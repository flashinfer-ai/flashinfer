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
kernel_cake_kimi_k3_attn_res_d7fe76d83967254ae38c(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
        float2 v_110 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 qp_111 = make_float2(q[8], q[9]);
        sq[4] = fma_f32x2_rn_noftz(v_110, v_110, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_110, qp_111, dot[4]);
        float2 v_112 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 qp_113 = make_float2(q[10], q[11]);
        sq[4] = fma_f32x2_rn_noftz(v_112, v_112, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_112, qp_113, dot[4]);
        float2 v_114 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 qp_115 = make_float2(q[12], q[13]);
        sq[4] = fma_f32x2_rn_noftz(v_114, v_114, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_114, qp_115, dot[4]);
        float2 v_116 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 qp_117 = make_float2(q[14], q[15]);
        sq[4] = fma_f32x2_rn_noftz(v_116, v_116, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_116, qp_117, dot[4]);
        int base_118 = (4 + group) * 1024 + thread * 8;
        const int woff_119 = 8;
        unsigned int sw_120[4];
        sw_120[0] = words[woff_119];
        sw_120[1] = words[woff_119 + 1];
        sw_120[2] = words[woff_119 + 2];
        sw_120[3] = words[woff_119 + 3];
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                : "r"(sw_120[_pair]));
        }
        fsrc[16] = _unpack_bf16_10[0];
        fsrc[17] = _unpack_bf16_10[1];
        fsrc[18] = _unpack_bf16_10[2];
        fsrc[19] = _unpack_bf16_10[3];
        fsrc[20] = _unpack_bf16_10[4];
        fsrc[21] = _unpack_bf16_10[5];
        fsrc[22] = _unpack_bf16_10[6];
        fsrc[23] = _unpack_bf16_10[7];
        float2 v_121 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 qp_122 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_121, v_121, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_121, qp_122, dot[0]);
        float2 v_123 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 qp_124 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_123, v_123, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_123, qp_124, dot[0]);
        float2 v_125 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 qp_126 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_125, v_125, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_125, qp_126, dot[0]);
        float2 v_127 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 qp_128 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_127, v_127, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_127, qp_128, dot[0]);
        unsigned int sw_129[4];
        sw_129[0] = words[14 + woff_119];
        sw_129[1] = words[14 + woff_119 + 1];
        sw_129[2] = words[14 + woff_119 + 2];
        sw_129[3] = words[14 + woff_119 + 3];
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                : "r"(sw_129[_pair]));
        }
        fsrc[44] = _unpack_bf16_11[0];
        fsrc[45] = _unpack_bf16_11[1];
        fsrc[46] = _unpack_bf16_11[2];
        fsrc[47] = _unpack_bf16_11[3];
        fsrc[48] = _unpack_bf16_11[4];
        fsrc[49] = _unpack_bf16_11[5];
        fsrc[50] = _unpack_bf16_11[6];
        fsrc[51] = _unpack_bf16_11[7];
        float2 v_130 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 qp_131 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_130, v_130, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_130, qp_131, dot[1]);
        float2 v_132 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 qp_133 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_132, v_132, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_132, qp_133, dot[1]);
        float2 v_134 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 qp_135 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_134, v_134, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_134, qp_135, dot[1]);
        float2 v_136 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 qp_137 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_136, v_136, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_136, qp_137, dot[1]);
        unsigned int sw_138[4];
        sw_138[0] = words[28 + woff_119];
        sw_138[1] = words[28 + woff_119 + 1];
        sw_138[2] = words[28 + woff_119 + 2];
        sw_138[3] = words[28 + woff_119 + 3];
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
        fsrc[72] = _unpack_bf16_12[0];
        fsrc[73] = _unpack_bf16_12[1];
        fsrc[74] = _unpack_bf16_12[2];
        fsrc[75] = _unpack_bf16_12[3];
        fsrc[76] = _unpack_bf16_12[4];
        fsrc[77] = _unpack_bf16_12[5];
        fsrc[78] = _unpack_bf16_12[6];
        fsrc[79] = _unpack_bf16_12[7];
        float2 v_139 = make_float2(_unpack_bf16_12[0], _unpack_bf16_12[1]);
        float2 qp_140 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_139, v_139, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_139, qp_140, dot[2]);
        float2 v_141 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 qp_142 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_141, v_141, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_141, qp_142, dot[2]);
        float2 v_143 = make_float2(_unpack_bf16_12[4], _unpack_bf16_12[5]);
        float2 qp_144 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_143, v_143, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_143, qp_144, dot[2]);
        float2 v_145 = make_float2(_unpack_bf16_12[6], _unpack_bf16_12[7]);
        float2 qp_146 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_145, v_145, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_145, qp_146, dot[2]);
        unsigned int sw_147[4];
        sw_147[0] = words[42 + woff_119];
        sw_147[1] = words[42 + woff_119 + 1];
        sw_147[2] = words[42 + woff_119 + 2];
        sw_147[3] = words[42 + woff_119 + 3];
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
        float2 v_148 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 qp_149 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_148, v_148, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_148, qp_149, dot[3]);
        float2 v_150 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 qp_151 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_150, v_150, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_150, qp_151, dot[3]);
        float2 v_152 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
        float2 qp_153 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_152, v_152, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_152, qp_153, dot[3]);
        float2 v_154 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
        float2 qp_155 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_154, v_154, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_154, qp_155, dot[3]);
        unsigned int sw_156[4];
        sw_156[0] = words[56 + woff_119];
        sw_156[1] = words[56 + woff_119 + 1];
        sw_156[2] = words[56 + woff_119 + 2];
        sw_156[3] = words[56 + woff_119 + 3];
        __nv_bfloat162 a_157 = __as_bf16x2(sw_156[0]);
        __nv_bfloat162 d_158 = __as_bf16x2(dwords[woff_119]);
        __nv_bfloat162 mixed_159 = a_157 + d_158;
        sw_156[0] = __as_u32(mixed_159);
        words[56 + woff_119] = sw_156[0];
        __nv_bfloat162 a_160 = __as_bf16x2(sw_156[1]);
        __nv_bfloat162 d_161 = __as_bf16x2(dwords[woff_119 + 1]);
        __nv_bfloat162 mixed_162 = a_160 + d_161;
        sw_156[1] = __as_u32(mixed_162);
        words[56 + woff_119 + 1] = sw_156[1];
        __nv_bfloat162 a_163 = __as_bf16x2(sw_156[2]);
        __nv_bfloat162 d_164 = __as_bf16x2(dwords[woff_119 + 2]);
        __nv_bfloat162 mixed_165 = a_163 + d_164;
        sw_156[2] = __as_u32(mixed_165);
        words[56 + woff_119 + 2] = sw_156[2];
        __nv_bfloat162 a_166 = __as_bf16x2(sw_156[3]);
        __nv_bfloat162 d_167 = __as_bf16x2(dwords[woff_119 + 3]);
        __nv_bfloat162 mixed_168 = a_166 + d_167;
        sw_156[3] = __as_u32(mixed_168);
        words[56 + woff_119 + 3] = sw_156[3];
        {
            int4 _iv4 = make_int4(sw_156[0 + 0], sw_156[0 + 1], sw_156[0 + 2], sw_156[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_118)) + 0) = _iv4;
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
                : "r"(sw_156[_pair]));
        }
        float2 v_169 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 qp_170 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_169, v_169, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_169, qp_170, dot[4]);
        float2 v_171 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 qp_172 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_171, v_171, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_171, qp_172, dot[4]);
        float2 v_173 = make_float2(_unpack_bf16_14[4], _unpack_bf16_14[5]);
        float2 qp_174 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_173, v_173, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_173, qp_174, dot[4]);
        float2 v_175 = make_float2(_unpack_bf16_14[6], _unpack_bf16_14[7]);
        float2 qp_176 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_175, v_175, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_175, qp_176, dot[4]);
        int base_177 = 6144 + group * 512 + thread * 4;
        const int woff_178 = 12;
        unsigned int sw_179[4];
        sw_179[0] = words[woff_178];
        sw_179[1] = words[woff_178 + 1];
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                : "r"(sw_179[_pair]));
        }
        fsrc[24] = _unpack_bf16_15[0];
        fsrc[25] = _unpack_bf16_15[1];
        fsrc[26] = _unpack_bf16_15[2];
        fsrc[27] = _unpack_bf16_15[3];
        float2 v_180 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        sq[0] = fma_f32x2_rn_noftz(v_180, v_180, sq[0]);
        float2 v_181 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        sq[0] = fma_f32x2_rn_noftz(v_181, v_181, sq[0]);
        float2 v_182 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_183 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_182, qp_183, dot[0]);
        float2 v_184 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_185 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_184, qp_185, dot[0]);
        unsigned int sw_186[4];
        sw_186[0] = words[14 + woff_178];
        sw_186[1] = words[14 + woff_178 + 1];
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(sw_186[_pair]));
        }
        fsrc[52] = _unpack_bf16_16[0];
        fsrc[53] = _unpack_bf16_16[1];
        fsrc[54] = _unpack_bf16_16[2];
        fsrc[55] = _unpack_bf16_16[3];
        float2 v_187 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        sq[1] = fma_f32x2_rn_noftz(v_187, v_187, sq[1]);
        float2 v_188 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        sq[1] = fma_f32x2_rn_noftz(v_188, v_188, sq[1]);
        float2 v_189 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_190 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_189, qp_190, dot[1]);
        float2 v_191 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_192 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_191, qp_192, dot[1]);
        unsigned int sw_193[4];
        sw_193[0] = words[28 + woff_178];
        sw_193[1] = words[28 + woff_178 + 1];
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(sw_193[_pair]));
        }
        fsrc[80] = _unpack_bf16_17[0];
        fsrc[81] = _unpack_bf16_17[1];
        fsrc[82] = _unpack_bf16_17[2];
        fsrc[83] = _unpack_bf16_17[3];
        float2 v_194 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        sq[2] = fma_f32x2_rn_noftz(v_194, v_194, sq[2]);
        float2 v_195 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        sq[2] = fma_f32x2_rn_noftz(v_195, v_195, sq[2]);
        float2 v_196 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_197 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_196, qp_197, dot[2]);
        float2 v_198 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_199 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_198, qp_199, dot[2]);
        unsigned int sw_200[4];
        sw_200[0] = words[42 + woff_178];
        sw_200[1] = words[42 + woff_178 + 1];
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                : "r"(sw_200[_pair]));
        }
        float2 v_201 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        sq[3] = fma_f32x2_rn_noftz(v_201, v_201, sq[3]);
        float2 v_202 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        sq[3] = fma_f32x2_rn_noftz(v_202, v_202, sq[3]);
        float2 v_203 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_204 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_203, qp_204, dot[3]);
        float2 v_205 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_206 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_205, qp_206, dot[3]);
        unsigned int sw_207[4];
        sw_207[0] = words[56 + woff_178];
        sw_207[1] = words[56 + woff_178 + 1];
        __nv_bfloat162 a_208 = __as_bf16x2(sw_207[0]);
        __nv_bfloat162 d_209 = __as_bf16x2(dwords[woff_178]);
        __nv_bfloat162 mixed_210 = a_208 + d_209;
        sw_207[0] = __as_u32(mixed_210);
        words[56 + woff_178] = sw_207[0];
        __nv_bfloat162 a_211 = __as_bf16x2(sw_207[1]);
        __nv_bfloat162 d_212 = __as_bf16x2(dwords[woff_178 + 1]);
        __nv_bfloat162 mixed_213 = a_211 + d_212;
        sw_207[1] = __as_u32(mixed_213);
        words[56 + woff_178 + 1] = sw_207[1];
        {
            int2 _iv2 = make_int2(sw_207[0 + 0], sw_207[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_177)) + 0) = _iv2;
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
                : "r"(sw_207[_pair]));
        }
        float2 v_214 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        sq[4] = fma_f32x2_rn_noftz(v_214, v_214, sq[4]);
        float2 v_215 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        sq[4] = fma_f32x2_rn_noftz(v_215, v_215, sq[4]);
        float2 v_216 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 qp_217 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_216, qp_217, dot[4]);
        float2 v_218 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_219 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_218, qp_219, dot[4]);
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
        unsigned long long bits_220 = 0;
        float2 _bit_cast_src_31 = pairs[1];
        bits_220 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_31)[0];
        unsigned long long peerbits_221 = __shfl_xor_sync(0xFFFFFFFF, bits_220, 16);
        float2 peer_222 = make_float2(0.0f, 0.0f);
        peer_222 = reinterpret_cast<float2*>(&peerbits_221)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_222);
        unsigned long long bits_223 = 0;
        float2 _bit_cast_src_32 = pairs[2];
        bits_223 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_32)[0];
        unsigned long long peerbits_224 = __shfl_xor_sync(0xFFFFFFFF, bits_223, 16);
        float2 peer_225 = make_float2(0.0f, 0.0f);
        peer_225 = reinterpret_cast<float2*>(&peerbits_224)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_225);
        unsigned long long bits_226 = 0;
        float2 _bit_cast_src_33 = pairs[3];
        bits_226 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_33)[0];
        unsigned long long peerbits_227 = __shfl_xor_sync(0xFFFFFFFF, bits_226, 16);
        float2 peer_228 = make_float2(0.0f, 0.0f);
        peer_228 = reinterpret_cast<float2*>(&peerbits_227)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_228);
        unsigned long long bits_229 = 0;
        float2 _bit_cast_src_34 = pairs[4];
        bits_229 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_34)[0];
        unsigned long long peerbits_230 = __shfl_xor_sync(0xFFFFFFFF, bits_229, 16);
        float2 peer_231 = make_float2(0.0f, 0.0f);
        peer_231 = reinterpret_cast<float2*>(&peerbits_230)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_231);
        unsigned long long bits_232 = 0;
        float2 _bit_cast_src_35 = pairs[0];
        bits_232 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_35)[0];
        unsigned long long peerbits_233 = __shfl_xor_sync(0xFFFFFFFF, bits_232, 8);
        float2 peer_234 = make_float2(0.0f, 0.0f);
        peer_234 = reinterpret_cast<float2*>(&peerbits_233)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_234);
        unsigned long long bits_235 = 0;
        float2 _bit_cast_src_36 = pairs[1];
        bits_235 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_36)[0];
        unsigned long long peerbits_236 = __shfl_xor_sync(0xFFFFFFFF, bits_235, 8);
        float2 peer_237 = make_float2(0.0f, 0.0f);
        peer_237 = reinterpret_cast<float2*>(&peerbits_236)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_237);
        unsigned long long bits_238 = 0;
        float2 _bit_cast_src_37 = pairs[2];
        bits_238 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_37)[0];
        unsigned long long peerbits_239 = __shfl_xor_sync(0xFFFFFFFF, bits_238, 8);
        float2 peer_240 = make_float2(0.0f, 0.0f);
        peer_240 = reinterpret_cast<float2*>(&peerbits_239)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_240);
        unsigned long long bits_241 = 0;
        float2 _bit_cast_src_38 = pairs[3];
        bits_241 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_38)[0];
        unsigned long long peerbits_242 = __shfl_xor_sync(0xFFFFFFFF, bits_241, 8);
        float2 peer_243 = make_float2(0.0f, 0.0f);
        peer_243 = reinterpret_cast<float2*>(&peerbits_242)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_243);
        unsigned long long bits_244 = 0;
        float2 _bit_cast_src_39 = pairs[4];
        bits_244 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_39)[0];
        unsigned long long peerbits_245 = __shfl_xor_sync(0xFFFFFFFF, bits_244, 8);
        float2 peer_246 = make_float2(0.0f, 0.0f);
        peer_246 = reinterpret_cast<float2*>(&peerbits_245)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_246);
        unsigned long long bits_247 = 0;
        float2 _bit_cast_src_40 = pairs[0];
        bits_247 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_40)[0];
        unsigned long long peerbits_248 = __shfl_xor_sync(0xFFFFFFFF, bits_247, 4);
        float2 peer_249 = make_float2(0.0f, 0.0f);
        peer_249 = reinterpret_cast<float2*>(&peerbits_248)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_249);
        unsigned long long bits_250 = 0;
        float2 _bit_cast_src_41 = pairs[1];
        bits_250 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_41)[0];
        unsigned long long peerbits_251 = __shfl_xor_sync(0xFFFFFFFF, bits_250, 4);
        float2 peer_252 = make_float2(0.0f, 0.0f);
        peer_252 = reinterpret_cast<float2*>(&peerbits_251)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_252);
        unsigned long long bits_253 = 0;
        float2 _bit_cast_src_42 = pairs[2];
        bits_253 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits_254 = __shfl_xor_sync(0xFFFFFFFF, bits_253, 4);
        float2 peer_255 = make_float2(0.0f, 0.0f);
        peer_255 = reinterpret_cast<float2*>(&peerbits_254)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_255);
        unsigned long long bits_256 = 0;
        float2 _bit_cast_src_43 = pairs[3];
        bits_256 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_257 = __shfl_xor_sync(0xFFFFFFFF, bits_256, 4);
        float2 peer_258 = make_float2(0.0f, 0.0f);
        peer_258 = reinterpret_cast<float2*>(&peerbits_257)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_258);
        unsigned long long bits_259 = 0;
        float2 _bit_cast_src_44 = pairs[4];
        bits_259 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_260 = __shfl_xor_sync(0xFFFFFFFF, bits_259, 4);
        float2 peer_261 = make_float2(0.0f, 0.0f);
        peer_261 = reinterpret_cast<float2*>(&peerbits_260)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_261);
        unsigned long long bits_262 = 0;
        float2 _bit_cast_src_45 = pairs[0];
        bits_262 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_263 = __shfl_xor_sync(0xFFFFFFFF, bits_262, 2);
        float2 peer_264 = make_float2(0.0f, 0.0f);
        peer_264 = reinterpret_cast<float2*>(&peerbits_263)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_264);
        unsigned long long bits_265 = 0;
        float2 _bit_cast_src_46 = pairs[1];
        bits_265 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_266 = __shfl_xor_sync(0xFFFFFFFF, bits_265, 2);
        float2 peer_267 = make_float2(0.0f, 0.0f);
        peer_267 = reinterpret_cast<float2*>(&peerbits_266)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_267);
        unsigned long long bits_268 = 0;
        float2 _bit_cast_src_47 = pairs[2];
        bits_268 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_269 = __shfl_xor_sync(0xFFFFFFFF, bits_268, 2);
        float2 peer_270 = make_float2(0.0f, 0.0f);
        peer_270 = reinterpret_cast<float2*>(&peerbits_269)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_270);
        unsigned long long bits_271 = 0;
        float2 _bit_cast_src_48 = pairs[3];
        bits_271 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_272 = __shfl_xor_sync(0xFFFFFFFF, bits_271, 2);
        float2 peer_273 = make_float2(0.0f, 0.0f);
        peer_273 = reinterpret_cast<float2*>(&peerbits_272)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_273);
        unsigned long long bits_274 = 0;
        float2 _bit_cast_src_49 = pairs[4];
        bits_274 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_275 = __shfl_xor_sync(0xFFFFFFFF, bits_274, 2);
        float2 peer_276 = make_float2(0.0f, 0.0f);
        peer_276 = reinterpret_cast<float2*>(&peerbits_275)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_276);
        unsigned long long bits_277 = 0;
        float2 _bit_cast_src_50 = pairs[0];
        bits_277 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_278 = __shfl_xor_sync(0xFFFFFFFF, bits_277, 1);
        float2 peer_279 = make_float2(0.0f, 0.0f);
        peer_279 = reinterpret_cast<float2*>(&peerbits_278)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_279);
        unsigned long long bits_280 = 0;
        float2 _bit_cast_src_51 = pairs[1];
        bits_280 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_281 = __shfl_xor_sync(0xFFFFFFFF, bits_280, 1);
        float2 peer_282 = make_float2(0.0f, 0.0f);
        peer_282 = reinterpret_cast<float2*>(&peerbits_281)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_282);
        unsigned long long bits_283 = 0;
        float2 _bit_cast_src_52 = pairs[2];
        bits_283 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_284 = __shfl_xor_sync(0xFFFFFFFF, bits_283, 1);
        float2 peer_285 = make_float2(0.0f, 0.0f);
        peer_285 = reinterpret_cast<float2*>(&peerbits_284)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_285);
        unsigned long long bits_286 = 0;
        float2 _bit_cast_src_53 = pairs[3];
        bits_286 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_287 = __shfl_xor_sync(0xFFFFFFFF, bits_286, 1);
        float2 peer_288 = make_float2(0.0f, 0.0f);
        peer_288 = reinterpret_cast<float2*>(&peerbits_287)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_288);
        unsigned long long bits_289 = 0;
        float2 _bit_cast_src_54 = pairs[4];
        bits_289 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_290 = __shfl_xor_sync(0xFFFFFFFF, bits_289, 1);
        float2 peer_291 = make_float2(0.0f, 0.0f);
        peer_291 = reinterpret_cast<float2*>(&peerbits_290)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_291);
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
        float _shfl_down_sync_75 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_75;
        float _shfl_down_sync_76 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_76;
        float _shfl_down_sync_77 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_77;
        float _shfl_down_sync_78 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_78;
        float _shfl_down_sync_79 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_79;
        float _shfl_down_sync_80 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_80;
        float logit = 0.0f;
        if (stat_n < 5 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[5];
        float _shfl_sync_81 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_81;
        float _shfl_sync_82 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_82;
        float _shfl_sync_83 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_83;
        float _shfl_sync_84 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_sync_84;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 1) {
            total_sq2 = stats[(stat_w * 5 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 5 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_85 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_85;
        float _shfl_down_sync_86 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_86;
        float _shfl_down_sync_87 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_87;
        float _shfl_down_sync_88 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_88;
        float _shfl_down_sync_89 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_89;
        float _shfl_down_sync_90 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_90;
        logit2 = 0.0f;
        if (stat_n < 1 && stat_w == 0) {
            float sigma2 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_sync_91 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_sync_91;
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
        float _exp2_92 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_92;
        sum_weights += weights[0];
        float _exp2_93 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_93;
        sum_weights += weights[1];
        float _exp2_94 = approx_exp2((logits[2] - max_new) * 1.442695f);
        weights[2] = _exp2_94;
        sum_weights += weights[2];
        float2 corr = make_float2(correction, correction);
        const int woff_292 = 0;
        int base_293 = group * 1024 + thread * 8;
        float2 a_294[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_294[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_295 = make_float2(acc[2], acc[3]);
        a_294[1] = mul_f32x2_noftz(previous_295, corr);
        float2 previous_296 = make_float2(acc[4], acc[5]);
        a_294[2] = mul_f32x2_noftz(previous_296, corr);
        float2 previous_297 = make_float2(acc[6], acc[7]);
        a_294[3] = mul_f32x2_noftz(previous_297, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_298 = make_float2(fsrc[0], fsrc[1]);
        a_294[0] = fma_f32x2_rn_noftz(weight, v_298, a_294[0]);
        float2 v_299 = make_float2(fsrc[2], fsrc[3]);
        a_294[1] = fma_f32x2_rn_noftz(weight, v_299, a_294[1]);
        float2 v_300 = make_float2(fsrc[4], fsrc[5]);
        a_294[2] = fma_f32x2_rn_noftz(weight, v_300, a_294[2]);
        float2 v_301 = make_float2(fsrc[6], fsrc[7]);
        a_294[3] = fma_f32x2_rn_noftz(weight, v_301, a_294[3]);
        float2 weight_302 = make_float2(weights[1], weights[1]);
        float2 v_303 = make_float2(fsrc[28], fsrc[29]);
        a_294[0] = fma_f32x2_rn_noftz(weight_302, v_303, a_294[0]);
        float2 v_304 = make_float2(fsrc[30], fsrc[31]);
        a_294[1] = fma_f32x2_rn_noftz(weight_302, v_304, a_294[1]);
        float2 v_305 = make_float2(fsrc[32], fsrc[33]);
        a_294[2] = fma_f32x2_rn_noftz(weight_302, v_305, a_294[2]);
        float2 v_306 = make_float2(fsrc[34], fsrc[35]);
        a_294[3] = fma_f32x2_rn_noftz(weight_302, v_306, a_294[3]);
        float2 weight_307 = make_float2(weights[2], weights[2]);
        float2 v_308 = make_float2(fsrc[56], fsrc[57]);
        a_294[0] = fma_f32x2_rn_noftz(weight_307, v_308, a_294[0]);
        float2 v_309 = make_float2(fsrc[58], fsrc[59]);
        a_294[1] = fma_f32x2_rn_noftz(weight_307, v_309, a_294[1]);
        float2 v_310 = make_float2(fsrc[60], fsrc[61]);
        a_294[2] = fma_f32x2_rn_noftz(weight_307, v_310, a_294[2]);
        float2 v_311 = make_float2(fsrc[62], fsrc[63]);
        a_294[3] = fma_f32x2_rn_noftz(weight_307, v_311, a_294[3]);
        acc[0] = a_294[0].x;
        acc[1] = a_294[0].y;
        acc[2] = a_294[1].x;
        acc[3] = a_294[1].y;
        acc[4] = a_294[2].x;
        acc[5] = a_294[2].y;
        acc[6] = a_294[3].x;
        acc[7] = a_294[3].y;
        const int woff_312 = 4;
        int base_313 = (2 + group) * 1024 + thread * 8;
        float2 a_314[4];
        float2 previous_315 = make_float2(acc[8], acc[9]);
        a_314[0] = mul_f32x2_noftz(previous_315, corr);
        float2 previous_316 = make_float2(acc[10], acc[11]);
        a_314[1] = mul_f32x2_noftz(previous_316, corr);
        float2 previous_317 = make_float2(acc[12], acc[13]);
        a_314[2] = mul_f32x2_noftz(previous_317, corr);
        float2 previous_318 = make_float2(acc[14], acc[15]);
        a_314[3] = mul_f32x2_noftz(previous_318, corr);
        float2 weight_319 = make_float2(weights[0], weights[0]);
        float2 v_320 = make_float2(fsrc[8], fsrc[9]);
        a_314[0] = fma_f32x2_rn_noftz(weight_319, v_320, a_314[0]);
        float2 v_321 = make_float2(fsrc[10], fsrc[11]);
        a_314[1] = fma_f32x2_rn_noftz(weight_319, v_321, a_314[1]);
        float2 v_322 = make_float2(fsrc[12], fsrc[13]);
        a_314[2] = fma_f32x2_rn_noftz(weight_319, v_322, a_314[2]);
        float2 v_323 = make_float2(fsrc[14], fsrc[15]);
        a_314[3] = fma_f32x2_rn_noftz(weight_319, v_323, a_314[3]);
        float2 weight_324 = make_float2(weights[1], weights[1]);
        float2 v_325 = make_float2(fsrc[36], fsrc[37]);
        a_314[0] = fma_f32x2_rn_noftz(weight_324, v_325, a_314[0]);
        float2 v_326 = make_float2(fsrc[38], fsrc[39]);
        a_314[1] = fma_f32x2_rn_noftz(weight_324, v_326, a_314[1]);
        float2 v_327 = make_float2(fsrc[40], fsrc[41]);
        a_314[2] = fma_f32x2_rn_noftz(weight_324, v_327, a_314[2]);
        float2 v_328 = make_float2(fsrc[42], fsrc[43]);
        a_314[3] = fma_f32x2_rn_noftz(weight_324, v_328, a_314[3]);
        float2 weight_329 = make_float2(weights[2], weights[2]);
        float2 v_330 = make_float2(fsrc[64], fsrc[65]);
        a_314[0] = fma_f32x2_rn_noftz(weight_329, v_330, a_314[0]);
        float2 v_331 = make_float2(fsrc[66], fsrc[67]);
        a_314[1] = fma_f32x2_rn_noftz(weight_329, v_331, a_314[1]);
        float2 v_332 = make_float2(fsrc[68], fsrc[69]);
        a_314[2] = fma_f32x2_rn_noftz(weight_329, v_332, a_314[2]);
        float2 v_333 = make_float2(fsrc[70], fsrc[71]);
        a_314[3] = fma_f32x2_rn_noftz(weight_329, v_333, a_314[3]);
        acc[8] = a_314[0].x;
        acc[9] = a_314[0].y;
        acc[10] = a_314[1].x;
        acc[11] = a_314[1].y;
        acc[12] = a_314[2].x;
        acc[13] = a_314[2].y;
        acc[14] = a_314[3].x;
        acc[15] = a_314[3].y;
        const int woff_334 = 8;
        int base_335 = (4 + group) * 1024 + thread * 8;
        float2 a_336[4];
        float2 previous_337 = make_float2(acc[16], acc[17]);
        a_336[0] = mul_f32x2_noftz(previous_337, corr);
        float2 previous_338 = make_float2(acc[18], acc[19]);
        a_336[1] = mul_f32x2_noftz(previous_338, corr);
        float2 previous_339 = make_float2(acc[20], acc[21]);
        a_336[2] = mul_f32x2_noftz(previous_339, corr);
        float2 previous_340 = make_float2(acc[22], acc[23]);
        a_336[3] = mul_f32x2_noftz(previous_340, corr);
        float2 weight_341 = make_float2(weights[0], weights[0]);
        float2 v_342 = make_float2(fsrc[16], fsrc[17]);
        a_336[0] = fma_f32x2_rn_noftz(weight_341, v_342, a_336[0]);
        float2 v_343 = make_float2(fsrc[18], fsrc[19]);
        a_336[1] = fma_f32x2_rn_noftz(weight_341, v_343, a_336[1]);
        float2 v_344 = make_float2(fsrc[20], fsrc[21]);
        a_336[2] = fma_f32x2_rn_noftz(weight_341, v_344, a_336[2]);
        float2 v_345 = make_float2(fsrc[22], fsrc[23]);
        a_336[3] = fma_f32x2_rn_noftz(weight_341, v_345, a_336[3]);
        float2 weight_346 = make_float2(weights[1], weights[1]);
        float2 v_347 = make_float2(fsrc[44], fsrc[45]);
        a_336[0] = fma_f32x2_rn_noftz(weight_346, v_347, a_336[0]);
        float2 v_348 = make_float2(fsrc[46], fsrc[47]);
        a_336[1] = fma_f32x2_rn_noftz(weight_346, v_348, a_336[1]);
        float2 v_349 = make_float2(fsrc[48], fsrc[49]);
        a_336[2] = fma_f32x2_rn_noftz(weight_346, v_349, a_336[2]);
        float2 v_350 = make_float2(fsrc[50], fsrc[51]);
        a_336[3] = fma_f32x2_rn_noftz(weight_346, v_350, a_336[3]);
        float2 weight_351 = make_float2(weights[2], weights[2]);
        float2 v_352 = make_float2(fsrc[72], fsrc[73]);
        a_336[0] = fma_f32x2_rn_noftz(weight_351, v_352, a_336[0]);
        float2 v_353 = make_float2(fsrc[74], fsrc[75]);
        a_336[1] = fma_f32x2_rn_noftz(weight_351, v_353, a_336[1]);
        float2 v_354 = make_float2(fsrc[76], fsrc[77]);
        a_336[2] = fma_f32x2_rn_noftz(weight_351, v_354, a_336[2]);
        float2 v_355 = make_float2(fsrc[78], fsrc[79]);
        a_336[3] = fma_f32x2_rn_noftz(weight_351, v_355, a_336[3]);
        acc[16] = a_336[0].x;
        acc[17] = a_336[0].y;
        acc[18] = a_336[1].x;
        acc[19] = a_336[1].y;
        acc[20] = a_336[2].x;
        acc[21] = a_336[2].y;
        acc[22] = a_336[3].x;
        acc[23] = a_336[3].y;
        const int woff_356 = 12;
        int base_357 = 6144 + group * 512 + thread * 4;
        float2 a_358[4];
        float2 previous_359 = make_float2(acc[24], acc[25]);
        a_358[0] = mul_f32x2_noftz(previous_359, corr);
        float2 previous_360 = make_float2(acc[26], acc[27]);
        a_358[1] = mul_f32x2_noftz(previous_360, corr);
        float2 weight_361 = make_float2(weights[0], weights[0]);
        float2 v_362 = make_float2(fsrc[24], fsrc[25]);
        a_358[0] = fma_f32x2_rn_noftz(weight_361, v_362, a_358[0]);
        float2 v_363 = make_float2(fsrc[26], fsrc[27]);
        a_358[1] = fma_f32x2_rn_noftz(weight_361, v_363, a_358[1]);
        float2 weight_364 = make_float2(weights[1], weights[1]);
        float2 v_365 = make_float2(fsrc[52], fsrc[53]);
        a_358[0] = fma_f32x2_rn_noftz(weight_364, v_365, a_358[0]);
        float2 v_366 = make_float2(fsrc[54], fsrc[55]);
        a_358[1] = fma_f32x2_rn_noftz(weight_364, v_366, a_358[1]);
        float2 weight_367 = make_float2(weights[2], weights[2]);
        float2 v_368 = make_float2(fsrc[80], fsrc[81]);
        a_358[0] = fma_f32x2_rn_noftz(weight_367, v_368, a_358[0]);
        float2 v_369 = make_float2(fsrc[82], fsrc[83]);
        a_358[1] = fma_f32x2_rn_noftz(weight_367, v_369, a_358[1]);
        acc[24] = a_358[0].x;
        acc[25] = a_358[0].y;
        acc[26] = a_358[1].x;
        acc[27] = a_358[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_370 = -3.4028235e+38f;
        max_chunk_370 = fmaxf(max_chunk_370, logits[3]);
        max_chunk_370 = fmaxf(max_chunk_370, logits[4]);
        float max_new_371 = fmaxf(max_running, max_chunk_370);
        float correction_372 = approx_exp2((max_running - max_new_371) * 1.442695f);
        float weights_373[5];
        float sum_weights_374 = 0.0f;
        float _exp2_95 = approx_exp2((logits[3] - max_new_371) * 1.442695f);
        weights_373[3] = _exp2_95;
        sum_weights_374 += weights_373[3];
        float _exp2_96 = approx_exp2((logits[4] - max_new_371) * 1.442695f);
        weights_373[4] = _exp2_96;
        sum_weights_374 += weights_373[4];
        float2 corr_375 = make_float2(correction_372, correction_372);
        const int woff_376 = 0;
        int base_377 = group * 1024 + thread * 8;
        float2 a_378[4];
        float2 previous_379 = make_float2(acc[0], acc[1]);
        a_378[0] = mul_f32x2_noftz(previous_379, corr_375);
        float2 previous_380 = make_float2(acc[2], acc[3]);
        a_378[1] = mul_f32x2_noftz(previous_380, corr_375);
        float2 previous_381 = make_float2(acc[4], acc[5]);
        a_378[2] = mul_f32x2_noftz(previous_381, corr_375);
        float2 previous_382 = make_float2(acc[6], acc[7]);
        a_378[3] = mul_f32x2_noftz(previous_382, corr_375);
        float2 weight_383 = make_float2(weights_373[3], weights_373[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_376];
        sw2[1] = words[42 + woff_376 + 1];
        sw2[2] = words[42 + woff_376 + 2];
        sw2[3] = words[42 + woff_376 + 3];
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
        float2 v_384 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        a_378[0] = fma_f32x2_rn_noftz(weight_383, v_384, a_378[0]);
        float2 v_385 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        a_378[1] = fma_f32x2_rn_noftz(weight_383, v_385, a_378[1]);
        float2 v_386 = make_float2(_unpack_bf16_20[4], _unpack_bf16_20[5]);
        a_378[2] = fma_f32x2_rn_noftz(weight_383, v_386, a_378[2]);
        float2 v_387 = make_float2(_unpack_bf16_20[6], _unpack_bf16_20[7]);
        a_378[3] = fma_f32x2_rn_noftz(weight_383, v_387, a_378[3]);
        float2 weight_388 = make_float2(weights_373[4], weights_373[4]);
        unsigned int sw2_389[4];
        sw2_389[0] = words[56 + woff_376];
        sw2_389[1] = words[56 + woff_376 + 1];
        sw2_389[2] = words[56 + woff_376 + 2];
        sw2_389[3] = words[56 + woff_376 + 3];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw2_389[_pair]));
        }
        float2 v_390 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        a_378[0] = fma_f32x2_rn_noftz(weight_388, v_390, a_378[0]);
        float2 v_391 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        a_378[1] = fma_f32x2_rn_noftz(weight_388, v_391, a_378[1]);
        float2 v_392 = make_float2(_unpack_bf16_21[4], _unpack_bf16_21[5]);
        a_378[2] = fma_f32x2_rn_noftz(weight_388, v_392, a_378[2]);
        float2 v_393 = make_float2(_unpack_bf16_21[6], _unpack_bf16_21[7]);
        a_378[3] = fma_f32x2_rn_noftz(weight_388, v_393, a_378[3]);
        acc[0] = a_378[0].x;
        acc[1] = a_378[0].y;
        acc[2] = a_378[1].x;
        acc[3] = a_378[1].y;
        acc[4] = a_378[2].x;
        acc[5] = a_378[2].y;
        acc[6] = a_378[3].x;
        acc[7] = a_378[3].y;
        const int woff_394 = 4;
        int base_395 = (2 + group) * 1024 + thread * 8;
        float2 a_396[4];
        float2 previous_397 = make_float2(acc[8], acc[9]);
        a_396[0] = mul_f32x2_noftz(previous_397, corr_375);
        float2 previous_398 = make_float2(acc[10], acc[11]);
        a_396[1] = mul_f32x2_noftz(previous_398, corr_375);
        float2 previous_399 = make_float2(acc[12], acc[13]);
        a_396[2] = mul_f32x2_noftz(previous_399, corr_375);
        float2 previous_400 = make_float2(acc[14], acc[15]);
        a_396[3] = mul_f32x2_noftz(previous_400, corr_375);
        float2 weight_401 = make_float2(weights_373[3], weights_373[3]);
        unsigned int sw2_402[4];
        sw2_402[0] = words[42 + woff_394];
        sw2_402[1] = words[42 + woff_394 + 1];
        sw2_402[2] = words[42 + woff_394 + 2];
        sw2_402[3] = words[42 + woff_394 + 3];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw2_402[_pair]));
        }
        float2 v_403 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        a_396[0] = fma_f32x2_rn_noftz(weight_401, v_403, a_396[0]);
        float2 v_404 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        a_396[1] = fma_f32x2_rn_noftz(weight_401, v_404, a_396[1]);
        float2 v_405 = make_float2(_unpack_bf16_22[4], _unpack_bf16_22[5]);
        a_396[2] = fma_f32x2_rn_noftz(weight_401, v_405, a_396[2]);
        float2 v_406 = make_float2(_unpack_bf16_22[6], _unpack_bf16_22[7]);
        a_396[3] = fma_f32x2_rn_noftz(weight_401, v_406, a_396[3]);
        float2 weight_407 = make_float2(weights_373[4], weights_373[4]);
        unsigned int sw2_408[4];
        sw2_408[0] = words[56 + woff_394];
        sw2_408[1] = words[56 + woff_394 + 1];
        sw2_408[2] = words[56 + woff_394 + 2];
        sw2_408[3] = words[56 + woff_394 + 3];
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                : "r"(sw2_408[_pair]));
        }
        float2 v_409 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        a_396[0] = fma_f32x2_rn_noftz(weight_407, v_409, a_396[0]);
        float2 v_410 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        a_396[1] = fma_f32x2_rn_noftz(weight_407, v_410, a_396[1]);
        float2 v_411 = make_float2(_unpack_bf16_23[4], _unpack_bf16_23[5]);
        a_396[2] = fma_f32x2_rn_noftz(weight_407, v_411, a_396[2]);
        float2 v_412 = make_float2(_unpack_bf16_23[6], _unpack_bf16_23[7]);
        a_396[3] = fma_f32x2_rn_noftz(weight_407, v_412, a_396[3]);
        acc[8] = a_396[0].x;
        acc[9] = a_396[0].y;
        acc[10] = a_396[1].x;
        acc[11] = a_396[1].y;
        acc[12] = a_396[2].x;
        acc[13] = a_396[2].y;
        acc[14] = a_396[3].x;
        acc[15] = a_396[3].y;
        const int woff_413 = 8;
        int base_414 = (4 + group) * 1024 + thread * 8;
        float2 a_415[4];
        float2 previous_416 = make_float2(acc[16], acc[17]);
        a_415[0] = mul_f32x2_noftz(previous_416, corr_375);
        float2 previous_417 = make_float2(acc[18], acc[19]);
        a_415[1] = mul_f32x2_noftz(previous_417, corr_375);
        float2 previous_418 = make_float2(acc[20], acc[21]);
        a_415[2] = mul_f32x2_noftz(previous_418, corr_375);
        float2 previous_419 = make_float2(acc[22], acc[23]);
        a_415[3] = mul_f32x2_noftz(previous_419, corr_375);
        float2 weight_420 = make_float2(weights_373[3], weights_373[3]);
        unsigned int sw2_421[4];
        sw2_421[0] = words[42 + woff_413];
        sw2_421[1] = words[42 + woff_413 + 1];
        sw2_421[2] = words[42 + woff_413 + 2];
        sw2_421[3] = words[42 + woff_413 + 3];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw2_421[_pair]));
        }
        float2 v_422 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        a_415[0] = fma_f32x2_rn_noftz(weight_420, v_422, a_415[0]);
        float2 v_423 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        a_415[1] = fma_f32x2_rn_noftz(weight_420, v_423, a_415[1]);
        float2 v_424 = make_float2(_unpack_bf16_24[4], _unpack_bf16_24[5]);
        a_415[2] = fma_f32x2_rn_noftz(weight_420, v_424, a_415[2]);
        float2 v_425 = make_float2(_unpack_bf16_24[6], _unpack_bf16_24[7]);
        a_415[3] = fma_f32x2_rn_noftz(weight_420, v_425, a_415[3]);
        float2 weight_426 = make_float2(weights_373[4], weights_373[4]);
        unsigned int sw2_427[4];
        sw2_427[0] = words[56 + woff_413];
        sw2_427[1] = words[56 + woff_413 + 1];
        sw2_427[2] = words[56 + woff_413 + 2];
        sw2_427[3] = words[56 + woff_413 + 3];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw2_427[_pair]));
        }
        float2 v_428 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        a_415[0] = fma_f32x2_rn_noftz(weight_426, v_428, a_415[0]);
        float2 v_429 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        a_415[1] = fma_f32x2_rn_noftz(weight_426, v_429, a_415[1]);
        float2 v_430 = make_float2(_unpack_bf16_25[4], _unpack_bf16_25[5]);
        a_415[2] = fma_f32x2_rn_noftz(weight_426, v_430, a_415[2]);
        float2 v_431 = make_float2(_unpack_bf16_25[6], _unpack_bf16_25[7]);
        a_415[3] = fma_f32x2_rn_noftz(weight_426, v_431, a_415[3]);
        acc[16] = a_415[0].x;
        acc[17] = a_415[0].y;
        acc[18] = a_415[1].x;
        acc[19] = a_415[1].y;
        acc[20] = a_415[2].x;
        acc[21] = a_415[2].y;
        acc[22] = a_415[3].x;
        acc[23] = a_415[3].y;
        const int woff_432 = 12;
        int base_433 = 6144 + group * 512 + thread * 4;
        float2 a_434[4];
        float2 previous_435 = make_float2(acc[24], acc[25]);
        a_434[0] = mul_f32x2_noftz(previous_435, corr_375);
        float2 previous_436 = make_float2(acc[26], acc[27]);
        a_434[1] = mul_f32x2_noftz(previous_436, corr_375);
        float2 weight_437 = make_float2(weights_373[3], weights_373[3]);
        unsigned int sw2_438[4];
        sw2_438[0] = words[42 + woff_432];
        sw2_438[1] = words[42 + woff_432 + 1];
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw2_438[_pair]));
        }
        float2 v_439 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        a_434[0] = fma_f32x2_rn_noftz(weight_437, v_439, a_434[0]);
        float2 v_440 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        a_434[1] = fma_f32x2_rn_noftz(weight_437, v_440, a_434[1]);
        float2 weight_441 = make_float2(weights_373[4], weights_373[4]);
        unsigned int sw2_442[4];
        sw2_442[0] = words[56 + woff_432];
        sw2_442[1] = words[56 + woff_432 + 1];
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw2_442[_pair]));
        }
        float2 v_443 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        a_434[0] = fma_f32x2_rn_noftz(weight_441, v_443, a_434[0]);
        float2 v_444 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        a_434[1] = fma_f32x2_rn_noftz(weight_441, v_444, a_434[1]);
        acc[24] = a_434[0].x;
        acc[25] = a_434[0].y;
        acc[26] = a_434[1].x;
        acc[27] = a_434[1].y;
        sum_running = __fmaf_rn(sum_running, correction_372, sum_weights_374);
        max_running = max_new_371;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_445 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_445, v_445, output_sq_pair);
        float2 v_446 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_446, v_446, output_sq_pair);
        float2 v_447 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_447, v_447, output_sq_pair);
        float2 v_448 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_448, v_448, output_sq_pair);
        float2 v_449 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_449, v_449, output_sq_pair);
        float2 v_450 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_450, v_450, output_sq_pair);
        float2 v_451 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_451, v_451, output_sq_pair);
        float2 v_452 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_452, v_452, output_sq_pair);
        float2 v_453 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_453, v_453, output_sq_pair);
        float2 v_454 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_454, v_454, output_sq_pair);
        float2 v_455 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_455, v_455, output_sq_pair);
        float2 v_456 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_456, v_456, output_sq_pair);
        float2 v_457 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_457, v_457, output_sq_pair);
        float2 v_458 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_458, v_458, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_98;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_100;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_101;
        if (lane == 0) {
            uint32_t _cluster_mapa_102;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_102) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_102), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_103;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_103) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_103), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_104 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_104;
        float _shfl_down_sync_105 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_105;
        float _shfl_down_sync_106 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_106;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_107 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_107;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_459 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_108 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_109 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_109));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_460 = 2;
        const int acc_idx_461 = value_idx_460;
        float2 _f32x2_mul_a_110 = make_float2(acc[acc_idx_461], acc[acc_idx_461 + 1]);
        float2 scaled_pair_462;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_462) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_111 = make_float2(wout[acc_idx_461], wout[acc_idx_461 + 1]);
        float2 normalized_pair_463;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_463) : "l"(*(const unsigned long long*)&scaled_pair_462), "l"(*(const unsigned long long*)&_f32x2_mul_b_111));
        output_values[value_idx_460] = normalized_pair_463.x;
        output_values[value_idx_460 + 1] = normalized_pair_463.y;
        const int value_idx_464 = 4;
        const int acc_idx_465 = value_idx_464;
        float2 _f32x2_mul_a_112 = make_float2(acc[acc_idx_465], acc[acc_idx_465 + 1]);
        float2 scaled_pair_466;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_466) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_113 = make_float2(wout[acc_idx_465], wout[acc_idx_465 + 1]);
        float2 normalized_pair_467;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_467) : "l"(*(const unsigned long long*)&scaled_pair_466), "l"(*(const unsigned long long*)&_f32x2_mul_b_113));
        output_values[value_idx_464] = normalized_pair_467.x;
        output_values[value_idx_464 + 1] = normalized_pair_467.y;
        const int value_idx_468 = 6;
        const int acc_idx_469 = value_idx_468;
        float2 _f32x2_mul_a_114 = make_float2(acc[acc_idx_469], acc[acc_idx_469 + 1]);
        float2 scaled_pair_470;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_470) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_115 = make_float2(wout[acc_idx_469], wout[acc_idx_469 + 1]);
        float2 normalized_pair_471;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_471) : "l"(*(const unsigned long long*)&scaled_pair_470), "l"(*(const unsigned long long*)&_f32x2_mul_b_115));
        output_values[value_idx_468] = normalized_pair_471.x;
        output_values[value_idx_468 + 1] = normalized_pair_471.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_459))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_472 = (2 + group) * 1024 + thread * 8;
        float output_values_473[8];
        const int value_idx_474 = 0;
        const int acc_idx_475 = 8 + value_idx_474;
        float2 _f32x2_mul_a_116 = make_float2(acc[acc_idx_475], acc[acc_idx_475 + 1]);
        float2 scaled_pair_476;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_476) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_117 = make_float2(wout[acc_idx_475], wout[acc_idx_475 + 1]);
        float2 normalized_pair_477;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_477) : "l"(*(const unsigned long long*)&scaled_pair_476), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
        output_values_473[value_idx_474] = normalized_pair_477.x;
        output_values_473[value_idx_474 + 1] = normalized_pair_477.y;
        const int value_idx_478 = 2;
        const int acc_idx_479 = 8 + value_idx_478;
        float2 _f32x2_mul_a_118 = make_float2(acc[acc_idx_479], acc[acc_idx_479 + 1]);
        float2 scaled_pair_480;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_480) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_119 = make_float2(wout[acc_idx_479], wout[acc_idx_479 + 1]);
        float2 normalized_pair_481;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_481) : "l"(*(const unsigned long long*)&scaled_pair_480), "l"(*(const unsigned long long*)&_f32x2_mul_b_119));
        output_values_473[value_idx_478] = normalized_pair_481.x;
        output_values_473[value_idx_478 + 1] = normalized_pair_481.y;
        const int value_idx_482 = 4;
        const int acc_idx_483 = 8 + value_idx_482;
        float2 _f32x2_mul_a_120 = make_float2(acc[acc_idx_483], acc[acc_idx_483 + 1]);
        float2 scaled_pair_484;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_484) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_121 = make_float2(wout[acc_idx_483], wout[acc_idx_483 + 1]);
        float2 normalized_pair_485;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_485) : "l"(*(const unsigned long long*)&scaled_pair_484), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
        output_values_473[value_idx_482] = normalized_pair_485.x;
        output_values_473[value_idx_482 + 1] = normalized_pair_485.y;
        const int value_idx_486 = 6;
        const int acc_idx_487 = 8 + value_idx_486;
        float2 _f32x2_mul_a_122 = make_float2(acc[acc_idx_487], acc[acc_idx_487 + 1]);
        float2 scaled_pair_488;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_488) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_123 = make_float2(wout[acc_idx_487], wout[acc_idx_487 + 1]);
        float2 normalized_pair_489;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_489) : "l"(*(const unsigned long long*)&scaled_pair_488), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
        output_values_473[value_idx_486] = normalized_pair_489.x;
        output_values_473[value_idx_486 + 1] = normalized_pair_489.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_473[_lp*2 + 0], output_values_473[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_473[_lp*2 + 2], output_values_473[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_473[_lp*2 + 4], output_values_473[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_473[_lp*2 + 6], output_values_473[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_472))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_490 = (4 + group) * 1024 + thread * 8;
        float output_values_491[8];
        const int value_idx_492 = 0;
        const int acc_idx_493 = 16 + value_idx_492;
        float2 _f32x2_mul_a_124 = make_float2(acc[acc_idx_493], acc[acc_idx_493 + 1]);
        float2 scaled_pair_494;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_494) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_125 = make_float2(wout[acc_idx_493], wout[acc_idx_493 + 1]);
        float2 normalized_pair_495;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_495) : "l"(*(const unsigned long long*)&scaled_pair_494), "l"(*(const unsigned long long*)&_f32x2_mul_b_125));
        output_values_491[value_idx_492] = normalized_pair_495.x;
        output_values_491[value_idx_492 + 1] = normalized_pair_495.y;
        const int value_idx_496 = 2;
        const int acc_idx_497 = 16 + value_idx_496;
        float2 _f32x2_mul_a_126 = make_float2(acc[acc_idx_497], acc[acc_idx_497 + 1]);
        float2 scaled_pair_498;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_498) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_127 = make_float2(wout[acc_idx_497], wout[acc_idx_497 + 1]);
        float2 normalized_pair_499;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_499) : "l"(*(const unsigned long long*)&scaled_pair_498), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
        output_values_491[value_idx_496] = normalized_pair_499.x;
        output_values_491[value_idx_496 + 1] = normalized_pair_499.y;
        const int value_idx_500 = 4;
        const int acc_idx_501 = 16 + value_idx_500;
        float2 _f32x2_mul_a_128 = make_float2(acc[acc_idx_501], acc[acc_idx_501 + 1]);
        float2 scaled_pair_502;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_502) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_129 = make_float2(wout[acc_idx_501], wout[acc_idx_501 + 1]);
        float2 normalized_pair_503;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_503) : "l"(*(const unsigned long long*)&scaled_pair_502), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
        output_values_491[value_idx_500] = normalized_pair_503.x;
        output_values_491[value_idx_500 + 1] = normalized_pair_503.y;
        const int value_idx_504 = 6;
        const int acc_idx_505 = 16 + value_idx_504;
        float2 _f32x2_mul_a_130 = make_float2(acc[acc_idx_505], acc[acc_idx_505 + 1]);
        float2 scaled_pair_506;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_506) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_131 = make_float2(wout[acc_idx_505], wout[acc_idx_505 + 1]);
        float2 normalized_pair_507;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_507) : "l"(*(const unsigned long long*)&scaled_pair_506), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
        output_values_491[value_idx_504] = normalized_pair_507.x;
        output_values_491[value_idx_504 + 1] = normalized_pair_507.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_491[_lp*2 + 0], output_values_491[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_491[_lp*2 + 2], output_values_491[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_491[_lp*2 + 4], output_values_491[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_491[_lp*2 + 6], output_values_491[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_490))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_508 = 6144 + group * 512 + thread * 4;
        float output_values_509[8];
        const int value_idx_510 = 0;
        const int acc_idx_511 = 24 + value_idx_510;
        float2 _f32x2_mul_a_132 = make_float2(acc[acc_idx_511], acc[acc_idx_511 + 1]);
        float2 scaled_pair_512;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_512) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_133 = make_float2(wout[acc_idx_511], wout[acc_idx_511 + 1]);
        float2 normalized_pair_513;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_513) : "l"(*(const unsigned long long*)&scaled_pair_512), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
        output_values_509[value_idx_510] = normalized_pair_513.x;
        output_values_509[value_idx_510 + 1] = normalized_pair_513.y;
        const int value_idx_514 = 2;
        const int acc_idx_515 = 24 + value_idx_514;
        float2 _f32x2_mul_a_134 = make_float2(acc[acc_idx_515], acc[acc_idx_515 + 1]);
        float2 scaled_pair_516;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_516) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_135 = make_float2(wout[acc_idx_515], wout[acc_idx_515 + 1]);
        float2 normalized_pair_517;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_517) : "l"(*(const unsigned long long*)&scaled_pair_516), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
        output_values_509[value_idx_514] = normalized_pair_517.x;
        output_values_509[value_idx_514 + 1] = normalized_pair_517.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_509[_lp*2 + 0], output_values_509[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_509[_lp*2 + 2], output_values_509[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_508))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
