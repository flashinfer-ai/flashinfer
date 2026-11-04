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
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 132
#define SMEM_REDUCE_SMEM_STRIDE 132
#define SMEM_RMS_SCALAR_OFF 132
#define SMEM_RMS_SCALAR_STAGE_BYTES 4
#define SMEM_RMS_SCALAR_STRIDE 4
#define SMEM_RMS_PARTIALS_OFF 136
#define SMEM_RMS_PARTIALS_STAGE_BYTES 32
#define SMEM_RMS_PARTIALS_STRIDE 32
#define SMEM_RS_STAGE_WORDS_OFF 0
#define SMEM_RS_STAGE_WORDS_STAGE_BYTES 3584
#define SMEM_RS_STAGE_WORDS_STRIDE 3584
#define SMEM_RS_SUM_WORDS_OFF 3584
#define SMEM_RS_SUM_WORDS_STAGE_BYTES 448
#define SMEM_RS_SUM_WORDS_STRIDE 448
#define SMEM_TOTAL 256
#define THREADS 224
#define LAUNCH_MIN_BLOCKS 5

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


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}





__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


extern "C" {

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_allreduce_union_ws8_bf16_two_shot_b5(__nv_bfloat16* __restrict__ active_expert_tokens, float* __restrict__ expert_scales, __nv_bfloat16* __restrict__ token_input, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ gamma, __nv_bfloat16* __restrict__ moe_allreduce_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int active_experts, float epsilon, float weight_bias)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[4032];
    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem + 168;
    #define rms_exchange_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* reduce_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_smem_addr = smem + 0;
    float* rms_scalar = reinterpret_cast<float*>(smem_raw + 132);
    const int rms_scalar_addr = smem + 132;
    float* rms_partials = reinterpret_cast<float*>(smem_raw + 136);
    const int rms_partials_addr = smem + 136;
    unsigned int* rs_stage_words = reinterpret_cast<unsigned int*>(smem_static_raw + 0);
    const int rs_stage_words_addr = (int)(unsigned long long)__cvta_generic_to_shared(rs_stage_words);
    unsigned int* rs_sum_words = reinterpret_cast<unsigned int*>(smem_static_raw + 3584);
    const int rs_sum_words_addr = (int)(unsigned long long)__cvta_generic_to_shared(rs_sum_words);
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_init(rms_exchange_addr, 1);
            mbarrier_init(rms_exchange_addr + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int early_thread = cta_rank * 224 + tid;
    int early_token = (((int)cluster_id < tokens) ? (int)cluster_id : tokens - 1);
    long long control_entry = 0;
    int elem = early_token * 7168 + early_thread * 8;
    float acc[8];
    unsigned int packed_acc[4];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        acc[j] = 0.0f;
    }
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        packed_acc[pair] = 0;
    }
    int block_taken = 0;
    if (active_experts == 8) {
        long long expert_elem_0 = (long long)0 * 7168 + (long long)elem;
        float expert_vec_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_0 + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&expert_vec_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_0 = expert_scales[early_token];
        long long expert_elem_1 = (long long)tokens * 7168 + (long long)elem;
        float expert_vec_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_1 + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_1[_blk].x), "=r"(_vld_1[_blk].y), "=r"(_vld_1[_blk].z), "=r"(_vld_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&expert_vec_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_1 = expert_scales[tokens + early_token];
        long long expert_elem_2 = (long long)(2 * tokens) * 7168 + (long long)elem;
        float expert_vec_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_2 + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_2[_blk].x), "=r"(_vld_2[_blk].y), "=r"(_vld_2[_blk].z), "=r"(_vld_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&expert_vec_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_2 = expert_scales[2 * tokens + early_token];
        long long expert_elem_3 = (long long)(3 * tokens) * 7168 + (long long)elem;
        float expert_vec_3[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_3 + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_3[_blk].x), "=r"(_vld_3[_blk].y), "=r"(_vld_3[_blk].z), "=r"(_vld_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&expert_vec_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_3 = expert_scales[3 * tokens + early_token];
        long long expert_elem_4 = (long long)(4 * tokens) * 7168 + (long long)elem;
        float expert_vec_4[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_4 + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_4[_blk].x), "=r"(_vld_4[_blk].y), "=r"(_vld_4[_blk].z), "=r"(_vld_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&expert_vec_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_4 = expert_scales[4 * tokens + early_token];
        long long expert_elem_5 = (long long)(5 * tokens) * 7168 + (long long)elem;
        float expert_vec_5[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_5 + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_5[_blk].x), "=r"(_vld_5[_blk].y), "=r"(_vld_5[_blk].z), "=r"(_vld_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&expert_vec_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_5 = expert_scales[5 * tokens + early_token];
        long long expert_elem_6 = (long long)(6 * tokens) * 7168 + (long long)elem;
        float expert_vec_6[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_6 + 0);
            uint4 _vld_6[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_6[_blk].x), "=r"(_vld_6[_blk].y), "=r"(_vld_6[_blk].z), "=r"(_vld_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&expert_vec_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_6 = expert_scales[6 * tokens + early_token];
        long long expert_elem_7 = (long long)(7 * tokens) * 7168 + (long long)elem;
        float expert_vec_7[8];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_7 + 0);
            uint4 _vld_7[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_7[_blk].x), "=r"(_vld_7[_blk].y), "=r"(_vld_7[_blk].z), "=r"(_vld_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&expert_vec_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_7 = expert_scales[7 * tokens + early_token];
        float scaled[8];
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            scaled[j_1] = expert_vec_0[j_1] * expert_scale_0;
        }
        unsigned int scaled_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_1 = 0; pair_1 < 4; pair_1++) {
            uint32_t pair_sum;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum) : "r"(packed_acc[pair_1]), "r"(scaled_bf16[pair_1]));
            packed_acc[pair_1] = pair_sum;
        }
        float scaled_0[8];
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            scaled_0[j_2] = expert_vec_1[j_2] * expert_scale_1;
        }
        unsigned int scaled_bf16_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
            scaled_bf16_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_2 = 0; pair_2 < 4; pair_2++) {
            uint32_t pair_sum_1;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_1) : "r"(packed_acc[pair_2]), "r"(scaled_bf16_0[pair_2]));
            packed_acc[pair_2] = pair_sum_1;
        }
        float scaled_1[8];
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            scaled_1[j_3] = expert_vec_2[j_3] * expert_scale_2;
        }
        unsigned int scaled_bf16_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
            scaled_bf16_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_3 = 0; pair_3 < 4; pair_3++) {
            uint32_t pair_sum_2;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_2) : "r"(packed_acc[pair_3]), "r"(scaled_bf16_1[pair_3]));
            packed_acc[pair_3] = pair_sum_2;
        }
        float scaled_2[8];
        #pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
            scaled_2[j_4] = expert_vec_3[j_4] * expert_scale_3;
        }
        unsigned int scaled_bf16_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
            scaled_bf16_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_4 = 0; pair_4 < 4; pair_4++) {
            uint32_t pair_sum_3;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_3) : "r"(packed_acc[pair_4]), "r"(scaled_bf16_2[pair_4]));
            packed_acc[pair_4] = pair_sum_3;
        }
        float scaled_3[8];
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            scaled_3[j_5] = expert_vec_4[j_5] * expert_scale_4;
        }
        unsigned int scaled_bf16_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_3[_lp*2 + 0], scaled_3[_lp*2+1 + 0]));
            scaled_bf16_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_5 = 0; pair_5 < 4; pair_5++) {
            uint32_t pair_sum_4;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_4) : "r"(packed_acc[pair_5]), "r"(scaled_bf16_3[pair_5]));
            packed_acc[pair_5] = pair_sum_4;
        }
        float scaled_4[8];
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            scaled_4[j_6] = expert_vec_5[j_6] * expert_scale_5;
        }
        unsigned int scaled_bf16_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
            scaled_bf16_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_6 = 0; pair_6 < 4; pair_6++) {
            uint32_t pair_sum_5;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_5) : "r"(packed_acc[pair_6]), "r"(scaled_bf16_4[pair_6]));
            packed_acc[pair_6] = pair_sum_5;
        }
        float scaled_5[8];
        #pragma unroll
        for (int j_7 = 0; j_7 < 8; j_7++) {
            scaled_5[j_7] = expert_vec_6[j_7] * expert_scale_6;
        }
        unsigned int scaled_bf16_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_5[_lp*2 + 0], scaled_5[_lp*2+1 + 0]));
            scaled_bf16_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_7 = 0; pair_7 < 4; pair_7++) {
            uint32_t pair_sum_6;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_6) : "r"(packed_acc[pair_7]), "r"(scaled_bf16_5[pair_7]));
            packed_acc[pair_7] = pair_sum_6;
        }
        float scaled_6[8];
        #pragma unroll
        for (int j_8 = 0; j_8 < 8; j_8++) {
            scaled_6[j_8] = expert_vec_7[j_8] * expert_scale_7;
        }
        unsigned int scaled_bf16_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
            scaled_bf16_6[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_8 = 0; pair_8 < 4; pair_8++) {
            uint32_t pair_sum_7;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_7) : "r"(packed_acc[pair_8]), "r"(scaled_bf16_6[pair_8]));
            packed_acc[pair_8] = pair_sum_7;
        }
        control_entry = workspace_tensor[24];
        block_taken = 1;
    }
    if (active_experts == 12) {
        long long expert_elem_8 = (long long)0 * 7168 + (long long)elem;
        float expert_vec_8[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_8 + 0);
            uint4 _vld_8[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_8[_blk].x), "=r"(_vld_8[_blk].y), "=r"(_vld_8[_blk].z), "=r"(_vld_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&expert_vec_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_8 = expert_scales[early_token];
        long long expert_elem_9 = (long long)tokens * 7168 + (long long)elem;
        float expert_vec_9[8];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_9 + 0);
            uint4 _vld_9[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_9[_blk].x), "=r"(_vld_9[_blk].y), "=r"(_vld_9[_blk].z), "=r"(_vld_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                    (&expert_vec_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_9 = expert_scales[tokens + early_token];
        long long expert_elem_10 = (long long)(2 * tokens) * 7168 + (long long)elem;
        float expert_vec_10[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_10 + 0);
            uint4 _vld_10[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_10[_blk].x), "=r"(_vld_10[_blk].y), "=r"(_vld_10[_blk].z), "=r"(_vld_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&expert_vec_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_10 = expert_scales[2 * tokens + early_token];
        long long expert_elem_11 = (long long)(3 * tokens) * 7168 + (long long)elem;
        float expert_vec_11[8];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_11 + 0);
            uint4 _vld_11[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_11[_blk].x), "=r"(_vld_11[_blk].y), "=r"(_vld_11[_blk].z), "=r"(_vld_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                    (&expert_vec_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_11 = expert_scales[3 * tokens + early_token];
        long long expert_elem_12 = (long long)(4 * tokens) * 7168 + (long long)elem;
        float expert_vec_12[8];
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_12 + 0);
            uint4 _vld_12[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_12[_blk].x), "=r"(_vld_12[_blk].y), "=r"(_vld_12[_blk].z), "=r"(_vld_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                    (&expert_vec_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_12 = expert_scales[4 * tokens + early_token];
        long long expert_elem_13 = (long long)(5 * tokens) * 7168 + (long long)elem;
        float expert_vec_13[8];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_13 + 0);
            uint4 _vld_13[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_13[_blk].x), "=r"(_vld_13[_blk].y), "=r"(_vld_13[_blk].z), "=r"(_vld_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                    (&expert_vec_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_13 = expert_scales[5 * tokens + early_token];
        long long expert_elem_14 = (long long)(6 * tokens) * 7168 + (long long)elem;
        float expert_vec_14[8];
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_14 + 0);
            uint4 _vld_14[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_14[_blk].x), "=r"(_vld_14[_blk].y), "=r"(_vld_14[_blk].z), "=r"(_vld_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                    (&expert_vec_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_14 = expert_scales[6 * tokens + early_token];
        long long expert_elem_15 = (long long)(7 * tokens) * 7168 + (long long)elem;
        float expert_vec_15[8];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_15 + 0);
            uint4 _vld_15[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_15[_blk].x), "=r"(_vld_15[_blk].y), "=r"(_vld_15[_blk].z), "=r"(_vld_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                    (&expert_vec_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_15 = expert_scales[7 * tokens + early_token];
        long long expert_elem_16 = (long long)(8 * tokens) * 7168 + (long long)elem;
        float expert_vec_16[8];
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_16 + 0);
            uint4 _vld_16[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_16[_blk].x), "=r"(_vld_16[_blk].y), "=r"(_vld_16[_blk].z), "=r"(_vld_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                    (&expert_vec_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_16 = expert_scales[8 * tokens + early_token];
        long long expert_elem_17 = (long long)(9 * tokens) * 7168 + (long long)elem;
        float expert_vec_17[8];
        {
            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_17 + 0);
            uint4 _vld_17[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_17[_blk].x), "=r"(_vld_17[_blk].y), "=r"(_vld_17[_blk].z), "=r"(_vld_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                    (&expert_vec_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_17 = expert_scales[9 * tokens + early_token];
        long long expert_elem_18 = (long long)(10 * tokens) * 7168 + (long long)elem;
        float expert_vec_18[8];
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_18 + 0);
            uint4 _vld_18[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_18[_blk].x), "=r"(_vld_18[_blk].y), "=r"(_vld_18[_blk].z), "=r"(_vld_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                    (&expert_vec_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_18 = expert_scales[10 * tokens + early_token];
        long long expert_elem_19 = (long long)(11 * tokens) * 7168 + (long long)elem;
        float expert_vec_19[8];
        {
            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_19 + 0);
            uint4 _vld_19[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_19[_blk].x), "=r"(_vld_19[_blk].y), "=r"(_vld_19[_blk].z), "=r"(_vld_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                    (&expert_vec_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_19 = expert_scales[11 * tokens + early_token];
        control_entry = workspace_tensor[24];
        float scaled_7[8];
        #pragma unroll
        for (int j_9 = 0; j_9 < 8; j_9++) {
            scaled_7[j_9] = expert_vec_8[j_9] * expert_scale_8;
        }
        unsigned int scaled_bf16_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_7[_lp*2 + 0], scaled_7[_lp*2+1 + 0]));
            scaled_bf16_7[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_9 = 0; pair_9 < 4; pair_9++) {
            uint32_t pair_sum_8;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_8) : "r"(packed_acc[pair_9]), "r"(scaled_bf16_7[pair_9]));
            packed_acc[pair_9] = pair_sum_8;
        }
        float scaled_7_1[8];
        #pragma unroll
        for (int j_10 = 0; j_10 < 8; j_10++) {
            scaled_7_1[j_10] = expert_vec_9[j_10] * expert_scale_9;
        }
        unsigned int scaled_bf16_7_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_7_1[_lp*2 + 0], scaled_7_1[_lp*2+1 + 0]));
            scaled_bf16_7_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_10 = 0; pair_10 < 4; pair_10++) {
            uint32_t pair_sum_9;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_9) : "r"(packed_acc[pair_10]), "r"(scaled_bf16_7_1[pair_10]));
            packed_acc[pair_10] = pair_sum_9;
        }
        float scaled_8[8];
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            scaled_8[j_11] = expert_vec_10[j_11] * expert_scale_10;
        }
        unsigned int scaled_bf16_8[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
            scaled_bf16_8[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_11 = 0; pair_11 < 4; pair_11++) {
            uint32_t pair_sum_10;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_10) : "r"(packed_acc[pair_11]), "r"(scaled_bf16_8[pair_11]));
            packed_acc[pair_11] = pair_sum_10;
        }
        float scaled_9[8];
        #pragma unroll
        for (int j_12 = 0; j_12 < 8; j_12++) {
            scaled_9[j_12] = expert_vec_11[j_12] * expert_scale_11;
        }
        unsigned int scaled_bf16_9[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_9[_lp*2 + 0], scaled_9[_lp*2+1 + 0]));
            scaled_bf16_9[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_12 = 0; pair_12 < 4; pair_12++) {
            uint32_t pair_sum_11;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_11) : "r"(packed_acc[pair_12]), "r"(scaled_bf16_9[pair_12]));
            packed_acc[pair_12] = pair_sum_11;
        }
        float scaled_10[8];
        #pragma unroll
        for (int j_13 = 0; j_13 < 8; j_13++) {
            scaled_10[j_13] = expert_vec_12[j_13] * expert_scale_12;
        }
        unsigned int scaled_bf16_10[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
            scaled_bf16_10[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_13 = 0; pair_13 < 4; pair_13++) {
            uint32_t pair_sum_12;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_12) : "r"(packed_acc[pair_13]), "r"(scaled_bf16_10[pair_13]));
            packed_acc[pair_13] = pair_sum_12;
        }
        float scaled_11[8];
        #pragma unroll
        for (int j_14 = 0; j_14 < 8; j_14++) {
            scaled_11[j_14] = expert_vec_13[j_14] * expert_scale_13;
        }
        unsigned int scaled_bf16_11[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_11[_lp*2 + 0], scaled_11[_lp*2+1 + 0]));
            scaled_bf16_11[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_14 = 0; pair_14 < 4; pair_14++) {
            uint32_t pair_sum_13;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_13) : "r"(packed_acc[pair_14]), "r"(scaled_bf16_11[pair_14]));
            packed_acc[pair_14] = pair_sum_13;
        }
        float scaled_12[8];
        #pragma unroll
        for (int j_15 = 0; j_15 < 8; j_15++) {
            scaled_12[j_15] = expert_vec_14[j_15] * expert_scale_14;
        }
        unsigned int scaled_bf16_12[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
            scaled_bf16_12[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_15 = 0; pair_15 < 4; pair_15++) {
            uint32_t pair_sum_14;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_14) : "r"(packed_acc[pair_15]), "r"(scaled_bf16_12[pair_15]));
            packed_acc[pair_15] = pair_sum_14;
        }
        float scaled_13[8];
        #pragma unroll
        for (int j_16 = 0; j_16 < 8; j_16++) {
            scaled_13[j_16] = expert_vec_15[j_16] * expert_scale_15;
        }
        unsigned int scaled_bf16_13[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_13[_lp*2 + 0], scaled_13[_lp*2+1 + 0]));
            scaled_bf16_13[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_16 = 0; pair_16 < 4; pair_16++) {
            uint32_t pair_sum_15;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_15) : "r"(packed_acc[pair_16]), "r"(scaled_bf16_13[pair_16]));
            packed_acc[pair_16] = pair_sum_15;
        }
        float scaled_14[8];
        #pragma unroll
        for (int j_17 = 0; j_17 < 8; j_17++) {
            scaled_14[j_17] = expert_vec_16[j_17] * expert_scale_16;
        }
        unsigned int scaled_bf16_14[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_14[_lp*2 + 0], scaled_14[_lp*2+1 + 0]));
            scaled_bf16_14[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_17 = 0; pair_17 < 4; pair_17++) {
            uint32_t pair_sum_16;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_16) : "r"(packed_acc[pair_17]), "r"(scaled_bf16_14[pair_17]));
            packed_acc[pair_17] = pair_sum_16;
        }
        float scaled_15[8];
        #pragma unroll
        for (int j_18 = 0; j_18 < 8; j_18++) {
            scaled_15[j_18] = expert_vec_17[j_18] * expert_scale_17;
        }
        unsigned int scaled_bf16_15[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_15[_lp*2 + 0], scaled_15[_lp*2+1 + 0]));
            scaled_bf16_15[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_18 = 0; pair_18 < 4; pair_18++) {
            uint32_t pair_sum_17;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_17) : "r"(packed_acc[pair_18]), "r"(scaled_bf16_15[pair_18]));
            packed_acc[pair_18] = pair_sum_17;
        }
        float scaled_16[8];
        #pragma unroll
        for (int j_19 = 0; j_19 < 8; j_19++) {
            scaled_16[j_19] = expert_vec_18[j_19] * expert_scale_18;
        }
        unsigned int scaled_bf16_16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_16[_lp*2 + 0], scaled_16[_lp*2+1 + 0]));
            scaled_bf16_16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_19 = 0; pair_19 < 4; pair_19++) {
            uint32_t pair_sum_18;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_18) : "r"(packed_acc[pair_19]), "r"(scaled_bf16_16[pair_19]));
            packed_acc[pair_19] = pair_sum_18;
        }
        float scaled_17[8];
        #pragma unroll
        for (int j_20 = 0; j_20 < 8; j_20++) {
            scaled_17[j_20] = expert_vec_19[j_20] * expert_scale_19;
        }
        unsigned int scaled_bf16_17[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_17[_lp*2 + 0], scaled_17[_lp*2+1 + 0]));
            scaled_bf16_17[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_20 = 0; pair_20 < 4; pair_20++) {
            uint32_t pair_sum_19;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_19) : "r"(packed_acc[pair_20]), "r"(scaled_bf16_17[pair_20]));
            packed_acc[pair_20] = pair_sum_19;
        }
        block_taken = 1;
    }
    if (active_experts == 16) {
        long long expert_elem_20 = (long long)0 * 7168 + (long long)elem;
        float expert_vec_20[8];
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_20 + 0);
            uint4 _vld_20[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_20[_blk].x), "=r"(_vld_20[_blk].y), "=r"(_vld_20[_blk].z), "=r"(_vld_20[_blk].w) : "l"((const void*)(_vptr_20 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                    (&expert_vec_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_20 = expert_scales[early_token];
        long long expert_elem_21 = (long long)tokens * 7168 + (long long)elem;
        float expert_vec_21[8];
        {
            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_21 + 0);
            uint4 _vld_21[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_21[_blk].x), "=r"(_vld_21[_blk].y), "=r"(_vld_21[_blk].z), "=r"(_vld_21[_blk].w) : "l"((const void*)(_vptr_21 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
                    (&expert_vec_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_21 = expert_scales[tokens + early_token];
        long long expert_elem_22 = (long long)(2 * tokens) * 7168 + (long long)elem;
        float expert_vec_22[8];
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_22 + 0);
            uint4 _vld_22[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_22[_blk].x), "=r"(_vld_22[_blk].y), "=r"(_vld_22[_blk].z), "=r"(_vld_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_22[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
                    (&expert_vec_22[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_22 = expert_scales[2 * tokens + early_token];
        long long expert_elem_23 = (long long)(3 * tokens) * 7168 + (long long)elem;
        float expert_vec_23[8];
        {
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_23 + 0);
            uint4 _vld_23[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_23[_blk].x), "=r"(_vld_23[_blk].y), "=r"(_vld_23[_blk].z), "=r"(_vld_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_23[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
                    (&expert_vec_23[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_23 = expert_scales[3 * tokens + early_token];
        long long expert_elem_24 = (long long)(4 * tokens) * 7168 + (long long)elem;
        float expert_vec_24[8];
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_24 + 0);
            uint4 _vld_24[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_24[_blk].x), "=r"(_vld_24[_blk].y), "=r"(_vld_24[_blk].z), "=r"(_vld_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_24[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                    (&expert_vec_24[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_24 = expert_scales[4 * tokens + early_token];
        long long expert_elem_25 = (long long)(5 * tokens) * 7168 + (long long)elem;
        float expert_vec_25[8];
        {
            const uint4* _vptr_25 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_25 + 0);
            uint4 _vld_25[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_25[_blk].x), "=r"(_vld_25[_blk].y), "=r"(_vld_25[_blk].z), "=r"(_vld_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_25[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                    (&expert_vec_25[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_25 = expert_scales[5 * tokens + early_token];
        long long expert_elem_26 = (long long)(6 * tokens) * 7168 + (long long)elem;
        float expert_vec_26[8];
        {
            const uint4* _vptr_26 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_26 + 0);
            uint4 _vld_26[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_26[_blk].x), "=r"(_vld_26[_blk].y), "=r"(_vld_26[_blk].z), "=r"(_vld_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
                    (&expert_vec_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_26 = expert_scales[6 * tokens + early_token];
        long long expert_elem_27 = (long long)(7 * tokens) * 7168 + (long long)elem;
        float expert_vec_27[8];
        {
            const uint4* _vptr_27 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_27 + 0);
            uint4 _vld_27[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_27[_blk].x), "=r"(_vld_27[_blk].y), "=r"(_vld_27[_blk].z), "=r"(_vld_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
                    (&expert_vec_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_27 = expert_scales[7 * tokens + early_token];
        long long expert_elem_28 = (long long)(8 * tokens) * 7168 + (long long)elem;
        float expert_vec_28[8];
        {
            const uint4* _vptr_28 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_28 + 0);
            uint4 _vld_28[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_28[_blk].x), "=r"(_vld_28[_blk].y), "=r"(_vld_28[_blk].z), "=r"(_vld_28[_blk].w) : "l"((const void*)(_vptr_28 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) << 16);
                    (&expert_vec_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_28 = expert_scales[8 * tokens + early_token];
        long long expert_elem_29 = (long long)(9 * tokens) * 7168 + (long long)elem;
        float expert_vec_29[8];
        {
            const uint4* _vptr_29 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_29 + 0);
            uint4 _vld_29[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_29[_blk].x), "=r"(_vld_29[_blk].y), "=r"(_vld_29[_blk].z), "=r"(_vld_29[_blk].w) : "l"((const void*)(_vptr_29 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
                    (&expert_vec_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_29 = expert_scales[9 * tokens + early_token];
        long long expert_elem_30 = (long long)(10 * tokens) * 7168 + (long long)elem;
        float expert_vec_30[8];
        {
            const uint4* _vptr_30 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_30 + 0);
            uint4 _vld_30[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_30[_blk].x), "=r"(_vld_30[_blk].y), "=r"(_vld_30[_blk].z), "=r"(_vld_30[_blk].w) : "l"((const void*)(_vptr_30 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_30[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                    (&expert_vec_30[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_30 = expert_scales[10 * tokens + early_token];
        long long expert_elem_31 = (long long)(11 * tokens) * 7168 + (long long)elem;
        float expert_vec_31[8];
        {
            const uint4* _vptr_31 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_31 + 0);
            uint4 _vld_31[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_31[_blk].x), "=r"(_vld_31[_blk].y), "=r"(_vld_31[_blk].z), "=r"(_vld_31[_blk].w) : "l"((const void*)(_vptr_31 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_31[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                    (&expert_vec_31[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_31 = expert_scales[11 * tokens + early_token];
        long long expert_elem_32 = (long long)(12 * tokens) * 7168 + (long long)elem;
        float expert_vec_32[8];
        {
            const uint4* _vptr_32 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_32 + 0);
            uint4 _vld_32[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_32[_blk].x), "=r"(_vld_32[_blk].y), "=r"(_vld_32[_blk].z), "=r"(_vld_32[_blk].w) : "l"((const void*)(_vptr_32 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_32[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                    (&expert_vec_32[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_32 = expert_scales[12 * tokens + early_token];
        long long expert_elem_33 = (long long)(13 * tokens) * 7168 + (long long)elem;
        float expert_vec_33[8];
        {
            const uint4* _vptr_33 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_33 + 0);
            uint4 _vld_33[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_33[_blk].x), "=r"(_vld_33[_blk].y), "=r"(_vld_33[_blk].z), "=r"(_vld_33[_blk].w) : "l"((const void*)(_vptr_33 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_33[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                    (&expert_vec_33[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_33 = expert_scales[13 * tokens + early_token];
        long long expert_elem_34 = (long long)(14 * tokens) * 7168 + (long long)elem;
        float expert_vec_34[8];
        {
            const uint4* _vptr_34 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_34 + 0);
            uint4 _vld_34[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_34[_blk].x), "=r"(_vld_34[_blk].y), "=r"(_vld_34[_blk].z), "=r"(_vld_34[_blk].w) : "l"((const void*)(_vptr_34 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_34[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                    (&expert_vec_34[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_34 = expert_scales[14 * tokens + early_token];
        long long expert_elem_35 = (long long)(15 * tokens) * 7168 + (long long)elem;
        float expert_vec_35[8];
        {
            const uint4* _vptr_35 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_35 + 0);
            uint4 _vld_35[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_35[_blk].x), "=r"(_vld_35[_blk].y), "=r"(_vld_35[_blk].z), "=r"(_vld_35[_blk].w) : "l"((const void*)(_vptr_35 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec_35[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
                    (&expert_vec_35[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale_35 = expert_scales[15 * tokens + early_token];
        control_entry = workspace_tensor[24];
        float scaled_18[8];
        #pragma unroll
        for (int j_21 = 0; j_21 < 8; j_21++) {
            scaled_18[j_21] = expert_vec_20[j_21] * expert_scale_20;
        }
        unsigned int scaled_bf16_18[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_18[_lp*2 + 0], scaled_18[_lp*2+1 + 0]));
            scaled_bf16_18[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_21 = 0; pair_21 < 4; pair_21++) {
            uint32_t pair_sum_20;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_20) : "r"(packed_acc[pair_21]), "r"(scaled_bf16_18[pair_21]));
            packed_acc[pair_21] = pair_sum_20;
        }
        float scaled_18_1[8];
        #pragma unroll
        for (int j_22 = 0; j_22 < 8; j_22++) {
            scaled_18_1[j_22] = expert_vec_21[j_22] * expert_scale_21;
        }
        unsigned int scaled_bf16_18_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_18_1[_lp*2 + 0], scaled_18_1[_lp*2+1 + 0]));
            scaled_bf16_18_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_22 = 0; pair_22 < 4; pair_22++) {
            uint32_t pair_sum_21;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_21) : "r"(packed_acc[pair_22]), "r"(scaled_bf16_18_1[pair_22]));
            packed_acc[pair_22] = pair_sum_21;
        }
        float scaled_19[8];
        #pragma unroll
        for (int j_23 = 0; j_23 < 8; j_23++) {
            scaled_19[j_23] = expert_vec_22[j_23] * expert_scale_22;
        }
        unsigned int scaled_bf16_19[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_19[_lp*2 + 0], scaled_19[_lp*2+1 + 0]));
            scaled_bf16_19[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_23 = 0; pair_23 < 4; pair_23++) {
            uint32_t pair_sum_22;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_22) : "r"(packed_acc[pair_23]), "r"(scaled_bf16_19[pair_23]));
            packed_acc[pair_23] = pair_sum_22;
        }
        float scaled_20[8];
        #pragma unroll
        for (int j_24 = 0; j_24 < 8; j_24++) {
            scaled_20[j_24] = expert_vec_23[j_24] * expert_scale_23;
        }
        unsigned int scaled_bf16_20[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_20[_lp*2 + 0], scaled_20[_lp*2+1 + 0]));
            scaled_bf16_20[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_24 = 0; pair_24 < 4; pair_24++) {
            uint32_t pair_sum_23;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_23) : "r"(packed_acc[pair_24]), "r"(scaled_bf16_20[pair_24]));
            packed_acc[pair_24] = pair_sum_23;
        }
        float scaled_21[8];
        #pragma unroll
        for (int j_25 = 0; j_25 < 8; j_25++) {
            scaled_21[j_25] = expert_vec_24[j_25] * expert_scale_24;
        }
        unsigned int scaled_bf16_21[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_21[_lp*2 + 0], scaled_21[_lp*2+1 + 0]));
            scaled_bf16_21[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_25 = 0; pair_25 < 4; pair_25++) {
            uint32_t pair_sum_24;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_24) : "r"(packed_acc[pair_25]), "r"(scaled_bf16_21[pair_25]));
            packed_acc[pair_25] = pair_sum_24;
        }
        float scaled_22[8];
        #pragma unroll
        for (int j_26 = 0; j_26 < 8; j_26++) {
            scaled_22[j_26] = expert_vec_25[j_26] * expert_scale_25;
        }
        unsigned int scaled_bf16_22[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_22[_lp*2 + 0], scaled_22[_lp*2+1 + 0]));
            scaled_bf16_22[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_26 = 0; pair_26 < 4; pair_26++) {
            uint32_t pair_sum_25;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_25) : "r"(packed_acc[pair_26]), "r"(scaled_bf16_22[pair_26]));
            packed_acc[pair_26] = pair_sum_25;
        }
        float scaled_23[8];
        #pragma unroll
        for (int j_27 = 0; j_27 < 8; j_27++) {
            scaled_23[j_27] = expert_vec_26[j_27] * expert_scale_26;
        }
        unsigned int scaled_bf16_23[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_23[_lp*2 + 0], scaled_23[_lp*2+1 + 0]));
            scaled_bf16_23[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_27 = 0; pair_27 < 4; pair_27++) {
            uint32_t pair_sum_26;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_26) : "r"(packed_acc[pair_27]), "r"(scaled_bf16_23[pair_27]));
            packed_acc[pair_27] = pair_sum_26;
        }
        float scaled_24[8];
        #pragma unroll
        for (int j_28 = 0; j_28 < 8; j_28++) {
            scaled_24[j_28] = expert_vec_27[j_28] * expert_scale_27;
        }
        unsigned int scaled_bf16_24[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_24[_lp*2 + 0], scaled_24[_lp*2+1 + 0]));
            scaled_bf16_24[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_28 = 0; pair_28 < 4; pair_28++) {
            uint32_t pair_sum_27;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_27) : "r"(packed_acc[pair_28]), "r"(scaled_bf16_24[pair_28]));
            packed_acc[pair_28] = pair_sum_27;
        }
        float scaled_25[8];
        #pragma unroll
        for (int j_29 = 0; j_29 < 8; j_29++) {
            scaled_25[j_29] = expert_vec_28[j_29] * expert_scale_28;
        }
        unsigned int scaled_bf16_25[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_25[_lp*2 + 0], scaled_25[_lp*2+1 + 0]));
            scaled_bf16_25[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_29 = 0; pair_29 < 4; pair_29++) {
            uint32_t pair_sum_28;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_28) : "r"(packed_acc[pair_29]), "r"(scaled_bf16_25[pair_29]));
            packed_acc[pair_29] = pair_sum_28;
        }
        float scaled_26[8];
        #pragma unroll
        for (int j_30 = 0; j_30 < 8; j_30++) {
            scaled_26[j_30] = expert_vec_29[j_30] * expert_scale_29;
        }
        unsigned int scaled_bf16_26[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_26[_lp*2 + 0], scaled_26[_lp*2+1 + 0]));
            scaled_bf16_26[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_30 = 0; pair_30 < 4; pair_30++) {
            uint32_t pair_sum_29;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_29) : "r"(packed_acc[pair_30]), "r"(scaled_bf16_26[pair_30]));
            packed_acc[pair_30] = pair_sum_29;
        }
        float scaled_27[8];
        #pragma unroll
        for (int j_31 = 0; j_31 < 8; j_31++) {
            scaled_27[j_31] = expert_vec_30[j_31] * expert_scale_30;
        }
        unsigned int scaled_bf16_27[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_27[_lp*2 + 0], scaled_27[_lp*2+1 + 0]));
            scaled_bf16_27[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_31 = 0; pair_31 < 4; pair_31++) {
            uint32_t pair_sum_30;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_30) : "r"(packed_acc[pair_31]), "r"(scaled_bf16_27[pair_31]));
            packed_acc[pair_31] = pair_sum_30;
        }
        float scaled_28[8];
        #pragma unroll
        for (int j_32 = 0; j_32 < 8; j_32++) {
            scaled_28[j_32] = expert_vec_31[j_32] * expert_scale_31;
        }
        unsigned int scaled_bf16_28[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_28[_lp*2 + 0], scaled_28[_lp*2+1 + 0]));
            scaled_bf16_28[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_32 = 0; pair_32 < 4; pair_32++) {
            uint32_t pair_sum_31;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_31) : "r"(packed_acc[pair_32]), "r"(scaled_bf16_28[pair_32]));
            packed_acc[pair_32] = pair_sum_31;
        }
        float scaled_29[8];
        #pragma unroll
        for (int j_33 = 0; j_33 < 8; j_33++) {
            scaled_29[j_33] = expert_vec_32[j_33] * expert_scale_32;
        }
        unsigned int scaled_bf16_29[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_29[_lp*2 + 0], scaled_29[_lp*2+1 + 0]));
            scaled_bf16_29[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_33 = 0; pair_33 < 4; pair_33++) {
            uint32_t pair_sum_32;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_32) : "r"(packed_acc[pair_33]), "r"(scaled_bf16_29[pair_33]));
            packed_acc[pair_33] = pair_sum_32;
        }
        float scaled_30[8];
        #pragma unroll
        for (int j_34 = 0; j_34 < 8; j_34++) {
            scaled_30[j_34] = expert_vec_33[j_34] * expert_scale_33;
        }
        unsigned int scaled_bf16_30[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_30[_lp*2 + 0], scaled_30[_lp*2+1 + 0]));
            scaled_bf16_30[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_34 = 0; pair_34 < 4; pair_34++) {
            uint32_t pair_sum_33;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_33) : "r"(packed_acc[pair_34]), "r"(scaled_bf16_30[pair_34]));
            packed_acc[pair_34] = pair_sum_33;
        }
        float scaled_31[8];
        #pragma unroll
        for (int j_35 = 0; j_35 < 8; j_35++) {
            scaled_31[j_35] = expert_vec_34[j_35] * expert_scale_34;
        }
        unsigned int scaled_bf16_31[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_31[_lp*2 + 0], scaled_31[_lp*2+1 + 0]));
            scaled_bf16_31[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_35 = 0; pair_35 < 4; pair_35++) {
            uint32_t pair_sum_34;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_34) : "r"(packed_acc[pair_35]), "r"(scaled_bf16_31[pair_35]));
            packed_acc[pair_35] = pair_sum_34;
        }
        float scaled_32[8];
        #pragma unroll
        for (int j_36 = 0; j_36 < 8; j_36++) {
            scaled_32[j_36] = expert_vec_35[j_36] * expert_scale_35;
        }
        unsigned int scaled_bf16_32[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_32[_lp*2 + 0], scaled_32[_lp*2+1 + 0]));
            scaled_bf16_32[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_36 = 0; pair_36 < 4; pair_36++) {
            uint32_t pair_sum_35;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_35) : "r"(packed_acc[pair_36]), "r"(scaled_bf16_32[pair_36]));
            packed_acc[pair_36] = pair_sum_35;
        }
        block_taken = 1;
    }
    if (block_taken == 0) {
        control_entry = workspace_tensor[24];
    }
    int expert_loop_end = ((block_taken == 1) ? 0 : active_experts);
    #pragma unroll 4
    for (int expert = 0; expert < expert_loop_end; expert++) {
        long long expert_elem = (long long)expert * (long long)tokens * 7168 + (long long)elem;
        float expert_vec[8];
        {
            const uint4* _vptr_36 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem + 0);
            uint4 _vld_36[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_36[_blk].x), "=r"(_vld_36[_blk].y), "=r"(_vld_36[_blk].z), "=r"(_vld_36[_blk].w) : "l"((const void*)(_vptr_36 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&expert_vec[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                    (&expert_vec[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale = expert_scales[expert * tokens + early_token];
        float scaled_33[8];
        #pragma unroll
        for (int j_37 = 0; j_37 < 8; j_37++) {
            scaled_33[j_37] = expert_vec[j_37] * expert_scale;
        }
        unsigned int scaled_bf16_33[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_33[_lp*2 + 0], scaled_33[_lp*2+1 + 0]));
            scaled_bf16_33[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_37 = 0; pair_37 < 4; pair_37++) {
            uint32_t pair_sum_36;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_36) : "r"(packed_acc[pair_37]), "r"(scaled_bf16_33[pair_37]));
            packed_acc[pair_37] = pair_sum_36;
        }
    }
    long long workspace_entry = workspace_tensor[16 + world_rank];
    long long owner_entry = workspace_tensor[16 + early_thread / 112];
    float _vec_load_37[8];
    {
        const uint4* _vptr_37 = reinterpret_cast<const uint4*>(token_input + elem + 0);
        uint4 _vld_37[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_vld_37[_blk].x), "=r"(_vld_37[_blk].y), "=r"(_vld_37[_blk].z), "=r"(_vld_37[_blk].w) : "l"((const void*)(_vptr_37 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
            uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_37[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                (&_vec_load_37[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
            }
        }
    }
    int _vec_load_38[1];
    {
        uint32_t _scalar_bits_38;
        asm volatile("ld.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_38) : "l"((const void*)(reinterpret_cast<int*>(control_entry) + 2 + (0))) : "memory");
        _vec_load_38[0] = (int32_t)_scalar_bits_38;
    }
    int _vec_load_39[1];
    {
        uint32_t _scalar_bits_39;
        asm volatile("ld.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_39) : "l"((const void*)(reinterpret_cast<int*>(control_entry) + 3 + (0))) : "memory");
        _vec_load_39[0] = (int32_t)_scalar_bits_39;
    }
    int _vec_load_40[1];
    {
        uint32_t _scalar_bits_40;
        asm volatile("ld.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_40) : "l"((const void*)(reinterpret_cast<int*>(control_entry) + 4 + (0))) : "memory");
        _vec_load_40[0] = (int32_t)_scalar_bits_40;
    }
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&acc[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_acc[_pair]) << 16);
        (&acc[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_acc[_pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int j_38 = 0; j_38 < 8; j_38++) {
        acc[j_38] = acc[j_38] + _vec_load_37[j_38];
    }
    unsigned int scaled_bf16_34[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        scaled_bf16_34[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&acc[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_34[_pair]) << 16);
        (&acc[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_34[_pair]) & 0xffff0000u);
    }
    #pragma unroll
    for (int j_39 = 0; j_39 < 8; j_39++) {
        acc[j_39] = ((acc[j_39] == 0.0f) ? 0.0f : acc[j_39]);
    }
    int rank = world_rank;
    long long control_address = control_entry;
    int* control = reinterpret_cast<int*>(control_address);
    unsigned int* completion = reinterpret_cast<unsigned int*>(control_address);
    int* flag_addr = control + 2;
    int* comm_size_addr = control + 3;
    int* clear_addr = control + 4;
    long long comm_stride_elems = (long long)_vec_load_39[0] / 2;
    long long workspace_address = workspace_entry;
    __nv_bfloat16* workspace_local = reinterpret_cast<__nv_bfloat16*>(workspace_address);
    int flag = _vec_load_38[0];
    int clear_size = _vec_load_40[0];
    int data_epoch = flag % 3;
    int clear_epoch = (flag + 2) % 3;
    int cluster_thread = cta_rank * 224 + tid;
    int token_stride = num_clusters;
    int access_stride = token_stride * 896;
    int total_access = tokens * 896;
    int first_access = cluster_id * 896 + (unsigned int)cluster_thread;
    int rs_items = tokens * 112;
    int publish_start = cluster_id;
    if ((int)cluster_id < tokens) {
        int owner = cluster_thread / 112;
        long long slot = (long long)data_epoch * comm_stride_elems + (long long)rank * (long long)rs_items * 8 + (long long)((int)cluster_id * 112 + (cluster_thread - owner * 112)) * 8;
        unsigned int scaled_bf16_33_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            scaled_bf16_33_1[_lp] = *(uint32_t*)&_bf2;
        }
        if (owner == 0) {
            {
                auto _gst_41 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_41));
                reinterpret_cast<int4*>(_gst_41)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 1) {
            {
                auto _gst_42 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_42));
                reinterpret_cast<int4*>(_gst_42)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 2) {
            {
                auto _gst_43 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_43));
                reinterpret_cast<int4*>(_gst_43)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 3) {
            {
                auto _gst_44 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_44));
                reinterpret_cast<int4*>(_gst_44)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 4) {
            {
                auto _gst_45 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_45));
                reinterpret_cast<int4*>(_gst_45)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 5) {
            {
                auto _gst_46 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_46));
                reinterpret_cast<int4*>(_gst_46)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 6) {
            {
                auto _gst_47 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_47));
                reinterpret_cast<int4*>(_gst_47)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
        if (owner == 7) {
            {
                auto _gst_48 = reinterpret_cast<__nv_bfloat16*>(owner_entry) + slot;
                __builtin_assume(__isGlobal(_gst_48));
                reinterpret_cast<int4*>(_gst_48)[0] = reinterpret_cast<int4*>(scaled_bf16_33_1)[0];
            }
        }
    }
    publish_start = (int)(cluster_id + num_clusters);
    #pragma unroll 1
    for (int token = publish_start; token < tokens; token += (int)num_clusters) {
        if (token + (int)num_clusters >= tokens) {
            int pub_rs_src = tid / 28;
            int pub_rs_item = cta_rank * 28 + tid % 28;
            __nv_bfloat16* pub_rs_base = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[16 + pub_rs_src]);
            unsigned int pub_rs_words[4];
            uint32_t sysv_ld[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld[0]), "=r"(sysv_ld[1]), "=r"(sysv_ld[2]), "=r"(sysv_ld[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(pub_rs_src * rs_items + (int)cluster_id * 112 + pub_rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word = 0; word < 4; word++) {
                pub_rs_words[word] = sysv_ld[word];
            }
            int pending = 0;
            #pragma unroll
            for (int word_1 = 0; word_1 < 4; word_1++) {
                pending = (((pub_rs_words[word_1] & 65535) == 32768) ? 1 : pending);
                pending = ((pub_rs_words[word_1] >> 16 == 32768) ? 1 : pending);
            }
            if (pending == 1) {
                uint32_t sysv_poll_group[4];
                uint32_t _sysv_pending_49 = 1u;
                do {
                    if (_sysv_pending_49 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group[0]), "=r"(sysv_poll_group[1]), "=r"(sysv_poll_group[2]), "=r"(sysv_poll_group[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(pub_rs_src * rs_items + (int)cluster_id * 112 + pub_rs_item) * 8)) : "memory");
                    }
                    if (_sysv_pending_49 & 1u) {
                        if (!((((sysv_poll_group[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_49 &= ~1u;
                    }
                } while (_sysv_pending_49 != 0u);
                #pragma unroll
                for (int word_2 = 0; word_2 < 4; word_2++) {
                    pub_rs_words[word_2] = sysv_poll_group[word_2];
                }
            }
            unsigned int ag_words[4];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rs_stage_words_addr + (unsigned int)(tid * 16)), "r"(pub_rs_words[0]), "r"(pub_rs_words[1]), "r"(pub_rs_words[2]), "r"(pub_rs_words[3]) : "memory");
            __syncthreads();
            if (tid < 112) {
                float lanes[2];
                unsigned int word_3[1];
                word_3[0] = rs_stage_words[tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                float received_lanes[2];
                word_3[0] = rs_stage_words[112 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_0 = 0; j_1_0 < 2; j_1_0++) {
                    lanes[j_1_0] = lanes[j_1_0] + received_lanes[j_1_0];
                }
                unsigned int lanes_bf16_7_0[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_0[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_0[_pair]) & 0xffff0000u);
                }
                word_3[0] = rs_stage_words[224 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 2; j_1_1++) {
                    lanes[j_1_1] = lanes[j_1_1] + received_lanes[j_1_1];
                }
                unsigned int lanes_bf16_7_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_1[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_1[_pair]) & 0xffff0000u);
                }
                word_3[0] = rs_stage_words[336 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 2; j_1_2++) {
                    lanes[j_1_2] = lanes[j_1_2] + received_lanes[j_1_2];
                }
                unsigned int lanes_bf16_7_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_2[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_2[_pair]) & 0xffff0000u);
                }
                word_3[0] = rs_stage_words[448 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_3 = 0; j_1_3 < 2; j_1_3++) {
                    lanes[j_1_3] = lanes[j_1_3] + received_lanes[j_1_3];
                }
                unsigned int lanes_bf16_7_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_3[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_3[_pair]) & 0xffff0000u);
                }
                word_3[0] = rs_stage_words[560 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_4 = 0; j_1_4 < 2; j_1_4++) {
                    lanes[j_1_4] = lanes[j_1_4] + received_lanes[j_1_4];
                }
                unsigned int lanes_bf16_7_4[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_4[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_4[_pair]) & 0xffff0000u);
                }
                word_3[0] = rs_stage_words[672 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_5 = 0; j_1_5 < 2; j_1_5++) {
                    lanes[j_1_5] = lanes[j_1_5] + received_lanes[j_1_5];
                }
                unsigned int lanes_bf16_7_5[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_5[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_5[_pair]) & 0xffff0000u);
                }
                word_3[0] = rs_stage_words[784 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_6 = 0; j_1_6 < 2; j_1_6++) {
                    lanes[j_1_6] = lanes[j_1_6] + received_lanes[j_1_6];
                }
                unsigned int lanes_bf16_7_6[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_7_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_6[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_6[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_34_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    scaled_bf16_34_1[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low = scaled_bf16_34_1[0] & 65535;
                unsigned int high = scaled_bf16_34_1[0] >> 16;
                low = ((low == 65535) ? 65534 : low);
                high = ((high == 65535) ? 65534 : high);
                low = ((low == 32768) ? 65535 : low);
                high = ((high == 32768) ? 65535 : high);
                rs_sum_words[tid] = low | high << 16;
            }
            __syncthreads();
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&ag_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words[(0) + 3]))
                : "r"(rs_sum_words_addr + (unsigned int)(tid % 28 * 16)) : "memory");
            {
                auto _gst_50 = pub_rs_base + ((long long)data_epoch * comm_stride_elems + (long long)tokens * 7168 + (long long)((int)cluster_id * 896 + rank * 112 + pub_rs_item) * 8);
                __builtin_assume(__isGlobal(_gst_50));
                reinterpret_cast<int4*>(_gst_50)[0] = reinterpret_cast<int4*>(ag_words)[0];
            }
        }
        int elem_2 = token * 7168 + cluster_thread * 8;
        float acc_0[8];
        unsigned int packed_acc_0[4];
        #pragma unroll
        for (int j_40 = 0; j_40 < 8; j_40++) {
            acc_0[j_40] = 0.0f;
        }
        #pragma unroll
        for (int pair_38 = 0; pair_38 < 4; pair_38++) {
            packed_acc_0[pair_38] = 0;
        }
        int block_taken_0 = 0;
        if (active_experts == 8) {
            long long expert_elem_36 = (long long)0 * 7168 + (long long)elem_2;
            float expert_vec_36[8];
            {
                const uint4* _vptr_51 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_36 + 0);
                uint4 _vld_51[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_51[_blk].x), "=r"(_vld_51[_blk].y), "=r"(_vld_51[_blk].z), "=r"(_vld_51[_blk].w) : "l"((const void*)(_vptr_51 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_36[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
                        (&expert_vec_36[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_36 = expert_scales[token];
            long long expert_elem_37 = (long long)tokens * 7168 + (long long)elem_2;
            float expert_vec_37[8];
            {
                const uint4* _vptr_52 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_37 + 0);
                uint4 _vld_52[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_52[_blk].x), "=r"(_vld_52[_blk].y), "=r"(_vld_52[_blk].z), "=r"(_vld_52[_blk].w) : "l"((const void*)(_vptr_52 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_52 = reinterpret_cast<uint32_t*>(&_vld_52[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_37[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) << 16);
                        (&expert_vec_37[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_37 = expert_scales[tokens + token];
            long long expert_elem_38 = (long long)(2 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_38[8];
            {
                const uint4* _vptr_53 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_38 + 0);
                uint4 _vld_53[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_53[_blk].x), "=r"(_vld_53[_blk].y), "=r"(_vld_53[_blk].z), "=r"(_vld_53[_blk].w) : "l"((const void*)(_vptr_53 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_53 = reinterpret_cast<uint32_t*>(&_vld_53[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_38[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) << 16);
                        (&expert_vec_38[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_38 = expert_scales[2 * tokens + token];
            long long expert_elem_39 = (long long)(3 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_39[8];
            {
                const uint4* _vptr_54 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_39 + 0);
                uint4 _vld_54[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_54[_blk].x), "=r"(_vld_54[_blk].y), "=r"(_vld_54[_blk].z), "=r"(_vld_54[_blk].w) : "l"((const void*)(_vptr_54 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_54 = reinterpret_cast<uint32_t*>(&_vld_54[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_39[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) << 16);
                        (&expert_vec_39[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_39 = expert_scales[3 * tokens + token];
            long long expert_elem_40 = (long long)(4 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_40[8];
            {
                const uint4* _vptr_55 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_40 + 0);
                uint4 _vld_55[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_55[_blk].x), "=r"(_vld_55[_blk].y), "=r"(_vld_55[_blk].z), "=r"(_vld_55[_blk].w) : "l"((const void*)(_vptr_55 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_40[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                        (&expert_vec_40[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_40 = expert_scales[4 * tokens + token];
            long long expert_elem_41 = (long long)(5 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_41[8];
            {
                const uint4* _vptr_56 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_41 + 0);
                uint4 _vld_56[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_56[_blk].x), "=r"(_vld_56[_blk].y), "=r"(_vld_56[_blk].z), "=r"(_vld_56[_blk].w) : "l"((const void*)(_vptr_56 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_41[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                        (&expert_vec_41[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_41 = expert_scales[5 * tokens + token];
            long long expert_elem_42 = (long long)(6 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_42[8];
            {
                const uint4* _vptr_57 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_42 + 0);
                uint4 _vld_57[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_57[_blk].x), "=r"(_vld_57[_blk].y), "=r"(_vld_57[_blk].z), "=r"(_vld_57[_blk].w) : "l"((const void*)(_vptr_57 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_42[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                        (&expert_vec_42[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_42 = expert_scales[6 * tokens + token];
            long long expert_elem_43 = (long long)(7 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_43[8];
            {
                const uint4* _vptr_58 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_43 + 0);
                uint4 _vld_58[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_58[_blk].x), "=r"(_vld_58[_blk].y), "=r"(_vld_58[_blk].z), "=r"(_vld_58[_blk].w) : "l"((const void*)(_vptr_58 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_43[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                        (&expert_vec_43[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_43 = expert_scales[7 * tokens + token];
            float scaled_34[8];
            #pragma unroll
            for (int j_41 = 0; j_41 < 8; j_41++) {
                scaled_34[j_41] = expert_vec_36[j_41] * expert_scale_36;
            }
            unsigned int scaled_bf16_35[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_34[_lp*2 + 0], scaled_34[_lp*2+1 + 0]));
                scaled_bf16_35[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_39 = 0; pair_39 < 4; pair_39++) {
                uint32_t pair_sum_37;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_37) : "r"(packed_acc_0[pair_39]), "r"(scaled_bf16_35[pair_39]));
                packed_acc_0[pair_39] = pair_sum_37;
            }
            float scaled_33_1[8];
            #pragma unroll
            for (int j_42 = 0; j_42 < 8; j_42++) {
                scaled_33_1[j_42] = expert_vec_37[j_42] * expert_scale_37;
            }
            unsigned int scaled_bf16_36[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_33_1[_lp*2 + 0], scaled_33_1[_lp*2+1 + 0]));
                scaled_bf16_36[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_40 = 0; pair_40 < 4; pair_40++) {
                uint32_t pair_sum_38;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_38) : "r"(packed_acc_0[pair_40]), "r"(scaled_bf16_36[pair_40]));
                packed_acc_0[pair_40] = pair_sum_38;
            }
            float scaled_34_1[8];
            #pragma unroll
            for (int j_43 = 0; j_43 < 8; j_43++) {
                scaled_34_1[j_43] = expert_vec_38[j_43] * expert_scale_38;
            }
            unsigned int scaled_bf16_37[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_34_1[_lp*2 + 0], scaled_34_1[_lp*2+1 + 0]));
                scaled_bf16_37[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_41 = 0; pair_41 < 4; pair_41++) {
                uint32_t pair_sum_39;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_39) : "r"(packed_acc_0[pair_41]), "r"(scaled_bf16_37[pair_41]));
                packed_acc_0[pair_41] = pair_sum_39;
            }
            float scaled_35[8];
            #pragma unroll
            for (int j_44 = 0; j_44 < 8; j_44++) {
                scaled_35[j_44] = expert_vec_39[j_44] * expert_scale_39;
            }
            unsigned int scaled_bf16_38[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_35[_lp*2 + 0], scaled_35[_lp*2+1 + 0]));
                scaled_bf16_38[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_42 = 0; pair_42 < 4; pair_42++) {
                uint32_t pair_sum_40;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_40) : "r"(packed_acc_0[pair_42]), "r"(scaled_bf16_38[pair_42]));
                packed_acc_0[pair_42] = pair_sum_40;
            }
            float scaled_36[8];
            #pragma unroll
            for (int j_45 = 0; j_45 < 8; j_45++) {
                scaled_36[j_45] = expert_vec_40[j_45] * expert_scale_40;
            }
            unsigned int scaled_bf16_39[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_36[_lp*2 + 0], scaled_36[_lp*2+1 + 0]));
                scaled_bf16_39[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_43 = 0; pair_43 < 4; pair_43++) {
                uint32_t pair_sum_41;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_41) : "r"(packed_acc_0[pair_43]), "r"(scaled_bf16_39[pair_43]));
                packed_acc_0[pair_43] = pair_sum_41;
            }
            float scaled_37[8];
            #pragma unroll
            for (int j_46 = 0; j_46 < 8; j_46++) {
                scaled_37[j_46] = expert_vec_41[j_46] * expert_scale_41;
            }
            unsigned int scaled_bf16_40[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_37[_lp*2 + 0], scaled_37[_lp*2+1 + 0]));
                scaled_bf16_40[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_44 = 0; pair_44 < 4; pair_44++) {
                uint32_t pair_sum_42;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_42) : "r"(packed_acc_0[pair_44]), "r"(scaled_bf16_40[pair_44]));
                packed_acc_0[pair_44] = pair_sum_42;
            }
            float scaled_38[8];
            #pragma unroll
            for (int j_47 = 0; j_47 < 8; j_47++) {
                scaled_38[j_47] = expert_vec_42[j_47] * expert_scale_42;
            }
            unsigned int scaled_bf16_41[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_38[_lp*2 + 0], scaled_38[_lp*2+1 + 0]));
                scaled_bf16_41[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_45 = 0; pair_45 < 4; pair_45++) {
                uint32_t pair_sum_43;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_43) : "r"(packed_acc_0[pair_45]), "r"(scaled_bf16_41[pair_45]));
                packed_acc_0[pair_45] = pair_sum_43;
            }
            float scaled_39[8];
            #pragma unroll
            for (int j_48 = 0; j_48 < 8; j_48++) {
                scaled_39[j_48] = expert_vec_43[j_48] * expert_scale_43;
            }
            unsigned int scaled_bf16_42[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_39[_lp*2 + 0], scaled_39[_lp*2+1 + 0]));
                scaled_bf16_42[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_46 = 0; pair_46 < 4; pair_46++) {
                uint32_t pair_sum_44;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_44) : "r"(packed_acc_0[pair_46]), "r"(scaled_bf16_42[pair_46]));
                packed_acc_0[pair_46] = pair_sum_44;
            }
            block_taken_0 = 1;
        }
        if (active_experts == 12) {
            long long expert_elem_44 = (long long)0 * 7168 + (long long)elem_2;
            float expert_vec_44[8];
            {
                const uint4* _vptr_59 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_44 + 0);
                uint4 _vld_59[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_59[_blk].x), "=r"(_vld_59[_blk].y), "=r"(_vld_59[_blk].z), "=r"(_vld_59[_blk].w) : "l"((const void*)(_vptr_59 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_44[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                        (&expert_vec_44[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_44 = expert_scales[token];
            long long expert_elem_45 = (long long)tokens * 7168 + (long long)elem_2;
            float expert_vec_45[8];
            {
                const uint4* _vptr_60 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_45 + 0);
                uint4 _vld_60[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_60[_blk].x), "=r"(_vld_60[_blk].y), "=r"(_vld_60[_blk].z), "=r"(_vld_60[_blk].w) : "l"((const void*)(_vptr_60 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_60 = reinterpret_cast<uint32_t*>(&_vld_60[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_45[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) << 16);
                        (&expert_vec_45[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_45 = expert_scales[tokens + token];
            long long expert_elem_46 = (long long)(2 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_46[8];
            {
                const uint4* _vptr_61 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_46 + 0);
                uint4 _vld_61[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_61[_blk].x), "=r"(_vld_61[_blk].y), "=r"(_vld_61[_blk].z), "=r"(_vld_61[_blk].w) : "l"((const void*)(_vptr_61 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_61 = reinterpret_cast<uint32_t*>(&_vld_61[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_46[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) << 16);
                        (&expert_vec_46[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_46 = expert_scales[2 * tokens + token];
            long long expert_elem_47 = (long long)(3 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_47[8];
            {
                const uint4* _vptr_62 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_47 + 0);
                uint4 _vld_62[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_62[_blk].x), "=r"(_vld_62[_blk].y), "=r"(_vld_62[_blk].z), "=r"(_vld_62[_blk].w) : "l"((const void*)(_vptr_62 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_62 = reinterpret_cast<uint32_t*>(&_vld_62[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_47[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) << 16);
                        (&expert_vec_47[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_47 = expert_scales[3 * tokens + token];
            long long expert_elem_48 = (long long)(4 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_48[8];
            {
                const uint4* _vptr_63 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_48 + 0);
                uint4 _vld_63[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_63[_blk].x), "=r"(_vld_63[_blk].y), "=r"(_vld_63[_blk].z), "=r"(_vld_63[_blk].w) : "l"((const void*)(_vptr_63 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_63 = reinterpret_cast<uint32_t*>(&_vld_63[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_48[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) << 16);
                        (&expert_vec_48[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_48 = expert_scales[4 * tokens + token];
            long long expert_elem_49 = (long long)(5 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_49[8];
            {
                const uint4* _vptr_64 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_49 + 0);
                uint4 _vld_64[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_64[_blk].x), "=r"(_vld_64[_blk].y), "=r"(_vld_64[_blk].z), "=r"(_vld_64[_blk].w) : "l"((const void*)(_vptr_64 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_49[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                        (&expert_vec_49[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_49 = expert_scales[5 * tokens + token];
            long long expert_elem_50 = (long long)(6 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_50[8];
            {
                const uint4* _vptr_65 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_50 + 0);
                uint4 _vld_65[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_65[_blk].x), "=r"(_vld_65[_blk].y), "=r"(_vld_65[_blk].z), "=r"(_vld_65[_blk].w) : "l"((const void*)(_vptr_65 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_50[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                        (&expert_vec_50[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_50 = expert_scales[6 * tokens + token];
            long long expert_elem_51 = (long long)(7 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_51[8];
            {
                const uint4* _vptr_66 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_51 + 0);
                uint4 _vld_66[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_66[_blk].x), "=r"(_vld_66[_blk].y), "=r"(_vld_66[_blk].z), "=r"(_vld_66[_blk].w) : "l"((const void*)(_vptr_66 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_51[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                        (&expert_vec_51[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_51 = expert_scales[7 * tokens + token];
            long long expert_elem_52 = (long long)(8 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_52[8];
            {
                const uint4* _vptr_67 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_52 + 0);
                uint4 _vld_67[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_67[_blk].x), "=r"(_vld_67[_blk].y), "=r"(_vld_67[_blk].z), "=r"(_vld_67[_blk].w) : "l"((const void*)(_vptr_67 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_52[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                        (&expert_vec_52[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_52 = expert_scales[8 * tokens + token];
            long long expert_elem_53 = (long long)(9 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_53[8];
            {
                const uint4* _vptr_68 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_53 + 0);
                uint4 _vld_68[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_68[_blk].x), "=r"(_vld_68[_blk].y), "=r"(_vld_68[_blk].z), "=r"(_vld_68[_blk].w) : "l"((const void*)(_vptr_68 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_68 = reinterpret_cast<uint32_t*>(&_vld_68[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_53[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_68[_pair]) << 16);
                        (&expert_vec_53[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_68[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_53 = expert_scales[9 * tokens + token];
            long long expert_elem_54 = (long long)(10 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_54[8];
            {
                const uint4* _vptr_69 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_54 + 0);
                uint4 _vld_69[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_69[_blk].x), "=r"(_vld_69[_blk].y), "=r"(_vld_69[_blk].z), "=r"(_vld_69[_blk].w) : "l"((const void*)(_vptr_69 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_69 = reinterpret_cast<uint32_t*>(&_vld_69[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_54[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) << 16);
                        (&expert_vec_54[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_54 = expert_scales[10 * tokens + token];
            long long expert_elem_55 = (long long)(11 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_55[8];
            {
                const uint4* _vptr_70 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_55 + 0);
                uint4 _vld_70[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_70[_blk].x), "=r"(_vld_70[_blk].y), "=r"(_vld_70[_blk].z), "=r"(_vld_70[_blk].w) : "l"((const void*)(_vptr_70 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_70 = reinterpret_cast<uint32_t*>(&_vld_70[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_55[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_70[_pair]) << 16);
                        (&expert_vec_55[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_70[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_55 = expert_scales[11 * tokens + token];
            float scaled_40[8];
            #pragma unroll
            for (int j_49 = 0; j_49 < 8; j_49++) {
                scaled_40[j_49] = expert_vec_44[j_49] * expert_scale_44;
            }
            unsigned int scaled_bf16_43[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_40[_lp*2 + 0], scaled_40[_lp*2+1 + 0]));
                scaled_bf16_43[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_47 = 0; pair_47 < 4; pair_47++) {
                uint32_t pair_sum_45;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_45) : "r"(packed_acc_0[pair_47]), "r"(scaled_bf16_43[pair_47]));
                packed_acc_0[pair_47] = pair_sum_45;
            }
            float scaled_40_1[8];
            #pragma unroll
            for (int j_50 = 0; j_50 < 8; j_50++) {
                scaled_40_1[j_50] = expert_vec_45[j_50] * expert_scale_45;
            }
            unsigned int scaled_bf16_44[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_40_1[_lp*2 + 0], scaled_40_1[_lp*2+1 + 0]));
                scaled_bf16_44[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_48 = 0; pair_48 < 4; pair_48++) {
                uint32_t pair_sum_46;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_46) : "r"(packed_acc_0[pair_48]), "r"(scaled_bf16_44[pair_48]));
                packed_acc_0[pair_48] = pair_sum_46;
            }
            float scaled_41[8];
            #pragma unroll
            for (int j_51 = 0; j_51 < 8; j_51++) {
                scaled_41[j_51] = expert_vec_46[j_51] * expert_scale_46;
            }
            unsigned int scaled_bf16_45[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_41[_lp*2 + 0], scaled_41[_lp*2+1 + 0]));
                scaled_bf16_45[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_49 = 0; pair_49 < 4; pair_49++) {
                uint32_t pair_sum_47;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_47) : "r"(packed_acc_0[pair_49]), "r"(scaled_bf16_45[pair_49]));
                packed_acc_0[pair_49] = pair_sum_47;
            }
            float scaled_42[8];
            #pragma unroll
            for (int j_52 = 0; j_52 < 8; j_52++) {
                scaled_42[j_52] = expert_vec_47[j_52] * expert_scale_47;
            }
            unsigned int scaled_bf16_46[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_42[_lp*2 + 0], scaled_42[_lp*2+1 + 0]));
                scaled_bf16_46[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_50 = 0; pair_50 < 4; pair_50++) {
                uint32_t pair_sum_48;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_48) : "r"(packed_acc_0[pair_50]), "r"(scaled_bf16_46[pair_50]));
                packed_acc_0[pair_50] = pair_sum_48;
            }
            float scaled_43[8];
            #pragma unroll
            for (int j_53 = 0; j_53 < 8; j_53++) {
                scaled_43[j_53] = expert_vec_48[j_53] * expert_scale_48;
            }
            unsigned int scaled_bf16_47[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_43[_lp*2 + 0], scaled_43[_lp*2+1 + 0]));
                scaled_bf16_47[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_51 = 0; pair_51 < 4; pair_51++) {
                uint32_t pair_sum_49;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_49) : "r"(packed_acc_0[pair_51]), "r"(scaled_bf16_47[pair_51]));
                packed_acc_0[pair_51] = pair_sum_49;
            }
            float scaled_44[8];
            #pragma unroll
            for (int j_54 = 0; j_54 < 8; j_54++) {
                scaled_44[j_54] = expert_vec_49[j_54] * expert_scale_49;
            }
            unsigned int scaled_bf16_48[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_44[_lp*2 + 0], scaled_44[_lp*2+1 + 0]));
                scaled_bf16_48[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_52 = 0; pair_52 < 4; pair_52++) {
                uint32_t pair_sum_50;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_50) : "r"(packed_acc_0[pair_52]), "r"(scaled_bf16_48[pair_52]));
                packed_acc_0[pair_52] = pair_sum_50;
            }
            float scaled_45[8];
            #pragma unroll
            for (int j_55 = 0; j_55 < 8; j_55++) {
                scaled_45[j_55] = expert_vec_50[j_55] * expert_scale_50;
            }
            unsigned int scaled_bf16_49[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_45[_lp*2 + 0], scaled_45[_lp*2+1 + 0]));
                scaled_bf16_49[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_53 = 0; pair_53 < 4; pair_53++) {
                uint32_t pair_sum_51;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_51) : "r"(packed_acc_0[pair_53]), "r"(scaled_bf16_49[pair_53]));
                packed_acc_0[pair_53] = pair_sum_51;
            }
            float scaled_46[8];
            #pragma unroll
            for (int j_56 = 0; j_56 < 8; j_56++) {
                scaled_46[j_56] = expert_vec_51[j_56] * expert_scale_51;
            }
            unsigned int scaled_bf16_50[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_46[_lp*2 + 0], scaled_46[_lp*2+1 + 0]));
                scaled_bf16_50[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_54 = 0; pair_54 < 4; pair_54++) {
                uint32_t pair_sum_52;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_52) : "r"(packed_acc_0[pair_54]), "r"(scaled_bf16_50[pair_54]));
                packed_acc_0[pair_54] = pair_sum_52;
            }
            float scaled_47[8];
            #pragma unroll
            for (int j_57 = 0; j_57 < 8; j_57++) {
                scaled_47[j_57] = expert_vec_52[j_57] * expert_scale_52;
            }
            unsigned int scaled_bf16_51[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_47[_lp*2 + 0], scaled_47[_lp*2+1 + 0]));
                scaled_bf16_51[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_55 = 0; pair_55 < 4; pair_55++) {
                uint32_t pair_sum_53;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_53) : "r"(packed_acc_0[pair_55]), "r"(scaled_bf16_51[pair_55]));
                packed_acc_0[pair_55] = pair_sum_53;
            }
            float scaled_48[8];
            #pragma unroll
            for (int j_58 = 0; j_58 < 8; j_58++) {
                scaled_48[j_58] = expert_vec_53[j_58] * expert_scale_53;
            }
            unsigned int scaled_bf16_52[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_48[_lp*2 + 0], scaled_48[_lp*2+1 + 0]));
                scaled_bf16_52[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_56 = 0; pair_56 < 4; pair_56++) {
                uint32_t pair_sum_54;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_54) : "r"(packed_acc_0[pair_56]), "r"(scaled_bf16_52[pair_56]));
                packed_acc_0[pair_56] = pair_sum_54;
            }
            float scaled_49[8];
            #pragma unroll
            for (int j_59 = 0; j_59 < 8; j_59++) {
                scaled_49[j_59] = expert_vec_54[j_59] * expert_scale_54;
            }
            unsigned int scaled_bf16_53[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_49[_lp*2 + 0], scaled_49[_lp*2+1 + 0]));
                scaled_bf16_53[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_57 = 0; pair_57 < 4; pair_57++) {
                uint32_t pair_sum_55;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_55) : "r"(packed_acc_0[pair_57]), "r"(scaled_bf16_53[pair_57]));
                packed_acc_0[pair_57] = pair_sum_55;
            }
            float scaled_50[8];
            #pragma unroll
            for (int j_60 = 0; j_60 < 8; j_60++) {
                scaled_50[j_60] = expert_vec_55[j_60] * expert_scale_55;
            }
            unsigned int scaled_bf16_54[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_50[_lp*2 + 0], scaled_50[_lp*2+1 + 0]));
                scaled_bf16_54[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_58 = 0; pair_58 < 4; pair_58++) {
                uint32_t pair_sum_56;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_56) : "r"(packed_acc_0[pair_58]), "r"(scaled_bf16_54[pair_58]));
                packed_acc_0[pair_58] = pair_sum_56;
            }
            block_taken_0 = 1;
        }
        if (active_experts == 16) {
            long long expert_elem_56 = (long long)0 * 7168 + (long long)elem_2;
            float expert_vec_56[8];
            {
                const uint4* _vptr_71 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_56 + 0);
                uint4 _vld_71[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_71[_blk].x), "=r"(_vld_71[_blk].y), "=r"(_vld_71[_blk].z), "=r"(_vld_71[_blk].w) : "l"((const void*)(_vptr_71 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_71 = reinterpret_cast<uint32_t*>(&_vld_71[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_56[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) << 16);
                        (&expert_vec_56[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_56 = expert_scales[token];
            long long expert_elem_57 = (long long)tokens * 7168 + (long long)elem_2;
            float expert_vec_57[8];
            {
                const uint4* _vptr_72 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_57 + 0);
                uint4 _vld_72[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_72[_blk].x), "=r"(_vld_72[_blk].y), "=r"(_vld_72[_blk].z), "=r"(_vld_72[_blk].w) : "l"((const void*)(_vptr_72 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_72 = reinterpret_cast<uint32_t*>(&_vld_72[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_57[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) << 16);
                        (&expert_vec_57[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_57 = expert_scales[tokens + token];
            long long expert_elem_58 = (long long)(2 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_58[8];
            {
                const uint4* _vptr_73 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_58 + 0);
                uint4 _vld_73[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_73[_blk].x), "=r"(_vld_73[_blk].y), "=r"(_vld_73[_blk].z), "=r"(_vld_73[_blk].w) : "l"((const void*)(_vptr_73 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_73 = reinterpret_cast<uint32_t*>(&_vld_73[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_58[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) << 16);
                        (&expert_vec_58[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_58 = expert_scales[2 * tokens + token];
            long long expert_elem_59 = (long long)(3 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_59[8];
            {
                const uint4* _vptr_74 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_59 + 0);
                uint4 _vld_74[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_74[_blk].x), "=r"(_vld_74[_blk].y), "=r"(_vld_74[_blk].z), "=r"(_vld_74[_blk].w) : "l"((const void*)(_vptr_74 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_74 = reinterpret_cast<uint32_t*>(&_vld_74[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_59[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) << 16);
                        (&expert_vec_59[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_59 = expert_scales[3 * tokens + token];
            long long expert_elem_60 = (long long)(4 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_60[8];
            {
                const uint4* _vptr_75 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_60 + 0);
                uint4 _vld_75[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_75[_blk].x), "=r"(_vld_75[_blk].y), "=r"(_vld_75[_blk].z), "=r"(_vld_75[_blk].w) : "l"((const void*)(_vptr_75 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_75 = reinterpret_cast<uint32_t*>(&_vld_75[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_60[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) << 16);
                        (&expert_vec_60[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_60 = expert_scales[4 * tokens + token];
            long long expert_elem_61 = (long long)(5 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_61[8];
            {
                const uint4* _vptr_76 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_61 + 0);
                uint4 _vld_76[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_76[_blk].x), "=r"(_vld_76[_blk].y), "=r"(_vld_76[_blk].z), "=r"(_vld_76[_blk].w) : "l"((const void*)(_vptr_76 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_76 = reinterpret_cast<uint32_t*>(&_vld_76[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_61[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) << 16);
                        (&expert_vec_61[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_61 = expert_scales[5 * tokens + token];
            long long expert_elem_62 = (long long)(6 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_62[8];
            {
                const uint4* _vptr_77 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_62 + 0);
                uint4 _vld_77[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_77[_blk].x), "=r"(_vld_77[_blk].y), "=r"(_vld_77[_blk].z), "=r"(_vld_77[_blk].w) : "l"((const void*)(_vptr_77 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_77 = reinterpret_cast<uint32_t*>(&_vld_77[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_62[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) << 16);
                        (&expert_vec_62[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_62 = expert_scales[6 * tokens + token];
            long long expert_elem_63 = (long long)(7 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_63[8];
            {
                const uint4* _vptr_78 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_63 + 0);
                uint4 _vld_78[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_78[_blk].x), "=r"(_vld_78[_blk].y), "=r"(_vld_78[_blk].z), "=r"(_vld_78[_blk].w) : "l"((const void*)(_vptr_78 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_63[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                        (&expert_vec_63[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_63 = expert_scales[7 * tokens + token];
            long long expert_elem_64 = (long long)(8 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_64[8];
            {
                const uint4* _vptr_79 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_64 + 0);
                uint4 _vld_79[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_79[_blk].x), "=r"(_vld_79[_blk].y), "=r"(_vld_79[_blk].z), "=r"(_vld_79[_blk].w) : "l"((const void*)(_vptr_79 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_64[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                        (&expert_vec_64[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_64 = expert_scales[8 * tokens + token];
            long long expert_elem_65 = (long long)(9 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_65[8];
            {
                const uint4* _vptr_80 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_65 + 0);
                uint4 _vld_80[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_80[_blk].x), "=r"(_vld_80[_blk].y), "=r"(_vld_80[_blk].z), "=r"(_vld_80[_blk].w) : "l"((const void*)(_vptr_80 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_65[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                        (&expert_vec_65[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_65 = expert_scales[9 * tokens + token];
            long long expert_elem_66 = (long long)(10 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_66[8];
            {
                const uint4* _vptr_81 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_66 + 0);
                uint4 _vld_81[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_81[_blk].x), "=r"(_vld_81[_blk].y), "=r"(_vld_81[_blk].z), "=r"(_vld_81[_blk].w) : "l"((const void*)(_vptr_81 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_66[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                        (&expert_vec_66[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_66 = expert_scales[10 * tokens + token];
            long long expert_elem_67 = (long long)(11 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_67[8];
            {
                const uint4* _vptr_82 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_67 + 0);
                uint4 _vld_82[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_82[_blk].x), "=r"(_vld_82[_blk].y), "=r"(_vld_82[_blk].z), "=r"(_vld_82[_blk].w) : "l"((const void*)(_vptr_82 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_82 = reinterpret_cast<uint32_t*>(&_vld_82[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_67[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_82[_pair]) << 16);
                        (&expert_vec_67[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_82[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_67 = expert_scales[11 * tokens + token];
            long long expert_elem_68 = (long long)(12 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_68[8];
            {
                const uint4* _vptr_83 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_68 + 0);
                uint4 _vld_83[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_83[_blk].x), "=r"(_vld_83[_blk].y), "=r"(_vld_83[_blk].z), "=r"(_vld_83[_blk].w) : "l"((const void*)(_vptr_83 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_68[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                        (&expert_vec_68[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_68 = expert_scales[12 * tokens + token];
            long long expert_elem_69 = (long long)(13 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_69[8];
            {
                const uint4* _vptr_84 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_69 + 0);
                uint4 _vld_84[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_84[_blk].x), "=r"(_vld_84[_blk].y), "=r"(_vld_84[_blk].z), "=r"(_vld_84[_blk].w) : "l"((const void*)(_vptr_84 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_84 = reinterpret_cast<uint32_t*>(&_vld_84[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_69[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) << 16);
                        (&expert_vec_69[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_69 = expert_scales[13 * tokens + token];
            long long expert_elem_70 = (long long)(14 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_70[8];
            {
                const uint4* _vptr_85 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_70 + 0);
                uint4 _vld_85[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_85[_blk].x), "=r"(_vld_85[_blk].y), "=r"(_vld_85[_blk].z), "=r"(_vld_85[_blk].w) : "l"((const void*)(_vptr_85 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_85 = reinterpret_cast<uint32_t*>(&_vld_85[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_70[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) << 16);
                        (&expert_vec_70[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_70 = expert_scales[14 * tokens + token];
            long long expert_elem_71 = (long long)(15 * tokens) * 7168 + (long long)elem_2;
            float expert_vec_71[8];
            {
                const uint4* _vptr_86 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_71 + 0);
                uint4 _vld_86[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_86[_blk].x), "=r"(_vld_86[_blk].y), "=r"(_vld_86[_blk].z), "=r"(_vld_86[_blk].w) : "l"((const void*)(_vptr_86 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_86 = reinterpret_cast<uint32_t*>(&_vld_86[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_71[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_86[_pair]) << 16);
                        (&expert_vec_71[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_86[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_71 = expert_scales[15 * tokens + token];
            float scaled_51[8];
            #pragma unroll
            for (int j_61 = 0; j_61 < 8; j_61++) {
                scaled_51[j_61] = expert_vec_56[j_61] * expert_scale_56;
            }
            unsigned int scaled_bf16_55[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_51[_lp*2 + 0], scaled_51[_lp*2+1 + 0]));
                scaled_bf16_55[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_59 = 0; pair_59 < 4; pair_59++) {
                uint32_t pair_sum_57;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_57) : "r"(packed_acc_0[pair_59]), "r"(scaled_bf16_55[pair_59]));
                packed_acc_0[pair_59] = pair_sum_57;
            }
            float scaled_51_1[8];
            #pragma unroll
            for (int j_62 = 0; j_62 < 8; j_62++) {
                scaled_51_1[j_62] = expert_vec_57[j_62] * expert_scale_57;
            }
            unsigned int scaled_bf16_56[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_51_1[_lp*2 + 0], scaled_51_1[_lp*2+1 + 0]));
                scaled_bf16_56[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_60 = 0; pair_60 < 4; pair_60++) {
                uint32_t pair_sum_58;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_58) : "r"(packed_acc_0[pair_60]), "r"(scaled_bf16_56[pair_60]));
                packed_acc_0[pair_60] = pair_sum_58;
            }
            float scaled_52[8];
            #pragma unroll
            for (int j_63 = 0; j_63 < 8; j_63++) {
                scaled_52[j_63] = expert_vec_58[j_63] * expert_scale_58;
            }
            unsigned int scaled_bf16_57[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_52[_lp*2 + 0], scaled_52[_lp*2+1 + 0]));
                scaled_bf16_57[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_61 = 0; pair_61 < 4; pair_61++) {
                uint32_t pair_sum_59;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_59) : "r"(packed_acc_0[pair_61]), "r"(scaled_bf16_57[pair_61]));
                packed_acc_0[pair_61] = pair_sum_59;
            }
            float scaled_53[8];
            #pragma unroll
            for (int j_64 = 0; j_64 < 8; j_64++) {
                scaled_53[j_64] = expert_vec_59[j_64] * expert_scale_59;
            }
            unsigned int scaled_bf16_58[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_53[_lp*2 + 0], scaled_53[_lp*2+1 + 0]));
                scaled_bf16_58[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_62 = 0; pair_62 < 4; pair_62++) {
                uint32_t pair_sum_60;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_60) : "r"(packed_acc_0[pair_62]), "r"(scaled_bf16_58[pair_62]));
                packed_acc_0[pair_62] = pair_sum_60;
            }
            float scaled_54[8];
            #pragma unroll
            for (int j_65 = 0; j_65 < 8; j_65++) {
                scaled_54[j_65] = expert_vec_60[j_65] * expert_scale_60;
            }
            unsigned int scaled_bf16_59[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_54[_lp*2 + 0], scaled_54[_lp*2+1 + 0]));
                scaled_bf16_59[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_63 = 0; pair_63 < 4; pair_63++) {
                uint32_t pair_sum_61;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_61) : "r"(packed_acc_0[pair_63]), "r"(scaled_bf16_59[pair_63]));
                packed_acc_0[pair_63] = pair_sum_61;
            }
            float scaled_55[8];
            #pragma unroll
            for (int j_66 = 0; j_66 < 8; j_66++) {
                scaled_55[j_66] = expert_vec_61[j_66] * expert_scale_61;
            }
            unsigned int scaled_bf16_60[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_55[_lp*2 + 0], scaled_55[_lp*2+1 + 0]));
                scaled_bf16_60[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_64 = 0; pair_64 < 4; pair_64++) {
                uint32_t pair_sum_62;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_62) : "r"(packed_acc_0[pair_64]), "r"(scaled_bf16_60[pair_64]));
                packed_acc_0[pair_64] = pair_sum_62;
            }
            float scaled_56[8];
            #pragma unroll
            for (int j_67 = 0; j_67 < 8; j_67++) {
                scaled_56[j_67] = expert_vec_62[j_67] * expert_scale_62;
            }
            unsigned int scaled_bf16_61[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_56[_lp*2 + 0], scaled_56[_lp*2+1 + 0]));
                scaled_bf16_61[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_65 = 0; pair_65 < 4; pair_65++) {
                uint32_t pair_sum_63;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_63) : "r"(packed_acc_0[pair_65]), "r"(scaled_bf16_61[pair_65]));
                packed_acc_0[pair_65] = pair_sum_63;
            }
            float scaled_57[8];
            #pragma unroll
            for (int j_68 = 0; j_68 < 8; j_68++) {
                scaled_57[j_68] = expert_vec_63[j_68] * expert_scale_63;
            }
            unsigned int scaled_bf16_62[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_57[_lp*2 + 0], scaled_57[_lp*2+1 + 0]));
                scaled_bf16_62[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_66 = 0; pair_66 < 4; pair_66++) {
                uint32_t pair_sum_64;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_64) : "r"(packed_acc_0[pair_66]), "r"(scaled_bf16_62[pair_66]));
                packed_acc_0[pair_66] = pair_sum_64;
            }
            float scaled_58[8];
            #pragma unroll
            for (int j_69 = 0; j_69 < 8; j_69++) {
                scaled_58[j_69] = expert_vec_64[j_69] * expert_scale_64;
            }
            unsigned int scaled_bf16_63[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_58[_lp*2 + 0], scaled_58[_lp*2+1 + 0]));
                scaled_bf16_63[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_67 = 0; pair_67 < 4; pair_67++) {
                uint32_t pair_sum_65;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_65) : "r"(packed_acc_0[pair_67]), "r"(scaled_bf16_63[pair_67]));
                packed_acc_0[pair_67] = pair_sum_65;
            }
            float scaled_59[8];
            #pragma unroll
            for (int j_70 = 0; j_70 < 8; j_70++) {
                scaled_59[j_70] = expert_vec_65[j_70] * expert_scale_65;
            }
            unsigned int scaled_bf16_64[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_59[_lp*2 + 0], scaled_59[_lp*2+1 + 0]));
                scaled_bf16_64[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_68 = 0; pair_68 < 4; pair_68++) {
                uint32_t pair_sum_66;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_66) : "r"(packed_acc_0[pair_68]), "r"(scaled_bf16_64[pair_68]));
                packed_acc_0[pair_68] = pair_sum_66;
            }
            float scaled_60[8];
            #pragma unroll
            for (int j_71 = 0; j_71 < 8; j_71++) {
                scaled_60[j_71] = expert_vec_66[j_71] * expert_scale_66;
            }
            unsigned int scaled_bf16_65[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_60[_lp*2 + 0], scaled_60[_lp*2+1 + 0]));
                scaled_bf16_65[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_69 = 0; pair_69 < 4; pair_69++) {
                uint32_t pair_sum_67;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_67) : "r"(packed_acc_0[pair_69]), "r"(scaled_bf16_65[pair_69]));
                packed_acc_0[pair_69] = pair_sum_67;
            }
            float scaled_61[8];
            #pragma unroll
            for (int j_72 = 0; j_72 < 8; j_72++) {
                scaled_61[j_72] = expert_vec_67[j_72] * expert_scale_67;
            }
            unsigned int scaled_bf16_66[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_61[_lp*2 + 0], scaled_61[_lp*2+1 + 0]));
                scaled_bf16_66[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_70 = 0; pair_70 < 4; pair_70++) {
                uint32_t pair_sum_68;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_68) : "r"(packed_acc_0[pair_70]), "r"(scaled_bf16_66[pair_70]));
                packed_acc_0[pair_70] = pair_sum_68;
            }
            float scaled_62[8];
            #pragma unroll
            for (int j_73 = 0; j_73 < 8; j_73++) {
                scaled_62[j_73] = expert_vec_68[j_73] * expert_scale_68;
            }
            unsigned int scaled_bf16_67[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_62[_lp*2 + 0], scaled_62[_lp*2+1 + 0]));
                scaled_bf16_67[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_71 = 0; pair_71 < 4; pair_71++) {
                uint32_t pair_sum_69;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_69) : "r"(packed_acc_0[pair_71]), "r"(scaled_bf16_67[pair_71]));
                packed_acc_0[pair_71] = pair_sum_69;
            }
            float scaled_63[8];
            #pragma unroll
            for (int j_74 = 0; j_74 < 8; j_74++) {
                scaled_63[j_74] = expert_vec_69[j_74] * expert_scale_69;
            }
            unsigned int scaled_bf16_68[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_63[_lp*2 + 0], scaled_63[_lp*2+1 + 0]));
                scaled_bf16_68[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_72 = 0; pair_72 < 4; pair_72++) {
                uint32_t pair_sum_70;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_70) : "r"(packed_acc_0[pair_72]), "r"(scaled_bf16_68[pair_72]));
                packed_acc_0[pair_72] = pair_sum_70;
            }
            float scaled_64[8];
            #pragma unroll
            for (int j_75 = 0; j_75 < 8; j_75++) {
                scaled_64[j_75] = expert_vec_70[j_75] * expert_scale_70;
            }
            unsigned int scaled_bf16_69[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_64[_lp*2 + 0], scaled_64[_lp*2+1 + 0]));
                scaled_bf16_69[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_73 = 0; pair_73 < 4; pair_73++) {
                uint32_t pair_sum_71;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_71) : "r"(packed_acc_0[pair_73]), "r"(scaled_bf16_69[pair_73]));
                packed_acc_0[pair_73] = pair_sum_71;
            }
            float scaled_65[8];
            #pragma unroll
            for (int j_76 = 0; j_76 < 8; j_76++) {
                scaled_65[j_76] = expert_vec_71[j_76] * expert_scale_71;
            }
            unsigned int scaled_bf16_70[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_65[_lp*2 + 0], scaled_65[_lp*2+1 + 0]));
                scaled_bf16_70[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_74 = 0; pair_74 < 4; pair_74++) {
                uint32_t pair_sum_72;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_72) : "r"(packed_acc_0[pair_74]), "r"(scaled_bf16_70[pair_74]));
                packed_acc_0[pair_74] = pair_sum_72;
            }
            block_taken_0 = 1;
        }
        int expert_loop_end_0 = ((block_taken_0 == 1) ? 0 : active_experts);
        #pragma unroll 4
        for (int expert_1 = 0; expert_1 < expert_loop_end_0; expert_1++) {
            long long expert_elem_72 = (long long)expert_1 * (long long)tokens * 7168 + (long long)elem_2;
            float expert_vec_72[8];
            {
                const uint4* _vptr_87 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_72 + 0);
                uint4 _vld_87[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_87[_blk].x), "=r"(_vld_87[_blk].y), "=r"(_vld_87[_blk].z), "=r"(_vld_87[_blk].w) : "l"((const void*)(_vptr_87 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_87 = reinterpret_cast<uint32_t*>(&_vld_87[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&expert_vec_72[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) << 16);
                        (&expert_vec_72[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_87[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_72 = expert_scales[expert_1 * tokens + token];
            float scaled_66[8];
            #pragma unroll
            for (int j_77 = 0; j_77 < 8; j_77++) {
                scaled_66[j_77] = expert_vec_72[j_77] * expert_scale_72;
            }
            unsigned int scaled_bf16_71[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_66[_lp*2 + 0], scaled_66[_lp*2+1 + 0]));
                scaled_bf16_71[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_75 = 0; pair_75 < 4; pair_75++) {
                uint32_t pair_sum_73;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(pair_sum_73) : "r"(packed_acc_0[pair_75]), "r"(scaled_bf16_71[pair_75]));
                packed_acc_0[pair_75] = pair_sum_73;
            }
        }
        float token_vec[8];
        {
            const uint4* _vptr_88 = reinterpret_cast<const uint4*>(token_input + elem_2 + 0);
            uint4 _vld_88[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_88[_blk].x), "=r"(_vld_88[_blk].y), "=r"(_vld_88[_blk].z), "=r"(_vld_88[_blk].w) : "l"((const void*)(_vptr_88 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_88 = reinterpret_cast<uint32_t*>(&_vld_88[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&token_vec[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) << 16);
                    (&token_vec[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&acc_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_acc_0[_pair]) << 16);
            (&acc_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_acc_0[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_78 = 0; j_78 < 8; j_78++) {
            acc_0[j_78] = acc_0[j_78] + token_vec[j_78];
        }
        unsigned int scaled_bf16_72[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
            scaled_bf16_72[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&acc_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_72[_pair]) << 16);
            (&acc_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_72[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_79 = 0; j_79 < 8; j_79++) {
            acc_0[j_79] = ((acc_0[j_79] == 0.0f) ? 0.0f : acc_0[j_79]);
        }
        int owner_1 = cluster_thread / 112;
        long long slot_1 = (long long)data_epoch * comm_stride_elems + (long long)rank * (long long)rs_items * 8 + (long long)(token * 112 + (cluster_thread - owner_1 * 112)) * 8;
        unsigned int scaled_bf16_73[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
            scaled_bf16_73[_lp] = *(uint32_t*)&_bf2;
        }
        if (owner_1 == 0) {
            {
                auto _gst_89 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[16]) + slot_1;
                __builtin_assume(__isGlobal(_gst_89));
                reinterpret_cast<int4*>(_gst_89)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 1) {
            {
                auto _gst_90 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[17]) + slot_1;
                __builtin_assume(__isGlobal(_gst_90));
                reinterpret_cast<int4*>(_gst_90)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 2) {
            {
                auto _gst_91 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[18]) + slot_1;
                __builtin_assume(__isGlobal(_gst_91));
                reinterpret_cast<int4*>(_gst_91)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 3) {
            {
                auto _gst_92 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[19]) + slot_1;
                __builtin_assume(__isGlobal(_gst_92));
                reinterpret_cast<int4*>(_gst_92)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 4) {
            {
                auto _gst_93 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[20]) + slot_1;
                __builtin_assume(__isGlobal(_gst_93));
                reinterpret_cast<int4*>(_gst_93)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 5) {
            {
                auto _gst_94 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[21]) + slot_1;
                __builtin_assume(__isGlobal(_gst_94));
                reinterpret_cast<int4*>(_gst_94)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 6) {
            {
                auto _gst_95 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[22]) + slot_1;
                __builtin_assume(__isGlobal(_gst_95));
                reinterpret_cast<int4*>(_gst_95)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
        if (owner_1 == 7) {
            {
                auto _gst_96 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[23]) + slot_1;
                __builtin_assume(__isGlobal(_gst_96));
                reinterpret_cast<int4*>(_gst_96)[0] = reinterpret_cast<int4*>(scaled_bf16_73)[0];
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        atomicAdd(&completion[0], 1);
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int lane_access = cluster_thread;
    int access = cluster_id * 896 + (unsigned int)lane_access;
    unsigned int rms_trip = 0;
    unsigned int rms_exchange_rms_trip = 0;
    int rs_src = tid / 28;
    int rs_item = cta_rank * 28 + tid % 28;
    int rs_slot_items = tokens * 112;
    __nv_bfloat16* rs_source_base = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[16 + rs_src]);
    if ((int)cluster_id < tokens) {
        if ((int)cluster_id + (int)num_clusters >= tokens) {
            unsigned int first_rs_words[4];
            uint32_t sysv_ld_1[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_1[0]), "=r"(sysv_ld_1[1]), "=r"(sysv_ld_1[2]), "=r"(sysv_ld_1[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(rs_src * rs_slot_items + (int)cluster_id * 112 + rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word_4 = 0; word_4 < 4; word_4++) {
                first_rs_words[word_4] = sysv_ld_1[word_4];
            }
            int pending_1 = 0;
            #pragma unroll
            for (int word_5 = 0; word_5 < 4; word_5++) {
                pending_1 = (((first_rs_words[word_5] & 65535) == 32768) ? 1 : pending_1);
                pending_1 = ((first_rs_words[word_5] >> 16 == 32768) ? 1 : pending_1);
            }
            if (pending_1 == 1) {
                uint32_t sysv_poll_group_1[4];
                uint32_t _sysv_pending_97 = 1u;
                do {
                    if (_sysv_pending_97 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_1[0]), "=r"(sysv_poll_group_1[1]), "=r"(sysv_poll_group_1[2]), "=r"(sysv_poll_group_1[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(rs_src * rs_slot_items + (int)cluster_id * 112 + rs_item) * 8)) : "memory");
                    }
                    if (_sysv_pending_97 & 1u) {
                        if (!((((sysv_poll_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_97 &= ~1u;
                    }
                } while (_sysv_pending_97 != 0u);
                #pragma unroll
                for (int word_6 = 0; word_6 < 4; word_6++) {
                    first_rs_words[word_6] = sysv_poll_group_1[word_6];
                }
            }
            unsigned int ag_words_1[4];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rs_stage_words_addr + (unsigned int)(tid * 16)), "r"(first_rs_words[0]), "r"(first_rs_words[1]), "r"(first_rs_words[2]), "r"(first_rs_words[3]) : "memory");
            __syncthreads();
            if (tid < 112) {
                float lanes_1[2];
                unsigned int word_7[1];
                word_7[0] = rs_stage_words[tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                float received_lanes_1[2];
                word_7[0] = rs_stage_words[112 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_7 = 0; j_1_7 < 2; j_1_7++) {
                    lanes_1[j_1_7] = lanes_1[j_1_7] + received_lanes_1[j_1_7];
                }
                unsigned int lanes_bf16_7_7[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_7[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_7[_pair]) & 0xffff0000u);
                }
                word_7[0] = rs_stage_words[224 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_8 = 0; j_1_8 < 2; j_1_8++) {
                    lanes_1[j_1_8] = lanes_1[j_1_8] + received_lanes_1[j_1_8];
                }
                unsigned int lanes_bf16_7_8[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_8[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_8[_pair]) & 0xffff0000u);
                }
                word_7[0] = rs_stage_words[336 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_9 = 0; j_1_9 < 2; j_1_9++) {
                    lanes_1[j_1_9] = lanes_1[j_1_9] + received_lanes_1[j_1_9];
                }
                unsigned int lanes_bf16_7_9[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_9[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_9[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_9[_pair]) & 0xffff0000u);
                }
                word_7[0] = rs_stage_words[448 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_10 = 0; j_1_10 < 2; j_1_10++) {
                    lanes_1[j_1_10] = lanes_1[j_1_10] + received_lanes_1[j_1_10];
                }
                unsigned int lanes_bf16_7_10[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_10[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_10[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_10[_pair]) & 0xffff0000u);
                }
                word_7[0] = rs_stage_words[560 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_11 = 0; j_1_11 < 2; j_1_11++) {
                    lanes_1[j_1_11] = lanes_1[j_1_11] + received_lanes_1[j_1_11];
                }
                unsigned int lanes_bf16_7_11[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_11[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_11[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_11[_pair]) & 0xffff0000u);
                }
                word_7[0] = rs_stage_words[672 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_12 = 0; j_1_12 < 2; j_1_12++) {
                    lanes_1[j_1_12] = lanes_1[j_1_12] + received_lanes_1[j_1_12];
                }
                unsigned int lanes_bf16_7_12[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_12[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_12[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_12[_pair]) & 0xffff0000u);
                }
                word_7[0] = rs_stage_words[784 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_13 = 0; j_1_13 < 2; j_1_13++) {
                    lanes_1[j_1_13] = lanes_1[j_1_13] + received_lanes_1[j_1_13];
                }
                unsigned int lanes_bf16_7_13[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_7_13[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_13[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_13[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_74[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    scaled_bf16_74[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_1 = scaled_bf16_74[0] & 65535;
                unsigned int high_1 = scaled_bf16_74[0] >> 16;
                low_1 = ((low_1 == 65535) ? 65534 : low_1);
                high_1 = ((high_1 == 65535) ? 65534 : high_1);
                low_1 = ((low_1 == 32768) ? 65535 : low_1);
                high_1 = ((high_1 == 32768) ? 65535 : high_1);
                rs_sum_words[tid] = low_1 | high_1 << 16;
            }
            __syncthreads();
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&ag_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words_1[(0) + 3]))
                : "r"(rs_sum_words_addr + (unsigned int)(tid % 28 * 16)) : "memory");
            {
                auto _gst_98 = rs_source_base + ((long long)data_epoch * comm_stride_elems + (long long)tokens * 7168 + (long long)((int)cluster_id * 896 + rank * 112 + rs_item) * 8);
                __builtin_assume(__isGlobal(_gst_98));
                reinterpret_cast<int4*>(_gst_98)[0] = reinterpret_cast<int4*>(ag_words_1)[0];
            }
        }
    }
    unsigned int _phase_rms_exchange = 0;
    #pragma unroll 1
    for (int token_1 = cluster_id; token_1 < tokens; token_1 += (int)num_clusters) {
        long long data_base = (long long)data_epoch * comm_stride_elems;
        int rs_token = token_1 + (int)num_clusters;
        unsigned int rs_words[4];
        if (rs_token < tokens) {
            uint32_t sysv_ld_2[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_2[0]), "=r"(sysv_ld_2[1]), "=r"(sysv_ld_2[2]), "=r"(sysv_ld_2[3]) : "l"(workspace_local + (data_base + (long long)(rs_src * rs_slot_items + rs_token * 112 + rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word_8 = 0; word_8 < 4; word_8++) {
                rs_words[word_8] = sysv_ld_2[word_8];
            }
        }
        int prefetch_elem = access * 8;
        float _vec_load_79[8];
        {
            const uint4* _vptr_99 = reinterpret_cast<const uint4*>(residual + prefetch_elem + 0);
            uint4 _vld_99[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_99[_blk].x), "=r"(_vld_99[_blk].y), "=r"(_vld_99[_blk].z), "=r"(_vld_99[_blk].w) : "l"((const void*)(_vptr_99 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_99 = reinterpret_cast<uint32_t*>(&_vld_99[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_79[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) << 16);
                    (&_vec_load_79[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_80[8];
        {
            const uint4* _vptr_100 = reinterpret_cast<const uint4*>(gamma + (lane_access * 8) + 0);
            uint4 _vld_100[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_100[_blk] = _vptr_100[_blk];
                uint32_t* _vpairs_100 = reinterpret_cast<uint32_t*>(&_vld_100[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_80[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_100[_pair]) << 16);
                    (&_vec_load_80[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_100[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int ag_probe[4];
        uint32_t sysv_ld_3[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_3[0]), "=r"(sysv_ld_3[1]), "=r"(sysv_ld_3[2]), "=r"(sysv_ld_3[3]) : "l"(workspace_local + (data_base + (long long)tokens * 7168 + (long long)access * 8)) : "memory");
        #pragma unroll
        for (int word_9 = 0; word_9 < 4; word_9++) {
            ag_probe[word_9] = sysv_ld_3[word_9];
        }
        if (rs_token < tokens) {
            int pending_2 = 0;
            #pragma unroll
            for (int word_10 = 0; word_10 < 4; word_10++) {
                pending_2 = (((rs_words[word_10] & 65535) == 32768) ? 1 : pending_2);
                pending_2 = ((rs_words[word_10] >> 16 == 32768) ? 1 : pending_2);
            }
            if (pending_2 == 1) {
                uint32_t sysv_poll_group_2[4];
                uint32_t _sysv_pending_101 = 1u;
                do {
                    if (_sysv_pending_101 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_2[0]), "=r"(sysv_poll_group_2[1]), "=r"(sysv_poll_group_2[2]), "=r"(sysv_poll_group_2[3]) : "l"(workspace_local + (data_base + (long long)(rs_src * rs_slot_items + rs_token * 112 + rs_item) * 8)) : "memory");
                    }
                    if (_sysv_pending_101 & 1u) {
                        if (!((((sysv_poll_group_2[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_101 &= ~1u;
                    }
                } while (_sysv_pending_101 != 0u);
                #pragma unroll
                for (int word_11 = 0; word_11 < 4; word_11++) {
                    rs_words[word_11] = sysv_poll_group_2[word_11];
                }
            }
            unsigned int ag_words_2[4];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(rs_stage_words_addr + (unsigned int)(tid * 16)), "r"(rs_words[0]), "r"(rs_words[1]), "r"(rs_words[2]), "r"(rs_words[3]) : "memory");
            __syncthreads();
            if (tid < 112) {
                float lanes_2[2];
                unsigned int word_12[1];
                word_12[0] = rs_stage_words[tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                float received_lanes_2[2];
                word_12[0] = rs_stage_words[112 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_14 = 0; j_1_14 < 2; j_1_14++) {
                    lanes_2[j_1_14] = lanes_2[j_1_14] + received_lanes_2[j_1_14];
                }
                unsigned int lanes_bf16_7_14[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_14[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_14[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_14[_pair]) & 0xffff0000u);
                }
                word_12[0] = rs_stage_words[224 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_15 = 0; j_1_15 < 2; j_1_15++) {
                    lanes_2[j_1_15] = lanes_2[j_1_15] + received_lanes_2[j_1_15];
                }
                unsigned int lanes_bf16_7_15[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_15[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_15[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_15[_pair]) & 0xffff0000u);
                }
                word_12[0] = rs_stage_words[336 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_16 = 0; j_1_16 < 2; j_1_16++) {
                    lanes_2[j_1_16] = lanes_2[j_1_16] + received_lanes_2[j_1_16];
                }
                unsigned int lanes_bf16_7_16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_16[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_16[_pair]) & 0xffff0000u);
                }
                word_12[0] = rs_stage_words[448 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_17 = 0; j_1_17 < 2; j_1_17++) {
                    lanes_2[j_1_17] = lanes_2[j_1_17] + received_lanes_2[j_1_17];
                }
                unsigned int lanes_bf16_7_17[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_17[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_17[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_17[_pair]) & 0xffff0000u);
                }
                word_12[0] = rs_stage_words[560 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_18 = 0; j_1_18 < 2; j_1_18++) {
                    lanes_2[j_1_18] = lanes_2[j_1_18] + received_lanes_2[j_1_18];
                }
                unsigned int lanes_bf16_7_18[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_18[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_18[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_18[_pair]) & 0xffff0000u);
                }
                word_12[0] = rs_stage_words[672 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_19 = 0; j_1_19 < 2; j_1_19++) {
                    lanes_2[j_1_19] = lanes_2[j_1_19] + received_lanes_2[j_1_19];
                }
                unsigned int lanes_bf16_7_19[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_19[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_19[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_19[_pair]) & 0xffff0000u);
                }
                word_12[0] = rs_stage_words[784 + tid];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_1_20 = 0; j_1_20 < 2; j_1_20++) {
                    lanes_2[j_1_20] = lanes_2[j_1_20] + received_lanes_2[j_1_20];
                }
                unsigned int lanes_bf16_7_20[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_7_20[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_20[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7_20[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_75[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    scaled_bf16_75[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_2 = scaled_bf16_75[0] & 65535;
                unsigned int high_2 = scaled_bf16_75[0] >> 16;
                low_2 = ((low_2 == 65535) ? 65534 : low_2);
                high_2 = ((high_2 == 65535) ? 65534 : high_2);
                low_2 = ((low_2 == 32768) ? 65535 : low_2);
                high_2 = ((high_2 == 32768) ? 65535 : high_2);
                rs_sum_words[tid] = low_2 | high_2 << 16;
            }
            __syncthreads();
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&ag_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ag_words_2[(0) + 3]))
                : "r"(rs_sum_words_addr + (unsigned int)(tid % 28 * 16)) : "memory");
            {
                auto _gst_102 = rs_source_base + (data_base + (long long)tokens * 7168 + (long long)(rs_token * 896 + rank * 112 + rs_item) * 8);
                __builtin_assume(__isGlobal(_gst_102));
                reinterpret_cast<int4*>(_gst_102)[0] = reinterpret_cast<int4*>(ag_words_2)[0];
            }
        }
        int pending_3 = 0;
        #pragma unroll
        for (int word_13 = 0; word_13 < 4; word_13++) {
            pending_3 = (((ag_probe[word_13] & 65535) == 32768) ? 1 : pending_3);
            pending_3 = ((ag_probe[word_13] >> 16 == 32768) ? 1 : pending_3);
        }
        if (pending_3 == 1) {
            uint32_t sysv_poll_group_3[4];
            uint32_t _sysv_pending_103 = 1u;
            do {
                if (_sysv_pending_103 & 1u) {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_3[0]), "=r"(sysv_poll_group_3[1]), "=r"(sysv_poll_group_3[2]), "=r"(sysv_poll_group_3[3]) : "l"(workspace_local + (data_base + (long long)tokens * 7168 + (long long)access * 8)) : "memory");
                }
                if (_sysv_pending_103 & 1u) {
                    if (!((((sysv_poll_group_3[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_103 &= ~1u;
                }
            } while (_sysv_pending_103 != 0u);
            #pragma unroll
            for (int word_14 = 0; word_14 < 4; word_14++) {
                ag_probe[word_14] = sysv_poll_group_3[word_14];
            }
        }
        unsigned int sum_words[4];
        #pragma unroll
        for (int word_15 = 0; word_15 < 4; word_15++) {
            unsigned int low_3 = ag_probe[word_15] & 65535;
            unsigned int high_3 = ag_probe[word_15] >> 16;
            low_3 = ((low_3 == 65535) ? 32768 : low_3);
            high_3 = ((high_3 == 65535) ? 32768 : high_3);
            sum_words[word_15] = low_3 | high_3 << 16;
        }
        float sum_value[8];
        float unpack_bf16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sum_words[_pair]) << 16);
            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sum_words[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_80 = 0; j_80 < 8; j_80++) {
            sum_value[j_80] = unpack_bf16[j_80];
        }
        int access_in_token = lane_access;
        int elem_0 = access * 8;
        unsigned int _packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 2], sum_value[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 4], sum_value[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 6], sum_value[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(moe_allreduce_out + elem_0), "r"(_packed_0[(0) + 0]), "r"(_packed_0[(0) + 1]), "r"(_packed_0[(0) + 2]), "r"(_packed_0[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
        #pragma unroll
        for (int j_81 = 0; j_81 < 8; j_81++) {
            _vec_load_79[j_81] = _vec_load_79[j_81] + sum_value[j_81];
        }
        unsigned int scaled_bf16_76[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_79[_lp*2 + 0], _vec_load_79[_lp*2+1 + 0]));
            scaled_bf16_76[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_79[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_76[_pair]) << 16);
            (&_vec_load_79[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_76[_pair]) & 0xffff0000u);
        }
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_79[_lp*2 + 0], _vec_load_79[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_79[_lp*2 + 2], _vec_load_79[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_79[_lp*2 + 4], _vec_load_79[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_79[_lp*2 + 6], _vec_load_79[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(residual_out + elem_0), "r"(_packed_1[(0) + 0]), "r"(_packed_1[(0) + 1]), "r"(_packed_1[(0) + 2]), "r"(_packed_1[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
        float square_sum = 0.0f;
        #pragma unroll
        for (int j_82 = 0; j_82 < 8; j_82++) {
            square_sum = __fmaf_rn(_vec_load_79[j_82], _vec_load_79[j_82], square_sum);
        }
        float _warp_reduce_104 = square_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_104 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_104, offset);
        square_sum = _warp_reduce_104;
        int rms_parity = (int)(rms_trip & 1);
        if (lane == 0) {
            reduce_smem[rms_parity * 7 + warp] = square_sum;
        }
        __syncthreads();
        float block_sum = 0.0f;
        if (tid < 7) {
            block_sum = reduce_smem[rms_parity * 7 + tid];
        }
        float _warp_reduce_105 = block_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_105 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_105, offset);
        block_sum = _warp_reduce_105;
        int slot_base = rms_parity * 4;
        if (tid < 4) {
            uint32_t _cluster_mapa_106;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_106) : "r"(rms_partials_addr), "r"(tid));
            uint32_t _mbar_peer_107;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_107) : "r"(rms_exchange_addr + (rms_exchange_rms_trip) * 8), "r"(tid));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_106 + (unsigned int)((slot_base + cta_rank) * 4)), "f"(block_sum), "r"(_mbar_peer_107) : "memory");
        }
        if (tid == 0) {
            mbarrier_arrive_expect_tx(rms_exchange_addr + (rms_exchange_rms_trip) * 8, 16);
        }
        unsigned int clr_words[4];
        #pragma unroll
        for (int clr_word = 0; clr_word < 4; clr_word++) {
            clr_words[clr_word] = 2147516416;
        }
        long long clr_base = (long long)clear_epoch * comm_stride_elems;
        int clr_high = access + tokens * 896;
        if (access < clear_size / 8) {
            asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(workspace_local + (clr_base + (long long)access * 8)), "r"(clr_words[(0) + 0]), "r"(clr_words[(0) + 1]), "r"(clr_words[(0) + 2]), "r"(clr_words[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
        }
        if (clr_high < clear_size / 8) {
            asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(workspace_local + (clr_base + (long long)clr_high * 8)), "r"(clr_words[(0) + 0]), "r"(clr_words[(0) + 1]), "r"(clr_words[(0) + 2]), "r"(clr_words[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
        }
        asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(rms_exchange_addr + (rms_exchange_rms_trip) * 8), "r"(uint32_t(_phase_rms_exchange)), "r"(uint32_t(10000000)) : "memory");
        asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
        float cluster_sum = 0.0f;
        cluster_sum = cluster_sum + rms_partials[slot_base];
        cluster_sum = cluster_sum + rms_partials[slot_base + 1];
        cluster_sum = cluster_sum + rms_partials[slot_base + 2];
        cluster_sum = cluster_sum + rms_partials[slot_base + 3];
        float rstd = rsqrtf(cluster_sum / 7168.0f + epsilon);
        rms_trip += 1;
        rms_exchange_rms_trip += 1;
        if (rms_exchange_rms_trip == 2) { rms_exchange_rms_trip = 0; _phase_rms_exchange ^= 1; }
        float norm_value[8];
        #pragma unroll
        for (int j_83 = 0; j_83 < 8; j_83++) {
            norm_value[j_83] = _vec_load_79[j_83] * rstd * (_vec_load_80[j_83] + weight_bias);
        }
        unsigned int scaled_bf16_77[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
            scaled_bf16_77[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(norm_out + elem_0), "r"(scaled_bf16_77[(0) + 0]), "r"(scaled_bf16_77[(0) + 1]), "r"(scaled_bf16_77[(0) + 2]), "r"(scaled_bf16_77[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
        access += access_stride;
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    unsigned int clear_words[4];
    #pragma unroll
    for (int word_16 = 0; word_16 < 4; word_16++) {
        clear_words[word_16] = 2147516416;
    }
    long long clear_base = (long long)clear_epoch * comm_stride_elems;
    #pragma unroll 4
    for (int access_1 = access + tokens * 896; access_1 < clear_size / 8; access_1 += access_stride) {
        asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(workspace_local + (clear_base + (long long)access_1 * 8)), "r"(clear_words[(0) + 0]), "r"(clear_words[(0) + 1]), "r"(clear_words[(0) + 2]), "r"(clear_words[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _sysv_wait_p_108 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_sysv_wait_p_108 != static_cast<int>(num_bids)) {}
            }
            flag_addr[0] = (flag + 1) % 3;
            clear_addr[0] = tokens * 7168 * 2;
            completion[0] = 0;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

} // extern "C"
