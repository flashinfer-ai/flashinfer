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
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 132
#define SMEM_REDUCE_SMEM_STRIDE 132
#define SMEM_RMS_SCALAR_OFF 132
#define SMEM_RMS_SCALAR_STAGE_BYTES 4
#define SMEM_RMS_SCALAR_STRIDE 4
#define SMEM_RMS_PARTIALS_OFF 136
#define SMEM_RMS_PARTIALS_STAGE_BYTES 32
#define SMEM_RMS_PARTIALS_STRIDE 32
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
kernel_cake_trtllm_moe_allreduce_union_ws4_bf16_two_shot_b5(__nv_bfloat16* __restrict__ active_expert_tokens, float* __restrict__ expert_scales, __nv_bfloat16* __restrict__ token_input, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ gamma, __nv_bfloat16* __restrict__ moe_allreduce_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int active_experts, float epsilon, float weight_bias)
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
        long long expert_elem0 = (long long)0 * 7168 + (long long)elem;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem0 + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale0 = expert_scales[early_token];
        long long expert_elem1 = (long long)tokens * 7168 + (long long)elem;
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem1 + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_1[_blk].x), "=r"(_vld_1[_blk].y), "=r"(_vld_1[_blk].z), "=r"(_vld_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale1 = expert_scales[tokens + early_token];
        long long expert_elem2 = (long long)(2 * tokens) * 7168 + (long long)elem;
        float _vec_load_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem2 + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_2[_blk].x), "=r"(_vld_2[_blk].y), "=r"(_vld_2[_blk].z), "=r"(_vld_2[_blk].w) : "l"((const void*)(_vptr_2 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale2 = expert_scales[2 * tokens + early_token];
        long long expert_elem3 = (long long)(3 * tokens) * 7168 + (long long)elem;
        float _vec_load_3[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem3 + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_3[_blk].x), "=r"(_vld_3[_blk].y), "=r"(_vld_3[_blk].z), "=r"(_vld_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale3 = expert_scales[3 * tokens + early_token];
        long long expert_elem4 = (long long)(4 * tokens) * 7168 + (long long)elem;
        float _vec_load_4[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem4 + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_4[_blk].x), "=r"(_vld_4[_blk].y), "=r"(_vld_4[_blk].z), "=r"(_vld_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale4 = expert_scales[4 * tokens + early_token];
        long long expert_elem5 = (long long)(5 * tokens) * 7168 + (long long)elem;
        float _vec_load_5[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem5 + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_5[_blk].x), "=r"(_vld_5[_blk].y), "=r"(_vld_5[_blk].z), "=r"(_vld_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale5 = expert_scales[5 * tokens + early_token];
        long long expert_elem6 = (long long)(6 * tokens) * 7168 + (long long)elem;
        float _vec_load_6[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem6 + 0);
            uint4 _vld_6[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_6[_blk].x), "=r"(_vld_6[_blk].y), "=r"(_vld_6[_blk].z), "=r"(_vld_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale6 = expert_scales[6 * tokens + early_token];
        long long expert_elem7 = (long long)(7 * tokens) * 7168 + (long long)elem;
        float _vec_load_7[8];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem7 + 0);
            uint4 _vld_7[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_7[_blk].x), "=r"(_vld_7[_blk].y), "=r"(_vld_7[_blk].z), "=r"(_vld_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale7 = expert_scales[7 * tokens + early_token];
        float scaled[8];
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            scaled[j_1] = _vec_load_0[j_1] * expert_scale0;
        }
        unsigned int scaled_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_1 = 0; pair_1 < 4; pair_1++) {
            uint32_t bf16x2_add;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add) : "r"(packed_acc[pair_1]), "r"(scaled_bf16[pair_1]));
            packed_acc[pair_1] = bf16x2_add;
        }
        float scaled_0[8];
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            scaled_0[j_2] = _vec_load_1[j_2] * expert_scale1;
        }
        unsigned int scaled_bf16_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
            scaled_bf16_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_2 = 0; pair_2 < 4; pair_2++) {
            uint32_t bf16x2_add_1;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_1) : "r"(packed_acc[pair_2]), "r"(scaled_bf16_0[pair_2]));
            packed_acc[pair_2] = bf16x2_add_1;
        }
        float scaled_1[8];
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            scaled_1[j_3] = _vec_load_2[j_3] * expert_scale2;
        }
        unsigned int scaled_bf16_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
            scaled_bf16_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_3 = 0; pair_3 < 4; pair_3++) {
            uint32_t bf16x2_add_2;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_2) : "r"(packed_acc[pair_3]), "r"(scaled_bf16_1[pair_3]));
            packed_acc[pair_3] = bf16x2_add_2;
        }
        float scaled_2[8];
        #pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
            scaled_2[j_4] = _vec_load_3[j_4] * expert_scale3;
        }
        unsigned int scaled_bf16_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
            scaled_bf16_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_4 = 0; pair_4 < 4; pair_4++) {
            uint32_t bf16x2_add_3;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_3) : "r"(packed_acc[pair_4]), "r"(scaled_bf16_2[pair_4]));
            packed_acc[pair_4] = bf16x2_add_3;
        }
        float scaled_3[8];
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            scaled_3[j_5] = _vec_load_4[j_5] * expert_scale4;
        }
        unsigned int scaled_bf16_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_3[_lp*2 + 0], scaled_3[_lp*2+1 + 0]));
            scaled_bf16_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_5 = 0; pair_5 < 4; pair_5++) {
            uint32_t bf16x2_add_4;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_4) : "r"(packed_acc[pair_5]), "r"(scaled_bf16_3[pair_5]));
            packed_acc[pair_5] = bf16x2_add_4;
        }
        float scaled_4[8];
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            scaled_4[j_6] = _vec_load_5[j_6] * expert_scale5;
        }
        unsigned int scaled_bf16_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
            scaled_bf16_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_6 = 0; pair_6 < 4; pair_6++) {
            uint32_t bf16x2_add_5;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_5) : "r"(packed_acc[pair_6]), "r"(scaled_bf16_4[pair_6]));
            packed_acc[pair_6] = bf16x2_add_5;
        }
        float scaled_5[8];
        #pragma unroll
        for (int j_7 = 0; j_7 < 8; j_7++) {
            scaled_5[j_7] = _vec_load_6[j_7] * expert_scale6;
        }
        unsigned int scaled_bf16_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_5[_lp*2 + 0], scaled_5[_lp*2+1 + 0]));
            scaled_bf16_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_7 = 0; pair_7 < 4; pair_7++) {
            uint32_t bf16x2_add_6;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_6) : "r"(packed_acc[pair_7]), "r"(scaled_bf16_5[pair_7]));
            packed_acc[pair_7] = bf16x2_add_6;
        }
        float scaled_6[8];
        #pragma unroll
        for (int j_8 = 0; j_8 < 8; j_8++) {
            scaled_6[j_8] = _vec_load_7[j_8] * expert_scale7;
        }
        unsigned int scaled_bf16_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_6[_lp*2 + 0], scaled_6[_lp*2+1 + 0]));
            scaled_bf16_6[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_8 = 0; pair_8 < 4; pair_8++) {
            uint32_t bf16x2_add_7;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_7) : "r"(packed_acc[pair_8]), "r"(scaled_bf16_6[pair_8]));
            packed_acc[pair_8] = bf16x2_add_7;
        }
        control_entry = workspace_tensor[12];
        block_taken = 1;
    }
    if (active_experts == 12) {
        long long expert_elem0_1 = (long long)0 * 7168 + (long long)elem;
        float _vec_load_8[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem0_1 + 0);
            uint4 _vld_8[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_8[_blk].x), "=r"(_vld_8[_blk].y), "=r"(_vld_8[_blk].z), "=r"(_vld_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale0_1 = expert_scales[early_token];
        long long expert_elem1_1 = (long long)tokens * 7168 + (long long)elem;
        float _vec_load_9[8];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem1_1 + 0);
            uint4 _vld_9[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_9[_blk].x), "=r"(_vld_9[_blk].y), "=r"(_vld_9[_blk].z), "=r"(_vld_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale1_1 = expert_scales[tokens + early_token];
        long long expert_elem2_1 = (long long)(2 * tokens) * 7168 + (long long)elem;
        float _vec_load_10[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem2_1 + 0);
            uint4 _vld_10[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_10[_blk].x), "=r"(_vld_10[_blk].y), "=r"(_vld_10[_blk].z), "=r"(_vld_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale2_1 = expert_scales[2 * tokens + early_token];
        long long expert_elem3_1 = (long long)(3 * tokens) * 7168 + (long long)elem;
        float _vec_load_11[8];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem3_1 + 0);
            uint4 _vld_11[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_11[_blk].x), "=r"(_vld_11[_blk].y), "=r"(_vld_11[_blk].z), "=r"(_vld_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale3_1 = expert_scales[3 * tokens + early_token];
        long long expert_elem4_1 = (long long)(4 * tokens) * 7168 + (long long)elem;
        float _vec_load_12[8];
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem4_1 + 0);
            uint4 _vld_12[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_12[_blk].x), "=r"(_vld_12[_blk].y), "=r"(_vld_12[_blk].z), "=r"(_vld_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale4_1 = expert_scales[4 * tokens + early_token];
        long long expert_elem5_1 = (long long)(5 * tokens) * 7168 + (long long)elem;
        float _vec_load_13[8];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem5_1 + 0);
            uint4 _vld_13[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_13[_blk].x), "=r"(_vld_13[_blk].y), "=r"(_vld_13[_blk].z), "=r"(_vld_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale5_1 = expert_scales[5 * tokens + early_token];
        long long expert_elem6_1 = (long long)(6 * tokens) * 7168 + (long long)elem;
        float _vec_load_14[8];
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem6_1 + 0);
            uint4 _vld_14[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_14[_blk].x), "=r"(_vld_14[_blk].y), "=r"(_vld_14[_blk].z), "=r"(_vld_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale6_1 = expert_scales[6 * tokens + early_token];
        long long expert_elem7_1 = (long long)(7 * tokens) * 7168 + (long long)elem;
        float _vec_load_15[8];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem7_1 + 0);
            uint4 _vld_15[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_15[_blk].x), "=r"(_vld_15[_blk].y), "=r"(_vld_15[_blk].z), "=r"(_vld_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale7_1 = expert_scales[7 * tokens + early_token];
        long long expert_elem8 = (long long)(8 * tokens) * 7168 + (long long)elem;
        float _vec_load_16[8];
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem8 + 0);
            uint4 _vld_16[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_16[_blk].x), "=r"(_vld_16[_blk].y), "=r"(_vld_16[_blk].z), "=r"(_vld_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale8 = expert_scales[8 * tokens + early_token];
        long long expert_elem9 = (long long)(9 * tokens) * 7168 + (long long)elem;
        float _vec_load_17[8];
        {
            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem9 + 0);
            uint4 _vld_17[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_17[_blk].x), "=r"(_vld_17[_blk].y), "=r"(_vld_17[_blk].z), "=r"(_vld_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale9 = expert_scales[9 * tokens + early_token];
        long long expert_elem10 = (long long)(10 * tokens) * 7168 + (long long)elem;
        float _vec_load_18[8];
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem10 + 0);
            uint4 _vld_18[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_18[_blk].x), "=r"(_vld_18[_blk].y), "=r"(_vld_18[_blk].z), "=r"(_vld_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale10 = expert_scales[10 * tokens + early_token];
        long long expert_elem11 = (long long)(11 * tokens) * 7168 + (long long)elem;
        float _vec_load_19[8];
        {
            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem11 + 0);
            uint4 _vld_19[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_19[_blk].x), "=r"(_vld_19[_blk].y), "=r"(_vld_19[_blk].z), "=r"(_vld_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale11 = expert_scales[11 * tokens + early_token];
        control_entry = workspace_tensor[12];
        float scaled_7[8];
        #pragma unroll
        for (int j_9 = 0; j_9 < 8; j_9++) {
            scaled_7[j_9] = _vec_load_8[j_9] * expert_scale0_1;
        }
        unsigned int scaled_bf16_7[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_7[_lp*2 + 0], scaled_7[_lp*2+1 + 0]));
            scaled_bf16_7[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_9 = 0; pair_9 < 4; pair_9++) {
            uint32_t bf16x2_add_8;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_8) : "r"(packed_acc[pair_9]), "r"(scaled_bf16_7[pair_9]));
            packed_acc[pair_9] = bf16x2_add_8;
        }
        float scaled_7_1[8];
        #pragma unroll
        for (int j_10 = 0; j_10 < 8; j_10++) {
            scaled_7_1[j_10] = _vec_load_9[j_10] * expert_scale1_1;
        }
        unsigned int scaled_bf16_7_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_7_1[_lp*2 + 0], scaled_7_1[_lp*2+1 + 0]));
            scaled_bf16_7_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_10 = 0; pair_10 < 4; pair_10++) {
            uint32_t bf16x2_add_9;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_9) : "r"(packed_acc[pair_10]), "r"(scaled_bf16_7_1[pair_10]));
            packed_acc[pair_10] = bf16x2_add_9;
        }
        float scaled_8[8];
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            scaled_8[j_11] = _vec_load_10[j_11] * expert_scale2_1;
        }
        unsigned int scaled_bf16_8[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_8[_lp*2 + 0], scaled_8[_lp*2+1 + 0]));
            scaled_bf16_8[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_11 = 0; pair_11 < 4; pair_11++) {
            uint32_t bf16x2_add_10;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_10) : "r"(packed_acc[pair_11]), "r"(scaled_bf16_8[pair_11]));
            packed_acc[pair_11] = bf16x2_add_10;
        }
        float scaled_9[8];
        #pragma unroll
        for (int j_12 = 0; j_12 < 8; j_12++) {
            scaled_9[j_12] = _vec_load_11[j_12] * expert_scale3_1;
        }
        unsigned int scaled_bf16_9[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_9[_lp*2 + 0], scaled_9[_lp*2+1 + 0]));
            scaled_bf16_9[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_12 = 0; pair_12 < 4; pair_12++) {
            uint32_t bf16x2_add_11;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_11) : "r"(packed_acc[pair_12]), "r"(scaled_bf16_9[pair_12]));
            packed_acc[pair_12] = bf16x2_add_11;
        }
        float scaled_10[8];
        #pragma unroll
        for (int j_13 = 0; j_13 < 8; j_13++) {
            scaled_10[j_13] = _vec_load_12[j_13] * expert_scale4_1;
        }
        unsigned int scaled_bf16_10[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_10[_lp*2 + 0], scaled_10[_lp*2+1 + 0]));
            scaled_bf16_10[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_13 = 0; pair_13 < 4; pair_13++) {
            uint32_t bf16x2_add_12;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_12) : "r"(packed_acc[pair_13]), "r"(scaled_bf16_10[pair_13]));
            packed_acc[pair_13] = bf16x2_add_12;
        }
        float scaled_11[8];
        #pragma unroll
        for (int j_14 = 0; j_14 < 8; j_14++) {
            scaled_11[j_14] = _vec_load_13[j_14] * expert_scale5_1;
        }
        unsigned int scaled_bf16_11[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_11[_lp*2 + 0], scaled_11[_lp*2+1 + 0]));
            scaled_bf16_11[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_14 = 0; pair_14 < 4; pair_14++) {
            uint32_t bf16x2_add_13;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_13) : "r"(packed_acc[pair_14]), "r"(scaled_bf16_11[pair_14]));
            packed_acc[pair_14] = bf16x2_add_13;
        }
        float scaled_12[8];
        #pragma unroll
        for (int j_15 = 0; j_15 < 8; j_15++) {
            scaled_12[j_15] = _vec_load_14[j_15] * expert_scale6_1;
        }
        unsigned int scaled_bf16_12[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_12[_lp*2 + 0], scaled_12[_lp*2+1 + 0]));
            scaled_bf16_12[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_15 = 0; pair_15 < 4; pair_15++) {
            uint32_t bf16x2_add_14;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_14) : "r"(packed_acc[pair_15]), "r"(scaled_bf16_12[pair_15]));
            packed_acc[pair_15] = bf16x2_add_14;
        }
        float scaled_13[8];
        #pragma unroll
        for (int j_16 = 0; j_16 < 8; j_16++) {
            scaled_13[j_16] = _vec_load_15[j_16] * expert_scale7_1;
        }
        unsigned int scaled_bf16_13[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_13[_lp*2 + 0], scaled_13[_lp*2+1 + 0]));
            scaled_bf16_13[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_16 = 0; pair_16 < 4; pair_16++) {
            uint32_t bf16x2_add_15;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_15) : "r"(packed_acc[pair_16]), "r"(scaled_bf16_13[pair_16]));
            packed_acc[pair_16] = bf16x2_add_15;
        }
        float scaled_14[8];
        #pragma unroll
        for (int j_17 = 0; j_17 < 8; j_17++) {
            scaled_14[j_17] = _vec_load_16[j_17] * expert_scale8;
        }
        unsigned int scaled_bf16_14[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_14[_lp*2 + 0], scaled_14[_lp*2+1 + 0]));
            scaled_bf16_14[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_17 = 0; pair_17 < 4; pair_17++) {
            uint32_t bf16x2_add_16;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_16) : "r"(packed_acc[pair_17]), "r"(scaled_bf16_14[pair_17]));
            packed_acc[pair_17] = bf16x2_add_16;
        }
        float scaled_15[8];
        #pragma unroll
        for (int j_18 = 0; j_18 < 8; j_18++) {
            scaled_15[j_18] = _vec_load_17[j_18] * expert_scale9;
        }
        unsigned int scaled_bf16_15[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_15[_lp*2 + 0], scaled_15[_lp*2+1 + 0]));
            scaled_bf16_15[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_18 = 0; pair_18 < 4; pair_18++) {
            uint32_t bf16x2_add_17;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_17) : "r"(packed_acc[pair_18]), "r"(scaled_bf16_15[pair_18]));
            packed_acc[pair_18] = bf16x2_add_17;
        }
        float scaled_16[8];
        #pragma unroll
        for (int j_19 = 0; j_19 < 8; j_19++) {
            scaled_16[j_19] = _vec_load_18[j_19] * expert_scale10;
        }
        unsigned int scaled_bf16_16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_16[_lp*2 + 0], scaled_16[_lp*2+1 + 0]));
            scaled_bf16_16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_19 = 0; pair_19 < 4; pair_19++) {
            uint32_t bf16x2_add_18;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_18) : "r"(packed_acc[pair_19]), "r"(scaled_bf16_16[pair_19]));
            packed_acc[pair_19] = bf16x2_add_18;
        }
        float scaled_17[8];
        #pragma unroll
        for (int j_20 = 0; j_20 < 8; j_20++) {
            scaled_17[j_20] = _vec_load_19[j_20] * expert_scale11;
        }
        unsigned int scaled_bf16_17[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_17[_lp*2 + 0], scaled_17[_lp*2+1 + 0]));
            scaled_bf16_17[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_20 = 0; pair_20 < 4; pair_20++) {
            uint32_t bf16x2_add_19;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_19) : "r"(packed_acc[pair_20]), "r"(scaled_bf16_17[pair_20]));
            packed_acc[pair_20] = bf16x2_add_19;
        }
        block_taken = 1;
    }
    if (active_experts == 16) {
        long long expert_elem0_2 = (long long)0 * 7168 + (long long)elem;
        float _vec_load_20[8];
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem0_2 + 0);
            uint4 _vld_20[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_20[_blk].x), "=r"(_vld_20[_blk].y), "=r"(_vld_20[_blk].z), "=r"(_vld_20[_blk].w) : "l"((const void*)(_vptr_20 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale0_2 = expert_scales[early_token];
        long long expert_elem1_2 = (long long)tokens * 7168 + (long long)elem;
        float _vec_load_21[8];
        {
            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem1_2 + 0);
            uint4 _vld_21[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_21[_blk].x), "=r"(_vld_21[_blk].y), "=r"(_vld_21[_blk].z), "=r"(_vld_21[_blk].w) : "l"((const void*)(_vptr_21 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale1_2 = expert_scales[tokens + early_token];
        long long expert_elem2_2 = (long long)(2 * tokens) * 7168 + (long long)elem;
        float _vec_load_22[8];
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem2_2 + 0);
            uint4 _vld_22[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_22[_blk].x), "=r"(_vld_22[_blk].y), "=r"(_vld_22[_blk].z), "=r"(_vld_22[_blk].w) : "l"((const void*)(_vptr_22 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_22[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
                    (&_vec_load_22[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale2_2 = expert_scales[2 * tokens + early_token];
        long long expert_elem3_2 = (long long)(3 * tokens) * 7168 + (long long)elem;
        float _vec_load_23[8];
        {
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem3_2 + 0);
            uint4 _vld_23[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_23[_blk].x), "=r"(_vld_23[_blk].y), "=r"(_vld_23[_blk].z), "=r"(_vld_23[_blk].w) : "l"((const void*)(_vptr_23 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_23[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
                    (&_vec_load_23[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale3_2 = expert_scales[3 * tokens + early_token];
        long long expert_elem4_2 = (long long)(4 * tokens) * 7168 + (long long)elem;
        float _vec_load_24[8];
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem4_2 + 0);
            uint4 _vld_24[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_24[_blk].x), "=r"(_vld_24[_blk].y), "=r"(_vld_24[_blk].z), "=r"(_vld_24[_blk].w) : "l"((const void*)(_vptr_24 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_24[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                    (&_vec_load_24[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale4_2 = expert_scales[4 * tokens + early_token];
        long long expert_elem5_2 = (long long)(5 * tokens) * 7168 + (long long)elem;
        float _vec_load_25[8];
        {
            const uint4* _vptr_25 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem5_2 + 0);
            uint4 _vld_25[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_25[_blk].x), "=r"(_vld_25[_blk].y), "=r"(_vld_25[_blk].z), "=r"(_vld_25[_blk].w) : "l"((const void*)(_vptr_25 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_25[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                    (&_vec_load_25[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale5_2 = expert_scales[5 * tokens + early_token];
        long long expert_elem6_2 = (long long)(6 * tokens) * 7168 + (long long)elem;
        float _vec_load_26[8];
        {
            const uint4* _vptr_26 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem6_2 + 0);
            uint4 _vld_26[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_26[_blk].x), "=r"(_vld_26[_blk].y), "=r"(_vld_26[_blk].z), "=r"(_vld_26[_blk].w) : "l"((const void*)(_vptr_26 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_26[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
                    (&_vec_load_26[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale6_2 = expert_scales[6 * tokens + early_token];
        long long expert_elem7_2 = (long long)(7 * tokens) * 7168 + (long long)elem;
        float _vec_load_27[8];
        {
            const uint4* _vptr_27 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem7_2 + 0);
            uint4 _vld_27[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_27[_blk].x), "=r"(_vld_27[_blk].y), "=r"(_vld_27[_blk].z), "=r"(_vld_27[_blk].w) : "l"((const void*)(_vptr_27 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_27[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
                    (&_vec_load_27[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale7_2 = expert_scales[7 * tokens + early_token];
        long long expert_elem8_1 = (long long)(8 * tokens) * 7168 + (long long)elem;
        float _vec_load_28[8];
        {
            const uint4* _vptr_28 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem8_1 + 0);
            uint4 _vld_28[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_28[_blk].x), "=r"(_vld_28[_blk].y), "=r"(_vld_28[_blk].z), "=r"(_vld_28[_blk].w) : "l"((const void*)(_vptr_28 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_28[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) << 16);
                    (&_vec_load_28[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale8_1 = expert_scales[8 * tokens + early_token];
        long long expert_elem9_1 = (long long)(9 * tokens) * 7168 + (long long)elem;
        float _vec_load_29[8];
        {
            const uint4* _vptr_29 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem9_1 + 0);
            uint4 _vld_29[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_29[_blk].x), "=r"(_vld_29[_blk].y), "=r"(_vld_29[_blk].z), "=r"(_vld_29[_blk].w) : "l"((const void*)(_vptr_29 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_29[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
                    (&_vec_load_29[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale9_1 = expert_scales[9 * tokens + early_token];
        long long expert_elem10_1 = (long long)(10 * tokens) * 7168 + (long long)elem;
        float _vec_load_30[8];
        {
            const uint4* _vptr_30 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem10_1 + 0);
            uint4 _vld_30[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_30[_blk].x), "=r"(_vld_30[_blk].y), "=r"(_vld_30[_blk].z), "=r"(_vld_30[_blk].w) : "l"((const void*)(_vptr_30 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_30[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                    (&_vec_load_30[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale10_1 = expert_scales[10 * tokens + early_token];
        long long expert_elem11_1 = (long long)(11 * tokens) * 7168 + (long long)elem;
        float _vec_load_31[8];
        {
            const uint4* _vptr_31 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem11_1 + 0);
            uint4 _vld_31[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_31[_blk].x), "=r"(_vld_31[_blk].y), "=r"(_vld_31[_blk].z), "=r"(_vld_31[_blk].w) : "l"((const void*)(_vptr_31 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_31[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                    (&_vec_load_31[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale11_1 = expert_scales[11 * tokens + early_token];
        long long expert_elem12 = (long long)(12 * tokens) * 7168 + (long long)elem;
        float _vec_load_32[8];
        {
            const uint4* _vptr_32 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem12 + 0);
            uint4 _vld_32[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_32[_blk].x), "=r"(_vld_32[_blk].y), "=r"(_vld_32[_blk].z), "=r"(_vld_32[_blk].w) : "l"((const void*)(_vptr_32 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_32[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                    (&_vec_load_32[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale12 = expert_scales[12 * tokens + early_token];
        long long expert_elem13 = (long long)(13 * tokens) * 7168 + (long long)elem;
        float _vec_load_33[8];
        {
            const uint4* _vptr_33 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem13 + 0);
            uint4 _vld_33[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_33[_blk].x), "=r"(_vld_33[_blk].y), "=r"(_vld_33[_blk].z), "=r"(_vld_33[_blk].w) : "l"((const void*)(_vptr_33 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_33[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                    (&_vec_load_33[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale13 = expert_scales[13 * tokens + early_token];
        long long expert_elem14 = (long long)(14 * tokens) * 7168 + (long long)elem;
        float _vec_load_34[8];
        {
            const uint4* _vptr_34 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem14 + 0);
            uint4 _vld_34[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_34[_blk].x), "=r"(_vld_34[_blk].y), "=r"(_vld_34[_blk].z), "=r"(_vld_34[_blk].w) : "l"((const void*)(_vptr_34 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_34[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                    (&_vec_load_34[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale14 = expert_scales[14 * tokens + early_token];
        long long expert_elem15 = (long long)(15 * tokens) * 7168 + (long long)elem;
        float _vec_load_35[8];
        {
            const uint4* _vptr_35 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem15 + 0);
            uint4 _vld_35[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_35[_blk].x), "=r"(_vld_35[_blk].y), "=r"(_vld_35[_blk].z), "=r"(_vld_35[_blk].w) : "l"((const void*)(_vptr_35 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_35[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
                    (&_vec_load_35[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale15 = expert_scales[15 * tokens + early_token];
        control_entry = workspace_tensor[12];
        float scaled_18[8];
        #pragma unroll
        for (int j_21 = 0; j_21 < 8; j_21++) {
            scaled_18[j_21] = _vec_load_20[j_21] * expert_scale0_2;
        }
        unsigned int scaled_bf16_18[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_18[_lp*2 + 0], scaled_18[_lp*2+1 + 0]));
            scaled_bf16_18[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_21 = 0; pair_21 < 4; pair_21++) {
            uint32_t bf16x2_add_20;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_20) : "r"(packed_acc[pair_21]), "r"(scaled_bf16_18[pair_21]));
            packed_acc[pair_21] = bf16x2_add_20;
        }
        float scaled_18_1[8];
        #pragma unroll
        for (int j_22 = 0; j_22 < 8; j_22++) {
            scaled_18_1[j_22] = _vec_load_21[j_22] * expert_scale1_2;
        }
        unsigned int scaled_bf16_18_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_18_1[_lp*2 + 0], scaled_18_1[_lp*2+1 + 0]));
            scaled_bf16_18_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_22 = 0; pair_22 < 4; pair_22++) {
            uint32_t bf16x2_add_21;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_21) : "r"(packed_acc[pair_22]), "r"(scaled_bf16_18_1[pair_22]));
            packed_acc[pair_22] = bf16x2_add_21;
        }
        float scaled_19[8];
        #pragma unroll
        for (int j_23 = 0; j_23 < 8; j_23++) {
            scaled_19[j_23] = _vec_load_22[j_23] * expert_scale2_2;
        }
        unsigned int scaled_bf16_19[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_19[_lp*2 + 0], scaled_19[_lp*2+1 + 0]));
            scaled_bf16_19[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_23 = 0; pair_23 < 4; pair_23++) {
            uint32_t bf16x2_add_22;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_22) : "r"(packed_acc[pair_23]), "r"(scaled_bf16_19[pair_23]));
            packed_acc[pair_23] = bf16x2_add_22;
        }
        float scaled_20[8];
        #pragma unroll
        for (int j_24 = 0; j_24 < 8; j_24++) {
            scaled_20[j_24] = _vec_load_23[j_24] * expert_scale3_2;
        }
        unsigned int scaled_bf16_20[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_20[_lp*2 + 0], scaled_20[_lp*2+1 + 0]));
            scaled_bf16_20[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_24 = 0; pair_24 < 4; pair_24++) {
            uint32_t bf16x2_add_23;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_23) : "r"(packed_acc[pair_24]), "r"(scaled_bf16_20[pair_24]));
            packed_acc[pair_24] = bf16x2_add_23;
        }
        float scaled_21[8];
        #pragma unroll
        for (int j_25 = 0; j_25 < 8; j_25++) {
            scaled_21[j_25] = _vec_load_24[j_25] * expert_scale4_2;
        }
        unsigned int scaled_bf16_21[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_21[_lp*2 + 0], scaled_21[_lp*2+1 + 0]));
            scaled_bf16_21[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_25 = 0; pair_25 < 4; pair_25++) {
            uint32_t bf16x2_add_24;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_24) : "r"(packed_acc[pair_25]), "r"(scaled_bf16_21[pair_25]));
            packed_acc[pair_25] = bf16x2_add_24;
        }
        float scaled_22[8];
        #pragma unroll
        for (int j_26 = 0; j_26 < 8; j_26++) {
            scaled_22[j_26] = _vec_load_25[j_26] * expert_scale5_2;
        }
        unsigned int scaled_bf16_22[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_22[_lp*2 + 0], scaled_22[_lp*2+1 + 0]));
            scaled_bf16_22[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_26 = 0; pair_26 < 4; pair_26++) {
            uint32_t bf16x2_add_25;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_25) : "r"(packed_acc[pair_26]), "r"(scaled_bf16_22[pair_26]));
            packed_acc[pair_26] = bf16x2_add_25;
        }
        float scaled_23[8];
        #pragma unroll
        for (int j_27 = 0; j_27 < 8; j_27++) {
            scaled_23[j_27] = _vec_load_26[j_27] * expert_scale6_2;
        }
        unsigned int scaled_bf16_23[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_23[_lp*2 + 0], scaled_23[_lp*2+1 + 0]));
            scaled_bf16_23[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_27 = 0; pair_27 < 4; pair_27++) {
            uint32_t bf16x2_add_26;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_26) : "r"(packed_acc[pair_27]), "r"(scaled_bf16_23[pair_27]));
            packed_acc[pair_27] = bf16x2_add_26;
        }
        float scaled_24[8];
        #pragma unroll
        for (int j_28 = 0; j_28 < 8; j_28++) {
            scaled_24[j_28] = _vec_load_27[j_28] * expert_scale7_2;
        }
        unsigned int scaled_bf16_24[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_24[_lp*2 + 0], scaled_24[_lp*2+1 + 0]));
            scaled_bf16_24[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_28 = 0; pair_28 < 4; pair_28++) {
            uint32_t bf16x2_add_27;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_27) : "r"(packed_acc[pair_28]), "r"(scaled_bf16_24[pair_28]));
            packed_acc[pair_28] = bf16x2_add_27;
        }
        float scaled_25[8];
        #pragma unroll
        for (int j_29 = 0; j_29 < 8; j_29++) {
            scaled_25[j_29] = _vec_load_28[j_29] * expert_scale8_1;
        }
        unsigned int scaled_bf16_25[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_25[_lp*2 + 0], scaled_25[_lp*2+1 + 0]));
            scaled_bf16_25[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_29 = 0; pair_29 < 4; pair_29++) {
            uint32_t bf16x2_add_28;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_28) : "r"(packed_acc[pair_29]), "r"(scaled_bf16_25[pair_29]));
            packed_acc[pair_29] = bf16x2_add_28;
        }
        float scaled_26[8];
        #pragma unroll
        for (int j_30 = 0; j_30 < 8; j_30++) {
            scaled_26[j_30] = _vec_load_29[j_30] * expert_scale9_1;
        }
        unsigned int scaled_bf16_26[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_26[_lp*2 + 0], scaled_26[_lp*2+1 + 0]));
            scaled_bf16_26[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_30 = 0; pair_30 < 4; pair_30++) {
            uint32_t bf16x2_add_29;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_29) : "r"(packed_acc[pair_30]), "r"(scaled_bf16_26[pair_30]));
            packed_acc[pair_30] = bf16x2_add_29;
        }
        float scaled_27[8];
        #pragma unroll
        for (int j_31 = 0; j_31 < 8; j_31++) {
            scaled_27[j_31] = _vec_load_30[j_31] * expert_scale10_1;
        }
        unsigned int scaled_bf16_27[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_27[_lp*2 + 0], scaled_27[_lp*2+1 + 0]));
            scaled_bf16_27[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_31 = 0; pair_31 < 4; pair_31++) {
            uint32_t bf16x2_add_30;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_30) : "r"(packed_acc[pair_31]), "r"(scaled_bf16_27[pair_31]));
            packed_acc[pair_31] = bf16x2_add_30;
        }
        float scaled_28[8];
        #pragma unroll
        for (int j_32 = 0; j_32 < 8; j_32++) {
            scaled_28[j_32] = _vec_load_31[j_32] * expert_scale11_1;
        }
        unsigned int scaled_bf16_28[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_28[_lp*2 + 0], scaled_28[_lp*2+1 + 0]));
            scaled_bf16_28[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_32 = 0; pair_32 < 4; pair_32++) {
            uint32_t bf16x2_add_31;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_31) : "r"(packed_acc[pair_32]), "r"(scaled_bf16_28[pair_32]));
            packed_acc[pair_32] = bf16x2_add_31;
        }
        float scaled_29[8];
        #pragma unroll
        for (int j_33 = 0; j_33 < 8; j_33++) {
            scaled_29[j_33] = _vec_load_32[j_33] * expert_scale12;
        }
        unsigned int scaled_bf16_29[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_29[_lp*2 + 0], scaled_29[_lp*2+1 + 0]));
            scaled_bf16_29[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_33 = 0; pair_33 < 4; pair_33++) {
            uint32_t bf16x2_add_32;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_32) : "r"(packed_acc[pair_33]), "r"(scaled_bf16_29[pair_33]));
            packed_acc[pair_33] = bf16x2_add_32;
        }
        float scaled_30[8];
        #pragma unroll
        for (int j_34 = 0; j_34 < 8; j_34++) {
            scaled_30[j_34] = _vec_load_33[j_34] * expert_scale13;
        }
        unsigned int scaled_bf16_30[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_30[_lp*2 + 0], scaled_30[_lp*2+1 + 0]));
            scaled_bf16_30[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_34 = 0; pair_34 < 4; pair_34++) {
            uint32_t bf16x2_add_33;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_33) : "r"(packed_acc[pair_34]), "r"(scaled_bf16_30[pair_34]));
            packed_acc[pair_34] = bf16x2_add_33;
        }
        float scaled_31[8];
        #pragma unroll
        for (int j_35 = 0; j_35 < 8; j_35++) {
            scaled_31[j_35] = _vec_load_34[j_35] * expert_scale14;
        }
        unsigned int scaled_bf16_31[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_31[_lp*2 + 0], scaled_31[_lp*2+1 + 0]));
            scaled_bf16_31[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_35 = 0; pair_35 < 4; pair_35++) {
            uint32_t bf16x2_add_34;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_34) : "r"(packed_acc[pair_35]), "r"(scaled_bf16_31[pair_35]));
            packed_acc[pair_35] = bf16x2_add_34;
        }
        float scaled_32[8];
        #pragma unroll
        for (int j_36 = 0; j_36 < 8; j_36++) {
            scaled_32[j_36] = _vec_load_35[j_36] * expert_scale15;
        }
        unsigned int scaled_bf16_32[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_32[_lp*2 + 0], scaled_32[_lp*2+1 + 0]));
            scaled_bf16_32[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_36 = 0; pair_36 < 4; pair_36++) {
            uint32_t bf16x2_add_35;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_35) : "r"(packed_acc[pair_36]), "r"(scaled_bf16_32[pair_36]));
            packed_acc[pair_36] = bf16x2_add_35;
        }
        block_taken = 1;
    }
    if (block_taken == 0) {
        control_entry = workspace_tensor[12];
    }
    int expert_loop_end = ((block_taken == 1) ? 0 : active_experts);
    #pragma unroll 4
    for (int expert = 0; expert < expert_loop_end; expert++) {
        long long expert_elem = (long long)expert * (long long)tokens * 7168 + (long long)elem;
        float _vec_load_36[8];
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
                    (&_vec_load_36[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                    (&_vec_load_36[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                }
            }
        }
        float expert_scale = expert_scales[expert * tokens + early_token];
        float scaled_33[8];
        #pragma unroll
        for (int j_37 = 0; j_37 < 8; j_37++) {
            scaled_33[j_37] = _vec_load_36[j_37] * expert_scale;
        }
        unsigned int scaled_bf16_33[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_33[_lp*2 + 0], scaled_33[_lp*2+1 + 0]));
            scaled_bf16_33[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int pair_37 = 0; pair_37 < 4; pair_37++) {
            uint32_t bf16x2_add_36;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_36) : "r"(packed_acc[pair_37]), "r"(scaled_bf16_33[pair_37]));
            packed_acc[pair_37] = bf16x2_add_36;
        }
    }
    long long workspace_entry = workspace_tensor[8 + world_rank];
    long long owner_entry = workspace_tensor[8 + early_thread / 224];
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
    int rs_items = tokens * 224;
    int publish_start = cluster_id;
    if ((int)cluster_id < tokens) {
        int owner = cluster_thread / 224;
        long long slot = (long long)data_epoch * comm_stride_elems + (long long)rank * (long long)rs_items * 8 + (long long)((int)cluster_id * 224 + (cluster_thread - owner * 224)) * 8;
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
    }
    publish_start = (int)(cluster_id + num_clusters);
    #pragma unroll 1
    for (int token = publish_start; token < tokens; token += (int)num_clusters) {
        if (token + (int)num_clusters >= tokens) {
            int pub_rs_src = cluster_thread % 4;
            int pub_rs_item = cluster_thread / 4;
            __nv_bfloat16* pub_rs_base = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[8 + pub_rs_src]);
            unsigned int pub_rs_words[4];
            uint32_t sysv_ld[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld[0]), "=r"(sysv_ld[1]), "=r"(sysv_ld[2]), "=r"(sysv_ld[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(pub_rs_src * rs_items + (int)cluster_id * 224 + pub_rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word = 0; word < 4; word++) {
                pub_rs_words[word] = sysv_ld[word];
            }
            int pub_lead_token = (int)cluster_id + (int)num_clusters;
            unsigned int pub_lead_words[4];
            if (pub_lead_token < token) {
                uint32_t sysv_ld_0[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_0[0]), "=r"(sysv_ld_0[1]), "=r"(sysv_ld_0[2]), "=r"(sysv_ld_0[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(pub_rs_src * rs_items + pub_lead_token * 224 + pub_rs_item) * 8)) : "memory");
                #pragma unroll
                for (int word_1 = 0; word_1 < 4; word_1++) {
                    pub_lead_words[word_1] = sysv_ld_0[word_1];
                }
            }
            if ((int)cluster_id < tokens) {
                int pending = 0;
                #pragma unroll
                for (int word_2 = 0; word_2 < 4; word_2++) {
                    pending = (((pub_rs_words[word_2] & 65535) == 32768) ? 1 : pending);
                    pending = ((pub_rs_words[word_2] >> 16 == 32768) ? 1 : pending);
                }
                if (pending == 1) {
                    uint32_t sysv_poll_group[4];
                    uint32_t _sysv_pending_45 = 1u;
                    do {
                        if (_sysv_pending_45 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group[0]), "=r"(sysv_poll_group[1]), "=r"(sysv_poll_group[2]), "=r"(sysv_poll_group[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(pub_rs_src * rs_items + (int)cluster_id * 224 + pub_rs_item) * 8)) : "memory");
                        }
                        if (_sysv_pending_45 & 1u) {
                            if (!((((sysv_poll_group[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_45 &= ~1u;
                        }
                    } while (_sysv_pending_45 != 0u);
                    #pragma unroll
                    for (int word_3 = 0; word_3 < 4; word_3++) {
                        pub_rs_words[word_3] = sysv_poll_group[word_3];
                    }
                }
                unsigned int ag_words[4];
                unsigned int _shfl_0 = __shfl_sync(4294967295, pub_rs_words[0], 0, 4);
                unsigned int _shfl_16_0 = __shfl_sync(4294967295, pub_rs_words[0], 1, 4);
                unsigned int _shfl_17_0 = __shfl_sync(4294967295, pub_rs_words[0], 2, 4);
                unsigned int _shfl_18_0 = __shfl_sync(4294967295, pub_rs_words[0], 3, 4);
                float lanes[2];
                unsigned int word_4[1];
                word_4[0] = _shfl_0;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) & 0xffff0000u);
                }
                float received_lanes[2];
                word_4[0] = _shfl_16_0;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_40 = 0; j_40 < 2; j_40++) {
                    lanes[j_40] = lanes[j_40] + received_lanes[j_40];
                }
                unsigned int lanes_bf16[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16[_pair]) & 0xffff0000u);
                }
                word_4[0] = _shfl_17_0;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_41 = 0; j_41 < 2; j_41++) {
                    lanes[j_41] = lanes[j_41] + received_lanes[j_41];
                }
                unsigned int lanes_bf16_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1[_pair]) & 0xffff0000u);
                }
                word_4[0] = _shfl_18_0;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) << 16);
                    (&received_lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_42 = 0; j_42 < 2; j_42++) {
                    lanes[j_42] = lanes[j_42] + received_lanes[j_42];
                }
                unsigned int lanes_bf16_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes[_lp*2 + 0], lanes[_lp*2+1 + 0]));
                    lanes_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2[_pair]) << 16);
                    (&lanes[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2[_pair]) & 0xffff0000u);
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
                ag_words[0] = low | high << 16;
                unsigned int _shfl_1 = __shfl_sync(4294967295, pub_rs_words[1], 0, 4);
                unsigned int _shfl_16_1 = __shfl_sync(4294967295, pub_rs_words[1], 1, 4);
                unsigned int _shfl_17_1 = __shfl_sync(4294967295, pub_rs_words[1], 2, 4);
                unsigned int _shfl_18_1 = __shfl_sync(4294967295, pub_rs_words[1], 3, 4);
                float lanes_0[2];
                unsigned int word_0[1];
                word_0[0] = _shfl_1;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) << 16);
                    (&lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) & 0xffff0000u);
                }
                float received_lanes_0[2];
                word_0[0] = _shfl_16_1;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) << 16);
                    (&received_lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_43 = 0; j_43 < 2; j_43++) {
                    lanes_0[j_43] = lanes_0[j_43] + received_lanes_0[j_43];
                }
                unsigned int lanes_bf16_0[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_0[_lp*2 + 0], lanes_0[_lp*2+1 + 0]));
                    lanes_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_0[_pair]) << 16);
                    (&lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_0[_pair]) & 0xffff0000u);
                }
                word_0[0] = _shfl_17_1;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) << 16);
                    (&received_lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_44 = 0; j_44 < 2; j_44++) {
                    lanes_0[j_44] = lanes_0[j_44] + received_lanes_0[j_44];
                }
                unsigned int lanes_bf16_1_0[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_0[_lp*2 + 0], lanes_0[_lp*2+1 + 0]));
                    lanes_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_0[_pair]) << 16);
                    (&lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_0[_pair]) & 0xffff0000u);
                }
                word_0[0] = _shfl_18_1;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) << 16);
                    (&received_lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_0[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_45 = 0; j_45 < 2; j_45++) {
                    lanes_0[j_45] = lanes_0[j_45] + received_lanes_0[j_45];
                }
                unsigned int lanes_bf16_2_0[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_0[_lp*2 + 0], lanes_0[_lp*2+1 + 0]));
                    lanes_bf16_2_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_0[_pair]) << 16);
                    (&lanes_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_0[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_35[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_0[_lp*2 + 0], lanes_0[_lp*2+1 + 0]));
                    scaled_bf16_35[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_0 = scaled_bf16_35[0] & 65535;
                unsigned int high_0 = scaled_bf16_35[0] >> 16;
                low_0 = ((low_0 == 65535) ? 65534 : low_0);
                high_0 = ((high_0 == 65535) ? 65534 : high_0);
                low_0 = ((low_0 == 32768) ? 65535 : low_0);
                high_0 = ((high_0 == 32768) ? 65535 : high_0);
                ag_words[1] = low_0 | high_0 << 16;
                unsigned int _shfl_2 = __shfl_sync(4294967295, pub_rs_words[2], 0, 4);
                unsigned int _shfl_16_2 = __shfl_sync(4294967295, pub_rs_words[2], 1, 4);
                unsigned int _shfl_17_2 = __shfl_sync(4294967295, pub_rs_words[2], 2, 4);
                unsigned int _shfl_18_2 = __shfl_sync(4294967295, pub_rs_words[2], 3, 4);
                float lanes_1[2];
                unsigned int word_1_1[1];
                word_1_1[0] = _shfl_2;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) & 0xffff0000u);
                }
                float received_lanes_1[2];
                word_1_1[0] = _shfl_16_2;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_46 = 0; j_46 < 2; j_46++) {
                    lanes_1[j_46] = lanes_1[j_46] + received_lanes_1[j_46];
                }
                unsigned int lanes_bf16_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_3[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_3[_pair]) & 0xffff0000u);
                }
                word_1_1[0] = _shfl_17_2;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_47 = 0; j_47 < 2; j_47++) {
                    lanes_1[j_47] = lanes_1[j_47] + received_lanes_1[j_47];
                }
                unsigned int lanes_bf16_1_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_1[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_1[_pair]) & 0xffff0000u);
                }
                word_1_1[0] = _shfl_18_2;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) << 16);
                    (&received_lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_1_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_48 = 0; j_48 < 2; j_48++) {
                    lanes_1[j_48] = lanes_1[j_48] + received_lanes_1[j_48];
                }
                unsigned int lanes_bf16_2_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    lanes_bf16_2_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_1[_pair]) << 16);
                    (&lanes_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_1[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_36[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_1[_lp*2 + 0], lanes_1[_lp*2+1 + 0]));
                    scaled_bf16_36[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_1 = scaled_bf16_36[0] & 65535;
                unsigned int high_1 = scaled_bf16_36[0] >> 16;
                low_1 = ((low_1 == 65535) ? 65534 : low_1);
                high_1 = ((high_1 == 65535) ? 65534 : high_1);
                low_1 = ((low_1 == 32768) ? 65535 : low_1);
                high_1 = ((high_1 == 32768) ? 65535 : high_1);
                ag_words[2] = low_1 | high_1 << 16;
                unsigned int _shfl_3 = __shfl_sync(4294967295, pub_rs_words[3], 0, 4);
                unsigned int _shfl_16_3 = __shfl_sync(4294967295, pub_rs_words[3], 1, 4);
                unsigned int _shfl_17_3 = __shfl_sync(4294967295, pub_rs_words[3], 2, 4);
                unsigned int _shfl_18_3 = __shfl_sync(4294967295, pub_rs_words[3], 3, 4);
                float lanes_2[2];
                unsigned int word_2_1[1];
                word_2_1[0] = _shfl_3;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) & 0xffff0000u);
                }
                float received_lanes_2[2];
                word_2_1[0] = _shfl_16_3;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_49 = 0; j_49 < 2; j_49++) {
                    lanes_2[j_49] = lanes_2[j_49] + received_lanes_2[j_49];
                }
                unsigned int lanes_bf16_4[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_4[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_4[_pair]) & 0xffff0000u);
                }
                word_2_1[0] = _shfl_17_3;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_50 = 0; j_50 < 2; j_50++) {
                    lanes_2[j_50] = lanes_2[j_50] + received_lanes_2[j_50];
                }
                unsigned int lanes_bf16_1_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_1_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_2[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_2[_pair]) & 0xffff0000u);
                }
                word_2_1[0] = _shfl_18_3;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) << 16);
                    (&received_lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_2_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_51 = 0; j_51 < 2; j_51++) {
                    lanes_2[j_51] = lanes_2[j_51] + received_lanes_2[j_51];
                }
                unsigned int lanes_bf16_2_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    lanes_bf16_2_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_2[_pair]) << 16);
                    (&lanes_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_2[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_37[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_2[_lp*2 + 0], lanes_2[_lp*2+1 + 0]));
                    scaled_bf16_37[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_2 = scaled_bf16_37[0] & 65535;
                unsigned int high_2 = scaled_bf16_37[0] >> 16;
                low_2 = ((low_2 == 65535) ? 65534 : low_2);
                high_2 = ((high_2 == 65535) ? 65534 : high_2);
                low_2 = ((low_2 == 32768) ? 65535 : low_2);
                high_2 = ((high_2 == 32768) ? 65535 : high_2);
                ag_words[3] = low_2 | high_2 << 16;
                {
                    auto _gst_46 = pub_rs_base + ((long long)data_epoch * comm_stride_elems + (long long)tokens * 7168 + (long long)((int)cluster_id * 896 + rank * 224 + pub_rs_item) * 8);
                    __builtin_assume(__isGlobal(_gst_46));
                    reinterpret_cast<int4*>(_gst_46)[0] = reinterpret_cast<int4*>(ag_words)[0];
                }
            }
            if (pub_lead_token < token) {
                int pending_1 = 0;
                #pragma unroll
                for (int word_5 = 0; word_5 < 4; word_5++) {
                    pending_1 = (((pub_lead_words[word_5] & 65535) == 32768) ? 1 : pending_1);
                    pending_1 = ((pub_lead_words[word_5] >> 16 == 32768) ? 1 : pending_1);
                }
                if (pending_1 == 1) {
                    uint32_t sysv_poll_group_1[4];
                    uint32_t _sysv_pending_47 = 1u;
                    do {
                        if (_sysv_pending_47 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_1[0]), "=r"(sysv_poll_group_1[1]), "=r"(sysv_poll_group_1[2]), "=r"(sysv_poll_group_1[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(pub_rs_src * rs_items + pub_lead_token * 224 + pub_rs_item) * 8)) : "memory");
                        }
                        if (_sysv_pending_47 & 1u) {
                            if (!((((sysv_poll_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_1[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_47 &= ~1u;
                        }
                    } while (_sysv_pending_47 != 0u);
                    #pragma unroll
                    for (int word_6 = 0; word_6 < 4; word_6++) {
                        pub_lead_words[word_6] = sysv_poll_group_1[word_6];
                    }
                }
                unsigned int ag_words_1[4];
                unsigned int _shfl_4 = __shfl_sync(4294967295, pub_lead_words[0], 0, 4);
                unsigned int _shfl_16_4 = __shfl_sync(4294967295, pub_lead_words[0], 1, 4);
                unsigned int _shfl_17_4 = __shfl_sync(4294967295, pub_lead_words[0], 2, 4);
                unsigned int _shfl_18_4 = __shfl_sync(4294967295, pub_lead_words[0], 3, 4);
                float lanes_3[2];
                unsigned int word_7[1];
                word_7[0] = _shfl_4;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                float received_lanes_3[2];
                word_7[0] = _shfl_16_4;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_52 = 0; j_52 < 2; j_52++) {
                    lanes_3[j_52] = lanes_3[j_52] + received_lanes_3[j_52];
                }
                unsigned int lanes_bf16_5[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3[_lp*2 + 0], lanes_3[_lp*2+1 + 0]));
                    lanes_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_5[_pair]) << 16);
                    (&lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_5[_pair]) & 0xffff0000u);
                }
                word_7[0] = _shfl_17_4;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_53 = 0; j_53 < 2; j_53++) {
                    lanes_3[j_53] = lanes_3[j_53] + received_lanes_3[j_53];
                }
                unsigned int lanes_bf16_1_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3[_lp*2 + 0], lanes_3[_lp*2+1 + 0]));
                    lanes_bf16_1_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_3[_pair]) << 16);
                    (&lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_3[_pair]) & 0xffff0000u);
                }
                word_7[0] = _shfl_18_4;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) << 16);
                    (&received_lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_54 = 0; j_54 < 2; j_54++) {
                    lanes_3[j_54] = lanes_3[j_54] + received_lanes_3[j_54];
                }
                unsigned int lanes_bf16_2_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3[_lp*2 + 0], lanes_3[_lp*2+1 + 0]));
                    lanes_bf16_2_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_3[_pair]) << 16);
                    (&lanes_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_3[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_38[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3[_lp*2 + 0], lanes_3[_lp*2+1 + 0]));
                    scaled_bf16_38[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_3 = scaled_bf16_38[0] & 65535;
                unsigned int high_3 = scaled_bf16_38[0] >> 16;
                low_3 = ((low_3 == 65535) ? 65534 : low_3);
                high_3 = ((high_3 == 65535) ? 65534 : high_3);
                low_3 = ((low_3 == 32768) ? 65535 : low_3);
                high_3 = ((high_3 == 32768) ? 65535 : high_3);
                ag_words_1[0] = low_3 | high_3 << 16;
                unsigned int _shfl_5 = __shfl_sync(4294967295, pub_lead_words[1], 0, 4);
                unsigned int _shfl_16_5 = __shfl_sync(4294967295, pub_lead_words[1], 1, 4);
                unsigned int _shfl_17_5 = __shfl_sync(4294967295, pub_lead_words[1], 2, 4);
                unsigned int _shfl_18_5 = __shfl_sync(4294967295, pub_lead_words[1], 3, 4);
                float lanes_3_1[2];
                unsigned int word_3_1[1];
                word_3_1[0] = _shfl_5;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) << 16);
                    (&lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) & 0xffff0000u);
                }
                float received_lanes_3_1[2];
                word_3_1[0] = _shfl_16_5;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) << 16);
                    (&received_lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_55 = 0; j_55 < 2; j_55++) {
                    lanes_3_1[j_55] = lanes_3_1[j_55] + received_lanes_3_1[j_55];
                }
                unsigned int lanes_bf16_5_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3_1[_lp*2 + 0], lanes_3_1[_lp*2+1 + 0]));
                    lanes_bf16_5_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_5_1[_pair]) << 16);
                    (&lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_5_1[_pair]) & 0xffff0000u);
                }
                word_3_1[0] = _shfl_17_5;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) << 16);
                    (&received_lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_56 = 0; j_56 < 2; j_56++) {
                    lanes_3_1[j_56] = lanes_3_1[j_56] + received_lanes_3_1[j_56];
                }
                unsigned int lanes_bf16_1_3_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3_1[_lp*2 + 0], lanes_3_1[_lp*2+1 + 0]));
                    lanes_bf16_1_3_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_3_1[_pair]) << 16);
                    (&lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_3_1[_pair]) & 0xffff0000u);
                }
                word_3_1[0] = _shfl_18_5;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) << 16);
                    (&received_lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_3_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_57 = 0; j_57 < 2; j_57++) {
                    lanes_3_1[j_57] = lanes_3_1[j_57] + received_lanes_3_1[j_57];
                }
                unsigned int lanes_bf16_2_3_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3_1[_lp*2 + 0], lanes_3_1[_lp*2+1 + 0]));
                    lanes_bf16_2_3_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_3_1[_pair]) << 16);
                    (&lanes_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_3_1[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_39[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_3_1[_lp*2 + 0], lanes_3_1[_lp*2+1 + 0]));
                    scaled_bf16_39[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_3_1 = scaled_bf16_39[0] & 65535;
                unsigned int high_3_1 = scaled_bf16_39[0] >> 16;
                low_3_1 = ((low_3_1 == 65535) ? 65534 : low_3_1);
                high_3_1 = ((high_3_1 == 65535) ? 65534 : high_3_1);
                low_3_1 = ((low_3_1 == 32768) ? 65535 : low_3_1);
                high_3_1 = ((high_3_1 == 32768) ? 65535 : high_3_1);
                ag_words_1[1] = low_3_1 | high_3_1 << 16;
                unsigned int _shfl_6 = __shfl_sync(4294967295, pub_lead_words[2], 0, 4);
                unsigned int _shfl_16_6 = __shfl_sync(4294967295, pub_lead_words[2], 1, 4);
                unsigned int _shfl_17_6 = __shfl_sync(4294967295, pub_lead_words[2], 2, 4);
                unsigned int _shfl_18_6 = __shfl_sync(4294967295, pub_lead_words[2], 3, 4);
                float lanes_4[2];
                unsigned int word_4_1[1];
                word_4_1[0] = _shfl_6;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) << 16);
                    (&lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) & 0xffff0000u);
                }
                float received_lanes_4[2];
                word_4_1[0] = _shfl_16_6;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) << 16);
                    (&received_lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_58 = 0; j_58 < 2; j_58++) {
                    lanes_4[j_58] = lanes_4[j_58] + received_lanes_4[j_58];
                }
                unsigned int lanes_bf16_6[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_4[_lp*2 + 0], lanes_4[_lp*2+1 + 0]));
                    lanes_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_6[_pair]) << 16);
                    (&lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_6[_pair]) & 0xffff0000u);
                }
                word_4_1[0] = _shfl_17_6;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) << 16);
                    (&received_lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_59 = 0; j_59 < 2; j_59++) {
                    lanes_4[j_59] = lanes_4[j_59] + received_lanes_4[j_59];
                }
                unsigned int lanes_bf16_1_4[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_4[_lp*2 + 0], lanes_4[_lp*2+1 + 0]));
                    lanes_bf16_1_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_4[_pair]) << 16);
                    (&lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_4[_pair]) & 0xffff0000u);
                }
                word_4_1[0] = _shfl_18_6;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) << 16);
                    (&received_lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_4_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_60 = 0; j_60 < 2; j_60++) {
                    lanes_4[j_60] = lanes_4[j_60] + received_lanes_4[j_60];
                }
                unsigned int lanes_bf16_2_4[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_4[_lp*2 + 0], lanes_4[_lp*2+1 + 0]));
                    lanes_bf16_2_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_4[_pair]) << 16);
                    (&lanes_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_4[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_40[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_4[_lp*2 + 0], lanes_4[_lp*2+1 + 0]));
                    scaled_bf16_40[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_4 = scaled_bf16_40[0] & 65535;
                unsigned int high_4 = scaled_bf16_40[0] >> 16;
                low_4 = ((low_4 == 65535) ? 65534 : low_4);
                high_4 = ((high_4 == 65535) ? 65534 : high_4);
                low_4 = ((low_4 == 32768) ? 65535 : low_4);
                high_4 = ((high_4 == 32768) ? 65535 : high_4);
                ag_words_1[2] = low_4 | high_4 << 16;
                unsigned int _shfl_7 = __shfl_sync(4294967295, pub_lead_words[3], 0, 4);
                unsigned int _shfl_16_7 = __shfl_sync(4294967295, pub_lead_words[3], 1, 4);
                unsigned int _shfl_17_7 = __shfl_sync(4294967295, pub_lead_words[3], 2, 4);
                unsigned int _shfl_18_7 = __shfl_sync(4294967295, pub_lead_words[3], 3, 4);
                float lanes_5[2];
                unsigned int word_5_1[1];
                word_5_1[0] = _shfl_7;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) << 16);
                    (&lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) & 0xffff0000u);
                }
                float received_lanes_5[2];
                word_5_1[0] = _shfl_16_7;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) << 16);
                    (&received_lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_61 = 0; j_61 < 2; j_61++) {
                    lanes_5[j_61] = lanes_5[j_61] + received_lanes_5[j_61];
                }
                unsigned int lanes_bf16_7[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_5[_lp*2 + 0], lanes_5[_lp*2+1 + 0]));
                    lanes_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7[_pair]) << 16);
                    (&lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_7[_pair]) & 0xffff0000u);
                }
                word_5_1[0] = _shfl_17_7;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) << 16);
                    (&received_lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_62 = 0; j_62 < 2; j_62++) {
                    lanes_5[j_62] = lanes_5[j_62] + received_lanes_5[j_62];
                }
                unsigned int lanes_bf16_1_5[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_5[_lp*2 + 0], lanes_5[_lp*2+1 + 0]));
                    lanes_bf16_1_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_5[_pair]) << 16);
                    (&lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_5[_pair]) & 0xffff0000u);
                }
                word_5_1[0] = _shfl_18_7;
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&received_lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) << 16);
                    (&received_lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_5_1[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_63 = 0; j_63 < 2; j_63++) {
                    lanes_5[j_63] = lanes_5[j_63] + received_lanes_5[j_63];
                }
                unsigned int lanes_bf16_2_5[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_5[_lp*2 + 0], lanes_5[_lp*2+1 + 0]));
                    lanes_bf16_2_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&lanes_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_5[_pair]) << 16);
                    (&lanes_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_5[_pair]) & 0xffff0000u);
                }
                unsigned int scaled_bf16_41[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_5[_lp*2 + 0], lanes_5[_lp*2+1 + 0]));
                    scaled_bf16_41[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int low_5 = scaled_bf16_41[0] & 65535;
                unsigned int high_5 = scaled_bf16_41[0] >> 16;
                low_5 = ((low_5 == 65535) ? 65534 : low_5);
                high_5 = ((high_5 == 65535) ? 65534 : high_5);
                low_5 = ((low_5 == 32768) ? 65535 : low_5);
                high_5 = ((high_5 == 32768) ? 65535 : high_5);
                ag_words_1[3] = low_5 | high_5 << 16;
                {
                    auto _gst_48 = pub_rs_base + ((long long)data_epoch * comm_stride_elems + (long long)tokens * 7168 + (long long)(pub_lead_token * 896 + rank * 224 + pub_rs_item) * 8);
                    __builtin_assume(__isGlobal(_gst_48));
                    reinterpret_cast<int4*>(_gst_48)[0] = reinterpret_cast<int4*>(ag_words_1)[0];
                }
            }
        }
        int elem_3 = token * 7168 + cluster_thread * 8;
        float acc_0[8];
        unsigned int packed_acc_0[4];
        #pragma unroll
        for (int j_64 = 0; j_64 < 8; j_64++) {
            acc_0[j_64] = 0.0f;
        }
        #pragma unroll
        for (int pair_38 = 0; pair_38 < 4; pair_38++) {
            packed_acc_0[pair_38] = 0;
        }
        int block_taken_0 = 0;
        if (active_experts == 8) {
            long long expert_elem0_3 = (long long)0 * 7168 + (long long)elem_3;
            float vec_load[8];
            {
                const uint4* _vptr_49 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem0_3 + 0);
                uint4 _vld_49[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_49[_blk].x), "=r"(_vld_49[_blk].y), "=r"(_vld_49[_blk].z), "=r"(_vld_49[_blk].w) : "l"((const void*)(_vptr_49 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                        (&vec_load[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale0_3 = expert_scales[token];
            long long expert_elem1_3 = (long long)tokens * 7168 + (long long)elem_3;
            float vec_load_1[8];
            {
                const uint4* _vptr_50 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem1_3 + 0);
                uint4 _vld_50[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_50[_blk].x), "=r"(_vld_50[_blk].y), "=r"(_vld_50[_blk].z), "=r"(_vld_50[_blk].w) : "l"((const void*)(_vptr_50 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_50 = reinterpret_cast<uint32_t*>(&_vld_50[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) << 16);
                        (&vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale1_3 = expert_scales[tokens + token];
            long long expert_elem2_3 = (long long)(2 * tokens) * 7168 + (long long)elem_3;
            float vec_load_2[8];
            {
                const uint4* _vptr_51 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem2_3 + 0);
                uint4 _vld_51[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_51[_blk].x), "=r"(_vld_51[_blk].y), "=r"(_vld_51[_blk].z), "=r"(_vld_51[_blk].w) : "l"((const void*)(_vptr_51 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
                        (&vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale2_3 = expert_scales[2 * tokens + token];
            long long expert_elem3_3 = (long long)(3 * tokens) * 7168 + (long long)elem_3;
            float vec_load_3[8];
            {
                const uint4* _vptr_52 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem3_3 + 0);
                uint4 _vld_52[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_52[_blk].x), "=r"(_vld_52[_blk].y), "=r"(_vld_52[_blk].z), "=r"(_vld_52[_blk].w) : "l"((const void*)(_vptr_52 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_52 = reinterpret_cast<uint32_t*>(&_vld_52[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) << 16);
                        (&vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale3_3 = expert_scales[3 * tokens + token];
            long long expert_elem4_3 = (long long)(4 * tokens) * 7168 + (long long)elem_3;
            float vec_load_4[8];
            {
                const uint4* _vptr_53 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem4_3 + 0);
                uint4 _vld_53[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_53[_blk].x), "=r"(_vld_53[_blk].y), "=r"(_vld_53[_blk].z), "=r"(_vld_53[_blk].w) : "l"((const void*)(_vptr_53 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_53 = reinterpret_cast<uint32_t*>(&_vld_53[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) << 16);
                        (&vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale4_3 = expert_scales[4 * tokens + token];
            long long expert_elem5_3 = (long long)(5 * tokens) * 7168 + (long long)elem_3;
            float vec_load_5[8];
            {
                const uint4* _vptr_54 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem5_3 + 0);
                uint4 _vld_54[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_54[_blk].x), "=r"(_vld_54[_blk].y), "=r"(_vld_54[_blk].z), "=r"(_vld_54[_blk].w) : "l"((const void*)(_vptr_54 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_54 = reinterpret_cast<uint32_t*>(&_vld_54[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) << 16);
                        (&vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale5_3 = expert_scales[5 * tokens + token];
            long long expert_elem6_3 = (long long)(6 * tokens) * 7168 + (long long)elem_3;
            float vec_load_6[8];
            {
                const uint4* _vptr_55 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem6_3 + 0);
                uint4 _vld_55[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_55[_blk].x), "=r"(_vld_55[_blk].y), "=r"(_vld_55[_blk].z), "=r"(_vld_55[_blk].w) : "l"((const void*)(_vptr_55 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                        (&vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale6_3 = expert_scales[6 * tokens + token];
            long long expert_elem7_3 = (long long)(7 * tokens) * 7168 + (long long)elem_3;
            float vec_load_7[8];
            {
                const uint4* _vptr_56 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem7_3 + 0);
                uint4 _vld_56[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_56[_blk].x), "=r"(_vld_56[_blk].y), "=r"(_vld_56[_blk].z), "=r"(_vld_56[_blk].w) : "l"((const void*)(_vptr_56 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                        (&vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale7_3 = expert_scales[7 * tokens + token];
            float scaled_34[8];
            #pragma unroll
            for (int j_65 = 0; j_65 < 8; j_65++) {
                scaled_34[j_65] = vec_load[j_65] * expert_scale0_3;
            }
            unsigned int scaled_bf16_42[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_34[_lp*2 + 0], scaled_34[_lp*2+1 + 0]));
                scaled_bf16_42[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_39 = 0; pair_39 < 4; pair_39++) {
                uint32_t bf16x2_add_37;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_37) : "r"(packed_acc_0[pair_39]), "r"(scaled_bf16_42[pair_39]));
                packed_acc_0[pair_39] = bf16x2_add_37;
            }
            float scaled_33_1[8];
            #pragma unroll
            for (int j_66 = 0; j_66 < 8; j_66++) {
                scaled_33_1[j_66] = vec_load_1[j_66] * expert_scale1_3;
            }
            unsigned int scaled_bf16_43[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_33_1[_lp*2 + 0], scaled_33_1[_lp*2+1 + 0]));
                scaled_bf16_43[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_40 = 0; pair_40 < 4; pair_40++) {
                uint32_t bf16x2_add_38;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_38) : "r"(packed_acc_0[pair_40]), "r"(scaled_bf16_43[pair_40]));
                packed_acc_0[pair_40] = bf16x2_add_38;
            }
            float scaled_34_1[8];
            #pragma unroll
            for (int j_67 = 0; j_67 < 8; j_67++) {
                scaled_34_1[j_67] = vec_load_2[j_67] * expert_scale2_3;
            }
            unsigned int scaled_bf16_44[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_34_1[_lp*2 + 0], scaled_34_1[_lp*2+1 + 0]));
                scaled_bf16_44[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_41 = 0; pair_41 < 4; pair_41++) {
                uint32_t bf16x2_add_39;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_39) : "r"(packed_acc_0[pair_41]), "r"(scaled_bf16_44[pair_41]));
                packed_acc_0[pair_41] = bf16x2_add_39;
            }
            float scaled_35[8];
            #pragma unroll
            for (int j_68 = 0; j_68 < 8; j_68++) {
                scaled_35[j_68] = vec_load_3[j_68] * expert_scale3_3;
            }
            unsigned int scaled_bf16_45[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_35[_lp*2 + 0], scaled_35[_lp*2+1 + 0]));
                scaled_bf16_45[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_42 = 0; pair_42 < 4; pair_42++) {
                uint32_t bf16x2_add_40;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_40) : "r"(packed_acc_0[pair_42]), "r"(scaled_bf16_45[pair_42]));
                packed_acc_0[pair_42] = bf16x2_add_40;
            }
            float scaled_36[8];
            #pragma unroll
            for (int j_69 = 0; j_69 < 8; j_69++) {
                scaled_36[j_69] = vec_load_4[j_69] * expert_scale4_3;
            }
            unsigned int scaled_bf16_46[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_36[_lp*2 + 0], scaled_36[_lp*2+1 + 0]));
                scaled_bf16_46[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_43 = 0; pair_43 < 4; pair_43++) {
                uint32_t bf16x2_add_41;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_41) : "r"(packed_acc_0[pair_43]), "r"(scaled_bf16_46[pair_43]));
                packed_acc_0[pair_43] = bf16x2_add_41;
            }
            float scaled_37[8];
            #pragma unroll
            for (int j_70 = 0; j_70 < 8; j_70++) {
                scaled_37[j_70] = vec_load_5[j_70] * expert_scale5_3;
            }
            unsigned int scaled_bf16_47[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_37[_lp*2 + 0], scaled_37[_lp*2+1 + 0]));
                scaled_bf16_47[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_44 = 0; pair_44 < 4; pair_44++) {
                uint32_t bf16x2_add_42;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_42) : "r"(packed_acc_0[pair_44]), "r"(scaled_bf16_47[pair_44]));
                packed_acc_0[pair_44] = bf16x2_add_42;
            }
            float scaled_38[8];
            #pragma unroll
            for (int j_71 = 0; j_71 < 8; j_71++) {
                scaled_38[j_71] = vec_load_6[j_71] * expert_scale6_3;
            }
            unsigned int scaled_bf16_48[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_38[_lp*2 + 0], scaled_38[_lp*2+1 + 0]));
                scaled_bf16_48[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_45 = 0; pair_45 < 4; pair_45++) {
                uint32_t bf16x2_add_43;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_43) : "r"(packed_acc_0[pair_45]), "r"(scaled_bf16_48[pair_45]));
                packed_acc_0[pair_45] = bf16x2_add_43;
            }
            float scaled_39[8];
            #pragma unroll
            for (int j_72 = 0; j_72 < 8; j_72++) {
                scaled_39[j_72] = vec_load_7[j_72] * expert_scale7_3;
            }
            unsigned int scaled_bf16_49[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_39[_lp*2 + 0], scaled_39[_lp*2+1 + 0]));
                scaled_bf16_49[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_46 = 0; pair_46 < 4; pair_46++) {
                uint32_t bf16x2_add_44;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_44) : "r"(packed_acc_0[pair_46]), "r"(scaled_bf16_49[pair_46]));
                packed_acc_0[pair_46] = bf16x2_add_44;
            }
            block_taken_0 = 1;
        }
        if (active_experts == 12) {
            long long expert_elem0_4 = (long long)0 * 7168 + (long long)elem_3;
            float vec_load_8[8];
            {
                const uint4* _vptr_57 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem0_4 + 0);
                uint4 _vld_57[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_57[_blk].x), "=r"(_vld_57[_blk].y), "=r"(_vld_57[_blk].z), "=r"(_vld_57[_blk].w) : "l"((const void*)(_vptr_57 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                        (&vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale0_4 = expert_scales[token];
            long long expert_elem1_4 = (long long)tokens * 7168 + (long long)elem_3;
            float vec_load_1_1[8];
            {
                const uint4* _vptr_58 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem1_4 + 0);
                uint4 _vld_58[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_58[_blk].x), "=r"(_vld_58[_blk].y), "=r"(_vld_58[_blk].z), "=r"(_vld_58[_blk].w) : "l"((const void*)(_vptr_58 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                        (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale1_4 = expert_scales[tokens + token];
            long long expert_elem2_4 = (long long)(2 * tokens) * 7168 + (long long)elem_3;
            float vec_load_2_1[8];
            {
                const uint4* _vptr_59 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem2_4 + 0);
                uint4 _vld_59[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_59[_blk].x), "=r"(_vld_59[_blk].y), "=r"(_vld_59[_blk].z), "=r"(_vld_59[_blk].w) : "l"((const void*)(_vptr_59 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                        (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale2_4 = expert_scales[2 * tokens + token];
            long long expert_elem3_4 = (long long)(3 * tokens) * 7168 + (long long)elem_3;
            float vec_load_3_1[8];
            {
                const uint4* _vptr_60 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem3_4 + 0);
                uint4 _vld_60[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_60[_blk].x), "=r"(_vld_60[_blk].y), "=r"(_vld_60[_blk].z), "=r"(_vld_60[_blk].w) : "l"((const void*)(_vptr_60 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_60 = reinterpret_cast<uint32_t*>(&_vld_60[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) << 16);
                        (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale3_4 = expert_scales[3 * tokens + token];
            long long expert_elem4_4 = (long long)(4 * tokens) * 7168 + (long long)elem_3;
            float vec_load_4_1[8];
            {
                const uint4* _vptr_61 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem4_4 + 0);
                uint4 _vld_61[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_61[_blk].x), "=r"(_vld_61[_blk].y), "=r"(_vld_61[_blk].z), "=r"(_vld_61[_blk].w) : "l"((const void*)(_vptr_61 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_61 = reinterpret_cast<uint32_t*>(&_vld_61[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) << 16);
                        (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale4_4 = expert_scales[4 * tokens + token];
            long long expert_elem5_4 = (long long)(5 * tokens) * 7168 + (long long)elem_3;
            float vec_load_5_1[8];
            {
                const uint4* _vptr_62 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem5_4 + 0);
                uint4 _vld_62[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_62[_blk].x), "=r"(_vld_62[_blk].y), "=r"(_vld_62[_blk].z), "=r"(_vld_62[_blk].w) : "l"((const void*)(_vptr_62 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_62 = reinterpret_cast<uint32_t*>(&_vld_62[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) << 16);
                        (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale5_4 = expert_scales[5 * tokens + token];
            long long expert_elem6_4 = (long long)(6 * tokens) * 7168 + (long long)elem_3;
            float vec_load_6_1[8];
            {
                const uint4* _vptr_63 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem6_4 + 0);
                uint4 _vld_63[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_63[_blk].x), "=r"(_vld_63[_blk].y), "=r"(_vld_63[_blk].z), "=r"(_vld_63[_blk].w) : "l"((const void*)(_vptr_63 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_63 = reinterpret_cast<uint32_t*>(&_vld_63[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) << 16);
                        (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale6_4 = expert_scales[6 * tokens + token];
            long long expert_elem7_4 = (long long)(7 * tokens) * 7168 + (long long)elem_3;
            float vec_load_7_1[8];
            {
                const uint4* _vptr_64 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem7_4 + 0);
                uint4 _vld_64[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_64[_blk].x), "=r"(_vld_64[_blk].y), "=r"(_vld_64[_blk].z), "=r"(_vld_64[_blk].w) : "l"((const void*)(_vptr_64 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                        (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale7_4 = expert_scales[7 * tokens + token];
            long long expert_elem8_2 = (long long)(8 * tokens) * 7168 + (long long)elem_3;
            float vec_load_8_1[8];
            {
                const uint4* _vptr_65 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem8_2 + 0);
                uint4 _vld_65[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_65[_blk].x), "=r"(_vld_65[_blk].y), "=r"(_vld_65[_blk].z), "=r"(_vld_65[_blk].w) : "l"((const void*)(_vptr_65 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_8_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                        (&vec_load_8_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale8_2 = expert_scales[8 * tokens + token];
            long long expert_elem9_2 = (long long)(9 * tokens) * 7168 + (long long)elem_3;
            float vec_load_9[8];
            {
                const uint4* _vptr_66 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem9_2 + 0);
                uint4 _vld_66[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_66[_blk].x), "=r"(_vld_66[_blk].y), "=r"(_vld_66[_blk].z), "=r"(_vld_66[_blk].w) : "l"((const void*)(_vptr_66 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                        (&vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale9_2 = expert_scales[9 * tokens + token];
            long long expert_elem10_2 = (long long)(10 * tokens) * 7168 + (long long)elem_3;
            float vec_load_10[8];
            {
                const uint4* _vptr_67 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem10_2 + 0);
                uint4 _vld_67[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_67[_blk].x), "=r"(_vld_67[_blk].y), "=r"(_vld_67[_blk].z), "=r"(_vld_67[_blk].w) : "l"((const void*)(_vptr_67 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                        (&vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale10_2 = expert_scales[10 * tokens + token];
            long long expert_elem11_2 = (long long)(11 * tokens) * 7168 + (long long)elem_3;
            float vec_load_11[8];
            {
                const uint4* _vptr_68 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem11_2 + 0);
                uint4 _vld_68[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_68[_blk].x), "=r"(_vld_68[_blk].y), "=r"(_vld_68[_blk].z), "=r"(_vld_68[_blk].w) : "l"((const void*)(_vptr_68 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_68 = reinterpret_cast<uint32_t*>(&_vld_68[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_68[_pair]) << 16);
                        (&vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_68[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale11_2 = expert_scales[11 * tokens + token];
            float scaled_40[8];
            #pragma unroll
            for (int j_73 = 0; j_73 < 8; j_73++) {
                scaled_40[j_73] = vec_load_8[j_73] * expert_scale0_4;
            }
            unsigned int scaled_bf16_50[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_40[_lp*2 + 0], scaled_40[_lp*2+1 + 0]));
                scaled_bf16_50[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_47 = 0; pair_47 < 4; pair_47++) {
                uint32_t bf16x2_add_45;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_45) : "r"(packed_acc_0[pair_47]), "r"(scaled_bf16_50[pair_47]));
                packed_acc_0[pair_47] = bf16x2_add_45;
            }
            float scaled_40_1[8];
            #pragma unroll
            for (int j_74 = 0; j_74 < 8; j_74++) {
                scaled_40_1[j_74] = vec_load_1_1[j_74] * expert_scale1_4;
            }
            unsigned int scaled_bf16_51[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_40_1[_lp*2 + 0], scaled_40_1[_lp*2+1 + 0]));
                scaled_bf16_51[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_48 = 0; pair_48 < 4; pair_48++) {
                uint32_t bf16x2_add_46;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_46) : "r"(packed_acc_0[pair_48]), "r"(scaled_bf16_51[pair_48]));
                packed_acc_0[pair_48] = bf16x2_add_46;
            }
            float scaled_41[8];
            #pragma unroll
            for (int j_75 = 0; j_75 < 8; j_75++) {
                scaled_41[j_75] = vec_load_2_1[j_75] * expert_scale2_4;
            }
            unsigned int scaled_bf16_52[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_41[_lp*2 + 0], scaled_41[_lp*2+1 + 0]));
                scaled_bf16_52[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_49 = 0; pair_49 < 4; pair_49++) {
                uint32_t bf16x2_add_47;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_47) : "r"(packed_acc_0[pair_49]), "r"(scaled_bf16_52[pair_49]));
                packed_acc_0[pair_49] = bf16x2_add_47;
            }
            float scaled_42[8];
            #pragma unroll
            for (int j_76 = 0; j_76 < 8; j_76++) {
                scaled_42[j_76] = vec_load_3_1[j_76] * expert_scale3_4;
            }
            unsigned int scaled_bf16_53[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_42[_lp*2 + 0], scaled_42[_lp*2+1 + 0]));
                scaled_bf16_53[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_50 = 0; pair_50 < 4; pair_50++) {
                uint32_t bf16x2_add_48;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_48) : "r"(packed_acc_0[pair_50]), "r"(scaled_bf16_53[pair_50]));
                packed_acc_0[pair_50] = bf16x2_add_48;
            }
            float scaled_43[8];
            #pragma unroll
            for (int j_77 = 0; j_77 < 8; j_77++) {
                scaled_43[j_77] = vec_load_4_1[j_77] * expert_scale4_4;
            }
            unsigned int scaled_bf16_54[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_43[_lp*2 + 0], scaled_43[_lp*2+1 + 0]));
                scaled_bf16_54[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_51 = 0; pair_51 < 4; pair_51++) {
                uint32_t bf16x2_add_49;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_49) : "r"(packed_acc_0[pair_51]), "r"(scaled_bf16_54[pair_51]));
                packed_acc_0[pair_51] = bf16x2_add_49;
            }
            float scaled_44[8];
            #pragma unroll
            for (int j_78 = 0; j_78 < 8; j_78++) {
                scaled_44[j_78] = vec_load_5_1[j_78] * expert_scale5_4;
            }
            unsigned int scaled_bf16_55[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_44[_lp*2 + 0], scaled_44[_lp*2+1 + 0]));
                scaled_bf16_55[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_52 = 0; pair_52 < 4; pair_52++) {
                uint32_t bf16x2_add_50;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_50) : "r"(packed_acc_0[pair_52]), "r"(scaled_bf16_55[pair_52]));
                packed_acc_0[pair_52] = bf16x2_add_50;
            }
            float scaled_45[8];
            #pragma unroll
            for (int j_79 = 0; j_79 < 8; j_79++) {
                scaled_45[j_79] = vec_load_6_1[j_79] * expert_scale6_4;
            }
            unsigned int scaled_bf16_56[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_45[_lp*2 + 0], scaled_45[_lp*2+1 + 0]));
                scaled_bf16_56[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_53 = 0; pair_53 < 4; pair_53++) {
                uint32_t bf16x2_add_51;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_51) : "r"(packed_acc_0[pair_53]), "r"(scaled_bf16_56[pair_53]));
                packed_acc_0[pair_53] = bf16x2_add_51;
            }
            float scaled_46[8];
            #pragma unroll
            for (int j_80 = 0; j_80 < 8; j_80++) {
                scaled_46[j_80] = vec_load_7_1[j_80] * expert_scale7_4;
            }
            unsigned int scaled_bf16_57[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_46[_lp*2 + 0], scaled_46[_lp*2+1 + 0]));
                scaled_bf16_57[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_54 = 0; pair_54 < 4; pair_54++) {
                uint32_t bf16x2_add_52;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_52) : "r"(packed_acc_0[pair_54]), "r"(scaled_bf16_57[pair_54]));
                packed_acc_0[pair_54] = bf16x2_add_52;
            }
            float scaled_47[8];
            #pragma unroll
            for (int j_81 = 0; j_81 < 8; j_81++) {
                scaled_47[j_81] = vec_load_8_1[j_81] * expert_scale8_2;
            }
            unsigned int scaled_bf16_58[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_47[_lp*2 + 0], scaled_47[_lp*2+1 + 0]));
                scaled_bf16_58[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_55 = 0; pair_55 < 4; pair_55++) {
                uint32_t bf16x2_add_53;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_53) : "r"(packed_acc_0[pair_55]), "r"(scaled_bf16_58[pair_55]));
                packed_acc_0[pair_55] = bf16x2_add_53;
            }
            float scaled_48[8];
            #pragma unroll
            for (int j_82 = 0; j_82 < 8; j_82++) {
                scaled_48[j_82] = vec_load_9[j_82] * expert_scale9_2;
            }
            unsigned int scaled_bf16_59[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_48[_lp*2 + 0], scaled_48[_lp*2+1 + 0]));
                scaled_bf16_59[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_56 = 0; pair_56 < 4; pair_56++) {
                uint32_t bf16x2_add_54;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_54) : "r"(packed_acc_0[pair_56]), "r"(scaled_bf16_59[pair_56]));
                packed_acc_0[pair_56] = bf16x2_add_54;
            }
            float scaled_49[8];
            #pragma unroll
            for (int j_83 = 0; j_83 < 8; j_83++) {
                scaled_49[j_83] = vec_load_10[j_83] * expert_scale10_2;
            }
            unsigned int scaled_bf16_60[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_49[_lp*2 + 0], scaled_49[_lp*2+1 + 0]));
                scaled_bf16_60[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_57 = 0; pair_57 < 4; pair_57++) {
                uint32_t bf16x2_add_55;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_55) : "r"(packed_acc_0[pair_57]), "r"(scaled_bf16_60[pair_57]));
                packed_acc_0[pair_57] = bf16x2_add_55;
            }
            float scaled_50[8];
            #pragma unroll
            for (int j_84 = 0; j_84 < 8; j_84++) {
                scaled_50[j_84] = vec_load_11[j_84] * expert_scale11_2;
            }
            unsigned int scaled_bf16_61[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_50[_lp*2 + 0], scaled_50[_lp*2+1 + 0]));
                scaled_bf16_61[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_58 = 0; pair_58 < 4; pair_58++) {
                uint32_t bf16x2_add_56;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_56) : "r"(packed_acc_0[pair_58]), "r"(scaled_bf16_61[pair_58]));
                packed_acc_0[pair_58] = bf16x2_add_56;
            }
            block_taken_0 = 1;
        }
        if (active_experts == 16) {
            long long expert_elem0_5 = (long long)0 * 7168 + (long long)elem_3;
            float vec_load_12[8];
            {
                const uint4* _vptr_69 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem0_5 + 0);
                uint4 _vld_69[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_69[_blk].x), "=r"(_vld_69[_blk].y), "=r"(_vld_69[_blk].z), "=r"(_vld_69[_blk].w) : "l"((const void*)(_vptr_69 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_69 = reinterpret_cast<uint32_t*>(&_vld_69[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) << 16);
                        (&vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale0_5 = expert_scales[token];
            long long expert_elem1_5 = (long long)tokens * 7168 + (long long)elem_3;
            float vec_load_1_2[8];
            {
                const uint4* _vptr_70 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem1_5 + 0);
                uint4 _vld_70[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_70[_blk].x), "=r"(_vld_70[_blk].y), "=r"(_vld_70[_blk].z), "=r"(_vld_70[_blk].w) : "l"((const void*)(_vptr_70 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_70 = reinterpret_cast<uint32_t*>(&_vld_70[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_1_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_70[_pair]) << 16);
                        (&vec_load_1_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_70[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale1_5 = expert_scales[tokens + token];
            long long expert_elem2_5 = (long long)(2 * tokens) * 7168 + (long long)elem_3;
            float vec_load_2_2[8];
            {
                const uint4* _vptr_71 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem2_5 + 0);
                uint4 _vld_71[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_71[_blk].x), "=r"(_vld_71[_blk].y), "=r"(_vld_71[_blk].z), "=r"(_vld_71[_blk].w) : "l"((const void*)(_vptr_71 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_71 = reinterpret_cast<uint32_t*>(&_vld_71[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_2_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) << 16);
                        (&vec_load_2_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale2_5 = expert_scales[2 * tokens + token];
            long long expert_elem3_5 = (long long)(3 * tokens) * 7168 + (long long)elem_3;
            float vec_load_3_2[8];
            {
                const uint4* _vptr_72 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem3_5 + 0);
                uint4 _vld_72[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_72[_blk].x), "=r"(_vld_72[_blk].y), "=r"(_vld_72[_blk].z), "=r"(_vld_72[_blk].w) : "l"((const void*)(_vptr_72 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_72 = reinterpret_cast<uint32_t*>(&_vld_72[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_3_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) << 16);
                        (&vec_load_3_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_72[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale3_5 = expert_scales[3 * tokens + token];
            long long expert_elem4_5 = (long long)(4 * tokens) * 7168 + (long long)elem_3;
            float vec_load_4_2[8];
            {
                const uint4* _vptr_73 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem4_5 + 0);
                uint4 _vld_73[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_73[_blk].x), "=r"(_vld_73[_blk].y), "=r"(_vld_73[_blk].z), "=r"(_vld_73[_blk].w) : "l"((const void*)(_vptr_73 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_73 = reinterpret_cast<uint32_t*>(&_vld_73[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_4_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) << 16);
                        (&vec_load_4_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_73[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale4_5 = expert_scales[4 * tokens + token];
            long long expert_elem5_5 = (long long)(5 * tokens) * 7168 + (long long)elem_3;
            float vec_load_5_2[8];
            {
                const uint4* _vptr_74 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem5_5 + 0);
                uint4 _vld_74[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_74[_blk].x), "=r"(_vld_74[_blk].y), "=r"(_vld_74[_blk].z), "=r"(_vld_74[_blk].w) : "l"((const void*)(_vptr_74 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_74 = reinterpret_cast<uint32_t*>(&_vld_74[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_5_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) << 16);
                        (&vec_load_5_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_74[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale5_5 = expert_scales[5 * tokens + token];
            long long expert_elem6_5 = (long long)(6 * tokens) * 7168 + (long long)elem_3;
            float vec_load_6_2[8];
            {
                const uint4* _vptr_75 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem6_5 + 0);
                uint4 _vld_75[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_75[_blk].x), "=r"(_vld_75[_blk].y), "=r"(_vld_75[_blk].z), "=r"(_vld_75[_blk].w) : "l"((const void*)(_vptr_75 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_75 = reinterpret_cast<uint32_t*>(&_vld_75[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_6_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) << 16);
                        (&vec_load_6_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale6_5 = expert_scales[6 * tokens + token];
            long long expert_elem7_5 = (long long)(7 * tokens) * 7168 + (long long)elem_3;
            float vec_load_7_2[8];
            {
                const uint4* _vptr_76 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem7_5 + 0);
                uint4 _vld_76[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_76[_blk].x), "=r"(_vld_76[_blk].y), "=r"(_vld_76[_blk].z), "=r"(_vld_76[_blk].w) : "l"((const void*)(_vptr_76 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_76 = reinterpret_cast<uint32_t*>(&_vld_76[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_7_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) << 16);
                        (&vec_load_7_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale7_5 = expert_scales[7 * tokens + token];
            long long expert_elem8_3 = (long long)(8 * tokens) * 7168 + (long long)elem_3;
            float vec_load_8_2[8];
            {
                const uint4* _vptr_77 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem8_3 + 0);
                uint4 _vld_77[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_77[_blk].x), "=r"(_vld_77[_blk].y), "=r"(_vld_77[_blk].z), "=r"(_vld_77[_blk].w) : "l"((const void*)(_vptr_77 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_77 = reinterpret_cast<uint32_t*>(&_vld_77[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_8_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) << 16);
                        (&vec_load_8_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale8_3 = expert_scales[8 * tokens + token];
            long long expert_elem9_3 = (long long)(9 * tokens) * 7168 + (long long)elem_3;
            float vec_load_9_1[8];
            {
                const uint4* _vptr_78 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem9_3 + 0);
                uint4 _vld_78[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_78[_blk].x), "=r"(_vld_78[_blk].y), "=r"(_vld_78[_blk].z), "=r"(_vld_78[_blk].w) : "l"((const void*)(_vptr_78 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_9_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                        (&vec_load_9_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale9_3 = expert_scales[9 * tokens + token];
            long long expert_elem10_3 = (long long)(10 * tokens) * 7168 + (long long)elem_3;
            float vec_load_10_1[8];
            {
                const uint4* _vptr_79 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem10_3 + 0);
                uint4 _vld_79[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_79[_blk].x), "=r"(_vld_79[_blk].y), "=r"(_vld_79[_blk].z), "=r"(_vld_79[_blk].w) : "l"((const void*)(_vptr_79 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_10_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                        (&vec_load_10_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale10_3 = expert_scales[10 * tokens + token];
            long long expert_elem11_3 = (long long)(11 * tokens) * 7168 + (long long)elem_3;
            float vec_load_11_1[8];
            {
                const uint4* _vptr_80 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem11_3 + 0);
                uint4 _vld_80[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_80[_blk].x), "=r"(_vld_80[_blk].y), "=r"(_vld_80[_blk].z), "=r"(_vld_80[_blk].w) : "l"((const void*)(_vptr_80 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_11_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                        (&vec_load_11_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale11_3 = expert_scales[11 * tokens + token];
            long long expert_elem12_1 = (long long)(12 * tokens) * 7168 + (long long)elem_3;
            float vec_load_12_1[8];
            {
                const uint4* _vptr_81 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem12_1 + 0);
                uint4 _vld_81[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_81[_blk].x), "=r"(_vld_81[_blk].y), "=r"(_vld_81[_blk].z), "=r"(_vld_81[_blk].w) : "l"((const void*)(_vptr_81 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_12_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                        (&vec_load_12_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale12_1 = expert_scales[12 * tokens + token];
            long long expert_elem13_1 = (long long)(13 * tokens) * 7168 + (long long)elem_3;
            float vec_load_13[8];
            {
                const uint4* _vptr_82 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem13_1 + 0);
                uint4 _vld_82[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_82[_blk].x), "=r"(_vld_82[_blk].y), "=r"(_vld_82[_blk].z), "=r"(_vld_82[_blk].w) : "l"((const void*)(_vptr_82 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_82 = reinterpret_cast<uint32_t*>(&_vld_82[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_82[_pair]) << 16);
                        (&vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_82[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale13_1 = expert_scales[13 * tokens + token];
            long long expert_elem14_1 = (long long)(14 * tokens) * 7168 + (long long)elem_3;
            float vec_load_14[8];
            {
                const uint4* _vptr_83 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem14_1 + 0);
                uint4 _vld_83[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_83[_blk].x), "=r"(_vld_83[_blk].y), "=r"(_vld_83[_blk].z), "=r"(_vld_83[_blk].w) : "l"((const void*)(_vptr_83 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                        (&vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale14_1 = expert_scales[14 * tokens + token];
            long long expert_elem15_1 = (long long)(15 * tokens) * 7168 + (long long)elem_3;
            float vec_load_15[8];
            {
                const uint4* _vptr_84 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem15_1 + 0);
                uint4 _vld_84[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_84[_blk].x), "=r"(_vld_84[_blk].y), "=r"(_vld_84[_blk].z), "=r"(_vld_84[_blk].w) : "l"((const void*)(_vptr_84 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_84 = reinterpret_cast<uint32_t*>(&_vld_84[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) << 16);
                        (&vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_84[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale15_1 = expert_scales[15 * tokens + token];
            float scaled_51[8];
            #pragma unroll
            for (int j_85 = 0; j_85 < 8; j_85++) {
                scaled_51[j_85] = vec_load_12[j_85] * expert_scale0_5;
            }
            unsigned int scaled_bf16_62[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_51[_lp*2 + 0], scaled_51[_lp*2+1 + 0]));
                scaled_bf16_62[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_59 = 0; pair_59 < 4; pair_59++) {
                uint32_t bf16x2_add_57;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_57) : "r"(packed_acc_0[pair_59]), "r"(scaled_bf16_62[pair_59]));
                packed_acc_0[pair_59] = bf16x2_add_57;
            }
            float scaled_51_1[8];
            #pragma unroll
            for (int j_86 = 0; j_86 < 8; j_86++) {
                scaled_51_1[j_86] = vec_load_1_2[j_86] * expert_scale1_5;
            }
            unsigned int scaled_bf16_63[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_51_1[_lp*2 + 0], scaled_51_1[_lp*2+1 + 0]));
                scaled_bf16_63[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_60 = 0; pair_60 < 4; pair_60++) {
                uint32_t bf16x2_add_58;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_58) : "r"(packed_acc_0[pair_60]), "r"(scaled_bf16_63[pair_60]));
                packed_acc_0[pair_60] = bf16x2_add_58;
            }
            float scaled_52[8];
            #pragma unroll
            for (int j_87 = 0; j_87 < 8; j_87++) {
                scaled_52[j_87] = vec_load_2_2[j_87] * expert_scale2_5;
            }
            unsigned int scaled_bf16_64[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_52[_lp*2 + 0], scaled_52[_lp*2+1 + 0]));
                scaled_bf16_64[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_61 = 0; pair_61 < 4; pair_61++) {
                uint32_t bf16x2_add_59;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_59) : "r"(packed_acc_0[pair_61]), "r"(scaled_bf16_64[pair_61]));
                packed_acc_0[pair_61] = bf16x2_add_59;
            }
            float scaled_53[8];
            #pragma unroll
            for (int j_88 = 0; j_88 < 8; j_88++) {
                scaled_53[j_88] = vec_load_3_2[j_88] * expert_scale3_5;
            }
            unsigned int scaled_bf16_65[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_53[_lp*2 + 0], scaled_53[_lp*2+1 + 0]));
                scaled_bf16_65[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_62 = 0; pair_62 < 4; pair_62++) {
                uint32_t bf16x2_add_60;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_60) : "r"(packed_acc_0[pair_62]), "r"(scaled_bf16_65[pair_62]));
                packed_acc_0[pair_62] = bf16x2_add_60;
            }
            float scaled_54[8];
            #pragma unroll
            for (int j_89 = 0; j_89 < 8; j_89++) {
                scaled_54[j_89] = vec_load_4_2[j_89] * expert_scale4_5;
            }
            unsigned int scaled_bf16_66[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_54[_lp*2 + 0], scaled_54[_lp*2+1 + 0]));
                scaled_bf16_66[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_63 = 0; pair_63 < 4; pair_63++) {
                uint32_t bf16x2_add_61;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_61) : "r"(packed_acc_0[pair_63]), "r"(scaled_bf16_66[pair_63]));
                packed_acc_0[pair_63] = bf16x2_add_61;
            }
            float scaled_55[8];
            #pragma unroll
            for (int j_90 = 0; j_90 < 8; j_90++) {
                scaled_55[j_90] = vec_load_5_2[j_90] * expert_scale5_5;
            }
            unsigned int scaled_bf16_67[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_55[_lp*2 + 0], scaled_55[_lp*2+1 + 0]));
                scaled_bf16_67[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_64 = 0; pair_64 < 4; pair_64++) {
                uint32_t bf16x2_add_62;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_62) : "r"(packed_acc_0[pair_64]), "r"(scaled_bf16_67[pair_64]));
                packed_acc_0[pair_64] = bf16x2_add_62;
            }
            float scaled_56[8];
            #pragma unroll
            for (int j_91 = 0; j_91 < 8; j_91++) {
                scaled_56[j_91] = vec_load_6_2[j_91] * expert_scale6_5;
            }
            unsigned int scaled_bf16_68[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_56[_lp*2 + 0], scaled_56[_lp*2+1 + 0]));
                scaled_bf16_68[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_65 = 0; pair_65 < 4; pair_65++) {
                uint32_t bf16x2_add_63;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_63) : "r"(packed_acc_0[pair_65]), "r"(scaled_bf16_68[pair_65]));
                packed_acc_0[pair_65] = bf16x2_add_63;
            }
            float scaled_57[8];
            #pragma unroll
            for (int j_92 = 0; j_92 < 8; j_92++) {
                scaled_57[j_92] = vec_load_7_2[j_92] * expert_scale7_5;
            }
            unsigned int scaled_bf16_69[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_57[_lp*2 + 0], scaled_57[_lp*2+1 + 0]));
                scaled_bf16_69[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_66 = 0; pair_66 < 4; pair_66++) {
                uint32_t bf16x2_add_64;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_64) : "r"(packed_acc_0[pair_66]), "r"(scaled_bf16_69[pair_66]));
                packed_acc_0[pair_66] = bf16x2_add_64;
            }
            float scaled_58[8];
            #pragma unroll
            for (int j_93 = 0; j_93 < 8; j_93++) {
                scaled_58[j_93] = vec_load_8_2[j_93] * expert_scale8_3;
            }
            unsigned int scaled_bf16_70[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_58[_lp*2 + 0], scaled_58[_lp*2+1 + 0]));
                scaled_bf16_70[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_67 = 0; pair_67 < 4; pair_67++) {
                uint32_t bf16x2_add_65;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_65) : "r"(packed_acc_0[pair_67]), "r"(scaled_bf16_70[pair_67]));
                packed_acc_0[pair_67] = bf16x2_add_65;
            }
            float scaled_59[8];
            #pragma unroll
            for (int j_94 = 0; j_94 < 8; j_94++) {
                scaled_59[j_94] = vec_load_9_1[j_94] * expert_scale9_3;
            }
            unsigned int scaled_bf16_71[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_59[_lp*2 + 0], scaled_59[_lp*2+1 + 0]));
                scaled_bf16_71[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_68 = 0; pair_68 < 4; pair_68++) {
                uint32_t bf16x2_add_66;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_66) : "r"(packed_acc_0[pair_68]), "r"(scaled_bf16_71[pair_68]));
                packed_acc_0[pair_68] = bf16x2_add_66;
            }
            float scaled_60[8];
            #pragma unroll
            for (int j_95 = 0; j_95 < 8; j_95++) {
                scaled_60[j_95] = vec_load_10_1[j_95] * expert_scale10_3;
            }
            unsigned int scaled_bf16_72[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_60[_lp*2 + 0], scaled_60[_lp*2+1 + 0]));
                scaled_bf16_72[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_69 = 0; pair_69 < 4; pair_69++) {
                uint32_t bf16x2_add_67;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_67) : "r"(packed_acc_0[pair_69]), "r"(scaled_bf16_72[pair_69]));
                packed_acc_0[pair_69] = bf16x2_add_67;
            }
            float scaled_61[8];
            #pragma unroll
            for (int j_96 = 0; j_96 < 8; j_96++) {
                scaled_61[j_96] = vec_load_11_1[j_96] * expert_scale11_3;
            }
            unsigned int scaled_bf16_73[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_61[_lp*2 + 0], scaled_61[_lp*2+1 + 0]));
                scaled_bf16_73[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_70 = 0; pair_70 < 4; pair_70++) {
                uint32_t bf16x2_add_68;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_68) : "r"(packed_acc_0[pair_70]), "r"(scaled_bf16_73[pair_70]));
                packed_acc_0[pair_70] = bf16x2_add_68;
            }
            float scaled_62[8];
            #pragma unroll
            for (int j_97 = 0; j_97 < 8; j_97++) {
                scaled_62[j_97] = vec_load_12_1[j_97] * expert_scale12_1;
            }
            unsigned int scaled_bf16_74[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_62[_lp*2 + 0], scaled_62[_lp*2+1 + 0]));
                scaled_bf16_74[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_71 = 0; pair_71 < 4; pair_71++) {
                uint32_t bf16x2_add_69;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_69) : "r"(packed_acc_0[pair_71]), "r"(scaled_bf16_74[pair_71]));
                packed_acc_0[pair_71] = bf16x2_add_69;
            }
            float scaled_63[8];
            #pragma unroll
            for (int j_98 = 0; j_98 < 8; j_98++) {
                scaled_63[j_98] = vec_load_13[j_98] * expert_scale13_1;
            }
            unsigned int scaled_bf16_75[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_63[_lp*2 + 0], scaled_63[_lp*2+1 + 0]));
                scaled_bf16_75[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_72 = 0; pair_72 < 4; pair_72++) {
                uint32_t bf16x2_add_70;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_70) : "r"(packed_acc_0[pair_72]), "r"(scaled_bf16_75[pair_72]));
                packed_acc_0[pair_72] = bf16x2_add_70;
            }
            float scaled_64[8];
            #pragma unroll
            for (int j_99 = 0; j_99 < 8; j_99++) {
                scaled_64[j_99] = vec_load_14[j_99] * expert_scale14_1;
            }
            unsigned int scaled_bf16_76[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_64[_lp*2 + 0], scaled_64[_lp*2+1 + 0]));
                scaled_bf16_76[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_73 = 0; pair_73 < 4; pair_73++) {
                uint32_t bf16x2_add_71;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_71) : "r"(packed_acc_0[pair_73]), "r"(scaled_bf16_76[pair_73]));
                packed_acc_0[pair_73] = bf16x2_add_71;
            }
            float scaled_65[8];
            #pragma unroll
            for (int j_100 = 0; j_100 < 8; j_100++) {
                scaled_65[j_100] = vec_load_15[j_100] * expert_scale15_1;
            }
            unsigned int scaled_bf16_77[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_65[_lp*2 + 0], scaled_65[_lp*2+1 + 0]));
                scaled_bf16_77[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_74 = 0; pair_74 < 4; pair_74++) {
                uint32_t bf16x2_add_72;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_72) : "r"(packed_acc_0[pair_74]), "r"(scaled_bf16_77[pair_74]));
                packed_acc_0[pair_74] = bf16x2_add_72;
            }
            block_taken_0 = 1;
        }
        int expert_loop_end_0 = ((block_taken_0 == 1) ? 0 : active_experts);
        #pragma unroll 4
        for (int expert_1 = 0; expert_1 < expert_loop_end_0; expert_1++) {
            long long expert_elem_1 = (long long)expert_1 * (long long)tokens * 7168 + (long long)elem_3;
            float vec_load_16[8];
            {
                const uint4* _vptr_85 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem_1 + 0);
                uint4 _vld_85[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                        : "=r"(_vld_85[_blk].x), "=r"(_vld_85[_blk].y), "=r"(_vld_85[_blk].z), "=r"(_vld_85[_blk].w) : "l"((const void*)(_vptr_85 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    uint32_t* _vpairs_85 = reinterpret_cast<uint32_t*>(&_vld_85[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) << 16);
                        (&vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) & 0xffff0000u);
                    }
                }
            }
            float expert_scale_1 = expert_scales[expert_1 * tokens + token];
            float scaled_66[8];
            #pragma unroll
            for (int j_101 = 0; j_101 < 8; j_101++) {
                scaled_66[j_101] = vec_load_16[j_101] * expert_scale_1;
            }
            unsigned int scaled_bf16_78[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_66[_lp*2 + 0], scaled_66[_lp*2+1 + 0]));
                scaled_bf16_78[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int pair_75 = 0; pair_75 < 4; pair_75++) {
                uint32_t bf16x2_add_73;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_add_73) : "r"(packed_acc_0[pair_75]), "r"(scaled_bf16_78[pair_75]));
                packed_acc_0[pair_75] = bf16x2_add_73;
            }
        }
        float vec_load_17[8];
        {
            const uint4* _vptr_86 = reinterpret_cast<const uint4*>(token_input + elem_3 + 0);
            uint4 _vld_86[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_86[_blk].x), "=r"(_vld_86[_blk].y), "=r"(_vld_86[_blk].z), "=r"(_vld_86[_blk].w) : "l"((const void*)(_vptr_86 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_86 = reinterpret_cast<uint32_t*>(&_vld_86[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_86[_pair]) << 16);
                    (&vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_86[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&acc_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_acc_0[_pair]) << 16);
            (&acc_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_acc_0[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_102 = 0; j_102 < 8; j_102++) {
            acc_0[j_102] = acc_0[j_102] + vec_load_17[j_102];
        }
        unsigned int scaled_bf16_79[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
            scaled_bf16_79[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&acc_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_79[_pair]) << 16);
            (&acc_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_79[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_103 = 0; j_103 < 8; j_103++) {
            acc_0[j_103] = ((acc_0[j_103] == 0.0f) ? 0.0f : acc_0[j_103]);
        }
        int owner_1 = cluster_thread / 224;
        long long slot_1 = (long long)data_epoch * comm_stride_elems + (long long)rank * (long long)rs_items * 8 + (long long)(token * 224 + (cluster_thread - owner_1 * 224)) * 8;
        unsigned int scaled_bf16_80[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
            scaled_bf16_80[_lp] = *(uint32_t*)&_bf2;
        }
        if (owner_1 == 0) {
            {
                auto _gst_87 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[8]) + slot_1;
                __builtin_assume(__isGlobal(_gst_87));
                reinterpret_cast<int4*>(_gst_87)[0] = reinterpret_cast<int4*>(scaled_bf16_80)[0];
            }
        }
        if (owner_1 == 1) {
            {
                auto _gst_88 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[9]) + slot_1;
                __builtin_assume(__isGlobal(_gst_88));
                reinterpret_cast<int4*>(_gst_88)[0] = reinterpret_cast<int4*>(scaled_bf16_80)[0];
            }
        }
        if (owner_1 == 2) {
            {
                auto _gst_89 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[10]) + slot_1;
                __builtin_assume(__isGlobal(_gst_89));
                reinterpret_cast<int4*>(_gst_89)[0] = reinterpret_cast<int4*>(scaled_bf16_80)[0];
            }
        }
        if (owner_1 == 3) {
            {
                auto _gst_90 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[11]) + slot_1;
                __builtin_assume(__isGlobal(_gst_90));
                reinterpret_cast<int4*>(_gst_90)[0] = reinterpret_cast<int4*>(scaled_bf16_80)[0];
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
    int rs_src = cluster_thread % 4;
    int rs_item = cluster_thread / 4;
    int rs_slot_items = tokens * 224;
    __nv_bfloat16* rs_source_base = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[8 + rs_src]);
    if ((int)cluster_id < tokens) {
        if ((int)cluster_id + (int)num_clusters >= tokens) {
            unsigned int first_rs_words[4];
            uint32_t sysv_ld_1[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_1[0]), "=r"(sysv_ld_1[1]), "=r"(sysv_ld_1[2]), "=r"(sysv_ld_1[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(rs_src * rs_slot_items + (int)cluster_id * 224 + rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word_8 = 0; word_8 < 4; word_8++) {
                first_rs_words[word_8] = sysv_ld_1[word_8];
            }
            int pending_2 = 0;
            #pragma unroll
            for (int word_9 = 0; word_9 < 4; word_9++) {
                pending_2 = (((first_rs_words[word_9] & 65535) == 32768) ? 1 : pending_2);
                pending_2 = ((first_rs_words[word_9] >> 16 == 32768) ? 1 : pending_2);
            }
            if (pending_2 == 1) {
                uint32_t sysv_poll_group_2[4];
                uint32_t _sysv_pending_91 = 1u;
                do {
                    if (_sysv_pending_91 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_2[0]), "=r"(sysv_poll_group_2[1]), "=r"(sysv_poll_group_2[2]), "=r"(sysv_poll_group_2[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(rs_src * rs_slot_items + (int)cluster_id * 224 + rs_item) * 8)) : "memory");
                    }
                    if (_sysv_pending_91 & 1u) {
                        if (!((((sysv_poll_group_2[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_2[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_91 &= ~1u;
                    }
                } while (_sysv_pending_91 != 0u);
                #pragma unroll
                for (int word_10 = 0; word_10 < 4; word_10++) {
                    first_rs_words[word_10] = sysv_poll_group_2[word_10];
                }
            }
            unsigned int ag_words_2[4];
            unsigned int _shfl_8 = __shfl_sync(4294967295, first_rs_words[0], 0, 4);
            unsigned int _shfl_16_8 = __shfl_sync(4294967295, first_rs_words[0], 1, 4);
            unsigned int _shfl_17_8 = __shfl_sync(4294967295, first_rs_words[0], 2, 4);
            unsigned int _shfl_18_8 = __shfl_sync(4294967295, first_rs_words[0], 3, 4);
            float lanes_6[2];
            unsigned int word_11[1];
            word_11[0] = _shfl_8;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) << 16);
                (&lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) & 0xffff0000u);
            }
            float received_lanes_6[2];
            word_11[0] = _shfl_16_8;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) << 16);
                (&received_lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_104 = 0; j_104 < 2; j_104++) {
                lanes_6[j_104] = lanes_6[j_104] + received_lanes_6[j_104];
            }
            unsigned int lanes_bf16_8[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6[_lp*2 + 0], lanes_6[_lp*2+1 + 0]));
                lanes_bf16_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_8[_pair]) << 16);
                (&lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_8[_pair]) & 0xffff0000u);
            }
            word_11[0] = _shfl_17_8;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) << 16);
                (&received_lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_105 = 0; j_105 < 2; j_105++) {
                lanes_6[j_105] = lanes_6[j_105] + received_lanes_6[j_105];
            }
            unsigned int lanes_bf16_1_6[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6[_lp*2 + 0], lanes_6[_lp*2+1 + 0]));
                lanes_bf16_1_6[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_6[_pair]) << 16);
                (&lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_6[_pair]) & 0xffff0000u);
            }
            word_11[0] = _shfl_18_8;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) << 16);
                (&received_lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_106 = 0; j_106 < 2; j_106++) {
                lanes_6[j_106] = lanes_6[j_106] + received_lanes_6[j_106];
            }
            unsigned int lanes_bf16_2_6[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6[_lp*2 + 0], lanes_6[_lp*2+1 + 0]));
                lanes_bf16_2_6[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_6[_pair]) << 16);
                (&lanes_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_6[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_81[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6[_lp*2 + 0], lanes_6[_lp*2+1 + 0]));
                scaled_bf16_81[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_6 = scaled_bf16_81[0] & 65535;
            unsigned int high_6 = scaled_bf16_81[0] >> 16;
            low_6 = ((low_6 == 65535) ? 65534 : low_6);
            high_6 = ((high_6 == 65535) ? 65534 : high_6);
            low_6 = ((low_6 == 32768) ? 65535 : low_6);
            high_6 = ((high_6 == 32768) ? 65535 : high_6);
            ag_words_2[0] = low_6 | high_6 << 16;
            unsigned int _shfl_9 = __shfl_sync(4294967295, first_rs_words[1], 0, 4);
            unsigned int _shfl_16_9 = __shfl_sync(4294967295, first_rs_words[1], 1, 4);
            unsigned int _shfl_17_9 = __shfl_sync(4294967295, first_rs_words[1], 2, 4);
            unsigned int _shfl_18_9 = __shfl_sync(4294967295, first_rs_words[1], 3, 4);
            float lanes_6_1[2];
            unsigned int word_6_1[1];
            word_6_1[0] = _shfl_9;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) << 16);
                (&lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_6_1[2];
            word_6_1[0] = _shfl_16_9;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) << 16);
                (&received_lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_107 = 0; j_107 < 2; j_107++) {
                lanes_6_1[j_107] = lanes_6_1[j_107] + received_lanes_6_1[j_107];
            }
            unsigned int lanes_bf16_8_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6_1[_lp*2 + 0], lanes_6_1[_lp*2+1 + 0]));
                lanes_bf16_8_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_8_1[_pair]) << 16);
                (&lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_8_1[_pair]) & 0xffff0000u);
            }
            word_6_1[0] = _shfl_17_9;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) << 16);
                (&received_lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_108 = 0; j_108 < 2; j_108++) {
                lanes_6_1[j_108] = lanes_6_1[j_108] + received_lanes_6_1[j_108];
            }
            unsigned int lanes_bf16_1_6_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6_1[_lp*2 + 0], lanes_6_1[_lp*2+1 + 0]));
                lanes_bf16_1_6_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_6_1[_pair]) << 16);
                (&lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_6_1[_pair]) & 0xffff0000u);
            }
            word_6_1[0] = _shfl_18_9;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) << 16);
                (&received_lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_6_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_109 = 0; j_109 < 2; j_109++) {
                lanes_6_1[j_109] = lanes_6_1[j_109] + received_lanes_6_1[j_109];
            }
            unsigned int lanes_bf16_2_6_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6_1[_lp*2 + 0], lanes_6_1[_lp*2+1 + 0]));
                lanes_bf16_2_6_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_6_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_6_1[_pair]) << 16);
                (&lanes_6_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_6_1[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_82[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_6_1[_lp*2 + 0], lanes_6_1[_lp*2+1 + 0]));
                scaled_bf16_82[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_6_1 = scaled_bf16_82[0] & 65535;
            unsigned int high_6_1 = scaled_bf16_82[0] >> 16;
            low_6_1 = ((low_6_1 == 65535) ? 65534 : low_6_1);
            high_6_1 = ((high_6_1 == 65535) ? 65534 : high_6_1);
            low_6_1 = ((low_6_1 == 32768) ? 65535 : low_6_1);
            high_6_1 = ((high_6_1 == 32768) ? 65535 : high_6_1);
            ag_words_2[1] = low_6_1 | high_6_1 << 16;
            unsigned int _shfl_10 = __shfl_sync(4294967295, first_rs_words[2], 0, 4);
            unsigned int _shfl_16_10 = __shfl_sync(4294967295, first_rs_words[2], 1, 4);
            unsigned int _shfl_17_10 = __shfl_sync(4294967295, first_rs_words[2], 2, 4);
            unsigned int _shfl_18_10 = __shfl_sync(4294967295, first_rs_words[2], 3, 4);
            float lanes_7[2];
            unsigned int word_7_1[1];
            word_7_1[0] = _shfl_10;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) << 16);
                (&lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_7[2];
            word_7_1[0] = _shfl_16_10;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) << 16);
                (&received_lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_110 = 0; j_110 < 2; j_110++) {
                lanes_7[j_110] = lanes_7[j_110] + received_lanes_7[j_110];
            }
            unsigned int lanes_bf16_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_7[_lp*2 + 0], lanes_7[_lp*2+1 + 0]));
                lanes_bf16_9[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_9[_pair]) << 16);
                (&lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_9[_pair]) & 0xffff0000u);
            }
            word_7_1[0] = _shfl_17_10;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) << 16);
                (&received_lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_111 = 0; j_111 < 2; j_111++) {
                lanes_7[j_111] = lanes_7[j_111] + received_lanes_7[j_111];
            }
            unsigned int lanes_bf16_1_7[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_7[_lp*2 + 0], lanes_7[_lp*2+1 + 0]));
                lanes_bf16_1_7[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_7[_pair]) << 16);
                (&lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_7[_pair]) & 0xffff0000u);
            }
            word_7_1[0] = _shfl_18_10;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) << 16);
                (&received_lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_7_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_112 = 0; j_112 < 2; j_112++) {
                lanes_7[j_112] = lanes_7[j_112] + received_lanes_7[j_112];
            }
            unsigned int lanes_bf16_2_7[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_7[_lp*2 + 0], lanes_7[_lp*2+1 + 0]));
                lanes_bf16_2_7[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_7[_pair]) << 16);
                (&lanes_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_7[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_83[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_7[_lp*2 + 0], lanes_7[_lp*2+1 + 0]));
                scaled_bf16_83[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_7 = scaled_bf16_83[0] & 65535;
            unsigned int high_7 = scaled_bf16_83[0] >> 16;
            low_7 = ((low_7 == 65535) ? 65534 : low_7);
            high_7 = ((high_7 == 65535) ? 65534 : high_7);
            low_7 = ((low_7 == 32768) ? 65535 : low_7);
            high_7 = ((high_7 == 32768) ? 65535 : high_7);
            ag_words_2[2] = low_7 | high_7 << 16;
            unsigned int _shfl_11 = __shfl_sync(4294967295, first_rs_words[3], 0, 4);
            unsigned int _shfl_16_11 = __shfl_sync(4294967295, first_rs_words[3], 1, 4);
            unsigned int _shfl_17_11 = __shfl_sync(4294967295, first_rs_words[3], 2, 4);
            unsigned int _shfl_18_11 = __shfl_sync(4294967295, first_rs_words[3], 3, 4);
            float lanes_8[2];
            unsigned int word_8_1[1];
            word_8_1[0] = _shfl_11;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) << 16);
                (&lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_8[2];
            word_8_1[0] = _shfl_16_11;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) << 16);
                (&received_lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_113 = 0; j_113 < 2; j_113++) {
                lanes_8[j_113] = lanes_8[j_113] + received_lanes_8[j_113];
            }
            unsigned int lanes_bf16_10[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_8[_lp*2 + 0], lanes_8[_lp*2+1 + 0]));
                lanes_bf16_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_10[_pair]) << 16);
                (&lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_10[_pair]) & 0xffff0000u);
            }
            word_8_1[0] = _shfl_17_11;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) << 16);
                (&received_lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_114 = 0; j_114 < 2; j_114++) {
                lanes_8[j_114] = lanes_8[j_114] + received_lanes_8[j_114];
            }
            unsigned int lanes_bf16_1_8[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_8[_lp*2 + 0], lanes_8[_lp*2+1 + 0]));
                lanes_bf16_1_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_8[_pair]) << 16);
                (&lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_8[_pair]) & 0xffff0000u);
            }
            word_8_1[0] = _shfl_18_11;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) << 16);
                (&received_lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_8_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_115 = 0; j_115 < 2; j_115++) {
                lanes_8[j_115] = lanes_8[j_115] + received_lanes_8[j_115];
            }
            unsigned int lanes_bf16_2_8[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_8[_lp*2 + 0], lanes_8[_lp*2+1 + 0]));
                lanes_bf16_2_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_8[_pair]) << 16);
                (&lanes_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_8[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_84[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_8[_lp*2 + 0], lanes_8[_lp*2+1 + 0]));
                scaled_bf16_84[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_8 = scaled_bf16_84[0] & 65535;
            unsigned int high_8 = scaled_bf16_84[0] >> 16;
            low_8 = ((low_8 == 65535) ? 65534 : low_8);
            high_8 = ((high_8 == 65535) ? 65534 : high_8);
            low_8 = ((low_8 == 32768) ? 65535 : low_8);
            high_8 = ((high_8 == 32768) ? 65535 : high_8);
            ag_words_2[3] = low_8 | high_8 << 16;
            {
                auto _gst_92 = rs_source_base + ((long long)data_epoch * comm_stride_elems + (long long)tokens * 7168 + (long long)((int)cluster_id * 896 + rank * 224 + rs_item) * 8);
                __builtin_assume(__isGlobal(_gst_92));
                reinterpret_cast<int4*>(_gst_92)[0] = reinterpret_cast<int4*>(ag_words_2)[0];
            }
        }
    }
    int lead_rs_token = (int)cluster_id + (int)num_clusters;
    if (lead_rs_token < tokens) {
        if (lead_rs_token + (int)num_clusters >= tokens) {
            unsigned int lead_rs_words[4];
            uint32_t sysv_ld_2[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_2[0]), "=r"(sysv_ld_2[1]), "=r"(sysv_ld_2[2]), "=r"(sysv_ld_2[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(rs_src * rs_slot_items + lead_rs_token * 224 + rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word_12 = 0; word_12 < 4; word_12++) {
                lead_rs_words[word_12] = sysv_ld_2[word_12];
            }
            int pending_3 = 0;
            #pragma unroll
            for (int word_13 = 0; word_13 < 4; word_13++) {
                pending_3 = (((lead_rs_words[word_13] & 65535) == 32768) ? 1 : pending_3);
                pending_3 = ((lead_rs_words[word_13] >> 16 == 32768) ? 1 : pending_3);
            }
            if (pending_3 == 1) {
                uint32_t sysv_poll_group_3[4];
                uint32_t _sysv_pending_93 = 1u;
                do {
                    if (_sysv_pending_93 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_3[0]), "=r"(sysv_poll_group_3[1]), "=r"(sysv_poll_group_3[2]), "=r"(sysv_poll_group_3[3]) : "l"(workspace_local + ((long long)data_epoch * comm_stride_elems + (long long)(rs_src * rs_slot_items + lead_rs_token * 224 + rs_item) * 8)) : "memory");
                    }
                    if (_sysv_pending_93 & 1u) {
                        if (!((((sysv_poll_group_3[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_3[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_93 &= ~1u;
                    }
                } while (_sysv_pending_93 != 0u);
                #pragma unroll
                for (int word_14 = 0; word_14 < 4; word_14++) {
                    lead_rs_words[word_14] = sysv_poll_group_3[word_14];
                }
            }
            unsigned int ag_words_3[4];
            unsigned int _shfl_12 = __shfl_sync(4294967295, lead_rs_words[0], 0, 4);
            unsigned int _shfl_16_12 = __shfl_sync(4294967295, lead_rs_words[0], 1, 4);
            unsigned int _shfl_17_12 = __shfl_sync(4294967295, lead_rs_words[0], 2, 4);
            unsigned int _shfl_18_12 = __shfl_sync(4294967295, lead_rs_words[0], 3, 4);
            float lanes_9[2];
            unsigned int word_15[1];
            word_15[0] = _shfl_12;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) << 16);
                (&lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) & 0xffff0000u);
            }
            float received_lanes_9[2];
            word_15[0] = _shfl_16_12;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) << 16);
                (&received_lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_116 = 0; j_116 < 2; j_116++) {
                lanes_9[j_116] = lanes_9[j_116] + received_lanes_9[j_116];
            }
            unsigned int lanes_bf16_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9[_lp*2 + 0], lanes_9[_lp*2+1 + 0]));
                lanes_bf16_11[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_11[_pair]) << 16);
                (&lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_11[_pair]) & 0xffff0000u);
            }
            word_15[0] = _shfl_17_12;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) << 16);
                (&received_lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_117 = 0; j_117 < 2; j_117++) {
                lanes_9[j_117] = lanes_9[j_117] + received_lanes_9[j_117];
            }
            unsigned int lanes_bf16_1_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9[_lp*2 + 0], lanes_9[_lp*2+1 + 0]));
                lanes_bf16_1_9[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_9[_pair]) << 16);
                (&lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_9[_pair]) & 0xffff0000u);
            }
            word_15[0] = _shfl_18_12;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) << 16);
                (&received_lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_15[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_118 = 0; j_118 < 2; j_118++) {
                lanes_9[j_118] = lanes_9[j_118] + received_lanes_9[j_118];
            }
            unsigned int lanes_bf16_2_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9[_lp*2 + 0], lanes_9[_lp*2+1 + 0]));
                lanes_bf16_2_9[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_9[_pair]) << 16);
                (&lanes_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_9[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_85[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9[_lp*2 + 0], lanes_9[_lp*2+1 + 0]));
                scaled_bf16_85[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_9 = scaled_bf16_85[0] & 65535;
            unsigned int high_9 = scaled_bf16_85[0] >> 16;
            low_9 = ((low_9 == 65535) ? 65534 : low_9);
            high_9 = ((high_9 == 65535) ? 65534 : high_9);
            low_9 = ((low_9 == 32768) ? 65535 : low_9);
            high_9 = ((high_9 == 32768) ? 65535 : high_9);
            ag_words_3[0] = low_9 | high_9 << 16;
            unsigned int _shfl_13 = __shfl_sync(4294967295, lead_rs_words[1], 0, 4);
            unsigned int _shfl_16_13 = __shfl_sync(4294967295, lead_rs_words[1], 1, 4);
            unsigned int _shfl_17_13 = __shfl_sync(4294967295, lead_rs_words[1], 2, 4);
            unsigned int _shfl_18_13 = __shfl_sync(4294967295, lead_rs_words[1], 3, 4);
            float lanes_9_1[2];
            unsigned int word_9_1[1];
            word_9_1[0] = _shfl_13;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) << 16);
                (&lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_9_1[2];
            word_9_1[0] = _shfl_16_13;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) << 16);
                (&received_lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_119 = 0; j_119 < 2; j_119++) {
                lanes_9_1[j_119] = lanes_9_1[j_119] + received_lanes_9_1[j_119];
            }
            unsigned int lanes_bf16_11_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9_1[_lp*2 + 0], lanes_9_1[_lp*2+1 + 0]));
                lanes_bf16_11_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_11_1[_pair]) << 16);
                (&lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_11_1[_pair]) & 0xffff0000u);
            }
            word_9_1[0] = _shfl_17_13;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) << 16);
                (&received_lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_120 = 0; j_120 < 2; j_120++) {
                lanes_9_1[j_120] = lanes_9_1[j_120] + received_lanes_9_1[j_120];
            }
            unsigned int lanes_bf16_1_9_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9_1[_lp*2 + 0], lanes_9_1[_lp*2+1 + 0]));
                lanes_bf16_1_9_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_9_1[_pair]) << 16);
                (&lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_9_1[_pair]) & 0xffff0000u);
            }
            word_9_1[0] = _shfl_18_13;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) << 16);
                (&received_lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_9_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_121 = 0; j_121 < 2; j_121++) {
                lanes_9_1[j_121] = lanes_9_1[j_121] + received_lanes_9_1[j_121];
            }
            unsigned int lanes_bf16_2_9_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9_1[_lp*2 + 0], lanes_9_1[_lp*2+1 + 0]));
                lanes_bf16_2_9_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_9_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_9_1[_pair]) << 16);
                (&lanes_9_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_9_1[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_86[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_9_1[_lp*2 + 0], lanes_9_1[_lp*2+1 + 0]));
                scaled_bf16_86[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_9_1 = scaled_bf16_86[0] & 65535;
            unsigned int high_9_1 = scaled_bf16_86[0] >> 16;
            low_9_1 = ((low_9_1 == 65535) ? 65534 : low_9_1);
            high_9_1 = ((high_9_1 == 65535) ? 65534 : high_9_1);
            low_9_1 = ((low_9_1 == 32768) ? 65535 : low_9_1);
            high_9_1 = ((high_9_1 == 32768) ? 65535 : high_9_1);
            ag_words_3[1] = low_9_1 | high_9_1 << 16;
            unsigned int _shfl_14 = __shfl_sync(4294967295, lead_rs_words[2], 0, 4);
            unsigned int _shfl_16_14 = __shfl_sync(4294967295, lead_rs_words[2], 1, 4);
            unsigned int _shfl_17_14 = __shfl_sync(4294967295, lead_rs_words[2], 2, 4);
            unsigned int _shfl_18_14 = __shfl_sync(4294967295, lead_rs_words[2], 3, 4);
            float lanes_10[2];
            unsigned int word_10_1[1];
            word_10_1[0] = _shfl_14;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) << 16);
                (&lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_10[2];
            word_10_1[0] = _shfl_16_14;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) << 16);
                (&received_lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_122 = 0; j_122 < 2; j_122++) {
                lanes_10[j_122] = lanes_10[j_122] + received_lanes_10[j_122];
            }
            unsigned int lanes_bf16_12[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_10[_lp*2 + 0], lanes_10[_lp*2+1 + 0]));
                lanes_bf16_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_12[_pair]) << 16);
                (&lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_12[_pair]) & 0xffff0000u);
            }
            word_10_1[0] = _shfl_17_14;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) << 16);
                (&received_lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_123 = 0; j_123 < 2; j_123++) {
                lanes_10[j_123] = lanes_10[j_123] + received_lanes_10[j_123];
            }
            unsigned int lanes_bf16_1_10[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_10[_lp*2 + 0], lanes_10[_lp*2+1 + 0]));
                lanes_bf16_1_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_10[_pair]) << 16);
                (&lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_10[_pair]) & 0xffff0000u);
            }
            word_10_1[0] = _shfl_18_14;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) << 16);
                (&received_lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_10_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_124 = 0; j_124 < 2; j_124++) {
                lanes_10[j_124] = lanes_10[j_124] + received_lanes_10[j_124];
            }
            unsigned int lanes_bf16_2_10[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_10[_lp*2 + 0], lanes_10[_lp*2+1 + 0]));
                lanes_bf16_2_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_10[_pair]) << 16);
                (&lanes_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_10[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_87[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_10[_lp*2 + 0], lanes_10[_lp*2+1 + 0]));
                scaled_bf16_87[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_10 = scaled_bf16_87[0] & 65535;
            unsigned int high_10 = scaled_bf16_87[0] >> 16;
            low_10 = ((low_10 == 65535) ? 65534 : low_10);
            high_10 = ((high_10 == 65535) ? 65534 : high_10);
            low_10 = ((low_10 == 32768) ? 65535 : low_10);
            high_10 = ((high_10 == 32768) ? 65535 : high_10);
            ag_words_3[2] = low_10 | high_10 << 16;
            unsigned int _shfl_15 = __shfl_sync(4294967295, lead_rs_words[3], 0, 4);
            unsigned int _shfl_16_15 = __shfl_sync(4294967295, lead_rs_words[3], 1, 4);
            unsigned int _shfl_17_15 = __shfl_sync(4294967295, lead_rs_words[3], 2, 4);
            unsigned int _shfl_18_15 = __shfl_sync(4294967295, lead_rs_words[3], 3, 4);
            float lanes_11[2];
            unsigned int word_11_1[1];
            word_11_1[0] = _shfl_15;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) << 16);
                (&lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_11[2];
            word_11_1[0] = _shfl_16_15;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) << 16);
                (&received_lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_125 = 0; j_125 < 2; j_125++) {
                lanes_11[j_125] = lanes_11[j_125] + received_lanes_11[j_125];
            }
            unsigned int lanes_bf16_13[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_11[_lp*2 + 0], lanes_11[_lp*2+1 + 0]));
                lanes_bf16_13[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_13[_pair]) << 16);
                (&lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_13[_pair]) & 0xffff0000u);
            }
            word_11_1[0] = _shfl_17_15;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) << 16);
                (&received_lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_126 = 0; j_126 < 2; j_126++) {
                lanes_11[j_126] = lanes_11[j_126] + received_lanes_11[j_126];
            }
            unsigned int lanes_bf16_1_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_11[_lp*2 + 0], lanes_11[_lp*2+1 + 0]));
                lanes_bf16_1_11[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_11[_pair]) << 16);
                (&lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_11[_pair]) & 0xffff0000u);
            }
            word_11_1[0] = _shfl_18_15;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) << 16);
                (&received_lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_11_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_127 = 0; j_127 < 2; j_127++) {
                lanes_11[j_127] = lanes_11[j_127] + received_lanes_11[j_127];
            }
            unsigned int lanes_bf16_2_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_11[_lp*2 + 0], lanes_11[_lp*2+1 + 0]));
                lanes_bf16_2_11[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_11[_pair]) << 16);
                (&lanes_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_11[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_88[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_11[_lp*2 + 0], lanes_11[_lp*2+1 + 0]));
                scaled_bf16_88[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_11 = scaled_bf16_88[0] & 65535;
            unsigned int high_11 = scaled_bf16_88[0] >> 16;
            low_11 = ((low_11 == 65535) ? 65534 : low_11);
            high_11 = ((high_11 == 65535) ? 65534 : high_11);
            low_11 = ((low_11 == 32768) ? 65535 : low_11);
            high_11 = ((high_11 == 32768) ? 65535 : high_11);
            ag_words_3[3] = low_11 | high_11 << 16;
            {
                auto _gst_94 = rs_source_base + ((long long)data_epoch * comm_stride_elems + (long long)tokens * 7168 + (long long)(lead_rs_token * 896 + rank * 224 + rs_item) * 8);
                __builtin_assume(__isGlobal(_gst_94));
                reinterpret_cast<int4*>(_gst_94)[0] = reinterpret_cast<int4*>(ag_words_3)[0];
            }
        }
    }
    unsigned int _phase_rms_exchange = 0;
    #pragma unroll 1
    for (int token_1 = cluster_id; token_1 < tokens; token_1 += (int)num_clusters) {
        long long data_base = (long long)data_epoch * comm_stride_elems;
        int rs_token = token_1 + 2 * (int)num_clusters;
        unsigned int rs_words[4];
        if (rs_token < tokens) {
            uint32_t sysv_ld_3[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_ld_3[0]), "=r"(sysv_ld_3[1]), "=r"(sysv_ld_3[2]), "=r"(sysv_ld_3[3]) : "l"(workspace_local + (data_base + (long long)(rs_src * rs_slot_items + rs_token * 224 + rs_item) * 8)) : "memory");
            #pragma unroll
            for (int word_16 = 0; word_16 < 4; word_16++) {
                rs_words[word_16] = sysv_ld_3[word_16];
            }
        }
        int prefetch_elem = access * 8;
        float _vec_load_79[8];
        {
            const uint4* _vptr_95 = reinterpret_cast<const uint4*>(residual + prefetch_elem + 0);
            uint4 _vld_95[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_95[_blk].x), "=r"(_vld_95[_blk].y), "=r"(_vld_95[_blk].z), "=r"(_vld_95[_blk].w) : "l"((const void*)(_vptr_95 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_95 = reinterpret_cast<uint32_t*>(&_vld_95[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_79[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_95[_pair]) << 16);
                    (&_vec_load_79[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_95[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_80[8];
        {
            const uint4* _vptr_96 = reinterpret_cast<const uint4*>(gamma + (lane_access * 8) + 0);
            uint4 _vld_96[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_96[_blk] = _vptr_96[_blk];
                uint32_t* _vpairs_96 = reinterpret_cast<uint32_t*>(&_vld_96[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_80[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_96[_pair]) << 16);
                    (&_vec_load_80[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_96[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int ag_probe[4];
        uint32_t _sysv_ld_2[4];
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_2[0]), "=r"(_sysv_ld_2[1]), "=r"(_sysv_ld_2[2]), "=r"(_sysv_ld_2[3]) : "l"(workspace_local + (data_base + (long long)tokens * 7168 + (long long)access * 8)) : "memory");
        #pragma unroll
        for (int word_17 = 0; word_17 < 4; word_17++) {
            ag_probe[word_17] = _sysv_ld_2[word_17];
        }
        if (rs_token < tokens) {
            int pending_4 = 0;
            #pragma unroll
            for (int word_18 = 0; word_18 < 4; word_18++) {
                pending_4 = (((rs_words[word_18] & 65535) == 32768) ? 1 : pending_4);
                pending_4 = ((rs_words[word_18] >> 16 == 32768) ? 1 : pending_4);
            }
            if (pending_4 == 1) {
                uint32_t sysv_poll_group_4[4];
                uint32_t _sysv_pending_97 = 1u;
                do {
                    if (_sysv_pending_97 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_4[0]), "=r"(sysv_poll_group_4[1]), "=r"(sysv_poll_group_4[2]), "=r"(sysv_poll_group_4[3]) : "l"(workspace_local + (data_base + (long long)(rs_src * rs_slot_items + rs_token * 224 + rs_item) * 8)) : "memory");
                    }
                    if (_sysv_pending_97 & 1u) {
                        if (!((((sysv_poll_group_4[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_4[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_97 &= ~1u;
                    }
                } while (_sysv_pending_97 != 0u);
                #pragma unroll
                for (int word_19 = 0; word_19 < 4; word_19++) {
                    rs_words[word_19] = sysv_poll_group_4[word_19];
                }
            }
            unsigned int ag_words_4[4];
            unsigned int _shfl_16 = __shfl_sync(4294967295, rs_words[0], 0, 4);
            unsigned int _shfl_1_0 = __shfl_sync(4294967295, rs_words[0], 1, 4);
            unsigned int _shfl_2_0 = __shfl_sync(4294967295, rs_words[0], 2, 4);
            unsigned int _shfl_3_0 = __shfl_sync(4294967295, rs_words[0], 3, 4);
            float lanes_12[2];
            unsigned int word_20[1];
            word_20[0] = _shfl_16;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) << 16);
                (&lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) & 0xffff0000u);
            }
            float received_lanes_12[2];
            word_20[0] = _shfl_1_0;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) << 16);
                (&received_lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_128 = 0; j_128 < 2; j_128++) {
                lanes_12[j_128] = lanes_12[j_128] + received_lanes_12[j_128];
            }
            unsigned int lanes_bf16_14[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12[_lp*2 + 0], lanes_12[_lp*2+1 + 0]));
                lanes_bf16_14[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_14[_pair]) << 16);
                (&lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_14[_pair]) & 0xffff0000u);
            }
            word_20[0] = _shfl_2_0;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) << 16);
                (&received_lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_129 = 0; j_129 < 2; j_129++) {
                lanes_12[j_129] = lanes_12[j_129] + received_lanes_12[j_129];
            }
            unsigned int lanes_bf16_1_12[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12[_lp*2 + 0], lanes_12[_lp*2+1 + 0]));
                lanes_bf16_1_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_12[_pair]) << 16);
                (&lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_12[_pair]) & 0xffff0000u);
            }
            word_20[0] = _shfl_3_0;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) << 16);
                (&received_lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_20[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_130 = 0; j_130 < 2; j_130++) {
                lanes_12[j_130] = lanes_12[j_130] + received_lanes_12[j_130];
            }
            unsigned int lanes_bf16_2_12[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12[_lp*2 + 0], lanes_12[_lp*2+1 + 0]));
                lanes_bf16_2_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_12[_pair]) << 16);
                (&lanes_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_12[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_89[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12[_lp*2 + 0], lanes_12[_lp*2+1 + 0]));
                scaled_bf16_89[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_12 = scaled_bf16_89[0] & 65535;
            unsigned int high_12 = scaled_bf16_89[0] >> 16;
            low_12 = ((low_12 == 65535) ? 65534 : low_12);
            high_12 = ((high_12 == 65535) ? 65534 : high_12);
            low_12 = ((low_12 == 32768) ? 65535 : low_12);
            high_12 = ((high_12 == 32768) ? 65535 : high_12);
            ag_words_4[0] = low_12 | high_12 << 16;
            unsigned int _shfl_17 = __shfl_sync(4294967295, rs_words[1], 0, 4);
            unsigned int _shfl_1_1 = __shfl_sync(4294967295, rs_words[1], 1, 4);
            unsigned int _shfl_2_1 = __shfl_sync(4294967295, rs_words[1], 2, 4);
            unsigned int _shfl_3_1 = __shfl_sync(4294967295, rs_words[1], 3, 4);
            float lanes_12_1[2];
            unsigned int word_12_1[1];
            word_12_1[0] = _shfl_17;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) << 16);
                (&lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_12_1[2];
            word_12_1[0] = _shfl_1_1;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) << 16);
                (&received_lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_131 = 0; j_131 < 2; j_131++) {
                lanes_12_1[j_131] = lanes_12_1[j_131] + received_lanes_12_1[j_131];
            }
            unsigned int lanes_bf16_14_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12_1[_lp*2 + 0], lanes_12_1[_lp*2+1 + 0]));
                lanes_bf16_14_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_14_1[_pair]) << 16);
                (&lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_14_1[_pair]) & 0xffff0000u);
            }
            word_12_1[0] = _shfl_2_1;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) << 16);
                (&received_lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_132 = 0; j_132 < 2; j_132++) {
                lanes_12_1[j_132] = lanes_12_1[j_132] + received_lanes_12_1[j_132];
            }
            unsigned int lanes_bf16_1_12_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12_1[_lp*2 + 0], lanes_12_1[_lp*2+1 + 0]));
                lanes_bf16_1_12_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_12_1[_pair]) << 16);
                (&lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_12_1[_pair]) & 0xffff0000u);
            }
            word_12_1[0] = _shfl_3_1;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) << 16);
                (&received_lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_12_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_133 = 0; j_133 < 2; j_133++) {
                lanes_12_1[j_133] = lanes_12_1[j_133] + received_lanes_12_1[j_133];
            }
            unsigned int lanes_bf16_2_12_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12_1[_lp*2 + 0], lanes_12_1[_lp*2+1 + 0]));
                lanes_bf16_2_12_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_12_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_12_1[_pair]) << 16);
                (&lanes_12_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_12_1[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_90[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_12_1[_lp*2 + 0], lanes_12_1[_lp*2+1 + 0]));
                scaled_bf16_90[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_12_1 = scaled_bf16_90[0] & 65535;
            unsigned int high_12_1 = scaled_bf16_90[0] >> 16;
            low_12_1 = ((low_12_1 == 65535) ? 65534 : low_12_1);
            high_12_1 = ((high_12_1 == 65535) ? 65534 : high_12_1);
            low_12_1 = ((low_12_1 == 32768) ? 65535 : low_12_1);
            high_12_1 = ((high_12_1 == 32768) ? 65535 : high_12_1);
            ag_words_4[1] = low_12_1 | high_12_1 << 16;
            unsigned int _shfl_18 = __shfl_sync(4294967295, rs_words[2], 0, 4);
            unsigned int _shfl_1_2 = __shfl_sync(4294967295, rs_words[2], 1, 4);
            unsigned int _shfl_2_2 = __shfl_sync(4294967295, rs_words[2], 2, 4);
            unsigned int _shfl_3_2 = __shfl_sync(4294967295, rs_words[2], 3, 4);
            float lanes_13[2];
            unsigned int word_13_1[1];
            word_13_1[0] = _shfl_18;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) << 16);
                (&lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_13[2];
            word_13_1[0] = _shfl_1_2;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) << 16);
                (&received_lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_134 = 0; j_134 < 2; j_134++) {
                lanes_13[j_134] = lanes_13[j_134] + received_lanes_13[j_134];
            }
            unsigned int lanes_bf16_15[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_13[_lp*2 + 0], lanes_13[_lp*2+1 + 0]));
                lanes_bf16_15[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_15[_pair]) << 16);
                (&lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_15[_pair]) & 0xffff0000u);
            }
            word_13_1[0] = _shfl_2_2;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) << 16);
                (&received_lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_135 = 0; j_135 < 2; j_135++) {
                lanes_13[j_135] = lanes_13[j_135] + received_lanes_13[j_135];
            }
            unsigned int lanes_bf16_1_13[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_13[_lp*2 + 0], lanes_13[_lp*2+1 + 0]));
                lanes_bf16_1_13[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_13[_pair]) << 16);
                (&lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_13[_pair]) & 0xffff0000u);
            }
            word_13_1[0] = _shfl_3_2;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) << 16);
                (&received_lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_13_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_136 = 0; j_136 < 2; j_136++) {
                lanes_13[j_136] = lanes_13[j_136] + received_lanes_13[j_136];
            }
            unsigned int lanes_bf16_2_13[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_13[_lp*2 + 0], lanes_13[_lp*2+1 + 0]));
                lanes_bf16_2_13[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_13[_pair]) << 16);
                (&lanes_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_13[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_91[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_13[_lp*2 + 0], lanes_13[_lp*2+1 + 0]));
                scaled_bf16_91[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_13 = scaled_bf16_91[0] & 65535;
            unsigned int high_13 = scaled_bf16_91[0] >> 16;
            low_13 = ((low_13 == 65535) ? 65534 : low_13);
            high_13 = ((high_13 == 65535) ? 65534 : high_13);
            low_13 = ((low_13 == 32768) ? 65535 : low_13);
            high_13 = ((high_13 == 32768) ? 65535 : high_13);
            ag_words_4[2] = low_13 | high_13 << 16;
            unsigned int _shfl_19 = __shfl_sync(4294967295, rs_words[3], 0, 4);
            unsigned int _shfl_1_3 = __shfl_sync(4294967295, rs_words[3], 1, 4);
            unsigned int _shfl_2_3 = __shfl_sync(4294967295, rs_words[3], 2, 4);
            unsigned int _shfl_3_3 = __shfl_sync(4294967295, rs_words[3], 3, 4);
            float lanes_14[2];
            unsigned int word_14_1[1];
            word_14_1[0] = _shfl_19;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) << 16);
                (&lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) & 0xffff0000u);
            }
            float received_lanes_14[2];
            word_14_1[0] = _shfl_1_3;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) << 16);
                (&received_lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_137 = 0; j_137 < 2; j_137++) {
                lanes_14[j_137] = lanes_14[j_137] + received_lanes_14[j_137];
            }
            unsigned int lanes_bf16_16[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_14[_lp*2 + 0], lanes_14[_lp*2+1 + 0]));
                lanes_bf16_16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_16[_pair]) << 16);
                (&lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_16[_pair]) & 0xffff0000u);
            }
            word_14_1[0] = _shfl_2_3;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) << 16);
                (&received_lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_138 = 0; j_138 < 2; j_138++) {
                lanes_14[j_138] = lanes_14[j_138] + received_lanes_14[j_138];
            }
            unsigned int lanes_bf16_1_14[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_14[_lp*2 + 0], lanes_14[_lp*2+1 + 0]));
                lanes_bf16_1_14[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_14[_pair]) << 16);
                (&lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_1_14[_pair]) & 0xffff0000u);
            }
            word_14_1[0] = _shfl_3_3;
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&received_lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) << 16);
                (&received_lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(word_14_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_139 = 0; j_139 < 2; j_139++) {
                lanes_14[j_139] = lanes_14[j_139] + received_lanes_14[j_139];
            }
            unsigned int lanes_bf16_2_14[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_14[_lp*2 + 0], lanes_14[_lp*2+1 + 0]));
                lanes_bf16_2_14[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&lanes_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_14[_pair]) << 16);
                (&lanes_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(lanes_bf16_2_14[_pair]) & 0xffff0000u);
            }
            unsigned int scaled_bf16_92[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lanes_14[_lp*2 + 0], lanes_14[_lp*2+1 + 0]));
                scaled_bf16_92[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int low_14 = scaled_bf16_92[0] & 65535;
            unsigned int high_14 = scaled_bf16_92[0] >> 16;
            low_14 = ((low_14 == 65535) ? 65534 : low_14);
            high_14 = ((high_14 == 65535) ? 65534 : high_14);
            low_14 = ((low_14 == 32768) ? 65535 : low_14);
            high_14 = ((high_14 == 32768) ? 65535 : high_14);
            ag_words_4[3] = low_14 | high_14 << 16;
            {
                auto _gst_98 = rs_source_base + (data_base + (long long)tokens * 7168 + (long long)(rs_token * 896 + rank * 224 + rs_item) * 8);
                __builtin_assume(__isGlobal(_gst_98));
                reinterpret_cast<int4*>(_gst_98)[0] = reinterpret_cast<int4*>(ag_words_4)[0];
            }
        }
        int pending_5 = 0;
        #pragma unroll
        for (int word_21 = 0; word_21 < 4; word_21++) {
            pending_5 = (((ag_probe[word_21] & 65535) == 32768) ? 1 : pending_5);
            pending_5 = ((ag_probe[word_21] >> 16 == 32768) ? 1 : pending_5);
        }
        if (pending_5 == 1) {
            uint32_t sysv_poll_group_5[4];
            uint32_t _sysv_pending_99 = 1u;
            do {
                if (_sysv_pending_99 & 1u) {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(sysv_poll_group_5[0]), "=r"(sysv_poll_group_5[1]), "=r"(sysv_poll_group_5[2]), "=r"(sysv_poll_group_5[3]) : "l"(workspace_local + (data_base + (long long)tokens * 7168 + (long long)access * 8)) : "memory");
                }
                if (_sysv_pending_99 & 1u) {
                    if (!((((sysv_poll_group_5[0] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[0] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[1] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[1] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[2] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[2] >> 16) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[3] >> 0) & 0xffffu) == 0x8000u) || (((sysv_poll_group_5[3] >> 16) & 0xffffu) == 0x8000u))) _sysv_pending_99 &= ~1u;
                }
            } while (_sysv_pending_99 != 0u);
            #pragma unroll
            for (int word_22 = 0; word_22 < 4; word_22++) {
                ag_probe[word_22] = sysv_poll_group_5[word_22];
            }
        }
        unsigned int sum_words[4];
        #pragma unroll
        for (int word_23 = 0; word_23 < 4; word_23++) {
            unsigned int low_15 = ag_probe[word_23] & 65535;
            unsigned int high_15 = ag_probe[word_23] >> 16;
            low_15 = ((low_15 == 65535) ? 32768 : low_15);
            high_15 = ((high_15 == 65535) ? 32768 : high_15);
            sum_words[word_23] = low_15 | high_15 << 16;
        }
        float sum_value[8];
        float unpack_bf16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sum_words[_pair]) << 16);
            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sum_words[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_140 = 0; j_140 < 8; j_140++) {
            sum_value[j_140] = unpack_bf16[j_140];
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
        for (int j_141 = 0; j_141 < 8; j_141++) {
            _vec_load_79[j_141] = _vec_load_79[j_141] + sum_value[j_141];
        }
        unsigned int scaled_bf16_93[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_79[_lp*2 + 0], _vec_load_79[_lp*2+1 + 0]));
            scaled_bf16_93[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_79[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_93[_pair]) << 16);
            (&_vec_load_79[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_93[_pair]) & 0xffff0000u);
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
        for (int j_142 = 0; j_142 < 8; j_142++) {
            square_sum = __fmaf_rn(_vec_load_79[j_142], _vec_load_79[j_142], square_sum);
        }
        float _warp_reduce_100 = square_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_100 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_100, offset);
        square_sum = _warp_reduce_100;
        int rms_parity = (int)(rms_trip & 1);
        if (lane == 0) {
            reduce_smem[rms_parity * 7 + warp] = square_sum;
        }
        __syncthreads();
        float block_sum = 0.0f;
        if (tid < 7) {
            block_sum = reduce_smem[rms_parity * 7 + tid];
        }
        float _warp_reduce_101 = block_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_101 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_101, offset);
        block_sum = _warp_reduce_101;
        int slot_base = rms_parity * 4;
        if (tid < 4) {
            uint32_t _cluster_mapa_102;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_102) : "r"(rms_partials_addr), "r"(tid));
            uint32_t _mbar_peer_103;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_103) : "r"(rms_exchange_addr + (rms_exchange_rms_trip) * 8), "r"(tid));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_102 + (unsigned int)((slot_base + cta_rank) * 4)), "f"(block_sum), "r"(_mbar_peer_103) : "memory");
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
        for (int j_143 = 0; j_143 < 8; j_143++) {
            norm_value[j_143] = _vec_load_79[j_143] * rstd * (_vec_load_80[j_143] + weight_bias);
        }
        unsigned int scaled_bf16_94[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
            scaled_bf16_94[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(norm_out + elem_0), "r"(scaled_bf16_94[(0) + 0]), "r"(scaled_bf16_94[(0) + 1]), "r"(scaled_bf16_94[(0) + 2]), "r"(scaled_bf16_94[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
        access += access_stride;
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    unsigned int clear_words[4];
    #pragma unroll
    for (int word_24 = 0; word_24 < 4; word_24++) {
        clear_words[word_24] = 2147516416;
    }
    long long clear_base = (long long)clear_epoch * comm_stride_elems;
    #pragma unroll 4
    for (int access_1 = access + tokens * 896; access_1 < clear_size / 8; access_1 += access_stride) {
        asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(workspace_local + (clear_base + (long long)access_1 * 8)), "r"(clear_words[(0) + 0]), "r"(clear_words[(0) + 1]), "r"(clear_words[(0) + 2]), "r"(clear_words[(0) + 3]), "l"(0x12F0000000000000ULL) : "memory");
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _sysv_wait_p_104 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_sysv_wait_p_104 != static_cast<int>(num_bids)) {}
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
