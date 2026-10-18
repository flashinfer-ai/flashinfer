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
#define NUM_TOKEN_PIPE_STAGES 2
#define SMEM_PREFIX_BUFS_OFF 0
#define SMEM_PREFIX_BUFS_STAGE_BYTES 14336
#define SMEM_PREFIX_BUFS_STRIDE 14336
#define SMEM_DELTA_BUFS_OFF 28672
#define SMEM_DELTA_BUFS_STAGE_BYTES 14336
#define SMEM_DELTA_BUFS_STRIDE 14336
#define SMEM_OUTPUT_NORM_BUF_OFF 57344
#define SMEM_OUTPUT_NORM_BUF_STAGE_BYTES 14336
#define SMEM_OUTPUT_NORM_BUF_STRIDE 14336
#define SMEM_STATS_OFF 71680
#define SMEM_STATS_STAGE_BYTES 32
#define SMEM_STATS_STRIDE 32
#define SMEM_TOTAL 71808
#define THREADS 288
#define LAUNCH_MIN_BLOCKS 1

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



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}



extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_attn_res_80715310f7aa0cb02593(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ output_norm_weight, float output_norm_eps, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 71712;
    #define ready_addr (mbar_base + 0)
    #define consumed_addr (mbar_base + 16)
    #define output_norm_ready_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* prefix_bufs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int prefix_bufs_addr = smem + 0;
    __nv_bfloat16* delta_bufs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 28672);
    const int delta_bufs_addr = smem + 28672;
    __nv_bfloat16* output_norm_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 57344);
    const int output_norm_buf_addr = smem + 57344;
    float* stats = reinterpret_cast<float*>(smem_raw + 71680);
    const int stats_addr = smem + 71680;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 5 barriers)
    // Mbarriers at smem_raw[71712..71752)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'token_pipe' ---
            // ready: 2 barriers, init_count=1
            mbarrier_init(smem + 71712, 1);
            mbarrier_init(smem + 71720, 1);
            // consumed: 2 barriers, init_count=256
            mbarrier_init(smem + 71728, 256);
            mbarrier_init(smem + 71736, 256);
            // output_norm_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 71744, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                mbarrier_arrive_expect_tx(output_norm_ready_addr, 14336);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(output_norm_buf_addr), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(output_norm_weight) + ((unsigned long long)0 * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(output_norm_ready_addr)
                    : "memory");
            }
            unsigned int producer_stage = 0;
            unsigned int producer_phase = 0;
            #pragma unroll 1
            for (int token = bid; token < M; token += num_bids) {
                mbarrier_wait(consumed_addr + (producer_stage) * 8, producer_phase ^ 1);
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(ready_addr + (producer_stage) * 8, 28672);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(prefix_bufs_addr + producer_stage * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(prefix) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (producer_stage) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(delta_bufs_addr + producer_stage * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(delta) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (producer_stage) * 8)
                        : "memory");
                }
                producer_stage += 1;
                if (producer_stage == 2) { producer_stage = 0; producer_phase ^= 1; }
            }
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    // ---- Role: consumers ----
    } else if (warp >= 1 && warp <= 8) {
        { // consumers_main
            int warp_id_in_role = (warp - 1);
            int consumer_warp = warp_id_in_role;
            int consumer_thread = consumer_warp * 32 + lane;
            int group = consumer_warp / 4;
            int thread_in_group = consumer_thread % 128;
            unsigned int consumer_stage = 0;
            unsigned int consumer_phase = 0;
            #pragma unroll 1
            for (int token_1 = bid; token_1 < M; token_1 += num_bids) {
                mbarrier_wait(ready_addr + (consumer_stage) * 8, consumer_phase);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float updated[28];
                float2 _f2_0 = make_float2(0.0f, 0.0f);
                float2 local_sum_pair = _f2_0;
                #pragma unroll
                for (int slice_idx = 0; slice_idx < 4; slice_idx++) {
                    if (slice_idx == 3) {
                        int hidden_base = 6144 + group * 512 + thread_in_group * 4;
                        float _prefix_bufs_reg_0[4];
                        {
                            uint32_t _smem_addr_0 = prefix_bufs_addr + (consumer_stage * 7168 + (unsigned int)hidden_base) * 2;
                            uint32_t _bf16x2_0[2];
                            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_bf16x2_0[0]), "=r"(_bf16x2_0[1]) : "r"(_smem_addr_0) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_prefix_bufs_reg_0[0])[0]), "=f"((&_prefix_bufs_reg_0[0])[1])
                                : "r"(_bf16x2_0[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_prefix_bufs_reg_0[2])[0]), "=f"((&_prefix_bufs_reg_0[2])[1])
                                : "r"(_bf16x2_0[1]));
                        }
                        float _delta_bufs_reg_0[4];
                        {
                            uint32_t _smem_addr_1 = delta_bufs_addr + (consumer_stage * 7168 + (unsigned int)hidden_base) * 2;
                            uint32_t _bf16x2_1[2];
                            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_bf16x2_1[0]), "=r"(_bf16x2_1[1]) : "r"(_smem_addr_1) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_delta_bufs_reg_0[0])[0]), "=f"((&_delta_bufs_reg_0[0])[1])
                                : "r"(_bf16x2_1[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_delta_bufs_reg_0[2])[0]), "=f"((&_delta_bufs_reg_0[2])[1])
                                : "r"(_bf16x2_1[1]));
                        }
                        #pragma unroll
                        for (int pair_idx = 0; pair_idx < 2; pair_idx++) {
                            const int value_idx = pair_idx * 2;
                            const int acc_idx = slice_idx * 8 + value_idx;
                            float2 _f2_1 = make_float2(_prefix_bufs_reg_0[value_idx], _prefix_bufs_reg_0[value_idx + 1]);
                            float2 _f2_2 = make_float2(_delta_bufs_reg_0[value_idx], _delta_bufs_reg_0[value_idx + 1]);
                            float2 updated_pair = add_f32x2_noftz(_f2_1, _f2_2);
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(updated_pair.x);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            updated[acc_idx] = _cvt_f32_0;
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(updated_pair.y);
                            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                            updated[acc_idx + 1] = _cvt_f32_1;
                        }
                        {
                            uint2 _pk2;
                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                            _pk[0] = __floats2bfloat162_rn(updated[slice_idx * 8 + 0], updated[slice_idx * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(updated[slice_idx * 8 + 2], updated[slice_idx * 8 + 3]);
                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base]) = _pk2;
                        }
                        #pragma unroll
                        for (int pair_idx_1 = 0; pair_idx_1 < 2; pair_idx_1++) {
                            const int acc_idx_1 = slice_idx * 8 + pair_idx_1 * 2;
                            float2 _f2_3 = make_float2(updated[acc_idx_1], updated[acc_idx_1 + 1]);
                            float2 value_pair = _f2_3;
                            local_sum_pair = fma_f32x2_rn_ftz(value_pair, value_pair, local_sum_pair);
                        }
                    } else {
                        int hidden_tile = slice_idx * 2 + group;
                        int hidden_base_1 = hidden_tile * 1024 + thread_in_group * 8;
                        float _prefix_bufs_reg_1[8];
                        {
                            uint32_t _smem_addr_2 = prefix_bufs_addr + (consumer_stage * 7168 + (unsigned int)hidden_base_1) * 2;
                            uint32_t _bf16x2_2[4];
                            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_bf16x2_2[0]), "=r"(_bf16x2_2[1]), "=r"(_bf16x2_2[2]), "=r"(_bf16x2_2[3]) : "r"(_smem_addr_2) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_prefix_bufs_reg_1[0])[0]), "=f"((&_prefix_bufs_reg_1[0])[1])
                                : "r"(_bf16x2_2[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_prefix_bufs_reg_1[2])[0]), "=f"((&_prefix_bufs_reg_1[2])[1])
                                : "r"(_bf16x2_2[1]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_prefix_bufs_reg_1[4])[0]), "=f"((&_prefix_bufs_reg_1[4])[1])
                                : "r"(_bf16x2_2[2]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_prefix_bufs_reg_1[6])[0]), "=f"((&_prefix_bufs_reg_1[6])[1])
                                : "r"(_bf16x2_2[3]));
                        }
                        float _delta_bufs_reg_1[8];
                        {
                            uint32_t _smem_addr_3 = delta_bufs_addr + (consumer_stage * 7168 + (unsigned int)hidden_base_1) * 2;
                            uint32_t _bf16x2_3[4];
                            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_bf16x2_3[0]), "=r"(_bf16x2_3[1]), "=r"(_bf16x2_3[2]), "=r"(_bf16x2_3[3]) : "r"(_smem_addr_3) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_delta_bufs_reg_1[0])[0]), "=f"((&_delta_bufs_reg_1[0])[1])
                                : "r"(_bf16x2_3[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_delta_bufs_reg_1[2])[0]), "=f"((&_delta_bufs_reg_1[2])[1])
                                : "r"(_bf16x2_3[1]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_delta_bufs_reg_1[4])[0]), "=f"((&_delta_bufs_reg_1[4])[1])
                                : "r"(_bf16x2_3[2]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_delta_bufs_reg_1[6])[0]), "=f"((&_delta_bufs_reg_1[6])[1])
                                : "r"(_bf16x2_3[3]));
                        }
                        #pragma unroll
                        for (int pair_idx_2 = 0; pair_idx_2 < 4; pair_idx_2++) {
                            const int value_idx_1 = pair_idx_2 * 2;
                            const int acc_idx_2 = slice_idx * 8 + value_idx_1;
                            float2 _f2_4 = make_float2(_prefix_bufs_reg_1[value_idx_1], _prefix_bufs_reg_1[value_idx_1 + 1]);
                            float2 _f2_5 = make_float2(_delta_bufs_reg_1[value_idx_1], _delta_bufs_reg_1[value_idx_1 + 1]);
                            float2 updated_pair_1 = add_f32x2_noftz(_f2_4, _f2_5);
                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(updated_pair_1.x);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                            updated[acc_idx_2] = _cvt_f32_2;
                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(updated_pair_1.y);
                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                            updated[acc_idx_2 + 1] = _cvt_f32_3;
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(updated[slice_idx * 8 + 0], updated[slice_idx * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(updated[slice_idx * 8 + 2], updated[slice_idx * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(updated[slice_idx * 8 + 4], updated[slice_idx * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(updated[slice_idx * 8 + 6], updated[slice_idx * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        #pragma unroll
                        for (int pair_idx_3 = 0; pair_idx_3 < 4; pair_idx_3++) {
                            const int acc_idx_3 = slice_idx * 8 + pair_idx_3 * 2;
                            float2 _f2_6 = make_float2(updated[acc_idx_3], updated[acc_idx_3 + 1]);
                            float2 value_pair_1 = _f2_6;
                            local_sum_pair = fma_f32x2_rn_ftz(value_pair_1, value_pair_1, local_sum_pair);
                        }
                    }
                }
                mbarrier_arrive(consumed_addr + (consumer_stage) * 8);
                if (token_1 == bid) {
                    mbarrier_wait(output_norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float local_sum_sq = local_sum_pair.x + local_sum_pair.y;
                float _warp_reduce_0 = local_sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                local_sum_sq = _warp_reduce_0;
                if (lane == 0) {
                    stats[consumer_warp] = local_sum_sq;
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                float total_sum_sq = ((lane < 8) ? stats[lane] : 0.0f);
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 4, 8);
                total_sum_sq += _shfl_down_0;
                float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 2, 8);
                total_sum_sq += _shfl_down_1;
                float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 1, 8);
                total_sum_sq += _shfl_down_2;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_0 = rsqrtf(total_sum_sq / 7168.0f + output_norm_eps);
                    output_rsigma = _rsqrt_0;
                }
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_0;
                #pragma unroll
                for (int slice_idx_1 = 0; slice_idx_1 < 4; slice_idx_1++) {
                    if (slice_idx_1 == 3) {
                        int hidden_base_2 = 6144 + group * 512 + thread_in_group * 4;
                        float _output_norm_buf_reg_0[4];
                        {
                            uint32_t _smem_addr_4 = output_norm_buf_addr + (hidden_base_2) * 2;
                            uint32_t _bf16x2_4[2];
                            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_bf16x2_4[0]), "=r"(_bf16x2_4[1]) : "r"(_smem_addr_4) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_0[0])[0]), "=f"((&_output_norm_buf_reg_0[0])[1])
                                : "r"(_bf16x2_4[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_0[2])[0]), "=f"((&_output_norm_buf_reg_0[2])[1])
                                : "r"(_bf16x2_4[1]));
                        }
                        float2 _f2_7 = make_float2(output_rsigma, output_rsigma);
                        float2 output_rsigma_pair = _f2_7;
                        #pragma unroll
                        for (int pair_idx_4 = 0; pair_idx_4 < 2; pair_idx_4++) {
                            const int value_idx_2 = pair_idx_4 * 2;
                            const int acc_idx_4 = slice_idx_1 * 8 + value_idx_2;
                            float2 _f2_8 = make_float2(updated[acc_idx_4], updated[acc_idx_4 + 1]);
                            float2 scaled_pair = mul_f32x2_noftz(_f2_8, output_rsigma_pair);
                            float2 _f2_9 = make_float2(_output_norm_buf_reg_0[value_idx_2], _output_norm_buf_reg_0[value_idx_2 + 1]);
                            float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_9);
                            updated[acc_idx_4] = normalized_pair.x;
                            updated[acc_idx_4 + 1] = normalized_pair.y;
                        }
                        {
                            uint2 _pk2;
                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                            _pk[0] = __floats2bfloat162_rn(updated[slice_idx_1 * 8 + 0], updated[slice_idx_1 * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(updated[slice_idx_1 * 8 + 2], updated[slice_idx_1 * 8 + 3]);
                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_2]) = _pk2;
                        }
                    } else {
                        int hidden_tile_1 = slice_idx_1 * 2 + group;
                        int hidden_base_3 = hidden_tile_1 * 1024 + thread_in_group * 8;
                        float _output_norm_buf_reg_1[8];
                        {
                            uint32_t _smem_addr_5 = output_norm_buf_addr + (hidden_base_3) * 2;
                            uint32_t _bf16x2_5[4];
                            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_bf16x2_5[0]), "=r"(_bf16x2_5[1]), "=r"(_bf16x2_5[2]), "=r"(_bf16x2_5[3]) : "r"(_smem_addr_5) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_1[0])[0]), "=f"((&_output_norm_buf_reg_1[0])[1])
                                : "r"(_bf16x2_5[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_1[2])[0]), "=f"((&_output_norm_buf_reg_1[2])[1])
                                : "r"(_bf16x2_5[1]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_1[4])[0]), "=f"((&_output_norm_buf_reg_1[4])[1])
                                : "r"(_bf16x2_5[2]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_1[6])[0]), "=f"((&_output_norm_buf_reg_1[6])[1])
                                : "r"(_bf16x2_5[3]));
                        }
                        float2 _f2_10 = make_float2(output_rsigma, output_rsigma);
                        float2 output_rsigma_pair_1 = _f2_10;
                        #pragma unroll
                        for (int pair_idx_5 = 0; pair_idx_5 < 4; pair_idx_5++) {
                            const int value_idx_3 = pair_idx_5 * 2;
                            const int acc_idx_5 = slice_idx_1 * 8 + value_idx_3;
                            float2 _f2_11 = make_float2(updated[acc_idx_5], updated[acc_idx_5 + 1]);
                            float2 scaled_pair_1 = mul_f32x2_noftz(_f2_11, output_rsigma_pair_1);
                            float2 _f2_12 = make_float2(_output_norm_buf_reg_1[value_idx_3], _output_norm_buf_reg_1[value_idx_3 + 1]);
                            float2 normalized_pair_1 = mul_f32x2_noftz(scaled_pair_1, _f2_12);
                            updated[acc_idx_5] = normalized_pair_1.x;
                            updated[acc_idx_5 + 1] = normalized_pair_1.y;
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(updated[slice_idx_1 * 8 + 0], updated[slice_idx_1 * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(updated[slice_idx_1 * 8 + 2], updated[slice_idx_1 * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(updated[slice_idx_1 * 8 + 4], updated[slice_idx_1 * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(updated[slice_idx_1 * 8 + 6], updated[slice_idx_1 * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_3 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; consumer_phase ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
