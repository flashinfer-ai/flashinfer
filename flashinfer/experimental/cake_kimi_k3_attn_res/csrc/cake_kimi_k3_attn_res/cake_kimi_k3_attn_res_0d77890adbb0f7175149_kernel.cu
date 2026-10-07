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
#define TMEM_NCOLS 192
#define TMEM__TMEM_VALUES_OFFSET 0
#define NUM_CHUNK_PIPE_STAGES 3
#define SMEM_V_BUFS_OFF 0
#define SMEM_V_BUFS_STAGE_BYTES 43008
#define SMEM_V_BUFS_STRIDE 43008
#define SMEM_DELTA_BUFS_OFF 129024
#define SMEM_DELTA_BUFS_STAGE_BYTES 14336
#define SMEM_DELTA_BUFS_STRIDE 14336
#define SMEM_OUTPUT_NORM_BUF_OFF 172032
#define SMEM_OUTPUT_NORM_BUF_STAGE_BYTES 14336
#define SMEM_OUTPUT_NORM_BUF_STRIDE 14336
#define SMEM_STATS_OFF 186368
#define SMEM_STATS_STAGE_BYTES 576
#define SMEM_STATS_STRIDE 576
#define SMEM_OUTPUT_STATS_OFF 186944
#define SMEM_OUTPUT_STATS_STAGE_BYTES 96
#define SMEM_OUTPUT_STATS_STRIDE 96
#define SMEM_TOTAL 187136
#define THREADS 288
#define NUM_BLOCKS 3
#define DEFER_TMEM_STORE_WAIT 1
#define EARLY_CONSUMED_RELEASE 1
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
}


__device__ __forceinline__ void tmem_st_x8_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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




__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_attn_res_0d77890adbb0f7175149(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 187040;
    #define ready_addr (mbar_base + 0)
    #define consumed_addr (mbar_base + 24)
    #define output_norm_ready_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* v_bufs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int v_bufs_addr = smem + 0;
    __nv_bfloat16* delta_bufs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int delta_bufs_addr = smem + 129024;
    __nv_bfloat16* output_norm_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 172032);
    const int output_norm_buf_addr = smem + 172032;
    float* stats = reinterpret_cast<float*>(smem_raw + 186368);
    const int stats_addr = smem + 186368;
    float* output_stats = reinterpret_cast<float*>(smem_raw + 186944);
    const int output_stats_addr = smem + 186944;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 7 barriers)
    // Mbarriers at smem_raw[187040..187096)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'chunk_pipe' ---
            // ready: 3 barriers, init_count=1
            mbarrier_init(smem + 187040, 1);
            mbarrier_init(smem + 187048, 1);
            mbarrier_init(smem + 187056, 1);
            // consumed: 3 barriers, init_count=256
            mbarrier_init(smem + 187064, 256);
            mbarrier_init(smem + 187072, 256);
            mbarrier_init(smem + 187080, 256);
            // output_norm_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 187088, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 187096);
    if (warp == 1) {
        int _tmem_hold = smem + 187096;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem__tmem_values = taddr;

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    mbarrier_arrive_expect_tx(output_norm_ready_addr, 14336);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(output_norm_buf_addr), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(output_norm_weight) + ((unsigned long long)0 * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(output_norm_ready_addr)
                        : "memory");
                }
            }
            unsigned int producer_stage = 0;
            unsigned int producer_phase = 0;
            #pragma unroll 1
            for (int token = bid; token < M; token += num_bids) {
                #pragma unroll
                for (int chunk = 0; chunk < (NUM_BLOCKS + 3) / 3; chunk++) {
                    const int source_base = chunk * 3;
                    const int remaining_sources = NUM_BLOCKS + 1 - source_base;
                    const int active_rows = ((remaining_sources > 3) ? 3 : remaining_sources);
                    mbarrier_wait(consumed_addr + (producer_stage) * 8, producer_phase ^ 1);
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        const int extra_delta_row = ((chunk == (NUM_BLOCKS + 3) / 3 - 1) ? 1 : 0);
                        mbarrier_arrive_expect_tx(ready_addr + (producer_stage) * 8, (active_rows + extra_delta_row) * 7168 * 2);
                        #pragma unroll
                        for (int source_in_chunk = 0; source_in_chunk < 3; source_in_chunk++) {
                            const int source = source_base + source_in_chunk;
                            if (source < NUM_BLOCKS + 1) {
                                if (source == NUM_BLOCKS) {
                                    asm volatile(
                                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                        "[%0], [%1], %2, [%3];"
                                        :: "r"(v_bufs_addr + (producer_stage * 3 * 7168 + (unsigned int)(source_in_chunk * 7168)) * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(prefix) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (producer_stage) * 8)
                                        : "memory");
                                    asm volatile(
                                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                        "[%0], [%1], %2, [%3];"
                                        :: "r"(delta_bufs_addr + producer_stage * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(delta) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (producer_stage) * 8)
                                        : "memory");
                                } else {
                                    asm volatile(
                                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                        "[%0], [%1], %2, [%3];"
                                        :: "r"(v_bufs_addr + (producer_stage * 3 * 7168 + (unsigned int)(source_in_chunk * 7168)) * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + (unsigned long long)source * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (producer_stage) * 8)
                                        : "memory");
                                }
                            }
                        }
                        if (NUM_BLOCKS == 8 && DEFER_TMEM_STORE_WAIT && EARLY_CONSUMED_RELEASE && !0 && !0 && !0 && chunk == 0 || NUM_BLOCKS == 4 && EARLY_CONSUMED_RELEASE && !0 && !0 && !0 && chunk == 0) {
                            if (token == bid) {
                                mbarrier_arrive_expect_tx(output_norm_ready_addr, 14336);
                                asm volatile(
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                    "[%0], [%1], %2, [%3];"
                                    :: "r"(output_norm_buf_addr), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(output_norm_weight) + ((unsigned long long)0 * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(output_norm_ready_addr)
                                    : "memory");
                            }
                        }
                    }
                    producer_stage += 1;
                    if (producer_stage == 3) { producer_stage = 0; producer_phase ^= 1; }
                }
            }
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    // ---- Role: consumers ----
    } else if (warp >= 1 && warp <= 8) {
        { // consumers_main
            int warp_id_in_role = (warp - 1);
            int consumer_warp = warp_id_in_role;
            int consumer_thread = consumer_warp * 32 + lane;
            int group = consumer_warp / 4;
            int thread_in_group = consumer_thread % 128;
            int group_tmem_base = taddr + (unsigned int)(group * 96);
            float q_cache[28];
            #pragma unroll
            for (int slice_idx = 0; slice_idx < 4; slice_idx++) {
                if (slice_idx == 3) {
                    int hidden_base = 6144 + group * 512 + thread_in_group * 4;
                    float _vec_load_0[4];
                    {
                        uint2 _vld_0;
                        _vld_0 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base);
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                    float _vec_load_1[4];
                    {
                        uint2 _vld_1;
                        _vld_1 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base);
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_1[0 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                    #pragma unroll
                    for (int pair_idx = 0; pair_idx < 2; pair_idx++) {
                        const int value_idx = pair_idx * 2;
                        const int cache_idx = slice_idx * 8 + value_idx;
                        float2 _f2_0 = make_float2(_vec_load_0[value_idx], _vec_load_0[value_idx + 1]);
                        float2 _f2_1 = make_float2(_vec_load_1[value_idx], _vec_load_1[value_idx + 1]);
                        float2 q_pair = mul_f32x2_noftz(_f2_0, _f2_1);
                        q_cache[cache_idx] = q_pair.x;
                        q_cache[cache_idx + 1] = q_pair.y;
                    }
                } else {
                    int hidden_tile = slice_idx * 2 + group;
                    int hidden_base_1 = hidden_tile * 1024 + thread_in_group * 8;
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_2[_pair]));
                            }
                        }
                    }
                    float _vec_load_3[8];
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1);
                        uint4 _vld_3[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_3[_blk] = _vptr_3[_blk];
                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_3[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int pair_idx_1 = 0; pair_idx_1 < 4; pair_idx_1++) {
                        const int value_idx_1 = pair_idx_1 * 2;
                        const int cache_idx_1 = slice_idx * 8 + value_idx_1;
                        float2 _f2_2 = make_float2(_vec_load_2[value_idx_1], _vec_load_2[value_idx_1 + 1]);
                        float2 _f2_3 = make_float2(_vec_load_3[value_idx_1], _vec_load_3[value_idx_1 + 1]);
                        float2 q_pair_1 = mul_f32x2_noftz(_f2_2, _f2_3);
                        q_cache[cache_idx_1] = q_pair_1.x;
                        q_cache[cache_idx_1 + 1] = q_pair_1.y;
                    }
                }
            }
            float block_carry[28];
            unsigned int consumer_stage = 0;
            unsigned int consumer_phase = 0;
            #pragma unroll 1
            for (int token_1 = bid; token_1 < M; token_1 += num_bids) {
                float acc32[28];
                #pragma unroll
                for (int value_idx_2 = 0; value_idx_2 < 28; value_idx_2++) {
                    acc32[value_idx_2] = 0.0f;
                }
                float m_running = -3.4028234663852886e+38f;
                float s_running = 0.0f;
                #pragma unroll
                for (int chunk_1 = 0; chunk_1 < (NUM_BLOCKS + 3) / 3; chunk_1++) {
                    const int source_base_1 = chunk_1 * 3;
                    const int remaining_sources_1 = NUM_BLOCKS + 1 - source_base_1;
                    const int active_rows_1 = ((remaining_sources_1 > 3) ? 3 : remaining_sources_1);
                    mbarrier_wait(ready_addr + (consumer_stage) * 8, consumer_phase);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    float2 local_sum_sq[3];
                    float2 local_dot[3];
                    #pragma unroll
                    for (int source_in_chunk_1 = 0; source_in_chunk_1 < 3; source_in_chunk_1++) {
                        float2 _f2_4 = make_float2(0.0f, 0.0f);
                        local_sum_sq[source_in_chunk_1] = _f2_4;
                        float2 _f2_5 = make_float2(0.0f, 0.0f);
                        local_dot[source_in_chunk_1] = _f2_5;
                    }
                    #pragma unroll
                    for (int slice_idx_1 = 0; slice_idx_1 < 4; slice_idx_1++) {
                        if (slice_idx_1 == 3) {
                            int hidden_base_2 = 6144 + group * 512 + thread_in_group * 4;
                            float values[8];
                            #pragma unroll
                            for (int source_in_chunk_2 = 0; source_in_chunk_2 < 3; source_in_chunk_2++) {
                                const int source_1 = source_base_1 + source_in_chunk_2;
                                if (source_1 < NUM_BLOCKS + 1) {
                                    unsigned int prefix_words[2];
                                    {
                                        float _v_bufs_reg_0[4];
                                        {
                                            uint32_t _smem_addr_4 = v_bufs_addr + (consumer_stage * 3 * 7168 + (unsigned int)(source_in_chunk_2 * 7168) + (unsigned int)hidden_base_2) * 2;
                                            uint32_t _bf16x2_4[2];
                                            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                                                : "=r"(_bf16x2_4[0]), "=r"(_bf16x2_4[1]) : "r"(_smem_addr_4) : "memory");
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_v_bufs_reg_0[0])[0]), "=f"((&_v_bufs_reg_0[0])[1])
                                                : "r"(_bf16x2_4[0]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_v_bufs_reg_0[2])[0]), "=f"((&_v_bufs_reg_0[2])[1])
                                                : "r"(_bf16x2_4[1]));
                                        }
                                        #pragma unroll
                                        for (int value_idx_3 = 0; value_idx_3 < 4; value_idx_3++) {
                                            values[value_idx_3] = _v_bufs_reg_0[value_idx_3];
                                        }
                                    }
                                    if (source_1 == NUM_BLOCKS) {
                                        float _delta_bufs_reg_0[4];
                                        {
                                            uint32_t _smem_addr_5 = delta_bufs_addr + (consumer_stage * 7168 + (unsigned int)hidden_base_2) * 2;
                                            uint32_t _bf16x2_5[2];
                                            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                                                : "=r"(_bf16x2_5[0]), "=r"(_bf16x2_5[1]) : "r"(_smem_addr_5) : "memory");
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_delta_bufs_reg_0[0])[0]), "=f"((&_delta_bufs_reg_0[0])[1])
                                                : "r"(_bf16x2_5[0]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_delta_bufs_reg_0[2])[0]), "=f"((&_delta_bufs_reg_0[2])[1])
                                                : "r"(_bf16x2_5[1]));
                                        }
                                        #pragma unroll
                                        for (int pair_idx_2 = 0; pair_idx_2 < 2; pair_idx_2++) {
                                            const int value_idx_4 = pair_idx_2 * 2;
                                            float2 _f2_6 = make_float2(values[value_idx_4], values[value_idx_4 + 1]);
                                            float2 _f2_7 = make_float2(_delta_bufs_reg_0[value_idx_4], _delta_bufs_reg_0[value_idx_4 + 1]);
                                            float2 updated_pair = add_f32x2_noftz(_f2_6, _f2_7);
                                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(updated_pair.x);
                                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                            values[value_idx_4] = _cvt_f32_0;
                                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(updated_pair.y);
                                            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                            values[value_idx_4 + 1] = _cvt_f32_1;
                                        }
                                        {
                                            uint2 _pk2;
                                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                            _pk[0] = __floats2bfloat162_rn(values[0 + 0], values[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(values[0 + 2], values[0 + 3]);
                                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_2]) = _pk2;
                                        }
                                    }
                                    #pragma unroll
                                    for (int pair_idx_3 = 0; pair_idx_3 < 2; pair_idx_3++) {
                                        const int value_idx_5 = pair_idx_3 * 2;
                                        float2 _f2_8 = make_float2(values[value_idx_5], values[value_idx_5 + 1]);
                                        float2 value_pair = _f2_8;
                                        float2 _f2_9 = make_float2(q_cache[slice_idx_1 * 8 + value_idx_5], q_cache[slice_idx_1 * 8 + value_idx_5 + 1]);
                                        float2 q_pair_2 = _f2_9;
                                        local_sum_sq[source_in_chunk_2] = fma_f32x2_rn_ftz(value_pair, value_pair, local_sum_sq[source_in_chunk_2]);
                                        local_dot[source_in_chunk_2] = fma_f32x2_rn_ftz(value_pair, q_pair_2, local_dot[source_in_chunk_2]);
                                    }
                                    {
                                        tmem_st_x4_f32(group_tmem_base + (slice_idx_1 * 3 + source_in_chunk_2) * 8, values);
                                    }
                                }
                            }
                        } else {
                            int hidden_tile_1 = slice_idx_1 * 2 + group;
                            int hidden_base_3 = hidden_tile_1 * 1024 + thread_in_group * 8;
                            float values_1[8];
                            #pragma unroll
                            for (int source_in_chunk_3 = 0; source_in_chunk_3 < 3; source_in_chunk_3++) {
                                const int source_2 = source_base_1 + source_in_chunk_3;
                                if (source_2 < NUM_BLOCKS + 1) {
                                    unsigned int prefix_words_1[4];
                                    {
                                        float _v_bufs_reg_1[8];
                                        {
                                            uint32_t _smem_addr_6 = v_bufs_addr + (consumer_stage * 3 * 7168 + (unsigned int)(source_in_chunk_3 * 7168) + (unsigned int)hidden_base_3) * 2;
                                            uint32_t _bf16x2_6[4];
                                            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(_bf16x2_6[0]), "=r"(_bf16x2_6[1]), "=r"(_bf16x2_6[2]), "=r"(_bf16x2_6[3]) : "r"(_smem_addr_6) : "memory");
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_v_bufs_reg_1[0])[0]), "=f"((&_v_bufs_reg_1[0])[1])
                                                : "r"(_bf16x2_6[0]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_v_bufs_reg_1[2])[0]), "=f"((&_v_bufs_reg_1[2])[1])
                                                : "r"(_bf16x2_6[1]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_v_bufs_reg_1[4])[0]), "=f"((&_v_bufs_reg_1[4])[1])
                                                : "r"(_bf16x2_6[2]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_v_bufs_reg_1[6])[0]), "=f"((&_v_bufs_reg_1[6])[1])
                                                : "r"(_bf16x2_6[3]));
                                        }
                                        #pragma unroll
                                        for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                                            values_1[value_idx_6] = _v_bufs_reg_1[value_idx_6];
                                        }
                                    }
                                    if (source_2 == NUM_BLOCKS) {
                                        float _delta_bufs_reg_1[8];
                                        {
                                            uint32_t _smem_addr_7 = delta_bufs_addr + (consumer_stage * 7168 + (unsigned int)hidden_base_3) * 2;
                                            uint32_t _bf16x2_7[4];
                                            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(_bf16x2_7[0]), "=r"(_bf16x2_7[1]), "=r"(_bf16x2_7[2]), "=r"(_bf16x2_7[3]) : "r"(_smem_addr_7) : "memory");
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_delta_bufs_reg_1[0])[0]), "=f"((&_delta_bufs_reg_1[0])[1])
                                                : "r"(_bf16x2_7[0]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_delta_bufs_reg_1[2])[0]), "=f"((&_delta_bufs_reg_1[2])[1])
                                                : "r"(_bf16x2_7[1]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_delta_bufs_reg_1[4])[0]), "=f"((&_delta_bufs_reg_1[4])[1])
                                                : "r"(_bf16x2_7[2]));
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_delta_bufs_reg_1[6])[0]), "=f"((&_delta_bufs_reg_1[6])[1])
                                                : "r"(_bf16x2_7[3]));
                                        }
                                        #pragma unroll
                                        for (int pair_idx_4 = 0; pair_idx_4 < 4; pair_idx_4++) {
                                            const int value_idx_7 = pair_idx_4 * 2;
                                            float2 _f2_10 = make_float2(values_1[value_idx_7], values_1[value_idx_7 + 1]);
                                            float2 _f2_11 = make_float2(_delta_bufs_reg_1[value_idx_7], _delta_bufs_reg_1[value_idx_7 + 1]);
                                            float2 updated_pair_1 = add_f32x2_noftz(_f2_10, _f2_11);
                                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(updated_pair_1.x);
                                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                            values_1[value_idx_7] = _cvt_f32_2;
                                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(updated_pair_1.y);
                                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                            values_1[value_idx_7 + 1] = _cvt_f32_3;
                                        }
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(values_1[0 + 0], values_1[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(values_1[0 + 2], values_1[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(values_1[0 + 4], values_1[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(values_1[0 + 6], values_1[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_3 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                    #pragma unroll
                                    for (int pair_idx_5 = 0; pair_idx_5 < 4; pair_idx_5++) {
                                        const int value_idx_8 = pair_idx_5 * 2;
                                        float2 _f2_12 = make_float2(values_1[value_idx_8], values_1[value_idx_8 + 1]);
                                        float2 value_pair_1 = _f2_12;
                                        float2 _f2_13 = make_float2(q_cache[slice_idx_1 * 8 + value_idx_8], q_cache[slice_idx_1 * 8 + value_idx_8 + 1]);
                                        float2 q_pair_3 = _f2_13;
                                        local_sum_sq[source_in_chunk_3] = fma_f32x2_rn_ftz(value_pair_1, value_pair_1, local_sum_sq[source_in_chunk_3]);
                                        local_dot[source_in_chunk_3] = fma_f32x2_rn_ftz(value_pair_1, q_pair_3, local_dot[source_in_chunk_3]);
                                    }
                                    {
                                        tmem_st_x8_f32(group_tmem_base + (slice_idx_1 * 3 + source_in_chunk_3) * 8, values_1);
                                    }
                                }
                            }
                        }
                    }
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(consumed_addr + (consumer_stage) * 8);
                    }
                    unsigned int stats_stage_base = consumer_stage * 48;
                    {
                        #pragma unroll
                        for (int source_in_chunk_4 = 0; source_in_chunk_4 < 3; source_in_chunk_4++) {
                            if (active_rows_1 > source_in_chunk_4) {
                                float2 _f2_20 = make_float2(local_sum_sq[source_in_chunk_4].x + local_sum_sq[source_in_chunk_4].y, local_dot[source_in_chunk_4].x + local_dot[source_in_chunk_4].y);
                                float2 reduce_pair = _f2_20;
                                unsigned long long packed = 0;
                                unsigned long long peer_packed = 0;
                                float2 _f2_21 = make_float2(0.0f, 0.0f);
                                float2 peer_pair = _f2_21;
                                packed = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                                unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, packed, 16);
                                peer_packed = _shfl_xor_5;
                                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                                packed = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                                unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, packed, 8);
                                peer_packed = _shfl_xor_6;
                                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                                packed = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                                unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, packed, 4);
                                peer_packed = _shfl_xor_7;
                                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                                packed = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                                unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, packed, 2);
                                peer_packed = _shfl_xor_8;
                                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                                packed = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                                unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, packed, 1);
                                peer_packed = _shfl_xor_9;
                                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                                if (lane == 0) {
                                    stats[stats_stage_base + (unsigned int)(((0) ? source_in_chunk_4 * 8 + consumer_warp : consumer_warp * 3 + source_in_chunk_4) * 2)] = reduce_pair.x;
                                    stats[stats_stage_base + (unsigned int)(((0) ? source_in_chunk_4 * 8 + consumer_warp : consumer_warp * 3 + source_in_chunk_4) * 2) + 1] = reduce_pair.y;
                                }
                            }
                        }
                    }
                    {
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    int stat_source = lane / 8;
                    int stat_warp = lane % 8;
                    float total_sum_sq = 0.0f;
                    float total_dot = 0.0f;
                    if (stat_source < active_rows_1) {
                        total_sum_sq = stats[stats_stage_base + (unsigned int)(((0) ? (int)lane : stat_warp * 3 + stat_source) * 2)];
                        total_dot = stats[stats_stage_base + (unsigned int)(((0) ? (int)lane : stat_warp * 3 + stat_source) * 2) + 1];
                    }
                    float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 4, 8);
                    total_sum_sq += _shfl_down_0;
                    float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
                    total_dot += _shfl_down_1;
                    float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 2, 8);
                    total_sum_sq += _shfl_down_2;
                    float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
                    total_dot += _shfl_down_3;
                    float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 1, 8);
                    total_sum_sq += _shfl_down_4;
                    float _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
                    total_dot += _shfl_down_5;
                    float local_logit = 0.0f;
                    if (stat_source < active_rows_1 && stat_warp == 0) {
                        float _rsqrt_0 = rsqrtf(total_sum_sq / 7168.0f + eps);
                        local_logit = total_dot * _rsqrt_0;
                    }
                    float logits[3];
                    #pragma unroll
                    for (int source_in_chunk_5 = 0; source_in_chunk_5 < 3; source_in_chunk_5++) {
                        logits[source_in_chunk_5] = 0.0f;
                        if (source_in_chunk_5 < 4) {
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, local_logit, source_in_chunk_5 * 8);
                            logits[source_in_chunk_5] = _shfl_0;
                        }
                    }
                    if (!EARLY_CONSUMED_RELEASE && chunk_1 != (NUM_BLOCKS + 3) / 3 - 1) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(consumed_addr + (consumer_stage) * 8);
                    }
                    float weights[3];
                    float new_max = 0.0f;
                    float correction = 0.0f;
                    float weight_sum = 0.0f;
                    #pragma unroll
                    for (int source_in_chunk_6 = 0; source_in_chunk_6 < 3; source_in_chunk_6++) {
                        weights[source_in_chunk_6] = 0.0f;
                    }
                    float chunk_max = -3.4028234663852886e+38f;
                    #pragma unroll
                    for (int source_in_chunk_7 = 0; source_in_chunk_7 < 3; source_in_chunk_7++) {
                        if (active_rows_1 > source_in_chunk_7) {
                            float _max_0 = max_noftz(chunk_max, logits[source_in_chunk_7]);
                            chunk_max = _max_0;
                        }
                    }
                    float _max_1 = max_noftz(m_running, chunk_max);
                    new_max = _max_1;
                    float _exp2_0 = approx_exp2((m_running - new_max) * 1.4426950408889634f);
                    correction = _exp2_0;
                    #pragma unroll
                    for (int source_in_chunk_8 = 0; source_in_chunk_8 < 3; source_in_chunk_8++) {
                        if (active_rows_1 > source_in_chunk_8) {
                            float _exp2_1 = approx_exp2((logits[source_in_chunk_8] - new_max) * 1.4426950408889634f);
                            weights[source_in_chunk_8] = _exp2_1;
                            weight_sum += weights[source_in_chunk_8];
                        }
                    }
                    #pragma unroll
                    for (int slice_idx_2 = 0; slice_idx_2 < 4; slice_idx_2++) {
                        if (slice_idx_2 == 3) {
                            float2 _f2_22 = make_float2(correction, correction);
                            float2 correction_pair = _f2_22;
                            #pragma unroll
                            for (int pair_idx_6 = 0; pair_idx_6 < 2; pair_idx_6++) {
                                const int acc_idx = slice_idx_2 * 8 + pair_idx_6 * 2;
                                float2 _f2_23 = make_float2(acc32[acc_idx], acc32[acc_idx + 1]);
                                float2 acc_pair = _f2_23;
                                float2 _mul_f32x2_0;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&acc_pair), "l"(*(const unsigned long long*)&correction_pair));
                                float2 scaled_pair = _mul_f32x2_0;
                                acc32[acc_idx] = scaled_pair.x;
                                acc32[acc_idx + 1] = scaled_pair.y;
                            }
                            #pragma unroll
                            for (int source_in_chunk_9 = 0; source_in_chunk_9 < 3; source_in_chunk_9++) {
                                if (active_rows_1 > source_in_chunk_9) {
                                    {
                                        float _tmem_load_0[4];
                                        tmem_ld_x4(&_tmem_load_0[0], group_tmem_base + (slice_idx_2 * 3 + source_in_chunk_9) * 8);
                                        float2 _f2_30 = make_float2(weights[source_in_chunk_9], weights[source_in_chunk_9]);
                                        float2 weight_pair = _f2_30;
                                        #pragma unroll
                                        for (int pair_idx_7 = 0; pair_idx_7 < 2; pair_idx_7++) {
                                            const int acc_idx_1 = slice_idx_2 * 8 + pair_idx_7 * 2;
                                            float2 _f2_31 = make_float2(_tmem_load_0[pair_idx_7 * 2], _tmem_load_0[pair_idx_7 * 2 + 1]);
                                            float2 cached_pair = _f2_31;
                                            float2 _f2_32 = make_float2(acc32[acc_idx_1], acc32[acc_idx_1 + 1]);
                                            float2 acc_pair_1 = _f2_32;
                                            float2 updated_pair_2 = fma_f32x2_rn_ftz(weight_pair, cached_pair, acc_pair_1);
                                            acc32[acc_idx_1] = updated_pair_2.x;
                                            acc32[acc_idx_1 + 1] = updated_pair_2.y;
                                        }
                                    }
                                }
                            }
                        } else {
                            float2 _f2_33 = make_float2(correction, correction);
                            float2 correction_pair_1 = _f2_33;
                            #pragma unroll
                            for (int pair_idx_8 = 0; pair_idx_8 < 4; pair_idx_8++) {
                                const int acc_idx_2 = slice_idx_2 * 8 + pair_idx_8 * 2;
                                float2 _f2_34 = make_float2(acc32[acc_idx_2], acc32[acc_idx_2 + 1]);
                                float2 acc_pair_2 = _f2_34;
                                float2 _mul_f32x2_1;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&acc_pair_2), "l"(*(const unsigned long long*)&correction_pair_1));
                                float2 scaled_pair_1 = _mul_f32x2_1;
                                acc32[acc_idx_2] = scaled_pair_1.x;
                                acc32[acc_idx_2 + 1] = scaled_pair_1.y;
                            }
                            #pragma unroll
                            for (int source_in_chunk_10 = 0; source_in_chunk_10 < 3; source_in_chunk_10++) {
                                if (active_rows_1 > source_in_chunk_10) {
                                    {
                                        float _tmem_load_1[8];
                                        tmem_ld_x8(&_tmem_load_1[0], group_tmem_base + (slice_idx_2 * 3 + source_in_chunk_10) * 8);
                                        float2 _f2_38 = make_float2(weights[source_in_chunk_10], weights[source_in_chunk_10]);
                                        float2 weight_pair_1 = _f2_38;
                                        #pragma unroll
                                        for (int pair_idx_9 = 0; pair_idx_9 < 4; pair_idx_9++) {
                                            const int acc_idx_3 = slice_idx_2 * 8 + pair_idx_9 * 2;
                                            float2 _f2_39 = make_float2(_tmem_load_1[pair_idx_9 * 2], _tmem_load_1[pair_idx_9 * 2 + 1]);
                                            float2 cached_pair_1 = _f2_39;
                                            float2 _f2_40 = make_float2(acc32[acc_idx_3], acc32[acc_idx_3 + 1]);
                                            float2 acc_pair_3 = _f2_40;
                                            float2 updated_pair_3 = fma_f32x2_rn_ftz(weight_pair_1, cached_pair_1, acc_pair_3);
                                            acc32[acc_idx_3] = updated_pair_3.x;
                                            acc32[acc_idx_3 + 1] = updated_pair_3.y;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    s_running = s_running * correction + weight_sum;
                    m_running = new_max;
                    if (EARLY_CONSUMED_RELEASE || chunk_1 != (NUM_BLOCKS + 3) / 3 - 1) {
                        consumer_stage += 1;
                        if (consumer_stage == 3) { consumer_stage = 0; consumer_phase ^= 1; }
                    }
                }
                if (token_1 == bid) {
                    mbarrier_wait(output_norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float output_norm_first_values[8];
                float2 _f2_41 = make_float2(0.0f, 0.0f);
                float2 local_output_sum_pair = _f2_41;
                #pragma unroll
                for (int pair_idx_10 = 0; pair_idx_10 < 14; pair_idx_10++) {
                    float2 _f2_42 = make_float2(acc32[pair_idx_10 * 2], acc32[pair_idx_10 * 2 + 1]);
                    float2 acc_pair_4 = _f2_42;
                    local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_4, acc_pair_4, local_output_sum_pair);
                }
                float local_output_sum_sq = local_output_sum_pair.x + local_output_sum_pair.y;
                float _warp_reduce_0 = local_output_sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                local_output_sum_sq = _warp_reduce_0;
                unsigned int output_stats_stage_base = consumer_stage * 8;
                if (lane == 0) {
                    output_stats[output_stats_stage_base + (unsigned int)consumer_warp] = local_output_sum_sq;
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                float total_output_sum_sq = ((lane < 8) ? output_stats[output_stats_stage_base + (unsigned int)lane] : 0.0f);
                float _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 4, 8);
                total_output_sum_sq += _shfl_down_12;
                float _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 2, 8);
                total_output_sum_sq += _shfl_down_13;
                float _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 1, 8);
                total_output_sum_sq += _shfl_down_14;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_2 = rsqrtf(total_output_sum_sq / 7168.0f + output_norm_eps * s_running * s_running);
                    output_rsigma = _rsqrt_2;
                }
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_2;
                #pragma unroll
                for (int slice_idx_3 = 0; slice_idx_3 < 4; slice_idx_3++) {
                    if (slice_idx_3 == 3) {
                        int hidden_base_4 = 6144 + group * 512 + thread_in_group * 4;
                        float _output_norm_buf_reg_1[4];
                        {
                            uint32_t _smem_addr_8 = output_norm_buf_addr + (hidden_base_4) * 2;
                            uint32_t _bf16x2_8[2];
                            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                                : "=r"(_bf16x2_8[0]), "=r"(_bf16x2_8[1]) : "r"(_smem_addr_8) : "memory");
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_1[0])[0]), "=f"((&_output_norm_buf_reg_1[0])[1])
                                : "r"(_bf16x2_8[0]));
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_output_norm_buf_reg_1[2])[0]), "=f"((&_output_norm_buf_reg_1[2])[1])
                                : "r"(_bf16x2_8[1]));
                        }
                        float2 _f2_43 = make_float2(output_rsigma, output_rsigma);
                        float2 output_rsigma_pair = _f2_43;
                        #pragma unroll
                        for (int pair_idx_11 = 0; pair_idx_11 < 2; pair_idx_11++) {
                            const int value_idx_9 = pair_idx_11 * 2;
                            const int acc_idx_4 = slice_idx_3 * 8 + value_idx_9;
                            float2 _f2_44 = make_float2(acc32[acc_idx_4], acc32[acc_idx_4 + 1]);
                            float2 scaled_pair_2 = mul_f32x2_noftz(_f2_44, output_rsigma_pair);
                            float2 _f2_45 = make_float2(_output_norm_buf_reg_1[value_idx_9], _output_norm_buf_reg_1[value_idx_9 + 1]);
                            float2 normalized_pair = mul_f32x2_noftz(scaled_pair_2, _f2_45);
                            acc32[acc_idx_4] = normalized_pair.x;
                            acc32[acc_idx_4 + 1] = normalized_pair.y;
                        }
                        {
                            uint2 _pk2;
                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                            _pk[0] = __floats2bfloat162_rn(acc32[slice_idx_3 * 8 + 0], acc32[slice_idx_3 * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(acc32[slice_idx_3 * 8 + 2], acc32[slice_idx_3 * 8 + 3]);
                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_4]) = _pk2;
                        }
                    } else {
                        int hidden_tile_2 = slice_idx_3 * 2 + group;
                        int hidden_base_5 = hidden_tile_2 * 1024 + thread_in_group * 8;
                        {
                            float _output_norm_buf_reg_3[8];
                            {
                                uint32_t _smem_addr_9 = output_norm_buf_addr + (hidden_base_5) * 2;
                                uint32_t _bf16x2_9[4];
                                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_bf16x2_9[0]), "=r"(_bf16x2_9[1]), "=r"(_bf16x2_9[2]), "=r"(_bf16x2_9[3]) : "r"(_smem_addr_9) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_output_norm_buf_reg_3[0])[0]), "=f"((&_output_norm_buf_reg_3[0])[1])
                                    : "r"(_bf16x2_9[0]));
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_output_norm_buf_reg_3[2])[0]), "=f"((&_output_norm_buf_reg_3[2])[1])
                                    : "r"(_bf16x2_9[1]));
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_output_norm_buf_reg_3[4])[0]), "=f"((&_output_norm_buf_reg_3[4])[1])
                                    : "r"(_bf16x2_9[2]));
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_output_norm_buf_reg_3[6])[0]), "=f"((&_output_norm_buf_reg_3[6])[1])
                                    : "r"(_bf16x2_9[3]));
                            }
                            float2 _f2_49 = make_float2(output_rsigma, output_rsigma);
                            float2 output_rsigma_pair_1 = _f2_49;
                            #pragma unroll
                            for (int pair_idx_12 = 0; pair_idx_12 < 4; pair_idx_12++) {
                                const int value_idx_10 = pair_idx_12 * 2;
                                const int acc_idx_5 = slice_idx_3 * 8 + value_idx_10;
                                float2 _f2_50 = make_float2(acc32[acc_idx_5], acc32[acc_idx_5 + 1]);
                                float2 scaled_pair_3 = mul_f32x2_noftz(_f2_50, output_rsigma_pair_1);
                                float2 _f2_51 = make_float2(_output_norm_buf_reg_3[value_idx_10], _output_norm_buf_reg_3[value_idx_10 + 1]);
                                float2 normalized_pair_1 = mul_f32x2_noftz(scaled_pair_3, _f2_51);
                                acc32[acc_idx_5] = normalized_pair_1.x;
                                acc32[acc_idx_5 + 1] = normalized_pair_1.y;
                            }
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(acc32[slice_idx_3 * 8 + 0], acc32[slice_idx_3 * 8 + 1]);
                                _pk[1] = __floats2bfloat162_rn(acc32[slice_idx_3 * 8 + 2], acc32[slice_idx_3 * 8 + 3]);
                                _pk[2] = __floats2bfloat162_rn(acc32[slice_idx_3 * 8 + 4], acc32[slice_idx_3 * 8 + 5]);
                                _pk[3] = __floats2bfloat162_rn(acc32[slice_idx_3 * 8 + 6], acc32[slice_idx_3 * 8 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_5 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            if (consumer_warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }

    // Cleanup
}

} // extern "C"
