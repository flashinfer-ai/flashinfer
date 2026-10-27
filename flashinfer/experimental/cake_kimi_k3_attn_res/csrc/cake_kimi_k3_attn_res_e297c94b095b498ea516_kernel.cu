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
#define TMEM_NCOLS 256
#define TMEM_VALUES_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SOURCE_BUF_OFF 0
#define SMEM_SOURCE_BUF_STAGE_BYTES 28672
#define SMEM_SOURCE_BUF_STRIDE 28672
#define SMEM_SOURCE_WORDS_OFF 0
#define SMEM_SOURCE_WORDS_STAGE_BYTES 114688
#define SMEM_SOURCE_WORDS_STRIDE 114688
#define SMEM_DELTA_BUF_OFF 114688
#define SMEM_DELTA_BUF_STAGE_BYTES 14336
#define SMEM_DELTA_BUF_STRIDE 14336
#define SMEM_DELTA_WORDS_OFF 114688
#define SMEM_DELTA_WORDS_STAGE_BYTES 28672
#define SMEM_DELTA_WORDS_STRIDE 28672
#define SMEM_OUTPUT_NORM_BUF_OFF 143360
#define SMEM_OUTPUT_NORM_BUF_STAGE_BYTES 14336
#define SMEM_OUTPUT_NORM_BUF_STRIDE 14336
#define SMEM_STATS_OFF 157696
#define SMEM_STATS_STAGE_BYTES 1024
#define SMEM_STATS_STRIDE 1024
#define SMEM_OUTPUT_STATS_OFF 158720
#define SMEM_OUTPUT_STATS_STAGE_BYTES 64
#define SMEM_OUTPUT_STATS_STRIDE 64
#define SMEM_TOTAL 158976
#define THREADS 384
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


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_attn_res_e297c94b095b498ea516(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 158784;
    #define ready_addr (mbar_base + 0)
    #define consumed_addr (mbar_base + 32)
    #define norm_ready_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* source_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int source_buf_addr = smem + 0;
    unsigned int* source_words = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int source_words_addr = smem + 0;
    __nv_bfloat16* delta_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 114688);
    const int delta_buf_addr = smem + 114688;
    unsigned int* delta_words = reinterpret_cast<unsigned int*>(smem_raw + 114688);
    const int delta_words_addr = smem + 114688;
    __nv_bfloat16* output_norm_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 143360);
    const int output_norm_buf_addr = smem + 143360;
    float* stats = reinterpret_cast<float*>(smem_raw + 157696);
    const int stats_addr = smem + 157696;
    float* output_stats = reinterpret_cast<float*>(smem_raw + 158720);
    const int output_stats_addr = smem + 158720;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[158784..158856)

    if (warp == 8) {
        uint32_t leader = elect_sync();
        if (leader) {
            // ready: 4 barriers, init_count=1
            mbarrier_init(smem + 158784, 1);
            mbarrier_init(smem + 158792, 1);
            mbarrier_init(smem + 158800, 1);
            mbarrier_init(smem + 158808, 1);
            // consumed: 4 barriers, init_count=128
            mbarrier_init(smem + 158816, 128);
            mbarrier_init(smem + 158824, 128);
            mbarrier_init(smem + 158832, 128);
            mbarrier_init(smem + 158840, 128);
            // norm_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 158848, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 158856);
    if (warp == 0) {
        int _tmem_hold = smem + 158856;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_values_tmem = taddr;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: producers ----
    if (warp >= 8 && warp <= 9) {
        { // producers_main
            unsigned int _phase_consumed = 1;
            if (elect_sync()) {
                int warp_id_in_role = warp - 8;
                int stream = warp_id_in_role;
                unsigned int stream_u = (unsigned int)stream;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                if (stream == 0) {
                    mbarrier_arrive_expect_tx(norm_ready_addr, 14336);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(output_norm_buf_addr), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(output_norm_weight) + ((unsigned long long)0 * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(norm_ready_addr)
                        : "memory");
                }
                unsigned int stage = 0;
                #pragma unroll 1
                for (int token = bid + stream * num_bids; token < M; token += 2 * num_bids) {
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, _phase_consumed);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(ready_addr + (stage + stream_u * 2) * 8, 28672);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + (stream_u * 2 + stage) * 2 * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + ((stream_u * 2 + stage) * 2 + 1) * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_consumed ^= 1; }
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, _phase_consumed);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(ready_addr + (stage + stream_u * 2) * 8, 28672);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + (stream_u * 2 + stage) * 2 * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + 2 * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + ((stream_u * 2 + stage) * 2 + 1) * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + 3 * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_consumed ^= 1; }
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, _phase_consumed);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(ready_addr + (stage + stream_u * 2) * 8, 28672);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + (stream_u * 2 + stage) * 2 * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + 4 * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + ((stream_u * 2 + stage) * 2 + 1) * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + 5 * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_consumed ^= 1; }
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, _phase_consumed);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(ready_addr + (stage + stream_u * 2) * 8, 28672);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + (stream_u * 2 + stage) * 2 * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(prefix) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(delta_buf_addr + stream_u * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(delta) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_consumed ^= 1; }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("barrier.sync 0, 320;" ::: "memory");
        }
    // ---- Role: consumers ----
    } else if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // consumers_main
            int warp_id_in_role_1 = warp - 0;
            int consumer_warp = warp_id_in_role_1;
            int stream_1 = consumer_warp / 4;
            unsigned int stream_u_1 = (unsigned int)stream_1;
            int warp_in_group = consumer_warp % 4;
            int thread_in_group = warp_in_group * 32 + lane;
            int tmem_base = taddr + (unsigned int)(stream_1 * 128) + (unsigned int)(warp % 4 * 32 << 16);
            int first_token = bid + stream_1 * num_bids;
            unsigned int stats_stream_base = stream_u_1 * 128;
            unsigned int output_stats_base = stream_u_1 * 8;
            float q_cache[56];
            int hidden_base = thread_in_group * 8;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(norm_weight + hidden_base);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
            float _vec_load_1[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(qk_weight + hidden_base);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_2 = make_float2(_vec_load_0[0], _vec_load_0[1]);
            float2 _f32x2_mul_b_3 = make_float2(_vec_load_1[0], _vec_load_1[1]);
            float2 q_pair;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
            q_cache[0] = q_pair.x;
            q_cache[1] = q_pair.y;
            float2 _f32x2_mul_a_4 = make_float2(_vec_load_0[2], _vec_load_0[3]);
            float2 _f32x2_mul_b_5 = make_float2(_vec_load_1[2], _vec_load_1[3]);
            float2 q_pair_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
            q_cache[2] = q_pair_0.x;
            q_cache[3] = q_pair_0.y;
            float2 _f32x2_mul_a_6 = make_float2(_vec_load_0[4], _vec_load_0[5]);
            float2 _f32x2_mul_b_7 = make_float2(_vec_load_1[4], _vec_load_1[5]);
            float2 q_pair_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
            q_cache[4] = q_pair_1.x;
            q_cache[5] = q_pair_1.y;
            float2 _f32x2_mul_a_8 = make_float2(_vec_load_0[6], _vec_load_0[7]);
            float2 _f32x2_mul_b_9 = make_float2(_vec_load_1[6], _vec_load_1[7]);
            float2 q_pair_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
            q_cache[6] = q_pair_2.x;
            q_cache[7] = q_pair_2.y;
            int hidden_base_3 = 2048 + thread_in_group * 8;
            float _vec_load_2[8];
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_3);
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
                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_10[_pair]));
                    }
                }
            }
            float _vec_load_3[8];
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_3);
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
                            : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_11[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_12 = make_float2(_vec_load_2[0], _vec_load_2[1]);
            float2 _f32x2_mul_b_13 = make_float2(_vec_load_3[0], _vec_load_3[1]);
            float2 q_pair_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
            q_cache[8] = q_pair_4.x;
            q_cache[9] = q_pair_4.y;
            float2 _f32x2_mul_a_14 = make_float2(_vec_load_2[2], _vec_load_2[3]);
            float2 _f32x2_mul_b_15 = make_float2(_vec_load_3[2], _vec_load_3[3]);
            float2 q_pair_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
            q_cache[10] = q_pair_5.x;
            q_cache[11] = q_pair_5.y;
            float2 _f32x2_mul_a_16 = make_float2(_vec_load_2[4], _vec_load_2[5]);
            float2 _f32x2_mul_b_17 = make_float2(_vec_load_3[4], _vec_load_3[5]);
            float2 q_pair_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
            q_cache[12] = q_pair_6.x;
            q_cache[13] = q_pair_6.y;
            float2 _f32x2_mul_a_18 = make_float2(_vec_load_2[6], _vec_load_2[7]);
            float2 _f32x2_mul_b_19 = make_float2(_vec_load_3[6], _vec_load_3[7]);
            float2 q_pair_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
            q_cache[14] = q_pair_7.x;
            q_cache[15] = q_pair_7.y;
            int hidden_base_8 = 4096 + thread_in_group * 8;
            float _vec_load_4[8];
            {
                const uint4* _vptr_20 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_8);
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
                            : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_20[_pair]));
                    }
                }
            }
            float _vec_load_5[8];
            {
                const uint4* _vptr_21 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_8);
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
                            : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_21[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_22 = make_float2(_vec_load_4[0], _vec_load_4[1]);
            float2 _f32x2_mul_b_23 = make_float2(_vec_load_5[0], _vec_load_5[1]);
            float2 q_pair_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
            q_cache[16] = q_pair_9.x;
            q_cache[17] = q_pair_9.y;
            float2 _f32x2_mul_a_24 = make_float2(_vec_load_4[2], _vec_load_4[3]);
            float2 _f32x2_mul_b_25 = make_float2(_vec_load_5[2], _vec_load_5[3]);
            float2 q_pair_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
            q_cache[18] = q_pair_10.x;
            q_cache[19] = q_pair_10.y;
            float2 _f32x2_mul_a_26 = make_float2(_vec_load_4[4], _vec_load_4[5]);
            float2 _f32x2_mul_b_27 = make_float2(_vec_load_5[4], _vec_load_5[5]);
            float2 q_pair_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
            q_cache[20] = q_pair_11.x;
            q_cache[21] = q_pair_11.y;
            float2 _f32x2_mul_a_28 = make_float2(_vec_load_4[6], _vec_load_4[7]);
            float2 _f32x2_mul_b_29 = make_float2(_vec_load_5[6], _vec_load_5[7]);
            float2 q_pair_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
            q_cache[22] = q_pair_12.x;
            q_cache[23] = q_pair_12.y;
            int hidden_base_13 = 6144 + thread_in_group * 4;
            float _vec_load_6[4];
            {
                uint2 _vld_30;
                _vld_30 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base_13);
                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_6[0 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _pair * 2])[1])
                        : "r"(_vpairs_30[_pair]));
                }
            }
            float _vec_load_7[4];
            {
                uint2 _vld_31;
                _vld_31 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base_13);
                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_7[0 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _pair * 2])[1])
                        : "r"(_vpairs_31[_pair]));
                }
            }
            float2 _f32x2_mul_a_32 = make_float2(_vec_load_6[0], _vec_load_6[1]);
            float2 _f32x2_mul_b_33 = make_float2(_vec_load_7[0], _vec_load_7[1]);
            float2 q_pair_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
            q_cache[24] = q_pair_14.x;
            q_cache[25] = q_pair_14.y;
            float2 _f32x2_mul_a_34 = make_float2(_vec_load_6[2], _vec_load_6[3]);
            float2 _f32x2_mul_b_35 = make_float2(_vec_load_7[2], _vec_load_7[3]);
            float2 q_pair_15;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
            q_cache[26] = q_pair_15.x;
            q_cache[27] = q_pair_15.y;
            int hidden_base_16 = 1024 + thread_in_group * 8;
            float _vec_load_8[8];
            {
                const uint4* _vptr_36 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_16);
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
                            : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_36[_pair]));
                    }
                }
            }
            float _vec_load_9[8];
            {
                const uint4* _vptr_37 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_16);
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
                            : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_37[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_38 = make_float2(_vec_load_8[0], _vec_load_8[1]);
            float2 _f32x2_mul_b_39 = make_float2(_vec_load_9[0], _vec_load_9[1]);
            float2 q_pair_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
            q_cache[28] = q_pair_17.x;
            q_cache[29] = q_pair_17.y;
            float2 _f32x2_mul_a_40 = make_float2(_vec_load_8[2], _vec_load_8[3]);
            float2 _f32x2_mul_b_41 = make_float2(_vec_load_9[2], _vec_load_9[3]);
            float2 q_pair_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
            q_cache[30] = q_pair_18.x;
            q_cache[31] = q_pair_18.y;
            float2 _f32x2_mul_a_42 = make_float2(_vec_load_8[4], _vec_load_8[5]);
            float2 _f32x2_mul_b_43 = make_float2(_vec_load_9[4], _vec_load_9[5]);
            float2 q_pair_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
            q_cache[32] = q_pair_19.x;
            q_cache[33] = q_pair_19.y;
            float2 _f32x2_mul_a_44 = make_float2(_vec_load_8[6], _vec_load_8[7]);
            float2 _f32x2_mul_b_45 = make_float2(_vec_load_9[6], _vec_load_9[7]);
            float2 q_pair_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
            q_cache[34] = q_pair_20.x;
            q_cache[35] = q_pair_20.y;
            int hidden_base_21 = 3072 + thread_in_group * 8;
            float _vec_load_10[8];
            {
                const uint4* _vptr_46 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_21);
                uint4 _vld_46[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_46[_blk] = _vptr_46[_blk];
                    uint32_t* _vpairs_46 = reinterpret_cast<uint32_t*>(&_vld_46[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_46[_pair]));
                    }
                }
            }
            float _vec_load_11[8];
            {
                const uint4* _vptr_47 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_21);
                uint4 _vld_47[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_47[_blk] = _vptr_47[_blk];
                    uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_47[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_48 = make_float2(_vec_load_10[0], _vec_load_10[1]);
            float2 _f32x2_mul_b_49 = make_float2(_vec_load_11[0], _vec_load_11[1]);
            float2 q_pair_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
            q_cache[36] = q_pair_22.x;
            q_cache[37] = q_pair_22.y;
            float2 _f32x2_mul_a_50 = make_float2(_vec_load_10[2], _vec_load_10[3]);
            float2 _f32x2_mul_b_51 = make_float2(_vec_load_11[2], _vec_load_11[3]);
            float2 q_pair_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            q_cache[38] = q_pair_23.x;
            q_cache[39] = q_pair_23.y;
            float2 _f32x2_mul_a_52 = make_float2(_vec_load_10[4], _vec_load_10[5]);
            float2 _f32x2_mul_b_53 = make_float2(_vec_load_11[4], _vec_load_11[5]);
            float2 q_pair_24;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            q_cache[40] = q_pair_24.x;
            q_cache[41] = q_pair_24.y;
            float2 _f32x2_mul_a_54 = make_float2(_vec_load_10[6], _vec_load_10[7]);
            float2 _f32x2_mul_b_55 = make_float2(_vec_load_11[6], _vec_load_11[7]);
            float2 q_pair_25;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
            q_cache[42] = q_pair_25.x;
            q_cache[43] = q_pair_25.y;
            int hidden_base_26 = 5120 + thread_in_group * 8;
            float _vec_load_12[8];
            {
                const uint4* _vptr_56 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_26);
                uint4 _vld_56[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_56[_blk] = _vptr_56[_blk];
                    uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_56[_pair]));
                    }
                }
            }
            float _vec_load_13[8];
            {
                const uint4* _vptr_57 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_26);
                uint4 _vld_57[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_57[_blk] = _vptr_57[_blk];
                    uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_57[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_58 = make_float2(_vec_load_12[0], _vec_load_12[1]);
            float2 _f32x2_mul_b_59 = make_float2(_vec_load_13[0], _vec_load_13[1]);
            float2 q_pair_27;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
            q_cache[44] = q_pair_27.x;
            q_cache[45] = q_pair_27.y;
            float2 _f32x2_mul_a_60 = make_float2(_vec_load_12[2], _vec_load_12[3]);
            float2 _f32x2_mul_b_61 = make_float2(_vec_load_13[2], _vec_load_13[3]);
            float2 q_pair_28;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
            q_cache[46] = q_pair_28.x;
            q_cache[47] = q_pair_28.y;
            float2 _f32x2_mul_a_62 = make_float2(_vec_load_12[4], _vec_load_12[5]);
            float2 _f32x2_mul_b_63 = make_float2(_vec_load_13[4], _vec_load_13[5]);
            float2 q_pair_29;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
            q_cache[48] = q_pair_29.x;
            q_cache[49] = q_pair_29.y;
            float2 _f32x2_mul_a_64 = make_float2(_vec_load_12[6], _vec_load_12[7]);
            float2 _f32x2_mul_b_65 = make_float2(_vec_load_13[6], _vec_load_13[7]);
            float2 q_pair_30;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
            q_cache[50] = q_pair_30.x;
            q_cache[51] = q_pair_30.y;
            int hidden_base_31 = 6656 + thread_in_group * 4;
            float _vec_load_14[4];
            {
                uint2 _vld_66;
                _vld_66 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base_31);
                uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_14[0 + _pair * 2])[0]), "=f"((&_vec_load_14[0 + _pair * 2])[1])
                        : "r"(_vpairs_66[_pair]));
                }
            }
            float _vec_load_15[4];
            {
                uint2 _vld_67;
                _vld_67 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base_31);
                uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_15[0 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _pair * 2])[1])
                        : "r"(_vpairs_67[_pair]));
                }
            }
            float2 _f32x2_mul_a_68 = make_float2(_vec_load_14[0], _vec_load_14[1]);
            float2 _f32x2_mul_b_69 = make_float2(_vec_load_15[0], _vec_load_15[1]);
            float2 q_pair_32;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
            q_cache[52] = q_pair_32.x;
            q_cache[53] = q_pair_32.y;
            float2 _f32x2_mul_a_70 = make_float2(_vec_load_14[2], _vec_load_14[3]);
            float2 _f32x2_mul_b_71 = make_float2(_vec_load_15[2], _vec_load_15[3]);
            float2 q_pair_33;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
            q_cache[54] = q_pair_33.x;
            q_cache[55] = q_pair_33.y;
            unsigned int consumer_stage = 0;
            unsigned int _phase_ready = 0;
            #pragma unroll 1
            for (int token_1 = bid + stream_1 * num_bids; token_1 < M; token_1 += 2 * num_bids) {
                float acc32[56];
                acc32[0] = 0.0f;
                acc32[1] = 0.0f;
                acc32[2] = 0.0f;
                acc32[3] = 0.0f;
                acc32[4] = 0.0f;
                acc32[5] = 0.0f;
                acc32[6] = 0.0f;
                acc32[7] = 0.0f;
                acc32[8] = 0.0f;
                acc32[9] = 0.0f;
                acc32[10] = 0.0f;
                acc32[11] = 0.0f;
                acc32[12] = 0.0f;
                acc32[13] = 0.0f;
                acc32[14] = 0.0f;
                acc32[15] = 0.0f;
                acc32[16] = 0.0f;
                acc32[17] = 0.0f;
                acc32[18] = 0.0f;
                acc32[19] = 0.0f;
                acc32[20] = 0.0f;
                acc32[21] = 0.0f;
                acc32[22] = 0.0f;
                acc32[23] = 0.0f;
                acc32[24] = 0.0f;
                acc32[25] = 0.0f;
                acc32[26] = 0.0f;
                acc32[27] = 0.0f;
                acc32[28] = 0.0f;
                acc32[29] = 0.0f;
                acc32[30] = 0.0f;
                acc32[31] = 0.0f;
                acc32[32] = 0.0f;
                acc32[33] = 0.0f;
                acc32[34] = 0.0f;
                acc32[35] = 0.0f;
                acc32[36] = 0.0f;
                acc32[37] = 0.0f;
                acc32[38] = 0.0f;
                acc32[39] = 0.0f;
                acc32[40] = 0.0f;
                acc32[41] = 0.0f;
                acc32[42] = 0.0f;
                acc32[43] = 0.0f;
                acc32[44] = 0.0f;
                acc32[45] = 0.0f;
                acc32[46] = 0.0f;
                acc32[47] = 0.0f;
                acc32[48] = 0.0f;
                acc32[49] = 0.0f;
                acc32[50] = 0.0f;
                acc32[51] = 0.0f;
                acc32[52] = 0.0f;
                acc32[53] = 0.0f;
                acc32[54] = 0.0f;
                acc32[55] = 0.0f;
                float m_running = -3.4028235e+38f;
                float s_running = 0.0f;
                unsigned int ring_stage = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage) * 8, _phase_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq[4];
                float2 local_dot[4];
                local_sum_sq[0] = make_float2(0.0f, 0.0f);
                local_dot[0] = make_float2(0.0f, 0.0f);
                local_sum_sq[1] = make_float2(0.0f, 0.0f);
                local_dot[1] = make_float2(0.0f, 0.0f);
                local_sum_sq[2] = make_float2(0.0f, 0.0f);
                local_dot[2] = make_float2(0.0f, 0.0f);
                local_sum_sq[3] = make_float2(0.0f, 0.0f);
                local_dot[3] = make_float2(0.0f, 0.0f);
                int hidden_base_0 = thread_in_group * 8;
                unsigned int smem_regs[4];
                #pragma unroll
                for (int _lr = 0; _lr < 4; _lr++) {
                    smem_regs[_lr] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_0 / 2) + (unsigned int)_lr];
                }
                float values[8];
                float _unpack_bf16_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_0[_pair * 2])[0]), "=f"((&_unpack_bf16_0[_pair * 2])[1])
                        : "r"(smem_regs[_pair]));
                }
                values[0] = _unpack_bf16_0[0];
                values[1] = _unpack_bf16_0[1];
                values[2] = _unpack_bf16_0[2];
                values[3] = _unpack_bf16_0[3];
                values[4] = _unpack_bf16_0[4];
                values[5] = _unpack_bf16_0[5];
                values[6] = _unpack_bf16_0[6];
                values[7] = _unpack_bf16_0[7];
                tmem_st_x8_f32(tmem_base, values);
                float2 value_pair = make_float2(values[0], values[1]);
                float2 q_pair_3 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair, value_pair, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair, q_pair_3, local_dot[0]);
                float2 value_pair_4 = make_float2(values[2], values[3]);
                float2 q_pair_8 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_4, value_pair_4, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_4, q_pair_8, local_dot[0]);
                float2 value_pair_9 = make_float2(values[4], values[5]);
                float2 q_pair_13 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_9, value_pair_9, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_9, q_pair_13, local_dot[0]);
                float2 value_pair_14 = make_float2(values[6], values[7]);
                float2 q_pair_16 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_14, value_pair_14, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_14, q_pair_16, local_dot[0]);
                unsigned int smem_regs_17[4];
                #pragma unroll
                for (int _lr_1 = 0; _lr_1 < 4; _lr_1++) {
                    smem_regs_17[_lr_1] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_0 / 2) + (unsigned int)_lr_1];
                }
                float values_18[8];
                float _unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
                        : "r"(smem_regs_17[_pair]));
                }
                values_18[0] = _unpack_bf16_1[0];
                values_18[1] = _unpack_bf16_1[1];
                values_18[2] = _unpack_bf16_1[2];
                values_18[3] = _unpack_bf16_1[3];
                values_18[4] = _unpack_bf16_1[4];
                values_18[5] = _unpack_bf16_1[5];
                values_18[6] = _unpack_bf16_1[6];
                values_18[7] = _unpack_bf16_1[7];
                tmem_st_x8_f32(tmem_base + 64, values_18);
                float2 value_pair_19 = make_float2(values_18[0], values_18[1]);
                float2 q_pair_21 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_19, value_pair_19, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_19, q_pair_21, local_dot[1]);
                float2 value_pair_22 = make_float2(values_18[2], values_18[3]);
                float2 q_pair_26 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_22, value_pair_22, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_22, q_pair_26, local_dot[1]);
                float2 value_pair_27 = make_float2(values_18[4], values_18[5]);
                float2 q_pair_31 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_27, value_pair_27, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_27, q_pair_31, local_dot[1]);
                float2 value_pair_32 = make_float2(values_18[6], values_18[7]);
                float2 q_pair_34 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_32, value_pair_32, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_32, q_pair_34, local_dot[1]);
                int hidden_base_35 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_36[4];
                #pragma unroll
                for (int _lr_2 = 0; _lr_2 < 4; _lr_2++) {
                    smem_regs_36[_lr_2] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_35 / 2) + (unsigned int)_lr_2];
                }
                float values_37[8];
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                        : "r"(smem_regs_36[_pair]));
                }
                values_37[0] = _unpack_bf16_2[0];
                values_37[1] = _unpack_bf16_2[1];
                values_37[2] = _unpack_bf16_2[2];
                values_37[3] = _unpack_bf16_2[3];
                values_37[4] = _unpack_bf16_2[4];
                values_37[5] = _unpack_bf16_2[5];
                values_37[6] = _unpack_bf16_2[6];
                values_37[7] = _unpack_bf16_2[7];
                tmem_st_x8_f32(tmem_base + 8, values_37);
                float2 value_pair_38 = make_float2(values_37[0], values_37[1]);
                float2 q_pair_39 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_38, value_pair_38, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_38, q_pair_39, local_dot[0]);
                float2 value_pair_40 = make_float2(values_37[2], values_37[3]);
                float2 q_pair_41 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_40, value_pair_40, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_40, q_pair_41, local_dot[0]);
                float2 value_pair_42 = make_float2(values_37[4], values_37[5]);
                float2 q_pair_43 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_42, value_pair_42, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_42, q_pair_43, local_dot[0]);
                float2 value_pair_44 = make_float2(values_37[6], values_37[7]);
                float2 q_pair_45 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_44, value_pair_44, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_44, q_pair_45, local_dot[0]);
                unsigned int smem_regs_46[4];
                #pragma unroll
                for (int _lr_3 = 0; _lr_3 < 4; _lr_3++) {
                    smem_regs_46[_lr_3] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_35 / 2) + (unsigned int)_lr_3];
                }
                float values_47[8];
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                        : "r"(smem_regs_46[_pair]));
                }
                values_47[0] = _unpack_bf16_3[0];
                values_47[1] = _unpack_bf16_3[1];
                values_47[2] = _unpack_bf16_3[2];
                values_47[3] = _unpack_bf16_3[3];
                values_47[4] = _unpack_bf16_3[4];
                values_47[5] = _unpack_bf16_3[5];
                values_47[6] = _unpack_bf16_3[6];
                values_47[7] = _unpack_bf16_3[7];
                tmem_st_x8_f32(tmem_base + 64 + 8, values_47);
                float2 value_pair_48 = make_float2(values_47[0], values_47[1]);
                float2 q_pair_49 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_48, value_pair_48, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_48, q_pair_49, local_dot[1]);
                float2 value_pair_50 = make_float2(values_47[2], values_47[3]);
                float2 q_pair_51 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_50, value_pair_50, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_50, q_pair_51, local_dot[1]);
                float2 value_pair_52 = make_float2(values_47[4], values_47[5]);
                float2 q_pair_53 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_52, value_pair_52, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_52, q_pair_53, local_dot[1]);
                float2 value_pair_54 = make_float2(values_47[6], values_47[7]);
                float2 q_pair_55 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_54, value_pair_54, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_54, q_pair_55, local_dot[1]);
                int hidden_base_56 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_57[4];
                #pragma unroll
                for (int _lr_4 = 0; _lr_4 < 4; _lr_4++) {
                    smem_regs_57[_lr_4] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_56 / 2) + (unsigned int)_lr_4];
                }
                float values_58[8];
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                        : "r"(smem_regs_57[_pair]));
                }
                values_58[0] = _unpack_bf16_4[0];
                values_58[1] = _unpack_bf16_4[1];
                values_58[2] = _unpack_bf16_4[2];
                values_58[3] = _unpack_bf16_4[3];
                values_58[4] = _unpack_bf16_4[4];
                values_58[5] = _unpack_bf16_4[5];
                values_58[6] = _unpack_bf16_4[6];
                values_58[7] = _unpack_bf16_4[7];
                tmem_st_x8_f32(tmem_base + 16, values_58);
                float2 value_pair_59 = make_float2(values_58[0], values_58[1]);
                float2 q_pair_60 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_59, value_pair_59, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_59, q_pair_60, local_dot[0]);
                float2 value_pair_61 = make_float2(values_58[2], values_58[3]);
                float2 q_pair_62 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_61, value_pair_61, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_61, q_pair_62, local_dot[0]);
                float2 value_pair_63 = make_float2(values_58[4], values_58[5]);
                float2 q_pair_64 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_63, value_pair_63, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_63, q_pair_64, local_dot[0]);
                float2 value_pair_65 = make_float2(values_58[6], values_58[7]);
                float2 q_pair_66 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_65, value_pair_65, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_65, q_pair_66, local_dot[0]);
                unsigned int smem_regs_67[4];
                #pragma unroll
                for (int _lr_5 = 0; _lr_5 < 4; _lr_5++) {
                    smem_regs_67[_lr_5] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_56 / 2) + (unsigned int)_lr_5];
                }
                float values_68[8];
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                        : "r"(smem_regs_67[_pair]));
                }
                values_68[0] = _unpack_bf16_5[0];
                values_68[1] = _unpack_bf16_5[1];
                values_68[2] = _unpack_bf16_5[2];
                values_68[3] = _unpack_bf16_5[3];
                values_68[4] = _unpack_bf16_5[4];
                values_68[5] = _unpack_bf16_5[5];
                values_68[6] = _unpack_bf16_5[6];
                values_68[7] = _unpack_bf16_5[7];
                tmem_st_x8_f32(tmem_base + 64 + 16, values_68);
                float2 value_pair_69 = make_float2(values_68[0], values_68[1]);
                float2 q_pair_70 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_69, value_pair_69, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_69, q_pair_70, local_dot[1]);
                float2 value_pair_71 = make_float2(values_68[2], values_68[3]);
                float2 q_pair_72 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_71, value_pair_71, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_71, q_pair_72, local_dot[1]);
                float2 value_pair_73 = make_float2(values_68[4], values_68[5]);
                float2 q_pair_74 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_73, value_pair_73, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_73, q_pair_74, local_dot[1]);
                float2 value_pair_75 = make_float2(values_68[6], values_68[7]);
                float2 q_pair_76 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_75, value_pair_75, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_75, q_pair_76, local_dot[1]);
                int hidden_base_77 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_78[2];
                #pragma unroll
                for (int _lr_6 = 0; _lr_6 < 2; _lr_6++) {
                    smem_regs_78[_lr_6] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_77 / 2) + (unsigned int)_lr_6];
                }
                float values_79[4];
                float _unpack_bf16_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                        : "r"(smem_regs_78[_pair]));
                }
                values_79[0] = _unpack_bf16_6[0];
                values_79[1] = _unpack_bf16_6[1];
                values_79[2] = _unpack_bf16_6[2];
                values_79[3] = _unpack_bf16_6[3];
                tmem_st_x4_f32(tmem_base + 24, values_79);
                float2 value_pair_80 = make_float2(values_79[0], values_79[1]);
                float2 q_pair_81 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_80, value_pair_80, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_80, q_pair_81, local_dot[0]);
                float2 value_pair_82 = make_float2(values_79[2], values_79[3]);
                float2 q_pair_83 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_82, value_pair_82, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_82, q_pair_83, local_dot[0]);
                unsigned int smem_regs_84[2];
                #pragma unroll
                for (int _lr_7 = 0; _lr_7 < 2; _lr_7++) {
                    smem_regs_84[_lr_7] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_77 / 2) + (unsigned int)_lr_7];
                }
                float values_85[4];
                float _unpack_bf16_7[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                        : "r"(smem_regs_84[_pair]));
                }
                values_85[0] = _unpack_bf16_7[0];
                values_85[1] = _unpack_bf16_7[1];
                values_85[2] = _unpack_bf16_7[2];
                values_85[3] = _unpack_bf16_7[3];
                tmem_st_x4_f32(tmem_base + 64 + 24, values_85);
                float2 value_pair_86 = make_float2(values_85[0], values_85[1]);
                float2 q_pair_87 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_86, value_pair_86, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_86, q_pair_87, local_dot[1]);
                float2 value_pair_88 = make_float2(values_85[2], values_85[3]);
                float2 q_pair_89 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_88, value_pair_88, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_88, q_pair_89, local_dot[1]);
                int hidden_base_90 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_91[4];
                #pragma unroll
                for (int _lr_8 = 0; _lr_8 < 4; _lr_8++) {
                    smem_regs_91[_lr_8] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_90 / 2) + (unsigned int)_lr_8];
                }
                float values_92[8];
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                        : "r"(smem_regs_91[_pair]));
                }
                values_92[0] = _unpack_bf16_8[0];
                values_92[1] = _unpack_bf16_8[1];
                values_92[2] = _unpack_bf16_8[2];
                values_92[3] = _unpack_bf16_8[3];
                values_92[4] = _unpack_bf16_8[4];
                values_92[5] = _unpack_bf16_8[5];
                values_92[6] = _unpack_bf16_8[6];
                values_92[7] = _unpack_bf16_8[7];
                tmem_st_x8_f32(tmem_base + 32, values_92);
                float2 value_pair_93 = make_float2(values_92[0], values_92[1]);
                float2 q_pair_94 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_93, value_pair_93, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_93, q_pair_94, local_dot[2]);
                float2 value_pair_95 = make_float2(values_92[2], values_92[3]);
                float2 q_pair_96 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_95, value_pair_95, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_95, q_pair_96, local_dot[2]);
                float2 value_pair_97 = make_float2(values_92[4], values_92[5]);
                float2 q_pair_98 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_97, value_pair_97, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_97, q_pair_98, local_dot[2]);
                float2 value_pair_99 = make_float2(values_92[6], values_92[7]);
                float2 q_pair_100 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_99, value_pair_99, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_99, q_pair_100, local_dot[2]);
                unsigned int smem_regs_101[4];
                #pragma unroll
                for (int _lr_9 = 0; _lr_9 < 4; _lr_9++) {
                    smem_regs_101[_lr_9] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_90 / 2) + (unsigned int)_lr_9];
                }
                float values_102[8];
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                        : "r"(smem_regs_101[_pair]));
                }
                values_102[0] = _unpack_bf16_9[0];
                values_102[1] = _unpack_bf16_9[1];
                values_102[2] = _unpack_bf16_9[2];
                values_102[3] = _unpack_bf16_9[3];
                values_102[4] = _unpack_bf16_9[4];
                values_102[5] = _unpack_bf16_9[5];
                values_102[6] = _unpack_bf16_9[6];
                values_102[7] = _unpack_bf16_9[7];
                tmem_st_x8_f32(tmem_base + 64 + 32, values_102);
                float2 value_pair_103 = make_float2(values_102[0], values_102[1]);
                float2 q_pair_104 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_103, value_pair_103, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_103, q_pair_104, local_dot[3]);
                float2 value_pair_105 = make_float2(values_102[2], values_102[3]);
                float2 q_pair_106 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_105, value_pair_105, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_105, q_pair_106, local_dot[3]);
                float2 value_pair_107 = make_float2(values_102[4], values_102[5]);
                float2 q_pair_108 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_107, value_pair_107, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_107, q_pair_108, local_dot[3]);
                float2 value_pair_109 = make_float2(values_102[6], values_102[7]);
                float2 q_pair_110 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_109, value_pair_109, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_109, q_pair_110, local_dot[3]);
                int hidden_base_111 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_112[4];
                #pragma unroll
                for (int _lr_10 = 0; _lr_10 < 4; _lr_10++) {
                    smem_regs_112[_lr_10] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_111 / 2) + (unsigned int)_lr_10];
                }
                float values_113[8];
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                        : "r"(smem_regs_112[_pair]));
                }
                values_113[0] = _unpack_bf16_10[0];
                values_113[1] = _unpack_bf16_10[1];
                values_113[2] = _unpack_bf16_10[2];
                values_113[3] = _unpack_bf16_10[3];
                values_113[4] = _unpack_bf16_10[4];
                values_113[5] = _unpack_bf16_10[5];
                values_113[6] = _unpack_bf16_10[6];
                values_113[7] = _unpack_bf16_10[7];
                tmem_st_x8_f32(tmem_base + 40, values_113);
                float2 value_pair_114 = make_float2(values_113[0], values_113[1]);
                float2 q_pair_115 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_114, value_pair_114, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_114, q_pair_115, local_dot[2]);
                float2 value_pair_116 = make_float2(values_113[2], values_113[3]);
                float2 q_pair_117 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_116, value_pair_116, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_116, q_pair_117, local_dot[2]);
                float2 value_pair_118 = make_float2(values_113[4], values_113[5]);
                float2 q_pair_119 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_118, value_pair_118, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_118, q_pair_119, local_dot[2]);
                float2 value_pair_120 = make_float2(values_113[6], values_113[7]);
                float2 q_pair_121 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_120, value_pair_120, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_120, q_pair_121, local_dot[2]);
                unsigned int smem_regs_122[4];
                #pragma unroll
                for (int _lr_11 = 0; _lr_11 < 4; _lr_11++) {
                    smem_regs_122[_lr_11] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_111 / 2) + (unsigned int)_lr_11];
                }
                float values_123[8];
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                        : "r"(smem_regs_122[_pair]));
                }
                values_123[0] = _unpack_bf16_11[0];
                values_123[1] = _unpack_bf16_11[1];
                values_123[2] = _unpack_bf16_11[2];
                values_123[3] = _unpack_bf16_11[3];
                values_123[4] = _unpack_bf16_11[4];
                values_123[5] = _unpack_bf16_11[5];
                values_123[6] = _unpack_bf16_11[6];
                values_123[7] = _unpack_bf16_11[7];
                tmem_st_x8_f32(tmem_base + 64 + 40, values_123);
                float2 value_pair_124 = make_float2(values_123[0], values_123[1]);
                float2 q_pair_125 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_124, value_pair_124, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_124, q_pair_125, local_dot[3]);
                float2 value_pair_126 = make_float2(values_123[2], values_123[3]);
                float2 q_pair_127 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_126, value_pair_126, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_126, q_pair_127, local_dot[3]);
                float2 value_pair_128 = make_float2(values_123[4], values_123[5]);
                float2 q_pair_129 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_128, value_pair_128, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_128, q_pair_129, local_dot[3]);
                float2 value_pair_130 = make_float2(values_123[6], values_123[7]);
                float2 q_pair_131 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_130, value_pair_130, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_130, q_pair_131, local_dot[3]);
                int hidden_base_132 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_133[4];
                #pragma unroll
                for (int _lr_12 = 0; _lr_12 < 4; _lr_12++) {
                    smem_regs_133[_lr_12] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_132 / 2) + (unsigned int)_lr_12];
                }
                float values_134[8];
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                        : "r"(smem_regs_133[_pair]));
                }
                values_134[0] = _unpack_bf16_12[0];
                values_134[1] = _unpack_bf16_12[1];
                values_134[2] = _unpack_bf16_12[2];
                values_134[3] = _unpack_bf16_12[3];
                values_134[4] = _unpack_bf16_12[4];
                values_134[5] = _unpack_bf16_12[5];
                values_134[6] = _unpack_bf16_12[6];
                values_134[7] = _unpack_bf16_12[7];
                tmem_st_x8_f32(tmem_base + 48, values_134);
                float2 value_pair_135 = make_float2(values_134[0], values_134[1]);
                float2 q_pair_136 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_135, value_pair_135, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_135, q_pair_136, local_dot[2]);
                float2 value_pair_137 = make_float2(values_134[2], values_134[3]);
                float2 q_pair_138 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_137, value_pair_137, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_137, q_pair_138, local_dot[2]);
                float2 value_pair_139 = make_float2(values_134[4], values_134[5]);
                float2 q_pair_140 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_139, value_pair_139, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_139, q_pair_140, local_dot[2]);
                float2 value_pair_141 = make_float2(values_134[6], values_134[7]);
                float2 q_pair_142 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_141, value_pair_141, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_141, q_pair_142, local_dot[2]);
                unsigned int smem_regs_143[4];
                #pragma unroll
                for (int _lr_13 = 0; _lr_13 < 4; _lr_13++) {
                    smem_regs_143[_lr_13] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_132 / 2) + (unsigned int)_lr_13];
                }
                float values_144[8];
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                        : "r"(smem_regs_143[_pair]));
                }
                values_144[0] = _unpack_bf16_13[0];
                values_144[1] = _unpack_bf16_13[1];
                values_144[2] = _unpack_bf16_13[2];
                values_144[3] = _unpack_bf16_13[3];
                values_144[4] = _unpack_bf16_13[4];
                values_144[5] = _unpack_bf16_13[5];
                values_144[6] = _unpack_bf16_13[6];
                values_144[7] = _unpack_bf16_13[7];
                tmem_st_x8_f32(tmem_base + 64 + 48, values_144);
                float2 value_pair_145 = make_float2(values_144[0], values_144[1]);
                float2 q_pair_146 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_145, value_pair_145, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_145, q_pair_146, local_dot[3]);
                float2 value_pair_147 = make_float2(values_144[2], values_144[3]);
                float2 q_pair_148 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_147, value_pair_147, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_147, q_pair_148, local_dot[3]);
                float2 value_pair_149 = make_float2(values_144[4], values_144[5]);
                float2 q_pair_150 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_149, value_pair_149, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_149, q_pair_150, local_dot[3]);
                float2 value_pair_151 = make_float2(values_144[6], values_144[7]);
                float2 q_pair_152 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_151, value_pair_151, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_151, q_pair_152, local_dot[3]);
                int hidden_base_153 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_154[2];
                #pragma unroll
                for (int _lr_14 = 0; _lr_14 < 2; _lr_14++) {
                    smem_regs_154[_lr_14] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_153 / 2) + (unsigned int)_lr_14];
                }
                float values_155[4];
                float _unpack_bf16_14[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                        : "r"(smem_regs_154[_pair]));
                }
                values_155[0] = _unpack_bf16_14[0];
                values_155[1] = _unpack_bf16_14[1];
                values_155[2] = _unpack_bf16_14[2];
                values_155[3] = _unpack_bf16_14[3];
                tmem_st_x4_f32(tmem_base + 56, values_155);
                float2 value_pair_156 = make_float2(values_155[0], values_155[1]);
                float2 q_pair_157 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_156, value_pair_156, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_156, q_pair_157, local_dot[2]);
                float2 value_pair_158 = make_float2(values_155[2], values_155[3]);
                float2 q_pair_159 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_158, value_pair_158, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_158, q_pair_159, local_dot[2]);
                unsigned int smem_regs_160[2];
                #pragma unroll
                for (int _lr_15 = 0; _lr_15 < 2; _lr_15++) {
                    smem_regs_160[_lr_15] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_153 / 2) + (unsigned int)_lr_15];
                }
                float values_161[4];
                float _unpack_bf16_15[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                        : "r"(smem_regs_160[_pair]));
                }
                values_161[0] = _unpack_bf16_15[0];
                values_161[1] = _unpack_bf16_15[1];
                values_161[2] = _unpack_bf16_15[2];
                values_161[3] = _unpack_bf16_15[3];
                tmem_st_x4_f32(tmem_base + 64 + 56, values_161);
                float2 value_pair_162 = make_float2(values_161[0], values_161[1]);
                float2 q_pair_163 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_162, value_pair_162, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_162, q_pair_163, local_dot[3]);
                float2 value_pair_164 = make_float2(values_161[2], values_161[3]);
                float2 q_pair_165 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_164, value_pair_164, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_164, q_pair_165, local_dot[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base = stats_stream_base;
                float2 reduce_pair = make_float2(local_sum_sq[0].x + local_sum_sq[0].y, local_dot[0].x + local_dot[0].y);
                unsigned long long packed_pair = 0;
                unsigned long long peer_packed = 0;
                float2 peer_pair = make_float2(0.0f, 0.0f);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 16);
                peer_packed = _shfl_xor_sync_72;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 8);
                peer_packed = _shfl_xor_sync_73;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 4);
                peer_packed = _shfl_xor_sync_74;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 2);
                peer_packed = _shfl_xor_sync_75;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 1);
                peer_packed = _shfl_xor_sync_76;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair.x;
                    stats[slot_base + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair.y;
                }
                float2 reduce_pair_166 = make_float2(local_sum_sq[1].x + local_sum_sq[1].y, local_dot[1].x + local_dot[1].y);
                unsigned long long packed_pair_167 = 0;
                unsigned long long peer_packed_168 = 0;
                float2 peer_pair_169 = make_float2(0.0f, 0.0f);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 16);
                peer_packed_168 = _shfl_xor_sync_77;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 8);
                peer_packed_168 = _shfl_xor_sync_78;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 4);
                peer_packed_168 = _shfl_xor_sync_79;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 2);
                peer_packed_168 = _shfl_xor_sync_80;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 1);
                peer_packed_168 = _shfl_xor_sync_81;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_166.x;
                    stats[slot_base + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_166.y;
                }
                float2 reduce_pair_170 = make_float2(local_sum_sq[2].x + local_sum_sq[2].y, local_dot[2].x + local_dot[2].y);
                unsigned long long packed_pair_171 = 0;
                unsigned long long peer_packed_172 = 0;
                float2 peer_pair_173 = make_float2(0.0f, 0.0f);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 16);
                peer_packed_172 = _shfl_xor_sync_82;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 8);
                peer_packed_172 = _shfl_xor_sync_83;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 4);
                peer_packed_172 = _shfl_xor_sync_84;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 2);
                peer_packed_172 = _shfl_xor_sync_85;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 1);
                peer_packed_172 = _shfl_xor_sync_86;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_170.x;
                    stats[slot_base + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_170.y;
                }
                float2 reduce_pair_174 = make_float2(local_sum_sq[3].x + local_sum_sq[3].y, local_dot[3].x + local_dot[3].y);
                unsigned long long packed_pair_175 = 0;
                unsigned long long peer_packed_176 = 0;
                float2 peer_pair_177 = make_float2(0.0f, 0.0f);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 16);
                peer_packed_176 = _shfl_xor_sync_87;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 8);
                peer_packed_176 = _shfl_xor_sync_88;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 4);
                peer_packed_176 = _shfl_xor_sync_89;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 2);
                peer_packed_176 = _shfl_xor_sync_90;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 1);
                peer_packed_176 = _shfl_xor_sync_91;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2)] = reduce_pair_174.x;
                    stats[slot_base + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2) + 1] = reduce_pair_174.y;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int stat_source = lane / 8;
                int stat_warp = lane % 8;
                float total_sum_sq = 0.0f;
                float total_dot = 0.0f;
                if (stat_source < 2) {
                    total_sum_sq = stats[slot_base + (unsigned int)((stat_warp * 2 + stat_source) * 2)];
                    total_dot = stats[slot_base + (unsigned int)((stat_warp * 2 + stat_source) * 2) + 1];
                }
                float _shfl_down_sync_92 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 4, 8);
                total_sum_sq += _shfl_down_sync_92;
                float _shfl_down_sync_93 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
                total_dot += _shfl_down_sync_93;
                float _shfl_down_sync_94 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 2, 8);
                total_sum_sq += _shfl_down_sync_94;
                float _shfl_down_sync_95 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
                total_dot += _shfl_down_sync_95;
                float _shfl_down_sync_96 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq, 1, 8);
                total_sum_sq += _shfl_down_sync_96;
                float _shfl_down_sync_97 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
                total_dot += _shfl_down_sync_97;
                float local_logit = 0.0f;
                if (stat_source < 2 && stat_warp == 0) {
                    float _rsqrt_98 = rsqrtf(total_sum_sq / 7168.0f + eps);
                    local_logit = total_dot * _rsqrt_98;
                }
                float logits[2];
                logits[0] = 0.0f;
                float _shfl_sync_99 = __shfl_sync(0xFFFFFFFF, local_logit, 0);
                logits[0] = _shfl_sync_99;
                logits[1] = 0.0f;
                float _shfl_sync_100 = __shfl_sync(0xFFFFFFFF, local_logit, 8);
                logits[1] = _shfl_sync_100;
                float weights[2];
                float new_max = 0.0f;
                float correction = 0.0f;
                float weight_sum = 0.0f;
                weights[0] = 0.0f;
                weights[1] = 0.0f;
                float chunk_max = -3.4028235e+38f;
                chunk_max = max_noftz(chunk_max, logits[0]);
                chunk_max = max_noftz(chunk_max, logits[1]);
                new_max = max_noftz(m_running, chunk_max);
                float _exp2_101 = approx_exp2((m_running - new_max) * 1.442695f);
                correction = _exp2_101;
                float _exp2_102 = approx_exp2((logits[0] - new_max) * 1.442695f);
                weights[0] = _exp2_102;
                weight_sum += weights[0];
                float _exp2_103 = approx_exp2((logits[1] - new_max) * 1.442695f);
                weights[1] = _exp2_103;
                weight_sum += weights[1];
                float2 correction_pair = make_float2(correction, correction);
                float2 acc_pair = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&acc_pair), "l"(*(const unsigned long long*)&correction_pair));
                acc32[0] = scaled_pair.x;
                acc32[1] = scaled_pair.y;
                float2 acc_pair_178 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_179;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_179) : "l"(*(const unsigned long long*)&acc_pair_178), "l"(*(const unsigned long long*)&correction_pair));
                acc32[2] = scaled_pair_179.x;
                acc32[3] = scaled_pair_179.y;
                float2 acc_pair_180 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_181;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_181) : "l"(*(const unsigned long long*)&acc_pair_180), "l"(*(const unsigned long long*)&correction_pair));
                acc32[4] = scaled_pair_181.x;
                acc32[5] = scaled_pair_181.y;
                float2 acc_pair_182 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_183;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_183) : "l"(*(const unsigned long long*)&acc_pair_182), "l"(*(const unsigned long long*)&correction_pair));
                acc32[6] = scaled_pair_183.x;
                acc32[7] = scaled_pair_183.y;
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], tmem_base);
                float2 weight_pair = make_float2(weights[0], weights[0]);
                float2 cached_pair = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                float2 acc_pair_184 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair = fma_f32x2_rn_ftz(weight_pair, cached_pair, acc_pair_184);
                acc32[0] = updated_pair.x;
                acc32[1] = updated_pair.y;
                float2 cached_pair_185 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
                float2 acc_pair_186 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_187 = fma_f32x2_rn_ftz(weight_pair, cached_pair_185, acc_pair_186);
                acc32[2] = updated_pair_187.x;
                acc32[3] = updated_pair_187.y;
                float2 cached_pair_188 = make_float2(_tmem_load_0[4], _tmem_load_0[5]);
                float2 acc_pair_189 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_190 = fma_f32x2_rn_ftz(weight_pair, cached_pair_188, acc_pair_189);
                acc32[4] = updated_pair_190.x;
                acc32[5] = updated_pair_190.y;
                float2 cached_pair_191 = make_float2(_tmem_load_0[6], _tmem_load_0[7]);
                float2 acc_pair_192 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_193 = fma_f32x2_rn_ftz(weight_pair, cached_pair_191, acc_pair_192);
                acc32[6] = updated_pair_193.x;
                acc32[7] = updated_pair_193.y;
                float _tmem_load_1[8];
                tmem_ld_x8(&_tmem_load_1[0], tmem_base + 64);
                float2 weight_pair_194 = make_float2(weights[1], weights[1]);
                float2 cached_pair_195 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                float2 acc_pair_196 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_197 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_195, acc_pair_196);
                acc32[0] = updated_pair_197.x;
                acc32[1] = updated_pair_197.y;
                float2 cached_pair_198 = make_float2(_tmem_load_1[2], _tmem_load_1[3]);
                float2 acc_pair_199 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_200 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_198, acc_pair_199);
                acc32[2] = updated_pair_200.x;
                acc32[3] = updated_pair_200.y;
                float2 cached_pair_201 = make_float2(_tmem_load_1[4], _tmem_load_1[5]);
                float2 acc_pair_202 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_203 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_201, acc_pair_202);
                acc32[4] = updated_pair_203.x;
                acc32[5] = updated_pair_203.y;
                float2 cached_pair_204 = make_float2(_tmem_load_1[6], _tmem_load_1[7]);
                float2 acc_pair_205 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_206 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_204, acc_pair_205);
                acc32[6] = updated_pair_206.x;
                acc32[7] = updated_pair_206.y;
                float2 correction_pair_207 = make_float2(correction, correction);
                float2 acc_pair_208 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_209;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_209) : "l"(*(const unsigned long long*)&acc_pair_208), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[8] = scaled_pair_209.x;
                acc32[9] = scaled_pair_209.y;
                float2 acc_pair_210 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_211;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_211) : "l"(*(const unsigned long long*)&acc_pair_210), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[10] = scaled_pair_211.x;
                acc32[11] = scaled_pair_211.y;
                float2 acc_pair_212 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_213;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_213) : "l"(*(const unsigned long long*)&acc_pair_212), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[12] = scaled_pair_213.x;
                acc32[13] = scaled_pair_213.y;
                float2 acc_pair_214 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_215;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_215) : "l"(*(const unsigned long long*)&acc_pair_214), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[14] = scaled_pair_215.x;
                acc32[15] = scaled_pair_215.y;
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], tmem_base + 8);
                float2 weight_pair_216 = make_float2(weights[0], weights[0]);
                float2 cached_pair_217 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                float2 acc_pair_218 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_219 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_217, acc_pair_218);
                acc32[8] = updated_pair_219.x;
                acc32[9] = updated_pair_219.y;
                float2 cached_pair_220 = make_float2(_tmem_load_2[2], _tmem_load_2[3]);
                float2 acc_pair_221 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_222 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_220, acc_pair_221);
                acc32[10] = updated_pair_222.x;
                acc32[11] = updated_pair_222.y;
                float2 cached_pair_223 = make_float2(_tmem_load_2[4], _tmem_load_2[5]);
                float2 acc_pair_224 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_225 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_223, acc_pair_224);
                acc32[12] = updated_pair_225.x;
                acc32[13] = updated_pair_225.y;
                float2 cached_pair_226 = make_float2(_tmem_load_2[6], _tmem_load_2[7]);
                float2 acc_pair_227 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_228 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_226, acc_pair_227);
                acc32[14] = updated_pair_228.x;
                acc32[15] = updated_pair_228.y;
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], tmem_base + 64 + 8);
                float2 weight_pair_229 = make_float2(weights[1], weights[1]);
                float2 cached_pair_230 = make_float2(_tmem_load_3[0], _tmem_load_3[1]);
                float2 acc_pair_231 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_232 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_230, acc_pair_231);
                acc32[8] = updated_pair_232.x;
                acc32[9] = updated_pair_232.y;
                float2 cached_pair_233 = make_float2(_tmem_load_3[2], _tmem_load_3[3]);
                float2 acc_pair_234 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_235 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_233, acc_pair_234);
                acc32[10] = updated_pair_235.x;
                acc32[11] = updated_pair_235.y;
                float2 cached_pair_236 = make_float2(_tmem_load_3[4], _tmem_load_3[5]);
                float2 acc_pair_237 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_238 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_236, acc_pair_237);
                acc32[12] = updated_pair_238.x;
                acc32[13] = updated_pair_238.y;
                float2 cached_pair_239 = make_float2(_tmem_load_3[6], _tmem_load_3[7]);
                float2 acc_pair_240 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_241 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_239, acc_pair_240);
                acc32[14] = updated_pair_241.x;
                acc32[15] = updated_pair_241.y;
                float2 correction_pair_242 = make_float2(correction, correction);
                float2 acc_pair_243 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_244;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_244) : "l"(*(const unsigned long long*)&acc_pair_243), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[16] = scaled_pair_244.x;
                acc32[17] = scaled_pair_244.y;
                float2 acc_pair_245 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_246;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_246) : "l"(*(const unsigned long long*)&acc_pair_245), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[18] = scaled_pair_246.x;
                acc32[19] = scaled_pair_246.y;
                float2 acc_pair_247 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_248;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_248) : "l"(*(const unsigned long long*)&acc_pair_247), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[20] = scaled_pair_248.x;
                acc32[21] = scaled_pair_248.y;
                float2 acc_pair_249 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_250;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_250) : "l"(*(const unsigned long long*)&acc_pair_249), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[22] = scaled_pair_250.x;
                acc32[23] = scaled_pair_250.y;
                float _tmem_load_4[8];
                tmem_ld_x8(&_tmem_load_4[0], tmem_base + 16);
                float2 weight_pair_251 = make_float2(weights[0], weights[0]);
                float2 cached_pair_252 = make_float2(_tmem_load_4[0], _tmem_load_4[1]);
                float2 acc_pair_253 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_254 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_252, acc_pair_253);
                acc32[16] = updated_pair_254.x;
                acc32[17] = updated_pair_254.y;
                float2 cached_pair_255 = make_float2(_tmem_load_4[2], _tmem_load_4[3]);
                float2 acc_pair_256 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_257 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_255, acc_pair_256);
                acc32[18] = updated_pair_257.x;
                acc32[19] = updated_pair_257.y;
                float2 cached_pair_258 = make_float2(_tmem_load_4[4], _tmem_load_4[5]);
                float2 acc_pair_259 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_260 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_258, acc_pair_259);
                acc32[20] = updated_pair_260.x;
                acc32[21] = updated_pair_260.y;
                float2 cached_pair_261 = make_float2(_tmem_load_4[6], _tmem_load_4[7]);
                float2 acc_pair_262 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_263 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_261, acc_pair_262);
                acc32[22] = updated_pair_263.x;
                acc32[23] = updated_pair_263.y;
                float _tmem_load_5[8];
                tmem_ld_x8(&_tmem_load_5[0], tmem_base + 64 + 16);
                float2 weight_pair_264 = make_float2(weights[1], weights[1]);
                float2 cached_pair_265 = make_float2(_tmem_load_5[0], _tmem_load_5[1]);
                float2 acc_pair_266 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_267 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_265, acc_pair_266);
                acc32[16] = updated_pair_267.x;
                acc32[17] = updated_pair_267.y;
                float2 cached_pair_268 = make_float2(_tmem_load_5[2], _tmem_load_5[3]);
                float2 acc_pair_269 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_270 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_268, acc_pair_269);
                acc32[18] = updated_pair_270.x;
                acc32[19] = updated_pair_270.y;
                float2 cached_pair_271 = make_float2(_tmem_load_5[4], _tmem_load_5[5]);
                float2 acc_pair_272 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_273 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_271, acc_pair_272);
                acc32[20] = updated_pair_273.x;
                acc32[21] = updated_pair_273.y;
                float2 cached_pair_274 = make_float2(_tmem_load_5[6], _tmem_load_5[7]);
                float2 acc_pair_275 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_276 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_274, acc_pair_275);
                acc32[22] = updated_pair_276.x;
                acc32[23] = updated_pair_276.y;
                float2 correction_pair_277 = make_float2(correction, correction);
                float2 acc_pair_278 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_279;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_279) : "l"(*(const unsigned long long*)&acc_pair_278), "l"(*(const unsigned long long*)&correction_pair_277));
                acc32[24] = scaled_pair_279.x;
                acc32[25] = scaled_pair_279.y;
                float2 acc_pair_280 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_281;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_281) : "l"(*(const unsigned long long*)&acc_pair_280), "l"(*(const unsigned long long*)&correction_pair_277));
                acc32[26] = scaled_pair_281.x;
                acc32[27] = scaled_pair_281.y;
                float _tmem_load_6[4];
                tmem_ld_x4(&_tmem_load_6[0], tmem_base + 24);
                float2 weight_pair_282 = make_float2(weights[0], weights[0]);
                float2 cached_pair_283 = make_float2(_tmem_load_6[0], _tmem_load_6[1]);
                float2 acc_pair_284 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_285 = fma_f32x2_rn_ftz(weight_pair_282, cached_pair_283, acc_pair_284);
                acc32[24] = updated_pair_285.x;
                acc32[25] = updated_pair_285.y;
                float2 cached_pair_286 = make_float2(_tmem_load_6[2], _tmem_load_6[3]);
                float2 acc_pair_287 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_288 = fma_f32x2_rn_ftz(weight_pair_282, cached_pair_286, acc_pair_287);
                acc32[26] = updated_pair_288.x;
                acc32[27] = updated_pair_288.y;
                float _tmem_load_7[4];
                tmem_ld_x4(&_tmem_load_7[0], tmem_base + 64 + 24);
                float2 weight_pair_289 = make_float2(weights[1], weights[1]);
                float2 cached_pair_290 = make_float2(_tmem_load_7[0], _tmem_load_7[1]);
                float2 acc_pair_291 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_292 = fma_f32x2_rn_ftz(weight_pair_289, cached_pair_290, acc_pair_291);
                acc32[24] = updated_pair_292.x;
                acc32[25] = updated_pair_292.y;
                float2 cached_pair_293 = make_float2(_tmem_load_7[2], _tmem_load_7[3]);
                float2 acc_pair_294 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_295 = fma_f32x2_rn_ftz(weight_pair_289, cached_pair_293, acc_pair_294);
                acc32[26] = updated_pair_295.x;
                acc32[27] = updated_pair_295.y;
                float2 correction_pair_296 = make_float2(correction, correction);
                float2 acc_pair_297 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_298;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_298) : "l"(*(const unsigned long long*)&acc_pair_297), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[28] = scaled_pair_298.x;
                acc32[29] = scaled_pair_298.y;
                float2 acc_pair_299 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_300;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_300) : "l"(*(const unsigned long long*)&acc_pair_299), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[30] = scaled_pair_300.x;
                acc32[31] = scaled_pair_300.y;
                float2 acc_pair_301 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_302;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_302) : "l"(*(const unsigned long long*)&acc_pair_301), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[32] = scaled_pair_302.x;
                acc32[33] = scaled_pair_302.y;
                float2 acc_pair_303 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_304;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_304) : "l"(*(const unsigned long long*)&acc_pair_303), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[34] = scaled_pair_304.x;
                acc32[35] = scaled_pair_304.y;
                float _tmem_load_8[8];
                tmem_ld_x8(&_tmem_load_8[0], tmem_base + 32);
                float2 weight_pair_305 = make_float2(weights[0], weights[0]);
                float2 cached_pair_306 = make_float2(_tmem_load_8[0], _tmem_load_8[1]);
                float2 acc_pair_307 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_308 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_306, acc_pair_307);
                acc32[28] = updated_pair_308.x;
                acc32[29] = updated_pair_308.y;
                float2 cached_pair_309 = make_float2(_tmem_load_8[2], _tmem_load_8[3]);
                float2 acc_pair_310 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_311 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_309, acc_pair_310);
                acc32[30] = updated_pair_311.x;
                acc32[31] = updated_pair_311.y;
                float2 cached_pair_312 = make_float2(_tmem_load_8[4], _tmem_load_8[5]);
                float2 acc_pair_313 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_314 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_312, acc_pair_313);
                acc32[32] = updated_pair_314.x;
                acc32[33] = updated_pair_314.y;
                float2 cached_pair_315 = make_float2(_tmem_load_8[6], _tmem_load_8[7]);
                float2 acc_pair_316 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_317 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_315, acc_pair_316);
                acc32[34] = updated_pair_317.x;
                acc32[35] = updated_pair_317.y;
                float _tmem_load_9[8];
                tmem_ld_x8(&_tmem_load_9[0], tmem_base + 64 + 32);
                float2 weight_pair_318 = make_float2(weights[1], weights[1]);
                float2 cached_pair_319 = make_float2(_tmem_load_9[0], _tmem_load_9[1]);
                float2 acc_pair_320 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_321 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_319, acc_pair_320);
                acc32[28] = updated_pair_321.x;
                acc32[29] = updated_pair_321.y;
                float2 cached_pair_322 = make_float2(_tmem_load_9[2], _tmem_load_9[3]);
                float2 acc_pair_323 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_324 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_322, acc_pair_323);
                acc32[30] = updated_pair_324.x;
                acc32[31] = updated_pair_324.y;
                float2 cached_pair_325 = make_float2(_tmem_load_9[4], _tmem_load_9[5]);
                float2 acc_pair_326 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_327 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_325, acc_pair_326);
                acc32[32] = updated_pair_327.x;
                acc32[33] = updated_pair_327.y;
                float2 cached_pair_328 = make_float2(_tmem_load_9[6], _tmem_load_9[7]);
                float2 acc_pair_329 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_330 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_328, acc_pair_329);
                acc32[34] = updated_pair_330.x;
                acc32[35] = updated_pair_330.y;
                float2 correction_pair_331 = make_float2(correction, correction);
                float2 acc_pair_332 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_333;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_333) : "l"(*(const unsigned long long*)&acc_pair_332), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[36] = scaled_pair_333.x;
                acc32[37] = scaled_pair_333.y;
                float2 acc_pair_334 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_335;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_335) : "l"(*(const unsigned long long*)&acc_pair_334), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[38] = scaled_pair_335.x;
                acc32[39] = scaled_pair_335.y;
                float2 acc_pair_336 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_337;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_337) : "l"(*(const unsigned long long*)&acc_pair_336), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[40] = scaled_pair_337.x;
                acc32[41] = scaled_pair_337.y;
                float2 acc_pair_338 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_339;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_339) : "l"(*(const unsigned long long*)&acc_pair_338), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[42] = scaled_pair_339.x;
                acc32[43] = scaled_pair_339.y;
                float _tmem_load_10[8];
                tmem_ld_x8(&_tmem_load_10[0], tmem_base + 40);
                float2 weight_pair_340 = make_float2(weights[0], weights[0]);
                float2 cached_pair_341 = make_float2(_tmem_load_10[0], _tmem_load_10[1]);
                float2 acc_pair_342 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_343 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_341, acc_pair_342);
                acc32[36] = updated_pair_343.x;
                acc32[37] = updated_pair_343.y;
                float2 cached_pair_344 = make_float2(_tmem_load_10[2], _tmem_load_10[3]);
                float2 acc_pair_345 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_346 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_344, acc_pair_345);
                acc32[38] = updated_pair_346.x;
                acc32[39] = updated_pair_346.y;
                float2 cached_pair_347 = make_float2(_tmem_load_10[4], _tmem_load_10[5]);
                float2 acc_pair_348 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_349 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_347, acc_pair_348);
                acc32[40] = updated_pair_349.x;
                acc32[41] = updated_pair_349.y;
                float2 cached_pair_350 = make_float2(_tmem_load_10[6], _tmem_load_10[7]);
                float2 acc_pair_351 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_352 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_350, acc_pair_351);
                acc32[42] = updated_pair_352.x;
                acc32[43] = updated_pair_352.y;
                float _tmem_load_11[8];
                tmem_ld_x8(&_tmem_load_11[0], tmem_base + 64 + 40);
                float2 weight_pair_353 = make_float2(weights[1], weights[1]);
                float2 cached_pair_354 = make_float2(_tmem_load_11[0], _tmem_load_11[1]);
                float2 acc_pair_355 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_356 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_354, acc_pair_355);
                acc32[36] = updated_pair_356.x;
                acc32[37] = updated_pair_356.y;
                float2 cached_pair_357 = make_float2(_tmem_load_11[2], _tmem_load_11[3]);
                float2 acc_pair_358 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_359 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_357, acc_pair_358);
                acc32[38] = updated_pair_359.x;
                acc32[39] = updated_pair_359.y;
                float2 cached_pair_360 = make_float2(_tmem_load_11[4], _tmem_load_11[5]);
                float2 acc_pair_361 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_362 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_360, acc_pair_361);
                acc32[40] = updated_pair_362.x;
                acc32[41] = updated_pair_362.y;
                float2 cached_pair_363 = make_float2(_tmem_load_11[6], _tmem_load_11[7]);
                float2 acc_pair_364 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_365 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_363, acc_pair_364);
                acc32[42] = updated_pair_365.x;
                acc32[43] = updated_pair_365.y;
                float2 correction_pair_366 = make_float2(correction, correction);
                float2 acc_pair_367 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_368;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_368) : "l"(*(const unsigned long long*)&acc_pair_367), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[44] = scaled_pair_368.x;
                acc32[45] = scaled_pair_368.y;
                float2 acc_pair_369 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_370;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_370) : "l"(*(const unsigned long long*)&acc_pair_369), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[46] = scaled_pair_370.x;
                acc32[47] = scaled_pair_370.y;
                float2 acc_pair_371 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_372;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_372) : "l"(*(const unsigned long long*)&acc_pair_371), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[48] = scaled_pair_372.x;
                acc32[49] = scaled_pair_372.y;
                float2 acc_pair_373 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_374;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_374) : "l"(*(const unsigned long long*)&acc_pair_373), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[50] = scaled_pair_374.x;
                acc32[51] = scaled_pair_374.y;
                float _tmem_load_12[8];
                tmem_ld_x8(&_tmem_load_12[0], tmem_base + 48);
                float2 weight_pair_375 = make_float2(weights[0], weights[0]);
                float2 cached_pair_376 = make_float2(_tmem_load_12[0], _tmem_load_12[1]);
                float2 acc_pair_377 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_378 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_376, acc_pair_377);
                acc32[44] = updated_pair_378.x;
                acc32[45] = updated_pair_378.y;
                float2 cached_pair_379 = make_float2(_tmem_load_12[2], _tmem_load_12[3]);
                float2 acc_pair_380 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_381 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_379, acc_pair_380);
                acc32[46] = updated_pair_381.x;
                acc32[47] = updated_pair_381.y;
                float2 cached_pair_382 = make_float2(_tmem_load_12[4], _tmem_load_12[5]);
                float2 acc_pair_383 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_384 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_382, acc_pair_383);
                acc32[48] = updated_pair_384.x;
                acc32[49] = updated_pair_384.y;
                float2 cached_pair_385 = make_float2(_tmem_load_12[6], _tmem_load_12[7]);
                float2 acc_pair_386 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_387 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_385, acc_pair_386);
                acc32[50] = updated_pair_387.x;
                acc32[51] = updated_pair_387.y;
                float _tmem_load_13[8];
                tmem_ld_x8(&_tmem_load_13[0], tmem_base + 64 + 48);
                float2 weight_pair_388 = make_float2(weights[1], weights[1]);
                float2 cached_pair_389 = make_float2(_tmem_load_13[0], _tmem_load_13[1]);
                float2 acc_pair_390 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_391 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_389, acc_pair_390);
                acc32[44] = updated_pair_391.x;
                acc32[45] = updated_pair_391.y;
                float2 cached_pair_392 = make_float2(_tmem_load_13[2], _tmem_load_13[3]);
                float2 acc_pair_393 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_394 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_392, acc_pair_393);
                acc32[46] = updated_pair_394.x;
                acc32[47] = updated_pair_394.y;
                float2 cached_pair_395 = make_float2(_tmem_load_13[4], _tmem_load_13[5]);
                float2 acc_pair_396 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_397 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_395, acc_pair_396);
                acc32[48] = updated_pair_397.x;
                acc32[49] = updated_pair_397.y;
                float2 cached_pair_398 = make_float2(_tmem_load_13[6], _tmem_load_13[7]);
                float2 acc_pair_399 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_400 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_398, acc_pair_399);
                acc32[50] = updated_pair_400.x;
                acc32[51] = updated_pair_400.y;
                float2 correction_pair_401 = make_float2(correction, correction);
                float2 acc_pair_402 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_403;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_403) : "l"(*(const unsigned long long*)&acc_pair_402), "l"(*(const unsigned long long*)&correction_pair_401));
                acc32[52] = scaled_pair_403.x;
                acc32[53] = scaled_pair_403.y;
                float2 acc_pair_404 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_405;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_405) : "l"(*(const unsigned long long*)&acc_pair_404), "l"(*(const unsigned long long*)&correction_pair_401));
                acc32[54] = scaled_pair_405.x;
                acc32[55] = scaled_pair_405.y;
                float _tmem_load_14[4];
                tmem_ld_x4(&_tmem_load_14[0], tmem_base + 56);
                float2 weight_pair_406 = make_float2(weights[0], weights[0]);
                float2 cached_pair_407 = make_float2(_tmem_load_14[0], _tmem_load_14[1]);
                float2 acc_pair_408 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_409 = fma_f32x2_rn_ftz(weight_pair_406, cached_pair_407, acc_pair_408);
                acc32[52] = updated_pair_409.x;
                acc32[53] = updated_pair_409.y;
                float2 cached_pair_410 = make_float2(_tmem_load_14[2], _tmem_load_14[3]);
                float2 acc_pair_411 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_412 = fma_f32x2_rn_ftz(weight_pair_406, cached_pair_410, acc_pair_411);
                acc32[54] = updated_pair_412.x;
                acc32[55] = updated_pair_412.y;
                float _tmem_load_15[4];
                tmem_ld_x4(&_tmem_load_15[0], tmem_base + 64 + 56);
                float2 weight_pair_413 = make_float2(weights[1], weights[1]);
                float2 cached_pair_414 = make_float2(_tmem_load_15[0], _tmem_load_15[1]);
                float2 acc_pair_415 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_416 = fma_f32x2_rn_ftz(weight_pair_413, cached_pair_414, acc_pair_415);
                acc32[52] = updated_pair_416.x;
                acc32[53] = updated_pair_416.y;
                float2 cached_pair_417 = make_float2(_tmem_load_15[2], _tmem_load_15[3]);
                float2 acc_pair_418 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_419 = fma_f32x2_rn_ftz(weight_pair_413, cached_pair_417, acc_pair_418);
                acc32[54] = updated_pair_419.x;
                acc32[55] = updated_pair_419.y;
                s_running = __fmaf_rn(s_running, correction, weight_sum);
                m_running = new_max;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; _phase_ready ^= 1; }
                unsigned int ring_stage_420 = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage_420) * 8, _phase_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq_421[4];
                float2 local_dot_422[4];
                local_sum_sq_421[0] = make_float2(0.0f, 0.0f);
                local_dot_422[0] = make_float2(0.0f, 0.0f);
                local_sum_sq_421[1] = make_float2(0.0f, 0.0f);
                local_dot_422[1] = make_float2(0.0f, 0.0f);
                local_sum_sq_421[2] = make_float2(0.0f, 0.0f);
                local_dot_422[2] = make_float2(0.0f, 0.0f);
                local_sum_sq_421[3] = make_float2(0.0f, 0.0f);
                local_dot_422[3] = make_float2(0.0f, 0.0f);
                int hidden_base_423 = thread_in_group * 8;
                unsigned int smem_regs_424[4];
                #pragma unroll
                for (int _lr_16 = 0; _lr_16 < 4; _lr_16++) {
                    smem_regs_424[_lr_16] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_423 / 2) + (unsigned int)_lr_16];
                }
                float values_425[8];
                float _unpack_bf16_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                        : "r"(smem_regs_424[_pair]));
                }
                values_425[0] = _unpack_bf16_16[0];
                values_425[1] = _unpack_bf16_16[1];
                values_425[2] = _unpack_bf16_16[2];
                values_425[3] = _unpack_bf16_16[3];
                values_425[4] = _unpack_bf16_16[4];
                values_425[5] = _unpack_bf16_16[5];
                values_425[6] = _unpack_bf16_16[6];
                values_425[7] = _unpack_bf16_16[7];
                tmem_st_x8_f32(tmem_base, values_425);
                float2 value_pair_426 = make_float2(values_425[0], values_425[1]);
                float2 q_pair_427 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_426, value_pair_426, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_426, q_pair_427, local_dot_422[0]);
                float2 value_pair_428 = make_float2(values_425[2], values_425[3]);
                float2 q_pair_429 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_428, value_pair_428, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_428, q_pair_429, local_dot_422[0]);
                float2 value_pair_430 = make_float2(values_425[4], values_425[5]);
                float2 q_pair_431 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_430, value_pair_430, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_430, q_pair_431, local_dot_422[0]);
                float2 value_pair_432 = make_float2(values_425[6], values_425[7]);
                float2 q_pair_433 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_432, value_pair_432, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_432, q_pair_433, local_dot_422[0]);
                unsigned int smem_regs_434[4];
                #pragma unroll
                for (int _lr_17 = 0; _lr_17 < 4; _lr_17++) {
                    smem_regs_434[_lr_17] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_423 / 2) + (unsigned int)_lr_17];
                }
                float values_435[8];
                float _unpack_bf16_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                        : "r"(smem_regs_434[_pair]));
                }
                values_435[0] = _unpack_bf16_17[0];
                values_435[1] = _unpack_bf16_17[1];
                values_435[2] = _unpack_bf16_17[2];
                values_435[3] = _unpack_bf16_17[3];
                values_435[4] = _unpack_bf16_17[4];
                values_435[5] = _unpack_bf16_17[5];
                values_435[6] = _unpack_bf16_17[6];
                values_435[7] = _unpack_bf16_17[7];
                tmem_st_x8_f32(tmem_base + 64, values_435);
                float2 value_pair_436 = make_float2(values_435[0], values_435[1]);
                float2 q_pair_437 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_436, value_pair_436, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_436, q_pair_437, local_dot_422[1]);
                float2 value_pair_438 = make_float2(values_435[2], values_435[3]);
                float2 q_pair_439 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_438, value_pair_438, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_438, q_pair_439, local_dot_422[1]);
                float2 value_pair_440 = make_float2(values_435[4], values_435[5]);
                float2 q_pair_441 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_440, value_pair_440, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_440, q_pair_441, local_dot_422[1]);
                float2 value_pair_442 = make_float2(values_435[6], values_435[7]);
                float2 q_pair_443 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_442, value_pair_442, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_442, q_pair_443, local_dot_422[1]);
                int hidden_base_444 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_445[4];
                #pragma unroll
                for (int _lr_18 = 0; _lr_18 < 4; _lr_18++) {
                    smem_regs_445[_lr_18] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_444 / 2) + (unsigned int)_lr_18];
                }
                float values_446[8];
                float _unpack_bf16_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                        : "r"(smem_regs_445[_pair]));
                }
                values_446[0] = _unpack_bf16_18[0];
                values_446[1] = _unpack_bf16_18[1];
                values_446[2] = _unpack_bf16_18[2];
                values_446[3] = _unpack_bf16_18[3];
                values_446[4] = _unpack_bf16_18[4];
                values_446[5] = _unpack_bf16_18[5];
                values_446[6] = _unpack_bf16_18[6];
                values_446[7] = _unpack_bf16_18[7];
                tmem_st_x8_f32(tmem_base + 8, values_446);
                float2 value_pair_447 = make_float2(values_446[0], values_446[1]);
                float2 q_pair_448 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_447, value_pair_447, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_447, q_pair_448, local_dot_422[0]);
                float2 value_pair_449 = make_float2(values_446[2], values_446[3]);
                float2 q_pair_450 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_449, value_pair_449, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_449, q_pair_450, local_dot_422[0]);
                float2 value_pair_451 = make_float2(values_446[4], values_446[5]);
                float2 q_pair_452 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_451, value_pair_451, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_451, q_pair_452, local_dot_422[0]);
                float2 value_pair_453 = make_float2(values_446[6], values_446[7]);
                float2 q_pair_454 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_453, value_pair_453, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_453, q_pair_454, local_dot_422[0]);
                unsigned int smem_regs_455[4];
                #pragma unroll
                for (int _lr_19 = 0; _lr_19 < 4; _lr_19++) {
                    smem_regs_455[_lr_19] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_444 / 2) + (unsigned int)_lr_19];
                }
                float values_456[8];
                float _unpack_bf16_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                        : "r"(smem_regs_455[_pair]));
                }
                values_456[0] = _unpack_bf16_19[0];
                values_456[1] = _unpack_bf16_19[1];
                values_456[2] = _unpack_bf16_19[2];
                values_456[3] = _unpack_bf16_19[3];
                values_456[4] = _unpack_bf16_19[4];
                values_456[5] = _unpack_bf16_19[5];
                values_456[6] = _unpack_bf16_19[6];
                values_456[7] = _unpack_bf16_19[7];
                tmem_st_x8_f32(tmem_base + 64 + 8, values_456);
                float2 value_pair_457 = make_float2(values_456[0], values_456[1]);
                float2 q_pair_458 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_457, value_pair_457, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_457, q_pair_458, local_dot_422[1]);
                float2 value_pair_459 = make_float2(values_456[2], values_456[3]);
                float2 q_pair_460 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_459, value_pair_459, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_459, q_pair_460, local_dot_422[1]);
                float2 value_pair_461 = make_float2(values_456[4], values_456[5]);
                float2 q_pair_462 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_461, value_pair_461, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_461, q_pair_462, local_dot_422[1]);
                float2 value_pair_463 = make_float2(values_456[6], values_456[7]);
                float2 q_pair_464 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_463, value_pair_463, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_463, q_pair_464, local_dot_422[1]);
                int hidden_base_465 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_466[4];
                #pragma unroll
                for (int _lr_20 = 0; _lr_20 < 4; _lr_20++) {
                    smem_regs_466[_lr_20] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_465 / 2) + (unsigned int)_lr_20];
                }
                float values_467[8];
                float _unpack_bf16_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                        : "r"(smem_regs_466[_pair]));
                }
                values_467[0] = _unpack_bf16_20[0];
                values_467[1] = _unpack_bf16_20[1];
                values_467[2] = _unpack_bf16_20[2];
                values_467[3] = _unpack_bf16_20[3];
                values_467[4] = _unpack_bf16_20[4];
                values_467[5] = _unpack_bf16_20[5];
                values_467[6] = _unpack_bf16_20[6];
                values_467[7] = _unpack_bf16_20[7];
                tmem_st_x8_f32(tmem_base + 16, values_467);
                float2 value_pair_468 = make_float2(values_467[0], values_467[1]);
                float2 q_pair_469 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_468, value_pair_468, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_468, q_pair_469, local_dot_422[0]);
                float2 value_pair_470 = make_float2(values_467[2], values_467[3]);
                float2 q_pair_471 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_470, value_pair_470, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_470, q_pair_471, local_dot_422[0]);
                float2 value_pair_472 = make_float2(values_467[4], values_467[5]);
                float2 q_pair_473 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_472, value_pair_472, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_472, q_pair_473, local_dot_422[0]);
                float2 value_pair_474 = make_float2(values_467[6], values_467[7]);
                float2 q_pair_475 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_474, value_pair_474, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_474, q_pair_475, local_dot_422[0]);
                unsigned int smem_regs_476[4];
                #pragma unroll
                for (int _lr_21 = 0; _lr_21 < 4; _lr_21++) {
                    smem_regs_476[_lr_21] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_465 / 2) + (unsigned int)_lr_21];
                }
                float values_477[8];
                float _unpack_bf16_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                        : "r"(smem_regs_476[_pair]));
                }
                values_477[0] = _unpack_bf16_21[0];
                values_477[1] = _unpack_bf16_21[1];
                values_477[2] = _unpack_bf16_21[2];
                values_477[3] = _unpack_bf16_21[3];
                values_477[4] = _unpack_bf16_21[4];
                values_477[5] = _unpack_bf16_21[5];
                values_477[6] = _unpack_bf16_21[6];
                values_477[7] = _unpack_bf16_21[7];
                tmem_st_x8_f32(tmem_base + 64 + 16, values_477);
                float2 value_pair_478 = make_float2(values_477[0], values_477[1]);
                float2 q_pair_479 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_478, value_pair_478, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_478, q_pair_479, local_dot_422[1]);
                float2 value_pair_480 = make_float2(values_477[2], values_477[3]);
                float2 q_pair_481 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_480, value_pair_480, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_480, q_pair_481, local_dot_422[1]);
                float2 value_pair_482 = make_float2(values_477[4], values_477[5]);
                float2 q_pair_483 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_482, value_pair_482, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_482, q_pair_483, local_dot_422[1]);
                float2 value_pair_484 = make_float2(values_477[6], values_477[7]);
                float2 q_pair_485 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_484, value_pair_484, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_484, q_pair_485, local_dot_422[1]);
                int hidden_base_486 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_487[2];
                #pragma unroll
                for (int _lr_22 = 0; _lr_22 < 2; _lr_22++) {
                    smem_regs_487[_lr_22] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_486 / 2) + (unsigned int)_lr_22];
                }
                float values_488[4];
                float _unpack_bf16_22[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                        : "r"(smem_regs_487[_pair]));
                }
                values_488[0] = _unpack_bf16_22[0];
                values_488[1] = _unpack_bf16_22[1];
                values_488[2] = _unpack_bf16_22[2];
                values_488[3] = _unpack_bf16_22[3];
                tmem_st_x4_f32(tmem_base + 24, values_488);
                float2 value_pair_489 = make_float2(values_488[0], values_488[1]);
                float2 q_pair_490 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_489, value_pair_489, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_489, q_pair_490, local_dot_422[0]);
                float2 value_pair_491 = make_float2(values_488[2], values_488[3]);
                float2 q_pair_492 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_491, value_pair_491, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_491, q_pair_492, local_dot_422[0]);
                unsigned int smem_regs_493[2];
                #pragma unroll
                for (int _lr_23 = 0; _lr_23 < 2; _lr_23++) {
                    smem_regs_493[_lr_23] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_486 / 2) + (unsigned int)_lr_23];
                }
                float values_494[4];
                float _unpack_bf16_23[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                        : "r"(smem_regs_493[_pair]));
                }
                values_494[0] = _unpack_bf16_23[0];
                values_494[1] = _unpack_bf16_23[1];
                values_494[2] = _unpack_bf16_23[2];
                values_494[3] = _unpack_bf16_23[3];
                tmem_st_x4_f32(tmem_base + 64 + 24, values_494);
                float2 value_pair_495 = make_float2(values_494[0], values_494[1]);
                float2 q_pair_496 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_495, value_pair_495, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_495, q_pair_496, local_dot_422[1]);
                float2 value_pair_497 = make_float2(values_494[2], values_494[3]);
                float2 q_pair_498 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_497, value_pair_497, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_497, q_pair_498, local_dot_422[1]);
                int hidden_base_499 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_500[4];
                #pragma unroll
                for (int _lr_24 = 0; _lr_24 < 4; _lr_24++) {
                    smem_regs_500[_lr_24] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_499 / 2) + (unsigned int)_lr_24];
                }
                float values_501[8];
                float _unpack_bf16_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                        : "r"(smem_regs_500[_pair]));
                }
                values_501[0] = _unpack_bf16_24[0];
                values_501[1] = _unpack_bf16_24[1];
                values_501[2] = _unpack_bf16_24[2];
                values_501[3] = _unpack_bf16_24[3];
                values_501[4] = _unpack_bf16_24[4];
                values_501[5] = _unpack_bf16_24[5];
                values_501[6] = _unpack_bf16_24[6];
                values_501[7] = _unpack_bf16_24[7];
                tmem_st_x8_f32(tmem_base + 32, values_501);
                float2 value_pair_502 = make_float2(values_501[0], values_501[1]);
                float2 q_pair_503 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_502, value_pair_502, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_502, q_pair_503, local_dot_422[2]);
                float2 value_pair_504 = make_float2(values_501[2], values_501[3]);
                float2 q_pair_505 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_504, value_pair_504, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_504, q_pair_505, local_dot_422[2]);
                float2 value_pair_506 = make_float2(values_501[4], values_501[5]);
                float2 q_pair_507 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_506, value_pair_506, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_506, q_pair_507, local_dot_422[2]);
                float2 value_pair_508 = make_float2(values_501[6], values_501[7]);
                float2 q_pair_509 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_508, value_pair_508, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_508, q_pair_509, local_dot_422[2]);
                unsigned int smem_regs_510[4];
                #pragma unroll
                for (int _lr_25 = 0; _lr_25 < 4; _lr_25++) {
                    smem_regs_510[_lr_25] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_499 / 2) + (unsigned int)_lr_25];
                }
                float values_511[8];
                float _unpack_bf16_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                        : "r"(smem_regs_510[_pair]));
                }
                values_511[0] = _unpack_bf16_25[0];
                values_511[1] = _unpack_bf16_25[1];
                values_511[2] = _unpack_bf16_25[2];
                values_511[3] = _unpack_bf16_25[3];
                values_511[4] = _unpack_bf16_25[4];
                values_511[5] = _unpack_bf16_25[5];
                values_511[6] = _unpack_bf16_25[6];
                values_511[7] = _unpack_bf16_25[7];
                tmem_st_x8_f32(tmem_base + 64 + 32, values_511);
                float2 value_pair_512 = make_float2(values_511[0], values_511[1]);
                float2 q_pair_513 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_512, value_pair_512, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_512, q_pair_513, local_dot_422[3]);
                float2 value_pair_514 = make_float2(values_511[2], values_511[3]);
                float2 q_pair_515 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_514, value_pair_514, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_514, q_pair_515, local_dot_422[3]);
                float2 value_pair_516 = make_float2(values_511[4], values_511[5]);
                float2 q_pair_517 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_516, value_pair_516, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_516, q_pair_517, local_dot_422[3]);
                float2 value_pair_518 = make_float2(values_511[6], values_511[7]);
                float2 q_pair_519 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_518, value_pair_518, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_518, q_pair_519, local_dot_422[3]);
                int hidden_base_520 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_521[4];
                #pragma unroll
                for (int _lr_26 = 0; _lr_26 < 4; _lr_26++) {
                    smem_regs_521[_lr_26] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_520 / 2) + (unsigned int)_lr_26];
                }
                float values_522[8];
                float _unpack_bf16_26[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                        : "r"(smem_regs_521[_pair]));
                }
                values_522[0] = _unpack_bf16_26[0];
                values_522[1] = _unpack_bf16_26[1];
                values_522[2] = _unpack_bf16_26[2];
                values_522[3] = _unpack_bf16_26[3];
                values_522[4] = _unpack_bf16_26[4];
                values_522[5] = _unpack_bf16_26[5];
                values_522[6] = _unpack_bf16_26[6];
                values_522[7] = _unpack_bf16_26[7];
                tmem_st_x8_f32(tmem_base + 40, values_522);
                float2 value_pair_523 = make_float2(values_522[0], values_522[1]);
                float2 q_pair_524 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_523, value_pair_523, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_523, q_pair_524, local_dot_422[2]);
                float2 value_pair_525 = make_float2(values_522[2], values_522[3]);
                float2 q_pair_526 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_525, value_pair_525, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_525, q_pair_526, local_dot_422[2]);
                float2 value_pair_527 = make_float2(values_522[4], values_522[5]);
                float2 q_pair_528 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_527, value_pair_527, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_527, q_pair_528, local_dot_422[2]);
                float2 value_pair_529 = make_float2(values_522[6], values_522[7]);
                float2 q_pair_530 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_529, value_pair_529, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_529, q_pair_530, local_dot_422[2]);
                unsigned int smem_regs_531[4];
                #pragma unroll
                for (int _lr_27 = 0; _lr_27 < 4; _lr_27++) {
                    smem_regs_531[_lr_27] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_520 / 2) + (unsigned int)_lr_27];
                }
                float values_532[8];
                float _unpack_bf16_27[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                        : "r"(smem_regs_531[_pair]));
                }
                values_532[0] = _unpack_bf16_27[0];
                values_532[1] = _unpack_bf16_27[1];
                values_532[2] = _unpack_bf16_27[2];
                values_532[3] = _unpack_bf16_27[3];
                values_532[4] = _unpack_bf16_27[4];
                values_532[5] = _unpack_bf16_27[5];
                values_532[6] = _unpack_bf16_27[6];
                values_532[7] = _unpack_bf16_27[7];
                tmem_st_x8_f32(tmem_base + 64 + 40, values_532);
                float2 value_pair_533 = make_float2(values_532[0], values_532[1]);
                float2 q_pair_534 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_533, value_pair_533, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_533, q_pair_534, local_dot_422[3]);
                float2 value_pair_535 = make_float2(values_532[2], values_532[3]);
                float2 q_pair_536 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_535, value_pair_535, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_535, q_pair_536, local_dot_422[3]);
                float2 value_pair_537 = make_float2(values_532[4], values_532[5]);
                float2 q_pair_538 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_537, value_pair_537, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_537, q_pair_538, local_dot_422[3]);
                float2 value_pair_539 = make_float2(values_532[6], values_532[7]);
                float2 q_pair_540 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_539, value_pair_539, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_539, q_pair_540, local_dot_422[3]);
                int hidden_base_541 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_542[4];
                #pragma unroll
                for (int _lr_28 = 0; _lr_28 < 4; _lr_28++) {
                    smem_regs_542[_lr_28] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_541 / 2) + (unsigned int)_lr_28];
                }
                float values_543[8];
                float _unpack_bf16_28[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                        : "r"(smem_regs_542[_pair]));
                }
                values_543[0] = _unpack_bf16_28[0];
                values_543[1] = _unpack_bf16_28[1];
                values_543[2] = _unpack_bf16_28[2];
                values_543[3] = _unpack_bf16_28[3];
                values_543[4] = _unpack_bf16_28[4];
                values_543[5] = _unpack_bf16_28[5];
                values_543[6] = _unpack_bf16_28[6];
                values_543[7] = _unpack_bf16_28[7];
                tmem_st_x8_f32(tmem_base + 48, values_543);
                float2 value_pair_544 = make_float2(values_543[0], values_543[1]);
                float2 q_pair_545 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_544, value_pair_544, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_544, q_pair_545, local_dot_422[2]);
                float2 value_pair_546 = make_float2(values_543[2], values_543[3]);
                float2 q_pair_547 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_546, value_pair_546, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_546, q_pair_547, local_dot_422[2]);
                float2 value_pair_548 = make_float2(values_543[4], values_543[5]);
                float2 q_pair_549 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_548, value_pair_548, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_548, q_pair_549, local_dot_422[2]);
                float2 value_pair_550 = make_float2(values_543[6], values_543[7]);
                float2 q_pair_551 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_550, value_pair_550, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_550, q_pair_551, local_dot_422[2]);
                unsigned int smem_regs_552[4];
                #pragma unroll
                for (int _lr_29 = 0; _lr_29 < 4; _lr_29++) {
                    smem_regs_552[_lr_29] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_541 / 2) + (unsigned int)_lr_29];
                }
                float values_553[8];
                float _unpack_bf16_29[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                        : "r"(smem_regs_552[_pair]));
                }
                values_553[0] = _unpack_bf16_29[0];
                values_553[1] = _unpack_bf16_29[1];
                values_553[2] = _unpack_bf16_29[2];
                values_553[3] = _unpack_bf16_29[3];
                values_553[4] = _unpack_bf16_29[4];
                values_553[5] = _unpack_bf16_29[5];
                values_553[6] = _unpack_bf16_29[6];
                values_553[7] = _unpack_bf16_29[7];
                tmem_st_x8_f32(tmem_base + 64 + 48, values_553);
                float2 value_pair_554 = make_float2(values_553[0], values_553[1]);
                float2 q_pair_555 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_554, value_pair_554, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_554, q_pair_555, local_dot_422[3]);
                float2 value_pair_556 = make_float2(values_553[2], values_553[3]);
                float2 q_pair_557 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_556, value_pair_556, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_556, q_pair_557, local_dot_422[3]);
                float2 value_pair_558 = make_float2(values_553[4], values_553[5]);
                float2 q_pair_559 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_558, value_pair_558, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_558, q_pair_559, local_dot_422[3]);
                float2 value_pair_560 = make_float2(values_553[6], values_553[7]);
                float2 q_pair_561 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_560, value_pair_560, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_560, q_pair_561, local_dot_422[3]);
                int hidden_base_562 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_563[2];
                #pragma unroll
                for (int _lr_30 = 0; _lr_30 < 2; _lr_30++) {
                    smem_regs_563[_lr_30] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_562 / 2) + (unsigned int)_lr_30];
                }
                float values_564[4];
                float _unpack_bf16_30[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                        : "r"(smem_regs_563[_pair]));
                }
                values_564[0] = _unpack_bf16_30[0];
                values_564[1] = _unpack_bf16_30[1];
                values_564[2] = _unpack_bf16_30[2];
                values_564[3] = _unpack_bf16_30[3];
                tmem_st_x4_f32(tmem_base + 56, values_564);
                float2 value_pair_565 = make_float2(values_564[0], values_564[1]);
                float2 q_pair_566 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_565, value_pair_565, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_565, q_pair_566, local_dot_422[2]);
                float2 value_pair_567 = make_float2(values_564[2], values_564[3]);
                float2 q_pair_568 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_567, value_pair_567, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_567, q_pair_568, local_dot_422[2]);
                unsigned int smem_regs_569[2];
                #pragma unroll
                for (int _lr_31 = 0; _lr_31 < 2; _lr_31++) {
                    smem_regs_569[_lr_31] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_562 / 2) + (unsigned int)_lr_31];
                }
                float values_570[4];
                float _unpack_bf16_31[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                        : "r"(smem_regs_569[_pair]));
                }
                values_570[0] = _unpack_bf16_31[0];
                values_570[1] = _unpack_bf16_31[1];
                values_570[2] = _unpack_bf16_31[2];
                values_570[3] = _unpack_bf16_31[3];
                tmem_st_x4_f32(tmem_base + 64 + 56, values_570);
                float2 value_pair_571 = make_float2(values_570[0], values_570[1]);
                float2 q_pair_572 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_571, value_pair_571, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_571, q_pair_572, local_dot_422[3]);
                float2 value_pair_573 = make_float2(values_570[2], values_570[3]);
                float2 q_pair_574 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_573, value_pair_573, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_573, q_pair_574, local_dot_422[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_420) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_575 = stats_stream_base + 32;
                float2 reduce_pair_576 = make_float2(local_sum_sq_421[0].x + local_sum_sq_421[0].y, local_dot_422[0].x + local_dot_422[0].y);
                unsigned long long packed_pair_577 = 0;
                unsigned long long peer_packed_578 = 0;
                float2 peer_pair_579 = make_float2(0.0f, 0.0f);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 16);
                peer_packed_578 = _shfl_xor_sync_104;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 8);
                peer_packed_578 = _shfl_xor_sync_105;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 4);
                peer_packed_578 = _shfl_xor_sync_106;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 2);
                peer_packed_578 = _shfl_xor_sync_107;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 1);
                peer_packed_578 = _shfl_xor_sync_108;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_576.x;
                    stats[slot_base_575 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_576.y;
                }
                float2 reduce_pair_580 = make_float2(local_sum_sq_421[1].x + local_sum_sq_421[1].y, local_dot_422[1].x + local_dot_422[1].y);
                unsigned long long packed_pair_581 = 0;
                unsigned long long peer_packed_582 = 0;
                float2 peer_pair_583 = make_float2(0.0f, 0.0f);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 16);
                peer_packed_582 = _shfl_xor_sync_109;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 8);
                peer_packed_582 = _shfl_xor_sync_110;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 4);
                peer_packed_582 = _shfl_xor_sync_111;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 2);
                peer_packed_582 = _shfl_xor_sync_112;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 1);
                peer_packed_582 = _shfl_xor_sync_113;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_580.x;
                    stats[slot_base_575 + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_580.y;
                }
                float2 reduce_pair_584 = make_float2(local_sum_sq_421[2].x + local_sum_sq_421[2].y, local_dot_422[2].x + local_dot_422[2].y);
                unsigned long long packed_pair_585 = 0;
                unsigned long long peer_packed_586 = 0;
                float2 peer_pair_587 = make_float2(0.0f, 0.0f);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 16);
                peer_packed_586 = _shfl_xor_sync_114;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 8);
                peer_packed_586 = _shfl_xor_sync_115;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 4);
                peer_packed_586 = _shfl_xor_sync_116;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 2);
                peer_packed_586 = _shfl_xor_sync_117;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 1);
                peer_packed_586 = _shfl_xor_sync_118;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_584.x;
                    stats[slot_base_575 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_584.y;
                }
                float2 reduce_pair_588 = make_float2(local_sum_sq_421[3].x + local_sum_sq_421[3].y, local_dot_422[3].x + local_dot_422[3].y);
                unsigned long long packed_pair_589 = 0;
                unsigned long long peer_packed_590 = 0;
                float2 peer_pair_591 = make_float2(0.0f, 0.0f);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 16);
                peer_packed_590 = _shfl_xor_sync_119;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 8);
                peer_packed_590 = _shfl_xor_sync_120;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 4);
                peer_packed_590 = _shfl_xor_sync_121;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 2);
                peer_packed_590 = _shfl_xor_sync_122;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 1);
                peer_packed_590 = _shfl_xor_sync_123;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2)] = reduce_pair_588.x;
                    stats[slot_base_575 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2) + 1] = reduce_pair_588.y;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int stat_source_592 = lane / 8;
                int stat_warp_593 = lane % 8;
                float total_sum_sq_594 = 0.0f;
                float total_dot_595 = 0.0f;
                if (stat_source_592 < 2) {
                    total_sum_sq_594 = stats[slot_base_575 + (unsigned int)((stat_warp_593 * 2 + stat_source_592) * 2)];
                    total_dot_595 = stats[slot_base_575 + (unsigned int)((stat_warp_593 * 2 + stat_source_592) * 2) + 1];
                }
                float _shfl_down_sync_124 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_594, 4, 8);
                total_sum_sq_594 += _shfl_down_sync_124;
                float _shfl_down_sync_125 = __shfl_down_sync(0xFFFFFFFF, total_dot_595, 4, 8);
                total_dot_595 += _shfl_down_sync_125;
                float _shfl_down_sync_126 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_594, 2, 8);
                total_sum_sq_594 += _shfl_down_sync_126;
                float _shfl_down_sync_127 = __shfl_down_sync(0xFFFFFFFF, total_dot_595, 2, 8);
                total_dot_595 += _shfl_down_sync_127;
                float _shfl_down_sync_128 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_594, 1, 8);
                total_sum_sq_594 += _shfl_down_sync_128;
                float _shfl_down_sync_129 = __shfl_down_sync(0xFFFFFFFF, total_dot_595, 1, 8);
                total_dot_595 += _shfl_down_sync_129;
                float local_logit_596 = 0.0f;
                if (stat_source_592 < 2 && stat_warp_593 == 0) {
                    float _rsqrt_130 = rsqrtf(total_sum_sq_594 / 7168.0f + eps);
                    local_logit_596 = total_dot_595 * _rsqrt_130;
                }
                float logits_597[2];
                logits_597[0] = 0.0f;
                float _shfl_sync_131 = __shfl_sync(0xFFFFFFFF, local_logit_596, 0);
                logits_597[0] = _shfl_sync_131;
                logits_597[1] = 0.0f;
                float _shfl_sync_132 = __shfl_sync(0xFFFFFFFF, local_logit_596, 8);
                logits_597[1] = _shfl_sync_132;
                float weights_598[2];
                float new_max_599 = 0.0f;
                float correction_600 = 0.0f;
                float weight_sum_601 = 0.0f;
                weights_598[0] = 0.0f;
                weights_598[1] = 0.0f;
                float chunk_max_602 = -3.4028235e+38f;
                chunk_max_602 = max_noftz(chunk_max_602, logits_597[0]);
                chunk_max_602 = max_noftz(chunk_max_602, logits_597[1]);
                new_max_599 = max_noftz(m_running, chunk_max_602);
                float _exp2_133 = approx_exp2((m_running - new_max_599) * 1.442695f);
                correction_600 = _exp2_133;
                float _exp2_134 = approx_exp2((logits_597[0] - new_max_599) * 1.442695f);
                weights_598[0] = _exp2_134;
                weight_sum_601 += weights_598[0];
                float _exp2_135 = approx_exp2((logits_597[1] - new_max_599) * 1.442695f);
                weights_598[1] = _exp2_135;
                weight_sum_601 += weights_598[1];
                float2 correction_pair_603 = make_float2(correction_600, correction_600);
                float2 acc_pair_604 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_605;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_605) : "l"(*(const unsigned long long*)&acc_pair_604), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[0] = scaled_pair_605.x;
                acc32[1] = scaled_pair_605.y;
                float2 acc_pair_606 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_607;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_607) : "l"(*(const unsigned long long*)&acc_pair_606), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[2] = scaled_pair_607.x;
                acc32[3] = scaled_pair_607.y;
                float2 acc_pair_608 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_609;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_609) : "l"(*(const unsigned long long*)&acc_pair_608), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[4] = scaled_pair_609.x;
                acc32[5] = scaled_pair_609.y;
                float2 acc_pair_610 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_611;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_611) : "l"(*(const unsigned long long*)&acc_pair_610), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[6] = scaled_pair_611.x;
                acc32[7] = scaled_pair_611.y;
                float _tmem_load_16[8];
                tmem_ld_x8(&_tmem_load_16[0], tmem_base);
                float2 weight_pair_612 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_613 = make_float2(_tmem_load_16[0], _tmem_load_16[1]);
                float2 acc_pair_614 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_615 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_613, acc_pair_614);
                acc32[0] = updated_pair_615.x;
                acc32[1] = updated_pair_615.y;
                float2 cached_pair_616 = make_float2(_tmem_load_16[2], _tmem_load_16[3]);
                float2 acc_pair_617 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_618 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_616, acc_pair_617);
                acc32[2] = updated_pair_618.x;
                acc32[3] = updated_pair_618.y;
                float2 cached_pair_619 = make_float2(_tmem_load_16[4], _tmem_load_16[5]);
                float2 acc_pair_620 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_621 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_619, acc_pair_620);
                acc32[4] = updated_pair_621.x;
                acc32[5] = updated_pair_621.y;
                float2 cached_pair_622 = make_float2(_tmem_load_16[6], _tmem_load_16[7]);
                float2 acc_pair_623 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_624 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_622, acc_pair_623);
                acc32[6] = updated_pair_624.x;
                acc32[7] = updated_pair_624.y;
                float _tmem_load_17[8];
                tmem_ld_x8(&_tmem_load_17[0], tmem_base + 64);
                float2 weight_pair_625 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_626 = make_float2(_tmem_load_17[0], _tmem_load_17[1]);
                float2 acc_pair_627 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_628 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_626, acc_pair_627);
                acc32[0] = updated_pair_628.x;
                acc32[1] = updated_pair_628.y;
                float2 cached_pair_629 = make_float2(_tmem_load_17[2], _tmem_load_17[3]);
                float2 acc_pair_630 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_631 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_629, acc_pair_630);
                acc32[2] = updated_pair_631.x;
                acc32[3] = updated_pair_631.y;
                float2 cached_pair_632 = make_float2(_tmem_load_17[4], _tmem_load_17[5]);
                float2 acc_pair_633 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_634 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_632, acc_pair_633);
                acc32[4] = updated_pair_634.x;
                acc32[5] = updated_pair_634.y;
                float2 cached_pair_635 = make_float2(_tmem_load_17[6], _tmem_load_17[7]);
                float2 acc_pair_636 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_637 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_635, acc_pair_636);
                acc32[6] = updated_pair_637.x;
                acc32[7] = updated_pair_637.y;
                float2 correction_pair_638 = make_float2(correction_600, correction_600);
                float2 acc_pair_639 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_640;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_640) : "l"(*(const unsigned long long*)&acc_pair_639), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[8] = scaled_pair_640.x;
                acc32[9] = scaled_pair_640.y;
                float2 acc_pair_641 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_642;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_642) : "l"(*(const unsigned long long*)&acc_pair_641), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[10] = scaled_pair_642.x;
                acc32[11] = scaled_pair_642.y;
                float2 acc_pair_643 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_644;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_644) : "l"(*(const unsigned long long*)&acc_pair_643), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[12] = scaled_pair_644.x;
                acc32[13] = scaled_pair_644.y;
                float2 acc_pair_645 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_646;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_646) : "l"(*(const unsigned long long*)&acc_pair_645), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[14] = scaled_pair_646.x;
                acc32[15] = scaled_pair_646.y;
                float _tmem_load_18[8];
                tmem_ld_x8(&_tmem_load_18[0], tmem_base + 8);
                float2 weight_pair_647 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_648 = make_float2(_tmem_load_18[0], _tmem_load_18[1]);
                float2 acc_pair_649 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_650 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_648, acc_pair_649);
                acc32[8] = updated_pair_650.x;
                acc32[9] = updated_pair_650.y;
                float2 cached_pair_651 = make_float2(_tmem_load_18[2], _tmem_load_18[3]);
                float2 acc_pair_652 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_653 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_651, acc_pair_652);
                acc32[10] = updated_pair_653.x;
                acc32[11] = updated_pair_653.y;
                float2 cached_pair_654 = make_float2(_tmem_load_18[4], _tmem_load_18[5]);
                float2 acc_pair_655 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_656 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_654, acc_pair_655);
                acc32[12] = updated_pair_656.x;
                acc32[13] = updated_pair_656.y;
                float2 cached_pair_657 = make_float2(_tmem_load_18[6], _tmem_load_18[7]);
                float2 acc_pair_658 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_659 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_657, acc_pair_658);
                acc32[14] = updated_pair_659.x;
                acc32[15] = updated_pair_659.y;
                float _tmem_load_19[8];
                tmem_ld_x8(&_tmem_load_19[0], tmem_base + 64 + 8);
                float2 weight_pair_660 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_661 = make_float2(_tmem_load_19[0], _tmem_load_19[1]);
                float2 acc_pair_662 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_663 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_661, acc_pair_662);
                acc32[8] = updated_pair_663.x;
                acc32[9] = updated_pair_663.y;
                float2 cached_pair_664 = make_float2(_tmem_load_19[2], _tmem_load_19[3]);
                float2 acc_pair_665 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_666 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_664, acc_pair_665);
                acc32[10] = updated_pair_666.x;
                acc32[11] = updated_pair_666.y;
                float2 cached_pair_667 = make_float2(_tmem_load_19[4], _tmem_load_19[5]);
                float2 acc_pair_668 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_669 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_667, acc_pair_668);
                acc32[12] = updated_pair_669.x;
                acc32[13] = updated_pair_669.y;
                float2 cached_pair_670 = make_float2(_tmem_load_19[6], _tmem_load_19[7]);
                float2 acc_pair_671 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_672 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_670, acc_pair_671);
                acc32[14] = updated_pair_672.x;
                acc32[15] = updated_pair_672.y;
                float2 correction_pair_673 = make_float2(correction_600, correction_600);
                float2 acc_pair_674 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_675;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_675) : "l"(*(const unsigned long long*)&acc_pair_674), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[16] = scaled_pair_675.x;
                acc32[17] = scaled_pair_675.y;
                float2 acc_pair_676 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_677;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_677) : "l"(*(const unsigned long long*)&acc_pair_676), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[18] = scaled_pair_677.x;
                acc32[19] = scaled_pair_677.y;
                float2 acc_pair_678 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_679;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_679) : "l"(*(const unsigned long long*)&acc_pair_678), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[20] = scaled_pair_679.x;
                acc32[21] = scaled_pair_679.y;
                float2 acc_pair_680 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_681;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_681) : "l"(*(const unsigned long long*)&acc_pair_680), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[22] = scaled_pair_681.x;
                acc32[23] = scaled_pair_681.y;
                float _tmem_load_20[8];
                tmem_ld_x8(&_tmem_load_20[0], tmem_base + 16);
                float2 weight_pair_682 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_683 = make_float2(_tmem_load_20[0], _tmem_load_20[1]);
                float2 acc_pair_684 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_685 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_683, acc_pair_684);
                acc32[16] = updated_pair_685.x;
                acc32[17] = updated_pair_685.y;
                float2 cached_pair_686 = make_float2(_tmem_load_20[2], _tmem_load_20[3]);
                float2 acc_pair_687 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_688 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_686, acc_pair_687);
                acc32[18] = updated_pair_688.x;
                acc32[19] = updated_pair_688.y;
                float2 cached_pair_689 = make_float2(_tmem_load_20[4], _tmem_load_20[5]);
                float2 acc_pair_690 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_691 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_689, acc_pair_690);
                acc32[20] = updated_pair_691.x;
                acc32[21] = updated_pair_691.y;
                float2 cached_pair_692 = make_float2(_tmem_load_20[6], _tmem_load_20[7]);
                float2 acc_pair_693 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_694 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_692, acc_pair_693);
                acc32[22] = updated_pair_694.x;
                acc32[23] = updated_pair_694.y;
                float _tmem_load_21[8];
                tmem_ld_x8(&_tmem_load_21[0], tmem_base + 64 + 16);
                float2 weight_pair_695 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_696 = make_float2(_tmem_load_21[0], _tmem_load_21[1]);
                float2 acc_pair_697 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_698 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_696, acc_pair_697);
                acc32[16] = updated_pair_698.x;
                acc32[17] = updated_pair_698.y;
                float2 cached_pair_699 = make_float2(_tmem_load_21[2], _tmem_load_21[3]);
                float2 acc_pair_700 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_701 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_699, acc_pair_700);
                acc32[18] = updated_pair_701.x;
                acc32[19] = updated_pair_701.y;
                float2 cached_pair_702 = make_float2(_tmem_load_21[4], _tmem_load_21[5]);
                float2 acc_pair_703 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_704 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_702, acc_pair_703);
                acc32[20] = updated_pair_704.x;
                acc32[21] = updated_pair_704.y;
                float2 cached_pair_705 = make_float2(_tmem_load_21[6], _tmem_load_21[7]);
                float2 acc_pair_706 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_707 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_705, acc_pair_706);
                acc32[22] = updated_pair_707.x;
                acc32[23] = updated_pair_707.y;
                float2 correction_pair_708 = make_float2(correction_600, correction_600);
                float2 acc_pair_709 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_710;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_710) : "l"(*(const unsigned long long*)&acc_pair_709), "l"(*(const unsigned long long*)&correction_pair_708));
                acc32[24] = scaled_pair_710.x;
                acc32[25] = scaled_pair_710.y;
                float2 acc_pair_711 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_712;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_712) : "l"(*(const unsigned long long*)&acc_pair_711), "l"(*(const unsigned long long*)&correction_pair_708));
                acc32[26] = scaled_pair_712.x;
                acc32[27] = scaled_pair_712.y;
                float _tmem_load_22[4];
                tmem_ld_x4(&_tmem_load_22[0], tmem_base + 24);
                float2 weight_pair_713 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_714 = make_float2(_tmem_load_22[0], _tmem_load_22[1]);
                float2 acc_pair_715 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_716 = fma_f32x2_rn_ftz(weight_pair_713, cached_pair_714, acc_pair_715);
                acc32[24] = updated_pair_716.x;
                acc32[25] = updated_pair_716.y;
                float2 cached_pair_717 = make_float2(_tmem_load_22[2], _tmem_load_22[3]);
                float2 acc_pair_718 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_719 = fma_f32x2_rn_ftz(weight_pair_713, cached_pair_717, acc_pair_718);
                acc32[26] = updated_pair_719.x;
                acc32[27] = updated_pair_719.y;
                float _tmem_load_23[4];
                tmem_ld_x4(&_tmem_load_23[0], tmem_base + 64 + 24);
                float2 weight_pair_720 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_721 = make_float2(_tmem_load_23[0], _tmem_load_23[1]);
                float2 acc_pair_722 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_723 = fma_f32x2_rn_ftz(weight_pair_720, cached_pair_721, acc_pair_722);
                acc32[24] = updated_pair_723.x;
                acc32[25] = updated_pair_723.y;
                float2 cached_pair_724 = make_float2(_tmem_load_23[2], _tmem_load_23[3]);
                float2 acc_pair_725 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_726 = fma_f32x2_rn_ftz(weight_pair_720, cached_pair_724, acc_pair_725);
                acc32[26] = updated_pair_726.x;
                acc32[27] = updated_pair_726.y;
                float2 correction_pair_727 = make_float2(correction_600, correction_600);
                float2 acc_pair_728 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_729;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_729) : "l"(*(const unsigned long long*)&acc_pair_728), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[28] = scaled_pair_729.x;
                acc32[29] = scaled_pair_729.y;
                float2 acc_pair_730 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_731;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_731) : "l"(*(const unsigned long long*)&acc_pair_730), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[30] = scaled_pair_731.x;
                acc32[31] = scaled_pair_731.y;
                float2 acc_pair_732 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_733;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_733) : "l"(*(const unsigned long long*)&acc_pair_732), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[32] = scaled_pair_733.x;
                acc32[33] = scaled_pair_733.y;
                float2 acc_pair_734 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_735;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_735) : "l"(*(const unsigned long long*)&acc_pair_734), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[34] = scaled_pair_735.x;
                acc32[35] = scaled_pair_735.y;
                float _tmem_load_24[8];
                tmem_ld_x8(&_tmem_load_24[0], tmem_base + 32);
                float2 weight_pair_736 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_737 = make_float2(_tmem_load_24[0], _tmem_load_24[1]);
                float2 acc_pair_738 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_739 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_737, acc_pair_738);
                acc32[28] = updated_pair_739.x;
                acc32[29] = updated_pair_739.y;
                float2 cached_pair_740 = make_float2(_tmem_load_24[2], _tmem_load_24[3]);
                float2 acc_pair_741 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_742 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_740, acc_pair_741);
                acc32[30] = updated_pair_742.x;
                acc32[31] = updated_pair_742.y;
                float2 cached_pair_743 = make_float2(_tmem_load_24[4], _tmem_load_24[5]);
                float2 acc_pair_744 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_745 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_743, acc_pair_744);
                acc32[32] = updated_pair_745.x;
                acc32[33] = updated_pair_745.y;
                float2 cached_pair_746 = make_float2(_tmem_load_24[6], _tmem_load_24[7]);
                float2 acc_pair_747 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_748 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_746, acc_pair_747);
                acc32[34] = updated_pair_748.x;
                acc32[35] = updated_pair_748.y;
                float _tmem_load_25[8];
                tmem_ld_x8(&_tmem_load_25[0], tmem_base + 64 + 32);
                float2 weight_pair_749 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_750 = make_float2(_tmem_load_25[0], _tmem_load_25[1]);
                float2 acc_pair_751 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_752 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_750, acc_pair_751);
                acc32[28] = updated_pair_752.x;
                acc32[29] = updated_pair_752.y;
                float2 cached_pair_753 = make_float2(_tmem_load_25[2], _tmem_load_25[3]);
                float2 acc_pair_754 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_755 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_753, acc_pair_754);
                acc32[30] = updated_pair_755.x;
                acc32[31] = updated_pair_755.y;
                float2 cached_pair_756 = make_float2(_tmem_load_25[4], _tmem_load_25[5]);
                float2 acc_pair_757 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_758 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_756, acc_pair_757);
                acc32[32] = updated_pair_758.x;
                acc32[33] = updated_pair_758.y;
                float2 cached_pair_759 = make_float2(_tmem_load_25[6], _tmem_load_25[7]);
                float2 acc_pair_760 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_761 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_759, acc_pair_760);
                acc32[34] = updated_pair_761.x;
                acc32[35] = updated_pair_761.y;
                float2 correction_pair_762 = make_float2(correction_600, correction_600);
                float2 acc_pair_763 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_764;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_764) : "l"(*(const unsigned long long*)&acc_pair_763), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[36] = scaled_pair_764.x;
                acc32[37] = scaled_pair_764.y;
                float2 acc_pair_765 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_766;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_766) : "l"(*(const unsigned long long*)&acc_pair_765), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[38] = scaled_pair_766.x;
                acc32[39] = scaled_pair_766.y;
                float2 acc_pair_767 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_768;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_768) : "l"(*(const unsigned long long*)&acc_pair_767), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[40] = scaled_pair_768.x;
                acc32[41] = scaled_pair_768.y;
                float2 acc_pair_769 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_770;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_770) : "l"(*(const unsigned long long*)&acc_pair_769), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[42] = scaled_pair_770.x;
                acc32[43] = scaled_pair_770.y;
                float _tmem_load_26[8];
                tmem_ld_x8(&_tmem_load_26[0], tmem_base + 40);
                float2 weight_pair_771 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_772 = make_float2(_tmem_load_26[0], _tmem_load_26[1]);
                float2 acc_pair_773 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_774 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_772, acc_pair_773);
                acc32[36] = updated_pair_774.x;
                acc32[37] = updated_pair_774.y;
                float2 cached_pair_775 = make_float2(_tmem_load_26[2], _tmem_load_26[3]);
                float2 acc_pair_776 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_777 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_775, acc_pair_776);
                acc32[38] = updated_pair_777.x;
                acc32[39] = updated_pair_777.y;
                float2 cached_pair_778 = make_float2(_tmem_load_26[4], _tmem_load_26[5]);
                float2 acc_pair_779 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_780 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_778, acc_pair_779);
                acc32[40] = updated_pair_780.x;
                acc32[41] = updated_pair_780.y;
                float2 cached_pair_781 = make_float2(_tmem_load_26[6], _tmem_load_26[7]);
                float2 acc_pair_782 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_783 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_781, acc_pair_782);
                acc32[42] = updated_pair_783.x;
                acc32[43] = updated_pair_783.y;
                float _tmem_load_27[8];
                tmem_ld_x8(&_tmem_load_27[0], tmem_base + 64 + 40);
                float2 weight_pair_784 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_785 = make_float2(_tmem_load_27[0], _tmem_load_27[1]);
                float2 acc_pair_786 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_787 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_785, acc_pair_786);
                acc32[36] = updated_pair_787.x;
                acc32[37] = updated_pair_787.y;
                float2 cached_pair_788 = make_float2(_tmem_load_27[2], _tmem_load_27[3]);
                float2 acc_pair_789 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_790 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_788, acc_pair_789);
                acc32[38] = updated_pair_790.x;
                acc32[39] = updated_pair_790.y;
                float2 cached_pair_791 = make_float2(_tmem_load_27[4], _tmem_load_27[5]);
                float2 acc_pair_792 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_793 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_791, acc_pair_792);
                acc32[40] = updated_pair_793.x;
                acc32[41] = updated_pair_793.y;
                float2 cached_pair_794 = make_float2(_tmem_load_27[6], _tmem_load_27[7]);
                float2 acc_pair_795 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_796 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_794, acc_pair_795);
                acc32[42] = updated_pair_796.x;
                acc32[43] = updated_pair_796.y;
                float2 correction_pair_797 = make_float2(correction_600, correction_600);
                float2 acc_pair_798 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_799;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_799) : "l"(*(const unsigned long long*)&acc_pair_798), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[44] = scaled_pair_799.x;
                acc32[45] = scaled_pair_799.y;
                float2 acc_pair_800 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_801;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_801) : "l"(*(const unsigned long long*)&acc_pair_800), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[46] = scaled_pair_801.x;
                acc32[47] = scaled_pair_801.y;
                float2 acc_pair_802 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_803;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_803) : "l"(*(const unsigned long long*)&acc_pair_802), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[48] = scaled_pair_803.x;
                acc32[49] = scaled_pair_803.y;
                float2 acc_pair_804 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_805;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_805) : "l"(*(const unsigned long long*)&acc_pair_804), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[50] = scaled_pair_805.x;
                acc32[51] = scaled_pair_805.y;
                float _tmem_load_28[8];
                tmem_ld_x8(&_tmem_load_28[0], tmem_base + 48);
                float2 weight_pair_806 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_807 = make_float2(_tmem_load_28[0], _tmem_load_28[1]);
                float2 acc_pair_808 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_809 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_807, acc_pair_808);
                acc32[44] = updated_pair_809.x;
                acc32[45] = updated_pair_809.y;
                float2 cached_pair_810 = make_float2(_tmem_load_28[2], _tmem_load_28[3]);
                float2 acc_pair_811 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_812 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_810, acc_pair_811);
                acc32[46] = updated_pair_812.x;
                acc32[47] = updated_pair_812.y;
                float2 cached_pair_813 = make_float2(_tmem_load_28[4], _tmem_load_28[5]);
                float2 acc_pair_814 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_815 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_813, acc_pair_814);
                acc32[48] = updated_pair_815.x;
                acc32[49] = updated_pair_815.y;
                float2 cached_pair_816 = make_float2(_tmem_load_28[6], _tmem_load_28[7]);
                float2 acc_pair_817 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_818 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_816, acc_pair_817);
                acc32[50] = updated_pair_818.x;
                acc32[51] = updated_pair_818.y;
                float _tmem_load_29[8];
                tmem_ld_x8(&_tmem_load_29[0], tmem_base + 64 + 48);
                float2 weight_pair_819 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_820 = make_float2(_tmem_load_29[0], _tmem_load_29[1]);
                float2 acc_pair_821 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_822 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_820, acc_pair_821);
                acc32[44] = updated_pair_822.x;
                acc32[45] = updated_pair_822.y;
                float2 cached_pair_823 = make_float2(_tmem_load_29[2], _tmem_load_29[3]);
                float2 acc_pair_824 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_825 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_823, acc_pair_824);
                acc32[46] = updated_pair_825.x;
                acc32[47] = updated_pair_825.y;
                float2 cached_pair_826 = make_float2(_tmem_load_29[4], _tmem_load_29[5]);
                float2 acc_pair_827 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_828 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_826, acc_pair_827);
                acc32[48] = updated_pair_828.x;
                acc32[49] = updated_pair_828.y;
                float2 cached_pair_829 = make_float2(_tmem_load_29[6], _tmem_load_29[7]);
                float2 acc_pair_830 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_831 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_829, acc_pair_830);
                acc32[50] = updated_pair_831.x;
                acc32[51] = updated_pair_831.y;
                float2 correction_pair_832 = make_float2(correction_600, correction_600);
                float2 acc_pair_833 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_834;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_834) : "l"(*(const unsigned long long*)&acc_pair_833), "l"(*(const unsigned long long*)&correction_pair_832));
                acc32[52] = scaled_pair_834.x;
                acc32[53] = scaled_pair_834.y;
                float2 acc_pair_835 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_836;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_836) : "l"(*(const unsigned long long*)&acc_pair_835), "l"(*(const unsigned long long*)&correction_pair_832));
                acc32[54] = scaled_pair_836.x;
                acc32[55] = scaled_pair_836.y;
                float _tmem_load_30[4];
                tmem_ld_x4(&_tmem_load_30[0], tmem_base + 56);
                float2 weight_pair_837 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_838 = make_float2(_tmem_load_30[0], _tmem_load_30[1]);
                float2 acc_pair_839 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_840 = fma_f32x2_rn_ftz(weight_pair_837, cached_pair_838, acc_pair_839);
                acc32[52] = updated_pair_840.x;
                acc32[53] = updated_pair_840.y;
                float2 cached_pair_841 = make_float2(_tmem_load_30[2], _tmem_load_30[3]);
                float2 acc_pair_842 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_843 = fma_f32x2_rn_ftz(weight_pair_837, cached_pair_841, acc_pair_842);
                acc32[54] = updated_pair_843.x;
                acc32[55] = updated_pair_843.y;
                float _tmem_load_31[4];
                tmem_ld_x4(&_tmem_load_31[0], tmem_base + 64 + 56);
                float2 weight_pair_844 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_845 = make_float2(_tmem_load_31[0], _tmem_load_31[1]);
                float2 acc_pair_846 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_847 = fma_f32x2_rn_ftz(weight_pair_844, cached_pair_845, acc_pair_846);
                acc32[52] = updated_pair_847.x;
                acc32[53] = updated_pair_847.y;
                float2 cached_pair_848 = make_float2(_tmem_load_31[2], _tmem_load_31[3]);
                float2 acc_pair_849 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_850 = fma_f32x2_rn_ftz(weight_pair_844, cached_pair_848, acc_pair_849);
                acc32[54] = updated_pair_850.x;
                acc32[55] = updated_pair_850.y;
                s_running = __fmaf_rn(s_running, correction_600, weight_sum_601);
                m_running = new_max_599;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; _phase_ready ^= 1; }
                unsigned int ring_stage_851 = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage_851) * 8, _phase_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq_852[4];
                float2 local_dot_853[4];
                local_sum_sq_852[0] = make_float2(0.0f, 0.0f);
                local_dot_853[0] = make_float2(0.0f, 0.0f);
                local_sum_sq_852[1] = make_float2(0.0f, 0.0f);
                local_dot_853[1] = make_float2(0.0f, 0.0f);
                local_sum_sq_852[2] = make_float2(0.0f, 0.0f);
                local_dot_853[2] = make_float2(0.0f, 0.0f);
                local_sum_sq_852[3] = make_float2(0.0f, 0.0f);
                local_dot_853[3] = make_float2(0.0f, 0.0f);
                int hidden_base_854 = thread_in_group * 8;
                unsigned int smem_regs_855[4];
                #pragma unroll
                for (int _lr_32 = 0; _lr_32 < 4; _lr_32++) {
                    smem_regs_855[_lr_32] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_854 / 2) + (unsigned int)_lr_32];
                }
                float values_856[8];
                float _unpack_bf16_32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                        : "r"(smem_regs_855[_pair]));
                }
                values_856[0] = _unpack_bf16_32[0];
                values_856[1] = _unpack_bf16_32[1];
                values_856[2] = _unpack_bf16_32[2];
                values_856[3] = _unpack_bf16_32[3];
                values_856[4] = _unpack_bf16_32[4];
                values_856[5] = _unpack_bf16_32[5];
                values_856[6] = _unpack_bf16_32[6];
                values_856[7] = _unpack_bf16_32[7];
                tmem_st_x8_f32(tmem_base, values_856);
                float2 value_pair_857 = make_float2(values_856[0], values_856[1]);
                float2 q_pair_858 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_857, value_pair_857, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_857, q_pair_858, local_dot_853[0]);
                float2 value_pair_859 = make_float2(values_856[2], values_856[3]);
                float2 q_pair_860 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_859, value_pair_859, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_859, q_pair_860, local_dot_853[0]);
                float2 value_pair_861 = make_float2(values_856[4], values_856[5]);
                float2 q_pair_862 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_861, value_pair_861, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_861, q_pair_862, local_dot_853[0]);
                float2 value_pair_863 = make_float2(values_856[6], values_856[7]);
                float2 q_pair_864 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_863, value_pair_863, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_863, q_pair_864, local_dot_853[0]);
                unsigned int smem_regs_865[4];
                #pragma unroll
                for (int _lr_33 = 0; _lr_33 < 4; _lr_33++) {
                    smem_regs_865[_lr_33] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_854 / 2) + (unsigned int)_lr_33];
                }
                float values_866[8];
                float _unpack_bf16_33[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                        : "r"(smem_regs_865[_pair]));
                }
                values_866[0] = _unpack_bf16_33[0];
                values_866[1] = _unpack_bf16_33[1];
                values_866[2] = _unpack_bf16_33[2];
                values_866[3] = _unpack_bf16_33[3];
                values_866[4] = _unpack_bf16_33[4];
                values_866[5] = _unpack_bf16_33[5];
                values_866[6] = _unpack_bf16_33[6];
                values_866[7] = _unpack_bf16_33[7];
                tmem_st_x8_f32(tmem_base + 64, values_866);
                float2 value_pair_867 = make_float2(values_866[0], values_866[1]);
                float2 q_pair_868 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_867, value_pair_867, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_867, q_pair_868, local_dot_853[1]);
                float2 value_pair_869 = make_float2(values_866[2], values_866[3]);
                float2 q_pair_870 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_869, value_pair_869, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_869, q_pair_870, local_dot_853[1]);
                float2 value_pair_871 = make_float2(values_866[4], values_866[5]);
                float2 q_pair_872 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_871, value_pair_871, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_871, q_pair_872, local_dot_853[1]);
                float2 value_pair_873 = make_float2(values_866[6], values_866[7]);
                float2 q_pair_874 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_873, value_pair_873, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_873, q_pair_874, local_dot_853[1]);
                int hidden_base_875 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_876[4];
                #pragma unroll
                for (int _lr_34 = 0; _lr_34 < 4; _lr_34++) {
                    smem_regs_876[_lr_34] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_875 / 2) + (unsigned int)_lr_34];
                }
                float values_877[8];
                float _unpack_bf16_34[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                        : "r"(smem_regs_876[_pair]));
                }
                values_877[0] = _unpack_bf16_34[0];
                values_877[1] = _unpack_bf16_34[1];
                values_877[2] = _unpack_bf16_34[2];
                values_877[3] = _unpack_bf16_34[3];
                values_877[4] = _unpack_bf16_34[4];
                values_877[5] = _unpack_bf16_34[5];
                values_877[6] = _unpack_bf16_34[6];
                values_877[7] = _unpack_bf16_34[7];
                tmem_st_x8_f32(tmem_base + 8, values_877);
                float2 value_pair_878 = make_float2(values_877[0], values_877[1]);
                float2 q_pair_879 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_878, value_pair_878, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_878, q_pair_879, local_dot_853[0]);
                float2 value_pair_880 = make_float2(values_877[2], values_877[3]);
                float2 q_pair_881 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_880, value_pair_880, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_880, q_pair_881, local_dot_853[0]);
                float2 value_pair_882 = make_float2(values_877[4], values_877[5]);
                float2 q_pair_883 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_882, value_pair_882, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_882, q_pair_883, local_dot_853[0]);
                float2 value_pair_884 = make_float2(values_877[6], values_877[7]);
                float2 q_pair_885 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_884, value_pair_884, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_884, q_pair_885, local_dot_853[0]);
                unsigned int smem_regs_886[4];
                #pragma unroll
                for (int _lr_35 = 0; _lr_35 < 4; _lr_35++) {
                    smem_regs_886[_lr_35] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_875 / 2) + (unsigned int)_lr_35];
                }
                float values_887[8];
                float _unpack_bf16_35[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                        : "r"(smem_regs_886[_pair]));
                }
                values_887[0] = _unpack_bf16_35[0];
                values_887[1] = _unpack_bf16_35[1];
                values_887[2] = _unpack_bf16_35[2];
                values_887[3] = _unpack_bf16_35[3];
                values_887[4] = _unpack_bf16_35[4];
                values_887[5] = _unpack_bf16_35[5];
                values_887[6] = _unpack_bf16_35[6];
                values_887[7] = _unpack_bf16_35[7];
                tmem_st_x8_f32(tmem_base + 64 + 8, values_887);
                float2 value_pair_888 = make_float2(values_887[0], values_887[1]);
                float2 q_pair_889 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_888, value_pair_888, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_888, q_pair_889, local_dot_853[1]);
                float2 value_pair_890 = make_float2(values_887[2], values_887[3]);
                float2 q_pair_891 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_890, value_pair_890, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_890, q_pair_891, local_dot_853[1]);
                float2 value_pair_892 = make_float2(values_887[4], values_887[5]);
                float2 q_pair_893 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_892, value_pair_892, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_892, q_pair_893, local_dot_853[1]);
                float2 value_pair_894 = make_float2(values_887[6], values_887[7]);
                float2 q_pair_895 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_894, value_pair_894, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_894, q_pair_895, local_dot_853[1]);
                int hidden_base_896 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_897[4];
                #pragma unroll
                for (int _lr_36 = 0; _lr_36 < 4; _lr_36++) {
                    smem_regs_897[_lr_36] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_896 / 2) + (unsigned int)_lr_36];
                }
                float values_898[8];
                float _unpack_bf16_36[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                        : "r"(smem_regs_897[_pair]));
                }
                values_898[0] = _unpack_bf16_36[0];
                values_898[1] = _unpack_bf16_36[1];
                values_898[2] = _unpack_bf16_36[2];
                values_898[3] = _unpack_bf16_36[3];
                values_898[4] = _unpack_bf16_36[4];
                values_898[5] = _unpack_bf16_36[5];
                values_898[6] = _unpack_bf16_36[6];
                values_898[7] = _unpack_bf16_36[7];
                tmem_st_x8_f32(tmem_base + 16, values_898);
                float2 value_pair_899 = make_float2(values_898[0], values_898[1]);
                float2 q_pair_900 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_899, value_pair_899, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_899, q_pair_900, local_dot_853[0]);
                float2 value_pair_901 = make_float2(values_898[2], values_898[3]);
                float2 q_pair_902 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_901, value_pair_901, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_901, q_pair_902, local_dot_853[0]);
                float2 value_pair_903 = make_float2(values_898[4], values_898[5]);
                float2 q_pair_904 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_903, value_pair_903, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_903, q_pair_904, local_dot_853[0]);
                float2 value_pair_905 = make_float2(values_898[6], values_898[7]);
                float2 q_pair_906 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_905, value_pair_905, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_905, q_pair_906, local_dot_853[0]);
                unsigned int smem_regs_907[4];
                #pragma unroll
                for (int _lr_37 = 0; _lr_37 < 4; _lr_37++) {
                    smem_regs_907[_lr_37] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_896 / 2) + (unsigned int)_lr_37];
                }
                float values_908[8];
                float _unpack_bf16_37[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                        : "r"(smem_regs_907[_pair]));
                }
                values_908[0] = _unpack_bf16_37[0];
                values_908[1] = _unpack_bf16_37[1];
                values_908[2] = _unpack_bf16_37[2];
                values_908[3] = _unpack_bf16_37[3];
                values_908[4] = _unpack_bf16_37[4];
                values_908[5] = _unpack_bf16_37[5];
                values_908[6] = _unpack_bf16_37[6];
                values_908[7] = _unpack_bf16_37[7];
                tmem_st_x8_f32(tmem_base + 64 + 16, values_908);
                float2 value_pair_909 = make_float2(values_908[0], values_908[1]);
                float2 q_pair_910 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_909, value_pair_909, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_909, q_pair_910, local_dot_853[1]);
                float2 value_pair_911 = make_float2(values_908[2], values_908[3]);
                float2 q_pair_912 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_911, value_pair_911, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_911, q_pair_912, local_dot_853[1]);
                float2 value_pair_913 = make_float2(values_908[4], values_908[5]);
                float2 q_pair_914 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_913, value_pair_913, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_913, q_pair_914, local_dot_853[1]);
                float2 value_pair_915 = make_float2(values_908[6], values_908[7]);
                float2 q_pair_916 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_915, value_pair_915, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_915, q_pair_916, local_dot_853[1]);
                int hidden_base_917 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_918[2];
                #pragma unroll
                for (int _lr_38 = 0; _lr_38 < 2; _lr_38++) {
                    smem_regs_918[_lr_38] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_917 / 2) + (unsigned int)_lr_38];
                }
                float values_919[4];
                float _unpack_bf16_38[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                        : "r"(smem_regs_918[_pair]));
                }
                values_919[0] = _unpack_bf16_38[0];
                values_919[1] = _unpack_bf16_38[1];
                values_919[2] = _unpack_bf16_38[2];
                values_919[3] = _unpack_bf16_38[3];
                tmem_st_x4_f32(tmem_base + 24, values_919);
                float2 value_pair_920 = make_float2(values_919[0], values_919[1]);
                float2 q_pair_921 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_920, value_pair_920, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_920, q_pair_921, local_dot_853[0]);
                float2 value_pair_922 = make_float2(values_919[2], values_919[3]);
                float2 q_pair_923 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_922, value_pair_922, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_922, q_pair_923, local_dot_853[0]);
                unsigned int smem_regs_924[2];
                #pragma unroll
                for (int _lr_39 = 0; _lr_39 < 2; _lr_39++) {
                    smem_regs_924[_lr_39] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_917 / 2) + (unsigned int)_lr_39];
                }
                float values_925[4];
                float _unpack_bf16_39[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                        : "r"(smem_regs_924[_pair]));
                }
                values_925[0] = _unpack_bf16_39[0];
                values_925[1] = _unpack_bf16_39[1];
                values_925[2] = _unpack_bf16_39[2];
                values_925[3] = _unpack_bf16_39[3];
                tmem_st_x4_f32(tmem_base + 64 + 24, values_925);
                float2 value_pair_926 = make_float2(values_925[0], values_925[1]);
                float2 q_pair_927 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_926, value_pair_926, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_926, q_pair_927, local_dot_853[1]);
                float2 value_pair_928 = make_float2(values_925[2], values_925[3]);
                float2 q_pair_929 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_928, value_pair_928, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_928, q_pair_929, local_dot_853[1]);
                int hidden_base_930 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_931[4];
                #pragma unroll
                for (int _lr_40 = 0; _lr_40 < 4; _lr_40++) {
                    smem_regs_931[_lr_40] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_930 / 2) + (unsigned int)_lr_40];
                }
                float values_932[8];
                float _unpack_bf16_40[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                        : "r"(smem_regs_931[_pair]));
                }
                values_932[0] = _unpack_bf16_40[0];
                values_932[1] = _unpack_bf16_40[1];
                values_932[2] = _unpack_bf16_40[2];
                values_932[3] = _unpack_bf16_40[3];
                values_932[4] = _unpack_bf16_40[4];
                values_932[5] = _unpack_bf16_40[5];
                values_932[6] = _unpack_bf16_40[6];
                values_932[7] = _unpack_bf16_40[7];
                tmem_st_x8_f32(tmem_base + 32, values_932);
                float2 value_pair_933 = make_float2(values_932[0], values_932[1]);
                float2 q_pair_934 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_933, value_pair_933, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_933, q_pair_934, local_dot_853[2]);
                float2 value_pair_935 = make_float2(values_932[2], values_932[3]);
                float2 q_pair_936 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_935, value_pair_935, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_935, q_pair_936, local_dot_853[2]);
                float2 value_pair_937 = make_float2(values_932[4], values_932[5]);
                float2 q_pair_938 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_937, value_pair_937, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_937, q_pair_938, local_dot_853[2]);
                float2 value_pair_939 = make_float2(values_932[6], values_932[7]);
                float2 q_pair_940 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_939, value_pair_939, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_939, q_pair_940, local_dot_853[2]);
                unsigned int smem_regs_941[4];
                #pragma unroll
                for (int _lr_41 = 0; _lr_41 < 4; _lr_41++) {
                    smem_regs_941[_lr_41] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_930 / 2) + (unsigned int)_lr_41];
                }
                float values_942[8];
                float _unpack_bf16_41[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                        : "r"(smem_regs_941[_pair]));
                }
                values_942[0] = _unpack_bf16_41[0];
                values_942[1] = _unpack_bf16_41[1];
                values_942[2] = _unpack_bf16_41[2];
                values_942[3] = _unpack_bf16_41[3];
                values_942[4] = _unpack_bf16_41[4];
                values_942[5] = _unpack_bf16_41[5];
                values_942[6] = _unpack_bf16_41[6];
                values_942[7] = _unpack_bf16_41[7];
                tmem_st_x8_f32(tmem_base + 64 + 32, values_942);
                float2 value_pair_943 = make_float2(values_942[0], values_942[1]);
                float2 q_pair_944 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_943, value_pair_943, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_943, q_pair_944, local_dot_853[3]);
                float2 value_pair_945 = make_float2(values_942[2], values_942[3]);
                float2 q_pair_946 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_945, value_pair_945, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_945, q_pair_946, local_dot_853[3]);
                float2 value_pair_947 = make_float2(values_942[4], values_942[5]);
                float2 q_pair_948 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_947, value_pair_947, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_947, q_pair_948, local_dot_853[3]);
                float2 value_pair_949 = make_float2(values_942[6], values_942[7]);
                float2 q_pair_950 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_949, value_pair_949, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_949, q_pair_950, local_dot_853[3]);
                int hidden_base_951 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_952[4];
                #pragma unroll
                for (int _lr_42 = 0; _lr_42 < 4; _lr_42++) {
                    smem_regs_952[_lr_42] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_951 / 2) + (unsigned int)_lr_42];
                }
                float values_953[8];
                float _unpack_bf16_42[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                        : "r"(smem_regs_952[_pair]));
                }
                values_953[0] = _unpack_bf16_42[0];
                values_953[1] = _unpack_bf16_42[1];
                values_953[2] = _unpack_bf16_42[2];
                values_953[3] = _unpack_bf16_42[3];
                values_953[4] = _unpack_bf16_42[4];
                values_953[5] = _unpack_bf16_42[5];
                values_953[6] = _unpack_bf16_42[6];
                values_953[7] = _unpack_bf16_42[7];
                tmem_st_x8_f32(tmem_base + 40, values_953);
                float2 value_pair_954 = make_float2(values_953[0], values_953[1]);
                float2 q_pair_955 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_954, value_pair_954, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_954, q_pair_955, local_dot_853[2]);
                float2 value_pair_956 = make_float2(values_953[2], values_953[3]);
                float2 q_pair_957 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_956, value_pair_956, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_956, q_pair_957, local_dot_853[2]);
                float2 value_pair_958 = make_float2(values_953[4], values_953[5]);
                float2 q_pair_959 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_958, value_pair_958, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_958, q_pair_959, local_dot_853[2]);
                float2 value_pair_960 = make_float2(values_953[6], values_953[7]);
                float2 q_pair_961 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_960, value_pair_960, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_960, q_pair_961, local_dot_853[2]);
                unsigned int smem_regs_962[4];
                #pragma unroll
                for (int _lr_43 = 0; _lr_43 < 4; _lr_43++) {
                    smem_regs_962[_lr_43] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_951 / 2) + (unsigned int)_lr_43];
                }
                float values_963[8];
                float _unpack_bf16_43[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                        : "r"(smem_regs_962[_pair]));
                }
                values_963[0] = _unpack_bf16_43[0];
                values_963[1] = _unpack_bf16_43[1];
                values_963[2] = _unpack_bf16_43[2];
                values_963[3] = _unpack_bf16_43[3];
                values_963[4] = _unpack_bf16_43[4];
                values_963[5] = _unpack_bf16_43[5];
                values_963[6] = _unpack_bf16_43[6];
                values_963[7] = _unpack_bf16_43[7];
                tmem_st_x8_f32(tmem_base + 64 + 40, values_963);
                float2 value_pair_964 = make_float2(values_963[0], values_963[1]);
                float2 q_pair_965 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_964, value_pair_964, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_964, q_pair_965, local_dot_853[3]);
                float2 value_pair_966 = make_float2(values_963[2], values_963[3]);
                float2 q_pair_967 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_966, value_pair_966, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_966, q_pair_967, local_dot_853[3]);
                float2 value_pair_968 = make_float2(values_963[4], values_963[5]);
                float2 q_pair_969 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_968, value_pair_968, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_968, q_pair_969, local_dot_853[3]);
                float2 value_pair_970 = make_float2(values_963[6], values_963[7]);
                float2 q_pair_971 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_970, value_pair_970, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_970, q_pair_971, local_dot_853[3]);
                int hidden_base_972 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_973[4];
                #pragma unroll
                for (int _lr_44 = 0; _lr_44 < 4; _lr_44++) {
                    smem_regs_973[_lr_44] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_972 / 2) + (unsigned int)_lr_44];
                }
                float values_974[8];
                float _unpack_bf16_44[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_44[_pair * 2])[0]), "=f"((&_unpack_bf16_44[_pair * 2])[1])
                        : "r"(smem_regs_973[_pair]));
                }
                values_974[0] = _unpack_bf16_44[0];
                values_974[1] = _unpack_bf16_44[1];
                values_974[2] = _unpack_bf16_44[2];
                values_974[3] = _unpack_bf16_44[3];
                values_974[4] = _unpack_bf16_44[4];
                values_974[5] = _unpack_bf16_44[5];
                values_974[6] = _unpack_bf16_44[6];
                values_974[7] = _unpack_bf16_44[7];
                tmem_st_x8_f32(tmem_base + 48, values_974);
                float2 value_pair_975 = make_float2(values_974[0], values_974[1]);
                float2 q_pair_976 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_975, value_pair_975, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_975, q_pair_976, local_dot_853[2]);
                float2 value_pair_977 = make_float2(values_974[2], values_974[3]);
                float2 q_pair_978 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_977, value_pair_977, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_977, q_pair_978, local_dot_853[2]);
                float2 value_pair_979 = make_float2(values_974[4], values_974[5]);
                float2 q_pair_980 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_979, value_pair_979, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_979, q_pair_980, local_dot_853[2]);
                float2 value_pair_981 = make_float2(values_974[6], values_974[7]);
                float2 q_pair_982 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_981, value_pair_981, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_981, q_pair_982, local_dot_853[2]);
                unsigned int smem_regs_983[4];
                #pragma unroll
                for (int _lr_45 = 0; _lr_45 < 4; _lr_45++) {
                    smem_regs_983[_lr_45] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_972 / 2) + (unsigned int)_lr_45];
                }
                float values_984[8];
                float _unpack_bf16_45[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_45[_pair * 2])[0]), "=f"((&_unpack_bf16_45[_pair * 2])[1])
                        : "r"(smem_regs_983[_pair]));
                }
                values_984[0] = _unpack_bf16_45[0];
                values_984[1] = _unpack_bf16_45[1];
                values_984[2] = _unpack_bf16_45[2];
                values_984[3] = _unpack_bf16_45[3];
                values_984[4] = _unpack_bf16_45[4];
                values_984[5] = _unpack_bf16_45[5];
                values_984[6] = _unpack_bf16_45[6];
                values_984[7] = _unpack_bf16_45[7];
                tmem_st_x8_f32(tmem_base + 64 + 48, values_984);
                float2 value_pair_985 = make_float2(values_984[0], values_984[1]);
                float2 q_pair_986 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_985, value_pair_985, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_985, q_pair_986, local_dot_853[3]);
                float2 value_pair_987 = make_float2(values_984[2], values_984[3]);
                float2 q_pair_988 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_987, value_pair_987, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_987, q_pair_988, local_dot_853[3]);
                float2 value_pair_989 = make_float2(values_984[4], values_984[5]);
                float2 q_pair_990 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_989, value_pair_989, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_989, q_pair_990, local_dot_853[3]);
                float2 value_pair_991 = make_float2(values_984[6], values_984[7]);
                float2 q_pair_992 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_991, value_pair_991, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_991, q_pair_992, local_dot_853[3]);
                int hidden_base_993 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_994[2];
                #pragma unroll
                for (int _lr_46 = 0; _lr_46 < 2; _lr_46++) {
                    smem_regs_994[_lr_46] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_993 / 2) + (unsigned int)_lr_46];
                }
                float values_995[4];
                float _unpack_bf16_46[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_46[_pair * 2])[0]), "=f"((&_unpack_bf16_46[_pair * 2])[1])
                        : "r"(smem_regs_994[_pair]));
                }
                values_995[0] = _unpack_bf16_46[0];
                values_995[1] = _unpack_bf16_46[1];
                values_995[2] = _unpack_bf16_46[2];
                values_995[3] = _unpack_bf16_46[3];
                tmem_st_x4_f32(tmem_base + 56, values_995);
                float2 value_pair_996 = make_float2(values_995[0], values_995[1]);
                float2 q_pair_997 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_996, value_pair_996, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_996, q_pair_997, local_dot_853[2]);
                float2 value_pair_998 = make_float2(values_995[2], values_995[3]);
                float2 q_pair_999 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_998, value_pair_998, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_998, q_pair_999, local_dot_853[2]);
                unsigned int smem_regs_1000[2];
                #pragma unroll
                for (int _lr_47 = 0; _lr_47 < 2; _lr_47++) {
                    smem_regs_1000[_lr_47] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_993 / 2) + (unsigned int)_lr_47];
                }
                float values_1001[4];
                float _unpack_bf16_47[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_47[_pair * 2])[0]), "=f"((&_unpack_bf16_47[_pair * 2])[1])
                        : "r"(smem_regs_1000[_pair]));
                }
                values_1001[0] = _unpack_bf16_47[0];
                values_1001[1] = _unpack_bf16_47[1];
                values_1001[2] = _unpack_bf16_47[2];
                values_1001[3] = _unpack_bf16_47[3];
                tmem_st_x4_f32(tmem_base + 64 + 56, values_1001);
                float2 value_pair_1002 = make_float2(values_1001[0], values_1001[1]);
                float2 q_pair_1003 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1002, value_pair_1002, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1002, q_pair_1003, local_dot_853[3]);
                float2 value_pair_1004 = make_float2(values_1001[2], values_1001[3]);
                float2 q_pair_1005 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1004, value_pair_1004, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1004, q_pair_1005, local_dot_853[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_851) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1006 = stats_stream_base + 64;
                float2 reduce_pair_1007 = make_float2(local_sum_sq_852[0].x + local_sum_sq_852[0].y, local_dot_853[0].x + local_dot_853[0].y);
                unsigned long long packed_pair_1008 = 0;
                unsigned long long peer_packed_1009 = 0;
                float2 peer_pair_1010 = make_float2(0.0f, 0.0f);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 16);
                peer_packed_1009 = _shfl_xor_sync_136;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 8);
                peer_packed_1009 = _shfl_xor_sync_137;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 4);
                peer_packed_1009 = _shfl_xor_sync_138;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 2);
                peer_packed_1009 = _shfl_xor_sync_139;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 1);
                peer_packed_1009 = _shfl_xor_sync_140;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1007.x;
                    stats[slot_base_1006 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1007.y;
                }
                float2 reduce_pair_1011 = make_float2(local_sum_sq_852[1].x + local_sum_sq_852[1].y, local_dot_853[1].x + local_dot_853[1].y);
                unsigned long long packed_pair_1012 = 0;
                unsigned long long peer_packed_1013 = 0;
                float2 peer_pair_1014 = make_float2(0.0f, 0.0f);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 16);
                peer_packed_1013 = _shfl_xor_sync_141;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 8);
                peer_packed_1013 = _shfl_xor_sync_142;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 4);
                peer_packed_1013 = _shfl_xor_sync_143;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 2);
                peer_packed_1013 = _shfl_xor_sync_144;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 1);
                peer_packed_1013 = _shfl_xor_sync_145;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_1011.x;
                    stats[slot_base_1006 + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_1011.y;
                }
                float2 reduce_pair_1015 = make_float2(local_sum_sq_852[2].x + local_sum_sq_852[2].y, local_dot_853[2].x + local_dot_853[2].y);
                unsigned long long packed_pair_1016 = 0;
                unsigned long long peer_packed_1017 = 0;
                float2 peer_pair_1018 = make_float2(0.0f, 0.0f);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 16);
                peer_packed_1017 = _shfl_xor_sync_146;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 8);
                peer_packed_1017 = _shfl_xor_sync_147;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 4);
                peer_packed_1017 = _shfl_xor_sync_148;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 2);
                peer_packed_1017 = _shfl_xor_sync_149;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 1);
                peer_packed_1017 = _shfl_xor_sync_150;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1015.x;
                    stats[slot_base_1006 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1015.y;
                }
                float2 reduce_pair_1019 = make_float2(local_sum_sq_852[3].x + local_sum_sq_852[3].y, local_dot_853[3].x + local_dot_853[3].y);
                unsigned long long packed_pair_1020 = 0;
                unsigned long long peer_packed_1021 = 0;
                float2 peer_pair_1022 = make_float2(0.0f, 0.0f);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 16);
                peer_packed_1021 = _shfl_xor_sync_151;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 8);
                peer_packed_1021 = _shfl_xor_sync_152;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 4);
                peer_packed_1021 = _shfl_xor_sync_153;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 2);
                peer_packed_1021 = _shfl_xor_sync_154;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 1);
                peer_packed_1021 = _shfl_xor_sync_155;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2)] = reduce_pair_1019.x;
                    stats[slot_base_1006 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2) + 1] = reduce_pair_1019.y;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int stat_source_1023 = lane / 8;
                int stat_warp_1024 = lane % 8;
                float total_sum_sq_1025 = 0.0f;
                float total_dot_1026 = 0.0f;
                if (stat_source_1023 < 2) {
                    total_sum_sq_1025 = stats[slot_base_1006 + (unsigned int)((stat_warp_1024 * 2 + stat_source_1023) * 2)];
                    total_dot_1026 = stats[slot_base_1006 + (unsigned int)((stat_warp_1024 * 2 + stat_source_1023) * 2) + 1];
                }
                float _shfl_down_sync_156 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1025, 4, 8);
                total_sum_sq_1025 += _shfl_down_sync_156;
                float _shfl_down_sync_157 = __shfl_down_sync(0xFFFFFFFF, total_dot_1026, 4, 8);
                total_dot_1026 += _shfl_down_sync_157;
                float _shfl_down_sync_158 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1025, 2, 8);
                total_sum_sq_1025 += _shfl_down_sync_158;
                float _shfl_down_sync_159 = __shfl_down_sync(0xFFFFFFFF, total_dot_1026, 2, 8);
                total_dot_1026 += _shfl_down_sync_159;
                float _shfl_down_sync_160 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1025, 1, 8);
                total_sum_sq_1025 += _shfl_down_sync_160;
                float _shfl_down_sync_161 = __shfl_down_sync(0xFFFFFFFF, total_dot_1026, 1, 8);
                total_dot_1026 += _shfl_down_sync_161;
                float local_logit_1027 = 0.0f;
                if (stat_source_1023 < 2 && stat_warp_1024 == 0) {
                    float _rsqrt_162 = rsqrtf(total_sum_sq_1025 / 7168.0f + eps);
                    local_logit_1027 = total_dot_1026 * _rsqrt_162;
                }
                float logits_1028[2];
                logits_1028[0] = 0.0f;
                float _shfl_sync_163 = __shfl_sync(0xFFFFFFFF, local_logit_1027, 0);
                logits_1028[0] = _shfl_sync_163;
                logits_1028[1] = 0.0f;
                float _shfl_sync_164 = __shfl_sync(0xFFFFFFFF, local_logit_1027, 8);
                logits_1028[1] = _shfl_sync_164;
                float weights_1029[2];
                float new_max_1030 = 0.0f;
                float correction_1031 = 0.0f;
                float weight_sum_1032 = 0.0f;
                weights_1029[0] = 0.0f;
                weights_1029[1] = 0.0f;
                float chunk_max_1033 = -3.4028235e+38f;
                chunk_max_1033 = max_noftz(chunk_max_1033, logits_1028[0]);
                chunk_max_1033 = max_noftz(chunk_max_1033, logits_1028[1]);
                new_max_1030 = max_noftz(m_running, chunk_max_1033);
                float _exp2_165 = approx_exp2((m_running - new_max_1030) * 1.442695f);
                correction_1031 = _exp2_165;
                float _exp2_166 = approx_exp2((logits_1028[0] - new_max_1030) * 1.442695f);
                weights_1029[0] = _exp2_166;
                weight_sum_1032 += weights_1029[0];
                float _exp2_167 = approx_exp2((logits_1028[1] - new_max_1030) * 1.442695f);
                weights_1029[1] = _exp2_167;
                weight_sum_1032 += weights_1029[1];
                float2 correction_pair_1034 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1035 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1036;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1036) : "l"(*(const unsigned long long*)&acc_pair_1035), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[0] = scaled_pair_1036.x;
                acc32[1] = scaled_pair_1036.y;
                float2 acc_pair_1037 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1038;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1038) : "l"(*(const unsigned long long*)&acc_pair_1037), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[2] = scaled_pair_1038.x;
                acc32[3] = scaled_pair_1038.y;
                float2 acc_pair_1039 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1040;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1040) : "l"(*(const unsigned long long*)&acc_pair_1039), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[4] = scaled_pair_1040.x;
                acc32[5] = scaled_pair_1040.y;
                float2 acc_pair_1041 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1042;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1042) : "l"(*(const unsigned long long*)&acc_pair_1041), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[6] = scaled_pair_1042.x;
                acc32[7] = scaled_pair_1042.y;
                float _tmem_load_32[8];
                tmem_ld_x8(&_tmem_load_32[0], tmem_base);
                float2 weight_pair_1043 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1044 = make_float2(_tmem_load_32[0], _tmem_load_32[1]);
                float2 acc_pair_1045 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1046 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1044, acc_pair_1045);
                acc32[0] = updated_pair_1046.x;
                acc32[1] = updated_pair_1046.y;
                float2 cached_pair_1047 = make_float2(_tmem_load_32[2], _tmem_load_32[3]);
                float2 acc_pair_1048 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1049 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1047, acc_pair_1048);
                acc32[2] = updated_pair_1049.x;
                acc32[3] = updated_pair_1049.y;
                float2 cached_pair_1050 = make_float2(_tmem_load_32[4], _tmem_load_32[5]);
                float2 acc_pair_1051 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1052 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1050, acc_pair_1051);
                acc32[4] = updated_pair_1052.x;
                acc32[5] = updated_pair_1052.y;
                float2 cached_pair_1053 = make_float2(_tmem_load_32[6], _tmem_load_32[7]);
                float2 acc_pair_1054 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1055 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1053, acc_pair_1054);
                acc32[6] = updated_pair_1055.x;
                acc32[7] = updated_pair_1055.y;
                float _tmem_load_33[8];
                tmem_ld_x8(&_tmem_load_33[0], tmem_base + 64);
                float2 weight_pair_1056 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1057 = make_float2(_tmem_load_33[0], _tmem_load_33[1]);
                float2 acc_pair_1058 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1059 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1057, acc_pair_1058);
                acc32[0] = updated_pair_1059.x;
                acc32[1] = updated_pair_1059.y;
                float2 cached_pair_1060 = make_float2(_tmem_load_33[2], _tmem_load_33[3]);
                float2 acc_pair_1061 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1062 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1060, acc_pair_1061);
                acc32[2] = updated_pair_1062.x;
                acc32[3] = updated_pair_1062.y;
                float2 cached_pair_1063 = make_float2(_tmem_load_33[4], _tmem_load_33[5]);
                float2 acc_pair_1064 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1065 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1063, acc_pair_1064);
                acc32[4] = updated_pair_1065.x;
                acc32[5] = updated_pair_1065.y;
                float2 cached_pair_1066 = make_float2(_tmem_load_33[6], _tmem_load_33[7]);
                float2 acc_pair_1067 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1068 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1066, acc_pair_1067);
                acc32[6] = updated_pair_1068.x;
                acc32[7] = updated_pair_1068.y;
                float2 correction_pair_1069 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1070 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1071;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1071) : "l"(*(const unsigned long long*)&acc_pair_1070), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[8] = scaled_pair_1071.x;
                acc32[9] = scaled_pair_1071.y;
                float2 acc_pair_1072 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1073;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1073) : "l"(*(const unsigned long long*)&acc_pair_1072), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[10] = scaled_pair_1073.x;
                acc32[11] = scaled_pair_1073.y;
                float2 acc_pair_1074 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1075;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1075) : "l"(*(const unsigned long long*)&acc_pair_1074), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[12] = scaled_pair_1075.x;
                acc32[13] = scaled_pair_1075.y;
                float2 acc_pair_1076 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1077;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1077) : "l"(*(const unsigned long long*)&acc_pair_1076), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[14] = scaled_pair_1077.x;
                acc32[15] = scaled_pair_1077.y;
                float _tmem_load_34[8];
                tmem_ld_x8(&_tmem_load_34[0], tmem_base + 8);
                float2 weight_pair_1078 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1079 = make_float2(_tmem_load_34[0], _tmem_load_34[1]);
                float2 acc_pair_1080 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1081 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1079, acc_pair_1080);
                acc32[8] = updated_pair_1081.x;
                acc32[9] = updated_pair_1081.y;
                float2 cached_pair_1082 = make_float2(_tmem_load_34[2], _tmem_load_34[3]);
                float2 acc_pair_1083 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1084 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1082, acc_pair_1083);
                acc32[10] = updated_pair_1084.x;
                acc32[11] = updated_pair_1084.y;
                float2 cached_pair_1085 = make_float2(_tmem_load_34[4], _tmem_load_34[5]);
                float2 acc_pair_1086 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1087 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1085, acc_pair_1086);
                acc32[12] = updated_pair_1087.x;
                acc32[13] = updated_pair_1087.y;
                float2 cached_pair_1088 = make_float2(_tmem_load_34[6], _tmem_load_34[7]);
                float2 acc_pair_1089 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1090 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1088, acc_pair_1089);
                acc32[14] = updated_pair_1090.x;
                acc32[15] = updated_pair_1090.y;
                float _tmem_load_35[8];
                tmem_ld_x8(&_tmem_load_35[0], tmem_base + 64 + 8);
                float2 weight_pair_1091 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1092 = make_float2(_tmem_load_35[0], _tmem_load_35[1]);
                float2 acc_pair_1093 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1094 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1092, acc_pair_1093);
                acc32[8] = updated_pair_1094.x;
                acc32[9] = updated_pair_1094.y;
                float2 cached_pair_1095 = make_float2(_tmem_load_35[2], _tmem_load_35[3]);
                float2 acc_pair_1096 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1097 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1095, acc_pair_1096);
                acc32[10] = updated_pair_1097.x;
                acc32[11] = updated_pair_1097.y;
                float2 cached_pair_1098 = make_float2(_tmem_load_35[4], _tmem_load_35[5]);
                float2 acc_pair_1099 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1100 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1098, acc_pair_1099);
                acc32[12] = updated_pair_1100.x;
                acc32[13] = updated_pair_1100.y;
                float2 cached_pair_1101 = make_float2(_tmem_load_35[6], _tmem_load_35[7]);
                float2 acc_pair_1102 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1103 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1101, acc_pair_1102);
                acc32[14] = updated_pair_1103.x;
                acc32[15] = updated_pair_1103.y;
                float2 correction_pair_1104 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1105 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1106;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1106) : "l"(*(const unsigned long long*)&acc_pair_1105), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[16] = scaled_pair_1106.x;
                acc32[17] = scaled_pair_1106.y;
                float2 acc_pair_1107 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1108;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1108) : "l"(*(const unsigned long long*)&acc_pair_1107), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[18] = scaled_pair_1108.x;
                acc32[19] = scaled_pair_1108.y;
                float2 acc_pair_1109 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1110;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1110) : "l"(*(const unsigned long long*)&acc_pair_1109), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[20] = scaled_pair_1110.x;
                acc32[21] = scaled_pair_1110.y;
                float2 acc_pair_1111 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1112;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1112) : "l"(*(const unsigned long long*)&acc_pair_1111), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[22] = scaled_pair_1112.x;
                acc32[23] = scaled_pair_1112.y;
                float _tmem_load_36[8];
                tmem_ld_x8(&_tmem_load_36[0], tmem_base + 16);
                float2 weight_pair_1113 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1114 = make_float2(_tmem_load_36[0], _tmem_load_36[1]);
                float2 acc_pair_1115 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1116 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1114, acc_pair_1115);
                acc32[16] = updated_pair_1116.x;
                acc32[17] = updated_pair_1116.y;
                float2 cached_pair_1117 = make_float2(_tmem_load_36[2], _tmem_load_36[3]);
                float2 acc_pair_1118 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1119 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1117, acc_pair_1118);
                acc32[18] = updated_pair_1119.x;
                acc32[19] = updated_pair_1119.y;
                float2 cached_pair_1120 = make_float2(_tmem_load_36[4], _tmem_load_36[5]);
                float2 acc_pair_1121 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1122 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1120, acc_pair_1121);
                acc32[20] = updated_pair_1122.x;
                acc32[21] = updated_pair_1122.y;
                float2 cached_pair_1123 = make_float2(_tmem_load_36[6], _tmem_load_36[7]);
                float2 acc_pair_1124 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1125 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1123, acc_pair_1124);
                acc32[22] = updated_pair_1125.x;
                acc32[23] = updated_pair_1125.y;
                float _tmem_load_37[8];
                tmem_ld_x8(&_tmem_load_37[0], tmem_base + 64 + 16);
                float2 weight_pair_1126 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1127 = make_float2(_tmem_load_37[0], _tmem_load_37[1]);
                float2 acc_pair_1128 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1129 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1127, acc_pair_1128);
                acc32[16] = updated_pair_1129.x;
                acc32[17] = updated_pair_1129.y;
                float2 cached_pair_1130 = make_float2(_tmem_load_37[2], _tmem_load_37[3]);
                float2 acc_pair_1131 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1132 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1130, acc_pair_1131);
                acc32[18] = updated_pair_1132.x;
                acc32[19] = updated_pair_1132.y;
                float2 cached_pair_1133 = make_float2(_tmem_load_37[4], _tmem_load_37[5]);
                float2 acc_pair_1134 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1135 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1133, acc_pair_1134);
                acc32[20] = updated_pair_1135.x;
                acc32[21] = updated_pair_1135.y;
                float2 cached_pair_1136 = make_float2(_tmem_load_37[6], _tmem_load_37[7]);
                float2 acc_pair_1137 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1138 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1136, acc_pair_1137);
                acc32[22] = updated_pair_1138.x;
                acc32[23] = updated_pair_1138.y;
                float2 correction_pair_1139 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1140 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1141;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1141) : "l"(*(const unsigned long long*)&acc_pair_1140), "l"(*(const unsigned long long*)&correction_pair_1139));
                acc32[24] = scaled_pair_1141.x;
                acc32[25] = scaled_pair_1141.y;
                float2 acc_pair_1142 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1143;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1143) : "l"(*(const unsigned long long*)&acc_pair_1142), "l"(*(const unsigned long long*)&correction_pair_1139));
                acc32[26] = scaled_pair_1143.x;
                acc32[27] = scaled_pair_1143.y;
                float _tmem_load_38[4];
                tmem_ld_x4(&_tmem_load_38[0], tmem_base + 24);
                float2 weight_pair_1144 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1145 = make_float2(_tmem_load_38[0], _tmem_load_38[1]);
                float2 acc_pair_1146 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1147 = fma_f32x2_rn_ftz(weight_pair_1144, cached_pair_1145, acc_pair_1146);
                acc32[24] = updated_pair_1147.x;
                acc32[25] = updated_pair_1147.y;
                float2 cached_pair_1148 = make_float2(_tmem_load_38[2], _tmem_load_38[3]);
                float2 acc_pair_1149 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1150 = fma_f32x2_rn_ftz(weight_pair_1144, cached_pair_1148, acc_pair_1149);
                acc32[26] = updated_pair_1150.x;
                acc32[27] = updated_pair_1150.y;
                float _tmem_load_39[4];
                tmem_ld_x4(&_tmem_load_39[0], tmem_base + 64 + 24);
                float2 weight_pair_1151 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1152 = make_float2(_tmem_load_39[0], _tmem_load_39[1]);
                float2 acc_pair_1153 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1154 = fma_f32x2_rn_ftz(weight_pair_1151, cached_pair_1152, acc_pair_1153);
                acc32[24] = updated_pair_1154.x;
                acc32[25] = updated_pair_1154.y;
                float2 cached_pair_1155 = make_float2(_tmem_load_39[2], _tmem_load_39[3]);
                float2 acc_pair_1156 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1157 = fma_f32x2_rn_ftz(weight_pair_1151, cached_pair_1155, acc_pair_1156);
                acc32[26] = updated_pair_1157.x;
                acc32[27] = updated_pair_1157.y;
                float2 correction_pair_1158 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1159 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1160;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1160) : "l"(*(const unsigned long long*)&acc_pair_1159), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[28] = scaled_pair_1160.x;
                acc32[29] = scaled_pair_1160.y;
                float2 acc_pair_1161 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1162;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1162) : "l"(*(const unsigned long long*)&acc_pair_1161), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[30] = scaled_pair_1162.x;
                acc32[31] = scaled_pair_1162.y;
                float2 acc_pair_1163 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1164;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1164) : "l"(*(const unsigned long long*)&acc_pair_1163), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[32] = scaled_pair_1164.x;
                acc32[33] = scaled_pair_1164.y;
                float2 acc_pair_1165 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1166;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1166) : "l"(*(const unsigned long long*)&acc_pair_1165), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[34] = scaled_pair_1166.x;
                acc32[35] = scaled_pair_1166.y;
                float _tmem_load_40[8];
                tmem_ld_x8(&_tmem_load_40[0], tmem_base + 32);
                float2 weight_pair_1167 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1168 = make_float2(_tmem_load_40[0], _tmem_load_40[1]);
                float2 acc_pair_1169 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1170 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1168, acc_pair_1169);
                acc32[28] = updated_pair_1170.x;
                acc32[29] = updated_pair_1170.y;
                float2 cached_pair_1171 = make_float2(_tmem_load_40[2], _tmem_load_40[3]);
                float2 acc_pair_1172 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1173 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1171, acc_pair_1172);
                acc32[30] = updated_pair_1173.x;
                acc32[31] = updated_pair_1173.y;
                float2 cached_pair_1174 = make_float2(_tmem_load_40[4], _tmem_load_40[5]);
                float2 acc_pair_1175 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1176 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1174, acc_pair_1175);
                acc32[32] = updated_pair_1176.x;
                acc32[33] = updated_pair_1176.y;
                float2 cached_pair_1177 = make_float2(_tmem_load_40[6], _tmem_load_40[7]);
                float2 acc_pair_1178 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1179 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1177, acc_pair_1178);
                acc32[34] = updated_pair_1179.x;
                acc32[35] = updated_pair_1179.y;
                float _tmem_load_41[8];
                tmem_ld_x8(&_tmem_load_41[0], tmem_base + 64 + 32);
                float2 weight_pair_1180 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1181 = make_float2(_tmem_load_41[0], _tmem_load_41[1]);
                float2 acc_pair_1182 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1183 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1181, acc_pair_1182);
                acc32[28] = updated_pair_1183.x;
                acc32[29] = updated_pair_1183.y;
                float2 cached_pair_1184 = make_float2(_tmem_load_41[2], _tmem_load_41[3]);
                float2 acc_pair_1185 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1186 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1184, acc_pair_1185);
                acc32[30] = updated_pair_1186.x;
                acc32[31] = updated_pair_1186.y;
                float2 cached_pair_1187 = make_float2(_tmem_load_41[4], _tmem_load_41[5]);
                float2 acc_pair_1188 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1189 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1187, acc_pair_1188);
                acc32[32] = updated_pair_1189.x;
                acc32[33] = updated_pair_1189.y;
                float2 cached_pair_1190 = make_float2(_tmem_load_41[6], _tmem_load_41[7]);
                float2 acc_pair_1191 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1192 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1190, acc_pair_1191);
                acc32[34] = updated_pair_1192.x;
                acc32[35] = updated_pair_1192.y;
                float2 correction_pair_1193 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1194 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1195;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1195) : "l"(*(const unsigned long long*)&acc_pair_1194), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[36] = scaled_pair_1195.x;
                acc32[37] = scaled_pair_1195.y;
                float2 acc_pair_1196 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1197;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1197) : "l"(*(const unsigned long long*)&acc_pair_1196), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[38] = scaled_pair_1197.x;
                acc32[39] = scaled_pair_1197.y;
                float2 acc_pair_1198 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1199;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1199) : "l"(*(const unsigned long long*)&acc_pair_1198), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[40] = scaled_pair_1199.x;
                acc32[41] = scaled_pair_1199.y;
                float2 acc_pair_1200 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1201;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1201) : "l"(*(const unsigned long long*)&acc_pair_1200), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[42] = scaled_pair_1201.x;
                acc32[43] = scaled_pair_1201.y;
                float _tmem_load_42[8];
                tmem_ld_x8(&_tmem_load_42[0], tmem_base + 40);
                float2 weight_pair_1202 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1203 = make_float2(_tmem_load_42[0], _tmem_load_42[1]);
                float2 acc_pair_1204 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1205 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1203, acc_pair_1204);
                acc32[36] = updated_pair_1205.x;
                acc32[37] = updated_pair_1205.y;
                float2 cached_pair_1206 = make_float2(_tmem_load_42[2], _tmem_load_42[3]);
                float2 acc_pair_1207 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1208 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1206, acc_pair_1207);
                acc32[38] = updated_pair_1208.x;
                acc32[39] = updated_pair_1208.y;
                float2 cached_pair_1209 = make_float2(_tmem_load_42[4], _tmem_load_42[5]);
                float2 acc_pair_1210 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1211 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1209, acc_pair_1210);
                acc32[40] = updated_pair_1211.x;
                acc32[41] = updated_pair_1211.y;
                float2 cached_pair_1212 = make_float2(_tmem_load_42[6], _tmem_load_42[7]);
                float2 acc_pair_1213 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1214 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1212, acc_pair_1213);
                acc32[42] = updated_pair_1214.x;
                acc32[43] = updated_pair_1214.y;
                float _tmem_load_43[8];
                tmem_ld_x8(&_tmem_load_43[0], tmem_base + 64 + 40);
                float2 weight_pair_1215 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1216 = make_float2(_tmem_load_43[0], _tmem_load_43[1]);
                float2 acc_pair_1217 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1218 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1216, acc_pair_1217);
                acc32[36] = updated_pair_1218.x;
                acc32[37] = updated_pair_1218.y;
                float2 cached_pair_1219 = make_float2(_tmem_load_43[2], _tmem_load_43[3]);
                float2 acc_pair_1220 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1221 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1219, acc_pair_1220);
                acc32[38] = updated_pair_1221.x;
                acc32[39] = updated_pair_1221.y;
                float2 cached_pair_1222 = make_float2(_tmem_load_43[4], _tmem_load_43[5]);
                float2 acc_pair_1223 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1224 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1222, acc_pair_1223);
                acc32[40] = updated_pair_1224.x;
                acc32[41] = updated_pair_1224.y;
                float2 cached_pair_1225 = make_float2(_tmem_load_43[6], _tmem_load_43[7]);
                float2 acc_pair_1226 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1227 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1225, acc_pair_1226);
                acc32[42] = updated_pair_1227.x;
                acc32[43] = updated_pair_1227.y;
                float2 correction_pair_1228 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1229 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1230;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1230) : "l"(*(const unsigned long long*)&acc_pair_1229), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[44] = scaled_pair_1230.x;
                acc32[45] = scaled_pair_1230.y;
                float2 acc_pair_1231 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1232;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1232) : "l"(*(const unsigned long long*)&acc_pair_1231), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[46] = scaled_pair_1232.x;
                acc32[47] = scaled_pair_1232.y;
                float2 acc_pair_1233 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1234;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1234) : "l"(*(const unsigned long long*)&acc_pair_1233), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[48] = scaled_pair_1234.x;
                acc32[49] = scaled_pair_1234.y;
                float2 acc_pair_1235 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1236;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1236) : "l"(*(const unsigned long long*)&acc_pair_1235), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[50] = scaled_pair_1236.x;
                acc32[51] = scaled_pair_1236.y;
                float _tmem_load_44[8];
                tmem_ld_x8(&_tmem_load_44[0], tmem_base + 48);
                float2 weight_pair_1237 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1238 = make_float2(_tmem_load_44[0], _tmem_load_44[1]);
                float2 acc_pair_1239 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1240 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1238, acc_pair_1239);
                acc32[44] = updated_pair_1240.x;
                acc32[45] = updated_pair_1240.y;
                float2 cached_pair_1241 = make_float2(_tmem_load_44[2], _tmem_load_44[3]);
                float2 acc_pair_1242 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1243 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1241, acc_pair_1242);
                acc32[46] = updated_pair_1243.x;
                acc32[47] = updated_pair_1243.y;
                float2 cached_pair_1244 = make_float2(_tmem_load_44[4], _tmem_load_44[5]);
                float2 acc_pair_1245 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1246 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1244, acc_pair_1245);
                acc32[48] = updated_pair_1246.x;
                acc32[49] = updated_pair_1246.y;
                float2 cached_pair_1247 = make_float2(_tmem_load_44[6], _tmem_load_44[7]);
                float2 acc_pair_1248 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1249 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1247, acc_pair_1248);
                acc32[50] = updated_pair_1249.x;
                acc32[51] = updated_pair_1249.y;
                float _tmem_load_45[8];
                tmem_ld_x8(&_tmem_load_45[0], tmem_base + 64 + 48);
                float2 weight_pair_1250 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1251 = make_float2(_tmem_load_45[0], _tmem_load_45[1]);
                float2 acc_pair_1252 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1253 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1251, acc_pair_1252);
                acc32[44] = updated_pair_1253.x;
                acc32[45] = updated_pair_1253.y;
                float2 cached_pair_1254 = make_float2(_tmem_load_45[2], _tmem_load_45[3]);
                float2 acc_pair_1255 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1256 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1254, acc_pair_1255);
                acc32[46] = updated_pair_1256.x;
                acc32[47] = updated_pair_1256.y;
                float2 cached_pair_1257 = make_float2(_tmem_load_45[4], _tmem_load_45[5]);
                float2 acc_pair_1258 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1259 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1257, acc_pair_1258);
                acc32[48] = updated_pair_1259.x;
                acc32[49] = updated_pair_1259.y;
                float2 cached_pair_1260 = make_float2(_tmem_load_45[6], _tmem_load_45[7]);
                float2 acc_pair_1261 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1262 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1260, acc_pair_1261);
                acc32[50] = updated_pair_1262.x;
                acc32[51] = updated_pair_1262.y;
                float2 correction_pair_1263 = make_float2(correction_1031, correction_1031);
                float2 acc_pair_1264 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1265;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1265) : "l"(*(const unsigned long long*)&acc_pair_1264), "l"(*(const unsigned long long*)&correction_pair_1263));
                acc32[52] = scaled_pair_1265.x;
                acc32[53] = scaled_pair_1265.y;
                float2 acc_pair_1266 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1267;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1267) : "l"(*(const unsigned long long*)&acc_pair_1266), "l"(*(const unsigned long long*)&correction_pair_1263));
                acc32[54] = scaled_pair_1267.x;
                acc32[55] = scaled_pair_1267.y;
                float _tmem_load_46[4];
                tmem_ld_x4(&_tmem_load_46[0], tmem_base + 56);
                float2 weight_pair_1268 = make_float2(weights_1029[0], weights_1029[0]);
                float2 cached_pair_1269 = make_float2(_tmem_load_46[0], _tmem_load_46[1]);
                float2 acc_pair_1270 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1271 = fma_f32x2_rn_ftz(weight_pair_1268, cached_pair_1269, acc_pair_1270);
                acc32[52] = updated_pair_1271.x;
                acc32[53] = updated_pair_1271.y;
                float2 cached_pair_1272 = make_float2(_tmem_load_46[2], _tmem_load_46[3]);
                float2 acc_pair_1273 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1274 = fma_f32x2_rn_ftz(weight_pair_1268, cached_pair_1272, acc_pair_1273);
                acc32[54] = updated_pair_1274.x;
                acc32[55] = updated_pair_1274.y;
                float _tmem_load_47[4];
                tmem_ld_x4(&_tmem_load_47[0], tmem_base + 64 + 56);
                float2 weight_pair_1275 = make_float2(weights_1029[1], weights_1029[1]);
                float2 cached_pair_1276 = make_float2(_tmem_load_47[0], _tmem_load_47[1]);
                float2 acc_pair_1277 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1278 = fma_f32x2_rn_ftz(weight_pair_1275, cached_pair_1276, acc_pair_1277);
                acc32[52] = updated_pair_1278.x;
                acc32[53] = updated_pair_1278.y;
                float2 cached_pair_1279 = make_float2(_tmem_load_47[2], _tmem_load_47[3]);
                float2 acc_pair_1280 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1281 = fma_f32x2_rn_ftz(weight_pair_1275, cached_pair_1279, acc_pair_1280);
                acc32[54] = updated_pair_1281.x;
                acc32[55] = updated_pair_1281.y;
                s_running = __fmaf_rn(s_running, correction_1031, weight_sum_1032);
                m_running = new_max_1030;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; _phase_ready ^= 1; }
                unsigned int ring_stage_1282 = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage_1282) * 8, _phase_ready);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq_1283[4];
                float2 local_dot_1284[4];
                local_sum_sq_1283[0] = make_float2(0.0f, 0.0f);
                local_dot_1284[0] = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[1] = make_float2(0.0f, 0.0f);
                local_dot_1284[1] = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[2] = make_float2(0.0f, 0.0f);
                local_dot_1284[2] = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[3] = make_float2(0.0f, 0.0f);
                local_dot_1284[3] = make_float2(0.0f, 0.0f);
                int hidden_base_1285 = thread_in_group * 8;
                unsigned int smem_regs_1286[4];
                #pragma unroll
                for (int _lr_48 = 0; _lr_48 < 4; _lr_48++) {
                    smem_regs_1286[_lr_48] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1285 / 2) + (unsigned int)_lr_48];
                }
                float values_1287[8];
                unsigned int smem_regs_1288[4];
                #pragma unroll
                for (int _lr_49 = 0; _lr_49 < 4; _lr_49++) {
                    smem_regs_1288[_lr_49] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1285 / 2) + (unsigned int)_lr_49];
                }
                __nv_bfloat162 prefix_pair = __as_bf16x2(smem_regs_1286[0]);
                __nv_bfloat162 delta_pair = __as_bf16x2(smem_regs_1288[0]);
                __nv_bfloat162 updated_bf16_pair = prefix_pair + delta_pair;
                values_1287[0] = (float)updated_bf16_pair.x;
                values_1287[1] = (float)updated_bf16_pair.y;
                __nv_bfloat162 prefix_pair_1289 = __as_bf16x2(smem_regs_1286[1]);
                __nv_bfloat162 delta_pair_1290 = __as_bf16x2(smem_regs_1288[1]);
                __nv_bfloat162 updated_bf16_pair_1291 = prefix_pair_1289 + delta_pair_1290;
                values_1287[2] = (float)updated_bf16_pair_1291.x;
                values_1287[3] = (float)updated_bf16_pair_1291.y;
                __nv_bfloat162 prefix_pair_1292 = __as_bf16x2(smem_regs_1286[2]);
                __nv_bfloat162 delta_pair_1293 = __as_bf16x2(smem_regs_1288[2]);
                __nv_bfloat162 updated_bf16_pair_1294 = prefix_pair_1292 + delta_pair_1293;
                values_1287[4] = (float)updated_bf16_pair_1294.x;
                values_1287[5] = (float)updated_bf16_pair_1294.y;
                __nv_bfloat162 prefix_pair_1295 = __as_bf16x2(smem_regs_1286[3]);
                __nv_bfloat162 delta_pair_1296 = __as_bf16x2(smem_regs_1288[3]);
                __nv_bfloat162 updated_bf16_pair_1297 = prefix_pair_1295 + delta_pair_1296;
                values_1287[6] = (float)updated_bf16_pair_1297.x;
                values_1287[7] = (float)updated_bf16_pair_1297.y;
                unsigned int _packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1287[_lp*2 + 0], values_1287[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1287[_lp*2 + 2], values_1287[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1287[_lp*2 + 4], values_1287[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1287[_lp*2 + 6], values_1287[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1285))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                tmem_st_x8_f32(tmem_base, values_1287);
                float2 value_pair_1298 = make_float2(values_1287[0], values_1287[1]);
                float2 q_pair_1299 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1298, value_pair_1298, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1298, q_pair_1299, local_dot_1284[0]);
                float2 value_pair_1300 = make_float2(values_1287[2], values_1287[3]);
                float2 q_pair_1301 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1300, value_pair_1300, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1300, q_pair_1301, local_dot_1284[0]);
                float2 value_pair_1302 = make_float2(values_1287[4], values_1287[5]);
                float2 q_pair_1303 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1302, value_pair_1302, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1302, q_pair_1303, local_dot_1284[0]);
                float2 value_pair_1304 = make_float2(values_1287[6], values_1287[7]);
                float2 q_pair_1305 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1304, value_pair_1304, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1304, q_pair_1305, local_dot_1284[0]);
                int hidden_base_1306 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_1307[4];
                #pragma unroll
                for (int _lr_50 = 0; _lr_50 < 4; _lr_50++) {
                    smem_regs_1307[_lr_50] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1306 / 2) + (unsigned int)_lr_50];
                }
                float values_1308[8];
                unsigned int smem_regs_1309[4];
                #pragma unroll
                for (int _lr_51 = 0; _lr_51 < 4; _lr_51++) {
                    smem_regs_1309[_lr_51] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1306 / 2) + (unsigned int)_lr_51];
                }
                __nv_bfloat162 prefix_pair_1310 = __as_bf16x2(smem_regs_1307[0]);
                __nv_bfloat162 delta_pair_1311 = __as_bf16x2(smem_regs_1309[0]);
                __nv_bfloat162 updated_bf16_pair_1312 = prefix_pair_1310 + delta_pair_1311;
                values_1308[0] = (float)updated_bf16_pair_1312.x;
                values_1308[1] = (float)updated_bf16_pair_1312.y;
                __nv_bfloat162 prefix_pair_1313 = __as_bf16x2(smem_regs_1307[1]);
                __nv_bfloat162 delta_pair_1314 = __as_bf16x2(smem_regs_1309[1]);
                __nv_bfloat162 updated_bf16_pair_1315 = prefix_pair_1313 + delta_pair_1314;
                values_1308[2] = (float)updated_bf16_pair_1315.x;
                values_1308[3] = (float)updated_bf16_pair_1315.y;
                __nv_bfloat162 prefix_pair_1316 = __as_bf16x2(smem_regs_1307[2]);
                __nv_bfloat162 delta_pair_1317 = __as_bf16x2(smem_regs_1309[2]);
                __nv_bfloat162 updated_bf16_pair_1318 = prefix_pair_1316 + delta_pair_1317;
                values_1308[4] = (float)updated_bf16_pair_1318.x;
                values_1308[5] = (float)updated_bf16_pair_1318.y;
                __nv_bfloat162 prefix_pair_1319 = __as_bf16x2(smem_regs_1307[3]);
                __nv_bfloat162 delta_pair_1320 = __as_bf16x2(smem_regs_1309[3]);
                __nv_bfloat162 updated_bf16_pair_1321 = prefix_pair_1319 + delta_pair_1320;
                values_1308[6] = (float)updated_bf16_pair_1321.x;
                values_1308[7] = (float)updated_bf16_pair_1321.y;
                unsigned int _packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1308[_lp*2 + 0], values_1308[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1308[_lp*2 + 2], values_1308[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1308[_lp*2 + 4], values_1308[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1308[_lp*2 + 6], values_1308[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1306))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                tmem_st_x8_f32(tmem_base + 8, values_1308);
                float2 value_pair_1322 = make_float2(values_1308[0], values_1308[1]);
                float2 q_pair_1323 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1322, value_pair_1322, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1322, q_pair_1323, local_dot_1284[0]);
                float2 value_pair_1324 = make_float2(values_1308[2], values_1308[3]);
                float2 q_pair_1325 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1324, value_pair_1324, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1324, q_pair_1325, local_dot_1284[0]);
                float2 value_pair_1326 = make_float2(values_1308[4], values_1308[5]);
                float2 q_pair_1327 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1326, value_pair_1326, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1326, q_pair_1327, local_dot_1284[0]);
                float2 value_pair_1328 = make_float2(values_1308[6], values_1308[7]);
                float2 q_pair_1329 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1328, value_pair_1328, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1328, q_pair_1329, local_dot_1284[0]);
                int hidden_base_1330 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_1331[4];
                #pragma unroll
                for (int _lr_52 = 0; _lr_52 < 4; _lr_52++) {
                    smem_regs_1331[_lr_52] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1330 / 2) + (unsigned int)_lr_52];
                }
                float values_1332[8];
                unsigned int smem_regs_1333[4];
                #pragma unroll
                for (int _lr_53 = 0; _lr_53 < 4; _lr_53++) {
                    smem_regs_1333[_lr_53] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1330 / 2) + (unsigned int)_lr_53];
                }
                __nv_bfloat162 prefix_pair_1334 = __as_bf16x2(smem_regs_1331[0]);
                __nv_bfloat162 delta_pair_1335 = __as_bf16x2(smem_regs_1333[0]);
                __nv_bfloat162 updated_bf16_pair_1336 = prefix_pair_1334 + delta_pair_1335;
                values_1332[0] = (float)updated_bf16_pair_1336.x;
                values_1332[1] = (float)updated_bf16_pair_1336.y;
                __nv_bfloat162 prefix_pair_1337 = __as_bf16x2(smem_regs_1331[1]);
                __nv_bfloat162 delta_pair_1338 = __as_bf16x2(smem_regs_1333[1]);
                __nv_bfloat162 updated_bf16_pair_1339 = prefix_pair_1337 + delta_pair_1338;
                values_1332[2] = (float)updated_bf16_pair_1339.x;
                values_1332[3] = (float)updated_bf16_pair_1339.y;
                __nv_bfloat162 prefix_pair_1340 = __as_bf16x2(smem_regs_1331[2]);
                __nv_bfloat162 delta_pair_1341 = __as_bf16x2(smem_regs_1333[2]);
                __nv_bfloat162 updated_bf16_pair_1342 = prefix_pair_1340 + delta_pair_1341;
                values_1332[4] = (float)updated_bf16_pair_1342.x;
                values_1332[5] = (float)updated_bf16_pair_1342.y;
                __nv_bfloat162 prefix_pair_1343 = __as_bf16x2(smem_regs_1331[3]);
                __nv_bfloat162 delta_pair_1344 = __as_bf16x2(smem_regs_1333[3]);
                __nv_bfloat162 updated_bf16_pair_1345 = prefix_pair_1343 + delta_pair_1344;
                values_1332[6] = (float)updated_bf16_pair_1345.x;
                values_1332[7] = (float)updated_bf16_pair_1345.y;
                unsigned int _packed_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1332[_lp*2 + 0], values_1332[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1332[_lp*2 + 2], values_1332[_lp*2+1 + 2]));
                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1332[_lp*2 + 4], values_1332[_lp*2+1 + 4]));
                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1332[_lp*2 + 6], values_1332[_lp*2+1 + 6]));
                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1330))[0] = reinterpret_cast<int4*>(_packed_2)[0];
                tmem_st_x8_f32(tmem_base + 16, values_1332);
                float2 value_pair_1346 = make_float2(values_1332[0], values_1332[1]);
                float2 q_pair_1347 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1346, value_pair_1346, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1346, q_pair_1347, local_dot_1284[0]);
                float2 value_pair_1348 = make_float2(values_1332[2], values_1332[3]);
                float2 q_pair_1349 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1348, value_pair_1348, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1348, q_pair_1349, local_dot_1284[0]);
                float2 value_pair_1350 = make_float2(values_1332[4], values_1332[5]);
                float2 q_pair_1351 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1350, value_pair_1350, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1350, q_pair_1351, local_dot_1284[0]);
                float2 value_pair_1352 = make_float2(values_1332[6], values_1332[7]);
                float2 q_pair_1353 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1352, value_pair_1352, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1352, q_pair_1353, local_dot_1284[0]);
                int hidden_base_1354 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_1355[2];
                #pragma unroll
                for (int _lr_54 = 0; _lr_54 < 2; _lr_54++) {
                    smem_regs_1355[_lr_54] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1354 / 2) + (unsigned int)_lr_54];
                }
                float values_1356[4];
                unsigned int smem_regs_1357[2];
                #pragma unroll
                for (int _lr_55 = 0; _lr_55 < 2; _lr_55++) {
                    smem_regs_1357[_lr_55] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1354 / 2) + (unsigned int)_lr_55];
                }
                __nv_bfloat162 prefix_pair_1358 = __as_bf16x2(smem_regs_1355[0]);
                __nv_bfloat162 delta_pair_1359 = __as_bf16x2(smem_regs_1357[0]);
                __nv_bfloat162 updated_bf16_pair_1360 = prefix_pair_1358 + delta_pair_1359;
                values_1356[0] = (float)updated_bf16_pair_1360.x;
                values_1356[1] = (float)updated_bf16_pair_1360.y;
                __nv_bfloat162 prefix_pair_1361 = __as_bf16x2(smem_regs_1355[1]);
                __nv_bfloat162 delta_pair_1362 = __as_bf16x2(smem_regs_1357[1]);
                __nv_bfloat162 updated_bf16_pair_1363 = prefix_pair_1361 + delta_pair_1362;
                values_1356[2] = (float)updated_bf16_pair_1363.x;
                values_1356[3] = (float)updated_bf16_pair_1363.y;
                unsigned int _packed_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1356[_lp*2 + 0], values_1356[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1356[_lp*2 + 2], values_1356[_lp*2+1 + 2]));
                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(prefix + (token_1 * 7168 + hidden_base_1354))[0] = reinterpret_cast<int2*>(_packed_3)[0];
                tmem_st_x4_f32(tmem_base + 24, values_1356);
                float2 value_pair_1364 = make_float2(values_1356[0], values_1356[1]);
                float2 q_pair_1365 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1364, value_pair_1364, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1364, q_pair_1365, local_dot_1284[0]);
                float2 value_pair_1366 = make_float2(values_1356[2], values_1356[3]);
                float2 q_pair_1367 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1366, value_pair_1366, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1366, q_pair_1367, local_dot_1284[0]);
                int hidden_base_1368 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_1369[4];
                #pragma unroll
                for (int _lr_56 = 0; _lr_56 < 4; _lr_56++) {
                    smem_regs_1369[_lr_56] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1368 / 2) + (unsigned int)_lr_56];
                }
                float values_1370[8];
                unsigned int smem_regs_1371[4];
                #pragma unroll
                for (int _lr_57 = 0; _lr_57 < 4; _lr_57++) {
                    smem_regs_1371[_lr_57] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1368 / 2) + (unsigned int)_lr_57];
                }
                __nv_bfloat162 prefix_pair_1372 = __as_bf16x2(smem_regs_1369[0]);
                __nv_bfloat162 delta_pair_1373 = __as_bf16x2(smem_regs_1371[0]);
                __nv_bfloat162 updated_bf16_pair_1374 = prefix_pair_1372 + delta_pair_1373;
                values_1370[0] = (float)updated_bf16_pair_1374.x;
                values_1370[1] = (float)updated_bf16_pair_1374.y;
                __nv_bfloat162 prefix_pair_1375 = __as_bf16x2(smem_regs_1369[1]);
                __nv_bfloat162 delta_pair_1376 = __as_bf16x2(smem_regs_1371[1]);
                __nv_bfloat162 updated_bf16_pair_1377 = prefix_pair_1375 + delta_pair_1376;
                values_1370[2] = (float)updated_bf16_pair_1377.x;
                values_1370[3] = (float)updated_bf16_pair_1377.y;
                __nv_bfloat162 prefix_pair_1378 = __as_bf16x2(smem_regs_1369[2]);
                __nv_bfloat162 delta_pair_1379 = __as_bf16x2(smem_regs_1371[2]);
                __nv_bfloat162 updated_bf16_pair_1380 = prefix_pair_1378 + delta_pair_1379;
                values_1370[4] = (float)updated_bf16_pair_1380.x;
                values_1370[5] = (float)updated_bf16_pair_1380.y;
                __nv_bfloat162 prefix_pair_1381 = __as_bf16x2(smem_regs_1369[3]);
                __nv_bfloat162 delta_pair_1382 = __as_bf16x2(smem_regs_1371[3]);
                __nv_bfloat162 updated_bf16_pair_1383 = prefix_pair_1381 + delta_pair_1382;
                values_1370[6] = (float)updated_bf16_pair_1383.x;
                values_1370[7] = (float)updated_bf16_pair_1383.y;
                unsigned int _packed_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1370[_lp*2 + 0], values_1370[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1370[_lp*2 + 2], values_1370[_lp*2+1 + 2]));
                    _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1370[_lp*2 + 4], values_1370[_lp*2+1 + 4]));
                    _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1370[_lp*2 + 6], values_1370[_lp*2+1 + 6]));
                    _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1368))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                tmem_st_x8_f32(tmem_base + 32, values_1370);
                float2 value_pair_1384 = make_float2(values_1370[0], values_1370[1]);
                float2 q_pair_1385 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1384, value_pair_1384, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1384, q_pair_1385, local_dot_1284[2]);
                float2 value_pair_1386 = make_float2(values_1370[2], values_1370[3]);
                float2 q_pair_1387 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1386, value_pair_1386, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1386, q_pair_1387, local_dot_1284[2]);
                float2 value_pair_1388 = make_float2(values_1370[4], values_1370[5]);
                float2 q_pair_1389 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1388, value_pair_1388, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1388, q_pair_1389, local_dot_1284[2]);
                float2 value_pair_1390 = make_float2(values_1370[6], values_1370[7]);
                float2 q_pair_1391 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1390, value_pair_1390, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1390, q_pair_1391, local_dot_1284[2]);
                int hidden_base_1392 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_1393[4];
                #pragma unroll
                for (int _lr_58 = 0; _lr_58 < 4; _lr_58++) {
                    smem_regs_1393[_lr_58] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1392 / 2) + (unsigned int)_lr_58];
                }
                float values_1394[8];
                unsigned int smem_regs_1395[4];
                #pragma unroll
                for (int _lr_59 = 0; _lr_59 < 4; _lr_59++) {
                    smem_regs_1395[_lr_59] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1392 / 2) + (unsigned int)_lr_59];
                }
                __nv_bfloat162 prefix_pair_1396 = __as_bf16x2(smem_regs_1393[0]);
                __nv_bfloat162 delta_pair_1397 = __as_bf16x2(smem_regs_1395[0]);
                __nv_bfloat162 updated_bf16_pair_1398 = prefix_pair_1396 + delta_pair_1397;
                values_1394[0] = (float)updated_bf16_pair_1398.x;
                values_1394[1] = (float)updated_bf16_pair_1398.y;
                __nv_bfloat162 prefix_pair_1399 = __as_bf16x2(smem_regs_1393[1]);
                __nv_bfloat162 delta_pair_1400 = __as_bf16x2(smem_regs_1395[1]);
                __nv_bfloat162 updated_bf16_pair_1401 = prefix_pair_1399 + delta_pair_1400;
                values_1394[2] = (float)updated_bf16_pair_1401.x;
                values_1394[3] = (float)updated_bf16_pair_1401.y;
                __nv_bfloat162 prefix_pair_1402 = __as_bf16x2(smem_regs_1393[2]);
                __nv_bfloat162 delta_pair_1403 = __as_bf16x2(smem_regs_1395[2]);
                __nv_bfloat162 updated_bf16_pair_1404 = prefix_pair_1402 + delta_pair_1403;
                values_1394[4] = (float)updated_bf16_pair_1404.x;
                values_1394[5] = (float)updated_bf16_pair_1404.y;
                __nv_bfloat162 prefix_pair_1405 = __as_bf16x2(smem_regs_1393[3]);
                __nv_bfloat162 delta_pair_1406 = __as_bf16x2(smem_regs_1395[3]);
                __nv_bfloat162 updated_bf16_pair_1407 = prefix_pair_1405 + delta_pair_1406;
                values_1394[6] = (float)updated_bf16_pair_1407.x;
                values_1394[7] = (float)updated_bf16_pair_1407.y;
                unsigned int _packed_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1394[_lp*2 + 0], values_1394[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1394[_lp*2 + 2], values_1394[_lp*2+1 + 2]));
                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1394[_lp*2 + 4], values_1394[_lp*2+1 + 4]));
                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1394[_lp*2 + 6], values_1394[_lp*2+1 + 6]));
                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1392))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                tmem_st_x8_f32(tmem_base + 40, values_1394);
                float2 value_pair_1408 = make_float2(values_1394[0], values_1394[1]);
                float2 q_pair_1409 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1408, value_pair_1408, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1408, q_pair_1409, local_dot_1284[2]);
                float2 value_pair_1410 = make_float2(values_1394[2], values_1394[3]);
                float2 q_pair_1411 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1410, value_pair_1410, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1410, q_pair_1411, local_dot_1284[2]);
                float2 value_pair_1412 = make_float2(values_1394[4], values_1394[5]);
                float2 q_pair_1413 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1412, value_pair_1412, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1412, q_pair_1413, local_dot_1284[2]);
                float2 value_pair_1414 = make_float2(values_1394[6], values_1394[7]);
                float2 q_pair_1415 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1414, value_pair_1414, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1414, q_pair_1415, local_dot_1284[2]);
                int hidden_base_1416 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_1417[4];
                #pragma unroll
                for (int _lr_60 = 0; _lr_60 < 4; _lr_60++) {
                    smem_regs_1417[_lr_60] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1416 / 2) + (unsigned int)_lr_60];
                }
                float values_1418[8];
                unsigned int smem_regs_1419[4];
                #pragma unroll
                for (int _lr_61 = 0; _lr_61 < 4; _lr_61++) {
                    smem_regs_1419[_lr_61] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1416 / 2) + (unsigned int)_lr_61];
                }
                __nv_bfloat162 prefix_pair_1420 = __as_bf16x2(smem_regs_1417[0]);
                __nv_bfloat162 delta_pair_1421 = __as_bf16x2(smem_regs_1419[0]);
                __nv_bfloat162 updated_bf16_pair_1422 = prefix_pair_1420 + delta_pair_1421;
                values_1418[0] = (float)updated_bf16_pair_1422.x;
                values_1418[1] = (float)updated_bf16_pair_1422.y;
                __nv_bfloat162 prefix_pair_1423 = __as_bf16x2(smem_regs_1417[1]);
                __nv_bfloat162 delta_pair_1424 = __as_bf16x2(smem_regs_1419[1]);
                __nv_bfloat162 updated_bf16_pair_1425 = prefix_pair_1423 + delta_pair_1424;
                values_1418[2] = (float)updated_bf16_pair_1425.x;
                values_1418[3] = (float)updated_bf16_pair_1425.y;
                __nv_bfloat162 prefix_pair_1426 = __as_bf16x2(smem_regs_1417[2]);
                __nv_bfloat162 delta_pair_1427 = __as_bf16x2(smem_regs_1419[2]);
                __nv_bfloat162 updated_bf16_pair_1428 = prefix_pair_1426 + delta_pair_1427;
                values_1418[4] = (float)updated_bf16_pair_1428.x;
                values_1418[5] = (float)updated_bf16_pair_1428.y;
                __nv_bfloat162 prefix_pair_1429 = __as_bf16x2(smem_regs_1417[3]);
                __nv_bfloat162 delta_pair_1430 = __as_bf16x2(smem_regs_1419[3]);
                __nv_bfloat162 updated_bf16_pair_1431 = prefix_pair_1429 + delta_pair_1430;
                values_1418[6] = (float)updated_bf16_pair_1431.x;
                values_1418[7] = (float)updated_bf16_pair_1431.y;
                unsigned int _packed_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1418[_lp*2 + 0], values_1418[_lp*2+1 + 0]));
                    _packed_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1418[_lp*2 + 2], values_1418[_lp*2+1 + 2]));
                    _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1418[_lp*2 + 4], values_1418[_lp*2+1 + 4]));
                    _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1418[_lp*2 + 6], values_1418[_lp*2+1 + 6]));
                    _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1416))[0] = reinterpret_cast<int4*>(_packed_6)[0];
                tmem_st_x8_f32(tmem_base + 48, values_1418);
                float2 value_pair_1432 = make_float2(values_1418[0], values_1418[1]);
                float2 q_pair_1433 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1432, value_pair_1432, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1432, q_pair_1433, local_dot_1284[2]);
                float2 value_pair_1434 = make_float2(values_1418[2], values_1418[3]);
                float2 q_pair_1435 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1434, value_pair_1434, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1434, q_pair_1435, local_dot_1284[2]);
                float2 value_pair_1436 = make_float2(values_1418[4], values_1418[5]);
                float2 q_pair_1437 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1436, value_pair_1436, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1436, q_pair_1437, local_dot_1284[2]);
                float2 value_pair_1438 = make_float2(values_1418[6], values_1418[7]);
                float2 q_pair_1439 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1438, value_pair_1438, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1438, q_pair_1439, local_dot_1284[2]);
                int hidden_base_1440 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_1441[2];
                #pragma unroll
                for (int _lr_62 = 0; _lr_62 < 2; _lr_62++) {
                    smem_regs_1441[_lr_62] = source_words[ring_stage_1282 * 2 * 3584 + (unsigned int)(hidden_base_1440 / 2) + (unsigned int)_lr_62];
                }
                float values_1442[4];
                unsigned int smem_regs_1443[2];
                #pragma unroll
                for (int _lr_63 = 0; _lr_63 < 2; _lr_63++) {
                    smem_regs_1443[_lr_63] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1440 / 2) + (unsigned int)_lr_63];
                }
                __nv_bfloat162 prefix_pair_1444 = __as_bf16x2(smem_regs_1441[0]);
                __nv_bfloat162 delta_pair_1445 = __as_bf16x2(smem_regs_1443[0]);
                __nv_bfloat162 updated_bf16_pair_1446 = prefix_pair_1444 + delta_pair_1445;
                values_1442[0] = (float)updated_bf16_pair_1446.x;
                values_1442[1] = (float)updated_bf16_pair_1446.y;
                __nv_bfloat162 prefix_pair_1447 = __as_bf16x2(smem_regs_1441[1]);
                __nv_bfloat162 delta_pair_1448 = __as_bf16x2(smem_regs_1443[1]);
                __nv_bfloat162 updated_bf16_pair_1449 = prefix_pair_1447 + delta_pair_1448;
                values_1442[2] = (float)updated_bf16_pair_1449.x;
                values_1442[3] = (float)updated_bf16_pair_1449.y;
                unsigned int _packed_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1442[_lp*2 + 0], values_1442[_lp*2+1 + 0]));
                    _packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1442[_lp*2 + 2], values_1442[_lp*2+1 + 2]));
                    _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(prefix + (token_1 * 7168 + hidden_base_1440))[0] = reinterpret_cast<int2*>(_packed_7)[0];
                tmem_st_x4_f32(tmem_base + 56, values_1442);
                float2 value_pair_1450 = make_float2(values_1442[0], values_1442[1]);
                float2 q_pair_1451 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1450, value_pair_1450, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1450, q_pair_1451, local_dot_1284[2]);
                float2 value_pair_1452 = make_float2(values_1442[2], values_1442[3]);
                float2 q_pair_1453 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1452, value_pair_1452, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1452, q_pair_1453, local_dot_1284[2]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_1282) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1454 = stats_stream_base + 96;
                float2 reduce_pair_1455 = make_float2(local_sum_sq_1283[0].x + local_sum_sq_1283[0].y, local_dot_1284[0].x + local_dot_1284[0].y);
                unsigned long long packed_pair_1456 = 0;
                unsigned long long peer_packed_1457 = 0;
                float2 peer_pair_1458 = make_float2(0.0f, 0.0f);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 16);
                peer_packed_1457 = _shfl_xor_sync_168;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 8);
                peer_packed_1457 = _shfl_xor_sync_169;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 4);
                peer_packed_1457 = _shfl_xor_sync_170;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 2);
                peer_packed_1457 = _shfl_xor_sync_171;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 1);
                peer_packed_1457 = _shfl_xor_sync_172;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                if (lane == 0) {
                    stats[slot_base_1454 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1455.x;
                    stats[slot_base_1454 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1455.y;
                }
                float2 reduce_pair_1459 = make_float2(local_sum_sq_1283[2].x + local_sum_sq_1283[2].y, local_dot_1284[2].x + local_dot_1284[2].y);
                unsigned long long packed_pair_1460 = 0;
                unsigned long long peer_packed_1461 = 0;
                float2 peer_pair_1462 = make_float2(0.0f, 0.0f);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 16);
                peer_packed_1461 = _shfl_xor_sync_173;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 8);
                peer_packed_1461 = _shfl_xor_sync_174;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 4);
                peer_packed_1461 = _shfl_xor_sync_175;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 2);
                peer_packed_1461 = _shfl_xor_sync_176;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 1);
                peer_packed_1461 = _shfl_xor_sync_177;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                if (lane == 0) {
                    stats[slot_base_1454 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1459.x;
                    stats[slot_base_1454 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1459.y;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int stat_source_1463 = lane / 8;
                int stat_warp_1464 = lane % 8;
                float total_sum_sq_1465 = 0.0f;
                float total_dot_1466 = 0.0f;
                if (stat_source_1463 < 1) {
                    total_sum_sq_1465 = stats[slot_base_1454 + (unsigned int)((stat_warp_1464 * 2 + stat_source_1463) * 2)];
                    total_dot_1466 = stats[slot_base_1454 + (unsigned int)((stat_warp_1464 * 2 + stat_source_1463) * 2) + 1];
                }
                float _shfl_down_sync_178 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1465, 4, 8);
                total_sum_sq_1465 += _shfl_down_sync_178;
                float _shfl_down_sync_179 = __shfl_down_sync(0xFFFFFFFF, total_dot_1466, 4, 8);
                total_dot_1466 += _shfl_down_sync_179;
                float _shfl_down_sync_180 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1465, 2, 8);
                total_sum_sq_1465 += _shfl_down_sync_180;
                float _shfl_down_sync_181 = __shfl_down_sync(0xFFFFFFFF, total_dot_1466, 2, 8);
                total_dot_1466 += _shfl_down_sync_181;
                float _shfl_down_sync_182 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1465, 1, 8);
                total_sum_sq_1465 += _shfl_down_sync_182;
                float _shfl_down_sync_183 = __shfl_down_sync(0xFFFFFFFF, total_dot_1466, 1, 8);
                total_dot_1466 += _shfl_down_sync_183;
                float local_logit_1467 = 0.0f;
                if (stat_source_1463 < 1 && stat_warp_1464 == 0) {
                    float _rsqrt_184 = rsqrtf(total_sum_sq_1465 / 7168.0f + eps);
                    local_logit_1467 = total_dot_1466 * _rsqrt_184;
                }
                float logits_1468[2];
                logits_1468[0] = 0.0f;
                float _shfl_sync_185 = __shfl_sync(0xFFFFFFFF, local_logit_1467, 0);
                logits_1468[0] = _shfl_sync_185;
                logits_1468[1] = 0.0f;
                float _shfl_sync_186 = __shfl_sync(0xFFFFFFFF, local_logit_1467, 8);
                logits_1468[1] = _shfl_sync_186;
                float weights_1469[2];
                float new_max_1470 = 0.0f;
                float correction_1471 = 0.0f;
                float weight_sum_1472 = 0.0f;
                weights_1469[0] = 0.0f;
                weights_1469[1] = 0.0f;
                float chunk_max_1473 = -3.4028235e+38f;
                chunk_max_1473 = max_noftz(chunk_max_1473, logits_1468[0]);
                new_max_1470 = max_noftz(m_running, chunk_max_1473);
                float _exp2_187 = approx_exp2((m_running - new_max_1470) * 1.442695f);
                correction_1471 = _exp2_187;
                float _exp2_188 = approx_exp2((logits_1468[0] - new_max_1470) * 1.442695f);
                weights_1469[0] = _exp2_188;
                weight_sum_1472 += weights_1469[0];
                float2 correction_pair_1474 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1475 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1476;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1476) : "l"(*(const unsigned long long*)&acc_pair_1475), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[0] = scaled_pair_1476.x;
                acc32[1] = scaled_pair_1476.y;
                float2 acc_pair_1477 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1478;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1478) : "l"(*(const unsigned long long*)&acc_pair_1477), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[2] = scaled_pair_1478.x;
                acc32[3] = scaled_pair_1478.y;
                float2 acc_pair_1479 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1480;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1480) : "l"(*(const unsigned long long*)&acc_pair_1479), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[4] = scaled_pair_1480.x;
                acc32[5] = scaled_pair_1480.y;
                float2 acc_pair_1481 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1482;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1482) : "l"(*(const unsigned long long*)&acc_pair_1481), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[6] = scaled_pair_1482.x;
                acc32[7] = scaled_pair_1482.y;
                float _tmem_load_48[8];
                tmem_ld_x8(&_tmem_load_48[0], tmem_base);
                float2 weight_pair_1483 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1484 = make_float2(_tmem_load_48[0], _tmem_load_48[1]);
                float2 acc_pair_1485 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1486 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1484, acc_pair_1485);
                acc32[0] = updated_pair_1486.x;
                acc32[1] = updated_pair_1486.y;
                float2 cached_pair_1487 = make_float2(_tmem_load_48[2], _tmem_load_48[3]);
                float2 acc_pair_1488 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1489 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1487, acc_pair_1488);
                acc32[2] = updated_pair_1489.x;
                acc32[3] = updated_pair_1489.y;
                float2 cached_pair_1490 = make_float2(_tmem_load_48[4], _tmem_load_48[5]);
                float2 acc_pair_1491 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1492 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1490, acc_pair_1491);
                acc32[4] = updated_pair_1492.x;
                acc32[5] = updated_pair_1492.y;
                float2 cached_pair_1493 = make_float2(_tmem_load_48[6], _tmem_load_48[7]);
                float2 acc_pair_1494 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1495 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1493, acc_pair_1494);
                acc32[6] = updated_pair_1495.x;
                acc32[7] = updated_pair_1495.y;
                float2 correction_pair_1496 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1497 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1498;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1498) : "l"(*(const unsigned long long*)&acc_pair_1497), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[8] = scaled_pair_1498.x;
                acc32[9] = scaled_pair_1498.y;
                float2 acc_pair_1499 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1500;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1500) : "l"(*(const unsigned long long*)&acc_pair_1499), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[10] = scaled_pair_1500.x;
                acc32[11] = scaled_pair_1500.y;
                float2 acc_pair_1501 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1502;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1502) : "l"(*(const unsigned long long*)&acc_pair_1501), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[12] = scaled_pair_1502.x;
                acc32[13] = scaled_pair_1502.y;
                float2 acc_pair_1503 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1504;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1504) : "l"(*(const unsigned long long*)&acc_pair_1503), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[14] = scaled_pair_1504.x;
                acc32[15] = scaled_pair_1504.y;
                float _tmem_load_49[8];
                tmem_ld_x8(&_tmem_load_49[0], tmem_base + 8);
                float2 weight_pair_1505 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1506 = make_float2(_tmem_load_49[0], _tmem_load_49[1]);
                float2 acc_pair_1507 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1508 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1506, acc_pair_1507);
                acc32[8] = updated_pair_1508.x;
                acc32[9] = updated_pair_1508.y;
                float2 cached_pair_1509 = make_float2(_tmem_load_49[2], _tmem_load_49[3]);
                float2 acc_pair_1510 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1511 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1509, acc_pair_1510);
                acc32[10] = updated_pair_1511.x;
                acc32[11] = updated_pair_1511.y;
                float2 cached_pair_1512 = make_float2(_tmem_load_49[4], _tmem_load_49[5]);
                float2 acc_pair_1513 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1514 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1512, acc_pair_1513);
                acc32[12] = updated_pair_1514.x;
                acc32[13] = updated_pair_1514.y;
                float2 cached_pair_1515 = make_float2(_tmem_load_49[6], _tmem_load_49[7]);
                float2 acc_pair_1516 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1517 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1515, acc_pair_1516);
                acc32[14] = updated_pair_1517.x;
                acc32[15] = updated_pair_1517.y;
                float2 correction_pair_1518 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1519 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1520;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1520) : "l"(*(const unsigned long long*)&acc_pair_1519), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[16] = scaled_pair_1520.x;
                acc32[17] = scaled_pair_1520.y;
                float2 acc_pair_1521 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1522;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1522) : "l"(*(const unsigned long long*)&acc_pair_1521), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[18] = scaled_pair_1522.x;
                acc32[19] = scaled_pair_1522.y;
                float2 acc_pair_1523 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1524;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1524) : "l"(*(const unsigned long long*)&acc_pair_1523), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[20] = scaled_pair_1524.x;
                acc32[21] = scaled_pair_1524.y;
                float2 acc_pair_1525 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1526;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1526) : "l"(*(const unsigned long long*)&acc_pair_1525), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[22] = scaled_pair_1526.x;
                acc32[23] = scaled_pair_1526.y;
                float _tmem_load_50[8];
                tmem_ld_x8(&_tmem_load_50[0], tmem_base + 16);
                float2 weight_pair_1527 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1528 = make_float2(_tmem_load_50[0], _tmem_load_50[1]);
                float2 acc_pair_1529 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1530 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1528, acc_pair_1529);
                acc32[16] = updated_pair_1530.x;
                acc32[17] = updated_pair_1530.y;
                float2 cached_pair_1531 = make_float2(_tmem_load_50[2], _tmem_load_50[3]);
                float2 acc_pair_1532 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1533 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1531, acc_pair_1532);
                acc32[18] = updated_pair_1533.x;
                acc32[19] = updated_pair_1533.y;
                float2 cached_pair_1534 = make_float2(_tmem_load_50[4], _tmem_load_50[5]);
                float2 acc_pair_1535 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1536 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1534, acc_pair_1535);
                acc32[20] = updated_pair_1536.x;
                acc32[21] = updated_pair_1536.y;
                float2 cached_pair_1537 = make_float2(_tmem_load_50[6], _tmem_load_50[7]);
                float2 acc_pair_1538 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1539 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1537, acc_pair_1538);
                acc32[22] = updated_pair_1539.x;
                acc32[23] = updated_pair_1539.y;
                float2 correction_pair_1540 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1541 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1542;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1542) : "l"(*(const unsigned long long*)&acc_pair_1541), "l"(*(const unsigned long long*)&correction_pair_1540));
                acc32[24] = scaled_pair_1542.x;
                acc32[25] = scaled_pair_1542.y;
                float2 acc_pair_1543 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1544;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1544) : "l"(*(const unsigned long long*)&acc_pair_1543), "l"(*(const unsigned long long*)&correction_pair_1540));
                acc32[26] = scaled_pair_1544.x;
                acc32[27] = scaled_pair_1544.y;
                float _tmem_load_51[4];
                tmem_ld_x4(&_tmem_load_51[0], tmem_base + 24);
                float2 weight_pair_1545 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1546 = make_float2(_tmem_load_51[0], _tmem_load_51[1]);
                float2 acc_pair_1547 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1548 = fma_f32x2_rn_ftz(weight_pair_1545, cached_pair_1546, acc_pair_1547);
                acc32[24] = updated_pair_1548.x;
                acc32[25] = updated_pair_1548.y;
                float2 cached_pair_1549 = make_float2(_tmem_load_51[2], _tmem_load_51[3]);
                float2 acc_pair_1550 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1551 = fma_f32x2_rn_ftz(weight_pair_1545, cached_pair_1549, acc_pair_1550);
                acc32[26] = updated_pair_1551.x;
                acc32[27] = updated_pair_1551.y;
                float2 correction_pair_1552 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1553 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1554;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1554) : "l"(*(const unsigned long long*)&acc_pair_1553), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[28] = scaled_pair_1554.x;
                acc32[29] = scaled_pair_1554.y;
                float2 acc_pair_1555 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1556;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1556) : "l"(*(const unsigned long long*)&acc_pair_1555), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[30] = scaled_pair_1556.x;
                acc32[31] = scaled_pair_1556.y;
                float2 acc_pair_1557 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1558;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1558) : "l"(*(const unsigned long long*)&acc_pair_1557), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[32] = scaled_pair_1558.x;
                acc32[33] = scaled_pair_1558.y;
                float2 acc_pair_1559 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1560;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1560) : "l"(*(const unsigned long long*)&acc_pair_1559), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[34] = scaled_pair_1560.x;
                acc32[35] = scaled_pair_1560.y;
                float _tmem_load_52[8];
                tmem_ld_x8(&_tmem_load_52[0], tmem_base + 32);
                float2 weight_pair_1561 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1562 = make_float2(_tmem_load_52[0], _tmem_load_52[1]);
                float2 acc_pair_1563 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1564 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1562, acc_pair_1563);
                acc32[28] = updated_pair_1564.x;
                acc32[29] = updated_pair_1564.y;
                float2 cached_pair_1565 = make_float2(_tmem_load_52[2], _tmem_load_52[3]);
                float2 acc_pair_1566 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1567 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1565, acc_pair_1566);
                acc32[30] = updated_pair_1567.x;
                acc32[31] = updated_pair_1567.y;
                float2 cached_pair_1568 = make_float2(_tmem_load_52[4], _tmem_load_52[5]);
                float2 acc_pair_1569 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1570 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1568, acc_pair_1569);
                acc32[32] = updated_pair_1570.x;
                acc32[33] = updated_pair_1570.y;
                float2 cached_pair_1571 = make_float2(_tmem_load_52[6], _tmem_load_52[7]);
                float2 acc_pair_1572 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1573 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1571, acc_pair_1572);
                acc32[34] = updated_pair_1573.x;
                acc32[35] = updated_pair_1573.y;
                float2 correction_pair_1574 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1575 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1576;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1576) : "l"(*(const unsigned long long*)&acc_pair_1575), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[36] = scaled_pair_1576.x;
                acc32[37] = scaled_pair_1576.y;
                float2 acc_pair_1577 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1578;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1578) : "l"(*(const unsigned long long*)&acc_pair_1577), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[38] = scaled_pair_1578.x;
                acc32[39] = scaled_pair_1578.y;
                float2 acc_pair_1579 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1580;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1580) : "l"(*(const unsigned long long*)&acc_pair_1579), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[40] = scaled_pair_1580.x;
                acc32[41] = scaled_pair_1580.y;
                float2 acc_pair_1581 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1582;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1582) : "l"(*(const unsigned long long*)&acc_pair_1581), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[42] = scaled_pair_1582.x;
                acc32[43] = scaled_pair_1582.y;
                float _tmem_load_53[8];
                tmem_ld_x8(&_tmem_load_53[0], tmem_base + 40);
                float2 weight_pair_1583 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1584 = make_float2(_tmem_load_53[0], _tmem_load_53[1]);
                float2 acc_pair_1585 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1586 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1584, acc_pair_1585);
                acc32[36] = updated_pair_1586.x;
                acc32[37] = updated_pair_1586.y;
                float2 cached_pair_1587 = make_float2(_tmem_load_53[2], _tmem_load_53[3]);
                float2 acc_pair_1588 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1589 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1587, acc_pair_1588);
                acc32[38] = updated_pair_1589.x;
                acc32[39] = updated_pair_1589.y;
                float2 cached_pair_1590 = make_float2(_tmem_load_53[4], _tmem_load_53[5]);
                float2 acc_pair_1591 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1592 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1590, acc_pair_1591);
                acc32[40] = updated_pair_1592.x;
                acc32[41] = updated_pair_1592.y;
                float2 cached_pair_1593 = make_float2(_tmem_load_53[6], _tmem_load_53[7]);
                float2 acc_pair_1594 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1595 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1593, acc_pair_1594);
                acc32[42] = updated_pair_1595.x;
                acc32[43] = updated_pair_1595.y;
                float2 correction_pair_1596 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1597 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1598;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1598) : "l"(*(const unsigned long long*)&acc_pair_1597), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[44] = scaled_pair_1598.x;
                acc32[45] = scaled_pair_1598.y;
                float2 acc_pair_1599 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1600;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1600) : "l"(*(const unsigned long long*)&acc_pair_1599), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[46] = scaled_pair_1600.x;
                acc32[47] = scaled_pair_1600.y;
                float2 acc_pair_1601 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1602;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1602) : "l"(*(const unsigned long long*)&acc_pair_1601), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[48] = scaled_pair_1602.x;
                acc32[49] = scaled_pair_1602.y;
                float2 acc_pair_1603 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1604;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1604) : "l"(*(const unsigned long long*)&acc_pair_1603), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[50] = scaled_pair_1604.x;
                acc32[51] = scaled_pair_1604.y;
                float _tmem_load_54[8];
                tmem_ld_x8(&_tmem_load_54[0], tmem_base + 48);
                float2 weight_pair_1605 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1606 = make_float2(_tmem_load_54[0], _tmem_load_54[1]);
                float2 acc_pair_1607 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1608 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1606, acc_pair_1607);
                acc32[44] = updated_pair_1608.x;
                acc32[45] = updated_pair_1608.y;
                float2 cached_pair_1609 = make_float2(_tmem_load_54[2], _tmem_load_54[3]);
                float2 acc_pair_1610 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1611 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1609, acc_pair_1610);
                acc32[46] = updated_pair_1611.x;
                acc32[47] = updated_pair_1611.y;
                float2 cached_pair_1612 = make_float2(_tmem_load_54[4], _tmem_load_54[5]);
                float2 acc_pair_1613 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1614 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1612, acc_pair_1613);
                acc32[48] = updated_pair_1614.x;
                acc32[49] = updated_pair_1614.y;
                float2 cached_pair_1615 = make_float2(_tmem_load_54[6], _tmem_load_54[7]);
                float2 acc_pair_1616 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1617 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1615, acc_pair_1616);
                acc32[50] = updated_pair_1617.x;
                acc32[51] = updated_pair_1617.y;
                float2 correction_pair_1618 = make_float2(correction_1471, correction_1471);
                float2 acc_pair_1619 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1620;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1620) : "l"(*(const unsigned long long*)&acc_pair_1619), "l"(*(const unsigned long long*)&correction_pair_1618));
                acc32[52] = scaled_pair_1620.x;
                acc32[53] = scaled_pair_1620.y;
                float2 acc_pair_1621 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1622;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1622) : "l"(*(const unsigned long long*)&acc_pair_1621), "l"(*(const unsigned long long*)&correction_pair_1618));
                acc32[54] = scaled_pair_1622.x;
                acc32[55] = scaled_pair_1622.y;
                float _tmem_load_55[4];
                tmem_ld_x4(&_tmem_load_55[0], tmem_base + 56);
                float2 weight_pair_1623 = make_float2(weights_1469[0], weights_1469[0]);
                float2 cached_pair_1624 = make_float2(_tmem_load_55[0], _tmem_load_55[1]);
                float2 acc_pair_1625 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1626 = fma_f32x2_rn_ftz(weight_pair_1623, cached_pair_1624, acc_pair_1625);
                acc32[52] = updated_pair_1626.x;
                acc32[53] = updated_pair_1626.y;
                float2 cached_pair_1627 = make_float2(_tmem_load_55[2], _tmem_load_55[3]);
                float2 acc_pair_1628 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1629 = fma_f32x2_rn_ftz(weight_pair_1623, cached_pair_1627, acc_pair_1628);
                acc32[54] = updated_pair_1629.x;
                acc32[55] = updated_pair_1629.y;
                s_running = __fmaf_rn(s_running, correction_1471, weight_sum_1472);
                m_running = new_max_1470;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; _phase_ready ^= 1; }
                if (token_1 == first_token) {
                    mbarrier_wait(norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float2 local_output_sum_pair = make_float2(0.0f, 0.0f);
                float2 acc_pair_1630 = make_float2(acc32[0], acc32[1]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1630, acc_pair_1630, local_output_sum_pair);
                float2 acc_pair_1631 = make_float2(acc32[2], acc32[3]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1631, acc_pair_1631, local_output_sum_pair);
                float2 acc_pair_1632 = make_float2(acc32[4], acc32[5]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1632, acc_pair_1632, local_output_sum_pair);
                float2 acc_pair_1633 = make_float2(acc32[6], acc32[7]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1633, acc_pair_1633, local_output_sum_pair);
                float2 acc_pair_1634 = make_float2(acc32[8], acc32[9]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1634, acc_pair_1634, local_output_sum_pair);
                float2 acc_pair_1635 = make_float2(acc32[10], acc32[11]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1635, acc_pair_1635, local_output_sum_pair);
                float2 acc_pair_1636 = make_float2(acc32[12], acc32[13]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1636, acc_pair_1636, local_output_sum_pair);
                float2 acc_pair_1637 = make_float2(acc32[14], acc32[15]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1637, acc_pair_1637, local_output_sum_pair);
                float2 acc_pair_1638 = make_float2(acc32[16], acc32[17]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1638, acc_pair_1638, local_output_sum_pair);
                float2 acc_pair_1639 = make_float2(acc32[18], acc32[19]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1639, acc_pair_1639, local_output_sum_pair);
                float2 acc_pair_1640 = make_float2(acc32[20], acc32[21]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1640, acc_pair_1640, local_output_sum_pair);
                float2 acc_pair_1641 = make_float2(acc32[22], acc32[23]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1641, acc_pair_1641, local_output_sum_pair);
                float2 acc_pair_1642 = make_float2(acc32[24], acc32[25]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1642, acc_pair_1642, local_output_sum_pair);
                float2 acc_pair_1643 = make_float2(acc32[26], acc32[27]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1643, acc_pair_1643, local_output_sum_pair);
                float local_output_sum_sq = local_output_sum_pair.x + local_output_sum_pair.y;
                float _warp_reduce_189 = local_output_sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_189 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_189, offset);
                local_output_sum_sq = _warp_reduce_189;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)warp_in_group] = local_output_sum_sq;
                }
                float2 local_output_sum_pair_1644 = make_float2(0.0f, 0.0f);
                float2 acc_pair_1645 = make_float2(acc32[28], acc32[29]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1645, acc_pair_1645, local_output_sum_pair_1644);
                float2 acc_pair_1646 = make_float2(acc32[30], acc32[31]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1646, acc_pair_1646, local_output_sum_pair_1644);
                float2 acc_pair_1647 = make_float2(acc32[32], acc32[33]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1647, acc_pair_1647, local_output_sum_pair_1644);
                float2 acc_pair_1648 = make_float2(acc32[34], acc32[35]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1648, acc_pair_1648, local_output_sum_pair_1644);
                float2 acc_pair_1649 = make_float2(acc32[36], acc32[37]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1649, acc_pair_1649, local_output_sum_pair_1644);
                float2 acc_pair_1650 = make_float2(acc32[38], acc32[39]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1650, acc_pair_1650, local_output_sum_pair_1644);
                float2 acc_pair_1651 = make_float2(acc32[40], acc32[41]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1651, acc_pair_1651, local_output_sum_pair_1644);
                float2 acc_pair_1652 = make_float2(acc32[42], acc32[43]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1652, acc_pair_1652, local_output_sum_pair_1644);
                float2 acc_pair_1653 = make_float2(acc32[44], acc32[45]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1653, acc_pair_1653, local_output_sum_pair_1644);
                float2 acc_pair_1654 = make_float2(acc32[46], acc32[47]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1654, acc_pair_1654, local_output_sum_pair_1644);
                float2 acc_pair_1655 = make_float2(acc32[48], acc32[49]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1655, acc_pair_1655, local_output_sum_pair_1644);
                float2 acc_pair_1656 = make_float2(acc32[50], acc32[51]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1656, acc_pair_1656, local_output_sum_pair_1644);
                float2 acc_pair_1657 = make_float2(acc32[52], acc32[53]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1657, acc_pair_1657, local_output_sum_pair_1644);
                float2 acc_pair_1658 = make_float2(acc32[54], acc32[55]);
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1658, acc_pair_1658, local_output_sum_pair_1644);
                float local_output_sum_sq_1659 = local_output_sum_pair_1644.x + local_output_sum_pair_1644.y;
                float _warp_reduce_190 = local_output_sum_sq_1659;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_190 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_190, offset);
                local_output_sum_sq_1659 = _warp_reduce_190;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)(4 + warp_in_group)] = local_output_sum_sq_1659;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                float total_output_sum_sq = ((lane < 8) ? output_stats[output_stats_base + (unsigned int)lane] : 0.0f);
                float _shfl_down_sync_191 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 4, 8);
                total_output_sum_sq += _shfl_down_sync_191;
                float _shfl_down_sync_192 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 2, 8);
                total_output_sum_sq += _shfl_down_sync_192;
                float _shfl_down_sync_193 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 1, 8);
                total_output_sum_sq += _shfl_down_sync_193;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_194 = rsqrtf(__fmaf_rn(output_norm_eps * s_running, s_running, total_output_sum_sq / 7168.0f));
                    output_rsigma = _rsqrt_194;
                }
                float _shfl_sync_195 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_sync_195;
                int hidden_base_1660 = thread_in_group * 8;
                unsigned int smem_bf16x2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1660 * 2)) : "memory");
                float _unpack_bf16_48[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_48[_pair * 2])[0]), "=f"((&_unpack_bf16_48[_pair * 2])[1])
                        : "r"(smem_bf16x2[_pair]));
                }
                float2 output_rsigma_pair = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_196 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1661;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1661) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_197 = make_float2(_unpack_bf16_48[0], _unpack_bf16_48[1]);
                float2 normalized_pair;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair_1661), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
                acc32[0] = normalized_pair.x;
                acc32[1] = normalized_pair.y;
                float2 _f32x2_mul_a_198 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1662;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1662) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_199 = make_float2(_unpack_bf16_48[2], _unpack_bf16_48[3]);
                float2 normalized_pair_1663;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1663) : "l"(*(const unsigned long long*)&scaled_pair_1662), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
                acc32[2] = normalized_pair_1663.x;
                acc32[3] = normalized_pair_1663.y;
                float2 _f32x2_mul_a_200 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1664;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1664) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_201 = make_float2(_unpack_bf16_48[4], _unpack_bf16_48[5]);
                float2 normalized_pair_1665;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1665) : "l"(*(const unsigned long long*)&scaled_pair_1664), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
                acc32[4] = normalized_pair_1665.x;
                acc32[5] = normalized_pair_1665.y;
                float2 _f32x2_mul_a_202 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1666;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1666) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_203 = make_float2(_unpack_bf16_48[6], _unpack_bf16_48[7]);
                float2 normalized_pair_1667;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1667) : "l"(*(const unsigned long long*)&scaled_pair_1666), "l"(*(const unsigned long long*)&_f32x2_mul_b_203));
                acc32[6] = normalized_pair_1667.x;
                acc32[7] = normalized_pair_1667.y;
                unsigned int _packed_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 0], acc32[_lp*2+1 + 0]));
                    _packed_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 2], acc32[_lp*2+1 + 2]));
                    _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 4], acc32[_lp*2+1 + 4]));
                    _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 6], acc32[_lp*2+1 + 6]));
                    _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1660))[0] = reinterpret_cast<int4*>(_packed_8)[0];
                int hidden_base_1668 = 2048 + thread_in_group * 8;
                unsigned int smem_bf16x2_1669[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1669[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1669[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1669[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1669[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1668 * 2)) : "memory");
                float _unpack_bf16_49[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_49[_pair * 2])[0]), "=f"((&_unpack_bf16_49[_pair * 2])[1])
                        : "r"(smem_bf16x2_1669[_pair]));
                }
                float2 output_rsigma_pair_1670 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_204 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1671;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1671) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f32x2_mul_b_205 = make_float2(_unpack_bf16_49[0], _unpack_bf16_49[1]);
                float2 normalized_pair_1672;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1672) : "l"(*(const unsigned long long*)&scaled_pair_1671), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
                acc32[8] = normalized_pair_1672.x;
                acc32[9] = normalized_pair_1672.y;
                float2 _f32x2_mul_a_206 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1673;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1673) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f32x2_mul_b_207 = make_float2(_unpack_bf16_49[2], _unpack_bf16_49[3]);
                float2 normalized_pair_1674;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1674) : "l"(*(const unsigned long long*)&scaled_pair_1673), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
                acc32[10] = normalized_pair_1674.x;
                acc32[11] = normalized_pair_1674.y;
                float2 _f32x2_mul_a_208 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1675;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1675) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f32x2_mul_b_209 = make_float2(_unpack_bf16_49[4], _unpack_bf16_49[5]);
                float2 normalized_pair_1676;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1676) : "l"(*(const unsigned long long*)&scaled_pair_1675), "l"(*(const unsigned long long*)&_f32x2_mul_b_209));
                acc32[12] = normalized_pair_1676.x;
                acc32[13] = normalized_pair_1676.y;
                float2 _f32x2_mul_a_210 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1677;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1677) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f32x2_mul_b_211 = make_float2(_unpack_bf16_49[6], _unpack_bf16_49[7]);
                float2 normalized_pair_1678;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1678) : "l"(*(const unsigned long long*)&scaled_pair_1677), "l"(*(const unsigned long long*)&_f32x2_mul_b_211));
                acc32[14] = normalized_pair_1678.x;
                acc32[15] = normalized_pair_1678.y;
                unsigned int _packed_9[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 8], acc32[_lp*2+1 + 8]));
                    _packed_9[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 10], acc32[_lp*2+1 + 10]));
                    _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 12], acc32[_lp*2+1 + 12]));
                    _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 14], acc32[_lp*2+1 + 14]));
                    _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1668))[0] = reinterpret_cast<int4*>(_packed_9)[0];
                int hidden_base_1679 = 4096 + thread_in_group * 8;
                unsigned int smem_bf16x2_1680[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1680[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1680[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1680[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1680[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1679 * 2)) : "memory");
                float _unpack_bf16_50[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_50[_pair * 2])[0]), "=f"((&_unpack_bf16_50[_pair * 2])[1])
                        : "r"(smem_bf16x2_1680[_pair]));
                }
                float2 output_rsigma_pair_1681 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_212 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1682;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1682) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f32x2_mul_b_213 = make_float2(_unpack_bf16_50[0], _unpack_bf16_50[1]);
                float2 normalized_pair_1683;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1683) : "l"(*(const unsigned long long*)&scaled_pair_1682), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
                acc32[16] = normalized_pair_1683.x;
                acc32[17] = normalized_pair_1683.y;
                float2 _f32x2_mul_a_214 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1684;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1684) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f32x2_mul_b_215 = make_float2(_unpack_bf16_50[2], _unpack_bf16_50[3]);
                float2 normalized_pair_1685;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1685) : "l"(*(const unsigned long long*)&scaled_pair_1684), "l"(*(const unsigned long long*)&_f32x2_mul_b_215));
                acc32[18] = normalized_pair_1685.x;
                acc32[19] = normalized_pair_1685.y;
                float2 _f32x2_mul_a_216 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1686;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1686) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f32x2_mul_b_217 = make_float2(_unpack_bf16_50[4], _unpack_bf16_50[5]);
                float2 normalized_pair_1687;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1687) : "l"(*(const unsigned long long*)&scaled_pair_1686), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
                acc32[20] = normalized_pair_1687.x;
                acc32[21] = normalized_pair_1687.y;
                float2 _f32x2_mul_a_218 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1688;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1688) : "l"(*(const unsigned long long*)&_f32x2_mul_a_218), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f32x2_mul_b_219 = make_float2(_unpack_bf16_50[6], _unpack_bf16_50[7]);
                float2 normalized_pair_1689;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1689) : "l"(*(const unsigned long long*)&scaled_pair_1688), "l"(*(const unsigned long long*)&_f32x2_mul_b_219));
                acc32[22] = normalized_pair_1689.x;
                acc32[23] = normalized_pair_1689.y;
                unsigned int _packed_10[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 16], acc32[_lp*2+1 + 16]));
                    _packed_10[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 18], acc32[_lp*2+1 + 18]));
                    _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 20], acc32[_lp*2+1 + 20]));
                    _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 22], acc32[_lp*2+1 + 22]));
                    _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1679))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                int hidden_base_1690 = 6144 + thread_in_group * 4;
                unsigned int smem_bf16x2_1691[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1691[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1691[(0) + 1]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1690 * 2)) : "memory");
                float _unpack_bf16_51[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_51[_pair * 2])[0]), "=f"((&_unpack_bf16_51[_pair * 2])[1])
                        : "r"(smem_bf16x2_1691[_pair]));
                }
                float2 output_rsigma_pair_1692 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_220 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1693;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1693) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&output_rsigma_pair_1692));
                float2 _f32x2_mul_b_221 = make_float2(_unpack_bf16_51[0], _unpack_bf16_51[1]);
                float2 normalized_pair_1694;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1694) : "l"(*(const unsigned long long*)&scaled_pair_1693), "l"(*(const unsigned long long*)&_f32x2_mul_b_221));
                acc32[24] = normalized_pair_1694.x;
                acc32[25] = normalized_pair_1694.y;
                float2 _f32x2_mul_a_222 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1695;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1695) : "l"(*(const unsigned long long*)&_f32x2_mul_a_222), "l"(*(const unsigned long long*)&output_rsigma_pair_1692));
                float2 _f32x2_mul_b_223 = make_float2(_unpack_bf16_51[2], _unpack_bf16_51[3]);
                float2 normalized_pair_1696;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1696) : "l"(*(const unsigned long long*)&scaled_pair_1695), "l"(*(const unsigned long long*)&_f32x2_mul_b_223));
                acc32[26] = normalized_pair_1696.x;
                acc32[27] = normalized_pair_1696.y;
                unsigned int _packed_11[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 24], acc32[_lp*2+1 + 24]));
                    _packed_11[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 26], acc32[_lp*2+1 + 26]));
                    _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(out + (token_1 * 7168 + hidden_base_1690))[0] = reinterpret_cast<int2*>(_packed_11)[0];
                int hidden_base_1697 = 1024 + thread_in_group * 8;
                unsigned int smem_bf16x2_1698[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1698[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1698[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1698[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1698[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1697 * 2)) : "memory");
                float _unpack_bf16_52[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_52[_pair * 2])[0]), "=f"((&_unpack_bf16_52[_pair * 2])[1])
                        : "r"(smem_bf16x2_1698[_pair]));
                }
                float2 output_rsigma_pair_1699 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_224 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1700;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1700) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f32x2_mul_b_225 = make_float2(_unpack_bf16_52[0], _unpack_bf16_52[1]);
                float2 normalized_pair_1701;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1701) : "l"(*(const unsigned long long*)&scaled_pair_1700), "l"(*(const unsigned long long*)&_f32x2_mul_b_225));
                acc32[28] = normalized_pair_1701.x;
                acc32[29] = normalized_pair_1701.y;
                float2 _f32x2_mul_a_226 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1702;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1702) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f32x2_mul_b_227 = make_float2(_unpack_bf16_52[2], _unpack_bf16_52[3]);
                float2 normalized_pair_1703;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1703) : "l"(*(const unsigned long long*)&scaled_pair_1702), "l"(*(const unsigned long long*)&_f32x2_mul_b_227));
                acc32[30] = normalized_pair_1703.x;
                acc32[31] = normalized_pair_1703.y;
                float2 _f32x2_mul_a_228 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1704;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1704) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f32x2_mul_b_229 = make_float2(_unpack_bf16_52[4], _unpack_bf16_52[5]);
                float2 normalized_pair_1705;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1705) : "l"(*(const unsigned long long*)&scaled_pair_1704), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
                acc32[32] = normalized_pair_1705.x;
                acc32[33] = normalized_pair_1705.y;
                float2 _f32x2_mul_a_230 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1706;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1706) : "l"(*(const unsigned long long*)&_f32x2_mul_a_230), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f32x2_mul_b_231 = make_float2(_unpack_bf16_52[6], _unpack_bf16_52[7]);
                float2 normalized_pair_1707;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1707) : "l"(*(const unsigned long long*)&scaled_pair_1706), "l"(*(const unsigned long long*)&_f32x2_mul_b_231));
                acc32[34] = normalized_pair_1707.x;
                acc32[35] = normalized_pair_1707.y;
                unsigned int _packed_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 28], acc32[_lp*2+1 + 28]));
                    _packed_12[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 30], acc32[_lp*2+1 + 30]));
                    _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 32], acc32[_lp*2+1 + 32]));
                    _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 34], acc32[_lp*2+1 + 34]));
                    _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1697))[0] = reinterpret_cast<int4*>(_packed_12)[0];
                int hidden_base_1708 = 3072 + thread_in_group * 8;
                unsigned int smem_bf16x2_1709[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1709[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1709[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1709[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1709[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1708 * 2)) : "memory");
                float _unpack_bf16_53[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_53[_pair * 2])[0]), "=f"((&_unpack_bf16_53[_pair * 2])[1])
                        : "r"(smem_bf16x2_1709[_pair]));
                }
                float2 output_rsigma_pair_1710 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_232 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1711;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1711) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f32x2_mul_b_233 = make_float2(_unpack_bf16_53[0], _unpack_bf16_53[1]);
                float2 normalized_pair_1712;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1712) : "l"(*(const unsigned long long*)&scaled_pair_1711), "l"(*(const unsigned long long*)&_f32x2_mul_b_233));
                acc32[36] = normalized_pair_1712.x;
                acc32[37] = normalized_pair_1712.y;
                float2 _f32x2_mul_a_234 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1713;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1713) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f32x2_mul_b_235 = make_float2(_unpack_bf16_53[2], _unpack_bf16_53[3]);
                float2 normalized_pair_1714;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1714) : "l"(*(const unsigned long long*)&scaled_pair_1713), "l"(*(const unsigned long long*)&_f32x2_mul_b_235));
                acc32[38] = normalized_pair_1714.x;
                acc32[39] = normalized_pair_1714.y;
                float2 _f32x2_mul_a_236 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1715;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1715) : "l"(*(const unsigned long long*)&_f32x2_mul_a_236), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f32x2_mul_b_237 = make_float2(_unpack_bf16_53[4], _unpack_bf16_53[5]);
                float2 normalized_pair_1716;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1716) : "l"(*(const unsigned long long*)&scaled_pair_1715), "l"(*(const unsigned long long*)&_f32x2_mul_b_237));
                acc32[40] = normalized_pair_1716.x;
                acc32[41] = normalized_pair_1716.y;
                float2 _f32x2_mul_a_238 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1717;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1717) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f32x2_mul_b_239 = make_float2(_unpack_bf16_53[6], _unpack_bf16_53[7]);
                float2 normalized_pair_1718;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1718) : "l"(*(const unsigned long long*)&scaled_pair_1717), "l"(*(const unsigned long long*)&_f32x2_mul_b_239));
                acc32[42] = normalized_pair_1718.x;
                acc32[43] = normalized_pair_1718.y;
                unsigned int _packed_13[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 36], acc32[_lp*2+1 + 36]));
                    _packed_13[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 38], acc32[_lp*2+1 + 38]));
                    _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 40], acc32[_lp*2+1 + 40]));
                    _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 42], acc32[_lp*2+1 + 42]));
                    _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1708))[0] = reinterpret_cast<int4*>(_packed_13)[0];
                int hidden_base_1719 = 5120 + thread_in_group * 8;
                unsigned int smem_bf16x2_1720[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1720[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1720[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1720[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1720[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1719 * 2)) : "memory");
                float _unpack_bf16_54[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_54[_pair * 2])[0]), "=f"((&_unpack_bf16_54[_pair * 2])[1])
                        : "r"(smem_bf16x2_1720[_pair]));
                }
                float2 output_rsigma_pair_1721 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_240 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1722;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1722) : "l"(*(const unsigned long long*)&_f32x2_mul_a_240), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f32x2_mul_b_241 = make_float2(_unpack_bf16_54[0], _unpack_bf16_54[1]);
                float2 normalized_pair_1723;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1723) : "l"(*(const unsigned long long*)&scaled_pair_1722), "l"(*(const unsigned long long*)&_f32x2_mul_b_241));
                acc32[44] = normalized_pair_1723.x;
                acc32[45] = normalized_pair_1723.y;
                float2 _f32x2_mul_a_242 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1724;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1724) : "l"(*(const unsigned long long*)&_f32x2_mul_a_242), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f32x2_mul_b_243 = make_float2(_unpack_bf16_54[2], _unpack_bf16_54[3]);
                float2 normalized_pair_1725;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1725) : "l"(*(const unsigned long long*)&scaled_pair_1724), "l"(*(const unsigned long long*)&_f32x2_mul_b_243));
                acc32[46] = normalized_pair_1725.x;
                acc32[47] = normalized_pair_1725.y;
                float2 _f32x2_mul_a_244 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1726;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1726) : "l"(*(const unsigned long long*)&_f32x2_mul_a_244), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f32x2_mul_b_245 = make_float2(_unpack_bf16_54[4], _unpack_bf16_54[5]);
                float2 normalized_pair_1727;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1727) : "l"(*(const unsigned long long*)&scaled_pair_1726), "l"(*(const unsigned long long*)&_f32x2_mul_b_245));
                acc32[48] = normalized_pair_1727.x;
                acc32[49] = normalized_pair_1727.y;
                float2 _f32x2_mul_a_246 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1728;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1728) : "l"(*(const unsigned long long*)&_f32x2_mul_a_246), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f32x2_mul_b_247 = make_float2(_unpack_bf16_54[6], _unpack_bf16_54[7]);
                float2 normalized_pair_1729;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1729) : "l"(*(const unsigned long long*)&scaled_pair_1728), "l"(*(const unsigned long long*)&_f32x2_mul_b_247));
                acc32[50] = normalized_pair_1729.x;
                acc32[51] = normalized_pair_1729.y;
                unsigned int _packed_14[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 44], acc32[_lp*2+1 + 44]));
                    _packed_14[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 46], acc32[_lp*2+1 + 46]));
                    _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 48], acc32[_lp*2+1 + 48]));
                    _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 50], acc32[_lp*2+1 + 50]));
                    _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1719))[0] = reinterpret_cast<int4*>(_packed_14)[0];
                int hidden_base_1730 = 6656 + thread_in_group * 4;
                unsigned int smem_bf16x2_1731[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1731[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1731[(0) + 1]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1730 * 2)) : "memory");
                float _unpack_bf16_55[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_55[_pair * 2])[0]), "=f"((&_unpack_bf16_55[_pair * 2])[1])
                        : "r"(smem_bf16x2_1731[_pair]));
                }
                float2 output_rsigma_pair_1732 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_248 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1733;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1733) : "l"(*(const unsigned long long*)&_f32x2_mul_a_248), "l"(*(const unsigned long long*)&output_rsigma_pair_1732));
                float2 _f32x2_mul_b_249 = make_float2(_unpack_bf16_55[0], _unpack_bf16_55[1]);
                float2 normalized_pair_1734;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1734) : "l"(*(const unsigned long long*)&scaled_pair_1733), "l"(*(const unsigned long long*)&_f32x2_mul_b_249));
                acc32[52] = normalized_pair_1734.x;
                acc32[53] = normalized_pair_1734.y;
                float2 _f32x2_mul_a_250 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1735;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1735) : "l"(*(const unsigned long long*)&_f32x2_mul_a_250), "l"(*(const unsigned long long*)&output_rsigma_pair_1732));
                float2 _f32x2_mul_b_251 = make_float2(_unpack_bf16_55[2], _unpack_bf16_55[3]);
                float2 normalized_pair_1736;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1736) : "l"(*(const unsigned long long*)&scaled_pair_1735), "l"(*(const unsigned long long*)&_f32x2_mul_b_251));
                acc32[54] = normalized_pair_1736.x;
                acc32[55] = normalized_pair_1736.y;
                unsigned int _packed_15[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 52], acc32[_lp*2+1 + 52]));
                    _packed_15[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc32[_lp*2 + 54], acc32[_lp*2+1 + 54]));
                    _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(out + (token_1 * 7168 + hidden_base_1730))[0] = reinterpret_cast<int2*>(_packed_15)[0];
            }
            mbarrier_wait(norm_ready_addr, 0);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("barrier.sync 0, 320;" ::: "memory");
            if (consumer_warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }

    // Cleanup
}

} // extern "C"
