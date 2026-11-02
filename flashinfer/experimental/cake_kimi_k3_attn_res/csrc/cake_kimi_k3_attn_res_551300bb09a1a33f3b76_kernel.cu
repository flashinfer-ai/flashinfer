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
#define SMEM_STATS_STAGE_BYTES 768
#define SMEM_STATS_STRIDE 768
#define SMEM_OUTPUT_STATS_OFF 158464
#define SMEM_OUTPUT_STATS_STAGE_BYTES 64
#define SMEM_OUTPUT_STATS_STRIDE 64
#define SMEM_TOTAL 158720
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
kernel_cake_kimi_k3_attn_res_551300bb09a1a33f3b76(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 158528;
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
    float* output_stats = reinterpret_cast<float*>(smem_raw + 158464);
    const int output_stats_addr = smem + 158464;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[158528..158600)

    if (warp == 8) {
        uint32_t leader = elect_sync();
        if (leader) {
            // ready: 4 barriers, init_count=1
            mbarrier_init(smem + 158528, 1);
            mbarrier_init(smem + 158536, 1);
            mbarrier_init(smem + 158544, 1);
            mbarrier_init(smem + 158552, 1);
            // consumed: 4 barriers, init_count=128
            mbarrier_init(smem + 158560, 128);
            mbarrier_init(smem + 158568, 128);
            mbarrier_init(smem + 158576, 128);
            mbarrier_init(smem + 158584, 128);
            // norm_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 158592, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 158600);
    if (warp == 0) {
        int _tmem_hold = smem + 158600;
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
                    mbarrier_arrive_expect_tx(ready_addr + (stage + stream_u * 2) * 8, 43008);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + (stream_u * 2 + stage) * 2 * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + 4 * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
                        : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(source_buf_addr + ((stream_u * 2 + stage) * 2 + 1) * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(prefix) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage + stream_u * 2) * 8)
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
            unsigned int stats_stream_base = stream_u_1 * 96;
            unsigned int output_stats_base = stream_u_1 * 8;
            float q_cache[56];
            int hidden_base_1_0 = thread_in_group * 8;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1_0);
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
            float _vec_load_16_0[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1_0);
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
                            : "=f"((&_vec_load_16_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_2 = make_float2(_vec_load_0[0], _vec_load_0[1]);
            float2 _f32x2_mul_b_3 = make_float2(_vec_load_16_0[0], _vec_load_16_0[1]);
            float2 q_pair_35_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
            q_cache[0] = q_pair_35_0.x;
            q_cache[1] = q_pair_35_0.y;
            float2 _f32x2_mul_a_4 = make_float2(_vec_load_0[2], _vec_load_0[3]);
            float2 _f32x2_mul_b_5 = make_float2(_vec_load_16_0[2], _vec_load_16_0[3]);
            float2 q_pair_35_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
            q_cache[2] = q_pair_35_1.x;
            q_cache[3] = q_pair_35_1.y;
            float2 _f32x2_mul_a_6 = make_float2(_vec_load_0[4], _vec_load_0[5]);
            float2 _f32x2_mul_b_7 = make_float2(_vec_load_16_0[4], _vec_load_16_0[5]);
            float2 q_pair_35_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
            q_cache[4] = q_pair_35_2.x;
            q_cache[5] = q_pair_35_2.y;
            float2 _f32x2_mul_a_8 = make_float2(_vec_load_0[6], _vec_load_0[7]);
            float2 _f32x2_mul_b_9 = make_float2(_vec_load_16_0[6], _vec_load_16_0[7]);
            float2 q_pair_35_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
            q_cache[6] = q_pair_35_3.x;
            q_cache[7] = q_pair_35_3.y;
            int hidden_base_1_1 = 2048 + thread_in_group * 8;
            float _vec_load_1[8];
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1_1);
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
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_10[_pair]));
                    }
                }
            }
            float _vec_load_16_1[8];
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1_1);
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
                            : "=f"((&_vec_load_16_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_11[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_12 = make_float2(_vec_load_1[0], _vec_load_1[1]);
            float2 _f32x2_mul_b_13 = make_float2(_vec_load_16_1[0], _vec_load_16_1[1]);
            float2 q_pair_35_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
            q_cache[8] = q_pair_35_4.x;
            q_cache[9] = q_pair_35_4.y;
            float2 _f32x2_mul_a_14 = make_float2(_vec_load_1[2], _vec_load_1[3]);
            float2 _f32x2_mul_b_15 = make_float2(_vec_load_16_1[2], _vec_load_16_1[3]);
            float2 q_pair_35_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
            q_cache[10] = q_pair_35_5.x;
            q_cache[11] = q_pair_35_5.y;
            float2 _f32x2_mul_a_16 = make_float2(_vec_load_1[4], _vec_load_1[5]);
            float2 _f32x2_mul_b_17 = make_float2(_vec_load_16_1[4], _vec_load_16_1[5]);
            float2 q_pair_35_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
            q_cache[12] = q_pair_35_6.x;
            q_cache[13] = q_pair_35_6.y;
            float2 _f32x2_mul_a_18 = make_float2(_vec_load_1[6], _vec_load_1[7]);
            float2 _f32x2_mul_b_19 = make_float2(_vec_load_16_1[6], _vec_load_16_1[7]);
            float2 q_pair_35_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
            q_cache[14] = q_pair_35_7.x;
            q_cache[15] = q_pair_35_7.y;
            int hidden_base_1_2 = 4096 + thread_in_group * 8;
            float _vec_load_2[8];
            {
                const uint4* _vptr_20 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1_2);
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
                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_20[_pair]));
                    }
                }
            }
            float _vec_load_16_2[8];
            {
                const uint4* _vptr_21 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1_2);
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
                            : "=f"((&_vec_load_16_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_21[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_22 = make_float2(_vec_load_2[0], _vec_load_2[1]);
            float2 _f32x2_mul_b_23 = make_float2(_vec_load_16_2[0], _vec_load_16_2[1]);
            float2 q_pair_35_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
            q_cache[16] = q_pair_35_8.x;
            q_cache[17] = q_pair_35_8.y;
            float2 _f32x2_mul_a_24 = make_float2(_vec_load_2[2], _vec_load_2[3]);
            float2 _f32x2_mul_b_25 = make_float2(_vec_load_16_2[2], _vec_load_16_2[3]);
            float2 q_pair_35_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
            q_cache[18] = q_pair_35_9.x;
            q_cache[19] = q_pair_35_9.y;
            float2 _f32x2_mul_a_26 = make_float2(_vec_load_2[4], _vec_load_2[5]);
            float2 _f32x2_mul_b_27 = make_float2(_vec_load_16_2[4], _vec_load_16_2[5]);
            float2 q_pair_35_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
            q_cache[20] = q_pair_35_10.x;
            q_cache[21] = q_pair_35_10.y;
            float2 _f32x2_mul_a_28 = make_float2(_vec_load_2[6], _vec_load_2[7]);
            float2 _f32x2_mul_b_29 = make_float2(_vec_load_16_2[6], _vec_load_16_2[7]);
            float2 q_pair_35_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
            q_cache[22] = q_pair_35_11.x;
            q_cache[23] = q_pair_35_11.y;
            int hidden_base_1_3 = 6144 + thread_in_group * 4;
            float _vec_load_3[4];
            {
                uint2 _vld_30;
                _vld_30 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base_1_3);
                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_3[0 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _pair * 2])[1])
                        : "r"(_vpairs_30[_pair]));
                }
            }
            float _vec_load_16_3[4];
            {
                uint2 _vld_31;
                _vld_31 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base_1_3);
                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_16_3[0 + _pair * 2])[0]), "=f"((&_vec_load_16_3[0 + _pair * 2])[1])
                        : "r"(_vpairs_31[_pair]));
                }
            }
            float2 _f32x2_mul_a_32 = make_float2(_vec_load_3[0], _vec_load_3[1]);
            float2 _f32x2_mul_b_33 = make_float2(_vec_load_16_3[0], _vec_load_16_3[1]);
            float2 q_pair_35_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
            q_cache[24] = q_pair_35_12.x;
            q_cache[25] = q_pair_35_12.y;
            float2 _f32x2_mul_a_34 = make_float2(_vec_load_3[2], _vec_load_3[3]);
            float2 _f32x2_mul_b_35 = make_float2(_vec_load_16_3[2], _vec_load_16_3[3]);
            float2 q_pair_35_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
            q_cache[26] = q_pair_35_13.x;
            q_cache[27] = q_pair_35_13.y;
            int hidden_base_1_4 = 1024 + thread_in_group * 8;
            float _vec_load_4[8];
            {
                const uint4* _vptr_36 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1_4);
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
                            : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_36[_pair]));
                    }
                }
            }
            float _vec_load_16_4[8];
            {
                const uint4* _vptr_37 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1_4);
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
                            : "=f"((&_vec_load_16_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16_4[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_37[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_38 = make_float2(_vec_load_4[0], _vec_load_4[1]);
            float2 _f32x2_mul_b_39 = make_float2(_vec_load_16_4[0], _vec_load_16_4[1]);
            float2 q_pair_35_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
            q_cache[28] = q_pair_35_14.x;
            q_cache[29] = q_pair_35_14.y;
            float2 _f32x2_mul_a_40 = make_float2(_vec_load_4[2], _vec_load_4[3]);
            float2 _f32x2_mul_b_41 = make_float2(_vec_load_16_4[2], _vec_load_16_4[3]);
            float2 q_pair_35_15;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
            q_cache[30] = q_pair_35_15.x;
            q_cache[31] = q_pair_35_15.y;
            float2 _f32x2_mul_a_42 = make_float2(_vec_load_4[4], _vec_load_4[5]);
            float2 _f32x2_mul_b_43 = make_float2(_vec_load_16_4[4], _vec_load_16_4[5]);
            float2 q_pair_35_16;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
            q_cache[32] = q_pair_35_16.x;
            q_cache[33] = q_pair_35_16.y;
            float2 _f32x2_mul_a_44 = make_float2(_vec_load_4[6], _vec_load_4[7]);
            float2 _f32x2_mul_b_45 = make_float2(_vec_load_16_4[6], _vec_load_16_4[7]);
            float2 q_pair_35_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
            q_cache[34] = q_pair_35_17.x;
            q_cache[35] = q_pair_35_17.y;
            int hidden_base_1_5 = 3072 + thread_in_group * 8;
            float _vec_load_5[8];
            {
                const uint4* _vptr_46 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1_5);
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
                            : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_46[_pair]));
                    }
                }
            }
            float _vec_load_16_5[8];
            {
                const uint4* _vptr_47 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1_5);
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
                            : "=f"((&_vec_load_16_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16_5[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_47[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_48 = make_float2(_vec_load_5[0], _vec_load_5[1]);
            float2 _f32x2_mul_b_49 = make_float2(_vec_load_16_5[0], _vec_load_16_5[1]);
            float2 q_pair_35_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
            q_cache[36] = q_pair_35_18.x;
            q_cache[37] = q_pair_35_18.y;
            float2 _f32x2_mul_a_50 = make_float2(_vec_load_5[2], _vec_load_5[3]);
            float2 _f32x2_mul_b_51 = make_float2(_vec_load_16_5[2], _vec_load_16_5[3]);
            float2 q_pair_35_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
            q_cache[38] = q_pair_35_19.x;
            q_cache[39] = q_pair_35_19.y;
            float2 _f32x2_mul_a_52 = make_float2(_vec_load_5[4], _vec_load_5[5]);
            float2 _f32x2_mul_b_53 = make_float2(_vec_load_16_5[4], _vec_load_16_5[5]);
            float2 q_pair_35_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
            q_cache[40] = q_pair_35_20.x;
            q_cache[41] = q_pair_35_20.y;
            float2 _f32x2_mul_a_54 = make_float2(_vec_load_5[6], _vec_load_5[7]);
            float2 _f32x2_mul_b_55 = make_float2(_vec_load_16_5[6], _vec_load_16_5[7]);
            float2 q_pair_35_21;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
            q_cache[42] = q_pair_35_21.x;
            q_cache[43] = q_pair_35_21.y;
            int hidden_base_1_6 = 5120 + thread_in_group * 8;
            float _vec_load_6[8];
            {
                const uint4* _vptr_56 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_1_6);
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
                            : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_56[_pair]));
                    }
                }
            }
            float _vec_load_16_6[8];
            {
                const uint4* _vptr_57 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_1_6);
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
                            : "=f"((&_vec_load_16_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16_6[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_57[_pair]));
                    }
                }
            }
            float2 _f32x2_mul_a_58 = make_float2(_vec_load_6[0], _vec_load_6[1]);
            float2 _f32x2_mul_b_59 = make_float2(_vec_load_16_6[0], _vec_load_16_6[1]);
            float2 q_pair_35_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
            q_cache[44] = q_pair_35_22.x;
            q_cache[45] = q_pair_35_22.y;
            float2 _f32x2_mul_a_60 = make_float2(_vec_load_6[2], _vec_load_6[3]);
            float2 _f32x2_mul_b_61 = make_float2(_vec_load_16_6[2], _vec_load_16_6[3]);
            float2 q_pair_35_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
            q_cache[46] = q_pair_35_23.x;
            q_cache[47] = q_pair_35_23.y;
            float2 _f32x2_mul_a_62 = make_float2(_vec_load_6[4], _vec_load_6[5]);
            float2 _f32x2_mul_b_63 = make_float2(_vec_load_16_6[4], _vec_load_16_6[5]);
            float2 q_pair_35_24;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
            q_cache[48] = q_pair_35_24.x;
            q_cache[49] = q_pair_35_24.y;
            float2 _f32x2_mul_a_64 = make_float2(_vec_load_6[6], _vec_load_6[7]);
            float2 _f32x2_mul_b_65 = make_float2(_vec_load_16_6[6], _vec_load_16_6[7]);
            float2 q_pair_35_25;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
            q_cache[50] = q_pair_35_25.x;
            q_cache[51] = q_pair_35_25.y;
            int hidden_base_1_7 = 6656 + thread_in_group * 4;
            float _vec_load_7[4];
            {
                uint2 _vld_66;
                _vld_66 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base_1_7);
                uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_7[0 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _pair * 2])[1])
                        : "r"(_vpairs_66[_pair]));
                }
            }
            float _vec_load_16_7[4];
            {
                uint2 _vld_67;
                _vld_67 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base_1_7);
                uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_16_7[0 + _pair * 2])[0]), "=f"((&_vec_load_16_7[0 + _pair * 2])[1])
                        : "r"(_vpairs_67[_pair]));
                }
            }
            float2 _f32x2_mul_a_68 = make_float2(_vec_load_7[0], _vec_load_7[1]);
            float2 _f32x2_mul_b_69 = make_float2(_vec_load_16_7[0], _vec_load_16_7[1]);
            float2 q_pair_35_26;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
            q_cache[52] = q_pair_35_26.x;
            q_cache[53] = q_pair_35_26.y;
            float2 _f32x2_mul_a_70 = make_float2(_vec_load_7[2], _vec_load_7[3]);
            float2 _f32x2_mul_b_71 = make_float2(_vec_load_16_7[2], _vec_load_16_7[3]);
            float2 q_pair_35_27;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_35_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
            q_cache[54] = q_pair_35_27.x;
            q_cache[55] = q_pair_35_27.y;
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
                int hidden_base_1_8 = thread_in_group * 8;
                unsigned int smem_regs[4];
                #pragma unroll
                for (int _lr = 0; _lr < 4; _lr++) {
                    smem_regs[_lr] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_8 / 2) + (unsigned int)_lr];
                }
                float values_1_0[8];
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
                values_1_0[0] = _unpack_bf16_0[0];
                values_1_0[1] = _unpack_bf16_0[1];
                values_1_0[2] = _unpack_bf16_0[2];
                values_1_0[3] = _unpack_bf16_0[3];
                values_1_0[4] = _unpack_bf16_0[4];
                values_1_0[5] = _unpack_bf16_0[5];
                values_1_0[6] = _unpack_bf16_0[6];
                values_1_0[7] = _unpack_bf16_0[7];
                tmem_st_x8_f32(tmem_base, values_1_0);
                float2 value_pair_1_0 = make_float2(values_1_0[0], values_1_0[1]);
                float2 q_pair_35_28 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_0, value_pair_1_0, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_0, q_pair_35_28, local_dot[0]);
                float2 value_pair_1_1 = make_float2(values_1_0[2], values_1_0[3]);
                float2 q_pair_35_29 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_1, value_pair_1_1, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_1, q_pair_35_29, local_dot[0]);
                float2 value_pair_1_2 = make_float2(values_1_0[4], values_1_0[5]);
                float2 q_pair_35_30 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_2, value_pair_1_2, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_2, q_pair_35_30, local_dot[0]);
                float2 value_pair_1_3 = make_float2(values_1_0[6], values_1_0[7]);
                float2 q_pair_35_31 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_3, value_pair_1_3, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_3, q_pair_35_31, local_dot[0]);
                unsigned int smem_regs_0[4];
                #pragma unroll
                for (int _lr_1 = 0; _lr_1 < 4; _lr_1++) {
                    smem_regs_0[_lr_1] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_8 / 2) + (unsigned int)_lr_1];
                }
                float values_1_1[8];
                float _unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
                        : "r"(smem_regs_0[_pair]));
                }
                values_1_1[0] = _unpack_bf16_1[0];
                values_1_1[1] = _unpack_bf16_1[1];
                values_1_1[2] = _unpack_bf16_1[2];
                values_1_1[3] = _unpack_bf16_1[3];
                values_1_1[4] = _unpack_bf16_1[4];
                values_1_1[5] = _unpack_bf16_1[5];
                values_1_1[6] = _unpack_bf16_1[6];
                values_1_1[7] = _unpack_bf16_1[7];
                tmem_st_x8_f32(tmem_base + 64, values_1_1);
                float2 value_pair_1_4 = make_float2(values_1_1[0], values_1_1[1]);
                float2 q_pair_35_32 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_4, value_pair_1_4, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_4, q_pair_35_32, local_dot[1]);
                float2 value_pair_1_5 = make_float2(values_1_1[2], values_1_1[3]);
                float2 q_pair_35_33 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_5, value_pair_1_5, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_5, q_pair_35_33, local_dot[1]);
                float2 value_pair_1_6 = make_float2(values_1_1[4], values_1_1[5]);
                float2 q_pair_35_34 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_6, value_pair_1_6, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_6, q_pair_35_34, local_dot[1]);
                float2 value_pair_1_7 = make_float2(values_1_1[6], values_1_1[7]);
                float2 q_pair_35_35 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_7, value_pair_1_7, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_7, q_pair_35_35, local_dot[1]);
                int hidden_base_1_9 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_13[4];
                #pragma unroll
                for (int _lr_2 = 0; _lr_2 < 4; _lr_2++) {
                    smem_regs_13[_lr_2] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_9 / 2) + (unsigned int)_lr_2];
                }
                float values_1_2[8];
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                        : "r"(smem_regs_13[_pair]));
                }
                values_1_2[0] = _unpack_bf16_2[0];
                values_1_2[1] = _unpack_bf16_2[1];
                values_1_2[2] = _unpack_bf16_2[2];
                values_1_2[3] = _unpack_bf16_2[3];
                values_1_2[4] = _unpack_bf16_2[4];
                values_1_2[5] = _unpack_bf16_2[5];
                values_1_2[6] = _unpack_bf16_2[6];
                values_1_2[7] = _unpack_bf16_2[7];
                tmem_st_x8_f32(tmem_base + 8, values_1_2);
                float2 value_pair_1_8 = make_float2(values_1_2[0], values_1_2[1]);
                float2 q_pair_35_36 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_8, value_pair_1_8, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_8, q_pair_35_36, local_dot[0]);
                float2 value_pair_1_9 = make_float2(values_1_2[2], values_1_2[3]);
                float2 q_pair_35_37 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_9, value_pair_1_9, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_9, q_pair_35_37, local_dot[0]);
                float2 value_pair_1_10 = make_float2(values_1_2[4], values_1_2[5]);
                float2 q_pair_35_38 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_10, value_pair_1_10, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_10, q_pair_35_38, local_dot[0]);
                float2 value_pair_1_11 = make_float2(values_1_2[6], values_1_2[7]);
                float2 q_pair_35_39 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_11, value_pair_1_11, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_11, q_pair_35_39, local_dot[0]);
                unsigned int smem_regs_14[4];
                #pragma unroll
                for (int _lr_3 = 0; _lr_3 < 4; _lr_3++) {
                    smem_regs_14[_lr_3] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_9 / 2) + (unsigned int)_lr_3];
                }
                float values_1_3[8];
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                        : "r"(smem_regs_14[_pair]));
                }
                values_1_3[0] = _unpack_bf16_3[0];
                values_1_3[1] = _unpack_bf16_3[1];
                values_1_3[2] = _unpack_bf16_3[2];
                values_1_3[3] = _unpack_bf16_3[3];
                values_1_3[4] = _unpack_bf16_3[4];
                values_1_3[5] = _unpack_bf16_3[5];
                values_1_3[6] = _unpack_bf16_3[6];
                values_1_3[7] = _unpack_bf16_3[7];
                tmem_st_x8_f32(tmem_base + 64 + 8, values_1_3);
                float2 value_pair_1_12 = make_float2(values_1_3[0], values_1_3[1]);
                float2 q_pair_35_40 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_12, value_pair_1_12, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_12, q_pair_35_40, local_dot[1]);
                float2 value_pair_1_13 = make_float2(values_1_3[2], values_1_3[3]);
                float2 q_pair_35_41 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_13, value_pair_1_13, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_13, q_pair_35_41, local_dot[1]);
                float2 value_pair_1_14 = make_float2(values_1_3[4], values_1_3[5]);
                float2 q_pair_35_42 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_14, value_pair_1_14, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_14, q_pair_35_42, local_dot[1]);
                float2 value_pair_1_15 = make_float2(values_1_3[6], values_1_3[7]);
                float2 q_pair_35_43 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_15, value_pair_1_15, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_15, q_pair_35_43, local_dot[1]);
                int hidden_base_1_10 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_15[4];
                #pragma unroll
                for (int _lr_4 = 0; _lr_4 < 4; _lr_4++) {
                    smem_regs_15[_lr_4] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_10 / 2) + (unsigned int)_lr_4];
                }
                float values_1_4[8];
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                        : "r"(smem_regs_15[_pair]));
                }
                values_1_4[0] = _unpack_bf16_4[0];
                values_1_4[1] = _unpack_bf16_4[1];
                values_1_4[2] = _unpack_bf16_4[2];
                values_1_4[3] = _unpack_bf16_4[3];
                values_1_4[4] = _unpack_bf16_4[4];
                values_1_4[5] = _unpack_bf16_4[5];
                values_1_4[6] = _unpack_bf16_4[6];
                values_1_4[7] = _unpack_bf16_4[7];
                tmem_st_x8_f32(tmem_base + 16, values_1_4);
                float2 value_pair_1_16 = make_float2(values_1_4[0], values_1_4[1]);
                float2 q_pair_35_44 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_16, value_pair_1_16, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_16, q_pair_35_44, local_dot[0]);
                float2 value_pair_1_17 = make_float2(values_1_4[2], values_1_4[3]);
                float2 q_pair_35_45 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_17, value_pair_1_17, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_17, q_pair_35_45, local_dot[0]);
                float2 value_pair_1_18 = make_float2(values_1_4[4], values_1_4[5]);
                float2 q_pair_35_46 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_18, value_pair_1_18, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_18, q_pair_35_46, local_dot[0]);
                float2 value_pair_1_19 = make_float2(values_1_4[6], values_1_4[7]);
                float2 q_pair_35_47 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_19, value_pair_1_19, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_19, q_pair_35_47, local_dot[0]);
                unsigned int smem_regs_16[4];
                #pragma unroll
                for (int _lr_5 = 0; _lr_5 < 4; _lr_5++) {
                    smem_regs_16[_lr_5] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_10 / 2) + (unsigned int)_lr_5];
                }
                float values_1_5[8];
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                        : "r"(smem_regs_16[_pair]));
                }
                values_1_5[0] = _unpack_bf16_5[0];
                values_1_5[1] = _unpack_bf16_5[1];
                values_1_5[2] = _unpack_bf16_5[2];
                values_1_5[3] = _unpack_bf16_5[3];
                values_1_5[4] = _unpack_bf16_5[4];
                values_1_5[5] = _unpack_bf16_5[5];
                values_1_5[6] = _unpack_bf16_5[6];
                values_1_5[7] = _unpack_bf16_5[7];
                tmem_st_x8_f32(tmem_base + 64 + 16, values_1_5);
                float2 value_pair_1_20 = make_float2(values_1_5[0], values_1_5[1]);
                float2 q_pair_35_48 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_20, value_pair_1_20, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_20, q_pair_35_48, local_dot[1]);
                float2 value_pair_1_21 = make_float2(values_1_5[2], values_1_5[3]);
                float2 q_pair_35_49 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_21, value_pair_1_21, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_21, q_pair_35_49, local_dot[1]);
                float2 value_pair_1_22 = make_float2(values_1_5[4], values_1_5[5]);
                float2 q_pair_35_50 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_22, value_pair_1_22, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_22, q_pair_35_50, local_dot[1]);
                float2 value_pair_1_23 = make_float2(values_1_5[6], values_1_5[7]);
                float2 q_pair_35_51 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_23, value_pair_1_23, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_23, q_pair_35_51, local_dot[1]);
                int hidden_base_1_11 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_17[2];
                #pragma unroll
                for (int _lr_6 = 0; _lr_6 < 2; _lr_6++) {
                    smem_regs_17[_lr_6] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_11 / 2) + (unsigned int)_lr_6];
                }
                float values_1_6[4];
                float _unpack_bf16_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                        : "r"(smem_regs_17[_pair]));
                }
                values_1_6[0] = _unpack_bf16_6[0];
                values_1_6[1] = _unpack_bf16_6[1];
                values_1_6[2] = _unpack_bf16_6[2];
                values_1_6[3] = _unpack_bf16_6[3];
                tmem_st_x4_f32(tmem_base + 24, values_1_6);
                float2 value_pair_1_24 = make_float2(values_1_6[0], values_1_6[1]);
                float2 q_pair_35_52 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_24, value_pair_1_24, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_24, q_pair_35_52, local_dot[0]);
                float2 value_pair_1_25 = make_float2(values_1_6[2], values_1_6[3]);
                float2 q_pair_35_53 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_1_25, value_pair_1_25, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_1_25, q_pair_35_53, local_dot[0]);
                unsigned int smem_regs_18[2];
                #pragma unroll
                for (int _lr_7 = 0; _lr_7 < 2; _lr_7++) {
                    smem_regs_18[_lr_7] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_11 / 2) + (unsigned int)_lr_7];
                }
                float values_1_7[4];
                float _unpack_bf16_7[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                        : "r"(smem_regs_18[_pair]));
                }
                values_1_7[0] = _unpack_bf16_7[0];
                values_1_7[1] = _unpack_bf16_7[1];
                values_1_7[2] = _unpack_bf16_7[2];
                values_1_7[3] = _unpack_bf16_7[3];
                tmem_st_x4_f32(tmem_base + 64 + 24, values_1_7);
                float2 value_pair_1_26 = make_float2(values_1_7[0], values_1_7[1]);
                float2 q_pair_35_54 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_26, value_pair_1_26, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_26, q_pair_35_54, local_dot[1]);
                float2 value_pair_1_27 = make_float2(values_1_7[2], values_1_7[3]);
                float2 q_pair_35_55 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_1_27, value_pair_1_27, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_1_27, q_pair_35_55, local_dot[1]);
                int hidden_base_1_12 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_19[4];
                #pragma unroll
                for (int _lr_8 = 0; _lr_8 < 4; _lr_8++) {
                    smem_regs_19[_lr_8] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_12 / 2) + (unsigned int)_lr_8];
                }
                float values_1_8[8];
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                        : "r"(smem_regs_19[_pair]));
                }
                values_1_8[0] = _unpack_bf16_8[0];
                values_1_8[1] = _unpack_bf16_8[1];
                values_1_8[2] = _unpack_bf16_8[2];
                values_1_8[3] = _unpack_bf16_8[3];
                values_1_8[4] = _unpack_bf16_8[4];
                values_1_8[5] = _unpack_bf16_8[5];
                values_1_8[6] = _unpack_bf16_8[6];
                values_1_8[7] = _unpack_bf16_8[7];
                tmem_st_x8_f32(tmem_base + 32, values_1_8);
                float2 value_pair_1_28 = make_float2(values_1_8[0], values_1_8[1]);
                float2 q_pair_35_56 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_28, value_pair_1_28, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_28, q_pair_35_56, local_dot[2]);
                float2 value_pair_1_29 = make_float2(values_1_8[2], values_1_8[3]);
                float2 q_pair_35_57 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_29, value_pair_1_29, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_29, q_pair_35_57, local_dot[2]);
                float2 value_pair_1_30 = make_float2(values_1_8[4], values_1_8[5]);
                float2 q_pair_35_58 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_30, value_pair_1_30, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_30, q_pair_35_58, local_dot[2]);
                float2 value_pair_1_31 = make_float2(values_1_8[6], values_1_8[7]);
                float2 q_pair_35_59 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_31, value_pair_1_31, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_31, q_pair_35_59, local_dot[2]);
                unsigned int smem_regs_20[4];
                #pragma unroll
                for (int _lr_9 = 0; _lr_9 < 4; _lr_9++) {
                    smem_regs_20[_lr_9] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_12 / 2) + (unsigned int)_lr_9];
                }
                float values_1_9[8];
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                        : "r"(smem_regs_20[_pair]));
                }
                values_1_9[0] = _unpack_bf16_9[0];
                values_1_9[1] = _unpack_bf16_9[1];
                values_1_9[2] = _unpack_bf16_9[2];
                values_1_9[3] = _unpack_bf16_9[3];
                values_1_9[4] = _unpack_bf16_9[4];
                values_1_9[5] = _unpack_bf16_9[5];
                values_1_9[6] = _unpack_bf16_9[6];
                values_1_9[7] = _unpack_bf16_9[7];
                tmem_st_x8_f32(tmem_base + 64 + 32, values_1_9);
                float2 value_pair_1_32 = make_float2(values_1_9[0], values_1_9[1]);
                float2 q_pair_35_60 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_32, value_pair_1_32, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_32, q_pair_35_60, local_dot[3]);
                float2 value_pair_1_33 = make_float2(values_1_9[2], values_1_9[3]);
                float2 q_pair_35_61 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_33, value_pair_1_33, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_33, q_pair_35_61, local_dot[3]);
                float2 value_pair_1_34 = make_float2(values_1_9[4], values_1_9[5]);
                float2 q_pair_35_62 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_34, value_pair_1_34, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_34, q_pair_35_62, local_dot[3]);
                float2 value_pair_1_35 = make_float2(values_1_9[6], values_1_9[7]);
                float2 q_pair_35_63 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_35, value_pair_1_35, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_35, q_pair_35_63, local_dot[3]);
                int hidden_base_1_13 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_21[4];
                #pragma unroll
                for (int _lr_10 = 0; _lr_10 < 4; _lr_10++) {
                    smem_regs_21[_lr_10] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_13 / 2) + (unsigned int)_lr_10];
                }
                float values_1_10[8];
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                        : "r"(smem_regs_21[_pair]));
                }
                values_1_10[0] = _unpack_bf16_10[0];
                values_1_10[1] = _unpack_bf16_10[1];
                values_1_10[2] = _unpack_bf16_10[2];
                values_1_10[3] = _unpack_bf16_10[3];
                values_1_10[4] = _unpack_bf16_10[4];
                values_1_10[5] = _unpack_bf16_10[5];
                values_1_10[6] = _unpack_bf16_10[6];
                values_1_10[7] = _unpack_bf16_10[7];
                tmem_st_x8_f32(tmem_base + 40, values_1_10);
                float2 value_pair_1_36 = make_float2(values_1_10[0], values_1_10[1]);
                float2 q_pair_35_64 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_36, value_pair_1_36, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_36, q_pair_35_64, local_dot[2]);
                float2 value_pair_1_37 = make_float2(values_1_10[2], values_1_10[3]);
                float2 q_pair_35_65 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_37, value_pair_1_37, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_37, q_pair_35_65, local_dot[2]);
                float2 value_pair_1_38 = make_float2(values_1_10[4], values_1_10[5]);
                float2 q_pair_35_66 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_38, value_pair_1_38, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_38, q_pair_35_66, local_dot[2]);
                float2 value_pair_1_39 = make_float2(values_1_10[6], values_1_10[7]);
                float2 q_pair_35_67 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_39, value_pair_1_39, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_39, q_pair_35_67, local_dot[2]);
                unsigned int smem_regs_22[4];
                #pragma unroll
                for (int _lr_11 = 0; _lr_11 < 4; _lr_11++) {
                    smem_regs_22[_lr_11] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_13 / 2) + (unsigned int)_lr_11];
                }
                float values_1_11[8];
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                        : "r"(smem_regs_22[_pair]));
                }
                values_1_11[0] = _unpack_bf16_11[0];
                values_1_11[1] = _unpack_bf16_11[1];
                values_1_11[2] = _unpack_bf16_11[2];
                values_1_11[3] = _unpack_bf16_11[3];
                values_1_11[4] = _unpack_bf16_11[4];
                values_1_11[5] = _unpack_bf16_11[5];
                values_1_11[6] = _unpack_bf16_11[6];
                values_1_11[7] = _unpack_bf16_11[7];
                tmem_st_x8_f32(tmem_base + 64 + 40, values_1_11);
                float2 value_pair_1_40 = make_float2(values_1_11[0], values_1_11[1]);
                float2 q_pair_35_68 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_40, value_pair_1_40, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_40, q_pair_35_68, local_dot[3]);
                float2 value_pair_1_41 = make_float2(values_1_11[2], values_1_11[3]);
                float2 q_pair_35_69 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_41, value_pair_1_41, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_41, q_pair_35_69, local_dot[3]);
                float2 value_pair_1_42 = make_float2(values_1_11[4], values_1_11[5]);
                float2 q_pair_35_70 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_42, value_pair_1_42, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_42, q_pair_35_70, local_dot[3]);
                float2 value_pair_1_43 = make_float2(values_1_11[6], values_1_11[7]);
                float2 q_pair_35_71 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_43, value_pair_1_43, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_43, q_pair_35_71, local_dot[3]);
                int hidden_base_1_14 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_23[4];
                #pragma unroll
                for (int _lr_12 = 0; _lr_12 < 4; _lr_12++) {
                    smem_regs_23[_lr_12] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_14 / 2) + (unsigned int)_lr_12];
                }
                float values_1_12[8];
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                        : "r"(smem_regs_23[_pair]));
                }
                values_1_12[0] = _unpack_bf16_12[0];
                values_1_12[1] = _unpack_bf16_12[1];
                values_1_12[2] = _unpack_bf16_12[2];
                values_1_12[3] = _unpack_bf16_12[3];
                values_1_12[4] = _unpack_bf16_12[4];
                values_1_12[5] = _unpack_bf16_12[5];
                values_1_12[6] = _unpack_bf16_12[6];
                values_1_12[7] = _unpack_bf16_12[7];
                tmem_st_x8_f32(tmem_base + 48, values_1_12);
                float2 value_pair_1_44 = make_float2(values_1_12[0], values_1_12[1]);
                float2 q_pair_35_72 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_44, value_pair_1_44, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_44, q_pair_35_72, local_dot[2]);
                float2 value_pair_1_45 = make_float2(values_1_12[2], values_1_12[3]);
                float2 q_pair_35_73 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_45, value_pair_1_45, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_45, q_pair_35_73, local_dot[2]);
                float2 value_pair_1_46 = make_float2(values_1_12[4], values_1_12[5]);
                float2 q_pair_35_74 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_46, value_pair_1_46, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_46, q_pair_35_74, local_dot[2]);
                float2 value_pair_1_47 = make_float2(values_1_12[6], values_1_12[7]);
                float2 q_pair_35_75 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_47, value_pair_1_47, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_47, q_pair_35_75, local_dot[2]);
                unsigned int smem_regs_24[4];
                #pragma unroll
                for (int _lr_13 = 0; _lr_13 < 4; _lr_13++) {
                    smem_regs_24[_lr_13] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_14 / 2) + (unsigned int)_lr_13];
                }
                float values_1_13[8];
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                        : "r"(smem_regs_24[_pair]));
                }
                values_1_13[0] = _unpack_bf16_13[0];
                values_1_13[1] = _unpack_bf16_13[1];
                values_1_13[2] = _unpack_bf16_13[2];
                values_1_13[3] = _unpack_bf16_13[3];
                values_1_13[4] = _unpack_bf16_13[4];
                values_1_13[5] = _unpack_bf16_13[5];
                values_1_13[6] = _unpack_bf16_13[6];
                values_1_13[7] = _unpack_bf16_13[7];
                tmem_st_x8_f32(tmem_base + 64 + 48, values_1_13);
                float2 value_pair_1_48 = make_float2(values_1_13[0], values_1_13[1]);
                float2 q_pair_35_76 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_48, value_pair_1_48, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_48, q_pair_35_76, local_dot[3]);
                float2 value_pair_1_49 = make_float2(values_1_13[2], values_1_13[3]);
                float2 q_pair_35_77 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_49, value_pair_1_49, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_49, q_pair_35_77, local_dot[3]);
                float2 value_pair_1_50 = make_float2(values_1_13[4], values_1_13[5]);
                float2 q_pair_35_78 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_50, value_pair_1_50, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_50, q_pair_35_78, local_dot[3]);
                float2 value_pair_1_51 = make_float2(values_1_13[6], values_1_13[7]);
                float2 q_pair_35_79 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_51, value_pair_1_51, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_51, q_pair_35_79, local_dot[3]);
                int hidden_base_1_15 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_25[2];
                #pragma unroll
                for (int _lr_14 = 0; _lr_14 < 2; _lr_14++) {
                    smem_regs_25[_lr_14] = source_words[ring_stage * 2 * 3584 + (unsigned int)(hidden_base_1_15 / 2) + (unsigned int)_lr_14];
                }
                float values_1_14[4];
                float _unpack_bf16_14[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                        : "r"(smem_regs_25[_pair]));
                }
                values_1_14[0] = _unpack_bf16_14[0];
                values_1_14[1] = _unpack_bf16_14[1];
                values_1_14[2] = _unpack_bf16_14[2];
                values_1_14[3] = _unpack_bf16_14[3];
                tmem_st_x4_f32(tmem_base + 56, values_1_14);
                float2 value_pair_1_52 = make_float2(values_1_14[0], values_1_14[1]);
                float2 q_pair_35_80 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_52, value_pair_1_52, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_52, q_pair_35_80, local_dot[2]);
                float2 value_pair_1_53 = make_float2(values_1_14[2], values_1_14[3]);
                float2 q_pair_35_81 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_1_53, value_pair_1_53, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_1_53, q_pair_35_81, local_dot[2]);
                unsigned int smem_regs_26[2];
                #pragma unroll
                for (int _lr_15 = 0; _lr_15 < 2; _lr_15++) {
                    smem_regs_26[_lr_15] = source_words[(ring_stage * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_15 / 2) + (unsigned int)_lr_15];
                }
                float values_1_15[4];
                float _unpack_bf16_15[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                        : "r"(smem_regs_26[_pair]));
                }
                values_1_15[0] = _unpack_bf16_15[0];
                values_1_15[1] = _unpack_bf16_15[1];
                values_1_15[2] = _unpack_bf16_15[2];
                values_1_15[3] = _unpack_bf16_15[3];
                tmem_st_x4_f32(tmem_base + 64 + 56, values_1_15);
                float2 value_pair_1_54 = make_float2(values_1_15[0], values_1_15[1]);
                float2 q_pair_35_82 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_54, value_pair_1_54, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_54, q_pair_35_82, local_dot[3]);
                float2 value_pair_1_55 = make_float2(values_1_15[2], values_1_15[3]);
                float2 q_pair_35_83 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_1_55, value_pair_1_55, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_1_55, q_pair_35_83, local_dot[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base = stats_stream_base;
                float2 reduce_pair_1_0 = make_float2(local_sum_sq[0].x + local_sum_sq[0].y, local_dot[0].x + local_dot[0].y);
                unsigned long long packed_pair_1_0 = 0;
                unsigned long long peer_packed_1_0 = 0;
                float2 peer_pair_1_0 = make_float2(0.0f, 0.0f);
                packed_pair_1_0 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_0)[0];
                unsigned long long _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_0, 16);
                peer_packed_1_0 = _shfl_xor_sync_72;
                peer_pair_1_0 = reinterpret_cast<float2*>(&peer_packed_1_0)[0];
                reduce_pair_1_0 = add_f32x2(reduce_pair_1_0, peer_pair_1_0);
                packed_pair_1_0 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_0)[0];
                unsigned long long _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_0, 8);
                peer_packed_1_0 = _shfl_xor_sync_73;
                peer_pair_1_0 = reinterpret_cast<float2*>(&peer_packed_1_0)[0];
                reduce_pair_1_0 = add_f32x2(reduce_pair_1_0, peer_pair_1_0);
                packed_pair_1_0 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_0)[0];
                unsigned long long _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_0, 4);
                peer_packed_1_0 = _shfl_xor_sync_74;
                peer_pair_1_0 = reinterpret_cast<float2*>(&peer_packed_1_0)[0];
                reduce_pair_1_0 = add_f32x2(reduce_pair_1_0, peer_pair_1_0);
                packed_pair_1_0 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_0)[0];
                unsigned long long _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_0, 2);
                peer_packed_1_0 = _shfl_xor_sync_75;
                peer_pair_1_0 = reinterpret_cast<float2*>(&peer_packed_1_0)[0];
                reduce_pair_1_0 = add_f32x2(reduce_pair_1_0, peer_pair_1_0);
                packed_pair_1_0 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_0)[0];
                unsigned long long _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_0, 1);
                peer_packed_1_0 = _shfl_xor_sync_76;
                peer_pair_1_0 = reinterpret_cast<float2*>(&peer_packed_1_0)[0];
                reduce_pair_1_0 = add_f32x2(reduce_pair_1_0, peer_pair_1_0);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1_0.x;
                    stats[slot_base + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1_0.y;
                }
                float2 reduce_pair_1_1 = make_float2(local_sum_sq[1].x + local_sum_sq[1].y, local_dot[1].x + local_dot[1].y);
                unsigned long long packed_pair_1_1 = 0;
                unsigned long long peer_packed_1_1 = 0;
                float2 peer_pair_1_1 = make_float2(0.0f, 0.0f);
                packed_pair_1_1 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_1)[0];
                unsigned long long _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_1, 16);
                peer_packed_1_1 = _shfl_xor_sync_77;
                peer_pair_1_1 = reinterpret_cast<float2*>(&peer_packed_1_1)[0];
                reduce_pair_1_1 = add_f32x2(reduce_pair_1_1, peer_pair_1_1);
                packed_pair_1_1 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_1)[0];
                unsigned long long _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_1, 8);
                peer_packed_1_1 = _shfl_xor_sync_78;
                peer_pair_1_1 = reinterpret_cast<float2*>(&peer_packed_1_1)[0];
                reduce_pair_1_1 = add_f32x2(reduce_pair_1_1, peer_pair_1_1);
                packed_pair_1_1 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_1)[0];
                unsigned long long _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_1, 4);
                peer_packed_1_1 = _shfl_xor_sync_79;
                peer_pair_1_1 = reinterpret_cast<float2*>(&peer_packed_1_1)[0];
                reduce_pair_1_1 = add_f32x2(reduce_pair_1_1, peer_pair_1_1);
                packed_pair_1_1 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_1)[0];
                unsigned long long _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_1, 2);
                peer_packed_1_1 = _shfl_xor_sync_80;
                peer_pair_1_1 = reinterpret_cast<float2*>(&peer_packed_1_1)[0];
                reduce_pair_1_1 = add_f32x2(reduce_pair_1_1, peer_pair_1_1);
                packed_pair_1_1 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_1)[0];
                unsigned long long _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_1, 1);
                peer_packed_1_1 = _shfl_xor_sync_81;
                peer_pair_1_1 = reinterpret_cast<float2*>(&peer_packed_1_1)[0];
                reduce_pair_1_1 = add_f32x2(reduce_pair_1_1, peer_pair_1_1);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_1_1.x;
                    stats[slot_base + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_1_1.y;
                }
                float2 reduce_pair_1_2 = make_float2(local_sum_sq[2].x + local_sum_sq[2].y, local_dot[2].x + local_dot[2].y);
                unsigned long long packed_pair_1_2 = 0;
                unsigned long long peer_packed_1_2 = 0;
                float2 peer_pair_1_2 = make_float2(0.0f, 0.0f);
                packed_pair_1_2 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_2)[0];
                unsigned long long _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_2, 16);
                peer_packed_1_2 = _shfl_xor_sync_82;
                peer_pair_1_2 = reinterpret_cast<float2*>(&peer_packed_1_2)[0];
                reduce_pair_1_2 = add_f32x2(reduce_pair_1_2, peer_pair_1_2);
                packed_pair_1_2 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_2)[0];
                unsigned long long _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_2, 8);
                peer_packed_1_2 = _shfl_xor_sync_83;
                peer_pair_1_2 = reinterpret_cast<float2*>(&peer_packed_1_2)[0];
                reduce_pair_1_2 = add_f32x2(reduce_pair_1_2, peer_pair_1_2);
                packed_pair_1_2 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_2)[0];
                unsigned long long _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_2, 4);
                peer_packed_1_2 = _shfl_xor_sync_84;
                peer_pair_1_2 = reinterpret_cast<float2*>(&peer_packed_1_2)[0];
                reduce_pair_1_2 = add_f32x2(reduce_pair_1_2, peer_pair_1_2);
                packed_pair_1_2 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_2)[0];
                unsigned long long _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_2, 2);
                peer_packed_1_2 = _shfl_xor_sync_85;
                peer_pair_1_2 = reinterpret_cast<float2*>(&peer_packed_1_2)[0];
                reduce_pair_1_2 = add_f32x2(reduce_pair_1_2, peer_pair_1_2);
                packed_pair_1_2 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_2)[0];
                unsigned long long _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_2, 1);
                peer_packed_1_2 = _shfl_xor_sync_86;
                peer_pair_1_2 = reinterpret_cast<float2*>(&peer_packed_1_2)[0];
                reduce_pair_1_2 = add_f32x2(reduce_pair_1_2, peer_pair_1_2);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1_2.x;
                    stats[slot_base + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1_2.y;
                }
                float2 reduce_pair_1_3 = make_float2(local_sum_sq[3].x + local_sum_sq[3].y, local_dot[3].x + local_dot[3].y);
                unsigned long long packed_pair_1_3 = 0;
                unsigned long long peer_packed_1_3 = 0;
                float2 peer_pair_1_3 = make_float2(0.0f, 0.0f);
                packed_pair_1_3 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_3)[0];
                unsigned long long _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_3, 16);
                peer_packed_1_3 = _shfl_xor_sync_87;
                peer_pair_1_3 = reinterpret_cast<float2*>(&peer_packed_1_3)[0];
                reduce_pair_1_3 = add_f32x2(reduce_pair_1_3, peer_pair_1_3);
                packed_pair_1_3 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_3)[0];
                unsigned long long _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_3, 8);
                peer_packed_1_3 = _shfl_xor_sync_88;
                peer_pair_1_3 = reinterpret_cast<float2*>(&peer_packed_1_3)[0];
                reduce_pair_1_3 = add_f32x2(reduce_pair_1_3, peer_pair_1_3);
                packed_pair_1_3 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_3)[0];
                unsigned long long _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_3, 4);
                peer_packed_1_3 = _shfl_xor_sync_89;
                peer_pair_1_3 = reinterpret_cast<float2*>(&peer_packed_1_3)[0];
                reduce_pair_1_3 = add_f32x2(reduce_pair_1_3, peer_pair_1_3);
                packed_pair_1_3 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_3)[0];
                unsigned long long _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_3, 2);
                peer_packed_1_3 = _shfl_xor_sync_90;
                peer_pair_1_3 = reinterpret_cast<float2*>(&peer_packed_1_3)[0];
                reduce_pair_1_3 = add_f32x2(reduce_pair_1_3, peer_pair_1_3);
                packed_pair_1_3 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_3)[0];
                unsigned long long _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_3, 1);
                peer_packed_1_3 = _shfl_xor_sync_91;
                peer_pair_1_3 = reinterpret_cast<float2*>(&peer_packed_1_3)[0];
                reduce_pair_1_3 = add_f32x2(reduce_pair_1_3, peer_pair_1_3);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2)] = reduce_pair_1_3.x;
                    stats[slot_base + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2) + 1] = reduce_pair_1_3.y;
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
                float2 correction_pair_1_0 = make_float2(correction, correction);
                float2 acc_pair_1_0 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1_0;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_0) : "l"(*(const unsigned long long*)&acc_pair_1_0), "l"(*(const unsigned long long*)&correction_pair_1_0));
                acc32[0] = scaled_pair_1_0.x;
                acc32[1] = scaled_pair_1_0.y;
                float2 acc_pair_1_1 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_1) : "l"(*(const unsigned long long*)&acc_pair_1_1), "l"(*(const unsigned long long*)&correction_pair_1_0));
                acc32[2] = scaled_pair_1_1.x;
                acc32[3] = scaled_pair_1_1.y;
                float2 acc_pair_1_2 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_2) : "l"(*(const unsigned long long*)&acc_pair_1_2), "l"(*(const unsigned long long*)&correction_pair_1_0));
                acc32[4] = scaled_pair_1_2.x;
                acc32[5] = scaled_pair_1_2.y;
                float2 acc_pair_1_3 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_3) : "l"(*(const unsigned long long*)&acc_pair_1_3), "l"(*(const unsigned long long*)&correction_pair_1_0));
                acc32[6] = scaled_pair_1_3.x;
                acc32[7] = scaled_pair_1_3.y;
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], tmem_base);
                float2 weight_pair_1_0 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_0 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                float2 acc_pair_1_4 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1_0 = fma_f32x2_rn_ftz(weight_pair_1_0, cached_pair_1_0, acc_pair_1_4);
                acc32[0] = updated_pair_1_0.x;
                acc32[1] = updated_pair_1_0.y;
                float2 cached_pair_1_1 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
                float2 acc_pair_1_5 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1_1 = fma_f32x2_rn_ftz(weight_pair_1_0, cached_pair_1_1, acc_pair_1_5);
                acc32[2] = updated_pair_1_1.x;
                acc32[3] = updated_pair_1_1.y;
                float2 cached_pair_1_2 = make_float2(_tmem_load_0[4], _tmem_load_0[5]);
                float2 acc_pair_1_6 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1_2 = fma_f32x2_rn_ftz(weight_pair_1_0, cached_pair_1_2, acc_pair_1_6);
                acc32[4] = updated_pair_1_2.x;
                acc32[5] = updated_pair_1_2.y;
                float2 cached_pair_1_3 = make_float2(_tmem_load_0[6], _tmem_load_0[7]);
                float2 acc_pair_1_7 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1_3 = fma_f32x2_rn_ftz(weight_pair_1_0, cached_pair_1_3, acc_pair_1_7);
                acc32[6] = updated_pair_1_3.x;
                acc32[7] = updated_pair_1_3.y;
                float _tmem_load_1[8];
                tmem_ld_x8(&_tmem_load_1[0], tmem_base + 64);
                float2 weight_pair_1_1 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_4 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                float2 acc_pair_1_8 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1_4 = fma_f32x2_rn_ftz(weight_pair_1_1, cached_pair_1_4, acc_pair_1_8);
                acc32[0] = updated_pair_1_4.x;
                acc32[1] = updated_pair_1_4.y;
                float2 cached_pair_1_5 = make_float2(_tmem_load_1[2], _tmem_load_1[3]);
                float2 acc_pair_1_9 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1_5 = fma_f32x2_rn_ftz(weight_pair_1_1, cached_pair_1_5, acc_pair_1_9);
                acc32[2] = updated_pair_1_5.x;
                acc32[3] = updated_pair_1_5.y;
                float2 cached_pair_1_6 = make_float2(_tmem_load_1[4], _tmem_load_1[5]);
                float2 acc_pair_1_10 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1_6 = fma_f32x2_rn_ftz(weight_pair_1_1, cached_pair_1_6, acc_pair_1_10);
                acc32[4] = updated_pair_1_6.x;
                acc32[5] = updated_pair_1_6.y;
                float2 cached_pair_1_7 = make_float2(_tmem_load_1[6], _tmem_load_1[7]);
                float2 acc_pair_1_11 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1_7 = fma_f32x2_rn_ftz(weight_pair_1_1, cached_pair_1_7, acc_pair_1_11);
                acc32[6] = updated_pair_1_7.x;
                acc32[7] = updated_pair_1_7.y;
                float2 correction_pair_1_1 = make_float2(correction, correction);
                float2 acc_pair_1_12 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1_4;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_4) : "l"(*(const unsigned long long*)&acc_pair_1_12), "l"(*(const unsigned long long*)&correction_pair_1_1));
                acc32[8] = scaled_pair_1_4.x;
                acc32[9] = scaled_pair_1_4.y;
                float2 acc_pair_1_13 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1_5;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_5) : "l"(*(const unsigned long long*)&acc_pair_1_13), "l"(*(const unsigned long long*)&correction_pair_1_1));
                acc32[10] = scaled_pair_1_5.x;
                acc32[11] = scaled_pair_1_5.y;
                float2 acc_pair_1_14 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1_6;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_6) : "l"(*(const unsigned long long*)&acc_pair_1_14), "l"(*(const unsigned long long*)&correction_pair_1_1));
                acc32[12] = scaled_pair_1_6.x;
                acc32[13] = scaled_pair_1_6.y;
                float2 acc_pair_1_15 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1_7;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_7) : "l"(*(const unsigned long long*)&acc_pair_1_15), "l"(*(const unsigned long long*)&correction_pair_1_1));
                acc32[14] = scaled_pair_1_7.x;
                acc32[15] = scaled_pair_1_7.y;
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], tmem_base + 8);
                float2 weight_pair_1_2 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_8 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                float2 acc_pair_1_16 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1_8 = fma_f32x2_rn_ftz(weight_pair_1_2, cached_pair_1_8, acc_pair_1_16);
                acc32[8] = updated_pair_1_8.x;
                acc32[9] = updated_pair_1_8.y;
                float2 cached_pair_1_9 = make_float2(_tmem_load_2[2], _tmem_load_2[3]);
                float2 acc_pair_1_17 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1_9 = fma_f32x2_rn_ftz(weight_pair_1_2, cached_pair_1_9, acc_pair_1_17);
                acc32[10] = updated_pair_1_9.x;
                acc32[11] = updated_pair_1_9.y;
                float2 cached_pair_1_10 = make_float2(_tmem_load_2[4], _tmem_load_2[5]);
                float2 acc_pair_1_18 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1_10 = fma_f32x2_rn_ftz(weight_pair_1_2, cached_pair_1_10, acc_pair_1_18);
                acc32[12] = updated_pair_1_10.x;
                acc32[13] = updated_pair_1_10.y;
                float2 cached_pair_1_11 = make_float2(_tmem_load_2[6], _tmem_load_2[7]);
                float2 acc_pair_1_19 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1_11 = fma_f32x2_rn_ftz(weight_pair_1_2, cached_pair_1_11, acc_pair_1_19);
                acc32[14] = updated_pair_1_11.x;
                acc32[15] = updated_pair_1_11.y;
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], tmem_base + 64 + 8);
                float2 weight_pair_1_3 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_12 = make_float2(_tmem_load_3[0], _tmem_load_3[1]);
                float2 acc_pair_1_20 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1_12 = fma_f32x2_rn_ftz(weight_pair_1_3, cached_pair_1_12, acc_pair_1_20);
                acc32[8] = updated_pair_1_12.x;
                acc32[9] = updated_pair_1_12.y;
                float2 cached_pair_1_13 = make_float2(_tmem_load_3[2], _tmem_load_3[3]);
                float2 acc_pair_1_21 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1_13 = fma_f32x2_rn_ftz(weight_pair_1_3, cached_pair_1_13, acc_pair_1_21);
                acc32[10] = updated_pair_1_13.x;
                acc32[11] = updated_pair_1_13.y;
                float2 cached_pair_1_14 = make_float2(_tmem_load_3[4], _tmem_load_3[5]);
                float2 acc_pair_1_22 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1_14 = fma_f32x2_rn_ftz(weight_pair_1_3, cached_pair_1_14, acc_pair_1_22);
                acc32[12] = updated_pair_1_14.x;
                acc32[13] = updated_pair_1_14.y;
                float2 cached_pair_1_15 = make_float2(_tmem_load_3[6], _tmem_load_3[7]);
                float2 acc_pair_1_23 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1_15 = fma_f32x2_rn_ftz(weight_pair_1_3, cached_pair_1_15, acc_pair_1_23);
                acc32[14] = updated_pair_1_15.x;
                acc32[15] = updated_pair_1_15.y;
                float2 correction_pair_1_2 = make_float2(correction, correction);
                float2 acc_pair_1_24 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1_8;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_8) : "l"(*(const unsigned long long*)&acc_pair_1_24), "l"(*(const unsigned long long*)&correction_pair_1_2));
                acc32[16] = scaled_pair_1_8.x;
                acc32[17] = scaled_pair_1_8.y;
                float2 acc_pair_1_25 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1_9;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_9) : "l"(*(const unsigned long long*)&acc_pair_1_25), "l"(*(const unsigned long long*)&correction_pair_1_2));
                acc32[18] = scaled_pair_1_9.x;
                acc32[19] = scaled_pair_1_9.y;
                float2 acc_pair_1_26 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_10) : "l"(*(const unsigned long long*)&acc_pair_1_26), "l"(*(const unsigned long long*)&correction_pair_1_2));
                acc32[20] = scaled_pair_1_10.x;
                acc32[21] = scaled_pair_1_10.y;
                float2 acc_pair_1_27 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1_11;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_11) : "l"(*(const unsigned long long*)&acc_pair_1_27), "l"(*(const unsigned long long*)&correction_pair_1_2));
                acc32[22] = scaled_pair_1_11.x;
                acc32[23] = scaled_pair_1_11.y;
                float _tmem_load_4[8];
                tmem_ld_x8(&_tmem_load_4[0], tmem_base + 16);
                float2 weight_pair_1_4 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_16 = make_float2(_tmem_load_4[0], _tmem_load_4[1]);
                float2 acc_pair_1_28 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1_16 = fma_f32x2_rn_ftz(weight_pair_1_4, cached_pair_1_16, acc_pair_1_28);
                acc32[16] = updated_pair_1_16.x;
                acc32[17] = updated_pair_1_16.y;
                float2 cached_pair_1_17 = make_float2(_tmem_load_4[2], _tmem_load_4[3]);
                float2 acc_pair_1_29 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1_17 = fma_f32x2_rn_ftz(weight_pair_1_4, cached_pair_1_17, acc_pair_1_29);
                acc32[18] = updated_pair_1_17.x;
                acc32[19] = updated_pair_1_17.y;
                float2 cached_pair_1_18 = make_float2(_tmem_load_4[4], _tmem_load_4[5]);
                float2 acc_pair_1_30 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1_18 = fma_f32x2_rn_ftz(weight_pair_1_4, cached_pair_1_18, acc_pair_1_30);
                acc32[20] = updated_pair_1_18.x;
                acc32[21] = updated_pair_1_18.y;
                float2 cached_pair_1_19 = make_float2(_tmem_load_4[6], _tmem_load_4[7]);
                float2 acc_pair_1_31 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1_19 = fma_f32x2_rn_ftz(weight_pair_1_4, cached_pair_1_19, acc_pair_1_31);
                acc32[22] = updated_pair_1_19.x;
                acc32[23] = updated_pair_1_19.y;
                float _tmem_load_5[8];
                tmem_ld_x8(&_tmem_load_5[0], tmem_base + 64 + 16);
                float2 weight_pair_1_5 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_20 = make_float2(_tmem_load_5[0], _tmem_load_5[1]);
                float2 acc_pair_1_32 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1_20 = fma_f32x2_rn_ftz(weight_pair_1_5, cached_pair_1_20, acc_pair_1_32);
                acc32[16] = updated_pair_1_20.x;
                acc32[17] = updated_pair_1_20.y;
                float2 cached_pair_1_21 = make_float2(_tmem_load_5[2], _tmem_load_5[3]);
                float2 acc_pair_1_33 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1_21 = fma_f32x2_rn_ftz(weight_pair_1_5, cached_pair_1_21, acc_pair_1_33);
                acc32[18] = updated_pair_1_21.x;
                acc32[19] = updated_pair_1_21.y;
                float2 cached_pair_1_22 = make_float2(_tmem_load_5[4], _tmem_load_5[5]);
                float2 acc_pair_1_34 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1_22 = fma_f32x2_rn_ftz(weight_pair_1_5, cached_pair_1_22, acc_pair_1_34);
                acc32[20] = updated_pair_1_22.x;
                acc32[21] = updated_pair_1_22.y;
                float2 cached_pair_1_23 = make_float2(_tmem_load_5[6], _tmem_load_5[7]);
                float2 acc_pair_1_35 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1_23 = fma_f32x2_rn_ftz(weight_pair_1_5, cached_pair_1_23, acc_pair_1_35);
                acc32[22] = updated_pair_1_23.x;
                acc32[23] = updated_pair_1_23.y;
                float2 correction_pair_1_3 = make_float2(correction, correction);
                float2 acc_pair_1_36 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1_12;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_12) : "l"(*(const unsigned long long*)&acc_pair_1_36), "l"(*(const unsigned long long*)&correction_pair_1_3));
                acc32[24] = scaled_pair_1_12.x;
                acc32[25] = scaled_pair_1_12.y;
                float2 acc_pair_1_37 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1_13;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_13) : "l"(*(const unsigned long long*)&acc_pair_1_37), "l"(*(const unsigned long long*)&correction_pair_1_3));
                acc32[26] = scaled_pair_1_13.x;
                acc32[27] = scaled_pair_1_13.y;
                float _tmem_load_6[4];
                tmem_ld_x4(&_tmem_load_6[0], tmem_base + 24);
                float2 weight_pair_1_6 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_24 = make_float2(_tmem_load_6[0], _tmem_load_6[1]);
                float2 acc_pair_1_38 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1_24 = fma_f32x2_rn_ftz(weight_pair_1_6, cached_pair_1_24, acc_pair_1_38);
                acc32[24] = updated_pair_1_24.x;
                acc32[25] = updated_pair_1_24.y;
                float2 cached_pair_1_25 = make_float2(_tmem_load_6[2], _tmem_load_6[3]);
                float2 acc_pair_1_39 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1_25 = fma_f32x2_rn_ftz(weight_pair_1_6, cached_pair_1_25, acc_pair_1_39);
                acc32[26] = updated_pair_1_25.x;
                acc32[27] = updated_pair_1_25.y;
                float _tmem_load_7[4];
                tmem_ld_x4(&_tmem_load_7[0], tmem_base + 64 + 24);
                float2 weight_pair_1_7 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_26 = make_float2(_tmem_load_7[0], _tmem_load_7[1]);
                float2 acc_pair_1_40 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1_26 = fma_f32x2_rn_ftz(weight_pair_1_7, cached_pair_1_26, acc_pair_1_40);
                acc32[24] = updated_pair_1_26.x;
                acc32[25] = updated_pair_1_26.y;
                float2 cached_pair_1_27 = make_float2(_tmem_load_7[2], _tmem_load_7[3]);
                float2 acc_pair_1_41 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1_27 = fma_f32x2_rn_ftz(weight_pair_1_7, cached_pair_1_27, acc_pair_1_41);
                acc32[26] = updated_pair_1_27.x;
                acc32[27] = updated_pair_1_27.y;
                float2 correction_pair_1_4 = make_float2(correction, correction);
                float2 acc_pair_1_42 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1_14;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_14) : "l"(*(const unsigned long long*)&acc_pair_1_42), "l"(*(const unsigned long long*)&correction_pair_1_4));
                acc32[28] = scaled_pair_1_14.x;
                acc32[29] = scaled_pair_1_14.y;
                float2 acc_pair_1_43 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1_15;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_15) : "l"(*(const unsigned long long*)&acc_pair_1_43), "l"(*(const unsigned long long*)&correction_pair_1_4));
                acc32[30] = scaled_pair_1_15.x;
                acc32[31] = scaled_pair_1_15.y;
                float2 acc_pair_1_44 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1_16;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_16) : "l"(*(const unsigned long long*)&acc_pair_1_44), "l"(*(const unsigned long long*)&correction_pair_1_4));
                acc32[32] = scaled_pair_1_16.x;
                acc32[33] = scaled_pair_1_16.y;
                float2 acc_pair_1_45 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1_17;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_17) : "l"(*(const unsigned long long*)&acc_pair_1_45), "l"(*(const unsigned long long*)&correction_pair_1_4));
                acc32[34] = scaled_pair_1_17.x;
                acc32[35] = scaled_pair_1_17.y;
                float _tmem_load_8[8];
                tmem_ld_x8(&_tmem_load_8[0], tmem_base + 32);
                float2 weight_pair_1_8 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_28 = make_float2(_tmem_load_8[0], _tmem_load_8[1]);
                float2 acc_pair_1_46 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1_28 = fma_f32x2_rn_ftz(weight_pair_1_8, cached_pair_1_28, acc_pair_1_46);
                acc32[28] = updated_pair_1_28.x;
                acc32[29] = updated_pair_1_28.y;
                float2 cached_pair_1_29 = make_float2(_tmem_load_8[2], _tmem_load_8[3]);
                float2 acc_pair_1_47 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1_29 = fma_f32x2_rn_ftz(weight_pair_1_8, cached_pair_1_29, acc_pair_1_47);
                acc32[30] = updated_pair_1_29.x;
                acc32[31] = updated_pair_1_29.y;
                float2 cached_pair_1_30 = make_float2(_tmem_load_8[4], _tmem_load_8[5]);
                float2 acc_pair_1_48 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1_30 = fma_f32x2_rn_ftz(weight_pair_1_8, cached_pair_1_30, acc_pair_1_48);
                acc32[32] = updated_pair_1_30.x;
                acc32[33] = updated_pair_1_30.y;
                float2 cached_pair_1_31 = make_float2(_tmem_load_8[6], _tmem_load_8[7]);
                float2 acc_pair_1_49 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1_31 = fma_f32x2_rn_ftz(weight_pair_1_8, cached_pair_1_31, acc_pair_1_49);
                acc32[34] = updated_pair_1_31.x;
                acc32[35] = updated_pair_1_31.y;
                float _tmem_load_9[8];
                tmem_ld_x8(&_tmem_load_9[0], tmem_base + 64 + 32);
                float2 weight_pair_1_9 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_32 = make_float2(_tmem_load_9[0], _tmem_load_9[1]);
                float2 acc_pair_1_50 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1_32 = fma_f32x2_rn_ftz(weight_pair_1_9, cached_pair_1_32, acc_pair_1_50);
                acc32[28] = updated_pair_1_32.x;
                acc32[29] = updated_pair_1_32.y;
                float2 cached_pair_1_33 = make_float2(_tmem_load_9[2], _tmem_load_9[3]);
                float2 acc_pair_1_51 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1_33 = fma_f32x2_rn_ftz(weight_pair_1_9, cached_pair_1_33, acc_pair_1_51);
                acc32[30] = updated_pair_1_33.x;
                acc32[31] = updated_pair_1_33.y;
                float2 cached_pair_1_34 = make_float2(_tmem_load_9[4], _tmem_load_9[5]);
                float2 acc_pair_1_52 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1_34 = fma_f32x2_rn_ftz(weight_pair_1_9, cached_pair_1_34, acc_pair_1_52);
                acc32[32] = updated_pair_1_34.x;
                acc32[33] = updated_pair_1_34.y;
                float2 cached_pair_1_35 = make_float2(_tmem_load_9[6], _tmem_load_9[7]);
                float2 acc_pair_1_53 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1_35 = fma_f32x2_rn_ftz(weight_pair_1_9, cached_pair_1_35, acc_pair_1_53);
                acc32[34] = updated_pair_1_35.x;
                acc32[35] = updated_pair_1_35.y;
                float2 correction_pair_1_5 = make_float2(correction, correction);
                float2 acc_pair_1_54 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1_18;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_18) : "l"(*(const unsigned long long*)&acc_pair_1_54), "l"(*(const unsigned long long*)&correction_pair_1_5));
                acc32[36] = scaled_pair_1_18.x;
                acc32[37] = scaled_pair_1_18.y;
                float2 acc_pair_1_55 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1_19;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_19) : "l"(*(const unsigned long long*)&acc_pair_1_55), "l"(*(const unsigned long long*)&correction_pair_1_5));
                acc32[38] = scaled_pair_1_19.x;
                acc32[39] = scaled_pair_1_19.y;
                float2 acc_pair_1_56 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1_20;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_20) : "l"(*(const unsigned long long*)&acc_pair_1_56), "l"(*(const unsigned long long*)&correction_pair_1_5));
                acc32[40] = scaled_pair_1_20.x;
                acc32[41] = scaled_pair_1_20.y;
                float2 acc_pair_1_57 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1_21;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_21) : "l"(*(const unsigned long long*)&acc_pair_1_57), "l"(*(const unsigned long long*)&correction_pair_1_5));
                acc32[42] = scaled_pair_1_21.x;
                acc32[43] = scaled_pair_1_21.y;
                float _tmem_load_10[8];
                tmem_ld_x8(&_tmem_load_10[0], tmem_base + 40);
                float2 weight_pair_1_10 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_36 = make_float2(_tmem_load_10[0], _tmem_load_10[1]);
                float2 acc_pair_1_58 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1_36 = fma_f32x2_rn_ftz(weight_pair_1_10, cached_pair_1_36, acc_pair_1_58);
                acc32[36] = updated_pair_1_36.x;
                acc32[37] = updated_pair_1_36.y;
                float2 cached_pair_1_37 = make_float2(_tmem_load_10[2], _tmem_load_10[3]);
                float2 acc_pair_1_59 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1_37 = fma_f32x2_rn_ftz(weight_pair_1_10, cached_pair_1_37, acc_pair_1_59);
                acc32[38] = updated_pair_1_37.x;
                acc32[39] = updated_pair_1_37.y;
                float2 cached_pair_1_38 = make_float2(_tmem_load_10[4], _tmem_load_10[5]);
                float2 acc_pair_1_60 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1_38 = fma_f32x2_rn_ftz(weight_pair_1_10, cached_pair_1_38, acc_pair_1_60);
                acc32[40] = updated_pair_1_38.x;
                acc32[41] = updated_pair_1_38.y;
                float2 cached_pair_1_39 = make_float2(_tmem_load_10[6], _tmem_load_10[7]);
                float2 acc_pair_1_61 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1_39 = fma_f32x2_rn_ftz(weight_pair_1_10, cached_pair_1_39, acc_pair_1_61);
                acc32[42] = updated_pair_1_39.x;
                acc32[43] = updated_pair_1_39.y;
                float _tmem_load_11[8];
                tmem_ld_x8(&_tmem_load_11[0], tmem_base + 64 + 40);
                float2 weight_pair_1_11 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_40 = make_float2(_tmem_load_11[0], _tmem_load_11[1]);
                float2 acc_pair_1_62 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1_40 = fma_f32x2_rn_ftz(weight_pair_1_11, cached_pair_1_40, acc_pair_1_62);
                acc32[36] = updated_pair_1_40.x;
                acc32[37] = updated_pair_1_40.y;
                float2 cached_pair_1_41 = make_float2(_tmem_load_11[2], _tmem_load_11[3]);
                float2 acc_pair_1_63 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1_41 = fma_f32x2_rn_ftz(weight_pair_1_11, cached_pair_1_41, acc_pair_1_63);
                acc32[38] = updated_pair_1_41.x;
                acc32[39] = updated_pair_1_41.y;
                float2 cached_pair_1_42 = make_float2(_tmem_load_11[4], _tmem_load_11[5]);
                float2 acc_pair_1_64 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1_42 = fma_f32x2_rn_ftz(weight_pair_1_11, cached_pair_1_42, acc_pair_1_64);
                acc32[40] = updated_pair_1_42.x;
                acc32[41] = updated_pair_1_42.y;
                float2 cached_pair_1_43 = make_float2(_tmem_load_11[6], _tmem_load_11[7]);
                float2 acc_pair_1_65 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1_43 = fma_f32x2_rn_ftz(weight_pair_1_11, cached_pair_1_43, acc_pair_1_65);
                acc32[42] = updated_pair_1_43.x;
                acc32[43] = updated_pair_1_43.y;
                float2 correction_pair_1_6 = make_float2(correction, correction);
                float2 acc_pair_1_66 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1_22;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_22) : "l"(*(const unsigned long long*)&acc_pair_1_66), "l"(*(const unsigned long long*)&correction_pair_1_6));
                acc32[44] = scaled_pair_1_22.x;
                acc32[45] = scaled_pair_1_22.y;
                float2 acc_pair_1_67 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1_23;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_23) : "l"(*(const unsigned long long*)&acc_pair_1_67), "l"(*(const unsigned long long*)&correction_pair_1_6));
                acc32[46] = scaled_pair_1_23.x;
                acc32[47] = scaled_pair_1_23.y;
                float2 acc_pair_1_68 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1_24;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_24) : "l"(*(const unsigned long long*)&acc_pair_1_68), "l"(*(const unsigned long long*)&correction_pair_1_6));
                acc32[48] = scaled_pair_1_24.x;
                acc32[49] = scaled_pair_1_24.y;
                float2 acc_pair_1_69 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1_25;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_25) : "l"(*(const unsigned long long*)&acc_pair_1_69), "l"(*(const unsigned long long*)&correction_pair_1_6));
                acc32[50] = scaled_pair_1_25.x;
                acc32[51] = scaled_pair_1_25.y;
                float _tmem_load_12[8];
                tmem_ld_x8(&_tmem_load_12[0], tmem_base + 48);
                float2 weight_pair_1_12 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_44 = make_float2(_tmem_load_12[0], _tmem_load_12[1]);
                float2 acc_pair_1_70 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1_44 = fma_f32x2_rn_ftz(weight_pair_1_12, cached_pair_1_44, acc_pair_1_70);
                acc32[44] = updated_pair_1_44.x;
                acc32[45] = updated_pair_1_44.y;
                float2 cached_pair_1_45 = make_float2(_tmem_load_12[2], _tmem_load_12[3]);
                float2 acc_pair_1_71 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1_45 = fma_f32x2_rn_ftz(weight_pair_1_12, cached_pair_1_45, acc_pair_1_71);
                acc32[46] = updated_pair_1_45.x;
                acc32[47] = updated_pair_1_45.y;
                float2 cached_pair_1_46 = make_float2(_tmem_load_12[4], _tmem_load_12[5]);
                float2 acc_pair_1_72 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1_46 = fma_f32x2_rn_ftz(weight_pair_1_12, cached_pair_1_46, acc_pair_1_72);
                acc32[48] = updated_pair_1_46.x;
                acc32[49] = updated_pair_1_46.y;
                float2 cached_pair_1_47 = make_float2(_tmem_load_12[6], _tmem_load_12[7]);
                float2 acc_pair_1_73 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1_47 = fma_f32x2_rn_ftz(weight_pair_1_12, cached_pair_1_47, acc_pair_1_73);
                acc32[50] = updated_pair_1_47.x;
                acc32[51] = updated_pair_1_47.y;
                float _tmem_load_13[8];
                tmem_ld_x8(&_tmem_load_13[0], tmem_base + 64 + 48);
                float2 weight_pair_1_13 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_48 = make_float2(_tmem_load_13[0], _tmem_load_13[1]);
                float2 acc_pair_1_74 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1_48 = fma_f32x2_rn_ftz(weight_pair_1_13, cached_pair_1_48, acc_pair_1_74);
                acc32[44] = updated_pair_1_48.x;
                acc32[45] = updated_pair_1_48.y;
                float2 cached_pair_1_49 = make_float2(_tmem_load_13[2], _tmem_load_13[3]);
                float2 acc_pair_1_75 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1_49 = fma_f32x2_rn_ftz(weight_pair_1_13, cached_pair_1_49, acc_pair_1_75);
                acc32[46] = updated_pair_1_49.x;
                acc32[47] = updated_pair_1_49.y;
                float2 cached_pair_1_50 = make_float2(_tmem_load_13[4], _tmem_load_13[5]);
                float2 acc_pair_1_76 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1_50 = fma_f32x2_rn_ftz(weight_pair_1_13, cached_pair_1_50, acc_pair_1_76);
                acc32[48] = updated_pair_1_50.x;
                acc32[49] = updated_pair_1_50.y;
                float2 cached_pair_1_51 = make_float2(_tmem_load_13[6], _tmem_load_13[7]);
                float2 acc_pair_1_77 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1_51 = fma_f32x2_rn_ftz(weight_pair_1_13, cached_pair_1_51, acc_pair_1_77);
                acc32[50] = updated_pair_1_51.x;
                acc32[51] = updated_pair_1_51.y;
                float2 correction_pair_1_7 = make_float2(correction, correction);
                float2 acc_pair_1_78 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1_26;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_26) : "l"(*(const unsigned long long*)&acc_pair_1_78), "l"(*(const unsigned long long*)&correction_pair_1_7));
                acc32[52] = scaled_pair_1_26.x;
                acc32[53] = scaled_pair_1_26.y;
                float2 acc_pair_1_79 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1_27;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_27) : "l"(*(const unsigned long long*)&acc_pair_1_79), "l"(*(const unsigned long long*)&correction_pair_1_7));
                acc32[54] = scaled_pair_1_27.x;
                acc32[55] = scaled_pair_1_27.y;
                float _tmem_load_14[4];
                tmem_ld_x4(&_tmem_load_14[0], tmem_base + 56);
                float2 weight_pair_1_14 = make_float2(weights[0], weights[0]);
                float2 cached_pair_1_52 = make_float2(_tmem_load_14[0], _tmem_load_14[1]);
                float2 acc_pair_1_80 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1_52 = fma_f32x2_rn_ftz(weight_pair_1_14, cached_pair_1_52, acc_pair_1_80);
                acc32[52] = updated_pair_1_52.x;
                acc32[53] = updated_pair_1_52.y;
                float2 cached_pair_1_53 = make_float2(_tmem_load_14[2], _tmem_load_14[3]);
                float2 acc_pair_1_81 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1_53 = fma_f32x2_rn_ftz(weight_pair_1_14, cached_pair_1_53, acc_pair_1_81);
                acc32[54] = updated_pair_1_53.x;
                acc32[55] = updated_pair_1_53.y;
                float _tmem_load_15[4];
                tmem_ld_x4(&_tmem_load_15[0], tmem_base + 64 + 56);
                float2 weight_pair_1_15 = make_float2(weights[1], weights[1]);
                float2 cached_pair_1_54 = make_float2(_tmem_load_15[0], _tmem_load_15[1]);
                float2 acc_pair_1_82 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1_54 = fma_f32x2_rn_ftz(weight_pair_1_15, cached_pair_1_54, acc_pair_1_82);
                acc32[52] = updated_pair_1_54.x;
                acc32[53] = updated_pair_1_54.y;
                float2 cached_pair_1_55 = make_float2(_tmem_load_15[2], _tmem_load_15[3]);
                float2 acc_pair_1_83 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1_55 = fma_f32x2_rn_ftz(weight_pair_1_15, cached_pair_1_55, acc_pair_1_83);
                acc32[54] = updated_pair_1_55.x;
                acc32[55] = updated_pair_1_55.y;
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
                int hidden_base_1_16 = thread_in_group * 8;
                unsigned int smem_regs_27[4];
                #pragma unroll
                for (int _lr_16 = 0; _lr_16 < 4; _lr_16++) {
                    smem_regs_27[_lr_16] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_16 / 2) + (unsigned int)_lr_16];
                }
                float values_1_16[8];
                float _unpack_bf16_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                        : "r"(smem_regs_27[_pair]));
                }
                values_1_16[0] = _unpack_bf16_16[0];
                values_1_16[1] = _unpack_bf16_16[1];
                values_1_16[2] = _unpack_bf16_16[2];
                values_1_16[3] = _unpack_bf16_16[3];
                values_1_16[4] = _unpack_bf16_16[4];
                values_1_16[5] = _unpack_bf16_16[5];
                values_1_16[6] = _unpack_bf16_16[6];
                values_1_16[7] = _unpack_bf16_16[7];
                tmem_st_x8_f32(tmem_base, values_1_16);
                float2 value_pair_1_56 = make_float2(values_1_16[0], values_1_16[1]);
                float2 q_pair_35_84 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_56, value_pair_1_56, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_56, q_pair_35_84, local_dot_422[0]);
                float2 value_pair_1_57 = make_float2(values_1_16[2], values_1_16[3]);
                float2 q_pair_35_85 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_57, value_pair_1_57, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_57, q_pair_35_85, local_dot_422[0]);
                float2 value_pair_1_58 = make_float2(values_1_16[4], values_1_16[5]);
                float2 q_pair_35_86 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_58, value_pair_1_58, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_58, q_pair_35_86, local_dot_422[0]);
                float2 value_pair_1_59 = make_float2(values_1_16[6], values_1_16[7]);
                float2 q_pair_35_87 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_59, value_pair_1_59, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_59, q_pair_35_87, local_dot_422[0]);
                unsigned int smem_regs_28[4];
                #pragma unroll
                for (int _lr_17 = 0; _lr_17 < 4; _lr_17++) {
                    smem_regs_28[_lr_17] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_16 / 2) + (unsigned int)_lr_17];
                }
                float values_1_17[8];
                float _unpack_bf16_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                        : "r"(smem_regs_28[_pair]));
                }
                values_1_17[0] = _unpack_bf16_17[0];
                values_1_17[1] = _unpack_bf16_17[1];
                values_1_17[2] = _unpack_bf16_17[2];
                values_1_17[3] = _unpack_bf16_17[3];
                values_1_17[4] = _unpack_bf16_17[4];
                values_1_17[5] = _unpack_bf16_17[5];
                values_1_17[6] = _unpack_bf16_17[6];
                values_1_17[7] = _unpack_bf16_17[7];
                tmem_st_x8_f32(tmem_base + 64, values_1_17);
                float2 value_pair_1_60 = make_float2(values_1_17[0], values_1_17[1]);
                float2 q_pair_35_88 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_60, value_pair_1_60, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_60, q_pair_35_88, local_dot_422[1]);
                float2 value_pair_1_61 = make_float2(values_1_17[2], values_1_17[3]);
                float2 q_pair_35_89 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_61, value_pair_1_61, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_61, q_pair_35_89, local_dot_422[1]);
                float2 value_pair_1_62 = make_float2(values_1_17[4], values_1_17[5]);
                float2 q_pair_35_90 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_62, value_pair_1_62, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_62, q_pair_35_90, local_dot_422[1]);
                float2 value_pair_1_63 = make_float2(values_1_17[6], values_1_17[7]);
                float2 q_pair_35_91 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_63, value_pair_1_63, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_63, q_pair_35_91, local_dot_422[1]);
                int hidden_base_1_17 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_29[4];
                #pragma unroll
                for (int _lr_18 = 0; _lr_18 < 4; _lr_18++) {
                    smem_regs_29[_lr_18] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_17 / 2) + (unsigned int)_lr_18];
                }
                float values_1_18[8];
                float _unpack_bf16_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                        : "r"(smem_regs_29[_pair]));
                }
                values_1_18[0] = _unpack_bf16_18[0];
                values_1_18[1] = _unpack_bf16_18[1];
                values_1_18[2] = _unpack_bf16_18[2];
                values_1_18[3] = _unpack_bf16_18[3];
                values_1_18[4] = _unpack_bf16_18[4];
                values_1_18[5] = _unpack_bf16_18[5];
                values_1_18[6] = _unpack_bf16_18[6];
                values_1_18[7] = _unpack_bf16_18[7];
                tmem_st_x8_f32(tmem_base + 8, values_1_18);
                float2 value_pair_1_64 = make_float2(values_1_18[0], values_1_18[1]);
                float2 q_pair_35_92 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_64, value_pair_1_64, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_64, q_pair_35_92, local_dot_422[0]);
                float2 value_pair_1_65 = make_float2(values_1_18[2], values_1_18[3]);
                float2 q_pair_35_93 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_65, value_pair_1_65, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_65, q_pair_35_93, local_dot_422[0]);
                float2 value_pair_1_66 = make_float2(values_1_18[4], values_1_18[5]);
                float2 q_pair_35_94 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_66, value_pair_1_66, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_66, q_pair_35_94, local_dot_422[0]);
                float2 value_pair_1_67 = make_float2(values_1_18[6], values_1_18[7]);
                float2 q_pair_35_95 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_67, value_pair_1_67, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_67, q_pair_35_95, local_dot_422[0]);
                unsigned int smem_regs_30[4];
                #pragma unroll
                for (int _lr_19 = 0; _lr_19 < 4; _lr_19++) {
                    smem_regs_30[_lr_19] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_17 / 2) + (unsigned int)_lr_19];
                }
                float values_1_19[8];
                float _unpack_bf16_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                        : "r"(smem_regs_30[_pair]));
                }
                values_1_19[0] = _unpack_bf16_19[0];
                values_1_19[1] = _unpack_bf16_19[1];
                values_1_19[2] = _unpack_bf16_19[2];
                values_1_19[3] = _unpack_bf16_19[3];
                values_1_19[4] = _unpack_bf16_19[4];
                values_1_19[5] = _unpack_bf16_19[5];
                values_1_19[6] = _unpack_bf16_19[6];
                values_1_19[7] = _unpack_bf16_19[7];
                tmem_st_x8_f32(tmem_base + 64 + 8, values_1_19);
                float2 value_pair_1_68 = make_float2(values_1_19[0], values_1_19[1]);
                float2 q_pair_35_96 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_68, value_pair_1_68, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_68, q_pair_35_96, local_dot_422[1]);
                float2 value_pair_1_69 = make_float2(values_1_19[2], values_1_19[3]);
                float2 q_pair_35_97 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_69, value_pair_1_69, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_69, q_pair_35_97, local_dot_422[1]);
                float2 value_pair_1_70 = make_float2(values_1_19[4], values_1_19[5]);
                float2 q_pair_35_98 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_70, value_pair_1_70, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_70, q_pair_35_98, local_dot_422[1]);
                float2 value_pair_1_71 = make_float2(values_1_19[6], values_1_19[7]);
                float2 q_pair_35_99 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_71, value_pair_1_71, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_71, q_pair_35_99, local_dot_422[1]);
                int hidden_base_1_18 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_31[4];
                #pragma unroll
                for (int _lr_20 = 0; _lr_20 < 4; _lr_20++) {
                    smem_regs_31[_lr_20] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_18 / 2) + (unsigned int)_lr_20];
                }
                float values_1_20[8];
                float _unpack_bf16_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                        : "r"(smem_regs_31[_pair]));
                }
                values_1_20[0] = _unpack_bf16_20[0];
                values_1_20[1] = _unpack_bf16_20[1];
                values_1_20[2] = _unpack_bf16_20[2];
                values_1_20[3] = _unpack_bf16_20[3];
                values_1_20[4] = _unpack_bf16_20[4];
                values_1_20[5] = _unpack_bf16_20[5];
                values_1_20[6] = _unpack_bf16_20[6];
                values_1_20[7] = _unpack_bf16_20[7];
                tmem_st_x8_f32(tmem_base + 16, values_1_20);
                float2 value_pair_1_72 = make_float2(values_1_20[0], values_1_20[1]);
                float2 q_pair_35_100 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_72, value_pair_1_72, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_72, q_pair_35_100, local_dot_422[0]);
                float2 value_pair_1_73 = make_float2(values_1_20[2], values_1_20[3]);
                float2 q_pair_35_101 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_73, value_pair_1_73, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_73, q_pair_35_101, local_dot_422[0]);
                float2 value_pair_1_74 = make_float2(values_1_20[4], values_1_20[5]);
                float2 q_pair_35_102 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_74, value_pair_1_74, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_74, q_pair_35_102, local_dot_422[0]);
                float2 value_pair_1_75 = make_float2(values_1_20[6], values_1_20[7]);
                float2 q_pair_35_103 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_75, value_pair_1_75, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_75, q_pair_35_103, local_dot_422[0]);
                unsigned int smem_regs_32[4];
                #pragma unroll
                for (int _lr_21 = 0; _lr_21 < 4; _lr_21++) {
                    smem_regs_32[_lr_21] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_18 / 2) + (unsigned int)_lr_21];
                }
                float values_1_21[8];
                float _unpack_bf16_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                        : "r"(smem_regs_32[_pair]));
                }
                values_1_21[0] = _unpack_bf16_21[0];
                values_1_21[1] = _unpack_bf16_21[1];
                values_1_21[2] = _unpack_bf16_21[2];
                values_1_21[3] = _unpack_bf16_21[3];
                values_1_21[4] = _unpack_bf16_21[4];
                values_1_21[5] = _unpack_bf16_21[5];
                values_1_21[6] = _unpack_bf16_21[6];
                values_1_21[7] = _unpack_bf16_21[7];
                tmem_st_x8_f32(tmem_base + 64 + 16, values_1_21);
                float2 value_pair_1_76 = make_float2(values_1_21[0], values_1_21[1]);
                float2 q_pair_35_104 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_76, value_pair_1_76, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_76, q_pair_35_104, local_dot_422[1]);
                float2 value_pair_1_77 = make_float2(values_1_21[2], values_1_21[3]);
                float2 q_pair_35_105 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_77, value_pair_1_77, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_77, q_pair_35_105, local_dot_422[1]);
                float2 value_pair_1_78 = make_float2(values_1_21[4], values_1_21[5]);
                float2 q_pair_35_106 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_78, value_pair_1_78, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_78, q_pair_35_106, local_dot_422[1]);
                float2 value_pair_1_79 = make_float2(values_1_21[6], values_1_21[7]);
                float2 q_pair_35_107 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_79, value_pair_1_79, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_79, q_pair_35_107, local_dot_422[1]);
                int hidden_base_1_19 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_33[2];
                #pragma unroll
                for (int _lr_22 = 0; _lr_22 < 2; _lr_22++) {
                    smem_regs_33[_lr_22] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_19 / 2) + (unsigned int)_lr_22];
                }
                float values_1_22[4];
                float _unpack_bf16_22[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                        : "r"(smem_regs_33[_pair]));
                }
                values_1_22[0] = _unpack_bf16_22[0];
                values_1_22[1] = _unpack_bf16_22[1];
                values_1_22[2] = _unpack_bf16_22[2];
                values_1_22[3] = _unpack_bf16_22[3];
                tmem_st_x4_f32(tmem_base + 24, values_1_22);
                float2 value_pair_1_80 = make_float2(values_1_22[0], values_1_22[1]);
                float2 q_pair_35_108 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_80, value_pair_1_80, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_80, q_pair_35_108, local_dot_422[0]);
                float2 value_pair_1_81 = make_float2(values_1_22[2], values_1_22[3]);
                float2 q_pair_35_109 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_1_81, value_pair_1_81, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_1_81, q_pair_35_109, local_dot_422[0]);
                unsigned int smem_regs_34[2];
                #pragma unroll
                for (int _lr_23 = 0; _lr_23 < 2; _lr_23++) {
                    smem_regs_34[_lr_23] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_19 / 2) + (unsigned int)_lr_23];
                }
                float values_1_23[4];
                float _unpack_bf16_23[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                        : "r"(smem_regs_34[_pair]));
                }
                values_1_23[0] = _unpack_bf16_23[0];
                values_1_23[1] = _unpack_bf16_23[1];
                values_1_23[2] = _unpack_bf16_23[2];
                values_1_23[3] = _unpack_bf16_23[3];
                tmem_st_x4_f32(tmem_base + 64 + 24, values_1_23);
                float2 value_pair_1_82 = make_float2(values_1_23[0], values_1_23[1]);
                float2 q_pair_35_110 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_82, value_pair_1_82, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_82, q_pair_35_110, local_dot_422[1]);
                float2 value_pair_1_83 = make_float2(values_1_23[2], values_1_23[3]);
                float2 q_pair_35_111 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_1_83, value_pair_1_83, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_1_83, q_pair_35_111, local_dot_422[1]);
                int hidden_base_1_20 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_35[4];
                #pragma unroll
                for (int _lr_24 = 0; _lr_24 < 4; _lr_24++) {
                    smem_regs_35[_lr_24] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_20 / 2) + (unsigned int)_lr_24];
                }
                float values_1_24[8];
                float _unpack_bf16_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                        : "r"(smem_regs_35[_pair]));
                }
                values_1_24[0] = _unpack_bf16_24[0];
                values_1_24[1] = _unpack_bf16_24[1];
                values_1_24[2] = _unpack_bf16_24[2];
                values_1_24[3] = _unpack_bf16_24[3];
                values_1_24[4] = _unpack_bf16_24[4];
                values_1_24[5] = _unpack_bf16_24[5];
                values_1_24[6] = _unpack_bf16_24[6];
                values_1_24[7] = _unpack_bf16_24[7];
                tmem_st_x8_f32(tmem_base + 32, values_1_24);
                float2 value_pair_1_84 = make_float2(values_1_24[0], values_1_24[1]);
                float2 q_pair_35_112 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_84, value_pair_1_84, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_84, q_pair_35_112, local_dot_422[2]);
                float2 value_pair_1_85 = make_float2(values_1_24[2], values_1_24[3]);
                float2 q_pair_35_113 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_85, value_pair_1_85, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_85, q_pair_35_113, local_dot_422[2]);
                float2 value_pair_1_86 = make_float2(values_1_24[4], values_1_24[5]);
                float2 q_pair_35_114 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_86, value_pair_1_86, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_86, q_pair_35_114, local_dot_422[2]);
                float2 value_pair_1_87 = make_float2(values_1_24[6], values_1_24[7]);
                float2 q_pair_35_115 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_87, value_pair_1_87, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_87, q_pair_35_115, local_dot_422[2]);
                unsigned int smem_regs_36[4];
                #pragma unroll
                for (int _lr_25 = 0; _lr_25 < 4; _lr_25++) {
                    smem_regs_36[_lr_25] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_20 / 2) + (unsigned int)_lr_25];
                }
                float values_1_25[8];
                float _unpack_bf16_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                        : "r"(smem_regs_36[_pair]));
                }
                values_1_25[0] = _unpack_bf16_25[0];
                values_1_25[1] = _unpack_bf16_25[1];
                values_1_25[2] = _unpack_bf16_25[2];
                values_1_25[3] = _unpack_bf16_25[3];
                values_1_25[4] = _unpack_bf16_25[4];
                values_1_25[5] = _unpack_bf16_25[5];
                values_1_25[6] = _unpack_bf16_25[6];
                values_1_25[7] = _unpack_bf16_25[7];
                tmem_st_x8_f32(tmem_base + 64 + 32, values_1_25);
                float2 value_pair_1_88 = make_float2(values_1_25[0], values_1_25[1]);
                float2 q_pair_35_116 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_88, value_pair_1_88, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_88, q_pair_35_116, local_dot_422[3]);
                float2 value_pair_1_89 = make_float2(values_1_25[2], values_1_25[3]);
                float2 q_pair_35_117 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_89, value_pair_1_89, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_89, q_pair_35_117, local_dot_422[3]);
                float2 value_pair_1_90 = make_float2(values_1_25[4], values_1_25[5]);
                float2 q_pair_35_118 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_90, value_pair_1_90, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_90, q_pair_35_118, local_dot_422[3]);
                float2 value_pair_1_91 = make_float2(values_1_25[6], values_1_25[7]);
                float2 q_pair_35_119 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_91, value_pair_1_91, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_91, q_pair_35_119, local_dot_422[3]);
                int hidden_base_1_21 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_37[4];
                #pragma unroll
                for (int _lr_26 = 0; _lr_26 < 4; _lr_26++) {
                    smem_regs_37[_lr_26] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_21 / 2) + (unsigned int)_lr_26];
                }
                float values_1_26[8];
                float _unpack_bf16_26[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                        : "r"(smem_regs_37[_pair]));
                }
                values_1_26[0] = _unpack_bf16_26[0];
                values_1_26[1] = _unpack_bf16_26[1];
                values_1_26[2] = _unpack_bf16_26[2];
                values_1_26[3] = _unpack_bf16_26[3];
                values_1_26[4] = _unpack_bf16_26[4];
                values_1_26[5] = _unpack_bf16_26[5];
                values_1_26[6] = _unpack_bf16_26[6];
                values_1_26[7] = _unpack_bf16_26[7];
                tmem_st_x8_f32(tmem_base + 40, values_1_26);
                float2 value_pair_1_92 = make_float2(values_1_26[0], values_1_26[1]);
                float2 q_pair_35_120 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_92, value_pair_1_92, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_92, q_pair_35_120, local_dot_422[2]);
                float2 value_pair_1_93 = make_float2(values_1_26[2], values_1_26[3]);
                float2 q_pair_35_121 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_93, value_pair_1_93, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_93, q_pair_35_121, local_dot_422[2]);
                float2 value_pair_1_94 = make_float2(values_1_26[4], values_1_26[5]);
                float2 q_pair_35_122 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_94, value_pair_1_94, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_94, q_pair_35_122, local_dot_422[2]);
                float2 value_pair_1_95 = make_float2(values_1_26[6], values_1_26[7]);
                float2 q_pair_35_123 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_95, value_pair_1_95, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_95, q_pair_35_123, local_dot_422[2]);
                unsigned int smem_regs_38[4];
                #pragma unroll
                for (int _lr_27 = 0; _lr_27 < 4; _lr_27++) {
                    smem_regs_38[_lr_27] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_21 / 2) + (unsigned int)_lr_27];
                }
                float values_1_27[8];
                float _unpack_bf16_27[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                        : "r"(smem_regs_38[_pair]));
                }
                values_1_27[0] = _unpack_bf16_27[0];
                values_1_27[1] = _unpack_bf16_27[1];
                values_1_27[2] = _unpack_bf16_27[2];
                values_1_27[3] = _unpack_bf16_27[3];
                values_1_27[4] = _unpack_bf16_27[4];
                values_1_27[5] = _unpack_bf16_27[5];
                values_1_27[6] = _unpack_bf16_27[6];
                values_1_27[7] = _unpack_bf16_27[7];
                tmem_st_x8_f32(tmem_base + 64 + 40, values_1_27);
                float2 value_pair_1_96 = make_float2(values_1_27[0], values_1_27[1]);
                float2 q_pair_35_124 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_96, value_pair_1_96, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_96, q_pair_35_124, local_dot_422[3]);
                float2 value_pair_1_97 = make_float2(values_1_27[2], values_1_27[3]);
                float2 q_pair_35_125 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_97, value_pair_1_97, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_97, q_pair_35_125, local_dot_422[3]);
                float2 value_pair_1_98 = make_float2(values_1_27[4], values_1_27[5]);
                float2 q_pair_35_126 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_98, value_pair_1_98, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_98, q_pair_35_126, local_dot_422[3]);
                float2 value_pair_1_99 = make_float2(values_1_27[6], values_1_27[7]);
                float2 q_pair_35_127 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_99, value_pair_1_99, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_99, q_pair_35_127, local_dot_422[3]);
                int hidden_base_1_22 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_39[4];
                #pragma unroll
                for (int _lr_28 = 0; _lr_28 < 4; _lr_28++) {
                    smem_regs_39[_lr_28] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_22 / 2) + (unsigned int)_lr_28];
                }
                float values_1_28[8];
                float _unpack_bf16_28[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                        : "r"(smem_regs_39[_pair]));
                }
                values_1_28[0] = _unpack_bf16_28[0];
                values_1_28[1] = _unpack_bf16_28[1];
                values_1_28[2] = _unpack_bf16_28[2];
                values_1_28[3] = _unpack_bf16_28[3];
                values_1_28[4] = _unpack_bf16_28[4];
                values_1_28[5] = _unpack_bf16_28[5];
                values_1_28[6] = _unpack_bf16_28[6];
                values_1_28[7] = _unpack_bf16_28[7];
                tmem_st_x8_f32(tmem_base + 48, values_1_28);
                float2 value_pair_1_100 = make_float2(values_1_28[0], values_1_28[1]);
                float2 q_pair_35_128 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_100, value_pair_1_100, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_100, q_pair_35_128, local_dot_422[2]);
                float2 value_pair_1_101 = make_float2(values_1_28[2], values_1_28[3]);
                float2 q_pair_35_129 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_101, value_pair_1_101, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_101, q_pair_35_129, local_dot_422[2]);
                float2 value_pair_1_102 = make_float2(values_1_28[4], values_1_28[5]);
                float2 q_pair_35_130 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_102, value_pair_1_102, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_102, q_pair_35_130, local_dot_422[2]);
                float2 value_pair_1_103 = make_float2(values_1_28[6], values_1_28[7]);
                float2 q_pair_35_131 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_103, value_pair_1_103, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_103, q_pair_35_131, local_dot_422[2]);
                unsigned int smem_regs_40[4];
                #pragma unroll
                for (int _lr_29 = 0; _lr_29 < 4; _lr_29++) {
                    smem_regs_40[_lr_29] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_22 / 2) + (unsigned int)_lr_29];
                }
                float values_1_29[8];
                float _unpack_bf16_29[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                        : "r"(smem_regs_40[_pair]));
                }
                values_1_29[0] = _unpack_bf16_29[0];
                values_1_29[1] = _unpack_bf16_29[1];
                values_1_29[2] = _unpack_bf16_29[2];
                values_1_29[3] = _unpack_bf16_29[3];
                values_1_29[4] = _unpack_bf16_29[4];
                values_1_29[5] = _unpack_bf16_29[5];
                values_1_29[6] = _unpack_bf16_29[6];
                values_1_29[7] = _unpack_bf16_29[7];
                tmem_st_x8_f32(tmem_base + 64 + 48, values_1_29);
                float2 value_pair_1_104 = make_float2(values_1_29[0], values_1_29[1]);
                float2 q_pair_35_132 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_104, value_pair_1_104, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_104, q_pair_35_132, local_dot_422[3]);
                float2 value_pair_1_105 = make_float2(values_1_29[2], values_1_29[3]);
                float2 q_pair_35_133 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_105, value_pair_1_105, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_105, q_pair_35_133, local_dot_422[3]);
                float2 value_pair_1_106 = make_float2(values_1_29[4], values_1_29[5]);
                float2 q_pair_35_134 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_106, value_pair_1_106, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_106, q_pair_35_134, local_dot_422[3]);
                float2 value_pair_1_107 = make_float2(values_1_29[6], values_1_29[7]);
                float2 q_pair_35_135 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_107, value_pair_1_107, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_107, q_pair_35_135, local_dot_422[3]);
                int hidden_base_1_23 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_41[2];
                #pragma unroll
                for (int _lr_30 = 0; _lr_30 < 2; _lr_30++) {
                    smem_regs_41[_lr_30] = source_words[ring_stage_420 * 2 * 3584 + (unsigned int)(hidden_base_1_23 / 2) + (unsigned int)_lr_30];
                }
                float values_1_30[4];
                float _unpack_bf16_30[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                        : "r"(smem_regs_41[_pair]));
                }
                values_1_30[0] = _unpack_bf16_30[0];
                values_1_30[1] = _unpack_bf16_30[1];
                values_1_30[2] = _unpack_bf16_30[2];
                values_1_30[3] = _unpack_bf16_30[3];
                tmem_st_x4_f32(tmem_base + 56, values_1_30);
                float2 value_pair_1_108 = make_float2(values_1_30[0], values_1_30[1]);
                float2 q_pair_35_136 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_108, value_pair_1_108, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_108, q_pair_35_136, local_dot_422[2]);
                float2 value_pair_1_109 = make_float2(values_1_30[2], values_1_30[3]);
                float2 q_pair_35_137 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_1_109, value_pair_1_109, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_1_109, q_pair_35_137, local_dot_422[2]);
                unsigned int smem_regs_42[2];
                #pragma unroll
                for (int _lr_31 = 0; _lr_31 < 2; _lr_31++) {
                    smem_regs_42[_lr_31] = source_words[(ring_stage_420 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_23 / 2) + (unsigned int)_lr_31];
                }
                float values_1_31[4];
                float _unpack_bf16_31[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                        : "r"(smem_regs_42[_pair]));
                }
                values_1_31[0] = _unpack_bf16_31[0];
                values_1_31[1] = _unpack_bf16_31[1];
                values_1_31[2] = _unpack_bf16_31[2];
                values_1_31[3] = _unpack_bf16_31[3];
                tmem_st_x4_f32(tmem_base + 64 + 56, values_1_31);
                float2 value_pair_1_110 = make_float2(values_1_31[0], values_1_31[1]);
                float2 q_pair_35_138 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_110, value_pair_1_110, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_110, q_pair_35_138, local_dot_422[3]);
                float2 value_pair_1_111 = make_float2(values_1_31[2], values_1_31[3]);
                float2 q_pair_35_139 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_1_111, value_pair_1_111, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_1_111, q_pair_35_139, local_dot_422[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_420) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_575 = stats_stream_base + 32;
                float2 reduce_pair_1_4 = make_float2(local_sum_sq_421[0].x + local_sum_sq_421[0].y, local_dot_422[0].x + local_dot_422[0].y);
                unsigned long long packed_pair_1_4 = 0;
                unsigned long long peer_packed_1_4 = 0;
                float2 peer_pair_1_4 = make_float2(0.0f, 0.0f);
                packed_pair_1_4 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_4)[0];
                unsigned long long _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_4, 16);
                peer_packed_1_4 = _shfl_xor_sync_104;
                peer_pair_1_4 = reinterpret_cast<float2*>(&peer_packed_1_4)[0];
                reduce_pair_1_4 = add_f32x2(reduce_pair_1_4, peer_pair_1_4);
                packed_pair_1_4 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_4)[0];
                unsigned long long _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_4, 8);
                peer_packed_1_4 = _shfl_xor_sync_105;
                peer_pair_1_4 = reinterpret_cast<float2*>(&peer_packed_1_4)[0];
                reduce_pair_1_4 = add_f32x2(reduce_pair_1_4, peer_pair_1_4);
                packed_pair_1_4 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_4)[0];
                unsigned long long _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_4, 4);
                peer_packed_1_4 = _shfl_xor_sync_106;
                peer_pair_1_4 = reinterpret_cast<float2*>(&peer_packed_1_4)[0];
                reduce_pair_1_4 = add_f32x2(reduce_pair_1_4, peer_pair_1_4);
                packed_pair_1_4 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_4)[0];
                unsigned long long _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_4, 2);
                peer_packed_1_4 = _shfl_xor_sync_107;
                peer_pair_1_4 = reinterpret_cast<float2*>(&peer_packed_1_4)[0];
                reduce_pair_1_4 = add_f32x2(reduce_pair_1_4, peer_pair_1_4);
                packed_pair_1_4 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_4)[0];
                unsigned long long _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_4, 1);
                peer_packed_1_4 = _shfl_xor_sync_108;
                peer_pair_1_4 = reinterpret_cast<float2*>(&peer_packed_1_4)[0];
                reduce_pair_1_4 = add_f32x2(reduce_pair_1_4, peer_pair_1_4);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1_4.x;
                    stats[slot_base_575 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1_4.y;
                }
                float2 reduce_pair_1_5 = make_float2(local_sum_sq_421[1].x + local_sum_sq_421[1].y, local_dot_422[1].x + local_dot_422[1].y);
                unsigned long long packed_pair_1_5 = 0;
                unsigned long long peer_packed_1_5 = 0;
                float2 peer_pair_1_5 = make_float2(0.0f, 0.0f);
                packed_pair_1_5 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_5)[0];
                unsigned long long _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_5, 16);
                peer_packed_1_5 = _shfl_xor_sync_109;
                peer_pair_1_5 = reinterpret_cast<float2*>(&peer_packed_1_5)[0];
                reduce_pair_1_5 = add_f32x2(reduce_pair_1_5, peer_pair_1_5);
                packed_pair_1_5 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_5)[0];
                unsigned long long _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_5, 8);
                peer_packed_1_5 = _shfl_xor_sync_110;
                peer_pair_1_5 = reinterpret_cast<float2*>(&peer_packed_1_5)[0];
                reduce_pair_1_5 = add_f32x2(reduce_pair_1_5, peer_pair_1_5);
                packed_pair_1_5 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_5)[0];
                unsigned long long _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_5, 4);
                peer_packed_1_5 = _shfl_xor_sync_111;
                peer_pair_1_5 = reinterpret_cast<float2*>(&peer_packed_1_5)[0];
                reduce_pair_1_5 = add_f32x2(reduce_pair_1_5, peer_pair_1_5);
                packed_pair_1_5 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_5)[0];
                unsigned long long _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_5, 2);
                peer_packed_1_5 = _shfl_xor_sync_112;
                peer_pair_1_5 = reinterpret_cast<float2*>(&peer_packed_1_5)[0];
                reduce_pair_1_5 = add_f32x2(reduce_pair_1_5, peer_pair_1_5);
                packed_pair_1_5 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_5)[0];
                unsigned long long _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_5, 1);
                peer_packed_1_5 = _shfl_xor_sync_113;
                peer_pair_1_5 = reinterpret_cast<float2*>(&peer_packed_1_5)[0];
                reduce_pair_1_5 = add_f32x2(reduce_pair_1_5, peer_pair_1_5);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_1_5.x;
                    stats[slot_base_575 + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_1_5.y;
                }
                float2 reduce_pair_1_6 = make_float2(local_sum_sq_421[2].x + local_sum_sq_421[2].y, local_dot_422[2].x + local_dot_422[2].y);
                unsigned long long packed_pair_1_6 = 0;
                unsigned long long peer_packed_1_6 = 0;
                float2 peer_pair_1_6 = make_float2(0.0f, 0.0f);
                packed_pair_1_6 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_6)[0];
                unsigned long long _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_6, 16);
                peer_packed_1_6 = _shfl_xor_sync_114;
                peer_pair_1_6 = reinterpret_cast<float2*>(&peer_packed_1_6)[0];
                reduce_pair_1_6 = add_f32x2(reduce_pair_1_6, peer_pair_1_6);
                packed_pair_1_6 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_6)[0];
                unsigned long long _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_6, 8);
                peer_packed_1_6 = _shfl_xor_sync_115;
                peer_pair_1_6 = reinterpret_cast<float2*>(&peer_packed_1_6)[0];
                reduce_pair_1_6 = add_f32x2(reduce_pair_1_6, peer_pair_1_6);
                packed_pair_1_6 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_6)[0];
                unsigned long long _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_6, 4);
                peer_packed_1_6 = _shfl_xor_sync_116;
                peer_pair_1_6 = reinterpret_cast<float2*>(&peer_packed_1_6)[0];
                reduce_pair_1_6 = add_f32x2(reduce_pair_1_6, peer_pair_1_6);
                packed_pair_1_6 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_6)[0];
                unsigned long long _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_6, 2);
                peer_packed_1_6 = _shfl_xor_sync_117;
                peer_pair_1_6 = reinterpret_cast<float2*>(&peer_packed_1_6)[0];
                reduce_pair_1_6 = add_f32x2(reduce_pair_1_6, peer_pair_1_6);
                packed_pair_1_6 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_6)[0];
                unsigned long long _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_6, 1);
                peer_packed_1_6 = _shfl_xor_sync_118;
                peer_pair_1_6 = reinterpret_cast<float2*>(&peer_packed_1_6)[0];
                reduce_pair_1_6 = add_f32x2(reduce_pair_1_6, peer_pair_1_6);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1_6.x;
                    stats[slot_base_575 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1_6.y;
                }
                float2 reduce_pair_1_7 = make_float2(local_sum_sq_421[3].x + local_sum_sq_421[3].y, local_dot_422[3].x + local_dot_422[3].y);
                unsigned long long packed_pair_1_7 = 0;
                unsigned long long peer_packed_1_7 = 0;
                float2 peer_pair_1_7 = make_float2(0.0f, 0.0f);
                packed_pair_1_7 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_7)[0];
                unsigned long long _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_7, 16);
                peer_packed_1_7 = _shfl_xor_sync_119;
                peer_pair_1_7 = reinterpret_cast<float2*>(&peer_packed_1_7)[0];
                reduce_pair_1_7 = add_f32x2(reduce_pair_1_7, peer_pair_1_7);
                packed_pair_1_7 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_7)[0];
                unsigned long long _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_7, 8);
                peer_packed_1_7 = _shfl_xor_sync_120;
                peer_pair_1_7 = reinterpret_cast<float2*>(&peer_packed_1_7)[0];
                reduce_pair_1_7 = add_f32x2(reduce_pair_1_7, peer_pair_1_7);
                packed_pair_1_7 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_7)[0];
                unsigned long long _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_7, 4);
                peer_packed_1_7 = _shfl_xor_sync_121;
                peer_pair_1_7 = reinterpret_cast<float2*>(&peer_packed_1_7)[0];
                reduce_pair_1_7 = add_f32x2(reduce_pair_1_7, peer_pair_1_7);
                packed_pair_1_7 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_7)[0];
                unsigned long long _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_7, 2);
                peer_packed_1_7 = _shfl_xor_sync_122;
                peer_pair_1_7 = reinterpret_cast<float2*>(&peer_packed_1_7)[0];
                reduce_pair_1_7 = add_f32x2(reduce_pair_1_7, peer_pair_1_7);
                packed_pair_1_7 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_7)[0];
                unsigned long long _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_7, 1);
                peer_packed_1_7 = _shfl_xor_sync_123;
                peer_pair_1_7 = reinterpret_cast<float2*>(&peer_packed_1_7)[0];
                reduce_pair_1_7 = add_f32x2(reduce_pair_1_7, peer_pair_1_7);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2)] = reduce_pair_1_7.x;
                    stats[slot_base_575 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2) + 1] = reduce_pair_1_7.y;
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
                float2 correction_pair_1_8 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_84 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1_28;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_28) : "l"(*(const unsigned long long*)&acc_pair_1_84), "l"(*(const unsigned long long*)&correction_pair_1_8));
                acc32[0] = scaled_pair_1_28.x;
                acc32[1] = scaled_pair_1_28.y;
                float2 acc_pair_1_85 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1_29;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_29) : "l"(*(const unsigned long long*)&acc_pair_1_85), "l"(*(const unsigned long long*)&correction_pair_1_8));
                acc32[2] = scaled_pair_1_29.x;
                acc32[3] = scaled_pair_1_29.y;
                float2 acc_pair_1_86 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1_30;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_30) : "l"(*(const unsigned long long*)&acc_pair_1_86), "l"(*(const unsigned long long*)&correction_pair_1_8));
                acc32[4] = scaled_pair_1_30.x;
                acc32[5] = scaled_pair_1_30.y;
                float2 acc_pair_1_87 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1_31;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_31) : "l"(*(const unsigned long long*)&acc_pair_1_87), "l"(*(const unsigned long long*)&correction_pair_1_8));
                acc32[6] = scaled_pair_1_31.x;
                acc32[7] = scaled_pair_1_31.y;
                float _tmem_load_16[8];
                tmem_ld_x8(&_tmem_load_16[0], tmem_base);
                float2 weight_pair_1_16 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_56 = make_float2(_tmem_load_16[0], _tmem_load_16[1]);
                float2 acc_pair_1_88 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1_56 = fma_f32x2_rn_ftz(weight_pair_1_16, cached_pair_1_56, acc_pair_1_88);
                acc32[0] = updated_pair_1_56.x;
                acc32[1] = updated_pair_1_56.y;
                float2 cached_pair_1_57 = make_float2(_tmem_load_16[2], _tmem_load_16[3]);
                float2 acc_pair_1_89 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1_57 = fma_f32x2_rn_ftz(weight_pair_1_16, cached_pair_1_57, acc_pair_1_89);
                acc32[2] = updated_pair_1_57.x;
                acc32[3] = updated_pair_1_57.y;
                float2 cached_pair_1_58 = make_float2(_tmem_load_16[4], _tmem_load_16[5]);
                float2 acc_pair_1_90 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1_58 = fma_f32x2_rn_ftz(weight_pair_1_16, cached_pair_1_58, acc_pair_1_90);
                acc32[4] = updated_pair_1_58.x;
                acc32[5] = updated_pair_1_58.y;
                float2 cached_pair_1_59 = make_float2(_tmem_load_16[6], _tmem_load_16[7]);
                float2 acc_pair_1_91 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1_59 = fma_f32x2_rn_ftz(weight_pair_1_16, cached_pair_1_59, acc_pair_1_91);
                acc32[6] = updated_pair_1_59.x;
                acc32[7] = updated_pair_1_59.y;
                float _tmem_load_17[8];
                tmem_ld_x8(&_tmem_load_17[0], tmem_base + 64);
                float2 weight_pair_1_17 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_60 = make_float2(_tmem_load_17[0], _tmem_load_17[1]);
                float2 acc_pair_1_92 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1_60 = fma_f32x2_rn_ftz(weight_pair_1_17, cached_pair_1_60, acc_pair_1_92);
                acc32[0] = updated_pair_1_60.x;
                acc32[1] = updated_pair_1_60.y;
                float2 cached_pair_1_61 = make_float2(_tmem_load_17[2], _tmem_load_17[3]);
                float2 acc_pair_1_93 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1_61 = fma_f32x2_rn_ftz(weight_pair_1_17, cached_pair_1_61, acc_pair_1_93);
                acc32[2] = updated_pair_1_61.x;
                acc32[3] = updated_pair_1_61.y;
                float2 cached_pair_1_62 = make_float2(_tmem_load_17[4], _tmem_load_17[5]);
                float2 acc_pair_1_94 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1_62 = fma_f32x2_rn_ftz(weight_pair_1_17, cached_pair_1_62, acc_pair_1_94);
                acc32[4] = updated_pair_1_62.x;
                acc32[5] = updated_pair_1_62.y;
                float2 cached_pair_1_63 = make_float2(_tmem_load_17[6], _tmem_load_17[7]);
                float2 acc_pair_1_95 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1_63 = fma_f32x2_rn_ftz(weight_pair_1_17, cached_pair_1_63, acc_pair_1_95);
                acc32[6] = updated_pair_1_63.x;
                acc32[7] = updated_pair_1_63.y;
                float2 correction_pair_1_9 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_96 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1_32;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_32) : "l"(*(const unsigned long long*)&acc_pair_1_96), "l"(*(const unsigned long long*)&correction_pair_1_9));
                acc32[8] = scaled_pair_1_32.x;
                acc32[9] = scaled_pair_1_32.y;
                float2 acc_pair_1_97 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1_33;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_33) : "l"(*(const unsigned long long*)&acc_pair_1_97), "l"(*(const unsigned long long*)&correction_pair_1_9));
                acc32[10] = scaled_pair_1_33.x;
                acc32[11] = scaled_pair_1_33.y;
                float2 acc_pair_1_98 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1_34;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_34) : "l"(*(const unsigned long long*)&acc_pair_1_98), "l"(*(const unsigned long long*)&correction_pair_1_9));
                acc32[12] = scaled_pair_1_34.x;
                acc32[13] = scaled_pair_1_34.y;
                float2 acc_pair_1_99 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1_35;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_35) : "l"(*(const unsigned long long*)&acc_pair_1_99), "l"(*(const unsigned long long*)&correction_pair_1_9));
                acc32[14] = scaled_pair_1_35.x;
                acc32[15] = scaled_pair_1_35.y;
                float _tmem_load_18[8];
                tmem_ld_x8(&_tmem_load_18[0], tmem_base + 8);
                float2 weight_pair_1_18 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_64 = make_float2(_tmem_load_18[0], _tmem_load_18[1]);
                float2 acc_pair_1_100 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1_64 = fma_f32x2_rn_ftz(weight_pair_1_18, cached_pair_1_64, acc_pair_1_100);
                acc32[8] = updated_pair_1_64.x;
                acc32[9] = updated_pair_1_64.y;
                float2 cached_pair_1_65 = make_float2(_tmem_load_18[2], _tmem_load_18[3]);
                float2 acc_pair_1_101 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1_65 = fma_f32x2_rn_ftz(weight_pair_1_18, cached_pair_1_65, acc_pair_1_101);
                acc32[10] = updated_pair_1_65.x;
                acc32[11] = updated_pair_1_65.y;
                float2 cached_pair_1_66 = make_float2(_tmem_load_18[4], _tmem_load_18[5]);
                float2 acc_pair_1_102 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1_66 = fma_f32x2_rn_ftz(weight_pair_1_18, cached_pair_1_66, acc_pair_1_102);
                acc32[12] = updated_pair_1_66.x;
                acc32[13] = updated_pair_1_66.y;
                float2 cached_pair_1_67 = make_float2(_tmem_load_18[6], _tmem_load_18[7]);
                float2 acc_pair_1_103 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1_67 = fma_f32x2_rn_ftz(weight_pair_1_18, cached_pair_1_67, acc_pair_1_103);
                acc32[14] = updated_pair_1_67.x;
                acc32[15] = updated_pair_1_67.y;
                float _tmem_load_19[8];
                tmem_ld_x8(&_tmem_load_19[0], tmem_base + 64 + 8);
                float2 weight_pair_1_19 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_68 = make_float2(_tmem_load_19[0], _tmem_load_19[1]);
                float2 acc_pair_1_104 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1_68 = fma_f32x2_rn_ftz(weight_pair_1_19, cached_pair_1_68, acc_pair_1_104);
                acc32[8] = updated_pair_1_68.x;
                acc32[9] = updated_pair_1_68.y;
                float2 cached_pair_1_69 = make_float2(_tmem_load_19[2], _tmem_load_19[3]);
                float2 acc_pair_1_105 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1_69 = fma_f32x2_rn_ftz(weight_pair_1_19, cached_pair_1_69, acc_pair_1_105);
                acc32[10] = updated_pair_1_69.x;
                acc32[11] = updated_pair_1_69.y;
                float2 cached_pair_1_70 = make_float2(_tmem_load_19[4], _tmem_load_19[5]);
                float2 acc_pair_1_106 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1_70 = fma_f32x2_rn_ftz(weight_pair_1_19, cached_pair_1_70, acc_pair_1_106);
                acc32[12] = updated_pair_1_70.x;
                acc32[13] = updated_pair_1_70.y;
                float2 cached_pair_1_71 = make_float2(_tmem_load_19[6], _tmem_load_19[7]);
                float2 acc_pair_1_107 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1_71 = fma_f32x2_rn_ftz(weight_pair_1_19, cached_pair_1_71, acc_pair_1_107);
                acc32[14] = updated_pair_1_71.x;
                acc32[15] = updated_pair_1_71.y;
                float2 correction_pair_1_10 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_108 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1_36;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_36) : "l"(*(const unsigned long long*)&acc_pair_1_108), "l"(*(const unsigned long long*)&correction_pair_1_10));
                acc32[16] = scaled_pair_1_36.x;
                acc32[17] = scaled_pair_1_36.y;
                float2 acc_pair_1_109 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1_37;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_37) : "l"(*(const unsigned long long*)&acc_pair_1_109), "l"(*(const unsigned long long*)&correction_pair_1_10));
                acc32[18] = scaled_pair_1_37.x;
                acc32[19] = scaled_pair_1_37.y;
                float2 acc_pair_1_110 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1_38;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_38) : "l"(*(const unsigned long long*)&acc_pair_1_110), "l"(*(const unsigned long long*)&correction_pair_1_10));
                acc32[20] = scaled_pair_1_38.x;
                acc32[21] = scaled_pair_1_38.y;
                float2 acc_pair_1_111 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1_39;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_39) : "l"(*(const unsigned long long*)&acc_pair_1_111), "l"(*(const unsigned long long*)&correction_pair_1_10));
                acc32[22] = scaled_pair_1_39.x;
                acc32[23] = scaled_pair_1_39.y;
                float _tmem_load_20[8];
                tmem_ld_x8(&_tmem_load_20[0], tmem_base + 16);
                float2 weight_pair_1_20 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_72 = make_float2(_tmem_load_20[0], _tmem_load_20[1]);
                float2 acc_pair_1_112 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1_72 = fma_f32x2_rn_ftz(weight_pair_1_20, cached_pair_1_72, acc_pair_1_112);
                acc32[16] = updated_pair_1_72.x;
                acc32[17] = updated_pair_1_72.y;
                float2 cached_pair_1_73 = make_float2(_tmem_load_20[2], _tmem_load_20[3]);
                float2 acc_pair_1_113 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1_73 = fma_f32x2_rn_ftz(weight_pair_1_20, cached_pair_1_73, acc_pair_1_113);
                acc32[18] = updated_pair_1_73.x;
                acc32[19] = updated_pair_1_73.y;
                float2 cached_pair_1_74 = make_float2(_tmem_load_20[4], _tmem_load_20[5]);
                float2 acc_pair_1_114 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1_74 = fma_f32x2_rn_ftz(weight_pair_1_20, cached_pair_1_74, acc_pair_1_114);
                acc32[20] = updated_pair_1_74.x;
                acc32[21] = updated_pair_1_74.y;
                float2 cached_pair_1_75 = make_float2(_tmem_load_20[6], _tmem_load_20[7]);
                float2 acc_pair_1_115 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1_75 = fma_f32x2_rn_ftz(weight_pair_1_20, cached_pair_1_75, acc_pair_1_115);
                acc32[22] = updated_pair_1_75.x;
                acc32[23] = updated_pair_1_75.y;
                float _tmem_load_21[8];
                tmem_ld_x8(&_tmem_load_21[0], tmem_base + 64 + 16);
                float2 weight_pair_1_21 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_76 = make_float2(_tmem_load_21[0], _tmem_load_21[1]);
                float2 acc_pair_1_116 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1_76 = fma_f32x2_rn_ftz(weight_pair_1_21, cached_pair_1_76, acc_pair_1_116);
                acc32[16] = updated_pair_1_76.x;
                acc32[17] = updated_pair_1_76.y;
                float2 cached_pair_1_77 = make_float2(_tmem_load_21[2], _tmem_load_21[3]);
                float2 acc_pair_1_117 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1_77 = fma_f32x2_rn_ftz(weight_pair_1_21, cached_pair_1_77, acc_pair_1_117);
                acc32[18] = updated_pair_1_77.x;
                acc32[19] = updated_pair_1_77.y;
                float2 cached_pair_1_78 = make_float2(_tmem_load_21[4], _tmem_load_21[5]);
                float2 acc_pair_1_118 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1_78 = fma_f32x2_rn_ftz(weight_pair_1_21, cached_pair_1_78, acc_pair_1_118);
                acc32[20] = updated_pair_1_78.x;
                acc32[21] = updated_pair_1_78.y;
                float2 cached_pair_1_79 = make_float2(_tmem_load_21[6], _tmem_load_21[7]);
                float2 acc_pair_1_119 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1_79 = fma_f32x2_rn_ftz(weight_pair_1_21, cached_pair_1_79, acc_pair_1_119);
                acc32[22] = updated_pair_1_79.x;
                acc32[23] = updated_pair_1_79.y;
                float2 correction_pair_1_11 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_120 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1_40;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_40) : "l"(*(const unsigned long long*)&acc_pair_1_120), "l"(*(const unsigned long long*)&correction_pair_1_11));
                acc32[24] = scaled_pair_1_40.x;
                acc32[25] = scaled_pair_1_40.y;
                float2 acc_pair_1_121 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1_41;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_41) : "l"(*(const unsigned long long*)&acc_pair_1_121), "l"(*(const unsigned long long*)&correction_pair_1_11));
                acc32[26] = scaled_pair_1_41.x;
                acc32[27] = scaled_pair_1_41.y;
                float _tmem_load_22[4];
                tmem_ld_x4(&_tmem_load_22[0], tmem_base + 24);
                float2 weight_pair_1_22 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_80 = make_float2(_tmem_load_22[0], _tmem_load_22[1]);
                float2 acc_pair_1_122 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1_80 = fma_f32x2_rn_ftz(weight_pair_1_22, cached_pair_1_80, acc_pair_1_122);
                acc32[24] = updated_pair_1_80.x;
                acc32[25] = updated_pair_1_80.y;
                float2 cached_pair_1_81 = make_float2(_tmem_load_22[2], _tmem_load_22[3]);
                float2 acc_pair_1_123 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1_81 = fma_f32x2_rn_ftz(weight_pair_1_22, cached_pair_1_81, acc_pair_1_123);
                acc32[26] = updated_pair_1_81.x;
                acc32[27] = updated_pair_1_81.y;
                float _tmem_load_23[4];
                tmem_ld_x4(&_tmem_load_23[0], tmem_base + 64 + 24);
                float2 weight_pair_1_23 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_82 = make_float2(_tmem_load_23[0], _tmem_load_23[1]);
                float2 acc_pair_1_124 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1_82 = fma_f32x2_rn_ftz(weight_pair_1_23, cached_pair_1_82, acc_pair_1_124);
                acc32[24] = updated_pair_1_82.x;
                acc32[25] = updated_pair_1_82.y;
                float2 cached_pair_1_83 = make_float2(_tmem_load_23[2], _tmem_load_23[3]);
                float2 acc_pair_1_125 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1_83 = fma_f32x2_rn_ftz(weight_pair_1_23, cached_pair_1_83, acc_pair_1_125);
                acc32[26] = updated_pair_1_83.x;
                acc32[27] = updated_pair_1_83.y;
                float2 correction_pair_1_12 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_126 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1_42;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_42) : "l"(*(const unsigned long long*)&acc_pair_1_126), "l"(*(const unsigned long long*)&correction_pair_1_12));
                acc32[28] = scaled_pair_1_42.x;
                acc32[29] = scaled_pair_1_42.y;
                float2 acc_pair_1_127 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1_43;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_43) : "l"(*(const unsigned long long*)&acc_pair_1_127), "l"(*(const unsigned long long*)&correction_pair_1_12));
                acc32[30] = scaled_pair_1_43.x;
                acc32[31] = scaled_pair_1_43.y;
                float2 acc_pair_1_128 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1_44;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_44) : "l"(*(const unsigned long long*)&acc_pair_1_128), "l"(*(const unsigned long long*)&correction_pair_1_12));
                acc32[32] = scaled_pair_1_44.x;
                acc32[33] = scaled_pair_1_44.y;
                float2 acc_pair_1_129 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1_45;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_45) : "l"(*(const unsigned long long*)&acc_pair_1_129), "l"(*(const unsigned long long*)&correction_pair_1_12));
                acc32[34] = scaled_pair_1_45.x;
                acc32[35] = scaled_pair_1_45.y;
                float _tmem_load_24[8];
                tmem_ld_x8(&_tmem_load_24[0], tmem_base + 32);
                float2 weight_pair_1_24 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_84 = make_float2(_tmem_load_24[0], _tmem_load_24[1]);
                float2 acc_pair_1_130 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1_84 = fma_f32x2_rn_ftz(weight_pair_1_24, cached_pair_1_84, acc_pair_1_130);
                acc32[28] = updated_pair_1_84.x;
                acc32[29] = updated_pair_1_84.y;
                float2 cached_pair_1_85 = make_float2(_tmem_load_24[2], _tmem_load_24[3]);
                float2 acc_pair_1_131 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1_85 = fma_f32x2_rn_ftz(weight_pair_1_24, cached_pair_1_85, acc_pair_1_131);
                acc32[30] = updated_pair_1_85.x;
                acc32[31] = updated_pair_1_85.y;
                float2 cached_pair_1_86 = make_float2(_tmem_load_24[4], _tmem_load_24[5]);
                float2 acc_pair_1_132 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1_86 = fma_f32x2_rn_ftz(weight_pair_1_24, cached_pair_1_86, acc_pair_1_132);
                acc32[32] = updated_pair_1_86.x;
                acc32[33] = updated_pair_1_86.y;
                float2 cached_pair_1_87 = make_float2(_tmem_load_24[6], _tmem_load_24[7]);
                float2 acc_pair_1_133 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1_87 = fma_f32x2_rn_ftz(weight_pair_1_24, cached_pair_1_87, acc_pair_1_133);
                acc32[34] = updated_pair_1_87.x;
                acc32[35] = updated_pair_1_87.y;
                float _tmem_load_25[8];
                tmem_ld_x8(&_tmem_load_25[0], tmem_base + 64 + 32);
                float2 weight_pair_1_25 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_88 = make_float2(_tmem_load_25[0], _tmem_load_25[1]);
                float2 acc_pair_1_134 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1_88 = fma_f32x2_rn_ftz(weight_pair_1_25, cached_pair_1_88, acc_pair_1_134);
                acc32[28] = updated_pair_1_88.x;
                acc32[29] = updated_pair_1_88.y;
                float2 cached_pair_1_89 = make_float2(_tmem_load_25[2], _tmem_load_25[3]);
                float2 acc_pair_1_135 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1_89 = fma_f32x2_rn_ftz(weight_pair_1_25, cached_pair_1_89, acc_pair_1_135);
                acc32[30] = updated_pair_1_89.x;
                acc32[31] = updated_pair_1_89.y;
                float2 cached_pair_1_90 = make_float2(_tmem_load_25[4], _tmem_load_25[5]);
                float2 acc_pair_1_136 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1_90 = fma_f32x2_rn_ftz(weight_pair_1_25, cached_pair_1_90, acc_pair_1_136);
                acc32[32] = updated_pair_1_90.x;
                acc32[33] = updated_pair_1_90.y;
                float2 cached_pair_1_91 = make_float2(_tmem_load_25[6], _tmem_load_25[7]);
                float2 acc_pair_1_137 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1_91 = fma_f32x2_rn_ftz(weight_pair_1_25, cached_pair_1_91, acc_pair_1_137);
                acc32[34] = updated_pair_1_91.x;
                acc32[35] = updated_pair_1_91.y;
                float2 correction_pair_1_13 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_138 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1_46;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_46) : "l"(*(const unsigned long long*)&acc_pair_1_138), "l"(*(const unsigned long long*)&correction_pair_1_13));
                acc32[36] = scaled_pair_1_46.x;
                acc32[37] = scaled_pair_1_46.y;
                float2 acc_pair_1_139 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1_47;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_47) : "l"(*(const unsigned long long*)&acc_pair_1_139), "l"(*(const unsigned long long*)&correction_pair_1_13));
                acc32[38] = scaled_pair_1_47.x;
                acc32[39] = scaled_pair_1_47.y;
                float2 acc_pair_1_140 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1_48;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_48) : "l"(*(const unsigned long long*)&acc_pair_1_140), "l"(*(const unsigned long long*)&correction_pair_1_13));
                acc32[40] = scaled_pair_1_48.x;
                acc32[41] = scaled_pair_1_48.y;
                float2 acc_pair_1_141 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1_49;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_49) : "l"(*(const unsigned long long*)&acc_pair_1_141), "l"(*(const unsigned long long*)&correction_pair_1_13));
                acc32[42] = scaled_pair_1_49.x;
                acc32[43] = scaled_pair_1_49.y;
                float _tmem_load_26[8];
                tmem_ld_x8(&_tmem_load_26[0], tmem_base + 40);
                float2 weight_pair_1_26 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_92 = make_float2(_tmem_load_26[0], _tmem_load_26[1]);
                float2 acc_pair_1_142 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1_92 = fma_f32x2_rn_ftz(weight_pair_1_26, cached_pair_1_92, acc_pair_1_142);
                acc32[36] = updated_pair_1_92.x;
                acc32[37] = updated_pair_1_92.y;
                float2 cached_pair_1_93 = make_float2(_tmem_load_26[2], _tmem_load_26[3]);
                float2 acc_pair_1_143 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1_93 = fma_f32x2_rn_ftz(weight_pair_1_26, cached_pair_1_93, acc_pair_1_143);
                acc32[38] = updated_pair_1_93.x;
                acc32[39] = updated_pair_1_93.y;
                float2 cached_pair_1_94 = make_float2(_tmem_load_26[4], _tmem_load_26[5]);
                float2 acc_pair_1_144 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1_94 = fma_f32x2_rn_ftz(weight_pair_1_26, cached_pair_1_94, acc_pair_1_144);
                acc32[40] = updated_pair_1_94.x;
                acc32[41] = updated_pair_1_94.y;
                float2 cached_pair_1_95 = make_float2(_tmem_load_26[6], _tmem_load_26[7]);
                float2 acc_pair_1_145 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1_95 = fma_f32x2_rn_ftz(weight_pair_1_26, cached_pair_1_95, acc_pair_1_145);
                acc32[42] = updated_pair_1_95.x;
                acc32[43] = updated_pair_1_95.y;
                float _tmem_load_27[8];
                tmem_ld_x8(&_tmem_load_27[0], tmem_base + 64 + 40);
                float2 weight_pair_1_27 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_96 = make_float2(_tmem_load_27[0], _tmem_load_27[1]);
                float2 acc_pair_1_146 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1_96 = fma_f32x2_rn_ftz(weight_pair_1_27, cached_pair_1_96, acc_pair_1_146);
                acc32[36] = updated_pair_1_96.x;
                acc32[37] = updated_pair_1_96.y;
                float2 cached_pair_1_97 = make_float2(_tmem_load_27[2], _tmem_load_27[3]);
                float2 acc_pair_1_147 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1_97 = fma_f32x2_rn_ftz(weight_pair_1_27, cached_pair_1_97, acc_pair_1_147);
                acc32[38] = updated_pair_1_97.x;
                acc32[39] = updated_pair_1_97.y;
                float2 cached_pair_1_98 = make_float2(_tmem_load_27[4], _tmem_load_27[5]);
                float2 acc_pair_1_148 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1_98 = fma_f32x2_rn_ftz(weight_pair_1_27, cached_pair_1_98, acc_pair_1_148);
                acc32[40] = updated_pair_1_98.x;
                acc32[41] = updated_pair_1_98.y;
                float2 cached_pair_1_99 = make_float2(_tmem_load_27[6], _tmem_load_27[7]);
                float2 acc_pair_1_149 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1_99 = fma_f32x2_rn_ftz(weight_pair_1_27, cached_pair_1_99, acc_pair_1_149);
                acc32[42] = updated_pair_1_99.x;
                acc32[43] = updated_pair_1_99.y;
                float2 correction_pair_1_14 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_150 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1_50;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_50) : "l"(*(const unsigned long long*)&acc_pair_1_150), "l"(*(const unsigned long long*)&correction_pair_1_14));
                acc32[44] = scaled_pair_1_50.x;
                acc32[45] = scaled_pair_1_50.y;
                float2 acc_pair_1_151 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1_51;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_51) : "l"(*(const unsigned long long*)&acc_pair_1_151), "l"(*(const unsigned long long*)&correction_pair_1_14));
                acc32[46] = scaled_pair_1_51.x;
                acc32[47] = scaled_pair_1_51.y;
                float2 acc_pair_1_152 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1_52;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_52) : "l"(*(const unsigned long long*)&acc_pair_1_152), "l"(*(const unsigned long long*)&correction_pair_1_14));
                acc32[48] = scaled_pair_1_52.x;
                acc32[49] = scaled_pair_1_52.y;
                float2 acc_pair_1_153 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1_53;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_53) : "l"(*(const unsigned long long*)&acc_pair_1_153), "l"(*(const unsigned long long*)&correction_pair_1_14));
                acc32[50] = scaled_pair_1_53.x;
                acc32[51] = scaled_pair_1_53.y;
                float _tmem_load_28[8];
                tmem_ld_x8(&_tmem_load_28[0], tmem_base + 48);
                float2 weight_pair_1_28 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_100 = make_float2(_tmem_load_28[0], _tmem_load_28[1]);
                float2 acc_pair_1_154 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1_100 = fma_f32x2_rn_ftz(weight_pair_1_28, cached_pair_1_100, acc_pair_1_154);
                acc32[44] = updated_pair_1_100.x;
                acc32[45] = updated_pair_1_100.y;
                float2 cached_pair_1_101 = make_float2(_tmem_load_28[2], _tmem_load_28[3]);
                float2 acc_pair_1_155 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1_101 = fma_f32x2_rn_ftz(weight_pair_1_28, cached_pair_1_101, acc_pair_1_155);
                acc32[46] = updated_pair_1_101.x;
                acc32[47] = updated_pair_1_101.y;
                float2 cached_pair_1_102 = make_float2(_tmem_load_28[4], _tmem_load_28[5]);
                float2 acc_pair_1_156 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1_102 = fma_f32x2_rn_ftz(weight_pair_1_28, cached_pair_1_102, acc_pair_1_156);
                acc32[48] = updated_pair_1_102.x;
                acc32[49] = updated_pair_1_102.y;
                float2 cached_pair_1_103 = make_float2(_tmem_load_28[6], _tmem_load_28[7]);
                float2 acc_pair_1_157 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1_103 = fma_f32x2_rn_ftz(weight_pair_1_28, cached_pair_1_103, acc_pair_1_157);
                acc32[50] = updated_pair_1_103.x;
                acc32[51] = updated_pair_1_103.y;
                float _tmem_load_29[8];
                tmem_ld_x8(&_tmem_load_29[0], tmem_base + 64 + 48);
                float2 weight_pair_1_29 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_104 = make_float2(_tmem_load_29[0], _tmem_load_29[1]);
                float2 acc_pair_1_158 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1_104 = fma_f32x2_rn_ftz(weight_pair_1_29, cached_pair_1_104, acc_pair_1_158);
                acc32[44] = updated_pair_1_104.x;
                acc32[45] = updated_pair_1_104.y;
                float2 cached_pair_1_105 = make_float2(_tmem_load_29[2], _tmem_load_29[3]);
                float2 acc_pair_1_159 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1_105 = fma_f32x2_rn_ftz(weight_pair_1_29, cached_pair_1_105, acc_pair_1_159);
                acc32[46] = updated_pair_1_105.x;
                acc32[47] = updated_pair_1_105.y;
                float2 cached_pair_1_106 = make_float2(_tmem_load_29[4], _tmem_load_29[5]);
                float2 acc_pair_1_160 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1_106 = fma_f32x2_rn_ftz(weight_pair_1_29, cached_pair_1_106, acc_pair_1_160);
                acc32[48] = updated_pair_1_106.x;
                acc32[49] = updated_pair_1_106.y;
                float2 cached_pair_1_107 = make_float2(_tmem_load_29[6], _tmem_load_29[7]);
                float2 acc_pair_1_161 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1_107 = fma_f32x2_rn_ftz(weight_pair_1_29, cached_pair_1_107, acc_pair_1_161);
                acc32[50] = updated_pair_1_107.x;
                acc32[51] = updated_pair_1_107.y;
                float2 correction_pair_1_15 = make_float2(correction_600, correction_600);
                float2 acc_pair_1_162 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1_54;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_54) : "l"(*(const unsigned long long*)&acc_pair_1_162), "l"(*(const unsigned long long*)&correction_pair_1_15));
                acc32[52] = scaled_pair_1_54.x;
                acc32[53] = scaled_pair_1_54.y;
                float2 acc_pair_1_163 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1_55;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_55) : "l"(*(const unsigned long long*)&acc_pair_1_163), "l"(*(const unsigned long long*)&correction_pair_1_15));
                acc32[54] = scaled_pair_1_55.x;
                acc32[55] = scaled_pair_1_55.y;
                float _tmem_load_30[4];
                tmem_ld_x4(&_tmem_load_30[0], tmem_base + 56);
                float2 weight_pair_1_30 = make_float2(weights_598[0], weights_598[0]);
                float2 cached_pair_1_108 = make_float2(_tmem_load_30[0], _tmem_load_30[1]);
                float2 acc_pair_1_164 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1_108 = fma_f32x2_rn_ftz(weight_pair_1_30, cached_pair_1_108, acc_pair_1_164);
                acc32[52] = updated_pair_1_108.x;
                acc32[53] = updated_pair_1_108.y;
                float2 cached_pair_1_109 = make_float2(_tmem_load_30[2], _tmem_load_30[3]);
                float2 acc_pair_1_165 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1_109 = fma_f32x2_rn_ftz(weight_pair_1_30, cached_pair_1_109, acc_pair_1_165);
                acc32[54] = updated_pair_1_109.x;
                acc32[55] = updated_pair_1_109.y;
                float _tmem_load_31[4];
                tmem_ld_x4(&_tmem_load_31[0], tmem_base + 64 + 56);
                float2 weight_pair_1_31 = make_float2(weights_598[1], weights_598[1]);
                float2 cached_pair_1_110 = make_float2(_tmem_load_31[0], _tmem_load_31[1]);
                float2 acc_pair_1_166 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1_110 = fma_f32x2_rn_ftz(weight_pair_1_31, cached_pair_1_110, acc_pair_1_166);
                acc32[52] = updated_pair_1_110.x;
                acc32[53] = updated_pair_1_110.y;
                float2 cached_pair_1_111 = make_float2(_tmem_load_31[2], _tmem_load_31[3]);
                float2 acc_pair_1_167 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1_111 = fma_f32x2_rn_ftz(weight_pair_1_31, cached_pair_1_111, acc_pair_1_167);
                acc32[54] = updated_pair_1_111.x;
                acc32[55] = updated_pair_1_111.y;
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
                int hidden_base_1_24 = thread_in_group * 8;
                unsigned int smem_regs_43[4];
                #pragma unroll
                for (int _lr_32 = 0; _lr_32 < 4; _lr_32++) {
                    smem_regs_43[_lr_32] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_1_24 / 2) + (unsigned int)_lr_32];
                }
                float values_1_32[8];
                float _unpack_bf16_75[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_75[_pair * 2])[0]), "=f"((&_unpack_bf16_75[_pair * 2])[1])
                        : "r"(smem_regs_43[_pair]));
                }
                values_1_32[0] = _unpack_bf16_75[0];
                values_1_32[1] = _unpack_bf16_75[1];
                values_1_32[2] = _unpack_bf16_75[2];
                values_1_32[3] = _unpack_bf16_75[3];
                values_1_32[4] = _unpack_bf16_75[4];
                values_1_32[5] = _unpack_bf16_75[5];
                values_1_32[6] = _unpack_bf16_75[6];
                values_1_32[7] = _unpack_bf16_75[7];
                tmem_st_x8_f32(tmem_base, values_1_32);
                float2 value_pair_1_112 = make_float2(values_1_32[0], values_1_32[1]);
                float2 q_pair_35_140 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_112, value_pair_1_112, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_112, q_pair_35_140, local_dot_853[0]);
                float2 value_pair_1_113 = make_float2(values_1_32[2], values_1_32[3]);
                float2 q_pair_35_141 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_113, value_pair_1_113, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_113, q_pair_35_141, local_dot_853[0]);
                float2 value_pair_1_114 = make_float2(values_1_32[4], values_1_32[5]);
                float2 q_pair_35_142 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_114, value_pair_1_114, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_114, q_pair_35_142, local_dot_853[0]);
                float2 value_pair_1_115 = make_float2(values_1_32[6], values_1_32[7]);
                float2 q_pair_35_143 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_115, value_pair_1_115, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_115, q_pair_35_143, local_dot_853[0]);
                unsigned int smem_regs_44[4];
                #pragma unroll
                for (int _lr_33 = 0; _lr_33 < 4; _lr_33++) {
                    smem_regs_44[_lr_33] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_24 / 2) + (unsigned int)_lr_33];
                }
                float values_2_0[8];
                unsigned int smem_regs_45[4];
                #pragma unroll
                for (int _lr_34 = 0; _lr_34 < 4; _lr_34++) {
                    smem_regs_45[_lr_34] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1_24 / 2) + (unsigned int)_lr_34];
                }
                __nv_bfloat162 prefix_pair_1_0 = __as_bf16x2(smem_regs_44[0]);
                __nv_bfloat162 delta_pair_1_0 = __as_bf16x2(smem_regs_45[0]);
                __nv_bfloat162 updated_bf16_pair_1_0 = prefix_pair_1_0 + delta_pair_1_0;
                values_2_0[0] = (float)updated_bf16_pair_1_0.x;
                values_2_0[1] = (float)updated_bf16_pair_1_0.y;
                __nv_bfloat162 prefix_pair_1_1 = __as_bf16x2(smem_regs_44[1]);
                __nv_bfloat162 delta_pair_1_1 = __as_bf16x2(smem_regs_45[1]);
                __nv_bfloat162 updated_bf16_pair_1_1 = prefix_pair_1_1 + delta_pair_1_1;
                values_2_0[2] = (float)updated_bf16_pair_1_1.x;
                values_2_0[3] = (float)updated_bf16_pair_1_1.y;
                __nv_bfloat162 prefix_pair_1_2 = __as_bf16x2(smem_regs_44[2]);
                __nv_bfloat162 delta_pair_1_2 = __as_bf16x2(smem_regs_45[2]);
                __nv_bfloat162 updated_bf16_pair_1_2 = prefix_pair_1_2 + delta_pair_1_2;
                values_2_0[4] = (float)updated_bf16_pair_1_2.x;
                values_2_0[5] = (float)updated_bf16_pair_1_2.y;
                __nv_bfloat162 prefix_pair_1_3 = __as_bf16x2(smem_regs_44[3]);
                __nv_bfloat162 delta_pair_1_3 = __as_bf16x2(smem_regs_45[3]);
                __nv_bfloat162 updated_bf16_pair_1_3 = prefix_pair_1_3 + delta_pair_1_3;
                values_2_0[6] = (float)updated_bf16_pair_1_3.x;
                values_2_0[7] = (float)updated_bf16_pair_1_3.y;
                unsigned int _packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_0[_lp*2 + 0], values_2_0[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_0[_lp*2 + 2], values_2_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_0[_lp*2 + 4], values_2_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_0[_lp*2 + 6], values_2_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1_24))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                tmem_st_x8_f32(tmem_base + 64, values_2_0);
                float2 value_pair_1_116 = make_float2(values_2_0[0], values_2_0[1]);
                float2 q_pair_35_144 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_116, value_pair_1_116, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_116, q_pair_35_144, local_dot_853[1]);
                float2 value_pair_1_117 = make_float2(values_2_0[2], values_2_0[3]);
                float2 q_pair_35_145 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_117, value_pair_1_117, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_117, q_pair_35_145, local_dot_853[1]);
                float2 value_pair_1_118 = make_float2(values_2_0[4], values_2_0[5]);
                float2 q_pair_35_146 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_118, value_pair_1_118, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_118, q_pair_35_146, local_dot_853[1]);
                float2 value_pair_1_119 = make_float2(values_2_0[6], values_2_0[7]);
                float2 q_pair_35_147 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_119, value_pair_1_119, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_119, q_pair_35_147, local_dot_853[1]);
                int hidden_base_2_0 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_46[4];
                #pragma unroll
                for (int _lr_35 = 0; _lr_35 < 4; _lr_35++) {
                    smem_regs_46[_lr_35] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_2_0 / 2) + (unsigned int)_lr_35];
                }
                float values_3_0[8];
                float _unpack_bf16_48_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_48_0[_pair * 2])[0]), "=f"((&_unpack_bf16_48_0[_pair * 2])[1])
                        : "r"(smem_regs_46[_pair]));
                }
                values_3_0[0] = _unpack_bf16_48_0[0];
                values_3_0[1] = _unpack_bf16_48_0[1];
                values_3_0[2] = _unpack_bf16_48_0[2];
                values_3_0[3] = _unpack_bf16_48_0[3];
                values_3_0[4] = _unpack_bf16_48_0[4];
                values_3_0[5] = _unpack_bf16_48_0[5];
                values_3_0[6] = _unpack_bf16_48_0[6];
                values_3_0[7] = _unpack_bf16_48_0[7];
                tmem_st_x8_f32(tmem_base + 8, values_3_0);
                float2 value_pair_1_120 = make_float2(values_3_0[0], values_3_0[1]);
                float2 q_pair_35_148 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_120, value_pair_1_120, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_120, q_pair_35_148, local_dot_853[0]);
                float2 value_pair_1_121 = make_float2(values_3_0[2], values_3_0[3]);
                float2 q_pair_35_149 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_121, value_pair_1_121, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_121, q_pair_35_149, local_dot_853[0]);
                float2 value_pair_1_122 = make_float2(values_3_0[4], values_3_0[5]);
                float2 q_pair_35_150 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_122, value_pair_1_122, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_122, q_pair_35_150, local_dot_853[0]);
                float2 value_pair_1_123 = make_float2(values_3_0[6], values_3_0[7]);
                float2 q_pair_35_151 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_123, value_pair_1_123, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_123, q_pair_35_151, local_dot_853[0]);
                unsigned int smem_regs_47[4];
                #pragma unroll
                for (int _lr_36 = 0; _lr_36 < 4; _lr_36++) {
                    smem_regs_47[_lr_36] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_2_0 / 2) + (unsigned int)_lr_36];
                }
                float values_4_0[8];
                unsigned int smem_regs_48[4];
                #pragma unroll
                for (int _lr_37 = 0; _lr_37 < 4; _lr_37++) {
                    smem_regs_48[_lr_37] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_2_0 / 2) + (unsigned int)_lr_37];
                }
                __nv_bfloat162 prefix_pair_1_4 = __as_bf16x2(smem_regs_47[0]);
                __nv_bfloat162 delta_pair_1_4 = __as_bf16x2(smem_regs_48[0]);
                __nv_bfloat162 updated_bf16_pair_1_4 = prefix_pair_1_4 + delta_pair_1_4;
                values_4_0[0] = (float)updated_bf16_pair_1_4.x;
                values_4_0[1] = (float)updated_bf16_pair_1_4.y;
                __nv_bfloat162 prefix_pair_1_5 = __as_bf16x2(smem_regs_47[1]);
                __nv_bfloat162 delta_pair_1_5 = __as_bf16x2(smem_regs_48[1]);
                __nv_bfloat162 updated_bf16_pair_1_5 = prefix_pair_1_5 + delta_pair_1_5;
                values_4_0[2] = (float)updated_bf16_pair_1_5.x;
                values_4_0[3] = (float)updated_bf16_pair_1_5.y;
                __nv_bfloat162 prefix_pair_1_6 = __as_bf16x2(smem_regs_47[2]);
                __nv_bfloat162 delta_pair_1_6 = __as_bf16x2(smem_regs_48[2]);
                __nv_bfloat162 updated_bf16_pair_1_6 = prefix_pair_1_6 + delta_pair_1_6;
                values_4_0[4] = (float)updated_bf16_pair_1_6.x;
                values_4_0[5] = (float)updated_bf16_pair_1_6.y;
                __nv_bfloat162 prefix_pair_1_7 = __as_bf16x2(smem_regs_47[3]);
                __nv_bfloat162 delta_pair_1_7 = __as_bf16x2(smem_regs_48[3]);
                __nv_bfloat162 updated_bf16_pair_1_7 = prefix_pair_1_7 + delta_pair_1_7;
                values_4_0[6] = (float)updated_bf16_pair_1_7.x;
                values_4_0[7] = (float)updated_bf16_pair_1_7.y;
                unsigned int _packed_16_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_0[_lp*2 + 0], values_4_0[_lp*2+1 + 0]));
                    _packed_16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_0[_lp*2 + 2], values_4_0[_lp*2+1 + 2]));
                    _packed_16_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_0[_lp*2 + 4], values_4_0[_lp*2+1 + 4]));
                    _packed_16_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_0[_lp*2 + 6], values_4_0[_lp*2+1 + 6]));
                    _packed_16_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_2_0))[0] = reinterpret_cast<int4*>(_packed_16_0)[0];
                tmem_st_x8_f32(tmem_base + 64 + 8, values_4_0);
                float2 value_pair_1_124 = make_float2(values_4_0[0], values_4_0[1]);
                float2 q_pair_35_152 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_124, value_pair_1_124, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_124, q_pair_35_152, local_dot_853[1]);
                float2 value_pair_1_125 = make_float2(values_4_0[2], values_4_0[3]);
                float2 q_pair_35_153 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_125, value_pair_1_125, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_125, q_pair_35_153, local_dot_853[1]);
                float2 value_pair_1_126 = make_float2(values_4_0[4], values_4_0[5]);
                float2 q_pair_35_154 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_126, value_pair_1_126, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_126, q_pair_35_154, local_dot_853[1]);
                float2 value_pair_1_127 = make_float2(values_4_0[6], values_4_0[7]);
                float2 q_pair_35_155 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_127, value_pair_1_127, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_127, q_pair_35_155, local_dot_853[1]);
                int hidden_base_4_0 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_49[4];
                #pragma unroll
                for (int _lr_38 = 0; _lr_38 < 4; _lr_38++) {
                    smem_regs_49[_lr_38] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_4_0 / 2) + (unsigned int)_lr_38];
                }
                float values_5_0[8];
                float _unpack_bf16_49_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_49_0[_pair * 2])[0]), "=f"((&_unpack_bf16_49_0[_pair * 2])[1])
                        : "r"(smem_regs_49[_pair]));
                }
                values_5_0[0] = _unpack_bf16_49_0[0];
                values_5_0[1] = _unpack_bf16_49_0[1];
                values_5_0[2] = _unpack_bf16_49_0[2];
                values_5_0[3] = _unpack_bf16_49_0[3];
                values_5_0[4] = _unpack_bf16_49_0[4];
                values_5_0[5] = _unpack_bf16_49_0[5];
                values_5_0[6] = _unpack_bf16_49_0[6];
                values_5_0[7] = _unpack_bf16_49_0[7];
                tmem_st_x8_f32(tmem_base + 16, values_5_0);
                float2 value_pair_1_128 = make_float2(values_5_0[0], values_5_0[1]);
                float2 q_pair_35_156 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_128, value_pair_1_128, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_128, q_pair_35_156, local_dot_853[0]);
                float2 value_pair_1_129 = make_float2(values_5_0[2], values_5_0[3]);
                float2 q_pair_35_157 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_129, value_pair_1_129, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_129, q_pair_35_157, local_dot_853[0]);
                float2 value_pair_1_130 = make_float2(values_5_0[4], values_5_0[5]);
                float2 q_pair_35_158 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_130, value_pair_1_130, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_130, q_pair_35_158, local_dot_853[0]);
                float2 value_pair_1_131 = make_float2(values_5_0[6], values_5_0[7]);
                float2 q_pair_35_159 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_131, value_pair_1_131, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_131, q_pair_35_159, local_dot_853[0]);
                unsigned int smem_regs_50[4];
                #pragma unroll
                for (int _lr_39 = 0; _lr_39 < 4; _lr_39++) {
                    smem_regs_50[_lr_39] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_4_0 / 2) + (unsigned int)_lr_39];
                }
                float values_6_0[8];
                unsigned int smem_regs_51[4];
                #pragma unroll
                for (int _lr_40 = 0; _lr_40 < 4; _lr_40++) {
                    smem_regs_51[_lr_40] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_4_0 / 2) + (unsigned int)_lr_40];
                }
                __nv_bfloat162 prefix_pair_1_8 = __as_bf16x2(smem_regs_50[0]);
                __nv_bfloat162 delta_pair_1_8 = __as_bf16x2(smem_regs_51[0]);
                __nv_bfloat162 updated_bf16_pair_1_8 = prefix_pair_1_8 + delta_pair_1_8;
                values_6_0[0] = (float)updated_bf16_pair_1_8.x;
                values_6_0[1] = (float)updated_bf16_pair_1_8.y;
                __nv_bfloat162 prefix_pair_1_9 = __as_bf16x2(smem_regs_50[1]);
                __nv_bfloat162 delta_pair_1_9 = __as_bf16x2(smem_regs_51[1]);
                __nv_bfloat162 updated_bf16_pair_1_9 = prefix_pair_1_9 + delta_pair_1_9;
                values_6_0[2] = (float)updated_bf16_pair_1_9.x;
                values_6_0[3] = (float)updated_bf16_pair_1_9.y;
                __nv_bfloat162 prefix_pair_1_10 = __as_bf16x2(smem_regs_50[2]);
                __nv_bfloat162 delta_pair_1_10 = __as_bf16x2(smem_regs_51[2]);
                __nv_bfloat162 updated_bf16_pair_1_10 = prefix_pair_1_10 + delta_pair_1_10;
                values_6_0[4] = (float)updated_bf16_pair_1_10.x;
                values_6_0[5] = (float)updated_bf16_pair_1_10.y;
                __nv_bfloat162 prefix_pair_1_11 = __as_bf16x2(smem_regs_50[3]);
                __nv_bfloat162 delta_pair_1_11 = __as_bf16x2(smem_regs_51[3]);
                __nv_bfloat162 updated_bf16_pair_1_11 = prefix_pair_1_11 + delta_pair_1_11;
                values_6_0[6] = (float)updated_bf16_pair_1_11.x;
                values_6_0[7] = (float)updated_bf16_pair_1_11.y;
                unsigned int _packed_17_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_0[_lp*2 + 0], values_6_0[_lp*2+1 + 0]));
                    _packed_17_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_0[_lp*2 + 2], values_6_0[_lp*2+1 + 2]));
                    _packed_17_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_0[_lp*2 + 4], values_6_0[_lp*2+1 + 4]));
                    _packed_17_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_0[_lp*2 + 6], values_6_0[_lp*2+1 + 6]));
                    _packed_17_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_4_0))[0] = reinterpret_cast<int4*>(_packed_17_0)[0];
                tmem_st_x8_f32(tmem_base + 64 + 16, values_6_0);
                float2 value_pair_1_132 = make_float2(values_6_0[0], values_6_0[1]);
                float2 q_pair_35_160 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_132, value_pair_1_132, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_132, q_pair_35_160, local_dot_853[1]);
                float2 value_pair_1_133 = make_float2(values_6_0[2], values_6_0[3]);
                float2 q_pair_35_161 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_133, value_pair_1_133, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_133, q_pair_35_161, local_dot_853[1]);
                float2 value_pair_1_134 = make_float2(values_6_0[4], values_6_0[5]);
                float2 q_pair_35_162 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_134, value_pair_1_134, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_134, q_pair_35_162, local_dot_853[1]);
                float2 value_pair_1_135 = make_float2(values_6_0[6], values_6_0[7]);
                float2 q_pair_35_163 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_135, value_pair_1_135, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_135, q_pair_35_163, local_dot_853[1]);
                int hidden_base_5_0 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_52[2];
                #pragma unroll
                for (int _lr_41 = 0; _lr_41 < 2; _lr_41++) {
                    smem_regs_52[_lr_41] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_5_0 / 2) + (unsigned int)_lr_41];
                }
                float values_7_0[4];
                float _unpack_bf16_50_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_50_0[_pair * 2])[0]), "=f"((&_unpack_bf16_50_0[_pair * 2])[1])
                        : "r"(smem_regs_52[_pair]));
                }
                values_7_0[0] = _unpack_bf16_50_0[0];
                values_7_0[1] = _unpack_bf16_50_0[1];
                values_7_0[2] = _unpack_bf16_50_0[2];
                values_7_0[3] = _unpack_bf16_50_0[3];
                tmem_st_x4_f32(tmem_base + 24, values_7_0);
                float2 value_pair_1_136 = make_float2(values_7_0[0], values_7_0[1]);
                float2 q_pair_35_164 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_136, value_pair_1_136, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_136, q_pair_35_164, local_dot_853[0]);
                float2 value_pair_1_137 = make_float2(values_7_0[2], values_7_0[3]);
                float2 q_pair_35_165 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_1_137, value_pair_1_137, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_1_137, q_pair_35_165, local_dot_853[0]);
                unsigned int smem_regs_53[2];
                #pragma unroll
                for (int _lr_42 = 0; _lr_42 < 2; _lr_42++) {
                    smem_regs_53[_lr_42] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_5_0 / 2) + (unsigned int)_lr_42];
                }
                float values_8_0[4];
                unsigned int smem_regs_54[2];
                #pragma unroll
                for (int _lr_43 = 0; _lr_43 < 2; _lr_43++) {
                    smem_regs_54[_lr_43] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_5_0 / 2) + (unsigned int)_lr_43];
                }
                __nv_bfloat162 prefix_pair_1_12 = __as_bf16x2(smem_regs_53[0]);
                __nv_bfloat162 delta_pair_1_12 = __as_bf16x2(smem_regs_54[0]);
                __nv_bfloat162 updated_bf16_pair_1_12 = prefix_pair_1_12 + delta_pair_1_12;
                values_8_0[0] = (float)updated_bf16_pair_1_12.x;
                values_8_0[1] = (float)updated_bf16_pair_1_12.y;
                __nv_bfloat162 prefix_pair_1_13 = __as_bf16x2(smem_regs_53[1]);
                __nv_bfloat162 delta_pair_1_13 = __as_bf16x2(smem_regs_54[1]);
                __nv_bfloat162 updated_bf16_pair_1_13 = prefix_pair_1_13 + delta_pair_1_13;
                values_8_0[2] = (float)updated_bf16_pair_1_13.x;
                values_8_0[3] = (float)updated_bf16_pair_1_13.y;
                unsigned int _packed_18_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_8_0[_lp*2 + 0], values_8_0[_lp*2+1 + 0]));
                    _packed_18_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_8_0[_lp*2 + 2], values_8_0[_lp*2+1 + 2]));
                    _packed_18_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(prefix + (token_1 * 7168 + hidden_base_5_0))[0] = reinterpret_cast<int2*>(_packed_18_0)[0];
                tmem_st_x4_f32(tmem_base + 64 + 24, values_8_0);
                float2 value_pair_1_138 = make_float2(values_8_0[0], values_8_0[1]);
                float2 q_pair_35_166 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_138, value_pair_1_138, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_138, q_pair_35_166, local_dot_853[1]);
                float2 value_pair_1_139 = make_float2(values_8_0[2], values_8_0[3]);
                float2 q_pair_35_167 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_1_139, value_pair_1_139, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_1_139, q_pair_35_167, local_dot_853[1]);
                int hidden_base_1_25 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_55[4];
                #pragma unroll
                for (int _lr_44 = 0; _lr_44 < 4; _lr_44++) {
                    smem_regs_55[_lr_44] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_1_25 / 2) + (unsigned int)_lr_44];
                }
                float values_1_33[8];
                float _unpack_bf16_76[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_76[_pair * 2])[0]), "=f"((&_unpack_bf16_76[_pair * 2])[1])
                        : "r"(smem_regs_55[_pair]));
                }
                values_1_33[0] = _unpack_bf16_76[0];
                values_1_33[1] = _unpack_bf16_76[1];
                values_1_33[2] = _unpack_bf16_76[2];
                values_1_33[3] = _unpack_bf16_76[3];
                values_1_33[4] = _unpack_bf16_76[4];
                values_1_33[5] = _unpack_bf16_76[5];
                values_1_33[6] = _unpack_bf16_76[6];
                values_1_33[7] = _unpack_bf16_76[7];
                tmem_st_x8_f32(tmem_base + 32, values_1_33);
                float2 value_pair_1_140 = make_float2(values_1_33[0], values_1_33[1]);
                float2 q_pair_35_168 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_140, value_pair_1_140, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_140, q_pair_35_168, local_dot_853[2]);
                float2 value_pair_1_141 = make_float2(values_1_33[2], values_1_33[3]);
                float2 q_pair_35_169 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_141, value_pair_1_141, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_141, q_pair_35_169, local_dot_853[2]);
                float2 value_pair_1_142 = make_float2(values_1_33[4], values_1_33[5]);
                float2 q_pair_35_170 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_142, value_pair_1_142, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_142, q_pair_35_170, local_dot_853[2]);
                float2 value_pair_1_143 = make_float2(values_1_33[6], values_1_33[7]);
                float2 q_pair_35_171 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_143, value_pair_1_143, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_143, q_pair_35_171, local_dot_853[2]);
                unsigned int smem_regs_56[4];
                #pragma unroll
                for (int _lr_45 = 0; _lr_45 < 4; _lr_45++) {
                    smem_regs_56[_lr_45] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_1_25 / 2) + (unsigned int)_lr_45];
                }
                float values_2_1[8];
                unsigned int smem_regs_57[4];
                #pragma unroll
                for (int _lr_46 = 0; _lr_46 < 4; _lr_46++) {
                    smem_regs_57[_lr_46] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1_25 / 2) + (unsigned int)_lr_46];
                }
                __nv_bfloat162 prefix_pair_1_14 = __as_bf16x2(smem_regs_56[0]);
                __nv_bfloat162 delta_pair_1_14 = __as_bf16x2(smem_regs_57[0]);
                __nv_bfloat162 updated_bf16_pair_1_14 = prefix_pair_1_14 + delta_pair_1_14;
                values_2_1[0] = (float)updated_bf16_pair_1_14.x;
                values_2_1[1] = (float)updated_bf16_pair_1_14.y;
                __nv_bfloat162 prefix_pair_1_15 = __as_bf16x2(smem_regs_56[1]);
                __nv_bfloat162 delta_pair_1_15 = __as_bf16x2(smem_regs_57[1]);
                __nv_bfloat162 updated_bf16_pair_1_15 = prefix_pair_1_15 + delta_pair_1_15;
                values_2_1[2] = (float)updated_bf16_pair_1_15.x;
                values_2_1[3] = (float)updated_bf16_pair_1_15.y;
                __nv_bfloat162 prefix_pair_1_16 = __as_bf16x2(smem_regs_56[2]);
                __nv_bfloat162 delta_pair_1_16 = __as_bf16x2(smem_regs_57[2]);
                __nv_bfloat162 updated_bf16_pair_1_16 = prefix_pair_1_16 + delta_pair_1_16;
                values_2_1[4] = (float)updated_bf16_pair_1_16.x;
                values_2_1[5] = (float)updated_bf16_pair_1_16.y;
                __nv_bfloat162 prefix_pair_1_17 = __as_bf16x2(smem_regs_56[3]);
                __nv_bfloat162 delta_pair_1_17 = __as_bf16x2(smem_regs_57[3]);
                __nv_bfloat162 updated_bf16_pair_1_17 = prefix_pair_1_17 + delta_pair_1_17;
                values_2_1[6] = (float)updated_bf16_pair_1_17.x;
                values_2_1[7] = (float)updated_bf16_pair_1_17.y;
                unsigned int _packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_1[_lp*2 + 0], values_2_1[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_1[_lp*2 + 2], values_2_1[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_1[_lp*2 + 4], values_2_1[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_2_1[_lp*2 + 6], values_2_1[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_1_25))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                tmem_st_x8_f32(tmem_base + 64 + 32, values_2_1);
                float2 value_pair_1_144 = make_float2(values_2_1[0], values_2_1[1]);
                float2 q_pair_35_172 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_144, value_pair_1_144, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_144, q_pair_35_172, local_dot_853[3]);
                float2 value_pair_1_145 = make_float2(values_2_1[2], values_2_1[3]);
                float2 q_pair_35_173 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_145, value_pair_1_145, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_145, q_pair_35_173, local_dot_853[3]);
                float2 value_pair_1_146 = make_float2(values_2_1[4], values_2_1[5]);
                float2 q_pair_35_174 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_146, value_pair_1_146, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_146, q_pair_35_174, local_dot_853[3]);
                float2 value_pair_1_147 = make_float2(values_2_1[6], values_2_1[7]);
                float2 q_pair_35_175 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_147, value_pair_1_147, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_147, q_pair_35_175, local_dot_853[3]);
                int hidden_base_2_1 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_58[4];
                #pragma unroll
                for (int _lr_47 = 0; _lr_47 < 4; _lr_47++) {
                    smem_regs_58[_lr_47] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_2_1 / 2) + (unsigned int)_lr_47];
                }
                float values_3_1[8];
                float _unpack_bf16_48_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_48_1[_pair * 2])[0]), "=f"((&_unpack_bf16_48_1[_pair * 2])[1])
                        : "r"(smem_regs_58[_pair]));
                }
                values_3_1[0] = _unpack_bf16_48_1[0];
                values_3_1[1] = _unpack_bf16_48_1[1];
                values_3_1[2] = _unpack_bf16_48_1[2];
                values_3_1[3] = _unpack_bf16_48_1[3];
                values_3_1[4] = _unpack_bf16_48_1[4];
                values_3_1[5] = _unpack_bf16_48_1[5];
                values_3_1[6] = _unpack_bf16_48_1[6];
                values_3_1[7] = _unpack_bf16_48_1[7];
                tmem_st_x8_f32(tmem_base + 40, values_3_1);
                float2 value_pair_1_148 = make_float2(values_3_1[0], values_3_1[1]);
                float2 q_pair_35_176 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_148, value_pair_1_148, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_148, q_pair_35_176, local_dot_853[2]);
                float2 value_pair_1_149 = make_float2(values_3_1[2], values_3_1[3]);
                float2 q_pair_35_177 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_149, value_pair_1_149, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_149, q_pair_35_177, local_dot_853[2]);
                float2 value_pair_1_150 = make_float2(values_3_1[4], values_3_1[5]);
                float2 q_pair_35_178 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_150, value_pair_1_150, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_150, q_pair_35_178, local_dot_853[2]);
                float2 value_pair_1_151 = make_float2(values_3_1[6], values_3_1[7]);
                float2 q_pair_35_179 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_151, value_pair_1_151, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_151, q_pair_35_179, local_dot_853[2]);
                unsigned int smem_regs_59[4];
                #pragma unroll
                for (int _lr_48 = 0; _lr_48 < 4; _lr_48++) {
                    smem_regs_59[_lr_48] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_2_1 / 2) + (unsigned int)_lr_48];
                }
                float values_4_1[8];
                unsigned int smem_regs_60[4];
                #pragma unroll
                for (int _lr_49 = 0; _lr_49 < 4; _lr_49++) {
                    smem_regs_60[_lr_49] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_2_1 / 2) + (unsigned int)_lr_49];
                }
                __nv_bfloat162 prefix_pair_1_18 = __as_bf16x2(smem_regs_59[0]);
                __nv_bfloat162 delta_pair_1_18 = __as_bf16x2(smem_regs_60[0]);
                __nv_bfloat162 updated_bf16_pair_1_18 = prefix_pair_1_18 + delta_pair_1_18;
                values_4_1[0] = (float)updated_bf16_pair_1_18.x;
                values_4_1[1] = (float)updated_bf16_pair_1_18.y;
                __nv_bfloat162 prefix_pair_1_19 = __as_bf16x2(smem_regs_59[1]);
                __nv_bfloat162 delta_pair_1_19 = __as_bf16x2(smem_regs_60[1]);
                __nv_bfloat162 updated_bf16_pair_1_19 = prefix_pair_1_19 + delta_pair_1_19;
                values_4_1[2] = (float)updated_bf16_pair_1_19.x;
                values_4_1[3] = (float)updated_bf16_pair_1_19.y;
                __nv_bfloat162 prefix_pair_1_20 = __as_bf16x2(smem_regs_59[2]);
                __nv_bfloat162 delta_pair_1_20 = __as_bf16x2(smem_regs_60[2]);
                __nv_bfloat162 updated_bf16_pair_1_20 = prefix_pair_1_20 + delta_pair_1_20;
                values_4_1[4] = (float)updated_bf16_pair_1_20.x;
                values_4_1[5] = (float)updated_bf16_pair_1_20.y;
                __nv_bfloat162 prefix_pair_1_21 = __as_bf16x2(smem_regs_59[3]);
                __nv_bfloat162 delta_pair_1_21 = __as_bf16x2(smem_regs_60[3]);
                __nv_bfloat162 updated_bf16_pair_1_21 = prefix_pair_1_21 + delta_pair_1_21;
                values_4_1[6] = (float)updated_bf16_pair_1_21.x;
                values_4_1[7] = (float)updated_bf16_pair_1_21.y;
                unsigned int _packed_16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_1[_lp*2 + 0], values_4_1[_lp*2+1 + 0]));
                    _packed_16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_1[_lp*2 + 2], values_4_1[_lp*2+1 + 2]));
                    _packed_16_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_1[_lp*2 + 4], values_4_1[_lp*2+1 + 4]));
                    _packed_16_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_4_1[_lp*2 + 6], values_4_1[_lp*2+1 + 6]));
                    _packed_16_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_2_1))[0] = reinterpret_cast<int4*>(_packed_16_1)[0];
                tmem_st_x8_f32(tmem_base + 64 + 40, values_4_1);
                float2 value_pair_1_152 = make_float2(values_4_1[0], values_4_1[1]);
                float2 q_pair_35_180 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_152, value_pair_1_152, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_152, q_pair_35_180, local_dot_853[3]);
                float2 value_pair_1_153 = make_float2(values_4_1[2], values_4_1[3]);
                float2 q_pair_35_181 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_153, value_pair_1_153, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_153, q_pair_35_181, local_dot_853[3]);
                float2 value_pair_1_154 = make_float2(values_4_1[4], values_4_1[5]);
                float2 q_pair_35_182 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_154, value_pair_1_154, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_154, q_pair_35_182, local_dot_853[3]);
                float2 value_pair_1_155 = make_float2(values_4_1[6], values_4_1[7]);
                float2 q_pair_35_183 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_155, value_pair_1_155, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_155, q_pair_35_183, local_dot_853[3]);
                int hidden_base_4_1 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_61[4];
                #pragma unroll
                for (int _lr_50 = 0; _lr_50 < 4; _lr_50++) {
                    smem_regs_61[_lr_50] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_4_1 / 2) + (unsigned int)_lr_50];
                }
                float values_5_1[8];
                float _unpack_bf16_49_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_49_1[_pair * 2])[0]), "=f"((&_unpack_bf16_49_1[_pair * 2])[1])
                        : "r"(smem_regs_61[_pair]));
                }
                values_5_1[0] = _unpack_bf16_49_1[0];
                values_5_1[1] = _unpack_bf16_49_1[1];
                values_5_1[2] = _unpack_bf16_49_1[2];
                values_5_1[3] = _unpack_bf16_49_1[3];
                values_5_1[4] = _unpack_bf16_49_1[4];
                values_5_1[5] = _unpack_bf16_49_1[5];
                values_5_1[6] = _unpack_bf16_49_1[6];
                values_5_1[7] = _unpack_bf16_49_1[7];
                tmem_st_x8_f32(tmem_base + 48, values_5_1);
                float2 value_pair_1_156 = make_float2(values_5_1[0], values_5_1[1]);
                float2 q_pair_35_184 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_156, value_pair_1_156, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_156, q_pair_35_184, local_dot_853[2]);
                float2 value_pair_1_157 = make_float2(values_5_1[2], values_5_1[3]);
                float2 q_pair_35_185 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_157, value_pair_1_157, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_157, q_pair_35_185, local_dot_853[2]);
                float2 value_pair_1_158 = make_float2(values_5_1[4], values_5_1[5]);
                float2 q_pair_35_186 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_158, value_pair_1_158, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_158, q_pair_35_186, local_dot_853[2]);
                float2 value_pair_1_159 = make_float2(values_5_1[6], values_5_1[7]);
                float2 q_pair_35_187 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_159, value_pair_1_159, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_159, q_pair_35_187, local_dot_853[2]);
                unsigned int smem_regs_62[4];
                #pragma unroll
                for (int _lr_51 = 0; _lr_51 < 4; _lr_51++) {
                    smem_regs_62[_lr_51] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_4_1 / 2) + (unsigned int)_lr_51];
                }
                float values_6_1[8];
                unsigned int smem_regs_63[4];
                #pragma unroll
                for (int _lr_52 = 0; _lr_52 < 4; _lr_52++) {
                    smem_regs_63[_lr_52] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_4_1 / 2) + (unsigned int)_lr_52];
                }
                __nv_bfloat162 prefix_pair_1_22 = __as_bf16x2(smem_regs_62[0]);
                __nv_bfloat162 delta_pair_1_22 = __as_bf16x2(smem_regs_63[0]);
                __nv_bfloat162 updated_bf16_pair_1_22 = prefix_pair_1_22 + delta_pair_1_22;
                values_6_1[0] = (float)updated_bf16_pair_1_22.x;
                values_6_1[1] = (float)updated_bf16_pair_1_22.y;
                __nv_bfloat162 prefix_pair_1_23 = __as_bf16x2(smem_regs_62[1]);
                __nv_bfloat162 delta_pair_1_23 = __as_bf16x2(smem_regs_63[1]);
                __nv_bfloat162 updated_bf16_pair_1_23 = prefix_pair_1_23 + delta_pair_1_23;
                values_6_1[2] = (float)updated_bf16_pair_1_23.x;
                values_6_1[3] = (float)updated_bf16_pair_1_23.y;
                __nv_bfloat162 prefix_pair_1_24 = __as_bf16x2(smem_regs_62[2]);
                __nv_bfloat162 delta_pair_1_24 = __as_bf16x2(smem_regs_63[2]);
                __nv_bfloat162 updated_bf16_pair_1_24 = prefix_pair_1_24 + delta_pair_1_24;
                values_6_1[4] = (float)updated_bf16_pair_1_24.x;
                values_6_1[5] = (float)updated_bf16_pair_1_24.y;
                __nv_bfloat162 prefix_pair_1_25 = __as_bf16x2(smem_regs_62[3]);
                __nv_bfloat162 delta_pair_1_25 = __as_bf16x2(smem_regs_63[3]);
                __nv_bfloat162 updated_bf16_pair_1_25 = prefix_pair_1_25 + delta_pair_1_25;
                values_6_1[6] = (float)updated_bf16_pair_1_25.x;
                values_6_1[7] = (float)updated_bf16_pair_1_25.y;
                unsigned int _packed_17_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_1[_lp*2 + 0], values_6_1[_lp*2+1 + 0]));
                    _packed_17_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_1[_lp*2 + 2], values_6_1[_lp*2+1 + 2]));
                    _packed_17_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_1[_lp*2 + 4], values_6_1[_lp*2+1 + 4]));
                    _packed_17_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_6_1[_lp*2 + 6], values_6_1[_lp*2+1 + 6]));
                    _packed_17_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_4_1))[0] = reinterpret_cast<int4*>(_packed_17_1)[0];
                tmem_st_x8_f32(tmem_base + 64 + 48, values_6_1);
                float2 value_pair_1_160 = make_float2(values_6_1[0], values_6_1[1]);
                float2 q_pair_35_188 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_160, value_pair_1_160, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_160, q_pair_35_188, local_dot_853[3]);
                float2 value_pair_1_161 = make_float2(values_6_1[2], values_6_1[3]);
                float2 q_pair_35_189 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_161, value_pair_1_161, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_161, q_pair_35_189, local_dot_853[3]);
                float2 value_pair_1_162 = make_float2(values_6_1[4], values_6_1[5]);
                float2 q_pair_35_190 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_162, value_pair_1_162, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_162, q_pair_35_190, local_dot_853[3]);
                float2 value_pair_1_163 = make_float2(values_6_1[6], values_6_1[7]);
                float2 q_pair_35_191 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_163, value_pair_1_163, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_163, q_pair_35_191, local_dot_853[3]);
                int hidden_base_5_1 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_64[2];
                #pragma unroll
                for (int _lr_53 = 0; _lr_53 < 2; _lr_53++) {
                    smem_regs_64[_lr_53] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_5_1 / 2) + (unsigned int)_lr_53];
                }
                float values_7_1[4];
                float _unpack_bf16_50_1[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_50_1[_pair * 2])[0]), "=f"((&_unpack_bf16_50_1[_pair * 2])[1])
                        : "r"(smem_regs_64[_pair]));
                }
                values_7_1[0] = _unpack_bf16_50_1[0];
                values_7_1[1] = _unpack_bf16_50_1[1];
                values_7_1[2] = _unpack_bf16_50_1[2];
                values_7_1[3] = _unpack_bf16_50_1[3];
                tmem_st_x4_f32(tmem_base + 56, values_7_1);
                float2 value_pair_1_164 = make_float2(values_7_1[0], values_7_1[1]);
                float2 q_pair_35_192 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_164, value_pair_1_164, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_164, q_pair_35_192, local_dot_853[2]);
                float2 value_pair_1_165 = make_float2(values_7_1[2], values_7_1[3]);
                float2 q_pair_35_193 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1_165, value_pair_1_165, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1_165, q_pair_35_193, local_dot_853[2]);
                unsigned int smem_regs_65[2];
                #pragma unroll
                for (int _lr_54 = 0; _lr_54 < 2; _lr_54++) {
                    smem_regs_65[_lr_54] = source_words[(ring_stage_851 * 2 + 1) * 3584 + (unsigned int)(hidden_base_5_1 / 2) + (unsigned int)_lr_54];
                }
                float values_8_1[4];
                unsigned int smem_regs_66[2];
                #pragma unroll
                for (int _lr_55 = 0; _lr_55 < 2; _lr_55++) {
                    smem_regs_66[_lr_55] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_5_1 / 2) + (unsigned int)_lr_55];
                }
                __nv_bfloat162 prefix_pair_1_26 = __as_bf16x2(smem_regs_65[0]);
                __nv_bfloat162 delta_pair_1_26 = __as_bf16x2(smem_regs_66[0]);
                __nv_bfloat162 updated_bf16_pair_1_26 = prefix_pair_1_26 + delta_pair_1_26;
                values_8_1[0] = (float)updated_bf16_pair_1_26.x;
                values_8_1[1] = (float)updated_bf16_pair_1_26.y;
                __nv_bfloat162 prefix_pair_1_27 = __as_bf16x2(smem_regs_65[1]);
                __nv_bfloat162 delta_pair_1_27 = __as_bf16x2(smem_regs_66[1]);
                __nv_bfloat162 updated_bf16_pair_1_27 = prefix_pair_1_27 + delta_pair_1_27;
                values_8_1[2] = (float)updated_bf16_pair_1_27.x;
                values_8_1[3] = (float)updated_bf16_pair_1_27.y;
                unsigned int _packed_18_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_8_1[_lp*2 + 0], values_8_1[_lp*2+1 + 0]));
                    _packed_18_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_8_1[_lp*2 + 2], values_8_1[_lp*2+1 + 2]));
                    _packed_18_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(prefix + (token_1 * 7168 + hidden_base_5_1))[0] = reinterpret_cast<int2*>(_packed_18_1)[0];
                tmem_st_x4_f32(tmem_base + 64 + 56, values_8_1);
                float2 value_pair_1_166 = make_float2(values_8_1[0], values_8_1[1]);
                float2 q_pair_35_194 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_166, value_pair_1_166, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_166, q_pair_35_194, local_dot_853[3]);
                float2 value_pair_1_167 = make_float2(values_8_1[2], values_8_1[3]);
                float2 q_pair_35_195 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1_167, value_pair_1_167, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1_167, q_pair_35_195, local_dot_853[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_851) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1095 = stats_stream_base + 64;
                float2 reduce_pair_1_8 = make_float2(local_sum_sq_852[0].x + local_sum_sq_852[0].y, local_dot_853[0].x + local_dot_853[0].y);
                unsigned long long packed_pair_1_8 = 0;
                unsigned long long peer_packed_1_8 = 0;
                float2 peer_pair_1_8 = make_float2(0.0f, 0.0f);
                packed_pair_1_8 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_8)[0];
                unsigned long long _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_8, 16);
                peer_packed_1_8 = _shfl_xor_sync_136;
                peer_pair_1_8 = reinterpret_cast<float2*>(&peer_packed_1_8)[0];
                reduce_pair_1_8 = add_f32x2(reduce_pair_1_8, peer_pair_1_8);
                packed_pair_1_8 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_8)[0];
                unsigned long long _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_8, 8);
                peer_packed_1_8 = _shfl_xor_sync_137;
                peer_pair_1_8 = reinterpret_cast<float2*>(&peer_packed_1_8)[0];
                reduce_pair_1_8 = add_f32x2(reduce_pair_1_8, peer_pair_1_8);
                packed_pair_1_8 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_8)[0];
                unsigned long long _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_8, 4);
                peer_packed_1_8 = _shfl_xor_sync_138;
                peer_pair_1_8 = reinterpret_cast<float2*>(&peer_packed_1_8)[0];
                reduce_pair_1_8 = add_f32x2(reduce_pair_1_8, peer_pair_1_8);
                packed_pair_1_8 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_8)[0];
                unsigned long long _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_8, 2);
                peer_packed_1_8 = _shfl_xor_sync_139;
                peer_pair_1_8 = reinterpret_cast<float2*>(&peer_packed_1_8)[0];
                reduce_pair_1_8 = add_f32x2(reduce_pair_1_8, peer_pair_1_8);
                packed_pair_1_8 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_8)[0];
                unsigned long long _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_8, 1);
                peer_packed_1_8 = _shfl_xor_sync_140;
                peer_pair_1_8 = reinterpret_cast<float2*>(&peer_packed_1_8)[0];
                reduce_pair_1_8 = add_f32x2(reduce_pair_1_8, peer_pair_1_8);
                if (lane == 0) {
                    stats[slot_base_1095 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1_8.x;
                    stats[slot_base_1095 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1_8.y;
                }
                float2 reduce_pair_1_9 = make_float2(local_sum_sq_852[1].x + local_sum_sq_852[1].y, local_dot_853[1].x + local_dot_853[1].y);
                unsigned long long packed_pair_1_9 = 0;
                unsigned long long peer_packed_1_9 = 0;
                float2 peer_pair_1_9 = make_float2(0.0f, 0.0f);
                packed_pair_1_9 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_9)[0];
                unsigned long long _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_9, 16);
                peer_packed_1_9 = _shfl_xor_sync_141;
                peer_pair_1_9 = reinterpret_cast<float2*>(&peer_packed_1_9)[0];
                reduce_pair_1_9 = add_f32x2(reduce_pair_1_9, peer_pair_1_9);
                packed_pair_1_9 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_9)[0];
                unsigned long long _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_9, 8);
                peer_packed_1_9 = _shfl_xor_sync_142;
                peer_pair_1_9 = reinterpret_cast<float2*>(&peer_packed_1_9)[0];
                reduce_pair_1_9 = add_f32x2(reduce_pair_1_9, peer_pair_1_9);
                packed_pair_1_9 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_9)[0];
                unsigned long long _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_9, 4);
                peer_packed_1_9 = _shfl_xor_sync_143;
                peer_pair_1_9 = reinterpret_cast<float2*>(&peer_packed_1_9)[0];
                reduce_pair_1_9 = add_f32x2(reduce_pair_1_9, peer_pair_1_9);
                packed_pair_1_9 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_9)[0];
                unsigned long long _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_9, 2);
                peer_packed_1_9 = _shfl_xor_sync_144;
                peer_pair_1_9 = reinterpret_cast<float2*>(&peer_packed_1_9)[0];
                reduce_pair_1_9 = add_f32x2(reduce_pair_1_9, peer_pair_1_9);
                packed_pair_1_9 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_9)[0];
                unsigned long long _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_9, 1);
                peer_packed_1_9 = _shfl_xor_sync_145;
                peer_pair_1_9 = reinterpret_cast<float2*>(&peer_packed_1_9)[0];
                reduce_pair_1_9 = add_f32x2(reduce_pair_1_9, peer_pair_1_9);
                if (lane == 0) {
                    stats[slot_base_1095 + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_1_9.x;
                    stats[slot_base_1095 + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_1_9.y;
                }
                float2 reduce_pair_1_10 = make_float2(local_sum_sq_852[2].x + local_sum_sq_852[2].y, local_dot_853[2].x + local_dot_853[2].y);
                unsigned long long packed_pair_1_10 = 0;
                unsigned long long peer_packed_1_10 = 0;
                float2 peer_pair_1_10 = make_float2(0.0f, 0.0f);
                packed_pair_1_10 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_10)[0];
                unsigned long long _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_10, 16);
                peer_packed_1_10 = _shfl_xor_sync_146;
                peer_pair_1_10 = reinterpret_cast<float2*>(&peer_packed_1_10)[0];
                reduce_pair_1_10 = add_f32x2(reduce_pair_1_10, peer_pair_1_10);
                packed_pair_1_10 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_10)[0];
                unsigned long long _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_10, 8);
                peer_packed_1_10 = _shfl_xor_sync_147;
                peer_pair_1_10 = reinterpret_cast<float2*>(&peer_packed_1_10)[0];
                reduce_pair_1_10 = add_f32x2(reduce_pair_1_10, peer_pair_1_10);
                packed_pair_1_10 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_10)[0];
                unsigned long long _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_10, 4);
                peer_packed_1_10 = _shfl_xor_sync_148;
                peer_pair_1_10 = reinterpret_cast<float2*>(&peer_packed_1_10)[0];
                reduce_pair_1_10 = add_f32x2(reduce_pair_1_10, peer_pair_1_10);
                packed_pair_1_10 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_10)[0];
                unsigned long long _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_10, 2);
                peer_packed_1_10 = _shfl_xor_sync_149;
                peer_pair_1_10 = reinterpret_cast<float2*>(&peer_packed_1_10)[0];
                reduce_pair_1_10 = add_f32x2(reduce_pair_1_10, peer_pair_1_10);
                packed_pair_1_10 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_10)[0];
                unsigned long long _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_10, 1);
                peer_packed_1_10 = _shfl_xor_sync_150;
                peer_pair_1_10 = reinterpret_cast<float2*>(&peer_packed_1_10)[0];
                reduce_pair_1_10 = add_f32x2(reduce_pair_1_10, peer_pair_1_10);
                if (lane == 0) {
                    stats[slot_base_1095 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1_10.x;
                    stats[slot_base_1095 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1_10.y;
                }
                float2 reduce_pair_1_11 = make_float2(local_sum_sq_852[3].x + local_sum_sq_852[3].y, local_dot_853[3].x + local_dot_853[3].y);
                unsigned long long packed_pair_1_11 = 0;
                unsigned long long peer_packed_1_11 = 0;
                float2 peer_pair_1_11 = make_float2(0.0f, 0.0f);
                packed_pair_1_11 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_11)[0];
                unsigned long long _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_11, 16);
                peer_packed_1_11 = _shfl_xor_sync_151;
                peer_pair_1_11 = reinterpret_cast<float2*>(&peer_packed_1_11)[0];
                reduce_pair_1_11 = add_f32x2(reduce_pair_1_11, peer_pair_1_11);
                packed_pair_1_11 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_11)[0];
                unsigned long long _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_11, 8);
                peer_packed_1_11 = _shfl_xor_sync_152;
                peer_pair_1_11 = reinterpret_cast<float2*>(&peer_packed_1_11)[0];
                reduce_pair_1_11 = add_f32x2(reduce_pair_1_11, peer_pair_1_11);
                packed_pair_1_11 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_11)[0];
                unsigned long long _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_11, 4);
                peer_packed_1_11 = _shfl_xor_sync_153;
                peer_pair_1_11 = reinterpret_cast<float2*>(&peer_packed_1_11)[0];
                reduce_pair_1_11 = add_f32x2(reduce_pair_1_11, peer_pair_1_11);
                packed_pair_1_11 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_11)[0];
                unsigned long long _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_11, 2);
                peer_packed_1_11 = _shfl_xor_sync_154;
                peer_pair_1_11 = reinterpret_cast<float2*>(&peer_packed_1_11)[0];
                reduce_pair_1_11 = add_f32x2(reduce_pair_1_11, peer_pair_1_11);
                packed_pair_1_11 = reinterpret_cast<unsigned long long*>(&reduce_pair_1_11)[0];
                unsigned long long _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1_11, 1);
                peer_packed_1_11 = _shfl_xor_sync_155;
                peer_pair_1_11 = reinterpret_cast<float2*>(&peer_packed_1_11)[0];
                reduce_pair_1_11 = add_f32x2(reduce_pair_1_11, peer_pair_1_11);
                if (lane == 0) {
                    stats[slot_base_1095 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2)] = reduce_pair_1_11.x;
                    stats[slot_base_1095 + (unsigned int)(((4 + warp_in_group) * 2 + 1) * 2) + 1] = reduce_pair_1_11.y;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int stat_source_1112 = lane / 8;
                int stat_warp_1113 = lane % 8;
                float total_sum_sq_1114 = 0.0f;
                float total_dot_1115 = 0.0f;
                if (stat_source_1112 < 2) {
                    total_sum_sq_1114 = stats[slot_base_1095 + (unsigned int)((stat_warp_1113 * 2 + stat_source_1112) * 2)];
                    total_dot_1115 = stats[slot_base_1095 + (unsigned int)((stat_warp_1113 * 2 + stat_source_1112) * 2) + 1];
                }
                float _shfl_down_sync_156 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1114, 4, 8);
                total_sum_sq_1114 += _shfl_down_sync_156;
                float _shfl_down_sync_157 = __shfl_down_sync(0xFFFFFFFF, total_dot_1115, 4, 8);
                total_dot_1115 += _shfl_down_sync_157;
                float _shfl_down_sync_158 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1114, 2, 8);
                total_sum_sq_1114 += _shfl_down_sync_158;
                float _shfl_down_sync_159 = __shfl_down_sync(0xFFFFFFFF, total_dot_1115, 2, 8);
                total_dot_1115 += _shfl_down_sync_159;
                float _shfl_down_sync_160 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1114, 1, 8);
                total_sum_sq_1114 += _shfl_down_sync_160;
                float _shfl_down_sync_161 = __shfl_down_sync(0xFFFFFFFF, total_dot_1115, 1, 8);
                total_dot_1115 += _shfl_down_sync_161;
                float local_logit_1116 = 0.0f;
                if (stat_source_1112 < 2 && stat_warp_1113 == 0) {
                    float _rsqrt_162 = rsqrtf(total_sum_sq_1114 / 7168.0f + eps);
                    local_logit_1116 = total_dot_1115 * _rsqrt_162;
                }
                float logits_1117[2];
                logits_1117[0] = 0.0f;
                float _shfl_sync_163 = __shfl_sync(0xFFFFFFFF, local_logit_1116, 0);
                logits_1117[0] = _shfl_sync_163;
                logits_1117[1] = 0.0f;
                float _shfl_sync_164 = __shfl_sync(0xFFFFFFFF, local_logit_1116, 8);
                logits_1117[1] = _shfl_sync_164;
                float weights_1118[2];
                float new_max_1119 = 0.0f;
                float correction_1120 = 0.0f;
                float weight_sum_1121 = 0.0f;
                weights_1118[0] = 0.0f;
                weights_1118[1] = 0.0f;
                float chunk_max_1122 = -3.4028235e+38f;
                chunk_max_1122 = max_noftz(chunk_max_1122, logits_1117[0]);
                chunk_max_1122 = max_noftz(chunk_max_1122, logits_1117[1]);
                new_max_1119 = max_noftz(m_running, chunk_max_1122);
                float _exp2_165 = approx_exp2((m_running - new_max_1119) * 1.442695f);
                correction_1120 = _exp2_165;
                float _exp2_166 = approx_exp2((logits_1117[0] - new_max_1119) * 1.442695f);
                weights_1118[0] = _exp2_166;
                weight_sum_1121 += weights_1118[0];
                float _exp2_167 = approx_exp2((logits_1117[1] - new_max_1119) * 1.442695f);
                weights_1118[1] = _exp2_167;
                weight_sum_1121 += weights_1118[1];
                float2 correction_pair_1_16 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_168 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1_56;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_56) : "l"(*(const unsigned long long*)&acc_pair_1_168), "l"(*(const unsigned long long*)&correction_pair_1_16));
                acc32[0] = scaled_pair_1_56.x;
                acc32[1] = scaled_pair_1_56.y;
                float2 acc_pair_1_169 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1_57;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_57) : "l"(*(const unsigned long long*)&acc_pair_1_169), "l"(*(const unsigned long long*)&correction_pair_1_16));
                acc32[2] = scaled_pair_1_57.x;
                acc32[3] = scaled_pair_1_57.y;
                float2 acc_pair_1_170 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1_58;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_58) : "l"(*(const unsigned long long*)&acc_pair_1_170), "l"(*(const unsigned long long*)&correction_pair_1_16));
                acc32[4] = scaled_pair_1_58.x;
                acc32[5] = scaled_pair_1_58.y;
                float2 acc_pair_1_171 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1_59;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_59) : "l"(*(const unsigned long long*)&acc_pair_1_171), "l"(*(const unsigned long long*)&correction_pair_1_16));
                acc32[6] = scaled_pair_1_59.x;
                acc32[7] = scaled_pair_1_59.y;
                float _tmem_load_32[8];
                tmem_ld_x8(&_tmem_load_32[0], tmem_base);
                float2 weight_pair_1_32 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_112 = make_float2(_tmem_load_32[0], _tmem_load_32[1]);
                float2 acc_pair_1_172 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1_112 = fma_f32x2_rn_ftz(weight_pair_1_32, cached_pair_1_112, acc_pair_1_172);
                acc32[0] = updated_pair_1_112.x;
                acc32[1] = updated_pair_1_112.y;
                float2 cached_pair_1_113 = make_float2(_tmem_load_32[2], _tmem_load_32[3]);
                float2 acc_pair_1_173 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1_113 = fma_f32x2_rn_ftz(weight_pair_1_32, cached_pair_1_113, acc_pair_1_173);
                acc32[2] = updated_pair_1_113.x;
                acc32[3] = updated_pair_1_113.y;
                float2 cached_pair_1_114 = make_float2(_tmem_load_32[4], _tmem_load_32[5]);
                float2 acc_pair_1_174 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1_114 = fma_f32x2_rn_ftz(weight_pair_1_32, cached_pair_1_114, acc_pair_1_174);
                acc32[4] = updated_pair_1_114.x;
                acc32[5] = updated_pair_1_114.y;
                float2 cached_pair_1_115 = make_float2(_tmem_load_32[6], _tmem_load_32[7]);
                float2 acc_pair_1_175 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1_115 = fma_f32x2_rn_ftz(weight_pair_1_32, cached_pair_1_115, acc_pair_1_175);
                acc32[6] = updated_pair_1_115.x;
                acc32[7] = updated_pair_1_115.y;
                float _tmem_load_33[8];
                tmem_ld_x8(&_tmem_load_33[0], tmem_base + 64);
                float2 weight_pair_1_33 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_116 = make_float2(_tmem_load_33[0], _tmem_load_33[1]);
                float2 acc_pair_1_176 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1_116 = fma_f32x2_rn_ftz(weight_pair_1_33, cached_pair_1_116, acc_pair_1_176);
                acc32[0] = updated_pair_1_116.x;
                acc32[1] = updated_pair_1_116.y;
                float2 cached_pair_1_117 = make_float2(_tmem_load_33[2], _tmem_load_33[3]);
                float2 acc_pair_1_177 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1_117 = fma_f32x2_rn_ftz(weight_pair_1_33, cached_pair_1_117, acc_pair_1_177);
                acc32[2] = updated_pair_1_117.x;
                acc32[3] = updated_pair_1_117.y;
                float2 cached_pair_1_118 = make_float2(_tmem_load_33[4], _tmem_load_33[5]);
                float2 acc_pair_1_178 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1_118 = fma_f32x2_rn_ftz(weight_pair_1_33, cached_pair_1_118, acc_pair_1_178);
                acc32[4] = updated_pair_1_118.x;
                acc32[5] = updated_pair_1_118.y;
                float2 cached_pair_1_119 = make_float2(_tmem_load_33[6], _tmem_load_33[7]);
                float2 acc_pair_1_179 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1_119 = fma_f32x2_rn_ftz(weight_pair_1_33, cached_pair_1_119, acc_pair_1_179);
                acc32[6] = updated_pair_1_119.x;
                acc32[7] = updated_pair_1_119.y;
                float2 correction_pair_1_17 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_180 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1_60;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_60) : "l"(*(const unsigned long long*)&acc_pair_1_180), "l"(*(const unsigned long long*)&correction_pair_1_17));
                acc32[8] = scaled_pair_1_60.x;
                acc32[9] = scaled_pair_1_60.y;
                float2 acc_pair_1_181 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1_61;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_61) : "l"(*(const unsigned long long*)&acc_pair_1_181), "l"(*(const unsigned long long*)&correction_pair_1_17));
                acc32[10] = scaled_pair_1_61.x;
                acc32[11] = scaled_pair_1_61.y;
                float2 acc_pair_1_182 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1_62;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_62) : "l"(*(const unsigned long long*)&acc_pair_1_182), "l"(*(const unsigned long long*)&correction_pair_1_17));
                acc32[12] = scaled_pair_1_62.x;
                acc32[13] = scaled_pair_1_62.y;
                float2 acc_pair_1_183 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1_63;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_63) : "l"(*(const unsigned long long*)&acc_pair_1_183), "l"(*(const unsigned long long*)&correction_pair_1_17));
                acc32[14] = scaled_pair_1_63.x;
                acc32[15] = scaled_pair_1_63.y;
                float _tmem_load_34[8];
                tmem_ld_x8(&_tmem_load_34[0], tmem_base + 8);
                float2 weight_pair_1_34 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_120 = make_float2(_tmem_load_34[0], _tmem_load_34[1]);
                float2 acc_pair_1_184 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1_120 = fma_f32x2_rn_ftz(weight_pair_1_34, cached_pair_1_120, acc_pair_1_184);
                acc32[8] = updated_pair_1_120.x;
                acc32[9] = updated_pair_1_120.y;
                float2 cached_pair_1_121 = make_float2(_tmem_load_34[2], _tmem_load_34[3]);
                float2 acc_pair_1_185 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1_121 = fma_f32x2_rn_ftz(weight_pair_1_34, cached_pair_1_121, acc_pair_1_185);
                acc32[10] = updated_pair_1_121.x;
                acc32[11] = updated_pair_1_121.y;
                float2 cached_pair_1_122 = make_float2(_tmem_load_34[4], _tmem_load_34[5]);
                float2 acc_pair_1_186 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1_122 = fma_f32x2_rn_ftz(weight_pair_1_34, cached_pair_1_122, acc_pair_1_186);
                acc32[12] = updated_pair_1_122.x;
                acc32[13] = updated_pair_1_122.y;
                float2 cached_pair_1_123 = make_float2(_tmem_load_34[6], _tmem_load_34[7]);
                float2 acc_pair_1_187 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1_123 = fma_f32x2_rn_ftz(weight_pair_1_34, cached_pair_1_123, acc_pair_1_187);
                acc32[14] = updated_pair_1_123.x;
                acc32[15] = updated_pair_1_123.y;
                float _tmem_load_35[8];
                tmem_ld_x8(&_tmem_load_35[0], tmem_base + 64 + 8);
                float2 weight_pair_1_35 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_124 = make_float2(_tmem_load_35[0], _tmem_load_35[1]);
                float2 acc_pair_1_188 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1_124 = fma_f32x2_rn_ftz(weight_pair_1_35, cached_pair_1_124, acc_pair_1_188);
                acc32[8] = updated_pair_1_124.x;
                acc32[9] = updated_pair_1_124.y;
                float2 cached_pair_1_125 = make_float2(_tmem_load_35[2], _tmem_load_35[3]);
                float2 acc_pair_1_189 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1_125 = fma_f32x2_rn_ftz(weight_pair_1_35, cached_pair_1_125, acc_pair_1_189);
                acc32[10] = updated_pair_1_125.x;
                acc32[11] = updated_pair_1_125.y;
                float2 cached_pair_1_126 = make_float2(_tmem_load_35[4], _tmem_load_35[5]);
                float2 acc_pair_1_190 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1_126 = fma_f32x2_rn_ftz(weight_pair_1_35, cached_pair_1_126, acc_pair_1_190);
                acc32[12] = updated_pair_1_126.x;
                acc32[13] = updated_pair_1_126.y;
                float2 cached_pair_1_127 = make_float2(_tmem_load_35[6], _tmem_load_35[7]);
                float2 acc_pair_1_191 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1_127 = fma_f32x2_rn_ftz(weight_pair_1_35, cached_pair_1_127, acc_pair_1_191);
                acc32[14] = updated_pair_1_127.x;
                acc32[15] = updated_pair_1_127.y;
                float2 correction_pair_1_18 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_192 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1_64;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_64) : "l"(*(const unsigned long long*)&acc_pair_1_192), "l"(*(const unsigned long long*)&correction_pair_1_18));
                acc32[16] = scaled_pair_1_64.x;
                acc32[17] = scaled_pair_1_64.y;
                float2 acc_pair_1_193 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1_65;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_65) : "l"(*(const unsigned long long*)&acc_pair_1_193), "l"(*(const unsigned long long*)&correction_pair_1_18));
                acc32[18] = scaled_pair_1_65.x;
                acc32[19] = scaled_pair_1_65.y;
                float2 acc_pair_1_194 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1_66;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_66) : "l"(*(const unsigned long long*)&acc_pair_1_194), "l"(*(const unsigned long long*)&correction_pair_1_18));
                acc32[20] = scaled_pair_1_66.x;
                acc32[21] = scaled_pair_1_66.y;
                float2 acc_pair_1_195 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1_67;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_67) : "l"(*(const unsigned long long*)&acc_pair_1_195), "l"(*(const unsigned long long*)&correction_pair_1_18));
                acc32[22] = scaled_pair_1_67.x;
                acc32[23] = scaled_pair_1_67.y;
                float _tmem_load_36[8];
                tmem_ld_x8(&_tmem_load_36[0], tmem_base + 16);
                float2 weight_pair_1_36 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_128 = make_float2(_tmem_load_36[0], _tmem_load_36[1]);
                float2 acc_pair_1_196 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1_128 = fma_f32x2_rn_ftz(weight_pair_1_36, cached_pair_1_128, acc_pair_1_196);
                acc32[16] = updated_pair_1_128.x;
                acc32[17] = updated_pair_1_128.y;
                float2 cached_pair_1_129 = make_float2(_tmem_load_36[2], _tmem_load_36[3]);
                float2 acc_pair_1_197 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1_129 = fma_f32x2_rn_ftz(weight_pair_1_36, cached_pair_1_129, acc_pair_1_197);
                acc32[18] = updated_pair_1_129.x;
                acc32[19] = updated_pair_1_129.y;
                float2 cached_pair_1_130 = make_float2(_tmem_load_36[4], _tmem_load_36[5]);
                float2 acc_pair_1_198 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1_130 = fma_f32x2_rn_ftz(weight_pair_1_36, cached_pair_1_130, acc_pair_1_198);
                acc32[20] = updated_pair_1_130.x;
                acc32[21] = updated_pair_1_130.y;
                float2 cached_pair_1_131 = make_float2(_tmem_load_36[6], _tmem_load_36[7]);
                float2 acc_pair_1_199 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1_131 = fma_f32x2_rn_ftz(weight_pair_1_36, cached_pair_1_131, acc_pair_1_199);
                acc32[22] = updated_pair_1_131.x;
                acc32[23] = updated_pair_1_131.y;
                float _tmem_load_37[8];
                tmem_ld_x8(&_tmem_load_37[0], tmem_base + 64 + 16);
                float2 weight_pair_1_37 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_132 = make_float2(_tmem_load_37[0], _tmem_load_37[1]);
                float2 acc_pair_1_200 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1_132 = fma_f32x2_rn_ftz(weight_pair_1_37, cached_pair_1_132, acc_pair_1_200);
                acc32[16] = updated_pair_1_132.x;
                acc32[17] = updated_pair_1_132.y;
                float2 cached_pair_1_133 = make_float2(_tmem_load_37[2], _tmem_load_37[3]);
                float2 acc_pair_1_201 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1_133 = fma_f32x2_rn_ftz(weight_pair_1_37, cached_pair_1_133, acc_pair_1_201);
                acc32[18] = updated_pair_1_133.x;
                acc32[19] = updated_pair_1_133.y;
                float2 cached_pair_1_134 = make_float2(_tmem_load_37[4], _tmem_load_37[5]);
                float2 acc_pair_1_202 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1_134 = fma_f32x2_rn_ftz(weight_pair_1_37, cached_pair_1_134, acc_pair_1_202);
                acc32[20] = updated_pair_1_134.x;
                acc32[21] = updated_pair_1_134.y;
                float2 cached_pair_1_135 = make_float2(_tmem_load_37[6], _tmem_load_37[7]);
                float2 acc_pair_1_203 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1_135 = fma_f32x2_rn_ftz(weight_pair_1_37, cached_pair_1_135, acc_pair_1_203);
                acc32[22] = updated_pair_1_135.x;
                acc32[23] = updated_pair_1_135.y;
                float2 correction_pair_1_19 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_204 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1_68;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_68) : "l"(*(const unsigned long long*)&acc_pair_1_204), "l"(*(const unsigned long long*)&correction_pair_1_19));
                acc32[24] = scaled_pair_1_68.x;
                acc32[25] = scaled_pair_1_68.y;
                float2 acc_pair_1_205 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1_69;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_69) : "l"(*(const unsigned long long*)&acc_pair_1_205), "l"(*(const unsigned long long*)&correction_pair_1_19));
                acc32[26] = scaled_pair_1_69.x;
                acc32[27] = scaled_pair_1_69.y;
                float _tmem_load_38[4];
                tmem_ld_x4(&_tmem_load_38[0], tmem_base + 24);
                float2 weight_pair_1_38 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_136 = make_float2(_tmem_load_38[0], _tmem_load_38[1]);
                float2 acc_pair_1_206 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1_136 = fma_f32x2_rn_ftz(weight_pair_1_38, cached_pair_1_136, acc_pair_1_206);
                acc32[24] = updated_pair_1_136.x;
                acc32[25] = updated_pair_1_136.y;
                float2 cached_pair_1_137 = make_float2(_tmem_load_38[2], _tmem_load_38[3]);
                float2 acc_pair_1_207 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1_137 = fma_f32x2_rn_ftz(weight_pair_1_38, cached_pair_1_137, acc_pair_1_207);
                acc32[26] = updated_pair_1_137.x;
                acc32[27] = updated_pair_1_137.y;
                float _tmem_load_39[4];
                tmem_ld_x4(&_tmem_load_39[0], tmem_base + 64 + 24);
                float2 weight_pair_1_39 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_138 = make_float2(_tmem_load_39[0], _tmem_load_39[1]);
                float2 acc_pair_1_208 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1_138 = fma_f32x2_rn_ftz(weight_pair_1_39, cached_pair_1_138, acc_pair_1_208);
                acc32[24] = updated_pair_1_138.x;
                acc32[25] = updated_pair_1_138.y;
                float2 cached_pair_1_139 = make_float2(_tmem_load_39[2], _tmem_load_39[3]);
                float2 acc_pair_1_209 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1_139 = fma_f32x2_rn_ftz(weight_pair_1_39, cached_pair_1_139, acc_pair_1_209);
                acc32[26] = updated_pair_1_139.x;
                acc32[27] = updated_pair_1_139.y;
                float2 correction_pair_1_20 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_210 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1_70;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_70) : "l"(*(const unsigned long long*)&acc_pair_1_210), "l"(*(const unsigned long long*)&correction_pair_1_20));
                acc32[28] = scaled_pair_1_70.x;
                acc32[29] = scaled_pair_1_70.y;
                float2 acc_pair_1_211 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1_71;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_71) : "l"(*(const unsigned long long*)&acc_pair_1_211), "l"(*(const unsigned long long*)&correction_pair_1_20));
                acc32[30] = scaled_pair_1_71.x;
                acc32[31] = scaled_pair_1_71.y;
                float2 acc_pair_1_212 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1_72;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_72) : "l"(*(const unsigned long long*)&acc_pair_1_212), "l"(*(const unsigned long long*)&correction_pair_1_20));
                acc32[32] = scaled_pair_1_72.x;
                acc32[33] = scaled_pair_1_72.y;
                float2 acc_pair_1_213 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1_73;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_73) : "l"(*(const unsigned long long*)&acc_pair_1_213), "l"(*(const unsigned long long*)&correction_pair_1_20));
                acc32[34] = scaled_pair_1_73.x;
                acc32[35] = scaled_pair_1_73.y;
                float _tmem_load_40[8];
                tmem_ld_x8(&_tmem_load_40[0], tmem_base + 32);
                float2 weight_pair_1_40 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_140 = make_float2(_tmem_load_40[0], _tmem_load_40[1]);
                float2 acc_pair_1_214 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1_140 = fma_f32x2_rn_ftz(weight_pair_1_40, cached_pair_1_140, acc_pair_1_214);
                acc32[28] = updated_pair_1_140.x;
                acc32[29] = updated_pair_1_140.y;
                float2 cached_pair_1_141 = make_float2(_tmem_load_40[2], _tmem_load_40[3]);
                float2 acc_pair_1_215 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1_141 = fma_f32x2_rn_ftz(weight_pair_1_40, cached_pair_1_141, acc_pair_1_215);
                acc32[30] = updated_pair_1_141.x;
                acc32[31] = updated_pair_1_141.y;
                float2 cached_pair_1_142 = make_float2(_tmem_load_40[4], _tmem_load_40[5]);
                float2 acc_pair_1_216 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1_142 = fma_f32x2_rn_ftz(weight_pair_1_40, cached_pair_1_142, acc_pair_1_216);
                acc32[32] = updated_pair_1_142.x;
                acc32[33] = updated_pair_1_142.y;
                float2 cached_pair_1_143 = make_float2(_tmem_load_40[6], _tmem_load_40[7]);
                float2 acc_pair_1_217 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1_143 = fma_f32x2_rn_ftz(weight_pair_1_40, cached_pair_1_143, acc_pair_1_217);
                acc32[34] = updated_pair_1_143.x;
                acc32[35] = updated_pair_1_143.y;
                float _tmem_load_41[8];
                tmem_ld_x8(&_tmem_load_41[0], tmem_base + 64 + 32);
                float2 weight_pair_1_41 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_144 = make_float2(_tmem_load_41[0], _tmem_load_41[1]);
                float2 acc_pair_1_218 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1_144 = fma_f32x2_rn_ftz(weight_pair_1_41, cached_pair_1_144, acc_pair_1_218);
                acc32[28] = updated_pair_1_144.x;
                acc32[29] = updated_pair_1_144.y;
                float2 cached_pair_1_145 = make_float2(_tmem_load_41[2], _tmem_load_41[3]);
                float2 acc_pair_1_219 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1_145 = fma_f32x2_rn_ftz(weight_pair_1_41, cached_pair_1_145, acc_pair_1_219);
                acc32[30] = updated_pair_1_145.x;
                acc32[31] = updated_pair_1_145.y;
                float2 cached_pair_1_146 = make_float2(_tmem_load_41[4], _tmem_load_41[5]);
                float2 acc_pair_1_220 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1_146 = fma_f32x2_rn_ftz(weight_pair_1_41, cached_pair_1_146, acc_pair_1_220);
                acc32[32] = updated_pair_1_146.x;
                acc32[33] = updated_pair_1_146.y;
                float2 cached_pair_1_147 = make_float2(_tmem_load_41[6], _tmem_load_41[7]);
                float2 acc_pair_1_221 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1_147 = fma_f32x2_rn_ftz(weight_pair_1_41, cached_pair_1_147, acc_pair_1_221);
                acc32[34] = updated_pair_1_147.x;
                acc32[35] = updated_pair_1_147.y;
                float2 correction_pair_1_21 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_222 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1_74;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_74) : "l"(*(const unsigned long long*)&acc_pair_1_222), "l"(*(const unsigned long long*)&correction_pair_1_21));
                acc32[36] = scaled_pair_1_74.x;
                acc32[37] = scaled_pair_1_74.y;
                float2 acc_pair_1_223 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1_75;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_75) : "l"(*(const unsigned long long*)&acc_pair_1_223), "l"(*(const unsigned long long*)&correction_pair_1_21));
                acc32[38] = scaled_pair_1_75.x;
                acc32[39] = scaled_pair_1_75.y;
                float2 acc_pair_1_224 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1_76;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_76) : "l"(*(const unsigned long long*)&acc_pair_1_224), "l"(*(const unsigned long long*)&correction_pair_1_21));
                acc32[40] = scaled_pair_1_76.x;
                acc32[41] = scaled_pair_1_76.y;
                float2 acc_pair_1_225 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1_77;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_77) : "l"(*(const unsigned long long*)&acc_pair_1_225), "l"(*(const unsigned long long*)&correction_pair_1_21));
                acc32[42] = scaled_pair_1_77.x;
                acc32[43] = scaled_pair_1_77.y;
                float _tmem_load_42[8];
                tmem_ld_x8(&_tmem_load_42[0], tmem_base + 40);
                float2 weight_pair_1_42 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_148 = make_float2(_tmem_load_42[0], _tmem_load_42[1]);
                float2 acc_pair_1_226 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1_148 = fma_f32x2_rn_ftz(weight_pair_1_42, cached_pair_1_148, acc_pair_1_226);
                acc32[36] = updated_pair_1_148.x;
                acc32[37] = updated_pair_1_148.y;
                float2 cached_pair_1_149 = make_float2(_tmem_load_42[2], _tmem_load_42[3]);
                float2 acc_pair_1_227 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1_149 = fma_f32x2_rn_ftz(weight_pair_1_42, cached_pair_1_149, acc_pair_1_227);
                acc32[38] = updated_pair_1_149.x;
                acc32[39] = updated_pair_1_149.y;
                float2 cached_pair_1_150 = make_float2(_tmem_load_42[4], _tmem_load_42[5]);
                float2 acc_pair_1_228 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1_150 = fma_f32x2_rn_ftz(weight_pair_1_42, cached_pair_1_150, acc_pair_1_228);
                acc32[40] = updated_pair_1_150.x;
                acc32[41] = updated_pair_1_150.y;
                float2 cached_pair_1_151 = make_float2(_tmem_load_42[6], _tmem_load_42[7]);
                float2 acc_pair_1_229 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1_151 = fma_f32x2_rn_ftz(weight_pair_1_42, cached_pair_1_151, acc_pair_1_229);
                acc32[42] = updated_pair_1_151.x;
                acc32[43] = updated_pair_1_151.y;
                float _tmem_load_43[8];
                tmem_ld_x8(&_tmem_load_43[0], tmem_base + 64 + 40);
                float2 weight_pair_1_43 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_152 = make_float2(_tmem_load_43[0], _tmem_load_43[1]);
                float2 acc_pair_1_230 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1_152 = fma_f32x2_rn_ftz(weight_pair_1_43, cached_pair_1_152, acc_pair_1_230);
                acc32[36] = updated_pair_1_152.x;
                acc32[37] = updated_pair_1_152.y;
                float2 cached_pair_1_153 = make_float2(_tmem_load_43[2], _tmem_load_43[3]);
                float2 acc_pair_1_231 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1_153 = fma_f32x2_rn_ftz(weight_pair_1_43, cached_pair_1_153, acc_pair_1_231);
                acc32[38] = updated_pair_1_153.x;
                acc32[39] = updated_pair_1_153.y;
                float2 cached_pair_1_154 = make_float2(_tmem_load_43[4], _tmem_load_43[5]);
                float2 acc_pair_1_232 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1_154 = fma_f32x2_rn_ftz(weight_pair_1_43, cached_pair_1_154, acc_pair_1_232);
                acc32[40] = updated_pair_1_154.x;
                acc32[41] = updated_pair_1_154.y;
                float2 cached_pair_1_155 = make_float2(_tmem_load_43[6], _tmem_load_43[7]);
                float2 acc_pair_1_233 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1_155 = fma_f32x2_rn_ftz(weight_pair_1_43, cached_pair_1_155, acc_pair_1_233);
                acc32[42] = updated_pair_1_155.x;
                acc32[43] = updated_pair_1_155.y;
                float2 correction_pair_1_22 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_234 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1_78;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_78) : "l"(*(const unsigned long long*)&acc_pair_1_234), "l"(*(const unsigned long long*)&correction_pair_1_22));
                acc32[44] = scaled_pair_1_78.x;
                acc32[45] = scaled_pair_1_78.y;
                float2 acc_pair_1_235 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1_79;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_79) : "l"(*(const unsigned long long*)&acc_pair_1_235), "l"(*(const unsigned long long*)&correction_pair_1_22));
                acc32[46] = scaled_pair_1_79.x;
                acc32[47] = scaled_pair_1_79.y;
                float2 acc_pair_1_236 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1_80;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_80) : "l"(*(const unsigned long long*)&acc_pair_1_236), "l"(*(const unsigned long long*)&correction_pair_1_22));
                acc32[48] = scaled_pair_1_80.x;
                acc32[49] = scaled_pair_1_80.y;
                float2 acc_pair_1_237 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1_81;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_81) : "l"(*(const unsigned long long*)&acc_pair_1_237), "l"(*(const unsigned long long*)&correction_pair_1_22));
                acc32[50] = scaled_pair_1_81.x;
                acc32[51] = scaled_pair_1_81.y;
                float _tmem_load_44[8];
                tmem_ld_x8(&_tmem_load_44[0], tmem_base + 48);
                float2 weight_pair_1_44 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_156 = make_float2(_tmem_load_44[0], _tmem_load_44[1]);
                float2 acc_pair_1_238 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1_156 = fma_f32x2_rn_ftz(weight_pair_1_44, cached_pair_1_156, acc_pair_1_238);
                acc32[44] = updated_pair_1_156.x;
                acc32[45] = updated_pair_1_156.y;
                float2 cached_pair_1_157 = make_float2(_tmem_load_44[2], _tmem_load_44[3]);
                float2 acc_pair_1_239 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1_157 = fma_f32x2_rn_ftz(weight_pair_1_44, cached_pair_1_157, acc_pair_1_239);
                acc32[46] = updated_pair_1_157.x;
                acc32[47] = updated_pair_1_157.y;
                float2 cached_pair_1_158 = make_float2(_tmem_load_44[4], _tmem_load_44[5]);
                float2 acc_pair_1_240 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1_158 = fma_f32x2_rn_ftz(weight_pair_1_44, cached_pair_1_158, acc_pair_1_240);
                acc32[48] = updated_pair_1_158.x;
                acc32[49] = updated_pair_1_158.y;
                float2 cached_pair_1_159 = make_float2(_tmem_load_44[6], _tmem_load_44[7]);
                float2 acc_pair_1_241 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1_159 = fma_f32x2_rn_ftz(weight_pair_1_44, cached_pair_1_159, acc_pair_1_241);
                acc32[50] = updated_pair_1_159.x;
                acc32[51] = updated_pair_1_159.y;
                float _tmem_load_45[8];
                tmem_ld_x8(&_tmem_load_45[0], tmem_base + 64 + 48);
                float2 weight_pair_1_45 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_160 = make_float2(_tmem_load_45[0], _tmem_load_45[1]);
                float2 acc_pair_1_242 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1_160 = fma_f32x2_rn_ftz(weight_pair_1_45, cached_pair_1_160, acc_pair_1_242);
                acc32[44] = updated_pair_1_160.x;
                acc32[45] = updated_pair_1_160.y;
                float2 cached_pair_1_161 = make_float2(_tmem_load_45[2], _tmem_load_45[3]);
                float2 acc_pair_1_243 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1_161 = fma_f32x2_rn_ftz(weight_pair_1_45, cached_pair_1_161, acc_pair_1_243);
                acc32[46] = updated_pair_1_161.x;
                acc32[47] = updated_pair_1_161.y;
                float2 cached_pair_1_162 = make_float2(_tmem_load_45[4], _tmem_load_45[5]);
                float2 acc_pair_1_244 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1_162 = fma_f32x2_rn_ftz(weight_pair_1_45, cached_pair_1_162, acc_pair_1_244);
                acc32[48] = updated_pair_1_162.x;
                acc32[49] = updated_pair_1_162.y;
                float2 cached_pair_1_163 = make_float2(_tmem_load_45[6], _tmem_load_45[7]);
                float2 acc_pair_1_245 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1_163 = fma_f32x2_rn_ftz(weight_pair_1_45, cached_pair_1_163, acc_pair_1_245);
                acc32[50] = updated_pair_1_163.x;
                acc32[51] = updated_pair_1_163.y;
                float2 correction_pair_1_23 = make_float2(correction_1120, correction_1120);
                float2 acc_pair_1_246 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1_82;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_82) : "l"(*(const unsigned long long*)&acc_pair_1_246), "l"(*(const unsigned long long*)&correction_pair_1_23));
                acc32[52] = scaled_pair_1_82.x;
                acc32[53] = scaled_pair_1_82.y;
                float2 acc_pair_1_247 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1_83;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_83) : "l"(*(const unsigned long long*)&acc_pair_1_247), "l"(*(const unsigned long long*)&correction_pair_1_23));
                acc32[54] = scaled_pair_1_83.x;
                acc32[55] = scaled_pair_1_83.y;
                float _tmem_load_46[4];
                tmem_ld_x4(&_tmem_load_46[0], tmem_base + 56);
                float2 weight_pair_1_46 = make_float2(weights_1118[0], weights_1118[0]);
                float2 cached_pair_1_164 = make_float2(_tmem_load_46[0], _tmem_load_46[1]);
                float2 acc_pair_1_248 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1_164 = fma_f32x2_rn_ftz(weight_pair_1_46, cached_pair_1_164, acc_pair_1_248);
                acc32[52] = updated_pair_1_164.x;
                acc32[53] = updated_pair_1_164.y;
                float2 cached_pair_1_165 = make_float2(_tmem_load_46[2], _tmem_load_46[3]);
                float2 acc_pair_1_249 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1_165 = fma_f32x2_rn_ftz(weight_pair_1_46, cached_pair_1_165, acc_pair_1_249);
                acc32[54] = updated_pair_1_165.x;
                acc32[55] = updated_pair_1_165.y;
                float _tmem_load_47[4];
                tmem_ld_x4(&_tmem_load_47[0], tmem_base + 64 + 56);
                float2 weight_pair_1_47 = make_float2(weights_1118[1], weights_1118[1]);
                float2 cached_pair_1_166 = make_float2(_tmem_load_47[0], _tmem_load_47[1]);
                float2 acc_pair_1_250 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1_166 = fma_f32x2_rn_ftz(weight_pair_1_47, cached_pair_1_166, acc_pair_1_250);
                acc32[52] = updated_pair_1_166.x;
                acc32[53] = updated_pair_1_166.y;
                float2 cached_pair_1_167 = make_float2(_tmem_load_47[2], _tmem_load_47[3]);
                float2 acc_pair_1_251 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1_167 = fma_f32x2_rn_ftz(weight_pair_1_47, cached_pair_1_167, acc_pair_1_251);
                acc32[54] = updated_pair_1_167.x;
                acc32[55] = updated_pair_1_167.y;
                s_running = __fmaf_rn(s_running, correction_1120, weight_sum_1121);
                m_running = new_max_1119;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; _phase_ready ^= 1; }
                if (token_1 == first_token) {
                    mbarrier_wait(norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float2 local_output_sum_pair_1_0 = make_float2(0.0f, 0.0f);
                float2 acc_pair_1_252 = make_float2(acc32[0], acc32[1]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_252, acc_pair_1_252, local_output_sum_pair_1_0);
                float2 acc_pair_1_253 = make_float2(acc32[2], acc32[3]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_253, acc_pair_1_253, local_output_sum_pair_1_0);
                float2 acc_pair_1_254 = make_float2(acc32[4], acc32[5]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_254, acc_pair_1_254, local_output_sum_pair_1_0);
                float2 acc_pair_1_255 = make_float2(acc32[6], acc32[7]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_255, acc_pair_1_255, local_output_sum_pair_1_0);
                float2 acc_pair_1_256 = make_float2(acc32[8], acc32[9]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_256, acc_pair_1_256, local_output_sum_pair_1_0);
                float2 acc_pair_1_257 = make_float2(acc32[10], acc32[11]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_257, acc_pair_1_257, local_output_sum_pair_1_0);
                float2 acc_pair_1_258 = make_float2(acc32[12], acc32[13]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_258, acc_pair_1_258, local_output_sum_pair_1_0);
                float2 acc_pair_1_259 = make_float2(acc32[14], acc32[15]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_259, acc_pair_1_259, local_output_sum_pair_1_0);
                float2 acc_pair_1_260 = make_float2(acc32[16], acc32[17]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_260, acc_pair_1_260, local_output_sum_pair_1_0);
                float2 acc_pair_1_261 = make_float2(acc32[18], acc32[19]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_261, acc_pair_1_261, local_output_sum_pair_1_0);
                float2 acc_pair_1_262 = make_float2(acc32[20], acc32[21]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_262, acc_pair_1_262, local_output_sum_pair_1_0);
                float2 acc_pair_1_263 = make_float2(acc32[22], acc32[23]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_263, acc_pair_1_263, local_output_sum_pair_1_0);
                float2 acc_pair_1_264 = make_float2(acc32[24], acc32[25]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_264, acc_pair_1_264, local_output_sum_pair_1_0);
                float2 acc_pair_1_265 = make_float2(acc32[26], acc32[27]);
                local_output_sum_pair_1_0 = fma_f32x2_rn_ftz(acc_pair_1_265, acc_pair_1_265, local_output_sum_pair_1_0);
                float local_output_sum_sq_1_0 = local_output_sum_pair_1_0.x + local_output_sum_pair_1_0.y;
                float _warp_reduce_168 = local_output_sum_sq_1_0;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_168 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_168, offset);
                local_output_sum_sq_1_0 = _warp_reduce_168;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)warp_in_group] = local_output_sum_sq_1_0;
                }
                float2 local_output_sum_pair_1_1 = make_float2(0.0f, 0.0f);
                float2 acc_pair_1_266 = make_float2(acc32[28], acc32[29]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_266, acc_pair_1_266, local_output_sum_pair_1_1);
                float2 acc_pair_1_267 = make_float2(acc32[30], acc32[31]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_267, acc_pair_1_267, local_output_sum_pair_1_1);
                float2 acc_pair_1_268 = make_float2(acc32[32], acc32[33]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_268, acc_pair_1_268, local_output_sum_pair_1_1);
                float2 acc_pair_1_269 = make_float2(acc32[34], acc32[35]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_269, acc_pair_1_269, local_output_sum_pair_1_1);
                float2 acc_pair_1_270 = make_float2(acc32[36], acc32[37]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_270, acc_pair_1_270, local_output_sum_pair_1_1);
                float2 acc_pair_1_271 = make_float2(acc32[38], acc32[39]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_271, acc_pair_1_271, local_output_sum_pair_1_1);
                float2 acc_pair_1_272 = make_float2(acc32[40], acc32[41]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_272, acc_pair_1_272, local_output_sum_pair_1_1);
                float2 acc_pair_1_273 = make_float2(acc32[42], acc32[43]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_273, acc_pair_1_273, local_output_sum_pair_1_1);
                float2 acc_pair_1_274 = make_float2(acc32[44], acc32[45]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_274, acc_pair_1_274, local_output_sum_pair_1_1);
                float2 acc_pair_1_275 = make_float2(acc32[46], acc32[47]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_275, acc_pair_1_275, local_output_sum_pair_1_1);
                float2 acc_pair_1_276 = make_float2(acc32[48], acc32[49]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_276, acc_pair_1_276, local_output_sum_pair_1_1);
                float2 acc_pair_1_277 = make_float2(acc32[50], acc32[51]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_277, acc_pair_1_277, local_output_sum_pair_1_1);
                float2 acc_pair_1_278 = make_float2(acc32[52], acc32[53]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_278, acc_pair_1_278, local_output_sum_pair_1_1);
                float2 acc_pair_1_279 = make_float2(acc32[54], acc32[55]);
                local_output_sum_pair_1_1 = fma_f32x2_rn_ftz(acc_pair_1_279, acc_pair_1_279, local_output_sum_pair_1_1);
                float local_output_sum_sq_1_1 = local_output_sum_pair_1_1.x + local_output_sum_pair_1_1.y;
                float _warp_reduce_169 = local_output_sum_sq_1_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_169 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_169, offset);
                local_output_sum_sq_1_1 = _warp_reduce_169;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)(4 + warp_in_group)] = local_output_sum_sq_1_1;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                float total_output_sum_sq = ((lane < 8) ? output_stats[output_stats_base + (unsigned int)lane] : 0.0f);
                float _shfl_down_sync_170 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 4, 8);
                total_output_sum_sq += _shfl_down_sync_170;
                float _shfl_down_sync_171 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 2, 8);
                total_output_sum_sq += _shfl_down_sync_171;
                float _shfl_down_sync_172 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 1, 8);
                total_output_sum_sq += _shfl_down_sync_172;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_173 = rsqrtf(__fmaf_rn(output_norm_eps * s_running, s_running, total_output_sum_sq / 7168.0f));
                    output_rsigma = _rsqrt_173;
                }
                float _shfl_sync_174 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_sync_174;
                int hidden_base_1401 = thread_in_group * 8;
                unsigned int smem_bf16x2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1401 * 2)) : "memory");
                float unpack_bf16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16[_pair * 2])[0]), "=f"((&unpack_bf16[_pair * 2])[1])
                        : "r"(smem_bf16x2[_pair]));
                }
                float2 output_rsigma_pair = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_175 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1_84;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_176 = make_float2(unpack_bf16[0], unpack_bf16[1]);
                float2 normalized_pair_1_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_0) : "l"(*(const unsigned long long*)&scaled_pair_1_84), "l"(*(const unsigned long long*)&_f32x2_mul_b_176));
                acc32[0] = normalized_pair_1_0.x;
                acc32[1] = normalized_pair_1_0.y;
                float2 _f32x2_mul_a_177 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1_85;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_85) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_178 = make_float2(unpack_bf16[2], unpack_bf16[3]);
                float2 normalized_pair_1_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_1) : "l"(*(const unsigned long long*)&scaled_pair_1_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_178));
                acc32[2] = normalized_pair_1_1.x;
                acc32[3] = normalized_pair_1_1.y;
                float2 _f32x2_mul_a_179 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1_86;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_180 = make_float2(unpack_bf16[4], unpack_bf16[5]);
                float2 normalized_pair_1_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_2) : "l"(*(const unsigned long long*)&scaled_pair_1_86), "l"(*(const unsigned long long*)&_f32x2_mul_b_180));
                acc32[4] = normalized_pair_1_2.x;
                acc32[5] = normalized_pair_1_2.y;
                float2 _f32x2_mul_a_181 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1_87;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_87) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_182 = make_float2(unpack_bf16[6], unpack_bf16[7]);
                float2 normalized_pair_1_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_3) : "l"(*(const unsigned long long*)&scaled_pair_1_87), "l"(*(const unsigned long long*)&_f32x2_mul_b_182));
                acc32[6] = normalized_pair_1_3.x;
                acc32[7] = normalized_pair_1_3.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1401))[0] = reinterpret_cast<int4*>(_packed_8)[0];
                int hidden_base_1409 = 2048 + thread_in_group * 8;
                unsigned int smem_bf16x2_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_0[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1409 * 2)) : "memory");
                float unpack_bf16_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_0[_pair * 2])[0]), "=f"((&unpack_bf16_0[_pair * 2])[1])
                        : "r"(smem_bf16x2_0[_pair]));
                }
                float2 output_rsigma_pair_1411 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_183 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1_88;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&output_rsigma_pair_1411));
                float2 _f32x2_mul_b_184 = make_float2(unpack_bf16_0[0], unpack_bf16_0[1]);
                float2 normalized_pair_1_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_4) : "l"(*(const unsigned long long*)&scaled_pair_1_88), "l"(*(const unsigned long long*)&_f32x2_mul_b_184));
                acc32[8] = normalized_pair_1_4.x;
                acc32[9] = normalized_pair_1_4.y;
                float2 _f32x2_mul_a_185 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1_89;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&output_rsigma_pair_1411));
                float2 _f32x2_mul_b_186 = make_float2(unpack_bf16_0[2], unpack_bf16_0[3]);
                float2 normalized_pair_1_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_5) : "l"(*(const unsigned long long*)&scaled_pair_1_89), "l"(*(const unsigned long long*)&_f32x2_mul_b_186));
                acc32[10] = normalized_pair_1_5.x;
                acc32[11] = normalized_pair_1_5.y;
                float2 _f32x2_mul_a_187 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1_90;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&output_rsigma_pair_1411));
                float2 _f32x2_mul_b_188 = make_float2(unpack_bf16_0[4], unpack_bf16_0[5]);
                float2 normalized_pair_1_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_6) : "l"(*(const unsigned long long*)&scaled_pair_1_90), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
                acc32[12] = normalized_pair_1_6.x;
                acc32[13] = normalized_pair_1_6.y;
                float2 _f32x2_mul_a_189 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1_91;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_91) : "l"(*(const unsigned long long*)&_f32x2_mul_a_189), "l"(*(const unsigned long long*)&output_rsigma_pair_1411));
                float2 _f32x2_mul_b_190 = make_float2(unpack_bf16_0[6], unpack_bf16_0[7]);
                float2 normalized_pair_1_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_7) : "l"(*(const unsigned long long*)&scaled_pair_1_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_190));
                acc32[14] = normalized_pair_1_7.x;
                acc32[15] = normalized_pair_1_7.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1409))[0] = reinterpret_cast<int4*>(_packed_9)[0];
                int hidden_base_1420 = 4096 + thread_in_group * 8;
                unsigned int smem_bf16x2_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1420 * 2)) : "memory");
                float unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_1[_pair * 2])[0]), "=f"((&unpack_bf16_1[_pair * 2])[1])
                        : "r"(smem_bf16x2_1[_pair]));
                }
                float2 output_rsigma_pair_1422 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_191 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1_92;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&output_rsigma_pair_1422));
                float2 _f32x2_mul_b_192 = make_float2(unpack_bf16_1[0], unpack_bf16_1[1]);
                float2 normalized_pair_1_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_8) : "l"(*(const unsigned long long*)&scaled_pair_1_92), "l"(*(const unsigned long long*)&_f32x2_mul_b_192));
                acc32[16] = normalized_pair_1_8.x;
                acc32[17] = normalized_pair_1_8.y;
                float2 _f32x2_mul_a_193 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1_93;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_93) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&output_rsigma_pair_1422));
                float2 _f32x2_mul_b_194 = make_float2(unpack_bf16_1[2], unpack_bf16_1[3]);
                float2 normalized_pair_1_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_9) : "l"(*(const unsigned long long*)&scaled_pair_1_93), "l"(*(const unsigned long long*)&_f32x2_mul_b_194));
                acc32[18] = normalized_pair_1_9.x;
                acc32[19] = normalized_pair_1_9.y;
                float2 _f32x2_mul_a_195 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1_94;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_195), "l"(*(const unsigned long long*)&output_rsigma_pair_1422));
                float2 _f32x2_mul_b_196 = make_float2(unpack_bf16_1[4], unpack_bf16_1[5]);
                float2 normalized_pair_1_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_10) : "l"(*(const unsigned long long*)&scaled_pair_1_94), "l"(*(const unsigned long long*)&_f32x2_mul_b_196));
                acc32[20] = normalized_pair_1_10.x;
                acc32[21] = normalized_pair_1_10.y;
                float2 _f32x2_mul_a_197 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1_95;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_95) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&output_rsigma_pair_1422));
                float2 _f32x2_mul_b_198 = make_float2(unpack_bf16_1[6], unpack_bf16_1[7]);
                float2 normalized_pair_1_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_11) : "l"(*(const unsigned long long*)&scaled_pair_1_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_198));
                acc32[22] = normalized_pair_1_11.x;
                acc32[23] = normalized_pair_1_11.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1420))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                int hidden_base_1431 = 6144 + thread_in_group * 4;
                unsigned int smem_bf16x2_2[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_2[(0) + 1]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1431 * 2)) : "memory");
                float unpack_bf16_2[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_2[_pair * 2])[0]), "=f"((&unpack_bf16_2[_pair * 2])[1])
                        : "r"(smem_bf16x2_2[_pair]));
                }
                float2 output_rsigma_pair_1433 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_199 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1_96;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_96) : "l"(*(const unsigned long long*)&_f32x2_mul_a_199), "l"(*(const unsigned long long*)&output_rsigma_pair_1433));
                float2 _f32x2_mul_b_200 = make_float2(unpack_bf16_2[0], unpack_bf16_2[1]);
                float2 normalized_pair_1_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_12) : "l"(*(const unsigned long long*)&scaled_pair_1_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_200));
                acc32[24] = normalized_pair_1_12.x;
                acc32[25] = normalized_pair_1_12.y;
                float2 _f32x2_mul_a_201 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1_97;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_97) : "l"(*(const unsigned long long*)&_f32x2_mul_a_201), "l"(*(const unsigned long long*)&output_rsigma_pair_1433));
                float2 _f32x2_mul_b_202 = make_float2(unpack_bf16_2[2], unpack_bf16_2[3]);
                float2 normalized_pair_1_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_13) : "l"(*(const unsigned long long*)&scaled_pair_1_97), "l"(*(const unsigned long long*)&_f32x2_mul_b_202));
                acc32[26] = normalized_pair_1_13.x;
                acc32[27] = normalized_pair_1_13.y;
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
                reinterpret_cast<int2*>(out + (token_1 * 7168 + hidden_base_1431))[0] = reinterpret_cast<int2*>(_packed_11)[0];
                int hidden_base_1438 = 1024 + thread_in_group * 8;
                unsigned int smem_bf16x2_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_3[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1438 * 2)) : "memory");
                float unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_3[_pair * 2])[0]), "=f"((&unpack_bf16_3[_pair * 2])[1])
                        : "r"(smem_bf16x2_3[_pair]));
                }
                float2 output_rsigma_pair_1440 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_203 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1_98;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_203), "l"(*(const unsigned long long*)&output_rsigma_pair_1440));
                float2 _f32x2_mul_b_204 = make_float2(unpack_bf16_3[0], unpack_bf16_3[1]);
                float2 normalized_pair_1_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_14) : "l"(*(const unsigned long long*)&scaled_pair_1_98), "l"(*(const unsigned long long*)&_f32x2_mul_b_204));
                acc32[28] = normalized_pair_1_14.x;
                acc32[29] = normalized_pair_1_14.y;
                float2 _f32x2_mul_a_205 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1_99;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_99) : "l"(*(const unsigned long long*)&_f32x2_mul_a_205), "l"(*(const unsigned long long*)&output_rsigma_pair_1440));
                float2 _f32x2_mul_b_206 = make_float2(unpack_bf16_3[2], unpack_bf16_3[3]);
                float2 normalized_pair_1_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_15) : "l"(*(const unsigned long long*)&scaled_pair_1_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_206));
                acc32[30] = normalized_pair_1_15.x;
                acc32[31] = normalized_pair_1_15.y;
                float2 _f32x2_mul_a_207 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1_100;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_100) : "l"(*(const unsigned long long*)&_f32x2_mul_a_207), "l"(*(const unsigned long long*)&output_rsigma_pair_1440));
                float2 _f32x2_mul_b_208 = make_float2(unpack_bf16_3[4], unpack_bf16_3[5]);
                float2 normalized_pair_1_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_16) : "l"(*(const unsigned long long*)&scaled_pair_1_100), "l"(*(const unsigned long long*)&_f32x2_mul_b_208));
                acc32[32] = normalized_pair_1_16.x;
                acc32[33] = normalized_pair_1_16.y;
                float2 _f32x2_mul_a_209 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1_101;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_101) : "l"(*(const unsigned long long*)&_f32x2_mul_a_209), "l"(*(const unsigned long long*)&output_rsigma_pair_1440));
                float2 _f32x2_mul_b_210 = make_float2(unpack_bf16_3[6], unpack_bf16_3[7]);
                float2 normalized_pair_1_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_17) : "l"(*(const unsigned long long*)&scaled_pair_1_101), "l"(*(const unsigned long long*)&_f32x2_mul_b_210));
                acc32[34] = normalized_pair_1_17.x;
                acc32[35] = normalized_pair_1_17.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1438))[0] = reinterpret_cast<int4*>(_packed_12)[0];
                int hidden_base_1449 = 3072 + thread_in_group * 8;
                unsigned int smem_bf16x2_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_4[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1449 * 2)) : "memory");
                float unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_4[_pair * 2])[0]), "=f"((&unpack_bf16_4[_pair * 2])[1])
                        : "r"(smem_bf16x2_4[_pair]));
                }
                float2 output_rsigma_pair_1451 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_211 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1_102;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_102) : "l"(*(const unsigned long long*)&_f32x2_mul_a_211), "l"(*(const unsigned long long*)&output_rsigma_pair_1451));
                float2 _f32x2_mul_b_212 = make_float2(unpack_bf16_4[0], unpack_bf16_4[1]);
                float2 normalized_pair_1_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_18) : "l"(*(const unsigned long long*)&scaled_pair_1_102), "l"(*(const unsigned long long*)&_f32x2_mul_b_212));
                acc32[36] = normalized_pair_1_18.x;
                acc32[37] = normalized_pair_1_18.y;
                float2 _f32x2_mul_a_213 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1_103;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_103) : "l"(*(const unsigned long long*)&_f32x2_mul_a_213), "l"(*(const unsigned long long*)&output_rsigma_pair_1451));
                float2 _f32x2_mul_b_214 = make_float2(unpack_bf16_4[2], unpack_bf16_4[3]);
                float2 normalized_pair_1_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_19) : "l"(*(const unsigned long long*)&scaled_pair_1_103), "l"(*(const unsigned long long*)&_f32x2_mul_b_214));
                acc32[38] = normalized_pair_1_19.x;
                acc32[39] = normalized_pair_1_19.y;
                float2 _f32x2_mul_a_215 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1_104;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_104) : "l"(*(const unsigned long long*)&_f32x2_mul_a_215), "l"(*(const unsigned long long*)&output_rsigma_pair_1451));
                float2 _f32x2_mul_b_216 = make_float2(unpack_bf16_4[4], unpack_bf16_4[5]);
                float2 normalized_pair_1_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_20) : "l"(*(const unsigned long long*)&scaled_pair_1_104), "l"(*(const unsigned long long*)&_f32x2_mul_b_216));
                acc32[40] = normalized_pair_1_20.x;
                acc32[41] = normalized_pair_1_20.y;
                float2 _f32x2_mul_a_217 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1_105;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_217), "l"(*(const unsigned long long*)&output_rsigma_pair_1451));
                float2 _f32x2_mul_b_218 = make_float2(unpack_bf16_4[6], unpack_bf16_4[7]);
                float2 normalized_pair_1_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_21) : "l"(*(const unsigned long long*)&scaled_pair_1_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_218));
                acc32[42] = normalized_pair_1_21.x;
                acc32[43] = normalized_pair_1_21.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1449))[0] = reinterpret_cast<int4*>(_packed_13)[0];
                int hidden_base_1460 = 5120 + thread_in_group * 8;
                unsigned int smem_bf16x2_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_5[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1460 * 2)) : "memory");
                float unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_5[_pair * 2])[0]), "=f"((&unpack_bf16_5[_pair * 2])[1])
                        : "r"(smem_bf16x2_5[_pair]));
                }
                float2 output_rsigma_pair_1462 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_219 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1_106;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_106) : "l"(*(const unsigned long long*)&_f32x2_mul_a_219), "l"(*(const unsigned long long*)&output_rsigma_pair_1462));
                float2 _f32x2_mul_b_220 = make_float2(unpack_bf16_5[0], unpack_bf16_5[1]);
                float2 normalized_pair_1_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_22) : "l"(*(const unsigned long long*)&scaled_pair_1_106), "l"(*(const unsigned long long*)&_f32x2_mul_b_220));
                acc32[44] = normalized_pair_1_22.x;
                acc32[45] = normalized_pair_1_22.y;
                float2 _f32x2_mul_a_221 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1_107;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_107) : "l"(*(const unsigned long long*)&_f32x2_mul_a_221), "l"(*(const unsigned long long*)&output_rsigma_pair_1462));
                float2 _f32x2_mul_b_222 = make_float2(unpack_bf16_5[2], unpack_bf16_5[3]);
                float2 normalized_pair_1_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_23) : "l"(*(const unsigned long long*)&scaled_pair_1_107), "l"(*(const unsigned long long*)&_f32x2_mul_b_222));
                acc32[46] = normalized_pair_1_23.x;
                acc32[47] = normalized_pair_1_23.y;
                float2 _f32x2_mul_a_223 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1_108;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_108) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&output_rsigma_pair_1462));
                float2 _f32x2_mul_b_224 = make_float2(unpack_bf16_5[4], unpack_bf16_5[5]);
                float2 normalized_pair_1_24;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_24) : "l"(*(const unsigned long long*)&scaled_pair_1_108), "l"(*(const unsigned long long*)&_f32x2_mul_b_224));
                acc32[48] = normalized_pair_1_24.x;
                acc32[49] = normalized_pair_1_24.y;
                float2 _f32x2_mul_a_225 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1_109;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_109) : "l"(*(const unsigned long long*)&_f32x2_mul_a_225), "l"(*(const unsigned long long*)&output_rsigma_pair_1462));
                float2 _f32x2_mul_b_226 = make_float2(unpack_bf16_5[6], unpack_bf16_5[7]);
                float2 normalized_pair_1_25;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_25) : "l"(*(const unsigned long long*)&scaled_pair_1_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_226));
                acc32[50] = normalized_pair_1_25.x;
                acc32[51] = normalized_pair_1_25.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1460))[0] = reinterpret_cast<int4*>(_packed_14)[0];
                int hidden_base_1471 = 6656 + thread_in_group * 4;
                unsigned int smem_bf16x2_6[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_6[(0) + 1]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1471 * 2)) : "memory");
                float unpack_bf16_6[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&unpack_bf16_6[_pair * 2])[0]), "=f"((&unpack_bf16_6[_pair * 2])[1])
                        : "r"(smem_bf16x2_6[_pair]));
                }
                float2 output_rsigma_pair_1473 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_227 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1_110;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_227), "l"(*(const unsigned long long*)&output_rsigma_pair_1473));
                float2 _f32x2_mul_b_228 = make_float2(unpack_bf16_6[0], unpack_bf16_6[1]);
                float2 normalized_pair_1_26;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_26) : "l"(*(const unsigned long long*)&scaled_pair_1_110), "l"(*(const unsigned long long*)&_f32x2_mul_b_228));
                acc32[52] = normalized_pair_1_26.x;
                acc32[53] = normalized_pair_1_26.y;
                float2 _f32x2_mul_a_229 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1_111;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1_111) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&output_rsigma_pair_1473));
                float2 _f32x2_mul_b_230 = make_float2(unpack_bf16_6[2], unpack_bf16_6[3]);
                float2 normalized_pair_1_27;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1_27) : "l"(*(const unsigned long long*)&scaled_pair_1_111), "l"(*(const unsigned long long*)&_f32x2_mul_b_230));
                acc32[54] = normalized_pair_1_27.x;
                acc32[55] = normalized_pair_1_27.y;
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
                reinterpret_cast<int2*>(out + (token_1 * 7168 + hidden_base_1471))[0] = reinterpret_cast<int2*>(_packed_15)[0];
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
