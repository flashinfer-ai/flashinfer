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
kernel_cake_kimi_k3_attn_res_71700f054507bd256b1b(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
            unsigned int stats_stream_base = stream_u_1 * 96;
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
                unsigned int smem_regs_857[4];
                #pragma unroll
                for (int _lr_33 = 0; _lr_33 < 4; _lr_33++) {
                    smem_regs_857[_lr_33] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_854 / 2) + (unsigned int)_lr_33];
                }
                __nv_bfloat162 prefix_pair = __as_bf16x2(smem_regs_855[0]);
                __nv_bfloat162 delta_pair = __as_bf16x2(smem_regs_857[0]);
                __nv_bfloat162 updated_bf16_pair = prefix_pair + delta_pair;
                values_856[0] = (float)updated_bf16_pair.x;
                values_856[1] = (float)updated_bf16_pair.y;
                __nv_bfloat162 prefix_pair_858 = __as_bf16x2(smem_regs_855[1]);
                __nv_bfloat162 delta_pair_859 = __as_bf16x2(smem_regs_857[1]);
                __nv_bfloat162 updated_bf16_pair_860 = prefix_pair_858 + delta_pair_859;
                values_856[2] = (float)updated_bf16_pair_860.x;
                values_856[3] = (float)updated_bf16_pair_860.y;
                __nv_bfloat162 prefix_pair_861 = __as_bf16x2(smem_regs_855[2]);
                __nv_bfloat162 delta_pair_862 = __as_bf16x2(smem_regs_857[2]);
                __nv_bfloat162 updated_bf16_pair_863 = prefix_pair_861 + delta_pair_862;
                values_856[4] = (float)updated_bf16_pair_863.x;
                values_856[5] = (float)updated_bf16_pair_863.y;
                __nv_bfloat162 prefix_pair_864 = __as_bf16x2(smem_regs_855[3]);
                __nv_bfloat162 delta_pair_865 = __as_bf16x2(smem_regs_857[3]);
                __nv_bfloat162 updated_bf16_pair_866 = prefix_pair_864 + delta_pair_865;
                values_856[6] = (float)updated_bf16_pair_866.x;
                values_856[7] = (float)updated_bf16_pair_866.y;
                unsigned int _packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_856[_lp*2 + 0], values_856[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_856[_lp*2 + 2], values_856[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_856[_lp*2 + 4], values_856[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_856[_lp*2 + 6], values_856[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_854))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                tmem_st_x8_f32(tmem_base, values_856);
                float2 value_pair_867 = make_float2(values_856[0], values_856[1]);
                float2 q_pair_868 = make_float2(q_cache[0], q_cache[1]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_867, value_pair_867, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_867, q_pair_868, local_dot_853[0]);
                float2 value_pair_869 = make_float2(values_856[2], values_856[3]);
                float2 q_pair_870 = make_float2(q_cache[2], q_cache[3]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_869, value_pair_869, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_869, q_pair_870, local_dot_853[0]);
                float2 value_pair_871 = make_float2(values_856[4], values_856[5]);
                float2 q_pair_872 = make_float2(q_cache[4], q_cache[5]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_871, value_pair_871, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_871, q_pair_872, local_dot_853[0]);
                float2 value_pair_873 = make_float2(values_856[6], values_856[7]);
                float2 q_pair_874 = make_float2(q_cache[6], q_cache[7]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_873, value_pair_873, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_873, q_pair_874, local_dot_853[0]);
                int hidden_base_875 = 2048 + thread_in_group * 8;
                unsigned int smem_regs_876[4];
                #pragma unroll
                for (int _lr_34 = 0; _lr_34 < 4; _lr_34++) {
                    smem_regs_876[_lr_34] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_875 / 2) + (unsigned int)_lr_34];
                }
                float values_877[8];
                unsigned int smem_regs_878[4];
                #pragma unroll
                for (int _lr_35 = 0; _lr_35 < 4; _lr_35++) {
                    smem_regs_878[_lr_35] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_875 / 2) + (unsigned int)_lr_35];
                }
                __nv_bfloat162 prefix_pair_879 = __as_bf16x2(smem_regs_876[0]);
                __nv_bfloat162 delta_pair_880 = __as_bf16x2(smem_regs_878[0]);
                __nv_bfloat162 updated_bf16_pair_881 = prefix_pair_879 + delta_pair_880;
                values_877[0] = (float)updated_bf16_pair_881.x;
                values_877[1] = (float)updated_bf16_pair_881.y;
                __nv_bfloat162 prefix_pair_882 = __as_bf16x2(smem_regs_876[1]);
                __nv_bfloat162 delta_pair_883 = __as_bf16x2(smem_regs_878[1]);
                __nv_bfloat162 updated_bf16_pair_884 = prefix_pair_882 + delta_pair_883;
                values_877[2] = (float)updated_bf16_pair_884.x;
                values_877[3] = (float)updated_bf16_pair_884.y;
                __nv_bfloat162 prefix_pair_885 = __as_bf16x2(smem_regs_876[2]);
                __nv_bfloat162 delta_pair_886 = __as_bf16x2(smem_regs_878[2]);
                __nv_bfloat162 updated_bf16_pair_887 = prefix_pair_885 + delta_pair_886;
                values_877[4] = (float)updated_bf16_pair_887.x;
                values_877[5] = (float)updated_bf16_pair_887.y;
                __nv_bfloat162 prefix_pair_888 = __as_bf16x2(smem_regs_876[3]);
                __nv_bfloat162 delta_pair_889 = __as_bf16x2(smem_regs_878[3]);
                __nv_bfloat162 updated_bf16_pair_890 = prefix_pair_888 + delta_pair_889;
                values_877[6] = (float)updated_bf16_pair_890.x;
                values_877[7] = (float)updated_bf16_pair_890.y;
                unsigned int _packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_877[_lp*2 + 0], values_877[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_877[_lp*2 + 2], values_877[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_877[_lp*2 + 4], values_877[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_877[_lp*2 + 6], values_877[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_875))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                tmem_st_x8_f32(tmem_base + 8, values_877);
                float2 value_pair_891 = make_float2(values_877[0], values_877[1]);
                float2 q_pair_892 = make_float2(q_cache[8], q_cache[9]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_891, value_pair_891, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_891, q_pair_892, local_dot_853[0]);
                float2 value_pair_893 = make_float2(values_877[2], values_877[3]);
                float2 q_pair_894 = make_float2(q_cache[10], q_cache[11]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_893, value_pair_893, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_893, q_pair_894, local_dot_853[0]);
                float2 value_pair_895 = make_float2(values_877[4], values_877[5]);
                float2 q_pair_896 = make_float2(q_cache[12], q_cache[13]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_895, value_pair_895, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_895, q_pair_896, local_dot_853[0]);
                float2 value_pair_897 = make_float2(values_877[6], values_877[7]);
                float2 q_pair_898 = make_float2(q_cache[14], q_cache[15]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_897, value_pair_897, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_897, q_pair_898, local_dot_853[0]);
                int hidden_base_899 = 4096 + thread_in_group * 8;
                unsigned int smem_regs_900[4];
                #pragma unroll
                for (int _lr_36 = 0; _lr_36 < 4; _lr_36++) {
                    smem_regs_900[_lr_36] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_899 / 2) + (unsigned int)_lr_36];
                }
                float values_901[8];
                unsigned int smem_regs_902[4];
                #pragma unroll
                for (int _lr_37 = 0; _lr_37 < 4; _lr_37++) {
                    smem_regs_902[_lr_37] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_899 / 2) + (unsigned int)_lr_37];
                }
                __nv_bfloat162 prefix_pair_903 = __as_bf16x2(smem_regs_900[0]);
                __nv_bfloat162 delta_pair_904 = __as_bf16x2(smem_regs_902[0]);
                __nv_bfloat162 updated_bf16_pair_905 = prefix_pair_903 + delta_pair_904;
                values_901[0] = (float)updated_bf16_pair_905.x;
                values_901[1] = (float)updated_bf16_pair_905.y;
                __nv_bfloat162 prefix_pair_906 = __as_bf16x2(smem_regs_900[1]);
                __nv_bfloat162 delta_pair_907 = __as_bf16x2(smem_regs_902[1]);
                __nv_bfloat162 updated_bf16_pair_908 = prefix_pair_906 + delta_pair_907;
                values_901[2] = (float)updated_bf16_pair_908.x;
                values_901[3] = (float)updated_bf16_pair_908.y;
                __nv_bfloat162 prefix_pair_909 = __as_bf16x2(smem_regs_900[2]);
                __nv_bfloat162 delta_pair_910 = __as_bf16x2(smem_regs_902[2]);
                __nv_bfloat162 updated_bf16_pair_911 = prefix_pair_909 + delta_pair_910;
                values_901[4] = (float)updated_bf16_pair_911.x;
                values_901[5] = (float)updated_bf16_pair_911.y;
                __nv_bfloat162 prefix_pair_912 = __as_bf16x2(smem_regs_900[3]);
                __nv_bfloat162 delta_pair_913 = __as_bf16x2(smem_regs_902[3]);
                __nv_bfloat162 updated_bf16_pair_914 = prefix_pair_912 + delta_pair_913;
                values_901[6] = (float)updated_bf16_pair_914.x;
                values_901[7] = (float)updated_bf16_pair_914.y;
                unsigned int _packed_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_901[_lp*2 + 0], values_901[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_901[_lp*2 + 2], values_901[_lp*2+1 + 2]));
                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_901[_lp*2 + 4], values_901[_lp*2+1 + 4]));
                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_901[_lp*2 + 6], values_901[_lp*2+1 + 6]));
                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_899))[0] = reinterpret_cast<int4*>(_packed_2)[0];
                tmem_st_x8_f32(tmem_base + 16, values_901);
                float2 value_pair_915 = make_float2(values_901[0], values_901[1]);
                float2 q_pair_916 = make_float2(q_cache[16], q_cache[17]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_915, value_pair_915, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_915, q_pair_916, local_dot_853[0]);
                float2 value_pair_917 = make_float2(values_901[2], values_901[3]);
                float2 q_pair_918 = make_float2(q_cache[18], q_cache[19]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_917, value_pair_917, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_917, q_pair_918, local_dot_853[0]);
                float2 value_pair_919 = make_float2(values_901[4], values_901[5]);
                float2 q_pair_920 = make_float2(q_cache[20], q_cache[21]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_919, value_pair_919, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_919, q_pair_920, local_dot_853[0]);
                float2 value_pair_921 = make_float2(values_901[6], values_901[7]);
                float2 q_pair_922 = make_float2(q_cache[22], q_cache[23]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_921, value_pair_921, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_921, q_pair_922, local_dot_853[0]);
                int hidden_base_923 = 6144 + thread_in_group * 4;
                unsigned int smem_regs_924[2];
                #pragma unroll
                for (int _lr_38 = 0; _lr_38 < 2; _lr_38++) {
                    smem_regs_924[_lr_38] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_923 / 2) + (unsigned int)_lr_38];
                }
                float values_925[4];
                unsigned int smem_regs_926[2];
                #pragma unroll
                for (int _lr_39 = 0; _lr_39 < 2; _lr_39++) {
                    smem_regs_926[_lr_39] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_923 / 2) + (unsigned int)_lr_39];
                }
                __nv_bfloat162 prefix_pair_927 = __as_bf16x2(smem_regs_924[0]);
                __nv_bfloat162 delta_pair_928 = __as_bf16x2(smem_regs_926[0]);
                __nv_bfloat162 updated_bf16_pair_929 = prefix_pair_927 + delta_pair_928;
                values_925[0] = (float)updated_bf16_pair_929.x;
                values_925[1] = (float)updated_bf16_pair_929.y;
                __nv_bfloat162 prefix_pair_930 = __as_bf16x2(smem_regs_924[1]);
                __nv_bfloat162 delta_pair_931 = __as_bf16x2(smem_regs_926[1]);
                __nv_bfloat162 updated_bf16_pair_932 = prefix_pair_930 + delta_pair_931;
                values_925[2] = (float)updated_bf16_pair_932.x;
                values_925[3] = (float)updated_bf16_pair_932.y;
                unsigned int _packed_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_925[_lp*2 + 0], values_925[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_925[_lp*2 + 2], values_925[_lp*2+1 + 2]));
                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(prefix + (token_1 * 7168 + hidden_base_923))[0] = reinterpret_cast<int2*>(_packed_3)[0];
                tmem_st_x4_f32(tmem_base + 24, values_925);
                float2 value_pair_933 = make_float2(values_925[0], values_925[1]);
                float2 q_pair_934 = make_float2(q_cache[24], q_cache[25]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_933, value_pair_933, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_933, q_pair_934, local_dot_853[0]);
                float2 value_pair_935 = make_float2(values_925[2], values_925[3]);
                float2 q_pair_936 = make_float2(q_cache[26], q_cache[27]);
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_935, value_pair_935, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_935, q_pair_936, local_dot_853[0]);
                int hidden_base_937 = 1024 + thread_in_group * 8;
                unsigned int smem_regs_938[4];
                #pragma unroll
                for (int _lr_40 = 0; _lr_40 < 4; _lr_40++) {
                    smem_regs_938[_lr_40] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_937 / 2) + (unsigned int)_lr_40];
                }
                float values_939[8];
                unsigned int smem_regs_940[4];
                #pragma unroll
                for (int _lr_41 = 0; _lr_41 < 4; _lr_41++) {
                    smem_regs_940[_lr_41] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_937 / 2) + (unsigned int)_lr_41];
                }
                __nv_bfloat162 prefix_pair_941 = __as_bf16x2(smem_regs_938[0]);
                __nv_bfloat162 delta_pair_942 = __as_bf16x2(smem_regs_940[0]);
                __nv_bfloat162 updated_bf16_pair_943 = prefix_pair_941 + delta_pair_942;
                values_939[0] = (float)updated_bf16_pair_943.x;
                values_939[1] = (float)updated_bf16_pair_943.y;
                __nv_bfloat162 prefix_pair_944 = __as_bf16x2(smem_regs_938[1]);
                __nv_bfloat162 delta_pair_945 = __as_bf16x2(smem_regs_940[1]);
                __nv_bfloat162 updated_bf16_pair_946 = prefix_pair_944 + delta_pair_945;
                values_939[2] = (float)updated_bf16_pair_946.x;
                values_939[3] = (float)updated_bf16_pair_946.y;
                __nv_bfloat162 prefix_pair_947 = __as_bf16x2(smem_regs_938[2]);
                __nv_bfloat162 delta_pair_948 = __as_bf16x2(smem_regs_940[2]);
                __nv_bfloat162 updated_bf16_pair_949 = prefix_pair_947 + delta_pair_948;
                values_939[4] = (float)updated_bf16_pair_949.x;
                values_939[5] = (float)updated_bf16_pair_949.y;
                __nv_bfloat162 prefix_pair_950 = __as_bf16x2(smem_regs_938[3]);
                __nv_bfloat162 delta_pair_951 = __as_bf16x2(smem_regs_940[3]);
                __nv_bfloat162 updated_bf16_pair_952 = prefix_pair_950 + delta_pair_951;
                values_939[6] = (float)updated_bf16_pair_952.x;
                values_939[7] = (float)updated_bf16_pair_952.y;
                unsigned int _packed_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_939[_lp*2 + 0], values_939[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_939[_lp*2 + 2], values_939[_lp*2+1 + 2]));
                    _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_939[_lp*2 + 4], values_939[_lp*2+1 + 4]));
                    _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_939[_lp*2 + 6], values_939[_lp*2+1 + 6]));
                    _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_937))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                tmem_st_x8_f32(tmem_base + 32, values_939);
                float2 value_pair_953 = make_float2(values_939[0], values_939[1]);
                float2 q_pair_954 = make_float2(q_cache[28], q_cache[29]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_953, value_pair_953, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_953, q_pair_954, local_dot_853[2]);
                float2 value_pair_955 = make_float2(values_939[2], values_939[3]);
                float2 q_pair_956 = make_float2(q_cache[30], q_cache[31]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_955, value_pair_955, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_955, q_pair_956, local_dot_853[2]);
                float2 value_pair_957 = make_float2(values_939[4], values_939[5]);
                float2 q_pair_958 = make_float2(q_cache[32], q_cache[33]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_957, value_pair_957, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_957, q_pair_958, local_dot_853[2]);
                float2 value_pair_959 = make_float2(values_939[6], values_939[7]);
                float2 q_pair_960 = make_float2(q_cache[34], q_cache[35]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_959, value_pair_959, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_959, q_pair_960, local_dot_853[2]);
                int hidden_base_961 = 3072 + thread_in_group * 8;
                unsigned int smem_regs_962[4];
                #pragma unroll
                for (int _lr_42 = 0; _lr_42 < 4; _lr_42++) {
                    smem_regs_962[_lr_42] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_961 / 2) + (unsigned int)_lr_42];
                }
                float values_963[8];
                unsigned int smem_regs_964[4];
                #pragma unroll
                for (int _lr_43 = 0; _lr_43 < 4; _lr_43++) {
                    smem_regs_964[_lr_43] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_961 / 2) + (unsigned int)_lr_43];
                }
                __nv_bfloat162 prefix_pair_965 = __as_bf16x2(smem_regs_962[0]);
                __nv_bfloat162 delta_pair_966 = __as_bf16x2(smem_regs_964[0]);
                __nv_bfloat162 updated_bf16_pair_967 = prefix_pair_965 + delta_pair_966;
                values_963[0] = (float)updated_bf16_pair_967.x;
                values_963[1] = (float)updated_bf16_pair_967.y;
                __nv_bfloat162 prefix_pair_968 = __as_bf16x2(smem_regs_962[1]);
                __nv_bfloat162 delta_pair_969 = __as_bf16x2(smem_regs_964[1]);
                __nv_bfloat162 updated_bf16_pair_970 = prefix_pair_968 + delta_pair_969;
                values_963[2] = (float)updated_bf16_pair_970.x;
                values_963[3] = (float)updated_bf16_pair_970.y;
                __nv_bfloat162 prefix_pair_971 = __as_bf16x2(smem_regs_962[2]);
                __nv_bfloat162 delta_pair_972 = __as_bf16x2(smem_regs_964[2]);
                __nv_bfloat162 updated_bf16_pair_973 = prefix_pair_971 + delta_pair_972;
                values_963[4] = (float)updated_bf16_pair_973.x;
                values_963[5] = (float)updated_bf16_pair_973.y;
                __nv_bfloat162 prefix_pair_974 = __as_bf16x2(smem_regs_962[3]);
                __nv_bfloat162 delta_pair_975 = __as_bf16x2(smem_regs_964[3]);
                __nv_bfloat162 updated_bf16_pair_976 = prefix_pair_974 + delta_pair_975;
                values_963[6] = (float)updated_bf16_pair_976.x;
                values_963[7] = (float)updated_bf16_pair_976.y;
                unsigned int _packed_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_963[_lp*2 + 0], values_963[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_963[_lp*2 + 2], values_963[_lp*2+1 + 2]));
                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_963[_lp*2 + 4], values_963[_lp*2+1 + 4]));
                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_963[_lp*2 + 6], values_963[_lp*2+1 + 6]));
                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_961))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                tmem_st_x8_f32(tmem_base + 40, values_963);
                float2 value_pair_977 = make_float2(values_963[0], values_963[1]);
                float2 q_pair_978 = make_float2(q_cache[36], q_cache[37]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_977, value_pair_977, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_977, q_pair_978, local_dot_853[2]);
                float2 value_pair_979 = make_float2(values_963[2], values_963[3]);
                float2 q_pair_980 = make_float2(q_cache[38], q_cache[39]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_979, value_pair_979, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_979, q_pair_980, local_dot_853[2]);
                float2 value_pair_981 = make_float2(values_963[4], values_963[5]);
                float2 q_pair_982 = make_float2(q_cache[40], q_cache[41]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_981, value_pair_981, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_981, q_pair_982, local_dot_853[2]);
                float2 value_pair_983 = make_float2(values_963[6], values_963[7]);
                float2 q_pair_984 = make_float2(q_cache[42], q_cache[43]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_983, value_pair_983, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_983, q_pair_984, local_dot_853[2]);
                int hidden_base_985 = 5120 + thread_in_group * 8;
                unsigned int smem_regs_986[4];
                #pragma unroll
                for (int _lr_44 = 0; _lr_44 < 4; _lr_44++) {
                    smem_regs_986[_lr_44] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_985 / 2) + (unsigned int)_lr_44];
                }
                float values_987[8];
                unsigned int smem_regs_988[4];
                #pragma unroll
                for (int _lr_45 = 0; _lr_45 < 4; _lr_45++) {
                    smem_regs_988[_lr_45] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_985 / 2) + (unsigned int)_lr_45];
                }
                __nv_bfloat162 prefix_pair_989 = __as_bf16x2(smem_regs_986[0]);
                __nv_bfloat162 delta_pair_990 = __as_bf16x2(smem_regs_988[0]);
                __nv_bfloat162 updated_bf16_pair_991 = prefix_pair_989 + delta_pair_990;
                values_987[0] = (float)updated_bf16_pair_991.x;
                values_987[1] = (float)updated_bf16_pair_991.y;
                __nv_bfloat162 prefix_pair_992 = __as_bf16x2(smem_regs_986[1]);
                __nv_bfloat162 delta_pair_993 = __as_bf16x2(smem_regs_988[1]);
                __nv_bfloat162 updated_bf16_pair_994 = prefix_pair_992 + delta_pair_993;
                values_987[2] = (float)updated_bf16_pair_994.x;
                values_987[3] = (float)updated_bf16_pair_994.y;
                __nv_bfloat162 prefix_pair_995 = __as_bf16x2(smem_regs_986[2]);
                __nv_bfloat162 delta_pair_996 = __as_bf16x2(smem_regs_988[2]);
                __nv_bfloat162 updated_bf16_pair_997 = prefix_pair_995 + delta_pair_996;
                values_987[4] = (float)updated_bf16_pair_997.x;
                values_987[5] = (float)updated_bf16_pair_997.y;
                __nv_bfloat162 prefix_pair_998 = __as_bf16x2(smem_regs_986[3]);
                __nv_bfloat162 delta_pair_999 = __as_bf16x2(smem_regs_988[3]);
                __nv_bfloat162 updated_bf16_pair_1000 = prefix_pair_998 + delta_pair_999;
                values_987[6] = (float)updated_bf16_pair_1000.x;
                values_987[7] = (float)updated_bf16_pair_1000.y;
                unsigned int _packed_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_987[_lp*2 + 0], values_987[_lp*2+1 + 0]));
                    _packed_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_987[_lp*2 + 2], values_987[_lp*2+1 + 2]));
                    _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_987[_lp*2 + 4], values_987[_lp*2+1 + 4]));
                    _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_987[_lp*2 + 6], values_987[_lp*2+1 + 6]));
                    _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(prefix + (token_1 * 7168 + hidden_base_985))[0] = reinterpret_cast<int4*>(_packed_6)[0];
                tmem_st_x8_f32(tmem_base + 48, values_987);
                float2 value_pair_1001 = make_float2(values_987[0], values_987[1]);
                float2 q_pair_1002 = make_float2(q_cache[44], q_cache[45]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1001, value_pair_1001, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1001, q_pair_1002, local_dot_853[2]);
                float2 value_pair_1003 = make_float2(values_987[2], values_987[3]);
                float2 q_pair_1004 = make_float2(q_cache[46], q_cache[47]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1003, value_pair_1003, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1003, q_pair_1004, local_dot_853[2]);
                float2 value_pair_1005 = make_float2(values_987[4], values_987[5]);
                float2 q_pair_1006 = make_float2(q_cache[48], q_cache[49]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1005, value_pair_1005, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1005, q_pair_1006, local_dot_853[2]);
                float2 value_pair_1007 = make_float2(values_987[6], values_987[7]);
                float2 q_pair_1008 = make_float2(q_cache[50], q_cache[51]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1007, value_pair_1007, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1007, q_pair_1008, local_dot_853[2]);
                int hidden_base_1009 = 6656 + thread_in_group * 4;
                unsigned int smem_regs_1010[2];
                #pragma unroll
                for (int _lr_46 = 0; _lr_46 < 2; _lr_46++) {
                    smem_regs_1010[_lr_46] = source_words[ring_stage_851 * 2 * 3584 + (unsigned int)(hidden_base_1009 / 2) + (unsigned int)_lr_46];
                }
                float values_1011[4];
                unsigned int smem_regs_1012[2];
                #pragma unroll
                for (int _lr_47 = 0; _lr_47 < 2; _lr_47++) {
                    smem_regs_1012[_lr_47] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_1009 / 2) + (unsigned int)_lr_47];
                }
                __nv_bfloat162 prefix_pair_1013 = __as_bf16x2(smem_regs_1010[0]);
                __nv_bfloat162 delta_pair_1014 = __as_bf16x2(smem_regs_1012[0]);
                __nv_bfloat162 updated_bf16_pair_1015 = prefix_pair_1013 + delta_pair_1014;
                values_1011[0] = (float)updated_bf16_pair_1015.x;
                values_1011[1] = (float)updated_bf16_pair_1015.y;
                __nv_bfloat162 prefix_pair_1016 = __as_bf16x2(smem_regs_1010[1]);
                __nv_bfloat162 delta_pair_1017 = __as_bf16x2(smem_regs_1012[1]);
                __nv_bfloat162 updated_bf16_pair_1018 = prefix_pair_1016 + delta_pair_1017;
                values_1011[2] = (float)updated_bf16_pair_1018.x;
                values_1011[3] = (float)updated_bf16_pair_1018.y;
                unsigned int _packed_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1011[_lp*2 + 0], values_1011[_lp*2+1 + 0]));
                    _packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_1011[_lp*2 + 2], values_1011[_lp*2+1 + 2]));
                    _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(prefix + (token_1 * 7168 + hidden_base_1009))[0] = reinterpret_cast<int2*>(_packed_7)[0];
                tmem_st_x4_f32(tmem_base + 56, values_1011);
                float2 value_pair_1019 = make_float2(values_1011[0], values_1011[1]);
                float2 q_pair_1020 = make_float2(q_cache[52], q_cache[53]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1019, value_pair_1019, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1019, q_pair_1020, local_dot_853[2]);
                float2 value_pair_1021 = make_float2(values_1011[2], values_1011[3]);
                float2 q_pair_1022 = make_float2(q_cache[54], q_cache[55]);
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1021, value_pair_1021, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1021, q_pair_1022, local_dot_853[2]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_851) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1023 = stats_stream_base + 64;
                float2 reduce_pair_1024 = make_float2(local_sum_sq_852[0].x + local_sum_sq_852[0].y, local_dot_853[0].x + local_dot_853[0].y);
                unsigned long long packed_pair_1025 = 0;
                unsigned long long peer_packed_1026 = 0;
                float2 peer_pair_1027 = make_float2(0.0f, 0.0f);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 16);
                peer_packed_1026 = _shfl_xor_sync_136;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 8);
                peer_packed_1026 = _shfl_xor_sync_137;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 4);
                peer_packed_1026 = _shfl_xor_sync_138;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 2);
                peer_packed_1026 = _shfl_xor_sync_139;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 1);
                peer_packed_1026 = _shfl_xor_sync_140;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                if (lane == 0) {
                    stats[slot_base_1023 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1024.x;
                    stats[slot_base_1023 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1024.y;
                }
                float2 reduce_pair_1028 = make_float2(local_sum_sq_852[2].x + local_sum_sq_852[2].y, local_dot_853[2].x + local_dot_853[2].y);
                unsigned long long packed_pair_1029 = 0;
                unsigned long long peer_packed_1030 = 0;
                float2 peer_pair_1031 = make_float2(0.0f, 0.0f);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 16);
                peer_packed_1030 = _shfl_xor_sync_141;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 8);
                peer_packed_1030 = _shfl_xor_sync_142;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 4);
                peer_packed_1030 = _shfl_xor_sync_143;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 2);
                peer_packed_1030 = _shfl_xor_sync_144;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 1);
                peer_packed_1030 = _shfl_xor_sync_145;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                if (lane == 0) {
                    stats[slot_base_1023 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1028.x;
                    stats[slot_base_1023 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1028.y;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int stat_source_1032 = lane / 8;
                int stat_warp_1033 = lane % 8;
                float total_sum_sq_1034 = 0.0f;
                float total_dot_1035 = 0.0f;
                if (stat_source_1032 < 1) {
                    total_sum_sq_1034 = stats[slot_base_1023 + (unsigned int)((stat_warp_1033 * 2 + stat_source_1032) * 2)];
                    total_dot_1035 = stats[slot_base_1023 + (unsigned int)((stat_warp_1033 * 2 + stat_source_1032) * 2) + 1];
                }
                float _shfl_down_sync_146 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1034, 4, 8);
                total_sum_sq_1034 += _shfl_down_sync_146;
                float _shfl_down_sync_147 = __shfl_down_sync(0xFFFFFFFF, total_dot_1035, 4, 8);
                total_dot_1035 += _shfl_down_sync_147;
                float _shfl_down_sync_148 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1034, 2, 8);
                total_sum_sq_1034 += _shfl_down_sync_148;
                float _shfl_down_sync_149 = __shfl_down_sync(0xFFFFFFFF, total_dot_1035, 2, 8);
                total_dot_1035 += _shfl_down_sync_149;
                float _shfl_down_sync_150 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1034, 1, 8);
                total_sum_sq_1034 += _shfl_down_sync_150;
                float _shfl_down_sync_151 = __shfl_down_sync(0xFFFFFFFF, total_dot_1035, 1, 8);
                total_dot_1035 += _shfl_down_sync_151;
                float local_logit_1036 = 0.0f;
                if (stat_source_1032 < 1 && stat_warp_1033 == 0) {
                    float _rsqrt_152 = rsqrtf(total_sum_sq_1034 / 7168.0f + eps);
                    local_logit_1036 = total_dot_1035 * _rsqrt_152;
                }
                float logits_1037[2];
                logits_1037[0] = 0.0f;
                float _shfl_sync_153 = __shfl_sync(0xFFFFFFFF, local_logit_1036, 0);
                logits_1037[0] = _shfl_sync_153;
                logits_1037[1] = 0.0f;
                float _shfl_sync_154 = __shfl_sync(0xFFFFFFFF, local_logit_1036, 8);
                logits_1037[1] = _shfl_sync_154;
                float weights_1038[2];
                float new_max_1039 = 0.0f;
                float correction_1040 = 0.0f;
                float weight_sum_1041 = 0.0f;
                weights_1038[0] = 0.0f;
                weights_1038[1] = 0.0f;
                float chunk_max_1042 = -3.4028235e+38f;
                chunk_max_1042 = max_noftz(chunk_max_1042, logits_1037[0]);
                new_max_1039 = max_noftz(m_running, chunk_max_1042);
                float _exp2_155 = approx_exp2((m_running - new_max_1039) * 1.442695f);
                correction_1040 = _exp2_155;
                float _exp2_156 = approx_exp2((logits_1037[0] - new_max_1039) * 1.442695f);
                weights_1038[0] = _exp2_156;
                weight_sum_1041 += weights_1038[0];
                float2 correction_pair_1043 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1044 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1045;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1045) : "l"(*(const unsigned long long*)&acc_pair_1044), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[0] = scaled_pair_1045.x;
                acc32[1] = scaled_pair_1045.y;
                float2 acc_pair_1046 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1047;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1047) : "l"(*(const unsigned long long*)&acc_pair_1046), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[2] = scaled_pair_1047.x;
                acc32[3] = scaled_pair_1047.y;
                float2 acc_pair_1048 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1049;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1049) : "l"(*(const unsigned long long*)&acc_pair_1048), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[4] = scaled_pair_1049.x;
                acc32[5] = scaled_pair_1049.y;
                float2 acc_pair_1050 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1051;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1051) : "l"(*(const unsigned long long*)&acc_pair_1050), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[6] = scaled_pair_1051.x;
                acc32[7] = scaled_pair_1051.y;
                float _tmem_load_32[8];
                tmem_ld_x8(&_tmem_load_32[0], tmem_base);
                float2 weight_pair_1052 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1053 = make_float2(_tmem_load_32[0], _tmem_load_32[1]);
                float2 acc_pair_1054 = make_float2(acc32[0], acc32[1]);
                float2 updated_pair_1055 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1053, acc_pair_1054);
                acc32[0] = updated_pair_1055.x;
                acc32[1] = updated_pair_1055.y;
                float2 cached_pair_1056 = make_float2(_tmem_load_32[2], _tmem_load_32[3]);
                float2 acc_pair_1057 = make_float2(acc32[2], acc32[3]);
                float2 updated_pair_1058 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1056, acc_pair_1057);
                acc32[2] = updated_pair_1058.x;
                acc32[3] = updated_pair_1058.y;
                float2 cached_pair_1059 = make_float2(_tmem_load_32[4], _tmem_load_32[5]);
                float2 acc_pair_1060 = make_float2(acc32[4], acc32[5]);
                float2 updated_pair_1061 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1059, acc_pair_1060);
                acc32[4] = updated_pair_1061.x;
                acc32[5] = updated_pair_1061.y;
                float2 cached_pair_1062 = make_float2(_tmem_load_32[6], _tmem_load_32[7]);
                float2 acc_pair_1063 = make_float2(acc32[6], acc32[7]);
                float2 updated_pair_1064 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1062, acc_pair_1063);
                acc32[6] = updated_pair_1064.x;
                acc32[7] = updated_pair_1064.y;
                float2 correction_pair_1065 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1066 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1067;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1067) : "l"(*(const unsigned long long*)&acc_pair_1066), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[8] = scaled_pair_1067.x;
                acc32[9] = scaled_pair_1067.y;
                float2 acc_pair_1068 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1069;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1069) : "l"(*(const unsigned long long*)&acc_pair_1068), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[10] = scaled_pair_1069.x;
                acc32[11] = scaled_pair_1069.y;
                float2 acc_pair_1070 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1071;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1071) : "l"(*(const unsigned long long*)&acc_pair_1070), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[12] = scaled_pair_1071.x;
                acc32[13] = scaled_pair_1071.y;
                float2 acc_pair_1072 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1073;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1073) : "l"(*(const unsigned long long*)&acc_pair_1072), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[14] = scaled_pair_1073.x;
                acc32[15] = scaled_pair_1073.y;
                float _tmem_load_33[8];
                tmem_ld_x8(&_tmem_load_33[0], tmem_base + 8);
                float2 weight_pair_1074 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1075 = make_float2(_tmem_load_33[0], _tmem_load_33[1]);
                float2 acc_pair_1076 = make_float2(acc32[8], acc32[9]);
                float2 updated_pair_1077 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1075, acc_pair_1076);
                acc32[8] = updated_pair_1077.x;
                acc32[9] = updated_pair_1077.y;
                float2 cached_pair_1078 = make_float2(_tmem_load_33[2], _tmem_load_33[3]);
                float2 acc_pair_1079 = make_float2(acc32[10], acc32[11]);
                float2 updated_pair_1080 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1078, acc_pair_1079);
                acc32[10] = updated_pair_1080.x;
                acc32[11] = updated_pair_1080.y;
                float2 cached_pair_1081 = make_float2(_tmem_load_33[4], _tmem_load_33[5]);
                float2 acc_pair_1082 = make_float2(acc32[12], acc32[13]);
                float2 updated_pair_1083 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1081, acc_pair_1082);
                acc32[12] = updated_pair_1083.x;
                acc32[13] = updated_pair_1083.y;
                float2 cached_pair_1084 = make_float2(_tmem_load_33[6], _tmem_load_33[7]);
                float2 acc_pair_1085 = make_float2(acc32[14], acc32[15]);
                float2 updated_pair_1086 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1084, acc_pair_1085);
                acc32[14] = updated_pair_1086.x;
                acc32[15] = updated_pair_1086.y;
                float2 correction_pair_1087 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1088 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1089;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1089) : "l"(*(const unsigned long long*)&acc_pair_1088), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[16] = scaled_pair_1089.x;
                acc32[17] = scaled_pair_1089.y;
                float2 acc_pair_1090 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1091;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1091) : "l"(*(const unsigned long long*)&acc_pair_1090), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[18] = scaled_pair_1091.x;
                acc32[19] = scaled_pair_1091.y;
                float2 acc_pair_1092 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1093;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1093) : "l"(*(const unsigned long long*)&acc_pair_1092), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[20] = scaled_pair_1093.x;
                acc32[21] = scaled_pair_1093.y;
                float2 acc_pair_1094 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1095;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1095) : "l"(*(const unsigned long long*)&acc_pair_1094), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[22] = scaled_pair_1095.x;
                acc32[23] = scaled_pair_1095.y;
                float _tmem_load_34[8];
                tmem_ld_x8(&_tmem_load_34[0], tmem_base + 16);
                float2 weight_pair_1096 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1097 = make_float2(_tmem_load_34[0], _tmem_load_34[1]);
                float2 acc_pair_1098 = make_float2(acc32[16], acc32[17]);
                float2 updated_pair_1099 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1097, acc_pair_1098);
                acc32[16] = updated_pair_1099.x;
                acc32[17] = updated_pair_1099.y;
                float2 cached_pair_1100 = make_float2(_tmem_load_34[2], _tmem_load_34[3]);
                float2 acc_pair_1101 = make_float2(acc32[18], acc32[19]);
                float2 updated_pair_1102 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1100, acc_pair_1101);
                acc32[18] = updated_pair_1102.x;
                acc32[19] = updated_pair_1102.y;
                float2 cached_pair_1103 = make_float2(_tmem_load_34[4], _tmem_load_34[5]);
                float2 acc_pair_1104 = make_float2(acc32[20], acc32[21]);
                float2 updated_pair_1105 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1103, acc_pair_1104);
                acc32[20] = updated_pair_1105.x;
                acc32[21] = updated_pair_1105.y;
                float2 cached_pair_1106 = make_float2(_tmem_load_34[6], _tmem_load_34[7]);
                float2 acc_pair_1107 = make_float2(acc32[22], acc32[23]);
                float2 updated_pair_1108 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1106, acc_pair_1107);
                acc32[22] = updated_pair_1108.x;
                acc32[23] = updated_pair_1108.y;
                float2 correction_pair_1109 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1110 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1111;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1111) : "l"(*(const unsigned long long*)&acc_pair_1110), "l"(*(const unsigned long long*)&correction_pair_1109));
                acc32[24] = scaled_pair_1111.x;
                acc32[25] = scaled_pair_1111.y;
                float2 acc_pair_1112 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1113;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1113) : "l"(*(const unsigned long long*)&acc_pair_1112), "l"(*(const unsigned long long*)&correction_pair_1109));
                acc32[26] = scaled_pair_1113.x;
                acc32[27] = scaled_pair_1113.y;
                float _tmem_load_35[4];
                tmem_ld_x4(&_tmem_load_35[0], tmem_base + 24);
                float2 weight_pair_1114 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1115 = make_float2(_tmem_load_35[0], _tmem_load_35[1]);
                float2 acc_pair_1116 = make_float2(acc32[24], acc32[25]);
                float2 updated_pair_1117 = fma_f32x2_rn_ftz(weight_pair_1114, cached_pair_1115, acc_pair_1116);
                acc32[24] = updated_pair_1117.x;
                acc32[25] = updated_pair_1117.y;
                float2 cached_pair_1118 = make_float2(_tmem_load_35[2], _tmem_load_35[3]);
                float2 acc_pair_1119 = make_float2(acc32[26], acc32[27]);
                float2 updated_pair_1120 = fma_f32x2_rn_ftz(weight_pair_1114, cached_pair_1118, acc_pair_1119);
                acc32[26] = updated_pair_1120.x;
                acc32[27] = updated_pair_1120.y;
                float2 correction_pair_1121 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1122 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1123;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1123) : "l"(*(const unsigned long long*)&acc_pair_1122), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[28] = scaled_pair_1123.x;
                acc32[29] = scaled_pair_1123.y;
                float2 acc_pair_1124 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1125;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1125) : "l"(*(const unsigned long long*)&acc_pair_1124), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[30] = scaled_pair_1125.x;
                acc32[31] = scaled_pair_1125.y;
                float2 acc_pair_1126 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1127;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1127) : "l"(*(const unsigned long long*)&acc_pair_1126), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[32] = scaled_pair_1127.x;
                acc32[33] = scaled_pair_1127.y;
                float2 acc_pair_1128 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1129;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1129) : "l"(*(const unsigned long long*)&acc_pair_1128), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[34] = scaled_pair_1129.x;
                acc32[35] = scaled_pair_1129.y;
                float _tmem_load_36[8];
                tmem_ld_x8(&_tmem_load_36[0], tmem_base + 32);
                float2 weight_pair_1130 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1131 = make_float2(_tmem_load_36[0], _tmem_load_36[1]);
                float2 acc_pair_1132 = make_float2(acc32[28], acc32[29]);
                float2 updated_pair_1133 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1131, acc_pair_1132);
                acc32[28] = updated_pair_1133.x;
                acc32[29] = updated_pair_1133.y;
                float2 cached_pair_1134 = make_float2(_tmem_load_36[2], _tmem_load_36[3]);
                float2 acc_pair_1135 = make_float2(acc32[30], acc32[31]);
                float2 updated_pair_1136 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1134, acc_pair_1135);
                acc32[30] = updated_pair_1136.x;
                acc32[31] = updated_pair_1136.y;
                float2 cached_pair_1137 = make_float2(_tmem_load_36[4], _tmem_load_36[5]);
                float2 acc_pair_1138 = make_float2(acc32[32], acc32[33]);
                float2 updated_pair_1139 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1137, acc_pair_1138);
                acc32[32] = updated_pair_1139.x;
                acc32[33] = updated_pair_1139.y;
                float2 cached_pair_1140 = make_float2(_tmem_load_36[6], _tmem_load_36[7]);
                float2 acc_pair_1141 = make_float2(acc32[34], acc32[35]);
                float2 updated_pair_1142 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1140, acc_pair_1141);
                acc32[34] = updated_pair_1142.x;
                acc32[35] = updated_pair_1142.y;
                float2 correction_pair_1143 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1144 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1145;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1145) : "l"(*(const unsigned long long*)&acc_pair_1144), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[36] = scaled_pair_1145.x;
                acc32[37] = scaled_pair_1145.y;
                float2 acc_pair_1146 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1147;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1147) : "l"(*(const unsigned long long*)&acc_pair_1146), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[38] = scaled_pair_1147.x;
                acc32[39] = scaled_pair_1147.y;
                float2 acc_pair_1148 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1149;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1149) : "l"(*(const unsigned long long*)&acc_pair_1148), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[40] = scaled_pair_1149.x;
                acc32[41] = scaled_pair_1149.y;
                float2 acc_pair_1150 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1151;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1151) : "l"(*(const unsigned long long*)&acc_pair_1150), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[42] = scaled_pair_1151.x;
                acc32[43] = scaled_pair_1151.y;
                float _tmem_load_37[8];
                tmem_ld_x8(&_tmem_load_37[0], tmem_base + 40);
                float2 weight_pair_1152 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1153 = make_float2(_tmem_load_37[0], _tmem_load_37[1]);
                float2 acc_pair_1154 = make_float2(acc32[36], acc32[37]);
                float2 updated_pair_1155 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1153, acc_pair_1154);
                acc32[36] = updated_pair_1155.x;
                acc32[37] = updated_pair_1155.y;
                float2 cached_pair_1156 = make_float2(_tmem_load_37[2], _tmem_load_37[3]);
                float2 acc_pair_1157 = make_float2(acc32[38], acc32[39]);
                float2 updated_pair_1158 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1156, acc_pair_1157);
                acc32[38] = updated_pair_1158.x;
                acc32[39] = updated_pair_1158.y;
                float2 cached_pair_1159 = make_float2(_tmem_load_37[4], _tmem_load_37[5]);
                float2 acc_pair_1160 = make_float2(acc32[40], acc32[41]);
                float2 updated_pair_1161 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1159, acc_pair_1160);
                acc32[40] = updated_pair_1161.x;
                acc32[41] = updated_pair_1161.y;
                float2 cached_pair_1162 = make_float2(_tmem_load_37[6], _tmem_load_37[7]);
                float2 acc_pair_1163 = make_float2(acc32[42], acc32[43]);
                float2 updated_pair_1164 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1162, acc_pair_1163);
                acc32[42] = updated_pair_1164.x;
                acc32[43] = updated_pair_1164.y;
                float2 correction_pair_1165 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1166 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1167;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1167) : "l"(*(const unsigned long long*)&acc_pair_1166), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[44] = scaled_pair_1167.x;
                acc32[45] = scaled_pair_1167.y;
                float2 acc_pair_1168 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1169;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1169) : "l"(*(const unsigned long long*)&acc_pair_1168), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[46] = scaled_pair_1169.x;
                acc32[47] = scaled_pair_1169.y;
                float2 acc_pair_1170 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1171;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1171) : "l"(*(const unsigned long long*)&acc_pair_1170), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[48] = scaled_pair_1171.x;
                acc32[49] = scaled_pair_1171.y;
                float2 acc_pair_1172 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1173;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1173) : "l"(*(const unsigned long long*)&acc_pair_1172), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[50] = scaled_pair_1173.x;
                acc32[51] = scaled_pair_1173.y;
                float _tmem_load_38[8];
                tmem_ld_x8(&_tmem_load_38[0], tmem_base + 48);
                float2 weight_pair_1174 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1175 = make_float2(_tmem_load_38[0], _tmem_load_38[1]);
                float2 acc_pair_1176 = make_float2(acc32[44], acc32[45]);
                float2 updated_pair_1177 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1175, acc_pair_1176);
                acc32[44] = updated_pair_1177.x;
                acc32[45] = updated_pair_1177.y;
                float2 cached_pair_1178 = make_float2(_tmem_load_38[2], _tmem_load_38[3]);
                float2 acc_pair_1179 = make_float2(acc32[46], acc32[47]);
                float2 updated_pair_1180 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1178, acc_pair_1179);
                acc32[46] = updated_pair_1180.x;
                acc32[47] = updated_pair_1180.y;
                float2 cached_pair_1181 = make_float2(_tmem_load_38[4], _tmem_load_38[5]);
                float2 acc_pair_1182 = make_float2(acc32[48], acc32[49]);
                float2 updated_pair_1183 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1181, acc_pair_1182);
                acc32[48] = updated_pair_1183.x;
                acc32[49] = updated_pair_1183.y;
                float2 cached_pair_1184 = make_float2(_tmem_load_38[6], _tmem_load_38[7]);
                float2 acc_pair_1185 = make_float2(acc32[50], acc32[51]);
                float2 updated_pair_1186 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1184, acc_pair_1185);
                acc32[50] = updated_pair_1186.x;
                acc32[51] = updated_pair_1186.y;
                float2 correction_pair_1187 = make_float2(correction_1040, correction_1040);
                float2 acc_pair_1188 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1189;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1189) : "l"(*(const unsigned long long*)&acc_pair_1188), "l"(*(const unsigned long long*)&correction_pair_1187));
                acc32[52] = scaled_pair_1189.x;
                acc32[53] = scaled_pair_1189.y;
                float2 acc_pair_1190 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1191;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1191) : "l"(*(const unsigned long long*)&acc_pair_1190), "l"(*(const unsigned long long*)&correction_pair_1187));
                acc32[54] = scaled_pair_1191.x;
                acc32[55] = scaled_pair_1191.y;
                float _tmem_load_39[4];
                tmem_ld_x4(&_tmem_load_39[0], tmem_base + 56);
                float2 weight_pair_1192 = make_float2(weights_1038[0], weights_1038[0]);
                float2 cached_pair_1193 = make_float2(_tmem_load_39[0], _tmem_load_39[1]);
                float2 acc_pair_1194 = make_float2(acc32[52], acc32[53]);
                float2 updated_pair_1195 = fma_f32x2_rn_ftz(weight_pair_1192, cached_pair_1193, acc_pair_1194);
                acc32[52] = updated_pair_1195.x;
                acc32[53] = updated_pair_1195.y;
                float2 cached_pair_1196 = make_float2(_tmem_load_39[2], _tmem_load_39[3]);
                float2 acc_pair_1197 = make_float2(acc32[54], acc32[55]);
                float2 updated_pair_1198 = fma_f32x2_rn_ftz(weight_pair_1192, cached_pair_1196, acc_pair_1197);
                acc32[54] = updated_pair_1198.x;
                acc32[55] = updated_pair_1198.y;
                s_running = __fmaf_rn(s_running, correction_1040, weight_sum_1041);
                m_running = new_max_1039;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; _phase_ready ^= 1; }
                if (token_1 == first_token) {
                    mbarrier_wait(norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float2 local_output_sum_pair = make_float2(0.0f, 0.0f);
                float2 acc_pair_1199 = make_float2(acc32[0], acc32[1]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1199, acc_pair_1199, local_output_sum_pair);
                float2 acc_pair_1200 = make_float2(acc32[2], acc32[3]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1200, acc_pair_1200, local_output_sum_pair);
                float2 acc_pair_1201 = make_float2(acc32[4], acc32[5]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1201, acc_pair_1201, local_output_sum_pair);
                float2 acc_pair_1202 = make_float2(acc32[6], acc32[7]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1202, acc_pair_1202, local_output_sum_pair);
                float2 acc_pair_1203 = make_float2(acc32[8], acc32[9]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1203, acc_pair_1203, local_output_sum_pair);
                float2 acc_pair_1204 = make_float2(acc32[10], acc32[11]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1204, acc_pair_1204, local_output_sum_pair);
                float2 acc_pair_1205 = make_float2(acc32[12], acc32[13]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1205, acc_pair_1205, local_output_sum_pair);
                float2 acc_pair_1206 = make_float2(acc32[14], acc32[15]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1206, acc_pair_1206, local_output_sum_pair);
                float2 acc_pair_1207 = make_float2(acc32[16], acc32[17]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1207, acc_pair_1207, local_output_sum_pair);
                float2 acc_pair_1208 = make_float2(acc32[18], acc32[19]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1208, acc_pair_1208, local_output_sum_pair);
                float2 acc_pair_1209 = make_float2(acc32[20], acc32[21]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1209, acc_pair_1209, local_output_sum_pair);
                float2 acc_pair_1210 = make_float2(acc32[22], acc32[23]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1210, acc_pair_1210, local_output_sum_pair);
                float2 acc_pair_1211 = make_float2(acc32[24], acc32[25]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1211, acc_pair_1211, local_output_sum_pair);
                float2 acc_pair_1212 = make_float2(acc32[26], acc32[27]);
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1212, acc_pair_1212, local_output_sum_pair);
                float local_output_sum_sq = local_output_sum_pair.x + local_output_sum_pair.y;
                float _warp_reduce_157 = local_output_sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_157 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_157, offset);
                local_output_sum_sq = _warp_reduce_157;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)warp_in_group] = local_output_sum_sq;
                }
                float2 local_output_sum_pair_1213 = make_float2(0.0f, 0.0f);
                float2 acc_pair_1214 = make_float2(acc32[28], acc32[29]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1214, acc_pair_1214, local_output_sum_pair_1213);
                float2 acc_pair_1215 = make_float2(acc32[30], acc32[31]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1215, acc_pair_1215, local_output_sum_pair_1213);
                float2 acc_pair_1216 = make_float2(acc32[32], acc32[33]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1216, acc_pair_1216, local_output_sum_pair_1213);
                float2 acc_pair_1217 = make_float2(acc32[34], acc32[35]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1217, acc_pair_1217, local_output_sum_pair_1213);
                float2 acc_pair_1218 = make_float2(acc32[36], acc32[37]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1218, acc_pair_1218, local_output_sum_pair_1213);
                float2 acc_pair_1219 = make_float2(acc32[38], acc32[39]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1219, acc_pair_1219, local_output_sum_pair_1213);
                float2 acc_pair_1220 = make_float2(acc32[40], acc32[41]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1220, acc_pair_1220, local_output_sum_pair_1213);
                float2 acc_pair_1221 = make_float2(acc32[42], acc32[43]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1221, acc_pair_1221, local_output_sum_pair_1213);
                float2 acc_pair_1222 = make_float2(acc32[44], acc32[45]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1222, acc_pair_1222, local_output_sum_pair_1213);
                float2 acc_pair_1223 = make_float2(acc32[46], acc32[47]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1223, acc_pair_1223, local_output_sum_pair_1213);
                float2 acc_pair_1224 = make_float2(acc32[48], acc32[49]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1224, acc_pair_1224, local_output_sum_pair_1213);
                float2 acc_pair_1225 = make_float2(acc32[50], acc32[51]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1225, acc_pair_1225, local_output_sum_pair_1213);
                float2 acc_pair_1226 = make_float2(acc32[52], acc32[53]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1226, acc_pair_1226, local_output_sum_pair_1213);
                float2 acc_pair_1227 = make_float2(acc32[54], acc32[55]);
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1227, acc_pair_1227, local_output_sum_pair_1213);
                float local_output_sum_sq_1228 = local_output_sum_pair_1213.x + local_output_sum_pair_1213.y;
                float _warp_reduce_158 = local_output_sum_sq_1228;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_158 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_158, offset);
                local_output_sum_sq_1228 = _warp_reduce_158;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)(4 + warp_in_group)] = local_output_sum_sq_1228;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                float total_output_sum_sq = ((lane < 8) ? output_stats[output_stats_base + (unsigned int)lane] : 0.0f);
                float _shfl_down_sync_159 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 4, 8);
                total_output_sum_sq += _shfl_down_sync_159;
                float _shfl_down_sync_160 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 2, 8);
                total_output_sum_sq += _shfl_down_sync_160;
                float _shfl_down_sync_161 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 1, 8);
                total_output_sum_sq += _shfl_down_sync_161;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_162 = rsqrtf(__fmaf_rn(output_norm_eps * s_running, s_running, total_output_sum_sq / 7168.0f));
                    output_rsigma = _rsqrt_162;
                }
                float _shfl_sync_163 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_sync_163;
                int hidden_base_1229 = thread_in_group * 8;
                unsigned int smem_bf16x2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1229 * 2)) : "memory");
                float _unpack_bf16_32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                        : "r"(smem_bf16x2[_pair]));
                }
                float2 output_rsigma_pair = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_164 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1230;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1230) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_165 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
                float2 normalized_pair;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair_1230), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
                acc32[0] = normalized_pair.x;
                acc32[1] = normalized_pair.y;
                float2 _f32x2_mul_a_166 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1231;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1231) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_167 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
                float2 normalized_pair_1232;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1232) : "l"(*(const unsigned long long*)&scaled_pair_1231), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
                acc32[2] = normalized_pair_1232.x;
                acc32[3] = normalized_pair_1232.y;
                float2 _f32x2_mul_a_168 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1233;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1233) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_169 = make_float2(_unpack_bf16_32[4], _unpack_bf16_32[5]);
                float2 normalized_pair_1234;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1234) : "l"(*(const unsigned long long*)&scaled_pair_1233), "l"(*(const unsigned long long*)&_f32x2_mul_b_169));
                acc32[4] = normalized_pair_1234.x;
                acc32[5] = normalized_pair_1234.y;
                float2 _f32x2_mul_a_170 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1235;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1235) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f32x2_mul_b_171 = make_float2(_unpack_bf16_32[6], _unpack_bf16_32[7]);
                float2 normalized_pair_1236;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1236) : "l"(*(const unsigned long long*)&scaled_pair_1235), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
                acc32[6] = normalized_pair_1236.x;
                acc32[7] = normalized_pair_1236.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1229))[0] = reinterpret_cast<int4*>(_packed_8)[0];
                int hidden_base_1237 = 2048 + thread_in_group * 8;
                unsigned int smem_bf16x2_1238[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1238[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1238[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1238[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1238[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1237 * 2)) : "memory");
                float _unpack_bf16_33[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                        : "r"(smem_bf16x2_1238[_pair]));
                }
                float2 output_rsigma_pair_1239 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_172 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1240;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1240) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f32x2_mul_b_173 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
                float2 normalized_pair_1241;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1241) : "l"(*(const unsigned long long*)&scaled_pair_1240), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
                acc32[8] = normalized_pair_1241.x;
                acc32[9] = normalized_pair_1241.y;
                float2 _f32x2_mul_a_174 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1242;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1242) : "l"(*(const unsigned long long*)&_f32x2_mul_a_174), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f32x2_mul_b_175 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
                float2 normalized_pair_1243;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1243) : "l"(*(const unsigned long long*)&scaled_pair_1242), "l"(*(const unsigned long long*)&_f32x2_mul_b_175));
                acc32[10] = normalized_pair_1243.x;
                acc32[11] = normalized_pair_1243.y;
                float2 _f32x2_mul_a_176 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1244;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1244) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f32x2_mul_b_177 = make_float2(_unpack_bf16_33[4], _unpack_bf16_33[5]);
                float2 normalized_pair_1245;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1245) : "l"(*(const unsigned long long*)&scaled_pair_1244), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
                acc32[12] = normalized_pair_1245.x;
                acc32[13] = normalized_pair_1245.y;
                float2 _f32x2_mul_a_178 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1246;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1246) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f32x2_mul_b_179 = make_float2(_unpack_bf16_33[6], _unpack_bf16_33[7]);
                float2 normalized_pair_1247;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1247) : "l"(*(const unsigned long long*)&scaled_pair_1246), "l"(*(const unsigned long long*)&_f32x2_mul_b_179));
                acc32[14] = normalized_pair_1247.x;
                acc32[15] = normalized_pair_1247.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1237))[0] = reinterpret_cast<int4*>(_packed_9)[0];
                int hidden_base_1248 = 4096 + thread_in_group * 8;
                unsigned int smem_bf16x2_1249[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1249[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1249[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1249[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1249[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1248 * 2)) : "memory");
                float _unpack_bf16_34[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                        : "r"(smem_bf16x2_1249[_pair]));
                }
                float2 output_rsigma_pair_1250 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_180 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1251;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1251) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f32x2_mul_b_181 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
                float2 normalized_pair_1252;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1252) : "l"(*(const unsigned long long*)&scaled_pair_1251), "l"(*(const unsigned long long*)&_f32x2_mul_b_181));
                acc32[16] = normalized_pair_1252.x;
                acc32[17] = normalized_pair_1252.y;
                float2 _f32x2_mul_a_182 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1253;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1253) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f32x2_mul_b_183 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
                float2 normalized_pair_1254;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1254) : "l"(*(const unsigned long long*)&scaled_pair_1253), "l"(*(const unsigned long long*)&_f32x2_mul_b_183));
                acc32[18] = normalized_pair_1254.x;
                acc32[19] = normalized_pair_1254.y;
                float2 _f32x2_mul_a_184 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1255;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1255) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f32x2_mul_b_185 = make_float2(_unpack_bf16_34[4], _unpack_bf16_34[5]);
                float2 normalized_pair_1256;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1256) : "l"(*(const unsigned long long*)&scaled_pair_1255), "l"(*(const unsigned long long*)&_f32x2_mul_b_185));
                acc32[20] = normalized_pair_1256.x;
                acc32[21] = normalized_pair_1256.y;
                float2 _f32x2_mul_a_186 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1257;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1257) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f32x2_mul_b_187 = make_float2(_unpack_bf16_34[6], _unpack_bf16_34[7]);
                float2 normalized_pair_1258;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1258) : "l"(*(const unsigned long long*)&scaled_pair_1257), "l"(*(const unsigned long long*)&_f32x2_mul_b_187));
                acc32[22] = normalized_pair_1258.x;
                acc32[23] = normalized_pair_1258.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1248))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                int hidden_base_1259 = 6144 + thread_in_group * 4;
                unsigned int smem_bf16x2_1260[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1260[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1260[(0) + 1]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1259 * 2)) : "memory");
                float _unpack_bf16_35[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                        : "r"(smem_bf16x2_1260[_pair]));
                }
                float2 output_rsigma_pair_1261 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_188 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1262;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1262) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&output_rsigma_pair_1261));
                float2 _f32x2_mul_b_189 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
                float2 normalized_pair_1263;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1263) : "l"(*(const unsigned long long*)&scaled_pair_1262), "l"(*(const unsigned long long*)&_f32x2_mul_b_189));
                acc32[24] = normalized_pair_1263.x;
                acc32[25] = normalized_pair_1263.y;
                float2 _f32x2_mul_a_190 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1264;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1264) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&output_rsigma_pair_1261));
                float2 _f32x2_mul_b_191 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
                float2 normalized_pair_1265;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1265) : "l"(*(const unsigned long long*)&scaled_pair_1264), "l"(*(const unsigned long long*)&_f32x2_mul_b_191));
                acc32[26] = normalized_pair_1265.x;
                acc32[27] = normalized_pair_1265.y;
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
                reinterpret_cast<int2*>(out + (token_1 * 7168 + hidden_base_1259))[0] = reinterpret_cast<int2*>(_packed_11)[0];
                int hidden_base_1266 = 1024 + thread_in_group * 8;
                unsigned int smem_bf16x2_1267[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1267[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1267[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1267[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1267[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1266 * 2)) : "memory");
                float _unpack_bf16_36[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                        : "r"(smem_bf16x2_1267[_pair]));
                }
                float2 output_rsigma_pair_1268 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_192 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1269;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1269) : "l"(*(const unsigned long long*)&_f32x2_mul_a_192), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f32x2_mul_b_193 = make_float2(_unpack_bf16_36[0], _unpack_bf16_36[1]);
                float2 normalized_pair_1270;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1270) : "l"(*(const unsigned long long*)&scaled_pair_1269), "l"(*(const unsigned long long*)&_f32x2_mul_b_193));
                acc32[28] = normalized_pair_1270.x;
                acc32[29] = normalized_pair_1270.y;
                float2 _f32x2_mul_a_194 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1271;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1271) : "l"(*(const unsigned long long*)&_f32x2_mul_a_194), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f32x2_mul_b_195 = make_float2(_unpack_bf16_36[2], _unpack_bf16_36[3]);
                float2 normalized_pair_1272;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1272) : "l"(*(const unsigned long long*)&scaled_pair_1271), "l"(*(const unsigned long long*)&_f32x2_mul_b_195));
                acc32[30] = normalized_pair_1272.x;
                acc32[31] = normalized_pair_1272.y;
                float2 _f32x2_mul_a_196 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1273;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1273) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f32x2_mul_b_197 = make_float2(_unpack_bf16_36[4], _unpack_bf16_36[5]);
                float2 normalized_pair_1274;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1274) : "l"(*(const unsigned long long*)&scaled_pair_1273), "l"(*(const unsigned long long*)&_f32x2_mul_b_197));
                acc32[32] = normalized_pair_1274.x;
                acc32[33] = normalized_pair_1274.y;
                float2 _f32x2_mul_a_198 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1275;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1275) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f32x2_mul_b_199 = make_float2(_unpack_bf16_36[6], _unpack_bf16_36[7]);
                float2 normalized_pair_1276;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1276) : "l"(*(const unsigned long long*)&scaled_pair_1275), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
                acc32[34] = normalized_pair_1276.x;
                acc32[35] = normalized_pair_1276.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1266))[0] = reinterpret_cast<int4*>(_packed_12)[0];
                int hidden_base_1277 = 3072 + thread_in_group * 8;
                unsigned int smem_bf16x2_1278[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1278[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1278[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1278[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1278[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1277 * 2)) : "memory");
                float _unpack_bf16_37[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                        : "r"(smem_bf16x2_1278[_pair]));
                }
                float2 output_rsigma_pair_1279 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_200 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1280;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1280) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f32x2_mul_b_201 = make_float2(_unpack_bf16_37[0], _unpack_bf16_37[1]);
                float2 normalized_pair_1281;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1281) : "l"(*(const unsigned long long*)&scaled_pair_1280), "l"(*(const unsigned long long*)&_f32x2_mul_b_201));
                acc32[36] = normalized_pair_1281.x;
                acc32[37] = normalized_pair_1281.y;
                float2 _f32x2_mul_a_202 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1282;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1282) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f32x2_mul_b_203 = make_float2(_unpack_bf16_37[2], _unpack_bf16_37[3]);
                float2 normalized_pair_1283;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1283) : "l"(*(const unsigned long long*)&scaled_pair_1282), "l"(*(const unsigned long long*)&_f32x2_mul_b_203));
                acc32[38] = normalized_pair_1283.x;
                acc32[39] = normalized_pair_1283.y;
                float2 _f32x2_mul_a_204 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1284;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1284) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f32x2_mul_b_205 = make_float2(_unpack_bf16_37[4], _unpack_bf16_37[5]);
                float2 normalized_pair_1285;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1285) : "l"(*(const unsigned long long*)&scaled_pair_1284), "l"(*(const unsigned long long*)&_f32x2_mul_b_205));
                acc32[40] = normalized_pair_1285.x;
                acc32[41] = normalized_pair_1285.y;
                float2 _f32x2_mul_a_206 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1286;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1286) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f32x2_mul_b_207 = make_float2(_unpack_bf16_37[6], _unpack_bf16_37[7]);
                float2 normalized_pair_1287;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1287) : "l"(*(const unsigned long long*)&scaled_pair_1286), "l"(*(const unsigned long long*)&_f32x2_mul_b_207));
                acc32[42] = normalized_pair_1287.x;
                acc32[43] = normalized_pair_1287.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1277))[0] = reinterpret_cast<int4*>(_packed_13)[0];
                int hidden_base_1288 = 5120 + thread_in_group * 8;
                unsigned int smem_bf16x2_1289[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1289[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1289[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1289[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1289[(0) + 3]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1288 * 2)) : "memory");
                float _unpack_bf16_38[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                        : "r"(smem_bf16x2_1289[_pair]));
                }
                float2 output_rsigma_pair_1290 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_208 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1291;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1291) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f32x2_mul_b_209 = make_float2(_unpack_bf16_38[0], _unpack_bf16_38[1]);
                float2 normalized_pair_1292;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1292) : "l"(*(const unsigned long long*)&scaled_pair_1291), "l"(*(const unsigned long long*)&_f32x2_mul_b_209));
                acc32[44] = normalized_pair_1292.x;
                acc32[45] = normalized_pair_1292.y;
                float2 _f32x2_mul_a_210 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1293;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1293) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f32x2_mul_b_211 = make_float2(_unpack_bf16_38[2], _unpack_bf16_38[3]);
                float2 normalized_pair_1294;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1294) : "l"(*(const unsigned long long*)&scaled_pair_1293), "l"(*(const unsigned long long*)&_f32x2_mul_b_211));
                acc32[46] = normalized_pair_1294.x;
                acc32[47] = normalized_pair_1294.y;
                float2 _f32x2_mul_a_212 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1295;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1295) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f32x2_mul_b_213 = make_float2(_unpack_bf16_38[4], _unpack_bf16_38[5]);
                float2 normalized_pair_1296;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1296) : "l"(*(const unsigned long long*)&scaled_pair_1295), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
                acc32[48] = normalized_pair_1296.x;
                acc32[49] = normalized_pair_1296.y;
                float2 _f32x2_mul_a_214 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1297;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1297) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f32x2_mul_b_215 = make_float2(_unpack_bf16_38[6], _unpack_bf16_38[7]);
                float2 normalized_pair_1298;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1298) : "l"(*(const unsigned long long*)&scaled_pair_1297), "l"(*(const unsigned long long*)&_f32x2_mul_b_215));
                acc32[50] = normalized_pair_1298.x;
                acc32[51] = normalized_pair_1298.y;
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
                reinterpret_cast<int4*>(out + (token_1 * 7168 + hidden_base_1288))[0] = reinterpret_cast<int4*>(_packed_14)[0];
                int hidden_base_1299 = 6656 + thread_in_group * 4;
                unsigned int smem_bf16x2_1300[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1300[0])), "=r"(*reinterpret_cast<uint32_t*>(&smem_bf16x2_1300[(0) + 1]))
                    : "r"(output_norm_buf_addr + (unsigned int)(hidden_base_1299 * 2)) : "memory");
                float _unpack_bf16_39[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                        : "r"(smem_bf16x2_1300[_pair]));
                }
                float2 output_rsigma_pair_1301 = make_float2(output_rsigma, output_rsigma);
                float2 _f32x2_mul_a_216 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1302;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1302) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&output_rsigma_pair_1301));
                float2 _f32x2_mul_b_217 = make_float2(_unpack_bf16_39[0], _unpack_bf16_39[1]);
                float2 normalized_pair_1303;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1303) : "l"(*(const unsigned long long*)&scaled_pair_1302), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
                acc32[52] = normalized_pair_1303.x;
                acc32[53] = normalized_pair_1303.y;
                float2 _f32x2_mul_a_218 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1304;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1304) : "l"(*(const unsigned long long*)&_f32x2_mul_a_218), "l"(*(const unsigned long long*)&output_rsigma_pair_1301));
                float2 _f32x2_mul_b_219 = make_float2(_unpack_bf16_39[2], _unpack_bf16_39[3]);
                float2 normalized_pair_1305;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1305) : "l"(*(const unsigned long long*)&scaled_pair_1304), "l"(*(const unsigned long long*)&_f32x2_mul_b_219));
                acc32[54] = normalized_pair_1305.x;
                acc32[55] = normalized_pair_1305.y;
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
                reinterpret_cast<int2*>(out + (token_1 * 7168 + hidden_base_1299))[0] = reinterpret_cast<int2*>(_packed_15)[0];
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
