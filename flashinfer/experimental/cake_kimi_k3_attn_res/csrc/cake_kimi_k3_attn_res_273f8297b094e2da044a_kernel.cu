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
#define NUM_PIPE_STAGES 2
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
#define THREADS 320
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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kimi_k3_attn_res_273f8297b094e2da044a(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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

    if (warp == 0) {
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
    if (warp == 2) {
        int _tmem_hold = smem + 158856;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_values_tmem = taddr;

    // ---- Role: producers ----
    if (warp <= 1) {
        { // producers_main
            if (elect_sync()) {
                int warp_id_in_role = (warp - 0);
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
                unsigned int phase = 0;
                unsigned int phase_flipped = 1;
                #pragma unroll 1
                for (int token = bid + stream * num_bids; token < M; token += 2 * num_bids) {
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, phase_flipped);
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
                    if (stage == 2) { stage = 0; phase ^= 1; phase_flipped ^= 1; }
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, phase_flipped);
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
                    if (stage == 2) { stage = 0; phase ^= 1; phase_flipped ^= 1; }
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, phase_flipped);
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
                    if (stage == 2) { stage = 0; phase ^= 1; phase_flipped ^= 1; }
                    mbarrier_wait(consumed_addr + (stage + stream_u * 2) * 8, phase_flipped);
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
                    if (stage == 2) { stage = 0; phase ^= 1; phase_flipped ^= 1; }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("barrier.sync 0, 320;" ::: "memory");
        }
    // ---- Role: consumers ----
    } else if (warp >= 2 && warp <= 9) {
        { // consumers_main
            int warp_id_in_role_1 = (warp - 2);
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
            float2 _f2_0 = make_float2(_vec_load_0[0], _vec_load_0[1]);
            float2 _f2_1 = make_float2(_vec_load_1[0], _vec_load_1[1]);
            float2 q_pair;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
            q_cache[0] = q_pair.x;
            q_cache[1] = q_pair.y;
            float2 _f2_2 = make_float2(_vec_load_0[2], _vec_load_0[3]);
            float2 _f2_3 = make_float2(_vec_load_1[2], _vec_load_1[3]);
            float2 q_pair_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_0) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_3));
            q_cache[2] = q_pair_0.x;
            q_cache[3] = q_pair_0.y;
            float2 _f2_4 = make_float2(_vec_load_0[4], _vec_load_0[5]);
            float2 _f2_5 = make_float2(_vec_load_1[4], _vec_load_1[5]);
            float2 q_pair_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_1) : "l"(*(const unsigned long long*)&_f2_4), "l"(*(const unsigned long long*)&_f2_5));
            q_cache[4] = q_pair_1.x;
            q_cache[5] = q_pair_1.y;
            float2 _f2_6 = make_float2(_vec_load_0[6], _vec_load_0[7]);
            float2 _f2_7 = make_float2(_vec_load_1[6], _vec_load_1[7]);
            float2 q_pair_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_2) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_7));
            q_cache[6] = q_pair_2.x;
            q_cache[7] = q_pair_2.y;
            int hidden_base_3 = 2048 + thread_in_group * 8;
            float _vec_load_2[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_3);
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
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_3);
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
            float2 _f2_8 = make_float2(_vec_load_2[0], _vec_load_2[1]);
            float2 _f2_9 = make_float2(_vec_load_3[0], _vec_load_3[1]);
            float2 q_pair_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_4) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_9));
            q_cache[8] = q_pair_4.x;
            q_cache[9] = q_pair_4.y;
            float2 _f2_10 = make_float2(_vec_load_2[2], _vec_load_2[3]);
            float2 _f2_11 = make_float2(_vec_load_3[2], _vec_load_3[3]);
            float2 q_pair_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_5) : "l"(*(const unsigned long long*)&_f2_10), "l"(*(const unsigned long long*)&_f2_11));
            q_cache[10] = q_pair_5.x;
            q_cache[11] = q_pair_5.y;
            float2 _f2_12 = make_float2(_vec_load_2[4], _vec_load_2[5]);
            float2 _f2_13 = make_float2(_vec_load_3[4], _vec_load_3[5]);
            float2 q_pair_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_6) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_13));
            q_cache[12] = q_pair_6.x;
            q_cache[13] = q_pair_6.y;
            float2 _f2_14 = make_float2(_vec_load_2[6], _vec_load_2[7]);
            float2 _f2_15 = make_float2(_vec_load_3[6], _vec_load_3[7]);
            float2 q_pair_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_7) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_15));
            q_cache[14] = q_pair_7.x;
            q_cache[15] = q_pair_7.y;
            int hidden_base_8 = 4096 + thread_in_group * 8;
            float _vec_load_4[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_8);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_4[_pair]));
                    }
                }
            }
            float _vec_load_5[8];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_8);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_5[_pair]));
                    }
                }
            }
            float2 _f2_16 = make_float2(_vec_load_4[0], _vec_load_4[1]);
            float2 _f2_17 = make_float2(_vec_load_5[0], _vec_load_5[1]);
            float2 q_pair_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_9) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_17));
            q_cache[16] = q_pair_9.x;
            q_cache[17] = q_pair_9.y;
            float2 _f2_18 = make_float2(_vec_load_4[2], _vec_load_4[3]);
            float2 _f2_19 = make_float2(_vec_load_5[2], _vec_load_5[3]);
            float2 q_pair_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_10) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_19));
            q_cache[18] = q_pair_10.x;
            q_cache[19] = q_pair_10.y;
            float2 _f2_20 = make_float2(_vec_load_4[4], _vec_load_4[5]);
            float2 _f2_21 = make_float2(_vec_load_5[4], _vec_load_5[5]);
            float2 q_pair_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_11) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&_f2_21));
            q_cache[20] = q_pair_11.x;
            q_cache[21] = q_pair_11.y;
            float2 _f2_22 = make_float2(_vec_load_4[6], _vec_load_4[7]);
            float2 _f2_23 = make_float2(_vec_load_5[6], _vec_load_5[7]);
            float2 q_pair_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_12) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&_f2_23));
            q_cache[22] = q_pair_12.x;
            q_cache[23] = q_pair_12.y;
            int hidden_base_13 = 6144 + thread_in_group * 4;
            float _vec_load_6[4];
            {
                uint2 _vld_6;
                _vld_6 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base_13);
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_6[0 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
            float _vec_load_7[4];
            {
                uint2 _vld_7;
                _vld_7 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base_13);
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_7[0 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
            float2 _f2_24 = make_float2(_vec_load_6[0], _vec_load_6[1]);
            float2 _f2_25 = make_float2(_vec_load_7[0], _vec_load_7[1]);
            float2 q_pair_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_14) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_25));
            q_cache[24] = q_pair_14.x;
            q_cache[25] = q_pair_14.y;
            float2 _f2_26 = make_float2(_vec_load_6[2], _vec_load_6[3]);
            float2 _f2_27 = make_float2(_vec_load_7[2], _vec_load_7[3]);
            float2 q_pair_15;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_15) : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&_f2_27));
            q_cache[26] = q_pair_15.x;
            q_cache[27] = q_pair_15.y;
            int hidden_base_16 = 1024 + thread_in_group * 8;
            float _vec_load_8[8];
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_16);
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
            float _vec_load_9[8];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_16);
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
            float2 _f2_28 = make_float2(_vec_load_8[0], _vec_load_8[1]);
            float2 _f2_29 = make_float2(_vec_load_9[0], _vec_load_9[1]);
            float2 q_pair_17;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_17) : "l"(*(const unsigned long long*)&_f2_28), "l"(*(const unsigned long long*)&_f2_29));
            q_cache[28] = q_pair_17.x;
            q_cache[29] = q_pair_17.y;
            float2 _f2_30 = make_float2(_vec_load_8[2], _vec_load_8[3]);
            float2 _f2_31 = make_float2(_vec_load_9[2], _vec_load_9[3]);
            float2 q_pair_18;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_18) : "l"(*(const unsigned long long*)&_f2_30), "l"(*(const unsigned long long*)&_f2_31));
            q_cache[30] = q_pair_18.x;
            q_cache[31] = q_pair_18.y;
            float2 _f2_32 = make_float2(_vec_load_8[4], _vec_load_8[5]);
            float2 _f2_33 = make_float2(_vec_load_9[4], _vec_load_9[5]);
            float2 q_pair_19;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_19) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&_f2_33));
            q_cache[32] = q_pair_19.x;
            q_cache[33] = q_pair_19.y;
            float2 _f2_34 = make_float2(_vec_load_8[6], _vec_load_8[7]);
            float2 _f2_35 = make_float2(_vec_load_9[6], _vec_load_9[7]);
            float2 q_pair_20;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_20) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&_f2_35));
            q_cache[34] = q_pair_20.x;
            q_cache[35] = q_pair_20.y;
            int hidden_base_21 = 3072 + thread_in_group * 8;
            float _vec_load_10[8];
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_21);
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
                            : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_10[_pair]));
                    }
                }
            }
            float _vec_load_11[8];
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_21);
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
                            : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_11[_pair]));
                    }
                }
            }
            float2 _f2_36 = make_float2(_vec_load_10[0], _vec_load_10[1]);
            float2 _f2_37 = make_float2(_vec_load_11[0], _vec_load_11[1]);
            float2 q_pair_22;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_22) : "l"(*(const unsigned long long*)&_f2_36), "l"(*(const unsigned long long*)&_f2_37));
            q_cache[36] = q_pair_22.x;
            q_cache[37] = q_pair_22.y;
            float2 _f2_38 = make_float2(_vec_load_10[2], _vec_load_10[3]);
            float2 _f2_39 = make_float2(_vec_load_11[2], _vec_load_11[3]);
            float2 q_pair_23;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_23) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&_f2_39));
            q_cache[38] = q_pair_23.x;
            q_cache[39] = q_pair_23.y;
            float2 _f2_40 = make_float2(_vec_load_10[4], _vec_load_10[5]);
            float2 _f2_41 = make_float2(_vec_load_11[4], _vec_load_11[5]);
            float2 q_pair_24;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_24) : "l"(*(const unsigned long long*)&_f2_40), "l"(*(const unsigned long long*)&_f2_41));
            q_cache[40] = q_pair_24.x;
            q_cache[41] = q_pair_24.y;
            float2 _f2_42 = make_float2(_vec_load_10[6], _vec_load_10[7]);
            float2 _f2_43 = make_float2(_vec_load_11[6], _vec_load_11[7]);
            float2 q_pair_25;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_25) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&_f2_43));
            q_cache[42] = q_pair_25.x;
            q_cache[43] = q_pair_25.y;
            int hidden_base_26 = 5120 + thread_in_group * 8;
            float _vec_load_12[8];
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(norm_weight + hidden_base_26);
                uint4 _vld_12[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_12[_blk] = _vptr_12[_blk];
                    uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_12[_pair]));
                    }
                }
            }
            float _vec_load_13[8];
            {
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(qk_weight + hidden_base_26);
                uint4 _vld_13[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_13[_blk] = _vptr_13[_blk];
                    uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_13[_pair]));
                    }
                }
            }
            float2 _f2_44 = make_float2(_vec_load_12[0], _vec_load_12[1]);
            float2 _f2_45 = make_float2(_vec_load_13[0], _vec_load_13[1]);
            float2 q_pair_27;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_27) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&_f2_45));
            q_cache[44] = q_pair_27.x;
            q_cache[45] = q_pair_27.y;
            float2 _f2_46 = make_float2(_vec_load_12[2], _vec_load_12[3]);
            float2 _f2_47 = make_float2(_vec_load_13[2], _vec_load_13[3]);
            float2 q_pair_28;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_28) : "l"(*(const unsigned long long*)&_f2_46), "l"(*(const unsigned long long*)&_f2_47));
            q_cache[46] = q_pair_28.x;
            q_cache[47] = q_pair_28.y;
            float2 _f2_48 = make_float2(_vec_load_12[4], _vec_load_12[5]);
            float2 _f2_49 = make_float2(_vec_load_13[4], _vec_load_13[5]);
            float2 q_pair_29;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_29) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&_f2_49));
            q_cache[48] = q_pair_29.x;
            q_cache[49] = q_pair_29.y;
            float2 _f2_50 = make_float2(_vec_load_12[6], _vec_load_12[7]);
            float2 _f2_51 = make_float2(_vec_load_13[6], _vec_load_13[7]);
            float2 q_pair_30;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_30) : "l"(*(const unsigned long long*)&_f2_50), "l"(*(const unsigned long long*)&_f2_51));
            q_cache[50] = q_pair_30.x;
            q_cache[51] = q_pair_30.y;
            int hidden_base_31 = 6656 + thread_in_group * 4;
            float _vec_load_14[4];
            {
                uint2 _vld_14;
                _vld_14 = *reinterpret_cast<const uint2*>(norm_weight + hidden_base_31);
                uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_14[0 + _pair * 2])[0]), "=f"((&_vec_load_14[0 + _pair * 2])[1])
                        : "r"(_vpairs_14[_pair]));
                }
            }
            float _vec_load_15[4];
            {
                uint2 _vld_15;
                _vld_15 = *reinterpret_cast<const uint2*>(qk_weight + hidden_base_31);
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_15[0 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
            float2 _f2_52 = make_float2(_vec_load_14[0], _vec_load_14[1]);
            float2 _f2_53 = make_float2(_vec_load_15[0], _vec_load_15[1]);
            float2 q_pair_32;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_32) : "l"(*(const unsigned long long*)&_f2_52), "l"(*(const unsigned long long*)&_f2_53));
            q_cache[52] = q_pair_32.x;
            q_cache[53] = q_pair_32.y;
            float2 _f2_54 = make_float2(_vec_load_14[2], _vec_load_14[3]);
            float2 _f2_55 = make_float2(_vec_load_15[2], _vec_load_15[3]);
            float2 q_pair_33;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_33) : "l"(*(const unsigned long long*)&_f2_54), "l"(*(const unsigned long long*)&_f2_55));
            q_cache[54] = q_pair_33.x;
            q_cache[55] = q_pair_33.y;
            unsigned int consumer_stage = 0;
            unsigned int consumer_phase = 0;
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
                float m_running = -3.4028234663852886e+38f;
                float s_running = 0.0f;
                unsigned int ring_stage = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage) * 8, consumer_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq[4];
                float2 local_dot[4];
                float2 _f2_56 = make_float2(0.0f, 0.0f);
                local_sum_sq[0] = _f2_56;
                float2 _f2_57 = make_float2(0.0f, 0.0f);
                local_dot[0] = _f2_57;
                float2 _f2_58 = make_float2(0.0f, 0.0f);
                local_sum_sq[1] = _f2_58;
                float2 _f2_59 = make_float2(0.0f, 0.0f);
                local_dot[1] = _f2_59;
                float2 _f2_60 = make_float2(0.0f, 0.0f);
                local_sum_sq[2] = _f2_60;
                float2 _f2_61 = make_float2(0.0f, 0.0f);
                local_dot[2] = _f2_61;
                float2 _f2_62 = make_float2(0.0f, 0.0f);
                local_sum_sq[3] = _f2_62;
                float2 _f2_63 = make_float2(0.0f, 0.0f);
                local_dot[3] = _f2_63;
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
                float2 _f2_64 = make_float2(values[0], values[1]);
                float2 value_pair = _f2_64;
                float2 _f2_65 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_3 = _f2_65;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair, value_pair, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair, q_pair_3, local_dot[0]);
                float2 _f2_66 = make_float2(values[2], values[3]);
                float2 value_pair_4 = _f2_66;
                float2 _f2_67 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_8 = _f2_67;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_4, value_pair_4, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_4, q_pair_8, local_dot[0]);
                float2 _f2_68 = make_float2(values[4], values[5]);
                float2 value_pair_9 = _f2_68;
                float2 _f2_69 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_13 = _f2_69;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_9, value_pair_9, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_9, q_pair_13, local_dot[0]);
                float2 _f2_70 = make_float2(values[6], values[7]);
                float2 value_pair_14 = _f2_70;
                float2 _f2_71 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_16 = _f2_71;
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
                float2 _f2_72 = make_float2(values_18[0], values_18[1]);
                float2 value_pair_19 = _f2_72;
                float2 _f2_73 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_21 = _f2_73;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_19, value_pair_19, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_19, q_pair_21, local_dot[1]);
                float2 _f2_74 = make_float2(values_18[2], values_18[3]);
                float2 value_pair_22 = _f2_74;
                float2 _f2_75 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_26 = _f2_75;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_22, value_pair_22, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_22, q_pair_26, local_dot[1]);
                float2 _f2_76 = make_float2(values_18[4], values_18[5]);
                float2 value_pair_27 = _f2_76;
                float2 _f2_77 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_31 = _f2_77;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_27, value_pair_27, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_27, q_pair_31, local_dot[1]);
                float2 _f2_78 = make_float2(values_18[6], values_18[7]);
                float2 value_pair_32 = _f2_78;
                float2 _f2_79 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_34 = _f2_79;
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
                float2 _f2_80 = make_float2(values_37[0], values_37[1]);
                float2 value_pair_38 = _f2_80;
                float2 _f2_81 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_39 = _f2_81;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_38, value_pair_38, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_38, q_pair_39, local_dot[0]);
                float2 _f2_82 = make_float2(values_37[2], values_37[3]);
                float2 value_pair_40 = _f2_82;
                float2 _f2_83 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_41 = _f2_83;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_40, value_pair_40, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_40, q_pair_41, local_dot[0]);
                float2 _f2_84 = make_float2(values_37[4], values_37[5]);
                float2 value_pair_42 = _f2_84;
                float2 _f2_85 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_43 = _f2_85;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_42, value_pair_42, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_42, q_pair_43, local_dot[0]);
                float2 _f2_86 = make_float2(values_37[6], values_37[7]);
                float2 value_pair_44 = _f2_86;
                float2 _f2_87 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_45 = _f2_87;
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
                float2 _f2_88 = make_float2(values_47[0], values_47[1]);
                float2 value_pair_48 = _f2_88;
                float2 _f2_89 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_49 = _f2_89;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_48, value_pair_48, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_48, q_pair_49, local_dot[1]);
                float2 _f2_90 = make_float2(values_47[2], values_47[3]);
                float2 value_pair_50 = _f2_90;
                float2 _f2_91 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_51 = _f2_91;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_50, value_pair_50, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_50, q_pair_51, local_dot[1]);
                float2 _f2_92 = make_float2(values_47[4], values_47[5]);
                float2 value_pair_52 = _f2_92;
                float2 _f2_93 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_53 = _f2_93;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_52, value_pair_52, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_52, q_pair_53, local_dot[1]);
                float2 _f2_94 = make_float2(values_47[6], values_47[7]);
                float2 value_pair_54 = _f2_94;
                float2 _f2_95 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_55 = _f2_95;
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
                float2 _f2_96 = make_float2(values_58[0], values_58[1]);
                float2 value_pair_59 = _f2_96;
                float2 _f2_97 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_60 = _f2_97;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_59, value_pair_59, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_59, q_pair_60, local_dot[0]);
                float2 _f2_98 = make_float2(values_58[2], values_58[3]);
                float2 value_pair_61 = _f2_98;
                float2 _f2_99 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_62 = _f2_99;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_61, value_pair_61, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_61, q_pair_62, local_dot[0]);
                float2 _f2_100 = make_float2(values_58[4], values_58[5]);
                float2 value_pair_63 = _f2_100;
                float2 _f2_101 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_64 = _f2_101;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_63, value_pair_63, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_63, q_pair_64, local_dot[0]);
                float2 _f2_102 = make_float2(values_58[6], values_58[7]);
                float2 value_pair_65 = _f2_102;
                float2 _f2_103 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_66 = _f2_103;
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
                float2 _f2_104 = make_float2(values_68[0], values_68[1]);
                float2 value_pair_69 = _f2_104;
                float2 _f2_105 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_70 = _f2_105;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_69, value_pair_69, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_69, q_pair_70, local_dot[1]);
                float2 _f2_106 = make_float2(values_68[2], values_68[3]);
                float2 value_pair_71 = _f2_106;
                float2 _f2_107 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_72 = _f2_107;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_71, value_pair_71, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_71, q_pair_72, local_dot[1]);
                float2 _f2_108 = make_float2(values_68[4], values_68[5]);
                float2 value_pair_73 = _f2_108;
                float2 _f2_109 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_74 = _f2_109;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_73, value_pair_73, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_73, q_pair_74, local_dot[1]);
                float2 _f2_110 = make_float2(values_68[6], values_68[7]);
                float2 value_pair_75 = _f2_110;
                float2 _f2_111 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_76 = _f2_111;
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
                float2 _f2_112 = make_float2(values_79[0], values_79[1]);
                float2 value_pair_80 = _f2_112;
                float2 _f2_113 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_81 = _f2_113;
                local_sum_sq[0] = fma_f32x2_rn_ftz(value_pair_80, value_pair_80, local_sum_sq[0]);
                local_dot[0] = fma_f32x2_rn_ftz(value_pair_80, q_pair_81, local_dot[0]);
                float2 _f2_114 = make_float2(values_79[2], values_79[3]);
                float2 value_pair_82 = _f2_114;
                float2 _f2_115 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_83 = _f2_115;
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
                float2 _f2_116 = make_float2(values_85[0], values_85[1]);
                float2 value_pair_86 = _f2_116;
                float2 _f2_117 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_87 = _f2_117;
                local_sum_sq[1] = fma_f32x2_rn_ftz(value_pair_86, value_pair_86, local_sum_sq[1]);
                local_dot[1] = fma_f32x2_rn_ftz(value_pair_86, q_pair_87, local_dot[1]);
                float2 _f2_118 = make_float2(values_85[2], values_85[3]);
                float2 value_pair_88 = _f2_118;
                float2 _f2_119 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_89 = _f2_119;
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
                float2 _f2_120 = make_float2(values_92[0], values_92[1]);
                float2 value_pair_93 = _f2_120;
                float2 _f2_121 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_94 = _f2_121;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_93, value_pair_93, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_93, q_pair_94, local_dot[2]);
                float2 _f2_122 = make_float2(values_92[2], values_92[3]);
                float2 value_pair_95 = _f2_122;
                float2 _f2_123 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_96 = _f2_123;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_95, value_pair_95, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_95, q_pair_96, local_dot[2]);
                float2 _f2_124 = make_float2(values_92[4], values_92[5]);
                float2 value_pair_97 = _f2_124;
                float2 _f2_125 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_98 = _f2_125;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_97, value_pair_97, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_97, q_pair_98, local_dot[2]);
                float2 _f2_126 = make_float2(values_92[6], values_92[7]);
                float2 value_pair_99 = _f2_126;
                float2 _f2_127 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_100 = _f2_127;
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
                float2 _f2_128 = make_float2(values_102[0], values_102[1]);
                float2 value_pair_103 = _f2_128;
                float2 _f2_129 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_104 = _f2_129;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_103, value_pair_103, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_103, q_pair_104, local_dot[3]);
                float2 _f2_130 = make_float2(values_102[2], values_102[3]);
                float2 value_pair_105 = _f2_130;
                float2 _f2_131 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_106 = _f2_131;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_105, value_pair_105, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_105, q_pair_106, local_dot[3]);
                float2 _f2_132 = make_float2(values_102[4], values_102[5]);
                float2 value_pair_107 = _f2_132;
                float2 _f2_133 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_108 = _f2_133;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_107, value_pair_107, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_107, q_pair_108, local_dot[3]);
                float2 _f2_134 = make_float2(values_102[6], values_102[7]);
                float2 value_pair_109 = _f2_134;
                float2 _f2_135 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_110 = _f2_135;
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
                float2 _f2_136 = make_float2(values_113[0], values_113[1]);
                float2 value_pair_114 = _f2_136;
                float2 _f2_137 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_115 = _f2_137;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_114, value_pair_114, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_114, q_pair_115, local_dot[2]);
                float2 _f2_138 = make_float2(values_113[2], values_113[3]);
                float2 value_pair_116 = _f2_138;
                float2 _f2_139 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_117 = _f2_139;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_116, value_pair_116, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_116, q_pair_117, local_dot[2]);
                float2 _f2_140 = make_float2(values_113[4], values_113[5]);
                float2 value_pair_118 = _f2_140;
                float2 _f2_141 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_119 = _f2_141;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_118, value_pair_118, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_118, q_pair_119, local_dot[2]);
                float2 _f2_142 = make_float2(values_113[6], values_113[7]);
                float2 value_pair_120 = _f2_142;
                float2 _f2_143 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_121 = _f2_143;
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
                float2 _f2_144 = make_float2(values_123[0], values_123[1]);
                float2 value_pair_124 = _f2_144;
                float2 _f2_145 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_125 = _f2_145;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_124, value_pair_124, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_124, q_pair_125, local_dot[3]);
                float2 _f2_146 = make_float2(values_123[2], values_123[3]);
                float2 value_pair_126 = _f2_146;
                float2 _f2_147 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_127 = _f2_147;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_126, value_pair_126, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_126, q_pair_127, local_dot[3]);
                float2 _f2_148 = make_float2(values_123[4], values_123[5]);
                float2 value_pair_128 = _f2_148;
                float2 _f2_149 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_129 = _f2_149;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_128, value_pair_128, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_128, q_pair_129, local_dot[3]);
                float2 _f2_150 = make_float2(values_123[6], values_123[7]);
                float2 value_pair_130 = _f2_150;
                float2 _f2_151 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_131 = _f2_151;
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
                float2 _f2_152 = make_float2(values_134[0], values_134[1]);
                float2 value_pair_135 = _f2_152;
                float2 _f2_153 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_136 = _f2_153;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_135, value_pair_135, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_135, q_pair_136, local_dot[2]);
                float2 _f2_154 = make_float2(values_134[2], values_134[3]);
                float2 value_pair_137 = _f2_154;
                float2 _f2_155 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_138 = _f2_155;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_137, value_pair_137, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_137, q_pair_138, local_dot[2]);
                float2 _f2_156 = make_float2(values_134[4], values_134[5]);
                float2 value_pair_139 = _f2_156;
                float2 _f2_157 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_140 = _f2_157;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_139, value_pair_139, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_139, q_pair_140, local_dot[2]);
                float2 _f2_158 = make_float2(values_134[6], values_134[7]);
                float2 value_pair_141 = _f2_158;
                float2 _f2_159 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_142 = _f2_159;
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
                float2 _f2_160 = make_float2(values_144[0], values_144[1]);
                float2 value_pair_145 = _f2_160;
                float2 _f2_161 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_146 = _f2_161;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_145, value_pair_145, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_145, q_pair_146, local_dot[3]);
                float2 _f2_162 = make_float2(values_144[2], values_144[3]);
                float2 value_pair_147 = _f2_162;
                float2 _f2_163 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_148 = _f2_163;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_147, value_pair_147, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_147, q_pair_148, local_dot[3]);
                float2 _f2_164 = make_float2(values_144[4], values_144[5]);
                float2 value_pair_149 = _f2_164;
                float2 _f2_165 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_150 = _f2_165;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_149, value_pair_149, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_149, q_pair_150, local_dot[3]);
                float2 _f2_166 = make_float2(values_144[6], values_144[7]);
                float2 value_pair_151 = _f2_166;
                float2 _f2_167 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_152 = _f2_167;
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
                float2 _f2_168 = make_float2(values_155[0], values_155[1]);
                float2 value_pair_156 = _f2_168;
                float2 _f2_169 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_157 = _f2_169;
                local_sum_sq[2] = fma_f32x2_rn_ftz(value_pair_156, value_pair_156, local_sum_sq[2]);
                local_dot[2] = fma_f32x2_rn_ftz(value_pair_156, q_pair_157, local_dot[2]);
                float2 _f2_170 = make_float2(values_155[2], values_155[3]);
                float2 value_pair_158 = _f2_170;
                float2 _f2_171 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_159 = _f2_171;
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
                float2 _f2_172 = make_float2(values_161[0], values_161[1]);
                float2 value_pair_162 = _f2_172;
                float2 _f2_173 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_163 = _f2_173;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_162, value_pair_162, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_162, q_pair_163, local_dot[3]);
                float2 _f2_174 = make_float2(values_161[2], values_161[3]);
                float2 value_pair_164 = _f2_174;
                float2 _f2_175 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_165 = _f2_175;
                local_sum_sq[3] = fma_f32x2_rn_ftz(value_pair_164, value_pair_164, local_sum_sq[3]);
                local_dot[3] = fma_f32x2_rn_ftz(value_pair_164, q_pair_165, local_dot[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base = stats_stream_base;
                float2 _f2_176 = make_float2(local_sum_sq[0].x + local_sum_sq[0].y, local_dot[0].x + local_dot[0].y);
                float2 reduce_pair = _f2_176;
                unsigned long long packed_pair = 0;
                unsigned long long peer_packed = 0;
                float2 _f2_177 = make_float2(0.0f, 0.0f);
                float2 peer_pair = _f2_177;
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 16);
                peer_packed = _shfl_xor_0;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 8);
                peer_packed = _shfl_xor_1;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 4);
                peer_packed = _shfl_xor_2;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 2);
                peer_packed = _shfl_xor_3;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                packed_pair = reinterpret_cast<unsigned long long*>(&reduce_pair)[0];
                unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, packed_pair, 1);
                peer_packed = _shfl_xor_4;
                peer_pair = reinterpret_cast<float2*>(&peer_packed)[0];
                reduce_pair = add_f32x2(reduce_pair, peer_pair);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair.x;
                    stats[slot_base + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair.y;
                }
                float2 _f2_178 = make_float2(local_sum_sq[1].x + local_sum_sq[1].y, local_dot[1].x + local_dot[1].y);
                float2 reduce_pair_166 = _f2_178;
                unsigned long long packed_pair_167 = 0;
                unsigned long long peer_packed_168 = 0;
                float2 _f2_179 = make_float2(0.0f, 0.0f);
                float2 peer_pair_169 = _f2_179;
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 16);
                peer_packed_168 = _shfl_xor_5;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 8);
                peer_packed_168 = _shfl_xor_6;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 4);
                peer_packed_168 = _shfl_xor_7;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 2);
                peer_packed_168 = _shfl_xor_8;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                packed_pair_167 = reinterpret_cast<unsigned long long*>(&reduce_pair_166)[0];
                unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_167, 1);
                peer_packed_168 = _shfl_xor_9;
                peer_pair_169 = reinterpret_cast<float2*>(&peer_packed_168)[0];
                reduce_pair_166 = add_f32x2(reduce_pair_166, peer_pair_169);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_166.x;
                    stats[slot_base + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_166.y;
                }
                float2 _f2_180 = make_float2(local_sum_sq[2].x + local_sum_sq[2].y, local_dot[2].x + local_dot[2].y);
                float2 reduce_pair_170 = _f2_180;
                unsigned long long packed_pair_171 = 0;
                unsigned long long peer_packed_172 = 0;
                float2 _f2_181 = make_float2(0.0f, 0.0f);
                float2 peer_pair_173 = _f2_181;
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 16);
                peer_packed_172 = _shfl_xor_10;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 8);
                peer_packed_172 = _shfl_xor_11;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 4);
                peer_packed_172 = _shfl_xor_12;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 2);
                peer_packed_172 = _shfl_xor_13;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                packed_pair_171 = reinterpret_cast<unsigned long long*>(&reduce_pair_170)[0];
                unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_171, 1);
                peer_packed_172 = _shfl_xor_14;
                peer_pair_173 = reinterpret_cast<float2*>(&peer_packed_172)[0];
                reduce_pair_170 = add_f32x2(reduce_pair_170, peer_pair_173);
                if (lane == 0) {
                    stats[slot_base + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_170.x;
                    stats[slot_base + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_170.y;
                }
                float2 _f2_182 = make_float2(local_sum_sq[3].x + local_sum_sq[3].y, local_dot[3].x + local_dot[3].y);
                float2 reduce_pair_174 = _f2_182;
                unsigned long long packed_pair_175 = 0;
                unsigned long long peer_packed_176 = 0;
                float2 _f2_183 = make_float2(0.0f, 0.0f);
                float2 peer_pair_177 = _f2_183;
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 16);
                peer_packed_176 = _shfl_xor_15;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 8);
                peer_packed_176 = _shfl_xor_16;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 4);
                peer_packed_176 = _shfl_xor_17;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 2);
                peer_packed_176 = _shfl_xor_18;
                peer_pair_177 = reinterpret_cast<float2*>(&peer_packed_176)[0];
                reduce_pair_174 = add_f32x2(reduce_pair_174, peer_pair_177);
                packed_pair_175 = reinterpret_cast<unsigned long long*>(&reduce_pair_174)[0];
                unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_175, 1);
                peer_packed_176 = _shfl_xor_19;
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
                if (stat_source < 2 && stat_warp == 0) {
                    float _rsqrt_0 = rsqrtf(total_sum_sq / 7168.0f + eps);
                    local_logit = total_dot * _rsqrt_0;
                }
                float logits[2];
                logits[0] = 0.0f;
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, local_logit, 0);
                logits[0] = _shfl_0;
                logits[1] = 0.0f;
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, local_logit, 8);
                logits[1] = _shfl_1;
                float weights[2];
                float new_max = 0.0f;
                float correction = 0.0f;
                float weight_sum = 0.0f;
                weights[0] = 0.0f;
                weights[1] = 0.0f;
                float chunk_max = -3.4028234663852886e+38f;
                float _max_0 = max_noftz(chunk_max, logits[0]);
                chunk_max = _max_0;
                float _max_1 = max_noftz(chunk_max, logits[1]);
                chunk_max = _max_1;
                float _max_2 = max_noftz(m_running, chunk_max);
                new_max = _max_2;
                float _exp2_0 = approx_exp2((m_running - new_max) * 1.4426950408889634f);
                correction = _exp2_0;
                float _exp2_1 = approx_exp2((logits[0] - new_max) * 1.4426950408889634f);
                weights[0] = _exp2_1;
                weight_sum += weights[0];
                float _exp2_2 = approx_exp2((logits[1] - new_max) * 1.4426950408889634f);
                weights[1] = _exp2_2;
                weight_sum += weights[1];
                float2 _f2_184 = make_float2(correction, correction);
                float2 correction_pair = _f2_184;
                float2 _f2_185 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair = _f2_185;
                float2 scaled_pair;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&acc_pair), "l"(*(const unsigned long long*)&correction_pair));
                acc32[0] = scaled_pair.x;
                acc32[1] = scaled_pair.y;
                float2 _f2_186 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_178 = _f2_186;
                float2 scaled_pair_179;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_179) : "l"(*(const unsigned long long*)&acc_pair_178), "l"(*(const unsigned long long*)&correction_pair));
                acc32[2] = scaled_pair_179.x;
                acc32[3] = scaled_pair_179.y;
                float2 _f2_187 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_180 = _f2_187;
                float2 scaled_pair_181;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_181) : "l"(*(const unsigned long long*)&acc_pair_180), "l"(*(const unsigned long long*)&correction_pair));
                acc32[4] = scaled_pair_181.x;
                acc32[5] = scaled_pair_181.y;
                float2 _f2_188 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_182 = _f2_188;
                float2 scaled_pair_183;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_183) : "l"(*(const unsigned long long*)&acc_pair_182), "l"(*(const unsigned long long*)&correction_pair));
                acc32[6] = scaled_pair_183.x;
                acc32[7] = scaled_pair_183.y;
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], tmem_base);
                float2 _f2_189 = make_float2(weights[0], weights[0]);
                float2 weight_pair = _f2_189;
                float2 _f2_190 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                float2 cached_pair = _f2_190;
                float2 _f2_191 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_184 = _f2_191;
                float2 updated_pair = fma_f32x2_rn_ftz(weight_pair, cached_pair, acc_pair_184);
                acc32[0] = updated_pair.x;
                acc32[1] = updated_pair.y;
                float2 _f2_192 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
                float2 cached_pair_185 = _f2_192;
                float2 _f2_193 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_186 = _f2_193;
                float2 updated_pair_187 = fma_f32x2_rn_ftz(weight_pair, cached_pair_185, acc_pair_186);
                acc32[2] = updated_pair_187.x;
                acc32[3] = updated_pair_187.y;
                float2 _f2_194 = make_float2(_tmem_load_0[4], _tmem_load_0[5]);
                float2 cached_pair_188 = _f2_194;
                float2 _f2_195 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_189 = _f2_195;
                float2 updated_pair_190 = fma_f32x2_rn_ftz(weight_pair, cached_pair_188, acc_pair_189);
                acc32[4] = updated_pair_190.x;
                acc32[5] = updated_pair_190.y;
                float2 _f2_196 = make_float2(_tmem_load_0[6], _tmem_load_0[7]);
                float2 cached_pair_191 = _f2_196;
                float2 _f2_197 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_192 = _f2_197;
                float2 updated_pair_193 = fma_f32x2_rn_ftz(weight_pair, cached_pair_191, acc_pair_192);
                acc32[6] = updated_pair_193.x;
                acc32[7] = updated_pair_193.y;
                float _tmem_load_1[8];
                tmem_ld_x8(&_tmem_load_1[0], tmem_base + 64);
                float2 _f2_198 = make_float2(weights[1], weights[1]);
                float2 weight_pair_194 = _f2_198;
                float2 _f2_199 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                float2 cached_pair_195 = _f2_199;
                float2 _f2_200 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_196 = _f2_200;
                float2 updated_pair_197 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_195, acc_pair_196);
                acc32[0] = updated_pair_197.x;
                acc32[1] = updated_pair_197.y;
                float2 _f2_201 = make_float2(_tmem_load_1[2], _tmem_load_1[3]);
                float2 cached_pair_198 = _f2_201;
                float2 _f2_202 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_199 = _f2_202;
                float2 updated_pair_200 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_198, acc_pair_199);
                acc32[2] = updated_pair_200.x;
                acc32[3] = updated_pair_200.y;
                float2 _f2_203 = make_float2(_tmem_load_1[4], _tmem_load_1[5]);
                float2 cached_pair_201 = _f2_203;
                float2 _f2_204 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_202 = _f2_204;
                float2 updated_pair_203 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_201, acc_pair_202);
                acc32[4] = updated_pair_203.x;
                acc32[5] = updated_pair_203.y;
                float2 _f2_205 = make_float2(_tmem_load_1[6], _tmem_load_1[7]);
                float2 cached_pair_204 = _f2_205;
                float2 _f2_206 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_205 = _f2_206;
                float2 updated_pair_206 = fma_f32x2_rn_ftz(weight_pair_194, cached_pair_204, acc_pair_205);
                acc32[6] = updated_pair_206.x;
                acc32[7] = updated_pair_206.y;
                float2 _f2_207 = make_float2(correction, correction);
                float2 correction_pair_207 = _f2_207;
                float2 _f2_208 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_208 = _f2_208;
                float2 scaled_pair_209;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_209) : "l"(*(const unsigned long long*)&acc_pair_208), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[8] = scaled_pair_209.x;
                acc32[9] = scaled_pair_209.y;
                float2 _f2_209 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_210 = _f2_209;
                float2 scaled_pair_211;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_211) : "l"(*(const unsigned long long*)&acc_pair_210), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[10] = scaled_pair_211.x;
                acc32[11] = scaled_pair_211.y;
                float2 _f2_210 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_212 = _f2_210;
                float2 scaled_pair_213;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_213) : "l"(*(const unsigned long long*)&acc_pair_212), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[12] = scaled_pair_213.x;
                acc32[13] = scaled_pair_213.y;
                float2 _f2_211 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_214 = _f2_211;
                float2 scaled_pair_215;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_215) : "l"(*(const unsigned long long*)&acc_pair_214), "l"(*(const unsigned long long*)&correction_pair_207));
                acc32[14] = scaled_pair_215.x;
                acc32[15] = scaled_pair_215.y;
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], tmem_base + 8);
                float2 _f2_212 = make_float2(weights[0], weights[0]);
                float2 weight_pair_216 = _f2_212;
                float2 _f2_213 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                float2 cached_pair_217 = _f2_213;
                float2 _f2_214 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_218 = _f2_214;
                float2 updated_pair_219 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_217, acc_pair_218);
                acc32[8] = updated_pair_219.x;
                acc32[9] = updated_pair_219.y;
                float2 _f2_215 = make_float2(_tmem_load_2[2], _tmem_load_2[3]);
                float2 cached_pair_220 = _f2_215;
                float2 _f2_216 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_221 = _f2_216;
                float2 updated_pair_222 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_220, acc_pair_221);
                acc32[10] = updated_pair_222.x;
                acc32[11] = updated_pair_222.y;
                float2 _f2_217 = make_float2(_tmem_load_2[4], _tmem_load_2[5]);
                float2 cached_pair_223 = _f2_217;
                float2 _f2_218 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_224 = _f2_218;
                float2 updated_pair_225 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_223, acc_pair_224);
                acc32[12] = updated_pair_225.x;
                acc32[13] = updated_pair_225.y;
                float2 _f2_219 = make_float2(_tmem_load_2[6], _tmem_load_2[7]);
                float2 cached_pair_226 = _f2_219;
                float2 _f2_220 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_227 = _f2_220;
                float2 updated_pair_228 = fma_f32x2_rn_ftz(weight_pair_216, cached_pair_226, acc_pair_227);
                acc32[14] = updated_pair_228.x;
                acc32[15] = updated_pair_228.y;
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], tmem_base + 64 + 8);
                float2 _f2_221 = make_float2(weights[1], weights[1]);
                float2 weight_pair_229 = _f2_221;
                float2 _f2_222 = make_float2(_tmem_load_3[0], _tmem_load_3[1]);
                float2 cached_pair_230 = _f2_222;
                float2 _f2_223 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_231 = _f2_223;
                float2 updated_pair_232 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_230, acc_pair_231);
                acc32[8] = updated_pair_232.x;
                acc32[9] = updated_pair_232.y;
                float2 _f2_224 = make_float2(_tmem_load_3[2], _tmem_load_3[3]);
                float2 cached_pair_233 = _f2_224;
                float2 _f2_225 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_234 = _f2_225;
                float2 updated_pair_235 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_233, acc_pair_234);
                acc32[10] = updated_pair_235.x;
                acc32[11] = updated_pair_235.y;
                float2 _f2_226 = make_float2(_tmem_load_3[4], _tmem_load_3[5]);
                float2 cached_pair_236 = _f2_226;
                float2 _f2_227 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_237 = _f2_227;
                float2 updated_pair_238 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_236, acc_pair_237);
                acc32[12] = updated_pair_238.x;
                acc32[13] = updated_pair_238.y;
                float2 _f2_228 = make_float2(_tmem_load_3[6], _tmem_load_3[7]);
                float2 cached_pair_239 = _f2_228;
                float2 _f2_229 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_240 = _f2_229;
                float2 updated_pair_241 = fma_f32x2_rn_ftz(weight_pair_229, cached_pair_239, acc_pair_240);
                acc32[14] = updated_pair_241.x;
                acc32[15] = updated_pair_241.y;
                float2 _f2_230 = make_float2(correction, correction);
                float2 correction_pair_242 = _f2_230;
                float2 _f2_231 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_243 = _f2_231;
                float2 scaled_pair_244;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_244) : "l"(*(const unsigned long long*)&acc_pair_243), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[16] = scaled_pair_244.x;
                acc32[17] = scaled_pair_244.y;
                float2 _f2_232 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_245 = _f2_232;
                float2 scaled_pair_246;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_246) : "l"(*(const unsigned long long*)&acc_pair_245), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[18] = scaled_pair_246.x;
                acc32[19] = scaled_pair_246.y;
                float2 _f2_233 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_247 = _f2_233;
                float2 scaled_pair_248;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_248) : "l"(*(const unsigned long long*)&acc_pair_247), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[20] = scaled_pair_248.x;
                acc32[21] = scaled_pair_248.y;
                float2 _f2_234 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_249 = _f2_234;
                float2 scaled_pair_250;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_250) : "l"(*(const unsigned long long*)&acc_pair_249), "l"(*(const unsigned long long*)&correction_pair_242));
                acc32[22] = scaled_pair_250.x;
                acc32[23] = scaled_pair_250.y;
                float _tmem_load_4[8];
                tmem_ld_x8(&_tmem_load_4[0], tmem_base + 16);
                float2 _f2_235 = make_float2(weights[0], weights[0]);
                float2 weight_pair_251 = _f2_235;
                float2 _f2_236 = make_float2(_tmem_load_4[0], _tmem_load_4[1]);
                float2 cached_pair_252 = _f2_236;
                float2 _f2_237 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_253 = _f2_237;
                float2 updated_pair_254 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_252, acc_pair_253);
                acc32[16] = updated_pair_254.x;
                acc32[17] = updated_pair_254.y;
                float2 _f2_238 = make_float2(_tmem_load_4[2], _tmem_load_4[3]);
                float2 cached_pair_255 = _f2_238;
                float2 _f2_239 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_256 = _f2_239;
                float2 updated_pair_257 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_255, acc_pair_256);
                acc32[18] = updated_pair_257.x;
                acc32[19] = updated_pair_257.y;
                float2 _f2_240 = make_float2(_tmem_load_4[4], _tmem_load_4[5]);
                float2 cached_pair_258 = _f2_240;
                float2 _f2_241 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_259 = _f2_241;
                float2 updated_pair_260 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_258, acc_pair_259);
                acc32[20] = updated_pair_260.x;
                acc32[21] = updated_pair_260.y;
                float2 _f2_242 = make_float2(_tmem_load_4[6], _tmem_load_4[7]);
                float2 cached_pair_261 = _f2_242;
                float2 _f2_243 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_262 = _f2_243;
                float2 updated_pair_263 = fma_f32x2_rn_ftz(weight_pair_251, cached_pair_261, acc_pair_262);
                acc32[22] = updated_pair_263.x;
                acc32[23] = updated_pair_263.y;
                float _tmem_load_5[8];
                tmem_ld_x8(&_tmem_load_5[0], tmem_base + 64 + 16);
                float2 _f2_244 = make_float2(weights[1], weights[1]);
                float2 weight_pair_264 = _f2_244;
                float2 _f2_245 = make_float2(_tmem_load_5[0], _tmem_load_5[1]);
                float2 cached_pair_265 = _f2_245;
                float2 _f2_246 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_266 = _f2_246;
                float2 updated_pair_267 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_265, acc_pair_266);
                acc32[16] = updated_pair_267.x;
                acc32[17] = updated_pair_267.y;
                float2 _f2_247 = make_float2(_tmem_load_5[2], _tmem_load_5[3]);
                float2 cached_pair_268 = _f2_247;
                float2 _f2_248 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_269 = _f2_248;
                float2 updated_pair_270 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_268, acc_pair_269);
                acc32[18] = updated_pair_270.x;
                acc32[19] = updated_pair_270.y;
                float2 _f2_249 = make_float2(_tmem_load_5[4], _tmem_load_5[5]);
                float2 cached_pair_271 = _f2_249;
                float2 _f2_250 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_272 = _f2_250;
                float2 updated_pair_273 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_271, acc_pair_272);
                acc32[20] = updated_pair_273.x;
                acc32[21] = updated_pair_273.y;
                float2 _f2_251 = make_float2(_tmem_load_5[6], _tmem_load_5[7]);
                float2 cached_pair_274 = _f2_251;
                float2 _f2_252 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_275 = _f2_252;
                float2 updated_pair_276 = fma_f32x2_rn_ftz(weight_pair_264, cached_pair_274, acc_pair_275);
                acc32[22] = updated_pair_276.x;
                acc32[23] = updated_pair_276.y;
                float2 _f2_253 = make_float2(correction, correction);
                float2 correction_pair_277 = _f2_253;
                float2 _f2_254 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_278 = _f2_254;
                float2 scaled_pair_279;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_279) : "l"(*(const unsigned long long*)&acc_pair_278), "l"(*(const unsigned long long*)&correction_pair_277));
                acc32[24] = scaled_pair_279.x;
                acc32[25] = scaled_pair_279.y;
                float2 _f2_255 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_280 = _f2_255;
                float2 scaled_pair_281;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_281) : "l"(*(const unsigned long long*)&acc_pair_280), "l"(*(const unsigned long long*)&correction_pair_277));
                acc32[26] = scaled_pair_281.x;
                acc32[27] = scaled_pair_281.y;
                float _tmem_load_6[4];
                tmem_ld_x4(&_tmem_load_6[0], tmem_base + 24);
                float2 _f2_256 = make_float2(weights[0], weights[0]);
                float2 weight_pair_282 = _f2_256;
                float2 _f2_257 = make_float2(_tmem_load_6[0], _tmem_load_6[1]);
                float2 cached_pair_283 = _f2_257;
                float2 _f2_258 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_284 = _f2_258;
                float2 updated_pair_285 = fma_f32x2_rn_ftz(weight_pair_282, cached_pair_283, acc_pair_284);
                acc32[24] = updated_pair_285.x;
                acc32[25] = updated_pair_285.y;
                float2 _f2_259 = make_float2(_tmem_load_6[2], _tmem_load_6[3]);
                float2 cached_pair_286 = _f2_259;
                float2 _f2_260 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_287 = _f2_260;
                float2 updated_pair_288 = fma_f32x2_rn_ftz(weight_pair_282, cached_pair_286, acc_pair_287);
                acc32[26] = updated_pair_288.x;
                acc32[27] = updated_pair_288.y;
                float _tmem_load_7[4];
                tmem_ld_x4(&_tmem_load_7[0], tmem_base + 64 + 24);
                float2 _f2_261 = make_float2(weights[1], weights[1]);
                float2 weight_pair_289 = _f2_261;
                float2 _f2_262 = make_float2(_tmem_load_7[0], _tmem_load_7[1]);
                float2 cached_pair_290 = _f2_262;
                float2 _f2_263 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_291 = _f2_263;
                float2 updated_pair_292 = fma_f32x2_rn_ftz(weight_pair_289, cached_pair_290, acc_pair_291);
                acc32[24] = updated_pair_292.x;
                acc32[25] = updated_pair_292.y;
                float2 _f2_264 = make_float2(_tmem_load_7[2], _tmem_load_7[3]);
                float2 cached_pair_293 = _f2_264;
                float2 _f2_265 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_294 = _f2_265;
                float2 updated_pair_295 = fma_f32x2_rn_ftz(weight_pair_289, cached_pair_293, acc_pair_294);
                acc32[26] = updated_pair_295.x;
                acc32[27] = updated_pair_295.y;
                float2 _f2_266 = make_float2(correction, correction);
                float2 correction_pair_296 = _f2_266;
                float2 _f2_267 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_297 = _f2_267;
                float2 scaled_pair_298;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_298) : "l"(*(const unsigned long long*)&acc_pair_297), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[28] = scaled_pair_298.x;
                acc32[29] = scaled_pair_298.y;
                float2 _f2_268 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_299 = _f2_268;
                float2 scaled_pair_300;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_300) : "l"(*(const unsigned long long*)&acc_pair_299), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[30] = scaled_pair_300.x;
                acc32[31] = scaled_pair_300.y;
                float2 _f2_269 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_301 = _f2_269;
                float2 scaled_pair_302;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_302) : "l"(*(const unsigned long long*)&acc_pair_301), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[32] = scaled_pair_302.x;
                acc32[33] = scaled_pair_302.y;
                float2 _f2_270 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_303 = _f2_270;
                float2 scaled_pair_304;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_304) : "l"(*(const unsigned long long*)&acc_pair_303), "l"(*(const unsigned long long*)&correction_pair_296));
                acc32[34] = scaled_pair_304.x;
                acc32[35] = scaled_pair_304.y;
                float _tmem_load_8[8];
                tmem_ld_x8(&_tmem_load_8[0], tmem_base + 32);
                float2 _f2_271 = make_float2(weights[0], weights[0]);
                float2 weight_pair_305 = _f2_271;
                float2 _f2_272 = make_float2(_tmem_load_8[0], _tmem_load_8[1]);
                float2 cached_pair_306 = _f2_272;
                float2 _f2_273 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_307 = _f2_273;
                float2 updated_pair_308 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_306, acc_pair_307);
                acc32[28] = updated_pair_308.x;
                acc32[29] = updated_pair_308.y;
                float2 _f2_274 = make_float2(_tmem_load_8[2], _tmem_load_8[3]);
                float2 cached_pair_309 = _f2_274;
                float2 _f2_275 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_310 = _f2_275;
                float2 updated_pair_311 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_309, acc_pair_310);
                acc32[30] = updated_pair_311.x;
                acc32[31] = updated_pair_311.y;
                float2 _f2_276 = make_float2(_tmem_load_8[4], _tmem_load_8[5]);
                float2 cached_pair_312 = _f2_276;
                float2 _f2_277 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_313 = _f2_277;
                float2 updated_pair_314 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_312, acc_pair_313);
                acc32[32] = updated_pair_314.x;
                acc32[33] = updated_pair_314.y;
                float2 _f2_278 = make_float2(_tmem_load_8[6], _tmem_load_8[7]);
                float2 cached_pair_315 = _f2_278;
                float2 _f2_279 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_316 = _f2_279;
                float2 updated_pair_317 = fma_f32x2_rn_ftz(weight_pair_305, cached_pair_315, acc_pair_316);
                acc32[34] = updated_pair_317.x;
                acc32[35] = updated_pair_317.y;
                float _tmem_load_9[8];
                tmem_ld_x8(&_tmem_load_9[0], tmem_base + 64 + 32);
                float2 _f2_280 = make_float2(weights[1], weights[1]);
                float2 weight_pair_318 = _f2_280;
                float2 _f2_281 = make_float2(_tmem_load_9[0], _tmem_load_9[1]);
                float2 cached_pair_319 = _f2_281;
                float2 _f2_282 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_320 = _f2_282;
                float2 updated_pair_321 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_319, acc_pair_320);
                acc32[28] = updated_pair_321.x;
                acc32[29] = updated_pair_321.y;
                float2 _f2_283 = make_float2(_tmem_load_9[2], _tmem_load_9[3]);
                float2 cached_pair_322 = _f2_283;
                float2 _f2_284 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_323 = _f2_284;
                float2 updated_pair_324 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_322, acc_pair_323);
                acc32[30] = updated_pair_324.x;
                acc32[31] = updated_pair_324.y;
                float2 _f2_285 = make_float2(_tmem_load_9[4], _tmem_load_9[5]);
                float2 cached_pair_325 = _f2_285;
                float2 _f2_286 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_326 = _f2_286;
                float2 updated_pair_327 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_325, acc_pair_326);
                acc32[32] = updated_pair_327.x;
                acc32[33] = updated_pair_327.y;
                float2 _f2_287 = make_float2(_tmem_load_9[6], _tmem_load_9[7]);
                float2 cached_pair_328 = _f2_287;
                float2 _f2_288 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_329 = _f2_288;
                float2 updated_pair_330 = fma_f32x2_rn_ftz(weight_pair_318, cached_pair_328, acc_pair_329);
                acc32[34] = updated_pair_330.x;
                acc32[35] = updated_pair_330.y;
                float2 _f2_289 = make_float2(correction, correction);
                float2 correction_pair_331 = _f2_289;
                float2 _f2_290 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_332 = _f2_290;
                float2 scaled_pair_333;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_333) : "l"(*(const unsigned long long*)&acc_pair_332), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[36] = scaled_pair_333.x;
                acc32[37] = scaled_pair_333.y;
                float2 _f2_291 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_334 = _f2_291;
                float2 scaled_pair_335;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_335) : "l"(*(const unsigned long long*)&acc_pair_334), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[38] = scaled_pair_335.x;
                acc32[39] = scaled_pair_335.y;
                float2 _f2_292 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_336 = _f2_292;
                float2 scaled_pair_337;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_337) : "l"(*(const unsigned long long*)&acc_pair_336), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[40] = scaled_pair_337.x;
                acc32[41] = scaled_pair_337.y;
                float2 _f2_293 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_338 = _f2_293;
                float2 scaled_pair_339;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_339) : "l"(*(const unsigned long long*)&acc_pair_338), "l"(*(const unsigned long long*)&correction_pair_331));
                acc32[42] = scaled_pair_339.x;
                acc32[43] = scaled_pair_339.y;
                float _tmem_load_10[8];
                tmem_ld_x8(&_tmem_load_10[0], tmem_base + 40);
                float2 _f2_294 = make_float2(weights[0], weights[0]);
                float2 weight_pair_340 = _f2_294;
                float2 _f2_295 = make_float2(_tmem_load_10[0], _tmem_load_10[1]);
                float2 cached_pair_341 = _f2_295;
                float2 _f2_296 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_342 = _f2_296;
                float2 updated_pair_343 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_341, acc_pair_342);
                acc32[36] = updated_pair_343.x;
                acc32[37] = updated_pair_343.y;
                float2 _f2_297 = make_float2(_tmem_load_10[2], _tmem_load_10[3]);
                float2 cached_pair_344 = _f2_297;
                float2 _f2_298 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_345 = _f2_298;
                float2 updated_pair_346 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_344, acc_pair_345);
                acc32[38] = updated_pair_346.x;
                acc32[39] = updated_pair_346.y;
                float2 _f2_299 = make_float2(_tmem_load_10[4], _tmem_load_10[5]);
                float2 cached_pair_347 = _f2_299;
                float2 _f2_300 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_348 = _f2_300;
                float2 updated_pair_349 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_347, acc_pair_348);
                acc32[40] = updated_pair_349.x;
                acc32[41] = updated_pair_349.y;
                float2 _f2_301 = make_float2(_tmem_load_10[6], _tmem_load_10[7]);
                float2 cached_pair_350 = _f2_301;
                float2 _f2_302 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_351 = _f2_302;
                float2 updated_pair_352 = fma_f32x2_rn_ftz(weight_pair_340, cached_pair_350, acc_pair_351);
                acc32[42] = updated_pair_352.x;
                acc32[43] = updated_pair_352.y;
                float _tmem_load_11[8];
                tmem_ld_x8(&_tmem_load_11[0], tmem_base + 64 + 40);
                float2 _f2_303 = make_float2(weights[1], weights[1]);
                float2 weight_pair_353 = _f2_303;
                float2 _f2_304 = make_float2(_tmem_load_11[0], _tmem_load_11[1]);
                float2 cached_pair_354 = _f2_304;
                float2 _f2_305 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_355 = _f2_305;
                float2 updated_pair_356 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_354, acc_pair_355);
                acc32[36] = updated_pair_356.x;
                acc32[37] = updated_pair_356.y;
                float2 _f2_306 = make_float2(_tmem_load_11[2], _tmem_load_11[3]);
                float2 cached_pair_357 = _f2_306;
                float2 _f2_307 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_358 = _f2_307;
                float2 updated_pair_359 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_357, acc_pair_358);
                acc32[38] = updated_pair_359.x;
                acc32[39] = updated_pair_359.y;
                float2 _f2_308 = make_float2(_tmem_load_11[4], _tmem_load_11[5]);
                float2 cached_pair_360 = _f2_308;
                float2 _f2_309 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_361 = _f2_309;
                float2 updated_pair_362 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_360, acc_pair_361);
                acc32[40] = updated_pair_362.x;
                acc32[41] = updated_pair_362.y;
                float2 _f2_310 = make_float2(_tmem_load_11[6], _tmem_load_11[7]);
                float2 cached_pair_363 = _f2_310;
                float2 _f2_311 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_364 = _f2_311;
                float2 updated_pair_365 = fma_f32x2_rn_ftz(weight_pair_353, cached_pair_363, acc_pair_364);
                acc32[42] = updated_pair_365.x;
                acc32[43] = updated_pair_365.y;
                float2 _f2_312 = make_float2(correction, correction);
                float2 correction_pair_366 = _f2_312;
                float2 _f2_313 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_367 = _f2_313;
                float2 scaled_pair_368;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_368) : "l"(*(const unsigned long long*)&acc_pair_367), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[44] = scaled_pair_368.x;
                acc32[45] = scaled_pair_368.y;
                float2 _f2_314 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_369 = _f2_314;
                float2 scaled_pair_370;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_370) : "l"(*(const unsigned long long*)&acc_pair_369), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[46] = scaled_pair_370.x;
                acc32[47] = scaled_pair_370.y;
                float2 _f2_315 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_371 = _f2_315;
                float2 scaled_pair_372;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_372) : "l"(*(const unsigned long long*)&acc_pair_371), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[48] = scaled_pair_372.x;
                acc32[49] = scaled_pair_372.y;
                float2 _f2_316 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_373 = _f2_316;
                float2 scaled_pair_374;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_374) : "l"(*(const unsigned long long*)&acc_pair_373), "l"(*(const unsigned long long*)&correction_pair_366));
                acc32[50] = scaled_pair_374.x;
                acc32[51] = scaled_pair_374.y;
                float _tmem_load_12[8];
                tmem_ld_x8(&_tmem_load_12[0], tmem_base + 48);
                float2 _f2_317 = make_float2(weights[0], weights[0]);
                float2 weight_pair_375 = _f2_317;
                float2 _f2_318 = make_float2(_tmem_load_12[0], _tmem_load_12[1]);
                float2 cached_pair_376 = _f2_318;
                float2 _f2_319 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_377 = _f2_319;
                float2 updated_pair_378 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_376, acc_pair_377);
                acc32[44] = updated_pair_378.x;
                acc32[45] = updated_pair_378.y;
                float2 _f2_320 = make_float2(_tmem_load_12[2], _tmem_load_12[3]);
                float2 cached_pair_379 = _f2_320;
                float2 _f2_321 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_380 = _f2_321;
                float2 updated_pair_381 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_379, acc_pair_380);
                acc32[46] = updated_pair_381.x;
                acc32[47] = updated_pair_381.y;
                float2 _f2_322 = make_float2(_tmem_load_12[4], _tmem_load_12[5]);
                float2 cached_pair_382 = _f2_322;
                float2 _f2_323 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_383 = _f2_323;
                float2 updated_pair_384 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_382, acc_pair_383);
                acc32[48] = updated_pair_384.x;
                acc32[49] = updated_pair_384.y;
                float2 _f2_324 = make_float2(_tmem_load_12[6], _tmem_load_12[7]);
                float2 cached_pair_385 = _f2_324;
                float2 _f2_325 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_386 = _f2_325;
                float2 updated_pair_387 = fma_f32x2_rn_ftz(weight_pair_375, cached_pair_385, acc_pair_386);
                acc32[50] = updated_pair_387.x;
                acc32[51] = updated_pair_387.y;
                float _tmem_load_13[8];
                tmem_ld_x8(&_tmem_load_13[0], tmem_base + 64 + 48);
                float2 _f2_326 = make_float2(weights[1], weights[1]);
                float2 weight_pair_388 = _f2_326;
                float2 _f2_327 = make_float2(_tmem_load_13[0], _tmem_load_13[1]);
                float2 cached_pair_389 = _f2_327;
                float2 _f2_328 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_390 = _f2_328;
                float2 updated_pair_391 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_389, acc_pair_390);
                acc32[44] = updated_pair_391.x;
                acc32[45] = updated_pair_391.y;
                float2 _f2_329 = make_float2(_tmem_load_13[2], _tmem_load_13[3]);
                float2 cached_pair_392 = _f2_329;
                float2 _f2_330 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_393 = _f2_330;
                float2 updated_pair_394 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_392, acc_pair_393);
                acc32[46] = updated_pair_394.x;
                acc32[47] = updated_pair_394.y;
                float2 _f2_331 = make_float2(_tmem_load_13[4], _tmem_load_13[5]);
                float2 cached_pair_395 = _f2_331;
                float2 _f2_332 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_396 = _f2_332;
                float2 updated_pair_397 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_395, acc_pair_396);
                acc32[48] = updated_pair_397.x;
                acc32[49] = updated_pair_397.y;
                float2 _f2_333 = make_float2(_tmem_load_13[6], _tmem_load_13[7]);
                float2 cached_pair_398 = _f2_333;
                float2 _f2_334 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_399 = _f2_334;
                float2 updated_pair_400 = fma_f32x2_rn_ftz(weight_pair_388, cached_pair_398, acc_pair_399);
                acc32[50] = updated_pair_400.x;
                acc32[51] = updated_pair_400.y;
                float2 _f2_335 = make_float2(correction, correction);
                float2 correction_pair_401 = _f2_335;
                float2 _f2_336 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_402 = _f2_336;
                float2 scaled_pair_403;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_403) : "l"(*(const unsigned long long*)&acc_pair_402), "l"(*(const unsigned long long*)&correction_pair_401));
                acc32[52] = scaled_pair_403.x;
                acc32[53] = scaled_pair_403.y;
                float2 _f2_337 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_404 = _f2_337;
                float2 scaled_pair_405;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_405) : "l"(*(const unsigned long long*)&acc_pair_404), "l"(*(const unsigned long long*)&correction_pair_401));
                acc32[54] = scaled_pair_405.x;
                acc32[55] = scaled_pair_405.y;
                float _tmem_load_14[4];
                tmem_ld_x4(&_tmem_load_14[0], tmem_base + 56);
                float2 _f2_338 = make_float2(weights[0], weights[0]);
                float2 weight_pair_406 = _f2_338;
                float2 _f2_339 = make_float2(_tmem_load_14[0], _tmem_load_14[1]);
                float2 cached_pair_407 = _f2_339;
                float2 _f2_340 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_408 = _f2_340;
                float2 updated_pair_409 = fma_f32x2_rn_ftz(weight_pair_406, cached_pair_407, acc_pair_408);
                acc32[52] = updated_pair_409.x;
                acc32[53] = updated_pair_409.y;
                float2 _f2_341 = make_float2(_tmem_load_14[2], _tmem_load_14[3]);
                float2 cached_pair_410 = _f2_341;
                float2 _f2_342 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_411 = _f2_342;
                float2 updated_pair_412 = fma_f32x2_rn_ftz(weight_pair_406, cached_pair_410, acc_pair_411);
                acc32[54] = updated_pair_412.x;
                acc32[55] = updated_pair_412.y;
                float _tmem_load_15[4];
                tmem_ld_x4(&_tmem_load_15[0], tmem_base + 64 + 56);
                float2 _f2_343 = make_float2(weights[1], weights[1]);
                float2 weight_pair_413 = _f2_343;
                float2 _f2_344 = make_float2(_tmem_load_15[0], _tmem_load_15[1]);
                float2 cached_pair_414 = _f2_344;
                float2 _f2_345 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_415 = _f2_345;
                float2 updated_pair_416 = fma_f32x2_rn_ftz(weight_pair_413, cached_pair_414, acc_pair_415);
                acc32[52] = updated_pair_416.x;
                acc32[53] = updated_pair_416.y;
                float2 _f2_346 = make_float2(_tmem_load_15[2], _tmem_load_15[3]);
                float2 cached_pair_417 = _f2_346;
                float2 _f2_347 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_418 = _f2_347;
                float2 updated_pair_419 = fma_f32x2_rn_ftz(weight_pair_413, cached_pair_417, acc_pair_418);
                acc32[54] = updated_pair_419.x;
                acc32[55] = updated_pair_419.y;
                s_running = __fmaf_rn(s_running, correction, weight_sum);
                m_running = new_max;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; consumer_phase ^= 1; }
                unsigned int ring_stage_420 = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage_420) * 8, consumer_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq_421[4];
                float2 local_dot_422[4];
                float2 _f2_348 = make_float2(0.0f, 0.0f);
                local_sum_sq_421[0] = _f2_348;
                float2 _f2_349 = make_float2(0.0f, 0.0f);
                local_dot_422[0] = _f2_349;
                float2 _f2_350 = make_float2(0.0f, 0.0f);
                local_sum_sq_421[1] = _f2_350;
                float2 _f2_351 = make_float2(0.0f, 0.0f);
                local_dot_422[1] = _f2_351;
                float2 _f2_352 = make_float2(0.0f, 0.0f);
                local_sum_sq_421[2] = _f2_352;
                float2 _f2_353 = make_float2(0.0f, 0.0f);
                local_dot_422[2] = _f2_353;
                float2 _f2_354 = make_float2(0.0f, 0.0f);
                local_sum_sq_421[3] = _f2_354;
                float2 _f2_355 = make_float2(0.0f, 0.0f);
                local_dot_422[3] = _f2_355;
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
                float2 _f2_356 = make_float2(values_425[0], values_425[1]);
                float2 value_pair_426 = _f2_356;
                float2 _f2_357 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_427 = _f2_357;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_426, value_pair_426, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_426, q_pair_427, local_dot_422[0]);
                float2 _f2_358 = make_float2(values_425[2], values_425[3]);
                float2 value_pair_428 = _f2_358;
                float2 _f2_359 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_429 = _f2_359;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_428, value_pair_428, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_428, q_pair_429, local_dot_422[0]);
                float2 _f2_360 = make_float2(values_425[4], values_425[5]);
                float2 value_pair_430 = _f2_360;
                float2 _f2_361 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_431 = _f2_361;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_430, value_pair_430, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_430, q_pair_431, local_dot_422[0]);
                float2 _f2_362 = make_float2(values_425[6], values_425[7]);
                float2 value_pair_432 = _f2_362;
                float2 _f2_363 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_433 = _f2_363;
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
                float2 _f2_364 = make_float2(values_435[0], values_435[1]);
                float2 value_pair_436 = _f2_364;
                float2 _f2_365 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_437 = _f2_365;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_436, value_pair_436, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_436, q_pair_437, local_dot_422[1]);
                float2 _f2_366 = make_float2(values_435[2], values_435[3]);
                float2 value_pair_438 = _f2_366;
                float2 _f2_367 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_439 = _f2_367;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_438, value_pair_438, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_438, q_pair_439, local_dot_422[1]);
                float2 _f2_368 = make_float2(values_435[4], values_435[5]);
                float2 value_pair_440 = _f2_368;
                float2 _f2_369 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_441 = _f2_369;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_440, value_pair_440, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_440, q_pair_441, local_dot_422[1]);
                float2 _f2_370 = make_float2(values_435[6], values_435[7]);
                float2 value_pair_442 = _f2_370;
                float2 _f2_371 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_443 = _f2_371;
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
                float2 _f2_372 = make_float2(values_446[0], values_446[1]);
                float2 value_pair_447 = _f2_372;
                float2 _f2_373 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_448 = _f2_373;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_447, value_pair_447, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_447, q_pair_448, local_dot_422[0]);
                float2 _f2_374 = make_float2(values_446[2], values_446[3]);
                float2 value_pair_449 = _f2_374;
                float2 _f2_375 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_450 = _f2_375;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_449, value_pair_449, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_449, q_pair_450, local_dot_422[0]);
                float2 _f2_376 = make_float2(values_446[4], values_446[5]);
                float2 value_pair_451 = _f2_376;
                float2 _f2_377 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_452 = _f2_377;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_451, value_pair_451, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_451, q_pair_452, local_dot_422[0]);
                float2 _f2_378 = make_float2(values_446[6], values_446[7]);
                float2 value_pair_453 = _f2_378;
                float2 _f2_379 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_454 = _f2_379;
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
                float2 _f2_380 = make_float2(values_456[0], values_456[1]);
                float2 value_pair_457 = _f2_380;
                float2 _f2_381 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_458 = _f2_381;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_457, value_pair_457, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_457, q_pair_458, local_dot_422[1]);
                float2 _f2_382 = make_float2(values_456[2], values_456[3]);
                float2 value_pair_459 = _f2_382;
                float2 _f2_383 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_460 = _f2_383;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_459, value_pair_459, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_459, q_pair_460, local_dot_422[1]);
                float2 _f2_384 = make_float2(values_456[4], values_456[5]);
                float2 value_pair_461 = _f2_384;
                float2 _f2_385 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_462 = _f2_385;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_461, value_pair_461, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_461, q_pair_462, local_dot_422[1]);
                float2 _f2_386 = make_float2(values_456[6], values_456[7]);
                float2 value_pair_463 = _f2_386;
                float2 _f2_387 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_464 = _f2_387;
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
                float2 _f2_388 = make_float2(values_467[0], values_467[1]);
                float2 value_pair_468 = _f2_388;
                float2 _f2_389 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_469 = _f2_389;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_468, value_pair_468, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_468, q_pair_469, local_dot_422[0]);
                float2 _f2_390 = make_float2(values_467[2], values_467[3]);
                float2 value_pair_470 = _f2_390;
                float2 _f2_391 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_471 = _f2_391;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_470, value_pair_470, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_470, q_pair_471, local_dot_422[0]);
                float2 _f2_392 = make_float2(values_467[4], values_467[5]);
                float2 value_pair_472 = _f2_392;
                float2 _f2_393 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_473 = _f2_393;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_472, value_pair_472, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_472, q_pair_473, local_dot_422[0]);
                float2 _f2_394 = make_float2(values_467[6], values_467[7]);
                float2 value_pair_474 = _f2_394;
                float2 _f2_395 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_475 = _f2_395;
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
                float2 _f2_396 = make_float2(values_477[0], values_477[1]);
                float2 value_pair_478 = _f2_396;
                float2 _f2_397 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_479 = _f2_397;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_478, value_pair_478, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_478, q_pair_479, local_dot_422[1]);
                float2 _f2_398 = make_float2(values_477[2], values_477[3]);
                float2 value_pair_480 = _f2_398;
                float2 _f2_399 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_481 = _f2_399;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_480, value_pair_480, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_480, q_pair_481, local_dot_422[1]);
                float2 _f2_400 = make_float2(values_477[4], values_477[5]);
                float2 value_pair_482 = _f2_400;
                float2 _f2_401 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_483 = _f2_401;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_482, value_pair_482, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_482, q_pair_483, local_dot_422[1]);
                float2 _f2_402 = make_float2(values_477[6], values_477[7]);
                float2 value_pair_484 = _f2_402;
                float2 _f2_403 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_485 = _f2_403;
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
                float2 _f2_404 = make_float2(values_488[0], values_488[1]);
                float2 value_pair_489 = _f2_404;
                float2 _f2_405 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_490 = _f2_405;
                local_sum_sq_421[0] = fma_f32x2_rn_ftz(value_pair_489, value_pair_489, local_sum_sq_421[0]);
                local_dot_422[0] = fma_f32x2_rn_ftz(value_pair_489, q_pair_490, local_dot_422[0]);
                float2 _f2_406 = make_float2(values_488[2], values_488[3]);
                float2 value_pair_491 = _f2_406;
                float2 _f2_407 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_492 = _f2_407;
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
                float2 _f2_408 = make_float2(values_494[0], values_494[1]);
                float2 value_pair_495 = _f2_408;
                float2 _f2_409 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_496 = _f2_409;
                local_sum_sq_421[1] = fma_f32x2_rn_ftz(value_pair_495, value_pair_495, local_sum_sq_421[1]);
                local_dot_422[1] = fma_f32x2_rn_ftz(value_pair_495, q_pair_496, local_dot_422[1]);
                float2 _f2_410 = make_float2(values_494[2], values_494[3]);
                float2 value_pair_497 = _f2_410;
                float2 _f2_411 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_498 = _f2_411;
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
                float2 _f2_412 = make_float2(values_501[0], values_501[1]);
                float2 value_pair_502 = _f2_412;
                float2 _f2_413 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_503 = _f2_413;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_502, value_pair_502, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_502, q_pair_503, local_dot_422[2]);
                float2 _f2_414 = make_float2(values_501[2], values_501[3]);
                float2 value_pair_504 = _f2_414;
                float2 _f2_415 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_505 = _f2_415;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_504, value_pair_504, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_504, q_pair_505, local_dot_422[2]);
                float2 _f2_416 = make_float2(values_501[4], values_501[5]);
                float2 value_pair_506 = _f2_416;
                float2 _f2_417 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_507 = _f2_417;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_506, value_pair_506, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_506, q_pair_507, local_dot_422[2]);
                float2 _f2_418 = make_float2(values_501[6], values_501[7]);
                float2 value_pair_508 = _f2_418;
                float2 _f2_419 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_509 = _f2_419;
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
                float2 _f2_420 = make_float2(values_511[0], values_511[1]);
                float2 value_pair_512 = _f2_420;
                float2 _f2_421 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_513 = _f2_421;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_512, value_pair_512, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_512, q_pair_513, local_dot_422[3]);
                float2 _f2_422 = make_float2(values_511[2], values_511[3]);
                float2 value_pair_514 = _f2_422;
                float2 _f2_423 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_515 = _f2_423;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_514, value_pair_514, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_514, q_pair_515, local_dot_422[3]);
                float2 _f2_424 = make_float2(values_511[4], values_511[5]);
                float2 value_pair_516 = _f2_424;
                float2 _f2_425 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_517 = _f2_425;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_516, value_pair_516, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_516, q_pair_517, local_dot_422[3]);
                float2 _f2_426 = make_float2(values_511[6], values_511[7]);
                float2 value_pair_518 = _f2_426;
                float2 _f2_427 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_519 = _f2_427;
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
                float2 _f2_428 = make_float2(values_522[0], values_522[1]);
                float2 value_pair_523 = _f2_428;
                float2 _f2_429 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_524 = _f2_429;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_523, value_pair_523, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_523, q_pair_524, local_dot_422[2]);
                float2 _f2_430 = make_float2(values_522[2], values_522[3]);
                float2 value_pair_525 = _f2_430;
                float2 _f2_431 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_526 = _f2_431;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_525, value_pair_525, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_525, q_pair_526, local_dot_422[2]);
                float2 _f2_432 = make_float2(values_522[4], values_522[5]);
                float2 value_pair_527 = _f2_432;
                float2 _f2_433 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_528 = _f2_433;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_527, value_pair_527, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_527, q_pair_528, local_dot_422[2]);
                float2 _f2_434 = make_float2(values_522[6], values_522[7]);
                float2 value_pair_529 = _f2_434;
                float2 _f2_435 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_530 = _f2_435;
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
                float2 _f2_436 = make_float2(values_532[0], values_532[1]);
                float2 value_pair_533 = _f2_436;
                float2 _f2_437 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_534 = _f2_437;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_533, value_pair_533, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_533, q_pair_534, local_dot_422[3]);
                float2 _f2_438 = make_float2(values_532[2], values_532[3]);
                float2 value_pair_535 = _f2_438;
                float2 _f2_439 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_536 = _f2_439;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_535, value_pair_535, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_535, q_pair_536, local_dot_422[3]);
                float2 _f2_440 = make_float2(values_532[4], values_532[5]);
                float2 value_pair_537 = _f2_440;
                float2 _f2_441 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_538 = _f2_441;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_537, value_pair_537, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_537, q_pair_538, local_dot_422[3]);
                float2 _f2_442 = make_float2(values_532[6], values_532[7]);
                float2 value_pair_539 = _f2_442;
                float2 _f2_443 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_540 = _f2_443;
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
                float2 _f2_444 = make_float2(values_543[0], values_543[1]);
                float2 value_pair_544 = _f2_444;
                float2 _f2_445 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_545 = _f2_445;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_544, value_pair_544, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_544, q_pair_545, local_dot_422[2]);
                float2 _f2_446 = make_float2(values_543[2], values_543[3]);
                float2 value_pair_546 = _f2_446;
                float2 _f2_447 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_547 = _f2_447;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_546, value_pair_546, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_546, q_pair_547, local_dot_422[2]);
                float2 _f2_448 = make_float2(values_543[4], values_543[5]);
                float2 value_pair_548 = _f2_448;
                float2 _f2_449 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_549 = _f2_449;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_548, value_pair_548, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_548, q_pair_549, local_dot_422[2]);
                float2 _f2_450 = make_float2(values_543[6], values_543[7]);
                float2 value_pair_550 = _f2_450;
                float2 _f2_451 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_551 = _f2_451;
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
                float2 _f2_452 = make_float2(values_553[0], values_553[1]);
                float2 value_pair_554 = _f2_452;
                float2 _f2_453 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_555 = _f2_453;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_554, value_pair_554, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_554, q_pair_555, local_dot_422[3]);
                float2 _f2_454 = make_float2(values_553[2], values_553[3]);
                float2 value_pair_556 = _f2_454;
                float2 _f2_455 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_557 = _f2_455;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_556, value_pair_556, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_556, q_pair_557, local_dot_422[3]);
                float2 _f2_456 = make_float2(values_553[4], values_553[5]);
                float2 value_pair_558 = _f2_456;
                float2 _f2_457 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_559 = _f2_457;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_558, value_pair_558, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_558, q_pair_559, local_dot_422[3]);
                float2 _f2_458 = make_float2(values_553[6], values_553[7]);
                float2 value_pair_560 = _f2_458;
                float2 _f2_459 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_561 = _f2_459;
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
                float2 _f2_460 = make_float2(values_564[0], values_564[1]);
                float2 value_pair_565 = _f2_460;
                float2 _f2_461 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_566 = _f2_461;
                local_sum_sq_421[2] = fma_f32x2_rn_ftz(value_pair_565, value_pair_565, local_sum_sq_421[2]);
                local_dot_422[2] = fma_f32x2_rn_ftz(value_pair_565, q_pair_566, local_dot_422[2]);
                float2 _f2_462 = make_float2(values_564[2], values_564[3]);
                float2 value_pair_567 = _f2_462;
                float2 _f2_463 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_568 = _f2_463;
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
                float2 _f2_464 = make_float2(values_570[0], values_570[1]);
                float2 value_pair_571 = _f2_464;
                float2 _f2_465 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_572 = _f2_465;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_571, value_pair_571, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_571, q_pair_572, local_dot_422[3]);
                float2 _f2_466 = make_float2(values_570[2], values_570[3]);
                float2 value_pair_573 = _f2_466;
                float2 _f2_467 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_574 = _f2_467;
                local_sum_sq_421[3] = fma_f32x2_rn_ftz(value_pair_573, value_pair_573, local_sum_sq_421[3]);
                local_dot_422[3] = fma_f32x2_rn_ftz(value_pair_573, q_pair_574, local_dot_422[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_420) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_575 = stats_stream_base + 32;
                float2 _f2_468 = make_float2(local_sum_sq_421[0].x + local_sum_sq_421[0].y, local_dot_422[0].x + local_dot_422[0].y);
                float2 reduce_pair_576 = _f2_468;
                unsigned long long packed_pair_577 = 0;
                unsigned long long peer_packed_578 = 0;
                float2 _f2_469 = make_float2(0.0f, 0.0f);
                float2 peer_pair_579 = _f2_469;
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 16);
                peer_packed_578 = _shfl_xor_20;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 8);
                peer_packed_578 = _shfl_xor_21;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 4);
                peer_packed_578 = _shfl_xor_22;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 2);
                peer_packed_578 = _shfl_xor_23;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                packed_pair_577 = reinterpret_cast<unsigned long long*>(&reduce_pair_576)[0];
                unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_577, 1);
                peer_packed_578 = _shfl_xor_24;
                peer_pair_579 = reinterpret_cast<float2*>(&peer_packed_578)[0];
                reduce_pair_576 = add_f32x2(reduce_pair_576, peer_pair_579);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_576.x;
                    stats[slot_base_575 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_576.y;
                }
                float2 _f2_470 = make_float2(local_sum_sq_421[1].x + local_sum_sq_421[1].y, local_dot_422[1].x + local_dot_422[1].y);
                float2 reduce_pair_580 = _f2_470;
                unsigned long long packed_pair_581 = 0;
                unsigned long long peer_packed_582 = 0;
                float2 _f2_471 = make_float2(0.0f, 0.0f);
                float2 peer_pair_583 = _f2_471;
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 16);
                peer_packed_582 = _shfl_xor_25;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 8);
                peer_packed_582 = _shfl_xor_26;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 4);
                peer_packed_582 = _shfl_xor_27;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 2);
                peer_packed_582 = _shfl_xor_28;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                packed_pair_581 = reinterpret_cast<unsigned long long*>(&reduce_pair_580)[0];
                unsigned long long _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_581, 1);
                peer_packed_582 = _shfl_xor_29;
                peer_pair_583 = reinterpret_cast<float2*>(&peer_packed_582)[0];
                reduce_pair_580 = add_f32x2(reduce_pair_580, peer_pair_583);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_580.x;
                    stats[slot_base_575 + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_580.y;
                }
                float2 _f2_472 = make_float2(local_sum_sq_421[2].x + local_sum_sq_421[2].y, local_dot_422[2].x + local_dot_422[2].y);
                float2 reduce_pair_584 = _f2_472;
                unsigned long long packed_pair_585 = 0;
                unsigned long long peer_packed_586 = 0;
                float2 _f2_473 = make_float2(0.0f, 0.0f);
                float2 peer_pair_587 = _f2_473;
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 16);
                peer_packed_586 = _shfl_xor_30;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 8);
                peer_packed_586 = _shfl_xor_31;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 4);
                peer_packed_586 = _shfl_xor_32;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 2);
                peer_packed_586 = _shfl_xor_33;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                packed_pair_585 = reinterpret_cast<unsigned long long*>(&reduce_pair_584)[0];
                unsigned long long _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_585, 1);
                peer_packed_586 = _shfl_xor_34;
                peer_pair_587 = reinterpret_cast<float2*>(&peer_packed_586)[0];
                reduce_pair_584 = add_f32x2(reduce_pair_584, peer_pair_587);
                if (lane == 0) {
                    stats[slot_base_575 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_584.x;
                    stats[slot_base_575 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_584.y;
                }
                float2 _f2_474 = make_float2(local_sum_sq_421[3].x + local_sum_sq_421[3].y, local_dot_422[3].x + local_dot_422[3].y);
                float2 reduce_pair_588 = _f2_474;
                unsigned long long packed_pair_589 = 0;
                unsigned long long peer_packed_590 = 0;
                float2 _f2_475 = make_float2(0.0f, 0.0f);
                float2 peer_pair_591 = _f2_475;
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 16);
                peer_packed_590 = _shfl_xor_35;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 8);
                peer_packed_590 = _shfl_xor_36;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 4);
                peer_packed_590 = _shfl_xor_37;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 2);
                peer_packed_590 = _shfl_xor_38;
                peer_pair_591 = reinterpret_cast<float2*>(&peer_packed_590)[0];
                reduce_pair_588 = add_f32x2(reduce_pair_588, peer_pair_591);
                packed_pair_589 = reinterpret_cast<unsigned long long*>(&reduce_pair_588)[0];
                unsigned long long _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_589, 1);
                peer_packed_590 = _shfl_xor_39;
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
                float _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_594, 4, 8);
                total_sum_sq_594 += _shfl_down_6;
                float _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, total_dot_595, 4, 8);
                total_dot_595 += _shfl_down_7;
                float _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_594, 2, 8);
                total_sum_sq_594 += _shfl_down_8;
                float _shfl_down_9 = __shfl_down_sync(0xFFFFFFFF, total_dot_595, 2, 8);
                total_dot_595 += _shfl_down_9;
                float _shfl_down_10 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_594, 1, 8);
                total_sum_sq_594 += _shfl_down_10;
                float _shfl_down_11 = __shfl_down_sync(0xFFFFFFFF, total_dot_595, 1, 8);
                total_dot_595 += _shfl_down_11;
                float local_logit_596 = 0.0f;
                if (stat_source_592 < 2 && stat_warp_593 == 0) {
                    float _rsqrt_1 = rsqrtf(total_sum_sq_594 / 7168.0f + eps);
                    local_logit_596 = total_dot_595 * _rsqrt_1;
                }
                float logits_597[2];
                logits_597[0] = 0.0f;
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, local_logit_596, 0);
                logits_597[0] = _shfl_2;
                logits_597[1] = 0.0f;
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, local_logit_596, 8);
                logits_597[1] = _shfl_3;
                float weights_598[2];
                float new_max_599 = 0.0f;
                float correction_600 = 0.0f;
                float weight_sum_601 = 0.0f;
                weights_598[0] = 0.0f;
                weights_598[1] = 0.0f;
                float chunk_max_602 = -3.4028234663852886e+38f;
                float _max_3 = max_noftz(chunk_max_602, logits_597[0]);
                chunk_max_602 = _max_3;
                float _max_4 = max_noftz(chunk_max_602, logits_597[1]);
                chunk_max_602 = _max_4;
                float _max_5 = max_noftz(m_running, chunk_max_602);
                new_max_599 = _max_5;
                float _exp2_3 = approx_exp2((m_running - new_max_599) * 1.4426950408889634f);
                correction_600 = _exp2_3;
                float _exp2_4 = approx_exp2((logits_597[0] - new_max_599) * 1.4426950408889634f);
                weights_598[0] = _exp2_4;
                weight_sum_601 += weights_598[0];
                float _exp2_5 = approx_exp2((logits_597[1] - new_max_599) * 1.4426950408889634f);
                weights_598[1] = _exp2_5;
                weight_sum_601 += weights_598[1];
                float2 _f2_476 = make_float2(correction_600, correction_600);
                float2 correction_pair_603 = _f2_476;
                float2 _f2_477 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_604 = _f2_477;
                float2 scaled_pair_605;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_605) : "l"(*(const unsigned long long*)&acc_pair_604), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[0] = scaled_pair_605.x;
                acc32[1] = scaled_pair_605.y;
                float2 _f2_478 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_606 = _f2_478;
                float2 scaled_pair_607;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_607) : "l"(*(const unsigned long long*)&acc_pair_606), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[2] = scaled_pair_607.x;
                acc32[3] = scaled_pair_607.y;
                float2 _f2_479 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_608 = _f2_479;
                float2 scaled_pair_609;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_609) : "l"(*(const unsigned long long*)&acc_pair_608), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[4] = scaled_pair_609.x;
                acc32[5] = scaled_pair_609.y;
                float2 _f2_480 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_610 = _f2_480;
                float2 scaled_pair_611;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_611) : "l"(*(const unsigned long long*)&acc_pair_610), "l"(*(const unsigned long long*)&correction_pair_603));
                acc32[6] = scaled_pair_611.x;
                acc32[7] = scaled_pair_611.y;
                float _tmem_load_16[8];
                tmem_ld_x8(&_tmem_load_16[0], tmem_base);
                float2 _f2_481 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_612 = _f2_481;
                float2 _f2_482 = make_float2(_tmem_load_16[0], _tmem_load_16[1]);
                float2 cached_pair_613 = _f2_482;
                float2 _f2_483 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_614 = _f2_483;
                float2 updated_pair_615 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_613, acc_pair_614);
                acc32[0] = updated_pair_615.x;
                acc32[1] = updated_pair_615.y;
                float2 _f2_484 = make_float2(_tmem_load_16[2], _tmem_load_16[3]);
                float2 cached_pair_616 = _f2_484;
                float2 _f2_485 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_617 = _f2_485;
                float2 updated_pair_618 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_616, acc_pair_617);
                acc32[2] = updated_pair_618.x;
                acc32[3] = updated_pair_618.y;
                float2 _f2_486 = make_float2(_tmem_load_16[4], _tmem_load_16[5]);
                float2 cached_pair_619 = _f2_486;
                float2 _f2_487 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_620 = _f2_487;
                float2 updated_pair_621 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_619, acc_pair_620);
                acc32[4] = updated_pair_621.x;
                acc32[5] = updated_pair_621.y;
                float2 _f2_488 = make_float2(_tmem_load_16[6], _tmem_load_16[7]);
                float2 cached_pair_622 = _f2_488;
                float2 _f2_489 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_623 = _f2_489;
                float2 updated_pair_624 = fma_f32x2_rn_ftz(weight_pair_612, cached_pair_622, acc_pair_623);
                acc32[6] = updated_pair_624.x;
                acc32[7] = updated_pair_624.y;
                float _tmem_load_17[8];
                tmem_ld_x8(&_tmem_load_17[0], tmem_base + 64);
                float2 _f2_490 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_625 = _f2_490;
                float2 _f2_491 = make_float2(_tmem_load_17[0], _tmem_load_17[1]);
                float2 cached_pair_626 = _f2_491;
                float2 _f2_492 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_627 = _f2_492;
                float2 updated_pair_628 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_626, acc_pair_627);
                acc32[0] = updated_pair_628.x;
                acc32[1] = updated_pair_628.y;
                float2 _f2_493 = make_float2(_tmem_load_17[2], _tmem_load_17[3]);
                float2 cached_pair_629 = _f2_493;
                float2 _f2_494 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_630 = _f2_494;
                float2 updated_pair_631 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_629, acc_pair_630);
                acc32[2] = updated_pair_631.x;
                acc32[3] = updated_pair_631.y;
                float2 _f2_495 = make_float2(_tmem_load_17[4], _tmem_load_17[5]);
                float2 cached_pair_632 = _f2_495;
                float2 _f2_496 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_633 = _f2_496;
                float2 updated_pair_634 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_632, acc_pair_633);
                acc32[4] = updated_pair_634.x;
                acc32[5] = updated_pair_634.y;
                float2 _f2_497 = make_float2(_tmem_load_17[6], _tmem_load_17[7]);
                float2 cached_pair_635 = _f2_497;
                float2 _f2_498 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_636 = _f2_498;
                float2 updated_pair_637 = fma_f32x2_rn_ftz(weight_pair_625, cached_pair_635, acc_pair_636);
                acc32[6] = updated_pair_637.x;
                acc32[7] = updated_pair_637.y;
                float2 _f2_499 = make_float2(correction_600, correction_600);
                float2 correction_pair_638 = _f2_499;
                float2 _f2_500 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_639 = _f2_500;
                float2 scaled_pair_640;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_640) : "l"(*(const unsigned long long*)&acc_pair_639), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[8] = scaled_pair_640.x;
                acc32[9] = scaled_pair_640.y;
                float2 _f2_501 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_641 = _f2_501;
                float2 scaled_pair_642;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_642) : "l"(*(const unsigned long long*)&acc_pair_641), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[10] = scaled_pair_642.x;
                acc32[11] = scaled_pair_642.y;
                float2 _f2_502 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_643 = _f2_502;
                float2 scaled_pair_644;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_644) : "l"(*(const unsigned long long*)&acc_pair_643), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[12] = scaled_pair_644.x;
                acc32[13] = scaled_pair_644.y;
                float2 _f2_503 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_645 = _f2_503;
                float2 scaled_pair_646;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_646) : "l"(*(const unsigned long long*)&acc_pair_645), "l"(*(const unsigned long long*)&correction_pair_638));
                acc32[14] = scaled_pair_646.x;
                acc32[15] = scaled_pair_646.y;
                float _tmem_load_18[8];
                tmem_ld_x8(&_tmem_load_18[0], tmem_base + 8);
                float2 _f2_504 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_647 = _f2_504;
                float2 _f2_505 = make_float2(_tmem_load_18[0], _tmem_load_18[1]);
                float2 cached_pair_648 = _f2_505;
                float2 _f2_506 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_649 = _f2_506;
                float2 updated_pair_650 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_648, acc_pair_649);
                acc32[8] = updated_pair_650.x;
                acc32[9] = updated_pair_650.y;
                float2 _f2_507 = make_float2(_tmem_load_18[2], _tmem_load_18[3]);
                float2 cached_pair_651 = _f2_507;
                float2 _f2_508 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_652 = _f2_508;
                float2 updated_pair_653 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_651, acc_pair_652);
                acc32[10] = updated_pair_653.x;
                acc32[11] = updated_pair_653.y;
                float2 _f2_509 = make_float2(_tmem_load_18[4], _tmem_load_18[5]);
                float2 cached_pair_654 = _f2_509;
                float2 _f2_510 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_655 = _f2_510;
                float2 updated_pair_656 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_654, acc_pair_655);
                acc32[12] = updated_pair_656.x;
                acc32[13] = updated_pair_656.y;
                float2 _f2_511 = make_float2(_tmem_load_18[6], _tmem_load_18[7]);
                float2 cached_pair_657 = _f2_511;
                float2 _f2_512 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_658 = _f2_512;
                float2 updated_pair_659 = fma_f32x2_rn_ftz(weight_pair_647, cached_pair_657, acc_pair_658);
                acc32[14] = updated_pair_659.x;
                acc32[15] = updated_pair_659.y;
                float _tmem_load_19[8];
                tmem_ld_x8(&_tmem_load_19[0], tmem_base + 64 + 8);
                float2 _f2_513 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_660 = _f2_513;
                float2 _f2_514 = make_float2(_tmem_load_19[0], _tmem_load_19[1]);
                float2 cached_pair_661 = _f2_514;
                float2 _f2_515 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_662 = _f2_515;
                float2 updated_pair_663 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_661, acc_pair_662);
                acc32[8] = updated_pair_663.x;
                acc32[9] = updated_pair_663.y;
                float2 _f2_516 = make_float2(_tmem_load_19[2], _tmem_load_19[3]);
                float2 cached_pair_664 = _f2_516;
                float2 _f2_517 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_665 = _f2_517;
                float2 updated_pair_666 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_664, acc_pair_665);
                acc32[10] = updated_pair_666.x;
                acc32[11] = updated_pair_666.y;
                float2 _f2_518 = make_float2(_tmem_load_19[4], _tmem_load_19[5]);
                float2 cached_pair_667 = _f2_518;
                float2 _f2_519 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_668 = _f2_519;
                float2 updated_pair_669 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_667, acc_pair_668);
                acc32[12] = updated_pair_669.x;
                acc32[13] = updated_pair_669.y;
                float2 _f2_520 = make_float2(_tmem_load_19[6], _tmem_load_19[7]);
                float2 cached_pair_670 = _f2_520;
                float2 _f2_521 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_671 = _f2_521;
                float2 updated_pair_672 = fma_f32x2_rn_ftz(weight_pair_660, cached_pair_670, acc_pair_671);
                acc32[14] = updated_pair_672.x;
                acc32[15] = updated_pair_672.y;
                float2 _f2_522 = make_float2(correction_600, correction_600);
                float2 correction_pair_673 = _f2_522;
                float2 _f2_523 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_674 = _f2_523;
                float2 scaled_pair_675;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_675) : "l"(*(const unsigned long long*)&acc_pair_674), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[16] = scaled_pair_675.x;
                acc32[17] = scaled_pair_675.y;
                float2 _f2_524 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_676 = _f2_524;
                float2 scaled_pair_677;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_677) : "l"(*(const unsigned long long*)&acc_pair_676), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[18] = scaled_pair_677.x;
                acc32[19] = scaled_pair_677.y;
                float2 _f2_525 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_678 = _f2_525;
                float2 scaled_pair_679;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_679) : "l"(*(const unsigned long long*)&acc_pair_678), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[20] = scaled_pair_679.x;
                acc32[21] = scaled_pair_679.y;
                float2 _f2_526 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_680 = _f2_526;
                float2 scaled_pair_681;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_681) : "l"(*(const unsigned long long*)&acc_pair_680), "l"(*(const unsigned long long*)&correction_pair_673));
                acc32[22] = scaled_pair_681.x;
                acc32[23] = scaled_pair_681.y;
                float _tmem_load_20[8];
                tmem_ld_x8(&_tmem_load_20[0], tmem_base + 16);
                float2 _f2_527 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_682 = _f2_527;
                float2 _f2_528 = make_float2(_tmem_load_20[0], _tmem_load_20[1]);
                float2 cached_pair_683 = _f2_528;
                float2 _f2_529 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_684 = _f2_529;
                float2 updated_pair_685 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_683, acc_pair_684);
                acc32[16] = updated_pair_685.x;
                acc32[17] = updated_pair_685.y;
                float2 _f2_530 = make_float2(_tmem_load_20[2], _tmem_load_20[3]);
                float2 cached_pair_686 = _f2_530;
                float2 _f2_531 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_687 = _f2_531;
                float2 updated_pair_688 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_686, acc_pair_687);
                acc32[18] = updated_pair_688.x;
                acc32[19] = updated_pair_688.y;
                float2 _f2_532 = make_float2(_tmem_load_20[4], _tmem_load_20[5]);
                float2 cached_pair_689 = _f2_532;
                float2 _f2_533 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_690 = _f2_533;
                float2 updated_pair_691 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_689, acc_pair_690);
                acc32[20] = updated_pair_691.x;
                acc32[21] = updated_pair_691.y;
                float2 _f2_534 = make_float2(_tmem_load_20[6], _tmem_load_20[7]);
                float2 cached_pair_692 = _f2_534;
                float2 _f2_535 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_693 = _f2_535;
                float2 updated_pair_694 = fma_f32x2_rn_ftz(weight_pair_682, cached_pair_692, acc_pair_693);
                acc32[22] = updated_pair_694.x;
                acc32[23] = updated_pair_694.y;
                float _tmem_load_21[8];
                tmem_ld_x8(&_tmem_load_21[0], tmem_base + 64 + 16);
                float2 _f2_536 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_695 = _f2_536;
                float2 _f2_537 = make_float2(_tmem_load_21[0], _tmem_load_21[1]);
                float2 cached_pair_696 = _f2_537;
                float2 _f2_538 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_697 = _f2_538;
                float2 updated_pair_698 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_696, acc_pair_697);
                acc32[16] = updated_pair_698.x;
                acc32[17] = updated_pair_698.y;
                float2 _f2_539 = make_float2(_tmem_load_21[2], _tmem_load_21[3]);
                float2 cached_pair_699 = _f2_539;
                float2 _f2_540 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_700 = _f2_540;
                float2 updated_pair_701 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_699, acc_pair_700);
                acc32[18] = updated_pair_701.x;
                acc32[19] = updated_pair_701.y;
                float2 _f2_541 = make_float2(_tmem_load_21[4], _tmem_load_21[5]);
                float2 cached_pair_702 = _f2_541;
                float2 _f2_542 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_703 = _f2_542;
                float2 updated_pair_704 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_702, acc_pair_703);
                acc32[20] = updated_pair_704.x;
                acc32[21] = updated_pair_704.y;
                float2 _f2_543 = make_float2(_tmem_load_21[6], _tmem_load_21[7]);
                float2 cached_pair_705 = _f2_543;
                float2 _f2_544 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_706 = _f2_544;
                float2 updated_pair_707 = fma_f32x2_rn_ftz(weight_pair_695, cached_pair_705, acc_pair_706);
                acc32[22] = updated_pair_707.x;
                acc32[23] = updated_pair_707.y;
                float2 _f2_545 = make_float2(correction_600, correction_600);
                float2 correction_pair_708 = _f2_545;
                float2 _f2_546 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_709 = _f2_546;
                float2 scaled_pair_710;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_710) : "l"(*(const unsigned long long*)&acc_pair_709), "l"(*(const unsigned long long*)&correction_pair_708));
                acc32[24] = scaled_pair_710.x;
                acc32[25] = scaled_pair_710.y;
                float2 _f2_547 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_711 = _f2_547;
                float2 scaled_pair_712;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_712) : "l"(*(const unsigned long long*)&acc_pair_711), "l"(*(const unsigned long long*)&correction_pair_708));
                acc32[26] = scaled_pair_712.x;
                acc32[27] = scaled_pair_712.y;
                float _tmem_load_22[4];
                tmem_ld_x4(&_tmem_load_22[0], tmem_base + 24);
                float2 _f2_548 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_713 = _f2_548;
                float2 _f2_549 = make_float2(_tmem_load_22[0], _tmem_load_22[1]);
                float2 cached_pair_714 = _f2_549;
                float2 _f2_550 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_715 = _f2_550;
                float2 updated_pair_716 = fma_f32x2_rn_ftz(weight_pair_713, cached_pair_714, acc_pair_715);
                acc32[24] = updated_pair_716.x;
                acc32[25] = updated_pair_716.y;
                float2 _f2_551 = make_float2(_tmem_load_22[2], _tmem_load_22[3]);
                float2 cached_pair_717 = _f2_551;
                float2 _f2_552 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_718 = _f2_552;
                float2 updated_pair_719 = fma_f32x2_rn_ftz(weight_pair_713, cached_pair_717, acc_pair_718);
                acc32[26] = updated_pair_719.x;
                acc32[27] = updated_pair_719.y;
                float _tmem_load_23[4];
                tmem_ld_x4(&_tmem_load_23[0], tmem_base + 64 + 24);
                float2 _f2_553 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_720 = _f2_553;
                float2 _f2_554 = make_float2(_tmem_load_23[0], _tmem_load_23[1]);
                float2 cached_pair_721 = _f2_554;
                float2 _f2_555 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_722 = _f2_555;
                float2 updated_pair_723 = fma_f32x2_rn_ftz(weight_pair_720, cached_pair_721, acc_pair_722);
                acc32[24] = updated_pair_723.x;
                acc32[25] = updated_pair_723.y;
                float2 _f2_556 = make_float2(_tmem_load_23[2], _tmem_load_23[3]);
                float2 cached_pair_724 = _f2_556;
                float2 _f2_557 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_725 = _f2_557;
                float2 updated_pair_726 = fma_f32x2_rn_ftz(weight_pair_720, cached_pair_724, acc_pair_725);
                acc32[26] = updated_pair_726.x;
                acc32[27] = updated_pair_726.y;
                float2 _f2_558 = make_float2(correction_600, correction_600);
                float2 correction_pair_727 = _f2_558;
                float2 _f2_559 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_728 = _f2_559;
                float2 scaled_pair_729;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_729) : "l"(*(const unsigned long long*)&acc_pair_728), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[28] = scaled_pair_729.x;
                acc32[29] = scaled_pair_729.y;
                float2 _f2_560 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_730 = _f2_560;
                float2 scaled_pair_731;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_731) : "l"(*(const unsigned long long*)&acc_pair_730), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[30] = scaled_pair_731.x;
                acc32[31] = scaled_pair_731.y;
                float2 _f2_561 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_732 = _f2_561;
                float2 scaled_pair_733;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_733) : "l"(*(const unsigned long long*)&acc_pair_732), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[32] = scaled_pair_733.x;
                acc32[33] = scaled_pair_733.y;
                float2 _f2_562 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_734 = _f2_562;
                float2 scaled_pair_735;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_735) : "l"(*(const unsigned long long*)&acc_pair_734), "l"(*(const unsigned long long*)&correction_pair_727));
                acc32[34] = scaled_pair_735.x;
                acc32[35] = scaled_pair_735.y;
                float _tmem_load_24[8];
                tmem_ld_x8(&_tmem_load_24[0], tmem_base + 32);
                float2 _f2_563 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_736 = _f2_563;
                float2 _f2_564 = make_float2(_tmem_load_24[0], _tmem_load_24[1]);
                float2 cached_pair_737 = _f2_564;
                float2 _f2_565 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_738 = _f2_565;
                float2 updated_pair_739 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_737, acc_pair_738);
                acc32[28] = updated_pair_739.x;
                acc32[29] = updated_pair_739.y;
                float2 _f2_566 = make_float2(_tmem_load_24[2], _tmem_load_24[3]);
                float2 cached_pair_740 = _f2_566;
                float2 _f2_567 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_741 = _f2_567;
                float2 updated_pair_742 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_740, acc_pair_741);
                acc32[30] = updated_pair_742.x;
                acc32[31] = updated_pair_742.y;
                float2 _f2_568 = make_float2(_tmem_load_24[4], _tmem_load_24[5]);
                float2 cached_pair_743 = _f2_568;
                float2 _f2_569 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_744 = _f2_569;
                float2 updated_pair_745 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_743, acc_pair_744);
                acc32[32] = updated_pair_745.x;
                acc32[33] = updated_pair_745.y;
                float2 _f2_570 = make_float2(_tmem_load_24[6], _tmem_load_24[7]);
                float2 cached_pair_746 = _f2_570;
                float2 _f2_571 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_747 = _f2_571;
                float2 updated_pair_748 = fma_f32x2_rn_ftz(weight_pair_736, cached_pair_746, acc_pair_747);
                acc32[34] = updated_pair_748.x;
                acc32[35] = updated_pair_748.y;
                float _tmem_load_25[8];
                tmem_ld_x8(&_tmem_load_25[0], tmem_base + 64 + 32);
                float2 _f2_572 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_749 = _f2_572;
                float2 _f2_573 = make_float2(_tmem_load_25[0], _tmem_load_25[1]);
                float2 cached_pair_750 = _f2_573;
                float2 _f2_574 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_751 = _f2_574;
                float2 updated_pair_752 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_750, acc_pair_751);
                acc32[28] = updated_pair_752.x;
                acc32[29] = updated_pair_752.y;
                float2 _f2_575 = make_float2(_tmem_load_25[2], _tmem_load_25[3]);
                float2 cached_pair_753 = _f2_575;
                float2 _f2_576 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_754 = _f2_576;
                float2 updated_pair_755 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_753, acc_pair_754);
                acc32[30] = updated_pair_755.x;
                acc32[31] = updated_pair_755.y;
                float2 _f2_577 = make_float2(_tmem_load_25[4], _tmem_load_25[5]);
                float2 cached_pair_756 = _f2_577;
                float2 _f2_578 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_757 = _f2_578;
                float2 updated_pair_758 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_756, acc_pair_757);
                acc32[32] = updated_pair_758.x;
                acc32[33] = updated_pair_758.y;
                float2 _f2_579 = make_float2(_tmem_load_25[6], _tmem_load_25[7]);
                float2 cached_pair_759 = _f2_579;
                float2 _f2_580 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_760 = _f2_580;
                float2 updated_pair_761 = fma_f32x2_rn_ftz(weight_pair_749, cached_pair_759, acc_pair_760);
                acc32[34] = updated_pair_761.x;
                acc32[35] = updated_pair_761.y;
                float2 _f2_581 = make_float2(correction_600, correction_600);
                float2 correction_pair_762 = _f2_581;
                float2 _f2_582 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_763 = _f2_582;
                float2 scaled_pair_764;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_764) : "l"(*(const unsigned long long*)&acc_pair_763), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[36] = scaled_pair_764.x;
                acc32[37] = scaled_pair_764.y;
                float2 _f2_583 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_765 = _f2_583;
                float2 scaled_pair_766;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_766) : "l"(*(const unsigned long long*)&acc_pair_765), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[38] = scaled_pair_766.x;
                acc32[39] = scaled_pair_766.y;
                float2 _f2_584 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_767 = _f2_584;
                float2 scaled_pair_768;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_768) : "l"(*(const unsigned long long*)&acc_pair_767), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[40] = scaled_pair_768.x;
                acc32[41] = scaled_pair_768.y;
                float2 _f2_585 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_769 = _f2_585;
                float2 scaled_pair_770;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_770) : "l"(*(const unsigned long long*)&acc_pair_769), "l"(*(const unsigned long long*)&correction_pair_762));
                acc32[42] = scaled_pair_770.x;
                acc32[43] = scaled_pair_770.y;
                float _tmem_load_26[8];
                tmem_ld_x8(&_tmem_load_26[0], tmem_base + 40);
                float2 _f2_586 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_771 = _f2_586;
                float2 _f2_587 = make_float2(_tmem_load_26[0], _tmem_load_26[1]);
                float2 cached_pair_772 = _f2_587;
                float2 _f2_588 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_773 = _f2_588;
                float2 updated_pair_774 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_772, acc_pair_773);
                acc32[36] = updated_pair_774.x;
                acc32[37] = updated_pair_774.y;
                float2 _f2_589 = make_float2(_tmem_load_26[2], _tmem_load_26[3]);
                float2 cached_pair_775 = _f2_589;
                float2 _f2_590 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_776 = _f2_590;
                float2 updated_pair_777 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_775, acc_pair_776);
                acc32[38] = updated_pair_777.x;
                acc32[39] = updated_pair_777.y;
                float2 _f2_591 = make_float2(_tmem_load_26[4], _tmem_load_26[5]);
                float2 cached_pair_778 = _f2_591;
                float2 _f2_592 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_779 = _f2_592;
                float2 updated_pair_780 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_778, acc_pair_779);
                acc32[40] = updated_pair_780.x;
                acc32[41] = updated_pair_780.y;
                float2 _f2_593 = make_float2(_tmem_load_26[6], _tmem_load_26[7]);
                float2 cached_pair_781 = _f2_593;
                float2 _f2_594 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_782 = _f2_594;
                float2 updated_pair_783 = fma_f32x2_rn_ftz(weight_pair_771, cached_pair_781, acc_pair_782);
                acc32[42] = updated_pair_783.x;
                acc32[43] = updated_pair_783.y;
                float _tmem_load_27[8];
                tmem_ld_x8(&_tmem_load_27[0], tmem_base + 64 + 40);
                float2 _f2_595 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_784 = _f2_595;
                float2 _f2_596 = make_float2(_tmem_load_27[0], _tmem_load_27[1]);
                float2 cached_pair_785 = _f2_596;
                float2 _f2_597 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_786 = _f2_597;
                float2 updated_pair_787 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_785, acc_pair_786);
                acc32[36] = updated_pair_787.x;
                acc32[37] = updated_pair_787.y;
                float2 _f2_598 = make_float2(_tmem_load_27[2], _tmem_load_27[3]);
                float2 cached_pair_788 = _f2_598;
                float2 _f2_599 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_789 = _f2_599;
                float2 updated_pair_790 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_788, acc_pair_789);
                acc32[38] = updated_pair_790.x;
                acc32[39] = updated_pair_790.y;
                float2 _f2_600 = make_float2(_tmem_load_27[4], _tmem_load_27[5]);
                float2 cached_pair_791 = _f2_600;
                float2 _f2_601 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_792 = _f2_601;
                float2 updated_pair_793 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_791, acc_pair_792);
                acc32[40] = updated_pair_793.x;
                acc32[41] = updated_pair_793.y;
                float2 _f2_602 = make_float2(_tmem_load_27[6], _tmem_load_27[7]);
                float2 cached_pair_794 = _f2_602;
                float2 _f2_603 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_795 = _f2_603;
                float2 updated_pair_796 = fma_f32x2_rn_ftz(weight_pair_784, cached_pair_794, acc_pair_795);
                acc32[42] = updated_pair_796.x;
                acc32[43] = updated_pair_796.y;
                float2 _f2_604 = make_float2(correction_600, correction_600);
                float2 correction_pair_797 = _f2_604;
                float2 _f2_605 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_798 = _f2_605;
                float2 scaled_pair_799;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_799) : "l"(*(const unsigned long long*)&acc_pair_798), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[44] = scaled_pair_799.x;
                acc32[45] = scaled_pair_799.y;
                float2 _f2_606 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_800 = _f2_606;
                float2 scaled_pair_801;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_801) : "l"(*(const unsigned long long*)&acc_pair_800), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[46] = scaled_pair_801.x;
                acc32[47] = scaled_pair_801.y;
                float2 _f2_607 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_802 = _f2_607;
                float2 scaled_pair_803;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_803) : "l"(*(const unsigned long long*)&acc_pair_802), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[48] = scaled_pair_803.x;
                acc32[49] = scaled_pair_803.y;
                float2 _f2_608 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_804 = _f2_608;
                float2 scaled_pair_805;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_805) : "l"(*(const unsigned long long*)&acc_pair_804), "l"(*(const unsigned long long*)&correction_pair_797));
                acc32[50] = scaled_pair_805.x;
                acc32[51] = scaled_pair_805.y;
                float _tmem_load_28[8];
                tmem_ld_x8(&_tmem_load_28[0], tmem_base + 48);
                float2 _f2_609 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_806 = _f2_609;
                float2 _f2_610 = make_float2(_tmem_load_28[0], _tmem_load_28[1]);
                float2 cached_pair_807 = _f2_610;
                float2 _f2_611 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_808 = _f2_611;
                float2 updated_pair_809 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_807, acc_pair_808);
                acc32[44] = updated_pair_809.x;
                acc32[45] = updated_pair_809.y;
                float2 _f2_612 = make_float2(_tmem_load_28[2], _tmem_load_28[3]);
                float2 cached_pair_810 = _f2_612;
                float2 _f2_613 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_811 = _f2_613;
                float2 updated_pair_812 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_810, acc_pair_811);
                acc32[46] = updated_pair_812.x;
                acc32[47] = updated_pair_812.y;
                float2 _f2_614 = make_float2(_tmem_load_28[4], _tmem_load_28[5]);
                float2 cached_pair_813 = _f2_614;
                float2 _f2_615 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_814 = _f2_615;
                float2 updated_pair_815 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_813, acc_pair_814);
                acc32[48] = updated_pair_815.x;
                acc32[49] = updated_pair_815.y;
                float2 _f2_616 = make_float2(_tmem_load_28[6], _tmem_load_28[7]);
                float2 cached_pair_816 = _f2_616;
                float2 _f2_617 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_817 = _f2_617;
                float2 updated_pair_818 = fma_f32x2_rn_ftz(weight_pair_806, cached_pair_816, acc_pair_817);
                acc32[50] = updated_pair_818.x;
                acc32[51] = updated_pair_818.y;
                float _tmem_load_29[8];
                tmem_ld_x8(&_tmem_load_29[0], tmem_base + 64 + 48);
                float2 _f2_618 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_819 = _f2_618;
                float2 _f2_619 = make_float2(_tmem_load_29[0], _tmem_load_29[1]);
                float2 cached_pair_820 = _f2_619;
                float2 _f2_620 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_821 = _f2_620;
                float2 updated_pair_822 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_820, acc_pair_821);
                acc32[44] = updated_pair_822.x;
                acc32[45] = updated_pair_822.y;
                float2 _f2_621 = make_float2(_tmem_load_29[2], _tmem_load_29[3]);
                float2 cached_pair_823 = _f2_621;
                float2 _f2_622 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_824 = _f2_622;
                float2 updated_pair_825 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_823, acc_pair_824);
                acc32[46] = updated_pair_825.x;
                acc32[47] = updated_pair_825.y;
                float2 _f2_623 = make_float2(_tmem_load_29[4], _tmem_load_29[5]);
                float2 cached_pair_826 = _f2_623;
                float2 _f2_624 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_827 = _f2_624;
                float2 updated_pair_828 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_826, acc_pair_827);
                acc32[48] = updated_pair_828.x;
                acc32[49] = updated_pair_828.y;
                float2 _f2_625 = make_float2(_tmem_load_29[6], _tmem_load_29[7]);
                float2 cached_pair_829 = _f2_625;
                float2 _f2_626 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_830 = _f2_626;
                float2 updated_pair_831 = fma_f32x2_rn_ftz(weight_pair_819, cached_pair_829, acc_pair_830);
                acc32[50] = updated_pair_831.x;
                acc32[51] = updated_pair_831.y;
                float2 _f2_627 = make_float2(correction_600, correction_600);
                float2 correction_pair_832 = _f2_627;
                float2 _f2_628 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_833 = _f2_628;
                float2 scaled_pair_834;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_834) : "l"(*(const unsigned long long*)&acc_pair_833), "l"(*(const unsigned long long*)&correction_pair_832));
                acc32[52] = scaled_pair_834.x;
                acc32[53] = scaled_pair_834.y;
                float2 _f2_629 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_835 = _f2_629;
                float2 scaled_pair_836;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_836) : "l"(*(const unsigned long long*)&acc_pair_835), "l"(*(const unsigned long long*)&correction_pair_832));
                acc32[54] = scaled_pair_836.x;
                acc32[55] = scaled_pair_836.y;
                float _tmem_load_30[4];
                tmem_ld_x4(&_tmem_load_30[0], tmem_base + 56);
                float2 _f2_630 = make_float2(weights_598[0], weights_598[0]);
                float2 weight_pair_837 = _f2_630;
                float2 _f2_631 = make_float2(_tmem_load_30[0], _tmem_load_30[1]);
                float2 cached_pair_838 = _f2_631;
                float2 _f2_632 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_839 = _f2_632;
                float2 updated_pair_840 = fma_f32x2_rn_ftz(weight_pair_837, cached_pair_838, acc_pair_839);
                acc32[52] = updated_pair_840.x;
                acc32[53] = updated_pair_840.y;
                float2 _f2_633 = make_float2(_tmem_load_30[2], _tmem_load_30[3]);
                float2 cached_pair_841 = _f2_633;
                float2 _f2_634 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_842 = _f2_634;
                float2 updated_pair_843 = fma_f32x2_rn_ftz(weight_pair_837, cached_pair_841, acc_pair_842);
                acc32[54] = updated_pair_843.x;
                acc32[55] = updated_pair_843.y;
                float _tmem_load_31[4];
                tmem_ld_x4(&_tmem_load_31[0], tmem_base + 64 + 56);
                float2 _f2_635 = make_float2(weights_598[1], weights_598[1]);
                float2 weight_pair_844 = _f2_635;
                float2 _f2_636 = make_float2(_tmem_load_31[0], _tmem_load_31[1]);
                float2 cached_pair_845 = _f2_636;
                float2 _f2_637 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_846 = _f2_637;
                float2 updated_pair_847 = fma_f32x2_rn_ftz(weight_pair_844, cached_pair_845, acc_pair_846);
                acc32[52] = updated_pair_847.x;
                acc32[53] = updated_pair_847.y;
                float2 _f2_638 = make_float2(_tmem_load_31[2], _tmem_load_31[3]);
                float2 cached_pair_848 = _f2_638;
                float2 _f2_639 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_849 = _f2_639;
                float2 updated_pair_850 = fma_f32x2_rn_ftz(weight_pair_844, cached_pair_848, acc_pair_849);
                acc32[54] = updated_pair_850.x;
                acc32[55] = updated_pair_850.y;
                s_running = __fmaf_rn(s_running, correction_600, weight_sum_601);
                m_running = new_max_599;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; consumer_phase ^= 1; }
                unsigned int ring_stage_851 = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage_851) * 8, consumer_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq_852[4];
                float2 local_dot_853[4];
                float2 _f2_640 = make_float2(0.0f, 0.0f);
                local_sum_sq_852[0] = _f2_640;
                float2 _f2_641 = make_float2(0.0f, 0.0f);
                local_dot_853[0] = _f2_641;
                float2 _f2_642 = make_float2(0.0f, 0.0f);
                local_sum_sq_852[1] = _f2_642;
                float2 _f2_643 = make_float2(0.0f, 0.0f);
                local_dot_853[1] = _f2_643;
                float2 _f2_644 = make_float2(0.0f, 0.0f);
                local_sum_sq_852[2] = _f2_644;
                float2 _f2_645 = make_float2(0.0f, 0.0f);
                local_dot_853[2] = _f2_645;
                float2 _f2_646 = make_float2(0.0f, 0.0f);
                local_sum_sq_852[3] = _f2_646;
                float2 _f2_647 = make_float2(0.0f, 0.0f);
                local_dot_853[3] = _f2_647;
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
                float2 _f2_648 = make_float2(values_856[0], values_856[1]);
                float2 value_pair_857 = _f2_648;
                float2 _f2_649 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_858 = _f2_649;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_857, value_pair_857, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_857, q_pair_858, local_dot_853[0]);
                float2 _f2_650 = make_float2(values_856[2], values_856[3]);
                float2 value_pair_859 = _f2_650;
                float2 _f2_651 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_860 = _f2_651;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_859, value_pair_859, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_859, q_pair_860, local_dot_853[0]);
                float2 _f2_652 = make_float2(values_856[4], values_856[5]);
                float2 value_pair_861 = _f2_652;
                float2 _f2_653 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_862 = _f2_653;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_861, value_pair_861, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_861, q_pair_862, local_dot_853[0]);
                float2 _f2_654 = make_float2(values_856[6], values_856[7]);
                float2 value_pair_863 = _f2_654;
                float2 _f2_655 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_864 = _f2_655;
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
                float2 _f2_656 = make_float2(values_866[0], values_866[1]);
                float2 value_pair_867 = _f2_656;
                float2 _f2_657 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_868 = _f2_657;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_867, value_pair_867, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_867, q_pair_868, local_dot_853[1]);
                float2 _f2_658 = make_float2(values_866[2], values_866[3]);
                float2 value_pair_869 = _f2_658;
                float2 _f2_659 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_870 = _f2_659;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_869, value_pair_869, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_869, q_pair_870, local_dot_853[1]);
                float2 _f2_660 = make_float2(values_866[4], values_866[5]);
                float2 value_pair_871 = _f2_660;
                float2 _f2_661 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_872 = _f2_661;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_871, value_pair_871, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_871, q_pair_872, local_dot_853[1]);
                float2 _f2_662 = make_float2(values_866[6], values_866[7]);
                float2 value_pair_873 = _f2_662;
                float2 _f2_663 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_874 = _f2_663;
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
                float2 _f2_664 = make_float2(values_877[0], values_877[1]);
                float2 value_pair_878 = _f2_664;
                float2 _f2_665 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_879 = _f2_665;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_878, value_pair_878, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_878, q_pair_879, local_dot_853[0]);
                float2 _f2_666 = make_float2(values_877[2], values_877[3]);
                float2 value_pair_880 = _f2_666;
                float2 _f2_667 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_881 = _f2_667;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_880, value_pair_880, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_880, q_pair_881, local_dot_853[0]);
                float2 _f2_668 = make_float2(values_877[4], values_877[5]);
                float2 value_pair_882 = _f2_668;
                float2 _f2_669 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_883 = _f2_669;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_882, value_pair_882, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_882, q_pair_883, local_dot_853[0]);
                float2 _f2_670 = make_float2(values_877[6], values_877[7]);
                float2 value_pair_884 = _f2_670;
                float2 _f2_671 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_885 = _f2_671;
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
                float2 _f2_672 = make_float2(values_887[0], values_887[1]);
                float2 value_pair_888 = _f2_672;
                float2 _f2_673 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_889 = _f2_673;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_888, value_pair_888, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_888, q_pair_889, local_dot_853[1]);
                float2 _f2_674 = make_float2(values_887[2], values_887[3]);
                float2 value_pair_890 = _f2_674;
                float2 _f2_675 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_891 = _f2_675;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_890, value_pair_890, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_890, q_pair_891, local_dot_853[1]);
                float2 _f2_676 = make_float2(values_887[4], values_887[5]);
                float2 value_pair_892 = _f2_676;
                float2 _f2_677 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_893 = _f2_677;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_892, value_pair_892, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_892, q_pair_893, local_dot_853[1]);
                float2 _f2_678 = make_float2(values_887[6], values_887[7]);
                float2 value_pair_894 = _f2_678;
                float2 _f2_679 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_895 = _f2_679;
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
                float2 _f2_680 = make_float2(values_898[0], values_898[1]);
                float2 value_pair_899 = _f2_680;
                float2 _f2_681 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_900 = _f2_681;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_899, value_pair_899, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_899, q_pair_900, local_dot_853[0]);
                float2 _f2_682 = make_float2(values_898[2], values_898[3]);
                float2 value_pair_901 = _f2_682;
                float2 _f2_683 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_902 = _f2_683;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_901, value_pair_901, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_901, q_pair_902, local_dot_853[0]);
                float2 _f2_684 = make_float2(values_898[4], values_898[5]);
                float2 value_pair_903 = _f2_684;
                float2 _f2_685 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_904 = _f2_685;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_903, value_pair_903, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_903, q_pair_904, local_dot_853[0]);
                float2 _f2_686 = make_float2(values_898[6], values_898[7]);
                float2 value_pair_905 = _f2_686;
                float2 _f2_687 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_906 = _f2_687;
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
                float2 _f2_688 = make_float2(values_908[0], values_908[1]);
                float2 value_pair_909 = _f2_688;
                float2 _f2_689 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_910 = _f2_689;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_909, value_pair_909, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_909, q_pair_910, local_dot_853[1]);
                float2 _f2_690 = make_float2(values_908[2], values_908[3]);
                float2 value_pair_911 = _f2_690;
                float2 _f2_691 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_912 = _f2_691;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_911, value_pair_911, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_911, q_pair_912, local_dot_853[1]);
                float2 _f2_692 = make_float2(values_908[4], values_908[5]);
                float2 value_pair_913 = _f2_692;
                float2 _f2_693 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_914 = _f2_693;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_913, value_pair_913, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_913, q_pair_914, local_dot_853[1]);
                float2 _f2_694 = make_float2(values_908[6], values_908[7]);
                float2 value_pair_915 = _f2_694;
                float2 _f2_695 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_916 = _f2_695;
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
                float2 _f2_696 = make_float2(values_919[0], values_919[1]);
                float2 value_pair_920 = _f2_696;
                float2 _f2_697 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_921 = _f2_697;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_920, value_pair_920, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_920, q_pair_921, local_dot_853[0]);
                float2 _f2_698 = make_float2(values_919[2], values_919[3]);
                float2 value_pair_922 = _f2_698;
                float2 _f2_699 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_923 = _f2_699;
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
                float2 _f2_700 = make_float2(values_925[0], values_925[1]);
                float2 value_pair_926 = _f2_700;
                float2 _f2_701 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_927 = _f2_701;
                local_sum_sq_852[1] = fma_f32x2_rn_ftz(value_pair_926, value_pair_926, local_sum_sq_852[1]);
                local_dot_853[1] = fma_f32x2_rn_ftz(value_pair_926, q_pair_927, local_dot_853[1]);
                float2 _f2_702 = make_float2(values_925[2], values_925[3]);
                float2 value_pair_928 = _f2_702;
                float2 _f2_703 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_929 = _f2_703;
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
                float2 _f2_704 = make_float2(values_932[0], values_932[1]);
                float2 value_pair_933 = _f2_704;
                float2 _f2_705 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_934 = _f2_705;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_933, value_pair_933, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_933, q_pair_934, local_dot_853[2]);
                float2 _f2_706 = make_float2(values_932[2], values_932[3]);
                float2 value_pair_935 = _f2_706;
                float2 _f2_707 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_936 = _f2_707;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_935, value_pair_935, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_935, q_pair_936, local_dot_853[2]);
                float2 _f2_708 = make_float2(values_932[4], values_932[5]);
                float2 value_pair_937 = _f2_708;
                float2 _f2_709 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_938 = _f2_709;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_937, value_pair_937, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_937, q_pair_938, local_dot_853[2]);
                float2 _f2_710 = make_float2(values_932[6], values_932[7]);
                float2 value_pair_939 = _f2_710;
                float2 _f2_711 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_940 = _f2_711;
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
                float2 _f2_712 = make_float2(values_942[0], values_942[1]);
                float2 value_pair_943 = _f2_712;
                float2 _f2_713 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_944 = _f2_713;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_943, value_pair_943, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_943, q_pair_944, local_dot_853[3]);
                float2 _f2_714 = make_float2(values_942[2], values_942[3]);
                float2 value_pair_945 = _f2_714;
                float2 _f2_715 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_946 = _f2_715;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_945, value_pair_945, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_945, q_pair_946, local_dot_853[3]);
                float2 _f2_716 = make_float2(values_942[4], values_942[5]);
                float2 value_pair_947 = _f2_716;
                float2 _f2_717 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_948 = _f2_717;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_947, value_pair_947, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_947, q_pair_948, local_dot_853[3]);
                float2 _f2_718 = make_float2(values_942[6], values_942[7]);
                float2 value_pair_949 = _f2_718;
                float2 _f2_719 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_950 = _f2_719;
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
                float2 _f2_720 = make_float2(values_953[0], values_953[1]);
                float2 value_pair_954 = _f2_720;
                float2 _f2_721 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_955 = _f2_721;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_954, value_pair_954, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_954, q_pair_955, local_dot_853[2]);
                float2 _f2_722 = make_float2(values_953[2], values_953[3]);
                float2 value_pair_956 = _f2_722;
                float2 _f2_723 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_957 = _f2_723;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_956, value_pair_956, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_956, q_pair_957, local_dot_853[2]);
                float2 _f2_724 = make_float2(values_953[4], values_953[5]);
                float2 value_pair_958 = _f2_724;
                float2 _f2_725 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_959 = _f2_725;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_958, value_pair_958, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_958, q_pair_959, local_dot_853[2]);
                float2 _f2_726 = make_float2(values_953[6], values_953[7]);
                float2 value_pair_960 = _f2_726;
                float2 _f2_727 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_961 = _f2_727;
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
                float2 _f2_728 = make_float2(values_963[0], values_963[1]);
                float2 value_pair_964 = _f2_728;
                float2 _f2_729 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_965 = _f2_729;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_964, value_pair_964, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_964, q_pair_965, local_dot_853[3]);
                float2 _f2_730 = make_float2(values_963[2], values_963[3]);
                float2 value_pair_966 = _f2_730;
                float2 _f2_731 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_967 = _f2_731;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_966, value_pair_966, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_966, q_pair_967, local_dot_853[3]);
                float2 _f2_732 = make_float2(values_963[4], values_963[5]);
                float2 value_pair_968 = _f2_732;
                float2 _f2_733 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_969 = _f2_733;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_968, value_pair_968, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_968, q_pair_969, local_dot_853[3]);
                float2 _f2_734 = make_float2(values_963[6], values_963[7]);
                float2 value_pair_970 = _f2_734;
                float2 _f2_735 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_971 = _f2_735;
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
                float2 _f2_736 = make_float2(values_974[0], values_974[1]);
                float2 value_pair_975 = _f2_736;
                float2 _f2_737 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_976 = _f2_737;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_975, value_pair_975, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_975, q_pair_976, local_dot_853[2]);
                float2 _f2_738 = make_float2(values_974[2], values_974[3]);
                float2 value_pair_977 = _f2_738;
                float2 _f2_739 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_978 = _f2_739;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_977, value_pair_977, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_977, q_pair_978, local_dot_853[2]);
                float2 _f2_740 = make_float2(values_974[4], values_974[5]);
                float2 value_pair_979 = _f2_740;
                float2 _f2_741 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_980 = _f2_741;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_979, value_pair_979, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_979, q_pair_980, local_dot_853[2]);
                float2 _f2_742 = make_float2(values_974[6], values_974[7]);
                float2 value_pair_981 = _f2_742;
                float2 _f2_743 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_982 = _f2_743;
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
                float2 _f2_744 = make_float2(values_984[0], values_984[1]);
                float2 value_pair_985 = _f2_744;
                float2 _f2_745 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_986 = _f2_745;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_985, value_pair_985, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_985, q_pair_986, local_dot_853[3]);
                float2 _f2_746 = make_float2(values_984[2], values_984[3]);
                float2 value_pair_987 = _f2_746;
                float2 _f2_747 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_988 = _f2_747;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_987, value_pair_987, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_987, q_pair_988, local_dot_853[3]);
                float2 _f2_748 = make_float2(values_984[4], values_984[5]);
                float2 value_pair_989 = _f2_748;
                float2 _f2_749 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_990 = _f2_749;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_989, value_pair_989, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_989, q_pair_990, local_dot_853[3]);
                float2 _f2_750 = make_float2(values_984[6], values_984[7]);
                float2 value_pair_991 = _f2_750;
                float2 _f2_751 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_992 = _f2_751;
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
                float2 _f2_752 = make_float2(values_995[0], values_995[1]);
                float2 value_pair_996 = _f2_752;
                float2 _f2_753 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_997 = _f2_753;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_996, value_pair_996, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_996, q_pair_997, local_dot_853[2]);
                float2 _f2_754 = make_float2(values_995[2], values_995[3]);
                float2 value_pair_998 = _f2_754;
                float2 _f2_755 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_999 = _f2_755;
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
                float2 _f2_756 = make_float2(values_1001[0], values_1001[1]);
                float2 value_pair_1002 = _f2_756;
                float2 _f2_757 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_1003 = _f2_757;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1002, value_pair_1002, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1002, q_pair_1003, local_dot_853[3]);
                float2 _f2_758 = make_float2(values_1001[2], values_1001[3]);
                float2 value_pair_1004 = _f2_758;
                float2 _f2_759 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_1005 = _f2_759;
                local_sum_sq_852[3] = fma_f32x2_rn_ftz(value_pair_1004, value_pair_1004, local_sum_sq_852[3]);
                local_dot_853[3] = fma_f32x2_rn_ftz(value_pair_1004, q_pair_1005, local_dot_853[3]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_851) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1006 = stats_stream_base + 64;
                float2 _f2_760 = make_float2(local_sum_sq_852[0].x + local_sum_sq_852[0].y, local_dot_853[0].x + local_dot_853[0].y);
                float2 reduce_pair_1007 = _f2_760;
                unsigned long long packed_pair_1008 = 0;
                unsigned long long peer_packed_1009 = 0;
                float2 _f2_761 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1010 = _f2_761;
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 16);
                peer_packed_1009 = _shfl_xor_40;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 8);
                peer_packed_1009 = _shfl_xor_41;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 4);
                peer_packed_1009 = _shfl_xor_42;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 2);
                peer_packed_1009 = _shfl_xor_43;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                packed_pair_1008 = reinterpret_cast<unsigned long long*>(&reduce_pair_1007)[0];
                unsigned long long _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1008, 1);
                peer_packed_1009 = _shfl_xor_44;
                peer_pair_1010 = reinterpret_cast<float2*>(&peer_packed_1009)[0];
                reduce_pair_1007 = add_f32x2(reduce_pair_1007, peer_pair_1010);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1007.x;
                    stats[slot_base_1006 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1007.y;
                }
                float2 _f2_762 = make_float2(local_sum_sq_852[1].x + local_sum_sq_852[1].y, local_dot_853[1].x + local_dot_853[1].y);
                float2 reduce_pair_1011 = _f2_762;
                unsigned long long packed_pair_1012 = 0;
                unsigned long long peer_packed_1013 = 0;
                float2 _f2_763 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1014 = _f2_763;
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 16);
                peer_packed_1013 = _shfl_xor_45;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 8);
                peer_packed_1013 = _shfl_xor_46;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 4);
                peer_packed_1013 = _shfl_xor_47;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 2);
                peer_packed_1013 = _shfl_xor_48;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                packed_pair_1012 = reinterpret_cast<unsigned long long*>(&reduce_pair_1011)[0];
                unsigned long long _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1012, 1);
                peer_packed_1013 = _shfl_xor_49;
                peer_pair_1014 = reinterpret_cast<float2*>(&peer_packed_1013)[0];
                reduce_pair_1011 = add_f32x2(reduce_pair_1011, peer_pair_1014);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)((warp_in_group * 2 + 1) * 2)] = reduce_pair_1011.x;
                    stats[slot_base_1006 + (unsigned int)((warp_in_group * 2 + 1) * 2) + 1] = reduce_pair_1011.y;
                }
                float2 _f2_764 = make_float2(local_sum_sq_852[2].x + local_sum_sq_852[2].y, local_dot_853[2].x + local_dot_853[2].y);
                float2 reduce_pair_1015 = _f2_764;
                unsigned long long packed_pair_1016 = 0;
                unsigned long long peer_packed_1017 = 0;
                float2 _f2_765 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1018 = _f2_765;
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 16);
                peer_packed_1017 = _shfl_xor_50;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 8);
                peer_packed_1017 = _shfl_xor_51;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 4);
                peer_packed_1017 = _shfl_xor_52;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 2);
                peer_packed_1017 = _shfl_xor_53;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                packed_pair_1016 = reinterpret_cast<unsigned long long*>(&reduce_pair_1015)[0];
                unsigned long long _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1016, 1);
                peer_packed_1017 = _shfl_xor_54;
                peer_pair_1018 = reinterpret_cast<float2*>(&peer_packed_1017)[0];
                reduce_pair_1015 = add_f32x2(reduce_pair_1015, peer_pair_1018);
                if (lane == 0) {
                    stats[slot_base_1006 + (unsigned int)((4 + warp_in_group) * 2 * 2)] = reduce_pair_1015.x;
                    stats[slot_base_1006 + (unsigned int)((4 + warp_in_group) * 2 * 2) + 1] = reduce_pair_1015.y;
                }
                float2 _f2_766 = make_float2(local_sum_sq_852[3].x + local_sum_sq_852[3].y, local_dot_853[3].x + local_dot_853[3].y);
                float2 reduce_pair_1019 = _f2_766;
                unsigned long long packed_pair_1020 = 0;
                unsigned long long peer_packed_1021 = 0;
                float2 _f2_767 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1022 = _f2_767;
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 16);
                peer_packed_1021 = _shfl_xor_55;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 8);
                peer_packed_1021 = _shfl_xor_56;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 4);
                peer_packed_1021 = _shfl_xor_57;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 2);
                peer_packed_1021 = _shfl_xor_58;
                peer_pair_1022 = reinterpret_cast<float2*>(&peer_packed_1021)[0];
                reduce_pair_1019 = add_f32x2(reduce_pair_1019, peer_pair_1022);
                packed_pair_1020 = reinterpret_cast<unsigned long long*>(&reduce_pair_1019)[0];
                unsigned long long _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1020, 1);
                peer_packed_1021 = _shfl_xor_59;
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
                float _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1025, 4, 8);
                total_sum_sq_1025 += _shfl_down_12;
                float _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, total_dot_1026, 4, 8);
                total_dot_1026 += _shfl_down_13;
                float _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1025, 2, 8);
                total_sum_sq_1025 += _shfl_down_14;
                float _shfl_down_15 = __shfl_down_sync(0xFFFFFFFF, total_dot_1026, 2, 8);
                total_dot_1026 += _shfl_down_15;
                float _shfl_down_16 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1025, 1, 8);
                total_sum_sq_1025 += _shfl_down_16;
                float _shfl_down_17 = __shfl_down_sync(0xFFFFFFFF, total_dot_1026, 1, 8);
                total_dot_1026 += _shfl_down_17;
                float local_logit_1027 = 0.0f;
                if (stat_source_1023 < 2 && stat_warp_1024 == 0) {
                    float _rsqrt_2 = rsqrtf(total_sum_sq_1025 / 7168.0f + eps);
                    local_logit_1027 = total_dot_1026 * _rsqrt_2;
                }
                float logits_1028[2];
                logits_1028[0] = 0.0f;
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, local_logit_1027, 0);
                logits_1028[0] = _shfl_4;
                logits_1028[1] = 0.0f;
                float _shfl_5 = __shfl_sync(0xFFFFFFFF, local_logit_1027, 8);
                logits_1028[1] = _shfl_5;
                float weights_1029[2];
                float new_max_1030 = 0.0f;
                float correction_1031 = 0.0f;
                float weight_sum_1032 = 0.0f;
                weights_1029[0] = 0.0f;
                weights_1029[1] = 0.0f;
                float chunk_max_1033 = -3.4028234663852886e+38f;
                float _max_6 = max_noftz(chunk_max_1033, logits_1028[0]);
                chunk_max_1033 = _max_6;
                float _max_7 = max_noftz(chunk_max_1033, logits_1028[1]);
                chunk_max_1033 = _max_7;
                float _max_8 = max_noftz(m_running, chunk_max_1033);
                new_max_1030 = _max_8;
                float _exp2_6 = approx_exp2((m_running - new_max_1030) * 1.4426950408889634f);
                correction_1031 = _exp2_6;
                float _exp2_7 = approx_exp2((logits_1028[0] - new_max_1030) * 1.4426950408889634f);
                weights_1029[0] = _exp2_7;
                weight_sum_1032 += weights_1029[0];
                float _exp2_8 = approx_exp2((logits_1028[1] - new_max_1030) * 1.4426950408889634f);
                weights_1029[1] = _exp2_8;
                weight_sum_1032 += weights_1029[1];
                float2 _f2_768 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1034 = _f2_768;
                float2 _f2_769 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1035 = _f2_769;
                float2 scaled_pair_1036;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1036) : "l"(*(const unsigned long long*)&acc_pair_1035), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[0] = scaled_pair_1036.x;
                acc32[1] = scaled_pair_1036.y;
                float2 _f2_770 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1037 = _f2_770;
                float2 scaled_pair_1038;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1038) : "l"(*(const unsigned long long*)&acc_pair_1037), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[2] = scaled_pair_1038.x;
                acc32[3] = scaled_pair_1038.y;
                float2 _f2_771 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1039 = _f2_771;
                float2 scaled_pair_1040;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1040) : "l"(*(const unsigned long long*)&acc_pair_1039), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[4] = scaled_pair_1040.x;
                acc32[5] = scaled_pair_1040.y;
                float2 _f2_772 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1041 = _f2_772;
                float2 scaled_pair_1042;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1042) : "l"(*(const unsigned long long*)&acc_pair_1041), "l"(*(const unsigned long long*)&correction_pair_1034));
                acc32[6] = scaled_pair_1042.x;
                acc32[7] = scaled_pair_1042.y;
                float _tmem_load_32[8];
                tmem_ld_x8(&_tmem_load_32[0], tmem_base);
                float2 _f2_773 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1043 = _f2_773;
                float2 _f2_774 = make_float2(_tmem_load_32[0], _tmem_load_32[1]);
                float2 cached_pair_1044 = _f2_774;
                float2 _f2_775 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1045 = _f2_775;
                float2 updated_pair_1046 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1044, acc_pair_1045);
                acc32[0] = updated_pair_1046.x;
                acc32[1] = updated_pair_1046.y;
                float2 _f2_776 = make_float2(_tmem_load_32[2], _tmem_load_32[3]);
                float2 cached_pair_1047 = _f2_776;
                float2 _f2_777 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1048 = _f2_777;
                float2 updated_pair_1049 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1047, acc_pair_1048);
                acc32[2] = updated_pair_1049.x;
                acc32[3] = updated_pair_1049.y;
                float2 _f2_778 = make_float2(_tmem_load_32[4], _tmem_load_32[5]);
                float2 cached_pair_1050 = _f2_778;
                float2 _f2_779 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1051 = _f2_779;
                float2 updated_pair_1052 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1050, acc_pair_1051);
                acc32[4] = updated_pair_1052.x;
                acc32[5] = updated_pair_1052.y;
                float2 _f2_780 = make_float2(_tmem_load_32[6], _tmem_load_32[7]);
                float2 cached_pair_1053 = _f2_780;
                float2 _f2_781 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1054 = _f2_781;
                float2 updated_pair_1055 = fma_f32x2_rn_ftz(weight_pair_1043, cached_pair_1053, acc_pair_1054);
                acc32[6] = updated_pair_1055.x;
                acc32[7] = updated_pair_1055.y;
                float _tmem_load_33[8];
                tmem_ld_x8(&_tmem_load_33[0], tmem_base + 64);
                float2 _f2_782 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1056 = _f2_782;
                float2 _f2_783 = make_float2(_tmem_load_33[0], _tmem_load_33[1]);
                float2 cached_pair_1057 = _f2_783;
                float2 _f2_784 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1058 = _f2_784;
                float2 updated_pair_1059 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1057, acc_pair_1058);
                acc32[0] = updated_pair_1059.x;
                acc32[1] = updated_pair_1059.y;
                float2 _f2_785 = make_float2(_tmem_load_33[2], _tmem_load_33[3]);
                float2 cached_pair_1060 = _f2_785;
                float2 _f2_786 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1061 = _f2_786;
                float2 updated_pair_1062 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1060, acc_pair_1061);
                acc32[2] = updated_pair_1062.x;
                acc32[3] = updated_pair_1062.y;
                float2 _f2_787 = make_float2(_tmem_load_33[4], _tmem_load_33[5]);
                float2 cached_pair_1063 = _f2_787;
                float2 _f2_788 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1064 = _f2_788;
                float2 updated_pair_1065 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1063, acc_pair_1064);
                acc32[4] = updated_pair_1065.x;
                acc32[5] = updated_pair_1065.y;
                float2 _f2_789 = make_float2(_tmem_load_33[6], _tmem_load_33[7]);
                float2 cached_pair_1066 = _f2_789;
                float2 _f2_790 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1067 = _f2_790;
                float2 updated_pair_1068 = fma_f32x2_rn_ftz(weight_pair_1056, cached_pair_1066, acc_pair_1067);
                acc32[6] = updated_pair_1068.x;
                acc32[7] = updated_pair_1068.y;
                float2 _f2_791 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1069 = _f2_791;
                float2 _f2_792 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1070 = _f2_792;
                float2 scaled_pair_1071;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1071) : "l"(*(const unsigned long long*)&acc_pair_1070), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[8] = scaled_pair_1071.x;
                acc32[9] = scaled_pair_1071.y;
                float2 _f2_793 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1072 = _f2_793;
                float2 scaled_pair_1073;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1073) : "l"(*(const unsigned long long*)&acc_pair_1072), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[10] = scaled_pair_1073.x;
                acc32[11] = scaled_pair_1073.y;
                float2 _f2_794 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1074 = _f2_794;
                float2 scaled_pair_1075;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1075) : "l"(*(const unsigned long long*)&acc_pair_1074), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[12] = scaled_pair_1075.x;
                acc32[13] = scaled_pair_1075.y;
                float2 _f2_795 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1076 = _f2_795;
                float2 scaled_pair_1077;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1077) : "l"(*(const unsigned long long*)&acc_pair_1076), "l"(*(const unsigned long long*)&correction_pair_1069));
                acc32[14] = scaled_pair_1077.x;
                acc32[15] = scaled_pair_1077.y;
                float _tmem_load_34[8];
                tmem_ld_x8(&_tmem_load_34[0], tmem_base + 8);
                float2 _f2_796 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1078 = _f2_796;
                float2 _f2_797 = make_float2(_tmem_load_34[0], _tmem_load_34[1]);
                float2 cached_pair_1079 = _f2_797;
                float2 _f2_798 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1080 = _f2_798;
                float2 updated_pair_1081 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1079, acc_pair_1080);
                acc32[8] = updated_pair_1081.x;
                acc32[9] = updated_pair_1081.y;
                float2 _f2_799 = make_float2(_tmem_load_34[2], _tmem_load_34[3]);
                float2 cached_pair_1082 = _f2_799;
                float2 _f2_800 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1083 = _f2_800;
                float2 updated_pair_1084 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1082, acc_pair_1083);
                acc32[10] = updated_pair_1084.x;
                acc32[11] = updated_pair_1084.y;
                float2 _f2_801 = make_float2(_tmem_load_34[4], _tmem_load_34[5]);
                float2 cached_pair_1085 = _f2_801;
                float2 _f2_802 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1086 = _f2_802;
                float2 updated_pair_1087 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1085, acc_pair_1086);
                acc32[12] = updated_pair_1087.x;
                acc32[13] = updated_pair_1087.y;
                float2 _f2_803 = make_float2(_tmem_load_34[6], _tmem_load_34[7]);
                float2 cached_pair_1088 = _f2_803;
                float2 _f2_804 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1089 = _f2_804;
                float2 updated_pair_1090 = fma_f32x2_rn_ftz(weight_pair_1078, cached_pair_1088, acc_pair_1089);
                acc32[14] = updated_pair_1090.x;
                acc32[15] = updated_pair_1090.y;
                float _tmem_load_35[8];
                tmem_ld_x8(&_tmem_load_35[0], tmem_base + 64 + 8);
                float2 _f2_805 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1091 = _f2_805;
                float2 _f2_806 = make_float2(_tmem_load_35[0], _tmem_load_35[1]);
                float2 cached_pair_1092 = _f2_806;
                float2 _f2_807 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1093 = _f2_807;
                float2 updated_pair_1094 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1092, acc_pair_1093);
                acc32[8] = updated_pair_1094.x;
                acc32[9] = updated_pair_1094.y;
                float2 _f2_808 = make_float2(_tmem_load_35[2], _tmem_load_35[3]);
                float2 cached_pair_1095 = _f2_808;
                float2 _f2_809 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1096 = _f2_809;
                float2 updated_pair_1097 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1095, acc_pair_1096);
                acc32[10] = updated_pair_1097.x;
                acc32[11] = updated_pair_1097.y;
                float2 _f2_810 = make_float2(_tmem_load_35[4], _tmem_load_35[5]);
                float2 cached_pair_1098 = _f2_810;
                float2 _f2_811 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1099 = _f2_811;
                float2 updated_pair_1100 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1098, acc_pair_1099);
                acc32[12] = updated_pair_1100.x;
                acc32[13] = updated_pair_1100.y;
                float2 _f2_812 = make_float2(_tmem_load_35[6], _tmem_load_35[7]);
                float2 cached_pair_1101 = _f2_812;
                float2 _f2_813 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1102 = _f2_813;
                float2 updated_pair_1103 = fma_f32x2_rn_ftz(weight_pair_1091, cached_pair_1101, acc_pair_1102);
                acc32[14] = updated_pair_1103.x;
                acc32[15] = updated_pair_1103.y;
                float2 _f2_814 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1104 = _f2_814;
                float2 _f2_815 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1105 = _f2_815;
                float2 scaled_pair_1106;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1106) : "l"(*(const unsigned long long*)&acc_pair_1105), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[16] = scaled_pair_1106.x;
                acc32[17] = scaled_pair_1106.y;
                float2 _f2_816 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1107 = _f2_816;
                float2 scaled_pair_1108;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1108) : "l"(*(const unsigned long long*)&acc_pair_1107), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[18] = scaled_pair_1108.x;
                acc32[19] = scaled_pair_1108.y;
                float2 _f2_817 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1109 = _f2_817;
                float2 scaled_pair_1110;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1110) : "l"(*(const unsigned long long*)&acc_pair_1109), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[20] = scaled_pair_1110.x;
                acc32[21] = scaled_pair_1110.y;
                float2 _f2_818 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1111 = _f2_818;
                float2 scaled_pair_1112;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1112) : "l"(*(const unsigned long long*)&acc_pair_1111), "l"(*(const unsigned long long*)&correction_pair_1104));
                acc32[22] = scaled_pair_1112.x;
                acc32[23] = scaled_pair_1112.y;
                float _tmem_load_36[8];
                tmem_ld_x8(&_tmem_load_36[0], tmem_base + 16);
                float2 _f2_819 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1113 = _f2_819;
                float2 _f2_820 = make_float2(_tmem_load_36[0], _tmem_load_36[1]);
                float2 cached_pair_1114 = _f2_820;
                float2 _f2_821 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1115 = _f2_821;
                float2 updated_pair_1116 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1114, acc_pair_1115);
                acc32[16] = updated_pair_1116.x;
                acc32[17] = updated_pair_1116.y;
                float2 _f2_822 = make_float2(_tmem_load_36[2], _tmem_load_36[3]);
                float2 cached_pair_1117 = _f2_822;
                float2 _f2_823 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1118 = _f2_823;
                float2 updated_pair_1119 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1117, acc_pair_1118);
                acc32[18] = updated_pair_1119.x;
                acc32[19] = updated_pair_1119.y;
                float2 _f2_824 = make_float2(_tmem_load_36[4], _tmem_load_36[5]);
                float2 cached_pair_1120 = _f2_824;
                float2 _f2_825 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1121 = _f2_825;
                float2 updated_pair_1122 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1120, acc_pair_1121);
                acc32[20] = updated_pair_1122.x;
                acc32[21] = updated_pair_1122.y;
                float2 _f2_826 = make_float2(_tmem_load_36[6], _tmem_load_36[7]);
                float2 cached_pair_1123 = _f2_826;
                float2 _f2_827 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1124 = _f2_827;
                float2 updated_pair_1125 = fma_f32x2_rn_ftz(weight_pair_1113, cached_pair_1123, acc_pair_1124);
                acc32[22] = updated_pair_1125.x;
                acc32[23] = updated_pair_1125.y;
                float _tmem_load_37[8];
                tmem_ld_x8(&_tmem_load_37[0], tmem_base + 64 + 16);
                float2 _f2_828 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1126 = _f2_828;
                float2 _f2_829 = make_float2(_tmem_load_37[0], _tmem_load_37[1]);
                float2 cached_pair_1127 = _f2_829;
                float2 _f2_830 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1128 = _f2_830;
                float2 updated_pair_1129 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1127, acc_pair_1128);
                acc32[16] = updated_pair_1129.x;
                acc32[17] = updated_pair_1129.y;
                float2 _f2_831 = make_float2(_tmem_load_37[2], _tmem_load_37[3]);
                float2 cached_pair_1130 = _f2_831;
                float2 _f2_832 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1131 = _f2_832;
                float2 updated_pair_1132 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1130, acc_pair_1131);
                acc32[18] = updated_pair_1132.x;
                acc32[19] = updated_pair_1132.y;
                float2 _f2_833 = make_float2(_tmem_load_37[4], _tmem_load_37[5]);
                float2 cached_pair_1133 = _f2_833;
                float2 _f2_834 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1134 = _f2_834;
                float2 updated_pair_1135 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1133, acc_pair_1134);
                acc32[20] = updated_pair_1135.x;
                acc32[21] = updated_pair_1135.y;
                float2 _f2_835 = make_float2(_tmem_load_37[6], _tmem_load_37[7]);
                float2 cached_pair_1136 = _f2_835;
                float2 _f2_836 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1137 = _f2_836;
                float2 updated_pair_1138 = fma_f32x2_rn_ftz(weight_pair_1126, cached_pair_1136, acc_pair_1137);
                acc32[22] = updated_pair_1138.x;
                acc32[23] = updated_pair_1138.y;
                float2 _f2_837 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1139 = _f2_837;
                float2 _f2_838 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1140 = _f2_838;
                float2 scaled_pair_1141;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1141) : "l"(*(const unsigned long long*)&acc_pair_1140), "l"(*(const unsigned long long*)&correction_pair_1139));
                acc32[24] = scaled_pair_1141.x;
                acc32[25] = scaled_pair_1141.y;
                float2 _f2_839 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1142 = _f2_839;
                float2 scaled_pair_1143;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1143) : "l"(*(const unsigned long long*)&acc_pair_1142), "l"(*(const unsigned long long*)&correction_pair_1139));
                acc32[26] = scaled_pair_1143.x;
                acc32[27] = scaled_pair_1143.y;
                float _tmem_load_38[4];
                tmem_ld_x4(&_tmem_load_38[0], tmem_base + 24);
                float2 _f2_840 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1144 = _f2_840;
                float2 _f2_841 = make_float2(_tmem_load_38[0], _tmem_load_38[1]);
                float2 cached_pair_1145 = _f2_841;
                float2 _f2_842 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1146 = _f2_842;
                float2 updated_pair_1147 = fma_f32x2_rn_ftz(weight_pair_1144, cached_pair_1145, acc_pair_1146);
                acc32[24] = updated_pair_1147.x;
                acc32[25] = updated_pair_1147.y;
                float2 _f2_843 = make_float2(_tmem_load_38[2], _tmem_load_38[3]);
                float2 cached_pair_1148 = _f2_843;
                float2 _f2_844 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1149 = _f2_844;
                float2 updated_pair_1150 = fma_f32x2_rn_ftz(weight_pair_1144, cached_pair_1148, acc_pair_1149);
                acc32[26] = updated_pair_1150.x;
                acc32[27] = updated_pair_1150.y;
                float _tmem_load_39[4];
                tmem_ld_x4(&_tmem_load_39[0], tmem_base + 64 + 24);
                float2 _f2_845 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1151 = _f2_845;
                float2 _f2_846 = make_float2(_tmem_load_39[0], _tmem_load_39[1]);
                float2 cached_pair_1152 = _f2_846;
                float2 _f2_847 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1153 = _f2_847;
                float2 updated_pair_1154 = fma_f32x2_rn_ftz(weight_pair_1151, cached_pair_1152, acc_pair_1153);
                acc32[24] = updated_pair_1154.x;
                acc32[25] = updated_pair_1154.y;
                float2 _f2_848 = make_float2(_tmem_load_39[2], _tmem_load_39[3]);
                float2 cached_pair_1155 = _f2_848;
                float2 _f2_849 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1156 = _f2_849;
                float2 updated_pair_1157 = fma_f32x2_rn_ftz(weight_pair_1151, cached_pair_1155, acc_pair_1156);
                acc32[26] = updated_pair_1157.x;
                acc32[27] = updated_pair_1157.y;
                float2 _f2_850 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1158 = _f2_850;
                float2 _f2_851 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1159 = _f2_851;
                float2 scaled_pair_1160;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1160) : "l"(*(const unsigned long long*)&acc_pair_1159), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[28] = scaled_pair_1160.x;
                acc32[29] = scaled_pair_1160.y;
                float2 _f2_852 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1161 = _f2_852;
                float2 scaled_pair_1162;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1162) : "l"(*(const unsigned long long*)&acc_pair_1161), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[30] = scaled_pair_1162.x;
                acc32[31] = scaled_pair_1162.y;
                float2 _f2_853 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1163 = _f2_853;
                float2 scaled_pair_1164;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1164) : "l"(*(const unsigned long long*)&acc_pair_1163), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[32] = scaled_pair_1164.x;
                acc32[33] = scaled_pair_1164.y;
                float2 _f2_854 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1165 = _f2_854;
                float2 scaled_pair_1166;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1166) : "l"(*(const unsigned long long*)&acc_pair_1165), "l"(*(const unsigned long long*)&correction_pair_1158));
                acc32[34] = scaled_pair_1166.x;
                acc32[35] = scaled_pair_1166.y;
                float _tmem_load_40[8];
                tmem_ld_x8(&_tmem_load_40[0], tmem_base + 32);
                float2 _f2_855 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1167 = _f2_855;
                float2 _f2_856 = make_float2(_tmem_load_40[0], _tmem_load_40[1]);
                float2 cached_pair_1168 = _f2_856;
                float2 _f2_857 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1169 = _f2_857;
                float2 updated_pair_1170 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1168, acc_pair_1169);
                acc32[28] = updated_pair_1170.x;
                acc32[29] = updated_pair_1170.y;
                float2 _f2_858 = make_float2(_tmem_load_40[2], _tmem_load_40[3]);
                float2 cached_pair_1171 = _f2_858;
                float2 _f2_859 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1172 = _f2_859;
                float2 updated_pair_1173 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1171, acc_pair_1172);
                acc32[30] = updated_pair_1173.x;
                acc32[31] = updated_pair_1173.y;
                float2 _f2_860 = make_float2(_tmem_load_40[4], _tmem_load_40[5]);
                float2 cached_pair_1174 = _f2_860;
                float2 _f2_861 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1175 = _f2_861;
                float2 updated_pair_1176 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1174, acc_pair_1175);
                acc32[32] = updated_pair_1176.x;
                acc32[33] = updated_pair_1176.y;
                float2 _f2_862 = make_float2(_tmem_load_40[6], _tmem_load_40[7]);
                float2 cached_pair_1177 = _f2_862;
                float2 _f2_863 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1178 = _f2_863;
                float2 updated_pair_1179 = fma_f32x2_rn_ftz(weight_pair_1167, cached_pair_1177, acc_pair_1178);
                acc32[34] = updated_pair_1179.x;
                acc32[35] = updated_pair_1179.y;
                float _tmem_load_41[8];
                tmem_ld_x8(&_tmem_load_41[0], tmem_base + 64 + 32);
                float2 _f2_864 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1180 = _f2_864;
                float2 _f2_865 = make_float2(_tmem_load_41[0], _tmem_load_41[1]);
                float2 cached_pair_1181 = _f2_865;
                float2 _f2_866 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1182 = _f2_866;
                float2 updated_pair_1183 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1181, acc_pair_1182);
                acc32[28] = updated_pair_1183.x;
                acc32[29] = updated_pair_1183.y;
                float2 _f2_867 = make_float2(_tmem_load_41[2], _tmem_load_41[3]);
                float2 cached_pair_1184 = _f2_867;
                float2 _f2_868 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1185 = _f2_868;
                float2 updated_pair_1186 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1184, acc_pair_1185);
                acc32[30] = updated_pair_1186.x;
                acc32[31] = updated_pair_1186.y;
                float2 _f2_869 = make_float2(_tmem_load_41[4], _tmem_load_41[5]);
                float2 cached_pair_1187 = _f2_869;
                float2 _f2_870 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1188 = _f2_870;
                float2 updated_pair_1189 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1187, acc_pair_1188);
                acc32[32] = updated_pair_1189.x;
                acc32[33] = updated_pair_1189.y;
                float2 _f2_871 = make_float2(_tmem_load_41[6], _tmem_load_41[7]);
                float2 cached_pair_1190 = _f2_871;
                float2 _f2_872 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1191 = _f2_872;
                float2 updated_pair_1192 = fma_f32x2_rn_ftz(weight_pair_1180, cached_pair_1190, acc_pair_1191);
                acc32[34] = updated_pair_1192.x;
                acc32[35] = updated_pair_1192.y;
                float2 _f2_873 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1193 = _f2_873;
                float2 _f2_874 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1194 = _f2_874;
                float2 scaled_pair_1195;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1195) : "l"(*(const unsigned long long*)&acc_pair_1194), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[36] = scaled_pair_1195.x;
                acc32[37] = scaled_pair_1195.y;
                float2 _f2_875 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1196 = _f2_875;
                float2 scaled_pair_1197;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1197) : "l"(*(const unsigned long long*)&acc_pair_1196), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[38] = scaled_pair_1197.x;
                acc32[39] = scaled_pair_1197.y;
                float2 _f2_876 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1198 = _f2_876;
                float2 scaled_pair_1199;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1199) : "l"(*(const unsigned long long*)&acc_pair_1198), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[40] = scaled_pair_1199.x;
                acc32[41] = scaled_pair_1199.y;
                float2 _f2_877 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1200 = _f2_877;
                float2 scaled_pair_1201;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1201) : "l"(*(const unsigned long long*)&acc_pair_1200), "l"(*(const unsigned long long*)&correction_pair_1193));
                acc32[42] = scaled_pair_1201.x;
                acc32[43] = scaled_pair_1201.y;
                float _tmem_load_42[8];
                tmem_ld_x8(&_tmem_load_42[0], tmem_base + 40);
                float2 _f2_878 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1202 = _f2_878;
                float2 _f2_879 = make_float2(_tmem_load_42[0], _tmem_load_42[1]);
                float2 cached_pair_1203 = _f2_879;
                float2 _f2_880 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1204 = _f2_880;
                float2 updated_pair_1205 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1203, acc_pair_1204);
                acc32[36] = updated_pair_1205.x;
                acc32[37] = updated_pair_1205.y;
                float2 _f2_881 = make_float2(_tmem_load_42[2], _tmem_load_42[3]);
                float2 cached_pair_1206 = _f2_881;
                float2 _f2_882 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1207 = _f2_882;
                float2 updated_pair_1208 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1206, acc_pair_1207);
                acc32[38] = updated_pair_1208.x;
                acc32[39] = updated_pair_1208.y;
                float2 _f2_883 = make_float2(_tmem_load_42[4], _tmem_load_42[5]);
                float2 cached_pair_1209 = _f2_883;
                float2 _f2_884 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1210 = _f2_884;
                float2 updated_pair_1211 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1209, acc_pair_1210);
                acc32[40] = updated_pair_1211.x;
                acc32[41] = updated_pair_1211.y;
                float2 _f2_885 = make_float2(_tmem_load_42[6], _tmem_load_42[7]);
                float2 cached_pair_1212 = _f2_885;
                float2 _f2_886 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1213 = _f2_886;
                float2 updated_pair_1214 = fma_f32x2_rn_ftz(weight_pair_1202, cached_pair_1212, acc_pair_1213);
                acc32[42] = updated_pair_1214.x;
                acc32[43] = updated_pair_1214.y;
                float _tmem_load_43[8];
                tmem_ld_x8(&_tmem_load_43[0], tmem_base + 64 + 40);
                float2 _f2_887 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1215 = _f2_887;
                float2 _f2_888 = make_float2(_tmem_load_43[0], _tmem_load_43[1]);
                float2 cached_pair_1216 = _f2_888;
                float2 _f2_889 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1217 = _f2_889;
                float2 updated_pair_1218 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1216, acc_pair_1217);
                acc32[36] = updated_pair_1218.x;
                acc32[37] = updated_pair_1218.y;
                float2 _f2_890 = make_float2(_tmem_load_43[2], _tmem_load_43[3]);
                float2 cached_pair_1219 = _f2_890;
                float2 _f2_891 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1220 = _f2_891;
                float2 updated_pair_1221 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1219, acc_pair_1220);
                acc32[38] = updated_pair_1221.x;
                acc32[39] = updated_pair_1221.y;
                float2 _f2_892 = make_float2(_tmem_load_43[4], _tmem_load_43[5]);
                float2 cached_pair_1222 = _f2_892;
                float2 _f2_893 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1223 = _f2_893;
                float2 updated_pair_1224 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1222, acc_pair_1223);
                acc32[40] = updated_pair_1224.x;
                acc32[41] = updated_pair_1224.y;
                float2 _f2_894 = make_float2(_tmem_load_43[6], _tmem_load_43[7]);
                float2 cached_pair_1225 = _f2_894;
                float2 _f2_895 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1226 = _f2_895;
                float2 updated_pair_1227 = fma_f32x2_rn_ftz(weight_pair_1215, cached_pair_1225, acc_pair_1226);
                acc32[42] = updated_pair_1227.x;
                acc32[43] = updated_pair_1227.y;
                float2 _f2_896 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1228 = _f2_896;
                float2 _f2_897 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1229 = _f2_897;
                float2 scaled_pair_1230;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1230) : "l"(*(const unsigned long long*)&acc_pair_1229), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[44] = scaled_pair_1230.x;
                acc32[45] = scaled_pair_1230.y;
                float2 _f2_898 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1231 = _f2_898;
                float2 scaled_pair_1232;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1232) : "l"(*(const unsigned long long*)&acc_pair_1231), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[46] = scaled_pair_1232.x;
                acc32[47] = scaled_pair_1232.y;
                float2 _f2_899 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1233 = _f2_899;
                float2 scaled_pair_1234;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1234) : "l"(*(const unsigned long long*)&acc_pair_1233), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[48] = scaled_pair_1234.x;
                acc32[49] = scaled_pair_1234.y;
                float2 _f2_900 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1235 = _f2_900;
                float2 scaled_pair_1236;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1236) : "l"(*(const unsigned long long*)&acc_pair_1235), "l"(*(const unsigned long long*)&correction_pair_1228));
                acc32[50] = scaled_pair_1236.x;
                acc32[51] = scaled_pair_1236.y;
                float _tmem_load_44[8];
                tmem_ld_x8(&_tmem_load_44[0], tmem_base + 48);
                float2 _f2_901 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1237 = _f2_901;
                float2 _f2_902 = make_float2(_tmem_load_44[0], _tmem_load_44[1]);
                float2 cached_pair_1238 = _f2_902;
                float2 _f2_903 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1239 = _f2_903;
                float2 updated_pair_1240 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1238, acc_pair_1239);
                acc32[44] = updated_pair_1240.x;
                acc32[45] = updated_pair_1240.y;
                float2 _f2_904 = make_float2(_tmem_load_44[2], _tmem_load_44[3]);
                float2 cached_pair_1241 = _f2_904;
                float2 _f2_905 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1242 = _f2_905;
                float2 updated_pair_1243 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1241, acc_pair_1242);
                acc32[46] = updated_pair_1243.x;
                acc32[47] = updated_pair_1243.y;
                float2 _f2_906 = make_float2(_tmem_load_44[4], _tmem_load_44[5]);
                float2 cached_pair_1244 = _f2_906;
                float2 _f2_907 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1245 = _f2_907;
                float2 updated_pair_1246 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1244, acc_pair_1245);
                acc32[48] = updated_pair_1246.x;
                acc32[49] = updated_pair_1246.y;
                float2 _f2_908 = make_float2(_tmem_load_44[6], _tmem_load_44[7]);
                float2 cached_pair_1247 = _f2_908;
                float2 _f2_909 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1248 = _f2_909;
                float2 updated_pair_1249 = fma_f32x2_rn_ftz(weight_pair_1237, cached_pair_1247, acc_pair_1248);
                acc32[50] = updated_pair_1249.x;
                acc32[51] = updated_pair_1249.y;
                float _tmem_load_45[8];
                tmem_ld_x8(&_tmem_load_45[0], tmem_base + 64 + 48);
                float2 _f2_910 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1250 = _f2_910;
                float2 _f2_911 = make_float2(_tmem_load_45[0], _tmem_load_45[1]);
                float2 cached_pair_1251 = _f2_911;
                float2 _f2_912 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1252 = _f2_912;
                float2 updated_pair_1253 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1251, acc_pair_1252);
                acc32[44] = updated_pair_1253.x;
                acc32[45] = updated_pair_1253.y;
                float2 _f2_913 = make_float2(_tmem_load_45[2], _tmem_load_45[3]);
                float2 cached_pair_1254 = _f2_913;
                float2 _f2_914 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1255 = _f2_914;
                float2 updated_pair_1256 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1254, acc_pair_1255);
                acc32[46] = updated_pair_1256.x;
                acc32[47] = updated_pair_1256.y;
                float2 _f2_915 = make_float2(_tmem_load_45[4], _tmem_load_45[5]);
                float2 cached_pair_1257 = _f2_915;
                float2 _f2_916 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1258 = _f2_916;
                float2 updated_pair_1259 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1257, acc_pair_1258);
                acc32[48] = updated_pair_1259.x;
                acc32[49] = updated_pair_1259.y;
                float2 _f2_917 = make_float2(_tmem_load_45[6], _tmem_load_45[7]);
                float2 cached_pair_1260 = _f2_917;
                float2 _f2_918 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1261 = _f2_918;
                float2 updated_pair_1262 = fma_f32x2_rn_ftz(weight_pair_1250, cached_pair_1260, acc_pair_1261);
                acc32[50] = updated_pair_1262.x;
                acc32[51] = updated_pair_1262.y;
                float2 _f2_919 = make_float2(correction_1031, correction_1031);
                float2 correction_pair_1263 = _f2_919;
                float2 _f2_920 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1264 = _f2_920;
                float2 scaled_pair_1265;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1265) : "l"(*(const unsigned long long*)&acc_pair_1264), "l"(*(const unsigned long long*)&correction_pair_1263));
                acc32[52] = scaled_pair_1265.x;
                acc32[53] = scaled_pair_1265.y;
                float2 _f2_921 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1266 = _f2_921;
                float2 scaled_pair_1267;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1267) : "l"(*(const unsigned long long*)&acc_pair_1266), "l"(*(const unsigned long long*)&correction_pair_1263));
                acc32[54] = scaled_pair_1267.x;
                acc32[55] = scaled_pair_1267.y;
                float _tmem_load_46[4];
                tmem_ld_x4(&_tmem_load_46[0], tmem_base + 56);
                float2 _f2_922 = make_float2(weights_1029[0], weights_1029[0]);
                float2 weight_pair_1268 = _f2_922;
                float2 _f2_923 = make_float2(_tmem_load_46[0], _tmem_load_46[1]);
                float2 cached_pair_1269 = _f2_923;
                float2 _f2_924 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1270 = _f2_924;
                float2 updated_pair_1271 = fma_f32x2_rn_ftz(weight_pair_1268, cached_pair_1269, acc_pair_1270);
                acc32[52] = updated_pair_1271.x;
                acc32[53] = updated_pair_1271.y;
                float2 _f2_925 = make_float2(_tmem_load_46[2], _tmem_load_46[3]);
                float2 cached_pair_1272 = _f2_925;
                float2 _f2_926 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1273 = _f2_926;
                float2 updated_pair_1274 = fma_f32x2_rn_ftz(weight_pair_1268, cached_pair_1272, acc_pair_1273);
                acc32[54] = updated_pair_1274.x;
                acc32[55] = updated_pair_1274.y;
                float _tmem_load_47[4];
                tmem_ld_x4(&_tmem_load_47[0], tmem_base + 64 + 56);
                float2 _f2_927 = make_float2(weights_1029[1], weights_1029[1]);
                float2 weight_pair_1275 = _f2_927;
                float2 _f2_928 = make_float2(_tmem_load_47[0], _tmem_load_47[1]);
                float2 cached_pair_1276 = _f2_928;
                float2 _f2_929 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1277 = _f2_929;
                float2 updated_pair_1278 = fma_f32x2_rn_ftz(weight_pair_1275, cached_pair_1276, acc_pair_1277);
                acc32[52] = updated_pair_1278.x;
                acc32[53] = updated_pair_1278.y;
                float2 _f2_930 = make_float2(_tmem_load_47[2], _tmem_load_47[3]);
                float2 cached_pair_1279 = _f2_930;
                float2 _f2_931 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1280 = _f2_931;
                float2 updated_pair_1281 = fma_f32x2_rn_ftz(weight_pair_1275, cached_pair_1279, acc_pair_1280);
                acc32[54] = updated_pair_1281.x;
                acc32[55] = updated_pair_1281.y;
                s_running = __fmaf_rn(s_running, correction_1031, weight_sum_1032);
                m_running = new_max_1030;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; consumer_phase ^= 1; }
                unsigned int ring_stage_1282 = consumer_stage + stream_u_1 * 2;
                mbarrier_wait(ready_addr + (ring_stage_1282) * 8, consumer_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                float2 local_sum_sq_1283[4];
                float2 local_dot_1284[4];
                float2 _f2_932 = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[0] = _f2_932;
                float2 _f2_933 = make_float2(0.0f, 0.0f);
                local_dot_1284[0] = _f2_933;
                float2 _f2_934 = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[1] = _f2_934;
                float2 _f2_935 = make_float2(0.0f, 0.0f);
                local_dot_1284[1] = _f2_935;
                float2 _f2_936 = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[2] = _f2_936;
                float2 _f2_937 = make_float2(0.0f, 0.0f);
                local_dot_1284[2] = _f2_937;
                float2 _f2_938 = make_float2(0.0f, 0.0f);
                local_sum_sq_1283[3] = _f2_938;
                float2 _f2_939 = make_float2(0.0f, 0.0f);
                local_dot_1284[3] = _f2_939;
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
                float _cvt_f32_0 = (float)updated_bf16_pair.x;
                values_1287[0] = _cvt_f32_0;
                float _cvt_f32_1 = (float)updated_bf16_pair.y;
                values_1287[1] = _cvt_f32_1;
                __nv_bfloat162 prefix_pair_1289 = __as_bf16x2(smem_regs_1286[1]);
                __nv_bfloat162 delta_pair_1290 = __as_bf16x2(smem_regs_1288[1]);
                __nv_bfloat162 updated_bf16_pair_1291 = prefix_pair_1289 + delta_pair_1290;
                float _cvt_f32_2 = (float)updated_bf16_pair_1291.x;
                values_1287[2] = _cvt_f32_2;
                float _cvt_f32_3 = (float)updated_bf16_pair_1291.y;
                values_1287[3] = _cvt_f32_3;
                __nv_bfloat162 prefix_pair_1292 = __as_bf16x2(smem_regs_1286[2]);
                __nv_bfloat162 delta_pair_1293 = __as_bf16x2(smem_regs_1288[2]);
                __nv_bfloat162 updated_bf16_pair_1294 = prefix_pair_1292 + delta_pair_1293;
                float _cvt_f32_4 = (float)updated_bf16_pair_1294.x;
                values_1287[4] = _cvt_f32_4;
                float _cvt_f32_5 = (float)updated_bf16_pair_1294.y;
                values_1287[5] = _cvt_f32_5;
                __nv_bfloat162 prefix_pair_1295 = __as_bf16x2(smem_regs_1286[3]);
                __nv_bfloat162 delta_pair_1296 = __as_bf16x2(smem_regs_1288[3]);
                __nv_bfloat162 updated_bf16_pair_1297 = prefix_pair_1295 + delta_pair_1296;
                float _cvt_f32_6 = (float)updated_bf16_pair_1297.x;
                values_1287[6] = _cvt_f32_6;
                float _cvt_f32_7 = (float)updated_bf16_pair_1297.y;
                values_1287[7] = _cvt_f32_7;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_1287[0 + 0], values_1287[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1287[0 + 2], values_1287[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_1287[0 + 4], values_1287[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_1287[0 + 6], values_1287[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1285 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base, values_1287);
                float2 _f2_940 = make_float2(values_1287[0], values_1287[1]);
                float2 value_pair_1298 = _f2_940;
                float2 _f2_941 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_1299 = _f2_941;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1298, value_pair_1298, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1298, q_pair_1299, local_dot_1284[0]);
                float2 _f2_942 = make_float2(values_1287[2], values_1287[3]);
                float2 value_pair_1300 = _f2_942;
                float2 _f2_943 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_1301 = _f2_943;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1300, value_pair_1300, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1300, q_pair_1301, local_dot_1284[0]);
                float2 _f2_944 = make_float2(values_1287[4], values_1287[5]);
                float2 value_pair_1302 = _f2_944;
                float2 _f2_945 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_1303 = _f2_945;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1302, value_pair_1302, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1302, q_pair_1303, local_dot_1284[0]);
                float2 _f2_946 = make_float2(values_1287[6], values_1287[7]);
                float2 value_pair_1304 = _f2_946;
                float2 _f2_947 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_1305 = _f2_947;
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
                float _cvt_f32_8 = (float)updated_bf16_pair_1312.x;
                values_1308[0] = _cvt_f32_8;
                float _cvt_f32_9 = (float)updated_bf16_pair_1312.y;
                values_1308[1] = _cvt_f32_9;
                __nv_bfloat162 prefix_pair_1313 = __as_bf16x2(smem_regs_1307[1]);
                __nv_bfloat162 delta_pair_1314 = __as_bf16x2(smem_regs_1309[1]);
                __nv_bfloat162 updated_bf16_pair_1315 = prefix_pair_1313 + delta_pair_1314;
                float _cvt_f32_10 = (float)updated_bf16_pair_1315.x;
                values_1308[2] = _cvt_f32_10;
                float _cvt_f32_11 = (float)updated_bf16_pair_1315.y;
                values_1308[3] = _cvt_f32_11;
                __nv_bfloat162 prefix_pair_1316 = __as_bf16x2(smem_regs_1307[2]);
                __nv_bfloat162 delta_pair_1317 = __as_bf16x2(smem_regs_1309[2]);
                __nv_bfloat162 updated_bf16_pair_1318 = prefix_pair_1316 + delta_pair_1317;
                float _cvt_f32_12 = (float)updated_bf16_pair_1318.x;
                values_1308[4] = _cvt_f32_12;
                float _cvt_f32_13 = (float)updated_bf16_pair_1318.y;
                values_1308[5] = _cvt_f32_13;
                __nv_bfloat162 prefix_pair_1319 = __as_bf16x2(smem_regs_1307[3]);
                __nv_bfloat162 delta_pair_1320 = __as_bf16x2(smem_regs_1309[3]);
                __nv_bfloat162 updated_bf16_pair_1321 = prefix_pair_1319 + delta_pair_1320;
                float _cvt_f32_14 = (float)updated_bf16_pair_1321.x;
                values_1308[6] = _cvt_f32_14;
                float _cvt_f32_15 = (float)updated_bf16_pair_1321.y;
                values_1308[7] = _cvt_f32_15;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_1308[0 + 0], values_1308[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1308[0 + 2], values_1308[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_1308[0 + 4], values_1308[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_1308[0 + 6], values_1308[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1306 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 8, values_1308);
                float2 _f2_948 = make_float2(values_1308[0], values_1308[1]);
                float2 value_pair_1322 = _f2_948;
                float2 _f2_949 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_1323 = _f2_949;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1322, value_pair_1322, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1322, q_pair_1323, local_dot_1284[0]);
                float2 _f2_950 = make_float2(values_1308[2], values_1308[3]);
                float2 value_pair_1324 = _f2_950;
                float2 _f2_951 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_1325 = _f2_951;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1324, value_pair_1324, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1324, q_pair_1325, local_dot_1284[0]);
                float2 _f2_952 = make_float2(values_1308[4], values_1308[5]);
                float2 value_pair_1326 = _f2_952;
                float2 _f2_953 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_1327 = _f2_953;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1326, value_pair_1326, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1326, q_pair_1327, local_dot_1284[0]);
                float2 _f2_954 = make_float2(values_1308[6], values_1308[7]);
                float2 value_pair_1328 = _f2_954;
                float2 _f2_955 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_1329 = _f2_955;
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
                float _cvt_f32_16 = (float)updated_bf16_pair_1336.x;
                values_1332[0] = _cvt_f32_16;
                float _cvt_f32_17 = (float)updated_bf16_pair_1336.y;
                values_1332[1] = _cvt_f32_17;
                __nv_bfloat162 prefix_pair_1337 = __as_bf16x2(smem_regs_1331[1]);
                __nv_bfloat162 delta_pair_1338 = __as_bf16x2(smem_regs_1333[1]);
                __nv_bfloat162 updated_bf16_pair_1339 = prefix_pair_1337 + delta_pair_1338;
                float _cvt_f32_18 = (float)updated_bf16_pair_1339.x;
                values_1332[2] = _cvt_f32_18;
                float _cvt_f32_19 = (float)updated_bf16_pair_1339.y;
                values_1332[3] = _cvt_f32_19;
                __nv_bfloat162 prefix_pair_1340 = __as_bf16x2(smem_regs_1331[2]);
                __nv_bfloat162 delta_pair_1341 = __as_bf16x2(smem_regs_1333[2]);
                __nv_bfloat162 updated_bf16_pair_1342 = prefix_pair_1340 + delta_pair_1341;
                float _cvt_f32_20 = (float)updated_bf16_pair_1342.x;
                values_1332[4] = _cvt_f32_20;
                float _cvt_f32_21 = (float)updated_bf16_pair_1342.y;
                values_1332[5] = _cvt_f32_21;
                __nv_bfloat162 prefix_pair_1343 = __as_bf16x2(smem_regs_1331[3]);
                __nv_bfloat162 delta_pair_1344 = __as_bf16x2(smem_regs_1333[3]);
                __nv_bfloat162 updated_bf16_pair_1345 = prefix_pair_1343 + delta_pair_1344;
                float _cvt_f32_22 = (float)updated_bf16_pair_1345.x;
                values_1332[6] = _cvt_f32_22;
                float _cvt_f32_23 = (float)updated_bf16_pair_1345.y;
                values_1332[7] = _cvt_f32_23;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_1332[0 + 0], values_1332[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1332[0 + 2], values_1332[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_1332[0 + 4], values_1332[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_1332[0 + 6], values_1332[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1330 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 16, values_1332);
                float2 _f2_956 = make_float2(values_1332[0], values_1332[1]);
                float2 value_pair_1346 = _f2_956;
                float2 _f2_957 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_1347 = _f2_957;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1346, value_pair_1346, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1346, q_pair_1347, local_dot_1284[0]);
                float2 _f2_958 = make_float2(values_1332[2], values_1332[3]);
                float2 value_pair_1348 = _f2_958;
                float2 _f2_959 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_1349 = _f2_959;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1348, value_pair_1348, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1348, q_pair_1349, local_dot_1284[0]);
                float2 _f2_960 = make_float2(values_1332[4], values_1332[5]);
                float2 value_pair_1350 = _f2_960;
                float2 _f2_961 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_1351 = _f2_961;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1350, value_pair_1350, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1350, q_pair_1351, local_dot_1284[0]);
                float2 _f2_962 = make_float2(values_1332[6], values_1332[7]);
                float2 value_pair_1352 = _f2_962;
                float2 _f2_963 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_1353 = _f2_963;
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
                float _cvt_f32_24 = (float)updated_bf16_pair_1360.x;
                values_1356[0] = _cvt_f32_24;
                float _cvt_f32_25 = (float)updated_bf16_pair_1360.y;
                values_1356[1] = _cvt_f32_25;
                __nv_bfloat162 prefix_pair_1361 = __as_bf16x2(smem_regs_1355[1]);
                __nv_bfloat162 delta_pair_1362 = __as_bf16x2(smem_regs_1357[1]);
                __nv_bfloat162 updated_bf16_pair_1363 = prefix_pair_1361 + delta_pair_1362;
                float _cvt_f32_26 = (float)updated_bf16_pair_1363.x;
                values_1356[2] = _cvt_f32_26;
                float _cvt_f32_27 = (float)updated_bf16_pair_1363.y;
                values_1356[3] = _cvt_f32_27;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(values_1356[0 + 0], values_1356[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1356[0 + 2], values_1356[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1354]) = _pk2;
                }
                tmem_st_x4_f32(tmem_base + 24, values_1356);
                float2 _f2_964 = make_float2(values_1356[0], values_1356[1]);
                float2 value_pair_1364 = _f2_964;
                float2 _f2_965 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_1365 = _f2_965;
                local_sum_sq_1283[0] = fma_f32x2_rn_ftz(value_pair_1364, value_pair_1364, local_sum_sq_1283[0]);
                local_dot_1284[0] = fma_f32x2_rn_ftz(value_pair_1364, q_pair_1365, local_dot_1284[0]);
                float2 _f2_966 = make_float2(values_1356[2], values_1356[3]);
                float2 value_pair_1366 = _f2_966;
                float2 _f2_967 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_1367 = _f2_967;
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
                float _cvt_f32_28 = (float)updated_bf16_pair_1374.x;
                values_1370[0] = _cvt_f32_28;
                float _cvt_f32_29 = (float)updated_bf16_pair_1374.y;
                values_1370[1] = _cvt_f32_29;
                __nv_bfloat162 prefix_pair_1375 = __as_bf16x2(smem_regs_1369[1]);
                __nv_bfloat162 delta_pair_1376 = __as_bf16x2(smem_regs_1371[1]);
                __nv_bfloat162 updated_bf16_pair_1377 = prefix_pair_1375 + delta_pair_1376;
                float _cvt_f32_30 = (float)updated_bf16_pair_1377.x;
                values_1370[2] = _cvt_f32_30;
                float _cvt_f32_31 = (float)updated_bf16_pair_1377.y;
                values_1370[3] = _cvt_f32_31;
                __nv_bfloat162 prefix_pair_1378 = __as_bf16x2(smem_regs_1369[2]);
                __nv_bfloat162 delta_pair_1379 = __as_bf16x2(smem_regs_1371[2]);
                __nv_bfloat162 updated_bf16_pair_1380 = prefix_pair_1378 + delta_pair_1379;
                float _cvt_f32_32 = (float)updated_bf16_pair_1380.x;
                values_1370[4] = _cvt_f32_32;
                float _cvt_f32_33 = (float)updated_bf16_pair_1380.y;
                values_1370[5] = _cvt_f32_33;
                __nv_bfloat162 prefix_pair_1381 = __as_bf16x2(smem_regs_1369[3]);
                __nv_bfloat162 delta_pair_1382 = __as_bf16x2(smem_regs_1371[3]);
                __nv_bfloat162 updated_bf16_pair_1383 = prefix_pair_1381 + delta_pair_1382;
                float _cvt_f32_34 = (float)updated_bf16_pair_1383.x;
                values_1370[6] = _cvt_f32_34;
                float _cvt_f32_35 = (float)updated_bf16_pair_1383.y;
                values_1370[7] = _cvt_f32_35;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_1370[0 + 0], values_1370[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1370[0 + 2], values_1370[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_1370[0 + 4], values_1370[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_1370[0 + 6], values_1370[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1368 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 32, values_1370);
                float2 _f2_968 = make_float2(values_1370[0], values_1370[1]);
                float2 value_pair_1384 = _f2_968;
                float2 _f2_969 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_1385 = _f2_969;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1384, value_pair_1384, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1384, q_pair_1385, local_dot_1284[2]);
                float2 _f2_970 = make_float2(values_1370[2], values_1370[3]);
                float2 value_pair_1386 = _f2_970;
                float2 _f2_971 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_1387 = _f2_971;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1386, value_pair_1386, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1386, q_pair_1387, local_dot_1284[2]);
                float2 _f2_972 = make_float2(values_1370[4], values_1370[5]);
                float2 value_pair_1388 = _f2_972;
                float2 _f2_973 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_1389 = _f2_973;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1388, value_pair_1388, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1388, q_pair_1389, local_dot_1284[2]);
                float2 _f2_974 = make_float2(values_1370[6], values_1370[7]);
                float2 value_pair_1390 = _f2_974;
                float2 _f2_975 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_1391 = _f2_975;
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
                float _cvt_f32_36 = (float)updated_bf16_pair_1398.x;
                values_1394[0] = _cvt_f32_36;
                float _cvt_f32_37 = (float)updated_bf16_pair_1398.y;
                values_1394[1] = _cvt_f32_37;
                __nv_bfloat162 prefix_pair_1399 = __as_bf16x2(smem_regs_1393[1]);
                __nv_bfloat162 delta_pair_1400 = __as_bf16x2(smem_regs_1395[1]);
                __nv_bfloat162 updated_bf16_pair_1401 = prefix_pair_1399 + delta_pair_1400;
                float _cvt_f32_38 = (float)updated_bf16_pair_1401.x;
                values_1394[2] = _cvt_f32_38;
                float _cvt_f32_39 = (float)updated_bf16_pair_1401.y;
                values_1394[3] = _cvt_f32_39;
                __nv_bfloat162 prefix_pair_1402 = __as_bf16x2(smem_regs_1393[2]);
                __nv_bfloat162 delta_pair_1403 = __as_bf16x2(smem_regs_1395[2]);
                __nv_bfloat162 updated_bf16_pair_1404 = prefix_pair_1402 + delta_pair_1403;
                float _cvt_f32_40 = (float)updated_bf16_pair_1404.x;
                values_1394[4] = _cvt_f32_40;
                float _cvt_f32_41 = (float)updated_bf16_pair_1404.y;
                values_1394[5] = _cvt_f32_41;
                __nv_bfloat162 prefix_pair_1405 = __as_bf16x2(smem_regs_1393[3]);
                __nv_bfloat162 delta_pair_1406 = __as_bf16x2(smem_regs_1395[3]);
                __nv_bfloat162 updated_bf16_pair_1407 = prefix_pair_1405 + delta_pair_1406;
                float _cvt_f32_42 = (float)updated_bf16_pair_1407.x;
                values_1394[6] = _cvt_f32_42;
                float _cvt_f32_43 = (float)updated_bf16_pair_1407.y;
                values_1394[7] = _cvt_f32_43;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_1394[0 + 0], values_1394[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1394[0 + 2], values_1394[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_1394[0 + 4], values_1394[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_1394[0 + 6], values_1394[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1392 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 40, values_1394);
                float2 _f2_976 = make_float2(values_1394[0], values_1394[1]);
                float2 value_pair_1408 = _f2_976;
                float2 _f2_977 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_1409 = _f2_977;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1408, value_pair_1408, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1408, q_pair_1409, local_dot_1284[2]);
                float2 _f2_978 = make_float2(values_1394[2], values_1394[3]);
                float2 value_pair_1410 = _f2_978;
                float2 _f2_979 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_1411 = _f2_979;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1410, value_pair_1410, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1410, q_pair_1411, local_dot_1284[2]);
                float2 _f2_980 = make_float2(values_1394[4], values_1394[5]);
                float2 value_pair_1412 = _f2_980;
                float2 _f2_981 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_1413 = _f2_981;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1412, value_pair_1412, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1412, q_pair_1413, local_dot_1284[2]);
                float2 _f2_982 = make_float2(values_1394[6], values_1394[7]);
                float2 value_pair_1414 = _f2_982;
                float2 _f2_983 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_1415 = _f2_983;
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
                float _cvt_f32_44 = (float)updated_bf16_pair_1422.x;
                values_1418[0] = _cvt_f32_44;
                float _cvt_f32_45 = (float)updated_bf16_pair_1422.y;
                values_1418[1] = _cvt_f32_45;
                __nv_bfloat162 prefix_pair_1423 = __as_bf16x2(smem_regs_1417[1]);
                __nv_bfloat162 delta_pair_1424 = __as_bf16x2(smem_regs_1419[1]);
                __nv_bfloat162 updated_bf16_pair_1425 = prefix_pair_1423 + delta_pair_1424;
                float _cvt_f32_46 = (float)updated_bf16_pair_1425.x;
                values_1418[2] = _cvt_f32_46;
                float _cvt_f32_47 = (float)updated_bf16_pair_1425.y;
                values_1418[3] = _cvt_f32_47;
                __nv_bfloat162 prefix_pair_1426 = __as_bf16x2(smem_regs_1417[2]);
                __nv_bfloat162 delta_pair_1427 = __as_bf16x2(smem_regs_1419[2]);
                __nv_bfloat162 updated_bf16_pair_1428 = prefix_pair_1426 + delta_pair_1427;
                float _cvt_f32_48 = (float)updated_bf16_pair_1428.x;
                values_1418[4] = _cvt_f32_48;
                float _cvt_f32_49 = (float)updated_bf16_pair_1428.y;
                values_1418[5] = _cvt_f32_49;
                __nv_bfloat162 prefix_pair_1429 = __as_bf16x2(smem_regs_1417[3]);
                __nv_bfloat162 delta_pair_1430 = __as_bf16x2(smem_regs_1419[3]);
                __nv_bfloat162 updated_bf16_pair_1431 = prefix_pair_1429 + delta_pair_1430;
                float _cvt_f32_50 = (float)updated_bf16_pair_1431.x;
                values_1418[6] = _cvt_f32_50;
                float _cvt_f32_51 = (float)updated_bf16_pair_1431.y;
                values_1418[7] = _cvt_f32_51;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_1418[0 + 0], values_1418[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1418[0 + 2], values_1418[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_1418[0 + 4], values_1418[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_1418[0 + 6], values_1418[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1416 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 48, values_1418);
                float2 _f2_984 = make_float2(values_1418[0], values_1418[1]);
                float2 value_pair_1432 = _f2_984;
                float2 _f2_985 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_1433 = _f2_985;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1432, value_pair_1432, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1432, q_pair_1433, local_dot_1284[2]);
                float2 _f2_986 = make_float2(values_1418[2], values_1418[3]);
                float2 value_pair_1434 = _f2_986;
                float2 _f2_987 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_1435 = _f2_987;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1434, value_pair_1434, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1434, q_pair_1435, local_dot_1284[2]);
                float2 _f2_988 = make_float2(values_1418[4], values_1418[5]);
                float2 value_pair_1436 = _f2_988;
                float2 _f2_989 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_1437 = _f2_989;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1436, value_pair_1436, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1436, q_pair_1437, local_dot_1284[2]);
                float2 _f2_990 = make_float2(values_1418[6], values_1418[7]);
                float2 value_pair_1438 = _f2_990;
                float2 _f2_991 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_1439 = _f2_991;
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
                float _cvt_f32_52 = (float)updated_bf16_pair_1446.x;
                values_1442[0] = _cvt_f32_52;
                float _cvt_f32_53 = (float)updated_bf16_pair_1446.y;
                values_1442[1] = _cvt_f32_53;
                __nv_bfloat162 prefix_pair_1447 = __as_bf16x2(smem_regs_1441[1]);
                __nv_bfloat162 delta_pair_1448 = __as_bf16x2(smem_regs_1443[1]);
                __nv_bfloat162 updated_bf16_pair_1449 = prefix_pair_1447 + delta_pair_1448;
                float _cvt_f32_54 = (float)updated_bf16_pair_1449.x;
                values_1442[2] = _cvt_f32_54;
                float _cvt_f32_55 = (float)updated_bf16_pair_1449.y;
                values_1442[3] = _cvt_f32_55;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(values_1442[0 + 0], values_1442[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1442[0 + 2], values_1442[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1440]) = _pk2;
                }
                tmem_st_x4_f32(tmem_base + 56, values_1442);
                float2 _f2_992 = make_float2(values_1442[0], values_1442[1]);
                float2 value_pair_1450 = _f2_992;
                float2 _f2_993 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_1451 = _f2_993;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1450, value_pair_1450, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1450, q_pair_1451, local_dot_1284[2]);
                float2 _f2_994 = make_float2(values_1442[2], values_1442[3]);
                float2 value_pair_1452 = _f2_994;
                float2 _f2_995 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_1453 = _f2_995;
                local_sum_sq_1283[2] = fma_f32x2_rn_ftz(value_pair_1452, value_pair_1452, local_sum_sq_1283[2]);
                local_dot_1284[2] = fma_f32x2_rn_ftz(value_pair_1452, q_pair_1453, local_dot_1284[2]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_1282) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1454 = stats_stream_base + 96;
                float2 _f2_996 = make_float2(local_sum_sq_1283[0].x + local_sum_sq_1283[0].y, local_dot_1284[0].x + local_dot_1284[0].y);
                float2 reduce_pair_1455 = _f2_996;
                unsigned long long packed_pair_1456 = 0;
                unsigned long long peer_packed_1457 = 0;
                float2 _f2_997 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1458 = _f2_997;
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 16);
                peer_packed_1457 = _shfl_xor_60;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 8);
                peer_packed_1457 = _shfl_xor_61;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 4);
                peer_packed_1457 = _shfl_xor_62;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 2);
                peer_packed_1457 = _shfl_xor_63;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                packed_pair_1456 = reinterpret_cast<unsigned long long*>(&reduce_pair_1455)[0];
                unsigned long long _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1456, 1);
                peer_packed_1457 = _shfl_xor_64;
                peer_pair_1458 = reinterpret_cast<float2*>(&peer_packed_1457)[0];
                reduce_pair_1455 = add_f32x2(reduce_pair_1455, peer_pair_1458);
                if (lane == 0) {
                    stats[slot_base_1454 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1455.x;
                    stats[slot_base_1454 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1455.y;
                }
                float2 _f2_998 = make_float2(local_sum_sq_1283[2].x + local_sum_sq_1283[2].y, local_dot_1284[2].x + local_dot_1284[2].y);
                float2 reduce_pair_1459 = _f2_998;
                unsigned long long packed_pair_1460 = 0;
                unsigned long long peer_packed_1461 = 0;
                float2 _f2_999 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1462 = _f2_999;
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 16);
                peer_packed_1461 = _shfl_xor_65;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 8);
                peer_packed_1461 = _shfl_xor_66;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 4);
                peer_packed_1461 = _shfl_xor_67;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 2);
                peer_packed_1461 = _shfl_xor_68;
                peer_pair_1462 = reinterpret_cast<float2*>(&peer_packed_1461)[0];
                reduce_pair_1459 = add_f32x2(reduce_pair_1459, peer_pair_1462);
                packed_pair_1460 = reinterpret_cast<unsigned long long*>(&reduce_pair_1459)[0];
                unsigned long long _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1460, 1);
                peer_packed_1461 = _shfl_xor_69;
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
                float _shfl_down_18 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1465, 4, 8);
                total_sum_sq_1465 += _shfl_down_18;
                float _shfl_down_19 = __shfl_down_sync(0xFFFFFFFF, total_dot_1466, 4, 8);
                total_dot_1466 += _shfl_down_19;
                float _shfl_down_20 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1465, 2, 8);
                total_sum_sq_1465 += _shfl_down_20;
                float _shfl_down_21 = __shfl_down_sync(0xFFFFFFFF, total_dot_1466, 2, 8);
                total_dot_1466 += _shfl_down_21;
                float _shfl_down_22 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1465, 1, 8);
                total_sum_sq_1465 += _shfl_down_22;
                float _shfl_down_23 = __shfl_down_sync(0xFFFFFFFF, total_dot_1466, 1, 8);
                total_dot_1466 += _shfl_down_23;
                float local_logit_1467 = 0.0f;
                if (stat_source_1463 < 1 && stat_warp_1464 == 0) {
                    float _rsqrt_3 = rsqrtf(total_sum_sq_1465 / 7168.0f + eps);
                    local_logit_1467 = total_dot_1466 * _rsqrt_3;
                }
                float logits_1468[2];
                logits_1468[0] = 0.0f;
                float _shfl_6 = __shfl_sync(0xFFFFFFFF, local_logit_1467, 0);
                logits_1468[0] = _shfl_6;
                logits_1468[1] = 0.0f;
                float _shfl_7 = __shfl_sync(0xFFFFFFFF, local_logit_1467, 8);
                logits_1468[1] = _shfl_7;
                float weights_1469[2];
                float new_max_1470 = 0.0f;
                float correction_1471 = 0.0f;
                float weight_sum_1472 = 0.0f;
                weights_1469[0] = 0.0f;
                weights_1469[1] = 0.0f;
                float chunk_max_1473 = -3.4028234663852886e+38f;
                float _max_9 = max_noftz(chunk_max_1473, logits_1468[0]);
                chunk_max_1473 = _max_9;
                float _max_10 = max_noftz(m_running, chunk_max_1473);
                new_max_1470 = _max_10;
                float _exp2_9 = approx_exp2((m_running - new_max_1470) * 1.4426950408889634f);
                correction_1471 = _exp2_9;
                float _exp2_10 = approx_exp2((logits_1468[0] - new_max_1470) * 1.4426950408889634f);
                weights_1469[0] = _exp2_10;
                weight_sum_1472 += weights_1469[0];
                float2 _f2_1000 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1474 = _f2_1000;
                float2 _f2_1001 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1475 = _f2_1001;
                float2 scaled_pair_1476;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1476) : "l"(*(const unsigned long long*)&acc_pair_1475), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[0] = scaled_pair_1476.x;
                acc32[1] = scaled_pair_1476.y;
                float2 _f2_1002 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1477 = _f2_1002;
                float2 scaled_pair_1478;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1478) : "l"(*(const unsigned long long*)&acc_pair_1477), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[2] = scaled_pair_1478.x;
                acc32[3] = scaled_pair_1478.y;
                float2 _f2_1003 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1479 = _f2_1003;
                float2 scaled_pair_1480;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1480) : "l"(*(const unsigned long long*)&acc_pair_1479), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[4] = scaled_pair_1480.x;
                acc32[5] = scaled_pair_1480.y;
                float2 _f2_1004 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1481 = _f2_1004;
                float2 scaled_pair_1482;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1482) : "l"(*(const unsigned long long*)&acc_pair_1481), "l"(*(const unsigned long long*)&correction_pair_1474));
                acc32[6] = scaled_pair_1482.x;
                acc32[7] = scaled_pair_1482.y;
                float _tmem_load_48[8];
                tmem_ld_x8(&_tmem_load_48[0], tmem_base);
                float2 _f2_1005 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1483 = _f2_1005;
                float2 _f2_1006 = make_float2(_tmem_load_48[0], _tmem_load_48[1]);
                float2 cached_pair_1484 = _f2_1006;
                float2 _f2_1007 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1485 = _f2_1007;
                float2 updated_pair_1486 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1484, acc_pair_1485);
                acc32[0] = updated_pair_1486.x;
                acc32[1] = updated_pair_1486.y;
                float2 _f2_1008 = make_float2(_tmem_load_48[2], _tmem_load_48[3]);
                float2 cached_pair_1487 = _f2_1008;
                float2 _f2_1009 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1488 = _f2_1009;
                float2 updated_pair_1489 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1487, acc_pair_1488);
                acc32[2] = updated_pair_1489.x;
                acc32[3] = updated_pair_1489.y;
                float2 _f2_1010 = make_float2(_tmem_load_48[4], _tmem_load_48[5]);
                float2 cached_pair_1490 = _f2_1010;
                float2 _f2_1011 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1491 = _f2_1011;
                float2 updated_pair_1492 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1490, acc_pair_1491);
                acc32[4] = updated_pair_1492.x;
                acc32[5] = updated_pair_1492.y;
                float2 _f2_1012 = make_float2(_tmem_load_48[6], _tmem_load_48[7]);
                float2 cached_pair_1493 = _f2_1012;
                float2 _f2_1013 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1494 = _f2_1013;
                float2 updated_pair_1495 = fma_f32x2_rn_ftz(weight_pair_1483, cached_pair_1493, acc_pair_1494);
                acc32[6] = updated_pair_1495.x;
                acc32[7] = updated_pair_1495.y;
                float2 _f2_1014 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1496 = _f2_1014;
                float2 _f2_1015 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1497 = _f2_1015;
                float2 scaled_pair_1498;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1498) : "l"(*(const unsigned long long*)&acc_pair_1497), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[8] = scaled_pair_1498.x;
                acc32[9] = scaled_pair_1498.y;
                float2 _f2_1016 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1499 = _f2_1016;
                float2 scaled_pair_1500;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1500) : "l"(*(const unsigned long long*)&acc_pair_1499), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[10] = scaled_pair_1500.x;
                acc32[11] = scaled_pair_1500.y;
                float2 _f2_1017 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1501 = _f2_1017;
                float2 scaled_pair_1502;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1502) : "l"(*(const unsigned long long*)&acc_pair_1501), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[12] = scaled_pair_1502.x;
                acc32[13] = scaled_pair_1502.y;
                float2 _f2_1018 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1503 = _f2_1018;
                float2 scaled_pair_1504;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1504) : "l"(*(const unsigned long long*)&acc_pair_1503), "l"(*(const unsigned long long*)&correction_pair_1496));
                acc32[14] = scaled_pair_1504.x;
                acc32[15] = scaled_pair_1504.y;
                float _tmem_load_49[8];
                tmem_ld_x8(&_tmem_load_49[0], tmem_base + 8);
                float2 _f2_1019 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1505 = _f2_1019;
                float2 _f2_1020 = make_float2(_tmem_load_49[0], _tmem_load_49[1]);
                float2 cached_pair_1506 = _f2_1020;
                float2 _f2_1021 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1507 = _f2_1021;
                float2 updated_pair_1508 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1506, acc_pair_1507);
                acc32[8] = updated_pair_1508.x;
                acc32[9] = updated_pair_1508.y;
                float2 _f2_1022 = make_float2(_tmem_load_49[2], _tmem_load_49[3]);
                float2 cached_pair_1509 = _f2_1022;
                float2 _f2_1023 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1510 = _f2_1023;
                float2 updated_pair_1511 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1509, acc_pair_1510);
                acc32[10] = updated_pair_1511.x;
                acc32[11] = updated_pair_1511.y;
                float2 _f2_1024 = make_float2(_tmem_load_49[4], _tmem_load_49[5]);
                float2 cached_pair_1512 = _f2_1024;
                float2 _f2_1025 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1513 = _f2_1025;
                float2 updated_pair_1514 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1512, acc_pair_1513);
                acc32[12] = updated_pair_1514.x;
                acc32[13] = updated_pair_1514.y;
                float2 _f2_1026 = make_float2(_tmem_load_49[6], _tmem_load_49[7]);
                float2 cached_pair_1515 = _f2_1026;
                float2 _f2_1027 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1516 = _f2_1027;
                float2 updated_pair_1517 = fma_f32x2_rn_ftz(weight_pair_1505, cached_pair_1515, acc_pair_1516);
                acc32[14] = updated_pair_1517.x;
                acc32[15] = updated_pair_1517.y;
                float2 _f2_1028 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1518 = _f2_1028;
                float2 _f2_1029 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1519 = _f2_1029;
                float2 scaled_pair_1520;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1520) : "l"(*(const unsigned long long*)&acc_pair_1519), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[16] = scaled_pair_1520.x;
                acc32[17] = scaled_pair_1520.y;
                float2 _f2_1030 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1521 = _f2_1030;
                float2 scaled_pair_1522;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1522) : "l"(*(const unsigned long long*)&acc_pair_1521), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[18] = scaled_pair_1522.x;
                acc32[19] = scaled_pair_1522.y;
                float2 _f2_1031 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1523 = _f2_1031;
                float2 scaled_pair_1524;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1524) : "l"(*(const unsigned long long*)&acc_pair_1523), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[20] = scaled_pair_1524.x;
                acc32[21] = scaled_pair_1524.y;
                float2 _f2_1032 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1525 = _f2_1032;
                float2 scaled_pair_1526;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1526) : "l"(*(const unsigned long long*)&acc_pair_1525), "l"(*(const unsigned long long*)&correction_pair_1518));
                acc32[22] = scaled_pair_1526.x;
                acc32[23] = scaled_pair_1526.y;
                float _tmem_load_50[8];
                tmem_ld_x8(&_tmem_load_50[0], tmem_base + 16);
                float2 _f2_1033 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1527 = _f2_1033;
                float2 _f2_1034 = make_float2(_tmem_load_50[0], _tmem_load_50[1]);
                float2 cached_pair_1528 = _f2_1034;
                float2 _f2_1035 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1529 = _f2_1035;
                float2 updated_pair_1530 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1528, acc_pair_1529);
                acc32[16] = updated_pair_1530.x;
                acc32[17] = updated_pair_1530.y;
                float2 _f2_1036 = make_float2(_tmem_load_50[2], _tmem_load_50[3]);
                float2 cached_pair_1531 = _f2_1036;
                float2 _f2_1037 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1532 = _f2_1037;
                float2 updated_pair_1533 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1531, acc_pair_1532);
                acc32[18] = updated_pair_1533.x;
                acc32[19] = updated_pair_1533.y;
                float2 _f2_1038 = make_float2(_tmem_load_50[4], _tmem_load_50[5]);
                float2 cached_pair_1534 = _f2_1038;
                float2 _f2_1039 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1535 = _f2_1039;
                float2 updated_pair_1536 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1534, acc_pair_1535);
                acc32[20] = updated_pair_1536.x;
                acc32[21] = updated_pair_1536.y;
                float2 _f2_1040 = make_float2(_tmem_load_50[6], _tmem_load_50[7]);
                float2 cached_pair_1537 = _f2_1040;
                float2 _f2_1041 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1538 = _f2_1041;
                float2 updated_pair_1539 = fma_f32x2_rn_ftz(weight_pair_1527, cached_pair_1537, acc_pair_1538);
                acc32[22] = updated_pair_1539.x;
                acc32[23] = updated_pair_1539.y;
                float2 _f2_1042 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1540 = _f2_1042;
                float2 _f2_1043 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1541 = _f2_1043;
                float2 scaled_pair_1542;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1542) : "l"(*(const unsigned long long*)&acc_pair_1541), "l"(*(const unsigned long long*)&correction_pair_1540));
                acc32[24] = scaled_pair_1542.x;
                acc32[25] = scaled_pair_1542.y;
                float2 _f2_1044 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1543 = _f2_1044;
                float2 scaled_pair_1544;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1544) : "l"(*(const unsigned long long*)&acc_pair_1543), "l"(*(const unsigned long long*)&correction_pair_1540));
                acc32[26] = scaled_pair_1544.x;
                acc32[27] = scaled_pair_1544.y;
                float _tmem_load_51[4];
                tmem_ld_x4(&_tmem_load_51[0], tmem_base + 24);
                float2 _f2_1045 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1545 = _f2_1045;
                float2 _f2_1046 = make_float2(_tmem_load_51[0], _tmem_load_51[1]);
                float2 cached_pair_1546 = _f2_1046;
                float2 _f2_1047 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1547 = _f2_1047;
                float2 updated_pair_1548 = fma_f32x2_rn_ftz(weight_pair_1545, cached_pair_1546, acc_pair_1547);
                acc32[24] = updated_pair_1548.x;
                acc32[25] = updated_pair_1548.y;
                float2 _f2_1048 = make_float2(_tmem_load_51[2], _tmem_load_51[3]);
                float2 cached_pair_1549 = _f2_1048;
                float2 _f2_1049 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1550 = _f2_1049;
                float2 updated_pair_1551 = fma_f32x2_rn_ftz(weight_pair_1545, cached_pair_1549, acc_pair_1550);
                acc32[26] = updated_pair_1551.x;
                acc32[27] = updated_pair_1551.y;
                float2 _f2_1050 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1552 = _f2_1050;
                float2 _f2_1051 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1553 = _f2_1051;
                float2 scaled_pair_1554;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1554) : "l"(*(const unsigned long long*)&acc_pair_1553), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[28] = scaled_pair_1554.x;
                acc32[29] = scaled_pair_1554.y;
                float2 _f2_1052 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1555 = _f2_1052;
                float2 scaled_pair_1556;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1556) : "l"(*(const unsigned long long*)&acc_pair_1555), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[30] = scaled_pair_1556.x;
                acc32[31] = scaled_pair_1556.y;
                float2 _f2_1053 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1557 = _f2_1053;
                float2 scaled_pair_1558;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1558) : "l"(*(const unsigned long long*)&acc_pair_1557), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[32] = scaled_pair_1558.x;
                acc32[33] = scaled_pair_1558.y;
                float2 _f2_1054 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1559 = _f2_1054;
                float2 scaled_pair_1560;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1560) : "l"(*(const unsigned long long*)&acc_pair_1559), "l"(*(const unsigned long long*)&correction_pair_1552));
                acc32[34] = scaled_pair_1560.x;
                acc32[35] = scaled_pair_1560.y;
                float _tmem_load_52[8];
                tmem_ld_x8(&_tmem_load_52[0], tmem_base + 32);
                float2 _f2_1055 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1561 = _f2_1055;
                float2 _f2_1056 = make_float2(_tmem_load_52[0], _tmem_load_52[1]);
                float2 cached_pair_1562 = _f2_1056;
                float2 _f2_1057 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1563 = _f2_1057;
                float2 updated_pair_1564 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1562, acc_pair_1563);
                acc32[28] = updated_pair_1564.x;
                acc32[29] = updated_pair_1564.y;
                float2 _f2_1058 = make_float2(_tmem_load_52[2], _tmem_load_52[3]);
                float2 cached_pair_1565 = _f2_1058;
                float2 _f2_1059 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1566 = _f2_1059;
                float2 updated_pair_1567 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1565, acc_pair_1566);
                acc32[30] = updated_pair_1567.x;
                acc32[31] = updated_pair_1567.y;
                float2 _f2_1060 = make_float2(_tmem_load_52[4], _tmem_load_52[5]);
                float2 cached_pair_1568 = _f2_1060;
                float2 _f2_1061 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1569 = _f2_1061;
                float2 updated_pair_1570 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1568, acc_pair_1569);
                acc32[32] = updated_pair_1570.x;
                acc32[33] = updated_pair_1570.y;
                float2 _f2_1062 = make_float2(_tmem_load_52[6], _tmem_load_52[7]);
                float2 cached_pair_1571 = _f2_1062;
                float2 _f2_1063 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1572 = _f2_1063;
                float2 updated_pair_1573 = fma_f32x2_rn_ftz(weight_pair_1561, cached_pair_1571, acc_pair_1572);
                acc32[34] = updated_pair_1573.x;
                acc32[35] = updated_pair_1573.y;
                float2 _f2_1064 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1574 = _f2_1064;
                float2 _f2_1065 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1575 = _f2_1065;
                float2 scaled_pair_1576;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1576) : "l"(*(const unsigned long long*)&acc_pair_1575), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[36] = scaled_pair_1576.x;
                acc32[37] = scaled_pair_1576.y;
                float2 _f2_1066 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1577 = _f2_1066;
                float2 scaled_pair_1578;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1578) : "l"(*(const unsigned long long*)&acc_pair_1577), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[38] = scaled_pair_1578.x;
                acc32[39] = scaled_pair_1578.y;
                float2 _f2_1067 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1579 = _f2_1067;
                float2 scaled_pair_1580;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1580) : "l"(*(const unsigned long long*)&acc_pair_1579), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[40] = scaled_pair_1580.x;
                acc32[41] = scaled_pair_1580.y;
                float2 _f2_1068 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1581 = _f2_1068;
                float2 scaled_pair_1582;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1582) : "l"(*(const unsigned long long*)&acc_pair_1581), "l"(*(const unsigned long long*)&correction_pair_1574));
                acc32[42] = scaled_pair_1582.x;
                acc32[43] = scaled_pair_1582.y;
                float _tmem_load_53[8];
                tmem_ld_x8(&_tmem_load_53[0], tmem_base + 40);
                float2 _f2_1069 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1583 = _f2_1069;
                float2 _f2_1070 = make_float2(_tmem_load_53[0], _tmem_load_53[1]);
                float2 cached_pair_1584 = _f2_1070;
                float2 _f2_1071 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1585 = _f2_1071;
                float2 updated_pair_1586 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1584, acc_pair_1585);
                acc32[36] = updated_pair_1586.x;
                acc32[37] = updated_pair_1586.y;
                float2 _f2_1072 = make_float2(_tmem_load_53[2], _tmem_load_53[3]);
                float2 cached_pair_1587 = _f2_1072;
                float2 _f2_1073 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1588 = _f2_1073;
                float2 updated_pair_1589 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1587, acc_pair_1588);
                acc32[38] = updated_pair_1589.x;
                acc32[39] = updated_pair_1589.y;
                float2 _f2_1074 = make_float2(_tmem_load_53[4], _tmem_load_53[5]);
                float2 cached_pair_1590 = _f2_1074;
                float2 _f2_1075 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1591 = _f2_1075;
                float2 updated_pair_1592 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1590, acc_pair_1591);
                acc32[40] = updated_pair_1592.x;
                acc32[41] = updated_pair_1592.y;
                float2 _f2_1076 = make_float2(_tmem_load_53[6], _tmem_load_53[7]);
                float2 cached_pair_1593 = _f2_1076;
                float2 _f2_1077 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1594 = _f2_1077;
                float2 updated_pair_1595 = fma_f32x2_rn_ftz(weight_pair_1583, cached_pair_1593, acc_pair_1594);
                acc32[42] = updated_pair_1595.x;
                acc32[43] = updated_pair_1595.y;
                float2 _f2_1078 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1596 = _f2_1078;
                float2 _f2_1079 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1597 = _f2_1079;
                float2 scaled_pair_1598;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1598) : "l"(*(const unsigned long long*)&acc_pair_1597), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[44] = scaled_pair_1598.x;
                acc32[45] = scaled_pair_1598.y;
                float2 _f2_1080 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1599 = _f2_1080;
                float2 scaled_pair_1600;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1600) : "l"(*(const unsigned long long*)&acc_pair_1599), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[46] = scaled_pair_1600.x;
                acc32[47] = scaled_pair_1600.y;
                float2 _f2_1081 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1601 = _f2_1081;
                float2 scaled_pair_1602;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1602) : "l"(*(const unsigned long long*)&acc_pair_1601), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[48] = scaled_pair_1602.x;
                acc32[49] = scaled_pair_1602.y;
                float2 _f2_1082 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1603 = _f2_1082;
                float2 scaled_pair_1604;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1604) : "l"(*(const unsigned long long*)&acc_pair_1603), "l"(*(const unsigned long long*)&correction_pair_1596));
                acc32[50] = scaled_pair_1604.x;
                acc32[51] = scaled_pair_1604.y;
                float _tmem_load_54[8];
                tmem_ld_x8(&_tmem_load_54[0], tmem_base + 48);
                float2 _f2_1083 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1605 = _f2_1083;
                float2 _f2_1084 = make_float2(_tmem_load_54[0], _tmem_load_54[1]);
                float2 cached_pair_1606 = _f2_1084;
                float2 _f2_1085 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1607 = _f2_1085;
                float2 updated_pair_1608 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1606, acc_pair_1607);
                acc32[44] = updated_pair_1608.x;
                acc32[45] = updated_pair_1608.y;
                float2 _f2_1086 = make_float2(_tmem_load_54[2], _tmem_load_54[3]);
                float2 cached_pair_1609 = _f2_1086;
                float2 _f2_1087 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1610 = _f2_1087;
                float2 updated_pair_1611 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1609, acc_pair_1610);
                acc32[46] = updated_pair_1611.x;
                acc32[47] = updated_pair_1611.y;
                float2 _f2_1088 = make_float2(_tmem_load_54[4], _tmem_load_54[5]);
                float2 cached_pair_1612 = _f2_1088;
                float2 _f2_1089 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1613 = _f2_1089;
                float2 updated_pair_1614 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1612, acc_pair_1613);
                acc32[48] = updated_pair_1614.x;
                acc32[49] = updated_pair_1614.y;
                float2 _f2_1090 = make_float2(_tmem_load_54[6], _tmem_load_54[7]);
                float2 cached_pair_1615 = _f2_1090;
                float2 _f2_1091 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1616 = _f2_1091;
                float2 updated_pair_1617 = fma_f32x2_rn_ftz(weight_pair_1605, cached_pair_1615, acc_pair_1616);
                acc32[50] = updated_pair_1617.x;
                acc32[51] = updated_pair_1617.y;
                float2 _f2_1092 = make_float2(correction_1471, correction_1471);
                float2 correction_pair_1618 = _f2_1092;
                float2 _f2_1093 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1619 = _f2_1093;
                float2 scaled_pair_1620;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1620) : "l"(*(const unsigned long long*)&acc_pair_1619), "l"(*(const unsigned long long*)&correction_pair_1618));
                acc32[52] = scaled_pair_1620.x;
                acc32[53] = scaled_pair_1620.y;
                float2 _f2_1094 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1621 = _f2_1094;
                float2 scaled_pair_1622;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1622) : "l"(*(const unsigned long long*)&acc_pair_1621), "l"(*(const unsigned long long*)&correction_pair_1618));
                acc32[54] = scaled_pair_1622.x;
                acc32[55] = scaled_pair_1622.y;
                float _tmem_load_55[4];
                tmem_ld_x4(&_tmem_load_55[0], tmem_base + 56);
                float2 _f2_1095 = make_float2(weights_1469[0], weights_1469[0]);
                float2 weight_pair_1623 = _f2_1095;
                float2 _f2_1096 = make_float2(_tmem_load_55[0], _tmem_load_55[1]);
                float2 cached_pair_1624 = _f2_1096;
                float2 _f2_1097 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1625 = _f2_1097;
                float2 updated_pair_1626 = fma_f32x2_rn_ftz(weight_pair_1623, cached_pair_1624, acc_pair_1625);
                acc32[52] = updated_pair_1626.x;
                acc32[53] = updated_pair_1626.y;
                float2 _f2_1098 = make_float2(_tmem_load_55[2], _tmem_load_55[3]);
                float2 cached_pair_1627 = _f2_1098;
                float2 _f2_1099 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1628 = _f2_1099;
                float2 updated_pair_1629 = fma_f32x2_rn_ftz(weight_pair_1623, cached_pair_1627, acc_pair_1628);
                acc32[54] = updated_pair_1629.x;
                acc32[55] = updated_pair_1629.y;
                s_running = __fmaf_rn(s_running, correction_1471, weight_sum_1472);
                m_running = new_max_1470;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; consumer_phase ^= 1; }
                if (token_1 == first_token) {
                    mbarrier_wait(norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float2 _f2_1100 = make_float2(0.0f, 0.0f);
                float2 local_output_sum_pair = _f2_1100;
                float2 _f2_1101 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1630 = _f2_1101;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1630, acc_pair_1630, local_output_sum_pair);
                float2 _f2_1102 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1631 = _f2_1102;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1631, acc_pair_1631, local_output_sum_pair);
                float2 _f2_1103 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1632 = _f2_1103;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1632, acc_pair_1632, local_output_sum_pair);
                float2 _f2_1104 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1633 = _f2_1104;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1633, acc_pair_1633, local_output_sum_pair);
                float2 _f2_1105 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1634 = _f2_1105;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1634, acc_pair_1634, local_output_sum_pair);
                float2 _f2_1106 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1635 = _f2_1106;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1635, acc_pair_1635, local_output_sum_pair);
                float2 _f2_1107 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1636 = _f2_1107;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1636, acc_pair_1636, local_output_sum_pair);
                float2 _f2_1108 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1637 = _f2_1108;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1637, acc_pair_1637, local_output_sum_pair);
                float2 _f2_1109 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1638 = _f2_1109;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1638, acc_pair_1638, local_output_sum_pair);
                float2 _f2_1110 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1639 = _f2_1110;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1639, acc_pair_1639, local_output_sum_pair);
                float2 _f2_1111 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1640 = _f2_1111;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1640, acc_pair_1640, local_output_sum_pair);
                float2 _f2_1112 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1641 = _f2_1112;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1641, acc_pair_1641, local_output_sum_pair);
                float2 _f2_1113 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1642 = _f2_1113;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1642, acc_pair_1642, local_output_sum_pair);
                float2 _f2_1114 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1643 = _f2_1114;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1643, acc_pair_1643, local_output_sum_pair);
                float local_output_sum_sq = local_output_sum_pair.x + local_output_sum_pair.y;
                float _warp_reduce_0 = local_output_sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                local_output_sum_sq = _warp_reduce_0;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)warp_in_group] = local_output_sum_sq;
                }
                float2 _f2_1115 = make_float2(0.0f, 0.0f);
                float2 local_output_sum_pair_1644 = _f2_1115;
                float2 _f2_1116 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1645 = _f2_1116;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1645, acc_pair_1645, local_output_sum_pair_1644);
                float2 _f2_1117 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1646 = _f2_1117;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1646, acc_pair_1646, local_output_sum_pair_1644);
                float2 _f2_1118 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1647 = _f2_1118;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1647, acc_pair_1647, local_output_sum_pair_1644);
                float2 _f2_1119 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1648 = _f2_1119;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1648, acc_pair_1648, local_output_sum_pair_1644);
                float2 _f2_1120 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1649 = _f2_1120;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1649, acc_pair_1649, local_output_sum_pair_1644);
                float2 _f2_1121 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1650 = _f2_1121;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1650, acc_pair_1650, local_output_sum_pair_1644);
                float2 _f2_1122 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1651 = _f2_1122;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1651, acc_pair_1651, local_output_sum_pair_1644);
                float2 _f2_1123 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1652 = _f2_1123;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1652, acc_pair_1652, local_output_sum_pair_1644);
                float2 _f2_1124 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1653 = _f2_1124;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1653, acc_pair_1653, local_output_sum_pair_1644);
                float2 _f2_1125 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1654 = _f2_1125;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1654, acc_pair_1654, local_output_sum_pair_1644);
                float2 _f2_1126 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1655 = _f2_1126;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1655, acc_pair_1655, local_output_sum_pair_1644);
                float2 _f2_1127 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1656 = _f2_1127;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1656, acc_pair_1656, local_output_sum_pair_1644);
                float2 _f2_1128 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1657 = _f2_1128;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1657, acc_pair_1657, local_output_sum_pair_1644);
                float2 _f2_1129 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1658 = _f2_1129;
                local_output_sum_pair_1644 = fma_f32x2_rn_ftz(acc_pair_1658, acc_pair_1658, local_output_sum_pair_1644);
                float local_output_sum_sq_1659 = local_output_sum_pair_1644.x + local_output_sum_pair_1644.y;
                float _warp_reduce_1 = local_output_sum_sq_1659;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                local_output_sum_sq_1659 = _warp_reduce_1;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)(4 + warp_in_group)] = local_output_sum_sq_1659;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                float total_output_sum_sq = ((lane < 8) ? output_stats[output_stats_base + (unsigned int)lane] : 0.0f);
                float _shfl_down_24 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 4, 8);
                total_output_sum_sq += _shfl_down_24;
                float _shfl_down_25 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 2, 8);
                total_output_sum_sq += _shfl_down_25;
                float _shfl_down_26 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 1, 8);
                total_output_sum_sq += _shfl_down_26;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_4 = rsqrtf(__fmaf_rn(output_norm_eps * s_running, s_running, total_output_sum_sq / 7168.0f));
                    output_rsigma = _rsqrt_4;
                }
                float _shfl_8 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_8;
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
                float2 _f2_1130 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair = _f2_1130;
                float2 _f2_1131 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1661;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1661) : "l"(*(const unsigned long long*)&_f2_1131), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_1132 = make_float2(_unpack_bf16_48[0], _unpack_bf16_48[1]);
                float2 normalized_pair;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair_1661), "l"(*(const unsigned long long*)&_f2_1132));
                acc32[0] = normalized_pair.x;
                acc32[1] = normalized_pair.y;
                float2 _f2_1133 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1662;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1662) : "l"(*(const unsigned long long*)&_f2_1133), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_1134 = make_float2(_unpack_bf16_48[2], _unpack_bf16_48[3]);
                float2 normalized_pair_1663;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1663) : "l"(*(const unsigned long long*)&scaled_pair_1662), "l"(*(const unsigned long long*)&_f2_1134));
                acc32[2] = normalized_pair_1663.x;
                acc32[3] = normalized_pair_1663.y;
                float2 _f2_1135 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1664;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1664) : "l"(*(const unsigned long long*)&_f2_1135), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_1136 = make_float2(_unpack_bf16_48[4], _unpack_bf16_48[5]);
                float2 normalized_pair_1665;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1665) : "l"(*(const unsigned long long*)&scaled_pair_1664), "l"(*(const unsigned long long*)&_f2_1136));
                acc32[4] = normalized_pair_1665.x;
                acc32[5] = normalized_pair_1665.y;
                float2 _f2_1137 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1666;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1666) : "l"(*(const unsigned long long*)&_f2_1137), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_1138 = make_float2(_unpack_bf16_48[6], _unpack_bf16_48[7]);
                float2 normalized_pair_1667;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1667) : "l"(*(const unsigned long long*)&scaled_pair_1666), "l"(*(const unsigned long long*)&_f2_1138));
                acc32[6] = normalized_pair_1667.x;
                acc32[7] = normalized_pair_1667.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[0 + 0], acc32[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[0 + 2], acc32[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[0 + 4], acc32[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[0 + 6], acc32[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1660 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_1139 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1670 = _f2_1139;
                float2 _f2_1140 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1671;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1671) : "l"(*(const unsigned long long*)&_f2_1140), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f2_1141 = make_float2(_unpack_bf16_49[0], _unpack_bf16_49[1]);
                float2 normalized_pair_1672;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1672) : "l"(*(const unsigned long long*)&scaled_pair_1671), "l"(*(const unsigned long long*)&_f2_1141));
                acc32[8] = normalized_pair_1672.x;
                acc32[9] = normalized_pair_1672.y;
                float2 _f2_1142 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1673;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1673) : "l"(*(const unsigned long long*)&_f2_1142), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f2_1143 = make_float2(_unpack_bf16_49[2], _unpack_bf16_49[3]);
                float2 normalized_pair_1674;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1674) : "l"(*(const unsigned long long*)&scaled_pair_1673), "l"(*(const unsigned long long*)&_f2_1143));
                acc32[10] = normalized_pair_1674.x;
                acc32[11] = normalized_pair_1674.y;
                float2 _f2_1144 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1675;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1675) : "l"(*(const unsigned long long*)&_f2_1144), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f2_1145 = make_float2(_unpack_bf16_49[4], _unpack_bf16_49[5]);
                float2 normalized_pair_1676;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1676) : "l"(*(const unsigned long long*)&scaled_pair_1675), "l"(*(const unsigned long long*)&_f2_1145));
                acc32[12] = normalized_pair_1676.x;
                acc32[13] = normalized_pair_1676.y;
                float2 _f2_1146 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1677;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1677) : "l"(*(const unsigned long long*)&_f2_1146), "l"(*(const unsigned long long*)&output_rsigma_pair_1670));
                float2 _f2_1147 = make_float2(_unpack_bf16_49[6], _unpack_bf16_49[7]);
                float2 normalized_pair_1678;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1678) : "l"(*(const unsigned long long*)&scaled_pair_1677), "l"(*(const unsigned long long*)&_f2_1147));
                acc32[14] = normalized_pair_1678.x;
                acc32[15] = normalized_pair_1678.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[8 + 0], acc32[8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[8 + 2], acc32[8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[8 + 4], acc32[8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[8 + 6], acc32[8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1668 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_1148 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1681 = _f2_1148;
                float2 _f2_1149 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1682;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1682) : "l"(*(const unsigned long long*)&_f2_1149), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f2_1150 = make_float2(_unpack_bf16_50[0], _unpack_bf16_50[1]);
                float2 normalized_pair_1683;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1683) : "l"(*(const unsigned long long*)&scaled_pair_1682), "l"(*(const unsigned long long*)&_f2_1150));
                acc32[16] = normalized_pair_1683.x;
                acc32[17] = normalized_pair_1683.y;
                float2 _f2_1151 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1684;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1684) : "l"(*(const unsigned long long*)&_f2_1151), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f2_1152 = make_float2(_unpack_bf16_50[2], _unpack_bf16_50[3]);
                float2 normalized_pair_1685;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1685) : "l"(*(const unsigned long long*)&scaled_pair_1684), "l"(*(const unsigned long long*)&_f2_1152));
                acc32[18] = normalized_pair_1685.x;
                acc32[19] = normalized_pair_1685.y;
                float2 _f2_1153 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1686;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1686) : "l"(*(const unsigned long long*)&_f2_1153), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f2_1154 = make_float2(_unpack_bf16_50[4], _unpack_bf16_50[5]);
                float2 normalized_pair_1687;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1687) : "l"(*(const unsigned long long*)&scaled_pair_1686), "l"(*(const unsigned long long*)&_f2_1154));
                acc32[20] = normalized_pair_1687.x;
                acc32[21] = normalized_pair_1687.y;
                float2 _f2_1155 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1688;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1688) : "l"(*(const unsigned long long*)&_f2_1155), "l"(*(const unsigned long long*)&output_rsigma_pair_1681));
                float2 _f2_1156 = make_float2(_unpack_bf16_50[6], _unpack_bf16_50[7]);
                float2 normalized_pair_1689;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1689) : "l"(*(const unsigned long long*)&scaled_pair_1688), "l"(*(const unsigned long long*)&_f2_1156));
                acc32[22] = normalized_pair_1689.x;
                acc32[23] = normalized_pair_1689.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[16 + 0], acc32[16 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[16 + 2], acc32[16 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[16 + 4], acc32[16 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[16 + 6], acc32[16 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1679 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_1157 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1692 = _f2_1157;
                float2 _f2_1158 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1693;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1693) : "l"(*(const unsigned long long*)&_f2_1158), "l"(*(const unsigned long long*)&output_rsigma_pair_1692));
                float2 _f2_1159 = make_float2(_unpack_bf16_51[0], _unpack_bf16_51[1]);
                float2 normalized_pair_1694;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1694) : "l"(*(const unsigned long long*)&scaled_pair_1693), "l"(*(const unsigned long long*)&_f2_1159));
                acc32[24] = normalized_pair_1694.x;
                acc32[25] = normalized_pair_1694.y;
                float2 _f2_1160 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1695;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1695) : "l"(*(const unsigned long long*)&_f2_1160), "l"(*(const unsigned long long*)&output_rsigma_pair_1692));
                float2 _f2_1161 = make_float2(_unpack_bf16_51[2], _unpack_bf16_51[3]);
                float2 normalized_pair_1696;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1696) : "l"(*(const unsigned long long*)&scaled_pair_1695), "l"(*(const unsigned long long*)&_f2_1161));
                acc32[26] = normalized_pair_1696.x;
                acc32[27] = normalized_pair_1696.y;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(acc32[24 + 0], acc32[24 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[24 + 2], acc32[24 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1690]) = _pk2;
                }
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
                float2 _f2_1162 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1699 = _f2_1162;
                float2 _f2_1163 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1700;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1700) : "l"(*(const unsigned long long*)&_f2_1163), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f2_1164 = make_float2(_unpack_bf16_52[0], _unpack_bf16_52[1]);
                float2 normalized_pair_1701;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1701) : "l"(*(const unsigned long long*)&scaled_pair_1700), "l"(*(const unsigned long long*)&_f2_1164));
                acc32[28] = normalized_pair_1701.x;
                acc32[29] = normalized_pair_1701.y;
                float2 _f2_1165 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1702;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1702) : "l"(*(const unsigned long long*)&_f2_1165), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f2_1166 = make_float2(_unpack_bf16_52[2], _unpack_bf16_52[3]);
                float2 normalized_pair_1703;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1703) : "l"(*(const unsigned long long*)&scaled_pair_1702), "l"(*(const unsigned long long*)&_f2_1166));
                acc32[30] = normalized_pair_1703.x;
                acc32[31] = normalized_pair_1703.y;
                float2 _f2_1167 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1704;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1704) : "l"(*(const unsigned long long*)&_f2_1167), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f2_1168 = make_float2(_unpack_bf16_52[4], _unpack_bf16_52[5]);
                float2 normalized_pair_1705;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1705) : "l"(*(const unsigned long long*)&scaled_pair_1704), "l"(*(const unsigned long long*)&_f2_1168));
                acc32[32] = normalized_pair_1705.x;
                acc32[33] = normalized_pair_1705.y;
                float2 _f2_1169 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1706;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1706) : "l"(*(const unsigned long long*)&_f2_1169), "l"(*(const unsigned long long*)&output_rsigma_pair_1699));
                float2 _f2_1170 = make_float2(_unpack_bf16_52[6], _unpack_bf16_52[7]);
                float2 normalized_pair_1707;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1707) : "l"(*(const unsigned long long*)&scaled_pair_1706), "l"(*(const unsigned long long*)&_f2_1170));
                acc32[34] = normalized_pair_1707.x;
                acc32[35] = normalized_pair_1707.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[28 + 0], acc32[28 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[28 + 2], acc32[28 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[28 + 4], acc32[28 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[28 + 6], acc32[28 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1697 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_1171 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1710 = _f2_1171;
                float2 _f2_1172 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1711;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1711) : "l"(*(const unsigned long long*)&_f2_1172), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f2_1173 = make_float2(_unpack_bf16_53[0], _unpack_bf16_53[1]);
                float2 normalized_pair_1712;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1712) : "l"(*(const unsigned long long*)&scaled_pair_1711), "l"(*(const unsigned long long*)&_f2_1173));
                acc32[36] = normalized_pair_1712.x;
                acc32[37] = normalized_pair_1712.y;
                float2 _f2_1174 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1713;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1713) : "l"(*(const unsigned long long*)&_f2_1174), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f2_1175 = make_float2(_unpack_bf16_53[2], _unpack_bf16_53[3]);
                float2 normalized_pair_1714;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1714) : "l"(*(const unsigned long long*)&scaled_pair_1713), "l"(*(const unsigned long long*)&_f2_1175));
                acc32[38] = normalized_pair_1714.x;
                acc32[39] = normalized_pair_1714.y;
                float2 _f2_1176 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1715;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1715) : "l"(*(const unsigned long long*)&_f2_1176), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f2_1177 = make_float2(_unpack_bf16_53[4], _unpack_bf16_53[5]);
                float2 normalized_pair_1716;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1716) : "l"(*(const unsigned long long*)&scaled_pair_1715), "l"(*(const unsigned long long*)&_f2_1177));
                acc32[40] = normalized_pair_1716.x;
                acc32[41] = normalized_pair_1716.y;
                float2 _f2_1178 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1717;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1717) : "l"(*(const unsigned long long*)&_f2_1178), "l"(*(const unsigned long long*)&output_rsigma_pair_1710));
                float2 _f2_1179 = make_float2(_unpack_bf16_53[6], _unpack_bf16_53[7]);
                float2 normalized_pair_1718;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1718) : "l"(*(const unsigned long long*)&scaled_pair_1717), "l"(*(const unsigned long long*)&_f2_1179));
                acc32[42] = normalized_pair_1718.x;
                acc32[43] = normalized_pair_1718.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[36 + 0], acc32[36 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[36 + 2], acc32[36 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[36 + 4], acc32[36 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[36 + 6], acc32[36 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1708 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_1180 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1721 = _f2_1180;
                float2 _f2_1181 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1722;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1722) : "l"(*(const unsigned long long*)&_f2_1181), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f2_1182 = make_float2(_unpack_bf16_54[0], _unpack_bf16_54[1]);
                float2 normalized_pair_1723;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1723) : "l"(*(const unsigned long long*)&scaled_pair_1722), "l"(*(const unsigned long long*)&_f2_1182));
                acc32[44] = normalized_pair_1723.x;
                acc32[45] = normalized_pair_1723.y;
                float2 _f2_1183 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1724;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1724) : "l"(*(const unsigned long long*)&_f2_1183), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f2_1184 = make_float2(_unpack_bf16_54[2], _unpack_bf16_54[3]);
                float2 normalized_pair_1725;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1725) : "l"(*(const unsigned long long*)&scaled_pair_1724), "l"(*(const unsigned long long*)&_f2_1184));
                acc32[46] = normalized_pair_1725.x;
                acc32[47] = normalized_pair_1725.y;
                float2 _f2_1185 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1726;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1726) : "l"(*(const unsigned long long*)&_f2_1185), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f2_1186 = make_float2(_unpack_bf16_54[4], _unpack_bf16_54[5]);
                float2 normalized_pair_1727;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1727) : "l"(*(const unsigned long long*)&scaled_pair_1726), "l"(*(const unsigned long long*)&_f2_1186));
                acc32[48] = normalized_pair_1727.x;
                acc32[49] = normalized_pair_1727.y;
                float2 _f2_1187 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1728;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1728) : "l"(*(const unsigned long long*)&_f2_1187), "l"(*(const unsigned long long*)&output_rsigma_pair_1721));
                float2 _f2_1188 = make_float2(_unpack_bf16_54[6], _unpack_bf16_54[7]);
                float2 normalized_pair_1729;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1729) : "l"(*(const unsigned long long*)&scaled_pair_1728), "l"(*(const unsigned long long*)&_f2_1188));
                acc32[50] = normalized_pair_1729.x;
                acc32[51] = normalized_pair_1729.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[44 + 0], acc32[44 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[44 + 2], acc32[44 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[44 + 4], acc32[44 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[44 + 6], acc32[44 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1719 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_1189 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1732 = _f2_1189;
                float2 _f2_1190 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1733;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1733) : "l"(*(const unsigned long long*)&_f2_1190), "l"(*(const unsigned long long*)&output_rsigma_pair_1732));
                float2 _f2_1191 = make_float2(_unpack_bf16_55[0], _unpack_bf16_55[1]);
                float2 normalized_pair_1734;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1734) : "l"(*(const unsigned long long*)&scaled_pair_1733), "l"(*(const unsigned long long*)&_f2_1191));
                acc32[52] = normalized_pair_1734.x;
                acc32[53] = normalized_pair_1734.y;
                float2 _f2_1192 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1735;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1735) : "l"(*(const unsigned long long*)&_f2_1192), "l"(*(const unsigned long long*)&output_rsigma_pair_1732));
                float2 _f2_1193 = make_float2(_unpack_bf16_55[2], _unpack_bf16_55[3]);
                float2 normalized_pair_1736;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1736) : "l"(*(const unsigned long long*)&scaled_pair_1735), "l"(*(const unsigned long long*)&_f2_1193));
                acc32[54] = normalized_pair_1736.x;
                acc32[55] = normalized_pair_1736.y;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(acc32[52 + 0], acc32[52 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[52 + 2], acc32[52 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1730]) = _pk2;
                }
            }
            mbarrier_wait(norm_ready_addr, 0);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("barrier.sync 0, 320;" ::: "memory");
            if (consumer_warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }

    // Cleanup
}

} // extern "C"
