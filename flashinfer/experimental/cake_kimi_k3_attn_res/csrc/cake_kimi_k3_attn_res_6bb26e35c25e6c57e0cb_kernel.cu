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
#define SMEM_STATS_STAGE_BYTES 768
#define SMEM_STATS_STRIDE 768
#define SMEM_OUTPUT_STATS_OFF 158464
#define SMEM_OUTPUT_STATS_STAGE_BYTES 64
#define SMEM_OUTPUT_STATS_STRIDE 64
#define SMEM_TOTAL 158720
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
kernel_cake_kimi_k3_attn_res_6bb26e35c25e6c57e0cb(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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

    if (warp == 0) {
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
    if (warp == 2) {
        int _tmem_hold = smem + 158600;
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
                unsigned int smem_regs_857[4];
                #pragma unroll
                for (int _lr_33 = 0; _lr_33 < 4; _lr_33++) {
                    smem_regs_857[_lr_33] = delta_words[stream_u_1 * 3584 + (unsigned int)(hidden_base_854 / 2) + (unsigned int)_lr_33];
                }
                __nv_bfloat162 prefix_pair = __as_bf16x2(smem_regs_855[0]);
                __nv_bfloat162 delta_pair = __as_bf16x2(smem_regs_857[0]);
                __nv_bfloat162 updated_bf16_pair = prefix_pair + delta_pair;
                float _cvt_f32_0 = (float)updated_bf16_pair.x;
                values_856[0] = _cvt_f32_0;
                float _cvt_f32_1 = (float)updated_bf16_pair.y;
                values_856[1] = _cvt_f32_1;
                __nv_bfloat162 prefix_pair_858 = __as_bf16x2(smem_regs_855[1]);
                __nv_bfloat162 delta_pair_859 = __as_bf16x2(smem_regs_857[1]);
                __nv_bfloat162 updated_bf16_pair_860 = prefix_pair_858 + delta_pair_859;
                float _cvt_f32_2 = (float)updated_bf16_pair_860.x;
                values_856[2] = _cvt_f32_2;
                float _cvt_f32_3 = (float)updated_bf16_pair_860.y;
                values_856[3] = _cvt_f32_3;
                __nv_bfloat162 prefix_pair_861 = __as_bf16x2(smem_regs_855[2]);
                __nv_bfloat162 delta_pair_862 = __as_bf16x2(smem_regs_857[2]);
                __nv_bfloat162 updated_bf16_pair_863 = prefix_pair_861 + delta_pair_862;
                float _cvt_f32_4 = (float)updated_bf16_pair_863.x;
                values_856[4] = _cvt_f32_4;
                float _cvt_f32_5 = (float)updated_bf16_pair_863.y;
                values_856[5] = _cvt_f32_5;
                __nv_bfloat162 prefix_pair_864 = __as_bf16x2(smem_regs_855[3]);
                __nv_bfloat162 delta_pair_865 = __as_bf16x2(smem_regs_857[3]);
                __nv_bfloat162 updated_bf16_pair_866 = prefix_pair_864 + delta_pair_865;
                float _cvt_f32_6 = (float)updated_bf16_pair_866.x;
                values_856[6] = _cvt_f32_6;
                float _cvt_f32_7 = (float)updated_bf16_pair_866.y;
                values_856[7] = _cvt_f32_7;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_856[0 + 0], values_856[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_856[0 + 2], values_856[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_856[0 + 4], values_856[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_856[0 + 6], values_856[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_854 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base, values_856);
                float2 _f2_648 = make_float2(values_856[0], values_856[1]);
                float2 value_pair_867 = _f2_648;
                float2 _f2_649 = make_float2(q_cache[0], q_cache[1]);
                float2 q_pair_868 = _f2_649;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_867, value_pair_867, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_867, q_pair_868, local_dot_853[0]);
                float2 _f2_650 = make_float2(values_856[2], values_856[3]);
                float2 value_pair_869 = _f2_650;
                float2 _f2_651 = make_float2(q_cache[2], q_cache[3]);
                float2 q_pair_870 = _f2_651;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_869, value_pair_869, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_869, q_pair_870, local_dot_853[0]);
                float2 _f2_652 = make_float2(values_856[4], values_856[5]);
                float2 value_pair_871 = _f2_652;
                float2 _f2_653 = make_float2(q_cache[4], q_cache[5]);
                float2 q_pair_872 = _f2_653;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_871, value_pair_871, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_871, q_pair_872, local_dot_853[0]);
                float2 _f2_654 = make_float2(values_856[6], values_856[7]);
                float2 value_pair_873 = _f2_654;
                float2 _f2_655 = make_float2(q_cache[6], q_cache[7]);
                float2 q_pair_874 = _f2_655;
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
                float _cvt_f32_8 = (float)updated_bf16_pair_881.x;
                values_877[0] = _cvt_f32_8;
                float _cvt_f32_9 = (float)updated_bf16_pair_881.y;
                values_877[1] = _cvt_f32_9;
                __nv_bfloat162 prefix_pair_882 = __as_bf16x2(smem_regs_876[1]);
                __nv_bfloat162 delta_pair_883 = __as_bf16x2(smem_regs_878[1]);
                __nv_bfloat162 updated_bf16_pair_884 = prefix_pair_882 + delta_pair_883;
                float _cvt_f32_10 = (float)updated_bf16_pair_884.x;
                values_877[2] = _cvt_f32_10;
                float _cvt_f32_11 = (float)updated_bf16_pair_884.y;
                values_877[3] = _cvt_f32_11;
                __nv_bfloat162 prefix_pair_885 = __as_bf16x2(smem_regs_876[2]);
                __nv_bfloat162 delta_pair_886 = __as_bf16x2(smem_regs_878[2]);
                __nv_bfloat162 updated_bf16_pair_887 = prefix_pair_885 + delta_pair_886;
                float _cvt_f32_12 = (float)updated_bf16_pair_887.x;
                values_877[4] = _cvt_f32_12;
                float _cvt_f32_13 = (float)updated_bf16_pair_887.y;
                values_877[5] = _cvt_f32_13;
                __nv_bfloat162 prefix_pair_888 = __as_bf16x2(smem_regs_876[3]);
                __nv_bfloat162 delta_pair_889 = __as_bf16x2(smem_regs_878[3]);
                __nv_bfloat162 updated_bf16_pair_890 = prefix_pair_888 + delta_pair_889;
                float _cvt_f32_14 = (float)updated_bf16_pair_890.x;
                values_877[6] = _cvt_f32_14;
                float _cvt_f32_15 = (float)updated_bf16_pair_890.y;
                values_877[7] = _cvt_f32_15;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_877[0 + 0], values_877[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_877[0 + 2], values_877[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_877[0 + 4], values_877[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_877[0 + 6], values_877[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_875 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 8, values_877);
                float2 _f2_656 = make_float2(values_877[0], values_877[1]);
                float2 value_pair_891 = _f2_656;
                float2 _f2_657 = make_float2(q_cache[8], q_cache[9]);
                float2 q_pair_892 = _f2_657;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_891, value_pair_891, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_891, q_pair_892, local_dot_853[0]);
                float2 _f2_658 = make_float2(values_877[2], values_877[3]);
                float2 value_pair_893 = _f2_658;
                float2 _f2_659 = make_float2(q_cache[10], q_cache[11]);
                float2 q_pair_894 = _f2_659;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_893, value_pair_893, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_893, q_pair_894, local_dot_853[0]);
                float2 _f2_660 = make_float2(values_877[4], values_877[5]);
                float2 value_pair_895 = _f2_660;
                float2 _f2_661 = make_float2(q_cache[12], q_cache[13]);
                float2 q_pair_896 = _f2_661;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_895, value_pair_895, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_895, q_pair_896, local_dot_853[0]);
                float2 _f2_662 = make_float2(values_877[6], values_877[7]);
                float2 value_pair_897 = _f2_662;
                float2 _f2_663 = make_float2(q_cache[14], q_cache[15]);
                float2 q_pair_898 = _f2_663;
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
                float _cvt_f32_16 = (float)updated_bf16_pair_905.x;
                values_901[0] = _cvt_f32_16;
                float _cvt_f32_17 = (float)updated_bf16_pair_905.y;
                values_901[1] = _cvt_f32_17;
                __nv_bfloat162 prefix_pair_906 = __as_bf16x2(smem_regs_900[1]);
                __nv_bfloat162 delta_pair_907 = __as_bf16x2(smem_regs_902[1]);
                __nv_bfloat162 updated_bf16_pair_908 = prefix_pair_906 + delta_pair_907;
                float _cvt_f32_18 = (float)updated_bf16_pair_908.x;
                values_901[2] = _cvt_f32_18;
                float _cvt_f32_19 = (float)updated_bf16_pair_908.y;
                values_901[3] = _cvt_f32_19;
                __nv_bfloat162 prefix_pair_909 = __as_bf16x2(smem_regs_900[2]);
                __nv_bfloat162 delta_pair_910 = __as_bf16x2(smem_regs_902[2]);
                __nv_bfloat162 updated_bf16_pair_911 = prefix_pair_909 + delta_pair_910;
                float _cvt_f32_20 = (float)updated_bf16_pair_911.x;
                values_901[4] = _cvt_f32_20;
                float _cvt_f32_21 = (float)updated_bf16_pair_911.y;
                values_901[5] = _cvt_f32_21;
                __nv_bfloat162 prefix_pair_912 = __as_bf16x2(smem_regs_900[3]);
                __nv_bfloat162 delta_pair_913 = __as_bf16x2(smem_regs_902[3]);
                __nv_bfloat162 updated_bf16_pair_914 = prefix_pair_912 + delta_pair_913;
                float _cvt_f32_22 = (float)updated_bf16_pair_914.x;
                values_901[6] = _cvt_f32_22;
                float _cvt_f32_23 = (float)updated_bf16_pair_914.y;
                values_901[7] = _cvt_f32_23;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_901[0 + 0], values_901[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_901[0 + 2], values_901[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_901[0 + 4], values_901[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_901[0 + 6], values_901[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_899 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 16, values_901);
                float2 _f2_664 = make_float2(values_901[0], values_901[1]);
                float2 value_pair_915 = _f2_664;
                float2 _f2_665 = make_float2(q_cache[16], q_cache[17]);
                float2 q_pair_916 = _f2_665;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_915, value_pair_915, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_915, q_pair_916, local_dot_853[0]);
                float2 _f2_666 = make_float2(values_901[2], values_901[3]);
                float2 value_pair_917 = _f2_666;
                float2 _f2_667 = make_float2(q_cache[18], q_cache[19]);
                float2 q_pair_918 = _f2_667;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_917, value_pair_917, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_917, q_pair_918, local_dot_853[0]);
                float2 _f2_668 = make_float2(values_901[4], values_901[5]);
                float2 value_pair_919 = _f2_668;
                float2 _f2_669 = make_float2(q_cache[20], q_cache[21]);
                float2 q_pair_920 = _f2_669;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_919, value_pair_919, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_919, q_pair_920, local_dot_853[0]);
                float2 _f2_670 = make_float2(values_901[6], values_901[7]);
                float2 value_pair_921 = _f2_670;
                float2 _f2_671 = make_float2(q_cache[22], q_cache[23]);
                float2 q_pair_922 = _f2_671;
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
                float _cvt_f32_24 = (float)updated_bf16_pair_929.x;
                values_925[0] = _cvt_f32_24;
                float _cvt_f32_25 = (float)updated_bf16_pair_929.y;
                values_925[1] = _cvt_f32_25;
                __nv_bfloat162 prefix_pair_930 = __as_bf16x2(smem_regs_924[1]);
                __nv_bfloat162 delta_pair_931 = __as_bf16x2(smem_regs_926[1]);
                __nv_bfloat162 updated_bf16_pair_932 = prefix_pair_930 + delta_pair_931;
                float _cvt_f32_26 = (float)updated_bf16_pair_932.x;
                values_925[2] = _cvt_f32_26;
                float _cvt_f32_27 = (float)updated_bf16_pair_932.y;
                values_925[3] = _cvt_f32_27;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(values_925[0 + 0], values_925[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_925[0 + 2], values_925[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_923]) = _pk2;
                }
                tmem_st_x4_f32(tmem_base + 24, values_925);
                float2 _f2_672 = make_float2(values_925[0], values_925[1]);
                float2 value_pair_933 = _f2_672;
                float2 _f2_673 = make_float2(q_cache[24], q_cache[25]);
                float2 q_pair_934 = _f2_673;
                local_sum_sq_852[0] = fma_f32x2_rn_ftz(value_pair_933, value_pair_933, local_sum_sq_852[0]);
                local_dot_853[0] = fma_f32x2_rn_ftz(value_pair_933, q_pair_934, local_dot_853[0]);
                float2 _f2_674 = make_float2(values_925[2], values_925[3]);
                float2 value_pair_935 = _f2_674;
                float2 _f2_675 = make_float2(q_cache[26], q_cache[27]);
                float2 q_pair_936 = _f2_675;
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
                float _cvt_f32_28 = (float)updated_bf16_pair_943.x;
                values_939[0] = _cvt_f32_28;
                float _cvt_f32_29 = (float)updated_bf16_pair_943.y;
                values_939[1] = _cvt_f32_29;
                __nv_bfloat162 prefix_pair_944 = __as_bf16x2(smem_regs_938[1]);
                __nv_bfloat162 delta_pair_945 = __as_bf16x2(smem_regs_940[1]);
                __nv_bfloat162 updated_bf16_pair_946 = prefix_pair_944 + delta_pair_945;
                float _cvt_f32_30 = (float)updated_bf16_pair_946.x;
                values_939[2] = _cvt_f32_30;
                float _cvt_f32_31 = (float)updated_bf16_pair_946.y;
                values_939[3] = _cvt_f32_31;
                __nv_bfloat162 prefix_pair_947 = __as_bf16x2(smem_regs_938[2]);
                __nv_bfloat162 delta_pair_948 = __as_bf16x2(smem_regs_940[2]);
                __nv_bfloat162 updated_bf16_pair_949 = prefix_pair_947 + delta_pair_948;
                float _cvt_f32_32 = (float)updated_bf16_pair_949.x;
                values_939[4] = _cvt_f32_32;
                float _cvt_f32_33 = (float)updated_bf16_pair_949.y;
                values_939[5] = _cvt_f32_33;
                __nv_bfloat162 prefix_pair_950 = __as_bf16x2(smem_regs_938[3]);
                __nv_bfloat162 delta_pair_951 = __as_bf16x2(smem_regs_940[3]);
                __nv_bfloat162 updated_bf16_pair_952 = prefix_pair_950 + delta_pair_951;
                float _cvt_f32_34 = (float)updated_bf16_pair_952.x;
                values_939[6] = _cvt_f32_34;
                float _cvt_f32_35 = (float)updated_bf16_pair_952.y;
                values_939[7] = _cvt_f32_35;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_939[0 + 0], values_939[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_939[0 + 2], values_939[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_939[0 + 4], values_939[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_939[0 + 6], values_939[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_937 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 32, values_939);
                float2 _f2_676 = make_float2(values_939[0], values_939[1]);
                float2 value_pair_953 = _f2_676;
                float2 _f2_677 = make_float2(q_cache[28], q_cache[29]);
                float2 q_pair_954 = _f2_677;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_953, value_pair_953, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_953, q_pair_954, local_dot_853[2]);
                float2 _f2_678 = make_float2(values_939[2], values_939[3]);
                float2 value_pair_955 = _f2_678;
                float2 _f2_679 = make_float2(q_cache[30], q_cache[31]);
                float2 q_pair_956 = _f2_679;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_955, value_pair_955, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_955, q_pair_956, local_dot_853[2]);
                float2 _f2_680 = make_float2(values_939[4], values_939[5]);
                float2 value_pair_957 = _f2_680;
                float2 _f2_681 = make_float2(q_cache[32], q_cache[33]);
                float2 q_pair_958 = _f2_681;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_957, value_pair_957, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_957, q_pair_958, local_dot_853[2]);
                float2 _f2_682 = make_float2(values_939[6], values_939[7]);
                float2 value_pair_959 = _f2_682;
                float2 _f2_683 = make_float2(q_cache[34], q_cache[35]);
                float2 q_pair_960 = _f2_683;
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
                float _cvt_f32_36 = (float)updated_bf16_pair_967.x;
                values_963[0] = _cvt_f32_36;
                float _cvt_f32_37 = (float)updated_bf16_pair_967.y;
                values_963[1] = _cvt_f32_37;
                __nv_bfloat162 prefix_pair_968 = __as_bf16x2(smem_regs_962[1]);
                __nv_bfloat162 delta_pair_969 = __as_bf16x2(smem_regs_964[1]);
                __nv_bfloat162 updated_bf16_pair_970 = prefix_pair_968 + delta_pair_969;
                float _cvt_f32_38 = (float)updated_bf16_pair_970.x;
                values_963[2] = _cvt_f32_38;
                float _cvt_f32_39 = (float)updated_bf16_pair_970.y;
                values_963[3] = _cvt_f32_39;
                __nv_bfloat162 prefix_pair_971 = __as_bf16x2(smem_regs_962[2]);
                __nv_bfloat162 delta_pair_972 = __as_bf16x2(smem_regs_964[2]);
                __nv_bfloat162 updated_bf16_pair_973 = prefix_pair_971 + delta_pair_972;
                float _cvt_f32_40 = (float)updated_bf16_pair_973.x;
                values_963[4] = _cvt_f32_40;
                float _cvt_f32_41 = (float)updated_bf16_pair_973.y;
                values_963[5] = _cvt_f32_41;
                __nv_bfloat162 prefix_pair_974 = __as_bf16x2(smem_regs_962[3]);
                __nv_bfloat162 delta_pair_975 = __as_bf16x2(smem_regs_964[3]);
                __nv_bfloat162 updated_bf16_pair_976 = prefix_pair_974 + delta_pair_975;
                float _cvt_f32_42 = (float)updated_bf16_pair_976.x;
                values_963[6] = _cvt_f32_42;
                float _cvt_f32_43 = (float)updated_bf16_pair_976.y;
                values_963[7] = _cvt_f32_43;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_963[0 + 0], values_963[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_963[0 + 2], values_963[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_963[0 + 4], values_963[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_963[0 + 6], values_963[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_961 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 40, values_963);
                float2 _f2_684 = make_float2(values_963[0], values_963[1]);
                float2 value_pair_977 = _f2_684;
                float2 _f2_685 = make_float2(q_cache[36], q_cache[37]);
                float2 q_pair_978 = _f2_685;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_977, value_pair_977, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_977, q_pair_978, local_dot_853[2]);
                float2 _f2_686 = make_float2(values_963[2], values_963[3]);
                float2 value_pair_979 = _f2_686;
                float2 _f2_687 = make_float2(q_cache[38], q_cache[39]);
                float2 q_pair_980 = _f2_687;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_979, value_pair_979, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_979, q_pair_980, local_dot_853[2]);
                float2 _f2_688 = make_float2(values_963[4], values_963[5]);
                float2 value_pair_981 = _f2_688;
                float2 _f2_689 = make_float2(q_cache[40], q_cache[41]);
                float2 q_pair_982 = _f2_689;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_981, value_pair_981, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_981, q_pair_982, local_dot_853[2]);
                float2 _f2_690 = make_float2(values_963[6], values_963[7]);
                float2 value_pair_983 = _f2_690;
                float2 _f2_691 = make_float2(q_cache[42], q_cache[43]);
                float2 q_pair_984 = _f2_691;
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
                float _cvt_f32_44 = (float)updated_bf16_pair_991.x;
                values_987[0] = _cvt_f32_44;
                float _cvt_f32_45 = (float)updated_bf16_pair_991.y;
                values_987[1] = _cvt_f32_45;
                __nv_bfloat162 prefix_pair_992 = __as_bf16x2(smem_regs_986[1]);
                __nv_bfloat162 delta_pair_993 = __as_bf16x2(smem_regs_988[1]);
                __nv_bfloat162 updated_bf16_pair_994 = prefix_pair_992 + delta_pair_993;
                float _cvt_f32_46 = (float)updated_bf16_pair_994.x;
                values_987[2] = _cvt_f32_46;
                float _cvt_f32_47 = (float)updated_bf16_pair_994.y;
                values_987[3] = _cvt_f32_47;
                __nv_bfloat162 prefix_pair_995 = __as_bf16x2(smem_regs_986[2]);
                __nv_bfloat162 delta_pair_996 = __as_bf16x2(smem_regs_988[2]);
                __nv_bfloat162 updated_bf16_pair_997 = prefix_pair_995 + delta_pair_996;
                float _cvt_f32_48 = (float)updated_bf16_pair_997.x;
                values_987[4] = _cvt_f32_48;
                float _cvt_f32_49 = (float)updated_bf16_pair_997.y;
                values_987[5] = _cvt_f32_49;
                __nv_bfloat162 prefix_pair_998 = __as_bf16x2(smem_regs_986[3]);
                __nv_bfloat162 delta_pair_999 = __as_bf16x2(smem_regs_988[3]);
                __nv_bfloat162 updated_bf16_pair_1000 = prefix_pair_998 + delta_pair_999;
                float _cvt_f32_50 = (float)updated_bf16_pair_1000.x;
                values_987[6] = _cvt_f32_50;
                float _cvt_f32_51 = (float)updated_bf16_pair_1000.y;
                values_987[7] = _cvt_f32_51;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(values_987[0 + 0], values_987[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_987[0 + 2], values_987[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(values_987[0 + 4], values_987[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(values_987[0 + 6], values_987[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_985 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                tmem_st_x8_f32(tmem_base + 48, values_987);
                float2 _f2_692 = make_float2(values_987[0], values_987[1]);
                float2 value_pair_1001 = _f2_692;
                float2 _f2_693 = make_float2(q_cache[44], q_cache[45]);
                float2 q_pair_1002 = _f2_693;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1001, value_pair_1001, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1001, q_pair_1002, local_dot_853[2]);
                float2 _f2_694 = make_float2(values_987[2], values_987[3]);
                float2 value_pair_1003 = _f2_694;
                float2 _f2_695 = make_float2(q_cache[46], q_cache[47]);
                float2 q_pair_1004 = _f2_695;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1003, value_pair_1003, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1003, q_pair_1004, local_dot_853[2]);
                float2 _f2_696 = make_float2(values_987[4], values_987[5]);
                float2 value_pair_1005 = _f2_696;
                float2 _f2_697 = make_float2(q_cache[48], q_cache[49]);
                float2 q_pair_1006 = _f2_697;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1005, value_pair_1005, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1005, q_pair_1006, local_dot_853[2]);
                float2 _f2_698 = make_float2(values_987[6], values_987[7]);
                float2 value_pair_1007 = _f2_698;
                float2 _f2_699 = make_float2(q_cache[50], q_cache[51]);
                float2 q_pair_1008 = _f2_699;
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
                float _cvt_f32_52 = (float)updated_bf16_pair_1015.x;
                values_1011[0] = _cvt_f32_52;
                float _cvt_f32_53 = (float)updated_bf16_pair_1015.y;
                values_1011[1] = _cvt_f32_53;
                __nv_bfloat162 prefix_pair_1016 = __as_bf16x2(smem_regs_1010[1]);
                __nv_bfloat162 delta_pair_1017 = __as_bf16x2(smem_regs_1012[1]);
                __nv_bfloat162 updated_bf16_pair_1018 = prefix_pair_1016 + delta_pair_1017;
                float _cvt_f32_54 = (float)updated_bf16_pair_1018.x;
                values_1011[2] = _cvt_f32_54;
                float _cvt_f32_55 = (float)updated_bf16_pair_1018.y;
                values_1011[3] = _cvt_f32_55;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(values_1011[0 + 0], values_1011[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(values_1011[0 + 2], values_1011[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(prefix))[token_1 * 7168 + hidden_base_1009]) = _pk2;
                }
                tmem_st_x4_f32(tmem_base + 56, values_1011);
                float2 _f2_700 = make_float2(values_1011[0], values_1011[1]);
                float2 value_pair_1019 = _f2_700;
                float2 _f2_701 = make_float2(q_cache[52], q_cache[53]);
                float2 q_pair_1020 = _f2_701;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1019, value_pair_1019, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1019, q_pair_1020, local_dot_853[2]);
                float2 _f2_702 = make_float2(values_1011[2], values_1011[3]);
                float2 value_pair_1021 = _f2_702;
                float2 _f2_703 = make_float2(q_cache[54], q_cache[55]);
                float2 q_pair_1022 = _f2_703;
                local_sum_sq_852[2] = fma_f32x2_rn_ftz(value_pair_1021, value_pair_1021, local_sum_sq_852[2]);
                local_dot_853[2] = fma_f32x2_rn_ftz(value_pair_1021, q_pair_1022, local_dot_853[2]);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(consumed_addr + (ring_stage_851) * 8);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                unsigned int slot_base_1023 = stats_stream_base + 64;
                float2 _f2_704 = make_float2(local_sum_sq_852[0].x + local_sum_sq_852[0].y, local_dot_853[0].x + local_dot_853[0].y);
                float2 reduce_pair_1024 = _f2_704;
                unsigned long long packed_pair_1025 = 0;
                unsigned long long peer_packed_1026 = 0;
                float2 _f2_705 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1027 = _f2_705;
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 16);
                peer_packed_1026 = _shfl_xor_40;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 8);
                peer_packed_1026 = _shfl_xor_41;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 4);
                peer_packed_1026 = _shfl_xor_42;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 2);
                peer_packed_1026 = _shfl_xor_43;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                packed_pair_1025 = reinterpret_cast<unsigned long long*>(&reduce_pair_1024)[0];
                unsigned long long _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1025, 1);
                peer_packed_1026 = _shfl_xor_44;
                peer_pair_1027 = reinterpret_cast<float2*>(&peer_packed_1026)[0];
                reduce_pair_1024 = add_f32x2(reduce_pair_1024, peer_pair_1027);
                if (lane == 0) {
                    stats[slot_base_1023 + (unsigned int)(warp_in_group * 2 * 2)] = reduce_pair_1024.x;
                    stats[slot_base_1023 + (unsigned int)(warp_in_group * 2 * 2) + 1] = reduce_pair_1024.y;
                }
                float2 _f2_706 = make_float2(local_sum_sq_852[2].x + local_sum_sq_852[2].y, local_dot_853[2].x + local_dot_853[2].y);
                float2 reduce_pair_1028 = _f2_706;
                unsigned long long packed_pair_1029 = 0;
                unsigned long long peer_packed_1030 = 0;
                float2 _f2_707 = make_float2(0.0f, 0.0f);
                float2 peer_pair_1031 = _f2_707;
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 16);
                peer_packed_1030 = _shfl_xor_45;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 8);
                peer_packed_1030 = _shfl_xor_46;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 4);
                peer_packed_1030 = _shfl_xor_47;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 2);
                peer_packed_1030 = _shfl_xor_48;
                peer_pair_1031 = reinterpret_cast<float2*>(&peer_packed_1030)[0];
                reduce_pair_1028 = add_f32x2(reduce_pair_1028, peer_pair_1031);
                packed_pair_1029 = reinterpret_cast<unsigned long long*>(&reduce_pair_1028)[0];
                unsigned long long _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, packed_pair_1029, 1);
                peer_packed_1030 = _shfl_xor_49;
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
                float _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1034, 4, 8);
                total_sum_sq_1034 += _shfl_down_12;
                float _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, total_dot_1035, 4, 8);
                total_dot_1035 += _shfl_down_13;
                float _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1034, 2, 8);
                total_sum_sq_1034 += _shfl_down_14;
                float _shfl_down_15 = __shfl_down_sync(0xFFFFFFFF, total_dot_1035, 2, 8);
                total_dot_1035 += _shfl_down_15;
                float _shfl_down_16 = __shfl_down_sync(0xFFFFFFFF, total_sum_sq_1034, 1, 8);
                total_sum_sq_1034 += _shfl_down_16;
                float _shfl_down_17 = __shfl_down_sync(0xFFFFFFFF, total_dot_1035, 1, 8);
                total_dot_1035 += _shfl_down_17;
                float local_logit_1036 = 0.0f;
                if (stat_source_1032 < 1 && stat_warp_1033 == 0) {
                    float _rsqrt_2 = rsqrtf(total_sum_sq_1034 / 7168.0f + eps);
                    local_logit_1036 = total_dot_1035 * _rsqrt_2;
                }
                float logits_1037[2];
                logits_1037[0] = 0.0f;
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, local_logit_1036, 0);
                logits_1037[0] = _shfl_4;
                logits_1037[1] = 0.0f;
                float _shfl_5 = __shfl_sync(0xFFFFFFFF, local_logit_1036, 8);
                logits_1037[1] = _shfl_5;
                float weights_1038[2];
                float new_max_1039 = 0.0f;
                float correction_1040 = 0.0f;
                float weight_sum_1041 = 0.0f;
                weights_1038[0] = 0.0f;
                weights_1038[1] = 0.0f;
                float chunk_max_1042 = -3.4028234663852886e+38f;
                float _max_6 = max_noftz(chunk_max_1042, logits_1037[0]);
                chunk_max_1042 = _max_6;
                float _max_7 = max_noftz(m_running, chunk_max_1042);
                new_max_1039 = _max_7;
                float _exp2_6 = approx_exp2((m_running - new_max_1039) * 1.4426950408889634f);
                correction_1040 = _exp2_6;
                float _exp2_7 = approx_exp2((logits_1037[0] - new_max_1039) * 1.4426950408889634f);
                weights_1038[0] = _exp2_7;
                weight_sum_1041 += weights_1038[0];
                float2 _f2_708 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1043 = _f2_708;
                float2 _f2_709 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1044 = _f2_709;
                float2 scaled_pair_1045;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1045) : "l"(*(const unsigned long long*)&acc_pair_1044), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[0] = scaled_pair_1045.x;
                acc32[1] = scaled_pair_1045.y;
                float2 _f2_710 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1046 = _f2_710;
                float2 scaled_pair_1047;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1047) : "l"(*(const unsigned long long*)&acc_pair_1046), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[2] = scaled_pair_1047.x;
                acc32[3] = scaled_pair_1047.y;
                float2 _f2_711 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1048 = _f2_711;
                float2 scaled_pair_1049;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1049) : "l"(*(const unsigned long long*)&acc_pair_1048), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[4] = scaled_pair_1049.x;
                acc32[5] = scaled_pair_1049.y;
                float2 _f2_712 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1050 = _f2_712;
                float2 scaled_pair_1051;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1051) : "l"(*(const unsigned long long*)&acc_pair_1050), "l"(*(const unsigned long long*)&correction_pair_1043));
                acc32[6] = scaled_pair_1051.x;
                acc32[7] = scaled_pair_1051.y;
                float _tmem_load_32[8];
                tmem_ld_x8(&_tmem_load_32[0], tmem_base);
                float2 _f2_713 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1052 = _f2_713;
                float2 _f2_714 = make_float2(_tmem_load_32[0], _tmem_load_32[1]);
                float2 cached_pair_1053 = _f2_714;
                float2 _f2_715 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1054 = _f2_715;
                float2 updated_pair_1055 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1053, acc_pair_1054);
                acc32[0] = updated_pair_1055.x;
                acc32[1] = updated_pair_1055.y;
                float2 _f2_716 = make_float2(_tmem_load_32[2], _tmem_load_32[3]);
                float2 cached_pair_1056 = _f2_716;
                float2 _f2_717 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1057 = _f2_717;
                float2 updated_pair_1058 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1056, acc_pair_1057);
                acc32[2] = updated_pair_1058.x;
                acc32[3] = updated_pair_1058.y;
                float2 _f2_718 = make_float2(_tmem_load_32[4], _tmem_load_32[5]);
                float2 cached_pair_1059 = _f2_718;
                float2 _f2_719 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1060 = _f2_719;
                float2 updated_pair_1061 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1059, acc_pair_1060);
                acc32[4] = updated_pair_1061.x;
                acc32[5] = updated_pair_1061.y;
                float2 _f2_720 = make_float2(_tmem_load_32[6], _tmem_load_32[7]);
                float2 cached_pair_1062 = _f2_720;
                float2 _f2_721 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1063 = _f2_721;
                float2 updated_pair_1064 = fma_f32x2_rn_ftz(weight_pair_1052, cached_pair_1062, acc_pair_1063);
                acc32[6] = updated_pair_1064.x;
                acc32[7] = updated_pair_1064.y;
                float2 _f2_722 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1065 = _f2_722;
                float2 _f2_723 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1066 = _f2_723;
                float2 scaled_pair_1067;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1067) : "l"(*(const unsigned long long*)&acc_pair_1066), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[8] = scaled_pair_1067.x;
                acc32[9] = scaled_pair_1067.y;
                float2 _f2_724 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1068 = _f2_724;
                float2 scaled_pair_1069;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1069) : "l"(*(const unsigned long long*)&acc_pair_1068), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[10] = scaled_pair_1069.x;
                acc32[11] = scaled_pair_1069.y;
                float2 _f2_725 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1070 = _f2_725;
                float2 scaled_pair_1071;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1071) : "l"(*(const unsigned long long*)&acc_pair_1070), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[12] = scaled_pair_1071.x;
                acc32[13] = scaled_pair_1071.y;
                float2 _f2_726 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1072 = _f2_726;
                float2 scaled_pair_1073;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1073) : "l"(*(const unsigned long long*)&acc_pair_1072), "l"(*(const unsigned long long*)&correction_pair_1065));
                acc32[14] = scaled_pair_1073.x;
                acc32[15] = scaled_pair_1073.y;
                float _tmem_load_33[8];
                tmem_ld_x8(&_tmem_load_33[0], tmem_base + 8);
                float2 _f2_727 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1074 = _f2_727;
                float2 _f2_728 = make_float2(_tmem_load_33[0], _tmem_load_33[1]);
                float2 cached_pair_1075 = _f2_728;
                float2 _f2_729 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1076 = _f2_729;
                float2 updated_pair_1077 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1075, acc_pair_1076);
                acc32[8] = updated_pair_1077.x;
                acc32[9] = updated_pair_1077.y;
                float2 _f2_730 = make_float2(_tmem_load_33[2], _tmem_load_33[3]);
                float2 cached_pair_1078 = _f2_730;
                float2 _f2_731 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1079 = _f2_731;
                float2 updated_pair_1080 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1078, acc_pair_1079);
                acc32[10] = updated_pair_1080.x;
                acc32[11] = updated_pair_1080.y;
                float2 _f2_732 = make_float2(_tmem_load_33[4], _tmem_load_33[5]);
                float2 cached_pair_1081 = _f2_732;
                float2 _f2_733 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1082 = _f2_733;
                float2 updated_pair_1083 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1081, acc_pair_1082);
                acc32[12] = updated_pair_1083.x;
                acc32[13] = updated_pair_1083.y;
                float2 _f2_734 = make_float2(_tmem_load_33[6], _tmem_load_33[7]);
                float2 cached_pair_1084 = _f2_734;
                float2 _f2_735 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1085 = _f2_735;
                float2 updated_pair_1086 = fma_f32x2_rn_ftz(weight_pair_1074, cached_pair_1084, acc_pair_1085);
                acc32[14] = updated_pair_1086.x;
                acc32[15] = updated_pair_1086.y;
                float2 _f2_736 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1087 = _f2_736;
                float2 _f2_737 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1088 = _f2_737;
                float2 scaled_pair_1089;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1089) : "l"(*(const unsigned long long*)&acc_pair_1088), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[16] = scaled_pair_1089.x;
                acc32[17] = scaled_pair_1089.y;
                float2 _f2_738 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1090 = _f2_738;
                float2 scaled_pair_1091;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1091) : "l"(*(const unsigned long long*)&acc_pair_1090), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[18] = scaled_pair_1091.x;
                acc32[19] = scaled_pair_1091.y;
                float2 _f2_739 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1092 = _f2_739;
                float2 scaled_pair_1093;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1093) : "l"(*(const unsigned long long*)&acc_pair_1092), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[20] = scaled_pair_1093.x;
                acc32[21] = scaled_pair_1093.y;
                float2 _f2_740 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1094 = _f2_740;
                float2 scaled_pair_1095;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1095) : "l"(*(const unsigned long long*)&acc_pair_1094), "l"(*(const unsigned long long*)&correction_pair_1087));
                acc32[22] = scaled_pair_1095.x;
                acc32[23] = scaled_pair_1095.y;
                float _tmem_load_34[8];
                tmem_ld_x8(&_tmem_load_34[0], tmem_base + 16);
                float2 _f2_741 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1096 = _f2_741;
                float2 _f2_742 = make_float2(_tmem_load_34[0], _tmem_load_34[1]);
                float2 cached_pair_1097 = _f2_742;
                float2 _f2_743 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1098 = _f2_743;
                float2 updated_pair_1099 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1097, acc_pair_1098);
                acc32[16] = updated_pair_1099.x;
                acc32[17] = updated_pair_1099.y;
                float2 _f2_744 = make_float2(_tmem_load_34[2], _tmem_load_34[3]);
                float2 cached_pair_1100 = _f2_744;
                float2 _f2_745 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1101 = _f2_745;
                float2 updated_pair_1102 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1100, acc_pair_1101);
                acc32[18] = updated_pair_1102.x;
                acc32[19] = updated_pair_1102.y;
                float2 _f2_746 = make_float2(_tmem_load_34[4], _tmem_load_34[5]);
                float2 cached_pair_1103 = _f2_746;
                float2 _f2_747 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1104 = _f2_747;
                float2 updated_pair_1105 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1103, acc_pair_1104);
                acc32[20] = updated_pair_1105.x;
                acc32[21] = updated_pair_1105.y;
                float2 _f2_748 = make_float2(_tmem_load_34[6], _tmem_load_34[7]);
                float2 cached_pair_1106 = _f2_748;
                float2 _f2_749 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1107 = _f2_749;
                float2 updated_pair_1108 = fma_f32x2_rn_ftz(weight_pair_1096, cached_pair_1106, acc_pair_1107);
                acc32[22] = updated_pair_1108.x;
                acc32[23] = updated_pair_1108.y;
                float2 _f2_750 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1109 = _f2_750;
                float2 _f2_751 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1110 = _f2_751;
                float2 scaled_pair_1111;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1111) : "l"(*(const unsigned long long*)&acc_pair_1110), "l"(*(const unsigned long long*)&correction_pair_1109));
                acc32[24] = scaled_pair_1111.x;
                acc32[25] = scaled_pair_1111.y;
                float2 _f2_752 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1112 = _f2_752;
                float2 scaled_pair_1113;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1113) : "l"(*(const unsigned long long*)&acc_pair_1112), "l"(*(const unsigned long long*)&correction_pair_1109));
                acc32[26] = scaled_pair_1113.x;
                acc32[27] = scaled_pair_1113.y;
                float _tmem_load_35[4];
                tmem_ld_x4(&_tmem_load_35[0], tmem_base + 24);
                float2 _f2_753 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1114 = _f2_753;
                float2 _f2_754 = make_float2(_tmem_load_35[0], _tmem_load_35[1]);
                float2 cached_pair_1115 = _f2_754;
                float2 _f2_755 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1116 = _f2_755;
                float2 updated_pair_1117 = fma_f32x2_rn_ftz(weight_pair_1114, cached_pair_1115, acc_pair_1116);
                acc32[24] = updated_pair_1117.x;
                acc32[25] = updated_pair_1117.y;
                float2 _f2_756 = make_float2(_tmem_load_35[2], _tmem_load_35[3]);
                float2 cached_pair_1118 = _f2_756;
                float2 _f2_757 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1119 = _f2_757;
                float2 updated_pair_1120 = fma_f32x2_rn_ftz(weight_pair_1114, cached_pair_1118, acc_pair_1119);
                acc32[26] = updated_pair_1120.x;
                acc32[27] = updated_pair_1120.y;
                float2 _f2_758 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1121 = _f2_758;
                float2 _f2_759 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1122 = _f2_759;
                float2 scaled_pair_1123;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1123) : "l"(*(const unsigned long long*)&acc_pair_1122), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[28] = scaled_pair_1123.x;
                acc32[29] = scaled_pair_1123.y;
                float2 _f2_760 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1124 = _f2_760;
                float2 scaled_pair_1125;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1125) : "l"(*(const unsigned long long*)&acc_pair_1124), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[30] = scaled_pair_1125.x;
                acc32[31] = scaled_pair_1125.y;
                float2 _f2_761 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1126 = _f2_761;
                float2 scaled_pair_1127;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1127) : "l"(*(const unsigned long long*)&acc_pair_1126), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[32] = scaled_pair_1127.x;
                acc32[33] = scaled_pair_1127.y;
                float2 _f2_762 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1128 = _f2_762;
                float2 scaled_pair_1129;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1129) : "l"(*(const unsigned long long*)&acc_pair_1128), "l"(*(const unsigned long long*)&correction_pair_1121));
                acc32[34] = scaled_pair_1129.x;
                acc32[35] = scaled_pair_1129.y;
                float _tmem_load_36[8];
                tmem_ld_x8(&_tmem_load_36[0], tmem_base + 32);
                float2 _f2_763 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1130 = _f2_763;
                float2 _f2_764 = make_float2(_tmem_load_36[0], _tmem_load_36[1]);
                float2 cached_pair_1131 = _f2_764;
                float2 _f2_765 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1132 = _f2_765;
                float2 updated_pair_1133 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1131, acc_pair_1132);
                acc32[28] = updated_pair_1133.x;
                acc32[29] = updated_pair_1133.y;
                float2 _f2_766 = make_float2(_tmem_load_36[2], _tmem_load_36[3]);
                float2 cached_pair_1134 = _f2_766;
                float2 _f2_767 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1135 = _f2_767;
                float2 updated_pair_1136 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1134, acc_pair_1135);
                acc32[30] = updated_pair_1136.x;
                acc32[31] = updated_pair_1136.y;
                float2 _f2_768 = make_float2(_tmem_load_36[4], _tmem_load_36[5]);
                float2 cached_pair_1137 = _f2_768;
                float2 _f2_769 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1138 = _f2_769;
                float2 updated_pair_1139 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1137, acc_pair_1138);
                acc32[32] = updated_pair_1139.x;
                acc32[33] = updated_pair_1139.y;
                float2 _f2_770 = make_float2(_tmem_load_36[6], _tmem_load_36[7]);
                float2 cached_pair_1140 = _f2_770;
                float2 _f2_771 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1141 = _f2_771;
                float2 updated_pair_1142 = fma_f32x2_rn_ftz(weight_pair_1130, cached_pair_1140, acc_pair_1141);
                acc32[34] = updated_pair_1142.x;
                acc32[35] = updated_pair_1142.y;
                float2 _f2_772 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1143 = _f2_772;
                float2 _f2_773 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1144 = _f2_773;
                float2 scaled_pair_1145;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1145) : "l"(*(const unsigned long long*)&acc_pair_1144), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[36] = scaled_pair_1145.x;
                acc32[37] = scaled_pair_1145.y;
                float2 _f2_774 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1146 = _f2_774;
                float2 scaled_pair_1147;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1147) : "l"(*(const unsigned long long*)&acc_pair_1146), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[38] = scaled_pair_1147.x;
                acc32[39] = scaled_pair_1147.y;
                float2 _f2_775 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1148 = _f2_775;
                float2 scaled_pair_1149;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1149) : "l"(*(const unsigned long long*)&acc_pair_1148), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[40] = scaled_pair_1149.x;
                acc32[41] = scaled_pair_1149.y;
                float2 _f2_776 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1150 = _f2_776;
                float2 scaled_pair_1151;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1151) : "l"(*(const unsigned long long*)&acc_pair_1150), "l"(*(const unsigned long long*)&correction_pair_1143));
                acc32[42] = scaled_pair_1151.x;
                acc32[43] = scaled_pair_1151.y;
                float _tmem_load_37[8];
                tmem_ld_x8(&_tmem_load_37[0], tmem_base + 40);
                float2 _f2_777 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1152 = _f2_777;
                float2 _f2_778 = make_float2(_tmem_load_37[0], _tmem_load_37[1]);
                float2 cached_pair_1153 = _f2_778;
                float2 _f2_779 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1154 = _f2_779;
                float2 updated_pair_1155 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1153, acc_pair_1154);
                acc32[36] = updated_pair_1155.x;
                acc32[37] = updated_pair_1155.y;
                float2 _f2_780 = make_float2(_tmem_load_37[2], _tmem_load_37[3]);
                float2 cached_pair_1156 = _f2_780;
                float2 _f2_781 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1157 = _f2_781;
                float2 updated_pair_1158 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1156, acc_pair_1157);
                acc32[38] = updated_pair_1158.x;
                acc32[39] = updated_pair_1158.y;
                float2 _f2_782 = make_float2(_tmem_load_37[4], _tmem_load_37[5]);
                float2 cached_pair_1159 = _f2_782;
                float2 _f2_783 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1160 = _f2_783;
                float2 updated_pair_1161 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1159, acc_pair_1160);
                acc32[40] = updated_pair_1161.x;
                acc32[41] = updated_pair_1161.y;
                float2 _f2_784 = make_float2(_tmem_load_37[6], _tmem_load_37[7]);
                float2 cached_pair_1162 = _f2_784;
                float2 _f2_785 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1163 = _f2_785;
                float2 updated_pair_1164 = fma_f32x2_rn_ftz(weight_pair_1152, cached_pair_1162, acc_pair_1163);
                acc32[42] = updated_pair_1164.x;
                acc32[43] = updated_pair_1164.y;
                float2 _f2_786 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1165 = _f2_786;
                float2 _f2_787 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1166 = _f2_787;
                float2 scaled_pair_1167;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1167) : "l"(*(const unsigned long long*)&acc_pair_1166), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[44] = scaled_pair_1167.x;
                acc32[45] = scaled_pair_1167.y;
                float2 _f2_788 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1168 = _f2_788;
                float2 scaled_pair_1169;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1169) : "l"(*(const unsigned long long*)&acc_pair_1168), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[46] = scaled_pair_1169.x;
                acc32[47] = scaled_pair_1169.y;
                float2 _f2_789 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1170 = _f2_789;
                float2 scaled_pair_1171;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1171) : "l"(*(const unsigned long long*)&acc_pair_1170), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[48] = scaled_pair_1171.x;
                acc32[49] = scaled_pair_1171.y;
                float2 _f2_790 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1172 = _f2_790;
                float2 scaled_pair_1173;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1173) : "l"(*(const unsigned long long*)&acc_pair_1172), "l"(*(const unsigned long long*)&correction_pair_1165));
                acc32[50] = scaled_pair_1173.x;
                acc32[51] = scaled_pair_1173.y;
                float _tmem_load_38[8];
                tmem_ld_x8(&_tmem_load_38[0], tmem_base + 48);
                float2 _f2_791 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1174 = _f2_791;
                float2 _f2_792 = make_float2(_tmem_load_38[0], _tmem_load_38[1]);
                float2 cached_pair_1175 = _f2_792;
                float2 _f2_793 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1176 = _f2_793;
                float2 updated_pair_1177 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1175, acc_pair_1176);
                acc32[44] = updated_pair_1177.x;
                acc32[45] = updated_pair_1177.y;
                float2 _f2_794 = make_float2(_tmem_load_38[2], _tmem_load_38[3]);
                float2 cached_pair_1178 = _f2_794;
                float2 _f2_795 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1179 = _f2_795;
                float2 updated_pair_1180 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1178, acc_pair_1179);
                acc32[46] = updated_pair_1180.x;
                acc32[47] = updated_pair_1180.y;
                float2 _f2_796 = make_float2(_tmem_load_38[4], _tmem_load_38[5]);
                float2 cached_pair_1181 = _f2_796;
                float2 _f2_797 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1182 = _f2_797;
                float2 updated_pair_1183 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1181, acc_pair_1182);
                acc32[48] = updated_pair_1183.x;
                acc32[49] = updated_pair_1183.y;
                float2 _f2_798 = make_float2(_tmem_load_38[6], _tmem_load_38[7]);
                float2 cached_pair_1184 = _f2_798;
                float2 _f2_799 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1185 = _f2_799;
                float2 updated_pair_1186 = fma_f32x2_rn_ftz(weight_pair_1174, cached_pair_1184, acc_pair_1185);
                acc32[50] = updated_pair_1186.x;
                acc32[51] = updated_pair_1186.y;
                float2 _f2_800 = make_float2(correction_1040, correction_1040);
                float2 correction_pair_1187 = _f2_800;
                float2 _f2_801 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1188 = _f2_801;
                float2 scaled_pair_1189;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1189) : "l"(*(const unsigned long long*)&acc_pair_1188), "l"(*(const unsigned long long*)&correction_pair_1187));
                acc32[52] = scaled_pair_1189.x;
                acc32[53] = scaled_pair_1189.y;
                float2 _f2_802 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1190 = _f2_802;
                float2 scaled_pair_1191;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1191) : "l"(*(const unsigned long long*)&acc_pair_1190), "l"(*(const unsigned long long*)&correction_pair_1187));
                acc32[54] = scaled_pair_1191.x;
                acc32[55] = scaled_pair_1191.y;
                float _tmem_load_39[4];
                tmem_ld_x4(&_tmem_load_39[0], tmem_base + 56);
                float2 _f2_803 = make_float2(weights_1038[0], weights_1038[0]);
                float2 weight_pair_1192 = _f2_803;
                float2 _f2_804 = make_float2(_tmem_load_39[0], _tmem_load_39[1]);
                float2 cached_pair_1193 = _f2_804;
                float2 _f2_805 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1194 = _f2_805;
                float2 updated_pair_1195 = fma_f32x2_rn_ftz(weight_pair_1192, cached_pair_1193, acc_pair_1194);
                acc32[52] = updated_pair_1195.x;
                acc32[53] = updated_pair_1195.y;
                float2 _f2_806 = make_float2(_tmem_load_39[2], _tmem_load_39[3]);
                float2 cached_pair_1196 = _f2_806;
                float2 _f2_807 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1197 = _f2_807;
                float2 updated_pair_1198 = fma_f32x2_rn_ftz(weight_pair_1192, cached_pair_1196, acc_pair_1197);
                acc32[54] = updated_pair_1198.x;
                acc32[55] = updated_pair_1198.y;
                s_running = __fmaf_rn(s_running, correction_1040, weight_sum_1041);
                m_running = new_max_1039;
                consumer_stage += 1;
                if (consumer_stage == 2) { consumer_stage = 0; consumer_phase ^= 1; }
                if (token_1 == first_token) {
                    mbarrier_wait(norm_ready_addr, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                float2 _f2_808 = make_float2(0.0f, 0.0f);
                float2 local_output_sum_pair = _f2_808;
                float2 _f2_809 = make_float2(acc32[0], acc32[1]);
                float2 acc_pair_1199 = _f2_809;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1199, acc_pair_1199, local_output_sum_pair);
                float2 _f2_810 = make_float2(acc32[2], acc32[3]);
                float2 acc_pair_1200 = _f2_810;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1200, acc_pair_1200, local_output_sum_pair);
                float2 _f2_811 = make_float2(acc32[4], acc32[5]);
                float2 acc_pair_1201 = _f2_811;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1201, acc_pair_1201, local_output_sum_pair);
                float2 _f2_812 = make_float2(acc32[6], acc32[7]);
                float2 acc_pair_1202 = _f2_812;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1202, acc_pair_1202, local_output_sum_pair);
                float2 _f2_813 = make_float2(acc32[8], acc32[9]);
                float2 acc_pair_1203 = _f2_813;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1203, acc_pair_1203, local_output_sum_pair);
                float2 _f2_814 = make_float2(acc32[10], acc32[11]);
                float2 acc_pair_1204 = _f2_814;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1204, acc_pair_1204, local_output_sum_pair);
                float2 _f2_815 = make_float2(acc32[12], acc32[13]);
                float2 acc_pair_1205 = _f2_815;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1205, acc_pair_1205, local_output_sum_pair);
                float2 _f2_816 = make_float2(acc32[14], acc32[15]);
                float2 acc_pair_1206 = _f2_816;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1206, acc_pair_1206, local_output_sum_pair);
                float2 _f2_817 = make_float2(acc32[16], acc32[17]);
                float2 acc_pair_1207 = _f2_817;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1207, acc_pair_1207, local_output_sum_pair);
                float2 _f2_818 = make_float2(acc32[18], acc32[19]);
                float2 acc_pair_1208 = _f2_818;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1208, acc_pair_1208, local_output_sum_pair);
                float2 _f2_819 = make_float2(acc32[20], acc32[21]);
                float2 acc_pair_1209 = _f2_819;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1209, acc_pair_1209, local_output_sum_pair);
                float2 _f2_820 = make_float2(acc32[22], acc32[23]);
                float2 acc_pair_1210 = _f2_820;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1210, acc_pair_1210, local_output_sum_pair);
                float2 _f2_821 = make_float2(acc32[24], acc32[25]);
                float2 acc_pair_1211 = _f2_821;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1211, acc_pair_1211, local_output_sum_pair);
                float2 _f2_822 = make_float2(acc32[26], acc32[27]);
                float2 acc_pair_1212 = _f2_822;
                local_output_sum_pair = fma_f32x2_rn_ftz(acc_pair_1212, acc_pair_1212, local_output_sum_pair);
                float local_output_sum_sq = local_output_sum_pair.x + local_output_sum_pair.y;
                float _warp_reduce_0 = local_output_sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                local_output_sum_sq = _warp_reduce_0;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)warp_in_group] = local_output_sum_sq;
                }
                float2 _f2_823 = make_float2(0.0f, 0.0f);
                float2 local_output_sum_pair_1213 = _f2_823;
                float2 _f2_824 = make_float2(acc32[28], acc32[29]);
                float2 acc_pair_1214 = _f2_824;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1214, acc_pair_1214, local_output_sum_pair_1213);
                float2 _f2_825 = make_float2(acc32[30], acc32[31]);
                float2 acc_pair_1215 = _f2_825;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1215, acc_pair_1215, local_output_sum_pair_1213);
                float2 _f2_826 = make_float2(acc32[32], acc32[33]);
                float2 acc_pair_1216 = _f2_826;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1216, acc_pair_1216, local_output_sum_pair_1213);
                float2 _f2_827 = make_float2(acc32[34], acc32[35]);
                float2 acc_pair_1217 = _f2_827;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1217, acc_pair_1217, local_output_sum_pair_1213);
                float2 _f2_828 = make_float2(acc32[36], acc32[37]);
                float2 acc_pair_1218 = _f2_828;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1218, acc_pair_1218, local_output_sum_pair_1213);
                float2 _f2_829 = make_float2(acc32[38], acc32[39]);
                float2 acc_pair_1219 = _f2_829;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1219, acc_pair_1219, local_output_sum_pair_1213);
                float2 _f2_830 = make_float2(acc32[40], acc32[41]);
                float2 acc_pair_1220 = _f2_830;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1220, acc_pair_1220, local_output_sum_pair_1213);
                float2 _f2_831 = make_float2(acc32[42], acc32[43]);
                float2 acc_pair_1221 = _f2_831;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1221, acc_pair_1221, local_output_sum_pair_1213);
                float2 _f2_832 = make_float2(acc32[44], acc32[45]);
                float2 acc_pair_1222 = _f2_832;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1222, acc_pair_1222, local_output_sum_pair_1213);
                float2 _f2_833 = make_float2(acc32[46], acc32[47]);
                float2 acc_pair_1223 = _f2_833;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1223, acc_pair_1223, local_output_sum_pair_1213);
                float2 _f2_834 = make_float2(acc32[48], acc32[49]);
                float2 acc_pair_1224 = _f2_834;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1224, acc_pair_1224, local_output_sum_pair_1213);
                float2 _f2_835 = make_float2(acc32[50], acc32[51]);
                float2 acc_pair_1225 = _f2_835;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1225, acc_pair_1225, local_output_sum_pair_1213);
                float2 _f2_836 = make_float2(acc32[52], acc32[53]);
                float2 acc_pair_1226 = _f2_836;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1226, acc_pair_1226, local_output_sum_pair_1213);
                float2 _f2_837 = make_float2(acc32[54], acc32[55]);
                float2 acc_pair_1227 = _f2_837;
                local_output_sum_pair_1213 = fma_f32x2_rn_ftz(acc_pair_1227, acc_pair_1227, local_output_sum_pair_1213);
                float local_output_sum_sq_1228 = local_output_sum_pair_1213.x + local_output_sum_pair_1213.y;
                float _warp_reduce_1 = local_output_sum_sq_1228;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                local_output_sum_sq_1228 = _warp_reduce_1;
                if (lane == 0) {
                    output_stats[output_stats_base + (unsigned int)(4 + warp_in_group)] = local_output_sum_sq_1228;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 128;" :: "r"(8 + stream_1) : "memory");
                float total_output_sum_sq = ((lane < 8) ? output_stats[output_stats_base + (unsigned int)lane] : 0.0f);
                float _shfl_down_18 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 4, 8);
                total_output_sum_sq += _shfl_down_18;
                float _shfl_down_19 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 2, 8);
                total_output_sum_sq += _shfl_down_19;
                float _shfl_down_20 = __shfl_down_sync(0xFFFFFFFF, total_output_sum_sq, 1, 8);
                total_output_sum_sq += _shfl_down_20;
                float output_rsigma = 0.0f;
                if (lane == 0) {
                    float _rsqrt_3 = rsqrtf(__fmaf_rn(output_norm_eps * s_running, s_running, total_output_sum_sq / 7168.0f));
                    output_rsigma = _rsqrt_3;
                }
                float _shfl_6 = __shfl_sync(0xFFFFFFFF, output_rsigma, 0);
                output_rsigma = _shfl_6;
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
                float2 _f2_838 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair = _f2_838;
                float2 _f2_839 = make_float2(acc32[0], acc32[1]);
                float2 scaled_pair_1230;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1230) : "l"(*(const unsigned long long*)&_f2_839), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_840 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
                float2 normalized_pair;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair_1230), "l"(*(const unsigned long long*)&_f2_840));
                acc32[0] = normalized_pair.x;
                acc32[1] = normalized_pair.y;
                float2 _f2_841 = make_float2(acc32[2], acc32[3]);
                float2 scaled_pair_1231;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1231) : "l"(*(const unsigned long long*)&_f2_841), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_842 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
                float2 normalized_pair_1232;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1232) : "l"(*(const unsigned long long*)&scaled_pair_1231), "l"(*(const unsigned long long*)&_f2_842));
                acc32[2] = normalized_pair_1232.x;
                acc32[3] = normalized_pair_1232.y;
                float2 _f2_843 = make_float2(acc32[4], acc32[5]);
                float2 scaled_pair_1233;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1233) : "l"(*(const unsigned long long*)&_f2_843), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_844 = make_float2(_unpack_bf16_32[4], _unpack_bf16_32[5]);
                float2 normalized_pair_1234;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1234) : "l"(*(const unsigned long long*)&scaled_pair_1233), "l"(*(const unsigned long long*)&_f2_844));
                acc32[4] = normalized_pair_1234.x;
                acc32[5] = normalized_pair_1234.y;
                float2 _f2_845 = make_float2(acc32[6], acc32[7]);
                float2 scaled_pair_1235;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1235) : "l"(*(const unsigned long long*)&_f2_845), "l"(*(const unsigned long long*)&output_rsigma_pair));
                float2 _f2_846 = make_float2(_unpack_bf16_32[6], _unpack_bf16_32[7]);
                float2 normalized_pair_1236;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1236) : "l"(*(const unsigned long long*)&scaled_pair_1235), "l"(*(const unsigned long long*)&_f2_846));
                acc32[6] = normalized_pair_1236.x;
                acc32[7] = normalized_pair_1236.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[0 + 0], acc32[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[0 + 2], acc32[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[0 + 4], acc32[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[0 + 6], acc32[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1229 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_847 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1239 = _f2_847;
                float2 _f2_848 = make_float2(acc32[8], acc32[9]);
                float2 scaled_pair_1240;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1240) : "l"(*(const unsigned long long*)&_f2_848), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f2_849 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
                float2 normalized_pair_1241;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1241) : "l"(*(const unsigned long long*)&scaled_pair_1240), "l"(*(const unsigned long long*)&_f2_849));
                acc32[8] = normalized_pair_1241.x;
                acc32[9] = normalized_pair_1241.y;
                float2 _f2_850 = make_float2(acc32[10], acc32[11]);
                float2 scaled_pair_1242;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1242) : "l"(*(const unsigned long long*)&_f2_850), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f2_851 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
                float2 normalized_pair_1243;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1243) : "l"(*(const unsigned long long*)&scaled_pair_1242), "l"(*(const unsigned long long*)&_f2_851));
                acc32[10] = normalized_pair_1243.x;
                acc32[11] = normalized_pair_1243.y;
                float2 _f2_852 = make_float2(acc32[12], acc32[13]);
                float2 scaled_pair_1244;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1244) : "l"(*(const unsigned long long*)&_f2_852), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f2_853 = make_float2(_unpack_bf16_33[4], _unpack_bf16_33[5]);
                float2 normalized_pair_1245;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1245) : "l"(*(const unsigned long long*)&scaled_pair_1244), "l"(*(const unsigned long long*)&_f2_853));
                acc32[12] = normalized_pair_1245.x;
                acc32[13] = normalized_pair_1245.y;
                float2 _f2_854 = make_float2(acc32[14], acc32[15]);
                float2 scaled_pair_1246;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1246) : "l"(*(const unsigned long long*)&_f2_854), "l"(*(const unsigned long long*)&output_rsigma_pair_1239));
                float2 _f2_855 = make_float2(_unpack_bf16_33[6], _unpack_bf16_33[7]);
                float2 normalized_pair_1247;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1247) : "l"(*(const unsigned long long*)&scaled_pair_1246), "l"(*(const unsigned long long*)&_f2_855));
                acc32[14] = normalized_pair_1247.x;
                acc32[15] = normalized_pair_1247.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[8 + 0], acc32[8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[8 + 2], acc32[8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[8 + 4], acc32[8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[8 + 6], acc32[8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1237 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_856 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1250 = _f2_856;
                float2 _f2_857 = make_float2(acc32[16], acc32[17]);
                float2 scaled_pair_1251;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1251) : "l"(*(const unsigned long long*)&_f2_857), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f2_858 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
                float2 normalized_pair_1252;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1252) : "l"(*(const unsigned long long*)&scaled_pair_1251), "l"(*(const unsigned long long*)&_f2_858));
                acc32[16] = normalized_pair_1252.x;
                acc32[17] = normalized_pair_1252.y;
                float2 _f2_859 = make_float2(acc32[18], acc32[19]);
                float2 scaled_pair_1253;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1253) : "l"(*(const unsigned long long*)&_f2_859), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f2_860 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
                float2 normalized_pair_1254;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1254) : "l"(*(const unsigned long long*)&scaled_pair_1253), "l"(*(const unsigned long long*)&_f2_860));
                acc32[18] = normalized_pair_1254.x;
                acc32[19] = normalized_pair_1254.y;
                float2 _f2_861 = make_float2(acc32[20], acc32[21]);
                float2 scaled_pair_1255;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1255) : "l"(*(const unsigned long long*)&_f2_861), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f2_862 = make_float2(_unpack_bf16_34[4], _unpack_bf16_34[5]);
                float2 normalized_pair_1256;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1256) : "l"(*(const unsigned long long*)&scaled_pair_1255), "l"(*(const unsigned long long*)&_f2_862));
                acc32[20] = normalized_pair_1256.x;
                acc32[21] = normalized_pair_1256.y;
                float2 _f2_863 = make_float2(acc32[22], acc32[23]);
                float2 scaled_pair_1257;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1257) : "l"(*(const unsigned long long*)&_f2_863), "l"(*(const unsigned long long*)&output_rsigma_pair_1250));
                float2 _f2_864 = make_float2(_unpack_bf16_34[6], _unpack_bf16_34[7]);
                float2 normalized_pair_1258;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1258) : "l"(*(const unsigned long long*)&scaled_pair_1257), "l"(*(const unsigned long long*)&_f2_864));
                acc32[22] = normalized_pair_1258.x;
                acc32[23] = normalized_pair_1258.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[16 + 0], acc32[16 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[16 + 2], acc32[16 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[16 + 4], acc32[16 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[16 + 6], acc32[16 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1248 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_865 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1261 = _f2_865;
                float2 _f2_866 = make_float2(acc32[24], acc32[25]);
                float2 scaled_pair_1262;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1262) : "l"(*(const unsigned long long*)&_f2_866), "l"(*(const unsigned long long*)&output_rsigma_pair_1261));
                float2 _f2_867 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
                float2 normalized_pair_1263;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1263) : "l"(*(const unsigned long long*)&scaled_pair_1262), "l"(*(const unsigned long long*)&_f2_867));
                acc32[24] = normalized_pair_1263.x;
                acc32[25] = normalized_pair_1263.y;
                float2 _f2_868 = make_float2(acc32[26], acc32[27]);
                float2 scaled_pair_1264;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1264) : "l"(*(const unsigned long long*)&_f2_868), "l"(*(const unsigned long long*)&output_rsigma_pair_1261));
                float2 _f2_869 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
                float2 normalized_pair_1265;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1265) : "l"(*(const unsigned long long*)&scaled_pair_1264), "l"(*(const unsigned long long*)&_f2_869));
                acc32[26] = normalized_pair_1265.x;
                acc32[27] = normalized_pair_1265.y;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(acc32[24 + 0], acc32[24 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[24 + 2], acc32[24 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1259]) = _pk2;
                }
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
                float2 _f2_870 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1268 = _f2_870;
                float2 _f2_871 = make_float2(acc32[28], acc32[29]);
                float2 scaled_pair_1269;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1269) : "l"(*(const unsigned long long*)&_f2_871), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f2_872 = make_float2(_unpack_bf16_36[0], _unpack_bf16_36[1]);
                float2 normalized_pair_1270;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1270) : "l"(*(const unsigned long long*)&scaled_pair_1269), "l"(*(const unsigned long long*)&_f2_872));
                acc32[28] = normalized_pair_1270.x;
                acc32[29] = normalized_pair_1270.y;
                float2 _f2_873 = make_float2(acc32[30], acc32[31]);
                float2 scaled_pair_1271;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1271) : "l"(*(const unsigned long long*)&_f2_873), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f2_874 = make_float2(_unpack_bf16_36[2], _unpack_bf16_36[3]);
                float2 normalized_pair_1272;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1272) : "l"(*(const unsigned long long*)&scaled_pair_1271), "l"(*(const unsigned long long*)&_f2_874));
                acc32[30] = normalized_pair_1272.x;
                acc32[31] = normalized_pair_1272.y;
                float2 _f2_875 = make_float2(acc32[32], acc32[33]);
                float2 scaled_pair_1273;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1273) : "l"(*(const unsigned long long*)&_f2_875), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f2_876 = make_float2(_unpack_bf16_36[4], _unpack_bf16_36[5]);
                float2 normalized_pair_1274;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1274) : "l"(*(const unsigned long long*)&scaled_pair_1273), "l"(*(const unsigned long long*)&_f2_876));
                acc32[32] = normalized_pair_1274.x;
                acc32[33] = normalized_pair_1274.y;
                float2 _f2_877 = make_float2(acc32[34], acc32[35]);
                float2 scaled_pair_1275;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1275) : "l"(*(const unsigned long long*)&_f2_877), "l"(*(const unsigned long long*)&output_rsigma_pair_1268));
                float2 _f2_878 = make_float2(_unpack_bf16_36[6], _unpack_bf16_36[7]);
                float2 normalized_pair_1276;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1276) : "l"(*(const unsigned long long*)&scaled_pair_1275), "l"(*(const unsigned long long*)&_f2_878));
                acc32[34] = normalized_pair_1276.x;
                acc32[35] = normalized_pair_1276.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[28 + 0], acc32[28 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[28 + 2], acc32[28 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[28 + 4], acc32[28 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[28 + 6], acc32[28 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1266 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_879 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1279 = _f2_879;
                float2 _f2_880 = make_float2(acc32[36], acc32[37]);
                float2 scaled_pair_1280;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1280) : "l"(*(const unsigned long long*)&_f2_880), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f2_881 = make_float2(_unpack_bf16_37[0], _unpack_bf16_37[1]);
                float2 normalized_pair_1281;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1281) : "l"(*(const unsigned long long*)&scaled_pair_1280), "l"(*(const unsigned long long*)&_f2_881));
                acc32[36] = normalized_pair_1281.x;
                acc32[37] = normalized_pair_1281.y;
                float2 _f2_882 = make_float2(acc32[38], acc32[39]);
                float2 scaled_pair_1282;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1282) : "l"(*(const unsigned long long*)&_f2_882), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f2_883 = make_float2(_unpack_bf16_37[2], _unpack_bf16_37[3]);
                float2 normalized_pair_1283;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1283) : "l"(*(const unsigned long long*)&scaled_pair_1282), "l"(*(const unsigned long long*)&_f2_883));
                acc32[38] = normalized_pair_1283.x;
                acc32[39] = normalized_pair_1283.y;
                float2 _f2_884 = make_float2(acc32[40], acc32[41]);
                float2 scaled_pair_1284;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1284) : "l"(*(const unsigned long long*)&_f2_884), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f2_885 = make_float2(_unpack_bf16_37[4], _unpack_bf16_37[5]);
                float2 normalized_pair_1285;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1285) : "l"(*(const unsigned long long*)&scaled_pair_1284), "l"(*(const unsigned long long*)&_f2_885));
                acc32[40] = normalized_pair_1285.x;
                acc32[41] = normalized_pair_1285.y;
                float2 _f2_886 = make_float2(acc32[42], acc32[43]);
                float2 scaled_pair_1286;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1286) : "l"(*(const unsigned long long*)&_f2_886), "l"(*(const unsigned long long*)&output_rsigma_pair_1279));
                float2 _f2_887 = make_float2(_unpack_bf16_37[6], _unpack_bf16_37[7]);
                float2 normalized_pair_1287;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1287) : "l"(*(const unsigned long long*)&scaled_pair_1286), "l"(*(const unsigned long long*)&_f2_887));
                acc32[42] = normalized_pair_1287.x;
                acc32[43] = normalized_pair_1287.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[36 + 0], acc32[36 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[36 + 2], acc32[36 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[36 + 4], acc32[36 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[36 + 6], acc32[36 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1277 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_888 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1290 = _f2_888;
                float2 _f2_889 = make_float2(acc32[44], acc32[45]);
                float2 scaled_pair_1291;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1291) : "l"(*(const unsigned long long*)&_f2_889), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f2_890 = make_float2(_unpack_bf16_38[0], _unpack_bf16_38[1]);
                float2 normalized_pair_1292;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1292) : "l"(*(const unsigned long long*)&scaled_pair_1291), "l"(*(const unsigned long long*)&_f2_890));
                acc32[44] = normalized_pair_1292.x;
                acc32[45] = normalized_pair_1292.y;
                float2 _f2_891 = make_float2(acc32[46], acc32[47]);
                float2 scaled_pair_1293;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1293) : "l"(*(const unsigned long long*)&_f2_891), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f2_892 = make_float2(_unpack_bf16_38[2], _unpack_bf16_38[3]);
                float2 normalized_pair_1294;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1294) : "l"(*(const unsigned long long*)&scaled_pair_1293), "l"(*(const unsigned long long*)&_f2_892));
                acc32[46] = normalized_pair_1294.x;
                acc32[47] = normalized_pair_1294.y;
                float2 _f2_893 = make_float2(acc32[48], acc32[49]);
                float2 scaled_pair_1295;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1295) : "l"(*(const unsigned long long*)&_f2_893), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f2_894 = make_float2(_unpack_bf16_38[4], _unpack_bf16_38[5]);
                float2 normalized_pair_1296;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1296) : "l"(*(const unsigned long long*)&scaled_pair_1295), "l"(*(const unsigned long long*)&_f2_894));
                acc32[48] = normalized_pair_1296.x;
                acc32[49] = normalized_pair_1296.y;
                float2 _f2_895 = make_float2(acc32[50], acc32[51]);
                float2 scaled_pair_1297;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1297) : "l"(*(const unsigned long long*)&_f2_895), "l"(*(const unsigned long long*)&output_rsigma_pair_1290));
                float2 _f2_896 = make_float2(_unpack_bf16_38[6], _unpack_bf16_38[7]);
                float2 normalized_pair_1298;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1298) : "l"(*(const unsigned long long*)&scaled_pair_1297), "l"(*(const unsigned long long*)&_f2_896));
                acc32[50] = normalized_pair_1298.x;
                acc32[51] = normalized_pair_1298.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(acc32[44 + 0], acc32[44 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[44 + 2], acc32[44 + 3]);
                    _pk[2] = __floats2bfloat162_rn(acc32[44 + 4], acc32[44 + 5]);
                    _pk[3] = __floats2bfloat162_rn(acc32[44 + 6], acc32[44 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1288 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
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
                float2 _f2_897 = make_float2(output_rsigma, output_rsigma);
                float2 output_rsigma_pair_1301 = _f2_897;
                float2 _f2_898 = make_float2(acc32[52], acc32[53]);
                float2 scaled_pair_1302;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1302) : "l"(*(const unsigned long long*)&_f2_898), "l"(*(const unsigned long long*)&output_rsigma_pair_1301));
                float2 _f2_899 = make_float2(_unpack_bf16_39[0], _unpack_bf16_39[1]);
                float2 normalized_pair_1303;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1303) : "l"(*(const unsigned long long*)&scaled_pair_1302), "l"(*(const unsigned long long*)&_f2_899));
                acc32[52] = normalized_pair_1303.x;
                acc32[53] = normalized_pair_1303.y;
                float2 _f2_900 = make_float2(acc32[54], acc32[55]);
                float2 scaled_pair_1304;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1304) : "l"(*(const unsigned long long*)&_f2_900), "l"(*(const unsigned long long*)&output_rsigma_pair_1301));
                float2 _f2_901 = make_float2(_unpack_bf16_39[2], _unpack_bf16_39[3]);
                float2 normalized_pair_1305;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1305) : "l"(*(const unsigned long long*)&scaled_pair_1304), "l"(*(const unsigned long long*)&_f2_901));
                acc32[54] = normalized_pair_1305.x;
                acc32[55] = normalized_pair_1305.y;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(acc32[52 + 0], acc32[52 + 1]);
                    _pk[1] = __floats2bfloat162_rn(acc32[52 + 2], acc32[52 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + hidden_base_1299]) = _pk2;
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
