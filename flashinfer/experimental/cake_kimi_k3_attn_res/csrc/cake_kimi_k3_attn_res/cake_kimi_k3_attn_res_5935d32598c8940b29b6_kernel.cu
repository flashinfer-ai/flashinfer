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
#define TMEM_VALUES_TMEM_OFFSET 0
#define NUM_PIPE_STAGES 2
#define SMEM_SOURCE_BUF_OFF 0
#define SMEM_SOURCE_BUF_STAGE_BYTES 43008
#define SMEM_SOURCE_BUF_STRIDE 43008
#define SMEM_SOURCE_WORDS_OFF 0
#define SMEM_SOURCE_WORDS_STAGE_BYTES 86016
#define SMEM_SOURCE_WORDS_STRIDE 86016
#define SMEM_DELTA_BUF_OFF 86016
#define SMEM_DELTA_BUF_STAGE_BYTES 14336
#define SMEM_DELTA_BUF_STRIDE 14336
#define SMEM_DELTA_WORDS_OFF 86016
#define SMEM_DELTA_WORDS_STAGE_BYTES 28672
#define SMEM_DELTA_WORDS_STRIDE 28672
#define SMEM_OUTPUT_NORM_BUF_OFF 114688
#define SMEM_OUTPUT_NORM_BUF_STAGE_BYTES 14336
#define SMEM_OUTPUT_NORM_BUF_STRIDE 14336
#define SMEM_STATS_OFF 129024
#define SMEM_STATS_STAGE_BYTES 768
#define SMEM_STATS_STRIDE 768
#define SMEM_TOTAL 129840
#define THREADS 288

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



// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra DONE_RELAXED;\n\t"
        "bra LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
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




__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
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

__global__ __launch_bounds__(288, 1) void
kernel_cake_kimi_k3_attn_res_5935d32598c8940b29b6(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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

    const int mbar_base = smem + 129792;
    #define ready_addr (mbar_base + 0)
    #define consumed_addr (mbar_base + 16)
    #define norm_ready_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* source_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int source_buf_addr = smem + 0;
    unsigned int* source_words = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int source_words_addr = smem + 0;
    __nv_bfloat16* delta_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 86016);
    const int delta_buf_addr = smem + 86016;
    unsigned int* delta_words = reinterpret_cast<unsigned int*>(smem_raw + 86016);
    const int delta_words_addr = smem + 86016;
    __nv_bfloat16* output_norm_buf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 114688);
    const int output_norm_buf_addr = smem + 114688;
    float* stats = reinterpret_cast<float*>(smem_raw + 129024);
    const int stats_addr = smem + 129024;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 5 barriers)
    // Mbarriers at smem_raw[129792..129832)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // ready: 2 barriers, init_count=1
            mbarrier_init(smem + 129792, 1);
            mbarrier_init(smem + 129800, 1);
            // consumed: 2 barriers, init_count=256
            mbarrier_init(smem + 129808, 256);
            mbarrier_init(smem + 129816, 256);
            // norm_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 129824, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 129832);
    if (warp == 1) {
        int _tmem_hold = smem + 129832;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_values_tmem = taddr;

    // ---- Role: producer ----
    if (warp == 0) {
        { // producer_main
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(norm_ready_addr, 14336);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %2, [%3];"
                    :: "r"(output_norm_buf_addr), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(output_norm_weight) + ((unsigned long long)0 * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(norm_ready_addr)
                    : "memory");
            }
            if (elect_sync()) {
                unsigned int stage = 0;
                unsigned int phase = 0;
                int pdl_waited = 0;
                #pragma unroll 1
                for (int token = bid; token < M; token += num_bids) {
                    #pragma unroll
                    for (int chunk = 0; chunk < 3; chunk++) {
                        mbarrier_wait_relaxed(consumed_addr + (stage) * 8, phase ^ 1);
                        mbarrier_arrive_expect_tx(ready_addr + (stage) * 8, (3 + ((chunk == 2) ? 1 : 0)) * 7168 * 2);
                        #pragma unroll
                        for (int row = 0; row < 3; row++) {
                            if (chunk * 3 + row == 8) {
                                asm volatile(
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                    "[%0], [%1], %2, [%3];"
                                    :: "r"(source_buf_addr + (stage * 3 + (unsigned int)row) * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(prefix) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage) * 8)
                                    : "memory");
                            } else {
                                asm volatile(
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                    "[%0], [%1], %2, [%3];"
                                    :: "r"(source_buf_addr + (stage * 3 + (unsigned int)row) * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(blocks) + ((unsigned long long)((unsigned long long)token * blocks_m_stride + (unsigned long long)(chunk * 3 + row) * blocks_k_stride) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage) * 8)
                                    : "memory");
                            }
                        }
                        if (chunk == 2) {
                            if (pdl_waited == 0) {
                                asm volatile("griddepcontrol.wait;" ::: "memory");
                                pdl_waited = 1;
                            }
                            asm volatile(
                                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                "[%0], [%1], %2, [%3];"
                                :: "r"(delta_buf_addr + stage * 7168 * 2), "l"(reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(delta) + ((unsigned long long)(token * 7168) * (unsigned long long)2))), "r"((uint32_t)(14336)), "r"(ready_addr + (stage) * 8)
                                : "memory");
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; phase ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("barrier.sync 0, 288;" ::: "memory");
        }
    // ---- Role: consumers ----
    } else if (warp >= 1 && warp <= 8) {
        { // consumers_main
            int warp_id_in_role = (warp - 1);
            int warp_0 = warp_id_in_role;
            int group = warp_0 / 4;
            int thread = (warp_0 * 32 + lane) % 128;
            int tbase = taddr + (unsigned int)(group * 96);
            float q[28];
            #pragma unroll
            for (int si = 0; si < 4; si++) {
                if (si == 3) {
                    int base = ((si == 3) ? 6144 + group * 512 + thread * 4 : (si * 2 + group) * 1024 + thread * 8);
                    float _vec_load_0[4];
                    {
                        uint2 _vld_0;
                        _vld_0 = *reinterpret_cast<const uint2*>(norm_weight + base);
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
                        _vld_1 = *reinterpret_cast<const uint2*>(qk_weight + base);
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
                    for (int j = 0; j < 4; j++) {
                        q[si * 8 + j] = _vec_load_0[j] * _vec_load_1[j];
                    }
                } else {
                    int base_1 = ((si == 3) ? 6144 + group * 512 + thread * 4 : (si * 2 + group) * 1024 + thread * 8);
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(norm_weight + base_1);
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
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(qk_weight + base_1);
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
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        q[si * 8 + j_1] = _vec_load_2[j_1] * _vec_load_3[j_1];
                    }
                }
            }
            unsigned int cstage = 0;
            unsigned int cphase = 0;
            #pragma unroll 1
            for (int token_1 = bid; token_1 < M; token_1 += num_bids) {
                float acc[28];
                #pragma unroll
                for (int j_2 = 0; j_2 < 28; j_2++) {
                    acc[j_2] = 0.0f;
                }
                float max_running = -3.4028234663852886e+38f;
                float sum_running = 0.0f;
                #pragma unroll
                for (int chunk_1 = 0; chunk_1 < 3; chunk_1++) {
                    mbarrier_wait_relaxed(ready_addr + (cstage) * 8, cphase);
                    float2 sq[3];
                    float2 dot[3];
                    #pragma unroll
                    for (int row_1 = 0; row_1 < 3; row_1++) {
                        float2 _f2_0 = make_float2(0.0f, 0.0f);
                        sq[row_1] = _f2_0;
                        float2 _f2_1 = make_float2(0.0f, 0.0f);
                        dot[row_1] = _f2_1;
                    }
                    #pragma unroll
                    for (int si_1 = 0; si_1 < 4; si_1++) {
                        if (si_1 == 3) {
                            int base_2 = ((si_1 == 3) ? 6144 + group * 512 + thread * 4 : (si_1 * 2 + group) * 1024 + thread * 8);
                            #pragma unroll
                            for (int row_2 = 0; row_2 < 3; row_2++) {
                                unsigned int _source_words_reg_0[2];
                                {
                                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(source_words);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 2; _lr++)
                                        _source_words_reg_0[_lr] = _smem_ptr[((cstage * 3 + (unsigned int)row_2) * 3584 + (unsigned int)(base_2 / 2)) + _lr];
                                }
                                unsigned int words[2];
                                #pragma unroll
                                for (int j_3 = 0; j_3 < 2; j_3++) {
                                    words[j_3] = _source_words_reg_0[j_3];
                                }
                                if (chunk_1 * 3 + row_2 == 8) {
                                    #pragma unroll
                                    for (int j_4 = 0; j_4 < 2; j_4++) {
                                        __nv_bfloat162 a = __as_bf16x2(words[j_4]);
                                        __nv_bfloat162 d = __as_bf16x2(delta_words[cstage * 3584 + (unsigned int)(base_2 / 2) + (unsigned int)j_4]);
                                        __nv_bfloat162 mixed = a + d;
                                        words[j_4] = __as_u32(mixed);
                                    }
                                    {
                                        int2 _iv2 = make_int2(words[0 + 0], words[0 + 1]);
                                        *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (token_1 * 7168 + base_2)) + 0) = _iv2;
                                    }
                                }
                                float words_f32[4];
                                #pragma unroll
                                for (int _pair = 0; _pair < 2; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&words_f32[_pair * 2])[0]), "=f"((&words_f32[_pair * 2])[1])
                                        : "r"(words[_pair]));
                                }
                                tmem_st_x4_f32(tbase + (si_1 * 3 + row_2) * 8, words_f32);
                                if (si_1 == 3) {
                                    #pragma unroll
                                    for (int j_5 = 0; j_5 < 2; j_5++) {
                                        float2 _f2_2 = make_float2(words_f32[j_5 * 2], words_f32[j_5 * 2 + 1]);
                                        float2 v = _f2_2;
                                        sq[row_2] = fma_f32x2_rn_noftz(v, v, sq[row_2]);
                                    }
                                    #pragma unroll
                                    for (int j_6 = 0; j_6 < 2; j_6++) {
                                        float2 _f2_3 = make_float2(words_f32[j_6 * 2], words_f32[j_6 * 2 + 1]);
                                        float2 v_1 = _f2_3;
                                        float2 _f2_4 = make_float2(q[si_1 * 8 + j_6 * 2], q[si_1 * 8 + j_6 * 2 + 1]);
                                        float2 qp = _f2_4;
                                        dot[row_2] = fma_f32x2_rn_noftz(v_1, qp, dot[row_2]);
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 4; j_7++) {
                                        float2 _f2_5 = make_float2(words_f32[j_7 * 2], words_f32[j_7 * 2 + 1]);
                                        float2 v_2 = _f2_5;
                                        float2 _f2_6 = make_float2(q[si_1 * 8 + j_7 * 2], q[si_1 * 8 + j_7 * 2 + 1]);
                                        float2 qp_1 = _f2_6;
                                        sq[row_2] = fma_f32x2_rn_noftz(v_2, v_2, sq[row_2]);
                                        dot[row_2] = fma_f32x2_rn_noftz(v_2, qp_1, dot[row_2]);
                                    }
                                }
                            }
                        } else {
                            int base_3 = ((si_1 == 3) ? 6144 + group * 512 + thread * 4 : (si_1 * 2 + group) * 1024 + thread * 8);
                            #pragma unroll
                            for (int row_3 = 0; row_3 < 3; row_3++) {
                                unsigned int _source_words_reg_1[4];
                                {
                                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(source_words);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 4; _lr++)
                                        _source_words_reg_1[_lr] = _smem_ptr[((cstage * 3 + (unsigned int)row_3) * 3584 + (unsigned int)(base_3 / 2)) + _lr];
                                }
                                unsigned int words_1[4];
                                #pragma unroll
                                for (int j_8 = 0; j_8 < 4; j_8++) {
                                    words_1[j_8] = _source_words_reg_1[j_8];
                                }
                                if (chunk_1 * 3 + row_3 == 8) {
                                    #pragma unroll
                                    for (int j_9 = 0; j_9 < 4; j_9++) {
                                        __nv_bfloat162 a_1 = __as_bf16x2(words_1[j_9]);
                                        __nv_bfloat162 d_1 = __as_bf16x2(delta_words[cstage * 3584 + (unsigned int)(base_3 / 2) + (unsigned int)j_9]);
                                        __nv_bfloat162 mixed_1 = a_1 + d_1;
                                        words_1[j_9] = __as_u32(mixed_1);
                                    }
                                    {
                                        int4 _iv4 = make_int4(words_1[0 + 0], words_1[0 + 1], words_1[0 + 2], words_1[0 + 3]);
                                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (token_1 * 7168 + base_3)) + 0) = _iv4;
                                    }
                                }
                                float words_f32_1[8];
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&words_f32_1[_pair * 2])[0]), "=f"((&words_f32_1[_pair * 2])[1])
                                        : "r"(words_1[_pair]));
                                }
                                tmem_st_x8_f32(tbase + (si_1 * 3 + row_3) * 8, words_f32_1);
                                if (si_1 == 3) {
                                    #pragma unroll
                                    for (int j_10 = 0; j_10 < 2; j_10++) {
                                        float2 _f2_7 = make_float2(words_f32_1[j_10 * 2], words_f32_1[j_10 * 2 + 1]);
                                        float2 v_3 = _f2_7;
                                        sq[row_3] = fma_f32x2_rn_noftz(v_3, v_3, sq[row_3]);
                                    }
                                    #pragma unroll
                                    for (int j_11 = 0; j_11 < 2; j_11++) {
                                        float2 _f2_8 = make_float2(words_f32_1[j_11 * 2], words_f32_1[j_11 * 2 + 1]);
                                        float2 v_4 = _f2_8;
                                        float2 _f2_9 = make_float2(q[si_1 * 8 + j_11 * 2], q[si_1 * 8 + j_11 * 2 + 1]);
                                        float2 qp_2 = _f2_9;
                                        dot[row_3] = fma_f32x2_rn_noftz(v_4, qp_2, dot[row_3]);
                                    }
                                } else {
                                    #pragma unroll
                                    for (int j_12 = 0; j_12 < 4; j_12++) {
                                        float2 _f2_10 = make_float2(words_f32_1[j_12 * 2], words_f32_1[j_12 * 2 + 1]);
                                        float2 v_5 = _f2_10;
                                        float2 _f2_11 = make_float2(q[si_1 * 8 + j_12 * 2], q[si_1 * 8 + j_12 * 2 + 1]);
                                        float2 qp_3 = _f2_11;
                                        sq[row_3] = fma_f32x2_rn_noftz(v_5, v_5, sq[row_3]);
                                        dot[row_3] = fma_f32x2_rn_noftz(v_5, qp_3, dot[row_3]);
                                    }
                                }
                            }
                        }
                    }
                    mbarrier_arrive(consumed_addr + (cstage) * 8);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    float2 pairs[3];
                    #pragma unroll
                    for (int row_4 = 0; row_4 < 3; row_4++) {
                        float2 _f2_12 = make_float2(sq[row_4].x + sq[row_4].y, dot[row_4].x + dot[row_4].y);
                        pairs[row_4] = _f2_12;
                    }
                    #pragma unroll
                    for (int row_5 = 0; row_5 < 3; row_5++) {
                        unsigned long long bits = 0;
                        bits = reinterpret_cast<unsigned long long*>(&pairs[row_5])[0];
                        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
                        unsigned long long peerbits = _shfl_xor_0;
                        float2 _f2_13 = make_float2(0.0f, 0.0f);
                        float2 peer = _f2_13;
                        peer = reinterpret_cast<float2*>(&peerbits)[0];
                        pairs[row_5] = add_f32x2_noftz(pairs[row_5], peer);
                    }
                    #pragma unroll
                    for (int row_6 = 0; row_6 < 3; row_6++) {
                        unsigned long long bits_1 = 0;
                        bits_1 = reinterpret_cast<unsigned long long*>(&pairs[row_6])[0];
                        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_1, 8);
                        unsigned long long peerbits_1 = _shfl_xor_1;
                        float2 _f2_14 = make_float2(0.0f, 0.0f);
                        float2 peer_1 = _f2_14;
                        peer_1 = reinterpret_cast<float2*>(&peerbits_1)[0];
                        pairs[row_6] = add_f32x2_noftz(pairs[row_6], peer_1);
                    }
                    #pragma unroll
                    for (int row_7 = 0; row_7 < 3; row_7++) {
                        unsigned long long bits_2 = 0;
                        bits_2 = reinterpret_cast<unsigned long long*>(&pairs[row_7])[0];
                        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_2, 4);
                        unsigned long long peerbits_2 = _shfl_xor_2;
                        float2 _f2_15 = make_float2(0.0f, 0.0f);
                        float2 peer_2 = _f2_15;
                        peer_2 = reinterpret_cast<float2*>(&peerbits_2)[0];
                        pairs[row_7] = add_f32x2_noftz(pairs[row_7], peer_2);
                    }
                    #pragma unroll
                    for (int row_8 = 0; row_8 < 3; row_8++) {
                        unsigned long long bits_3 = 0;
                        bits_3 = reinterpret_cast<unsigned long long*>(&pairs[row_8])[0];
                        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_3, 2);
                        unsigned long long peerbits_3 = _shfl_xor_3;
                        float2 _f2_16 = make_float2(0.0f, 0.0f);
                        float2 peer_3 = _f2_16;
                        peer_3 = reinterpret_cast<float2*>(&peerbits_3)[0];
                        pairs[row_8] = add_f32x2_noftz(pairs[row_8], peer_3);
                    }
                    #pragma unroll
                    for (int row_9 = 0; row_9 < 3; row_9++) {
                        unsigned long long bits_4 = 0;
                        bits_4 = reinterpret_cast<unsigned long long*>(&pairs[row_9])[0];
                        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_4, 1);
                        unsigned long long peerbits_4 = _shfl_xor_4;
                        float2 _f2_17 = make_float2(0.0f, 0.0f);
                        float2 peer_4 = _f2_17;
                        peer_4 = reinterpret_cast<float2*>(&peerbits_4)[0];
                        pairs[row_9] = add_f32x2_noftz(pairs[row_9], peer_4);
                    }
                    if (lane == 0) {
                        #pragma unroll
                        for (int row_10 = 0; row_10 < 3; row_10++) {
                            stats[chunk_1 * 8 * 3 * 2 + (warp_0 * 3 + row_10) * 2] = pairs[row_10].x;
                            stats[chunk_1 * 8 * 3 * 2 + (warp_0 * 3 + row_10) * 2 + 1] = pairs[row_10].y;
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    int stat_n = lane / 8;
                    int stat_w = lane % 8;
                    float total_sq = 0.0f;
                    float total_dot = 0.0f;
                    if (stat_n < 3) {
                        total_sq = stats[chunk_1 * 8 * 3 * 2 + (stat_w * 3 + stat_n) * 2];
                        total_dot = stats[chunk_1 * 8 * 3 * 2 + (stat_w * 3 + stat_n) * 2 + 1];
                    }
                    float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
                    total_sq += _shfl_down_0;
                    float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
                    total_dot += _shfl_down_1;
                    float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
                    total_sq += _shfl_down_2;
                    float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
                    total_dot += _shfl_down_3;
                    float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
                    total_sq += _shfl_down_4;
                    float _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
                    total_dot += _shfl_down_5;
                    float logit = 0.0f;
                    if (stat_n < 3 && stat_w == 0) {
                        float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
                        float sigma = _rsqrt_0;
                        logit = total_dot * sigma;
                    }
                    float logits[3];
                    #pragma unroll
                    for (int row_11 = 0; row_11 < 3; row_11++) {
                        float _shfl_0;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(logit), "r"(row_11 * 8));
                        logits[row_11] = _shfl_0;
                    }
                    float max_chunk = -3.4028234663852886e+38f;
                    #pragma unroll
                    for (int row_12 = 0; row_12 < 3; row_12++) {
                        float _fmax_0 = fmaxf(max_chunk, logits[row_12]);
                        max_chunk = _fmax_0;
                    }
                    float _fmax_1 = fmaxf(max_running, max_chunk);
                    float max_new = _fmax_1;
                    float _exp2_0 = approx_exp2((max_running - max_new) * 1.4426950408889634f);
                    float correction = _exp2_0;
                    float weights[3];
                    float sum_weights = 0.0f;
                    #pragma unroll
                    for (int row_13 = 0; row_13 < 3; row_13++) {
                        float _exp2_1 = approx_exp2((logits[row_13] - max_new) * 1.4426950408889634f);
                        weights[row_13] = _exp2_1;
                        sum_weights += weights[row_13];
                    }
                    #pragma unroll
                    for (int si_2 = 0; si_2 < 4; si_2++) {
                        if (si_2 == 3) {
                            float2 a_2[2];
                            float2 _f2_18 = make_float2(correction, correction);
                            float2 corr = _f2_18;
                            #pragma unroll
                            for (int j_13 = 0; j_13 < 2; j_13++) {
                                float2 _f2_19 = make_float2(acc[si_2 * 8 + j_13 * 2], acc[si_2 * 8 + j_13 * 2 + 1]);
                                float2 previous = _f2_19;
                                a_2[j_13] = mul_f32x2_noftz(previous, corr);
                            }
                            float cached[12];
                            #pragma unroll
                            for (int row_14 = 0; row_14 < 3; row_14++) {
                                float _tmem_load_0[4];
                                tmem_ld_x4(&_tmem_load_0[0], tbase + (si_2 * 3 + row_14) * 8);
                                #pragma unroll
                                for (int j_14 = 0; j_14 < 4; j_14++) {
                                    cached[row_14 * 4 + j_14] = _tmem_load_0[j_14];
                                }
                            }
                            #pragma unroll
                            for (int row_15 = 0; row_15 < 3; row_15++) {
                                float2 _f2_20 = make_float2(weights[row_15], weights[row_15]);
                                float2 weight = _f2_20;
                                #pragma unroll
                                for (int j_15 = 0; j_15 < 2; j_15++) {
                                    float2 _f2_21 = make_float2(cached[row_15 * 4 + j_15 * 2], cached[row_15 * 4 + j_15 * 2 + 1]);
                                    float2 v_6 = _f2_21;
                                    a_2[j_15] = fma_f32x2_rn_noftz(weight, v_6, a_2[j_15]);
                                }
                            }
                            #pragma unroll
                            for (int j_16 = 0; j_16 < 2; j_16++) {
                                acc[si_2 * 8 + j_16 * 2] = a_2[j_16].x;
                                acc[si_2 * 8 + j_16 * 2 + 1] = a_2[j_16].y;
                            }
                        } else {
                            float2 a_3[4];
                            float2 _f2_22 = make_float2(correction, correction);
                            float2 corr_1 = _f2_22;
                            #pragma unroll
                            for (int j_17 = 0; j_17 < 4; j_17++) {
                                float2 _f2_23 = make_float2(acc[si_2 * 8 + j_17 * 2], acc[si_2 * 8 + j_17 * 2 + 1]);
                                float2 previous_1 = _f2_23;
                                a_3[j_17] = mul_f32x2_noftz(previous_1, corr_1);
                            }
                            float cached_1[24];
                            #pragma unroll
                            for (int row_16 = 0; row_16 < 3; row_16++) {
                                float _tmem_load_1[8];
                                tmem_ld_x8(&_tmem_load_1[0], tbase + (si_2 * 3 + row_16) * 8);
                                #pragma unroll
                                for (int j_18 = 0; j_18 < 8; j_18++) {
                                    cached_1[row_16 * 8 + j_18] = _tmem_load_1[j_18];
                                }
                            }
                            #pragma unroll
                            for (int row_17 = 0; row_17 < 3; row_17++) {
                                float2 _f2_24 = make_float2(weights[row_17], weights[row_17]);
                                float2 weight_1 = _f2_24;
                                #pragma unroll
                                for (int j_19 = 0; j_19 < 4; j_19++) {
                                    float2 _f2_25 = make_float2(cached_1[row_17 * 8 + j_19 * 2], cached_1[row_17 * 8 + j_19 * 2 + 1]);
                                    float2 v_7 = _f2_25;
                                    a_3[j_19] = fma_f32x2_rn_noftz(weight_1, v_7, a_3[j_19]);
                                }
                            }
                            #pragma unroll
                            for (int j_20 = 0; j_20 < 4; j_20++) {
                                acc[si_2 * 8 + j_20 * 2] = a_3[j_20].x;
                                acc[si_2 * 8 + j_20 * 2 + 1] = a_3[j_20].y;
                            }
                        }
                    }
                    sum_running = sum_running * correction + sum_weights;
                    max_running = max_new;
                    cstage += 1;
                    if (cstage == 2) { cstage = 0; cphase ^= 1; }
                }
                float2 _f2_26 = make_float2(0.0f, 0.0f);
                float2 output_sq_pair = _f2_26;
                #pragma unroll
                for (int j_21 = 0; j_21 < 14; j_21++) {
                    float2 _f2_27 = make_float2(acc[j_21 * 2], acc[j_21 * 2 + 1]);
                    float2 v_8 = _f2_27;
                    output_sq_pair = fma_f32x2_rn_noftz(v_8, v_8, output_sq_pair);
                }
                if (token_1 == bid) {
                    mbarrier_wait_relaxed(norm_ready_addr, 0);
                }
                float output_sq = output_sq_pair.x + output_sq_pair.y;
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
                output_sq += _shfl_xor_5;
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
                output_sq += _shfl_xor_6;
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
                output_sq += _shfl_xor_7;
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
                output_sq += _shfl_xor_8;
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
                output_sq += _shfl_xor_9;
                if (lane == 0) {
                    stats[144 + warp_0 * 3 * 2] = output_sq;
                    stats[144 + warp_0 * 3 * 2 + 1] = 0.0f;
                }
                asm volatile("barrier.sync 8, 256;" ::: "memory");
                float output_total = ((lane < 8) ? stats[144 + lane * 3 * 2] : 0.0f);
                float _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
                output_total += _shfl_down_6;
                float _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
                output_total += _shfl_down_7;
                float _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
                output_total += _shfl_down_8;
                if (lane == 0) {
                    float _rsqrt_1 = rsqrtf(output_total / 7168.0f + output_norm_eps * sum_running * sum_running);
                    output_total = _rsqrt_1;
                }
                float _shfl_1;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(output_total), "r"(0));
                float rsigma = _shfl_1;
                #pragma unroll
                for (int si_3 = 0; si_3 < 4; si_3++) {
                    if (si_3 == 3) {
                        int base_4 = ((si_3 == 3) ? 6144 + group * 512 + thread * 4 : (si_3 * 2 + group) * 1024 + thread * 8);
                        float _output_norm_buf_reg_0[4];
                        {
                            uint32_t _smem_addr_4 = output_norm_buf_addr + (base_4) * 2;
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
                        float output_values[4];
                        #pragma unroll
                        for (int j_22 = 0; j_22 < 4; j_22++) {
                            float weight_2 = _output_norm_buf_reg_0[j_22];
                            output_values[j_22] = acc[si_3 * 8 + j_22] * rsigma * weight_2;
                        }
                        {
                            uint2 _pk2;
                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + base_4]) = _pk2;
                        }
                    } else {
                        int base_5 = ((si_3 == 3) ? 6144 + group * 512 + thread * 4 : (si_3 * 2 + group) * 1024 + thread * 8);
                        float _output_norm_buf_reg_1[8];
                        {
                            uint32_t _smem_addr_5 = output_norm_buf_addr + (base_5) * 2;
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
                        float output_values_1[8];
                        #pragma unroll
                        for (int j_23 = 0; j_23 < 8; j_23++) {
                            float weight_3 = _output_norm_buf_reg_1[j_23];
                            output_values_1[j_23] = acc[si_3 * 8 + j_23] * rsigma * weight_3;
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(output_values_1[0 + 0], output_values_1[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(output_values_1[0 + 2], output_values_1[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(output_values_1[0 + 4], output_values_1[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(output_values_1[0 + 6], output_values_1[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[token_1 * 7168 + base_5 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
            asm volatile("barrier.sync 0, 288;" ::: "memory");
            if (warp_0 == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }

    // Cleanup
}

} // extern "C"
