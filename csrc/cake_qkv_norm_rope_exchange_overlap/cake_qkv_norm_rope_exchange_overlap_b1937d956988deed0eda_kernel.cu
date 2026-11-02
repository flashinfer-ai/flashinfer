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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 6
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_STAGING_OFF 197632
#define SMEM_STAGING_STAGE_BYTES 34816
#define SMEM_STAGING_STRIDE 34816
#define SMEM_TOTAL 232448
#define THREADS 192

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ cos_sin, __nv_bfloat16* __restrict__ q_scale, __nv_bfloat16* __restrict__ k_scale, long long* __restrict__ tiles, long long* __restrict__ dst_ptrs, long long* __restrict__ ready_ptrs, unsigned int* __restrict__ counters, int num_tiles, int M, int K, int S_local, int head_stride, long long batch_stride, long long slot_elem_offset, int pad_base, int num_words, int rank, int world, unsigned int ready_target)
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

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 48)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_STAGING_OFF);
    const int staging_addr = smem + SMEM_STAGING_OFF;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // mma_done: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
            unsigned int load_stage = 0;
            unsigned int load_phase = 1;
            #pragma unroll 1
            for (int tile = bid; tile < num_tiles; tile += num_bids) {
                int desc = tile * 8;
                int mt = (int)tiles[desc];
                int n0 = (int)tiles[desc + 1];
                int off_m = mt * 128;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K / 64; iter_k++) {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(mma_done_addr + (load_stage) * 8, load_phase);
                    mbarrier_wait_token(mma_done_addr + (load_stage) * 8, load_phase, _mbar_token_0);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_a_addr + load_stage * 32768, (&A), 0, off_m, iter_k, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_b_addr + load_stage * 32768, (&B), 0, n0, iter_k, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 32768);
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; }
                    if (load_stage == 0) {
                        load_phase = load_phase ^ 1;
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int _tile = bid; _tile < num_tiles; _tile += num_bids) {
                mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K / 64; iter_k_1++) {
                    mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, ((init_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_tma_stage) * 8);
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 6) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (mma_epi_stage) * 8);
                mma_epi_stage += 1;
                if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            const int stg_base = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < num_tiles; tile_1 += num_bids) {
                int desc_1 = tile_1 * 8;
                int mt_1 = (int)tiles[desc_1];
                int dest = (int)tiles[desc_1 + 2];
                int norm_kind = (int)tiles[desc_1 + 3];
                long long base_elem = tiles[desc_1 + 4];
                long long row_stride = tiles[desc_1 + 5];
                int word = (int)tiles[desc_1 + 6];
                unsigned int last_ticket = (unsigned int)tiles[desc_1 + 7];
                int m = mt_1 * 128 + epi_tid;
                int b = m / S_local;
                int s = m - b * S_local;
                int cs_row = ((m < M) ? m : 0);
                long long cs_base = (long long)cs_row * 128;
                long long row_elem = slot_elem_offset + base_elem + (long long)b * batch_stride + (long long)s * row_stride;
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(dst_ptrs[dest]);
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 128;
                #pragma unroll
                for (int head_i = 0; head_i < 1; head_i++) {
                    int tmem_base = lane_addr + head_i * 128;
                    long long head_elem = row_elem + (long long)(head_i * head_stride);
                    if (norm_kind == 0) {
                        #pragma unroll
                        for (int chunk = 0; chunk < 16; chunk++) {
                            float _tmem_load_0[8];
                            tmem_ld_x8(&_tmem_load_0[0], tmem_base + chunk * 8);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            uint32_t _tmem_load_0_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                                _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(staging_addr + (unsigned int)(stg_base + epi_tid * 272 + chunk * 16)), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                        }
                    } else {
                        float partial[8];
                        float _tmem_load_1[8];
                        tmem_ld_x8(&_tmem_load_1[0], tmem_base);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            partial[j] = _tmem_load_1[j] * _tmem_load_1[j];
                        }
                        #pragma unroll
                        for (int chunk_1 = 1; chunk_1 < 16; chunk_1++) {
                            float _tmem_load_2[8];
                            tmem_ld_x8(&_tmem_load_2[0], tmem_base + chunk_1 * 8);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                partial[j_1] = partial[j_1] + _tmem_load_2[j_1] * _tmem_load_2[j_1];
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            partial[i] = partial[i] + partial[i + 4];
                        }
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 2; i_1++) {
                            partial[i_1] = partial[i_1] + partial[i_1 + 2];
                        }
                        float sum_sq = partial[0] + partial[1];
                        float _rsqrt_0 = rsqrtf(sum_sq * 0.0078125f + 1e-06f);
                        float rrms = _rsqrt_0;
                        #pragma unroll
                        for (int chunk_2 = 0; chunk_2 < 16; chunk_2++) {
                            float _tmem_load_3[8];
                            tmem_ld_x8(&_tmem_load_3[0], tmem_base + chunk_2 * 8);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((norm_kind == 1) ? q_scale + (chunk_2 * 8) : k_scale + (chunk_2 * 8)) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(cos_sin + (cs_base + (long long)(chunk_2 * 8 / 2)) + 0);
                                _vec_load_1[0 + 0] = _v4.x;
                                _vec_load_1[0 + 1] = _v4.y;
                                _vec_load_1[0 + 2] = _v4.z;
                                _vec_load_1[0 + 3] = _v4.w;
                            }
                            float _vec_load_2[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(cos_sin + (cs_base + (long long)(64 + chunk_2 * 8 / 2)) + 0);
                                _vec_load_2[0 + 0] = _v4.x;
                                _vec_load_2[0 + 1] = _v4.y;
                                _vec_load_2[0 + 2] = _v4.z;
                                _vec_load_2[0 + 3] = _v4.w;
                            }
                            float out[8];
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 4; j_2++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_3[2 * j_2] * rrms);
                                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                float x0 = _cvt_f32_0;
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_3[2 * j_2 + 1] * rrms);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                float x1 = _cvt_f32_1;
                                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(x0 * _vec_load_0[2 * j_2]);
                                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                float xs0 = _cvt_f32_2;
                                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(x1 * _vec_load_0[2 * j_2 + 1]);
                                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                float xs1 = _cvt_f32_3;
                                out[2 * j_2] = xs0 * _vec_load_1[j_2] - xs1 * _vec_load_2[j_2];
                                out[2 * j_2 + 1] = xs1 * _vec_load_1[j_2] + xs0 * _vec_load_2[j_2];
                            }
                            uint32_t out_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                                out_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(staging_addr + (unsigned int)(stg_base + epi_tid * 272 + chunk_2 * 8 * 2)), "r"(out_bf16[0]), "r"(out_bf16[1]), "r"(out_bf16[2]), "r"(out_bf16[3]) : "memory");
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                    int ro_chunk = epi_tid % 16;
                    int ro_lane_row = epi_tid / 16;
                    #pragma unroll
                    for (int ro_pass = 0; ro_pass < 16; ro_pass++) {
                        int ro_row = ro_pass * 8 + ro_lane_row;
                        int ro_smem = stg_base + ro_row * 272 + ro_chunk * 16;
                        unsigned int words[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                            : "r"(staging_addr + (unsigned int)ro_smem));
                        int ro_m = mt_1 * 128 + ro_row;
                        int ro_b = ro_m / S_local;
                        int ro_s = ro_m - ro_b * S_local;
                        long long ro_elem = slot_elem_offset + base_elem + (long long)ro_b * batch_stride + (long long)ro_s * row_stride + (long long)(head_i * head_stride + ro_chunk * 8);
                        if (ro_m < M) {
                            asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(dst + ro_elem), "r"((words)[0]), "r"((words)[1]), "r"((words)[2]), "r"((words)[3]) : "memory");
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 2) {
                    if (elect_sync()) {
                        uint32_t _atomic_inc_old_0;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_0) : "l"(&counters[dest * num_words + word]), "r"(static_cast<uint32_t>(last_ticket)) : "memory");
                        if (_atomic_inc_old_0 == last_ticket) {
                            unsigned int* signal = reinterpret_cast<unsigned int*>(ready_ptrs[dest]);
                            __threadfence_system();
                            asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(signal) + (pad_base + word))), "r"(static_cast<unsigned int>(ready_target)) : "memory");
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
