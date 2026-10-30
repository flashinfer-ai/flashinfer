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
#define TMEM_ACC_OFFSET 0
#define NUM_RING_STAGES 2
#define NUM_ACCS_STAGES 2
#define NUM_TK_PIPE_STAGES 2
#define SMEM_DO_S_OFF 1024
#define SMEM_DO_S_STAGE_BYTES 16384
#define SMEM_DO_S_STRIDE 24576
#define SMEM_AQK_S_OFF 17408
#define SMEM_AQK_S_STAGE_BYTES 8192
#define SMEM_AQK_S_STRIDE 24576
#define SMEM_DO_MN_OFF 1024
#define SMEM_DO_MN_STAGE_BYTES 16384
#define SMEM_DO_MN_STRIDE 24576
#define SMEM_AQK_MN_OFF 17408
#define SMEM_AQK_MN_STAGE_BYTES 8192
#define SMEM_AQK_MN_STRIDE 24576
#define SMEM_TK_S_OFF 50176
#define SMEM_TK_S_STAGE_BYTES 16
#define SMEM_TK_S_STRIDE 16
#define SMEM_TOTAL 50304
#define THREADS 192
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_3f3dd0a8504039fd93f0(const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap aqk_tma, __nv_bfloat16* __restrict__ do_ptr, float* __restrict__ g_cum, __nv_bfloat16* __restrict__ do_g, __nv_bfloat16* __restrict__ dv_intra, int num_heads, float scale, int do_tstride, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, unsigned int* __restrict__ ticket)
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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define acc_full_addr (mbar_base + 32)
    #define acc_empty_addr (mbar_base + 48)
    #define tk_full_addr (mbar_base + 64)
    #define tk_empty_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* do_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_s_addr = smem + 1024;
    __nv_bfloat16* aqk_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int aqk_s_addr = smem + 17408;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_mn_addr = smem + 1024;
    __nv_bfloat16* aqk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int aqk_mn_addr = smem + 17408;
    unsigned int* tk_s = reinterpret_cast<unsigned int*>(smem_raw + 50176);
    const int tk_s_addr = smem + 50176;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=129
            mbarrier_init(smem + 16, 129);
            mbarrier_init(smem + 24, 129);
            // --- pipeline 'accs' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            // --- pipeline 'tk_pipe' ---
            // tk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // tk_empty: 2 barriers, init_count=160
            mbarrier_init(smem + 80, 160);
            mbarrier_init(smem + 88, 160);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int num_items_c = seq_chunk_start[num_seqs] * num_heads;
            unsigned int epi_buf = 0;
            const int wq = warp;
            const int row_base = wq * 32 << 16;
            const int lane_half = lane >> 4;
            const int s_row = wq * 16 + (lane & 15);
            const int col_half = lane_half * 16;
            unsigned int do_stage = 0;
            const int t_do = wq * 16 + (lane & 15);
            const int v0 = lane_half * 64;
            const unsigned int do_row = (unsigned int)(lane_half * 8192 + t_do * 128);
            const int sw7 = lane & 7;
            const int rsel = lane >> 3 & 1;
            const int sw_l = lane & 7 ^ rsel;
            const unsigned int do_row_l = (unsigned int)(lane_half * 8192 + (wq * 16 + rsel) * 128);
            const int col_l = lane_half * 64 + (lane & 7) * 8;
            const long long nh64 = (long long)num_heads;
            float zero8[8];
            zero8[0] = 0.0f;
            zero8[1] = 0.0f;
            zero8[2] = 0.0f;
            zero8[3] = 0.0f;
            zero8[4] = 0.0f;
            zero8[5] = 0.0f;
            zero8[6] = 0.0f;
            zero8[7] = 0.0f;
            unsigned int ck_stage = 0;
            int clen_c = 0;
            float g_c = 0.0f;
            int row0_c = 0;
            int head_c = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_tk_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _j = 0; _j < num_items_c + 1; _j++) {
                if (_j > 0) {
                    mbarrier_wait(tiles_full_addr + (do_stage) * 8, _phase_tiles_full);
                }
                mbarrier_wait(tk_full_addr + (ck_stage) * 8, _phase_tk_full);
                unsigned int tku[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku[0])) : "r"(tk_s_addr + ck_stage * 4) : "memory");
                int u = (int)tku[0];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(tk_empty_addr + (ck_stage) * 8);
                ck_stage += 1;
                if (ck_stage == 2) { ck_stage = 0; _phase_tk_full ^= 1; }
                int clen_n = 0;
                float g_n = 0.0f;
                int row0_n = 0;
                int head_n = 0;
                if (u < num_items_c) {
                    unsigned int it = (unsigned int)u;
                    int chunk = it / (unsigned int)num_heads;
                    int head = it % (unsigned int)num_heads;
                    int row0 = chunk * 64;
                    head_n = head;
                    row0_n = row0;
                    clen_n = chunk_len[chunk];
                    g_n = g_cum[(long long)(row0 + t_do) * nh64 + (long long)head];
                }
                if (_j > 0) {
                    long long h64 = (long long)head_c;
                    float _exp_0 = expf(g_c);
                    float eg_own = _exp_0 * scale;
                    unsigned int src_l = do_s_addr + do_stage * 24576 + do_row_l;
                    long long dog_l = ((long long)(row0_c + wq * 16 + rsel) * nh64 + h64) * 128 + (long long)col_l;
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        float eg_c = __shfl_sync(0xFFFFFFFF, eg_own, 2 * c + rsel);
                        unsigned int pk[4];
                        if (wq * 16 + 2 * c + rsel < clen_c) {
                            unsigned int w[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&w[0])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 3]))
                                : "r"(src_l + (unsigned int)(c * 256) + (unsigned int)((sw_l ^ 2 * c & 7) << 4)));
                            float dv8[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&dv8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w[_pair]) << 16);
                                (&dv8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w[_pair]) & 0xffff0000u);
                            }
                            float og[8];
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                og[j] = eg_c * dv8[j];
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(og[_lp*2 + 0], og[_lp*2+1 + 0]));
                                pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(og[_lp*2 + 2], og[_lp*2+1 + 2]));
                                pk[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(og[_lp*2 + 4], og[_lp*2+1 + 4]));
                                pk[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(og[_lp*2 + 6], og[_lp*2+1 + 6]));
                                pk[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 0], zero8[_lp*2+1 + 0]));
                                pk[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 2], zero8[_lp*2+1 + 2]));
                                pk[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 4], zero8[_lp*2+1 + 4]));
                                pk[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero8[_lp*2 + 6], zero8[_lp*2+1 + 6]));
                                pk[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                        }
                        reinterpret_cast<int4*>(do_g + (dog_l + (long long)c * (nh64 * 256)))[0] = reinterpret_cast<int4*>(pk)[0];
                    }
                    mbarrier_arrive(tiles_empty_addr + (do_stage) * 8);
                    do_stage += 1;
                    if (do_stage == 2) { do_stage = 0; _phase_tiles_full ^= 1; }
                    mbarrier_wait(acc_full_addr + (epi_buf) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int buf_col = (int)epi_buf * 128;
                    long long dvi_base = ((long long)(row0_c + s_row) * nh64 + h64) * 128 + (long long)col_half;
                    #pragma unroll
                    for (int blk = 0; blk < 4; blk++) {
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col + blk * 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float out16[16];
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 16; j_1++) {
                            out16[j_1] = ((s_row < clen_c) ? _tmem_load_0[j_1] : 0.0f);
                        }
                        unsigned int _packed_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 0], out16[_lp*2+1 + 0]));
                            _packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 2], out16[_lp*2+1 + 2]));
                            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 4], out16[_lp*2+1 + 4]));
                            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 6], out16[_lp*2+1 + 6]));
                            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 8], out16[_lp*2+1 + 8]));
                            _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 10], out16[_lp*2+1 + 10]));
                            _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 12], out16[_lp*2+1 + 12]));
                            _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out16[_lp*2 + 14], out16[_lp*2+1 + 14]));
                            _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dv_intra + (dvi_base + (long long)(blk * 32)))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(acc_empty_addr + (epi_buf) * 8);
                    epi_buf += 1;
                    if (epi_buf == 2) { epi_buf = 0; _phase_acc_full ^= 1; }
                }
                if (u >= num_items_c) {
                    break;
                }
                clen_c = clen_n;
                g_c = g_n;
                row0_c = row0_n;
                head_c = head_n;
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int num_items_l = seq_chunk_start[num_seqs] * num_heads;
            const int grid_l = (int)num_bids;
            unsigned int ld_stage = 0;
            unsigned int lk_stage = 0;
            unsigned int _phase_tk_empty = 1;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                const unsigned int tk_limit = (unsigned int)(num_items_l - 1);
                unsigned int drawn = (unsigned int)bid;
                unsigned int tkw[1];
                #pragma unroll 1
                for (int _j_1 = 0; _j_1 < num_items_l + 1; _j_1++) {
                    mbarrier_wait(tk_empty_addr + (lk_stage) * 8, _phase_tk_empty);
                    tkw[0] = drawn;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tk_s_addr + lk_stage * 4), "r"(tkw[0]) : "memory");
                    mbarrier_arrive(tk_full_addr + (lk_stage) * 8);
                    lk_stage += 1;
                    if (lk_stage == 2) { lk_stage = 0; _phase_tk_empty ^= 1; }
                    int u_1 = (int)drawn;
                    if (num_items_l > grid_l) {
                        if ((unsigned int)u_1 == tk_limit + (unsigned int)grid_l) {
                            asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(ticket)), "r"(static_cast<unsigned int>(0)) : "memory");
                        }
                    }
                    if (u_1 >= num_items_l) {
                        break;
                    }
                    if (num_items_l > grid_l) {
                        unsigned int _atomic_fetch_add_0 = atomicAdd(ticket, 1);
                        drawn = _atomic_fetch_add_0 + (unsigned int)grid_l;
                    } else {
                        drawn = (unsigned int)(u_1 + grid_l);
                    }
                    unsigned int it_1 = (unsigned int)u_1;
                    int chunk_1 = it_1 / (unsigned int)num_heads;
                    int head_1 = it_1 % (unsigned int)num_heads;
                    int row0_1 = chunk_1 * 64;
                    int trow = chunk_bos[chunk_1];
                    mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 24576);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(do_s_addr + ld_stage * 24576 + (unsigned int)(seg * 8192), (&do_tma), seg * 64, head_1, trow, tiles_full_addr + (ld_stage) * 8);
                    }
                    tma_3d_gmem2smem(aqk_s_addr + ld_stage * 24576, (&aqk_tma), 0, head_1, row0_1, tiles_full_addr + (ld_stage) * 8);
                    ld_stage += 1;
                    if (ld_stage == 2) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int num_items_m = seq_chunk_start[num_seqs] * num_heads;
            unsigned int mma_stage = 0;
            unsigned int mma_buf = 0;
            const int lane_m = lane;
            unsigned int mk_stage = 0;
            unsigned int _phase_tk_full_1 = 0;
            unsigned int _phase_tiles_full_1 = 0;
            unsigned int _phase_acc_empty = 1;
            #pragma unroll 1
            for (int _j_2 = 0; _j_2 < num_items_m + 1; _j_2++) {
                mbarrier_wait(tk_full_addr + (mk_stage) * 8, _phase_tk_full_1);
                unsigned int tku_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku_1[0])) : "r"(tk_s_addr + mk_stage * 4) : "memory");
                int u_2 = (int)tku_1[0];
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(tk_empty_addr + (mk_stage) * 8);
                mk_stage += 1;
                if (mk_stage == 2) { mk_stage = 0; _phase_tk_full_1 ^= 1; }
                if (u_2 >= num_items_m) {
                    break;
                }
                unsigned int it_2 = (unsigned int)u_2;
                int chunk_2 = it_2 / (unsigned int)num_heads;
                int head_2 = it_2 % (unsigned int)num_heads;
                int row0_2 = chunk_2 * 64;
                int clen_m = chunk_len[chunk_2];
                mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full_1);
                if (clen_m < 64) {
                    unsigned int zeros[4];
                    zeros[0] = 0;
                    zeros[1] = 0;
                    zeros[2] = 0;
                    zeros[3] = 0;
                    int npad = (64 - clen_m) * 16;
                    #pragma unroll 1
                    for (int idx = lane_m; idx < npad; idx += 32) {
                        int row = clen_m + (idx >> 4);
                        int rem = idx & 15;
                        int off = row * 128 + (rem >> 3) * 8192 + (rem & 7) * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(do_s_addr + mma_stage * 24576 + (unsigned int)off), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((((aqk_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 1536;
                    int _smem_desc_lo_1 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 1536;
#else
                    int _smem_desc_lo_0 = ((((aqk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 1536;
                    int _smem_desc_lo_1 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 1536;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69305488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (mma_buf * 128))), "r"(0) : "memory");
                    tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                    tcgen05_commit(acc_full_addr + (mma_buf) * 8);
                }
                mma_stage += 1;
                if (mma_stage == 2) { mma_stage = 0; _phase_tiles_full_1 ^= 1; }
                mma_buf += 1;
                if (mma_buf == 2) { mma_buf = 0; _phase_acc_empty ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
