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
#define TMEM_ACC_DH_OFFSET 0
#define TMEM_ACC_DV_OFFSET 128
#define NUM_K_PIPE_STAGES 1
#define NUM_REST_PIPE_STAGES 1
#define NUM_ONE_STAGES 1
#define SMEM_K_PANEL_OFF 1024
#define SMEM_K_PANEL_STAGE_BYTES 8192
#define SMEM_K_PANEL_STRIDE 8192
#define SMEM_Q_PANEL_OFF 17408
#define SMEM_Q_PANEL_STAGE_BYTES 8192
#define SMEM_Q_PANEL_STRIDE 8192
#define SMEM_W_PANEL_OFF 33792
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_DO_PANEL_OFF 50176
#define SMEM_DO_PANEL_STAGE_BYTES 8192
#define SMEM_DO_PANEL_STRIDE 8192
#define SMEM_K_OP_OFF 1024
#define SMEM_K_OP_STAGE_BYTES 16384
#define SMEM_K_OP_STRIDE 16384
#define SMEM_Q_MN_OFF 17408
#define SMEM_Q_MN_STAGE_BYTES 16384
#define SMEM_Q_MN_STRIDE 16384
#define SMEM_W_MN_OFF 33792
#define SMEM_W_MN_STAGE_BYTES 16384
#define SMEM_W_MN_STRIDE 16384
#define SMEM_DO_MN_OFF 50176
#define SMEM_DO_MN_STAGE_BYTES 16384
#define SMEM_DO_MN_STRIDE 16384
#define SMEM_DHS_OFF 66560
#define SMEM_DHS_STAGE_BYTES 32768
#define SMEM_DHS_STRIDE 32768
#define SMEM_DHS_MN_OFF 66560
#define SMEM_DHS_MN_STAGE_BYTES 32768
#define SMEM_DHS_MN_STRIDE 32768
#define SMEM_BDV_OFF 99328
#define SMEM_BDV_STAGE_BYTES 16384
#define SMEM_BDV_STRIDE 16384
#define SMEM_BDV_MN_OFF 99328
#define SMEM_BDV_MN_STAGE_BYTES 16384
#define SMEM_BDV_MN_STRIDE 16384
#define SMEM_TOTAL 115712
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


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
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
kernel_cake_gdn_train_94ff8219a97e6835ca9e(const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_g_tma, __nv_bfloat16* __restrict__ dv_intra, float* __restrict__ rowscale, float* __restrict__ gamma, int num_heads, int num_qk_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ dht, int* __restrict__ dht_index, int dht_hg, int dht_n, float* __restrict__ out, int* __restrict__ out_index, int out_hg, int out_n, int tile_stride, __nv_bfloat16* __restrict__ dh_tape, __nv_bfloat16* __restrict__ dv_new)
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
    #define k_full_addr (mbar_base + 0)
    #define k_empty_addr (mbar_base + 8)
    #define rest_full_addr (mbar_base + 16)
    #define rest_empty_addr (mbar_base + 24)
    #define dhs_ready_addr (mbar_base + 32)
    #define dv_ready_addr (mbar_base + 40)
    #define bdv_ready_addr (mbar_base + 48)
    #define dh_ready_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* k_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int k_panel_addr = smem + 1024;
    __nv_bfloat16* q_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int q_panel_addr = smem + 17408;
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_panel_addr = smem + 33792;
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int do_panel_addr = smem + 50176;
    __nv_bfloat16* k_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int k_op_addr = smem + 1024;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int q_mn_addr = smem + 17408;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_mn_addr = smem + 33792;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int do_mn_addr = smem + 50176;
    __nv_bfloat16* dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dhs_addr = smem + 66560;
    __nv_bfloat16* dhs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dhs_mn_addr = smem + 66560;
    __nv_bfloat16* bdv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_addr = smem + 99328;
    __nv_bfloat16* bdv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_mn_addr = smem + 99328;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'rest_pipe' ---
            // rest_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // rest_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // dhs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // dv_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dh = taddr;
    const int tmem_acc_dv = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int hb = blockIdx.x;
            int vbase = 0;
            int head = hb % num_heads;
            int walk = hb / num_heads;
            int seg = walk_seg[walk];
            int c0 = seq_chunk_start[seg];
            int nc = seq_chunk_start[seg + 1] - c0;
            int row0 = c0 * 64;
            int qk_head = head / (num_heads / num_qk_heads);
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int m128_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int m64_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            int dht_slot = dht_index[seg];
            if (dht_slot >= 0) {
                int dht_tile = (head / dht_hg * dht_n + dht_slot) * dht_hg + head % dht_hg;
                long long dht_base = (long long)dht_tile * (long long)tile_stride * 16384 + (long long)(k_row * 128);
                #pragma unroll
                for (int blk = 0; blk < 8; blk++) {
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(blk * 16 + q * 4));
                            init_row[q * 4 + 0] = _v4.x;
                            init_row[q * 4 + 1] = _v4.y;
                            init_row[q * 4 + 2] = _v4.z;
                            init_row[q * 4 + 3] = _v4.w;
                        }
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk * 16), init_row);
                }
            } else {
                init_row[0] = 0.0f;
                init_row[1] = 0.0f;
                init_row[2] = 0.0f;
                init_row[3] = 0.0f;
                init_row[4] = 0.0f;
                init_row[5] = 0.0f;
                init_row[6] = 0.0f;
                init_row[7] = 0.0f;
                init_row[8] = 0.0f;
                init_row[9] = 0.0f;
                init_row[10] = 0.0f;
                init_row[11] = 0.0f;
                init_row[12] = 0.0f;
                init_row[13] = 0.0f;
                init_row[14] = 0.0f;
                init_row[15] = 0.0f;
                #pragma unroll
                for (int blk_1 = 0; blk_1 < 8; blk_1++) {
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk_1 * 16), init_row);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int _phase_dv_ready_0 = 0;
            unsigned int _phase_dh_ready_0 = 0;
            #pragma unroll 1
            for (int rc = 0; rc < nc; rc++) {
                int c = nc - 1 - rc;
                int gc = c0 + c;
                long long dh_base = ((long long)gc * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                float decay = gamma[(long long)gc * (long long)num_heads + (long long)head];
                #pragma unroll
                for (int g = 0; g < 2; g++) {
                    float dh4[64];
                    tmem_ld_x16(&dh4[0], taddr + (unsigned int)m128_row_base + (unsigned int)(g * 64));
                    tmem_ld_x16(&dh4[16], taddr + (unsigned int)m128_row_base + (unsigned int)(g * 64 + 16));
                    tmem_ld_x16(&dh4[32], taddr + (unsigned int)m128_row_base + (unsigned int)(g * 64 + 32));
                    tmem_ld_x16(&dh4[48], taddr + (unsigned int)m128_row_base + (unsigned int)(g * 64 + 48));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 4; q_1++) {
                        unsigned int _packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 0], (dh4 + q_1 * 16)[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 2], (dh4 + q_1 * 16)[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 4], (dh4 + q_1 * 16)[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 6], (dh4 + q_1 * 16)[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 8], (dh4 + q_1 * 16)[_lp*2+1 + 8]));
                            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 10], (dh4 + q_1 * 16)[_lp*2+1 + 10]));
                            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 12], (dh4 + q_1 * 16)[_lp*2+1 + 12]));
                            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((dh4 + q_1 * 16)[_lp*2 + 14], (dh4 + q_1 * 16)[_lp*2+1 + 14]));
                            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_tape + (dh_base + (long long)(g * 64 + q_1 * 16)))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                        }
                    }
                    unsigned int dh4_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh4[_lp*2 + 0], dh4[_lp*2+1 + 0]));
                        dh4_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_2 = 0; q_2 < 4; q_2++) {
                        #pragma unroll
                        for (int s = 0; s < 2; s++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((g * 64 + q_2 * 16 + s * 8) / 64 * 16384 + k_row * 128 + (g * 64 + q_2 * 16 + s * 8) % 64 * 2 ^ ((g * 64 + q_2 * 16 + s * 8) / 64 * 16384 + k_row * 128 + (g * 64 + q_2 * 16 + s * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dh4_bf16[q_2 * 8 + s * 4]), "r"(dh4_bf16[q_2 * 8 + s * 4 + 1]), "r"(dh4_bf16[q_2 * 8 + s * 4 + 2]), "r"(dh4_bf16[q_2 * 8 + s * 4 + 3]) : "memory");
                        }
                    }
                    #pragma unroll
                    for (int q_3 = 0; q_3 < 4; q_3++) {
                        float sc[16];
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            sc[j] = dh4[q_3 * 16 + j] * decay;
                        }
                        tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(g * 64 + q_3 * 16), sc);
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long t_base = ((long long)trow * (long long)num_heads + (long long)head) * 128;
                float rs_t = rowscale[(long long)trow * (long long)num_heads + (long long)head];
                bool valid = t_row < chunk_len[gc];
                float dv4[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[1])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[2])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[3])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[4])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[5])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[6])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[7])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[8])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[9])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[10])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[11])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[12])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[13])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[14])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv4[16])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[17])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[18])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[19])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[20])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[21])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[22])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[23])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[24])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[25])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[26])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[27])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[28])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[29])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[30])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[31]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv4[32])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[33])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[34])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[35])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[36])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[37])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[38])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[39])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[40])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[41])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[42])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[43])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[44])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[45])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[46])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[47]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dv4[48])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[49])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[50])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[51])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[52])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[53])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[54])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[55])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[56])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[57])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[58])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[59])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[60])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[61])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[62])), "=r"(*reinterpret_cast<uint32_t*>(&dv4[63]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int blk_2 = 0; blk_2 < 4; blk_2++) {
                    int col = blk_2 * 32 + lane_half * 16;
                    float d1[16];
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(dv_intra + t_base + (long long)col);
                        uint4 _vld_2[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                (&d1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    float bdvv[16];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 16; j_1++) {
                        bdvv[j_1] = ((valid) ? __fmaf_rn(dv4[blk_2 * 16 + j_1], rs_t, d1[j_1]) : 0.0f);
                    }
                    unsigned int _packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 0], bdvv[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 2], bdvv[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 4], bdvv[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 6], bdvv[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 8], bdvv[_lp*2+1 + 8]));
                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 10], bdvv[_lp*2+1 + 10]));
                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 12], bdvv[_lp*2+1 + 12]));
                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 14], bdvv[_lp*2+1 + 14]));
                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dv_new + (t_base + (long long)col))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                    }
                    unsigned int bdvv_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 0], bdvv[_lp*2+1 + 0]));
                        bdvv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 2; q_4++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 ^ ((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16[q_4 * 4]), "r"(bdvv_bf16[q_4 * 4 + 1]), "r"(bdvv_bf16[q_4 * 4 + 2]), "r"(bdvv_bf16[q_4 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int hb_1 = blockIdx.x;
            int vbase_1 = 0;
            int head_1 = hb_1 % num_heads;
            int walk_1 = hb_1 / num_heads;
            int seg_1 = walk_seg[walk_1];
            int c0_1 = seq_chunk_start[seg_1];
            int nc_1 = seq_chunk_start[seg_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int qk_head_1 = head_1 / (num_heads / num_qk_heads);
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_rest_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 0; rc_1 < nc_1; rc_1++) {
                    int c_1 = nc_1 - 1 - rc_1;
                    int trow_1 = row0_1 + c_1 * 64;
                    int tok = chunk_bos[c0_1 + c_1];
                    mbarrier_wait(k_empty_addr, _phase_k_empty);
                    mbarrier_arrive_expect_tx(k_full_addr, 16384);
                    #pragma unroll
                    for (int panel = 0; panel < 2; panel++) {
                        tma_3d_gmem2smem(k_panel_addr + (unsigned int)(panel * 8192), (&k_tma), panel * 64, qk_head_1, tok, k_full_addr);
                    }
                    _phase_k_empty ^= 1;
                    mbarrier_wait(rest_empty_addr, _phase_rest_empty);
                    mbarrier_arrive_expect_tx(rest_full_addr, 49152);
                    #pragma unroll
                    for (int panel_1 = 0; panel_1 < 2; panel_1++) {
                        tma_3d_gmem2smem(q_panel_addr + (unsigned int)(panel_1 * 8192), (&q_tma), panel_1 * 64, qk_head_1, tok, rest_full_addr);
                        tma_3d_gmem2smem(w_panel_addr + (unsigned int)(panel_1 * 8192), (&w_tma), panel_1 * 64, head_1, trow_1, rest_full_addr);
                    }
                    #pragma unroll
                    for (int panel_2 = 0; panel_2 < 2; panel_2++) {
                        tma_3d_gmem2smem(do_panel_addr + (unsigned int)(panel_2 * 8192), (&do_g_tma), panel_2 * 64, head_1, trow_1, rest_full_addr);
                    }
                    _phase_rest_empty ^= 1;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int hb_2 = blockIdx.x;
            int vbase_2 = 0;
            int head_2 = hb_2 % num_heads;
            int walk_2 = hb_2 / num_heads;
            int seg_2 = walk_seg[walk_2];
            int c0_2 = seq_chunk_start[seg_2];
            int nc_2 = seq_chunk_start[seg_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int qk_head_2 = head_2 / (num_heads / num_qk_heads);
            unsigned int _phase_k_full = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_rest_full = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(k_full_addr, _phase_k_full);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((k_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_1 = (((dhs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = ((k_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_1 = (((dhs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_dv), "r"(0) : "memory");
                    tcgen05_commit(dv_ready_addr);
                    tcgen05_commit(k_empty_addr);
                    _phase_k_full ^= 1;
                    mbarrier_wait(rest_full_addr, _phase_rest_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_3 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = (((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_3 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_dh), "r"(1) : "memory");
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((w_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_5 = (((bdv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_4 = (((w_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_5 = (((bdv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_dh), "r"(1), "r"((((uint32_t)(136414352)) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
                    tcgen05_commit(dh_ready_addr);
                    tcgen05_commit(rest_empty_addr);
                    _phase_rest_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
