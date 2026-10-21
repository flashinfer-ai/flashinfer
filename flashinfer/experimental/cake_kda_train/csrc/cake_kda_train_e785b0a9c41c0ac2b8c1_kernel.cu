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
#define TMEM_NCOLS 512
#define TMEM_ACC1_OFFSET 0
#define TMEM_ACC2_OFFSET 256
#define NUM_ONE_STAGES 1
#define NUM_RING_STAGES 2
#define NUM_ACCS_STAGES 2
#define SMEM_DO_S_OFF 1024
#define SMEM_DO_S_STAGE_BYTES 16384
#define SMEM_DO_S_STRIDE 98304
#define SMEM_VNEW_S_OFF 33792
#define SMEM_VNEW_S_STAGE_BYTES 16384
#define SMEM_VNEW_S_STRIDE 98304
#define SMEM_AQK_S_OFF 66560
#define SMEM_AQK_S_STAGE_BYTES 8192
#define SMEM_AQK_S_STRIDE 98304
#define SMEM_BLOCKS_OFF 1024
#define SMEM_BLOCKS_STAGE_BYTES 8192
#define SMEM_BLOCKS_STRIDE 8192
#define SMEM_DO_K_OFF 1024
#define SMEM_DO_K_STAGE_BYTES 32768
#define SMEM_DO_K_STRIDE 98304
#define SMEM_VNEW_K_OFF 33792
#define SMEM_VNEW_K_STAGE_BYTES 32768
#define SMEM_VNEW_K_STRIDE 98304
#define SMEM_DO_MN_OFF 1024
#define SMEM_DO_MN_STAGE_BYTES 32768
#define SMEM_DO_MN_STRIDE 98304
#define SMEM_AQK_MN_OFF 66560
#define SMEM_AQK_MN_STAGE_BYTES 32768
#define SMEM_AQK_MN_STRIDE 98304
#define SMEM_TOTAL 197632
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

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_e785b0a9c41c0ac2b8c1(const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap vnew_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ dAqk, __nv_bfloat16* __restrict__ dv1, int num_heads, float scale, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, int num_pairs)
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
    #define zero_done_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* do_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_s_addr = smem + 1024;
    __nv_bfloat16* vnew_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vnew_s_addr = smem + 33792;
    __nv_bfloat16* aqk_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int aqk_s_addr = smem + 66560;
    __nv_bfloat16* blocks = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int blocks_addr = smem + 1024;
    __nv_bfloat16* do_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_k_addr = smem + 1024;
    __nv_bfloat16* vnew_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vnew_k_addr = smem + 33792;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_mn_addr = smem + 1024;
    __nv_bfloat16* aqk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int aqk_mn_addr = smem + 66560;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'accs' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // acc_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            mbarrier_init(smem + 56, 256);
            // --- pipeline 'one' ---
            // zero_done: 1 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 72);
    if (warp == 0) {
        int _tmem_hold = smem + 72;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc1 = taddr;
    const int tmem_acc2 = taddr + 256;

    // ---- Role: epi_a ----
    if (warp <= 3) {
        { // epi_a_main
            const int epi_warp = warp;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int zeros[4];
            zeros[0] = 0;
            zeros[1] = 0;
            zeros[2] = 0;
            zeros[3] = 0;
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                #pragma unroll
                for (int blk = 1; blk < 3; blk++) {
                    int base = blocks_addr + (unsigned int)((s * 12 + 8 + blk) * 8192);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base + (j * 256 + epi_tid) * 16), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(zero_done_addr);
            unsigned int epi_buf = 0;
            int num_items_e = num_pairs * num_heads;
            const int quad = (epi_warp & 3) * 32;
            const int row_base = quad << 16;
            const int row = quad + lane;
            const int chunk = row / 64;
            const int t_local = row - chunk * 64;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int item = bid; item < num_items_e; item += num_bids) {
                int it_e = (int)item;
                int pair_e = it_e / num_heads;
                int head_e = it_e - pair_e * num_heads;
                int grow = pair_e * 128 + row;
                int dA_base = (grow * num_heads + head_e) * 64;
                int buf_col = (int)epi_buf * 128 + chunk * 64;
                mbarrier_wait(acc_full_addr + (epi_buf) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int grp = 0; grp < 1; grp++) {
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col + grp * 64)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(buf_col + grp * 64) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int q = 0; q < 8; q++) {
                        float out8[8];
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            int s_local = grp * 64 + q * 8 + j_1;
                            float val = _tmem_load_0[q * 8 + j_1] * scale;
                            if (s_local > t_local) {
                                val = 0.0f;
                            }
                            out8[j_1] = val;
                        }
                        {
                            unsigned _stv8_0_0 = __float_as_uint(out8[0 + 0]);
                            unsigned _stv8_0_1 = __float_as_uint(out8[0 + 1]);
                            unsigned _stv8_0_2 = __float_as_uint(out8[0 + 2]);
                            unsigned _stv8_0_3 = __float_as_uint(out8[0 + 3]);
                            unsigned _stv8_0_4 = __float_as_uint(out8[0 + 4]);
                            unsigned _stv8_0_5 = __float_as_uint(out8[0 + 5]);
                            unsigned _stv8_0_6 = __float_as_uint(out8[0 + 6]);
                            unsigned _stv8_0_7 = __float_as_uint(out8[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dAqk + (dA_base + grp * 64 + q * 8) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr + (epi_buf) * 8);
                epi_buf += 1;
                if (epi_buf == 2) { epi_buf = 0; _phase_acc_full ^= 1; }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: epi_v ----
    if (warp >= 4 && warp <= 7) {
        { // epi_v_main
            const int epi_warp_1 = warp;
            const int epi_tid_1 = epi_warp_1 * 32 + lane;
            unsigned int zeros_1[4];
            zeros_1[0] = 0;
            zeros_1[1] = 0;
            zeros_1[2] = 0;
            zeros_1[3] = 0;
            #pragma unroll
            for (int s_1 = 0; s_1 < 2; s_1++) {
                #pragma unroll
                for (int blk_1 = 1; blk_1 < 3; blk_1++) {
                    int base_1 = blocks_addr + (unsigned int)((s_1 * 12 + 8 + blk_1) * 8192);
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 2; j_2++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base_1 + (j_2 * 256 + epi_tid_1) * 16), "r"(zeros_1[0]), "r"(zeros_1[1]), "r"(zeros_1[2]), "r"(zeros_1[3]) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(zero_done_addr);
            unsigned int epi_buf_1 = 0;
            int num_items_e_1 = num_pairs * num_heads;
            const int quad_1 = (epi_warp_1 & 3) * 32;
            const int row_base_1 = quad_1 << 16;
            const int row_1 = quad_1 + lane;
            unsigned int _phase_acc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int item_1 = bid; item_1 < num_items_e_1; item_1 += num_bids) {
                int it_e_1 = (int)item_1;
                int pair_e_1 = it_e_1 / num_heads;
                int head_e_1 = it_e_1 - pair_e_1 * num_heads;
                int grow_1 = pair_e_1 * 128 + row_1;
                long long dv_base = ((long long)grow_1 * (long long)num_heads + (long long)head_e_1) * 128;
                int buf_col_1 = (int)epi_buf_1 * 128;
                mbarrier_wait(acc_full_addr + (epi_buf_1) * 8, _phase_acc_full_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int grp_1 = 0; grp_1 < 2; grp_1++) {
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(taddr + 256 + (unsigned int)row_base_1 + (unsigned int)(buf_col_1 + grp_1 * 64)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(taddr + 256 + (unsigned int)row_base_1 + (unsigned int)(buf_col_1 + grp_1 * 64) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 4; j_3++) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 0], _tmem_load_1[j_3 * 16 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 2], _tmem_load_1[j_3 * 16 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 4], _tmem_load_1[j_3 * 16 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 6], _tmem_load_1[j_3 * 16 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 8], _tmem_load_1[j_3 * 16 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 10], _tmem_load_1[j_3 * 16 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 12], _tmem_load_1[j_3 * 16 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_1[j_3 * 16 + 14], _tmem_load_1[j_3 * 16 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(dv1 + (dv_base + (long long)(grp_1 * 64 + j_3 * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr + (epi_buf_1) * 8);
                epi_buf_1 += 1;
                if (epi_buf_1 == 2) { epi_buf_1 = 0; _phase_acc_full_1 ^= 1; }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            unsigned int ld_stage = 0;
            int num_items_l = num_pairs * num_heads;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                int nbos0 = 0;
                int nbos1 = 0;
                #pragma unroll 1
                for (unsigned int item_2 = bid; item_2 < num_items_l; item_2 += num_bids) {
                    int it_l = (int)item_2;
                    int pair_l = it_l / num_heads;
                    int head_l = it_l - pair_l * num_heads;
                    int row0_l = pair_l * 128;
                    int bos0_l = nbos0;
                    int bos1_l = nbos1;
                    mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 81920);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(do_s_addr + ld_stage * 98304 + (unsigned int)(seg * 16384), (&do_tma), seg * 64, head_l, row0_l, tiles_full_addr + (ld_stage) * 8);
                        tma_3d_gmem2smem(vnew_s_addr + ld_stage * 98304 + (unsigned int)(seg * 16384), (&vnew_tma), seg * 64, head_l, row0_l, tiles_full_addr + (ld_stage) * 8);
                    }
                    tma_3d_gmem2smem(aqk_s_addr + ld_stage * 98304, (&aqk_tma), 0, head_l, row0_l, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(aqk_s_addr + ld_stage * 98304 + 24576, (&aqk_tma), 0, head_l, row0_l + 64, tiles_full_addr + (ld_stage) * 8);
                    ld_stage += 1;
                    if (ld_stage == 2) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_buf = 0;
            int num_items_m = num_pairs * num_heads;
            unsigned int _phase_zero_done_0 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_acc_empty = 1;
            if (elect_sync()) {
                mbarrier_wait(zero_done_addr, _phase_zero_done_0);
                _phase_zero_done_0 ^= 1;
                #pragma unroll 1
                for (unsigned int item_3 = bid; item_3 < num_items_m; item_3 += num_bids) {
                    mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full);
                    mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = (((do_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 6144;
                    int _mma_b_lo_0 = (((vnew_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 6144;
#else
                    int _mma_a_lo_0 = (((do_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
                    int _mma_b_lo_0 = (((vnew_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
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
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc1 + (mma_buf * 128))), "r"(0));
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_1 = ((((aqk_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 6144;
                    int _mma_b_lo_1 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 6144;
#else
                    int _mma_a_lo_1 = ((((aqk_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
                    int _mma_b_lo_1 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_acc2 + (mma_buf * 128))), "r"(0));
                    tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                    tcgen05_commit(acc_full_addr + (mma_buf) * 8);
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_tiles_full ^= 1; }
                    mma_buf += 1;
                    if (mma_buf == 2) { mma_buf = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

