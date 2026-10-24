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
#define TMEM_ACC_U_OFFSET 0
#define TMEM_ACC_W_OFFSET 256
#define NUM_ONE_STAGES 1
#define NUM_RING_STAGES 2
#define NUM_ACCS_STAGES 2
#define SMEM_AKK_BLOCKS_OFF 1024
#define SMEM_AKK_BLOCKS_STAGE_BYTES 8192
#define SMEM_AKK_BLOCKS_STRIDE 8192
#define SMEM_VB_S_OFF 33792
#define SMEM_VB_S_STAGE_BYTES 32768
#define SMEM_VB_S_STRIDE 98304
#define SMEM_KB_S_OFF 66560
#define SMEM_KB_S_STAGE_BYTES 32768
#define SMEM_KB_S_STRIDE 98304
#define SMEM_AKK_K_OFF 1024
#define SMEM_AKK_K_STAGE_BYTES 32768
#define SMEM_AKK_K_STRIDE 98304
#define SMEM_VB_MN_OFF 33792
#define SMEM_VB_MN_STAGE_BYTES 32768
#define SMEM_VB_MN_STRIDE 98304
#define SMEM_KB_MN_OFF 66560
#define SMEM_KB_MN_STAGE_BYTES 32768
#define SMEM_KB_MN_STRIDE 98304
#define SMEM_TOTAL 197632
#define THREADS 448
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

__global__ __launch_bounds__(448, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_18bd735cf970f8dfa1b4(const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap kb_tma, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ u_out, __nv_bfloat16* __restrict__ w_out, int num_heads, int num_pairs)
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
    #define v_full_addr (mbar_base + 16)
    #define tiles_empty_addr (mbar_base + 32)
    #define vb_ready_addr (mbar_base + 48)
    #define acc_full_addr (mbar_base + 64)
    #define acc_empty_addr (mbar_base + 80)
    #define zero_done_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* akk_blocks = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int akk_blocks_addr = smem + 1024;
    __nv_bfloat16* vb_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vb_s_addr = smem + 33792;
    __nv_bfloat16* kb_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int kb_s_addr = smem + 66560;
    __nv_bfloat16* akk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int akk_k_addr = smem + 1024;
    __nv_bfloat16* vb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vb_mn_addr = smem + 33792;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int kb_mn_addr = smem + 66560;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // vb_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            // --- pipeline 'accs' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // acc_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            mbarrier_init(smem + 88, 256);
            // --- pipeline 'one' ---
            // zero_done: 1 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 0) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_u = taddr;
    const int tmem_acc_w = taddr + 256;

    // ---- Role: epi_u ----
    if (warp <= 3) {
        { // epi_u_main
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
                    int base = akk_blocks_addr + (unsigned int)((s * 12 + blk) * 8192);
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
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int item = bid; item < num_items_e; item += num_bids) {
                int it_e = (int)item;
                int pair_e = it_e / num_heads;
                int head_e = it_e - pair_e * num_heads;
                int grow = pair_e * 128 + quad + lane;
                long long out_base = ((long long)grow * (long long)num_heads + (long long)head_e) * 128;
                int buf_col = (int)epi_buf * 128;
                mbarrier_wait(acc_full_addr + (epi_buf) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int grp = 0; grp < 2; grp++) {
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
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        unsigned int _packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 0], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 2], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 4], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 6], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 8], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 8]));
                            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 10], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 10]));
                            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 12], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 12]));
                            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1 * 16)[_lp*2 + 14], (_tmem_load_0 + j_1 * 16)[_lp*2+1 + 14]));
                            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (out_base + (long long)(grp * 64 + j_1 * 16)))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
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
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: epi_w ----
    if (warp >= 4 && warp <= 7) {
        { // epi_w_main
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
                    int base_1 = akk_blocks_addr + (unsigned int)((s_1 * 12 + blk_1) * 8192);
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
            unsigned int _phase_acc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int item_1 = bid; item_1 < num_items_e_1; item_1 += num_bids) {
                int it_e_1 = (int)item_1;
                int pair_e_1 = it_e_1 / num_heads;
                int head_e_1 = it_e_1 - pair_e_1 * num_heads;
                int grow_1 = pair_e_1 * 128 + quad_1 + lane;
                long long out_base_1 = ((long long)grow_1 * (long long)num_heads + (long long)head_e_1) * 128;
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
                        unsigned int _packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 0], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 2], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 4], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 6], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 8], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 8]));
                            _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 10], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 10]));
                            _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 12], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 12]));
                            _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_3 * 16)[_lp*2 + 14], (_tmem_load_1 + j_3 * 16)[_lp*2+1 + 14]));
                            _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (out_base_1 + (long long)(grp_1 * 64 + j_3 * 16)))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr + (epi_buf_1) * 8);
                epi_buf_1 += 1;
                if (epi_buf_1 == 2) { epi_buf_1 = 0; _phase_acc_full_1 ^= 1; }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            unsigned int ld_stage = 0;
            int num_items_l = num_pairs * num_heads;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                int first_l = (int)bid;
                int pair_f = first_l / num_heads;
                int chunk_f0 = pair_f * 2;
                int chunk_f1 = pair_f * 2 + 1;
                int nbos0 = chunk_bos[chunk_f0];
                int nbos1 = chunk_bos[chunk_f1];
                #pragma unroll 1
                for (unsigned int item_2 = bid; item_2 < num_items_l; item_2 += num_bids) {
                    int it_l = (int)item_2;
                    int pair_l = it_l / num_heads;
                    int head_l = it_l - pair_l * num_heads;
                    int row0_l = pair_l * 128;
                    int bos0_l = nbos0;
                    int bos1_l = nbos1;
                    int it_n = it_l + (int)num_bids;
                    if (it_n >= num_items_l) {
                        it_n = it_l;
                    }
                    int pair_n = it_n / num_heads;
                    int chunk_n0 = pair_n * 2;
                    int chunk_n1 = pair_n * 2 + 1;
                    nbos0 = chunk_bos[chunk_n0];
                    nbos1 = chunk_bos[chunk_n1];
                    mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(v_full_addr + (ld_stage) * 8, 32768);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 49152);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(vb_s_addr + ld_stage * 98304 + (unsigned int)(seg * 16384), (&v_tma), seg * 64, head_l, bos0_l, v_full_addr + (ld_stage) * 8);
                        tma_3d_gmem2smem(vb_s_addr + ld_stage * 98304 + (unsigned int)(seg * 16384 + 8192), (&v_tma), seg * 64, head_l, bos1_l, v_full_addr + (ld_stage) * 8);
                        tma_3d_gmem2smem(kb_s_addr + ld_stage * 98304 + (unsigned int)(seg * 16384), (&kb_tma), seg * 64, head_l, row0_l, tiles_full_addr + (ld_stage) * 8);
                    }
                    int akk_row0 = row0_l;
                    int akk_row1 = row0_l + 64;
                    tma_3d_gmem2smem(akk_k_addr + ld_stage * 98304, (&akk_tma), 0, head_l, akk_row0, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(akk_k_addr + ld_stage * 98304 + 24576, (&akk_tma), 0, head_l, akk_row1, tiles_full_addr + (ld_stage) * 8);
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
                for (unsigned int item_m = bid; item_m < num_items_m; item_m += num_bids) {
                    mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full);
                    mbarrier_wait(vb_ready_addr + (mma_stage) * 8, _phase_tiles_full);
                    mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((akk_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 6144;
                    int _smem_desc_lo_1 = ((((vb_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 6144;
#else
                    int _smem_desc_lo_0 = (((akk_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
                    int _smem_desc_lo_1 = ((((vb_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc_u + (mma_buf * 128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((akk_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 6144;
                    int _smem_desc_lo_3 = ((((kb_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 6144;
#else
                    int _smem_desc_lo_2 = (((akk_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
                    int _smem_desc_lo_3 = ((((kb_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
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
                    "mov.b32 id, 136381584;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc_w + (mma_buf * 128))), "r"(0) : "memory");
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
    // ---- Role: cvt ----
    if (warp >= 10 && warp <= 13) {
        { // cvt_main
            const int cvt_tid = (warp - 10) * 32 + lane;
            const int c_of = cvt_tid >> 6;
            const int t_of = cvt_tid & 63;
            const int swz = cvt_tid & 7;
            unsigned int cvt_stage = 0;
            int num_items_c = num_pairs * num_heads;
            int it_f = (int)bid;
            int pair_cf = it_f / num_heads;
            int head_cf = it_f - pair_cf * num_heads;
            int chunk_cf = pair_cf * 2 + c_of;
            int trow_f = chunk_bos[chunk_cf];
            int clen_f = chunk_len[chunk_cf];
            int clast_f = clen_f - 1;
            if (clast_f < 0) {
                clast_f = 0;
            }
            int t_cf = t_of;
            if (t_cf > clast_f) {
                t_cf = clast_f;
            }
            long long bidx_f = (long long)(trow_f + t_cf) * (long long)num_heads + (long long)head_cf;
            float bt_pending = (float)beta[bidx_f];
            int valid_pending = 0;
            if (t_of < clen_f) {
                valid_pending = 1;
            }
            unsigned int _phase_v_full = 0;
            #pragma unroll 1
            for (unsigned int item_3 = bid; item_3 < num_items_c; item_3 += num_bids) {
                float bt = 0.0f;
                if (valid_pending == 1) {
                    bt = bt_pending;
                }
                unsigned int mask = (unsigned int)(-valid_pending);
                int it_n_1 = (int)item_3 + (int)num_bids;
                if (it_n_1 >= num_items_c) {
                    it_n_1 = (int)item_3;
                }
                int pair_n_1 = it_n_1 / num_heads;
                int head_n = it_n_1 - pair_n_1 * num_heads;
                int chunk_n = pair_n_1 * 2 + c_of;
                int trow_n = chunk_bos[chunk_n];
                int clen_n = chunk_len[chunk_n];
                int clast_n = clen_n - 1;
                if (clast_n < 0) {
                    clast_n = 0;
                }
                int t_cn = t_of;
                if (t_cn > clast_n) {
                    t_cn = clast_n;
                }
                long long bidx_n = (long long)(trow_n + t_cn) * (long long)num_heads + (long long)head_n;
                bt_pending = (float)beta[bidx_n];
                valid_pending = 0;
                if (t_of < clen_n) {
                    valid_pending = 1;
                }
                mbarrier_wait(v_full_addr + (cvt_stage) * 8, _phase_v_full);
                unsigned int row_addr = vb_s_addr + cvt_stage * 98304 + (unsigned int)(cvt_tid * 128);
                #pragma unroll
                for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 8; j_4++) {
                        unsigned int addr = row_addr + (unsigned int)(seg_1 * 16384 + ((swz ^ j_4) << 4));
                        unsigned int raw[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                            : "r"(addr));
                        unsigned int sel[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            sel[i] = raw[i] & mask;
                        }
                        float _unpack_bf16_0[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel[_pair]) & 0xffff0000u);
                        }
                        float prod[8];
                        #pragma unroll
                        for (int e = 0; e < 8; e++) {
                            prod[e] = _unpack_bf16_0[e] * bt;
                        }
                        unsigned int packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod[_lp*2 + 0], prod[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vb_ready_addr + (cvt_stage) * 8);
                cvt_stage += 1;
                if (cvt_stage == 2) { cvt_stage = 0; _phase_v_full ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
