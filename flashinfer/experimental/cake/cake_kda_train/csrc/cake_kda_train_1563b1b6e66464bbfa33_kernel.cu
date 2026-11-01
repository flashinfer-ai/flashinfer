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
#define TMEM_ACC_W_OFFSET 0
#define NUM_ONE_STAGES 1
#define NUM_RING_STAGES 2
#define NUM_ACCS_STAGES 2
#define NUM_WORK_STAGES 4
#define SMEM_AKK_BLOCKS_OFF 1024
#define SMEM_AKK_BLOCKS_STAGE_BYTES 8192
#define SMEM_AKK_BLOCKS_STRIDE 8192
#define SMEM_KB_S_OFF 66560
#define SMEM_KB_S_STAGE_BYTES 32768
#define SMEM_KB_S_STRIDE 98304
#define SMEM_AKK_K_OFF 1024
#define SMEM_AKK_K_STAGE_BYTES 32768
#define SMEM_AKK_K_STRIDE 98304
#define SMEM_KB_MN_OFF 66560
#define SMEM_KB_MN_STAGE_BYTES 32768
#define SMEM_KB_MN_STRIDE 98304
#define SMEM_WORK_RESP_OFF 197632
#define SMEM_WORK_RESP_STAGE_BYTES 16
#define SMEM_WORK_RESP_STRIDE 16
#define SMEM_TOTAL 197760
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
kernel_cake_kda_train_1563b1b6e66464bbfa33(const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap kb_tma, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ w_out, int num_heads, int num_pairs)
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
    #define work_full_addr (mbar_base + 72)
    #define work_empty_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* akk_blocks = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int akk_blocks_addr = smem + 1024;
    __nv_bfloat16* kb_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int kb_s_addr = smem + 66560;
    __nv_bfloat16* akk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int akk_k_addr = smem + 1024;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int kb_mn_addr = smem + 66560;
    unsigned int* work_resp = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int work_resp_addr = smem + 197632;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

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
            // --- pipeline 'work' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // work_empty: 4 barriers, init_count=289
            mbarrier_init(smem + 104, 289);
            mbarrier_init(smem + 112, 289);
            mbarrier_init(smem + 120, 289);
            mbarrier_init(smem + 128, 289);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 0) {
        int _tmem_hold = smem + 136;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_w = taddr;

    // ---- Role: epi_lo ----
    if (warp <= 3) {
        { // epi_lo_main
            const int epi_warp_1_0 = warp;
            const int epi_tid_1_0 = epi_warp_1_0 * 32 + lane;
            unsigned int zeros_1_0[4];
            zeros_1_0[0] = 0;
            zeros_1_0[1] = 0;
            zeros_1_0[2] = 0;
            zeros_1_0[3] = 0;
            #pragma unroll
            for (int s_1_0 = 0; s_1_0 < 2; s_1_0++) {
                #pragma unroll
                for (int blk_1_0 = 1; blk_1_0 < 3; blk_1_0++) {
                    int base_1_0 = akk_blocks_addr + (unsigned int)((s_1_0 * 12 + blk_1_0) * 8192);
                    #pragma unroll
                    for (int j_1_0 = 0; j_1_0 < 2; j_1_0++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base_1_0 + (j_1_0 * 256 + epi_tid_1_0) * 16), "r"(zeros_1_0[0]), "r"(zeros_1_0[1]), "r"(zeros_1_0[2]), "r"(zeros_1_0[3]) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(zero_done_addr);
            unsigned int epi_buf_1_0 = 0;
            unsigned int work_stage_e_1_0 = 0;
            int num_items_e_1_0 = num_pairs * num_heads;
            const int quad_1_0 = (epi_warp_1_0 & 3) * 32;
            const int row_base_1_0 = quad_1_0 << 16;
            int it_e_1_0 = (int)bid;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _item_1_0 = 0; _item_1_0 < num_items_e_1_0; _item_1_0++) {
                mbarrier_wait(work_full_addr + (work_stage_e_1_0) * 8, _phase_work_full);
                uint32_t valid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid)
                    : "r"(work_resp_addr + work_stage_e_1_0 * 16 + 0 * 16)
                    : "memory");
                uint32_t cta = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta)
                    : "r"(work_resp_addr + work_stage_e_1_0 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_e_1_0) * 8);
                work_stage_e_1_0 += 1;
                if (work_stage_e_1_0 == 4) { work_stage_e_1_0 = 0; _phase_work_full ^= 1; }
                int pair_e_1_0 = it_e_1_0 / num_heads;
                int head_e_1_0 = it_e_1_0 - pair_e_1_0 * num_heads;
                int grow_1_0 = pair_e_1_0 * 128 + quad_1_0 + lane;
                long long out_base_1_0 = ((long long)grow_1_0 * (long long)num_heads + (long long)head_e_1_0) * 128;
                int buf_col_1_0 = (int)epi_buf_1_0 * 128;
                mbarrier_wait(acc_full_addr + (epi_buf_1_0) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)row_base_1_0 + (unsigned int)buf_col_1_0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)row_base_1_0 + (unsigned int)buf_col_1_0 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                    unsigned int _packed_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 0], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 2], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 4], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 6], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 8], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 8]));
                        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 10], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 10]));
                        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 12], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 12]));
                        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + j_1_1 * 16)[_lp*2 + 14], (_tmem_load_0 + j_1_1 * 16)[_lp*2+1 + 14]));
                        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(w_out + (out_base_1_0 + (long long)(j_1_1 * 16)))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr + (epi_buf_1_0) * 8);
                epi_buf_1_0 += 1;
                if (epi_buf_1_0 == 2) { epi_buf_1_0 = 0; _phase_acc_full ^= 1; }
                if (valid == 0) {
                    break;
                }
                it_e_1_0 = (int)cta;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: epi_hi ----
    if (warp >= 4 && warp <= 7) {
        { // epi_hi_main
            const int epi_warp_1_1 = warp;
            const int epi_tid_1_1 = epi_warp_1_1 * 32 + lane;
            unsigned int zeros_1_1[4];
            zeros_1_1[0] = 0;
            zeros_1_1[1] = 0;
            zeros_1_1[2] = 0;
            zeros_1_1[3] = 0;
            #pragma unroll
            for (int s_1_1 = 0; s_1_1 < 2; s_1_1++) {
                #pragma unroll
                for (int blk_1_1 = 1; blk_1_1 < 3; blk_1_1++) {
                    int base_1_1 = akk_blocks_addr + (unsigned int)((s_1_1 * 12 + blk_1_1) * 8192);
                    #pragma unroll
                    for (int j_1_2 = 0; j_1_2 < 2; j_1_2++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base_1_1 + (j_1_2 * 256 + epi_tid_1_1) * 16), "r"(zeros_1_1[0]), "r"(zeros_1_1[1]), "r"(zeros_1_1[2]), "r"(zeros_1_1[3]) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(zero_done_addr);
            unsigned int epi_buf_1_1 = 0;
            unsigned int work_stage_e_1_1 = 0;
            int num_items_e_1_1 = num_pairs * num_heads;
            const int quad_1_1 = (epi_warp_1_1 & 3) * 32;
            const int row_base_1_1 = quad_1_1 << 16;
            int it_e_1_1 = (int)bid;
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_acc_full_1 = 0;
            #pragma unroll 1
            for (int _item_1_1 = 0; _item_1_1 < num_items_e_1_1; _item_1_1++) {
                mbarrier_wait(work_full_addr + (work_stage_e_1_1) * 8, _phase_work_full_1);
                uint32_t valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid_1)
                    : "r"(work_resp_addr + work_stage_e_1_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t cta_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta_1)
                    : "r"(work_resp_addr + work_stage_e_1_1 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_e_1_1) * 8);
                work_stage_e_1_1 += 1;
                if (work_stage_e_1_1 == 4) { work_stage_e_1_1 = 0; _phase_work_full_1 ^= 1; }
                int pair_e_1_1 = it_e_1_1 / num_heads;
                int head_e_1_1 = it_e_1_1 - pair_e_1_1 * num_heads;
                int grow_1_1 = pair_e_1_1 * 128 + quad_1_1 + lane;
                long long out_base_1_1 = ((long long)grow_1_1 * (long long)num_heads + (long long)head_e_1_1) * 128;
                int buf_col_1_1 = (int)epi_buf_1_1 * 128;
                mbarrier_wait(acc_full_addr + (epi_buf_1_1) * 8, _phase_acc_full_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(taddr + 64 + (unsigned int)row_base_1_1 + (unsigned int)buf_col_1_1));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(taddr + 64 + (unsigned int)row_base_1_1 + (unsigned int)buf_col_1_1 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int j_1_3 = 0; j_1_3 < 4; j_1_3++) {
                    unsigned int _packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 0], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 2], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 4], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 6], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 8], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 8]));
                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 10], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 10]));
                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 12], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 12]));
                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + j_1_3 * 16)[_lp*2 + 14], (_tmem_load_1 + j_1_3 * 16)[_lp*2+1 + 14]));
                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(w_out + (out_base_1_1 + (long long)(64 + j_1_3 * 16)))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr + (epi_buf_1_1) * 8);
                epi_buf_1_1 += 1;
                if (epi_buf_1_1 == 2) { epi_buf_1_1 = 0; _phase_acc_full_1 ^= 1; }
                if (valid_1 == 0) {
                    break;
                }
                it_e_1_1 = (int)cta_1;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            unsigned int ld_stage = 0;
            unsigned int work_stage_l = 0;
            int num_items_l = num_pairs * num_heads;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            if (elect_sync()) {
                int it_l = (int)bid;
                int pair_f = it_l / num_heads;
                int chunk_f0 = pair_f * 2;
                int chunk_f1 = pair_f * 2 + 1;
                int nbos0 = chunk_bos[chunk_f0];
                int nbos1 = chunk_bos[chunk_f1];
                #pragma unroll 1
                for (int _item_l = 0; _item_l < num_items_l; _item_l++) {
                    mbarrier_wait(work_empty_addr + (work_stage_l) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_l) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_resp_addr + work_stage_l * 16 + 0 * 16), "r"(work_full_addr + work_stage_l * 8)
                        : "memory");
                    int pair_l = it_l / num_heads;
                    int head_l = it_l - pair_l * num_heads;
                    int row0_l = pair_l * 128;
                    int bos0_l = nbos0;
                    int bos1_l = nbos1;
                    mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 49152);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kb_s_addr + ld_stage * 98304 + (unsigned int)(seg * 16384), (&kb_tma), seg * 64, head_l, row0_l, tiles_full_addr + (ld_stage) * 8);
                    }
                    tma_3d_gmem2smem(akk_k_addr + ld_stage * 98304, (&akk_tma), 0, head_l, bos0_l, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(akk_k_addr + ld_stage * 98304 + 24576, (&akk_tma), 0, head_l, bos1_l, tiles_full_addr + (ld_stage) * 8);
                    ld_stage += 1;
                    if (ld_stage == 2) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_2);
                    uint32_t valid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(valid_2)
                        : "r"(work_resp_addr + work_stage_l * 16 + 0 * 16)
                        : "memory");
                    uint32_t cta_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(cta_2)
                        : "r"(work_resp_addr + work_stage_l * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                    work_stage_l += 1;
                    if (work_stage_l == 4) { work_stage_l = 0; _phase_work_empty ^= 1; _phase_work_full_2 ^= 1; }
                    if (valid_2 == 0) {
                        break;
                    }
                    it_l = (int)cta_2;
                    int pair_n = it_l / num_heads;
                    nbos0 = chunk_bos[pair_n * 2];
                    nbos1 = chunk_bos[pair_n * 2 + 1];
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_buf = 0;
            unsigned int work_stage_m = 0;
            int num_items_m = num_pairs * num_heads;
            const int lane_m = lane;
            unsigned int _phase_zero_done_0 = 0;
            mbarrier_wait(zero_done_addr, _phase_zero_done_0);
            _phase_zero_done_0 ^= 1;
            int it_w = (int)bid;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_acc_empty = 1;
            #pragma unroll 1
            for (int _item_w = 0; _item_w < num_items_m; _item_w++) {
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_3);
                uint32_t valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid_3)
                    : "r"(work_resp_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                uint32_t cta_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta_3)
                    : "r"(work_resp_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_3 ^= 1; }
                int pair_w = it_w / num_heads;
                int clen_w0 = chunk_len[pair_w * 2];
                int clen_w1 = chunk_len[pair_w * 2 + 1];
                mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full);
                if (clen_w0 + clen_w1 < 128) {
                    unsigned int zeros[4];
                    zeros[0] = 0;
                    zeros[1] = 0;
                    zeros[2] = 0;
                    zeros[3] = 0;
                    int npad = (64 - clen_w0) * 8;
                    #pragma unroll 1
                    for (int idx = lane_m; idx < npad; idx += 32) {
                        int row = clen_w0 + (idx >> 3);
                        int off = row * 128 + (idx & 7) * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(akk_k_addr + mma_stage * 98304 + (unsigned int)off), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                    }
                    unsigned int zeros_0[4];
                    zeros_0[0] = 0;
                    zeros_0[1] = 0;
                    zeros_0[2] = 0;
                    zeros_0[3] = 0;
                    int npad_0 = (64 - clen_w1) * 8;
                    #pragma unroll 1
                    for (int idx_1 = lane_m; idx_1 < npad_0; idx_1 += 32) {
                        int row_1 = clen_w1 + (idx_1 >> 3);
                        int off_1 = row_1 * 128 + (idx_1 & 7) * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(akk_k_addr + mma_stage * 98304 + 24576 + (unsigned int)off_1), "r"(zeros_0[0]), "r"(zeros_0[1]), "r"(zeros_0[2]), "r"(zeros_0[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((akk_k_addr) >> 4) & 0x7FFF) + (mma_stage) * 6144;
                    int _smem_desc_lo_1 = ((((kb_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mma_stage) * 6144;
#else
                    int _smem_desc_lo_0 = (((akk_k_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
                    int _smem_desc_lo_1 = ((((kb_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc_w + (mma_buf * 128))), "r"(0) : "memory");
                    tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                    tcgen05_commit(acc_full_addr + (mma_buf) * 8);
                }
                mma_stage += 1;
                if (mma_stage == 2) { mma_stage = 0; _phase_tiles_full ^= 1; }
                mma_buf += 1;
                if (mma_buf == 2) { mma_buf = 0; _phase_acc_empty ^= 1; }
                if (valid_3 == 0) {
                    break;
                }
                it_w = (int)cta_3;
            }
        }
    }

    // Cleanup
}

} // extern "C"
