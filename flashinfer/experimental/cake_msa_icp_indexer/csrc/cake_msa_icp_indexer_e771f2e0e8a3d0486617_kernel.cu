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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 2
#define NUM_K_PIPE_STAGES 8
#define NUM_ACC_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 66560
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_TOTAL 197632
#define THREADS 320

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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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
#if __CUDA_ARCH__ == 1030
#else

#endif


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_msa_icp_indexer_e771f2e0e8a3d0486617(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, int* __restrict__ block_table, float* __restrict__ max_score, int* __restrict__ qo_lens, int* __restrict__ kv_lens, int* __restrict__ qo_indptr, int* __restrict__ qo_offset, long long* __restrict__ work_range, long long* __restrict__ work_info, int rank, int compact, int bt_stride, long long score_stride_t, long long score_stride_h, int score_cols)
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
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 32)
    #define k_empty_addr (mbar_base + 96)
    #define acc_full_addr (mbar_base + 160)
    #define acc_empty_addr (mbar_base + 192)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 8 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // k_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 4 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // acc_empty: 4 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epi ----
    if (warp <= 7) {
        { // epi_main
            long long wr = work_range[blockIdx.x];
            int j0 = (int)(wr & 65535);
            int j1 = (int)(wr >> 16 & 65535);
            int c0 = (int)(wr >> 32 & 65535);
            int c1 = (int)(wr >> 48 & 65535);
            int j_end = ((c1 > 0) ? j1 + 1 : j1);
            int wg = warp / 4;
            unsigned int acc = 0;
            int quarter = warp % 4;
            int lane_1 = lane;
            int lrow = quarter * 32 + lane_1;
            int head = lrow % 4;
            unsigned int lane_base = (unsigned int)(quarter * 32) << 16;
            int own = rank * 2 / 2;
            int wcols = 1;
            if (compact == 0) {
                wcols = 2;
            }
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int j = j0; j < j_end; j++) {
                long long wi = work_info[j];
                long long prev = 0;
                if (j > 0) {
                    prev = work_info[j - 1] >> 32;
                }
                int b = (int)(wi & 65535);
                int it = (int)(wi >> 16 & 65535);
                int ce = (int)((wi >> 32) - prev);
                if (j == j1) {
                    ce = c1;
                }
                int cb = ((j == j0) ? c0 : 0);
                int kt0 = cb / 2;
                int kt1 = (ce + 2 - 1) / 2;
                if (kt1 > kt0) {
                    int q_len = qo_lens[b];
                    int kv_len = kv_lens[b];
                    int q_off = qo_offset[b];
                    int t_base = qo_indptr[b];
                    int c_first = kt0 * wcols;
                    int use_stage = 0;
                    int windows[2];
                    int r_0 = it * 64 + lrow / 4;
                    int u_0 = 0;
                    if (r_0 < q_len) {
                        int _min_0 = ((kv_len) < (q_off + r_0 + 1) ? (kv_len) : (q_off + r_0 + 1));
                        u_0 = _min_0;
                    }
                    int cpre_0 = (((u_0 + 127) / 128) < (score_cols) ? ((u_0 + 127) / 128) : (score_cols));
                    if (compact != 0) {
                        int nvb_0 = (u_0 + 127) / 128;
                        cpre_0 = 0;
                        if (nvb_0 > own) {
                            int _min_1 = (((nvb_0 - 1 - own) / 2 + 1) < (score_cols) ? ((nvb_0 - 1 - own) / 2 + 1) : (score_cols));
                            cpre_0 = _min_1;
                        }
                    }
                    windows[0] = c_first - c_first % 32;
                    long long off_0 = (long long)(t_base + r_0) * score_stride_t + (long long)head * score_stride_h;
                    int r_1 = it * 64 + 32 + lrow / 4;
                    int u_1 = 0;
                    if (r_1 < q_len) {
                        int _min_2 = ((kv_len) < (q_off + r_1 + 1) ? (kv_len) : (q_off + r_1 + 1));
                        u_1 = _min_2;
                    }
                    int cpre_1 = (((u_1 + 127) / 128) < (score_cols) ? ((u_1 + 127) / 128) : (score_cols));
                    if (compact != 0) {
                        int nvb_1 = (u_1 + 127) / 128;
                        cpre_1 = 0;
                        if (nvb_1 > own) {
                            int _min_3 = (((nvb_1 - 1 - own) / 2 + 1) < (score_cols) ? ((nvb_1 - 1 - own) / 2 + 1) : (score_cols));
                            cpre_1 = _min_3;
                        }
                    }
                    windows[1] = c_first - c_first % 32;
                    long long off_1 = (long long)(t_base + r_1) * score_stride_t + (long long)head * score_stride_h;
                    if (use_stage == 0) {
                        #pragma unroll 1
                        for (int kt_0 = kt0; kt_0 < kt1; kt_0++) {
                            int mine_0 = 1;
                            mine_0 = ((0 == wg) ? 1 : 0);
                            if (mine_0 != 0) {
                                mbarrier_wait(acc_full_addr + ((unsigned int)wg + acc) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float v_0[128];
                                unsigned int tile_addr_0 = taddr + lane_base + ((unsigned int)wg + acc) * 128;
#if __CUDA_ARCH__ == 1030
                                float pm_0 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
#else
#endif
                                asm volatile(
#if __CUDA_ARCH__ == 1030
                                    "tcgen05.ld.red.sync.aligned.32x32b.x128.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
                                    : "=f"(v_0[0]), "=f"(v_0[1]), "=f"(v_0[2]), "=f"(v_0[3]), "=f"(v_0[4]), "=f"(v_0[5]), "=f"(v_0[6]), "=f"(v_0[7]), "=f"(v_0[8]), "=f"(v_0[9]), "=f"(v_0[10]), "=f"(v_0[11]), "=f"(v_0[12]), "=f"(v_0[13]), "=f"(v_0[14]), "=f"(v_0[15]), "=f"(v_0[16]), "=f"(v_0[17]), "=f"(v_0[18]), "=f"(v_0[19]), "=f"(v_0[20]), "=f"(v_0[21]), "=f"(v_0[22]), "=f"(v_0[23]), "=f"(v_0[24]), "=f"(v_0[25]), "=f"(v_0[26]), "=f"(v_0[27]), "=f"(v_0[28]), "=f"(v_0[29]), "=f"(v_0[30]), "=f"(v_0[31]), "=f"(v_0[32]), "=f"(v_0[33]), "=f"(v_0[34]), "=f"(v_0[35]), "=f"(v_0[36]), "=f"(v_0[37]), "=f"(v_0[38]), "=f"(v_0[39]), "=f"(v_0[40]), "=f"(v_0[41]), "=f"(v_0[42]), "=f"(v_0[43]), "=f"(v_0[44]), "=f"(v_0[45]), "=f"(v_0[46]), "=f"(v_0[47]), "=f"(v_0[48]), "=f"(v_0[49]), "=f"(v_0[50]), "=f"(v_0[51]), "=f"(v_0[52]), "=f"(v_0[53]), "=f"(v_0[54]), "=f"(v_0[55]), "=f"(v_0[56]), "=f"(v_0[57]), "=f"(v_0[58]), "=f"(v_0[59]), "=f"(v_0[60]), "=f"(v_0[61]), "=f"(v_0[62]), "=f"(v_0[63]), "=f"(v_0[64]), "=f"(v_0[65]), "=f"(v_0[66]), "=f"(v_0[67]), "=f"(v_0[68]), "=f"(v_0[69]), "=f"(v_0[70]), "=f"(v_0[71]), "=f"(v_0[72]), "=f"(v_0[73]), "=f"(v_0[74]), "=f"(v_0[75]), "=f"(v_0[76]), "=f"(v_0[77]), "=f"(v_0[78]), "=f"(v_0[79]), "=f"(v_0[80]), "=f"(v_0[81]), "=f"(v_0[82]), "=f"(v_0[83]), "=f"(v_0[84]), "=f"(v_0[85]), "=f"(v_0[86]), "=f"(v_0[87]), "=f"(v_0[88]), "=f"(v_0[89]), "=f"(v_0[90]), "=f"(v_0[91]), "=f"(v_0[92]), "=f"(v_0[93]), "=f"(v_0[94]), "=f"(v_0[95]), "=f"(v_0[96]), "=f"(v_0[97]), "=f"(v_0[98]), "=f"(v_0[99]), "=f"(v_0[100]), "=f"(v_0[101]), "=f"(v_0[102]), "=f"(v_0[103]), "=f"(v_0[104]), "=f"(v_0[105]), "=f"(v_0[106]), "=f"(v_0[107]), "=f"(v_0[108]), "=f"(v_0[109]), "=f"(v_0[110]), "=f"(v_0[111]), "=f"(v_0[112]), "=f"(v_0[113]), "=f"(v_0[114]), "=f"(v_0[115]), "=f"(v_0[116]), "=f"(v_0[117]), "=f"(v_0[118]), "=f"(v_0[119]), "=f"(v_0[120]), "=f"(v_0[121]), "=f"(v_0[122]), "=f"(v_0[123]), "=f"(v_0[124]), "=f"(v_0[125]), "=f"(v_0[126]), "=f"(v_0[127]), "=f"(pm_0)
#else
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_0[0]), "=f"(v_0[1]), "=f"(v_0[2]), "=f"(v_0[3]), "=f"(v_0[4]), "=f"(v_0[5]), "=f"(v_0[6]), "=f"(v_0[7]), "=f"(v_0[8]), "=f"(v_0[9]), "=f"(v_0[10]), "=f"(v_0[11]), "=f"(v_0[12]), "=f"(v_0[13]), "=f"(v_0[14]), "=f"(v_0[15]), "=f"(v_0[16]), "=f"(v_0[17]), "=f"(v_0[18]), "=f"(v_0[19]), "=f"(v_0[20]), "=f"(v_0[21]), "=f"(v_0[22]), "=f"(v_0[23]), "=f"(v_0[24]), "=f"(v_0[25]), "=f"(v_0[26]), "=f"(v_0[27]), "=f"(v_0[28]), "=f"(v_0[29]), "=f"(v_0[30]), "=f"(v_0[31])
#endif
                                    : "r"(tile_addr_0 + 0));
#if __CUDA_ARCH__ == 1030
#else
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_0[32]), "=f"(v_0[33]), "=f"(v_0[34]), "=f"(v_0[35]), "=f"(v_0[36]), "=f"(v_0[37]), "=f"(v_0[38]), "=f"(v_0[39]), "=f"(v_0[40]), "=f"(v_0[41]), "=f"(v_0[42]), "=f"(v_0[43]), "=f"(v_0[44]), "=f"(v_0[45]), "=f"(v_0[46]), "=f"(v_0[47]), "=f"(v_0[48]), "=f"(v_0[49]), "=f"(v_0[50]), "=f"(v_0[51]), "=f"(v_0[52]), "=f"(v_0[53]), "=f"(v_0[54]), "=f"(v_0[55]), "=f"(v_0[56]), "=f"(v_0[57]), "=f"(v_0[58]), "=f"(v_0[59]), "=f"(v_0[60]), "=f"(v_0[61]), "=f"(v_0[62]), "=f"(v_0[63])
                                    : "r"(tile_addr_0 + 32));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_0[64]), "=f"(v_0[65]), "=f"(v_0[66]), "=f"(v_0[67]), "=f"(v_0[68]), "=f"(v_0[69]), "=f"(v_0[70]), "=f"(v_0[71]), "=f"(v_0[72]), "=f"(v_0[73]), "=f"(v_0[74]), "=f"(v_0[75]), "=f"(v_0[76]), "=f"(v_0[77]), "=f"(v_0[78]), "=f"(v_0[79]), "=f"(v_0[80]), "=f"(v_0[81]), "=f"(v_0[82]), "=f"(v_0[83]), "=f"(v_0[84]), "=f"(v_0[85]), "=f"(v_0[86]), "=f"(v_0[87]), "=f"(v_0[88]), "=f"(v_0[89]), "=f"(v_0[90]), "=f"(v_0[91]), "=f"(v_0[92]), "=f"(v_0[93]), "=f"(v_0[94]), "=f"(v_0[95])
                                    : "r"(tile_addr_0 + 64));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_0[96]), "=f"(v_0[97]), "=f"(v_0[98]), "=f"(v_0[99]), "=f"(v_0[100]), "=f"(v_0[101]), "=f"(v_0[102]), "=f"(v_0[103]), "=f"(v_0[104]), "=f"(v_0[105]), "=f"(v_0[106]), "=f"(v_0[107]), "=f"(v_0[108]), "=f"(v_0[109]), "=f"(v_0[110]), "=f"(v_0[111]), "=f"(v_0[112]), "=f"(v_0[113]), "=f"(v_0[114]), "=f"(v_0[115]), "=f"(v_0[116]), "=f"(v_0[117]), "=f"(v_0[118]), "=f"(v_0[119]), "=f"(v_0[120]), "=f"(v_0[121]), "=f"(v_0[122]), "=f"(v_0[123]), "=f"(v_0[124]), "=f"(v_0[125]), "=f"(v_0[126]), "=f"(v_0[127])
                                    : "r"(tile_addr_0 + 96));
#endif
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + ((unsigned int)wg + acc) * 8);
                                acc += 2;
                                if (acc >= 4) { acc -= 4; _phase_acc_full ^= 1; }
                                int row_stage_0 = (quarter * 32 + lane_1) * 33 - windows[0];
                                int lim_0 = cpre_0;
                                int p_0 = kt_0;
                                int nvis_0 = ((u_0 - 256 * p_0 - 128 * rank) > (0) ? (u_0 - 256 * p_0 - 128 * rank) : (0));
                                float m_0 = -CUDART_INF_F;
                                int masked_0 = 1;
#if __CUDA_ARCH__ == 1030
                                m_0 = pm_0;
                                int _warp_vote_4 = __any_sync(0xFFFFFFFF, nvis_0 < 128);
                                masked_0 = _warp_vote_4;
#else
#endif
                                if (masked_0 != 0) {
                                    m_0 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_5 = __any_sync(0xFFFFFFFF, nvis_0 < 32);
                                    if (_warp_vote_5 != 0) {
                                        uint32_t _thresh_mask_6;
#else
                                    int _warp_vote_4 = __any_sync(0xFFFFFFFF, nvis_0 < 32);
                                    if (_warp_vote_4 != 0) {
                                        uint32_t _thresh_mask_5;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_7 = nvis_0;
                                            if (_lim_7 <= 0) { _thresh_mask_6 = 0u; }
                                            else if (_lim_7 >= 32) { _thresh_mask_6 = 0xFFFFFFFFu; }
#else
                                            int _lim_6 = nvis_0;
                                            if (_lim_6 <= 0) { _thresh_mask_5 = 0u; }
                                            else if (_lim_6 >= 32) { _thresh_mask_5 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_6) : "r"(_lim_7));
#else
                                                    "}" : "=r"(_thresh_mask_5) : "r"(_lim_6));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_6 & (1u << 0))) v_0[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 1))) v_0[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 2))) v_0[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 3))) v_0[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 4))) v_0[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 5))) v_0[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 6))) v_0[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 7))) v_0[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 8))) v_0[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 9))) v_0[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 10))) v_0[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 11))) v_0[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 12))) v_0[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 13))) v_0[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 14))) v_0[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 15))) v_0[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 16))) v_0[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 17))) v_0[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 18))) v_0[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 19))) v_0[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 20))) v_0[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 21))) v_0[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 22))) v_0[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 23))) v_0[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 24))) v_0[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 25))) v_0[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 26))) v_0[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 27))) v_0[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 28))) v_0[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 29))) v_0[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 30))) v_0[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_6 & (1u << 31))) v_0[31] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_5 & (1u << 0))) v_0[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 1))) v_0[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 2))) v_0[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 3))) v_0[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 4))) v_0[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 5))) v_0[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 6))) v_0[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 7))) v_0[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 8))) v_0[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 9))) v_0[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 10))) v_0[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 11))) v_0[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 12))) v_0[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 13))) v_0[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 14))) v_0[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 15))) v_0[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 16))) v_0[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 17))) v_0[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 18))) v_0[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 19))) v_0[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 20))) v_0[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 21))) v_0[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 22))) v_0[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 23))) v_0[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 24))) v_0[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 25))) v_0[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 26))) v_0[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 27))) v_0[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 28))) v_0[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 29))) v_0[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 30))) v_0[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_5 & (1u << 31))) v_0[31] = -CUDART_INF_F;
#endif
                                    }
                                    float part_0 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_9 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[0], v_0[1]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[2], v_0[3]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[4], v_0[5]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[6], v_0[7]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[8], v_0[9]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[10], v_0[11]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[12], v_0[13]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[14], v_0[15]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[16], v_0[17]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[18], v_0[19]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[20], v_0[21]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[22], v_0[23]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[24], v_0[25]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[26], v_0[27]));
                                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(v_0[28], v_0[29]));
                                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(v_0[30], v_0[31]));
                                    float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                                    part_0 = _reg_array_reduce_8;
#else
                                    float2 _reg_reduce_max2_8 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[0], v_0[1]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[2], v_0[3]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[4], v_0[5]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[6], v_0[7]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[8], v_0[9]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[10], v_0[11]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[12], v_0[13]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[14], v_0[15]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[16], v_0[17]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[18], v_0[19]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[20], v_0[21]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[22], v_0[23]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[24], v_0[25]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[26], v_0[27]));
                                    _reg_reduce_max2_8.x = max_noftz(_reg_reduce_max2_8.x, max_noftz(v_0[28], v_0[29]));
                                    _reg_reduce_max2_8.y = max_noftz(_reg_reduce_max2_8.y, max_noftz(v_0[30], v_0[31]));
                                    float _reg_array_reduce_7 = row_max_reduce(_reg_reduce_max2_8);
                                    part_0 = _reg_array_reduce_7;
#endif
                                    m_0 = fmaxf(m_0, part_0);
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_10 = __any_sync(0xFFFFFFFF, nvis_0 < 64);
                                    if (_warp_vote_10 != 0) {
                                        uint32_t _thresh_mask_11;
#else
                                    int _warp_vote_9 = __any_sync(0xFFFFFFFF, nvis_0 < 64);
                                    if (_warp_vote_9 != 0) {
                                        uint32_t _thresh_mask_10;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_12 = nvis_0 - 32;
                                            if (_lim_12 <= 0) { _thresh_mask_11 = 0u; }
                                            else if (_lim_12 >= 32) { _thresh_mask_11 = 0xFFFFFFFFu; }
#else
                                            int _lim_11 = nvis_0 - 32;
                                            if (_lim_11 <= 0) { _thresh_mask_10 = 0u; }
                                            else if (_lim_11 >= 32) { _thresh_mask_10 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_11) : "r"(_lim_12));
#else
                                                    "}" : "=r"(_thresh_mask_10) : "r"(_lim_11));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_11 & (1u << 0))) v_0[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 1))) v_0[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 2))) v_0[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 3))) v_0[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 4))) v_0[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 5))) v_0[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 6))) v_0[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 7))) v_0[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 8))) v_0[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 9))) v_0[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 10))) v_0[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 11))) v_0[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 12))) v_0[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 13))) v_0[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 14))) v_0[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 15))) v_0[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 16))) v_0[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 17))) v_0[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 18))) v_0[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 19))) v_0[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 20))) v_0[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 21))) v_0[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 22))) v_0[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 23))) v_0[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 24))) v_0[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 25))) v_0[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 26))) v_0[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 27))) v_0[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 28))) v_0[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 29))) v_0[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 30))) v_0[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_11 & (1u << 31))) v_0[63] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_10 & (1u << 0))) v_0[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 1))) v_0[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 2))) v_0[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 3))) v_0[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 4))) v_0[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 5))) v_0[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 6))) v_0[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 7))) v_0[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 8))) v_0[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 9))) v_0[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 10))) v_0[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 11))) v_0[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 12))) v_0[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 13))) v_0[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 14))) v_0[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 15))) v_0[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 16))) v_0[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 17))) v_0[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 18))) v_0[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 19))) v_0[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 20))) v_0[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 21))) v_0[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 22))) v_0[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 23))) v_0[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 24))) v_0[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 25))) v_0[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 26))) v_0[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 27))) v_0[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 28))) v_0[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 29))) v_0[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 30))) v_0[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 31))) v_0[63] = -CUDART_INF_F;
#endif
                                    }
                                    float part_1 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_14 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(32) + 0], _reg_reduce_max2_14);
                                    float _reg_array_reduce_13 = row_max_reduce(_reg_reduce_max2_14);
                                    part_1 = _reg_array_reduce_13;
#else
                                    float2 _reg_reduce_max2_13 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(32) + 0], _reg_reduce_max2_13);
                                    float _reg_array_reduce_12 = row_max_reduce(_reg_reduce_max2_13);
                                    part_1 = _reg_array_reduce_12;
#endif
                                    m_0 = fmaxf(m_0, part_1);
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_15 = __any_sync(0xFFFFFFFF, nvis_0 < 96);
                                    if (_warp_vote_15 != 0) {
                                        uint32_t _thresh_mask_16;
#else
                                    int _warp_vote_14 = __any_sync(0xFFFFFFFF, nvis_0 < 96);
                                    if (_warp_vote_14 != 0) {
                                        uint32_t _thresh_mask_15;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_17 = nvis_0 - 64;
                                            if (_lim_17 <= 0) { _thresh_mask_16 = 0u; }
                                            else if (_lim_17 >= 32) { _thresh_mask_16 = 0xFFFFFFFFu; }
#else
                                            int _lim_16 = nvis_0 - 64;
                                            if (_lim_16 <= 0) { _thresh_mask_15 = 0u; }
                                            else if (_lim_16 >= 32) { _thresh_mask_15 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_16) : "r"(_lim_17));
#else
                                                    "}" : "=r"(_thresh_mask_15) : "r"(_lim_16));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_16 & (1u << 0))) v_0[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 1))) v_0[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 2))) v_0[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 3))) v_0[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 4))) v_0[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 5))) v_0[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 6))) v_0[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 7))) v_0[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 8))) v_0[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 9))) v_0[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 10))) v_0[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 11))) v_0[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 12))) v_0[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 13))) v_0[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 14))) v_0[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 15))) v_0[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 16))) v_0[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 17))) v_0[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 18))) v_0[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 19))) v_0[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 20))) v_0[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 21))) v_0[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 22))) v_0[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 23))) v_0[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 24))) v_0[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 25))) v_0[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 26))) v_0[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 27))) v_0[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 28))) v_0[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 29))) v_0[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 30))) v_0[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 31))) v_0[95] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_15 & (1u << 0))) v_0[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 1))) v_0[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 2))) v_0[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 3))) v_0[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 4))) v_0[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 5))) v_0[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 6))) v_0[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 7))) v_0[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 8))) v_0[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 9))) v_0[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 10))) v_0[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 11))) v_0[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 12))) v_0[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 13))) v_0[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 14))) v_0[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 15))) v_0[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 16))) v_0[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 17))) v_0[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 18))) v_0[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 19))) v_0[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 20))) v_0[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 21))) v_0[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 22))) v_0[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 23))) v_0[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 24))) v_0[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 25))) v_0[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 26))) v_0[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 27))) v_0[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 28))) v_0[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 29))) v_0[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 30))) v_0[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_15 & (1u << 31))) v_0[95] = -CUDART_INF_F;
#endif
                                    }
                                    float part_2 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_19 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(64) + 0], _reg_reduce_max2_19);
                                    float _reg_array_reduce_18 = row_max_reduce(_reg_reduce_max2_19);
                                    part_2 = _reg_array_reduce_18;
#else
                                    float2 _reg_reduce_max2_18 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(64) + 0], _reg_reduce_max2_18);
                                    float _reg_array_reduce_17 = row_max_reduce(_reg_reduce_max2_18);
                                    part_2 = _reg_array_reduce_17;
#endif
                                    m_0 = fmaxf(m_0, part_2);
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_20 = __any_sync(0xFFFFFFFF, nvis_0 < 128);
                                    if (_warp_vote_20 != 0) {
                                        uint32_t _thresh_mask_21;
#else
                                    int _warp_vote_19 = __any_sync(0xFFFFFFFF, nvis_0 < 128);
                                    if (_warp_vote_19 != 0) {
                                        uint32_t _thresh_mask_20;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_22 = nvis_0 - 96;
                                            if (_lim_22 <= 0) { _thresh_mask_21 = 0u; }
                                            else if (_lim_22 >= 32) { _thresh_mask_21 = 0xFFFFFFFFu; }
#else
                                            int _lim_21 = nvis_0 - 96;
                                            if (_lim_21 <= 0) { _thresh_mask_20 = 0u; }
                                            else if (_lim_21 >= 32) { _thresh_mask_20 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_21) : "r"(_lim_22));
#else
                                                    "}" : "=r"(_thresh_mask_20) : "r"(_lim_21));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_21 & (1u << 0))) v_0[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 1))) v_0[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 2))) v_0[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 3))) v_0[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 4))) v_0[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 5))) v_0[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 6))) v_0[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 7))) v_0[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 8))) v_0[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 9))) v_0[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 10))) v_0[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 11))) v_0[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 12))) v_0[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 13))) v_0[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 14))) v_0[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 15))) v_0[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 16))) v_0[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 17))) v_0[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 18))) v_0[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 19))) v_0[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 20))) v_0[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 21))) v_0[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 22))) v_0[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 23))) v_0[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 24))) v_0[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 25))) v_0[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 26))) v_0[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 27))) v_0[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 28))) v_0[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 29))) v_0[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 30))) v_0[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_21 & (1u << 31))) v_0[127] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_20 & (1u << 0))) v_0[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 1))) v_0[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 2))) v_0[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 3))) v_0[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 4))) v_0[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 5))) v_0[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 6))) v_0[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 7))) v_0[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 8))) v_0[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 9))) v_0[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 10))) v_0[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 11))) v_0[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 12))) v_0[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 13))) v_0[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 14))) v_0[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 15))) v_0[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 16))) v_0[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 17))) v_0[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 18))) v_0[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 19))) v_0[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 20))) v_0[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 21))) v_0[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 22))) v_0[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 23))) v_0[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 24))) v_0[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 25))) v_0[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 26))) v_0[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 27))) v_0[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 28))) v_0[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 29))) v_0[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 30))) v_0[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_20 & (1u << 31))) v_0[127] = -CUDART_INF_F;
#endif
                                    }
                                    float part_3 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_24 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(96) + 0], _reg_reduce_max2_24);
                                    float _reg_array_reduce_23 = row_max_reduce(_reg_reduce_max2_24);
                                    part_3 = _reg_array_reduce_23;
#else
                                    float2 _reg_reduce_max2_23 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(96) + 0], _reg_reduce_max2_23);
                                    float _reg_array_reduce_22 = row_max_reduce(_reg_reduce_max2_23);
                                    part_3 = _reg_array_reduce_22;
#endif
                                    m_0 = fmaxf(m_0, part_3);
                                }
                                m_0 = ((m_0 == 0.0f) ? 0.0f : m_0);
                                if (compact != 0) {
                                    if (p_0 < lim_0) {
                                        max_score[off_0 + (long long)p_0] = m_0;
                                    }
                                } else {
                                    if (2 * p_0 + own < lim_0) {
                                        max_score[off_0 + (long long)(2 * p_0 + own)] = m_0;
                                    }
                                    if (2 * p_0 + 1 - own < lim_0) {
                                        max_score[off_0 + (long long)(2 * p_0 + 1 - own)] = -CUDART_INF_F;
                                    }
                                }
                            }
                            int mine_1 = 1;
                            mine_1 = ((1 == wg) ? 1 : 0);
                            if (mine_1 != 0) {
                                mbarrier_wait(acc_full_addr + ((unsigned int)wg + acc) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float v_1[128];
                                unsigned int tile_addr_1 = taddr + lane_base + ((unsigned int)wg + acc) * 128;
#if __CUDA_ARCH__ == 1030
                                float pm_1 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
#else
#endif
                                asm volatile(
#if __CUDA_ARCH__ == 1030
                                    "tcgen05.ld.red.sync.aligned.32x32b.x128.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
                                    : "=f"(v_1[0]), "=f"(v_1[1]), "=f"(v_1[2]), "=f"(v_1[3]), "=f"(v_1[4]), "=f"(v_1[5]), "=f"(v_1[6]), "=f"(v_1[7]), "=f"(v_1[8]), "=f"(v_1[9]), "=f"(v_1[10]), "=f"(v_1[11]), "=f"(v_1[12]), "=f"(v_1[13]), "=f"(v_1[14]), "=f"(v_1[15]), "=f"(v_1[16]), "=f"(v_1[17]), "=f"(v_1[18]), "=f"(v_1[19]), "=f"(v_1[20]), "=f"(v_1[21]), "=f"(v_1[22]), "=f"(v_1[23]), "=f"(v_1[24]), "=f"(v_1[25]), "=f"(v_1[26]), "=f"(v_1[27]), "=f"(v_1[28]), "=f"(v_1[29]), "=f"(v_1[30]), "=f"(v_1[31]), "=f"(v_1[32]), "=f"(v_1[33]), "=f"(v_1[34]), "=f"(v_1[35]), "=f"(v_1[36]), "=f"(v_1[37]), "=f"(v_1[38]), "=f"(v_1[39]), "=f"(v_1[40]), "=f"(v_1[41]), "=f"(v_1[42]), "=f"(v_1[43]), "=f"(v_1[44]), "=f"(v_1[45]), "=f"(v_1[46]), "=f"(v_1[47]), "=f"(v_1[48]), "=f"(v_1[49]), "=f"(v_1[50]), "=f"(v_1[51]), "=f"(v_1[52]), "=f"(v_1[53]), "=f"(v_1[54]), "=f"(v_1[55]), "=f"(v_1[56]), "=f"(v_1[57]), "=f"(v_1[58]), "=f"(v_1[59]), "=f"(v_1[60]), "=f"(v_1[61]), "=f"(v_1[62]), "=f"(v_1[63]), "=f"(v_1[64]), "=f"(v_1[65]), "=f"(v_1[66]), "=f"(v_1[67]), "=f"(v_1[68]), "=f"(v_1[69]), "=f"(v_1[70]), "=f"(v_1[71]), "=f"(v_1[72]), "=f"(v_1[73]), "=f"(v_1[74]), "=f"(v_1[75]), "=f"(v_1[76]), "=f"(v_1[77]), "=f"(v_1[78]), "=f"(v_1[79]), "=f"(v_1[80]), "=f"(v_1[81]), "=f"(v_1[82]), "=f"(v_1[83]), "=f"(v_1[84]), "=f"(v_1[85]), "=f"(v_1[86]), "=f"(v_1[87]), "=f"(v_1[88]), "=f"(v_1[89]), "=f"(v_1[90]), "=f"(v_1[91]), "=f"(v_1[92]), "=f"(v_1[93]), "=f"(v_1[94]), "=f"(v_1[95]), "=f"(v_1[96]), "=f"(v_1[97]), "=f"(v_1[98]), "=f"(v_1[99]), "=f"(v_1[100]), "=f"(v_1[101]), "=f"(v_1[102]), "=f"(v_1[103]), "=f"(v_1[104]), "=f"(v_1[105]), "=f"(v_1[106]), "=f"(v_1[107]), "=f"(v_1[108]), "=f"(v_1[109]), "=f"(v_1[110]), "=f"(v_1[111]), "=f"(v_1[112]), "=f"(v_1[113]), "=f"(v_1[114]), "=f"(v_1[115]), "=f"(v_1[116]), "=f"(v_1[117]), "=f"(v_1[118]), "=f"(v_1[119]), "=f"(v_1[120]), "=f"(v_1[121]), "=f"(v_1[122]), "=f"(v_1[123]), "=f"(v_1[124]), "=f"(v_1[125]), "=f"(v_1[126]), "=f"(v_1[127]), "=f"(pm_1)
#else
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_1[0]), "=f"(v_1[1]), "=f"(v_1[2]), "=f"(v_1[3]), "=f"(v_1[4]), "=f"(v_1[5]), "=f"(v_1[6]), "=f"(v_1[7]), "=f"(v_1[8]), "=f"(v_1[9]), "=f"(v_1[10]), "=f"(v_1[11]), "=f"(v_1[12]), "=f"(v_1[13]), "=f"(v_1[14]), "=f"(v_1[15]), "=f"(v_1[16]), "=f"(v_1[17]), "=f"(v_1[18]), "=f"(v_1[19]), "=f"(v_1[20]), "=f"(v_1[21]), "=f"(v_1[22]), "=f"(v_1[23]), "=f"(v_1[24]), "=f"(v_1[25]), "=f"(v_1[26]), "=f"(v_1[27]), "=f"(v_1[28]), "=f"(v_1[29]), "=f"(v_1[30]), "=f"(v_1[31])
#endif
                                    : "r"(tile_addr_1 + 0));
#if __CUDA_ARCH__ == 1030
#else
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_1[32]), "=f"(v_1[33]), "=f"(v_1[34]), "=f"(v_1[35]), "=f"(v_1[36]), "=f"(v_1[37]), "=f"(v_1[38]), "=f"(v_1[39]), "=f"(v_1[40]), "=f"(v_1[41]), "=f"(v_1[42]), "=f"(v_1[43]), "=f"(v_1[44]), "=f"(v_1[45]), "=f"(v_1[46]), "=f"(v_1[47]), "=f"(v_1[48]), "=f"(v_1[49]), "=f"(v_1[50]), "=f"(v_1[51]), "=f"(v_1[52]), "=f"(v_1[53]), "=f"(v_1[54]), "=f"(v_1[55]), "=f"(v_1[56]), "=f"(v_1[57]), "=f"(v_1[58]), "=f"(v_1[59]), "=f"(v_1[60]), "=f"(v_1[61]), "=f"(v_1[62]), "=f"(v_1[63])
                                    : "r"(tile_addr_1 + 32));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_1[64]), "=f"(v_1[65]), "=f"(v_1[66]), "=f"(v_1[67]), "=f"(v_1[68]), "=f"(v_1[69]), "=f"(v_1[70]), "=f"(v_1[71]), "=f"(v_1[72]), "=f"(v_1[73]), "=f"(v_1[74]), "=f"(v_1[75]), "=f"(v_1[76]), "=f"(v_1[77]), "=f"(v_1[78]), "=f"(v_1[79]), "=f"(v_1[80]), "=f"(v_1[81]), "=f"(v_1[82]), "=f"(v_1[83]), "=f"(v_1[84]), "=f"(v_1[85]), "=f"(v_1[86]), "=f"(v_1[87]), "=f"(v_1[88]), "=f"(v_1[89]), "=f"(v_1[90]), "=f"(v_1[91]), "=f"(v_1[92]), "=f"(v_1[93]), "=f"(v_1[94]), "=f"(v_1[95])
                                    : "r"(tile_addr_1 + 64));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(v_1[96]), "=f"(v_1[97]), "=f"(v_1[98]), "=f"(v_1[99]), "=f"(v_1[100]), "=f"(v_1[101]), "=f"(v_1[102]), "=f"(v_1[103]), "=f"(v_1[104]), "=f"(v_1[105]), "=f"(v_1[106]), "=f"(v_1[107]), "=f"(v_1[108]), "=f"(v_1[109]), "=f"(v_1[110]), "=f"(v_1[111]), "=f"(v_1[112]), "=f"(v_1[113]), "=f"(v_1[114]), "=f"(v_1[115]), "=f"(v_1[116]), "=f"(v_1[117]), "=f"(v_1[118]), "=f"(v_1[119]), "=f"(v_1[120]), "=f"(v_1[121]), "=f"(v_1[122]), "=f"(v_1[123]), "=f"(v_1[124]), "=f"(v_1[125]), "=f"(v_1[126]), "=f"(v_1[127])
                                    : "r"(tile_addr_1 + 96));
#endif
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + ((unsigned int)wg + acc) * 8);
                                acc += 2;
                                if (acc >= 4) { acc -= 4; _phase_acc_full ^= 1; }
                                int row_stage_1 = ((4 + quarter) * 32 + lane_1) * 33 - windows[1];
                                int lim_1 = cpre_1;
                                int p_1 = kt_0;
                                int nvis_1 = ((u_1 - 256 * p_1 - 128 * rank) > (0) ? (u_1 - 256 * p_1 - 128 * rank) : (0));
                                float m_1 = -CUDART_INF_F;
                                int masked_1 = 1;
#if __CUDA_ARCH__ == 1030
                                m_1 = pm_1;
                                int _warp_vote_25 = __any_sync(0xFFFFFFFF, nvis_1 < 128);
                                masked_1 = _warp_vote_25;
#else
#endif
                                if (masked_1 != 0) {
                                    m_1 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_26 = __any_sync(0xFFFFFFFF, nvis_1 < 32);
                                    if (_warp_vote_26 != 0) {
                                        uint32_t _thresh_mask_27;
#else
                                    int _warp_vote_24 = __any_sync(0xFFFFFFFF, nvis_1 < 32);
                                    if (_warp_vote_24 != 0) {
                                        uint32_t _thresh_mask_25;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_28 = nvis_1;
                                            if (_lim_28 <= 0) { _thresh_mask_27 = 0u; }
                                            else if (_lim_28 >= 32) { _thresh_mask_27 = 0xFFFFFFFFu; }
#else
                                            int _lim_26 = nvis_1;
                                            if (_lim_26 <= 0) { _thresh_mask_25 = 0u; }
                                            else if (_lim_26 >= 32) { _thresh_mask_25 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_27) : "r"(_lim_28));
#else
                                                    "}" : "=r"(_thresh_mask_25) : "r"(_lim_26));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_27 & (1u << 0))) v_1[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 1))) v_1[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 2))) v_1[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 3))) v_1[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 4))) v_1[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 5))) v_1[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 6))) v_1[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 7))) v_1[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 8))) v_1[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 9))) v_1[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 10))) v_1[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 11))) v_1[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 12))) v_1[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 13))) v_1[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 14))) v_1[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 15))) v_1[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 16))) v_1[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 17))) v_1[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 18))) v_1[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 19))) v_1[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 20))) v_1[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 21))) v_1[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 22))) v_1[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 23))) v_1[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 24))) v_1[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 25))) v_1[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 26))) v_1[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 27))) v_1[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 28))) v_1[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 29))) v_1[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 30))) v_1[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_27 & (1u << 31))) v_1[31] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_25 & (1u << 0))) v_1[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 1))) v_1[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 2))) v_1[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 3))) v_1[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 4))) v_1[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 5))) v_1[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 6))) v_1[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 7))) v_1[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 8))) v_1[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 9))) v_1[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 10))) v_1[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 11))) v_1[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 12))) v_1[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 13))) v_1[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 14))) v_1[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 15))) v_1[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 16))) v_1[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 17))) v_1[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 18))) v_1[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 19))) v_1[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 20))) v_1[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 21))) v_1[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 22))) v_1[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 23))) v_1[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 24))) v_1[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 25))) v_1[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 26))) v_1[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 27))) v_1[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 28))) v_1[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 29))) v_1[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 30))) v_1[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_25 & (1u << 31))) v_1[31] = -CUDART_INF_F;
#endif
                                    }
                                    float part_4 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_30 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[0], v_1[1]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[2], v_1[3]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[4], v_1[5]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[6], v_1[7]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[8], v_1[9]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[10], v_1[11]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[12], v_1[13]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[14], v_1[15]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[16], v_1[17]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[18], v_1[19]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[20], v_1[21]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[22], v_1[23]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[24], v_1[25]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[26], v_1[27]));
                                    _reg_reduce_max2_30.x = max_noftz(_reg_reduce_max2_30.x, max_noftz(v_1[28], v_1[29]));
                                    _reg_reduce_max2_30.y = max_noftz(_reg_reduce_max2_30.y, max_noftz(v_1[30], v_1[31]));
                                    float _reg_array_reduce_29 = row_max_reduce(_reg_reduce_max2_30);
                                    part_4 = _reg_array_reduce_29;
#else
                                    float2 _reg_reduce_max2_28 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[0], v_1[1]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[2], v_1[3]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[4], v_1[5]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[6], v_1[7]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[8], v_1[9]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[10], v_1[11]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[12], v_1[13]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[14], v_1[15]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[16], v_1[17]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[18], v_1[19]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[20], v_1[21]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[22], v_1[23]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[24], v_1[25]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[26], v_1[27]));
                                    _reg_reduce_max2_28.x = max_noftz(_reg_reduce_max2_28.x, max_noftz(v_1[28], v_1[29]));
                                    _reg_reduce_max2_28.y = max_noftz(_reg_reduce_max2_28.y, max_noftz(v_1[30], v_1[31]));
                                    float _reg_array_reduce_27 = row_max_reduce(_reg_reduce_max2_28);
                                    part_4 = _reg_array_reduce_27;
#endif
                                    m_1 = fmaxf(m_1, part_4);
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_31 = __any_sync(0xFFFFFFFF, nvis_1 < 64);
                                    if (_warp_vote_31 != 0) {
                                        uint32_t _thresh_mask_32;
#else
                                    int _warp_vote_29 = __any_sync(0xFFFFFFFF, nvis_1 < 64);
                                    if (_warp_vote_29 != 0) {
                                        uint32_t _thresh_mask_30;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_33 = nvis_1 - 32;
                                            if (_lim_33 <= 0) { _thresh_mask_32 = 0u; }
                                            else if (_lim_33 >= 32) { _thresh_mask_32 = 0xFFFFFFFFu; }
#else
                                            int _lim_31 = nvis_1 - 32;
                                            if (_lim_31 <= 0) { _thresh_mask_30 = 0u; }
                                            else if (_lim_31 >= 32) { _thresh_mask_30 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_32) : "r"(_lim_33));
#else
                                                    "}" : "=r"(_thresh_mask_30) : "r"(_lim_31));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_32 & (1u << 0))) v_1[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 1))) v_1[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 2))) v_1[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 3))) v_1[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 4))) v_1[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 5))) v_1[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 6))) v_1[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 7))) v_1[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 8))) v_1[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 9))) v_1[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 10))) v_1[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 11))) v_1[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 12))) v_1[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 13))) v_1[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 14))) v_1[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 15))) v_1[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 16))) v_1[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 17))) v_1[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 18))) v_1[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 19))) v_1[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 20))) v_1[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 21))) v_1[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 22))) v_1[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 23))) v_1[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 24))) v_1[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 25))) v_1[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 26))) v_1[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 27))) v_1[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 28))) v_1[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 29))) v_1[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 30))) v_1[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_32 & (1u << 31))) v_1[63] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_30 & (1u << 0))) v_1[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 1))) v_1[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 2))) v_1[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 3))) v_1[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 4))) v_1[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 5))) v_1[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 6))) v_1[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 7))) v_1[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 8))) v_1[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 9))) v_1[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 10))) v_1[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 11))) v_1[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 12))) v_1[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 13))) v_1[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 14))) v_1[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 15))) v_1[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 16))) v_1[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 17))) v_1[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 18))) v_1[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 19))) v_1[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 20))) v_1[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 21))) v_1[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 22))) v_1[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 23))) v_1[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 24))) v_1[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 25))) v_1[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 26))) v_1[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 27))) v_1[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 28))) v_1[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 29))) v_1[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 30))) v_1[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_30 & (1u << 31))) v_1[63] = -CUDART_INF_F;
#endif
                                    }
                                    float part_5 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_35 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(32) + 0], _reg_reduce_max2_35);
                                    float _reg_array_reduce_34 = row_max_reduce(_reg_reduce_max2_35);
                                    part_5 = _reg_array_reduce_34;
#else
                                    float2 _reg_reduce_max2_33 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(32) + 0], _reg_reduce_max2_33);
                                    float _reg_array_reduce_32 = row_max_reduce(_reg_reduce_max2_33);
                                    part_5 = _reg_array_reduce_32;
#endif
                                    m_1 = fmaxf(m_1, part_5);
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_36 = __any_sync(0xFFFFFFFF, nvis_1 < 96);
                                    if (_warp_vote_36 != 0) {
                                        uint32_t _thresh_mask_37;
#else
                                    int _warp_vote_34 = __any_sync(0xFFFFFFFF, nvis_1 < 96);
                                    if (_warp_vote_34 != 0) {
                                        uint32_t _thresh_mask_35;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_38 = nvis_1 - 64;
                                            if (_lim_38 <= 0) { _thresh_mask_37 = 0u; }
                                            else if (_lim_38 >= 32) { _thresh_mask_37 = 0xFFFFFFFFu; }
#else
                                            int _lim_36 = nvis_1 - 64;
                                            if (_lim_36 <= 0) { _thresh_mask_35 = 0u; }
                                            else if (_lim_36 >= 32) { _thresh_mask_35 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_37) : "r"(_lim_38));
#else
                                                    "}" : "=r"(_thresh_mask_35) : "r"(_lim_36));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_37 & (1u << 0))) v_1[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 1))) v_1[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 2))) v_1[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 3))) v_1[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 4))) v_1[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 5))) v_1[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 6))) v_1[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 7))) v_1[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 8))) v_1[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 9))) v_1[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 10))) v_1[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 11))) v_1[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 12))) v_1[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 13))) v_1[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 14))) v_1[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 15))) v_1[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 16))) v_1[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 17))) v_1[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 18))) v_1[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 19))) v_1[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 20))) v_1[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 21))) v_1[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 22))) v_1[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 23))) v_1[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 24))) v_1[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 25))) v_1[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 26))) v_1[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 27))) v_1[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 28))) v_1[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 29))) v_1[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 30))) v_1[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_37 & (1u << 31))) v_1[95] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_35 & (1u << 0))) v_1[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 1))) v_1[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 2))) v_1[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 3))) v_1[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 4))) v_1[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 5))) v_1[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 6))) v_1[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 7))) v_1[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 8))) v_1[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 9))) v_1[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 10))) v_1[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 11))) v_1[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 12))) v_1[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 13))) v_1[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 14))) v_1[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 15))) v_1[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 16))) v_1[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 17))) v_1[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 18))) v_1[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 19))) v_1[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 20))) v_1[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 21))) v_1[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 22))) v_1[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 23))) v_1[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 24))) v_1[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 25))) v_1[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 26))) v_1[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 27))) v_1[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 28))) v_1[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 29))) v_1[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 30))) v_1[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_35 & (1u << 31))) v_1[95] = -CUDART_INF_F;
#endif
                                    }
                                    float part_6 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_40 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(64) + 0], _reg_reduce_max2_40);
                                    float _reg_array_reduce_39 = row_max_reduce(_reg_reduce_max2_40);
                                    part_6 = _reg_array_reduce_39;
#else
                                    float2 _reg_reduce_max2_38 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(64) + 0], _reg_reduce_max2_38);
                                    float _reg_array_reduce_37 = row_max_reduce(_reg_reduce_max2_38);
                                    part_6 = _reg_array_reduce_37;
#endif
                                    m_1 = fmaxf(m_1, part_6);
#if __CUDA_ARCH__ == 1030
                                    int _warp_vote_41 = __any_sync(0xFFFFFFFF, nvis_1 < 128);
                                    if (_warp_vote_41 != 0) {
                                        uint32_t _thresh_mask_42;
#else
                                    int _warp_vote_39 = __any_sync(0xFFFFFFFF, nvis_1 < 128);
                                    if (_warp_vote_39 != 0) {
                                        uint32_t _thresh_mask_40;
#endif
                                        {
#if __CUDA_ARCH__ == 1030
                                            int _lim_43 = nvis_1 - 96;
                                            if (_lim_43 <= 0) { _thresh_mask_42 = 0u; }
                                            else if (_lim_43 >= 32) { _thresh_mask_42 = 0xFFFFFFFFu; }
#else
                                            int _lim_41 = nvis_1 - 96;
                                            if (_lim_41 <= 0) { _thresh_mask_40 = 0u; }
                                            else if (_lim_41 >= 32) { _thresh_mask_40 = 0xFFFFFFFFu; }
#endif
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
#if __CUDA_ARCH__ == 1030
                                                    "}" : "=r"(_thresh_mask_42) : "r"(_lim_43));
#else
                                                    "}" : "=r"(_thresh_mask_40) : "r"(_lim_41));
#endif
                                            }
                                        }
#if __CUDA_ARCH__ == 1030
                                        if (!(_thresh_mask_42 & (1u << 0))) v_1[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 1))) v_1[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 2))) v_1[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 3))) v_1[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 4))) v_1[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 5))) v_1[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 6))) v_1[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 7))) v_1[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 8))) v_1[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 9))) v_1[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 10))) v_1[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 11))) v_1[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 12))) v_1[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 13))) v_1[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 14))) v_1[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 15))) v_1[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 16))) v_1[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 17))) v_1[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 18))) v_1[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 19))) v_1[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 20))) v_1[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 21))) v_1[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 22))) v_1[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 23))) v_1[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 24))) v_1[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 25))) v_1[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 26))) v_1[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 27))) v_1[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 28))) v_1[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 29))) v_1[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 30))) v_1[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_42 & (1u << 31))) v_1[127] = -CUDART_INF_F;
#else
                                        if (!(_thresh_mask_40 & (1u << 0))) v_1[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 1))) v_1[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 2))) v_1[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 3))) v_1[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 4))) v_1[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 5))) v_1[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 6))) v_1[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 7))) v_1[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 8))) v_1[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 9))) v_1[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 10))) v_1[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 11))) v_1[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 12))) v_1[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 13))) v_1[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 14))) v_1[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 15))) v_1[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 16))) v_1[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 17))) v_1[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 18))) v_1[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 19))) v_1[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 20))) v_1[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 21))) v_1[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 22))) v_1[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 23))) v_1[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 24))) v_1[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 25))) v_1[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 26))) v_1[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 27))) v_1[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 28))) v_1[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 29))) v_1[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 30))) v_1[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 31))) v_1[127] = -CUDART_INF_F;
#endif
                                    }
                                    float part_7 = -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                                    float2 _reg_reduce_max2_45 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(96) + 0], _reg_reduce_max2_45);
                                    float _reg_array_reduce_44 = row_max_reduce(_reg_reduce_max2_45);
                                    part_7 = _reg_array_reduce_44;
#else
                                    float2 _reg_reduce_max2_43 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(96) + 0], _reg_reduce_max2_43);
                                    float _reg_array_reduce_42 = row_max_reduce(_reg_reduce_max2_43);
                                    part_7 = _reg_array_reduce_42;
#endif
                                    m_1 = fmaxf(m_1, part_7);
                                }
                                m_1 = ((m_1 == 0.0f) ? 0.0f : m_1);
                                if (compact != 0) {
                                    if (p_1 < lim_1) {
                                        max_score[off_1 + (long long)p_1] = m_1;
                                    }
                                } else {
                                    if (2 * p_1 + own < lim_1) {
                                        max_score[off_1 + (long long)(2 * p_1 + own)] = m_1;
                                    }
                                    if (2 * p_1 + 1 - own < lim_1) {
                                        max_score[off_1 + (long long)(2 * p_1 + 1 - own)] = -CUDART_INF_F;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            long long wr_1 = work_range[blockIdx.x];
            int j0_1 = (int)(wr_1 & 65535);
            int j1_1 = (int)(wr_1 >> 16 & 65535);
            int c0_1 = (int)(wr_1 >> 32 & 65535);
            int c1_1 = (int)(wr_1 >> 48 & 65535);
            int j_end_1 = ((c1_1 > 0) ? j1_1 + 1 : j1_1);
            unsigned int q_slot = 0;
            unsigned int k_stage = 0;
            int lane_0 = lane;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (int j_1 = j0_1; j_1 < j_end_1; j_1++) {
                long long wi_1 = work_info[j_1];
                long long prev_1 = 0;
                if (j_1 > 0) {
                    prev_1 = work_info[j_1 - 1] >> 32;
                }
                int b_1 = (int)(wi_1 & 65535);
                int it_1 = (int)(wi_1 >> 16 & 65535);
                int ce_1 = (int)((wi_1 >> 32) - prev_1);
                if (j_1 == j1_1) {
                    ce_1 = c1_1;
                }
                int cb_1 = ((j_1 == j0_1) ? c0_1 : 0);
                int kt0_1 = cb_1 / 2;
                int kt1_1 = (ce_1 + 2 - 1) / 2;
                if (kt1_1 > kt0_1) {
                    int pages_end = (ce_1 + 1) / 2;
                    int q_row = (qo_indptr[b_1] + it_1 * 64) * 4;
                    mbarrier_wait(q_empty_addr + (q_slot) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_slot) * 8, 32768);
                        tma_2d_gmem2smem(smem_q_addr + q_slot * 2 * 16384, (&q), 0, q_row, q_full_addr + (q_slot) * 8);
                        tma_2d_gmem2smem(smem_q_addr + (q_slot * 2 + 1) * 16384, (&q), 0, q_row + 128, q_full_addr + (q_slot) * 8);
                    }
                    q_slot += 1;
                    if (q_slot == 2) { q_slot = 0; _phase_q_empty ^= 1; }
                    int bt_base = b_1 * bt_stride;
                    int p0 = kt0_1;
                    int ids = 0;
                    int ids_next = 0;
                    if (p0 + lane_0 < pages_end) {
                        ids = block_table[bt_base + p0 + lane_0];
                    }
                    if (p0 + 32 + lane_0 < pages_end) {
                        ids_next = block_table[bt_base + p0 + 32 + lane_0];
                    }
                    #pragma unroll 1
                    for (int kt = kt0_1; kt < kt1_1; kt++) {
                        int pl = kt - p0;
                        if (pl % 32 == 0 && pl > 0) {
                            ids = ids_next;
                            ids_next = 0;
                            if (p0 + pl + 32 + lane_0 < pages_end) {
                                ids_next = block_table[bt_base + p0 + pl + 32 + lane_0];
                            }
                        }
                        int npg = ((pages_end - kt) < (1) ? (pages_end - kt) : (1));
                        int pid_0 = __shfl_sync(0xFFFFFFFF, ids, pl % 32);
                        mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                        if (elect_sync()) {
                            if (npg > 0) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 16384);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384, (&k), 0, 0, pid_0, k_full_addr + (k_stage) * 8);
                            }
                            mbarrier_arrive(k_full_addr + (k_stage) * 8);
                        }
                        k_stage += 1;
                        if (k_stage == 8) { k_stage = 0; _phase_k_empty ^= 1; }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            long long wr_2 = work_range[blockIdx.x];
            int j0_2 = (int)(wr_2 & 65535);
            int j1_2 = (int)(wr_2 >> 16 & 65535);
            int c0_2 = (int)(wr_2 >> 32 & 65535);
            int c1_2 = (int)(wr_2 >> 48 & 65535);
            int j_end_2 = ((c1_2 > 0) ? j1_2 + 1 : j1_2);
            unsigned int q_slot_1 = 0;
            unsigned int k_stage_1 = 0;
            unsigned int acc_g_0 = 0;
            unsigned int acc_g_1 = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_acc_empty_1 = 1;
            #pragma unroll 1
            for (int j_2 = j0_2; j_2 < j_end_2; j_2++) {
                long long wi_2 = work_info[j_2];
                long long prev_2 = 0;
                if (j_2 > 0) {
                    prev_2 = work_info[j_2 - 1] >> 32;
                }
                int b_2 = (int)(wi_2 & 65535);
                int it_2 = (int)(wi_2 >> 16 & 65535);
                int ce_2 = (int)((wi_2 >> 32) - prev_2);
                if (j_2 == j1_2) {
                    ce_2 = c1_2;
                }
                int cb_2 = ((j_2 == j0_2) ? c0_2 : 0);
                int kt0_2 = cb_2 / 2;
                int kt1_2 = (ce_2 + 2 - 1) / 2;
                if (kt1_2 > kt0_2) {
                    mbarrier_wait(q_full_addr + (q_slot_1) * 8, _phase_q_full);
                    #pragma unroll 1
                    for (int kt_1 = kt0_2; kt_1 < kt1_2; kt_1++) {
                        mbarrier_wait(k_full_addr + (k_stage_1) * 8, _phase_k_full);
                        int b_lo = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                        mbarrier_wait(acc_empty_addr + (0 + acc_g_0) * 8, _phase_acc_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_slot_1 * 2) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(b_lo);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((0 + acc_g_0) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((0 + acc_g_0) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((0 + acc_g_0) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((0 + acc_g_0) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                            }
                        }
                        elect_commit(acc_full_addr + (0 + acc_g_0) * 8);
                        mbarrier_wait(acc_empty_addr + (1 + acc_g_1) * 8, _phase_acc_empty_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int a_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_slot_1 * 2 + 1) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(a_lo_1);
                            uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(b_lo);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((1 + acc_g_1) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((1 + acc_g_1) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((1 + acc_g_1) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f8f6f4((tmem_acc + ((1 + acc_g_1) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136314896, 1);
                            }
                        }
                        elect_commit(acc_full_addr + (1 + acc_g_1) * 8);
                        elect_commit(k_empty_addr + (k_stage_1) * 8);
                        k_stage_1 += 1;
                        if (k_stage_1 == 8) { k_stage_1 = 0; _phase_k_full ^= 1; }
                        acc_g_0 += 2;
                        if (acc_g_0 >= 4) { acc_g_0 -= 4; _phase_acc_empty ^= 1; }
                        acc_g_1 += 2;
                        if (acc_g_1 >= 4) { acc_g_1 -= 4; _phase_acc_empty_1 ^= 1; }
                    }
                    elect_commit(q_empty_addr + (q_slot_1) * 8);
                    q_slot_1 += 1;
                    if (q_slot_1 == 2) { q_slot_1 = 0; _phase_q_full ^= 1; }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
