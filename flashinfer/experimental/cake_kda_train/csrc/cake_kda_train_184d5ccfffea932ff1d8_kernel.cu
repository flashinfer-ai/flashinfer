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
#define TMEM_NCOLS 384
#define TMEM_ST_OFFSET 0
#define NUM_M_PIPE_STAGES 3
#define NUM_STEP_PIPE_STAGES 3
#define SMEM_M_K_OFF 1024
#define SMEM_M_K_STAGE_BYTES 65536
#define SMEM_M_K_STRIDE 65536
#define SMEM_M_MN_OFF 1024
#define SMEM_M_MN_STAGE_BYTES 65536
#define SMEM_M_MN_STRIDE 65536
#define SMEM_TOTAL 197632
#define THREADS 576
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






__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(576, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_184d5ccfffea932ff1d8(const __grid_constant__ CUtensorMap m_k_tma, const __grid_constant__ CUtensorMap m_mn_tma, float* __restrict__ b, float* __restrict__ h_out, int num_heads, int* __restrict__ chain_seg_start, int reverse)
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
    #define m_full_addr (mbar_base + 0)
    #define m_empty_addr (mbar_base + 24)
    #define s_ready_addr (mbar_base + 48)
    #define mma_done_addr (mbar_base + 72)
    #define h_read_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* m_k = reinterpret_cast<float*>(smem_raw + 1024);
    const int m_k_addr = smem + 1024;
    float* m_mn = reinterpret_cast<float*>(smem_raw + 1024);
    const int m_mn_addr = smem + 1024;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'm_pipe' ---
            // m_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // m_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'step_pipe' ---
            // s_ready: 3 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            mbarrier_init(smem + 56, 256);
            mbarrier_init(smem + 64, 256);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // h_read: 3 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 0) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_st = taddr;

    // ---- Role: readout ----
    if (warp <= 7) {
        { // readout_main
            int half = blockIdx.x % 2;
            int pair = blockIdx.x / 2;
            int head = pair % num_heads;
            int chain = pair / num_heads;
            int seg0 = chain_seg_start[chain];
            int nseg = chain_seg_start[chain + 1] - seg0;
            int steps = nseg - 1;
            int quad = warp % 4;
            int khalf = warp / 4 % 2;
            int lane_v = half * 64 + quad * 16 + (lane & 15);
            int row_base = quad * 32 << 16;
            int kcol = khalf * 64;
            int k0 = kcol + (lane >> 4) * 32;
            long long k_off = (long long)k0 * 128;
            int first = seg0;
            if (reverse == 1) {
                first = seg0 + nseg - 1;
            }
            long long base = ((long long)first * (long long)num_heads + (long long)head) * 16384 + (long long)lane_v;
            long long zero_base = base + k_off;
            float zeros[32];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            zeros[8] = 0.0f;
            zeros[9] = 0.0f;
            zeros[10] = 0.0f;
            zeros[11] = 0.0f;
            zeros[12] = 0.0f;
            zeros[13] = 0.0f;
            zeros[14] = 0.0f;
            zeros[15] = 0.0f;
            zeros[16] = 0.0f;
            zeros[17] = 0.0f;
            zeros[18] = 0.0f;
            zeros[19] = 0.0f;
            zeros[20] = 0.0f;
            zeros[21] = 0.0f;
            zeros[22] = 0.0f;
            zeros[23] = 0.0f;
            zeros[24] = 0.0f;
            zeros[25] = 0.0f;
            zeros[26] = 0.0f;
            zeros[27] = 0.0f;
            zeros[28] = 0.0f;
            zeros[29] = 0.0f;
            zeros[30] = 0.0f;
            zeros[31] = 0.0f;
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                *(reinterpret_cast<float*>(h_out) + (zero_base + (long long)(k * 128))) = zeros[0];
            }
            unsigned int rd = 0;
            unsigned int _phase_mma_done = 0;
            #pragma unroll 1
            for (int step = 0; step < steps; step++) {
                int nxt = (step + 1) % 3;
                int cur = step % 3;
                int s = step;
                if (reverse == 1) {
                    s = nseg - 1 - step;
                }
                int segment = seg0 + s;
                int sin = segment;
                int sout = sin + 1;
                if (reverse == 1) {
                    sout = sin - 1;
                }
                mbarrier_wait(mma_done_addr + (rd) * 8, _phase_mma_done);
                rd += 1;
                if (rd == 3) { rd = 0; _phase_mma_done ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long base_0 = ((long long)sout * (long long)num_heads + (long long)head) * 16384 + (long long)lane_v;
                long long o_base = base_0 + k_off;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + (unsigned int)row_base + (unsigned int)(nxt * 128 + kcol)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int k_1 = 0; k_1 < 32; k_1++) {
                    *(reinterpret_cast<float*>(h_out) + (o_base + (long long)(k_1 * 128))) = _tmem_load_0[k_1];
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(h_read_addr + (cur) * 8);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 512;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: offset ----
    if (warp >= 8 && warp <= 15) {
        { // offset_main
            int half_1 = blockIdx.x % 2;
            int pair_1 = blockIdx.x / 2;
            int head_1 = pair_1 % num_heads;
            int chain_1 = pair_1 / num_heads;
            int seg0_1 = chain_seg_start[chain_1];
            int nseg_1 = chain_seg_start[chain_1 + 1] - seg0_1;
            int steps_1 = nseg_1 - 1;
            int quad_1 = warp % 4;
            int khalf_1 = warp / 4 % 2;
            int lane_v_1 = half_1 * 64 + quad_1 * 16 + (lane & 15);
            int row_base_1 = quad_1 * 32 << 16;
            int kcol_1 = khalf_1 * 64;
            int k0_1 = kcol_1 + (lane >> 4) * 32;
            long long k_off_1 = (long long)k0_1 * 128;
            float zeros_1[32];
            zeros_1[0] = 0.0f;
            zeros_1[1] = 0.0f;
            zeros_1[2] = 0.0f;
            zeros_1[3] = 0.0f;
            zeros_1[4] = 0.0f;
            zeros_1[5] = 0.0f;
            zeros_1[6] = 0.0f;
            zeros_1[7] = 0.0f;
            zeros_1[8] = 0.0f;
            zeros_1[9] = 0.0f;
            zeros_1[10] = 0.0f;
            zeros_1[11] = 0.0f;
            zeros_1[12] = 0.0f;
            zeros_1[13] = 0.0f;
            zeros_1[14] = 0.0f;
            zeros_1[15] = 0.0f;
            zeros_1[16] = 0.0f;
            zeros_1[17] = 0.0f;
            zeros_1[18] = 0.0f;
            zeros_1[19] = 0.0f;
            zeros_1[20] = 0.0f;
            zeros_1[21] = 0.0f;
            zeros_1[22] = 0.0f;
            zeros_1[23] = 0.0f;
            zeros_1[24] = 0.0f;
            zeros_1[25] = 0.0f;
            zeros_1[26] = 0.0f;
            zeros_1[27] = 0.0f;
            zeros_1[28] = 0.0f;
            zeros_1[29] = 0.0f;
            zeros_1[30] = 0.0f;
            zeros_1[31] = 0.0f;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x32.b32"
                " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                :: "r"(taddr + (unsigned int)row_base_1 + (unsigned int)kcol_1), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&zeros_1[31])));
            float bcol[32];
            if (steps_1 > 0) {
                int s_1 = 0;
                if (reverse == 1) {
                    s_1 = nseg_1 - 1;
                }
                int segment_1 = seg0_1 + s_1;
                long long base_1 = ((long long)segment_1 * (long long)num_heads + (long long)head_1) * 16384 + (long long)lane_v_1;
                #pragma unroll
                for (int k_2 = 0; k_2 < 32; k_2++) {
                    {
                        bcol[k_2] = *reinterpret_cast<const float*>(b + (base_1 + k_off_1 + (long long)(k_2 * 128)));
                    }
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x32.b32"
                    " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + (unsigned int)row_base_1 + (unsigned int)(128 + kcol_1)), "r"(*reinterpret_cast<const uint32_t*>(&bcol[0])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[1])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[2])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[3])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[4])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[5])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[6])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[7])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[8])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[9])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[10])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[11])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[12])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[13])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[14])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[15])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[16])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[17])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[18])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[19])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[20])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[21])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[22])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[23])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[24])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[25])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[26])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[27])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[28])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[29])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[30])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[31])));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(s_ready_addr);
            if (steps_1 > 1) {
                int s_2 = 1;
                if (reverse == 1) {
                    s_2 = nseg_1 - 1 - 1;
                }
                int segment_2 = seg0_1 + s_2;
                long long base_2 = ((long long)segment_2 * (long long)num_heads + (long long)head_1) * 16384 + (long long)lane_v_1;
                #pragma unroll
                for (int k_3 = 0; k_3 < 32; k_3++) {
                    {
                        bcol[k_3] = *reinterpret_cast<const float*>(b + (base_2 + k_off_1 + (long long)(k_3 * 128)));
                    }
                }
            }
            unsigned int od = 0;
            unsigned int oh = 0;
            unsigned int _phase_mma_done_1 = 0;
            unsigned int _phase_h_read = 0;
            #pragma unroll 1
            for (int step_1 = 0; step_1 < steps_1 - 1; step_1++) {
                int tgt = (step_1 + 2) % 3;
                int nxt_stage = (step_1 + 1) % 3;
                if (step_1 >= 1) {
                    mbarrier_wait(mma_done_addr + (od) * 8, _phase_mma_done_1);
                    od += 1;
                    if (od == 3) { od = 0; _phase_mma_done_1 ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                if (step_1 >= 2) {
                    mbarrier_wait(h_read_addr + (oh) * 8, _phase_h_read);
                    oh += 1;
                    if (oh == 3) { oh = 0; _phase_h_read ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x32.b32"
                    " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + (unsigned int)row_base_1 + (unsigned int)(tgt * 128 + kcol_1)), "r"(*reinterpret_cast<const uint32_t*>(&bcol[0])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[1])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[2])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[3])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[4])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[5])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[6])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[7])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[8])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[9])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[10])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[11])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[12])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[13])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[14])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[15])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[16])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[17])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[18])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[19])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[20])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[21])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[22])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[23])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[24])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[25])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[26])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[27])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[28])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[29])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[30])), "r"(*reinterpret_cast<const uint32_t*>(&bcol[31])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_ready_addr + (nxt_stage) * 8);
                if (steps_1 > step_1 + 2) {
                    int s_3 = step_1 + 2;
                    if (reverse == 1) {
                        s_3 = nseg_1 - 1 - (step_1 + 2);
                    }
                    int segment_3 = seg0_1 + s_3;
                    long long base_3 = ((long long)segment_3 * (long long)num_heads + (long long)head_1) * 16384 + (long long)lane_v_1;
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 32; k_4++) {
                        {
                            bcol[k_4] = *reinterpret_cast<const float*>(b + (base_3 + k_off_1 + (long long)(k_4 * 128)));
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 512;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 16) {
        { // load_main
            int half_2 = blockIdx.x % 2;
            int pair_2 = blockIdx.x / 2;
            int head_2 = pair_2 % num_heads;
            int chain_2 = pair_2 / num_heads;
            int seg0_2 = chain_seg_start[chain_2];
            int nseg_2 = chain_seg_start[chain_2 + 1] - seg0_2;
            int steps_2 = nseg_2 - 1;
            unsigned int ms = 0;
            unsigned int _phase_m_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int step_2 = 0; step_2 < steps_2; step_2++) {
                    int s_4 = step_2;
                    if (reverse == 1) {
                        s_4 = nseg_2 - 1 - step_2;
                    }
                    int segment_4 = seg0_2 + s_4;
                    int tile = segment_4 * num_heads + head_2;
                    mbarrier_wait(m_empty_addr + (ms) * 8, _phase_m_empty);
                    mbarrier_arrive_expect_tx(m_full_addr + (ms) * 8, 65536);
                    if (reverse == 1) {
                        tma_4d_gmem2smem(m_mn_addr + ms * 65536, (&m_mn_tma), 0, 0, 0, tile, m_full_addr + (ms) * 8);
                    } else {
                        tma_4d_gmem2smem(m_k_addr + ms * 65536, (&m_k_tma), 0, 0, 0, tile, m_full_addr + (ms) * 8);
                    }
                    ms += 1;
                    if (ms == 3) { ms = 0; _phase_m_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 17) {
        { // mma_main
            int half_3 = blockIdx.x % 2;
            int pair_3 = blockIdx.x / 2;
            int head_3 = pair_3 % num_heads;
            int chain_3 = pair_3 / num_heads;
            int seg0_3 = chain_seg_start[chain_3];
            int nseg_3 = chain_seg_start[chain_3 + 1] - seg0_3;
            int steps_3 = nseg_3 - 1;
            unsigned int ms_1 = 0;
            unsigned int rs = 0;
            unsigned int ds = 0;
            unsigned int _phase_m_full = 0;
            unsigned int _phase_s_ready = 0;
            unsigned int _phase_mma_done_2 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int step_3 = 0; step_3 < steps_3; step_3++) {
                    int cur_1 = step_3 % 3;
                    int nxt_1 = (step_3 + 1) % 3;
                    mbarrier_wait(m_full_addr + (ms_1) * 8, _phase_m_full);
                    mbarrier_wait(s_ready_addr + (rs) * 8, _phase_s_ready);
                    rs += 1;
                    if (rs == 3) { rs = 0; _phase_s_ready ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (reverse == 1) {
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = ((((m_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (ms_1) * 4096;
#else
                        int _smem_desc_lo_0 = ((((m_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (ms_1) * 4096;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x20004020;\n\t"
                    "mov.b32 id, 69273872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_st + (nxt_1 * 128))), "r"(_smem_desc_lo_0), "r"(tmem_st + cur_1 * 128), "r"(1) : "memory");
                    } else {
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_1 = (((m_k_addr) >> 4) & 0x7FFF) + (ms_1) * 4096;
#else
                        int _smem_desc_lo_1 = (((m_k_addr) >> 4) & 0x3FFF) + (ms_1) * 4096;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69208336;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_st + (nxt_1 * 128))), "r"(_smem_desc_lo_1), "r"(tmem_st + cur_1 * 128), "r"(1) : "memory");
                    }
                    tcgen05_commit(mma_done_addr + (cur_1) * 8);
                    tcgen05_commit(m_empty_addr + (ms_1) * 8);
                    ms_1 += 1;
                    if (ms_1 == 3) { ms_1 = 0; _phase_m_full ^= 1; }
                    mbarrier_wait(mma_done_addr + (ds) * 8, _phase_mma_done_2);
                    ds += 1;
                    if (ds == 3) { ds = 0; _phase_mma_done_2 ^= 1; }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
