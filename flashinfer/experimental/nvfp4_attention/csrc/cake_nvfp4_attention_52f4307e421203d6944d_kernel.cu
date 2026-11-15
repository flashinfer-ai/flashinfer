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
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_A_SMEM_OFF 1024
#define SMEM_A_SMEM_STAGE_BYTES 16384
#define SMEM_A_SMEM_STRIDE 16384
#define SMEM_B_SMEM_OFF 66560
#define SMEM_B_SMEM_STAGE_BYTES 16384
#define SMEM_B_SMEM_STRIDE 16384
#define SMEM_STAGE_SMEM_OFF 132096
#define SMEM_STAGE_SMEM_STAGE_BYTES 65536
#define SMEM_STAGE_SMEM_STRIDE 65536
#define SMEM_TOTAL 197632
#define THREADS 128
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


__device__ __forceinline__ void tcgen05_mma_tf32(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n\t"
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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_attention_52f4307e421203d6944d(const __grid_constant__ CUtensorMap QM, const __grid_constant__ CUtensorMap KC, __nv_bfloat16* __restrict__ tab, int kv_stride, int num_q_blocks, int total_bh)
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
    #define full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 8)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* a_smem = reinterpret_cast<float*>(smem_raw + 1024);
    const int a_smem_addr = smem + 1024;
    float* b_smem = reinterpret_cast<float*>(smem_raw + 66560);
    const int b_smem_addr = smem + 66560;
    float* stage_smem = reinterpret_cast<float*>(smem_raw + 132096);
    const int stage_smem_addr = smem + 132096;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // mma_done: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 16);
    if (warp == 0) {
        int _tmem_hold = smem + 16;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int m_tiles = (num_q_blocks + 128 - 1) / 128;
    int n_tiles = kv_stride / 128;
    int items_per_bh = m_tiles * n_tiles;
    int total_items = total_bh * items_per_bh;
    int num_groups = (num_q_blocks + 8 - 1) / 8;
    int num_key_tiles = kv_stride / 64;
    int qb_local = warp * 32 + lane;
    float zero4[4];
    zero4[0] = 0.0f;
    zero4[1] = 0.0f;
    zero4[2] = 0.0f;
    zero4[3] = 0.0f;
    float xs[8];
    float lo[8];
    unsigned int words[8];
    int tmem_row_base = make_warp_uniform(warp * 32 << 16);
    float vals[128];
    int first = bid;
    if (first < total_items) {
        int bh = first / items_per_bh;
        int rem = first - bh * items_per_bh;
        int m = rem / n_tiles;
        int n = rem - m * n_tiles;
        int qm_row = bh * num_q_blocks + m * 128;
        int kc_row = bh * kv_stride + n * 128;
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive_expect_tx(full_addr, 131072);
                tma_2d_gmem2smem(a_smem_addr, (&QM), 0, qm_row, full_addr);
                tma_2d_gmem2smem(b_smem_addr, (&KC), 0, kc_row, full_addr);
                tma_2d_gmem2smem(a_smem_addr + 16384, (&QM), 32, qm_row, full_addr);
                tma_2d_gmem2smem(b_smem_addr + 16384, (&KC), 32, kc_row, full_addr);
                tma_2d_gmem2smem(a_smem_addr + 32768, (&QM), 64, qm_row, full_addr);
                tma_2d_gmem2smem(b_smem_addr + 32768, (&KC), 64, kc_row, full_addr);
                tma_2d_gmem2smem(a_smem_addr + 49152, (&QM), 96, qm_row, full_addr);
                tma_2d_gmem2smem(b_smem_addr + 49152, (&KC), 96, kc_row, full_addr);
            }
        }
    }
    unsigned int _phase_full_0 = 0;
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int item = bid; item < total_items; item += num_bids) {
        int bh_1 = item / items_per_bh;
        int rem_1 = item - bh_1 * items_per_bh;
        int m_1 = rem_1 / n_tiles;
        int n_1 = rem_1 - m_1 * n_tiles;
        if (warp == 0) {
            mbarrier_wait(full_addr, _phase_full_0);
            _phase_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_0 = make_warp_uniform((((a_smem_addr) >> 4) & 0x3FFF) + (0) * 1024);
            int _mma_b_lo_0 = make_warp_uniform((((b_smem_addr) >> 4) & 0x3FFF) + (0) * 1024);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136317200, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136317200, 1);
                }
            }
            int _mma_a_lo_1 = make_warp_uniform((((a_smem_addr) >> 4) & 0x3FFF) + (1) * 1024);
            int _mma_b_lo_1 = make_warp_uniform((((b_smem_addr) >> 4) & 0x3FFF) + (1) * 1024);
            {
                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136317200, 1);
                }
            }
            int _mma_a_lo_2 = make_warp_uniform((((a_smem_addr) >> 4) & 0x3FFF) + (2) * 1024);
            int _mma_b_lo_2 = make_warp_uniform((((b_smem_addr) >> 4) & 0x3FFF) + (2) * 1024);
            {
                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136317200, 1);
                }
            }
            int _mma_a_lo_3 = make_warp_uniform((((a_smem_addr) >> 4) & 0x3FFF) + (3) * 1024);
            int _mma_b_lo_3 = make_warp_uniform((((b_smem_addr) >> 4) & 0x3FFF) + (3) * 1024);
            {
                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136317200, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_tf32(tmem_acc, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 136317200, 1);
                }
            }
            elect_commit(mma_done_addr);
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        int nxt = item + num_bids;
        if (nxt < total_items) {
            int bh_0 = nxt / items_per_bh;
            int rem_1_1 = nxt - bh_0 * items_per_bh;
            int m_2 = rem_1_1 / n_tiles;
            int n_3 = rem_1_1 - m_2 * n_tiles;
            int qm_row_1 = bh_0 * num_q_blocks + m_2 * 128;
            int kc_row_1 = bh_0 * kv_stride + n_3 * 128;
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive_expect_tx(full_addr, 131072);
                    tma_2d_gmem2smem(a_smem_addr, (&QM), 0, qm_row_1, full_addr);
                    tma_2d_gmem2smem(b_smem_addr, (&KC), 0, kc_row_1, full_addr);
                    tma_2d_gmem2smem(a_smem_addr + 16384, (&QM), 32, qm_row_1, full_addr);
                    tma_2d_gmem2smem(b_smem_addr + 16384, (&KC), 32, kc_row_1, full_addr);
                    tma_2d_gmem2smem(a_smem_addr + 32768, (&QM), 64, qm_row_1, full_addr);
                    tma_2d_gmem2smem(b_smem_addr + 32768, (&KC), 64, kc_row_1, full_addr);
                    tma_2d_gmem2smem(a_smem_addr + 49152, (&QM), 96, qm_row_1, full_addr);
                    tma_2d_gmem2smem(b_smem_addr + 49152, (&KC), 96, kc_row_1, full_addr);
                }
            }
        }
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(vals[0]), "=f"(vals[1]), "=f"(vals[2]), "=f"(vals[3]), "=f"(vals[4]), "=f"(vals[5]), "=f"(vals[6]), "=f"(vals[7]), "=f"(vals[8]), "=f"(vals[9]), "=f"(vals[10]), "=f"(vals[11]), "=f"(vals[12]), "=f"(vals[13]), "=f"(vals[14]), "=f"(vals[15]), "=f"(vals[16]), "=f"(vals[17]), "=f"(vals[18]), "=f"(vals[19]), "=f"(vals[20]), "=f"(vals[21]), "=f"(vals[22]), "=f"(vals[23]), "=f"(vals[24]), "=f"(vals[25]), "=f"(vals[26]), "=f"(vals[27]), "=f"(vals[28]), "=f"(vals[29]), "=f"(vals[30]), "=f"(vals[31])
            : "r"(taddr + (unsigned int)0 + (unsigned int)tmem_row_base));
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(vals[32]), "=f"(vals[33]), "=f"(vals[34]), "=f"(vals[35]), "=f"(vals[36]), "=f"(vals[37]), "=f"(vals[38]), "=f"(vals[39]), "=f"(vals[40]), "=f"(vals[41]), "=f"(vals[42]), "=f"(vals[43]), "=f"(vals[44]), "=f"(vals[45]), "=f"(vals[46]), "=f"(vals[47]), "=f"(vals[48]), "=f"(vals[49]), "=f"(vals[50]), "=f"(vals[51]), "=f"(vals[52]), "=f"(vals[53]), "=f"(vals[54]), "=f"(vals[55]), "=f"(vals[56]), "=f"(vals[57]), "=f"(vals[58]), "=f"(vals[59]), "=f"(vals[60]), "=f"(vals[61]), "=f"(vals[62]), "=f"(vals[63])
            : "r"(taddr + (unsigned int)0 + (unsigned int)tmem_row_base + 32));
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(vals[64]), "=f"(vals[65]), "=f"(vals[66]), "=f"(vals[67]), "=f"(vals[68]), "=f"(vals[69]), "=f"(vals[70]), "=f"(vals[71]), "=f"(vals[72]), "=f"(vals[73]), "=f"(vals[74]), "=f"(vals[75]), "=f"(vals[76]), "=f"(vals[77]), "=f"(vals[78]), "=f"(vals[79]), "=f"(vals[80]), "=f"(vals[81]), "=f"(vals[82]), "=f"(vals[83]), "=f"(vals[84]), "=f"(vals[85]), "=f"(vals[86]), "=f"(vals[87]), "=f"(vals[88]), "=f"(vals[89]), "=f"(vals[90]), "=f"(vals[91]), "=f"(vals[92]), "=f"(vals[93]), "=f"(vals[94]), "=f"(vals[95])
            : "r"(taddr + (unsigned int)0 + (unsigned int)tmem_row_base + 64));
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(vals[96]), "=f"(vals[97]), "=f"(vals[98]), "=f"(vals[99]), "=f"(vals[100]), "=f"(vals[101]), "=f"(vals[102]), "=f"(vals[103]), "=f"(vals[104]), "=f"(vals[105]), "=f"(vals[106]), "=f"(vals[107]), "=f"(vals[108]), "=f"(vals[109]), "=f"(vals[110]), "=f"(vals[111]), "=f"(vals[112]), "=f"(vals[113]), "=f"(vals[114]), "=f"(vals[115]), "=f"(vals[116]), "=f"(vals[117]), "=f"(vals[118]), "=f"(vals[119]), "=f"(vals[120]), "=f"(vals[121]), "=f"(vals[122]), "=f"(vals[123]), "=f"(vals[124]), "=f"(vals[125]), "=f"(vals[126]), "=f"(vals[127])
            : "r"(taddr + (unsigned int)0 + (unsigned int)tmem_row_base + 96));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int qb = m_1 * 128 + qb_local;
        unsigned int stage_col = stage_smem_addr + (unsigned int)(qb_local * 4);
        if (qb < num_q_blocks) {
            {
                uint32_t _addr_4 = static_cast<uint32_t>(stage_col);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(vals[0]) : "memory");
            }
            {
                uint32_t _addr_5 = static_cast<uint32_t>(stage_col + 512);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(vals[1]) : "memory");
            }
            {
                uint32_t _addr_6 = static_cast<uint32_t>(stage_col + 1024);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(vals[2]) : "memory");
            }
            {
                uint32_t _addr_7 = static_cast<uint32_t>(stage_col + 1536);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(vals[3]) : "memory");
            }
            {
                uint32_t _addr_8 = static_cast<uint32_t>(stage_col + 2048);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(vals[4]) : "memory");
            }
            {
                uint32_t _addr_9 = static_cast<uint32_t>(stage_col + 2560);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(vals[5]) : "memory");
            }
            {
                uint32_t _addr_10 = static_cast<uint32_t>(stage_col + 3072);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(vals[6]) : "memory");
            }
            {
                uint32_t _addr_11 = static_cast<uint32_t>(stage_col + 3584);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(vals[7]) : "memory");
            }
            {
                uint32_t _addr_12 = static_cast<uint32_t>(stage_col + 4096);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(vals[8]) : "memory");
            }
            {
                uint32_t _addr_13 = static_cast<uint32_t>(stage_col + 4608);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(vals[9]) : "memory");
            }
            {
                uint32_t _addr_14 = static_cast<uint32_t>(stage_col + 5120);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(vals[10]) : "memory");
            }
            {
                uint32_t _addr_15 = static_cast<uint32_t>(stage_col + 5632);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(vals[11]) : "memory");
            }
            {
                uint32_t _addr_16 = static_cast<uint32_t>(stage_col + 6144);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(vals[12]) : "memory");
            }
            {
                uint32_t _addr_17 = static_cast<uint32_t>(stage_col + 6656);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(vals[13]) : "memory");
            }
            {
                uint32_t _addr_18 = static_cast<uint32_t>(stage_col + 7168);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(vals[14]) : "memory");
            }
            {
                uint32_t _addr_19 = static_cast<uint32_t>(stage_col + 7680);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(vals[15]) : "memory");
            }
            {
                uint32_t _addr_20 = static_cast<uint32_t>(stage_col + 8192);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(vals[16]) : "memory");
            }
            {
                uint32_t _addr_21 = static_cast<uint32_t>(stage_col + 8704);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(vals[17]) : "memory");
            }
            {
                uint32_t _addr_22 = static_cast<uint32_t>(stage_col + 9216);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(vals[18]) : "memory");
            }
            {
                uint32_t _addr_23 = static_cast<uint32_t>(stage_col + 9728);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(vals[19]) : "memory");
            }
            {
                uint32_t _addr_24 = static_cast<uint32_t>(stage_col + 10240);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(vals[20]) : "memory");
            }
            {
                uint32_t _addr_25 = static_cast<uint32_t>(stage_col + 10752);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(vals[21]) : "memory");
            }
            {
                uint32_t _addr_26 = static_cast<uint32_t>(stage_col + 11264);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(vals[22]) : "memory");
            }
            {
                uint32_t _addr_27 = static_cast<uint32_t>(stage_col + 11776);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(vals[23]) : "memory");
            }
            {
                uint32_t _addr_28 = static_cast<uint32_t>(stage_col + 12288);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(vals[24]) : "memory");
            }
            {
                uint32_t _addr_29 = static_cast<uint32_t>(stage_col + 12800);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(vals[25]) : "memory");
            }
            {
                uint32_t _addr_30 = static_cast<uint32_t>(stage_col + 13312);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(vals[26]) : "memory");
            }
            {
                uint32_t _addr_31 = static_cast<uint32_t>(stage_col + 13824);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(vals[27]) : "memory");
            }
            {
                uint32_t _addr_32 = static_cast<uint32_t>(stage_col + 14336);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(vals[28]) : "memory");
            }
            {
                uint32_t _addr_33 = static_cast<uint32_t>(stage_col + 14848);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(vals[29]) : "memory");
            }
            {
                uint32_t _addr_34 = static_cast<uint32_t>(stage_col + 15360);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(vals[30]) : "memory");
            }
            {
                uint32_t _addr_35 = static_cast<uint32_t>(stage_col + 15872);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(vals[31]) : "memory");
            }
            {
                uint32_t _addr_36 = static_cast<uint32_t>(stage_col + 16384);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(vals[32]) : "memory");
            }
            {
                uint32_t _addr_37 = static_cast<uint32_t>(stage_col + 16896);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(vals[33]) : "memory");
            }
            {
                uint32_t _addr_38 = static_cast<uint32_t>(stage_col + 17408);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(vals[34]) : "memory");
            }
            {
                uint32_t _addr_39 = static_cast<uint32_t>(stage_col + 17920);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(vals[35]) : "memory");
            }
            {
                uint32_t _addr_40 = static_cast<uint32_t>(stage_col + 18432);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(vals[36]) : "memory");
            }
            {
                uint32_t _addr_41 = static_cast<uint32_t>(stage_col + 18944);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(vals[37]) : "memory");
            }
            {
                uint32_t _addr_42 = static_cast<uint32_t>(stage_col + 19456);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(vals[38]) : "memory");
            }
            {
                uint32_t _addr_43 = static_cast<uint32_t>(stage_col + 19968);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(vals[39]) : "memory");
            }
            {
                uint32_t _addr_44 = static_cast<uint32_t>(stage_col + 20480);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(vals[40]) : "memory");
            }
            {
                uint32_t _addr_45 = static_cast<uint32_t>(stage_col + 20992);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(vals[41]) : "memory");
            }
            {
                uint32_t _addr_46 = static_cast<uint32_t>(stage_col + 21504);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(vals[42]) : "memory");
            }
            {
                uint32_t _addr_47 = static_cast<uint32_t>(stage_col + 22016);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(vals[43]) : "memory");
            }
            {
                uint32_t _addr_48 = static_cast<uint32_t>(stage_col + 22528);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(vals[44]) : "memory");
            }
            {
                uint32_t _addr_49 = static_cast<uint32_t>(stage_col + 23040);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(vals[45]) : "memory");
            }
            {
                uint32_t _addr_50 = static_cast<uint32_t>(stage_col + 23552);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(vals[46]) : "memory");
            }
            {
                uint32_t _addr_51 = static_cast<uint32_t>(stage_col + 24064);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(vals[47]) : "memory");
            }
            {
                uint32_t _addr_52 = static_cast<uint32_t>(stage_col + 24576);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(vals[48]) : "memory");
            }
            {
                uint32_t _addr_53 = static_cast<uint32_t>(stage_col + 25088);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(vals[49]) : "memory");
            }
            {
                uint32_t _addr_54 = static_cast<uint32_t>(stage_col + 25600);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(vals[50]) : "memory");
            }
            {
                uint32_t _addr_55 = static_cast<uint32_t>(stage_col + 26112);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(vals[51]) : "memory");
            }
            {
                uint32_t _addr_56 = static_cast<uint32_t>(stage_col + 26624);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(vals[52]) : "memory");
            }
            {
                uint32_t _addr_57 = static_cast<uint32_t>(stage_col + 27136);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(vals[53]) : "memory");
            }
            {
                uint32_t _addr_58 = static_cast<uint32_t>(stage_col + 27648);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(vals[54]) : "memory");
            }
            {
                uint32_t _addr_59 = static_cast<uint32_t>(stage_col + 28160);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(vals[55]) : "memory");
            }
            {
                uint32_t _addr_60 = static_cast<uint32_t>(stage_col + 28672);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(vals[56]) : "memory");
            }
            {
                uint32_t _addr_61 = static_cast<uint32_t>(stage_col + 29184);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(vals[57]) : "memory");
            }
            {
                uint32_t _addr_62 = static_cast<uint32_t>(stage_col + 29696);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(vals[58]) : "memory");
            }
            {
                uint32_t _addr_63 = static_cast<uint32_t>(stage_col + 30208);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(vals[59]) : "memory");
            }
            {
                uint32_t _addr_64 = static_cast<uint32_t>(stage_col + 30720);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(vals[60]) : "memory");
            }
            {
                uint32_t _addr_65 = static_cast<uint32_t>(stage_col + 31232);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(vals[61]) : "memory");
            }
            {
                uint32_t _addr_66 = static_cast<uint32_t>(stage_col + 31744);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(vals[62]) : "memory");
            }
            {
                uint32_t _addr_67 = static_cast<uint32_t>(stage_col + 32256);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(vals[63]) : "memory");
            }
            {
                uint32_t _addr_68 = static_cast<uint32_t>(stage_col + 32768);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(vals[64]) : "memory");
            }
            {
                uint32_t _addr_69 = static_cast<uint32_t>(stage_col + 33280);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(vals[65]) : "memory");
            }
            {
                uint32_t _addr_70 = static_cast<uint32_t>(stage_col + 33792);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(vals[66]) : "memory");
            }
            {
                uint32_t _addr_71 = static_cast<uint32_t>(stage_col + 34304);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_71), "f"(vals[67]) : "memory");
            }
            {
                uint32_t _addr_72 = static_cast<uint32_t>(stage_col + 34816);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(vals[68]) : "memory");
            }
            {
                uint32_t _addr_73 = static_cast<uint32_t>(stage_col + 35328);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(vals[69]) : "memory");
            }
            {
                uint32_t _addr_74 = static_cast<uint32_t>(stage_col + 35840);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(vals[70]) : "memory");
            }
            {
                uint32_t _addr_75 = static_cast<uint32_t>(stage_col + 36352);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(vals[71]) : "memory");
            }
            {
                uint32_t _addr_76 = static_cast<uint32_t>(stage_col + 36864);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(vals[72]) : "memory");
            }
            {
                uint32_t _addr_77 = static_cast<uint32_t>(stage_col + 37376);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(vals[73]) : "memory");
            }
            {
                uint32_t _addr_78 = static_cast<uint32_t>(stage_col + 37888);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(vals[74]) : "memory");
            }
            {
                uint32_t _addr_79 = static_cast<uint32_t>(stage_col + 38400);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(vals[75]) : "memory");
            }
            {
                uint32_t _addr_80 = static_cast<uint32_t>(stage_col + 38912);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(vals[76]) : "memory");
            }
            {
                uint32_t _addr_81 = static_cast<uint32_t>(stage_col + 39424);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(vals[77]) : "memory");
            }
            {
                uint32_t _addr_82 = static_cast<uint32_t>(stage_col + 39936);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(vals[78]) : "memory");
            }
            {
                uint32_t _addr_83 = static_cast<uint32_t>(stage_col + 40448);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(vals[79]) : "memory");
            }
            {
                uint32_t _addr_84 = static_cast<uint32_t>(stage_col + 40960);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(vals[80]) : "memory");
            }
            {
                uint32_t _addr_85 = static_cast<uint32_t>(stage_col + 41472);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(vals[81]) : "memory");
            }
            {
                uint32_t _addr_86 = static_cast<uint32_t>(stage_col + 41984);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_86), "f"(vals[82]) : "memory");
            }
            {
                uint32_t _addr_87 = static_cast<uint32_t>(stage_col + 42496);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(vals[83]) : "memory");
            }
            {
                uint32_t _addr_88 = static_cast<uint32_t>(stage_col + 43008);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(vals[84]) : "memory");
            }
            {
                uint32_t _addr_89 = static_cast<uint32_t>(stage_col + 43520);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(vals[85]) : "memory");
            }
            {
                uint32_t _addr_90 = static_cast<uint32_t>(stage_col + 44032);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(vals[86]) : "memory");
            }
            {
                uint32_t _addr_91 = static_cast<uint32_t>(stage_col + 44544);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(vals[87]) : "memory");
            }
            {
                uint32_t _addr_92 = static_cast<uint32_t>(stage_col + 45056);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(vals[88]) : "memory");
            }
            {
                uint32_t _addr_93 = static_cast<uint32_t>(stage_col + 45568);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(vals[89]) : "memory");
            }
            {
                uint32_t _addr_94 = static_cast<uint32_t>(stage_col + 46080);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(vals[90]) : "memory");
            }
            {
                uint32_t _addr_95 = static_cast<uint32_t>(stage_col + 46592);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(vals[91]) : "memory");
            }
            {
                uint32_t _addr_96 = static_cast<uint32_t>(stage_col + 47104);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(vals[92]) : "memory");
            }
            {
                uint32_t _addr_97 = static_cast<uint32_t>(stage_col + 47616);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(vals[93]) : "memory");
            }
            {
                uint32_t _addr_98 = static_cast<uint32_t>(stage_col + 48128);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(vals[94]) : "memory");
            }
            {
                uint32_t _addr_99 = static_cast<uint32_t>(stage_col + 48640);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_99), "f"(vals[95]) : "memory");
            }
            {
                uint32_t _addr_100 = static_cast<uint32_t>(stage_col + 49152);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_100), "f"(vals[96]) : "memory");
            }
            {
                uint32_t _addr_101 = static_cast<uint32_t>(stage_col + 49664);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(vals[97]) : "memory");
            }
            {
                uint32_t _addr_102 = static_cast<uint32_t>(stage_col + 50176);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(vals[98]) : "memory");
            }
            {
                uint32_t _addr_103 = static_cast<uint32_t>(stage_col + 50688);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(vals[99]) : "memory");
            }
            {
                uint32_t _addr_104 = static_cast<uint32_t>(stage_col + 51200);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(vals[100]) : "memory");
            }
            {
                uint32_t _addr_105 = static_cast<uint32_t>(stage_col + 51712);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(vals[101]) : "memory");
            }
            {
                uint32_t _addr_106 = static_cast<uint32_t>(stage_col + 52224);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(vals[102]) : "memory");
            }
            {
                uint32_t _addr_107 = static_cast<uint32_t>(stage_col + 52736);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(vals[103]) : "memory");
            }
            {
                uint32_t _addr_108 = static_cast<uint32_t>(stage_col + 53248);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_108), "f"(vals[104]) : "memory");
            }
            {
                uint32_t _addr_109 = static_cast<uint32_t>(stage_col + 53760);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_109), "f"(vals[105]) : "memory");
            }
            {
                uint32_t _addr_110 = static_cast<uint32_t>(stage_col + 54272);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_110), "f"(vals[106]) : "memory");
            }
            {
                uint32_t _addr_111 = static_cast<uint32_t>(stage_col + 54784);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_111), "f"(vals[107]) : "memory");
            }
            {
                uint32_t _addr_112 = static_cast<uint32_t>(stage_col + 55296);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_112), "f"(vals[108]) : "memory");
            }
            {
                uint32_t _addr_113 = static_cast<uint32_t>(stage_col + 55808);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_113), "f"(vals[109]) : "memory");
            }
            {
                uint32_t _addr_114 = static_cast<uint32_t>(stage_col + 56320);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_114), "f"(vals[110]) : "memory");
            }
            {
                uint32_t _addr_115 = static_cast<uint32_t>(stage_col + 56832);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_115), "f"(vals[111]) : "memory");
            }
            {
                uint32_t _addr_116 = static_cast<uint32_t>(stage_col + 57344);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(vals[112]) : "memory");
            }
            {
                uint32_t _addr_117 = static_cast<uint32_t>(stage_col + 57856);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_117), "f"(vals[113]) : "memory");
            }
            {
                uint32_t _addr_118 = static_cast<uint32_t>(stage_col + 58368);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_118), "f"(vals[114]) : "memory");
            }
            {
                uint32_t _addr_119 = static_cast<uint32_t>(stage_col + 58880);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_119), "f"(vals[115]) : "memory");
            }
            {
                uint32_t _addr_120 = static_cast<uint32_t>(stage_col + 59392);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_120), "f"(vals[116]) : "memory");
            }
            {
                uint32_t _addr_121 = static_cast<uint32_t>(stage_col + 59904);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_121), "f"(vals[117]) : "memory");
            }
            {
                uint32_t _addr_122 = static_cast<uint32_t>(stage_col + 60416);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_122), "f"(vals[118]) : "memory");
            }
            {
                uint32_t _addr_123 = static_cast<uint32_t>(stage_col + 60928);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_123), "f"(vals[119]) : "memory");
            }
            {
                uint32_t _addr_124 = static_cast<uint32_t>(stage_col + 61440);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_124), "f"(vals[120]) : "memory");
            }
            {
                uint32_t _addr_125 = static_cast<uint32_t>(stage_col + 61952);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_125), "f"(vals[121]) : "memory");
            }
            {
                uint32_t _addr_126 = static_cast<uint32_t>(stage_col + 62464);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_126), "f"(vals[122]) : "memory");
            }
            {
                uint32_t _addr_127 = static_cast<uint32_t>(stage_col + 62976);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_127), "f"(vals[123]) : "memory");
            }
            {
                uint32_t _addr_128 = static_cast<uint32_t>(stage_col + 63488);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_128), "f"(vals[124]) : "memory");
            }
            {
                uint32_t _addr_129 = static_cast<uint32_t>(stage_col + 64000);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_129), "f"(vals[125]) : "memory");
            }
            {
                uint32_t _addr_130 = static_cast<uint32_t>(stage_col + 64512);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_130), "f"(vals[126]) : "memory");
            }
            {
                uint32_t _addr_131 = static_cast<uint32_t>(stage_col + 65024);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_131), "f"(vals[127]) : "memory");
            }
        } else {
            {
                uint32_t _addr_132 = static_cast<uint32_t>(stage_col);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_132), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_133 = static_cast<uint32_t>(stage_col + 512);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_133), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_134 = static_cast<uint32_t>(stage_col + 1024);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_134), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_135 = static_cast<uint32_t>(stage_col + 1536);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_135), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_136 = static_cast<uint32_t>(stage_col + 2048);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_136), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_137 = static_cast<uint32_t>(stage_col + 2560);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_137), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_138 = static_cast<uint32_t>(stage_col + 3072);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_138), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_139 = static_cast<uint32_t>(stage_col + 3584);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_139), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_140 = static_cast<uint32_t>(stage_col + 4096);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_140), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_141 = static_cast<uint32_t>(stage_col + 4608);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_141), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_142 = static_cast<uint32_t>(stage_col + 5120);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_142), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_143 = static_cast<uint32_t>(stage_col + 5632);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_143), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_144 = static_cast<uint32_t>(stage_col + 6144);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_144), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_145 = static_cast<uint32_t>(stage_col + 6656);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_145), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_146 = static_cast<uint32_t>(stage_col + 7168);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_146), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_147 = static_cast<uint32_t>(stage_col + 7680);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_147), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_148 = static_cast<uint32_t>(stage_col + 8192);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_148), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_149 = static_cast<uint32_t>(stage_col + 8704);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_149), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_150 = static_cast<uint32_t>(stage_col + 9216);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_150), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_151 = static_cast<uint32_t>(stage_col + 9728);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_151), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_152 = static_cast<uint32_t>(stage_col + 10240);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_152), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_153 = static_cast<uint32_t>(stage_col + 10752);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_153), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_154 = static_cast<uint32_t>(stage_col + 11264);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_154), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_155 = static_cast<uint32_t>(stage_col + 11776);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_155), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_156 = static_cast<uint32_t>(stage_col + 12288);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_156), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_157 = static_cast<uint32_t>(stage_col + 12800);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_157), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_158 = static_cast<uint32_t>(stage_col + 13312);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_158), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_159 = static_cast<uint32_t>(stage_col + 13824);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_159), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_160 = static_cast<uint32_t>(stage_col + 14336);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_160), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_161 = static_cast<uint32_t>(stage_col + 14848);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_161), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_162 = static_cast<uint32_t>(stage_col + 15360);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_162), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_163 = static_cast<uint32_t>(stage_col + 15872);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_163), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_164 = static_cast<uint32_t>(stage_col + 16384);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_164), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_165 = static_cast<uint32_t>(stage_col + 16896);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_165), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_166 = static_cast<uint32_t>(stage_col + 17408);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_166), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_167 = static_cast<uint32_t>(stage_col + 17920);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_167), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_168 = static_cast<uint32_t>(stage_col + 18432);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_168), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_169 = static_cast<uint32_t>(stage_col + 18944);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_169), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_170 = static_cast<uint32_t>(stage_col + 19456);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_170), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_171 = static_cast<uint32_t>(stage_col + 19968);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_171), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_172 = static_cast<uint32_t>(stage_col + 20480);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_172), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_173 = static_cast<uint32_t>(stage_col + 20992);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_173), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_174 = static_cast<uint32_t>(stage_col + 21504);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_174), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_175 = static_cast<uint32_t>(stage_col + 22016);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_175), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_176 = static_cast<uint32_t>(stage_col + 22528);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_176), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_177 = static_cast<uint32_t>(stage_col + 23040);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_177), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_178 = static_cast<uint32_t>(stage_col + 23552);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_178), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_179 = static_cast<uint32_t>(stage_col + 24064);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_179), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_180 = static_cast<uint32_t>(stage_col + 24576);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_180), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_181 = static_cast<uint32_t>(stage_col + 25088);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_181), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_182 = static_cast<uint32_t>(stage_col + 25600);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_182), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_183 = static_cast<uint32_t>(stage_col + 26112);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_183), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_184 = static_cast<uint32_t>(stage_col + 26624);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_184), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_185 = static_cast<uint32_t>(stage_col + 27136);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_185), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_186 = static_cast<uint32_t>(stage_col + 27648);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_186), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_187 = static_cast<uint32_t>(stage_col + 28160);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_187), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_188 = static_cast<uint32_t>(stage_col + 28672);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_188), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_189 = static_cast<uint32_t>(stage_col + 29184);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_189), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_190 = static_cast<uint32_t>(stage_col + 29696);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_190), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_191 = static_cast<uint32_t>(stage_col + 30208);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_191), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_192 = static_cast<uint32_t>(stage_col + 30720);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_192), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_193 = static_cast<uint32_t>(stage_col + 31232);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_193), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_194 = static_cast<uint32_t>(stage_col + 31744);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_194), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_195 = static_cast<uint32_t>(stage_col + 32256);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_195), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_196 = static_cast<uint32_t>(stage_col + 32768);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_196), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_197 = static_cast<uint32_t>(stage_col + 33280);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_197), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_198 = static_cast<uint32_t>(stage_col + 33792);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_198), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_199 = static_cast<uint32_t>(stage_col + 34304);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_199), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_200 = static_cast<uint32_t>(stage_col + 34816);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_200), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_201 = static_cast<uint32_t>(stage_col + 35328);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_201), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_202 = static_cast<uint32_t>(stage_col + 35840);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_202), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_203 = static_cast<uint32_t>(stage_col + 36352);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_203), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_204 = static_cast<uint32_t>(stage_col + 36864);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_204), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_205 = static_cast<uint32_t>(stage_col + 37376);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_205), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_206 = static_cast<uint32_t>(stage_col + 37888);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_206), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_207 = static_cast<uint32_t>(stage_col + 38400);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_207), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_208 = static_cast<uint32_t>(stage_col + 38912);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_208), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_209 = static_cast<uint32_t>(stage_col + 39424);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_209), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_210 = static_cast<uint32_t>(stage_col + 39936);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_210), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_211 = static_cast<uint32_t>(stage_col + 40448);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_211), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_212 = static_cast<uint32_t>(stage_col + 40960);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_212), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_213 = static_cast<uint32_t>(stage_col + 41472);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_213), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_214 = static_cast<uint32_t>(stage_col + 41984);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_214), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_215 = static_cast<uint32_t>(stage_col + 42496);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_215), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_216 = static_cast<uint32_t>(stage_col + 43008);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_216), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_217 = static_cast<uint32_t>(stage_col + 43520);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_217), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_218 = static_cast<uint32_t>(stage_col + 44032);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_218), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_219 = static_cast<uint32_t>(stage_col + 44544);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_219), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_220 = static_cast<uint32_t>(stage_col + 45056);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_220), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_221 = static_cast<uint32_t>(stage_col + 45568);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_221), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_222 = static_cast<uint32_t>(stage_col + 46080);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_222), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_223 = static_cast<uint32_t>(stage_col + 46592);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_223), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_224 = static_cast<uint32_t>(stage_col + 47104);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_224), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_225 = static_cast<uint32_t>(stage_col + 47616);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_225), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_226 = static_cast<uint32_t>(stage_col + 48128);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_226), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_227 = static_cast<uint32_t>(stage_col + 48640);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_227), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_228 = static_cast<uint32_t>(stage_col + 49152);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_228), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_229 = static_cast<uint32_t>(stage_col + 49664);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_229), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_230 = static_cast<uint32_t>(stage_col + 50176);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_230), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_231 = static_cast<uint32_t>(stage_col + 50688);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_231), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_232 = static_cast<uint32_t>(stage_col + 51200);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_232), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_233 = static_cast<uint32_t>(stage_col + 51712);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_233), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_234 = static_cast<uint32_t>(stage_col + 52224);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_234), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_235 = static_cast<uint32_t>(stage_col + 52736);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_235), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_236 = static_cast<uint32_t>(stage_col + 53248);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_236), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_237 = static_cast<uint32_t>(stage_col + 53760);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_237), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_238 = static_cast<uint32_t>(stage_col + 54272);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_238), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_239 = static_cast<uint32_t>(stage_col + 54784);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_239), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_240 = static_cast<uint32_t>(stage_col + 55296);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_240), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_241 = static_cast<uint32_t>(stage_col + 55808);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_241), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_242 = static_cast<uint32_t>(stage_col + 56320);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_242), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_243 = static_cast<uint32_t>(stage_col + 56832);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_243), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_244 = static_cast<uint32_t>(stage_col + 57344);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_244), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_245 = static_cast<uint32_t>(stage_col + 57856);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_245), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_246 = static_cast<uint32_t>(stage_col + 58368);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_246), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_247 = static_cast<uint32_t>(stage_col + 58880);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_247), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_248 = static_cast<uint32_t>(stage_col + 59392);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_248), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_249 = static_cast<uint32_t>(stage_col + 59904);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_249), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_250 = static_cast<uint32_t>(stage_col + 60416);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_250), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_251 = static_cast<uint32_t>(stage_col + 60928);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_251), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_252 = static_cast<uint32_t>(stage_col + 61440);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_252), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_253 = static_cast<uint32_t>(stage_col + 61952);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_253), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_254 = static_cast<uint32_t>(stage_col + 62464);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_254), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_255 = static_cast<uint32_t>(stage_col + 62976);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_255), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_256 = static_cast<uint32_t>(stage_col + 63488);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_256), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_257 = static_cast<uint32_t>(stage_col + 64000);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_257), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_258 = static_cast<uint32_t>(stage_col + 64512);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_258), "f"(zero4[0]) : "memory");
            }
            {
                uint32_t _addr_259 = static_cast<uint32_t>(stage_col + 65024);
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_259), "f"(zero4[0]) : "memory");
            }
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
        int key_tile = n_1 * 2 + qb_local / 64;
        int row_in_tile = qb_local - qb_local / 64 * 64;
        unsigned int stage_row = stage_smem_addr + (unsigned int)(qb_local * 512);
        int group_0 = m_1 * 16;
        if (group_0 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 16));
            unsigned int _packed_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_0[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_0[_pair]) << 16);
                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_0[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_1[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_1[_pair]) << 16);
                (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_1[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_0[0];
            lo[4] = xs[4] - _unpack_bf16_1[0];
            lo[1] = xs[1] - _unpack_bf16_0[1];
            lo[5] = xs[5] - _unpack_bf16_1[1];
            lo[2] = xs[2] - _unpack_bf16_0[2];
            lo[6] = xs[6] - _unpack_bf16_1[2];
            lo[3] = xs[3] - _unpack_bf16_0[3];
            lo[7] = xs[7] - _unpack_bf16_1[3];
            unsigned int _packed_2[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_3[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_0[0];
            words[1] = _packed_0[1];
            words[2] = _packed_2[0];
            words[3] = _packed_2[1];
            words[4] = _packed_1[0];
            words[5] = _packed_1[1];
            words[6] = _packed_3[0];
            words[7] = _packed_3[1];
            long long row_0 = (((long long)bh_1 * (long long)num_groups + (long long)group_0) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_0 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_0 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_1 = m_1 * 16 + 1;
        if (group_1 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 32));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 48));
            unsigned int _packed_4[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_5[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_2[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_4[_pair]) << 16);
                (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_4[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_3[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_5[_pair]) << 16);
                (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_5[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_2[0];
            lo[4] = xs[4] - _unpack_bf16_3[0];
            lo[1] = xs[1] - _unpack_bf16_2[1];
            lo[5] = xs[5] - _unpack_bf16_3[1];
            lo[2] = xs[2] - _unpack_bf16_2[2];
            lo[6] = xs[6] - _unpack_bf16_3[2];
            lo[3] = xs[3] - _unpack_bf16_2[3];
            lo[7] = xs[7] - _unpack_bf16_3[3];
            unsigned int _packed_6[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_7[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_7[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_4[0];
            words[1] = _packed_4[1];
            words[2] = _packed_6[0];
            words[3] = _packed_6[1];
            words[4] = _packed_5[0];
            words[5] = _packed_5[1];
            words[6] = _packed_7[0];
            words[7] = _packed_7[1];
            long long row_1 = (((long long)bh_1 * (long long)num_groups + (long long)group_1) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_1 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_1 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_2 = m_1 * 16 + 2;
        if (group_2 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 64));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 80));
            unsigned int _packed_8[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_9[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_9[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_4[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_8[_pair]) << 16);
                (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_8[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_5[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_9[_pair]) << 16);
                (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_9[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_4[0];
            lo[4] = xs[4] - _unpack_bf16_5[0];
            lo[1] = xs[1] - _unpack_bf16_4[1];
            lo[5] = xs[5] - _unpack_bf16_5[1];
            lo[2] = xs[2] - _unpack_bf16_4[2];
            lo[6] = xs[6] - _unpack_bf16_5[2];
            lo[3] = xs[3] - _unpack_bf16_4[3];
            lo[7] = xs[7] - _unpack_bf16_5[3];
            unsigned int _packed_10[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_11[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_11[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_8[0];
            words[1] = _packed_8[1];
            words[2] = _packed_10[0];
            words[3] = _packed_10[1];
            words[4] = _packed_9[0];
            words[5] = _packed_9[1];
            words[6] = _packed_11[0];
            words[7] = _packed_11[1];
            long long row_2 = (((long long)bh_1 * (long long)num_groups + (long long)group_2) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_2 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_2 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_3 = m_1 * 16 + 3;
        if (group_3 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 96));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 112));
            unsigned int _packed_12[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_13[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_13[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_6[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_12[_pair]) << 16);
                (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_12[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_7[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_13[_pair]) << 16);
                (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_13[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_6[0];
            lo[4] = xs[4] - _unpack_bf16_7[0];
            lo[1] = xs[1] - _unpack_bf16_6[1];
            lo[5] = xs[5] - _unpack_bf16_7[1];
            lo[2] = xs[2] - _unpack_bf16_6[2];
            lo[6] = xs[6] - _unpack_bf16_7[2];
            lo[3] = xs[3] - _unpack_bf16_6[3];
            lo[7] = xs[7] - _unpack_bf16_7[3];
            unsigned int _packed_14[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_14[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_15[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_15[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_12[0];
            words[1] = _packed_12[1];
            words[2] = _packed_14[0];
            words[3] = _packed_14[1];
            words[4] = _packed_13[0];
            words[5] = _packed_13[1];
            words[6] = _packed_15[0];
            words[7] = _packed_15[1];
            long long row_3 = (((long long)bh_1 * (long long)num_groups + (long long)group_3) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_3 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_3 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_4 = m_1 * 16 + 4;
        if (group_4 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 128));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 144));
            unsigned int _packed_16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_16[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_17[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_17[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_17[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_8[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_16[_pair]) << 16);
                (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_16[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_9[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_17[_pair]) << 16);
                (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_17[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_8[0];
            lo[4] = xs[4] - _unpack_bf16_9[0];
            lo[1] = xs[1] - _unpack_bf16_8[1];
            lo[5] = xs[5] - _unpack_bf16_9[1];
            lo[2] = xs[2] - _unpack_bf16_8[2];
            lo[6] = xs[6] - _unpack_bf16_9[2];
            lo[3] = xs[3] - _unpack_bf16_8[3];
            lo[7] = xs[7] - _unpack_bf16_9[3];
            unsigned int _packed_18[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_18[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_18[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_19[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_19[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_19[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_16[0];
            words[1] = _packed_16[1];
            words[2] = _packed_18[0];
            words[3] = _packed_18[1];
            words[4] = _packed_17[0];
            words[5] = _packed_17[1];
            words[6] = _packed_19[0];
            words[7] = _packed_19[1];
            long long row_4 = (((long long)bh_1 * (long long)num_groups + (long long)group_4) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_4 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_4 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_5 = m_1 * 16 + 5;
        if (group_5 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 160));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 176));
            unsigned int _packed_20[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_20[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_20[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_21[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_21[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_21[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_10[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_20[_pair]) << 16);
                (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_20[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_11[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_21[_pair]) << 16);
                (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_21[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_10[0];
            lo[4] = xs[4] - _unpack_bf16_11[0];
            lo[1] = xs[1] - _unpack_bf16_10[1];
            lo[5] = xs[5] - _unpack_bf16_11[1];
            lo[2] = xs[2] - _unpack_bf16_10[2];
            lo[6] = xs[6] - _unpack_bf16_11[2];
            lo[3] = xs[3] - _unpack_bf16_10[3];
            lo[7] = xs[7] - _unpack_bf16_11[3];
            unsigned int _packed_22[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_22[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_22[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_23[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_23[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_23[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_20[0];
            words[1] = _packed_20[1];
            words[2] = _packed_22[0];
            words[3] = _packed_22[1];
            words[4] = _packed_21[0];
            words[5] = _packed_21[1];
            words[6] = _packed_23[0];
            words[7] = _packed_23[1];
            long long row_5 = (((long long)bh_1 * (long long)num_groups + (long long)group_5) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_5 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_5 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_6 = m_1 * 16 + 6;
        if (group_6 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 192));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 208));
            unsigned int _packed_24[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_24[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_24[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_25[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_25[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_25[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_12[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_24[_pair]) << 16);
                (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_24[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_13[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_25[_pair]) << 16);
                (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_25[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_12[0];
            lo[4] = xs[4] - _unpack_bf16_13[0];
            lo[1] = xs[1] - _unpack_bf16_12[1];
            lo[5] = xs[5] - _unpack_bf16_13[1];
            lo[2] = xs[2] - _unpack_bf16_12[2];
            lo[6] = xs[6] - _unpack_bf16_13[2];
            lo[3] = xs[3] - _unpack_bf16_12[3];
            lo[7] = xs[7] - _unpack_bf16_13[3];
            unsigned int _packed_26[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_26[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_26[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_27[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_27[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_27[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_24[0];
            words[1] = _packed_24[1];
            words[2] = _packed_26[0];
            words[3] = _packed_26[1];
            words[4] = _packed_25[0];
            words[5] = _packed_25[1];
            words[6] = _packed_27[0];
            words[7] = _packed_27[1];
            long long row_6 = (((long long)bh_1 * (long long)num_groups + (long long)group_6) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_6 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_6 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_7 = m_1 * 16 + 7;
        if (group_7 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 224));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 240));
            unsigned int _packed_28[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_28[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_28[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_29[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_29[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_29[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_14[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_28[_pair]) << 16);
                (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_28[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_15[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_29[_pair]) << 16);
                (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_29[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_14[0];
            lo[4] = xs[4] - _unpack_bf16_15[0];
            lo[1] = xs[1] - _unpack_bf16_14[1];
            lo[5] = xs[5] - _unpack_bf16_15[1];
            lo[2] = xs[2] - _unpack_bf16_14[2];
            lo[6] = xs[6] - _unpack_bf16_15[2];
            lo[3] = xs[3] - _unpack_bf16_14[3];
            lo[7] = xs[7] - _unpack_bf16_15[3];
            unsigned int _packed_30[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_30[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_30[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_31[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_31[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_31[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_28[0];
            words[1] = _packed_28[1];
            words[2] = _packed_30[0];
            words[3] = _packed_30[1];
            words[4] = _packed_29[0];
            words[5] = _packed_29[1];
            words[6] = _packed_31[0];
            words[7] = _packed_31[1];
            long long row_7 = (((long long)bh_1 * (long long)num_groups + (long long)group_7) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_7 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_7 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_8 = m_1 * 16 + 8;
        if (group_8 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 256));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 272));
            unsigned int _packed_32[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_32[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_32[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_33[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_33[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_33[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_16[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_32[_pair]) << 16);
                (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_32[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_17[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_33[_pair]) << 16);
                (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_33[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_16[0];
            lo[4] = xs[4] - _unpack_bf16_17[0];
            lo[1] = xs[1] - _unpack_bf16_16[1];
            lo[5] = xs[5] - _unpack_bf16_17[1];
            lo[2] = xs[2] - _unpack_bf16_16[2];
            lo[6] = xs[6] - _unpack_bf16_17[2];
            lo[3] = xs[3] - _unpack_bf16_16[3];
            lo[7] = xs[7] - _unpack_bf16_17[3];
            unsigned int _packed_34[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_34[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_34[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_35[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_35[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_35[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_32[0];
            words[1] = _packed_32[1];
            words[2] = _packed_34[0];
            words[3] = _packed_34[1];
            words[4] = _packed_33[0];
            words[5] = _packed_33[1];
            words[6] = _packed_35[0];
            words[7] = _packed_35[1];
            long long row_8 = (((long long)bh_1 * (long long)num_groups + (long long)group_8) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_8 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_8 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_9 = m_1 * 16 + 9;
        if (group_9 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 288));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 304));
            unsigned int _packed_36[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_36[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_36[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_37[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_37[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_37[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_18[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_36[_pair]) << 16);
                (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_36[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_19[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_37[_pair]) << 16);
                (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_37[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_18[0];
            lo[4] = xs[4] - _unpack_bf16_19[0];
            lo[1] = xs[1] - _unpack_bf16_18[1];
            lo[5] = xs[5] - _unpack_bf16_19[1];
            lo[2] = xs[2] - _unpack_bf16_18[2];
            lo[6] = xs[6] - _unpack_bf16_19[2];
            lo[3] = xs[3] - _unpack_bf16_18[3];
            lo[7] = xs[7] - _unpack_bf16_19[3];
            unsigned int _packed_38[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_38[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_38[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_39[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_39[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_39[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_36[0];
            words[1] = _packed_36[1];
            words[2] = _packed_38[0];
            words[3] = _packed_38[1];
            words[4] = _packed_37[0];
            words[5] = _packed_37[1];
            words[6] = _packed_39[0];
            words[7] = _packed_39[1];
            long long row_9 = (((long long)bh_1 * (long long)num_groups + (long long)group_9) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_9 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_9 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_10 = m_1 * 16 + 10;
        if (group_10 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 320));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 336));
            unsigned int _packed_40[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_40[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_40[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_41[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_41[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_41[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_20[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_40[_pair]) << 16);
                (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_40[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_21[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_41[_pair]) << 16);
                (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_41[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_20[0];
            lo[4] = xs[4] - _unpack_bf16_21[0];
            lo[1] = xs[1] - _unpack_bf16_20[1];
            lo[5] = xs[5] - _unpack_bf16_21[1];
            lo[2] = xs[2] - _unpack_bf16_20[2];
            lo[6] = xs[6] - _unpack_bf16_21[2];
            lo[3] = xs[3] - _unpack_bf16_20[3];
            lo[7] = xs[7] - _unpack_bf16_21[3];
            unsigned int _packed_42[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_42[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_42[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_43[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_43[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_43[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_40[0];
            words[1] = _packed_40[1];
            words[2] = _packed_42[0];
            words[3] = _packed_42[1];
            words[4] = _packed_41[0];
            words[5] = _packed_41[1];
            words[6] = _packed_43[0];
            words[7] = _packed_43[1];
            long long row_10 = (((long long)bh_1 * (long long)num_groups + (long long)group_10) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_10 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_10 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_11 = m_1 * 16 + 11;
        if (group_11 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 352));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 368));
            unsigned int _packed_44[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_44[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_44[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_45[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_45[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_45[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_22[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_44[_pair]) << 16);
                (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_44[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_23[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_45[_pair]) << 16);
                (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_45[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_22[0];
            lo[4] = xs[4] - _unpack_bf16_23[0];
            lo[1] = xs[1] - _unpack_bf16_22[1];
            lo[5] = xs[5] - _unpack_bf16_23[1];
            lo[2] = xs[2] - _unpack_bf16_22[2];
            lo[6] = xs[6] - _unpack_bf16_23[2];
            lo[3] = xs[3] - _unpack_bf16_22[3];
            lo[7] = xs[7] - _unpack_bf16_23[3];
            unsigned int _packed_46[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_46[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_46[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_47[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_47[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_47[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_44[0];
            words[1] = _packed_44[1];
            words[2] = _packed_46[0];
            words[3] = _packed_46[1];
            words[4] = _packed_45[0];
            words[5] = _packed_45[1];
            words[6] = _packed_47[0];
            words[7] = _packed_47[1];
            long long row_11 = (((long long)bh_1 * (long long)num_groups + (long long)group_11) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_11 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_11 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_12 = m_1 * 16 + 12;
        if (group_12 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 384));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 400));
            unsigned int _packed_48[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_48[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_48[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_49[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_49[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_49[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_24[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_48[_pair]) << 16);
                (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_48[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_25[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_49[_pair]) << 16);
                (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_49[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_24[0];
            lo[4] = xs[4] - _unpack_bf16_25[0];
            lo[1] = xs[1] - _unpack_bf16_24[1];
            lo[5] = xs[5] - _unpack_bf16_25[1];
            lo[2] = xs[2] - _unpack_bf16_24[2];
            lo[6] = xs[6] - _unpack_bf16_25[2];
            lo[3] = xs[3] - _unpack_bf16_24[3];
            lo[7] = xs[7] - _unpack_bf16_25[3];
            unsigned int _packed_50[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_50[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_50[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_51[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_51[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_51[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_48[0];
            words[1] = _packed_48[1];
            words[2] = _packed_50[0];
            words[3] = _packed_50[1];
            words[4] = _packed_49[0];
            words[5] = _packed_49[1];
            words[6] = _packed_51[0];
            words[7] = _packed_51[1];
            long long row_12 = (((long long)bh_1 * (long long)num_groups + (long long)group_12) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_12 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_12 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_13 = m_1 * 16 + 13;
        if (group_13 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 416));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 432));
            unsigned int _packed_52[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_52[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_52[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_53[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_53[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_53[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_26[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_52[_pair]) << 16);
                (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_52[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_27[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_53[_pair]) << 16);
                (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_53[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_26[0];
            lo[4] = xs[4] - _unpack_bf16_27[0];
            lo[1] = xs[1] - _unpack_bf16_26[1];
            lo[5] = xs[5] - _unpack_bf16_27[1];
            lo[2] = xs[2] - _unpack_bf16_26[2];
            lo[6] = xs[6] - _unpack_bf16_27[2];
            lo[3] = xs[3] - _unpack_bf16_26[3];
            lo[7] = xs[7] - _unpack_bf16_27[3];
            unsigned int _packed_54[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_54[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_54[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_55[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_55[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_55[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_52[0];
            words[1] = _packed_52[1];
            words[2] = _packed_54[0];
            words[3] = _packed_54[1];
            words[4] = _packed_53[0];
            words[5] = _packed_53[1];
            words[6] = _packed_55[0];
            words[7] = _packed_55[1];
            long long row_13 = (((long long)bh_1 * (long long)num_groups + (long long)group_13) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_13 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_13 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_14 = m_1 * 16 + 14;
        if (group_14 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 448));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 464));
            unsigned int _packed_56[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_56[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_56[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_57[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_57[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_57[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_28[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_56[_pair]) << 16);
                (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_56[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_29[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_57[_pair]) << 16);
                (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_57[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_28[0];
            lo[4] = xs[4] - _unpack_bf16_29[0];
            lo[1] = xs[1] - _unpack_bf16_28[1];
            lo[5] = xs[5] - _unpack_bf16_29[1];
            lo[2] = xs[2] - _unpack_bf16_28[2];
            lo[6] = xs[6] - _unpack_bf16_29[2];
            lo[3] = xs[3] - _unpack_bf16_28[3];
            lo[7] = xs[7] - _unpack_bf16_29[3];
            unsigned int _packed_58[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_58[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_58[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_59[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_59[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_59[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_56[0];
            words[1] = _packed_56[1];
            words[2] = _packed_58[0];
            words[3] = _packed_58[1];
            words[4] = _packed_57[0];
            words[5] = _packed_57[1];
            words[6] = _packed_59[0];
            words[7] = _packed_59[1];
            long long row_14 = (((long long)bh_1 * (long long)num_groups + (long long)group_14) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_14 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_14 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        int group_15 = m_1 * 16 + 15;
        if (group_15 < num_groups) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[0])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(0) + 3]))
                : "r"(stage_row + 480));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xs[4])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xs[(4) + 3]))
                : "r"(stage_row + 496));
            unsigned int _packed_60[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                _packed_60[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                _packed_60[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_61[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                _packed_61[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                _packed_61[_lp + 1] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_30[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_60[_pair]) << 16);
                (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_60[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_31[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_packed_61[_pair]) << 16);
                (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_packed_61[_pair]) & 0xffff0000u);
            }
            lo[0] = xs[0] - _unpack_bf16_30[0];
            lo[4] = xs[4] - _unpack_bf16_31[0];
            lo[1] = xs[1] - _unpack_bf16_30[1];
            lo[5] = xs[5] - _unpack_bf16_31[1];
            lo[2] = xs[2] - _unpack_bf16_30[2];
            lo[6] = xs[6] - _unpack_bf16_31[2];
            lo[3] = xs[3] - _unpack_bf16_30[3];
            lo[7] = xs[7] - _unpack_bf16_31[3];
            unsigned int _packed_62[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                _packed_62[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 2], lo[_lp*2+1 + 2]));
                _packed_62[_lp + 1] = *(uint32_t*)&_bf2;
            }
            unsigned int _packed_63[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 4], lo[_lp*2+1 + 4]));
                _packed_63[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 6], lo[_lp*2+1 + 6]));
                _packed_63[_lp + 1] = *(uint32_t*)&_bf2;
            }
            words[0] = _packed_60[0];
            words[1] = _packed_60[1];
            words[2] = _packed_62[0];
            words[3] = _packed_62[1];
            words[4] = _packed_61[0];
            words[5] = _packed_61[1];
            words[6] = _packed_63[0];
            words[7] = _packed_63[1];
            long long row_15 = (((long long)bh_1 * (long long)num_groups + (long long)group_15) * (long long)num_key_tiles + (long long)key_tile) * 64 + (long long)row_in_tile;
            reinterpret_cast<int4*>(tab + (row_15 * 16))[0] = reinterpret_cast<int4*>(words)[0];
            reinterpret_cast<int4*>(tab + (row_15 * 16 + 8))[0] = reinterpret_cast<int4*>(words + (4))[0];
        }
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncthreads();
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
