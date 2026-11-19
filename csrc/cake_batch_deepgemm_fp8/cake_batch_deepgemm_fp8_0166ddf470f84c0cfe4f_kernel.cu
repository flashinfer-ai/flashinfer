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

template <typename T, int N>
struct CakeParamArray {
    T v[N];
    __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}





__device__ __forceinline__ void tcgen05_mma_mxf8f6f4_bs_k64_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x7FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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





__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
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



__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}




extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_batch_deepgemm_fp8_0166ddf470f84c0cfe4f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C_tma, const __grid_constant__ CUtensorMap SFA_packed, const __grid_constant__ CUtensorMap SFB_packed, int* __restrict__ masked_m, unsigned int num_groups, unsigned int shape_m, unsigned int N, unsigned int K, int g_cut, int* __restrict__ counters0, int* __restrict__ counters1, const CakeParamArray<int, 256> sm_table)
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
    #define work_full_addr (mbar_base + 0)
    #define work_empty_addr (mbar_base + 32)
    #define tma_full_addr (mbar_base + 64)
    #define tma_free_addr (mbar_base + 112)
    #define scale_ready_addr (mbar_base + 160)
    #define mainloop_done_addr (mbar_base + 208)
    #define epilogue_done_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int epi_staging_addr = smem + 1024;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_a_addr = smem + 5120;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_b_addr = smem + 9216;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 41984);
    const int smem_sfa_addr = smem + 41984;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 42496);
    const int smem_sfb_addr = smem + 42496;
    unsigned int* smem_sfw = reinterpret_cast<unsigned int*>(smem_raw + 43008);
    const int smem_sfw_addr = smem + 43008;
    unsigned int* work_slot = reinterpret_cast<unsigned int*>(smem_raw + 238592);
    const int work_slot_addr = smem + 238592;

    // Kernel setup ops
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA_packed))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB_packed))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");
    unsigned int ee_lane = (unsigned int)lane;
    unsigned int ee_groups = (unsigned int)64;
    unsigned int ee_grid_n = (unsigned int)32;
    unsigned int ee_last = ee_groups - 1;
    unsigned int ee_m_blocks = 0;
    unsigned int ee_m = (unsigned int)masked_m[((ee_lane < ee_groups) ? ee_lane : ee_last)];
    unsigned int ee_lane_hi = ee_lane + 32;
    unsigned int ee_m_hi = (unsigned int)masked_m[((ee_lane_hi < ee_groups) ? ee_lane_hi : ee_last)];
    unsigned int ee_first = (unsigned int)bid - (unsigned int)cta_rank;

    // Mbarrier init (30 barriers at smem_raw[0..240))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // work_empty: 4 barriers, init_count=416
            mbarrier_init(smem + 32, 416);
            mbarrier_init(smem + 40, 416);
            mbarrier_init(smem + 48, 416);
            mbarrier_init(smem + 56, 416);
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // tma_free: 6 barriers, init_count=33
            mbarrier_init(smem + 112, 33);
            mbarrier_init(smem + 120, 33);
            mbarrier_init(smem + 128, 33);
            mbarrier_init(smem + 136, 33);
            mbarrier_init(smem + 144, 33);
            mbarrier_init(smem + 152, 33);
            // scale_ready: 6 barriers, init_count=64
            mbarrier_init(smem + 160, 64);
            mbarrier_init(smem + 168, 64);
            mbarrier_init(smem + 176, 64);
            mbarrier_init(smem + 184, 64);
            mbarrier_init(smem + 192, 64);
            mbarrier_init(smem + 200, 64);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 224, 8);
            mbarrier_init(smem + 232, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned int ee_blocks = 0;
    if (ee_lane < ee_groups) {
        ee_blocks = (ee_m + 32 - 1) / 32;
    }
    unsigned int ee_blocks_hi = 0;
    if (ee_lane_hi < ee_groups) {
        ee_blocks_hi = (ee_m_hi + 32 - 1) / 32;
    }
    uint32_t ws_incl = ee_blocks;
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, ws_incl, 1, 32); if (lane >= 1) { ws_incl += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, ws_incl, 2, 32); if (lane >= 2) { ws_incl += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, ws_incl, 4, 32); if (lane >= 4) { ws_incl += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, ws_incl, 8, 32); if (lane >= 8) { ws_incl += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, ws_incl, 16, 32); if (lane >= 16) { ws_incl += peer; } }
    unsigned int ws_one = 1;
    unsigned int ws_zero = 0;
    unsigned int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, ws_incl, 31);
    ee_m_blocks = _shfl_sync_0;
    unsigned int ws_32 = 32;
    uint32_t _warp_scan_sum_1 = ee_blocks_hi;
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 1, 32); if (lane >= 1) { _warp_scan_sum_1 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 2, 32); if (lane >= 2) { _warp_scan_sum_1 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 4, 32); if (lane >= 4) { _warp_scan_sum_1 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 8, 32); if (lane >= 8) { _warp_scan_sum_1 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_1, 16, 32); if (lane >= 16) { _warp_scan_sum_1 += peer; } }
    unsigned int ws_incl_hi = _warp_scan_sum_1 + ee_m_blocks;
    unsigned int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, ws_incl_hi, 31);
    ee_m_blocks = _shfl_sync_2;

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 240);
    if (warp == 2) {
        int _tmem_hold = smem + 240;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_accum = taddr;
    const unsigned int tmem_tmem_sfb = taddr + 64;
    const unsigned int tmem_tmem_sfa = taddr + 72;
    const unsigned int tmem_tmem_sfa1 = taddr + 76;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int grid_n = (unsigned int)32;
            unsigned int num_k_blocks = (unsigned int)28;
            unsigned int sf_packed_cols = (unsigned int)14;
            unsigned int cta_rank_0 = (unsigned int)cta_rank;
            unsigned int num_bids_u = (unsigned int)num_bids;
            unsigned int bid_u = (unsigned int)bid;
            unsigned int current_group_idx = 0;
            unsigned int current_m_cumsum = 0;
            unsigned int group_bound = (unsigned int)64;
            unsigned int max_m_blocks = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles = group_bound * max_m_blocks * grid_n;
            unsigned int max_scheduler_iters = max_total_tiles / 2 + 1;
            unsigned int ld_work_stage = 0;
            unsigned int di_m = 0;
            if ((unsigned int)lane < (unsigned int)64) {
                di_m = (unsigned int)masked_m[(unsigned int)lane];
            }
            unsigned int di_blocks = (di_m + 32 - 1) / 32;
            unsigned int di_own = (((unsigned int)lane < (unsigned int)g_cut) ? di_blocks : (unsigned int)0);
            unsigned int _warp_redux_0;
            asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_0) : "r"(di_own));
            unsigned int di_items0 = _warp_redux_0 * grid_n / 2;
            unsigned int _warp_redux_1;
            asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_1) : "r"(di_blocks - di_own));
            unsigned int di_items1 = _warp_redux_1 * grid_n / 2;
            int dom_items0 = (int)di_items0;
            int dom_items1 = (int)di_items1;
            unsigned int di_m_0 = 0;
            if ((unsigned int)lane + 32 < (unsigned int)64) {
                di_m_0 = (unsigned int)masked_m[(unsigned int)lane + 32];
            }
            unsigned int di_blocks_0 = (di_m_0 + 32 - 1) / 32;
            unsigned int di_own_0 = (((unsigned int)lane + 32 < (unsigned int)g_cut) ? di_blocks_0 : (unsigned int)0);
            unsigned int _warp_redux_2;
            asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_2) : "r"(di_own_0));
            unsigned int di_items0_0 = _warp_redux_2 * grid_n / 2;
            unsigned int _warp_redux_3;
            asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3) : "r"(di_blocks_0 - di_own_0));
            unsigned int di_items1_0 = _warp_redux_3 * grid_n / 2;
            dom_items0 = dom_items0 + (int)di_items0_0;
            dom_items1 = dom_items1 + (int)di_items1_0;
            int dom_home = 0;
            unsigned int _phase_work_empty = 1;
            if (elect_sync()) {
                if (cta_rank == 0) {
                    uint32_t _read_special_reg_4;
                    asm volatile("mov.u32 %0, %%smid;" : "=r"(_read_special_reg_4));
                    int sm = (int)_read_special_reg_4;
                    int entry = sm_table[sm];
                    uint32_t _read_special_reg_5;
                    asm("mov.u32 %0, %%locality_domain_id;" : "=r"(_read_special_reg_5));
                    int hw_die = (int)_read_special_reg_5;
                    int table_die = entry >> 20 & 15;
                    int die_xor = table_die ^ hw_die;
                    dom_home = table_die;
                    mbarrier_wait_cluster_hint(work_empty_addr + (ld_work_stage) * 8, _phase_work_empty, 10000000);
                    int claim_die = dom_home;
                    int claim_item = 0;
                    if (dom_home == 0) {
                        int _atomic_fetch_add_6 = atomicAdd(counters0, 1);
                        claim_item = _atomic_fetch_add_6;
                        if (claim_item >= dom_items0) {
                            claim_die = 1;
                            int _atomic_fetch_add_7 = atomicAdd(counters1, 1);
                            claim_item = _atomic_fetch_add_7;
                        }
                    } else {
                        int _atomic_fetch_add_8 = atomicAdd(counters1, 1);
                        claim_item = _atomic_fetch_add_8;
                        if (claim_item >= dom_items1) {
                            claim_die = 0;
                            int _atomic_fetch_add_9 = atomicAdd(counters0, 1);
                            claim_item = _atomic_fetch_add_9;
                        }
                    }
                    int claim_valid = 0;
                    if (claim_die == 0) {
                        if (claim_item < dom_items0) {
                            claim_valid = 1;
                        }
                    } else if (claim_item < dom_items1) {
                        claim_valid = 1;
                    }
                    int c_pair = claim_item + ((claim_die == 1) ? dom_items0 : 0);
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(work_full_addr + (ld_work_stage) * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(work_full_addr + (ld_work_stage) * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                    uint32_t _cluster_mapa_10;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_10) : "r"(work_slot_addr + ld_work_stage * 16), "r"(0));
                    uint32_t _mbar_peer_11;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_11) : "r"(work_full_addr + (ld_work_stage) * 8), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_10), "r"((unsigned int)claim_valid), "r"((unsigned int)c_pair), "r"((unsigned int)claim_die), "r"(0), "r"(_mbar_peer_11) : "memory");
                    uint32_t _cluster_mapa_12;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_12) : "r"(work_slot_addr + ld_work_stage * 16), "r"(1));
                    uint32_t _mbar_peer_13;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_13) : "r"(work_full_addr + (ld_work_stage) * 8), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_12), "r"((unsigned int)claim_valid), "r"((unsigned int)c_pair), "r"((unsigned int)claim_die), "r"(0), "r"(_mbar_peer_13) : "memory");
                }
            }
            unsigned int _phase_work_full = 0;
            unsigned int _phase_tma_free = 1;
            #pragma unroll 1
            for (int current_iter = 0; current_iter < max_scheduler_iters; current_iter++) {
                unsigned int next_block_idx = (unsigned int)current_iter * num_bids_u + bid_u;
                mbarrier_wait(work_full_addr + (ld_work_stage) * 8, _phase_work_full);
                unsigned int posted[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&posted[0])), "=r"(*reinterpret_cast<uint32_t*>(&posted[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&posted[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&posted[(0) + 3]))
                    : "r"(work_slot_addr + ld_work_stage * 16));
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (ld_work_stage) * 8), "r"(0) : "memory");
                ld_work_stage += 1;
                if (ld_work_stage == 4) { ld_work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                int w_valid = (int)posted[0];
                unsigned int w_pair = posted[1];
                next_block_idx = w_pair * 2 + (unsigned int)cta_rank;
                int ld_valid = w_valid;
                if (ld_valid == 0) {
                    break;
                }
                if (elect_sync()) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (ld_work_stage) * 8, _phase_work_empty, 10000000);
                        int claim_die_1 = dom_home;
                        int claim_item_1 = 0;
                        if (dom_home == 0) {
                            int _atomic_fetch_add_14 = atomicAdd(counters0, 1);
                            claim_item_1 = _atomic_fetch_add_14;
                            if (claim_item_1 >= dom_items0) {
                                claim_die_1 = 1;
                                int _atomic_fetch_add_15 = atomicAdd(counters1, 1);
                                claim_item_1 = _atomic_fetch_add_15;
                            }
                        } else {
                            int _atomic_fetch_add_16 = atomicAdd(counters1, 1);
                            claim_item_1 = _atomic_fetch_add_16;
                            if (claim_item_1 >= dom_items1) {
                                claim_die_1 = 0;
                                int _atomic_fetch_add_17 = atomicAdd(counters0, 1);
                                claim_item_1 = _atomic_fetch_add_17;
                            }
                        }
                        int claim_valid_1 = 0;
                        if (claim_die_1 == 0) {
                            if (claim_item_1 < dom_items0) {
                                claim_valid_1 = 1;
                            }
                        } else if (claim_item_1 < dom_items1) {
                            claim_valid_1 = 1;
                        }
                        int c_pair_1 = claim_item_1 + ((claim_die_1 == 1) ? dom_items0 : 0);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (ld_work_stage) * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (ld_work_stage) * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        uint32_t _cluster_mapa_18;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_18) : "r"(work_slot_addr + ld_work_stage * 16), "r"(0));
                        uint32_t _mbar_peer_19;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_19) : "r"(work_full_addr + (ld_work_stage) * 8), "r"(0));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_18), "r"((unsigned int)claim_valid_1), "r"((unsigned int)c_pair_1), "r"((unsigned int)claim_die_1), "r"(0), "r"(_mbar_peer_19) : "memory");
                        uint32_t _cluster_mapa_20;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_20) : "r"(work_slot_addr + ld_work_stage * 16), "r"(1));
                        uint32_t _mbar_peer_21;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_21) : "r"(work_full_addr + (ld_work_stage) * 8), "r"(1));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_20), "r"((unsigned int)claim_valid_1), "r"((unsigned int)c_pair_1), "r"((unsigned int)claim_die_1), "r"(0), "r"(_mbar_peer_21) : "memory");
                    }
                }
                int has_block = 0;
                unsigned int m_blocks_g = 0;
                unsigned int ws_le = ((next_block_idx >= ws_incl * grid_n) ? ws_one : ws_zero);
                unsigned int ws_g;
                asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(ws_g) : "r"(ws_le));
                unsigned int ws_le_hi = ((next_block_idx >= ws_incl_hi * grid_n) ? ws_one : ws_zero);
                unsigned int _warp_redux_22;
                asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_22) : "r"(ws_le_hi));
                ws_g = ws_g + _warp_redux_22;
                if (ws_g < group_bound) {
                    unsigned int ws_lane = ((ws_g >= ws_32) ? ws_g - ws_32 : ws_g);
                    unsigned int ws_src_b = ((ws_g >= ws_32) ? ee_blocks_hi : ee_blocks);
                    unsigned int ws_src_i = ((ws_g >= ws_32) ? ws_incl_hi : ws_incl);
                    unsigned int ws_blocks = __shfl_sync(0xFFFFFFFF, ws_src_b, ws_lane);
                    current_group_idx = ws_g;
                    unsigned int _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, ws_src_i, ws_lane);
                    current_m_cumsum = _shfl_sync_23 - ws_blocks;
                    m_blocks_g = ws_blocks;
                    has_block = 1;
                }
                if (has_block == 0) {
                    break;
                }
                unsigned int group_m = (unsigned int)masked_m[current_group_idx];
                unsigned int block_idx = next_block_idx - current_m_cumsum * grid_n;
                unsigned int blocks_per_l2_group = m_blocks_g * 2;
                unsigned int l2_group = block_idx / blocks_per_l2_group;
                unsigned int first_n_block = l2_group * 2;
                unsigned int in_l2_group = block_idx % blocks_per_l2_group;
                unsigned int m_block = in_l2_group / 2;
                unsigned int n_block = first_n_block + in_l2_group % 2;
                unsigned int off_m = m_block * 32;
                unsigned int off_n = n_block * 128;
                unsigned int remaining_m = group_m - off_m;
                unsigned int runtime_mma_n = (((remaining_m + 15) / 16 * 16) < (32) ? ((remaining_m + 15) / 16 * 16) : (32));
                unsigned int load_block_m = runtime_mma_n / 2;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < num_k_blocks; iter_k++) {
                    mbarrier_wait_hint(tma_free_addr + (load_stage) * 8, _phase_tma_free, 10000000);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_b_addr + load_stage * 38912), "l"((&B)), "r"(0), "r"(off_n), "r"((unsigned int)iter_k * 2), "r"(current_group_idx),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_a_addr + load_stage * 38912), "l"((&A)), "r"(0), "r"(off_m + cta_rank_0 * load_block_m), "r"((unsigned int)iter_k * 2), "r"(current_group_idx),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x1000000000000000ULL) : "memory");
                        if (iter_k % 2 == 0) {
                            unsigned int sf_outer = current_group_idx * sf_packed_cols + (unsigned int)iter_k / 2;
                            tma_2d_gmem2smem(smem_sfa_addr + load_stage * 38912, (&SFA_packed), off_m, sf_outer, tma_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(smem_sfb_addr + load_stage * 38912, (&SFB_packed), off_n, sf_outer, tma_full_addr + (load_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 36864 + ((iter_k % 2 == 0) ? 640 : 0));
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_tma_free ^= 1; }
                }
            }
            if (elect_sync()) {
                if (cta_rank == 0) {
                    __threadfence();
                    int retired = atomicAdd(counters0 + 1, 1);
                    if (retired == (int)num_clusters - 1) {
                        __threadfence();
                        counters0[0] = 0;
                        counters0[1] = 0;
                    }
                    __threadfence();
                    int retired_0 = atomicAdd(counters1 + 1, 1);
                    if (retired_0 == (int)num_clusters - 1) {
                        __threadfence();
                        counters1[0] = 0;
                        counters1[1] = 0;
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int tma_stage = 0;
            unsigned int epi_stage = 0;
            unsigned int grid_n_1 = (unsigned int)32;
            unsigned int num_k_blocks_1 = (unsigned int)28;
            unsigned int num_bids_u_1 = (unsigned int)num_bids;
            unsigned int bid_u_1 = (unsigned int)bid;
            unsigned int current_group_idx_1 = 0;
            unsigned int current_m_cumsum_1 = 0;
            unsigned int group_bound_1 = (unsigned int)64;
            unsigned int max_m_blocks_1 = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles_1 = group_bound_1 * max_m_blocks_1 * grid_n_1;
            unsigned int max_scheduler_iters_1 = max_total_tiles_1 / 2 + 1;
            unsigned int mm_work_stage = 0;
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_scale_ready = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int current_iter_1 = 0; current_iter_1 < max_scheduler_iters_1; current_iter_1++) {
                    unsigned int next_block_idx_1 = (unsigned int)current_iter_1 * num_bids_u_1 + bid_u_1;
                    mbarrier_wait(work_full_addr + (mm_work_stage) * 8, _phase_work_full_1);
                    unsigned int posted_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&posted_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&posted_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&posted_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&posted_1[(0) + 3]))
                        : "r"(work_slot_addr + mm_work_stage * 16));
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (mm_work_stage) * 8), "r"(0) : "memory");
                    mm_work_stage += 1;
                    if (mm_work_stage == 4) { mm_work_stage = 0; _phase_work_full_1 ^= 1; }
                    int w_valid_1 = (int)posted_1[0];
                    unsigned int w_pair_1 = posted_1[1];
                    next_block_idx_1 = w_pair_1 * 2 + (unsigned int)cta_rank;
                    int mm_valid = w_valid_1;
                    if (mm_valid == 0) {
                        break;
                    }
                    int has_block_1 = 0;
                    unsigned int m_blocks_g_1 = 0;
                    unsigned int ws_le_1 = ((next_block_idx_1 >= ws_incl * grid_n_1) ? ws_one : ws_zero);
                    unsigned int ws_g_1;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(ws_g_1) : "r"(ws_le_1));
                    unsigned int ws_le_hi_1 = ((next_block_idx_1 >= ws_incl_hi * grid_n_1) ? ws_one : ws_zero);
                    unsigned int _warp_redux_0_1;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_0_1) : "r"(ws_le_hi_1));
                    ws_g_1 = ws_g_1 + _warp_redux_0_1;
                    if (ws_g_1 < group_bound_1) {
                        unsigned int ws_lane_1 = ((ws_g_1 >= ws_32) ? ws_g_1 - ws_32 : ws_g_1);
                        unsigned int ws_src_b_1 = ((ws_g_1 >= ws_32) ? ee_blocks_hi : ee_blocks);
                        unsigned int ws_src_i_1 = ((ws_g_1 >= ws_32) ? ws_incl_hi : ws_incl);
                        unsigned int ws_blocks_1 = __shfl_sync(0xFFFFFFFF, ws_src_b_1, ws_lane_1);
                        current_group_idx_1 = ws_g_1;
                        unsigned int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, ws_src_i_1, ws_lane_1);
                        current_m_cumsum_1 = _shfl_sync_1 - ws_blocks_1;
                        m_blocks_g_1 = ws_blocks_1;
                        has_block_1 = 1;
                    }
                    if (has_block_1 == 0) {
                        break;
                    }
                    unsigned int group_m_1 = (unsigned int)masked_m[current_group_idx_1];
                    unsigned int block_idx_1 = next_block_idx_1 - current_m_cumsum_1 * grid_n_1;
                    unsigned int blocks_per_l2_group_1 = m_blocks_g_1 * 2;
                    unsigned int l2_group_1 = block_idx_1 / blocks_per_l2_group_1;
                    unsigned int first_n_block_1 = l2_group_1 * 2;
                    unsigned int in_l2_group_1 = block_idx_1 % blocks_per_l2_group_1;
                    unsigned int m_block_1 = in_l2_group_1 / 2;
                    unsigned int remaining_m_1 = group_m_1 - m_block_1 * 32;
                    unsigned int runtime_mma_n_1 = (((remaining_m_1 + 15) / 16 * 16) < (32) ? ((remaining_m_1 + 15) / 16 * 16) : (32));
                    mbarrier_wait_hint(epilogue_done_addr + (epi_stage) * 8, _phase_epilogue_done, 10000000);
                    #pragma unroll 4
                    for (int iter_k_1 = 0; iter_k_1 < num_k_blocks_1; iter_k_1++) {
                        mbarrier_wait_hint(scale_ready_addr + (tma_stage) * 8, _phase_scale_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        uint32_t _elected_2 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfw_addr) >> 4) + (tma_stage) * 2432)), _elected_2);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfw_addr) >> 4) + (tma_stage) * 2432 + 32)), _elected_2);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 2432)), _elected_2);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa1, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 2432)), _elected_2);
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x7FFF) + (tma_stage) * 2432;
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x7FFF) + (tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8f6f4_bs_k64_cta2_leader((tmem_accum + (epi_stage * 32)), a_desc + 0, b_desc + 0,
                                ((0x90880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)), (int)((tmem_tmem_sfb + 0)), (int)((tmem_tmem_sfa + 0)), ((init_flag) ? 0 : 1), _elected_2);
                            tcgen05_mma_mxf8f6f4_bs_k64_cta2_leader((tmem_accum + (epi_stage * 32)), a_desc + 4, b_desc + 4,
                                ((0xd0880020U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)), (int)((tmem_tmem_sfb + 0) | 0x80000000), (int)((tmem_tmem_sfa + 0) | 0x80000000), 1, _elected_2);
                        }
                        int _smem_desc_lo_5 = (((smem_b_addr + 16384) >> 4) & 0x7FFF) + (tma_stage) * 2432;
                        int _smem_desc_lo_6 = (((smem_a_addr + 2048) >> 4) & 0x7FFF) + (tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8f6f4_bs_k64_cta2_leader((tmem_accum + (epi_stage * 32)), a_desc + 0, b_desc + 0,
                                ((0x90880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)), (int)((tmem_tmem_sfb + 4 + 0)), (int)((tmem_tmem_sfa1 + 0)), 1, _elected_2);
                            tcgen05_mma_mxf8f6f4_bs_k64_cta2_leader((tmem_accum + (epi_stage * 32)), a_desc + 4, b_desc + 4,
                                ((0xd0880020U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)), (int)((tmem_tmem_sfb + 4 + 0) | 0x80000000), (int)((tmem_tmem_sfa1 + 0) | 0x80000000), 1, _elected_2);
                        }
                        elect_commit_cg2_multicast(tma_free_addr + (tma_stage) * 8, (uint16_t)(3));
                        tma_stage += 1;
                        if (tma_stage == 6) { tma_stage = 0; _phase_scale_ready ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (epi_stage) * 8, (uint16_t)(3));
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int transpose_stage = 0;
            unsigned int grid_n_2 = (unsigned int)32;
            unsigned int num_k_blocks_2 = (unsigned int)28;
            unsigned int num_bids_u_2 = (unsigned int)num_bids;
            unsigned int bid_u_2 = (unsigned int)bid;
            unsigned int current_group_idx_2 = 0;
            unsigned int current_m_cumsum_2 = 0;
            unsigned int group_bound_2 = (unsigned int)64;
            unsigned int max_m_blocks_2 = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles_2 = group_bound_2 * max_m_blocks_2 * grid_n_2;
            unsigned int max_scheduler_iters_2 = max_total_tiles_2 / 2 + 1;
            unsigned int sf_k64[5];
            unsigned int tp_work_stage = 0;
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int current_iter_2 = 0; current_iter_2 < max_scheduler_iters_2; current_iter_2++) {
                unsigned int next_block_idx_2 = (unsigned int)current_iter_2 * num_bids_u_2 + bid_u_2;
                mbarrier_wait(work_full_addr + (tp_work_stage) * 8, _phase_work_full_2);
                unsigned int posted_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&posted_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&posted_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&posted_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&posted_2[(0) + 3]))
                    : "r"(work_slot_addr + tp_work_stage * 16));
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (tp_work_stage) * 8), "r"(0) : "memory");
                tp_work_stage += 1;
                if (tp_work_stage == 4) { tp_work_stage = 0; _phase_work_full_2 ^= 1; }
                int w_valid_2 = (int)posted_2[0];
                unsigned int w_pair_2 = posted_2[1];
                next_block_idx_2 = w_pair_2 * 2 + (unsigned int)cta_rank;
                int tp_valid = w_valid_2;
                if (tp_valid == 0) {
                    break;
                }
                int has_block_2 = 0;
                unsigned int ws_le_2 = ((next_block_idx_2 >= ws_incl * grid_n_2) ? ws_one : ws_zero);
                unsigned int ws_g_2;
                asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(ws_g_2) : "r"(ws_le_2));
                unsigned int ws_le_hi_2 = ((next_block_idx_2 >= ws_incl_hi * grid_n_2) ? ws_one : ws_zero);
                unsigned int _warp_redux_0_2;
                asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_0_2) : "r"(ws_le_hi_2));
                ws_g_2 = ws_g_2 + _warp_redux_0_2;
                if (ws_g_2 < group_bound_2) {
                    unsigned int ws_lane_2 = ((ws_g_2 >= ws_32) ? ws_g_2 - ws_32 : ws_g_2);
                    unsigned int ws_src_b_2 = ((ws_g_2 >= ws_32) ? ee_blocks_hi : ee_blocks);
                    unsigned int ws_src_i_2 = ((ws_g_2 >= ws_32) ? ws_incl_hi : ws_incl);
                    unsigned int ws_blocks_2 = __shfl_sync(0xFFFFFFFF, ws_src_b_2, ws_lane_2);
                    current_group_idx_2 = ws_g_2;
                    unsigned int _shfl_sync_1_1 = __shfl_sync(0xFFFFFFFF, ws_src_i_2, ws_lane_2);
                    current_m_cumsum_2 = _shfl_sync_1_1 - ws_blocks_2;
                    has_block_2 = 1;
                }
                if (has_block_2 == 0) {
                    break;
                }
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < num_k_blocks_2; iter_k_2++) {
                    mbarrier_wait_hint(tma_full_addr + (transpose_stage) * 8, _phase_tma_full, 10000000);
                    if (iter_k_2 % 2 == 0) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_k64[0])) : "r"(smem_sfb_addr + transpose_stage * 38912 + (unsigned int)(lane * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_k64[1])) : "r"(smem_sfb_addr + transpose_stage * 38912 + (unsigned int)((32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_k64[2])) : "r"(smem_sfb_addr + transpose_stage * 38912 + (unsigned int)((64 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_k64[3])) : "r"(smem_sfb_addr + transpose_stage * 38912 + (unsigned int)((96 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sf_k64[4])) : "r"(smem_sfa_addr + transpose_stage * 38912 + (unsigned int)(lane * 4)));
                        __syncwarp();
                    }
                    unsigned int sf_sel_lo = ((iter_k_2 % 2 == 1) ? (unsigned int)8738 : (unsigned int)0);
                    unsigned int sf_sel_hi = sf_sel_lo + 4369;
                    unsigned int sf_zero = 0;
                    uint32_t _prmt_2;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_2) : "r"(sf_k64[0]), "r"(sf_zero), "r"(sf_sel_lo));
                    uint32_t _prmt_3;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_3) : "r"(sf_k64[1]), "r"(sf_zero), "r"(sf_sel_lo));
                    uint32_t _prmt_4;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_4) : "r"(sf_k64[2]), "r"(sf_zero), "r"(sf_sel_lo));
                    uint32_t _prmt_5;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_5) : "r"(sf_k64[3]), "r"(sf_zero), "r"(sf_sel_lo));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sfw_addr + transpose_stage * 38912 + (unsigned int)(lane * 16)), "r"(_prmt_2), "r"(_prmt_3), "r"(_prmt_4), "r"(_prmt_5));
                    uint32_t _prmt_6;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_6) : "r"(sf_k64[0]), "r"(sf_zero), "r"(sf_sel_hi));
                    uint32_t _prmt_7;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_7) : "r"(sf_k64[1]), "r"(sf_zero), "r"(sf_sel_hi));
                    uint32_t _prmt_8;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_8) : "r"(sf_k64[2]), "r"(sf_zero), "r"(sf_sel_hi));
                    uint32_t _prmt_9;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_9) : "r"(sf_k64[3]), "r"(sf_zero), "r"(sf_sel_hi));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sfw_addr + transpose_stage * 38912 + (unsigned int)(512 + lane * 16)), "r"(_prmt_6), "r"(_prmt_7), "r"(_prmt_8), "r"(_prmt_9));
                    uint32_t sf_act_lo;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sf_act_lo) : "r"(sf_k64[4]), "r"(sf_zero), "r"(sf_sel_lo));
                    uint32_t sf_act_hi;
                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sf_act_hi) : "r"(sf_k64[4]), "r"(sf_zero), "r"(sf_sel_hi));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sfa_addr + transpose_stage * 38912 + (unsigned int)(lane * 16)), "r"(sf_act_lo), "r"(sf_act_lo), "r"(sf_act_lo), "r"(sf_act_lo));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sfb_addr + transpose_stage * 38912 + (unsigned int)(lane * 16)), "r"(sf_act_hi), "r"(sf_act_hi), "r"(sf_act_hi), "r"(sf_act_hi));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_ready_addr + (transpose_stage) * 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(tma_free_addr + (transpose_stage) * 8);
                    transpose_stage += 1;
                    if (transpose_stage == 6) { transpose_stage = 0; _phase_tma_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int epi_stage_1 = 0;
            unsigned int grid_n_3 = (unsigned int)32;
            unsigned int num_bids_u_3 = (unsigned int)num_bids;
            unsigned int bid_u_3 = (unsigned int)bid;
            unsigned int current_group_idx_3 = 0;
            unsigned int current_m_cumsum_3 = 0;
            unsigned int group_bound_3 = (unsigned int)64;
            unsigned int max_m_blocks_3 = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles_3 = group_bound_3 * max_m_blocks_3 * grid_n_3;
            unsigned int max_scheduler_iters_3 = max_total_tiles_3 / 2 + 1;
            const int epi_warp = warp % 4;
            unsigned int ep_work_stage = 0;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int current_iter_3 = 0; current_iter_3 < max_scheduler_iters_3; current_iter_3++) {
                unsigned int next_block_idx_3 = (unsigned int)current_iter_3 * num_bids_u_3 + bid_u_3;
                mbarrier_wait(work_full_addr + (ep_work_stage) * 8, _phase_work_full_3);
                unsigned int posted_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&posted_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&posted_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&posted_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&posted_3[(0) + 3]))
                    : "r"(work_slot_addr + ep_work_stage * 16));
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (ep_work_stage) * 8), "r"(0) : "memory");
                ep_work_stage += 1;
                if (ep_work_stage == 4) { ep_work_stage = 0; _phase_work_full_3 ^= 1; }
                int w_valid_3 = (int)posted_3[0];
                unsigned int w_pair_3 = posted_3[1];
                next_block_idx_3 = w_pair_3 * 2 + (unsigned int)cta_rank;
                int ep_valid = w_valid_3;
                if (ep_valid == 0) {
                    break;
                }
                int has_block_3 = 0;
                unsigned int m_blocks_g_2 = 0;
                unsigned int ws_le_3 = ((next_block_idx_3 >= ws_incl * grid_n_3) ? ws_one : ws_zero);
                unsigned int ws_g_3;
                asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(ws_g_3) : "r"(ws_le_3));
                unsigned int ws_le_hi_3 = ((next_block_idx_3 >= ws_incl_hi * grid_n_3) ? ws_one : ws_zero);
                unsigned int _warp_redux_0_3;
                asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_0_3) : "r"(ws_le_hi_3));
                ws_g_3 = ws_g_3 + _warp_redux_0_3;
                if (ws_g_3 < group_bound_3) {
                    unsigned int ws_lane_3 = ((ws_g_3 >= ws_32) ? ws_g_3 - ws_32 : ws_g_3);
                    unsigned int ws_src_b_3 = ((ws_g_3 >= ws_32) ? ee_blocks_hi : ee_blocks);
                    unsigned int ws_src_i_3 = ((ws_g_3 >= ws_32) ? ws_incl_hi : ws_incl);
                    unsigned int ws_blocks_3 = __shfl_sync(0xFFFFFFFF, ws_src_b_3, ws_lane_3);
                    current_group_idx_3 = ws_g_3;
                    unsigned int _shfl_sync_1_2 = __shfl_sync(0xFFFFFFFF, ws_src_i_3, ws_lane_3);
                    current_m_cumsum_3 = _shfl_sync_1_2 - ws_blocks_3;
                    m_blocks_g_2 = ws_blocks_3;
                    has_block_3 = 1;
                }
                if (has_block_3 == 0) {
                    break;
                }
                unsigned int block_idx_2 = next_block_idx_3 - current_m_cumsum_3 * grid_n_3;
                unsigned int blocks_per_l2_group_2 = m_blocks_g_2 * 2;
                unsigned int l2_group_2 = block_idx_2 / blocks_per_l2_group_2;
                unsigned int first_n_block_2 = l2_group_2 * 2;
                unsigned int in_l2_group_2 = block_idx_2 % blocks_per_l2_group_2;
                unsigned int m_block_2 = in_l2_group_2 / 2;
                unsigned int n_block_1 = first_n_block_2 + in_l2_group_2 % 2;
                mbarrier_wait_hint(mainloop_done_addr + (epi_stage_1) * 8, _phase_mainloop_done, 10000000);
                unsigned int off_n_1 = n_block_1 * 128;
                unsigned int flat_row = current_group_idx_3 * shape_m + m_block_2 * 32;
                int tmem_base = taddr + (unsigned int)((int)epi_stage_1 * 32) + (unsigned int)(epi_warp * 32 << 16);
                #pragma unroll
                for (int strip = 0; strip < 2; strip++) {
                    int col_strip = strip * 16;
                    if (epi_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int stage_base = epi_staging_addr + 0 * 4096;
                    #pragma unroll
                    for (int atom = 0; atom < 2; atom++) {
                        int tmem_addr = tmem_base + col_strip + atom * 8;
                        float _tmem_load_0[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                            : "r"(tmem_addr));
                        float _tmem_load_1[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                            : "r"(tmem_addr + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                        int outer_atom = epi_warp / 2 * 2048;
                        int inner_atom = atom * 1024;
                        int lane_row = lane % 8;
                        int lane_col = epi_warp % 2 * 4 + lane / 8;
                        int smem_ptr = stage_base + outer_atom + inner_atom + lane_row * 128 + (lane_col ^ lane_row) * 16;
                        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(smem_ptr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_0)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_1)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_2)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_3))
                            : "memory");
                    }
                    if (strip == 1) {
                        if (elect_sync()) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((epilogue_done_addr + (epi_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            if (m_block_2 * 32 + (unsigned int)col_strip < shape_m) {
                                tma_store_2d((&C_tma), off_n_1, flat_row + (unsigned int)col_strip, stage_base);
                                tma_store_2d((&C_tma), off_n_1 + 64, flat_row + (unsigned int)col_strip, stage_base + 2048);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    __syncwarp();
                }
                epi_stage_1 += 1;
                if (epi_stage_1 == 2) { epi_stage_1 = 0; _phase_mainloop_done ^= 1; }
            }
            if (epi_warp == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
