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
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
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




__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}



__device__ __forceinline__ void tcgen05_commit_cg2_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

template <typename T> __device__ __forceinline__ T cake_floor_mod(T a, T b) {
    const T r = a % b;
    return (r != 0) & ((r ^ b) < 0) ? r + b : r;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_mega_moe_nvfp4_c223d36d635d46b021f0(const __grid_constant__ CUtensorMap W1, const __grid_constant__ CUtensorMap S1, const __grid_constant__ CUtensorMap W2, const __grid_constant__ CUtensorMap S2, const __grid_constant__ CUtensorMap pool_b, const __grid_constant__ CUtensorMap pool_sfb, const __grid_constant__ CUtensorMap fc1_out_st, const __grid_constant__ CUtensorMap fc1_out_ld, const __grid_constant__ CUtensorMap fc1_out_sfb, const __grid_constant__ CUtensorMap pool_b16, const __grid_constant__ CUtensorMap fc1_out_ld16, const __grid_constant__ CUtensorMap pool_sfb1, const __grid_constant__ CUtensorMap fc1_out_sfb1, const __grid_constant__ CUtensorMap pool_bh, const __grid_constant__ CUtensorMap fc1_out_ldh, uint8_t* __restrict__ pool_data, uint8_t* __restrict__ pool_sf, uint8_t* __restrict__ fc1_out_sf, int* __restrict__ pool_expert_base, int* __restrict__ expert_counts, float* __restrict__ fc1_alpha, float* __restrict__ fc2_alpha, float* __restrict__ norm_const, unsigned int* __restrict__ fc1_ready, unsigned int* __restrict__ fc1_done, unsigned int* __restrict__ sync_words, int* __restrict__ sorted_metadata_ready, int* __restrict__ push_table_ready, int* __restrict__ router_size_counter, int* __restrict__ router_histogram_done, int* __restrict__ source_base_ready, int num_ctas, int e_pad, int block_words, int size_counter_words, int push_ctas, uint8_t* __restrict__ dedup_rows, unsigned int* __restrict__ fc2_done, int dedup_cap, int mega_pg_world, int mega_pg_rank, const __grid_constant__ CakePeerPointerTable<unsigned int> mega_pg_flags, uint8_t* activation, const __grid_constant__ CakePeerPointerTable<uint8_t> activation_peers, uint8_t* activation_sf, const __grid_constant__ CakePeerPointerTable<uint8_t> activation_sf_peers, int* sizes, const __grid_constant__ CakePeerPointerTable<int> sizes_peers, int* sizes_ready, const __grid_constant__ CakePeerPointerTable<int> sizes_ready_peers, int* metadata_ready, const __grid_constant__ CakePeerPointerTable<int> metadata_ready_peers, long long* token_src_metadata, const __grid_constant__ CakePeerPointerTable<long long> token_src_metadata_peers, int* x_ready, const __grid_constant__ CakePeerPointerTable<int> x_ready_peers, float* fc1_topk_scores, const __grid_constant__ CakePeerPointerTable<float> fc1_topk_scores_peers, __nv_bfloat16* pre_reduced, const __grid_constant__ CakePeerPointerTable<__nv_bfloat16> pre_reduced_peers)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(pool_expert_base));
    __builtin_assume(__isGlobal(pool_expert_base));
    asm volatile("" : "+l"(expert_counts));
    __builtin_assume(__isGlobal(expert_counts));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define af_addr (mbar_base + 0)
    #define ae_addr (mbar_base + 48)
    #define cf_addr (mbar_base + 96)
    #define ce_addr (mbar_base + 112)
    #define pull_all_addr (mbar_base + 128)
    #define cfill_all_addr (mbar_base + 192)
    #define efill_bar_0_addr (mbar_base + 256)
    #define efill_bar_1_addr (mbar_base + 272)
    #define efill_bar_2_addr (mbar_base + 288)
    #define efill_bar_3_addr (mbar_base + 304)
    #define sfull_addr (mbar_base + 320)
    #define sempty_addr (mbar_base + 352)
    #define work_full_addr (mbar_base + 384)
    #define work_empty_addr (mbar_base + 408)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    unsigned int cluster_size;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cluster_size));

    // Prologue
    uint8_t* sm_epi = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int sm_epi_addr = smem + 1024;
    uint8_t* sm_f2 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int sm_f2_addr = smem + 1024;
    uint8_t* sm_fc1 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int sm_fc1_addr = smem + 37888;
    uint8_t* sm_pull = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int sm_pull_addr = smem + 46080;
    unsigned int* work_cells = reinterpret_cast<unsigned int*>(smem_raw + 78848);
    const int work_cells_addr = smem + 78848;
    uint8_t* sm_a = reinterpret_cast<uint8_t*>(smem_raw + 79872);
    const int sm_a_addr = smem + 79872;
    uint8_t* sm_b = reinterpret_cast<uint8_t*>(smem_raw + 178176);
    const int sm_b_addr = smem + 178176;
    uint8_t* sm_sfa = reinterpret_cast<uint8_t*>(smem_raw + 276480);
    const int sm_sfa_addr = smem + 276480;
    uint8_t* sm_sfb = reinterpret_cast<uint8_t*>(smem_raw + 288768);
    const int sm_sfb_addr = smem + 288768;

    // Mbarrier init (54 barriers at smem_raw[0..432))

    if (warp == 0) {
        // --- pipeline 'ap' ---
        // af: 6 barriers, init_count=4
        // ae: 6 barriers, init_count=2
        // --- pipeline 'cp' ---
        // cf: 2 barriers, init_count=1
        // ce: 2 barriers, init_count=256
        // pull_all: 8 barriers, init_count=1
        // cfill_all: 8 barriers, init_count=1
        // efill_bar_0: 2 barriers, init_count=1
        // efill_bar_1: 2 barriers, init_count=1
        // efill_bar_2: 2 barriers, init_count=1
        // efill_bar_3: 2 barriers, init_count=1
        // --- pipeline 'sp' ---
        // sfull: 4 barriers, init_count=128
        // sempty: 4 barriers, init_count=1
        // --- pipeline 'wq' ---
        // work_full: 3 barriers, init_count=1
        // work_empty: 3 barriers, init_count=960
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(16), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(14), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(6), "r"((uint32_t)(4)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 960;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(19), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(12), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(1)));
        if (lane < 22) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (576 columns, 576 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 432);
    if (warp == 0) {
        int _tmem_hold = smem + 432;
        asm volatile("tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(576) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int accum = taddr;
    const unsigned int tm_sfa = taddr + 512;
    const unsigned int tm_sfb = taddr + 528;
    const unsigned int allocation_padding = taddr + 560;

    // ---- Role: epi ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        { // epi_main
            unsigned int cst = 0;
            unsigned int sst = 0;
            int cta = cta_rank & 1;
            unsigned long long peer_base[8];
            peer_base[0] = 0ull;
            if (0 < mega_pg_world) {
                peer_base[0] = (unsigned long long)pre_reduced_peers.ptrs[0];
            }
            peer_base[1] = 0ull;
            if (1 < mega_pg_world) {
                peer_base[1] = (unsigned long long)pre_reduced_peers.ptrs[1];
            }
            peer_base[2] = 0ull;
            if (2 < mega_pg_world) {
                peer_base[2] = (unsigned long long)pre_reduced_peers.ptrs[2];
            }
            peer_base[3] = 0ull;
            if (3 < mega_pg_world) {
                peer_base[3] = (unsigned long long)pre_reduced_peers.ptrs[3];
            }
            peer_base[4] = 0ull;
            if (4 < mega_pg_world) {
                peer_base[4] = (unsigned long long)pre_reduced_peers.ptrs[4];
            }
            peer_base[5] = 0ull;
            if (5 < mega_pg_world) {
                peer_base[5] = (unsigned long long)pre_reduced_peers.ptrs[5];
            }
            peer_base[6] = 0ull;
            if (6 < mega_pg_world) {
                peer_base[6] = (unsigned long long)pre_reduced_peers.ptrs[6];
            }
            peer_base[7] = 0ull;
            if (7 < mega_pg_world) {
                peer_base[7] = (unsigned long long)pre_reduced_peers.ptrs[7];
            }
            {
                int _poll_v_0;
                do {
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<int*>(push_table_ready)) : "memory");
                } while (_poll_v_0 < static_cast<int>(1));
            }
            int epi_copy = metadata_ready[0] & 1;
            unsigned int wst = 0;
            int no_pend[1];
            int fpend[2];
            fpend[0] = -1;
            fpend[1] = 0;
            int nit = 0;
            int part = 0;
            part = part + (expert_counts[0] + 256 - 1 >> 8);
            part = part + (expert_counts[1] + 256 - 1 >> 8);
            part = part + (expert_counts[2] + 256 - 1 >> 8);
            part = part + (expert_counts[3] + 256 - 1 >> 8);
            part = part + (expert_counts[4] + 256 - 1 >> 8);
            part = part + (expert_counts[5] + 256 - 1 >> 8);
            part = part + (expert_counts[6] + 256 - 1 >> 8);
            part = part + (expert_counts[7] + 256 - 1 >> 8);
            part = part + (expert_counts[8] + 256 - 1 >> 8);
            part = part + (expert_counts[9] + 256 - 1 >> 8);
            part = part + (expert_counts[10] + 256 - 1 >> 8);
            part = part + (expert_counts[11] + 256 - 1 >> 8);
            int total = part * 26;
            int ticket = 0;
            int s1 = 0;
            int sb1 = 0;
            int d1 = 0;
            int f1 = 0;
            int b1 = 0;
            int c1 = expert_counts[0];
            int n1 = (c1 + 256 - 1 >> 8) * 12;
            int s2 = 0;
            int sb2 = 0;
            int d2 = 0;
            int f2 = 0;
            int b2 = 0;
            int c2 = expert_counts[0];
            int n2 = (c2 + 256 - 1 >> 8) * 14;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_cf = 0;
            unsigned int _phase_sempty = 1;
            #pragma unroll 1
            for (int it = 0; it < total + 1; it++) {
                mbarrier_wait(work_full_addr + (wst) * 8, _phase_work_full);
                ticket = (int)work_cells[wst * 4];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wst) * 8), "r"(1), "r"((uint32_t)((1 + (ticket >> 31)) * (2 - ((int)cluster_size >> 2)))) : "memory");
                wst += 1;
                if (wst == 3) { wst = 0; _phase_work_full ^= 1; }
                if (ticket == 2147483647) {
                    break;
                }
                int ph = ticket >> 30;
                int rid = ticket & 1073741823;
                int e = 0;
                int local = 0;
                int db = 0;
                int fb = 0;
                int bb = 0;
                int cnt = 0;
                int block = 0;
                int m_pair = 0;
                if (ph == 0) {
                    while (rid >= sb1 + n1) {
                        sb1 = sb1 + n1;
                        d1 = d1 + (c1 + 63 & -64);
                        f1 = f1 + (c1 + 127 & -128);
                        b1 = b1 + (c1 + 255 >> 8);
                        s1 = s1 + 1;
                        c1 = 0;
                        if (s1 < 12) {
                            c1 = expert_counts[s1];
                        }
                        n1 = (c1 + 256 - 1 >> 8) * 12;
                    }
                    e = s1;
                    local = rid - sb1;
                    db = d1;
                    fb = f1;
                    bb = b1;
                    cnt = c1;
                    block = local / 12;
                    m_pair = local - block * 12;
                } else {
                    while (rid >= sb2 + n2) {
                        sb2 = sb2 + n2;
                        d2 = d2 + (c2 + 63 & -64);
                        f2 = f2 + (c2 + 127 & -128);
                        b2 = b2 + (c2 + 255 >> 8);
                        s2 = s2 + 1;
                        c2 = 0;
                        if (s2 < 12) {
                            c2 = expert_counts[s2];
                        }
                        n2 = (c2 + 256 - 1 >> 8) * 14;
                    }
                    e = s2;
                    local = rid - sb2;
                    db = d2;
                    fb = f2;
                    bb = b2;
                    cnt = c2;
                    block = local / 14;
                    m_pair = local - block * 14;
                }
                int valid = ((cnt - block * 256) < (256) ? (cnt - block * 256) : (256));
                int nsub = 2 - ((int)cluster_size >> 2);
                #pragma unroll 1
                for (int sub = 0; sub < nsub; sub++) {
                    int m_one = m_pair * 2 + (cta_rank >> 1) + sub;
                    if (ph == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        mbarrier_wait(cf_addr + (cst) * 8, _phase_cf);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int acc_col = cst * 256;
                        #pragma unroll 1
                        for (int s0 = 0; s0 < valid; s0 += 64) {
                            float alpha = fc1_alpha[e];
                            float norm = norm_const[e];
                            int lane_1 = lane;
                            int a = lane_1 >> 4;
                            int ee = lane_1 >> 2 & 3;
                            int q = lane_1 & 3;
                            float scores[2];
                            scores[0] = fc1_topk_scores[(db + block * 256 + s0 + 16 * (2 * a) + 8 * (q >> 1) + 2 * ee + (q & 1) >> 5 << 6) + (epi_copy << 5) + (db + block * 256 + s0 + 16 * (2 * a) + 8 * (q >> 1) + 2 * ee + (q & 1) & 31)];
                            scores[1] = fc1_topk_scores[(db + block * 256 + s0 + 16 * (2 * a + 1) + 8 * (q >> 1) + 2 * ee + (q & 1) >> 5 << 6) + (epi_copy << 5) + (db + block * 256 + s0 + 16 * (2 * a + 1) + 8 * (q >> 1) + 2 * ee + (q & 1) & 31)];
                            unsigned int sc0 = taddr + (unsigned int)(warp * 32 << 16) + acc_col + (unsigned int)s0;
                            float gv[32];
                            float uv[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&gv[0])), "=r"(*reinterpret_cast<uint32_t*>(&gv[1])), "=r"(*reinterpret_cast<uint32_t*>(&gv[2])), "=r"(*reinterpret_cast<uint32_t*>(&gv[3])), "=r"(*reinterpret_cast<uint32_t*>(&gv[4])), "=r"(*reinterpret_cast<uint32_t*>(&gv[5])), "=r"(*reinterpret_cast<uint32_t*>(&gv[6])), "=r"(*reinterpret_cast<uint32_t*>(&gv[7])), "=r"(*reinterpret_cast<uint32_t*>(&gv[8])), "=r"(*reinterpret_cast<uint32_t*>(&gv[9])), "=r"(*reinterpret_cast<uint32_t*>(&gv[10])), "=r"(*reinterpret_cast<uint32_t*>(&gv[11])), "=r"(*reinterpret_cast<uint32_t*>(&gv[12])), "=r"(*reinterpret_cast<uint32_t*>(&gv[13])), "=r"(*reinterpret_cast<uint32_t*>(&gv[14])), "=r"(*reinterpret_cast<uint32_t*>(&gv[15])), "=r"(*reinterpret_cast<uint32_t*>(&gv[16])), "=r"(*reinterpret_cast<uint32_t*>(&gv[17])), "=r"(*reinterpret_cast<uint32_t*>(&gv[18])), "=r"(*reinterpret_cast<uint32_t*>(&gv[19])), "=r"(*reinterpret_cast<uint32_t*>(&gv[20])), "=r"(*reinterpret_cast<uint32_t*>(&gv[21])), "=r"(*reinterpret_cast<uint32_t*>(&gv[22])), "=r"(*reinterpret_cast<uint32_t*>(&gv[23])), "=r"(*reinterpret_cast<uint32_t*>(&gv[24])), "=r"(*reinterpret_cast<uint32_t*>(&gv[25])), "=r"(*reinterpret_cast<uint32_t*>(&gv[26])), "=r"(*reinterpret_cast<uint32_t*>(&gv[27])), "=r"(*reinterpret_cast<uint32_t*>(&gv[28])), "=r"(*reinterpret_cast<uint32_t*>(&gv[29])), "=r"(*reinterpret_cast<uint32_t*>(&gv[30])), "=r"(*reinterpret_cast<uint32_t*>(&gv[31]))
                                : "r"(sc0 + 0));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&uv[0])), "=r"(*reinterpret_cast<uint32_t*>(&uv[1])), "=r"(*reinterpret_cast<uint32_t*>(&uv[2])), "=r"(*reinterpret_cast<uint32_t*>(&uv[3])), "=r"(*reinterpret_cast<uint32_t*>(&uv[4])), "=r"(*reinterpret_cast<uint32_t*>(&uv[5])), "=r"(*reinterpret_cast<uint32_t*>(&uv[6])), "=r"(*reinterpret_cast<uint32_t*>(&uv[7])), "=r"(*reinterpret_cast<uint32_t*>(&uv[8])), "=r"(*reinterpret_cast<uint32_t*>(&uv[9])), "=r"(*reinterpret_cast<uint32_t*>(&uv[10])), "=r"(*reinterpret_cast<uint32_t*>(&uv[11])), "=r"(*reinterpret_cast<uint32_t*>(&uv[12])), "=r"(*reinterpret_cast<uint32_t*>(&uv[13])), "=r"(*reinterpret_cast<uint32_t*>(&uv[14])), "=r"(*reinterpret_cast<uint32_t*>(&uv[15])), "=r"(*reinterpret_cast<uint32_t*>(&uv[16])), "=r"(*reinterpret_cast<uint32_t*>(&uv[17])), "=r"(*reinterpret_cast<uint32_t*>(&uv[18])), "=r"(*reinterpret_cast<uint32_t*>(&uv[19])), "=r"(*reinterpret_cast<uint32_t*>(&uv[20])), "=r"(*reinterpret_cast<uint32_t*>(&uv[21])), "=r"(*reinterpret_cast<uint32_t*>(&uv[22])), "=r"(*reinterpret_cast<uint32_t*>(&uv[23])), "=r"(*reinterpret_cast<uint32_t*>(&uv[24])), "=r"(*reinterpret_cast<uint32_t*>(&uv[25])), "=r"(*reinterpret_cast<uint32_t*>(&uv[26])), "=r"(*reinterpret_cast<uint32_t*>(&uv[27])), "=r"(*reinterpret_cast<uint32_t*>(&uv[28])), "=r"(*reinterpret_cast<uint32_t*>(&uv[29])), "=r"(*reinterpret_cast<uint32_t*>(&uv[30])), "=r"(*reinterpret_cast<uint32_t*>(&uv[31]))
                                : "r"(sc0 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float2 alpha2 = make_float2(alpha, alpha);
                            float2 nl2 = make_float2(-1.442695f, -1.442695f);
                            float2 one2 = make_float2(1.0f, 1.0f);
                            float act[32];
                            float2 _f32x2_mul_a_1 = make_float2(gv[0], gv[1]);
                            float2 ga_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_2 = make_float2(uv[0], uv[1]);
                            float2 ua_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&alpha2));
                            float g0_0;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_0) : "f"(ga_0.x), "f"(10.0f));
                            float g1_0;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_0) : "f"(ga_0.y), "f"(10.0f));
                            float u0_0;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_0) : "f"(ua_0.x), "f"(10.0f));
                            float u1_0;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_0) : "f"(ua_0.y), "f"(10.0f));
                            float2 g2_0 = make_float2(g0_0, g1_0);
                            float2 t2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_3 = approx_exp2(t2_0.x);
                            float _exp2_4 = approx_exp2(t2_0.y);
                            float2 ex2_0 = add_f32x2_noftz(make_float2(_exp2_3, _exp2_4), one2);
                            float _rcp_5 = approx_rcp(ex2_0.x);
                            float _rcp_6 = approx_rcp(ex2_0.y);
                            float2 sig2_0 = make_float2(_rcp_5, _rcp_6);
                            float2 _f32x2_mul_a_7 = make_float2(u0_0, u1_0);
                            float2 ug2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&g2_0));
                            float2 act2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_0) : "l"(*(const unsigned long long*)&ug2_0), "l"(*(const unsigned long long*)&sig2_0));
                            act[0] = act2_0.x;
                            act[1] = act2_0.y;
                            float2 _f32x2_mul_a_8 = make_float2(gv[2], gv[3]);
                            float2 ga_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_9 = make_float2(uv[2], uv[3]);
                            float2 ua_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&alpha2));
                            float g0_1;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_1) : "f"(ga_1.x), "f"(10.0f));
                            float g1_1;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_1) : "f"(ga_1.y), "f"(10.0f));
                            float u0_1;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_1) : "f"(ua_1.x), "f"(10.0f));
                            float u1_1;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_1) : "f"(ua_1.y), "f"(10.0f));
                            float2 g2_1 = make_float2(g0_1, g1_1);
                            float2 t2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_10 = approx_exp2(t2_1.x);
                            float _exp2_11 = approx_exp2(t2_1.y);
                            float2 ex2_1 = add_f32x2_noftz(make_float2(_exp2_10, _exp2_11), one2);
                            float _rcp_12 = approx_rcp(ex2_1.x);
                            float _rcp_13 = approx_rcp(ex2_1.y);
                            float2 sig2_1 = make_float2(_rcp_12, _rcp_13);
                            float2 _f32x2_mul_a_14 = make_float2(u0_1, u1_1);
                            float2 ug2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&g2_1));
                            float2 act2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_1) : "l"(*(const unsigned long long*)&ug2_1), "l"(*(const unsigned long long*)&sig2_1));
                            act[2] = act2_1.x;
                            act[3] = act2_1.y;
                            float2 _f32x2_mul_a_15 = make_float2(gv[4], gv[5]);
                            float2 ga_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_16 = make_float2(uv[4], uv[5]);
                            float2 ua_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&alpha2));
                            float g0_2;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_2) : "f"(ga_2.x), "f"(10.0f));
                            float g1_2;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_2) : "f"(ga_2.y), "f"(10.0f));
                            float u0_2;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_2) : "f"(ua_2.x), "f"(10.0f));
                            float u1_2;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_2) : "f"(ua_2.y), "f"(10.0f));
                            float2 g2_2 = make_float2(g0_2, g1_2);
                            float2 t2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_17 = approx_exp2(t2_2.x);
                            float _exp2_18 = approx_exp2(t2_2.y);
                            float2 ex2_2 = add_f32x2_noftz(make_float2(_exp2_17, _exp2_18), one2);
                            float _rcp_19 = approx_rcp(ex2_2.x);
                            float _rcp_20 = approx_rcp(ex2_2.y);
                            float2 sig2_2 = make_float2(_rcp_19, _rcp_20);
                            float2 _f32x2_mul_a_21 = make_float2(u0_2, u1_2);
                            float2 ug2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&g2_2));
                            float2 act2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_2) : "l"(*(const unsigned long long*)&ug2_2), "l"(*(const unsigned long long*)&sig2_2));
                            act[4] = act2_2.x;
                            act[5] = act2_2.y;
                            float2 _f32x2_mul_a_22 = make_float2(gv[6], gv[7]);
                            float2 ga_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_23 = make_float2(uv[6], uv[7]);
                            float2 ua_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&alpha2));
                            float g0_3;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_3) : "f"(ga_3.x), "f"(10.0f));
                            float g1_3;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_3) : "f"(ga_3.y), "f"(10.0f));
                            float u0_3;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_3) : "f"(ua_3.x), "f"(10.0f));
                            float u1_3;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_3) : "f"(ua_3.y), "f"(10.0f));
                            float2 g2_3 = make_float2(g0_3, g1_3);
                            float2 t2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_24 = approx_exp2(t2_3.x);
                            float _exp2_25 = approx_exp2(t2_3.y);
                            float2 ex2_3 = add_f32x2_noftz(make_float2(_exp2_24, _exp2_25), one2);
                            float _rcp_26 = approx_rcp(ex2_3.x);
                            float _rcp_27 = approx_rcp(ex2_3.y);
                            float2 sig2_3 = make_float2(_rcp_26, _rcp_27);
                            float2 _f32x2_mul_a_28 = make_float2(u0_3, u1_3);
                            float2 ug2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&g2_3));
                            float2 act2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_3) : "l"(*(const unsigned long long*)&ug2_3), "l"(*(const unsigned long long*)&sig2_3));
                            act[6] = act2_3.x;
                            act[7] = act2_3.y;
                            float2 _f32x2_mul_a_29 = make_float2(gv[8], gv[9]);
                            float2 ga_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_30 = make_float2(uv[8], uv[9]);
                            float2 ua_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&alpha2));
                            float g0_4;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_4) : "f"(ga_4.x), "f"(10.0f));
                            float g1_4;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_4) : "f"(ga_4.y), "f"(10.0f));
                            float u0_4;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_4) : "f"(ua_4.x), "f"(10.0f));
                            float u1_4;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_4) : "f"(ua_4.y), "f"(10.0f));
                            float2 g2_4 = make_float2(g0_4, g1_4);
                            float2 t2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_31 = approx_exp2(t2_4.x);
                            float _exp2_32 = approx_exp2(t2_4.y);
                            float2 ex2_4 = add_f32x2_noftz(make_float2(_exp2_31, _exp2_32), one2);
                            float _rcp_33 = approx_rcp(ex2_4.x);
                            float _rcp_34 = approx_rcp(ex2_4.y);
                            float2 sig2_4 = make_float2(_rcp_33, _rcp_34);
                            float2 _f32x2_mul_a_35 = make_float2(u0_4, u1_4);
                            float2 ug2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&g2_4));
                            float2 act2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_4) : "l"(*(const unsigned long long*)&ug2_4), "l"(*(const unsigned long long*)&sig2_4));
                            act[8] = act2_4.x;
                            act[9] = act2_4.y;
                            float2 _f32x2_mul_a_36 = make_float2(gv[10], gv[11]);
                            float2 ga_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_37 = make_float2(uv[10], uv[11]);
                            float2 ua_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&alpha2));
                            float g0_5;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_5) : "f"(ga_5.x), "f"(10.0f));
                            float g1_5;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_5) : "f"(ga_5.y), "f"(10.0f));
                            float u0_5;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_5) : "f"(ua_5.x), "f"(10.0f));
                            float u1_5;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_5) : "f"(ua_5.y), "f"(10.0f));
                            float2 g2_5 = make_float2(g0_5, g1_5);
                            float2 t2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_38 = approx_exp2(t2_5.x);
                            float _exp2_39 = approx_exp2(t2_5.y);
                            float2 ex2_5 = add_f32x2_noftz(make_float2(_exp2_38, _exp2_39), one2);
                            float _rcp_40 = approx_rcp(ex2_5.x);
                            float _rcp_41 = approx_rcp(ex2_5.y);
                            float2 sig2_5 = make_float2(_rcp_40, _rcp_41);
                            float2 _f32x2_mul_a_42 = make_float2(u0_5, u1_5);
                            float2 ug2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&g2_5));
                            float2 act2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_5) : "l"(*(const unsigned long long*)&ug2_5), "l"(*(const unsigned long long*)&sig2_5));
                            act[10] = act2_5.x;
                            act[11] = act2_5.y;
                            float2 _f32x2_mul_a_43 = make_float2(gv[12], gv[13]);
                            float2 ga_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_44 = make_float2(uv[12], uv[13]);
                            float2 ua_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&alpha2));
                            float g0_6;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_6) : "f"(ga_6.x), "f"(10.0f));
                            float g1_6;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_6) : "f"(ga_6.y), "f"(10.0f));
                            float u0_6;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_6) : "f"(ua_6.x), "f"(10.0f));
                            float u1_6;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_6) : "f"(ua_6.y), "f"(10.0f));
                            float2 g2_6 = make_float2(g0_6, g1_6);
                            float2 t2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_45 = approx_exp2(t2_6.x);
                            float _exp2_46 = approx_exp2(t2_6.y);
                            float2 ex2_6 = add_f32x2_noftz(make_float2(_exp2_45, _exp2_46), one2);
                            float _rcp_47 = approx_rcp(ex2_6.x);
                            float _rcp_48 = approx_rcp(ex2_6.y);
                            float2 sig2_6 = make_float2(_rcp_47, _rcp_48);
                            float2 _f32x2_mul_a_49 = make_float2(u0_6, u1_6);
                            float2 ug2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&g2_6));
                            float2 act2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_6) : "l"(*(const unsigned long long*)&ug2_6), "l"(*(const unsigned long long*)&sig2_6));
                            act[12] = act2_6.x;
                            act[13] = act2_6.y;
                            float2 _f32x2_mul_a_50 = make_float2(gv[14], gv[15]);
                            float2 ga_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_51 = make_float2(uv[14], uv[15]);
                            float2 ua_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&alpha2));
                            float g0_7;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_7) : "f"(ga_7.x), "f"(10.0f));
                            float g1_7;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_7) : "f"(ga_7.y), "f"(10.0f));
                            float u0_7;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_7) : "f"(ua_7.x), "f"(10.0f));
                            float u1_7;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_7) : "f"(ua_7.y), "f"(10.0f));
                            float2 g2_7 = make_float2(g0_7, g1_7);
                            float2 t2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_52 = approx_exp2(t2_7.x);
                            float _exp2_53 = approx_exp2(t2_7.y);
                            float2 ex2_7 = add_f32x2_noftz(make_float2(_exp2_52, _exp2_53), one2);
                            float _rcp_54 = approx_rcp(ex2_7.x);
                            float _rcp_55 = approx_rcp(ex2_7.y);
                            float2 sig2_7 = make_float2(_rcp_54, _rcp_55);
                            float2 _f32x2_mul_a_56 = make_float2(u0_7, u1_7);
                            float2 ug2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&g2_7));
                            float2 act2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_7) : "l"(*(const unsigned long long*)&ug2_7), "l"(*(const unsigned long long*)&sig2_7));
                            act[14] = act2_7.x;
                            act[15] = act2_7.y;
                            float2 _f32x2_mul_a_57 = make_float2(gv[16], gv[17]);
                            float2 ga_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_58 = make_float2(uv[16], uv[17]);
                            float2 ua_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&alpha2));
                            float g0_8;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_8) : "f"(ga_8.x), "f"(10.0f));
                            float g1_8;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_8) : "f"(ga_8.y), "f"(10.0f));
                            float u0_8;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_8) : "f"(ua_8.x), "f"(10.0f));
                            float u1_8;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_8) : "f"(ua_8.y), "f"(10.0f));
                            float2 g2_8 = make_float2(g0_8, g1_8);
                            float2 t2_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_8) : "l"(*(const unsigned long long*)&g2_8), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_59 = approx_exp2(t2_8.x);
                            float _exp2_60 = approx_exp2(t2_8.y);
                            float2 ex2_8 = add_f32x2_noftz(make_float2(_exp2_59, _exp2_60), one2);
                            float _rcp_61 = approx_rcp(ex2_8.x);
                            float _rcp_62 = approx_rcp(ex2_8.y);
                            float2 sig2_8 = make_float2(_rcp_61, _rcp_62);
                            float2 _f32x2_mul_a_63 = make_float2(u0_8, u1_8);
                            float2 ug2_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&g2_8));
                            float2 act2_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_8) : "l"(*(const unsigned long long*)&ug2_8), "l"(*(const unsigned long long*)&sig2_8));
                            act[16] = act2_8.x;
                            act[17] = act2_8.y;
                            float2 _f32x2_mul_a_64 = make_float2(gv[18], gv[19]);
                            float2 ga_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_65 = make_float2(uv[18], uv[19]);
                            float2 ua_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&alpha2));
                            float g0_9;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_9) : "f"(ga_9.x), "f"(10.0f));
                            float g1_9;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_9) : "f"(ga_9.y), "f"(10.0f));
                            float u0_9;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_9) : "f"(ua_9.x), "f"(10.0f));
                            float u1_9;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_9) : "f"(ua_9.y), "f"(10.0f));
                            float2 g2_9 = make_float2(g0_9, g1_9);
                            float2 t2_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_9) : "l"(*(const unsigned long long*)&g2_9), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_66 = approx_exp2(t2_9.x);
                            float _exp2_67 = approx_exp2(t2_9.y);
                            float2 ex2_9 = add_f32x2_noftz(make_float2(_exp2_66, _exp2_67), one2);
                            float _rcp_68 = approx_rcp(ex2_9.x);
                            float _rcp_69 = approx_rcp(ex2_9.y);
                            float2 sig2_9 = make_float2(_rcp_68, _rcp_69);
                            float2 _f32x2_mul_a_70 = make_float2(u0_9, u1_9);
                            float2 ug2_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&g2_9));
                            float2 act2_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_9) : "l"(*(const unsigned long long*)&ug2_9), "l"(*(const unsigned long long*)&sig2_9));
                            act[18] = act2_9.x;
                            act[19] = act2_9.y;
                            float2 _f32x2_mul_a_71 = make_float2(gv[20], gv[21]);
                            float2 ga_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_72 = make_float2(uv[20], uv[21]);
                            float2 ua_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&alpha2));
                            float g0_10;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_10) : "f"(ga_10.x), "f"(10.0f));
                            float g1_10;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_10) : "f"(ga_10.y), "f"(10.0f));
                            float u0_10;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_10) : "f"(ua_10.x), "f"(10.0f));
                            float u1_10;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_10) : "f"(ua_10.y), "f"(10.0f));
                            float2 g2_10 = make_float2(g0_10, g1_10);
                            float2 t2_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_10) : "l"(*(const unsigned long long*)&g2_10), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_73 = approx_exp2(t2_10.x);
                            float _exp2_74 = approx_exp2(t2_10.y);
                            float2 ex2_10 = add_f32x2_noftz(make_float2(_exp2_73, _exp2_74), one2);
                            float _rcp_75 = approx_rcp(ex2_10.x);
                            float _rcp_76 = approx_rcp(ex2_10.y);
                            float2 sig2_10 = make_float2(_rcp_75, _rcp_76);
                            float2 _f32x2_mul_a_77 = make_float2(u0_10, u1_10);
                            float2 ug2_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&g2_10));
                            float2 act2_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_10) : "l"(*(const unsigned long long*)&ug2_10), "l"(*(const unsigned long long*)&sig2_10));
                            act[20] = act2_10.x;
                            act[21] = act2_10.y;
                            float2 _f32x2_mul_a_78 = make_float2(gv[22], gv[23]);
                            float2 ga_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_79 = make_float2(uv[22], uv[23]);
                            float2 ua_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&alpha2));
                            float g0_11;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_11) : "f"(ga_11.x), "f"(10.0f));
                            float g1_11;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_11) : "f"(ga_11.y), "f"(10.0f));
                            float u0_11;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_11) : "f"(ua_11.x), "f"(10.0f));
                            float u1_11;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_11) : "f"(ua_11.y), "f"(10.0f));
                            float2 g2_11 = make_float2(g0_11, g1_11);
                            float2 t2_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_11) : "l"(*(const unsigned long long*)&g2_11), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_80 = approx_exp2(t2_11.x);
                            float _exp2_81 = approx_exp2(t2_11.y);
                            float2 ex2_11 = add_f32x2_noftz(make_float2(_exp2_80, _exp2_81), one2);
                            float _rcp_82 = approx_rcp(ex2_11.x);
                            float _rcp_83 = approx_rcp(ex2_11.y);
                            float2 sig2_11 = make_float2(_rcp_82, _rcp_83);
                            float2 _f32x2_mul_a_84 = make_float2(u0_11, u1_11);
                            float2 ug2_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&g2_11));
                            float2 act2_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_11) : "l"(*(const unsigned long long*)&ug2_11), "l"(*(const unsigned long long*)&sig2_11));
                            act[22] = act2_11.x;
                            act[23] = act2_11.y;
                            float2 _f32x2_mul_a_85 = make_float2(gv[24], gv[25]);
                            float2 ga_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_86 = make_float2(uv[24], uv[25]);
                            float2 ua_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&alpha2));
                            float g0_12;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_12) : "f"(ga_12.x), "f"(10.0f));
                            float g1_12;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_12) : "f"(ga_12.y), "f"(10.0f));
                            float u0_12;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_12) : "f"(ua_12.x), "f"(10.0f));
                            float u1_12;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_12) : "f"(ua_12.y), "f"(10.0f));
                            float2 g2_12 = make_float2(g0_12, g1_12);
                            float2 t2_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_12) : "l"(*(const unsigned long long*)&g2_12), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_87 = approx_exp2(t2_12.x);
                            float _exp2_88 = approx_exp2(t2_12.y);
                            float2 ex2_12 = add_f32x2_noftz(make_float2(_exp2_87, _exp2_88), one2);
                            float _rcp_89 = approx_rcp(ex2_12.x);
                            float _rcp_90 = approx_rcp(ex2_12.y);
                            float2 sig2_12 = make_float2(_rcp_89, _rcp_90);
                            float2 _f32x2_mul_a_91 = make_float2(u0_12, u1_12);
                            float2 ug2_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&g2_12));
                            float2 act2_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_12) : "l"(*(const unsigned long long*)&ug2_12), "l"(*(const unsigned long long*)&sig2_12));
                            act[24] = act2_12.x;
                            act[25] = act2_12.y;
                            float2 _f32x2_mul_a_92 = make_float2(gv[26], gv[27]);
                            float2 ga_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_93 = make_float2(uv[26], uv[27]);
                            float2 ua_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&alpha2));
                            float g0_13;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_13) : "f"(ga_13.x), "f"(10.0f));
                            float g1_13;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_13) : "f"(ga_13.y), "f"(10.0f));
                            float u0_13;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_13) : "f"(ua_13.x), "f"(10.0f));
                            float u1_13;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_13) : "f"(ua_13.y), "f"(10.0f));
                            float2 g2_13 = make_float2(g0_13, g1_13);
                            float2 t2_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_13) : "l"(*(const unsigned long long*)&g2_13), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_94 = approx_exp2(t2_13.x);
                            float _exp2_95 = approx_exp2(t2_13.y);
                            float2 ex2_13 = add_f32x2_noftz(make_float2(_exp2_94, _exp2_95), one2);
                            float _rcp_96 = approx_rcp(ex2_13.x);
                            float _rcp_97 = approx_rcp(ex2_13.y);
                            float2 sig2_13 = make_float2(_rcp_96, _rcp_97);
                            float2 _f32x2_mul_a_98 = make_float2(u0_13, u1_13);
                            float2 ug2_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&g2_13));
                            float2 act2_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_13) : "l"(*(const unsigned long long*)&ug2_13), "l"(*(const unsigned long long*)&sig2_13));
                            act[26] = act2_13.x;
                            act[27] = act2_13.y;
                            float2 _f32x2_mul_a_99 = make_float2(gv[28], gv[29]);
                            float2 ga_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_99), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_100 = make_float2(uv[28], uv[29]);
                            float2 ua_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&alpha2));
                            float g0_14;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_14) : "f"(ga_14.x), "f"(10.0f));
                            float g1_14;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_14) : "f"(ga_14.y), "f"(10.0f));
                            float u0_14;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_14) : "f"(ua_14.x), "f"(10.0f));
                            float u1_14;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_14) : "f"(ua_14.y), "f"(10.0f));
                            float2 g2_14 = make_float2(g0_14, g1_14);
                            float2 t2_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_14) : "l"(*(const unsigned long long*)&g2_14), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_101 = approx_exp2(t2_14.x);
                            float _exp2_102 = approx_exp2(t2_14.y);
                            float2 ex2_14 = add_f32x2_noftz(make_float2(_exp2_101, _exp2_102), one2);
                            float _rcp_103 = approx_rcp(ex2_14.x);
                            float _rcp_104 = approx_rcp(ex2_14.y);
                            float2 sig2_14 = make_float2(_rcp_103, _rcp_104);
                            float2 _f32x2_mul_a_105 = make_float2(u0_14, u1_14);
                            float2 ug2_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&g2_14));
                            float2 act2_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_14) : "l"(*(const unsigned long long*)&ug2_14), "l"(*(const unsigned long long*)&sig2_14));
                            act[28] = act2_14.x;
                            act[29] = act2_14.y;
                            float2 _f32x2_mul_a_106 = make_float2(gv[30], gv[31]);
                            float2 ga_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ga_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&alpha2));
                            float2 _f32x2_mul_a_107 = make_float2(uv[30], uv[31]);
                            float2 ua_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ua_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&alpha2));
                            float g0_15;
                            asm("min.f32 %0, %1, %2;" : "=f"(g0_15) : "f"(ga_15.x), "f"(10.0f));
                            float g1_15;
                            asm("min.f32 %0, %1, %2;" : "=f"(g1_15) : "f"(ga_15.y), "f"(10.0f));
                            float u0_15;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u0_15) : "f"(ua_15.x), "f"(10.0f));
                            float u1_15;
                            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(u1_15) : "f"(ua_15.y), "f"(10.0f));
                            float2 g2_15 = make_float2(g0_15, g1_15);
                            float2 t2_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t2_15) : "l"(*(const unsigned long long*)&g2_15), "l"(*(const unsigned long long*)&nl2));
                            float _exp2_108 = approx_exp2(t2_15.x);
                            float _exp2_109 = approx_exp2(t2_15.y);
                            float2 ex2_15 = add_f32x2_noftz(make_float2(_exp2_108, _exp2_109), one2);
                            float _rcp_110 = approx_rcp(ex2_15.x);
                            float _rcp_111 = approx_rcp(ex2_15.y);
                            float2 sig2_15 = make_float2(_rcp_110, _rcp_111);
                            float2 _f32x2_mul_a_112 = make_float2(u0_15, u1_15);
                            float2 ug2_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&ug2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&g2_15));
                            float2 act2_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&act2_15) : "l"(*(const unsigned long long*)&ug2_15), "l"(*(const unsigned long long*)&sig2_15));
                            act[30] = act2_15.x;
                            act[31] = act2_15.y;
                            int w1 = lane_1 >> 2 & 1;
                            float st1[32];
                            st1[0] = act[0];
                            if (w1 != 0) {
                                st1[0] = act[16];
                            }
                            st1[1] = act[2];
                            if (w1 != 0) {
                                st1[1] = act[18];
                            }
                            st1[2] = act[1];
                            if (w1 != 0) {
                                st1[2] = act[17];
                            }
                            st1[3] = act[3];
                            if (w1 != 0) {
                                st1[3] = act[19];
                            }
                            st1[4] = act[4];
                            if (w1 != 0) {
                                st1[4] = act[20];
                            }
                            st1[5] = act[6];
                            if (w1 != 0) {
                                st1[5] = act[22];
                            }
                            st1[6] = act[5];
                            if (w1 != 0) {
                                st1[6] = act[21];
                            }
                            st1[7] = act[7];
                            if (w1 != 0) {
                                st1[7] = act[23];
                            }
                            st1[8] = act[8];
                            if (w1 != 0) {
                                st1[8] = act[24];
                            }
                            st1[9] = act[10];
                            if (w1 != 0) {
                                st1[9] = act[26];
                            }
                            st1[10] = act[9];
                            if (w1 != 0) {
                                st1[10] = act[25];
                            }
                            st1[11] = act[11];
                            if (w1 != 0) {
                                st1[11] = act[27];
                            }
                            st1[12] = act[12];
                            if (w1 != 0) {
                                st1[12] = act[28];
                            }
                            st1[13] = act[14];
                            if (w1 != 0) {
                                st1[13] = act[30];
                            }
                            st1[14] = act[13];
                            if (w1 != 0) {
                                st1[14] = act[29];
                            }
                            st1[15] = act[15];
                            if (w1 != 0) {
                                st1[15] = act[31];
                            }
                            st1[16] = act[16];
                            if (w1 != 0) {
                                st1[16] = act[0];
                            }
                            st1[17] = act[18];
                            if (w1 != 0) {
                                st1[17] = act[2];
                            }
                            st1[18] = act[17];
                            if (w1 != 0) {
                                st1[18] = act[1];
                            }
                            st1[19] = act[19];
                            if (w1 != 0) {
                                st1[19] = act[3];
                            }
                            st1[20] = act[20];
                            if (w1 != 0) {
                                st1[20] = act[4];
                            }
                            st1[21] = act[22];
                            if (w1 != 0) {
                                st1[21] = act[6];
                            }
                            st1[22] = act[21];
                            if (w1 != 0) {
                                st1[22] = act[5];
                            }
                            st1[23] = act[23];
                            if (w1 != 0) {
                                st1[23] = act[7];
                            }
                            st1[24] = act[24];
                            if (w1 != 0) {
                                st1[24] = act[8];
                            }
                            st1[25] = act[26];
                            if (w1 != 0) {
                                st1[25] = act[10];
                            }
                            st1[26] = act[25];
                            if (w1 != 0) {
                                st1[26] = act[9];
                            }
                            st1[27] = act[27];
                            if (w1 != 0) {
                                st1[27] = act[11];
                            }
                            st1[28] = act[28];
                            if (w1 != 0) {
                                st1[28] = act[12];
                            }
                            st1[29] = act[30];
                            if (w1 != 0) {
                                st1[29] = act[14];
                            }
                            st1[30] = act[29];
                            if (w1 != 0) {
                                st1[30] = act[13];
                            }
                            st1[31] = act[31];
                            if (w1 != 0) {
                                st1[31] = act[15];
                            }
                            tmem_st_x32_f32(sc0, st1);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            float h0[16];
                            float h1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&h0[0])), "=r"(*reinterpret_cast<uint32_t*>(&h0[1])), "=r"(*reinterpret_cast<uint32_t*>(&h0[2])), "=r"(*reinterpret_cast<uint32_t*>(&h0[3])), "=r"(*reinterpret_cast<uint32_t*>(&h0[4])), "=r"(*reinterpret_cast<uint32_t*>(&h0[5])), "=r"(*reinterpret_cast<uint32_t*>(&h0[6])), "=r"(*reinterpret_cast<uint32_t*>(&h0[7])), "=r"(*reinterpret_cast<uint32_t*>(&h0[8])), "=r"(*reinterpret_cast<uint32_t*>(&h0[9])), "=r"(*reinterpret_cast<uint32_t*>(&h0[10])), "=r"(*reinterpret_cast<uint32_t*>(&h0[11])), "=r"(*reinterpret_cast<uint32_t*>(&h0[12])), "=r"(*reinterpret_cast<uint32_t*>(&h0[13])), "=r"(*reinterpret_cast<uint32_t*>(&h0[14])), "=r"(*reinterpret_cast<uint32_t*>(&h0[15]))
                                : "r"(sc0 + 0));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&h1[0])), "=r"(*reinterpret_cast<uint32_t*>(&h1[1])), "=r"(*reinterpret_cast<uint32_t*>(&h1[2])), "=r"(*reinterpret_cast<uint32_t*>(&h1[3])), "=r"(*reinterpret_cast<uint32_t*>(&h1[4])), "=r"(*reinterpret_cast<uint32_t*>(&h1[5])), "=r"(*reinterpret_cast<uint32_t*>(&h1[6])), "=r"(*reinterpret_cast<uint32_t*>(&h1[7])), "=r"(*reinterpret_cast<uint32_t*>(&h1[8])), "=r"(*reinterpret_cast<uint32_t*>(&h1[9])), "=r"(*reinterpret_cast<uint32_t*>(&h1[10])), "=r"(*reinterpret_cast<uint32_t*>(&h1[11])), "=r"(*reinterpret_cast<uint32_t*>(&h1[12])), "=r"(*reinterpret_cast<uint32_t*>(&h1[13])), "=r"(*reinterpret_cast<uint32_t*>(&h1[14])), "=r"(*reinterpret_cast<uint32_t*>(&h1[15]))
                                : "r"(sc0 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float st2[16];
                            st2[0] = h0[8];
                            st2[1] = h0[9];
                            st2[2] = h0[10];
                            st2[3] = h0[11];
                            st2[4] = h1[8];
                            st2[5] = h1[9];
                            st2[6] = h1[10];
                            st2[7] = h1[11];
                            st2[8] = h0[12];
                            st2[9] = h0[13];
                            st2[10] = h0[14];
                            st2[11] = h0[15];
                            st2[12] = h1[12];
                            st2[13] = h1[13];
                            st2[14] = h1[14];
                            st2[15] = h1[15];
                            tmem_st_x16_f32(sc0 + 32, st2);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            float rhi[16];
                            float rlo[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rhi[0])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[1])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[2])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[3])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[4])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[5])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[6])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[7])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[8])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[9])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[10])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[11])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[12])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[13])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[14])), "=r"(*reinterpret_cast<uint32_t*>(&rhi[15]))
                                : "r"(sc0 + (unsigned int)((16 << 16) + 32)));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rlo[0])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[1])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[2])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[3])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[4])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[5])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[6])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[7])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[8])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[9])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[10])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[11])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[12])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[13])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[14])), "=r"(*reinterpret_cast<uint32_t*>(&rlo[15]))
                                : "r"(sc0 + 16));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int m_cta = m_one * 2 + cta;
                            int sf_col = m_cta * 4 + warp;
                            mbarrier_wait(sempty_addr + (sst) * 8, _phase_sempty);
                            int tok_0 = 16 * (2 * a) + 8 * (q >> 1) + 2 * ee + (q & 1);
                            float x_0[16];
                            float own_0 = h0[0];
                            float other_0 = rhi[0];
                            if (a != 0) {
                                other_0 = rlo[0];
                            }
                            x_0[0] = own_0;
                            if (a != 0) {
                                x_0[0] = other_0;
                            }
                            float own_1 = h0[0];
                            float other_1 = rhi[0];
                            if (a != 0) {
                                other_1 = rlo[0];
                            }
                            x_0[1] = own_1;
                            if (a != 1) {
                                x_0[1] = other_1;
                            }
                            float own_2 = h0[2];
                            float other_2 = rhi[2];
                            if (a != 0) {
                                other_2 = rlo[2];
                            }
                            x_0[2] = own_2;
                            if (a != 0) {
                                x_0[2] = other_2;
                            }
                            float own_3 = h0[2];
                            float other_3 = rhi[2];
                            if (a != 0) {
                                other_3 = rlo[2];
                            }
                            x_0[3] = own_3;
                            if (a != 1) {
                                x_0[3] = other_3;
                            }
                            float own_4 = h1[0];
                            float other_4 = rhi[4];
                            if (a != 0) {
                                other_4 = rlo[4];
                            }
                            x_0[4] = own_4;
                            if (a != 0) {
                                x_0[4] = other_4;
                            }
                            float own_5 = h1[0];
                            float other_5 = rhi[4];
                            if (a != 0) {
                                other_5 = rlo[4];
                            }
                            x_0[5] = own_5;
                            if (a != 1) {
                                x_0[5] = other_5;
                            }
                            float own_6 = h1[2];
                            float other_6 = rhi[6];
                            if (a != 0) {
                                other_6 = rlo[6];
                            }
                            x_0[6] = own_6;
                            if (a != 0) {
                                x_0[6] = other_6;
                            }
                            float own_7 = h1[2];
                            float other_7 = rhi[6];
                            if (a != 0) {
                                other_7 = rlo[6];
                            }
                            x_0[7] = own_7;
                            if (a != 1) {
                                x_0[7] = other_7;
                            }
                            float own_8 = h0[1];
                            float other_8 = rhi[1];
                            if (a != 0) {
                                other_8 = rlo[1];
                            }
                            x_0[8] = own_8;
                            if (a != 0) {
                                x_0[8] = other_8;
                            }
                            float own_9 = h0[1];
                            float other_9 = rhi[1];
                            if (a != 0) {
                                other_9 = rlo[1];
                            }
                            x_0[9] = own_9;
                            if (a != 1) {
                                x_0[9] = other_9;
                            }
                            float own_10 = h0[3];
                            float other_10 = rhi[3];
                            if (a != 0) {
                                other_10 = rlo[3];
                            }
                            x_0[10] = own_10;
                            if (a != 0) {
                                x_0[10] = other_10;
                            }
                            float own_11 = h0[3];
                            float other_11 = rhi[3];
                            if (a != 0) {
                                other_11 = rlo[3];
                            }
                            x_0[11] = own_11;
                            if (a != 1) {
                                x_0[11] = other_11;
                            }
                            float own_12 = h1[1];
                            float other_12 = rhi[5];
                            if (a != 0) {
                                other_12 = rlo[5];
                            }
                            x_0[12] = own_12;
                            if (a != 0) {
                                x_0[12] = other_12;
                            }
                            float own_13 = h1[1];
                            float other_13 = rhi[5];
                            if (a != 0) {
                                other_13 = rlo[5];
                            }
                            x_0[13] = own_13;
                            if (a != 1) {
                                x_0[13] = other_13;
                            }
                            float own_14 = h1[3];
                            float other_14 = rhi[7];
                            if (a != 0) {
                                other_14 = rlo[7];
                            }
                            x_0[14] = own_14;
                            if (a != 0) {
                                x_0[14] = other_14;
                            }
                            float own_15 = h1[3];
                            float other_15 = rhi[7];
                            if (a != 0) {
                                other_15 = rlo[7];
                            }
                            x_0[15] = own_15;
                            if (a != 1) {
                                x_0[15] = other_15;
                            }
                            float2 sc2_0 = make_float2(scores[0], scores[0]);
                            float2 _f32x2_mul_a_113 = make_float2(x_0[0], x_0[1]);
                            float2 xs_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[0] = xs_0.x;
                            x_0[1] = xs_0.y;
                            float2 _f32x2_mul_a_114 = make_float2(x_0[2], x_0[3]);
                            float2 xs_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[2] = xs_1.x;
                            x_0[3] = xs_1.y;
                            float2 _f32x2_mul_a_115 = make_float2(x_0[4], x_0[5]);
                            float2 xs_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[4] = xs_2.x;
                            x_0[5] = xs_2.y;
                            float2 _f32x2_mul_a_116 = make_float2(x_0[6], x_0[7]);
                            float2 xs_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[6] = xs_3.x;
                            x_0[7] = xs_3.y;
                            float2 _f32x2_mul_a_117 = make_float2(x_0[8], x_0[9]);
                            float2 xs_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[8] = xs_4.x;
                            x_0[9] = xs_4.y;
                            float2 _f32x2_mul_a_118 = make_float2(x_0[10], x_0[11]);
                            float2 xs_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[10] = xs_5.x;
                            x_0[11] = xs_5.y;
                            float2 _f32x2_mul_a_119 = make_float2(x_0[12], x_0[13]);
                            float2 xs_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[12] = xs_6.x;
                            x_0[13] = xs_6.y;
                            float2 _f32x2_mul_a_120 = make_float2(x_0[14], x_0[15]);
                            float2 xs_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&sc2_0));
                            x_0[14] = xs_7.x;
                            x_0[15] = xs_7.y;
                            float amax_0 = x_0[0];
                            float _max_121;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_121) : "f"(amax_0), "f"(x_0[1]));
                            amax_0 = _max_121;
                            float _max_122;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_122) : "f"(amax_0), "f"(x_0[2]));
                            amax_0 = _max_122;
                            float _max_123;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_123) : "f"(amax_0), "f"(x_0[3]));
                            amax_0 = _max_123;
                            float _max_124;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_124) : "f"(amax_0), "f"(x_0[4]));
                            amax_0 = _max_124;
                            float _max_125;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_125) : "f"(amax_0), "f"(x_0[5]));
                            amax_0 = _max_125;
                            float _max_126;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_126) : "f"(amax_0), "f"(x_0[6]));
                            amax_0 = _max_126;
                            float _max_127;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_127) : "f"(amax_0), "f"(x_0[7]));
                            amax_0 = _max_127;
                            float _max_128;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_128) : "f"(amax_0), "f"(x_0[8]));
                            amax_0 = _max_128;
                            float _max_129;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_129) : "f"(amax_0), "f"(x_0[9]));
                            amax_0 = _max_129;
                            float _max_130;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_130) : "f"(amax_0), "f"(x_0[10]));
                            amax_0 = _max_130;
                            float _max_131;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_131) : "f"(amax_0), "f"(x_0[11]));
                            amax_0 = _max_131;
                            float _max_132;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_132) : "f"(amax_0), "f"(x_0[12]));
                            amax_0 = _max_132;
                            float _max_133;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_133) : "f"(amax_0), "f"(x_0[13]));
                            amax_0 = _max_133;
                            float _max_134;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_134) : "f"(amax_0), "f"(x_0[14]));
                            amax_0 = _max_134;
                            float _max_135;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_135) : "f"(amax_0), "f"(x_0[15]));
                            amax_0 = _max_135;
                            float amax_mag_0 = __uint_as_float(__as_u32(amax_0) & 2147483647);
                            float sfc_0;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfc_0) : "f"(norm), "f"(mul_f32_rn_noftz(amax_mag_0, 0.16666667f)));
                            uint16_t sf_pair_0;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_0) : "f"(0.0f), "f"(sfc_0));
                            unsigned int sf_byte_0 = (unsigned int)sf_pair_0 & 255;
                            uint32_t sf_f16_0;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16_0) : "h"((uint16_t)((uint16_t)sf_byte_0)));
                            float sf_rt_0;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rt_0) : "h"((uint16_t)(sf_f16_0)));
                            float _rcp_136 = approx_rcp(sf_rt_0);
                            float enc_0;
                            asm("min.f32 %0, %1, %2;" : "=f"(enc_0) : "f"(mul_f32_rn_noftz(norm, _rcp_136)), "f"(3.4028235e+38f));
                            float mask_0;
                            asm("min.f32 %0, %1, %2;" : "=f"(mask_0) : "f"(mul_f32_rn_noftz(sf_rt_0, 1e+30f)), "f"(1.0f));
                            float enc_m_0;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(enc_m_0) : "f"(enc_0), "f"(mask_0));
                            float2 em2_0 = make_float2(enc_m_0, enc_m_0);
                            float2 _f32x2_mul_a_137 = make_float2(x_0[0], x_0[1]);
                            float2 xe_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&em2_0));
                            x_0[0] = xe_0.x;
                            x_0[1] = xe_0.y;
                            float2 _f32x2_mul_a_138 = make_float2(x_0[2], x_0[3]);
                            float2 xe_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&em2_0));
                            x_0[2] = xe_1.x;
                            x_0[3] = xe_1.y;
                            float2 _f32x2_mul_a_139 = make_float2(x_0[4], x_0[5]);
                            float2 xe_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&em2_0));
                            x_0[4] = xe_2.x;
                            x_0[5] = xe_2.y;
                            float2 _f32x2_mul_a_140 = make_float2(x_0[6], x_0[7]);
                            float2 xe_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&em2_0));
                            x_0[6] = xe_3.x;
                            x_0[7] = xe_3.y;
                            float2 _f32x2_mul_a_141 = make_float2(x_0[8], x_0[9]);
                            float2 xe_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&em2_0));
                            x_0[8] = xe_4.x;
                            x_0[9] = xe_4.y;
                            float2 _f32x2_mul_a_142 = make_float2(x_0[10], x_0[11]);
                            float2 xe_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&em2_0));
                            x_0[10] = xe_5.x;
                            x_0[11] = xe_5.y;
                            float2 _f32x2_mul_a_143 = make_float2(x_0[12], x_0[13]);
                            float2 xe_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&em2_0));
                            x_0[12] = xe_6.x;
                            x_0[13] = xe_6.y;
                            float2 _f32x2_mul_a_144 = make_float2(x_0[14], x_0[15]);
                            float2 xe_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&em2_0));
                            x_0[14] = xe_7.x;
                            x_0[15] = xe_7.y;
                            unsigned int packed_0[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_0[0]) : "f"(x_0[0]), "f"(x_0[1]), "f"(x_0[2]), "f"(x_0[3]), "f"(x_0[4]), "f"(x_0[5]), "f"(x_0[6]), "f"(x_0[7]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_0[1]) : "f"(x_0[8]), "f"(x_0[9]), "f"(x_0[10]), "f"(x_0[11]), "f"(x_0[12]), "f"(x_0[13]), "f"(x_0[14]), "f"(x_0[15]));
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(sm_fc1_addr + sst * 2048 + (unsigned int)(tok_0 * 32 + warp * 8)), "r"(packed_0[0]), "r"(packed_0[1]) : "memory");
                            int sf_row_0 = fb + block * 256 + s0 + tok_0;
                            int sf_offset_0 = ((sf_row_0 >> 7) * 48 + (sf_col >> 2)) * 512 + (sf_row_0 & 31) * 16 + (sf_row_0 >> 5 & 3) * 4 + (sf_col & 3);
                            fc1_out_sf[sf_offset_0] = (uint8_t)sf_byte_0;
                            int tok_1 = 16 * (2 * a + 1) + 8 * (q >> 1) + 2 * ee + (q & 1);
                            float x_1[16];
                            float own_16 = h0[4];
                            float other_16 = rhi[8];
                            if (a != 0) {
                                other_16 = rlo[8];
                            }
                            x_1[0] = own_16;
                            if (a != 0) {
                                x_1[0] = other_16;
                            }
                            float own_17 = h0[4];
                            float other_17 = rhi[8];
                            if (a != 0) {
                                other_17 = rlo[8];
                            }
                            x_1[1] = own_17;
                            if (a != 1) {
                                x_1[1] = other_17;
                            }
                            float own_18 = h0[6];
                            float other_18 = rhi[10];
                            if (a != 0) {
                                other_18 = rlo[10];
                            }
                            x_1[2] = own_18;
                            if (a != 0) {
                                x_1[2] = other_18;
                            }
                            float own_19 = h0[6];
                            float other_19 = rhi[10];
                            if (a != 0) {
                                other_19 = rlo[10];
                            }
                            x_1[3] = own_19;
                            if (a != 1) {
                                x_1[3] = other_19;
                            }
                            float own_20 = h1[4];
                            float other_20 = rhi[12];
                            if (a != 0) {
                                other_20 = rlo[12];
                            }
                            x_1[4] = own_20;
                            if (a != 0) {
                                x_1[4] = other_20;
                            }
                            float own_21 = h1[4];
                            float other_21 = rhi[12];
                            if (a != 0) {
                                other_21 = rlo[12];
                            }
                            x_1[5] = own_21;
                            if (a != 1) {
                                x_1[5] = other_21;
                            }
                            float own_22 = h1[6];
                            float other_22 = rhi[14];
                            if (a != 0) {
                                other_22 = rlo[14];
                            }
                            x_1[6] = own_22;
                            if (a != 0) {
                                x_1[6] = other_22;
                            }
                            float own_23 = h1[6];
                            float other_23 = rhi[14];
                            if (a != 0) {
                                other_23 = rlo[14];
                            }
                            x_1[7] = own_23;
                            if (a != 1) {
                                x_1[7] = other_23;
                            }
                            float own_24 = h0[5];
                            float other_24 = rhi[9];
                            if (a != 0) {
                                other_24 = rlo[9];
                            }
                            x_1[8] = own_24;
                            if (a != 0) {
                                x_1[8] = other_24;
                            }
                            float own_25 = h0[5];
                            float other_25 = rhi[9];
                            if (a != 0) {
                                other_25 = rlo[9];
                            }
                            x_1[9] = own_25;
                            if (a != 1) {
                                x_1[9] = other_25;
                            }
                            float own_26 = h0[7];
                            float other_26 = rhi[11];
                            if (a != 0) {
                                other_26 = rlo[11];
                            }
                            x_1[10] = own_26;
                            if (a != 0) {
                                x_1[10] = other_26;
                            }
                            float own_27 = h0[7];
                            float other_27 = rhi[11];
                            if (a != 0) {
                                other_27 = rlo[11];
                            }
                            x_1[11] = own_27;
                            if (a != 1) {
                                x_1[11] = other_27;
                            }
                            float own_28 = h1[5];
                            float other_28 = rhi[13];
                            if (a != 0) {
                                other_28 = rlo[13];
                            }
                            x_1[12] = own_28;
                            if (a != 0) {
                                x_1[12] = other_28;
                            }
                            float own_29 = h1[5];
                            float other_29 = rhi[13];
                            if (a != 0) {
                                other_29 = rlo[13];
                            }
                            x_1[13] = own_29;
                            if (a != 1) {
                                x_1[13] = other_29;
                            }
                            float own_30 = h1[7];
                            float other_30 = rhi[15];
                            if (a != 0) {
                                other_30 = rlo[15];
                            }
                            x_1[14] = own_30;
                            if (a != 0) {
                                x_1[14] = other_30;
                            }
                            float own_31 = h1[7];
                            float other_31 = rhi[15];
                            if (a != 0) {
                                other_31 = rlo[15];
                            }
                            x_1[15] = own_31;
                            if (a != 1) {
                                x_1[15] = other_31;
                            }
                            float2 sc2_1 = make_float2(scores[1], scores[1]);
                            float2 _f32x2_mul_a_145 = make_float2(x_1[0], x_1[1]);
                            float2 xs_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_145), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[0] = xs_8.x;
                            x_1[1] = xs_8.y;
                            float2 _f32x2_mul_a_146 = make_float2(x_1[2], x_1[3]);
                            float2 xs_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[2] = xs_9.x;
                            x_1[3] = xs_9.y;
                            float2 _f32x2_mul_a_147 = make_float2(x_1[4], x_1[5]);
                            float2 xs_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_147), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[4] = xs_10.x;
                            x_1[5] = xs_10.y;
                            float2 _f32x2_mul_a_148 = make_float2(x_1[6], x_1[7]);
                            float2 xs_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[6] = xs_11.x;
                            x_1[7] = xs_11.y;
                            float2 _f32x2_mul_a_149 = make_float2(x_1[8], x_1[9]);
                            float2 xs_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[8] = xs_12.x;
                            x_1[9] = xs_12.y;
                            float2 _f32x2_mul_a_150 = make_float2(x_1[10], x_1[11]);
                            float2 xs_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[10] = xs_13.x;
                            x_1[11] = xs_13.y;
                            float2 _f32x2_mul_a_151 = make_float2(x_1[12], x_1[13]);
                            float2 xs_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[12] = xs_14.x;
                            x_1[13] = xs_14.y;
                            float2 _f32x2_mul_a_152 = make_float2(x_1[14], x_1[15]);
                            float2 xs_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xs_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&sc2_1));
                            x_1[14] = xs_15.x;
                            x_1[15] = xs_15.y;
                            float amax_1 = x_1[0];
                            float _max_153;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_153) : "f"(amax_1), "f"(x_1[1]));
                            amax_1 = _max_153;
                            float _max_154;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_154) : "f"(amax_1), "f"(x_1[2]));
                            amax_1 = _max_154;
                            float _max_155;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_155) : "f"(amax_1), "f"(x_1[3]));
                            amax_1 = _max_155;
                            float _max_156;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_156) : "f"(amax_1), "f"(x_1[4]));
                            amax_1 = _max_156;
                            float _max_157;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_157) : "f"(amax_1), "f"(x_1[5]));
                            amax_1 = _max_157;
                            float _max_158;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_158) : "f"(amax_1), "f"(x_1[6]));
                            amax_1 = _max_158;
                            float _max_159;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_159) : "f"(amax_1), "f"(x_1[7]));
                            amax_1 = _max_159;
                            float _max_160;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_160) : "f"(amax_1), "f"(x_1[8]));
                            amax_1 = _max_160;
                            float _max_161;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_161) : "f"(amax_1), "f"(x_1[9]));
                            amax_1 = _max_161;
                            float _max_162;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_162) : "f"(amax_1), "f"(x_1[10]));
                            amax_1 = _max_162;
                            float _max_163;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_163) : "f"(amax_1), "f"(x_1[11]));
                            amax_1 = _max_163;
                            float _max_164;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_164) : "f"(amax_1), "f"(x_1[12]));
                            amax_1 = _max_164;
                            float _max_165;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_165) : "f"(amax_1), "f"(x_1[13]));
                            amax_1 = _max_165;
                            float _max_166;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_166) : "f"(amax_1), "f"(x_1[14]));
                            amax_1 = _max_166;
                            float _max_167;
                            asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_167) : "f"(amax_1), "f"(x_1[15]));
                            amax_1 = _max_167;
                            float amax_mag_1 = __uint_as_float(__as_u32(amax_1) & 2147483647);
                            float sfc_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfc_1) : "f"(norm), "f"(mul_f32_rn_noftz(amax_mag_1, 0.16666667f)));
                            uint16_t sf_pair_1;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_1) : "f"(0.0f), "f"(sfc_1));
                            unsigned int sf_byte_1 = (unsigned int)sf_pair_1 & 255;
                            uint32_t sf_f16_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16_1) : "h"((uint16_t)((uint16_t)sf_byte_1)));
                            float sf_rt_1;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rt_1) : "h"((uint16_t)(sf_f16_1)));
                            float _rcp_168 = approx_rcp(sf_rt_1);
                            float enc_1;
                            asm("min.f32 %0, %1, %2;" : "=f"(enc_1) : "f"(mul_f32_rn_noftz(norm, _rcp_168)), "f"(3.4028235e+38f));
                            float mask_1;
                            asm("min.f32 %0, %1, %2;" : "=f"(mask_1) : "f"(mul_f32_rn_noftz(sf_rt_1, 1e+30f)), "f"(1.0f));
                            float enc_m_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(enc_m_1) : "f"(enc_1), "f"(mask_1));
                            float2 em2_1 = make_float2(enc_m_1, enc_m_1);
                            float2 _f32x2_mul_a_169 = make_float2(x_1[0], x_1[1]);
                            float2 xe_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&em2_1));
                            x_1[0] = xe_8.x;
                            x_1[1] = xe_8.y;
                            float2 _f32x2_mul_a_170 = make_float2(x_1[2], x_1[3]);
                            float2 xe_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&em2_1));
                            x_1[2] = xe_9.x;
                            x_1[3] = xe_9.y;
                            float2 _f32x2_mul_a_171 = make_float2(x_1[4], x_1[5]);
                            float2 xe_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&em2_1));
                            x_1[4] = xe_10.x;
                            x_1[5] = xe_10.y;
                            float2 _f32x2_mul_a_172 = make_float2(x_1[6], x_1[7]);
                            float2 xe_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&em2_1));
                            x_1[6] = xe_11.x;
                            x_1[7] = xe_11.y;
                            float2 _f32x2_mul_a_173 = make_float2(x_1[8], x_1[9]);
                            float2 xe_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&em2_1));
                            x_1[8] = xe_12.x;
                            x_1[9] = xe_12.y;
                            float2 _f32x2_mul_a_174 = make_float2(x_1[10], x_1[11]);
                            float2 xe_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_174), "l"(*(const unsigned long long*)&em2_1));
                            x_1[10] = xe_13.x;
                            x_1[11] = xe_13.y;
                            float2 _f32x2_mul_a_175 = make_float2(x_1[12], x_1[13]);
                            float2 xe_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&em2_1));
                            x_1[12] = xe_14.x;
                            x_1[13] = xe_14.y;
                            float2 _f32x2_mul_a_176 = make_float2(x_1[14], x_1[15]);
                            float2 xe_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xe_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&em2_1));
                            x_1[14] = xe_15.x;
                            x_1[15] = xe_15.y;
                            unsigned int packed_1[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[0]) : "f"(x_1[0]), "f"(x_1[1]), "f"(x_1[2]), "f"(x_1[3]), "f"(x_1[4]), "f"(x_1[5]), "f"(x_1[6]), "f"(x_1[7]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[1]) : "f"(x_1[8]), "f"(x_1[9]), "f"(x_1[10]), "f"(x_1[11]), "f"(x_1[12]), "f"(x_1[13]), "f"(x_1[14]), "f"(x_1[15]));
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(sm_fc1_addr + sst * 2048 + (unsigned int)(tok_1 * 32 + warp * 8)), "r"(packed_1[0]), "r"(packed_1[1]) : "memory");
                            int sf_row_1 = fb + block * 256 + s0 + tok_1;
                            int sf_offset_1 = ((sf_row_1 >> 7) * 48 + (sf_col >> 2)) * 512 + (sf_row_1 & 31) * 16 + (sf_row_1 >> 5 & 3) * 4 + (sf_col & 3);
                            fc1_out_sf[sf_offset_1] = (uint8_t)sf_byte_1;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(sfull_addr + (sst) * 8);
                            sst += 1;
                            if (sst == 4) { sst = 0; _phase_sempty ^= 1; }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((ce_addr + (cst) * 8) & 0xFEFFFFFF) : "memory");
                        cst += 1;
                        if (cst == 2) { cst = 0; _phase_cf ^= 1; }
                        if (fpend[0] >= 0) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            if (warp == 0) {
                                if (elect_sync()) {
                                    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc2_done) + (fpend[0]))), "r"(static_cast<unsigned int>((unsigned int)fpend[1])) : "memory");
                                }
                            }
                            fpend[0] = -1;
                            fpend[1] = 0;
                        }
                    }
                    if (ph == 1) {
                        float alpha_1 = fc2_alpha[e];
                        int t = warp * 32 + lane;
                        int lane_2 = lane;
                        int odd = lane_2 & 1;
                        int word = t - odd >> 1;
                        int h0_1 = m_one * 256 + cta * 128;
                        unsigned int dlo[4];
                        unsigned int dhi[4];
                        long long mrec[4] = {0};
                        int tq0_0 = warp * 16 + (lane_2 & 15);
                        int mine0_0 = lane_2 >> 4 ^ 0 ^ 1;
                        if (tq0_0 < valid && mine0_0 != 0) {
                            mrec[0] = token_src_metadata[(db + block * 256 + tq0_0 >> 4 << 5) + (epi_copy << 4) + (db + block * 256 + tq0_0 & 15)];
                        }
                        int tq0_1 = 64 + warp * 16 + (lane_2 & 15);
                        int mine0_1 = lane_2 >> 4 ^ 1 ^ 1;
                        if (tq0_1 < valid && mine0_1 != 0) {
                            mrec[1] = token_src_metadata[(db + block * 256 + tq0_1 >> 4 << 5) + (epi_copy << 4) + (db + block * 256 + tq0_1 & 15)];
                        }
                        int tq0_2 = 128 + warp * 16 + (lane_2 & 15);
                        int mine0_2 = lane_2 >> 4 ^ 0 ^ 1;
                        if (tq0_2 < valid && mine0_2 != 0) {
                            mrec[2] = token_src_metadata[(db + block * 256 + tq0_2 >> 4 << 5) + (epi_copy << 4) + (db + block * 256 + tq0_2 & 15)];
                        }
                        int tq0_3 = 192 + warp * 16 + (lane_2 & 15);
                        int mine0_3 = lane_2 >> 4 ^ 1 ^ 1;
                        if (tq0_3 < valid && mine0_3 != 0) {
                            mrec[3] = token_src_metadata[(db + block * 256 + tq0_3 >> 4 << 5) + (epi_copy << 4) + (db + block * 256 + tq0_3 & 15)];
                        }
                        dlo[0] = 0;
                        dhi[0] = 0;
                        int tq_0 = warp * 16 + (lane_2 & 15);
                        int mine_0 = 1;
                        mine_0 = lane_2 >> 4 ^ 0 ^ 1;
                        if (tq_0 < valid && mine_0 != 0) {
                            long long record_0 = mrec[0];
                            int lo_0 = (int)(record_0 & 4294967295ll);
                            int rank_0 = cake_floor_mod<int>(lo_0 >> 24 & 255, mega_pg_world);
                            unsigned long long d_0 = 0ull;
                            int order_0 = (int)(record_0 >> 32ll & 255ll);
                            int copies_0 = (int)(record_0 >> 40ll & 255ll) + 1;
                            int dslot_0 = mega_pg_rank;
                            int direct_0 = 0;
                            if (copies_0 == 1) {
                                direct_0 = 1;
                            }
                            if (direct_0 != 0) {
                                unsigned long long base1_0 = peer_base[0];
                                if (rank_0 == 1) {
                                    base1_0 = peer_base[1];
                                }
                                if (rank_0 == 2) {
                                    base1_0 = peer_base[2];
                                }
                                if (rank_0 == 3) {
                                    base1_0 = peer_base[3];
                                }
                                if (rank_0 == 4) {
                                    base1_0 = peer_base[4];
                                }
                                if (rank_0 == 5) {
                                    base1_0 = peer_base[5];
                                }
                                if (rank_0 == 6) {
                                    base1_0 = peer_base[6];
                                }
                                if (rank_0 == 7) {
                                    base1_0 = peer_base[7];
                                }
                                d_0 = base1_0 + (unsigned long long)(((lo_0 & 65535) * 8 + dslot_0) * 7168 + h0_1) * 2ull;
                            } else {
                                d_0 = (unsigned long long)dedup_rows + (unsigned long long)(rank_0 * dedup_cap + (lo_0 & 65535)) * 86016ull + (unsigned long long)((h0_1 * 2 / 2048 * 6 + order_0) * 2048 + h0_1 * 2 % 2048);
                            }
                            dlo[0] = (unsigned int)(d_0 & 4294967295ull);
                            dhi[0] = (unsigned int)(d_0 >> 32ull);
                        }
                        dlo[1] = 0;
                        dhi[1] = 0;
                        int tq_1 = 64 + warp * 16 + (lane_2 & 15);
                        int mine_1 = 1;
                        mine_1 = lane_2 >> 4 ^ 1 ^ 1;
                        if (tq_1 < valid && mine_1 != 0) {
                            long long record_1 = mrec[1];
                            int lo_1 = (int)(record_1 & 4294967295ll);
                            int rank_1 = cake_floor_mod<int>(lo_1 >> 24 & 255, mega_pg_world);
                            unsigned long long d_1 = 0ull;
                            int order_1 = (int)(record_1 >> 32ll & 255ll);
                            int copies_1 = (int)(record_1 >> 40ll & 255ll) + 1;
                            int dslot_1 = mega_pg_rank;
                            int direct_1 = 0;
                            if (copies_1 == 1) {
                                direct_1 = 1;
                            }
                            if (direct_1 != 0) {
                                unsigned long long base1_1 = peer_base[0];
                                if (rank_1 == 1) {
                                    base1_1 = peer_base[1];
                                }
                                if (rank_1 == 2) {
                                    base1_1 = peer_base[2];
                                }
                                if (rank_1 == 3) {
                                    base1_1 = peer_base[3];
                                }
                                if (rank_1 == 4) {
                                    base1_1 = peer_base[4];
                                }
                                if (rank_1 == 5) {
                                    base1_1 = peer_base[5];
                                }
                                if (rank_1 == 6) {
                                    base1_1 = peer_base[6];
                                }
                                if (rank_1 == 7) {
                                    base1_1 = peer_base[7];
                                }
                                d_1 = base1_1 + (unsigned long long)(((lo_1 & 65535) * 8 + dslot_1) * 7168 + h0_1) * 2ull;
                            } else {
                                d_1 = (unsigned long long)dedup_rows + (unsigned long long)(rank_1 * dedup_cap + (lo_1 & 65535)) * 86016ull + (unsigned long long)((h0_1 * 2 / 2048 * 6 + order_1) * 2048 + h0_1 * 2 % 2048);
                            }
                            dlo[1] = (unsigned int)(d_1 & 4294967295ull);
                            dhi[1] = (unsigned int)(d_1 >> 32ull);
                        }
                        dlo[2] = 0;
                        dhi[2] = 0;
                        int tq_2 = 128 + warp * 16 + (lane_2 & 15);
                        int mine_2 = 1;
                        mine_2 = lane_2 >> 4 ^ 0 ^ 1;
                        if (tq_2 < valid && mine_2 != 0) {
                            long long record_2 = mrec[2];
                            int lo_2 = (int)(record_2 & 4294967295ll);
                            int rank_2 = cake_floor_mod<int>(lo_2 >> 24 & 255, mega_pg_world);
                            unsigned long long d_2 = 0ull;
                            int order_2 = (int)(record_2 >> 32ll & 255ll);
                            int copies_2 = (int)(record_2 >> 40ll & 255ll) + 1;
                            int dslot_2 = mega_pg_rank;
                            int direct_2 = 0;
                            if (copies_2 == 1) {
                                direct_2 = 1;
                            }
                            if (direct_2 != 0) {
                                unsigned long long base1_2 = peer_base[0];
                                if (rank_2 == 1) {
                                    base1_2 = peer_base[1];
                                }
                                if (rank_2 == 2) {
                                    base1_2 = peer_base[2];
                                }
                                if (rank_2 == 3) {
                                    base1_2 = peer_base[3];
                                }
                                if (rank_2 == 4) {
                                    base1_2 = peer_base[4];
                                }
                                if (rank_2 == 5) {
                                    base1_2 = peer_base[5];
                                }
                                if (rank_2 == 6) {
                                    base1_2 = peer_base[6];
                                }
                                if (rank_2 == 7) {
                                    base1_2 = peer_base[7];
                                }
                                d_2 = base1_2 + (unsigned long long)(((lo_2 & 65535) * 8 + dslot_2) * 7168 + h0_1) * 2ull;
                            } else {
                                d_2 = (unsigned long long)dedup_rows + (unsigned long long)(rank_2 * dedup_cap + (lo_2 & 65535)) * 86016ull + (unsigned long long)((h0_1 * 2 / 2048 * 6 + order_2) * 2048 + h0_1 * 2 % 2048);
                            }
                            dlo[2] = (unsigned int)(d_2 & 4294967295ull);
                            dhi[2] = (unsigned int)(d_2 >> 32ull);
                        }
                        dlo[3] = 0;
                        dhi[3] = 0;
                        int tq_3 = 192 + warp * 16 + (lane_2 & 15);
                        int mine_3 = 1;
                        mine_3 = lane_2 >> 4 ^ 1 ^ 1;
                        if (tq_3 < valid && mine_3 != 0) {
                            long long record_3 = mrec[3];
                            int lo_3 = (int)(record_3 & 4294967295ll);
                            int rank_3 = cake_floor_mod<int>(lo_3 >> 24 & 255, mega_pg_world);
                            unsigned long long d_3 = 0ull;
                            int order_3 = (int)(record_3 >> 32ll & 255ll);
                            int copies_3 = (int)(record_3 >> 40ll & 255ll) + 1;
                            int dslot_3 = mega_pg_rank;
                            int direct_3 = 0;
                            if (copies_3 == 1) {
                                direct_3 = 1;
                            }
                            if (direct_3 != 0) {
                                unsigned long long base1_3 = peer_base[0];
                                if (rank_3 == 1) {
                                    base1_3 = peer_base[1];
                                }
                                if (rank_3 == 2) {
                                    base1_3 = peer_base[2];
                                }
                                if (rank_3 == 3) {
                                    base1_3 = peer_base[3];
                                }
                                if (rank_3 == 4) {
                                    base1_3 = peer_base[4];
                                }
                                if (rank_3 == 5) {
                                    base1_3 = peer_base[5];
                                }
                                if (rank_3 == 6) {
                                    base1_3 = peer_base[6];
                                }
                                if (rank_3 == 7) {
                                    base1_3 = peer_base[7];
                                }
                                d_3 = base1_3 + (unsigned long long)(((lo_3 & 65535) * 8 + dslot_3) * 7168 + h0_1) * 2ull;
                            } else {
                                d_3 = (unsigned long long)dedup_rows + (unsigned long long)(rank_3 * dedup_cap + (lo_3 & 65535)) * 86016ull + (unsigned long long)((h0_1 * 2 / 2048 * 6 + order_3) * 2048 + h0_1 * 2 % 2048);
                            }
                            dlo[3] = (unsigned int)(d_3 & 4294967295ull);
                            dhi[3] = (unsigned int)(d_3 >> 32ull);
                        }
                        mbarrier_wait(cf_addr + (cst) * 8, _phase_cf);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int acc_col_1 = cst * 256;
                        if (fpend[0] != bb + block || fpend[1] >= 8) {
                            if (fpend[0] >= 0) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                                asm volatile("fence.proxy.async.global;" ::: "memory");
                                asm volatile("barrier.sync 2, 128;" ::: "memory");
                                if (warp == 0) {
                                    if (elect_sync()) {
                                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc2_done) + (fpend[0]))), "r"(static_cast<unsigned int>((unsigned int)fpend[1])) : "memory");
                                    }
                                }
                                fpend[0] = -1;
                                fpend[1] = 0;
                            }
                        }
                        if (0 < valid) {
                            unsigned int buf_0 = sm_f2_addr + (unsigned int)(nit * 4 & 1) * 18432;
                            float vals_0[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[31]))
                                : "r"(taddr + (unsigned int)(warp * 32 << 16) + acc_col_1 + 0));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&vals_0[63]))
                                : "r"(taddr + (unsigned int)(warp * 32 + 16 << 16) + acc_col_1 + 0));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (64 >= valid) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((ce_addr + (cst) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            float2 alpha2_0 = make_float2(alpha_1, alpha_1);
                            float2 _f32x2_mul_a_177 = make_float2(vals_0[0], vals_0[1]);
                            float2 va2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[0] = va2_0.x;
                            vals_0[1] = va2_0.y;
                            float2 _f32x2_mul_a_178 = make_float2(vals_0[2], vals_0[3]);
                            float2 va2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[2] = va2_1.x;
                            vals_0[3] = va2_1.y;
                            float2 _f32x2_mul_a_179 = make_float2(vals_0[4], vals_0[5]);
                            float2 va2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[4] = va2_2.x;
                            vals_0[5] = va2_2.y;
                            float2 _f32x2_mul_a_180 = make_float2(vals_0[6], vals_0[7]);
                            float2 va2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[6] = va2_3.x;
                            vals_0[7] = va2_3.y;
                            float2 _f32x2_mul_a_181 = make_float2(vals_0[8], vals_0[9]);
                            float2 va2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[8] = va2_4.x;
                            vals_0[9] = va2_4.y;
                            float2 _f32x2_mul_a_182 = make_float2(vals_0[10], vals_0[11]);
                            float2 va2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[10] = va2_5.x;
                            vals_0[11] = va2_5.y;
                            float2 _f32x2_mul_a_183 = make_float2(vals_0[12], vals_0[13]);
                            float2 va2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[12] = va2_6.x;
                            vals_0[13] = va2_6.y;
                            float2 _f32x2_mul_a_184 = make_float2(vals_0[14], vals_0[15]);
                            float2 va2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[14] = va2_7.x;
                            vals_0[15] = va2_7.y;
                            float2 _f32x2_mul_a_185 = make_float2(vals_0[16], vals_0[17]);
                            float2 va2_8;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[16] = va2_8.x;
                            vals_0[17] = va2_8.y;
                            float2 _f32x2_mul_a_186 = make_float2(vals_0[18], vals_0[19]);
                            float2 va2_9;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[18] = va2_9.x;
                            vals_0[19] = va2_9.y;
                            float2 _f32x2_mul_a_187 = make_float2(vals_0[20], vals_0[21]);
                            float2 va2_10;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[20] = va2_10.x;
                            vals_0[21] = va2_10.y;
                            float2 _f32x2_mul_a_188 = make_float2(vals_0[22], vals_0[23]);
                            float2 va2_11;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[22] = va2_11.x;
                            vals_0[23] = va2_11.y;
                            float2 _f32x2_mul_a_189 = make_float2(vals_0[24], vals_0[25]);
                            float2 va2_12;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_189), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[24] = va2_12.x;
                            vals_0[25] = va2_12.y;
                            float2 _f32x2_mul_a_190 = make_float2(vals_0[26], vals_0[27]);
                            float2 va2_13;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_190), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[26] = va2_13.x;
                            vals_0[27] = va2_13.y;
                            float2 _f32x2_mul_a_191 = make_float2(vals_0[28], vals_0[29]);
                            float2 va2_14;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[28] = va2_14.x;
                            vals_0[29] = va2_14.y;
                            float2 _f32x2_mul_a_192 = make_float2(vals_0[30], vals_0[31]);
                            float2 va2_15;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_192), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[30] = va2_15.x;
                            vals_0[31] = va2_15.y;
                            float2 _f32x2_mul_a_193 = make_float2(vals_0[32], vals_0[33]);
                            float2 va2_16;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[32] = va2_16.x;
                            vals_0[33] = va2_16.y;
                            float2 _f32x2_mul_a_194 = make_float2(vals_0[34], vals_0[35]);
                            float2 va2_17;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_194), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[34] = va2_17.x;
                            vals_0[35] = va2_17.y;
                            float2 _f32x2_mul_a_195 = make_float2(vals_0[36], vals_0[37]);
                            float2 va2_18;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_195), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[36] = va2_18.x;
                            vals_0[37] = va2_18.y;
                            float2 _f32x2_mul_a_196 = make_float2(vals_0[38], vals_0[39]);
                            float2 va2_19;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_196), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[38] = va2_19.x;
                            vals_0[39] = va2_19.y;
                            float2 _f32x2_mul_a_197 = make_float2(vals_0[40], vals_0[41]);
                            float2 va2_20;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[40] = va2_20.x;
                            vals_0[41] = va2_20.y;
                            float2 _f32x2_mul_a_198 = make_float2(vals_0[42], vals_0[43]);
                            float2 va2_21;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[42] = va2_21.x;
                            vals_0[43] = va2_21.y;
                            float2 _f32x2_mul_a_199 = make_float2(vals_0[44], vals_0[45]);
                            float2 va2_22;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_199), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[44] = va2_22.x;
                            vals_0[45] = va2_22.y;
                            float2 _f32x2_mul_a_200 = make_float2(vals_0[46], vals_0[47]);
                            float2 va2_23;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[46] = va2_23.x;
                            vals_0[47] = va2_23.y;
                            float2 _f32x2_mul_a_201 = make_float2(vals_0[48], vals_0[49]);
                            float2 va2_24;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_201), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[48] = va2_24.x;
                            vals_0[49] = va2_24.y;
                            float2 _f32x2_mul_a_202 = make_float2(vals_0[50], vals_0[51]);
                            float2 va2_25;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[50] = va2_25.x;
                            vals_0[51] = va2_25.y;
                            float2 _f32x2_mul_a_203 = make_float2(vals_0[52], vals_0[53]);
                            float2 va2_26;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_203), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[52] = va2_26.x;
                            vals_0[53] = va2_26.y;
                            float2 _f32x2_mul_a_204 = make_float2(vals_0[54], vals_0[55]);
                            float2 va2_27;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[54] = va2_27.x;
                            vals_0[55] = va2_27.y;
                            float2 _f32x2_mul_a_205 = make_float2(vals_0[56], vals_0[57]);
                            float2 va2_28;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_205), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[56] = va2_28.x;
                            vals_0[57] = va2_28.y;
                            float2 _f32x2_mul_a_206 = make_float2(vals_0[58], vals_0[59]);
                            float2 va2_29;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[58] = va2_29.x;
                            vals_0[59] = va2_29.y;
                            float2 _f32x2_mul_a_207 = make_float2(vals_0[60], vals_0[61]);
                            float2 va2_30;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_207), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[60] = va2_30.x;
                            vals_0[61] = va2_30.y;
                            float2 _f32x2_mul_a_208 = make_float2(vals_0[62], vals_0[63]);
                            float2 va2_31;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&alpha2_0));
                            vals_0[62] = va2_31.x;
                            vals_0[63] = va2_31.y;
                            unsigned int pa_0[2];
                            unsigned int pb_0[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                                pa_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 32], vals_0[_lp*2+1 + 32]));
                                pb_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_209 = static_cast<uint32_t>(buf_0 + (unsigned int)((lane_2 & 7) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_209), "r"(*reinterpret_cast<const uint32_t*>(&pa_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_0[1]))
                                : "memory");
                            unsigned int pa_1[2];
                            unsigned int pb_1[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 4], vals_0[_lp*2+1 + 4]));
                                pa_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 36], vals_0[_lp*2+1 + 36]));
                                pb_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_210 = static_cast<uint32_t>(buf_0 + (unsigned int)((8 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_210), "r"(*reinterpret_cast<const uint32_t*>(&pa_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_1[1]))
                                : "memory");
                            unsigned int pa_2[2];
                            unsigned int pb_2[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 8], vals_0[_lp*2+1 + 8]));
                                pa_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 40], vals_0[_lp*2+1 + 40]));
                                pb_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_211 = static_cast<uint32_t>(buf_0 + (unsigned int)((16 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_211), "r"(*reinterpret_cast<const uint32_t*>(&pa_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_2[1]))
                                : "memory");
                            unsigned int pa_3[2];
                            unsigned int pb_3[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 12], vals_0[_lp*2+1 + 12]));
                                pa_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 44], vals_0[_lp*2+1 + 44]));
                                pb_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_212 = static_cast<uint32_t>(buf_0 + (unsigned int)((24 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_212), "r"(*reinterpret_cast<const uint32_t*>(&pa_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_3[1]))
                                : "memory");
                            unsigned int pa_4[2];
                            unsigned int pb_4[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 16], vals_0[_lp*2+1 + 16]));
                                pa_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 48], vals_0[_lp*2+1 + 48]));
                                pb_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_213 = static_cast<uint32_t>(buf_0 + (unsigned int)((32 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_213), "r"(*reinterpret_cast<const uint32_t*>(&pa_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_4[1]))
                                : "memory");
                            unsigned int pa_5[2];
                            unsigned int pb_5[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 20], vals_0[_lp*2+1 + 20]));
                                pa_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 52], vals_0[_lp*2+1 + 52]));
                                pb_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_214 = static_cast<uint32_t>(buf_0 + (unsigned int)((40 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_214), "r"(*reinterpret_cast<const uint32_t*>(&pa_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_5[1]))
                                : "memory");
                            unsigned int pa_6[2];
                            unsigned int pb_6[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 24], vals_0[_lp*2+1 + 24]));
                                pa_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 56], vals_0[_lp*2+1 + 56]));
                                pb_6[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_215 = static_cast<uint32_t>(buf_0 + (unsigned int)((48 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_215), "r"(*reinterpret_cast<const uint32_t*>(&pa_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_6[1]))
                                : "memory");
                            unsigned int pa_7[2];
                            unsigned int pb_7[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 28], vals_0[_lp*2+1 + 28]));
                                pa_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 60], vals_0[_lp*2+1 + 60]));
                                pb_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_216 = static_cast<uint32_t>(buf_0 + (unsigned int)((56 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_216), "r"(*reinterpret_cast<const uint32_t*>(&pa_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_7[1]))
                                : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            unsigned long long d64_0 = (unsigned long long)dhi[0] << 32ull | (unsigned long long)dlo[0];
                            if (d64_0 != 0ull) {
                                int row_0 = warp * 16 + (lane_2 & 15);
                                unsigned int src_0 = buf_0 + (unsigned int)(row_0 * 272);
                                {
                                    void* _cpbulk_dst_217 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(d64_0));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                                        :: "l"(_cpbulk_dst_217), "r"(src_0), "r"((uint32_t)(256)), "l"(0x12F0000000000000ULL)
                                        : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        } else {
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (64 < valid) {
                            unsigned int buf_1 = sm_f2_addr + (unsigned int)(1 + nit * 4 & 1) * 18432;
                            float vals_1[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[31]))
                                : "r"(taddr + (unsigned int)(warp * 32 << 16) + acc_col_1 + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&vals_1[63]))
                                : "r"(taddr + (unsigned int)(warp * 32 + 16 << 16) + acc_col_1 + 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (128 >= valid) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((ce_addr + (cst) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            float2 alpha2_1 = make_float2(alpha_1, alpha_1);
                            float2 _f32x2_mul_a_218 = make_float2(vals_1[0], vals_1[1]);
                            float2 va2_32;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_218), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[0] = va2_32.x;
                            vals_1[1] = va2_32.y;
                            float2 _f32x2_mul_a_219 = make_float2(vals_1[2], vals_1[3]);
                            float2 va2_33;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_219), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[2] = va2_33.x;
                            vals_1[3] = va2_33.y;
                            float2 _f32x2_mul_a_220 = make_float2(vals_1[4], vals_1[5]);
                            float2 va2_34;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[4] = va2_34.x;
                            vals_1[5] = va2_34.y;
                            float2 _f32x2_mul_a_221 = make_float2(vals_1[6], vals_1[7]);
                            float2 va2_35;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_221), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[6] = va2_35.x;
                            vals_1[7] = va2_35.y;
                            float2 _f32x2_mul_a_222 = make_float2(vals_1[8], vals_1[9]);
                            float2 va2_36;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_222), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[8] = va2_36.x;
                            vals_1[9] = va2_36.y;
                            float2 _f32x2_mul_a_223 = make_float2(vals_1[10], vals_1[11]);
                            float2 va2_37;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_223), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[10] = va2_37.x;
                            vals_1[11] = va2_37.y;
                            float2 _f32x2_mul_a_224 = make_float2(vals_1[12], vals_1[13]);
                            float2 va2_38;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[12] = va2_38.x;
                            vals_1[13] = va2_38.y;
                            float2 _f32x2_mul_a_225 = make_float2(vals_1[14], vals_1[15]);
                            float2 va2_39;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_225), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[14] = va2_39.x;
                            vals_1[15] = va2_39.y;
                            float2 _f32x2_mul_a_226 = make_float2(vals_1[16], vals_1[17]);
                            float2 va2_40;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[16] = va2_40.x;
                            vals_1[17] = va2_40.y;
                            float2 _f32x2_mul_a_227 = make_float2(vals_1[18], vals_1[19]);
                            float2 va2_41;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_227), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[18] = va2_41.x;
                            vals_1[19] = va2_41.y;
                            float2 _f32x2_mul_a_228 = make_float2(vals_1[20], vals_1[21]);
                            float2 va2_42;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[20] = va2_42.x;
                            vals_1[21] = va2_42.y;
                            float2 _f32x2_mul_a_229 = make_float2(vals_1[22], vals_1[23]);
                            float2 va2_43;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[22] = va2_43.x;
                            vals_1[23] = va2_43.y;
                            float2 _f32x2_mul_a_230 = make_float2(vals_1[24], vals_1[25]);
                            float2 va2_44;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_230), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[24] = va2_44.x;
                            vals_1[25] = va2_44.y;
                            float2 _f32x2_mul_a_231 = make_float2(vals_1[26], vals_1[27]);
                            float2 va2_45;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_231), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[26] = va2_45.x;
                            vals_1[27] = va2_45.y;
                            float2 _f32x2_mul_a_232 = make_float2(vals_1[28], vals_1[29]);
                            float2 va2_46;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[28] = va2_46.x;
                            vals_1[29] = va2_46.y;
                            float2 _f32x2_mul_a_233 = make_float2(vals_1[30], vals_1[31]);
                            float2 va2_47;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_233), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[30] = va2_47.x;
                            vals_1[31] = va2_47.y;
                            float2 _f32x2_mul_a_234 = make_float2(vals_1[32], vals_1[33]);
                            float2 va2_48;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[32] = va2_48.x;
                            vals_1[33] = va2_48.y;
                            float2 _f32x2_mul_a_235 = make_float2(vals_1[34], vals_1[35]);
                            float2 va2_49;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_235), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[34] = va2_49.x;
                            vals_1[35] = va2_49.y;
                            float2 _f32x2_mul_a_236 = make_float2(vals_1[36], vals_1[37]);
                            float2 va2_50;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_236), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[36] = va2_50.x;
                            vals_1[37] = va2_50.y;
                            float2 _f32x2_mul_a_237 = make_float2(vals_1[38], vals_1[39]);
                            float2 va2_51;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_237), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[38] = va2_51.x;
                            vals_1[39] = va2_51.y;
                            float2 _f32x2_mul_a_238 = make_float2(vals_1[40], vals_1[41]);
                            float2 va2_52;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[40] = va2_52.x;
                            vals_1[41] = va2_52.y;
                            float2 _f32x2_mul_a_239 = make_float2(vals_1[42], vals_1[43]);
                            float2 va2_53;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_239), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[42] = va2_53.x;
                            vals_1[43] = va2_53.y;
                            float2 _f32x2_mul_a_240 = make_float2(vals_1[44], vals_1[45]);
                            float2 va2_54;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_240), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[44] = va2_54.x;
                            vals_1[45] = va2_54.y;
                            float2 _f32x2_mul_a_241 = make_float2(vals_1[46], vals_1[47]);
                            float2 va2_55;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_241), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[46] = va2_55.x;
                            vals_1[47] = va2_55.y;
                            float2 _f32x2_mul_a_242 = make_float2(vals_1[48], vals_1[49]);
                            float2 va2_56;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_242), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[48] = va2_56.x;
                            vals_1[49] = va2_56.y;
                            float2 _f32x2_mul_a_243 = make_float2(vals_1[50], vals_1[51]);
                            float2 va2_57;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_243), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[50] = va2_57.x;
                            vals_1[51] = va2_57.y;
                            float2 _f32x2_mul_a_244 = make_float2(vals_1[52], vals_1[53]);
                            float2 va2_58;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_244), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[52] = va2_58.x;
                            vals_1[53] = va2_58.y;
                            float2 _f32x2_mul_a_245 = make_float2(vals_1[54], vals_1[55]);
                            float2 va2_59;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_245), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[54] = va2_59.x;
                            vals_1[55] = va2_59.y;
                            float2 _f32x2_mul_a_246 = make_float2(vals_1[56], vals_1[57]);
                            float2 va2_60;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_246), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[56] = va2_60.x;
                            vals_1[57] = va2_60.y;
                            float2 _f32x2_mul_a_247 = make_float2(vals_1[58], vals_1[59]);
                            float2 va2_61;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_247), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[58] = va2_61.x;
                            vals_1[59] = va2_61.y;
                            float2 _f32x2_mul_a_248 = make_float2(vals_1[60], vals_1[61]);
                            float2 va2_62;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_248), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[60] = va2_62.x;
                            vals_1[61] = va2_62.y;
                            float2 _f32x2_mul_a_249 = make_float2(vals_1[62], vals_1[63]);
                            float2 va2_63;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_249), "l"(*(const unsigned long long*)&alpha2_1));
                            vals_1[62] = va2_63.x;
                            vals_1[63] = va2_63.y;
                            unsigned int pa_8[2];
                            unsigned int pb_8[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                                pa_8[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 32], vals_1[_lp*2+1 + 32]));
                                pb_8[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_250 = static_cast<uint32_t>(buf_1 + (unsigned int)((lane_2 & 7) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_250), "r"(*reinterpret_cast<const uint32_t*>(&pa_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_8[1]))
                                : "memory");
                            unsigned int pa_9[2];
                            unsigned int pb_9[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 4], vals_1[_lp*2+1 + 4]));
                                pa_9[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 36], vals_1[_lp*2+1 + 36]));
                                pb_9[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_251 = static_cast<uint32_t>(buf_1 + (unsigned int)((8 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_251), "r"(*reinterpret_cast<const uint32_t*>(&pa_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_9[1]))
                                : "memory");
                            unsigned int pa_10[2];
                            unsigned int pb_10[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 8], vals_1[_lp*2+1 + 8]));
                                pa_10[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 40], vals_1[_lp*2+1 + 40]));
                                pb_10[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_252 = static_cast<uint32_t>(buf_1 + (unsigned int)((16 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_252), "r"(*reinterpret_cast<const uint32_t*>(&pa_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_10[1]))
                                : "memory");
                            unsigned int pa_11[2];
                            unsigned int pb_11[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 12], vals_1[_lp*2+1 + 12]));
                                pa_11[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 44], vals_1[_lp*2+1 + 44]));
                                pb_11[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_253 = static_cast<uint32_t>(buf_1 + (unsigned int)((24 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_253), "r"(*reinterpret_cast<const uint32_t*>(&pa_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_11[1]))
                                : "memory");
                            unsigned int pa_12[2];
                            unsigned int pb_12[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 16], vals_1[_lp*2+1 + 16]));
                                pa_12[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 48], vals_1[_lp*2+1 + 48]));
                                pb_12[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_254 = static_cast<uint32_t>(buf_1 + (unsigned int)((32 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_254), "r"(*reinterpret_cast<const uint32_t*>(&pa_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_12[1]))
                                : "memory");
                            unsigned int pa_13[2];
                            unsigned int pb_13[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 20], vals_1[_lp*2+1 + 20]));
                                pa_13[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 52], vals_1[_lp*2+1 + 52]));
                                pb_13[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_255 = static_cast<uint32_t>(buf_1 + (unsigned int)((40 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_255), "r"(*reinterpret_cast<const uint32_t*>(&pa_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_13[1]))
                                : "memory");
                            unsigned int pa_14[2];
                            unsigned int pb_14[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 24], vals_1[_lp*2+1 + 24]));
                                pa_14[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 56], vals_1[_lp*2+1 + 56]));
                                pb_14[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_256 = static_cast<uint32_t>(buf_1 + (unsigned int)((48 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_256), "r"(*reinterpret_cast<const uint32_t*>(&pa_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_14[1]))
                                : "memory");
                            unsigned int pa_15[2];
                            unsigned int pb_15[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 28], vals_1[_lp*2+1 + 28]));
                                pa_15[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 60], vals_1[_lp*2+1 + 60]));
                                pb_15[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_257 = static_cast<uint32_t>(buf_1 + (unsigned int)((56 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_257), "r"(*reinterpret_cast<const uint32_t*>(&pa_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_15[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_15[1]))
                                : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            unsigned long long d64_1 = (unsigned long long)dhi[1] << 32ull | (unsigned long long)dlo[1];
                            if (d64_1 != 0ull) {
                                int row_1 = warp * 16 + (lane_2 & 15);
                                unsigned int src_1 = buf_1 + (unsigned int)(row_1 * 272);
                                {
                                    void* _cpbulk_dst_258 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(d64_1));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                                        :: "l"(_cpbulk_dst_258), "r"(src_1), "r"((uint32_t)(256)), "l"(0x12F0000000000000ULL)
                                        : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        } else {
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (128 < valid) {
                            unsigned int buf_2 = sm_f2_addr + (unsigned int)(2 + nit * 4 & 1) * 18432;
                            float vals_2[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[31]))
                                : "r"(taddr + (unsigned int)(warp * 32 << 16) + acc_col_1 + 128));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&vals_2[63]))
                                : "r"(taddr + (unsigned int)(warp * 32 + 16 << 16) + acc_col_1 + 128));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (192 >= valid) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((ce_addr + (cst) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            float2 alpha2_2 = make_float2(alpha_1, alpha_1);
                            float2 _f32x2_mul_a_259 = make_float2(vals_2[0], vals_2[1]);
                            float2 va2_64;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_259), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[0] = va2_64.x;
                            vals_2[1] = va2_64.y;
                            float2 _f32x2_mul_a_260 = make_float2(vals_2[2], vals_2[3]);
                            float2 va2_65;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_65) : "l"(*(const unsigned long long*)&_f32x2_mul_a_260), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[2] = va2_65.x;
                            vals_2[3] = va2_65.y;
                            float2 _f32x2_mul_a_261 = make_float2(vals_2[4], vals_2[5]);
                            float2 va2_66;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_66) : "l"(*(const unsigned long long*)&_f32x2_mul_a_261), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[4] = va2_66.x;
                            vals_2[5] = va2_66.y;
                            float2 _f32x2_mul_a_262 = make_float2(vals_2[6], vals_2[7]);
                            float2 va2_67;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_67) : "l"(*(const unsigned long long*)&_f32x2_mul_a_262), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[6] = va2_67.x;
                            vals_2[7] = va2_67.y;
                            float2 _f32x2_mul_a_263 = make_float2(vals_2[8], vals_2[9]);
                            float2 va2_68;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_68) : "l"(*(const unsigned long long*)&_f32x2_mul_a_263), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[8] = va2_68.x;
                            vals_2[9] = va2_68.y;
                            float2 _f32x2_mul_a_264 = make_float2(vals_2[10], vals_2[11]);
                            float2 va2_69;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_69) : "l"(*(const unsigned long long*)&_f32x2_mul_a_264), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[10] = va2_69.x;
                            vals_2[11] = va2_69.y;
                            float2 _f32x2_mul_a_265 = make_float2(vals_2[12], vals_2[13]);
                            float2 va2_70;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_70) : "l"(*(const unsigned long long*)&_f32x2_mul_a_265), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[12] = va2_70.x;
                            vals_2[13] = va2_70.y;
                            float2 _f32x2_mul_a_266 = make_float2(vals_2[14], vals_2[15]);
                            float2 va2_71;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_71) : "l"(*(const unsigned long long*)&_f32x2_mul_a_266), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[14] = va2_71.x;
                            vals_2[15] = va2_71.y;
                            float2 _f32x2_mul_a_267 = make_float2(vals_2[16], vals_2[17]);
                            float2 va2_72;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_72) : "l"(*(const unsigned long long*)&_f32x2_mul_a_267), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[16] = va2_72.x;
                            vals_2[17] = va2_72.y;
                            float2 _f32x2_mul_a_268 = make_float2(vals_2[18], vals_2[19]);
                            float2 va2_73;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_73) : "l"(*(const unsigned long long*)&_f32x2_mul_a_268), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[18] = va2_73.x;
                            vals_2[19] = va2_73.y;
                            float2 _f32x2_mul_a_269 = make_float2(vals_2[20], vals_2[21]);
                            float2 va2_74;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_74) : "l"(*(const unsigned long long*)&_f32x2_mul_a_269), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[20] = va2_74.x;
                            vals_2[21] = va2_74.y;
                            float2 _f32x2_mul_a_270 = make_float2(vals_2[22], vals_2[23]);
                            float2 va2_75;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_75) : "l"(*(const unsigned long long*)&_f32x2_mul_a_270), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[22] = va2_75.x;
                            vals_2[23] = va2_75.y;
                            float2 _f32x2_mul_a_271 = make_float2(vals_2[24], vals_2[25]);
                            float2 va2_76;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_76) : "l"(*(const unsigned long long*)&_f32x2_mul_a_271), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[24] = va2_76.x;
                            vals_2[25] = va2_76.y;
                            float2 _f32x2_mul_a_272 = make_float2(vals_2[26], vals_2[27]);
                            float2 va2_77;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_77) : "l"(*(const unsigned long long*)&_f32x2_mul_a_272), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[26] = va2_77.x;
                            vals_2[27] = va2_77.y;
                            float2 _f32x2_mul_a_273 = make_float2(vals_2[28], vals_2[29]);
                            float2 va2_78;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_78) : "l"(*(const unsigned long long*)&_f32x2_mul_a_273), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[28] = va2_78.x;
                            vals_2[29] = va2_78.y;
                            float2 _f32x2_mul_a_274 = make_float2(vals_2[30], vals_2[31]);
                            float2 va2_79;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_79) : "l"(*(const unsigned long long*)&_f32x2_mul_a_274), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[30] = va2_79.x;
                            vals_2[31] = va2_79.y;
                            float2 _f32x2_mul_a_275 = make_float2(vals_2[32], vals_2[33]);
                            float2 va2_80;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_80) : "l"(*(const unsigned long long*)&_f32x2_mul_a_275), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[32] = va2_80.x;
                            vals_2[33] = va2_80.y;
                            float2 _f32x2_mul_a_276 = make_float2(vals_2[34], vals_2[35]);
                            float2 va2_81;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_81) : "l"(*(const unsigned long long*)&_f32x2_mul_a_276), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[34] = va2_81.x;
                            vals_2[35] = va2_81.y;
                            float2 _f32x2_mul_a_277 = make_float2(vals_2[36], vals_2[37]);
                            float2 va2_82;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_82) : "l"(*(const unsigned long long*)&_f32x2_mul_a_277), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[36] = va2_82.x;
                            vals_2[37] = va2_82.y;
                            float2 _f32x2_mul_a_278 = make_float2(vals_2[38], vals_2[39]);
                            float2 va2_83;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_83) : "l"(*(const unsigned long long*)&_f32x2_mul_a_278), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[38] = va2_83.x;
                            vals_2[39] = va2_83.y;
                            float2 _f32x2_mul_a_279 = make_float2(vals_2[40], vals_2[41]);
                            float2 va2_84;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_279), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[40] = va2_84.x;
                            vals_2[41] = va2_84.y;
                            float2 _f32x2_mul_a_280 = make_float2(vals_2[42], vals_2[43]);
                            float2 va2_85;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_85) : "l"(*(const unsigned long long*)&_f32x2_mul_a_280), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[42] = va2_85.x;
                            vals_2[43] = va2_85.y;
                            float2 _f32x2_mul_a_281 = make_float2(vals_2[44], vals_2[45]);
                            float2 va2_86;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_281), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[44] = va2_86.x;
                            vals_2[45] = va2_86.y;
                            float2 _f32x2_mul_a_282 = make_float2(vals_2[46], vals_2[47]);
                            float2 va2_87;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_87) : "l"(*(const unsigned long long*)&_f32x2_mul_a_282), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[46] = va2_87.x;
                            vals_2[47] = va2_87.y;
                            float2 _f32x2_mul_a_283 = make_float2(vals_2[48], vals_2[49]);
                            float2 va2_88;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_283), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[48] = va2_88.x;
                            vals_2[49] = va2_88.y;
                            float2 _f32x2_mul_a_284 = make_float2(vals_2[50], vals_2[51]);
                            float2 va2_89;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_284), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[50] = va2_89.x;
                            vals_2[51] = va2_89.y;
                            float2 _f32x2_mul_a_285 = make_float2(vals_2[52], vals_2[53]);
                            float2 va2_90;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_90) : "l"(*(const unsigned long long*)&_f32x2_mul_a_285), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[52] = va2_90.x;
                            vals_2[53] = va2_90.y;
                            float2 _f32x2_mul_a_286 = make_float2(vals_2[54], vals_2[55]);
                            float2 va2_91;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_91) : "l"(*(const unsigned long long*)&_f32x2_mul_a_286), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[54] = va2_91.x;
                            vals_2[55] = va2_91.y;
                            float2 _f32x2_mul_a_287 = make_float2(vals_2[56], vals_2[57]);
                            float2 va2_92;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_92) : "l"(*(const unsigned long long*)&_f32x2_mul_a_287), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[56] = va2_92.x;
                            vals_2[57] = va2_92.y;
                            float2 _f32x2_mul_a_288 = make_float2(vals_2[58], vals_2[59]);
                            float2 va2_93;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_93) : "l"(*(const unsigned long long*)&_f32x2_mul_a_288), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[58] = va2_93.x;
                            vals_2[59] = va2_93.y;
                            float2 _f32x2_mul_a_289 = make_float2(vals_2[60], vals_2[61]);
                            float2 va2_94;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_94) : "l"(*(const unsigned long long*)&_f32x2_mul_a_289), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[60] = va2_94.x;
                            vals_2[61] = va2_94.y;
                            float2 _f32x2_mul_a_290 = make_float2(vals_2[62], vals_2[63]);
                            float2 va2_95;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_95) : "l"(*(const unsigned long long*)&_f32x2_mul_a_290), "l"(*(const unsigned long long*)&alpha2_2));
                            vals_2[62] = va2_95.x;
                            vals_2[63] = va2_95.y;
                            unsigned int pa_16[2];
                            unsigned int pb_16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 0], vals_2[_lp*2+1 + 0]));
                                pa_16[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 32], vals_2[_lp*2+1 + 32]));
                                pb_16[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_291 = static_cast<uint32_t>(buf_2 + (unsigned int)((lane_2 & 7) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_291), "r"(*reinterpret_cast<const uint32_t*>(&pa_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_16[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_16[1]))
                                : "memory");
                            unsigned int pa_17[2];
                            unsigned int pb_17[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 4], vals_2[_lp*2+1 + 4]));
                                pa_17[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 36], vals_2[_lp*2+1 + 36]));
                                pb_17[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_292 = static_cast<uint32_t>(buf_2 + (unsigned int)((8 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_292), "r"(*reinterpret_cast<const uint32_t*>(&pa_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_17[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_17[1]))
                                : "memory");
                            unsigned int pa_18[2];
                            unsigned int pb_18[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 8], vals_2[_lp*2+1 + 8]));
                                pa_18[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 40], vals_2[_lp*2+1 + 40]));
                                pb_18[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_293 = static_cast<uint32_t>(buf_2 + (unsigned int)((16 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_293), "r"(*reinterpret_cast<const uint32_t*>(&pa_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_18[1]))
                                : "memory");
                            unsigned int pa_19[2];
                            unsigned int pb_19[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 12], vals_2[_lp*2+1 + 12]));
                                pa_19[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 44], vals_2[_lp*2+1 + 44]));
                                pb_19[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_294 = static_cast<uint32_t>(buf_2 + (unsigned int)((24 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_294), "r"(*reinterpret_cast<const uint32_t*>(&pa_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_19[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_19[1]))
                                : "memory");
                            unsigned int pa_20[2];
                            unsigned int pb_20[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 16], vals_2[_lp*2+1 + 16]));
                                pa_20[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 48], vals_2[_lp*2+1 + 48]));
                                pb_20[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_295 = static_cast<uint32_t>(buf_2 + (unsigned int)((32 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_295), "r"(*reinterpret_cast<const uint32_t*>(&pa_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_20[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_20[1]))
                                : "memory");
                            unsigned int pa_21[2];
                            unsigned int pb_21[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 20], vals_2[_lp*2+1 + 20]));
                                pa_21[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 52], vals_2[_lp*2+1 + 52]));
                                pb_21[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_296 = static_cast<uint32_t>(buf_2 + (unsigned int)((40 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_296), "r"(*reinterpret_cast<const uint32_t*>(&pa_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_21[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_21[1]))
                                : "memory");
                            unsigned int pa_22[2];
                            unsigned int pb_22[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 24], vals_2[_lp*2+1 + 24]));
                                pa_22[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 56], vals_2[_lp*2+1 + 56]));
                                pb_22[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_297 = static_cast<uint32_t>(buf_2 + (unsigned int)((48 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_297), "r"(*reinterpret_cast<const uint32_t*>(&pa_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_22[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_22[1]))
                                : "memory");
                            unsigned int pa_23[2];
                            unsigned int pb_23[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 28], vals_2[_lp*2+1 + 28]));
                                pa_23[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 60], vals_2[_lp*2+1 + 60]));
                                pb_23[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_298 = static_cast<uint32_t>(buf_2 + (unsigned int)((56 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_298), "r"(*reinterpret_cast<const uint32_t*>(&pa_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_23[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_23[1]))
                                : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            unsigned long long d64_2 = (unsigned long long)dhi[2] << 32ull | (unsigned long long)dlo[2];
                            if (d64_2 != 0ull) {
                                int row_2 = warp * 16 + (lane_2 & 15);
                                unsigned int src_2 = buf_2 + (unsigned int)(row_2 * 272);
                                {
                                    void* _cpbulk_dst_299 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(d64_2));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                                        :: "l"(_cpbulk_dst_299), "r"(src_2), "r"((uint32_t)(256)), "l"(0x12F0000000000000ULL)
                                        : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        } else {
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (192 < valid) {
                            unsigned int buf_3 = sm_f2_addr + (unsigned int)(3 + nit * 4 & 1) * 18432;
                            float vals_3[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[31]))
                                : "r"(taddr + (unsigned int)(warp * 32 << 16) + acc_col_1 + 192));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&vals_3[32])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[33])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[34])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[35])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[36])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[37])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[38])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[39])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[40])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[41])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[42])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[43])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[44])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[45])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[46])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[47])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[48])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[49])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[50])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[51])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[52])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[53])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[54])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[55])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[56])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[57])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[58])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[59])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[60])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[61])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[62])), "=r"(*reinterpret_cast<uint32_t*>(&vals_3[63]))
                                : "r"(taddr + (unsigned int)(warp * 32 + 16 << 16) + acc_col_1 + 192));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (256 >= valid) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((ce_addr + (cst) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            float2 alpha2_3 = make_float2(alpha_1, alpha_1);
                            float2 _f32x2_mul_a_300 = make_float2(vals_3[0], vals_3[1]);
                            float2 va2_96;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_96) : "l"(*(const unsigned long long*)&_f32x2_mul_a_300), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[0] = va2_96.x;
                            vals_3[1] = va2_96.y;
                            float2 _f32x2_mul_a_301 = make_float2(vals_3[2], vals_3[3]);
                            float2 va2_97;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_97) : "l"(*(const unsigned long long*)&_f32x2_mul_a_301), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[2] = va2_97.x;
                            vals_3[3] = va2_97.y;
                            float2 _f32x2_mul_a_302 = make_float2(vals_3[4], vals_3[5]);
                            float2 va2_98;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_98) : "l"(*(const unsigned long long*)&_f32x2_mul_a_302), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[4] = va2_98.x;
                            vals_3[5] = va2_98.y;
                            float2 _f32x2_mul_a_303 = make_float2(vals_3[6], vals_3[7]);
                            float2 va2_99;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_99) : "l"(*(const unsigned long long*)&_f32x2_mul_a_303), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[6] = va2_99.x;
                            vals_3[7] = va2_99.y;
                            float2 _f32x2_mul_a_304 = make_float2(vals_3[8], vals_3[9]);
                            float2 va2_100;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_100) : "l"(*(const unsigned long long*)&_f32x2_mul_a_304), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[8] = va2_100.x;
                            vals_3[9] = va2_100.y;
                            float2 _f32x2_mul_a_305 = make_float2(vals_3[10], vals_3[11]);
                            float2 va2_101;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_101) : "l"(*(const unsigned long long*)&_f32x2_mul_a_305), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[10] = va2_101.x;
                            vals_3[11] = va2_101.y;
                            float2 _f32x2_mul_a_306 = make_float2(vals_3[12], vals_3[13]);
                            float2 va2_102;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_102) : "l"(*(const unsigned long long*)&_f32x2_mul_a_306), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[12] = va2_102.x;
                            vals_3[13] = va2_102.y;
                            float2 _f32x2_mul_a_307 = make_float2(vals_3[14], vals_3[15]);
                            float2 va2_103;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_103) : "l"(*(const unsigned long long*)&_f32x2_mul_a_307), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[14] = va2_103.x;
                            vals_3[15] = va2_103.y;
                            float2 _f32x2_mul_a_308 = make_float2(vals_3[16], vals_3[17]);
                            float2 va2_104;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_104) : "l"(*(const unsigned long long*)&_f32x2_mul_a_308), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[16] = va2_104.x;
                            vals_3[17] = va2_104.y;
                            float2 _f32x2_mul_a_309 = make_float2(vals_3[18], vals_3[19]);
                            float2 va2_105;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_309), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[18] = va2_105.x;
                            vals_3[19] = va2_105.y;
                            float2 _f32x2_mul_a_310 = make_float2(vals_3[20], vals_3[21]);
                            float2 va2_106;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_106) : "l"(*(const unsigned long long*)&_f32x2_mul_a_310), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[20] = va2_106.x;
                            vals_3[21] = va2_106.y;
                            float2 _f32x2_mul_a_311 = make_float2(vals_3[22], vals_3[23]);
                            float2 va2_107;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_107) : "l"(*(const unsigned long long*)&_f32x2_mul_a_311), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[22] = va2_107.x;
                            vals_3[23] = va2_107.y;
                            float2 _f32x2_mul_a_312 = make_float2(vals_3[24], vals_3[25]);
                            float2 va2_108;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_108) : "l"(*(const unsigned long long*)&_f32x2_mul_a_312), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[24] = va2_108.x;
                            vals_3[25] = va2_108.y;
                            float2 _f32x2_mul_a_313 = make_float2(vals_3[26], vals_3[27]);
                            float2 va2_109;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_109) : "l"(*(const unsigned long long*)&_f32x2_mul_a_313), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[26] = va2_109.x;
                            vals_3[27] = va2_109.y;
                            float2 _f32x2_mul_a_314 = make_float2(vals_3[28], vals_3[29]);
                            float2 va2_110;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_110) : "l"(*(const unsigned long long*)&_f32x2_mul_a_314), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[28] = va2_110.x;
                            vals_3[29] = va2_110.y;
                            float2 _f32x2_mul_a_315 = make_float2(vals_3[30], vals_3[31]);
                            float2 va2_111;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_111) : "l"(*(const unsigned long long*)&_f32x2_mul_a_315), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[30] = va2_111.x;
                            vals_3[31] = va2_111.y;
                            float2 _f32x2_mul_a_316 = make_float2(vals_3[32], vals_3[33]);
                            float2 va2_112;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_112) : "l"(*(const unsigned long long*)&_f32x2_mul_a_316), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[32] = va2_112.x;
                            vals_3[33] = va2_112.y;
                            float2 _f32x2_mul_a_317 = make_float2(vals_3[34], vals_3[35]);
                            float2 va2_113;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_113) : "l"(*(const unsigned long long*)&_f32x2_mul_a_317), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[34] = va2_113.x;
                            vals_3[35] = va2_113.y;
                            float2 _f32x2_mul_a_318 = make_float2(vals_3[36], vals_3[37]);
                            float2 va2_114;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_114) : "l"(*(const unsigned long long*)&_f32x2_mul_a_318), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[36] = va2_114.x;
                            vals_3[37] = va2_114.y;
                            float2 _f32x2_mul_a_319 = make_float2(vals_3[38], vals_3[39]);
                            float2 va2_115;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_115) : "l"(*(const unsigned long long*)&_f32x2_mul_a_319), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[38] = va2_115.x;
                            vals_3[39] = va2_115.y;
                            float2 _f32x2_mul_a_320 = make_float2(vals_3[40], vals_3[41]);
                            float2 va2_116;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_116) : "l"(*(const unsigned long long*)&_f32x2_mul_a_320), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[40] = va2_116.x;
                            vals_3[41] = va2_116.y;
                            float2 _f32x2_mul_a_321 = make_float2(vals_3[42], vals_3[43]);
                            float2 va2_117;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_117) : "l"(*(const unsigned long long*)&_f32x2_mul_a_321), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[42] = va2_117.x;
                            vals_3[43] = va2_117.y;
                            float2 _f32x2_mul_a_322 = make_float2(vals_3[44], vals_3[45]);
                            float2 va2_118;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_118) : "l"(*(const unsigned long long*)&_f32x2_mul_a_322), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[44] = va2_118.x;
                            vals_3[45] = va2_118.y;
                            float2 _f32x2_mul_a_323 = make_float2(vals_3[46], vals_3[47]);
                            float2 va2_119;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_119) : "l"(*(const unsigned long long*)&_f32x2_mul_a_323), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[46] = va2_119.x;
                            vals_3[47] = va2_119.y;
                            float2 _f32x2_mul_a_324 = make_float2(vals_3[48], vals_3[49]);
                            float2 va2_120;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_120) : "l"(*(const unsigned long long*)&_f32x2_mul_a_324), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[48] = va2_120.x;
                            vals_3[49] = va2_120.y;
                            float2 _f32x2_mul_a_325 = make_float2(vals_3[50], vals_3[51]);
                            float2 va2_121;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_121) : "l"(*(const unsigned long long*)&_f32x2_mul_a_325), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[50] = va2_121.x;
                            vals_3[51] = va2_121.y;
                            float2 _f32x2_mul_a_326 = make_float2(vals_3[52], vals_3[53]);
                            float2 va2_122;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_122) : "l"(*(const unsigned long long*)&_f32x2_mul_a_326), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[52] = va2_122.x;
                            vals_3[53] = va2_122.y;
                            float2 _f32x2_mul_a_327 = make_float2(vals_3[54], vals_3[55]);
                            float2 va2_123;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_123) : "l"(*(const unsigned long long*)&_f32x2_mul_a_327), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[54] = va2_123.x;
                            vals_3[55] = va2_123.y;
                            float2 _f32x2_mul_a_328 = make_float2(vals_3[56], vals_3[57]);
                            float2 va2_124;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_124) : "l"(*(const unsigned long long*)&_f32x2_mul_a_328), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[56] = va2_124.x;
                            vals_3[57] = va2_124.y;
                            float2 _f32x2_mul_a_329 = make_float2(vals_3[58], vals_3[59]);
                            float2 va2_125;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_125) : "l"(*(const unsigned long long*)&_f32x2_mul_a_329), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[58] = va2_125.x;
                            vals_3[59] = va2_125.y;
                            float2 _f32x2_mul_a_330 = make_float2(vals_3[60], vals_3[61]);
                            float2 va2_126;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_126) : "l"(*(const unsigned long long*)&_f32x2_mul_a_330), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[60] = va2_126.x;
                            vals_3[61] = va2_126.y;
                            float2 _f32x2_mul_a_331 = make_float2(vals_3[62], vals_3[63]);
                            float2 va2_127;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&va2_127) : "l"(*(const unsigned long long*)&_f32x2_mul_a_331), "l"(*(const unsigned long long*)&alpha2_3));
                            vals_3[62] = va2_127.x;
                            vals_3[63] = va2_127.y;
                            unsigned int pa_24[2];
                            unsigned int pb_24[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 0], vals_3[_lp*2+1 + 0]));
                                pa_24[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 32], vals_3[_lp*2+1 + 32]));
                                pb_24[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_332 = static_cast<uint32_t>(buf_3 + (unsigned int)((lane_2 & 7) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_332), "r"(*reinterpret_cast<const uint32_t*>(&pa_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_24[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_24[1]))
                                : "memory");
                            unsigned int pa_25[2];
                            unsigned int pb_25[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 4], vals_3[_lp*2+1 + 4]));
                                pa_25[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 36], vals_3[_lp*2+1 + 36]));
                                pb_25[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_333 = static_cast<uint32_t>(buf_3 + (unsigned int)((8 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_333), "r"(*reinterpret_cast<const uint32_t*>(&pa_25[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_25[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_25[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_25[1]))
                                : "memory");
                            unsigned int pa_26[2];
                            unsigned int pb_26[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 8], vals_3[_lp*2+1 + 8]));
                                pa_26[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 40], vals_3[_lp*2+1 + 40]));
                                pb_26[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_334 = static_cast<uint32_t>(buf_3 + (unsigned int)((16 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_334), "r"(*reinterpret_cast<const uint32_t*>(&pa_26[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_26[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_26[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_26[1]))
                                : "memory");
                            unsigned int pa_27[2];
                            unsigned int pb_27[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 12], vals_3[_lp*2+1 + 12]));
                                pa_27[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 44], vals_3[_lp*2+1 + 44]));
                                pb_27[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_335 = static_cast<uint32_t>(buf_3 + (unsigned int)((24 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_335), "r"(*reinterpret_cast<const uint32_t*>(&pa_27[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_27[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_27[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_27[1]))
                                : "memory");
                            unsigned int pa_28[2];
                            unsigned int pb_28[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 16], vals_3[_lp*2+1 + 16]));
                                pa_28[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 48], vals_3[_lp*2+1 + 48]));
                                pb_28[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_336 = static_cast<uint32_t>(buf_3 + (unsigned int)((32 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_336), "r"(*reinterpret_cast<const uint32_t*>(&pa_28[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_28[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_28[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_28[1]))
                                : "memory");
                            unsigned int pa_29[2];
                            unsigned int pb_29[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 20], vals_3[_lp*2+1 + 20]));
                                pa_29[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 52], vals_3[_lp*2+1 + 52]));
                                pb_29[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_337 = static_cast<uint32_t>(buf_3 + (unsigned int)((40 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_337), "r"(*reinterpret_cast<const uint32_t*>(&pa_29[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_29[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_29[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_29[1]))
                                : "memory");
                            unsigned int pa_30[2];
                            unsigned int pb_30[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 24], vals_3[_lp*2+1 + 24]));
                                pa_30[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 56], vals_3[_lp*2+1 + 56]));
                                pb_30[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_338 = static_cast<uint32_t>(buf_3 + (unsigned int)((48 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_338), "r"(*reinterpret_cast<const uint32_t*>(&pa_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_30[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_30[1]))
                                : "memory");
                            unsigned int pa_31[2];
                            unsigned int pb_31[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 28], vals_3[_lp*2+1 + 28]));
                                pa_31[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 60], vals_3[_lp*2+1 + 60]));
                                pb_31[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_339 = static_cast<uint32_t>(buf_3 + (unsigned int)((56 + (lane_2 & 7)) * 272 + (warp * 32 + (lane_2 >> 3) * 8) * 2));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_339), "r"(*reinterpret_cast<const uint32_t*>(&pa_31[0])), "r"(*reinterpret_cast<const uint32_t*>(&pa_31[1])), "r"(*reinterpret_cast<const uint32_t*>(&pb_31[0])), "r"(*reinterpret_cast<const uint32_t*>(&pb_31[1]))
                                : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            unsigned long long d64_3 = (unsigned long long)dhi[3] << 32ull | (unsigned long long)dlo[3];
                            if (d64_3 != 0ull) {
                                int row_3 = warp * 16 + (lane_2 & 15);
                                unsigned int src_3 = buf_3 + (unsigned int)(row_3 * 272);
                                {
                                    void* _cpbulk_dst_340 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(d64_3));
                                    asm volatile(
                                        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                                        :: "l"(_cpbulk_dst_340), "r"(src_3), "r"((uint32_t)(256)), "l"(0x12F0000000000000ULL)
                                        : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        } else {
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        cst += 1;
                        if (cst == 2) { cst = 0; _phase_cf ^= 1; }
                        fpend[0] = bb + block;
                        fpend[1] = fpend[1] + 1;
                    }
                    nit = nit + 1;
                }
            }
            if (fpend[0] >= 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
                asm volatile("fence.proxy.async.global;" ::: "memory");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc2_done) + (fpend[0]))), "r"(static_cast<unsigned int>((unsigned int)fpend[1])) : "memory");
                    }
                }
                fpend[0] = -1;
                fpend[1] = 0;
            }
            asm volatile("cp.async.bulk.wait_group 0;");
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            if (warp == 0) {
                int me = mega_pg_rank;
                int fi = 0;
                int fw = 0;
                int nf = 0;
                int nw = 0;
                float dacc[32];
                int cd_copy = metadata_ready[0] & 1;
                int nxt = 0;
                int got = 0;
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_341 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                    got = (int)_atomic_fetch_add_341;
                }
                int _shfl_sync_342 = __shfl_sync(0xFFFFFFFF, got, 0);
                nxt = _shfl_sync_342;
                int dnext = nxt;
                int dprefix = 0;
                int cbegin = 0;
                int cblk = 0;
                int bprefix = 0;
                int tblocks = 0;
                tblocks = tblocks + (expert_counts[0] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[1] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[2] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[3] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[4] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[5] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[6] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[7] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[8] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[9] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[10] + 256 - 1 >> 8);
                tblocks = tblocks + (expert_counts[11] + 256 - 1 >> 8);
                #pragma unroll 1
                for (int ce_ = 0; ce_ < 12; ce_++) {
                    int ccount = expert_counts[ce_];
                    int cend = cbegin + ccount;
                    int cbase = pool_expert_base[ce_];
                    while (dnext < cend) {
                        long long grec = token_src_metadata[(cbase + (dnext - cbegin) >> 4 << 5) + (cd_copy << 4) + (cbase + (dnext - cbegin) & 15)];
                        if ((unsigned int)(grec & 4294967295ll) == 2147483648u || (unsigned int)((unsigned long long)grec >> 32ull) == 2147483648u) {
                            uint32_t both[2];
                            do {
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both[0]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase + (dnext - cbegin) >> 4 << 5) + (cd_copy << 4) + (cbase + (dnext - cbegin) & 15)))) : "memory");
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both[1]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase + (dnext - cbegin) >> 4 << 5) + (cd_copy << 4) + (cbase + (dnext - cbegin) & 15))) + 1) : "memory");
                            } while (
                                (both[0] == 0x80000000u) || (both[1] == 0x80000000u)
                            );
                            grec = (long long)((unsigned long long)(unsigned int)both[1] << 32ull | (unsigned long long)(unsigned int)both[0]);
                        }
                        int dlo_1 = (int)(grec & 4294967295ll);
                        int dhi_1 = (int)(grec >> 32ll);
                        int dcn = (dhi_1 >> 8 & 255) + 1;
                        int dwork = 0;
                        if (dcn > 1) {
                            if ((dhi_1 & 255) == dcn - 1) {
                                dwork = 1;
                            }
                        }
                        if (dwork != 0) {
                            unsigned int dsleep = 500;
                            int gblock = cblk + (dnext - cbegin >> 8);
                            while (bprefix <= gblock) {
                                int bbase = bprefix & -32;
                                unsigned int bfirst = 32;
                                int pb = bbase + lane;
                                if (pb < tblocks) {
                                    unsigned int _bval;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_bval) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pb)) : "memory");
                                    if ((unsigned int)_bval < 56) {
                                        bfirst = (unsigned int)lane;
                                    }
                                }
                                unsigned int bend_lane;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(bend_lane) : "r"(bfirst));
                                int _max_343 = ((bprefix) > (bbase + (int)bend_lane) ? (bprefix) : (bbase + (int)bend_lane));
                                bprefix = _max_343;
                                if (bprefix <= gblock) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep));
                                    unsigned int _min_344 = ((dsleep * 2) < (1000) ? (dsleep * 2) : (1000));
                                    dsleep = _min_344;
                                }
                            }
                            while (dprefix <= -1) {
                                int pbase = dprefix & -32;
                                int pex = pbase + lane;
                                unsigned int pfirst = 32;
                                if (pex < 12) {
                                    int pcount = expert_counts[pex];
                                    unsigned int _pval;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_pval) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pex)) : "memory");
                                    if ((unsigned int)_pval < (unsigned int)((pcount + 256 - 1 >> 8) * 56)) {
                                        pfirst = (unsigned int)lane;
                                    }
                                }
                                unsigned int pend_lane;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(pend_lane) : "r"(pfirst));
                                int _max_345 = ((dprefix) > (pbase + (int)pend_lane) ? (dprefix) : (pbase + (int)pend_lane));
                                dprefix = _max_345;
                                if (dprefix <= ce_) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep));
                                    unsigned int _min_346 = ((dsleep * 2) < (1000) ? (dsleep * 2) : (1000));
                                    dsleep = _min_346;
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            int dsrc = cake_floor_mod<int>(dlo_1 >> 24 & 255, mega_pg_world);
                            int dtok = dlo_1 & 65535;
                            unsigned long long dblock = (unsigned long long)dedup_rows + (unsigned long long)(dsrc * dedup_cap + dtok) * 86016ull;
                            int dpeer = dsrc;
                            unsigned long long ddst = (unsigned long long)pre_reduced_peers.ptrs[dpeer] + (unsigned long long)((dtok * 8 + me) * 7168 * 2);
                            int dper = dcn + 1 >> 1;
                            int dfills = 7 * dper;
                            __syncwarp();
                            int d0n = ((dcn) < (2) ? (dcn) : (2));
                            int dpair = dsrc * dedup_cap + dtok;
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive_expect_tx(efill_bar_0_addr + ((int)((unsigned int)fi % 2)) * 8, d0n * 2048);
                                asm volatile(
                                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1], %2, [%3], %4;"
                                    :: "r"(sm_epi_addr + 0 + (unsigned int)((nf & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock)), "r"((unsigned)(d0n * 2048)), "r"(efill_bar_0_addr + ((int)((unsigned int)fi % 2)) * 8),
                                       "l"(0x12F0000000000000ULL) : "memory");
                            }
                            fi += 1;
                            nf = nf + 1;
                            #pragma unroll 1
                            for (int f = 0; f < dfills; f++) {
                                int fc = f / dper;
                                int fj = f - fc * dper;
                                int fn = ((dcn - fj * 2) < (2) ? (dcn - fj * 2) : (2));
                                int fill_live = 0;
                                if (f + 1 < dfills) {
                                    fill_live = 1;
                                }
                                int gc_ = (f + 1) / dper;
                                int gj = f + 1 - gc_ * dper;
                                int _min_347 = ((dcn - gj * 2) < (2) ? (dcn - gj * 2) : (2));
                                int gn = _min_347 * fill_live;
                                if (elect_sync()) {
                                    if (fill_live != 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    }
                                    mbarrier_arrive_expect_tx(efill_bar_0_addr + ((int)((unsigned int)fi % 2)) * 8, gn * 2048);
                                    if (fill_live != 0) {
                                        asm volatile(
                                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1], %2, [%3], %4;"
                                            :: "r"(sm_epi_addr + 0 + (unsigned int)((nf & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock + (unsigned long long)((gc_ * 6 + gj * 2) * 2048))), "r"((unsigned)(gn * 2048)), "r"(efill_bar_0_addr + ((int)((unsigned int)fi % 2)) * 8),
                                               "l"(0x12F0000000000000ULL) : "memory");
                                    }
                                }
                                fi += 1;
                                nf = nf + 1;
                                if (fj == 0) {
                                    dacc[0] = 0.0f;
                                    dacc[1] = 0.0f;
                                    dacc[2] = 0.0f;
                                    dacc[3] = 0.0f;
                                    dacc[4] = 0.0f;
                                    dacc[5] = 0.0f;
                                    dacc[6] = 0.0f;
                                    dacc[7] = 0.0f;
                                    dacc[8] = 0.0f;
                                    dacc[9] = 0.0f;
                                    dacc[10] = 0.0f;
                                    dacc[11] = 0.0f;
                                    dacc[12] = 0.0f;
                                    dacc[13] = 0.0f;
                                    dacc[14] = 0.0f;
                                    dacc[15] = 0.0f;
                                    dacc[16] = 0.0f;
                                    dacc[17] = 0.0f;
                                    dacc[18] = 0.0f;
                                    dacc[19] = 0.0f;
                                    dacc[20] = 0.0f;
                                    dacc[21] = 0.0f;
                                    dacc[22] = 0.0f;
                                    dacc[23] = 0.0f;
                                    dacc[24] = 0.0f;
                                    dacc[25] = 0.0f;
                                    dacc[26] = 0.0f;
                                    dacc[27] = 0.0f;
                                    dacc[28] = 0.0f;
                                    dacc[29] = 0.0f;
                                    dacc[30] = 0.0f;
                                    dacc[31] = 0.0f;
                                }
                                int _stage_efill_bar_0 = (int)((unsigned int)fw % 2);
                                int _parity_efill_bar_0 = (int)((unsigned int)fw / 2 % 2);
                                mbarrier_wait(efill_bar_0_addr + (_stage_efill_bar_0) * 8, _parity_efill_bar_0);
                                unsigned int dbuf = sm_epi_addr + 0 + (unsigned int)((nw & 1) * 4096);
                                if (0 < fn) {
                                    unsigned int dw__0[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__0[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__0[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)(lane * 16)));
                                    float dsum_0[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_0[0]), "=&f"(dsum_0[1]) : "r"(dw__0[0]), "f"(dacc[0]), "f"(dacc[1]));
                                    dacc[0] = dsum_0[0];
                                    dacc[1] = dsum_0[1];
                                    float dsum_1[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_1[0]), "=&f"(dsum_1[1]) : "r"(dw__0[1]), "f"(dacc[2]), "f"(dacc[3]));
                                    dacc[2] = dsum_1[0];
                                    dacc[3] = dsum_1[1];
                                    float dsum_2[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_2[0]), "=&f"(dsum_2[1]) : "r"(dw__0[2]), "f"(dacc[4]), "f"(dacc[5]));
                                    dacc[4] = dsum_2[0];
                                    dacc[5] = dsum_2[1];
                                    float dsum_3[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_3[0]), "=&f"(dsum_3[1]) : "r"(dw__0[3]), "f"(dacc[6]), "f"(dacc[7]));
                                    dacc[6] = dsum_3[0];
                                    dacc[7] = dsum_3[1];
                                    unsigned int dw__1[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__1[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__1[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)((32 + lane) * 16)));
                                    float dsum_4[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_4[0]), "=&f"(dsum_4[1]) : "r"(dw__1[0]), "f"(dacc[8]), "f"(dacc[9]));
                                    dacc[8] = dsum_4[0];
                                    dacc[9] = dsum_4[1];
                                    float dsum_5[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_5[0]), "=&f"(dsum_5[1]) : "r"(dw__1[1]), "f"(dacc[10]), "f"(dacc[11]));
                                    dacc[10] = dsum_5[0];
                                    dacc[11] = dsum_5[1];
                                    float dsum_6[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_6[0]), "=&f"(dsum_6[1]) : "r"(dw__1[2]), "f"(dacc[12]), "f"(dacc[13]));
                                    dacc[12] = dsum_6[0];
                                    dacc[13] = dsum_6[1];
                                    float dsum_7[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_7[0]), "=&f"(dsum_7[1]) : "r"(dw__1[3]), "f"(dacc[14]), "f"(dacc[15]));
                                    dacc[14] = dsum_7[0];
                                    dacc[15] = dsum_7[1];
                                    unsigned int dw__2[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__2[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__2[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)((64 + lane) * 16)));
                                    float dsum_8[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_8[0]), "=&f"(dsum_8[1]) : "r"(dw__2[0]), "f"(dacc[16]), "f"(dacc[17]));
                                    dacc[16] = dsum_8[0];
                                    dacc[17] = dsum_8[1];
                                    float dsum_9[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_9[0]), "=&f"(dsum_9[1]) : "r"(dw__2[1]), "f"(dacc[18]), "f"(dacc[19]));
                                    dacc[18] = dsum_9[0];
                                    dacc[19] = dsum_9[1];
                                    float dsum_10[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_10[0]), "=&f"(dsum_10[1]) : "r"(dw__2[2]), "f"(dacc[20]), "f"(dacc[21]));
                                    dacc[20] = dsum_10[0];
                                    dacc[21] = dsum_10[1];
                                    float dsum_11[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_11[0]), "=&f"(dsum_11[1]) : "r"(dw__2[3]), "f"(dacc[22]), "f"(dacc[23]));
                                    dacc[22] = dsum_11[0];
                                    dacc[23] = dsum_11[1];
                                    unsigned int dw__3[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__3[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__3[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)((96 + lane) * 16)));
                                    float dsum_12[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_12[0]), "=&f"(dsum_12[1]) : "r"(dw__3[0]), "f"(dacc[24]), "f"(dacc[25]));
                                    dacc[24] = dsum_12[0];
                                    dacc[25] = dsum_12[1];
                                    float dsum_13[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_13[0]), "=&f"(dsum_13[1]) : "r"(dw__3[1]), "f"(dacc[26]), "f"(dacc[27]));
                                    dacc[26] = dsum_13[0];
                                    dacc[27] = dsum_13[1];
                                    float dsum_14[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_14[0]), "=&f"(dsum_14[1]) : "r"(dw__3[2]), "f"(dacc[28]), "f"(dacc[29]));
                                    dacc[28] = dsum_14[0];
                                    dacc[29] = dsum_14[1];
                                    float dsum_15[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_15[0]), "=&f"(dsum_15[1]) : "r"(dw__3[3]), "f"(dacc[30]), "f"(dacc[31]));
                                    dacc[30] = dsum_15[0];
                                    dacc[31] = dsum_15[1];
                                }
                                if (1 < fn) {
                                    unsigned int dw__4[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__4[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__4[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)(2048 + lane * 16)));
                                    float dsum_16[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_16[0]), "=&f"(dsum_16[1]) : "r"(dw__4[0]), "f"(dacc[0]), "f"(dacc[1]));
                                    dacc[0] = dsum_16[0];
                                    dacc[1] = dsum_16[1];
                                    float dsum_17[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_17[0]), "=&f"(dsum_17[1]) : "r"(dw__4[1]), "f"(dacc[2]), "f"(dacc[3]));
                                    dacc[2] = dsum_17[0];
                                    dacc[3] = dsum_17[1];
                                    float dsum_18[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_18[0]), "=&f"(dsum_18[1]) : "r"(dw__4[2]), "f"(dacc[4]), "f"(dacc[5]));
                                    dacc[4] = dsum_18[0];
                                    dacc[5] = dsum_18[1];
                                    float dsum_19[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_19[0]), "=&f"(dsum_19[1]) : "r"(dw__4[3]), "f"(dacc[6]), "f"(dacc[7]));
                                    dacc[6] = dsum_19[0];
                                    dacc[7] = dsum_19[1];
                                    unsigned int dw__5[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__5[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__5[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)(2048 + (32 + lane) * 16)));
                                    float dsum_20[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_20[0]), "=&f"(dsum_20[1]) : "r"(dw__5[0]), "f"(dacc[8]), "f"(dacc[9]));
                                    dacc[8] = dsum_20[0];
                                    dacc[9] = dsum_20[1];
                                    float dsum_21[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_21[0]), "=&f"(dsum_21[1]) : "r"(dw__5[1]), "f"(dacc[10]), "f"(dacc[11]));
                                    dacc[10] = dsum_21[0];
                                    dacc[11] = dsum_21[1];
                                    float dsum_22[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_22[0]), "=&f"(dsum_22[1]) : "r"(dw__5[2]), "f"(dacc[12]), "f"(dacc[13]));
                                    dacc[12] = dsum_22[0];
                                    dacc[13] = dsum_22[1];
                                    float dsum_23[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_23[0]), "=&f"(dsum_23[1]) : "r"(dw__5[3]), "f"(dacc[14]), "f"(dacc[15]));
                                    dacc[14] = dsum_23[0];
                                    dacc[15] = dsum_23[1];
                                    unsigned int dw__6[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__6[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__6[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)(2048 + (64 + lane) * 16)));
                                    float dsum_24[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_24[0]), "=&f"(dsum_24[1]) : "r"(dw__6[0]), "f"(dacc[16]), "f"(dacc[17]));
                                    dacc[16] = dsum_24[0];
                                    dacc[17] = dsum_24[1];
                                    float dsum_25[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_25[0]), "=&f"(dsum_25[1]) : "r"(dw__6[1]), "f"(dacc[18]), "f"(dacc[19]));
                                    dacc[18] = dsum_25[0];
                                    dacc[19] = dsum_25[1];
                                    float dsum_26[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_26[0]), "=&f"(dsum_26[1]) : "r"(dw__6[2]), "f"(dacc[20]), "f"(dacc[21]));
                                    dacc[20] = dsum_26[0];
                                    dacc[21] = dsum_26[1];
                                    float dsum_27[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_27[0]), "=&f"(dsum_27[1]) : "r"(dw__6[3]), "f"(dacc[22]), "f"(dacc[23]));
                                    dacc[22] = dsum_27[0];
                                    dacc[23] = dsum_27[1];
                                    unsigned int dw__7[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__7[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__7[(0) + 3]))
                                        : "r"(dbuf + (unsigned int)(2048 + (96 + lane) * 16)));
                                    float dsum_28[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_28[0]), "=&f"(dsum_28[1]) : "r"(dw__7[0]), "f"(dacc[24]), "f"(dacc[25]));
                                    dacc[24] = dsum_28[0];
                                    dacc[25] = dsum_28[1];
                                    float dsum_29[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_29[0]), "=&f"(dsum_29[1]) : "r"(dw__7[1]), "f"(dacc[26]), "f"(dacc[27]));
                                    dacc[26] = dsum_29[0];
                                    dacc[27] = dsum_29[1];
                                    float dsum_30[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_30[0]), "=&f"(dsum_30[1]) : "r"(dw__7[2]), "f"(dacc[28]), "f"(dacc[29]));
                                    dacc[28] = dsum_30[0];
                                    dacc[29] = dsum_30[1];
                                    float dsum_31[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_31[0]), "=&f"(dsum_31[1]) : "r"(dw__7[3]), "f"(dacc[30]), "f"(dacc[31]));
                                    dacc[30] = dsum_31[0];
                                    dacc[31] = dsum_31[1];
                                }
                                fw += 1;
                                nw = nw + 1;
                                if (fj == dper - 1) {
                                    unsigned int dpk_0[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc[_lp*2 + 0], dacc[_lp*2+1 + 0]));
                                        dpk_0[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf + (unsigned int)(lane * 16)), "r"(dpk_0[0]), "r"(dpk_0[1]), "r"(dpk_0[2]), "r"(dpk_0[3]) : "memory");
                                    unsigned int dpk_1[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc[_lp*2 + 8], dacc[_lp*2+1 + 8]));
                                        dpk_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf + (unsigned int)((32 + lane) * 16)), "r"(dpk_1[0]), "r"(dpk_1[1]), "r"(dpk_1[2]), "r"(dpk_1[3]) : "memory");
                                    unsigned int dpk_2[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc[_lp*2 + 16], dacc[_lp*2+1 + 16]));
                                        dpk_2[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf + (unsigned int)((64 + lane) * 16)), "r"(dpk_2[0]), "r"(dpk_2[1]), "r"(dpk_2[2]), "r"(dpk_2[3]) : "memory");
                                    unsigned int dpk_3[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc[_lp*2 + 24], dacc[_lp*2+1 + 24]));
                                        dpk_3[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf + (unsigned int)((96 + lane) * 16)), "r"(dpk_3[0]), "r"(dpk_3[1]), "r"(dpk_3[2]), "r"(dpk_3[3]) : "memory");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (elect_sync()) {
                                        {
                                            void* _cpbulk_dst_348 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(ddst + (unsigned long long)(fc * 2048)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_348), "r"(dbuf), "r"((uint32_t)(2048))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                __syncwarp();
                            }
                            int _stage_efill_bar_0_1 = (int)((unsigned int)fw % 2);
                            int _parity_efill_bar_0_1 = (int)((unsigned int)fw / 2 % 2);
                            mbarrier_wait(efill_bar_0_addr + (_stage_efill_bar_0_1) * 8, _parity_efill_bar_0_1);
                            fw += 1;
                            nw = nw + 1;
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncwarp();
                        }
                        int nxt_0 = dnext;
                        int got_0 = 0;
                        if (elect_sync()) {
                            unsigned int _atomic_fetch_add_349 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                            got_0 = (int)_atomic_fetch_add_349;
                        }
                        int _shfl_sync_350 = __shfl_sync(0xFFFFFFFF, got_0, 0);
                        nxt_0 = _shfl_sync_350;
                        dnext = nxt_0;
                    }
                    cblk = cblk + (ccount + 256 - 1 >> 8);
                    cbegin = cend;
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
            if (warp == 1) {
                int me_1 = mega_pg_rank;
                int fi_1 = 0;
                int fw_1 = 0;
                int nf_1 = 0;
                int nw_1 = 0;
                float dacc_1[32];
                int cd_copy_1 = metadata_ready[0] & 1;
                int nxt_1 = 0;
                int got_1 = 0;
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_351 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                    got_1 = (int)_atomic_fetch_add_351;
                }
                int _shfl_sync_352 = __shfl_sync(0xFFFFFFFF, got_1, 0);
                nxt_1 = _shfl_sync_352;
                int dnext_1 = nxt_1;
                int dprefix_1 = 0;
                int cbegin_1 = 0;
                int cblk_1 = 0;
                int bprefix_1 = 0;
                int tblocks_1 = 0;
                tblocks_1 = tblocks_1 + (expert_counts[0] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[1] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[2] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[3] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[4] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[5] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[6] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[7] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[8] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[9] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[10] + 256 - 1 >> 8);
                tblocks_1 = tblocks_1 + (expert_counts[11] + 256 - 1 >> 8);
                #pragma unroll 1
                for (int ce__1 = 0; ce__1 < 12; ce__1++) {
                    int ccount_1 = expert_counts[ce__1];
                    int cend_1 = cbegin_1 + ccount_1;
                    int cbase_1 = pool_expert_base[ce__1];
                    while (dnext_1 < cend_1) {
                        long long grec_1 = token_src_metadata[(cbase_1 + (dnext_1 - cbegin_1) >> 4 << 5) + (cd_copy_1 << 4) + (cbase_1 + (dnext_1 - cbegin_1) & 15)];
                        if ((unsigned int)(grec_1 & 4294967295ll) == 2147483648u || (unsigned int)((unsigned long long)grec_1 >> 32ull) == 2147483648u) {
                            uint32_t both_1[2];
                            do {
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_1[0]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_1 + (dnext_1 - cbegin_1) >> 4 << 5) + (cd_copy_1 << 4) + (cbase_1 + (dnext_1 - cbegin_1) & 15)))) : "memory");
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_1[1]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_1 + (dnext_1 - cbegin_1) >> 4 << 5) + (cd_copy_1 << 4) + (cbase_1 + (dnext_1 - cbegin_1) & 15))) + 1) : "memory");
                            } while (
                                (both_1[0] == 0x80000000u) || (both_1[1] == 0x80000000u)
                            );
                            grec_1 = (long long)((unsigned long long)(unsigned int)both_1[1] << 32ull | (unsigned long long)(unsigned int)both_1[0]);
                        }
                        int dlo_2 = (int)(grec_1 & 4294967295ll);
                        int dhi_2 = (int)(grec_1 >> 32ll);
                        int dcn_1 = (dhi_2 >> 8 & 255) + 1;
                        int dwork_1 = 0;
                        if (dcn_1 > 1) {
                            if ((dhi_2 & 255) == dcn_1 - 1) {
                                dwork_1 = 1;
                            }
                        }
                        if (dwork_1 != 0) {
                            unsigned int dsleep_1 = 500;
                            int gblock_1 = cblk_1 + (dnext_1 - cbegin_1 >> 8);
                            while (bprefix_1 <= gblock_1) {
                                int bbase_1 = bprefix_1 & -32;
                                unsigned int bfirst_1 = 32;
                                int pb_32 = bbase_1 + lane;
                                if (pb_32 < tblocks_1) {
                                    unsigned int _bval_1;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_bval_1) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pb_32)) : "memory");
                                    if ((unsigned int)_bval_1 < 56) {
                                        bfirst_1 = (unsigned int)lane;
                                    }
                                }
                                unsigned int bend_lane_1;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(bend_lane_1) : "r"(bfirst_1));
                                int _max_353 = ((bprefix_1) > (bbase_1 + (int)bend_lane_1) ? (bprefix_1) : (bbase_1 + (int)bend_lane_1));
                                bprefix_1 = _max_353;
                                if (bprefix_1 <= gblock_1) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_1));
                                    unsigned int _min_354 = ((dsleep_1 * 2) < (1000) ? (dsleep_1 * 2) : (1000));
                                    dsleep_1 = _min_354;
                                }
                            }
                            while (dprefix_1 <= -1) {
                                int pbase_1 = dprefix_1 & -32;
                                int pex_1 = pbase_1 + lane;
                                unsigned int pfirst_1 = 32;
                                if (pex_1 < 12) {
                                    int pcount_1 = expert_counts[pex_1];
                                    unsigned int _pval_1;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_pval_1) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pex_1)) : "memory");
                                    if ((unsigned int)_pval_1 < (unsigned int)((pcount_1 + 256 - 1 >> 8) * 56)) {
                                        pfirst_1 = (unsigned int)lane;
                                    }
                                }
                                unsigned int pend_lane_1;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(pend_lane_1) : "r"(pfirst_1));
                                int _max_355 = ((dprefix_1) > (pbase_1 + (int)pend_lane_1) ? (dprefix_1) : (pbase_1 + (int)pend_lane_1));
                                dprefix_1 = _max_355;
                                if (dprefix_1 <= ce__1) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_1));
                                    unsigned int _min_356 = ((dsleep_1 * 2) < (1000) ? (dsleep_1 * 2) : (1000));
                                    dsleep_1 = _min_356;
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            int dsrc_1 = cake_floor_mod<int>(dlo_2 >> 24 & 255, mega_pg_world);
                            int dtok_1 = dlo_2 & 65535;
                            unsigned long long dblock_1 = (unsigned long long)dedup_rows + (unsigned long long)(dsrc_1 * dedup_cap + dtok_1) * 86016ull;
                            int dpeer_1 = dsrc_1;
                            unsigned long long ddst_1 = (unsigned long long)pre_reduced_peers.ptrs[dpeer_1] + (unsigned long long)((dtok_1 * 8 + me_1) * 7168 * 2);
                            int dper_1 = dcn_1 + 1 >> 1;
                            int dfills_1 = 7 * dper_1;
                            __syncwarp();
                            int d0n_1 = ((dcn_1) < (2) ? (dcn_1) : (2));
                            int dpair_1 = dsrc_1 * dedup_cap + dtok_1;
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive_expect_tx(efill_bar_1_addr + ((int)((unsigned int)fi_1 % 2)) * 8, d0n_1 * 2048);
                                asm volatile(
                                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1], %2, [%3], %4;"
                                    :: "r"(sm_epi_addr + 8192 + (unsigned int)((nf_1 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_1)), "r"((unsigned)(d0n_1 * 2048)), "r"(efill_bar_1_addr + ((int)((unsigned int)fi_1 % 2)) * 8),
                                       "l"(0x12F0000000000000ULL) : "memory");
                            }
                            fi_1 += 1;
                            nf_1 = nf_1 + 1;
                            #pragma unroll 1
                            for (int f_1 = 0; f_1 < dfills_1; f_1++) {
                                int fc_1 = f_1 / dper_1;
                                int fj_1 = f_1 - fc_1 * dper_1;
                                int fn_1 = ((dcn_1 - fj_1 * 2) < (2) ? (dcn_1 - fj_1 * 2) : (2));
                                int fill_live_1 = 0;
                                if (f_1 + 1 < dfills_1) {
                                    fill_live_1 = 1;
                                }
                                int gc__1 = (f_1 + 1) / dper_1;
                                int gj_1 = f_1 + 1 - gc__1 * dper_1;
                                int _min_357 = ((dcn_1 - gj_1 * 2) < (2) ? (dcn_1 - gj_1 * 2) : (2));
                                int gn_1 = _min_357 * fill_live_1;
                                if (elect_sync()) {
                                    if (fill_live_1 != 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    }
                                    mbarrier_arrive_expect_tx(efill_bar_1_addr + ((int)((unsigned int)fi_1 % 2)) * 8, gn_1 * 2048);
                                    if (fill_live_1 != 0) {
                                        asm volatile(
                                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1], %2, [%3], %4;"
                                            :: "r"(sm_epi_addr + 8192 + (unsigned int)((nf_1 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_1 + (unsigned long long)((gc__1 * 6 + gj_1 * 2) * 2048))), "r"((unsigned)(gn_1 * 2048)), "r"(efill_bar_1_addr + ((int)((unsigned int)fi_1 % 2)) * 8),
                                               "l"(0x12F0000000000000ULL) : "memory");
                                    }
                                }
                                fi_1 += 1;
                                nf_1 = nf_1 + 1;
                                if (fj_1 == 0) {
                                    dacc_1[0] = 0.0f;
                                    dacc_1[1] = 0.0f;
                                    dacc_1[2] = 0.0f;
                                    dacc_1[3] = 0.0f;
                                    dacc_1[4] = 0.0f;
                                    dacc_1[5] = 0.0f;
                                    dacc_1[6] = 0.0f;
                                    dacc_1[7] = 0.0f;
                                    dacc_1[8] = 0.0f;
                                    dacc_1[9] = 0.0f;
                                    dacc_1[10] = 0.0f;
                                    dacc_1[11] = 0.0f;
                                    dacc_1[12] = 0.0f;
                                    dacc_1[13] = 0.0f;
                                    dacc_1[14] = 0.0f;
                                    dacc_1[15] = 0.0f;
                                    dacc_1[16] = 0.0f;
                                    dacc_1[17] = 0.0f;
                                    dacc_1[18] = 0.0f;
                                    dacc_1[19] = 0.0f;
                                    dacc_1[20] = 0.0f;
                                    dacc_1[21] = 0.0f;
                                    dacc_1[22] = 0.0f;
                                    dacc_1[23] = 0.0f;
                                    dacc_1[24] = 0.0f;
                                    dacc_1[25] = 0.0f;
                                    dacc_1[26] = 0.0f;
                                    dacc_1[27] = 0.0f;
                                    dacc_1[28] = 0.0f;
                                    dacc_1[29] = 0.0f;
                                    dacc_1[30] = 0.0f;
                                    dacc_1[31] = 0.0f;
                                }
                                int _stage_efill_bar_1 = (int)((unsigned int)fw_1 % 2);
                                int _parity_efill_bar_1 = (int)((unsigned int)fw_1 / 2 % 2);
                                mbarrier_wait(efill_bar_1_addr + (_stage_efill_bar_1) * 8, _parity_efill_bar_1);
                                unsigned int dbuf_1 = sm_epi_addr + 8192 + (unsigned int)((nw_1 & 1) * 4096);
                                if (0 < fn_1) {
                                    unsigned int dw__8[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__8[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__8[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)(lane * 16)));
                                    float dsum_32[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_32[0]), "=&f"(dsum_32[1]) : "r"(dw__8[0]), "f"(dacc_1[0]), "f"(dacc_1[1]));
                                    dacc_1[0] = dsum_32[0];
                                    dacc_1[1] = dsum_32[1];
                                    float dsum_33[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_33[0]), "=&f"(dsum_33[1]) : "r"(dw__8[1]), "f"(dacc_1[2]), "f"(dacc_1[3]));
                                    dacc_1[2] = dsum_33[0];
                                    dacc_1[3] = dsum_33[1];
                                    float dsum_34[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_34[0]), "=&f"(dsum_34[1]) : "r"(dw__8[2]), "f"(dacc_1[4]), "f"(dacc_1[5]));
                                    dacc_1[4] = dsum_34[0];
                                    dacc_1[5] = dsum_34[1];
                                    float dsum_35[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_35[0]), "=&f"(dsum_35[1]) : "r"(dw__8[3]), "f"(dacc_1[6]), "f"(dacc_1[7]));
                                    dacc_1[6] = dsum_35[0];
                                    dacc_1[7] = dsum_35[1];
                                    unsigned int dw__9[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__9[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__9[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)((32 + lane) * 16)));
                                    float dsum_36[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_36[0]), "=&f"(dsum_36[1]) : "r"(dw__9[0]), "f"(dacc_1[8]), "f"(dacc_1[9]));
                                    dacc_1[8] = dsum_36[0];
                                    dacc_1[9] = dsum_36[1];
                                    float dsum_37[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_37[0]), "=&f"(dsum_37[1]) : "r"(dw__9[1]), "f"(dacc_1[10]), "f"(dacc_1[11]));
                                    dacc_1[10] = dsum_37[0];
                                    dacc_1[11] = dsum_37[1];
                                    float dsum_38[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_38[0]), "=&f"(dsum_38[1]) : "r"(dw__9[2]), "f"(dacc_1[12]), "f"(dacc_1[13]));
                                    dacc_1[12] = dsum_38[0];
                                    dacc_1[13] = dsum_38[1];
                                    float dsum_39[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_39[0]), "=&f"(dsum_39[1]) : "r"(dw__9[3]), "f"(dacc_1[14]), "f"(dacc_1[15]));
                                    dacc_1[14] = dsum_39[0];
                                    dacc_1[15] = dsum_39[1];
                                    unsigned int dw__10[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__10[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__10[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)((64 + lane) * 16)));
                                    float dsum_40[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_40[0]), "=&f"(dsum_40[1]) : "r"(dw__10[0]), "f"(dacc_1[16]), "f"(dacc_1[17]));
                                    dacc_1[16] = dsum_40[0];
                                    dacc_1[17] = dsum_40[1];
                                    float dsum_41[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_41[0]), "=&f"(dsum_41[1]) : "r"(dw__10[1]), "f"(dacc_1[18]), "f"(dacc_1[19]));
                                    dacc_1[18] = dsum_41[0];
                                    dacc_1[19] = dsum_41[1];
                                    float dsum_42[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_42[0]), "=&f"(dsum_42[1]) : "r"(dw__10[2]), "f"(dacc_1[20]), "f"(dacc_1[21]));
                                    dacc_1[20] = dsum_42[0];
                                    dacc_1[21] = dsum_42[1];
                                    float dsum_43[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_43[0]), "=&f"(dsum_43[1]) : "r"(dw__10[3]), "f"(dacc_1[22]), "f"(dacc_1[23]));
                                    dacc_1[22] = dsum_43[0];
                                    dacc_1[23] = dsum_43[1];
                                    unsigned int dw__11[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__11[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__11[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)((96 + lane) * 16)));
                                    float dsum_44[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_44[0]), "=&f"(dsum_44[1]) : "r"(dw__11[0]), "f"(dacc_1[24]), "f"(dacc_1[25]));
                                    dacc_1[24] = dsum_44[0];
                                    dacc_1[25] = dsum_44[1];
                                    float dsum_45[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_45[0]), "=&f"(dsum_45[1]) : "r"(dw__11[1]), "f"(dacc_1[26]), "f"(dacc_1[27]));
                                    dacc_1[26] = dsum_45[0];
                                    dacc_1[27] = dsum_45[1];
                                    float dsum_46[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_46[0]), "=&f"(dsum_46[1]) : "r"(dw__11[2]), "f"(dacc_1[28]), "f"(dacc_1[29]));
                                    dacc_1[28] = dsum_46[0];
                                    dacc_1[29] = dsum_46[1];
                                    float dsum_47[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_47[0]), "=&f"(dsum_47[1]) : "r"(dw__11[3]), "f"(dacc_1[30]), "f"(dacc_1[31]));
                                    dacc_1[30] = dsum_47[0];
                                    dacc_1[31] = dsum_47[1];
                                }
                                if (1 < fn_1) {
                                    unsigned int dw__12[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__12[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__12[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)(2048 + lane * 16)));
                                    float dsum_48[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_48[0]), "=&f"(dsum_48[1]) : "r"(dw__12[0]), "f"(dacc_1[0]), "f"(dacc_1[1]));
                                    dacc_1[0] = dsum_48[0];
                                    dacc_1[1] = dsum_48[1];
                                    float dsum_49[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_49[0]), "=&f"(dsum_49[1]) : "r"(dw__12[1]), "f"(dacc_1[2]), "f"(dacc_1[3]));
                                    dacc_1[2] = dsum_49[0];
                                    dacc_1[3] = dsum_49[1];
                                    float dsum_50[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_50[0]), "=&f"(dsum_50[1]) : "r"(dw__12[2]), "f"(dacc_1[4]), "f"(dacc_1[5]));
                                    dacc_1[4] = dsum_50[0];
                                    dacc_1[5] = dsum_50[1];
                                    float dsum_51[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_51[0]), "=&f"(dsum_51[1]) : "r"(dw__12[3]), "f"(dacc_1[6]), "f"(dacc_1[7]));
                                    dacc_1[6] = dsum_51[0];
                                    dacc_1[7] = dsum_51[1];
                                    unsigned int dw__13[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__13[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__13[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)(2048 + (32 + lane) * 16)));
                                    float dsum_52[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_52[0]), "=&f"(dsum_52[1]) : "r"(dw__13[0]), "f"(dacc_1[8]), "f"(dacc_1[9]));
                                    dacc_1[8] = dsum_52[0];
                                    dacc_1[9] = dsum_52[1];
                                    float dsum_53[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_53[0]), "=&f"(dsum_53[1]) : "r"(dw__13[1]), "f"(dacc_1[10]), "f"(dacc_1[11]));
                                    dacc_1[10] = dsum_53[0];
                                    dacc_1[11] = dsum_53[1];
                                    float dsum_54[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_54[0]), "=&f"(dsum_54[1]) : "r"(dw__13[2]), "f"(dacc_1[12]), "f"(dacc_1[13]));
                                    dacc_1[12] = dsum_54[0];
                                    dacc_1[13] = dsum_54[1];
                                    float dsum_55[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_55[0]), "=&f"(dsum_55[1]) : "r"(dw__13[3]), "f"(dacc_1[14]), "f"(dacc_1[15]));
                                    dacc_1[14] = dsum_55[0];
                                    dacc_1[15] = dsum_55[1];
                                    unsigned int dw__14[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__14[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__14[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)(2048 + (64 + lane) * 16)));
                                    float dsum_56[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_56[0]), "=&f"(dsum_56[1]) : "r"(dw__14[0]), "f"(dacc_1[16]), "f"(dacc_1[17]));
                                    dacc_1[16] = dsum_56[0];
                                    dacc_1[17] = dsum_56[1];
                                    float dsum_57[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_57[0]), "=&f"(dsum_57[1]) : "r"(dw__14[1]), "f"(dacc_1[18]), "f"(dacc_1[19]));
                                    dacc_1[18] = dsum_57[0];
                                    dacc_1[19] = dsum_57[1];
                                    float dsum_58[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_58[0]), "=&f"(dsum_58[1]) : "r"(dw__14[2]), "f"(dacc_1[20]), "f"(dacc_1[21]));
                                    dacc_1[20] = dsum_58[0];
                                    dacc_1[21] = dsum_58[1];
                                    float dsum_59[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_59[0]), "=&f"(dsum_59[1]) : "r"(dw__14[3]), "f"(dacc_1[22]), "f"(dacc_1[23]));
                                    dacc_1[22] = dsum_59[0];
                                    dacc_1[23] = dsum_59[1];
                                    unsigned int dw__15[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__15[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__15[(0) + 3]))
                                        : "r"(dbuf_1 + (unsigned int)(2048 + (96 + lane) * 16)));
                                    float dsum_60[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_60[0]), "=&f"(dsum_60[1]) : "r"(dw__15[0]), "f"(dacc_1[24]), "f"(dacc_1[25]));
                                    dacc_1[24] = dsum_60[0];
                                    dacc_1[25] = dsum_60[1];
                                    float dsum_61[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_61[0]), "=&f"(dsum_61[1]) : "r"(dw__15[1]), "f"(dacc_1[26]), "f"(dacc_1[27]));
                                    dacc_1[26] = dsum_61[0];
                                    dacc_1[27] = dsum_61[1];
                                    float dsum_62[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_62[0]), "=&f"(dsum_62[1]) : "r"(dw__15[2]), "f"(dacc_1[28]), "f"(dacc_1[29]));
                                    dacc_1[28] = dsum_62[0];
                                    dacc_1[29] = dsum_62[1];
                                    float dsum_63[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_63[0]), "=&f"(dsum_63[1]) : "r"(dw__15[3]), "f"(dacc_1[30]), "f"(dacc_1[31]));
                                    dacc_1[30] = dsum_63[0];
                                    dacc_1[31] = dsum_63[1];
                                }
                                fw_1 += 1;
                                nw_1 = nw_1 + 1;
                                if (fj_1 == dper_1 - 1) {
                                    unsigned int dpk_4[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_1[_lp*2 + 0], dacc_1[_lp*2+1 + 0]));
                                        dpk_4[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_1 + (unsigned int)(lane * 16)), "r"(dpk_4[0]), "r"(dpk_4[1]), "r"(dpk_4[2]), "r"(dpk_4[3]) : "memory");
                                    unsigned int dpk_5[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_1[_lp*2 + 8], dacc_1[_lp*2+1 + 8]));
                                        dpk_5[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_1 + (unsigned int)((32 + lane) * 16)), "r"(dpk_5[0]), "r"(dpk_5[1]), "r"(dpk_5[2]), "r"(dpk_5[3]) : "memory");
                                    unsigned int dpk_6[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_1[_lp*2 + 16], dacc_1[_lp*2+1 + 16]));
                                        dpk_6[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_1 + (unsigned int)((64 + lane) * 16)), "r"(dpk_6[0]), "r"(dpk_6[1]), "r"(dpk_6[2]), "r"(dpk_6[3]) : "memory");
                                    unsigned int dpk_7[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_1[_lp*2 + 24], dacc_1[_lp*2+1 + 24]));
                                        dpk_7[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_1 + (unsigned int)((96 + lane) * 16)), "r"(dpk_7[0]), "r"(dpk_7[1]), "r"(dpk_7[2]), "r"(dpk_7[3]) : "memory");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (elect_sync()) {
                                        {
                                            void* _cpbulk_dst_358 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(ddst_1 + (unsigned long long)(fc_1 * 2048)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_358), "r"(dbuf_1), "r"((uint32_t)(2048))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                __syncwarp();
                            }
                            int _stage_efill_bar_1_1 = (int)((unsigned int)fw_1 % 2);
                            int _parity_efill_bar_1_1 = (int)((unsigned int)fw_1 / 2 % 2);
                            mbarrier_wait(efill_bar_1_addr + (_stage_efill_bar_1_1) * 8, _parity_efill_bar_1_1);
                            fw_1 += 1;
                            nw_1 = nw_1 + 1;
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncwarp();
                        }
                        int nxt_1_1 = dnext_1;
                        int got_1_1 = 0;
                        if (elect_sync()) {
                            unsigned int _atomic_fetch_add_359 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                            got_1_1 = (int)_atomic_fetch_add_359;
                        }
                        int _shfl_sync_360 = __shfl_sync(0xFFFFFFFF, got_1_1, 0);
                        nxt_1_1 = _shfl_sync_360;
                        dnext_1 = nxt_1_1;
                    }
                    cblk_1 = cblk_1 + (ccount_1 + 256 - 1 >> 8);
                    cbegin_1 = cend_1;
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
            if (warp == 2) {
                int me_2 = mega_pg_rank;
                int fi_2 = 0;
                int fw_2 = 0;
                int nf_2 = 0;
                int nw_2 = 0;
                float dacc_2[32];
                int cd_copy_2 = metadata_ready[0] & 1;
                int nxt_2 = 0;
                int got_2 = 0;
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_361 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                    got_2 = (int)_atomic_fetch_add_361;
                }
                int _shfl_sync_362 = __shfl_sync(0xFFFFFFFF, got_2, 0);
                nxt_2 = _shfl_sync_362;
                int dnext_2 = nxt_2;
                int dprefix_2 = 0;
                int cbegin_2 = 0;
                int cblk_2 = 0;
                int bprefix_2 = 0;
                int tblocks_2 = 0;
                tblocks_2 = tblocks_2 + (expert_counts[0] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[1] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[2] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[3] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[4] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[5] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[6] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[7] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[8] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[9] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[10] + 256 - 1 >> 8);
                tblocks_2 = tblocks_2 + (expert_counts[11] + 256 - 1 >> 8);
                #pragma unroll 1
                for (int ce__2 = 0; ce__2 < 12; ce__2++) {
                    int ccount_2 = expert_counts[ce__2];
                    int cend_2 = cbegin_2 + ccount_2;
                    int cbase_2 = pool_expert_base[ce__2];
                    while (dnext_2 < cend_2) {
                        long long grec_2 = token_src_metadata[(cbase_2 + (dnext_2 - cbegin_2) >> 4 << 5) + (cd_copy_2 << 4) + (cbase_2 + (dnext_2 - cbegin_2) & 15)];
                        if ((unsigned int)(grec_2 & 4294967295ll) == 2147483648u || (unsigned int)((unsigned long long)grec_2 >> 32ull) == 2147483648u) {
                            uint32_t both_2[2];
                            do {
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_2[0]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_2 + (dnext_2 - cbegin_2) >> 4 << 5) + (cd_copy_2 << 4) + (cbase_2 + (dnext_2 - cbegin_2) & 15)))) : "memory");
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_2[1]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_2 + (dnext_2 - cbegin_2) >> 4 << 5) + (cd_copy_2 << 4) + (cbase_2 + (dnext_2 - cbegin_2) & 15))) + 1) : "memory");
                            } while (
                                (both_2[0] == 0x80000000u) || (both_2[1] == 0x80000000u)
                            );
                            grec_2 = (long long)((unsigned long long)(unsigned int)both_2[1] << 32ull | (unsigned long long)(unsigned int)both_2[0]);
                        }
                        int dlo_3 = (int)(grec_2 & 4294967295ll);
                        int dhi_3 = (int)(grec_2 >> 32ll);
                        int dcn_2 = (dhi_3 >> 8 & 255) + 1;
                        int dwork_2 = 0;
                        if (dcn_2 > 1) {
                            if ((dhi_3 & 255) == dcn_2 - 1) {
                                dwork_2 = 1;
                            }
                        }
                        if (dwork_2 != 0) {
                            unsigned int dsleep_2 = 500;
                            int gblock_2 = cblk_2 + (dnext_2 - cbegin_2 >> 8);
                            while (bprefix_2 <= gblock_2) {
                                int bbase_2 = bprefix_2 & -32;
                                unsigned int bfirst_2 = 32;
                                int pb_33 = bbase_2 + lane;
                                if (pb_33 < tblocks_2) {
                                    unsigned int _bval_2;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_bval_2) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pb_33)) : "memory");
                                    if ((unsigned int)_bval_2 < 56) {
                                        bfirst_2 = (unsigned int)lane;
                                    }
                                }
                                unsigned int bend_lane_2;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(bend_lane_2) : "r"(bfirst_2));
                                int _max_363 = ((bprefix_2) > (bbase_2 + (int)bend_lane_2) ? (bprefix_2) : (bbase_2 + (int)bend_lane_2));
                                bprefix_2 = _max_363;
                                if (bprefix_2 <= gblock_2) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_2));
                                    unsigned int _min_364 = ((dsleep_2 * 2) < (1000) ? (dsleep_2 * 2) : (1000));
                                    dsleep_2 = _min_364;
                                }
                            }
                            while (dprefix_2 <= -1) {
                                int pbase_2 = dprefix_2 & -32;
                                int pex_2 = pbase_2 + lane;
                                unsigned int pfirst_2 = 32;
                                if (pex_2 < 12) {
                                    int pcount_2 = expert_counts[pex_2];
                                    unsigned int _pval_2;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_pval_2) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pex_2)) : "memory");
                                    if ((unsigned int)_pval_2 < (unsigned int)((pcount_2 + 256 - 1 >> 8) * 56)) {
                                        pfirst_2 = (unsigned int)lane;
                                    }
                                }
                                unsigned int pend_lane_2;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(pend_lane_2) : "r"(pfirst_2));
                                int _max_365 = ((dprefix_2) > (pbase_2 + (int)pend_lane_2) ? (dprefix_2) : (pbase_2 + (int)pend_lane_2));
                                dprefix_2 = _max_365;
                                if (dprefix_2 <= ce__2) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_2));
                                    unsigned int _min_366 = ((dsleep_2 * 2) < (1000) ? (dsleep_2 * 2) : (1000));
                                    dsleep_2 = _min_366;
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            int dsrc_2 = cake_floor_mod<int>(dlo_3 >> 24 & 255, mega_pg_world);
                            int dtok_2 = dlo_3 & 65535;
                            unsigned long long dblock_2 = (unsigned long long)dedup_rows + (unsigned long long)(dsrc_2 * dedup_cap + dtok_2) * 86016ull;
                            int dpeer_2 = dsrc_2;
                            unsigned long long ddst_2 = (unsigned long long)pre_reduced_peers.ptrs[dpeer_2] + (unsigned long long)((dtok_2 * 8 + me_2) * 7168 * 2);
                            int dper_2 = dcn_2 + 1 >> 1;
                            int dfills_2 = 7 * dper_2;
                            __syncwarp();
                            int d0n_2 = ((dcn_2) < (2) ? (dcn_2) : (2));
                            int dpair_2 = dsrc_2 * dedup_cap + dtok_2;
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive_expect_tx(efill_bar_2_addr + ((int)((unsigned int)fi_2 % 2)) * 8, d0n_2 * 2048);
                                asm volatile(
                                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1], %2, [%3], %4;"
                                    :: "r"(sm_epi_addr + 16384 + (unsigned int)((nf_2 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_2)), "r"((unsigned)(d0n_2 * 2048)), "r"(efill_bar_2_addr + ((int)((unsigned int)fi_2 % 2)) * 8),
                                       "l"(0x12F0000000000000ULL) : "memory");
                            }
                            fi_2 += 1;
                            nf_2 = nf_2 + 1;
                            #pragma unroll 1
                            for (int f_2 = 0; f_2 < dfills_2; f_2++) {
                                int fc_2 = f_2 / dper_2;
                                int fj_2 = f_2 - fc_2 * dper_2;
                                int fn_2 = ((dcn_2 - fj_2 * 2) < (2) ? (dcn_2 - fj_2 * 2) : (2));
                                int fill_live_2 = 0;
                                if (f_2 + 1 < dfills_2) {
                                    fill_live_2 = 1;
                                }
                                int gc__2 = (f_2 + 1) / dper_2;
                                int gj_2 = f_2 + 1 - gc__2 * dper_2;
                                int _min_367 = ((dcn_2 - gj_2 * 2) < (2) ? (dcn_2 - gj_2 * 2) : (2));
                                int gn_2 = _min_367 * fill_live_2;
                                if (elect_sync()) {
                                    if (fill_live_2 != 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    }
                                    mbarrier_arrive_expect_tx(efill_bar_2_addr + ((int)((unsigned int)fi_2 % 2)) * 8, gn_2 * 2048);
                                    if (fill_live_2 != 0) {
                                        asm volatile(
                                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1], %2, [%3], %4;"
                                            :: "r"(sm_epi_addr + 16384 + (unsigned int)((nf_2 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_2 + (unsigned long long)((gc__2 * 6 + gj_2 * 2) * 2048))), "r"((unsigned)(gn_2 * 2048)), "r"(efill_bar_2_addr + ((int)((unsigned int)fi_2 % 2)) * 8),
                                               "l"(0x12F0000000000000ULL) : "memory");
                                    }
                                }
                                fi_2 += 1;
                                nf_2 = nf_2 + 1;
                                if (fj_2 == 0) {
                                    dacc_2[0] = 0.0f;
                                    dacc_2[1] = 0.0f;
                                    dacc_2[2] = 0.0f;
                                    dacc_2[3] = 0.0f;
                                    dacc_2[4] = 0.0f;
                                    dacc_2[5] = 0.0f;
                                    dacc_2[6] = 0.0f;
                                    dacc_2[7] = 0.0f;
                                    dacc_2[8] = 0.0f;
                                    dacc_2[9] = 0.0f;
                                    dacc_2[10] = 0.0f;
                                    dacc_2[11] = 0.0f;
                                    dacc_2[12] = 0.0f;
                                    dacc_2[13] = 0.0f;
                                    dacc_2[14] = 0.0f;
                                    dacc_2[15] = 0.0f;
                                    dacc_2[16] = 0.0f;
                                    dacc_2[17] = 0.0f;
                                    dacc_2[18] = 0.0f;
                                    dacc_2[19] = 0.0f;
                                    dacc_2[20] = 0.0f;
                                    dacc_2[21] = 0.0f;
                                    dacc_2[22] = 0.0f;
                                    dacc_2[23] = 0.0f;
                                    dacc_2[24] = 0.0f;
                                    dacc_2[25] = 0.0f;
                                    dacc_2[26] = 0.0f;
                                    dacc_2[27] = 0.0f;
                                    dacc_2[28] = 0.0f;
                                    dacc_2[29] = 0.0f;
                                    dacc_2[30] = 0.0f;
                                    dacc_2[31] = 0.0f;
                                }
                                int _stage_efill_bar_2 = (int)((unsigned int)fw_2 % 2);
                                int _parity_efill_bar_2 = (int)((unsigned int)fw_2 / 2 % 2);
                                mbarrier_wait(efill_bar_2_addr + (_stage_efill_bar_2) * 8, _parity_efill_bar_2);
                                unsigned int dbuf_2 = sm_epi_addr + 16384 + (unsigned int)((nw_2 & 1) * 4096);
                                if (0 < fn_2) {
                                    unsigned int dw__16[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__16[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__16[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)(lane * 16)));
                                    float dsum_64[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_64[0]), "=&f"(dsum_64[1]) : "r"(dw__16[0]), "f"(dacc_2[0]), "f"(dacc_2[1]));
                                    dacc_2[0] = dsum_64[0];
                                    dacc_2[1] = dsum_64[1];
                                    float dsum_65[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_65[0]), "=&f"(dsum_65[1]) : "r"(dw__16[1]), "f"(dacc_2[2]), "f"(dacc_2[3]));
                                    dacc_2[2] = dsum_65[0];
                                    dacc_2[3] = dsum_65[1];
                                    float dsum_66[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_66[0]), "=&f"(dsum_66[1]) : "r"(dw__16[2]), "f"(dacc_2[4]), "f"(dacc_2[5]));
                                    dacc_2[4] = dsum_66[0];
                                    dacc_2[5] = dsum_66[1];
                                    float dsum_67[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_67[0]), "=&f"(dsum_67[1]) : "r"(dw__16[3]), "f"(dacc_2[6]), "f"(dacc_2[7]));
                                    dacc_2[6] = dsum_67[0];
                                    dacc_2[7] = dsum_67[1];
                                    unsigned int dw__17[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__17[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__17[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)((32 + lane) * 16)));
                                    float dsum_68[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_68[0]), "=&f"(dsum_68[1]) : "r"(dw__17[0]), "f"(dacc_2[8]), "f"(dacc_2[9]));
                                    dacc_2[8] = dsum_68[0];
                                    dacc_2[9] = dsum_68[1];
                                    float dsum_69[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_69[0]), "=&f"(dsum_69[1]) : "r"(dw__17[1]), "f"(dacc_2[10]), "f"(dacc_2[11]));
                                    dacc_2[10] = dsum_69[0];
                                    dacc_2[11] = dsum_69[1];
                                    float dsum_70[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_70[0]), "=&f"(dsum_70[1]) : "r"(dw__17[2]), "f"(dacc_2[12]), "f"(dacc_2[13]));
                                    dacc_2[12] = dsum_70[0];
                                    dacc_2[13] = dsum_70[1];
                                    float dsum_71[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_71[0]), "=&f"(dsum_71[1]) : "r"(dw__17[3]), "f"(dacc_2[14]), "f"(dacc_2[15]));
                                    dacc_2[14] = dsum_71[0];
                                    dacc_2[15] = dsum_71[1];
                                    unsigned int dw__18[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__18[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__18[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)((64 + lane) * 16)));
                                    float dsum_72[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_72[0]), "=&f"(dsum_72[1]) : "r"(dw__18[0]), "f"(dacc_2[16]), "f"(dacc_2[17]));
                                    dacc_2[16] = dsum_72[0];
                                    dacc_2[17] = dsum_72[1];
                                    float dsum_73[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_73[0]), "=&f"(dsum_73[1]) : "r"(dw__18[1]), "f"(dacc_2[18]), "f"(dacc_2[19]));
                                    dacc_2[18] = dsum_73[0];
                                    dacc_2[19] = dsum_73[1];
                                    float dsum_74[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_74[0]), "=&f"(dsum_74[1]) : "r"(dw__18[2]), "f"(dacc_2[20]), "f"(dacc_2[21]));
                                    dacc_2[20] = dsum_74[0];
                                    dacc_2[21] = dsum_74[1];
                                    float dsum_75[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_75[0]), "=&f"(dsum_75[1]) : "r"(dw__18[3]), "f"(dacc_2[22]), "f"(dacc_2[23]));
                                    dacc_2[22] = dsum_75[0];
                                    dacc_2[23] = dsum_75[1];
                                    unsigned int dw__19[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__19[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__19[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)((96 + lane) * 16)));
                                    float dsum_76[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_76[0]), "=&f"(dsum_76[1]) : "r"(dw__19[0]), "f"(dacc_2[24]), "f"(dacc_2[25]));
                                    dacc_2[24] = dsum_76[0];
                                    dacc_2[25] = dsum_76[1];
                                    float dsum_77[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_77[0]), "=&f"(dsum_77[1]) : "r"(dw__19[1]), "f"(dacc_2[26]), "f"(dacc_2[27]));
                                    dacc_2[26] = dsum_77[0];
                                    dacc_2[27] = dsum_77[1];
                                    float dsum_78[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_78[0]), "=&f"(dsum_78[1]) : "r"(dw__19[2]), "f"(dacc_2[28]), "f"(dacc_2[29]));
                                    dacc_2[28] = dsum_78[0];
                                    dacc_2[29] = dsum_78[1];
                                    float dsum_79[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_79[0]), "=&f"(dsum_79[1]) : "r"(dw__19[3]), "f"(dacc_2[30]), "f"(dacc_2[31]));
                                    dacc_2[30] = dsum_79[0];
                                    dacc_2[31] = dsum_79[1];
                                }
                                if (1 < fn_2) {
                                    unsigned int dw__20[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__20[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__20[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)(2048 + lane * 16)));
                                    float dsum_80[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_80[0]), "=&f"(dsum_80[1]) : "r"(dw__20[0]), "f"(dacc_2[0]), "f"(dacc_2[1]));
                                    dacc_2[0] = dsum_80[0];
                                    dacc_2[1] = dsum_80[1];
                                    float dsum_81[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_81[0]), "=&f"(dsum_81[1]) : "r"(dw__20[1]), "f"(dacc_2[2]), "f"(dacc_2[3]));
                                    dacc_2[2] = dsum_81[0];
                                    dacc_2[3] = dsum_81[1];
                                    float dsum_82[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_82[0]), "=&f"(dsum_82[1]) : "r"(dw__20[2]), "f"(dacc_2[4]), "f"(dacc_2[5]));
                                    dacc_2[4] = dsum_82[0];
                                    dacc_2[5] = dsum_82[1];
                                    float dsum_83[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_83[0]), "=&f"(dsum_83[1]) : "r"(dw__20[3]), "f"(dacc_2[6]), "f"(dacc_2[7]));
                                    dacc_2[6] = dsum_83[0];
                                    dacc_2[7] = dsum_83[1];
                                    unsigned int dw__21[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__21[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__21[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)(2048 + (32 + lane) * 16)));
                                    float dsum_84[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_84[0]), "=&f"(dsum_84[1]) : "r"(dw__21[0]), "f"(dacc_2[8]), "f"(dacc_2[9]));
                                    dacc_2[8] = dsum_84[0];
                                    dacc_2[9] = dsum_84[1];
                                    float dsum_85[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_85[0]), "=&f"(dsum_85[1]) : "r"(dw__21[1]), "f"(dacc_2[10]), "f"(dacc_2[11]));
                                    dacc_2[10] = dsum_85[0];
                                    dacc_2[11] = dsum_85[1];
                                    float dsum_86[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_86[0]), "=&f"(dsum_86[1]) : "r"(dw__21[2]), "f"(dacc_2[12]), "f"(dacc_2[13]));
                                    dacc_2[12] = dsum_86[0];
                                    dacc_2[13] = dsum_86[1];
                                    float dsum_87[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_87[0]), "=&f"(dsum_87[1]) : "r"(dw__21[3]), "f"(dacc_2[14]), "f"(dacc_2[15]));
                                    dacc_2[14] = dsum_87[0];
                                    dacc_2[15] = dsum_87[1];
                                    unsigned int dw__22[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__22[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__22[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)(2048 + (64 + lane) * 16)));
                                    float dsum_88[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_88[0]), "=&f"(dsum_88[1]) : "r"(dw__22[0]), "f"(dacc_2[16]), "f"(dacc_2[17]));
                                    dacc_2[16] = dsum_88[0];
                                    dacc_2[17] = dsum_88[1];
                                    float dsum_89[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_89[0]), "=&f"(dsum_89[1]) : "r"(dw__22[1]), "f"(dacc_2[18]), "f"(dacc_2[19]));
                                    dacc_2[18] = dsum_89[0];
                                    dacc_2[19] = dsum_89[1];
                                    float dsum_90[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_90[0]), "=&f"(dsum_90[1]) : "r"(dw__22[2]), "f"(dacc_2[20]), "f"(dacc_2[21]));
                                    dacc_2[20] = dsum_90[0];
                                    dacc_2[21] = dsum_90[1];
                                    float dsum_91[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_91[0]), "=&f"(dsum_91[1]) : "r"(dw__22[3]), "f"(dacc_2[22]), "f"(dacc_2[23]));
                                    dacc_2[22] = dsum_91[0];
                                    dacc_2[23] = dsum_91[1];
                                    unsigned int dw__23[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__23[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__23[(0) + 3]))
                                        : "r"(dbuf_2 + (unsigned int)(2048 + (96 + lane) * 16)));
                                    float dsum_92[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_92[0]), "=&f"(dsum_92[1]) : "r"(dw__23[0]), "f"(dacc_2[24]), "f"(dacc_2[25]));
                                    dacc_2[24] = dsum_92[0];
                                    dacc_2[25] = dsum_92[1];
                                    float dsum_93[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_93[0]), "=&f"(dsum_93[1]) : "r"(dw__23[1]), "f"(dacc_2[26]), "f"(dacc_2[27]));
                                    dacc_2[26] = dsum_93[0];
                                    dacc_2[27] = dsum_93[1];
                                    float dsum_94[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_94[0]), "=&f"(dsum_94[1]) : "r"(dw__23[2]), "f"(dacc_2[28]), "f"(dacc_2[29]));
                                    dacc_2[28] = dsum_94[0];
                                    dacc_2[29] = dsum_94[1];
                                    float dsum_95[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_95[0]), "=&f"(dsum_95[1]) : "r"(dw__23[3]), "f"(dacc_2[30]), "f"(dacc_2[31]));
                                    dacc_2[30] = dsum_95[0];
                                    dacc_2[31] = dsum_95[1];
                                }
                                fw_2 += 1;
                                nw_2 = nw_2 + 1;
                                if (fj_2 == dper_2 - 1) {
                                    unsigned int dpk_8[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_2[_lp*2 + 0], dacc_2[_lp*2+1 + 0]));
                                        dpk_8[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_2 + (unsigned int)(lane * 16)), "r"(dpk_8[0]), "r"(dpk_8[1]), "r"(dpk_8[2]), "r"(dpk_8[3]) : "memory");
                                    unsigned int dpk_9[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_2[_lp*2 + 8], dacc_2[_lp*2+1 + 8]));
                                        dpk_9[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_2 + (unsigned int)((32 + lane) * 16)), "r"(dpk_9[0]), "r"(dpk_9[1]), "r"(dpk_9[2]), "r"(dpk_9[3]) : "memory");
                                    unsigned int dpk_10[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_2[_lp*2 + 16], dacc_2[_lp*2+1 + 16]));
                                        dpk_10[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_2 + (unsigned int)((64 + lane) * 16)), "r"(dpk_10[0]), "r"(dpk_10[1]), "r"(dpk_10[2]), "r"(dpk_10[3]) : "memory");
                                    unsigned int dpk_11[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_2[_lp*2 + 24], dacc_2[_lp*2+1 + 24]));
                                        dpk_11[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_2 + (unsigned int)((96 + lane) * 16)), "r"(dpk_11[0]), "r"(dpk_11[1]), "r"(dpk_11[2]), "r"(dpk_11[3]) : "memory");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (elect_sync()) {
                                        {
                                            void* _cpbulk_dst_368 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(ddst_2 + (unsigned long long)(fc_2 * 2048)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_368), "r"(dbuf_2), "r"((uint32_t)(2048))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                __syncwarp();
                            }
                            int _stage_efill_bar_2_1 = (int)((unsigned int)fw_2 % 2);
                            int _parity_efill_bar_2_1 = (int)((unsigned int)fw_2 / 2 % 2);
                            mbarrier_wait(efill_bar_2_addr + (_stage_efill_bar_2_1) * 8, _parity_efill_bar_2_1);
                            fw_2 += 1;
                            nw_2 = nw_2 + 1;
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncwarp();
                        }
                        int nxt_2_1 = dnext_2;
                        int got_2_1 = 0;
                        if (elect_sync()) {
                            unsigned int _atomic_fetch_add_369 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                            got_2_1 = (int)_atomic_fetch_add_369;
                        }
                        int _shfl_sync_370 = __shfl_sync(0xFFFFFFFF, got_2_1, 0);
                        nxt_2_1 = _shfl_sync_370;
                        dnext_2 = nxt_2_1;
                    }
                    cblk_2 = cblk_2 + (ccount_2 + 256 - 1 >> 8);
                    cbegin_2 = cend_2;
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
            if (warp == 3) {
                int me_3 = mega_pg_rank;
                int fi_3 = 0;
                int fw_3 = 0;
                int nf_3 = 0;
                int nw_3 = 0;
                float dacc_3[32];
                int cd_copy_3 = metadata_ready[0] & 1;
                int nxt_3 = 0;
                int got_3 = 0;
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_371 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                    got_3 = (int)_atomic_fetch_add_371;
                }
                int _shfl_sync_372 = __shfl_sync(0xFFFFFFFF, got_3, 0);
                nxt_3 = _shfl_sync_372;
                int dnext_3 = nxt_3;
                int dprefix_3 = 0;
                int cbegin_3 = 0;
                int cblk_3 = 0;
                int bprefix_3 = 0;
                int tblocks_3 = 0;
                tblocks_3 = tblocks_3 + (expert_counts[0] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[1] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[2] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[3] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[4] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[5] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[6] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[7] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[8] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[9] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[10] + 256 - 1 >> 8);
                tblocks_3 = tblocks_3 + (expert_counts[11] + 256 - 1 >> 8);
                #pragma unroll 1
                for (int ce__3 = 0; ce__3 < 12; ce__3++) {
                    int ccount_3 = expert_counts[ce__3];
                    int cend_3 = cbegin_3 + ccount_3;
                    int cbase_3 = pool_expert_base[ce__3];
                    while (dnext_3 < cend_3) {
                        long long grec_3 = token_src_metadata[(cbase_3 + (dnext_3 - cbegin_3) >> 4 << 5) + (cd_copy_3 << 4) + (cbase_3 + (dnext_3 - cbegin_3) & 15)];
                        if ((unsigned int)(grec_3 & 4294967295ll) == 2147483648u || (unsigned int)((unsigned long long)grec_3 >> 32ull) == 2147483648u) {
                            uint32_t both_3[2];
                            do {
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_3[0]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_3 + (dnext_3 - cbegin_3) >> 4 << 5) + (cd_copy_3 << 4) + (cbase_3 + (dnext_3 - cbegin_3) & 15)))) : "memory");
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_3[1]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_3 + (dnext_3 - cbegin_3) >> 4 << 5) + (cd_copy_3 << 4) + (cbase_3 + (dnext_3 - cbegin_3) & 15))) + 1) : "memory");
                            } while (
                                (both_3[0] == 0x80000000u) || (both_3[1] == 0x80000000u)
                            );
                            grec_3 = (long long)((unsigned long long)(unsigned int)both_3[1] << 32ull | (unsigned long long)(unsigned int)both_3[0]);
                        }
                        int dlo_4 = (int)(grec_3 & 4294967295ll);
                        int dhi_4 = (int)(grec_3 >> 32ll);
                        int dcn_3 = (dhi_4 >> 8 & 255) + 1;
                        int dwork_3 = 0;
                        if (dcn_3 > 1) {
                            if ((dhi_4 & 255) == dcn_3 - 1) {
                                dwork_3 = 1;
                            }
                        }
                        if (dwork_3 != 0) {
                            unsigned int dsleep_3 = 500;
                            int gblock_3 = cblk_3 + (dnext_3 - cbegin_3 >> 8);
                            while (bprefix_3 <= gblock_3) {
                                int bbase_3 = bprefix_3 & -32;
                                unsigned int bfirst_3 = 32;
                                int pb_34 = bbase_3 + lane;
                                if (pb_34 < tblocks_3) {
                                    unsigned int _bval_3;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_bval_3) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pb_34)) : "memory");
                                    if ((unsigned int)_bval_3 < 56) {
                                        bfirst_3 = (unsigned int)lane;
                                    }
                                }
                                unsigned int bend_lane_3;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(bend_lane_3) : "r"(bfirst_3));
                                int _max_373 = ((bprefix_3) > (bbase_3 + (int)bend_lane_3) ? (bprefix_3) : (bbase_3 + (int)bend_lane_3));
                                bprefix_3 = _max_373;
                                if (bprefix_3 <= gblock_3) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_3));
                                    unsigned int _min_374 = ((dsleep_3 * 2) < (1000) ? (dsleep_3 * 2) : (1000));
                                    dsleep_3 = _min_374;
                                }
                            }
                            while (dprefix_3 <= -1) {
                                int pbase_3 = dprefix_3 & -32;
                                int pex_3 = pbase_3 + lane;
                                unsigned int pfirst_3 = 32;
                                if (pex_3 < 12) {
                                    int pcount_3 = expert_counts[pex_3];
                                    unsigned int _pval_3;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_pval_3) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pex_3)) : "memory");
                                    if ((unsigned int)_pval_3 < (unsigned int)((pcount_3 + 256 - 1 >> 8) * 56)) {
                                        pfirst_3 = (unsigned int)lane;
                                    }
                                }
                                unsigned int pend_lane_3;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(pend_lane_3) : "r"(pfirst_3));
                                int _max_375 = ((dprefix_3) > (pbase_3 + (int)pend_lane_3) ? (dprefix_3) : (pbase_3 + (int)pend_lane_3));
                                dprefix_3 = _max_375;
                                if (dprefix_3 <= ce__3) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_3));
                                    unsigned int _min_376 = ((dsleep_3 * 2) < (1000) ? (dsleep_3 * 2) : (1000));
                                    dsleep_3 = _min_376;
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            int dsrc_3 = cake_floor_mod<int>(dlo_4 >> 24 & 255, mega_pg_world);
                            int dtok_3 = dlo_4 & 65535;
                            unsigned long long dblock_3 = (unsigned long long)dedup_rows + (unsigned long long)(dsrc_3 * dedup_cap + dtok_3) * 86016ull;
                            int dpeer_3 = dsrc_3;
                            unsigned long long ddst_3 = (unsigned long long)pre_reduced_peers.ptrs[dpeer_3] + (unsigned long long)((dtok_3 * 8 + me_3) * 7168 * 2);
                            int dper_3 = dcn_3 + 1 >> 1;
                            int dfills_3 = 7 * dper_3;
                            __syncwarp();
                            int d0n_3 = ((dcn_3) < (2) ? (dcn_3) : (2));
                            int dpair_3 = dsrc_3 * dedup_cap + dtok_3;
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive_expect_tx(efill_bar_3_addr + ((int)((unsigned int)fi_3 % 2)) * 8, d0n_3 * 2048);
                                asm volatile(
                                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1], %2, [%3], %4;"
                                    :: "r"(sm_epi_addr + 24576 + (unsigned int)((nf_3 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_3)), "r"((unsigned)(d0n_3 * 2048)), "r"(efill_bar_3_addr + ((int)((unsigned int)fi_3 % 2)) * 8),
                                       "l"(0x12F0000000000000ULL) : "memory");
                            }
                            fi_3 += 1;
                            nf_3 = nf_3 + 1;
                            #pragma unroll 1
                            for (int f_3 = 0; f_3 < dfills_3; f_3++) {
                                int fc_3 = f_3 / dper_3;
                                int fj_3 = f_3 - fc_3 * dper_3;
                                int fn_3 = ((dcn_3 - fj_3 * 2) < (2) ? (dcn_3 - fj_3 * 2) : (2));
                                int fill_live_3 = 0;
                                if (f_3 + 1 < dfills_3) {
                                    fill_live_3 = 1;
                                }
                                int gc__3 = (f_3 + 1) / dper_3;
                                int gj_3 = f_3 + 1 - gc__3 * dper_3;
                                int _min_377 = ((dcn_3 - gj_3 * 2) < (2) ? (dcn_3 - gj_3 * 2) : (2));
                                int gn_3 = _min_377 * fill_live_3;
                                if (elect_sync()) {
                                    if (fill_live_3 != 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    }
                                    mbarrier_arrive_expect_tx(efill_bar_3_addr + ((int)((unsigned int)fi_3 % 2)) * 8, gn_3 * 2048);
                                    if (fill_live_3 != 0) {
                                        asm volatile(
                                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1], %2, [%3], %4;"
                                            :: "r"(sm_epi_addr + 24576 + (unsigned int)((nf_3 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_3 + (unsigned long long)((gc__3 * 6 + gj_3 * 2) * 2048))), "r"((unsigned)(gn_3 * 2048)), "r"(efill_bar_3_addr + ((int)((unsigned int)fi_3 % 2)) * 8),
                                               "l"(0x12F0000000000000ULL) : "memory");
                                    }
                                }
                                fi_3 += 1;
                                nf_3 = nf_3 + 1;
                                if (fj_3 == 0) {
                                    dacc_3[0] = 0.0f;
                                    dacc_3[1] = 0.0f;
                                    dacc_3[2] = 0.0f;
                                    dacc_3[3] = 0.0f;
                                    dacc_3[4] = 0.0f;
                                    dacc_3[5] = 0.0f;
                                    dacc_3[6] = 0.0f;
                                    dacc_3[7] = 0.0f;
                                    dacc_3[8] = 0.0f;
                                    dacc_3[9] = 0.0f;
                                    dacc_3[10] = 0.0f;
                                    dacc_3[11] = 0.0f;
                                    dacc_3[12] = 0.0f;
                                    dacc_3[13] = 0.0f;
                                    dacc_3[14] = 0.0f;
                                    dacc_3[15] = 0.0f;
                                    dacc_3[16] = 0.0f;
                                    dacc_3[17] = 0.0f;
                                    dacc_3[18] = 0.0f;
                                    dacc_3[19] = 0.0f;
                                    dacc_3[20] = 0.0f;
                                    dacc_3[21] = 0.0f;
                                    dacc_3[22] = 0.0f;
                                    dacc_3[23] = 0.0f;
                                    dacc_3[24] = 0.0f;
                                    dacc_3[25] = 0.0f;
                                    dacc_3[26] = 0.0f;
                                    dacc_3[27] = 0.0f;
                                    dacc_3[28] = 0.0f;
                                    dacc_3[29] = 0.0f;
                                    dacc_3[30] = 0.0f;
                                    dacc_3[31] = 0.0f;
                                }
                                int _stage_efill_bar_3 = (int)((unsigned int)fw_3 % 2);
                                int _parity_efill_bar_3 = (int)((unsigned int)fw_3 / 2 % 2);
                                mbarrier_wait(efill_bar_3_addr + (_stage_efill_bar_3) * 8, _parity_efill_bar_3);
                                unsigned int dbuf_3 = sm_epi_addr + 24576 + (unsigned int)((nw_3 & 1) * 4096);
                                if (0 < fn_3) {
                                    unsigned int dw__24[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__24[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__24[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)(lane * 16)));
                                    float dsum_96[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_96[0]), "=&f"(dsum_96[1]) : "r"(dw__24[0]), "f"(dacc_3[0]), "f"(dacc_3[1]));
                                    dacc_3[0] = dsum_96[0];
                                    dacc_3[1] = dsum_96[1];
                                    float dsum_97[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_97[0]), "=&f"(dsum_97[1]) : "r"(dw__24[1]), "f"(dacc_3[2]), "f"(dacc_3[3]));
                                    dacc_3[2] = dsum_97[0];
                                    dacc_3[3] = dsum_97[1];
                                    float dsum_98[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_98[0]), "=&f"(dsum_98[1]) : "r"(dw__24[2]), "f"(dacc_3[4]), "f"(dacc_3[5]));
                                    dacc_3[4] = dsum_98[0];
                                    dacc_3[5] = dsum_98[1];
                                    float dsum_99[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_99[0]), "=&f"(dsum_99[1]) : "r"(dw__24[3]), "f"(dacc_3[6]), "f"(dacc_3[7]));
                                    dacc_3[6] = dsum_99[0];
                                    dacc_3[7] = dsum_99[1];
                                    unsigned int dw__25[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__25[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__25[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)((32 + lane) * 16)));
                                    float dsum_100[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_100[0]), "=&f"(dsum_100[1]) : "r"(dw__25[0]), "f"(dacc_3[8]), "f"(dacc_3[9]));
                                    dacc_3[8] = dsum_100[0];
                                    dacc_3[9] = dsum_100[1];
                                    float dsum_101[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_101[0]), "=&f"(dsum_101[1]) : "r"(dw__25[1]), "f"(dacc_3[10]), "f"(dacc_3[11]));
                                    dacc_3[10] = dsum_101[0];
                                    dacc_3[11] = dsum_101[1];
                                    float dsum_102[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_102[0]), "=&f"(dsum_102[1]) : "r"(dw__25[2]), "f"(dacc_3[12]), "f"(dacc_3[13]));
                                    dacc_3[12] = dsum_102[0];
                                    dacc_3[13] = dsum_102[1];
                                    float dsum_103[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_103[0]), "=&f"(dsum_103[1]) : "r"(dw__25[3]), "f"(dacc_3[14]), "f"(dacc_3[15]));
                                    dacc_3[14] = dsum_103[0];
                                    dacc_3[15] = dsum_103[1];
                                    unsigned int dw__26[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__26[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__26[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)((64 + lane) * 16)));
                                    float dsum_104[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_104[0]), "=&f"(dsum_104[1]) : "r"(dw__26[0]), "f"(dacc_3[16]), "f"(dacc_3[17]));
                                    dacc_3[16] = dsum_104[0];
                                    dacc_3[17] = dsum_104[1];
                                    float dsum_105[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_105[0]), "=&f"(dsum_105[1]) : "r"(dw__26[1]), "f"(dacc_3[18]), "f"(dacc_3[19]));
                                    dacc_3[18] = dsum_105[0];
                                    dacc_3[19] = dsum_105[1];
                                    float dsum_106[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_106[0]), "=&f"(dsum_106[1]) : "r"(dw__26[2]), "f"(dacc_3[20]), "f"(dacc_3[21]));
                                    dacc_3[20] = dsum_106[0];
                                    dacc_3[21] = dsum_106[1];
                                    float dsum_107[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_107[0]), "=&f"(dsum_107[1]) : "r"(dw__26[3]), "f"(dacc_3[22]), "f"(dacc_3[23]));
                                    dacc_3[22] = dsum_107[0];
                                    dacc_3[23] = dsum_107[1];
                                    unsigned int dw__27[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__27[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__27[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)((96 + lane) * 16)));
                                    float dsum_108[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_108[0]), "=&f"(dsum_108[1]) : "r"(dw__27[0]), "f"(dacc_3[24]), "f"(dacc_3[25]));
                                    dacc_3[24] = dsum_108[0];
                                    dacc_3[25] = dsum_108[1];
                                    float dsum_109[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_109[0]), "=&f"(dsum_109[1]) : "r"(dw__27[1]), "f"(dacc_3[26]), "f"(dacc_3[27]));
                                    dacc_3[26] = dsum_109[0];
                                    dacc_3[27] = dsum_109[1];
                                    float dsum_110[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_110[0]), "=&f"(dsum_110[1]) : "r"(dw__27[2]), "f"(dacc_3[28]), "f"(dacc_3[29]));
                                    dacc_3[28] = dsum_110[0];
                                    dacc_3[29] = dsum_110[1];
                                    float dsum_111[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_111[0]), "=&f"(dsum_111[1]) : "r"(dw__27[3]), "f"(dacc_3[30]), "f"(dacc_3[31]));
                                    dacc_3[30] = dsum_111[0];
                                    dacc_3[31] = dsum_111[1];
                                }
                                if (1 < fn_3) {
                                    unsigned int dw__28[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__28[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__28[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)(2048 + lane * 16)));
                                    float dsum_112[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_112[0]), "=&f"(dsum_112[1]) : "r"(dw__28[0]), "f"(dacc_3[0]), "f"(dacc_3[1]));
                                    dacc_3[0] = dsum_112[0];
                                    dacc_3[1] = dsum_112[1];
                                    float dsum_113[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_113[0]), "=&f"(dsum_113[1]) : "r"(dw__28[1]), "f"(dacc_3[2]), "f"(dacc_3[3]));
                                    dacc_3[2] = dsum_113[0];
                                    dacc_3[3] = dsum_113[1];
                                    float dsum_114[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_114[0]), "=&f"(dsum_114[1]) : "r"(dw__28[2]), "f"(dacc_3[4]), "f"(dacc_3[5]));
                                    dacc_3[4] = dsum_114[0];
                                    dacc_3[5] = dsum_114[1];
                                    float dsum_115[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_115[0]), "=&f"(dsum_115[1]) : "r"(dw__28[3]), "f"(dacc_3[6]), "f"(dacc_3[7]));
                                    dacc_3[6] = dsum_115[0];
                                    dacc_3[7] = dsum_115[1];
                                    unsigned int dw__29[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__29[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__29[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)(2048 + (32 + lane) * 16)));
                                    float dsum_116[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_116[0]), "=&f"(dsum_116[1]) : "r"(dw__29[0]), "f"(dacc_3[8]), "f"(dacc_3[9]));
                                    dacc_3[8] = dsum_116[0];
                                    dacc_3[9] = dsum_116[1];
                                    float dsum_117[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_117[0]), "=&f"(dsum_117[1]) : "r"(dw__29[1]), "f"(dacc_3[10]), "f"(dacc_3[11]));
                                    dacc_3[10] = dsum_117[0];
                                    dacc_3[11] = dsum_117[1];
                                    float dsum_118[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_118[0]), "=&f"(dsum_118[1]) : "r"(dw__29[2]), "f"(dacc_3[12]), "f"(dacc_3[13]));
                                    dacc_3[12] = dsum_118[0];
                                    dacc_3[13] = dsum_118[1];
                                    float dsum_119[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_119[0]), "=&f"(dsum_119[1]) : "r"(dw__29[3]), "f"(dacc_3[14]), "f"(dacc_3[15]));
                                    dacc_3[14] = dsum_119[0];
                                    dacc_3[15] = dsum_119[1];
                                    unsigned int dw__30[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__30[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__30[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)(2048 + (64 + lane) * 16)));
                                    float dsum_120[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_120[0]), "=&f"(dsum_120[1]) : "r"(dw__30[0]), "f"(dacc_3[16]), "f"(dacc_3[17]));
                                    dacc_3[16] = dsum_120[0];
                                    dacc_3[17] = dsum_120[1];
                                    float dsum_121[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_121[0]), "=&f"(dsum_121[1]) : "r"(dw__30[1]), "f"(dacc_3[18]), "f"(dacc_3[19]));
                                    dacc_3[18] = dsum_121[0];
                                    dacc_3[19] = dsum_121[1];
                                    float dsum_122[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_122[0]), "=&f"(dsum_122[1]) : "r"(dw__30[2]), "f"(dacc_3[20]), "f"(dacc_3[21]));
                                    dacc_3[20] = dsum_122[0];
                                    dacc_3[21] = dsum_122[1];
                                    float dsum_123[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_123[0]), "=&f"(dsum_123[1]) : "r"(dw__30[3]), "f"(dacc_3[22]), "f"(dacc_3[23]));
                                    dacc_3[22] = dsum_123[0];
                                    dacc_3[23] = dsum_123[1];
                                    unsigned int dw__31[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__31[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__31[(0) + 3]))
                                        : "r"(dbuf_3 + (unsigned int)(2048 + (96 + lane) * 16)));
                                    float dsum_124[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_124[0]), "=&f"(dsum_124[1]) : "r"(dw__31[0]), "f"(dacc_3[24]), "f"(dacc_3[25]));
                                    dacc_3[24] = dsum_124[0];
                                    dacc_3[25] = dsum_124[1];
                                    float dsum_125[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_125[0]), "=&f"(dsum_125[1]) : "r"(dw__31[1]), "f"(dacc_3[26]), "f"(dacc_3[27]));
                                    dacc_3[26] = dsum_125[0];
                                    dacc_3[27] = dsum_125[1];
                                    float dsum_126[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_126[0]), "=&f"(dsum_126[1]) : "r"(dw__31[2]), "f"(dacc_3[28]), "f"(dacc_3[29]));
                                    dacc_3[28] = dsum_126[0];
                                    dacc_3[29] = dsum_126[1];
                                    float dsum_127[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_127[0]), "=&f"(dsum_127[1]) : "r"(dw__31[3]), "f"(dacc_3[30]), "f"(dacc_3[31]));
                                    dacc_3[30] = dsum_127[0];
                                    dacc_3[31] = dsum_127[1];
                                }
                                fw_3 += 1;
                                nw_3 = nw_3 + 1;
                                if (fj_3 == dper_3 - 1) {
                                    unsigned int dpk_12[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_3[_lp*2 + 0], dacc_3[_lp*2+1 + 0]));
                                        dpk_12[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_3 + (unsigned int)(lane * 16)), "r"(dpk_12[0]), "r"(dpk_12[1]), "r"(dpk_12[2]), "r"(dpk_12[3]) : "memory");
                                    unsigned int dpk_13[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_3[_lp*2 + 8], dacc_3[_lp*2+1 + 8]));
                                        dpk_13[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_3 + (unsigned int)((32 + lane) * 16)), "r"(dpk_13[0]), "r"(dpk_13[1]), "r"(dpk_13[2]), "r"(dpk_13[3]) : "memory");
                                    unsigned int dpk_14[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_3[_lp*2 + 16], dacc_3[_lp*2+1 + 16]));
                                        dpk_14[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_3 + (unsigned int)((64 + lane) * 16)), "r"(dpk_14[0]), "r"(dpk_14[1]), "r"(dpk_14[2]), "r"(dpk_14[3]) : "memory");
                                    unsigned int dpk_15[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_3[_lp*2 + 24], dacc_3[_lp*2+1 + 24]));
                                        dpk_15[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_3 + (unsigned int)((96 + lane) * 16)), "r"(dpk_15[0]), "r"(dpk_15[1]), "r"(dpk_15[2]), "r"(dpk_15[3]) : "memory");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (elect_sync()) {
                                        {
                                            void* _cpbulk_dst_378 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(ddst_3 + (unsigned long long)(fc_3 * 2048)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_378), "r"(dbuf_3), "r"((uint32_t)(2048))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                __syncwarp();
                            }
                            int _stage_efill_bar_3_1 = (int)((unsigned int)fw_3 % 2);
                            int _parity_efill_bar_3_1 = (int)((unsigned int)fw_3 / 2 % 2);
                            mbarrier_wait(efill_bar_3_addr + (_stage_efill_bar_3_1) * 8, _parity_efill_bar_3_1);
                            fw_3 += 1;
                            nw_3 = nw_3 + 1;
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncwarp();
                        }
                        int nxt_3_1 = dnext_3;
                        int got_3_1 = 0;
                        if (elect_sync()) {
                            unsigned int _atomic_fetch_add_379 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                            got_3_1 = (int)_atomic_fetch_add_379;
                        }
                        int _shfl_sync_380 = __shfl_sync(0xFFFFFFFF, got_3_1, 0);
                        nxt_3_1 = _shfl_sync_380;
                        dnext_3 = nxt_3_1;
                    }
                    cblk_3 = cblk_3 + (ccount_3 + 256 - 1 >> 8);
                    cbegin_3 = cend_3;
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 4, 288;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // mma_main
            unsigned int stage = 0;
            unsigned int cst_1 = 0;
            {
                int _poll_v_0;
                do {
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<int*>(push_table_ready)) : "memory");
                } while (_poll_v_0 < static_cast<int>(1));
            }
            unsigned int wst_1 = 0;
            int no_pend_1[1];
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_ce = 1;
            unsigned int _phase_af = 0;
            unsigned int _phase_work_empty = 0;
            unsigned int _phase_work_empty_1 = 1;
            if ((cta_rank & 1) == 0) {
                int nit_1 = 0;
                int part_1 = 0;
                part_1 = part_1 + (expert_counts[0] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[1] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[2] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[3] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[4] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[5] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[6] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[7] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[8] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[9] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[10] + 256 - 1 >> 8);
                part_1 = part_1 + (expert_counts[11] + 256 - 1 >> 8);
                int total_1 = part_1 * 26;
                int ticket_1 = 0;
                int s1_1 = 0;
                int sb1_1 = 0;
                int d1_1 = 0;
                int f1_1 = 0;
                int b1_1 = 0;
                int c1_1 = expert_counts[0];
                int n1_1 = (c1_1 + 256 - 1 >> 8) * 12;
                int s2_1 = 0;
                int sb2_1 = 0;
                int d2_1 = 0;
                int f2_1 = 0;
                int b2_1 = 0;
                int c2_1 = expert_counts[0];
                int n2_1 = (c2_1 + 256 - 1 >> 8) * 14;
                #pragma unroll 1
                for (int it_1 = 0; it_1 < total_1 + 1; it_1++) {
                    mbarrier_wait(work_full_addr + (wst_1) * 8, _phase_work_full_1);
                    ticket_1 = (int)work_cells[wst_1 * 4];
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(work_empty_addr + (wst_1) * 8), "r"(1), "r"((uint32_t)((1 + (ticket_1 >> 31)) * (2 - ((int)cluster_size >> 2)))) : "memory");
                    wst_1 += 1;
                    if (wst_1 == 3) { wst_1 = 0; _phase_work_full_1 ^= 1; _phase_work_empty_1 ^= 1; }
                    if (ticket_1 == 2147483647) {
                        break;
                    }
                    int ph_1 = ticket_1 >> 30;
                    int rid_1 = ticket_1 & 1073741823;
                    int e_1 = 0;
                    int local_1 = 0;
                    int db_1 = 0;
                    int fb_1 = 0;
                    int bb_1 = 0;
                    int cnt_1 = 0;
                    int block_1 = 0;
                    int m_pair_1 = 0;
                    if (ph_1 == 0) {
                        while (rid_1 >= sb1_1 + n1_1) {
                            sb1_1 = sb1_1 + n1_1;
                            d1_1 = d1_1 + (c1_1 + 63 & -64);
                            f1_1 = f1_1 + (c1_1 + 127 & -128);
                            b1_1 = b1_1 + (c1_1 + 255 >> 8);
                            s1_1 = s1_1 + 1;
                            c1_1 = 0;
                            if (s1_1 < 12) {
                                c1_1 = expert_counts[s1_1];
                            }
                            n1_1 = (c1_1 + 256 - 1 >> 8) * 12;
                        }
                        e_1 = s1_1;
                        local_1 = rid_1 - sb1_1;
                        db_1 = d1_1;
                        fb_1 = f1_1;
                        bb_1 = b1_1;
                        cnt_1 = c1_1;
                        block_1 = local_1 / 12;
                        m_pair_1 = local_1 - block_1 * 12;
                    } else {
                        while (rid_1 >= sb2_1 + n2_1) {
                            sb2_1 = sb2_1 + n2_1;
                            d2_1 = d2_1 + (c2_1 + 63 & -64);
                            f2_1 = f2_1 + (c2_1 + 127 & -128);
                            b2_1 = b2_1 + (c2_1 + 255 >> 8);
                            s2_1 = s2_1 + 1;
                            c2_1 = 0;
                            if (s2_1 < 12) {
                                c2_1 = expert_counts[s2_1];
                            }
                            n2_1 = (c2_1 + 256 - 1 >> 8) * 14;
                        }
                        e_1 = s2_1;
                        local_1 = rid_1 - sb2_1;
                        db_1 = d2_1;
                        fb_1 = f2_1;
                        bb_1 = b2_1;
                        cnt_1 = c2_1;
                        block_1 = local_1 / 14;
                        m_pair_1 = local_1 - block_1 * 14;
                    }
                    int valid_1 = ((cnt_1 - block_1 * 256) < (256) ? (cnt_1 - block_1 * 256) : (256));
                    int nsub_1 = 2 - ((int)cluster_size >> 2);
                    #pragma unroll 1
                    for (int sub_1 = 0; sub_1 < nsub_1; sub_1++) {
                        int m_one_1 = m_pair_1 * 2 + (cta_rank >> 1) + sub_1;
                        if (ph_1 == 0) {
                            if ((cta_rank & 1) == 0) {
                                int n_rt = valid_1 + 15 & -16;
                                mbarrier_wait(ce_addr + (cst_1) * 8, _phase_ce);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                unsigned int acc_col_2 = cst_1 * 256;
                                if (n_rt <= 128) {
                                    if ((int)cluster_size >> 2 != 0) {
                                        #pragma unroll 1
                                        for (int kt = 0; kt < 28; kt++) {
                                            mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            uint32_t _elected_1 = elect_sync();
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_2), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_3), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_4), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_5), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_6), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_7), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_8), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_9), "r"(_elected_1)
                                                    : "memory");
                                            }
                                            int a_lo_0 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_0 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_0) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_0) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt == 0) ? 0 : 1));
                                                }
                                            }
                                            int a_lo_1 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_1 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_1) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_1) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 8 + 0, 1);
                                                }
                                            }
                                            elect_commit_cg2_multicast(ae_addr + (stage) * 8, (uint16_t)(15));
                                            stage += 1;
                                            if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                        }
                                    } else {
                                        #pragma unroll 1
                                        for (int kt_1 = 0; kt_1 < 28; kt_1++) {
                                            mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            uint32_t _elected_10 = elect_sync();
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_11), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_12), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_13), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_14), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_15), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_16), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_17), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_18), "r"(_elected_10)
                                                    : "memory");
                                            }
                                            int a_lo_2 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_2 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_2) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_2) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_1 == 0) ? 0 : 1));
                                                }
                                            }
                                            int a_lo_3 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_3 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_3) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_3) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 8 + 0, 1);
                                                }
                                            }
                                            uint32_t _elected_19 = elect_sync();
                                            tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_19);
                                            tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_19);
                                            stage += 1;
                                            if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                        }
                                    }
                                } else if ((int)cluster_size >> 2 != 0) {
                                    #pragma unroll 1
                                    for (int kt_2 = 0; kt_2 < 28; kt_2++) {
                                        mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        uint32_t _elected_20 = elect_sync();
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_21 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_21), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_22 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_22), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_23 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_23), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_24 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_24), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_25 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_25), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_26 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_26), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_27 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_27), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_28 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2560)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_28), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_29 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 16)), "l"(_tcgen05_cp_desc_29), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_30 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3072)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 20)), "l"(_tcgen05_cp_desc_30), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_31 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 24)), "l"(_tcgen05_cp_desc_31), "r"(_elected_20)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_32 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3584)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 28)), "l"(_tcgen05_cp_desc_32), "r"(_elected_20)
                                                : "memory");
                                        }
                                        int a_lo_4 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_4 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_4) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_4) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_2 == 0) ? 0 : 1));
                                            }
                                        }
                                        int a_lo_5 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_5 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_5) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_5) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 16 + 0, 1);
                                            }
                                        }
                                        elect_commit_cg2_multicast(ae_addr + (stage) * 8, (uint16_t)(15));
                                        stage += 1;
                                        if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int kt_3 = 0; kt_3 < 28; kt_3++) {
                                        mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        uint32_t _elected_33 = elect_sync();
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_34 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_34), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_35 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_35), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_36 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_36), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_37 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_37), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_38 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_38), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_39 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_39), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_40 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_40), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_41 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2560)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_41), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_42 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 16)), "l"(_tcgen05_cp_desc_42), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_43 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3072)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 20)), "l"(_tcgen05_cp_desc_43), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_44 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 24)), "l"(_tcgen05_cp_desc_44), "r"(_elected_33)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_45 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3584)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 28)), "l"(_tcgen05_cp_desc_45), "r"(_elected_33)
                                                : "memory");
                                        }
                                        int a_lo_6 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_6 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_6) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_6) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_3 == 0) ? 0 : 1));
                                            }
                                        }
                                        int a_lo_7 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_7 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_7) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_7) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_2)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 16 + 0, 1);
                                            }
                                        }
                                        uint32_t _elected_46 = elect_sync();
                                        tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_46);
                                        tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_46);
                                        stage += 1;
                                        if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                    }
                                }
                                elect_commit_cg2_multicast(cf_addr + (cst_1) * 8, (uint16_t)(3 << (cta_rank & 2)));
                                cst_1 += 1;
                                if (cst_1 == 2) { cst_1 = 0; _phase_ce ^= 1; }
                            }
                        }
                        if (ph_1 == 1) {
                            if ((cta_rank & 1) == 0) {
                                int n_rt_1 = valid_1 + 15 & -16;
                                mbarrier_wait(ce_addr + (cst_1) * 8, _phase_ce);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                unsigned int acc_col_3 = cst_1 * 256;
                                if (n_rt_1 <= 128) {
                                    if ((int)cluster_size >> 2 != 0) {
                                        #pragma unroll 1
                                        for (int kt_4 = 0; kt_4 < 12; kt_4++) {
                                            mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            uint32_t _elected_47 = elect_sync();
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_48 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_48), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_49 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_49), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_50 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_50), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_51 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_51), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_52 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_52), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_53 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_53), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_54 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_54), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_55 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_55), "r"(_elected_47)
                                                    : "memory");
                                            }
                                            int a_lo_8 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_8 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_8) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_8) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_4 == 0) ? 0 : 1));
                                                }
                                            }
                                            int a_lo_9 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_9 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_9) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_9) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 8 + 0, 1);
                                                }
                                            }
                                            elect_commit_cg2_multicast(ae_addr + (stage) * 8, (uint16_t)(15));
                                            stage += 1;
                                            if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                        }
                                    } else {
                                        #pragma unroll 1
                                        for (int kt_5 = 0; kt_5 < 12; kt_5++) {
                                            mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                            asm volatile("tcgen05.fence::after_thread_sync;");
                                            uint32_t _elected_56 = elect_sync();
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_57 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_57), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_58 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_58), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_59 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_59), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_60 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_60), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_61 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_61), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_62 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_62), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_63 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_63), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                            #endif
                                            {
                                                uint64_t _tcgen05_cp_desc_64 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                                asm volatile(
                                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                    "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                    :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_64), "r"(_elected_56)
                                                    : "memory");
                                            }
                                            int a_lo_10 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_10 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_10) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_10) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_5 == 0) ? 0 : 1));
                                                }
                                            }
                                            int a_lo_11 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            int b_lo_11 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                            if (elect_sync()) {
                                                {
                                                    uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_11) | ((uint64_t)0x40004040 << 32);
                                                    uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_11) | ((uint64_t)0x40004040 << 32);

                                                    tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                        ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 8 + 0, 1);
                                                }
                                            }
                                            uint32_t _elected_65 = elect_sync();
                                            tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_65);
                                            tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_65);
                                            stage += 1;
                                            if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                        }
                                    }
                                } else if ((int)cluster_size >> 2 != 0) {
                                    #pragma unroll 1
                                    for (int kt_6 = 0; kt_6 < 12; kt_6++) {
                                        mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        uint32_t _elected_66 = elect_sync();
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_67 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_67), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_68 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_68), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_69 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_69), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_70 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_70), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_71 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_71), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_72 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_72), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_73 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_73), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_74 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2560)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_74), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_75 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 16)), "l"(_tcgen05_cp_desc_75), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_76 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3072)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 20)), "l"(_tcgen05_cp_desc_76), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_77 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 24)), "l"(_tcgen05_cp_desc_77), "r"(_elected_66)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_78 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3584)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 28)), "l"(_tcgen05_cp_desc_78), "r"(_elected_66)
                                                : "memory");
                                        }
                                        int a_lo_12 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_12 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_12) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_12) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_6 == 0) ? 0 : 1));
                                            }
                                        }
                                        int a_lo_13 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_13 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_13) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_13) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 16 + 0, 1);
                                            }
                                        }
                                        elect_commit_cg2_multicast(ae_addr + (stage) * 8, (uint16_t)(15));
                                        stage += 1;
                                        if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int kt_7 = 0; kt_7 < 12; kt_7++) {
                                        mbarrier_wait(af_addr + (stage) * 8, _phase_af);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        uint32_t _elected_79 = elect_sync();
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_80 = ((((uint64_t)(sm_sfa_addr + stage * 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512)), "l"(_tcgen05_cp_desc_80), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_81 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 4)), "l"(_tcgen05_cp_desc_81), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_82 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 8)), "l"(_tcgen05_cp_desc_82), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_83 = ((((uint64_t)(sm_sfa_addr + stage * 2048 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 512 + 12)), "l"(_tcgen05_cp_desc_83), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_84 = ((((uint64_t)(sm_sfb_addr + stage * 4096)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528)), "l"(_tcgen05_cp_desc_84), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_85 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2048)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 4)), "l"(_tcgen05_cp_desc_85), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_86 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 512)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 8)), "l"(_tcgen05_cp_desc_86), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_87 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 2560)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 12)), "l"(_tcgen05_cp_desc_87), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_88 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1024)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 16)), "l"(_tcgen05_cp_desc_88), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_89 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3072)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 20)), "l"(_tcgen05_cp_desc_89), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_90 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 1536)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 24)), "l"(_tcgen05_cp_desc_90), "r"(_elected_79)
                                                : "memory");
                                        }
                                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                        #endif
                                        {
                                            uint64_t _tcgen05_cp_desc_91 = ((((uint64_t)(sm_sfb_addr + stage * 4096 + 3584)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                            asm volatile(
                                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                                "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
                                                :: "r"((uint32_t)(taddr + 528 + 28)), "l"(_tcgen05_cp_desc_91), "r"(_elected_79)
                                                : "memory");
                                        }
                                        int a_lo_14 = make_warp_uniform((((sm_a_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_14 = make_warp_uniform((((sm_b_addr) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_14) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_14) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 0 + 0, tm_sfb + 0 + 0, ((kt_7 == 0) ? 0 : 1));
                                            }
                                        }
                                        int a_lo_15 = make_warp_uniform((((sm_a_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        int b_lo_15 = make_warp_uniform((((sm_b_addr + 64) >> 4) & 0x7FFF) + (stage) * 1024);
                                        if (elect_sync()) {
                                            {
                                                uint64_t a_desc = ((uint64_t)(uint32_t)a_lo_15) | ((uint64_t)0x40004040 << 32);
                                                uint64_t b_desc = ((uint64_t)(uint32_t)b_lo_15) | ((uint64_t)0x40004040 << 32);

                                                tcgen05_mma_mxf4nvf4_bs_block16_k128_cta2((accum + (acc_col_3)), a_desc + 0, b_desc + 0,
                                                    ((0x10401488U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(n_rt_1) >> 3) << 17)), tm_sfa + 8 + 0, tm_sfb + 16 + 0, 1);
                                            }
                                        }
                                        uint32_t _elected_92 = elect_sync();
                                        tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_92);
                                        tcgen05_commit_cg2_multicast_leader(ae_addr + (stage) * 8, (uint16_t)(3), _elected_92);
                                        stage += 1;
                                        if (stage == 6) { stage = 0; _phase_af ^= 1; }
                                    }
                                }
                                elect_commit_cg2_multicast(cf_addr + (cst_1) * 8, (uint16_t)(3 << (cta_rank & 2)));
                                cst_1 += 1;
                                if (cst_1 == 2) { cst_1 = 0; _phase_ce ^= 1; }
                            }
                        }
                        nit_1 = nit_1 + 1;
                    }
                }
            } else if (cta_rank == 1) {
                int part_2 = 0;
                part_2 = part_2 + (expert_counts[0] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[1] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[2] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[3] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[4] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[5] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[6] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[7] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[8] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[9] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[10] + 256 - 1 >> 8);
                part_2 = part_2 + (expert_counts[11] + 256 - 1 >> 8);
                int total_2 = part_2 * 26;
                uint32_t _read_special_reg_93;
                asm("mov.u32 %0, %%locality_domain_id;" : "=r"(_read_special_reg_93));
                int dom = (int)_read_special_reg_93 & 1;
                int dom_open = 1;
                unsigned int wtr = 0;
                int fb_small = 1 - ((int)cluster_size >> 2);
                int part_0 = 0;
                part_0 = part_0 + (expert_counts[0] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[1] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[2] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[3] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[4] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[5] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[6] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[7] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[8] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[9] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[10] + 256 - 1 >> 8);
                part_0 = part_0 + (expert_counts[11] + 256 - 1 >> 8);
                int rc_part = part_0;
                int rc_n1 = rc_part * 12;
                int rc_n2 = rc_part * 14;
                int hold = -1;
                int last2 = 0;
                int open1 = 1;
                int open2 = 1;
                int q_s = 0;
                int q_sb = 0;
                int q_b = 0;
                int q_c = expert_counts[0];
                int q_n = (q_c + 256 - 1 >> 8) * 14;
                int cy_pro = 3;
                int cy_pos = (int)cluster_id % 13;
                int cy_sync = 0;
                int cy_w = num_bids >> 2;
                int cy_wm = (((cy_w + 14 - 1) / 14 * 12) < (3 * cy_w) ? ((cy_w + 14 - 1) / 14 * 12) : (3 * cy_w));
                int _min_94 = ((cy_wm) < (rc_n1) ? (cy_wm) : (rc_n1));
                cy_wm = _min_94;
                int dom_cut1 = rc_n1 * 850 / 1000 / 12 * 12;
                int bal_all = 0;
                int bal_n0 = 0;
                if (elect_sync()) {
                    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (7))), "r"(static_cast<unsigned int>((unsigned int)cluster_size << 16 | (unsigned int)cluster_size * (unsigned int)(1 - dom))) : "memory");
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (7))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)num_ctas << 16)) >= static_cast<unsigned int>(2147483648u));
                    }
                    unsigned int _atomic_fetch_add_95 = atomicAdd(&sync_words[7], 0);
                    int bal_word = (int)_atomic_fetch_add_95;
                    bal_all = bal_word >> 16 & 65535;
                    bal_n0 = bal_word & 65535;
                }
                #pragma unroll 1
                for (int _c = 0; _c < total_2 + 1; _c++) {
                    if (fb_small != 0) {
                        if (_c >= 1) {
                            mbarrier_wait_cluster_hint(work_empty_addr + (wtr) * 8, _phase_work_empty, 10000000);
                            wtr += 1;
                            if (wtr == 3) { wtr = 0; _phase_work_empty ^= 1; }
                        }
                    }
                    mbarrier_wait_cluster_hint(work_empty_addr + (wst_1) * 8, _phase_work_empty_1, 10000000);
                    int got_4 = 0;
                    if (elect_sync()) {
                        unsigned int claimed4 = 2147483647;
                        int want1 = 1;
                        if (cy_pro <= 0) {
                            if (cy_pos * 7 % 13 < 7) {
                                want1 = 0;
                            }
                        }
                        if (open1 == 0) {
                            want1 = 0;
                        }
                        if (open2 == 0) {
                            want1 = 1;
                        }
                        if (want1 == 0) {
                            if (cy_sync == 0) {
                                {
                                unsigned int _acquire_observed;
                                do {
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (1))) : "memory");
                                } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)cy_wm)) >= static_cast<unsigned int>(2147483648u));
                                }
                                cy_sync = 1;
                            }
                            unsigned int _atomic_fetch_add_96 = atomicAdd(&sync_words[3], 1);
                            int c2_2 = (int)_atomic_fetch_add_96;
                            if (c2_2 < rc_n2) {
                                claimed4 = (unsigned int)(1073741824 | c2_2);
                            } else {
                                open2 = 0;
                            }
                        }
                        if (claimed4 == 2147483647) {
                            if (open1 != 0) {
                                int c1_2 = 0;
                                if (dom_open != 0) {
                                    unsigned int _atomic_fetch_add_97 = atomicAdd(&sync_words[4 + dom], 1);
                                    int dk1 = (int)_atomic_fetch_add_97;
                                    int bal_nd = bal_n0 + dom * (bal_all - 2 * bal_n0);
                                    int bal_b = dk1 * bal_all / (12 * bal_nd);
                                    while (dom * (bal_b + 1) * 12 + (1 - 2 * dom) * ((bal_b + 1) * 12 * bal_n0 / bal_all) <= dk1) {
                                        bal_b = bal_b + 1;
                                    }
                                    while (dom * bal_b * 12 + (1 - 2 * dom) * (bal_b * 12 * bal_n0 / bal_all) > dk1) {
                                        bal_b = bal_b - 1;
                                    }
                                    int bal_m = dk1 - (dom * bal_b * 12 + (1 - 2 * dom) * (bal_b * 12 * bal_n0 / bal_all)) + dom * ((bal_b + 1) * 12 * bal_n0 / bal_all - bal_b * 12 * bal_n0 / bal_all);
                                    c1_2 = bal_b * 12 + bal_m;
                                    if (c1_2 >= dom_cut1) {
                                        dom_open = 0;
                                    }
                                }
                                if (dom_open == 0) {
                                    unsigned int _atomic_fetch_add_98 = atomicAdd(&sync_words[6], 1);
                                    c1_2 = dom_cut1 + (int)_atomic_fetch_add_98;
                                }
                                asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sync_words) + (1))), "r"(static_cast<unsigned int>(1)) : "memory");
                                if (c1_2 < rc_n1) {
                                    claimed4 = (unsigned int)c1_2;
                                } else {
                                    open1 = 0;
                                }
                            }
                        }
                        if (claimed4 == 2147483647) {
                            if (open2 != 0) {
                                unsigned int _atomic_fetch_add_99 = atomicAdd(&sync_words[3], 1);
                                int c3 = (int)_atomic_fetch_add_99;
                                if (c3 < rc_n2) {
                                    claimed4 = (unsigned int)(1073741824 | c3);
                                } else {
                                    open2 = 0;
                                }
                            }
                        }
                        if (claimed4 != 2147483647) {
                            if (cy_pro > 0) {
                                cy_pro = cy_pro - 1;
                            } else {
                                cy_pos = cy_pos + 1;
                                if (cy_pos == 13) {
                                    cy_pos = 0;
                                }
                            }
                        }
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(work_cells_addr + wst_1 * 16), "r"(claimed4) : "memory");
                        int cbig = (int)cluster_size >> 2;
                        int sendp = 0;
                        sendp = 1;
                        if (sendp != 0) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + (wst_1) * 8), "r"(0), "r"((uint32_t)(4)) : "memory");
                            uint32_t _cluster_mapa_100;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_100) : "r"(work_cells_addr + wst_1 * 16), "r"(0));
                            uint32_t _mbar_peer_101;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_101) : "r"(work_full_addr + (wst_1) * 8), "r"(0));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                                :: "r"(_cluster_mapa_100), "r"(claimed4), "r"(_mbar_peer_101) : "memory");
                        }
                        sendp = 1;
                        sendp = cbig;
                        if (sendp != 0) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + (wst_1) * 8), "r"(2), "r"((uint32_t)(4)) : "memory");
                            uint32_t _cluster_mapa_102;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_102) : "r"(work_cells_addr + wst_1 * 16), "r"(2));
                            uint32_t _mbar_peer_103;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_103) : "r"(work_full_addr + (wst_1) * 8), "r"(2));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                                :: "r"(_cluster_mapa_102), "r"(claimed4), "r"(_mbar_peer_103) : "memory");
                        }
                        sendp = 1;
                        sendp = cbig;
                        if (sendp != 0) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + (wst_1) * 8), "r"(3), "r"((uint32_t)(4)) : "memory");
                            uint32_t _cluster_mapa_104;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_cluster_mapa_104) : "r"(work_cells_addr + wst_1 * 16), "r"(3));
                            uint32_t _mbar_peer_105;
                            asm(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mbar_peer_105) : "r"(work_full_addr + (wst_1) * 8), "r"(3));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                                :: "r"(_cluster_mapa_104), "r"(claimed4), "r"(_mbar_peer_105) : "memory");
                        }
                        mbarrier_arrive(work_full_addr + (wst_1) * 8);
                        got_4 = (int)claimed4;
                    }
                    int _shfl_sync_106 = __shfl_sync(0xFFFFFFFF, got_4, 0);
                    got_4 = _shfl_sync_106;
                    wst_1 += 1;
                    if (wst_1 == 3) { wst_1 = 0; _phase_work_full_1 ^= 1; _phase_work_empty_1 ^= 1; }
                    if (got_4 == 2147483647) {
                        break;
                    }
                }
                #pragma unroll 1
                for (int _d = 0; _d < 3; _d++) {
                    mbarrier_wait_cluster_hint(work_empty_addr + (wst_1) * 8, _phase_work_empty_1, 10000000);
                    wst_1 += 1;
                    if (wst_1 == 3) { wst_1 = 0; _phase_work_full_1 ^= 1; _phase_work_empty_1 ^= 1; }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: tma_a ----
    if (warp == 5) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // tma_a_main
            unsigned int stage_1 = 0;
            int cta_1 = cta_rank & 1;
            {
                int _poll_v_0;
                do {
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<int*>(push_table_ready)) : "memory");
                } while (_poll_v_0 < static_cast<int>(1));
            }
            unsigned int wst_2 = 0;
            int no_pend_2[1];
            int ait = 0;
            int part_3 = 0;
            part_3 = part_3 + (expert_counts[0] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[1] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[2] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[3] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[4] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[5] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[6] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[7] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[8] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[9] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[10] + 256 - 1 >> 8);
            part_3 = part_3 + (expert_counts[11] + 256 - 1 >> 8);
            int total_3 = part_3 * 26;
            int ticket_2 = 0;
            int s1_2 = 0;
            int sb1_2 = 0;
            int d1_2 = 0;
            int f1_2 = 0;
            int b1_2 = 0;
            int c1_3 = expert_counts[0];
            int n1_2 = (c1_3 + 256 - 1 >> 8) * 12;
            int s2_2 = 0;
            int sb2_2 = 0;
            int d2_2 = 0;
            int f2_2 = 0;
            int b2_2 = 0;
            int c2_3 = expert_counts[0];
            int n2_2 = (c2_3 + 256 - 1 >> 8) * 14;
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_ae = 1;
            #pragma unroll 1
            for (int it_2 = 0; it_2 < total_3 + 1; it_2++) {
                mbarrier_wait(work_full_addr + (wst_2) * 8, _phase_work_full_2);
                ticket_2 = (int)work_cells[wst_2 * 4];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wst_2) * 8), "r"(1), "r"((uint32_t)((1 + (ticket_2 >> 31)) * (2 - ((int)cluster_size >> 2)))) : "memory");
                wst_2 += 1;
                if (wst_2 == 3) { wst_2 = 0; _phase_work_full_2 ^= 1; }
                if (ticket_2 == 2147483647) {
                    break;
                }
                int ph_2 = ticket_2 >> 30;
                int rid_2 = ticket_2 & 1073741823;
                int e_2 = 0;
                int local_2 = 0;
                int db_2 = 0;
                int fb_2 = 0;
                int bb_2 = 0;
                int cnt_2 = 0;
                int block_2 = 0;
                int m_pair_2 = 0;
                if (ph_2 == 0) {
                    while (rid_2 >= sb1_2 + n1_2) {
                        sb1_2 = sb1_2 + n1_2;
                        d1_2 = d1_2 + (c1_3 + 63 & -64);
                        f1_2 = f1_2 + (c1_3 + 127 & -128);
                        b1_2 = b1_2 + (c1_3 + 255 >> 8);
                        s1_2 = s1_2 + 1;
                        c1_3 = 0;
                        if (s1_2 < 12) {
                            c1_3 = expert_counts[s1_2];
                        }
                        n1_2 = (c1_3 + 256 - 1 >> 8) * 12;
                    }
                    e_2 = s1_2;
                    local_2 = rid_2 - sb1_2;
                    db_2 = d1_2;
                    fb_2 = f1_2;
                    bb_2 = b1_2;
                    cnt_2 = c1_3;
                    block_2 = local_2 / 12;
                    m_pair_2 = local_2 - block_2 * 12;
                } else {
                    while (rid_2 >= sb2_2 + n2_2) {
                        sb2_2 = sb2_2 + n2_2;
                        d2_2 = d2_2 + (c2_3 + 63 & -64);
                        f2_2 = f2_2 + (c2_3 + 127 & -128);
                        b2_2 = b2_2 + (c2_3 + 255 >> 8);
                        s2_2 = s2_2 + 1;
                        c2_3 = 0;
                        if (s2_2 < 12) {
                            c2_3 = expert_counts[s2_2];
                        }
                        n2_2 = (c2_3 + 256 - 1 >> 8) * 14;
                    }
                    e_2 = s2_2;
                    local_2 = rid_2 - sb2_2;
                    db_2 = d2_2;
                    fb_2 = f2_2;
                    bb_2 = b2_2;
                    cnt_2 = c2_3;
                    block_2 = local_2 / 14;
                    m_pair_2 = local_2 - block_2 * 14;
                }
                int valid_2 = ((cnt_2 - block_2 * 256) < (256) ? (cnt_2 - block_2 * 256) : (256));
                int nsub_2 = 2 - ((int)cluster_size >> 2);
                #pragma unroll 1
                for (int sub_2 = 0; sub_2 < nsub_2; sub_2++) {
                    int m_one_2 = m_pair_2 * 2 + (cta_rank >> 1) + sub_2;
                    if (ph_2 == 0) {
                        int row = m_one_2 * 256 + cta_1 * 128;
                        int atom = m_one_2 * 2 + cta_1;
                        #pragma unroll 1
                        for (int kt_8 = 0; kt_8 < 28; kt_8++) {
                            mbarrier_wait(ae_addr + (stage_1) * 8, _phase_ae);
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((af_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(18432)) : "memory");
                                tma_3d_gmem2smem_cta2(sm_a_addr + stage_1 * 16384, (&W1), kt_8 * 256, row, e_2, ((af_addr + (stage_1) * 8) & 0xFEFFFFFF));
                                tma_4d_gmem2smem_cta2(sm_sfa_addr + stage_1 * 2048, (&S1), 0, kt_8 * 4, atom, e_2, ((af_addr + (stage_1) * 8) & 0xFEFFFFFF));
                            }
                            stage_1 += 1;
                            if (stage_1 == 6) { stage_1 = 0; _phase_ae ^= 1; }
                        }
                    }
                    if (ph_2 == 1) {
                        int row_4 = m_one_2 * 256 + cta_1 * 128;
                        int atom_1 = m_one_2 * 2 + cta_1;
                        #pragma unroll 1
                        for (int kt_9 = 0; kt_9 < 12; kt_9++) {
                            mbarrier_wait(ae_addr + (stage_1) * 8, _phase_ae);
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((af_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(18432)) : "memory");
                                tma_3d_gmem2smem_cta2(sm_a_addr + stage_1 * 16384, (&W2), kt_9 * 256, row_4, e_2, ((af_addr + (stage_1) * 8) & 0xFEFFFFFF));
                                tma_4d_gmem2smem_cta2(sm_sfa_addr + stage_1 * 2048, (&S2), 0, kt_9 * 4, atom_1, e_2, ((af_addr + (stage_1) * 8) & 0xFEFFFFFF));
                            }
                            stage_1 += 1;
                            if (stage_1 == 6) { stage_1 = 0; _phase_ae ^= 1; }
                        }
                    }
                    ait = ait + 1;
                }
            }
            #pragma unroll 1
            for (int _s = 0; _s < 6; _s++) {
                mbarrier_wait(ae_addr + (stage_1) * 8, _phase_ae);
                stage_1 += 1;
                if (stage_1 == 6) { stage_1 = 0; _phase_ae ^= 1; }
            }
        }
    }
    // ---- Role: tma_b ----
    if (warp == 6) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // tma_b_main
            unsigned int stage_2 = 0;
            int cta_2 = cta_rank & 1;
            {
                int _poll_v_0;
                do {
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<int*>(push_table_ready)) : "memory");
                } while (_poll_v_0 < static_cast<int>(1));
            }
            unsigned int wst_3 = 0;
            int no_pend_3[1];
            int nit_2 = 0;
            int part_4 = 0;
            part_4 = part_4 + (expert_counts[0] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[1] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[2] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[3] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[4] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[5] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[6] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[7] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[8] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[9] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[10] + 256 - 1 >> 8);
            part_4 = part_4 + (expert_counts[11] + 256 - 1 >> 8);
            int total_4 = part_4 * 26;
            int ticket_3 = 0;
            int s1_3 = 0;
            int sb1_3 = 0;
            int d1_3 = 0;
            int f1_3 = 0;
            int b1_3 = 0;
            int c1_4 = expert_counts[0];
            int n1_3 = (c1_4 + 256 - 1 >> 8) * 12;
            int s2_3 = 0;
            int sb2_3 = 0;
            int d2_3 = 0;
            int f2_3 = 0;
            int b2_3 = 0;
            int c2_4 = expert_counts[0];
            int n2_3 = (c2_4 + 256 - 1 >> 8) * 14;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_ae_1 = 1;
            #pragma unroll 1
            for (int it_3 = 0; it_3 < total_4 + 1; it_3++) {
                mbarrier_wait(work_full_addr + (wst_3) * 8, _phase_work_full_3);
                ticket_3 = (int)work_cells[wst_3 * 4];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wst_3) * 8), "r"(1), "r"((uint32_t)((1 + (ticket_3 >> 31)) * (2 - ((int)cluster_size >> 2)))) : "memory");
                wst_3 += 1;
                if (wst_3 == 3) { wst_3 = 0; _phase_work_full_3 ^= 1; }
                if (ticket_3 == 2147483647) {
                    break;
                }
                int ph_3 = ticket_3 >> 30;
                int rid_3 = ticket_3 & 1073741823;
                int e_3 = 0;
                int local_3 = 0;
                int db_3 = 0;
                int fb_3 = 0;
                int bb_3 = 0;
                int cnt_3 = 0;
                int block_3 = 0;
                int m_pair_3 = 0;
                if (ph_3 == 0) {
                    while (rid_3 >= sb1_3 + n1_3) {
                        sb1_3 = sb1_3 + n1_3;
                        d1_3 = d1_3 + (c1_4 + 63 & -64);
                        f1_3 = f1_3 + (c1_4 + 127 & -128);
                        b1_3 = b1_3 + (c1_4 + 255 >> 8);
                        s1_3 = s1_3 + 1;
                        c1_4 = 0;
                        if (s1_3 < 12) {
                            c1_4 = expert_counts[s1_3];
                        }
                        n1_3 = (c1_4 + 256 - 1 >> 8) * 12;
                    }
                    e_3 = s1_3;
                    local_3 = rid_3 - sb1_3;
                    db_3 = d1_3;
                    fb_3 = f1_3;
                    bb_3 = b1_3;
                    cnt_3 = c1_4;
                    block_3 = local_3 / 12;
                    m_pair_3 = local_3 - block_3 * 12;
                } else {
                    while (rid_3 >= sb2_3 + n2_3) {
                        sb2_3 = sb2_3 + n2_3;
                        d2_3 = d2_3 + (c2_4 + 63 & -64);
                        f2_3 = f2_3 + (c2_4 + 127 & -128);
                        b2_3 = b2_3 + (c2_4 + 255 >> 8);
                        s2_3 = s2_3 + 1;
                        c2_4 = 0;
                        if (s2_3 < 12) {
                            c2_4 = expert_counts[s2_3];
                        }
                        n2_3 = (c2_4 + 256 - 1 >> 8) * 14;
                    }
                    e_3 = s2_3;
                    local_3 = rid_3 - sb2_3;
                    db_3 = d2_3;
                    fb_3 = f2_3;
                    bb_3 = b2_3;
                    cnt_3 = c2_4;
                    block_3 = local_3 / 14;
                    m_pair_3 = local_3 - block_3 * 14;
                }
                int valid_3 = ((cnt_3 - block_3 * 256) < (256) ? (cnt_3 - block_3 * 256) : (256));
                int nsub_3 = 2 - ((int)cluster_size >> 2);
                #pragma unroll 1
                for (int sub_3 = 0; sub_3 < nsub_3; sub_3++) {
                    int m_one_3 = m_pair_3 * 2 + (cta_rank >> 1) + sub_3;
                    if (ph_3 == 0) {
                        int n_rt_2 = valid_3 + 15 & -16;
                        int row_5 = db_3 + block_3 * 256 + cta_2 * (n_rt_2 / 2);
                        int sf_atom = (fb_3 + block_3 * 256) / 128;
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(fc1_ready) + (bb_3 + block_3))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)valid_3)) >= static_cast<unsigned int>(2147483648u));
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        int small = 0;
                        int one_atom = 0;
                        if (n_rt_2 <= 32) {
                            small = 1;
                        }
                        if (n_rt_2 <= 128) {
                            one_atom = 1;
                        }
                        unsigned int b_bytes = 16384;
                        if (small != 0) {
                            b_bytes = 2048;
                        }
                        unsigned int sfb_bytes = 4096;
                        if (one_atom != 0) {
                            sfb_bytes = 2048;
                        }
                        int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, row_5, 0);
                        row_5 = _shfl_sync_1;
                        int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, sf_atom, 0);
                        sf_atom = _shfl_sync_2;
                        if (small != 0) {
                            #pragma unroll 1
                            for (int kt_10 = 0; kt_10 < 28; kt_10++) {
                                mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes + sfb_bytes)) : "memory");
                                    tma_2d_gmem2smem_cta2(sm_b_addr + stage_2 * 16384, (&pool_b16), kt_10 * 256, row_5, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                    tma_3d_gmem2smem_cta2(sm_sfb_addr + stage_2 * 4096, (&pool_sfb1), 0, kt_10 * 4, sf_atom, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                }
                                stage_2 += 1;
                                if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                            }
                        } else if ((int)cluster_size >> 2 != 0) {
                            if (one_atom != 0) {
                                #pragma unroll 1
                                for (int kt_11 = 0; kt_11 < 28; kt_11++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes + sfb_bytes)) : "memory");
                                        int pid = cta_rank >> 1;
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3}], [%4], %5;"
                                            :: "r"(sm_b_addr + stage_2 * 16384 + (unsigned int)(pid * 8192)), "l"((&pool_bh)), "r"(kt_11 * 256), "r"(row_5 + pid * 64),
                                               "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(5 << cta_2)) : "memory");
                                        tma_3d_gmem2smem_cta2(sm_sfb_addr + stage_2 * 4096, (&pool_sfb1), 0, kt_11 * 4, sf_atom, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            } else {
                                #pragma unroll 1
                                for (int kt_12 = 0; kt_12 < 28; kt_12++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes + sfb_bytes)) : "memory");
                                        int pid_1 = cta_rank >> 1;
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3}], [%4], %5;"
                                            :: "r"(sm_b_addr + stage_2 * 16384 + (unsigned int)(pid_1 * 8192)), "l"((&pool_bh)), "r"(kt_12 * 256), "r"(row_5 + pid_1 * 64),
                                               "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(5 << cta_2)) : "memory");
                                        if ((kt_12 & 1) == (int)cta_rank >> 1) {
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(sm_sfb_addr + stage_2 * 4096 + (unsigned int)(cta_2 * 2048)), "l"((&pool_sfb1)), "r"(0), "r"(kt_12 * 4), "r"(sf_atom + cta_2),
                                                   "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(15)) : "memory");
                                        }
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            }
                        } else {
                            if (one_atom != 0) {
                                #pragma unroll 1
                                for (int kt_13 = 0; kt_13 < 28; kt_13++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes + sfb_bytes)) : "memory");
                                        tma_2d_gmem2smem_cta2(sm_b_addr + stage_2 * 16384, (&pool_b), kt_13 * 256, row_5, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                        tma_3d_gmem2smem_cta2(sm_sfb_addr + stage_2 * 4096, (&pool_sfb1), 0, kt_13 * 4, sf_atom, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            } else {
                                #pragma unroll 1
                                for (int kt_14 = 0; kt_14 < 28; kt_14++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes + sfb_bytes)) : "memory");
                                        tma_2d_gmem2smem_cta2(sm_b_addr + stage_2 * 16384, (&pool_b), kt_14 * 256, row_5, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sm_sfb_addr + stage_2 * 4096 + (unsigned int)(cta_2 * 2048)), "l"((&pool_sfb1)), "r"(0), "r"(kt_14 * 4), "r"(sf_atom + cta_2),
                                               "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3 << (cta_rank & 2))) : "memory");
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            }
                        }
                    }
                    if (ph_3 == 1) {
                        int n_rt_3 = valid_3 + 15 & -16;
                        int row_6 = db_3 + block_3 * 256 + cta_2 * (n_rt_3 / 2);
                        int sf_atom_1 = (fb_3 + block_3 * 256) / 128;
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(fc1_done) + (bb_3 + block_3))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(48)) >= static_cast<unsigned int>(2147483648u));
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        int small_1 = 0;
                        int one_atom_1 = 0;
                        if (n_rt_3 <= 32) {
                            small_1 = 1;
                        }
                        if (n_rt_3 <= 128) {
                            one_atom_1 = 1;
                        }
                        unsigned int b_bytes_1 = 16384;
                        if (small_1 != 0) {
                            b_bytes_1 = 2048;
                        }
                        unsigned int sfb_bytes_1 = 4096;
                        if (one_atom_1 != 0) {
                            sfb_bytes_1 = 2048;
                        }
                        int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, row_6, 0);
                        row_6 = _shfl_sync_3;
                        int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, sf_atom_1, 0);
                        sf_atom_1 = _shfl_sync_4;
                        if (small_1 != 0) {
                            #pragma unroll 1
                            for (int kt_15 = 0; kt_15 < 12; kt_15++) {
                                mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes_1 + sfb_bytes_1)) : "memory");
                                    tma_2d_gmem2smem_cta2(sm_b_addr + stage_2 * 16384, (&fc1_out_ld16), kt_15 * 256, row_6, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                    tma_3d_gmem2smem_cta2(sm_sfb_addr + stage_2 * 4096, (&fc1_out_sfb1), 0, kt_15 * 4, sf_atom_1, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                }
                                stage_2 += 1;
                                if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                            }
                        } else if ((int)cluster_size >> 2 != 0) {
                            if (one_atom_1 != 0) {
                                #pragma unroll 1
                                for (int kt_16 = 0; kt_16 < 12; kt_16++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes_1 + sfb_bytes_1)) : "memory");
                                        int pid_2 = cta_rank >> 1;
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3}], [%4], %5;"
                                            :: "r"(sm_b_addr + stage_2 * 16384 + (unsigned int)(pid_2 * 8192)), "l"((&fc1_out_ldh)), "r"(kt_16 * 256), "r"(row_6 + pid_2 * 64),
                                               "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(5 << cta_2)) : "memory");
                                        tma_3d_gmem2smem_cta2(sm_sfb_addr + stage_2 * 4096, (&fc1_out_sfb1), 0, kt_16 * 4, sf_atom_1, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            } else {
                                #pragma unroll 1
                                for (int kt_17 = 0; kt_17 < 12; kt_17++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes_1 + sfb_bytes_1)) : "memory");
                                        int pid_3 = cta_rank >> 1;
                                        asm volatile(
                                            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3}], [%4], %5;"
                                            :: "r"(sm_b_addr + stage_2 * 16384 + (unsigned int)(pid_3 * 8192)), "l"((&fc1_out_ldh)), "r"(kt_17 * 256), "r"(row_6 + pid_3 * 64),
                                               "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(5 << cta_2)) : "memory");
                                        if ((kt_17 & 1) == (int)cta_rank >> 1) {
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(sm_sfb_addr + stage_2 * 4096 + (unsigned int)(cta_2 * 2048)), "l"((&fc1_out_sfb1)), "r"(0), "r"(kt_17 * 4), "r"(sf_atom_1 + cta_2),
                                                   "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(15)) : "memory");
                                        }
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            }
                        } else {
                            if (one_atom_1 != 0) {
                                #pragma unroll 1
                                for (int kt_18 = 0; kt_18 < 12; kt_18++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes_1 + sfb_bytes_1)) : "memory");
                                        tma_2d_gmem2smem_cta2(sm_b_addr + stage_2 * 16384, (&fc1_out_ld), kt_18 * 256, row_6, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                        tma_3d_gmem2smem_cta2(sm_sfb_addr + stage_2 * 4096, (&fc1_out_sfb1), 0, kt_18 * 4, sf_atom_1, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            } else {
                                #pragma unroll 1
                                for (int kt_19 = 0; kt_19 < 12; kt_19++) {
                                    mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((af_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(b_bytes_1 + sfb_bytes_1)) : "memory");
                                        tma_2d_gmem2smem_cta2(sm_b_addr + stage_2 * 16384, (&fc1_out_ld), kt_19 * 256, row_6, ((af_addr + (stage_2) * 8) & 0xFEFFFFFF));
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sm_sfb_addr + stage_2 * 4096 + (unsigned int)(cta_2 * 2048)), "l"((&fc1_out_sfb1)), "r"(0), "r"(kt_19 * 4), "r"(sf_atom_1 + cta_2),
                                               "r"(((af_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3 << (cta_rank & 2))) : "memory");
                                    }
                                    stage_2 += 1;
                                    if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
                                }
                            }
                        }
                    }
                    nit_2 = nit_2 + 1;
                }
            }
            #pragma unroll 1
            for (int _s_1 = 0; _s_1 < 6; _s_1++) {
                mbarrier_wait(ae_addr + (stage_2) * 8, _phase_ae_1);
                stage_2 += 1;
                if (stage_2 == 6) { stage_2 = 0; _phase_ae_1 ^= 1; }
            }
        }
    }
    // ---- Role: store ----
    if (warp == 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // store_main
            unsigned int sst_1 = 0;
            int cta_3 = cta_rank & 1;
            int pend[1];
            pend[0] = -1;
            {
                int _poll_v_0;
                do {
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<int*>(push_table_ready)) : "memory");
                } while (_poll_v_0 < static_cast<int>(1));
            }
            unsigned int wst_4 = 0;
            int no_pend_4[1];
            int part_5 = 0;
            part_5 = part_5 + (expert_counts[0] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[1] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[2] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[3] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[4] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[5] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[6] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[7] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[8] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[9] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[10] + 256 - 1 >> 8);
            part_5 = part_5 + (expert_counts[11] + 256 - 1 >> 8);
            int total_5 = part_5 * 26;
            int ticket_4 = 0;
            int s1_4 = 0;
            int sb1_4 = 0;
            int d1_4 = 0;
            int f1_4 = 0;
            int b1_4 = 0;
            int c1_5 = expert_counts[0];
            int n1_4 = (c1_5 + 256 - 1 >> 8) * 12;
            int s2_4 = 0;
            int sb2_4 = 0;
            int d2_4 = 0;
            int f2_4 = 0;
            int b2_4 = 0;
            int c2_5 = expert_counts[0];
            int n2_4 = (c2_5 + 256 - 1 >> 8) * 14;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_sfull = 0;
            #pragma unroll 1
            for (int it_4 = 0; it_4 < total_5 + 1; it_4++) {
                if (pend[0] >= 0) {
                    uint32_t ready = mbarrier_try_wait(work_full_addr + (wst_4) * 8, _phase_work_full_4);
                    if (ready == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group 0;");
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_done) + (pend[0]))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                        pend[0] = -1;
                    }
                }
                mbarrier_wait(work_full_addr + (wst_4) * 8, _phase_work_full_4);
                ticket_4 = (int)work_cells[wst_4 * 4];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                    "}"
                    :: "r"(work_empty_addr + (wst_4) * 8), "r"(1), "r"((uint32_t)((1 + (ticket_4 >> 31)) * (2 - ((int)cluster_size >> 2)))) : "memory");
                wst_4 += 1;
                if (wst_4 == 3) { wst_4 = 0; _phase_work_full_4 ^= 1; }
                if (ticket_4 == 2147483647) {
                    break;
                }
                int ph_4 = ticket_4 >> 30;
                int rid_4 = ticket_4 & 1073741823;
                int e_4 = 0;
                int local_4 = 0;
                int db_4 = 0;
                int fb_4 = 0;
                int bb_4 = 0;
                int cnt_4 = 0;
                int block_4 = 0;
                int m_pair_4 = 0;
                if (ph_4 == 0) {
                    while (rid_4 >= sb1_4 + n1_4) {
                        sb1_4 = sb1_4 + n1_4;
                        d1_4 = d1_4 + (c1_5 + 63 & -64);
                        f1_4 = f1_4 + (c1_5 + 127 & -128);
                        b1_4 = b1_4 + (c1_5 + 255 >> 8);
                        s1_4 = s1_4 + 1;
                        c1_5 = 0;
                        if (s1_4 < 12) {
                            c1_5 = expert_counts[s1_4];
                        }
                        n1_4 = (c1_5 + 256 - 1 >> 8) * 12;
                    }
                    e_4 = s1_4;
                    local_4 = rid_4 - sb1_4;
                    db_4 = d1_4;
                    fb_4 = f1_4;
                    bb_4 = b1_4;
                    cnt_4 = c1_5;
                    block_4 = local_4 / 12;
                    m_pair_4 = local_4 - block_4 * 12;
                } else {
                    while (rid_4 >= sb2_4 + n2_4) {
                        sb2_4 = sb2_4 + n2_4;
                        d2_4 = d2_4 + (c2_5 + 63 & -64);
                        f2_4 = f2_4 + (c2_5 + 127 & -128);
                        b2_4 = b2_4 + (c2_5 + 255 >> 8);
                        s2_4 = s2_4 + 1;
                        c2_5 = 0;
                        if (s2_4 < 12) {
                            c2_5 = expert_counts[s2_4];
                        }
                        n2_4 = (c2_5 + 256 - 1 >> 8) * 14;
                    }
                    e_4 = s2_4;
                    local_4 = rid_4 - sb2_4;
                    db_4 = d2_4;
                    fb_4 = f2_4;
                    bb_4 = b2_4;
                    cnt_4 = c2_5;
                    block_4 = local_4 / 14;
                    m_pair_4 = local_4 - block_4 * 14;
                }
                int valid_4 = ((cnt_4 - block_4 * 256) < (256) ? (cnt_4 - block_4 * 256) : (256));
                int nsub_4 = 2 - ((int)cluster_size >> 2);
                #pragma unroll 1
                for (int sub_4 = 0; sub_4 < nsub_4; sub_4++) {
                    int m_one_4 = m_pair_4 * 2 + (cta_rank >> 1) + sub_4;
                    if (ph_4 == 0) {
                        int m_cta_1 = m_one_4 * 2 + cta_3;
                        #pragma unroll 1
                        for (int s0_1 = 0; s0_1 < valid_4; s0_1 += 64) {
                            if (pend[0] >= 0) {
                                uint32_t ready_1 = mbarrier_try_wait(sfull_addr + (sst_1) * 8, _phase_sfull);
                                if (ready_1 == 0) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group 0;");
                                        asm volatile("fence.proxy.async.global;" ::: "memory");
                                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_done) + (pend[0]))), "r"(static_cast<unsigned int>(1)) : "memory");
                                    }
                                    pend[0] = -1;
                                }
                            }
                            mbarrier_wait(sfull_addr + (sst_1) * 8, _phase_sfull);
                            if (elect_sync()) {
                                tma_store_2d((&fc1_out_st), m_cta_1 * 32, db_4 + block_4 * 256 + s0_1, sm_fc1_addr + sst_1 * 2048);
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                                mbarrier_arrive(sempty_addr + (sst_1) * 8);
                            }
                            sst_1 += 1;
                            if (sst_1 == 4) { sst_1 = 0; _phase_sfull ^= 1; }
                        }
                        if (elect_sync()) {
                            #pragma unroll 1
                            for (int _g = valid_4 + 63 >> 6; _g < 4; _g++) {
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            if (pend[0] >= 0) {
                                asm volatile("cp.async.bulk.wait_group 4;");
                                asm volatile("fence.proxy.async.global;" ::: "memory");
                                asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_done) + (pend[0]))), "r"(static_cast<unsigned int>(1)) : "memory");
                            }
                        }
                        pend[0] = bb_4 + block_4;
                    }
                }
            }
            if (pend[0] >= 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_done) + (pend[0]))), "r"(static_cast<unsigned int>(1)) : "memory");
                }
                pend[0] = -1;
            }
            asm volatile("barrier.sync 4, 288;" ::: "memory");
        }
    }
    // ---- Role: dall ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // dall_main
            int dwv = (int)warp - 8;
            int pk = dwv * 2;
            unsigned int stage_addr = sm_pull_addr + (unsigned int)(dwv * 8192);
            {
                int _poll_v_0;
                do {
                    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<int*>(push_table_ready)) : "memory");
                } while (_poll_v_0 < static_cast<int>(1));
            }
            int meta_launch = metadata_ready[0];
            int meta_copy = meta_launch & 1;
            if (lane < mega_pg_world) {
                {
                    int _poll_v_1;
                    do {
                        asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(_poll_v_1) : "l"((reinterpret_cast<int*>(x_ready) + (lane))) : "memory");
                    } while (_poll_v_1 < static_cast<int>(meta_launch + 1));
                }
            }
            __syncwarp();
            int gw = bid * 4 + dwv;
            int gcount = num_bids * 4;
            int rows = 0;
            rows = rows + expert_counts[0];
            rows = rows + expert_counts[1];
            rows = rows + expert_counts[2];
            rows = rows + expert_counts[3];
            rows = rows + expert_counts[4];
            rows = rows + expert_counts[5];
            rows = rows + expert_counts[6];
            rows = rows + expert_counts[7];
            rows = rows + expert_counts[8];
            rows = rows + expert_counts[9];
            rows = rows + expert_counts[10];
            rows = rows + expert_counts[11];
            int nrows = 0;
            if (gw < rows) {
                nrows = (rows - gw + gcount - 1) / gcount;
            }
            int e_5 = 0;
            int begin = 0;
            int end = expert_counts[0];
            int sf_begin = 0;
            int block_begin = 0;
            int pw = dwv * 2;
            int slot_pool[2];
            int slot_sf[2];
            int slot_block[2];
            unsigned int _phase_pull_all = 0;
            #pragma unroll 1
            for (int k = 0; k < nrows; k++) {
                int cur_pool = 0;
                int cur_sf = 0;
                int cur_block = 0;
                if (k < nrows) {
                    int d = gw + k * gcount;
                    while (d >= end) {
                        int count_e = expert_counts[e_5];
                        sf_begin = sf_begin + (count_e + 127 & -128);
                        block_begin = block_begin + (count_e + 255 >> 8);
                        begin = end;
                        e_5 = e_5 + 1;
                        end = end + expert_counts[e_5];
                    }
                    int t_1 = d - begin;
                    cur_pool = pool_expert_base[e_5] + t_1;
                    cur_sf = sf_begin + t_1;
                    cur_block = block_begin + (t_1 >> 8);
                    long long frec = token_src_metadata[(cur_pool >> 4 << 5) + (meta_copy << 4) + (cur_pool & 15)];
                    unsigned int fscore = __as_u32(fc1_topk_scores[(cur_pool >> 5 << 6) + (meta_copy << 5) + (cur_pool & 31)]);
                    if ((unsigned int)(frec & 4294967295ll) == 2147483648u || (unsigned int)((unsigned long long)frec >> 32ull) == 2147483648u || fscore == 2147483648u) {
                        uint32_t three[3];
                        do {
                            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(three[0]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cur_pool >> 4 << 5) + (meta_copy << 4) + (cur_pool & 15)))) : "memory");
                            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(three[1]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cur_pool >> 4 << 5) + (meta_copy << 4) + (cur_pool & 15))) + 1) : "memory");
                            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(three[2]) : "l"(reinterpret_cast<unsigned int*>(fc1_topk_scores + ((cur_pool >> 5 << 6) + (meta_copy << 5) + (cur_pool & 31)))) : "memory");
                        } while (
                            (three[0] == 0x80000000u) || (three[1] == 0x80000000u) || (three[2] == 0x80000000u)
                        );
                        frec = (long long)((unsigned long long)(unsigned int)three[1] << 32ull | (unsigned long long)(unsigned int)three[0]);
                    }
                    int lo = (int)(frec & 4294967295ll);
                    int src_token = lo & 65535;
                    int src_rank = cake_floor_mod<int>(lo >> 24 & 255, mega_pg_world);
                    int ks = k % 2;
                    if (ks == 0) {
                        slot_pool[0] = cur_pool;
                        slot_sf[0] = cur_sf;
                        slot_block[0] = cur_block;
                    }
                    if (ks == 1) {
                        slot_pool[1] = cur_pool;
                        slot_sf[1] = cur_sf;
                        slot_block[1] = cur_block;
                    }
                    unsigned int slot = stage_addr + (unsigned int)(ks * 4032);
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(pull_all_addr + (pk) * 8, 4032);
                        cp_async_bulk_gmem2smem(slot, reinterpret_cast<uint8_t*>(activation_peers.ptrs[src_rank]) + (unsigned long long)((unsigned long long)src_token * 3584ull), 3584, pull_all_addr + (pk) * 8);
                        cp_async_bulk_gmem2smem(slot + 3584, reinterpret_cast<uint8_t*>(activation_sf_peers.ptrs[src_rank]) + (unsigned long long)((unsigned long long)src_token * 448ull), 448, pull_all_addr + (pk) * 8);
                    }
                    pk += 1;
                    if (pk == (dwv * 2 + 2)) { pk = dwv * 2; }
                }
                if (k >= 0) {
                    mbarrier_wait(pull_all_addr + (pw) * 8, _phase_pull_all);
                    pw += 1;
                    if (pw == (dwv * 2 + 2)) { pw = dwv * 2; _phase_pull_all ^= 1; }
                    int ps = k % 2;
                    int prev_pool = slot_pool[0];
                    int prev_sf = slot_sf[0];
                    int prev_block = slot_block[0];
                    if (ps == 1) {
                        prev_pool = slot_pool[1];
                        prev_sf = slot_sf[1];
                        prev_block = slot_block[1];
                    }
                    unsigned int pslot = stage_addr + (unsigned int)(ps * 4032);
                    if (elect_sync()) {
                        {
                            void* _cpbulk_dst_2 = reinterpret_cast<void*>(pool_data + (unsigned long long)((unsigned long long)prev_pool * 3584ull));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_2), "r"(pslot), "r"((uint32_t)(3584))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    int w_0 = lane;
                    if (w_0 < 112) {
                        unsigned int word_0[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word_0[0])) : "r"(pslot + (unsigned int)(3584 + w_0 * 4)));
                        int sf_offset_2 = ((prev_sf >> 7) * 112 + w_0) * 512 + (prev_sf & 31) * 16 + (prev_sf >> 5 & 3) * 4;
                        reinterpret_cast<unsigned int*>(pool_sf + sf_offset_2)[0] = word_0[0];
                    }
                    int w_1 = 32 + lane;
                    if (w_1 < 112) {
                        unsigned int word_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word_1[0])) : "r"(pslot + (unsigned int)(3584 + w_1 * 4)));
                        int sf_offset_3 = ((prev_sf >> 7) * 112 + w_1) * 512 + (prev_sf & 31) * 16 + (prev_sf >> 5 & 3) * 4;
                        reinterpret_cast<unsigned int*>(pool_sf + sf_offset_3)[0] = word_1[0];
                    }
                    int w_2 = 64 + lane;
                    if (w_2 < 112) {
                        unsigned int word_2[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word_2[0])) : "r"(pslot + (unsigned int)(3584 + w_2 * 4)));
                        int sf_offset_4 = ((prev_sf >> 7) * 112 + w_2) * 512 + (prev_sf & 31) * 16 + (prev_sf >> 5 & 3) * 4;
                        reinterpret_cast<unsigned int*>(pool_sf + sf_offset_4)[0] = word_2[0];
                    }
                    int w_3 = 96 + lane;
                    if (w_3 < 112) {
                        unsigned int word_3[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word_3[0])) : "r"(pslot + (unsigned int)(3584 + w_3 * 4)));
                        int sf_offset_5 = ((prev_sf >> 7) * 112 + w_3) * 512 + (prev_sf & 31) * 16 + (prev_sf >> 5 & 3) * 4;
                        reinterpret_cast<unsigned int*>(pool_sf + sf_offset_5)[0] = word_3[0];
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(fc1_ready) + (prev_block))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
            unsigned int _phase_cfill_all = 0;
            if (dwv < 3) {
                int me_4 = mega_pg_rank;
                int cdw = (int)warp - 8;
                int fi_4 = cdw * 2;
                int fw_4 = cdw * 2;
                int nf_4 = 0;
                int nw_4 = 0;
                float dacc_4[32];
                int cd_copy_4 = metadata_ready[0] & 1;
                int nxt_4 = gw - gcount + gcount;
                int got_5 = 0;
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_3 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                    got_5 = (int)_atomic_fetch_add_3;
                }
                int _shfl_sync_4_1 = __shfl_sync(0xFFFFFFFF, got_5, 0);
                nxt_4 = _shfl_sync_4_1;
                int dnext_4 = nxt_4;
                int dprefix_4 = 0;
                int cbegin_4 = 0;
                int cblk_4 = 0;
                int bprefix_4 = 0;
                int tblocks_4 = 0;
                tblocks_4 = tblocks_4 + (expert_counts[0] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[1] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[2] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[3] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[4] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[5] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[6] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[7] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[8] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[9] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[10] + 256 - 1 >> 8);
                tblocks_4 = tblocks_4 + (expert_counts[11] + 256 - 1 >> 8);
                #pragma unroll 1
                for (int ce__4 = 0; ce__4 < 12; ce__4++) {
                    int ccount_4 = expert_counts[ce__4];
                    int cend_4 = cbegin_4 + ccount_4;
                    int cbase_4 = pool_expert_base[ce__4];
                    while (dnext_4 < cend_4) {
                        long long grec_4 = token_src_metadata[(cbase_4 + (dnext_4 - cbegin_4) >> 4 << 5) + (cd_copy_4 << 4) + (cbase_4 + (dnext_4 - cbegin_4) & 15)];
                        if ((unsigned int)(grec_4 & 4294967295ll) == 2147483648u || (unsigned int)((unsigned long long)grec_4 >> 32ull) == 2147483648u) {
                            uint32_t both_4[2];
                            do {
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_4[0]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_4 + (dnext_4 - cbegin_4) >> 4 << 5) + (cd_copy_4 << 4) + (cbase_4 + (dnext_4 - cbegin_4) & 15)))) : "memory");
                                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(both_4[1]) : "l"(reinterpret_cast<unsigned int*>(token_src_metadata + ((cbase_4 + (dnext_4 - cbegin_4) >> 4 << 5) + (cd_copy_4 << 4) + (cbase_4 + (dnext_4 - cbegin_4) & 15))) + 1) : "memory");
                            } while (
                                (both_4[0] == 0x80000000u) || (both_4[1] == 0x80000000u)
                            );
                            grec_4 = (long long)((unsigned long long)(unsigned int)both_4[1] << 32ull | (unsigned long long)(unsigned int)both_4[0]);
                        }
                        int dlo_5 = (int)(grec_4 & 4294967295ll);
                        int dhi_5 = (int)(grec_4 >> 32ll);
                        int dcn_4 = (dhi_5 >> 8 & 255) + 1;
                        int dwork_4 = 0;
                        if (dcn_4 > 1) {
                            if ((dhi_5 & 255) == dcn_4 - 1) {
                                dwork_4 = 1;
                            }
                        }
                        if (dwork_4 != 0) {
                            unsigned int dsleep_4 = 500;
                            int gblock_4 = cblk_4 + (dnext_4 - cbegin_4 >> 8);
                            while (bprefix_4 <= gblock_4) {
                                int bbase_4 = bprefix_4 & -32;
                                unsigned int bfirst_4 = 32;
                                int pb_35 = bbase_4 + lane;
                                if (pb_35 < tblocks_4) {
                                    unsigned int _bval_4;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_bval_4) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pb_35)) : "memory");
                                    if ((unsigned int)_bval_4 < 56) {
                                        bfirst_4 = (unsigned int)lane;
                                    }
                                }
                                unsigned int bend_lane_4;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(bend_lane_4) : "r"(bfirst_4));
                                int _max_5 = ((bprefix_4) > (bbase_4 + (int)bend_lane_4) ? (bprefix_4) : (bbase_4 + (int)bend_lane_4));
                                bprefix_4 = _max_5;
                                if (bprefix_4 <= gblock_4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_4));
                                    unsigned int _min_6 = ((dsleep_4 * 2) < (1000) ? (dsleep_4 * 2) : (1000));
                                    dsleep_4 = _min_6;
                                }
                            }
                            while (dprefix_4 <= -1) {
                                int pbase_4 = dprefix_4 & -32;
                                int pex_4 = pbase_4 + lane;
                                unsigned int pfirst_4 = 32;
                                if (pex_4 < 12) {
                                    int pcount_4 = expert_counts[pex_4];
                                    unsigned int _pval_4;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_pval_4) : "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done) + pex_4)) : "memory");
                                    if ((unsigned int)_pval_4 < (unsigned int)((pcount_4 + 256 - 1 >> 8) * 56)) {
                                        pfirst_4 = (unsigned int)lane;
                                    }
                                }
                                unsigned int pend_lane_4;
                                asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(pend_lane_4) : "r"(pfirst_4));
                                int _max_7 = ((dprefix_4) > (pbase_4 + (int)pend_lane_4) ? (dprefix_4) : (pbase_4 + (int)pend_lane_4));
                                dprefix_4 = _max_7;
                                if (dprefix_4 <= ce__4) {
                                    asm volatile("nanosleep.u32 %0;" :: "r"(dsleep_4));
                                    unsigned int _min_8 = ((dsleep_4 * 2) < (1000) ? (dsleep_4 * 2) : (1000));
                                    dsleep_4 = _min_8;
                                }
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            int dsrc_4 = cake_floor_mod<int>(dlo_5 >> 24 & 255, mega_pg_world);
                            int dtok_4 = dlo_5 & 65535;
                            unsigned long long dblock_4 = (unsigned long long)dedup_rows + (unsigned long long)(dsrc_4 * dedup_cap + dtok_4) * 86016ull;
                            int dpeer_4 = dsrc_4;
                            unsigned long long ddst_4 = (unsigned long long)pre_reduced_peers.ptrs[dpeer_4] + (unsigned long long)((dtok_4 * 8 + me_4) * 7168 * 2);
                            int dper_4 = dcn_4 + 1 >> 1;
                            int dfills_4 = 7 * dper_4;
                            __syncwarp();
                            int d0n_4 = ((dcn_4) < (2) ? (dcn_4) : (2));
                            int dpair_4 = dsrc_4 * dedup_cap + dtok_4;
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive_expect_tx(cfill_all_addr + (fi_4) * 8, d0n_4 * 2048);
                                asm volatile(
                                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1], %2, [%3], %4;"
                                    :: "r"(stage_addr + (unsigned int)((nf_4 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_4)), "r"((unsigned)(d0n_4 * 2048)), "r"(cfill_all_addr + (fi_4) * 8),
                                       "l"(0x12F0000000000000ULL) : "memory");
                            }
                            fi_4 += 1;
                            if (fi_4 == (cdw * 2 + 2)) { fi_4 = cdw * 2; }
                            nf_4 = nf_4 + 1;
                            #pragma unroll 1
                            for (int f_4 = 0; f_4 < dfills_4; f_4++) {
                                int fc_4 = f_4 / dper_4;
                                int fj_4 = f_4 - fc_4 * dper_4;
                                int fn_4 = ((dcn_4 - fj_4 * 2) < (2) ? (dcn_4 - fj_4 * 2) : (2));
                                int fill_live_4 = 0;
                                if (f_4 + 1 < dfills_4) {
                                    fill_live_4 = 1;
                                }
                                int gc__4 = (f_4 + 1) / dper_4;
                                int gj_4 = f_4 + 1 - gc__4 * dper_4;
                                int _min_9 = ((dcn_4 - gj_4 * 2) < (2) ? (dcn_4 - gj_4 * 2) : (2));
                                int gn_4 = _min_9 * fill_live_4;
                                if (elect_sync()) {
                                    if (fill_live_4 != 0) {
                                        asm volatile("cp.async.bulk.wait_group.read 0;");
                                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    }
                                    mbarrier_arrive_expect_tx(cfill_all_addr + (fi_4) * 8, gn_4 * 2048);
                                    if (fill_live_4 != 0) {
                                        asm volatile(
                                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1], %2, [%3], %4;"
                                            :: "r"(stage_addr + (unsigned int)((nf_4 & 1) * 4096)), "l"(reinterpret_cast<uint8_t*>(dblock_4 + (unsigned long long)((gc__4 * 6 + gj_4 * 2) * 2048))), "r"((unsigned)(gn_4 * 2048)), "r"(cfill_all_addr + (fi_4) * 8),
                                               "l"(0x12F0000000000000ULL) : "memory");
                                    }
                                }
                                fi_4 += 1;
                                if (fi_4 == (cdw * 2 + 2)) { fi_4 = cdw * 2; }
                                nf_4 = nf_4 + 1;
                                if (fj_4 == 0) {
                                    dacc_4[0] = 0.0f;
                                    dacc_4[1] = 0.0f;
                                    dacc_4[2] = 0.0f;
                                    dacc_4[3] = 0.0f;
                                    dacc_4[4] = 0.0f;
                                    dacc_4[5] = 0.0f;
                                    dacc_4[6] = 0.0f;
                                    dacc_4[7] = 0.0f;
                                    dacc_4[8] = 0.0f;
                                    dacc_4[9] = 0.0f;
                                    dacc_4[10] = 0.0f;
                                    dacc_4[11] = 0.0f;
                                    dacc_4[12] = 0.0f;
                                    dacc_4[13] = 0.0f;
                                    dacc_4[14] = 0.0f;
                                    dacc_4[15] = 0.0f;
                                    dacc_4[16] = 0.0f;
                                    dacc_4[17] = 0.0f;
                                    dacc_4[18] = 0.0f;
                                    dacc_4[19] = 0.0f;
                                    dacc_4[20] = 0.0f;
                                    dacc_4[21] = 0.0f;
                                    dacc_4[22] = 0.0f;
                                    dacc_4[23] = 0.0f;
                                    dacc_4[24] = 0.0f;
                                    dacc_4[25] = 0.0f;
                                    dacc_4[26] = 0.0f;
                                    dacc_4[27] = 0.0f;
                                    dacc_4[28] = 0.0f;
                                    dacc_4[29] = 0.0f;
                                    dacc_4[30] = 0.0f;
                                    dacc_4[31] = 0.0f;
                                }
                                mbarrier_wait(cfill_all_addr + (fw_4) * 8, _phase_cfill_all);
                                unsigned int dbuf_4 = stage_addr + (unsigned int)((nw_4 & 1) * 4096);
                                if (0 < fn_4) {
                                    unsigned int dw__32[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__32[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__32[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)(lane * 16)));
                                    float dsum_128[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_128[0]), "=&f"(dsum_128[1]) : "r"(dw__32[0]), "f"(dacc_4[0]), "f"(dacc_4[1]));
                                    dacc_4[0] = dsum_128[0];
                                    dacc_4[1] = dsum_128[1];
                                    float dsum_129[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_129[0]), "=&f"(dsum_129[1]) : "r"(dw__32[1]), "f"(dacc_4[2]), "f"(dacc_4[3]));
                                    dacc_4[2] = dsum_129[0];
                                    dacc_4[3] = dsum_129[1];
                                    float dsum_130[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_130[0]), "=&f"(dsum_130[1]) : "r"(dw__32[2]), "f"(dacc_4[4]), "f"(dacc_4[5]));
                                    dacc_4[4] = dsum_130[0];
                                    dacc_4[5] = dsum_130[1];
                                    float dsum_131[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_131[0]), "=&f"(dsum_131[1]) : "r"(dw__32[3]), "f"(dacc_4[6]), "f"(dacc_4[7]));
                                    dacc_4[6] = dsum_131[0];
                                    dacc_4[7] = dsum_131[1];
                                    unsigned int dw__33[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__33[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__33[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)((32 + lane) * 16)));
                                    float dsum_132[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_132[0]), "=&f"(dsum_132[1]) : "r"(dw__33[0]), "f"(dacc_4[8]), "f"(dacc_4[9]));
                                    dacc_4[8] = dsum_132[0];
                                    dacc_4[9] = dsum_132[1];
                                    float dsum_133[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_133[0]), "=&f"(dsum_133[1]) : "r"(dw__33[1]), "f"(dacc_4[10]), "f"(dacc_4[11]));
                                    dacc_4[10] = dsum_133[0];
                                    dacc_4[11] = dsum_133[1];
                                    float dsum_134[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_134[0]), "=&f"(dsum_134[1]) : "r"(dw__33[2]), "f"(dacc_4[12]), "f"(dacc_4[13]));
                                    dacc_4[12] = dsum_134[0];
                                    dacc_4[13] = dsum_134[1];
                                    float dsum_135[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_135[0]), "=&f"(dsum_135[1]) : "r"(dw__33[3]), "f"(dacc_4[14]), "f"(dacc_4[15]));
                                    dacc_4[14] = dsum_135[0];
                                    dacc_4[15] = dsum_135[1];
                                    unsigned int dw__34[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__34[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__34[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)((64 + lane) * 16)));
                                    float dsum_136[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_136[0]), "=&f"(dsum_136[1]) : "r"(dw__34[0]), "f"(dacc_4[16]), "f"(dacc_4[17]));
                                    dacc_4[16] = dsum_136[0];
                                    dacc_4[17] = dsum_136[1];
                                    float dsum_137[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_137[0]), "=&f"(dsum_137[1]) : "r"(dw__34[1]), "f"(dacc_4[18]), "f"(dacc_4[19]));
                                    dacc_4[18] = dsum_137[0];
                                    dacc_4[19] = dsum_137[1];
                                    float dsum_138[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_138[0]), "=&f"(dsum_138[1]) : "r"(dw__34[2]), "f"(dacc_4[20]), "f"(dacc_4[21]));
                                    dacc_4[20] = dsum_138[0];
                                    dacc_4[21] = dsum_138[1];
                                    float dsum_139[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_139[0]), "=&f"(dsum_139[1]) : "r"(dw__34[3]), "f"(dacc_4[22]), "f"(dacc_4[23]));
                                    dacc_4[22] = dsum_139[0];
                                    dacc_4[23] = dsum_139[1];
                                    unsigned int dw__35[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__35[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__35[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)((96 + lane) * 16)));
                                    float dsum_140[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_140[0]), "=&f"(dsum_140[1]) : "r"(dw__35[0]), "f"(dacc_4[24]), "f"(dacc_4[25]));
                                    dacc_4[24] = dsum_140[0];
                                    dacc_4[25] = dsum_140[1];
                                    float dsum_141[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_141[0]), "=&f"(dsum_141[1]) : "r"(dw__35[1]), "f"(dacc_4[26]), "f"(dacc_4[27]));
                                    dacc_4[26] = dsum_141[0];
                                    dacc_4[27] = dsum_141[1];
                                    float dsum_142[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_142[0]), "=&f"(dsum_142[1]) : "r"(dw__35[2]), "f"(dacc_4[28]), "f"(dacc_4[29]));
                                    dacc_4[28] = dsum_142[0];
                                    dacc_4[29] = dsum_142[1];
                                    float dsum_143[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_143[0]), "=&f"(dsum_143[1]) : "r"(dw__35[3]), "f"(dacc_4[30]), "f"(dacc_4[31]));
                                    dacc_4[30] = dsum_143[0];
                                    dacc_4[31] = dsum_143[1];
                                }
                                if (1 < fn_4) {
                                    unsigned int dw__36[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__36[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__36[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)(2048 + lane * 16)));
                                    float dsum_144[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_144[0]), "=&f"(dsum_144[1]) : "r"(dw__36[0]), "f"(dacc_4[0]), "f"(dacc_4[1]));
                                    dacc_4[0] = dsum_144[0];
                                    dacc_4[1] = dsum_144[1];
                                    float dsum_145[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_145[0]), "=&f"(dsum_145[1]) : "r"(dw__36[1]), "f"(dacc_4[2]), "f"(dacc_4[3]));
                                    dacc_4[2] = dsum_145[0];
                                    dacc_4[3] = dsum_145[1];
                                    float dsum_146[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_146[0]), "=&f"(dsum_146[1]) : "r"(dw__36[2]), "f"(dacc_4[4]), "f"(dacc_4[5]));
                                    dacc_4[4] = dsum_146[0];
                                    dacc_4[5] = dsum_146[1];
                                    float dsum_147[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_147[0]), "=&f"(dsum_147[1]) : "r"(dw__36[3]), "f"(dacc_4[6]), "f"(dacc_4[7]));
                                    dacc_4[6] = dsum_147[0];
                                    dacc_4[7] = dsum_147[1];
                                    unsigned int dw__37[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__37[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__37[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)(2048 + (32 + lane) * 16)));
                                    float dsum_148[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_148[0]), "=&f"(dsum_148[1]) : "r"(dw__37[0]), "f"(dacc_4[8]), "f"(dacc_4[9]));
                                    dacc_4[8] = dsum_148[0];
                                    dacc_4[9] = dsum_148[1];
                                    float dsum_149[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_149[0]), "=&f"(dsum_149[1]) : "r"(dw__37[1]), "f"(dacc_4[10]), "f"(dacc_4[11]));
                                    dacc_4[10] = dsum_149[0];
                                    dacc_4[11] = dsum_149[1];
                                    float dsum_150[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_150[0]), "=&f"(dsum_150[1]) : "r"(dw__37[2]), "f"(dacc_4[12]), "f"(dacc_4[13]));
                                    dacc_4[12] = dsum_150[0];
                                    dacc_4[13] = dsum_150[1];
                                    float dsum_151[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_151[0]), "=&f"(dsum_151[1]) : "r"(dw__37[3]), "f"(dacc_4[14]), "f"(dacc_4[15]));
                                    dacc_4[14] = dsum_151[0];
                                    dacc_4[15] = dsum_151[1];
                                    unsigned int dw__38[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__38[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__38[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)(2048 + (64 + lane) * 16)));
                                    float dsum_152[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_152[0]), "=&f"(dsum_152[1]) : "r"(dw__38[0]), "f"(dacc_4[16]), "f"(dacc_4[17]));
                                    dacc_4[16] = dsum_152[0];
                                    dacc_4[17] = dsum_152[1];
                                    float dsum_153[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_153[0]), "=&f"(dsum_153[1]) : "r"(dw__38[1]), "f"(dacc_4[18]), "f"(dacc_4[19]));
                                    dacc_4[18] = dsum_153[0];
                                    dacc_4[19] = dsum_153[1];
                                    float dsum_154[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_154[0]), "=&f"(dsum_154[1]) : "r"(dw__38[2]), "f"(dacc_4[20]), "f"(dacc_4[21]));
                                    dacc_4[20] = dsum_154[0];
                                    dacc_4[21] = dsum_154[1];
                                    float dsum_155[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_155[0]), "=&f"(dsum_155[1]) : "r"(dw__38[3]), "f"(dacc_4[22]), "f"(dacc_4[23]));
                                    dacc_4[22] = dsum_155[0];
                                    dacc_4[23] = dsum_155[1];
                                    unsigned int dw__39[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&dw__39[0])), "=r"(*reinterpret_cast<uint32_t*>(&dw__39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dw__39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dw__39[(0) + 3]))
                                        : "r"(dbuf_4 + (unsigned int)(2048 + (96 + lane) * 16)));
                                    float dsum_156[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_156[0]), "=&f"(dsum_156[1]) : "r"(dw__39[0]), "f"(dacc_4[24]), "f"(dacc_4[25]));
                                    dacc_4[24] = dsum_156[0];
                                    dacc_4[25] = dsum_156[1];
                                    float dsum_157[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_157[0]), "=&f"(dsum_157[1]) : "r"(dw__39[1]), "f"(dacc_4[26]), "f"(dacc_4[27]));
                                    dacc_4[26] = dsum_157[0];
                                    dacc_4[27] = dsum_157[1];
                                    float dsum_158[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_158[0]), "=&f"(dsum_158[1]) : "r"(dw__39[2]), "f"(dacc_4[28]), "f"(dacc_4[29]));
                                    dacc_4[28] = dsum_158[0];
                                    dacc_4[29] = dsum_158[1];
                                    float dsum_159[2];
                                    asm(
                                        "{\n\t"
                                        ".reg .b64 c, d;\n\t"
                                        "mov.b64 c, {%3, %4};\n\t"
                                        "add.rn.f32x2.bf16x2.f32x2 d, %2, c;\n\t"
                                        "mov.b64 {%0, %1}, d;\n\t"
                                        "}\n"
                                        : "=&f"(dsum_159[0]), "=&f"(dsum_159[1]) : "r"(dw__39[3]), "f"(dacc_4[30]), "f"(dacc_4[31]));
                                    dacc_4[30] = dsum_159[0];
                                    dacc_4[31] = dsum_159[1];
                                }
                                fw_4 += 1;
                                if (fw_4 == (cdw * 2 + 2)) { fw_4 = cdw * 2; _phase_cfill_all ^= 1; }
                                nw_4 = nw_4 + 1;
                                if (fj_4 == dper_4 - 1) {
                                    unsigned int dpk_16[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_4[_lp*2 + 0], dacc_4[_lp*2+1 + 0]));
                                        dpk_16[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_4 + (unsigned int)(lane * 16)), "r"(dpk_16[0]), "r"(dpk_16[1]), "r"(dpk_16[2]), "r"(dpk_16[3]) : "memory");
                                    unsigned int dpk_17[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_4[_lp*2 + 8], dacc_4[_lp*2+1 + 8]));
                                        dpk_17[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_4 + (unsigned int)((32 + lane) * 16)), "r"(dpk_17[0]), "r"(dpk_17[1]), "r"(dpk_17[2]), "r"(dpk_17[3]) : "memory");
                                    unsigned int dpk_18[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_4[_lp*2 + 16], dacc_4[_lp*2+1 + 16]));
                                        dpk_18[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_4 + (unsigned int)((64 + lane) * 16)), "r"(dpk_18[0]), "r"(dpk_18[1]), "r"(dpk_18[2]), "r"(dpk_18[3]) : "memory");
                                    unsigned int dpk_19[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dacc_4[_lp*2 + 24], dacc_4[_lp*2+1 + 24]));
                                        dpk_19[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dbuf_4 + (unsigned int)((96 + lane) * 16)), "r"(dpk_19[0]), "r"(dpk_19[1]), "r"(dpk_19[2]), "r"(dpk_19[3]) : "memory");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (elect_sync()) {
                                        {
                                            void* _cpbulk_dst_10 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(ddst_4 + (unsigned long long)(fc_4 * 2048)));
                                            asm volatile(
                                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                                :: "l"(_cpbulk_dst_10), "r"(dbuf_4), "r"((uint32_t)(2048))
                                                : "memory");
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                                __syncwarp();
                            }
                            mbarrier_wait(cfill_all_addr + (fw_4) * 8, _phase_cfill_all);
                            fw_4 += 1;
                            if (fw_4 == (cdw * 2 + 2)) { fw_4 = cdw * 2; _phase_cfill_all ^= 1; }
                            nw_4 = nw_4 + 1;
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                            __syncwarp();
                        }
                        int nxt_4_1 = dnext_4 + gcount;
                        int got_4_1 = 0;
                        if (elect_sync()) {
                            unsigned int _atomic_fetch_add_11 = atomicAdd(&reinterpret_cast<unsigned int*>(sync_words)[2], 1);
                            got_4_1 = (int)_atomic_fetch_add_11;
                        }
                        int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, got_4_1, 0);
                        nxt_4_1 = _shfl_sync_12;
                        dnext_4 = nxt_4_1;
                    }
                    cblk_4 = cblk_4 + (ccount_4 + 256 - 1 >> 8);
                    cbegin_4 = cend_4;
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
            int stale_rows = metadata_ready[1 + (meta_copy ^ 1)];
            #pragma unroll 1
            for (int r = gw * 32 + lane; r < stale_rows; r += gcount * 32) {
                token_src_metadata[(r >> 4 << 5) + ((meta_copy ^ 1) << 4) + (r & 15)] = -9223372034707292160ll;
                reinterpret_cast<unsigned int*>(fc1_topk_scores)[(r >> 5 << 6) + ((meta_copy ^ 1) << 5) + (r & 31)] = 2147483648u;
            }
            asm volatile("barrier.sync 4, 288;" ::: "memory");
            int dw0 = 0;
            if (dwv == 0) {
                dw0 = 1;
            }
            if (dw0 != 0) {
                if (elect_sync()) {
                    asm volatile("fence.release.sys;" ::: "memory");
                    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(sync_words)), "r"(static_cast<unsigned int>(1)) : "memory");
                }
            }
            if (bid == 0) {
                if (dw0 != 0) {
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(sync_words) + (0))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)num_ctas)) >= static_cast<unsigned int>(2147483648u));
                        }
                    }
                    __syncwarp();
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
                int t0 = dwv * 32 + lane;
                #pragma unroll 1
                for (int i = t0; i < block_words; i += 128) {
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc1_ready)) + (i))), "r"(static_cast<unsigned int>(0)) : "memory");
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc1_done)) + (i))), "r"(static_cast<unsigned int>(0)) : "memory");
                }
                #pragma unroll 1
                for (int i_1 = t0; i_1 < e_pad; i_1 += 128) {
                    asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sizes)) + (i_1))), "r"(static_cast<unsigned int>(0)) : "memory");
                }
                #pragma unroll 1
                for (int i_2 = t0; i_2 < size_counter_words; i_2 += 128) {
                    router_size_counter[i_2] = 0;
                }
                if (t0 == 0) {
                    asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(metadata_ready))), "r"(static_cast<unsigned int>((unsigned int)(meta_launch + 1))) : "memory");
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(metadata_ready)) + (1 + meta_copy))), "r"(static_cast<unsigned int>((unsigned int)(pool_expert_base[11] + expert_counts[11]))) : "memory");
                }
                #pragma unroll 1
                for (int i_3 = t0; i_3 < block_words; i_3 += 128) {
                    asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(fc2_done)) + (i_3))), "r"(static_cast<unsigned int>(0)) : "memory");
                }
                if (dw0 != 0) {
                    if (elect_sync()) {
                        asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sizes_ready))), "r"(static_cast<unsigned int>(0)) : "memory");
                        sorted_metadata_ready[0] = 0;
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(push_table_ready))), "r"(static_cast<unsigned int>(0)) : "memory");
                        router_histogram_done[0] = 0;
                        source_base_ready[0] = 0;
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (0))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (1))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (4))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (5))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (6))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (7))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (3))), "r"(static_cast<unsigned int>(0)) : "memory");
                        asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sync_words)) + (2))), "r"(static_cast<unsigned int>(0)) : "memory");
                    }
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (dw0 != 0) {
                    if (elect_sync()) {
                        asm volatile("fence.release.sys;" ::: "memory");
                    }
                    __syncwarp();
                    // nvlink_barrier(mega_pg_flags) phase=0 owner_warp=8
                    {
                        const int __ws = mega_pg_world;
                        const int __me = mega_pg_rank;
                        const int __warp = warp;
                        const int __lane = lane;
                        if (__warp == 8) {
                            unsigned* __local_epoch = mega_pg_flags.ptrs[__me] + 0;
                            unsigned __previous_epoch;
                            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                                : "=r"(__previous_epoch) : "l"(__local_epoch) : "memory");
                            const unsigned __epoch = __previous_epoch + 1u;
                            const int __bank = (int)(__epoch & 1u);
                            const int __mailbox_base = 2 + (0 * 2 + __bank) * __ws;
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            if (__lane < __ws) {
                                unsigned* __peer_mailbox = mega_pg_flags.ptrs[__lane] + __mailbox_base + __me;
                                asm volatile("st.release.sys.global.u32 [%0], %1;"
                                    :: "l"(__peer_mailbox), "r"(__epoch) : "memory");
                                unsigned* __local_mailbox = mega_pg_flags.ptrs[__me] + __mailbox_base + __lane;
                                while (true) {
                                    unsigned __v;
                                    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_mailbox) : "memory");
                                    if (__v != __epoch) continue;
                                    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__v) : "l"(__local_mailbox) : "memory");
                                    if (__v == __epoch) break;
                                }
                                asm volatile("fence.proxy.alias;" ::: "memory");
                            }
                            __syncwarp();
                            asm volatile("fence.proxy.async.global;" ::: "memory");
                            if (__lane == 0) {
                                asm volatile("st.release.sys.global.u32 [%0], %1;"
                                    :: "l"(__local_epoch), "r"(__epoch) : "memory");
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(576));
    }
}

} // extern "C"
