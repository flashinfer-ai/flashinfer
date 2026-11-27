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
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
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



__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}








__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
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


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(480) __cluster_dims__(2,1,1) void
kernel_cake_fp4_mla_decode_2e6c290c0084109bce41(const __grid_constant__ CUtensorMap kv_k, const __grid_constant__ CUtensorMap kv_v, unsigned int* __restrict__ kv_sf, unsigned int* __restrict__ v_pool, unsigned int* __restrict__ q, unsigned int* __restrict__ q_sf, int* __restrict__ page_table, int* __restrict__ kv_lens, float* __restrict__ q_gs, float* __restrict__ kv_gs, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, float sm_scale, int num_heads, int q_len, int num_split, int page_stride, int page_offset, int batch, const __grid_constant__ CUtensorMap ksf_map, const __grid_constant__ CUtensorMap vsf_map, const __grid_constant__ CUtensorMap out_map, const __grid_constant__ CUtensorMap po_map)
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
    #define k_empty_addr (mbar_base + 16)
    #define raw_full_addr (mbar_base + 32)
    #define raw_empty_addr (mbar_base + 48)
    #define vt_full_addr (mbar_base + 64)
    #define vt_empty_addr (mbar_base + 80)
    #define vs_full_addr (mbar_base + 96)
    #define vs_empty_addr (mbar_base + 128)
    #define q_full_addr (mbar_base + 160)
    #define q_empty_addr (mbar_base + 168)
    #define o_free_addr (mbar_base + 176)
    #define s_full_addr (mbar_base + 184)
    #define s_empty_addr (mbar_base + 200)
    #define p_full_addr (mbar_base + 216)
    #define pv_done_addr (mbar_base + 232)
    #define m_ready_addr (mbar_base + 248)
    #define l_ready_addr (mbar_base + 264)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_k_addr = smem + 1024;
    uint8_t* _smem_raw = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_raw_addr = smem + 41984;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 74752);
    const int smem_vt_addr = smem + 74752;
    unsigned int* _smem_ks = reinterpret_cast<unsigned int*>(smem_raw + 107520);
    const int smem_ks_addr = smem + 107520;
    unsigned int* _smem_vs = reinterpret_cast<unsigned int*>(smem_raw + 128000);
    const int smem_vs_addr = smem + 128000;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 144384);
    const int smem_p_addr = smem + 144384;
    float* xch_g = reinterpret_cast<float*>(smem_raw + 152576);
    const int _xch_g_addr = smem + 152576;
    float* xch_l = reinterpret_cast<float*>(smem_raw + 160768);
    const int _xch_l_addr = smem + 160768;
    float* xch_m = reinterpret_cast<float*>(smem_raw + 161792);
    const int _xch_m_addr = smem + 161792;
    float* xch_s = reinterpret_cast<float*>(smem_raw + 162816);
    const int _xch_s_addr = smem + 162816;
    unsigned int* xch_w = reinterpret_cast<unsigned int*>(smem_raw + 163840);
    const int _xch_w_addr = smem + 163840;
    unsigned int* _smem_os = reinterpret_cast<unsigned int*>(smem_raw + 165888);
    const int smem_os_addr = smem + 165888;

    // Kernel setup ops
    int m_tiles = (q_len * num_heads + 127) / 128;
    int n_all = num_split * m_tiles * batch;
    int clu = blockIdx.x / 2;
    int n_clu = gridDim.x / 2;
    int n_items = ((clu < n_all) ? (n_all - clu + n_clu - 1) / n_clu : 0);

    // Mbarrier init (35 barriers at smem_raw[0..280))

    if (warp == 0) {
        // --- pipeline 'k_pipe' ---
        // k_full: 2 barriers, init_count=2
        // k_empty: 2 barriers, init_count=1
        // --- pipeline 'raw_pipe' ---
        // raw_full: 2 barriers, init_count=1
        // raw_empty: 2 barriers, init_count=128
        // --- pipeline 'vt_pipe' ---
        // vt_full: 2 barriers, init_count=256
        // vt_empty: 2 barriers, init_count=1
        // --- pipeline 'vs_pipe' ---
        // vs_full: 4 barriers, init_count=2
        // vs_empty: 4 barriers, init_count=1
        // q_full: 1 barriers, init_count=256
        // q_empty: 1 barriers, init_count=1
        // o_free: 1 barriers, init_count=8
        // s_full: 2 barriers, init_count=1
        // s_empty: 2 barriers, init_count=256
        // p_full: 2 barriers, init_count=256
        // pv_done: 2 barriers, init_count=1
        // m_ready: 2 barriers, init_count=128
        // l_ready: 2 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(31), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(29), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(25), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(23), "r"((uint32_t)(8)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(22), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(21), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(16), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(8), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(2), "r"((uint32_t)(2)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        if (lane < 3) {
            mbarrier_init(smem + 256 + lane * 8, 128);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 280);
    if (warp == 0) {
        int _tmem_hold = smem + 280;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;
    const unsigned int tmem_q = taddr + 320;
    const unsigned int tmem_sfa = taddr + 400;
    const unsigned int tmem_sfb = taddr + 440;
    const unsigned int tmem_sfap = taddr + 480;
    const unsigned int tmem_sfbv = taddr + 496;

    // ---- Role: smx ----
    if (warp <= 7) {
        { // smx_main
            int grp = make_warp_uniform(warp / 4);
            int qs = make_warp_uniform(warp % 4);
            int half = qs / 2;
            int lane_base_s = qs * 32 << 16;
            int r = qs % 2 * 32 + lane;
            int row_local = cta_rank * 64 + r;
            int mine = qs * 32 + lane;
            int other = (qs ^ 2) * 32 + lane;
            int xbar = make_warp_uniform(1 + 2 * grp + qs % 2);
            float gq = *q_gs;
            float gk = *kv_gs;
            float c_log2 = sm_scale * 1.442695f / (gq * gk);
            float m = -CUDART_INF_F;
            float l_sum = 0.0f;
            float s_vals[64];
            int h_s = 0;
            int g0_s = 0;
            int og = make_warp_uniform(1 - warp / 4);
            if (grp == 0) {
                xch_m[128 + mine] = -CUDART_INF_F;
                xch_s[128 + mine] = 0.0f;
            }
            unsigned int _phase_s_full = 0;
            unsigned int _phase_m_ready = 1;
            unsigned int _phase_m_ready_1 = 0;
            unsigned int _phase_l_ready = 1;
            unsigned int _phase_l_ready_1 = 0;
            unsigned int _phase_pv_done = 1;
            unsigned int _phase_pv_done_1 = 0;
            unsigned int _phase_pv_done_2 = 1;
            #pragma unroll 1
            for (int k_it = 0; k_it < n_items; k_it++) {
                int it = clu + k_it * n_clu;
                int split_i = it % num_split;
                int rest = it / num_split;
                int b_i = rest / m_tiles;
                int rows_b = q_len * num_heads;
                int row0 = rest % m_tiles * 128;
                int rows_left = rows_b - row0;
                int rows_valid = ((rows_left > 128) ? 128 : rows_left);
                int row_base = b_i * rows_b + row0;
                int kv_len_i = kv_lens[b_i];
                int t_last = (row0 + rows_valid - 1) / num_heads;
                int kv_end_raw = kv_len_i - (q_len - 1 - t_last);
                int kv_end = ((kv_end_raw < 0) ? 0 : kv_end_raw);
                int n_pages = (kv_end + 128 - 1) / 128;
                int pps = (n_pages + num_split - 1) / num_split;
                int p_start = split_i * pps;
                int p_end_raw = p_start + pps;
                int p_end = ((p_end_raw > n_pages) ? n_pages : p_end_raw);
                int n_raw = p_end - p_start;
                int empty = ((n_raw < 1) ? 1 : 0);
                int my_n = ((n_raw < 1) ? 1 : n_raw);
                int my_start = ((n_raw < 1) ? 0 : p_start);
                int keys = ((empty != 0) ? 0 : kv_end);
                int row_ok_c = row_local < rows_valid;
                int token = (row0 + row_local) / num_heads;
                int vis_raw = kv_len_i - (q_len - 1 - token);
                int vis_c = ((row_ok_c != 0 && keys > 0) ? vis_raw : 0);
                m = -CUDART_INF_F;
                l_sum = 0.0f;
                int j0 = (grp + g0_s) % 2;
                int n_grp = (my_n - j0 + 2 - 1) / 2;
                #pragma unroll 1
                for (int jj = 0; jj < n_grp; jj++) {
                    int j = j0 + 2 * jj;
                    int xs = 2 * grp + h_s % 2;
                    int tok0 = 128 * (my_start + j) + 64 * half;
                    mbarrier_wait(s_full_addr + (grp) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float bmax0 = -CUDART_INF_F;
                    float bmax1 = -CUDART_INF_F;
                    float bmax2 = -CUDART_INF_F;
                    float bmax3 = -CUDART_INF_F;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17];"
                        : "=f"(s_vals[0]), "=f"(s_vals[1]), "=f"(s_vals[2]), "=f"(s_vals[3]), "=f"(s_vals[4]), "=f"(s_vals[5]), "=f"(s_vals[6]), "=f"(s_vals[7]), "=f"(s_vals[8]), "=f"(s_vals[9]), "=f"(s_vals[10]), "=f"(s_vals[11]), "=f"(s_vals[12]), "=f"(s_vals[13]), "=f"(s_vals[14]), "=f"(s_vals[15]), "=f"(bmax0)
                        : "r"(taddr + (unsigned int)lane_base_s + 256));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17];"
                        : "=f"(s_vals[16]), "=f"(s_vals[17]), "=f"(s_vals[18]), "=f"(s_vals[19]), "=f"(s_vals[20]), "=f"(s_vals[21]), "=f"(s_vals[22]), "=f"(s_vals[23]), "=f"(s_vals[24]), "=f"(s_vals[25]), "=f"(s_vals[26]), "=f"(s_vals[27]), "=f"(s_vals[28]), "=f"(s_vals[29]), "=f"(s_vals[30]), "=f"(s_vals[31]), "=f"(bmax1)
                        : "r"(taddr + (unsigned int)lane_base_s + 272));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17];"
                        : "=f"(s_vals[32]), "=f"(s_vals[33]), "=f"(s_vals[34]), "=f"(s_vals[35]), "=f"(s_vals[36]), "=f"(s_vals[37]), "=f"(s_vals[38]), "=f"(s_vals[39]), "=f"(s_vals[40]), "=f"(s_vals[41]), "=f"(s_vals[42]), "=f"(s_vals[43]), "=f"(s_vals[44]), "=f"(s_vals[45]), "=f"(s_vals[46]), "=f"(s_vals[47]), "=f"(bmax2)
                        : "r"(taddr + (unsigned int)lane_base_s + 288));
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17];"
                        : "=f"(s_vals[48]), "=f"(s_vals[49]), "=f"(s_vals[50]), "=f"(s_vals[51]), "=f"(s_vals[52]), "=f"(s_vals[53]), "=f"(s_vals[54]), "=f"(s_vals[55]), "=f"(s_vals[56]), "=f"(s_vals[57]), "=f"(s_vals[58]), "=f"(s_vals[59]), "=f"(s_vals[60]), "=f"(s_vals[61]), "=f"(s_vals[62]), "=f"(s_vals[63]), "=f"(bmax3)
                        : "r"(taddr + (unsigned int)lane_base_s + 304));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + (grp) * 8), "r"(0) : "memory");
                    _phase_s_full ^= 1;
                    int live = vis_c - tok0;
                    int _warp_vote_0 = __any_sync(0xFFFFFFFF, live < 64);
                    if (_warp_vote_0 != 0) {
                        s_vals[0] = ((live > 0) ? s_vals[0] : -CUDART_INF_F);
                        s_vals[1] = ((live > 1) ? s_vals[1] : -CUDART_INF_F);
                        s_vals[2] = ((live > 2) ? s_vals[2] : -CUDART_INF_F);
                        s_vals[3] = ((live > 3) ? s_vals[3] : -CUDART_INF_F);
                        s_vals[4] = ((live > 4) ? s_vals[4] : -CUDART_INF_F);
                        s_vals[5] = ((live > 5) ? s_vals[5] : -CUDART_INF_F);
                        s_vals[6] = ((live > 6) ? s_vals[6] : -CUDART_INF_F);
                        s_vals[7] = ((live > 7) ? s_vals[7] : -CUDART_INF_F);
                        s_vals[8] = ((live > 8) ? s_vals[8] : -CUDART_INF_F);
                        s_vals[9] = ((live > 9) ? s_vals[9] : -CUDART_INF_F);
                        s_vals[10] = ((live > 10) ? s_vals[10] : -CUDART_INF_F);
                        s_vals[11] = ((live > 11) ? s_vals[11] : -CUDART_INF_F);
                        s_vals[12] = ((live > 12) ? s_vals[12] : -CUDART_INF_F);
                        s_vals[13] = ((live > 13) ? s_vals[13] : -CUDART_INF_F);
                        s_vals[14] = ((live > 14) ? s_vals[14] : -CUDART_INF_F);
                        s_vals[15] = ((live > 15) ? s_vals[15] : -CUDART_INF_F);
                        s_vals[16] = ((live > 16) ? s_vals[16] : -CUDART_INF_F);
                        s_vals[17] = ((live > 17) ? s_vals[17] : -CUDART_INF_F);
                        s_vals[18] = ((live > 18) ? s_vals[18] : -CUDART_INF_F);
                        s_vals[19] = ((live > 19) ? s_vals[19] : -CUDART_INF_F);
                        s_vals[20] = ((live > 20) ? s_vals[20] : -CUDART_INF_F);
                        s_vals[21] = ((live > 21) ? s_vals[21] : -CUDART_INF_F);
                        s_vals[22] = ((live > 22) ? s_vals[22] : -CUDART_INF_F);
                        s_vals[23] = ((live > 23) ? s_vals[23] : -CUDART_INF_F);
                        s_vals[24] = ((live > 24) ? s_vals[24] : -CUDART_INF_F);
                        s_vals[25] = ((live > 25) ? s_vals[25] : -CUDART_INF_F);
                        s_vals[26] = ((live > 26) ? s_vals[26] : -CUDART_INF_F);
                        s_vals[27] = ((live > 27) ? s_vals[27] : -CUDART_INF_F);
                        s_vals[28] = ((live > 28) ? s_vals[28] : -CUDART_INF_F);
                        s_vals[29] = ((live > 29) ? s_vals[29] : -CUDART_INF_F);
                        s_vals[30] = ((live > 30) ? s_vals[30] : -CUDART_INF_F);
                        s_vals[31] = ((live > 31) ? s_vals[31] : -CUDART_INF_F);
                        s_vals[32] = ((live > 32) ? s_vals[32] : -CUDART_INF_F);
                        s_vals[33] = ((live > 33) ? s_vals[33] : -CUDART_INF_F);
                        s_vals[34] = ((live > 34) ? s_vals[34] : -CUDART_INF_F);
                        s_vals[35] = ((live > 35) ? s_vals[35] : -CUDART_INF_F);
                        s_vals[36] = ((live > 36) ? s_vals[36] : -CUDART_INF_F);
                        s_vals[37] = ((live > 37) ? s_vals[37] : -CUDART_INF_F);
                        s_vals[38] = ((live > 38) ? s_vals[38] : -CUDART_INF_F);
                        s_vals[39] = ((live > 39) ? s_vals[39] : -CUDART_INF_F);
                        s_vals[40] = ((live > 40) ? s_vals[40] : -CUDART_INF_F);
                        s_vals[41] = ((live > 41) ? s_vals[41] : -CUDART_INF_F);
                        s_vals[42] = ((live > 42) ? s_vals[42] : -CUDART_INF_F);
                        s_vals[43] = ((live > 43) ? s_vals[43] : -CUDART_INF_F);
                        s_vals[44] = ((live > 44) ? s_vals[44] : -CUDART_INF_F);
                        s_vals[45] = ((live > 45) ? s_vals[45] : -CUDART_INF_F);
                        s_vals[46] = ((live > 46) ? s_vals[46] : -CUDART_INF_F);
                        s_vals[47] = ((live > 47) ? s_vals[47] : -CUDART_INF_F);
                        s_vals[48] = ((live > 48) ? s_vals[48] : -CUDART_INF_F);
                        s_vals[49] = ((live > 49) ? s_vals[49] : -CUDART_INF_F);
                        s_vals[50] = ((live > 50) ? s_vals[50] : -CUDART_INF_F);
                        s_vals[51] = ((live > 51) ? s_vals[51] : -CUDART_INF_F);
                        s_vals[52] = ((live > 52) ? s_vals[52] : -CUDART_INF_F);
                        s_vals[53] = ((live > 53) ? s_vals[53] : -CUDART_INF_F);
                        s_vals[54] = ((live > 54) ? s_vals[54] : -CUDART_INF_F);
                        s_vals[55] = ((live > 55) ? s_vals[55] : -CUDART_INF_F);
                        s_vals[56] = ((live > 56) ? s_vals[56] : -CUDART_INF_F);
                        s_vals[57] = ((live > 57) ? s_vals[57] : -CUDART_INF_F);
                        s_vals[58] = ((live > 58) ? s_vals[58] : -CUDART_INF_F);
                        s_vals[59] = ((live > 59) ? s_vals[59] : -CUDART_INF_F);
                        s_vals[60] = ((live > 60) ? s_vals[60] : -CUDART_INF_F);
                        s_vals[61] = ((live > 61) ? s_vals[61] : -CUDART_INF_F);
                        s_vals[62] = ((live > 62) ? s_vals[62] : -CUDART_INF_F);
                        s_vals[63] = ((live > 63) ? s_vals[63] : -CUDART_INF_F);
                        float _reg_array_reduce_1 = s_vals[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _reg_array_reduce_1 = max_noftz(_reg_array_reduce_1, s_vals[_lr]);
                        }
                        bmax0 = _reg_array_reduce_1;
                        float _reg_array_reduce_2 = s_vals[(16) + 0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _reg_array_reduce_2 = max_noftz(_reg_array_reduce_2, s_vals[(16) + _lr]);
                        }
                        bmax1 = _reg_array_reduce_2;
                        float _reg_array_reduce_3 = s_vals[(32) + 0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _reg_array_reduce_3 = max_noftz(_reg_array_reduce_3, s_vals[(32) + _lr]);
                        }
                        bmax2 = _reg_array_reduce_3;
                        float _reg_array_reduce_4 = s_vals[(48) + 0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _reg_array_reduce_4 = max_noftz(_reg_array_reduce_4, s_vals[(48) + _lr]);
                        }
                        bmax3 = _reg_array_reduce_4;
                    }
                    float g[8];
                    g[0] = ((bmax0 > -CUDART_INF_F) ? c_log2 * bmax0 : -CUDART_INF_F);
                    g[1] = ((bmax1 > -CUDART_INF_F) ? c_log2 * bmax1 : -CUDART_INF_F);
                    g[2] = ((bmax2 > -CUDART_INF_F) ? c_log2 * bmax2 : -CUDART_INF_F);
                    g[3] = ((bmax3 > -CUDART_INF_F) ? c_log2 * bmax3 : -CUDART_INF_F);
                    xch_g[(xs * 128 + mine) * 4] = g[0];
                    xch_g[(xs * 128 + mine) * 4 + 1] = g[1];
                    xch_g[(xs * 128 + mine) * 4 + 2] = g[2];
                    xch_g[(xs * 128 + mine) * 4 + 3] = g[3];
                    asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                    g[4] = xch_g[(xs * 128 + other) * 4];
                    g[5] = xch_g[(xs * 128 + other) * 4 + 1];
                    g[6] = xch_g[(xs * 128 + other) * 4 + 2];
                    g[7] = xch_g[(xs * 128 + other) * 4 + 3];
                    float page_max = g[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 8; _lr++) {
                        page_max = max_noftz(page_max, g[_lr]);
                    }
                    const bool _selected_wait_cond_5 = grp == 0;
                    mbarrier_wait((_selected_wait_cond_5 ? (m_ready_addr + (og) * 8) : (m_ready_addr + (og) * 8)), (_selected_wait_cond_5 ? (_phase_m_ready) : (_phase_m_ready_1)));
                    _phase_m_ready ^= 1;
                    _phase_m_ready_1 ^= 1;
                    m = xch_m[og * 128 + mine];
                    int move = page_max > -CUDART_INF_F && (m == -CUDART_INF_F || page_max > m + 3.0f);
                    int rescale = move != 0 && m > -CUDART_INF_F;
                    float _exp2_6 = approx_exp2(m - page_max);
                    float alpha = ((rescale != 0) ? _exp2_6 : 1.0f);
                    if (move != 0) {
                        m = page_max;
                    }
                    xch_m[grp * 128 + mine] = ((j == my_n - 1) ? -CUDART_INF_F : m);
                    mbarrier_arrive(m_ready_addr + (grp) * 8);
                    unsigned int codes[8];
                    unsigned int sf_pair[2];
                    float sf_fit[4];
                    float b_sums[4];
                    float b_weights[4];
                    float g_safe[4];
                    float qv[4];
                    float qq[4];
                    g_safe[0] = ((g[0] > -CUDART_INF_F) ? g[0] : 0.0f);
                    const float2 _fma_b2_7 = {c_log2, c_log2};
                    const float2 _fma_c2_8 = {-g_safe[0], -g_safe[0]};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_7, _fma_c2_8);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le] = approx_exp2(s_vals[_le]);
                    }
                    float t8[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 0))[_la], reinterpret_cast<const float2*>((s_vals + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8)[_la], reinterpret_cast<const float2*>((t8 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8)[_la], reinterpret_cast<const float2*>((t8 + 2))[_la]);
                    b_sums[0] = t8[0] + t8[1];
                    const float2 _scale2_9 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_ls], _scale2_9);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(s_vals[0]), "f"(s_vals[1]), "f"(s_vals[2]), "f"(s_vals[3]), "f"(s_vals[4]), "f"(s_vals[5]), "f"(s_vals[6]), "f"(s_vals[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(s_vals[8]), "f"(s_vals[9]), "f"(s_vals[10]), "f"(s_vals[11]), "f"(s_vals[12]), "f"(s_vals[13]), "f"(s_vals[14]), "f"(s_vals[15]));
                    float q_b[16];
                    unsigned int qn[8];
                    uint32_t _cvt_f16x2_e2m1x2_10;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_10) : "r"((uint32_t)(codes[0] >> 0)));
                    qn[0] = _cvt_f16x2_e2m1x2_10;
                    float qw_0[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_0[0]), "=&f"(qw_0[1]) : "r"(qn[0]), "f"(0.0f), "f"(0.0f));
                    q_b[0] = qw_0[0];
                    q_b[1] = qw_0[1];
                    uint32_t _cvt_f16x2_e2m1x2_11;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_11) : "r"((uint32_t)(codes[0] >> 8)));
                    qn[1] = _cvt_f16x2_e2m1x2_11;
                    float qw_1[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_1[0]), "=&f"(qw_1[1]) : "r"(qn[1]), "f"(0.0f), "f"(0.0f));
                    q_b[2] = qw_1[0];
                    q_b[3] = qw_1[1];
                    uint32_t _cvt_f16x2_e2m1x2_12;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_12) : "r"((uint32_t)(codes[0] >> 16)));
                    qn[2] = _cvt_f16x2_e2m1x2_12;
                    float qw_2[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_2[0]), "=&f"(qw_2[1]) : "r"(qn[2]), "f"(0.0f), "f"(0.0f));
                    q_b[4] = qw_2[0];
                    q_b[5] = qw_2[1];
                    uint32_t _cvt_f16x2_e2m1x2_13;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_13) : "r"((uint32_t)(codes[0] >> 24)));
                    qn[3] = _cvt_f16x2_e2m1x2_13;
                    float qw_3[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_3[0]), "=&f"(qw_3[1]) : "r"(qn[3]), "f"(0.0f), "f"(0.0f));
                    q_b[6] = qw_3[0];
                    q_b[7] = qw_3[1];
                    uint32_t _cvt_f16x2_e2m1x2_14;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_14) : "r"((uint32_t)(codes[1] >> 0)));
                    qn[4] = _cvt_f16x2_e2m1x2_14;
                    float qw_4[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_4[0]), "=&f"(qw_4[1]) : "r"(qn[4]), "f"(0.0f), "f"(0.0f));
                    q_b[8] = qw_4[0];
                    q_b[9] = qw_4[1];
                    uint32_t _cvt_f16x2_e2m1x2_15;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_15) : "r"((uint32_t)(codes[1] >> 8)));
                    qn[5] = _cvt_f16x2_e2m1x2_15;
                    float qw_5[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_5[0]), "=&f"(qw_5[1]) : "r"(qn[5]), "f"(0.0f), "f"(0.0f));
                    q_b[10] = qw_5[0];
                    q_b[11] = qw_5[1];
                    uint32_t _cvt_f16x2_e2m1x2_16;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_16) : "r"((uint32_t)(codes[1] >> 16)));
                    qn[6] = _cvt_f16x2_e2m1x2_16;
                    float qw_6[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_6[0]), "=&f"(qw_6[1]) : "r"(qn[6]), "f"(0.0f), "f"(0.0f));
                    q_b[12] = qw_6[0];
                    q_b[13] = qw_6[1];
                    uint32_t _cvt_f16x2_e2m1x2_17;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_17) : "r"((uint32_t)(codes[1] >> 24)));
                    qn[7] = _cvt_f16x2_e2m1x2_17;
                    float qw_7[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_7[0]), "=&f"(qw_7[1]) : "r"(qn[7]), "f"(0.0f), "f"(0.0f));
                    q_b[14] = qw_7[0];
                    q_b[15] = qw_7[1];
                    float qe[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b)[_ls], reinterpret_cast<const float2*>((s_vals + 0))[_ls]);
                    float t8_0[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_0)[_la] = add_f32x2(reinterpret_cast<float2*>((qe + 0))[_la], reinterpret_cast<const float2*>((qe + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_0)[_la], reinterpret_cast<const float2*>((t8_0 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_0)[_la], reinterpret_cast<const float2*>((t8_0 + 2))[_la]);
                    qv[0] = t8_0[0] + t8_0[1];
                    uint32_t a0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0) : "r"(qn[0]), "r"(qn[0]), "r"(0));
                    uint32_t b0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0) : "r"(qn[4]), "r"(qn[4]), "r"(0));
                    uint32_t a1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1) : "r"(qn[1]), "r"(qn[1]), "r"(a0));
                    uint32_t b1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1) : "r"(qn[5]), "r"(qn[5]), "r"(b0));
                    uint32_t a2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2) : "r"(qn[2]), "r"(qn[2]), "r"(a1));
                    uint32_t b2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2) : "r"(qn[6]), "r"(qn[6]), "r"(b1));
                    uint32_t a3;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3) : "r"(qn[3]), "r"(qn[3]), "r"(a2));
                    uint32_t b3;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3) : "r"(qn[7]), "r"(qn[7]), "r"(b2));
                    uint32_t v;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v) : "r"(a3), "r"(b3));
                    float w[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w[0]), "=&f"(w[1]) : "r"(v), "f"(0.0f), "f"(0.0f));
                    qq[0] = w[0] + w[1];
                    g_safe[1] = ((g[1] > -CUDART_INF_F) ? g[1] : 0.0f);
                    const float2 _fma_b2_18 = {c_log2, c_log2};
                    const float2 _fma_c2_19 = {-g_safe[1], -g_safe[1]};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_18, _fma_c2_19);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le + 16] = approx_exp2(s_vals[_le + 16]);
                    }
                    float t8_1[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_1)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 16))[_la], reinterpret_cast<const float2*>((s_vals + 24))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_1)[_la], reinterpret_cast<const float2*>((t8_1 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_1)[_la], reinterpret_cast<const float2*>((t8_1 + 2))[_la]);
                    b_sums[1] = t8_1[0] + t8_1[1];
                    const float2 _scale2_20 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_ls], _scale2_20);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[2]) : "f"(s_vals[16]), "f"(s_vals[17]), "f"(s_vals[18]), "f"(s_vals[19]), "f"(s_vals[20]), "f"(s_vals[21]), "f"(s_vals[22]), "f"(s_vals[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[3]) : "f"(s_vals[24]), "f"(s_vals[25]), "f"(s_vals[26]), "f"(s_vals[27]), "f"(s_vals[28]), "f"(s_vals[29]), "f"(s_vals[30]), "f"(s_vals[31]));
                    float q_b_0[16];
                    unsigned int qn_0[8];
                    uint32_t _cvt_f16x2_e2m1x2_21;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_21) : "r"((uint32_t)(codes[2] >> 0)));
                    qn_0[0] = _cvt_f16x2_e2m1x2_21;
                    float qw_8[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_8[0]), "=&f"(qw_8[1]) : "r"(qn_0[0]), "f"(0.0f), "f"(0.0f));
                    q_b_0[0] = qw_8[0];
                    q_b_0[1] = qw_8[1];
                    uint32_t _cvt_f16x2_e2m1x2_22;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_22) : "r"((uint32_t)(codes[2] >> 8)));
                    qn_0[1] = _cvt_f16x2_e2m1x2_22;
                    float qw_9[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_9[0]), "=&f"(qw_9[1]) : "r"(qn_0[1]), "f"(0.0f), "f"(0.0f));
                    q_b_0[2] = qw_9[0];
                    q_b_0[3] = qw_9[1];
                    uint32_t _cvt_f16x2_e2m1x2_23;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_23) : "r"((uint32_t)(codes[2] >> 16)));
                    qn_0[2] = _cvt_f16x2_e2m1x2_23;
                    float qw_10[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_10[0]), "=&f"(qw_10[1]) : "r"(qn_0[2]), "f"(0.0f), "f"(0.0f));
                    q_b_0[4] = qw_10[0];
                    q_b_0[5] = qw_10[1];
                    uint32_t _cvt_f16x2_e2m1x2_24;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_24) : "r"((uint32_t)(codes[2] >> 24)));
                    qn_0[3] = _cvt_f16x2_e2m1x2_24;
                    float qw_11[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_11[0]), "=&f"(qw_11[1]) : "r"(qn_0[3]), "f"(0.0f), "f"(0.0f));
                    q_b_0[6] = qw_11[0];
                    q_b_0[7] = qw_11[1];
                    uint32_t _cvt_f16x2_e2m1x2_25;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_25) : "r"((uint32_t)(codes[3] >> 0)));
                    qn_0[4] = _cvt_f16x2_e2m1x2_25;
                    float qw_12[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_12[0]), "=&f"(qw_12[1]) : "r"(qn_0[4]), "f"(0.0f), "f"(0.0f));
                    q_b_0[8] = qw_12[0];
                    q_b_0[9] = qw_12[1];
                    uint32_t _cvt_f16x2_e2m1x2_26;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_26) : "r"((uint32_t)(codes[3] >> 8)));
                    qn_0[5] = _cvt_f16x2_e2m1x2_26;
                    float qw_13[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_13[0]), "=&f"(qw_13[1]) : "r"(qn_0[5]), "f"(0.0f), "f"(0.0f));
                    q_b_0[10] = qw_13[0];
                    q_b_0[11] = qw_13[1];
                    uint32_t _cvt_f16x2_e2m1x2_27;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_27) : "r"((uint32_t)(codes[3] >> 16)));
                    qn_0[6] = _cvt_f16x2_e2m1x2_27;
                    float qw_14[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_14[0]), "=&f"(qw_14[1]) : "r"(qn_0[6]), "f"(0.0f), "f"(0.0f));
                    q_b_0[12] = qw_14[0];
                    q_b_0[13] = qw_14[1];
                    uint32_t _cvt_f16x2_e2m1x2_28;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_28) : "r"((uint32_t)(codes[3] >> 24)));
                    qn_0[7] = _cvt_f16x2_e2m1x2_28;
                    float qw_15[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_15[0]), "=&f"(qw_15[1]) : "r"(qn_0[7]), "f"(0.0f), "f"(0.0f));
                    q_b_0[14] = qw_15[0];
                    q_b_0[15] = qw_15[1];
                    float qe_0[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_0)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_0)[_ls], reinterpret_cast<const float2*>((s_vals + 16))[_ls]);
                    float t8_2[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_2)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_0 + 0))[_la], reinterpret_cast<const float2*>((qe_0 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_2)[_la], reinterpret_cast<const float2*>((t8_2 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_2)[_la], reinterpret_cast<const float2*>((t8_2 + 2))[_la]);
                    qv[1] = t8_2[0] + t8_2[1];
                    uint32_t a0_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_0) : "r"(qn_0[0]), "r"(qn_0[0]), "r"(0));
                    uint32_t b0_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_0) : "r"(qn_0[4]), "r"(qn_0[4]), "r"(0));
                    uint32_t a1_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_0) : "r"(qn_0[1]), "r"(qn_0[1]), "r"(a0_0));
                    uint32_t b1_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_0) : "r"(qn_0[5]), "r"(qn_0[5]), "r"(b0_0));
                    uint32_t a2_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_0) : "r"(qn_0[2]), "r"(qn_0[2]), "r"(a1_0));
                    uint32_t b2_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_0) : "r"(qn_0[6]), "r"(qn_0[6]), "r"(b1_0));
                    uint32_t a3_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_0) : "r"(qn_0[3]), "r"(qn_0[3]), "r"(a2_0));
                    uint32_t b3_0;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_0) : "r"(qn_0[7]), "r"(qn_0[7]), "r"(b2_0));
                    uint32_t v_16;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v_16) : "r"(a3_0), "r"(b3_0));
                    float w_0[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w_0[0]), "=&f"(w_0[1]) : "r"(v_16), "f"(0.0f), "f"(0.0f));
                    qq[1] = w_0[0] + w_0[1];
                    g_safe[2] = ((g[2] > -CUDART_INF_F) ? g[2] : 0.0f);
                    const float2 _fma_b2_29 = {c_log2, c_log2};
                    const float2 _fma_c2_30 = {-g_safe[2], -g_safe[2]};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_29, _fma_c2_30);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le + 32] = approx_exp2(s_vals[_le + 32]);
                    }
                    float t8_3[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_3)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 32))[_la], reinterpret_cast<const float2*>((s_vals + 40))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_3)[_la], reinterpret_cast<const float2*>((t8_3 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_3)[_la], reinterpret_cast<const float2*>((t8_3 + 2))[_la]);
                    b_sums[2] = t8_3[0] + t8_3[1];
                    const float2 _scale2_31 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_ls], _scale2_31);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[4]) : "f"(s_vals[32]), "f"(s_vals[33]), "f"(s_vals[34]), "f"(s_vals[35]), "f"(s_vals[36]), "f"(s_vals[37]), "f"(s_vals[38]), "f"(s_vals[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[5]) : "f"(s_vals[40]), "f"(s_vals[41]), "f"(s_vals[42]), "f"(s_vals[43]), "f"(s_vals[44]), "f"(s_vals[45]), "f"(s_vals[46]), "f"(s_vals[47]));
                    float q_b_1[16];
                    unsigned int qn_1[8];
                    uint32_t _cvt_f16x2_e2m1x2_32;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_32) : "r"((uint32_t)(codes[4] >> 0)));
                    qn_1[0] = _cvt_f16x2_e2m1x2_32;
                    float qw_16[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_16[0]), "=&f"(qw_16[1]) : "r"(qn_1[0]), "f"(0.0f), "f"(0.0f));
                    q_b_1[0] = qw_16[0];
                    q_b_1[1] = qw_16[1];
                    uint32_t _cvt_f16x2_e2m1x2_33;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_33) : "r"((uint32_t)(codes[4] >> 8)));
                    qn_1[1] = _cvt_f16x2_e2m1x2_33;
                    float qw_17[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_17[0]), "=&f"(qw_17[1]) : "r"(qn_1[1]), "f"(0.0f), "f"(0.0f));
                    q_b_1[2] = qw_17[0];
                    q_b_1[3] = qw_17[1];
                    uint32_t _cvt_f16x2_e2m1x2_34;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_34) : "r"((uint32_t)(codes[4] >> 16)));
                    qn_1[2] = _cvt_f16x2_e2m1x2_34;
                    float qw_18[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_18[0]), "=&f"(qw_18[1]) : "r"(qn_1[2]), "f"(0.0f), "f"(0.0f));
                    q_b_1[4] = qw_18[0];
                    q_b_1[5] = qw_18[1];
                    uint32_t _cvt_f16x2_e2m1x2_35;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_35) : "r"((uint32_t)(codes[4] >> 24)));
                    qn_1[3] = _cvt_f16x2_e2m1x2_35;
                    float qw_19[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_19[0]), "=&f"(qw_19[1]) : "r"(qn_1[3]), "f"(0.0f), "f"(0.0f));
                    q_b_1[6] = qw_19[0];
                    q_b_1[7] = qw_19[1];
                    uint32_t _cvt_f16x2_e2m1x2_36;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_36) : "r"((uint32_t)(codes[5] >> 0)));
                    qn_1[4] = _cvt_f16x2_e2m1x2_36;
                    float qw_20[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_20[0]), "=&f"(qw_20[1]) : "r"(qn_1[4]), "f"(0.0f), "f"(0.0f));
                    q_b_1[8] = qw_20[0];
                    q_b_1[9] = qw_20[1];
                    uint32_t _cvt_f16x2_e2m1x2_37;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_37) : "r"((uint32_t)(codes[5] >> 8)));
                    qn_1[5] = _cvt_f16x2_e2m1x2_37;
                    float qw_21[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_21[0]), "=&f"(qw_21[1]) : "r"(qn_1[5]), "f"(0.0f), "f"(0.0f));
                    q_b_1[10] = qw_21[0];
                    q_b_1[11] = qw_21[1];
                    uint32_t _cvt_f16x2_e2m1x2_38;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_38) : "r"((uint32_t)(codes[5] >> 16)));
                    qn_1[6] = _cvt_f16x2_e2m1x2_38;
                    float qw_22[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_22[0]), "=&f"(qw_22[1]) : "r"(qn_1[6]), "f"(0.0f), "f"(0.0f));
                    q_b_1[12] = qw_22[0];
                    q_b_1[13] = qw_22[1];
                    uint32_t _cvt_f16x2_e2m1x2_39;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_39) : "r"((uint32_t)(codes[5] >> 24)));
                    qn_1[7] = _cvt_f16x2_e2m1x2_39;
                    float qw_23[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_23[0]), "=&f"(qw_23[1]) : "r"(qn_1[7]), "f"(0.0f), "f"(0.0f));
                    q_b_1[14] = qw_23[0];
                    q_b_1[15] = qw_23[1];
                    float qe_1[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_1)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_1)[_ls], reinterpret_cast<const float2*>((s_vals + 32))[_ls]);
                    float t8_4[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_4)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_1 + 0))[_la], reinterpret_cast<const float2*>((qe_1 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_4)[_la], reinterpret_cast<const float2*>((t8_4 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_4)[_la], reinterpret_cast<const float2*>((t8_4 + 2))[_la]);
                    qv[2] = t8_4[0] + t8_4[1];
                    uint32_t a0_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_1) : "r"(qn_1[0]), "r"(qn_1[0]), "r"(0));
                    uint32_t b0_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_1) : "r"(qn_1[4]), "r"(qn_1[4]), "r"(0));
                    uint32_t a1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_1) : "r"(qn_1[1]), "r"(qn_1[1]), "r"(a0_1));
                    uint32_t b1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_1) : "r"(qn_1[5]), "r"(qn_1[5]), "r"(b0_1));
                    uint32_t a2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_1) : "r"(qn_1[2]), "r"(qn_1[2]), "r"(a1_1));
                    uint32_t b2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_1) : "r"(qn_1[6]), "r"(qn_1[6]), "r"(b1_1));
                    uint32_t a3_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_1) : "r"(qn_1[3]), "r"(qn_1[3]), "r"(a2_1));
                    uint32_t b3_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_1) : "r"(qn_1[7]), "r"(qn_1[7]), "r"(b2_1));
                    uint32_t v_17;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v_17) : "r"(a3_1), "r"(b3_1));
                    float w_1[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w_1[0]), "=&f"(w_1[1]) : "r"(v_17), "f"(0.0f), "f"(0.0f));
                    qq[2] = w_1[0] + w_1[1];
                    g_safe[3] = ((g[3] > -CUDART_INF_F) ? g[3] : 0.0f);
                    const float2 _fma_b2_40 = {c_log2, c_log2};
                    const float2 _fma_c2_41 = {-g_safe[3], -g_safe[3]};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_40, _fma_c2_41);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le + 48] = approx_exp2(s_vals[_le + 48]);
                    }
                    float t8_5[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_5)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 48))[_la], reinterpret_cast<const float2*>((s_vals + 56))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_5)[_la], reinterpret_cast<const float2*>((t8_5 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_5)[_la], reinterpret_cast<const float2*>((t8_5 + 2))[_la]);
                    b_sums[3] = t8_5[0] + t8_5[1];
                    const float2 _scale2_42 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_ls], _scale2_42);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[6]) : "f"(s_vals[48]), "f"(s_vals[49]), "f"(s_vals[50]), "f"(s_vals[51]), "f"(s_vals[52]), "f"(s_vals[53]), "f"(s_vals[54]), "f"(s_vals[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[7]) : "f"(s_vals[56]), "f"(s_vals[57]), "f"(s_vals[58]), "f"(s_vals[59]), "f"(s_vals[60]), "f"(s_vals[61]), "f"(s_vals[62]), "f"(s_vals[63]));
                    float q_b_2[16];
                    unsigned int qn_2[8];
                    uint32_t _cvt_f16x2_e2m1x2_43;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_43) : "r"((uint32_t)(codes[6] >> 0)));
                    qn_2[0] = _cvt_f16x2_e2m1x2_43;
                    float qw_24[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_24[0]), "=&f"(qw_24[1]) : "r"(qn_2[0]), "f"(0.0f), "f"(0.0f));
                    q_b_2[0] = qw_24[0];
                    q_b_2[1] = qw_24[1];
                    uint32_t _cvt_f16x2_e2m1x2_44;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_44) : "r"((uint32_t)(codes[6] >> 8)));
                    qn_2[1] = _cvt_f16x2_e2m1x2_44;
                    float qw_25[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_25[0]), "=&f"(qw_25[1]) : "r"(qn_2[1]), "f"(0.0f), "f"(0.0f));
                    q_b_2[2] = qw_25[0];
                    q_b_2[3] = qw_25[1];
                    uint32_t _cvt_f16x2_e2m1x2_45;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_45) : "r"((uint32_t)(codes[6] >> 16)));
                    qn_2[2] = _cvt_f16x2_e2m1x2_45;
                    float qw_26[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_26[0]), "=&f"(qw_26[1]) : "r"(qn_2[2]), "f"(0.0f), "f"(0.0f));
                    q_b_2[4] = qw_26[0];
                    q_b_2[5] = qw_26[1];
                    uint32_t _cvt_f16x2_e2m1x2_46;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_46) : "r"((uint32_t)(codes[6] >> 24)));
                    qn_2[3] = _cvt_f16x2_e2m1x2_46;
                    float qw_27[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_27[0]), "=&f"(qw_27[1]) : "r"(qn_2[3]), "f"(0.0f), "f"(0.0f));
                    q_b_2[6] = qw_27[0];
                    q_b_2[7] = qw_27[1];
                    uint32_t _cvt_f16x2_e2m1x2_47;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_47) : "r"((uint32_t)(codes[7] >> 0)));
                    qn_2[4] = _cvt_f16x2_e2m1x2_47;
                    float qw_28[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_28[0]), "=&f"(qw_28[1]) : "r"(qn_2[4]), "f"(0.0f), "f"(0.0f));
                    q_b_2[8] = qw_28[0];
                    q_b_2[9] = qw_28[1];
                    uint32_t _cvt_f16x2_e2m1x2_48;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_48) : "r"((uint32_t)(codes[7] >> 8)));
                    qn_2[5] = _cvt_f16x2_e2m1x2_48;
                    float qw_29[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_29[0]), "=&f"(qw_29[1]) : "r"(qn_2[5]), "f"(0.0f), "f"(0.0f));
                    q_b_2[10] = qw_29[0];
                    q_b_2[11] = qw_29[1];
                    uint32_t _cvt_f16x2_e2m1x2_49;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_49) : "r"((uint32_t)(codes[7] >> 16)));
                    qn_2[6] = _cvt_f16x2_e2m1x2_49;
                    float qw_30[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_30[0]), "=&f"(qw_30[1]) : "r"(qn_2[6]), "f"(0.0f), "f"(0.0f));
                    q_b_2[12] = qw_30[0];
                    q_b_2[13] = qw_30[1];
                    uint32_t _cvt_f16x2_e2m1x2_50;
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_cvt_f16x2_e2m1x2_50) : "r"((uint32_t)(codes[7] >> 24)));
                    qn_2[7] = _cvt_f16x2_e2m1x2_50;
                    float qw_31[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_31[0]), "=&f"(qw_31[1]) : "r"(qn_2[7]), "f"(0.0f), "f"(0.0f));
                    q_b_2[14] = qw_31[0];
                    q_b_2[15] = qw_31[1];
                    float qe_2[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_2)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_2)[_ls], reinterpret_cast<const float2*>((s_vals + 48))[_ls]);
                    float t8_6[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_6)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_2 + 0))[_la], reinterpret_cast<const float2*>((qe_2 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_6)[_la], reinterpret_cast<const float2*>((t8_6 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_6)[_la], reinterpret_cast<const float2*>((t8_6 + 2))[_la]);
                    qv[3] = t8_6[0] + t8_6[1];
                    uint32_t a0_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_2) : "r"(qn_2[0]), "r"(qn_2[0]), "r"(0));
                    uint32_t b0_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_2) : "r"(qn_2[4]), "r"(qn_2[4]), "r"(0));
                    uint32_t a1_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_2) : "r"(qn_2[1]), "r"(qn_2[1]), "r"(a0_2));
                    uint32_t b1_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_2) : "r"(qn_2[5]), "r"(qn_2[5]), "r"(b0_2));
                    uint32_t a2_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_2) : "r"(qn_2[2]), "r"(qn_2[2]), "r"(a1_2));
                    uint32_t b2_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_2) : "r"(qn_2[6]), "r"(qn_2[6]), "r"(b1_2));
                    uint32_t a3_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_2) : "r"(qn_2[3]), "r"(qn_2[3]), "r"(a2_2));
                    uint32_t b3_2;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_2) : "r"(qn_2[7]), "r"(qn_2[7]), "r"(b2_2));
                    uint32_t v_18;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v_18) : "r"(a3_2), "r"(b3_2));
                    float w_2[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w_2[0]), "=&f"(w_2[1]) : "r"(v_18), "f"(0.0f), "f"(0.0f));
                    qq[3] = w_2[0] + w_2[1];
                    float d_w[4];
                    const float2 _sub2_51 = {m, m};
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        reinterpret_cast<float2*>(d_w)[_ls] = sub_f32x2(reinterpret_cast<float2*>(g_safe)[_ls], _sub2_51);
                    float d_s[4];
                    const float2 _sub2_52 = {4.0f, 4.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        reinterpret_cast<float2*>(d_s)[_ls] = sub_f32x2(reinterpret_cast<float2*>(d_w)[_ls], _sub2_52);
                    float e_s[4];
                    float r_q[4];
                    float _exp2_53 = approx_exp2(d_w[0]);
                    b_weights[0] = ((g[0] > -CUDART_INF_F) ? _exp2_53 : 0.0f);
                    float _exp2_54 = approx_exp2(d_s[0]);
                    e_s[0] = _exp2_54;
                    float _rcp_55 = approx_rcp(qq[0]);
                    r_q[0] = _rcp_55;
                    float _exp2_56 = approx_exp2(d_w[1]);
                    b_weights[1] = ((g[1] > -CUDART_INF_F) ? _exp2_56 : 0.0f);
                    float _exp2_57 = approx_exp2(d_s[1]);
                    e_s[1] = _exp2_57;
                    float _rcp_58 = approx_rcp(qq[1]);
                    r_q[1] = _rcp_58;
                    float _exp2_59 = approx_exp2(d_w[2]);
                    b_weights[2] = ((g[2] > -CUDART_INF_F) ? _exp2_59 : 0.0f);
                    float _exp2_60 = approx_exp2(d_s[2]);
                    e_s[2] = _exp2_60;
                    float _rcp_61 = approx_rcp(qq[2]);
                    r_q[2] = _rcp_61;
                    float _exp2_62 = approx_exp2(d_w[3]);
                    b_weights[3] = ((g[3] > -CUDART_INF_F) ? _exp2_62 : 0.0f);
                    float _exp2_63 = approx_exp2(d_s[3]);
                    e_s[3] = _exp2_63;
                    float _rcp_64 = approx_rcp(qq[3]);
                    r_q[3] = _rcp_64;
                    float fit[4];
                    const float2 _scale2_65 = {448.0f, 448.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        reinterpret_cast<float2*>(fit)[_ls] = mul_f32x2(reinterpret_cast<float2*>(e_s)[_ls], _scale2_65);
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(fit)[_ls], reinterpret_cast<const float2*>(qv)[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(fit)[_ls], reinterpret_cast<const float2*>(r_q)[_ls]);
                    sf_fit[0] = ((g[0] > -CUDART_INF_F && qq[0] > 0.0f) ? fit[0] : 0.0f);
                    sf_fit[1] = ((g[1] > -CUDART_INF_F && qq[1] > 0.0f) ? fit[1] : 0.0f);
                    sf_fit[2] = ((g[2] > -CUDART_INF_F && qq[2] > 0.0f) ? fit[2] : 0.0f);
                    sf_fit[3] = ((g[3] > -CUDART_INF_F && qq[3] > 0.0f) ? fit[3] : 0.0f);
                    unsigned int sf_word[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_fit[0]), "f"(sf_fit[1]),
                                               "f"(sf_fit[2]), "f"(sf_fit[3]));
                        sf_word[0] = _packed;
                    }
                    unsigned int sf_mine = sf_word[0];
                    xch_w[xs * 128 + mine] = sf_mine;
                    asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                    unsigned int sf_other = xch_w[xs * 128 + other];
                    sf_pair[0] = sf_mine;
                    sf_pair[1] = sf_other;
                    const bool _selected_wait_cond_66 = grp == 0;
                    mbarrier_wait((_selected_wait_cond_66 ? (l_ready_addr + (og) * 8) : (l_ready_addr + (og) * 8)), (_selected_wait_cond_66 ? (_phase_l_ready) : (_phase_l_ready_1)));
                    _phase_l_ready ^= 1;
                    _phase_l_ready_1 ^= 1;
                    l_sum = xch_s[og * 128 + mine];
                    l_sum = __fmaf_rn(l_sum, alpha, b_sums[0] * b_weights[0]);
                    l_sum = __fmaf_rn(b_sums[1], b_weights[1], l_sum);
                    l_sum = __fmaf_rn(b_sums[2], b_weights[2], l_sum);
                    l_sum = __fmaf_rn(b_sums[3], b_weights[3], l_sum);
                    xch_s[grp * 128 + mine] = ((j == my_n - 1) ? 0.0f : l_sum);
                    mbarrier_arrive(l_ready_addr + (grp) * 8);
                    const bool _selected_wait_cond_67 = grp == 0;
                    mbarrier_wait((_selected_wait_cond_67 ? (pv_done_addr + (og) * 8) : (pv_done_addr + (og) * 8)), (_selected_wait_cond_67 ? (_phase_pv_done) : (_phase_pv_done_1)));
                    _phase_pv_done ^= 1;
                    _phase_pv_done_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _warp_vote_68 = __any_sync(0xFFFFFFFF, rescale != 0);
                    if (_warp_vote_68 != 0) {
                        float o_chunk_0[32];
                        unsigned int o_addr_0 = taddr + (unsigned int)lane_base_s;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_0[0]), "=f"(o_chunk_0[1]), "=f"(o_chunk_0[2]), "=f"(o_chunk_0[3]), "=f"(o_chunk_0[4]), "=f"(o_chunk_0[5]), "=f"(o_chunk_0[6]), "=f"(o_chunk_0[7]), "=f"(o_chunk_0[8]), "=f"(o_chunk_0[9]), "=f"(o_chunk_0[10]), "=f"(o_chunk_0[11]), "=f"(o_chunk_0[12]), "=f"(o_chunk_0[13]), "=f"(o_chunk_0[14]), "=f"(o_chunk_0[15]), "=f"(o_chunk_0[16]), "=f"(o_chunk_0[17]), "=f"(o_chunk_0[18]), "=f"(o_chunk_0[19]), "=f"(o_chunk_0[20]), "=f"(o_chunk_0[21]), "=f"(o_chunk_0[22]), "=f"(o_chunk_0[23]), "=f"(o_chunk_0[24]), "=f"(o_chunk_0[25]), "=f"(o_chunk_0[26]), "=f"(o_chunk_0[27]), "=f"(o_chunk_0[28]), "=f"(o_chunk_0[29]), "=f"(o_chunk_0[30]), "=f"(o_chunk_0[31])
                            : "r"(o_addr_0));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_69 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_0)[_ls], _scale2_69);
                        tmem_st_x32_f32(o_addr_0, o_chunk_0);
                        float o_chunk_1[32];
                        unsigned int o_addr_1 = taddr + (unsigned int)lane_base_s + 32;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_1[0]), "=f"(o_chunk_1[1]), "=f"(o_chunk_1[2]), "=f"(o_chunk_1[3]), "=f"(o_chunk_1[4]), "=f"(o_chunk_1[5]), "=f"(o_chunk_1[6]), "=f"(o_chunk_1[7]), "=f"(o_chunk_1[8]), "=f"(o_chunk_1[9]), "=f"(o_chunk_1[10]), "=f"(o_chunk_1[11]), "=f"(o_chunk_1[12]), "=f"(o_chunk_1[13]), "=f"(o_chunk_1[14]), "=f"(o_chunk_1[15]), "=f"(o_chunk_1[16]), "=f"(o_chunk_1[17]), "=f"(o_chunk_1[18]), "=f"(o_chunk_1[19]), "=f"(o_chunk_1[20]), "=f"(o_chunk_1[21]), "=f"(o_chunk_1[22]), "=f"(o_chunk_1[23]), "=f"(o_chunk_1[24]), "=f"(o_chunk_1[25]), "=f"(o_chunk_1[26]), "=f"(o_chunk_1[27]), "=f"(o_chunk_1[28]), "=f"(o_chunk_1[29]), "=f"(o_chunk_1[30]), "=f"(o_chunk_1[31])
                            : "r"(o_addr_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_70 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_1)[_ls], _scale2_70);
                        tmem_st_x32_f32(o_addr_1, o_chunk_1);
                        float o_chunk_2[32];
                        unsigned int o_addr_2 = taddr + (unsigned int)lane_base_s + 64;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_2[0]), "=f"(o_chunk_2[1]), "=f"(o_chunk_2[2]), "=f"(o_chunk_2[3]), "=f"(o_chunk_2[4]), "=f"(o_chunk_2[5]), "=f"(o_chunk_2[6]), "=f"(o_chunk_2[7]), "=f"(o_chunk_2[8]), "=f"(o_chunk_2[9]), "=f"(o_chunk_2[10]), "=f"(o_chunk_2[11]), "=f"(o_chunk_2[12]), "=f"(o_chunk_2[13]), "=f"(o_chunk_2[14]), "=f"(o_chunk_2[15]), "=f"(o_chunk_2[16]), "=f"(o_chunk_2[17]), "=f"(o_chunk_2[18]), "=f"(o_chunk_2[19]), "=f"(o_chunk_2[20]), "=f"(o_chunk_2[21]), "=f"(o_chunk_2[22]), "=f"(o_chunk_2[23]), "=f"(o_chunk_2[24]), "=f"(o_chunk_2[25]), "=f"(o_chunk_2[26]), "=f"(o_chunk_2[27]), "=f"(o_chunk_2[28]), "=f"(o_chunk_2[29]), "=f"(o_chunk_2[30]), "=f"(o_chunk_2[31])
                            : "r"(o_addr_2));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_71 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_2)[_ls], _scale2_71);
                        tmem_st_x32_f32(o_addr_2, o_chunk_2);
                        float o_chunk_3[32];
                        unsigned int o_addr_3 = taddr + (unsigned int)lane_base_s + 96;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_3[0]), "=f"(o_chunk_3[1]), "=f"(o_chunk_3[2]), "=f"(o_chunk_3[3]), "=f"(o_chunk_3[4]), "=f"(o_chunk_3[5]), "=f"(o_chunk_3[6]), "=f"(o_chunk_3[7]), "=f"(o_chunk_3[8]), "=f"(o_chunk_3[9]), "=f"(o_chunk_3[10]), "=f"(o_chunk_3[11]), "=f"(o_chunk_3[12]), "=f"(o_chunk_3[13]), "=f"(o_chunk_3[14]), "=f"(o_chunk_3[15]), "=f"(o_chunk_3[16]), "=f"(o_chunk_3[17]), "=f"(o_chunk_3[18]), "=f"(o_chunk_3[19]), "=f"(o_chunk_3[20]), "=f"(o_chunk_3[21]), "=f"(o_chunk_3[22]), "=f"(o_chunk_3[23]), "=f"(o_chunk_3[24]), "=f"(o_chunk_3[25]), "=f"(o_chunk_3[26]), "=f"(o_chunk_3[27]), "=f"(o_chunk_3[28]), "=f"(o_chunk_3[29]), "=f"(o_chunk_3[30]), "=f"(o_chunk_3[31])
                            : "r"(o_addr_3));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_72 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_3)[_ls], _scale2_72);
                        tmem_st_x32_f32(o_addr_3, o_chunk_3);
                        float o_chunk_4[32];
                        unsigned int o_addr_4 = taddr + (unsigned int)lane_base_s + 128;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_4[0]), "=f"(o_chunk_4[1]), "=f"(o_chunk_4[2]), "=f"(o_chunk_4[3]), "=f"(o_chunk_4[4]), "=f"(o_chunk_4[5]), "=f"(o_chunk_4[6]), "=f"(o_chunk_4[7]), "=f"(o_chunk_4[8]), "=f"(o_chunk_4[9]), "=f"(o_chunk_4[10]), "=f"(o_chunk_4[11]), "=f"(o_chunk_4[12]), "=f"(o_chunk_4[13]), "=f"(o_chunk_4[14]), "=f"(o_chunk_4[15]), "=f"(o_chunk_4[16]), "=f"(o_chunk_4[17]), "=f"(o_chunk_4[18]), "=f"(o_chunk_4[19]), "=f"(o_chunk_4[20]), "=f"(o_chunk_4[21]), "=f"(o_chunk_4[22]), "=f"(o_chunk_4[23]), "=f"(o_chunk_4[24]), "=f"(o_chunk_4[25]), "=f"(o_chunk_4[26]), "=f"(o_chunk_4[27]), "=f"(o_chunk_4[28]), "=f"(o_chunk_4[29]), "=f"(o_chunk_4[30]), "=f"(o_chunk_4[31])
                            : "r"(o_addr_4));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_73 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_4)[_ls], _scale2_73);
                        tmem_st_x32_f32(o_addr_4, o_chunk_4);
                        float o_chunk_5[32];
                        unsigned int o_addr_5 = taddr + (unsigned int)lane_base_s + 160;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_5[0]), "=f"(o_chunk_5[1]), "=f"(o_chunk_5[2]), "=f"(o_chunk_5[3]), "=f"(o_chunk_5[4]), "=f"(o_chunk_5[5]), "=f"(o_chunk_5[6]), "=f"(o_chunk_5[7]), "=f"(o_chunk_5[8]), "=f"(o_chunk_5[9]), "=f"(o_chunk_5[10]), "=f"(o_chunk_5[11]), "=f"(o_chunk_5[12]), "=f"(o_chunk_5[13]), "=f"(o_chunk_5[14]), "=f"(o_chunk_5[15]), "=f"(o_chunk_5[16]), "=f"(o_chunk_5[17]), "=f"(o_chunk_5[18]), "=f"(o_chunk_5[19]), "=f"(o_chunk_5[20]), "=f"(o_chunk_5[21]), "=f"(o_chunk_5[22]), "=f"(o_chunk_5[23]), "=f"(o_chunk_5[24]), "=f"(o_chunk_5[25]), "=f"(o_chunk_5[26]), "=f"(o_chunk_5[27]), "=f"(o_chunk_5[28]), "=f"(o_chunk_5[29]), "=f"(o_chunk_5[30]), "=f"(o_chunk_5[31])
                            : "r"(o_addr_5));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_74 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_5)[_ls], _scale2_74);
                        tmem_st_x32_f32(o_addr_5, o_chunk_5);
                        float o_chunk_6[32];
                        unsigned int o_addr_6 = taddr + (unsigned int)lane_base_s + 192;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_6[0]), "=f"(o_chunk_6[1]), "=f"(o_chunk_6[2]), "=f"(o_chunk_6[3]), "=f"(o_chunk_6[4]), "=f"(o_chunk_6[5]), "=f"(o_chunk_6[6]), "=f"(o_chunk_6[7]), "=f"(o_chunk_6[8]), "=f"(o_chunk_6[9]), "=f"(o_chunk_6[10]), "=f"(o_chunk_6[11]), "=f"(o_chunk_6[12]), "=f"(o_chunk_6[13]), "=f"(o_chunk_6[14]), "=f"(o_chunk_6[15]), "=f"(o_chunk_6[16]), "=f"(o_chunk_6[17]), "=f"(o_chunk_6[18]), "=f"(o_chunk_6[19]), "=f"(o_chunk_6[20]), "=f"(o_chunk_6[21]), "=f"(o_chunk_6[22]), "=f"(o_chunk_6[23]), "=f"(o_chunk_6[24]), "=f"(o_chunk_6[25]), "=f"(o_chunk_6[26]), "=f"(o_chunk_6[27]), "=f"(o_chunk_6[28]), "=f"(o_chunk_6[29]), "=f"(o_chunk_6[30]), "=f"(o_chunk_6[31])
                            : "r"(o_addr_6));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_75 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_6)[_ls], _scale2_75);
                        tmem_st_x32_f32(o_addr_6, o_chunk_6);
                        float o_chunk_7[32];
                        unsigned int o_addr_7 = taddr + (unsigned int)lane_base_s + 224;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_chunk_7[0]), "=f"(o_chunk_7[1]), "=f"(o_chunk_7[2]), "=f"(o_chunk_7[3]), "=f"(o_chunk_7[4]), "=f"(o_chunk_7[5]), "=f"(o_chunk_7[6]), "=f"(o_chunk_7[7]), "=f"(o_chunk_7[8]), "=f"(o_chunk_7[9]), "=f"(o_chunk_7[10]), "=f"(o_chunk_7[11]), "=f"(o_chunk_7[12]), "=f"(o_chunk_7[13]), "=f"(o_chunk_7[14]), "=f"(o_chunk_7[15]), "=f"(o_chunk_7[16]), "=f"(o_chunk_7[17]), "=f"(o_chunk_7[18]), "=f"(o_chunk_7[19]), "=f"(o_chunk_7[20]), "=f"(o_chunk_7[21]), "=f"(o_chunk_7[22]), "=f"(o_chunk_7[23]), "=f"(o_chunk_7[24]), "=f"(o_chunk_7[25]), "=f"(o_chunk_7[26]), "=f"(o_chunk_7[27]), "=f"(o_chunk_7[28]), "=f"(o_chunk_7[29]), "=f"(o_chunk_7[30]), "=f"(o_chunk_7[31])
                            : "r"(o_addr_7));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_76 = {alpha, alpha};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_7)[_ls], _scale2_76);
                        tmem_st_x32_f32(o_addr_7, o_chunk_7);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_p_addr + (unsigned int)(grp * 4096 + r * 64 + (2 * half ^ r >> 1 & 3) * 16)), "r"(codes[0]), "r"(codes[1]), "r"(codes[2]), "r"(codes[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_p_addr + (unsigned int)(grp * 4096 + r * 64 + (2 * half + 1 ^ r >> 1 & 3) * 16)), "r"(codes[4]), "r"(codes[5]), "r"(codes[6]), "r"(codes[7]) : "memory");
                    unsigned int w_lo = ((half == 0) ? sf_pair[0] : sf_pair[1]);
                    unsigned int w_hi = ((half == 0) ? sf_pair[1] : sf_pair[0]);
                    unsigned int sfa_p[8];
                    sfa_p[0] = w_lo;
                    sfa_p[4] = w_hi;
                    sfa_p[1] = w_lo;
                    sfa_p[5] = w_hi;
                    sfa_p[2] = w_lo;
                    sfa_p[6] = w_hi;
                    sfa_p[3] = w_lo;
                    sfa_p[7] = w_hi;
                    tmem_st_x8_u32(taddr + (unsigned int)lane_base_s + (unsigned int)(480 + 8 * grp), (const uint32_t*)sfa_p);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (grp) * 8), "r"(0) : "memory");
                    _phase_pv_done_2 ^= 1;
                    h_s = h_s + 1;
                }
                int last_grp = (g0_s + my_n - 1) % 2;
                if (last_grp == grp) {
                    xch_l[grp * 128 + mine] = l_sum;
                    asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                    float l_row = l_sum + xch_l[grp * 128 + other];
                    float gk_e = *kv_gs;
                    float _rcp_77 = approx_rcp(gk_e * l_row);
                    float inv = ((l_row > 0.0f) ? 0.005952381f * _rcp_77 : 0.0f);
                    int direct = num_split == 1;
                    int row_g = row_base + row_local;
                    int o_base = (row_g * num_split + split_i) * 512 + 128 * half;
                    mbarrier_wait(pv_done_addr + (grp) * 8, _phase_pv_done_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int box0 = smem_os_addr + (unsigned int)((4 * grp + qs) * 4096);
                    int row_w = row0 + cta_rank * 64 + qs % 2 * 32;
                    unsigned int box_0 = box0 + 0;
                    float o_pr_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_0[0]), "=f"(o_pr_0[1]), "=f"(o_pr_0[2]), "=f"(o_pr_0[3]), "=f"(o_pr_0[4]), "=f"(o_pr_0[5]), "=f"(o_pr_0[6]), "=f"(o_pr_0[7]), "=f"(o_pr_0[8]), "=f"(o_pr_0[9]), "=f"(o_pr_0[10]), "=f"(o_pr_0[11]), "=f"(o_pr_0[12]), "=f"(o_pr_0[13]), "=f"(o_pr_0[14]), "=f"(o_pr_0[15]), "=f"(o_pr_0[16]), "=f"(o_pr_0[17]), "=f"(o_pr_0[18]), "=f"(o_pr_0[19]), "=f"(o_pr_0[20]), "=f"(o_pr_0[21]), "=f"(o_pr_0[22]), "=f"(o_pr_0[23]), "=f"(o_pr_0[24]), "=f"(o_pr_0[25]), "=f"(o_pr_0[26]), "=f"(o_pr_0[27]), "=f"(o_pr_0[28]), "=f"(o_pr_0[29]), "=f"(o_pr_0[30]), "=f"(o_pr_0[31])
                        : "r"(taddr + (unsigned int)lane_base_s));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_0[32]), "=f"(o_pr_0[33]), "=f"(o_pr_0[34]), "=f"(o_pr_0[35]), "=f"(o_pr_0[36]), "=f"(o_pr_0[37]), "=f"(o_pr_0[38]), "=f"(o_pr_0[39]), "=f"(o_pr_0[40]), "=f"(o_pr_0[41]), "=f"(o_pr_0[42]), "=f"(o_pr_0[43]), "=f"(o_pr_0[44]), "=f"(o_pr_0[45]), "=f"(o_pr_0[46]), "=f"(o_pr_0[47]), "=f"(o_pr_0[48]), "=f"(o_pr_0[49]), "=f"(o_pr_0[50]), "=f"(o_pr_0[51]), "=f"(o_pr_0[52]), "=f"(o_pr_0[53]), "=f"(o_pr_0[54]), "=f"(o_pr_0[55]), "=f"(o_pr_0[56]), "=f"(o_pr_0[57]), "=f"(o_pr_0[58]), "=f"(o_pr_0[59]), "=f"(o_pr_0[60]), "=f"(o_pr_0[61]), "=f"(o_pr_0[62]), "=f"(o_pr_0[63])
                        : "r"(taddr + (unsigned int)lane_base_s + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_78 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_0 + 0))[_ls], _scale2_78);
                    unsigned int o_bf16_0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_0 + 0)[_lp*2 + 0], (o_pr_0 + 0)[_lp*2+1 + 0]));
                        o_bf16_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (0 ^ lane % 8) * 16)), "r"(o_bf16_0[0]), "r"(o_bf16_0[1]), "r"(o_bf16_0[2]), "r"(o_bf16_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (1 ^ lane % 8) * 16)), "r"(o_bf16_0[4]), "r"(o_bf16_0[5]), "r"(o_bf16_0[6]), "r"(o_bf16_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (2 ^ lane % 8) * 16)), "r"(o_bf16_0[8]), "r"(o_bf16_0[9]), "r"(o_bf16_0[10]), "r"(o_bf16_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (3 ^ lane % 8) * 16)), "r"(o_bf16_0[12]), "r"(o_bf16_0[13]), "r"(o_bf16_0[14]), "r"(o_bf16_0[15]) : "memory");
                    const float2 _scale2_79 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_0 + 32))[_ls], _scale2_79);
                    unsigned int o_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_0 + 32)[_lp*2 + 0], (o_pr_0 + 32)[_lp*2+1 + 0]));
                        o_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (4 ^ lane % 8) * 16)), "r"(o_bf16_1[0]), "r"(o_bf16_1[1]), "r"(o_bf16_1[2]), "r"(o_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (5 ^ lane % 8) * 16)), "r"(o_bf16_1[4]), "r"(o_bf16_1[5]), "r"(o_bf16_1[6]), "r"(o_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (6 ^ lane % 8) * 16)), "r"(o_bf16_1[8]), "r"(o_bf16_1[9]), "r"(o_bf16_1[10]), "r"(o_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_0 + (unsigned int)(lane * 128 + (7 ^ lane % 8) * 16)), "r"(o_bf16_1[12]), "r"(o_bf16_1[13]), "r"(o_bf16_1[14]), "r"(o_bf16_1[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    int dim_b_0 = 128 * half;
                    if (elect_sync()) {
                        if (direct != 0) {
                            tma_store_3d((&out_map), dim_b_0, row_w, b_i, box_0);
                        } else {
                            tma_store_4d((&po_map), dim_b_0, split_i, row_w, b_i, box_0);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    unsigned int box_1 = box0 + 0;
                    float o_pr_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_1[0]), "=f"(o_pr_1[1]), "=f"(o_pr_1[2]), "=f"(o_pr_1[3]), "=f"(o_pr_1[4]), "=f"(o_pr_1[5]), "=f"(o_pr_1[6]), "=f"(o_pr_1[7]), "=f"(o_pr_1[8]), "=f"(o_pr_1[9]), "=f"(o_pr_1[10]), "=f"(o_pr_1[11]), "=f"(o_pr_1[12]), "=f"(o_pr_1[13]), "=f"(o_pr_1[14]), "=f"(o_pr_1[15]), "=f"(o_pr_1[16]), "=f"(o_pr_1[17]), "=f"(o_pr_1[18]), "=f"(o_pr_1[19]), "=f"(o_pr_1[20]), "=f"(o_pr_1[21]), "=f"(o_pr_1[22]), "=f"(o_pr_1[23]), "=f"(o_pr_1[24]), "=f"(o_pr_1[25]), "=f"(o_pr_1[26]), "=f"(o_pr_1[27]), "=f"(o_pr_1[28]), "=f"(o_pr_1[29]), "=f"(o_pr_1[30]), "=f"(o_pr_1[31])
                        : "r"(taddr + (unsigned int)lane_base_s + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_1[32]), "=f"(o_pr_1[33]), "=f"(o_pr_1[34]), "=f"(o_pr_1[35]), "=f"(o_pr_1[36]), "=f"(o_pr_1[37]), "=f"(o_pr_1[38]), "=f"(o_pr_1[39]), "=f"(o_pr_1[40]), "=f"(o_pr_1[41]), "=f"(o_pr_1[42]), "=f"(o_pr_1[43]), "=f"(o_pr_1[44]), "=f"(o_pr_1[45]), "=f"(o_pr_1[46]), "=f"(o_pr_1[47]), "=f"(o_pr_1[48]), "=f"(o_pr_1[49]), "=f"(o_pr_1[50]), "=f"(o_pr_1[51]), "=f"(o_pr_1[52]), "=f"(o_pr_1[53]), "=f"(o_pr_1[54]), "=f"(o_pr_1[55]), "=f"(o_pr_1[56]), "=f"(o_pr_1[57]), "=f"(o_pr_1[58]), "=f"(o_pr_1[59]), "=f"(o_pr_1[60]), "=f"(o_pr_1[61]), "=f"(o_pr_1[62]), "=f"(o_pr_1[63])
                        : "r"(taddr + (unsigned int)lane_base_s + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_80 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_1 + 0))[_ls], _scale2_80);
                    unsigned int o_bf16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_1 + 0)[_lp*2 + 0], (o_pr_1 + 0)[_lp*2+1 + 0]));
                        o_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (0 ^ lane % 8) * 16)), "r"(o_bf16_2[0]), "r"(o_bf16_2[1]), "r"(o_bf16_2[2]), "r"(o_bf16_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (1 ^ lane % 8) * 16)), "r"(o_bf16_2[4]), "r"(o_bf16_2[5]), "r"(o_bf16_2[6]), "r"(o_bf16_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (2 ^ lane % 8) * 16)), "r"(o_bf16_2[8]), "r"(o_bf16_2[9]), "r"(o_bf16_2[10]), "r"(o_bf16_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (3 ^ lane % 8) * 16)), "r"(o_bf16_2[12]), "r"(o_bf16_2[13]), "r"(o_bf16_2[14]), "r"(o_bf16_2[15]) : "memory");
                    const float2 _scale2_81 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_1 + 32))[_ls], _scale2_81);
                    unsigned int o_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_1 + 32)[_lp*2 + 0], (o_pr_1 + 32)[_lp*2+1 + 0]));
                        o_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (4 ^ lane % 8) * 16)), "r"(o_bf16_3[0]), "r"(o_bf16_3[1]), "r"(o_bf16_3[2]), "r"(o_bf16_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (5 ^ lane % 8) * 16)), "r"(o_bf16_3[4]), "r"(o_bf16_3[5]), "r"(o_bf16_3[6]), "r"(o_bf16_3[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (6 ^ lane % 8) * 16)), "r"(o_bf16_3[8]), "r"(o_bf16_3[9]), "r"(o_bf16_3[10]), "r"(o_bf16_3[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_1 + (unsigned int)(lane * 128 + (7 ^ lane % 8) * 16)), "r"(o_bf16_3[12]), "r"(o_bf16_3[13]), "r"(o_bf16_3[14]), "r"(o_bf16_3[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    int dim_b_1 = 128 * half + 64;
                    if (elect_sync()) {
                        if (direct != 0) {
                            tma_store_3d((&out_map), dim_b_1, row_w, b_i, box_1);
                        } else {
                            tma_store_4d((&po_map), dim_b_1, split_i, row_w, b_i, box_1);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    unsigned int box_2 = box0 + 0;
                    float o_pr_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_2[0]), "=f"(o_pr_2[1]), "=f"(o_pr_2[2]), "=f"(o_pr_2[3]), "=f"(o_pr_2[4]), "=f"(o_pr_2[5]), "=f"(o_pr_2[6]), "=f"(o_pr_2[7]), "=f"(o_pr_2[8]), "=f"(o_pr_2[9]), "=f"(o_pr_2[10]), "=f"(o_pr_2[11]), "=f"(o_pr_2[12]), "=f"(o_pr_2[13]), "=f"(o_pr_2[14]), "=f"(o_pr_2[15]), "=f"(o_pr_2[16]), "=f"(o_pr_2[17]), "=f"(o_pr_2[18]), "=f"(o_pr_2[19]), "=f"(o_pr_2[20]), "=f"(o_pr_2[21]), "=f"(o_pr_2[22]), "=f"(o_pr_2[23]), "=f"(o_pr_2[24]), "=f"(o_pr_2[25]), "=f"(o_pr_2[26]), "=f"(o_pr_2[27]), "=f"(o_pr_2[28]), "=f"(o_pr_2[29]), "=f"(o_pr_2[30]), "=f"(o_pr_2[31])
                        : "r"(taddr + (unsigned int)lane_base_s + 128));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_2[32]), "=f"(o_pr_2[33]), "=f"(o_pr_2[34]), "=f"(o_pr_2[35]), "=f"(o_pr_2[36]), "=f"(o_pr_2[37]), "=f"(o_pr_2[38]), "=f"(o_pr_2[39]), "=f"(o_pr_2[40]), "=f"(o_pr_2[41]), "=f"(o_pr_2[42]), "=f"(o_pr_2[43]), "=f"(o_pr_2[44]), "=f"(o_pr_2[45]), "=f"(o_pr_2[46]), "=f"(o_pr_2[47]), "=f"(o_pr_2[48]), "=f"(o_pr_2[49]), "=f"(o_pr_2[50]), "=f"(o_pr_2[51]), "=f"(o_pr_2[52]), "=f"(o_pr_2[53]), "=f"(o_pr_2[54]), "=f"(o_pr_2[55]), "=f"(o_pr_2[56]), "=f"(o_pr_2[57]), "=f"(o_pr_2[58]), "=f"(o_pr_2[59]), "=f"(o_pr_2[60]), "=f"(o_pr_2[61]), "=f"(o_pr_2[62]), "=f"(o_pr_2[63])
                        : "r"(taddr + (unsigned int)lane_base_s + 160));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_82 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_2 + 0))[_ls], _scale2_82);
                    unsigned int o_bf16_4[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_2 + 0)[_lp*2 + 0], (o_pr_2 + 0)[_lp*2+1 + 0]));
                        o_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (0 ^ lane % 8) * 16)), "r"(o_bf16_4[0]), "r"(o_bf16_4[1]), "r"(o_bf16_4[2]), "r"(o_bf16_4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (1 ^ lane % 8) * 16)), "r"(o_bf16_4[4]), "r"(o_bf16_4[5]), "r"(o_bf16_4[6]), "r"(o_bf16_4[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (2 ^ lane % 8) * 16)), "r"(o_bf16_4[8]), "r"(o_bf16_4[9]), "r"(o_bf16_4[10]), "r"(o_bf16_4[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (3 ^ lane % 8) * 16)), "r"(o_bf16_4[12]), "r"(o_bf16_4[13]), "r"(o_bf16_4[14]), "r"(o_bf16_4[15]) : "memory");
                    const float2 _scale2_83 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_2 + 32))[_ls], _scale2_83);
                    unsigned int o_bf16_5[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_2 + 32)[_lp*2 + 0], (o_pr_2 + 32)[_lp*2+1 + 0]));
                        o_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (4 ^ lane % 8) * 16)), "r"(o_bf16_5[0]), "r"(o_bf16_5[1]), "r"(o_bf16_5[2]), "r"(o_bf16_5[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (5 ^ lane % 8) * 16)), "r"(o_bf16_5[4]), "r"(o_bf16_5[5]), "r"(o_bf16_5[6]), "r"(o_bf16_5[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (6 ^ lane % 8) * 16)), "r"(o_bf16_5[8]), "r"(o_bf16_5[9]), "r"(o_bf16_5[10]), "r"(o_bf16_5[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_2 + (unsigned int)(lane * 128 + (7 ^ lane % 8) * 16)), "r"(o_bf16_5[12]), "r"(o_bf16_5[13]), "r"(o_bf16_5[14]), "r"(o_bf16_5[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    int dim_b_2 = 256 + 128 * half;
                    if (elect_sync()) {
                        if (direct != 0) {
                            tma_store_3d((&out_map), dim_b_2, row_w, b_i, box_2);
                        } else {
                            tma_store_4d((&po_map), dim_b_2, split_i, row_w, b_i, box_2);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    unsigned int box_3 = box0 + 0;
                    float o_pr_3[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_3[0]), "=f"(o_pr_3[1]), "=f"(o_pr_3[2]), "=f"(o_pr_3[3]), "=f"(o_pr_3[4]), "=f"(o_pr_3[5]), "=f"(o_pr_3[6]), "=f"(o_pr_3[7]), "=f"(o_pr_3[8]), "=f"(o_pr_3[9]), "=f"(o_pr_3[10]), "=f"(o_pr_3[11]), "=f"(o_pr_3[12]), "=f"(o_pr_3[13]), "=f"(o_pr_3[14]), "=f"(o_pr_3[15]), "=f"(o_pr_3[16]), "=f"(o_pr_3[17]), "=f"(o_pr_3[18]), "=f"(o_pr_3[19]), "=f"(o_pr_3[20]), "=f"(o_pr_3[21]), "=f"(o_pr_3[22]), "=f"(o_pr_3[23]), "=f"(o_pr_3[24]), "=f"(o_pr_3[25]), "=f"(o_pr_3[26]), "=f"(o_pr_3[27]), "=f"(o_pr_3[28]), "=f"(o_pr_3[29]), "=f"(o_pr_3[30]), "=f"(o_pr_3[31])
                        : "r"(taddr + (unsigned int)lane_base_s + 192));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_pr_3[32]), "=f"(o_pr_3[33]), "=f"(o_pr_3[34]), "=f"(o_pr_3[35]), "=f"(o_pr_3[36]), "=f"(o_pr_3[37]), "=f"(o_pr_3[38]), "=f"(o_pr_3[39]), "=f"(o_pr_3[40]), "=f"(o_pr_3[41]), "=f"(o_pr_3[42]), "=f"(o_pr_3[43]), "=f"(o_pr_3[44]), "=f"(o_pr_3[45]), "=f"(o_pr_3[46]), "=f"(o_pr_3[47]), "=f"(o_pr_3[48]), "=f"(o_pr_3[49]), "=f"(o_pr_3[50]), "=f"(o_pr_3[51]), "=f"(o_pr_3[52]), "=f"(o_pr_3[53]), "=f"(o_pr_3[54]), "=f"(o_pr_3[55]), "=f"(o_pr_3[56]), "=f"(o_pr_3[57]), "=f"(o_pr_3[58]), "=f"(o_pr_3[59]), "=f"(o_pr_3[60]), "=f"(o_pr_3[61]), "=f"(o_pr_3[62]), "=f"(o_pr_3[63])
                        : "r"(taddr + (unsigned int)lane_base_s + 224));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_84 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_3 + 0))[_ls], _scale2_84);
                    unsigned int o_bf16_6[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_3 + 0)[_lp*2 + 0], (o_pr_3 + 0)[_lp*2+1 + 0]));
                        o_bf16_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (0 ^ lane % 8) * 16)), "r"(o_bf16_6[0]), "r"(o_bf16_6[1]), "r"(o_bf16_6[2]), "r"(o_bf16_6[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (1 ^ lane % 8) * 16)), "r"(o_bf16_6[4]), "r"(o_bf16_6[5]), "r"(o_bf16_6[6]), "r"(o_bf16_6[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (2 ^ lane % 8) * 16)), "r"(o_bf16_6[8]), "r"(o_bf16_6[9]), "r"(o_bf16_6[10]), "r"(o_bf16_6[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (3 ^ lane % 8) * 16)), "r"(o_bf16_6[12]), "r"(o_bf16_6[13]), "r"(o_bf16_6[14]), "r"(o_bf16_6[15]) : "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_free_addr), "r"(0) : "memory");
                    }
                    const float2 _scale2_85 = {inv, inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_pr_3 + 32))[_ls], _scale2_85);
                    unsigned int o_bf16_7[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_pr_3 + 32)[_lp*2 + 0], (o_pr_3 + 32)[_lp*2+1 + 0]));
                        o_bf16_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (4 ^ lane % 8) * 16)), "r"(o_bf16_7[0]), "r"(o_bf16_7[1]), "r"(o_bf16_7[2]), "r"(o_bf16_7[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (5 ^ lane % 8) * 16)), "r"(o_bf16_7[4]), "r"(o_bf16_7[5]), "r"(o_bf16_7[6]), "r"(o_bf16_7[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (6 ^ lane % 8) * 16)), "r"(o_bf16_7[8]), "r"(o_bf16_7[9]), "r"(o_bf16_7[10]), "r"(o_bf16_7[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(box_3 + (unsigned int)(lane * 128 + (7 ^ lane % 8) * 16)), "r"(o_bf16_7[12]), "r"(o_bf16_7[13]), "r"(o_bf16_7[14]), "r"(o_bf16_7[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    int dim_b_3 = 256 + 128 * half + 64;
                    if (elect_sync()) {
                        if (direct != 0) {
                            tma_store_3d((&out_map), dim_b_3, row_w, b_i, box_3);
                        } else {
                            tma_store_4d((&po_map), dim_b_3, split_i, row_w, b_i, box_3);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (row_ok_c != 0 && half == 0 && direct == 0) {
                        partial_max[row_g * num_split + split_i] = m;
                        partial_sum[row_g * num_split + split_i] = l_row;
                    }
                }
                g0_s = g0_s + my_n;
            }
            mbarrier_wait(pv_done_addr + (grp) * 8, _phase_pv_done_2);
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            __syncwarp();
        }
    }
    // ---- Role: xpose ----
    if (warp >= 8 && warp <= 11) {
        { // xpose_main
            int xw = make_warp_uniform(warp - 8);
            int par_row = lane & 15;
            int raw_lane = (lane >> 4) * 4096 + par_row * 64 + (xw ^ par_row >> 1 & 3) * 16;
            int swap_x = lane >> 3 & 1 ^ lane >> 2 & 1;
            int dim_row = xw * 32 + (lane & 7) * 2 + (lane >> 4) * 16 + swap_x;
            int raw_x = 0;
            int vt_x = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_vt_empty = 1;
            #pragma unroll 1
            for (int k_it_1 = 0; k_it_1 < n_items; k_it_1++) {
                int it_1 = clu + k_it_1 * n_clu;
                int split_i_1 = it_1 % num_split;
                int rest_1 = it_1 / num_split;
                int b_i_1 = rest_1 / m_tiles;
                int rows_b_1 = q_len * num_heads;
                int row0_1 = rest_1 % m_tiles * 128;
                int rows_left_1 = rows_b_1 - row0_1;
                int rows_valid_1 = ((rows_left_1 > 128) ? 128 : rows_left_1);
                int row_base_1 = b_i_1 * rows_b_1 + row0_1;
                int kv_len_i_1 = kv_lens[b_i_1];
                int t_last_1 = (row0_1 + rows_valid_1 - 1) / num_heads;
                int kv_end_raw_1 = kv_len_i_1 - (q_len - 1 - t_last_1);
                int kv_end_1 = ((kv_end_raw_1 < 0) ? 0 : kv_end_raw_1);
                int n_pages_1 = (kv_end_1 + 128 - 1) / 128;
                int pps_1 = (n_pages_1 + num_split - 1) / num_split;
                int p_start_1 = split_i_1 * pps_1;
                int p_end_raw_1 = p_start_1 + pps_1;
                int p_end_1 = ((p_end_raw_1 > n_pages_1) ? n_pages_1 : p_end_raw_1);
                int n_raw_1 = p_end_1 - p_start_1;
                int empty_1 = ((n_raw_1 < 1) ? 1 : 0);
                int my_n_1 = ((n_raw_1 < 1) ? 1 : n_raw_1);
                int my_start_1 = ((n_raw_1 < 1) ? 0 : p_start_1);
                int keys_1 = ((empty_1 != 0) ? 0 : kv_end_1);
                int qc = make_warp_uniform(warp % 4);
                int lane_base_c = qc * 32 << 16;
                int row_c = cta_rank * 64 + qc % 2 * 32 + lane;
                int row_ok = row_c < rows_valid_1;
                int q_row = ((row_ok != 0) ? row_base_1 + row_c : row_base_1);
                unsigned int a_words[80];
                {
                    uint4 _uv4_0 = *reinterpret_cast<const uint4*>(q + q_row * 80 + 0);
                    a_words[0 + 0] = _uv4_0.x;
                    a_words[0 + 1] = _uv4_0.y;
                    a_words[0 + 2] = _uv4_0.z;
                    a_words[0 + 3] = _uv4_0.w;
                }
                {
                    uint4 _uv4_1 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 4) + 0);
                    a_words[4 + 0] = _uv4_1.x;
                    a_words[4 + 1] = _uv4_1.y;
                    a_words[4 + 2] = _uv4_1.z;
                    a_words[4 + 3] = _uv4_1.w;
                }
                {
                    uint4 _uv4_2 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 8) + 0);
                    a_words[8 + 0] = _uv4_2.x;
                    a_words[8 + 1] = _uv4_2.y;
                    a_words[8 + 2] = _uv4_2.z;
                    a_words[8 + 3] = _uv4_2.w;
                }
                {
                    uint4 _uv4_3 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 12) + 0);
                    a_words[12 + 0] = _uv4_3.x;
                    a_words[12 + 1] = _uv4_3.y;
                    a_words[12 + 2] = _uv4_3.z;
                    a_words[12 + 3] = _uv4_3.w;
                }
                {
                    uint4 _uv4_4 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 16) + 0);
                    a_words[16 + 0] = _uv4_4.x;
                    a_words[16 + 1] = _uv4_4.y;
                    a_words[16 + 2] = _uv4_4.z;
                    a_words[16 + 3] = _uv4_4.w;
                }
                {
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 20) + 0);
                    a_words[20 + 0] = _uv4_5.x;
                    a_words[20 + 1] = _uv4_5.y;
                    a_words[20 + 2] = _uv4_5.z;
                    a_words[20 + 3] = _uv4_5.w;
                }
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 24) + 0);
                    a_words[24 + 0] = _uv4_6.x;
                    a_words[24 + 1] = _uv4_6.y;
                    a_words[24 + 2] = _uv4_6.z;
                    a_words[24 + 3] = _uv4_6.w;
                }
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 28) + 0);
                    a_words[28 + 0] = _uv4_7.x;
                    a_words[28 + 1] = _uv4_7.y;
                    a_words[28 + 2] = _uv4_7.z;
                    a_words[28 + 3] = _uv4_7.w;
                }
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 32) + 0);
                    a_words[32 + 0] = _uv4_8.x;
                    a_words[32 + 1] = _uv4_8.y;
                    a_words[32 + 2] = _uv4_8.z;
                    a_words[32 + 3] = _uv4_8.w;
                }
                {
                    uint4 _uv4_9 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 36) + 0);
                    a_words[36 + 0] = _uv4_9.x;
                    a_words[36 + 1] = _uv4_9.y;
                    a_words[36 + 2] = _uv4_9.z;
                    a_words[36 + 3] = _uv4_9.w;
                }
                {
                    uint4 _uv4_10 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 40) + 0);
                    a_words[40 + 0] = _uv4_10.x;
                    a_words[40 + 1] = _uv4_10.y;
                    a_words[40 + 2] = _uv4_10.z;
                    a_words[40 + 3] = _uv4_10.w;
                }
                {
                    uint4 _uv4_11 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 44) + 0);
                    a_words[44 + 0] = _uv4_11.x;
                    a_words[44 + 1] = _uv4_11.y;
                    a_words[44 + 2] = _uv4_11.z;
                    a_words[44 + 3] = _uv4_11.w;
                }
                {
                    uint4 _uv4_12 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 48) + 0);
                    a_words[48 + 0] = _uv4_12.x;
                    a_words[48 + 1] = _uv4_12.y;
                    a_words[48 + 2] = _uv4_12.z;
                    a_words[48 + 3] = _uv4_12.w;
                }
                {
                    uint4 _uv4_13 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 52) + 0);
                    a_words[52 + 0] = _uv4_13.x;
                    a_words[52 + 1] = _uv4_13.y;
                    a_words[52 + 2] = _uv4_13.z;
                    a_words[52 + 3] = _uv4_13.w;
                }
                {
                    uint4 _uv4_14 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 56) + 0);
                    a_words[56 + 0] = _uv4_14.x;
                    a_words[56 + 1] = _uv4_14.y;
                    a_words[56 + 2] = _uv4_14.z;
                    a_words[56 + 3] = _uv4_14.w;
                }
                {
                    uint4 _uv4_15 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 60) + 0);
                    a_words[60 + 0] = _uv4_15.x;
                    a_words[60 + 1] = _uv4_15.y;
                    a_words[60 + 2] = _uv4_15.z;
                    a_words[60 + 3] = _uv4_15.w;
                }
                unsigned int rope[16];
                {
                    uint4 _uv4_16 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 64) + 0);
                    rope[0 + 0] = _uv4_16.x;
                    rope[0 + 1] = _uv4_16.y;
                    rope[0 + 2] = _uv4_16.z;
                    rope[0 + 3] = _uv4_16.w;
                }
                {
                    uint4 _uv4_17 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 64 + 4) + 0);
                    rope[4 + 0] = _uv4_17.x;
                    rope[4 + 1] = _uv4_17.y;
                    rope[4 + 2] = _uv4_17.z;
                    rope[4 + 3] = _uv4_17.w;
                }
                {
                    uint4 _uv4_18 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 64 + 8) + 0);
                    rope[8 + 0] = _uv4_18.x;
                    rope[8 + 1] = _uv4_18.y;
                    rope[8 + 2] = _uv4_18.z;
                    rope[8 + 3] = _uv4_18.w;
                }
                {
                    uint4 _uv4_19 = *reinterpret_cast<const uint4*>(q + (q_row * 80 + 64 + 12) + 0);
                    rope[12 + 0] = _uv4_19.x;
                    rope[12 + 1] = _uv4_19.y;
                    rope[12 + 2] = _uv4_19.z;
                    rope[12 + 3] = _uv4_19.w;
                }
                unsigned int sfa_words[10];
                int qsf_base = q_row / 128 * 1280 + q_row % 32 * 4 + q_row % 128 / 32;
                {
                    sfa_words[0] = *reinterpret_cast<const unsigned int*>(q_sf + qsf_base + 0);
                }
                {
                    sfa_words[1] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 128) + 0);
                }
                {
                    sfa_words[2] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 256) + 0);
                }
                {
                    sfa_words[3] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 384) + 0);
                }
                {
                    sfa_words[4] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 512) + 0);
                }
                {
                    sfa_words[5] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 640) + 0);
                }
                {
                    sfa_words[6] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 768) + 0);
                }
                {
                    sfa_words[7] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 896) + 0);
                }
                {
                    sfa_words[8] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 1024) + 0);
                }
                {
                    sfa_words[9] = *reinterpret_cast<const unsigned int*>(q_sf + (qsf_base + 1152) + 0);
                }
                a_words[64] = rope[0];
                a_words[65] = rope[1];
                a_words[72] = rope[2];
                a_words[73] = rope[3];
                a_words[66] = rope[4];
                a_words[67] = rope[5];
                a_words[74] = rope[6];
                a_words[75] = rope[7];
                a_words[68] = rope[8];
                a_words[69] = rope[9];
                a_words[76] = rope[10];
                a_words[77] = rope[11];
                a_words[70] = rope[12];
                a_words[71] = rope[13];
                a_words[78] = rope[14];
                a_words[79] = rope[15];
                unsigned int rope_lo = sfa_words[8];
                unsigned int rope_hi = sfa_words[9];
                uint32_t _prmt_20;
                asm("prmt.b32 %0, %1, %2, 0x6420;" : "=r"(_prmt_20) : "r"(rope_lo), "r"(rope_hi));
                sfa_words[8] = _prmt_20;
                uint32_t _prmt_21;
                asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(_prmt_21) : "r"(rope_lo), "r"(rope_hi));
                sfa_words[9] = _prmt_21;
                if (row_ok == 0) {
                    a_words[0] = 0;
                    a_words[1] = 0;
                    a_words[2] = 0;
                    a_words[3] = 0;
                    a_words[4] = 0;
                    a_words[5] = 0;
                    a_words[6] = 0;
                    a_words[7] = 0;
                    a_words[8] = 0;
                    a_words[9] = 0;
                    a_words[10] = 0;
                    a_words[11] = 0;
                    a_words[12] = 0;
                    a_words[13] = 0;
                    a_words[14] = 0;
                    a_words[15] = 0;
                    a_words[16] = 0;
                    a_words[17] = 0;
                    a_words[18] = 0;
                    a_words[19] = 0;
                    a_words[20] = 0;
                    a_words[21] = 0;
                    a_words[22] = 0;
                    a_words[23] = 0;
                    a_words[24] = 0;
                    a_words[25] = 0;
                    a_words[26] = 0;
                    a_words[27] = 0;
                    a_words[28] = 0;
                    a_words[29] = 0;
                    a_words[30] = 0;
                    a_words[31] = 0;
                    a_words[32] = 0;
                    a_words[33] = 0;
                    a_words[34] = 0;
                    a_words[35] = 0;
                    a_words[36] = 0;
                    a_words[37] = 0;
                    a_words[38] = 0;
                    a_words[39] = 0;
                    a_words[40] = 0;
                    a_words[41] = 0;
                    a_words[42] = 0;
                    a_words[43] = 0;
                    a_words[44] = 0;
                    a_words[45] = 0;
                    a_words[46] = 0;
                    a_words[47] = 0;
                    a_words[48] = 0;
                    a_words[49] = 0;
                    a_words[50] = 0;
                    a_words[51] = 0;
                    a_words[52] = 0;
                    a_words[53] = 0;
                    a_words[54] = 0;
                    a_words[55] = 0;
                    a_words[56] = 0;
                    a_words[57] = 0;
                    a_words[58] = 0;
                    a_words[59] = 0;
                    a_words[60] = 0;
                    a_words[61] = 0;
                    a_words[62] = 0;
                    a_words[63] = 0;
                    a_words[64] = 0;
                    a_words[65] = 0;
                    a_words[66] = 0;
                    a_words[67] = 0;
                    a_words[68] = 0;
                    a_words[69] = 0;
                    a_words[70] = 0;
                    a_words[71] = 0;
                    a_words[72] = 0;
                    a_words[73] = 0;
                    a_words[74] = 0;
                    a_words[75] = 0;
                    a_words[76] = 0;
                    a_words[77] = 0;
                    a_words[78] = 0;
                    a_words[79] = 0;
                    sfa_words[0] = 0;
                    sfa_words[1] = 0;
                    sfa_words[2] = 0;
                    sfa_words[3] = 0;
                    sfa_words[4] = 0;
                    sfa_words[5] = 0;
                    sfa_words[6] = 0;
                    sfa_words[7] = 0;
                    sfa_words[8] = 0;
                    sfa_words[9] = 0;
                }
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320), "r"(a_words[0]), "r"(a_words[1]), "r"(a_words[2]), "r"(a_words[3]), "r"(a_words[4]), "r"(a_words[5]), "r"(a_words[6]), "r"(a_words[7]), "r"(a_words[8]), "r"(a_words[9]), "r"(a_words[10]), "r"(a_words[11]), "r"(a_words[12]), "r"(a_words[13]), "r"(a_words[14]), "r"(a_words[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 336), "r"((a_words + 16)[0]), "r"((a_words + 16)[1]), "r"((a_words + 16)[2]), "r"((a_words + 16)[3]), "r"((a_words + 16)[4]), "r"((a_words + 16)[5]), "r"((a_words + 16)[6]), "r"((a_words + 16)[7]), "r"((a_words + 16)[8]), "r"((a_words + 16)[9]), "r"((a_words + 16)[10]), "r"((a_words + 16)[11]), "r"((a_words + 16)[12]), "r"((a_words + 16)[13]), "r"((a_words + 16)[14]), "r"((a_words + 16)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 352), "r"((a_words + 32)[0]), "r"((a_words + 32)[1]), "r"((a_words + 32)[2]), "r"((a_words + 32)[3]), "r"((a_words + 32)[4]), "r"((a_words + 32)[5]), "r"((a_words + 32)[6]), "r"((a_words + 32)[7]), "r"((a_words + 32)[8]), "r"((a_words + 32)[9]), "r"((a_words + 32)[10]), "r"((a_words + 32)[11]), "r"((a_words + 32)[12]), "r"((a_words + 32)[13]), "r"((a_words + 32)[14]), "r"((a_words + 32)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 368), "r"((a_words + 48)[0]), "r"((a_words + 48)[1]), "r"((a_words + 48)[2]), "r"((a_words + 48)[3]), "r"((a_words + 48)[4]), "r"((a_words + 48)[5]), "r"((a_words + 48)[6]), "r"((a_words + 48)[7]), "r"((a_words + 48)[8]), "r"((a_words + 48)[9]), "r"((a_words + 48)[10]), "r"((a_words + 48)[11]), "r"((a_words + 48)[12]), "r"((a_words + 48)[13]), "r"((a_words + 48)[14]), "r"((a_words + 48)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 384), "r"((a_words + 64)[0]), "r"((a_words + 64)[1]), "r"((a_words + 64)[2]), "r"((a_words + 64)[3]), "r"((a_words + 64)[4]), "r"((a_words + 64)[5]), "r"((a_words + 64)[6]), "r"((a_words + 64)[7]), "r"((a_words + 64)[8]), "r"((a_words + 64)[9]), "r"((a_words + 64)[10]), "r"((a_words + 64)[11]), "r"((a_words + 64)[12]), "r"((a_words + 64)[13]), "r"((a_words + 64)[14]), "r"((a_words + 64)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 400), "r"(sfa_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 401), "r"(sfa_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 402), "r"(sfa_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 403), "r"(sfa_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 404), "r"((sfa_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 405), "r"((sfa_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 406), "r"((sfa_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 407), "r"((sfa_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 408), "r"((sfa_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 409), "r"((sfa_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 410), "r"((sfa_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 411), "r"((sfa_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 412), "r"((sfa_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 413), "r"((sfa_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 414), "r"((sfa_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 415), "r"((sfa_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416), "r"((sfa_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 417), "r"((sfa_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 418), "r"((sfa_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 419), "r"((sfa_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 420), "r"((sfa_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 421), "r"((sfa_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 422), "r"((sfa_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 423), "r"((sfa_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 424), "r"((sfa_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 425), "r"((sfa_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 426), "r"((sfa_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 427), "r"((sfa_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 428), "r"((sfa_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 429), "r"((sfa_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 430), "r"((sfa_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 431), "r"((sfa_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 432), "r"((sfa_words + 8)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 433), "r"((sfa_words + 8)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 434), "r"((sfa_words + 8)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 435), "r"((sfa_words + 8)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 436), "r"((sfa_words + 9)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 437), "r"((sfa_words + 9)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 438), "r"((sfa_words + 9)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 439), "r"((sfa_words + 9)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(q_full_addr), "r"(0) : "memory");
                #pragma unroll 1
                for (int j_1 = 0; j_1 < my_n_1; j_1++) {
                    mbarrier_wait(raw_full_addr + (raw_x) * 8, _phase_raw_full);
                    mbarrier_wait(vt_empty_addr + (vt_x) * 8, _phase_vt_empty);
                    unsigned int lw[32];
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[0]), "=r"(lw[1]), "=r"(lw[2]), "=r"(lw[3])
                        : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[4]), "=r"(lw[5]), "=r"(lw[6]), "=r"(lw[7])
                        : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + 1024 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[8]), "=r"(lw[9]), "=r"(lw[10]), "=r"(lw[11])
                        : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + 2048 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[12]), "=r"(lw[13]), "=r"(lw[14]), "=r"(lw[15])
                        : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + 3072 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[16]), "=r"(lw[17]), "=r"(lw[18]), "=r"(lw[19])
                        : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[20]), "=r"(lw[21]), "=r"(lw[22]), "=r"(lw[23])
                        : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + 1024 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[24]), "=r"(lw[25]), "=r"(lw[26]), "=r"(lw[27])
                        : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + 2048 + raw_lane))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(lw[28]), "=r"(lw[29]), "=r"(lw[30]), "=r"(lw[31])
                        : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + 3072 + raw_lane))
                        : "memory");
                    unsigned int pw_0[4];
                    unsigned int u_0 = lw[0];
                    unsigned int v_0 = lw[2];
                    unsigned int even_0 = u_0 & 252645135 | v_0 << 4 & 4042322160u;
                    unsigned int odd_0 = u_0 >> 4 & 252645135 | v_0 & 4042322160u;
                    pw_0[0] = ((lane >= 16) ? odd_0 : even_0);
                    pw_0[1] = ((lane >= 16) ? even_0 : odd_0);
                    unsigned int u_1 = lw[1];
                    unsigned int v_1 = lw[3];
                    unsigned int even_1 = u_1 & 252645135 | v_1 << 4 & 4042322160u;
                    unsigned int odd_1 = u_1 >> 4 & 252645135 | v_1 & 4042322160u;
                    pw_0[2] = ((lane >= 16) ? odd_1 : even_1);
                    pw_0[3] = ((lane >= 16) ? even_1 : odd_1);
                    uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (0 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[3]))
                        : "memory");
                    unsigned int pw_1[4];
                    unsigned int u_2 = lw[4];
                    unsigned int v_2 = lw[6];
                    unsigned int even_2 = u_2 & 252645135 | v_2 << 4 & 4042322160u;
                    unsigned int odd_2 = u_2 >> 4 & 252645135 | v_2 & 4042322160u;
                    pw_1[0] = ((lane >= 16) ? odd_2 : even_2);
                    pw_1[1] = ((lane >= 16) ? even_2 : odd_2);
                    unsigned int u_3 = lw[5];
                    unsigned int v_3 = lw[7];
                    unsigned int even_3 = u_3 & 252645135 | v_3 << 4 & 4042322160u;
                    unsigned int odd_3 = u_3 >> 4 & 252645135 | v_3 & 4042322160u;
                    pw_1[2] = ((lane >= 16) ? odd_3 : even_3);
                    pw_1[3] = ((lane >= 16) ? even_3 : odd_3);
                    uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (1 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[3]))
                        : "memory");
                    unsigned int pw_2[4];
                    unsigned int u_4 = lw[8];
                    unsigned int v_4 = lw[10];
                    unsigned int even_4 = u_4 & 252645135 | v_4 << 4 & 4042322160u;
                    unsigned int odd_4 = u_4 >> 4 & 252645135 | v_4 & 4042322160u;
                    pw_2[0] = ((lane >= 16) ? odd_4 : even_4);
                    pw_2[1] = ((lane >= 16) ? even_4 : odd_4);
                    unsigned int u_5 = lw[9];
                    unsigned int v_5 = lw[11];
                    unsigned int even_5 = u_5 & 252645135 | v_5 << 4 & 4042322160u;
                    unsigned int odd_5 = u_5 >> 4 & 252645135 | v_5 & 4042322160u;
                    pw_2[2] = ((lane >= 16) ? odd_5 : even_5);
                    pw_2[3] = ((lane >= 16) ? even_5 : odd_5);
                    uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (2 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[3]))
                        : "memory");
                    unsigned int pw_3[4];
                    unsigned int u_6 = lw[12];
                    unsigned int v_6 = lw[14];
                    unsigned int even_6 = u_6 & 252645135 | v_6 << 4 & 4042322160u;
                    unsigned int odd_6 = u_6 >> 4 & 252645135 | v_6 & 4042322160u;
                    pw_3[0] = ((lane >= 16) ? odd_6 : even_6);
                    pw_3[1] = ((lane >= 16) ? even_6 : odd_6);
                    unsigned int u_7 = lw[13];
                    unsigned int v_7 = lw[15];
                    unsigned int even_7 = u_7 & 252645135 | v_7 << 4 & 4042322160u;
                    unsigned int odd_7 = u_7 >> 4 & 252645135 | v_7 & 4042322160u;
                    pw_3[2] = ((lane >= 16) ? odd_7 : even_7);
                    pw_3[3] = ((lane >= 16) ? even_7 : odd_7);
                    uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (3 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[3]))
                        : "memory");
                    unsigned int pw_4[4];
                    unsigned int u_8 = lw[16];
                    unsigned int v_8 = lw[18];
                    unsigned int even_8 = u_8 & 252645135 | v_8 << 4 & 4042322160u;
                    unsigned int odd_8 = u_8 >> 4 & 252645135 | v_8 & 4042322160u;
                    pw_4[0] = ((lane >= 16) ? odd_8 : even_8);
                    pw_4[1] = ((lane >= 16) ? even_8 : odd_8);
                    unsigned int u_9 = lw[17];
                    unsigned int v_9 = lw[19];
                    unsigned int even_9 = u_9 & 252645135 | v_9 << 4 & 4042322160u;
                    unsigned int odd_9 = u_9 >> 4 & 252645135 | v_9 & 4042322160u;
                    pw_4[2] = ((lane >= 16) ? odd_9 : even_9);
                    pw_4[3] = ((lane >= 16) ? even_9 : odd_9);
                    uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (0 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[3]))
                        : "memory");
                    unsigned int pw_5[4];
                    unsigned int u_10 = lw[20];
                    unsigned int v_10 = lw[22];
                    unsigned int even_10 = u_10 & 252645135 | v_10 << 4 & 4042322160u;
                    unsigned int odd_10 = u_10 >> 4 & 252645135 | v_10 & 4042322160u;
                    pw_5[0] = ((lane >= 16) ? odd_10 : even_10);
                    pw_5[1] = ((lane >= 16) ? even_10 : odd_10);
                    unsigned int u_11 = lw[21];
                    unsigned int v_11 = lw[23];
                    unsigned int even_11 = u_11 & 252645135 | v_11 << 4 & 4042322160u;
                    unsigned int odd_11 = u_11 >> 4 & 252645135 | v_11 & 4042322160u;
                    pw_5[2] = ((lane >= 16) ? odd_11 : even_11);
                    pw_5[3] = ((lane >= 16) ? even_11 : odd_11);
                    uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (1 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[3]))
                        : "memory");
                    unsigned int pw_6[4];
                    unsigned int u_12 = lw[24];
                    unsigned int v_12 = lw[26];
                    unsigned int even_12 = u_12 & 252645135 | v_12 << 4 & 4042322160u;
                    unsigned int odd_12 = u_12 >> 4 & 252645135 | v_12 & 4042322160u;
                    pw_6[0] = ((lane >= 16) ? odd_12 : even_12);
                    pw_6[1] = ((lane >= 16) ? even_12 : odd_12);
                    unsigned int u_13 = lw[25];
                    unsigned int v_13 = lw[27];
                    unsigned int even_13 = u_13 & 252645135 | v_13 << 4 & 4042322160u;
                    unsigned int odd_13 = u_13 >> 4 & 252645135 | v_13 & 4042322160u;
                    pw_6[2] = ((lane >= 16) ? odd_13 : even_13);
                    pw_6[3] = ((lane >= 16) ? even_13 : odd_13);
                    uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (2 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[3]))
                        : "memory");
                    unsigned int pw_7[4];
                    unsigned int u_14 = lw[28];
                    unsigned int v_14 = lw[30];
                    unsigned int even_14 = u_14 & 252645135 | v_14 << 4 & 4042322160u;
                    unsigned int odd_14 = u_14 >> 4 & 252645135 | v_14 & 4042322160u;
                    pw_7[0] = ((lane >= 16) ? odd_14 : even_14);
                    pw_7[1] = ((lane >= 16) ? even_14 : odd_14);
                    unsigned int u_15 = lw[29];
                    unsigned int v_15 = lw[31];
                    unsigned int even_15 = u_15 & 252645135 | v_15 << 4 & 4042322160u;
                    unsigned int odd_15 = u_15 >> 4 & 252645135 | v_15 & 4042322160u;
                    pw_7[2] = ((lane >= 16) ? odd_15 : even_15);
                    pw_7[3] = ((lane >= 16) ? even_15 : odd_15);
                    uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (3 ^ dim_row >> 1 & 3) * 16));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(raw_empty_addr + (raw_x) * 8);
                    raw_x += 1;
                    if (raw_x == 2) { raw_x = 0; _phase_raw_full ^= 1; }
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(vt_full_addr + (vt_x) * 8), "r"(0) : "memory");
                    vt_x += 1;
                    if (vt_x == 2) { vt_x = 0; _phase_vt_empty ^= 1; }
                }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            mbarrier_wait(vt_empty_addr + (vt_x) * 8, _phase_vt_empty);
            vt_x += 1;
            if (vt_x == 2) { vt_x = 0; _phase_vt_empty ^= 1; }
            mbarrier_wait(vt_empty_addr + (vt_x) * 8, _phase_vt_empty);
            vt_x += 1;
            if (vt_x == 2) { vt_x = 0; _phase_vt_empty ^= 1; }
        }
    }
    // ---- Role: load ----
    if (warp == 12) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            int k_l = 0;
            int raw_l = 0;
            int vs_l = 0;
            unsigned int kb_next[20];
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_raw_empty = 1;
            unsigned int _phase_vs_empty = 1;
            #pragma unroll 1
            for (int k_it_2 = 0; k_it_2 < n_items; k_it_2++) {
                int it_2 = clu + k_it_2 * n_clu;
                int split_i_2 = it_2 % num_split;
                int rest_2 = it_2 / num_split;
                int b_i_2 = rest_2 / m_tiles;
                int rows_b_2 = q_len * num_heads;
                int row0_2 = rest_2 % m_tiles * 128;
                int rows_left_2 = rows_b_2 - row0_2;
                int rows_valid_2 = ((rows_left_2 > 128) ? 128 : rows_left_2);
                int row_base_2 = b_i_2 * rows_b_2 + row0_2;
                int kv_len_i_2 = kv_lens[b_i_2];
                int t_last_2 = (row0_2 + rows_valid_2 - 1) / num_heads;
                int kv_end_raw_2 = kv_len_i_2 - (q_len - 1 - t_last_2);
                int kv_end_2 = ((kv_end_raw_2 < 0) ? 0 : kv_end_raw_2);
                int n_pages_2 = (kv_end_2 + 128 - 1) / 128;
                int pps_2 = (n_pages_2 + num_split - 1) / num_split;
                int p_start_2 = split_i_2 * pps_2;
                int p_end_raw_2 = p_start_2 + pps_2;
                int p_end_2 = ((p_end_raw_2 > n_pages_2) ? n_pages_2 : p_end_raw_2);
                int n_raw_2 = p_end_2 - p_start_2;
                int empty_2 = ((n_raw_2 < 1) ? 1 : 0);
                int my_n_2 = ((n_raw_2 < 1) ? 1 : n_raw_2);
                int my_start_2 = ((n_raw_2 < 1) ? 0 : p_start_2);
                int keys_2 = ((empty_2 != 0) ? 0 : kv_end_2);
                int n_last_l = my_n_2 - 1;
                int pid = page_table[b_i_2 * page_stride + my_start_2];
                int phys = ((pid < 0) ? 0 : pid) + page_offset;
                {
                    const uint2* _ivptr_0 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 4 * lane + 2) + 0);
                    uint2 _ivld_0;
                    _ivld_0 = *_ivptr_0;
                    kb_next[0 + 0] = _ivld_0.x;
                    kb_next[0 + 1] = _ivld_0.y;
                }
                {
                    const uint2* _ivptr_1 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 128 + 4 * lane + 2) + 0);
                    uint2 _ivld_1;
                    _ivld_1 = *_ivptr_1;
                    kb_next[2 + 0] = _ivld_1.x;
                    kb_next[2 + 1] = _ivld_1.y;
                }
                {
                    const uint2* _ivptr_2 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 256 + 4 * lane + 2) + 0);
                    uint2 _ivld_2;
                    _ivld_2 = *_ivptr_2;
                    kb_next[4 + 0] = _ivld_2.x;
                    kb_next[4 + 1] = _ivld_2.y;
                }
                {
                    const uint2* _ivptr_3 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 384 + 4 * lane + 2) + 0);
                    uint2 _ivld_3;
                    _ivld_3 = *_ivptr_3;
                    kb_next[6 + 0] = _ivld_3.x;
                    kb_next[6 + 1] = _ivld_3.y;
                }
                {
                    const uint2* _ivptr_4 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 512 + 4 * lane + 2) + 0);
                    uint2 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    kb_next[8 + 0] = _ivld_4.x;
                    kb_next[8 + 1] = _ivld_4.y;
                }
                {
                    const uint2* _ivptr_5 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 640 + 4 * lane + 2) + 0);
                    uint2 _ivld_5;
                    _ivld_5 = *_ivptr_5;
                    kb_next[10 + 0] = _ivld_5.x;
                    kb_next[10 + 1] = _ivld_5.y;
                }
                {
                    const uint2* _ivptr_6 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 768 + 4 * lane + 2) + 0);
                    uint2 _ivld_6;
                    _ivld_6 = *_ivptr_6;
                    kb_next[12 + 0] = _ivld_6.x;
                    kb_next[12 + 1] = _ivld_6.y;
                }
                {
                    const uint2* _ivptr_7 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 896 + 4 * lane + 2) + 0);
                    uint2 _ivld_7;
                    _ivld_7 = *_ivptr_7;
                    kb_next[14 + 0] = _ivld_7.x;
                    kb_next[14 + 1] = _ivld_7.y;
                }
                {
                    const uint2* _ivptr_8 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 1024 + 4 * lane + 2) + 0);
                    uint2 _ivld_8;
                    _ivld_8 = *_ivptr_8;
                    kb_next[16 + 0] = _ivld_8.x;
                    kb_next[16 + 1] = _ivld_8.y;
                }
                {
                    const uint2* _ivptr_9 = reinterpret_cast<const uint2*>(kv_sf + (phys * 1280 + 1152 + 4 * lane + 2) + 0);
                    uint2 _ivld_9;
                    _ivld_9 = *_ivptr_9;
                    kb_next[18 + 0] = _ivld_9.x;
                    kb_next[18 + 1] = _ivld_9.y;
                }
                #pragma unroll 1
                for (int j_2 = 0; j_2 < my_n_2; j_2++) {
                    int pid_0 = page_table[b_i_2 * page_stride + (my_start_2 + j_2)];
                    int phys_0 = ((pid_0 < 0) ? 0 : pid_0) + page_offset;
                    int page_l = phys_0;
                    unsigned int kb[20];
                    kb[0] = kb_next[0];
                    kb[1] = kb_next[1];
                    kb[2] = kb_next[2];
                    kb[3] = kb_next[3];
                    kb[4] = kb_next[4];
                    kb[5] = kb_next[5];
                    kb[6] = kb_next[6];
                    kb[7] = kb_next[7];
                    kb[8] = kb_next[8];
                    kb[9] = kb_next[9];
                    kb[10] = kb_next[10];
                    kb[11] = kb_next[11];
                    kb[12] = kb_next[12];
                    kb[13] = kb_next[13];
                    kb[14] = kb_next[14];
                    kb[15] = kb_next[15];
                    kb[16] = kb_next[16];
                    kb[17] = kb_next[17];
                    kb[18] = kb_next[18];
                    kb[19] = kb_next[19];
                    int j1 = j_2 + 1;
                    int pid_1 = page_table[b_i_2 * page_stride + (my_start_2 + ((j1 < n_last_l) ? j1 : n_last_l))];
                    int phys_1 = ((pid_1 < 0) ? 0 : pid_1) + page_offset;
                    {
                        const uint2* _ivptr_10 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 4 * lane + 2) + 0);
                        uint2 _ivld_10;
                        _ivld_10 = *_ivptr_10;
                        kb_next[0 + 0] = _ivld_10.x;
                        kb_next[0 + 1] = _ivld_10.y;
                    }
                    {
                        const uint2* _ivptr_11 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 128 + 4 * lane + 2) + 0);
                        uint2 _ivld_11;
                        _ivld_11 = *_ivptr_11;
                        kb_next[2 + 0] = _ivld_11.x;
                        kb_next[2 + 1] = _ivld_11.y;
                    }
                    {
                        const uint2* _ivptr_12 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 256 + 4 * lane + 2) + 0);
                        uint2 _ivld_12;
                        _ivld_12 = *_ivptr_12;
                        kb_next[4 + 0] = _ivld_12.x;
                        kb_next[4 + 1] = _ivld_12.y;
                    }
                    {
                        const uint2* _ivptr_13 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 384 + 4 * lane + 2) + 0);
                        uint2 _ivld_13;
                        _ivld_13 = *_ivptr_13;
                        kb_next[6 + 0] = _ivld_13.x;
                        kb_next[6 + 1] = _ivld_13.y;
                    }
                    {
                        const uint2* _ivptr_14 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 512 + 4 * lane + 2) + 0);
                        uint2 _ivld_14;
                        _ivld_14 = *_ivptr_14;
                        kb_next[8 + 0] = _ivld_14.x;
                        kb_next[8 + 1] = _ivld_14.y;
                    }
                    {
                        const uint2* _ivptr_15 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 640 + 4 * lane + 2) + 0);
                        uint2 _ivld_15;
                        _ivld_15 = *_ivptr_15;
                        kb_next[10 + 0] = _ivld_15.x;
                        kb_next[10 + 1] = _ivld_15.y;
                    }
                    {
                        const uint2* _ivptr_16 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 768 + 4 * lane + 2) + 0);
                        uint2 _ivld_16;
                        _ivld_16 = *_ivptr_16;
                        kb_next[12 + 0] = _ivld_16.x;
                        kb_next[12 + 1] = _ivld_16.y;
                    }
                    {
                        const uint2* _ivptr_17 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 896 + 4 * lane + 2) + 0);
                        uint2 _ivld_17;
                        _ivld_17 = *_ivptr_17;
                        kb_next[14 + 0] = _ivld_17.x;
                        kb_next[14 + 1] = _ivld_17.y;
                    }
                    {
                        const uint2* _ivptr_18 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 1024 + 4 * lane + 2) + 0);
                        uint2 _ivld_18;
                        _ivld_18 = *_ivptr_18;
                        kb_next[16 + 0] = _ivld_18.x;
                        kb_next[16 + 1] = _ivld_18.y;
                    }
                    {
                        const uint2* _ivptr_19 = reinterpret_cast<const uint2*>(kv_sf + (phys_1 * 1280 + 1152 + 4 * lane + 2) + 0);
                        uint2 _ivld_19;
                        _ivld_19 = *_ivptr_19;
                        kb_next[18 + 0] = _ivld_19.x;
                        kb_next[18 + 1] = _ivld_19.y;
                    }
                    mbarrier_wait(k_empty_addr + (k_l) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)(k_l * 5 * 4096), (&kv_k), 0, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 1) * 4096), (&kv_k), 64, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 2) * 4096), (&kv_k), 128, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 3) * 4096), (&kv_k), 192, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 4) * 4096), (&kv_k), 256, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                        tma_5d_gmem2smem_cta2(smem_ks_addr + (unsigned int)(k_l * 10240), (&ksf_map), 0, 0, 0, 0, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 16 * (lane % 8))), "r"(kb[0]), "r"(kb[1]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 128 + 16 * (lane % 8))), "r"(kb[2]), "r"(kb[3]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 256 + 16 * (lane % 8))), "r"(kb[4]), "r"(kb[5]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 384 + 16 * (lane % 8))), "r"(kb[6]), "r"(kb[7]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 512 + 16 * (lane % 8))), "r"(kb[8]), "r"(kb[9]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 640 + 16 * (lane % 8))), "r"(kb[10]), "r"(kb[11]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 768 + 16 * (lane % 8))), "r"(kb[12]), "r"(kb[13]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 896 + 16 * (lane % 8))), "r"(kb[14]), "r"(kb[15]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 1024 + 16 * (lane % 8))), "r"(kb[16]), "r"(kb[17]), "r"(0), "r"(0) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ks_addr + (unsigned int)(k_l * 10240 + 5120 + 1280 * (lane / 8) + 1152 + 16 * (lane % 8))), "r"(kb[18]), "r"(kb[19]), "r"(0), "r"(0) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (k_l) * 8) & 0xFEFFFFFF), "r"((uint32_t)(25600)) : "memory");
                    }
                    k_l += 1;
                    if (k_l == 2) { k_l = 0; _phase_k_empty ^= 1; }
                    mbarrier_wait(raw_empty_addr + (raw_l) * 8, _phase_raw_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(raw_full_addr + (raw_l) * 8, 16384);
                        tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(raw_l * 2 * 8192), (&kv_v), 64 * cta_rank, 0, 0, page_l, raw_full_addr + (raw_l) * 8);
                        tma_4d_gmem2smem(smem_raw_addr + (unsigned int)((raw_l * 2 + 1) * 8192), (&kv_v), 64 * (2 + cta_rank), 0, 0, page_l, raw_full_addr + (raw_l) * 8);
                    }
                    raw_l += 1;
                    if (raw_l == 2) { raw_l = 0; _phase_raw_empty ^= 1; }
                    mbarrier_wait(vs_empty_addr + (vs_l) * 8, _phase_vs_empty);
                    int page_keys_l = keys_2 - 128 * (my_start_2 + j_2);
                    if (page_keys_l >= 128) {
                        if (elect_sync()) {
                            tma_5d_gmem2smem_cta2(smem_vs_addr + (unsigned int)(vs_l * 4096), (&vsf_map), 0, 0, 0, 0, page_l, ((vs_full_addr + (vs_l) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((vs_full_addr + (vs_l) * 8) & 0xFEFFFFFF), "r"((uint32_t)(4096)) : "memory");
                        }
                    } else {
                        unsigned int vw_0[4];
                        {
                            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 4 * lane) + 0);
                            vw_0[0 + 0] = _uv4_20.x;
                            vw_0[0 + 1] = _uv4_20.y;
                            vw_0[0 + 2] = _uv4_20.z;
                            vw_0[0 + 3] = _uv4_20.w;
                        }
                        unsigned int live_mask_0 = 0;
                        if (0 < page_keys_l) {
                            live_mask_0 = live_mask_0 | 255;
                        }
                        if (16 < page_keys_l) {
                            live_mask_0 = live_mask_0 | 65280;
                        }
                        if (32 < page_keys_l) {
                            live_mask_0 = live_mask_0 | 16711680;
                        }
                        if (48 < page_keys_l) {
                            live_mask_0 = live_mask_0 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 16 * lane)), "r"(vw_0[0] & live_mask_0), "r"(vw_0[1] & live_mask_0), "r"(vw_0[2] & live_mask_0), "r"(vw_0[3] & live_mask_0) : "memory");
                        unsigned int vw_1[4];
                        {
                            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 256 + 4 * lane) + 0);
                            vw_1[0 + 0] = _uv4_21.x;
                            vw_1[0 + 1] = _uv4_21.y;
                            vw_1[0 + 2] = _uv4_21.z;
                            vw_1[0 + 3] = _uv4_21.w;
                        }
                        unsigned int live_mask_1 = 0;
                        if (0 < page_keys_l) {
                            live_mask_1 = live_mask_1 | 255;
                        }
                        if (16 < page_keys_l) {
                            live_mask_1 = live_mask_1 | 65280;
                        }
                        if (32 < page_keys_l) {
                            live_mask_1 = live_mask_1 | 16711680;
                        }
                        if (48 < page_keys_l) {
                            live_mask_1 = live_mask_1 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 512 + 16 * lane)), "r"(vw_1[0] & live_mask_1), "r"(vw_1[1] & live_mask_1), "r"(vw_1[2] & live_mask_1), "r"(vw_1[3] & live_mask_1) : "memory");
                        unsigned int vw_2[4];
                        {
                            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 128 + 4 * lane) + 0);
                            vw_2[0 + 0] = _uv4_22.x;
                            vw_2[0 + 1] = _uv4_22.y;
                            vw_2[0 + 2] = _uv4_22.z;
                            vw_2[0 + 3] = _uv4_22.w;
                        }
                        unsigned int live_mask_2 = 0;
                        if (64 < page_keys_l) {
                            live_mask_2 = live_mask_2 | 255;
                        }
                        if (80 < page_keys_l) {
                            live_mask_2 = live_mask_2 | 65280;
                        }
                        if (96 < page_keys_l) {
                            live_mask_2 = live_mask_2 | 16711680;
                        }
                        if (112 < page_keys_l) {
                            live_mask_2 = live_mask_2 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 1024 + 16 * lane)), "r"(vw_2[0] & live_mask_2), "r"(vw_2[1] & live_mask_2), "r"(vw_2[2] & live_mask_2), "r"(vw_2[3] & live_mask_2) : "memory");
                        unsigned int vw_3[4];
                        {
                            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 256 + 128 + 4 * lane) + 0);
                            vw_3[0 + 0] = _uv4_23.x;
                            vw_3[0 + 1] = _uv4_23.y;
                            vw_3[0 + 2] = _uv4_23.z;
                            vw_3[0 + 3] = _uv4_23.w;
                        }
                        unsigned int live_mask_3 = 0;
                        if (64 < page_keys_l) {
                            live_mask_3 = live_mask_3 | 255;
                        }
                        if (80 < page_keys_l) {
                            live_mask_3 = live_mask_3 | 65280;
                        }
                        if (96 < page_keys_l) {
                            live_mask_3 = live_mask_3 | 16711680;
                        }
                        if (112 < page_keys_l) {
                            live_mask_3 = live_mask_3 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 1536 + 16 * lane)), "r"(vw_3[0] & live_mask_3), "r"(vw_3[1] & live_mask_3), "r"(vw_3[2] & live_mask_3), "r"(vw_3[3] & live_mask_3) : "memory");
                        unsigned int vw_4[4];
                        {
                            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 512 + 4 * lane) + 0);
                            vw_4[0 + 0] = _uv4_24.x;
                            vw_4[0 + 1] = _uv4_24.y;
                            vw_4[0 + 2] = _uv4_24.z;
                            vw_4[0 + 3] = _uv4_24.w;
                        }
                        unsigned int live_mask_4 = 0;
                        if (0 < page_keys_l) {
                            live_mask_4 = live_mask_4 | 255;
                        }
                        if (16 < page_keys_l) {
                            live_mask_4 = live_mask_4 | 65280;
                        }
                        if (32 < page_keys_l) {
                            live_mask_4 = live_mask_4 | 16711680;
                        }
                        if (48 < page_keys_l) {
                            live_mask_4 = live_mask_4 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 2048 + 16 * lane)), "r"(vw_4[0] & live_mask_4), "r"(vw_4[1] & live_mask_4), "r"(vw_4[2] & live_mask_4), "r"(vw_4[3] & live_mask_4) : "memory");
                        unsigned int vw_5[4];
                        {
                            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 768 + 4 * lane) + 0);
                            vw_5[0 + 0] = _uv4_25.x;
                            vw_5[0 + 1] = _uv4_25.y;
                            vw_5[0 + 2] = _uv4_25.z;
                            vw_5[0 + 3] = _uv4_25.w;
                        }
                        unsigned int live_mask_5 = 0;
                        if (0 < page_keys_l) {
                            live_mask_5 = live_mask_5 | 255;
                        }
                        if (16 < page_keys_l) {
                            live_mask_5 = live_mask_5 | 65280;
                        }
                        if (32 < page_keys_l) {
                            live_mask_5 = live_mask_5 | 16711680;
                        }
                        if (48 < page_keys_l) {
                            live_mask_5 = live_mask_5 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 2560 + 16 * lane)), "r"(vw_5[0] & live_mask_5), "r"(vw_5[1] & live_mask_5), "r"(vw_5[2] & live_mask_5), "r"(vw_5[3] & live_mask_5) : "memory");
                        unsigned int vw_6[4];
                        {
                            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 512 + 128 + 4 * lane) + 0);
                            vw_6[0 + 0] = _uv4_26.x;
                            vw_6[0 + 1] = _uv4_26.y;
                            vw_6[0 + 2] = _uv4_26.z;
                            vw_6[0 + 3] = _uv4_26.w;
                        }
                        unsigned int live_mask_6 = 0;
                        if (64 < page_keys_l) {
                            live_mask_6 = live_mask_6 | 255;
                        }
                        if (80 < page_keys_l) {
                            live_mask_6 = live_mask_6 | 65280;
                        }
                        if (96 < page_keys_l) {
                            live_mask_6 = live_mask_6 | 16711680;
                        }
                        if (112 < page_keys_l) {
                            live_mask_6 = live_mask_6 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 3072 + 16 * lane)), "r"(vw_6[0] & live_mask_6), "r"(vw_6[1] & live_mask_6), "r"(vw_6[2] & live_mask_6), "r"(vw_6[3] & live_mask_6) : "memory");
                        unsigned int vw_7[4];
                        {
                            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(v_pool + (page_l * 1024 + 768 + 128 + 4 * lane) + 0);
                            vw_7[0 + 0] = _uv4_27.x;
                            vw_7[0 + 1] = _uv4_27.y;
                            vw_7[0 + 2] = _uv4_27.z;
                            vw_7[0 + 3] = _uv4_27.w;
                        }
                        unsigned int live_mask_7 = 0;
                        if (64 < page_keys_l) {
                            live_mask_7 = live_mask_7 | 255;
                        }
                        if (80 < page_keys_l) {
                            live_mask_7 = live_mask_7 | 65280;
                        }
                        if (96 < page_keys_l) {
                            live_mask_7 = live_mask_7 | 16711680;
                        }
                        if (112 < page_keys_l) {
                            live_mask_7 = live_mask_7 | 4278190080u;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_vs_addr + (unsigned int)(vs_l * 4096 + 3584 + 16 * lane)), "r"(vw_7[0] & live_mask_7), "r"(vw_7[1] & live_mask_7), "r"(vw_7[2] & live_mask_7), "r"(vw_7[3] & live_mask_7) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(vs_full_addr + (vs_l) * 8), "r"(0) : "memory");
                        }
                    }
                    vs_l += 1;
                    if (vs_l == 4) { vs_l = 0; _phase_vs_empty ^= 1; }
                }
            }
            mbarrier_wait(k_empty_addr + (k_l) * 8, _phase_k_empty);
            k_l += 1;
            if (k_l == 2) { k_l = 0; _phase_k_empty ^= 1; }
            mbarrier_wait(k_empty_addr + (k_l) * 8, _phase_k_empty);
            k_l += 1;
            if (k_l == 2) { k_l = 0; _phase_k_empty ^= 1; }
            mbarrier_wait(vs_empty_addr + (vs_l) * 8, _phase_vs_empty);
            vs_l += 1;
            if (vs_l == 4) { vs_l = 0; _phase_vs_empty ^= 1; }
            mbarrier_wait(vs_empty_addr + (vs_l) * 8, _phase_vs_empty);
            vs_l += 1;
            if (vs_l == 4) { vs_l = 0; _phase_vs_empty ^= 1; }
            mbarrier_wait(vs_empty_addr + (vs_l) * 8, _phase_vs_empty);
            vs_l += 1;
            if (vs_l == 4) { vs_l = 0; _phase_vs_empty ^= 1; }
            mbarrier_wait(vs_empty_addr + (vs_l) * 8, _phase_vs_empty);
            vs_l += 1;
            if (vs_l == 4) { vs_l = 0; _phase_vs_empty ^= 1; }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_s_empty = 0;
            if (cta_rank == 0) {
                int k_m = 0;
                int s_m = 0;
                int se_m = 0;
                #pragma unroll 1
                for (int k_it_3 = 0; k_it_3 < n_items; k_it_3++) {
                    int it_3 = clu + k_it_3 * n_clu;
                    int split_i_3 = it_3 % num_split;
                    int rest_3 = it_3 / num_split;
                    int b_i_3 = rest_3 / m_tiles;
                    int rows_b_3 = q_len * num_heads;
                    int row0_3 = rest_3 % m_tiles * 128;
                    int rows_left_3 = rows_b_3 - row0_3;
                    int rows_valid_3 = ((rows_left_3 > 128) ? 128 : rows_left_3);
                    int row_base_3 = b_i_3 * rows_b_3 + row0_3;
                    int kv_len_i_3 = kv_lens[b_i_3];
                    int t_last_3 = (row0_3 + rows_valid_3 - 1) / num_heads;
                    int kv_end_raw_3 = kv_len_i_3 - (q_len - 1 - t_last_3);
                    int kv_end_3 = ((kv_end_raw_3 < 0) ? 0 : kv_end_raw_3);
                    int n_pages_3 = (kv_end_3 + 128 - 1) / 128;
                    int pps_3 = (n_pages_3 + num_split - 1) / num_split;
                    int p_start_3 = split_i_3 * pps_3;
                    int p_end_raw_3 = p_start_3 + pps_3;
                    int p_end_3 = ((p_end_raw_3 > n_pages_3) ? n_pages_3 : p_end_raw_3);
                    int n_raw_3 = p_end_3 - p_start_3;
                    int empty_3 = ((n_raw_3 < 1) ? 1 : 0);
                    int my_n_3 = ((n_raw_3 < 1) ? 1 : n_raw_3);
                    int my_start_3 = ((n_raw_3 < 1) ? 0 : p_start_3);
                    int keys_3 = ((empty_3 != 0) ? 0 : kv_end_3);
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait_cluster_hint(k_full_addr + (k_m) * 8, _phase_k_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_0 = elect_sync();
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 0)), "l"(_tcgen05_cp_desc_1), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 128))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_2), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 256))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_3), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 384))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_4), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 512))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_5), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 640))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_6), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 768))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_7), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 896))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_8), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 1024))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 32)), "l"(_tcgen05_cp_desc_9), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 1152))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_sfb + 36)), "l"(_tcgen05_cp_desc_10), "r"(_elected_0)
                            : "memory");
                    }
                    #pragma unroll 1
                    for (int j_3 = 0; j_3 < my_n_3; j_3++) {
                        if (elect_sync()) {
                            int _smem_desc_lo_11 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_11);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 0), "r"(tmem_sfb + 0), "r"(0) : "memory");
                            }
                            int _smem_desc_lo_12 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_12);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 4), "r"(tmem_sfb + 4), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_13 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 1) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_13);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 8), "r"(tmem_sfb + 8), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_14 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 1) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_14);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 24), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 12), "r"(tmem_sfb + 12), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_15 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 2) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_15);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 16), "r"(tmem_sfb + 16), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_16 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 2) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_16);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 40), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 20), "r"(tmem_sfb + 20), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_17 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 3) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_17);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 24), "r"(tmem_sfb + 24), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_18 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 3) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_18);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 56), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 28), "r"(tmem_sfb + 28), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_19 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_19);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 64), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 32), "r"(tmem_sfb + 32), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_20 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_20);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 72), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 36), "r"(tmem_sfb + 32), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_21 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_21);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 64), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 32), "r"(tmem_sfb + 36), "r"(1) : "memory");
                            }
                            tcgen05_commit_cg2_multicast(s_full_addr + (s_m) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(k_empty_addr + (k_m) * 8, (uint16_t)(3));
                            if (j_3 == my_n_3 - 1) {
                                tcgen05_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                            }
                        }
                        k_m += 1;
                        if (k_m == 2) { k_m = 0; _phase_k_full ^= 1; }
                        if (j_3 + 1 < my_n_3) {
                            mbarrier_wait(s_full_addr + (s_m) * 8, _phase_s_full_1);
                            mbarrier_wait_cluster_hint(k_full_addr + (k_m) * 8, _phase_k_full, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            uint32_t _elected_22 = elect_sync();
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_23 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 0)), "l"(_tcgen05_cp_desc_23), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_24 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 128))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_24), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_25 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 256))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_25), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_26 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 384))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_26), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_27 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 512))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_27), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_28 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 640))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_28), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_29 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 768))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_29), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_30 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 896))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_30), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_31 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 1024))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 32)), "l"(_tcgen05_cp_desc_31), "r"(_elected_22)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_32 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 1152))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                    "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                    :: "r"((uint32_t)(tmem_sfb + 36)), "l"(_tcgen05_cp_desc_32), "r"(_elected_22)
                                    : "memory");
                            }
                        }
                        s_m += 1;
                        if (s_m == 2) { s_m = 0; _phase_s_full_1 ^= 1; }
                        mbarrier_wait(s_empty_addr + (se_m) * 8, _phase_s_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        se_m += 1;
                        if (se_m == 2) { se_m = 0; _phase_s_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: pvi ----
    if (warp == 14) {
        { // pvi_main
            unsigned int _phase_o_free_0 = 1;
            unsigned int _phase_vt_full = 0;
            unsigned int _phase_vs_full = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_pv_done_3 = 0;
            if (cta_rank == 0) {
                int vt_m = 0;
                int vs_m = 0;
                int p_m = 0;
                int pv_m = 0;
                #pragma unroll 1
                for (int k_it_4 = 0; k_it_4 < n_items; k_it_4++) {
                    int it_4 = clu + k_it_4 * n_clu;
                    int split_i_4 = it_4 % num_split;
                    int rest_4 = it_4 / num_split;
                    int b_i_4 = rest_4 / m_tiles;
                    int rows_b_4 = q_len * num_heads;
                    int row0_4 = rest_4 % m_tiles * 128;
                    int rows_left_4 = rows_b_4 - row0_4;
                    int rows_valid_4 = ((rows_left_4 > 128) ? 128 : rows_left_4);
                    int row_base_4 = b_i_4 * rows_b_4 + row0_4;
                    int kv_len_i_4 = kv_lens[b_i_4];
                    int t_last_4 = (row0_4 + rows_valid_4 - 1) / num_heads;
                    int kv_end_raw_4 = kv_len_i_4 - (q_len - 1 - t_last_4);
                    int kv_end_4 = ((kv_end_raw_4 < 0) ? 0 : kv_end_raw_4);
                    int n_pages_4 = (kv_end_4 + 128 - 1) / 128;
                    int pps_4 = (n_pages_4 + num_split - 1) / num_split;
                    int p_start_4 = split_i_4 * pps_4;
                    int p_end_raw_4 = p_start_4 + pps_4;
                    int p_end_4 = ((p_end_raw_4 > n_pages_4) ? n_pages_4 : p_end_raw_4);
                    int n_raw_4 = p_end_4 - p_start_4;
                    int empty_4 = ((n_raw_4 < 1) ? 1 : 0);
                    int my_n_4 = ((n_raw_4 < 1) ? 1 : n_raw_4);
                    int my_start_4 = ((n_raw_4 < 1) ? 0 : p_start_4);
                    int keys_4 = ((empty_4 != 0) ? 0 : kv_end_4);
                    mbarrier_wait(o_free_addr, _phase_o_free_0);
                    _phase_o_free_0 ^= 1;
                    #pragma unroll 1
                    for (int jv = 0; jv < my_n_4; jv++) {
                        mbarrier_wait(vt_full_addr + (vt_m) * 8, _phase_vt_full);
                        mbarrier_wait_cluster_hint(vs_full_addr + (vs_m) * 8, _phase_vs_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_0 = elect_sync();
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_vs_addr + (unsigned int)(vs_m * 4096))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfbv + 0)), "l"(_tcgen05_cp_desc_1), "r"(_elected_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_vs_addr + (unsigned int)(vs_m * 4096 + 1024))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfbv + 4)), "l"(_tcgen05_cp_desc_2), "r"(_elected_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_vs_addr + (unsigned int)(vs_m * 4096 + 2048))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfbv + 8)), "l"(_tcgen05_cp_desc_3), "r"(_elected_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_vs_addr + (unsigned int)(vs_m * 4096 + 3072))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfbv + 12)), "l"(_tcgen05_cp_desc_4), "r"(_elected_0)
                                : "memory");
                        }
                        mbarrier_wait(p_full_addr + (p_m) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_5 = elect_sync();
                        int _smem_desc_lo_6 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_m) * 256;
                        int _smem_desc_lo_7 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_acc, a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + 8 * p_m + 0, tmem_sfbv + 0 + 0, ((jv == 0) ? 0 : 1), _elected_5);
                        }
                        int _smem_desc_lo_8 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_m) * 256;
                        int _smem_desc_lo_9 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (vt_m * 2) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_9) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_acc, a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + (8 * p_m + 4) + 0, tmem_sfbv + 4 + 0, 1, _elected_5);
                        }
                        int _smem_desc_lo_10 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_m) * 256;
                        int _smem_desc_lo_11_1 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11_1) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader((tmem_acc + (128)), a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + 8 * p_m + 0, tmem_sfbv + 8 + 0, ((jv == 0) ? 0 : 1), _elected_5);
                        }
                        int _smem_desc_lo_12_1 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_m) * 256;
                        int _smem_desc_lo_13_1 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12_1) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13_1) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader((tmem_acc + (128)), a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + (8 * p_m + 4) + 0, tmem_sfbv + 12 + 0, 1, _elected_5);
                        }
                        tcgen05_commit_cg2_multicast_leader(pv_done_addr + (pv_m) * 8, (uint16_t)(3), _elected_5);
                        tcgen05_commit_cg2_multicast_leader(vt_empty_addr + (vt_m) * 8, (uint16_t)(3), _elected_5);
                        tcgen05_commit_cg2_multicast_leader(vs_empty_addr + (vs_m) * 8, (uint16_t)(3), _elected_5);
                        vt_m += 1;
                        if (vt_m == 2) { vt_m = 0; _phase_vt_full ^= 1; }
                        vs_m += 1;
                        if (vs_m == 4) { vs_m = 0; _phase_vs_full ^= 1; }
                        p_m += 1;
                        if (p_m == 2) { p_m = 0; _phase_p_full ^= 1; }
                        mbarrier_wait(pv_done_addr + (pv_m) * 8, _phase_pv_done_3);
                        pv_m += 1;
                        if (pv_m == 2) { pv_m = 0; _phase_pv_done_3 ^= 1; }
                    }
                }
                mbarrier_wait(o_free_addr, _phase_o_free_0);
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
