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

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(448) __cluster_dims__(2,1,1) void
kernel_cake_fp4_mla_decode_6146c060a9da20cc3573(const __grid_constant__ CUtensorMap kv_k, const __grid_constant__ CUtensorMap kv_v, unsigned int* __restrict__ kv_sf, unsigned int* __restrict__ q, unsigned int* __restrict__ q_sf, int* __restrict__ page_table, int* __restrict__ kv_lens, float* __restrict__ q_gs, float* __restrict__ kv_gs, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, float sm_scale, int num_heads, int q_len, int num_split, int page_stride, int page_offset)
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
    #define q_full_addr (mbar_base + 96)
    #define sfk_full_addr (mbar_base + 104)
    #define sfk_empty_addr (mbar_base + 112)
    #define s_full_addr (mbar_base + 120)
    #define s_empty_addr (mbar_base + 128)
    #define sfv_full_addr (mbar_base + 136)
    #define p_full_addr (mbar_base + 144)
    #define pv_done_addr (mbar_base + 152)

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
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 107520);
    const int smem_p_addr = smem + 107520;
    float* xch_g = reinterpret_cast<float*>(smem_raw + 111616);
    const int _xch_g_addr = smem + 111616;
    float* xch_l = reinterpret_cast<float*>(smem_raw + 115712);
    const int _xch_l_addr = smem + 115712;
    unsigned int* xch_w = reinterpret_cast<unsigned int*>(smem_raw + 116224);
    const int _xch_w_addr = smem + 116224;

    // Mbarrier init (20 barriers at smem_raw[0..160))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // k_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'raw_pipe' ---
            // raw_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // raw_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            // --- pipeline 'vt_pipe' ---
            // vt_full: 2 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            mbarrier_init(smem + 72, 256);
            // vt_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // q_full: 1 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            // sfk_full: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            // sfk_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // s_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            // sfv_full: 1 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 484 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
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
    const unsigned int tmem_sfap = taddr + 460;
    const unsigned int tmem_sfbv = taddr + 468;

    // ---- Role: smx ----
    if (warp <= 3) {
        { // smx_main
            int split_i = blockIdx.x / 2;
            int b_i = blockIdx.z;
            int rows_b = q_len * num_heads;
            int row0 = blockIdx.y * 128;
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
            int qs = make_warp_uniform(warp % 4);
            int half = qs / 2;
            int lane_base_s = qs * 32 << 16;
            int r = qs % 2 * 32 + lane;
            int row_local = cta_rank * 64 + r;
            int mine = qs * 32 + lane;
            int other = (qs ^ 2) * 32 + lane;
            int xbar = make_warp_uniform(1 + qs % 2);
            int row_ok = row_local < rows_valid;
            int token = (row0 + row_local) / num_heads;
            int vis_raw = kv_len_i - (q_len - 1 - token);
            int vis = ((row_ok != 0 && keys > 0) ? vis_raw : 0);
            int dither = vis >= 32768;
            float gq = *q_gs;
            float gk = *kv_gs;
            float c_log2 = sm_scale * 1.442695f / (gq * gk);
            float m = -CUDART_INF_F;
            float l_sum = 0.0f;
            float s_vals[64];
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_pv_done_0 = 1;
            #pragma unroll 1
            for (int j = 0; j < my_n; j++) {
                int tok0 = 128 * (my_start + j) + 64 * half;
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(s_vals[0]), "=f"(s_vals[1]), "=f"(s_vals[2]), "=f"(s_vals[3]), "=f"(s_vals[4]), "=f"(s_vals[5]), "=f"(s_vals[6]), "=f"(s_vals[7]), "=f"(s_vals[8]), "=f"(s_vals[9]), "=f"(s_vals[10]), "=f"(s_vals[11]), "=f"(s_vals[12]), "=f"(s_vals[13]), "=f"(s_vals[14]), "=f"(s_vals[15]), "=f"(s_vals[16]), "=f"(s_vals[17]), "=f"(s_vals[18]), "=f"(s_vals[19]), "=f"(s_vals[20]), "=f"(s_vals[21]), "=f"(s_vals[22]), "=f"(s_vals[23]), "=f"(s_vals[24]), "=f"(s_vals[25]), "=f"(s_vals[26]), "=f"(s_vals[27]), "=f"(s_vals[28]), "=f"(s_vals[29]), "=f"(s_vals[30]), "=f"(s_vals[31])
                    : "r"(taddr + (unsigned int)lane_base_s + 256));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(s_vals[32]), "=f"(s_vals[33]), "=f"(s_vals[34]), "=f"(s_vals[35]), "=f"(s_vals[36]), "=f"(s_vals[37]), "=f"(s_vals[38]), "=f"(s_vals[39]), "=f"(s_vals[40]), "=f"(s_vals[41]), "=f"(s_vals[42]), "=f"(s_vals[43]), "=f"(s_vals[44]), "=f"(s_vals[45]), "=f"(s_vals[46]), "=f"(s_vals[47]), "=f"(s_vals[48]), "=f"(s_vals[49]), "=f"(s_vals[50]), "=f"(s_vals[51]), "=f"(s_vals[52]), "=f"(s_vals[53]), "=f"(s_vals[54]), "=f"(s_vals[55]), "=f"(s_vals[56]), "=f"(s_vals[57]), "=f"(s_vals[58]), "=f"(s_vals[59]), "=f"(s_vals[60]), "=f"(s_vals[61]), "=f"(s_vals[62]), "=f"(s_vals[63])
                    : "r"(taddr + (unsigned int)lane_base_s + 288));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(s_empty_addr), "r"(0) : "memory");
                _phase_s_full_0 ^= 1;
                int live = vis - tok0;
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
                }
                float g[8];
                float b_max_0 = s_vals[0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_0 = max_noftz(b_max_0, s_vals[_lr]);
                }
                g[0] = ((b_max_0 > -CUDART_INF_F) ? c_log2 * b_max_0 : -CUDART_INF_F);
                xch_g[(j % 2 * 128 + mine) * 4] = g[0];
                float b_max_1 = s_vals[(16) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_1 = max_noftz(b_max_1, s_vals[(16) + _lr]);
                }
                g[1] = ((b_max_1 > -CUDART_INF_F) ? c_log2 * b_max_1 : -CUDART_INF_F);
                xch_g[(j % 2 * 128 + mine) * 4 + 1] = g[1];
                float b_max_2 = s_vals[(32) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_2 = max_noftz(b_max_2, s_vals[(32) + _lr]);
                }
                g[2] = ((b_max_2 > -CUDART_INF_F) ? c_log2 * b_max_2 : -CUDART_INF_F);
                xch_g[(j % 2 * 128 + mine) * 4 + 2] = g[2];
                float b_max_3 = s_vals[(48) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_3 = max_noftz(b_max_3, s_vals[(48) + _lr]);
                }
                g[3] = ((b_max_3 > -CUDART_INF_F) ? c_log2 * b_max_3 : -CUDART_INF_F);
                xch_g[(j % 2 * 128 + mine) * 4 + 3] = g[3];
                asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                g[4] = xch_g[(j % 2 * 128 + other) * 4];
                g[5] = xch_g[(j % 2 * 128 + other) * 4 + 1];
                g[6] = xch_g[(j % 2 * 128 + other) * 4 + 2];
                g[7] = xch_g[(j % 2 * 128 + other) * 4 + 3];
                float page_max = g[0];
                #pragma unroll
                for (int _lr = 1; _lr < 8; _lr++) {
                    page_max = max_noftz(page_max, g[_lr]);
                }
                int move = page_max > -CUDART_INF_F && (m == -CUDART_INF_F || page_max > m + 3.0f);
                int rescale = move != 0 && m > -CUDART_INF_F;
                float _exp2_1 = approx_exp2(m - page_max);
                float alpha = ((rescale != 0) ? _exp2_1 : 1.0f);
                if (move != 0) {
                    m = page_max;
                }
                l_sum = l_sum * alpha;
                unsigned int codes[8];
                unsigned int sf_pair[2];
                float sf_fit[4];
                float sf_near[4];
                float _exp2_2 = approx_exp2(g[0] - m - 4.0f);
                sf_near[0] = ((g[0] > -CUDART_INF_F) ? 448.0f * _exp2_2 : 0.0f);
                float _exp2_3 = approx_exp2(g[1] - m - 4.0f);
                sf_near[1] = ((g[1] > -CUDART_INF_F) ? 448.0f * _exp2_3 : 0.0f);
                float _exp2_4 = approx_exp2(g[2] - m - 4.0f);
                sf_near[2] = ((g[2] > -CUDART_INF_F) ? 448.0f * _exp2_4 : 0.0f);
                float _exp2_5 = approx_exp2(g[3] - m - 4.0f);
                sf_near[3] = ((g[3] > -CUDART_INF_F) ? 448.0f * _exp2_5 : 0.0f);
                unsigned int near_word[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(sf_near[0]), "f"(sf_near[1]),
                                           "f"(sf_near[2]), "f"(sf_near[3]));
                    near_word[0] = _packed;
                }
                float six_rho_d[4];
                uint16_t _e4m3x2_decode_6 = (uint16_t)(near_word[0] >> 0 & 0xFFu);
                uint32_t _f16x2_decode_6;
                float sf_near_r_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_6) : "h"(_e4m3x2_decode_6));
                uint16_t _f16_decode_6 = (uint16_t)_f16x2_decode_6;
                asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_0) : "h"(_f16_decode_6));
                float _rcp_7 = approx_rcp(sf_near_r_0);
                six_rho_d[0] = ((sf_near_r_0 > 0.0f) ? 6.0f * sf_near[0] * _rcp_7 : 0.0f);
                uint16_t _e4m3x2_decode_8 = (uint16_t)(near_word[0] >> 8 & 0xFFu);
                uint32_t _f16x2_decode_8;
                float sf_near_r_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_8) : "h"(_e4m3x2_decode_8));
                uint16_t _f16_decode_8 = (uint16_t)_f16x2_decode_8;
                asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_1) : "h"(_f16_decode_8));
                float _rcp_9 = approx_rcp(sf_near_r_1);
                six_rho_d[1] = ((sf_near_r_1 > 0.0f) ? 6.0f * sf_near[1] * _rcp_9 : 0.0f);
                uint16_t _e4m3x2_decode_10 = (uint16_t)(near_word[0] >> 16 & 0xFFu);
                uint32_t _f16x2_decode_10;
                float sf_near_r_2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_10) : "h"(_e4m3x2_decode_10));
                uint16_t _f16_decode_10 = (uint16_t)_f16x2_decode_10;
                asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_2) : "h"(_f16_decode_10));
                float _rcp_11 = approx_rcp(sf_near_r_2);
                six_rho_d[2] = ((sf_near_r_2 > 0.0f) ? 6.0f * sf_near[2] * _rcp_11 : 0.0f);
                uint16_t _e4m3x2_decode_12 = (uint16_t)(near_word[0] >> 24 & 0xFFu);
                uint32_t _f16x2_decode_12;
                float sf_near_r_3;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_decode_12));
                uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
                asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_3) : "h"(_f16_decode_12));
                float _rcp_13 = approx_rcp(sf_near_r_3);
                six_rho_d[3] = ((sf_near_r_3 > 0.0f) ? 6.0f * sf_near[3] * _rcp_13 : 0.0f);
                int g_live_0 = g[0] > -CUDART_INF_F;
                float g_safe_0 = ((g_live_0 != 0) ? g[0] : 0.0f);
                const float2 _fma_b2_14 = {c_log2, c_log2};
                const float2 _fma_c2_15 = {-g_safe_0, -g_safe_0};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_14, _fma_c2_15);
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
                float b_sum_0 = t8[0] + t8[1];
                float _exp2_16 = approx_exp2(g_safe_0 - m);
                float b_weight_0 = ((g_live_0 != 0) ? _exp2_16 : 0.0f);
                l_sum = l_sum + b_sum_0 * b_weight_0;
                float e_min_0 = s_vals[0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    e_min_0 = fminf(e_min_0, s_vals[_lr]);
                }
                int flat_0 = dither != 0 && e_min_0 >= 0.5f;
                float _exp2_17 = approx_exp2(g_safe_0 - m - 4.0f);
                float sf_ex_0 = 448.0f * _exp2_17;
                if (flat_0 != 0) {
                    const float2 _fma_b2_18 = {six_rho_d[0], six_rho_d[0]};
                    const float2 _fma_c2_19 = {0.0f, 0.0f};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_18, _fma_c2_19);
                    uint16_t _cvt_rs_x4_f32_20;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_20) : "f"(s_vals[3]), "f"(s_vals[2]), "f"(s_vals[1]), "f"(s_vals[0]), "r"(3103267870u));
                    unsigned int w_lo_0 = _cvt_rs_x4_f32_20;
                    uint16_t _cvt_rs_x4_f32_21;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_21) : "f"(s_vals[7]), "f"(s_vals[6]), "f"(s_vals[5]), "f"(s_vals[4]), "r"(2564298778u));
                    unsigned int w_hi_0 = _cvt_rs_x4_f32_21;
                    codes[0] = w_lo_0 | w_hi_0 << 16;
                    uint16_t _cvt_rs_x4_f32_22;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_22) : "f"(s_vals[11]), "f"(s_vals[10]), "f"(s_vals[9]), "f"(s_vals[8]), "r"(2833781782u));
                    unsigned int w_lo_1 = _cvt_rs_x4_f32_22;
                    uint16_t _cvt_rs_x4_f32_23;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_23) : "f"(s_vals[15]), "f"(s_vals[14]), "f"(s_vals[13]), "f"(s_vals[12]), "r"(2294812690u));
                    unsigned int w_hi_1 = _cvt_rs_x4_f32_23;
                    codes[1] = w_lo_1 | w_hi_1 << 16;
                    sf_fit[0] = sf_near[0];
                } else {
                    const float2 _scale2_24 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_ls], _scale2_24);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(s_vals[0]), "f"(s_vals[1]), "f"(s_vals[2]), "f"(s_vals[3]), "f"(s_vals[4]), "f"(s_vals[5]), "f"(s_vals[6]), "f"(s_vals[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(s_vals[8]), "f"(s_vals[9]), "f"(s_vals[10]), "f"(s_vals[11]), "f"(s_vals[12]), "f"(s_vals[13]), "f"(s_vals[14]), "f"(s_vals[15]));
                    float q_b[16];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[0]));
                        (&q_b[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b[0]), "=f"(q_b[1]), "=f"(q_b[2]), "=f"(q_b[3]), "=f"(q_b[4]), "=f"(q_b[5]), "=f"(q_b[6]), "=f"(q_b[7]) : "r"(codes[0]));
                        #endif
                    }
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[1]));
                        (&q_b[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b[8]), "=f"(q_b[9]), "=f"(q_b[10]), "=f"(q_b[11]), "=f"(q_b[12]), "=f"(q_b[13]), "=f"(q_b[14]), "=f"(q_b[15]) : "r"(codes[1]));
                        #endif
                    }
                    float qe[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b)[_ls], reinterpret_cast<const float2*>((s_vals + 0))[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_b)[_ls], reinterpret_cast<const float2*>(q_b)[_ls]);
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
                    float qv = t8_0[0] + t8_0[1];
                    float t8_1[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_1)[_la] = add_f32x2(reinterpret_cast<float2*>((q_b + 0))[_la], reinterpret_cast<const float2*>((q_b + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_1)[_la], reinterpret_cast<const float2*>((t8_1 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_1)[_la], reinterpret_cast<const float2*>((t8_1 + 2))[_la]);
                    float qq = t8_1[0] + t8_1[1];
                    float _rcp_25 = approx_rcp(qq);
                    sf_fit[0] = ((g_live_0 != 0 && qq > 0.0f) ? sf_ex_0 * qv * _rcp_25 : 0.0f);
                }
                int g_live_1 = g[1] > -CUDART_INF_F;
                float g_safe_1 = ((g_live_1 != 0) ? g[1] : 0.0f);
                const float2 _fma_b2_26 = {c_log2, c_log2};
                const float2 _fma_c2_27 = {-g_safe_1, -g_safe_1};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_26, _fma_c2_27);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    s_vals[_le + 16] = approx_exp2(s_vals[_le + 16]);
                }
                float t8_2[8];
                #pragma unroll
                for (int _la = 0; _la < 4; _la++)
                    reinterpret_cast<float2*>(t8_2)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 16))[_la], reinterpret_cast<const float2*>((s_vals + 24))[_la]);
                #pragma unroll
                for (int _la = 0; _la < 2; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(t8_2)[_la], reinterpret_cast<const float2*>((t8_2 + 4))[_la]);
                #pragma unroll
                for (int _la = 0; _la < 1; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(t8_2)[_la], reinterpret_cast<const float2*>((t8_2 + 2))[_la]);
                float b_sum_1 = t8_2[0] + t8_2[1];
                float _exp2_28 = approx_exp2(g_safe_1 - m);
                float b_weight_1 = ((g_live_1 != 0) ? _exp2_28 : 0.0f);
                l_sum = l_sum + b_sum_1 * b_weight_1;
                float e_min_1 = s_vals[(16) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    e_min_1 = fminf(e_min_1, s_vals[(16) + _lr]);
                }
                int flat_1 = dither != 0 && e_min_1 >= 0.5f;
                float _exp2_29 = approx_exp2(g_safe_1 - m - 4.0f);
                float sf_ex_1 = 448.0f * _exp2_29;
                if (flat_1 != 0) {
                    const float2 _fma_b2_30 = {six_rho_d[1], six_rho_d[1]};
                    const float2 _fma_c2_31 = {0.0f, 0.0f};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_30, _fma_c2_31);
                    uint16_t _cvt_rs_x4_f32_32;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_32) : "f"(s_vals[19]), "f"(s_vals[18]), "f"(s_vals[17]), "f"(s_vals[16]), "r"(3103267870u));
                    unsigned int w_lo_2 = _cvt_rs_x4_f32_32;
                    uint16_t _cvt_rs_x4_f32_33;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_33) : "f"(s_vals[23]), "f"(s_vals[22]), "f"(s_vals[21]), "f"(s_vals[20]), "r"(2564298778u));
                    unsigned int w_hi_2 = _cvt_rs_x4_f32_33;
                    codes[2] = w_lo_2 | w_hi_2 << 16;
                    uint16_t _cvt_rs_x4_f32_34;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_34) : "f"(s_vals[27]), "f"(s_vals[26]), "f"(s_vals[25]), "f"(s_vals[24]), "r"(2833781782u));
                    unsigned int w_lo_3 = _cvt_rs_x4_f32_34;
                    uint16_t _cvt_rs_x4_f32_35;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_35) : "f"(s_vals[31]), "f"(s_vals[30]), "f"(s_vals[29]), "f"(s_vals[28]), "r"(2294812690u));
                    unsigned int w_hi_3 = _cvt_rs_x4_f32_35;
                    codes[3] = w_lo_3 | w_hi_3 << 16;
                    sf_fit[1] = sf_near[1];
                } else {
                    const float2 _scale2_36 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_ls], _scale2_36);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[2]) : "f"(s_vals[16]), "f"(s_vals[17]), "f"(s_vals[18]), "f"(s_vals[19]), "f"(s_vals[20]), "f"(s_vals[21]), "f"(s_vals[22]), "f"(s_vals[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[3]) : "f"(s_vals[24]), "f"(s_vals[25]), "f"(s_vals[26]), "f"(s_vals[27]), "f"(s_vals[28]), "f"(s_vals[29]), "f"(s_vals[30]), "f"(s_vals[31]));
                    float q_b_1[16];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[2 + 0]));
                        (&q_b_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b_1[0]), "=f"(q_b_1[1]), "=f"(q_b_1[2]), "=f"(q_b_1[3]), "=f"(q_b_1[4]), "=f"(q_b_1[5]), "=f"(q_b_1[6]), "=f"(q_b_1[7]) : "r"(codes[2 + 0]));
                        #endif
                    }
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[2 + 1]));
                        (&q_b_1[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b_1[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b_1[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b_1[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b_1[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b_1[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b_1[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b_1[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b_1[8]), "=f"(q_b_1[9]), "=f"(q_b_1[10]), "=f"(q_b_1[11]), "=f"(q_b_1[12]), "=f"(q_b_1[13]), "=f"(q_b_1[14]), "=f"(q_b_1[15]) : "r"(codes[2 + 1]));
                        #endif
                    }
                    float qe_1[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_1)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_1)[_ls], reinterpret_cast<const float2*>((s_vals + 16))[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_b_1)[_ls], reinterpret_cast<const float2*>(q_b_1)[_ls]);
                    float t8_3[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_3)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_1 + 0))[_la], reinterpret_cast<const float2*>((qe_1 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_3)[_la], reinterpret_cast<const float2*>((t8_3 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_3)[_la], reinterpret_cast<const float2*>((t8_3 + 2))[_la]);
                    float qv_1 = t8_3[0] + t8_3[1];
                    float t8_4[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_4)[_la] = add_f32x2(reinterpret_cast<float2*>((q_b_1 + 0))[_la], reinterpret_cast<const float2*>((q_b_1 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_4)[_la], reinterpret_cast<const float2*>((t8_4 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_4)[_la], reinterpret_cast<const float2*>((t8_4 + 2))[_la]);
                    float qq_1 = t8_4[0] + t8_4[1];
                    float _rcp_37 = approx_rcp(qq_1);
                    sf_fit[1] = ((g_live_1 != 0 && qq_1 > 0.0f) ? sf_ex_1 * qv_1 * _rcp_37 : 0.0f);
                }
                int g_live_2 = g[2] > -CUDART_INF_F;
                float g_safe_2 = ((g_live_2 != 0) ? g[2] : 0.0f);
                const float2 _fma_b2_38 = {c_log2, c_log2};
                const float2 _fma_c2_39 = {-g_safe_2, -g_safe_2};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_38, _fma_c2_39);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    s_vals[_le + 32] = approx_exp2(s_vals[_le + 32]);
                }
                float t8_5[8];
                #pragma unroll
                for (int _la = 0; _la < 4; _la++)
                    reinterpret_cast<float2*>(t8_5)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 32))[_la], reinterpret_cast<const float2*>((s_vals + 40))[_la]);
                #pragma unroll
                for (int _la = 0; _la < 2; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(t8_5)[_la], reinterpret_cast<const float2*>((t8_5 + 4))[_la]);
                #pragma unroll
                for (int _la = 0; _la < 1; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(t8_5)[_la], reinterpret_cast<const float2*>((t8_5 + 2))[_la]);
                float b_sum_2 = t8_5[0] + t8_5[1];
                float _exp2_40 = approx_exp2(g_safe_2 - m);
                float b_weight_2 = ((g_live_2 != 0) ? _exp2_40 : 0.0f);
                l_sum = l_sum + b_sum_2 * b_weight_2;
                float e_min_2 = s_vals[(32) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    e_min_2 = fminf(e_min_2, s_vals[(32) + _lr]);
                }
                int flat_2 = dither != 0 && e_min_2 >= 0.5f;
                float _exp2_41 = approx_exp2(g_safe_2 - m - 4.0f);
                float sf_ex_2 = 448.0f * _exp2_41;
                if (flat_2 != 0) {
                    const float2 _fma_b2_42 = {six_rho_d[2], six_rho_d[2]};
                    const float2 _fma_c2_43 = {0.0f, 0.0f};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_42, _fma_c2_43);
                    uint16_t _cvt_rs_x4_f32_44;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_44) : "f"(s_vals[35]), "f"(s_vals[34]), "f"(s_vals[33]), "f"(s_vals[32]), "r"(3103267870u));
                    unsigned int w_lo_4 = _cvt_rs_x4_f32_44;
                    uint16_t _cvt_rs_x4_f32_45;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_45) : "f"(s_vals[39]), "f"(s_vals[38]), "f"(s_vals[37]), "f"(s_vals[36]), "r"(2564298778u));
                    unsigned int w_hi_4 = _cvt_rs_x4_f32_45;
                    codes[4] = w_lo_4 | w_hi_4 << 16;
                    uint16_t _cvt_rs_x4_f32_46;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_46) : "f"(s_vals[43]), "f"(s_vals[42]), "f"(s_vals[41]), "f"(s_vals[40]), "r"(2833781782u));
                    unsigned int w_lo_5 = _cvt_rs_x4_f32_46;
                    uint16_t _cvt_rs_x4_f32_47;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_47) : "f"(s_vals[47]), "f"(s_vals[46]), "f"(s_vals[45]), "f"(s_vals[44]), "r"(2294812690u));
                    unsigned int w_hi_5 = _cvt_rs_x4_f32_47;
                    codes[5] = w_lo_5 | w_hi_5 << 16;
                    sf_fit[2] = sf_near[2];
                } else {
                    const float2 _scale2_48 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_ls], _scale2_48);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[4]) : "f"(s_vals[32]), "f"(s_vals[33]), "f"(s_vals[34]), "f"(s_vals[35]), "f"(s_vals[36]), "f"(s_vals[37]), "f"(s_vals[38]), "f"(s_vals[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[5]) : "f"(s_vals[40]), "f"(s_vals[41]), "f"(s_vals[42]), "f"(s_vals[43]), "f"(s_vals[44]), "f"(s_vals[45]), "f"(s_vals[46]), "f"(s_vals[47]));
                    float q_b_2[16];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[4 + 0]));
                        (&q_b_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b_2[0]), "=f"(q_b_2[1]), "=f"(q_b_2[2]), "=f"(q_b_2[3]), "=f"(q_b_2[4]), "=f"(q_b_2[5]), "=f"(q_b_2[6]), "=f"(q_b_2[7]) : "r"(codes[4 + 0]));
                        #endif
                    }
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[4 + 1]));
                        (&q_b_2[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b_2[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b_2[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b_2[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b_2[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b_2[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b_2[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b_2[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b_2[8]), "=f"(q_b_2[9]), "=f"(q_b_2[10]), "=f"(q_b_2[11]), "=f"(q_b_2[12]), "=f"(q_b_2[13]), "=f"(q_b_2[14]), "=f"(q_b_2[15]) : "r"(codes[4 + 1]));
                        #endif
                    }
                    float qe_2[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_2)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_2)[_ls], reinterpret_cast<const float2*>((s_vals + 32))[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_b_2)[_ls], reinterpret_cast<const float2*>(q_b_2)[_ls]);
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
                    float qv_2 = t8_6[0] + t8_6[1];
                    float t8_7[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_7)[_la] = add_f32x2(reinterpret_cast<float2*>((q_b_2 + 0))[_la], reinterpret_cast<const float2*>((q_b_2 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_7)[_la], reinterpret_cast<const float2*>((t8_7 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_7)[_la], reinterpret_cast<const float2*>((t8_7 + 2))[_la]);
                    float qq_2 = t8_7[0] + t8_7[1];
                    float _rcp_49 = approx_rcp(qq_2);
                    sf_fit[2] = ((g_live_2 != 0 && qq_2 > 0.0f) ? sf_ex_2 * qv_2 * _rcp_49 : 0.0f);
                }
                int g_live_3 = g[3] > -CUDART_INF_F;
                float g_safe_3 = ((g_live_3 != 0) ? g[3] : 0.0f);
                const float2 _fma_b2_50 = {c_log2, c_log2};
                const float2 _fma_c2_51 = {-g_safe_3, -g_safe_3};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_50, _fma_c2_51);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    s_vals[_le + 48] = approx_exp2(s_vals[_le + 48]);
                }
                float t8_8[8];
                #pragma unroll
                for (int _la = 0; _la < 4; _la++)
                    reinterpret_cast<float2*>(t8_8)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 48))[_la], reinterpret_cast<const float2*>((s_vals + 56))[_la]);
                #pragma unroll
                for (int _la = 0; _la < 2; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(t8_8)[_la], reinterpret_cast<const float2*>((t8_8 + 4))[_la]);
                #pragma unroll
                for (int _la = 0; _la < 1; _la++)
                    add_f32x2_inplace(&reinterpret_cast<float2*>(t8_8)[_la], reinterpret_cast<const float2*>((t8_8 + 2))[_la]);
                float b_sum_3 = t8_8[0] + t8_8[1];
                float _exp2_52 = approx_exp2(g_safe_3 - m);
                float b_weight_3 = ((g_live_3 != 0) ? _exp2_52 : 0.0f);
                l_sum = l_sum + b_sum_3 * b_weight_3;
                float e_min_3 = s_vals[(48) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    e_min_3 = fminf(e_min_3, s_vals[(48) + _lr]);
                }
                int flat_3 = dither != 0 && e_min_3 >= 0.5f;
                float _exp2_53 = approx_exp2(g_safe_3 - m - 4.0f);
                float sf_ex_3 = 448.0f * _exp2_53;
                if (flat_3 != 0) {
                    const float2 _fma_b2_54 = {six_rho_d[3], six_rho_d[3]};
                    const float2 _fma_c2_55 = {0.0f, 0.0f};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_54, _fma_c2_55);
                    uint16_t _cvt_rs_x4_f32_56;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_56) : "f"(s_vals[51]), "f"(s_vals[50]), "f"(s_vals[49]), "f"(s_vals[48]), "r"(3103267870u));
                    unsigned int w_lo_6 = _cvt_rs_x4_f32_56;
                    uint16_t _cvt_rs_x4_f32_57;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_57) : "f"(s_vals[55]), "f"(s_vals[54]), "f"(s_vals[53]), "f"(s_vals[52]), "r"(2564298778u));
                    unsigned int w_hi_6 = _cvt_rs_x4_f32_57;
                    codes[6] = w_lo_6 | w_hi_6 << 16;
                    uint16_t _cvt_rs_x4_f32_58;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_58) : "f"(s_vals[59]), "f"(s_vals[58]), "f"(s_vals[57]), "f"(s_vals[56]), "r"(2833781782u));
                    unsigned int w_lo_7 = _cvt_rs_x4_f32_58;
                    uint16_t _cvt_rs_x4_f32_59;
                    asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_59) : "f"(s_vals[63]), "f"(s_vals[62]), "f"(s_vals[61]), "f"(s_vals[60]), "r"(2294812690u));
                    unsigned int w_hi_7 = _cvt_rs_x4_f32_59;
                    codes[7] = w_lo_7 | w_hi_7 << 16;
                    sf_fit[3] = sf_near[3];
                } else {
                    const float2 _scale2_60 = {6.0f, 6.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_ls], _scale2_60);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[6]) : "f"(s_vals[48]), "f"(s_vals[49]), "f"(s_vals[50]), "f"(s_vals[51]), "f"(s_vals[52]), "f"(s_vals[53]), "f"(s_vals[54]), "f"(s_vals[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[7]) : "f"(s_vals[56]), "f"(s_vals[57]), "f"(s_vals[58]), "f"(s_vals[59]), "f"(s_vals[60]), "f"(s_vals[61]), "f"(s_vals[62]), "f"(s_vals[63]));
                    float q_b_3[16];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[6 + 0]));
                        (&q_b_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b_3[0]), "=f"(q_b_3[1]), "=f"(q_b_3[2]), "=f"(q_b_3[3]), "=f"(q_b_3[4]), "=f"(q_b_3[5]), "=f"(q_b_3[6]), "=f"(q_b_3[7]) : "r"(codes[6 + 0]));
                        #endif
                    }
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(codes[6 + 1]));
                        (&q_b_3[8])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&q_b_3[8])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&q_b_3[10])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&q_b_3[10])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&q_b_3[12])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&q_b_3[12])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&q_b_3[14])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&q_b_3[14])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(q_b_3[8]), "=f"(q_b_3[9]), "=f"(q_b_3[10]), "=f"(q_b_3[11]), "=f"(q_b_3[12]), "=f"(q_b_3[13]), "=f"(q_b_3[14]), "=f"(q_b_3[15]) : "r"(codes[6 + 1]));
                        #endif
                    }
                    float qe_3[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_3)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_3)[_ls], reinterpret_cast<const float2*>((s_vals + 48))[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_b_3)[_ls], reinterpret_cast<const float2*>(q_b_3)[_ls]);
                    float t8_9[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_9)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_3 + 0))[_la], reinterpret_cast<const float2*>((qe_3 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_9)[_la], reinterpret_cast<const float2*>((t8_9 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_9)[_la], reinterpret_cast<const float2*>((t8_9 + 2))[_la]);
                    float qv_3 = t8_9[0] + t8_9[1];
                    float t8_10[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_10)[_la] = add_f32x2(reinterpret_cast<float2*>((q_b_3 + 0))[_la], reinterpret_cast<const float2*>((q_b_3 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_10)[_la], reinterpret_cast<const float2*>((t8_10 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_10)[_la], reinterpret_cast<const float2*>((t8_10 + 2))[_la]);
                    float qq_3 = t8_10[0] + t8_10[1];
                    float _rcp_61 = approx_rcp(qq_3);
                    sf_fit[3] = ((g_live_3 != 0 && qq_3 > 0.0f) ? sf_ex_3 * qv_3 * _rcp_61 : 0.0f);
                }
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
                xch_w[j % 2 * 128 + mine] = sf_mine;
                asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                unsigned int sf_other = xch_w[j % 2 * 128 + other];
                sf_pair[0] = sf_mine;
                sf_pair[1] = sf_other;
                mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _warp_vote_62 = __any_sync(0xFFFFFFFF, rescale != 0);
                if (_warp_vote_62 != 0) {
                    float o_chunk_0[32];
                    unsigned int o_addr_0 = taddr + (unsigned int)lane_base_s;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_0[0]), "=f"(o_chunk_0[1]), "=f"(o_chunk_0[2]), "=f"(o_chunk_0[3]), "=f"(o_chunk_0[4]), "=f"(o_chunk_0[5]), "=f"(o_chunk_0[6]), "=f"(o_chunk_0[7]), "=f"(o_chunk_0[8]), "=f"(o_chunk_0[9]), "=f"(o_chunk_0[10]), "=f"(o_chunk_0[11]), "=f"(o_chunk_0[12]), "=f"(o_chunk_0[13]), "=f"(o_chunk_0[14]), "=f"(o_chunk_0[15]), "=f"(o_chunk_0[16]), "=f"(o_chunk_0[17]), "=f"(o_chunk_0[18]), "=f"(o_chunk_0[19]), "=f"(o_chunk_0[20]), "=f"(o_chunk_0[21]), "=f"(o_chunk_0[22]), "=f"(o_chunk_0[23]), "=f"(o_chunk_0[24]), "=f"(o_chunk_0[25]), "=f"(o_chunk_0[26]), "=f"(o_chunk_0[27]), "=f"(o_chunk_0[28]), "=f"(o_chunk_0[29]), "=f"(o_chunk_0[30]), "=f"(o_chunk_0[31])
                        : "r"(o_addr_0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_63 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_0)[_ls], _scale2_63);
                    tmem_st_x32_f32(o_addr_0, o_chunk_0);
                    float o_chunk_1[32];
                    unsigned int o_addr_1 = taddr + (unsigned int)lane_base_s + 32;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_1[0]), "=f"(o_chunk_1[1]), "=f"(o_chunk_1[2]), "=f"(o_chunk_1[3]), "=f"(o_chunk_1[4]), "=f"(o_chunk_1[5]), "=f"(o_chunk_1[6]), "=f"(o_chunk_1[7]), "=f"(o_chunk_1[8]), "=f"(o_chunk_1[9]), "=f"(o_chunk_1[10]), "=f"(o_chunk_1[11]), "=f"(o_chunk_1[12]), "=f"(o_chunk_1[13]), "=f"(o_chunk_1[14]), "=f"(o_chunk_1[15]), "=f"(o_chunk_1[16]), "=f"(o_chunk_1[17]), "=f"(o_chunk_1[18]), "=f"(o_chunk_1[19]), "=f"(o_chunk_1[20]), "=f"(o_chunk_1[21]), "=f"(o_chunk_1[22]), "=f"(o_chunk_1[23]), "=f"(o_chunk_1[24]), "=f"(o_chunk_1[25]), "=f"(o_chunk_1[26]), "=f"(o_chunk_1[27]), "=f"(o_chunk_1[28]), "=f"(o_chunk_1[29]), "=f"(o_chunk_1[30]), "=f"(o_chunk_1[31])
                        : "r"(o_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_64 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_1)[_ls], _scale2_64);
                    tmem_st_x32_f32(o_addr_1, o_chunk_1);
                    float o_chunk_2[32];
                    unsigned int o_addr_2 = taddr + (unsigned int)lane_base_s + 64;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_2[0]), "=f"(o_chunk_2[1]), "=f"(o_chunk_2[2]), "=f"(o_chunk_2[3]), "=f"(o_chunk_2[4]), "=f"(o_chunk_2[5]), "=f"(o_chunk_2[6]), "=f"(o_chunk_2[7]), "=f"(o_chunk_2[8]), "=f"(o_chunk_2[9]), "=f"(o_chunk_2[10]), "=f"(o_chunk_2[11]), "=f"(o_chunk_2[12]), "=f"(o_chunk_2[13]), "=f"(o_chunk_2[14]), "=f"(o_chunk_2[15]), "=f"(o_chunk_2[16]), "=f"(o_chunk_2[17]), "=f"(o_chunk_2[18]), "=f"(o_chunk_2[19]), "=f"(o_chunk_2[20]), "=f"(o_chunk_2[21]), "=f"(o_chunk_2[22]), "=f"(o_chunk_2[23]), "=f"(o_chunk_2[24]), "=f"(o_chunk_2[25]), "=f"(o_chunk_2[26]), "=f"(o_chunk_2[27]), "=f"(o_chunk_2[28]), "=f"(o_chunk_2[29]), "=f"(o_chunk_2[30]), "=f"(o_chunk_2[31])
                        : "r"(o_addr_2));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_65 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_2)[_ls], _scale2_65);
                    tmem_st_x32_f32(o_addr_2, o_chunk_2);
                    float o_chunk_3[32];
                    unsigned int o_addr_3 = taddr + (unsigned int)lane_base_s + 96;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_3[0]), "=f"(o_chunk_3[1]), "=f"(o_chunk_3[2]), "=f"(o_chunk_3[3]), "=f"(o_chunk_3[4]), "=f"(o_chunk_3[5]), "=f"(o_chunk_3[6]), "=f"(o_chunk_3[7]), "=f"(o_chunk_3[8]), "=f"(o_chunk_3[9]), "=f"(o_chunk_3[10]), "=f"(o_chunk_3[11]), "=f"(o_chunk_3[12]), "=f"(o_chunk_3[13]), "=f"(o_chunk_3[14]), "=f"(o_chunk_3[15]), "=f"(o_chunk_3[16]), "=f"(o_chunk_3[17]), "=f"(o_chunk_3[18]), "=f"(o_chunk_3[19]), "=f"(o_chunk_3[20]), "=f"(o_chunk_3[21]), "=f"(o_chunk_3[22]), "=f"(o_chunk_3[23]), "=f"(o_chunk_3[24]), "=f"(o_chunk_3[25]), "=f"(o_chunk_3[26]), "=f"(o_chunk_3[27]), "=f"(o_chunk_3[28]), "=f"(o_chunk_3[29]), "=f"(o_chunk_3[30]), "=f"(o_chunk_3[31])
                        : "r"(o_addr_3));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_66 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_3)[_ls], _scale2_66);
                    tmem_st_x32_f32(o_addr_3, o_chunk_3);
                    float o_chunk_4[32];
                    unsigned int o_addr_4 = taddr + (unsigned int)lane_base_s + 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_4[0]), "=f"(o_chunk_4[1]), "=f"(o_chunk_4[2]), "=f"(o_chunk_4[3]), "=f"(o_chunk_4[4]), "=f"(o_chunk_4[5]), "=f"(o_chunk_4[6]), "=f"(o_chunk_4[7]), "=f"(o_chunk_4[8]), "=f"(o_chunk_4[9]), "=f"(o_chunk_4[10]), "=f"(o_chunk_4[11]), "=f"(o_chunk_4[12]), "=f"(o_chunk_4[13]), "=f"(o_chunk_4[14]), "=f"(o_chunk_4[15]), "=f"(o_chunk_4[16]), "=f"(o_chunk_4[17]), "=f"(o_chunk_4[18]), "=f"(o_chunk_4[19]), "=f"(o_chunk_4[20]), "=f"(o_chunk_4[21]), "=f"(o_chunk_4[22]), "=f"(o_chunk_4[23]), "=f"(o_chunk_4[24]), "=f"(o_chunk_4[25]), "=f"(o_chunk_4[26]), "=f"(o_chunk_4[27]), "=f"(o_chunk_4[28]), "=f"(o_chunk_4[29]), "=f"(o_chunk_4[30]), "=f"(o_chunk_4[31])
                        : "r"(o_addr_4));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_67 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_4)[_ls], _scale2_67);
                    tmem_st_x32_f32(o_addr_4, o_chunk_4);
                    float o_chunk_5[32];
                    unsigned int o_addr_5 = taddr + (unsigned int)lane_base_s + 160;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_5[0]), "=f"(o_chunk_5[1]), "=f"(o_chunk_5[2]), "=f"(o_chunk_5[3]), "=f"(o_chunk_5[4]), "=f"(o_chunk_5[5]), "=f"(o_chunk_5[6]), "=f"(o_chunk_5[7]), "=f"(o_chunk_5[8]), "=f"(o_chunk_5[9]), "=f"(o_chunk_5[10]), "=f"(o_chunk_5[11]), "=f"(o_chunk_5[12]), "=f"(o_chunk_5[13]), "=f"(o_chunk_5[14]), "=f"(o_chunk_5[15]), "=f"(o_chunk_5[16]), "=f"(o_chunk_5[17]), "=f"(o_chunk_5[18]), "=f"(o_chunk_5[19]), "=f"(o_chunk_5[20]), "=f"(o_chunk_5[21]), "=f"(o_chunk_5[22]), "=f"(o_chunk_5[23]), "=f"(o_chunk_5[24]), "=f"(o_chunk_5[25]), "=f"(o_chunk_5[26]), "=f"(o_chunk_5[27]), "=f"(o_chunk_5[28]), "=f"(o_chunk_5[29]), "=f"(o_chunk_5[30]), "=f"(o_chunk_5[31])
                        : "r"(o_addr_5));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_68 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_5)[_ls], _scale2_68);
                    tmem_st_x32_f32(o_addr_5, o_chunk_5);
                    float o_chunk_6[32];
                    unsigned int o_addr_6 = taddr + (unsigned int)lane_base_s + 192;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_6[0]), "=f"(o_chunk_6[1]), "=f"(o_chunk_6[2]), "=f"(o_chunk_6[3]), "=f"(o_chunk_6[4]), "=f"(o_chunk_6[5]), "=f"(o_chunk_6[6]), "=f"(o_chunk_6[7]), "=f"(o_chunk_6[8]), "=f"(o_chunk_6[9]), "=f"(o_chunk_6[10]), "=f"(o_chunk_6[11]), "=f"(o_chunk_6[12]), "=f"(o_chunk_6[13]), "=f"(o_chunk_6[14]), "=f"(o_chunk_6[15]), "=f"(o_chunk_6[16]), "=f"(o_chunk_6[17]), "=f"(o_chunk_6[18]), "=f"(o_chunk_6[19]), "=f"(o_chunk_6[20]), "=f"(o_chunk_6[21]), "=f"(o_chunk_6[22]), "=f"(o_chunk_6[23]), "=f"(o_chunk_6[24]), "=f"(o_chunk_6[25]), "=f"(o_chunk_6[26]), "=f"(o_chunk_6[27]), "=f"(o_chunk_6[28]), "=f"(o_chunk_6[29]), "=f"(o_chunk_6[30]), "=f"(o_chunk_6[31])
                        : "r"(o_addr_6));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_69 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_6)[_ls], _scale2_69);
                    tmem_st_x32_f32(o_addr_6, o_chunk_6);
                    float o_chunk_7[32];
                    unsigned int o_addr_7 = taddr + (unsigned int)lane_base_s + 224;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_7[0]), "=f"(o_chunk_7[1]), "=f"(o_chunk_7[2]), "=f"(o_chunk_7[3]), "=f"(o_chunk_7[4]), "=f"(o_chunk_7[5]), "=f"(o_chunk_7[6]), "=f"(o_chunk_7[7]), "=f"(o_chunk_7[8]), "=f"(o_chunk_7[9]), "=f"(o_chunk_7[10]), "=f"(o_chunk_7[11]), "=f"(o_chunk_7[12]), "=f"(o_chunk_7[13]), "=f"(o_chunk_7[14]), "=f"(o_chunk_7[15]), "=f"(o_chunk_7[16]), "=f"(o_chunk_7[17]), "=f"(o_chunk_7[18]), "=f"(o_chunk_7[19]), "=f"(o_chunk_7[20]), "=f"(o_chunk_7[21]), "=f"(o_chunk_7[22]), "=f"(o_chunk_7[23]), "=f"(o_chunk_7[24]), "=f"(o_chunk_7[25]), "=f"(o_chunk_7[26]), "=f"(o_chunk_7[27]), "=f"(o_chunk_7[28]), "=f"(o_chunk_7[29]), "=f"(o_chunk_7[30]), "=f"(o_chunk_7[31])
                        : "r"(o_addr_7));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_70 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_7)[_ls], _scale2_70);
                    tmem_st_x32_f32(o_addr_7, o_chunk_7);
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_p_addr + (unsigned int)(r * 64 + (2 * half ^ r >> 1 & 3) * 16)), "r"(codes[0]), "r"(codes[1]), "r"(codes[2]), "r"(codes[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_p_addr + (unsigned int)(r * 64 + (2 * half + 1 ^ r >> 1 & 3) * 16)), "r"(codes[4]), "r"(codes[5]), "r"(codes[6]), "r"(codes[7]) : "memory");
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
                tmem_st_x8_u32(taddr + (unsigned int)lane_base_s + 460, (const uint32_t*)sfa_p);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(p_full_addr), "r"(0) : "memory");
                _phase_pv_done_0 ^= 1;
            }
            xch_l[mine] = l_sum;
            asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
            float l_row = l_sum + xch_l[other];
            float gk_e = *kv_gs;
            float _rcp_71 = approx_rcp(gk_e * l_row);
            float inv = ((l_row > 0.0f) ? 0.005952381f * _rcp_71 : 0.0f);
            int direct = num_split == 1;
            int row_g = row_base + row_local;
            int o_base = (row_g * num_split + split_i) * 512 + 128 * half;
            mbarrier_wait(pv_done_addr, _phase_pv_done_0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float o_vals_0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_0[0]), "=f"(o_vals_0[1]), "=f"(o_vals_0[2]), "=f"(o_vals_0[3]), "=f"(o_vals_0[4]), "=f"(o_vals_0[5]), "=f"(o_vals_0[6]), "=f"(o_vals_0[7]), "=f"(o_vals_0[8]), "=f"(o_vals_0[9]), "=f"(o_vals_0[10]), "=f"(o_vals_0[11]), "=f"(o_vals_0[12]), "=f"(o_vals_0[13]), "=f"(o_vals_0[14]), "=f"(o_vals_0[15]), "=f"(o_vals_0[16]), "=f"(o_vals_0[17]), "=f"(o_vals_0[18]), "=f"(o_vals_0[19]), "=f"(o_vals_0[20]), "=f"(o_vals_0[21]), "=f"(o_vals_0[22]), "=f"(o_vals_0[23]), "=f"(o_vals_0[24]), "=f"(o_vals_0[25]), "=f"(o_vals_0[26]), "=f"(o_vals_0[27]), "=f"(o_vals_0[28]), "=f"(o_vals_0[29]), "=f"(o_vals_0[30]), "=f"(o_vals_0[31])
                : "r"(taddr + (unsigned int)lane_base_s));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_72 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_0)[_ls], _scale2_72);
            unsigned int o_bf16_0[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_0[_lp*2 + 0], o_vals_0[_lp*2+1 + 0]));
                o_bf16_0[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_0 = 0;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_0))[0] = reinterpret_cast<int4*>(o_bf16_0)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_0))[0] = reinterpret_cast<int4*>(o_bf16_0)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_0 + 8))[0] = reinterpret_cast<int4*>(o_bf16_0 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_0 + 8))[0] = reinterpret_cast<int4*>(o_bf16_0 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_0 + 16))[0] = reinterpret_cast<int4*>(o_bf16_0 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_0 + 16))[0] = reinterpret_cast<int4*>(o_bf16_0 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_0 + 24))[0] = reinterpret_cast<int4*>(o_bf16_0 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_0 + 24))[0] = reinterpret_cast<int4*>(o_bf16_0 + (12))[0];
                }
            }
            float o_vals_1[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_1[0]), "=f"(o_vals_1[1]), "=f"(o_vals_1[2]), "=f"(o_vals_1[3]), "=f"(o_vals_1[4]), "=f"(o_vals_1[5]), "=f"(o_vals_1[6]), "=f"(o_vals_1[7]), "=f"(o_vals_1[8]), "=f"(o_vals_1[9]), "=f"(o_vals_1[10]), "=f"(o_vals_1[11]), "=f"(o_vals_1[12]), "=f"(o_vals_1[13]), "=f"(o_vals_1[14]), "=f"(o_vals_1[15]), "=f"(o_vals_1[16]), "=f"(o_vals_1[17]), "=f"(o_vals_1[18]), "=f"(o_vals_1[19]), "=f"(o_vals_1[20]), "=f"(o_vals_1[21]), "=f"(o_vals_1[22]), "=f"(o_vals_1[23]), "=f"(o_vals_1[24]), "=f"(o_vals_1[25]), "=f"(o_vals_1[26]), "=f"(o_vals_1[27]), "=f"(o_vals_1[28]), "=f"(o_vals_1[29]), "=f"(o_vals_1[30]), "=f"(o_vals_1[31])
                : "r"(taddr + (unsigned int)lane_base_s + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_73 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_1)[_ls], _scale2_73);
            unsigned int o_bf16_1[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_1[_lp*2 + 0], o_vals_1[_lp*2+1 + 0]));
                o_bf16_1[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_1 = 32;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_1))[0] = reinterpret_cast<int4*>(o_bf16_1)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_1))[0] = reinterpret_cast<int4*>(o_bf16_1)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_1 + 8))[0] = reinterpret_cast<int4*>(o_bf16_1 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_1 + 8))[0] = reinterpret_cast<int4*>(o_bf16_1 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_1 + 16))[0] = reinterpret_cast<int4*>(o_bf16_1 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_1 + 16))[0] = reinterpret_cast<int4*>(o_bf16_1 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_1 + 24))[0] = reinterpret_cast<int4*>(o_bf16_1 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_1 + 24))[0] = reinterpret_cast<int4*>(o_bf16_1 + (12))[0];
                }
            }
            float o_vals_2[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_2[0]), "=f"(o_vals_2[1]), "=f"(o_vals_2[2]), "=f"(o_vals_2[3]), "=f"(o_vals_2[4]), "=f"(o_vals_2[5]), "=f"(o_vals_2[6]), "=f"(o_vals_2[7]), "=f"(o_vals_2[8]), "=f"(o_vals_2[9]), "=f"(o_vals_2[10]), "=f"(o_vals_2[11]), "=f"(o_vals_2[12]), "=f"(o_vals_2[13]), "=f"(o_vals_2[14]), "=f"(o_vals_2[15]), "=f"(o_vals_2[16]), "=f"(o_vals_2[17]), "=f"(o_vals_2[18]), "=f"(o_vals_2[19]), "=f"(o_vals_2[20]), "=f"(o_vals_2[21]), "=f"(o_vals_2[22]), "=f"(o_vals_2[23]), "=f"(o_vals_2[24]), "=f"(o_vals_2[25]), "=f"(o_vals_2[26]), "=f"(o_vals_2[27]), "=f"(o_vals_2[28]), "=f"(o_vals_2[29]), "=f"(o_vals_2[30]), "=f"(o_vals_2[31])
                : "r"(taddr + (unsigned int)lane_base_s + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_74 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_2)[_ls], _scale2_74);
            unsigned int o_bf16_2[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_2[_lp*2 + 0], o_vals_2[_lp*2+1 + 0]));
                o_bf16_2[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_2 = 64;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_2))[0] = reinterpret_cast<int4*>(o_bf16_2)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_2))[0] = reinterpret_cast<int4*>(o_bf16_2)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_2 + 8))[0] = reinterpret_cast<int4*>(o_bf16_2 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_2 + 8))[0] = reinterpret_cast<int4*>(o_bf16_2 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_2 + 16))[0] = reinterpret_cast<int4*>(o_bf16_2 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_2 + 16))[0] = reinterpret_cast<int4*>(o_bf16_2 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_2 + 24))[0] = reinterpret_cast<int4*>(o_bf16_2 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_2 + 24))[0] = reinterpret_cast<int4*>(o_bf16_2 + (12))[0];
                }
            }
            float o_vals_3[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_3[0]), "=f"(o_vals_3[1]), "=f"(o_vals_3[2]), "=f"(o_vals_3[3]), "=f"(o_vals_3[4]), "=f"(o_vals_3[5]), "=f"(o_vals_3[6]), "=f"(o_vals_3[7]), "=f"(o_vals_3[8]), "=f"(o_vals_3[9]), "=f"(o_vals_3[10]), "=f"(o_vals_3[11]), "=f"(o_vals_3[12]), "=f"(o_vals_3[13]), "=f"(o_vals_3[14]), "=f"(o_vals_3[15]), "=f"(o_vals_3[16]), "=f"(o_vals_3[17]), "=f"(o_vals_3[18]), "=f"(o_vals_3[19]), "=f"(o_vals_3[20]), "=f"(o_vals_3[21]), "=f"(o_vals_3[22]), "=f"(o_vals_3[23]), "=f"(o_vals_3[24]), "=f"(o_vals_3[25]), "=f"(o_vals_3[26]), "=f"(o_vals_3[27]), "=f"(o_vals_3[28]), "=f"(o_vals_3[29]), "=f"(o_vals_3[30]), "=f"(o_vals_3[31])
                : "r"(taddr + (unsigned int)lane_base_s + 96));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_75 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_3)[_ls], _scale2_75);
            unsigned int o_bf16_3[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_3[_lp*2 + 0], o_vals_3[_lp*2+1 + 0]));
                o_bf16_3[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_3 = 96;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_3))[0] = reinterpret_cast<int4*>(o_bf16_3)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_3))[0] = reinterpret_cast<int4*>(o_bf16_3)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_3 + 8))[0] = reinterpret_cast<int4*>(o_bf16_3 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_3 + 8))[0] = reinterpret_cast<int4*>(o_bf16_3 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_3 + 16))[0] = reinterpret_cast<int4*>(o_bf16_3 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_3 + 16))[0] = reinterpret_cast<int4*>(o_bf16_3 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_3 + 24))[0] = reinterpret_cast<int4*>(o_bf16_3 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_3 + 24))[0] = reinterpret_cast<int4*>(o_bf16_3 + (12))[0];
                }
            }
            float o_vals_4[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_4[0]), "=f"(o_vals_4[1]), "=f"(o_vals_4[2]), "=f"(o_vals_4[3]), "=f"(o_vals_4[4]), "=f"(o_vals_4[5]), "=f"(o_vals_4[6]), "=f"(o_vals_4[7]), "=f"(o_vals_4[8]), "=f"(o_vals_4[9]), "=f"(o_vals_4[10]), "=f"(o_vals_4[11]), "=f"(o_vals_4[12]), "=f"(o_vals_4[13]), "=f"(o_vals_4[14]), "=f"(o_vals_4[15]), "=f"(o_vals_4[16]), "=f"(o_vals_4[17]), "=f"(o_vals_4[18]), "=f"(o_vals_4[19]), "=f"(o_vals_4[20]), "=f"(o_vals_4[21]), "=f"(o_vals_4[22]), "=f"(o_vals_4[23]), "=f"(o_vals_4[24]), "=f"(o_vals_4[25]), "=f"(o_vals_4[26]), "=f"(o_vals_4[27]), "=f"(o_vals_4[28]), "=f"(o_vals_4[29]), "=f"(o_vals_4[30]), "=f"(o_vals_4[31])
                : "r"(taddr + (unsigned int)lane_base_s + 128));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_76 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_4)[_ls], _scale2_76);
            unsigned int o_bf16_4[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_4[_lp*2 + 0], o_vals_4[_lp*2+1 + 0]));
                o_bf16_4[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_4 = 256;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_4))[0] = reinterpret_cast<int4*>(o_bf16_4)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_4))[0] = reinterpret_cast<int4*>(o_bf16_4)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_4 + 8))[0] = reinterpret_cast<int4*>(o_bf16_4 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_4 + 8))[0] = reinterpret_cast<int4*>(o_bf16_4 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_4 + 16))[0] = reinterpret_cast<int4*>(o_bf16_4 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_4 + 16))[0] = reinterpret_cast<int4*>(o_bf16_4 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_4 + 24))[0] = reinterpret_cast<int4*>(o_bf16_4 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_4 + 24))[0] = reinterpret_cast<int4*>(o_bf16_4 + (12))[0];
                }
            }
            float o_vals_5[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_5[0]), "=f"(o_vals_5[1]), "=f"(o_vals_5[2]), "=f"(o_vals_5[3]), "=f"(o_vals_5[4]), "=f"(o_vals_5[5]), "=f"(o_vals_5[6]), "=f"(o_vals_5[7]), "=f"(o_vals_5[8]), "=f"(o_vals_5[9]), "=f"(o_vals_5[10]), "=f"(o_vals_5[11]), "=f"(o_vals_5[12]), "=f"(o_vals_5[13]), "=f"(o_vals_5[14]), "=f"(o_vals_5[15]), "=f"(o_vals_5[16]), "=f"(o_vals_5[17]), "=f"(o_vals_5[18]), "=f"(o_vals_5[19]), "=f"(o_vals_5[20]), "=f"(o_vals_5[21]), "=f"(o_vals_5[22]), "=f"(o_vals_5[23]), "=f"(o_vals_5[24]), "=f"(o_vals_5[25]), "=f"(o_vals_5[26]), "=f"(o_vals_5[27]), "=f"(o_vals_5[28]), "=f"(o_vals_5[29]), "=f"(o_vals_5[30]), "=f"(o_vals_5[31])
                : "r"(taddr + (unsigned int)lane_base_s + 160));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_77 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_5)[_ls], _scale2_77);
            unsigned int o_bf16_5[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_5[_lp*2 + 0], o_vals_5[_lp*2+1 + 0]));
                o_bf16_5[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_5 = 288;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_5))[0] = reinterpret_cast<int4*>(o_bf16_5)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_5))[0] = reinterpret_cast<int4*>(o_bf16_5)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_5 + 8))[0] = reinterpret_cast<int4*>(o_bf16_5 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_5 + 8))[0] = reinterpret_cast<int4*>(o_bf16_5 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_5 + 16))[0] = reinterpret_cast<int4*>(o_bf16_5 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_5 + 16))[0] = reinterpret_cast<int4*>(o_bf16_5 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_5 + 24))[0] = reinterpret_cast<int4*>(o_bf16_5 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_5 + 24))[0] = reinterpret_cast<int4*>(o_bf16_5 + (12))[0];
                }
            }
            float o_vals_6[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_6[0]), "=f"(o_vals_6[1]), "=f"(o_vals_6[2]), "=f"(o_vals_6[3]), "=f"(o_vals_6[4]), "=f"(o_vals_6[5]), "=f"(o_vals_6[6]), "=f"(o_vals_6[7]), "=f"(o_vals_6[8]), "=f"(o_vals_6[9]), "=f"(o_vals_6[10]), "=f"(o_vals_6[11]), "=f"(o_vals_6[12]), "=f"(o_vals_6[13]), "=f"(o_vals_6[14]), "=f"(o_vals_6[15]), "=f"(o_vals_6[16]), "=f"(o_vals_6[17]), "=f"(o_vals_6[18]), "=f"(o_vals_6[19]), "=f"(o_vals_6[20]), "=f"(o_vals_6[21]), "=f"(o_vals_6[22]), "=f"(o_vals_6[23]), "=f"(o_vals_6[24]), "=f"(o_vals_6[25]), "=f"(o_vals_6[26]), "=f"(o_vals_6[27]), "=f"(o_vals_6[28]), "=f"(o_vals_6[29]), "=f"(o_vals_6[30]), "=f"(o_vals_6[31])
                : "r"(taddr + (unsigned int)lane_base_s + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_78 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_6)[_ls], _scale2_78);
            unsigned int o_bf16_6[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_6[_lp*2 + 0], o_vals_6[_lp*2+1 + 0]));
                o_bf16_6[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_6 = 320;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_6))[0] = reinterpret_cast<int4*>(o_bf16_6)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_6))[0] = reinterpret_cast<int4*>(o_bf16_6)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_6 + 8))[0] = reinterpret_cast<int4*>(o_bf16_6 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_6 + 8))[0] = reinterpret_cast<int4*>(o_bf16_6 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_6 + 16))[0] = reinterpret_cast<int4*>(o_bf16_6 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_6 + 16))[0] = reinterpret_cast<int4*>(o_bf16_6 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_6 + 24))[0] = reinterpret_cast<int4*>(o_bf16_6 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_6 + 24))[0] = reinterpret_cast<int4*>(o_bf16_6 + (12))[0];
                }
            }
            float o_vals_7[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(o_vals_7[0]), "=f"(o_vals_7[1]), "=f"(o_vals_7[2]), "=f"(o_vals_7[3]), "=f"(o_vals_7[4]), "=f"(o_vals_7[5]), "=f"(o_vals_7[6]), "=f"(o_vals_7[7]), "=f"(o_vals_7[8]), "=f"(o_vals_7[9]), "=f"(o_vals_7[10]), "=f"(o_vals_7[11]), "=f"(o_vals_7[12]), "=f"(o_vals_7[13]), "=f"(o_vals_7[14]), "=f"(o_vals_7[15]), "=f"(o_vals_7[16]), "=f"(o_vals_7[17]), "=f"(o_vals_7[18]), "=f"(o_vals_7[19]), "=f"(o_vals_7[20]), "=f"(o_vals_7[21]), "=f"(o_vals_7[22]), "=f"(o_vals_7[23]), "=f"(o_vals_7[24]), "=f"(o_vals_7[25]), "=f"(o_vals_7[26]), "=f"(o_vals_7[27]), "=f"(o_vals_7[28]), "=f"(o_vals_7[29]), "=f"(o_vals_7[30]), "=f"(o_vals_7[31])
                : "r"(taddr + (unsigned int)lane_base_s + 224));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_79 = {inv, inv};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_7)[_ls], _scale2_79);
            unsigned int o_bf16_7[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_7[_lp*2 + 0], o_vals_7[_lp*2+1 + 0]));
                o_bf16_7[_lp] = *(uint32_t*)&_bf2;
            }
            int dim0_7 = 352;
            if (row_ok != 0) {
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_7))[0] = reinterpret_cast<int4*>(o_bf16_7)[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_7))[0] = reinterpret_cast<int4*>(o_bf16_7)[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_7 + 8))[0] = reinterpret_cast<int4*>(o_bf16_7 + (4))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_7 + 8))[0] = reinterpret_cast<int4*>(o_bf16_7 + (4))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_7 + 16))[0] = reinterpret_cast<int4*>(o_bf16_7 + (8))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_7 + 16))[0] = reinterpret_cast<int4*>(o_bf16_7 + (8))[0];
                }
                if (direct != 0) {
                    reinterpret_cast<int4*>(out + (row_g * 512 + 128 * half + dim0_7 + 24))[0] = reinterpret_cast<int4*>(o_bf16_7 + (12))[0];
                } else {
                    reinterpret_cast<int4*>(partial_O + (o_base + dim0_7 + 24))[0] = reinterpret_cast<int4*>(o_bf16_7 + (12))[0];
                }
            }
            if (row_ok != 0 && half == 0 && direct == 0) {
                partial_max[row_g * num_split + split_i] = m;
                partial_sum[row_g * num_split + split_i] = l_row;
            }
        }
    }
    // ---- Role: corr ----
    if (warp >= 4 && warp <= 7) {
        { // corr_main
            int split_i_1 = blockIdx.x / 2;
            int b_i_1 = blockIdx.z;
            int rows_b_1 = q_len * num_heads;
            int row0_1 = blockIdx.y * 128;
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
            int row_ok_1 = row_c < rows_valid_1;
            int q_row = ((row_ok_1 != 0) ? row_base_1 + row_c : row_base_1);
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
            if (row_ok_1 == 0) {
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
            int tok_b = qc / 2 * 64 + lane;
            int byte_sel = lane / 16;
            unsigned int v_sel[2];
            v_sel[0] = 64 + 17 * (unsigned int)byte_sel;
            v_sel[1] = 98 + 17 * (unsigned int)byte_sel;
            int n_last = my_n_1 - 1;
            unsigned int kw_next[20];
            int pid = page_table[b_i_1 * page_stride + my_start_1];
            int phys = ((pid < 0) ? 0 : pid) + page_offset;
            int ksf_base = phys * 1280 + lane * 4 + qc / 2 * 2;
            {
                kw_next[0] = *reinterpret_cast<const unsigned int*>(kv_sf + ksf_base + 0);
            }
            {
                kw_next[1] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 128) + 0);
            }
            {
                kw_next[2] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 256) + 0);
            }
            {
                kw_next[3] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 384) + 0);
            }
            {
                kw_next[4] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 512) + 0);
            }
            {
                kw_next[5] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 640) + 0);
            }
            {
                kw_next[6] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 768) + 0);
            }
            {
                kw_next[7] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 896) + 0);
            }
            {
                kw_next[8] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1024) + 0);
            }
            {
                kw_next[9] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1152) + 0);
            }
            {
                kw_next[10] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1) + 0);
            }
            {
                kw_next[11] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 128) + 0);
            }
            {
                kw_next[12] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 256) + 0);
            }
            {
                kw_next[13] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 384) + 0);
            }
            {
                kw_next[14] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 512) + 0);
            }
            {
                kw_next[15] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 640) + 0);
            }
            {
                kw_next[16] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 768) + 0);
            }
            {
                kw_next[17] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 896) + 0);
            }
            {
                kw_next[18] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 1024) + 0);
            }
            {
                kw_next[19] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base + 1 + 1152) + 0);
            }
            int h_l = lane >> 4;
            int s_l = lane >> 3 & 1;
            int jj_l = lane >> 1 & 3;
            int cc_l = lane & 1;
            unsigned int v_word = kv_sf[phys * 1280 + (4 * h_l + 2 * (qc / 2) + cc_l) * 128 + 64 * (jj_l % 2) + 2 * s_l + jj_l / 2];
            unsigned int v_next = v_word;
            int pid_0 = page_table[b_i_1 * page_stride + (my_start_1 + ((n_last >= 1) ? 1 : 0))];
            int phys_0 = ((pid_0 < 0) ? 0 : pid_0) + page_offset;
            int page_n = phys_0;
            unsigned int _phase_sfk_empty_0 = 1;
            unsigned int _phase_pv_done_0_1 = 1;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < my_n_1; j_1++) {
                int pg_c = my_start_1 + j_1;
                int page_keys = keys_1 - 128 * pg_c;
                unsigned int kw[20];
                kw[0] = kw_next[0];
                kw[1] = kw_next[1];
                kw[2] = kw_next[2];
                kw[3] = kw_next[3];
                kw[4] = kw_next[4];
                kw[5] = kw_next[5];
                kw[6] = kw_next[6];
                kw[7] = kw_next[7];
                kw[8] = kw_next[8];
                kw[9] = kw_next[9];
                kw[10] = kw_next[10];
                kw[11] = kw_next[11];
                kw[12] = kw_next[12];
                kw[13] = kw_next[13];
                kw[14] = kw_next[14];
                kw[15] = kw_next[15];
                kw[16] = kw_next[16];
                kw[17] = kw_next[17];
                kw[18] = kw_next[18];
                kw[19] = kw_next[19];
                unsigned int v_cur = v_next;
                int ksf_base_0 = page_n * 1280 + lane * 4 + qc / 2 * 2;
                {
                    kw_next[0] = *reinterpret_cast<const unsigned int*>(kv_sf + ksf_base_0 + 0);
                }
                {
                    kw_next[1] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 128) + 0);
                }
                {
                    kw_next[2] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 256) + 0);
                }
                {
                    kw_next[3] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 384) + 0);
                }
                {
                    kw_next[4] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 512) + 0);
                }
                {
                    kw_next[5] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 640) + 0);
                }
                {
                    kw_next[6] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 768) + 0);
                }
                {
                    kw_next[7] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 896) + 0);
                }
                {
                    kw_next[8] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1024) + 0);
                }
                {
                    kw_next[9] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1152) + 0);
                }
                {
                    kw_next[10] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1) + 0);
                }
                {
                    kw_next[11] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 128) + 0);
                }
                {
                    kw_next[12] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 256) + 0);
                }
                {
                    kw_next[13] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 384) + 0);
                }
                {
                    kw_next[14] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 512) + 0);
                }
                {
                    kw_next[15] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 640) + 0);
                }
                {
                    kw_next[16] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 768) + 0);
                }
                {
                    kw_next[17] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 896) + 0);
                }
                {
                    kw_next[18] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 1024) + 0);
                }
                {
                    kw_next[19] = *reinterpret_cast<const unsigned int*>(kv_sf + (ksf_base_0 + 1 + 1152) + 0);
                }
                int h_l_0 = lane >> 4;
                int s_l_0 = lane >> 3 & 1;
                int jj_l_0 = lane >> 1 & 3;
                int cc_l_0 = lane & 1;
                unsigned int v_word_0 = kv_sf[page_n * 1280 + (4 * h_l_0 + 2 * (qc / 2) + cc_l_0) * 128 + 64 * (jj_l_0 % 2) + 2 * s_l_0 + jj_l_0 / 2];
                v_next = v_word_0;
                int j2 = j_1 + 2;
                int pid_1 = page_table[b_i_1 * page_stride + (my_start_1 + ((j2 < n_last) ? j2 : n_last))];
                int phys_1 = ((pid_1 < 0) ? 0 : pid_1) + page_offset;
                page_n = phys_1;
                if (tok_b >= page_keys) {
                    kw[0] = 0;
                    kw[1] = 0;
                    kw[2] = 0;
                    kw[3] = 0;
                    kw[4] = 0;
                    kw[5] = 0;
                    kw[6] = 0;
                    kw[7] = 0;
                    kw[8] = 0;
                    kw[9] = 0;
                }
                if (tok_b + 32 >= page_keys) {
                    kw[10] = 0;
                    kw[11] = 0;
                    kw[12] = 0;
                    kw[13] = 0;
                    kw[14] = 0;
                    kw[15] = 0;
                    kw[16] = 0;
                    kw[17] = 0;
                    kw[18] = 0;
                    kw[19] = 0;
                }
                mbarrier_wait(sfk_empty_addr, _phase_sfk_empty_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 440), "r"(kw[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 442), "r"((kw + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 444), "r"((kw + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 446), "r"((kw + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 448), "r"((kw + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 450), "r"((kw + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 452), "r"((kw + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 454), "r"((kw + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 456), "r"((kw + 8)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 458), "r"((kw + 9)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 441), "r"((kw + 10)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 443), "r"((kw + 11)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 445), "r"((kw + 12)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 447), "r"((kw + 13)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 449), "r"((kw + 14)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 451), "r"((kw + 15)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 453), "r"((kw + 16)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 455), "r"((kw + 17)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 457), "r"((kw + 18)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 459), "r"((kw + 19)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sfk_full_addr), "r"(0) : "memory");
                _phase_sfk_empty_0 ^= 1;
                unsigned int vw[16];
                unsigned int kj_0[4];
                unsigned int _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, v_cur, 0);
                kj_0[0] = _shfl_sync_22;
                unsigned int _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, v_cur, 2);
                kj_0[1] = _shfl_sync_23;
                unsigned int _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, v_cur, 4);
                kj_0[2] = _shfl_sync_24;
                unsigned int _shfl_sync_25 = __shfl_sync(0xFFFFFFFF, v_cur, 6);
                kj_0[3] = _shfl_sync_25;
                uint32_t lo_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_0) : "r"(kj_0[0]), "r"(kj_0[1]), "r"(v_sel[0]));
                uint32_t hi_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_0) : "r"(kj_0[2]), "r"(kj_0[3]), "r"(v_sel[0]));
                uint32_t _prmt_26;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_26) : "r"(lo_0), "r"(hi_0));
                vw[0] = _prmt_26;
                uint32_t lo_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_1) : "r"(kj_0[0]), "r"(kj_0[1]), "r"(v_sel[1]));
                uint32_t hi_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_1) : "r"(kj_0[2]), "r"(kj_0[3]), "r"(v_sel[1]));
                uint32_t _prmt_27;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_27) : "r"(lo_1), "r"(hi_1));
                vw[1] = _prmt_27;
                unsigned int kj_1[4];
                unsigned int _shfl_sync_28 = __shfl_sync(0xFFFFFFFF, v_cur, 1);
                kj_1[0] = _shfl_sync_28;
                unsigned int _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, v_cur, 3);
                kj_1[1] = _shfl_sync_29;
                unsigned int _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, v_cur, 5);
                kj_1[2] = _shfl_sync_30;
                unsigned int _shfl_sync_31 = __shfl_sync(0xFFFFFFFF, v_cur, 7);
                kj_1[3] = _shfl_sync_31;
                uint32_t lo_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_2) : "r"(kj_1[0]), "r"(kj_1[1]), "r"(v_sel[0]));
                uint32_t hi_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_2) : "r"(kj_1[2]), "r"(kj_1[3]), "r"(v_sel[0]));
                uint32_t _prmt_32;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_32) : "r"(lo_2), "r"(hi_2));
                vw[2] = _prmt_32;
                uint32_t lo_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_3) : "r"(kj_1[0]), "r"(kj_1[1]), "r"(v_sel[1]));
                uint32_t hi_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_3) : "r"(kj_1[2]), "r"(kj_1[3]), "r"(v_sel[1]));
                uint32_t _prmt_33;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_33) : "r"(lo_3), "r"(hi_3));
                vw[3] = _prmt_33;
                unsigned int kj_2[4];
                unsigned int _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, v_cur, 8);
                kj_2[0] = _shfl_sync_34;
                unsigned int _shfl_sync_35 = __shfl_sync(0xFFFFFFFF, v_cur, 10);
                kj_2[1] = _shfl_sync_35;
                unsigned int _shfl_sync_36 = __shfl_sync(0xFFFFFFFF, v_cur, 12);
                kj_2[2] = _shfl_sync_36;
                unsigned int _shfl_sync_37 = __shfl_sync(0xFFFFFFFF, v_cur, 14);
                kj_2[3] = _shfl_sync_37;
                uint32_t lo_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_4) : "r"(kj_2[0]), "r"(kj_2[1]), "r"(v_sel[0]));
                uint32_t hi_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_4) : "r"(kj_2[2]), "r"(kj_2[3]), "r"(v_sel[0]));
                uint32_t _prmt_38;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_38) : "r"(lo_4), "r"(hi_4));
                vw[4] = _prmt_38;
                uint32_t lo_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_5) : "r"(kj_2[0]), "r"(kj_2[1]), "r"(v_sel[1]));
                uint32_t hi_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_5) : "r"(kj_2[2]), "r"(kj_2[3]), "r"(v_sel[1]));
                uint32_t _prmt_39;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_39) : "r"(lo_5), "r"(hi_5));
                vw[5] = _prmt_39;
                unsigned int kj_3[4];
                unsigned int _shfl_sync_40 = __shfl_sync(0xFFFFFFFF, v_cur, 9);
                kj_3[0] = _shfl_sync_40;
                unsigned int _shfl_sync_41 = __shfl_sync(0xFFFFFFFF, v_cur, 11);
                kj_3[1] = _shfl_sync_41;
                unsigned int _shfl_sync_42 = __shfl_sync(0xFFFFFFFF, v_cur, 13);
                kj_3[2] = _shfl_sync_42;
                unsigned int _shfl_sync_43 = __shfl_sync(0xFFFFFFFF, v_cur, 15);
                kj_3[3] = _shfl_sync_43;
                uint32_t lo_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_6) : "r"(kj_3[0]), "r"(kj_3[1]), "r"(v_sel[0]));
                uint32_t hi_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_6) : "r"(kj_3[2]), "r"(kj_3[3]), "r"(v_sel[0]));
                uint32_t _prmt_44;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_44) : "r"(lo_6), "r"(hi_6));
                vw[6] = _prmt_44;
                uint32_t lo_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_7) : "r"(kj_3[0]), "r"(kj_3[1]), "r"(v_sel[1]));
                uint32_t hi_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_7) : "r"(kj_3[2]), "r"(kj_3[3]), "r"(v_sel[1]));
                uint32_t _prmt_45;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_45) : "r"(lo_7), "r"(hi_7));
                vw[7] = _prmt_45;
                unsigned int kj_4[4];
                unsigned int _shfl_sync_46 = __shfl_sync(0xFFFFFFFF, v_cur, 16);
                kj_4[0] = _shfl_sync_46;
                unsigned int _shfl_sync_47 = __shfl_sync(0xFFFFFFFF, v_cur, 18);
                kj_4[1] = _shfl_sync_47;
                unsigned int _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, v_cur, 20);
                kj_4[2] = _shfl_sync_48;
                unsigned int _shfl_sync_49 = __shfl_sync(0xFFFFFFFF, v_cur, 22);
                kj_4[3] = _shfl_sync_49;
                uint32_t lo_8;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_8) : "r"(kj_4[0]), "r"(kj_4[1]), "r"(v_sel[0]));
                uint32_t hi_8;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_8) : "r"(kj_4[2]), "r"(kj_4[3]), "r"(v_sel[0]));
                uint32_t _prmt_50;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_50) : "r"(lo_8), "r"(hi_8));
                vw[8] = _prmt_50;
                uint32_t lo_9;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_9) : "r"(kj_4[0]), "r"(kj_4[1]), "r"(v_sel[1]));
                uint32_t hi_9;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_9) : "r"(kj_4[2]), "r"(kj_4[3]), "r"(v_sel[1]));
                uint32_t _prmt_51;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_51) : "r"(lo_9), "r"(hi_9));
                vw[9] = _prmt_51;
                unsigned int kj_5[4];
                unsigned int _shfl_sync_52 = __shfl_sync(0xFFFFFFFF, v_cur, 17);
                kj_5[0] = _shfl_sync_52;
                unsigned int _shfl_sync_53 = __shfl_sync(0xFFFFFFFF, v_cur, 19);
                kj_5[1] = _shfl_sync_53;
                unsigned int _shfl_sync_54 = __shfl_sync(0xFFFFFFFF, v_cur, 21);
                kj_5[2] = _shfl_sync_54;
                unsigned int _shfl_sync_55 = __shfl_sync(0xFFFFFFFF, v_cur, 23);
                kj_5[3] = _shfl_sync_55;
                uint32_t lo_10;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_10) : "r"(kj_5[0]), "r"(kj_5[1]), "r"(v_sel[0]));
                uint32_t hi_10;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_10) : "r"(kj_5[2]), "r"(kj_5[3]), "r"(v_sel[0]));
                uint32_t _prmt_56;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_56) : "r"(lo_10), "r"(hi_10));
                vw[10] = _prmt_56;
                uint32_t lo_11;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_11) : "r"(kj_5[0]), "r"(kj_5[1]), "r"(v_sel[1]));
                uint32_t hi_11;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_11) : "r"(kj_5[2]), "r"(kj_5[3]), "r"(v_sel[1]));
                uint32_t _prmt_57;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_57) : "r"(lo_11), "r"(hi_11));
                vw[11] = _prmt_57;
                unsigned int kj_6[4];
                unsigned int _shfl_sync_58 = __shfl_sync(0xFFFFFFFF, v_cur, 24);
                kj_6[0] = _shfl_sync_58;
                unsigned int _shfl_sync_59 = __shfl_sync(0xFFFFFFFF, v_cur, 26);
                kj_6[1] = _shfl_sync_59;
                unsigned int _shfl_sync_60 = __shfl_sync(0xFFFFFFFF, v_cur, 28);
                kj_6[2] = _shfl_sync_60;
                unsigned int _shfl_sync_61 = __shfl_sync(0xFFFFFFFF, v_cur, 30);
                kj_6[3] = _shfl_sync_61;
                uint32_t lo_12;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_12) : "r"(kj_6[0]), "r"(kj_6[1]), "r"(v_sel[0]));
                uint32_t hi_12;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_12) : "r"(kj_6[2]), "r"(kj_6[3]), "r"(v_sel[0]));
                uint32_t _prmt_62;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_62) : "r"(lo_12), "r"(hi_12));
                vw[12] = _prmt_62;
                uint32_t lo_13;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_13) : "r"(kj_6[0]), "r"(kj_6[1]), "r"(v_sel[1]));
                uint32_t hi_13;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_13) : "r"(kj_6[2]), "r"(kj_6[3]), "r"(v_sel[1]));
                uint32_t _prmt_63;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_63) : "r"(lo_13), "r"(hi_13));
                vw[13] = _prmt_63;
                unsigned int kj_7[4];
                unsigned int _shfl_sync_64 = __shfl_sync(0xFFFFFFFF, v_cur, 25);
                kj_7[0] = _shfl_sync_64;
                unsigned int _shfl_sync_65 = __shfl_sync(0xFFFFFFFF, v_cur, 27);
                kj_7[1] = _shfl_sync_65;
                unsigned int _shfl_sync_66 = __shfl_sync(0xFFFFFFFF, v_cur, 29);
                kj_7[2] = _shfl_sync_66;
                unsigned int _shfl_sync_67 = __shfl_sync(0xFFFFFFFF, v_cur, 31);
                kj_7[3] = _shfl_sync_67;
                uint32_t lo_14;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_14) : "r"(kj_7[0]), "r"(kj_7[1]), "r"(v_sel[0]));
                uint32_t hi_14;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_14) : "r"(kj_7[2]), "r"(kj_7[3]), "r"(v_sel[0]));
                uint32_t _prmt_68;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_68) : "r"(lo_14), "r"(hi_14));
                vw[14] = _prmt_68;
                uint32_t lo_15;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo_15) : "r"(kj_7[0]), "r"(kj_7[1]), "r"(v_sel[1]));
                uint32_t hi_15;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi_15) : "r"(kj_7[2]), "r"(kj_7[3]), "r"(v_sel[1]));
                uint32_t _prmt_69;
                asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_69) : "r"(lo_15), "r"(hi_15));
                vw[15] = _prmt_69;
                if (page_keys < 128) {
                    unsigned int live_mask_0 = 0;
                    if (0 < page_keys) {
                        live_mask_0 = live_mask_0 | 255;
                    }
                    if (16 < page_keys) {
                        live_mask_0 = live_mask_0 | 65280;
                    }
                    if (32 < page_keys) {
                        live_mask_0 = live_mask_0 | 16711680;
                    }
                    if (48 < page_keys) {
                        live_mask_0 = live_mask_0 | 4278190080u;
                    }
                    vw[0] = vw[0] & live_mask_0;
                    vw[1] = vw[1] & live_mask_0;
                    vw[2] = vw[2] & live_mask_0;
                    vw[3] = vw[3] & live_mask_0;
                    vw[8] = vw[8] & live_mask_0;
                    vw[9] = vw[9] & live_mask_0;
                    vw[10] = vw[10] & live_mask_0;
                    vw[11] = vw[11] & live_mask_0;
                    unsigned int live_mask_1 = 0;
                    if (64 < page_keys) {
                        live_mask_1 = live_mask_1 | 255;
                    }
                    if (80 < page_keys) {
                        live_mask_1 = live_mask_1 | 65280;
                    }
                    if (96 < page_keys) {
                        live_mask_1 = live_mask_1 | 16711680;
                    }
                    if (112 < page_keys) {
                        live_mask_1 = live_mask_1 | 4278190080u;
                    }
                    vw[4] = vw[4] & live_mask_1;
                    vw[5] = vw[5] & live_mask_1;
                    vw[6] = vw[6] & live_mask_1;
                    vw[7] = vw[7] & live_mask_1;
                    vw[12] = vw[12] & live_mask_1;
                    vw[13] = vw[13] & live_mask_1;
                    vw[14] = vw[14] & live_mask_1;
                    vw[15] = vw[15] & live_mask_1;
                }
                mbarrier_wait(pv_done_addr, _phase_pv_done_0_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 468), "r"(vw[0]), "r"(vw[1]), "r"(vw[2]), "r"(vw[3]), "r"(vw[4]), "r"(vw[5]), "r"(vw[6]), "r"(vw[7]), "r"(vw[8]), "r"(vw[9]), "r"(vw[10]), "r"(vw[11]), "r"(vw[12]), "r"(vw[13]), "r"(vw[14]), "r"(vw[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sfv_full_addr), "r"(0) : "memory");
                _phase_pv_done_0_1 ^= 1;
            }
            mbarrier_wait(sfk_empty_addr, _phase_sfk_empty_0);
            mbarrier_wait(pv_done_addr, _phase_pv_done_0_1);
        }
    }
    // ---- Role: xpose ----
    if (warp >= 8 && warp <= 11) {
        { // xpose_main
            int split_i_2 = blockIdx.x / 2;
            int b_i_2 = blockIdx.z;
            int rows_b_2 = q_len * num_heads;
            int row0_2 = blockIdx.y * 128;
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
            int xw = make_warp_uniform(warp - 8);
            int src_row = (lane & 1) + (lane >> 1 & 1) * 8 + (lane >> 2 & 3) * 2 + (lane >> 4) * 16;
            int raw_lane = src_row * 64 + (xw ^ src_row >> 1 & 3) * 16;
            int src_lane = (lane & 3) << 2 | lane >> 2 & 3;
            int dim_row = xw * 32 + lane;
            unsigned int x_sel = ((lane >= 16) ? 30258 : 21520);
            int raw_x = 0;
            int vt_x = 0;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_vt_empty = 1;
            #pragma unroll 1
            for (int j_2 = 0; j_2 < my_n_2; j_2++) {
                mbarrier_wait(raw_full_addr + (raw_x) * 8, _phase_raw_full);
                mbarrier_wait(vt_empty_addr + (vt_x) * 8, _phase_vt_empty);
                unsigned int lw[32];
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[0]), "=r"(lw[1]), "=r"(lw[2]), "=r"(lw[3])
                    : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[4]), "=r"(lw[5]), "=r"(lw[6]), "=r"(lw[7])
                    : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + 2048 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[8]), "=r"(lw[9]), "=r"(lw[10]), "=r"(lw[11])
                    : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + 4096 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[12]), "=r"(lw[13]), "=r"(lw[14]), "=r"(lw[15])
                    : "r"(smem_raw_addr + (unsigned int)(raw_x * 2 * 8192 + 6144 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[16]), "=r"(lw[17]), "=r"(lw[18]), "=r"(lw[19])
                    : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[20]), "=r"(lw[21]), "=r"(lw[22]), "=r"(lw[23])
                    : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + 2048 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[24]), "=r"(lw[25]), "=r"(lw[26]), "=r"(lw[27])
                    : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + 4096 + raw_lane))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(lw[28]), "=r"(lw[29]), "=r"(lw[30]), "=r"(lw[31])
                    : "r"(smem_raw_addr + (unsigned int)((raw_x * 2 + 1) * 8192 + 6144 + raw_lane))
                    : "memory");
                unsigned int sw_0[8];
                unsigned int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, lw[0], src_lane);
                sw_0[0] = _shfl_sync_0;
                unsigned int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, lw[0], src_lane + 16);
                sw_0[4] = _shfl_sync_1;
                unsigned int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, lw[1], src_lane);
                sw_0[1] = _shfl_sync_2;
                unsigned int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, lw[1], src_lane + 16);
                sw_0[5] = _shfl_sync_3;
                unsigned int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, lw[2], src_lane);
                sw_0[2] = _shfl_sync_4;
                unsigned int _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, lw[2], src_lane + 16);
                sw_0[6] = _shfl_sync_5;
                unsigned int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, lw[3], src_lane);
                sw_0[3] = _shfl_sync_6;
                unsigned int _shfl_sync_7 = __shfl_sync(0xFFFFFFFF, lw[3], src_lane + 16);
                sw_0[7] = _shfl_sync_7;
                uint32_t first0_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_0) : "r"(sw_0[0]), "r"(sw_0[2]), "r"(x_sel));
                uint32_t first1_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_0) : "r"(sw_0[1]), "r"(sw_0[3]), "r"(x_sel));
                uint32_t second0_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_0) : "r"(sw_0[4]), "r"(sw_0[6]), "r"(x_sel));
                uint32_t second1_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_0) : "r"(sw_0[5]), "r"(sw_0[7]), "r"(x_sel));
                unsigned int pw_0[4];
                pw_0[0] = first0_0 & 983055 | first0_0 >> 4 & 15728880 | (first0_0 >> 4 & 983055 | first0_0 >> 8 & 15728880) << 8;
                pw_0[1] = second0_0 & 983055 | second0_0 >> 4 & 15728880 | (second0_0 >> 4 & 983055 | second0_0 >> 8 & 15728880) << 8;
                pw_0[2] = first1_0 & 983055 | first1_0 >> 4 & 15728880 | (first1_0 >> 4 & 983055 | first1_0 >> 8 & 15728880) << 8;
                pw_0[3] = second1_0 & 983055 | second1_0 >> 4 & 15728880 | (second1_0 >> 4 & 983055 | second1_0 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_8 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (0 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_8), "r"(pw_0[0]), "r"(pw_0[1]), "r"(pw_0[2]), "r"(pw_0[3])
                    : "memory");
                unsigned int sw_1[8];
                unsigned int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, lw[4], src_lane);
                sw_1[0] = _shfl_sync_9;
                unsigned int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, lw[4], src_lane + 16);
                sw_1[4] = _shfl_sync_10;
                unsigned int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, lw[5], src_lane);
                sw_1[1] = _shfl_sync_11;
                unsigned int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, lw[5], src_lane + 16);
                sw_1[5] = _shfl_sync_12;
                unsigned int _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, lw[6], src_lane);
                sw_1[2] = _shfl_sync_13;
                unsigned int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, lw[6], src_lane + 16);
                sw_1[6] = _shfl_sync_14;
                unsigned int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, lw[7], src_lane);
                sw_1[3] = _shfl_sync_15;
                unsigned int _shfl_sync_16 = __shfl_sync(0xFFFFFFFF, lw[7], src_lane + 16);
                sw_1[7] = _shfl_sync_16;
                uint32_t first0_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_1) : "r"(sw_1[0]), "r"(sw_1[2]), "r"(x_sel));
                uint32_t first1_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_1) : "r"(sw_1[1]), "r"(sw_1[3]), "r"(x_sel));
                uint32_t second0_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_1) : "r"(sw_1[4]), "r"(sw_1[6]), "r"(x_sel));
                uint32_t second1_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_1) : "r"(sw_1[5]), "r"(sw_1[7]), "r"(x_sel));
                unsigned int pw_1[4];
                pw_1[0] = first0_1 & 983055 | first0_1 >> 4 & 15728880 | (first0_1 >> 4 & 983055 | first0_1 >> 8 & 15728880) << 8;
                pw_1[1] = second0_1 & 983055 | second0_1 >> 4 & 15728880 | (second0_1 >> 4 & 983055 | second0_1 >> 8 & 15728880) << 8;
                pw_1[2] = first1_1 & 983055 | first1_1 >> 4 & 15728880 | (first1_1 >> 4 & 983055 | first1_1 >> 8 & 15728880) << 8;
                pw_1[3] = second1_1 & 983055 | second1_1 >> 4 & 15728880 | (second1_1 >> 4 & 983055 | second1_1 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_17 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (1 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_17), "r"(pw_1[0]), "r"(pw_1[1]), "r"(pw_1[2]), "r"(pw_1[3])
                    : "memory");
                unsigned int sw_2[8];
                unsigned int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, lw[8], src_lane);
                sw_2[0] = _shfl_sync_18;
                unsigned int _shfl_sync_19 = __shfl_sync(0xFFFFFFFF, lw[8], src_lane + 16);
                sw_2[4] = _shfl_sync_19;
                unsigned int _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, lw[9], src_lane);
                sw_2[1] = _shfl_sync_20;
                unsigned int _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, lw[9], src_lane + 16);
                sw_2[5] = _shfl_sync_21;
                unsigned int _shfl_sync_22_1 = __shfl_sync(0xFFFFFFFF, lw[10], src_lane);
                sw_2[2] = _shfl_sync_22_1;
                unsigned int _shfl_sync_23_1 = __shfl_sync(0xFFFFFFFF, lw[10], src_lane + 16);
                sw_2[6] = _shfl_sync_23_1;
                unsigned int _shfl_sync_24_1 = __shfl_sync(0xFFFFFFFF, lw[11], src_lane);
                sw_2[3] = _shfl_sync_24_1;
                unsigned int _shfl_sync_25_1 = __shfl_sync(0xFFFFFFFF, lw[11], src_lane + 16);
                sw_2[7] = _shfl_sync_25_1;
                uint32_t first0_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_2) : "r"(sw_2[0]), "r"(sw_2[2]), "r"(x_sel));
                uint32_t first1_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_2) : "r"(sw_2[1]), "r"(sw_2[3]), "r"(x_sel));
                uint32_t second0_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_2) : "r"(sw_2[4]), "r"(sw_2[6]), "r"(x_sel));
                uint32_t second1_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_2) : "r"(sw_2[5]), "r"(sw_2[7]), "r"(x_sel));
                unsigned int pw_2[4];
                pw_2[0] = first0_2 & 983055 | first0_2 >> 4 & 15728880 | (first0_2 >> 4 & 983055 | first0_2 >> 8 & 15728880) << 8;
                pw_2[1] = second0_2 & 983055 | second0_2 >> 4 & 15728880 | (second0_2 >> 4 & 983055 | second0_2 >> 8 & 15728880) << 8;
                pw_2[2] = first1_2 & 983055 | first1_2 >> 4 & 15728880 | (first1_2 >> 4 & 983055 | first1_2 >> 8 & 15728880) << 8;
                pw_2[3] = second1_2 & 983055 | second1_2 >> 4 & 15728880 | (second1_2 >> 4 & 983055 | second1_2 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_26 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (2 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_26), "r"(pw_2[0]), "r"(pw_2[1]), "r"(pw_2[2]), "r"(pw_2[3])
                    : "memory");
                unsigned int sw_3[8];
                unsigned int _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, lw[12], src_lane);
                sw_3[0] = _shfl_sync_27;
                unsigned int _shfl_sync_28_1 = __shfl_sync(0xFFFFFFFF, lw[12], src_lane + 16);
                sw_3[4] = _shfl_sync_28_1;
                unsigned int _shfl_sync_29_1 = __shfl_sync(0xFFFFFFFF, lw[13], src_lane);
                sw_3[1] = _shfl_sync_29_1;
                unsigned int _shfl_sync_30_1 = __shfl_sync(0xFFFFFFFF, lw[13], src_lane + 16);
                sw_3[5] = _shfl_sync_30_1;
                unsigned int _shfl_sync_31_1 = __shfl_sync(0xFFFFFFFF, lw[14], src_lane);
                sw_3[2] = _shfl_sync_31_1;
                unsigned int _shfl_sync_32 = __shfl_sync(0xFFFFFFFF, lw[14], src_lane + 16);
                sw_3[6] = _shfl_sync_32;
                unsigned int _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, lw[15], src_lane);
                sw_3[3] = _shfl_sync_33;
                unsigned int _shfl_sync_34_1 = __shfl_sync(0xFFFFFFFF, lw[15], src_lane + 16);
                sw_3[7] = _shfl_sync_34_1;
                uint32_t first0_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_3) : "r"(sw_3[0]), "r"(sw_3[2]), "r"(x_sel));
                uint32_t first1_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_3) : "r"(sw_3[1]), "r"(sw_3[3]), "r"(x_sel));
                uint32_t second0_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_3) : "r"(sw_3[4]), "r"(sw_3[6]), "r"(x_sel));
                uint32_t second1_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_3) : "r"(sw_3[5]), "r"(sw_3[7]), "r"(x_sel));
                unsigned int pw_3[4];
                pw_3[0] = first0_3 & 983055 | first0_3 >> 4 & 15728880 | (first0_3 >> 4 & 983055 | first0_3 >> 8 & 15728880) << 8;
                pw_3[1] = second0_3 & 983055 | second0_3 >> 4 & 15728880 | (second0_3 >> 4 & 983055 | second0_3 >> 8 & 15728880) << 8;
                pw_3[2] = first1_3 & 983055 | first1_3 >> 4 & 15728880 | (first1_3 >> 4 & 983055 | first1_3 >> 8 & 15728880) << 8;
                pw_3[3] = second1_3 & 983055 | second1_3 >> 4 & 15728880 | (second1_3 >> 4 & 983055 | second1_3 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_35 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (3 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_35), "r"(pw_3[0]), "r"(pw_3[1]), "r"(pw_3[2]), "r"(pw_3[3])
                    : "memory");
                unsigned int sw_4[8];
                unsigned int _shfl_sync_36_1 = __shfl_sync(0xFFFFFFFF, lw[16], src_lane);
                sw_4[0] = _shfl_sync_36_1;
                unsigned int _shfl_sync_37_1 = __shfl_sync(0xFFFFFFFF, lw[16], src_lane + 16);
                sw_4[4] = _shfl_sync_37_1;
                unsigned int _shfl_sync_38 = __shfl_sync(0xFFFFFFFF, lw[17], src_lane);
                sw_4[1] = _shfl_sync_38;
                unsigned int _shfl_sync_39 = __shfl_sync(0xFFFFFFFF, lw[17], src_lane + 16);
                sw_4[5] = _shfl_sync_39;
                unsigned int _shfl_sync_40_1 = __shfl_sync(0xFFFFFFFF, lw[18], src_lane);
                sw_4[2] = _shfl_sync_40_1;
                unsigned int _shfl_sync_41_1 = __shfl_sync(0xFFFFFFFF, lw[18], src_lane + 16);
                sw_4[6] = _shfl_sync_41_1;
                unsigned int _shfl_sync_42_1 = __shfl_sync(0xFFFFFFFF, lw[19], src_lane);
                sw_4[3] = _shfl_sync_42_1;
                unsigned int _shfl_sync_43_1 = __shfl_sync(0xFFFFFFFF, lw[19], src_lane + 16);
                sw_4[7] = _shfl_sync_43_1;
                uint32_t first0_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_4) : "r"(sw_4[0]), "r"(sw_4[2]), "r"(x_sel));
                uint32_t first1_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_4) : "r"(sw_4[1]), "r"(sw_4[3]), "r"(x_sel));
                uint32_t second0_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_4) : "r"(sw_4[4]), "r"(sw_4[6]), "r"(x_sel));
                uint32_t second1_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_4) : "r"(sw_4[5]), "r"(sw_4[7]), "r"(x_sel));
                unsigned int pw_4[4];
                pw_4[0] = first0_4 & 983055 | first0_4 >> 4 & 15728880 | (first0_4 >> 4 & 983055 | first0_4 >> 8 & 15728880) << 8;
                pw_4[1] = second0_4 & 983055 | second0_4 >> 4 & 15728880 | (second0_4 >> 4 & 983055 | second0_4 >> 8 & 15728880) << 8;
                pw_4[2] = first1_4 & 983055 | first1_4 >> 4 & 15728880 | (first1_4 >> 4 & 983055 | first1_4 >> 8 & 15728880) << 8;
                pw_4[3] = second1_4 & 983055 | second1_4 >> 4 & 15728880 | (second1_4 >> 4 & 983055 | second1_4 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_44 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (0 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_44), "r"(pw_4[0]), "r"(pw_4[1]), "r"(pw_4[2]), "r"(pw_4[3])
                    : "memory");
                unsigned int sw_5[8];
                unsigned int _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, lw[20], src_lane);
                sw_5[0] = _shfl_sync_45;
                unsigned int _shfl_sync_46_1 = __shfl_sync(0xFFFFFFFF, lw[20], src_lane + 16);
                sw_5[4] = _shfl_sync_46_1;
                unsigned int _shfl_sync_47_1 = __shfl_sync(0xFFFFFFFF, lw[21], src_lane);
                sw_5[1] = _shfl_sync_47_1;
                unsigned int _shfl_sync_48_1 = __shfl_sync(0xFFFFFFFF, lw[21], src_lane + 16);
                sw_5[5] = _shfl_sync_48_1;
                unsigned int _shfl_sync_49_1 = __shfl_sync(0xFFFFFFFF, lw[22], src_lane);
                sw_5[2] = _shfl_sync_49_1;
                unsigned int _shfl_sync_50 = __shfl_sync(0xFFFFFFFF, lw[22], src_lane + 16);
                sw_5[6] = _shfl_sync_50;
                unsigned int _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, lw[23], src_lane);
                sw_5[3] = _shfl_sync_51;
                unsigned int _shfl_sync_52_1 = __shfl_sync(0xFFFFFFFF, lw[23], src_lane + 16);
                sw_5[7] = _shfl_sync_52_1;
                uint32_t first0_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_5) : "r"(sw_5[0]), "r"(sw_5[2]), "r"(x_sel));
                uint32_t first1_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_5) : "r"(sw_5[1]), "r"(sw_5[3]), "r"(x_sel));
                uint32_t second0_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_5) : "r"(sw_5[4]), "r"(sw_5[6]), "r"(x_sel));
                uint32_t second1_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_5) : "r"(sw_5[5]), "r"(sw_5[7]), "r"(x_sel));
                unsigned int pw_5[4];
                pw_5[0] = first0_5 & 983055 | first0_5 >> 4 & 15728880 | (first0_5 >> 4 & 983055 | first0_5 >> 8 & 15728880) << 8;
                pw_5[1] = second0_5 & 983055 | second0_5 >> 4 & 15728880 | (second0_5 >> 4 & 983055 | second0_5 >> 8 & 15728880) << 8;
                pw_5[2] = first1_5 & 983055 | first1_5 >> 4 & 15728880 | (first1_5 >> 4 & 983055 | first1_5 >> 8 & 15728880) << 8;
                pw_5[3] = second1_5 & 983055 | second1_5 >> 4 & 15728880 | (second1_5 >> 4 & 983055 | second1_5 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_53 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (1 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_53), "r"(pw_5[0]), "r"(pw_5[1]), "r"(pw_5[2]), "r"(pw_5[3])
                    : "memory");
                unsigned int sw_6[8];
                unsigned int _shfl_sync_54_1 = __shfl_sync(0xFFFFFFFF, lw[24], src_lane);
                sw_6[0] = _shfl_sync_54_1;
                unsigned int _shfl_sync_55_1 = __shfl_sync(0xFFFFFFFF, lw[24], src_lane + 16);
                sw_6[4] = _shfl_sync_55_1;
                unsigned int _shfl_sync_56 = __shfl_sync(0xFFFFFFFF, lw[25], src_lane);
                sw_6[1] = _shfl_sync_56;
                unsigned int _shfl_sync_57 = __shfl_sync(0xFFFFFFFF, lw[25], src_lane + 16);
                sw_6[5] = _shfl_sync_57;
                unsigned int _shfl_sync_58_1 = __shfl_sync(0xFFFFFFFF, lw[26], src_lane);
                sw_6[2] = _shfl_sync_58_1;
                unsigned int _shfl_sync_59_1 = __shfl_sync(0xFFFFFFFF, lw[26], src_lane + 16);
                sw_6[6] = _shfl_sync_59_1;
                unsigned int _shfl_sync_60_1 = __shfl_sync(0xFFFFFFFF, lw[27], src_lane);
                sw_6[3] = _shfl_sync_60_1;
                unsigned int _shfl_sync_61_1 = __shfl_sync(0xFFFFFFFF, lw[27], src_lane + 16);
                sw_6[7] = _shfl_sync_61_1;
                uint32_t first0_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_6) : "r"(sw_6[0]), "r"(sw_6[2]), "r"(x_sel));
                uint32_t first1_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_6) : "r"(sw_6[1]), "r"(sw_6[3]), "r"(x_sel));
                uint32_t second0_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_6) : "r"(sw_6[4]), "r"(sw_6[6]), "r"(x_sel));
                uint32_t second1_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_6) : "r"(sw_6[5]), "r"(sw_6[7]), "r"(x_sel));
                unsigned int pw_6[4];
                pw_6[0] = first0_6 & 983055 | first0_6 >> 4 & 15728880 | (first0_6 >> 4 & 983055 | first0_6 >> 8 & 15728880) << 8;
                pw_6[1] = second0_6 & 983055 | second0_6 >> 4 & 15728880 | (second0_6 >> 4 & 983055 | second0_6 >> 8 & 15728880) << 8;
                pw_6[2] = first1_6 & 983055 | first1_6 >> 4 & 15728880 | (first1_6 >> 4 & 983055 | first1_6 >> 8 & 15728880) << 8;
                pw_6[3] = second1_6 & 983055 | second1_6 >> 4 & 15728880 | (second1_6 >> 4 & 983055 | second1_6 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_62 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (2 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_62), "r"(pw_6[0]), "r"(pw_6[1]), "r"(pw_6[2]), "r"(pw_6[3])
                    : "memory");
                unsigned int sw_7[8];
                unsigned int _shfl_sync_63 = __shfl_sync(0xFFFFFFFF, lw[28], src_lane);
                sw_7[0] = _shfl_sync_63;
                unsigned int _shfl_sync_64_1 = __shfl_sync(0xFFFFFFFF, lw[28], src_lane + 16);
                sw_7[4] = _shfl_sync_64_1;
                unsigned int _shfl_sync_65_1 = __shfl_sync(0xFFFFFFFF, lw[29], src_lane);
                sw_7[1] = _shfl_sync_65_1;
                unsigned int _shfl_sync_66_1 = __shfl_sync(0xFFFFFFFF, lw[29], src_lane + 16);
                sw_7[5] = _shfl_sync_66_1;
                unsigned int _shfl_sync_67_1 = __shfl_sync(0xFFFFFFFF, lw[30], src_lane);
                sw_7[2] = _shfl_sync_67_1;
                unsigned int _shfl_sync_68 = __shfl_sync(0xFFFFFFFF, lw[30], src_lane + 16);
                sw_7[6] = _shfl_sync_68;
                unsigned int _shfl_sync_69 = __shfl_sync(0xFFFFFFFF, lw[31], src_lane);
                sw_7[3] = _shfl_sync_69;
                unsigned int _shfl_sync_70 = __shfl_sync(0xFFFFFFFF, lw[31], src_lane + 16);
                sw_7[7] = _shfl_sync_70;
                uint32_t first0_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first0_7) : "r"(sw_7[0]), "r"(sw_7[2]), "r"(x_sel));
                uint32_t first1_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(first1_7) : "r"(sw_7[1]), "r"(sw_7[3]), "r"(x_sel));
                uint32_t second0_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second0_7) : "r"(sw_7[4]), "r"(sw_7[6]), "r"(x_sel));
                uint32_t second1_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(second1_7) : "r"(sw_7[5]), "r"(sw_7[7]), "r"(x_sel));
                unsigned int pw_7[4];
                pw_7[0] = first0_7 & 983055 | first0_7 >> 4 & 15728880 | (first0_7 >> 4 & 983055 | first0_7 >> 8 & 15728880) << 8;
                pw_7[1] = second0_7 & 983055 | second0_7 >> 4 & 15728880 | (second0_7 >> 4 & 983055 | second0_7 >> 8 & 15728880) << 8;
                pw_7[2] = first1_7 & 983055 | first1_7 >> 4 & 15728880 | (first1_7 >> 4 & 983055 | first1_7 >> 8 & 15728880) << 8;
                pw_7[3] = second1_7 & 983055 | second1_7 >> 4 & 15728880 | (second1_7 >> 4 & 983055 | second1_7 >> 8 & 15728880) << 8;
                uint32_t _stmatrix_b8_addr_71 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (3 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_71), "r"(pw_7[0]), "r"(pw_7[1]), "r"(pw_7[2]), "r"(pw_7[3])
                    : "memory");
                mbarrier_arrive(raw_empty_addr + (raw_x) * 8);
                raw_x += 1;
                if (raw_x == 2) { raw_x = 0; _phase_raw_full ^= 1; }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
            int split_i_3 = blockIdx.x / 2;
            int b_i_3 = blockIdx.z;
            int rows_b_3 = q_len * num_heads;
            int row0_3 = blockIdx.y * 128;
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
            int k_l = 0;
            int raw_l = 0;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_raw_empty = 1;
            #pragma unroll 1
            for (int j_3 = 0; j_3 < my_n_3; j_3++) {
                int pid_2 = page_table[b_i_3 * page_stride + (my_start_3 + j_3)];
                int phys_2 = ((pid_2 < 0) ? 0 : pid_2) + page_offset;
                int page_l = phys_2;
                mbarrier_wait(k_empty_addr + (k_l) * 8, _phase_k_empty);
                if (elect_sync()) {
                    tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)(k_l * 5 * 4096), (&kv_k), 0, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 1) * 4096), (&kv_k), 64, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 2) * 4096), (&kv_k), 128, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 3) * 4096), (&kv_k), 192, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_k_addr + (unsigned int)((k_l * 5 + 4) * 4096), (&kv_k), 256, cta_rank * 64, page_l, ((k_full_addr + (k_l) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((k_full_addr + (k_l) * 8) & 0xFEFFFFFF), "r"((uint32_t)(20480)) : "memory");
                }
                k_l += 1;
                if (k_l == 2) { k_l = 0; _phase_k_empty ^= 1; }
                mbarrier_wait(raw_empty_addr + (raw_l) * 8, _phase_raw_empty);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(raw_full_addr + (raw_l) * 8, 16384);
                    tma_3d_gmem2smem(smem_raw_addr + (unsigned int)(raw_l * 2 * 8192), (&kv_v), 64 * cta_rank, 0, page_l, raw_full_addr + (raw_l) * 8);
                    tma_3d_gmem2smem(smem_raw_addr + (unsigned int)((raw_l * 2 + 1) * 8192), (&kv_v), 64 * (2 + cta_rank), 0, page_l, raw_full_addr + (raw_l) * 8);
                }
                raw_l += 1;
                if (raw_l == 2) { raw_l = 0; _phase_raw_empty ^= 1; }
            }
            mbarrier_wait(k_empty_addr + (k_l) * 8, _phase_k_empty);
            k_l += 1;
            if (k_l == 2) { k_l = 0; _phase_k_empty ^= 1; }
            mbarrier_wait(k_empty_addr + (k_l) * 8, _phase_k_empty);
            k_l += 1;
            if (k_l == 2) { k_l = 0; _phase_k_empty ^= 1; }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_sfk_full_0 = 0;
            unsigned int _phase_s_empty_0 = 1;
            unsigned int _phase_vt_full = 0;
            unsigned int _phase_sfv_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            if (cta_rank == 0) {
                int split_i_4 = blockIdx.x / 2;
                int b_i_4 = blockIdx.z;
                int rows_b_4 = q_len * num_heads;
                int row0_4 = blockIdx.y * 128;
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
                int k_m = 0;
                int vt_m = 0;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                #pragma unroll 1
                for (int j_4 = 0; j_4 < my_n_4; j_4++) {
                    mbarrier_wait_cluster_hint(k_full_addr + (k_m) * 8, _phase_k_full, 10000000);
                    mbarrier_wait(sfk_full_addr, _phase_sfk_full_0);
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_0 = elect_sync();
                    int _smem_desc_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_1);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 0), "r"(tmem_sfb + 0), "r"(0), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_2 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_2);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 4), "r"(tmem_sfb + 2), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_3 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 1) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_3);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 8), "r"(tmem_sfb + 4), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_4 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 1) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_4);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 24), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 12), "r"(tmem_sfb + 6), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_5 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 2) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_5);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 16), "r"(tmem_sfb + 8), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_6 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 2) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_6);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 40), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 20), "r"(tmem_sfb + 10), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_7 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 3) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_7);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 24), "r"(tmem_sfb + 12), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_8 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 3) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_8);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 56), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 28), "r"(tmem_sfb + 14), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_9 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_9);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 64), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 32), "r"(tmem_sfb + 16), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_10 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_10);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 72), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 36), "r"(tmem_sfb + 16), "r"(1), "r"(_elected_0) : "memory");
                    }
                    int _smem_desc_lo_11 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_11);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 64), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 32), "r"(tmem_sfb + 18), "r"(1), "r"(_elected_0) : "memory");
                    }
                    tcgen05_commit_cg2_multicast_leader(s_full_addr, (uint16_t)(3), _elected_0);
                    tcgen05_commit_cg2_multicast_leader(sfk_empty_addr, (uint16_t)(3), _elected_0);
                    tcgen05_commit_cg2_multicast_leader(k_empty_addr + (k_m) * 8, (uint16_t)(3), _elected_0);
                    k_m += 1;
                    if (k_m == 2) { k_m = 0; _phase_k_full ^= 1; }
                    _phase_sfk_full_0 ^= 1;
                    _phase_s_empty_0 ^= 1;
                    if (j_4 >= 1) {
                        mbarrier_wait(vt_full_addr + (vt_m) * 8, _phase_vt_full);
                        mbarrier_wait(sfv_full_addr, _phase_sfv_full_0);
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_12 = elect_sync();
                        int _smem_desc_lo_13 = ((smem_p_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_14 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_14) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_acc, a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + 0 + 0, tmem_sfbv + 0 + 0, (((int)(j_4 == 1)) ? 0 : 1), _elected_12);
                        }
                        int _smem_desc_lo_15 = ((smem_p_addr + 32) >> 4) & 0x3FFF;
                        int _smem_desc_lo_16 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (vt_m * 2) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_acc, a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + 4 + 0, tmem_sfbv + 4 + 0, 1, _elected_12);
                        }
                        int _smem_desc_lo_17 = ((smem_p_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_18 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_17) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_18) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader((tmem_acc + (128)), a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + 0 + 0, tmem_sfbv + 8 + 0, (((int)(j_4 == 1)) ? 0 : 1), _elected_12);
                        }
                        int _smem_desc_lo_19 = ((smem_p_addr + 32) >> 4) & 0x3FFF;
                        int _smem_desc_lo_20 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader((tmem_acc + (128)), a_desc + 0, b_desc + 0,
                                0x8400480U, tmem_sfap + 4 + 0, tmem_sfbv + 12 + 0, 1, _elected_12);
                        }
                        tcgen05_commit_cg2_multicast_leader(pv_done_addr, (uint16_t)(3), _elected_12);
                        tcgen05_commit_cg2_multicast_leader(vt_empty_addr + (vt_m) * 8, (uint16_t)(3), _elected_12);
                        vt_m += 1;
                        if (vt_m == 2) { vt_m = 0; _phase_vt_full ^= 1; }
                        _phase_sfv_full_0 ^= 1;
                        _phase_p_full_0 ^= 1;
                    }
                }
                mbarrier_wait(vt_full_addr + (vt_m) * 8, _phase_vt_full);
                mbarrier_wait(sfv_full_addr, _phase_sfv_full_0);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_21 = elect_sync();
                int _smem_desc_lo_22 = ((smem_p_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_23 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2) * 512;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_22) | ((uint64_t)0x80004020 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_23) | ((uint64_t)0x80004020 << 32);

                    tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_acc, a_desc + 0, b_desc + 0,
                        0x8400480U, tmem_sfap + 0 + 0, tmem_sfbv + 0 + 0, (((int)(my_n_4 == 1)) ? 0 : 1), _elected_21);
                }
                int _smem_desc_lo_24 = ((smem_p_addr + 32) >> 4) & 0x3FFF;
                int _smem_desc_lo_25 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (vt_m * 2) * 512;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_24) | ((uint64_t)0x80004020 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25) | ((uint64_t)0x80004020 << 32);

                    tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_acc, a_desc + 0, b_desc + 0,
                        0x8400480U, tmem_sfap + 4 + 0, tmem_sfbv + 4 + 0, 1, _elected_21);
                }
                int _smem_desc_lo_26 = ((smem_p_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_27 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_26) | ((uint64_t)0x80004020 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_27) | ((uint64_t)0x80004020 << 32);

                    tcgen05_mma_mxf4nvf4_bs_cta2_leader((tmem_acc + (128)), a_desc + 0, b_desc + 0,
                        0x8400480U, tmem_sfap + 0 + 0, tmem_sfbv + 8 + 0, (((int)(my_n_4 == 1)) ? 0 : 1), _elected_21);
                }
                int _smem_desc_lo_28 = ((smem_p_addr + 32) >> 4) & 0x3FFF;
                int _smem_desc_lo_29 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_28) | ((uint64_t)0x80004020 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29) | ((uint64_t)0x80004020 << 32);

                    tcgen05_mma_mxf4nvf4_bs_cta2_leader((tmem_acc + (128)), a_desc + 0, b_desc + 0,
                        0x8400480U, tmem_sfap + 4 + 0, tmem_sfbv + 12 + 0, 1, _elected_21);
                }
                tcgen05_commit_cg2_multicast_leader(pv_done_addr, (uint16_t)(3), _elected_21);
                tcgen05_commit_cg2_multicast_leader(vt_empty_addr + (vt_m) * 8, (uint16_t)(3), _elected_21);
                vt_m += 1;
                if (vt_m == 2) { vt_m = 0; _phase_vt_full ^= 1; }
                _phase_sfv_full_0 ^= 1;
                _phase_p_full_0 ^= 1;
                mbarrier_wait(s_empty_addr, _phase_s_empty_0);
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
