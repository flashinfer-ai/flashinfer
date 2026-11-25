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


__device__ __forceinline__ void tma_5d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
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

__global__ __launch_bounds__(480) __cluster_dims__(2,1,1) void
kernel_cake_fp4_mla_decode_6146c060a9da20cc3573(const __grid_constant__ CUtensorMap kv_k, const __grid_constant__ CUtensorMap kv_v, unsigned int* __restrict__ kv_sf, unsigned int* __restrict__ v_pool, unsigned int* __restrict__ q, unsigned int* __restrict__ q_sf, int* __restrict__ page_table, int* __restrict__ kv_lens, float* __restrict__ q_gs, float* __restrict__ kv_gs, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, float sm_scale, int num_heads, int q_len, int num_split, int page_stride, int page_offset, const __grid_constant__ CUtensorMap ksf_map, const __grid_constant__ CUtensorMap vsf_map)
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
    #define s_full_addr (mbar_base + 168)
    #define s_empty_addr (mbar_base + 184)
    #define p_full_addr (mbar_base + 200)
    #define pv_done_addr (mbar_base + 216)
    #define m_ready_addr (mbar_base + 232)
    #define l_ready_addr (mbar_base + 248)

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
    float* xch_m = reinterpret_cast<float*>(smem_raw + 161280);
    const int _xch_m_addr = smem + 161280;
    float* xch_s = reinterpret_cast<float*>(smem_raw + 162304);
    const int _xch_s_addr = smem + 162304;
    unsigned int* xch_w = reinterpret_cast<unsigned int*>(smem_raw + 163328);
    const int _xch_w_addr = smem + 163328;

    // Mbarrier init (33 barriers at smem_raw[0..264))

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
            // --- pipeline 'vs_pipe' ---
            // vs_full: 4 barriers, init_count=2
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // vs_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // q_full: 1 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            mbarrier_init(smem + 208, 256);
            // pv_done: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // m_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            mbarrier_init(smem + 240, 128);
            // l_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 248, 128);
            mbarrier_init(smem + 256, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 0) {
        int _tmem_hold = smem + 264;
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
            if (grp == 1) {
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
                int qc = make_warp_uniform(warp % 4);
                int lane_base_c = qc * 32 << 16;
                int row_c = cta_rank * 64 + qc % 2 * 32 + lane;
                int row_ok = row_c < rows_valid;
                int q_row = ((row_ok != 0) ? row_base + row_c : row_base);
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
            }
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
            int qs = make_warp_uniform(warp % 4);
            int half = qs / 2;
            int lane_base_s = qs * 32 << 16;
            int r = qs % 2 * 32 + lane;
            int row_local = cta_rank * 64 + r;
            int mine = qs * 32 + lane;
            int other = (qs ^ 2) * 32 + lane;
            int xbar = make_warp_uniform(1 + 2 * grp + qs % 2);
            int row_ok_1 = row_local < rows_valid_1;
            int token = (row0_1 + row_local) / num_heads;
            int vis_raw = kv_len_i_1 - (q_len - 1 - token);
            int vis = ((row_ok_1 != 0 && keys_1 > 0) ? vis_raw : 0);
            int dither = vis >= 32768;
            int warp_dither = __any_sync(0xFFFFFFFF, dither != 0);
            float gq = *q_gs;
            float gk = *kv_gs;
            float c_log2 = sm_scale * 1.442695f / (gq * gk);
            float m = -CUDART_INF_F;
            float l_sum = 0.0f;
            float s_vals[64];
            int n_grp = (my_n_1 - grp + 2 - 1) / 2;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_pv_done = 1;
            #pragma unroll 1
            for (int jj = 0; jj < n_grp; jj++) {
                int j = 2 * jj + grp;
                int xs = 2 * grp + jj % 2;
                int tok0 = 128 * (my_start_1 + j) + 64 * half;
                mbarrier_wait(s_full_addr + (grp) * 8, _phase_s_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                float bmax0 = -CUDART_INF_F;
                float bmax1 = -CUDART_INF_F;
                float bmax2 = -CUDART_INF_F;
                float bmax3 = -CUDART_INF_F;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                #endif
#else
#endif
                asm volatile(
#if __CUDA_ARCH__ == 1030
                    "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17];"
                    : "=f"(s_vals[0]), "=f"(s_vals[1]), "=f"(s_vals[2]), "=f"(s_vals[3]), "=f"(s_vals[4]), "=f"(s_vals[5]), "=f"(s_vals[6]), "=f"(s_vals[7]), "=f"(s_vals[8]), "=f"(s_vals[9]), "=f"(s_vals[10]), "=f"(s_vals[11]), "=f"(s_vals[12]), "=f"(s_vals[13]), "=f"(s_vals[14]), "=f"(s_vals[15]), "=f"(bmax0)
#else
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(s_vals[0]), "=f"(s_vals[1]), "=f"(s_vals[2]), "=f"(s_vals[3]), "=f"(s_vals[4]), "=f"(s_vals[5]), "=f"(s_vals[6]), "=f"(s_vals[7]), "=f"(s_vals[8]), "=f"(s_vals[9]), "=f"(s_vals[10]), "=f"(s_vals[11]), "=f"(s_vals[12]), "=f"(s_vals[13]), "=f"(s_vals[14]), "=f"(s_vals[15]), "=f"(s_vals[16]), "=f"(s_vals[17]), "=f"(s_vals[18]), "=f"(s_vals[19]), "=f"(s_vals[20]), "=f"(s_vals[21]), "=f"(s_vals[22]), "=f"(s_vals[23]), "=f"(s_vals[24]), "=f"(s_vals[25]), "=f"(s_vals[26]), "=f"(s_vals[27]), "=f"(s_vals[28]), "=f"(s_vals[29]), "=f"(s_vals[30]), "=f"(s_vals[31])
#endif
                    : "r"(taddr + (unsigned int)lane_base_s + 256));
#if __CUDA_ARCH__ == 1030
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                #endif
#else
#endif
                asm volatile(
#if __CUDA_ARCH__ == 1030
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
#else
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(s_vals[32]), "=f"(s_vals[33]), "=f"(s_vals[34]), "=f"(s_vals[35]), "=f"(s_vals[36]), "=f"(s_vals[37]), "=f"(s_vals[38]), "=f"(s_vals[39]), "=f"(s_vals[40]), "=f"(s_vals[41]), "=f"(s_vals[42]), "=f"(s_vals[43]), "=f"(s_vals[44]), "=f"(s_vals[45]), "=f"(s_vals[46]), "=f"(s_vals[47]), "=f"(s_vals[48]), "=f"(s_vals[49]), "=f"(s_vals[50]), "=f"(s_vals[51]), "=f"(s_vals[52]), "=f"(s_vals[53]), "=f"(s_vals[54]), "=f"(s_vals[55]), "=f"(s_vals[56]), "=f"(s_vals[57]), "=f"(s_vals[58]), "=f"(s_vals[59]), "=f"(s_vals[60]), "=f"(s_vals[61]), "=f"(s_vals[62]), "=f"(s_vals[63])
#endif
                    : "r"(taddr + (unsigned int)lane_base_s + 288));
#if __CUDA_ARCH__ == 1030
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                #endif
                asm volatile(
                    "tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, [%17];"
                    : "=f"(s_vals[48]), "=f"(s_vals[49]), "=f"(s_vals[50]), "=f"(s_vals[51]), "=f"(s_vals[52]), "=f"(s_vals[53]), "=f"(s_vals[54]), "=f"(s_vals[55]), "=f"(s_vals[56]), "=f"(s_vals[57]), "=f"(s_vals[58]), "=f"(s_vals[59]), "=f"(s_vals[60]), "=f"(s_vals[61]), "=f"(s_vals[62]), "=f"(s_vals[63]), "=f"(bmax3)
                    : "r"(taddr + (unsigned int)lane_base_s + 304));
#else
#endif
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
                int live = vis - tok0;
                int _warp_vote_22 = __any_sync(0xFFFFFFFF, live < 64);
                if (_warp_vote_22 != 0) {
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
#if __CUDA_ARCH__ == 1030
                    float _reg_array_reduce_23 = s_vals[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_23 = max_noftz(_reg_array_reduce_23, s_vals[_lr]);
                    }
                    bmax0 = _reg_array_reduce_23;
                    float _reg_array_reduce_24 = s_vals[(16) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_24 = max_noftz(_reg_array_reduce_24, s_vals[(16) + _lr]);
                    }
                    bmax1 = _reg_array_reduce_24;
                    float _reg_array_reduce_25 = s_vals[(32) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_25 = max_noftz(_reg_array_reduce_25, s_vals[(32) + _lr]);
                    }
                    bmax2 = _reg_array_reduce_25;
                    float _reg_array_reduce_26 = s_vals[(48) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_26 = max_noftz(_reg_array_reduce_26, s_vals[(48) + _lr]);
                    }
                    bmax3 = _reg_array_reduce_26;
#else
#endif
                }
                float g[8];
#if __CUDA_ARCH__ == 1030
                g[0] = ((bmax0 > -CUDART_INF_F) ? c_log2 * bmax0 : -CUDART_INF_F);
                g[1] = ((bmax1 > -CUDART_INF_F) ? c_log2 * bmax1 : -CUDART_INF_F);
                g[2] = ((bmax2 > -CUDART_INF_F) ? c_log2 * bmax2 : -CUDART_INF_F);
                g[3] = ((bmax3 > -CUDART_INF_F) ? c_log2 * bmax3 : -CUDART_INF_F);
#else
                float b_max_0 = s_vals[0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_0 = max_noftz(b_max_0, s_vals[_lr]);
                }
                g[0] = ((b_max_0 > -CUDART_INF_F) ? c_log2 * b_max_0 : -CUDART_INF_F);
#endif
                xch_g[(xs * 128 + mine) * 4] = g[0];
#if __CUDA_ARCH__ == 1030
#else
                float b_max_1 = s_vals[(16) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_1 = max_noftz(b_max_1, s_vals[(16) + _lr]);
                }
                g[1] = ((b_max_1 > -CUDART_INF_F) ? c_log2 * b_max_1 : -CUDART_INF_F);
#endif
                xch_g[(xs * 128 + mine) * 4 + 1] = g[1];
#if __CUDA_ARCH__ == 1030
#else
                float b_max_2 = s_vals[(32) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_2 = max_noftz(b_max_2, s_vals[(32) + _lr]);
                }
                g[2] = ((b_max_2 > -CUDART_INF_F) ? c_log2 * b_max_2 : -CUDART_INF_F);
#endif
                xch_g[(xs * 128 + mine) * 4 + 2] = g[2];
#if __CUDA_ARCH__ == 1030
#else
                float b_max_3 = s_vals[(48) + 0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    b_max_3 = max_noftz(b_max_3, s_vals[(48) + _lr]);
                }
                g[3] = ((b_max_3 > -CUDART_INF_F) ? c_log2 * b_max_3 : -CUDART_INF_F);
#endif
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
                if (grp == 0) {
                    if (jj > 0) {
                        int _parity_m_ready = (int)((unsigned int)jj % 2) ^ 1;
                        mbarrier_wait(m_ready_addr + 8, _parity_m_ready);
                        m = xch_m[128 + mine];
                    }
                } else {
                    int _parity_m_ready_1 = (int)((unsigned int)jj % 2);
                    mbarrier_wait(m_ready_addr, _parity_m_ready_1);
                    m = xch_m[mine];
                }
                int move = page_max > -CUDART_INF_F && (m == -CUDART_INF_F || page_max > m + 3.0f);
                int rescale = move != 0 && m > -CUDART_INF_F;
#if __CUDA_ARCH__ == 1030
                float _exp2_27 = approx_exp2(m - page_max);
                float alpha = ((rescale != 0) ? _exp2_27 : 1.0f);
#else
                float _exp2_23 = approx_exp2(m - page_max);
                float alpha = ((rescale != 0) ? _exp2_23 : 1.0f);
#endif
                if (move != 0) {
                    m = page_max;
                }
                xch_m[grp * 128 + mine] = m;
                mbarrier_arrive(m_ready_addr + (grp) * 8);
                unsigned int codes[8];
                unsigned int sf_pair[2];
                float sf_fit[4];
                float b_sums[4];
                float b_weights[4];
                if (warp_dither != 0) {
                    float sf_near[4];
#if __CUDA_ARCH__ == 1030
                    float _exp2_28 = approx_exp2(g[0] - m - 4.0f);
                    sf_near[0] = ((g[0] > -CUDART_INF_F) ? 448.0f * _exp2_28 : 0.0f);
                    float _exp2_29 = approx_exp2(g[1] - m - 4.0f);
                    sf_near[1] = ((g[1] > -CUDART_INF_F) ? 448.0f * _exp2_29 : 0.0f);
                    float _exp2_30 = approx_exp2(g[2] - m - 4.0f);
                    sf_near[2] = ((g[2] > -CUDART_INF_F) ? 448.0f * _exp2_30 : 0.0f);
                    float _exp2_31 = approx_exp2(g[3] - m - 4.0f);
                    sf_near[3] = ((g[3] > -CUDART_INF_F) ? 448.0f * _exp2_31 : 0.0f);
#else
                    float _exp2_24 = approx_exp2(g[0] - m - 4.0f);
                    sf_near[0] = ((g[0] > -CUDART_INF_F) ? 448.0f * _exp2_24 : 0.0f);
                    float _exp2_25 = approx_exp2(g[1] - m - 4.0f);
                    sf_near[1] = ((g[1] > -CUDART_INF_F) ? 448.0f * _exp2_25 : 0.0f);
                    float _exp2_26 = approx_exp2(g[2] - m - 4.0f);
                    sf_near[2] = ((g[2] > -CUDART_INF_F) ? 448.0f * _exp2_26 : 0.0f);
                    float _exp2_27 = approx_exp2(g[3] - m - 4.0f);
                    sf_near[3] = ((g[3] > -CUDART_INF_F) ? 448.0f * _exp2_27 : 0.0f);
#endif
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
#if __CUDA_ARCH__ == 1030
                    uint16_t _e4m3x2_decode_32 = (uint16_t)(near_word[0] >> 0 & 0xFFu);
                    uint32_t _f16x2_decode_32;
#else
                    uint16_t _e4m3x2_decode_28 = (uint16_t)(near_word[0] >> 0 & 0xFFu);
                    uint32_t _f16x2_decode_28;
#endif
                    float sf_near_r_0;
#if __CUDA_ARCH__ == 1030
#else
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_28) : "h"(_e4m3x2_decode_28));
                    uint16_t _f16_decode_28 = (uint16_t)_f16x2_decode_28;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_0) : "h"(_f16_decode_28));
                    float _rcp_29 = approx_rcp(sf_near_r_0);
                    six_rho_d[0] = ((sf_near_r_0 > 0.0f) ? 6.0f * sf_near[0] * _rcp_29 : 0.0f);
                    uint16_t _e4m3x2_decode_30 = (uint16_t)(near_word[0] >> 8 & 0xFFu);
                    uint32_t _f16x2_decode_30;
                    float sf_near_r_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_30) : "h"(_e4m3x2_decode_30));
                    uint16_t _f16_decode_30 = (uint16_t)_f16x2_decode_30;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_1) : "h"(_f16_decode_30));
                    float _rcp_31 = approx_rcp(sf_near_r_1);
                    six_rho_d[1] = ((sf_near_r_1 > 0.0f) ? 6.0f * sf_near[1] * _rcp_31 : 0.0f);
                    uint16_t _e4m3x2_decode_32 = (uint16_t)(near_word[0] >> 16 & 0xFFu);
                    uint32_t _f16x2_decode_32;
                    float sf_near_r_2;
#endif
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_32) : "h"(_e4m3x2_decode_32));
                    uint16_t _f16_decode_32 = (uint16_t)_f16x2_decode_32;
#if __CUDA_ARCH__ == 1030
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_0) : "h"(_f16_decode_32));
                    float _rcp_33 = approx_rcp(sf_near_r_0);
                    six_rho_d[0] = ((sf_near_r_0 > 0.0f) ? 6.0f * sf_near[0] * _rcp_33 : 0.0f);
                    uint16_t _e4m3x2_decode_34 = (uint16_t)(near_word[0] >> 8 & 0xFFu);
#else
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_2) : "h"(_f16_decode_32));
                    float _rcp_33 = approx_rcp(sf_near_r_2);
                    six_rho_d[2] = ((sf_near_r_2 > 0.0f) ? 6.0f * sf_near[2] * _rcp_33 : 0.0f);
                    uint16_t _e4m3x2_decode_34 = (uint16_t)(near_word[0] >> 24 & 0xFFu);
#endif
                    uint32_t _f16x2_decode_34;
#if __CUDA_ARCH__ == 1030
                    float sf_near_r_1;
#else
                    float sf_near_r_3;
#endif
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_34) : "h"(_e4m3x2_decode_34));
                    uint16_t _f16_decode_34 = (uint16_t)_f16x2_decode_34;
#if __CUDA_ARCH__ == 1030
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_1) : "h"(_f16_decode_34));
                    float _rcp_35 = approx_rcp(sf_near_r_1);
                    six_rho_d[1] = ((sf_near_r_1 > 0.0f) ? 6.0f * sf_near[1] * _rcp_35 : 0.0f);
                    uint16_t _e4m3x2_decode_36 = (uint16_t)(near_word[0] >> 16 & 0xFFu);
                    uint32_t _f16x2_decode_36;
                    float sf_near_r_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_36) : "h"(_e4m3x2_decode_36));
                    uint16_t _f16_decode_36 = (uint16_t)_f16x2_decode_36;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_2) : "h"(_f16_decode_36));
                    float _rcp_37 = approx_rcp(sf_near_r_2);
                    six_rho_d[2] = ((sf_near_r_2 > 0.0f) ? 6.0f * sf_near[2] * _rcp_37 : 0.0f);
                    uint16_t _e4m3x2_decode_38 = (uint16_t)(near_word[0] >> 24 & 0xFFu);
                    uint32_t _f16x2_decode_38;
                    float sf_near_r_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_38) : "h"(_e4m3x2_decode_38));
                    uint16_t _f16_decode_38 = (uint16_t)_f16x2_decode_38;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_3) : "h"(_f16_decode_38));
                    float _rcp_39 = approx_rcp(sf_near_r_3);
                    six_rho_d[3] = ((sf_near_r_3 > 0.0f) ? 6.0f * sf_near[3] * _rcp_39 : 0.0f);
#else
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_near_r_3) : "h"(_f16_decode_34));
                    float _rcp_35 = approx_rcp(sf_near_r_3);
                    six_rho_d[3] = ((sf_near_r_3 > 0.0f) ? 6.0f * sf_near[3] * _rcp_35 : 0.0f);
#endif
                    int g_live = g[0] > -CUDART_INF_F;
                    float g_safe = ((g_live != 0) ? g[0] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_40 = {c_log2, c_log2};
                    const float2 _fma_c2_41 = {-g_safe, -g_safe};
#else
                    const float2 _fma_b2_36 = {c_log2, c_log2};
                    const float2 _fma_c2_37 = {-g_safe, -g_safe};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_40, _fma_c2_41);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_36, _fma_c2_37);
#endif
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
#if __CUDA_ARCH__ == 1030
                    float _exp2_42 = approx_exp2(g_safe - m);
                    b_weights[0] = ((g_live != 0) ? _exp2_42 : 0.0f);
                    float _exp2_43 = approx_exp2(g_safe - m - 4.0f);
                    float sf_ex = 448.0f * _exp2_43;
#else
                    float _exp2_38 = approx_exp2(g_safe - m);
                    b_weights[0] = ((g_live != 0) ? _exp2_38 : 0.0f);
                    float _exp2_39 = approx_exp2(g_safe - m - 4.0f);
                    float sf_ex = 448.0f * _exp2_39;
#endif
                    float e_min = s_vals[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        e_min = fminf(e_min, s_vals[_lr]);
                    }
                    int flat = dither != 0 && e_min >= 0.5f;
                    if (flat != 0) {
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_44 = {six_rho_d[0], six_rho_d[0]};
                        const float2 _fma_c2_45 = {0.0f, 0.0f};
#else
                        const float2 _fma_b2_40 = {six_rho_d[0], six_rho_d[0]};
                        const float2 _fma_c2_41 = {0.0f, 0.0f};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_44, _fma_c2_45);
                        uint16_t _cvt_rs_x4_f32_46;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_46) : "f"(s_vals[3]), "f"(s_vals[2]), "f"(s_vals[1]), "f"(s_vals[0]), "r"(3103267870u));
                        unsigned int w_lo_0 = _cvt_rs_x4_f32_46;
                        uint16_t _cvt_rs_x4_f32_47;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_47) : "f"(s_vals[7]), "f"(s_vals[6]), "f"(s_vals[5]), "f"(s_vals[4]), "r"(2564298778u));
                        unsigned int w_hi_0 = _cvt_rs_x4_f32_47;
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_40, _fma_c2_41);
                        uint16_t _cvt_rs_x4_f32_42;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_42) : "f"(s_vals[3]), "f"(s_vals[2]), "f"(s_vals[1]), "f"(s_vals[0]), "r"(3103267870u));
                        unsigned int w_lo_0 = _cvt_rs_x4_f32_42;
                        uint16_t _cvt_rs_x4_f32_43;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_43) : "f"(s_vals[7]), "f"(s_vals[6]), "f"(s_vals[5]), "f"(s_vals[4]), "r"(2564298778u));
                        unsigned int w_hi_0 = _cvt_rs_x4_f32_43;
#endif
                        codes[0] = w_lo_0 | w_hi_0 << 16;
#if __CUDA_ARCH__ == 1030
                        uint16_t _cvt_rs_x4_f32_48;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_48) : "f"(s_vals[11]), "f"(s_vals[10]), "f"(s_vals[9]), "f"(s_vals[8]), "r"(2833781782u));
                        unsigned int w_lo_1 = _cvt_rs_x4_f32_48;
                        uint16_t _cvt_rs_x4_f32_49;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_49) : "f"(s_vals[15]), "f"(s_vals[14]), "f"(s_vals[13]), "f"(s_vals[12]), "r"(2294812690u));
                        unsigned int w_hi_1 = _cvt_rs_x4_f32_49;
#else
                        uint16_t _cvt_rs_x4_f32_44;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_44) : "f"(s_vals[11]), "f"(s_vals[10]), "f"(s_vals[9]), "f"(s_vals[8]), "r"(2833781782u));
                        unsigned int w_lo_1 = _cvt_rs_x4_f32_44;
                        uint16_t _cvt_rs_x4_f32_45;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_45) : "f"(s_vals[15]), "f"(s_vals[14]), "f"(s_vals[13]), "f"(s_vals[12]), "r"(2294812690u));
                        unsigned int w_hi_1 = _cvt_rs_x4_f32_45;
#endif
                        codes[1] = w_lo_1 | w_hi_1 << 16;
                        sf_fit[0] = sf_near[0];
                    } else {
                        float qv[4];
                        float qq[4];
#if __CUDA_ARCH__ == 1030
                        const float2 _scale2_50 = {6.0f, 6.0f};
#else
                        const float2 _scale2_46 = {6.0f, 6.0f};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_ls], _scale2_50);
#else
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_ls], _scale2_46);
#endif
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(s_vals[0]), "f"(s_vals[1]), "f"(s_vals[2]), "f"(s_vals[3]), "f"(s_vals[4]), "f"(s_vals[5]), "f"(s_vals[6]), "f"(s_vals[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(s_vals[8]), "f"(s_vals[9]), "f"(s_vals[10]), "f"(s_vals[11]), "f"(s_vals[12]), "f"(s_vals[13]), "f"(s_vals[14]), "f"(s_vals[15]));
                        float q_b[16];
                        unsigned int qn[8];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_51;
#else
                        uint32_t _cvt_f16x2_e2m1x2_47;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_51) : "r"((uint32_t)(codes[0] >> 0)));
                        qn[0] = _cvt_f16x2_e2m1x2_51;
#else
                : "=r"(_cvt_f16x2_e2m1x2_47) : "r"((uint32_t)(codes[0] >> 0)));
                        qn[0] = _cvt_f16x2_e2m1x2_47;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_52;
#else
                        uint32_t _cvt_f16x2_e2m1x2_48;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_52) : "r"((uint32_t)(codes[0] >> 8)));
                        qn[1] = _cvt_f16x2_e2m1x2_52;
#else
                : "=r"(_cvt_f16x2_e2m1x2_48) : "r"((uint32_t)(codes[0] >> 8)));
                        qn[1] = _cvt_f16x2_e2m1x2_48;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_53;
#else
                        uint32_t _cvt_f16x2_e2m1x2_49;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_53) : "r"((uint32_t)(codes[0] >> 16)));
                        qn[2] = _cvt_f16x2_e2m1x2_53;
#else
                : "=r"(_cvt_f16x2_e2m1x2_49) : "r"((uint32_t)(codes[0] >> 16)));
                        qn[2] = _cvt_f16x2_e2m1x2_49;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_54;
#else
                        uint32_t _cvt_f16x2_e2m1x2_50;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_54) : "r"((uint32_t)(codes[0] >> 24)));
                        qn[3] = _cvt_f16x2_e2m1x2_54;
#else
                : "=r"(_cvt_f16x2_e2m1x2_50) : "r"((uint32_t)(codes[0] >> 24)));
                        qn[3] = _cvt_f16x2_e2m1x2_50;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_55;
#else
                        uint32_t _cvt_f16x2_e2m1x2_51;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_55) : "r"((uint32_t)(codes[1] >> 0)));
                        qn[4] = _cvt_f16x2_e2m1x2_55;
#else
                : "=r"(_cvt_f16x2_e2m1x2_51) : "r"((uint32_t)(codes[1] >> 0)));
                        qn[4] = _cvt_f16x2_e2m1x2_51;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_56;
#else
                        uint32_t _cvt_f16x2_e2m1x2_52;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_56) : "r"((uint32_t)(codes[1] >> 8)));
                        qn[5] = _cvt_f16x2_e2m1x2_56;
#else
                : "=r"(_cvt_f16x2_e2m1x2_52) : "r"((uint32_t)(codes[1] >> 8)));
                        qn[5] = _cvt_f16x2_e2m1x2_52;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_57;
#else
                        uint32_t _cvt_f16x2_e2m1x2_53;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_57) : "r"((uint32_t)(codes[1] >> 16)));
                        qn[6] = _cvt_f16x2_e2m1x2_57;
#else
                : "=r"(_cvt_f16x2_e2m1x2_53) : "r"((uint32_t)(codes[1] >> 16)));
                        qn[6] = _cvt_f16x2_e2m1x2_53;
#endif
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
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_58;
#else
                        uint32_t _cvt_f16x2_e2m1x2_54;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_58) : "r"((uint32_t)(codes[1] >> 24)));
                        qn[7] = _cvt_f16x2_e2m1x2_58;
#else
                : "=r"(_cvt_f16x2_e2m1x2_54) : "r"((uint32_t)(codes[1] >> 24)));
                        qn[7] = _cvt_f16x2_e2m1x2_54;
#endif
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
#if __CUDA_ARCH__ == 1030
                        float _rcp_59 = approx_rcp(qq[0]);
                        sf_fit[0] = ((g_live != 0 && qq[0] > 0.0f) ? sf_ex * qv[0] * _rcp_59 : 0.0f);
#else
                        float _rcp_55 = approx_rcp(qq[0]);
                        sf_fit[0] = ((g_live != 0 && qq[0] > 0.0f) ? sf_ex * qv[0] * _rcp_55 : 0.0f);
#endif
                    }
                    int g_live_0 = g[1] > -CUDART_INF_F;
                    float g_safe_0 = ((g_live_0 != 0) ? g[1] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_60 = {c_log2, c_log2};
                    const float2 _fma_c2_61 = {-g_safe_0, -g_safe_0};
#else
                    const float2 _fma_b2_56 = {c_log2, c_log2};
                    const float2 _fma_c2_57 = {-g_safe_0, -g_safe_0};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_60, _fma_c2_61);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_56, _fma_c2_57);
#endif
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
#if __CUDA_ARCH__ == 1030
                    float _exp2_62 = approx_exp2(g_safe_0 - m);
                    b_weights[1] = ((g_live_0 != 0) ? _exp2_62 : 0.0f);
                    float _exp2_63 = approx_exp2(g_safe_0 - m - 4.0f);
                    float sf_ex_0 = 448.0f * _exp2_63;
#else
                    float _exp2_58 = approx_exp2(g_safe_0 - m);
                    b_weights[1] = ((g_live_0 != 0) ? _exp2_58 : 0.0f);
                    float _exp2_59 = approx_exp2(g_safe_0 - m - 4.0f);
                    float sf_ex_0 = 448.0f * _exp2_59;
#endif
                    float e_min_0 = s_vals[(16) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        e_min_0 = fminf(e_min_0, s_vals[(16) + _lr]);
                    }
                    int flat_0 = dither != 0 && e_min_0 >= 0.5f;
                    if (flat_0 != 0) {
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_64 = {six_rho_d[1], six_rho_d[1]};
                        const float2 _fma_c2_65 = {0.0f, 0.0f};
#else
                        const float2 _fma_b2_60 = {six_rho_d[1], six_rho_d[1]};
                        const float2 _fma_c2_61 = {0.0f, 0.0f};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_64, _fma_c2_65);
                        uint16_t _cvt_rs_x4_f32_66;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_66) : "f"(s_vals[19]), "f"(s_vals[18]), "f"(s_vals[17]), "f"(s_vals[16]), "r"(3103267870u));
                        unsigned int w_lo_2 = _cvt_rs_x4_f32_66;
                        uint16_t _cvt_rs_x4_f32_67;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_67) : "f"(s_vals[23]), "f"(s_vals[22]), "f"(s_vals[21]), "f"(s_vals[20]), "r"(2564298778u));
                        unsigned int w_hi_2 = _cvt_rs_x4_f32_67;
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_60, _fma_c2_61);
                        uint16_t _cvt_rs_x4_f32_62;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_62) : "f"(s_vals[19]), "f"(s_vals[18]), "f"(s_vals[17]), "f"(s_vals[16]), "r"(3103267870u));
                        unsigned int w_lo_2 = _cvt_rs_x4_f32_62;
                        uint16_t _cvt_rs_x4_f32_63;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_63) : "f"(s_vals[23]), "f"(s_vals[22]), "f"(s_vals[21]), "f"(s_vals[20]), "r"(2564298778u));
                        unsigned int w_hi_2 = _cvt_rs_x4_f32_63;
#endif
                        codes[2] = w_lo_2 | w_hi_2 << 16;
#if __CUDA_ARCH__ == 1030
                        uint16_t _cvt_rs_x4_f32_68;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_68) : "f"(s_vals[27]), "f"(s_vals[26]), "f"(s_vals[25]), "f"(s_vals[24]), "r"(2833781782u));
                        unsigned int w_lo_3 = _cvt_rs_x4_f32_68;
                        uint16_t _cvt_rs_x4_f32_69;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_69) : "f"(s_vals[31]), "f"(s_vals[30]), "f"(s_vals[29]), "f"(s_vals[28]), "r"(2294812690u));
                        unsigned int w_hi_3 = _cvt_rs_x4_f32_69;
#else
                        uint16_t _cvt_rs_x4_f32_64;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_64) : "f"(s_vals[27]), "f"(s_vals[26]), "f"(s_vals[25]), "f"(s_vals[24]), "r"(2833781782u));
                        unsigned int w_lo_3 = _cvt_rs_x4_f32_64;
                        uint16_t _cvt_rs_x4_f32_65;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_65) : "f"(s_vals[31]), "f"(s_vals[30]), "f"(s_vals[29]), "f"(s_vals[28]), "r"(2294812690u));
                        unsigned int w_hi_3 = _cvt_rs_x4_f32_65;
#endif
                        codes[3] = w_lo_3 | w_hi_3 << 16;
                        sf_fit[1] = sf_near[1];
                    } else {
                        float qv_1[4];
                        float qq_1[4];
#if __CUDA_ARCH__ == 1030
                        const float2 _scale2_70 = {6.0f, 6.0f};
#else
                        const float2 _scale2_66 = {6.0f, 6.0f};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_ls], _scale2_70);
#else
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_ls], _scale2_66);
#endif
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[2]) : "f"(s_vals[16]), "f"(s_vals[17]), "f"(s_vals[18]), "f"(s_vals[19]), "f"(s_vals[20]), "f"(s_vals[21]), "f"(s_vals[22]), "f"(s_vals[23]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[3]) : "f"(s_vals[24]), "f"(s_vals[25]), "f"(s_vals[26]), "f"(s_vals[27]), "f"(s_vals[28]), "f"(s_vals[29]), "f"(s_vals[30]), "f"(s_vals[31]));
                        float q_b_1[16];
                        unsigned int qn_1[8];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_71;
#else
                        uint32_t _cvt_f16x2_e2m1x2_67;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_71) : "r"((uint32_t)(codes[2] >> 0)));
                        qn_1[0] = _cvt_f16x2_e2m1x2_71;
#else
                : "=r"(_cvt_f16x2_e2m1x2_67) : "r"((uint32_t)(codes[2] >> 0)));
                        qn_1[0] = _cvt_f16x2_e2m1x2_67;
#endif
                        float qw_8[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_8[0]), "=&f"(qw_8[1]) : "r"(qn_1[0]), "f"(0.0f), "f"(0.0f));
                        q_b_1[0] = qw_8[0];
                        q_b_1[1] = qw_8[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_72;
#else
                        uint32_t _cvt_f16x2_e2m1x2_68;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_72) : "r"((uint32_t)(codes[2] >> 8)));
                        qn_1[1] = _cvt_f16x2_e2m1x2_72;
#else
                : "=r"(_cvt_f16x2_e2m1x2_68) : "r"((uint32_t)(codes[2] >> 8)));
                        qn_1[1] = _cvt_f16x2_e2m1x2_68;
#endif
                        float qw_9[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_9[0]), "=&f"(qw_9[1]) : "r"(qn_1[1]), "f"(0.0f), "f"(0.0f));
                        q_b_1[2] = qw_9[0];
                        q_b_1[3] = qw_9[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_73;
#else
                        uint32_t _cvt_f16x2_e2m1x2_69;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_73) : "r"((uint32_t)(codes[2] >> 16)));
                        qn_1[2] = _cvt_f16x2_e2m1x2_73;
#else
                : "=r"(_cvt_f16x2_e2m1x2_69) : "r"((uint32_t)(codes[2] >> 16)));
                        qn_1[2] = _cvt_f16x2_e2m1x2_69;
#endif
                        float qw_10[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_10[0]), "=&f"(qw_10[1]) : "r"(qn_1[2]), "f"(0.0f), "f"(0.0f));
                        q_b_1[4] = qw_10[0];
                        q_b_1[5] = qw_10[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_74;
#else
                        uint32_t _cvt_f16x2_e2m1x2_70;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_74) : "r"((uint32_t)(codes[2] >> 24)));
                        qn_1[3] = _cvt_f16x2_e2m1x2_74;
#else
                : "=r"(_cvt_f16x2_e2m1x2_70) : "r"((uint32_t)(codes[2] >> 24)));
                        qn_1[3] = _cvt_f16x2_e2m1x2_70;
#endif
                        float qw_11[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_11[0]), "=&f"(qw_11[1]) : "r"(qn_1[3]), "f"(0.0f), "f"(0.0f));
                        q_b_1[6] = qw_11[0];
                        q_b_1[7] = qw_11[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_75;
#else
                        uint32_t _cvt_f16x2_e2m1x2_71;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_75) : "r"((uint32_t)(codes[3] >> 0)));
                        qn_1[4] = _cvt_f16x2_e2m1x2_75;
#else
                : "=r"(_cvt_f16x2_e2m1x2_71) : "r"((uint32_t)(codes[3] >> 0)));
                        qn_1[4] = _cvt_f16x2_e2m1x2_71;
#endif
                        float qw_12[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_12[0]), "=&f"(qw_12[1]) : "r"(qn_1[4]), "f"(0.0f), "f"(0.0f));
                        q_b_1[8] = qw_12[0];
                        q_b_1[9] = qw_12[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_76;
#else
                        uint32_t _cvt_f16x2_e2m1x2_72;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_76) : "r"((uint32_t)(codes[3] >> 8)));
                        qn_1[5] = _cvt_f16x2_e2m1x2_76;
#else
                : "=r"(_cvt_f16x2_e2m1x2_72) : "r"((uint32_t)(codes[3] >> 8)));
                        qn_1[5] = _cvt_f16x2_e2m1x2_72;
#endif
                        float qw_13[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_13[0]), "=&f"(qw_13[1]) : "r"(qn_1[5]), "f"(0.0f), "f"(0.0f));
                        q_b_1[10] = qw_13[0];
                        q_b_1[11] = qw_13[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_77;
#else
                        uint32_t _cvt_f16x2_e2m1x2_73;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_77) : "r"((uint32_t)(codes[3] >> 16)));
                        qn_1[6] = _cvt_f16x2_e2m1x2_77;
#else
                : "=r"(_cvt_f16x2_e2m1x2_73) : "r"((uint32_t)(codes[3] >> 16)));
                        qn_1[6] = _cvt_f16x2_e2m1x2_73;
#endif
                        float qw_14[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_14[0]), "=&f"(qw_14[1]) : "r"(qn_1[6]), "f"(0.0f), "f"(0.0f));
                        q_b_1[12] = qw_14[0];
                        q_b_1[13] = qw_14[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_78;
#else
                        uint32_t _cvt_f16x2_e2m1x2_74;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_78) : "r"((uint32_t)(codes[3] >> 24)));
                        qn_1[7] = _cvt_f16x2_e2m1x2_78;
#else
                : "=r"(_cvt_f16x2_e2m1x2_74) : "r"((uint32_t)(codes[3] >> 24)));
                        qn_1[7] = _cvt_f16x2_e2m1x2_74;
#endif
                        float qw_15[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_15[0]), "=&f"(qw_15[1]) : "r"(qn_1[7]), "f"(0.0f), "f"(0.0f));
                        q_b_1[14] = qw_15[0];
                        q_b_1[15] = qw_15[1];
                        float qe_1[16];
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(qe_1)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_1)[_ls], reinterpret_cast<const float2*>((s_vals + 16))[_ls]);
                        float t8_2[8];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            reinterpret_cast<float2*>(t8_2)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_1 + 0))[_la], reinterpret_cast<const float2*>((qe_1 + 8))[_la]);
                        #pragma unroll
                        for (int _la = 0; _la < 2; _la++)
                            add_f32x2_inplace(&reinterpret_cast<float2*>(t8_2)[_la], reinterpret_cast<const float2*>((t8_2 + 4))[_la]);
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            add_f32x2_inplace(&reinterpret_cast<float2*>(t8_2)[_la], reinterpret_cast<const float2*>((t8_2 + 2))[_la]);
                        qv_1[1] = t8_2[0] + t8_2[1];
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
                        uint32_t v_1;
                        asm("add.f16x2 %0, %1, %2;" : "=r"(v_1) : "r"(a3_1), "r"(b3_1));
                        float w_1[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(w_1[0]), "=&f"(w_1[1]) : "r"(v_1), "f"(0.0f), "f"(0.0f));
                        qq_1[1] = w_1[0] + w_1[1];
#if __CUDA_ARCH__ == 1030
                        float _rcp_79 = approx_rcp(qq_1[1]);
                        sf_fit[1] = ((g_live_0 != 0 && qq_1[1] > 0.0f) ? sf_ex_0 * qv_1[1] * _rcp_79 : 0.0f);
#else
                        float _rcp_75 = approx_rcp(qq_1[1]);
                        sf_fit[1] = ((g_live_0 != 0 && qq_1[1] > 0.0f) ? sf_ex_0 * qv_1[1] * _rcp_75 : 0.0f);
#endif
                    }
                    int g_live_1 = g[2] > -CUDART_INF_F;
                    float g_safe_1 = ((g_live_1 != 0) ? g[2] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_80 = {c_log2, c_log2};
                    const float2 _fma_c2_81 = {-g_safe_1, -g_safe_1};
#else
                    const float2 _fma_b2_76 = {c_log2, c_log2};
                    const float2 _fma_c2_77 = {-g_safe_1, -g_safe_1};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_80, _fma_c2_81);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_76, _fma_c2_77);
#endif
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
#if __CUDA_ARCH__ == 1030
                    float _exp2_82 = approx_exp2(g_safe_1 - m);
                    b_weights[2] = ((g_live_1 != 0) ? _exp2_82 : 0.0f);
                    float _exp2_83 = approx_exp2(g_safe_1 - m - 4.0f);
                    float sf_ex_1 = 448.0f * _exp2_83;
#else
                    float _exp2_78 = approx_exp2(g_safe_1 - m);
                    b_weights[2] = ((g_live_1 != 0) ? _exp2_78 : 0.0f);
                    float _exp2_79 = approx_exp2(g_safe_1 - m - 4.0f);
                    float sf_ex_1 = 448.0f * _exp2_79;
#endif
                    float e_min_1 = s_vals[(32) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        e_min_1 = fminf(e_min_1, s_vals[(32) + _lr]);
                    }
                    int flat_1 = dither != 0 && e_min_1 >= 0.5f;
                    if (flat_1 != 0) {
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_84 = {six_rho_d[2], six_rho_d[2]};
                        const float2 _fma_c2_85 = {0.0f, 0.0f};
#else
                        const float2 _fma_b2_80 = {six_rho_d[2], six_rho_d[2]};
                        const float2 _fma_c2_81 = {0.0f, 0.0f};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_84, _fma_c2_85);
                        uint16_t _cvt_rs_x4_f32_86;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_86) : "f"(s_vals[35]), "f"(s_vals[34]), "f"(s_vals[33]), "f"(s_vals[32]), "r"(3103267870u));
                        unsigned int w_lo_4 = _cvt_rs_x4_f32_86;
                        uint16_t _cvt_rs_x4_f32_87;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_87) : "f"(s_vals[39]), "f"(s_vals[38]), "f"(s_vals[37]), "f"(s_vals[36]), "r"(2564298778u));
                        unsigned int w_hi_4 = _cvt_rs_x4_f32_87;
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_80, _fma_c2_81);
                        uint16_t _cvt_rs_x4_f32_82;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_82) : "f"(s_vals[35]), "f"(s_vals[34]), "f"(s_vals[33]), "f"(s_vals[32]), "r"(3103267870u));
                        unsigned int w_lo_4 = _cvt_rs_x4_f32_82;
                        uint16_t _cvt_rs_x4_f32_83;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_83) : "f"(s_vals[39]), "f"(s_vals[38]), "f"(s_vals[37]), "f"(s_vals[36]), "r"(2564298778u));
                        unsigned int w_hi_4 = _cvt_rs_x4_f32_83;
#endif
                        codes[4] = w_lo_4 | w_hi_4 << 16;
#if __CUDA_ARCH__ == 1030
                        uint16_t _cvt_rs_x4_f32_88;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_88) : "f"(s_vals[43]), "f"(s_vals[42]), "f"(s_vals[41]), "f"(s_vals[40]), "r"(2833781782u));
                        unsigned int w_lo_5 = _cvt_rs_x4_f32_88;
                        uint16_t _cvt_rs_x4_f32_89;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_89) : "f"(s_vals[47]), "f"(s_vals[46]), "f"(s_vals[45]), "f"(s_vals[44]), "r"(2294812690u));
                        unsigned int w_hi_5 = _cvt_rs_x4_f32_89;
#else
                        uint16_t _cvt_rs_x4_f32_84;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_84) : "f"(s_vals[43]), "f"(s_vals[42]), "f"(s_vals[41]), "f"(s_vals[40]), "r"(2833781782u));
                        unsigned int w_lo_5 = _cvt_rs_x4_f32_84;
                        uint16_t _cvt_rs_x4_f32_85;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_85) : "f"(s_vals[47]), "f"(s_vals[46]), "f"(s_vals[45]), "f"(s_vals[44]), "r"(2294812690u));
                        unsigned int w_hi_5 = _cvt_rs_x4_f32_85;
#endif
                        codes[5] = w_lo_5 | w_hi_5 << 16;
                        sf_fit[2] = sf_near[2];
                    } else {
                        float qv_2[4];
                        float qq_2[4];
#if __CUDA_ARCH__ == 1030
                        const float2 _scale2_90 = {6.0f, 6.0f};
#else
                        const float2 _scale2_86 = {6.0f, 6.0f};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_ls], _scale2_90);
#else
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_ls], _scale2_86);
#endif
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[4]) : "f"(s_vals[32]), "f"(s_vals[33]), "f"(s_vals[34]), "f"(s_vals[35]), "f"(s_vals[36]), "f"(s_vals[37]), "f"(s_vals[38]), "f"(s_vals[39]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[5]) : "f"(s_vals[40]), "f"(s_vals[41]), "f"(s_vals[42]), "f"(s_vals[43]), "f"(s_vals[44]), "f"(s_vals[45]), "f"(s_vals[46]), "f"(s_vals[47]));
                        float q_b_2[16];
                        unsigned int qn_2[8];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_91;
#else
                        uint32_t _cvt_f16x2_e2m1x2_87;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_91) : "r"((uint32_t)(codes[4] >> 0)));
                        qn_2[0] = _cvt_f16x2_e2m1x2_91;
#else
                : "=r"(_cvt_f16x2_e2m1x2_87) : "r"((uint32_t)(codes[4] >> 0)));
                        qn_2[0] = _cvt_f16x2_e2m1x2_87;
#endif
                        float qw_16[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_16[0]), "=&f"(qw_16[1]) : "r"(qn_2[0]), "f"(0.0f), "f"(0.0f));
                        q_b_2[0] = qw_16[0];
                        q_b_2[1] = qw_16[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_92;
#else
                        uint32_t _cvt_f16x2_e2m1x2_88;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_92) : "r"((uint32_t)(codes[4] >> 8)));
                        qn_2[1] = _cvt_f16x2_e2m1x2_92;
#else
                : "=r"(_cvt_f16x2_e2m1x2_88) : "r"((uint32_t)(codes[4] >> 8)));
                        qn_2[1] = _cvt_f16x2_e2m1x2_88;
#endif
                        float qw_17[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_17[0]), "=&f"(qw_17[1]) : "r"(qn_2[1]), "f"(0.0f), "f"(0.0f));
                        q_b_2[2] = qw_17[0];
                        q_b_2[3] = qw_17[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_93;
#else
                        uint32_t _cvt_f16x2_e2m1x2_89;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_93) : "r"((uint32_t)(codes[4] >> 16)));
                        qn_2[2] = _cvt_f16x2_e2m1x2_93;
#else
                : "=r"(_cvt_f16x2_e2m1x2_89) : "r"((uint32_t)(codes[4] >> 16)));
                        qn_2[2] = _cvt_f16x2_e2m1x2_89;
#endif
                        float qw_18[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_18[0]), "=&f"(qw_18[1]) : "r"(qn_2[2]), "f"(0.0f), "f"(0.0f));
                        q_b_2[4] = qw_18[0];
                        q_b_2[5] = qw_18[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_94;
#else
                        uint32_t _cvt_f16x2_e2m1x2_90;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_94) : "r"((uint32_t)(codes[4] >> 24)));
                        qn_2[3] = _cvt_f16x2_e2m1x2_94;
#else
                : "=r"(_cvt_f16x2_e2m1x2_90) : "r"((uint32_t)(codes[4] >> 24)));
                        qn_2[3] = _cvt_f16x2_e2m1x2_90;
#endif
                        float qw_19[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_19[0]), "=&f"(qw_19[1]) : "r"(qn_2[3]), "f"(0.0f), "f"(0.0f));
                        q_b_2[6] = qw_19[0];
                        q_b_2[7] = qw_19[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_95;
#else
                        uint32_t _cvt_f16x2_e2m1x2_91;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_95) : "r"((uint32_t)(codes[5] >> 0)));
                        qn_2[4] = _cvt_f16x2_e2m1x2_95;
#else
                : "=r"(_cvt_f16x2_e2m1x2_91) : "r"((uint32_t)(codes[5] >> 0)));
                        qn_2[4] = _cvt_f16x2_e2m1x2_91;
#endif
                        float qw_20[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_20[0]), "=&f"(qw_20[1]) : "r"(qn_2[4]), "f"(0.0f), "f"(0.0f));
                        q_b_2[8] = qw_20[0];
                        q_b_2[9] = qw_20[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_96;
#else
                        uint32_t _cvt_f16x2_e2m1x2_92;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_96) : "r"((uint32_t)(codes[5] >> 8)));
                        qn_2[5] = _cvt_f16x2_e2m1x2_96;
#else
                : "=r"(_cvt_f16x2_e2m1x2_92) : "r"((uint32_t)(codes[5] >> 8)));
                        qn_2[5] = _cvt_f16x2_e2m1x2_92;
#endif
                        float qw_21[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_21[0]), "=&f"(qw_21[1]) : "r"(qn_2[5]), "f"(0.0f), "f"(0.0f));
                        q_b_2[10] = qw_21[0];
                        q_b_2[11] = qw_21[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_97;
#else
                        uint32_t _cvt_f16x2_e2m1x2_93;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_97) : "r"((uint32_t)(codes[5] >> 16)));
                        qn_2[6] = _cvt_f16x2_e2m1x2_97;
#else
                : "=r"(_cvt_f16x2_e2m1x2_93) : "r"((uint32_t)(codes[5] >> 16)));
                        qn_2[6] = _cvt_f16x2_e2m1x2_93;
#endif
                        float qw_22[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_22[0]), "=&f"(qw_22[1]) : "r"(qn_2[6]), "f"(0.0f), "f"(0.0f));
                        q_b_2[12] = qw_22[0];
                        q_b_2[13] = qw_22[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_98;
#else
                        uint32_t _cvt_f16x2_e2m1x2_94;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_98) : "r"((uint32_t)(codes[5] >> 24)));
                        qn_2[7] = _cvt_f16x2_e2m1x2_98;
#else
                : "=r"(_cvt_f16x2_e2m1x2_94) : "r"((uint32_t)(codes[5] >> 24)));
                        qn_2[7] = _cvt_f16x2_e2m1x2_94;
#endif
                        float qw_23[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_23[0]), "=&f"(qw_23[1]) : "r"(qn_2[7]), "f"(0.0f), "f"(0.0f));
                        q_b_2[14] = qw_23[0];
                        q_b_2[15] = qw_23[1];
                        float qe_2[16];
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(qe_2)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_2)[_ls], reinterpret_cast<const float2*>((s_vals + 32))[_ls]);
                        float t8_4[8];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            reinterpret_cast<float2*>(t8_4)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_2 + 0))[_la], reinterpret_cast<const float2*>((qe_2 + 8))[_la]);
                        #pragma unroll
                        for (int _la = 0; _la < 2; _la++)
                            add_f32x2_inplace(&reinterpret_cast<float2*>(t8_4)[_la], reinterpret_cast<const float2*>((t8_4 + 4))[_la]);
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            add_f32x2_inplace(&reinterpret_cast<float2*>(t8_4)[_la], reinterpret_cast<const float2*>((t8_4 + 2))[_la]);
                        qv_2[2] = t8_4[0] + t8_4[1];
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
                        uint32_t v_2;
                        asm("add.f16x2 %0, %1, %2;" : "=r"(v_2) : "r"(a3_2), "r"(b3_2));
                        float w_2[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(w_2[0]), "=&f"(w_2[1]) : "r"(v_2), "f"(0.0f), "f"(0.0f));
                        qq_2[2] = w_2[0] + w_2[1];
#if __CUDA_ARCH__ == 1030
                        float _rcp_99 = approx_rcp(qq_2[2]);
                        sf_fit[2] = ((g_live_1 != 0 && qq_2[2] > 0.0f) ? sf_ex_1 * qv_2[2] * _rcp_99 : 0.0f);
#else
                        float _rcp_95 = approx_rcp(qq_2[2]);
                        sf_fit[2] = ((g_live_1 != 0 && qq_2[2] > 0.0f) ? sf_ex_1 * qv_2[2] * _rcp_95 : 0.0f);
#endif
                    }
                    int g_live_2 = g[3] > -CUDART_INF_F;
                    float g_safe_2 = ((g_live_2 != 0) ? g[3] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_100 = {c_log2, c_log2};
                    const float2 _fma_c2_101 = {-g_safe_2, -g_safe_2};
#else
                    const float2 _fma_b2_96 = {c_log2, c_log2};
                    const float2 _fma_c2_97 = {-g_safe_2, -g_safe_2};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_100, _fma_c2_101);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_96, _fma_c2_97);
#endif
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
#if __CUDA_ARCH__ == 1030
                    float _exp2_102 = approx_exp2(g_safe_2 - m);
                    b_weights[3] = ((g_live_2 != 0) ? _exp2_102 : 0.0f);
                    float _exp2_103 = approx_exp2(g_safe_2 - m - 4.0f);
                    float sf_ex_2 = 448.0f * _exp2_103;
#else
                    float _exp2_98 = approx_exp2(g_safe_2 - m);
                    b_weights[3] = ((g_live_2 != 0) ? _exp2_98 : 0.0f);
                    float _exp2_99 = approx_exp2(g_safe_2 - m - 4.0f);
                    float sf_ex_2 = 448.0f * _exp2_99;
#endif
                    float e_min_2 = s_vals[(48) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        e_min_2 = fminf(e_min_2, s_vals[(48) + _lr]);
                    }
                    int flat_2 = dither != 0 && e_min_2 >= 0.5f;
                    if (flat_2 != 0) {
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_104 = {six_rho_d[3], six_rho_d[3]};
                        const float2 _fma_c2_105 = {0.0f, 0.0f};
#else
                        const float2 _fma_b2_100 = {six_rho_d[3], six_rho_d[3]};
                        const float2 _fma_c2_101 = {0.0f, 0.0f};
#endif
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_104, _fma_c2_105);
                        uint16_t _cvt_rs_x4_f32_106;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_106) : "f"(s_vals[51]), "f"(s_vals[50]), "f"(s_vals[49]), "f"(s_vals[48]), "r"(3103267870u));
                        unsigned int w_lo_6 = _cvt_rs_x4_f32_106;
                        uint16_t _cvt_rs_x4_f32_107;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_107) : "f"(s_vals[55]), "f"(s_vals[54]), "f"(s_vals[53]), "f"(s_vals[52]), "r"(2564298778u));
                        unsigned int w_hi_6 = _cvt_rs_x4_f32_107;
#else
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_100, _fma_c2_101);
                        uint16_t _cvt_rs_x4_f32_102;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_102) : "f"(s_vals[51]), "f"(s_vals[50]), "f"(s_vals[49]), "f"(s_vals[48]), "r"(3103267870u));
                        unsigned int w_lo_6 = _cvt_rs_x4_f32_102;
                        uint16_t _cvt_rs_x4_f32_103;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_103) : "f"(s_vals[55]), "f"(s_vals[54]), "f"(s_vals[53]), "f"(s_vals[52]), "r"(2564298778u));
                        unsigned int w_hi_6 = _cvt_rs_x4_f32_103;
#endif
                        codes[6] = w_lo_6 | w_hi_6 << 16;
#if __CUDA_ARCH__ == 1030
                        uint16_t _cvt_rs_x4_f32_108;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_108) : "f"(s_vals[59]), "f"(s_vals[58]), "f"(s_vals[57]), "f"(s_vals[56]), "r"(2833781782u));
                        unsigned int w_lo_7 = _cvt_rs_x4_f32_108;
                        uint16_t _cvt_rs_x4_f32_109;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_109) : "f"(s_vals[63]), "f"(s_vals[62]), "f"(s_vals[61]), "f"(s_vals[60]), "r"(2294812690u));
                        unsigned int w_hi_7 = _cvt_rs_x4_f32_109;
#else
                        uint16_t _cvt_rs_x4_f32_104;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_104) : "f"(s_vals[59]), "f"(s_vals[58]), "f"(s_vals[57]), "f"(s_vals[56]), "r"(2833781782u));
                        unsigned int w_lo_7 = _cvt_rs_x4_f32_104;
                        uint16_t _cvt_rs_x4_f32_105;
                        asm("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;" : "=h"(_cvt_rs_x4_f32_105) : "f"(s_vals[63]), "f"(s_vals[62]), "f"(s_vals[61]), "f"(s_vals[60]), "r"(2294812690u));
                        unsigned int w_hi_7 = _cvt_rs_x4_f32_105;
#endif
                        codes[7] = w_lo_7 | w_hi_7 << 16;
                        sf_fit[3] = sf_near[3];
                    } else {
                        float qv_3[4];
                        float qq_3[4];
#if __CUDA_ARCH__ == 1030
                        const float2 _scale2_110 = {6.0f, 6.0f};
#else
                        const float2 _scale2_106 = {6.0f, 6.0f};
#endif
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_ls], _scale2_110);
#else
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_ls], _scale2_106);
#endif
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[6]) : "f"(s_vals[48]), "f"(s_vals[49]), "f"(s_vals[50]), "f"(s_vals[51]), "f"(s_vals[52]), "f"(s_vals[53]), "f"(s_vals[54]), "f"(s_vals[55]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[7]) : "f"(s_vals[56]), "f"(s_vals[57]), "f"(s_vals[58]), "f"(s_vals[59]), "f"(s_vals[60]), "f"(s_vals[61]), "f"(s_vals[62]), "f"(s_vals[63]));
                        float q_b_3[16];
                        unsigned int qn_3[8];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_111;
#else
                        uint32_t _cvt_f16x2_e2m1x2_107;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_111) : "r"((uint32_t)(codes[6] >> 0)));
                        qn_3[0] = _cvt_f16x2_e2m1x2_111;
#else
                : "=r"(_cvt_f16x2_e2m1x2_107) : "r"((uint32_t)(codes[6] >> 0)));
                        qn_3[0] = _cvt_f16x2_e2m1x2_107;
#endif
                        float qw_24[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_24[0]), "=&f"(qw_24[1]) : "r"(qn_3[0]), "f"(0.0f), "f"(0.0f));
                        q_b_3[0] = qw_24[0];
                        q_b_3[1] = qw_24[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_112;
#else
                        uint32_t _cvt_f16x2_e2m1x2_108;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_112) : "r"((uint32_t)(codes[6] >> 8)));
                        qn_3[1] = _cvt_f16x2_e2m1x2_112;
#else
                : "=r"(_cvt_f16x2_e2m1x2_108) : "r"((uint32_t)(codes[6] >> 8)));
                        qn_3[1] = _cvt_f16x2_e2m1x2_108;
#endif
                        float qw_25[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_25[0]), "=&f"(qw_25[1]) : "r"(qn_3[1]), "f"(0.0f), "f"(0.0f));
                        q_b_3[2] = qw_25[0];
                        q_b_3[3] = qw_25[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_113;
#else
                        uint32_t _cvt_f16x2_e2m1x2_109;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_113) : "r"((uint32_t)(codes[6] >> 16)));
                        qn_3[2] = _cvt_f16x2_e2m1x2_113;
#else
                : "=r"(_cvt_f16x2_e2m1x2_109) : "r"((uint32_t)(codes[6] >> 16)));
                        qn_3[2] = _cvt_f16x2_e2m1x2_109;
#endif
                        float qw_26[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_26[0]), "=&f"(qw_26[1]) : "r"(qn_3[2]), "f"(0.0f), "f"(0.0f));
                        q_b_3[4] = qw_26[0];
                        q_b_3[5] = qw_26[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_114;
#else
                        uint32_t _cvt_f16x2_e2m1x2_110;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_114) : "r"((uint32_t)(codes[6] >> 24)));
                        qn_3[3] = _cvt_f16x2_e2m1x2_114;
#else
                : "=r"(_cvt_f16x2_e2m1x2_110) : "r"((uint32_t)(codes[6] >> 24)));
                        qn_3[3] = _cvt_f16x2_e2m1x2_110;
#endif
                        float qw_27[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_27[0]), "=&f"(qw_27[1]) : "r"(qn_3[3]), "f"(0.0f), "f"(0.0f));
                        q_b_3[6] = qw_27[0];
                        q_b_3[7] = qw_27[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_115;
#else
                        uint32_t _cvt_f16x2_e2m1x2_111;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_115) : "r"((uint32_t)(codes[7] >> 0)));
                        qn_3[4] = _cvt_f16x2_e2m1x2_115;
#else
                : "=r"(_cvt_f16x2_e2m1x2_111) : "r"((uint32_t)(codes[7] >> 0)));
                        qn_3[4] = _cvt_f16x2_e2m1x2_111;
#endif
                        float qw_28[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_28[0]), "=&f"(qw_28[1]) : "r"(qn_3[4]), "f"(0.0f), "f"(0.0f));
                        q_b_3[8] = qw_28[0];
                        q_b_3[9] = qw_28[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_116;
#else
                        uint32_t _cvt_f16x2_e2m1x2_112;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_116) : "r"((uint32_t)(codes[7] >> 8)));
                        qn_3[5] = _cvt_f16x2_e2m1x2_116;
#else
                : "=r"(_cvt_f16x2_e2m1x2_112) : "r"((uint32_t)(codes[7] >> 8)));
                        qn_3[5] = _cvt_f16x2_e2m1x2_112;
#endif
                        float qw_29[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_29[0]), "=&f"(qw_29[1]) : "r"(qn_3[5]), "f"(0.0f), "f"(0.0f));
                        q_b_3[10] = qw_29[0];
                        q_b_3[11] = qw_29[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_117;
#else
                        uint32_t _cvt_f16x2_e2m1x2_113;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_117) : "r"((uint32_t)(codes[7] >> 16)));
                        qn_3[6] = _cvt_f16x2_e2m1x2_117;
#else
                : "=r"(_cvt_f16x2_e2m1x2_113) : "r"((uint32_t)(codes[7] >> 16)));
                        qn_3[6] = _cvt_f16x2_e2m1x2_113;
#endif
                        float qw_30[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_30[0]), "=&f"(qw_30[1]) : "r"(qn_3[6]), "f"(0.0f), "f"(0.0f));
                        q_b_3[12] = qw_30[0];
                        q_b_3[13] = qw_30[1];
#if __CUDA_ARCH__ == 1030
                        uint32_t _cvt_f16x2_e2m1x2_118;
#else
                        uint32_t _cvt_f16x2_e2m1x2_114;
#endif
                        asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_118) : "r"((uint32_t)(codes[7] >> 24)));
                        qn_3[7] = _cvt_f16x2_e2m1x2_118;
#else
                : "=r"(_cvt_f16x2_e2m1x2_114) : "r"((uint32_t)(codes[7] >> 24)));
                        qn_3[7] = _cvt_f16x2_e2m1x2_114;
#endif
                        float qw_31[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(qw_31[0]), "=&f"(qw_31[1]) : "r"(qn_3[7]), "f"(0.0f), "f"(0.0f));
                        q_b_3[14] = qw_31[0];
                        q_b_3[15] = qw_31[1];
                        float qe_3[16];
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(qe_3)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_3)[_ls], reinterpret_cast<const float2*>((s_vals + 48))[_ls]);
                        float t8_6[8];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            reinterpret_cast<float2*>(t8_6)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_3 + 0))[_la], reinterpret_cast<const float2*>((qe_3 + 8))[_la]);
                        #pragma unroll
                        for (int _la = 0; _la < 2; _la++)
                            add_f32x2_inplace(&reinterpret_cast<float2*>(t8_6)[_la], reinterpret_cast<const float2*>((t8_6 + 4))[_la]);
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            add_f32x2_inplace(&reinterpret_cast<float2*>(t8_6)[_la], reinterpret_cast<const float2*>((t8_6 + 2))[_la]);
                        qv_3[3] = t8_6[0] + t8_6[1];
                        uint32_t a0_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_3) : "r"(qn_3[0]), "r"(qn_3[0]), "r"(0));
                        uint32_t b0_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_3) : "r"(qn_3[4]), "r"(qn_3[4]), "r"(0));
                        uint32_t a1_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_3) : "r"(qn_3[1]), "r"(qn_3[1]), "r"(a0_3));
                        uint32_t b1_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_3) : "r"(qn_3[5]), "r"(qn_3[5]), "r"(b0_3));
                        uint32_t a2_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_3) : "r"(qn_3[2]), "r"(qn_3[2]), "r"(a1_3));
                        uint32_t b2_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_3) : "r"(qn_3[6]), "r"(qn_3[6]), "r"(b1_3));
                        uint32_t a3_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_3) : "r"(qn_3[3]), "r"(qn_3[3]), "r"(a2_3));
                        uint32_t b3_3;
                        asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_3) : "r"(qn_3[7]), "r"(qn_3[7]), "r"(b2_3));
                        uint32_t v_3;
                        asm("add.f16x2 %0, %1, %2;" : "=r"(v_3) : "r"(a3_3), "r"(b3_3));
                        float w_3[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.f16 %0, lo, %3;\n\t"
                            "add.rn.f32.f16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(w_3[0]), "=&f"(w_3[1]) : "r"(v_3), "f"(0.0f), "f"(0.0f));
                        qq_3[3] = w_3[0] + w_3[1];
#if __CUDA_ARCH__ == 1030
                        float _rcp_119 = approx_rcp(qq_3[3]);
                        sf_fit[3] = ((g_live_2 != 0 && qq_3[3] > 0.0f) ? sf_ex_2 * qv_3[3] * _rcp_119 : 0.0f);
#else
                        float _rcp_115 = approx_rcp(qq_3[3]);
                        sf_fit[3] = ((g_live_2 != 0 && qq_3[3] > 0.0f) ? sf_ex_2 * qv_3[3] * _rcp_115 : 0.0f);
#endif
                    }
                } else {
                    int g_live_3 = g[0] > -CUDART_INF_F;
                    float g_safe_3 = ((g_live_3 != 0) ? g[0] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_120 = {c_log2, c_log2};
                    const float2 _fma_c2_121 = {-g_safe_3, -g_safe_3};
#else
                    const float2 _fma_b2_116 = {c_log2, c_log2};
                    const float2 _fma_c2_117 = {-g_safe_3, -g_safe_3};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_120, _fma_c2_121);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_lf], _fma_b2_116, _fma_c2_117);
#endif
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le] = approx_exp2(s_vals[_le]);
                    }
                    float t8_7[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_7)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 0))[_la], reinterpret_cast<const float2*>((s_vals + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_7)[_la], reinterpret_cast<const float2*>((t8_7 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_7)[_la], reinterpret_cast<const float2*>((t8_7 + 2))[_la]);
                    b_sums[0] = t8_7[0] + t8_7[1];
#if __CUDA_ARCH__ == 1030
                    float _exp2_122 = approx_exp2(g_safe_3 - m);
                    b_weights[0] = ((g_live_3 != 0) ? _exp2_122 : 0.0f);
                    float _exp2_123 = approx_exp2(g_safe_3 - m - 4.0f);
                    float sf_ex_3 = 448.0f * _exp2_123;
#else
                    float _exp2_118 = approx_exp2(g_safe_3 - m);
                    b_weights[0] = ((g_live_3 != 0) ? _exp2_118 : 0.0f);
                    float _exp2_119 = approx_exp2(g_safe_3 - m - 4.0f);
                    float sf_ex_3 = 448.0f * _exp2_119;
#endif
                    float qv_4[4];
                    float qq_4[4];
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_124 = {6.0f, 6.0f};
#else
                    const float2 _scale2_120 = {6.0f, 6.0f};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_ls], _scale2_124);
#else
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 0))[_ls], _scale2_120);
#endif
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[0]) : "f"(s_vals[0]), "f"(s_vals[1]), "f"(s_vals[2]), "f"(s_vals[3]), "f"(s_vals[4]), "f"(s_vals[5]), "f"(s_vals[6]), "f"(s_vals[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[1]) : "f"(s_vals[8]), "f"(s_vals[9]), "f"(s_vals[10]), "f"(s_vals[11]), "f"(s_vals[12]), "f"(s_vals[13]), "f"(s_vals[14]), "f"(s_vals[15]));
                    float q_b_4[16];
                    unsigned int qn_4[8];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_125;
#else
                    uint32_t _cvt_f16x2_e2m1x2_121;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_125) : "r"((uint32_t)(codes[0] >> 0)));
                    qn_4[0] = _cvt_f16x2_e2m1x2_125;
#else
                : "=r"(_cvt_f16x2_e2m1x2_121) : "r"((uint32_t)(codes[0] >> 0)));
                    qn_4[0] = _cvt_f16x2_e2m1x2_121;
#endif
                    float qw_32[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_32[0]), "=&f"(qw_32[1]) : "r"(qn_4[0]), "f"(0.0f), "f"(0.0f));
                    q_b_4[0] = qw_32[0];
                    q_b_4[1] = qw_32[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_126;
#else
                    uint32_t _cvt_f16x2_e2m1x2_122;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_126) : "r"((uint32_t)(codes[0] >> 8)));
                    qn_4[1] = _cvt_f16x2_e2m1x2_126;
#else
                : "=r"(_cvt_f16x2_e2m1x2_122) : "r"((uint32_t)(codes[0] >> 8)));
                    qn_4[1] = _cvt_f16x2_e2m1x2_122;
#endif
                    float qw_33[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_33[0]), "=&f"(qw_33[1]) : "r"(qn_4[1]), "f"(0.0f), "f"(0.0f));
                    q_b_4[2] = qw_33[0];
                    q_b_4[3] = qw_33[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_127;
#else
                    uint32_t _cvt_f16x2_e2m1x2_123;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_127) : "r"((uint32_t)(codes[0] >> 16)));
                    qn_4[2] = _cvt_f16x2_e2m1x2_127;
#else
                : "=r"(_cvt_f16x2_e2m1x2_123) : "r"((uint32_t)(codes[0] >> 16)));
                    qn_4[2] = _cvt_f16x2_e2m1x2_123;
#endif
                    float qw_34[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_34[0]), "=&f"(qw_34[1]) : "r"(qn_4[2]), "f"(0.0f), "f"(0.0f));
                    q_b_4[4] = qw_34[0];
                    q_b_4[5] = qw_34[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_128;
#else
                    uint32_t _cvt_f16x2_e2m1x2_124;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_128) : "r"((uint32_t)(codes[0] >> 24)));
                    qn_4[3] = _cvt_f16x2_e2m1x2_128;
#else
                : "=r"(_cvt_f16x2_e2m1x2_124) : "r"((uint32_t)(codes[0] >> 24)));
                    qn_4[3] = _cvt_f16x2_e2m1x2_124;
#endif
                    float qw_35[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_35[0]), "=&f"(qw_35[1]) : "r"(qn_4[3]), "f"(0.0f), "f"(0.0f));
                    q_b_4[6] = qw_35[0];
                    q_b_4[7] = qw_35[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_129;
#else
                    uint32_t _cvt_f16x2_e2m1x2_125;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_129) : "r"((uint32_t)(codes[1] >> 0)));
                    qn_4[4] = _cvt_f16x2_e2m1x2_129;
#else
                : "=r"(_cvt_f16x2_e2m1x2_125) : "r"((uint32_t)(codes[1] >> 0)));
                    qn_4[4] = _cvt_f16x2_e2m1x2_125;
#endif
                    float qw_36[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_36[0]), "=&f"(qw_36[1]) : "r"(qn_4[4]), "f"(0.0f), "f"(0.0f));
                    q_b_4[8] = qw_36[0];
                    q_b_4[9] = qw_36[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_130;
#else
                    uint32_t _cvt_f16x2_e2m1x2_126;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_130) : "r"((uint32_t)(codes[1] >> 8)));
                    qn_4[5] = _cvt_f16x2_e2m1x2_130;
#else
                : "=r"(_cvt_f16x2_e2m1x2_126) : "r"((uint32_t)(codes[1] >> 8)));
                    qn_4[5] = _cvt_f16x2_e2m1x2_126;
#endif
                    float qw_37[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_37[0]), "=&f"(qw_37[1]) : "r"(qn_4[5]), "f"(0.0f), "f"(0.0f));
                    q_b_4[10] = qw_37[0];
                    q_b_4[11] = qw_37[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_131;
#else
                    uint32_t _cvt_f16x2_e2m1x2_127;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_131) : "r"((uint32_t)(codes[1] >> 16)));
                    qn_4[6] = _cvt_f16x2_e2m1x2_131;
#else
                : "=r"(_cvt_f16x2_e2m1x2_127) : "r"((uint32_t)(codes[1] >> 16)));
                    qn_4[6] = _cvt_f16x2_e2m1x2_127;
#endif
                    float qw_38[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_38[0]), "=&f"(qw_38[1]) : "r"(qn_4[6]), "f"(0.0f), "f"(0.0f));
                    q_b_4[12] = qw_38[0];
                    q_b_4[13] = qw_38[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_132;
#else
                    uint32_t _cvt_f16x2_e2m1x2_128;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_132) : "r"((uint32_t)(codes[1] >> 24)));
                    qn_4[7] = _cvt_f16x2_e2m1x2_132;
#else
                : "=r"(_cvt_f16x2_e2m1x2_128) : "r"((uint32_t)(codes[1] >> 24)));
                    qn_4[7] = _cvt_f16x2_e2m1x2_128;
#endif
                    float qw_39[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_39[0]), "=&f"(qw_39[1]) : "r"(qn_4[7]), "f"(0.0f), "f"(0.0f));
                    q_b_4[14] = qw_39[0];
                    q_b_4[15] = qw_39[1];
                    float qe_4[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_4)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_4)[_ls], reinterpret_cast<const float2*>((s_vals + 0))[_ls]);
                    float t8_7_1[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_7_1)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_4 + 0))[_la], reinterpret_cast<const float2*>((qe_4 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_7_1)[_la], reinterpret_cast<const float2*>((t8_7_1 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_7_1)[_la], reinterpret_cast<const float2*>((t8_7_1 + 2))[_la]);
                    qv_4[0] = t8_7_1[0] + t8_7_1[1];
                    uint32_t a0_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_4) : "r"(qn_4[0]), "r"(qn_4[0]), "r"(0));
                    uint32_t b0_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_4) : "r"(qn_4[4]), "r"(qn_4[4]), "r"(0));
                    uint32_t a1_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_4) : "r"(qn_4[1]), "r"(qn_4[1]), "r"(a0_4));
                    uint32_t b1_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_4) : "r"(qn_4[5]), "r"(qn_4[5]), "r"(b0_4));
                    uint32_t a2_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_4) : "r"(qn_4[2]), "r"(qn_4[2]), "r"(a1_4));
                    uint32_t b2_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_4) : "r"(qn_4[6]), "r"(qn_4[6]), "r"(b1_4));
                    uint32_t a3_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_4) : "r"(qn_4[3]), "r"(qn_4[3]), "r"(a2_4));
                    uint32_t b3_4;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_4) : "r"(qn_4[7]), "r"(qn_4[7]), "r"(b2_4));
                    uint32_t v_4;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v_4) : "r"(a3_4), "r"(b3_4));
                    float w_4[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w_4[0]), "=&f"(w_4[1]) : "r"(v_4), "f"(0.0f), "f"(0.0f));
                    qq_4[0] = w_4[0] + w_4[1];
#if __CUDA_ARCH__ == 1030
                    float _rcp_133 = approx_rcp(qq_4[0]);
                    sf_fit[0] = ((g_live_3 != 0 && qq_4[0] > 0.0f) ? sf_ex_3 * qv_4[0] * _rcp_133 : 0.0f);
#else
                    float _rcp_129 = approx_rcp(qq_4[0]);
                    sf_fit[0] = ((g_live_3 != 0 && qq_4[0] > 0.0f) ? sf_ex_3 * qv_4[0] * _rcp_129 : 0.0f);
#endif
                    int g_live_3_1 = g[1] > -CUDART_INF_F;
                    float g_safe_3_1 = ((g_live_3_1 != 0) ? g[1] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_134 = {c_log2, c_log2};
                    const float2 _fma_c2_135 = {-g_safe_3_1, -g_safe_3_1};
#else
                    const float2 _fma_b2_130 = {c_log2, c_log2};
                    const float2 _fma_c2_131 = {-g_safe_3_1, -g_safe_3_1};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_134, _fma_c2_135);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_lf], _fma_b2_130, _fma_c2_131);
#endif
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le + 16] = approx_exp2(s_vals[_le + 16]);
                    }
                    float t8_8[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_8)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 16))[_la], reinterpret_cast<const float2*>((s_vals + 24))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_8)[_la], reinterpret_cast<const float2*>((t8_8 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_8)[_la], reinterpret_cast<const float2*>((t8_8 + 2))[_la]);
                    b_sums[1] = t8_8[0] + t8_8[1];
#if __CUDA_ARCH__ == 1030
                    float _exp2_136 = approx_exp2(g_safe_3_1 - m);
                    b_weights[1] = ((g_live_3_1 != 0) ? _exp2_136 : 0.0f);
                    float _exp2_137 = approx_exp2(g_safe_3_1 - m - 4.0f);
                    float sf_ex_3_1 = 448.0f * _exp2_137;
#else
                    float _exp2_132 = approx_exp2(g_safe_3_1 - m);
                    b_weights[1] = ((g_live_3_1 != 0) ? _exp2_132 : 0.0f);
                    float _exp2_133 = approx_exp2(g_safe_3_1 - m - 4.0f);
                    float sf_ex_3_1 = 448.0f * _exp2_133;
#endif
                    float qv_0[4];
                    float qq_0[4];
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_138 = {6.0f, 6.0f};
#else
                    const float2 _scale2_134 = {6.0f, 6.0f};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_ls], _scale2_138);
#else
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 16))[_ls], _scale2_134);
#endif
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[2]) : "f"(s_vals[16]), "f"(s_vals[17]), "f"(s_vals[18]), "f"(s_vals[19]), "f"(s_vals[20]), "f"(s_vals[21]), "f"(s_vals[22]), "f"(s_vals[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[3]) : "f"(s_vals[24]), "f"(s_vals[25]), "f"(s_vals[26]), "f"(s_vals[27]), "f"(s_vals[28]), "f"(s_vals[29]), "f"(s_vals[30]), "f"(s_vals[31]));
                    float q_b_0[16];
                    unsigned int qn_0[8];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_139;
#else
                    uint32_t _cvt_f16x2_e2m1x2_135;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_139) : "r"((uint32_t)(codes[2] >> 0)));
                    qn_0[0] = _cvt_f16x2_e2m1x2_139;
#else
                : "=r"(_cvt_f16x2_e2m1x2_135) : "r"((uint32_t)(codes[2] >> 0)));
                    qn_0[0] = _cvt_f16x2_e2m1x2_135;
#endif
                    float qw_40[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_40[0]), "=&f"(qw_40[1]) : "r"(qn_0[0]), "f"(0.0f), "f"(0.0f));
                    q_b_0[0] = qw_40[0];
                    q_b_0[1] = qw_40[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_140;
#else
                    uint32_t _cvt_f16x2_e2m1x2_136;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_140) : "r"((uint32_t)(codes[2] >> 8)));
                    qn_0[1] = _cvt_f16x2_e2m1x2_140;
#else
                : "=r"(_cvt_f16x2_e2m1x2_136) : "r"((uint32_t)(codes[2] >> 8)));
                    qn_0[1] = _cvt_f16x2_e2m1x2_136;
#endif
                    float qw_41[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_41[0]), "=&f"(qw_41[1]) : "r"(qn_0[1]), "f"(0.0f), "f"(0.0f));
                    q_b_0[2] = qw_41[0];
                    q_b_0[3] = qw_41[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_141;
#else
                    uint32_t _cvt_f16x2_e2m1x2_137;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_141) : "r"((uint32_t)(codes[2] >> 16)));
                    qn_0[2] = _cvt_f16x2_e2m1x2_141;
#else
                : "=r"(_cvt_f16x2_e2m1x2_137) : "r"((uint32_t)(codes[2] >> 16)));
                    qn_0[2] = _cvt_f16x2_e2m1x2_137;
#endif
                    float qw_42[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_42[0]), "=&f"(qw_42[1]) : "r"(qn_0[2]), "f"(0.0f), "f"(0.0f));
                    q_b_0[4] = qw_42[0];
                    q_b_0[5] = qw_42[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_142;
#else
                    uint32_t _cvt_f16x2_e2m1x2_138;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_142) : "r"((uint32_t)(codes[2] >> 24)));
                    qn_0[3] = _cvt_f16x2_e2m1x2_142;
#else
                : "=r"(_cvt_f16x2_e2m1x2_138) : "r"((uint32_t)(codes[2] >> 24)));
                    qn_0[3] = _cvt_f16x2_e2m1x2_138;
#endif
                    float qw_43[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_43[0]), "=&f"(qw_43[1]) : "r"(qn_0[3]), "f"(0.0f), "f"(0.0f));
                    q_b_0[6] = qw_43[0];
                    q_b_0[7] = qw_43[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_143;
#else
                    uint32_t _cvt_f16x2_e2m1x2_139;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_143) : "r"((uint32_t)(codes[3] >> 0)));
                    qn_0[4] = _cvt_f16x2_e2m1x2_143;
#else
                : "=r"(_cvt_f16x2_e2m1x2_139) : "r"((uint32_t)(codes[3] >> 0)));
                    qn_0[4] = _cvt_f16x2_e2m1x2_139;
#endif
                    float qw_44[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_44[0]), "=&f"(qw_44[1]) : "r"(qn_0[4]), "f"(0.0f), "f"(0.0f));
                    q_b_0[8] = qw_44[0];
                    q_b_0[9] = qw_44[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_144;
#else
                    uint32_t _cvt_f16x2_e2m1x2_140;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_144) : "r"((uint32_t)(codes[3] >> 8)));
                    qn_0[5] = _cvt_f16x2_e2m1x2_144;
#else
                : "=r"(_cvt_f16x2_e2m1x2_140) : "r"((uint32_t)(codes[3] >> 8)));
                    qn_0[5] = _cvt_f16x2_e2m1x2_140;
#endif
                    float qw_45[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_45[0]), "=&f"(qw_45[1]) : "r"(qn_0[5]), "f"(0.0f), "f"(0.0f));
                    q_b_0[10] = qw_45[0];
                    q_b_0[11] = qw_45[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_145;
#else
                    uint32_t _cvt_f16x2_e2m1x2_141;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_145) : "r"((uint32_t)(codes[3] >> 16)));
                    qn_0[6] = _cvt_f16x2_e2m1x2_145;
#else
                : "=r"(_cvt_f16x2_e2m1x2_141) : "r"((uint32_t)(codes[3] >> 16)));
                    qn_0[6] = _cvt_f16x2_e2m1x2_141;
#endif
                    float qw_46[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_46[0]), "=&f"(qw_46[1]) : "r"(qn_0[6]), "f"(0.0f), "f"(0.0f));
                    q_b_0[12] = qw_46[0];
                    q_b_0[13] = qw_46[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_146;
#else
                    uint32_t _cvt_f16x2_e2m1x2_142;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_146) : "r"((uint32_t)(codes[3] >> 24)));
                    qn_0[7] = _cvt_f16x2_e2m1x2_146;
#else
                : "=r"(_cvt_f16x2_e2m1x2_142) : "r"((uint32_t)(codes[3] >> 24)));
                    qn_0[7] = _cvt_f16x2_e2m1x2_142;
#endif
                    float qw_47[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_47[0]), "=&f"(qw_47[1]) : "r"(qn_0[7]), "f"(0.0f), "f"(0.0f));
                    q_b_0[14] = qw_47[0];
                    q_b_0[15] = qw_47[1];
                    float qe_0[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_0)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_0)[_ls], reinterpret_cast<const float2*>((s_vals + 16))[_ls]);
                    float t8_9[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_9)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_0 + 0))[_la], reinterpret_cast<const float2*>((qe_0 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_9)[_la], reinterpret_cast<const float2*>((t8_9 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_9)[_la], reinterpret_cast<const float2*>((t8_9 + 2))[_la]);
                    qv_0[1] = t8_9[0] + t8_9[1];
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
                    qq_0[1] = w_0[0] + w_0[1];
#if __CUDA_ARCH__ == 1030
                    float _rcp_147 = approx_rcp(qq_0[1]);
                    sf_fit[1] = ((g_live_3_1 != 0 && qq_0[1] > 0.0f) ? sf_ex_3_1 * qv_0[1] * _rcp_147 : 0.0f);
#else
                    float _rcp_143 = approx_rcp(qq_0[1]);
                    sf_fit[1] = ((g_live_3_1 != 0 && qq_0[1] > 0.0f) ? sf_ex_3_1 * qv_0[1] * _rcp_143 : 0.0f);
#endif
                    int g_live_4 = g[2] > -CUDART_INF_F;
                    float g_safe_4 = ((g_live_4 != 0) ? g[2] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_148 = {c_log2, c_log2};
                    const float2 _fma_c2_149 = {-g_safe_4, -g_safe_4};
#else
                    const float2 _fma_b2_144 = {c_log2, c_log2};
                    const float2 _fma_c2_145 = {-g_safe_4, -g_safe_4};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_148, _fma_c2_149);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_lf], _fma_b2_144, _fma_c2_145);
#endif
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le + 32] = approx_exp2(s_vals[_le + 32]);
                    }
                    float t8_10[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_10)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 32))[_la], reinterpret_cast<const float2*>((s_vals + 40))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_10)[_la], reinterpret_cast<const float2*>((t8_10 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_10)[_la], reinterpret_cast<const float2*>((t8_10 + 2))[_la]);
                    b_sums[2] = t8_10[0] + t8_10[1];
#if __CUDA_ARCH__ == 1030
                    float _exp2_150 = approx_exp2(g_safe_4 - m);
                    b_weights[2] = ((g_live_4 != 0) ? _exp2_150 : 0.0f);
                    float _exp2_151 = approx_exp2(g_safe_4 - m - 4.0f);
                    float sf_ex_4 = 448.0f * _exp2_151;
#else
                    float _exp2_146 = approx_exp2(g_safe_4 - m);
                    b_weights[2] = ((g_live_4 != 0) ? _exp2_146 : 0.0f);
                    float _exp2_147 = approx_exp2(g_safe_4 - m - 4.0f);
                    float sf_ex_4 = 448.0f * _exp2_147;
#endif
                    float qv_1_1[4];
                    float qq_1_1[4];
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_152 = {6.0f, 6.0f};
#else
                    const float2 _scale2_148 = {6.0f, 6.0f};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_ls], _scale2_152);
#else
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 32))[_ls], _scale2_148);
#endif
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[4]) : "f"(s_vals[32]), "f"(s_vals[33]), "f"(s_vals[34]), "f"(s_vals[35]), "f"(s_vals[36]), "f"(s_vals[37]), "f"(s_vals[38]), "f"(s_vals[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[5]) : "f"(s_vals[40]), "f"(s_vals[41]), "f"(s_vals[42]), "f"(s_vals[43]), "f"(s_vals[44]), "f"(s_vals[45]), "f"(s_vals[46]), "f"(s_vals[47]));
                    float q_b_1_1[16];
                    unsigned int qn_1_1[8];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_153;
#else
                    uint32_t _cvt_f16x2_e2m1x2_149;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_153) : "r"((uint32_t)(codes[4] >> 0)));
                    qn_1_1[0] = _cvt_f16x2_e2m1x2_153;
#else
                : "=r"(_cvt_f16x2_e2m1x2_149) : "r"((uint32_t)(codes[4] >> 0)));
                    qn_1_1[0] = _cvt_f16x2_e2m1x2_149;
#endif
                    float qw_48[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_48[0]), "=&f"(qw_48[1]) : "r"(qn_1_1[0]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[0] = qw_48[0];
                    q_b_1_1[1] = qw_48[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_154;
#else
                    uint32_t _cvt_f16x2_e2m1x2_150;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_154) : "r"((uint32_t)(codes[4] >> 8)));
                    qn_1_1[1] = _cvt_f16x2_e2m1x2_154;
#else
                : "=r"(_cvt_f16x2_e2m1x2_150) : "r"((uint32_t)(codes[4] >> 8)));
                    qn_1_1[1] = _cvt_f16x2_e2m1x2_150;
#endif
                    float qw_49[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_49[0]), "=&f"(qw_49[1]) : "r"(qn_1_1[1]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[2] = qw_49[0];
                    q_b_1_1[3] = qw_49[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_155;
#else
                    uint32_t _cvt_f16x2_e2m1x2_151;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_155) : "r"((uint32_t)(codes[4] >> 16)));
                    qn_1_1[2] = _cvt_f16x2_e2m1x2_155;
#else
                : "=r"(_cvt_f16x2_e2m1x2_151) : "r"((uint32_t)(codes[4] >> 16)));
                    qn_1_1[2] = _cvt_f16x2_e2m1x2_151;
#endif
                    float qw_50[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_50[0]), "=&f"(qw_50[1]) : "r"(qn_1_1[2]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[4] = qw_50[0];
                    q_b_1_1[5] = qw_50[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_156;
#else
                    uint32_t _cvt_f16x2_e2m1x2_152;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_156) : "r"((uint32_t)(codes[4] >> 24)));
                    qn_1_1[3] = _cvt_f16x2_e2m1x2_156;
#else
                : "=r"(_cvt_f16x2_e2m1x2_152) : "r"((uint32_t)(codes[4] >> 24)));
                    qn_1_1[3] = _cvt_f16x2_e2m1x2_152;
#endif
                    float qw_51[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_51[0]), "=&f"(qw_51[1]) : "r"(qn_1_1[3]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[6] = qw_51[0];
                    q_b_1_1[7] = qw_51[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_157;
#else
                    uint32_t _cvt_f16x2_e2m1x2_153;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_157) : "r"((uint32_t)(codes[5] >> 0)));
                    qn_1_1[4] = _cvt_f16x2_e2m1x2_157;
#else
                : "=r"(_cvt_f16x2_e2m1x2_153) : "r"((uint32_t)(codes[5] >> 0)));
                    qn_1_1[4] = _cvt_f16x2_e2m1x2_153;
#endif
                    float qw_52[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_52[0]), "=&f"(qw_52[1]) : "r"(qn_1_1[4]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[8] = qw_52[0];
                    q_b_1_1[9] = qw_52[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_158;
#else
                    uint32_t _cvt_f16x2_e2m1x2_154;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_158) : "r"((uint32_t)(codes[5] >> 8)));
                    qn_1_1[5] = _cvt_f16x2_e2m1x2_158;
#else
                : "=r"(_cvt_f16x2_e2m1x2_154) : "r"((uint32_t)(codes[5] >> 8)));
                    qn_1_1[5] = _cvt_f16x2_e2m1x2_154;
#endif
                    float qw_53[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_53[0]), "=&f"(qw_53[1]) : "r"(qn_1_1[5]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[10] = qw_53[0];
                    q_b_1_1[11] = qw_53[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_159;
#else
                    uint32_t _cvt_f16x2_e2m1x2_155;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_159) : "r"((uint32_t)(codes[5] >> 16)));
                    qn_1_1[6] = _cvt_f16x2_e2m1x2_159;
#else
                : "=r"(_cvt_f16x2_e2m1x2_155) : "r"((uint32_t)(codes[5] >> 16)));
                    qn_1_1[6] = _cvt_f16x2_e2m1x2_155;
#endif
                    float qw_54[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_54[0]), "=&f"(qw_54[1]) : "r"(qn_1_1[6]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[12] = qw_54[0];
                    q_b_1_1[13] = qw_54[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_160;
#else
                    uint32_t _cvt_f16x2_e2m1x2_156;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_160) : "r"((uint32_t)(codes[5] >> 24)));
                    qn_1_1[7] = _cvt_f16x2_e2m1x2_160;
#else
                : "=r"(_cvt_f16x2_e2m1x2_156) : "r"((uint32_t)(codes[5] >> 24)));
                    qn_1_1[7] = _cvt_f16x2_e2m1x2_156;
#endif
                    float qw_55[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_55[0]), "=&f"(qw_55[1]) : "r"(qn_1_1[7]), "f"(0.0f), "f"(0.0f));
                    q_b_1_1[14] = qw_55[0];
                    q_b_1_1[15] = qw_55[1];
                    float qe_1_1[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_1_1)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_1_1)[_ls], reinterpret_cast<const float2*>((s_vals + 32))[_ls]);
                    float t8_11[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_11)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_1_1 + 0))[_la], reinterpret_cast<const float2*>((qe_1_1 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_11)[_la], reinterpret_cast<const float2*>((t8_11 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_11)[_la], reinterpret_cast<const float2*>((t8_11 + 2))[_la]);
                    qv_1_1[2] = t8_11[0] + t8_11[1];
                    uint32_t a0_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_1_1) : "r"(qn_1_1[0]), "r"(qn_1_1[0]), "r"(0));
                    uint32_t b0_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_1_1) : "r"(qn_1_1[4]), "r"(qn_1_1[4]), "r"(0));
                    uint32_t a1_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_1_1) : "r"(qn_1_1[1]), "r"(qn_1_1[1]), "r"(a0_1_1));
                    uint32_t b1_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_1_1) : "r"(qn_1_1[5]), "r"(qn_1_1[5]), "r"(b0_1_1));
                    uint32_t a2_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_1_1) : "r"(qn_1_1[2]), "r"(qn_1_1[2]), "r"(a1_1_1));
                    uint32_t b2_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_1_1) : "r"(qn_1_1[6]), "r"(qn_1_1[6]), "r"(b1_1_1));
                    uint32_t a3_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_1_1) : "r"(qn_1_1[3]), "r"(qn_1_1[3]), "r"(a2_1_1));
                    uint32_t b3_1_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_1_1) : "r"(qn_1_1[7]), "r"(qn_1_1[7]), "r"(b2_1_1));
                    uint32_t v_17;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v_17) : "r"(a3_1_1), "r"(b3_1_1));
                    float w_1_1[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w_1_1[0]), "=&f"(w_1_1[1]) : "r"(v_17), "f"(0.0f), "f"(0.0f));
                    qq_1_1[2] = w_1_1[0] + w_1_1[1];
#if __CUDA_ARCH__ == 1030
                    float _rcp_161 = approx_rcp(qq_1_1[2]);
                    sf_fit[2] = ((g_live_4 != 0 && qq_1_1[2] > 0.0f) ? sf_ex_4 * qv_1_1[2] * _rcp_161 : 0.0f);
#else
                    float _rcp_157 = approx_rcp(qq_1_1[2]);
                    sf_fit[2] = ((g_live_4 != 0 && qq_1_1[2] > 0.0f) ? sf_ex_4 * qv_1_1[2] * _rcp_157 : 0.0f);
#endif
                    int g_live_5 = g[3] > -CUDART_INF_F;
                    float g_safe_5 = ((g_live_5 != 0) ? g[3] : 0.0f);
#if __CUDA_ARCH__ == 1030
                    const float2 _fma_b2_162 = {c_log2, c_log2};
                    const float2 _fma_c2_163 = {-g_safe_5, -g_safe_5};
#else
                    const float2 _fma_b2_158 = {c_log2, c_log2};
                    const float2 _fma_c2_159 = {-g_safe_5, -g_safe_5};
#endif
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
#if __CUDA_ARCH__ == 1030
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_162, _fma_c2_163);
#else
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_lf], _fma_b2_158, _fma_c2_159);
#endif
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        s_vals[_le + 48] = approx_exp2(s_vals[_le + 48]);
                    }
                    float t8_12[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_12)[_la] = add_f32x2(reinterpret_cast<float2*>((s_vals + 48))[_la], reinterpret_cast<const float2*>((s_vals + 56))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_12)[_la], reinterpret_cast<const float2*>((t8_12 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_12)[_la], reinterpret_cast<const float2*>((t8_12 + 2))[_la]);
                    b_sums[3] = t8_12[0] + t8_12[1];
#if __CUDA_ARCH__ == 1030
                    float _exp2_164 = approx_exp2(g_safe_5 - m);
                    b_weights[3] = ((g_live_5 != 0) ? _exp2_164 : 0.0f);
                    float _exp2_165 = approx_exp2(g_safe_5 - m - 4.0f);
                    float sf_ex_5 = 448.0f * _exp2_165;
#else
                    float _exp2_160 = approx_exp2(g_safe_5 - m);
                    b_weights[3] = ((g_live_5 != 0) ? _exp2_160 : 0.0f);
                    float _exp2_161 = approx_exp2(g_safe_5 - m - 4.0f);
                    float sf_ex_5 = 448.0f * _exp2_161;
#endif
                    float qv_2_1[4];
                    float qq_2_1[4];
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_166 = {6.0f, 6.0f};
#else
                    const float2 _scale2_162 = {6.0f, 6.0f};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_ls], _scale2_166);
#else
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((s_vals + 48))[_ls], _scale2_162);
#endif
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[6]) : "f"(s_vals[48]), "f"(s_vals[49]), "f"(s_vals[50]), "f"(s_vals[51]), "f"(s_vals[52]), "f"(s_vals[53]), "f"(s_vals[54]), "f"(s_vals[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(codes[7]) : "f"(s_vals[56]), "f"(s_vals[57]), "f"(s_vals[58]), "f"(s_vals[59]), "f"(s_vals[60]), "f"(s_vals[61]), "f"(s_vals[62]), "f"(s_vals[63]));
                    float q_b_2_1[16];
                    unsigned int qn_2_1[8];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_167;
#else
                    uint32_t _cvt_f16x2_e2m1x2_163;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_167) : "r"((uint32_t)(codes[6] >> 0)));
                    qn_2_1[0] = _cvt_f16x2_e2m1x2_167;
#else
                : "=r"(_cvt_f16x2_e2m1x2_163) : "r"((uint32_t)(codes[6] >> 0)));
                    qn_2_1[0] = _cvt_f16x2_e2m1x2_163;
#endif
                    float qw_56[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_56[0]), "=&f"(qw_56[1]) : "r"(qn_2_1[0]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[0] = qw_56[0];
                    q_b_2_1[1] = qw_56[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_168;
#else
                    uint32_t _cvt_f16x2_e2m1x2_164;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_168) : "r"((uint32_t)(codes[6] >> 8)));
                    qn_2_1[1] = _cvt_f16x2_e2m1x2_168;
#else
                : "=r"(_cvt_f16x2_e2m1x2_164) : "r"((uint32_t)(codes[6] >> 8)));
                    qn_2_1[1] = _cvt_f16x2_e2m1x2_164;
#endif
                    float qw_57[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_57[0]), "=&f"(qw_57[1]) : "r"(qn_2_1[1]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[2] = qw_57[0];
                    q_b_2_1[3] = qw_57[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_169;
#else
                    uint32_t _cvt_f16x2_e2m1x2_165;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_169) : "r"((uint32_t)(codes[6] >> 16)));
                    qn_2_1[2] = _cvt_f16x2_e2m1x2_169;
#else
                : "=r"(_cvt_f16x2_e2m1x2_165) : "r"((uint32_t)(codes[6] >> 16)));
                    qn_2_1[2] = _cvt_f16x2_e2m1x2_165;
#endif
                    float qw_58[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_58[0]), "=&f"(qw_58[1]) : "r"(qn_2_1[2]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[4] = qw_58[0];
                    q_b_2_1[5] = qw_58[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_170;
#else
                    uint32_t _cvt_f16x2_e2m1x2_166;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_170) : "r"((uint32_t)(codes[6] >> 24)));
                    qn_2_1[3] = _cvt_f16x2_e2m1x2_170;
#else
                : "=r"(_cvt_f16x2_e2m1x2_166) : "r"((uint32_t)(codes[6] >> 24)));
                    qn_2_1[3] = _cvt_f16x2_e2m1x2_166;
#endif
                    float qw_59[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_59[0]), "=&f"(qw_59[1]) : "r"(qn_2_1[3]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[6] = qw_59[0];
                    q_b_2_1[7] = qw_59[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_171;
#else
                    uint32_t _cvt_f16x2_e2m1x2_167;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_171) : "r"((uint32_t)(codes[7] >> 0)));
                    qn_2_1[4] = _cvt_f16x2_e2m1x2_171;
#else
                : "=r"(_cvt_f16x2_e2m1x2_167) : "r"((uint32_t)(codes[7] >> 0)));
                    qn_2_1[4] = _cvt_f16x2_e2m1x2_167;
#endif
                    float qw_60[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_60[0]), "=&f"(qw_60[1]) : "r"(qn_2_1[4]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[8] = qw_60[0];
                    q_b_2_1[9] = qw_60[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_172;
#else
                    uint32_t _cvt_f16x2_e2m1x2_168;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_172) : "r"((uint32_t)(codes[7] >> 8)));
                    qn_2_1[5] = _cvt_f16x2_e2m1x2_172;
#else
                : "=r"(_cvt_f16x2_e2m1x2_168) : "r"((uint32_t)(codes[7] >> 8)));
                    qn_2_1[5] = _cvt_f16x2_e2m1x2_168;
#endif
                    float qw_61[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_61[0]), "=&f"(qw_61[1]) : "r"(qn_2_1[5]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[10] = qw_61[0];
                    q_b_2_1[11] = qw_61[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_173;
#else
                    uint32_t _cvt_f16x2_e2m1x2_169;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_173) : "r"((uint32_t)(codes[7] >> 16)));
                    qn_2_1[6] = _cvt_f16x2_e2m1x2_173;
#else
                : "=r"(_cvt_f16x2_e2m1x2_169) : "r"((uint32_t)(codes[7] >> 16)));
                    qn_2_1[6] = _cvt_f16x2_e2m1x2_169;
#endif
                    float qw_62[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_62[0]), "=&f"(qw_62[1]) : "r"(qn_2_1[6]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[12] = qw_62[0];
                    q_b_2_1[13] = qw_62[1];
#if __CUDA_ARCH__ == 1030
                    uint32_t _cvt_f16x2_e2m1x2_174;
#else
                    uint32_t _cvt_f16x2_e2m1x2_170;
#endif
                    asm("{ .reg .b8 _b;                        \n\t"
                "  mov.b32 {_b, _, _, _}, %1;           \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
#if __CUDA_ARCH__ == 1030
                : "=r"(_cvt_f16x2_e2m1x2_174) : "r"((uint32_t)(codes[7] >> 24)));
                    qn_2_1[7] = _cvt_f16x2_e2m1x2_174;
#else
                : "=r"(_cvt_f16x2_e2m1x2_170) : "r"((uint32_t)(codes[7] >> 24)));
                    qn_2_1[7] = _cvt_f16x2_e2m1x2_170;
#endif
                    float qw_63[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(qw_63[0]), "=&f"(qw_63[1]) : "r"(qn_2_1[7]), "f"(0.0f), "f"(0.0f));
                    q_b_2_1[14] = qw_63[0];
                    q_b_2_1[15] = qw_63[1];
                    float qe_2_1[16];
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        reinterpret_cast<float2*>(qe_2_1)[_ls] = mul_f32x2(reinterpret_cast<float2*>(q_b_2_1)[_ls], reinterpret_cast<const float2*>((s_vals + 48))[_ls]);
                    float t8_13[8];
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        reinterpret_cast<float2*>(t8_13)[_la] = add_f32x2(reinterpret_cast<float2*>((qe_2_1 + 0))[_la], reinterpret_cast<const float2*>((qe_2_1 + 8))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 2; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_13)[_la], reinterpret_cast<const float2*>((t8_13 + 4))[_la]);
                    #pragma unroll
                    for (int _la = 0; _la < 1; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(t8_13)[_la], reinterpret_cast<const float2*>((t8_13 + 2))[_la]);
                    qv_2_1[3] = t8_13[0] + t8_13[1];
                    uint32_t a0_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a0_2_1) : "r"(qn_2_1[0]), "r"(qn_2_1[0]), "r"(0));
                    uint32_t b0_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b0_2_1) : "r"(qn_2_1[4]), "r"(qn_2_1[4]), "r"(0));
                    uint32_t a1_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a1_2_1) : "r"(qn_2_1[1]), "r"(qn_2_1[1]), "r"(a0_2_1));
                    uint32_t b1_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b1_2_1) : "r"(qn_2_1[5]), "r"(qn_2_1[5]), "r"(b0_2_1));
                    uint32_t a2_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a2_2_1) : "r"(qn_2_1[2]), "r"(qn_2_1[2]), "r"(a1_2_1));
                    uint32_t b2_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b2_2_1) : "r"(qn_2_1[6]), "r"(qn_2_1[6]), "r"(b1_2_1));
                    uint32_t a3_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(a3_2_1) : "r"(qn_2_1[3]), "r"(qn_2_1[3]), "r"(a2_2_1));
                    uint32_t b3_2_1;
                    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(b3_2_1) : "r"(qn_2_1[7]), "r"(qn_2_1[7]), "r"(b2_2_1));
                    uint32_t v_18;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(v_18) : "r"(a3_2_1), "r"(b3_2_1));
                    float w_2_1[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.f16 %0, lo, %3;\n\t"
                        "add.rn.f32.f16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(w_2_1[0]), "=&f"(w_2_1[1]) : "r"(v_18), "f"(0.0f), "f"(0.0f));
                    qq_2_1[3] = w_2_1[0] + w_2_1[1];
#if __CUDA_ARCH__ == 1030
                    float _rcp_175 = approx_rcp(qq_2_1[3]);
                    sf_fit[3] = ((g_live_5 != 0 && qq_2_1[3] > 0.0f) ? sf_ex_5 * qv_2_1[3] * _rcp_175 : 0.0f);
#else
                    float _rcp_171 = approx_rcp(qq_2_1[3]);
                    sf_fit[3] = ((g_live_5 != 0 && qq_2_1[3] > 0.0f) ? sf_ex_5 * qv_2_1[3] * _rcp_171 : 0.0f);
#endif
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
                xch_w[xs * 128 + mine] = sf_mine;
                asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                unsigned int sf_other = xch_w[xs * 128 + other];
                sf_pair[0] = sf_mine;
                sf_pair[1] = sf_other;
                if (grp == 0) {
                    if (jj > 0) {
                        int _parity_l_ready = (int)((unsigned int)jj % 2) ^ 1;
                        mbarrier_wait(l_ready_addr + 8, _parity_l_ready);
                        l_sum = xch_s[128 + mine];
                    }
                } else {
                    int _parity_l_ready_1 = (int)((unsigned int)jj % 2);
                    mbarrier_wait(l_ready_addr, _parity_l_ready_1);
                    l_sum = xch_s[mine];
                }
                l_sum = __fmaf_rn(l_sum, alpha, b_sums[0] * b_weights[0]);
                l_sum = __fmaf_rn(b_sums[1], b_weights[1], l_sum);
                l_sum = __fmaf_rn(b_sums[2], b_weights[2], l_sum);
                l_sum = __fmaf_rn(b_sums[3], b_weights[3], l_sum);
                xch_s[grp * 128 + mine] = l_sum;
                mbarrier_arrive(l_ready_addr + (grp) * 8);
                if (grp == 0) {
                    if (jj > 0) {
                        int _parity_pv_done = (int)((unsigned int)jj % 2) ^ 1;
                        mbarrier_wait(pv_done_addr + 8, _parity_pv_done);
                    }
                } else {
                    int _parity_pv_done_1 = (int)((unsigned int)jj % 2);
                    mbarrier_wait(pv_done_addr, _parity_pv_done_1);
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                int _warp_vote_176 = __any_sync(0xFFFFFFFF, rescale != 0);
                if (_warp_vote_176 != 0) {
#else
                int _warp_vote_172 = __any_sync(0xFFFFFFFF, rescale != 0);
                if (_warp_vote_172 != 0) {
#endif
                    float o_chunk_0[32];
                    unsigned int o_addr_0 = taddr + (unsigned int)lane_base_s;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_0[0]), "=f"(o_chunk_0[1]), "=f"(o_chunk_0[2]), "=f"(o_chunk_0[3]), "=f"(o_chunk_0[4]), "=f"(o_chunk_0[5]), "=f"(o_chunk_0[6]), "=f"(o_chunk_0[7]), "=f"(o_chunk_0[8]), "=f"(o_chunk_0[9]), "=f"(o_chunk_0[10]), "=f"(o_chunk_0[11]), "=f"(o_chunk_0[12]), "=f"(o_chunk_0[13]), "=f"(o_chunk_0[14]), "=f"(o_chunk_0[15]), "=f"(o_chunk_0[16]), "=f"(o_chunk_0[17]), "=f"(o_chunk_0[18]), "=f"(o_chunk_0[19]), "=f"(o_chunk_0[20]), "=f"(o_chunk_0[21]), "=f"(o_chunk_0[22]), "=f"(o_chunk_0[23]), "=f"(o_chunk_0[24]), "=f"(o_chunk_0[25]), "=f"(o_chunk_0[26]), "=f"(o_chunk_0[27]), "=f"(o_chunk_0[28]), "=f"(o_chunk_0[29]), "=f"(o_chunk_0[30]), "=f"(o_chunk_0[31])
                        : "r"(o_addr_0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_177 = {alpha, alpha};
#else
                    const float2 _scale2_173 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_0)[_ls], _scale2_177);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_0)[_ls], _scale2_173);
#endif
                    tmem_st_x32_f32(o_addr_0, o_chunk_0);
                    float o_chunk_1[32];
                    unsigned int o_addr_1 = taddr + (unsigned int)lane_base_s + 32;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_1[0]), "=f"(o_chunk_1[1]), "=f"(o_chunk_1[2]), "=f"(o_chunk_1[3]), "=f"(o_chunk_1[4]), "=f"(o_chunk_1[5]), "=f"(o_chunk_1[6]), "=f"(o_chunk_1[7]), "=f"(o_chunk_1[8]), "=f"(o_chunk_1[9]), "=f"(o_chunk_1[10]), "=f"(o_chunk_1[11]), "=f"(o_chunk_1[12]), "=f"(o_chunk_1[13]), "=f"(o_chunk_1[14]), "=f"(o_chunk_1[15]), "=f"(o_chunk_1[16]), "=f"(o_chunk_1[17]), "=f"(o_chunk_1[18]), "=f"(o_chunk_1[19]), "=f"(o_chunk_1[20]), "=f"(o_chunk_1[21]), "=f"(o_chunk_1[22]), "=f"(o_chunk_1[23]), "=f"(o_chunk_1[24]), "=f"(o_chunk_1[25]), "=f"(o_chunk_1[26]), "=f"(o_chunk_1[27]), "=f"(o_chunk_1[28]), "=f"(o_chunk_1[29]), "=f"(o_chunk_1[30]), "=f"(o_chunk_1[31])
                        : "r"(o_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_178 = {alpha, alpha};
#else
                    const float2 _scale2_174 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_1)[_ls], _scale2_178);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_1)[_ls], _scale2_174);
#endif
                    tmem_st_x32_f32(o_addr_1, o_chunk_1);
                    float o_chunk_2[32];
                    unsigned int o_addr_2 = taddr + (unsigned int)lane_base_s + 64;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_2[0]), "=f"(o_chunk_2[1]), "=f"(o_chunk_2[2]), "=f"(o_chunk_2[3]), "=f"(o_chunk_2[4]), "=f"(o_chunk_2[5]), "=f"(o_chunk_2[6]), "=f"(o_chunk_2[7]), "=f"(o_chunk_2[8]), "=f"(o_chunk_2[9]), "=f"(o_chunk_2[10]), "=f"(o_chunk_2[11]), "=f"(o_chunk_2[12]), "=f"(o_chunk_2[13]), "=f"(o_chunk_2[14]), "=f"(o_chunk_2[15]), "=f"(o_chunk_2[16]), "=f"(o_chunk_2[17]), "=f"(o_chunk_2[18]), "=f"(o_chunk_2[19]), "=f"(o_chunk_2[20]), "=f"(o_chunk_2[21]), "=f"(o_chunk_2[22]), "=f"(o_chunk_2[23]), "=f"(o_chunk_2[24]), "=f"(o_chunk_2[25]), "=f"(o_chunk_2[26]), "=f"(o_chunk_2[27]), "=f"(o_chunk_2[28]), "=f"(o_chunk_2[29]), "=f"(o_chunk_2[30]), "=f"(o_chunk_2[31])
                        : "r"(o_addr_2));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_179 = {alpha, alpha};
#else
                    const float2 _scale2_175 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_2)[_ls], _scale2_179);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_2)[_ls], _scale2_175);
#endif
                    tmem_st_x32_f32(o_addr_2, o_chunk_2);
                    float o_chunk_3[32];
                    unsigned int o_addr_3 = taddr + (unsigned int)lane_base_s + 96;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_3[0]), "=f"(o_chunk_3[1]), "=f"(o_chunk_3[2]), "=f"(o_chunk_3[3]), "=f"(o_chunk_3[4]), "=f"(o_chunk_3[5]), "=f"(o_chunk_3[6]), "=f"(o_chunk_3[7]), "=f"(o_chunk_3[8]), "=f"(o_chunk_3[9]), "=f"(o_chunk_3[10]), "=f"(o_chunk_3[11]), "=f"(o_chunk_3[12]), "=f"(o_chunk_3[13]), "=f"(o_chunk_3[14]), "=f"(o_chunk_3[15]), "=f"(o_chunk_3[16]), "=f"(o_chunk_3[17]), "=f"(o_chunk_3[18]), "=f"(o_chunk_3[19]), "=f"(o_chunk_3[20]), "=f"(o_chunk_3[21]), "=f"(o_chunk_3[22]), "=f"(o_chunk_3[23]), "=f"(o_chunk_3[24]), "=f"(o_chunk_3[25]), "=f"(o_chunk_3[26]), "=f"(o_chunk_3[27]), "=f"(o_chunk_3[28]), "=f"(o_chunk_3[29]), "=f"(o_chunk_3[30]), "=f"(o_chunk_3[31])
                        : "r"(o_addr_3));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_180 = {alpha, alpha};
#else
                    const float2 _scale2_176 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_3)[_ls], _scale2_180);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_3)[_ls], _scale2_176);
#endif
                    tmem_st_x32_f32(o_addr_3, o_chunk_3);
                    float o_chunk_4[32];
                    unsigned int o_addr_4 = taddr + (unsigned int)lane_base_s + 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_4[0]), "=f"(o_chunk_4[1]), "=f"(o_chunk_4[2]), "=f"(o_chunk_4[3]), "=f"(o_chunk_4[4]), "=f"(o_chunk_4[5]), "=f"(o_chunk_4[6]), "=f"(o_chunk_4[7]), "=f"(o_chunk_4[8]), "=f"(o_chunk_4[9]), "=f"(o_chunk_4[10]), "=f"(o_chunk_4[11]), "=f"(o_chunk_4[12]), "=f"(o_chunk_4[13]), "=f"(o_chunk_4[14]), "=f"(o_chunk_4[15]), "=f"(o_chunk_4[16]), "=f"(o_chunk_4[17]), "=f"(o_chunk_4[18]), "=f"(o_chunk_4[19]), "=f"(o_chunk_4[20]), "=f"(o_chunk_4[21]), "=f"(o_chunk_4[22]), "=f"(o_chunk_4[23]), "=f"(o_chunk_4[24]), "=f"(o_chunk_4[25]), "=f"(o_chunk_4[26]), "=f"(o_chunk_4[27]), "=f"(o_chunk_4[28]), "=f"(o_chunk_4[29]), "=f"(o_chunk_4[30]), "=f"(o_chunk_4[31])
                        : "r"(o_addr_4));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_181 = {alpha, alpha};
#else
                    const float2 _scale2_177 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_4)[_ls], _scale2_181);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_4)[_ls], _scale2_177);
#endif
                    tmem_st_x32_f32(o_addr_4, o_chunk_4);
                    float o_chunk_5[32];
                    unsigned int o_addr_5 = taddr + (unsigned int)lane_base_s + 160;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_5[0]), "=f"(o_chunk_5[1]), "=f"(o_chunk_5[2]), "=f"(o_chunk_5[3]), "=f"(o_chunk_5[4]), "=f"(o_chunk_5[5]), "=f"(o_chunk_5[6]), "=f"(o_chunk_5[7]), "=f"(o_chunk_5[8]), "=f"(o_chunk_5[9]), "=f"(o_chunk_5[10]), "=f"(o_chunk_5[11]), "=f"(o_chunk_5[12]), "=f"(o_chunk_5[13]), "=f"(o_chunk_5[14]), "=f"(o_chunk_5[15]), "=f"(o_chunk_5[16]), "=f"(o_chunk_5[17]), "=f"(o_chunk_5[18]), "=f"(o_chunk_5[19]), "=f"(o_chunk_5[20]), "=f"(o_chunk_5[21]), "=f"(o_chunk_5[22]), "=f"(o_chunk_5[23]), "=f"(o_chunk_5[24]), "=f"(o_chunk_5[25]), "=f"(o_chunk_5[26]), "=f"(o_chunk_5[27]), "=f"(o_chunk_5[28]), "=f"(o_chunk_5[29]), "=f"(o_chunk_5[30]), "=f"(o_chunk_5[31])
                        : "r"(o_addr_5));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_182 = {alpha, alpha};
#else
                    const float2 _scale2_178 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_5)[_ls], _scale2_182);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_5)[_ls], _scale2_178);
#endif
                    tmem_st_x32_f32(o_addr_5, o_chunk_5);
                    float o_chunk_6[32];
                    unsigned int o_addr_6 = taddr + (unsigned int)lane_base_s + 192;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_6[0]), "=f"(o_chunk_6[1]), "=f"(o_chunk_6[2]), "=f"(o_chunk_6[3]), "=f"(o_chunk_6[4]), "=f"(o_chunk_6[5]), "=f"(o_chunk_6[6]), "=f"(o_chunk_6[7]), "=f"(o_chunk_6[8]), "=f"(o_chunk_6[9]), "=f"(o_chunk_6[10]), "=f"(o_chunk_6[11]), "=f"(o_chunk_6[12]), "=f"(o_chunk_6[13]), "=f"(o_chunk_6[14]), "=f"(o_chunk_6[15]), "=f"(o_chunk_6[16]), "=f"(o_chunk_6[17]), "=f"(o_chunk_6[18]), "=f"(o_chunk_6[19]), "=f"(o_chunk_6[20]), "=f"(o_chunk_6[21]), "=f"(o_chunk_6[22]), "=f"(o_chunk_6[23]), "=f"(o_chunk_6[24]), "=f"(o_chunk_6[25]), "=f"(o_chunk_6[26]), "=f"(o_chunk_6[27]), "=f"(o_chunk_6[28]), "=f"(o_chunk_6[29]), "=f"(o_chunk_6[30]), "=f"(o_chunk_6[31])
                        : "r"(o_addr_6));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_183 = {alpha, alpha};
#else
                    const float2 _scale2_179 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_6)[_ls], _scale2_183);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_6)[_ls], _scale2_179);
#endif
                    tmem_st_x32_f32(o_addr_6, o_chunk_6);
                    float o_chunk_7[32];
                    unsigned int o_addr_7 = taddr + (unsigned int)lane_base_s + 224;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_chunk_7[0]), "=f"(o_chunk_7[1]), "=f"(o_chunk_7[2]), "=f"(o_chunk_7[3]), "=f"(o_chunk_7[4]), "=f"(o_chunk_7[5]), "=f"(o_chunk_7[6]), "=f"(o_chunk_7[7]), "=f"(o_chunk_7[8]), "=f"(o_chunk_7[9]), "=f"(o_chunk_7[10]), "=f"(o_chunk_7[11]), "=f"(o_chunk_7[12]), "=f"(o_chunk_7[13]), "=f"(o_chunk_7[14]), "=f"(o_chunk_7[15]), "=f"(o_chunk_7[16]), "=f"(o_chunk_7[17]), "=f"(o_chunk_7[18]), "=f"(o_chunk_7[19]), "=f"(o_chunk_7[20]), "=f"(o_chunk_7[21]), "=f"(o_chunk_7[22]), "=f"(o_chunk_7[23]), "=f"(o_chunk_7[24]), "=f"(o_chunk_7[25]), "=f"(o_chunk_7[26]), "=f"(o_chunk_7[27]), "=f"(o_chunk_7[28]), "=f"(o_chunk_7[29]), "=f"(o_chunk_7[30]), "=f"(o_chunk_7[31])
                        : "r"(o_addr_7));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_184 = {alpha, alpha};
#else
                    const float2 _scale2_180 = {alpha, alpha};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_7)[_ls], _scale2_184);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk_7)[_ls], _scale2_180);
#endif
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
                _phase_pv_done ^= 1;
            }
            int last_grp = (my_n_1 - 1) % 2;
            if (last_grp == grp) {
                xch_l[mine] = l_sum;
                asm volatile("barrier.sync %0, 64;" :: "r"(xbar) : "memory");
                float l_row = l_sum + xch_l[other];
                float gk_e = *kv_gs;
#if __CUDA_ARCH__ == 1030
                float _rcp_185 = approx_rcp(gk_e * l_row);
                float inv = ((l_row > 0.0f) ? 0.005952381f * _rcp_185 : 0.0f);
#else
                float _rcp_181 = approx_rcp(gk_e * l_row);
                float inv = ((l_row > 0.0f) ? 0.005952381f * _rcp_181 : 0.0f);
#endif
                int direct = num_split == 1;
                int row_g = row_base_1 + row_local;
                int o_base = (row_g * num_split + split_i_1) * 512 + 128 * half;
                mbarrier_wait(pv_done_addr + (grp) * 8, _phase_pv_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float o_vals_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_vals_0[0]), "=f"(o_vals_0[1]), "=f"(o_vals_0[2]), "=f"(o_vals_0[3]), "=f"(o_vals_0[4]), "=f"(o_vals_0[5]), "=f"(o_vals_0[6]), "=f"(o_vals_0[7]), "=f"(o_vals_0[8]), "=f"(o_vals_0[9]), "=f"(o_vals_0[10]), "=f"(o_vals_0[11]), "=f"(o_vals_0[12]), "=f"(o_vals_0[13]), "=f"(o_vals_0[14]), "=f"(o_vals_0[15]), "=f"(o_vals_0[16]), "=f"(o_vals_0[17]), "=f"(o_vals_0[18]), "=f"(o_vals_0[19]), "=f"(o_vals_0[20]), "=f"(o_vals_0[21]), "=f"(o_vals_0[22]), "=f"(o_vals_0[23]), "=f"(o_vals_0[24]), "=f"(o_vals_0[25]), "=f"(o_vals_0[26]), "=f"(o_vals_0[27]), "=f"(o_vals_0[28]), "=f"(o_vals_0[29]), "=f"(o_vals_0[30]), "=f"(o_vals_0[31])
                    : "r"(taddr + (unsigned int)lane_base_s));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_186 = {inv, inv};
#else
                const float2 _scale2_182 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_0)[_ls], _scale2_186);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_0)[_ls], _scale2_182);
#endif
                unsigned int o_bf16_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_0[_lp*2 + 0], o_vals_0[_lp*2+1 + 0]));
                    o_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_0 = 0;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_187 = {inv, inv};
#else
                const float2 _scale2_183 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_1)[_ls], _scale2_187);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_1)[_ls], _scale2_183);
#endif
                unsigned int o_bf16_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_1[_lp*2 + 0], o_vals_1[_lp*2+1 + 0]));
                    o_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_1 = 32;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_188 = {inv, inv};
#else
                const float2 _scale2_184 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_2)[_ls], _scale2_188);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_2)[_ls], _scale2_184);
#endif
                unsigned int o_bf16_2[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_2[_lp*2 + 0], o_vals_2[_lp*2+1 + 0]));
                    o_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_2 = 64;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_189 = {inv, inv};
#else
                const float2 _scale2_185 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_3)[_ls], _scale2_189);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_3)[_ls], _scale2_185);
#endif
                unsigned int o_bf16_3[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_3[_lp*2 + 0], o_vals_3[_lp*2+1 + 0]));
                    o_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_3 = 96;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_190 = {inv, inv};
#else
                const float2 _scale2_186 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_4)[_ls], _scale2_190);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_4)[_ls], _scale2_186);
#endif
                unsigned int o_bf16_4[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_4[_lp*2 + 0], o_vals_4[_lp*2+1 + 0]));
                    o_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_4 = 256;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_191 = {inv, inv};
#else
                const float2 _scale2_187 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_5)[_ls], _scale2_191);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_5)[_ls], _scale2_187);
#endif
                unsigned int o_bf16_5[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_5[_lp*2 + 0], o_vals_5[_lp*2+1 + 0]));
                    o_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_5 = 288;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_192 = {inv, inv};
#else
                const float2 _scale2_188 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_6)[_ls], _scale2_192);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_6)[_ls], _scale2_188);
#endif
                unsigned int o_bf16_6[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_6[_lp*2 + 0], o_vals_6[_lp*2+1 + 0]));
                    o_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_6 = 320;
                if (row_ok_1 != 0) {
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
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_193 = {inv, inv};
#else
                const float2 _scale2_189 = {inv, inv};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_7)[_ls], _scale2_193);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_vals_7)[_ls], _scale2_189);
#endif
                unsigned int o_bf16_7[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_vals_7[_lp*2 + 0], o_vals_7[_lp*2+1 + 0]));
                    o_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                int dim0_7 = 352;
                if (row_ok_1 != 0) {
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
                if (row_ok_1 != 0 && half == 0 && direct == 0) {
                    partial_max[row_g * num_split + split_i_1] = m;
                    partial_sum[row_g * num_split + split_i_1] = l_row;
                }
            } else if (n_grp > 0) {
                mbarrier_wait(pv_done_addr + (grp) * 8, _phase_pv_done);
            }
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
            int src_row = (lane >> 2 & 3) * 8 + (lane >> 2 & 1 ^ lane >> 4) * 4 + (lane & 3);
            int raw_lane = src_row * 64 + (xw ^ src_row >> 1 & 3) * 16;
            unsigned int sel_u = (((lane & 1) == 1) ? 8292 : 25632);
            unsigned int sel_v = (((lane & 1) == 1) ? 12661 : 30001);
            int swap_x = lane >> 3 & 1 ^ lane >> 2 & 1;
            int dim_row = xw * 32 + (lane & 7) * 2 + (lane >> 4) * 16 + swap_x;
            int raw_x = 0;
            int vt_x = 0;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_vt_empty = 1;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < my_n_2; j_1++) {
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
                unsigned int pw_0[4];
                uint32_t u_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_0) : "r"(lw[0]), "r"(lw[2]), "r"(sel_u));
                uint32_t v_0;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_0) : "r"(lw[0]), "r"(lw[2]), "r"(sel_v));
                unsigned int even_0 = u_0 & 252645135 | v_0 << 4 & 4042322160u;
                unsigned int odd_0 = u_0 >> 4 & 252645135 | v_0 & 4042322160u;
                pw_0[0] = ((lane >= 16) ? odd_0 : even_0);
                pw_0[1] = ((lane >= 16) ? even_0 : odd_0);
                uint32_t u_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_1) : "r"(lw[1]), "r"(lw[3]), "r"(sel_u));
                uint32_t v_1_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_1_1) : "r"(lw[1]), "r"(lw[3]), "r"(sel_v));
                unsigned int even_1 = u_1 & 252645135 | v_1_1 << 4 & 4042322160u;
                unsigned int odd_1 = u_1 >> 4 & 252645135 | v_1_1 & 4042322160u;
                pw_0[2] = ((lane >= 16) ? odd_1 : even_1);
                pw_0[3] = ((lane >= 16) ? even_1 : odd_1);
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (0 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_0[3]))
                    : "memory");
                unsigned int pw_1[4];
                uint32_t u_2;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_2) : "r"(lw[4]), "r"(lw[6]), "r"(sel_u));
                uint32_t v_2_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_2_1) : "r"(lw[4]), "r"(lw[6]), "r"(sel_v));
                unsigned int even_2 = u_2 & 252645135 | v_2_1 << 4 & 4042322160u;
                unsigned int odd_2 = u_2 >> 4 & 252645135 | v_2_1 & 4042322160u;
                pw_1[0] = ((lane >= 16) ? odd_2 : even_2);
                pw_1[1] = ((lane >= 16) ? even_2 : odd_2);
                uint32_t u_3;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_3) : "r"(lw[5]), "r"(lw[7]), "r"(sel_u));
                uint32_t v_3_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_3_1) : "r"(lw[5]), "r"(lw[7]), "r"(sel_v));
                unsigned int even_3 = u_3 & 252645135 | v_3_1 << 4 & 4042322160u;
                unsigned int odd_3 = u_3 >> 4 & 252645135 | v_3_1 & 4042322160u;
                pw_1[2] = ((lane >= 16) ? odd_3 : even_3);
                pw_1[3] = ((lane >= 16) ? even_3 : odd_3);
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (1 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_1[3]))
                    : "memory");
                unsigned int pw_2[4];
                uint32_t u_4;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_4) : "r"(lw[8]), "r"(lw[10]), "r"(sel_u));
                uint32_t v_4_1;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_4_1) : "r"(lw[8]), "r"(lw[10]), "r"(sel_v));
                unsigned int even_4 = u_4 & 252645135 | v_4_1 << 4 & 4042322160u;
                unsigned int odd_4 = u_4 >> 4 & 252645135 | v_4_1 & 4042322160u;
                pw_2[0] = ((lane >= 16) ? odd_4 : even_4);
                pw_2[1] = ((lane >= 16) ? even_4 : odd_4);
                uint32_t u_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_5) : "r"(lw[9]), "r"(lw[11]), "r"(sel_u));
                uint32_t v_5;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_5) : "r"(lw[9]), "r"(lw[11]), "r"(sel_v));
                unsigned int even_5 = u_5 & 252645135 | v_5 << 4 & 4042322160u;
                unsigned int odd_5 = u_5 >> 4 & 252645135 | v_5 & 4042322160u;
                pw_2[2] = ((lane >= 16) ? odd_5 : even_5);
                pw_2[3] = ((lane >= 16) ? even_5 : odd_5);
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (2 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_2[3]))
                    : "memory");
                unsigned int pw_3[4];
                uint32_t u_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_6) : "r"(lw[12]), "r"(lw[14]), "r"(sel_u));
                uint32_t v_6;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_6) : "r"(lw[12]), "r"(lw[14]), "r"(sel_v));
                unsigned int even_6 = u_6 & 252645135 | v_6 << 4 & 4042322160u;
                unsigned int odd_6 = u_6 >> 4 & 252645135 | v_6 & 4042322160u;
                pw_3[0] = ((lane >= 16) ? odd_6 : even_6);
                pw_3[1] = ((lane >= 16) ? even_6 : odd_6);
                uint32_t u_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_7) : "r"(lw[13]), "r"(lw[15]), "r"(sel_u));
                uint32_t v_7;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_7) : "r"(lw[13]), "r"(lw[15]), "r"(sel_v));
                unsigned int even_7 = u_7 & 252645135 | v_7 << 4 & 4042322160u;
                unsigned int odd_7 = u_7 >> 4 & 252645135 | v_7 & 4042322160u;
                pw_3[2] = ((lane >= 16) ? odd_7 : even_7);
                pw_3[3] = ((lane >= 16) ? even_7 : odd_7);
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)(vt_x * 2 * 8192 + dim_row * 64 + (3 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_3[3]))
                    : "memory");
                unsigned int pw_4[4];
                uint32_t u_8;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_8) : "r"(lw[16]), "r"(lw[18]), "r"(sel_u));
                uint32_t v_8;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_8) : "r"(lw[16]), "r"(lw[18]), "r"(sel_v));
                unsigned int even_8 = u_8 & 252645135 | v_8 << 4 & 4042322160u;
                unsigned int odd_8 = u_8 >> 4 & 252645135 | v_8 & 4042322160u;
                pw_4[0] = ((lane >= 16) ? odd_8 : even_8);
                pw_4[1] = ((lane >= 16) ? even_8 : odd_8);
                uint32_t u_9;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_9) : "r"(lw[17]), "r"(lw[19]), "r"(sel_u));
                uint32_t v_9;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_9) : "r"(lw[17]), "r"(lw[19]), "r"(sel_v));
                unsigned int even_9 = u_9 & 252645135 | v_9 << 4 & 4042322160u;
                unsigned int odd_9 = u_9 >> 4 & 252645135 | v_9 & 4042322160u;
                pw_4[2] = ((lane >= 16) ? odd_9 : even_9);
                pw_4[3] = ((lane >= 16) ? even_9 : odd_9);
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (0 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_4[3]))
                    : "memory");
                unsigned int pw_5[4];
                uint32_t u_10;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_10) : "r"(lw[20]), "r"(lw[22]), "r"(sel_u));
                uint32_t v_10;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_10) : "r"(lw[20]), "r"(lw[22]), "r"(sel_v));
                unsigned int even_10 = u_10 & 252645135 | v_10 << 4 & 4042322160u;
                unsigned int odd_10 = u_10 >> 4 & 252645135 | v_10 & 4042322160u;
                pw_5[0] = ((lane >= 16) ? odd_10 : even_10);
                pw_5[1] = ((lane >= 16) ? even_10 : odd_10);
                uint32_t u_11;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_11) : "r"(lw[21]), "r"(lw[23]), "r"(sel_u));
                uint32_t v_11;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_11) : "r"(lw[21]), "r"(lw[23]), "r"(sel_v));
                unsigned int even_11 = u_11 & 252645135 | v_11 << 4 & 4042322160u;
                unsigned int odd_11 = u_11 >> 4 & 252645135 | v_11 & 4042322160u;
                pw_5[2] = ((lane >= 16) ? odd_11 : even_11);
                pw_5[3] = ((lane >= 16) ? even_11 : odd_11);
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (1 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_5[3]))
                    : "memory");
                unsigned int pw_6[4];
                uint32_t u_12;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_12) : "r"(lw[24]), "r"(lw[26]), "r"(sel_u));
                uint32_t v_12;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_12) : "r"(lw[24]), "r"(lw[26]), "r"(sel_v));
                unsigned int even_12 = u_12 & 252645135 | v_12 << 4 & 4042322160u;
                unsigned int odd_12 = u_12 >> 4 & 252645135 | v_12 & 4042322160u;
                pw_6[0] = ((lane >= 16) ? odd_12 : even_12);
                pw_6[1] = ((lane >= 16) ? even_12 : odd_12);
                uint32_t u_13;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_13) : "r"(lw[25]), "r"(lw[27]), "r"(sel_u));
                uint32_t v_13;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_13) : "r"(lw[25]), "r"(lw[27]), "r"(sel_v));
                unsigned int even_13 = u_13 & 252645135 | v_13 << 4 & 4042322160u;
                unsigned int odd_13 = u_13 >> 4 & 252645135 | v_13 & 4042322160u;
                pw_6[2] = ((lane >= 16) ? odd_13 : even_13);
                pw_6[3] = ((lane >= 16) ? even_13 : odd_13);
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (2 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_6[3]))
                    : "memory");
                unsigned int pw_7[4];
                uint32_t u_14;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_14) : "r"(lw[28]), "r"(lw[30]), "r"(sel_u));
                uint32_t v_14;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_14) : "r"(lw[28]), "r"(lw[30]), "r"(sel_v));
                unsigned int even_14 = u_14 & 252645135 | v_14 << 4 & 4042322160u;
                unsigned int odd_14 = u_14 >> 4 & 252645135 | v_14 & 4042322160u;
                pw_7[0] = ((lane >= 16) ? odd_14 : even_14);
                pw_7[1] = ((lane >= 16) ? even_14 : odd_14);
                uint32_t u_15;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(u_15) : "r"(lw[29]), "r"(lw[31]), "r"(sel_u));
                uint32_t v_15;
                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v_15) : "r"(lw[29]), "r"(lw[31]), "r"(sel_v));
                unsigned int even_15 = u_15 & 252645135 | v_15 << 4 & 4042322160u;
                unsigned int odd_15 = u_15 >> 4 & 252645135 | v_15 & 4042322160u;
                pw_7[2] = ((lane >= 16) ? odd_15 : even_15);
                pw_7[3] = ((lane >= 16) ? even_15 : odd_15);
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(smem_vt_addr + (unsigned int)((vt_x * 2 + 1) * 8192 + dim_row * 64 + (3 ^ dim_row >> 1 & 3) * 16));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&pw_7[3]))
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
            int vs_l = 0;
            int n_last_l = my_n_3 - 1;
            unsigned int kb_next[20];
            int pid = page_table[b_i_3 * page_stride + my_start_3];
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
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_raw_empty = 1;
            unsigned int _phase_vs_empty = 1;
            #pragma unroll 1
            for (int j_2 = 0; j_2 < my_n_3; j_2++) {
                int pid_0 = page_table[b_i_3 * page_stride + (my_start_3 + j_2)];
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
                int pid_1 = page_table[b_i_3 * page_stride + (my_start_3 + ((j1 < n_last_l) ? j1 : n_last_l))];
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
                    tma_3d_gmem2smem(smem_raw_addr + (unsigned int)(raw_l * 2 * 8192), (&kv_v), 64 * cta_rank, 0, page_l, raw_full_addr + (raw_l) * 8);
                    tma_3d_gmem2smem(smem_raw_addr + (unsigned int)((raw_l * 2 + 1) * 8192), (&kv_v), 64 * (2 + cta_rank), 0, page_l, raw_full_addr + (raw_l) * 8);
                }
                raw_l += 1;
                if (raw_l == 2) { raw_l = 0; _phase_raw_empty ^= 1; }
                mbarrier_wait(vs_empty_addr + (vs_l) * 8, _phase_vs_empty);
                int page_keys_l = keys_3 - 128 * (my_start_3 + j_2);
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
                int s_m = 0;
                int se_m = 0;
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
                for (int j_3 = 0; j_3 < my_n_4; j_3++) {
                    uint32_t _elected_11 = elect_sync();
                    int _smem_desc_lo_12 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_12);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 0), "r"(tmem_sfb + 0), "r"(0), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_13 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_13);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 8), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 4), "r"(tmem_sfb + 4), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_14 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 1) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_14);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 8), "r"(tmem_sfb + 8), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_15 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 1) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_15);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 24), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 12), "r"(tmem_sfb + 12), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_16 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 2) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_16);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 16), "r"(tmem_sfb + 16), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_17 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 2) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_17);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 40), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 20), "r"(tmem_sfb + 20), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_18 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 3) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_18);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 24), "r"(tmem_sfb + 24), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_19 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 3) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_19);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 56), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 28), "r"(tmem_sfb + 28), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_20 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_20);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 64), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 32), "r"(tmem_sfb + 32), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_21 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_21);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 72), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 36), "r"(tmem_sfb + 32), "r"(1), "r"(_elected_11) : "memory");
                    }
                    int _smem_desc_lo_22 = (((smem_k_addr + 32) >> 4) & 0x3FFF) + (k_m * 5 + 4) * 256;
                    {
                        const uint32_t b_desc_lo = make_warp_uniform(_smem_desc_lo_22);
                        asm volatile("{\n\t.reg .pred leader, p;\n\tsetp.ne.b32 leader, %7, 0;\n\tsetp.ne.b32 p, %6, 0;\n\t@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (256))), "r"(tmem_q + 64), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8200480U), "r"(tmem_sfa + 32), "r"(tmem_sfb + 36), "r"(1), "r"(_elected_11) : "memory");
                    }
                    tcgen05_commit_cg2_multicast_leader(s_full_addr + (s_m) * 8, (uint16_t)(3), _elected_11);
                    tcgen05_commit_cg2_multicast_leader(k_empty_addr + (k_m) * 8, (uint16_t)(3), _elected_11);
                    k_m += 1;
                    if (k_m == 2) { k_m = 0; _phase_k_full ^= 1; }
                    if (j_3 + 1 < my_n_4) {
                        mbarrier_wait(s_full_addr + (s_m) * 8, _phase_s_full_1);
                        mbarrier_wait_cluster_hint(k_full_addr + (k_m) * 8, _phase_k_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_23 = elect_sync();
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_24 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 0)), "l"(_tcgen05_cp_desc_24), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_25 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 128))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_25), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_26 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 256))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_26), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_27 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 384))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_27), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_28 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 512))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_28), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_29 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 640))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_29), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_30 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 768))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_30), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_31 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 896))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_31), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_32 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 1024))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 32)), "l"(_tcgen05_cp_desc_32), "r"(_elected_23)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_33 = ((((uint64_t)(smem_ks_addr + (unsigned int)(k_m * 10240 + 1152))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1280)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                                "@leader tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n\t}\n"
                                :: "r"((uint32_t)(tmem_sfb + 36)), "l"(_tcgen05_cp_desc_33), "r"(_elected_23)
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
    // ---- Role: pvi ----
    if (warp == 14) {
        { // pvi_main
            unsigned int _phase_vt_full = 0;
            unsigned int _phase_vs_full = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_pv_done_1 = 0;
            if (cta_rank == 0) {
                int split_i_5 = blockIdx.x / 2;
                int b_i_5 = blockIdx.z;
                int rows_b_5 = q_len * num_heads;
                int row0_5 = blockIdx.y * 128;
                int rows_left_5 = rows_b_5 - row0_5;
                int rows_valid_5 = ((rows_left_5 > 128) ? 128 : rows_left_5);
                int row_base_5 = b_i_5 * rows_b_5 + row0_5;
                int kv_len_i_5 = kv_lens[b_i_5];
                int t_last_5 = (row0_5 + rows_valid_5 - 1) / num_heads;
                int kv_end_raw_5 = kv_len_i_5 - (q_len - 1 - t_last_5);
                int kv_end_5 = ((kv_end_raw_5 < 0) ? 0 : kv_end_raw_5);
                int n_pages_5 = (kv_end_5 + 128 - 1) / 128;
                int pps_5 = (n_pages_5 + num_split - 1) / num_split;
                int p_start_5 = split_i_5 * pps_5;
                int p_end_raw_5 = p_start_5 + pps_5;
                int p_end_5 = ((p_end_raw_5 > n_pages_5) ? n_pages_5 : p_end_raw_5);
                int n_raw_5 = p_end_5 - p_start_5;
                int empty_5 = ((n_raw_5 < 1) ? 1 : 0);
                int my_n_5 = ((n_raw_5 < 1) ? 1 : n_raw_5);
                int my_start_5 = ((n_raw_5 < 1) ? 0 : p_start_5);
                int keys_5 = ((empty_5 != 0) ? 0 : kv_end_5);
                int vt_m = 0;
                int vs_m = 0;
                int p_m = 0;
                int pv_m = 0;
                #pragma unroll 1
                for (int jv = 0; jv < my_n_5; jv++) {
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
                    int _smem_desc_lo_11 = (((smem_vt_addr) >> 4) & 0x3FFF) + (vt_m * 2 + 1) * 512;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x80004020 << 32);

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
                    mbarrier_wait(pv_done_addr + (pv_m) * 8, _phase_pv_done_1);
                    pv_m += 1;
                    if (pv_m == 2) { pv_m = 0; _phase_pv_done_1 ^= 1; }
                }
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
