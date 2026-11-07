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



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_gqa_det_train_bd93d4235142169368da(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap dO, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, float* __restrict__ lse_log2, float* __restrict__ dpsum, __nv_bfloat16* __restrict__ ds_ws, int* __restrict__ tile_base, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, float* __restrict__ dv_out, int num_segments, int h_kv, int group, int rows_pad, float softmax_scale_log2, int n0, int h0, int tile_stride)
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
    #define kv_full_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 8)
    #define q_empty_addr (mbar_base + 24)
    #define do_full_addr (mbar_base + 40)
    #define do_empty_addr (mbar_base + 56)
    #define stats_full_addr (mbar_base + 72)
    #define stats_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 104)
    #define dp_full_addr (mbar_base + 120)
    #define ds_full_addr (mbar_base + 136)
    #define st_full_addr (mbar_base + 144)
    #define ds_empty_addr (mbar_base + 152)
    #define dkv_full_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int sq_addr = smem + 66560;
    __nv_bfloat16* sdo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int sdo_addr = smem + 132096;
    __nv_bfloat16* sdot = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int sdot_addr = smem + 132096;
    __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sk_addr = smem + 1024;
    __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int sv_addr = smem + 33792;
    __nv_bfloat16* sp = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int sp_addr = smem + 197632;
    __nv_bfloat16* sdst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int sdst_addr = smem + 214016;
    float* sst = reinterpret_cast<float*>(smem_raw + 230400);
    const int sst_addr = smem + 230400;

    // Kernel setup ops
    if (warp == 9) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 9) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dO))) : "memory"); }
    if (warp == 9) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 9) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (21 barriers at smem_raw[0..168))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // kv_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // --- pipeline 'do_pipe' ---
            // do_full: 2 barriers, init_count=2
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // do_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'st_pipe' ---
            // stats_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // stats_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            // --- pipeline 'tm_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // dp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // ds_full: 1 barriers, init_count=512
            mbarrier_init(smem + 136, 512);
            // st_full: 1 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            // ds_empty: 1 barriers, init_count=2
            mbarrier_init(smem + 152, 2);
            // dkv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 8) {
        int _tmem_hold = smem + 168;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_dv = taddr;
    const unsigned int tmem_s = taddr + 128;
    const unsigned int tmem_dp = taddr + 256;

    // ---- Role: compute ----
    if (warp <= 7) {
        { // compute_main
            int seg = blockIdx.z;
            int n_blk = n0 + (blockIdx.x >> 1);
            int q0 = cu_seqlens_q[seg];
            int q_len = cu_seqlens_q[seg + 1] - q0;
            int k0 = cu_seqlens_k[seg];
            int k_len = cu_seqlens_k[seg + 1] - k0;
            int found = ((n_blk * 128 < k_len) ? 1 : 0);
            int off = k_len - q_len;
            int first_key = n_blk * 128 - off;
            int m_min = ((first_key > 0) ? first_key : 0) >> 6;
            int m_max = q_len + 63 >> 6;
            int num_m = ((found != 0) ? m_max - m_min : 0);
            int row_off = q0 * 2 + seg * 128 >> 7 << 7;
            int rank = cta_rank & 1;
            int h = h0 + blockIdx.y;
            int n_groups = group >> 1;
            const int quarter = warp % 4;
            const int wg = warp >> 2;
            const int row = (quarter & 1) * 32 + lane;
            const int half = quarter >> 1;
            const int col0 = half * 64 + wg * 32;
            const unsigned int lane_base = (unsigned int)(quarter * 32 << 16);
            int key = n_blk * 128 + rank * 64 + row;
            int prow = half * 64 + row;
            int t = 0;
            unsigned int tc = 0;
            unsigned int sc = 0;
            unsigned int _phase_stats_full = 0;
            unsigned int _phase_s_full = 0;
            #pragma unroll 1
            for (int g = 0; g < n_groups; g++) {
                #pragma unroll 1
                for (int mi = 0; mi < num_m; mi++) {
                    int m = m_min + mi;
                    mbarrier_wait(stats_full_addr + (sc) * 8, _phase_stats_full);
                    mbarrier_wait(s_full_addr + (tc) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float s[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(s[0]), "=f"(s[1]), "=f"(s[2]), "=f"(s[3]), "=f"(s[4]), "=f"(s[5]), "=f"(s[6]), "=f"(s[7]), "=f"(s[8]), "=f"(s[9]), "=f"(s[10]), "=f"(s[11]), "=f"(s[12]), "=f"(s[13]), "=f"(s[14]), "=f"(s[15]), "=f"(s[16]), "=f"(s[17]), "=f"(s[18]), "=f"(s[19]), "=f"(s[20]), "=f"(s[21]), "=f"(s[22]), "=f"(s[23]), "=f"(s[24]), "=f"(s[25]), "=f"(s[26]), "=f"(s[27]), "=f"(s[28]), "=f"(s[29]), "=f"(s[30]), "=f"(s[31])
                        : "r"(taddr + (unsigned int)(128 + wg * 32) + tc * 64 + lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    int lo = (key - off - m * 64) * 2 - col0;
                    int hi = (q_len - m * 64) * 2 - col0;
                    int hi_k = ((key < k_len) ? hi : 0);
                    uint32_t _thresh_mask_0;
                    {
                        int _lim_1 = hi_k;
                        if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                        else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                        }
                    }
                    uint32_t _thresh_mask_2;
                    {
                        int _lim_3 = lo;
                        if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                        else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                        }
                    }
                    unsigned int keep = _thresh_mask_0 & ~_thresh_mask_2;
                    if (!(keep & (1u << 0))) s[0] = -CUDART_INF_F;
                    if (!(keep & (1u << 1))) s[1] = -CUDART_INF_F;
                    if (!(keep & (1u << 2))) s[2] = -CUDART_INF_F;
                    if (!(keep & (1u << 3))) s[3] = -CUDART_INF_F;
                    if (!(keep & (1u << 4))) s[4] = -CUDART_INF_F;
                    if (!(keep & (1u << 5))) s[5] = -CUDART_INF_F;
                    if (!(keep & (1u << 6))) s[6] = -CUDART_INF_F;
                    if (!(keep & (1u << 7))) s[7] = -CUDART_INF_F;
                    if (!(keep & (1u << 8))) s[8] = -CUDART_INF_F;
                    if (!(keep & (1u << 9))) s[9] = -CUDART_INF_F;
                    if (!(keep & (1u << 10))) s[10] = -CUDART_INF_F;
                    if (!(keep & (1u << 11))) s[11] = -CUDART_INF_F;
                    if (!(keep & (1u << 12))) s[12] = -CUDART_INF_F;
                    if (!(keep & (1u << 13))) s[13] = -CUDART_INF_F;
                    if (!(keep & (1u << 14))) s[14] = -CUDART_INF_F;
                    if (!(keep & (1u << 15))) s[15] = -CUDART_INF_F;
                    if (!(keep & (1u << 16))) s[16] = -CUDART_INF_F;
                    if (!(keep & (1u << 17))) s[17] = -CUDART_INF_F;
                    if (!(keep & (1u << 18))) s[18] = -CUDART_INF_F;
                    if (!(keep & (1u << 19))) s[19] = -CUDART_INF_F;
                    if (!(keep & (1u << 20))) s[20] = -CUDART_INF_F;
                    if (!(keep & (1u << 21))) s[21] = -CUDART_INF_F;
                    if (!(keep & (1u << 22))) s[22] = -CUDART_INF_F;
                    if (!(keep & (1u << 23))) s[23] = -CUDART_INF_F;
                    if (!(keep & (1u << 24))) s[24] = -CUDART_INF_F;
                    if (!(keep & (1u << 25))) s[25] = -CUDART_INF_F;
                    if (!(keep & (1u << 26))) s[26] = -CUDART_INF_F;
                    if (!(keep & (1u << 27))) s[27] = -CUDART_INF_F;
                    if (!(keep & (1u << 28))) s[28] = -CUDART_INF_F;
                    if (!(keep & (1u << 29))) s[29] = -CUDART_INF_F;
                    if (!(keep & (1u << 30))) s[30] = -CUDART_INF_F;
                    if (!(keep & (1u << 31))) s[31] = -CUDART_INF_F;
                    float lse[32];
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        lse[j] = sst[(int)sc * 256 + (col0 + j)];
                    }
                    float2 scale2 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    #pragma unroll
                    for (int pair = 0; pair < 16; pair++) {
                        float2 x = fma_sub_f32x2_rn_ftz(make_float2(s[pair * 2], s[pair * 2 + 1]), scale2, make_float2(lse[pair * 2], lse[pair * 2 + 1]));
                        s[pair * 2] = x.x;
                        s[pair * 2 + 1] = x.y;
                    }
                    unsigned int p16[16];
                    softmax_frag_exp2_cast(&s[0], p16, 0);
                    int _parity_ds_empty = (int)((unsigned int)t % 2) ^ 1;
                    mbarrier_wait_cluster_hint(ds_empty_addr, _parity_ds_empty, 10000000);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sp_addr + (unsigned int)(prow * 128 + (wg * 4 ^ prow & 7) * 16))), "r"(p16[0]), "r"(p16[1]), "r"(p16[2]), "r"(p16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sp_addr + (unsigned int)(prow * 128 + (wg * 4 + 1 ^ prow & 7) * 16))), "r"(p16[4]), "r"(p16[5]), "r"(p16[6]), "r"(p16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sp_addr + (unsigned int)(prow * 128 + (wg * 4 + 2 ^ prow & 7) * 16))), "r"(p16[8]), "r"(p16[9]), "r"(p16[10]), "r"(p16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sp_addr + (unsigned int)(prow * 128 + (wg * 4 + 3 ^ prow & 7) * 16))), "r"(p16[12]), "r"(p16[13]), "r"(p16[14]), "r"(p16[15]) : "memory");
                    mbarrier_wait(dp_full_addr + (tc) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float dp[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(dp[0]), "=f"(dp[1]), "=f"(dp[2]), "=f"(dp[3]), "=f"(dp[4]), "=f"(dp[5]), "=f"(dp[6]), "=f"(dp[7]), "=f"(dp[8]), "=f"(dp[9]), "=f"(dp[10]), "=f"(dp[11]), "=f"(dp[12]), "=f"(dp[13]), "=f"(dp[14]), "=f"(dp[15]), "=f"(dp[16]), "=f"(dp[17]), "=f"(dp[18]), "=f"(dp[19]), "=f"(dp[20]), "=f"(dp[21]), "=f"(dp[22]), "=f"(dp[23]), "=f"(dp[24]), "=f"(dp[25]), "=f"(dp[26]), "=f"(dp[27]), "=f"(dp[28]), "=f"(dp[29]), "=f"(dp[30]), "=f"(dp[31])
                        : "r"(taddr + (unsigned int)(256 + wg * 32) + tc * 64 + lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float dps[32];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 32; j_1++) {
                        dps[j_1] = sst[(int)sc * 256 + 128 + (col0 + j_1)];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(dp)[_ls], reinterpret_cast<const float2*>(dps)[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(dp)[_ls], reinterpret_cast<const float2*>(s)[_ls]);
                    unsigned int ds16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dp[_lp*2 + 0], dp[_lp*2+1 + 0]));
                        ds16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sdst_addr + (unsigned int)(prow * 128 + (wg * 4 ^ prow & 7) * 16))), "r"(ds16[0]), "r"(ds16[1]), "r"(ds16[2]), "r"(ds16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sdst_addr + (unsigned int)(prow * 128 + (wg * 4 + 1 ^ prow & 7) * 16))), "r"(ds16[4]), "r"(ds16[5]), "r"(ds16[6]), "r"(ds16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sdst_addr + (unsigned int)(prow * 128 + (wg * 4 + 2 ^ prow & 7) * 16))), "r"(ds16[8]), "r"(ds16[9]), "r"(ds16[10]), "r"(ds16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sdst_addr + (unsigned int)(prow * 128 + (wg * 4 + 3 ^ prow & 7) * 16))), "r"(ds16[12]), "r"(ds16[13]), "r"(ds16[14]), "r"(ds16[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                        :: "r"((ds_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(st_full_addr);
                    mbarrier_arrive(stats_empty_addr + (sc) * 8);
                    tc += 1;
                    if (tc == 2) { tc = 0; _phase_s_full ^= 1; }
                    sc += 1;
                    if (sc == 2) { sc = 0; _phase_stats_full ^= 1; }
                    t += 1;
                }
            }
            int _parity_ds_empty_1 = (int)((unsigned int)t % 2) ^ 1;
            mbarrier_wait_cluster_hint(ds_empty_addr, _parity_ds_empty_1, 10000000);
            mbarrier_wait(dkv_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int hd0 = half * 128 + wg * 64;
            int store_row = ((key < k_len) ? 1 : 0);
            if (found == 0) {
                store_row = 0;
            }
            long long out_row = (long long)(k0 + key);
            float acc[64];
            if (num_m > 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
                    : "r"(taddr + (unsigned int)(wg * 64) + lane_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(acc[32]), "=f"(acc[33]), "=f"(acc[34]), "=f"(acc[35]), "=f"(acc[36]), "=f"(acc[37]), "=f"(acc[38]), "=f"(acc[39]), "=f"(acc[40]), "=f"(acc[41]), "=f"(acc[42]), "=f"(acc[43]), "=f"(acc[44]), "=f"(acc[45]), "=f"(acc[46]), "=f"(acc[47]), "=f"(acc[48]), "=f"(acc[49]), "=f"(acc[50]), "=f"(acc[51]), "=f"(acc[52]), "=f"(acc[53]), "=f"(acc[54]), "=f"(acc[55]), "=f"(acc[56]), "=f"(acc[57]), "=f"(acc[58]), "=f"(acc[59]), "=f"(acc[60]), "=f"(acc[61]), "=f"(acc[62]), "=f"(acc[63])
                    : "r"(taddr + (unsigned int)(wg * 64 + 32) + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            } else {
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    acc[i] = 0.0f;
                }
            }
            long long base = (out_row * (long long)h_kv + (long long)h) * 256ll + (long long)hd0;
            if (store_row != 0) {
                #pragma unroll
                for (int i_1 = 0; i_1 < 16; i_1++) {
                    {
                        float4 _v4 = make_float4(acc[i_1 * 4 + 0], acc[i_1 * 4 + 1], acc[i_1 * 4 + 2], acc[i_1 * 4 + 3]);
                        *reinterpret_cast<float4*>((dv_out + (base + (long long)(i_1 * 4))) + 0) = _v4;
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int seg_1 = blockIdx.z;
            int n_blk_1 = n0 + (blockIdx.x >> 1);
            int q0_1 = cu_seqlens_q[seg_1];
            int q_len_1 = cu_seqlens_q[seg_1 + 1] - q0_1;
            int k0_1 = cu_seqlens_k[seg_1];
            int k_len_1 = cu_seqlens_k[seg_1 + 1] - k0_1;
            int found_1 = ((n_blk_1 * 128 < k_len_1) ? 1 : 0);
            int off_1 = k_len_1 - q_len_1;
            int first_key_1 = n_blk_1 * 128 - off_1;
            int m_min_1 = ((first_key_1 > 0) ? first_key_1 : 0) >> 6;
            int m_max_1 = q_len_1 + 63 >> 6;
            int num_m_1 = ((found_1 != 0) ? m_max_1 - m_min_1 : 0);
            int row_off_1 = q0_1 * 2 + seg_1 * 128 >> 7 << 7;
            int n_groups_1 = group >> 1;
            int trips = n_groups_1 * num_m_1;
            unsigned int qc = 0;
            unsigned int dc = 0;
            unsigned int tc_1 = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_do_full = 0;
            if ((cta_rank & 1) == 0) {
                if (elect_sync()) {
                    mbarrier_wait_cluster_hint(kv_full_addr, 0, 10000000);
                    #pragma unroll 1
                    for (int t_1 = 0; t_1 < trips + 1; t_1++) {
                        if (t_1 < trips) {
                            mbarrier_wait_cluster_hint(q_full_addr + (qc) * 8, _phase_q_full, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_0 = ((sk_addr) >> 4) & 0x7FFF;
                            int _smem_desc_lo_1 = (((sq_addr) >> 4) & 0x7FFF) + (qc) * 2048;
#else
                            int _smem_desc_lo_0 = ((sk_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_1 = (((sq_addr) >> 4) & 0x3FFF) + (qc) * 2048;
#endif
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_s + (tc_1 * 64))), "r"(0) : "memory");
                            tcgen05_commit_cg2_multicast(s_full_addr + (tc_1) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(q_empty_addr + (qc) * 8, (uint16_t)(3));
                            qc += 1;
                            if (qc == 2) { qc = 0; _phase_q_full ^= 1; }
                            mbarrier_wait_cluster_hint(do_full_addr + (dc) * 8, _phase_do_full, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_2 = ((sv_addr) >> 4) & 0x7FFF;
                            int _smem_desc_lo_3 = (((sdo_addr) >> 4) & 0x7FFF) + (dc) * 2048;
#else
                            int _smem_desc_lo_2 = ((sv_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_3 = (((sdo_addr) >> 4) & 0x3FFF) + (dc) * 2048;
#endif
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_dp + (tc_1 * 64))), "r"(0) : "memory");
                            tcgen05_commit_cg2_multicast(dp_full_addr + (tc_1) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(do_empty_addr + (dc) * 8, (uint16_t)(3));
                            dc += 1;
                            if (dc == 2) { dc = 0; _phase_do_full ^= 1; }
                            tc_1 += 1;
                            if (tc_1 == 2) { tc_1 = 0; }
                        }
                        if (t_1 > 0) {
                            int first = ((t_1 == 1) ? 1 : 0);
                            mbarrier_wait_cluster_hint(do_full_addr + (dc) * 8, _phase_do_full, 10000000);
                            int _parity_ds_full = (int)((unsigned int)t_1 % 2) ^ 1;
                            mbarrier_wait_cluster_hint(ds_full_addr, _parity_ds_full, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_4 = ((sp_addr) >> 4) & 0x7FFF;
                            int _smem_desc_lo_5 = ((((sdot_addr) >> 4) & 0x7FFF) | 0x4000000) + (dc) * 2048;
#else
                            int _smem_desc_lo_4 = ((sp_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_5 = ((((sdot_addr) >> 4) & 0x3FFF) | 0x4000000) + (dc) * 2048;
#endif
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_dv), "r"(((first) ? 0 : 1)) : "memory");
                            tcgen05_commit_cg2_multicast(ds_empty_addr, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(do_empty_addr + (dc) * 8, (uint16_t)(3));
                            dc += 1;
                            if (dc == 2) { dc = 0; _phase_do_full ^= 1; }
                        }
                    }
                    tcgen05_commit_cg2_multicast(dkv_full_addr, (uint16_t)(3));
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 9) {
        { // load_main
            int seg_2 = blockIdx.z;
            int n_blk_2 = n0 + (blockIdx.x >> 1);
            int q0_2 = cu_seqlens_q[seg_2];
            int q_len_2 = cu_seqlens_q[seg_2 + 1] - q0_2;
            int k0_2 = cu_seqlens_k[seg_2];
            int k_len_2 = cu_seqlens_k[seg_2 + 1] - k0_2;
            int found_2 = ((n_blk_2 * 128 < k_len_2) ? 1 : 0);
            int off_2 = k_len_2 - q_len_2;
            int first_key_2 = n_blk_2 * 128 - off_2;
            int m_min_2 = ((first_key_2 > 0) ? first_key_2 : 0) >> 6;
            int m_max_2 = q_len_2 + 63 >> 6;
            int num_m_2 = ((found_2 != 0) ? m_max_2 - m_min_2 : 0);
            int row_off_2 = q0_2 * 2 + seg_2 * 128 >> 7 << 7;
            int rank_1 = cta_rank & 1;
            int h_1 = h0 + blockIdx.y;
            int kv_row0 = k0_2 + n_blk_2 * 128;
            int n_groups_2 = group >> 1;
            int t_2 = 0;
            unsigned int qc_1 = 0;
            unsigned int dc_1 = 0;
            unsigned int sc_1 = 0;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_stats_empty = 1;
            unsigned int _phase_do_empty = 1;
            if (elect_sync()) {
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                    :: "r"((kv_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                tma_3d_gmem2smem_cta2(sk_addr, (&K), 0, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sv_addr, (&V), 0, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sk_addr + 8192, (&K), 64, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sv_addr + 8192, (&V), 64, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sk_addr + 16384, (&K), 128, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sv_addr + 16384, (&V), 128, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sk_addr + 24576, (&K), 192, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                tma_3d_gmem2smem_cta2(sv_addr + 24576, (&V), 192, h_1, kv_row0 + rank_1 * 64, ((kv_full_addr) & 0xFEFFFFFF));
                int prev_head0 = 0;
                int prev_tok = 0;
                #pragma unroll 1
                for (int g_1 = 0; g_1 < n_groups_2; g_1++) {
                    int head0 = h_1 * group + g_1 * 2;
                    int srow = (h_1 * n_groups_2 + g_1) * rows_pad + row_off_2;
                    #pragma unroll 1
                    for (int mi_1 = 0; mi_1 < num_m_2; mi_1++) {
                        int tok = q0_2 + (m_min_2 + mi_1) * 64;
                        int stat_row = srow + (m_min_2 + mi_1) * 128;
                        mbarrier_wait(q_empty_addr + (qc_1) * 8, _phase_q_empty);
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr + (qc_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_3d_gmem2smem_cta2(sq_addr + qc_1 * 32768, (&Q), 0, head0, tok + rank_1 * 32, ((q_full_addr + (qc_1) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sq_addr + qc_1 * 32768 + 8192, (&Q), 64, head0, tok + rank_1 * 32, ((q_full_addr + (qc_1) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sq_addr + qc_1 * 32768 + 16384, (&Q), 128, head0, tok + rank_1 * 32, ((q_full_addr + (qc_1) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sq_addr + qc_1 * 32768 + 24576, (&Q), 192, head0, tok + rank_1 * 32, ((q_full_addr + (qc_1) * 8) & 0xFEFFFFFF));
                        qc_1 += 1;
                        if (qc_1 == 2) { qc_1 = 0; _phase_q_empty ^= 1; }
                        mbarrier_wait(stats_empty_addr + (sc_1) * 8, _phase_stats_empty);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(stats_full_addr + (sc_1) * 8, 1024);
                        cp_async_bulk_gmem2smem(sst_addr + sc_1 * 1024, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(lse_log2) + (unsigned long long)stat_row * 4ull), 512, stats_full_addr + (sc_1) * 8);
                        cp_async_bulk_gmem2smem(sst_addr + sc_1 * 1024 + 512, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dpsum) + (unsigned long long)stat_row * 4ull), 512, stats_full_addr + (sc_1) * 8);
                        sc_1 += 1;
                        if (sc_1 == 2) { sc_1 = 0; _phase_stats_empty ^= 1; }
                        mbarrier_wait(do_empty_addr + (dc_1) * 8, _phase_do_empty);
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        tma_3d_gmem2smem_cta2(sdo_addr + dc_1 * 32768, (&dO), 0, head0, tok + rank_1 * 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sdo_addr + dc_1 * 32768 + 8192, (&dO), 64, head0, tok + rank_1 * 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sdo_addr + dc_1 * 32768 + 16384, (&dO), 128, head0, tok + rank_1 * 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(sdo_addr + dc_1 * 32768 + 24576, (&dO), 192, head0, tok + rank_1 * 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                        dc_1 += 1;
                        if (dc_1 == 2) { dc_1 = 0; _phase_do_empty ^= 1; }
                        if (t_2 > 0) {
                            mbarrier_wait(do_empty_addr + (dc_1) * 8, _phase_do_empty);
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768, (&dO), rank_1 * 128, prev_head0, prev_tok, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768 + 8192, (&dO), rank_1 * 128, prev_head0, prev_tok + 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768 + 16384, (&dO), rank_1 * 128 + 64, prev_head0, prev_tok, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768 + 24576, (&dO), rank_1 * 128 + 64, prev_head0, prev_tok + 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                            dc_1 += 1;
                            if (dc_1 == 2) { dc_1 = 0; _phase_do_empty ^= 1; }
                        }
                        prev_head0 = head0;
                        prev_tok = tok;
                        t_2 += 1;
                    }
                }
                if (t_2 > 0) {
                    mbarrier_wait(do_empty_addr + (dc_1) * 8, _phase_do_empty);
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768, (&dO), rank_1 * 128, prev_head0, prev_tok, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768 + 8192, (&dO), rank_1 * 128, prev_head0, prev_tok + 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768 + 16384, (&dO), rank_1 * 128 + 64, prev_head0, prev_tok, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(sdot_addr + dc_1 * 32768 + 24576, (&dO), rank_1 * 128 + 64, prev_head0, prev_tok + 32, ((do_full_addr + (dc_1) * 8) & 0xFEFFFFFF));
                    dc_1 += 1;
                    if (dc_1 == 2) { dc_1 = 0; _phase_do_empty ^= 1; }
                }
                mbarrier_wait(q_empty_addr + (qc_1) * 8, _phase_q_empty);
                qc_1 += 1;
                if (qc_1 == 2) { qc_1 = 0; _phase_q_empty ^= 1; }
                mbarrier_wait(q_empty_addr + (qc_1) * 8, _phase_q_empty);
                qc_1 += 1;
                if (qc_1 == 2) { qc_1 = 0; _phase_q_empty ^= 1; }
                mbarrier_wait(do_empty_addr + (dc_1) * 8, _phase_do_empty);
                dc_1 += 1;
                if (dc_1 == 2) { dc_1 = 0; _phase_do_empty ^= 1; }
                mbarrier_wait(do_empty_addr + (dc_1) * 8, _phase_do_empty);
                dc_1 += 1;
                if (dc_1 == 2) { dc_1 = 0; _phase_do_empty ^= 1; }
            }
        }
    }
    // ---- Role: store ----
    if (warp == 10) {
        { // store_main
            int seg_3 = blockIdx.z;
            int n_blk_3 = n0 + (blockIdx.x >> 1);
            int q0_3 = cu_seqlens_q[seg_3];
            int q_len_3 = cu_seqlens_q[seg_3 + 1] - q0_3;
            int k0_3 = cu_seqlens_k[seg_3];
            int k_len_3 = cu_seqlens_k[seg_3 + 1] - k0_3;
            int found_3 = ((n_blk_3 * 128 < k_len_3) ? 1 : 0);
            int off_3 = k_len_3 - q_len_3;
            int first_key_3 = n_blk_3 * 128 - off_3;
            int m_min_3 = ((first_key_3 > 0) ? first_key_3 : 0) >> 6;
            int m_max_3 = q_len_3 + 63 >> 6;
            int num_m_3 = ((found_3 != 0) ? m_max_3 - m_min_3 : 0);
            int row_off_3 = q0_3 * 2 + seg_3 * 128 >> 7 << 7;
            int rank_2 = cta_rank & 1;
            int n_groups_3 = group >> 1;
            int t_3 = 0;
            int first_tile = 0;
            if (found_3 != 0) {
                first_tile = tile_base[n_blk_3 * num_segments + blockIdx.z] - tile_base[n0 * num_segments];
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int g_2 = 0; g_2 < n_groups_3; g_2++) {
                    long long head_tile = (long long)(blockIdx.y * n_groups_3 + g_2) * (long long)tile_stride + (long long)first_tile;
                    #pragma unroll 1
                    for (int mi_2 = 0; mi_2 < num_m_3; mi_2++) {
                        int _parity_st_full = (int)((unsigned int)t_3 % 2);
                        mbarrier_wait(st_full_addr, _parity_st_full);
                        long long tile = (head_tile + (long long)mi_2) * 16384ll;
                        {
                            void* _cpbulk_dst_0 = reinterpret_cast<void*>(ds_ws + (tile + 0ll + (long long)(rank_2 * 4096)));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_0), "r"(sdst_addr + 0), "r"((uint32_t)(8192))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_1 = reinterpret_cast<void*>(ds_ws + (tile + 8192ll + (long long)(rank_2 * 4096)));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_1), "r"(sdst_addr + 8192), "r"((uint32_t)(8192))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(ds_empty_addr);
                        t_3 += 1;
                    }
                }
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 8) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
