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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}


__device__ __forceinline__ uint32_t mbarrier_test_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}






__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}


// Qmul4 (QMUL4.E4M3x4.E2M1x4.E4M3x4.SATFINITE) as the instruction: byte k of the result is the E2M1 value in one
// nibble of src times the E4M3 value in one byte of scale, rounded once to E4M3 (to nearest even, saturating to
// +-448). mul .e2m1p4x4 reads nibble 2k of a word, mul .e2m1x4 nibble k of a 16-bit half; both times byte k of
// their scale word.
//   qmul4_direct   nibble 2k      byte k
//   qmul4_b<j>     nibble 2k      byte j
//   qmul4_lower4   nibble k       byte 0
//   qmul4_higher4  nibble k + 4   byte 0

// Byte kByte of scale in all four bytes.
template <int kByte>
__device__ __forceinline__ uint32_t qmul4_broadcast(uint32_t scale) {
  uint32_t y;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(y) : "r"(scale), "n"(kByte * 0x1111));
  return y;
}

__device__ __forceinline__ uint32_t qmul4_lower4(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("{\n.reg .b16 h0, h1;\nmov.b32 {h0, h1}, %1;\nmul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, h0, %2;\n}\n"
      : "=r"(y)
      : "r"(src), "r"(qmul4_broadcast<0>(scale)));
  return y;
}

__device__ __forceinline__ uint32_t qmul4_higher4(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("{\n.reg .b16 h0, h1;\nmov.b32 {h0, h1}, %1;\nmul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, h1, %2;\n}\n"
      : "=r"(y)
      : "r"(src), "r"(qmul4_broadcast<0>(scale)));
  return y;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_mla_nvfp4_paged_decode_32a47bbfa05aebbc97c5(const __grid_constant__ CUtensorMap tmap_qn, const __grid_constant__ CUtensorMap tmap_qs, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_ks, const __grid_constant__ CUtensorMap tmap_kr, const __grid_constant__ CUtensorMap tmap_po, float* __restrict__ q_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, float* __restrict__ lse, int* __restrict__ seq_lens, int* __restrict__ kv_len_global, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq, int page_shift, int cp_world, int cp_rank, int has_lse)
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
    #define q_empty_addr (mbar_base + 8)
    #define raw_full_addr (mbar_base + 16)
    #define raw_empty_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 64)
    #define sf_full_addr (mbar_base + 72)
    #define sf_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 88)
    #define s_empty_addr (mbar_base + 96)
    #define v_full_addr (mbar_base + 104)
    #define v_empty_addr (mbar_base + 120)
    #define corr_sig_addr (mbar_base + 136)
    #define corr_empty_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 152)
    #define p_empty_addr (mbar_base + 168)
    #define pv_done_addr (mbar_base + 184)
    #define o_full_addr (mbar_base + 192)
    #define o_empty_addr (mbar_base + 200)
    #define epi_done_addr (mbar_base + 208)
    #define shift_full_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_qs = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_qs_addr = smem + 17408;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_qr_addr = smem + 19456;
    uint16_t* smem_raw_1 = reinterpret_cast<uint16_t*>(smem_raw + 23552);
    const int smem_raw_addr = smem + 23552;
    uint8_t* smem_kq = reinterpret_cast<uint8_t*>(smem_raw + 23552);
    const int smem_kq_addr = smem + 23552;
    uint8_t* smem_ko = reinterpret_cast<uint8_t*>(smem_raw + 39936);
    const int smem_ko_addr = smem + 39936;
    uint8_t* smem_ks = reinterpret_cast<uint8_t*>(smem_raw + 48128);
    const int smem_ks_addr = smem + 48128;
    uint8_t* smem_kr = reinterpret_cast<uint8_t*>(smem_raw + 52224);
    const int smem_kr_addr = smem + 52224;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 121856);
    const int smem_v_addr = smem + 121856;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 121856);
    const int smem_epi_addr = smem + 121856;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 187392);
    const int smem_p_addr = smem + 187392;
    float* row_state = reinterpret_cast<float*>(smem_raw + 203776);
    const int row_state_addr = smem + 203776;
    float* smem_xchg_max = reinterpret_cast<float*>(smem_raw + 204288);
    const int smem_xchg_max_addr = smem + 204288;
    float* smem_xchg_sum = reinterpret_cast<float*>(smem_raw + 204800);
    const int smem_xchg_sum_addr = smem + 204800;
    unsigned int* smem_shift_mul = reinterpret_cast<unsigned int*>(smem_raw + 205312);
    const int smem_shift_mul_addr = smem + 205312;
#if __CUDA_ARCH__ == 1030
    uint16_t* smem_shift_fac = reinterpret_cast<uint16_t*>(smem_raw + 206848);
#else
    unsigned int* smem_shift_fac = reinterpret_cast<unsigned int*>(smem_raw + 206848);
#endif
    const int smem_shift_fac_addr = smem + 206848;
#if __CUDA_ARCH__ == 1030

    // Kernel setup ops
    int q_end_k = cum_seq_lens_q[blockIdx.z + 1];
    int q_start_k = cum_seq_lens_q[blockIdx.z];
    int kv_len_k = seq_lens[blockIdx.z];
    int g_len_k = kv_len_global[blockIdx.z];
#else
#endif

    // Mbarrier init (30 barriers at smem_raw[0..240))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=129
            mbarrier_init(smem + 8, 129);
            // --- pipeline 'raw_pipe' ---
            // raw_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // raw_empty: 3 barriers, init_count=321
            mbarrier_init(smem + 40, 321);
            mbarrier_init(smem + 48, 321);
            mbarrier_init(smem + 56, 321);
            // sfa_full: 1 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            // sf_full: 1 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            // sf_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // s_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=384
            mbarrier_init(smem + 104, 384);
            mbarrier_init(smem + 112, 384);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // corr_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 152, 512);
            mbarrier_init(smem + 160, 512);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // o_full: 1 barriers, init_count=129
            mbarrier_init(smem + 192, 129);
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            // epi_done: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // --- pipeline 'raw_pipe' ---
#if __CUDA_ARCH__ == 1030
            // shift_full: 3 barriers, init_count=192
            mbarrier_init(smem + 216, 192);
            mbarrier_init(smem + 224, 192);
            mbarrier_init(smem + 232, 192);
#else
            // shift_full: 3 barriers, init_count=128
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
#endif
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 240);
    if (warp == 0) {
        int _tmem_hold = smem + 240;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;
    const unsigned int sfa_0 = taddr + 320;
    const unsigned int sfa_1 = taddr + 328;
    const unsigned int sfa_2 = taddr + 336;
    const unsigned int sfa_3 = taddr + 344;
    const unsigned int sfb_0 = taddr + 352;
    const unsigned int sfb_1 = taddr + 360;
    const unsigned int sfb_2 = taddr + 368;
    const unsigned int sfb_3 = taddr + 376;
    const unsigned int sfb_4 = taddr + 384;
    const unsigned int sfb_5 = taddr + 392;
    const unsigned int sfb_6 = taddr + 400;
    const unsigned int sfb_7 = taddr + 408;
    const unsigned int tmem_q = taddr + 416;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            int quadrant = make_warp_uniform(warp % 4);
            int n_half = make_warp_uniform(quadrant / 2);
            int row_c = quadrant % 2 * 32 + lane;
            int lane_base = quadrant * 32 << 16;
            int xchg_idx = n_half * 64 + row_c;
            int xchg_bar = make_warp_uniform(2 + quadrant % 2);
            int row_in_tile = cta_rank * 64 + row_c;
            int warp_row0_local = cta_rank * 64 + quadrant % 2 * 32;
            unsigned int warp_tiles_base = smem_epi_addr + (unsigned int)(quadrant * 16384);
            int p_s = 0;
            int sh_s = 0;
            int cx = blockIdx.x / 2;
            int n_clusters = gridDim.x / 2;
#if __CUDA_ARCH__ == 1030
            int q_end_r = q_end_k;
            int q_start_r = q_start_k;
            int kv_len_r = kv_len_k;
            int g_len_r = g_len_k;
#else
            int q_end_r = cum_seq_lens_q[blockIdx.z + 1];
            int q_start_r = cum_seq_lens_q[blockIdx.z];
            int kv_len_r = seq_lens[blockIdx.z];
            int g_len_r = kv_len_global[blockIdx.z];
#endif
            int pieces = ((q_end_r - q_start_r) * num_heads + 127) / 128 * num_split;
            int left = pieces - cx;
            int n_pieces_raw = (left + n_clusters - 1) / n_clusters;
            int n_pieces = ((left <= 0) ? 0 : n_pieces_raw);
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_shift_full = 0;
            unsigned int _phase_p_empty = 1;
            unsigned int _phase_corr_empty_0 = 1;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int pi_s = 0; pi_s < n_pieces; pi_s++) {
                int split_idx = (cx + pi_s * n_clusters) % num_split;
                int q_len_b = q_end_r - q_start_r;
                int rows_b = q_len_b * num_heads;
                int m_tile = (rows_b + 127) / 128 - 1 - (cx + pi_s * n_clusters) / num_split;
                int row0 = m_tile * 128;
                int rows_left = rows_b - row0;
                int rows_pos = ((rows_left < 0) ? 0 : rows_left);
                int rows_valid = ((rows_pos > 128) ? 128 : rows_pos);
                int row_base = q_start_r * num_heads + row0;
                int t_last = (row0 + rows_valid - 1) / num_heads;
                int num = g_len_r - q_len_b + t_last - cp_rank;
                int vis_raw = num / cp_world + 1;
                int vis_cap = ((vis_raw > kv_len_r) ? kv_len_r : vis_raw);
                int vis = ((num < 0) ? 0 : vis_cap);
                int kv_end = ((rows_valid == 0) ? 0 : vis);
                int n_tiles_total = (kv_end + 127) / 128;
                int tiles_per_split = (n_tiles_total + num_split - 1) / num_split;
                int my_start_raw = split_idx * tiles_per_split;
                int my_end_raw = my_start_raw + tiles_per_split;
                int my_end = ((my_end_raw > n_tiles_total) ? n_tiles_total : my_end_raw);
                int my_n_raw = my_end - my_start_raw;
                int empty = ((my_n_raw < 1) ? 1 : 0);
                int my_n_tiles = ((my_n_raw < 1) ? 1 : my_n_raw);
                int my_start = ((my_n_raw < 1) ? 0 : my_start_raw);
                int row_is_query = row_in_tile < rows_valid;
                int row_valid = row_is_query & (int)(empty == 0);
                int token_s = (row0 + row_in_tile) / num_heads;
                int out_row = row_base + row_in_tile;
                float q_scale_row = 0.0f;
                if (row_is_query != 0) {
                    q_scale_row = q_scale[out_row];
                }
                float scale_row = softmax_scale_log2 * q_scale_row;
                int num_0 = g_len_r - q_len_b + token_s - cp_rank;
                int vis_raw_0 = num_0 / cp_world + 1;
                int vis_cap_0 = ((vis_raw_0 > kv_len_r) ? kv_len_r : vis_raw_0);
                int vis_0 = ((num_0 < 0) ? 0 : vis_cap_0);
                int row_limit_off = ((row_is_query != 0) ? 0 : 1073741824);
                int row_limit = ((row_valid != 0) ? vis_0 : row_limit_off);
                float row_max = ((row_is_query != 0) ? -CUDART_INF_F : 1.048576e+06f);
                float row_sum = 0.0f;
                #pragma unroll 1
                for (int it_s = 0; it_s < my_n_tiles; it_s++) {
                    float sv0[32];
                    float sv1[32];
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv0[0]), "=f"(sv0[1]), "=f"(sv0[2]), "=f"(sv0[3]), "=f"(sv0[4]), "=f"(sv0[5]), "=f"(sv0[6]), "=f"(sv0[7]), "=f"(sv0[8]), "=f"(sv0[9]), "=f"(sv0[10]), "=f"(sv0[11]), "=f"(sv0[12]), "=f"(sv0[13]), "=f"(sv0[14]), "=f"(sv0[15]), "=f"(sv0[16]), "=f"(sv0[17]), "=f"(sv0[18]), "=f"(sv0[19]), "=f"(sv0[20]), "=f"(sv0[21]), "=f"(sv0[22]), "=f"(sv0[23]), "=f"(sv0[24]), "=f"(sv0[25]), "=f"(sv0[26]), "=f"(sv0[27]), "=f"(sv0[28]), "=f"(sv0[29]), "=f"(sv0[30]), "=f"(sv0[31])
                        : "r"(taddr + (unsigned int)lane_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv1[0]), "=f"(sv1[1]), "=f"(sv1[2]), "=f"(sv1[3]), "=f"(sv1[4]), "=f"(sv1[5]), "=f"(sv1[6]), "=f"(sv1[7]), "=f"(sv1[8]), "=f"(sv1[9]), "=f"(sv1[10]), "=f"(sv1[11]), "=f"(sv1[12]), "=f"(sv1[13]), "=f"(sv1[14]), "=f"(sv1[15]), "=f"(sv1[16]), "=f"(sv1[17]), "=f"(sv1[18]), "=f"(sv1[19]), "=f"(sv1[20]), "=f"(sv1[21]), "=f"(sv1[22]), "=f"(sv1[23]), "=f"(sv1[24]), "=f"(sv1[25]), "=f"(sv1[26]), "=f"(sv1[27]), "=f"(sv1[28]), "=f"(sv1[29]), "=f"(sv1[30]), "=f"(sv1[31])
                        : "r"(taddr + (unsigned int)lane_base + 32));
                    uint32_t sh_tok = mbarrier_test_wait(shift_full_addr + (sh_s) * 8, _phase_shift_full);
                    mbarrier_wait(p_empty_addr + (p_s) * 8, _phase_p_empty);
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
#if __CUDA_ARCH__ == 1030
                    unsigned int fac_0[16];
                    unsigned int fac_1[16];
#else
                    unsigned int fac_0[32];
                    unsigned int fac_1[32];
                    mbarrier_wait_token(shift_full_addr + (sh_s) * 8, _phase_shift_full, sh_tok);
                    unsigned int fac_addr = smem_shift_fac_addr + (unsigned int)((sh_s * 128 + n_half * 32) * 4);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(0) + 3]))
                        : "r"(fac_addr));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(4) + 3]))
                        : "r"(fac_addr + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(8) + 3]))
                        : "r"(fac_addr + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(12) + 3]))
                        : "r"(fac_addr + 48));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(16) + 3]))
                        : "r"(fac_addr + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(20) + 3]))
                        : "r"(fac_addr + 80));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(24) + 3]))
                        : "r"(fac_addr + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(28) + 3]))
                        : "r"(fac_addr + 112));
                    unsigned int fac_addr_0 = smem_shift_fac_addr + (unsigned int)((sh_s * 128 + 64 + n_half * 32) * 4);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(0) + 3]))
                        : "r"(fac_addr_0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(4) + 3]))
                        : "r"(fac_addr_0 + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(8) + 3]))
                        : "r"(fac_addr_0 + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(12) + 3]))
                        : "r"(fac_addr_0 + 48));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(16) + 3]))
                        : "r"(fac_addr_0 + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(20) + 3]))
                        : "r"(fac_addr_0 + 80));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(24) + 3]))
                        : "r"(fac_addr_0 + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(28) + 3]))
                        : "r"(fac_addr_0 + 112));
#endif
                    int tile_base_s = (my_start + it_s) * 128 + n_half * 32;
                    int _max_0 = ((row_limit - tile_base_s) > (0) ? (row_limit - tile_base_s) : (0));
                    int valid_0 = ((_max_0) < (32) ? (_max_0) : (32));
                    int _max_1 = ((row_limit - tile_base_s - 64) > (0) ? (row_limit - tile_base_s - 64) : (0));
                    int valid_1 = ((_max_1) < (32) ? (_max_1) : (32));
                    if (valid_0 < 32) {
                        uint32_t _thresh_mask_2;
                        {
                            int _lim_3 = valid_0;
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
                        if (!(_thresh_mask_2 & (1u << 0))) sv0[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 1))) sv0[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 2))) sv0[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 3))) sv0[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 4))) sv0[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 5))) sv0[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 6))) sv0[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 7))) sv0[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 8))) sv0[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 9))) sv0[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 10))) sv0[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 11))) sv0[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 12))) sv0[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 13))) sv0[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 14))) sv0[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 15))) sv0[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 16))) sv0[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 17))) sv0[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 18))) sv0[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 19))) sv0[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 20))) sv0[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 21))) sv0[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 22))) sv0[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 23))) sv0[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 24))) sv0[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 25))) sv0[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 26))) sv0[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 27))) sv0[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 28))) sv0[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 29))) sv0[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 30))) sv0[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 31))) sv0[31] = -CUDART_INF_F;
                    }
                    if (valid_1 < 32) {
                        uint32_t _thresh_mask_4;
                        {
                            int _lim_5 = valid_1;
                            if (_lim_5 <= 0) { _thresh_mask_4 = 0u; }
                            else if (_lim_5 >= 32) { _thresh_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_4) : "r"(_lim_5));
                            }
                        }
                        if (!(_thresh_mask_4 & (1u << 0))) sv1[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 1))) sv1[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 2))) sv1[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 3))) sv1[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 4))) sv1[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 5))) sv1[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 6))) sv1[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 7))) sv1[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 8))) sv1[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 9))) sv1[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 10))) sv1[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 11))) sv1[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 12))) sv1[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 13))) sv1[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 14))) sv1[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 15))) sv1[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 16))) sv1[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 17))) sv1[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 18))) sv1[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 19))) sv1[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 20))) sv1[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 21))) sv1[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 22))) sv1[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 23))) sv1[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 24))) sv1[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 25))) sv1[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 26))) sv1[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 27))) sv1[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 28))) sv1[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 29))) sv1[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 30))) sv1[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 31))) sv1[31] = -CUDART_INF_F;
                    }
                    float2 _reg_reduce_max2_7 = {-CUDART_INF_F, -CUDART_INF_F};
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[0], sv0[1]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[2], sv0[3]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[4], sv0[5]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[6], sv0[7]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[8], sv0[9]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[10], sv0[11]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[12], sv0[13]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[14], sv0[15]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[16], sv0[17]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[18], sv0[19]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[20], sv0[21]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[22], sv0[23]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[24], sv0[25]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[26], sv0[27]));
                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[28], sv0[29]));
                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[30], sv0[31]));
                    float _reg_array_reduce_6 = row_max_reduce(_reg_reduce_max2_7);
                    float2 _reg_reduce_max2_9 = {-CUDART_INF_F, -CUDART_INF_F};
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[0], sv1[1]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[2], sv1[3]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[4], sv1[5]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[6], sv1[7]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[8], sv1[9]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[10], sv1[11]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[12], sv1[13]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[14], sv1[15]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[16], sv1[17]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[18], sv1[19]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[20], sv1[21]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[22], sv1[23]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[24], sv1[25]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[26], sv1[27]));
                    _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[28], sv1[29]));
                    _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[30], sv1[31]));
                    float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                    float tile_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(tile_max) : "f"(_reg_array_reduce_6), "f"(_reg_array_reduce_8));
                    float tile_max_scaled = ((tile_max > -CUDART_INF_F) ? tile_max * scale_row : -CUDART_INF_F);
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(row_max), "f"(tile_max_scaled));
                    smem_xchg_max[xchg_idx] = new_max;
                    asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                    new_max = max_noftz(new_max, smem_xchg_max[xchg_idx ^ 64]);
                    asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                    int keep_max = row_max > -CUDART_INF_F && new_max - row_max <= 2.0f;
                    float fresh_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                    float safe_max = ((keep_max != 0) ? row_max : fresh_max);
                    if (keep_max != 0) {
                        new_max = row_max;
                    }
#if __CUDA_ARCH__ == 1030
                    float alpha_ex2 = approx_exp2(row_max - fresh_max);
                    int alpha_live = row_max > -CUDART_INF_F && keep_max == 0;
#else
                    float _exp2_10 = approx_exp2(row_max - safe_max);
                    float alpha = ((row_max > -CUDART_INF_F) ? _exp2_10 : 1.0f);
#endif
                    row_max = new_max;
                    float exp_bias = 6.0f - safe_max;
                    mbarrier_wait(corr_empty_addr, _phase_corr_empty_0);
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait_token(shift_full_addr + (sh_s) * 8, _phase_shift_full, sh_tok);
                    float alpha = ((alpha_live != 0) ? alpha_ex2 : 1.0f);
#else
#endif
                    if (n_half == 0) {
                        row_state[row_c] = alpha;
                    }
                    mbarrier_arrive(corr_sig_addr);
                    _phase_corr_empty_0 ^= 1;
#if __CUDA_ARCH__ == 1030
                    unsigned int fac_addr = smem_shift_fac_addr + (unsigned int)((sh_s * 128 + n_half * 32) * 2);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(0) + 3]))
                        : "r"(fac_addr));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(4) + 3]))
                        : "r"(fac_addr + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(8) + 3]))
                        : "r"(fac_addr + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_0[(12) + 3]))
                        : "r"(fac_addr + 48));
                    unsigned int fac_addr_0 = smem_shift_fac_addr + (unsigned int)((sh_s * 128 + 64 + n_half * 32) * 2);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(0) + 3]))
                        : "r"(fac_addr_0));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(4) + 3]))
                        : "r"(fac_addr_0 + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(8) + 3]))
                        : "r"(fac_addr_0 + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fac_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fac_1[(12) + 3]))
                        : "r"(fac_addr_0 + 48));
                    const float2 _fma_b2_10 = {scale_row, scale_row};
                    const float2 _fma_c2_11 = {exp_bias, exp_bias};
                    float2 _fma_pair_12 = fma_f32x2(make_float2(sv0[0], sv0[1]), _fma_b2_10, _fma_c2_11);
                    sv0[0] = _fma_pair_12.x;
                    sv0[1] = _fma_pair_12.y;
                    float2 _fma_pair_13 = fma_f32x2(make_float2(sv0[2], sv0[3]), _fma_b2_10, _fma_c2_11);
                    sv0[2] = _fma_pair_13.x;
                    sv0[3] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(sv0[4], sv0[5]), _fma_b2_10, _fma_c2_11);
                    sv0[4] = _fma_pair_14.x;
                    sv0[5] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(sv0[6], sv0[7]), _fma_b2_10, _fma_c2_11);
                    sv0[6] = _fma_pair_15.x;
                    sv0[7] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(sv0[8], sv0[9]), _fma_b2_10, _fma_c2_11);
                    sv0[8] = _fma_pair_16.x;
                    sv0[9] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(sv0[10], sv0[11]), _fma_b2_10, _fma_c2_11);
                    sv0[10] = _fma_pair_17.x;
                    sv0[11] = _fma_pair_17.y;
                    float2 _fma_pair_18 = fma_f32x2(make_float2(sv0[12], sv0[13]), _fma_b2_10, _fma_c2_11);
                    sv0[12] = _fma_pair_18.x;
                    sv0[13] = _fma_pair_18.y;
                    float2 _fma_pair_19 = fma_f32x2(make_float2(sv0[14], sv0[15]), _fma_b2_10, _fma_c2_11);
                    sv0[14] = _fma_pair_19.x;
                    sv0[15] = _fma_pair_19.y;
                    float2 _fma_pair_20 = fma_f32x2(make_float2(sv0[16], sv0[17]), _fma_b2_10, _fma_c2_11);
                    sv0[16] = _fma_pair_20.x;
                    sv0[17] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(sv0[18], sv0[19]), _fma_b2_10, _fma_c2_11);
                    sv0[18] = _fma_pair_21.x;
                    sv0[19] = _fma_pair_21.y;
                    float2 _fma_pair_22 = fma_f32x2(make_float2(sv0[20], sv0[21]), _fma_b2_10, _fma_c2_11);
                    sv0[20] = _fma_pair_22.x;
                    sv0[21] = _fma_pair_22.y;
                    float2 _fma_pair_23 = fma_f32x2(make_float2(sv0[22], sv0[23]), _fma_b2_10, _fma_c2_11);
                    sv0[22] = _fma_pair_23.x;
                    sv0[23] = _fma_pair_23.y;
                    float2 _fma_pair_24 = fma_f32x2(make_float2(sv0[24], sv0[25]), _fma_b2_10, _fma_c2_11);
                    sv0[24] = _fma_pair_24.x;
                    sv0[25] = _fma_pair_24.y;
                    float2 _fma_pair_25 = fma_f32x2(make_float2(sv0[26], sv0[27]), _fma_b2_10, _fma_c2_11);
                    sv0[26] = _fma_pair_25.x;
                    sv0[27] = _fma_pair_25.y;
                    float2 _fma_pair_26 = fma_f32x2(make_float2(sv0[28], sv0[29]), _fma_b2_10, _fma_c2_11);
                    sv0[28] = _fma_pair_26.x;
                    sv0[29] = _fma_pair_26.y;
                    float2 _fma_pair_27 = fma_f32x2(make_float2(sv0[30], sv0[31]), _fma_b2_10, _fma_c2_11);
                    sv0[30] = _fma_pair_27.x;
                    sv0[31] = _fma_pair_27.y;
                    const float2 _fma_b2_28 = {scale_row, scale_row};
                    const float2 _fma_c2_29 = {exp_bias, exp_bias};
                    float2 _fma_pair_30 = fma_f32x2(make_float2(sv1[0], sv1[1]), _fma_b2_28, _fma_c2_29);
                    sv1[0] = _fma_pair_30.x;
                    sv1[1] = _fma_pair_30.y;
                    float2 _fma_pair_31 = fma_f32x2(make_float2(sv1[2], sv1[3]), _fma_b2_28, _fma_c2_29);
                    sv1[2] = _fma_pair_31.x;
                    sv1[3] = _fma_pair_31.y;
                    float2 _fma_pair_32 = fma_f32x2(make_float2(sv1[4], sv1[5]), _fma_b2_28, _fma_c2_29);
                    sv1[4] = _fma_pair_32.x;
                    sv1[5] = _fma_pair_32.y;
                    float2 _fma_pair_33 = fma_f32x2(make_float2(sv1[6], sv1[7]), _fma_b2_28, _fma_c2_29);
                    sv1[6] = _fma_pair_33.x;
                    sv1[7] = _fma_pair_33.y;
                    float2 _fma_pair_34 = fma_f32x2(make_float2(sv1[8], sv1[9]), _fma_b2_28, _fma_c2_29);
                    sv1[8] = _fma_pair_34.x;
                    sv1[9] = _fma_pair_34.y;
                    float2 _fma_pair_35 = fma_f32x2(make_float2(sv1[10], sv1[11]), _fma_b2_28, _fma_c2_29);
                    sv1[10] = _fma_pair_35.x;
                    sv1[11] = _fma_pair_35.y;
                    float2 _fma_pair_36 = fma_f32x2(make_float2(sv1[12], sv1[13]), _fma_b2_28, _fma_c2_29);
                    sv1[12] = _fma_pair_36.x;
                    sv1[13] = _fma_pair_36.y;
                    float2 _fma_pair_37 = fma_f32x2(make_float2(sv1[14], sv1[15]), _fma_b2_28, _fma_c2_29);
                    sv1[14] = _fma_pair_37.x;
                    sv1[15] = _fma_pair_37.y;
                    float2 _fma_pair_38 = fma_f32x2(make_float2(sv1[16], sv1[17]), _fma_b2_28, _fma_c2_29);
                    sv1[16] = _fma_pair_38.x;
                    sv1[17] = _fma_pair_38.y;
                    float2 _fma_pair_39 = fma_f32x2(make_float2(sv1[18], sv1[19]), _fma_b2_28, _fma_c2_29);
                    sv1[18] = _fma_pair_39.x;
                    sv1[19] = _fma_pair_39.y;
                    float2 _fma_pair_40 = fma_f32x2(make_float2(sv1[20], sv1[21]), _fma_b2_28, _fma_c2_29);
                    sv1[20] = _fma_pair_40.x;
                    sv1[21] = _fma_pair_40.y;
                    float2 _fma_pair_41 = fma_f32x2(make_float2(sv1[22], sv1[23]), _fma_b2_28, _fma_c2_29);
                    sv1[22] = _fma_pair_41.x;
                    sv1[23] = _fma_pair_41.y;
                    float2 _fma_pair_42 = fma_f32x2(make_float2(sv1[24], sv1[25]), _fma_b2_28, _fma_c2_29);
                    sv1[24] = _fma_pair_42.x;
                    sv1[25] = _fma_pair_42.y;
                    float2 _fma_pair_43 = fma_f32x2(make_float2(sv1[26], sv1[27]), _fma_b2_28, _fma_c2_29);
                    sv1[26] = _fma_pair_43.x;
                    sv1[27] = _fma_pair_43.y;
                    float2 _fma_pair_44 = fma_f32x2(make_float2(sv1[28], sv1[29]), _fma_b2_28, _fma_c2_29);
                    sv1[28] = _fma_pair_44.x;
                    sv1[29] = _fma_pair_44.y;
                    float2 _fma_pair_45 = fma_f32x2(make_float2(sv1[30], sv1[31]), _fma_b2_28, _fma_c2_29);
                    sv1[30] = _fma_pair_45.x;
                    sv1[31] = _fma_pair_45.y;
#else
                    const float2 _fma_b2_11 = {scale_row, scale_row};
                    const float2 _fma_c2_12 = {exp_bias, exp_bias};
                    float2 _fma_pair_13 = fma_f32x2(make_float2(sv0[0], sv0[1]), _fma_b2_11, _fma_c2_12);
                    sv0[0] = _fma_pair_13.x;
                    sv0[1] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(sv0[2], sv0[3]), _fma_b2_11, _fma_c2_12);
                    sv0[2] = _fma_pair_14.x;
                    sv0[3] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(sv0[4], sv0[5]), _fma_b2_11, _fma_c2_12);
                    sv0[4] = _fma_pair_15.x;
                    sv0[5] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(sv0[6], sv0[7]), _fma_b2_11, _fma_c2_12);
                    sv0[6] = _fma_pair_16.x;
                    sv0[7] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(sv0[8], sv0[9]), _fma_b2_11, _fma_c2_12);
                    sv0[8] = _fma_pair_17.x;
                    sv0[9] = _fma_pair_17.y;
                    float2 _fma_pair_18 = fma_f32x2(make_float2(sv0[10], sv0[11]), _fma_b2_11, _fma_c2_12);
                    sv0[10] = _fma_pair_18.x;
                    sv0[11] = _fma_pair_18.y;
                    float2 _fma_pair_19 = fma_f32x2(make_float2(sv0[12], sv0[13]), _fma_b2_11, _fma_c2_12);
                    sv0[12] = _fma_pair_19.x;
                    sv0[13] = _fma_pair_19.y;
                    float2 _fma_pair_20 = fma_f32x2(make_float2(sv0[14], sv0[15]), _fma_b2_11, _fma_c2_12);
                    sv0[14] = _fma_pair_20.x;
                    sv0[15] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(sv0[16], sv0[17]), _fma_b2_11, _fma_c2_12);
                    sv0[16] = _fma_pair_21.x;
                    sv0[17] = _fma_pair_21.y;
                    float2 _fma_pair_22 = fma_f32x2(make_float2(sv0[18], sv0[19]), _fma_b2_11, _fma_c2_12);
                    sv0[18] = _fma_pair_22.x;
                    sv0[19] = _fma_pair_22.y;
                    float2 _fma_pair_23 = fma_f32x2(make_float2(sv0[20], sv0[21]), _fma_b2_11, _fma_c2_12);
                    sv0[20] = _fma_pair_23.x;
                    sv0[21] = _fma_pair_23.y;
                    float2 _fma_pair_24 = fma_f32x2(make_float2(sv0[22], sv0[23]), _fma_b2_11, _fma_c2_12);
                    sv0[22] = _fma_pair_24.x;
                    sv0[23] = _fma_pair_24.y;
                    float2 _fma_pair_25 = fma_f32x2(make_float2(sv0[24], sv0[25]), _fma_b2_11, _fma_c2_12);
                    sv0[24] = _fma_pair_25.x;
                    sv0[25] = _fma_pair_25.y;
                    float2 _fma_pair_26 = fma_f32x2(make_float2(sv0[26], sv0[27]), _fma_b2_11, _fma_c2_12);
                    sv0[26] = _fma_pair_26.x;
                    sv0[27] = _fma_pair_26.y;
                    float2 _fma_pair_27 = fma_f32x2(make_float2(sv0[28], sv0[29]), _fma_b2_11, _fma_c2_12);
                    sv0[28] = _fma_pair_27.x;
                    sv0[29] = _fma_pair_27.y;
                    float2 _fma_pair_28 = fma_f32x2(make_float2(sv0[30], sv0[31]), _fma_b2_11, _fma_c2_12);
                    sv0[30] = _fma_pair_28.x;
                    sv0[31] = _fma_pair_28.y;
                    const float2 _fma_b2_29 = {scale_row, scale_row};
                    const float2 _fma_c2_30 = {exp_bias, exp_bias};
                    float2 _fma_pair_31 = fma_f32x2(make_float2(sv1[0], sv1[1]), _fma_b2_29, _fma_c2_30);
                    sv1[0] = _fma_pair_31.x;
                    sv1[1] = _fma_pair_31.y;
                    float2 _fma_pair_32 = fma_f32x2(make_float2(sv1[2], sv1[3]), _fma_b2_29, _fma_c2_30);
                    sv1[2] = _fma_pair_32.x;
                    sv1[3] = _fma_pair_32.y;
                    float2 _fma_pair_33 = fma_f32x2(make_float2(sv1[4], sv1[5]), _fma_b2_29, _fma_c2_30);
                    sv1[4] = _fma_pair_33.x;
                    sv1[5] = _fma_pair_33.y;
                    float2 _fma_pair_34 = fma_f32x2(make_float2(sv1[6], sv1[7]), _fma_b2_29, _fma_c2_30);
                    sv1[6] = _fma_pair_34.x;
                    sv1[7] = _fma_pair_34.y;
                    float2 _fma_pair_35 = fma_f32x2(make_float2(sv1[8], sv1[9]), _fma_b2_29, _fma_c2_30);
                    sv1[8] = _fma_pair_35.x;
                    sv1[9] = _fma_pair_35.y;
                    float2 _fma_pair_36 = fma_f32x2(make_float2(sv1[10], sv1[11]), _fma_b2_29, _fma_c2_30);
                    sv1[10] = _fma_pair_36.x;
                    sv1[11] = _fma_pair_36.y;
                    float2 _fma_pair_37 = fma_f32x2(make_float2(sv1[12], sv1[13]), _fma_b2_29, _fma_c2_30);
                    sv1[12] = _fma_pair_37.x;
                    sv1[13] = _fma_pair_37.y;
                    float2 _fma_pair_38 = fma_f32x2(make_float2(sv1[14], sv1[15]), _fma_b2_29, _fma_c2_30);
                    sv1[14] = _fma_pair_38.x;
                    sv1[15] = _fma_pair_38.y;
                    float2 _fma_pair_39 = fma_f32x2(make_float2(sv1[16], sv1[17]), _fma_b2_29, _fma_c2_30);
                    sv1[16] = _fma_pair_39.x;
                    sv1[17] = _fma_pair_39.y;
                    float2 _fma_pair_40 = fma_f32x2(make_float2(sv1[18], sv1[19]), _fma_b2_29, _fma_c2_30);
                    sv1[18] = _fma_pair_40.x;
                    sv1[19] = _fma_pair_40.y;
                    float2 _fma_pair_41 = fma_f32x2(make_float2(sv1[20], sv1[21]), _fma_b2_29, _fma_c2_30);
                    sv1[20] = _fma_pair_41.x;
                    sv1[21] = _fma_pair_41.y;
                    float2 _fma_pair_42 = fma_f32x2(make_float2(sv1[22], sv1[23]), _fma_b2_29, _fma_c2_30);
                    sv1[22] = _fma_pair_42.x;
                    sv1[23] = _fma_pair_42.y;
                    float2 _fma_pair_43 = fma_f32x2(make_float2(sv1[24], sv1[25]), _fma_b2_29, _fma_c2_30);
                    sv1[24] = _fma_pair_43.x;
                    sv1[25] = _fma_pair_43.y;
                    float2 _fma_pair_44 = fma_f32x2(make_float2(sv1[26], sv1[27]), _fma_b2_29, _fma_c2_30);
                    sv1[26] = _fma_pair_44.x;
                    sv1[27] = _fma_pair_44.y;
                    float2 _fma_pair_45 = fma_f32x2(make_float2(sv1[28], sv1[29]), _fma_b2_29, _fma_c2_30);
                    sv1[28] = _fma_pair_45.x;
                    sv1[29] = _fma_pair_45.y;
                    float2 _fma_pair_46 = fma_f32x2(make_float2(sv1[30], sv1[31]), _fma_b2_29, _fma_c2_30);
                    sv1[30] = _fma_pair_46.x;
                    sv1[31] = _fma_pair_46.y;
#endif
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv0[_le] = approx_exp2(sv0[_le]);
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        sv1[_le] = approx_exp2(sv1[_le]);
                    }
#if __CUDA_ARCH__ == 1030
                    float2 _reg_reduce_sum2_47 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[0], sv0[1]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[2], sv0[3]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[4], sv0[5]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[6], sv0[7]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[8], sv0[9]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[10], sv0[11]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[12], sv0[13]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[14], sv0[15]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[16], sv0[17]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[18], sv0[19]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[20], sv0[21]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[22], sv0[23]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[24], sv0[25]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[26], sv0[27]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[28], sv0[29]));
                    _reg_reduce_sum2_47 = add_f32x2(_reg_reduce_sum2_47, make_float2(sv0[30], sv0[31]));
                    float _reg_array_reduce_46 = _reg_reduce_sum2_47.x + _reg_reduce_sum2_47.y;
                    float2 _reg_reduce_sum2_49 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[0], sv1[1]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[2], sv1[3]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[4], sv1[5]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[6], sv1[7]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[8], sv1[9]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[10], sv1[11]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[12], sv1[13]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[14], sv1[15]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[16], sv1[17]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[18], sv1[19]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[20], sv1[21]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[22], sv1[23]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[24], sv1[25]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[26], sv1[27]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[28], sv1[29]));
                    _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv1[30], sv1[31]));
                    float _reg_array_reduce_48 = _reg_reduce_sum2_49.x + _reg_reduce_sum2_49.y;
                    float block_sum = _reg_array_reduce_46 + _reg_array_reduce_48;
#else
                    float2 _reg_reduce_sum2_48 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[0], sv0[1]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[2], sv0[3]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[4], sv0[5]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[6], sv0[7]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[8], sv0[9]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[10], sv0[11]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[12], sv0[13]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[14], sv0[15]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[16], sv0[17]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[18], sv0[19]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[20], sv0[21]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[22], sv0[23]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[24], sv0[25]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[26], sv0[27]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[28], sv0[29]));
                    _reg_reduce_sum2_48 = add_f32x2(_reg_reduce_sum2_48, make_float2(sv0[30], sv0[31]));
                    float _reg_array_reduce_47 = _reg_reduce_sum2_48.x + _reg_reduce_sum2_48.y;
                    float2 _reg_reduce_sum2_50 = make_float2(0.0f, 0.0f);
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[0], sv1[1]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[2], sv1[3]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[4], sv1[5]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[6], sv1[7]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[8], sv1[9]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[10], sv1[11]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[12], sv1[13]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[14], sv1[15]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[16], sv1[17]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[18], sv1[19]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[20], sv1[21]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[22], sv1[23]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[24], sv1[25]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[26], sv1[27]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[28], sv1[29]));
                    _reg_reduce_sum2_50 = add_f32x2(_reg_reduce_sum2_50, make_float2(sv1[30], sv1[31]));
                    float _reg_array_reduce_49 = _reg_reduce_sum2_50.x + _reg_reduce_sum2_50.y;
                    float block_sum = _reg_array_reduce_47 + _reg_array_reduce_49;
#endif
                    row_sum = __fmaf_rn(row_sum, alpha, block_sum);
                    float factors[32];
#if __CUDA_ARCH__ == 1030
                    factors[0] = __uint_as_float(fac_0[0] << 16);
                    factors[1] = __uint_as_float(fac_0[0] & 4294901760u);
                    factors[2] = __uint_as_float(fac_0[1] << 16);
                    factors[3] = __uint_as_float(fac_0[1] & 4294901760u);
                    factors[4] = __uint_as_float(fac_0[2] << 16);
                    factors[5] = __uint_as_float(fac_0[2] & 4294901760u);
                    factors[6] = __uint_as_float(fac_0[3] << 16);
                    factors[7] = __uint_as_float(fac_0[3] & 4294901760u);
                    factors[8] = __uint_as_float(fac_0[4] << 16);
                    factors[9] = __uint_as_float(fac_0[4] & 4294901760u);
                    factors[10] = __uint_as_float(fac_0[5] << 16);
                    factors[11] = __uint_as_float(fac_0[5] & 4294901760u);
                    factors[12] = __uint_as_float(fac_0[6] << 16);
                    factors[13] = __uint_as_float(fac_0[6] & 4294901760u);
                    factors[14] = __uint_as_float(fac_0[7] << 16);
                    factors[15] = __uint_as_float(fac_0[7] & 4294901760u);
                    factors[16] = __uint_as_float(fac_0[8] << 16);
                    factors[17] = __uint_as_float(fac_0[8] & 4294901760u);
                    factors[18] = __uint_as_float(fac_0[9] << 16);
                    factors[19] = __uint_as_float(fac_0[9] & 4294901760u);
                    factors[20] = __uint_as_float(fac_0[10] << 16);
                    factors[21] = __uint_as_float(fac_0[10] & 4294901760u);
                    factors[22] = __uint_as_float(fac_0[11] << 16);
                    factors[23] = __uint_as_float(fac_0[11] & 4294901760u);
                    factors[24] = __uint_as_float(fac_0[12] << 16);
                    factors[25] = __uint_as_float(fac_0[12] & 4294901760u);
                    factors[26] = __uint_as_float(fac_0[13] << 16);
                    factors[27] = __uint_as_float(fac_0[13] & 4294901760u);
                    factors[28] = __uint_as_float(fac_0[14] << 16);
                    factors[29] = __uint_as_float(fac_0[14] & 4294901760u);
                    factors[30] = __uint_as_float(fac_0[15] << 16);
                    factors[31] = __uint_as_float(fac_0[15] & 4294901760u);
#else
                    factors[0] = __uint_as_float(fac_0[0]);
                    factors[1] = __uint_as_float(fac_0[1]);
                    factors[2] = __uint_as_float(fac_0[2]);
                    factors[3] = __uint_as_float(fac_0[3]);
                    factors[4] = __uint_as_float(fac_0[4]);
                    factors[5] = __uint_as_float(fac_0[5]);
                    factors[6] = __uint_as_float(fac_0[6]);
                    factors[7] = __uint_as_float(fac_0[7]);
                    factors[8] = __uint_as_float(fac_0[8]);
                    factors[9] = __uint_as_float(fac_0[9]);
                    factors[10] = __uint_as_float(fac_0[10]);
                    factors[11] = __uint_as_float(fac_0[11]);
                    factors[12] = __uint_as_float(fac_0[12]);
                    factors[13] = __uint_as_float(fac_0[13]);
                    factors[14] = __uint_as_float(fac_0[14]);
                    factors[15] = __uint_as_float(fac_0[15]);
                    factors[16] = __uint_as_float(fac_0[16]);
                    factors[17] = __uint_as_float(fac_0[17]);
                    factors[18] = __uint_as_float(fac_0[18]);
                    factors[19] = __uint_as_float(fac_0[19]);
                    factors[20] = __uint_as_float(fac_0[20]);
                    factors[21] = __uint_as_float(fac_0[21]);
                    factors[22] = __uint_as_float(fac_0[22]);
                    factors[23] = __uint_as_float(fac_0[23]);
                    factors[24] = __uint_as_float(fac_0[24]);
                    factors[25] = __uint_as_float(fac_0[25]);
                    factors[26] = __uint_as_float(fac_0[26]);
                    factors[27] = __uint_as_float(fac_0[27]);
                    factors[28] = __uint_as_float(fac_0[28]);
                    factors[29] = __uint_as_float(fac_0[29]);
                    factors[30] = __uint_as_float(fac_0[30]);
                    factors[31] = __uint_as_float(fac_0[31]);
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(sv0)[_ls], reinterpret_cast<const float2*>(factors)[_ls]);
                    float factors_0[32];
#if __CUDA_ARCH__ == 1030
                    factors_0[0] = __uint_as_float(fac_1[0] << 16);
                    factors_0[1] = __uint_as_float(fac_1[0] & 4294901760u);
                    factors_0[2] = __uint_as_float(fac_1[1] << 16);
                    factors_0[3] = __uint_as_float(fac_1[1] & 4294901760u);
                    factors_0[4] = __uint_as_float(fac_1[2] << 16);
                    factors_0[5] = __uint_as_float(fac_1[2] & 4294901760u);
                    factors_0[6] = __uint_as_float(fac_1[3] << 16);
                    factors_0[7] = __uint_as_float(fac_1[3] & 4294901760u);
                    factors_0[8] = __uint_as_float(fac_1[4] << 16);
                    factors_0[9] = __uint_as_float(fac_1[4] & 4294901760u);
                    factors_0[10] = __uint_as_float(fac_1[5] << 16);
                    factors_0[11] = __uint_as_float(fac_1[5] & 4294901760u);
                    factors_0[12] = __uint_as_float(fac_1[6] << 16);
                    factors_0[13] = __uint_as_float(fac_1[6] & 4294901760u);
                    factors_0[14] = __uint_as_float(fac_1[7] << 16);
                    factors_0[15] = __uint_as_float(fac_1[7] & 4294901760u);
                    factors_0[16] = __uint_as_float(fac_1[8] << 16);
                    factors_0[17] = __uint_as_float(fac_1[8] & 4294901760u);
                    factors_0[18] = __uint_as_float(fac_1[9] << 16);
                    factors_0[19] = __uint_as_float(fac_1[9] & 4294901760u);
                    factors_0[20] = __uint_as_float(fac_1[10] << 16);
                    factors_0[21] = __uint_as_float(fac_1[10] & 4294901760u);
                    factors_0[22] = __uint_as_float(fac_1[11] << 16);
                    factors_0[23] = __uint_as_float(fac_1[11] & 4294901760u);
                    factors_0[24] = __uint_as_float(fac_1[12] << 16);
                    factors_0[25] = __uint_as_float(fac_1[12] & 4294901760u);
                    factors_0[26] = __uint_as_float(fac_1[13] << 16);
                    factors_0[27] = __uint_as_float(fac_1[13] & 4294901760u);
                    factors_0[28] = __uint_as_float(fac_1[14] << 16);
                    factors_0[29] = __uint_as_float(fac_1[14] & 4294901760u);
                    factors_0[30] = __uint_as_float(fac_1[15] << 16);
                    factors_0[31] = __uint_as_float(fac_1[15] & 4294901760u);
#else
                    factors_0[0] = __uint_as_float(fac_1[0]);
                    factors_0[1] = __uint_as_float(fac_1[1]);
                    factors_0[2] = __uint_as_float(fac_1[2]);
                    factors_0[3] = __uint_as_float(fac_1[3]);
                    factors_0[4] = __uint_as_float(fac_1[4]);
                    factors_0[5] = __uint_as_float(fac_1[5]);
                    factors_0[6] = __uint_as_float(fac_1[6]);
                    factors_0[7] = __uint_as_float(fac_1[7]);
                    factors_0[8] = __uint_as_float(fac_1[8]);
                    factors_0[9] = __uint_as_float(fac_1[9]);
                    factors_0[10] = __uint_as_float(fac_1[10]);
                    factors_0[11] = __uint_as_float(fac_1[11]);
                    factors_0[12] = __uint_as_float(fac_1[12]);
                    factors_0[13] = __uint_as_float(fac_1[13]);
                    factors_0[14] = __uint_as_float(fac_1[14]);
                    factors_0[15] = __uint_as_float(fac_1[15]);
                    factors_0[16] = __uint_as_float(fac_1[16]);
                    factors_0[17] = __uint_as_float(fac_1[17]);
                    factors_0[18] = __uint_as_float(fac_1[18]);
                    factors_0[19] = __uint_as_float(fac_1[19]);
                    factors_0[20] = __uint_as_float(fac_1[20]);
                    factors_0[21] = __uint_as_float(fac_1[21]);
                    factors_0[22] = __uint_as_float(fac_1[22]);
                    factors_0[23] = __uint_as_float(fac_1[23]);
                    factors_0[24] = __uint_as_float(fac_1[24]);
                    factors_0[25] = __uint_as_float(fac_1[25]);
                    factors_0[26] = __uint_as_float(fac_1[26]);
                    factors_0[27] = __uint_as_float(fac_1[27]);
                    factors_0[28] = __uint_as_float(fac_1[28]);
                    factors_0[29] = __uint_as_float(fac_1[29]);
                    factors_0[30] = __uint_as_float(fac_1[30]);
                    factors_0[31] = __uint_as_float(fac_1[31]);
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(sv1)[_ls], reinterpret_cast<const float2*>(factors_0)[_ls]);
                    sh_s += 1;
                    if (sh_s == 3) { sh_s = 0; _phase_shift_full ^= 1; }
                    unsigned int p_base = smem_p_addr + (unsigned int)(p_s * 8192);
                    unsigned int p_fp8_0[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[0]), "f"(sv0[1]),
                                               "f"(sv0[2]), "f"(sv0[3]));
                        p_fp8_0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[4]), "f"(sv0[5]),
                                               "f"(sv0[6]), "f"(sv0[7]));
                        p_fp8_0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[8]), "f"(sv0[9]),
                                               "f"(sv0[10]), "f"(sv0[11]));
                        p_fp8_0[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[12]), "f"(sv0[13]),
                                               "f"(sv0[14]), "f"(sv0[15]));
                        p_fp8_0[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[16]), "f"(sv0[17]),
                                               "f"(sv0[18]), "f"(sv0[19]));
                        p_fp8_0[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[20]), "f"(sv0[21]),
                                               "f"(sv0[22]), "f"(sv0[23]));
                        p_fp8_0[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[24]), "f"(sv0[25]),
                                               "f"(sv0[26]), "f"(sv0[27]));
                        p_fp8_0[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[28]), "f"(sv0[29]),
                                               "f"(sv0[30]), "f"(sv0[31]));
                        p_fp8_0[7] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (unsigned int)(row_c * 128 + n_half * 32 ^ (row_c * 128 + n_half * 32 >> 7 & 7) << 4))), "r"(p_fp8_0[0]), "r"(p_fp8_0[1]), "r"(p_fp8_0[2]), "r"(p_fp8_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (unsigned int)(row_c * 128 + (n_half * 32 + 16) ^ (row_c * 128 + (n_half * 32 + 16) >> 7 & 7) << 4))), "r"(p_fp8_0[4]), "r"(p_fp8_0[5]), "r"(p_fp8_0[6]), "r"(p_fp8_0[7]) : "memory");
                    unsigned int p_fp8_1[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[0]), "f"(sv1[1]),
                                               "f"(sv1[2]), "f"(sv1[3]));
                        p_fp8_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[4]), "f"(sv1[5]),
                                               "f"(sv1[6]), "f"(sv1[7]));
                        p_fp8_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[8]), "f"(sv1[9]),
                                               "f"(sv1[10]), "f"(sv1[11]));
                        p_fp8_1[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[12]), "f"(sv1[13]),
                                               "f"(sv1[14]), "f"(sv1[15]));
                        p_fp8_1[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[16]), "f"(sv1[17]),
                                               "f"(sv1[18]), "f"(sv1[19]));
                        p_fp8_1[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[20]), "f"(sv1[21]),
                                               "f"(sv1[22]), "f"(sv1[23]));
                        p_fp8_1[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[24]), "f"(sv1[25]),
                                               "f"(sv1[26]), "f"(sv1[27]));
                        p_fp8_1[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[28]), "f"(sv1[29]),
                                               "f"(sv1[30]), "f"(sv1[31]));
                        p_fp8_1[7] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (unsigned int)(row_c * 128 + (64 + n_half * 32) ^ (row_c * 128 + (64 + n_half * 32) >> 7 & 7) << 4))), "r"(p_fp8_1[0]), "r"(p_fp8_1[1]), "r"(p_fp8_1[2]), "r"(p_fp8_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (unsigned int)(row_c * 128 + (64 + n_half * 32 + 16) ^ (row_c * 128 + (64 + n_half * 32 + 16) >> 7 & 7) << 4))), "r"(p_fp8_1[4]), "r"(p_fp8_1[5]), "r"(p_fp8_1[6]), "r"(p_fp8_1[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (p_s) * 8), "r"(0) : "memory");
                    p_s += 1;
                    if (p_s == 2) { p_s = 0; _phase_p_empty ^= 1; }
                }
                smem_xchg_sum[xchg_idx] = row_sum;
                asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                row_sum = row_sum + smem_xchg_sum[xchg_idx ^ 64];
                asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                int direct_out = num_split == 1;
                float out_scale = ((direct_out != 0) ? bmm2_scale : 1.0f);
#if __CUDA_ARCH__ == 1030
                float _rcp_50 = approx_rcp(row_sum);
                float inv_sum = ((row_sum > 0.0f) ? _rcp_50 * (out_scale * 8.0f) : 0.0f);
#else
                float _rcp_51 = approx_rcp(row_sum);
                float inv_sum = ((row_sum > 0.0f) ? _rcp_51 * (out_scale * 8.0f) : 0.0f);
#endif
                int o_elem_base = (out_row * num_split + split_idx) * 512 + n_half * 128;
                int warp_bulk = rows_valid >= warp_row0_local + 32 && empty == 0;
                int row_g0 = row_base + warp_row0_local;
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float o_ra[32];
                float o_rb[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 96));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_51 = {inv_sum, inv_sum};
#else
                const float2 _scale2_52 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_51);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_52);
#endif
                unsigned int o_bf16_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                    o_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_0 = warp_tiles_base;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_0 + (unsigned int)(lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_bf16_0[0]), "r"(o_bf16_0[1]), "r"(o_bf16_0[2]), "r"(o_bf16_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_0 + (unsigned int)(lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_bf16_0[4]), "r"(o_bf16_0[5]), "r"(o_bf16_0[6]), "r"(o_bf16_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_0 + (unsigned int)(lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_bf16_0[8]), "r"(o_bf16_0[9]), "r"(o_bf16_0[10]), "r"(o_bf16_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_0 + (unsigned int)(lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_bf16_0[12]), "r"(o_bf16_0[13]), "r"(o_bf16_0[14]), "r"(o_bf16_0[15]) : "memory");
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + o_elem_base)[0] = reinterpret_cast<int4*>(o_bf16_0)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 8))[0] = reinterpret_cast<int4*>(o_bf16_0 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 16))[0] = reinterpret_cast<int4*>(o_bf16_0 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 24))[0] = reinterpret_cast<int4*>(o_bf16_0 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 128));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_52 = {inv_sum, inv_sum};
#else
                const float2 _scale2_53 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_52);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_53);
#endif
                unsigned int o_bf16_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                    o_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_1 = warp_tiles_base;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (unsigned int)(lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_bf16_1[0]), "r"(o_bf16_1[1]), "r"(o_bf16_1[2]), "r"(o_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (unsigned int)(lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_bf16_1[4]), "r"(o_bf16_1[5]), "r"(o_bf16_1[6]), "r"(o_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (unsigned int)(lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_bf16_1[8]), "r"(o_bf16_1[9]), "r"(o_bf16_1[10]), "r"(o_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_1 + (unsigned int)(lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_bf16_1[12]), "r"(o_bf16_1[13]), "r"(o_bf16_1[14]), "r"(o_bf16_1[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_store_3d((&tmap_po), n_half * 128, split_idx, row_g0, warp_tile_1);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 32))[0] = reinterpret_cast<int4*>(o_bf16_1)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 40))[0] = reinterpret_cast<int4*>(o_bf16_1 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 48))[0] = reinterpret_cast<int4*>(o_bf16_1 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 56))[0] = reinterpret_cast<int4*>(o_bf16_1 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 160));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_53 = {inv_sum, inv_sum};
#else
                const float2 _scale2_54 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_53);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_54);
#endif
                unsigned int o_bf16_2[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                    o_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_2 = warp_tiles_base + 4096;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (unsigned int)(lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_bf16_2[0]), "r"(o_bf16_2[1]), "r"(o_bf16_2[2]), "r"(o_bf16_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (unsigned int)(lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_bf16_2[4]), "r"(o_bf16_2[5]), "r"(o_bf16_2[6]), "r"(o_bf16_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (unsigned int)(lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_bf16_2[8]), "r"(o_bf16_2[9]), "r"(o_bf16_2[10]), "r"(o_bf16_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_2 + (unsigned int)(lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_bf16_2[12]), "r"(o_bf16_2[13]), "r"(o_bf16_2[14]), "r"(o_bf16_2[15]) : "memory");
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 64))[0] = reinterpret_cast<int4*>(o_bf16_2)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 72))[0] = reinterpret_cast<int4*>(o_bf16_2 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 80))[0] = reinterpret_cast<int4*>(o_bf16_2 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 88))[0] = reinterpret_cast<int4*>(o_bf16_2 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 192));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_54 = {inv_sum, inv_sum};
#else
                const float2 _scale2_55 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_54);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_55);
#endif
                unsigned int o_bf16_3[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                    o_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_3 = warp_tiles_base + 4096;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (unsigned int)(lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_bf16_3[0]), "r"(o_bf16_3[1]), "r"(o_bf16_3[2]), "r"(o_bf16_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (unsigned int)(lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_bf16_3[4]), "r"(o_bf16_3[5]), "r"(o_bf16_3[6]), "r"(o_bf16_3[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (unsigned int)(lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_bf16_3[8]), "r"(o_bf16_3[9]), "r"(o_bf16_3[10]), "r"(o_bf16_3[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_3 + (unsigned int)(lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_bf16_3[12]), "r"(o_bf16_3[13]), "r"(o_bf16_3[14]), "r"(o_bf16_3[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_store_3d((&tmap_po), n_half * 128 + 64, split_idx, row_g0, warp_tile_3);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 96))[0] = reinterpret_cast<int4*>(o_bf16_3)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 104))[0] = reinterpret_cast<int4*>(o_bf16_3 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 112))[0] = reinterpret_cast<int4*>(o_bf16_3 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 120))[0] = reinterpret_cast<int4*>(o_bf16_3 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 224));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_55 = {inv_sum, inv_sum};
#else
                const float2 _scale2_56 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_55);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_56);
#endif
                unsigned int o_bf16_4[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                    o_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_4 = warp_tiles_base + 8192;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (unsigned int)(lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_bf16_4[0]), "r"(o_bf16_4[1]), "r"(o_bf16_4[2]), "r"(o_bf16_4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (unsigned int)(lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_bf16_4[4]), "r"(o_bf16_4[5]), "r"(o_bf16_4[6]), "r"(o_bf16_4[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (unsigned int)(lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_bf16_4[8]), "r"(o_bf16_4[9]), "r"(o_bf16_4[10]), "r"(o_bf16_4[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_4 + (unsigned int)(lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_bf16_4[12]), "r"(o_bf16_4[13]), "r"(o_bf16_4[14]), "r"(o_bf16_4[15]) : "memory");
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 256))[0] = reinterpret_cast<int4*>(o_bf16_4)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 264))[0] = reinterpret_cast<int4*>(o_bf16_4 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 272))[0] = reinterpret_cast<int4*>(o_bf16_4 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 280))[0] = reinterpret_cast<int4*>(o_bf16_4 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 256));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_56 = {inv_sum, inv_sum};
#else
                const float2 _scale2_57 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_56);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_57);
#endif
                unsigned int o_bf16_5[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                    o_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_5 = warp_tiles_base + 8192;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (unsigned int)(lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_bf16_5[0]), "r"(o_bf16_5[1]), "r"(o_bf16_5[2]), "r"(o_bf16_5[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (unsigned int)(lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_bf16_5[4]), "r"(o_bf16_5[5]), "r"(o_bf16_5[6]), "r"(o_bf16_5[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (unsigned int)(lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_bf16_5[8]), "r"(o_bf16_5[9]), "r"(o_bf16_5[10]), "r"(o_bf16_5[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_5 + (unsigned int)(lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_bf16_5[12]), "r"(o_bf16_5[13]), "r"(o_bf16_5[14]), "r"(o_bf16_5[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_store_3d((&tmap_po), n_half * 128 + 256, split_idx, row_g0, warp_tile_5);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 288))[0] = reinterpret_cast<int4*>(o_bf16_5)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 296))[0] = reinterpret_cast<int4*>(o_bf16_5 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 304))[0] = reinterpret_cast<int4*>(o_bf16_5 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 312))[0] = reinterpret_cast<int4*>(o_bf16_5 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 288));
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_57 = {inv_sum, inv_sum};
#else
                const float2 _scale2_58 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_57);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_58);
#endif
                unsigned int o_bf16_6[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                    o_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_6 = warp_tiles_base + 12288;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (unsigned int)(lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_bf16_6[0]), "r"(o_bf16_6[1]), "r"(o_bf16_6[2]), "r"(o_bf16_6[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (unsigned int)(lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_bf16_6[4]), "r"(o_bf16_6[5]), "r"(o_bf16_6[6]), "r"(o_bf16_6[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (unsigned int)(lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_bf16_6[8]), "r"(o_bf16_6[9]), "r"(o_bf16_6[10]), "r"(o_bf16_6[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_6 + (unsigned int)(lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_bf16_6[12]), "r"(o_bf16_6[13]), "r"(o_bf16_6[14]), "r"(o_bf16_6[15]) : "memory");
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 320))[0] = reinterpret_cast<int4*>(o_bf16_6)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 328))[0] = reinterpret_cast<int4*>(o_bf16_6 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 336))[0] = reinterpret_cast<int4*>(o_bf16_6 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 344))[0] = reinterpret_cast<int4*>(o_bf16_6 + (12))[0];
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_empty_addr), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1030
                const float2 _scale2_58 = {inv_sum, inv_sum};
#else
                const float2 _scale2_59 = {inv_sum, inv_sum};
#endif
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_58);
#else
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_59);
#endif
                unsigned int o_bf16_7[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                    o_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                if (warp_bulk != 0) {
                    unsigned int warp_tile_7 = warp_tiles_base + 12288;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (unsigned int)(lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_bf16_7[0]), "r"(o_bf16_7[1]), "r"(o_bf16_7[2]), "r"(o_bf16_7[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (unsigned int)(lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_bf16_7[4]), "r"(o_bf16_7[5]), "r"(o_bf16_7[6]), "r"(o_bf16_7[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (unsigned int)(lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_bf16_7[8]), "r"(o_bf16_7[9]), "r"(o_bf16_7[10]), "r"(o_bf16_7[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile_7 + (unsigned int)(lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_bf16_7[12]), "r"(o_bf16_7[13]), "r"(o_bf16_7[14]), "r"(o_bf16_7[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_store_3d((&tmap_po), n_half * 128 + 320, split_idx, row_g0, warp_tile_7);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else if (row_valid != 0) {
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 352))[0] = reinterpret_cast<int4*>(o_bf16_7)[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 360))[0] = reinterpret_cast<int4*>(o_bf16_7 + (4))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 368))[0] = reinterpret_cast<int4*>(o_bf16_7 + (8))[0];
                    reinterpret_cast<int4*>(partial_O + (o_elem_base + 376))[0] = reinterpret_cast<int4*>(o_bf16_7 + (12))[0];
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                __syncwarp();
                mbarrier_arrive(epi_done_addr);
                _phase_o_full_0 ^= 1;
                if (row_is_query != 0 && n_half == 0) {
                    int stat_off = out_row * num_split + split_idx;
                    if (direct_out != 0) {
                        if (has_lse != 0) {
                            float safe_t = ((row_sum > 0.0f) ? row_sum : 1.0f);
#if __CUDA_ARCH__ == 1030
                            float _log2_59;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_59) : "f"(safe_t));
                            float lse_l2 = row_max + _log2_59 - 6.0f;
#else
                            float _log2_60;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_60) : "f"(safe_t));
                            float lse_l2 = row_max + _log2_60 - 6.0f;
#endif
                            lse[out_row] = ((row_sum > 0.0f) ? lse_l2 * 0.6931472f : -CUDART_INF_F);
                        }
                    } else {
                        partial_max[stat_off] = row_max;
                        partial_sum[stat_off] = row_sum;
                    }
                }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_main
            int quadrant_c = make_warp_uniform(warp % 4);
            int row_cc = quadrant_c % 2 * 32 + lane;
            int lane_base_c = quadrant_c * 32 << 16;
#if __CUDA_ARCH__ == 1030
#else
            int shift_warp_c = make_warp_uniform(warp % 2);
#endif
            int raw_c = 0;
            int p_c = 0;
            int pub_tile = 0;
            int cx_1 = blockIdx.x / 2;
            int n_clusters_1 = gridDim.x / 2;
#if __CUDA_ARCH__ == 1030
            int q_end_r_1 = q_end_k;
            int q_start_r_1 = q_start_k;
            int kv_len_r_1 = kv_len_k;
            int g_len_r_1 = g_len_k;
#else
            int q_end_r_1 = cum_seq_lens_q[blockIdx.z + 1];
            int q_start_r_1 = cum_seq_lens_q[blockIdx.z];
            int kv_len_r_1 = seq_lens[blockIdx.z];
            int g_len_r_1 = kv_len_global[blockIdx.z];
#endif
            int pieces_1 = ((q_end_r_1 - q_start_r_1) * num_heads + 127) / 128 * num_split;
            int left_1 = pieces_1 - cx_1;
            int n_pieces_raw_1 = (left_1 + n_clusters_1 - 1) / n_clusters_1;
            int n_pieces_1 = ((left_1 <= 0) ? 0 : n_pieces_raw_1);
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_sf_empty_0 = 1;
            unsigned int _phase_p_empty_1 = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_pv_done_0 = 1;
            #pragma unroll 1
            for (int pi_c = 0; pi_c < n_pieces_1; pi_c++) {
                int split_idx_1 = (cx_1 + pi_c * n_clusters_1) % num_split;
                int q_len_b_1 = q_end_r_1 - q_start_r_1;
                int rows_b_1 = q_len_b_1 * num_heads;
                int m_tile_1 = (rows_b_1 + 127) / 128 - 1 - (cx_1 + pi_c * n_clusters_1) / num_split;
                int row0_1 = m_tile_1 * 128;
                int rows_left_1 = rows_b_1 - row0_1;
                int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
                int rows_valid_1 = ((rows_pos_1 > 128) ? 128 : rows_pos_1);
                int row_base_1 = q_start_r_1 * num_heads + row0_1;
                int t_last_1 = (row0_1 + rows_valid_1 - 1) / num_heads;
                int num_1 = g_len_r_1 - q_len_b_1 + t_last_1 - cp_rank;
                int vis_raw_1 = num_1 / cp_world + 1;
                int vis_cap_1 = ((vis_raw_1 > kv_len_r_1) ? kv_len_r_1 : vis_raw_1);
                int vis_1 = ((num_1 < 0) ? 0 : vis_cap_1);
                int kv_end_1 = ((rows_valid_1 == 0) ? 0 : vis_1);
                int n_tiles_total_1 = (kv_end_1 + 127) / 128;
                int tiles_per_split_1 = (n_tiles_total_1 + num_split - 1) / num_split;
                int my_start_raw_1 = split_idx_1 * tiles_per_split_1;
                int my_end_raw_1 = my_start_raw_1 + tiles_per_split_1;
                int my_end_1 = ((my_end_raw_1 > n_tiles_total_1) ? n_tiles_total_1 : my_end_raw_1);
                int my_n_raw_1 = my_end_1 - my_start_raw_1;
                int empty_1 = ((my_n_raw_1 < 1) ? 1 : 0);
                int my_n_tiles_1 = ((my_n_raw_1 < 1) ? 1 : my_n_raw_1);
                int my_start_1 = ((my_n_raw_1 < 1) ? 0 : my_start_raw_1);
                pub_tile = 0;
                int q_dead = cta_rank * 64 + quadrant_c % 2 * 32 >= rows_valid_1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int q_words[8];
                unsigned int qs_addr = smem_qs_addr + (unsigned int)(row_cc * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(0) + 3]))
                    : "r"(qs_addr));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words[(4) + 3]))
                    : "r"(qs_addr + 16));
                unsigned int q_row_words[64];
                int q_swz = row_cc >> 1 & 3;
                unsigned int q_atom_addr_0 = smem_q_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(0) + 3]))
                    : "r"(q_atom_addr_0 + (unsigned int)((q_swz ^ 0) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(4) + 3]))
                    : "r"(q_atom_addr_0 + (unsigned int)((q_swz ^ 1) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(8) + 3]))
                    : "r"(q_atom_addr_0 + (unsigned int)((q_swz ^ 2) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(12) + 3]))
                    : "r"(q_atom_addr_0 + (unsigned int)((q_swz ^ 3) * 16)));
                unsigned int q_atom_addr_1 = smem_q_addr + (unsigned int)(4096 + row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[16])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(16) + 3]))
                    : "r"(q_atom_addr_1 + (unsigned int)((q_swz ^ 0) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[20])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(20) + 3]))
                    : "r"(q_atom_addr_1 + (unsigned int)((q_swz ^ 1) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[24])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(24) + 3]))
                    : "r"(q_atom_addr_1 + (unsigned int)((q_swz ^ 2) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[28])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(28) + 3]))
                    : "r"(q_atom_addr_1 + (unsigned int)((q_swz ^ 3) * 16)));
                unsigned int q_atom_addr_2 = smem_q_addr + (unsigned int)(8192 + row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[32])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(32) + 3]))
                    : "r"(q_atom_addr_2 + (unsigned int)((q_swz ^ 0) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[36])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(36) + 3]))
                    : "r"(q_atom_addr_2 + (unsigned int)((q_swz ^ 1) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[40])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(40) + 3]))
                    : "r"(q_atom_addr_2 + (unsigned int)((q_swz ^ 2) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[44])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(44) + 3]))
                    : "r"(q_atom_addr_2 + (unsigned int)((q_swz ^ 3) * 16)));
                unsigned int q_atom_addr_3 = smem_q_addr + (unsigned int)(12288 + row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[48])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(48) + 3]))
                    : "r"(q_atom_addr_3 + (unsigned int)((q_swz ^ 0) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[52])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(52) + 3]))
                    : "r"(q_atom_addr_3 + (unsigned int)((q_swz ^ 1) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[56])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(56) + 3]))
                    : "r"(q_atom_addr_3 + (unsigned int)((q_swz ^ 2) * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[60])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words[(60) + 3]))
                    : "r"(q_atom_addr_3 + (unsigned int)((q_swz ^ 3) * 16)));
                if (q_dead != 0) {
                    q_words[0] = 0;
                    q_words[1] = 0;
                    q_words[2] = 0;
                    q_words[3] = 0;
                    q_words[4] = 0;
                    q_words[5] = 0;
                    q_words[6] = 0;
                    q_words[7] = 0;
                    q_row_words[0] = 0;
                    q_row_words[1] = 0;
                    q_row_words[2] = 0;
                    q_row_words[3] = 0;
                    q_row_words[4] = 0;
                    q_row_words[5] = 0;
                    q_row_words[6] = 0;
                    q_row_words[7] = 0;
                    q_row_words[8] = 0;
                    q_row_words[9] = 0;
                    q_row_words[10] = 0;
                    q_row_words[11] = 0;
                    q_row_words[12] = 0;
                    q_row_words[13] = 0;
                    q_row_words[14] = 0;
                    q_row_words[15] = 0;
                    q_row_words[16] = 0;
                    q_row_words[17] = 0;
                    q_row_words[18] = 0;
                    q_row_words[19] = 0;
                    q_row_words[20] = 0;
                    q_row_words[21] = 0;
                    q_row_words[22] = 0;
                    q_row_words[23] = 0;
                    q_row_words[24] = 0;
                    q_row_words[25] = 0;
                    q_row_words[26] = 0;
                    q_row_words[27] = 0;
                    q_row_words[28] = 0;
                    q_row_words[29] = 0;
                    q_row_words[30] = 0;
                    q_row_words[31] = 0;
                    q_row_words[32] = 0;
                    q_row_words[33] = 0;
                    q_row_words[34] = 0;
                    q_row_words[35] = 0;
                    q_row_words[36] = 0;
                    q_row_words[37] = 0;
                    q_row_words[38] = 0;
                    q_row_words[39] = 0;
                    q_row_words[40] = 0;
                    q_row_words[41] = 0;
                    q_row_words[42] = 0;
                    q_row_words[43] = 0;
                    q_row_words[44] = 0;
                    q_row_words[45] = 0;
                    q_row_words[46] = 0;
                    q_row_words[47] = 0;
                    q_row_words[48] = 0;
                    q_row_words[49] = 0;
                    q_row_words[50] = 0;
                    q_row_words[51] = 0;
                    q_row_words[52] = 0;
                    q_row_words[53] = 0;
                    q_row_words[54] = 0;
                    q_row_words[55] = 0;
                    q_row_words[56] = 0;
                    q_row_words[57] = 0;
                    q_row_words[58] = 0;
                    q_row_words[59] = 0;
                    q_row_words[60] = 0;
                    q_row_words[61] = 0;
                    q_row_words[62] = 0;
                    q_row_words[63] = 0;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416), "r"(q_row_words[0]), "r"(q_row_words[1]), "r"(q_row_words[2]), "r"(q_row_words[3]), "r"(q_row_words[4]), "r"(q_row_words[5]), "r"(q_row_words[6]), "r"(q_row_words[7]), "r"(q_row_words[8]), "r"(q_row_words[9]), "r"(q_row_words[10]), "r"(q_row_words[11]), "r"(q_row_words[12]), "r"(q_row_words[13]), "r"(q_row_words[14]), "r"(q_row_words[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 432), "r"((q_row_words + 16)[0]), "r"((q_row_words + 16)[1]), "r"((q_row_words + 16)[2]), "r"((q_row_words + 16)[3]), "r"((q_row_words + 16)[4]), "r"((q_row_words + 16)[5]), "r"((q_row_words + 16)[6]), "r"((q_row_words + 16)[7]), "r"((q_row_words + 16)[8]), "r"((q_row_words + 16)[9]), "r"((q_row_words + 16)[10]), "r"((q_row_words + 16)[11]), "r"((q_row_words + 16)[12]), "r"((q_row_words + 16)[13]), "r"((q_row_words + 16)[14]), "r"((q_row_words + 16)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 448), "r"((q_row_words + 32)[0]), "r"((q_row_words + 32)[1]), "r"((q_row_words + 32)[2]), "r"((q_row_words + 32)[3]), "r"((q_row_words + 32)[4]), "r"((q_row_words + 32)[5]), "r"((q_row_words + 32)[6]), "r"((q_row_words + 32)[7]), "r"((q_row_words + 32)[8]), "r"((q_row_words + 32)[9]), "r"((q_row_words + 32)[10]), "r"((q_row_words + 32)[11]), "r"((q_row_words + 32)[12]), "r"((q_row_words + 32)[13]), "r"((q_row_words + 32)[14]), "r"((q_row_words + 32)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 464), "r"((q_row_words + 48)[0]), "r"((q_row_words + 48)[1]), "r"((q_row_words + 48)[2]), "r"((q_row_words + 48)[3]), "r"((q_row_words + 48)[4]), "r"((q_row_words + 48)[5]), "r"((q_row_words + 48)[6]), "r"((q_row_words + 48)[7]), "r"((q_row_words + 48)[8]), "r"((q_row_words + 48)[9]), "r"((q_row_words + 48)[10]), "r"((q_row_words + 48)[11]), "r"((q_row_words + 48)[12]), "r"((q_row_words + 48)[13]), "r"((q_row_words + 48)[14]), "r"((q_row_words + 48)[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320), "r"(q_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 321), "r"(q_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 322), "r"(q_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 323), "r"(q_words[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 324), "r"((q_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 325), "r"((q_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 326), "r"((q_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 327), "r"((q_words + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 328), "r"((q_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 329), "r"((q_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 330), "r"((q_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 331), "r"((q_words + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 332), "r"((q_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 333), "r"((q_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 334), "r"((q_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 335), "r"((q_words + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 336), "r"((q_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 337), "r"((q_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 338), "r"((q_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 339), "r"((q_words + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 340), "r"((q_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 341), "r"((q_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 342), "r"((q_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 343), "r"((q_words + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 344), "r"((q_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 345), "r"((q_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 346), "r"((q_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 347), "r"((q_words + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 348), "r"((q_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 349), "r"((q_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 350), "r"((q_words + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 351), "r"((q_words + 7)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sfa_full_addr), "r"(0) : "memory");
                mbarrier_arrive(q_empty_addr);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(raw_full_addr + (raw_c) * 8, _phase_raw_full);
                mbarrier_wait(sf_empty_addr, _phase_sf_empty_0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int tok_c_0 = quadrant_c / 2 * 32 + lane;
                unsigned int ks_addr_0 = smem_ks_addr + (unsigned int)(raw_c * 32768 + tok_c_0 * 32);
                unsigned int k_words_0[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[(0) + 3]))
                    : "r"(ks_addr_0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_0[(4) + 3]))
                    : "r"(ks_addr_0 + 16));
#if __CUDA_ARCH__ == 1030
                if ((my_start_1 + pub_tile) * 128 + tok_c_0 >= kv_end_1 || empty_1 != 0) {
                    k_words_0[0] = 0;
                    k_words_0[1] = 0;
                    k_words_0[2] = 0;
                    k_words_0[3] = 0;
                    k_words_0[4] = 0;
                    k_words_0[5] = 0;
                    k_words_0[6] = 0;
                    k_words_0[7] = 0;
                }
#else
#endif
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 352), "r"(k_words_0[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 356), "r"((k_words_0 + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 360), "r"((k_words_0 + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 364), "r"((k_words_0 + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 368), "r"((k_words_0 + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 372), "r"((k_words_0 + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 376), "r"((k_words_0 + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 380), "r"((k_words_0 + 7)[0]));
                int tok_c_1 = 64 + quadrant_c / 2 * 32 + lane;
                unsigned int ks_addr_1 = smem_ks_addr + (unsigned int)(raw_c * 32768 + tok_c_1 * 32);
                unsigned int k_words_1[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[(0) + 3]))
                    : "r"(ks_addr_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_1[(4) + 3]))
                    : "r"(ks_addr_1 + 16));
#if __CUDA_ARCH__ == 1030
                if ((my_start_1 + pub_tile) * 128 + tok_c_1 >= kv_end_1 || empty_1 != 0) {
                    k_words_1[0] = 0;
                    k_words_1[1] = 0;
                    k_words_1[2] = 0;
                    k_words_1[3] = 0;
                    k_words_1[4] = 0;
                    k_words_1[5] = 0;
                    k_words_1[6] = 0;
                    k_words_1[7] = 0;
                }
#else
#endif
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 384), "r"(k_words_1[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 388), "r"((k_words_1 + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 392), "r"((k_words_1 + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 396), "r"((k_words_1 + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 400), "r"((k_words_1 + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 404), "r"((k_words_1 + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 408), "r"((k_words_1 + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 412), "r"((k_words_1 + 7)[0]));
                mbarrier_arrive(raw_empty_addr + (raw_c) * 8);
#if __CUDA_ARCH__ == 1030
                raw_c += 1;
                if (raw_c == 3) { raw_c = 0; _phase_raw_full ^= 1; }
#else
#endif
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sf_full_addr), "r"(0) : "memory");
                _phase_sf_empty_0 ^= 1;
#if __CUDA_ARCH__ == 1030
#else
                if (shift_warp_c != 0) {
                    int shift_pos_0 = (my_start_1 + pub_tile) * 128 + tok_c_0;
                    int shift_masked_0 = shift_pos_0 >= kv_end_1 || empty_1 != 0;
                    unsigned int shift_live_0 = ((shift_masked_0 != 0) ? 0 : 1);
                    unsigned int above_1 = 0;
                    unsigned int above_2 = 0;
                    unsigned int above_3 = 0;
                    above_1 = above_1 | k_words_0[0] + 370546198;
                    above_2 = above_2 | k_words_0[0] + 235802126;
                    above_3 = above_3 | k_words_0[0] + 101058054;
                    above_1 = above_1 | k_words_0[1] + 370546198;
                    above_2 = above_2 | k_words_0[1] + 235802126;
                    above_3 = above_3 | k_words_0[1] + 101058054;
                    above_1 = above_1 | k_words_0[2] + 370546198;
                    above_2 = above_2 | k_words_0[2] + 235802126;
                    above_3 = above_3 | k_words_0[2] + 101058054;
                    above_1 = above_1 | k_words_0[3] + 370546198;
                    above_2 = above_2 | k_words_0[3] + 235802126;
                    above_3 = above_3 | k_words_0[3] + 101058054;
                    above_1 = above_1 | k_words_0[4] + 370546198;
                    above_2 = above_2 | k_words_0[4] + 235802126;
                    above_3 = above_3 | k_words_0[4] + 101058054;
                    above_1 = above_1 | k_words_0[5] + 370546198;
                    above_2 = above_2 | k_words_0[5] + 235802126;
                    above_3 = above_3 | k_words_0[5] + 101058054;
                    above_1 = above_1 | k_words_0[6] + 370546198;
                    above_2 = above_2 | k_words_0[6] + 235802126;
                    above_3 = above_3 | k_words_0[6] + 101058054;
                    above_1 = above_1 | k_words_0[7] + 370546198;
                    above_2 = above_2 | k_words_0[7] + 235802126;
                    above_3 = above_3 | k_words_0[7] + 101058054;
                    unsigned int _min_0 = ((above_1 & 2155905152u) < (shift_live_0) ? (above_1 & 2155905152u) : (shift_live_0));
                    unsigned int _min_1 = ((above_2 & 2155905152u) < (shift_live_0) ? (above_2 & 2155905152u) : (shift_live_0));
                    unsigned int _min_2 = ((above_3 & 2155905152u) < (shift_live_0) ? (above_3 & 2155905152u) : (shift_live_0));
                    unsigned int shift = _min_0 + _min_1 + _min_2;
                    unsigned int shift_mul_0 = shift_live_0 * 1006648320 - shift * 67109888;
                    unsigned int shift_off_0 = (unsigned int)(raw_c * 512 + tok_c_0 * 4);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + shift_off_0), "r"(shift_mul_0) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_fac_addr + shift_off_0), "r"(124 + shift << 23) : "memory");
                    int shift_pos_1 = (my_start_1 + pub_tile) * 128 + tok_c_1;
                    int shift_masked_1 = shift_pos_1 >= kv_end_1 || empty_1 != 0;
                    unsigned int shift_live_1 = ((shift_masked_1 != 0) ? 0 : 1);
                    unsigned int above_1_0 = 0;
                    unsigned int above_2_0 = 0;
                    unsigned int above_3_0 = 0;
                    above_1_0 = above_1_0 | k_words_1[0] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[0] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[0] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[1] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[1] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[1] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[2] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[2] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[2] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[3] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[3] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[3] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[4] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[4] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[4] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[5] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[5] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[5] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[6] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[6] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[6] + 101058054;
                    above_1_0 = above_1_0 | k_words_1[7] + 370546198;
                    above_2_0 = above_2_0 | k_words_1[7] + 235802126;
                    above_3_0 = above_3_0 | k_words_1[7] + 101058054;
                    unsigned int _min_3 = ((above_1_0 & 2155905152u) < (shift_live_1) ? (above_1_0 & 2155905152u) : (shift_live_1));
                    unsigned int _min_4 = ((above_2_0 & 2155905152u) < (shift_live_1) ? (above_2_0 & 2155905152u) : (shift_live_1));
                    unsigned int _min_5 = ((above_3_0 & 2155905152u) < (shift_live_1) ? (above_3_0 & 2155905152u) : (shift_live_1));
                    unsigned int shift_0 = _min_3 + _min_4 + _min_5;
                    unsigned int shift_mul_1 = shift_live_1 * 1006648320 - shift_0 * 67109888;
                    unsigned int shift_off_1 = (unsigned int)(raw_c * 512 + tok_c_1 * 4);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + shift_off_1), "r"(shift_mul_1) : "memory");
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_fac_addr + shift_off_1), "r"(124 + shift_0 << 23) : "memory");
                }
                mbarrier_arrive(shift_full_addr + (raw_c) * 8);
                raw_c += 1;
                if (raw_c == 3) { raw_c = 0; _phase_raw_full ^= 1; }
#endif
                pub_tile = pub_tile + 1;
                if (my_n_tiles_1 > 1) {
                    mbarrier_wait(raw_full_addr + (raw_c) * 8, _phase_raw_full);
                    mbarrier_wait(sf_empty_addr, _phase_sf_empty_0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int tok_c_2 = quadrant_c / 2 * 32 + lane;
                    unsigned int ks_addr_2 = smem_ks_addr + (unsigned int)(raw_c * 32768 + tok_c_2 * 32);
                    unsigned int k_words_2[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[(0) + 3]))
                        : "r"(ks_addr_2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_2[(4) + 3]))
                        : "r"(ks_addr_2 + 16));
#if __CUDA_ARCH__ == 1030
                    if ((my_start_1 + pub_tile) * 128 + tok_c_2 >= kv_end_1 || empty_1 != 0) {
                        k_words_2[0] = 0;
                        k_words_2[1] = 0;
                        k_words_2[2] = 0;
                        k_words_2[3] = 0;
                        k_words_2[4] = 0;
                        k_words_2[5] = 0;
                        k_words_2[6] = 0;
                        k_words_2[7] = 0;
                    }
#else
#endif
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 352), "r"(k_words_2[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 356), "r"((k_words_2 + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 360), "r"((k_words_2 + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 364), "r"((k_words_2 + 3)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 368), "r"((k_words_2 + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 372), "r"((k_words_2 + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 376), "r"((k_words_2 + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 380), "r"((k_words_2 + 7)[0]));
                    int tok_c_3 = 64 + quadrant_c / 2 * 32 + lane;
                    unsigned int ks_addr_3 = smem_ks_addr + (unsigned int)(raw_c * 32768 + tok_c_3 * 32);
                    unsigned int k_words_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[(0) + 3]))
                        : "r"(ks_addr_3));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_3[(4) + 3]))
                        : "r"(ks_addr_3 + 16));
#if __CUDA_ARCH__ == 1030
                    if ((my_start_1 + pub_tile) * 128 + tok_c_3 >= kv_end_1 || empty_1 != 0) {
                        k_words_3[0] = 0;
                        k_words_3[1] = 0;
                        k_words_3[2] = 0;
                        k_words_3[3] = 0;
                        k_words_3[4] = 0;
                        k_words_3[5] = 0;
                        k_words_3[6] = 0;
                        k_words_3[7] = 0;
                    }
#else
#endif
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 384), "r"(k_words_3[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 388), "r"((k_words_3 + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 392), "r"((k_words_3 + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 396), "r"((k_words_3 + 3)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 400), "r"((k_words_3 + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 404), "r"((k_words_3 + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 408), "r"((k_words_3 + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)lane_base_c + 412), "r"((k_words_3 + 7)[0]));
                    mbarrier_arrive(raw_empty_addr + (raw_c) * 8);
#if __CUDA_ARCH__ == 1030
                    raw_c += 1;
                    if (raw_c == 3) { raw_c = 0; _phase_raw_full ^= 1; }
#else
#endif
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(sf_full_addr), "r"(0) : "memory");
                    _phase_sf_empty_0 ^= 1;
#if __CUDA_ARCH__ == 1030
#else
                    if (shift_warp_c != 0) {
                        int shift_pos_2 = (my_start_1 + pub_tile) * 128 + tok_c_2;
                        int shift_masked_2 = shift_pos_2 >= kv_end_1 || empty_1 != 0;
                        unsigned int shift_live_2 = ((shift_masked_2 != 0) ? 0 : 1);
                        unsigned int above_1_1 = 0;
                        unsigned int above_2_1 = 0;
                        unsigned int above_3_1 = 0;
                        above_1_1 = above_1_1 | k_words_2[0] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[0] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[0] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[1] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[1] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[1] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[2] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[2] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[2] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[3] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[3] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[3] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[4] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[4] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[4] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[5] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[5] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[5] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[6] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[6] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[6] + 101058054;
                        above_1_1 = above_1_1 | k_words_2[7] + 370546198;
                        above_2_1 = above_2_1 | k_words_2[7] + 235802126;
                        above_3_1 = above_3_1 | k_words_2[7] + 101058054;
                        unsigned int _min_6 = ((above_1_1 & 2155905152u) < (shift_live_2) ? (above_1_1 & 2155905152u) : (shift_live_2));
                        unsigned int _min_7 = ((above_2_1 & 2155905152u) < (shift_live_2) ? (above_2_1 & 2155905152u) : (shift_live_2));
                        unsigned int _min_8 = ((above_3_1 & 2155905152u) < (shift_live_2) ? (above_3_1 & 2155905152u) : (shift_live_2));
                        unsigned int shift_1 = _min_6 + _min_7 + _min_8;
                        unsigned int shift_mul_2 = shift_live_2 * 1006648320 - shift_1 * 67109888;
                        unsigned int shift_off_2 = (unsigned int)(raw_c * 512 + tok_c_2 * 4);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + shift_off_2), "r"(shift_mul_2) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_fac_addr + shift_off_2), "r"(124 + shift_1 << 23) : "memory");
                        int shift_pos_3 = (my_start_1 + pub_tile) * 128 + tok_c_3;
                        int shift_masked_3 = shift_pos_3 >= kv_end_1 || empty_1 != 0;
                        unsigned int shift_live_3 = ((shift_masked_3 != 0) ? 0 : 1);
                        unsigned int above_1_1_1 = 0;
                        unsigned int above_2_1_1 = 0;
                        unsigned int above_3_1_1 = 0;
                        above_1_1_1 = above_1_1_1 | k_words_3[0] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[0] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[0] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[1] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[1] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[1] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[2] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[2] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[2] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[3] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[3] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[3] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[4] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[4] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[4] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[5] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[5] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[5] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[6] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[6] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[6] + 101058054;
                        above_1_1_1 = above_1_1_1 | k_words_3[7] + 370546198;
                        above_2_1_1 = above_2_1_1 | k_words_3[7] + 235802126;
                        above_3_1_1 = above_3_1_1 | k_words_3[7] + 101058054;
                        unsigned int _min_9 = ((above_1_1_1 & 2155905152u) < (shift_live_3) ? (above_1_1_1 & 2155905152u) : (shift_live_3));
                        unsigned int _min_10 = ((above_2_1_1 & 2155905152u) < (shift_live_3) ? (above_2_1_1 & 2155905152u) : (shift_live_3));
                        unsigned int _min_11 = ((above_3_1_1 & 2155905152u) < (shift_live_3) ? (above_3_1_1 & 2155905152u) : (shift_live_3));
                        unsigned int shift_1_1 = _min_9 + _min_10 + _min_11;
                        unsigned int shift_mul_3 = shift_live_3 * 1006648320 - shift_1_1 * 67109888;
                        unsigned int shift_off_3 = (unsigned int)(raw_c * 512 + tok_c_3 * 4);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + shift_off_3), "r"(shift_mul_3) : "memory");
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_fac_addr + shift_off_3), "r"(124 + shift_1_1 << 23) : "memory");
                    }
                    mbarrier_arrive(shift_full_addr + (raw_c) * 8);
                    raw_c += 1;
                    if (raw_c == 3) { raw_c = 0; _phase_raw_full ^= 1; }
#endif
                    pub_tile = pub_tile + 1;
                }
                #pragma unroll 1
                for (int it_c = 0; it_c < my_n_tiles_1; it_c++) {
                    mbarrier_wait(p_empty_addr + (p_c) * 8, _phase_p_empty_1);
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    float alpha_c = row_state[row_cc];
                    mbarrier_arrive(corr_empty_addr);
                    _phase_corr_sig_0 ^= 1;
                    int need_rescale = 0;
                    if (it_c > 0) {
#if __CUDA_ARCH__ == 1030
                        int _warp_vote_0 = __all_sync(0xFFFFFFFF, alpha_c == 1.0f);
                        need_rescale = _warp_vote_0 == 0;
#else
                        int _warp_vote_12 = __all_sync(0xFFFFFFFF, alpha_c == 1.0f);
                        need_rescale = _warp_vote_12 == 0;
#endif
                    }
                    if (need_rescale != 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int col = 0; col < 256; col += 32) {
                            unsigned int o_addr = taddr + (unsigned int)lane_base_c + (unsigned int)(64 + col);
                            float o_chunk[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(o_chunk[0]), "=f"(o_chunk[1]), "=f"(o_chunk[2]), "=f"(o_chunk[3]), "=f"(o_chunk[4]), "=f"(o_chunk[5]), "=f"(o_chunk[6]), "=f"(o_chunk[7]), "=f"(o_chunk[8]), "=f"(o_chunk[9]), "=f"(o_chunk[10]), "=f"(o_chunk[11]), "=f"(o_chunk[12]), "=f"(o_chunk[13]), "=f"(o_chunk[14]), "=f"(o_chunk[15]), "=f"(o_chunk[16]), "=f"(o_chunk[17]), "=f"(o_chunk[18]), "=f"(o_chunk[19]), "=f"(o_chunk[20]), "=f"(o_chunk[21]), "=f"(o_chunk[22]), "=f"(o_chunk[23]), "=f"(o_chunk[24]), "=f"(o_chunk[25]), "=f"(o_chunk[26]), "=f"(o_chunk[27]), "=f"(o_chunk[28]), "=f"(o_chunk[29]), "=f"(o_chunk[30]), "=f"(o_chunk[31])
                                : "r"(o_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
#if __CUDA_ARCH__ == 1030
                            const float2 _scale2_1 = {alpha_c, alpha_c};
#else
                            const float2 _scale2_13 = {alpha_c, alpha_c};
#endif
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk)[_ls], _scale2_1);
#else
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_chunk)[_ls], _scale2_13);
#endif
                            tmem_st_x32_f32(o_addr, o_chunk);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (p_c) * 8), "r"(0) : "memory");
                    if (it_c > 0 && need_rescale == 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                    }
                    _phase_pv_done_0 ^= 1;
                    p_c += 1;
                    if (p_c == 2) { p_c = 0; _phase_p_empty_1 ^= 1; }
                    if (it_c + 2 < my_n_tiles_1) {
                        mbarrier_wait(raw_full_addr + (raw_c) * 8, _phase_raw_full);
                        mbarrier_wait(sf_empty_addr, _phase_sf_empty_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int tok_c_4 = quadrant_c / 2 * 32 + lane;
                        unsigned int ks_addr_4 = smem_ks_addr + (unsigned int)(raw_c * 32768 + tok_c_4 * 32);
                        unsigned int k_words_4[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[(0) + 3]))
                            : "r"(ks_addr_4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_4[(4) + 3]))
                            : "r"(ks_addr_4 + 16));
#if __CUDA_ARCH__ == 1030
                        if ((my_start_1 + pub_tile) * 128 + tok_c_4 >= kv_end_1 || empty_1 != 0) {
                            k_words_4[0] = 0;
                            k_words_4[1] = 0;
                            k_words_4[2] = 0;
                            k_words_4[3] = 0;
                            k_words_4[4] = 0;
                            k_words_4[5] = 0;
                            k_words_4[6] = 0;
                            k_words_4[7] = 0;
                        }
#else
#endif
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 352), "r"(k_words_4[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 356), "r"((k_words_4 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 360), "r"((k_words_4 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 364), "r"((k_words_4 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 368), "r"((k_words_4 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 372), "r"((k_words_4 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 376), "r"((k_words_4 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 380), "r"((k_words_4 + 7)[0]));
                        int tok_c_5 = 64 + quadrant_c / 2 * 32 + lane;
                        unsigned int ks_addr_5 = smem_ks_addr + (unsigned int)(raw_c * 32768 + tok_c_5 * 32);
                        unsigned int k_words_5[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[(0) + 3]))
                            : "r"(ks_addr_5));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_5[(4) + 3]))
                            : "r"(ks_addr_5 + 16));
#if __CUDA_ARCH__ == 1030
                        if ((my_start_1 + pub_tile) * 128 + tok_c_5 >= kv_end_1 || empty_1 != 0) {
                            k_words_5[0] = 0;
                            k_words_5[1] = 0;
                            k_words_5[2] = 0;
                            k_words_5[3] = 0;
                            k_words_5[4] = 0;
                            k_words_5[5] = 0;
                            k_words_5[6] = 0;
                            k_words_5[7] = 0;
                        }
#else
#endif
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 384), "r"(k_words_5[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 388), "r"((k_words_5 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 392), "r"((k_words_5 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 396), "r"((k_words_5 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 400), "r"((k_words_5 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 404), "r"((k_words_5 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 408), "r"((k_words_5 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)lane_base_c + 412), "r"((k_words_5 + 7)[0]));
                        mbarrier_arrive(raw_empty_addr + (raw_c) * 8);
#if __CUDA_ARCH__ == 1030
                        raw_c += 1;
                        if (raw_c == 3) { raw_c = 0; _phase_raw_full ^= 1; }
#else
#endif
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(sf_full_addr), "r"(0) : "memory");
                        _phase_sf_empty_0 ^= 1;
#if __CUDA_ARCH__ == 1030
#else
                        if (shift_warp_c != 0) {
                            int shift_pos_4 = (my_start_1 + pub_tile) * 128 + tok_c_4;
                            int shift_masked_4 = shift_pos_4 >= kv_end_1 || empty_1 != 0;
                            unsigned int shift_live_4 = ((shift_masked_4 != 0) ? 0 : 1);
                            unsigned int above_1_2 = 0;
                            unsigned int above_2_2 = 0;
                            unsigned int above_3_2 = 0;
                            above_1_2 = above_1_2 | k_words_4[0] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[0] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[0] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[1] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[1] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[1] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[2] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[2] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[2] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[3] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[3] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[3] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[4] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[4] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[4] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[5] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[5] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[5] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[6] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[6] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[6] + 101058054;
                            above_1_2 = above_1_2 | k_words_4[7] + 370546198;
                            above_2_2 = above_2_2 | k_words_4[7] + 235802126;
                            above_3_2 = above_3_2 | k_words_4[7] + 101058054;
                            unsigned int _min_14 = ((above_1_2 & 2155905152u) < (shift_live_4) ? (above_1_2 & 2155905152u) : (shift_live_4));
                            unsigned int _min_15 = ((above_2_2 & 2155905152u) < (shift_live_4) ? (above_2_2 & 2155905152u) : (shift_live_4));
                            unsigned int _min_16 = ((above_3_2 & 2155905152u) < (shift_live_4) ? (above_3_2 & 2155905152u) : (shift_live_4));
                            unsigned int shift_2 = _min_14 + _min_15 + _min_16;
                            unsigned int shift_mul_4 = shift_live_4 * 1006648320 - shift_2 * 67109888;
                            unsigned int shift_off_4 = (unsigned int)(raw_c * 512 + tok_c_4 * 4);
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + shift_off_4), "r"(shift_mul_4) : "memory");
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_fac_addr + shift_off_4), "r"(124 + shift_2 << 23) : "memory");
                            int shift_pos_5 = (my_start_1 + pub_tile) * 128 + tok_c_5;
                            int shift_masked_5 = shift_pos_5 >= kv_end_1 || empty_1 != 0;
                            unsigned int shift_live_5 = ((shift_masked_5 != 0) ? 0 : 1);
                            unsigned int above_1_2_1 = 0;
                            unsigned int above_2_2_1 = 0;
                            unsigned int above_3_2_1 = 0;
                            above_1_2_1 = above_1_2_1 | k_words_5[0] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[0] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[0] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[1] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[1] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[1] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[2] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[2] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[2] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[3] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[3] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[3] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[4] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[4] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[4] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[5] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[5] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[5] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[6] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[6] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[6] + 101058054;
                            above_1_2_1 = above_1_2_1 | k_words_5[7] + 370546198;
                            above_2_2_1 = above_2_2_1 | k_words_5[7] + 235802126;
                            above_3_2_1 = above_3_2_1 | k_words_5[7] + 101058054;
                            unsigned int _min_17 = ((above_1_2_1 & 2155905152u) < (shift_live_5) ? (above_1_2_1 & 2155905152u) : (shift_live_5));
                            unsigned int _min_18 = ((above_2_2_1 & 2155905152u) < (shift_live_5) ? (above_2_2_1 & 2155905152u) : (shift_live_5));
                            unsigned int _min_19 = ((above_3_2_1 & 2155905152u) < (shift_live_5) ? (above_3_2_1 & 2155905152u) : (shift_live_5));
                            unsigned int shift_2_1 = _min_17 + _min_18 + _min_19;
                            unsigned int shift_mul_5 = shift_live_5 * 1006648320 - shift_2_1 * 67109888;
                            unsigned int shift_off_5 = (unsigned int)(raw_c * 512 + tok_c_5 * 4);
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + shift_off_5), "r"(shift_mul_5) : "memory");
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_fac_addr + shift_off_5), "r"(124 + shift_2_1 << 23) : "memory");
                        }
                        mbarrier_arrive(shift_full_addr + (raw_c) * 8);
                        raw_c += 1;
                        if (raw_c == 3) { raw_c = 0; _phase_raw_full ^= 1; }
#endif
                        pub_tile = pub_tile + 1;
                    }
                }
                mbarrier_arrive(o_full_addr);
                mbarrier_wait(pv_done_addr, _phase_pv_done_0);
            }
            mbarrier_wait(p_empty_addr + (p_c) * 8, _phase_p_empty_1);
            p_c += 1;
            if (p_c == 2) { p_c = 0; _phase_p_empty_1 ^= 1; }
            mbarrier_wait(p_empty_addr + (p_c) * 8, _phase_p_empty_1);
            p_c += 1;
            if (p_c == 2) { p_c = 0; _phase_p_empty_1 ^= 1; }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 13) {
        { // transform_main
            int raw_t = 0;
            int v_t = 0;
            int role_tid = (warp - 8) * 32 + lane;
            int rank_t = cta_rank;
#if __CUDA_ARCH__ == 1030
            int _uniform_w = make_warp_uniform(warp % 4);
            int shift_warp = _uniform_w < 2;
            int shift_tok = (warp - 8) / 4 * 64 + warp % 4 * 32 + lane;
            unsigned int shift_row_rel = (unsigned int)(24576 + shift_tok * 32);
            unsigned int shift_mul_rel = (unsigned int)(shift_tok * 4);
            unsigned int shift_fac_rel = (unsigned int)(shift_tok * 2);
#else
#endif
            int group = role_tid / 8;
            int half = group % 2;
            int token = group / 2 * 2 + role_tid % 8 / 4;
            int chunk = role_tid % 4;
            int atom = half * 2 + rank_t;
            int row = token % 32;
            int other = token % 64 / 32 ^ rank_t;
            int region = (1 - other) * (atom * 4096) + other * (16384 + half * 4096);
            int src_rel = region + token / 64 * 2048 + row * 64 + (chunk ^ row >> 1 & 3) * 16;
            int scale_rel = 24576 + token * 32 + atom * 8 + chunk * 2;
            int v_off = token * 128 + chunk * 32;
            int v_rel = half * 16384 + (v_off ^ (v_off >> 7 & 7) << 4);
#if __CUDA_ARCH__ == 1030
            int mul_rel = token * 4;
#else
            int mul_rel = (role_tid / 16 * 2 + role_tid % 8 / 4) * 4;
#endif
            int group_0 = (role_tid + 192) / 8;
            int half_0 = group_0 % 2;
            int token_0 = group_0 / 2 * 2 + (role_tid + 192) % 8 / 4;
            int chunk_0 = (role_tid + 192) % 4;
            int atom_0 = half_0 * 2 + rank_t;
            int row_0 = token_0 % 32;
            int other_0 = token_0 % 64 / 32 ^ rank_t;
            int region_0 = (1 - other_0) * (atom_0 * 4096) + other_0 * (16384 + half_0 * 4096);
            int src_rel_0 = region_0 + token_0 / 64 * 2048 + row_0 * 64 + (chunk_0 ^ row_0 >> 1 & 3) * 16;
            int scale_rel_0 = 24576 + token_0 * 32 + atom_0 * 8 + chunk_0 * 2;
            int v_off_0 = token_0 * 128 + chunk_0 * 32;
            int v_rel_0 = half_0 * 16384 + (v_off_0 ^ (v_off_0 >> 7 & 7) << 4);
#if __CUDA_ARCH__ == 1030
            int mul_rel_0 = token_0 * 4;
#else
            int mul_rel_0 = ((role_tid + 192 - 192) / 16 * 2 + (role_tid + 192 - 192) % 8 / 4) * 4 + 96;
#endif
            int group_1 = (role_tid + 384) / 8;
            int half_1 = group_1 % 2;
            int token_1 = group_1 / 2 * 2 + (role_tid + 384) % 8 / 4;
            int chunk_1 = (role_tid + 384) % 4;
            int atom_1 = half_1 * 2 + rank_t;
            int row_1 = token_1 % 32;
            int other_1 = token_1 % 64 / 32 ^ rank_t;
            int region_1 = (1 - other_1) * (atom_1 * 4096) + other_1 * (16384 + half_1 * 4096);
            int src_rel_1 = region_1 + token_1 / 64 * 2048 + row_1 * 64 + (chunk_1 ^ row_1 >> 1 & 3) * 16;
            int scale_rel_1 = 24576 + token_1 * 32 + atom_1 * 8 + chunk_1 * 2;
            int v_off_1 = token_1 * 128 + chunk_1 * 32;
            int v_rel_1 = half_1 * 16384 + (v_off_1 ^ (v_off_1 >> 7 & 7) << 4);
#if __CUDA_ARCH__ == 1030
            int mul_rel_1 = token_1 * 4;
#else
            int mul_rel_1 = ((role_tid + 384 - 384) / 16 * 2 + (role_tid + 384 - 384) % 8 / 4) * 4 + 192;
#endif
            int group_2 = (role_tid + 576) / 8;
            int half_2 = group_2 % 2;
            int token_2 = group_2 / 2 * 2 + (role_tid + 576) % 8 / 4;
            int chunk_2 = (role_tid + 576) % 4;
            int atom_2 = half_2 * 2 + rank_t;
            int row_2 = token_2 % 32;
            int other_2 = token_2 % 64 / 32 ^ rank_t;
            int region_2 = (1 - other_2) * (atom_2 * 4096) + other_2 * (16384 + half_2 * 4096);
            int src_rel_2 = region_2 + token_2 / 64 * 2048 + row_2 * 64 + (chunk_2 ^ row_2 >> 1 & 3) * 16;
            int scale_rel_2 = 24576 + token_2 * 32 + atom_2 * 8 + chunk_2 * 2;
            int v_off_2 = token_2 * 128 + chunk_2 * 32;
            int v_rel_2 = half_2 * 16384 + (v_off_2 ^ (v_off_2 >> 7 & 7) << 4);
#if __CUDA_ARCH__ == 1030
            int mul_rel_2 = token_2 * 4;
#else
            int mul_rel_2 = ((role_tid + 576 - 576) / 16 * 2 + (role_tid + 576 - 576) % 8 / 4) * 4 + 288;
#endif
            int group_3 = (role_tid + 768) / 8;
            int half_3 = group_3 % 2;
            int token_3 = group_3 / 2 * 2 + (role_tid + 768) % 8 / 4;
            int chunk_3 = (role_tid + 768) % 4;
            int atom_3 = half_3 * 2 + rank_t;
            int row_3 = token_3 % 32;
            int other_3 = token_3 % 64 / 32 ^ rank_t;
            int region_3 = (1 - other_3) * (atom_3 * 4096) + other_3 * (16384 + half_3 * 4096);
            int src_rel_3 = region_3 + token_3 / 64 * 2048 + row_3 * 64 + (chunk_3 ^ row_3 >> 1 & 3) * 16;
            int scale_rel_3 = 24576 + token_3 * 32 + atom_3 * 8 + chunk_3 * 2;
            int v_off_3 = token_3 * 128 + chunk_3 * 32;
            int v_rel_3 = half_3 * 16384 + (v_off_3 ^ (v_off_3 >> 7 & 7) << 4);
#if __CUDA_ARCH__ == 1030
            int mul_rel_3 = token_3 * 4;
#else
            int mul_rel_3 = ((role_tid + 768 - 768) / 16 * 2 + (role_tid + 768 - 768) % 8 / 4) * 4 + 384;
#endif
            int group_4 = (role_tid + 896) / 8;
            int half_4 = group_4 % 2;
            int token_4 = group_4 / 2 * 2 + (role_tid + 896) % 8 / 4;
            int chunk_4 = (role_tid + 896) % 4;
            int atom_4 = half_4 * 2 + rank_t;
            int row_4 = token_4 % 32;
            int other_4 = token_4 % 64 / 32 ^ rank_t;
            int region_4 = (1 - other_4) * (atom_4 * 4096) + other_4 * (16384 + half_4 * 4096);
            int src_rel_4 = region_4 + token_4 / 64 * 2048 + row_4 * 64 + (chunk_4 ^ row_4 >> 1 & 3) * 16;
            int scale_rel_4 = 24576 + token_4 * 32 + atom_4 * 8 + chunk_4 * 2;
            int v_off_4 = token_4 * 128 + chunk_4 * 32;
            int v_rel_4 = half_4 * 16384 + (v_off_4 ^ (v_off_4 >> 7 & 7) << 4);
#if __CUDA_ARCH__ == 1030
            int mul_rel_4 = token_4 * 4;
#else
            int mul_rel_4 = ((role_tid + 896 - 896) / 16 * 2 + (role_tid + 896 - 896) % 8 / 4) * 4 + 448;
#endif
            int cx_2 = blockIdx.x / 2;
            int n_clusters_2 = gridDim.x / 2;
#if __CUDA_ARCH__ == 1030
            int q_end_r_2 = q_end_k;
            int q_start_r_2 = q_start_k;
            int kv_len_r_2 = kv_len_k;
            int g_len_r_2 = g_len_k;
#else
            int q_end_r_2 = cum_seq_lens_q[blockIdx.z + 1];
            int q_start_r_2 = cum_seq_lens_q[blockIdx.z];
            int kv_len_r_2 = seq_lens[blockIdx.z];
            int g_len_r_2 = kv_len_global[blockIdx.z];
#endif
            int pieces_2 = ((q_end_r_2 - q_start_r_2) * num_heads + 127) / 128 * num_split;
            int left_2 = pieces_2 - cx_2;
            int n_pieces_raw_2 = (left_2 + n_clusters_2 - 1) / n_clusters_2;
            int n_pieces_2 = ((left_2 <= 0) ? 0 : n_pieces_raw_2);
            unsigned int _phase_epi_done_0 = 1;
            unsigned int _phase_raw_full_1 = 0;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (int pi_t = 0; pi_t < n_pieces_2; pi_t++) {
                int split_idx_2 = (cx_2 + pi_t * n_clusters_2) % num_split;
                int q_len_b_2 = q_end_r_2 - q_start_r_2;
                int rows_b_2 = q_len_b_2 * num_heads;
                int m_tile_2 = (rows_b_2 + 127) / 128 - 1 - (cx_2 + pi_t * n_clusters_2) / num_split;
                int row0_2 = m_tile_2 * 128;
                int rows_left_2 = rows_b_2 - row0_2;
                int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
                int rows_valid_2 = ((rows_pos_2 > 128) ? 128 : rows_pos_2);
                int row_base_2 = q_start_r_2 * num_heads + row0_2;
                int t_last_2 = (row0_2 + rows_valid_2 - 1) / num_heads;
                int num_2 = g_len_r_2 - q_len_b_2 + t_last_2 - cp_rank;
                int vis_raw_2 = num_2 / cp_world + 1;
                int vis_cap_2 = ((vis_raw_2 > kv_len_r_2) ? kv_len_r_2 : vis_raw_2);
                int vis_2 = ((num_2 < 0) ? 0 : vis_cap_2);
                int kv_end_2 = ((rows_valid_2 == 0) ? 0 : vis_2);
                int n_tiles_total_2 = (kv_end_2 + 127) / 128;
                int tiles_per_split_2 = (n_tiles_total_2 + num_split - 1) / num_split;
                int my_start_raw_2 = split_idx_2 * tiles_per_split_2;
                int my_end_raw_2 = my_start_raw_2 + tiles_per_split_2;
                int my_end_2 = ((my_end_raw_2 > n_tiles_total_2) ? n_tiles_total_2 : my_end_raw_2);
                int my_n_raw_2 = my_end_2 - my_start_raw_2;
                int empty_2 = ((my_n_raw_2 < 1) ? 1 : 0);
                int my_n_tiles_2 = ((my_n_raw_2 < 1) ? 1 : my_n_raw_2);
                int my_start_2 = ((my_n_raw_2 < 1) ? 0 : my_start_raw_2);
#if __CUDA_ARCH__ == 1030
                int shift_lim = ((empty_2 != 0) ? 0 : kv_end_2 - my_start_2 * 128 - shift_tok);
#else
#endif
                if (pi_t > 0) {
                    mbarrier_wait(epi_done_addr, _phase_epi_done_0);
                }
                _phase_epi_done_0 ^= 1;
                #pragma unroll 1
                for (int it_t = 0; it_t < my_n_tiles_2; it_t++) {
                    mbarrier_wait(raw_full_addr + (raw_t) * 8, _phase_raw_full_1);
                    unsigned int raw_base = smem_raw_addr + (unsigned int)(raw_t * 32768);
                    unsigned int packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                        : "r"(raw_base + (unsigned int)src_rel));
                    unsigned int scale_word[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(raw_base + (unsigned int)scale_rel));
                    unsigned int scale_pair = scale_word[0];
                    unsigned int packed_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                        : "r"(raw_base + (unsigned int)src_rel_0));
                    unsigned int scale_word_0[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_0[0])) : "r"(raw_base + (unsigned int)scale_rel_0));
                    unsigned int scale_pair_0 = scale_word_0[0];
                    unsigned int packed_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                        : "r"(raw_base + (unsigned int)src_rel_1));
                    unsigned int scale_word_1[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_1[0])) : "r"(raw_base + (unsigned int)scale_rel_1));
                    unsigned int scale_pair_1 = scale_word_1[0];
#if __CUDA_ARCH__ == 1030
                    if (shift_warp != 0) {
                        unsigned int shift_addr = raw_base + shift_row_rel;
                        unsigned int shift_words[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shift_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&shift_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shift_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shift_words[(0) + 3]))
                            : "r"(shift_addr));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shift_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&shift_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shift_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shift_words[(4) + 3]))
                            : "r"(shift_addr + 16));
                        int masked = it_t * 128 >= shift_lim;
                        unsigned int above_1 = 0;
                        unsigned int above_2 = 0;
                        unsigned int above_3 = 0;
                        above_1 = above_1 | shift_words[0] + 370546198;
                        above_2 = above_2 | shift_words[0] + 235802126;
                        above_3 = above_3 | shift_words[0] + 101058054;
                        above_1 = above_1 | shift_words[1] + 370546198;
                        above_2 = above_2 | shift_words[1] + 235802126;
                        above_3 = above_3 | shift_words[1] + 101058054;
                        above_1 = above_1 | shift_words[2] + 370546198;
                        above_2 = above_2 | shift_words[2] + 235802126;
                        above_3 = above_3 | shift_words[2] + 101058054;
                        above_1 = above_1 | shift_words[3] + 370546198;
                        above_2 = above_2 | shift_words[3] + 235802126;
                        above_3 = above_3 | shift_words[3] + 101058054;
                        above_1 = above_1 | shift_words[4] + 370546198;
                        above_2 = above_2 | shift_words[4] + 235802126;
                        above_3 = above_3 | shift_words[4] + 101058054;
                        above_1 = above_1 | shift_words[5] + 370546198;
                        above_2 = above_2 | shift_words[5] + 235802126;
                        above_3 = above_3 | shift_words[5] + 101058054;
                        above_1 = above_1 | shift_words[6] + 370546198;
                        above_2 = above_2 | shift_words[6] + 235802126;
                        above_3 = above_3 | shift_words[6] + 101058054;
                        above_1 = above_1 | shift_words[7] + 370546198;
                        above_2 = above_2 | shift_words[7] + 235802126;
                        above_3 = above_3 | shift_words[7] + 101058054;
                        unsigned int shift = (unsigned int)((above_1 & 2155905152u) != 0) + (unsigned int)((above_2 & 2155905152u) != 0) + (unsigned int)((above_3 & 2155905152u) != 0);
                        unsigned int shift_k = ((masked != 0) ? 0 : shift);
                        unsigned int shift_mul = ((masked != 0) ? 0 : (15 - shift_k) * 67109888);
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_shift_mul_addr + (unsigned int)(raw_t * 512) + shift_mul_rel), "r"(shift_mul) : "memory");
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(smem_shift_fac_addr + (unsigned int)(raw_t * 256) + shift_fac_rel), "h"((uint16_t)((uint16_t)(124 + shift_k << 7))) : "memory");
                    }
                    mbarrier_arrive(shift_full_addr + (raw_t) * 8);
#else
#endif
                    mbarrier_wait(v_empty_addr + (v_t) * 8, _phase_v_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    unsigned int mul_base = smem_shift_mul_addr + (unsigned int)(raw_t * 512);
                    unsigned int v_base = smem_v_addr + (unsigned int)(v_t * 32768);
                    mbarrier_wait(shift_full_addr + (raw_t) * 8, _phase_raw_full_1);
                    unsigned int mul_word[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&mul_word[0])) : "r"(mul_base + (unsigned int)mul_rel));
                    unsigned int multiplier = mul_word[0];
                    unsigned int mul_word_0[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&mul_word_0[0])) : "r"(mul_base + (unsigned int)mul_rel_0));
                    unsigned int multiplier_0 = mul_word_0[0];
                    unsigned int mul_word_1[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&mul_word_1[0])) : "r"(mul_base + (unsigned int)mul_rel_1));
                    unsigned int multiplier_1 = mul_word_1[0];
                    unsigned int mul_word_2[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&mul_word_2[0])) : "r"(mul_base + (unsigned int)mul_rel_2));
                    unsigned int multiplier_2 = mul_word_2[0];
                    unsigned int mul_word_3[1];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&mul_word_3[0])) : "r"(mul_base + (unsigned int)mul_rel_3));
                    unsigned int multiplier_3 = mul_word_3[0];
#if __CUDA_ARCH__ == 1030
                    unsigned int kept = ((multiplier != 0) ? scale_pair : 0);
#else
                    unsigned int kept = ((scale_pair) < (multiplier) ? (scale_pair) : (multiplier));
#endif
                    uint32_t _cvt_f16x2_e4m3x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_0) : "h"((uint16_t)(kept)));
                    uint32_t scaled;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(scaled) : "r"(_cvt_f16x2_e4m3x2_0), "r"(multiplier));
                    uint16_t _cvt_satfinite_e4m3x2_f16x2_1;
                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_f16x2_1) : "r"(scaled));
                    unsigned int shifted = _cvt_satfinite_e4m3x2_f16x2_1;
                    unsigned int s0 = shifted;
#if __CUDA_ARCH__ == 1030
                    unsigned int s1 = shifted >> 8;
#else
                    uint32_t s1;
                    asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(s1) : "r"(shifted), "r"(0));
#endif
                    unsigned int converted[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted[0] = qmul4_lower4(packed[0], s0);
                    converted[1] = qmul4_higher4(packed[0], s0);
                    converted[2] = qmul4_lower4(packed[1], s0);
                    converted[3] = qmul4_higher4(packed[1], s0);
                    converted[4] = qmul4_lower4(packed[2], s1);
                    converted[5] = qmul4_higher4(packed[2], s1);
                    converted[6] = qmul4_lower4(packed[3], s1);
                    converted[7] = qmul4_higher4(packed[3], s1);
                    #else
                    {
                        uint16_t _qmul4_pair_2_sb0 = (uint16_t)((s0) & 0xFFu);
                        uint16_t _qmul4_pair_2_s0 = (uint16_t)(_qmul4_pair_2_sb0 | (_qmul4_pair_2_sb0 << 8));
                        uint16_t _qmul4_pair_2_sb1 = (uint16_t)((s1) & 0xFFu);
                        uint16_t _qmul4_pair_2_s1 = (uint16_t)(_qmul4_pair_2_sb1 | (_qmul4_pair_2_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted[0]), "=&r"(converted[1]), "=&r"(converted[2]), "=&r"(converted[3]), "=&r"(converted[4]), "=&r"(converted[5]), "=&r"(converted[6]), "=&r"(converted[7])
                            : "h"(_qmul4_pair_2_s0), "h"(_qmul4_pair_2_s1), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)v_rel), "r"(converted[0]), "r"(converted[1]), "r"(converted[2]), "r"(converted[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)(v_rel ^ 16)), "r"(converted[4]), "r"(converted[5]), "r"(converted[6]), "r"(converted[7]) : "memory");
                    unsigned int packed_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                        : "r"(raw_base + (unsigned int)src_rel_2));
                    unsigned int scale_word_2[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_2[0])) : "r"(raw_base + (unsigned int)scale_rel_2));
                    unsigned int scale_pair_2 = scale_word_2[0];
#if __CUDA_ARCH__ == 1030
                    unsigned int kept_0 = ((multiplier_0 != 0) ? scale_pair_0 : 0);
#else
                    unsigned int kept_0 = ((scale_pair_0) < (multiplier_0) ? (scale_pair_0) : (multiplier_0));
#endif
                    uint32_t _cvt_f16x2_e4m3x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_3) : "h"((uint16_t)(kept_0)));
                    uint32_t scaled_0;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(scaled_0) : "r"(_cvt_f16x2_e4m3x2_3), "r"(multiplier_0));
                    uint16_t _cvt_satfinite_e4m3x2_f16x2_4;
                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_f16x2_4) : "r"(scaled_0));
                    unsigned int shifted_0 = _cvt_satfinite_e4m3x2_f16x2_4;
                    unsigned int s0_0 = shifted_0;
#if __CUDA_ARCH__ == 1030
                    unsigned int s1_0 = shifted_0 >> 8;
#else
                    uint32_t s1_0;
                    asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(s1_0) : "r"(shifted_0), "r"(0));
#endif
                    unsigned int converted_0[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_0[0] = qmul4_lower4(packed_0[0], s0_0);
                    converted_0[1] = qmul4_higher4(packed_0[0], s0_0);
                    converted_0[2] = qmul4_lower4(packed_0[1], s0_0);
                    converted_0[3] = qmul4_higher4(packed_0[1], s0_0);
                    converted_0[4] = qmul4_lower4(packed_0[2], s1_0);
                    converted_0[5] = qmul4_higher4(packed_0[2], s1_0);
                    converted_0[6] = qmul4_lower4(packed_0[3], s1_0);
                    converted_0[7] = qmul4_higher4(packed_0[3], s1_0);
                    #else
                    {
                        uint16_t _qmul4_pair_5_sb0 = (uint16_t)((s0_0) & 0xFFu);
                        uint16_t _qmul4_pair_5_s0 = (uint16_t)(_qmul4_pair_5_sb0 | (_qmul4_pair_5_sb0 << 8));
                        uint16_t _qmul4_pair_5_sb1 = (uint16_t)((s1_0) & 0xFFu);
                        uint16_t _qmul4_pair_5_s1 = (uint16_t)(_qmul4_pair_5_sb1 | (_qmul4_pair_5_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_0[0]), "=&r"(converted_0[1]), "=&r"(converted_0[2]), "=&r"(converted_0[3]), "=&r"(converted_0[4]), "=&r"(converted_0[5]), "=&r"(converted_0[6]), "=&r"(converted_0[7])
                            : "h"(_qmul4_pair_5_s0), "h"(_qmul4_pair_5_s1), "r"(packed_0[0]), "r"(packed_0[1]), "r"(packed_0[2]), "r"(packed_0[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)v_rel_0), "r"(converted_0[0]), "r"(converted_0[1]), "r"(converted_0[2]), "r"(converted_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)(v_rel_0 ^ 16)), "r"(converted_0[4]), "r"(converted_0[5]), "r"(converted_0[6]), "r"(converted_0[7]) : "memory");
                    unsigned int packed_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 3]))
                        : "r"(raw_base + (unsigned int)src_rel_3));
                    unsigned int scale_word_3[1];
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_3[0])) : "r"(raw_base + (unsigned int)scale_rel_3));
                    unsigned int scale_pair_3 = scale_word_3[0];
#if __CUDA_ARCH__ == 1030
                    unsigned int kept_1 = ((multiplier_1 != 0) ? scale_pair_1 : 0);
#else
                    unsigned int kept_1 = ((scale_pair_1) < (multiplier_1) ? (scale_pair_1) : (multiplier_1));
#endif
                    uint32_t _cvt_f16x2_e4m3x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_6) : "h"((uint16_t)(kept_1)));
                    uint32_t scaled_1;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(scaled_1) : "r"(_cvt_f16x2_e4m3x2_6), "r"(multiplier_1));
                    uint16_t _cvt_satfinite_e4m3x2_f16x2_7;
                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_f16x2_7) : "r"(scaled_1));
                    unsigned int shifted_1 = _cvt_satfinite_e4m3x2_f16x2_7;
                    unsigned int s0_1 = shifted_1;
#if __CUDA_ARCH__ == 1030
                    unsigned int s1_1 = shifted_1 >> 8;
#else
                    uint32_t s1_1;
                    asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(s1_1) : "r"(shifted_1), "r"(0));
#endif
                    unsigned int converted_1[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_1[0] = qmul4_lower4(packed_1[0], s0_1);
                    converted_1[1] = qmul4_higher4(packed_1[0], s0_1);
                    converted_1[2] = qmul4_lower4(packed_1[1], s0_1);
                    converted_1[3] = qmul4_higher4(packed_1[1], s0_1);
                    converted_1[4] = qmul4_lower4(packed_1[2], s1_1);
                    converted_1[5] = qmul4_higher4(packed_1[2], s1_1);
                    converted_1[6] = qmul4_lower4(packed_1[3], s1_1);
                    converted_1[7] = qmul4_higher4(packed_1[3], s1_1);
                    #else
                    {
                        uint16_t _qmul4_pair_8_sb0 = (uint16_t)((s0_1) & 0xFFu);
                        uint16_t _qmul4_pair_8_s0 = (uint16_t)(_qmul4_pair_8_sb0 | (_qmul4_pair_8_sb0 << 8));
                        uint16_t _qmul4_pair_8_sb1 = (uint16_t)((s1_1) & 0xFFu);
                        uint16_t _qmul4_pair_8_s1 = (uint16_t)(_qmul4_pair_8_sb1 | (_qmul4_pair_8_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_1[0]), "=&r"(converted_1[1]), "=&r"(converted_1[2]), "=&r"(converted_1[3]), "=&r"(converted_1[4]), "=&r"(converted_1[5]), "=&r"(converted_1[6]), "=&r"(converted_1[7])
                            : "h"(_qmul4_pair_8_s0), "h"(_qmul4_pair_8_s1), "r"(packed_1[0]), "r"(packed_1[1]), "r"(packed_1[2]), "r"(packed_1[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)v_rel_1), "r"(converted_1[0]), "r"(converted_1[1]), "r"(converted_1[2]), "r"(converted_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)(v_rel_1 ^ 16)), "r"(converted_1[4]), "r"(converted_1[5]), "r"(converted_1[6]), "r"(converted_1[7]) : "memory");
#if __CUDA_ARCH__ == 1030
                    unsigned int kept_2 = ((multiplier_2 != 0) ? scale_pair_2 : 0);
#else
                    unsigned int kept_2 = ((scale_pair_2) < (multiplier_2) ? (scale_pair_2) : (multiplier_2));
#endif
                    uint32_t _cvt_f16x2_e4m3x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_9) : "h"((uint16_t)(kept_2)));
                    uint32_t scaled_2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(scaled_2) : "r"(_cvt_f16x2_e4m3x2_9), "r"(multiplier_2));
                    uint16_t _cvt_satfinite_e4m3x2_f16x2_10;
                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_f16x2_10) : "r"(scaled_2));
                    unsigned int shifted_2 = _cvt_satfinite_e4m3x2_f16x2_10;
                    unsigned int s0_2 = shifted_2;
#if __CUDA_ARCH__ == 1030
                    unsigned int s1_2 = shifted_2 >> 8;
#else
                    uint32_t s1_2;
                    asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(s1_2) : "r"(shifted_2), "r"(0));
#endif
                    unsigned int converted_2[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_2[0] = qmul4_lower4(packed_2[0], s0_2);
                    converted_2[1] = qmul4_higher4(packed_2[0], s0_2);
                    converted_2[2] = qmul4_lower4(packed_2[1], s0_2);
                    converted_2[3] = qmul4_higher4(packed_2[1], s0_2);
                    converted_2[4] = qmul4_lower4(packed_2[2], s1_2);
                    converted_2[5] = qmul4_higher4(packed_2[2], s1_2);
                    converted_2[6] = qmul4_lower4(packed_2[3], s1_2);
                    converted_2[7] = qmul4_higher4(packed_2[3], s1_2);
                    #else
                    {
                        uint16_t _qmul4_pair_11_sb0 = (uint16_t)((s0_2) & 0xFFu);
                        uint16_t _qmul4_pair_11_s0 = (uint16_t)(_qmul4_pair_11_sb0 | (_qmul4_pair_11_sb0 << 8));
                        uint16_t _qmul4_pair_11_sb1 = (uint16_t)((s1_2) & 0xFFu);
                        uint16_t _qmul4_pair_11_s1 = (uint16_t)(_qmul4_pair_11_sb1 | (_qmul4_pair_11_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_2[0]), "=&r"(converted_2[1]), "=&r"(converted_2[2]), "=&r"(converted_2[3]), "=&r"(converted_2[4]), "=&r"(converted_2[5]), "=&r"(converted_2[6]), "=&r"(converted_2[7])
                            : "h"(_qmul4_pair_11_s0), "h"(_qmul4_pair_11_s1), "r"(packed_2[0]), "r"(packed_2[1]), "r"(packed_2[2]), "r"(packed_2[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)v_rel_2), "r"(converted_2[0]), "r"(converted_2[1]), "r"(converted_2[2]), "r"(converted_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)(v_rel_2 ^ 16)), "r"(converted_2[4]), "r"(converted_2[5]), "r"(converted_2[6]), "r"(converted_2[7]) : "memory");
                    if (role_tid >= 64 && role_tid < 128) {
                        unsigned int packed_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                            : "r"(raw_base + (unsigned int)src_rel_4));
                        unsigned int scale_word_4[1];
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word_4[0])) : "r"(raw_base + (unsigned int)scale_rel_4));
                        unsigned int scale_pair_4 = scale_word_4[0];
                        unsigned int mul_word_4[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&mul_word_4[0])) : "r"(mul_base + (unsigned int)mul_rel_4));
                        unsigned int multiplier_4 = mul_word_4[0];
#if __CUDA_ARCH__ == 1030
                        unsigned int kept_3 = ((multiplier_4 != 0) ? scale_pair_4 : 0);
#else
                        unsigned int kept_3 = ((scale_pair_4) < (multiplier_4) ? (scale_pair_4) : (multiplier_4));
#endif
                        uint32_t _cvt_f16x2_e4m3x2_12;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_12) : "h"((uint16_t)(kept_3)));
                        uint32_t scaled_3;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(scaled_3) : "r"(_cvt_f16x2_e4m3x2_12), "r"(multiplier_4));
                        uint16_t _cvt_satfinite_e4m3x2_f16x2_13;
                        asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_f16x2_13) : "r"(scaled_3));
                        unsigned int shifted_3 = _cvt_satfinite_e4m3x2_f16x2_13;
                        unsigned int s0_3 = shifted_3;
#if __CUDA_ARCH__ == 1030
                        unsigned int s1_3 = shifted_3 >> 8;
#else
                        uint32_t s1_3;
                        asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(s1_3) : "r"(shifted_3), "r"(0));
#endif
                        unsigned int converted_3[8];
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                        converted_3[0] = qmul4_lower4(packed_4[0], s0_3);
                        converted_3[1] = qmul4_higher4(packed_4[0], s0_3);
                        converted_3[2] = qmul4_lower4(packed_4[1], s0_3);
                        converted_3[3] = qmul4_higher4(packed_4[1], s0_3);
                        converted_3[4] = qmul4_lower4(packed_4[2], s1_3);
                        converted_3[5] = qmul4_higher4(packed_4[2], s1_3);
                        converted_3[6] = qmul4_lower4(packed_4[3], s1_3);
                        converted_3[7] = qmul4_higher4(packed_4[3], s1_3);
                        #else
                        {
                            uint16_t _qmul4_pair_14_sb0 = (uint16_t)((s0_3) & 0xFFu);
                            uint16_t _qmul4_pair_14_s0 = (uint16_t)(_qmul4_pair_14_sb0 | (_qmul4_pair_14_sb0 << 8));
                            uint16_t _qmul4_pair_14_sb1 = (uint16_t)((s1_3) & 0xFFu);
                            uint16_t _qmul4_pair_14_s1 = (uint16_t)(_qmul4_pair_14_sb1 | (_qmul4_pair_14_sb1 << 8));
                            asm volatile("{\n"
                                ".reg .b8 b0, b1, b2, b3;\n"
                                ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                                ".reg .b16 e0, e1, e2, e3;\n"
                                "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                                "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                                "mov.b32 {b0, b1, b2, b3}, %10;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %0, {e0, e1};\n"
                                "mov.b32 %1, {e2, e3};\n"
                                "mov.b32 {b0, b1, b2, b3}, %11;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %2, {e0, e1};\n"
                                "mov.b32 %3, {e2, e3};\n"
                                "mov.b32 {b0, b1, b2, b3}, %12;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %4, {e0, e1};\n"
                                "mov.b32 %5, {e2, e3};\n"
                                "mov.b32 {b0, b1, b2, b3}, %13;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %6, {e0, e1};\n"
                                "mov.b32 %7, {e2, e3};\n"
                                "}\n"
                                : "=&r"(converted_3[0]), "=&r"(converted_3[1]), "=&r"(converted_3[2]), "=&r"(converted_3[3]), "=&r"(converted_3[4]), "=&r"(converted_3[5]), "=&r"(converted_3[6]), "=&r"(converted_3[7])
                                : "h"(_qmul4_pair_14_s0), "h"(_qmul4_pair_14_s1), "r"(packed_4[0]), "r"(packed_4[1]), "r"(packed_4[2]), "r"(packed_4[3]));
                        }
                        #endif
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)v_rel_4), "r"(converted_3[0]), "r"(converted_3[1]), "r"(converted_3[2]), "r"(converted_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)(v_rel_4 ^ 16)), "r"(converted_3[4]), "r"(converted_3[5]), "r"(converted_3[6]), "r"(converted_3[7]) : "memory");
                    }
#if __CUDA_ARCH__ == 1030
                    unsigned int kept_4 = ((multiplier_3 != 0) ? scale_pair_3 : 0);
#else
                    unsigned int kept_4 = ((scale_pair_3) < (multiplier_3) ? (scale_pair_3) : (multiplier_3));
#endif
                    uint32_t _cvt_f16x2_e4m3x2_15;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_cvt_f16x2_e4m3x2_15) : "h"((uint16_t)(kept_4)));
                    uint32_t scaled_4;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(scaled_4) : "r"(_cvt_f16x2_e4m3x2_15), "r"(multiplier_3));
                    uint16_t _cvt_satfinite_e4m3x2_f16x2_16;
                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_f16x2_16) : "r"(scaled_4));
                    unsigned int shifted_4 = _cvt_satfinite_e4m3x2_f16x2_16;
                    unsigned int s0_4 = shifted_4;
#if __CUDA_ARCH__ == 1030
                    unsigned int s1_4 = shifted_4 >> 8;
#else
                    uint32_t s1_4;
                    asm("prmt.b32 %0, %1, %2, 0x1111;" : "=r"(s1_4) : "r"(shifted_4), "r"(0));
#endif
                    unsigned int converted_4[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_4[0] = qmul4_lower4(packed_3[0], s0_4);
                    converted_4[1] = qmul4_higher4(packed_3[0], s0_4);
                    converted_4[2] = qmul4_lower4(packed_3[1], s0_4);
                    converted_4[3] = qmul4_higher4(packed_3[1], s0_4);
                    converted_4[4] = qmul4_lower4(packed_3[2], s1_4);
                    converted_4[5] = qmul4_higher4(packed_3[2], s1_4);
                    converted_4[6] = qmul4_lower4(packed_3[3], s1_4);
                    converted_4[7] = qmul4_higher4(packed_3[3], s1_4);
                    #else
                    {
                        uint16_t _qmul4_pair_17_sb0 = (uint16_t)((s0_4) & 0xFFu);
                        uint16_t _qmul4_pair_17_s0 = (uint16_t)(_qmul4_pair_17_sb0 | (_qmul4_pair_17_sb0 << 8));
                        uint16_t _qmul4_pair_17_sb1 = (uint16_t)((s1_4) & 0xFFu);
                        uint16_t _qmul4_pair_17_s1 = (uint16_t)(_qmul4_pair_17_sb1 | (_qmul4_pair_17_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_4[0]), "=&r"(converted_4[1]), "=&r"(converted_4[2]), "=&r"(converted_4[3]), "=&r"(converted_4[4]), "=&r"(converted_4[5]), "=&r"(converted_4[6]), "=&r"(converted_4[7])
                            : "h"(_qmul4_pair_17_s0), "h"(_qmul4_pair_17_s1), "r"(packed_3[0]), "r"(packed_3[1]), "r"(packed_3[2]), "r"(packed_3[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)v_rel_3), "r"(converted_4[0]), "r"(converted_4[1]), "r"(converted_4[2]), "r"(converted_4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_base + (unsigned int)(v_rel_3 ^ 16)), "r"(converted_4[4]), "r"(converted_4[5]), "r"(converted_4[6]), "r"(converted_4[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(v_full_addr + (v_t) * 8), "r"(0) : "memory");
                    mbarrier_arrive(raw_empty_addr + (raw_t) * 8);
                    raw_t += 1;
                    if (raw_t == 3) { raw_t = 0; _phase_raw_full_1 ^= 1; }
                    v_t += 1;
                    if (v_t == 2) { v_t = 0; _phase_v_empty ^= 1; }
                }
            }
            mbarrier_wait(v_empty_addr + (v_t) * 8, _phase_v_empty);
            v_t += 1;
            if (v_t == 2) { v_t = 0; _phase_v_empty ^= 1; }
            mbarrier_wait(v_empty_addr + (v_t) * 8, _phase_v_empty);
            v_t += 1;
            if (v_t == 2) { v_t = 0; _phase_v_empty ^= 1; }
        }
    }
    // ---- Role: load ----
    if (warp == 14) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            int rank_l = cta_rank;
            int raw_l = 0;
            int cx_3 = blockIdx.x / 2;
            int n_clusters_3 = gridDim.x / 2;
#if __CUDA_ARCH__ == 1030
            int q_end_r_3 = q_end_k;
            int q_start_r_3 = q_start_k;
            int kv_len_r_3 = kv_len_k;
            int g_len_r_3 = g_len_k;
#else
            int q_end_r_3 = cum_seq_lens_q[blockIdx.z + 1];
            int q_start_r_3 = cum_seq_lens_q[blockIdx.z];
            int kv_len_r_3 = seq_lens[blockIdx.z];
            int g_len_r_3 = kv_len_global[blockIdx.z];
#endif
            int pieces_3 = ((q_end_r_3 - q_start_r_3) * num_heads + 127) / 128 * num_split;
            int left_3 = pieces_3 - cx_3;
            int n_pieces_raw_3 = (left_3 + n_clusters_3 - 1) / n_clusters_3;
            int n_pieces_3 = ((left_3 <= 0) ? 0 : n_pieces_raw_3);
            int pt_base_l = blockIdx.z * max_pages_per_seq;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_raw_empty = 1;
            #pragma unroll 1
            for (int pi_l = 0; pi_l < n_pieces_3; pi_l++) {
                int split_idx_3 = (cx_3 + pi_l * n_clusters_3) % num_split;
                int q_len_b_3 = q_end_r_3 - q_start_r_3;
                int rows_b_3 = q_len_b_3 * num_heads;
                int m_tile_3 = (rows_b_3 + 127) / 128 - 1 - (cx_3 + pi_l * n_clusters_3) / num_split;
                int row0_3 = m_tile_3 * 128;
                int rows_left_3 = rows_b_3 - row0_3;
                int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
                int rows_valid_3 = ((rows_pos_3 > 128) ? 128 : rows_pos_3);
                int row_base_3 = q_start_r_3 * num_heads + row0_3;
                int t_last_3 = (row0_3 + rows_valid_3 - 1) / num_heads;
                int num_3 = g_len_r_3 - q_len_b_3 + t_last_3 - cp_rank;
                int vis_raw_3 = num_3 / cp_world + 1;
                int vis_cap_3 = ((vis_raw_3 > kv_len_r_3) ? kv_len_r_3 : vis_raw_3);
                int vis_3 = ((num_3 < 0) ? 0 : vis_cap_3);
                int kv_end_3 = ((rows_valid_3 == 0) ? 0 : vis_3);
                int n_tiles_total_3 = (kv_end_3 + 127) / 128;
                int tiles_per_split_3 = (n_tiles_total_3 + num_split - 1) / num_split;
                int my_start_raw_3 = split_idx_3 * tiles_per_split_3;
                int my_end_raw_3 = my_start_raw_3 + tiles_per_split_3;
                int my_end_3 = ((my_end_raw_3 > n_tiles_total_3) ? n_tiles_total_3 : my_end_raw_3);
                int my_n_raw_3 = my_end_3 - my_start_raw_3;
                int empty_3 = ((my_n_raw_3 < 1) ? 1 : 0);
                int my_n_tiles_3 = ((my_n_raw_3 < 1) ? 1 : my_n_raw_3);
                int my_start_3 = ((my_n_raw_3 < 1) ? 0 : my_start_raw_3);
                int q_row0 = row_base_3 + rank_l * 64;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 22528);
                    tma_2d_gmem2smem(smem_q_addr, (&tmap_qn), 0, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q_addr + 4096, (&tmap_qn), 64, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q_addr + 8192, (&tmap_qn), 128, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q_addr + 12288, (&tmap_qn), 192, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_qs_addr, (&tmap_qs), 0, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 0, q_row0, q_full_addr);
                }
                _phase_q_empty_0 ^= 1;
                #pragma unroll 1
                for (int it_l = 0; it_l < my_n_tiles_3; it_l++) {
                    mbarrier_wait(raw_empty_addr + (raw_l) * 8, _phase_raw_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(raw_full_addr + (raw_l) * 8, 32768);
                        if (rank_l == 0) {
                            int tok_raw = (my_start_3 + it_l) * 128;
                            int tok = ((tok_raw >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw);
                            int pidx = tok >> page_shift;
                            int off = tok - (pidx << page_shift);
                            int g_raw = page_table[pt_base_l + pidx];
                            int g = ((g_raw < 0) ? 0 : g_raw);
                            int tok_raw_0 = (my_start_3 + it_l) * 128 + 32;
                            int tok_0 = ((tok_raw_0 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_0);
                            int pidx_0 = tok_0 >> page_shift;
                            int off_0 = tok_0 - (pidx_0 << page_shift);
                            int g_raw_0 = page_table[pt_base_l + pidx_0];
                            int g_0 = ((g_raw_0 < 0) ? 0 : g_raw_0);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768), (&tmap_k), 0, off, g, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 4096), (&tmap_k), 64, off, g, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 8192), (&tmap_k), 128, off, g, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 12288), (&tmap_k), 192, off, g, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768), (&tmap_k), 0, off_0, g_0, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768 + 4096), (&tmap_k), 128, off_0, g_0, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768), (&tmap_ks), 0, off, g, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768 + 1024), (&tmap_ks), 0, off_0, g_0, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(raw_l * 32768), (&tmap_kr), 0, off, g, raw_full_addr + (raw_l) * 8);
                        }
                        if (rank_l == 1) {
                            int tok_raw_1 = (my_start_3 + it_l) * 128 + 32;
                            int tok_1 = ((tok_raw_1 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_1);
                            int pidx_1 = tok_1 >> page_shift;
                            int off_1 = tok_1 - (pidx_1 << page_shift);
                            int g_raw_1 = page_table[pt_base_l + pidx_1];
                            int g_1 = ((g_raw_1 < 0) ? 0 : g_raw_1);
                            int tok_raw_1_1 = (my_start_3 + it_l) * 128;
                            int tok_1_1 = ((tok_raw_1_1 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_1_1);
                            int pidx_1_1 = tok_1_1 >> page_shift;
                            int off_1_1 = tok_1_1 - (pidx_1_1 << page_shift);
                            int g_raw_1_1 = page_table[pt_base_l + pidx_1_1];
                            int g_1_1 = ((g_raw_1_1 < 0) ? 0 : g_raw_1_1);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768), (&tmap_k), 0, off_1, g_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 4096), (&tmap_k), 64, off_1, g_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 8192), (&tmap_k), 128, off_1, g_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 12288), (&tmap_k), 192, off_1, g_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768), (&tmap_k), 64, off_1_1, g_1_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768 + 4096), (&tmap_k), 192, off_1_1, g_1_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768 + 1024), (&tmap_ks), 0, off_1, g_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768), (&tmap_ks), 0, off_1_1, g_1_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(raw_l * 32768), (&tmap_kr), 0, off_1, g_1, raw_full_addr + (raw_l) * 8);
                        }
                        if (rank_l == 0) {
                            int tok_raw_2 = (my_start_3 + it_l) * 128 + 64;
                            int tok_2 = ((tok_raw_2 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_2);
                            int pidx_2 = tok_2 >> page_shift;
                            int off_2 = tok_2 - (pidx_2 << page_shift);
                            int g_raw_2 = page_table[pt_base_l + pidx_2];
                            int g_2 = ((g_raw_2 < 0) ? 0 : g_raw_2);
                            int tok_raw_2_1 = (my_start_3 + it_l) * 128 + 96;
                            int tok_2_1 = ((tok_raw_2_1 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_2_1);
                            int pidx_2_1 = tok_2_1 >> page_shift;
                            int off_2_1 = tok_2_1 - (pidx_2_1 << page_shift);
                            int g_raw_2_1 = page_table[pt_base_l + pidx_2_1];
                            int g_2_1 = ((g_raw_2_1 < 0) ? 0 : g_raw_2_1);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_k), 0, off_2, g_2, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 4096 + 2048), (&tmap_k), 64, off_2, g_2, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 8192 + 2048), (&tmap_k), 128, off_2, g_2, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 12288 + 2048), (&tmap_k), 192, off_2, g_2, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_k), 0, off_2_1, g_2_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768 + 4096 + 2048), (&tmap_k), 128, off_2_1, g_2_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_ks), 0, off_2, g_2, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768 + 3072), (&tmap_ks), 0, off_2_1, g_2_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_kr), 0, off_2, g_2, raw_full_addr + (raw_l) * 8);
                        }
                        if (rank_l == 1) {
                            int tok_raw_3 = (my_start_3 + it_l) * 128 + 96;
                            int tok_3 = ((tok_raw_3 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_3);
                            int pidx_3 = tok_3 >> page_shift;
                            int off_3 = tok_3 - (pidx_3 << page_shift);
                            int g_raw_3 = page_table[pt_base_l + pidx_3];
                            int g_3 = ((g_raw_3 < 0) ? 0 : g_raw_3);
                            int tok_raw_3_1 = (my_start_3 + it_l) * 128 + 64;
                            int tok_3_1 = ((tok_raw_3_1 >= kv_end_3 || empty_3 != 0) ? 0 : tok_raw_3_1);
                            int pidx_3_1 = tok_3_1 >> page_shift;
                            int off_3_1 = tok_3_1 - (pidx_3_1 << page_shift);
                            int g_raw_3_1 = page_table[pt_base_l + pidx_3_1];
                            int g_3_1 = ((g_raw_3_1 < 0) ? 0 : g_raw_3_1);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_k), 0, off_3, g_3, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 4096 + 2048), (&tmap_k), 64, off_3, g_3, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 8192 + 2048), (&tmap_k), 128, off_3, g_3, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kq_addr + (unsigned int)(raw_l * 32768 + 12288 + 2048), (&tmap_k), 192, off_3, g_3, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_k), 64, off_3_1, g_3_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ko_addr + (unsigned int)(raw_l * 32768 + 4096 + 2048), (&tmap_k), 192, off_3_1, g_3_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768 + 3072), (&tmap_ks), 0, off_3, g_3, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_ks_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_ks), 0, off_3_1, g_3_1, raw_full_addr + (raw_l) * 8);
                            tma_3d_gmem2smem(smem_kr_addr + (unsigned int)(raw_l * 32768 + 2048), (&tmap_kr), 0, off_3, g_3, raw_full_addr + (raw_l) * 8);
                        }
                    }
                    raw_l += 1;
                    if (raw_l == 3) { raw_l = 0; _phase_raw_empty ^= 1; }
                }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
        }
    }
    // ---- Role: mma ----
    if (warp == 15) {
        { // mma_main
            int raw_m = 0;
            int v_m = 0;
            int p_m = 0;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_sfa_full_0 = 0;
            unsigned int _phase_raw_full_2 = 0;
            unsigned int _phase_sf_full_0 = 0;
            unsigned int _phase_s_empty_0 = 1;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_p_full = 0;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    int cx_4 = blockIdx.x / 2;
                    int n_clusters_4 = gridDim.x / 2;
#if __CUDA_ARCH__ == 1030
                    int q_end_r_4 = q_end_k;
                    int q_start_r_4 = q_start_k;
                    int kv_len_r_4 = kv_len_k;
                    int g_len_r_4 = g_len_k;
#else
                    int q_end_r_4 = cum_seq_lens_q[blockIdx.z + 1];
                    int q_start_r_4 = cum_seq_lens_q[blockIdx.z];
                    int kv_len_r_4 = seq_lens[blockIdx.z];
                    int g_len_r_4 = kv_len_global[blockIdx.z];
#endif
                    int pieces_4 = ((q_end_r_4 - q_start_r_4) * num_heads + 127) / 128 * num_split;
                    int left_4 = pieces_4 - cx_4;
                    int n_pieces_raw_4 = (left_4 + n_clusters_4 - 1) / n_clusters_4;
                    int n_pieces_4 = ((left_4 <= 0) ? 0 : n_pieces_raw_4);
                    #pragma unroll 1
                    for (int pi_m = 0; pi_m < n_pieces_4; pi_m++) {
                        int split_idx_4 = (cx_4 + pi_m * n_clusters_4) % num_split;
                        int q_len_b_4 = q_end_r_4 - q_start_r_4;
                        int rows_b_4 = q_len_b_4 * num_heads;
                        int m_tile_4 = (rows_b_4 + 127) / 128 - 1 - (cx_4 + pi_m * n_clusters_4) / num_split;
                        int row0_4 = m_tile_4 * 128;
                        int rows_left_4 = rows_b_4 - row0_4;
                        int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
                        int rows_valid_4 = ((rows_pos_4 > 128) ? 128 : rows_pos_4);
                        int row_base_4 = q_start_r_4 * num_heads + row0_4;
                        int t_last_4 = (row0_4 + rows_valid_4 - 1) / num_heads;
                        int num_4 = g_len_r_4 - q_len_b_4 + t_last_4 - cp_rank;
                        int vis_raw_4 = num_4 / cp_world + 1;
                        int vis_cap_4 = ((vis_raw_4 > kv_len_r_4) ? kv_len_r_4 : vis_raw_4);
                        int vis_4 = ((num_4 < 0) ? 0 : vis_cap_4);
                        int kv_end_4 = ((rows_valid_4 == 0) ? 0 : vis_4);
                        int n_tiles_total_4 = (kv_end_4 + 127) / 128;
                        int tiles_per_split_4 = (n_tiles_total_4 + num_split - 1) / num_split;
                        int my_start_raw_4 = split_idx_4 * tiles_per_split_4;
                        int my_end_raw_4 = my_start_raw_4 + tiles_per_split_4;
                        int my_end_4 = ((my_end_raw_4 > n_tiles_total_4) ? n_tiles_total_4 : my_end_raw_4);
                        int my_n_raw_4 = my_end_4 - my_start_raw_4;
                        int empty_4 = ((my_n_raw_4 < 1) ? 1 : 0);
                        int my_n_tiles_4 = ((my_n_raw_4 < 1) ? 1 : my_n_raw_4);
                        int my_start_4 = ((my_n_raw_4 < 1) ? 0 : my_start_raw_4);
                        mbarrier_wait(q_full_addr, _phase_q_full_0_1);
                        mbarrier_wait(sfa_full_addr, _phase_sfa_full_0);
                        #pragma unroll 1
                        for (int it_m = 0; it_m < my_n_tiles_4; it_m++) {
                            mbarrier_wait(raw_full_addr + (raw_m) * 8, _phase_raw_full_2);
                            mbarrier_wait(sf_full_addr, _phase_sf_full_0);
                            mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _smem_desc_lo_0 = (((smem_kq_addr) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_0);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"(tmem_q + 0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_0), "r"(sfb_0), "r"(0) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"((tmem_q + 0 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_0 + 4)), "r"((sfb_0 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_1 = (((smem_kq_addr + 4096) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_1);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"(tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_1), "r"(sfb_1), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"((tmem_q + 16 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_1 + 4)), "r"((sfb_1 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_2 = (((smem_kq_addr + 8192) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_2);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"(tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_2), "r"(sfb_2), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"((tmem_q + 32 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_2 + 4)), "r"((sfb_2 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_3 = (((smem_kq_addr + 12288) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_3);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"(tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_3), "r"(sfb_3), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_acc), "r"((tmem_q + 48 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_3 + 4)), "r"((sfb_3 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_4 = ((smem_qr_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_5 = (((smem_kr_addr) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135266320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc), "r"(1) : "memory");
                            int _smem_desc_lo_6 = (((smem_kq_addr + 2048) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_6);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"(tmem_q + 0), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_0), "r"(sfb_4), "r"(0) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"((tmem_q + 0 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_0 + 4)), "r"((sfb_4 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_7 = (((smem_kq_addr + 6144) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_7);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"(tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_1), "r"(sfb_5), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"((tmem_q + 16 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_1 + 4)), "r"((sfb_5 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_8 = (((smem_kq_addr + 10240) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_8);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"(tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_2), "r"(sfb_6), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"((tmem_q + 32 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_2 + 4)), "r"((sfb_6 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_9 = (((smem_kq_addr + 14336) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            {
                                const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_9);
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"(tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(sfa_3), "r"(sfb_7), "r"(1) : "memory");
                                asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_acc + (32))), "r"((tmem_q + 48 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((sfa_3 + 4)), "r"((sfb_7 + 4)), "r"(1) : "memory");
                            }
                            int _smem_desc_lo_10 = ((smem_qr_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_11 = (((smem_kr_addr + 2048) >> 4) & 0x3FFF) + (raw_m) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135266320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"((tmem_acc + (32))), "r"(1) : "memory");
                            tcgen05_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(sf_empty_addr, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(raw_empty_addr + (raw_m) * 8, (uint16_t)(3));
                            _phase_s_empty_0 ^= 1;
                            _phase_sf_full_0 ^= 1;
                            raw_m += 1;
                            if (raw_m == 3) { raw_m = 0; _phase_raw_full_2 ^= 1; }
                            if (it_m == 1) {
                                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            }
                            if (it_m >= 1) {
                                mbarrier_wait(v_full_addr + (v_m) * 8, _phase_v_full);
                                mbarrier_wait(p_full_addr + (p_m) * 8, _phase_p_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _smem_desc_lo_12 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_m) * 512;
                                int _smem_desc_lo_13 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_m) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"((tmem_acc + (64))), "r"((((int)(it_m == 1)) ? 0 : 1)) : "memory");
                                int _smem_desc_lo_14 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_m) * 512;
                                int _smem_desc_lo_15 = ((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (v_m) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"((tmem_acc + (192))), "r"((((int)(it_m == 1)) ? 0 : 1)) : "memory");
                                tcgen05_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(p_empty_addr + (p_m) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(v_empty_addr + (v_m) * 8, (uint16_t)(3));
                                p_m += 1;
                                if (p_m == 2) { p_m = 0; _phase_p_full ^= 1; }
                                v_m += 1;
                                if (v_m == 2) { v_m = 0; _phase_v_full ^= 1; }
                            }
                        }
                        tcgen05_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        if (my_n_tiles_4 == 1) {
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                        }
                        mbarrier_wait(v_full_addr + (v_m) * 8, _phase_v_full);
                        mbarrier_wait(p_full_addr + (p_m) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_16 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_m) * 512;
                        int _smem_desc_lo_17 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_m) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"((tmem_acc + (64))), "r"((((int)(my_n_tiles_4 == 1)) ? 0 : 1)) : "memory");
                        int _smem_desc_lo_18 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_m) * 512;
                        int _smem_desc_lo_19 = ((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (v_m) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"((tmem_acc + (192))), "r"((((int)(my_n_tiles_4 == 1)) ? 0 : 1)) : "memory");
                        tcgen05_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                        tcgen05_commit_cg2_multicast(p_empty_addr + (p_m) * 8, (uint16_t)(3));
                        tcgen05_commit_cg2_multicast(v_empty_addr + (v_m) * 8, (uint16_t)(3));
                        p_m += 1;
                        if (p_m == 2) { p_m = 0; _phase_p_full ^= 1; }
                        v_m += 1;
                        if (v_m == 2) { v_m = 0; _phase_v_full ^= 1; }
                        tcgen05_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        _phase_q_full_0_1 ^= 1;
                        _phase_sfa_full_0 ^= 1;
                        _phase_o_empty_0 ^= 1;
                    }
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                }
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
