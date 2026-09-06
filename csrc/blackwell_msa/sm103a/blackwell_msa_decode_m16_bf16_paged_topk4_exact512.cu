typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) BlackwellMsaTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define BLACKWELL_MSA_INF CUDART_INF_F
#define TMEM_NCOLS 96
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 16
#define TMEM_TMEM_O0_OFFSET 32
#define TMEM_TMEM_O1_OFFSET 48
#define TMEM_TMEM_STATS0_OFFSET 64
#define TMEM_TMEM_STATS1_OFFSET 80
#define NUM_DECODE_KV_STAGES 4
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_EXCH0_U32_OFF 1152
#define SMEM_SMEM_EXCH0_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_U32_STRIDE 256
#define SMEM_SMEM_EXCH1_U32_OFF 1408
#define SMEM_SMEM_EXCH1_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_U32_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 4096
#define SMEM_SMEM_QT_STRIDE 4096
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 4096
#define SMEM_SMEM_P0_STRIDE 4096
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 4096
#define SMEM_SMEM_P1_STRIDE 4096
#define SMEM_SMEM_PAGE_INDICES_OFF 145408
#define SMEM_SMEM_PAGE_INDICES_STAGE_BYTES 2048
#define SMEM_SMEM_PAGE_INDICES_STRIDE 2048
#define SMEM_TOTAL 148480
#define THREADS 512
#define MAX_REQUESTS_CONST 512

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
        :: "r"(mbar_addr), "r"(count));
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

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_minimax_sparse_decode_m16_paged_b2_q1_kv257_h8_hkv1_k4_sm100(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap Q_prefill, __nv_bfloat16* __restrict__ Q_prefill_raw, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_prefill_pair, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_prefill_pair, const __grid_constant__ CUtensorMap KV, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ qo_indptr, int* __restrict__ kv_indptr, int* __restrict__ kv_len_arr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int* __restrict__ task_q_tile, int* __restrict__ task_split, int* __restrict__ task_kv_tile_begin, int* __restrict__ task_kv_tile_end, int* __restrict__ task_qo_begin, int* __restrict__ task_qo_end, int* __restrict__ task_page_begin, int* __restrict__ task_page_end, int* __restrict__ status, int num_requests, int num_q_heads, int num_kv_heads, int max_kv_tiles, int max_splits, int max_task_claims, float softmax_scale_log2, int attention_mode, int is_causal, int derive_q_offset, int record_tasks, int msa_max_pages, int msa_split_policy)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr0_addr = smem + 1024;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_corr1_addr = smem + 1088;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_p0_addr = smem + 137216;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    int* smem_page_indices = reinterpret_cast<int*>(smem_raw + 145408);
    const int smem_page_indices_addr = smem + 145408;

    // Mbarrier init (9 groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'decode_kv' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // decode_done: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // decode_inputs_reusable: 1 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 72)
    #define p_full_addr (mbar_base + 88)
    #define corr_sig_addr (mbar_base + 104)
    #define o_full_addr (mbar_base + 120)
    #define decode_done_addr (mbar_base + 128)
    #define decode_inputs_reusable_addr (mbar_base + 136)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 16;
    const int tmem_tmem_o0 = taddr + 32;
    const int tmem_tmem_o1 = taddr + 48;
    const int tmem_tmem_stats0 = taddr + 64;
    const int tmem_tmem_stats1 = taddr + 80;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            int group_size = 8;
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s_native = taddr + (unsigned int)(((is_wg1 != 0) ? 16 : 0));
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 80 : 64)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
            unsigned int* my_exch_u32_ptr = ((is_wg1 != 0) ? smem_exch1_u32 : smem_exch0_u32);
            float* my_corr_ptr = ((is_wg1 != 0) ? smem_corr1 : smem_corr0);
            unsigned int* base = ((is_wg1 != 0) ? reinterpret_cast<unsigned int*>(smem_p1) : reinterpret_cast<unsigned int*>(smem_p0));
            int direct_decode = 1;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int task_iter = 0; task_iter < 1; task_iter++) {
                int ticket = -1;
                ticket = blockIdx.x * gridDim.y + blockIdx.y + task_iter * gridDim.x * gridDim.y;
                if (ticket >= 2) {
                    ticket = -1;
                }
                if (ticket < 0) {
                    break;
                }
                int kind = 1;
                int direct_request = 0;
                direct_request = ticket / 1 / 1;
                int kv_tile_begin = 0;
                int direct_batch = direct_request;
                direct_batch = direct_request / 1;
                int direct_kv_len = kv_len_arr[direct_batch];
                int kv_tile_end = (direct_kv_len + 128 - 1) / 128;
                kv_tile_end = 4;
                int direct_kv_pairs = 2;
                int direct_split = ticket % 1;
                kv_tile_begin = 2 * (direct_kv_pairs * direct_split / 1);
                kv_tile_end = 2 * (direct_kv_pairs * (direct_split + 1) / 1);
                int num_n_blocks = 4;
                int num_pairs = num_n_blocks / 2;
                float row_max_pair[4];
                float row_sum_pair[4];
                row_max_pair[0] = -BLACKWELL_MSA_INF;
                row_max_pair[1] = -BLACKWELL_MSA_INF;
                row_max_pair[2] = -BLACKWELL_MSA_INF;
                row_max_pair[3] = -BLACKWELL_MSA_INF;
                row_sum_pair[0] = 0.0f;
                row_sum_pair[1] = 0.0f;
                row_sum_pair[2] = 0.0f;
                row_sum_pair[3] = 0.0f;
                int max_decode_pairs = 2;
                #pragma unroll 1
                for (int pair = 0; pair < 2; pair++) {
                    if (2 <= pair) {
                        break;
                    }
                    if (is_wg1 != 0) {
                        mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                        _phase_s_full_1 ^= 1;
                    } else {
                        mbarrier_wait(s_full_addr, _phase_s_full_0);
                        _phase_s_full_0 ^= 1;
                    }
                    float _tmem_load_0[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                        : "r"(my_tmem_s_native)
                        : "memory");
                    float _tmem_load_1[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7]))
                        : "r"(my_tmem_s_native + 1048576)
                        : "memory");
                    float sv[16];
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        sv[c] = _tmem_load_0[c];
                        sv[c + 8] = _tmem_load_1[c];
                    }
                    int valid_cols = smem_page_indices[pair * 2 + is_wg1];
                    int token_in_block = warp_in_wg * 32 + lane / 4;
                    if (token_in_block >= valid_cols) {
                        sv[0] = -BLACKWELL_MSA_INF;
                        sv[1] = -BLACKWELL_MSA_INF;
                        sv[4] = -BLACKWELL_MSA_INF;
                        sv[5] = -BLACKWELL_MSA_INF;
                    }
                    if (valid_cols <= token_in_block + 8) {
                        sv[2] = -BLACKWELL_MSA_INF;
                        sv[3] = -BLACKWELL_MSA_INF;
                        sv[6] = -BLACKWELL_MSA_INF;
                        sv[7] = -BLACKWELL_MSA_INF;
                    }
                    if (valid_cols <= token_in_block + 16) {
                        sv[8] = -BLACKWELL_MSA_INF;
                        sv[9] = -BLACKWELL_MSA_INF;
                        sv[12] = -BLACKWELL_MSA_INF;
                        sv[13] = -BLACKWELL_MSA_INF;
                    }
                    if (valid_cols <= token_in_block + 24) {
                        sv[10] = -BLACKWELL_MSA_INF;
                        sv[11] = -BLACKWELL_MSA_INF;
                        sv[14] = -BLACKWELL_MSA_INF;
                        sv[15] = -BLACKWELL_MSA_INF;
                    }
                    float partial_max[4];
                    float _max_0 = max_noftz(sv[0], sv[2]);
                    float _max_1 = max_noftz(sv[8], sv[10]);
                    float _max_2 = max_noftz(_max_0, _max_1);
                    partial_max[0] = _max_2;
                    float _max_3 = max_noftz(sv[1], sv[3]);
                    float _max_4 = max_noftz(sv[9], sv[11]);
                    float _max_5 = max_noftz(_max_3, _max_4);
                    partial_max[1] = _max_5;
                    float _max_6 = max_noftz(sv[4], sv[6]);
                    float _max_7 = max_noftz(sv[12], sv[14]);
                    float _max_8 = max_noftz(_max_6, _max_7);
                    partial_max[2] = _max_8;
                    float _max_9 = max_noftz(sv[5], sv[7]);
                    float _max_10 = max_noftz(sv[13], sv[15]);
                    float _max_11 = max_noftz(_max_9, _max_10);
                    partial_max[3] = _max_11;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 4; c_1++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, partial_max[c_1], 16);
                        float _max_12 = max_noftz(partial_max[c_1], _shfl_xor_0);
                        partial_max[c_1] = _max_12;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, partial_max[c_1], 8);
                        float _max_13 = max_noftz(partial_max[c_1], _shfl_xor_1);
                        partial_max[c_1] = _max_13;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, partial_max[c_1], 4);
                        float _max_14 = max_noftz(partial_max[c_1], _shfl_xor_2);
                        partial_max[c_1] = _max_14;
                    }
                    if (wg_tid < 16) {
                        uint32_t _amf_u_0 = __float_as_uint(-BLACKWELL_MSA_INF);
                        uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                        my_exch_u32_ptr[wg_tid] = _amf_enc_0;
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    int head_pair_base = lane % 4 * 2;
                    float old_max_pair[4];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 4; c_2++) {
                        old_max_pair[c_2] = row_max_pair[c_2];
                    }
                    if (lane < 4) {
                        uint32_t _amf_u_1 = __float_as_uint(partial_max[0]);
                        uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                        atomicMax(&my_exch_u32_ptr[head_pair_base], _amf_enc_1);
                        uint32_t _amf_u_2 = __float_as_uint(partial_max[1]);
                        uint32_t _amf_mask_2 = -int32_t(_amf_u_2 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_2 ^ _amf_mask_2;
                        atomicMax(&my_exch_u32_ptr[head_pair_base + 1], _amf_enc_2);
                        uint32_t _amf_u_3 = __float_as_uint(partial_max[2]);
                        uint32_t _amf_mask_3 = -int32_t(_amf_u_3 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_3 = _amf_u_3 ^ _amf_mask_3;
                        atomicMax(&my_exch_u32_ptr[head_pair_base + 8], _amf_enc_3);
                        uint32_t _amf_u_4 = __float_as_uint(partial_max[3]);
                        uint32_t _amf_mask_4 = -int32_t(_amf_u_4 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_4 = _amf_u_4 ^ _amf_mask_4;
                        atomicMax(&my_exch_u32_ptr[head_pair_base + 9], _amf_enc_4);
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    float new_max_pair[4];
                    uint32_t _amf_u_5 = my_exch_u32_ptr[head_pair_base];
                    uint32_t _amf_mask_5 = ((_amf_u_5 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_5 ^ _amf_mask_5);
                    new_max_pair[0] = _amf_dec_0;
                    uint32_t _amf_u_6 = my_exch_u32_ptr[head_pair_base + 1];
                    uint32_t _amf_mask_6 = ((_amf_u_6 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_6 ^ _amf_mask_6);
                    new_max_pair[1] = _amf_dec_1;
                    uint32_t _amf_u_7 = my_exch_u32_ptr[head_pair_base + 8];
                    uint32_t _amf_mask_7 = ((_amf_u_7 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_2 = __uint_as_float(_amf_u_7 ^ _amf_mask_7);
                    new_max_pair[2] = _amf_dec_2;
                    uint32_t _amf_u_8 = my_exch_u32_ptr[head_pair_base + 9];
                    uint32_t _amf_mask_8 = ((_amf_u_8 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_3 = __uint_as_float(_amf_u_8 ^ _amf_mask_8);
                    new_max_pair[3] = _amf_dec_3;
                    float acc_scale_pair[4];
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 4; c_3++) {
                        float delta = softmax_scale_log2 * (old_max_pair[c_3] - new_max_pair[c_3]);
                        float _exp2_0 = approx_exp2(delta);
                        acc_scale_pair[c_3] = ((old_max_pair[c_3] > -BLACKWELL_MSA_INF) ? _exp2_0 : 1.0f);
                        row_max_pair[c_3] = new_max_pair[c_3];
                    }
                    float acc_scale[16];
                    #pragma unroll
                    for (int c_4 = 0; c_4 < 16; c_4++) {
                        int scale_slot = c_4 % 2 + ((c_4 >= 8) ? 2 : 0);
                        int scale_lane = c_4 % 8 / 2;
                        float _shfl_0 = __shfl_sync(0xFFFFFFFF, acc_scale_pair[scale_slot], scale_lane);
                        acc_scale[c_4] = _shfl_0;
                    }
                    tmem_st_x16_f32(my_tmem_stats, acc_scale);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(corr_sig_addr + 8);
                    } else {
                        mbarrier_arrive(corr_sig_addr);
                    }
                    float exp_vals[16];
                    #pragma unroll
                    for (int head_slot = 0; head_slot < 4; head_slot++) {
                        int head_base = head_slot / 2 * 4 + head_slot % 2;
                        float safe_max = ((new_max_pair[head_slot] == -BLACKWELL_MSA_INF) ? 0.0f : new_max_pair[head_slot]);
                        float max_scaled = safe_max * softmax_scale_log2;
                        #pragma unroll
                        for (int k_pos = 0; k_pos < 4; k_pos++) {
                            int value_idx = head_base + k_pos % 2 * 2 + k_pos / 2 * 8;
                            float _exp2_1 = approx_exp2(sv[value_idx] * softmax_scale_log2 - max_scaled);
                            exp_vals[value_idx] = _exp2_1;
                        }
                    }
                    float pair_sum[4];
                    pair_sum[0] = exp_vals[0] + exp_vals[2] + exp_vals[8] + exp_vals[10];
                    pair_sum[1] = exp_vals[1] + exp_vals[3] + exp_vals[9] + exp_vals[11];
                    pair_sum[2] = exp_vals[4] + exp_vals[6] + exp_vals[12] + exp_vals[14];
                    pair_sum[3] = exp_vals[5] + exp_vals[7] + exp_vals[13] + exp_vals[15];
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 4; c_5++) {
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_5], 16);
                        pair_sum[c_5] = pair_sum[c_5] + _shfl_xor_3;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_5], 8);
                        pair_sum[c_5] = pair_sum[c_5] + _shfl_xor_4;
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_5], 4);
                        pair_sum[c_5] = pair_sum[c_5] + _shfl_xor_5;
                        row_sum_pair[c_5] = row_sum_pair[c_5] * acc_scale_pair[c_5] + pair_sum[c_5];
                    }
                    unsigned int p_packed[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(exp_vals[_lp*2 + 0], exp_vals[_lp*2+1 + 0]));
                        p_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int k_panel = warp_in_wg / 2;
                    int warp_in_panel = warp_in_wg % 2;
                    int matrix = lane / 8;
                    int matrix_row = lane % 8;
                    int matrix_row_group = matrix / 2;
                    int matrix_col = warp_in_panel * 4 + matrix % 2;
                    int p_row_offset = k_panel * 16 * 128 + (matrix_row_group * 8 + matrix_row) * 128;
                    int swizzled_col0 = matrix_col ^ matrix_row;
                    const void* _stmatrix_ptr_9 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(base) + (p_row_offset + swizzled_col0 * 16));
                    uint64_t _stmatrix_addr64_9;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_9) : "l"(_stmatrix_ptr_9));
                    uint32_t _stmatrix_addr_9;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_9) : "l"(_stmatrix_addr64_9));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[3]))
                        : "memory");
                    int swizzled_col1 = matrix_col + 2 ^ matrix_row;
                    const void* _stmatrix_ptr_10 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(base) + (p_row_offset + swizzled_col1 * 16));
                    uint64_t _stmatrix_addr64_10;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_10) : "l"(_stmatrix_ptr_10));
                    uint32_t _stmatrix_addr_10;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_10) : "l"(_stmatrix_addr64_10));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[4])), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[5])), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[6])), "r"(*reinterpret_cast<const uint32_t*>(&p_packed[7]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_addr + 8);
                    } else {
                        mbarrier_arrive(p_full_addr);
                    }
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                int head_pair_base_1 = lane % 4 * 2;
                if (lane < 4) {
                    my_exch_ptr[warp_in_wg * 16 + head_pair_base_1] = row_sum_pair[0];
                    my_exch_ptr[warp_in_wg * 16 + head_pair_base_1 + 1] = row_sum_pair[1];
                    my_exch_ptr[warp_in_wg * 16 + head_pair_base_1 + 8] = row_sum_pair[2];
                    my_exch_ptr[warp_in_wg * 16 + head_pair_base_1 + 9] = row_sum_pair[3];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                if (warp_in_wg == 0 && lane < 4) {
                    #pragma unroll
                    for (int head_slot_1 = 0; head_slot_1 < 4; head_slot_1++) {
                        int head_idx = head_pair_base_1 + head_slot_1 / 2 * 8 + head_slot_1 % 2;
                        float total_sum = my_exch_ptr[head_idx] + my_exch_ptr[16 + head_idx] + my_exch_ptr[32 + head_idx] + my_exch_ptr[48 + head_idx];
                        my_corr_ptr[head_idx] = total_sum;
                        my_exch_ptr[head_idx] = row_max_pair[head_slot_1];
                    }
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                if (is_wg1 != 0) {
                    mbarrier_arrive(corr_sig_addr + 8);
                } else {
                    mbarrier_arrive(corr_sig_addr);
                }
                mbarrier_arrive(decode_inputs_reusable_addr);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            int d_idx = warp % 4 * 32 + lane;
            int direct_decode_1 = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (int task_iter_1 = 0; task_iter_1 < 1; task_iter_1++) {
                int ticket_1 = -1;
                ticket_1 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_1 * gridDim.x * gridDim.y;
                if (ticket_1 >= 2) {
                    ticket_1 = -1;
                }
                if (ticket_1 < 0) {
                    break;
                }
                int kind_1 = 1;
                int request = 0;
                int kv_head = 0;
                int split = 0;
                int splits = 1;
                int logical_output = ticket_1 / 1;
                request = logical_output / 1;
                kv_head = logical_output - request * 1;
                split = ticket_1 - logical_output * 1;
                splits = 1;
                int kv_tile_begin_1 = 0;
                int direct_batch_1 = request;
                direct_batch_1 = request / 1;
                int direct_kv_len_1 = kv_len_arr[direct_batch_1];
                int kv_tile_end_1 = (direct_kv_len_1 + 128 - 1) / 128;
                int qo_begin = qo_indptr[request];
                kv_tile_end_1 = 4;
                qo_begin = request;
                int direct_kv_pairs_1 = 2;
                kv_tile_begin_1 = 2 * (direct_kv_pairs_1 * 0 / 1);
                kv_tile_end_1 = 2 * (direct_kv_pairs_1 * 1 / 1);
                int num_n_blocks_1 = 4;
                int group_size_1 = 8;
                int num_pairs_1 = num_n_blocks_1 / 2;
                int max_decode_pairs_1 = 2;
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 2; pair_1++) {
                    if (2 <= pair_1) {
                        break;
                    }
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 64 + (unsigned int)corr_row);
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 32 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h = 0; h < 16; h++) {
                        _tmem_load_6[h] = _tmem_load_6[h] * _tmem_load_5[h];
                    }
                    tmem_st_x16_f32(taddr + 32 + (unsigned int)corr_row, _tmem_load_6);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 80 + (unsigned int)corr_row);
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + 48 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_1 = 0; h_1 < 16; h_1++) {
                        _tmem_load_8[h_1] = _tmem_load_8[h_1] * _tmem_load_7[h_1];
                    }
                    tmem_st_x16_f32(taddr + 48 + (unsigned int)corr_row, _tmem_load_8);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + 8);
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                float scale0[16];
                float scale1[16];
                float final_sum[16];
                float final_max[16];
                #pragma unroll
                for (int c_6 = 0; c_6 < 16; c_6++) {
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, smem_exch0[c_6], c_6);
                    float _shfl_2 = __shfl_sync(0xFFFFFFFF, smem_exch1[c_6], c_6);
                    float _shfl_3 = __shfl_sync(0xFFFFFFFF, smem_corr0[c_6], c_6);
                    float _shfl_4 = __shfl_sync(0xFFFFFFFF, smem_corr1[c_6], c_6);
                    float _max_17 = max_noftz(_shfl_1, _shfl_2);
                    float fm = _max_17;
                    final_max[c_6] = fm;
                    float d0 = ((_shfl_1 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_1 - fm));
                    float d1 = ((_shfl_2 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_2 - fm));
                    float _exp2_3 = approx_exp2(d0);
                    scale0[c_6] = _exp2_3;
                    float _exp2_4 = approx_exp2(d1);
                    scale1[c_6] = _exp2_4;
                    final_sum[c_6] = _shfl_3 * scale0[c_6] + _shfl_4 * scale1[c_6];
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float inv_sum[16];
                #pragma unroll
                for (int h_2 = 0; h_2 < 16; h_2++) {
                    float _rcp_0 = approx_rcp(final_sum[h_2]);
                    inv_sum[h_2] = ((final_sum[h_2] > 0.0f) ? _rcp_0 : 0.0f);
                }
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + 32 + (unsigned int)corr_row);
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + 48 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_3 = 0; h_3 < 16; h_3++) {
                    if (8 > h_3) {
                        float merged = _tmem_load_9[h_3] * scale0[h_3] + _tmem_load_10[h_3] * scale1[h_3];
                        int q_row = qo_begin * 8 + kv_head * 8 + h_3;
                        int out_idx = q_row * 128 + d_idx;
                        if (d_idx == 0) {
                            float natural_lse = -BLACKWELL_MSA_INF;
                            if (final_sum[h_3] > 0.0f) {
                                float _log2_0;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum[h_3]));
                                natural_lse = final_max[h_3] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                            }
                            *(reinterpret_cast<float*>(msa_lse + q_row) + (0)) = natural_lse;
                        }
                        *(reinterpret_cast<__nv_bfloat16*>(O + out_idx) + (0)) = __float2bfloat16_rn(merged * inv_sum[h_3]);
                    }
                }
                mbarrier_arrive(decode_done_addr);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int direct_decode_2 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_decode_done_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            #pragma unroll 1
            for (int task_iter_2 = 0; task_iter_2 < 1; task_iter_2++) {
                int ticket_2 = -1;
                ticket_2 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_2 * gridDim.x * gridDim.y;
                if (ticket_2 >= 2) {
                    ticket_2 = -1;
                }
                if (ticket_2 < 0) {
                    break;
                }
                int kind_2 = 1;
                int direct_request_1 = 0;
                direct_request_1 = ticket_2 / 1 / 1;
                int kv_tile_begin_2 = 0;
                int direct_batch_2 = direct_request_1;
                direct_batch_2 = direct_request_1 / 1;
                int direct_kv_len_2 = kv_len_arr[direct_batch_2];
                int kv_tile_end_2 = (direct_kv_len_2 + 128 - 1) / 128;
                kv_tile_end_2 = 4;
                int direct_kv_pairs_2 = 2;
                int direct_split_1 = ticket_2 % 1;
                kv_tile_begin_2 = 2 * (direct_kv_pairs_2 * direct_split_1 / 1);
                kv_tile_end_2 = 2 * (direct_kv_pairs_2 * (direct_split_1 + 1) / 1);
                int num_n_blocks_2 = 4;
                int num_pairs_2 = num_n_blocks_2 / 2;
                int inst0_stage = 0;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                int _mma_a_lo_8 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_8 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_tmem_s0), "r"(0));
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr);
                int max_decode_pairs_2 = 2;
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < 1; pair_2++) {
                    if (pair_2 >= 1) {
                        break;
                    }
                    int s0 = inst0_stage;
                    int s1 = (inst0_stage + 1) % 4;
                    int s0_next = (inst0_stage + 2) % 4;
                    mbarrier_wait(kv_full_addr + (s1) * 8, 0);
                    int _mma_a_lo_9 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1) * 2048);
                    int _mma_b_lo_9 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"(tmem_tmem_s1), "r"(0));
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (s1) * 8);
                    mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 2048);
                    int _mma_b_lo_10 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (s0) * 8);
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_11 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_9), "r"(tmem_tmem_s0), "r"(0));
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_12 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 2048);
                    int _mma_b_lo_12 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (s1) * 8);
                    inst0_stage = s0_next;
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                int s0_last = inst0_stage;
                int s1_last = (inst0_stage + 1) % 4;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 0);
                int _mma_a_lo_13 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_last) * 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_8), "r"(tmem_tmem_s1), "r"(0));
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (s1_last) * 8);
                mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_14 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
                int _mma_b_lo_14 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                elect_commit(kv_empty_addr + (s0_last) * 8);
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_15 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
                int _mma_b_lo_15 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_15), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                elect_commit(kv_empty_addr + (s1_last) * 8);
                elect_commit(o_full_addr);
                mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                _phase_decode_done_0 ^= 1;
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 14) {
        { // producer_main
            int direct_decode_3 = 1;
            unsigned int _phase_decode_inputs_reusable_0 = 0;
            #pragma unroll 1
            for (int task_iter_3 = 0; task_iter_3 < 1; task_iter_3++) {
                int ticket_3 = -1;
                ticket_3 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_3 * gridDim.x * gridDim.y;
                if (ticket_3 >= 2) {
                    ticket_3 = -1;
                }
                if (ticket_3 < 0) {
                    break;
                }
                int kind_3 = 1;
                int direct_request_2 = 0;
                int kv_head_1 = 0;
                direct_request_2 = ticket_3 / 1 / 1;
                kv_head_1 = ticket_3 / 1 % 1;
                int q_tile = 0;
                int kv_tile_begin_3 = 0;
                int direct_batch_3 = direct_request_2;
                direct_batch_3 = direct_request_2 / 1;
                int direct_kv_len_3 = kv_len_arr[direct_batch_3];
                int kv_tile_end_3 = (direct_kv_len_3 + 128 - 1) / 128;
                int qo_begin_1 = qo_indptr[direct_request_2];
                int page_begin = kv_indptr[direct_request_2];
                kv_tile_end_3 = 4;
                qo_begin_1 = direct_request_2;
                page_begin = kv_indptr[direct_batch_3];
                int direct_kv_pairs_3 = 2;
                int direct_split_2 = ticket_3 % 1;
                kv_tile_begin_3 = 2 * (direct_kv_pairs_3 * direct_split_2 / 1);
                kv_tile_end_3 = 2 * (direct_kv_pairs_3 * (direct_split_2 + 1) / 1);
                int num_n_blocks_3 = 4;
                int group_size_2 = 8;
                if (elect_sync()) {
                    int q_row_1 = qo_begin_1 * 8 + kv_head_1 * 8 + q_tile * 16;
                    mbarrier_arrive_expect_tx(q_full_addr, 4096);
                    tma_3d_gmem2smem(smem_qt_addr, (&Q), 0, q_row_1, 0, q_full_addr);
                }
                int native_num_n_blocks = num_n_blocks_3;
                if (elect_sync()) {
                    int native_kv_stage = 0;
                    int native_kv_phase = 1;
                    int native_prefill = 4;
                    #pragma unroll
                    for (int native_ni = 0; native_ni < 4; native_ni++) {
                        if (4 <= native_ni) {
                            break;
                        }
                        int native_n_block = 3 - native_ni;
                        int native_pg0 = 0;
                        int native_pg1 = 0;
                        int msa_token_base = 0;
                        int msa_page_head = 0;
                        int msa_valid_cols = 128;
                        int batch = direct_request_2 / 1;
                        int query_in_batch = direct_request_2 - batch * 1;
                        int selected_block = task_kind[(kv_head_1 * 2 + direct_request_2) * 4 + native_n_block];
                        int kv_len = task_kv_head[batch];
                        int valid_cols_1 = 0;
                        if (selected_block >= 0) {
                            int block_start = selected_block * 128;
                            valid_cols_1 = kv_len - block_start;
                            if (valid_cols_1 > 128) {
                                valid_cols_1 = 128;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                            int query_position = kv_len - 1 + query_in_batch;
                            int causal_cols = query_position - block_start + 1;
                            if (valid_cols_1 > causal_cols) {
                                valid_cols_1 = causal_cols;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                        }
                        int token_base = 0;
                        int page_head = 0;
                        int physical_page = 0;
                        if (selected_block >= 0) {
                            physical_page = kv_indices[batch * 3 + selected_block];
                            if (physical_page < 0) {
                                valid_cols_1 = 0;
                                physical_page = 0;
                            }
                        }
                        page_head = physical_page * 1 + kv_head_1;
                        msa_token_base = token_base;
                        msa_page_head = page_head;
                        msa_valid_cols = valid_cols_1;
                        smem_page_indices[native_ni] = msa_valid_cols;
                        smem_page_indices[4 + native_ni] = msa_token_base;
                        smem_page_indices[8 + native_ni] = msa_page_head;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (native_kv_stage) * 8, native_kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (native_kv_stage) * 8, 32768);
                        int native_dst = smem_kv_addr + (unsigned int)(native_kv_stage * 32768);
                        int token0 = msa_token_base;
                        int token1 = msa_token_base + 64;
                        token0 = 0;
                        token1 = 64;
                        tma_4d_gmem2smem(native_dst, (&K), 0, token0, 0, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                        tma_4d_gmem2smem(native_dst + 8192, (&K), 0, token1, 0, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                        tma_4d_gmem2smem(native_dst + 16384, (&K), 0, token0, 1, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                        tma_4d_gmem2smem(native_dst + 24576, (&K), 0, token1, 1, msa_page_head, kv_full_addr + (native_kv_stage) * 8);
                        native_kv_stage += 1;
                        if (native_kv_stage == 4) { native_kv_stage = 0; native_kv_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int native_ni_1 = 0; native_ni_1 < 4; native_ni_1++) {
                        if (4 <= native_ni_1) {
                            break;
                        }
                        int native_stage = native_ni_1 % 4;
                        int native_n_block_1 = 3 - native_ni_1;
                        int native_pg0_1 = 0;
                        int native_pg1_1 = 0;
                        int msa_token_base_1 = 0;
                        int msa_page_head_1 = 0;
                        int msa_valid_cols_1 = 128;
                        msa_valid_cols_1 = smem_page_indices[native_ni_1];
                        msa_token_base_1 = smem_page_indices[4 + native_ni_1];
                        msa_page_head_1 = smem_page_indices[8 + native_ni_1];
                        mbarrier_wait(kv_empty_addr + (native_stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (native_stage) * 8, 32768);
                        int native_dst_1 = smem_kv_addr + (unsigned int)(native_stage * 32768);
                        int token0_1 = msa_token_base_1;
                        int token1_1 = msa_token_base_1 + 64;
                        token0_1 = 0;
                        token1_1 = 64;
                        tma_4d_gmem2smem(native_dst_1, (&V), 0, token0_1, 0, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                        tma_4d_gmem2smem(native_dst_1 + 8192, (&V), 0, token1_1, 0, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                        tma_4d_gmem2smem(native_dst_1 + 16384, (&V), 0, token0_1, 1, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                        tma_4d_gmem2smem(native_dst_1 + 24576, (&V), 0, token1_1, 1, msa_page_head_1, kv_full_addr + (native_stage) * 8);
                        int native_next_ni = native_ni_1 + 4;
                        if (native_next_ni < 4) {
                            int native_next_n = 3 - native_next_ni;
                            int native_npg0 = 0;
                            int native_npg1 = 0;
                            int msa_next_token_base = 0;
                            int msa_next_page_head = 0;
                            int msa_next_valid_cols = 128;
                            int batch_1 = direct_request_2 / 1;
                            int query_in_batch_1 = direct_request_2 - batch_1 * 1;
                            int selected_block_1 = task_kind[(kv_head_1 * 2 + direct_request_2) * 4 + native_next_n];
                            int kv_len_1 = task_kv_head[batch_1];
                            int valid_cols_2 = 0;
                            if (selected_block_1 >= 0) {
                                int block_start_1 = selected_block_1 * 128;
                                valid_cols_2 = kv_len_1 - block_start_1;
                                if (valid_cols_2 > 128) {
                                    valid_cols_2 = 128;
                                }
                                if (valid_cols_2 < 0) {
                                    valid_cols_2 = 0;
                                }
                                int query_position_1 = kv_len_1 - 1 + query_in_batch_1;
                                int causal_cols_1 = query_position_1 - block_start_1 + 1;
                                if (valid_cols_2 > causal_cols_1) {
                                    valid_cols_2 = causal_cols_1;
                                }
                                if (valid_cols_2 < 0) {
                                    valid_cols_2 = 0;
                                }
                            }
                            int token_base_1 = 0;
                            int page_head_1 = 0;
                            int physical_page_1 = 0;
                            if (selected_block_1 >= 0) {
                                physical_page_1 = kv_indices[batch_1 * 3 + selected_block_1];
                                if (physical_page_1 < 0) {
                                    valid_cols_2 = 0;
                                    physical_page_1 = 0;
                                }
                            }
                            page_head_1 = physical_page_1 * 1 + kv_head_1;
                            msa_next_token_base = token_base_1;
                            msa_next_page_head = page_head_1;
                            msa_next_valid_cols = valid_cols_2;
                            smem_page_indices[native_next_ni] = msa_next_valid_cols;
                            smem_page_indices[4 + native_next_ni] = msa_next_token_base;
                            smem_page_indices[8 + native_next_ni] = msa_next_page_head;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_wait(kv_empty_addr + (native_stage) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (native_stage) * 8, 32768);
                            int native_kdst = smem_kv_addr + (unsigned int)(native_stage * 32768);
                            int token0_2 = msa_next_token_base;
                            int token1_2 = msa_next_token_base + 64;
                            token0_2 = 0;
                            token1_2 = 64;
                            tma_4d_gmem2smem(native_kdst, (&K), 0, token0_2, 0, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                            tma_4d_gmem2smem(native_kdst + 8192, (&K), 0, token1_2, 0, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                            tma_4d_gmem2smem(native_kdst + 16384, (&K), 0, token0_2, 1, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                            tma_4d_gmem2smem(native_kdst + 24576, (&K), 0, token1_2, 1, msa_next_page_head, kv_full_addr + (native_stage) * 8);
                        }
                    }
                }
                num_n_blocks_3 = 0;
                int kv_stage = 0;
                int kv_phase = 1;
                int prefill = ((num_n_blocks_3 < 4) ? num_n_blocks_3 : 4);
                mbarrier_wait(decode_inputs_reusable_addr, _phase_decode_inputs_reusable_0);
                _phase_decode_inputs_reusable_0 ^= 1;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
