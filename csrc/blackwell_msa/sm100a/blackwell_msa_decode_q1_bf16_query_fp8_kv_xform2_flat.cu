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
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 4096
#define SMEM_SMEM_QT_STRIDE 4096
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_KV_FP8_OFF 151168
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 4096
#define SMEM_SMEM_P0_STRIDE 4096
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 4096
#define SMEM_SMEM_P1_STRIDE 4096
#define SMEM_SMEM_PAGE_INDICES_OFF 145408
#define SMEM_SMEM_PAGE_INDICES_STAGE_BYTES 2048
#define SMEM_SMEM_PAGE_INDICES_STRIDE 2048
#define SMEM_DECODE_ROW_MAX_OFF 150144
#define SMEM_DECODE_ROW_MAX_STAGE_BYTES 512
#define SMEM_DECODE_ROW_MAX_STRIDE 512
#define SMEM_DECODE_ROW_SUM_OFF 150656
#define SMEM_DECODE_ROW_SUM_STAGE_BYTES 512
#define SMEM_DECODE_ROW_SUM_STRIDE 512
#define SMEM_TOTAL 216704
#define THREADS 512

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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_blackwell_batch_attention_msa_decode_q1_fp8_flat_xform2_v1(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int num_requests, float softmax_scale_log2, int msa_max_pages)
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
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 151168);
    const int smem_kv_fp8_addr = smem + 151168;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_p0_addr = smem + 137216;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    int* smem_page_indices = reinterpret_cast<int*>(smem_raw + 145408);
    const int smem_page_indices_addr = smem + 145408;
    float* decode_row_max = reinterpret_cast<float*>(smem_raw + 150144);
    const int decode_row_max_addr = smem + 150144;
    float* decode_row_sum = reinterpret_cast<float*>(smem_raw + 150656);
    const int decode_row_sum_addr = smem + 150656;

    // Mbarrier init (10 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

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
            // kv_src_full: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 120, 256);
            mbarrier_init(smem + 128, 256);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            mbarrier_init(smem + 144, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // decode_done: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_src_full_addr (mbar_base + 40)
    #define kv_empty_addr (mbar_base + 72)
    #define s_full_addr (mbar_base + 104)
    #define p_full_addr (mbar_base + 120)
    #define corr_sig_addr (mbar_base + 136)
    #define corr_done_addr (mbar_base + 152)
    #define o_full_addr (mbar_base + 168)
    #define decode_done_addr (mbar_base + 184)
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
            int group_size = 16;
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s = taddr + (unsigned int)(((is_wg1 != 0) ? 16 : 0)) + (unsigned int)(tmem_row_base_v << 16);
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 80 : 64)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
            float* my_corr_ptr = ((is_wg1 != 0) ? smem_corr1 : smem_corr0);
            unsigned int* base = ((is_wg1 != 0) ? reinterpret_cast<unsigned int*>(smem_p1) : reinterpret_cast<unsigned int*>(smem_p0));
            int direct_request = 0;
            direct_request = blockIdx.x;
            int kv_tile_begin = 0;
            int kv_tile_end = 16;
            int num_n_blocks = kv_tile_end - kv_tile_begin;
            int num_pairs = num_n_blocks / 2;
            const int row_state_base = warp * 16;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                decode_row_max[row_state_base + c] = -BLACKWELL_MSA_INF;
                decode_row_sum[row_state_base + c] = 0.0f;
            }
            int max_decode_pairs = 8;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int pair = 0; pair < max_decode_pairs; pair++) {
                if (num_pairs <= pair) {
                    break;
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                    _phase_s_full_1 ^= 1;
                } else {
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                }
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], my_tmem_s);
                int valid_cols = smem_page_indices[pair * 2 + is_wg1];
                int token_in_block = warp_in_wg * 32 + lane;
                if (token_in_block >= valid_cols) {
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 16; c_1++) {
                        _tmem_load_0[c_1] = -BLACKWELL_MSA_INF;
                    }
                }
                float partial_max[16];
                #pragma unroll
                for (int c_2 = 0; c_2 < 16; c_2++) {
                    partial_max[c_2] = _tmem_load_0[c_2];
                }
                #pragma unroll
                for (int c_3 = 0; c_3 < 16; c_3++) {
                    float _warp_reduce_0 = partial_max[c_3];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                    partial_max[c_3] = _warp_reduce_0;
                }
                if (lane < 16) {
                    my_exch_ptr[warp_in_wg * 16 + lane] = partial_max[lane];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                float tile_max[16];
                if (lane < 16) {
                    float _max_0 = max_noftz(my_exch_ptr[lane], my_exch_ptr[16 + lane]);
                    float _max_1 = max_noftz(my_exch_ptr[32 + lane], my_exch_ptr[48 + lane]);
                    float _max_2 = max_noftz(_max_0, _max_1);
                    tile_max[lane] = _max_2;
                }
                #pragma unroll
                for (int c_4 = 0; c_4 < 16; c_4++) {
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, tile_max[c_4], c_4);
                    tile_max[c_4] = _shfl_0;
                }
                float acc_scale[16];
                #pragma unroll
                for (int c_5 = 0; c_5 < 16; c_5++) {
                    float old_max = decode_row_max[row_state_base + c_5];
                    float _max_3 = max_noftz(old_max, tile_max[c_5]);
                    float new_max = _max_3;
                    decode_row_max[row_state_base + c_5] = new_max;
                    float delta = softmax_scale_log2 * (old_max - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    acc_scale[c_5] = ((old_max > -BLACKWELL_MSA_INF) ? _exp2_0 : 1.0f);
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
                for (int c_6 = 0; c_6 < 16; c_6++) {
                    float new_max_1 = decode_row_max[row_state_base + c_6];
                    float safe_max = ((new_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : new_max_1);
                    float max_scaled = safe_max * softmax_scale_log2;
                    float _exp2_1 = approx_exp2(_tmem_load_0[c_6] * softmax_scale_log2 - max_scaled);
                    exp_vals[c_6] = _exp2_1;
                }
                float warp_sum[16];
                #pragma unroll
                for (int c_7 = 0; c_7 < 16; c_7++) {
                    warp_sum[c_7] = exp_vals[c_7];
                }
                #pragma unroll
                for (int c_8 = 0; c_8 < 16; c_8++) {
                    float _warp_reduce_1 = warp_sum[c_8];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                    warp_sum[c_8] = _warp_reduce_1;
                }
                #pragma unroll
                for (int c_9 = 0; c_9 < 16; c_9++) {
                    float old_sum = decode_row_sum[row_state_base + c_9];
                    decode_row_sum[row_state_base + c_9] = old_sum * acc_scale[c_9] + warp_sum[c_9];
                }
                #pragma unroll
                for (int h = 0; h < 16; h++) {
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(exp_vals[h]);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        const void* _ptr_0 = reinterpret_cast<const void*>((reinterpret_cast<uint8_t*>(base) + (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 ^ (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 >> 7 & 7) << 4)));
                        uint64_t _addr64_0;
                        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_addr64_0) : "l"(_ptr_0));
                        uint32_t _addr_0;
                        asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_addr_0) : "l"(_addr64_0));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                }
                asm volatile("fence.proxy.async;");
                if (is_wg1 != 0) {
                    mbarrier_arrive(p_full_addr + 8);
                } else {
                    mbarrier_arrive(p_full_addr);
                }
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (lane < 16) {
                my_exch_ptr[warp_in_wg * 16 + lane] = decode_row_sum[row_state_base + lane];
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            float total_sum[16];
            if (lane < 16) {
                total_sum[lane] = my_exch_ptr[lane] + my_exch_ptr[16 + lane] + my_exch_ptr[32 + lane] + my_exch_ptr[48 + lane];
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (warp_in_wg == 0 && lane < 16) {
                my_corr_ptr[lane] = total_sum[lane];
                my_exch_ptr[lane] = decode_row_max[row_state_base + lane];
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (is_wg1 != 0) {
                mbarrier_arrive(corr_sig_addr + 8);
            } else {
                mbarrier_arrive(corr_sig_addr);
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
            int request = 0;
            int kv_head = 0;
            request = blockIdx.x;
            kv_head = blockIdx.y;
            int kv_tile_begin_1 = 0;
            int kv_tile_end_1 = 16;
            int qo_begin = request;
            int num_n_blocks_1 = kv_tile_end_1 - kv_tile_begin_1;
            int group_size_1 = 16;
            int num_pairs_1 = num_n_blocks_1 / 2;
            int max_decode_pairs_1 = 8;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            #pragma unroll 1
            for (int pair_1 = 0; pair_1 < max_decode_pairs_1; pair_1++) {
                if (num_pairs_1 <= pair_1) {
                    break;
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 64 + (unsigned int)corr_row);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 32 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_1 = 0; h_1 < 16; h_1++) {
                    _tmem_load_2[h_1] = _tmem_load_2[h_1] * _tmem_load_1[h_1];
                }
                tmem_st_x16_f32(taddr + 32 + (unsigned int)corr_row, _tmem_load_2);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 80 + (unsigned int)corr_row);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + 48 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_2 = 0; h_2 < 16; h_2++) {
                    _tmem_load_4[h_2] = _tmem_load_4[h_2] * _tmem_load_3[h_2];
                }
                tmem_st_x16_f32(taddr + 48 + (unsigned int)corr_row, _tmem_load_4);
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
            for (int c_10 = 0; c_10 < 16; c_10++) {
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, smem_exch0[c_10], c_10);
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, smem_exch1[c_10], c_10);
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, smem_corr0[c_10], c_10);
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, smem_corr1[c_10], c_10);
                float _max_4 = max_noftz(_shfl_1, _shfl_2);
                float fm = _max_4;
                final_max[c_10] = fm;
                float d0 = ((_shfl_1 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_1 - fm));
                float d1 = ((_shfl_2 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_2 - fm));
                float _exp2_2 = approx_exp2(d0);
                scale0[c_10] = _exp2_2;
                float _exp2_3 = approx_exp2(d1);
                scale1[c_10] = _exp2_3;
                final_sum[c_10] = _shfl_3 * scale0[c_10] + _shfl_4 * scale1[c_10];
            }
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sum[16];
            #pragma unroll
            for (int h_3 = 0; h_3 < 16; h_3++) {
                float _rcp_0 = approx_rcp(final_sum[h_3]);
                inv_sum[h_3] = ((final_sum[h_3] > 0.0f) ? _rcp_0 : 0.0f);
            }
            float _tmem_load_5[16];
            tmem_ld_x16(&_tmem_load_5[0], taddr + 32 + (unsigned int)corr_row);
            float _tmem_load_6[16];
            tmem_ld_x16(&_tmem_load_6[0], taddr + 48 + (unsigned int)corr_row);
            #pragma unroll
            for (int h_4 = 0; h_4 < 16; h_4++) {
                if (group_size_1 > h_4) {
                    float merged = _tmem_load_5[h_4] * scale0[h_4] + _tmem_load_6[h_4] * scale1[h_4];
                    int q_row = qo_begin * 64 + kv_head * group_size_1 + h_4;
                    int out_idx = q_row * 128 + d_idx;
                    if (d_idx == 0) {
                        float natural_lse = -BLACKWELL_MSA_INF;
                        if (final_sum[h_4] > 0.0f) {
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum[h_4]));
                            natural_lse = final_max[h_4] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                        }
                        *(reinterpret_cast<float*>(msa_lse + q_row) + (0)) = natural_lse;
                    }
                    *(reinterpret_cast<__nv_bfloat16*>(O + out_idx) + (0)) = __float2bfloat16_rn(merged * inv_sum[h_4]);
                }
            }
            mbarrier_arrive(decode_done_addr);
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int direct_request_1 = 0;
            direct_request_1 = blockIdx.x;
            int kv_tile_begin_2 = 0;
            int kv_tile_end_2 = 16;
            int num_n_blocks_2 = kv_tile_end_2 - kv_tile_begin_2;
            int num_pairs_2 = num_n_blocks_2 / 2;
            int inst0_stage = 0;
            int first_pv0 = 1;
            int first_pv1 = 1;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            mbarrier_wait(kv_full_addr, 0);
            int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
            int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_s0), "r"(0));
            elect_commit(s_full_addr);
            elect_commit(kv_empty_addr);
            int max_decode_pairs_2 = 8;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            #pragma unroll 1
            for (int pair_2 = 0; pair_2 < max_decode_pairs_2 - 1; pair_2++) {
                if (pair_2 >= num_pairs_2 - 1) {
                    break;
                }
                int s0 = inst0_stage;
                int s1 = (inst0_stage + 1) % 4;
                int s0_next = (inst0_stage + 2) % 4;
                mbarrier_wait(kv_full_addr + (s1) * 8, 0);
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1) * 2048);
                int _mma_b_lo_1 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_tmem_s1), "r"(0));
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (s1) * 8);
                mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 2048);
                int _mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                elect_commit(kv_empty_addr + (s0) * 8);
                mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_1), "r"(tmem_tmem_s0), "r"(0));
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr + (s0_next) * 8);
                mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 2048);
                int _mma_b_lo_4 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                elect_commit(kv_empty_addr + (s1) * 8);
                inst0_stage = s0_next;
                first_pv0 = 0;
                first_pv1 = 0;
            }
            int s0_last = inst0_stage;
            int s1_last = (inst0_stage + 1) % 4;
            mbarrier_wait(kv_full_addr + (s1_last) * 8, 0);
            int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_last) * 2048);
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_0), "r"(tmem_tmem_s1), "r"(0));
            elect_commit(s_full_addr + 8);
            elect_commit(kv_empty_addr + (s1_last) * 8);
            mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
            int _mma_b_lo_6 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
            elect_commit(kv_empty_addr + (s0_last) * 8);
            mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
            _phase_p_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
            int _mma_b_lo_7 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
            elect_commit(kv_empty_addr + (s1_last) * 8);
            elect_commit(o_full_addr);
            unsigned int _phase_decode_done_0 = 0;
            mbarrier_wait(decode_done_addr, _phase_decode_done_0);
            _phase_decode_done_0 ^= 1;
        }
    }
    // ---- Role: producer ----
    if (warp == 13) {
        { // producer_main
            int direct_request_2 = 0;
            int kv_head_1 = 0;
            direct_request_2 = blockIdx.x;
            kv_head_1 = blockIdx.y;
            int q_tile = 0;
            int kv_tile_begin_3 = 0;
            int kv_tile_end_3 = 16;
            int qo_begin_1 = direct_request_2;
            int num_n_blocks_3 = kv_tile_end_3 - kv_tile_begin_3;
            int group_size_2 = 16;
            if (elect_sync()) {
                int q_row_1 = qo_begin_1 * 64 + kv_head_1 * group_size_2 + q_tile * 16;
                mbarrier_arrive_expect_tx(q_full_addr, 4096);
                tma_3d_gmem2smem(smem_qt_addr, (&Q), 0, q_row_1, 0, q_full_addr);
            }
            int native_num_n_blocks = num_n_blocks_3;
            if (elect_sync()) {
                int native_kv_stage = 0;
                int native_kv_phase = 1;
                int native_prefill = ((native_num_n_blocks < 4) ? native_num_n_blocks : 4);
                #pragma unroll
                for (int native_ni = 0; native_ni < 4; native_ni++) {
                    if (native_prefill <= native_ni) {
                        break;
                    }
                    int native_n_block = kv_tile_end_3 - 1 - native_ni;
                    int native_pg0 = 0;
                    int native_pg1 = 0;
                    int msa_token_base = 0;
                    int msa_page_head = 0;
                    int msa_valid_cols = 128;
                    int batch = direct_request_2;
                    int query_in_batch = direct_request_2 - batch;
                    int selected_block = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_n_block];
                    int kv_len = task_kv_head[batch];
                    {
                        kv_len = kv_indptr[batch + 1] - kv_indptr[batch];
                    }
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
                        {
                            int query_position = kv_len - 1 + query_in_batch;
                            int causal_cols = query_position - block_start + 1;
                            if (valid_cols_1 > causal_cols) {
                                valid_cols_1 = causal_cols;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                        }
                    }
                    int token_base = 0;
                    int page_head = 0;
                    {
                        int safe_block = ((selected_block >= 0) ? selected_block : 0);
                        token_base = kv_indptr[batch] + safe_block * 128;
                        page_head = kv_head_1;
                    }
                    msa_token_base = token_base;
                    msa_page_head = page_head;
                    msa_valid_cols = valid_cols_1;
                    smem_page_indices[native_ni] = msa_valid_cols;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_wait(kv_empty_addr + (native_kv_stage) * 8, native_kv_phase);
                    mbarrier_arrive_expect_tx(kv_src_full_addr + (native_kv_stage) * 8, 16384);
                    int native_dst = smem_kv_addr + (unsigned int)(native_kv_stage * 32768);
                    int token0 = msa_token_base;
                    int token1 = msa_token_base + 64;
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_kv_stage * 16384), (&K), 0, token0, msa_page_head, kv_src_full_addr + (native_kv_stage) * 8);
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_kv_stage * 16384) + 8192, (&K), 0, token1, msa_page_head, kv_src_full_addr + (native_kv_stage) * 8);
                    native_kv_stage += 1;
                    if (native_kv_stage == 4) { native_kv_stage = 0; native_kv_phase ^= 1; }
                }
                #pragma unroll 1
                for (int native_ni_1 = 0; native_ni_1 < 16; native_ni_1++) {
                    if (native_num_n_blocks <= native_ni_1) {
                        break;
                    }
                    int native_stage = native_ni_1 % 4;
                    int native_n_block_1 = kv_tile_end_3 - 1 - native_ni_1;
                    int native_pg0_1 = 0;
                    int native_pg1_1 = 0;
                    int msa_token_base_1 = 0;
                    int msa_page_head_1 = 0;
                    int msa_valid_cols_1 = 128;
                    int batch_1 = direct_request_2;
                    int query_in_batch_1 = direct_request_2 - batch_1;
                    int selected_block_1 = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_n_block_1];
                    int kv_len_1 = task_kv_head[batch_1];
                    {
                        kv_len_1 = kv_indptr[batch_1 + 1] - kv_indptr[batch_1];
                    }
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
                        {
                            int query_position_1 = kv_len_1 - 1 + query_in_batch_1;
                            int causal_cols_1 = query_position_1 - block_start_1 + 1;
                            if (valid_cols_2 > causal_cols_1) {
                                valid_cols_2 = causal_cols_1;
                            }
                            if (valid_cols_2 < 0) {
                                valid_cols_2 = 0;
                            }
                        }
                    }
                    int token_base_1 = 0;
                    int page_head_1 = 0;
                    {
                        int safe_block_1 = ((selected_block_1 >= 0) ? selected_block_1 : 0);
                        token_base_1 = kv_indptr[batch_1] + safe_block_1 * 128;
                        page_head_1 = kv_head_1;
                    }
                    msa_token_base_1 = token_base_1;
                    msa_page_head_1 = page_head_1;
                    msa_valid_cols_1 = valid_cols_2;
                    mbarrier_wait(kv_empty_addr + (native_stage) * 8, 0);
                    mbarrier_arrive_expect_tx(kv_src_full_addr + (native_stage) * 8, 16384);
                    int native_dst_1 = smem_kv_addr + (unsigned int)(native_stage * 32768);
                    int token0_1 = msa_token_base_1;
                    int token1_1 = msa_token_base_1 + 64;
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384), (&V), 0, token0_1, msa_page_head_1, kv_src_full_addr + (native_stage) * 8);
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384) + 8192, (&V), 0, token1_1, msa_page_head_1, kv_src_full_addr + (native_stage) * 8);
                    int native_next_ni = native_ni_1 + 4;
                    if (native_next_ni < native_num_n_blocks) {
                        int native_next_n = kv_tile_end_3 - 1 - native_next_ni;
                        int native_npg0 = 0;
                        int native_npg1 = 0;
                        int msa_next_token_base = 0;
                        int msa_next_page_head = 0;
                        int msa_next_valid_cols = 128;
                        int batch_0 = direct_request_2;
                        int query_in_batch_1_1 = direct_request_2 - batch_0;
                        int selected_block_2 = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_next_n];
                        int kv_len_3 = task_kv_head[batch_0];
                        {
                            kv_len_3 = kv_indptr[batch_0 + 1] - kv_indptr[batch_0];
                        }
                        int valid_cols_4 = 0;
                        if (selected_block_2 >= 0) {
                            int block_start_2 = selected_block_2 * 128;
                            valid_cols_4 = kv_len_3 - block_start_2;
                            if (valid_cols_4 > 128) {
                                valid_cols_4 = 128;
                            }
                            if (valid_cols_4 < 0) {
                                valid_cols_4 = 0;
                            }
                            {
                                int query_position_2 = kv_len_3 - 1 + query_in_batch_1_1;
                                int causal_cols_2 = query_position_2 - block_start_2 + 1;
                                if (valid_cols_4 > causal_cols_2) {
                                    valid_cols_4 = causal_cols_2;
                                }
                                if (valid_cols_4 < 0) {
                                    valid_cols_4 = 0;
                                }
                            }
                        }
                        int token_base_5 = 0;
                        int page_head_6 = 0;
                        {
                            int safe_block_2 = ((selected_block_2 >= 0) ? selected_block_2 : 0);
                            token_base_5 = kv_indptr[batch_0] + safe_block_2 * 128;
                            page_head_6 = kv_head_1;
                        }
                        msa_next_token_base = token_base_5;
                        msa_next_page_head = page_head_6;
                        msa_next_valid_cols = valid_cols_4;
                        smem_page_indices[native_next_ni] = msa_next_valid_cols;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (native_stage) * 8, 1);
                        mbarrier_arrive_expect_tx(kv_src_full_addr + (native_stage) * 8, 16384);
                        int native_kdst = smem_kv_addr + (unsigned int)(native_stage * 32768);
                        int token0_7 = msa_next_token_base;
                        int token1_8 = msa_next_token_base + 64;
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384), (&K), 0, token0_7, msa_next_page_head, kv_src_full_addr + (native_stage) * 8);
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384) + 8192, (&K), 0, token1_8, msa_next_page_head, kv_src_full_addr + (native_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: producer_aux ----
    if (warp >= 14 && warp <= 15) {
        { // producer_aux_main
            int num_n_blocks_4 = 16;
            int prefill = ((num_n_blocks_4 < 4) ? num_n_blocks_4 : 4);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                if (prefill <= ni) {
                    break;
                }
                mbarrier_wait(kv_src_full_addr + (ni) * 8, 0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(ni * 16384) - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(ni * 32768) - smem);
                    const int _tid = (int)threadIdx.x - (14) * 32;
                    #pragma unroll 4
                    for (int _off = _tid; _off < 2048; _off += 64) {
                        uint64_t _src64 = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        uint32_t _out_x16x2[4];
                        #pragma unroll
                        for (int _cv = 0; _cv < 4; ++_cv) {
                            uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                            #else
                            uint32_t _f16x2;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                            uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                            uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                            float _f0;
                            float _f1;
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                            #endif
                        }
                        uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                        int _elt = _off * 8;
                        int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                        int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                        int _swz_off = _byte_off ^ ((_row % 8) * 16);
                        *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("barrier.sync 10, 64;" ::: "memory");
                if (warp == 14) {
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (ni) * 8);
                    }
                }
            }
            #pragma unroll 1
            for (int ni_1 = 0; ni_1 < 16; ni_1++) {
                if (num_n_blocks_4 <= ni_1) {
                    break;
                }
                int stage = ni_1 % 4;
                mbarrier_wait(kv_src_full_addr + (stage) * 8, 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(stage * 16384) - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(stage * 32768) - smem);
                    const int _tid = (int)threadIdx.x - (14) * 32;
                    #pragma unroll 4
                    for (int _off = _tid; _off < 2048; _off += 64) {
                        uint64_t _src64 = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        uint32_t _out_x16x2[4];
                        #pragma unroll
                        for (int _cv = 0; _cv < 4; ++_cv) {
                            uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                            #else
                            uint32_t _f16x2;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                            uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                            uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                            float _f0;
                            float _f1;
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                            #endif
                        }
                        uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                        int _elt = _off * 8;
                        int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                        int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                        int _swz_off = _byte_off ^ ((_row % 8) * 16);
                        *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("barrier.sync 10, 64;" ::: "memory");
                if (warp == 14) {
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (stage) * 8);
                    }
                }
                int next_ni = ni_1 + 4;
                if (next_ni < num_n_blocks_4) {
                    mbarrier_wait(kv_src_full_addr + (stage) * 8, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(stage * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(stage * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (14) * 32;
                        #pragma unroll 4
                        for (int _off = _tid; _off < 2048; _off += 64) {
                            uint64_t _src64 = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    asm volatile("barrier.sync 10, 64;" ::: "memory");
                    if (warp == 14) {
                        if (elect_sync()) {
                            mbarrier_arrive(kv_full_addr + (stage) * 8);
                        }
                    }
                }
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

