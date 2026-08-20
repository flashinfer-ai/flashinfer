typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_CB_TMEM_OFFSET 0
#define TMEM_Q_TMEM_OFFSET 256
#define TMEM_INTRA_TMEM_OFFSET 320
#define TMEM_STATE_DELTA_TMEM_OFFSET 384
#define TMEM_INTER_TMEM_OFFSET 448
#define NUM_INPUT_PIPE_STAGES 2
#define NUM_INTRA1_ACC_PIPE_STAGES 2
#define SMEM_SMEM_B_OFF 1024
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_C_OFF 66560
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_X_TMA_OFF 132096
#define SMEM_SMEM_X_TMA_STAGE_BYTES 16384
#define SMEM_SMEM_X_TMA_STRIDE 16384
#define SMEM_SMEM_X_OFF 132096
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_SCALED_B_OFF 164864
#define SMEM_SMEM_SCALED_B_STAGE_BYTES 32768
#define SMEM_SMEM_SCALED_B_STRIDE 32768
#define SMEM_SMEM_STATE_OFF 164864
#define SMEM_SMEM_STATE_STAGE_BYTES 16384
#define SMEM_SMEM_STATE_STRIDE 16384
#define SMEM_SMEM_DELTA_ALL_OFF 197632
#define SMEM_SMEM_DELTA_ALL_STAGE_BYTES 512
#define SMEM_SMEM_DELTA_ALL_STRIDE 512
#define SMEM_SMEM_CUMSUM_ALL_OFF 198144
#define SMEM_SMEM_CUMSUM_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_CUMSUM_ALL_STRIDE 1024
#define SMEM_SMEM_Y_OFF 199168
#define SMEM_SMEM_Y_STAGE_BYTES 16384
#define SMEM_SMEM_Y_STRIDE 16384
#define SMEM_TOTAL 231936
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


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
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

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
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

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
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
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
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


__device__ __forceinline__ void tma_store_5d(
    const void *tmap, int x, int y, int z, int w, int v, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_mamba_ssd_q_tmem_alias_bf16_varlen(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap b_map, const __grid_constant__ CUtensorMap c_map, const __grid_constant__ CUtensorMap out_map, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, __nv_bfloat16* __restrict__ delta_precomputed, float* __restrict__ cumsum_precomputed, float* __restrict__ A, __nv_bfloat16* __restrict__ B_tensor, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ z, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ initial_states, __nv_bfloat16* __restrict__ final_states, __nv_bfloat16* __restrict__ checkpoint_states, int* __restrict__ checkpoint_token_indices, int* __restrict__ checkpoint_state_slots, int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, int* __restrict__ seq_chunk_cumsum, __nv_bfloat16* __restrict__ out_native, int nheads, int ngroups, int batch, int seqlen, int nchunks, int sequence_count, int num_logical_chunks, int mode_varlen, int has_seq_chunk_cumsum, int seq_idx_int64, int D_mode, int has_z, int has_initial, int dt_softplus, float dt_min, float dt_max, int write_final_states, int checkpoint_state_count)
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
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_b_addr = smem + 1024;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_c_addr = smem + 66560;
    __nv_bfloat16* smem_x_tma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_x_tma_addr = smem + 132096;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_x_addr = smem + 132096;
    __nv_bfloat16* smem_scaled_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_scaled_b_addr = smem + 164864;
    __nv_bfloat16* smem_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_state_addr = smem + 164864;
    __nv_bfloat16* smem_delta_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_delta_all_addr = smem + 197632;
    float* smem_cumsum_all = reinterpret_cast<float*>(smem_raw + 198144);
    const int smem_cumsum_all_addr = smem + 198144;
    __nv_bfloat16* smem_y = reinterpret_cast<__nv_bfloat16*>(smem_raw + 199168);
    const int smem_y_addr = smem + 199168;

    // Mbarrier init (17 groups, 25 barriers)
    // Mbarriers at smem_raw[0..200)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'input_pipe' ---
            // bc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // bc_empty: 2 barriers, init_count=3
            mbarrier_init(smem + 16, 3);
            mbarrier_init(smem + 24, 3);
            // x_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // x_empty: 2 barriers, init_count=3
            mbarrier_init(smem + 48, 3);
            mbarrier_init(smem + 56, 3);
            // aux_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // aux_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // --- pipeline 'intra1_acc_pipe' ---
            // cb_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // cb_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // scaled_b_full: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // state_operand_full: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // state_read_released: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // state_delta_full: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // intra_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // inter_full: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // outputs_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 200);
    if (warp == 12) {
        int _tmem_hold = smem + 200;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define bc_full_addr (mbar_base + 0)
    #define bc_empty_addr (mbar_base + 16)
    #define x_full_addr (mbar_base + 32)
    #define x_empty_addr (mbar_base + 48)
    #define aux_full_addr (mbar_base + 64)
    #define aux_empty_addr (mbar_base + 80)
    #define cb_full_addr (mbar_base + 96)
    #define cb_empty_addr (mbar_base + 112)
    #define q_full_addr (mbar_base + 128)
    #define q_empty_addr (mbar_base + 136)
    #define scaled_b_full_addr (mbar_base + 144)
    #define state_operand_full_addr (mbar_base + 152)
    #define state_read_released_addr (mbar_base + 160)
    #define state_delta_full_addr (mbar_base + 168)
    #define intra_full_addr (mbar_base + 176)
    #define inter_full_addr (mbar_base + 184)
    #define outputs_empty_addr (mbar_base + 192)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_cb_tmem = taddr;
    const int tmem_q_tmem = taddr + 256;
    const int tmem_intra_tmem = taddr + 320;
    const int tmem_state_delta_tmem = taddr + 384;
    const int tmem_inter_tmem = taddr + 448;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: mma_inter ----
    if (warp == 0) {
        { // mma_inter_main
            unsigned int input_stage = 0;
            unsigned int _phase_bc_full = 0;
            unsigned int _phase_state_operand_full_0 = 0;
            unsigned int _phase_outputs_empty_0 = 1;
            unsigned int _phase_scaled_b_full_0 = 0;
            unsigned int _phase_x_full = 0;
            #pragma unroll 1
            for (unsigned int work = bid; work < sequence_count * nheads; work += num_bids) {
                int sequence = work / (unsigned int)nheads;
                int first_logical = sequence * nchunks;
                int logical_end = first_logical + nchunks;
                first_logical = num_logical_chunks;
                logical_end = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical = seq_chunk_cumsum[sequence];
                    logical_end = seq_chunk_cumsum[sequence + 1];
                } else {
                    #pragma unroll 1
                    for (int probe = 0; probe < num_logical_chunks; probe++) {
                        int probe_start = chunk_indices[probe] * 128 + chunk_offsets[probe];
                        int probe_sequence = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence = (int)seq_idx_i64[probe_start];
                        } else {
                            probe_sequence = seq_idx_i32[probe_start];
                        }
                        if (probe_sequence == sequence) {
                            if (first_logical == num_logical_chunks) {
                                first_logical = probe;
                            }
                            logical_end = probe + 1;
                        }
                    }
                }
                int _uniform_6 = make_warp_uniform(first_logical);
                first_logical = _uniform_6;
                int _uniform_7 = make_warp_uniform(logical_end);
                logical_end = _uniform_7;
                #pragma unroll 1
                for (int logical = first_logical; logical < logical_end; logical++) {
                    int physical_chunk = logical - sequence * nchunks;
                    physical_chunk = chunk_indices[logical];
                    mbarrier_wait(bc_full_addr + (input_stage) * 8, _phase_bc_full);
                    mbarrier_wait(state_operand_full_addr, _phase_state_operand_full_0);
                    _phase_state_operand_full_0 ^= 1;
                    mbarrier_wait(outputs_empty_addr, _phase_outputs_empty_0);
                    _phase_outputs_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform((((smem_c_addr) >> 4) & 0x3FFF) + (input_stage) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_state_addr) >> 4) & 0x3FFF) + (0) * 1024);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_inter_tmem), "r"(0));
                    elect_commit(inter_full_addr);
                    elect_commit(state_read_released_addr);
                    mbarrier_wait(scaled_b_full_addr, _phase_scaled_b_full_0);
                    _phase_scaled_b_full_0 ^= 1;
                    mbarrier_wait(x_full_addr + (input_stage) * 8, _phase_x_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_3 = make_warp_uniform((((smem_scaled_b_addr) >> 4) & 0x3FFF) + (0) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x2000000) + (input_stage) * 1024);
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
                    "mov.b32 id, 135333008;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_state_delta_tmem), "r"(0));
                    elect_commit(state_delta_full_addr);
                    elect_commit(bc_empty_addr + (input_stage) * 8);
                    elect_commit(x_empty_addr + (input_stage) * 8);
                    input_stage += 1;
                    if (input_stage == 2) { input_stage = 0; _phase_bc_full ^= 1; _phase_x_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma_intra ----
    if (warp == 1) {
        { // mma_intra_main
            unsigned int input_stage_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_bc_full_1 = 0;
            unsigned int _phase_cb_empty = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_outputs_empty_0_1 = 1;
            unsigned int _phase_x_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_1 = bid; work_1 < sequence_count * nheads; work_1 += num_bids) {
                int sequence_1 = work_1 / (unsigned int)nheads;
                int first_logical_1 = sequence_1 * nchunks;
                int logical_end_1 = first_logical_1 + nchunks;
                first_logical_1 = num_logical_chunks;
                logical_end_1 = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical_1 = seq_chunk_cumsum[sequence_1];
                    logical_end_1 = seq_chunk_cumsum[sequence_1 + 1];
                } else {
                    #pragma unroll 1
                    for (int probe_1 = 0; probe_1 < num_logical_chunks; probe_1++) {
                        int probe_start_1 = chunk_indices[probe_1] * 128 + chunk_offsets[probe_1];
                        int probe_sequence_1 = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence_1 = (int)seq_idx_i64[probe_start_1];
                        } else {
                            probe_sequence_1 = seq_idx_i32[probe_start_1];
                        }
                        if (probe_sequence_1 == sequence_1) {
                            if (first_logical_1 == num_logical_chunks) {
                                first_logical_1 = probe_1;
                            }
                            logical_end_1 = probe_1 + 1;
                        }
                    }
                }
                int _uniform_4 = make_warp_uniform(first_logical_1);
                first_logical_1 = _uniform_4;
                int _uniform_5 = make_warp_uniform(logical_end_1);
                logical_end_1 = _uniform_5;
                #pragma unroll 1
                for (int logical_1 = first_logical_1; logical_1 < logical_end_1; logical_1++) {
                    int physical_chunk_1 = logical_1 - sequence_1 * nchunks;
                    physical_chunk_1 = chunk_indices[logical_1];
                    mbarrier_wait(bc_full_addr + (input_stage_1) * 8, _phase_bc_full_1);
                    mbarrier_wait(cb_empty_addr + (acc_stage) * 8, _phase_cb_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_c_addr) >> 4) & 0x3FFF) + (input_stage_1) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (input_stage_1) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_cb_tmem + (acc_stage * 128))), "r"(0));
                    elect_commit(cb_full_addr + (acc_stage) * 8);
                    elect_commit(bc_empty_addr + (input_stage_1) * 8);
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(outputs_empty_addr, _phase_outputs_empty_0_1);
                    _phase_outputs_empty_0_1 ^= 1;
                    mbarrier_wait(x_full_addr + (input_stage_1) * 8, _phase_x_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_1 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x4000000) + (input_stage_1) * 1024);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135333008;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_intra_tmem), "r"(_mma_b_lo_1), "r"(tmem_q_tmem), "r"(0));
                    elect_commit(intra_full_addr);
                    elect_commit(q_empty_addr);
                    elect_commit(x_empty_addr + (input_stage_1) * 8);
                    input_stage_1 += 1;
                    if (input_stage_1 == 2) { input_stage_1 = 0; _phase_bc_full_1 ^= 1; _phase_x_full_1 ^= 1; }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_cb_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_bc ----
    if (warp == 2) {
        { // load_bc_main
            unsigned int input_stage_2 = 0;
            unsigned int _phase_bc_empty = 1;
            #pragma unroll 1
            for (unsigned int work_2 = bid; work_2 < sequence_count * nheads; work_2 += num_bids) {
                int sequence_2 = work_2 / (unsigned int)nheads;
                int head = work_2 % (unsigned int)nheads;
                int group = head * ngroups / nheads;
                int first_logical_2 = sequence_2 * nchunks;
                int logical_end_2 = first_logical_2 + nchunks;
                first_logical_2 = num_logical_chunks;
                logical_end_2 = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical_2 = seq_chunk_cumsum[sequence_2];
                    logical_end_2 = seq_chunk_cumsum[sequence_2 + 1];
                } else {
                    #pragma unroll 1
                    for (int probe_2 = 0; probe_2 < num_logical_chunks; probe_2++) {
                        int probe_start_2 = chunk_indices[probe_2] * 128 + chunk_offsets[probe_2];
                        int probe_sequence_2 = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence_2 = (int)seq_idx_i64[probe_start_2];
                        } else {
                            probe_sequence_2 = seq_idx_i32[probe_start_2];
                        }
                        if (probe_sequence_2 == sequence_2) {
                            if (first_logical_2 == num_logical_chunks) {
                                first_logical_2 = probe_2;
                            }
                            logical_end_2 = probe_2 + 1;
                        }
                    }
                }
                int _uniform_0 = make_warp_uniform(first_logical_2);
                first_logical_2 = _uniform_0;
                int _uniform_1 = make_warp_uniform(logical_end_2);
                logical_end_2 = _uniform_1;
                #pragma unroll 1
                for (int logical_2 = first_logical_2; logical_2 < logical_end_2; logical_2++) {
                    int physical_chunk_2 = logical_2 - sequence_2 * nchunks;
                    int segment_offset = 0;
                    int segment_limit = 128;
                    physical_chunk_2 = chunk_indices[logical_2];
                    segment_offset = chunk_offsets[logical_2];
                    if (logical_2 + 1 < num_logical_chunks) {
                        int next_chunk = chunk_indices[logical_2 + 1];
                        if (next_chunk == physical_chunk_2) {
                            segment_limit = chunk_offsets[logical_2 + 1];
                        }
                    }
                    mbarrier_wait(bc_empty_addr + (input_stage_2) * 8, _phase_bc_empty);
                    int physical_batch = 0;
                    int token_in_batch = physical_chunk_2 * 128;
                    if (elect_sync()) {
                        tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768, (&b_map), 0, 0, group, token_in_batch, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768 + 16384, (&b_map), 0, 1, group, token_in_batch, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768, (&c_map), 0, 0, group, token_in_batch, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768 + 16384, (&c_map), 0, 1, group, token_in_batch, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        mbarrier_arrive_expect_tx(bc_full_addr + (input_stage_2) * 8, 65536);
                    }
                    input_stage_2 += 1;
                    if (input_stage_2 == 2) { input_stage_2 = 0; _phase_bc_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_x_dt ----
    if (warp == 3) {
        { // load_x_dt_main
            unsigned int input_stage_3 = 0;
            unsigned int _phase_aux_empty = 1;
            unsigned int _phase_x_empty = 1;
            #pragma unroll 1
            for (unsigned int work_3 = bid; work_3 < sequence_count * nheads; work_3 += num_bids) {
                int sequence_3 = work_3 / (unsigned int)nheads;
                int head_1 = work_3 % (unsigned int)nheads;
                int first_logical_3 = sequence_3 * nchunks;
                int logical_end_3 = first_logical_3 + nchunks;
                first_logical_3 = num_logical_chunks;
                logical_end_3 = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical_3 = seq_chunk_cumsum[sequence_3];
                    logical_end_3 = seq_chunk_cumsum[sequence_3 + 1];
                } else {
                    #pragma unroll 1
                    for (int probe_3 = 0; probe_3 < num_logical_chunks; probe_3++) {
                        int probe_start_3 = chunk_indices[probe_3] * 128 + chunk_offsets[probe_3];
                        int probe_sequence_3 = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence_3 = (int)seq_idx_i64[probe_start_3];
                        } else {
                            probe_sequence_3 = seq_idx_i32[probe_start_3];
                        }
                        if (probe_sequence_3 == sequence_3) {
                            if (first_logical_3 == num_logical_chunks) {
                                first_logical_3 = probe_3;
                            }
                            logical_end_3 = probe_3 + 1;
                        }
                    }
                }
                int _uniform_2 = make_warp_uniform(first_logical_3);
                first_logical_3 = _uniform_2;
                int _uniform_3 = make_warp_uniform(logical_end_3);
                logical_end_3 = _uniform_3;
                #pragma unroll 1
                for (int logical_3 = first_logical_3; logical_3 < logical_end_3; logical_3++) {
                    int physical_chunk_3 = logical_3 - sequence_3 * nchunks;
                    int segment_offset_1 = 0;
                    int segment_limit_1 = 128;
                    physical_chunk_3 = chunk_indices[logical_3];
                    segment_offset_1 = chunk_offsets[logical_3];
                    if (logical_3 + 1 < num_logical_chunks) {
                        int next_chunk_1 = chunk_indices[logical_3 + 1];
                        if (next_chunk_1 == physical_chunk_3) {
                            segment_limit_1 = chunk_offsets[logical_3 + 1];
                        }
                    }
                    mbarrier_wait(aux_empty_addr + (input_stage_3) * 8, _phase_aux_empty);
                    mbarrier_wait(x_empty_addr + (input_stage_3) * 8, _phase_x_empty);
                    int physical_batch_1 = 0;
                    int token_in_batch_1 = physical_chunk_3 * 128;
                    int physical_start = physical_batch_1 * seqlen + token_in_batch_1;
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_x_tma_addr + input_stage_3 * 16384, (&x_map), 0, head_1, token_in_batch_1, physical_batch_1, x_full_addr + (input_stage_3) * 8);
                        mbarrier_arrive_expect_tx(x_full_addr + (input_stage_3) * 8, 16384);
                    }
                    int tile = logical_3 * nheads + head_1;
                    int group_start = lane * 4;
                    #pragma unroll
                    for (int local = 0; local < 4; local++) {
                        int physical_token = group_start + local;
                        int factor_token = physical_token - segment_offset_1;
                        float cumsum_value = 0.0f;
                        __nv_bfloat16 delta_value = 0.0f;
                        if (physical_token >= segment_offset_1 && physical_token < segment_limit_1) {
                            cumsum_value = cumsum_precomputed[tile * 128 + factor_token];
                            delta_value = delta_precomputed[tile * 128 + factor_token];
                        }
                        smem_cumsum_all[input_stage_3 * 128 + (unsigned int)physical_token] = cumsum_value;
                        smem_delta_all[input_stage_3 * 128 + (unsigned int)physical_token] = delta_value;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(aux_full_addr + (input_stage_3) * 8);
                    }
                    input_stage_3 += 1;
                    if (input_stage_3 == 2) { input_stage_3 = 0; _phase_aux_empty ^= 1; _phase_x_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: pre_inter ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // pre_inter_main
            unsigned int input_stage_4 = 0;
            int taddr_0 = taddr;
            float state_values[64];
            unsigned int _phase_aux_full = 0;
            unsigned int _phase_bc_full_2 = 0;
            unsigned int _phase_state_read_released_0 = 0;
            unsigned int _phase_state_delta_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_4 = bid; work_4 < sequence_count * nheads; work_4 += num_bids) {
                int sequence_4 = work_4 / (unsigned int)nheads;
                int head_2 = work_4 % (unsigned int)nheads;
                int state_head_base = (sequence_4 * nheads + head_2) * 64 * 128;
                #pragma unroll
                for (int dim_chunk = 0; dim_chunk < 2; dim_chunk++) {
                    #pragma unroll
                    for (int row_half = 0; row_half < 2; row_half++) {
                        int state_fragment_offset = (dim_chunk * 2 + row_half) * 16;
                        int state_row_origin = warp % 4 * 32 + row_half * 16;
                        #pragma unroll
                        for (int state_local = 0; state_local < 16; state_local++) {
                            int state_reg = state_local % 4;
                            int state_repeat = state_local / 4;
                            int state_row = state_row_origin + lane / 4 + state_reg / 2 * 8;
                            int dim = dim_chunk * 32 + state_repeat * 8 + lane % 4 * 2 + state_reg % 2;
                            float initial_value = 0.0f;
                            if (has_initial != 0) {
                                initial_value = (float)initial_states[state_head_base + dim * 128 + state_row];
                            }
                            state_values[state_fragment_offset + state_local] = initial_value;
                        }
                    }
                }
                int first_logical_4 = sequence_4 * nchunks;
                int logical_end_4 = first_logical_4 + nchunks;
                first_logical_4 = num_logical_chunks;
                logical_end_4 = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical_4 = seq_chunk_cumsum[sequence_4];
                    logical_end_4 = seq_chunk_cumsum[sequence_4 + 1];
                } else {
                    #pragma unroll 1
                    for (int probe_4 = 0; probe_4 < num_logical_chunks; probe_4++) {
                        int probe_start_4 = chunk_indices[probe_4] * 128 + chunk_offsets[probe_4];
                        int probe_sequence_4 = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence_4 = (int)seq_idx_i64[probe_start_4];
                        } else {
                            probe_sequence_4 = seq_idx_i32[probe_start_4];
                        }
                        if (probe_sequence_4 == sequence_4) {
                            if (first_logical_4 == num_logical_chunks) {
                                first_logical_4 = probe_4;
                            }
                            logical_end_4 = probe_4 + 1;
                        }
                    }
                }
                int _uniform_10 = make_warp_uniform(first_logical_4);
                first_logical_4 = _uniform_10;
                int _uniform_11 = make_warp_uniform(logical_end_4);
                logical_end_4 = _uniform_11;
                #pragma unroll 1
                for (int logical_4 = first_logical_4; logical_4 < logical_end_4; logical_4++) {
                    int physical_chunk_4 = logical_4 - sequence_4 * nchunks;
                    int segment_offset_2 = 0;
                    int segment_limit_2 = 128;
                    physical_chunk_4 = chunk_indices[logical_4];
                    segment_offset_2 = chunk_offsets[logical_4];
                    if (logical_4 + 1 < num_logical_chunks) {
                        int next_chunk_2 = chunk_indices[logical_4 + 1];
                        if (next_chunk_2 == physical_chunk_4) {
                            segment_limit_2 = chunk_offsets[logical_4 + 1];
                        }
                    }
                    {
                        #pragma unroll
                        for (int dim_chunk_1 = 0; dim_chunk_1 < 2; dim_chunk_1++) {
                            #pragma unroll
                            for (int row_half_1 = 0; row_half_1 < 2; row_half_1++) {
                                int state_fragment_offset_1 = (dim_chunk_1 * 2 + row_half_1) * 16;
                                float state_fragment[16];
                                #pragma unroll
                                for (int state_local_1 = 0; state_local_1 < 16; state_local_1++) {
                                    state_fragment[state_local_1] = state_values[state_fragment_offset_1 + state_local_1];
                                }
                                uint32_t state_fragment_bf16[8];
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_fragment[_lp*2 + 0], state_fragment[_lp*2+1 + 0]));
                                    state_fragment_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int state_row_origin_1 = warp % 4 * 32 + row_half_1 * 16;
                                #pragma unroll
                                for (int stsm_repeat = 0; stsm_repeat < 2; stsm_repeat++) {
                                    int stsm_matrix = lane / 8;
                                    int stsm_row = dim_chunk_1 * 32 + stsm_repeat * 16 + stsm_matrix / 2 * 8 + lane % 8;
                                    int stsm_col = state_row_origin_1 + stsm_matrix % 2 * 8;
                                    int packed_base = stsm_repeat * 4;
                                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((smem_state_addr + (unsigned int)(stsm_col / 64 * 8192 + stsm_row * 128 + stsm_col % 64 * 2 ^ (stsm_col / 64 * 8192 + stsm_row * 128 + stsm_col % 64 * 2 >> 7 & 7) << 4)));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base])), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base + 3]))
                                        : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                mbarrier_arrive(state_operand_full_addr);
                            }
                        }
                        mbarrier_wait(aux_full_addr + (input_stage_4) * 8, _phase_aux_full);
                        mbarrier_wait(bc_full_addr + (input_stage_4) * 8, _phase_bc_full_2);
                        float last_cumsum = smem_cumsum_all[input_stage_4 * 128 + (unsigned int)(segment_limit_2 - 1)];
                        mbarrier_wait(state_read_released_addr, _phase_state_read_released_0);
                        _phase_state_read_released_0 ^= 1;
                        int b_state_base = (warp % 4 * 4 + lane / 8) * 8;
                        int b_col_lane = lane % 8;
                        #pragma unroll 1
                        for (int b_col_iter = 0; b_col_iter < 16; b_col_iter++) {
                            int col = b_col_iter * 8 + b_col_lane;
                            float scaled_b_values[8];
                            #pragma unroll
                            for (int b_local = 0; b_local < 8; b_local++) {
                                scaled_b_values[b_local] = 0.0f;
                            }
                            if (col >= segment_offset_2 && col < segment_limit_2) {
                                unsigned int b_packed[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&b_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 3]))
                                    : "r"((smem_b_addr + input_stage_4 * 32768 + (unsigned int)(b_state_base / 64 * 16384 + col * 128 + b_state_base % 64 * 2 ^ (b_state_base / 64 * 16384 + col * 128 + b_state_base % 64 * 2 >> 7 & 7) << 4))));
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&scaled_b_values[_pair * 2])[0]), "=f"((&scaled_b_values[_pair * 2])[1])
                                        : "r"(b_packed[_pair]));
                                }
                                float _exp2_1 = approx_exp2((last_cumsum - smem_cumsum_all[input_stage_4 * 128 + (unsigned int)col]) * 1.4426950408889634f);
                                float b_scale = _exp2_1;
                                float _cvt_f32_0 = __bfloat162float(smem_delta_all[input_stage_4 * 128 + (unsigned int)col]);
                                b_scale *= _cvt_f32_0;
                                const float2 _scale2_1 = {b_scale, b_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(scaled_b_values)[_ls], _scale2_1);
                            }
                            #pragma unroll
                            for (int b_local_1 = 0; b_local_1 < 8; b_local_1++) {
                                {
                                    __nv_bfloat16 _bval_2 = __float2bfloat16_rn(scaled_b_values[b_local_1]);
                                    uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                                    uint32_t _addr_2 = static_cast<uint32_t>((smem_scaled_b_addr + (unsigned int)(col / 64 * 16384 + (b_state_base + b_local_1) * 128 + col % 64 * 2 ^ (col / 64 * 16384 + (b_state_base + b_local_1) * 128 + col % 64 * 2 >> 7 & 7) << 4)));
                                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                mbarrier_arrive(scaled_b_full_addr);
                                mbarrier_arrive(bc_empty_addr + (input_stage_4) * 8);
                            }
                        }
                        mbarrier_wait(state_delta_full_addr, _phase_state_delta_full_0);
                        _phase_state_delta_full_0 ^= 1;
                        float segment_base = 0.0f;
                        if (segment_offset_2 > 0) {
                            segment_base = smem_cumsum_all[input_stage_4 * 128 + (unsigned int)(segment_offset_2 - 1)];
                        }
                        float _exp2_2 = approx_exp2((last_cumsum - segment_base) * 1.4426950408889634f);
                        float last_decay = _exp2_2;
                        const float2 _scale2_3 = {last_decay, last_decay};
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(state_values)[_ls], _scale2_3);
                        #pragma unroll
                        for (int dim_chunk_2 = 0; dim_chunk_2 < 2; dim_chunk_2++) {
                            #pragma unroll
                            for (int row_half_2 = 0; row_half_2 < 2; row_half_2++) {
                                int state_row_origin_2 = warp % 4 * 32 + row_half_2 * 16;
                                float _tmem_load_1[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                    : "r"(taddr_0 + (state_row_origin_2 << 16) + 384 + dim_chunk_2 * 32)
                                    : "memory");
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                int state_fragment_offset_2 = (dim_chunk_2 * 2 + row_half_2) * 16;
                                #pragma unroll
                                for (int state_local_2 = 0; state_local_2 < 16; state_local_2++) {
                                    int state_index = state_fragment_offset_2 + state_local_2;
                                    state_values[state_index] = state_values[state_index] + _tmem_load_1[state_local_2];
                                }
                            }
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                mbarrier_arrive(aux_empty_addr + (input_stage_4) * 8);
                            }
                        }
                        input_stage_4 += 1;
                        if (input_stage_4 == 2) { input_stage_4 = 0; _phase_aux_full ^= 1; _phase_bc_full_2 ^= 1; }
                    }
                    if (checkpoint_state_count > 0) {
                        int checkpoint_token = checkpoint_token_indices[sequence_4];
                        int segment_end = physical_chunk_4 * 128 + segment_limit_2;
                        if (checkpoint_token == segment_end) {
                            int checkpoint_slot = checkpoint_state_slots[sequence_4];
                            if (checkpoint_slot >= 0 && checkpoint_slot < checkpoint_state_count) {
                                int checkpoint_head_base = (checkpoint_slot * nheads + head_2) * 64 * 128;
                                #pragma unroll
                                for (int dim_chunk_3 = 0; dim_chunk_3 < 2; dim_chunk_3++) {
                                    #pragma unroll
                                    for (int row_half_3 = 0; row_half_3 < 2; row_half_3++) {
                                        int state_fragment_offset_3 = (dim_chunk_3 * 2 + row_half_3) * 16;
                                        int state_row_origin_3 = warp % 4 * 32 + row_half_3 * 16;
                                        #pragma unroll
                                        for (int state_local_3 = 0; state_local_3 < 16; state_local_3++) {
                                            int state_reg_1 = state_local_3 % 4;
                                            int state_repeat_1 = state_local_3 / 4;
                                            int state_row_1 = state_row_origin_3 + lane / 4 + state_reg_1 / 2 * 8;
                                            int dim_1 = dim_chunk_3 * 32 + state_repeat_1 * 8 + lane % 4 * 2 + state_reg_1 % 2;
                                            checkpoint_states[checkpoint_head_base + dim_1 * 128 + state_row_1] = state_values[state_fragment_offset_3 + state_local_3];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (write_final_states != 0) {
                    #pragma unroll
                    for (int dim_chunk_4 = 0; dim_chunk_4 < 2; dim_chunk_4++) {
                        #pragma unroll
                        for (int row_half_4 = 0; row_half_4 < 2; row_half_4++) {
                            int state_fragment_offset_4 = (dim_chunk_4 * 2 + row_half_4) * 16;
                            int state_row_origin_4 = warp % 4 * 32 + row_half_4 * 16;
                            #pragma unroll
                            for (int state_local_4 = 0; state_local_4 < 16; state_local_4++) {
                                int state_reg_2 = state_local_4 % 4;
                                int state_repeat_2 = state_local_4 / 4;
                                int state_row_2 = state_row_origin_4 + lane / 4 + state_reg_2 / 2 * 8;
                                int dim_2 = dim_chunk_4 * 32 + state_repeat_2 * 8 + lane % 4 * 2 + state_reg_2 % 2;
                                final_states[state_head_base + dim_2 * 128 + state_row_2] = state_values[state_fragment_offset_4 + state_local_4];
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: pre_intra ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // pre_intra_main
            unsigned int input_stage_5 = 0;
            unsigned int acc_stage_1 = 0;
            int row = warp % 4 * 32 + lane;
            int row_tmem_base = warp % 4 * 32 << 16;
            int taddr_0_1 = taddr;
            unsigned int _phase_cb_full = 0;
            unsigned int _phase_aux_full_1 = 0;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int work_5 = bid; work_5 < sequence_count * nheads; work_5 += num_bids) {
                int sequence_5 = work_5 / (unsigned int)nheads;
                int first_logical_5 = sequence_5 * nchunks;
                int logical_end_5 = first_logical_5 + nchunks;
                first_logical_5 = num_logical_chunks;
                logical_end_5 = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical_5 = seq_chunk_cumsum[sequence_5];
                    logical_end_5 = seq_chunk_cumsum[sequence_5 + 1];
                } else {
                    #pragma unroll 1
                    for (int probe_5 = 0; probe_5 < num_logical_chunks; probe_5++) {
                        int probe_start_5 = chunk_indices[probe_5] * 128 + chunk_offsets[probe_5];
                        int probe_sequence_5 = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence_5 = (int)seq_idx_i64[probe_start_5];
                        } else {
                            probe_sequence_5 = seq_idx_i32[probe_start_5];
                        }
                        if (probe_sequence_5 == sequence_5) {
                            if (first_logical_5 == num_logical_chunks) {
                                first_logical_5 = probe_5;
                            }
                            logical_end_5 = probe_5 + 1;
                        }
                    }
                }
                int _uniform_8 = make_warp_uniform(first_logical_5);
                first_logical_5 = _uniform_8;
                int _uniform_9 = make_warp_uniform(logical_end_5);
                logical_end_5 = _uniform_9;
                #pragma unroll 1
                for (int logical_5 = first_logical_5; logical_5 < logical_end_5; logical_5++) {
                    int physical_chunk_5 = logical_5 - sequence_5 * nchunks;
                    int segment_offset_3 = 0;
                    int segment_limit_3 = 128;
                    physical_chunk_5 = chunk_indices[logical_5];
                    segment_offset_3 = chunk_offsets[logical_5];
                    if (logical_5 + 1 < num_logical_chunks) {
                        int next_chunk_3 = chunk_indices[logical_5 + 1];
                        if (next_chunk_3 == physical_chunk_5) {
                            segment_limit_3 = chunk_offsets[logical_5 + 1];
                        }
                    }
                    {
                        mbarrier_wait(cb_full_addr + (acc_stage_1) * 8, _phase_cb_full);
                        mbarrier_wait(aux_full_addr + (input_stage_5) * 8, _phase_aux_full_1);
                        mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                        _phase_q_empty_0 ^= 1;
                        float row_cumsum = smem_cumsum_all[input_stage_5 * 128 + (unsigned int)row];
                        float segment_base_1 = 0.0f;
                        if (segment_offset_3 > 0) {
                            segment_base_1 = smem_cumsum_all[input_stage_5 * 128 + (unsigned int)(segment_offset_3 - 1)];
                        }
                        #pragma unroll
                        for (int col_chunk = 0; col_chunk < 4; col_chunk++) {
                            float _tmem_load_0[32];
                            tmem_ld_x32(&_tmem_load_0[0], (unsigned int)(taddr_0_1 + row_tmem_base) + acc_stage_1 * 128 + (unsigned int)(col_chunk * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float q_values[32];
                            #pragma unroll
                            for (int local_col = 0; local_col < 32; local_col++) {
                                int col_1 = col_chunk * 32 + local_col;
                                float q_value = 0.0f;
                                if (row >= segment_offset_3 && row < segment_limit_3 && col_1 >= segment_offset_3 && col_1 <= row) {
                                    float _exp2_0 = approx_exp2((row_cumsum - smem_cumsum_all[input_stage_5 * 128 + (unsigned int)col_1]) * 1.4426950408889634f);
                                    float decay = _exp2_0;
                                    q_value = decay * (float)smem_delta_all[input_stage_5 * 128 + (unsigned int)col_1];
                                    q_value *= _tmem_load_0[local_col];
                                }
                                q_values[local_col] = q_value;
                            }
                            uint32_t q_values_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_values[_lp*2 + 0], q_values[_lp*2+1 + 0]));
                                q_values_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr_0_1 + row_tmem_base + 256 + col_chunk * 16), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&q_values_bf16[15]))
                                : "memory");
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (warp == 8) {
                            if (elect_sync()) {
                                mbarrier_arrive(q_full_addr);
                                mbarrier_arrive(cb_empty_addr + (acc_stage_1) * 8);
                            }
                        }
                        input_stage_5 += 1;
                        if (input_stage_5 == 2) { input_stage_5 = 0; _phase_aux_full_1 ^= 1; }
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_cb_full ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // epilogue_main
            unsigned int input_stage_6 = 0;
            unsigned int output_stage = 0;
            int output_issued = 0;
            int row_1 = warp % 4 * 32 + lane;
            int row_tmem_base_1 = warp % 4 * 32 << 16;
            int taddr_0_2 = taddr;
            unsigned int _phase_intra_full_0 = 0;
            unsigned int _phase_inter_full_0 = 0;
            unsigned int _phase_aux_full_2 = 0;
            unsigned int _phase_x_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_6 = bid; work_6 < sequence_count * nheads; work_6 += num_bids) {
                int sequence_6 = work_6 / (unsigned int)nheads;
                int head_3 = work_6 % (unsigned int)nheads;
                float d_head_value = 0.0f;
                if (D_mode == 1) {
                    d_head_value = (float)D[head_3];
                }
                int first_logical_6 = sequence_6 * nchunks;
                int logical_end_6 = first_logical_6 + nchunks;
                first_logical_6 = num_logical_chunks;
                logical_end_6 = num_logical_chunks;
                if (has_seq_chunk_cumsum != 0) {
                    first_logical_6 = seq_chunk_cumsum[sequence_6];
                    logical_end_6 = seq_chunk_cumsum[sequence_6 + 1];
                } else {
                    #pragma unroll 1
                    for (int probe_6 = 0; probe_6 < num_logical_chunks; probe_6++) {
                        int probe_start_6 = chunk_indices[probe_6] * 128 + chunk_offsets[probe_6];
                        int probe_sequence_6 = 0;
                        if (seq_idx_int64 != 0) {
                            probe_sequence_6 = (int)seq_idx_i64[probe_start_6];
                        } else {
                            probe_sequence_6 = seq_idx_i32[probe_start_6];
                        }
                        if (probe_sequence_6 == sequence_6) {
                            if (first_logical_6 == num_logical_chunks) {
                                first_logical_6 = probe_6;
                            }
                            logical_end_6 = probe_6 + 1;
                        }
                    }
                }
                int _uniform_12 = make_warp_uniform(first_logical_6);
                first_logical_6 = _uniform_12;
                int _uniform_13 = make_warp_uniform(logical_end_6);
                logical_end_6 = _uniform_13;
                #pragma unroll 1
                for (int logical_6 = first_logical_6; logical_6 < logical_end_6; logical_6++) {
                    int physical_chunk_6 = logical_6 - sequence_6 * nchunks;
                    int segment_offset_4 = 0;
                    int segment_limit_4 = 128;
                    physical_chunk_6 = chunk_indices[logical_6];
                    segment_offset_4 = chunk_offsets[logical_6];
                    if (logical_6 + 1 < num_logical_chunks) {
                        int next_chunk_4 = chunk_indices[logical_6 + 1];
                        if (next_chunk_4 == physical_chunk_6) {
                            segment_limit_4 = chunk_offsets[logical_6 + 1];
                        }
                    }
                    {
                        mbarrier_wait(intra_full_addr, _phase_intra_full_0);
                        _phase_intra_full_0 ^= 1;
                        mbarrier_wait(inter_full_addr, _phase_inter_full_0);
                        _phase_inter_full_0 ^= 1;
                        mbarrier_wait(aux_full_addr + (input_stage_6) * 8, _phase_aux_full_2);
                        mbarrier_wait(x_full_addr + (input_stage_6) * 8, _phase_x_full_2);
                        int physical_batch_2 = 0;
                        float segment_base_2 = 0.0f;
                        if (segment_offset_4 > 0) {
                            segment_base_2 = smem_cumsum_all[input_stage_6 * 128 + (unsigned int)(segment_offset_4 - 1)];
                        }
                        if (!1 && output_issued >= 2) {
                            int warp_id_in_role = (warp - 12);
                            if (warp_id_in_role == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
                        }
                        #pragma unroll
                        for (int dim_chunk_5 = 0; dim_chunk_5 < 2; dim_chunk_5++) {
                            float _tmem_load_2[32];
                            tmem_ld_x32(&_tmem_load_2[0], taddr_0_2 + row_tmem_base_1 + 320 + dim_chunk_5 * 32);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float _tmem_load_3[32];
                            tmem_ld_x32(&_tmem_load_3[0], taddr_0_2 + row_tmem_base_1 + 448 + dim_chunk_5 * 32);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (row_1 >= segment_offset_4 && row_1 < segment_limit_4) {
                                int token_in_batch_2 = physical_chunk_6 * 128 + row_1;
                                int token = physical_batch_2 * seqlen + token_in_batch_2;
                                float _exp2_3 = approx_exp2((smem_cumsum_all[input_stage_6 * 128 + (unsigned int)row_1] - segment_base_2) * 1.4426950408889634f);
                                float decay_1 = _exp2_3;
                                #pragma unroll
                                for (int local_pair = 0; local_pair < 16; local_pair++) {
                                    int dim_pair_base = dim_chunk_5 * 32 + local_pair * 2;
                                    unsigned int x_packed[1];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])) : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)(dim_pair_base / 64 * 16384 + row_1 * 128 + dim_pair_base % 64 * 2 ^ (dim_pair_base / 64 * 16384 + row_1 * 128 + dim_pair_base % 64 * 2 >> 7 & 7) << 4))));
                                    float x_packed_f32[2];
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 1; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&x_packed_f32[_pair * 2])[0]), "=f"((&x_packed_f32[_pair * 2])[1])
                                            : "r"(x_packed[_pair]));
                                    }
                                    #pragma unroll
                                    for (int pair_lane = 0; pair_lane < 2; pair_lane++) {
                                        int local_dim = local_pair * 2 + pair_lane;
                                        int dim_3 = dim_pair_base + pair_lane;
                                        float _fma_0 = __fmaf_rn(_tmem_load_3[local_dim], decay_1, _tmem_load_2[local_dim]);
                                        float value = _fma_0;
                                        float x_value = x_packed_f32[pair_lane];
                                        if (D_mode == 1) {
                                            float _fma_1 = __fmaf_rn(x_value, d_head_value, value);
                                            value = _fma_1;
                                        }
                                        if (D_mode == 2) {
                                            float _fma_2 = __fmaf_rn(x_value, (float)D[head_3 * 64 + dim_3], value);
                                            value = _fma_2;
                                        }
                                        if (has_z != 0) {
                                            float z_value = (float)z[token * nheads * 64 + head_3 * 64 + dim_3];
                                            float _expf_0 = __expf(-z_value);
                                            float _rcp_0 = approx_rcp(1.0f + _expf_0);
                                            value *= z_value * _rcp_0;
                                        }
                                        int out_index = (((physical_batch_2 * nheads + head_3) * 64 + dim_3) * nchunks + physical_chunk_6) * 128 + row_1;
                                        {
                                            out_native[out_index] = value;
                                        }
                                    }
                                }
                            }
                        }
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (warp == 12) {
                            if (elect_sync()) {
                                mbarrier_arrive(outputs_empty_addr);
                                mbarrier_arrive(aux_empty_addr + (input_stage_6) * 8);
                                mbarrier_arrive(x_empty_addr + (input_stage_6) * 8);
                            }
                        }
                        input_stage_6 += 1;
                        if (input_stage_6 == 2) { input_stage_6 = 0; _phase_aux_full_2 ^= 1; _phase_x_full_2 ^= 1; }
                    }
                }
            }
            int warp_id_in_role_1 = (warp - 12);
            if (warp_id_in_role_1 == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 12) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

