// clang-format off
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_U_ACC_OFFSET 240
#define TMEM_TMEM_U2_INP_OFFSET 240
#define TMEM_TMEM_OUT_OFFSET 192
#define NUM_CHUNK_PIPE_STAGES 8
#define NUM_TMA_PIPE_STAGES 8
#define NUM_OUTPUT_ACC_PIPE_STAGES 3
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 4096
#define SMEM_SMEM_QD_STRIDE 15360
#define SMEM_SMEM_KD_OFF 5120
#define SMEM_SMEM_KD_STAGE_BYTES 4096
#define SMEM_SMEM_KD_STRIDE 15360
#define SMEM_SMEM_KD_LEFT_OFF 5120
#define SMEM_SMEM_KD_LEFT_STAGE_BYTES 2048
#define SMEM_SMEM_KD_LEFT_STRIDE 15360
#define SMEM_SMEM_KD_RIGHT_OFF 7168
#define SMEM_SMEM_KD_RIGHT_STAGE_BYTES 2048
#define SMEM_SMEM_KD_RIGHT_STRIDE 15360
#define SMEM_SMEM_KR_TRANS_OFF 9216
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 4096
#define SMEM_SMEM_KR_TRANS_STRIDE 15360
#define SMEM_SMEM_KR_TRANS_LEFT_OFF 9216
#define SMEM_SMEM_KR_TRANS_LEFT_STAGE_BYTES 2048
#define SMEM_SMEM_KR_TRANS_LEFT_STRIDE 15360
#define SMEM_SMEM_KR_TRANS_RIGHT_OFF 11264
#define SMEM_SMEM_KR_TRANS_RIGHT_STAGE_BYTES 2048
#define SMEM_SMEM_KR_TRANS_RIGHT_STRIDE 15360
#define SMEM_SMEM_MQK_TRANS_OFF 13312
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 512
#define SMEM_SMEM_MQK_TRANS_STRIDE 15360
#define SMEM_SMEM_V_OFF 13824
#define SMEM_SMEM_V_STAGE_BYTES 2048
#define SMEM_SMEM_V_STRIDE 15360
#define SMEM_SMEM_OUT_OFF 123904
#define SMEM_SMEM_OUT_STAGE_BYTES 2048
#define SMEM_SMEM_OUT_STRIDE 2048
#define SMEM_SMEM_GT_ALL_OFF 15872
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 108032
#define SMEM_SMEM_GT_ALL_STRIDE 108032
#define SMEM_TOTAL 138240
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


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
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


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
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
kernel_flashkda_bf16_bt16_chain_m64(__nv_bfloat16* __restrict__ ws_qd, CakeTensorMap const* ws_qd_tma, __nv_bfloat16* __restrict__ ws_kd, CakeTensorMap const* ws_kd_tma, __nv_bfloat16* __restrict__ ws_w, CakeTensorMap const* ws_w_tma, __nv_bfloat16* __restrict__ ws_qk, CakeTensorMap const* ws_qk_tma, float* __restrict__ ws_diag, CakeTensorMap const* ws_diag_tma, __nv_bfloat16* __restrict__ v, CakeTensorMap const* v_tma, long long* __restrict__ cu_seqlens, int* __restrict__ cu_chunks, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, CakeTensorMap const* out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(ws_qd_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(ws_kd_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(ws_w_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(ws_qk_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(ws_diag_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_kd_addr = smem + 5120;
    __nv_bfloat16* smem_kd_left = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_kd_left_addr = smem + 5120;
    __nv_bfloat16* smem_kd_right = reinterpret_cast<__nv_bfloat16*>(smem_raw + 7168);
    const int smem_kd_right_addr = smem + 7168;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kr_trans_addr = smem + 9216;
    __nv_bfloat16* smem_kr_trans_left = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kr_trans_left_addr = smem + 9216;
    __nv_bfloat16* smem_kr_trans_right = reinterpret_cast<__nv_bfloat16*>(smem_raw + 11264);
    const int smem_kr_trans_right_addr = smem + 11264;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_mqk_trans_addr = smem + 13312;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13824);
    const int smem_v_addr = smem + 13824;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int smem_out_addr = smem + 123904;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 15872);
    const int smem_gt_all_addr = smem + 15872;

    // Mbarrier init (11 groups, 76 barriers)
    // Mbarriers at smem_raw[0..608)

    if (warp == 0) {
        // --- pipeline 'tma_pipe' ---
        // qk_full: 8 barriers, init_count=1
        // --- pipeline 'chunk_pipe' ---
        // smem_free: 8 barriers, init_count=2
        // state_inp_ready: 8 barriers, init_count=8
        // state_inp_left_ready: 8 barriers, init_count=4
        // old_out_ready: 8 barriers, init_count=1
        // u2_inp_ready: 8 barriers, init_count=8
        // recurrence_left_done: 8 barriers, init_count=2
        // recurrence_right_done: 8 barriers, init_count=2
        // output_ready: 8 barriers, init_count=1
        // --- pipeline 'output_acc_pipe' ---
        // out_empty: 3 barriers, init_count=4
        // tmem_dealloc_ready: 1 barriers, init_count=3
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 64 + _bar * 8, 2);
        }
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 128 + _bar * 8, 8);
        }
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 192 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 256 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 320 + _bar * 8, 8);
        }
        for (int _bar = lane; _bar < 16; _bar += 32) {
            mbarrier_init(smem + 384 + _bar * 8, 2);
        }
        for (int _bar = lane; _bar < 8; _bar += 32) {
            mbarrier_init(smem + 512 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 3; _bar += 32) {
            mbarrier_init(smem + 576 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 1; _bar += 32) {
            mbarrier_init(smem + 600 + _bar * 8, 3);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 608);
    if (warp == 0) {
        int _tmem_hold = smem + 608;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define smem_free_addr (mbar_base + 64)
    #define state_inp_ready_addr (mbar_base + 128)
    #define state_inp_left_ready_addr (mbar_base + 192)
    #define old_out_ready_addr (mbar_base + 256)
    #define u2_inp_ready_addr (mbar_base + 320)
    #define recurrence_left_done_addr (mbar_base + 384)
    #define recurrence_right_done_addr (mbar_base + 448)
    #define output_ready_addr (mbar_base + 512)
    #define out_empty_addr (mbar_base + 576)
    #define tmem_dealloc_ready_addr (mbar_base + 600)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_u_acc = taddr + 240;
    const int tmem_tmem_u2_inp = taddr + 240;
    const int tmem_tmem_out = taddr + 192;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // epilogue_main
            int split_task_idx = blockIdx.x;
            int task_idx = split_task_idx / 2;
            int value_split_idx = split_task_idx % 2;
            int value_row_offset = value_split_idx * 64;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int seq_len = (int)(eos - bos);
            int num_chunks = cu_chunks[seq_idx + 1] - cu_chunks[seq_idx];
            int warp_id_in_role = (warp - 0);
            int epilogue_local_warp = warp_id_in_role;
            int warp_in_wg = warp % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int lane_quad = lane & 3;
            int local_row_top = warp_in_wg * 16 + lane / 4;
            int local_row_bot = local_row_top + 8;
            int state_row_top = value_row_offset + local_row_top;
            int state_row_bot = value_row_offset + local_row_bot;
            unsigned int epilogue_stage = 0;
            unsigned int epilogue_output_acc_stage = 0;
            unsigned int output_stage = 0;
            unsigned int _phase_output_ready = 0;
            #pragma unroll 1
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                mbarrier_wait(output_ready_addr + (epilogue_stage) * 8, _phase_output_ready);
                int chunk_is_full = ((seq_len >= (chunk_idx + 1) * 16) ? 1 : 0);
                if (chunk_is_full != 0) {
                    float _tmem_load_5[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
                        : "r"(taddr + 192 + epilogue_output_acc_stage * 16 + (unsigned int)tmem_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_0 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_0);
                    if (elect_sync()) {
                        mbarrier_arrive(out_empty_addr + (epilogue_output_acc_stage) * 8);
                    }
                    if (epilogue_local_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 6;");
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 2048;
                    unsigned int out_packed[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                        out_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int token_group = 0; token_group < 1; token_group++) {
                        int mtx_idx = lane / 8;
                        int row_addr = lane & 7;
                        int dim_base = epilogue_local_warp * 16 + (mtx_idx & 1) * 8;
                        int token_base = token_group * 16 + mtx_idx / 2 * 8;
                        int token_addr = token_base + row_addr;
                        int token_pair = token_addr / 2;
                        int token_parity = token_addr & 1;
                        int raw_row = token_pair;
                        int raw_col = (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                        int stsm_offset = (raw_row * 128 + raw_col) * 2;
                        const int pack_base = token_group * 4;
                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(out_tma, 0, (int)(bos + (long long)(chunk_idx * 16)), head_idx, value_split_idx, smem_out_addr + output_stage * 2048);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    output_stage = (output_stage + 1) % 7;
                } else {
                    float _tmem_load_6[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7]))
                        : "r"(taddr + 192 + epilogue_output_acc_stage * 16 + (unsigned int)tmem_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_2 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_2);
                    if (elect_sync()) {
                        mbarrier_arrive(out_empty_addr + (epilogue_output_acc_stage) * 8);
                    }
                    #pragma unroll
                    for (int token_group_1 = 0; token_group_1 < 2; token_group_1++) {
                        int token_pair_1 = token_group_1 * 8 + lane_quad * 2;
                        const int out_reg_base = token_group_1 * 4;
                        long long out_token_0 = bos + (long long)(chunk_idx * 16 + token_pair_1);
                        long long out_token_1 = out_token_0 + 1;
                        if (out_token_0 < eos) {
                            long long out_idx_top_0 = (out_token_0 * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row_top;
                            long long out_idx_bot_0 = (out_token_0 * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row_bot;
                            out[out_idx_top_0] = _tmem_load_6[out_reg_base];
                            out[out_idx_bot_0] = _tmem_load_6[out_reg_base + 2];
                        }
                        if (out_token_1 < eos) {
                            long long out_idx_top_1 = (out_token_1 * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row_top;
                            long long out_idx_bot_1 = (out_token_1 * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row_bot;
                            out[out_idx_top_1] = _tmem_load_6[out_reg_base + 1];
                            out[out_idx_bot_1] = _tmem_load_6[out_reg_base + 3];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 8) { epilogue_stage = 0; _phase_output_ready ^= 1; }
                epilogue_output_acc_stage += 1;
                if (epilogue_output_acc_stage == 3) { epilogue_output_acc_stage = 0; }
            }
            if (epilogue_local_warp == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (epilogue_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: compute_left ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_left_main
            int split_task_idx_1 = blockIdx.x;
            int task_idx_1 = split_task_idx_1 / 2;
            int value_split_idx_1 = split_task_idx_1 % 2;
            int value_row_offset_1 = value_split_idx_1 * 64;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_1 = cu_chunks[seq_idx_1 + 1] - cu_chunks[seq_idx_1];
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int lane_quad_1 = lane & 3;
            int local_row_top_1 = warp_in_wg_1 * 16 + lane / 4;
            int local_row_bot_1 = local_row_top_1 + 8;
            int state_row_top_1 = value_row_offset_1 + local_row_top_1;
            int state_row_bot_1 = value_row_offset_1 + local_row_bot_1;
            int warp_id_in_role_1 = (warp - 4);
            int compute_left_local_warp = warp_id_in_role_1;
            long long state_head_base = ((long long)seq_idx_1 * (long long)num_heads + (long long)head_idx_1) * 128 * 128;
            long long state_base_top = state_head_base + (long long)state_row_top_1 * 128;
            long long state_base_bot = state_head_base + (long long)state_row_bot_1 * 128;
            float state_init[32];
            state_init[0] = 0.0f;
            state_init[1] = 0.0f;
            state_init[2] = 0.0f;
            state_init[3] = 0.0f;
            state_init[4] = 0.0f;
            state_init[5] = 0.0f;
            state_init[6] = 0.0f;
            state_init[7] = 0.0f;
            state_init[8] = 0.0f;
            state_init[9] = 0.0f;
            state_init[10] = 0.0f;
            state_init[11] = 0.0f;
            state_init[12] = 0.0f;
            state_init[13] = 0.0f;
            state_init[14] = 0.0f;
            state_init[15] = 0.0f;
            state_init[16] = 0.0f;
            state_init[17] = 0.0f;
            state_init[18] = 0.0f;
            state_init[19] = 0.0f;
            state_init[20] = 0.0f;
            state_init[21] = 0.0f;
            state_init[22] = 0.0f;
            state_init[23] = 0.0f;
            state_init[24] = 0.0f;
            state_init[25] = 0.0f;
            state_init[26] = 0.0f;
            state_init[27] = 0.0f;
            state_init[28] = 0.0f;
            state_init[29] = 0.0f;
            state_init[30] = 0.0f;
            state_init[31] = 0.0f;
            if (use_initial_state != 0) {
                #pragma unroll
                for (int state_col_group = 0; state_col_group < 8; state_col_group++) {
                    int state_col_pair = state_col_group * 8 + lane_quad_1 * 2;
                    const int state_reg_base = state_col_group * 4;
                    state_init[state_reg_base] = (float)initial_state[state_base_top + (long long)state_col_pair];
                    state_init[state_reg_base + 1] = (float)initial_state[state_base_top + (long long)state_col_pair + 1];
                    state_init[state_reg_base + 2] = (float)initial_state[state_base_bot + (long long)state_col_pair];
                    state_init[state_reg_base + 3] = (float)initial_state[state_base_bot + (long long)state_col_pair + 1];
                }
            }
            asm volatile(
                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                :: "r"(taddr + 64 + (unsigned int)tmem_row_base_1), "r"(*reinterpret_cast<const uint32_t*>(&state_init[0])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[1])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[2])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[3])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[4])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[5])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[6])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[7])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[8])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[9])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[10])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[11])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[12])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[13])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[14])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[15])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[16])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[17])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[18])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[19])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[20])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[21])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[22])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[23])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[24])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[25])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[26])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[27])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[28])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[29])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[30])), "r"(*reinterpret_cast<const uint32_t*>(&state_init[31])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int compute_left_stage = 0;
            unsigned int compute_left_tma_stage = 0;
            unsigned int compute_left_tma_phase = 0;
            unsigned int _phase_recurrence_left_done = 0;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_1; _chunk_idx++) {
                int state_addr = taddr + 64 + (unsigned int)tmem_row_base_1;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(state_addr));
                mbarrier_wait(qk_full_addr + (compute_left_tma_stage) * 8, compute_left_tma_phase);
                uint32_t _tmem_load_0_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)tmem_row_base_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_left_stage) * 8);
                }
                #pragma unroll
                for (int state_col_group_1 = 0; state_col_group_1 < 8; state_col_group_1++) {
                    int state_col_pair_1 = state_col_group_1 * 8 + lane_quad_1 * 2;
                    const int state_reg_base_1 = state_col_group_1 * 4;
                    float state_scale_0 = smem_gt_all[compute_left_stage * 3840 + (unsigned int)state_col_pair_1];
                    float state_scale_1 = smem_gt_all[compute_left_stage * 3840 + (unsigned int)state_col_pair_1 + 1];
                    float2 _f2_0 = make_float2(_tmem_load_0[state_reg_base_1], _tmem_load_0[state_reg_base_1 + 1]);
                    float2 _f2_1 = make_float2(state_scale_0, state_scale_1);
                    float2 state_top_pair = mul_f32x2(_f2_0, _f2_1);
                    float2 _f2_2 = make_float2(_tmem_load_0[state_reg_base_1 + 2], _tmem_load_0[state_reg_base_1 + 3]);
                    float2 _f2_3 = make_float2(state_scale_0, state_scale_1);
                    float2 state_bot_pair = mul_f32x2(_f2_2, _f2_3);
                    _tmem_load_0[state_reg_base_1] = state_top_pair.x;
                    _tmem_load_0[state_reg_base_1 + 1] = state_top_pair.y;
                    _tmem_load_0[state_reg_base_1 + 2] = state_bot_pair.x;
                    _tmem_load_0[state_reg_base_1 + 3] = state_bot_pair.y;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(state_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[31])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_left_stage) * 8);
                }
                mbarrier_wait(recurrence_left_done_addr + (compute_left_stage) * 8, _phase_recurrence_left_done);
                compute_left_stage += 1;
                if (compute_left_stage == 8) { compute_left_stage = 0; _phase_recurrence_left_done ^= 1; }
                compute_left_tma_stage += 1;
                if (compute_left_tma_stage == 8) { compute_left_tma_stage = 0; compute_left_tma_phase ^= 1; }
            }
            if (store_final_state != 0) {
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(taddr + 64 + (unsigned int)tmem_row_base_1));
                #pragma unroll
                for (int state_col_group_2 = 0; state_col_group_2 < 8; state_col_group_2++) {
                    int state_col_pair_2 = state_col_group_2 * 8 + lane_quad_1 * 2;
                    const int state_reg_base_2 = state_col_group_2 * 4;
                    final_state[state_base_top + (long long)state_col_pair_2] = _tmem_load_1[state_reg_base_2];
                    final_state[state_base_top + (long long)state_col_pair_2 + 1] = _tmem_load_1[state_reg_base_2 + 1];
                    final_state[state_base_bot + (long long)state_col_pair_2] = _tmem_load_1[state_reg_base_2 + 2];
                    final_state[state_base_bot + (long long)state_col_pair_2 + 1] = _tmem_load_1[state_reg_base_2 + 3];
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (compute_left_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: compute_right ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_right_main
            int split_task_idx_2 = blockIdx.x;
            int task_idx_2 = split_task_idx_2 / 2;
            int value_split_idx_2 = split_task_idx_2 % 2;
            int value_row_offset_2 = value_split_idx_2 * 64;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            int head_idx_2 = task_idx_2 % num_heads;
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_2 = cu_chunks[seq_idx_2 + 1] - cu_chunks[seq_idx_2];
            int warp_in_wg_2 = warp % 4;
            const int tmem_row_base_2 = warp_in_wg_2 * 32 << 16;
            int lane_quad_2 = lane & 3;
            int local_row_top_2 = warp_in_wg_2 * 16 + lane / 4;
            int local_row_bot_2 = local_row_top_2 + 8;
            int state_row_top_2 = value_row_offset_2 + local_row_top_2;
            int state_row_bot_2 = value_row_offset_2 + local_row_bot_2;
            int warp_id_in_role_2 = (warp - 8);
            int compute_right_local_warp = warp_id_in_role_2;
            long long state_head_base_1 = ((long long)seq_idx_2 * (long long)num_heads + (long long)head_idx_2) * 128 * 128;
            long long state_base_top_1 = state_head_base_1 + (long long)state_row_top_2 * 128;
            long long state_base_bot_1 = state_head_base_1 + (long long)state_row_bot_2 * 128;
            #pragma unroll
            for (int state_col_half = 1; state_col_half < 2; state_col_half++) {
                float state_init_1[32];
                state_init_1[0] = 0.0f;
                state_init_1[1] = 0.0f;
                state_init_1[2] = 0.0f;
                state_init_1[3] = 0.0f;
                state_init_1[4] = 0.0f;
                state_init_1[5] = 0.0f;
                state_init_1[6] = 0.0f;
                state_init_1[7] = 0.0f;
                state_init_1[8] = 0.0f;
                state_init_1[9] = 0.0f;
                state_init_1[10] = 0.0f;
                state_init_1[11] = 0.0f;
                state_init_1[12] = 0.0f;
                state_init_1[13] = 0.0f;
                state_init_1[14] = 0.0f;
                state_init_1[15] = 0.0f;
                state_init_1[16] = 0.0f;
                state_init_1[17] = 0.0f;
                state_init_1[18] = 0.0f;
                state_init_1[19] = 0.0f;
                state_init_1[20] = 0.0f;
                state_init_1[21] = 0.0f;
                state_init_1[22] = 0.0f;
                state_init_1[23] = 0.0f;
                state_init_1[24] = 0.0f;
                state_init_1[25] = 0.0f;
                state_init_1[26] = 0.0f;
                state_init_1[27] = 0.0f;
                state_init_1[28] = 0.0f;
                state_init_1[29] = 0.0f;
                state_init_1[30] = 0.0f;
                state_init_1[31] = 0.0f;
                if (use_initial_state != 0) {
                    #pragma unroll
                    for (int state_col_group_3 = 0; state_col_group_3 < 8; state_col_group_3++) {
                        int state_col_pair_3 = state_col_half * 64 + state_col_group_3 * 8 + lane_quad_2 * 2;
                        const int state_reg_base_3 = state_col_group_3 * 4;
                        state_init_1[state_reg_base_3] = (float)initial_state[state_base_top_1 + (long long)state_col_pair_3];
                        state_init_1[state_reg_base_3 + 1] = (float)initial_state[state_base_top_1 + (long long)state_col_pair_3 + 1];
                        state_init_1[state_reg_base_3 + 2] = (float)initial_state[state_base_bot_1 + (long long)state_col_pair_3];
                        state_init_1[state_reg_base_3 + 3] = (float)initial_state[state_base_bot_1 + (long long)state_col_pair_3 + 1];
                    }
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 64 + (unsigned int)tmem_row_base_2 + (unsigned int)(state_col_half * 64)), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_1[31])));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int compute_stage = 0;
            unsigned int compute_tma_stage = 0;
            unsigned int compute_tma_phase = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_recurrence_right_done = 0;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < num_chunks_2; chunk_idx_1++) {
                int state_tail_addr = taddr + 64 + (unsigned int)tmem_row_base_2 + 64;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(state_tail_addr));
                mbarrier_wait(qk_full_addr + (compute_tma_stage) * 8, compute_tma_phase);
                uint32_t _tmem_load_2_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)tmem_row_base_2 + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                #pragma unroll
                for (int state_col_group_4 = 0; state_col_group_4 < 8; state_col_group_4++) {
                    int state_col_pair_4 = 64 + state_col_group_4 * 8 + lane_quad_2 * 2;
                    const int state_reg_base_4 = state_col_group_4 * 4;
                    float state_scale_0_1 = smem_gt_all[compute_stage * 3840 + (unsigned int)state_col_pair_4];
                    float state_scale_1_1 = smem_gt_all[compute_stage * 3840 + (unsigned int)state_col_pair_4 + 1];
                    float2 _f2_4 = make_float2(_tmem_load_2[state_reg_base_4], _tmem_load_2[state_reg_base_4 + 1]);
                    float2 _f2_5 = make_float2(state_scale_0_1, state_scale_1_1);
                    float2 state_top_pair_1 = mul_f32x2(_f2_4, _f2_5);
                    float2 _f2_6 = make_float2(_tmem_load_2[state_reg_base_4 + 2], _tmem_load_2[state_reg_base_4 + 3]);
                    float2 _f2_7 = make_float2(state_scale_0_1, state_scale_1_1);
                    float2 state_bot_pair_1 = mul_f32x2(_f2_6, _f2_7);
                    _tmem_load_2[state_reg_base_4] = state_top_pair_1.x;
                    _tmem_load_2[state_reg_base_4 + 1] = state_top_pair_1.y;
                    _tmem_load_2[state_reg_base_4 + 2] = state_bot_pair_1.x;
                    _tmem_load_2[state_reg_base_4 + 3] = state_bot_pair_1.y;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(state_tail_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[31])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int v_stage_addr = smem_v_addr + compute_stage * 15360;
                unsigned int v_prefetch_bits[4];
                #pragma unroll
                for (int token_group_2 = 0; token_group_2 < 2; token_group_2++) {
                    const int v_prefetch_reg_base = token_group_2 * 2;
                    int v_ld_matrix = lane / 8 & 1;
                    int v_ld_token = token_group_2 * 8 + (lane & 7);
                    int v_ld_row = warp_in_wg_2 * 16 + v_ld_matrix * 8;
                    int v_ld_row_addr = v_stage_addr + v_ld_token * 64 * 2;
                    int v_ld_addr = (v_ld_row_addr + (v_ld_row * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(v_prefetch_bits[v_prefetch_reg_base]), "=r"(v_prefetch_bits[v_prefetch_reg_base + 1])
                        : "r"(v_ld_addr)
                        : "memory");
                }
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 192 + 48 + (unsigned int)tmem_row_base_2));
                uint32_t _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int residual_packed[4];
                #pragma unroll
                for (int token_group_3 = 0; token_group_3 < 2; token_group_3++) {
                    const int residual_pack_base = token_group_3 * 2;
                    uint32_t _bf16x2_sub_0;
                    asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_0) : "r"(v_prefetch_bits[residual_pack_base]), "r"(_tmem_load_3_bf16[residual_pack_base]));
                    residual_packed[residual_pack_base] = _bf16x2_sub_0;
                    uint32_t _bf16x2_sub_1;
                    asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_1) : "r"(v_prefetch_bits[residual_pack_base + 1]), "r"(_tmem_load_3_bf16[residual_pack_base + 1]));
                    residual_packed[residual_pack_base + 1] = _bf16x2_sub_1;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x2.b32"
                    " [%0], {%1, %2, %3, %4};"
                    :: "r"(taddr + 192 + 48 + (unsigned int)tmem_row_base_2), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[3])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(recurrence_right_done_addr + (compute_stage) * 8, _phase_recurrence_right_done);
                compute_stage += 1;
                if (compute_stage == 8) { compute_stage = 0; _phase_old_out_ready ^= 1; _phase_recurrence_right_done ^= 1; }
                compute_tma_stage += 1;
                if (compute_tma_stage == 8) { compute_tma_stage = 0; compute_tma_phase ^= 1; }
            }
            if (store_final_state != 0) {
                #pragma unroll
                for (int state_col_half_1 = 1; state_col_half_1 < 2; state_col_half_1++) {
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base_2 + (unsigned int)(state_col_half_1 * 64)));
                    #pragma unroll
                    for (int state_col_group_5 = 0; state_col_group_5 < 8; state_col_group_5++) {
                        int state_col_pair_5 = state_col_half_1 * 64 + state_col_group_5 * 8 + lane_quad_2 * 2;
                        const int state_reg_base_5 = state_col_group_5 * 4;
                        final_state[state_base_top_1 + (long long)state_col_pair_5] = _tmem_load_4[state_reg_base_5];
                        final_state[state_base_top_1 + (long long)state_col_pair_5 + 1] = _tmem_load_4[state_reg_base_5 + 1];
                        final_state[state_base_bot_1 + (long long)state_col_pair_5] = _tmem_load_4[state_reg_base_5 + 2];
                        final_state[state_base_bot_1 + (long long)state_col_pair_5 + 1] = _tmem_load_4[state_reg_base_5 + 3];
                    }
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (compute_right_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: mma_q ----
    if (warp == 12) {
        { // mma_q_main
            int split_task_idx_3 = blockIdx.x;
            int task_idx_3 = split_task_idx_3 / 2;
            int seq_idx_3 = seq_order[task_idx_3 / num_heads];
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_3 = cu_chunks[seq_idx_3 + 1] - cu_chunks[seq_idx_3];
            unsigned int mma_q_stage = 0;
            unsigned int mma_q_output_acc_stage = 0;
            unsigned int mma_q_tma_stage = 0;
            unsigned int mma_q_tma_phase = 0;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_out_empty = 1;
            unsigned int _phase_u2_inp_ready = 0;
            #pragma unroll 1
            for (int _chunk_idx_1 = 0; _chunk_idx_1 < num_chunks_3; _chunk_idx_1++) {
                mbarrier_wait(qk_full_addr + (mma_q_tma_stage) * 8, mma_q_tma_phase);
                mbarrier_wait(state_inp_ready_addr + (mma_q_stage) * 8, _phase_state_inp_ready);
                mbarrier_wait(out_empty_addr + (mma_q_output_acc_stage) * 8, _phase_out_empty);
                int _mma_b_lo_0 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 960);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_out + (mma_q_output_acc_stage * 16))), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                elect_commit(recurrence_left_done_addr + (mma_q_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_q_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_1 = make_warp_uniform(((((smem_mqk_trans_addr) >> 4) & 0x3FFF) | 0x200000) + (mma_q_stage) * 960);
                mma_ts_step((tmem_tmem_out + (mma_q_output_acc_stage * 16)), tmem_tmem_u2_inp, _mma_b_lo_1, 0xC0004010, 67437712, 1);
                elect_commit(recurrence_right_done_addr + (mma_q_stage) * 8);
                elect_commit2(output_ready_addr + (mma_q_stage) * 8, smem_free_addr + (mma_q_stage) * 8);
                mma_q_stage += 1;
                if (mma_q_stage == 8) { mma_q_stage = 0; _phase_state_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; }
                mma_q_output_acc_stage += 1;
                if (mma_q_output_acc_stage == 3) { mma_q_output_acc_stage = 0; _phase_out_empty ^= 1; }
                mma_q_tma_stage += 1;
                if (mma_q_tma_stage == 8) { mma_q_tma_stage = 0; mma_q_tma_phase ^= 1; }
            }
        }
    }
    // ---- Role: mma_state ----
    if (warp == 13) {
        { // mma_state_main
            int split_task_idx_4 = blockIdx.x;
            int task_idx_4 = split_task_idx_4 / 2;
            int seq_idx_4 = seq_order[task_idx_4 / num_heads];
            long long bos_4 = cu_seqlens[seq_idx_4];
            long long eos_4 = cu_seqlens[seq_idx_4 + 1];
            int seq_len_4 = (int)(eos_4 - bos_4);
            int num_chunks_4 = cu_chunks[seq_idx_4 + 1] - cu_chunks[seq_idx_4];
            unsigned int mma_stage = 0;
            unsigned int mma_tma_stage = 0;
            unsigned int mma_tma_phase = 0;
            unsigned int _phase_state_inp_left_ready = 0;
            unsigned int _phase_state_inp_ready_1 = 0;
            unsigned int _phase_u2_inp_ready_1 = 0;
            #pragma unroll 1
            for (int _chunk_idx_2 = 0; _chunk_idx_2 < num_chunks_4; _chunk_idx_2++) {
                mbarrier_wait(qk_full_addr + (mma_tma_stage) * 8, mma_tma_phase);
                {
                    mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready_1);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 960);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67372176;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_4), "r"(tmem_tmem_state_inp), "r"(0));
                }
                elect_commit(old_out_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready_1);
                int _mma_b_lo_5 = make_warp_uniform(((((smem_kr_trans_left_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_stage) * 960);
                mma_ts_step(tmem_tmem_state, tmem_tmem_u2_inp, _mma_b_lo_5, 0x40004040, 68224144, 1);
                elect_commit(recurrence_left_done_addr + (mma_stage) * 8);
                int _mma_b_lo_6 = make_warp_uniform(((((smem_kr_trans_right_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_stage) * 960);
                mma_ts_step((tmem_tmem_state + (64)), tmem_tmem_u2_inp, _mma_b_lo_6, 0x40004040, 68224144, 1);
                elect_commit2(recurrence_right_done_addr + (mma_stage) * 8, smem_free_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 8) { mma_stage = 0; _phase_state_inp_left_ready ^= 1; _phase_state_inp_ready_1 ^= 1; _phase_u2_inp_ready_1 ^= 1; }
                mma_tma_stage += 1;
                if (mma_tma_stage == 8) { mma_tma_stage = 0; mma_tma_phase ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: factors ----
    if (warp == 14) {
        { // factors_main
            int split_task_idx_5 = blockIdx.x;
            int task_idx_5 = split_task_idx_5 / 2;
            int value_split_idx_3 = split_task_idx_5 % 2;
            int value_row_offset_3 = value_split_idx_3 * 64;
            int seq_idx_5 = seq_order[task_idx_5 / num_heads];
            int head_idx_3 = task_idx_5 % num_heads;
            long long bos_5 = cu_seqlens[seq_idx_5];
            long long eos_5 = cu_seqlens[seq_idx_5 + 1];
            int seq_len_5 = (int)(eos_5 - bos_5);
            int num_chunks_5 = cu_chunks[seq_idx_5 + 1] - cu_chunks[seq_idx_5];
            int chunk_base = cu_chunks[seq_idx_5];
            unsigned int factor_stage = 0;
            unsigned int factor_tma_stage = 0;
            unsigned int factor_tma_phase = 0;
            unsigned int _phase_smem_free = 1;
            #pragma unroll 1
            for (int chunk_idx_2 = 0; chunk_idx_2 < num_chunks_5; chunk_idx_2++) {
                mbarrier_wait(smem_free_addr + (factor_stage) * 8, _phase_smem_free);
                int ws_chunk = chunk_base + chunk_idx_2;
                int ws_row = ws_chunk * 16;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(qk_full_addr + (factor_tma_stage) * 8, 15360);
                    tma_4d_gmem2smem(smem_qd_addr + factor_stage * 15360, ws_qd_tma, 0, ws_row, head_idx_3, 0, qk_full_addr + (factor_tma_stage) * 8);
                    tma_4d_gmem2smem(smem_kd_addr + factor_stage * 15360, ws_kd_tma, 0, ws_row, head_idx_3, 0, qk_full_addr + (factor_tma_stage) * 8);
                    tma_4d_gmem2smem(smem_kr_trans_addr + factor_stage * 15360, ws_w_tma, 0, ws_row, head_idx_3, 0, qk_full_addr + (factor_tma_stage) * 8);
                    tma_5d_gmem2smem(smem_mqk_trans_addr + factor_stage * 15360, ws_qk_tma, 0, 0, ws_chunk, head_idx_3, 0, qk_full_addr + (factor_tma_stage) * 8);
                    tma_4d_gmem2smem(smem_gt_all_addr + factor_stage * 15360, ws_diag_tma, 0, ws_chunk, head_idx_3, 0, qk_full_addr + (factor_tma_stage) * 8);
                    tma_3d_gmem2smem(smem_v_addr + factor_stage * 15360, v_tma, value_row_offset_3, head_idx_3, (int)(bos_5 + (long long)(chunk_idx_2 * 16)), qk_full_addr + (factor_tma_stage) * 8);
                }
                factor_stage += 1;
                if (factor_stage == 8) { factor_stage = 0; _phase_smem_free ^= 1; }
                factor_tma_stage += 1;
                if (factor_tma_stage == 8) { factor_tma_stage = 0; factor_tma_phase ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
