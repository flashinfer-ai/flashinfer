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
#define TMEM_NCOLS 512
#define TMEM_SCORES0_OFFSET 0
#define TMEM_SCORES1_OFFSET 128
#define TMEM_OUTPUT0_OFFSET 256
#define TMEM_OUTPUT1_OFFSET 384
#define NUM_KV_PIPE_STAGES 3
#define SMEM_SCALE_SMEM_OFF 1024
#define SMEM_SCALE_SMEM_STAGE_BYTES 4096
#define SMEM_SCALE_SMEM_STRIDE 4096
#define SMEM_Q0_SMEM_OFF 5120
#define SMEM_Q0_SMEM_STAGE_BYTES 32768
#define SMEM_Q0_SMEM_STRIDE 32768
#define SMEM_Q1_SMEM_OFF 37888
#define SMEM_Q1_SMEM_STAGE_BYTES 32768
#define SMEM_Q1_SMEM_STRIDE 32768
#define SMEM_KV_SMEM_OFF 70656
#define SMEM_KV_SMEM_STAGE_BYTES 32768
#define SMEM_KV_SMEM_STRIDE 32768
#define SMEM_V_SMEM_OFF 70656
#define SMEM_V_SMEM_STAGE_BYTES 32768
#define SMEM_V_SMEM_STRIDE 32768
#define SMEM_Q2K_SMEM_OFF 168960
#define SMEM_Q2K_SMEM_STAGE_BYTES 32768
#define SMEM_Q2K_SMEM_STRIDE 32768
#define SMEM_KV_FP8_SMEM_OFF 185344
#define SMEM_KV_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_KV_FP8_SMEM_STRIDE 16384
#define SMEM_TOTAL 201728
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


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(*x0_ptr, -127.0f), x1 = max_noftz(*x1_ptr, -127.0f);
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
    *x0_ptr = r0; *x1_ptr = r1;
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



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
    }
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
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

__global__ __launch_bounds__(512, 1) void
kernel_minimax_sparse_prefill_union_sm100(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, __half* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int* __restrict__ q2k_indices, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int total_q, int num_q_heads, int num_kv_heads, int topk, int batch_size, int uniform_q_len, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_softmax_lse, int return_temperature_lse)
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
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 1024);
    const int scale_smem_addr = smem + 1024;
    __half* q0_smem = reinterpret_cast<__half*>(smem_raw + 5120);
    const int q0_smem_addr = smem + 5120;
    __half* q1_smem = reinterpret_cast<__half*>(smem_raw + 37888);
    const int q1_smem_addr = smem + 37888;
    __half* kv_smem = reinterpret_cast<__half*>(smem_raw + 70656);
    const int kv_smem_addr = smem + 70656;
    __half* v_smem = reinterpret_cast<__half*>(smem_raw + 70656);
    const int v_smem_addr = smem + 70656;
    int* q2k_smem = reinterpret_cast<int*>(smem_raw + 168960);
    const int q2k_smem_addr = smem + 168960;
    uint8_t* kv_fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 185344);
    const int kv_fp8_smem_addr = smem + 185344;

    // Mbarrier init (15 groups, 30 barriers)
    // Mbarriers at smem_raw[0..240)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kv_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q2k_full: 1 barriers, init_count=32
            mbarrier_init(smem + 16, 32);
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // v_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // kv_src_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // kv_src_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 152, 256);
            mbarrier_init(smem + 160, 256);
            // p_full_tail: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            mbarrier_init(smem + 192, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 240);
    if (warp == 0) {
        int _tmem_hold = smem + 240;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q2k_full_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 24)
    #define kv_empty_addr (mbar_base + 48)
    #define v_full_addr (mbar_base + 72)
    #define v_empty_addr (mbar_base + 96)
    #define kv_src_full_addr (mbar_base + 120)
    #define kv_src_empty_addr (mbar_base + 128)
    #define s_full_addr (mbar_base + 136)
    #define p_full_addr (mbar_base + 152)
    #define p_full_tail_addr (mbar_base + 168)
    #define corr_sig_addr (mbar_base + 184)
    #define corr_done_addr (mbar_base + 200)
    #define o_full_addr (mbar_base + 216)
    #define tmem_dealloc_addr (mbar_base + 232)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores0 = taddr;
    const int tmem_scores1 = taddr + 128;
    const int tmem_output0 = taddr + 256;
    const int tmem_output1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int linear_tile = blockIdx.x;
            int batch = 0;
            int q_tile = 0;
            int tile_prefix = 0;
            int tile_active = 0;
            const int tile_size = 256;
            if (uniform_q_len > 0) {
                int uniform_tiles = (uniform_q_len + tile_size - 1) / tile_size;
                batch = linear_tile / uniform_tiles;
                q_tile = linear_tile - batch * uniform_tiles;
                if (batch < batch_size) {
                    tile_active = 1;
                }
            } else {
                for (int candidate_batch = 0; candidate_batch < batch_size; candidate_batch++) {
                    int candidate_q_begin = cu_seqlens_q[candidate_batch];
                    int candidate_q_len = cu_seqlens_q[candidate_batch + 1] - candidate_q_begin;
                    int candidate_tiles = (candidate_q_len + tile_size - 1) / tile_size;
                    if (linear_tile >= tile_prefix && linear_tile < tile_prefix + candidate_tiles) {
                        batch = candidate_batch;
                        q_tile = linear_tile - tile_prefix;
                        tile_active = 1;
                    }
                    tile_prefix = tile_prefix + candidate_tiles;
                }
            }
            int work_head = blockIdx.y;
            int group_size = num_q_heads / num_kv_heads;
            int q_head = work_head;
            int kv_head = q_head / group_size;
            int q_begin = cu_seqlens_q[batch];
            int q_len = cu_seqlens_q[batch + 1] - q_begin;
            int q_local_base = q_tile * 256;
            int q_valid = q_len - q_local_base;
            if (q_valid > 256) {
                q_valid = 256;
            }
            if (q_valid < 0) {
                q_valid = 0;
            }
            if (tile_active == 0) {
                q_valid = 0;
            }
            int query_base = q_begin + q_local_base;
            int k_start = cu_seqlens_k[batch];
            int kv_len = kv_lens[batch];
            if (max_pages == 0) {
                kv_len = cu_seqlens_k[batch + 1] - k_start;
            }
            int query_offset = q_offsets[batch];
            if (derive_q_offset != 0) {
                query_offset = kv_len - q_len;
            }
            int num_n_blocks = (kv_len + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens = query_offset + q_local_base + q_valid;
                int visible_blocks = (visible_tokens + 128 - 1) / 128;
                if (num_n_blocks > visible_blocks) {
                    num_n_blocks = visible_blocks;
                }
            }
            if (q_valid == 0 || num_n_blocks <= 0) {
                num_n_blocks = 1;
            }
            int stage = make_warp_uniform(warp / 4);
            int stage_offset = make_warp_uniform(stage * 128);
            const int tmem_row_base = warp % 4 * 32 << 16;
            int my_row = warp % 4 * 32 + lane;
            int stage_query_offset = stage * 128;
            int query_in_stage = my_row;
            const int stage_query_capacity = 128;
            const int selection_rows = 256;
            int stage_valid = q_valid - stage_query_offset;
            if (stage_valid > stage_query_capacity) {
                stage_valid = stage_query_capacity;
            }
            if (stage_valid < 0) {
                stage_valid = 0;
            }
            int query_in_tile = stage_query_offset + query_in_stage;
            int row_valid = ((query_in_stage < stage_valid) ? 1 : 0);
            float row_max = -BLACKWELL_MSA_INF;
            float row_sum = 0.0f;
            float temperature_sum = 0.0f;
            unsigned int selection_mask = 0;
            unsigned int selection_mask_high = 0;
            unsigned int _phase_q2k_full_0 = 0;
            mbarrier_wait(q2k_full_addr, _phase_q2k_full_0);
            _phase_q2k_full_0 ^= 1;
            {
                if (row_valid != 0 && num_n_blocks <= 32) {
                    #pragma unroll 1
                    for (int slot = 0; slot < topk; slot++) {
                        int selected_block = q2k_smem[slot * selection_rows + query_in_tile];
                        if (selected_block >= 0 && selected_block < 32) {
                            selection_mask = selection_mask | (unsigned int)1 << (unsigned int)selected_block;
                        }
                    }
                }
            }
            int folded_valid_limit = kv_len;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                int n_block = num_n_blocks - 1 - n_iter;
                mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                _phase_s_full ^= 1;
                int selected = 0;
                if (row_valid != 0) {
                    {
                        if (num_n_blocks <= 32) {
                            selected = (int)((selection_mask & (unsigned int)1 << (unsigned int)n_block) != 0);
                        } else {
                            #pragma unroll 1
                            for (int slot_1 = 0; slot_1 < topk; slot_1++) {
                                selected = selected | (int)(q2k_smem[slot_1 * selection_rows + query_in_tile] == n_block);
                            }
                        }
                    }
                }
                int valid_cols = 0;
                if (selected != 0) {
                    {
                        valid_cols = kv_len - n_block * 128;
                        if (causal != 0) {
                            int query_position = query_offset + q_local_base + query_in_tile;
                            int causal_cols = query_position - n_block * 128 + 1;
                            if (valid_cols > causal_cols) {
                                valid_cols = causal_cols;
                            }
                        }
                    }
                    if (valid_cols > 128) {
                        valid_cols = 128;
                    }
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                }
                int score_base = taddr + (unsigned int)stage_offset + (unsigned int)tmem_row_base;
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(score_base)
                    : "memory");
                int body_valid = valid_cols;
                if (body_valid < 0) {
                    body_valid = 0;
                }
                if (body_valid > 0 && body_valid < 64) {
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = body_valid;
                        if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                        else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                        }
                    }
                    #pragma unroll
                    for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                        if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -BLACKWELL_MSA_INF;
                    }
                    uint32_t _slice_lo_mask_1;
                    {
                        int _lim_2 = body_valid - 32;
                        if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                        else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
                        }
                    }
                    #pragma unroll
                    for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                        if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -BLACKWELL_MSA_INF;
                    }
                }
                float2 _reg_reduce_max2_4 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_4);
                row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_4);
                float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_4);
                float tile_max = _tmem_load_0_max;
                if (body_valid <= 0) {
                    tile_max = -BLACKWELL_MSA_INF;
                }
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(score_base + 64)
                    : "memory");
                int tail_valid = valid_cols - 64;
                if (tail_valid < 0) {
                    tail_valid = 0;
                }
                if (valid_cols > 0 && tail_valid < 64) {
                    uint32_t _slice_lo_mask_2;
                    {
                        int _lim_5 = tail_valid;
                        if (_lim_5 <= 0) { _slice_lo_mask_2 = 0u; }
                        else if (_lim_5 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_5));
                        }
                    }
                    #pragma unroll
                    for (int _i_6 = 0; _i_6 < 32; _i_6++) {
                        if (!(_slice_lo_mask_2 & (1u << _i_6))) _tmem_load_1[0 + _i_6] = -BLACKWELL_MSA_INF;
                    }
                    uint32_t _slice_lo_mask_3;
                    {
                        int _lim_7 = tail_valid - 32;
                        if (_lim_7 <= 0) { _slice_lo_mask_3 = 0u; }
                        else if (_lim_7 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_7));
                        }
                    }
                    #pragma unroll
                    for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                        if (!(_slice_lo_mask_3 & (1u << _i_8))) _tmem_load_1[32 + _i_8] = -BLACKWELL_MSA_INF;
                    }
                }
                float2 _reg_reduce_max2_9 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_9);
                row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_9);
                float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_9);
                float tail_max = _tmem_load_1_max;
                if (tail_valid <= 0) {
                    tail_max = -BLACKWELL_MSA_INF;
                }
                float _max_0 = max_noftz(tile_max, tail_max);
                tile_max = _max_0;
                float _max_1 = max_noftz(tile_max, row_max);
                float new_max = _max_1;
                float safe_max = ((new_max == -BLACKWELL_MSA_INF) ? 0.0f : new_max);
                float new_max_scaled = safe_max * softmax_scale_log2;
                float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                float acc_scale_log2 = _fma_0;
                float acc_scale;
                float temperature_acc_scale;
                float selected_max;
                if (acc_scale_log2 >= -8.0f) {
                    selected_max = row_max;
                    safe_max = ((row_max == -BLACKWELL_MSA_INF) ? 0.0f : row_max);
                    acc_scale = 1.0f;
                    temperature_acc_scale = 1.0f;
                    new_max_scaled = safe_max * softmax_scale_log2;
                } else {
                    selected_max = new_max;
                    float _exp2_0 = approx_exp2(acc_scale_log2);
                    acc_scale = ((row_max > -BLACKWELL_MSA_INF) ? _exp2_0 : 1.0f);
                    float _exp2_1 = approx_exp2(acc_scale_log2 * lse_temperature_scale);
                    temperature_acc_scale = ((row_max > -BLACKWELL_MSA_INF) ? _exp2_1 : 1.0f);
                }
                row_max = selected_max;
                scale_smem[stage_offset + my_row] = acc_scale;
                mbarrier_arrive(corr_sig_addr + (stage) * 8);
                float score_bias = ((valid_cols > 0) ? -new_max_scaled : -BLACKWELL_MSA_INF);
                float block_temperature_sum = 0.0f;
                float block_sum = 0.0f;
                int p_base = taddr + (unsigned int)stage_offset + 64 + (unsigned int)tmem_row_base;
                if (return_temperature_lse != 0) {
                    const float2 _fma_b2_10 = {softmax_scale_log2 * lse_temperature_scale, softmax_scale_log2 * lse_temperature_scale};
                    const float2 _fma_c2_11 = {score_bias * lse_temperature_scale, score_bias * lse_temperature_scale};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_10, _fma_c2_11);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                    }
                    float2 _reg_reduce_sum2_12 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_12);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_12);
                    float _tmem_load_1_sum = _reg_reduce_sum2_12.x + _reg_reduce_sum2_12.y;
                    block_temperature_sum = _tmem_load_1_sum;
                    const float2 _fma_b2_13 = {softmax_scale_log2 * lse_temperature_scale, softmax_scale_log2 * lse_temperature_scale};
                    const float2 _fma_c2_14 = {score_bias * lse_temperature_scale, score_bias * lse_temperature_scale};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_13, _fma_c2_14);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_15 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_15);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_15);
                    float _tmem_load_0_sum = _reg_reduce_sum2_15.x + _reg_reduce_sum2_15.y;
                    block_temperature_sum += _tmem_load_0_sum;
                    float _tmem_load_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31]), "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                        : "r"(score_base + 64)
                        : "memory");
                    if (valid_cols > 0 && tail_valid < 64) {
                        uint32_t _slice_lo_mask_4;
                        {
                            int _lim_16 = tail_valid;
                            if (_lim_16 <= 0) { _slice_lo_mask_4 = 0u; }
                            else if (_lim_16 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_16));
                            }
                        }
                        #pragma unroll
                        for (int _i_17 = 0; _i_17 < 32; _i_17++) {
                            if (!(_slice_lo_mask_4 & (1u << _i_17))) _tmem_load_2[0 + _i_17] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_5;
                        {
                            int _lim_18 = tail_valid - 32;
                            if (_lim_18 <= 0) { _slice_lo_mask_5 = 0u; }
                            else if (_lim_18 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_18));
                            }
                        }
                        #pragma unroll
                        for (int _i_19 = 0; _i_19 < 32; _i_19++) {
                            if (!(_slice_lo_mask_5 & (1u << _i_19))) _tmem_load_2[32 + _i_19] = -BLACKWELL_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_20 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_21 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_lf], _fma_b2_20, _fma_c2_21);
                    uint32_t _tmem_load_2_bf16[16];
                    softmax_frag_exp2_cast(&_tmem_load_2[0], _tmem_load_2_bf16, 1);
                    uint32_t _tmem_load_2_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                        _tmem_load_2_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16[15]))
                        : "memory");
                    float _tmem_load_3[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31]), "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                        : "r"(score_base)
                        : "memory");
                    if (body_valid > 0 && body_valid < 64) {
                        uint32_t _slice_lo_mask_6;
                        {
                            int _lim_22 = body_valid;
                            if (_lim_22 <= 0) { _slice_lo_mask_6 = 0u; }
                            else if (_lim_22 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_22));
                            }
                        }
                        #pragma unroll
                        for (int _i_23 = 0; _i_23 < 32; _i_23++) {
                            if (!(_slice_lo_mask_6 & (1u << _i_23))) _tmem_load_3[0 + _i_23] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_7;
                        {
                            int _lim_24 = body_valid - 32;
                            if (_lim_24 <= 0) { _slice_lo_mask_7 = 0u; }
                            else if (_lim_24 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_24));
                            }
                        }
                        #pragma unroll
                        for (int _i_25 = 0; _i_25 < 32; _i_25++) {
                            if (!(_slice_lo_mask_7 & (1u << _i_25))) _tmem_load_3[32 + _i_25] = -BLACKWELL_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_26 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_27 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_lf], _fma_b2_26, _fma_c2_27);
                    uint32_t _tmem_load_3_bf16[32];
                    softmax_frag_exp2_cast(&_tmem_load_3[0], _tmem_load_3_bf16, 0);
                    softmax_frag_exp2_cast(&_tmem_load_3[32], &_tmem_load_3_bf16[16], 0);
                    uint32_t _tmem_load_3_f16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                        _tmem_load_3_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_f16[31]))
                        : "memory");
                    float2 _reg_reduce_sum2_28 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_3[0], &_reg_reduce_sum2_28);
                    softmax_block_sum(&_tmem_load_3[32], &_reg_reduce_sum2_28);
                    float _tmem_load_3_sum = _reg_reduce_sum2_28.x + _reg_reduce_sum2_28.y;
                    block_sum += _tmem_load_3_sum;
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    uint32_t _tmem_load_2_bf16_0[16];
                    softmax_frag_exp2_cast(&_tmem_load_2[32], _tmem_load_2_bf16_0, 0);
                    uint32_t _tmem_load_2_f16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_2[_lp*2 + 32], _tmem_load_2[_lp*2+1 + 32]));
                        _tmem_load_2_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_f16_1[15]))
                        : "memory");
                    float2 _reg_reduce_sum2_29 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_2[0], &_reg_reduce_sum2_29);
                    softmax_block_sum(&_tmem_load_2[32], &_reg_reduce_sum2_29);
                    float _tmem_load_2_sum = _reg_reduce_sum2_29.x + _reg_reduce_sum2_29.y;
                    block_sum += _tmem_load_2_sum;
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_tail_addr + (stage) * 8);
                } else {
                    const float2 _fma_b2_30 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_31 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_30, _fma_c2_31);
                    uint32_t _tmem_load_1_bf16[16];
                    softmax_frag_exp2_cast(&_tmem_load_1[0], _tmem_load_1_bf16, 1);
                    uint32_t _tmem_load_1_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16[15]))
                        : "memory");
                    const float2 _fma_b2_32 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_33 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_32, _fma_c2_33);
                    uint32_t _tmem_load_0_bf16[32];
                    softmax_frag_exp2_cast(&_tmem_load_0[0], _tmem_load_0_bf16, 0);
                    softmax_frag_exp2_cast(&_tmem_load_0[32], &_tmem_load_0_bf16[16], 0);
                    uint32_t _tmem_load_0_f16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_f16[31]))
                        : "memory");
                    float _fma_1 = __fmaf_rn(row_sum, acc_scale, _tmem_load_0[0]);
                    _tmem_load_0[0] = _fma_1;
                    float2 _reg_reduce_sum2_34 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_34);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_34);
                    float _tmem_load_0_sum_1 = _reg_reduce_sum2_34.x + _reg_reduce_sum2_34.y;
                    block_sum = _tmem_load_0_sum_1;
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    uint32_t _tmem_load_1_bf16_0[16];
                    softmax_frag_exp2_cast(&_tmem_load_1[32], _tmem_load_1_bf16_0, 0);
                    uint32_t _tmem_load_1_f16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 32], _tmem_load_1[_lp*2+1 + 32]));
                        _tmem_load_1_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_base + 48), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_f16_1[15]))
                        : "memory");
                    float2 _reg_reduce_sum2_35 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_35);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_35);
                    float _tmem_load_1_sum_1 = _reg_reduce_sum2_35.x + _reg_reduce_sum2_35.y;
                    block_sum += _tmem_load_1_sum_1;
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_tail_addr + (stage) * 8);
                }
                mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                _phase_corr_done ^= 1;
                if (return_temperature_lse != 0) {
                    row_sum = row_sum * acc_scale + block_sum;
                    temperature_sum = temperature_sum * temperature_acc_scale + block_temperature_sum;
                } else {
                    row_sum = block_sum;
                }
            }
            scale_smem[256 + stage_offset + my_row] = row_sum;
            scale_smem[512 + stage_offset + my_row] = row_max;
            scale_smem[768 + stage_offset + my_row] = temperature_sum;
            mbarrier_arrive(corr_sig_addr + (stage) * 8);
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // correction_main
            int linear_tile_1 = blockIdx.x;
            int batch_1 = 0;
            int q_tile_1 = 0;
            int tile_prefix_1 = 0;
            int tile_active_1 = 0;
            const int tile_size_1 = 256;
            if (uniform_q_len > 0) {
                int uniform_tiles_1 = (uniform_q_len + tile_size_1 - 1) / tile_size_1;
                batch_1 = linear_tile_1 / uniform_tiles_1;
                q_tile_1 = linear_tile_1 - batch_1 * uniform_tiles_1;
                if (batch_1 < batch_size) {
                    tile_active_1 = 1;
                }
            } else {
                for (int candidate_batch_1 = 0; candidate_batch_1 < batch_size; candidate_batch_1++) {
                    int candidate_q_begin_1 = cu_seqlens_q[candidate_batch_1];
                    int candidate_q_len_1 = cu_seqlens_q[candidate_batch_1 + 1] - candidate_q_begin_1;
                    int candidate_tiles_1 = (candidate_q_len_1 + tile_size_1 - 1) / tile_size_1;
                    if (linear_tile_1 >= tile_prefix_1 && linear_tile_1 < tile_prefix_1 + candidate_tiles_1) {
                        batch_1 = candidate_batch_1;
                        q_tile_1 = linear_tile_1 - tile_prefix_1;
                        tile_active_1 = 1;
                    }
                    tile_prefix_1 = tile_prefix_1 + candidate_tiles_1;
                }
            }
            int work_head_1 = blockIdx.y;
            int group_size_1 = num_q_heads / num_kv_heads;
            int q_head_1 = work_head_1;
            int kv_head_1 = q_head_1 / group_size_1;
            int q_begin_1 = cu_seqlens_q[batch_1];
            int q_len_1 = cu_seqlens_q[batch_1 + 1] - q_begin_1;
            int q_local_base_1 = q_tile_1 * 256;
            int q_valid_1 = q_len_1 - q_local_base_1;
            if (q_valid_1 > 256) {
                q_valid_1 = 256;
            }
            if (q_valid_1 < 0) {
                q_valid_1 = 0;
            }
            if (tile_active_1 == 0) {
                q_valid_1 = 0;
            }
            int query_base_1 = q_begin_1 + q_local_base_1;
            int k_start_1 = cu_seqlens_k[batch_1];
            int kv_len_1 = kv_lens[batch_1];
            if (max_pages == 0) {
                kv_len_1 = cu_seqlens_k[batch_1 + 1] - k_start_1;
            }
            int query_offset_1 = q_offsets[batch_1];
            if (derive_q_offset != 0) {
                query_offset_1 = kv_len_1 - q_len_1;
            }
            int num_n_blocks_1 = (kv_len_1 + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens_1 = query_offset_1 + q_local_base_1 + q_valid_1;
                int visible_blocks_1 = (visible_tokens_1 + 128 - 1) / 128;
                if (num_n_blocks_1 > visible_blocks_1) {
                    num_n_blocks_1 = visible_blocks_1;
                }
            }
            if (q_valid_1 == 0 || num_n_blocks_1 <= 0) {
                num_n_blocks_1 = 1;
            }
            const int tmem_row_base_1 = warp % 4 * 32 << 16;
            int my_row_1 = warp % 4 * 32 + lane;
            mbarrier_arrive(p_full_addr);
            mbarrier_arrive(p_full_addr + 8);
            unsigned int _phase_corr_sig_0 = 0;
            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
            _phase_corr_sig_0 ^= 1;
            mbarrier_arrive(corr_done_addr);
            unsigned int _phase_corr_sig_1 = 0;
            mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
            _phase_corr_sig_1 ^= 1;
            #pragma unroll 1
            for (int _ = 1; _ < num_n_blocks_1; _++) {
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                float acc_scale0 = scale_smem[my_row_1];
                int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale0 < 1.0f);
                if (_vote_0 != 0) {
                    #pragma unroll
                    for (int col = 0; col < 8; col++) {
                        int output_addr0 = taddr + (unsigned int)TMEM_OUTPUT0_OFFSET + (unsigned int)tmem_row_base_1 + (unsigned int)(col * 16);
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], output_addr0);
                        const float2 _scale2_0 = {acc_scale0, acc_scale0};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_0);
                        tmem_st_x16_f32(output_addr0, _tmem_load_4);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                mbarrier_arrive(p_full_addr);
                mbarrier_arrive(corr_done_addr + 8);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                float acc_scale1 = scale_smem[128 + my_row_1];
                int _vote_1 = __any_sync(0xFFFFFFFF, acc_scale1 < 1.0f);
                if (_vote_1 != 0) {
                    #pragma unroll
                    for (int col_1 = 0; col_1 < 8; col_1++) {
                        int output_addr1 = taddr + (unsigned int)TMEM_OUTPUT1_OFFSET + (unsigned int)tmem_row_base_1 + (unsigned int)(col_1 * 16);
                        float _tmem_load_5[16];
                        tmem_ld_x16(&_tmem_load_5[0], output_addr1);
                        const float2 _scale2_1 = {acc_scale1, acc_scale1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_1);
                        tmem_st_x16_f32(output_addr1, _tmem_load_5);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                mbarrier_arrive(p_full_addr + 8);
                mbarrier_arrive(corr_done_addr);
            }
            mbarrier_arrive(corr_done_addr + 8);
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            unsigned int _phase_o_full_1 = 0;
            mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
            _phase_o_full_1 ^= 1;
            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
            _phase_corr_sig_0 ^= 1;
            mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
            _phase_corr_sig_1 ^= 1;
            #pragma unroll
            for (int stage_1 = 0; stage_1 < 2; stage_1++) {
                int stage_offset_1 = stage_1 * 128;
                int output_offset = ((stage_1 == 0) ? TMEM_OUTPUT0_OFFSET : TMEM_OUTPUT1_OFFSET);
                float final_sum = scale_smem[256 + stage_offset_1 + my_row_1];
                float final_max = scale_smem[512 + stage_offset_1 + my_row_1];
                float final_temperature_sum = scale_smem[768 + stage_offset_1 + my_row_1];
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = ((final_sum > 0.0f && final_sum == final_sum) ? _rcp_0 : 0.0f);
                int stage_query_offset_1 = stage_1 * 128;
                int query_in_stage_1 = my_row_1;
                const int stage_query_capacity_1 = 128;
                int output_head = q_head_1;
                int query = query_base_1 + stage_query_offset_1 + query_in_stage_1;
                int stage_valid_1 = q_valid_1 - stage_query_offset_1;
                if (stage_valid_1 > stage_query_capacity_1) {
                    stage_valid_1 = stage_query_capacity_1;
                }
                if (stage_valid_1 < 0) {
                    stage_valid_1 = 0;
                }
                long long output_row = ((long long)query * (long long)num_q_heads + (long long)output_head) * 128;
                #pragma unroll
                for (int col_2 = 0; col_2 < 8; col_2++) {
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)output_offset + (unsigned int)tmem_row_base_1 + (unsigned int)(col_2 * 16));
                    if (query_in_stage_1 < stage_valid_1) {
                        {
                            const float2 _prescale2_2 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_6[0])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 16; _ps++)
                                _tmem_load_6[0 + _ps] *= inv_sum;
                            #endif
                            __half2 _pk[8];
                            _pk[0] = __floats2half2_rn(_tmem_load_6[0 + 0], _tmem_load_6[0 + 1]);
                            _pk[1] = __floats2half2_rn(_tmem_load_6[0 + 2], _tmem_load_6[0 + 3]);
                            _pk[2] = __floats2half2_rn(_tmem_load_6[0 + 4], _tmem_load_6[0 + 5]);
                            _pk[3] = __floats2half2_rn(_tmem_load_6[0 + 6], _tmem_load_6[0 + 7]);
                            _pk[4] = __floats2half2_rn(_tmem_load_6[0 + 8], _tmem_load_6[0 + 9]);
                            _pk[5] = __floats2half2_rn(_tmem_load_6[0 + 10], _tmem_load_6[0 + 11]);
                            _pk[6] = __floats2half2_rn(_tmem_load_6[0 + 12], _tmem_load_6[0 + 13]);
                            _pk[7] = __floats2half2_rn(_tmem_load_6[0 + 14], _tmem_load_6[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__half*)(out + (output_row + (long long)(col_2 * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__half*)(out + (output_row + (long long)(col_2 * 16))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                    }
                }
                if (query_in_stage_1 < stage_valid_1) {
                    int stat_idx = query * num_q_heads + output_head;
                    if (return_softmax_lse != 0) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum));
                        lse[stat_idx] = ((final_sum > 0.0f) ? final_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                    }
                    if (return_temperature_lse != 0) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(final_temperature_sum));
                        temperature_lse[stat_idx] = ((final_temperature_sum > 0.0f) ? final_max * softmax_scale_log2 * 0.6931471805599453f * lse_temperature_scale + _log2_1 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int linear_tile_2 = blockIdx.x;
            int batch_2 = 0;
            int q_tile_2 = 0;
            int tile_prefix_2 = 0;
            int tile_active_2 = 0;
            const int tile_size_2 = 256;
            if (uniform_q_len > 0) {
                int uniform_tiles_2 = (uniform_q_len + tile_size_2 - 1) / tile_size_2;
                batch_2 = linear_tile_2 / uniform_tiles_2;
                q_tile_2 = linear_tile_2 - batch_2 * uniform_tiles_2;
                if (batch_2 < batch_size) {
                    tile_active_2 = 1;
                }
            } else {
                for (int candidate_batch_2 = 0; candidate_batch_2 < batch_size; candidate_batch_2++) {
                    int candidate_q_begin_2 = cu_seqlens_q[candidate_batch_2];
                    int candidate_q_len_2 = cu_seqlens_q[candidate_batch_2 + 1] - candidate_q_begin_2;
                    int candidate_tiles_2 = (candidate_q_len_2 + tile_size_2 - 1) / tile_size_2;
                    if (linear_tile_2 >= tile_prefix_2 && linear_tile_2 < tile_prefix_2 + candidate_tiles_2) {
                        batch_2 = candidate_batch_2;
                        q_tile_2 = linear_tile_2 - tile_prefix_2;
                        tile_active_2 = 1;
                    }
                    tile_prefix_2 = tile_prefix_2 + candidate_tiles_2;
                }
            }
            int work_head_2 = blockIdx.y;
            int group_size_2 = num_q_heads / num_kv_heads;
            int q_head_2 = work_head_2;
            int kv_head_2 = q_head_2 / group_size_2;
            int q_begin_2 = cu_seqlens_q[batch_2];
            int q_len_2 = cu_seqlens_q[batch_2 + 1] - q_begin_2;
            int q_local_base_2 = q_tile_2 * 256;
            int q_valid_2 = q_len_2 - q_local_base_2;
            if (q_valid_2 > 256) {
                q_valid_2 = 256;
            }
            if (q_valid_2 < 0) {
                q_valid_2 = 0;
            }
            if (tile_active_2 == 0) {
                q_valid_2 = 0;
            }
            int query_base_2 = q_begin_2 + q_local_base_2;
            int k_start_2 = cu_seqlens_k[batch_2];
            int kv_len_2 = kv_lens[batch_2];
            if (max_pages == 0) {
                kv_len_2 = cu_seqlens_k[batch_2 + 1] - k_start_2;
            }
            int query_offset_2 = q_offsets[batch_2];
            if (derive_q_offset != 0) {
                query_offset_2 = kv_len_2 - q_len_2;
            }
            int num_n_blocks_2 = (kv_len_2 + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens_2 = query_offset_2 + q_local_base_2 + q_valid_2;
                int visible_blocks_2 = (visible_tokens_2 + 128 - 1) / 128;
                if (num_n_blocks_2 > visible_blocks_2) {
                    num_n_blocks_2 = visible_blocks_2;
                }
            }
            if (q_valid_2 == 0 || num_n_blocks_2 <= 0) {
                num_n_blocks_2 = 1;
            }
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_q_full_1 = 0;
            mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
            _phase_q_full_1 ^= 1;
            mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
            int _mma_a_lo_0 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
            int _mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (kv_stage) * 2048);
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores0), "r"(0));
            elect_commit(s_full_addr);
            int _mma_a_lo_1 = make_warp_uniform(((q1_smem_addr) >> 4) & 0x3FFF);
            int _mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (kv_stage) * 2048);
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores1), "r"(0));
            elect_commit(s_full_addr + 8);
            elect_commit(kv_empty_addr + (kv_stage) * 8);
            kv_stage += 1;
            if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
            int first_pv = 1;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_tail_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_tail_1 = 0;
            #pragma unroll 1
            for (int __1 = 0; __1 < num_n_blocks_2 - 1; __1++) {
                unsigned int v_stage = kv_stage;
                unsigned int v_phase = kv_phase;
                {
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                }
                {
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                }
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                int _mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output0), "r"(_mma_b_lo_2), "r"(tmem_scores0 + 64), "r"(((first_pv) ? 0 : 1)));
                mbarrier_wait(p_full_tail_addr, _phase_p_full_tail_0);
                _phase_p_full_tail_0 ^= 1;
                int _mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output0), "r"(_mma_b_lo_3), "r"(tmem_scores0 + 64), "r"(1));
                unsigned int k_stage = kv_stage;
                unsigned int k_phase = kv_phase;
                kv_stage += 1;
                if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                int _mma_a_lo_4 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_4 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_scores0), "r"(0));
                elect_commit(s_full_addr);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                int _mma_b_lo_5 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output1), "r"(_mma_b_lo_5), "r"(tmem_scores1 + 64), "r"(((first_pv) ? 0 : 1)));
                mbarrier_wait(p_full_tail_addr + 8, _phase_p_full_tail_1);
                _phase_p_full_tail_1 ^= 1;
                int _mma_b_lo_6 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output1), "r"(_mma_b_lo_6), "r"(tmem_scores1 + 64), "r"(1));
                {
                    elect_commit(kv_empty_addr + (v_stage) * 8);
                }
                int _mma_a_lo_7 = make_warp_uniform(((q1_smem_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_7 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_scores1), "r"(0));
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (k_stage) * 8);
                first_pv = 0;
            }
            {
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
            }
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            int _mma_b_lo_8 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output0), "r"(_mma_b_lo_8), "r"(tmem_scores0 + 64), "r"(((first_pv) ? 0 : 1)));
            mbarrier_wait(p_full_tail_addr, _phase_p_full_tail_0);
            _phase_p_full_tail_0 ^= 1;
            int _mma_b_lo_9 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output0), "r"(_mma_b_lo_9), "r"(tmem_scores0 + 64), "r"(1));
            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
            _phase_p_full_1 ^= 1;
            int _mma_b_lo_10 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    "}\n"
                    :: "r"(tmem_output1), "r"(_mma_b_lo_10), "r"(tmem_scores1 + 64), "r"(((first_pv) ? 0 : 1)));
            mbarrier_wait(p_full_tail_addr + 8, _phase_p_full_tail_1);
            _phase_p_full_tail_1 ^= 1;
            int _mma_b_lo_11 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
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
                    "mov.b32 id, 136380432;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output1), "r"(_mma_b_lo_11), "r"(tmem_scores1 + 64), "r"(1));
            {
                elect_commit(kv_empty_addr + (kv_stage) * 8);
            }
            elect_commit2(o_full_addr, o_full_addr + 8);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 14) {
        { // load_warp_main
            int linear_tile_3 = blockIdx.x;
            int batch_3 = 0;
            int q_tile_3 = 0;
            int tile_prefix_3 = 0;
            int tile_active_3 = 0;
            const int tile_size_3 = 256;
            if (uniform_q_len > 0) {
                int uniform_tiles_3 = (uniform_q_len + tile_size_3 - 1) / tile_size_3;
                batch_3 = linear_tile_3 / uniform_tiles_3;
                q_tile_3 = linear_tile_3 - batch_3 * uniform_tiles_3;
                if (batch_3 < batch_size) {
                    tile_active_3 = 1;
                }
            } else {
                for (int candidate_batch_3 = 0; candidate_batch_3 < batch_size; candidate_batch_3++) {
                    int candidate_q_begin_3 = cu_seqlens_q[candidate_batch_3];
                    int candidate_q_len_3 = cu_seqlens_q[candidate_batch_3 + 1] - candidate_q_begin_3;
                    int candidate_tiles_3 = (candidate_q_len_3 + tile_size_3 - 1) / tile_size_3;
                    if (linear_tile_3 >= tile_prefix_3 && linear_tile_3 < tile_prefix_3 + candidate_tiles_3) {
                        batch_3 = candidate_batch_3;
                        q_tile_3 = linear_tile_3 - tile_prefix_3;
                        tile_active_3 = 1;
                    }
                    tile_prefix_3 = tile_prefix_3 + candidate_tiles_3;
                }
            }
            int work_head_3 = blockIdx.y;
            int group_size_3 = num_q_heads / num_kv_heads;
            int q_head_3 = work_head_3;
            int kv_head_3 = q_head_3 / group_size_3;
            int q_begin_3 = cu_seqlens_q[batch_3];
            int q_len_3 = cu_seqlens_q[batch_3 + 1] - q_begin_3;
            int q_local_base_3 = q_tile_3 * 256;
            int q_valid_3 = q_len_3 - q_local_base_3;
            if (q_valid_3 > 256) {
                q_valid_3 = 256;
            }
            if (q_valid_3 < 0) {
                q_valid_3 = 0;
            }
            if (tile_active_3 == 0) {
                q_valid_3 = 0;
            }
            int query_base_3 = q_begin_3 + q_local_base_3;
            int k_start_3 = cu_seqlens_k[batch_3];
            int kv_len_3 = kv_lens[batch_3];
            if (max_pages == 0) {
                kv_len_3 = cu_seqlens_k[batch_3 + 1] - k_start_3;
            }
            int query_offset_3 = q_offsets[batch_3];
            if (derive_q_offset != 0) {
                query_offset_3 = kv_len_3 - q_len_3;
            }
            int num_n_blocks_3 = (kv_len_3 + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens_3 = query_offset_3 + q_local_base_3 + q_valid_3;
                int visible_blocks_3 = (visible_tokens_3 + 128 - 1) / 128;
                if (num_n_blocks_3 > visible_blocks_3) {
                    num_n_blocks_3 = visible_blocks_3;
                }
            }
            if (q_valid_3 == 0 || num_n_blocks_3 <= 0) {
                num_n_blocks_3 = 1;
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 32768);
                {
                    tma_4d_gmem2smem(q0_smem_addr, (&q), 0, query_base_3, q_head_3, 0, q_full_addr);
                }
                mbarrier_arrive_expect_tx(q_full_addr + 8, 32768);
                {
                    tma_4d_gmem2smem(q1_smem_addr, (&q), 0, query_base_3 + 128, q_head_3, 0, q_full_addr + 8);
                }
            }
            int selection_base = (kv_head_3 * total_q + query_base_3) * topk;
            int selection_elems = q_valid_3 * topk;
            const int selection_rows_1 = 256;
            #pragma unroll 1
            for (int index_offset = lane * 4; index_offset < selection_elems; index_offset += 128) {
                int _vec_load_0[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(q2k_indices + (selection_base + index_offset) + 0);
                    _vec_load_0[0 + 0] = _iv4.x;
                    _vec_load_0[0 + 1] = _iv4.y;
                    _vec_load_0[0 + 2] = _iv4.z;
                    _vec_load_0[0 + 3] = _iv4.w;
                }
                for (int index_in_vec = 0; index_in_vec < 4; index_in_vec++) {
                    int flat_index = index_offset + index_in_vec;
                    int query_row = flat_index / topk;
                    int slot_2 = flat_index - query_row * topk;
                    q2k_smem[slot_2 * selection_rows_1 + query_row] = _vec_load_0[index_in_vec];
                }
            }
            asm volatile("barrier.sync 8, 32;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q2k_full_addr);
            unsigned int load_stage = 0;
            unsigned int kv_src_empty_phase = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int n_iter_1 = 0; n_iter_1 < num_n_blocks_3; n_iter_1++) {
                int n_block_1 = num_n_blocks_3 - 1 - n_iter_1;
                int token_base = k_start_3 + n_block_1 * 128;
                int page_head = kv_head_3;
                {
                    int physical_page = page_table[batch_3 * max_pages + n_block_1];
                    if (physical_page < 0) {
                        physical_page = 0;
                    }
                    token_base = 0;
                    page_head = physical_page * num_kv_heads + kv_head_3;
                }
                {
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 32768);
                        int token0 = token_base;
                        int token1 = token_base + 64;
                        {
                            token0 = 0;
                            token1 = 64;
                        }
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768, (&k), 0, token0, 0, page_head, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768 + 8192, (&k), 0, token1, 0, page_head, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768 + 16384, (&k), 0, token0, 1, page_head, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768 + 24576, (&k), 0, token1, 1, page_head, kv_full_addr + (load_stage) * 8);
                    }
                }
                load_stage += 1;
                if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                {
                    {
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 32768);
                            int token0_1 = token_base;
                            int token1_1 = token_base + 64;
                            {
                                token0_1 = 0;
                                token1_1 = 64;
                            }
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768, (&v), 0, token0_1, 0, page_head, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768 + 8192, (&v), 0, token1_1, 0, page_head, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768 + 16384, (&v), 0, token0_1, 1, page_head, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768 + 24576, (&v), 0, token1_1, 1, page_head, kv_full_addr + (load_stage) * 8);
                        }
                    }
                }
                {
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp == 13 || warp == 15) {
        { // transform_main
            unsigned int _phase_v_empty = 1;
        }
    }

    // Cleanup
}

} // extern "C"

