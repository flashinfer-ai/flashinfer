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
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 66560
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 66560
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_ACC_SCALE_OFF 197632
#define SMEM_SMEM_ACC_SCALE_STAGE_BYTES 512
#define SMEM_SMEM_ACC_SCALE_STRIDE 512
#define SMEM_SMEM_SUM_OFF 198144
#define SMEM_SMEM_SUM_STAGE_BYTES 256
#define SMEM_SMEM_SUM_STRIDE 256
#define SMEM_SMEM_INDICES_OFF 198400
#define SMEM_SMEM_INDICES_STAGE_BYTES 2560
#define SMEM_SMEM_INDICES_STRIDE 2560
#define SMEM_TOTAL 200960
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


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
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
kernel_cake_dsv4_bf16_h64_prefill(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_swa_kv, CakeTensorMap const* tmap_compressed_kv, __nv_bfloat16* __restrict__ O, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int sparse_topk, int has_sinks)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_swa_kv)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_compressed_kv)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_kv_addr = smem + 66560;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_v_addr = smem + 66560;
    float* smem_acc_scale = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_acc_scale_addr = smem + 197632;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 198144);
    const int smem_sum_addr = smem + 198144;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 198400);
    const int smem_indices_addr = smem + 198400;

    // Mbarrier init (14 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=3
            mbarrier_init(smem + 8, 3);
            mbarrier_init(smem + 16, 3);
            mbarrier_init(smem + 24, 3);
            mbarrier_init(smem + 32, 3);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // p_full: 2 barriers, init_count=4
            mbarrier_init(smem + 88, 4);
            mbarrier_init(smem + 96, 4);
            // stats: 2 barriers, init_count=4
            mbarrier_init(smem + 104, 4);
            mbarrier_init(smem + 112, 4);
            // corr_done: 2 barriers, init_count=8
            mbarrier_init(smem + 120, 8);
            mbarrier_init(smem + 128, 8);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // staged_o2_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // staged_o2_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 152, 4);
            // staged_o3_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // staged_o3_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 168, 4);
            // sum_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 176, 4);
            // tmem_dealloc: 1 barriers, init_count=15
            mbarrier_init(smem + 184, 15);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
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
    #define stats_addr (mbar_base + 104)
    #define corr_done_addr (mbar_base + 120)
    #define pv_done_addr (mbar_base + 136)
    #define staged_o2_ready_addr (mbar_base + 144)
    #define staged_o2_empty_addr (mbar_base + 152)
    #define staged_o3_ready_addr (mbar_base + 160)
    #define staged_o3_empty_addr (mbar_base + 168)
    #define sum_ready_addr (mbar_base + 176)
    #define tmem_dealloc_addr (mbar_base + 184)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            int query_idx = blockIdx.x;
            int active_topk = sparse_topk_lens[query_idx];
            int tile_count = (active_topk + 128 - 1) / 128;
            const int warp_in_compute = warp;
            const int tmem_row_origin = warp_in_compute * 32;
            const int logical_row_origin = warp_in_compute * 16;
            const int my_row = logical_row_origin + lane % 16;
            const int col_half = lane / 16;
            float row_max = -CAKE_INF;
            float row_sum = 0.0f;
            if (has_sinks != 0 && my_row < num_heads) {
                row_max = sinks[my_row] * 1.4426950408889634f / softmax_scale_log2;
                row_sum = 1.0f;
            }
            #pragma unroll
            for (int tile = 0; tile < 5; tile++) {
                if (tile_count > tile) {
                    int phase = tile & 1;
                    int wait_phase = tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (phase) * 8, wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int score_col = ((phase != 0) ? 128 : 0);
                    int score_addr = taddr + (unsigned int)score_col + (unsigned int)(tmem_row_origin << 16);
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(score_addr)
                        : "memory");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                        : "r"(score_addr + 32)
                        : "memory");
                    int valid_cols = active_topk - tile * 128 - col_half * 64;
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                    if (valid_cols > 64) {
                        valid_cols = 64;
                    }
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = valid_cols;
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
                        if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -CAKE_INF;
                    }
                    uint32_t _slice_lo_mask_1;
                    {
                        int _lim_2 = valid_cols - 32;
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
                        if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_4);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_4);
                    float tile_max = _tmem_load_0_max;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                    float _max_0 = max_noftz(tile_max, _shfl_xor_0);
                    tile_max = _max_0;
                    float _max_1 = max_noftz(row_max, tile_max);
                    float new_max = _max_1;
                    float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, (-new_max) * softmax_scale_log2);
                    float delta = _fma_0;
                    float _exp2_0 = approx_exp2(delta);
                    float acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_6 = {-max_scaled, -max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_5, _fma_c2_6);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_7);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_7);
                    float _tmem_load_0_sum = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                    float block_sum = _tmem_load_0_sum;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_sum, 16);
                    block_sum = block_sum + _shfl_xor_1;
                    float _fma_1 = __fmaf_rn(row_sum, acc_scale, block_sum);
                    row_sum = _fma_1;
                    row_max = new_max;
                    unsigned int packed_p[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        packed_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    int p_addr = taddr + (unsigned int)score_col + 64 + (unsigned int)(tmem_row_origin << 16);
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15]))
                        : "memory");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[15]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (col_half == 0) {
                        smem_acc_scale[phase * 64 + my_row] = acc_scale;
                        if (tile == tile_count - 1) {
                            smem_sum[my_row] = row_sum;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(stats_addr + (phase) * 8);
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (phase) * 8);
                    }
                }
            }
            if (elect_sync()) {
                mbarrier_arrive(sum_ready_addr);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (elect_sync()) {
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: correction_o2 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // correction_o2_main
            float output_scale = bmm2_scale[0];
            int query_idx_1 = blockIdx.x;
            int active_topk_1 = sparse_topk_lens[query_idx_1];
            int tile_count_1 = (active_topk_1 + 128 - 1) / 128;
            const int warp_in_role = warp - 4;
            const int tmem_row_origin_1 = warp_in_role * 32;
            const int logical_row_origin_1 = warp_in_role * 16;
            const int my_row_1 = logical_row_origin_1 + lane % 16;
            const int col_half_1 = lane / 16;
            float staged_o2[64];
            #pragma unroll
            for (int tile_1 = 0; tile_1 < 5; tile_1++) {
                if (tile_count_1 > tile_1) {
                    int phase_1 = tile_1 & 1;
                    int wait_phase_1 = tile_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (phase_1) * 8, wait_phase_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc_scale_1 = smem_acc_scale[phase_1 * 64 + my_row_1];
                    if (tile_1 > 0) {
                        int prev_phase = tile_1 - 1 & 1;
                        mbarrier_wait(pv_done_addr, prev_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        int any_rescale = _vote_0;
                        if (any_rescale != 0) {
                            #pragma unroll
                            for (int v_stage = 0; v_stage < 2; v_stage++) {
                                int output_col = 256 + v_stage * 128;
                                int output_addr = taddr + (unsigned int)output_col + (unsigned int)(tmem_row_origin_1 << 16);
                                float _tmem_load_1[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                                    : "r"(output_addr)
                                    : "memory");
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                                    : "r"(output_addr + 32)
                                    : "memory");
                                const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(output_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[63]))
                                    : "memory");
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            const float2 _scale2_1 = {acc_scale_1, acc_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(staged_o2)[_ls], _scale2_1);
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(corr_done_addr + (phase_1) * 8);
                    }
                    int staged_col = ((phase_1 != 0) ? 0 : 128);
                    int staged_addr = taddr + (unsigned int)staged_col + (unsigned int)(tmem_row_origin_1 << 16);
                    mbarrier_wait(staged_o2_ready_addr, tile_1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                        : "r"(staged_addr)
                        : "memory");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                        : "r"(staged_addr + 32)
                        : "memory");
                    #pragma unroll
                    for (int col = 0; col < 64; col++) {
                        staged_o2[col] = ((tile_1 == 0) ? _tmem_load_2[col] : staged_o2[col] + _tmem_load_2[col]);
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(staged_o2_empty_addr);
                    }
                }
            }
            mbarrier_wait(pv_done_addr, tile_count_1 - 1 & 1);
            unsigned int _phase_sum_ready_0 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
            _phase_sum_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _rcp_0 = approx_rcp(smem_sum[my_row_1]);
            float inv_sum = _rcp_0;
            int output_base = (query_idx_1 * num_heads + my_row_1) * 512;
            #pragma unroll
            for (int v_stage_1 = 0; v_stage_1 < 2; v_stage_1++) {
                int output_col_1 = 256 + v_stage_1 * 128;
                int output_addr_1 = taddr + (unsigned int)output_col_1 + (unsigned int)(tmem_row_origin_1 << 16);
                float _tmem_load_3[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(output_addr_1)
                    : "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[63]))
                    : "r"(output_addr_1 + 32)
                    : "memory");
                int col_base = v_stage_1 * 128 + col_half_1 * 64;
                if (my_row_1 < num_heads) {
                    #pragma unroll
                    for (int offset = 0; offset < 64; offset += 8) {
                        {
                            const float2 _prescale2_2 = {inv_sum * output_scale, inv_sum * output_scale};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[offset])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                _tmem_load_3[offset + _ps] *= inv_sum * output_scale;
                            #endif
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_3[offset + 0], _tmem_load_3[offset + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_3[offset + 2], _tmem_load_3[offset + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_3[offset + 4], _tmem_load_3[offset + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_3[offset + 6], _tmem_load_3[offset + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_base + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
            int staged_col_base = 256 + col_half_1 * 64;
            if (my_row_1 < num_heads) {
                #pragma unroll
                for (int offset_1 = 0; offset_1 < 64; offset_1 += 8) {
                    {
                        const float2 _prescale2_3 = {inv_sum * output_scale, inv_sum * output_scale};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&staged_o2[offset_1])[_ps], _prescale2_3);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            staged_o2[offset_1 + _ps] *= inv_sum * output_scale;
                        #endif
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(staged_o2[offset_1 + 0], staged_o2[offset_1 + 1]);
                        _pk[1] = __floats2bfloat162_rn(staged_o2[offset_1 + 2], staged_o2[offset_1 + 3]);
                        _pk[2] = __floats2bfloat162_rn(staged_o2[offset_1 + 4], staged_o2[offset_1 + 5]);
                        _pk[3] = __floats2bfloat162_rn(staged_o2[offset_1 + 6], staged_o2[offset_1 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_base + staged_col_base + offset_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (elect_sync()) {
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: correction_o3 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // correction_o3_main
            float output_scale_1 = bmm2_scale[0];
            int query_idx_2 = blockIdx.x;
            int active_topk_2 = sparse_topk_lens[query_idx_2];
            int tile_count_2 = (active_topk_2 + 128 - 1) / 128;
            const int warp_in_role_1 = warp - 8;
            const int tmem_row_origin_2 = warp_in_role_1 * 32;
            const int logical_row_origin_2 = warp_in_role_1 * 16;
            const int my_row_2 = logical_row_origin_2 + lane % 16;
            const int col_half_2 = lane / 16;
            float staged_o3[64];
            #pragma unroll
            for (int tile_2 = 0; tile_2 < 5; tile_2++) {
                if (tile_count_2 > tile_2) {
                    int phase_2 = tile_2 & 1;
                    int wait_phase_2 = tile_2 >> 1 & 1;
                    mbarrier_wait(stats_addr + (phase_2) * 8, wait_phase_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc_scale_2 = smem_acc_scale[phase_2 * 64 + my_row_2];
                    if (tile_2 > 0) {
                        int _vote_1 = __any_sync(0xFFFFFFFF, acc_scale_2 < 1.0f);
                        int any_rescale_1 = _vote_1;
                        if (any_rescale_1 != 0) {
                            const float2 _scale2_0 = {acc_scale_2, acc_scale_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(staged_o3)[_ls], _scale2_0);
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(corr_done_addr + (phase_2) * 8);
                    }
                    int staged_col_1 = ((phase_2 != 0) ? 0 : 128);
                    int staged_addr_1 = taddr + (unsigned int)staged_col_1 + (unsigned int)(tmem_row_origin_2 << 16);
                    mbarrier_wait(staged_o3_ready_addr, tile_2 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_4[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                        : "r"(staged_addr_1)
                        : "memory");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[63]))
                        : "r"(staged_addr_1 + 32)
                        : "memory");
                    #pragma unroll
                    for (int col_1 = 0; col_1 < 64; col_1++) {
                        staged_o3[col_1] = ((tile_2 == 0) ? _tmem_load_4[col_1] : staged_o3[col_1] + _tmem_load_4[col_1]);
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(staged_o3_empty_addr);
                    }
                }
            }
            unsigned int _phase_sum_ready_0_1 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0_1);
            _phase_sum_ready_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _rcp_1 = approx_rcp(smem_sum[my_row_2]);
            float inv_sum_1 = _rcp_1;
            int output_base_1 = (query_idx_2 * num_heads + my_row_2) * 512;
            int staged_col_base_1 = 384 + col_half_2 * 64;
            if (my_row_2 < num_heads) {
                #pragma unroll
                for (int offset_2 = 0; offset_2 < 64; offset_2 += 8) {
                    {
                        const float2 _prescale2_1 = {inv_sum_1 * output_scale_1, inv_sum_1 * output_scale_1};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&staged_o3[offset_2])[_ps], _prescale2_1);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            staged_o3[offset_2 + _ps] *= inv_sum_1 * output_scale_1;
                        #endif
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(staged_o3[offset_2 + 0], staged_o3[offset_2 + 1]);
                        _pk[1] = __floats2bfloat162_rn(staged_o3[offset_2 + 2], staged_o3[offset_2 + 3]);
                        _pk[2] = __floats2bfloat162_rn(staged_o3[offset_2 + 4], staged_o3[offset_2 + 5]);
                        _pk[3] = __floats2bfloat162_rn(staged_o3[offset_2 + 6], staged_o3[offset_2 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_base_1 + staged_col_base_1 + offset_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (elect_sync()) {
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int query_idx_3 = blockIdx.x;
            int active_topk_3 = sparse_topk_lens[query_idx_3];
            int tile_count_3 = (active_topk_3 + 128 - 1) / 128;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (int tile_3 = 0; tile_3 < 5; tile_3++) {
                if (tile_count_3 > tile_3) {
                    int phase_3 = tile_3 & 1;
                    int score_col_1 = ((phase_3 != 0) ? 128 : 0);
                    #pragma unroll
                    for (int k_stage = 0; k_stage < 4; k_stage++) {
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, phase_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (k_stage * 2) * 512);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem + (score_col_1))), "r"(((k_stage == 0) ? 0 : 1)));
                        int _mma_a_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (k_stage * 2 + 1) * 512);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (score_col_1))), "r"(1));
                        if (k_stage == 3) {
                            elect_commit(s_full_addr + (phase_3) * 8);
                        }
                    }
                    int wait_phase_3 = tile_3 >> 1 & 1;
                    mbarrier_wait(p_full_addr + (phase_3) * 8, wait_phase_3);
                    mbarrier_wait(corr_done_addr + (phase_3) * 8, wait_phase_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int v_stage_2 = 0; v_stage_2 < 2; v_stage_2++) {
                        mbarrier_wait(kv_full_addr + (v_stage_2) * 8, phase_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int output_addr_2 = 256 + v_stage_2 * 128;
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_2) * 2048);
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"((tmem_tmem + (output_addr_2))), "r"(_mma_b_lo_2), "r"(tmem_tmem + (score_col_1 + 64)), "r"(((tile_3 == 0) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (v_stage_2) * 8);
                    }
                    #pragma unroll
                    for (int v_stage_3 = 2; v_stage_3 < 4; v_stage_3++) {
                        mbarrier_wait(kv_full_addr + (v_stage_3) * 8, phase_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int staged_col_2 = ((phase_3 != 0) ? 0 : 128);
                        int _mma_b_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_3) * 2048);
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
                    "mov.b32 id, 69272720;\n\t"
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
                    :: "r"((tmem_tmem + (staged_col_2))), "r"(_mma_b_lo_3), "r"(tmem_tmem + (score_col_1 + 64)), "r"(0));
                        if (v_stage_3 == 2) {
                            elect_commit(staged_o2_ready_addr);
                        } else {
                            elect_commit(staged_o3_ready_addr);
                        }
                        elect_commit(kv_empty_addr + (v_stage_3) * 8);
                        if (v_stage_3 == 2) {
                            mbarrier_wait(staged_o2_empty_addr, tile_3 & 1);
                        } else {
                            mbarrier_wait(staged_o3_empty_addr, tile_3 & 1);
                        }
                    }
                    elect_commit(pv_done_addr);
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_warp_rank = warp - 13;
            int query_idx_4 = blockIdx.x;
            int sparse_base = query_idx_4 * sparse_topk;
            int active_topk_4 = sparse_topk_lens[query_idx_4];
            int tile_count_4 = (active_topk_4 + 128 - 1) / 128;
            unsigned int load_kv_stage = 0;
            unsigned int load_kv_phase = 1;
            if (load_warp_rank == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    #pragma unroll
                    for (int q_stage = 0; q_stage < 8; q_stage++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 8192), tmap_q, 0, 0, q_stage, query_idx_4, q_full_addr);
                    }
                }
            }
            #pragma unroll
            for (int sparse_tile = 0; sparse_tile < 5; sparse_tile++) {
                if (tile_count_4 > sparse_tile) {
                    if (load_warp_rank == sparse_tile % 3) {
                        int index_offset = sparse_tile * 128 + lane * 4;
                        int _vec_load_0[4];
                        {
                            int4 _iv4 = *reinterpret_cast<const int4*>(sparse_indices + (sparse_base + index_offset) + 0);
                            _vec_load_0[0 + 0] = _iv4.x;
                            _vec_load_0[0 + 1] = _iv4.y;
                            _vec_load_0[0 + 2] = _iv4.z;
                            _vec_load_0[0 + 3] = _iv4.w;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)(index_offset * 4)), "r"(_vec_load_0[0]), "r"(_vec_load_0[1]), "r"(_vec_load_0[2]), "r"(_vec_load_0[3]) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 96;" ::: "memory");
            #pragma unroll
            for (int tile_4 = 0; tile_4 < 5; tile_4++) {
                if (tile_count_4 > tile_4) {
                    #pragma unroll
                    for (int k_stage_1 = 0; k_stage_1 < 4; k_stage_1++) {
                        if (tile_4 > 0) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                        }
                        if (elect_sync()) {
                            int producer_groups = ((load_warp_rank < 2) ? 11 : 10);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, producer_groups * 512 * 2);
                        }
                        #pragma unroll 1
                        for (int group = load_warp_rank; group < 32; group += 3) {
                            int index_base = tile_4 * 128 + group * 4;
                            int raw_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                                : "r"(smem_indices_addr + (unsigned int)(index_base * 4)));
                            int row0 = ((raw_rows[0] >= 0) ? raw_rows[0] : 0);
                            int row1 = ((raw_rows[1] >= 0) ? raw_rows[1] : 0);
                            int row2 = ((raw_rows[2] >= 0) ? raw_rows[2] : 0);
                            int row3 = ((raw_rows[3] >= 0) ? raw_rows[3] : 0);
                            int dst_k = smem_kv_addr + load_kv_stage * 32768;
                            if (elect_sync()) {
                                if (tile_4 == 0) {
                                    tma_gather4_gmem2smem(dst_k + group * 512, tmap_swa_kv, k_stage_1 * 128, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                                    tma_gather4_gmem2smem(dst_k + 16384 + group * 512, tmap_swa_kv, k_stage_1 * 128 + 64, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                                } else {
                                    tma_gather4_gmem2smem(dst_k + group * 512, tmap_compressed_kv, k_stage_1 * 128, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                                    tma_gather4_gmem2smem(dst_k + 16384 + group * 512, tmap_compressed_kv, k_stage_1 * 128 + 64, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                                }
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 4) { load_kv_stage = 0; load_kv_phase ^= 1; }
                    }
                }
            }
            if (elect_sync()) {
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"

