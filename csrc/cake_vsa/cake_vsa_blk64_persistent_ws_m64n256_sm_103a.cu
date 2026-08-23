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
#define TMEM_NCOLS 320
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBABILITIES_OFFSET 256
#define TMEM_OUTPUT_OFFSET 128
#define NUM_KV_PIPE_STAGES 3
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 65536
#define SMEM_KV_SMEM_STRIDE 65536
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 65536
#define SMEM_V_SMEM_STRIDE 65536
#define SMEM_SCALE_SMEM_OFF 214016
#define SMEM_SCALE_SMEM_STAGE_BYTES 1536
#define SMEM_SCALE_SMEM_STRIDE 1536
#define SMEM_PARTIAL_SMEM_OFF 215552
#define SMEM_PARTIAL_SMEM_STAGE_BYTES 16384
#define SMEM_PARTIAL_SMEM_STRIDE 16384
#define SMEM_TOTAL 231936
#define THREADS 384

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

__global__ __launch_bounds__(384, 1) void
kernel_flashinfer_vsa_blk64_persistent_per_head_m64n256_ws_sm100(CakeTensorMap const* q, CakeTensorMap const* k, CakeTensorMap const* v, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int* __restrict__ q2k_indices, int* __restrict__ q2k_num, int* __restrict__ kv_block_lens, int max_kv_blocks, int sequence_q, int query_blocks, int total_tiles, int tiles_per_cta, int num_heads, float softmax_scale_log2, int return_lse)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* kv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 214016);
    const int scale_smem_addr = smem + 214016;
    float* partial_smem = reinterpret_cast<float*>(smem_raw + 215552);
    const int partial_smem_addr = smem + 215552;

    // Mbarrier init (13 groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // s_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            // p_lastsplit: 1 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            // corr_done: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // tile_done: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 320 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 0) {
        int _tmem_hold = smem + 136;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 64)
    #define s_empty_addr (mbar_base + 72)
    #define p_full_addr (mbar_base + 80)
    #define p_lastsplit_addr (mbar_base + 88)
    #define p_empty_addr (mbar_base + 96)
    #define corr_sig_addr (mbar_base + 104)
    #define corr_done_addr (mbar_base + 112)
    #define o_full_addr (mbar_base + 120)
    #define tile_done_addr (mbar_base + 128)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_probabilities = taddr + 256;
    const int tmem_output = taddr + 128;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // softmax_main
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_corr_done_0 = 0;
            #pragma unroll 1
            for (int tile_iter = 0; tile_iter < tiles_per_cta; tile_iter++) {
                int tile_idx = blockIdx.x + tile_iter * gridDim.x;
                if (tile_idx < total_tiles) {
                    int q_block = tile_idx % query_blocks;
                    int head = tile_idx / query_blocks;
                    int query_base = q_block * 64;
                    int q_valid = sequence_q - query_base;
                    if (q_valid > 64) {
                        q_valid = 64;
                    }
                    if (q_valid < 0) {
                        q_valid = 0;
                    }
                    int row_id = head * query_blocks + q_block;
                    int row_begin = row_id * max_kv_blocks;
                    int selected_count = q2k_num[row_id];
                    int group_count = (selected_count + 4 - 1) / 4;
                    const int warp_in_role = warp;
                    const int tmem_row_origin = warp_in_role * 32;
                    const int warp_col = warp_in_role / 2;
                    int my_row = warp_in_role % 2 * 32 + lane;
                    int stat_slot = warp_in_role * 32 + lane;
                    int row_valid = ((my_row < q_valid) ? 1 : 0);
                    float row_max = -CAKE_INF;
                    float row_sum = 0.0f;
                    #pragma unroll 1
                    for (int group_index = 0; group_index < group_count; group_index++) {
                        mbarrier_wait(s_full_addr, _phase_s_full_0);
                        _phase_s_full_0 ^= 1;
                        int entry_lo = group_index * 4 + 2 * warp_col;
                        int entry_hi = entry_lo + 1;
                        int valid_lo = 0;
                        int valid_hi = 0;
                        if (entry_lo < selected_count) {
                            int block_lo = q2k_indices[row_begin + entry_lo];
                            valid_lo = kv_block_lens[block_lo];
                        }
                        if (entry_hi < selected_count) {
                            int block_hi = q2k_indices[row_begin + entry_hi];
                            valid_hi = kv_block_lens[block_hi];
                        }
                        if (row_valid == 0) {
                            valid_lo = 0;
                            valid_hi = 0;
                        }
                        if (valid_lo > 64) {
                            valid_lo = 64;
                        }
                        if (valid_hi > 64) {
                            valid_hi = 64;
                        }
                        int valid_cols = valid_lo + valid_hi;
                        int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                        float _tmem_load_0[128];
                        tmem_ld_x32(&_tmem_load_0[0], score_addr);
                        tmem_ld_x32(&_tmem_load_0[32], score_addr + 32);
                        tmem_ld_x32(&_tmem_load_0[64], score_addr + 64);
                        tmem_ld_x32(&_tmem_load_0[96], score_addr + 96);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        mbarrier_arrive(s_empty_addr);
                        if (valid_lo < 64) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_0 = valid_lo;
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
                            uint32_t _slice_hi_mask_0;
                            {
                                int _lim_1 = 64;
                                if (_lim_1 <= 0) { _slice_hi_mask_0 = 0u; }
                                else if (_lim_1 >= 32) { _slice_hi_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_0) : "r"(_lim_1));
                                }
                            }
                            #pragma unroll
                            for (int _i_2 = 0; _i_2 < 32; _i_2++) {
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << _i_2))) _tmem_load_0[0 + _i_2] = -CAKE_INF;
                            }
                            uint32_t _slice_lo_mask_1;
                            {
                                int _lim_3 = valid_lo - 32;
                                if (_lim_3 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_3 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_3));
                                }
                            }
                            uint32_t _slice_hi_mask_1;
                            {
                                int _lim_4 = 32;
                                if (_lim_4 <= 0) { _slice_hi_mask_1 = 0u; }
                                else if (_lim_4 >= 32) { _slice_hi_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_1) : "r"(_lim_4));
                                }
                            }
                            #pragma unroll
                            for (int _i_5 = 0; _i_5 < 32; _i_5++) {
                                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << _i_5))) _tmem_load_0[32 + _i_5] = -CAKE_INF;
                            }
                            uint32_t _slice_lo_mask_2;
                            {
                                int _lim_6 = valid_lo - 64;
                                if (_lim_6 <= 0) { _slice_lo_mask_2 = 0u; }
                                else if (_lim_6 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_6));
                                }
                            }
                            uint32_t _slice_hi_mask_2;
                            {
                                int _lim_7 = 0;
                                if (_lim_7 <= 0) { _slice_hi_mask_2 = 0u; }
                                else if (_lim_7 >= 32) { _slice_hi_mask_2 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_2) : "r"(_lim_7));
                                }
                            }
                            #pragma unroll
                            for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << _i_8))) _tmem_load_0[64 + _i_8] = -CAKE_INF;
                            }
                            uint32_t _slice_lo_mask_3;
                            {
                                int _lim_9 = valid_lo - 96;
                                if (_lim_9 <= 0) { _slice_lo_mask_3 = 0u; }
                                else if (_lim_9 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_9));
                                }
                            }
                            uint32_t _slice_hi_mask_3;
                            {
                                int _lim_10 = -32;
                                if (_lim_10 <= 0) { _slice_hi_mask_3 = 0u; }
                                else if (_lim_10 >= 32) { _slice_hi_mask_3 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_3) : "r"(_lim_10));
                                }
                            }
                            #pragma unroll
                            for (int _i_11 = 0; _i_11 < 32; _i_11++) {
                                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << _i_11))) _tmem_load_0[96 + _i_11] = -CAKE_INF;
                            }
                        }
                        if (valid_hi < 64) {
                            uint32_t _slice_lo_mask_4;
                            {
                                int _lim_12 = 64 + valid_hi;
                                if (_lim_12 <= 0) { _slice_lo_mask_4 = 0u; }
                                else if (_lim_12 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_12));
                                }
                            }
                            uint32_t _slice_hi_mask_4;
                            {
                                int _lim_13 = 128;
                                if (_lim_13 <= 0) { _slice_hi_mask_4 = 0u; }
                                else if (_lim_13 >= 32) { _slice_hi_mask_4 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_4) : "r"(_lim_13));
                                }
                            }
                            #pragma unroll
                            for (int _i_14 = 0; _i_14 < 32; _i_14++) {
                                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << _i_14))) _tmem_load_0[0 + _i_14] = -CAKE_INF;
                            }
                            uint32_t _slice_lo_mask_5;
                            {
                                int _lim_15 = 64 + valid_hi - 32;
                                if (_lim_15 <= 0) { _slice_lo_mask_5 = 0u; }
                                else if (_lim_15 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_15));
                                }
                            }
                            uint32_t _slice_hi_mask_5;
                            {
                                int _lim_16 = 96;
                                if (_lim_16 <= 0) { _slice_hi_mask_5 = 0u; }
                                else if (_lim_16 >= 32) { _slice_hi_mask_5 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_5) : "r"(_lim_16));
                                }
                            }
                            #pragma unroll
                            for (int _i_17 = 0; _i_17 < 32; _i_17++) {
                                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << _i_17))) _tmem_load_0[32 + _i_17] = -CAKE_INF;
                            }
                            uint32_t _slice_lo_mask_6;
                            {
                                int _lim_18 = 64 + valid_hi - 64;
                                if (_lim_18 <= 0) { _slice_lo_mask_6 = 0u; }
                                else if (_lim_18 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_18));
                                }
                            }
                            uint32_t _slice_hi_mask_6;
                            {
                                int _lim_19 = 64;
                                if (_lim_19 <= 0) { _slice_hi_mask_6 = 0u; }
                                else if (_lim_19 >= 32) { _slice_hi_mask_6 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_6) : "r"(_lim_19));
                                }
                            }
                            #pragma unroll
                            for (int _i_20 = 0; _i_20 < 32; _i_20++) {
                                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << _i_20))) _tmem_load_0[64 + _i_20] = -CAKE_INF;
                            }
                            uint32_t _slice_lo_mask_7;
                            {
                                int _lim_21 = 64 + valid_hi - 96;
                                if (_lim_21 <= 0) { _slice_lo_mask_7 = 0u; }
                                else if (_lim_21 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_21));
                                }
                            }
                            uint32_t _slice_hi_mask_7;
                            {
                                int _lim_22 = 32;
                                if (_lim_22 <= 0) { _slice_hi_mask_7 = 0u; }
                                else if (_lim_22 >= 32) { _slice_hi_mask_7 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_7) : "r"(_lim_22));
                                }
                            }
                            #pragma unroll
                            for (int _i_23 = 0; _i_23 < 32; _i_23++) {
                                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << _i_23))) _tmem_load_0[96 + _i_23] = -CAKE_INF;
                            }
                        }
                        float2 _reg_reduce_max2_24 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_24);
                        row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_24);
                        row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_24);
                        row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_24);
                        float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_24);
                        float tile_max = _tmem_load_0_max;
                        if (valid_cols <= 0) {
                            tile_max = -CAKE_INF;
                        }
                        float _max_0 = max_noftz(tile_max, row_max);
                        float new_max = _max_0;
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * softmax_scale_log2;
                        float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                        float acc_scale_log2 = _fma_0;
                        float acc_scale;
                        float selected_max;
                        if (acc_scale_log2 >= -8.0f) {
                            selected_max = row_max;
                            safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                            acc_scale = 1.0f;
                            new_max_scaled = safe_max * softmax_scale_log2;
                        } else {
                            selected_max = new_max;
                            float _exp2_0 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                        }
                        row_max = selected_max;
                        scale_smem[stat_slot] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(corr_sig_addr);
                        float score_bias = ((valid_cols > 0) ? -new_max_scaled : -CAKE_INF);
                        const float2 _fma_b2_25 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_26 = {score_bias, score_bias};
                        #pragma unroll
                        for (int _lf = 0; _lf < 64; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_25, _fma_c2_26);
                        #pragma unroll
                        for (int _le = 0; _le < 64; _le++) {
                            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                        }
                        unsigned int packed_p_lo[32];
                        #pragma unroll
                        for (int _lp = 0; _lp < 32; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            packed_p_lo[_lp] = *(uint32_t*)&_bf2;
                        }
                        int p_addr = taddr + (unsigned int)TMEM_PROBABILITIES_OFFSET + (unsigned int)(tmem_row_origin << 16);
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                        _phase_p_empty_0 ^= 1;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x32.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[15])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[16])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[17])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[18])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[19])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[20])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[21])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[22])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[23])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[24])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[25])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[26])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[27])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[28])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[29])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[30])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_lo[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        #pragma unroll
                        for (int _le = 0; _le < 64; _le++) {
                            _tmem_load_0[_le + 64] = approx_exp2(_tmem_load_0[_le + 64]);
                        }
                        unsigned int packed_p_hi[32];
                        #pragma unroll
                        for (int _lp = 0; _lp < 32; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 64], _tmem_load_0[_lp*2+1 + 64]));
                            packed_p_hi[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x32.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[15])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[16])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[17])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[18])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[19])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[20])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[21])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[22])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[23])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[24])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[25])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[26])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[27])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[28])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[29])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[30])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_hi[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(p_lastsplit_addr);
                        float2 _reg_reduce_sum2_27 = make_float2(0.0f, 0.0f);
                        softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_27);
                        softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_27);
                        softmax_block_sum(&_tmem_load_0[64], &_reg_reduce_sum2_27);
                        softmax_block_sum(&_tmem_load_0[96], &_reg_reduce_sum2_27);
                        float _tmem_load_0_sum = _reg_reduce_sum2_27.x + _reg_reduce_sum2_27.y;
                        float block_sum = _tmem_load_0_sum;
                        mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                        _phase_corr_done_0 ^= 1;
                        row_sum = row_sum * acc_scale + block_sum;
                    }
                    scale_smem[128 + stat_slot] = row_sum;
                    scale_smem[256 + stat_slot] = row_max;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(corr_sig_addr);
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // correction_main
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int tile_iter_1 = 0; tile_iter_1 < tiles_per_cta; tile_iter_1++) {
                int tile_idx_1 = blockIdx.x + tile_iter_1 * gridDim.x;
                if (tile_idx_1 < total_tiles) {
                    int q_block_1 = tile_idx_1 % query_blocks;
                    int head_1 = tile_idx_1 / query_blocks;
                    int query_base_1 = q_block_1 * 64;
                    int q_valid_1 = sequence_q - query_base_1;
                    if (q_valid_1 > 64) {
                        q_valid_1 = 64;
                    }
                    if (q_valid_1 < 0) {
                        q_valid_1 = 0;
                    }
                    int row_id_1 = head_1 * query_blocks + q_block_1;
                    int row_begin_1 = row_id_1 * max_kv_blocks;
                    int selected_count_1 = q2k_num[row_id_1];
                    int group_count_1 = (selected_count_1 + 4 - 1) / 4;
                    const int warp_in_role_1 = warp - 4;
                    const int tmem_row_origin_1 = warp_in_role_1 * 32;
                    int my_row_1 = warp_in_role_1 % 2 * 32 + lane;
                    int stat_slot_1 = warp_in_role_1 * 32 + lane;
                    int partner_slot = (warp_in_role_1 ^ 2) * 32 + lane;
                    int row_addr = tmem_row_origin_1 << 16;
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    #pragma unroll 1
                    for (int _group_index = 1; _group_index < group_count_1; _group_index++) {
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        float acc_scale_1 = scale_smem[stat_slot_1];
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (_vote_0 != 0) {
                            float _tmem_load_1[128];
                            tmem_ld_x32(&_tmem_load_1[0], taddr + 128 + (unsigned int)row_addr);
                            tmem_ld_x32(&_tmem_load_1[32], taddr + 128 + (unsigned int)row_addr + 32);
                            tmem_ld_x32(&_tmem_load_1[64], taddr + 128 + (unsigned int)row_addr + 64);
                            tmem_ld_x32(&_tmem_load_1[96], taddr + 128 + (unsigned int)row_addr + 96);
                            const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x128.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};"
                                :: "r"(taddr + 128 + (unsigned int)row_addr), "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]), "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]), "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]), "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]), "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]), "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]), "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]), "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]), "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]), "f"(_tmem_load_1[18]), "f"(_tmem_load_1[19]), "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]), "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]), "f"(_tmem_load_1[24]), "f"(_tmem_load_1[25]), "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]), "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]), "f"(_tmem_load_1[30]), "f"(_tmem_load_1[31]), "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]), "f"(_tmem_load_1[34]), "f"(_tmem_load_1[35]), "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]), "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]), "f"(_tmem_load_1[40]), "f"(_tmem_load_1[41]), "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]), "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]), "f"(_tmem_load_1[46]), "f"(_tmem_load_1[47]), "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]), "f"(_tmem_load_1[50]), "f"(_tmem_load_1[51]), "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]), "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]), "f"(_tmem_load_1[56]), "f"(_tmem_load_1[57]), "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]), "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]), "f"(_tmem_load_1[62]), "f"(_tmem_load_1[63]), "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]), "f"(_tmem_load_1[66]), "f"(_tmem_load_1[67]), "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]), "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]), "f"(_tmem_load_1[72]), "f"(_tmem_load_1[73]), "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]), "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]), "f"(_tmem_load_1[78]), "f"(_tmem_load_1[79]), "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]), "f"(_tmem_load_1[82]), "f"(_tmem_load_1[83]), "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]), "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]), "f"(_tmem_load_1[88]), "f"(_tmem_load_1[89]), "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]), "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]), "f"(_tmem_load_1[94]), "f"(_tmem_load_1[95]), "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]), "f"(_tmem_load_1[98]), "f"(_tmem_load_1[99]), "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]), "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]), "f"(_tmem_load_1[104]), "f"(_tmem_load_1[105]), "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]), "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]), "f"(_tmem_load_1[110]), "f"(_tmem_load_1[111]), "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]), "f"(_tmem_load_1[114]), "f"(_tmem_load_1[115]), "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]), "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]), "f"(_tmem_load_1[120]), "f"(_tmem_load_1[121]), "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]), "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]), "f"(_tmem_load_1[126]), "f"(_tmem_load_1[127]));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(p_full_addr);
                        mbarrier_arrive(corr_done_addr);
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_sum = scale_smem[128 + stat_slot_1];
                    float final_max = scale_smem[256 + stat_slot_1];
                    float partner_sum = scale_smem[128 + partner_slot];
                    float partner_max = scale_smem[256 + partner_slot];
                    float _max_1 = max_noftz(final_max, partner_max);
                    float max_total = _max_1;
                    float max_total_safe = ((max_total == -CAKE_INF) ? 0.0f : max_total);
                    float _exp2_1 = approx_exp2((final_max - max_total_safe) * softmax_scale_log2);
                    float local_scale = ((final_sum > 0.0f) ? _exp2_1 : 0.0f);
                    float _exp2_2 = approx_exp2((partner_max - max_total_safe) * softmax_scale_log2);
                    float partner_scale = ((partner_sum > 0.0f) ? _exp2_2 : 0.0f);
                    float sum_total = final_sum * local_scale + partner_sum * partner_scale;
                    float _rcp_0 = approx_rcp(sum_total);
                    float local_weight = ((sum_total > 0.0f) ? local_scale * _rcp_0 : 0.0f);
                    float _tmem_load_2[128];
                    tmem_ld_x32(&_tmem_load_2[0], taddr + 128 + (unsigned int)row_addr);
                    tmem_ld_x32(&_tmem_load_2[32], taddr + 128 + (unsigned int)row_addr + 32);
                    tmem_ld_x32(&_tmem_load_2[64], taddr + 128 + (unsigned int)row_addr + 64);
                    tmem_ld_x32(&_tmem_load_2[96], taddr + 128 + (unsigned int)row_addr + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    const float2 _scale2_1 = {local_weight, local_weight};
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_1);
                    int query = query_base_1 + my_row_1;
                    int output_row = (query * num_heads + head_1) * 128;
                    #pragma unroll
                    for (int chunk = 0; chunk < 4; chunk++) {
                        const int buffer_index = chunk % 2;
                        if (warp_in_role_1 >= 2) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)((buffer_index * 64 + my_row_1) * 32 * 4)), "f"(_tmem_load_2[chunk * 32]), "f"(_tmem_load_2[chunk * 32 + 1]), "f"(_tmem_load_2[chunk * 32 + 2]), "f"(_tmem_load_2[chunk * 32 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 4) * 4)), "f"(_tmem_load_2[chunk * 32 + 4]), "f"(_tmem_load_2[chunk * 32 + 4 + 1]), "f"(_tmem_load_2[chunk * 32 + 4 + 2]), "f"(_tmem_load_2[chunk * 32 + 4 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 8) * 4)), "f"(_tmem_load_2[chunk * 32 + 8]), "f"(_tmem_load_2[chunk * 32 + 8 + 1]), "f"(_tmem_load_2[chunk * 32 + 8 + 2]), "f"(_tmem_load_2[chunk * 32 + 8 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 12) * 4)), "f"(_tmem_load_2[chunk * 32 + 12]), "f"(_tmem_load_2[chunk * 32 + 12 + 1]), "f"(_tmem_load_2[chunk * 32 + 12 + 2]), "f"(_tmem_load_2[chunk * 32 + 12 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 16) * 4)), "f"(_tmem_load_2[chunk * 32 + 16]), "f"(_tmem_load_2[chunk * 32 + 16 + 1]), "f"(_tmem_load_2[chunk * 32 + 16 + 2]), "f"(_tmem_load_2[chunk * 32 + 16 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 20) * 4)), "f"(_tmem_load_2[chunk * 32 + 20]), "f"(_tmem_load_2[chunk * 32 + 20 + 1]), "f"(_tmem_load_2[chunk * 32 + 20 + 2]), "f"(_tmem_load_2[chunk * 32 + 20 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 24) * 4)), "f"(_tmem_load_2[chunk * 32 + 24]), "f"(_tmem_load_2[chunk * 32 + 24 + 1]), "f"(_tmem_load_2[chunk * 32 + 24 + 2]), "f"(_tmem_load_2[chunk * 32 + 24 + 3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(((buffer_index * 64 + my_row_1) * 32 + 28) * 4)), "f"(_tmem_load_2[chunk * 32 + 28]), "f"(_tmem_load_2[chunk * 32 + 28 + 1]), "f"(_tmem_load_2[chunk * 32 + 28 + 2]), "f"(_tmem_load_2[chunk * 32 + 28 + 3]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp_in_role_1 < 2 && my_row_1 < q_valid_1) {
                            const int chunk_base = chunk * 32;
                            int row_base = (buffer_index * 64 + my_row_1) * 32;
                            #pragma unroll
                            for (int offset = 0; offset < 32; offset += 4) {
                                float _partial_smem_reg_0[4];
                                {
                                    const float* _smem_ptr = (const float*)(smem_raw + 215552 + (unsigned int)((row_base + offset) * 4));
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 4; _lr++)
                                        _partial_smem_reg_0[_lr] = _smem_ptr[_lr];
                                }
                                #pragma unroll
                                for (int elem = 0; elem < 4; elem++) {
                                    _tmem_load_2[chunk_base + offset + elem] = _tmem_load_2[chunk_base + offset + elem] + _partial_smem_reg_0[elem];
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_2[chunk_base + offset + 0], _tmem_load_2[chunk_base + offset + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_2[chunk_base + offset + 2], _tmem_load_2[chunk_base + offset + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out + (output_row + chunk_base + offset)))[0]) = _pk2;
                                }
                            }
                        }
                    }
                    if (warp_in_role_1 < 2 && my_row_1 < q_valid_1 && return_lse != 0) {
                        int stat_idx = query * num_heads + head_1;
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(sum_total));
                        lse[stat_idx] = ((sum_total > 0.0f) ? max_total_safe * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_INF);
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tile_done_addr);
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_lastsplit_0 = 0;
            unsigned int _phase_tile_done_0 = 0;
            #pragma unroll 1
            for (int tile_iter_2 = 0; tile_iter_2 < tiles_per_cta; tile_iter_2++) {
                int tile_idx_2 = blockIdx.x + tile_iter_2 * gridDim.x;
                if (tile_idx_2 < total_tiles) {
                    int q_block_2 = tile_idx_2 % query_blocks;
                    int head_2 = tile_idx_2 / query_blocks;
                    int query_base_2 = q_block_2 * 64;
                    int q_valid_2 = sequence_q - query_base_2;
                    if (q_valid_2 > 64) {
                        q_valid_2 = 64;
                    }
                    if (q_valid_2 < 0) {
                        q_valid_2 = 0;
                    }
                    int row_id_2 = head_2 * query_blocks + q_block_2;
                    int row_begin_2 = row_id_2 * max_kv_blocks;
                    int selected_count_2 = q2k_num[row_id_2];
                    int group_count_2 = (selected_count_2 + 4 - 1) / 4;
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int first_pv = 1;
                    unsigned int k_stage = kv_stage;
                    unsigned int k_phase = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 4096);
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
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    if (group_count_2 == 1) {
                        elect_commit2(s_full_addr, q_empty_addr);
                    } else {
                        elect_commit(s_full_addr);
                    }
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    #pragma unroll 1
                    for (int group_index_1 = 0; group_index_1 < group_count_2 - 1; group_index_1++) {
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        k_stage = kv_stage;
                        k_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 4096);
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
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores), "r"(0));
                        if (group_index_1 + 2 == group_count_2) {
                            elect_commit2(s_full_addr, q_empty_addr);
                        } else {
                            elect_commit(s_full_addr);
                        }
                        elect_commit(kv_empty_addr + (k_stage) * 8);
                        unsigned int v_stage = kv_stage;
                        unsigned int v_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_2), "r"(tmem_probabilities), "r"(((first_pv) ? 0 : 1)));
                        mbarrier_wait(p_lastsplit_addr, _phase_p_lastsplit_0);
                        _phase_p_lastsplit_0 ^= 1;
                        int _mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_3), "r"(tmem_probabilities), "r"(1));
                        first_pv = 0;
                        elect_commit(p_empty_addr);
                        elect_commit(kv_empty_addr + (v_stage) * 8);
                    }
                    unsigned int final_v_stage = kv_stage;
                    unsigned int final_v_phase = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (final_v_stage) * 8, final_v_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_4 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (final_v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_4), "r"(tmem_probabilities), "r"(((first_pv) ? 0 : 1)));
                    mbarrier_wait(p_lastsplit_addr, _phase_p_lastsplit_0);
                    _phase_p_lastsplit_0 ^= 1;
                    int _mma_b_lo_5 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (final_v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_5), "r"(tmem_probabilities), "r"(1));
                    elect_commit2(o_full_addr, p_empty_addr);
                    elect_commit(kv_empty_addr + (final_v_stage) * 8);
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    _phase_s_empty_0 ^= 1;
                    mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                    _phase_tile_done_0 ^= 1;
                }
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            unsigned int load_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int tile_iter_3 = 0; tile_iter_3 < tiles_per_cta; tile_iter_3++) {
                int tile_idx_3 = blockIdx.x + tile_iter_3 * gridDim.x;
                if (tile_idx_3 < total_tiles) {
                    int q_block_3 = tile_idx_3 % query_blocks;
                    int head_3 = tile_idx_3 / query_blocks;
                    int query_base_3 = q_block_3 * 64;
                    int q_valid_3 = sequence_q - query_base_3;
                    if (q_valid_3 > 64) {
                        q_valid_3 = 64;
                    }
                    if (q_valid_3 < 0) {
                        q_valid_3 = 0;
                    }
                    int row_id_3 = head_3 * query_blocks + q_block_3;
                    int row_begin_3 = row_id_3 * max_kv_blocks;
                    int selected_count_3 = q2k_num[row_id_3];
                    int group_count_3 = (selected_count_3 + 4 - 1) / 4;
                    mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                    _phase_q_empty_0 ^= 1;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr, 16384);
                        tma_4d_gmem2smem(q_smem_addr, q, 0, head_3, query_base_3, 0, q_full_addr);
                    }
                    int entry0 = 0;
                    int entry1 = entry0 + 1;
                    int entry2 = entry0 + 2;
                    int entry3 = entry0 + 3;
                    int block0 = q2k_indices[row_begin_3 + entry0];
                    int block1 = block0;
                    if (entry1 < selected_count_3) {
                        block1 = q2k_indices[row_begin_3 + entry1];
                    }
                    int block2 = block1;
                    if (entry2 < selected_count_3) {
                        block2 = q2k_indices[row_begin_3 + entry2];
                    }
                    int block3 = block2;
                    if (entry3 < selected_count_3) {
                        block3 = q2k_indices[row_begin_3 + entry3];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, k, 0, block0 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, k, 0, block0 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, k, 0, block2 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, k, 0, block2 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, k, 0, block1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, k, 0, block1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, k, 0, block3 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, k, 0, block3 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    if (group_count_3 > 1) {
                        int entry0_0 = 4;
                        int entry1_1 = entry0_0 + 1;
                        int entry2_2 = entry0_0 + 2;
                        int entry3_3 = entry0_0 + 3;
                        int block0_4 = q2k_indices[row_begin_3 + entry0_0];
                        int block1_5 = block0_4;
                        if (entry1_1 < selected_count_3) {
                            block1_5 = q2k_indices[row_begin_3 + entry1_1];
                        }
                        int block2_6 = block1_5;
                        if (entry2_2 < selected_count_3) {
                            block2_6 = q2k_indices[row_begin_3 + entry2_2];
                        }
                        int block3_7 = block2_6;
                        if (entry3_3 < selected_count_3) {
                            block3_7 = q2k_indices[row_begin_3 + entry3_3];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, k, 0, block0_4 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, k, 0, block0_4 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, k, 0, block2_6 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, k, 0, block2_6 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, k, 0, block1_5 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, k, 0, block1_5 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, k, 0, block3_7 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, k, 0, block3_7 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        #pragma unroll 1
                        for (int group_index_2 = 0; group_index_2 < group_count_3 - 2; group_index_2++) {
                            int entry0_1 = group_index_2 * 4;
                            int entry1_2 = entry0_1 + 1;
                            int entry2_3 = entry0_1 + 2;
                            int entry3_4 = entry0_1 + 3;
                            int block0_5 = q2k_indices[row_begin_3 + entry0_1];
                            int block1_6 = block0_5;
                            if (entry1_2 < selected_count_3) {
                                block1_6 = q2k_indices[row_begin_3 + entry1_2];
                            }
                            int block2_7 = block1_6;
                            if (entry2_3 < selected_count_3) {
                                block2_7 = q2k_indices[row_begin_3 + entry2_3];
                            }
                            int block3_8 = block2_7;
                            if (entry3_4 < selected_count_3) {
                                block3_8 = q2k_indices[row_begin_3 + entry3_4];
                            }
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, v, 0, block0_5 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, v, 0, block0_5 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, v, 0, block2_7 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, v, 0, block2_7 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, v, 0, block1_6 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, v, 0, block1_6 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, v, 0, block3_8 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, v, 0, block3_8 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            }
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                            int entry0_9 = (group_index_2 + 2) * 4;
                            int entry1_10 = entry0_9 + 1;
                            int entry2_11 = entry0_9 + 2;
                            int entry3_12 = entry0_9 + 3;
                            int block0_13 = q2k_indices[row_begin_3 + entry0_9];
                            int block1_14 = block0_13;
                            if (entry1_10 < selected_count_3) {
                                block1_14 = q2k_indices[row_begin_3 + entry1_10];
                            }
                            int block2_15 = block1_14;
                            if (entry2_11 < selected_count_3) {
                                block2_15 = q2k_indices[row_begin_3 + entry2_11];
                            }
                            int block3_16 = block2_15;
                            if (entry3_12 < selected_count_3) {
                                block3_16 = q2k_indices[row_begin_3 + entry3_12];
                            }
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, k, 0, block0_13 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, k, 0, block0_13 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, k, 0, block2_15 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, k, 0, block2_15 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, k, 0, block1_14 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, k, 0, block1_14 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, k, 0, block3_16 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, k, 0, block3_16 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            }
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        int entry0_8 = (group_count_3 - 2) * 4;
                        int entry1_9 = entry0_8 + 1;
                        int entry2_10 = entry0_8 + 2;
                        int entry3_11 = entry0_8 + 3;
                        int block0_12 = q2k_indices[row_begin_3 + entry0_8];
                        int block1_13 = block0_12;
                        if (entry1_9 < selected_count_3) {
                            block1_13 = q2k_indices[row_begin_3 + entry1_9];
                        }
                        int block2_14 = block1_13;
                        if (entry2_10 < selected_count_3) {
                            block2_14 = q2k_indices[row_begin_3 + entry2_10];
                        }
                        int block3_15 = block2_14;
                        if (entry3_11 < selected_count_3) {
                            block3_15 = q2k_indices[row_begin_3 + entry3_11];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, v, 0, block0_12 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, v, 0, block0_12 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, v, 0, block2_14 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, v, 0, block2_14 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, v, 0, block1_13 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, v, 0, block1_13 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, v, 0, block3_15 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, v, 0, block3_15 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        int entry0_16 = (group_count_3 - 1) * 4;
                        int entry1_17 = entry0_16 + 1;
                        int entry2_18 = entry0_16 + 2;
                        int entry3_19 = entry0_16 + 3;
                        int block0_20 = q2k_indices[row_begin_3 + entry0_16];
                        int block1_21 = block0_20;
                        if (entry1_17 < selected_count_3) {
                            block1_21 = q2k_indices[row_begin_3 + entry1_17];
                        }
                        int block2_22 = block1_21;
                        if (entry2_18 < selected_count_3) {
                            block2_22 = q2k_indices[row_begin_3 + entry2_18];
                        }
                        int block3_23 = block2_22;
                        if (entry3_19 < selected_count_3) {
                            block3_23 = q2k_indices[row_begin_3 + entry3_19];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, v, 0, block0_20 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, v, 0, block0_20 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, v, 0, block2_22 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, v, 0, block2_22 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, v, 0, block1_21 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, v, 0, block1_21 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, v, 0, block3_23 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, v, 0, block3_23 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    } else {
                        int entry0_0_1 = 0;
                        int entry1_1_1 = entry0_0_1 + 1;
                        int entry2_2_1 = entry0_0_1 + 2;
                        int entry3_3_1 = entry0_0_1 + 3;
                        int block0_4_1 = q2k_indices[row_begin_3 + entry0_0_1];
                        int block1_5_1 = block0_4_1;
                        if (entry1_1_1 < selected_count_3) {
                            block1_5_1 = q2k_indices[row_begin_3 + entry1_1_1];
                        }
                        int block2_6_1 = block1_5_1;
                        if (entry2_2_1 < selected_count_3) {
                            block2_6_1 = q2k_indices[row_begin_3 + entry2_2_1];
                        }
                        int block3_7_1 = block2_6_1;
                        if (entry3_3_1 < selected_count_3) {
                            block3_7_1 = q2k_indices[row_begin_3 + entry3_3_1];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, v, 0, block0_4_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, v, 0, block0_4_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, v, 0, block2_6_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, v, 0, block2_6_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, v, 0, block1_5_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, v, 0, block1_5_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, v, 0, block3_7_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, v, 0, block3_7_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 10 && warp <= 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
