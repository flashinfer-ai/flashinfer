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
#define TMEM_NCOLS 384
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
#define SMEM_SMEM_SUM_OFF 197632
#define SMEM_SMEM_SUM_STAGE_BYTES 256
#define SMEM_SMEM_SUM_STRIDE 256
#define SMEM_SMEM_LSE_OFF 197888
#define SMEM_SMEM_LSE_STAGE_BYTES 256
#define SMEM_SMEM_LSE_STRIDE 256
#define SMEM_SMEM_INDEX_FLAGS_OFF 197632
#define SMEM_SMEM_INDEX_FLAGS_STAGE_BYTES 16
#define SMEM_SMEM_INDEX_FLAGS_STRIDE 16
#define SMEM_SMEM_INDICES_OFF 198144
#define SMEM_SMEM_INDICES_STAGE_BYTES 512
#define SMEM_SMEM_INDICES_STRIDE 512
#define SMEM_TOTAL 198656
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
kernel_cake_dsv4_bf16_h8_h32(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_swa_kv, CakeTensorMap const* tmap_compressed_kv, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int sparse_topk, int num_splits, int has_sinks)
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
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_sum_addr = smem + 197632;
    float* smem_lse = reinterpret_cast<float*>(smem_raw + 197888);
    const int smem_lse_addr = smem + 197888;
    int* smem_index_flags = reinterpret_cast<int*>(smem_raw + 197632);
    const int smem_index_flags_addr = smem + 197632;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 198144);
    const int smem_indices_addr = smem + 198144;

    // Mbarrier init (8 groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=4
            mbarrier_init(smem + 8, 4);
            mbarrier_init(smem + 16, 4);
            mbarrier_init(smem + 24, 4);
            mbarrier_init(smem + 32, 4);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // tmem_dealloc: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 0) {
        int _tmem_hold = smem + 112;
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
    #define p_full_addr (mbar_base + 80)
    #define sum_ready_addr (mbar_base + 88)
    #define o_done_addr (mbar_base + 96)
    #define tmem_dealloc_addr (mbar_base + 104)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            int split_work = blockIdx.x >> 2;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int active_topk = sparse_topk_lens[query_idx];
            const int warp_in_compute = warp;
            const int tmem_row_origin = warp_in_compute * 32;
            const int logical_row_origin = warp_in_compute * 16;
            const int my_row = logical_row_origin + lane % 16;
            const int col_half = lane / 16;
            unsigned int _phase_s_full_0 = 0;
            mbarrier_wait(s_full_addr, _phase_s_full_0);
            _phase_s_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
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
            int valid_cols = active_topk - split_idx * 128 - col_half * 64;
            valid_cols = ((valid_cols < 0) ? 0 : valid_cols);
            valid_cols = ((valid_cols > 64) ? 64 : valid_cols);
            int causal_cols = smem_index_flags[col_half * 2 + 1];
            valid_cols = ((valid_cols < causal_cols) ? valid_cols : causal_cols);
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
            float row_max = _tmem_load_0_max;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, row_max, 16);
            float _max_0 = max_noftz(row_max, _shfl_xor_0);
            row_max = _max_0;
            if (has_sinks != 0 && split_idx == 0 && my_row < num_heads) {
                float sink_unscaled = sinks[my_row] * 1.4426950408889634f / softmax_scale_log2;
                float _max_1 = max_noftz(row_max, sink_unscaled);
                row_max = _max_1;
            }
            float safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
            float max_scaled = safe_max * softmax_scale_log2;
            float2 _f2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
            float2 _f2_1 = make_float2(-max_scaled, -max_scaled);
            float2 _f2_2 = make_float2(0.0f, 0.0f);
            float2 sum0 = _f2_2;
            float2 _f2_3 = make_float2(0.0f, 0.0f);
            float2 sum1 = _f2_3;
            float2 _f2_4 = make_float2(0.0f, 0.0f);
            float2 sum2 = _f2_4;
            float2 _f2_5 = make_float2(0.0f, 0.0f);
            float2 sum3 = _f2_5;
            #pragma unroll
            for (int i = 0; i < 64; i += 8) {
                float2 _f2_6 = make_float2(_tmem_load_0[i], _tmem_load_0[i + 1]);
                float2 _f2_7 = make_float2(_tmem_load_0[i + 2], _tmem_load_0[i + 3]);
                float2 _f2_8 = make_float2(_tmem_load_0[i + 4], _tmem_load_0[i + 5]);
                float2 _f2_9 = make_float2(_tmem_load_0[i + 6], _tmem_load_0[i + 7]);
                float2 affine0 = fma_f32x2(_f2_6, _f2_0, _f2_1);
                float2 affine1 = fma_f32x2(_f2_7, _f2_0, _f2_1);
                float2 affine2 = fma_f32x2(_f2_8, _f2_0, _f2_1);
                float2 affine3 = fma_f32x2(_f2_9, _f2_0, _f2_1);
                float _exp2_0 = approx_exp2(affine0.x);
                float exp0 = _exp2_0;
                float _exp2_1 = approx_exp2(affine0.y);
                float exp1 = _exp2_1;
                float _exp2_2 = approx_exp2(affine1.x);
                float exp2 = _exp2_2;
                float _exp2_3 = approx_exp2(affine1.y);
                float exp3 = _exp2_3;
                float _exp2_4 = approx_exp2(affine2.x);
                float exp4 = _exp2_4;
                float _exp2_5 = approx_exp2(affine2.y);
                float exp5 = _exp2_5;
                float _exp2_6 = approx_exp2(affine3.x);
                float exp6 = _exp2_6;
                float _exp2_7 = approx_exp2(affine3.y);
                float exp7 = _exp2_7;
                _tmem_load_0[i] = exp0;
                _tmem_load_0[i + 1] = exp1;
                _tmem_load_0[i + 2] = exp2;
                _tmem_load_0[i + 3] = exp3;
                _tmem_load_0[i + 4] = exp4;
                _tmem_load_0[i + 5] = exp5;
                _tmem_load_0[i + 6] = exp6;
                _tmem_load_0[i + 7] = exp7;
                float2 _f2_10 = make_float2(exp0, exp1);
                sum0 = add_f32x2(sum0, _f2_10);
                float2 _f2_11 = make_float2(exp2, exp3);
                sum1 = add_f32x2(sum1, _f2_11);
                float2 _f2_12 = make_float2(exp4, exp5);
                sum2 = add_f32x2(sum2, _f2_12);
                float2 _f2_13 = make_float2(exp6, exp7);
                sum3 = add_f32x2(sum3, _f2_13);
            }
            float2 sum01 = add_f32x2(sum0, sum1);
            float2 sum23 = add_f32x2(sum2, sum3);
            float2 row_sum_pair = add_f32x2(sum01, sum23);
            float row_sum = row_sum_pair.x + row_sum_pair.y;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 16);
            row_sum = row_sum + _shfl_xor_1;
            if (has_sinks != 0 && split_idx == 0 && my_row < num_heads) {
                float _exp2_8 = approx_exp2(sinks[my_row] * 1.4426950408889634f - max_scaled);
                row_sum = row_sum + _exp2_8;
            }
            unsigned int packed_p[32];
            #pragma unroll
            for (int _lp = 0; _lp < 32; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                packed_p[_lp] = *(uint32_t*)&_bf2;
            }
            int p_addr = taddr + 128 + (unsigned int)(tmem_row_origin << 16);
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
                smem_sum[my_row] = row_sum;
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                smem_lse[my_row] = ((row_sum > 0.0f) ? max_scaled + _log2_0 : -CAKE_INF);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(p_full_addr);
            mbarrier_arrive(sum_ready_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // epilogue_main
            float output_scale = bmm2_scale[0];
            int work_idx = blockIdx.x;
            int v_chunk = work_idx & 3;
            int split_work_1 = work_idx >> 2;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            const int warp_in_role = warp - 4;
            const int tmem_row_origin_1 = warp_in_role * 32;
            const int logical_row_origin_1 = warp_in_role * 16;
            const int my_row_1 = logical_row_origin_1 + lane % 16;
            const int col_half_1 = lane / 16;
            const int row_addr = tmem_row_origin_1 << 16;
            unsigned int _phase_o_done_0 = 0;
            mbarrier_wait(o_done_addr, _phase_o_done_0);
            _phase_o_done_0 ^= 1;
            unsigned int _phase_sum_ready_0 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
            _phase_sum_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float row_sum_1 = smem_sum[my_row_1];
            float _rcp_0 = approx_rcp(row_sum_1);
            float inv_sum = ((row_sum_1 > 0.0f) ? _rcp_0 : 0.0f);
            int output_base = ((query_idx_1 * num_heads + my_row_1) * num_splits + split_idx_1) * 512 + v_chunk * 128;
            float _tmem_load_1[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                : "r"(taddr + 256 + (unsigned int)row_addr)
                : "memory");
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                : "r"(taddr + 256 + (unsigned int)row_addr + 32)
                : "memory");
            int col_base = col_half_1 * 64;
            if (my_row_1 < num_heads) {
                #pragma unroll
                for (int offset = 0; offset < 64; offset += 8) {
                    {
                        const float2 _prescale2_0 = {inv_sum * output_scale, inv_sum * output_scale};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_1[offset])[_ps], _prescale2_0);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_1[offset + _ps] *= inv_sum * output_scale;
                        #endif
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_1[offset + 0], _tmem_load_1[offset + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_1[offset + 2], _tmem_load_1[offset + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_1[offset + 4], _tmem_load_1[offset + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_1[offset + 6], _tmem_load_1[offset + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + (output_base + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
                if (v_chunk == 0 && col_half_1 == 0) {
                    int lse_offset = (query_idx_1 * num_heads + my_row_1) * num_splits + split_idx_1;
                    partial_lse[lse_offset] = smem_lse[my_row_1];
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_kv_full = 0;
            #pragma unroll
            for (int k_stage = 0; k_stage < 4; k_stage++) {
                mbarrier_wait(kv_full_addr + (k_stage) * 8, _phase_kv_full);
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem), "r"(((k_stage == 0) ? 0 : 1)));
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_tmem), "r"(1));
                if (k_stage == 3) {
                    elect_commit(s_full_addr);
                }
                elect_commit(kv_empty_addr + (k_stage) * 8);
            }
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(kv_full_addr, 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 2048);
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
                    :: "r"((tmem_tmem + (256))), "r"(_mma_b_lo_2), "r"(tmem_tmem + 128), "r"(0));
            elect_commit(o_done_addr);
            elect_commit(kv_empty_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: empty ----
    if (warp >= 9 && warp <= 11) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // load_warp_main
            const int load_warp_rank = warp - 12;
            int work_idx_1 = blockIdx.x;
            int v_chunk_1 = work_idx_1 & 3;
            int split_work_2 = work_idx_1 >> 2;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int sparse_base = query_idx_2 * sparse_topk;
            if (load_warp_rank == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    #pragma unroll
                    for (int q_stage = 0; q_stage < 8; q_stage++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 8192), tmap_q, 0, 0, q_stage, query_idx_2, q_full_addr);
                    }
                }
            }
            int index_offset = load_warp_rank * 32 + lane;
            int global_index_offset = split_idx_2 * 128 + index_offset;
            int sparse_row = -1;
            if (global_index_offset < sparse_topk) {
                sparse_row = sparse_indices[sparse_base + global_index_offset];
            }
            smem_indices[index_offset] = sparse_row;
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, sparse_row < 0);
            unsigned int invalid_indices = _vote_0;
            int valid_prefix = 32;
            if (invalid_indices != 0) {
                int _ffs_0 = __ffs(invalid_indices);
                valid_prefix = _ffs_0 - 1;
            }
            if (elect_sync()) {
                smem_index_flags[load_warp_rank] = valid_prefix;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if ((load_warp_rank & 1) != 0) {
                if (elect_sync()) {
                    int lower_prefix = smem_index_flags[load_warp_rank - 1];
                    int upper_prefix = smem_index_flags[load_warp_rank];
                    smem_index_flags[load_warp_rank] = ((lower_prefix < 32) ? lower_prefix : 32 + upper_prefix);
                }
            }
            unsigned int _phase_kv_empty = 1;
            #pragma unroll
            for (int k_stage_1 = 0; k_stage_1 < 4; k_stage_1++) {
                mbarrier_wait(kv_empty_addr + (k_stage_1) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (k_stage_1) * 8, 8192);
                }
                #pragma unroll
                for (int local_group = 0; local_group < 8; local_group++) {
                    int dst_k = smem_kv_addr + (unsigned int)(k_stage_1 * 32768);
                    int group = load_warp_rank * 8 + local_group;
                    int group_offset = group * 4;
                    int raw_rows[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                        : "r"(smem_indices_addr + (unsigned int)(group_offset * 4)));
                    int raw0 = raw_rows[0];
                    int raw1 = raw_rows[1];
                    int raw2 = raw_rows[2];
                    int raw3 = raw_rows[3];
                    int row0 = ((raw0 >= 0) ? raw0 : 0);
                    int row1 = ((raw1 >= 0) ? raw1 : 0);
                    int row2 = ((raw2 >= 0) ? raw2 : 0);
                    int row3 = ((raw3 >= 0) ? raw3 : 0);
                    if (elect_sync()) {
                        if (split_idx_2 == 0) {
                            tma_gather4_gmem2smem(dst_k + group * 512, tmap_swa_kv, k_stage_1 * 128, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                            tma_gather4_gmem2smem(dst_k + 16384 + group * 512, tmap_swa_kv, k_stage_1 * 128 + 64, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                        } else {
                            tma_gather4_gmem2smem(dst_k + group * 512, tmap_compressed_kv, k_stage_1 * 128, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                            tma_gather4_gmem2smem(dst_k + 16384 + group * 512, tmap_compressed_kv, k_stage_1 * 128 + 64, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                        }
                    }
                }
            }
            #pragma unroll
            for (int v_stage = 0; v_stage < 1; v_stage++) {
                mbarrier_wait(kv_empty_addr + (v_stage) * 8, 0);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (v_stage) * 8, 8192);
                }
                #pragma unroll
                for (int local_group_1 = 0; local_group_1 < 8; local_group_1++) {
                    int dst_v = smem_v_addr + (unsigned int)(v_stage * 32768);
                    int v_col = v_chunk_1 * 128 + v_stage * 64;
                    int group_1 = load_warp_rank * 8 + local_group_1;
                    int group_offset_1 = group_1 * 4;
                    int raw_rows_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                        : "r"(smem_indices_addr + (unsigned int)(group_offset_1 * 4)));
                    int raw0_1 = raw_rows_1[0];
                    int raw1_1 = raw_rows_1[1];
                    int raw2_1 = raw_rows_1[2];
                    int raw3_1 = raw_rows_1[3];
                    int row0_1 = ((raw0_1 >= 0) ? raw0_1 : 0);
                    int row1_1 = ((raw1_1 >= 0) ? raw1_1 : 0);
                    int row2_1 = ((raw2_1 >= 0) ? raw2_1 : 0);
                    int row3_1 = ((raw3_1 >= 0) ? raw3_1 : 0);
                    if (elect_sync()) {
                        if (split_idx_2 == 0) {
                            tma_gather4_gmem2smem(dst_v + group_1 * 512, tmap_swa_kv, v_col, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (v_stage) * 8);
                            tma_gather4_gmem2smem(dst_v + 16384 + group_1 * 512, tmap_swa_kv, v_col + 64, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (v_stage) * 8);
                        } else {
                            tma_gather4_gmem2smem(dst_v + group_1 * 512, tmap_compressed_kv, v_col, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (v_stage) * 8);
                            tma_gather4_gmem2smem(dst_v + 16384 + group_1 * 512, tmap_compressed_kv, v_col + 64, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (v_stage) * 8);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

