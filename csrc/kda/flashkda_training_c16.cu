typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

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

__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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

__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 272
#define TMEM_TMEM_STATE_OFFSET 0
#define TMEM_TMEM_STATE_INP_OFFSET 128
#define TMEM_TMEM_Q_STATE_OFFSET 192
#define TMEM_TMEM_STATE_K_OFFSET 224
#define TMEM_TMEM_U_ACC_OFFSET 240
#define TMEM_TMEM_Y_INP_OFFSET 256
#define TMEM_TMEM_U_INP_OFFSET 264
#define NUM_SCHED_PIPE_STAGES 8
#define NUM_RAW_PIPE_STAGES 5
#define NUM_RAW_BAR_PIPE_STAGES 6
#define NUM_DECAY_PIPE_STAGES 2
#define NUM_INTERMEDIATE_PIPE_STAGES 2
#define NUM_DIAG_PIPE_STAGES 4
#define NUM_STATE_PIPE_STAGES 1
#define NUM_O_PIPE_STAGES 2
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 4096
#define SMEM_SMEM_Q_STRIDE 4096
#define SMEM_SMEM_K_OFF 21504
#define SMEM_SMEM_K_STAGE_BYTES 4096
#define SMEM_SMEM_K_STRIDE 4096
#define SMEM_SMEM_V_OFF 41984
#define SMEM_SMEM_V_STAGE_BYTES 4096
#define SMEM_SMEM_V_STRIDE 4096
#define SMEM_SMEM_G_OFF 62464
#define SMEM_SMEM_G_STAGE_BYTES 8192
#define SMEM_SMEM_G_STRIDE 8192
#define SMEM_SMEM_BETA_OFF 103424
#define SMEM_SMEM_BETA_STAGE_BYTES 32
#define SMEM_SMEM_BETA_STRIDE 32
#define SMEM_SMEM_BETA_ALL_OFF 103424
#define SMEM_SMEM_BETA_ALL_STAGE_BYTES 192
#define SMEM_SMEM_BETA_ALL_STRIDE 192
#define SMEM_SMEM_K_INV_OFF 104448
#define SMEM_SMEM_K_INV_STAGE_BYTES 4096
#define SMEM_SMEM_K_INV_STRIDE 4096
#define SMEM_SMEM_K_DECAY_OFF 112640
#define SMEM_SMEM_K_DECAY_STAGE_BYTES 4096
#define SMEM_SMEM_K_DECAY_STRIDE 4096
#define SMEM_SMEM_Q_DECAY_OFF 120832
#define SMEM_SMEM_Q_DECAY_STAGE_BYTES 4096
#define SMEM_SMEM_Q_DECAY_STRIDE 4096
#define SMEM_SMEM_K_RESTORE_OFF 129024
#define SMEM_SMEM_K_RESTORE_STAGE_BYTES 4096
#define SMEM_SMEM_K_RESTORE_STRIDE 4096
#define SMEM_SMEM_K_RESTORE_MN_OFF 129024
#define SMEM_SMEM_K_RESTORE_MN_STAGE_BYTES 4096
#define SMEM_SMEM_K_RESTORE_MN_STRIDE 4096
#define SMEM_SMEM_TINV_OFF 137216
#define SMEM_SMEM_TINV_STAGE_BYTES 512
#define SMEM_SMEM_TINV_STRIDE 1024
#define SMEM_SMEM_A_OFF 137728
#define SMEM_SMEM_A_STAGE_BYTES 512
#define SMEM_SMEM_A_STRIDE 1024
#define SMEM_SMEM_TINV_MN_OFF 137216
#define SMEM_SMEM_TINV_MN_STAGE_BYTES 512
#define SMEM_SMEM_TINV_MN_STRIDE 1024
#define SMEM_SMEM_A_MN_OFF 137728
#define SMEM_SMEM_A_MN_STAGE_BYTES 512
#define SMEM_SMEM_A_MN_STRIDE 1024
#define SMEM_SMEM_STATE_DIAG_OFF 139264
#define SMEM_SMEM_STATE_DIAG_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG_STRIDE 512
#define SMEM_SMEM_O_OFF 155648
#define SMEM_SMEM_O_STAGE_BYTES 4096
#define SMEM_SMEM_O_STRIDE 4096
#define SMEM_SMEM_CHECKPOINT_OFF 163840
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_TINV_SCRATCH_OFF 229376
#define SMEM_TINV_SCRATCH_STAGE_BYTES 512
#define SMEM_TINV_SCRATCH_STRIDE 512
#define SMEM_SCHED_SLOT_OFF 229888
#define SMEM_SCHED_SLOT_STAGE_BYTES 4
#define SMEM_SCHED_SLOT_STRIDE 4
#define SMEM_SMEM_Q_ALL_OFF 1024
#define SMEM_SMEM_Q_ALL_STAGE_BYTES 20480
#define SMEM_SMEM_Q_ALL_STRIDE 20480
#define SMEM_SMEM_K_ALL_OFF 21504
#define SMEM_SMEM_K_ALL_STAGE_BYTES 20480
#define SMEM_SMEM_K_ALL_STRIDE 20480
#define SMEM_SMEM_G_ALL_OFF 62464
#define SMEM_SMEM_G_ALL_STAGE_BYTES 40960
#define SMEM_SMEM_G_ALL_STRIDE 40960
#define SMEM_SMEM_V_ALL_OFF 41984
#define SMEM_SMEM_V_ALL_STAGE_BYTES 20480
#define SMEM_SMEM_V_ALL_STRIDE 20480
#define SMEM_SMEM_O_ALL_OFF 155648
#define SMEM_SMEM_O_ALL_STAGE_BYTES 8192
#define SMEM_SMEM_O_ALL_STRIDE 8192
#define SMEM_TOTAL 230016
#define THREADS 512
#define USE_INITIAL_STATE 1
#define STORE_FINAL_STATE 1
#define ENABLE_CHECKPOINTS 1
#define STORE_BETA_ACTIVE 1
#define G_INPUT_BF16 1


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_flashkda_forward_checkpoint_c16(unsigned int* __restrict__ dynamic_counter, const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap g_tma, __nv_bfloat16* __restrict__ g, const __grid_constant__ CUtensorMap out_tma, const __grid_constant__ CUtensorMap checkpoint_tma, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ beta_active_out, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, long long* __restrict__ checkpoint_cu_starts, int* __restrict__ work_items, float* __restrict__ initial_state, float* __restrict__ final_state, int total_work_items, int uniform_work_items, int num_qk_heads, int num_heads, int beta_active_stride, int checkpoint_every_n_tokens, float scale, float lower_bound)
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
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_k_addr = smem + 21504;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_v_addr = smem + 41984;
    float* smem_g = reinterpret_cast<float*>(smem_raw + 62464);
    const int smem_g_addr = smem + 62464;
    __nv_bfloat16* smem_beta = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int smem_beta_addr = smem + 103424;
    __nv_bfloat16* smem_beta_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int smem_beta_all_addr = smem + 103424;
    __nv_bfloat16* smem_k_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 104448);
    const int smem_k_inv_addr = smem + 104448;
    __nv_bfloat16* smem_k_decay = reinterpret_cast<__nv_bfloat16*>(smem_raw + 112640);
    const int smem_k_decay_addr = smem + 112640;
    __nv_bfloat16* smem_q_decay = reinterpret_cast<__nv_bfloat16*>(smem_raw + 120832);
    const int smem_q_decay_addr = smem + 120832;
    __nv_bfloat16* smem_k_restore = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_k_restore_addr = smem + 129024;
    __nv_bfloat16* smem_k_restore_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_k_restore_mn_addr = smem + 129024;
    __nv_bfloat16* smem_tinv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_tinv_addr = smem + 137216;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137728);
    const int smem_a_addr = smem + 137728;
    __nv_bfloat16* smem_tinv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_tinv_mn_addr = smem + 137216;
    __nv_bfloat16* smem_a_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137728);
    const int smem_a_mn_addr = smem + 137728;
    __nv_bfloat16* smem_state_diag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 139264);
    const int smem_state_diag_addr = smem + 139264;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 155648);
    const int smem_o_addr = smem + 155648;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int smem_checkpoint_addr = smem + 163840;
    __nv_bfloat16* tinv_scratch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 229376);
    const int tinv_scratch_addr = smem + 229376;
    unsigned int* sched_slot = reinterpret_cast<unsigned int*>(smem_raw + 229888);
    const int sched_slot_addr = smem + 229888;
    __nv_bfloat16* smem_q_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_all_addr = smem + 1024;
    __nv_bfloat16* smem_k_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_k_all_addr = smem + 21504;
    float* smem_g_all = reinterpret_cast<float*>(smem_raw + 62464);
    const int smem_g_all_addr = smem + 62464;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_v_all_addr = smem + 41984;
    __nv_bfloat16* smem_o_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 155648);
    const int smem_o_all_addr = smem + 155648;

    // Mbarrier init (37 groups, 116 barriers)
    // Mbarriers at smem_raw[0..928)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'sched_pipe' ---
            // sched_ready: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // sched_done: 8 barriers, init_count=15
            mbarrier_init(smem + 64, 15);
            mbarrier_init(smem + 72, 15);
            mbarrier_init(smem + 80, 15);
            mbarrier_init(smem + 88, 15);
            mbarrier_init(smem + 96, 15);
            mbarrier_init(smem + 104, 15);
            mbarrier_init(smem + 112, 15);
            mbarrier_init(smem + 120, 15);
            // --- pipeline 'raw_bar_pipe' ---
            // q_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // k_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // v_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // g_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // --- pipeline 'raw_pipe' ---
            // q_done: 5 barriers, init_count=128
            mbarrier_init(smem + 320, 128);
            mbarrier_init(smem + 328, 128);
            mbarrier_init(smem + 336, 128);
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // k_done: 5 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            mbarrier_init(smem + 368, 128);
            mbarrier_init(smem + 376, 128);
            mbarrier_init(smem + 384, 128);
            mbarrier_init(smem + 392, 128);
            // g_done: 5 barriers, init_count=128
            mbarrier_init(smem + 400, 128);
            mbarrier_init(smem + 408, 128);
            mbarrier_init(smem + 416, 128);
            mbarrier_init(smem + 424, 128);
            mbarrier_init(smem + 432, 128);
            // v_done: 5 barriers, init_count=128
            mbarrier_init(smem + 440, 128);
            mbarrier_init(smem + 448, 128);
            mbarrier_init(smem + 456, 128);
            mbarrier_init(smem + 464, 128);
            mbarrier_init(smem + 472, 128);
            // --- pipeline 'raw_bar_pipe' ---
            // beta_ready: 6 barriers, init_count=32
            mbarrier_init(smem + 480, 32);
            mbarrier_init(smem + 488, 32);
            mbarrier_init(smem + 496, 32);
            mbarrier_init(smem + 504, 32);
            mbarrier_init(smem + 512, 32);
            mbarrier_init(smem + 520, 32);
            // beta_done: 6 barriers, init_count=160
            mbarrier_init(smem + 528, 160);
            mbarrier_init(smem + 536, 160);
            mbarrier_init(smem + 544, 160);
            mbarrier_init(smem + 552, 160);
            mbarrier_init(smem + 560, 160);
            mbarrier_init(smem + 568, 160);
            // --- pipeline 'decay_pipe' ---
            // k_decay_inv_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 576, 128);
            mbarrier_init(smem + 584, 128);
            // --- pipeline 'diag_pipe' ---
            // qk_scale_ready: 4 barriers, init_count=128
            mbarrier_init(smem + 592, 128);
            mbarrier_init(smem + 600, 128);
            mbarrier_init(smem + 608, 128);
            mbarrier_init(smem + 616, 128);
            // --- pipeline 'decay_pipe' ---
            // decay_tcgen_done: 2 barriers, init_count=1
            mbarrier_init(smem + 624, 1);
            mbarrier_init(smem + 632, 1);
            // decay_super_done: 2 barriers, init_count=64
            mbarrier_init(smem + 640, 64);
            mbarrier_init(smem + 648, 64);
            // k_restore_done: 2 barriers, init_count=1
            mbarrier_init(smem + 656, 1);
            mbarrier_init(smem + 664, 1);
            // --- pipeline 'diag_pipe' ---
            // state_diag_done: 4 barriers, init_count=1
            mbarrier_init(smem + 672, 1);
            mbarrier_init(smem + 680, 1);
            mbarrier_init(smem + 688, 1);
            mbarrier_init(smem + 696, 1);
            // --- pipeline 'intermediate_pipe' ---
            // tinv_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 704, 32);
            mbarrier_init(smem + 712, 32);
            // tinv_done: 2 barriers, init_count=1
            mbarrier_init(smem + 720, 1);
            mbarrier_init(smem + 728, 1);
            // a_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 736, 32);
            mbarrier_init(smem + 744, 32);
            // a_done: 2 barriers, init_count=1
            mbarrier_init(smem + 752, 1);
            mbarrier_init(smem + 760, 1);
            // --- pipeline 'state_pipe' ---
            // state_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 768, 128);
            // state_read_done: 1 barriers, init_count=128
            mbarrier_init(smem + 776, 128);
            // state_acc_done: 1 barriers, init_count=1
            mbarrier_init(smem + 784, 1);
            // state_k_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 792, 1);
            // y_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 800, 128);
            // u_acc_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 808, 1);
            // u_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 816, 128);
            // o_acc_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 824, 1);
            // --- pipeline 'o_pipe' ---
            // o_acc_done: 2 barriers, init_count=128
            mbarrier_init(smem + 832, 128);
            mbarrier_init(smem + 840, 128);
            // o_tma_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 848, 128);
            mbarrier_init(smem + 856, 128);
            // o_tma_done: 2 barriers, init_count=32
            mbarrier_init(smem + 864, 32);
            mbarrier_init(smem + 872, 32);
            // --- pipeline 'checkpoint_pipe' ---
            // checkpoint_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 880, 128);
            mbarrier_init(smem + 888, 128);
            // checkpoint_done: 2 barriers, init_count=32
            mbarrier_init(smem + 896, 32);
            mbarrier_init(smem + 904, 32);
            // consumers_done: 1 barriers, init_count=15
            mbarrier_init(smem + 912, 15);
            // cleanup_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 920, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 928);
    if (warp == 13) {
        int _tmem_hold = smem + 928;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define sched_ready_addr (mbar_base + 0)
    #define sched_done_addr (mbar_base + 64)
    #define q_ready_addr (mbar_base + 128)
    #define k_ready_addr (mbar_base + 176)
    #define v_ready_addr (mbar_base + 224)
    #define g_ready_addr (mbar_base + 272)
    #define q_done_addr (mbar_base + 320)
    #define k_done_addr (mbar_base + 360)
    #define g_done_addr (mbar_base + 400)
    #define v_done_addr (mbar_base + 440)
    #define beta_ready_addr (mbar_base + 480)
    #define beta_done_addr (mbar_base + 528)
    #define k_decay_inv_ready_addr (mbar_base + 576)
    #define qk_scale_ready_addr (mbar_base + 592)
    #define decay_tcgen_done_addr (mbar_base + 624)
    #define decay_super_done_addr (mbar_base + 640)
    #define k_restore_done_addr (mbar_base + 656)
    #define state_diag_done_addr (mbar_base + 672)
    #define tinv_ready_addr (mbar_base + 704)
    #define tinv_done_addr (mbar_base + 720)
    #define a_ready_addr (mbar_base + 736)
    #define a_done_addr (mbar_base + 752)
    #define state_inp_ready_addr (mbar_base + 768)
    #define state_read_done_addr (mbar_base + 776)
    #define state_acc_done_addr (mbar_base + 784)
    #define state_k_ready_addr (mbar_base + 792)
    #define y_inp_ready_addr (mbar_base + 800)
    #define u_acc_ready_addr (mbar_base + 808)
    #define u_inp_ready_addr (mbar_base + 816)
    #define o_acc_ready_addr (mbar_base + 824)
    #define o_acc_done_addr (mbar_base + 832)
    #define o_tma_ready_addr (mbar_base + 848)
    #define o_tma_done_addr (mbar_base + 864)
    #define checkpoint_ready_addr (mbar_base + 880)
    #define checkpoint_done_addr (mbar_base + 896)
    #define consumers_done_addr (mbar_base + 912)
    #define cleanup_ready_addr (mbar_base + 920)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr;
    const int tmem_tmem_state_inp = taddr + 128;
    const int tmem_tmem_q_state = taddr + 192;
    const int tmem_tmem_state_k = taddr + 224;
    const int tmem_tmem_u_acc = taddr + 240;
    const int tmem_tmem_y_inp = taddr + 256;
    const int tmem_tmem_u_inp = taddr + 264;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: cg0 ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // cg0_main
            unsigned int sched_stage_cg0 = 0;
            int cumulative_chunk_cg0 = 0;
            int instance_id = (warp - 0) / 4;
            int cg0_instance = instance_id;
            int warp_id_in_role = (warp - 0);
            int cg0_local_warp = warp_id_in_role - cg0_instance * 4;
            int cg0_tid = cg0_local_warp * 32 + lane;
            unsigned int _phase_sched_ready = 0;
            #pragma unroll 1
            for (int _ = 0; _ < total_work_items + 1; _++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_cg0) * 8, _phase_sched_ready);
                unsigned int slot[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot[0])) : "r"(sched_slot_addr + sched_stage_cg0 * 4));
                unsigned int tile_cg0 = slot[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_cg0) * 8);
                }
                sched_stage_cg0 += 1;
                if (sched_stage_cg0 == 8) { sched_stage_cg0 = 0; _phase_sched_ready ^= 1; }
                if (tile_cg0 >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_cg0 = (int)tile_cg0 * 8;
                int _vec_load_2[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_cg0 + 4);
                    _vec_load_2[0 + 0] = _iv4.x;
                    _vec_load_2[0 + 1] = _iv4.y;
                    _vec_load_2[0 + 2] = _iv4.z;
                    _vec_load_2[0 + 3] = _iv4.w;
                }
                int head_cg0 = work_items[item_base_cg0 + 1];
                int wend_cg0 = work_items[item_base_cg0 + 3];
                int cstart_cg0 = _vec_load_2[0];
                long long bos_cg0 = (long long)_vec_load_2[2];
                long long eos_cg0 = (long long)_vec_load_2[3];
                int chunks_cg0 = wend_cg0 - cstart_cg0;
                float _expf_0 = __expf(A_log[head_cg0]);
                float gate_rate_cg0 = _expf_0;
                float gate_bias_cg0 = dt_bias[head_cg0 * 128 + cg0_tid];
                asm volatile("barrier.sync 10, 256;" ::: "memory");
                int first_cumulative_cg0 = cumulative_chunk_cg0 + cg0_instance;
                int raw_stage_cg0 = first_cumulative_cg0 % 5;
                int raw_bar_stage_cg0 = first_cumulative_cg0 % 6;
                int decay_stage_cg0 = first_cumulative_cg0 % 2;
                int diag_stage_cg0 = first_cumulative_cg0 % 4;
                int raw_bar_phase_cg0 = first_cumulative_cg0 / 6 & 1;
                int decay_free_phase_cg0 = first_cumulative_cg0 / 2 + 1 & 1;
                int diag_free_phase_cg0 = first_cumulative_cg0 / 4 + 1 & 1;
                #pragma unroll 1
                for (int chunk_cg0 = cg0_instance; chunk_cg0 < chunks_cg0; chunk_cg0 += 2) {
                    int cumulative_cg0 = cumulative_chunk_cg0 + chunk_cg0;
                    int logical_chunk_cg0 = cstart_cg0 + chunk_cg0;
                    if (cg0_local_warp == 0) {
                        mbarrier_wait(beta_done_addr + (raw_bar_stage_cg0) * 8, (unsigned int)(cumulative_cg0 / 6 + 1 & 1));
                        if (lane < 16) {
                            long long beta_token_cg0 = bos_cg0 + (long long)logical_chunk_cg0 * 16 + (long long)lane;
                            float beta_value_cg0 = 0.0f;
                            if (beta_token_cg0 < eos_cg0) {
                                float beta_logit_cg0 = (float)beta[beta_token_cg0 * (long long)num_heads + (long long)head_cg0];
                                float _tanh_approx_0;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(beta_logit_cg0 * 0.5f));
                                beta_value_cg0 = _tanh_approx_0 * 0.5f + 0.5f;
                            }
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(beta_value_cg0);
                            __nv_bfloat16 beta_active_cg0 = _cvt_bf16_0;
                            smem_beta_all[(int)raw_bar_stage_cg0 * 16 + lane] = beta_active_cg0;
                            if (STORE_BETA_ACTIVE != 0 && beta_token_cg0 < eos_cg0) {
                                beta_active_out[beta_token_cg0 * (long long)beta_active_stride + (long long)head_cg0] = beta_active_cg0;
                            }
                        }
                        mbarrier_arrive(beta_ready_addr + (raw_bar_stage_cg0) * 8);
                    }
                    mbarrier_wait(g_ready_addr + (raw_bar_stage_cg0) * 8, raw_bar_phase_cg0);
                    float gate_raw_cg0[16];
                    float gate_log_cg0[16];
                    float gate_prefix_regs_cg0[16];
                    #pragma unroll
                    for (int token_gate_cg0 = 0; token_gate_cg0 < 16; token_gate_cg0++) {
                        {
                            long long gate_token_cg0 = bos_cg0 + (long long)logical_chunk_cg0 * 16 + (long long)token_gate_cg0;
                            gate_raw_cg0[token_gate_cg0] = (float)g[(gate_token_cg0 * (long long)num_heads + (long long)head_cg0) * 128 + (long long)cg0_tid];
                        }
                    }
                    #pragma unroll
                    for (int gate_row_cg0 = 0; gate_row_cg0 < 16; gate_row_cg0++) {
                        float gate_arg_cg0 = gate_rate_cg0 * (gate_raw_cg0[gate_row_cg0] + gate_bias_cg0);
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(gate_arg_cg0 * 0.5f));
                        float gate_value_cg0 = _tanh_approx_1 * 0.5f + 0.5f;
                        long long valid_gate_cg0 = bos_cg0 + (long long)logical_chunk_cg0 * 16 + (long long)gate_row_cg0;
                        gate_log_cg0[gate_row_cg0] = 0.0f;
                        if (valid_gate_cg0 < eos_cg0) {
                            gate_log_cg0[gate_row_cg0] = lower_bound * 1.4426950408889634f * gate_value_cg0;
                        }
                    }
                    float gate_prefix_cg0 = 0.0f;
                    #pragma unroll
                    for (int gate_pair_idx_cg0 = 0; gate_pair_idx_cg0 < 8; gate_pair_idx_cg0++) {
                        int gate_row0_cg0 = gate_pair_idx_cg0 * 2;
                        int gate_row1_cg0 = gate_row0_cg0 + 1;
                        float2 _f2_0 = make_float2(gate_prefix_cg0, gate_log_cg0[gate_row0_cg0]);
                        float2 _f2_1 = make_float2(gate_log_cg0[gate_row0_cg0], gate_log_cg0[gate_row1_cg0]);
                        float2 gate_pair_sum_cg0 = add_f32x2(_f2_0, _f2_1);
                        gate_prefix_regs_cg0[gate_row0_cg0] = gate_pair_sum_cg0.x;
                        gate_prefix_cg0 += gate_pair_sum_cg0.y;
                        gate_prefix_regs_cg0[gate_row1_cg0] = gate_prefix_cg0;
                    }
                    float gate_last_cg0 = 1.0f;
                    #pragma unroll
                    for (int token_gate_cg0_1 = 0; token_gate_cg0_1 < 16; token_gate_cg0_1++) {
                        float _exp2_0 = approx_exp2(gate_prefix_regs_cg0[token_gate_cg0_1]);
                        gate_last_cg0 = _exp2_0;
                        int segment = cg0_tid / 32;
                        int segment_col = cg0_tid - segment * 32;
                        int swizzled_col = segment_col ^ (token_gate_cg0_1 & 7) * 4;
                        smem_g_all[raw_stage_cg0 * 16 * 128 + segment * 16 * 32 + token_gate_cg0_1 * 32 + swizzled_col] = gate_last_cg0;
                    }
                    mbarrier_wait(state_diag_done_addr + (diag_stage_cg0) * 8, diag_free_phase_cg0);
                    int diag_block_cg0 = cg0_tid / 16;
                    int diag_coord_cg0 = cg0_tid - diag_block_cg0 * 16;
                    int diag_storage_stage_cg0 = (int)diag_stage_cg0 * 8 + diag_block_cg0;
                    if (cumulative_cg0 < 4) {
                        #pragma unroll
                        for (int diag_half_cg0 = 0; diag_half_cg0 < 2; diag_half_cg0++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_state_diag_addr + (unsigned int)(diag_storage_stage_cg0 * 512) + (unsigned int)(diag_half_cg0 * 8 / 16 * 512 + diag_coord_cg0 * 32 + diag_half_cg0 * 8 % 16 * 2 ^ (diag_half_cg0 * 8 / 16 * 512 + diag_coord_cg0 * 32 + diag_half_cg0 * 8 % 16 * 2 >> 7 & 1) << 4))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(gate_last_cg0);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        uint32_t _addr_0 = static_cast<uint32_t>((smem_state_diag_addr + (unsigned int)(diag_storage_stage_cg0 * 512) + (unsigned int)(diag_coord_cg0 / 16 * 512 + diag_coord_cg0 * 32 + diag_coord_cg0 % 16 * 2 ^ (diag_coord_cg0 / 16 * 512 + diag_coord_cg0 * 32 + diag_coord_cg0 % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                    if (cg0_instance == 0) {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    }
                    mbarrier_wait(q_ready_addr + (raw_bar_stage_cg0) * 8, raw_bar_phase_cg0);
                    mbarrier_wait(k_ready_addr + (raw_bar_stage_cg0) * 8, raw_bar_phase_cg0);
                    int decay_row_cg0 = cg0_local_warp * 4 + lane / 8;
                    int decay_lane_cg0 = lane & 7;
                    float q_values_cg0[16];
                    float k_values_cg0[16];
                    float2 _f2_2 = make_float2(0.0f, 0.0f);
                    float2 qk_sq_even_cg0 = _f2_2;
                    float2 _f2_3 = make_float2(0.0f, 0.0f);
                    float2 qk_sq_odd_cg0 = _f2_3;
                    #pragma unroll
                    for (int dim_half_cg0 = 0; dim_half_cg0 < 2; dim_half_cg0++) {
                        int dim_base_cg0 = dim_half_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int q_words_cg0[4];
                        unsigned int k_words_cg0[4];
                        int segment_1 = dim_base_cg0 / 64;
                        int segment_col_1 = dim_base_cg0 - segment_1 * 64;
                        int swizzled_col_1 = segment_col_1 ^ (decay_row_cg0 & 7) * 8;
                        int raw_index_cg0 = raw_stage_cg0 * 16 * 128 + segment_1 * 16 * 64 + decay_row_cg0 * 64 + swizzled_col_1;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_cg0[(0) + 3]))
                            : "r"(smem_q_all_addr + (unsigned int)(raw_index_cg0 * 2)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_cg0[(0) + 3]))
                            : "r"(smem_k_all_addr + (unsigned int)(raw_index_cg0 * 2)));
                        float q_words_cg0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&q_words_cg0_f32[_pair * 2])[0]), "=f"((&q_words_cg0_f32[_pair * 2])[1])
                                : "r"(q_words_cg0[_pair]));
                        }
                        float k_words_cg0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&k_words_cg0_f32[_pair * 2])[0]), "=f"((&k_words_cg0_f32[_pair * 2])[1])
                                : "r"(k_words_cg0[_pair]));
                        }
                        #pragma unroll
                        for (int dim_local_cg0 = 0; dim_local_cg0 < 8; dim_local_cg0++) {
                            int reg_cg0 = dim_half_cg0 * 8 + dim_local_cg0;
                            q_values_cg0[reg_cg0] = q_words_cg0_f32[dim_local_cg0];
                            k_values_cg0[reg_cg0] = k_words_cg0_f32[dim_local_cg0];
                        }
                        #pragma unroll
                        for (int dim_pair_sq_cg0 = 0; dim_pair_sq_cg0 < 4; dim_pair_sq_cg0++) {
                            int even_reg_cg0 = dim_half_cg0 * 8 + dim_pair_sq_cg0 * 2;
                            int odd_reg_cg0 = even_reg_cg0 + 1;
                            float2 _f2_4 = make_float2(q_values_cg0[even_reg_cg0], k_values_cg0[even_reg_cg0]);
                            float2 qk_even_cg0 = _f2_4;
                            float2 _f2_5 = make_float2(q_values_cg0[odd_reg_cg0], k_values_cg0[odd_reg_cg0]);
                            float2 qk_odd_cg0 = _f2_5;
                            qk_sq_even_cg0 = fma_f32x2(qk_even_cg0, qk_even_cg0, qk_sq_even_cg0);
                            qk_sq_odd_cg0 = fma_f32x2(qk_odd_cg0, qk_odd_cg0, qk_sq_odd_cg0);
                        }
                    }
                    float q_sq_cg0 = qk_sq_even_cg0.x + qk_sq_odd_cg0.x;
                    float k_sq_cg0 = qk_sq_even_cg0.y + qk_sq_odd_cg0.y;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sq_cg0, 4);
                    q_sq_cg0 += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sq_cg0, 4);
                    k_sq_cg0 += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sq_cg0, 2);
                    q_sq_cg0 += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sq_cg0, 2);
                    k_sq_cg0 += _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sq_cg0, 1);
                    q_sq_cg0 += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sq_cg0, 1);
                    k_sq_cg0 += _shfl_xor_5;
                    float _max_0 = max_noftz(q_sq_cg0, 1e-12f);
                    float _rsqrt_0 = rsqrtf(_max_0);
                    float q_inv_norm_cg0 = _rsqrt_0;
                    float _max_1 = max_noftz(k_sq_cg0, 1e-12f);
                    float _rsqrt_1 = rsqrtf(_max_1);
                    float k_inv_norm_cg0 = _rsqrt_1;
                    float exp_g_regs_cg0[16];
                    float exp_g_last_regs_cg0[16];
                    #pragma unroll
                    for (int dim_half_prefix_cg0 = 0; dim_half_prefix_cg0 < 2; dim_half_prefix_cg0++) {
                        int dim_base_prefix_cg0 = dim_half_prefix_cg0 * 64 + decay_lane_cg0 * 8;
                        #pragma unroll
                        for (int f32_group_cg0 = 0; f32_group_cg0 < 2; f32_group_cg0++) {
                            int f32_dim_base_cg0 = dim_base_prefix_cg0 + f32_group_cg0 * 4;
                            unsigned int exp_g_words_cg0[4];
                            unsigned int exp_g_last_words_cg0[4];
                            int segment_2 = f32_dim_base_cg0 / 32;
                            int segment_col_2 = f32_dim_base_cg0 - segment_2 * 32;
                            int swizzled_col_2 = segment_col_2 ^ (decay_row_cg0 & 7) * 4;
                            int exp_g_index_cg0 = raw_stage_cg0 * 16 * 128 + segment_2 * 16 * 32 + decay_row_cg0 * 32 + swizzled_col_2;
                            int segment_0 = f32_dim_base_cg0 / 32;
                            int segment_col_1_1 = f32_dim_base_cg0 - segment_0 * 32;
                            int swizzled_col_2_1 = segment_col_1_1 ^ 28;
                            int exp_g_last_index_cg0 = raw_stage_cg0 * 16 * 128 + segment_0 * 16 * 32 + 480 + swizzled_col_2_1;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_words_cg0[(0) + 3]))
                                : "r"(smem_g_all_addr + (unsigned int)(exp_g_index_cg0 * 4)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&exp_g_last_words_cg0[(0) + 3]))
                                : "r"(smem_g_all_addr + (unsigned int)(exp_g_last_index_cg0 * 4)));
                            #pragma unroll
                            for (int prefix_word_cg0 = 0; prefix_word_cg0 < 4; prefix_word_cg0++) {
                                int prefix_reg_cg0 = dim_half_prefix_cg0 * 8 + f32_group_cg0 * 4 + prefix_word_cg0;
                                exp_g_regs_cg0[prefix_reg_cg0] = __uint_as_float(exp_g_words_cg0[prefix_word_cg0]);
                                exp_g_last_regs_cg0[prefix_reg_cg0] = __uint_as_float(exp_g_last_words_cg0[prefix_word_cg0]);
                            }
                        }
                    }
                    __nv_bfloat162 k_inv_pairs_all_cg0[8];
                    float2 _f2_6 = make_float2(k_inv_norm_cg0, k_inv_norm_cg0);
                    float2 k_inv_norm_pair_cg0 = _f2_6;
                    #pragma unroll
                    for (int dim_half_k_cg0 = 0; dim_half_k_cg0 < 2; dim_half_k_cg0++) {
                        int dim_base_k_cg0 = dim_half_k_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int k_decay_words_cg0[4];
                        #pragma unroll
                        for (int dim_pair_k_cg0 = 0; dim_pair_k_cg0 < 4; dim_pair_k_cg0++) {
                            int dim_local0_k_cg0 = dim_pair_k_cg0 * 2;
                            int dim_local1_k_cg0 = dim_local0_k_cg0 + 1;
                            int reg_k0_cg0 = dim_half_k_cg0 * 8 + dim_local0_k_cg0;
                            int reg_k1_cg0 = reg_k0_cg0 + 1;
                            float prefix_k0_cg0 = exp_g_regs_cg0[reg_k0_cg0];
                            float prefix_k1_cg0 = exp_g_regs_cg0[reg_k1_cg0];
                            float2 _f2_7 = make_float2(k_values_cg0[reg_k0_cg0], k_values_cg0[reg_k1_cg0]);
                            float2 k_norm_pair_cg0 = mul_f32x2(_f2_7, k_inv_norm_pair_cg0);
                            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(k_norm_pair_cg0.x, k_norm_pair_cg0.y));
                            __nv_bfloat162 k_norm_bf16x2_cg0 = _bf16x2_0;
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(prefix_k0_cg0, prefix_k1_cg0));
                            __nv_bfloat162 prefix_bf16x2_cg0 = _bf16x2_1;
                            float _rcp_0 = approx_rcp(prefix_k0_cg0);
                            float _rcp_1 = approx_rcp(prefix_k1_cg0);
                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_rcp_0, _rcp_1));
                            __nv_bfloat162 reciprocal_bf16x2_cg0 = _bf16x2_2;
                            __nv_bfloat162 k_inv_bf16x2_cg0 = k_norm_bf16x2_cg0 * reciprocal_bf16x2_cg0;
                            __nv_bfloat162 k_decay_bf16x2_cg0 = k_norm_bf16x2_cg0 * prefix_bf16x2_cg0;
                            int k_word_cg0 = dim_half_k_cg0 * 4 + dim_pair_k_cg0;
                            k_inv_pairs_all_cg0[k_word_cg0] = k_inv_bf16x2_cg0;
                            k_decay_words_cg0[dim_pair_k_cg0] = __as_u32(k_decay_bf16x2_cg0);
                        }
                        if (dim_half_k_cg0 == 0) {
                            mbarrier_wait(decay_super_done_addr + (decay_stage_cg0) * 8, decay_free_phase_cg0);
                            mbarrier_wait(decay_tcgen_done_addr + (decay_stage_cg0) * 8, decay_free_phase_cg0);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_k_inv_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 ^ (dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4])), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4 + 1])), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4 + 2])), "r"(__as_u32(k_inv_pairs_all_cg0[dim_half_k_cg0 * 4 + 3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_k_decay_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 ^ (dim_base_k_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_k_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(k_decay_words_cg0[0]), "r"(k_decay_words_cg0[1]), "r"(k_decay_words_cg0[2]), "r"(k_decay_words_cg0[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(k_decay_inv_ready_addr + (decay_stage_cg0) * 8);
                    mbarrier_arrive(q_done_addr + (raw_stage_cg0) * 8);
                    mbarrier_arrive(k_done_addr + (raw_stage_cg0) * 8);
                    mbarrier_arrive(g_done_addr + (raw_stage_cg0) * 8);
                    float2 _f2_8 = make_float2(q_inv_norm_cg0, q_inv_norm_cg0);
                    float2 q_inv_pair_cg0 = _f2_8;
                    #pragma unroll
                    for (int dim_half_q_cg0 = 0; dim_half_q_cg0 < 2; dim_half_q_cg0++) {
                        int dim_base_q_cg0 = dim_half_q_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int q_decay_words_cg0[4];
                        #pragma unroll
                        for (int dim_pair_q_cg0 = 0; dim_pair_q_cg0 < 4; dim_pair_q_cg0++) {
                            int dim_local0_q_cg0 = dim_pair_q_cg0 * 2;
                            int dim_local1_q_cg0 = dim_local0_q_cg0 + 1;
                            int reg_q0_cg0 = dim_half_q_cg0 * 8 + dim_local0_q_cg0;
                            int reg_q1_cg0 = reg_q0_cg0 + 1;
                            float2 _f2_9 = make_float2(q_values_cg0[reg_q0_cg0], q_values_cg0[reg_q1_cg0]);
                            float2 q_norm_pair_cg0 = mul_f32x2(_f2_9, q_inv_pair_cg0);
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(q_norm_pair_cg0.x, q_norm_pair_cg0.y));
                            __nv_bfloat162 q_norm_bf16x2_cg0 = _bf16x2_3;
                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(exp_g_regs_cg0[reg_q0_cg0], exp_g_regs_cg0[reg_q1_cg0]));
                            __nv_bfloat162 q_prefix_bf16x2_cg0 = _bf16x2_4;
                            __nv_bfloat162 q_decay_bf16x2_cg0 = q_norm_bf16x2_cg0 * q_prefix_bf16x2_cg0;
                            q_decay_words_cg0[dim_pair_q_cg0] = __as_u32(q_decay_bf16x2_cg0);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_q_decay_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_q_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_q_cg0 % 64 * 2 ^ (dim_base_q_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_q_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(q_decay_words_cg0[0]), "r"(q_decay_words_cg0[1]), "r"(q_decay_words_cg0[2]), "r"(q_decay_words_cg0[3]) : "memory");
                    }
                    mbarrier_wait(k_restore_done_addr + (decay_stage_cg0) * 8, decay_free_phase_cg0);
                    #pragma unroll
                    for (int dim_half_restore_cg0 = 0; dim_half_restore_cg0 < 2; dim_half_restore_cg0++) {
                        int dim_base_restore_cg0 = dim_half_restore_cg0 * 64 + decay_lane_cg0 * 8;
                        unsigned int k_restore_words_cg0[4];
                        #pragma unroll
                        for (int dim_pair_restore_cg0 = 0; dim_pair_restore_cg0 < 4; dim_pair_restore_cg0++) {
                            int dim_local0_restore_cg0 = dim_pair_restore_cg0 * 2;
                            int dim_local1_restore_cg0 = dim_local0_restore_cg0 + 1;
                            int reg_restore0_cg0 = dim_half_restore_cg0 * 8 + dim_local0_restore_cg0;
                            int reg_restore1_cg0 = reg_restore0_cg0 + 1;
                            int restore_word_cg0 = dim_half_restore_cg0 * 4 + dim_pair_restore_cg0;
                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(exp_g_last_regs_cg0[reg_restore0_cg0], exp_g_last_regs_cg0[reg_restore1_cg0]));
                            __nv_bfloat162 k_restore_bf16x2_cg0 = k_inv_pairs_all_cg0[restore_word_cg0] * _bf16x2_5;
                            k_restore_words_cg0[dim_pair_restore_cg0] = __as_u32(k_restore_bf16x2_cg0);
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_k_restore_addr + (unsigned int)(decay_stage_cg0 * 4096) + (unsigned int)(dim_base_restore_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_restore_cg0 % 64 * 2 ^ (dim_base_restore_cg0 / 64 * 2048 + decay_row_cg0 * 128 + dim_base_restore_cg0 % 64 * 2 >> 7 & 7) << 4))), "r"(k_restore_words_cg0[0]), "r"(k_restore_words_cg0[1]), "r"(k_restore_words_cg0[2]), "r"(k_restore_words_cg0[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(qk_scale_ready_addr + (diag_stage_cg0) * 8);
                    raw_stage_cg0 += 2;
                    if (raw_stage_cg0 >= 5) {
                        raw_stage_cg0 -= 5;
                    }
                    raw_bar_stage_cg0 += 2;
                    if (raw_bar_stage_cg0 >= 6) {
                        raw_bar_stage_cg0 -= 6;
                        raw_bar_phase_cg0 ^= 1;
                    }
                    decay_free_phase_cg0 ^= 1;
                    diag_stage_cg0 += 2;
                    if (diag_stage_cg0 >= 4) {
                        diag_stage_cg0 -= 4;
                        diag_free_phase_cg0 ^= 1;
                    }
                }
                cumulative_chunk_cg0 += chunks_cg0;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }
    // ---- Role: cg1 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
        { // cg1_main
            unsigned int sched_stage_cg1 = 0;
            int cumulative_chunk_cg1 = 0;
            int warp_in_wg = warp % 4;
            int value_row_cg1 = warp_in_wg * 32 + lane;
            int value_dim_base_cg1 = warp_in_wg * 32;
            const int tmem_row_base_cg1 = warp_in_wg * 32 << 16;
            int ov_token_cg1 = lane / 16 * 8 + (lane & 7);
            int ov_col_cg1 = (lane / 8 & 1) * 8;
            unsigned int _phase_sched_ready_1 = 0;
            #pragma unroll 1
            for (int __1 = 0; __1 < total_work_items + 1; __1++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_cg1) * 8, _phase_sched_ready_1);
                unsigned int slot_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_1[0])) : "r"(sched_slot_addr + sched_stage_cg1 * 4));
                unsigned int tile_cg1 = slot_1[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_cg1) * 8);
                }
                sched_stage_cg1 += 1;
                if (sched_stage_cg1 == 8) { sched_stage_cg1 = 0; _phase_sched_ready_1 ^= 1; }
                if (tile_cg1 >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_cg1 = (int)tile_cg1 * 8;
                int _vec_load_4[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_cg1);
                    _vec_load_4[0 + 0] = _iv4.x;
                    _vec_load_4[0 + 1] = _iv4.y;
                    _vec_load_4[0 + 2] = _iv4.z;
                    _vec_load_4[0 + 3] = _iv4.w;
                }
                int _vec_load_5[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_cg1 + 4);
                    _vec_load_5[0 + 0] = _iv4.x;
                    _vec_load_5[0 + 1] = _iv4.y;
                    _vec_load_5[0 + 2] = _iv4.z;
                    _vec_load_5[0 + 3] = _iv4.w;
                }
                int seq_cg1 = _vec_load_4[0];
                int head_cg1 = _vec_load_4[1];
                int wstart_cg1 = _vec_load_4[2];
                int wend_cg1 = _vec_load_4[3];
                int cstart_cg1 = _vec_load_5[0];
                long long bos_cg1 = (long long)_vec_load_5[2];
                long long eos_cg1 = (long long)_vec_load_5[3];
                int sequence_chunks_cg1 = (int)((eos_cg1 - bos_cg1) / 16);
                int chunks_cg1 = wend_cg1 - cstart_cg1;
                long long state_base_cg1 = (((long long)seq_cg1 * (long long)num_heads + (long long)head_cg1) * 128 + (long long)value_row_cg1) * 128;
                #pragma unroll
                for (int state_block_cg1 = 0; state_block_cg1 < 4; state_block_cg1++) {
                    float state_init_cg1[32];
                    state_init_cg1[0] = 0.0f;
                    state_init_cg1[1] = 0.0f;
                    state_init_cg1[2] = 0.0f;
                    state_init_cg1[3] = 0.0f;
                    state_init_cg1[4] = 0.0f;
                    state_init_cg1[5] = 0.0f;
                    state_init_cg1[6] = 0.0f;
                    state_init_cg1[7] = 0.0f;
                    state_init_cg1[8] = 0.0f;
                    state_init_cg1[9] = 0.0f;
                    state_init_cg1[10] = 0.0f;
                    state_init_cg1[11] = 0.0f;
                    state_init_cg1[12] = 0.0f;
                    state_init_cg1[13] = 0.0f;
                    state_init_cg1[14] = 0.0f;
                    state_init_cg1[15] = 0.0f;
                    state_init_cg1[16] = 0.0f;
                    state_init_cg1[17] = 0.0f;
                    state_init_cg1[18] = 0.0f;
                    state_init_cg1[19] = 0.0f;
                    state_init_cg1[20] = 0.0f;
                    state_init_cg1[21] = 0.0f;
                    state_init_cg1[22] = 0.0f;
                    state_init_cg1[23] = 0.0f;
                    state_init_cg1[24] = 0.0f;
                    state_init_cg1[25] = 0.0f;
                    state_init_cg1[26] = 0.0f;
                    state_init_cg1[27] = 0.0f;
                    state_init_cg1[28] = 0.0f;
                    state_init_cg1[29] = 0.0f;
                    state_init_cg1[30] = 0.0f;
                    state_init_cg1[31] = 0.0f;
                    if (USE_INITIAL_STATE != 0 && cstart_cg1 == 0) {
                        #pragma unroll
                        for (int state_vec_cg1 = 0; state_vec_cg1 < 4; state_vec_cg1++) {
                            {
                                unsigned _ldv8_0_0;
                                unsigned _ldv8_0_1;
                                unsigned _ldv8_0_2;
                                unsigned _ldv8_0_3;
                                unsigned _ldv8_0_4;
                                unsigned _ldv8_0_5;
                                unsigned _ldv8_0_6;
                                unsigned _ldv8_0_7;
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(initial_state + (state_base_cg1 + (long long)(state_block_cg1 * 32) + (long long)(state_vec_cg1 * 8)))) : "memory");
                                state_init_cg1[state_vec_cg1 * 8 + 0] = __uint_as_float(_ldv8_0_0);
                                state_init_cg1[state_vec_cg1 * 8 + 1] = __uint_as_float(_ldv8_0_1);
                                state_init_cg1[state_vec_cg1 * 8 + 2] = __uint_as_float(_ldv8_0_2);
                                state_init_cg1[state_vec_cg1 * 8 + 3] = __uint_as_float(_ldv8_0_3);
                                state_init_cg1[state_vec_cg1 * 8 + 4] = __uint_as_float(_ldv8_0_4);
                                state_init_cg1[state_vec_cg1 * 8 + 5] = __uint_as_float(_ldv8_0_5);
                                state_init_cg1[state_vec_cg1 * 8 + 6] = __uint_as_float(_ldv8_0_6);
                                state_init_cg1[state_vec_cg1 * 8 + 7] = __uint_as_float(_ldv8_0_7);
                            }
                        }
                        #pragma unroll
                        for (int state_elem_cg1 = 0; state_elem_cg1 < 32; state_elem_cg1++) {
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(state_init_cg1[state_elem_cg1]);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_1);
                            state_init_cg1[state_elem_cg1] = _cvt_f32_2;
                        }
                    }
                    tmem_st_x32_f32(taddr + (unsigned int)tmem_row_base_cg1 + (unsigned int)(state_block_cg1 * 32), state_init_cg1);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (chunks_cg1 > 0) {
                    int cumulative_cg1 = cumulative_chunk_cg1;
                    unsigned int raw_stage_cg1 = (unsigned int)(cumulative_cg1 % 5);
                    unsigned int raw_bar_stage_cg1 = (unsigned int)(cumulative_cg1 % 6);
                    unsigned int o_stage_cg1 = (unsigned int)(cumulative_cg1 % 2);
                    unsigned int checkpoint_stage_cg1 = (unsigned int)(cumulative_cg1 % 2);
                    unsigned int state_phase_cg1 = (unsigned int)(cumulative_cg1 & 1);
                    float _tmem_load_0[128];
                    tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)tmem_row_base_cg1);
                    tmem_ld_x16(&_tmem_load_0[16], taddr + (unsigned int)tmem_row_base_cg1 + 16);
                    tmem_ld_x16(&_tmem_load_0[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                    tmem_ld_x16(&_tmem_load_0[48], taddr + (unsigned int)tmem_row_base_cg1 + 48);
                    tmem_ld_x16(&_tmem_load_0[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                    tmem_ld_x16(&_tmem_load_0[80], taddr + (unsigned int)tmem_row_base_cg1 + 80);
                    tmem_ld_x16(&_tmem_load_0[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                    tmem_ld_x16(&_tmem_load_0[112], taddr + (unsigned int)tmem_row_base_cg1 + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        mbarrier_wait(checkpoint_done_addr + (checkpoint_stage_cg1) * 8, (unsigned int)(cumulative_cg1 / 2 + 1 & 1));
                    }
                    unsigned int state_words_cg1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 ^ (value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 16 ^ (value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 8, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 32 ^ (value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 48 ^ (value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 32], _tmem_load_0[_lp*2+1 + 32]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 16, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 64 ^ (value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 80 ^ (value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 48], _tmem_load_0[_lp*2+1 + 48]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 24, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 96 ^ (value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 112 ^ (value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 64], _tmem_load_0[_lp*2+1 + 64]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 32, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 ^ (16384 + value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 16 ^ (16384 + value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 80], _tmem_load_0[_lp*2+1 + 80]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 40, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 32 ^ (16384 + value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 48 ^ (16384 + value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 96], _tmem_load_0[_lp*2+1 + 96]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 48, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 64 ^ (16384 + value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 80 ^ (16384 + value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 112], _tmem_load_0[_lp*2+1 + 112]));
                        state_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 56, (const uint32_t*)state_words_cg1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 96 ^ (16384 + value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1[0]), "r"(state_words_cg1[1]), "r"(state_words_cg1[2]), "r"(state_words_cg1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 112 ^ (16384 + value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1[4]), "r"(state_words_cg1[5]), "r"(state_words_cg1[6]), "r"(state_words_cg1[7]) : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(state_inp_ready_addr);
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(state_read_done_addr);
                        mbarrier_arrive(checkpoint_ready_addr + (checkpoint_stage_cg1) * 8);
                    }
                    mbarrier_wait(v_ready_addr + (raw_bar_stage_cg1) * 8, (unsigned int)(cumulative_cg1 / 6 & 1));
                    mbarrier_wait(beta_ready_addr + (raw_bar_stage_cg1) * 8, (unsigned int)(cumulative_cg1 / 6 & 1));
                    mbarrier_wait(state_k_ready_addr, state_phase_cg1);
                    float _tmem_load_3[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1));
                    float _tmem_load_4[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1 + 1048576));
                    unsigned int raw_v_words_lo_cg1[4];
                    unsigned int raw_v_words_hi_cg1[4];
                    int segment_3 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                    int segment_col_3 = value_dim_base_cg1 + ov_col_cg1 - segment_3 * 64;
                    int swizzled_col_3 = segment_col_3 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_lo_cg1[0]), "=r"(raw_v_words_lo_cg1[1]), "=r"(raw_v_words_lo_cg1[2]), "=r"(raw_v_words_lo_cg1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1 * 16 * 128 + (unsigned int)(segment_3 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_3) * 2)
                        : "memory");
                    int segment_0_1 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                    int segment_col_1_2 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_1 * 64;
                    int swizzled_col_2_2 = segment_col_1_2 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_hi_cg1[0]), "=r"(raw_v_words_hi_cg1[1]), "=r"(raw_v_words_hi_cg1[2]), "=r"(raw_v_words_hi_cg1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1 * 16 * 128 + (unsigned int)(segment_0_1 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_2_2) * 2)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __nv_bfloat162 beta_pairs_cg1[4];
                    #pragma unroll
                    for (int beta_reg_cg1 = 0; beta_reg_cg1 < 4; beta_reg_cg1++) {
                        int beta_packed_col_cg1 = beta_reg_cg1 / 2 * 4 + (lane & 3);
                        int beta_token0_cg1 = beta_packed_col_cg1 * 2;
                        int beta_token1_cg1 = beta_token0_cg1 + 1;
                        __nv_bfloat16 beta0_cg1 = smem_beta_all[(int)raw_bar_stage_cg1 * 16 + beta_token0_cg1];
                        __nv_bfloat16 beta1_cg1 = smem_beta_all[(int)raw_bar_stage_cg1 * 16 + beta_token1_cg1];
                        float _cvt_f32_3 = __bfloat162float(beta0_cg1);
                        float _cvt_f32_4 = __bfloat162float(beta1_cg1);
                        __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_cvt_f32_3, _cvt_f32_4));
                        beta_pairs_cg1[beta_reg_cg1] = _bf16x2_6;
                    }
                    unsigned int y_words_lo_cg1[4];
                    unsigned int y_words_hi_cg1[4];
                    #pragma unroll
                    for (int rhs_reg_cg1 = 0; rhs_reg_cg1 < 4; rhs_reg_cg1++) {
                        int rhs_raw_matrix_cg1 = rhs_reg_cg1;
                        int rhs_frag_pair_cg1 = rhs_reg_cg1 * 2;
                        __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_tmem_load_3[rhs_frag_pair_cg1], _tmem_load_3[rhs_frag_pair_cg1 + 1]));
                        __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_tmem_load_4[rhs_frag_pair_cg1], _tmem_load_4[rhs_frag_pair_cg1 + 1]));
                        uint32_t _bf16x2_sub_0;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_0) : "r"(raw_v_words_lo_cg1[rhs_raw_matrix_cg1]), "r"(__as_u32(_bf16x2_7)));
                        uint32_t _bf16x2_sub_1;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_1) : "r"(raw_v_words_hi_cg1[rhs_raw_matrix_cg1]), "r"(__as_u32(_bf16x2_8)));
                        __nv_bfloat162 rhs_diff_pair_lo_cg1 = __as_bf16x2(_bf16x2_sub_0);
                        __nv_bfloat162 rhs_diff_pair_hi_cg1 = __as_bf16x2(_bf16x2_sub_1);
                        __nv_bfloat162 y_pair_lo_cg1 = beta_pairs_cg1[rhs_reg_cg1] * rhs_diff_pair_lo_cg1;
                        __nv_bfloat162 y_pair_hi_cg1 = beta_pairs_cg1[rhs_reg_cg1] * rhs_diff_pair_hi_cg1;
                        y_words_lo_cg1[rhs_reg_cg1] = __as_u32(y_pair_lo_cg1);
                        y_words_hi_cg1[rhs_reg_cg1] = __as_u32(y_pair_hi_cg1);
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1[3])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1[3])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(v_done_addr + (raw_stage_cg1) * 8);
                    mbarrier_arrive(beta_done_addr + (raw_bar_stage_cg1) * 8);
                    mbarrier_arrive(y_inp_ready_addr);
                    mbarrier_wait(u_acc_ready_addr, state_phase_cg1);
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 240 + (unsigned int)tmem_row_base_cg1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int u_words_cg1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                        u_words_cg1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 264 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)u_words_cg1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(u_inp_ready_addr);
                }
                #pragma unroll 1
                for (int chunk_cg1 = 1; chunk_cg1 < chunks_cg1; chunk_cg1++) {
                    int cumulative_cg1_1 = cumulative_chunk_cg1 + chunk_cg1;
                    unsigned int raw_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 5);
                    unsigned int raw_bar_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 6);
                    unsigned int o_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 2);
                    unsigned int checkpoint_stage_cg1_1 = (unsigned int)(cumulative_cg1_1 % 2);
                    unsigned int state_phase_cg1_1 = (unsigned int)(cumulative_cg1_1 & 1);
                    {
                        mbarrier_wait(state_acc_done_addr, (unsigned int)(cumulative_cg1_1 - 1 & 1));
                    }
                    float _tmem_load_6[128];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)tmem_row_base_cg1);
                    tmem_ld_x16(&_tmem_load_6[16], taddr + (unsigned int)tmem_row_base_cg1 + 16);
                    tmem_ld_x16(&_tmem_load_6[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                    tmem_ld_x16(&_tmem_load_6[48], taddr + (unsigned int)tmem_row_base_cg1 + 48);
                    tmem_ld_x16(&_tmem_load_6[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                    tmem_ld_x16(&_tmem_load_6[80], taddr + (unsigned int)tmem_row_base_cg1 + 80);
                    tmem_ld_x16(&_tmem_load_6[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                    tmem_ld_x16(&_tmem_load_6[112], taddr + (unsigned int)tmem_row_base_cg1 + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        mbarrier_wait(checkpoint_done_addr + (checkpoint_stage_cg1_1) * 8, (unsigned int)(cumulative_cg1_1 / 2 + 1 & 1));
                    }
                    unsigned int state_words_cg1_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 ^ (value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 16 ^ (value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 16], _tmem_load_6[_lp*2+1 + 16]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 8, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 32 ^ (value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 48 ^ (value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 32], _tmem_load_6[_lp*2+1 + 32]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 16, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 64 ^ (value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 80 ^ (value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 48], _tmem_load_6[_lp*2+1 + 48]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 24, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 96 ^ (value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(value_row_cg1 * 128 + 112 ^ (value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 64], _tmem_load_6[_lp*2+1 + 64]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 32, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 ^ (16384 + value_row_cg1 * 128 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 16 ^ (16384 + value_row_cg1 * 128 + 16 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 80], _tmem_load_6[_lp*2+1 + 80]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 40, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 32 ^ (16384 + value_row_cg1 * 128 + 32 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 48 ^ (16384 + value_row_cg1 * 128 + 48 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 96], _tmem_load_6[_lp*2+1 + 96]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 48, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 64 ^ (16384 + value_row_cg1 * 128 + 64 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 80 ^ (16384 + value_row_cg1 * 128 + 80 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 112], _tmem_load_6[_lp*2+1 + 112]));
                        state_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base_cg1 + 56, (const uint32_t*)state_words_cg1_1);
                    {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 96 ^ (16384 + value_row_cg1 * 128 + 96 >> 7 & 7) << 4))), "r"(state_words_cg1_1[0]), "r"(state_words_cg1_1[1]), "r"(state_words_cg1_1[2]), "r"(state_words_cg1_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_checkpoint_addr + checkpoint_stage_cg1_1 * 32768 + (unsigned int)(16384 + value_row_cg1 * 128 + 112 ^ (16384 + value_row_cg1 * 128 + 112 >> 7 & 7) << 4))), "r"(state_words_cg1_1[4]), "r"(state_words_cg1_1[5]), "r"(state_words_cg1_1[6]), "r"(state_words_cg1_1[7]) : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(state_inp_ready_addr);
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(state_read_done_addr);
                        mbarrier_arrive(checkpoint_ready_addr + (checkpoint_stage_cg1_1) * 8);
                    }
                    {
                        int previous_event_cg1 = cumulative_cg1_1 - 1;
                        unsigned int previous_o_stage_cg1 = (unsigned int)(previous_event_cg1 % 2);
                        mbarrier_wait(o_acc_ready_addr, (unsigned int)(previous_event_cg1 & 1));
                        mbarrier_wait(o_tma_done_addr + (previous_o_stage_cg1) * 8, (unsigned int)(previous_event_cg1 / 2 + 1 & 1));
                        int output_col_cg1 = 192 + (int)previous_o_stage_cg1 * 16;
                        float _tmem_load_7[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7]))
                            : "r"(taddr + (unsigned int)output_col_cg1 + (unsigned int)tmem_row_base_cg1));
                        float _tmem_load_8[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7]))
                            : "r"(taddr + (unsigned int)output_col_cg1 + (unsigned int)tmem_row_base_cg1 + 1048576));
                        const float2 _scale2_1 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_1);
                        const float2 _scale2_2 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_2);
                        uint32_t _tmem_load_7_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                            _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _tmem_load_8_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                            _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int segment_4 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                        int segment_col_4 = value_dim_base_cg1 + ov_col_cg1 - segment_4 * 64;
                        int swizzled_col_4 = segment_col_4 ^ (ov_token_cg1 & 7) * 8;
                        int segment_0_2 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                        int segment_col_1_3 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_2 * 64;
                        int swizzled_col_2_3 = segment_col_1_3 ^ (ov_token_cg1 & 7) * 8;
                        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)previous_o_stage_cg1 * 16 * 128 + segment_4 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_4) * 2)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[3]))
                            : "memory");
                        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)previous_o_stage_cg1 * 16 * 128 + segment_0_2 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_2_3) * 2)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[3]))
                            : "memory");
                        mbarrier_arrive(o_acc_done_addr + (previous_o_stage_cg1) * 8);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(o_tma_ready_addr + (previous_o_stage_cg1) * 8);
                    }
                    mbarrier_wait(v_ready_addr + (raw_bar_stage_cg1_1) * 8, (unsigned int)(cumulative_cg1_1 / 6 & 1));
                    mbarrier_wait(beta_ready_addr + (raw_bar_stage_cg1_1) * 8, (unsigned int)(cumulative_cg1_1 / 6 & 1));
                    mbarrier_wait(state_k_ready_addr, state_phase_cg1_1);
                    float _tmem_load_9[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1));
                    float _tmem_load_10[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7]))
                        : "r"(taddr + 224 + (unsigned int)tmem_row_base_cg1 + 1048576));
                    unsigned int raw_v_words_lo_cg1_1[4];
                    unsigned int raw_v_words_hi_cg1_1[4];
                    int segment_5 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                    int segment_col_5 = value_dim_base_cg1 + ov_col_cg1 - segment_5 * 64;
                    int swizzled_col_5 = segment_col_5 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_lo_cg1_1[0]), "=r"(raw_v_words_lo_cg1_1[1]), "=r"(raw_v_words_lo_cg1_1[2]), "=r"(raw_v_words_lo_cg1_1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1_1 * 16 * 128 + (unsigned int)(segment_5 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_5) * 2)
                        : "memory");
                    int segment_0_3 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                    int segment_col_1_4 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_3 * 64;
                    int swizzled_col_2_4 = segment_col_1_4 ^ (ov_token_cg1 & 7) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(raw_v_words_hi_cg1_1[0]), "=r"(raw_v_words_hi_cg1_1[1]), "=r"(raw_v_words_hi_cg1_1[2]), "=r"(raw_v_words_hi_cg1_1[3])
                        : "r"(smem_v_all_addr + (raw_stage_cg1_1 * 16 * 128 + (unsigned int)(segment_0_3 * 16 * 64) + (unsigned int)(ov_token_cg1 * 64) + (unsigned int)swizzled_col_2_4) * 2)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __nv_bfloat162 beta_pairs_cg1_1[4];
                    #pragma unroll
                    for (int beta_reg_cg1_1 = 0; beta_reg_cg1_1 < 4; beta_reg_cg1_1++) {
                        int beta_packed_col_cg1_1 = beta_reg_cg1_1 / 2 * 4 + (lane & 3);
                        int beta_token0_cg1_1 = beta_packed_col_cg1_1 * 2;
                        int beta_token1_cg1_1 = beta_token0_cg1_1 + 1;
                        __nv_bfloat16 beta0_cg1_1 = smem_beta_all[(int)raw_bar_stage_cg1_1 * 16 + beta_token0_cg1_1];
                        __nv_bfloat16 beta1_cg1_1 = smem_beta_all[(int)raw_bar_stage_cg1_1 * 16 + beta_token1_cg1_1];
                        float _cvt_f32_5 = __bfloat162float(beta0_cg1_1);
                        float _cvt_f32_6 = __bfloat162float(beta1_cg1_1);
                        __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_cvt_f32_5, _cvt_f32_6));
                        beta_pairs_cg1_1[beta_reg_cg1_1] = _bf16x2_9;
                    }
                    unsigned int y_words_lo_cg1_1[4];
                    unsigned int y_words_hi_cg1_1[4];
                    #pragma unroll
                    for (int rhs_reg_cg1_1 = 0; rhs_reg_cg1_1 < 4; rhs_reg_cg1_1++) {
                        int rhs_raw_matrix_cg1_1 = rhs_reg_cg1_1;
                        int rhs_frag_pair_cg1_1 = rhs_reg_cg1_1 * 2;
                        __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_tmem_load_9[rhs_frag_pair_cg1_1], _tmem_load_9[rhs_frag_pair_cg1_1 + 1]));
                        __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_tmem_load_10[rhs_frag_pair_cg1_1], _tmem_load_10[rhs_frag_pair_cg1_1 + 1]));
                        uint32_t _bf16x2_sub_2;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_2) : "r"(raw_v_words_lo_cg1_1[rhs_raw_matrix_cg1_1]), "r"(__as_u32(_bf16x2_10)));
                        uint32_t _bf16x2_sub_3;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_3) : "r"(raw_v_words_hi_cg1_1[rhs_raw_matrix_cg1_1]), "r"(__as_u32(_bf16x2_11)));
                        __nv_bfloat162 rhs_diff_pair_lo_cg1_1 = __as_bf16x2(_bf16x2_sub_2);
                        __nv_bfloat162 rhs_diff_pair_hi_cg1_1 = __as_bf16x2(_bf16x2_sub_3);
                        __nv_bfloat162 y_pair_lo_cg1_1 = beta_pairs_cg1_1[rhs_reg_cg1_1] * rhs_diff_pair_lo_cg1_1;
                        __nv_bfloat162 y_pair_hi_cg1_1 = beta_pairs_cg1_1[rhs_reg_cg1_1] * rhs_diff_pair_hi_cg1_1;
                        y_words_lo_cg1_1[rhs_reg_cg1_1] = __as_u32(y_pair_lo_cg1_1);
                        y_words_hi_cg1_1[rhs_reg_cg1_1] = __as_u32(y_pair_hi_cg1_1);
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo_cg1_1[3])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x2.b32"
                        " [%0], {%1, %2, %3, %4};"
                        :: "r"(taddr + 256 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi_cg1_1[3])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(v_done_addr + (raw_stage_cg1_1) * 8);
                    mbarrier_arrive(beta_done_addr + (raw_bar_stage_cg1_1) * 8);
                    mbarrier_arrive(y_inp_ready_addr);
                    mbarrier_wait(u_acc_ready_addr, state_phase_cg1_1);
                    float _tmem_load_11[16];
                    tmem_ld_x16(&_tmem_load_11[0], taddr + 240 + (unsigned int)tmem_row_base_cg1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int u_words_cg1_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                        u_words_cg1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 264 + (unsigned int)tmem_row_base_cg1, (const uint32_t*)u_words_cg1_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(u_inp_ready_addr);
                }
                if (chunks_cg1 > 0) {
                    int final_event_cg1 = cumulative_chunk_cg1 + chunks_cg1 - 1;
                    unsigned int final_o_stage_cg1 = (unsigned int)(final_event_cg1 % 2);
                    mbarrier_wait(state_acc_done_addr, (unsigned int)(final_event_cg1 & 1));
                    mbarrier_wait(o_acc_ready_addr, (unsigned int)(final_event_cg1 & 1));
                    mbarrier_wait(o_tma_done_addr + (final_o_stage_cg1) * 8, (unsigned int)(final_event_cg1 / 2 + 1 & 1));
                    int output_col_cg1_1 = 192 + (int)final_o_stage_cg1 * 16;
                    float _tmem_load_12[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7]))
                        : "r"(taddr + (unsigned int)output_col_cg1_1 + (unsigned int)tmem_row_base_cg1));
                    float _tmem_load_13[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7]))
                        : "r"(taddr + (unsigned int)output_col_cg1_1 + (unsigned int)tmem_row_base_cg1 + 1048576));
                    const float2 _scale2_5 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_5);
                    const float2 _scale2_6 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_6);
                    uint32_t _tmem_load_12_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                        _tmem_load_12_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _tmem_load_13_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                        _tmem_load_13_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int segment_6 = (value_dim_base_cg1 + ov_col_cg1) / 64;
                    int segment_col_6 = value_dim_base_cg1 + ov_col_cg1 - segment_6 * 64;
                    int swizzled_col_6 = segment_col_6 ^ (ov_token_cg1 & 7) * 8;
                    int segment_0_4 = (value_dim_base_cg1 + 16 + ov_col_cg1) / 64;
                    int segment_col_1_5 = value_dim_base_cg1 + 16 + ov_col_cg1 - segment_0_4 * 64;
                    int swizzled_col_2_5 = segment_col_1_5 ^ (ov_token_cg1 & 7) * 8;
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)final_o_stage_cg1 * 16 * 128 + segment_6 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_6) * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)(smem_o_all_addr + (unsigned int)(((int)final_o_stage_cg1 * 16 * 128 + segment_0_4 * 16 * 64 + ov_token_cg1 * 64 + swizzled_col_2_5) * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[3]))
                        : "memory");
                    mbarrier_arrive(o_acc_done_addr + (final_o_stage_cg1) * 8);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(o_tma_ready_addr + (final_o_stage_cg1) * 8);
                    {
                        if (wend_cg1 == sequence_chunks_cg1) {
                            float _tmem_load_14[128];
                            tmem_ld_x16(&_tmem_load_14[0], taddr + (unsigned int)tmem_row_base_cg1);
                            tmem_ld_x16(&_tmem_load_14[16], taddr + (unsigned int)tmem_row_base_cg1 + 16);
                            tmem_ld_x16(&_tmem_load_14[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                            tmem_ld_x16(&_tmem_load_14[48], taddr + (unsigned int)tmem_row_base_cg1 + 48);
                            tmem_ld_x16(&_tmem_load_14[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                            tmem_ld_x16(&_tmem_load_14[80], taddr + (unsigned int)tmem_row_base_cg1 + 80);
                            tmem_ld_x16(&_tmem_load_14[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                            tmem_ld_x16(&_tmem_load_14[112], taddr + (unsigned int)tmem_row_base_cg1 + 112);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int final_vec_cg1 = 0; final_vec_cg1 < 16; final_vec_cg1++) {
                                {
                                    unsigned _stv8_9_0 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 0]);
                                    unsigned _stv8_9_1 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 1]);
                                    unsigned _stv8_9_2 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 2]);
                                    unsigned _stv8_9_3 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 3]);
                                    unsigned _stv8_9_4 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 4]);
                                    unsigned _stv8_9_5 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 5]);
                                    unsigned _stv8_9_6 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 6]);
                                    unsigned _stv8_9_7 = __float_as_uint(_tmem_load_14[final_vec_cg1 * 8 + 7]);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(final_state + (state_base_cg1 + (long long)(final_vec_cg1 * 8)) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
                                }
                            }
                        }
                    }
                }
                cumulative_chunk_cg1 += chunks_cg1;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }
    // ---- Role: super_mma ----
    if (warp == 12) {
        { // super_mma_main
            unsigned int sched_stage_super = 0;
            int cumulative_chunk_super = 0;
            int lhs_row_super = lane % 8 + (lane / 8 & 1) * 8;
            int lhs_col_super = lane / 16 * 8;
            int rhs_row_super = lane % 8 + lane / 16 * 8;
            int rhs_col_super = (lane / 8 & 1) * 8;
            unsigned int _phase_sched_ready_2 = 0;
            #pragma unroll 1
            for (int __2 = 0; __2 < total_work_items + 1; __2++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_super) * 8, _phase_sched_ready_2);
                unsigned int slot_2[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_2[0])) : "r"(sched_slot_addr + sched_stage_super * 4));
                unsigned int tile_super = slot_2[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_super) * 8);
                }
                sched_stage_super += 1;
                if (sched_stage_super == 8) { sched_stage_super = 0; _phase_sched_ready_2 ^= 1; }
                if (tile_super >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_super = (int)tile_super * 8;
                int chunks_super = work_items[item_base_super + 3] - work_items[item_base_super + 4];
                #pragma unroll 1
                for (int chunk_super = 0; chunk_super < chunks_super; chunk_super++) {
                    int cumulative_super = cumulative_chunk_super + chunk_super;
                    unsigned int decay_stage_super = (unsigned int)(cumulative_super % 2);
                    unsigned int raw_bar_stage_super = (unsigned int)(cumulative_super % 6);
                    unsigned int intermediate_stage_super = (unsigned int)(cumulative_super % 2);
                    mbarrier_wait(k_decay_inv_ready_addr + (decay_stage_super) * 8, (unsigned int)(cumulative_super / 2 & 1));
                    mbarrier_wait(beta_ready_addr + (raw_bar_stage_super) * 8, (unsigned int)(cumulative_super / 6 & 1));
                    mbarrier_wait(tinv_done_addr + (intermediate_stage_super) * 8, (unsigned int)(cumulative_super / 2 + 1 & 1));
                    float kk_acc_super[8];
                    #pragma unroll
                    for (int k_block_super = 0; k_block_super < 8; k_block_super++) {
                        unsigned int a_frag_super[4];
                        unsigned int b_frag_super[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_super[0]), "=r"(a_frag_super[1]), "=r"(a_frag_super[2]), "=r"(a_frag_super[3])
                            : "r"((smem_k_decay_addr + decay_stage_super * 4096 + (unsigned int)((k_block_super * 16 + lhs_col_super) / 64 * 2048 + lhs_row_super * 128 + (k_block_super * 16 + lhs_col_super) % 64 * 2 ^ ((k_block_super * 16 + lhs_col_super) / 64 * 2048 + lhs_row_super * 128 + (k_block_super * 16 + lhs_col_super) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_super[0]), "=r"(b_frag_super[1]), "=r"(b_frag_super[2]), "=r"(b_frag_super[3])
                            : "r"((smem_k_inv_addr + decay_stage_super * 4096 + (unsigned int)((k_block_super * 16 + rhs_col_super) / 64 * 2048 + rhs_row_super * 128 + (k_block_super * 16 + rhs_col_super) % 64 * 2 ^ ((k_block_super * 16 + rhs_col_super) / 64 * 2048 + rhs_row_super * 128 + (k_block_super * 16 + rhs_col_super) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(kk_acc_super[0]), "=f"(kk_acc_super[1]), "=f"(kk_acc_super[2]), "=f"(kk_acc_super[3])
                            : "r"(a_frag_super[0]), "r"(a_frag_super[1]), "r"(a_frag_super[2]), "r"(a_frag_super[3]), "r"(b_frag_super[0]), "r"(b_frag_super[1]), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[0])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[1])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[2])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[3])));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(kk_acc_super[4]), "=f"(kk_acc_super[(4) + 1]), "=f"(kk_acc_super[(4) + 2]), "=f"(kk_acc_super[(4) + 3])
                            : "r"(a_frag_super[0]), "r"(a_frag_super[1]), "r"(a_frag_super[2]), "r"(a_frag_super[3]), "r"(b_frag_super[2]), "r"(b_frag_super[(2) + 1]), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[4])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[(4) + 1])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[(4) + 2])), "f"(((k_block_super == 0) ? 0.0f : kk_acc_super[(4) + 3])));
                    }
                    int beta_stage_base_super = (int)raw_bar_stage_super * 16;
                    __nv_bfloat16 beta_lo_bf_super = smem_beta_all[beta_stage_base_super + lane / 4];
                    __nv_bfloat16 beta_hi_bf_super = smem_beta_all[beta_stage_base_super + lane / 4 + 8];
                    float _cvt_f32_0 = __bfloat162float(beta_lo_bf_super);
                    float beta_lo_super = _cvt_f32_0;
                    float _cvt_f32_1 = __bfloat162float(beta_hi_bf_super);
                    float beta_hi_super = _cvt_f32_1;
                    float l_values_super[8];
                    float tinv_acc_super[8];
                    #pragma unroll
                    for (int accum_super = 0; accum_super < 8; accum_super++) {
                        int row_super = lane / 4 + accum_super % 4 / 2 * 8;
                        int col_super = accum_super / 4 * 8 + (lane & 3) * 2 + (accum_super & 1);
                        l_values_super[accum_super] = 0.0f;
                        if (row_super > col_super) {
                            float beta_scale_super = beta_lo_super;
                            if (accum_super % 4 >= 2) {
                                beta_scale_super = beta_hi_super;
                            }
                            l_values_super[accum_super] = kk_acc_super[accum_super] * beta_scale_super;
                        }
                        tinv_acc_super[accum_super] = -l_values_super[accum_super];
                        if (row_super == col_super) {
                            tinv_acc_super[accum_super] = 1.0f;
                        }
                    }
                    unsigned int lpow_words_super[4];
                    unsigned int lpow_trans_super[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l_values_super[_lp*2 + 0], l_values_super[_lp*2+1 + 0]));
                        lpow_words_super[_lp] = *(uint32_t*)&_bf2;
                    }
                    int store_row_super = lane % 16;
                    int store_col_super = lane / 16 * 8;
                    int linear = store_row_super * 16 + store_col_super;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(tinv_scratch_addr + (unsigned int)((linear ^ (linear >> 6 & 1) * 8) * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[0])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[1])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[2])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[3]))
                        : "memory");
                    __syncwarp();
                    int load_row_super = lane % 16;
                    #pragma unroll
                    for (int load_half_super = 0; load_half_super < 2; load_half_super++) {
                        int linear_0 = load_row_super * 16 + load_half_super * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(lpow_trans_super[load_half_super * 2]), "=r"(lpow_trans_super[load_half_super * 2 + 1])
                            : "r"(tinv_scratch_addr + (unsigned int)((linear_0 ^ (linear_0 >> 6 & 1) * 8) * 2))
                            : "memory");
                    }
                    #pragma unroll
                    for (int neumann_super = 0; neumann_super < 3; neumann_super++) {
                        float square_acc_super[8];
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(square_acc_super[0]), "=f"(square_acc_super[1]), "=f"(square_acc_super[2]), "=f"(square_acc_super[3])
                            : "r"(lpow_words_super[0]), "r"(lpow_words_super[1]), "r"(lpow_words_super[2]), "r"(lpow_words_super[3]), "r"(lpow_trans_super[0]), "r"(lpow_trans_super[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(square_acc_super[4]), "=f"(square_acc_super[(4) + 1]), "=f"(square_acc_super[(4) + 2]), "=f"(square_acc_super[(4) + 3])
                            : "r"(lpow_words_super[0]), "r"(lpow_words_super[1]), "r"(lpow_words_super[2]), "r"(lpow_words_super[3]), "r"(lpow_trans_super[2]), "r"(lpow_trans_super[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(square_acc_super[_lp*2 + 0], square_acc_super[_lp*2+1 + 0]));
                            lpow_words_super[_lp] = *(uint32_t*)&_bf2;
                        }
                        int store_row_super_0 = lane % 16;
                        int store_col_super_1 = lane / 16 * 8;
                        int linear_2 = store_row_super_0 * 16 + store_col_super_1;
                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(tinv_scratch_addr + (unsigned int)((linear_2 ^ (linear_2 >> 6 & 1) * 8) * 2)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[0])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[1])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[2])), "r"(*reinterpret_cast<const uint32_t*>(&lpow_words_super[3]))
                            : "memory");
                        __syncwarp();
                        int load_row_super_3 = lane % 16;
                        #pragma unroll
                        for (int load_half_super_1 = 0; load_half_super_1 < 2; load_half_super_1++) {
                            int linear_0_1 = load_row_super_3 * 16 + load_half_super_1 * 8;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                : "=r"(lpow_trans_super[load_half_super_1 * 2]), "=r"(lpow_trans_super[load_half_super_1 * 2 + 1])
                                : "r"(tinv_scratch_addr + (unsigned int)((linear_0_1 ^ (linear_0_1 >> 6 & 1) * 8) * 2))
                                : "memory");
                        }
                        unsigned int tinv_words_super[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tinv_acc_super[_lp*2 + 0], tinv_acc_super[_lp*2+1 + 0]));
                            tinv_words_super[_lp] = *(uint32_t*)&_bf2;
                        }
                        float update_acc_super[8];
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(update_acc_super[0]), "=f"(update_acc_super[1]), "=f"(update_acc_super[2]), "=f"(update_acc_super[3])
                            : "r"(tinv_words_super[0]), "r"(tinv_words_super[1]), "r"(tinv_words_super[2]), "r"(tinv_words_super[3]), "r"(lpow_trans_super[0]), "r"(lpow_trans_super[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(update_acc_super[4]), "=f"(update_acc_super[(4) + 1]), "=f"(update_acc_super[(4) + 2]), "=f"(update_acc_super[(4) + 3])
                            : "r"(tinv_words_super[0]), "r"(tinv_words_super[1]), "r"(tinv_words_super[2]), "r"(tinv_words_super[3]), "r"(lpow_trans_super[2]), "r"(lpow_trans_super[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        float tinv_words_super_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&tinv_words_super_f32[_pair * 2])[0]), "=f"((&tinv_words_super_f32[_pair * 2])[1])
                                : "r"(tinv_words_super[_pair]));
                        }
                        #pragma unroll
                        for (int update_super = 0; update_super < 8; update_super++) {
                            tinv_acc_super[update_super] = tinv_words_super_f32[update_super] + update_acc_super[update_super];
                        }
                    }
                    unsigned int tinv_publish_super[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tinv_acc_super[_lp*2 + 0], tinv_acc_super[_lp*2+1 + 0]));
                        tinv_publish_super[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(smem_tinv_addr + intermediate_stage_super * 1024 + (unsigned int)(lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 ^ (lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 >> 7 & 1) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[0])), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[1])), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[2])), "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_super[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(tinv_ready_addr + (intermediate_stage_super) * 8);
                    mbarrier_arrive(beta_done_addr + (raw_bar_stage_super) * 8);
                    mbarrier_arrive(decay_super_done_addr + (decay_stage_super) * 8);
                }
                cumulative_chunk_super += chunks_super;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }
    // ---- Role: tcgen ----
    if (warp == 13) {
        { // tcgen_main
            float tmem_seed_tcgen[1];
            tmem_seed_tcgen[0] = 0.0f;
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                " [%0], {%1};"
                :: "r"(taddr), "r"(*reinterpret_cast<const uint32_t*>(&tmem_seed_tcgen[0])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int sched_stage_tcgen = 0;
            int cumulative_chunk_tcgen = 0;
            unsigned int _phase_sched_ready_3 = 0;
            #pragma unroll 1
            for (int __3 = 0; __3 < total_work_items + 1; __3++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_tcgen) * 8, _phase_sched_ready_3);
                unsigned int slot_3[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_3[0])) : "r"(sched_slot_addr + sched_stage_tcgen * 4));
                unsigned int tile_tcgen = slot_3[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_tcgen) * 8);
                }
                sched_stage_tcgen += 1;
                if (sched_stage_tcgen == 8) { sched_stage_tcgen = 0; _phase_sched_ready_3 ^= 1; }
                if (tile_tcgen >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_tcgen = (int)tile_tcgen * 8;
                int chunks_tcgen = work_items[item_base_tcgen + 3] - work_items[item_base_tcgen + 4];
                #pragma unroll 1
                for (int chunk_tcgen = 0; chunk_tcgen < chunks_tcgen; chunk_tcgen++) {
                    int cumulative_tcgen = cumulative_chunk_tcgen + chunk_tcgen;
                    unsigned int state_phase_tcgen = (unsigned int)(cumulative_tcgen & 1);
                    unsigned int decay_stage_tcgen = (unsigned int)(cumulative_tcgen % 2);
                    unsigned int diag_stage_tcgen = (unsigned int)(cumulative_tcgen % 4);
                    unsigned int intermediate_stage_tcgen = (unsigned int)(cumulative_tcgen % 2);
                    unsigned int o_stage_tcgen = (unsigned int)(cumulative_tcgen % 2);
                    mbarrier_wait(k_decay_inv_ready_addr + (decay_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 & 1));
                    mbarrier_wait(state_inp_ready_addr, state_phase_tcgen);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_decay_addr) >> 4) & 0x3FFF) + (decay_stage_tcgen) * 256);
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(tmem_tmem_state_k), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                    elect_commit(state_k_ready_addr);
                    mbarrier_wait(qk_scale_ready_addr + (diag_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 4 & 1));
                    mbarrier_wait(o_acc_done_addr + (o_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 + 1 & 1));
                    int _mma_b_lo_1 = make_warp_uniform((((smem_q_decay_addr) >> 4) & 0x3FFF) + (decay_stage_tcgen) * 256);
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"((tmem_tmem_q_state + ((int)o_stage_tcgen * 16))), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0));
                    elect_commit(decay_tcgen_done_addr + (decay_stage_tcgen) * 8);
                    {
                        mbarrier_wait(state_read_done_addr, state_phase_tcgen);
                    }
                    #pragma unroll
                    for (int diag_block_tcgen = 0; diag_block_tcgen < 8; diag_block_tcgen++) {
                        int _mma_b_lo_2 = make_warp_uniform((((smem_state_diag_addr) >> 4) & 0x3FFF) + ((int)diag_stage_tcgen * 8 + diag_block_tcgen) * 32);
                        mma_ts_step((tmem_tmem_state + (diag_block_tcgen * 16)), tmem_tmem_state_inp + diag_block_tcgen * 8, _mma_b_lo_2, 0xC0004010, 134481040, 0);
                    }
                    elect_commit(state_diag_done_addr + (diag_stage_tcgen) * 8);
                    mbarrier_wait(tinv_ready_addr + (intermediate_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 & 1));
                    mbarrier_wait(y_inp_ready_addr, state_phase_tcgen);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_tinv_mn_addr) >> 4) & 0x3FFF) | 0x200000) + (intermediate_stage_tcgen) * 64);
                    mma_ts_step(tmem_tmem_u_acc, tmem_tmem_y_inp, _mma_b_lo_3, 0xC0004010, 134546576, 0);
                    elect_commit2(tinv_done_addr + (intermediate_stage_tcgen) * 8, u_acc_ready_addr);
                    mbarrier_wait(u_inp_ready_addr, state_phase_tcgen);
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_k_restore_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (decay_stage_tcgen) * 256);
                    mma_ts_step(tmem_tmem_state, tmem_tmem_u_inp, _mma_b_lo_4, 0x40004040, 136381584, 1);
                    elect_commit2(k_restore_done_addr + (decay_stage_tcgen) * 8, state_acc_done_addr);
                    mbarrier_wait(a_ready_addr + (intermediate_stage_tcgen) * 8, (unsigned int)(cumulative_tcgen / 2 & 1));
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_a_mn_addr) >> 4) & 0x3FFF) | 0x200000) + (intermediate_stage_tcgen) * 64);
                    mma_ts_step((tmem_tmem_q_state + ((int)o_stage_tcgen * 16)), tmem_tmem_u_inp, _mma_b_lo_5, 0xC0004010, 134546576, 1);
                    elect_commit2(o_acc_ready_addr, a_done_addr + (intermediate_stage_tcgen) * 8);
                }
                cumulative_chunk_tcgen += chunks_tcgen;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
            unsigned int _phase_cleanup_ready_0 = 0;
            mbarrier_wait(cleanup_ready_addr, _phase_cleanup_ready_0);
            _phase_cleanup_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: tma ----
    if (warp == 14) {
        { // tma_main
            unsigned int sched_stage_tma = 0;
            int cumulative_chunk_tma = 0;
            unsigned int _phase_sched_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int sched_iter_tma = 0; sched_iter_tma < total_work_items + 1; sched_iter_tma++) {
                    mbarrier_wait(sched_done_addr + (sched_stage_tma) * 8, _phase_sched_done);
                    unsigned int tile_tma = blockIdx.x;
                    if (uniform_work_items != 0) {
                        tile_tma = (unsigned int)blockIdx.x + (unsigned int)sched_iter_tma * (unsigned int)gridDim.x;
                    } else if (sched_iter_tma > 0) {
                        unsigned int _atomic_old_0 = atomicAdd(dynamic_counter, 1);
                        tile_tma = (unsigned int)gridDim.x + _atomic_old_0;
                    }
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sched_slot_addr + sched_stage_tma * 4), "r"(tile_tma));
                    mbarrier_arrive(sched_ready_addr + (sched_stage_tma) * 8);
                    sched_stage_tma += 1;
                    if (sched_stage_tma == 8) { sched_stage_tma = 0; _phase_sched_done ^= 1; }
                    if (tile_tma >= (unsigned int)total_work_items) {
                        break;
                    }
                    int item_base_tma = (int)tile_tma * 8;
                    int _vec_load_0[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_tma);
                        _vec_load_0[0 + 0] = _iv4.x;
                        _vec_load_0[0 + 1] = _iv4.y;
                        _vec_load_0[0 + 2] = _iv4.z;
                        _vec_load_0[0 + 3] = _iv4.w;
                    }
                    int _vec_load_1[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_tma + 4);
                        _vec_load_1[0 + 0] = _iv4.x;
                        _vec_load_1[0 + 1] = _iv4.y;
                        _vec_load_1[0 + 2] = _iv4.z;
                        _vec_load_1[0 + 3] = _iv4.w;
                    }
                    int head_tma = _vec_load_0[1];
                    int qk_head_tma = head_tma * num_qk_heads / num_heads;
                    int wend_tma = _vec_load_0[3];
                    int cstart_tma = _vec_load_1[0];
                    long long bos_tma = (long long)_vec_load_1[2];
                    int chunks_tma = wend_tma - cstart_tma;
                    #pragma unroll 1
                    for (int chunk_tma = 0; chunk_tma < chunks_tma; chunk_tma++) {
                        int cumulative_tma = cumulative_chunk_tma + chunk_tma;
                        unsigned int raw_stage_tma = (unsigned int)(cumulative_tma % 5);
                        unsigned int raw_bar_stage_tma = (unsigned int)(cumulative_tma % 6);
                        unsigned int raw_done_phase_tma = (unsigned int)(cumulative_tma / 5 + 1 & 1);
                        mbarrier_wait(q_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_wait(k_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_wait(v_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_wait(g_done_addr + (raw_stage_tma) * 8, raw_done_phase_tma);
                        mbarrier_arrive_expect_tx(q_ready_addr + (raw_bar_stage_tma) * 8, 4096);
                        mbarrier_arrive_expect_tx(k_ready_addr + (raw_bar_stage_tma) * 8, 4096);
                        mbarrier_arrive_expect_tx(v_ready_addr + (raw_bar_stage_tma) * 8, 4096);
                        {
                            mbarrier_arrive(g_ready_addr + (raw_bar_stage_tma) * 8);
                        }
                        int logical_chunk_tma = cstart_tma + chunk_tma;
                        int token_tma = (int)(bos_tma + (long long)logical_chunk_tma * 16);
                        #pragma unroll
                        for (int segment_tma = 0; segment_tma < 2; segment_tma++) {
                            int segment_offset_tma = segment_tma * 16 * 64 * 2;
                            int segment_dim_tma = segment_tma * 64;
                            tma_3d_gmem2smem(smem_q_addr + raw_stage_tma * 4096 + (unsigned int)segment_offset_tma, (&q_tma), segment_dim_tma, qk_head_tma, token_tma, q_ready_addr + (raw_bar_stage_tma) * 8);
                            tma_3d_gmem2smem(smem_k_addr + raw_stage_tma * 4096 + (unsigned int)segment_offset_tma, (&k_tma), segment_dim_tma, qk_head_tma, token_tma, k_ready_addr + (raw_bar_stage_tma) * 8);
                            tma_3d_gmem2smem(smem_v_addr + raw_stage_tma * 4096 + (unsigned int)segment_offset_tma, (&v_tma), segment_dim_tma, head_tma, token_tma, v_ready_addr + (raw_bar_stage_tma) * 8);
                        }
                    }
                    cumulative_chunk_tma += chunks_tma;
                }
            }
            unsigned int _phase_consumers_done_0 = 0;
            mbarrier_wait(consumers_done_addr, _phase_consumers_done_0);
            _phase_consumers_done_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive(cleanup_ready_addr);
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp == 15) {
        { // epilogue_main
            unsigned int sched_stage_epi = 0;
            int cumulative_chunk_epi = 0;
            int lhs_row_epi = lane % 8 + (lane / 8 & 1) * 8;
            int lhs_col_epi = lane / 16 * 8;
            int rhs_row_epi = lane % 8 + lane / 16 * 8;
            int rhs_col_epi = (lane / 8 & 1) * 8;
            unsigned int _phase_sched_ready_4 = 0;
            #pragma unroll 1
            for (int __4 = 0; __4 < total_work_items + 1; __4++) {
                mbarrier_wait(sched_ready_addr + (sched_stage_epi) * 8, _phase_sched_ready_4);
                unsigned int slot_4[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&slot_4[0])) : "r"(sched_slot_addr + sched_stage_epi * 4));
                unsigned int tile_epi = slot_4[0];
                if (elect_sync()) {
                    mbarrier_arrive(sched_done_addr + (sched_stage_epi) * 8);
                }
                sched_stage_epi += 1;
                if (sched_stage_epi == 8) { sched_stage_epi = 0; _phase_sched_ready_4 ^= 1; }
                if (tile_epi >= (unsigned int)total_work_items) {
                    break;
                }
                int item_base_epi = (int)tile_epi * 8;
                int _vec_load_3[4];
                {
                    int4 _iv4 = *reinterpret_cast<const int4*>(work_items + item_base_epi);
                    _vec_load_3[0 + 0] = _iv4.x;
                    _vec_load_3[0 + 1] = _iv4.y;
                    _vec_load_3[0 + 2] = _iv4.z;
                    _vec_load_3[0 + 3] = _iv4.w;
                }
                int seq_epi = _vec_load_3[0];
                int head_epi = _vec_load_3[1];
                int wstart_epi = _vec_load_3[2];
                int wend_epi = _vec_load_3[3];
                int cstart_epi = work_items[item_base_epi + 4];
                long long bos_epi = (long long)work_items[item_base_epi + 6];
                int chunks_epi = wend_epi - cstart_epi;
                long long checkpoint_base_epi = checkpoint_cu_starts[seq_epi];
                if (ENABLE_CHECKPOINTS != 0 && chunks_epi > 0) {
                    int checkpoint_event_epi = cumulative_chunk_epi;
                    unsigned int checkpoint_stage_epi = (unsigned int)(checkpoint_event_epi % 2);
                    mbarrier_wait(checkpoint_ready_addr + (checkpoint_stage_epi) * 8, (unsigned int)(checkpoint_event_epi / 2 & 1));
                    if (wstart_epi == 0) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int segment_checkpoint_epi = 0; segment_checkpoint_epi < 2; segment_checkpoint_epi++) {
                                tma_store_4d((&checkpoint_tma), segment_checkpoint_epi * 64, 0, head_epi, checkpoint_base_epi, smem_checkpoint_addr + checkpoint_stage_epi * 32768 + (unsigned int)(segment_checkpoint_epi * 128 * 64 * 2));
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    mbarrier_arrive(checkpoint_done_addr + (checkpoint_stage_epi) * 8);
                }
                #pragma unroll 1
                for (int chunk_epi = 0; chunk_epi < chunks_epi; chunk_epi++) {
                    int cumulative_epi = cumulative_chunk_epi + chunk_epi;
                    int logical_chunk_epi = cstart_epi + chunk_epi;
                    unsigned int decay_stage_epi = (unsigned int)(cumulative_epi % 2);
                    unsigned int diag_stage_epi = (unsigned int)(cumulative_epi % 4);
                    unsigned int intermediate_stage_epi = (unsigned int)(cumulative_epi % 2);
                    mbarrier_wait(qk_scale_ready_addr + (diag_stage_epi) * 8, (unsigned int)(cumulative_epi / 4 & 1));
                    mbarrier_wait(a_done_addr + (intermediate_stage_epi) * 8, (unsigned int)(cumulative_epi / 2 + 1 & 1));
                    float a_acc_epi[8];
                    #pragma unroll
                    for (int k_block_epi = 0; k_block_epi < 8; k_block_epi++) {
                        unsigned int a_frag_epi[4];
                        unsigned int b_frag_epi[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag_epi[0]), "=r"(a_frag_epi[1]), "=r"(a_frag_epi[2]), "=r"(a_frag_epi[3])
                            : "r"((smem_q_decay_addr + decay_stage_epi * 4096 + (unsigned int)((k_block_epi * 16 + lhs_col_epi) / 64 * 2048 + lhs_row_epi * 128 + (k_block_epi * 16 + lhs_col_epi) % 64 * 2 ^ ((k_block_epi * 16 + lhs_col_epi) / 64 * 2048 + lhs_row_epi * 128 + (k_block_epi * 16 + lhs_col_epi) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_epi[0]), "=r"(b_frag_epi[1]), "=r"(b_frag_epi[2]), "=r"(b_frag_epi[3])
                            : "r"((smem_k_inv_addr + decay_stage_epi * 4096 + (unsigned int)((k_block_epi * 16 + rhs_col_epi) / 64 * 2048 + rhs_row_epi * 128 + (k_block_epi * 16 + rhs_col_epi) % 64 * 2 ^ ((k_block_epi * 16 + rhs_col_epi) / 64 * 2048 + rhs_row_epi * 128 + (k_block_epi * 16 + rhs_col_epi) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(a_acc_epi[0]), "=f"(a_acc_epi[1]), "=f"(a_acc_epi[2]), "=f"(a_acc_epi[3])
                            : "r"(a_frag_epi[0]), "r"(a_frag_epi[1]), "r"(a_frag_epi[2]), "r"(a_frag_epi[3]), "r"(b_frag_epi[0]), "r"(b_frag_epi[1]), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[0])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[1])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[2])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[3])));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(a_acc_epi[4]), "=f"(a_acc_epi[(4) + 1]), "=f"(a_acc_epi[(4) + 2]), "=f"(a_acc_epi[(4) + 3])
                            : "r"(a_frag_epi[0]), "r"(a_frag_epi[1]), "r"(a_frag_epi[2]), "r"(a_frag_epi[3]), "r"(b_frag_epi[2]), "r"(b_frag_epi[(2) + 1]), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[4])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[(4) + 1])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[(4) + 2])), "f"(((k_block_epi == 0) ? 0.0f : a_acc_epi[(4) + 3])));
                    }
                    float a_values_epi[8];
                    #pragma unroll
                    for (int accum_epi = 0; accum_epi < 8; accum_epi++) {
                        int row_epi = lane / 4 + accum_epi % 4 / 2 * 8;
                        int col_epi = accum_epi / 4 * 8 + (lane & 3) * 2 + (accum_epi & 1);
                        a_values_epi[accum_epi] = 0.0f;
                        if (row_epi >= col_epi) {
                            a_values_epi[accum_epi] = a_acc_epi[accum_epi];
                        }
                    }
                    unsigned int a_words_epi[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(a_values_epi[_lp*2 + 0], a_values_epi[_lp*2+1 + 0]));
                        a_words_epi[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(smem_a_addr + intermediate_stage_epi * 1024 + (unsigned int)(lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 ^ (lane / 16 * 8 / 16 * 512 + lane % 16 * 32 + lane / 16 * 8 % 16 * 2 >> 7 & 1) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[0])), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[1])), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[2])), "r"(*reinterpret_cast<const uint32_t*>(&a_words_epi[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(a_ready_addr + (intermediate_stage_epi) * 8);
                    mbarrier_arrive(decay_super_done_addr + (decay_stage_epi) * 8);
                    if (chunk_epi > 0) {
                        {
                            int checkpoint_event_loop_epi = cumulative_epi;
                            unsigned int checkpoint_stage_loop_epi = (unsigned int)(checkpoint_event_loop_epi % 2);
                            mbarrier_wait(checkpoint_ready_addr + (checkpoint_stage_loop_epi) * 8, (unsigned int)(checkpoint_event_loop_epi / 2 & 1));
                            if (logical_chunk_epi >= wstart_epi) {
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int checkpoint_segment_loop_epi = 0; checkpoint_segment_loop_epi < 2; checkpoint_segment_loop_epi++) {
                                        tma_store_4d((&checkpoint_tma), checkpoint_segment_loop_epi * 64, 0, head_epi, checkpoint_base_epi + (long long)logical_chunk_epi, smem_checkpoint_addr + checkpoint_stage_loop_epi * 32768 + (unsigned int)(checkpoint_segment_loop_epi * 128 * 64 * 2));
                                    }
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            mbarrier_arrive(checkpoint_done_addr + (checkpoint_stage_loop_epi) * 8);
                        }
                        int output_event_epi = cumulative_epi - 1;
                        unsigned int output_stage_epi = (unsigned int)(output_event_epi % 2);
                        mbarrier_wait(o_tma_ready_addr + (output_stage_epi) * 8, (unsigned int)(output_event_epi / 2 & 1));
                        if (logical_chunk_epi > wstart_epi) {
                            if (elect_sync()) {
                                #pragma unroll
                                for (int output_segment_epi = 0; output_segment_epi < 2; output_segment_epi++) {
                                    tma_store_3d((&out_tma), output_segment_epi * 64, head_epi, (int)(bos_epi + (long long)(logical_chunk_epi - 1) * 16), smem_o_addr + output_stage_epi * 4096 + (unsigned int)(output_segment_epi * 16 * 64 * 2));
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        mbarrier_arrive(o_tma_done_addr + (output_stage_epi) * 8);
                    }
                }
                if (chunks_epi > 0) {
                    int last_event_epi = cumulative_chunk_epi + chunks_epi - 1;
                    unsigned int last_o_stage_epi = (unsigned int)(last_event_epi % 2);
                    mbarrier_wait(o_tma_ready_addr + (last_o_stage_epi) * 8, (unsigned int)(last_event_epi / 2 & 1));
                    if (elect_sync()) {
                        #pragma unroll
                        for (int last_segment_epi = 0; last_segment_epi < 2; last_segment_epi++) {
                            tma_store_3d((&out_tma), last_segment_epi * 64, head_epi, (int)(bos_epi + (long long)(wend_epi - 1) * 16), smem_o_addr + last_o_stage_epi * 4096 + (unsigned int)(last_segment_epi * 16 * 64 * 2));
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(o_tma_done_addr + (last_o_stage_epi) * 8);
                }
                cumulative_chunk_epi += chunks_epi;
            }
            if (elect_sync()) {
                mbarrier_arrive(consumers_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef ENABLE_CHECKPOINTS
#undef G_INPUT_BF16
#undef LOOM_INF
#undef NUM_CHECKPOINT_PIPE_STAGES
#undef NUM_DECAY_PIPE_STAGES
#undef NUM_DIAG_PIPE_STAGES
#undef NUM_INTERMEDIATE_PIPE_STAGES
#undef NUM_O_PIPE_STAGES
#undef NUM_RAW_BAR_PIPE_STAGES
#undef NUM_RAW_PIPE_STAGES
#undef NUM_SCHED_PIPE_STAGES
#undef NUM_STATE_PIPE_STAGES
#undef SMEM_SCHED_SLOT_OFF
#undef SMEM_SCHED_SLOT_STAGE_BYTES
#undef SMEM_SCHED_SLOT_STRIDE
#undef SMEM_SMEM_A_MN_OFF
#undef SMEM_SMEM_A_MN_STAGE_BYTES
#undef SMEM_SMEM_A_MN_STRIDE
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_BETA_ALL_OFF
#undef SMEM_SMEM_BETA_ALL_STAGE_BYTES
#undef SMEM_SMEM_BETA_ALL_STRIDE
#undef SMEM_SMEM_BETA_OFF
#undef SMEM_SMEM_BETA_STAGE_BYTES
#undef SMEM_SMEM_BETA_STRIDE
#undef SMEM_SMEM_CHECKPOINT_OFF
#undef SMEM_SMEM_CHECKPOINT_STAGE_BYTES
#undef SMEM_SMEM_CHECKPOINT_STRIDE
#undef SMEM_SMEM_G_ALL_OFF
#undef SMEM_SMEM_G_ALL_STAGE_BYTES
#undef SMEM_SMEM_G_ALL_STRIDE
#undef SMEM_SMEM_G_OFF
#undef SMEM_SMEM_G_STAGE_BYTES
#undef SMEM_SMEM_G_STRIDE
#undef SMEM_SMEM_K_ALL_OFF
#undef SMEM_SMEM_K_ALL_STAGE_BYTES
#undef SMEM_SMEM_K_ALL_STRIDE
#undef SMEM_SMEM_K_DECAY_OFF
#undef SMEM_SMEM_K_DECAY_STAGE_BYTES
#undef SMEM_SMEM_K_DECAY_STRIDE
#undef SMEM_SMEM_K_INV_OFF
#undef SMEM_SMEM_K_INV_STAGE_BYTES
#undef SMEM_SMEM_K_INV_STRIDE
#undef SMEM_SMEM_K_OFF
#undef SMEM_SMEM_K_RESTORE_MN_OFF
#undef SMEM_SMEM_K_RESTORE_MN_STAGE_BYTES
#undef SMEM_SMEM_K_RESTORE_MN_STRIDE
#undef SMEM_SMEM_K_RESTORE_OFF
#undef SMEM_SMEM_K_RESTORE_STAGE_BYTES
#undef SMEM_SMEM_K_RESTORE_STRIDE
#undef SMEM_SMEM_K_STAGE_BYTES
#undef SMEM_SMEM_K_STRIDE
#undef SMEM_SMEM_O_ALL_OFF
#undef SMEM_SMEM_O_ALL_STAGE_BYTES
#undef SMEM_SMEM_O_ALL_STRIDE
#undef SMEM_SMEM_O_OFF
#undef SMEM_SMEM_O_STAGE_BYTES
#undef SMEM_SMEM_O_STRIDE
#undef SMEM_SMEM_Q_ALL_OFF
#undef SMEM_SMEM_Q_ALL_STAGE_BYTES
#undef SMEM_SMEM_Q_ALL_STRIDE
#undef SMEM_SMEM_Q_DECAY_OFF
#undef SMEM_SMEM_Q_DECAY_STAGE_BYTES
#undef SMEM_SMEM_Q_DECAY_STRIDE
#undef SMEM_SMEM_Q_OFF
#undef SMEM_SMEM_Q_STAGE_BYTES
#undef SMEM_SMEM_Q_STRIDE
#undef SMEM_SMEM_STATE_DIAG_OFF
#undef SMEM_SMEM_STATE_DIAG_STAGE_BYTES
#undef SMEM_SMEM_STATE_DIAG_STRIDE
#undef SMEM_SMEM_TINV_MN_OFF
#undef SMEM_SMEM_TINV_MN_STAGE_BYTES
#undef SMEM_SMEM_TINV_MN_STRIDE
#undef SMEM_SMEM_TINV_OFF
#undef SMEM_SMEM_TINV_STAGE_BYTES
#undef SMEM_SMEM_TINV_STRIDE
#undef SMEM_SMEM_V_ALL_OFF
#undef SMEM_SMEM_V_ALL_STAGE_BYTES
#undef SMEM_SMEM_V_ALL_STRIDE
#undef SMEM_SMEM_V_OFF
#undef SMEM_SMEM_V_STAGE_BYTES
#undef SMEM_SMEM_V_STRIDE
#undef SMEM_TINV_SCRATCH_OFF
#undef SMEM_TINV_SCRATCH_STAGE_BYTES
#undef SMEM_TINV_SCRATCH_STRIDE
#undef SMEM_TOTAL
#undef STORE_BETA_ACTIVE
#undef STORE_FINAL_STATE
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_Q_STATE_OFFSET
#undef TMEM_TMEM_STATE_INP_OFFSET
#undef TMEM_TMEM_STATE_K_OFFSET
#undef TMEM_TMEM_STATE_OFFSET
#undef TMEM_TMEM_U_ACC_OFFSET
#undef TMEM_TMEM_U_INP_OFFSET
#undef TMEM_TMEM_Y_INP_OFFSET
#undef USE_INITIAL_STATE
#undef a_done_addr
#undef a_ready_addr
#undef beta_done_addr
#undef beta_ready_addr
#undef checkpoint_done_addr
#undef checkpoint_ready_addr
#undef cleanup_ready_addr
#undef consumers_done_addr
#undef decay_super_done_addr
#undef decay_tcgen_done_addr
#undef g_done_addr
#undef g_ready_addr
#undef k_decay_inv_ready_addr
#undef k_done_addr
#undef k_ready_addr
#undef k_restore_done_addr
#undef o_acc_done_addr
#undef o_acc_ready_addr
#undef o_tma_done_addr
#undef o_tma_ready_addr
#undef q_done_addr
#undef q_ready_addr
#undef qk_scale_ready_addr
#undef sched_done_addr
#undef sched_ready_addr
#undef sched_slot_addr
#undef smem_a_addr
#undef smem_a_mn_addr
#undef smem_beta_addr
#undef smem_beta_all_addr
#undef smem_checkpoint_addr
#undef smem_g_addr
#undef smem_g_all_addr
#undef smem_k_addr
#undef smem_k_all_addr
#undef smem_k_decay_addr
#undef smem_k_inv_addr
#undef smem_k_restore_addr
#undef smem_k_restore_mn_addr
#undef smem_o_addr
#undef smem_o_all_addr
#undef smem_q_addr
#undef smem_q_all_addr
#undef smem_q_decay_addr
#undef smem_state_diag_addr
#undef smem_tinv_addr
#undef smem_tinv_mn_addr
#undef smem_v_addr
#undef smem_v_all_addr
#undef state_acc_done_addr
#undef state_diag_done_addr
#undef state_inp_ready_addr
#undef state_k_ready_addr
#undef state_read_done_addr
#undef tinv_done_addr
#undef tinv_ready_addr
#undef tinv_scratch_addr
#undef u_acc_ready_addr
#undef u_inp_ready_addr
#undef v_done_addr
#undef v_ready_addr
#undef y_inp_ready_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_ENVELOPE_OFFSET 0
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_Y_OFFSET 432
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_U_OFFSET 336
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DU_OFFSET 352
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_STATE_K_OFFSET 320
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DU_INP_OFFSET 440
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DY_OFFSET 320
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_NEG_DY_OFFSET 432
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DO_INP_OFFSET 440
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DSTATE_OFFSET 0
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DSTATE_INP_OFFSET 128
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DK_RESTORE_OFFSET 416
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DQ_OFFSET 368
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DK_DECAY_OFFSET 384
#define TMEM_FLASHKDA_BWD_PERSISTENT_C16_DK_INV_OFFSET 400
#define NUM_SCHED_PIPE_STAGES 8
#define NUM_RAW_PIPE_STAGES 2
#define NUM_OPERAND_PIPE_STAGES 2
#define NUM_INTERMEDIATE_PIPE_STAGES 2
#define NUM_TCGEN_DATA_PIPE_STAGES 1
#define NUM_DSTATE_RECURRENCE_PIPE_STAGES 1
#define NUM_U_SMEM_PIPE_STAGES 1
#define NUM_DSTATE_SMEM_PIPE_STAGES 1
#define NUM_BOUNDARY_PIPE_STAGES 1
#define NUM_BOUNDARY_LOCAL_GRAD_PIPE_STAGES 1
#define NUM_DK_RESTORE_PIPE_STAGES 1
#define NUM_DY_SMEM_PIPE_STAGES 1
#define NUM_BETA_DY_SMEM_PIPE_STAGES 2
#define NUM_DBETA_M_PIPE_STAGES 1
#define NUM_LOCAL_GRAD_PIPE_STAGES 1
#define NUM_QK_RAW_PIPE_STAGES 4
#define NUM_G_PREFIX_PIPE_STAGES 2
#define NUM_STATE_SMEM_PIPE_STAGES 2
#define SMEM_STATE_OPERAND_OFF 1024
#define SMEM_STATE_OPERAND_STAGE_BYTES 32768
#define SMEM_STATE_OPERAND_STRIDE 32768
#define SMEM_STATE_OPERAND_MN_OFF 1024
#define SMEM_STATE_OPERAND_MN_STAGE_BYTES 32768
#define SMEM_STATE_OPERAND_MN_STRIDE 32768
#define SMEM_STATE_PANEL_OFF 1024
#define SMEM_STATE_PANEL_STAGE_BYTES 16384
#define SMEM_STATE_PANEL_STRIDE 16384
#define SMEM_STATE_OPERAND_ALL_OFF 1024
#define SMEM_STATE_OPERAND_ALL_STAGE_BYTES 65536
#define SMEM_STATE_OPERAND_ALL_STRIDE 65536
#define SMEM_WORK_ITEM_OFF 230368
#define SMEM_WORK_ITEM_STAGE_BYTES 4
#define SMEM_WORK_ITEM_STRIDE 4
#define SMEM_RAW_Q_OFF 66560
#define SMEM_RAW_Q_STAGE_BYTES 4096
#define SMEM_RAW_Q_STRIDE 4096
#define SMEM_RAW_K_OFF 74752
#define SMEM_RAW_K_STAGE_BYTES 4096
#define SMEM_RAW_K_STRIDE 4096
#define SMEM_RAW_G_OFF 82944
#define SMEM_RAW_G_STAGE_BYTES 4096
#define SMEM_RAW_G_STRIDE 4096
#define SMEM_RAW_DO_OFF 91136
#define SMEM_RAW_DO_STAGE_BYTES 4096
#define SMEM_RAW_DO_STRIDE 4096
#define SMEM_RAW_DO_AMAJ_OFF 91136
#define SMEM_RAW_DO_AMAJ_STAGE_BYTES 4096
#define SMEM_RAW_DO_AMAJ_STRIDE 4096
#define SMEM_RAW_V_OFF 99328
#define SMEM_RAW_V_STAGE_BYTES 4096
#define SMEM_RAW_V_STRIDE 4096
#define SMEM_BETA_DY_SMEM_OFF 107520
#define SMEM_BETA_DY_SMEM_STAGE_BYTES 4096
#define SMEM_BETA_DY_SMEM_STRIDE 4096
#define SMEM_RAW_V_ALL_OFF 99328
#define SMEM_RAW_V_ALL_STAGE_BYTES 8192
#define SMEM_RAW_V_ALL_STRIDE 8192
#define SMEM_BETA_SMEM_OFF 220160
#define SMEM_BETA_SMEM_STAGE_BYTES 256
#define SMEM_BETA_SMEM_STRIDE 256
#define SMEM_BETA_SMEM_ALL_OFF 220160
#define SMEM_BETA_SMEM_ALL_STAGE_BYTES 512
#define SMEM_BETA_SMEM_ALL_STRIDE 512
#define SMEM_DBETA_M_SMEM_OFF 220672
#define SMEM_DBETA_M_SMEM_STAGE_BYTES 64
#define SMEM_DBETA_M_SMEM_STRIDE 64
#define SMEM_DBETA_RED_SMEM_OFF 220736
#define SMEM_DBETA_RED_SMEM_STAGE_BYTES 256
#define SMEM_DBETA_RED_SMEM_STRIDE 256
#define SMEM_QK_NORM_SMEM_OFF 220992
#define SMEM_QK_NORM_SMEM_STAGE_BYTES 128
#define SMEM_QK_NORM_SMEM_STRIDE 128
#define SMEM_QK_NORM_SMEM_ALL_OFF 220992
#define SMEM_QK_NORM_SMEM_ALL_STAGE_BYTES 512
#define SMEM_QK_NORM_SMEM_ALL_STRIDE 512
#define SMEM_QK_RED_SMEM_OFF 221504
#define SMEM_QK_RED_SMEM_STAGE_BYTES 256
#define SMEM_QK_RED_SMEM_STRIDE 256
#define SMEM_RAW_DO_ALL_OFF 91136
#define SMEM_RAW_DO_ALL_STAGE_BYTES 8192
#define SMEM_RAW_DO_ALL_STRIDE 8192
#define SMEM_RAW_Q_ALL_OFF 66560
#define SMEM_RAW_Q_ALL_STAGE_BYTES 8192
#define SMEM_RAW_Q_ALL_STRIDE 8192
#define SMEM_RAW_K_ALL_OFF 74752
#define SMEM_RAW_K_ALL_STAGE_BYTES 8192
#define SMEM_RAW_K_ALL_STRIDE 8192
#define SMEM_RAW_G_ALL_OFF 82944
#define SMEM_RAW_G_ALL_STAGE_BYTES 8192
#define SMEM_RAW_G_ALL_STRIDE 8192
#define SMEM_G_PREFIX_OFF 115712
#define SMEM_G_PREFIX_STAGE_BYTES 8192
#define SMEM_G_PREFIX_STRIDE 8192
#define SMEM_G_PREFIX_ALL_OFF 115712
#define SMEM_G_PREFIX_ALL_STAGE_BYTES 16384
#define SMEM_G_PREFIX_ALL_STRIDE 16384
#define SMEM_K_INV_OPERAND_OFF 132096
#define SMEM_K_INV_OPERAND_STAGE_BYTES 4096
#define SMEM_K_INV_OPERAND_STRIDE 4096
#define SMEM_K_INV_LEAD16_OFF 132096
#define SMEM_K_INV_LEAD16_STAGE_BYTES 4096
#define SMEM_K_INV_LEAD16_STRIDE 4096
#define SMEM_K_INV_AMAJ_OFF 132096
#define SMEM_K_INV_AMAJ_STAGE_BYTES 4096
#define SMEM_K_INV_AMAJ_STRIDE 4096
#define SMEM_K_DECAY_OPERAND_OFF 140288
#define SMEM_K_DECAY_OPERAND_STAGE_BYTES 4096
#define SMEM_K_DECAY_OPERAND_STRIDE 4096
#define SMEM_K_DECAY_LEAD16_OFF 140288
#define SMEM_K_DECAY_LEAD16_STAGE_BYTES 4096
#define SMEM_K_DECAY_LEAD16_STRIDE 4096
#define SMEM_K_DECAY_TRANS_OFF 140288
#define SMEM_K_DECAY_TRANS_STAGE_BYTES 4096
#define SMEM_K_DECAY_TRANS_STRIDE 4096
#define SMEM_Q_DECAY_OPERAND_OFF 148480
#define SMEM_Q_DECAY_OPERAND_STAGE_BYTES 4096
#define SMEM_Q_DECAY_OPERAND_STRIDE 4096
#define SMEM_Q_DECAY_TRANS_OFF 148480
#define SMEM_Q_DECAY_TRANS_STAGE_BYTES 4096
#define SMEM_Q_DECAY_TRANS_STRIDE 4096
#define SMEM_K_RESTORE_OPERAND_OFF 156672
#define SMEM_K_RESTORE_OPERAND_STAGE_BYTES 4096
#define SMEM_K_RESTORE_OPERAND_STRIDE 4096
#define SMEM_K_RESTORE_LEAD16_OFF 156672
#define SMEM_K_RESTORE_LEAD16_STAGE_BYTES 4096
#define SMEM_K_RESTORE_LEAD16_STRIDE 4096
#define SMEM_K_INV_ALL_OFF 132096
#define SMEM_K_INV_ALL_STAGE_BYTES 8192
#define SMEM_K_INV_ALL_STRIDE 8192
#define SMEM_K_DECAY_ALL_OFF 140288
#define SMEM_K_DECAY_ALL_STAGE_BYTES 8192
#define SMEM_K_DECAY_ALL_STRIDE 8192
#define SMEM_Q_DECAY_ALL_OFF 148480
#define SMEM_Q_DECAY_ALL_STAGE_BYTES 8192
#define SMEM_Q_DECAY_ALL_STRIDE 8192
#define SMEM_K_RESTORE_ALL_OFF 156672
#define SMEM_K_RESTORE_ALL_STAGE_BYTES 8192
#define SMEM_K_RESTORE_ALL_STRIDE 8192
#define SMEM_TINV_SCRATCH_OFF 164864
#define SMEM_TINV_SCRATCH_STAGE_BYTES 512
#define SMEM_TINV_SCRATCH_STRIDE 512
#define SMEM_INTERMEDIATE_A_OFF 165376
#define SMEM_INTERMEDIATE_A_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_A_STRIDE 2560
#define SMEM_INTERMEDIATE_TINV_OFF 165888
#define SMEM_INTERMEDIATE_TINV_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_TINV_STRIDE 2560
#define SMEM_INTERMEDIATE_DA_OFF 166400
#define SMEM_INTERMEDIATE_DA_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_DA_STRIDE 2560
#define SMEM_INTERMEDIATE_DM_OFF 166912
#define SMEM_INTERMEDIATE_DM_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_DM_STRIDE 2560
#define SMEM_INTERMEDIATE_NDM_OFF 167424
#define SMEM_INTERMEDIATE_NDM_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_NDM_STRIDE 2560
#define SMEM_INTERMEDIATE_A_MN_OFF 165376
#define SMEM_INTERMEDIATE_A_MN_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_A_MN_STRIDE 2560
#define SMEM_INTERMEDIATE_TINV_MN_OFF 165888
#define SMEM_INTERMEDIATE_TINV_MN_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_TINV_MN_STRIDE 2560
#define SMEM_INTERMEDIATE_DA_MN_OFF 166400
#define SMEM_INTERMEDIATE_DA_MN_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_DA_MN_STRIDE 2560
#define SMEM_INTERMEDIATE_NDM_MN_OFF 167424
#define SMEM_INTERMEDIATE_NDM_MN_STAGE_BYTES 512
#define SMEM_INTERMEDIATE_NDM_MN_STRIDE 2560
#define SMEM_STATE_SCALE_DIAG_OFF 170496
#define SMEM_STATE_SCALE_DIAG_STAGE_BYTES 512
#define SMEM_STATE_SCALE_DIAG_STRIDE 512
#define SMEM_U_SMEM_OFF 179200
#define SMEM_U_SMEM_STAGE_BYTES 4096
#define SMEM_U_SMEM_STRIDE 4096
#define SMEM_U_SMEM_ALL_OFF 179200
#define SMEM_U_SMEM_ALL_STAGE_BYTES 4096
#define SMEM_U_SMEM_ALL_STRIDE 4096
#define SMEM_U_LEAD16_OFF 179200
#define SMEM_U_LEAD16_STAGE_BYTES 4096
#define SMEM_U_LEAD16_STRIDE 4096
#define SMEM_DSTATE_SMEM_OFF 183296
#define SMEM_DSTATE_SMEM_STAGE_BYTES 32768
#define SMEM_DSTATE_SMEM_STRIDE 32768
#define SMEM_DSTATE_SMEM_MN_OFF 183296
#define SMEM_DSTATE_SMEM_MN_STAGE_BYTES 32768
#define SMEM_DSTATE_SMEM_MN_STRIDE 32768
#define SMEM_DSTATE_SMEM_ALL_OFF 183296
#define SMEM_DSTATE_SMEM_ALL_STAGE_BYTES 32768
#define SMEM_DSTATE_SMEM_ALL_STRIDE 32768
#define SMEM_DY_SMEM_OFF 216064
#define SMEM_DY_SMEM_STAGE_BYTES 4096
#define SMEM_DY_SMEM_STRIDE 4096
#define SMEM_DY_SMEM_ALL_OFF 216064
#define SMEM_DY_SMEM_ALL_STAGE_BYTES 4096
#define SMEM_DY_SMEM_ALL_STRIDE 4096
#define SMEM_DEBUG_DU_SMEM_OFF 221760
#define SMEM_DEBUG_DU_SMEM_STAGE_BYTES 4096
#define SMEM_DEBUG_DU_SMEM_STRIDE 4096
#define SMEM_DEBUG_DU_SMEM_ALL_OFF 221760
#define SMEM_DEBUG_DU_SMEM_ALL_STAGE_BYTES 4096
#define SMEM_DEBUG_DU_SMEM_ALL_STRIDE 4096
#define SMEM_BOUNDARY_STATE_SMEM_OFF 221760
#define SMEM_BOUNDARY_STATE_SMEM_STAGE_BYTES 512
#define SMEM_BOUNDARY_STATE_SMEM_STRIDE 512
#define SMEM_TOTAL 230400
#define THREADS 512
#define validate_outputs 0
#define USE_DSTATE_IN 1

extern "C" {

__global__ __launch_bounds__(512, 1) void kernel_flashkda_backward_persistent_c16(
    unsigned int* __restrict__ dynamic_counter, const __grid_constant__ CUtensorMap q_tma,
    const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap g_tma,
    const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap v_tma,
    const __grid_constant__ CUtensorMap state_tma, float* __restrict__ dfinal_state,
    const __grid_constant__ CUtensorMap dv_tma, __nv_bfloat16* __restrict__ dq_out,
    __nv_bfloat16* __restrict__ dk_out, float* __restrict__ dgate_out,
    float* __restrict__ dgate_boundary_out, float* __restrict__ dinitial_state,
    float* __restrict__ A_log, float* __restrict__ dt_bias,
    const __grid_constant__ CUtensorMap beta_tma, float* __restrict__ dbeta,
    long long* __restrict__ cu_seqlens, long long* __restrict__ checkpoint_cu_starts,
    int* __restrict__ work_items, unsigned int* __restrict__ visits, float* __restrict__ observed,
    float* __restrict__ raw_observed, float* __restrict__ gate_observed,
    float* __restrict__ operand_observed, float* __restrict__ kk_observed,
    float* __restrict__ tinv_observed, float* __restrict__ a_observed,
    float* __restrict__ tcgen_observed, float* __restrict__ dstate_observed,
    float* __restrict__ dk_restore_observed, float* __restrict__ da_observed,
    float* __restrict__ dm_observed, float* __restrict__ local_grad_observed,
    float* __restrict__ assembled_grad_observed, int total_work_items, int uniform_work_items,
    int observed_chunks, int num_qk_heads, int num_heads, int enable_kk, int enable_tinv,
    float scale, float lower_bound) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // Kernel setup ops
  __nv_bfloat16* state_operand = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int state_operand_addr = smem + 1024;
  __nv_bfloat16* state_operand_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int state_operand_mn_addr = smem + 1024;
  __nv_bfloat16* state_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int state_panel_addr = smem + 1024;
  __nv_bfloat16* state_operand_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int state_operand_all_addr = smem + 1024;
  unsigned int* work_item = reinterpret_cast<unsigned int*>(smem_raw + 230368);
  const int work_item_addr = smem + 230368;
  __nv_bfloat16* raw_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
  const int raw_q_addr = smem + 66560;
  __nv_bfloat16* raw_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
  const int raw_k_addr = smem + 74752;
  __nv_bfloat16* raw_g = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
  const int raw_g_addr = smem + 82944;
  __nv_bfloat16* raw_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
  const int raw_do_addr = smem + 91136;
  __nv_bfloat16* raw_do_amaj = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
  const int raw_do_amaj_addr = smem + 91136;
  __nv_bfloat16* raw_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
  const int raw_v_addr = smem + 99328;
  __nv_bfloat16* beta_dy_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
  const int beta_dy_smem_addr = smem + 107520;
  __nv_bfloat16* raw_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
  const int raw_v_all_addr = smem + 99328;
  __nv_bfloat16* beta_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 220160);
  const int beta_smem_addr = smem + 220160;
  __nv_bfloat16* beta_smem_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 220160);
  const int beta_smem_all_addr = smem + 220160;
  float* dbeta_m_smem = reinterpret_cast<float*>(smem_raw + 220672);
  const int dbeta_m_smem_addr = smem + 220672;
  float* dbeta_red_smem = reinterpret_cast<float*>(smem_raw + 220736);
  const int dbeta_red_smem_addr = smem + 220736;
  float* qk_norm_smem = reinterpret_cast<float*>(smem_raw + 220992);
  const int qk_norm_smem_addr = smem + 220992;
  float* qk_norm_smem_all = reinterpret_cast<float*>(smem_raw + 220992);
  const int qk_norm_smem_all_addr = smem + 220992;
  float* qk_red_smem = reinterpret_cast<float*>(smem_raw + 221504);
  const int qk_red_smem_addr = smem + 221504;
  __nv_bfloat16* raw_do_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
  const int raw_do_all_addr = smem + 91136;
  __nv_bfloat16* raw_q_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
  const int raw_q_all_addr = smem + 66560;
  __nv_bfloat16* raw_k_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
  const int raw_k_all_addr = smem + 74752;
  __nv_bfloat16* raw_g_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
  const int raw_g_all_addr = smem + 82944;
  float* g_prefix = reinterpret_cast<float*>(smem_raw + 115712);
  const int g_prefix_addr = smem + 115712;
  float* g_prefix_all = reinterpret_cast<float*>(smem_raw + 115712);
  const int g_prefix_all_addr = smem + 115712;
  __nv_bfloat16* k_inv_operand = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
  const int k_inv_operand_addr = smem + 132096;
  __nv_bfloat16* k_inv_lead16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
  const int k_inv_lead16_addr = smem + 132096;
  __nv_bfloat16* k_inv_amaj = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
  const int k_inv_amaj_addr = smem + 132096;
  __nv_bfloat16* k_decay_operand = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
  const int k_decay_operand_addr = smem + 140288;
  __nv_bfloat16* k_decay_lead16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
  const int k_decay_lead16_addr = smem + 140288;
  __nv_bfloat16* k_decay_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
  const int k_decay_trans_addr = smem + 140288;
  __nv_bfloat16* q_decay_operand = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
  const int q_decay_operand_addr = smem + 148480;
  __nv_bfloat16* q_decay_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
  const int q_decay_trans_addr = smem + 148480;
  __nv_bfloat16* k_restore_operand = reinterpret_cast<__nv_bfloat16*>(smem_raw + 156672);
  const int k_restore_operand_addr = smem + 156672;
  __nv_bfloat16* k_restore_lead16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 156672);
  const int k_restore_lead16_addr = smem + 156672;
  __nv_bfloat16* k_inv_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
  const int k_inv_all_addr = smem + 132096;
  __nv_bfloat16* k_decay_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
  const int k_decay_all_addr = smem + 140288;
  __nv_bfloat16* q_decay_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
  const int q_decay_all_addr = smem + 148480;
  __nv_bfloat16* k_restore_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 156672);
  const int k_restore_all_addr = smem + 156672;
  __nv_bfloat16* tinv_scratch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
  const int tinv_scratch_addr = smem + 164864;
  __nv_bfloat16* intermediate_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165376);
  const int intermediate_a_addr = smem + 165376;
  __nv_bfloat16* intermediate_tinv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165888);
  const int intermediate_tinv_addr = smem + 165888;
  __nv_bfloat16* intermediate_da = reinterpret_cast<__nv_bfloat16*>(smem_raw + 166400);
  const int intermediate_da_addr = smem + 166400;
  __nv_bfloat16* intermediate_dm = reinterpret_cast<__nv_bfloat16*>(smem_raw + 166912);
  const int intermediate_dm_addr = smem + 166912;
  __nv_bfloat16* intermediate_ndm = reinterpret_cast<__nv_bfloat16*>(smem_raw + 167424);
  const int intermediate_ndm_addr = smem + 167424;
  __nv_bfloat16* intermediate_a_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165376);
  const int intermediate_a_mn_addr = smem + 165376;
  __nv_bfloat16* intermediate_tinv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165888);
  const int intermediate_tinv_mn_addr = smem + 165888;
  __nv_bfloat16* intermediate_da_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 166400);
  const int intermediate_da_mn_addr = smem + 166400;
  __nv_bfloat16* intermediate_ndm_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 167424);
  const int intermediate_ndm_mn_addr = smem + 167424;
  __nv_bfloat16* state_scale_diag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 170496);
  const int state_scale_diag_addr = smem + 170496;
  __nv_bfloat16* u_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 179200);
  const int u_smem_addr = smem + 179200;
  __nv_bfloat16* u_smem_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 179200);
  const int u_smem_all_addr = smem + 179200;
  __nv_bfloat16* u_lead16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 179200);
  const int u_lead16_addr = smem + 179200;
  __nv_bfloat16* dstate_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 183296);
  const int dstate_smem_addr = smem + 183296;
  __nv_bfloat16* dstate_smem_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 183296);
  const int dstate_smem_mn_addr = smem + 183296;
  __nv_bfloat16* dstate_smem_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 183296);
  const int dstate_smem_all_addr = smem + 183296;
  __nv_bfloat16* dy_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 216064);
  const int dy_smem_addr = smem + 216064;
  __nv_bfloat16* dy_smem_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 216064);
  const int dy_smem_all_addr = smem + 216064;
  __nv_bfloat16* debug_du_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 221760);
  const int debug_du_smem_addr = smem + 221760;
  __nv_bfloat16* debug_du_smem_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 221760);
  const int debug_du_smem_all_addr = smem + 221760;
  float* boundary_state_smem = reinterpret_cast<float*>(smem_raw + 221760);
  const int boundary_state_smem_addr = smem + 221760;

  // Mbarrier init (50 groups, 89 barriers)
  // Mbarriers at smem_raw[0..712)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // --- pipeline 'sched_pipe' ---
      // sched_ready: 8 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      mbarrier_init(smem + 16, 1);
      mbarrier_init(smem + 24, 1);
      mbarrier_init(smem + 32, 1);
      mbarrier_init(smem + 40, 1);
      mbarrier_init(smem + 48, 1);
      mbarrier_init(smem + 56, 1);
      // sched_done: 8 barriers, init_count=15
      mbarrier_init(smem + 64, 15);
      mbarrier_init(smem + 72, 15);
      mbarrier_init(smem + 80, 15);
      mbarrier_init(smem + 88, 15);
      mbarrier_init(smem + 96, 15);
      mbarrier_init(smem + 104, 15);
      mbarrier_init(smem + 112, 15);
      mbarrier_init(smem + 120, 15);
      // --- pipeline 'raw_pipe' ---
      // raw_ready: 2 barriers, init_count=1
      mbarrier_init(smem + 128, 1);
      mbarrier_init(smem + 136, 1);
      // raw_done: 2 barriers, init_count=165
      mbarrier_init(smem + 144, 165);
      mbarrier_init(smem + 152, 165);
      // --- pipeline 'g_prefix_pipe' ---
      // g_prefix_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 160, 128);
      mbarrier_init(smem + 168, 128);
      // g_prefix_done: 2 barriers, init_count=128
      mbarrier_init(smem + 176, 128);
      mbarrier_init(smem + 184, 128);
      // --- pipeline 'state_smem_pipe' ---
      // state_ready: 2 barriers, init_count=1
      mbarrier_init(smem + 192, 1);
      mbarrier_init(smem + 200, 1);
      // state_slot_done: 2 barriers, init_count=1
      mbarrier_init(smem + 208, 1);
      mbarrier_init(smem + 216, 1);
      // state_cg2_done: 2 barriers, init_count=128
      mbarrier_init(smem + 224, 128);
      mbarrier_init(smem + 232, 128);
      // state_k_ready: 2 barriers, init_count=1
      mbarrier_init(smem + 240, 1);
      mbarrier_init(smem + 248, 1);
      // state_k_done: 2 barriers, init_count=4
      mbarrier_init(smem + 256, 4);
      mbarrier_init(smem + 264, 4);
      // --- pipeline 'operand_pipe' ---
      // k_decay_inv_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 272, 128);
      mbarrier_init(smem + 280, 128);
      // q_decay_k_restore_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 288, 128);
      mbarrier_init(smem + 296, 128);
      // decay_done: 2 barriers, init_count=1
      mbarrier_init(smem + 304, 1);
      mbarrier_init(smem + 312, 1);
      // --- pipeline 'intermediate_pipe' ---
      // tinv_ready: 2 barriers, init_count=32
      mbarrier_init(smem + 320, 32);
      mbarrier_init(smem + 328, 32);
      // a_ready: 2 barriers, init_count=32
      mbarrier_init(smem + 336, 32);
      mbarrier_init(smem + 344, 32);
      // da_ready: 2 barriers, init_count=32
      mbarrier_init(smem + 352, 32);
      mbarrier_init(smem + 360, 32);
      // dm_ready: 2 barriers, init_count=32
      mbarrier_init(smem + 368, 32);
      mbarrier_init(smem + 376, 32);
      // intermediate_done: 2 barriers, init_count=1
      mbarrier_init(smem + 384, 1);
      mbarrier_init(smem + 392, 1);
      // --- pipeline 'tcgen_data_pipe' ---
      // tcgen_inputs_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 400, 4);
      // tcgen_inputs_done: 1 barriers, init_count=1
      mbarrier_init(smem + 408, 1);
      // tcgen_products_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 416, 1);
      // tcgen_products_done: 1 barriers, init_count=4
      mbarrier_init(smem + 424, 4);
      // du_inp_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 432, 4);
      // dy_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 440, 1);
      // dy_done: 1 barriers, init_count=4
      mbarrier_init(smem + 448, 4);
      // neg_dy_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 456, 4);
      // dstate_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 464, 1);
      // --- pipeline 'dstate_recurrence_pipe' ---
      // dstate_inp_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 472, 4);
      // --- pipeline 'tcgen_data_pipe' ---
      // dstate_done: 1 barriers, init_count=8
      mbarrier_init(smem + 480, 8);
      // --- pipeline 'u_smem_pipe' ---
      // u_smem_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 488, 4);
      // --- pipeline 'dy_smem_pipe' ---
      // dy_smem_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 496, 4);
      // --- pipeline 'beta_dy_smem_pipe' ---
      // beta_dy_smem_ready: 2 barriers, init_count=4
      mbarrier_init(smem + 504, 4);
      mbarrier_init(smem + 512, 4);
      // beta_dy_smem_done: 2 barriers, init_count=2
      mbarrier_init(smem + 520, 2);
      mbarrier_init(smem + 528, 2);
      // --- pipeline 'dbeta_m_pipe' ---
      // dbeta_m_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 536, 1);
      // dbeta_m_done: 1 barriers, init_count=1
      mbarrier_init(smem + 544, 1);
      // --- pipeline 'dstate_smem_pipe' ---
      // dstate_smem_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 552, 4);
      // dstate_smem_done: 1 barriers, init_count=1
      mbarrier_init(smem + 560, 1);
      // --- pipeline 'boundary_pipe' ---
      // boundary_smem_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 568, 4);
      // boundary_state_ready: 1 barriers, init_count=4
      mbarrier_init(smem + 576, 4);
      // boundary_acc_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 584, 1);
      // --- pipeline 'boundary_local_grad_pipe' ---
      // boundary_local_grad_free: 1 barriers, init_count=4
      mbarrier_init(smem + 592, 4);
      // --- pipeline 'dk_restore_pipe' ---
      // dk_restore_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 600, 1);
      // dk_restore_done: 1 barriers, init_count=4
      mbarrier_init(smem + 608, 4);
      // --- pipeline 'local_grad_pipe' ---
      // local_grad_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 616, 1);
      // local_grad_done: 1 barriers, init_count=4
      mbarrier_init(smem + 624, 4);
      // --- pipeline 'qk_raw_pipe' ---
      // qk_raw_ready: 4 barriers, init_count=128
      mbarrier_init(smem + 632, 128);
      mbarrier_init(smem + 640, 128);
      mbarrier_init(smem + 648, 128);
      mbarrier_init(smem + 656, 128);
      // qk_raw_done: 4 barriers, init_count=128
      mbarrier_init(smem + 664, 128);
      mbarrier_init(smem + 672, 128);
      mbarrier_init(smem + 680, 128);
      mbarrier_init(smem + 688, 128);
      // consumers_done: 1 barriers, init_count=15
      mbarrier_init(smem + 696, 15);
      // cleanup_ready: 1 barriers, init_count=1
      mbarrier_init(smem + 704, 1);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
  }

  __syncwarp();

  // TMEM alloc (512 columns, 512 used)
  volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 712);
  if (warp == 13) {
    int _tmem_hold = smem + 712;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(_tmem_hold),
        "r"(512)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");

  const int mbar_base = smem;
#define sched_ready_addr (mbar_base + 0)
#define sched_done_addr (mbar_base + 64)
#define raw_ready_addr (mbar_base + 128)
#define raw_done_addr (mbar_base + 144)
#define g_prefix_ready_addr (mbar_base + 160)
#define g_prefix_done_addr (mbar_base + 176)
#define state_ready_addr (mbar_base + 192)
#define state_slot_done_addr (mbar_base + 208)
#define state_cg2_done_addr (mbar_base + 224)
#define state_k_ready_addr (mbar_base + 240)
#define state_k_done_addr (mbar_base + 256)
#define k_decay_inv_ready_addr (mbar_base + 272)
#define q_decay_k_restore_ready_addr (mbar_base + 288)
#define decay_done_addr (mbar_base + 304)
#define tinv_ready_addr (mbar_base + 320)
#define a_ready_addr (mbar_base + 336)
#define da_ready_addr (mbar_base + 352)
#define dm_ready_addr (mbar_base + 368)
#define intermediate_done_addr (mbar_base + 384)
#define tcgen_inputs_ready_addr (mbar_base + 400)
#define tcgen_inputs_done_addr (mbar_base + 408)
#define tcgen_products_ready_addr (mbar_base + 416)
#define tcgen_products_done_addr (mbar_base + 424)
#define du_inp_ready_addr (mbar_base + 432)
#define dy_ready_addr (mbar_base + 440)
#define dy_done_addr (mbar_base + 448)
#define neg_dy_ready_addr (mbar_base + 456)
#define dstate_ready_addr (mbar_base + 464)
#define dstate_inp_ready_addr (mbar_base + 472)
#define dstate_done_addr (mbar_base + 480)
#define u_smem_ready_addr (mbar_base + 488)
#define dy_smem_ready_addr (mbar_base + 496)
#define beta_dy_smem_ready_addr (mbar_base + 504)
#define beta_dy_smem_done_addr (mbar_base + 520)
#define dbeta_m_ready_addr (mbar_base + 536)
#define dbeta_m_done_addr (mbar_base + 544)
#define dstate_smem_ready_addr (mbar_base + 552)
#define dstate_smem_done_addr (mbar_base + 560)
#define boundary_smem_ready_addr (mbar_base + 568)
#define boundary_state_ready_addr (mbar_base + 576)
#define boundary_acc_ready_addr (mbar_base + 584)
#define boundary_local_grad_free_addr (mbar_base + 592)
#define dk_restore_ready_addr (mbar_base + 600)
#define dk_restore_done_addr (mbar_base + 608)
#define local_grad_ready_addr (mbar_base + 616)
#define local_grad_done_addr (mbar_base + 624)
#define qk_raw_ready_addr (mbar_base + 632)
#define qk_raw_done_addr (mbar_base + 664)
#define consumers_done_addr (mbar_base + 696)
#define cleanup_ready_addr (mbar_base + 704)
  const int taddr = tmem_addr_storage[0];

  // Kernel post-init ops
  const int tmem_flashkda_bwd_persistent_c16_envelope = taddr;
  const int tmem_flashkda_bwd_persistent_c16_y = taddr + 432;
  const int tmem_flashkda_bwd_persistent_c16_u = taddr + 336;
  const int tmem_flashkda_bwd_persistent_c16_du = taddr + 352;
  const int tmem_flashkda_bwd_persistent_c16_state_k = taddr + 320;
  const int tmem_flashkda_bwd_persistent_c16_du_inp = taddr + 440;
  const int tmem_flashkda_bwd_persistent_c16_dy = taddr + 320;
  const int tmem_flashkda_bwd_persistent_c16_neg_dy = taddr + 432;
  const int tmem_flashkda_bwd_persistent_c16_do_inp = taddr + 440;
  const int tmem_flashkda_bwd_persistent_c16_dstate = taddr;
  const int tmem_flashkda_bwd_persistent_c16_dstate_inp = taddr + 128;
  const int tmem_flashkda_bwd_persistent_c16_dk_restore = taddr + 416;
  const int tmem_flashkda_bwd_persistent_c16_dq = taddr + 368;
  const int tmem_flashkda_bwd_persistent_c16_dk_decay = taddr + 384;
  const int tmem_flashkda_bwd_persistent_c16_dk_inv = taddr + 400;

  // ---- Register redistribution for WGs split across roles ----
  // Dec phase frees registers before any WG attempts inc.
  if (warp >= 12 && warp <= 15) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
  }

  // ---- Role: compute0 ----
  if (warp <= 3) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
    {  // compute0_main
      unsigned int sched_stage0 = 0;
      unsigned int raw_stage0 = 0;
      unsigned int operand_stage0 = 0;
      unsigned int qk_raw_stage0 = 0;
      unsigned int g_prefix_stage0 = 0;
      int warp_id_in_role = (warp - 0);
      const int tmem_row_base0 = warp_id_in_role * 32 << 16;
      unsigned int _phase_sched_ready = 0;
      unsigned int _phase_raw_ready = 0;
      unsigned int _phase_g_prefix_done = 1;
      unsigned int _phase_qk_raw_done = 1;
      unsigned int _phase_decay_done = 1;
#pragma unroll 1
      for (int _ = 0; _ < total_work_items; _++) {
        mbarrier_wait(sched_ready_addr + (sched_stage0) * 8, _phase_sched_ready);
        unsigned int ticket_words[1];
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&ticket_words[0]))
                     : "r"(work_item_addr + sched_stage0 * 4));
        unsigned int tile0 = ticket_words[0];
        if (elect_sync()) {
          mbarrier_arrive(sched_done_addr + (sched_stage0) * 8);
        }
        sched_stage0 += 1;
        if (sched_stage0 == 8) {
          sched_stage0 = 0;
          _phase_sched_ready ^= 1;
        }
        if (tile0 >= (unsigned int)total_work_items) {
          break;
        }
        int item_base0 = (int)tile0 * 8;
        int head0 = work_items[item_base0 + 1];
        int write_start0 = work_items[item_base0 + 2];
        int write_end0 = work_items[item_base0 + 3];
        int compute_end0 = work_items[item_base0 + 5];
        int role_tid0 = warp_id_in_role * 32 + lane;
        float raw_sum0 = 0.0f;
        float gate_sum0 = 0.0f;
        float _expf_0 = __expf(A_log[head0]);
        float gate_rate0 = _expf_0;
        float gate_bias0 = dt_bias[head0 * 128 + role_tid0];
#pragma unroll 1
        for (int reverse0 = 0; reverse0 < compute_end0 - write_start0; reverse0++) {
          mbarrier_wait(raw_ready_addr + (raw_stage0) * 8, _phase_raw_ready);
          mbarrier_wait(g_prefix_done_addr + (g_prefix_stage0) * 8, _phase_g_prefix_done);
          float gate_prefix0 = 0.0f;
          float gate_last0 = 0.0f;
          float q_raw_values0[16];
          float k_raw_values0[16];
#pragma unroll
          for (int token0 = 0; token0 < 16; token0++) {
            int segment = role_tid0 / 64;
            int segment_col = role_tid0 - segment * 64;
            int swizzled_col = segment_col ^ (token0 & 7) * 8;
            int raw_index0 =
                (int)raw_stage0 * 16 * 128 + segment * 16 * 64 + token0 * 64 + swizzled_col;
            __nv_bfloat16 raw_q_value0 = raw_q_all[raw_index0];
            __nv_bfloat16 raw_k_value0 = raw_k_all[raw_index0];
            __nv_bfloat16 raw_g_value0 = raw_g_all[raw_index0];
            float _cvt_f32_0 = __bfloat162float(raw_q_value0);
            q_raw_values0[token0] = _cvt_f32_0;
            float _cvt_f32_1 = __bfloat162float(raw_k_value0);
            k_raw_values0[token0] = _cvt_f32_1;
            float _cvt_f32_2 = __bfloat162float(raw_q_value0);
            float _cvt_f32_3 = __bfloat162float(raw_k_value0);
            float _cvt_f32_4 = __bfloat162float(raw_g_value0);
            raw_sum0 += _cvt_f32_2 + _cvt_f32_3 + _cvt_f32_4;
            float _cvt_f32_5 = __bfloat162float(raw_g_value0);
            float gate_arg0 = gate_rate0 * (_cvt_f32_5 + gate_bias0);
            float _tanh_approx_0;
            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(gate_arg0 * 0.5f));
            float gate_sigmoid0 = _tanh_approx_0 * 0.5f + 0.5f;
            gate_prefix0 += lower_bound * 1.4426950408889634f * gate_sigmoid0;
            float _exp2_0 = approx_exp2(gate_prefix0);
            float gate_exp0 = _exp2_0;
            gate_last0 = gate_exp0;
            gate_sum0 += gate_exp0;
            int segment_0 = role_tid0 / 32;
            int segment_col_1 = role_tid0 - segment_0 * 32;
            int swizzled_col_2 = segment_col_1 ^ (token0 & 7) * 4;
            g_prefix_all[(int)raw_stage0 * 16 * 128 + segment_0 * 16 * 32 + token0 * 32 +
                         swizzled_col_2] = gate_exp0;
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(g_prefix_ready_addr + (g_prefix_stage0) * 8);
          unsigned int q_raw_words0[8];
          unsigned int k_raw_words0[8];
#pragma unroll
          for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(q_raw_values0[_lp * 2 + 0], q_raw_values0[_lp * 2 + 1 + 0]));
            q_raw_words0[_lp] = *(uint32_t*)&_bf2;
          }
#pragma unroll
          for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(k_raw_values0[_lp * 2 + 0], k_raw_values0[_lp * 2 + 1 + 0]));
            k_raw_words0[_lp] = *(uint32_t*)&_bf2;
          }
          mbarrier_wait(qk_raw_done_addr + (qk_raw_stage0) * 8, _phase_qk_raw_done);
          tmem_st_x8_u32(taddr + 448 + qk_raw_stage0 % 4 * 8 + (unsigned int)tmem_row_base0,
                         (const uint32_t*)q_raw_words0);
          tmem_st_x8_u32(taddr + 480 + qk_raw_stage0 % 4 * 8 + (unsigned int)tmem_row_base0,
                         (const uint32_t*)k_raw_words0);
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          mbarrier_arrive(qk_raw_ready_addr + (qk_raw_stage0) * 8);
          asm volatile("barrier.sync 8, 128;" ::: "memory");
          mbarrier_wait(decay_done_addr + (operand_stage0) * 8, _phase_decay_done);
          int state_scale_block0 = role_tid0 / 16;
          int state_scale_row0 = role_tid0 - state_scale_block0 * 16;
          float state_scale_values0[16];
          state_scale_values0[0] = 0.0f;
          state_scale_values0[1] = 0.0f;
          state_scale_values0[2] = 0.0f;
          state_scale_values0[3] = 0.0f;
          state_scale_values0[4] = 0.0f;
          state_scale_values0[5] = 0.0f;
          state_scale_values0[6] = 0.0f;
          state_scale_values0[7] = 0.0f;
          state_scale_values0[8] = 0.0f;
          state_scale_values0[9] = 0.0f;
          state_scale_values0[10] = 0.0f;
          state_scale_values0[11] = 0.0f;
          state_scale_values0[12] = 0.0f;
          state_scale_values0[13] = 0.0f;
          state_scale_values0[14] = 0.0f;
          state_scale_values0[15] = 0.0f;
          state_scale_values0[state_scale_row0] = gate_last0;
          unsigned int state_scale_words0[8];
#pragma unroll
          for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(
                state_scale_values0[_lp * 2 + 0], state_scale_values0[_lp * 2 + 1 + 0]));
            state_scale_words0[_lp] = *(uint32_t*)&_bf2;
          }
          int state_scale_stage0 = (int)operand_stage0 * 8 + state_scale_block0;
          asm volatile(
              "st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                  (state_scale_diag_addr + (unsigned int)(state_scale_stage0 * 512) +
                   (unsigned int)(state_scale_row0 * 32 ^ (state_scale_row0 * 32 >> 7 & 1) << 4))),
              "r"(state_scale_words0[0]), "r"(state_scale_words0[1]), "r"(state_scale_words0[2]),
              "r"(state_scale_words0[3])
              : "memory");
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                           (state_scale_diag_addr + (unsigned int)(state_scale_stage0 * 512) +
                            (unsigned int)(state_scale_row0 * 32 + 16 ^
                                           (state_scale_row0 * 32 + 16 >> 7 & 1) << 4))),
                       "r"(state_scale_words0[4]), "r"(state_scale_words0[5]),
                       "r"(state_scale_words0[6]), "r"(state_scale_words0[7])
                       : "memory");
          int decay_row0 = warp_id_in_role * 4 + lane / 8;
          int decay_lane0 = lane & 7;
          float q_row0[16];
          float k_row0[16];
          float q_sq0 = 0.0f;
          float k_sq0 = 0.0f;
#pragma unroll
          for (int dim_half0 = 0; dim_half0 < 2; dim_half0++) {
            int dim_base0 = dim_half0 * 64 + decay_lane0 * 8;
            unsigned int q_words0[4];
            unsigned int k_words0[4];
            int segment_1 = dim_base0 / 64;
            int segment_col_2 = dim_base0 - segment_1 * 64;
            int swizzled_col_1 = segment_col_2 ^ (decay_row0 & 7) * 8;
            int qk_index0 =
                (int)raw_stage0 * 16 * 128 + segment_1 * 16 * 64 + decay_row0 * 64 + swizzled_col_1;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&q_words0[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&q_words0[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&q_words0[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&q_words0[(0) + 3]))
                         : "r"(raw_q_all_addr + (unsigned int)(qk_index0 * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(*reinterpret_cast<uint32_t*>(&k_words0[0])),
                           "=r"(*reinterpret_cast<uint32_t*>(&k_words0[(0) + 1])),
                           "=r"(*reinterpret_cast<uint32_t*>(&k_words0[(0) + 2])),
                           "=r"(*reinterpret_cast<uint32_t*>(&k_words0[(0) + 3]))
                         : "r"(raw_k_all_addr + (unsigned int)(qk_index0 * 2)));
            float q_words0_f32[8];
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile(
                  "{\n\t"
                  "shl.b32 %0, %2, 16;\n\t"
                  "and.b32 %1, %2, 0xffff0000;\n\t"
                  "}\n"
                  : "=f"((&q_words0_f32[_pair * 2])[0]), "=f"((&q_words0_f32[_pair * 2])[1])
                  : "r"(q_words0[_pair]));
            }
            float k_words0_f32[8];
#pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
              asm volatile(
                  "{\n\t"
                  "shl.b32 %0, %2, 16;\n\t"
                  "and.b32 %1, %2, 0xffff0000;\n\t"
                  "}\n"
                  : "=f"((&k_words0_f32[_pair * 2])[0]), "=f"((&k_words0_f32[_pair * 2])[1])
                  : "r"(k_words0[_pair]));
            }
#pragma unroll
            for (int dim_local0 = 0; dim_local0 < 8; dim_local0++) {
              int row_reg0 = dim_half0 * 8 + dim_local0;
              q_row0[row_reg0] = q_words0_f32[dim_local0];
              k_row0[row_reg0] = k_words0_f32[dim_local0];
              float _fma_0 = __fmaf_rn(q_row0[row_reg0], q_row0[row_reg0], q_sq0);
              q_sq0 = _fma_0;
              float _fma_1 = __fmaf_rn(k_row0[row_reg0], k_row0[row_reg0], k_sq0);
              k_sq0 = _fma_1;
            }
          }
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sq0, 4);
          q_sq0 += _shfl_xor_0;
          float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, q_sq0, 2);
          q_sq0 += _shfl_xor_1;
          float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sq0, 1);
          q_sq0 += _shfl_xor_2;
          float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sq0, 4);
          k_sq0 += _shfl_xor_3;
          float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, k_sq0, 2);
          k_sq0 += _shfl_xor_4;
          float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sq0, 1);
          k_sq0 += _shfl_xor_5;
          float _rsqrt_0 = rsqrtf(q_sq0 + 1e-06f);
          float q_inv_norm0 = _rsqrt_0;
          float _rsqrt_1 = rsqrtf(k_sq0 + 1e-06f);
          float k_inv_norm0 = _rsqrt_1;
          if (decay_lane0 == 0) {
            int qk_norm_stage_base0 = (int)(qk_raw_stage0 % 4) * 2 * 16;
            qk_norm_smem_all[qk_norm_stage_base0 + decay_row0] = q_inv_norm0;
            qk_norm_smem_all[qk_norm_stage_base0 + 16 + decay_row0] = k_inv_norm0;
          }
#pragma unroll
          for (int dim_half1 = 0; dim_half1 < 2; dim_half1++) {
            int dim_base1 = dim_half1 * 64 + decay_lane0 * 8;
            float k_inv_values0[8];
            float k_decay_values0[8];
            float q_decay_values0[8];
            float k_restore_values0[8];
#pragma unroll
            for (int dim_local1 = 0; dim_local1 < 8; dim_local1++) {
              int row_reg1 = dim_half1 * 8 + dim_local1;
              int dim1 = dim_base1 + dim_local1;
              int segment_2 = dim1 / 32;
              int segment_col_3 = dim1 - segment_2 * 32;
              int swizzled_col_3 = segment_col_3 ^ (decay_row0 & 7) * 4;
              int prefix_index0 = (int)raw_stage0 * 16 * 128 + segment_2 * 16 * 32 +
                                  decay_row0 * 32 + swizzled_col_3;
              int segment_0_1 = dim1 / 32;
              int segment_col_1_1 = dim1 - segment_0_1 * 32;
              int swizzled_col_2_1 = segment_col_1_1 ^ 28;
              int prefix_last_index0 =
                  (int)raw_stage0 * 16 * 128 + segment_0_1 * 16 * 32 + 480 + swizzled_col_2_1;
              float prefix0 = g_prefix_all[prefix_index0];
              float prefix_last0 = g_prefix_all[prefix_last_index0];
              __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(q_row0[row_reg1] * q_inv_norm0 * scale);
              __nv_bfloat16 q_norm0 = _cvt_bf16_0;
              __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(k_row0[row_reg1] * k_inv_norm0);
              __nv_bfloat16 k_norm0 = _cvt_bf16_1;
              float _cvt_f32_6 = __bfloat162float(k_norm0);
              float _rcp_0 = approx_rcp(prefix0);
              __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_cvt_f32_6 * _rcp_0);
              __nv_bfloat16 k_inv_value0 = _cvt_bf16_2;
              float _cvt_f32_7 = __bfloat162float(k_inv_value0);
              k_inv_values0[dim_local1] = _cvt_f32_7;
              float _cvt_f32_9 = __bfloat162float(k_norm0);
              __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_cvt_f32_9 * prefix0);
              __nv_bfloat16 k_decay_value0 = _cvt_bf16_3;
              float _cvt_f32_10 = __bfloat162float(k_decay_value0);
              k_decay_values0[dim_local1] = _cvt_f32_10;
              float _cvt_f32_11 = __bfloat162float(q_norm0);
              __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_cvt_f32_11 * prefix0);
              __nv_bfloat16 q_decay_value0 = _cvt_bf16_4;
              float _cvt_f32_12 = __bfloat162float(q_decay_value0);
              q_decay_values0[dim_local1] = _cvt_f32_12;
              float _cvt_f32_13 = __bfloat162float(k_norm0);
              float _rcp_1 = approx_rcp(prefix0);
              __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_cvt_f32_13 * prefix_last0 * _rcp_1);
              __nv_bfloat16 k_restore_value0 = _cvt_bf16_5;
              float _cvt_f32_14 = __bfloat162float(k_restore_value0);
              k_restore_values0[dim_local1] = _cvt_f32_14;
            }
            unsigned int k_inv_words0[4];
            unsigned int k_decay_words0[4];
            unsigned int q_decay_words0[4];
            unsigned int k_restore_words0[4];
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(k_inv_values0[_lp * 2 + 0], k_inv_values0[_lp * 2 + 1 + 0]));
              k_inv_words0[_lp] = *(uint32_t*)&_bf2;
            }
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(k_decay_values0[_lp * 2 + 0], k_decay_values0[_lp * 2 + 1 + 0]));
              k_decay_words0[_lp] = *(uint32_t*)&_bf2;
            }
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(q_decay_values0[_lp * 2 + 0], q_decay_values0[_lp * 2 + 1 + 0]));
              q_decay_words0[_lp] = *(uint32_t*)&_bf2;
            }
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(k_restore_values0[_lp * 2 + 0], k_restore_values0[_lp * 2 + 1 + 0]));
              k_restore_words0[_lp] = *(uint32_t*)&_bf2;
            }
            int segment_3 = dim_base1 / 64;
            int segment_col_4 = dim_base1 - segment_3 * 64;
            int swizzled_col_4 = segment_col_4 ^ (decay_row0 & 7) * 8;
            int operand_index0 = (int)operand_stage0 * 16 * 128 + segment_3 * 16 * 64 +
                                 decay_row0 * 64 + swizzled_col_4;
            int operand_addr0 = operand_index0 * 2;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(k_inv_all_addr +
                                                                       (unsigned int)operand_addr0),
                         "r"(k_inv_words0[0]), "r"(k_inv_words0[1]), "r"(k_inv_words0[2]),
                         "r"(k_inv_words0[3])
                         : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(k_decay_all_addr +
                                                                       (unsigned int)operand_addr0),
                         "r"(k_decay_words0[0]), "r"(k_decay_words0[1]), "r"(k_decay_words0[2]),
                         "r"(k_decay_words0[3])
                         : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(q_decay_all_addr +
                                                                       (unsigned int)operand_addr0),
                         "r"(q_decay_words0[0]), "r"(q_decay_words0[1]), "r"(q_decay_words0[2]),
                         "r"(q_decay_words0[3])
                         : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(k_restore_all_addr +
                                                                       (unsigned int)operand_addr0),
                         "r"(k_restore_words0[0]), "r"(k_restore_words0[1]),
                         "r"(k_restore_words0[2]), "r"(k_restore_words0[3])
                         : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(k_decay_inv_ready_addr + (operand_stage0) * 8);
          mbarrier_arrive(q_decay_k_restore_ready_addr + (operand_stage0) * 8);
          mbarrier_arrive(raw_done_addr + (raw_stage0) * 8);
          raw_stage0 += 1;
          if (raw_stage0 == 2) {
            raw_stage0 = 0;
            _phase_raw_ready ^= 1;
          }
          operand_stage0 += 1;
          if (operand_stage0 == 2) {
            operand_stage0 = 0;
            _phase_decay_done ^= 1;
          }
          qk_raw_stage0 += 1;
          if (qk_raw_stage0 == 4) {
            qk_raw_stage0 = 0;
            _phase_qk_raw_done ^= 1;
          }
          g_prefix_stage0 += 1;
          if (g_prefix_stage0 == 2) {
            g_prefix_stage0 = 0;
            _phase_g_prefix_done ^= 1;
          }
        }
      }
      if (elect_sync()) {
        mbarrier_arrive(consumers_done_addr);
      }
    }
  }
  // ---- Role: compute1 ----
  if (warp >= 4 && warp <= 7) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
    {  // compute1_main
      unsigned int sched_stage1 = 0;
      unsigned int raw_stage1 = 0;
      unsigned int state_smem_stage1 = 0;
      unsigned int tcgen_data_stage1 = 0;
      unsigned int dstate_smem_stage1 = 0;
      unsigned int beta_dy_smem_stage1 = 0;
      unsigned int dbeta_m_stage1 = 0;
      unsigned int boundary_stage1 = 0;
      int dstate_smem_slot_acquired1 = 0;
      int warp_id_in_role_1 = (warp - 4);
      int value_dim_base1 = warp_id_in_role_1 * 32;
      int role_tid1 = warp_id_in_role_1 * 32 + lane;
      int ov_token1 = lane / 16 * 8 + (lane & 7);
      int ov_col1 = (lane / 8 & 1) * 8;
      const int tmem_row_base1 = warp_id_in_role_1 * 32 << 16;
      unsigned int _phase_sched_ready_1 = 0;
      unsigned int _phase_dstate_smem_done = 1;
      unsigned int _phase_tcgen_inputs_done = 1;
      unsigned int _phase_raw_ready_1 = 0;
      unsigned int _phase_state_k_ready = 0;
      unsigned int _phase_tcgen_products_ready = 0;
      unsigned int _phase_dy_ready = 0;
      unsigned int _phase_beta_dy_smem_done = 1;
      unsigned int _phase_dbeta_m_ready = 0;
      unsigned int _phase_dstate_ready = 0;
      unsigned int _phase_boundary_acc_ready = 0;
#pragma unroll 1
      for (int __1 = 0; __1 < total_work_items; __1++) {
        mbarrier_wait(sched_ready_addr + (sched_stage1) * 8, _phase_sched_ready_1);
        unsigned int ticket_words_1[1];
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&ticket_words_1[0]))
                     : "r"(work_item_addr + sched_stage1 * 4));
        unsigned int tile1 = ticket_words_1[0];
        if (elect_sync()) {
          mbarrier_arrive(sched_done_addr + (sched_stage1) * 8);
        }
        sched_stage1 += 1;
        if (sched_stage1 == 8) {
          sched_stage1 = 0;
          _phase_sched_ready_1 ^= 1;
        }
        if (tile1 >= (unsigned int)total_work_items) {
          break;
        }
        int item_base1 = (int)tile1 * 8;
        int sequence1 = work_items[item_base1];
        int head1 = work_items[item_base1 + 1];
        int write_start1 = work_items[item_base1 + 2];
        int write_end1 = work_items[item_base1 + 3];
        int compute_end1 = work_items[item_base1 + 5];
        float checksum_sum1 = 0.0f;
        long long bos1 = cu_seqlens[sequence1];
        long long eos1 = cu_seqlens[sequence1 + 1];
        int sequence_chunks1 = (int)((eos1 - bos1) / 16);
        int seed_dfinal1 = (int)(compute_end1 == sequence_chunks1);
        if (dstate_smem_slot_acquired1 == 0) {
          mbarrier_wait(dstate_smem_done_addr + (dstate_smem_stage1) * 8, _phase_dstate_smem_done);
        }
#pragma unroll
        for (int dstate_init_block1 = 0; dstate_init_block1 < 4; dstate_init_block1++) {
          float dstate_init_values1[32];
          dstate_init_values1[0] = 0.0f;
          dstate_init_values1[1] = 0.0f;
          dstate_init_values1[2] = 0.0f;
          dstate_init_values1[3] = 0.0f;
          dstate_init_values1[4] = 0.0f;
          dstate_init_values1[5] = 0.0f;
          dstate_init_values1[6] = 0.0f;
          dstate_init_values1[7] = 0.0f;
          dstate_init_values1[8] = 0.0f;
          dstate_init_values1[9] = 0.0f;
          dstate_init_values1[10] = 0.0f;
          dstate_init_values1[11] = 0.0f;
          dstate_init_values1[12] = 0.0f;
          dstate_init_values1[13] = 0.0f;
          dstate_init_values1[14] = 0.0f;
          dstate_init_values1[15] = 0.0f;
          dstate_init_values1[16] = 0.0f;
          dstate_init_values1[17] = 0.0f;
          dstate_init_values1[18] = 0.0f;
          dstate_init_values1[19] = 0.0f;
          dstate_init_values1[20] = 0.0f;
          dstate_init_values1[21] = 0.0f;
          dstate_init_values1[22] = 0.0f;
          dstate_init_values1[23] = 0.0f;
          dstate_init_values1[24] = 0.0f;
          dstate_init_values1[25] = 0.0f;
          dstate_init_values1[26] = 0.0f;
          dstate_init_values1[27] = 0.0f;
          dstate_init_values1[28] = 0.0f;
          dstate_init_values1[29] = 0.0f;
          dstate_init_values1[30] = 0.0f;
          dstate_init_values1[31] = 0.0f;
          if (USE_DSTATE_IN != 0 && seed_dfinal1 != 0) {
            long long dstate_init_base1 =
                (((long long)sequence1 * (long long)num_heads + (long long)head1) * 128 +
                 (long long)role_tid1) *
                    128 +
                (long long)(dstate_init_block1 * 32);
#pragma unroll
            for (int dstate_init_vec1 = 0; dstate_init_vec1 < 4; dstate_init_vec1++) {
              {
                unsigned _ldv8_0_0;
                unsigned _ldv8_0_1;
                unsigned _ldv8_0_2;
                unsigned _ldv8_0_3;
                unsigned _ldv8_0_4;
                unsigned _ldv8_0_5;
                unsigned _ldv8_0_6;
                unsigned _ldv8_0_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3),
                      "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7)
                    : "l"((const void*)(dfinal_state +
                                        (dstate_init_base1 + (long long)(dstate_init_vec1 * 8))))
                    : "memory");
                dstate_init_values1[dstate_init_vec1 * 8 + 0] = __uint_as_float(_ldv8_0_0);
                dstate_init_values1[dstate_init_vec1 * 8 + 1] = __uint_as_float(_ldv8_0_1);
                dstate_init_values1[dstate_init_vec1 * 8 + 2] = __uint_as_float(_ldv8_0_2);
                dstate_init_values1[dstate_init_vec1 * 8 + 3] = __uint_as_float(_ldv8_0_3);
                dstate_init_values1[dstate_init_vec1 * 8 + 4] = __uint_as_float(_ldv8_0_4);
                dstate_init_values1[dstate_init_vec1 * 8 + 5] = __uint_as_float(_ldv8_0_5);
                dstate_init_values1[dstate_init_vec1 * 8 + 6] = __uint_as_float(_ldv8_0_6);
                dstate_init_values1[dstate_init_vec1 * 8 + 7] = __uint_as_float(_ldv8_0_7);
              }
            }
          }
          tmem_st_x32_f32(
              taddr + (unsigned int)(dstate_init_block1 * 32) + (unsigned int)tmem_row_base1,
              dstate_init_values1);
          uint32_t dstate_init_values1_bf16[16];
#pragma unroll
          for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(
                dstate_init_values1[_lp * 2 + 0], dstate_init_values1[_lp * 2 + 1 + 0]));
            dstate_init_values1_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          asm volatile(
              "tcgen05.st.sync.aligned.32x32b.x16.b32"
              " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                  "r"(taddr + 128 + (unsigned int)(dstate_init_block1 * 16) +
                      (unsigned int)tmem_row_base1),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[3])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[4])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[5])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[6])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[7])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[8])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[9])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[10])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[11])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[12])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[13])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[14])),
              "r"(*reinterpret_cast<const uint32_t*>(&dstate_init_values1_bf16[15])));
#pragma unroll
          for (int dstate_init_atom1 = 0; dstate_init_atom1 < 4; dstate_init_atom1++) {
            int dstate_init_col1 = dstate_init_block1 * 32 + dstate_init_atom1 * 8;
            int segment_4 = dstate_init_col1 / 64;
            int segment_col_5 = dstate_init_col1 - segment_4 * 64;
            int swizzled_col_5 = segment_col_5 ^ (role_tid1 & 7) * 8;
            int dstate_init_smem_index1 = segment_4 * 128 * 64 + role_tid1 * 64 + swizzled_col_5;
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                             dstate_smem_all_addr + (unsigned int)(dstate_init_smem_index1 * 2)),
                         "r"(dstate_init_values1_bf16[dstate_init_atom1 * 4]),
                         "r"(dstate_init_values1_bf16[dstate_init_atom1 * 4 + 1]),
                         "r"(dstate_init_values1_bf16[dstate_init_atom1 * 4 + 2]),
                         "r"(dstate_init_values1_bf16[dstate_init_atom1 * 4 + 3])
                         : "memory");
          }
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (elect_sync()) {
          mbarrier_arrive(dstate_inp_ready_addr);
          mbarrier_arrive(dstate_smem_ready_addr + (dstate_smem_stage1) * 8);
        }
        dstate_smem_stage1 += 1;
        if (dstate_smem_stage1 == 1) {
          dstate_smem_stage1 = 0;
          _phase_dstate_smem_done ^= 1;
        }
        dstate_smem_slot_acquired1 = 0;
#pragma unroll 1
        for (int reverse1 = 0; reverse1 < compute_end1 - write_start1; reverse1++) {
          mbarrier_wait(tcgen_inputs_done_addr + (tcgen_data_stage1) * 8, _phase_tcgen_inputs_done);
          mbarrier_wait(raw_ready_addr + (raw_stage1) * 8, _phase_raw_ready_1);
          int beta_lane_token1 = (lane & 3) * 2;
          int beta_stage_base1 = (int)(raw_stage1 % 2) * 16 * 8;
          int beta_head1 = head1 % 8;
          __nv_bfloat16 beta_c0_bf1 =
              beta_smem_all[beta_stage_base1 + beta_lane_token1 * 8 + beta_head1];
          __nv_bfloat16 beta_c1_bf1 =
              beta_smem_all[beta_stage_base1 + (beta_lane_token1 + 1) * 8 + beta_head1];
          __nv_bfloat16 beta_c8_bf1 =
              beta_smem_all[beta_stage_base1 + (beta_lane_token1 + 8) * 8 + beta_head1];
          __nv_bfloat16 beta_c9_bf1 =
              beta_smem_all[beta_stage_base1 + (beta_lane_token1 + 9) * 8 + beta_head1];
          float _cvt_f32_15 = __bfloat162float(beta_c0_bf1);
          float beta_c0_1 = _cvt_f32_15;
          float _cvt_f32_16 = __bfloat162float(beta_c1_bf1);
          float beta_c1_1 = _cvt_f32_16;
          float _cvt_f32_17 = __bfloat162float(beta_c8_bf1);
          float beta_c8_1 = _cvt_f32_17;
          float _cvt_f32_18 = __bfloat162float(beta_c9_bf1);
          float beta_c9_1 = _cvt_f32_18;
          float beta_values1[4];
          beta_values1[0] = beta_c8_1;
          beta_values1[1] = beta_c9_1;
          beta_values1[2] = beta_c0_1;
          beta_values1[3] = beta_c1_1;
          unsigned int beta_words1[2];
#pragma unroll
          for (int _lp = 0; _lp < 2; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(beta_values1[_lp * 2 + 0], beta_values1[_lp * 2 + 1 + 0]));
            beta_words1[_lp] = *(uint32_t*)&_bf2;
          }
          mbarrier_wait(state_k_ready_addr + (state_smem_stage1) * 8, _phase_state_k_ready);
          float _tmem_load_0[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
              : "r"(taddr + 320 + (unsigned int)tmem_row_base1));
          float _tmem_load_1[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7]))
              : "r"(taddr + 320 + (unsigned int)tmem_row_base1 + 1048576));
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(state_k_done_addr + (state_smem_stage1) * 8);
          }
          uint32_t _tmem_load_0_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_0[_lp * 2 + 0], _tmem_load_0[_lp * 2 + 1 + 0]));
            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _tmem_load_1_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_1[_lp * 2 + 0], _tmem_load_1[_lp * 2 + 1 + 0]));
            _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          unsigned int v_fragment_lo1[4];
          unsigned int v_fragment_hi1[4];
          int segment_5 = (value_dim_base1 + ov_col1) / 64;
          int segment_col_6 = value_dim_base1 + ov_col1 - segment_5 * 64;
          int swizzled_col_6 = segment_col_6 ^ (ov_token1 & 7) * 8;
          unsigned int v_addr_lo1 =
              raw_v_all_addr + (unsigned int)(((int)raw_stage1 * 16 * 128 + segment_5 * 16 * 64 +
                                               ov_token1 * 64 + swizzled_col_6) *
                                              2);
          int segment_0_2 = (value_dim_base1 + 16 + ov_col1) / 64;
          int segment_col_1_2 = value_dim_base1 + 16 + ov_col1 - segment_0_2 * 64;
          int swizzled_col_2_2 = segment_col_1_2 ^ (ov_token1 & 7) * 8;
          unsigned int v_addr_hi1 =
              raw_v_all_addr + (unsigned int)(((int)raw_stage1 * 16 * 128 + segment_0_2 * 16 * 64 +
                                               ov_token1 * 64 + swizzled_col_2_2) *
                                              2);
          int segment_3_1 = (value_dim_base1 + ov_col1) / 64;
          int segment_col_4_1 = value_dim_base1 + ov_col1 - segment_3_1 * 64;
          int swizzled_col_5_1 = segment_col_4_1 ^ (ov_token1 & 7) * 8;
          unsigned int do_addr_lo1 =
              raw_do_all_addr + (unsigned int)(((int)raw_stage1 * 16 * 128 + segment_3_1 * 16 * 64 +
                                                ov_token1 * 64 + swizzled_col_5_1) *
                                               2);
          int segment_6 = (value_dim_base1 + 16 + ov_col1) / 64;
          int segment_col_7 = value_dim_base1 + 16 + ov_col1 - segment_6 * 64;
          int swizzled_col_8 = segment_col_7 ^ (ov_token1 & 7) * 8;
          unsigned int do_addr_hi1 =
              raw_do_all_addr + (unsigned int)(((int)raw_stage1 * 16 * 128 + segment_6 * 16 * 64 +
                                                ov_token1 * 64 + swizzled_col_8) *
                                               2);
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(v_fragment_lo1[0]), "=r"(v_fragment_lo1[1]), "=r"(v_fragment_lo1[2]),
                         "=r"(v_fragment_lo1[3])
                       : "r"(v_addr_lo1)
                       : "memory");
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(v_fragment_hi1[0]), "=r"(v_fragment_hi1[1]), "=r"(v_fragment_hi1[2]),
                         "=r"(v_fragment_hi1[3])
                       : "r"(v_addr_hi1)
                       : "memory");
          unsigned int do_fragment_lo1[4];
          unsigned int do_fragment_hi1[4];
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(do_fragment_lo1[0]), "=r"(do_fragment_lo1[1]),
                         "=r"(do_fragment_lo1[2]), "=r"(do_fragment_lo1[3])
                       : "r"(do_addr_lo1)
                       : "memory");
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                       : "=r"(do_fragment_hi1[0]), "=r"(do_fragment_hi1[1]),
                         "=r"(do_fragment_hi1[2]), "=r"(do_fragment_hi1[3])
                       : "r"(do_addr_hi1)
                       : "memory");
          unsigned int diff_words_lo1[4];
          unsigned int diff_words_hi1[4];
          __nv_bfloat162 y_pairs_lo1[4];
          __nv_bfloat162 y_pairs_hi1[4];
          unsigned int y_words_lo1[4];
          unsigned int y_words_hi1[4];
          unsigned int do_words_lo1[4];
          unsigned int do_words_hi1[4];
#pragma unroll
          for (int y_reg1 = 0; y_reg1 < 4; y_reg1++) {
            const int raw_reg1 = (1 - y_reg1 / 2) * 2 + (y_reg1 & 1);
            const int out_reg1 = y_reg1 ^ 2;
            uint32_t _bf16x2_sub_0;
            asm volatile("sub.rn.bf16x2 %0, %1, %2;"
                         : "=r"(_bf16x2_sub_0)
                         : "r"(v_fragment_lo1[raw_reg1]), "r"(_tmem_load_0_bf16[out_reg1]));
            diff_words_lo1[out_reg1] = _bf16x2_sub_0;
            uint32_t _bf16x2_sub_1;
            asm volatile("sub.rn.bf16x2 %0, %1, %2;"
                         : "=r"(_bf16x2_sub_1)
                         : "r"(v_fragment_hi1[raw_reg1]), "r"(_tmem_load_1_bf16[out_reg1]));
            diff_words_hi1[out_reg1] = _bf16x2_sub_1;
            y_pairs_lo1[out_reg1] =
                reinterpret_cast<__nv_bfloat162*>(diff_words_lo1 + out_reg1)[0] *
                reinterpret_cast<__nv_bfloat162*>(beta_words1 + y_reg1 / 2)[0];
            y_pairs_hi1[out_reg1] =
                reinterpret_cast<__nv_bfloat162*>(diff_words_hi1 + out_reg1)[0] *
                reinterpret_cast<__nv_bfloat162*>(beta_words1 + y_reg1 / 2)[0];
            y_words_lo1[out_reg1] = reinterpret_cast<unsigned int*>(y_pairs_lo1 + out_reg1)[0];
            y_words_hi1[out_reg1] = reinterpret_cast<unsigned int*>(y_pairs_hi1 + out_reg1)[0];
            do_words_lo1[out_reg1] = do_fragment_lo1[raw_reg1];
            do_words_hi1[out_reg1] = do_fragment_hi1[raw_reg1];
          }
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 432 + (unsigned int)tmem_row_base1),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo1[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo1[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo1[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_lo1[3])));
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 432 + (unsigned int)tmem_row_base1 +
                                               1048576),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi1[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi1[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi1[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&y_words_hi1[3])));
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 440 + (unsigned int)tmem_row_base1),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_lo1[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_lo1[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_lo1[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_lo1[3])));
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 440 + (unsigned int)tmem_row_base1 +
                                               1048576),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_hi1[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_hi1[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_hi1[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&do_words_hi1[3])));
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(tcgen_inputs_ready_addr + (tcgen_data_stage1) * 8);
            mbarrier_arrive(raw_done_addr + (raw_stage1) * 8);
          }
          mbarrier_wait(tcgen_products_ready_addr + (tcgen_data_stage1) * 8,
                        _phase_tcgen_products_ready);
          float _tmem_load_2[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
              : "r"(taddr + 336 + (unsigned int)tmem_row_base1));
          float _tmem_load_3[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
              : "r"(taddr + 336 + (unsigned int)tmem_row_base1 + 1048576));
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          uint32_t _tmem_load_2_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_2[_lp * 2 + 0], _tmem_load_2[_lp * 2 + 1 + 0]));
            _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _tmem_load_3_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_3[_lp * 2 + 0], _tmem_load_3[_lp * 2 + 1 + 0]));
            _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          int segment_9 = (value_dim_base1 + ov_col1) / 64;
          int segment_col_10 = value_dim_base1 + ov_col1 - segment_9 * 64;
          int swizzled_col_11 = segment_col_10 ^ (ov_token1 & 7) * 8;
          unsigned int u_addr_lo1 =
              u_smem_addr +
              (unsigned int)((segment_9 * 16 * 64 + ov_token1 * 64 + swizzled_col_11) * 2);
          int segment_12 = (value_dim_base1 + 16 + ov_col1) / 64;
          int segment_col_13 = value_dim_base1 + 16 + ov_col1 - segment_12 * 64;
          int swizzled_col_14 = segment_col_13 ^ (ov_token1 & 7) * 8;
          unsigned int u_addr_hi1 =
              u_smem_addr +
              (unsigned int)((segment_12 * 16 * 64 + ov_token1 * 64 + swizzled_col_14) * 2);
          uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)u_addr_lo1);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                  _stmatrix_addr_1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[3]))
              : "memory");
          uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)u_addr_hi1);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                  _stmatrix_addr_2),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[3]))
              : "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(u_smem_ready_addr);
          }
          float _tmem_load_4[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
              : "r"(taddr + 352 + (unsigned int)tmem_row_base1));
          float _tmem_load_5[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
              : "r"(taddr + 352 + (unsigned int)tmem_row_base1 + 1048576));
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          checksum_sum1 += _tmem_load_2[0] + _tmem_load_4[0];
          uint32_t _tmem_load_4_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_4[_lp * 2 + 0], _tmem_load_4[_lp * 2 + 1 + 0]));
            _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _tmem_load_5_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_5[_lp * 2 + 0], _tmem_load_5[_lp * 2 + 1 + 0]));
            _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 440 + (unsigned int)tmem_row_base1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[3])));
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 440 + (unsigned int)tmem_row_base1 +
                                               1048576),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[3])));
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(du_inp_ready_addr + (tcgen_data_stage1) * 8);
            mbarrier_arrive(tcgen_products_done_addr + (tcgen_data_stage1) * 8);
          }
          mbarrier_wait(dy_ready_addr + (tcgen_data_stage1) * 8, _phase_dy_ready);
          float _tmem_load_6[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7]))
              : "r"(taddr + 320 + (unsigned int)tmem_row_base1));
          float _tmem_load_7[8];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7]))
              : "r"(taddr + 320 + (unsigned int)tmem_row_base1 + 1048576));
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          float diff_words_lo1_f32[8];
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&diff_words_lo1_f32[_pair * 2])[0]),
                  "=f"((&diff_words_lo1_f32[_pair * 2])[1])
                : "r"(diff_words_lo1[_pair]));
          }
          float diff_words_hi1_f32[8];
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&diff_words_hi1_f32[_pair * 2])[0]),
                  "=f"((&diff_words_hi1_f32[_pair * 2])[1])
                : "r"(diff_words_hi1[_pair]));
          }
          float dbeta_partial1[4];
          dbeta_partial1[0] = 0.0f;
          dbeta_partial1[1] = 0.0f;
          dbeta_partial1[2] = 0.0f;
          dbeta_partial1[3] = 0.0f;
#pragma unroll
          for (int dbeta_pair1 = 0; dbeta_pair1 < 4; dbeta_pair1++) {
            const int dbeta_elem1 = dbeta_pair1 * 2;
            const int dbeta_slot_lo1 = 2 * (dbeta_pair1 / 2);
            const int dbeta_slot_hi1 = dbeta_slot_lo1 + 1;
            dbeta_partial1[dbeta_slot_lo1] =
                dbeta_partial1[dbeta_slot_lo1] +
                _tmem_load_6[dbeta_elem1] * diff_words_lo1_f32[dbeta_elem1] +
                _tmem_load_7[dbeta_elem1] * diff_words_hi1_f32[dbeta_elem1];
            dbeta_partial1[dbeta_slot_hi1] =
                dbeta_partial1[dbeta_slot_hi1] +
                _tmem_load_6[dbeta_elem1 + 1] * diff_words_lo1_f32[dbeta_elem1 + 1] +
                _tmem_load_7[dbeta_elem1 + 1] * diff_words_hi1_f32[dbeta_elem1 + 1];
          }
          uint32_t _tmem_load_6_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_6[_lp * 2 + 0], _tmem_load_6[_lp * 2 + 1 + 0]));
            _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _tmem_load_7_bf16[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_7[_lp * 2 + 0], _tmem_load_7[_lp * 2 + 1 + 0]));
            _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          int segment_15 = (value_dim_base1 + ov_col1) / 64;
          int segment_col_16 = value_dim_base1 + ov_col1 - segment_15 * 64;
          int swizzled_col_17 = segment_col_16 ^ (ov_token1 & 7) * 8;
          unsigned int dy_addr_lo1 =
              dy_smem_addr +
              (unsigned int)((segment_15 * 16 * 64 + ov_token1 * 64 + swizzled_col_17) * 2);
          int segment_18 = (value_dim_base1 + 16 + ov_col1) / 64;
          int segment_col_19 = value_dim_base1 + 16 + ov_col1 - segment_18 * 64;
          int swizzled_col_20 = segment_col_19 ^ (ov_token1 & 7) * 8;
          unsigned int dy_addr_hi1 =
              dy_smem_addr +
              (unsigned int)((segment_18 * 16 * 64 + ov_token1 * 64 + swizzled_col_20) * 2);
          uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)dy_addr_lo1);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                  _stmatrix_addr_3),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[3]))
              : "memory");
          uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)dy_addr_hi1);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                  _stmatrix_addr_4),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[3]))
              : "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(dy_smem_ready_addr);
          }
#pragma unroll
          for (int beta_dy_pair1 = 0; beta_dy_pair1 < 4; beta_dy_pair1++) {
            const int beta_dy_elem1 = beta_dy_pair1 * 2;
            float beta_dy_lo1 = beta_c0_1;
            float beta_dy_hi1 = beta_c1_1;
            if (beta_dy_pair1 >= 2) {
              beta_dy_lo1 = beta_c8_1;
              beta_dy_hi1 = beta_c9_1;
            }
            _tmem_load_6[beta_dy_elem1] = _tmem_load_6[beta_dy_elem1] * beta_dy_lo1;
            _tmem_load_6[beta_dy_elem1 + 1] = _tmem_load_6[beta_dy_elem1 + 1] * beta_dy_hi1;
            _tmem_load_7[beta_dy_elem1] = _tmem_load_7[beta_dy_elem1] * beta_dy_lo1;
            _tmem_load_7[beta_dy_elem1 + 1] = _tmem_load_7[beta_dy_elem1 + 1] * beta_dy_hi1;
          }
          uint32_t _tmem_load_6_bf16_21[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_6[_lp * 2 + 0], _tmem_load_6[_lp * 2 + 1 + 0]));
            _tmem_load_6_bf16_21[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _tmem_load_7_bf16_22[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_7[_lp * 2 + 0], _tmem_load_7[_lp * 2 + 1 + 0]));
            _tmem_load_7_bf16_22[_lp] = *(uint32_t*)&_bf2;
          }
          int segment_23 = (value_dim_base1 + ov_col1) / 64;
          int segment_col_24 = value_dim_base1 + ov_col1 - segment_23 * 64;
          int swizzled_col_25 = segment_col_24 ^ (ov_token1 & 7) * 8;
          unsigned int beta_dy_addr_lo1 =
              beta_dy_smem_addr + beta_dy_smem_stage1 * 4096 +
              (unsigned int)((segment_23 * 16 * 64 + ov_token1 * 64 + swizzled_col_25) * 2);
          int segment_26 = (value_dim_base1 + 16 + ov_col1) / 64;
          int segment_col_27 = value_dim_base1 + 16 + ov_col1 - segment_26 * 64;
          int swizzled_col_28 = segment_col_27 ^ (ov_token1 & 7) * 8;
          unsigned int beta_dy_addr_hi1 =
              beta_dy_smem_addr + beta_dy_smem_stage1 * 4096 +
              (unsigned int)((segment_26 * 16 * 64 + ov_token1 * 64 + swizzled_col_28) * 2);
          mbarrier_wait(beta_dy_smem_done_addr + (beta_dy_smem_stage1) * 8,
                        _phase_beta_dy_smem_done);
          uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)beta_dy_addr_lo1);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                  _stmatrix_addr_5),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_21[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_21[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_21[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_21[3]))
              : "memory");
          uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)beta_dy_addr_hi1);
          asm volatile(
              "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                  _stmatrix_addr_6),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_22[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_22[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_22[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_22[3]))
              : "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(beta_dy_smem_ready_addr + (beta_dy_smem_stage1) * 8);
          }
          beta_dy_smem_stage1 += 1;
          if (beta_dy_smem_stage1 == 2) {
            beta_dy_smem_stage1 = 0;
            _phase_beta_dy_smem_done ^= 1;
          }
          const float2 _scale2_7 = {-1.0f, -1.0f};
#pragma unroll
          for (int _ls = 0; _ls < 4; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_7);
          const float2 _scale2_8 = {-1.0f, -1.0f};
#pragma unroll
          for (int _ls = 0; _ls < 4; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_8);
          uint32_t _tmem_load_6_bf16_29[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_6[_lp * 2 + 0], _tmem_load_6[_lp * 2 + 1 + 0]));
            _tmem_load_6_bf16_29[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _tmem_load_7_bf16_30[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_7[_lp * 2 + 0], _tmem_load_7[_lp * 2 + 1 + 0]));
            _tmem_load_7_bf16_30[_lp] = *(uint32_t*)&_bf2;
          }
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 432 + (unsigned int)tmem_row_base1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_29[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_29[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_29[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16_29[3])));
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x2.b32"
              " [%0], {%1, %2, %3, %4};" ::"r"(taddr + 432 + (unsigned int)tmem_row_base1 +
                                               1048576),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_30[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_30[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_30[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16_30[3])));
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(neg_dy_ready_addr + (tcgen_data_stage1) * 8);
            mbarrier_arrive(dy_done_addr + (tcgen_data_stage1) * 8);
          }
#pragma unroll
          for (int dbeta_slot1 = 0; dbeta_slot1 < 4; dbeta_slot1++) {
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, dbeta_partial1[dbeta_slot1], 4);
            dbeta_partial1[dbeta_slot1] = dbeta_partial1[dbeta_slot1] + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, dbeta_partial1[dbeta_slot1], 8);
            dbeta_partial1[dbeta_slot1] = dbeta_partial1[dbeta_slot1] + _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, dbeta_partial1[dbeta_slot1], 16);
            dbeta_partial1[dbeta_slot1] = dbeta_partial1[dbeta_slot1] + _shfl_xor_8;
          }
          if (lane < 4) {
#pragma unroll
            for (int dbeta_slot1_1 = 0; dbeta_slot1_1 < 4; dbeta_slot1_1++) {
              int dbeta_token_slot1 =
                  (lane & 3) * 2 + (dbeta_slot1_1 & 1) + 8 * (dbeta_slot1_1 / 2);
              dbeta_red_smem[warp_id_in_role_1 * 16 + dbeta_token_slot1] =
                  dbeta_partial1[dbeta_slot1_1];
            }
          }
          asm volatile("barrier.sync 9, 128;" ::: "memory");
          mbarrier_wait(dbeta_m_ready_addr + (dbeta_m_stage1) * 8, _phase_dbeta_m_ready);
          if (role_tid1 < 16) {
            float dbeta_value1 = dbeta_m_smem[role_tid1];
#pragma unroll
            for (int dbeta_warp1 = 0; dbeta_warp1 < 4; dbeta_warp1++) {
              float dbeta_part1 = dbeta_red_smem[dbeta_warp1 * 16 + role_tid1];
              dbeta_value1 += dbeta_part1;
            }
            int dbeta_chunk1 = compute_end1 - 1 - reverse1;
            long long dbeta_token1 = bos1 + (long long)dbeta_chunk1 * 16 + (long long)role_tid1;
            if (dbeta_chunk1 < write_end1) {
              dbeta[dbeta_token1 * (long long)num_heads + (long long)head1] = dbeta_value1;
            }
          }
          asm volatile("barrier.sync 9, 128;" ::: "memory");
          if (warp_id_in_role_1 == 0) {
            if (elect_sync()) {
              mbarrier_arrive(dbeta_m_done_addr + (dbeta_m_stage1) * 8);
            }
          }
          dbeta_m_stage1 += 1;
          if (dbeta_m_stage1 == 1) {
            dbeta_m_stage1 = 0;
            _phase_dbeta_m_ready ^= 1;
          }
          mbarrier_wait(dstate_ready_addr + (tcgen_data_stage1) * 8, _phase_dstate_ready);
          mbarrier_wait(dstate_smem_done_addr + (dstate_smem_stage1) * 8, _phase_dstate_smem_done);
          dstate_smem_slot_acquired1 = 1;
          int boundary_chunk1 = compute_end1 - 1 - reverse1;
          long long boundary_checkpoint1 = bos1 / 16 + (long long)boundary_chunk1;
#pragma unroll
          for (int dstate_block1 = 0; dstate_block1 < 4; dstate_block1++) {
            float _tmem_load_8[32];
            tmem_ld_x32(&_tmem_load_8[0],
                        taddr + (unsigned int)(dstate_block1 * 32) + (unsigned int)tmem_row_base1);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t _tmem_load_8_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_8[_lp * 2 + 0], _tmem_load_8[_lp * 2 + 1 + 0]));
              _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            if (reverse1 + 1 < compute_end1 - write_start1) {
              asm volatile(
                  "tcgen05.st.sync.aligned.32x32b.x16.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(taddr + 128 + (unsigned int)(dstate_block1 * 16) +
                                (unsigned int)tmem_row_base1),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[15])));
            }
#pragma unroll
            for (int dstate_atom1 = 0; dstate_atom1 < 4; dstate_atom1++) {
              int dstate_col_base1 = dstate_block1 * 32 + dstate_atom1 * 8;
              int segment_1_1 = dstate_col_base1 / 64;
              int segment_col_2_1 = dstate_col_base1 - segment_1_1 * 64;
              int swizzled_col_3_1 = segment_col_2_1 ^ (role_tid1 & 7) * 8;
              int dstate_smem_index1 = segment_1_1 * 128 * 64 + role_tid1 * 64 + swizzled_col_3_1;
              asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                               dstate_smem_all_addr + (unsigned int)(dstate_smem_index1 * 2)),
                           "r"(_tmem_load_8_bf16[dstate_atom1 * 4]),
                           "r"(_tmem_load_8_bf16[dstate_atom1 * 4 + 1]),
                           "r"(_tmem_load_8_bf16[dstate_atom1 * 4 + 2]),
                           "r"(_tmem_load_8_bf16[dstate_atom1 * 4 + 3])
                           : "memory");
            }
            if (write_start1 == 0 && reverse1 + 1 == compute_end1 - write_start1) {
              long long dinitial_base1 =
                  (((long long)sequence1 * (long long)num_heads + (long long)head1) * 128 +
                   (long long)role_tid1) *
                      128 +
                  (long long)(dstate_block1 * 32);
#pragma unroll
              for (int dinitial_vec1 = 0; dinitial_vec1 < 4; dinitial_vec1++) {
                {
                  unsigned _stv8_9_0 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 0]);
                  unsigned _stv8_9_1 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 1]);
                  unsigned _stv8_9_2 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 2]);
                  unsigned _stv8_9_3 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 3]);
                  unsigned _stv8_9_4 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 4]);
                  unsigned _stv8_9_5 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 5]);
                  unsigned _stv8_9_6 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 6]);
                  unsigned _stv8_9_7 = __float_as_uint(_tmem_load_8[dinitial_vec1 * 8 + 7]);
                  asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(
                                   (void*)(dinitial_state +
                                           (dinitial_base1 + (long long)(dinitial_vec1 * 8)))),
                               "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3),
                               "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7)
                               : "memory");
                }
              }
            }
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          asm volatile("barrier.sync 9, 128;" ::: "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(boundary_smem_ready_addr + (boundary_stage1) * 8);
          }
          float boundary_diag_pair1[2];
#pragma unroll
          for (int boundary_diag_block1 = 0; boundary_diag_block1 < 2; boundary_diag_block1++) {
            int boundary_channel_base1 = warp_id_in_role_1 * 32 + boundary_diag_block1 * 16;
            float boundary_diag_acc1[8];
#pragma unroll
            for (int boundary_k1 = 0; boundary_k1 < 128; boundary_k1 += 16) {
              unsigned int boundary_a_frag1[4];
              unsigned int boundary_b_frag1[4];
              int boundary_lane_matrix1 = lane / 8;
              int boundary_lane_row1 = lane & 7;
              int boundary_a_row1 =
                  boundary_k1 + boundary_lane_row1 + boundary_lane_matrix1 / 2 * 8;
              int boundary_a_col1 = boundary_channel_base1 + (boundary_lane_matrix1 & 1) * 8;
              int segment_1_2 = boundary_a_col1 / 64;
              int segment_col_2_2 = boundary_a_col1 - segment_1_2 * 64;
              int swizzled_col_3_2 = segment_col_2_2 ^ (boundary_a_row1 & 7) * 8;
              asm volatile(
                  "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(boundary_a_frag1[0]), "=r"(boundary_a_frag1[1]), "=r"(boundary_a_frag1[2]),
                    "=r"(boundary_a_frag1[3])
                  : "r"(state_operand_all_addr +
                        (unsigned int)(((int)state_smem_stage1 * 128 * 128 +
                                        (segment_1_2 * 128 * 64 + boundary_a_row1 * 64 +
                                         swizzled_col_3_2)) *
                                       2))
                  : "memory");
#pragma unroll
              for (int boundary_n_half1 = 0; boundary_n_half1 < 2; boundary_n_half1++) {
                int boundary_b_row1 = boundary_k1 + lane % 16;
                int boundary_b_col1 = boundary_channel_base1 + boundary_n_half1 * 8;
                int segment_2_1 = boundary_b_col1 / 64;
                int segment_col_3_1 = boundary_b_col1 - segment_2_1 * 64;
                int swizzled_col_4_1 = segment_col_3_1 ^ (boundary_b_row1 & 7) * 8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(boundary_b_frag1[boundary_n_half1 * 2]),
                               "=r"(boundary_b_frag1[boundary_n_half1 * 2 + 1])
                             : "r"(dstate_smem_all_addr +
                                   (unsigned int)((segment_2_1 * 128 * 64 + boundary_b_row1 * 64 +
                                                   swizzled_col_4_1) *
                                                  2))
                             : "memory");
              }
              asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                  "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                  : "=f"(boundary_diag_acc1[0]), "=f"(boundary_diag_acc1[1]),
                    "=f"(boundary_diag_acc1[2]), "=f"(boundary_diag_acc1[3])
                  : "r"(boundary_a_frag1[0]), "r"(boundary_a_frag1[1]), "r"(boundary_a_frag1[2]),
                    "r"(boundary_a_frag1[3]), "r"(boundary_b_frag1[0]), "r"(boundary_b_frag1[1]),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[0])),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[1])),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[2])),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[3])));
              asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                  "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                  : "=f"(boundary_diag_acc1[4]), "=f"(boundary_diag_acc1[(4) + 1]),
                    "=f"(boundary_diag_acc1[(4) + 2]), "=f"(boundary_diag_acc1[(4) + 3])
                  : "r"(boundary_a_frag1[0]), "r"(boundary_a_frag1[1]), "r"(boundary_a_frag1[2]),
                    "r"(boundary_a_frag1[3]), "r"(boundary_b_frag1[2]),
                    "r"(boundary_b_frag1[(2) + 1]),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[4])),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[(4) + 1])),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[(4) + 2])),
                    "f"(((boundary_k1 == 0) ? 0.0f : boundary_diag_acc1[(4) + 3])));
            }
            float boundary_diag_lo1 = boundary_diag_acc1[0];
            float boundary_diag_hi1 = boundary_diag_acc1[6];
            if ((lane / 4 & 1) != 0) {
              boundary_diag_lo1 = boundary_diag_acc1[1];
              boundary_diag_hi1 = boundary_diag_acc1[7];
            }
            int boundary_target_row1 = lane & 15;
            int boundary_source_lane1 =
                (boundary_target_row1 & 7) * 4 + (boundary_target_row1 & 7) / 2;
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, boundary_diag_lo1, boundary_source_lane1);
            float boundary_diag_lower1 = _shfl_0;
            float _shfl_1 = __shfl_sync(0xFFFFFFFF, boundary_diag_hi1, boundary_source_lane1);
            float boundary_diag_upper1 = _shfl_1;
            boundary_diag_pair1[boundary_diag_block1] = boundary_diag_lower1;
            if (boundary_target_row1 >= 8) {
              boundary_diag_pair1[boundary_diag_block1] = boundary_diag_upper1;
            }
          }
          float dgate_last_state1 = boundary_diag_pair1[0];
          if (lane >= 16) {
            dgate_last_state1 = boundary_diag_pair1[1];
          }
          boundary_state_smem[role_tid1] = dgate_last_state1;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(boundary_state_ready_addr + (boundary_stage1) * 8);
          }
          if (reverse1 + 1 < compute_end1 - write_start1) {
            if (elect_sync()) {
              mbarrier_arrive(dstate_inp_ready_addr);
              mbarrier_arrive(dstate_smem_ready_addr + (dstate_smem_stage1) * 8);
            }
            dstate_smem_stage1 += 1;
            if (dstate_smem_stage1 == 1) {
              dstate_smem_stage1 = 0;
              _phase_dstate_smem_done ^= 1;
            }
            dstate_smem_slot_acquired1 = 0;
          }
          if (boundary_chunk1 < write_end1) {
            long long boundary_output_index1 =
                (boundary_checkpoint1 * (long long)num_heads + (long long)head1) * 128 +
                (long long)role_tid1;
            dgate_boundary_out[boundary_output_index1] = dgate_last_state1;
          }
          mbarrier_wait(boundary_acc_ready_addr + (boundary_stage1) * 8, _phase_boundary_acc_ready);
          if (elect_sync()) {
            mbarrier_arrive(dstate_done_addr + (tcgen_data_stage1) * 8);
          }
          boundary_stage1 += 1;
          if (boundary_stage1 == 1) {
            boundary_stage1 = 0;
            _phase_boundary_acc_ready ^= 1;
          }
          raw_stage1 += 1;
          if (raw_stage1 == 2) {
            raw_stage1 = 0;
            _phase_raw_ready_1 ^= 1;
          }
          state_smem_stage1 += 1;
          if (state_smem_stage1 == 2) {
            state_smem_stage1 = 0;
            _phase_state_k_ready ^= 1;
          }
          tcgen_data_stage1 += 1;
          if (tcgen_data_stage1 == 1) {
            tcgen_data_stage1 = 0;
            _phase_tcgen_inputs_done ^= 1;
            _phase_tcgen_products_ready ^= 1;
            _phase_dy_ready ^= 1;
            _phase_dstate_ready ^= 1;
          }
        }
        if (validate_outputs != 0 && role_tid1 == 0) {
          observed[tile1] = checksum_sum1;
        }
      }
      if (elect_sync()) {
        mbarrier_arrive(consumers_done_addr);
      }
    }
  }
  // ---- Role: compute2 ----
  if (warp >= 8 && warp <= 11) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
    {  // compute2_main
      unsigned int sched_stage2 = 0;
      unsigned int dk_restore_stage2 = 0;
      unsigned int local_grad_stage2 = 0;
      unsigned int qk_raw_stage2 = 0;
      unsigned int g_prefix_stage2 = 0;
      unsigned int state_smem_stage2 = 0;
      unsigned int boundary_stage2 = 0;
      int warp_id_in_role_2 = (warp - 8);
      int role_tid2 = warp_id_in_role_2 * 32 + lane;
      const int tmem_row_base2 = warp_id_in_role_2 * 32 << 16;
      unsigned int _phase_sched_ready_2 = 0;
      unsigned int _phase_qk_raw_ready = 0;
      unsigned int _phase_g_prefix_ready = 0;
      unsigned int _phase_state_ready = 0;
      unsigned int _phase_local_grad_ready = 0;
      unsigned int _phase_boundary_acc_ready_1 = 0;
      unsigned int _phase_boundary_state_ready = 0;
      unsigned int _phase_dk_restore_ready = 0;
#pragma unroll 1
      for (int __2 = 0; __2 < total_work_items; __2++) {
        mbarrier_wait(sched_ready_addr + (sched_stage2) * 8, _phase_sched_ready_2);
        unsigned int ticket_words_2[1];
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&ticket_words_2[0]))
                     : "r"(work_item_addr + sched_stage2 * 4));
        unsigned int tile2 = ticket_words_2[0];
        if (elect_sync()) {
          mbarrier_arrive(sched_done_addr + (sched_stage2) * 8);
        }
        sched_stage2 += 1;
        if (sched_stage2 == 8) {
          sched_stage2 = 0;
          _phase_sched_ready_2 ^= 1;
        }
        if (tile2 >= (unsigned int)total_work_items) {
          break;
        }
        int item_base2 = (int)tile2 * 8;
        int sequence2 = work_items[item_base2];
        int head2 = work_items[item_base2 + 1];
        int write_start2 = work_items[item_base2 + 2];
        int write_end2 = work_items[item_base2 + 3];
        int compute_end2 = work_items[item_base2 + 5];
        long long bos2 = cu_seqlens[sequence2];
#pragma unroll 1
        for (int reverse2 = 0; reverse2 < compute_end2 - write_start2; reverse2++) {
          mbarrier_wait(qk_raw_ready_addr + (qk_raw_stage2) * 8, _phase_qk_raw_ready);
          mbarrier_wait(g_prefix_ready_addr + (g_prefix_stage2) * 8, _phase_g_prefix_ready);
          mbarrier_wait(state_ready_addr + (state_smem_stage2) * 8, _phase_state_ready);
          int norm_stage_base2 = (int)(qk_raw_stage2 % 4) * 2 * 16;
          float eg_values2[16];
#pragma unroll
          for (int token_gate2 = 0; token_gate2 < 16; token_gate2++) {
            int segment_7 = role_tid2 / 32;
            int segment_col_8 = role_tid2 - segment_7 * 32;
            int swizzled_col_7 = segment_col_8 ^ (token_gate2 & 7) * 4;
            eg_values2[token_gate2] =
                g_prefix_all[(int)g_prefix_stage2 * 16 * 128 + segment_7 * 16 * 32 +
                             token_gate2 * 32 + swizzled_col_7];
          }
          mbarrier_arrive(g_prefix_done_addr + (g_prefix_stage2) * 8);
          mbarrier_wait(local_grad_ready_addr + (local_grad_stage2) * 8, _phase_local_grad_ready);
          asm volatile("tcgen05.fence::after_thread_sync;");
          float _tmem_load_9[16];
          tmem_ld_x16(&_tmem_load_9[0], taddr + 368 + (unsigned int)tmem_row_base2);
          float _tmem_load_10[16];
          tmem_ld_x16(&_tmem_load_10[0], taddr + 384 + (unsigned int)tmem_row_base2);
          float _tmem_load_11[16];
          tmem_ld_x16(&_tmem_load_11[0], taddr + 400 + (unsigned int)tmem_row_base2);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          asm volatile("tcgen05.fence::before_thread_sync;");
          if (elect_sync()) {
            mbarrier_arrive(local_grad_done_addr + (local_grad_stage2) * 8);
            mbarrier_arrive(boundary_local_grad_free_addr + (local_grad_stage2) * 8);
            mbarrier_arrive(dstate_done_addr + (local_grad_stage2) * 8);
          }
          mbarrier_wait(boundary_acc_ready_addr + (boundary_stage2) * 8,
                        _phase_boundary_acc_ready_1);
          mbarrier_wait(boundary_state_ready_addr + (boundary_stage2) * 8,
                        _phase_boundary_state_ready);
          float dgate_last_state2 = boundary_state_smem[role_tid2];
          boundary_stage2 += 1;
          if (boundary_stage2 == 1) {
            boundary_stage2 = 0;
            _phase_boundary_acc_ready_1 ^= 1;
            _phase_boundary_state_ready ^= 1;
          }
          mbarrier_arrive(state_cg2_done_addr + (state_smem_stage2) * 8);
          float2 dgate_last_restore_acc2[2];
#pragma unroll
          for (int restore_acc_init2 = 0; restore_acc_init2 < 2; restore_acc_init2++) {
            float2 _f2_0 = make_float2(0.0f, 0.0f);
            dgate_last_restore_acc2[restore_acc_init2] = _f2_0;
          }
#pragma unroll
          for (int token_pair2 = 0; token_pair2 < 8; token_pair2++) {
            const int token_scale0_2 = token_pair2 * 2;
            const int token_scale1_2 = token_scale0_2 + 1;
            float eg0_2 = eg_values2[token_scale0_2];
            float eg1_2 = eg_values2[token_scale1_2];
            float2 _f2_1 = make_float2(_tmem_load_9[token_scale0_2], _tmem_load_9[token_scale1_2]);
            float2 _f2_2 = make_float2(eg0_2 * scale, eg1_2 * scale);
            float2 dq_pair2 = mul_f32x2(_f2_1, _f2_2);
            float2 _f2_3 = make_float2(-eg0_2, -eg1_2);
            float2 _f2_4 =
                make_float2(_tmem_load_10[token_scale0_2], _tmem_load_10[token_scale1_2]);
            float2 dk_decay_pair2 = mul_f32x2(_f2_3, _f2_4);
            float2 _f2_5 =
                make_float2(_tmem_load_11[token_scale0_2], _tmem_load_11[token_scale1_2]);
            float _rcp_2 = approx_rcp(eg0_2);
            float _rcp_3 = approx_rcp(eg1_2);
            float2 _f2_6 = make_float2(_rcp_2, _rcp_3);
            float2 dk_inv_pair2 = fma_f32x2(_f2_5, _f2_6, dk_decay_pair2);
            _tmem_load_9[token_scale0_2] = dq_pair2.x;
            _tmem_load_9[token_scale1_2] = dq_pair2.y;
            _tmem_load_10[token_scale0_2] = dk_decay_pair2.x;
            _tmem_load_10[token_scale1_2] = dk_decay_pair2.y;
            _tmem_load_11[token_scale0_2] = dk_inv_pair2.x;
            _tmem_load_11[token_scale1_2] = dk_inv_pair2.y;
          }
          unsigned int raw_words_bits2[8];
          float q_raw_values2[16];
          float k_raw_values2[16];
          float _tmem_load_12[8];
          tmem_ld_x8(&_tmem_load_12[0],
                     taddr + 480 + qk_raw_stage2 % 4 * 8 + (unsigned int)tmem_row_base2);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          raw_words_bits2[0] = __as_u32(_tmem_load_12[0]);
          raw_words_bits2[1] = __as_u32(_tmem_load_12[1]);
          raw_words_bits2[2] = __as_u32(_tmem_load_12[2]);
          raw_words_bits2[3] = __as_u32(_tmem_load_12[3]);
          raw_words_bits2[4] = __as_u32(_tmem_load_12[4]);
          raw_words_bits2[5] = __as_u32(_tmem_load_12[5]);
          raw_words_bits2[6] = __as_u32(_tmem_load_12[6]);
          raw_words_bits2[7] = __as_u32(_tmem_load_12[7]);
#pragma unroll
          for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_raw_values2[_pair * 2])[0]), "=f"((&k_raw_values2[_pair * 2])[1])
                : "r"(raw_words_bits2[_pair]));
          }
          if (USE_DSTATE_IN != 0 || reverse2 > 0) {
            mbarrier_wait(dk_restore_ready_addr + (dk_restore_stage2) * 8, _phase_dk_restore_ready);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_13[16];
            tmem_ld_x16(&_tmem_load_13[0], taddr + 416 + (unsigned int)tmem_row_base2);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
            for (int dk_pair2 = 0; dk_pair2 < 8; dk_pair2++) {
              const int dk_col0_2 = dk_pair2 * 2;
              const int dk_col1_2 = dk_col0_2 + 1;
              float _rcp_4 = approx_rcp(eg_values2[dk_col0_2]);
              float restore_scale0_2 = eg_values2[15] * _rcp_4;
              float _rcp_5 = approx_rcp(eg_values2[dk_col1_2]);
              float restore_scale1_2 = eg_values2[15] * _rcp_5;
              float2 _f2_7 = make_float2(restore_scale0_2, restore_scale1_2);
              float2 _f2_8 = make_float2(_tmem_load_13[dk_col0_2], _tmem_load_13[dk_col1_2]);
              float2 restore_hat_pair2 = mul_f32x2(_f2_7, _f2_8);
              float2 _f2_9 = make_float2(_tmem_load_11[dk_col0_2], _tmem_load_11[dk_col1_2]);
              float2 dk_inv_restore_pair2 = add_f32x2(_f2_9, restore_hat_pair2);
              _tmem_load_11[dk_col0_2] = dk_inv_restore_pair2.x;
              _tmem_load_11[dk_col1_2] = dk_inv_restore_pair2.y;
              float k_restore_norm0_2 = qk_norm_smem_all[norm_stage_base2 + 16 + dk_col0_2];
              float k_restore_norm1_2 = qk_norm_smem_all[norm_stage_base2 + 16 + dk_col1_2];
              float2 _f2_10 = make_float2(k_raw_values2[dk_col0_2], k_raw_values2[dk_col1_2]);
              float2 _f2_11 = make_float2(k_restore_norm0_2, k_restore_norm1_2);
              float2 k_normalized_pair2 = mul_f32x2(_f2_10, _f2_11);
              float2 restore_gate_pair2 = mul_f32x2(k_normalized_pair2, restore_hat_pair2);
              const int restore_acc_index2 = dk_pair2 % 2;
              dgate_last_restore_acc2[restore_acc_index2] =
                  add_f32x2(dgate_last_restore_acc2[restore_acc_index2], restore_gate_pair2);
            }
            if (elect_sync()) {
              mbarrier_arrive(dk_restore_done_addr + (dk_restore_stage2) * 8);
            }
            dk_restore_stage2 += 1;
            if (dk_restore_stage2 == 1) {
              dk_restore_stage2 = 0;
              _phase_dk_restore_ready ^= 1;
            }
          }
          float _tmem_load_14[8];
          tmem_ld_x8(&_tmem_load_14[0],
                     taddr + 448 + qk_raw_stage2 % 4 * 8 + (unsigned int)tmem_row_base2);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          raw_words_bits2[0] = __as_u32(_tmem_load_14[0]);
          raw_words_bits2[1] = __as_u32(_tmem_load_14[1]);
          raw_words_bits2[2] = __as_u32(_tmem_load_14[2]);
          raw_words_bits2[3] = __as_u32(_tmem_load_14[3]);
          raw_words_bits2[4] = __as_u32(_tmem_load_14[4]);
          raw_words_bits2[5] = __as_u32(_tmem_load_14[5]);
          raw_words_bits2[6] = __as_u32(_tmem_load_14[6]);
          raw_words_bits2[7] = __as_u32(_tmem_load_14[7]);
#pragma unroll
          for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_raw_values2[_pair * 2])[0]), "=f"((&q_raw_values2[_pair * 2])[1])
                : "r"(raw_words_bits2[_pair]));
          }
          float dgate_last_restore2 = dgate_last_restore_acc2[0].x + dgate_last_restore_acc2[0].y +
                                      (dgate_last_restore_acc2[1].x + dgate_last_restore_acc2[1].y);
#pragma unroll
          for (int gate_pair2 = 0; gate_pair2 < 8; gate_pair2++) {
            const int gate_token0_2 = gate_pair2 * 2;
            const int gate_token1_2 = gate_token0_2 + 1;
            float q_gate_norm0_2 = qk_norm_smem_all[norm_stage_base2 + gate_token0_2];
            float q_gate_norm1_2 = qk_norm_smem_all[norm_stage_base2 + gate_token1_2];
            float k_gate_norm0_2 = qk_norm_smem_all[norm_stage_base2 + 16 + gate_token0_2];
            float k_gate_norm1_2 = qk_norm_smem_all[norm_stage_base2 + 16 + gate_token1_2];
            float2 _f2_12 = make_float2(q_raw_values2[gate_token0_2], q_raw_values2[gate_token1_2]);
            float2 _f2_13 = make_float2(q_gate_norm0_2, q_gate_norm1_2);
            float2 q_normalized_pair2 = mul_f32x2(_f2_12, _f2_13);
            float2 _f2_14 = make_float2(k_raw_values2[gate_token0_2], k_raw_values2[gate_token1_2]);
            float2 _f2_15 = make_float2(k_gate_norm0_2, k_gate_norm1_2);
            float2 k_normalized_pair2_1 = mul_f32x2(_f2_14, _f2_15);
            float2 _f2_16 = make_float2(2.0f, 2.0f);
            float2 _f2_17 = make_float2(_tmem_load_10[gate_token0_2], _tmem_load_10[gate_token1_2]);
            float2 _f2_18 = make_float2(_tmem_load_11[gate_token0_2], _tmem_load_11[gate_token1_2]);
            float2 gate_residual_pair2 = fma_sub_f32x2(_f2_16, _f2_17, _f2_18);
            float2 _f2_19 = make_float2(_tmem_load_9[gate_token0_2], _tmem_load_9[gate_token1_2]);
            float2 dgate_pair2 = fma_f32x2(k_normalized_pair2_1, gate_residual_pair2,
                                           mul_f32x2(q_normalized_pair2, _f2_19));
            _tmem_load_10[gate_token0_2] = dgate_pair2.x;
            _tmem_load_10[gate_token1_2] = dgate_pair2.y;
          }
          _tmem_load_10[15] = _tmem_load_10[15] + dgate_last_restore2;
          if (USE_DSTATE_IN != 0 || reverse2 > 0) {
            _tmem_load_10[15] = _tmem_load_10[15] + eg_values2[15] * dgate_last_state2;
          }
#pragma unroll
          for (int gate_suffix2 = 15; gate_suffix2 >= 1; gate_suffix2--) {
            _tmem_load_10[gate_suffix2 - 1] =
                _tmem_load_10[gate_suffix2 - 1] + _tmem_load_10[gate_suffix2];
          }
          float anchored_dgate2 = dgate_last_state2;
#pragma unroll
          for (int gate_reconcile2 = 0; gate_reconcile2 < 8; gate_reconcile2++) {
            float suffix_here2 = _tmem_load_10[gate_reconcile2];
            float suffix_next2 = _tmem_load_10[gate_reconcile2 + 1];
            _tmem_load_10[gate_reconcile2] = anchored_dgate2;
            anchored_dgate2 -= suffix_here2 - suffix_next2;
          }
          float _tmem_load_15[8];
          tmem_ld_x8(&_tmem_load_15[0],
                     taddr + 448 + qk_raw_stage2 % 4 * 8 + (unsigned int)tmem_row_base2);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          raw_words_bits2[0] = __as_u32(_tmem_load_15[0]);
          raw_words_bits2[1] = __as_u32(_tmem_load_15[1]);
          raw_words_bits2[2] = __as_u32(_tmem_load_15[2]);
          raw_words_bits2[3] = __as_u32(_tmem_load_15[3]);
          raw_words_bits2[4] = __as_u32(_tmem_load_15[4]);
          raw_words_bits2[5] = __as_u32(_tmem_load_15[5]);
          raw_words_bits2[6] = __as_u32(_tmem_load_15[6]);
          raw_words_bits2[7] = __as_u32(_tmem_load_15[7]);
#pragma unroll
          for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_raw_values2[_pair * 2])[0]), "=f"((&q_raw_values2[_pair * 2])[1])
                : "r"(raw_words_bits2[_pair]));
          }
          float dot_values2[16];
#pragma unroll
          for (int qdot_pair2 = 0; qdot_pair2 < 8; qdot_pair2++) {
            const int qdot_token0_2 = qdot_pair2 * 2;
            const int qdot_token1_2 = qdot_token0_2 + 1;
            float qdot_norm0_2 = qk_norm_smem_all[norm_stage_base2 + qdot_token0_2];
            float qdot_norm1_2 = qk_norm_smem_all[norm_stage_base2 + qdot_token1_2];
            float2 _f2_20 = make_float2(_tmem_load_9[qdot_token0_2], _tmem_load_9[qdot_token1_2]);
            float2 _f2_21 = make_float2(q_raw_values2[qdot_token0_2], q_raw_values2[qdot_token1_2]);
            float2 _f2_22 = make_float2(qdot_norm0_2, qdot_norm1_2);
            float2 qdot_pair_value2 = mul_f32x2(_f2_20, mul_f32x2(_f2_21, _f2_22));
            dot_values2[qdot_token0_2] = qdot_pair_value2.x;
            dot_values2[qdot_token1_2] = qdot_pair_value2.y;
#pragma unroll
            for (int qdot_step2 = 0; qdot_step2 < 5; qdot_step2++) {
              const int qdot_delta2 = 16 >> qdot_step2;
              float _shfl_xor_9 =
                  __shfl_xor_sync(0xFFFFFFFF, dot_values2[qdot_token0_2], qdot_delta2);
              float qdot_shuffle0_2 = _shfl_xor_9;
              float _shfl_xor_10 =
                  __shfl_xor_sync(0xFFFFFFFF, dot_values2[qdot_token1_2], qdot_delta2);
              float qdot_shuffle1_2 = _shfl_xor_10;
              float2 _f2_23 = make_float2(dot_values2[qdot_token0_2], dot_values2[qdot_token1_2]);
              float2 _f2_24 = make_float2(qdot_shuffle0_2, qdot_shuffle1_2);
              qdot_pair_value2 = add_f32x2(_f2_23, _f2_24);
              dot_values2[qdot_token0_2] = qdot_pair_value2.x;
              dot_values2[qdot_token1_2] = qdot_pair_value2.y;
            }
          }
          if (lane == 0) {
#pragma unroll
            for (int qdot_store2 = 0; qdot_store2 < 16; qdot_store2++) {
              qk_red_smem[warp_id_in_role_2 * 16 + qdot_store2] = dot_values2[qdot_store2];
            }
          }
          asm volatile("barrier.sync 10, 128;" ::: "memory");
#pragma unroll
          for (int qproj_pair2 = 0; qproj_pair2 < 8; qproj_pair2++) {
            const int qproj_token0_2 = qproj_pair2 * 2;
            const int qproj_token1_2 = qproj_token0_2 + 1;
            float2 _f2_25 = make_float2(0.0f, 0.0f);
            float2 qdot_total_pair2 = _f2_25;
#pragma unroll
            for (int qdot_warp2 = 0; qdot_warp2 < 4; qdot_warp2++) {
              float qdot_total0_2 = qk_red_smem[qdot_warp2 * 16 + qproj_token0_2];
              float qdot_total1_2 = qk_red_smem[qdot_warp2 * 16 + qproj_token1_2];
              float2 _f2_26 = make_float2(qdot_total0_2, qdot_total1_2);
              qdot_total_pair2 = add_f32x2(qdot_total_pair2, _f2_26);
            }
            float qproj_norm0_2 = qk_norm_smem_all[norm_stage_base2 + qproj_token0_2];
            float qproj_norm1_2 = qk_norm_smem_all[norm_stage_base2 + qproj_token1_2];
            float2 _f2_27 = make_float2(qproj_norm0_2, qproj_norm1_2);
            float2 qproj_norm_pair2 = _f2_27;
            float2 _f2_28 =
                make_float2(q_raw_values2[qproj_token0_2], q_raw_values2[qproj_token1_2]);
            float2 qproj_correction_pair2 =
                mul_f32x2(mul_f32x2(_f2_28, qproj_norm_pair2), qdot_total_pair2);
            float2 _f2_29 = make_float2(_tmem_load_9[qproj_token0_2], _tmem_load_9[qproj_token1_2]);
            float2 qproj_result_pair2 =
                mul_f32x2(sub_f32x2(_f2_29, qproj_correction_pair2), qproj_norm_pair2);
            _tmem_load_9[qproj_token0_2] = qproj_result_pair2.x;
            _tmem_load_9[qproj_token1_2] = qproj_result_pair2.y;
          }
          asm volatile("barrier.sync 10, 128;" ::: "memory");
          float _tmem_load_16[8];
          tmem_ld_x8(&_tmem_load_16[0],
                     taddr + 480 + qk_raw_stage2 % 4 * 8 + (unsigned int)tmem_row_base2);
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          raw_words_bits2[0] = __as_u32(_tmem_load_16[0]);
          raw_words_bits2[1] = __as_u32(_tmem_load_16[1]);
          raw_words_bits2[2] = __as_u32(_tmem_load_16[2]);
          raw_words_bits2[3] = __as_u32(_tmem_load_16[3]);
          raw_words_bits2[4] = __as_u32(_tmem_load_16[4]);
          raw_words_bits2[5] = __as_u32(_tmem_load_16[5]);
          raw_words_bits2[6] = __as_u32(_tmem_load_16[6]);
          raw_words_bits2[7] = __as_u32(_tmem_load_16[7]);
#pragma unroll
          for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_raw_values2[_pair * 2])[0]), "=f"((&k_raw_values2[_pair * 2])[1])
                : "r"(raw_words_bits2[_pair]));
          }
#pragma unroll
          for (int kdot_pair2 = 0; kdot_pair2 < 8; kdot_pair2++) {
            const int kdot_token0_2 = kdot_pair2 * 2;
            const int kdot_token1_2 = kdot_token0_2 + 1;
            float kdot_norm0_2 = qk_norm_smem_all[norm_stage_base2 + 16 + kdot_token0_2];
            float kdot_norm1_2 = qk_norm_smem_all[norm_stage_base2 + 16 + kdot_token1_2];
            float2 _f2_30 = make_float2(_tmem_load_11[kdot_token0_2], _tmem_load_11[kdot_token1_2]);
            float2 _f2_31 = make_float2(k_raw_values2[kdot_token0_2], k_raw_values2[kdot_token1_2]);
            float2 _f2_32 = make_float2(kdot_norm0_2, kdot_norm1_2);
            float2 kdot_pair_value2 = mul_f32x2(_f2_30, mul_f32x2(_f2_31, _f2_32));
            dot_values2[kdot_token0_2] = kdot_pair_value2.x;
            dot_values2[kdot_token1_2] = kdot_pair_value2.y;
#pragma unroll
            for (int kdot_step2 = 0; kdot_step2 < 5; kdot_step2++) {
              const int kdot_delta2 = 16 >> kdot_step2;
              float _shfl_xor_11 =
                  __shfl_xor_sync(0xFFFFFFFF, dot_values2[kdot_token0_2], kdot_delta2);
              float kdot_shuffle0_2 = _shfl_xor_11;
              float _shfl_xor_12 =
                  __shfl_xor_sync(0xFFFFFFFF, dot_values2[kdot_token1_2], kdot_delta2);
              float kdot_shuffle1_2 = _shfl_xor_12;
              float2 _f2_33 = make_float2(dot_values2[kdot_token0_2], dot_values2[kdot_token1_2]);
              float2 _f2_34 = make_float2(kdot_shuffle0_2, kdot_shuffle1_2);
              kdot_pair_value2 = add_f32x2(_f2_33, _f2_34);
              dot_values2[kdot_token0_2] = kdot_pair_value2.x;
              dot_values2[kdot_token1_2] = kdot_pair_value2.y;
            }
          }
          if (lane == 0) {
#pragma unroll
            for (int kdot_store2 = 0; kdot_store2 < 16; kdot_store2++) {
              qk_red_smem[warp_id_in_role_2 * 16 + kdot_store2] = dot_values2[kdot_store2];
            }
          }
          asm volatile("barrier.sync 10, 128;" ::: "memory");
#pragma unroll
          for (int kproj_pair2 = 0; kproj_pair2 < 8; kproj_pair2++) {
            const int kproj_token0_2 = kproj_pair2 * 2;
            const int kproj_token1_2 = kproj_token0_2 + 1;
            float2 _f2_35 = make_float2(0.0f, 0.0f);
            float2 kdot_total_pair2 = _f2_35;
#pragma unroll
            for (int kdot_warp2 = 0; kdot_warp2 < 4; kdot_warp2++) {
              float kdot_total0_2 = qk_red_smem[kdot_warp2 * 16 + kproj_token0_2];
              float kdot_total1_2 = qk_red_smem[kdot_warp2 * 16 + kproj_token1_2];
              float2 _f2_36 = make_float2(kdot_total0_2, kdot_total1_2);
              kdot_total_pair2 = add_f32x2(kdot_total_pair2, _f2_36);
            }
            float kproj_norm0_2 = qk_norm_smem_all[norm_stage_base2 + 16 + kproj_token0_2];
            float kproj_norm1_2 = qk_norm_smem_all[norm_stage_base2 + 16 + kproj_token1_2];
            float2 _f2_37 = make_float2(kproj_norm0_2, kproj_norm1_2);
            float2 kproj_norm_pair2 = _f2_37;
            float2 _f2_38 =
                make_float2(k_raw_values2[kproj_token0_2], k_raw_values2[kproj_token1_2]);
            float2 kproj_correction_pair2 =
                mul_f32x2(mul_f32x2(_f2_38, kproj_norm_pair2), kdot_total_pair2);
            float2 _f2_39 =
                make_float2(_tmem_load_11[kproj_token0_2], _tmem_load_11[kproj_token1_2]);
            float2 kproj_result_pair2 =
                mul_f32x2(sub_f32x2(_f2_39, kproj_correction_pair2), kproj_norm_pair2);
            _tmem_load_11[kproj_token0_2] = kproj_result_pair2.x;
            _tmem_load_11[kproj_token1_2] = kproj_result_pair2.y;
          }
          asm volatile("barrier.sync 10, 128;" ::: "memory");
          int output_chunk2 = compute_end2 - 1 - reverse2;
          long long output_token_base2 = bos2 + (long long)output_chunk2 * 16;
          if (output_chunk2 < write_end2) {
#pragma unroll
            for (int output_token2 = 0; output_token2 < 16; output_token2++) {
              long long output_index2 =
                  ((output_token_base2 + (long long)output_token2) * (long long)num_heads +
                   (long long)head2) *
                      128 +
                  (long long)role_tid2;
              __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_tmem_load_9[output_token2]);
              dq_out[output_index2] = _cvt_bf16_6;
              __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_tmem_load_11[output_token2]);
              dk_out[output_index2] = _cvt_bf16_7;
              dgate_out[output_index2] = _tmem_load_10[output_token2];
            }
          }
          mbarrier_arrive(qk_raw_done_addr + (qk_raw_stage2) * 8);
          local_grad_stage2 += 1;
          if (local_grad_stage2 == 1) {
            local_grad_stage2 = 0;
            _phase_local_grad_ready ^= 1;
          }
          qk_raw_stage2 += 1;
          if (qk_raw_stage2 == 4) {
            qk_raw_stage2 = 0;
            _phase_qk_raw_ready ^= 1;
          }
          g_prefix_stage2 += 1;
          if (g_prefix_stage2 == 2) {
            g_prefix_stage2 = 0;
            _phase_g_prefix_ready ^= 1;
          }
          state_smem_stage2 += 1;
          if (state_smem_stage2 == 2) {
            state_smem_stage2 = 0;
            _phase_state_ready ^= 1;
          }
        }
      }
      if (elect_sync()) {
        mbarrier_arrive(consumers_done_addr);
      }
    }
  }
  // ---- Role: super_mma ----
  if (warp == 12) {
    {  // super_mma_main
      unsigned int sched_stage3 = 0;
      unsigned int raw_stage3 = 0;
      unsigned int operand_stage3 = 0;
      unsigned int intermediate_stage3 = 0;
      unsigned int u_smem_stage3 = 0;
      unsigned int dy_smem_stage3 = 0;
      unsigned int dbeta_m_stage3 = 0;
      int lhs_row3 = lane % 8 + (lane / 8 & 1) * 8;
      int lhs_col_offset3 = lane / 16 * 8;
      int rhs_row3 = lane % 8 + lane / 16 * 8;
      int rhs_col_offset3 = (lane / 8 & 1) * 8;
      unsigned int _phase_sched_ready_3 = 0;
      unsigned int _phase_raw_ready_2 = 0;
      unsigned int _phase_k_decay_inv_ready = 0;
      unsigned int _phase_intermediate_done = 1;
      unsigned int _phase_u_smem_ready = 0;
      unsigned int _phase_dy_smem_ready = 0;
      unsigned int _phase_dbeta_m_done = 1;
#pragma unroll 1
      for (int __3 = 0; __3 < total_work_items; __3++) {
        mbarrier_wait(sched_ready_addr + (sched_stage3) * 8, _phase_sched_ready_3);
        unsigned int ticket_words_3[1];
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&ticket_words_3[0]))
                     : "r"(work_item_addr + sched_stage3 * 4));
        unsigned int tile3 = ticket_words_3[0];
        if (elect_sync()) {
          mbarrier_arrive(sched_done_addr + (sched_stage3) * 8);
        }
        sched_stage3 += 1;
        if (sched_stage3 == 8) {
          sched_stage3 = 0;
          _phase_sched_ready_3 ^= 1;
        }
        if (tile3 >= (unsigned int)total_work_items) {
          break;
        }
        int item_base3 = (int)tile3 * 8;
        int head3 = work_items[item_base3 + 1];
        int write_start3 = work_items[item_base3 + 2];
        int write_end3 = work_items[item_base3 + 3];
        int compute_end3 = work_items[item_base3 + 5];
        float kk_sum3[8];
        kk_sum3[0] = 0.0f;
        kk_sum3[1] = 0.0f;
        kk_sum3[2] = 0.0f;
        kk_sum3[3] = 0.0f;
        kk_sum3[4] = 0.0f;
        kk_sum3[5] = 0.0f;
        kk_sum3[6] = 0.0f;
        kk_sum3[7] = 0.0f;
        float tinv_sum3[8];
        tinv_sum3[0] = 0.0f;
        tinv_sum3[1] = 0.0f;
        tinv_sum3[2] = 0.0f;
        tinv_sum3[3] = 0.0f;
        tinv_sum3[4] = 0.0f;
        tinv_sum3[5] = 0.0f;
        tinv_sum3[6] = 0.0f;
        tinv_sum3[7] = 0.0f;
#pragma unroll 1
        for (int reverse3 = 0; reverse3 < compute_end3 - write_start3; reverse3++) {
          mbarrier_wait(raw_ready_addr + (raw_stage3) * 8, _phase_raw_ready_2);
          int beta_stage_base3 = (int)(raw_stage3 % 2) * 16 * 8;
          int beta_head3 = head3 % 8;
          int beta_row_lo3 = lane / 4;
          int beta_row_hi3 = beta_row_lo3 + 8;
          __nv_bfloat16 beta_lo_bf3 =
              beta_smem_all[beta_stage_base3 + beta_row_lo3 * 8 + beta_head3];
          __nv_bfloat16 beta_hi_bf3 =
              beta_smem_all[beta_stage_base3 + beta_row_hi3 * 8 + beta_head3];
          float _cvt_f32_22 = __bfloat162float(beta_lo_bf3);
          float beta_lo3 = _cvt_f32_22;
          float _cvt_f32_23 = __bfloat162float(beta_hi_bf3);
          float beta_hi3 = _cvt_f32_23;
          mbarrier_wait(k_decay_inv_ready_addr + (operand_stage3) * 8, _phase_k_decay_inv_ready);
          float kk_acc3[8];
          kk_acc3[0] = 0.0f;
          kk_acc3[1] = 0.0f;
          kk_acc3[2] = 0.0f;
          kk_acc3[3] = 0.0f;
          kk_acc3[4] = 0.0f;
          kk_acc3[5] = 0.0f;
          kk_acc3[6] = 0.0f;
          kk_acc3[7] = 0.0f;
          mbarrier_wait(intermediate_done_addr + (intermediate_stage3) * 8,
                        _phase_intermediate_done);
          if (enable_kk != 0) {
#pragma unroll
            for (int k_block3 = 0; k_block3 < 8; k_block3++) {
              int a_col3 = k_block3 * 16 + lhs_col_offset3;
              int b_col3 = k_block3 * 16 + rhs_col_offset3;
              unsigned int a_frag3[4];
              unsigned int b_frag3[4];
              int segment_8 = a_col3 / 64;
              int segment_col_9 = a_col3 - segment_8 * 64;
              int swizzled_col_9 = segment_col_9 ^ (lhs_row3 & 7) * 8;
              asm volatile(
                  "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(a_frag3[0]), "=r"(a_frag3[1]), "=r"(a_frag3[2]), "=r"(a_frag3[3])
                  : "r"(k_decay_all_addr +
                        (unsigned int)(((int)operand_stage3 * 16 * 128 + segment_8 * 16 * 64 +
                                        lhs_row3 * 64 + swizzled_col_9) *
                                       2))
                  : "memory");
              int segment_0_3 = b_col3 / 64;
              int segment_col_1_3 = b_col3 - segment_0_3 * 64;
              int swizzled_col_2_3 = segment_col_1_3 ^ (rhs_row3 & 7) * 8;
              asm volatile(
                  "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(b_frag3[0]), "=r"(b_frag3[1]), "=r"(b_frag3[2]), "=r"(b_frag3[3])
                  : "r"(k_inv_all_addr +
                        (unsigned int)(((int)operand_stage3 * 16 * 128 + segment_0_3 * 16 * 64 +
                                        rhs_row3 * 64 + swizzled_col_2_3) *
                                       2))
                  : "memory");
              asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                  "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                  : "=f"(kk_acc3[0]), "=f"(kk_acc3[1]), "=f"(kk_acc3[2]), "=f"(kk_acc3[3])
                  : "r"(a_frag3[0]), "r"(a_frag3[1]), "r"(a_frag3[2]), "r"(a_frag3[3]),
                    "r"(b_frag3[0]), "r"(b_frag3[1]), "f"(((k_block3 == 0) ? 0.0f : kk_acc3[0])),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[1])),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[2])),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[3])));
              asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                  "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                  : "=f"(kk_acc3[4]), "=f"(kk_acc3[(4) + 1]), "=f"(kk_acc3[(4) + 2]),
                    "=f"(kk_acc3[(4) + 3])
                  : "r"(a_frag3[0]), "r"(a_frag3[1]), "r"(a_frag3[2]), "r"(a_frag3[3]),
                    "r"(b_frag3[2]), "r"(b_frag3[(2) + 1]),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[4])),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[(4) + 1])),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[(4) + 2])),
                    "f"(((k_block3 == 0) ? 0.0f : kk_acc3[(4) + 3])));
            }
#pragma unroll
            for (int accum3 = 0; accum3 < 8; accum3++) {
              kk_sum3[accum3] = kk_sum3[accum3] + kk_acc3[accum3];
            }
            if (enable_tinv != 0) {
              float l_values3[8];
              float tinv_acc3[8];
#pragma unroll
              for (int accum_t3 = 0; accum_t3 < 8; accum_t3++) {
                int accum_row3 = lane / 4 + accum_t3 % 4 / 2 * 8;
                int accum_col3 = accum_t3 / 4 * 8 + (lane & 3) * 2 + (accum_t3 & 1);
                l_values3[accum_t3] = 0.0f;
                if (accum_row3 > accum_col3) {
                  float beta_scale3 = beta_lo3;
                  if (accum_t3 % 4 >= 2) {
                    beta_scale3 = beta_hi3;
                  }
                  l_values3[accum_t3] = kk_acc3[accum_t3] * beta_scale3;
                }
                tinv_acc3[accum_t3] = -l_values3[accum_t3];
                if (accum_row3 == accum_col3) {
                  tinv_acc3[accum_t3] = 1.0f;
                }
              }
              unsigned int lpow_words3[4];
              unsigned int lpow_trans3[4];
#pragma unroll
              for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(l_values3[_lp * 2 + 0], l_values3[_lp * 2 + 1 + 0]));
                lpow_words3[_lp] = *(uint32_t*)&_bf2;
              }
              int store_row3 = lane % 16;
              int store_col3 = lane / 16 * 8;
              int linear = store_row3 * 16 + store_col3;
              uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(
                  (unsigned long long)(tinv_scratch_addr +
                                       (unsigned int)((linear ^ (linear >> 6 & 1) * 8) * 2)));
              asm volatile(
                  "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                      _stmatrix_addr_0),
                  "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[3]))
                  : "memory");
              __syncwarp();
              int load_row3 = lane % 16;
#pragma unroll
              for (int n_half3 = 0; n_half3 < 2; n_half3++) {
                int load_col3 = n_half3 * 8;
                int linear_0 = load_row3 * 16 + load_col3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(lpow_trans3[n_half3 * 2]), "=r"(lpow_trans3[n_half3 * 2 + 1])
                             : "r"(tinv_scratch_addr +
                                   (unsigned int)((linear_0 ^ (linear_0 >> 6 & 1) * 8) * 2))
                             : "memory");
              }
#pragma unroll
              for (int neumann_round3 = 0; neumann_round3 < 3; neumann_round3++) {
                float square_acc3[8];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                    "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(square_acc3[0]), "=f"(square_acc3[1]), "=f"(square_acc3[2]),
                      "=f"(square_acc3[3])
                    : "r"(lpow_words3[0]), "r"(lpow_words3[1]), "r"(lpow_words3[2]),
                      "r"(lpow_words3[3]), "r"(lpow_trans3[0]), "r"(lpow_trans3[1]), "f"(0.0f),
                      "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                    "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(square_acc3[4]), "=f"(square_acc3[(4) + 1]), "=f"(square_acc3[(4) + 2]),
                      "=f"(square_acc3[(4) + 3])
                    : "r"(lpow_words3[0]), "r"(lpow_words3[1]), "r"(lpow_words3[2]),
                      "r"(lpow_words3[3]), "r"(lpow_trans3[2]), "r"(lpow_trans3[(2) + 1]),
                      "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
#pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                  __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                      make_float2(square_acc3[_lp * 2 + 0], square_acc3[_lp * 2 + 1 + 0]));
                  lpow_words3[_lp] = *(uint32_t*)&_bf2;
                }
                int store_row3_0 = lane % 16;
                int store_col3_1 = lane / 16 * 8;
                int linear_2 = store_row3_0 * 16 + store_col3_1;
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(
                    (unsigned long long)(tinv_scratch_addr +
                                         (unsigned int)((linear_2 ^ (linear_2 >> 6 & 1) * 8) * 2)));
                asm volatile(
                    "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                        _stmatrix_addr_1),
                    "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[0])),
                    "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[1])),
                    "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[2])),
                    "r"(*reinterpret_cast<const uint32_t*>(&lpow_words3[3]))
                    : "memory");
                __syncwarp();
                int load_row3_3 = lane % 16;
#pragma unroll
                for (int n_half3_1 = 0; n_half3_1 < 2; n_half3_1++) {
                  int load_col3_1 = n_half3_1 * 8;
                  int linear_0_1 = load_row3_3 * 16 + load_col3_1;
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                               : "=r"(lpow_trans3[n_half3_1 * 2]),
                                 "=r"(lpow_trans3[n_half3_1 * 2 + 1])
                               : "r"(tinv_scratch_addr +
                                     (unsigned int)((linear_0_1 ^ (linear_0_1 >> 6 & 1) * 8) * 2))
                               : "memory");
                }
                unsigned int tinv_words3[4];
#pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                  __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                      make_float2(tinv_acc3[_lp * 2 + 0], tinv_acc3[_lp * 2 + 1 + 0]));
                  tinv_words3[_lp] = *(uint32_t*)&_bf2;
                }
                float update_acc3[8];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                    "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(update_acc3[0]), "=f"(update_acc3[1]), "=f"(update_acc3[2]),
                      "=f"(update_acc3[3])
                    : "r"(tinv_words3[0]), "r"(tinv_words3[1]), "r"(tinv_words3[2]),
                      "r"(tinv_words3[3]), "r"(lpow_trans3[0]), "r"(lpow_trans3[1]), "f"(0.0f),
                      "f"(0.0f), "f"(0.0f), "f"(0.0f));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, "
                    "%5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(update_acc3[4]), "=f"(update_acc3[(4) + 1]), "=f"(update_acc3[(4) + 2]),
                      "=f"(update_acc3[(4) + 3])
                    : "r"(tinv_words3[0]), "r"(tinv_words3[1]), "r"(tinv_words3[2]),
                      "r"(tinv_words3[3]), "r"(lpow_trans3[2]), "r"(lpow_trans3[(2) + 1]),
                      "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                float tinv_words3_f32[8];
#pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                  asm volatile(
                      "{\n\t"
                      "shl.b32 %0, %2, 16;\n\t"
                      "and.b32 %1, %2, 0xffff0000;\n\t"
                      "}\n"
                      : "=f"((&tinv_words3_f32[_pair * 2])[0]),
                        "=f"((&tinv_words3_f32[_pair * 2])[1])
                      : "r"(tinv_words3[_pair]));
                }
#pragma unroll
                for (int accum_u3 = 0; accum_u3 < 8; accum_u3++) {
                  tinv_acc3[accum_u3] = tinv_words3_f32[accum_u3] + update_acc3[accum_u3];
                }
              }
#pragma unroll
              for (int accum_s3 = 0; accum_s3 < 8; accum_s3++) {
                tinv_sum3[accum_s3] = tinv_sum3[accum_s3] + tinv_acc3[accum_s3];
              }
              unsigned int tinv_publish_words3[4];
#pragma unroll
              for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(tinv_acc3[_lp * 2 + 0], tinv_acc3[_lp * 2 + 1 + 0]));
                tinv_publish_words3[_lp] = *(uint32_t*)&_bf2;
              }
              int publish_row3 = lane % 16;
              int publish_col3 = lane / 16 * 8;
              uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(
                  (unsigned long long)(intermediate_tinv_addr + intermediate_stage3 * 2560 +
                                       (unsigned int)(publish_col3 / 16 * 512 + publish_row3 * 32 +
                                                          publish_col3 % 16 * 2 ^
                                                      (publish_col3 / 16 * 512 + publish_row3 * 32 +
                                                               publish_col3 % 16 * 2 >>
                                                           7 &
                                                       1) << 4)));
              asm volatile(
                  "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                      _stmatrix_addr_2),
                  "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_words3[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_words3[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_words3[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&tinv_publish_words3[3]))
                  : "memory");
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(tinv_ready_addr + (intermediate_stage3) * 8);
          mbarrier_wait(u_smem_ready_addr + (u_smem_stage3) * 8, _phase_u_smem_ready);
          mbarrier_wait(dy_smem_ready_addr + (dy_smem_stage3) * 8, _phase_dy_smem_ready);
          float dm_acc3[8];
#pragma unroll
          for (int k_block_dm3 = 0; k_block_dm3 < 8; k_block_dm3++) {
            int a_col_dm3 = k_block_dm3 * 16 + lhs_col_offset3;
            int b_col_dm3 = k_block_dm3 * 16 + rhs_col_offset3;
            unsigned int dy_frag3[4];
            unsigned int u_frag3[4];
            int segment_10 = a_col_dm3 / 64;
            int segment_col_11 = a_col_dm3 - segment_10 * 64;
            int swizzled_col_10 = segment_col_11 ^ (lhs_row3 & 7) * 8;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(dy_frag3[0]), "=r"(dy_frag3[1]), "=r"(dy_frag3[2]), "=r"(dy_frag3[3])
                : "r"(dy_smem_all_addr +
                      (unsigned int)((segment_10 * 16 * 64 + lhs_row3 * 64 + swizzled_col_10) * 2))
                : "memory");
            int segment_0_4 = b_col_dm3 / 64;
            int segment_col_1_4 = b_col_dm3 - segment_0_4 * 64;
            int swizzled_col_2_4 = segment_col_1_4 ^ (rhs_row3 & 7) * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(u_frag3[0]), "=r"(u_frag3[1]), "=r"(u_frag3[2]), "=r"(u_frag3[3])
                         : "r"(u_smem_addr + (unsigned int)((segment_0_4 * 16 * 64 + rhs_row3 * 64 +
                                                             swizzled_col_2_4) *
                                                            2))
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(dm_acc3[0]), "=f"(dm_acc3[1]), "=f"(dm_acc3[2]), "=f"(dm_acc3[3])
                : "r"(dy_frag3[0]), "r"(dy_frag3[1]), "r"(dy_frag3[2]), "r"(dy_frag3[3]),
                  "r"(u_frag3[0]), "r"(u_frag3[1]), "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[0])),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[1])),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[2])),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[3])));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(dm_acc3[4]), "=f"(dm_acc3[(4) + 1]), "=f"(dm_acc3[(4) + 2]),
                  "=f"(dm_acc3[(4) + 3])
                : "r"(dy_frag3[0]), "r"(dy_frag3[1]), "r"(dy_frag3[2]), "r"(dy_frag3[3]),
                  "r"(u_frag3[2]), "r"(u_frag3[(2) + 1]),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[4])),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[(4) + 1])),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[(4) + 2])),
                  "f"(((k_block_dm3 == 0) ? 0.0f : dm_acc3[(4) + 3])));
          }
          float dm_strict3[8];
          float ndm_strict3[8];
#pragma unroll
          for (int accum_dm3 = 0; accum_dm3 < 8; accum_dm3++) {
            int dm_row3 = lane / 4 + accum_dm3 % 4 / 2 * 8;
            int dm_col3 = accum_dm3 / 4 * 8 + (lane & 3) * 2 + (accum_dm3 & 1);
            dm_strict3[accum_dm3] = 0.0f;
            if (dm_row3 > dm_col3) {
              float beta_dm_scale3 = beta_lo3;
              if (accum_dm3 % 4 >= 2) {
                beta_dm_scale3 = beta_hi3;
              }
              dm_strict3[accum_dm3] = dm_acc3[accum_dm3] * beta_dm_scale3;
            }
            ndm_strict3[accum_dm3] = -dm_strict3[accum_dm3];
          }
          unsigned int dm_words3[4];
          unsigned int ndm_words3[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(dm_strict3[_lp * 2 + 0], dm_strict3[_lp * 2 + 1 + 0]));
            dm_words3[_lp] = *(uint32_t*)&_bf2;
          }
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(ndm_strict3[_lp * 2 + 0], ndm_strict3[_lp * 2 + 1 + 0]));
            ndm_words3[_lp] = *(uint32_t*)&_bf2;
          }
          int dm_publish_row3 = lane % 16;
          int dm_publish_col3 = lane / 16 * 8;
          uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((
              unsigned long long)(intermediate_dm_addr + intermediate_stage3 * 2560 +
                                  (unsigned int)(dm_publish_col3 / 16 * 512 + dm_publish_row3 * 32 +
                                                     dm_publish_col3 % 16 * 2 ^
                                                 (dm_publish_col3 / 16 * 512 +
                                                          dm_publish_row3 * 32 +
                                                          dm_publish_col3 % 16 * 2 >>
                                                      7 &
                                                  1) << 4)));
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                           _stmatrix_addr_3),
                       "r"(*reinterpret_cast<const uint32_t*>(&dm_words3[0])),
                       "r"(*reinterpret_cast<const uint32_t*>(&dm_words3[1])),
                       "r"(*reinterpret_cast<const uint32_t*>(&dm_words3[2])),
                       "r"(*reinterpret_cast<const uint32_t*>(&dm_words3[3]))
                       : "memory");
          uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((
              unsigned long long)(intermediate_ndm_addr + intermediate_stage3 * 2560 +
                                  (unsigned int)(dm_publish_col3 / 16 * 512 + dm_publish_row3 * 32 +
                                                     dm_publish_col3 % 16 * 2 ^
                                                 (dm_publish_col3 / 16 * 512 +
                                                          dm_publish_row3 * 32 +
                                                          dm_publish_col3 % 16 * 2 >>
                                                      7 &
                                                  1) << 4)));
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                           _stmatrix_addr_4),
                       "r"(*reinterpret_cast<const uint32_t*>(&ndm_words3[0])),
                       "r"(*reinterpret_cast<const uint32_t*>(&ndm_words3[1])),
                       "r"(*reinterpret_cast<const uint32_t*>(&ndm_words3[2])),
                       "r"(*reinterpret_cast<const uint32_t*>(&ndm_words3[3]))
                       : "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(dm_ready_addr + (intermediate_stage3) * 8);
          float dbeta_m_lo3 = 0.0f;
          float dbeta_m_hi3 = 0.0f;
#pragma unroll
          for (int dbeta_accum3 = 0; dbeta_accum3 < 8; dbeta_accum3++) {
            int dbeta_row3 = lane / 4 + dbeta_accum3 % 4 / 2 * 8;
            int dbeta_col3 = dbeta_accum3 / 4 * 8 + (lane & 3) * 2 + (dbeta_accum3 & 1);
            float dbeta_m_part3 = 0.0f;
            if (dbeta_row3 > dbeta_col3) {
              dbeta_m_part3 = dm_acc3[dbeta_accum3] * kk_acc3[dbeta_accum3];
            }
            if (dbeta_accum3 % 4 < 2) {
              dbeta_m_lo3 += dbeta_m_part3;
            } else {
              dbeta_m_hi3 += dbeta_m_part3;
            }
          }
          float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, dbeta_m_lo3, 1);
          dbeta_m_lo3 += _shfl_xor_13;
          float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, dbeta_m_lo3, 2);
          dbeta_m_lo3 += _shfl_xor_14;
          float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, dbeta_m_hi3, 1);
          dbeta_m_hi3 += _shfl_xor_15;
          float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, dbeta_m_hi3, 2);
          dbeta_m_hi3 += _shfl_xor_16;
          mbarrier_wait(dbeta_m_done_addr + (dbeta_m_stage3) * 8, _phase_dbeta_m_done);
          if (lane % 4 == 0) {
            dbeta_m_smem[beta_row_lo3] = -dbeta_m_lo3;
            dbeta_m_smem[beta_row_hi3] = -dbeta_m_hi3;
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(dbeta_m_ready_addr + (dbeta_m_stage3) * 8);
          }
          dbeta_m_stage3 += 1;
          if (dbeta_m_stage3 == 1) {
            dbeta_m_stage3 = 0;
            _phase_dbeta_m_done ^= 1;
          }
          raw_stage3 += 1;
          if (raw_stage3 == 2) {
            raw_stage3 = 0;
            _phase_raw_ready_2 ^= 1;
          }
          operand_stage3 += 1;
          if (operand_stage3 == 2) {
            operand_stage3 = 0;
            _phase_k_decay_inv_ready ^= 1;
          }
          intermediate_stage3 += 1;
          if (intermediate_stage3 == 2) {
            intermediate_stage3 = 0;
            _phase_intermediate_done ^= 1;
          }
          u_smem_stage3 += 1;
          if (u_smem_stage3 == 1) {
            u_smem_stage3 = 0;
            _phase_u_smem_ready ^= 1;
          }
          dy_smem_stage3 += 1;
          if (dy_smem_stage3 == 1) {
            dy_smem_stage3 = 0;
            _phase_dy_smem_ready ^= 1;
          }
        }
        if (validate_outputs != 0 && enable_kk != 0) {
          int row_lo3 = lane / 4;
          int col_pair3 = (lane & 3) * 2;
#pragma unroll
          for (int accum4 = 0; accum4 < 8; accum4++) {
            int output_row3 = row_lo3 + accum4 % 4 / 2 * 8;
            int output_col3 = accum4 / 4 * 8 + col_pair3 + (accum4 & 1);
            kk_observed[(long long)tile3 * 16 * 16 + (long long)output_row3 * 16 +
                        (long long)output_col3] = kk_sum3[accum4];
            if (enable_tinv != 0) {
              tinv_observed[(long long)tile3 * 16 * 16 + (long long)output_row3 * 16 +
                            (long long)output_col3] = tinv_sum3[accum4];
            }
          }
        }
      }
      if (elect_sync()) {
        mbarrier_arrive(consumers_done_addr);
      }
    }
  }
  // ---- Role: tcgen ----
  if (warp == 13) {
    {  // tcgen_main
      float tmem_seed[1];
      tmem_seed[0] = 0.0f;
      asm volatile(
          "tcgen05.st.sync.aligned.32x32b.x1.b32"
          " [%0], {%1};" ::"r"(taddr),
          "r"(*reinterpret_cast<const uint32_t*>(&tmem_seed[0])));
      asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      unsigned int sched_stage4 = 0;
      unsigned int raw_stage4 = 0;
      unsigned int state_smem_stage4 = 0;
      unsigned int operand_stage4 = 0;
      unsigned int intermediate_stage4 = 0;
      unsigned int tcgen_data_stage4 = 0;
      unsigned int dstate_recurrence_stage4 = 0;
      unsigned int u_smem_stage4 = 0;
      unsigned int dstate_smem_stage4 = 0;
      unsigned int dk_restore_stage4 = 0;
      unsigned int local_grad_stage4 = 0;
      unsigned int beta_dy_smem_stage4 = 0;
      unsigned int boundary_stage4 = 0;
      unsigned int boundary_local_grad_stage4 = 0;
      unsigned int _phase_sched_ready_4 = 0;
      unsigned int _phase_dstate_done = 1;
      unsigned int _phase_local_grad_done = 1;
      unsigned int _phase_k_decay_inv_ready_1 = 0;
      unsigned int _phase_state_k_done = 1;
      unsigned int _phase_state_ready_1 = 0;
      unsigned int _phase_raw_ready_3 = 0;
      unsigned int _phase_q_decay_k_restore_ready = 0;
      unsigned int _phase_tinv_ready = 0;
      unsigned int _phase_a_ready = 0;
      unsigned int _phase_tcgen_inputs_ready = 0;
      unsigned int _phase_tcgen_products_done = 1;
      unsigned int _phase_dy_done = 1;
      unsigned int _phase_dstate_inp_ready = 0;
      unsigned int _phase_tcgen_products_ready_1 = 0;
      unsigned int _phase_du_inp_ready = 0;
      unsigned int _phase_u_smem_ready_1 = 0;
      unsigned int _phase_dstate_smem_ready = 0;
      unsigned int _phase_dk_restore_done = 1;
      unsigned int _phase_dy_ready_1 = 0;
      unsigned int _phase_neg_dy_ready = 0;
      unsigned int _phase_dstate_ready_1 = 0;
      unsigned int _phase_beta_dy_smem_ready = 0;
      unsigned int _phase_da_ready = 0;
      unsigned int _phase_dm_ready = 0;
      unsigned int _phase_boundary_smem_ready = 0;
      unsigned int _phase_boundary_local_grad_free = 0;
#pragma unroll 1
      for (int __4 = 0; __4 < total_work_items; __4++) {
        mbarrier_wait(sched_ready_addr + (sched_stage4) * 8, _phase_sched_ready_4);
        unsigned int ticket_words_4[1];
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&ticket_words_4[0]))
                     : "r"(work_item_addr + sched_stage4 * 4));
        unsigned int tile4 = ticket_words_4[0];
        if (elect_sync()) {
          mbarrier_arrive(sched_done_addr + (sched_stage4) * 8);
        }
        sched_stage4 += 1;
        if (sched_stage4 == 8) {
          sched_stage4 = 0;
          _phase_sched_ready_4 ^= 1;
        }
        if (tile4 >= (unsigned int)total_work_items) {
          break;
        }
        int item_base4 = (int)tile4 * 8;
        int write_start4 = work_items[item_base4 + 2];
        int write_end4 = work_items[item_base4 + 3];
        int compute_end4 = work_items[item_base4 + 5];
        float operand_sums4[16];
        operand_sums4[0] = 0.0f;
        operand_sums4[1] = 0.0f;
        operand_sums4[2] = 0.0f;
        operand_sums4[3] = 0.0f;
        operand_sums4[4] = 0.0f;
        operand_sums4[5] = 0.0f;
        operand_sums4[6] = 0.0f;
        operand_sums4[7] = 0.0f;
        operand_sums4[8] = 0.0f;
        operand_sums4[9] = 0.0f;
        operand_sums4[10] = 0.0f;
        operand_sums4[11] = 0.0f;
        operand_sums4[12] = 0.0f;
        operand_sums4[13] = 0.0f;
        operand_sums4[14] = 0.0f;
        operand_sums4[15] = 0.0f;
#pragma unroll 1
        for (int reverse4 = 0; reverse4 < compute_end4 - write_start4; reverse4++) {
          mbarrier_wait(dstate_done_addr + (tcgen_data_stage4) * 8, _phase_dstate_done);
          mbarrier_wait(local_grad_done_addr + (local_grad_stage4) * 8, _phase_local_grad_done);
          asm volatile("tcgen05.fence::after_thread_sync;");
          mbarrier_wait(k_decay_inv_ready_addr + (operand_stage4) * 8, _phase_k_decay_inv_ready_1);
          mbarrier_wait(state_k_done_addr + (state_smem_stage4) * 8, _phase_state_k_done);
          mbarrier_wait(state_ready_addr + (state_smem_stage4) * 8, _phase_state_ready_1);
          mbarrier_wait(raw_ready_addr + (raw_stage4) * 8, _phase_raw_ready_3);
          int _mma_a_lo_0 = make_warp_uniform((((state_operand_addr) >> 4) & 0x3FFF) +
                                              (state_smem_stage4) * 2048);
          int _mma_b_lo_0 =
              make_warp_uniform((((k_decay_lead16_addr) >> 4) & 0x3FFF) + (operand_stage4) * 256);
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
              "}\n" ::"r"(_mma_a_lo_0),
              "r"(_mma_b_lo_0), "r"(tmem_flashkda_bwd_persistent_c16_state_k), "r"(0));
          int _mma_a_lo_1 = make_warp_uniform(
              ((((state_operand_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (state_smem_stage4) * 2048);
          int _mma_b_lo_1 =
              make_warp_uniform(((((raw_do_addr) >> 4) & 0x3FFF) | 0x800000) + (raw_stage4) * 256);
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
              "}\n" ::"r"(_mma_a_lo_1),
              "r"(_mma_b_lo_1), "r"(tmem_flashkda_bwd_persistent_c16_dq), "r"(0));
          elect_commit(state_k_ready_addr + (state_smem_stage4) * 8);
          mbarrier_wait(q_decay_k_restore_ready_addr + (operand_stage4) * 8,
                        _phase_q_decay_k_restore_ready);
          mbarrier_wait(tinv_ready_addr + (intermediate_stage4) * 8, _phase_tinv_ready);
          mbarrier_wait(a_ready_addr + (intermediate_stage4) * 8, _phase_a_ready);
          mbarrier_wait(tcgen_inputs_ready_addr + (tcgen_data_stage4) * 8,
                        _phase_tcgen_inputs_ready);
          mbarrier_wait(tcgen_products_done_addr + (tcgen_data_stage4) * 8,
                        _phase_tcgen_products_done);
          mbarrier_wait(dy_done_addr + (tcgen_data_stage4) * 8, _phase_dy_done);
          mbarrier_wait(dstate_inp_ready_addr + (dstate_recurrence_stage4) * 8,
                        _phase_dstate_inp_ready);
          if (USE_DSTATE_IN != 0 || reverse4 > 0) {
#pragma unroll
            for (int dstate_block4 = 0; dstate_block4 < 8; dstate_block4++) {
              int _mma_b_lo_2 = make_warp_uniform((((state_scale_diag_addr) >> 4) & 0x3FFF) +
                                                  ((int)operand_stage4 * 8 + dstate_block4) * 32);
              mma_ts_step((tmem_flashkda_bwd_persistent_c16_dstate + (dstate_block4 * 16)),
                          tmem_flashkda_bwd_persistent_c16_dstate_inp + dstate_block4 * 8,
                          _mma_b_lo_2, 0xC0004010, 134481040, 0);
            }
            int _mma_b_lo_3 = make_warp_uniform((((k_restore_lead16_addr) >> 4) & 0x3FFF) +
                                                (operand_stage4) * 256);
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
                "mov.b32 id, 134481040;\n\t"
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
                "}\n" ::"r"(tmem_flashkda_bwd_persistent_c16_du),
                "r"(_mma_b_lo_3), "r"(tmem_flashkda_bwd_persistent_c16_dstate_inp), "r"(0));
          }
          dstate_recurrence_stage4 += 1;
          if (dstate_recurrence_stage4 == 1) {
            dstate_recurrence_stage4 = 0;
            _phase_dstate_inp_ready ^= 1;
          }
          int _mma_b_lo_4 = make_warp_uniform(((((q_decay_trans_addr) >> 4) & 0x3FFF) | 0x800000) +
                                              (operand_stage4) * 256);
          mma_ts_step(tmem_flashkda_bwd_persistent_c16_dstate,
                      tmem_flashkda_bwd_persistent_c16_do_inp, _mma_b_lo_4, 0x40004040, 136381584,
                      ((reverse4 == 0 && USE_DSTATE_IN == 0) ? 0 : 1));
          int _mma_b_lo_5 = make_warp_uniform((((intermediate_tinv_addr) >> 4) & 0x3FFF) +
                                              (intermediate_stage4) * 160);
          mma_ts_step(tmem_flashkda_bwd_persistent_c16_u, tmem_flashkda_bwd_persistent_c16_y,
                      _mma_b_lo_5, 0xC0004010, 134481040, 0);
          int _mma_a_lo_6 = make_warp_uniform(((((raw_do_amaj_addr) >> 4) & 0x3FFF) | 0x800000) +
                                              (raw_stage4) * 256);
          int _mma_b_lo_6 =
              make_warp_uniform(((((intermediate_a_mn_addr) >> 4) & 0x3FFF) | 0x200000) +
                                (intermediate_stage4) * 160);
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
              "mov.b32 bdhi, 0xC0004010;\n\t"
              "mov.b32 id, 134579344;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
              "}\n" ::"r"(_mma_a_lo_6),
              "r"(_mma_b_lo_6), "r"(tmem_flashkda_bwd_persistent_c16_du),
              "r"(((reverse4 == 0 && USE_DSTATE_IN == 0) ? 0 : 1)));
          elect_commit(tcgen_products_ready_addr + (tcgen_data_stage4) * 8);
          mbarrier_wait(tcgen_products_ready_addr + (tcgen_data_stage4) * 8,
                        _phase_tcgen_products_ready_1);
          mbarrier_wait(du_inp_ready_addr + (tcgen_data_stage4) * 8, _phase_du_inp_ready);
          int _mma_b_lo_7 =
              make_warp_uniform(((((intermediate_tinv_mn_addr) >> 4) & 0x3FFF) | 0x200000) +
                                (intermediate_stage4) * 160);
          mma_ts_step(tmem_flashkda_bwd_persistent_c16_dy, tmem_flashkda_bwd_persistent_c16_du_inp,
                      _mma_b_lo_7, 0xC0004010, 134546576, 0);
          elect_commit(dy_ready_addr + (tcgen_data_stage4) * 8);
          mbarrier_wait(u_smem_ready_addr + (u_smem_stage4) * 8, _phase_u_smem_ready_1);
          if (USE_DSTATE_IN != 0 || reverse4 > 0) {
            mbarrier_wait(dstate_smem_ready_addr + (dstate_smem_stage4) * 8,
                          _phase_dstate_smem_ready);
            mbarrier_wait(dk_restore_done_addr + (dk_restore_stage4) * 8, _phase_dk_restore_done);
            int _mma_a_lo_8 =
                make_warp_uniform((((dstate_smem_mn_addr) >> 4) & 0x3FFF) | 0x4000000);
            int _mma_b_lo_8 = make_warp_uniform((((u_lead16_addr) >> 4) & 0x3FFF) | 0x800000);
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
                "}\n" ::"r"(_mma_a_lo_8),
                "r"(_mma_b_lo_8), "r"(tmem_flashkda_bwd_persistent_c16_dk_restore), "r"(0));
            elect_commit2(dk_restore_ready_addr + (dk_restore_stage4) * 8,
                          dstate_smem_done_addr + (dstate_smem_stage4) * 8);
            dstate_smem_stage4 += 1;
            if (dstate_smem_stage4 == 1) {
              dstate_smem_stage4 = 0;
              _phase_dstate_smem_ready ^= 1;
            }
            dk_restore_stage4 += 1;
            if (dk_restore_stage4 == 1) {
              dk_restore_stage4 = 0;
              _phase_dk_restore_done ^= 1;
            }
          }
          u_smem_stage4 += 1;
          if (u_smem_stage4 == 1) {
            u_smem_stage4 = 0;
            _phase_u_smem_ready_1 ^= 1;
          }
          mbarrier_wait(dy_ready_addr + (tcgen_data_stage4) * 8, _phase_dy_ready_1);
          mbarrier_wait(neg_dy_ready_addr + (tcgen_data_stage4) * 8, _phase_neg_dy_ready);
          int _mma_b_lo_9 = make_warp_uniform(((((k_decay_trans_addr) >> 4) & 0x3FFF) | 0x800000) +
                                              (operand_stage4) * 256);
          mma_ts_step(tmem_flashkda_bwd_persistent_c16_dstate,
                      tmem_flashkda_bwd_persistent_c16_neg_dy, _mma_b_lo_9, 0x40004040, 136381584,
                      1);
          elect_commit(dstate_ready_addr + (tcgen_data_stage4) * 8);
          mbarrier_wait(dstate_ready_addr + (tcgen_data_stage4) * 8, _phase_dstate_ready_1);
          mbarrier_wait(beta_dy_smem_ready_addr + (beta_dy_smem_stage4) * 8,
                        _phase_beta_dy_smem_ready);
          int _mma_a_lo_10 = make_warp_uniform((((state_operand_addr) >> 4) & 0x3FFF) +
                                               (state_smem_stage4) * 2048);
          int _mma_b_lo_10 = make_warp_uniform((((beta_dy_smem_addr) >> 4) & 0x3FFF) +
                                               (beta_dy_smem_stage4) * 256);
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
              "}\n" ::"r"(_mma_a_lo_10),
              "r"(_mma_b_lo_10), "r"(tmem_flashkda_bwd_persistent_c16_dk_decay), "r"(0));
          mbarrier_wait(da_ready_addr + (intermediate_stage4) * 8, _phase_da_ready);
          mbarrier_wait(dm_ready_addr + (intermediate_stage4) * 8, _phase_dm_ready);
          int _mma_a_lo_11 = make_warp_uniform(((((q_decay_trans_addr) >> 4) & 0x3FFF) | 0x800000) +
                                               (operand_stage4) * 256);
          int _mma_b_lo_11 =
              make_warp_uniform(((((intermediate_da_mn_addr) >> 4) & 0x3FFF) | 0x200000) +
                                (intermediate_stage4) * 160);
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
              "mov.b32 bdhi, 0xC0004010;\n\t"
              "mov.b32 id, 134579344;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
              "}\n" ::"r"(_mma_a_lo_11),
              "r"(_mma_b_lo_11), "r"(tmem_flashkda_bwd_persistent_c16_dk_inv), "r"(0));
          int _mma_a_lo_12 = make_warp_uniform(((((k_inv_amaj_addr) >> 4) & 0x3FFF) | 0x800000) +
                                               (operand_stage4) * 256);
          int _mma_b_lo_12 = make_warp_uniform(
              ((((intermediate_da_addr) >> 4) & 0x3FFF) | 0x200000) + (intermediate_stage4) * 160);
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
              "mov.b32 bdhi, 0xC0004010;\n\t"
              "mov.b32 id, 134513808;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
              "}\n" ::"r"(_mma_a_lo_12),
              "r"(_mma_b_lo_12), "r"(tmem_flashkda_bwd_persistent_c16_dq), "r"(1));
          int _mma_a_lo_13 = make_warp_uniform(((((k_decay_trans_addr) >> 4) & 0x3FFF) | 0x800000) +
                                               (operand_stage4) * 256);
          int _mma_b_lo_13 =
              make_warp_uniform(((((intermediate_ndm_mn_addr) >> 4) & 0x3FFF) | 0x200000) +
                                (intermediate_stage4) * 160);
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
              "mov.b32 bdhi, 0xC0004010;\n\t"
              "mov.b32 id, 134579344;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
              "}\n" ::"r"(_mma_a_lo_13),
              "r"(_mma_b_lo_13), "r"(tmem_flashkda_bwd_persistent_c16_dk_inv), "r"(1));
          int _mma_a_lo_14 = make_warp_uniform(((((k_inv_amaj_addr) >> 4) & 0x3FFF) | 0x800000) +
                                               (operand_stage4) * 256);
          int _mma_b_lo_14 = make_warp_uniform(
              ((((intermediate_dm_addr) >> 4) & 0x3FFF) | 0x200000) + (intermediate_stage4) * 160);
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
              "mov.b32 bdhi, 0xC0004010;\n\t"
              "mov.b32 id, 134513808;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
              "}\n" ::"r"(_mma_a_lo_14),
              "r"(_mma_b_lo_14), "r"(tmem_flashkda_bwd_persistent_c16_dk_decay), "r"(1));
          elect_commit(local_grad_ready_addr + (local_grad_stage4) * 8);
          elect_commit(beta_dy_smem_done_addr + (beta_dy_smem_stage4) * 8);
          mbarrier_wait(boundary_smem_ready_addr + (boundary_stage4) * 8,
                        _phase_boundary_smem_ready);
          mbarrier_wait(boundary_local_grad_free_addr + (boundary_local_grad_stage4) * 8,
                        _phase_boundary_local_grad_free);
          int _mma_a_lo_15 = make_warp_uniform((((dstate_smem_mn_addr) >> 4) & 0x3FFF) | 0x4000000);
          int _mma_b_lo_15 = make_warp_uniform((((u_lead16_addr) >> 4) & 0x3FFF) | 0x800000);
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
              "}\n" ::"r"(_mma_a_lo_15),
              "r"(_mma_b_lo_15), "r"(tmem_flashkda_bwd_persistent_c16_dk_decay), "r"(0));
          elect_commit2(boundary_acc_ready_addr + (boundary_stage4) * 8,
                        state_slot_done_addr + (state_smem_stage4) * 8);
          boundary_stage4 += 1;
          if (boundary_stage4 == 1) {
            boundary_stage4 = 0;
            _phase_boundary_smem_ready ^= 1;
          }
          boundary_local_grad_stage4 += 1;
          if (boundary_local_grad_stage4 == 1) {
            boundary_local_grad_stage4 = 0;
            _phase_boundary_local_grad_free ^= 1;
          }
          local_grad_stage4 += 1;
          if (local_grad_stage4 == 1) {
            local_grad_stage4 = 0;
            _phase_local_grad_done ^= 1;
          }
          beta_dy_smem_stage4 += 1;
          if (beta_dy_smem_stage4 == 2) {
            beta_dy_smem_stage4 = 0;
            _phase_beta_dy_smem_ready ^= 1;
          }
          if (elect_sync()) {
            mbarrier_arrive(tcgen_inputs_done_addr + (tcgen_data_stage4) * 8);
            mbarrier_arrive(raw_done_addr + (raw_stage4) * 8);
            mbarrier_arrive(intermediate_done_addr + (intermediate_stage4) * 8);
            mbarrier_arrive(decay_done_addr + (operand_stage4) * 8);
          }
          operand_stage4 += 1;
          if (operand_stage4 == 2) {
            operand_stage4 = 0;
            _phase_k_decay_inv_ready_1 ^= 1;
            _phase_q_decay_k_restore_ready ^= 1;
          }
          raw_stage4 += 1;
          if (raw_stage4 == 2) {
            raw_stage4 = 0;
            _phase_raw_ready_3 ^= 1;
          }
          state_smem_stage4 += 1;
          if (state_smem_stage4 == 2) {
            state_smem_stage4 = 0;
            _phase_state_k_done ^= 1;
            _phase_state_ready_1 ^= 1;
          }
          intermediate_stage4 += 1;
          if (intermediate_stage4 == 2) {
            intermediate_stage4 = 0;
            _phase_tinv_ready ^= 1;
            _phase_a_ready ^= 1;
            _phase_da_ready ^= 1;
            _phase_dm_ready ^= 1;
          }
          tcgen_data_stage4 += 1;
          if (tcgen_data_stage4 == 1) {
            tcgen_data_stage4 = 0;
            _phase_dstate_done ^= 1;
            _phase_tcgen_inputs_ready ^= 1;
            _phase_tcgen_products_done ^= 1;
            _phase_dy_done ^= 1;
            _phase_tcgen_products_ready_1 ^= 1;
            _phase_du_inp_ready ^= 1;
            _phase_dy_ready_1 ^= 1;
            _phase_neg_dy_ready ^= 1;
            _phase_dstate_ready_1 ^= 1;
          }
        }
      }
      if (elect_sync()) {
        mbarrier_arrive(consumers_done_addr);
      }
      unsigned int _phase_cleanup_ready_0 = 0;
      mbarrier_wait(cleanup_ready_addr, _phase_cleanup_ready_0);
      _phase_cleanup_ready_0 ^= 1;
      int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
      asm volatile(
          "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(_tmem_dealloc_addr),
          "r"(512));
    }
  }
  // ---- Role: tma ----
  if (warp == 14) {
    {  // tma_main
      unsigned int sched_stage6 = 0;
      unsigned int raw_stage6 = 0;
      unsigned int state_smem_stage6 = 0;
      unsigned int _phase_sched_done = 1;
      unsigned int _phase_state_slot_done = 1;
      unsigned int _phase_state_cg2_done = 1;
      unsigned int _phase_raw_done = 1;
      if (elect_sync()) {
#pragma unroll 1
        for (int sched_iter6 = 0; sched_iter6 < total_work_items; sched_iter6++) {
          mbarrier_wait(sched_done_addr + (sched_stage6) * 8, _phase_sched_done);
          unsigned int tile6 = blockIdx.x;
          if (uniform_work_items != 0) {
            tile6 = (unsigned int)blockIdx.x + (unsigned int)sched_iter6 * (unsigned int)gridDim.x;
          } else {
            unsigned int _atomic_old_0 = atomicAdd(dynamic_counter, 1);
            tile6 = _atomic_old_0;
          }
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(work_item_addr + sched_stage6 * 4),
                       "r"(tile6));
          mbarrier_arrive(sched_ready_addr + (sched_stage6) * 8);
          sched_stage6 += 1;
          if (sched_stage6 == 8) {
            sched_stage6 = 0;
            _phase_sched_done ^= 1;
          }
          if (tile6 >= (unsigned int)total_work_items) {
            break;
          }
          int item_base6 = (int)tile6 * 8;
          int sequence6 = work_items[item_base6];
          int head6 = work_items[item_base6 + 1];
          int qk_head6 = head6 * num_qk_heads / num_heads;
          int write_start6 = work_items[item_base6 + 2];
          int write_end6 = work_items[item_base6 + 3];
          int compute_end6 = work_items[item_base6 + 5];
          long long bos6 = cu_seqlens[sequence6];
#pragma unroll 1
          for (int reverse6 = 0; reverse6 < compute_end6 - write_start6; reverse6++) {
            int chunk6 = compute_end6 - 1 - reverse6;
            int token6 = (int)(bos6 + (long long)chunk6 * 16);
            int checkpoint6 = (int)(checkpoint_cu_starts[sequence6] + (long long)chunk6);
            mbarrier_wait(state_slot_done_addr + (state_smem_stage6) * 8, _phase_state_slot_done);
            mbarrier_wait(state_cg2_done_addr + (state_smem_stage6) * 8, _phase_state_cg2_done);
            mbarrier_arrive_expect_tx(state_ready_addr + (state_smem_stage6) * 8, 32768);
#pragma unroll
            for (int state_segment6 = 0; state_segment6 < 2; state_segment6++) {
              tma_4d_gmem2smem(
                  state_panel_addr +
                      (state_smem_stage6 % 2 * 2 + (unsigned int)state_segment6) * 16384,
                  (&state_tma), state_segment6 * 64, 0, head6, checkpoint6,
                  state_ready_addr + (state_smem_stage6) * 8);
            }
            state_smem_stage6 += 1;
            if (state_smem_stage6 == 2) {
              state_smem_stage6 = 0;
              _phase_state_slot_done ^= 1;
              _phase_state_cg2_done ^= 1;
            }
            mbarrier_wait(raw_done_addr + (raw_stage6) * 8, _phase_raw_done);
            mbarrier_arrive_expect_tx(raw_ready_addr + (raw_stage6) * 8, 20736);
            tma_2d_gmem2smem(beta_smem_addr + raw_stage6 * 256, (&beta_tma), head6 / 8 * 8, token6,
                             raw_ready_addr + (raw_stage6) * 8);
#pragma unroll
            for (int raw_segment6 = 0; raw_segment6 < 2; raw_segment6++) {
              int raw_segment_offset6 = raw_segment6 * 16 * 64 * 2;
              int raw_channel6 = raw_segment6 * 64;
              tma_3d_gmem2smem(raw_q_addr + raw_stage6 * 4096 + (unsigned int)raw_segment_offset6,
                               (&q_tma), raw_channel6, qk_head6, token6,
                               raw_ready_addr + (raw_stage6) * 8);
              tma_3d_gmem2smem(raw_k_addr + raw_stage6 * 4096 + (unsigned int)raw_segment_offset6,
                               (&k_tma), raw_channel6, qk_head6, token6,
                               raw_ready_addr + (raw_stage6) * 8);
              tma_3d_gmem2smem(raw_g_addr + raw_stage6 * 4096 + (unsigned int)raw_segment_offset6,
                               (&g_tma), raw_channel6, head6, token6,
                               raw_ready_addr + (raw_stage6) * 8);
              tma_3d_gmem2smem(raw_do_addr + raw_stage6 * 4096 + (unsigned int)raw_segment_offset6,
                               (&do_tma), raw_channel6, head6, token6,
                               raw_ready_addr + (raw_stage6) * 8);
              tma_3d_gmem2smem(raw_v_addr + raw_stage6 * 4096 + (unsigned int)raw_segment_offset6,
                               (&v_tma), raw_channel6, head6, token6,
                               raw_ready_addr + (raw_stage6) * 8);
            }
            raw_stage6 += 1;
            if (raw_stage6 == 2) {
              raw_stage6 = 0;
              _phase_raw_done ^= 1;
            }
          }
        }
      }
      unsigned int _phase_consumers_done_0 = 0;
      mbarrier_wait(consumers_done_addr, _phase_consumers_done_0);
      _phase_consumers_done_0 ^= 1;
      if (elect_sync()) {
        mbarrier_arrive(cleanup_ready_addr);
      }
    }
  }
  // ---- Role: epilogue ----
  if (warp == 15) {
    {  // epilogue_main
      unsigned int sched_stage5 = 0;
      unsigned int raw_stage5 = 0;
      unsigned int operand_stage5 = 0;
      unsigned int intermediate_stage5 = 0;
      unsigned int u_smem_stage5 = 0;
      unsigned int dv_stage5 = 0;
      int lhs_row5 = lane % 8 + (lane / 8 & 1) * 8;
      int lhs_col_offset5 = lane / 16 * 8;
      int rhs_row5 = lane % 8 + lane / 16 * 8;
      int rhs_col_offset5 = (lane / 8 & 1) * 8;
      unsigned int _phase_sched_ready_5 = 0;
      unsigned int _phase_q_decay_k_restore_ready_1 = 0;
      unsigned int _phase_raw_ready_4 = 0;
      unsigned int _phase_intermediate_done_1 = 1;
      unsigned int _phase_beta_dy_smem_ready_1 = 0;
      unsigned int _phase_u_smem_ready_2 = 0;
#pragma unroll 1
      for (int __5 = 0; __5 < total_work_items; __5++) {
        mbarrier_wait(sched_ready_addr + (sched_stage5) * 8, _phase_sched_ready_5);
        unsigned int ticket_words_5[1];
        asm volatile("ld.shared.b32 %0, [%1];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&ticket_words_5[0]))
                     : "r"(work_item_addr + sched_stage5 * 4));
        unsigned int tile5 = ticket_words_5[0];
        if (elect_sync()) {
          mbarrier_arrive(sched_done_addr + (sched_stage5) * 8);
        }
        sched_stage5 += 1;
        if (sched_stage5 == 8) {
          sched_stage5 = 0;
          _phase_sched_ready_5 ^= 1;
        }
        if (tile5 >= (unsigned int)total_work_items) {
          break;
        }
        int item_base5 = (int)tile5 * 8;
        int sequence5 = work_items[item_base5];
        int head5 = work_items[item_base5 + 1];
        int write_start5 = work_items[item_base5 + 2];
        int write_end5 = work_items[item_base5 + 3];
        int compute_end5 = work_items[item_base5 + 5];
        long long bos5 = cu_seqlens[sequence5];
        float a_sum5[8];
        a_sum5[0] = 0.0f;
        a_sum5[1] = 0.0f;
        a_sum5[2] = 0.0f;
        a_sum5[3] = 0.0f;
        a_sum5[4] = 0.0f;
        a_sum5[5] = 0.0f;
        a_sum5[6] = 0.0f;
        a_sum5[7] = 0.0f;
#pragma unroll 1
        for (int reverse5 = 0; reverse5 < compute_end5 - write_start5; reverse5++) {
          mbarrier_wait(q_decay_k_restore_ready_addr + (operand_stage5) * 8,
                        _phase_q_decay_k_restore_ready_1);
          mbarrier_wait(raw_ready_addr + (raw_stage5) * 8, _phase_raw_ready_4);
          int dv_chunk5 = compute_end5 - 1 - reverse5;
          int dv_token5 = (int)(bos5 + (long long)dv_chunk5 * 16);
          mbarrier_wait(intermediate_done_addr + (intermediate_stage5) * 8,
                        _phase_intermediate_done_1);
          float a_acc5[8];
#pragma unroll
          for (int k_block5 = 0; k_block5 < 8; k_block5++) {
            int a_col5 = k_block5 * 16 + lhs_col_offset5;
            int b_col5 = k_block5 * 16 + rhs_col_offset5;
            unsigned int a_frag5[4];
            unsigned int b_frag5[4];
            int segment_11 = a_col5 / 64;
            int segment_col_12 = a_col5 - segment_11 * 64;
            int swizzled_col_12 = segment_col_12 ^ (lhs_row5 & 7) * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(a_frag5[0]), "=r"(a_frag5[1]), "=r"(a_frag5[2]), "=r"(a_frag5[3])
                         : "r"(q_decay_all_addr + (unsigned int)(((int)operand_stage5 * 16 * 128 +
                                                                  segment_11 * 16 * 64 +
                                                                  lhs_row5 * 64 + swizzled_col_12) *
                                                                 2))
                         : "memory");
            int segment_0_5 = b_col5 / 64;
            int segment_col_1_5 = b_col5 - segment_0_5 * 64;
            int swizzled_col_2_5 = segment_col_1_5 ^ (rhs_row5 & 7) * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(b_frag5[0]), "=r"(b_frag5[1]), "=r"(b_frag5[2]), "=r"(b_frag5[3])
                         : "r"(k_inv_all_addr + (unsigned int)(((int)operand_stage5 * 16 * 128 +
                                                                segment_0_5 * 16 * 64 +
                                                                rhs_row5 * 64 + swizzled_col_2_5) *
                                                               2))
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(a_acc5[0]), "=f"(a_acc5[1]), "=f"(a_acc5[2]), "=f"(a_acc5[3])
                : "r"(a_frag5[0]), "r"(a_frag5[1]), "r"(a_frag5[2]), "r"(a_frag5[3]),
                  "r"(b_frag5[0]), "r"(b_frag5[1]), "f"(((k_block5 == 0) ? 0.0f : a_acc5[0])),
                  "f"(((k_block5 == 0) ? 0.0f : a_acc5[1])),
                  "f"(((k_block5 == 0) ? 0.0f : a_acc5[2])),
                  "f"(((k_block5 == 0) ? 0.0f : a_acc5[3])));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(a_acc5[4]), "=f"(a_acc5[(4) + 1]), "=f"(a_acc5[(4) + 2]),
                  "=f"(a_acc5[(4) + 3])
                : "r"(a_frag5[0]), "r"(a_frag5[1]), "r"(a_frag5[2]), "r"(a_frag5[3]),
                  "r"(b_frag5[2]), "r"(b_frag5[(2) + 1]), "f"(((k_block5 == 0) ? 0.0f : a_acc5[4])),
                  "f"(((k_block5 == 0) ? 0.0f : a_acc5[(4) + 1])),
                  "f"(((k_block5 == 0) ? 0.0f : a_acc5[(4) + 2])),
                  "f"(((k_block5 == 0) ? 0.0f : a_acc5[(4) + 3])));
          }
          float a_values5[8];
#pragma unroll
          for (int accum5 = 0; accum5 < 8; accum5++) {
            int accum_row5 = lane / 4 + accum5 % 4 / 2 * 8;
            int accum_col5 = accum5 / 4 * 8 + (lane & 3) * 2 + (accum5 & 1);
            a_values5[accum5] = 0.0f;
            if (accum_row5 >= accum_col5) {
              a_values5[accum5] = a_acc5[accum5];
            }
            a_sum5[accum5] = a_sum5[accum5] + a_values5[accum5];
          }
          unsigned int a_words5[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(a_values5[_lp * 2 + 0], a_values5[_lp * 2 + 1 + 0]));
            a_words5[_lp] = *(uint32_t*)&_bf2;
          }
          int publish_row5 = lane % 16;
          int publish_col5 = lane / 16 * 8;
          uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(
              (unsigned long long)(intermediate_a_addr + intermediate_stage5 * 2560 +
                                   (unsigned int)(publish_col5 / 16 * 512 + publish_row5 * 32 +
                                                      publish_col5 % 16 * 2 ^
                                                  (publish_col5 / 16 * 512 + publish_row5 * 32 +
                                                           publish_col5 % 16 * 2 >>
                                                       7 &
                                                   1) << 4)));
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                           _stmatrix_addr_0),
                       "r"(*reinterpret_cast<const uint32_t*>(&a_words5[0])),
                       "r"(*reinterpret_cast<const uint32_t*>(&a_words5[1])),
                       "r"(*reinterpret_cast<const uint32_t*>(&a_words5[2])),
                       "r"(*reinterpret_cast<const uint32_t*>(&a_words5[3]))
                       : "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(a_ready_addr + (intermediate_stage5) * 8);
          mbarrier_wait(beta_dy_smem_ready_addr + (dv_stage5) * 8, _phase_beta_dy_smem_ready_1);
          if (dv_chunk5 < write_end5) {
            if (elect_sync()) {
#pragma unroll
              for (int dv_segment5 = 0; dv_segment5 < 2; dv_segment5++) {
                tma_store_3d((&dv_tma), dv_segment5 * 64, head5, dv_token5,
                             beta_dy_smem_addr + dv_stage5 * 4096 +
                                 (unsigned int)(dv_segment5 * 16 * 64 * 2));
              }
            }
          }
          asm volatile("cp.async.bulk.commit_group;");
          mbarrier_wait(u_smem_ready_addr + (u_smem_stage5) * 8, _phase_u_smem_ready_2);
          float da_acc5[8];
#pragma unroll
          for (int k_block_da5 = 0; k_block_da5 < 8; k_block_da5++) {
            int a_col_da5 = k_block_da5 * 16 + lhs_col_offset5;
            int b_col_da5 = k_block_da5 * 16 + rhs_col_offset5;
            unsigned int do_frag5[4];
            unsigned int u_frag5[4];
            int segment_13 = a_col_da5 / 64;
            int segment_col_14 = a_col_da5 - segment_13 * 64;
            int swizzled_col_13 = segment_col_14 ^ (lhs_row5 & 7) * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(do_frag5[0]), "=r"(do_frag5[1]), "=r"(do_frag5[2]),
                           "=r"(do_frag5[3])
                         : "r"(raw_do_all_addr +
                               (unsigned int)(((int)raw_stage5 * 16 * 128 + segment_13 * 16 * 64 +
                                               lhs_row5 * 64 + swizzled_col_13) *
                                              2))
                         : "memory");
            int segment_0_6 = b_col_da5 / 64;
            int segment_col_1_6 = b_col_da5 - segment_0_6 * 64;
            int swizzled_col_2_6 = segment_col_1_6 ^ (rhs_row5 & 7) * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(u_frag5[0]), "=r"(u_frag5[1]), "=r"(u_frag5[2]), "=r"(u_frag5[3])
                         : "r"(u_smem_addr + (unsigned int)((segment_0_6 * 16 * 64 + rhs_row5 * 64 +
                                                             swizzled_col_2_6) *
                                                            2))
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(da_acc5[0]), "=f"(da_acc5[1]), "=f"(da_acc5[2]), "=f"(da_acc5[3])
                : "r"(do_frag5[0]), "r"(do_frag5[1]), "r"(do_frag5[2]), "r"(do_frag5[3]),
                  "r"(u_frag5[0]), "r"(u_frag5[1]), "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[0])),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[1])),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[2])),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[3])));
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, "
                "%6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(da_acc5[4]), "=f"(da_acc5[(4) + 1]), "=f"(da_acc5[(4) + 2]),
                  "=f"(da_acc5[(4) + 3])
                : "r"(do_frag5[0]), "r"(do_frag5[1]), "r"(do_frag5[2]), "r"(do_frag5[3]),
                  "r"(u_frag5[2]), "r"(u_frag5[(2) + 1]),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[4])),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[(4) + 1])),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[(4) + 2])),
                  "f"(((k_block_da5 == 0) ? 0.0f : da_acc5[(4) + 3])));
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(raw_done_addr + (raw_stage5) * 8);
          float da_values5[8];
#pragma unroll
          for (int accum_da5 = 0; accum_da5 < 8; accum_da5++) {
            int da_row5 = lane / 4 + accum_da5 % 4 / 2 * 8;
            int da_col5 = accum_da5 / 4 * 8 + (lane & 3) * 2 + (accum_da5 & 1);
            da_values5[accum_da5] = 0.0f;
            if (da_row5 >= da_col5) {
              da_values5[accum_da5] = da_acc5[accum_da5];
            }
          }
          unsigned int da_words5[4];
#pragma unroll
          for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(da_values5[_lp * 2 + 0], da_values5[_lp * 2 + 1 + 0]));
            da_words5[_lp] = *(uint32_t*)&_bf2;
          }
          uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(
              (unsigned long long)(intermediate_da_addr + intermediate_stage5 * 2560 +
                                   (unsigned int)(publish_col5 / 16 * 512 + publish_row5 * 32 +
                                                      publish_col5 % 16 * 2 ^
                                                  (publish_col5 / 16 * 512 + publish_row5 * 32 +
                                                           publish_col5 % 16 * 2 >>
                                                       7 &
                                                   1) << 4)));
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                           _stmatrix_addr_1),
                       "r"(*reinterpret_cast<const uint32_t*>(&da_words5[0])),
                       "r"(*reinterpret_cast<const uint32_t*>(&da_words5[1])),
                       "r"(*reinterpret_cast<const uint32_t*>(&da_words5[2])),
                       "r"(*reinterpret_cast<const uint32_t*>(&da_words5[3]))
                       : "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(da_ready_addr + (intermediate_stage5) * 8);
          asm volatile("cp.async.bulk.wait_group 0;");
          if (elect_sync()) {
            mbarrier_arrive(beta_dy_smem_done_addr + (dv_stage5) * 8);
          }
          dv_stage5 += 1;
          if (dv_stage5 == 2) {
            dv_stage5 = 0;
            _phase_beta_dy_smem_ready_1 ^= 1;
          }
          raw_stage5 += 1;
          if (raw_stage5 == 2) {
            raw_stage5 = 0;
            _phase_raw_ready_4 ^= 1;
          }
          operand_stage5 += 1;
          if (operand_stage5 == 2) {
            operand_stage5 = 0;
            _phase_q_decay_k_restore_ready_1 ^= 1;
          }
          intermediate_stage5 += 1;
          if (intermediate_stage5 == 2) {
            intermediate_stage5 = 0;
            _phase_intermediate_done_1 ^= 1;
          }
          u_smem_stage5 += 1;
          if (u_smem_stage5 == 1) {
            u_smem_stage5 = 0;
            _phase_u_smem_ready_2 ^= 1;
          }
        }
      }
      if (elect_sync()) {
        mbarrier_arrive(consumers_done_addr);
      }
    }
  }

  // Cleanup
}

}  // extern "C"

#undef LOOM_INF
#undef NUM_BETA_DY_SMEM_PIPE_STAGES
#undef NUM_BOUNDARY_LOCAL_GRAD_PIPE_STAGES
#undef NUM_BOUNDARY_PIPE_STAGES
#undef NUM_DBETA_M_PIPE_STAGES
#undef NUM_DK_RESTORE_PIPE_STAGES
#undef NUM_DSTATE_RECURRENCE_PIPE_STAGES
#undef NUM_DSTATE_SMEM_PIPE_STAGES
#undef NUM_DY_SMEM_PIPE_STAGES
#undef NUM_G_PREFIX_PIPE_STAGES
#undef NUM_INTERMEDIATE_PIPE_STAGES
#undef NUM_LOCAL_GRAD_PIPE_STAGES
#undef NUM_OPERAND_PIPE_STAGES
#undef NUM_QK_RAW_PIPE_STAGES
#undef NUM_RAW_PIPE_STAGES
#undef NUM_SCHED_PIPE_STAGES
#undef NUM_STATE_SMEM_PIPE_STAGES
#undef NUM_TCGEN_DATA_PIPE_STAGES
#undef NUM_U_SMEM_PIPE_STAGES
#undef SMEM_BETA_DY_SMEM_OFF
#undef SMEM_BETA_DY_SMEM_STAGE_BYTES
#undef SMEM_BETA_DY_SMEM_STRIDE
#undef SMEM_BETA_SMEM_ALL_OFF
#undef SMEM_BETA_SMEM_ALL_STAGE_BYTES
#undef SMEM_BETA_SMEM_ALL_STRIDE
#undef SMEM_BETA_SMEM_OFF
#undef SMEM_BETA_SMEM_STAGE_BYTES
#undef SMEM_BETA_SMEM_STRIDE
#undef SMEM_BOUNDARY_STATE_SMEM_OFF
#undef SMEM_BOUNDARY_STATE_SMEM_STAGE_BYTES
#undef SMEM_BOUNDARY_STATE_SMEM_STRIDE
#undef SMEM_DBETA_M_SMEM_OFF
#undef SMEM_DBETA_M_SMEM_STAGE_BYTES
#undef SMEM_DBETA_M_SMEM_STRIDE
#undef SMEM_DBETA_RED_SMEM_OFF
#undef SMEM_DBETA_RED_SMEM_STAGE_BYTES
#undef SMEM_DBETA_RED_SMEM_STRIDE
#undef SMEM_DEBUG_DU_SMEM_ALL_OFF
#undef SMEM_DEBUG_DU_SMEM_ALL_STAGE_BYTES
#undef SMEM_DEBUG_DU_SMEM_ALL_STRIDE
#undef SMEM_DEBUG_DU_SMEM_OFF
#undef SMEM_DEBUG_DU_SMEM_STAGE_BYTES
#undef SMEM_DEBUG_DU_SMEM_STRIDE
#undef SMEM_DSTATE_SMEM_ALL_OFF
#undef SMEM_DSTATE_SMEM_ALL_STAGE_BYTES
#undef SMEM_DSTATE_SMEM_ALL_STRIDE
#undef SMEM_DSTATE_SMEM_MN_OFF
#undef SMEM_DSTATE_SMEM_MN_STAGE_BYTES
#undef SMEM_DSTATE_SMEM_MN_STRIDE
#undef SMEM_DSTATE_SMEM_OFF
#undef SMEM_DSTATE_SMEM_STAGE_BYTES
#undef SMEM_DSTATE_SMEM_STRIDE
#undef SMEM_DY_SMEM_ALL_OFF
#undef SMEM_DY_SMEM_ALL_STAGE_BYTES
#undef SMEM_DY_SMEM_ALL_STRIDE
#undef SMEM_DY_SMEM_OFF
#undef SMEM_DY_SMEM_STAGE_BYTES
#undef SMEM_DY_SMEM_STRIDE
#undef SMEM_G_PREFIX_ALL_OFF
#undef SMEM_G_PREFIX_ALL_STAGE_BYTES
#undef SMEM_G_PREFIX_ALL_STRIDE
#undef SMEM_G_PREFIX_OFF
#undef SMEM_G_PREFIX_STAGE_BYTES
#undef SMEM_G_PREFIX_STRIDE
#undef SMEM_INTERMEDIATE_A_MN_OFF
#undef SMEM_INTERMEDIATE_A_MN_STAGE_BYTES
#undef SMEM_INTERMEDIATE_A_MN_STRIDE
#undef SMEM_INTERMEDIATE_A_OFF
#undef SMEM_INTERMEDIATE_A_STAGE_BYTES
#undef SMEM_INTERMEDIATE_A_STRIDE
#undef SMEM_INTERMEDIATE_DA_MN_OFF
#undef SMEM_INTERMEDIATE_DA_MN_STAGE_BYTES
#undef SMEM_INTERMEDIATE_DA_MN_STRIDE
#undef SMEM_INTERMEDIATE_DA_OFF
#undef SMEM_INTERMEDIATE_DA_STAGE_BYTES
#undef SMEM_INTERMEDIATE_DA_STRIDE
#undef SMEM_INTERMEDIATE_DM_OFF
#undef SMEM_INTERMEDIATE_DM_STAGE_BYTES
#undef SMEM_INTERMEDIATE_DM_STRIDE
#undef SMEM_INTERMEDIATE_NDM_MN_OFF
#undef SMEM_INTERMEDIATE_NDM_MN_STAGE_BYTES
#undef SMEM_INTERMEDIATE_NDM_MN_STRIDE
#undef SMEM_INTERMEDIATE_NDM_OFF
#undef SMEM_INTERMEDIATE_NDM_STAGE_BYTES
#undef SMEM_INTERMEDIATE_NDM_STRIDE
#undef SMEM_INTERMEDIATE_TINV_MN_OFF
#undef SMEM_INTERMEDIATE_TINV_MN_STAGE_BYTES
#undef SMEM_INTERMEDIATE_TINV_MN_STRIDE
#undef SMEM_INTERMEDIATE_TINV_OFF
#undef SMEM_INTERMEDIATE_TINV_STAGE_BYTES
#undef SMEM_INTERMEDIATE_TINV_STRIDE
#undef SMEM_K_DECAY_ALL_OFF
#undef SMEM_K_DECAY_ALL_STAGE_BYTES
#undef SMEM_K_DECAY_ALL_STRIDE
#undef SMEM_K_DECAY_LEAD16_OFF
#undef SMEM_K_DECAY_LEAD16_STAGE_BYTES
#undef SMEM_K_DECAY_LEAD16_STRIDE
#undef SMEM_K_DECAY_OPERAND_OFF
#undef SMEM_K_DECAY_OPERAND_STAGE_BYTES
#undef SMEM_K_DECAY_OPERAND_STRIDE
#undef SMEM_K_DECAY_TRANS_OFF
#undef SMEM_K_DECAY_TRANS_STAGE_BYTES
#undef SMEM_K_DECAY_TRANS_STRIDE
#undef SMEM_K_INV_ALL_OFF
#undef SMEM_K_INV_ALL_STAGE_BYTES
#undef SMEM_K_INV_ALL_STRIDE
#undef SMEM_K_INV_AMAJ_OFF
#undef SMEM_K_INV_AMAJ_STAGE_BYTES
#undef SMEM_K_INV_AMAJ_STRIDE
#undef SMEM_K_INV_LEAD16_OFF
#undef SMEM_K_INV_LEAD16_STAGE_BYTES
#undef SMEM_K_INV_LEAD16_STRIDE
#undef SMEM_K_INV_OPERAND_OFF
#undef SMEM_K_INV_OPERAND_STAGE_BYTES
#undef SMEM_K_INV_OPERAND_STRIDE
#undef SMEM_K_RESTORE_ALL_OFF
#undef SMEM_K_RESTORE_ALL_STAGE_BYTES
#undef SMEM_K_RESTORE_ALL_STRIDE
#undef SMEM_K_RESTORE_LEAD16_OFF
#undef SMEM_K_RESTORE_LEAD16_STAGE_BYTES
#undef SMEM_K_RESTORE_LEAD16_STRIDE
#undef SMEM_K_RESTORE_OPERAND_OFF
#undef SMEM_K_RESTORE_OPERAND_STAGE_BYTES
#undef SMEM_K_RESTORE_OPERAND_STRIDE
#undef SMEM_QK_NORM_SMEM_ALL_OFF
#undef SMEM_QK_NORM_SMEM_ALL_STAGE_BYTES
#undef SMEM_QK_NORM_SMEM_ALL_STRIDE
#undef SMEM_QK_NORM_SMEM_OFF
#undef SMEM_QK_NORM_SMEM_STAGE_BYTES
#undef SMEM_QK_NORM_SMEM_STRIDE
#undef SMEM_QK_RED_SMEM_OFF
#undef SMEM_QK_RED_SMEM_STAGE_BYTES
#undef SMEM_QK_RED_SMEM_STRIDE
#undef SMEM_Q_DECAY_ALL_OFF
#undef SMEM_Q_DECAY_ALL_STAGE_BYTES
#undef SMEM_Q_DECAY_ALL_STRIDE
#undef SMEM_Q_DECAY_OPERAND_OFF
#undef SMEM_Q_DECAY_OPERAND_STAGE_BYTES
#undef SMEM_Q_DECAY_OPERAND_STRIDE
#undef SMEM_Q_DECAY_TRANS_OFF
#undef SMEM_Q_DECAY_TRANS_STAGE_BYTES
#undef SMEM_Q_DECAY_TRANS_STRIDE
#undef SMEM_RAW_DO_ALL_OFF
#undef SMEM_RAW_DO_ALL_STAGE_BYTES
#undef SMEM_RAW_DO_ALL_STRIDE
#undef SMEM_RAW_DO_AMAJ_OFF
#undef SMEM_RAW_DO_AMAJ_STAGE_BYTES
#undef SMEM_RAW_DO_AMAJ_STRIDE
#undef SMEM_RAW_DO_OFF
#undef SMEM_RAW_DO_STAGE_BYTES
#undef SMEM_RAW_DO_STRIDE
#undef SMEM_RAW_G_ALL_OFF
#undef SMEM_RAW_G_ALL_STAGE_BYTES
#undef SMEM_RAW_G_ALL_STRIDE
#undef SMEM_RAW_G_OFF
#undef SMEM_RAW_G_STAGE_BYTES
#undef SMEM_RAW_G_STRIDE
#undef SMEM_RAW_K_ALL_OFF
#undef SMEM_RAW_K_ALL_STAGE_BYTES
#undef SMEM_RAW_K_ALL_STRIDE
#undef SMEM_RAW_K_OFF
#undef SMEM_RAW_K_STAGE_BYTES
#undef SMEM_RAW_K_STRIDE
#undef SMEM_RAW_Q_ALL_OFF
#undef SMEM_RAW_Q_ALL_STAGE_BYTES
#undef SMEM_RAW_Q_ALL_STRIDE
#undef SMEM_RAW_Q_OFF
#undef SMEM_RAW_Q_STAGE_BYTES
#undef SMEM_RAW_Q_STRIDE
#undef SMEM_RAW_V_ALL_OFF
#undef SMEM_RAW_V_ALL_STAGE_BYTES
#undef SMEM_RAW_V_ALL_STRIDE
#undef SMEM_RAW_V_OFF
#undef SMEM_RAW_V_STAGE_BYTES
#undef SMEM_RAW_V_STRIDE
#undef SMEM_STATE_OPERAND_ALL_OFF
#undef SMEM_STATE_OPERAND_ALL_STAGE_BYTES
#undef SMEM_STATE_OPERAND_ALL_STRIDE
#undef SMEM_STATE_OPERAND_MN_OFF
#undef SMEM_STATE_OPERAND_MN_STAGE_BYTES
#undef SMEM_STATE_OPERAND_MN_STRIDE
#undef SMEM_STATE_OPERAND_OFF
#undef SMEM_STATE_OPERAND_STAGE_BYTES
#undef SMEM_STATE_OPERAND_STRIDE
#undef SMEM_STATE_PANEL_OFF
#undef SMEM_STATE_PANEL_STAGE_BYTES
#undef SMEM_STATE_PANEL_STRIDE
#undef SMEM_STATE_SCALE_DIAG_OFF
#undef SMEM_STATE_SCALE_DIAG_STAGE_BYTES
#undef SMEM_STATE_SCALE_DIAG_STRIDE
#undef SMEM_TINV_SCRATCH_OFF
#undef SMEM_TINV_SCRATCH_STAGE_BYTES
#undef SMEM_TINV_SCRATCH_STRIDE
#undef SMEM_TOTAL
#undef SMEM_U_LEAD16_OFF
#undef SMEM_U_LEAD16_STAGE_BYTES
#undef SMEM_U_LEAD16_STRIDE
#undef SMEM_U_SMEM_ALL_OFF
#undef SMEM_U_SMEM_ALL_STAGE_BYTES
#undef SMEM_U_SMEM_ALL_STRIDE
#undef SMEM_U_SMEM_OFF
#undef SMEM_U_SMEM_STAGE_BYTES
#undef SMEM_U_SMEM_STRIDE
#undef SMEM_WORK_ITEM_OFF
#undef SMEM_WORK_ITEM_STAGE_BYTES
#undef SMEM_WORK_ITEM_STRIDE
#undef THREADS
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DK_DECAY_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DK_INV_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DK_RESTORE_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DO_INP_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DQ_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DSTATE_INP_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DSTATE_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DU_INP_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DU_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_DY_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_ENVELOPE_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_NEG_DY_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_STATE_K_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_U_OFFSET
#undef TMEM_FLASHKDA_BWD_PERSISTENT_C16_Y_OFFSET
#undef TMEM_NCOLS
#undef USE_DSTATE_IN
#undef a_ready_addr
#undef beta_dy_smem_addr
#undef beta_dy_smem_done_addr
#undef beta_dy_smem_ready_addr
#undef beta_smem_addr
#undef beta_smem_all_addr
#undef boundary_acc_ready_addr
#undef boundary_local_grad_free_addr
#undef boundary_smem_ready_addr
#undef boundary_state_ready_addr
#undef boundary_state_smem_addr
#undef cleanup_ready_addr
#undef consumers_done_addr
#undef da_ready_addr
#undef dbeta_m_done_addr
#undef dbeta_m_ready_addr
#undef dbeta_m_smem_addr
#undef dbeta_red_smem_addr
#undef debug_du_smem_addr
#undef debug_du_smem_all_addr
#undef decay_done_addr
#undef dk_restore_done_addr
#undef dk_restore_ready_addr
#undef dm_ready_addr
#undef dstate_done_addr
#undef dstate_inp_ready_addr
#undef dstate_ready_addr
#undef dstate_smem_addr
#undef dstate_smem_all_addr
#undef dstate_smem_done_addr
#undef dstate_smem_mn_addr
#undef dstate_smem_ready_addr
#undef du_inp_ready_addr
#undef dy_done_addr
#undef dy_ready_addr
#undef dy_smem_addr
#undef dy_smem_all_addr
#undef dy_smem_ready_addr
#undef g_prefix_addr
#undef g_prefix_all_addr
#undef g_prefix_done_addr
#undef g_prefix_ready_addr
#undef intermediate_a_addr
#undef intermediate_a_mn_addr
#undef intermediate_da_addr
#undef intermediate_da_mn_addr
#undef intermediate_dm_addr
#undef intermediate_done_addr
#undef intermediate_ndm_addr
#undef intermediate_ndm_mn_addr
#undef intermediate_tinv_addr
#undef intermediate_tinv_mn_addr
#undef k_decay_all_addr
#undef k_decay_inv_ready_addr
#undef k_decay_lead16_addr
#undef k_decay_operand_addr
#undef k_decay_trans_addr
#undef k_inv_all_addr
#undef k_inv_amaj_addr
#undef k_inv_lead16_addr
#undef k_inv_operand_addr
#undef k_restore_all_addr
#undef k_restore_lead16_addr
#undef k_restore_operand_addr
#undef local_grad_done_addr
#undef local_grad_ready_addr
#undef neg_dy_ready_addr
#undef q_decay_all_addr
#undef q_decay_k_restore_ready_addr
#undef q_decay_operand_addr
#undef q_decay_trans_addr
#undef qk_norm_smem_addr
#undef qk_norm_smem_all_addr
#undef qk_raw_done_addr
#undef qk_raw_ready_addr
#undef qk_red_smem_addr
#undef raw_do_addr
#undef raw_do_all_addr
#undef raw_do_amaj_addr
#undef raw_done_addr
#undef raw_g_addr
#undef raw_g_all_addr
#undef raw_k_addr
#undef raw_k_all_addr
#undef raw_q_addr
#undef raw_q_all_addr
#undef raw_ready_addr
#undef raw_v_addr
#undef raw_v_all_addr
#undef sched_done_addr
#undef sched_ready_addr
#undef state_cg2_done_addr
#undef state_k_done_addr
#undef state_k_ready_addr
#undef state_operand_addr
#undef state_operand_all_addr
#undef state_operand_mn_addr
#undef state_panel_addr
#undef state_ready_addr
#undef state_scale_diag_addr
#undef state_slot_done_addr
#undef tcgen_inputs_done_addr
#undef tcgen_inputs_ready_addr
#undef tcgen_products_done_addr
#undef tcgen_products_ready_addr
#undef tinv_ready_addr
#undef tinv_scratch_addr
#undef u_lead16_addr
#undef u_smem_addr
#undef u_smem_all_addr
#undef u_smem_ready_addr
#undef validate_outputs
#undef work_item_addr
