typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) LoomTensorMap {
  uint64_t opaque[16];
};
template <int N>
struct __align__(128) LoomTensorMapPack {
  LoomTensorMap maps[N];
};

typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;

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
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count));
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
      : "r"(mbar_addr), "r"(phase)
      : "memory");
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
      : "r"(mbar_addr), "r"(phase)
      : "memory");
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
      "}\n" ::"r"(mbar_addr),
      "r"(phase), "r"(ticks)
      : "memory");
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
      "}\n" ::"r"(mbar_addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
  if (token == 0) {
    mbarrier_wait(mbar_addr, phase);
  }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase,
                                                            uint32_t token) {
  if (token == 0) {
    mbarrier_wait_cluster(mbar_addr, phase);
  }
}

__device__ __forceinline__ void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc,
                                                uint32_t i_desc, int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}\n" ::"r"(taddr),
      "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3FFFFULL) >> 4ULL; }

__device__ __forceinline__ void mma_ss_step(int a_lo, int b_lo, int taddr, uint32_t i_desc,
                                            int enable_d, uint32_t a_dhi, uint32_t b_dhi) {
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
      "}\n" ::"r"(a_lo),
      "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}

__device__ __forceinline__ void mma_ts_step(int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
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
      "}\n" ::"r"(taddr_out),
      "r"(taddr_a), "r"(b_lo), "r"(b_dhi), "r"(i_desc), "r"(enable_d));
}

__device__ __forceinline__ void elect_commit(int mbar_addr) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%0];\n\t"
      "}\n" ::"r"(mbar_addr));
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32"
      " {%0, %1, %2, %3, %4, %5, %6, %7,"
      "  %8, %9, %10, %11, %12, %13, %14, %15,"
      "  %16, %17, %18, %19, %20, %21, %22, %23,"
      "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]), "=f"(dst[4]), "=f"(dst[5]),
        "=f"(dst[6]), "=f"(dst[7]), "=f"(dst[8]), "=f"(dst[9]), "=f"(dst[10]), "=f"(dst[11]),
        "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]), "=f"(dst[16]), "=f"(dst[17]),
        "=f"(dst[18]), "=f"(dst[19]), "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
        "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]), "=f"(dst[28]), "=f"(dst[29]),
        "=f"(dst[30]), "=f"(dst[31])
      : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32"
      " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
      "  %9, %10, %11, %12, %13, %14, %15, %16,"
      "  %17, %18, %19, %20, %21, %22, %23, %24,"
      "  %25, %26, %27, %28, %29, %30, %31, %32};" ::"r"(tmem_addr),
      "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]), "f"(src[4]), "f"(src[5]), "f"(src[6]),
      "f"(src[7]), "f"(src[8]), "f"(src[9]), "f"(src[10]), "f"(src[11]), "f"(src[12]), "f"(src[13]),
      "f"(src[14]), "f"(src[15]), "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
      "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]), "f"(src[24]), "f"(src[25]),
      "f"(src[26]), "f"(src[27]), "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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
      : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b), "l"(*(unsigned long long*)&c));
  *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
  asm("mul.rn.ftz.f32x2 %0, %0, %1;"
      : "+l"(*(unsigned long long*)a)
      : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
  asm("add.rn.ftz.f32x2 %0, %0, %1;"
      : "+l"(*(unsigned long long*)a)
      : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
  asm("sub.rn.ftz.f32x2 %0, %0, %1;"
      : "+l"(*(unsigned long long*)a)
      : "l"(*(unsigned long long*)&b));
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

__device__ __forceinline__ void fma_scale_x32(float* sv, const float2* scale2,
                                              const float2* neg_max2) {
  float2* sv_2 = reinterpret_cast<float2*>(sv);

#pragma unroll

  for (int j = 0; j < 16; j++) fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
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
  asm volatile(
      "{\n\t"
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
  return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tma_store_3d(const void* tmap, int x, int y, int z,
                                             unsigned smem_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
      " [%0, {%1, %2, %3}], [%4];" ::"l"(tmap),
      "r"(x), "r"(y), "r"(z), "r"(smem_addr)
      : "memory");
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%0];" ::"r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
  uint32_t result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(result) : "r"(val));
  return result;
}

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_STATE_OFFSET 0
#define TMEM_TMEM_Q_STATE_OFFSET 128
#define TMEM_TMEM_STATE_INP_OFFSET 192
#define TMEM_TMEM_CG0_SHARED_ACC_OFFSET 256
#define TMEM_TMEM_CG1_SHARED_ACC_OFFSET 384
#define TMEM_TMEM_SHARED_INP_OFFSET 448
#define NUM_K_PIPE_STAGES 3
#define NUM_Q_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define NUM_G_PIPE_STAGES 2
#define NUM_GATE_PIPE_STAGES 5
#define NUM_BETA_PIPE_STAGES 5
#define NUM_ONE_STAGE_STAGES 1
#define NUM_CG0_ACC_PIPE_STAGES 2
#define NUM_AINV_PIPE_STAGES 3
#define NUM_QK_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 33792
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_K_TRANS_MMA_OFF 33792
#define SMEM_SMEM_K_TRANS_MMA_STAGE_BYTES 16384
#define SMEM_SMEM_K_TRANS_MMA_STRIDE 16384
#define SMEM_SMEM_V_OFF 82944
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_G_OFF 115712
#define SMEM_SMEM_G_STAGE_BYTES 16384
#define SMEM_SMEM_G_STRIDE 16384
#define SMEM_SMEM_G_TRANS_MMA_OFF 115712
#define SMEM_SMEM_G_TRANS_MMA_STAGE_BYTES 16384
#define SMEM_SMEM_G_TRANS_MMA_STRIDE 16384
#define SMEM_SMEM_V_MMA_OFF 82944
#define SMEM_SMEM_V_MMA_STAGE_BYTES 16384
#define SMEM_SMEM_V_MMA_STRIDE 16384
#define SMEM_SMEM_AINV_OFF 148480
#define SMEM_SMEM_AINV_STAGE_BYTES 8192
#define SMEM_SMEM_AINV_STRIDE 8192
#define SMEM_SMEM_AINV_RM_OFF 148480
#define SMEM_SMEM_AINV_RM_STAGE_BYTES 8192
#define SMEM_SMEM_AINV_RM_STRIDE 8192
#define SMEM_SMEM_QK_OFF 173056
#define SMEM_SMEM_QK_STAGE_BYTES 8192
#define SMEM_SMEM_QK_STRIDE 8192
#define SMEM_SMEM_O_OFF 189440
#define SMEM_SMEM_O_STAGE_BYTES 16384
#define SMEM_SMEM_O_STRIDE 16384
#define SMEM_SMEM_G_TOTAL_OFF 222208
#define SMEM_SMEM_G_TOTAL_STAGE_BYTES 512
#define SMEM_SMEM_G_TOTAL_STRIDE 512
#define SMEM_SMEM_BETA_OFF 224768
#define SMEM_SMEM_BETA_STAGE_BYTES 256
#define SMEM_SMEM_BETA_STRIDE 256
#define SMEM_TOTAL 226048
#define THREADS 384
#define USE_INITIAL_STATE 1
#define STORE_FINAL_STATE 1
#define ENABLE_CHECKPOINTS 0
#define IS_GQA 0

extern "C" {

__global__ __launch_bounds__(384, 1) void kernel_flashkda_blackwell_prefill_fp32_state_initial(
    LoomTensorMap const* Q, LoomTensorMap const* K, LoomTensorMap const* V, LoomTensorMap const* G,
    LoomTensorMap const* O, __nv_bfloat16* __restrict__ beta, float* __restrict__ A_log,
    float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens,
    float* __restrict__ initial_state, float* __restrict__ output_state,
    float* __restrict__ checkpoint_state, int* __restrict__ cu_checkpoints,
    uint8_t* __restrict__ tensormap_workspace, int checkpoint_every_n_tokens, float scale,
    int num_seqs, int num_q_heads, int num_v_heads, int total_tiles, float lower_bound) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  if (tid == 0) {
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(Q))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(K))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(V))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(G))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"((uint64_t)(O))
                 : "memory");
  }
  __syncthreads();

  // Kernel setup ops
  __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
  const int smem_q_addr = smem + 1024;
  __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
  const int smem_k_addr = smem + 33792;
  __nv_bfloat16* smem_k_trans_mma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
  const int smem_k_trans_mma_addr = smem + 33792;
  __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
  const int smem_v_addr = smem + 82944;
  __nv_bfloat16* smem_g = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
  const int smem_g_addr = smem + 115712;
  __nv_bfloat16* smem_g_trans_mma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
  const int smem_g_trans_mma_addr = smem + 115712;
  __nv_bfloat16* smem_v_mma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
  const int smem_v_mma_addr = smem + 82944;
  __nv_bfloat16* smem_ainv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
  const int smem_ainv_addr = smem + 148480;
  __nv_bfloat16* smem_ainv_rm = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
  const int smem_ainv_rm_addr = smem + 148480;
  __nv_bfloat16* smem_qk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
  const int smem_qk_addr = smem + 173056;
  __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
  const int smem_o_addr = smem + 189440;
  float* smem_g_total = reinterpret_cast<float*>(smem_raw + 222208);
  const int smem_g_total_addr = smem + 222208;
  float* smem_beta = reinterpret_cast<float*>(smem_raw + 224768);
  const int smem_beta_addr = smem + 224768;

  // Mbarrier init (34 groups, 73 barriers)
  // Mbarriers at smem_raw[0..584)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {
      // --- pipeline 'k_pipe' ---
      // load_k_full: 3 barriers, init_count=1
      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      mbarrier_init(smem + 16, 1);
      // load_k_empty: 3 barriers, init_count=2
      mbarrier_init(smem + 24, 2);
      mbarrier_init(smem + 32, 2);
      mbarrier_init(smem + 40, 2);
      // --- pipeline 'q_pipe' ---
      // load_q_full: 2 barriers, init_count=1
      mbarrier_init(smem + 48, 1);
      mbarrier_init(smem + 56, 1);
      // --- pipeline 'v_pipe' ---
      // load_v_full: 2 barriers, init_count=1
      mbarrier_init(smem + 64, 1);
      mbarrier_init(smem + 72, 1);
      // --- pipeline 'g_pipe' ---
      // load_g_full: 2 barriers, init_count=1
      mbarrier_init(smem + 80, 1);
      mbarrier_init(smem + 88, 1);
      // load_g_empty: 2 barriers, init_count=1
      mbarrier_init(smem + 96, 1);
      mbarrier_init(smem + 104, 1);
      // qkg_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 112, 128);
      mbarrier_init(smem + 120, 128);
      // ki_mma_consumed: 2 barriers, init_count=1
      mbarrier_init(smem + 128, 1);
      mbarrier_init(smem + 136, 1);
      // kr_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 144, 128);
      mbarrier_init(smem + 152, 128);
      // --- pipeline 'gate_pipe' ---
      // load_gate_full: 5 barriers, init_count=128
      mbarrier_init(smem + 160, 128);
      mbarrier_init(smem + 168, 128);
      mbarrier_init(smem + 176, 128);
      mbarrier_init(smem + 184, 128);
      mbarrier_init(smem + 192, 128);
      // --- pipeline 'beta_pipe' ---
      // load_beta_full: 5 barriers, init_count=32
      mbarrier_init(smem + 200, 32);
      mbarrier_init(smem + 208, 32);
      mbarrier_init(smem + 216, 32);
      mbarrier_init(smem + 224, 32);
      mbarrier_init(smem + 232, 32);
      // q_state_acc_full: 1 barriers, init_count=1
      mbarrier_init(smem + 240, 1);
      // q_state_acc_empty: 1 barriers, init_count=128
      mbarrier_init(smem + 248, 128);
      // kv_acc_full: 1 barriers, init_count=1
      mbarrier_init(smem + 256, 1);
      // kv_acc_empty: 1 barriers, init_count=128
      mbarrier_init(smem + 264, 128);
      // initial_state_loaded: 1 barriers, init_count=4
      mbarrier_init(smem + 272, 4);
      // --- pipeline 'cg0_acc_pipe' ---
      // cg0_shared_acc_full: 2 barriers, init_count=1
      mbarrier_init(smem + 280, 1);
      mbarrier_init(smem + 288, 1);
      // cg0_shared_acc_empty: 2 barriers, init_count=128
      mbarrier_init(smem + 296, 128);
      mbarrier_init(smem + 304, 128);
      // --- pipeline 'one_stage' ---
      // cg1_shared_acc_full: 1 barriers, init_count=1
      mbarrier_init(smem + 312, 1);
      // cg1_shared_acc_empty: 1 barriers, init_count=128
      mbarrier_init(smem + 320, 128);
      // --- pipeline 'ainv_pipe' ---
      // ainv_ready: 3 barriers, init_count=128
      mbarrier_init(smem + 328, 128);
      mbarrier_init(smem + 336, 128);
      mbarrier_init(smem + 344, 128);
      // --- pipeline 'qk_pipe' ---
      // qk_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 352, 128);
      mbarrier_init(smem + 360, 128);
      // state_inp_ready: 1 barriers, init_count=128
      mbarrier_init(smem + 368, 128);
      // vks_ready: 1 barriers, init_count=128
      mbarrier_init(smem + 376, 128);
      // nv_ready: 1 barriers, init_count=128
      mbarrier_init(smem + 384, 128);
      // decay_v_ready: 1 barriers, init_count=128
      mbarrier_init(smem + 392, 128);
      // --- pipeline 'o_pipe' ---
      // o_store_ready: 2 barriers, init_count=128
      mbarrier_init(smem + 400, 128);
      mbarrier_init(smem + 408, 128);
      // --- pipeline 'gate_pipe' ---
      // gate_cg1_empty: 5 barriers, init_count=128
      mbarrier_init(smem + 416, 128);
      mbarrier_init(smem + 424, 128);
      mbarrier_init(smem + 432, 128);
      mbarrier_init(smem + 440, 128);
      mbarrier_init(smem + 448, 128);
      // --- pipeline 'beta_pipe' ---
      // beta_smem_empty: 5 barriers, init_count=128
      mbarrier_init(smem + 456, 128);
      mbarrier_init(smem + 464, 128);
      mbarrier_init(smem + 472, 128);
      mbarrier_init(smem + 480, 128);
      mbarrier_init(smem + 488, 128);
      // --- pipeline 'qk_pipe' ---
      // qk_smem_empty: 2 barriers, init_count=1
      mbarrier_init(smem + 496, 1);
      mbarrier_init(smem + 504, 1);
      // --- pipeline 'ainv_pipe' ---
      // ainv_smem_empty: 3 barriers, init_count=1
      mbarrier_init(smem + 512, 1);
      mbarrier_init(smem + 520, 1);
      mbarrier_init(smem + 528, 1);
      // --- pipeline 'q_pipe' ---
      // q_smem_empty: 2 barriers, init_count=2
      mbarrier_init(smem + 536, 2);
      mbarrier_init(smem + 544, 2);
      // --- pipeline 'v_pipe' ---
      // v_smem_empty: 2 barriers, init_count=4
      mbarrier_init(smem + 552, 4);
      mbarrier_init(smem + 560, 4);
      // --- pipeline 'o_pipe' ---
      // o_smem_empty: 2 barriers, init_count=32
      mbarrier_init(smem + 568, 32);
      mbarrier_init(smem + 576, 32);
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
  }

  __syncwarp();

  // TMEM alloc (512 columns, 512 used)
  volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 584);
  if (warp == 4) {
    int _tmem_hold = smem + 584;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(_tmem_hold),
        "r"(512)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");

  const int mbar_base = smem;
#define load_k_full_addr (mbar_base + 0)
#define load_k_empty_addr (mbar_base + 24)
#define load_q_full_addr (mbar_base + 48)
#define load_v_full_addr (mbar_base + 64)
#define load_g_full_addr (mbar_base + 80)
#define load_g_empty_addr (mbar_base + 96)
#define qkg_ready_addr (mbar_base + 112)
#define ki_mma_consumed_addr (mbar_base + 128)
#define kr_ready_addr (mbar_base + 144)
#define load_gate_full_addr (mbar_base + 160)
#define load_beta_full_addr (mbar_base + 200)
#define q_state_acc_full_addr (mbar_base + 240)
#define q_state_acc_empty_addr (mbar_base + 248)
#define kv_acc_full_addr (mbar_base + 256)
#define kv_acc_empty_addr (mbar_base + 264)
#define initial_state_loaded_addr (mbar_base + 272)
#define cg0_shared_acc_full_addr (mbar_base + 280)
#define cg0_shared_acc_empty_addr (mbar_base + 296)
#define cg1_shared_acc_full_addr (mbar_base + 312)
#define cg1_shared_acc_empty_addr (mbar_base + 320)
#define ainv_ready_addr (mbar_base + 328)
#define qk_ready_addr (mbar_base + 352)
#define state_inp_ready_addr (mbar_base + 368)
#define vks_ready_addr (mbar_base + 376)
#define nv_ready_addr (mbar_base + 384)
#define decay_v_ready_addr (mbar_base + 392)
#define o_store_ready_addr (mbar_base + 400)
#define gate_cg1_empty_addr (mbar_base + 416)
#define beta_smem_empty_addr (mbar_base + 456)
#define qk_smem_empty_addr (mbar_base + 496)
#define ainv_smem_empty_addr (mbar_base + 512)
#define q_smem_empty_addr (mbar_base + 536)
#define v_smem_empty_addr (mbar_base + 552)
#define o_smem_empty_addr (mbar_base + 568)
  const int taddr = tmem_addr_storage[0];

  // Kernel post-init ops
  const int tmem_tmem_state = taddr;
  const int tmem_tmem_q_state = taddr + 128;
  const int tmem_tmem_state_inp = taddr + 192;
  const int tmem_tmem_cg0_shared_acc = taddr + 256;
  const int tmem_tmem_cg1_shared_acc = taddr + 384;
  const int tmem_tmem_shared_inp = taddr + 448;

  // ---- Register redistribution for WGs split across roles ----
  // Dec phase frees registers before any WG attempts inc.
  if (warp >= 8 && warp <= 11) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
  }

  // ---- Role: compute_group_0 ----
  if (warp <= 3) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
    {  // compute_group_0_main
      unsigned int k_pre_stage = 0;
      unsigned int k_pre_phase = 0;
      unsigned int q_pre_stage = 0;
      unsigned int q_pre_phase = 0;
      unsigned int g_pre_stage = 0;
      unsigned int g_pre_phase = 0;
      unsigned int g_kr_stage = 0;
      unsigned int g_kr_phase = 0;
      unsigned int gate_cg0_stage = 0;
      unsigned int gate_cg0_phase = 1;
      unsigned int beta_cg0_stage = 0;
      unsigned int beta_cg0_phase = 0;
      unsigned int ainv_cg0_stage = 0;
      unsigned int ainv_cg0_phase = 1;
      unsigned int qk_cg0_stage = 0;
      unsigned int qk_cg0_phase = 1;
#pragma unroll 1
      for (unsigned int tile = bid; tile < total_tiles; tile += num_bids) {
        int num_o_heads = ((num_q_heads >= num_v_heads) ? num_q_heads : num_v_heads);
        int batch_idx = tile / (unsigned int)num_o_heads;
        int head_idx = tile % (unsigned int)num_o_heads;
        int qk_head_idx =
            ((num_q_heads >= num_v_heads) ? head_idx : head_idx / (num_v_heads / num_q_heads));
        int v_head_idx =
            ((num_v_heads >= num_q_heads) ? head_idx : head_idx / (num_q_heads / num_v_heads));
        int batch_start = (int)cu_seqlens[batch_idx];
        int batch_end = (int)cu_seqlens[batch_idx + 1];
        int seqlen_b = batch_end - batch_start;
        int num_pairs_b = (seqlen_b + 32 - 1) / 32;
        int num_chunks_b = num_pairs_b * 2;
#pragma unroll 1
        for (int chunk_idx = 0; chunk_idx < num_chunks_b; chunk_idx += 2) {
          int chunk_offset = batch_start + chunk_idx * 16;
          int _cg0_marker = batch_idx + head_idx + chunk_offset + batch_end;
          int warp_id_in_role = (warp - 0);
          int warp_id_in_role_cg0 = warp_id_in_role;
          int row_cg0 = warp_id_in_role_cg0 * 32 + lane;
          int lane_quad_cg0 = lane & 3;
          int lane_row_cg0 = lane / 4;
          int qk_warp_row_base_cg0 = warp_id_in_role_cg0 * 16;
          int qk_tmem_row_base_cg0 = warp_id_in_role_cg0 * 32 << 16;
          unsigned int k0_stage = k_pre_stage;
          mbarrier_wait(load_k_full_addr + (k0_stage) * 8, k_pre_phase);
          k_pre_stage += 1;
          if (k_pre_stage == 3) {
            k_pre_stage = 0;
            k_pre_phase ^= 1;
          }
          unsigned int q0_stage = q_pre_stage;
          mbarrier_wait(load_q_full_addr + (q0_stage) * 8, q_pre_phase);
          q_pre_stage += 1;
          if (q_pre_stage == 2) {
            q_pre_stage = 0;
            q_pre_phase ^= 1;
          }
          unsigned int g0_stage = g_pre_stage;
          mbarrier_wait(load_g_full_addr + (g0_stage) * 8, g_pre_phase);
          g_pre_stage += 1;
          if (g_pre_stage == 2) {
            g_pre_stage = 0;
            g_pre_phase ^= 1;
          }
          unsigned int gate0_stage = gate_cg0_stage;
          mbarrier_wait(gate_cg1_empty_addr + (gate0_stage) * 8, gate_cg0_phase);
          gate_cg0_stage += 1;
          if (gate_cg0_stage == 5) {
            gate_cg0_stage = 0;
            gate_cg0_phase ^= 1;
          }
          int norm_row = tid / 8;
          int norm_lane = tid & 7;
          int norm_col_base = norm_lane * 16;
          float q_sum = 0.0f;
          float k_sum = 0.0f;
#pragma unroll
          for (int norm_j = 0; norm_j < 16; norm_j++) {
            int norm_col = norm_col_base + norm_j;
            float q_raw = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_q) +
                                                  (q0_stage * 16384)) +
                 (norm_col / 64 * 8192 + norm_row * 128 + norm_col % 64 * 2 ^
                  (norm_col / 64 * 8192 + norm_row * 128 + norm_col % 64 * 2 >> 7 & 7) << 4)))[0];
            float k_raw = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_k) +
                                                  (k0_stage * 16384)) +
                 (norm_col / 64 * 8192 + norm_row * 128 + norm_col % 64 * 2 ^
                  (norm_col / 64 * 8192 + norm_row * 128 + norm_col % 64 * 2 >> 7 & 7) << 4)))[0];
            float _fma_0 = __fmaf_rn(q_raw, q_raw, q_sum);
            q_sum = _fma_0;
            float _fma_1 = __fmaf_rn(k_raw, k_raw, k_sum);
            k_sum = _fma_1;
          }
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
          q_sum += _shfl_xor_0;
          float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
          k_sum += _shfl_xor_1;
          float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
          q_sum += _shfl_xor_2;
          float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
          k_sum += _shfl_xor_3;
          float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
          q_sum += _shfl_xor_4;
          float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
          k_sum += _shfl_xor_5;
          float _rsqrt_0 = rsqrtf(q_sum + 1e-06f);
          float q_inv_norm = _rsqrt_0;
          float _rsqrt_1 = rsqrtf(k_sum + 1e-06f);
          float k_inv_norm = _rsqrt_1;
          int row_is_valid = ((batch_end > chunk_offset + norm_row) ? 1 : 0);
#pragma unroll
          for (int norm_j_1 = 0; norm_j_1 < 16; norm_j_1++) {
            int norm_col_1 = norm_col_base + norm_j_1;
            float q_norm = 0.0f;
            float k_norm = 0.0f;
            if (row_is_valid != 0) {
              q_norm = (float)reinterpret_cast<const __nv_bfloat16*>((
                           reinterpret_cast<const uint8_t*>(
                               reinterpret_cast<const uint8_t*>(smem_q) + (q0_stage * 16384)) +
                           (norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 ^
                            (norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 >> 7 & 7)
                                << 4)))[0] *
                       q_inv_norm;
              k_norm = (float)reinterpret_cast<const __nv_bfloat16*>((
                           reinterpret_cast<const uint8_t*>(
                               reinterpret_cast<const uint8_t*>(smem_k) + (k0_stage * 16384)) +
                           (norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 ^
                            (norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 >> 7 & 7)
                                << 4)))[0] *
                       k_inv_norm;
            }
            {
              __nv_bfloat16 _bval_0 = __float2bfloat16_rn(q_norm);
              uint16_t _bits_0 = *(uint16_t*)&_bval_0;
              uint32_t _addr_0 = static_cast<uint32_t>(
                  (smem_q_addr + q0_stage * 16384 +
                   (unsigned int)(norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 ^
                                  (norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 >>
                                       7 &
                                   7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_0), "h"(_bits_0) : "memory");
            }
            {
              __nv_bfloat16 _bval_1 = __float2bfloat16_rn(k_norm);
              uint16_t _bits_1 = *(uint16_t*)&_bval_1;
              uint32_t _addr_1 = static_cast<uint32_t>(
                  (smem_k_addr + k0_stage * 16384 +
                   (unsigned int)(norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 ^
                                  (norm_col_1 / 64 * 8192 + norm_row * 128 + norm_col_1 % 64 * 2 >>
                                       7 &
                                   7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_1), "h"(_bits_1) : "memory");
            }
          }
          asm volatile("barrier.sync 13, 128;" ::: "memory");
          int gate_col = tid;
          float gate_bias = dt_bias[head_idx * 128 + gate_col];
          float _expf_0 = __expf(A_log[head_idx]);
          float gate_rate = _expf_0;
          float prefix_log2 = 0.0f;
#pragma unroll
          for (int gate_row = 0; gate_row < 16; gate_row++) {
            float q_norm_1 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_q) +
                                                  (q0_stage * 16384)) +
                 (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 ^
                  (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 >> 7 & 7) << 4)))[0];
            float k_norm_1 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_k) +
                                                  (k0_stage * 16384)) +
                 (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 ^
                  (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 >> 7 & 7) << 4)))[0];
            float gate_log2 = 0.0f;
            if (batch_end > chunk_offset + gate_row) {
              float gate_raw = (float)reinterpret_cast<const __nv_bfloat16*>(
                  (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_g) +
                                                    (g0_stage * 16384)) +
                   (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 ^
                    (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 >> 7 & 7) << 4)))[0];
              float gate_arg = gate_rate * (gate_raw + gate_bias);
              float _expf_1 = __expf(-gate_arg);
              float _rcp_0 = approx_rcp(1.0f + _expf_1);
              gate_log2 = lower_bound * 1.4426950408889634f * _rcp_0;
            }
            prefix_log2 += gate_log2;
            float _exp2_0 = approx_exp2(prefix_log2);
            float q_decay = _exp2_0;
            {
              __nv_bfloat16 _bval_2 = __float2bfloat16_rn(q_norm_1 * q_decay * scale);
              uint16_t _bits_2 = *(uint16_t*)&_bval_2;
              uint32_t _addr_2 = static_cast<uint32_t>(
                  (smem_q_addr + q0_stage * 16384 +
                   (unsigned int)(gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 ^
                                  (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 >> 7 &
                                   7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_2), "h"(_bits_2) : "memory");
            }
            {
              __nv_bfloat16 _bval_3 = __float2bfloat16_rn(k_norm_1 * q_decay);
              uint16_t _bits_3 = *(uint16_t*)&_bval_3;
              uint32_t _addr_3 = static_cast<uint32_t>(
                  (smem_k_addr + k0_stage * 16384 +
                   (unsigned int)(gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 ^
                                  (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 >> 7 &
                                   7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_3), "h"(_bits_3) : "memory");
            }
            {
              __nv_bfloat16 _bval_4 = __float2bfloat16_rn(k_norm_1 / q_decay);
              uint16_t _bits_4 = *(uint16_t*)&_bval_4;
              uint32_t _addr_4 = static_cast<uint32_t>(
                  (smem_g_addr + g0_stage * 16384 +
                   (unsigned int)(gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 ^
                                  (gate_col / 64 * 8192 + gate_row * 128 + gate_col % 64 * 2 >> 7 &
                                   7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_4), "h"(_bits_4) : "memory");
            }
          }
          float _exp2_1 = approx_exp2(prefix_log2);
          smem_g_total[gate0_stage * 128 + (unsigned int)gate_col] = _exp2_1;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(qkg_ready_addr + (g0_stage) * 8);
          mbarrier_arrive(load_gate_full_addr + (gate0_stage) * 8);
          unsigned int k1_stage = k_pre_stage;
          mbarrier_wait(load_k_full_addr + (k1_stage) * 8, k_pre_phase);
          k_pre_stage += 1;
          if (k_pre_stage == 3) {
            k_pre_stage = 0;
            k_pre_phase ^= 1;
          }
          unsigned int q1_stage = q_pre_stage;
          mbarrier_wait(load_q_full_addr + (q1_stage) * 8, q_pre_phase);
          q_pre_stage += 1;
          if (q_pre_stage == 2) {
            q_pre_stage = 0;
            q_pre_phase ^= 1;
          }
          unsigned int g1_stage = g_pre_stage;
          mbarrier_wait(load_g_full_addr + (g1_stage) * 8, g_pre_phase);
          g_pre_stage += 1;
          if (g_pre_stage == 2) {
            g_pre_stage = 0;
            g_pre_phase ^= 1;
          }
          unsigned int gate1_stage = gate_cg0_stage;
          mbarrier_wait(gate_cg1_empty_addr + (gate1_stage) * 8, gate_cg0_phase);
          gate_cg0_stage += 1;
          if (gate_cg0_stage == 5) {
            gate_cg0_stage = 0;
            gate_cg0_phase ^= 1;
          }
          int norm_row_0 = tid / 8;
          int norm_lane_1 = tid & 7;
          int norm_col_base_2 = norm_lane_1 * 16;
          float q_sum_3 = 0.0f;
          float k_sum_4 = 0.0f;
#pragma unroll
          for (int norm_j_2 = 0; norm_j_2 < 16; norm_j_2++) {
            int norm_col_2 = norm_col_base_2 + norm_j_2;
            float q_raw_1 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_q) +
                                                  (q1_stage * 16384)) +
                 (norm_col_2 / 64 * 8192 + norm_row_0 * 128 + norm_col_2 % 64 * 2 ^
                  (norm_col_2 / 64 * 8192 + norm_row_0 * 128 + norm_col_2 % 64 * 2 >> 7 & 7)
                      << 4)))[0];
            float k_raw_1 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_k) +
                                                  (k1_stage * 16384)) +
                 (norm_col_2 / 64 * 8192 + norm_row_0 * 128 + norm_col_2 % 64 * 2 ^
                  (norm_col_2 / 64 * 8192 + norm_row_0 * 128 + norm_col_2 % 64 * 2 >> 7 & 7)
                      << 4)))[0];
            float _fma_2 = __fmaf_rn(q_raw_1, q_raw_1, q_sum_3);
            q_sum_3 = _fma_2;
            float _fma_3 = __fmaf_rn(k_raw_1, k_raw_1, k_sum_4);
            k_sum_4 = _fma_3;
          }
          float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sum_3, 4);
          q_sum_3 += _shfl_xor_6;
          float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sum_4, 4);
          k_sum_4 += _shfl_xor_7;
          float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, q_sum_3, 2);
          q_sum_3 += _shfl_xor_8;
          float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, k_sum_4, 2);
          k_sum_4 += _shfl_xor_9;
          float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, q_sum_3, 1);
          q_sum_3 += _shfl_xor_10;
          float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, k_sum_4, 1);
          k_sum_4 += _shfl_xor_11;
          float _rsqrt_2 = rsqrtf(q_sum_3 + 1e-06f);
          float q_inv_norm_5 = _rsqrt_2;
          float _rsqrt_3 = rsqrtf(k_sum_4 + 1e-06f);
          float k_inv_norm_6 = _rsqrt_3;
          int row_is_valid_7 = ((batch_end > chunk_offset + 16 + norm_row_0) ? 1 : 0);
#pragma unroll
          for (int norm_j_3 = 0; norm_j_3 < 16; norm_j_3++) {
            int norm_col_3 = norm_col_base_2 + norm_j_3;
            float q_norm_2 = 0.0f;
            float k_norm_2 = 0.0f;
            if (row_is_valid_7 != 0) {
              q_norm_2 =
                  (float)reinterpret_cast<const __nv_bfloat16*>(
                      (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_q) +
                                                        (q1_stage * 16384)) +
                       (norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 ^
                        (norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 >> 7 & 7)
                            << 4)))[0] *
                  q_inv_norm_5;
              k_norm_2 =
                  (float)reinterpret_cast<const __nv_bfloat16*>(
                      (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_k) +
                                                        (k1_stage * 16384)) +
                       (norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 ^
                        (norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 >> 7 & 7)
                            << 4)))[0] *
                  k_inv_norm_6;
            }
            {
              __nv_bfloat16 _bval_5 = __float2bfloat16_rn(q_norm_2);
              uint16_t _bits_5 = *(uint16_t*)&_bval_5;
              uint32_t _addr_5 = static_cast<uint32_t>((
                  smem_q_addr + q1_stage * 16384 +
                  (unsigned int)(norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 ^
                                 (norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 >>
                                      7 &
                                  7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_5), "h"(_bits_5) : "memory");
            }
            {
              __nv_bfloat16 _bval_6 = __float2bfloat16_rn(k_norm_2);
              uint16_t _bits_6 = *(uint16_t*)&_bval_6;
              uint32_t _addr_6 = static_cast<uint32_t>((
                  smem_k_addr + k1_stage * 16384 +
                  (unsigned int)(norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 ^
                                 (norm_col_3 / 64 * 8192 + norm_row_0 * 128 + norm_col_3 % 64 * 2 >>
                                      7 &
                                  7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_6), "h"(_bits_6) : "memory");
            }
          }
          asm volatile("barrier.sync 13, 128;" ::: "memory");
          int gate_col_8 = tid;
          float gate_bias_9 = dt_bias[head_idx * 128 + gate_col_8];
          float _expf_2 = __expf(A_log[head_idx]);
          float gate_rate_10 = _expf_2;
          float prefix_log2_11 = 0.0f;
#pragma unroll
          for (int gate_row_1 = 0; gate_row_1 < 16; gate_row_1++) {
            float q_norm_3 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_q) +
                                                  (q1_stage * 16384)) +
                 (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 ^
                  (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 >> 7 & 7)
                      << 4)))[0];
            float k_norm_3 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_k) +
                                                  (k1_stage * 16384)) +
                 (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 ^
                  (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 >> 7 & 7)
                      << 4)))[0];
            float gate_log2_1 = 0.0f;
            if (batch_end > chunk_offset + 16 + gate_row_1) {
              float gate_raw_1 = (float)reinterpret_cast<const __nv_bfloat16*>(
                  (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_g) +
                                                    (g1_stage * 16384)) +
                   (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 ^
                    (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 >> 7 & 7)
                        << 4)))[0];
              float gate_arg_1 = gate_rate_10 * (gate_raw_1 + gate_bias_9);
              float _expf_3 = __expf(-gate_arg_1);
              float _rcp_1 = approx_rcp(1.0f + _expf_3);
              gate_log2_1 = lower_bound * 1.4426950408889634f * _rcp_1;
            }
            prefix_log2_11 += gate_log2_1;
            float _exp2_2 = approx_exp2(prefix_log2_11);
            float q_decay_1 = _exp2_2;
            {
              __nv_bfloat16 _bval_7 = __float2bfloat16_rn(q_norm_3 * q_decay_1 * scale);
              uint16_t _bits_7 = *(uint16_t*)&_bval_7;
              uint32_t _addr_7 = static_cast<uint32_t>((
                  smem_q_addr + q1_stage * 16384 +
                  (unsigned int)(gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 ^
                                 (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 >>
                                      7 &
                                  7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_7), "h"(_bits_7) : "memory");
            }
            {
              __nv_bfloat16 _bval_8 = __float2bfloat16_rn(k_norm_3 * q_decay_1);
              uint16_t _bits_8 = *(uint16_t*)&_bval_8;
              uint32_t _addr_8 = static_cast<uint32_t>((
                  smem_k_addr + k1_stage * 16384 +
                  (unsigned int)(gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 ^
                                 (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 >>
                                      7 &
                                  7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_8), "h"(_bits_8) : "memory");
            }
            {
              __nv_bfloat16 _bval_9 = __float2bfloat16_rn(k_norm_3 / q_decay_1);
              uint16_t _bits_9 = *(uint16_t*)&_bval_9;
              uint32_t _addr_9 = static_cast<uint32_t>((
                  smem_g_addr + g1_stage * 16384 +
                  (unsigned int)(gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 ^
                                 (gate_col_8 / 64 * 8192 + gate_row_1 * 128 + gate_col_8 % 64 * 2 >>
                                      7 &
                                  7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_9), "h"(_bits_9) : "memory");
            }
          }
          float _exp2_3 = approx_exp2(prefix_log2_11);
          smem_g_total[gate1_stage * 128 + (unsigned int)gate_col_8] = _exp2_3;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(qkg_ready_addr + (g1_stage) * 8);
          mbarrier_arrive(load_gate_full_addr + (gate1_stage) * 8);
          float qk_transfer0_cg0[32];
          float qk_transfer1_cg0[32];
#pragma unroll
          for (int qk_j_cg0 = 0; qk_j_cg0 < 32; qk_j_cg0++) {
            int qk_repeat_cg0 = qk_j_cg0 / 4;
            int qk_reg_cg0 = qk_j_cg0 & 3;
            int qk_row_cg0 = qk_warp_row_base_cg0 + lane_row_cg0 + qk_reg_cg0 / 2 * 8;
            int qk_col_cg0 = qk_repeat_cg0 * 8 + lane_quad_cg0 * 2 + (qk_reg_cg0 & 1);
            qk_transfer0_cg0[qk_j_cg0] =
                ((qk_row_cg0 < 16 && qk_col_cg0 < 16 && qk_row_cg0 >= qk_col_cg0) ? 1.0f : 0.0f);
            qk_transfer1_cg0[qk_j_cg0] =
                ((qk_row_cg0 < 16 && qk_col_cg0 < 16 && qk_row_cg0 >= qk_col_cg0) ? 1.0f : 0.0f);
          }
          unsigned int beta0_stage = beta_cg0_stage;
          mbarrier_wait(load_beta_full_addr + (beta0_stage) * 8, beta_cg0_phase);
          beta_cg0_stage += 1;
          if (beta_cg0_stage == 5) {
            beta_cg0_stage = 0;
            beta_cg0_phase ^= 1;
          }
          unsigned int beta1_stage = beta_cg0_stage;
          mbarrier_wait(load_beta_full_addr + (beta1_stage) * 8, beta_cg0_phase);
          beta_cg0_stage += 1;
          if (beta_cg0_stage == 5) {
            beta_cg0_stage = 0;
            beta_cg0_phase ^= 1;
          }
          int beta0_elem_base = beta0_stage * 64;
          int beta1_elem_base = beta1_stage * 64;
          unsigned int ainv0_stage = ainv_cg0_stage;
          mbarrier_wait(ainv_smem_empty_addr + (ainv0_stage) * 8, ainv_cg0_phase);
          ainv_cg0_stage += 1;
          if (ainv_cg0_stage == 3) {
            ainv_cg0_stage = 0;
            ainv_cg0_phase ^= 1;
          }
          unsigned int ainv1_stage = ainv_cg0_stage;
          mbarrier_wait(ainv_smem_empty_addr + (ainv1_stage) * 8, ainv_cg0_phase);
          ainv_cg0_stage += 1;
          if (ainv_cg0_stage == 3) {
            ainv_cg0_stage = 0;
            ainv_cg0_phase ^= 1;
          }
          mbarrier_wait(cg0_shared_acc_full_addr, 0);
          int kk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
          float _tmem_load_0[32];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x8.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
              : "r"(kk_addr));
          int kk_beta_row0_cg0 = qk_warp_row_base_cg0 + lane_row_cg0;
          int kk_beta_row1_cg0 = kk_beta_row0_cg0 + 8;
          float kk_beta0_cg0 = smem_beta[beta0_elem_base + kk_beta_row0_cg0];
          float kk_beta1_cg0 = smem_beta[beta0_elem_base + kk_beta_row1_cg0];
          int kk_stsm_row_cg0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
          int kk_stsm_col_lane_cg0 = (lane & 16) / 2;
#pragma unroll
          for (int qk_repeat_cg0_1 = 0; qk_repeat_cg0_1 < 4; qk_repeat_cg0_1++) {
            const int qk_j0_cg0 = qk_repeat_cg0_1 * 8;
            float _t0[8];
#pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
              reinterpret_cast<float2*>(_t0)[_ls] =
                  mul_f32x2(reinterpret_cast<float2*>((_tmem_load_0 + qk_j0_cg0))[_ls],
                            reinterpret_cast<const float2*>((qk_transfer0_cg0 + qk_j0_cg0))[_ls]);
            const float2 _scale2_10 = {kk_beta0_cg0, kk_beta0_cg0};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 0))[_ls], _scale2_10);
            const float2 _scale2_11 = {kk_beta1_cg0, kk_beta1_cg0};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 2))[_ls], _scale2_11);
            const float2 _scale2_12 = {kk_beta0_cg0, kk_beta0_cg0};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 4))[_ls], _scale2_12);
            const float2 _scale2_13 = {kk_beta1_cg0, kk_beta1_cg0};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 6))[_ls], _scale2_13);
            uint32_t _t0_bf16[4];
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 =
                  __float22bfloat162_rn(make_float2(_t0[_lp * 2 + 0], _t0[_lp * 2 + 1 + 0]));
              _t0_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int kk_stsm_addr_cg0 =
                smem_ainv_rm_addr + ainv0_stage * 8192 +
                (unsigned int)(kk_stsm_row_cg0 * 128 +
                                   (qk_repeat_cg0_1 * 16 + kk_stsm_col_lane_cg0) * 2 ^
                               (kk_stsm_row_cg0 * 128 +
                                        (qk_repeat_cg0_1 * 16 + kk_stsm_col_lane_cg0) * 2 >>
                                    7 &
                                7) << 4);
            uint32_t _stmatrix_addr_14 =
                static_cast<uint32_t>((unsigned long long)kk_stsm_addr_cg0);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                             _stmatrix_addr_14),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[0])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[1])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[2])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[3]))
                         : "memory");
          }
          mbarrier_arrive(cg0_shared_acc_empty_addr);
          mbarrier_wait(cg0_shared_acc_full_addr + 8, 0);
          int kk_addr_12 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
          float _tmem_load_1[32];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x8.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
              : "r"(kk_addr_12));
          int kk_beta_row0_cg0_13 = qk_warp_row_base_cg0 + lane_row_cg0;
          int kk_beta_row1_cg0_14 = kk_beta_row0_cg0_13 + 8;
          float kk_beta0_cg0_15 = smem_beta[beta1_elem_base + kk_beta_row0_cg0_13];
          float kk_beta1_cg0_16 = smem_beta[beta1_elem_base + kk_beta_row1_cg0_14];
          int kk_stsm_row_cg0_17 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
          int kk_stsm_col_lane_cg0_18 = (lane & 16) / 2;
#pragma unroll
          for (int qk_repeat_cg0_2 = 0; qk_repeat_cg0_2 < 4; qk_repeat_cg0_2++) {
            const int qk_j0_cg0_1 = qk_repeat_cg0_2 * 8;
            float _t1[8];
#pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
              reinterpret_cast<float2*>(_t1)[_ls] =
                  mul_f32x2(reinterpret_cast<float2*>((_tmem_load_1 + qk_j0_cg0_1))[_ls],
                            reinterpret_cast<const float2*>((qk_transfer1_cg0 + qk_j0_cg0_1))[_ls]);
            const float2 _scale2_15 = {kk_beta0_cg0_15, kk_beta0_cg0_15};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 0))[_ls], _scale2_15);
            const float2 _scale2_16 = {kk_beta1_cg0_16, kk_beta1_cg0_16};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 2))[_ls], _scale2_16);
            const float2 _scale2_17 = {kk_beta0_cg0_15, kk_beta0_cg0_15};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 4))[_ls], _scale2_17);
            const float2 _scale2_18 = {kk_beta1_cg0_16, kk_beta1_cg0_16};
#pragma unroll
            for (int _ls = 0; _ls < 1; _ls++)
              mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 6))[_ls], _scale2_18);
            uint32_t _t1_bf16[4];
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 =
                  __float22bfloat162_rn(make_float2(_t1[_lp * 2 + 0], _t1[_lp * 2 + 1 + 0]));
              _t1_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int kk_stsm_addr_cg0_1 =
                smem_ainv_rm_addr + ainv1_stage * 8192 +
                (unsigned int)(kk_stsm_row_cg0_17 * 128 +
                                   (qk_repeat_cg0_2 * 16 + kk_stsm_col_lane_cg0_18) * 2 ^
                               (kk_stsm_row_cg0_17 * 128 +
                                        (qk_repeat_cg0_2 * 16 + kk_stsm_col_lane_cg0_18) * 2 >>
                                    7 &
                                7) << 4);
            uint32_t _stmatrix_addr_19 =
                static_cast<uint32_t>((unsigned long long)kk_stsm_addr_cg0_1);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                             _stmatrix_addr_19),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[0])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[1])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[2])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[3]))
                         : "memory");
          }
          mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
          asm volatile("barrier.sync 12, 128;" ::: "memory");
          int inverse_group_cg0 = warp_id_in_role_cg0 / 2;
          int inverse_local_warp_cg0 = warp_id_in_role_cg0 & 1;
          unsigned int inverse_stage_cg0 = ainv0_stage;
          if (inverse_group_cg0 == 1) {
            inverse_stage_cg0 = ainv1_stage;
          }
          int inverse_row_cg0 = inverse_local_warp_cg0 * 32 + lane;
          int diag_block_cg0 = inverse_row_cg0 / 8;
          int lane_in_diag_cg0 = lane & 7;
          int diag_col_base_cg0 = diag_block_cg0 * 8;
          float inv_row_cg0[8];
          if (inverse_row_cg0 < 64) {
#pragma unroll
            for (int inv_j = 0; inv_j < 8; inv_j++) {
              int inv_col_cg0 = diag_col_base_cg0 + inv_j;
              inv_row_cg0[inv_j] = reinterpret_cast<const __nv_bfloat16*>(
                  reinterpret_cast<const uint8_t*>(smem_ainv_rm) +
                  (inverse_stage_cg0 * 8192 +
                   (unsigned int)(inverse_row_cg0 * 128 + inv_col_cg0 * 2 ^
                                  (inverse_row_cg0 * 128 + inv_col_cg0 * 2 >> 7 & 7) << 4)))[0];
              if (lane_in_diag_cg0 == inv_j) {
                inv_row_cg0[inv_j] = 1.0f;
              }
            }
            int diag_group_base_cg0 = lane - lane_in_diag_cg0;
#pragma unroll
            for (int src_row_cg0 = 0; src_row_cg0 < 7; src_row_cg0++) {
              float row_scale_cg0 = -1.0f * inv_row_cg0[src_row_cg0];
#pragma unroll
              for (int prev_col_cg0 = 0; prev_col_cg0 < src_row_cg0; prev_col_cg0++) {
                int pivot_lane_cg0 = diag_group_base_cg0 + src_row_cg0;
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, inv_row_cg0[prev_col_cg0], pivot_lane_cg0);
                float shfl_val_cg0 = _shfl_0;
                if (lane_in_diag_cg0 > src_row_cg0) {
                  inv_row_cg0[prev_col_cg0] =
                      inv_row_cg0[prev_col_cg0] + row_scale_cg0 * shfl_val_cg0;
                }
              }
              if (lane_in_diag_cg0 > src_row_cg0) {
                inv_row_cg0[src_row_cg0] = row_scale_cg0;
              }
            }
            uint32_t inv_row_cg0_bf16[4];
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(inv_row_cg0[_lp * 2 + 0], inv_row_cg0[_lp * 2 + 1 + 0]));
              inv_row_cg0_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int inv_diag_addr_cg0 =
                smem_ainv_rm_addr + inverse_stage_cg0 * 8192 +
                (unsigned int)(inverse_row_cg0 * 128 + diag_col_base_cg0 * 2 ^
                               (inverse_row_cg0 * 128 + diag_col_base_cg0 * 2 >> 7 & 7) << 4);
#pragma unroll
            for (int inv_store_j_cg0 = 0; inv_store_j_cg0 < 4; inv_store_j_cg0++) {
              asm volatile("st.shared.b32 [%0], %1;" ::"r"(inv_diag_addr_cg0 + inv_store_j_cg0 * 4),
                           "r"((inv_row_cg0_bf16[inv_store_j_cg0])));
            }
          }
          asm volatile("barrier.sync 12, 128;" ::: "memory");
          if (inverse_local_warp_cg0 == 0) {
            int inv16_lane_row_cg0 = lane & 7;
            int inv16_d_addr_cg0 =
                smem_ainv_rm_addr + inverse_stage_cg0 * 8192 +
                (unsigned int)((8 + inv16_lane_row_cg0) * 128 + 16 ^
                               ((8 + inv16_lane_row_cg0) * 128 + 16 >> 7 & 7) << 4);
            int inv16_c_addr_cg0 = smem_ainv_rm_addr + inverse_stage_cg0 * 8192 +
                                   (unsigned int)((8 + inv16_lane_row_cg0) * 128 ^
                                                  ((8 + inv16_lane_row_cg0) * 128 >> 7 & 7) << 4);
            int inv16_a_addr_cg0 =
                smem_ainv_rm_addr + inverse_stage_cg0 * 8192 +
                (unsigned int)(inv16_lane_row_cg0 * 128 ^ (inv16_lane_row_cg0 * 128 >> 7 & 7) << 4);
            int inv16_o_addr_cg0 = smem_ainv_rm_addr + inverse_stage_cg0 * 8192 +
                                   (unsigned int)((8 + inv16_lane_row_cg0) * 128 ^
                                                  ((8 + inv16_lane_row_cg0) * 128 >> 7 & 7) << 4);
            unsigned int inv16_d_frag_cg0[2];
            unsigned int inv16_c_frag_cg0[1];
            float inv16_dc_acc_cg0[4];
            unsigned int inv16_dc_bf16_cg0[2];
            unsigned int inv16_a_frag_cg0[1];
            float inv16_o_acc_cg0[4];
            unsigned int inv16_o_bf16_cg0[2];
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         : "=r"(inv16_d_frag_cg0[0])
                         : "r"(inv16_d_addr_cg0)
                         : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                         : "=r"(inv16_d_frag_cg0[1])
                         : "r"(inv16_d_addr_cg0)
                         : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                         : "=r"(inv16_c_frag_cg0[0])
                         : "r"(inv16_c_addr_cg0)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, "
                "{%6}, {%7, %8, %9, %10};\n"
                : "=f"(inv16_dc_acc_cg0[0]), "=f"(inv16_dc_acc_cg0[1]), "=f"(inv16_dc_acc_cg0[2]),
                  "=f"(inv16_dc_acc_cg0[3])
                : "r"(inv16_d_frag_cg0[0]), "r"(inv16_d_frag_cg0[1]), "r"(inv16_c_frag_cg0[0]),
                  "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
#pragma unroll
            for (int inv16_i_cg0 = 0; inv16_i_cg0 < 4; inv16_i_cg0++) {
              inv16_dc_acc_cg0[inv16_i_cg0] = inv16_dc_acc_cg0[inv16_i_cg0] * -1.0f;
            }
#pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(inv16_dc_acc_cg0[_lp * 2 + 0], inv16_dc_acc_cg0[_lp * 2 + 1 + 0]));
              inv16_dc_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                         : "=r"(inv16_a_frag_cg0[0])
                         : "r"(inv16_a_addr_cg0)
                         : "memory");
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, "
                "{%6}, {%7, %8, %9, %10};\n"
                : "=f"(inv16_o_acc_cg0[0]), "=f"(inv16_o_acc_cg0[1]), "=f"(inv16_o_acc_cg0[2]),
                  "=f"(inv16_o_acc_cg0[3])
                : "r"(inv16_dc_bf16_cg0[0]), "r"(inv16_dc_bf16_cg0[1]), "r"(inv16_a_frag_cg0[0]),
                  "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
#pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(inv16_o_acc_cg0[_lp * 2 + 0], inv16_o_acc_cg0[_lp * 2 + 1 + 0]));
              inv16_o_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _stmatrix_addr_20 =
                static_cast<uint32_t>((unsigned long long)inv16_o_addr_cg0);
            asm volatile(
                "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n" ::"r"(_stmatrix_addr_20),
                "r"(*reinterpret_cast<const uint32_t*>(&inv16_o_bf16_cg0[0]))
                : "memory");
          }
          asm volatile("barrier.sync 12, 128;" ::: "memory");
          asm volatile("barrier.sync 12, 128;" ::: "memory");
          asm volatile("barrier.sync 12, 128;" ::: "memory");
          if (warp_id_in_role_cg0 < 4) {
            unsigned int ainv_beta_ld_bits_cg0[4];
            int ainv_beta_tile_row_cg0 = warp_id_in_role_cg0 * 16 + (lane & 7) + (lane & 8);
            int ainv_beta_tile_col_lane_cg0 = (lane & 16) / 2;
            int ainv_beta_pair_col_cg0 = (lane & 3) * 2;
#pragma unroll 1
            for (int beta_col_tile_cg0 = 0; beta_col_tile_cg0 < 4; beta_col_tile_cg0++) {
              int ainv_beta_tile_col_cg0 = beta_col_tile_cg0 * 16 + ainv_beta_tile_col_lane_cg0;
              int ainv_beta_ld_addr_cg0 =
                  smem_ainv_rm_addr + ainv0_stage * 8192 +
                  (unsigned int)(ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 ^
                                 (ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 >> 7 &
                                  7) << 4);
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                           : "=r"(ainv_beta_ld_bits_cg0[0]), "=r"(ainv_beta_ld_bits_cg0[1]),
                             "=r"(ainv_beta_ld_bits_cg0[2]), "=r"(ainv_beta_ld_bits_cg0[3])
                           : "r"(ainv_beta_ld_addr_cg0)
                           : "memory");
              int ainv_beta_col0_cg0 = beta_col_tile_cg0 * 16 + ainv_beta_pair_col_cg0;
              int ainv_beta_col8_cg0 = ainv_beta_col0_cg0 + 8;
              float ainv_beta_scale0_lo_cg0 = smem_beta[beta0_elem_base + ainv_beta_col0_cg0];
              float ainv_beta_scale0_hi_cg0 = smem_beta[beta0_elem_base + ainv_beta_col0_cg0 + 1];
              float ainv_beta_scale8_lo_cg0 = smem_beta[beta0_elem_base + ainv_beta_col8_cg0];
              float ainv_beta_scale8_hi_cg0 = smem_beta[beta0_elem_base + ainv_beta_col8_cg0 + 1];
              uint32_t _bf16x2_scale_0;
              {
                uint32_t _bf16x2_pair_21 = ainv_beta_ld_bits_cg0[0];
                float _bf16x2_lo_21;
                float _bf16x2_hi_21;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_21)
                             : "h"((uint16_t)(_bf16x2_pair_21 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_21)
                             : "h"((uint16_t)(_bf16x2_pair_21 >> 16)));
                _bf16x2_lo_21 *= ainv_beta_scale0_lo_cg0;
                _bf16x2_hi_21 *= ainv_beta_scale0_hi_cg0;
                uint32_t _bf16x2_out_21;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_21)
                             : "f"(_bf16x2_hi_21), "f"(_bf16x2_lo_21));
                _bf16x2_scale_0 = _bf16x2_out_21;
              }
              uint32_t _bf16x2_scale_1;
              {
                uint32_t _bf16x2_pair_22 = ainv_beta_ld_bits_cg0[1];
                float _bf16x2_lo_22;
                float _bf16x2_hi_22;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_22)
                             : "h"((uint16_t)(_bf16x2_pair_22 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_22)
                             : "h"((uint16_t)(_bf16x2_pair_22 >> 16)));
                _bf16x2_lo_22 *= ainv_beta_scale0_lo_cg0;
                _bf16x2_hi_22 *= ainv_beta_scale0_hi_cg0;
                uint32_t _bf16x2_out_22;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_22)
                             : "f"(_bf16x2_hi_22), "f"(_bf16x2_lo_22));
                _bf16x2_scale_1 = _bf16x2_out_22;
              }
              uint32_t _bf16x2_scale_2;
              {
                uint32_t _bf16x2_pair_23 = ainv_beta_ld_bits_cg0[2];
                float _bf16x2_lo_23;
                float _bf16x2_hi_23;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_23)
                             : "h"((uint16_t)(_bf16x2_pair_23 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_23)
                             : "h"((uint16_t)(_bf16x2_pair_23 >> 16)));
                _bf16x2_lo_23 *= ainv_beta_scale8_lo_cg0;
                _bf16x2_hi_23 *= ainv_beta_scale8_hi_cg0;
                uint32_t _bf16x2_out_23;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_23)
                             : "f"(_bf16x2_hi_23), "f"(_bf16x2_lo_23));
                _bf16x2_scale_2 = _bf16x2_out_23;
              }
              uint32_t _bf16x2_scale_3;
              {
                uint32_t _bf16x2_pair_24 = ainv_beta_ld_bits_cg0[3];
                float _bf16x2_lo_24;
                float _bf16x2_hi_24;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_24)
                             : "h"((uint16_t)(_bf16x2_pair_24 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_24)
                             : "h"((uint16_t)(_bf16x2_pair_24 >> 16)));
                _bf16x2_lo_24 *= ainv_beta_scale8_lo_cg0;
                _bf16x2_hi_24 *= ainv_beta_scale8_hi_cg0;
                uint32_t _bf16x2_out_24;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_24)
                             : "f"(_bf16x2_hi_24), "f"(_bf16x2_lo_24));
                _bf16x2_scale_3 = _bf16x2_out_24;
              }
              uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(
                  (unsigned long long)(smem_ainv_addr + ainv0_stage * 8192 +
                                       (unsigned int)(ainv_beta_tile_row_cg0 * 128 +
                                                          ainv_beta_tile_col_cg0 * 2 ^
                                                      (ainv_beta_tile_row_cg0 * 128 +
                                                               ainv_beta_tile_col_cg0 * 2 >>
                                                           7 &
                                                       7) << 4)));
              asm volatile(
                  "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                      _stmatrix_addr_25),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_0)),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_1)),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_2)),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_3))
                  : "memory");
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(ainv_ready_addr + (ainv0_stage) * 8);
          mbarrier_arrive(beta_smem_empty_addr + (beta0_stage) * 8);
          if (warp_id_in_role_cg0 < 4) {
            unsigned int ainv_beta_ld_bits_cg0_1[4];
            int ainv_beta_tile_row_cg0_1 = warp_id_in_role_cg0 * 16 + (lane & 7) + (lane & 8);
            int ainv_beta_tile_col_lane_cg0_1 = (lane & 16) / 2;
            int ainv_beta_pair_col_cg0_1 = (lane & 3) * 2;
#pragma unroll 1
            for (int beta_col_tile_cg0_1 = 0; beta_col_tile_cg0_1 < 4; beta_col_tile_cg0_1++) {
              int ainv_beta_tile_col_cg0_1 =
                  beta_col_tile_cg0_1 * 16 + ainv_beta_tile_col_lane_cg0_1;
              int ainv_beta_ld_addr_cg0_1 =
                  smem_ainv_rm_addr + ainv1_stage * 8192 +
                  (unsigned int)(ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 ^
                                 (ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 >>
                                      7 &
                                  7) << 4);
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                           : "=r"(ainv_beta_ld_bits_cg0_1[0]), "=r"(ainv_beta_ld_bits_cg0_1[1]),
                             "=r"(ainv_beta_ld_bits_cg0_1[2]), "=r"(ainv_beta_ld_bits_cg0_1[3])
                           : "r"(ainv_beta_ld_addr_cg0_1)
                           : "memory");
              int ainv_beta_col0_cg0_1 = beta_col_tile_cg0_1 * 16 + ainv_beta_pair_col_cg0_1;
              int ainv_beta_col8_cg0_1 = ainv_beta_col0_cg0_1 + 8;
              float ainv_beta_scale0_lo_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col0_cg0_1];
              float ainv_beta_scale0_hi_cg0_1 =
                  smem_beta[beta1_elem_base + ainv_beta_col0_cg0_1 + 1];
              float ainv_beta_scale8_lo_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col8_cg0_1];
              float ainv_beta_scale8_hi_cg0_1 =
                  smem_beta[beta1_elem_base + ainv_beta_col8_cg0_1 + 1];
              uint32_t _bf16x2_scale_4;
              {
                uint32_t _bf16x2_pair_26 = ainv_beta_ld_bits_cg0_1[0];
                float _bf16x2_lo_26;
                float _bf16x2_hi_26;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_26)
                             : "h"((uint16_t)(_bf16x2_pair_26 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_26)
                             : "h"((uint16_t)(_bf16x2_pair_26 >> 16)));
                _bf16x2_lo_26 *= ainv_beta_scale0_lo_cg0_1;
                _bf16x2_hi_26 *= ainv_beta_scale0_hi_cg0_1;
                uint32_t _bf16x2_out_26;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_26)
                             : "f"(_bf16x2_hi_26), "f"(_bf16x2_lo_26));
                _bf16x2_scale_4 = _bf16x2_out_26;
              }
              uint32_t _bf16x2_scale_5;
              {
                uint32_t _bf16x2_pair_27 = ainv_beta_ld_bits_cg0_1[1];
                float _bf16x2_lo_27;
                float _bf16x2_hi_27;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_27)
                             : "h"((uint16_t)(_bf16x2_pair_27 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_27)
                             : "h"((uint16_t)(_bf16x2_pair_27 >> 16)));
                _bf16x2_lo_27 *= ainv_beta_scale0_lo_cg0_1;
                _bf16x2_hi_27 *= ainv_beta_scale0_hi_cg0_1;
                uint32_t _bf16x2_out_27;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_27)
                             : "f"(_bf16x2_hi_27), "f"(_bf16x2_lo_27));
                _bf16x2_scale_5 = _bf16x2_out_27;
              }
              uint32_t _bf16x2_scale_6;
              {
                uint32_t _bf16x2_pair_28 = ainv_beta_ld_bits_cg0_1[2];
                float _bf16x2_lo_28;
                float _bf16x2_hi_28;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_28)
                             : "h"((uint16_t)(_bf16x2_pair_28 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_28)
                             : "h"((uint16_t)(_bf16x2_pair_28 >> 16)));
                _bf16x2_lo_28 *= ainv_beta_scale8_lo_cg0_1;
                _bf16x2_hi_28 *= ainv_beta_scale8_hi_cg0_1;
                uint32_t _bf16x2_out_28;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_28)
                             : "f"(_bf16x2_hi_28), "f"(_bf16x2_lo_28));
                _bf16x2_scale_6 = _bf16x2_out_28;
              }
              uint32_t _bf16x2_scale_7;
              {
                uint32_t _bf16x2_pair_29 = ainv_beta_ld_bits_cg0_1[3];
                float _bf16x2_lo_29;
                float _bf16x2_hi_29;
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_lo_29)
                             : "h"((uint16_t)(_bf16x2_pair_29 & 0xFFFFu)));
                asm volatile("cvt.f32.bf16 %0, %1;"
                             : "=f"(_bf16x2_hi_29)
                             : "h"((uint16_t)(_bf16x2_pair_29 >> 16)));
                _bf16x2_lo_29 *= ainv_beta_scale8_lo_cg0_1;
                _bf16x2_hi_29 *= ainv_beta_scale8_hi_cg0_1;
                uint32_t _bf16x2_out_29;
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                             : "=r"(_bf16x2_out_29)
                             : "f"(_bf16x2_hi_29), "f"(_bf16x2_lo_29));
                _bf16x2_scale_7 = _bf16x2_out_29;
              }
              uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(
                  (unsigned long long)(smem_ainv_addr + ainv1_stage * 8192 +
                                       (unsigned int)(ainv_beta_tile_row_cg0_1 * 128 +
                                                          ainv_beta_tile_col_cg0_1 * 2 ^
                                                      (ainv_beta_tile_row_cg0_1 * 128 +
                                                               ainv_beta_tile_col_cg0_1 * 2 >>
                                                           7 &
                                                       7) << 4)));
              asm volatile(
                  "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                      _stmatrix_addr_30),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_4)),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_5)),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_6)),
                  "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_7))
                  : "memory");
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(ainv_ready_addr + (ainv1_stage) * 8);
          mbarrier_arrive(beta_smem_empty_addr + (beta1_stage) * 8);
          unsigned int qk0_stage = qk_cg0_stage;
          mbarrier_wait(qk_smem_empty_addr + (qk0_stage) * 8, qk_cg0_phase);
          qk_cg0_stage += 1;
          if (qk_cg0_stage == 2) {
            qk_cg0_stage = 0;
            qk_cg0_phase ^= 1;
          }
          unsigned int qk1_stage = qk_cg0_stage;
          mbarrier_wait(qk_smem_empty_addr + (qk1_stage) * 8, qk_cg0_phase);
          qk_cg0_stage += 1;
          if (qk_cg0_stage == 2) {
            qk_cg0_stage = 0;
            qk_cg0_phase ^= 1;
          }
          mbarrier_wait(cg0_shared_acc_full_addr, 1);
          int qk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
          float _tmem_load_2[32];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x8.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
              : "r"(qk_addr));
          int qk_stsm_row_cg0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
          int qk_stsm_col_lane_cg0 = (lane & 16) / 2;
#pragma unroll
          for (int qk_repeat_cg0_3 = 0; qk_repeat_cg0_3 < 4; qk_repeat_cg0_3++) {
            const int qk_j0_cg0_2 = qk_repeat_cg0_3 * 8;
            float _t2[8];
#pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
              reinterpret_cast<float2*>(_t2)[_ls] =
                  mul_f32x2(reinterpret_cast<float2*>((_tmem_load_2 + qk_j0_cg0_2))[_ls],
                            reinterpret_cast<const float2*>((qk_transfer0_cg0 + qk_j0_cg0_2))[_ls]);
            uint32_t _t2_bf16[4];
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 =
                  __float22bfloat162_rn(make_float2(_t2[_lp * 2 + 0], _t2[_lp * 2 + 1 + 0]));
              _t2_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _stmatrix_addr_31 = static_cast<uint32_t>((
                unsigned long long)(smem_qk_addr + qk0_stage * 8192 +
                                    (unsigned int)(qk_stsm_row_cg0 * 128 + (qk_repeat_cg0_3 * 16 +
                                                                            qk_stsm_col_lane_cg0) *
                                                                               2 ^
                                                   (qk_stsm_row_cg0 * 128 + (qk_repeat_cg0_3 * 16 +
                                                                             qk_stsm_col_lane_cg0) *
                                                                                2 >>
                                                        7 &
                                                    7) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                             _stmatrix_addr_31),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t2_bf16[0])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t2_bf16[1])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t2_bf16[2])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t2_bf16[3]))
                         : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(cg0_shared_acc_empty_addr);
          mbarrier_arrive(qk_ready_addr + (qk0_stage) * 8);
          mbarrier_wait(cg0_shared_acc_full_addr + 8, 1);
          int qk_addr_19 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
          float _tmem_load_3[32];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x8.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
              : "r"(qk_addr_19));
          int qk_stsm_row_cg0_20 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
          int qk_stsm_col_lane_cg0_21 = (lane & 16) / 2;
#pragma unroll
          for (int qk_repeat_cg0_4 = 0; qk_repeat_cg0_4 < 4; qk_repeat_cg0_4++) {
            const int qk_j0_cg0_3 = qk_repeat_cg0_4 * 8;
            float _t3[8];
#pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
              reinterpret_cast<float2*>(_t3)[_ls] =
                  mul_f32x2(reinterpret_cast<float2*>((_tmem_load_3 + qk_j0_cg0_3))[_ls],
                            reinterpret_cast<const float2*>((qk_transfer1_cg0 + qk_j0_cg0_3))[_ls]);
            uint32_t _t3_bf16[4];
#pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
              __nv_bfloat162 _bf2 =
                  __float22bfloat162_rn(make_float2(_t3[_lp * 2 + 0], _t3[_lp * 2 + 1 + 0]));
              _t3_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(
                (unsigned long long)(smem_qk_addr + qk1_stage * 8192 +
                                     (unsigned int)(qk_stsm_row_cg0_20 * 128 +
                                                        (qk_repeat_cg0_4 * 16 +
                                                         qk_stsm_col_lane_cg0_21) *
                                                            2 ^
                                                    (qk_stsm_row_cg0_20 * 128 +
                                                             (qk_repeat_cg0_4 * 16 +
                                                              qk_stsm_col_lane_cg0_21) *
                                                                 2 >>
                                                         7 &
                                                     7) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                             _stmatrix_addr_32),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[0])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[1])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[2])),
                         "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[3]))
                         : "memory");
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
          mbarrier_arrive(qk_ready_addr + (qk1_stage) * 8);
          mbarrier_wait(ki_mma_consumed_addr + (g0_stage) * 8, g_kr_phase);
          int kr_col = tid;
          float g_total = smem_g_total[gate0_stage * 128 + (unsigned int)kr_col];
#pragma unroll
          for (int kr_row = 0; kr_row < 16; kr_row++) {
            float ki = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_g) +
                                                  (g0_stage * 16384)) +
                 (kr_col / 64 * 8192 + kr_row * 128 + kr_col % 64 * 2 ^
                  (kr_col / 64 * 8192 + kr_row * 128 + kr_col % 64 * 2 >> 7 & 7) << 4)))[0];
            {
              __nv_bfloat16 _bval_33 = __float2bfloat16_rn(ki * g_total);
              uint16_t _bits_33 = *(uint16_t*)&_bval_33;
              uint32_t _addr_33 = static_cast<uint32_t>(
                  (smem_g_addr + g0_stage * 16384 +
                   (unsigned int)(kr_col / 64 * 8192 + kr_row * 128 + kr_col % 64 * 2 ^
                                  (kr_col / 64 * 8192 + kr_row * 128 + kr_col % 64 * 2 >> 7 & 7)
                                      << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_33), "h"(_bits_33) : "memory");
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(kr_ready_addr + (g0_stage) * 8);
          g_kr_stage += 1;
          if (g_kr_stage == 2) {
            g_kr_stage = 0;
            g_kr_phase ^= 1;
          }
          mbarrier_wait(ki_mma_consumed_addr + (g1_stage) * 8, g_kr_phase);
          int kr_col_22 = tid;
          float g_total_23 = smem_g_total[gate1_stage * 128 + (unsigned int)kr_col_22];
#pragma unroll
          for (int kr_row_1 = 0; kr_row_1 < 16; kr_row_1++) {
            float ki_1 = (float)reinterpret_cast<const __nv_bfloat16*>(
                (reinterpret_cast<const uint8_t*>(reinterpret_cast<const uint8_t*>(smem_g) +
                                                  (g1_stage * 16384)) +
                 (kr_col_22 / 64 * 8192 + kr_row_1 * 128 + kr_col_22 % 64 * 2 ^
                  (kr_col_22 / 64 * 8192 + kr_row_1 * 128 + kr_col_22 % 64 * 2 >> 7 & 7) << 4)))[0];
            {
              __nv_bfloat16 _bval_34 = __float2bfloat16_rn(ki_1 * g_total_23);
              uint16_t _bits_34 = *(uint16_t*)&_bval_34;
              uint32_t _addr_34 = static_cast<uint32_t>((
                  smem_g_addr + g1_stage * 16384 +
                  (unsigned int)(kr_col_22 / 64 * 8192 + kr_row_1 * 128 + kr_col_22 % 64 * 2 ^
                                 (kr_col_22 / 64 * 8192 + kr_row_1 * 128 + kr_col_22 % 64 * 2 >> 7 &
                                  7) << 4)));
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(_addr_34), "h"(_bits_34) : "memory");
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(kr_ready_addr + (g1_stage) * 8);
          g_kr_stage += 1;
          if (g_kr_stage == 2) {
            g_kr_stage = 0;
            g_kr_phase ^= 1;
          }
        }
      }
    }
  }
  // ---- Role: compute_group_1 ----
  if (warp >= 4 && warp <= 7) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
    {  // compute_group_1_main
      unsigned int gate_cg1_stage = 0;
      unsigned int gate_cg1_phase = 0;
      unsigned int v_cg1_stage = 0;
      unsigned int v_cg1_phase = 0;
      unsigned int o_cg1_stage = 0;
      unsigned int o_cg1_phase = 1;
      unsigned int _phase_initial_state_loaded_0 = 0;
      unsigned int _phase_kv_acc_full_0 = 0;
      unsigned int _phase_cg1_shared_acc_full_0 = 0;
      unsigned int _phase_q_state_acc_full_0 = 0;
#pragma unroll 1
      for (unsigned int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
        int num_o_heads_1 = ((num_q_heads >= num_v_heads) ? num_q_heads : num_v_heads);
        int batch_idx_1 = tile_1 / (unsigned int)num_o_heads_1;
        int head_idx_1 = tile_1 % (unsigned int)num_o_heads_1;
        int qk_head_idx_1 =
            ((num_q_heads >= num_v_heads) ? head_idx_1 : head_idx_1 / (num_v_heads / num_q_heads));
        int v_head_idx_1 =
            ((num_v_heads >= num_q_heads) ? head_idx_1 : head_idx_1 / (num_q_heads / num_v_heads));
        int batch_start_1 = (int)cu_seqlens[batch_idx_1];
        int batch_end_1 = (int)cu_seqlens[batch_idx_1 + 1];
        int seqlen_b_1 = batch_end_1 - batch_start_1;
        int num_pairs_b_1 = (seqlen_b_1 + 32 - 1) / 32;
        int num_chunks_b_1 = num_pairs_b_1 * 2;
        if (USE_INITIAL_STATE != 0 && num_chunks_b_1 > 0) {
          int warp_in_wg = warp % 4;
          int state_tmem_row_base_init = warp_in_wg * 32 << 16;
          int warp_id_in_role_1 = (warp - 4);
          int state_warp_init = warp_id_in_role_1;
          int state_row_init = state_warp_init * 32 + lane;
          int state_base_init =
              (batch_idx_1 * num_o_heads_1 + head_idx_1) * 128 * 128 + state_row_init * 128;
#pragma unroll
          for (int state_col_block_init = 0; state_col_block_init < 4; state_col_block_init++) {
            float state_init_frag[32];
#pragma unroll
            for (int state_vec_init = 0; state_vec_init < 8; state_vec_init++) {
              {
                float4 _v4 = *reinterpret_cast<const float4*>(initial_state + state_base_init +
                                                              state_col_block_init * 32 +
                                                              state_vec_init * 4);
                state_init_frag[state_vec_init * 4 + 0] = _v4.x;
                state_init_frag[state_vec_init * 4 + 1] = _v4.y;
                state_init_frag[state_vec_init * 4 + 2] = _v4.z;
                state_init_frag[state_vec_init * 4 + 3] = _v4.w;
              }
            }
            int state_init_addr = taddr + (unsigned int)state_tmem_row_base_init +
                                  (unsigned int)(state_col_block_init * 32);
            tmem_st_x32_f32(state_init_addr, state_init_frag);
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(initial_state_loaded_addr);
          }
          mbarrier_wait(initial_state_loaded_addr, _phase_initial_state_loaded_0);
          _phase_initial_state_loaded_0 ^= 1;
          if (state_warp_init == 0) {
            if (elect_sync()) {
              mbarrier_arrive(kv_acc_full_addr);
            }
          }
        }
        if (num_chunks_b_1 > 0) {
          {
            int chunk_offset_1 = batch_start_1;
            int _cg1_marker = batch_idx_1 + head_idx_1 + chunk_offset_1 + batch_end_1 +
                              checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE +
                              ENABLE_CHECKPOINTS;
            mbarrier_wait(load_gate_full_addr + (gate_cg1_stage) * 8, gate_cg1_phase);
            int gate_cg1_elem_base = gate_cg1_stage * 128;
            int warp_in_wg_1 = warp % 4;
            int tmem_row_base_v = warp_in_wg_1 * 32 << 16;
            {
              mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
              _phase_kv_acc_full_0 ^= 1;
              int state_addr_0 = taddr + (unsigned int)tmem_row_base_v;
              int state_addr_1 = state_addr_0 + 32;
              int state_addr_2 = state_addr_0 + 64;
              int state_addr_3 = state_addr_0 + 96;
              float _tmem_load_4[32];
              tmem_ld_x32(&_tmem_load_4[0], state_addr_0);
              float _tmem_load_5[32];
              tmem_ld_x32(&_tmem_load_5[0], state_addr_1);
              float _tmem_load_6[32];
              tmem_ld_x32(&_tmem_load_6[0], state_addr_2);
              float _tmem_load_7[32];
              tmem_ld_x32(&_tmem_load_7[0], state_addr_3);
              int state_inp_addr_0 = taddr + 192 + (unsigned int)tmem_row_base_v;
              int state_inp_addr_1 = state_inp_addr_0 + 16;
              int state_inp_addr_2 = state_inp_addr_0 + 32;
              int state_inp_addr_3 = state_inp_addr_0 + 48;
              uint32_t _tmem_load_4_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_4[_lp * 2 + 0], _tmem_load_4[_lp * 2 + 1 + 0]));
                _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
              }
              asm volatile(
                  "tcgen05.st.sync.aligned.32x32b.x16.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(state_inp_addr_0),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[15])));
              uint32_t _tmem_load_5_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_5[_lp * 2 + 0], _tmem_load_5[_lp * 2 + 1 + 0]));
                _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
              }
              asm volatile(
                  "tcgen05.st.sync.aligned.32x32b.x16.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(state_inp_addr_1),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[15])));
              uint32_t _tmem_load_6_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_6[_lp * 2 + 0], _tmem_load_6[_lp * 2 + 1 + 0]));
                _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
              }
              asm volatile(
                  "tcgen05.st.sync.aligned.32x32b.x16.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(state_inp_addr_2),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[15])));
              uint32_t _tmem_load_7_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_7[_lp * 2 + 0], _tmem_load_7[_lp * 2 + 1 + 0]));
                _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
              }
              asm volatile(
                  "tcgen05.st.sync.aligned.32x32b.x16.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(state_inp_addr_3),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7_bf16[15])));
#pragma unroll
              for (int state_col_cg1 = 0; state_col_cg1 < 32; state_col_cg1++) {
                _tmem_load_4[state_col_cg1] =
                    _tmem_load_4[state_col_cg1] * smem_g_total[gate_cg1_elem_base + state_col_cg1];
                _tmem_load_5[state_col_cg1] = _tmem_load_5[state_col_cg1] *
                                              smem_g_total[gate_cg1_elem_base + 32 + state_col_cg1];
                _tmem_load_6[state_col_cg1] = _tmem_load_6[state_col_cg1] *
                                              smem_g_total[gate_cg1_elem_base + 64 + state_col_cg1];
                _tmem_load_7[state_col_cg1] = _tmem_load_7[state_col_cg1] *
                                              smem_g_total[gate_cg1_elem_base + 96 + state_col_cg1];
              }
              tmem_st_x32_f32(state_addr_0, _tmem_load_4);
              tmem_st_x32_f32(state_addr_1, _tmem_load_5);
              tmem_st_x32_f32(state_addr_2, _tmem_load_6);
              tmem_st_x32_f32(state_addr_3, _tmem_load_7);
              asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
              mbarrier_arrive(state_inp_ready_addr);
              mbarrier_arrive(kv_acc_empty_addr);
            }
            mbarrier_arrive(gate_cg1_empty_addr + (gate_cg1_stage) * 8);
            gate_cg1_stage += 1;
            if (gate_cg1_stage == 5) {
              gate_cg1_stage = 0;
              gate_cg1_phase ^= 1;
            }
            mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, v_cg1_phase);
            int v_stage_addr_cg1 = smem_v_addr + v_cg1_stage * 16384;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            {
              unsigned int v_frag_lo_cg1[16];
              unsigned int v_frag_hi_cg1[16];
              unsigned int v_ld_bits_cg1[4];
#pragma unroll
              for (int v_frag_repeat_cg1 = 0; v_frag_repeat_cg1 < 8; v_frag_repeat_cg1++) {
                int v_ld_mtx_cg1 = lane / 8;
                int v_ld_token_cg1 = v_frag_repeat_cg1 * 8 + (lane & 7);
                int warp_id_in_role_2 = (warp - 4);
                int v_ld_dv_cg1 = warp_id_in_role_2 * 32 + v_ld_mtx_cg1 * 8;
                int v_d0_cg1 = v_ld_dv_cg1 & 63;
                int v_d1_cg1 = v_ld_dv_cg1 / 64;
                int v_t0_cg1 = v_ld_token_cg1 & 15;
                int v_t1_cg1 = v_ld_token_cg1 / 16;
                int v_ld_byte_off_cg1 =
                    (v_d0_cg1 + v_d1_cg1 * 4096 + v_t0_cg1 * 64 + v_t1_cg1 * 1024) * 2;
                v_ld_byte_off_cg1 = v_ld_byte_off_cg1 ^ (v_ld_byte_off_cg1 >> 7 & 7) << 4;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(v_ld_bits_cg1[0]), "=r"(v_ld_bits_cg1[1]), "=r"(v_ld_bits_cg1[2]),
                      "=r"(v_ld_bits_cg1[3])
                    : "r"(v_stage_addr_cg1 + v_ld_byte_off_cg1)
                    : "memory");
                const int v_frag_j0_cg1 = v_frag_repeat_cg1 * 2;
                v_frag_lo_cg1[v_frag_j0_cg1] = v_ld_bits_cg1[0];
                v_frag_lo_cg1[v_frag_j0_cg1 + 1] = v_ld_bits_cg1[1];
                v_frag_hi_cg1[v_frag_j0_cg1] = v_ld_bits_cg1[2];
                v_frag_hi_cg1[v_frag_j0_cg1 + 1] = v_ld_bits_cg1[3];
              }
              mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
              _phase_cg1_shared_acc_full_0 ^= 1;
              int ks_addr_lo_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
              int ks_addr_hi_cg1 = ks_addr_lo_cg1 + 1048576;
              float _tmem_load_8[32];
              asm volatile(
                  "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                  " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                  "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                  "[%32];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[16])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[17])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[18])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[19])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[20])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[21])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[22])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[23])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[24])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[25])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[26])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[27])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[28])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[29])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[30])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[31]))
                  : "r"(ks_addr_lo_cg1));
              float _tmem_load_9[32];
              asm volatile(
                  "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                  " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                  "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                  "[%32];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[16])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[17])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[18])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[19])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[20])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[21])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[22])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[23])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[24])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[25])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[26])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[27])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[28])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[29])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[30])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[31]))
                  : "r"(ks_addr_hi_cg1));
              mbarrier_arrive(cg1_shared_acc_empty_addr);
              uint32_t _tmem_load_8_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_8[_lp * 2 + 0], _tmem_load_8[_lp * 2 + 1 + 0]));
                _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
              }
              uint32_t _tmem_load_9_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_9[_lp * 2 + 0], _tmem_load_9[_lp * 2 + 1 + 0]));
                _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
              }
              unsigned int vks_packed_lo_cg1[16];
              unsigned int vks_packed_hi_cg1[16];
#pragma unroll
              for (int frag_pair_j = 0; frag_pair_j < 16; frag_pair_j++) {
                uint32_t _bf16x2_sub_0;
                asm volatile("sub.rn.bf16x2 %0, %1, %2;"
                             : "=r"(_bf16x2_sub_0)
                             : "r"(v_frag_lo_cg1[frag_pair_j]),
                               "r"(_tmem_load_8_bf16[frag_pair_j]));
                vks_packed_lo_cg1[frag_pair_j] = _bf16x2_sub_0;
                uint32_t _bf16x2_sub_1;
                asm volatile("sub.rn.bf16x2 %0, %1, %2;"
                             : "=r"(_bf16x2_sub_1)
                             : "r"(v_frag_hi_cg1[frag_pair_j]),
                               "r"(_tmem_load_9_bf16[frag_pair_j]));
                vks_packed_hi_cg1[frag_pair_j] = _bf16x2_sub_1;
              }
              int vks_addr_lo_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
              int vks_addr_hi_cg1 = vks_addr_lo_cg1 + 1048576;
              asm volatile(
                  "tcgen05.st.sync.aligned.16x128b.x8.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(vks_addr_lo_cg1),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[15])));
              asm volatile(
                  "tcgen05.st.sync.aligned.16x128b.x8.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                  "%16};" ::"r"(vks_addr_hi_cg1),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[15])));
              asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            mbarrier_arrive(vks_ready_addr);
            {
              mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
              _phase_q_state_acc_full_0 ^= 1;
#pragma unroll
              for (int dim_half_qs = 0; dim_half_qs < 2; dim_half_qs++) {
                int qs_addr = taddr + 128 + (unsigned int)tmem_row_base_v +
                              (unsigned int)(dim_half_qs * 16 << 16);
                float _tmem_load_10[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                    "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                    "[%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[16])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[17])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[18])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[19])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[20])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[21])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[22])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[23])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[24])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[25])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[26])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[27])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[28])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[29])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[30])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[31]))
                    : "r"(qs_addr));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
                    "%31, %32};" ::"r"(qs_addr),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[0])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[1])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[2])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[3])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[4])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[5])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[6])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[7])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[8])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[9])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[10])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[11])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[12])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[13])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[14])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[15])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[16])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[17])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[18])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[19])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[20])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[21])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[22])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[23])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[24])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[25])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[26])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[27])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[28])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[29])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[30])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_10[31])));
              }
              asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
              mbarrier_arrive(q_state_acc_empty_addr);
            }
            mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
            _phase_cg1_shared_acc_full_0 ^= 1;
            if (elect_sync()) {
              mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
            }
            v_cg1_stage += 1;
            if (v_cg1_stage == 2) {
              v_cg1_stage = 0;
              v_cg1_phase ^= 1;
            }
            int nv_src_addr_lo_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
            float _tmem_load_11[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[16])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[17])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[18])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[19])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[20])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[21])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[22])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[23])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[24])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[25])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[26])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[27])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[28])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[29])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[30])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[31]))
                : "r"(nv_src_addr_lo_cg1));
            uint32_t _tmem_load_11_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_11[_lp * 2 + 0], _tmem_load_11[_lp * 2 + 1 + 0]));
              _tmem_load_11_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int nv_dst_addr_lo_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(nv_dst_addr_lo_cg1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[15])));
            int nv_src_addr_hi_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v + 1048576;
            float _tmem_load_12[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[16])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[17])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[18])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[19])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[20])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[21])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[22])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[23])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[24])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[25])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[26])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[27])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[28])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[29])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[30])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[31]))
                : "r"(nv_src_addr_hi_cg1));
            uint32_t _tmem_load_12_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_12[_lp * 2 + 0], _tmem_load_12[_lp * 2 + 1 + 0]));
              _tmem_load_12_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            mbarrier_arrive(cg1_shared_acc_empty_addr);
            int nv_dst_addr_hi_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v + 1048576;
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(nv_dst_addr_hi_cg1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[15])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(nv_ready_addr);
            int decay_dst_addr_lo_cg1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v;
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(decay_dst_addr_lo_cg1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_bf16[15])));
            int decay_dst_addr_hi_cg1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v + 1048576;
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(decay_dst_addr_hi_cg1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_12_bf16[15])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(decay_v_ready_addr);
            mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, o_cg1_phase);
            int o_stage_addr_cg1 = smem_o_addr + o_cg1_stage * 16384;
            mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
            _phase_q_state_acc_full_0 ^= 1;
#pragma unroll
            for (int dim_half_cg1 = 0; dim_half_cg1 < 2; dim_half_cg1++) {
              int dim_half_row_cg1 = dim_half_cg1 * 16;
              int q_state_addr = taddr + 128 + (unsigned int)tmem_row_base_v +
                                 (unsigned int)(dim_half_row_cg1 << 16);
              float _tmem_load_13[32];
              asm volatile(
                  "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                  " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                  "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                  "[%32];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[16])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[17])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[18])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[19])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[20])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[21])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[22])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[23])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[24])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[25])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[26])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[27])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[28])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[29])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[30])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[31]))
                  : "r"(q_state_addr));
              uint32_t _tmem_load_13_bf16[16];
#pragma unroll
              for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                    make_float2(_tmem_load_13[_lp * 2 + 0], _tmem_load_13[_lp * 2 + 1 + 0]));
                _tmem_load_13_bf16[_lp] = *(uint32_t*)&_bf2;
              }
#pragma unroll
              for (int token_group_cg1 = 0; token_group_cg1 < 4; token_group_cg1++) {
                int o_mtx_idx_cg1 = lane / 8;
                int o_row_addr_cg1 = lane & 7;
                int warp_id_in_role_3 = (warp - 4);
                int o_dim_base_cg1 =
                    warp_id_in_role_3 * 32 + dim_half_row_cg1 + (o_mtx_idx_cg1 & 1) * 8;
                int o_token_base_cg1 = token_group_cg1 * 16 + o_mtx_idx_cg1 / 2 * 8;
                int o_token_addr_cg1 = o_token_base_cg1 + o_row_addr_cg1;
                int o_token_pair_cg1 = o_token_addr_cg1 / 2;
                int o_token_parity_cg1 = o_token_addr_cg1 & 1;
                int o_raw_row_cg1 = o_token_pair_cg1 + o_dim_base_cg1 / 64 * 32;
                int o_raw_col_cg1 =
                    (o_dim_base_cg1 & 63 ^ (o_token_pair_cg1 & 3) << 4 ^ o_token_parity_cg1 << 3) +
                    o_token_parity_cg1 * 64;
                int o_stsm_offset_cg1 = (o_raw_row_cg1 * 128 + o_raw_col_cg1) * 2;
                const int o_pack_base_cg1 = token_group_cg1 * 4;
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(
                    (unsigned long long)(o_stage_addr_cg1 + o_stsm_offset_cg1));
                asm volatile(
                    "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::
                        "r"(_stmatrix_addr_1),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[o_pack_base_cg1])),
                    "r"(*reinterpret_cast<const uint32_t*>(
                        &_tmem_load_13_bf16[o_pack_base_cg1 + 1])),
                    "r"(*reinterpret_cast<const uint32_t*>(
                        &_tmem_load_13_bf16[o_pack_base_cg1 + 2])),
                    "r"(*reinterpret_cast<const uint32_t*>(
                        &_tmem_load_13_bf16[o_pack_base_cg1 + 3]))
                    : "memory");
              }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_state_acc_empty_addr);
            mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
            o_cg1_stage += 1;
            if (o_cg1_stage == 2) {
              o_cg1_stage = 0;
              o_cg1_phase ^= 1;
            }
          }
        }
#pragma unroll 1
        for (int chunk_idx_1 = 1; chunk_idx_1 < num_chunks_b_1; chunk_idx_1++) {
          int chunk_offset_2 = batch_start_1 + chunk_idx_1 * 16;
          int _cg1_marker_1 = batch_idx_1 + head_idx_1 + chunk_offset_2 + batch_end_1 +
                              checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE +
                              ENABLE_CHECKPOINTS;
          mbarrier_wait(load_gate_full_addr + (gate_cg1_stage) * 8, gate_cg1_phase);
          int gate_cg1_elem_base_1 = gate_cg1_stage * 128;
          int warp_in_wg_2 = warp % 4;
          int tmem_row_base_v_1 = warp_in_wg_2 * 32 << 16;
          {
            mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
            _phase_kv_acc_full_0 ^= 1;
            int state_addr_0_1 = taddr + (unsigned int)tmem_row_base_v_1;
            int state_addr_1_1 = state_addr_0_1 + 32;
            int state_addr_2_1 = state_addr_0_1 + 64;
            int state_addr_3_1 = state_addr_0_1 + 96;
            float _tmem_load_24[32];
            tmem_ld_x32(&_tmem_load_24[0], state_addr_0_1);
            float _tmem_load_25[32];
            tmem_ld_x32(&_tmem_load_25[0], state_addr_1_1);
            float _tmem_load_26[32];
            tmem_ld_x32(&_tmem_load_26[0], state_addr_2_1);
            float _tmem_load_27[32];
            tmem_ld_x32(&_tmem_load_27[0], state_addr_3_1);
            int state_inp_addr_0_1 = taddr + 192 + (unsigned int)tmem_row_base_v_1;
            int state_inp_addr_1_1 = state_inp_addr_0_1 + 16;
            int state_inp_addr_2_1 = state_inp_addr_0_1 + 32;
            int state_inp_addr_3_1 = state_inp_addr_0_1 + 48;
            uint32_t _tmem_load_24_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_24[_lp * 2 + 0], _tmem_load_24[_lp * 2 + 1 + 0]));
              _tmem_load_24_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(state_inp_addr_0_1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_24_bf16[15])));
            uint32_t _tmem_load_25_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_25[_lp * 2 + 0], _tmem_load_25[_lp * 2 + 1 + 0]));
              _tmem_load_25_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(state_inp_addr_1_1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_25_bf16[15])));
            uint32_t _tmem_load_26_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_26[_lp * 2 + 0], _tmem_load_26[_lp * 2 + 1 + 0]));
              _tmem_load_26_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(state_inp_addr_2_1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_26_bf16[15])));
            uint32_t _tmem_load_27_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_27[_lp * 2 + 0], _tmem_load_27[_lp * 2 + 1 + 0]));
              _tmem_load_27_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(state_inp_addr_3_1),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_27_bf16[15])));
#pragma unroll
            for (int state_col_cg1_1 = 0; state_col_cg1_1 < 32; state_col_cg1_1++) {
              _tmem_load_24[state_col_cg1_1] = _tmem_load_24[state_col_cg1_1] *
                                               smem_g_total[gate_cg1_elem_base_1 + state_col_cg1_1];
              _tmem_load_25[state_col_cg1_1] =
                  _tmem_load_25[state_col_cg1_1] *
                  smem_g_total[gate_cg1_elem_base_1 + 32 + state_col_cg1_1];
              _tmem_load_26[state_col_cg1_1] =
                  _tmem_load_26[state_col_cg1_1] *
                  smem_g_total[gate_cg1_elem_base_1 + 64 + state_col_cg1_1];
              _tmem_load_27[state_col_cg1_1] =
                  _tmem_load_27[state_col_cg1_1] *
                  smem_g_total[gate_cg1_elem_base_1 + 96 + state_col_cg1_1];
            }
            tmem_st_x32_f32(state_addr_0_1, _tmem_load_24);
            tmem_st_x32_f32(state_addr_1_1, _tmem_load_25);
            tmem_st_x32_f32(state_addr_2_1, _tmem_load_26);
            tmem_st_x32_f32(state_addr_3_1, _tmem_load_27);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(state_inp_ready_addr);
            mbarrier_arrive(kv_acc_empty_addr);
          }
          mbarrier_arrive(gate_cg1_empty_addr + (gate_cg1_stage) * 8);
          gate_cg1_stage += 1;
          if (gate_cg1_stage == 5) {
            gate_cg1_stage = 0;
            gate_cg1_phase ^= 1;
          }
          mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, v_cg1_phase);
          int v_stage_addr_cg1_1 = smem_v_addr + v_cg1_stage * 16384;
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          {
            unsigned int v_frag_lo_cg1_1[16];
            unsigned int v_frag_hi_cg1_1[16];
            unsigned int v_ld_bits_cg1_1[4];
#pragma unroll
            for (int v_frag_repeat_cg1_1 = 0; v_frag_repeat_cg1_1 < 8; v_frag_repeat_cg1_1++) {
              int v_ld_mtx_cg1_1 = lane / 8;
              int v_ld_token_cg1_1 = v_frag_repeat_cg1_1 * 8 + (lane & 7);
              int warp_id_in_role_4 = (warp - 4);
              int v_ld_dv_cg1_1 = warp_id_in_role_4 * 32 + v_ld_mtx_cg1_1 * 8;
              int v_d0_cg1_1 = v_ld_dv_cg1_1 & 63;
              int v_d1_cg1_1 = v_ld_dv_cg1_1 / 64;
              int v_t0_cg1_1 = v_ld_token_cg1_1 & 15;
              int v_t1_cg1_1 = v_ld_token_cg1_1 / 16;
              int v_ld_byte_off_cg1_1 =
                  (v_d0_cg1_1 + v_d1_cg1_1 * 4096 + v_t0_cg1_1 * 64 + v_t1_cg1_1 * 1024) * 2;
              v_ld_byte_off_cg1_1 = v_ld_byte_off_cg1_1 ^ (v_ld_byte_off_cg1_1 >> 7 & 7) << 4;
              asm volatile(
                  "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(v_ld_bits_cg1_1[0]), "=r"(v_ld_bits_cg1_1[1]), "=r"(v_ld_bits_cg1_1[2]),
                    "=r"(v_ld_bits_cg1_1[3])
                  : "r"(v_stage_addr_cg1_1 + v_ld_byte_off_cg1_1)
                  : "memory");
              const int v_frag_j0_cg1_1 = v_frag_repeat_cg1_1 * 2;
              v_frag_lo_cg1_1[v_frag_j0_cg1_1] = v_ld_bits_cg1_1[0];
              v_frag_lo_cg1_1[v_frag_j0_cg1_1 + 1] = v_ld_bits_cg1_1[1];
              v_frag_hi_cg1_1[v_frag_j0_cg1_1] = v_ld_bits_cg1_1[2];
              v_frag_hi_cg1_1[v_frag_j0_cg1_1 + 1] = v_ld_bits_cg1_1[3];
            }
            mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
            _phase_cg1_shared_acc_full_0 ^= 1;
            int ks_addr_lo_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
            int ks_addr_hi_cg1_1 = ks_addr_lo_cg1_1 + 1048576;
            float _tmem_load_28[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[3])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[4])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[5])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[6])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[7])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[8])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[9])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[10])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[11])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[12])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[13])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[14])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[15])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[16])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[17])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[18])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[19])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[20])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[21])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[22])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[23])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[24])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[25])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[26])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[27])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[28])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[29])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[30])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_28[31]))
                : "r"(ks_addr_lo_cg1_1));
            float _tmem_load_29[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[3])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[4])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[5])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[6])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[7])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[8])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[9])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[10])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[11])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[12])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[13])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[14])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[15])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[16])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[17])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[18])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[19])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[20])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[21])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[22])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[23])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[24])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[25])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[26])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[27])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[28])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[29])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[30])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_29[31]))
                : "r"(ks_addr_hi_cg1_1));
            mbarrier_arrive(cg1_shared_acc_empty_addr);
            uint32_t _tmem_load_28_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_28[_lp * 2 + 0], _tmem_load_28[_lp * 2 + 1 + 0]));
              _tmem_load_28_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _tmem_load_29_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_29[_lp * 2 + 0], _tmem_load_29[_lp * 2 + 1 + 0]));
              _tmem_load_29_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int vks_packed_lo_cg1_1[16];
            unsigned int vks_packed_hi_cg1_1[16];
#pragma unroll
            for (int frag_pair_j_1 = 0; frag_pair_j_1 < 16; frag_pair_j_1++) {
              uint32_t _bf16x2_sub_4;
              asm volatile("sub.rn.bf16x2 %0, %1, %2;"
                           : "=r"(_bf16x2_sub_4)
                           : "r"(v_frag_lo_cg1_1[frag_pair_j_1]),
                             "r"(_tmem_load_28_bf16[frag_pair_j_1]));
              vks_packed_lo_cg1_1[frag_pair_j_1] = _bf16x2_sub_4;
              uint32_t _bf16x2_sub_5;
              asm volatile("sub.rn.bf16x2 %0, %1, %2;"
                           : "=r"(_bf16x2_sub_5)
                           : "r"(v_frag_hi_cg1_1[frag_pair_j_1]),
                             "r"(_tmem_load_29_bf16[frag_pair_j_1]));
              vks_packed_hi_cg1_1[frag_pair_j_1] = _bf16x2_sub_5;
            }
            int vks_addr_lo_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
            int vks_addr_hi_cg1_1 = vks_addr_lo_cg1_1 + 1048576;
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(vks_addr_lo_cg1_1),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[15])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                    "r"(vks_addr_hi_cg1_1),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[0])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[1])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[2])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[3])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[4])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[5])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[6])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[7])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[8])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[9])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[10])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[11])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[12])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[13])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[14])),
                "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[15])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          }
          mbarrier_arrive(vks_ready_addr);
          {
            mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
            _phase_q_state_acc_full_0 ^= 1;
#pragma unroll
            for (int dim_half_qs_1 = 0; dim_half_qs_1 < 2; dim_half_qs_1++) {
              int qs_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1 +
                              (unsigned int)(dim_half_qs_1 * 16 << 16);
              float _tmem_load_30[32];
              asm volatile(
                  "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                  " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                  "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                  "[%32];"
                  : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[0])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[1])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[2])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[3])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[4])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[5])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[6])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[7])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[8])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[9])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[10])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[11])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[12])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[13])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[14])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[15])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[16])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[17])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[18])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[19])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[20])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[21])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[22])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[23])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[24])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[25])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[26])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[27])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[28])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[29])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[30])),
                    "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_30[31]))
                  : "r"(qs_addr_1));
              asm volatile(
                  "tcgen05.st.sync.aligned.16x256b.x8.b32"
                  " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                  "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
                  "%32};" ::"r"(qs_addr_1),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[0])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[1])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[2])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[3])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[4])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[5])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[6])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[7])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[8])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[9])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[10])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[11])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[12])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[13])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[14])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[15])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[16])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[17])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[18])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[19])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[20])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[21])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[22])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[23])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[24])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[25])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[26])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[27])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[28])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[29])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[30])),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_30[31])));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(q_state_acc_empty_addr);
          }
          mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
          _phase_cg1_shared_acc_full_0 ^= 1;
          if (elect_sync()) {
            mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
          }
          v_cg1_stage += 1;
          if (v_cg1_stage == 2) {
            v_cg1_stage = 0;
            v_cg1_phase ^= 1;
          }
          int nv_src_addr_lo_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
          float _tmem_load_31[32];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x8.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_31[31]))
              : "r"(nv_src_addr_lo_cg1_1));
          uint32_t _tmem_load_31_bf16[16];
#pragma unroll
          for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_31[_lp * 2 + 0], _tmem_load_31[_lp * 2 + 1 + 0]));
            _tmem_load_31_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          int nv_dst_addr_lo_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x8.b32"
              " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                  "r"(nv_dst_addr_lo_cg1_1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[3])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[4])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[5])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[6])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[7])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[8])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[9])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[10])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[11])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[12])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[13])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[14])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[15])));
          int nv_src_addr_hi_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1 + 1048576;
          float _tmem_load_32[32];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x8.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_32[31]))
              : "r"(nv_src_addr_hi_cg1_1));
          uint32_t _tmem_load_32_bf16[16];
#pragma unroll
          for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                make_float2(_tmem_load_32[_lp * 2 + 0], _tmem_load_32[_lp * 2 + 1 + 0]));
            _tmem_load_32_bf16[_lp] = *(uint32_t*)&_bf2;
          }
          mbarrier_arrive(cg1_shared_acc_empty_addr);
          int nv_dst_addr_hi_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1 + 1048576;
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x8.b32"
              " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                  "r"(nv_dst_addr_hi_cg1_1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[3])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[4])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[5])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[6])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[7])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[8])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[9])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[10])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[11])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[12])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[13])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[14])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[15])));
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          mbarrier_arrive(nv_ready_addr);
          int decay_dst_addr_lo_cg1_1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v_1;
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x8.b32"
              " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                  "r"(decay_dst_addr_lo_cg1_1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[3])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[4])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[5])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[6])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[7])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[8])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[9])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[10])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[11])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[12])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[13])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[14])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_31_bf16[15])));
          int decay_dst_addr_hi_cg1_1 =
              taddr + 448 + 32 + (unsigned int)tmem_row_base_v_1 + 1048576;
          asm volatile(
              "tcgen05.st.sync.aligned.16x128b.x8.b32"
              " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::
                  "r"(decay_dst_addr_hi_cg1_1),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[0])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[1])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[2])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[3])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[4])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[5])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[6])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[7])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[8])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[9])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[10])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[11])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[12])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[13])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[14])),
              "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_32_bf16[15])));
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          mbarrier_arrive(decay_v_ready_addr);
          mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, o_cg1_phase);
          int o_stage_addr_cg1_1 = smem_o_addr + o_cg1_stage * 16384;
          mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
          _phase_q_state_acc_full_0 ^= 1;
#pragma unroll
          for (int dim_half_cg1_1 = 0; dim_half_cg1_1 < 2; dim_half_cg1_1++) {
            int dim_half_row_cg1_1 = dim_half_cg1_1 * 16;
            int q_state_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1 +
                                 (unsigned int)(dim_half_row_cg1_1 << 16);
            float _tmem_load_33[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
                "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[0])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[1])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[2])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[3])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[4])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[5])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[6])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[7])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[8])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[9])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[10])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[11])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[12])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[13])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[14])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[15])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[16])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[17])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[18])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[19])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[20])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[21])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[22])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[23])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[24])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[25])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[26])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[27])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[28])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[29])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[30])),
                  "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_33[31]))
                : "r"(q_state_addr_1));
            uint32_t _tmem_load_33_bf16[16];
#pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
              __nv_bfloat162 _bf2 = __float22bfloat162_rn(
                  make_float2(_tmem_load_33[_lp * 2 + 0], _tmem_load_33[_lp * 2 + 1 + 0]));
              _tmem_load_33_bf16[_lp] = *(uint32_t*)&_bf2;
            }
#pragma unroll
            for (int token_group_cg1_1 = 0; token_group_cg1_1 < 4; token_group_cg1_1++) {
              int o_mtx_idx_cg1_1 = lane / 8;
              int o_row_addr_cg1_1 = lane & 7;
              int warp_id_in_role_5 = (warp - 4);
              int o_dim_base_cg1_1 =
                  warp_id_in_role_5 * 32 + dim_half_row_cg1_1 + (o_mtx_idx_cg1_1 & 1) * 8;
              int o_token_base_cg1_1 = token_group_cg1_1 * 16 + o_mtx_idx_cg1_1 / 2 * 8;
              int o_token_addr_cg1_1 = o_token_base_cg1_1 + o_row_addr_cg1_1;
              int o_token_pair_cg1_1 = o_token_addr_cg1_1 / 2;
              int o_token_parity_cg1_1 = o_token_addr_cg1_1 & 1;
              int o_raw_row_cg1_1 = o_token_pair_cg1_1 + o_dim_base_cg1_1 / 64 * 32;
              int o_raw_col_cg1_1 = (o_dim_base_cg1_1 & 63 ^ (o_token_pair_cg1_1 & 3) << 4 ^
                                     o_token_parity_cg1_1 << 3) +
                                    o_token_parity_cg1_1 * 64;
              int o_stsm_offset_cg1_1 = (o_raw_row_cg1_1 * 128 + o_raw_col_cg1_1) * 2;
              const int o_pack_base_cg1_1 = token_group_cg1_1 * 4;
              uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(
                  (unsigned long long)(o_stage_addr_cg1_1 + o_stsm_offset_cg1_1));
              asm volatile(
                  "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                      _stmatrix_addr_2),
                  "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_33_bf16[o_pack_base_cg1_1])),
                  "r"(*reinterpret_cast<const uint32_t*>(
                      &_tmem_load_33_bf16[o_pack_base_cg1_1 + 1])),
                  "r"(*reinterpret_cast<const uint32_t*>(
                      &_tmem_load_33_bf16[o_pack_base_cg1_1 + 2])),
                  "r"(*reinterpret_cast<const uint32_t*>(
                      &_tmem_load_33_bf16[o_pack_base_cg1_1 + 3]))
                  : "memory");
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(q_state_acc_empty_addr);
          mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
          o_cg1_stage += 1;
          if (o_cg1_stage == 2) {
            o_cg1_stage = 0;
            o_cg1_phase ^= 1;
          }
        }
        if (STORE_FINAL_STATE != 0 && num_chunks_b_1 > 0) {
          mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
          _phase_kv_acc_full_0 ^= 1;
          int warp_in_wg_3 = warp % 4;
          int state_tmem_row_base_cg1 = warp_in_wg_3 * 32 << 16;
          int warp_id_in_role_6 = (warp - 4);
          int state_row_cg1 = warp_id_in_role_6 * 32 + lane;
          int state_base_idx_cg1 =
              (batch_idx_1 * num_o_heads_1 + head_idx_1) * 128 * 128 + state_row_cg1 * 128;
#pragma unroll
          for (int state_col_block_cg1 = 0; state_col_block_cg1 < 4; state_col_block_cg1++) {
            int final_state_addr_cg1 = taddr + (unsigned int)state_tmem_row_base_cg1 +
                                       (unsigned int)(state_col_block_cg1 * 32);
            float _tmem_load_34[32];
            tmem_ld_x32(&_tmem_load_34[0], final_state_addr_cg1);
#pragma unroll
            for (int state_vec_idx_cg1 = 0; state_vec_idx_cg1 < 8; state_vec_idx_cg1++) {
              asm volatile(
                  "st.global.L1::no_allocate.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                      output_state +
                      (state_base_idx_cg1 + state_col_block_cg1 * 32 + state_vec_idx_cg1 * 4) + 0),
                  "f"(_tmem_load_34[state_vec_idx_cg1 * 4 + 0]),
                  "f"(_tmem_load_34[state_vec_idx_cg1 * 4 + 1]),
                  "f"(_tmem_load_34[state_vec_idx_cg1 * 4 + 2]),
                  "f"(_tmem_load_34[state_vec_idx_cg1 * 4 + 3])
                  : "memory");
            }
          }
          mbarrier_arrive(kv_acc_empty_addr);
        }
        if (STORE_FINAL_STATE != 0 && USE_INITIAL_STATE != 0 && num_chunks_b_1 == 0) {
          int warp_id_in_role_7 = (warp - 4);
          int empty_state_row = warp_id_in_role_7 * 32 + lane;
          int empty_state_base =
              ((batch_idx_1 * num_o_heads_1 + head_idx_1) * 128 + empty_state_row) * 128;
#pragma unroll
          for (int empty_state_vec = 0; empty_state_vec < 32; empty_state_vec++) {
            int empty_state_col = empty_state_vec * 4;
            float _vec_load_0[4];
            {
              float4 _v4 = *reinterpret_cast<const float4*>(initial_state + empty_state_base +
                                                            empty_state_col);
              _vec_load_0[0 + 0] = _v4.x;
              _vec_load_0[0 + 1] = _v4.y;
              _vec_load_0[0 + 2] = _v4.z;
              _vec_load_0[0 + 3] = _v4.w;
            }
            {
              float4 _v4 = make_float4(_vec_load_0[0 + 0], _vec_load_0[0 + 1], _vec_load_0[0 + 2],
                                       _vec_load_0[0 + 3]);
              *reinterpret_cast<float4*>(output_state + (empty_state_base + empty_state_col) + 0) =
                  _v4;
            }
          }
        }
      }
    }
  }
  // ---- Role: mma ----
  if (warp == 8) {
    {  // mma_main
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(Q)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(K)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(V)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(G)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)(O)) : "memory");
      unsigned int k_cg0_stage = 0;
      unsigned int k_cg0_phase = 0;
      unsigned int q_cg0_stage = 0;
      unsigned int q_cg0_phase = 0;
      unsigned int g_cg0_stage = 0;
      unsigned int g_cg0_phase = 0;
#pragma unroll 1
      for (unsigned int tile_2 = bid; tile_2 < total_tiles; tile_2 += num_bids) {
        int num_o_heads_2 = ((num_q_heads >= num_v_heads) ? num_q_heads : num_v_heads);
        int batch_idx_2 = tile_2 / (unsigned int)num_o_heads_2;
        int head_idx_2 = tile_2 % (unsigned int)num_o_heads_2;
        int qk_head_idx_2 =
            ((num_q_heads >= num_v_heads) ? head_idx_2 : head_idx_2 / (num_v_heads / num_q_heads));
        int v_head_idx_2 =
            ((num_v_heads >= num_q_heads) ? head_idx_2 : head_idx_2 / (num_q_heads / num_v_heads));
        int batch_start_2 = (int)cu_seqlens[batch_idx_2];
        int batch_end_2 = (int)cu_seqlens[batch_idx_2 + 1];
        int seqlen_b_2 = batch_end_2 - batch_start_2;
        int num_pairs_b_2 = (seqlen_b_2 + 32 - 1) / 32;
        int num_chunks_b_2 = num_pairs_b_2 * 2;
        int num_pairs_b_0 = num_chunks_b_2 / 2;
#pragma unroll 1
        for (int _pair_idx = 0; _pair_idx < num_pairs_b_0; _pair_idx++) {
          unsigned int k0_stage_1 = k_cg0_stage;
          k_cg0_stage += 1;
          if (k_cg0_stage == 3) {
            k_cg0_stage = 0;
            k_cg0_phase ^= 1;
          }
          unsigned int q0_stage_1 = q_cg0_stage;
          q_cg0_stage += 1;
          if (q_cg0_stage == 2) {
            q_cg0_stage = 0;
            q_cg0_phase ^= 1;
          }
          unsigned int g0_stage_1 = g_cg0_stage;
          mbarrier_wait(qkg_ready_addr + (g0_stage_1) * 8, g_cg0_phase);
          g_cg0_stage += 1;
          if (g_cg0_stage == 2) {
            g_cg0_stage = 0;
            g_cg0_phase ^= 1;
          }
          mbarrier_wait(cg0_shared_acc_empty_addr, 1);
          int _mma_a_lo_0 =
              make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k0_stage_1) * 1024);
          int _mma_b_lo_0 =
              make_warp_uniform((((smem_g_addr) >> 4) & 0x3FFF) + (g0_stage_1) * 1024);
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
              "mov.b32 id, 68158608;\n\t"
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
              "add.u32 alo, alo, 506;\n\t"
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
              "}\n" ::"r"(_mma_a_lo_0),
              "r"(_mma_b_lo_0), "r"(tmem_tmem_cg0_shared_acc), "r"(0));
          elect_commit(cg0_shared_acc_full_addr);
          unsigned int k1_stage_1 = k_cg0_stage;
          k_cg0_stage += 1;
          if (k_cg0_stage == 3) {
            k_cg0_stage = 0;
            k_cg0_phase ^= 1;
          }
          unsigned int q1_stage_1 = q_cg0_stage;
          q_cg0_stage += 1;
          if (q_cg0_stage == 2) {
            q_cg0_stage = 0;
            q_cg0_phase ^= 1;
          }
          unsigned int g1_stage_1 = g_cg0_stage;
          mbarrier_wait(qkg_ready_addr + (g1_stage_1) * 8, g_cg0_phase);
          g_cg0_stage += 1;
          if (g_cg0_stage == 2) {
            g_cg0_stage = 0;
            g_cg0_phase ^= 1;
          }
          mbarrier_wait(cg0_shared_acc_empty_addr + 8, 1);
          int _mma_a_lo_1 =
              make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k1_stage_1) * 1024);
          int _mma_b_lo_1 =
              make_warp_uniform((((smem_g_addr) >> 4) & 0x3FFF) + (g1_stage_1) * 1024);
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
              "mov.b32 id, 68158608;\n\t"
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
              "add.u32 alo, alo, 506;\n\t"
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
              "}\n" ::"r"(_mma_a_lo_1),
              "r"(_mma_b_lo_1), "r"((tmem_tmem_cg0_shared_acc + (64))), "r"(0));
          elect_commit(cg0_shared_acc_full_addr + 8);
          mbarrier_wait(cg0_shared_acc_empty_addr, 0);
          int _mma_a_lo_2 =
              make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q0_stage_1) * 1024);
          int _mma_b_lo_2 =
              make_warp_uniform((((smem_g_addr) >> 4) & 0x3FFF) + (g0_stage_1) * 1024);
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
              "mov.b32 id, 68158608;\n\t"
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
              "add.u32 alo, alo, 506;\n\t"
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
              "}\n" ::"r"(_mma_a_lo_2),
              "r"(_mma_b_lo_2), "r"(tmem_tmem_cg0_shared_acc), "r"(0));
          elect_commit(cg0_shared_acc_full_addr);
          elect_commit(ki_mma_consumed_addr + (g0_stage_1) * 8);
          mbarrier_wait(cg0_shared_acc_empty_addr + 8, 0);
          int _mma_a_lo_3 =
              make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q1_stage_1) * 1024);
          int _mma_b_lo_3 =
              make_warp_uniform((((smem_g_addr) >> 4) & 0x3FFF) + (g1_stage_1) * 1024);
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
              "mov.b32 id, 68158608;\n\t"
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
              "add.u32 alo, alo, 506;\n\t"
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
              "}\n" ::"r"(_mma_a_lo_3),
              "r"(_mma_b_lo_3), "r"((tmem_tmem_cg0_shared_acc + (64))), "r"(0));
          elect_commit(cg0_shared_acc_full_addr + 8);
          elect_commit(ki_mma_consumed_addr + (g1_stage_1) * 8);
          elect_commit(load_k_empty_addr + (k0_stage_1) * 8);
          elect_commit(load_k_empty_addr + (k1_stage_1) * 8);
          elect_commit(q_smem_empty_addr + (q0_stage_1) * 8);
          elect_commit(q_smem_empty_addr + (q1_stage_1) * 8);
        }
      }
    }
  }
  // ---- Role: tma_qkv ----
  if (warp == 9) {
    {  // tma_qkv_main
      int cta_slot_base = bid * 1280;
      if (elect_sync()) {
#pragma unroll
        for (int descriptor_stage = 0; descriptor_stage < 2; descriptor_stage++) {
          {
            const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(Q);
            uint64_t* __tm_dst = reinterpret_cast<uint64_t*>(
                (uint64_t)(tensormap_workspace + (cta_slot_base + descriptor_stage * 128)));
#pragma unroll
            for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
              __tm_dst[__tm_i] = __tm_src[__tm_i];
            }
          }
        }
#pragma unroll
        for (int descriptor_stage_1 = 0; descriptor_stage_1 < 3; descriptor_stage_1++) {
          {
            const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(K);
            uint64_t* __tm_dst = reinterpret_cast<uint64_t*>(
                (uint64_t)(tensormap_workspace + (cta_slot_base + 256 + descriptor_stage_1 * 128)));
#pragma unroll
            for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
              __tm_dst[__tm_i] = __tm_src[__tm_i];
            }
          }
        }
#pragma unroll
        for (int descriptor_stage_2 = 0; descriptor_stage_2 < 2; descriptor_stage_2++) {
          {
            const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(V);
            uint64_t* __tm_dst = reinterpret_cast<uint64_t*>(
                (uint64_t)(tensormap_workspace + (cta_slot_base + 640 + descriptor_stage_2 * 128)));
#pragma unroll
            for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
              __tm_dst[__tm_i] = __tm_src[__tm_i];
            }
          }
        }
#pragma unroll
        for (int descriptor_stage_3 = 0; descriptor_stage_3 < 2; descriptor_stage_3++) {
          {
            const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(G);
            uint64_t* __tm_dst = reinterpret_cast<uint64_t*>(
                (uint64_t)(tensormap_workspace + (cta_slot_base + 896 + descriptor_stage_3 * 128)));
#pragma unroll
            for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
              __tm_dst[__tm_i] = __tm_src[__tm_i];
            }
          }
        }
        asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
      }
      unsigned int k_stage = 0;
      unsigned int k_empty_phase_tma = 1;
      unsigned int q_stage = 0;
      unsigned int q_empty_phase_tma = 1;
      unsigned int v_stage = 0;
      unsigned int v_empty_phase_tma = 1;
      unsigned int g_stage = 0;
      unsigned int g_empty_phase_tma = 1;
#pragma unroll 1
      for (unsigned int tile_3 = bid; tile_3 < total_tiles; tile_3 += num_bids) {
        int num_o_heads_3 = ((num_q_heads >= num_v_heads) ? num_q_heads : num_v_heads);
        int batch_idx_3 = tile_3 / (unsigned int)num_o_heads_3;
        int head_idx_3 = tile_3 % (unsigned int)num_o_heads_3;
        int qk_head_idx_3 =
            ((num_q_heads >= num_v_heads) ? head_idx_3 : head_idx_3 / (num_v_heads / num_q_heads));
        int v_head_idx_3 =
            ((num_v_heads >= num_q_heads) ? head_idx_3 : head_idx_3 / (num_q_heads / num_v_heads));
        int batch_start_3 = (int)cu_seqlens[batch_idx_3];
        int batch_end_3 = (int)cu_seqlens[batch_idx_3 + 1];
        int seqlen_b_3 = batch_end_3 - batch_start_3;
        int num_pairs_b_3 = (seqlen_b_3 + 32 - 1) / 32;
        int num_chunks_b_3 = num_pairs_b_3 * 2;
#pragma unroll 1
        for (int chunk_idx_2 = 0; chunk_idx_2 < num_chunks_b_3; chunk_idx_2++) {
          int chunk_offset_3 = batch_start_3 + chunk_idx_2 * 16;
          int chunk_end = chunk_offset_3 + 16;
          if (chunk_end > batch_end_3) {
            chunk_end = batch_end_3;
          }
          if (elect_sync()) {
            mbarrier_wait(load_k_empty_addr + (k_stage) * 8, k_empty_phase_tma);
            asm volatile(
                "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(
                    (uint64_t)(tensormap_workspace + (cta_slot_base + 256 + k_stage * 128))),
                "r"((uint32_t)(chunk_end))
                : "memory");
            asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"((
                uint64_t)(tensormap_workspace + (cta_slot_base + 256 + k_stage * 128)))
                         : "memory");
            mbarrier_arrive_expect_tx(load_k_full_addr + (k_stage) * 8, 16384);
#pragma unroll
            for (int dim_half = 0; dim_half < 2; dim_half++) {
              tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + (unsigned int)(dim_half * 8192),
                               tensormap_workspace + (cta_slot_base + 256 + k_stage * 128),
                               dim_half * 64, chunk_offset_3, qk_head_idx_3,
                               load_k_full_addr + (k_stage) * 8);
            }
            mbarrier_wait(q_smem_empty_addr + (q_stage) * 8, q_empty_phase_tma);
            asm volatile(
                "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" ::"l"((
                    uint64_t)(tensormap_workspace + ((unsigned int)cta_slot_base + q_stage * 128))),
                "r"((uint32_t)(chunk_end))
                : "memory");
            asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"((
                uint64_t)(tensormap_workspace + ((unsigned int)cta_slot_base + q_stage * 128)))
                         : "memory");
            mbarrier_arrive_expect_tx(load_q_full_addr + (q_stage) * 8, 16384);
#pragma unroll
            for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
              tma_3d_gmem2smem(smem_q_addr + q_stage * 16384 + (unsigned int)(dim_half_1 * 8192),
                               tensormap_workspace + ((unsigned int)cta_slot_base + q_stage * 128),
                               dim_half_1 * 64, chunk_offset_3, qk_head_idx_3,
                               load_q_full_addr + (q_stage) * 8);
            }
            mbarrier_wait(v_smem_empty_addr + (v_stage) * 8, v_empty_phase_tma);
            asm volatile(
                "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(
                    (uint64_t)(tensormap_workspace + (cta_slot_base + 640 + v_stage * 128))),
                "r"((uint32_t)(chunk_end))
                : "memory");
            asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"((
                uint64_t)(tensormap_workspace + (cta_slot_base + 640 + v_stage * 128)))
                         : "memory");
            mbarrier_arrive_expect_tx(load_v_full_addr + (v_stage) * 8, 16384);
#pragma unroll
            for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
              tma_3d_gmem2smem(smem_v_addr + v_stage * 16384 + (unsigned int)(dim_half_2 * 8192),
                               tensormap_workspace + (cta_slot_base + 640 + v_stage * 128),
                               dim_half_2 * 64, chunk_offset_3, v_head_idx_3,
                               load_v_full_addr + (v_stage) * 8);
            }
            mbarrier_wait(load_g_empty_addr + (g_stage) * 8, g_empty_phase_tma);
            asm volatile(
                "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(
                    (uint64_t)(tensormap_workspace + (cta_slot_base + 896 + g_stage * 128))),
                "r"((uint32_t)(chunk_end))
                : "memory");
            asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"((
                uint64_t)(tensormap_workspace + (cta_slot_base + 896 + g_stage * 128)))
                         : "memory");
            mbarrier_arrive_expect_tx(load_g_full_addr + (g_stage) * 8, 16384);
#pragma unroll
            for (int dim_half_3 = 0; dim_half_3 < 2; dim_half_3++) {
              tma_3d_gmem2smem(smem_g_addr + g_stage * 16384 + (unsigned int)(dim_half_3 * 8192),
                               tensormap_workspace + (cta_slot_base + 896 + g_stage * 128),
                               dim_half_3 * 64, chunk_offset_3, v_head_idx_3,
                               load_g_full_addr + (g_stage) * 8);
            }
          }
          k_stage += 1;
          if (k_stage == 3) {
            k_stage = 0;
            k_empty_phase_tma ^= 1;
          }
          q_stage += 1;
          if (q_stage == 2) {
            q_stage = 0;
            q_empty_phase_tma ^= 1;
          }
          v_stage += 1;
          if (v_stage == 2) {
            v_stage = 0;
            v_empty_phase_tma ^= 1;
          }
          g_stage += 1;
          if (g_stage == 2) {
            g_stage = 0;
            g_empty_phase_tma ^= 1;
          }
        }
      }
    }
  }
  // ---- Role: mma_cg1 ----
  if (warp == 10) {
    {  // mma_cg1_main
      unsigned int k_stage_1 = 0;
      unsigned int k_phase_mma = 0;
      unsigned int q_stage_1 = 0;
      unsigned int q_phase_mma = 0;
      unsigned int g_stage_mma = 0;
      unsigned int g_phase_mma = 0;
      unsigned int v_stage_mma = 0;
      unsigned int v_phase_mma = 0;
      unsigned int ainv_stage_mma = 0;
      unsigned int ainv_phase_mma = 0;
      unsigned int qk_stage_mma = 0;
      unsigned int qk_phase_mma = 0;
      unsigned int q_state_acc_mma_stage = 0;
      unsigned int q_state_acc_mma_phase = 1;
      unsigned int kv_acc_mma_stage = 0;
      unsigned int kv_acc_mma_phase = 1;
      unsigned int _phase_state_inp_ready_0 = 0;
      unsigned int _phase_cg1_shared_acc_empty_0 = 1;
      unsigned int _phase_vks_ready_0 = 0;
      unsigned int _phase_nv_ready_0 = 0;
      unsigned int _phase_decay_v_ready_0 = 0;
#pragma unroll 1
      for (unsigned int tile_4 = bid; tile_4 < total_tiles; tile_4 += num_bids) {
        int num_o_heads_4 = ((num_q_heads >= num_v_heads) ? num_q_heads : num_v_heads);
        int batch_idx_4 = tile_4 / (unsigned int)num_o_heads_4;
        int head_idx_4 = tile_4 % (unsigned int)num_o_heads_4;
        int qk_head_idx_4 =
            ((num_q_heads >= num_v_heads) ? head_idx_4 : head_idx_4 / (num_v_heads / num_q_heads));
        int v_head_idx_4 =
            ((num_v_heads >= num_q_heads) ? head_idx_4 : head_idx_4 / (num_q_heads / num_v_heads));
        int batch_start_4 = (int)cu_seqlens[batch_idx_4];
        int batch_end_4 = (int)cu_seqlens[batch_idx_4 + 1];
        int seqlen_b_4 = batch_end_4 - batch_start_4;
        int num_pairs_b_4 = (seqlen_b_4 + 32 - 1) / 32;
        int num_chunks_b_4 = num_pairs_b_4 * 2;
        if (num_chunks_b_4 > 0) {
          {
            kv_acc_mma_stage += 1;
            if (kv_acc_mma_stage == 1) {
              kv_acc_mma_stage = 0;
              kv_acc_mma_phase ^= 1;
            }
            int chunk_offset_4 = batch_start_4;
            int _mma_marker = batch_idx_4 + head_idx_4 + chunk_offset_4 + batch_end_4 + 512;
            mbarrier_wait(qkg_ready_addr + (g_stage_mma) * 8, g_phase_mma);
            {
              mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
              _phase_state_inp_ready_0 ^= 1;
              mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
              _phase_cg1_shared_acc_empty_0 ^= 1;
              int _mma_b_lo_4 =
                  make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
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
                  "mov.b32 id, 135267472;\n\t"
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
                  "add.u32 blo, blo, 506;\n\t"
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
                  "}\n" ::"r"(tmem_tmem_cg1_shared_acc),
                  "r"(_mma_b_lo_4), "r"(tmem_tmem_state_inp), "r"(0));
              elect_commit(cg1_shared_acc_full_addr);
              mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8,
                            q_state_acc_mma_phase);
              int _mma_b_lo_5 =
                  make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_1) * 1024);
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
                  "mov.b32 id, 135267472;\n\t"
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
                  "add.u32 blo, blo, 506;\n\t"
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
                  "}\n" ::"r"(tmem_tmem_q_state),
                  "r"(_mma_b_lo_5), "r"(tmem_tmem_state_inp), "r"(0));
              elect_commit(q_state_acc_full_addr);
              q_state_acc_mma_stage += 1;
              if (q_state_acc_mma_stage == 1) {
                q_state_acc_mma_stage = 0;
                q_state_acc_mma_phase ^= 1;
              }
            }
            if (elect_sync()) {
              mbarrier_arrive(q_smem_empty_addr + (q_stage_1) * 8);
            }
            mbarrier_wait(vks_ready_addr, _phase_vks_ready_0);
            _phase_vks_ready_0 ^= 1;
            mbarrier_wait(ainv_ready_addr + (ainv_stage_mma) * 8, ainv_phase_mma);
            mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
            _phase_cg1_shared_acc_empty_0 ^= 1;
            {
              int _mma_b_lo_6 =
                  make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_mma) * 512);
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
                  "mov.b32 id, 135267472;\n\t"
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
                  "}\n" ::"r"(tmem_tmem_cg1_shared_acc),
                  "r"(_mma_b_lo_6), "r"(tmem_tmem_shared_inp), "r"(0));
            }
            elect_commit(cg1_shared_acc_full_addr);
            if (elect_sync()) {
              mbarrier_arrive(ainv_smem_empty_addr + (ainv_stage_mma) * 8);
            }
            ainv_stage_mma += 1;
            if (ainv_stage_mma == 3) {
              ainv_stage_mma = 0;
              ainv_phase_mma ^= 1;
            }
            mbarrier_wait(qk_ready_addr + (qk_stage_mma) * 8, qk_phase_mma);
            mbarrier_wait(nv_ready_addr, _phase_nv_ready_0);
            _phase_nv_ready_0 ^= 1;
            mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8,
                          q_state_acc_mma_phase);
            int _mma_b_lo_8 =
                make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_mma) * 512);
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
                "mov.b32 id, 135267472;\n\t"
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
                "}\n" ::"r"(tmem_tmem_q_state),
                "r"(_mma_b_lo_8), "r"(tmem_tmem_shared_inp), "r"(((!1) ? 0 : 1)));
            elect_commit(q_state_acc_full_addr);
            q_state_acc_mma_stage += 1;
            if (q_state_acc_mma_stage == 1) {
              q_state_acc_mma_stage = 0;
              q_state_acc_mma_phase ^= 1;
            }
            if (elect_sync()) {
              mbarrier_arrive(qk_smem_empty_addr + (qk_stage_mma) * 8);
            }
            qk_stage_mma += 1;
            if (qk_stage_mma == 2) {
              qk_stage_mma = 0;
              qk_phase_mma ^= 1;
            }
            mbarrier_wait(kv_acc_empty_addr + (kv_acc_mma_stage) * 8, kv_acc_mma_phase);
            mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
            _phase_decay_v_ready_0 ^= 1;
            mbarrier_wait(kr_ready_addr + (g_stage_mma) * 8, g_phase_mma);
            int _mma_b_lo_9 = make_warp_uniform(
                ((((smem_g_trans_mma_addr) >> 4) & 0x3FFF) | 0x2000000) + (g_stage_mma) * 1024);
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
                "mov.b32 id, 136381584;\n\t"
                "mov.b32 ta, %2;\n\t"
                "mov.b32 blo, %1;\n\t"
                "mov.b64 db, {blo, dhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                "add.u32 ta, ta, 8;\n\t"
                "add.u32 blo, blo, 128;\n\t"
                "mov.b64 db, {blo, dhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                "add.u32 ta, ta, 8;\n\t"
                "add.u32 blo, blo, 128;\n\t"
                "mov.b64 db, {blo, dhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                "add.u32 ta, ta, 8;\n\t"
                "add.u32 blo, blo, 128;\n\t"
                "mov.b64 db, {blo, dhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                "}\n" ::"r"(tmem_tmem_state),
                "r"(_mma_b_lo_9), "r"(tmem_tmem_shared_inp + 32), "r"(((!1) ? 0 : 1)));
            elect_commit(kv_acc_full_addr);
            elect_commit(load_k_empty_addr + (k_stage_1) * 8);
            elect_commit(load_g_empty_addr + (g_stage_mma) * 8);
            kv_acc_mma_stage += 1;
            if (kv_acc_mma_stage == 1) {
              kv_acc_mma_stage = 0;
              kv_acc_mma_phase ^= 1;
            }
            k_stage_1 += 1;
            if (k_stage_1 == 3) {
              k_stage_1 = 0;
              k_phase_mma ^= 1;
            }
            q_stage_1 += 1;
            if (q_stage_1 == 2) {
              q_stage_1 = 0;
              q_phase_mma ^= 1;
            }
            g_stage_mma += 1;
            if (g_stage_mma == 2) {
              g_stage_mma = 0;
              g_phase_mma ^= 1;
            }
            v_stage_mma += 1;
            if (v_stage_mma == 2) {
              v_stage_mma = 0;
              v_phase_mma ^= 1;
            }
          }
        }
#pragma unroll 1
        for (int chunk_idx_3 = 1; chunk_idx_3 < num_chunks_b_4; chunk_idx_3++) {
          int chunk_offset_5 = batch_start_4 + chunk_idx_3 * 16;
          int _mma_marker_1 = batch_idx_4 + head_idx_4 + chunk_offset_5 + batch_end_4 + 512;
          mbarrier_wait(qkg_ready_addr + (g_stage_mma) * 8, g_phase_mma);
          {
            mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
            _phase_state_inp_ready_0 ^= 1;
            mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
            _phase_cg1_shared_acc_empty_0 ^= 1;
            int _mma_b_lo_16 =
                make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
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
                "mov.b32 id, 135267472;\n\t"
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
                "add.u32 blo, blo, 506;\n\t"
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
                "}\n" ::"r"(tmem_tmem_cg1_shared_acc),
                "r"(_mma_b_lo_16), "r"(tmem_tmem_state_inp), "r"(0));
            elect_commit(cg1_shared_acc_full_addr);
            mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8,
                          q_state_acc_mma_phase);
            int _mma_b_lo_17 =
                make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_1) * 1024);
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
                "mov.b32 id, 135267472;\n\t"
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
                "add.u32 blo, blo, 506;\n\t"
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
                "}\n" ::"r"(tmem_tmem_q_state),
                "r"(_mma_b_lo_17), "r"(tmem_tmem_state_inp), "r"(0));
            elect_commit(q_state_acc_full_addr);
            q_state_acc_mma_stage += 1;
            if (q_state_acc_mma_stage == 1) {
              q_state_acc_mma_stage = 0;
              q_state_acc_mma_phase ^= 1;
            }
          }
          if (elect_sync()) {
            mbarrier_arrive(q_smem_empty_addr + (q_stage_1) * 8);
          }
          mbarrier_wait(vks_ready_addr, _phase_vks_ready_0);
          _phase_vks_ready_0 ^= 1;
          mbarrier_wait(ainv_ready_addr + (ainv_stage_mma) * 8, ainv_phase_mma);
          mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
          _phase_cg1_shared_acc_empty_0 ^= 1;
          {
            int _mma_b_lo_18 =
                make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_mma) * 512);
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
                "mov.b32 id, 135267472;\n\t"
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
                "}\n" ::"r"(tmem_tmem_cg1_shared_acc),
                "r"(_mma_b_lo_18), "r"(tmem_tmem_shared_inp), "r"(0));
          }
          elect_commit(cg1_shared_acc_full_addr);
          if (elect_sync()) {
            mbarrier_arrive(ainv_smem_empty_addr + (ainv_stage_mma) * 8);
          }
          ainv_stage_mma += 1;
          if (ainv_stage_mma == 3) {
            ainv_stage_mma = 0;
            ainv_phase_mma ^= 1;
          }
          mbarrier_wait(qk_ready_addr + (qk_stage_mma) * 8, qk_phase_mma);
          mbarrier_wait(nv_ready_addr, _phase_nv_ready_0);
          _phase_nv_ready_0 ^= 1;
          mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8,
                        q_state_acc_mma_phase);
          int _mma_b_lo_20 =
              make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_mma) * 512);
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
              "mov.b32 id, 135267472;\n\t"
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
              "}\n" ::"r"(tmem_tmem_q_state),
              "r"(_mma_b_lo_20), "r"(tmem_tmem_shared_inp), "r"(((!1) ? 0 : 1)));
          elect_commit(q_state_acc_full_addr);
          q_state_acc_mma_stage += 1;
          if (q_state_acc_mma_stage == 1) {
            q_state_acc_mma_stage = 0;
            q_state_acc_mma_phase ^= 1;
          }
          if (elect_sync()) {
            mbarrier_arrive(qk_smem_empty_addr + (qk_stage_mma) * 8);
          }
          qk_stage_mma += 1;
          if (qk_stage_mma == 2) {
            qk_stage_mma = 0;
            qk_phase_mma ^= 1;
          }
          mbarrier_wait(kv_acc_empty_addr + (kv_acc_mma_stage) * 8, kv_acc_mma_phase);
          mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
          _phase_decay_v_ready_0 ^= 1;
          mbarrier_wait(kr_ready_addr + (g_stage_mma) * 8, g_phase_mma);
          int _mma_b_lo_21 = make_warp_uniform(
              ((((smem_g_trans_mma_addr) >> 4) & 0x3FFF) | 0x2000000) + (g_stage_mma) * 1024);
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
              "mov.b32 id, 136381584;\n\t"
              "mov.b32 ta, %2;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 db, {blo, dhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
              "add.u32 ta, ta, 8;\n\t"
              "add.u32 blo, blo, 128;\n\t"
              "mov.b64 db, {blo, dhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
              "add.u32 ta, ta, 8;\n\t"
              "add.u32 blo, blo, 128;\n\t"
              "mov.b64 db, {blo, dhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
              "add.u32 ta, ta, 8;\n\t"
              "add.u32 blo, blo, 128;\n\t"
              "mov.b64 db, {blo, dhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
              "}\n" ::"r"(tmem_tmem_state),
              "r"(_mma_b_lo_21), "r"(tmem_tmem_shared_inp + 32), "r"(((!1) ? 0 : 1)));
          elect_commit(kv_acc_full_addr);
          elect_commit(load_k_empty_addr + (k_stage_1) * 8);
          elect_commit(load_g_empty_addr + (g_stage_mma) * 8);
          kv_acc_mma_stage += 1;
          if (kv_acc_mma_stage == 1) {
            kv_acc_mma_stage = 0;
            kv_acc_mma_phase ^= 1;
          }
          k_stage_1 += 1;
          if (k_stage_1 == 3) {
            k_stage_1 = 0;
            k_phase_mma ^= 1;
          }
          q_stage_1 += 1;
          if (q_stage_1 == 2) {
            q_stage_1 = 0;
            q_phase_mma ^= 1;
          }
          g_stage_mma += 1;
          if (g_stage_mma == 2) {
            g_stage_mma = 0;
            g_phase_mma ^= 1;
          }
          v_stage_mma += 1;
          if (v_stage_mma == 2) {
            v_stage_mma = 0;
            v_phase_mma ^= 1;
          }
        }
        if (STORE_FINAL_STATE != 0 && num_chunks_b_4 > 0) {
          mbarrier_wait(kv_acc_empty_addr + (kv_acc_mma_stage) * 8, kv_acc_mma_phase);
        }
      }
    }
  }
  // ---- Role: output_epilogue ----
  if (warp == 11) {
    {  // output_epilogue_main
      int cta_slot_base_epi = bid * 1280;
      unsigned int gate_prod_stage = 0;
      unsigned int gate_prod_phase = 1;
      unsigned int o_epi_stage = 0;
      unsigned int o_epi_phase = 0;
      if (elect_sync()) {
        {
          const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(O);
          uint64_t* __tm_dst = reinterpret_cast<uint64_t*>(
              (uint64_t)(tensormap_workspace + (cta_slot_base_epi + 1152)));
#pragma unroll
          for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
            __tm_dst[__tm_i] = __tm_src[__tm_i];
          }
        }
        asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
      }
#pragma unroll 1
      for (unsigned int tile_5 = bid; tile_5 < total_tiles; tile_5 += num_bids) {
        int num_o_heads_5 = ((num_q_heads >= num_v_heads) ? num_q_heads : num_v_heads);
        int batch_idx_5 = tile_5 / (unsigned int)num_o_heads_5;
        int head_idx_5 = tile_5 % (unsigned int)num_o_heads_5;
        int qk_head_idx_5 =
            ((num_q_heads >= num_v_heads) ? head_idx_5 : head_idx_5 / (num_v_heads / num_q_heads));
        int v_head_idx_5 =
            ((num_v_heads >= num_q_heads) ? head_idx_5 : head_idx_5 / (num_q_heads / num_v_heads));
        int batch_start_5 = (int)cu_seqlens[batch_idx_5];
        int batch_end_5 = (int)cu_seqlens[batch_idx_5 + 1];
        int seqlen_b_5 = batch_end_5 - batch_start_5;
        int num_pairs_b_5 = (seqlen_b_5 + 32 - 1) / 32;
        int num_chunks_b_5 = num_pairs_b_5 * 2;
        if (elect_sync()) {
          asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(
                           (uint64_t)(tensormap_workspace + (cta_slot_base_epi + 1152))),
                       "r"((uint32_t)(batch_end_5))
                       : "memory");
          asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
        }
        int num_valid_chunks_b = (batch_end_5 - batch_start_5 + 16 - 1) / 16;
        if (num_chunks_b_5 > 0) {
#pragma unroll
          for (int prefetch_idx = 0; prefetch_idx < 2; prefetch_idx++) {
            int prefetch_offset = batch_start_5 + prefetch_idx * 16;
            mbarrier_wait(beta_smem_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
            int gb_lane = lane;
            int beta_elem_base = gate_prod_stage * 64;
            int token_0 = prefetch_offset + gb_lane;
            int beta_idx_0 = token_0 * num_o_heads_5 + head_idx_5;
            float beta_val_0 = 0.0f;
            if (gb_lane < 16 && token_0 < batch_end_5) {
              __nv_bfloat16 beta_logit_0 = beta[beta_idx_0];
              float _cvt_f32_0 = __bfloat162float(beta_logit_0);
              float _expf_4 = __expf(-_cvt_f32_0);
              float _rcp_2 = approx_rcp(1.0f + _expf_4);
              beta_val_0 = _rcp_2;
            }
            smem_beta[beta_elem_base + gb_lane] = beta_val_0;
            smem_beta[beta_elem_base + gb_lane + 32] = 0.0f;
            mbarrier_arrive(load_beta_full_addr + (gate_prod_stage) * 8);
            gate_prod_stage += 1;
            if (gate_prod_stage == 5) {
              gate_prod_stage = 0;
              gate_prod_phase ^= 1;
            }
          }
          if (num_chunks_b_5 > 2) {
#pragma unroll
            for (int prefetch_idx_1 = 2; prefetch_idx_1 < 4; prefetch_idx_1++) {
              int prefetch_offset_1 = batch_start_5 + prefetch_idx_1 * 16;
              mbarrier_wait(beta_smem_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
              int gb_lane_1 = lane;
              int beta_elem_base_1 = gate_prod_stage * 64;
              int token_0_1 = prefetch_offset_1 + gb_lane_1;
              int beta_idx_0_1 = token_0_1 * num_o_heads_5 + head_idx_5;
              float beta_val_0_1 = 0.0f;
              if (gb_lane_1 < 16 && token_0_1 < batch_end_5) {
                __nv_bfloat16 beta_logit_0_1 = beta[beta_idx_0_1];
                float _cvt_f32_1 = __bfloat162float(beta_logit_0_1);
                float _expf_5 = __expf(-_cvt_f32_1);
                float _rcp_3 = approx_rcp(1.0f + _expf_5);
                beta_val_0_1 = _rcp_3;
              }
              smem_beta[beta_elem_base_1 + gb_lane_1] = beta_val_0_1;
              smem_beta[beta_elem_base_1 + gb_lane_1 + 32] = 0.0f;
              mbarrier_arrive(load_beta_full_addr + (gate_prod_stage) * 8);
              gate_prod_stage += 1;
              if (gate_prod_stage == 5) {
                gate_prod_stage = 0;
                gate_prod_phase ^= 1;
              }
            }
          }
        }
#pragma unroll 1
        for (int chunk_idx_4 = 0; chunk_idx_4 < num_chunks_b_5; chunk_idx_4++) {
          int chunk_offset_6 = batch_start_5 + chunk_idx_4 * 16;
          int prefetch_idx_2 = chunk_idx_4 + 4;
          if (prefetch_idx_2 < num_chunks_b_5) {
            int prefetch_offset_2 = batch_start_5 + prefetch_idx_2 * 16;
            mbarrier_wait(beta_smem_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
            int gb_lane_2 = lane;
            int beta_elem_base_2 = gate_prod_stage * 64;
            int token_0_2 = prefetch_offset_2 + gb_lane_2;
            int beta_idx_0_2 = token_0_2 * num_o_heads_5 + head_idx_5;
            float beta_val_0_2 = 0.0f;
            if (gb_lane_2 < 16 && token_0_2 < batch_end_5) {
              __nv_bfloat16 beta_logit_0_2 = beta[beta_idx_0_2];
              float _cvt_f32_2 = __bfloat162float(beta_logit_0_2);
              float _expf_6 = __expf(-_cvt_f32_2);
              float _rcp_4 = approx_rcp(1.0f + _expf_6);
              beta_val_0_2 = _rcp_4;
            }
            smem_beta[beta_elem_base_2 + gb_lane_2] = beta_val_0_2;
            smem_beta[beta_elem_base_2 + gb_lane_2 + 32] = 0.0f;
            mbarrier_arrive(load_beta_full_addr + (gate_prod_stage) * 8);
            gate_prod_stage += 1;
            if (gate_prod_stage == 5) {
              gate_prod_stage = 0;
              gate_prod_phase ^= 1;
            }
          }
          mbarrier_wait(o_store_ready_addr + (o_epi_stage) * 8, o_epi_phase);
          if (elect_sync()) {
            if (chunk_idx_4 == 0) {
              asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(
                               (uint64_t)(tensormap_workspace + (cta_slot_base_epi + 1152)))
                           : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
            for (int dim_half_4 = 0; dim_half_4 < 2; dim_half_4++) {
              tma_store_3d(tensormap_workspace + (cta_slot_base_epi + 1152), dim_half_4 * 64,
                           chunk_offset_6, head_idx_5,
                           smem_o_addr + o_epi_stage * 16384 + (unsigned int)(dim_half_4 * 8192));
            }
          }
          asm volatile("cp.async.bulk.commit_group;");
          asm volatile("cp.async.bulk.wait_group 0;");
          mbarrier_arrive(o_smem_empty_addr + (o_epi_stage) * 8);
          o_epi_stage += 1;
          if (o_epi_stage == 2) {
            o_epi_stage = 0;
            o_epi_phase ^= 1;
          }
        }
      }
    }
  }

  // Cleanup
  __syncthreads();  // barrier before TMEM dealloc

  if (warp == 4) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr_storage[0]),
        "r"(512));
  }
}

}  // extern "C"

#undef ENABLE_CHECKPOINTS
#undef IS_GQA
#undef LOOM_INF
#undef NUM_AINV_PIPE_STAGES
#undef NUM_BETA_PIPE_STAGES
#undef NUM_CG0_ACC_PIPE_STAGES
#undef NUM_GATE_PIPE_STAGES
#undef NUM_G_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef NUM_ONE_STAGE_STAGES
#undef NUM_O_PIPE_STAGES
#undef NUM_QK_PIPE_STAGES
#undef NUM_Q_PIPE_STAGES
#undef NUM_V_PIPE_STAGES
#undef SMEM_SMEM_AINV_OFF
#undef SMEM_SMEM_AINV_RM_OFF
#undef SMEM_SMEM_AINV_RM_STAGE_BYTES
#undef SMEM_SMEM_AINV_RM_STRIDE
#undef SMEM_SMEM_AINV_STAGE_BYTES
#undef SMEM_SMEM_AINV_STRIDE
#undef SMEM_SMEM_BETA_OFF
#undef SMEM_SMEM_BETA_STAGE_BYTES
#undef SMEM_SMEM_BETA_STRIDE
#undef SMEM_SMEM_G_OFF
#undef SMEM_SMEM_G_STAGE_BYTES
#undef SMEM_SMEM_G_STRIDE
#undef SMEM_SMEM_G_TOTAL_OFF
#undef SMEM_SMEM_G_TOTAL_STAGE_BYTES
#undef SMEM_SMEM_G_TOTAL_STRIDE
#undef SMEM_SMEM_G_TRANS_MMA_OFF
#undef SMEM_SMEM_G_TRANS_MMA_STAGE_BYTES
#undef SMEM_SMEM_G_TRANS_MMA_STRIDE
#undef SMEM_SMEM_K_OFF
#undef SMEM_SMEM_K_STAGE_BYTES
#undef SMEM_SMEM_K_STRIDE
#undef SMEM_SMEM_K_TRANS_MMA_OFF
#undef SMEM_SMEM_K_TRANS_MMA_STAGE_BYTES
#undef SMEM_SMEM_K_TRANS_MMA_STRIDE
#undef SMEM_SMEM_O_OFF
#undef SMEM_SMEM_O_STAGE_BYTES
#undef SMEM_SMEM_O_STRIDE
#undef SMEM_SMEM_QK_OFF
#undef SMEM_SMEM_QK_STAGE_BYTES
#undef SMEM_SMEM_QK_STRIDE
#undef SMEM_SMEM_Q_OFF
#undef SMEM_SMEM_Q_STAGE_BYTES
#undef SMEM_SMEM_Q_STRIDE
#undef SMEM_SMEM_V_MMA_OFF
#undef SMEM_SMEM_V_MMA_STAGE_BYTES
#undef SMEM_SMEM_V_MMA_STRIDE
#undef SMEM_SMEM_V_OFF
#undef SMEM_SMEM_V_STAGE_BYTES
#undef SMEM_SMEM_V_STRIDE
#undef SMEM_TOTAL
#undef STORE_FINAL_STATE
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_CG0_SHARED_ACC_OFFSET
#undef TMEM_TMEM_CG1_SHARED_ACC_OFFSET
#undef TMEM_TMEM_Q_STATE_OFFSET
#undef TMEM_TMEM_SHARED_INP_OFFSET
#undef TMEM_TMEM_STATE_INP_OFFSET
#undef TMEM_TMEM_STATE_OFFSET
#undef USE_INITIAL_STATE
#undef ainv_ready_addr
#undef ainv_smem_empty_addr
#undef beta_smem_empty_addr
#undef cg0_shared_acc_empty_addr
#undef cg0_shared_acc_full_addr
#undef cg1_shared_acc_empty_addr
#undef cg1_shared_acc_full_addr
#undef decay_v_ready_addr
#undef gate_cg1_empty_addr
#undef initial_state_loaded_addr
#undef ki_mma_consumed_addr
#undef kr_ready_addr
#undef kv_acc_empty_addr
#undef kv_acc_full_addr
#undef load_beta_full_addr
#undef load_g_empty_addr
#undef load_g_full_addr
#undef load_gate_full_addr
#undef load_k_empty_addr
#undef load_k_full_addr
#undef load_q_full_addr
#undef load_v_full_addr
#undef nv_ready_addr
#undef o_smem_empty_addr
#undef o_store_ready_addr
#undef q_smem_empty_addr
#undef q_state_acc_empty_addr
#undef q_state_acc_full_addr
#undef qk_ready_addr
#undef qk_smem_empty_addr
#undef qkg_ready_addr
#undef smem_ainv_addr
#undef smem_ainv_rm_addr
#undef smem_beta_addr
#undef smem_g_addr
#undef smem_g_total_addr
#undef smem_g_trans_mma_addr
#undef smem_k_addr
#undef smem_k_trans_mma_addr
#undef smem_o_addr
#undef smem_q_addr
#undef smem_qk_addr
#undef smem_v_addr
#undef smem_v_mma_addr
#undef state_inp_ready_addr
#undef v_smem_empty_addr
#undef vks_ready_addr
