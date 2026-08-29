// Warp-specialized SM100 UMMA/TMEM prefill path for group_size==16, head_dim=128.
// Roles (16 warps, 512 threads, 1 CTA/SM):
//   warps 0-7  : softmax consumers (stage = warp/4; 128 TMEM lanes each)
//   warps 8-11 : correction (TMEM O rescale) + epilogue gmem store
//   warp  12   : MMA leader (elect.sync inside each mma group) + TMEM dealloc
//   warp  13   : transform (idle; reserved)
//   warp  14   : producer (meta, q2k copy, Q + K/V TMA stream)
//   warp  15   : transform (idle; reserved)
// Barrier protocol mirrors the proven SM100 dataflow: kv_full/kv_empty slot
// ring (3 x 32KB), q_full[2], s_full[2], p_full[2] + p_full_tail[2],
// corr_sig[2], corr_done[2], o_full[2], q2k_full, tmem_dealloc bar.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

#include "msa_vibecuda_common.h"

namespace msa_umma_g16 {

constexpr int kTile = 16;   // queries per CTA tile
constexpr int kGroup = 16;  // q heads per kv head
constexpr int kRows = 256;  // 16 queries x 16 heads per tile
constexpr int kHead = 128;
constexpr int kThreads = 512;
constexpr int kKVSlots = 3;
// smem byte offsets (match the proven SM100 dataflow layout)
constexpr int kScaleOff = 1024;  // 4 KB: [0..255] acc_scale, [256..511] l, [512..767] max
constexpr int kQ0Off = 5120;     // 32 KB
constexpr int kQ1Off = 37888;    // 32 KB
constexpr int kRingOff = 70656;  // 3 slots x 32768 B (K and V share the ring)
constexpr int kQ2KOff = 168960;

// ---- flat params (flat KV only; paged stays on the HMMA path) ----
struct G16Params {
  const void* q;  // [total_q, num_q_heads, 128]
  const void* k;  // [total_k, num_kv_heads, 128]
  const void* v;
  const int* q2k;   // [num_kv_heads, total_q, topk]
  const int* cu_q;  // [nbatch+1]
  const int* cu_k;  // [nbatch+1]
  void* out;        // [total_q, num_q_heads, 128]
  int total_q, total_k, num_q_heads, num_kv_heads, topk, nbatch;
  int causal;
  float scale_log2e;  // head_dim^-0.5 * log2(e)
};

// ==== ANCHOR:PTX_HELPERS ====
__device__ __forceinline__ int smem_u32(const void* p) {
  return static_cast<int>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred P1;\n\telect.sync _|P1, 0xFFFFFFFF;\n\t"
      "selp.u32 %0, 1, 0, P1;\n\t}"
      : "=r"(pred));
  return pred != 0;
}

__device__ __forceinline__ int warp_uni(int x) { return __shfl_sync(0xFFFFFFFFu, x, 0); }

__device__ __forceinline__ void mbar_init(int addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(int addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}

__device__ __forceinline__ void mbar_arrive_tx(int addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(int addr, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n"
      "LAB_WAIT%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE%=;\n\t"
      "bra LAB_WAIT%=;\n"
      "DONE%=:\n\t}" ::"r"(addr),
      "r"(parity)
      : "memory");
}

// 4D TMA load; dst must be a completed expect_tx mbarrier.
__device__ __forceinline__ void tma_load_4d(int dst, const CUtensorMap* map, int c0, int c1, int c2,
                                            int c3, int mbar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void tc_commit(int mbar) {
  asm volatile(
      "{\n\t.reg .pred leader;\n\telect.sync _|leader, 0xFFFFFFFF;\n\t"
      "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}" ::
          "r"(mbar));
}

__device__ __forceinline__ void tc_alloc(int smem_dst, int ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_dst),
               "r"(ncols));
}

__device__ __forceinline__ void tc_dealloc(int taddr, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}

__device__ __forceinline__ void tc_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ __forceinline__ void tc_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tc_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// tcgen05.ld 32x32b x64: each lane reads 64 consecutive 32b TMEM cells on its own lane row.
__device__ __forceinline__ void tmem_ld64(float* d, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]),
        "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
        "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]),
        "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
        "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]),
        "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]),
        "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]),
        "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
        "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]),
        "=f"(d[63])
      : "r"(addr));
}

// tcgen05.st 32x32b x32: each lane writes 32 consecutive packed bf16x2 cells.
__device__ __forceinline__ void tmem_st32(int addr, const uint32_t* s) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(addr),
      "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
      "r"(s[8]), "r"(s[9]), "r"(s[10]), "r"(s[11]), "r"(s[12]), "r"(s[13]), "r"(s[14]), "r"(s[15]),
      "r"(s[16]), "r"(s[17]), "r"(s[18]), "r"(s[19]), "r"(s[20]), "r"(s[21]), "r"(s[22]),
      "r"(s[23]), "r"(s[24]), "r"(s[25]), "r"(s[26]), "r"(s[27]), "r"(s[28]), "r"(s[29]),
      "r"(s[30]), "r"(s[31]));
}

__device__ __forceinline__ void tmem_ld16(float* d, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]),
        "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
        "=f"(d[14]), "=f"(d[15])
      : "r"(addr));
}

__device__ __forceinline__ void tmem_st16f(int addr, const float* s) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(addr),
      "f"(s[0]), "f"(s[1]), "f"(s[2]), "f"(s[3]), "f"(s[4]), "f"(s[5]), "f"(s[6]), "f"(s[7]),
      "f"(s[8]), "f"(s[9]), "f"(s[10]), "f"(s[11]), "f"(s[12]), "f"(s[13]), "f"(s[14]), "f"(s[15]));
}

__device__ __forceinline__ void tmem_st16u(int addr, const uint32_t* s) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(addr),
      "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
      "r"(s[8]), "r"(s[9]), "r"(s[10]), "r"(s[11]), "r"(s[12]), "r"(s[13]), "r"(s[14]), "r"(s[15])
      : "memory");
}

__device__ __forceinline__ void sreg_inc_192() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
}
__device__ __forceinline__ void sreg_dec_80() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
}
__device__ __forceinline__ void sreg_dec_48() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ float fast_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// packed f32x2 (SM10x) helpers — two lanes of FP32 work per instruction
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("add.rn.ftz.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}
__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}
__device__ __forceinline__ uint64_t pack2(float lo, float hi) {
  uint64_t r;
  float2 f = make_float2(lo, hi);
  r = *(uint64_t*)&f;
  return r;
}

// ==== ANCHOR:MMA_GROUPS ====
// Instruction descriptor for kind::f16 M128 QK (D=f32, A/B=bf16); f16 variant = idesc-0x480.
constexpr uint32_t kIdescQK = 136316048u;
constexpr uint32_t kIdescPV = 136381584u;
// SMEM desc hi word: SBO=1024B swizzle layout, 128B swizzle atom.
constexpr uint32_t kDescHi = 0x40004040u;

// QK^T for one 128-row stage: A = Q smem desc (a_lo), B = K smem desc (b_lo).
// 8 k-steps of 16 dims; desc lo walks +2,2,2,1018,2,2,2 (16B units).
__device__ __forceinline__ void mma_qk_group(int a_lo_in, int b_lo_in, int tmem_d, uint32_t idesc,
                                             int first) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader, p0, p1;\n\t"
      ".reg .b32 adhi, alo, bdhi, blo, idesc;\n\t"
      ".reg .b64 da, db;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 adhi, %4;\n\t"
      "mov.b32 bdhi, %4;\n\t"
      "mov.b32 idesc, %5;\n\t"
      "mov.b32 alo, %0;\n\t"
      "mov.b32 blo, %1;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p0;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "add.u32 alo, alo, 1018;\n\t"
      "add.u32 blo, blo, 1018;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, adhi};\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, idesc, p1;\n\t"
      "}\n" ::"r"(a_lo_in),
      "r"(b_lo_in), "r"(tmem_d), "r"(first), "r"(kDescHi), "r"(idesc));
}

// P*V: A = P in TMEM (tmem_a, 64 cells = 128 packed tokens), B = V smem desc.
// First half: tokens 0..95 (A cells +0,+8,...,+40); B lo walks +128 per kstep.
__device__ __forceinline__ void mma_pv_head(int v_lo_in, int tmem_d, int tmem_a, uint32_t idesc,
                                            int first) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader, p0, p1;\n\t"
      ".reg .b32 bdhi, blo, idesc;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "setp.ne.b32 p0, %4, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 bdhi, %5;\n\t"
      "mov.b32 idesc, %6;\n\t"
      "mov.b32 blo, %1;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3], db, idesc, p0;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 8], db, idesc, p1;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 16], db, idesc, p1;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 24], db, idesc, p1;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 32], db, idesc, p1;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 40], db, idesc, p1;\n\t"
      "}\n" ::"r"(tmem_d),
      "r"(v_lo_in), "r"(0), "r"(tmem_a), "r"(first), "r"(kDescHi), "r"(idesc)
      : "memory");
}

// Second half: tokens 96..127 (A cells +48,+56); B lo = base + 768 then +128.
__device__ __forceinline__ void mma_pv_tail(int v_lo_in, int tmem_d, int tmem_a, uint32_t idesc) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader, p1;\n\t"
      ".reg .b32 bdhi, blo, idesc;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 bdhi, %4;\n\t"
      "mov.b32 idesc, %5;\n\t"
      "add.u32 blo, %1, 768;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 48], db, idesc, p1;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, bdhi};\n\t"
      "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%3 + 56], db, idesc, p1;\n\t"
      "}\n" ::"r"(tmem_d),
      "r"(v_lo_in), "r"(0), "r"(tmem_a), "r"(kDescHi), "r"(idesc)
      : "memory");
}

// ==== ANCHOR:KERNEL ====
// Round-18 A/B (fixture0 = b1 q4096 flat g16 causal): base 235.2us, reverse
// 225.1us (-4.3%), 2-CTA parity split 286.7us (+22% loss even with NO DSM
// merge — a strict lower bound on the full split+merge design, so the merge
// variant is arithmetically pre-refuted on this 6.95-wave saturated grid).
// REVERSE is the retained win and defaults on. The round-18 split-probe
// launcher/dispatch arms were pruned in round 22 (AC-1 cleanup); the
// measured evidence lives in ncu_evidence.md.
#ifndef MSA_G16_REVERSE
#define MSA_G16_REVERSE 1
#endif

__device__ __forceinline__ uint32_t g16_cluster_rank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

struct TileMeta {
  int batch, q_local_base, q_valid, query_base, k_start, kv_len, qoff, num_n_blocks, kv_head,
      q_head;
};

__device__ __forceinline__ TileMeta compute_meta(const G16Params& p) {
  TileMeta m;
  // grid is (num_kv_heads, tiles): kv_head varies fastest in issue order so
  // the 4 CTAs covering complementary kv_head slices of the same token block
  // are scheduled together — the strided flat-KV lines are fetched once into
  // L2 and shared instead of four temporally-separated DRAM misses.
#ifdef __CUDA_ARCH__
  int linear_tile = blockIdx.y;
#if MSA_G16_REVERSE
  if (p.causal) linear_tile = (int)gridDim.y - 1 - linear_tile;
#endif
#else
  int linear_tile = blockIdx.y;
#endif
  int tile_prefix = 0, tile_active = 0, q_tile = 0, batch = 0;
  for (int b = 0; b < p.nbatch; b++) {
    int qb = p.cu_q[b];
    int ql = p.cu_q[b + 1] - qb;
    int tiles = (ql + kTile - 1) / kTile;
    if (linear_tile >= tile_prefix && linear_tile < tile_prefix + tiles) {
      batch = b;
      q_tile = linear_tile - tile_prefix;
      tile_active = 1;
    }
    tile_prefix += tiles;
  }
  m.batch = batch;
  m.kv_head = blockIdx.x;
  m.q_head = m.kv_head * kGroup;
  int qb = p.cu_q[batch];
  int ql = p.cu_q[batch + 1] - qb;
  m.q_local_base = q_tile * kTile;
  m.q_valid = tile_active ? min(max(ql - m.q_local_base, 0), kTile) : 0;
  m.query_base = qb + m.q_local_base;
  m.k_start = p.cu_k[batch];
  m.kv_len = p.cu_k[batch + 1] - m.k_start;
  m.qoff = p.causal ? (m.kv_len - ql) : 0;
  int nb = (m.kv_len + 127) / 128;
  if (p.causal) {
    int vis = (m.qoff + m.q_local_base + m.q_valid + 127) / 128;
    nb = min(nb, vis);
  }
  if (m.q_valid == 0 || nb <= 0) nb = 1;
  m.num_n_blocks = nb;
  return m;
}

// Delayed-max protocol (round 13): softmax stores P with the STALE row
// basis immediately after the TMEM score load (no max/drift chain on the
// P-store critical path); the true tile max is computed after the store.
// Correction rescales the O accumulator AFTER the block's PV commits
// (post-add rescale), gated by pgate (MMA->corr) / rdone (corr->MMA).

template <typename QT, bool IS_BF16, int MODE = 0>
__global__ void __launch_bounds__(512, 1)
    msa_g16_umma_kernel(const __grid_constant__ G16Params p,
                        const __grid_constant__ CUtensorMap q_map,
                        const __grid_constant__ CUtensorMap k_map,
                        const __grid_constant__ CUtensorMap v_map) {
  const CUtensorMap* qtm = &q_map;
  const CUtensorMap* ktm = &k_map;
  const CUtensorMap* vtm = &v_map;
  const int tid = threadIdx.x;
  const int warp = warp_uni(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  const int smem = smem_u32(smem_raw);

  // Upper-bound grid: tiles past the real tile count exit before any setup
  // so they don't stream KV or allocate TMEM.
  if (compute_meta(p).q_valid == 0) return;

  __syncthreads();

  // Round-18 MODE>0 probe: two-CTA cluster parity split of the KV block
  // stream. crank selects this CTA's half; blocks map as
  // n_block = first_blk - 2*i. MODE==0 keeps the single-CTA stride-1 sweep.
  const int crank = (MODE > 0) ? (int)g16_cluster_rank() : 0;
  int nb_r, first_blk, step;
  {
    const int nb = compute_meta(p).num_n_blocks;
    if (MODE > 0) {
      nb_r = (nb + 1 - crank) >> 1;
      first_blk = nb - 1 - ((((nb - 1) & 1) != crank) ? 1 : 0);
      step = 2;
    } else {
      nb_r = nb;
      first_blk = nb - 1;
      step = 1;
    }
    if (nb_r <= 0) return;  // probe only: rank with no blocks exits (nb==1)
  }

  float* scale_smem = reinterpret_cast<float*>(smem_raw + kScaleOff);
  int* q2k_smem = reinterpret_cast<int*>(smem_raw + kQ2KOff);

  // mbarrier layout (bytes): q_full[2] @0, q2k_full @16, kv_full[3] @24,
  // kv_empty[3] @48, s_full[2] @72, p_full[2] @88, p_tail[2] @104,
  // corr_sig[2] @120, corr_done[2] @136, o_full[2] @152, tmem_dealloc @168,
  // pgate[2] @176, (tmem_hold @192), rdone[2] @200.
  const int b_qfull = smem + 0;
  const int b_q2kfull = smem + 16;
  const int b_kvfull = smem + 24;
  const int b_kvempty = smem + 48;
  const int b_sfull = smem + 72;
  const int b_pfull = smem + 88;
  const int b_ptail = smem + 104;
  const int b_corrsig = smem + 120;
  const int b_corrdone = smem + 136;
  const int b_ofull = smem + 152;
  const int b_dealloc = smem + 168;
  const int b_pgate = smem + 176;
  const int b_rdone = smem + 200;

  if (warp == 0 && elect_one()) {
    mbar_init(b_qfull + 0, 1);
    mbar_init(b_qfull + 8, 1);
    mbar_init(b_q2kfull, 32);
    for (int s = 0; s < 3; s++) mbar_init(b_kvfull + s * 8, 1);
    for (int s = 0; s < 3; s++) mbar_init(b_kvempty + s * 8, 1);
    mbar_init(b_sfull + 0, 1);
    mbar_init(b_sfull + 8, 1);
    mbar_init(b_pfull + 0, 128);
    mbar_init(b_pfull + 8, 128);
    mbar_init(b_ptail + 0, 128);
    mbar_init(b_ptail + 8, 128);
    mbar_init(b_corrsig + 0, 128);
    mbar_init(b_corrsig + 8, 128);
    mbar_init(b_corrdone + 0, 128);
    mbar_init(b_corrdone + 8, 128);
    mbar_init(b_ofull + 0, 1);
    mbar_init(b_ofull + 8, 1);
    mbar_init(b_dealloc, 128);
    // delayed-max protocol: MMA commits pgate after each stage's PV;
    // correction arrives rdone after the post-PV O rescale.
    mbar_init(b_pgate + 0, 1);
    mbar_init(b_pgate + 8, 1);
    mbar_init(b_rdone + 0, 128);
    mbar_init(b_rdone + 8, 128);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  volatile int* tmem_hold = (volatile int*)(smem_raw + 192);
  if (warp == 0) tc_alloc(smem + 192, 512);
  __syncthreads();
  tc_fence_after_sync();
  const int taddr = tmem_hold[0];

  if (warp >= 12) sreg_dec_48();

  const int trow_base = (warp % 4) * 32 << 16;
  const int my_row = (warp % 4) * 32 + lane;
  const uint32_t idesc_qk = kIdescQK - (IS_BF16 ? 0u : 0x480u);
  const uint32_t idesc_pv = kIdescPV - (IS_BF16 ? 0u : 0x480u);

  // ================= Role: softmax (warps 0-7) =================
  if (warp <= 7) {
    sreg_inc_192();
    const TileMeta m = compute_meta(p);
    const int stage = warp_uni(warp / 4);
    const int sbase = stage * 128;
    const int query_in_stage = my_row / 16;
    const int stage_valid = min(max(m.q_valid - stage * 8, 0), 8);
    const int query_in_tile = stage * 8 + query_in_stage;
    const int row_valid = (query_in_stage < stage_valid) ? 1 : 0;
    // running basis in scaled (exp2-log2) units; basis 0 is always safe
    // (a single drift >= 8 exp2 units triggers a rescale, bounding P <= 2^8+).
    float row_basis_scaled = 0.0f;
    float row_sum = 0.0f;

    mbar_wait(b_q2kfull, 0);
    unsigned int selection_mask = 0;
    if ((lane % 16) == 0 && row_valid && m.num_n_blocks <= 32) {
      for (int slot = 0; slot < p.topk; slot++) {
        int sel = q2k_smem[slot * 16 + query_in_tile];
        if (sel >= 0 && sel < 32) selection_mask |= (1u << sel);
      }
    }
    selection_mask = __shfl_sync(0xFFFFFFFFu, selection_mask, lane / 16 * 16);

    unsigned phase_sfull = 0, phase_corrdone = 0;
    const float scale = p.scale_log2e;
#pragma unroll 1
    for (int n_iter = 0; n_iter < nb_r; n_iter++) {
      const int n_block = first_blk - step * n_iter;
      mbar_wait(b_sfull + stage * 8, phase_sfull);
      phase_sfull ^= 1;
      int selected = 0;
      if (row_valid) {
        if (m.num_n_blocks <= 32) {
          selected = (selection_mask >> n_block) & 1u;
        } else {
          for (int slot = 0; slot < p.topk; slot++)
            selected |= (q2k_smem[slot * 16 + query_in_tile] == n_block);
        }
      }
      int valid_cols = 0;
      if (selected) {
        valid_cols = m.kv_len - n_block * 128;
        if (valid_cols > 128) valid_cols = 128;
        if (p.causal) {
          int qp = m.qoff + m.q_local_base + query_in_tile;
          int cc = qp - n_block * 128 + 1;
          if (valid_cols > cc) valid_cols = cc;
        }
        if (valid_cols < 0) valid_cols = 0;
      }
      const int p_base_skip = taddr + sbase + 64 + trow_base;
      if (!__any_sync(0xFFFFFFFFu, selected)) {
        // No valid token anywhere in this warp for this block: P is exactly
        // zero in any basis, the row sums don't change, and no drift is
        // possible. Store zeros and keep the arrival pattern identical to the
        // slow path, but skip the score load, exp2 sweep, and max chain.
        uint32_t pkz[32];
#pragma unroll
        for (int j = 0; j < 32; j++) pkz[j] = 0u;
        tmem_st32(p_base_skip, pkz);
        tmem_st16u(p_base_skip + 32, pkz);
        tc_wait_st();
        mbar_arrive(b_pfull + stage * 8);
        tmem_st16u(p_base_skip + 48, pkz);
        tc_wait_st();
        mbar_arrive(b_ptail + stage * 8);
        scale_smem[sbase + my_row] = 1.0f;
        mbar_arrive(b_corrsig + stage * 8);
        mbar_wait(b_corrdone + stage * 8, phase_corrdone);
        phase_corrdone ^= 1;
        continue;
      }
      const int sc_base = taddr + sbase + trow_base;
      float s0[64], s1[64];
      tmem_ld64(s0, sc_base);
      tmem_ld64(s1, sc_base + 64);
      // Selection is uniform per 16-lane group (rows are query*qhead pairs and
      // every qhead of a (kv_head, query) shares one block list), so the two
      // halves of the warp can diverge cheaply: an unselected half runs a
      // zeros-only body and skips the whole exp2/max chain (~50% of groups
      // for topk=16/32-block prefills), while a selected half runs the full
      // pipeline. The per-element -INF sweep for causal-diagonal rows is
      // additionally gated on a warp vote (~3/32 block iterations).
      const int p_base = taddr + sbase + 64 + trow_base;
      // warp-converged vote BEFORE the 16-lane-group divergence below (a
      // full-mask vote inside the diverged branch deadlocks).
      const bool warp_partial = __any_sync(0xFFFFFFFFu, valid_cols > 0 && valid_cols < 128);
      const float score_bias = -row_basis_scaled;
      const uint64_t scale2 = pack2(scale, scale);
      const uint64_t bias2 = pack2(score_bias, score_bias);
      const uint64_t* s0w = reinterpret_cast<const uint64_t*>(s0);
      const uint64_t* s1w = reinterpret_cast<const uint64_t*>(s1);
      uint64_t sum0 = pack2(0.f, 0.f), sum1 = pack2(0.f, 0.f);
      uint32_t pk[32];
      int body_v = 0, tail_v = 0;
      if (valid_cols > 0) {
        // delayed-max: exponentiate with the STALE row basis right away so
        // the P store does not wait for the tile-max chain.
        body_v = min(valid_cols, 128);
        tail_v = max(valid_cols - 64, 0);
        if (warp_partial) {
#pragma unroll
          for (int j = 0; j < 64; j++) {
            if (j >= body_v && body_v < 64) s0[j] = -CUDART_INF_F;
            if (j >= tail_v && tail_v < 64) s1[j] = -CUDART_INF_F;
          }
        }
#pragma unroll
        for (int j = 0; j < 32; j++) {
          uint64_t x = fma2(s0w[j], scale2, bias2);
          float2 e = *(float2*)&x;
          e.x = fast_exp2(e.x);
          e.y = fast_exp2(e.y);
          sum0 = add2(sum0, *(uint64_t*)&e);
          pk[j] = IS_BF16 ? pack_bf16x2(e.x, e.y) : pack_f16x2(e.x, e.y);
        }
      } else {
        // zero row: exact-zero P in any basis without the exp2 sweep.
#pragma unroll
        for (int j = 0; j < 32; j++) pk[j] = 0u;
      }
      tmem_st32(p_base, pk);
      if (valid_cols > 0) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
          uint64_t x = fma2(s1w[j], scale2, bias2);
          float2 e = *(float2*)&x;
          e.x = fast_exp2(e.x);
          e.y = fast_exp2(e.y);
          sum1 = add2(sum1, *(uint64_t*)&e);
          pk[j] = IS_BF16 ? pack_bf16x2(e.x, e.y) : pack_f16x2(e.x, e.y);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 16; j++) pk[j] = 0u;
      }
      tmem_st16u(p_base + 32, pk);
      tc_wait_st();
      mbar_arrive(b_pfull + stage * 8);
      if (valid_cols > 0) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
          uint64_t x = fma2(s1w[16 + j], scale2, bias2);
          float2 e = *(float2*)&x;
          e.x = fast_exp2(e.x);
          e.y = fast_exp2(e.y);
          sum1 = add2(sum1, *(uint64_t*)&e);
          pk[j] = IS_BF16 ? pack_bf16x2(e.x, e.y) : pack_f16x2(e.x, e.y);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 16; j++) pk[j] = 0u;
      }
      tmem_st16u(p_base + 48, pk);
      tc_wait_st();
      mbar_arrive(b_ptail + stage * 8);
      const uint64_t stot = add2(sum0, sum1);
      const float2 sf = *(const float2*)&stot;
      const float block_sum = sf.x + sf.y;

      // true tile max + drift bookkeeping (off the P-store critical path);
      // correction consumes acc_scale and rescales O after this block's PV.
      // Zero rows skip the chain entirely: max stays -INF (no drift).
      float tmax0 = -CUDART_INF_F, tmax1 = -CUDART_INF_F;
      if (valid_cols > 0) {
        float mx0 = -CUDART_INF_F, mx1 = -CUDART_INF_F, mx2 = -CUDART_INF_F, mx3 = -CUDART_INF_F;
        float my0 = -CUDART_INF_F, my1 = -CUDART_INF_F, my2 = -CUDART_INF_F, my3 = -CUDART_INF_F;
#pragma unroll
        for (int j = 0; j < 16; j++) {
          mx0 = fmaxf(mx0, s0[2 * j]);
          mx1 = fmaxf(mx1, s0[2 * j + 1]);
          mx2 = fmaxf(mx2, s0[32 + 2 * j]);
          mx3 = fmaxf(mx3, s0[33 + 2 * j]);
          my0 = fmaxf(my0, s1[2 * j]);
          my1 = fmaxf(my1, s1[2 * j + 1]);
          my2 = fmaxf(my2, s1[32 + 2 * j]);
          my3 = fmaxf(my3, s1[33 + 2 * j]);
        }
        tmax0 = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
        tmax1 = fmaxf(fmaxf(my0, my1), fmaxf(my2, my3));
        if (tail_v <= 0) tmax1 = -CUDART_INF_F;
      }
      const float tile_max_scaled = fmaxf(tmax0, tmax1) * scale;
      float acc_scale = 1.0f;
      if (tile_max_scaled - row_basis_scaled >= 8.0f) {
        // drift >= 8 exp2 units: adopt the new basis; PV_i was accumulated
        // in the stale basis and correction rescales O post-add by acc_scale.
        acc_scale = fast_exp2(row_basis_scaled - tile_max_scaled);
        row_basis_scaled = tile_max_scaled;
      }
      scale_smem[sbase + my_row] = acc_scale;
      mbar_arrive(b_corrsig + stage * 8);

      mbar_wait(b_corrdone + stage * 8, phase_corrdone);
      phase_corrdone ^= 1;
      // stale-basis folding: this block's P (and its sum) were computed in
      // the OLD basis, so both sides move to the new basis on drift.
      row_sum = (row_sum + block_sum) * acc_scale;  // acc_scale == 1 unless drift
    }
    scale_smem[256 + sbase + my_row] = row_sum;
    mbar_arrive(b_corrsig + stage * 8);
    return;
  }
  // ==== ANCHOR:ROLE_CORR ====
  // ================= Role: correction + epilogue (warps 8-11) =================
  if (warp >= 8 && warp <= 11) {
    sreg_dec_80();
    const TileMeta m = compute_meta(p);
    // delayed-max: PV_0 needs no prior rescale — pre-arrive rdone for both
    // stages; afterwards each block's rdone comes from the post-PV rescale.
    // The O ld/st rescale runs only when drift actually occurred.
    mbar_arrive(b_rdone + 0);
    mbar_arrive(b_rdone + 8);
    unsigned ph_sig0 = 0, ph_sig1 = 0, ph_gate0 = 0, ph_gate1 = 0;
#pragma unroll 1
    for (int i = 0; i < nb_r; i++) {
      mbar_wait(b_corrsig + 0, ph_sig0);
      ph_sig0 ^= 1;
      float f0 = scale_smem[my_row];
      // corrdone only guards scale_smem reuse — arrive immediately so the
      // softmax loop is not serialized behind this block's PV/pgate chain.
      mbar_arrive(b_corrdone + 0);
      mbar_wait(b_corrsig + 8, ph_sig1);
      ph_sig1 ^= 1;
      float f1 = scale_smem[128 + my_row];
      mbar_arrive(b_corrdone + 8);
      mbar_wait(b_pgate + 0, ph_gate0);
      ph_gate0 ^= 1;
      if (__any_sync(0xFFFFFFFFu, f0 < 1.0f)) {
        const uint64_t f02 = pack2(f0, f0);
#pragma unroll
        for (int c = 0; c < 8; c++) {
          float o[16];
          const int oa = taddr + 256 + trow_base + c * 16;
          tmem_ld16(o, oa);
          uint64_t* ow = reinterpret_cast<uint64_t*>(o);
#pragma unroll
          for (int j = 0; j < 8; j++) ow[j] = mul2(ow[j], f02);
          tmem_st16f(oa, o);
        }
        tc_wait_st();
      }
      mbar_arrive(b_rdone + 0);
      mbar_wait(b_pgate + 8, ph_gate1);
      ph_gate1 ^= 1;
      if (__any_sync(0xFFFFFFFFu, f1 < 1.0f)) {
        const uint64_t f12 = pack2(f1, f1);
#pragma unroll
        for (int c = 0; c < 8; c++) {
          float o[16];
          const int oa = taddr + 384 + trow_base + c * 16;
          tmem_ld16(o, oa);
          uint64_t* ow = reinterpret_cast<uint64_t*>(o);
#pragma unroll
          for (int j = 0; j < 8; j++) ow[j] = mul2(ow[j], f12);
          tmem_st16f(oa, o);
        }
        tc_wait_st();
      }
      mbar_arrive(b_rdone + 8);
    }
    // Final drain v2 (round 25): per-stage waits so the stage-0 drain overlaps
    // the trailing stage-1 PV, and a 2-deep tcgen05.ld software pipeline so
    // chunk c+1's TMEM latency is hidden by chunk c's convert+store. All
    // tcgen05.ld sites stay warp-uniform; only convert/store is row-guarded.
#pragma unroll
    for (int stage = 0; stage < 2; stage++) {
      const int sbase = stage * 128;
      const int stage_valid = min(max(m.q_valid - stage * 8, 0), 8);
      const int query_in_stage = my_row / 16;
      mbar_wait(b_corrsig + stage * 8, stage == 0 ? ph_sig0 : ph_sig1);
      float inv_sum = 0.0f;
      if (query_in_stage < stage_valid) {
        const float l = scale_smem[256 + sbase + my_row];
        inv_sum = (l > 0.0f && l == l) ? (1.0f / l) : 0.0f;
      }
      mbar_wait(b_ofull + stage * 8, 0);
      const int query = m.query_base + stage * 8 + query_in_stage;
      const int ohead = m.q_head + (my_row % 16);
      const long orow = ((long)(query * p.num_q_heads + ohead)) * kHead;
      const int obase = taddr + 256 + sbase + trow_base;
      float o0[16], o1[16], o2[16];
      tmem_ld16(o0, obase);
      tmem_ld16(o1, obase + 16);
      tmem_ld16(o2, obase + 32);
#pragma unroll
      for (int c = 0; c < 8; c++) {
        float* ocur = (c % 3 == 0) ? o0 : ((c % 3 == 1) ? o1 : o2);
        if (query_in_stage < stage_valid) {
          uint32_t pk[8];
#pragma unroll
          for (int j = 0; j < 8; j++)
            pk[j] = IS_BF16 ? pack_bf16x2(ocur[2 * j] * inv_sum, ocur[2 * j + 1] * inv_sum)
                            : pack_f16x2(ocur[2 * j] * inv_sum, ocur[2 * j + 1] * inv_sum);
          *reinterpret_cast<uint4*>(reinterpret_cast<char*>(p.out) +
                                    (orow + c * 16) * (long)sizeof(QT)) =
              *reinterpret_cast<uint4*>(&pk[0]);
          *reinterpret_cast<uint4*>(reinterpret_cast<char*>(p.out) +
                                    (orow + c * 16 + 8) * (long)sizeof(QT)) =
              *reinterpret_cast<uint4*>(&pk[4]);
        }
        // refetch into the just-freed buffer: chunk c+3 for the next lap
        if (c + 3 < 8) tmem_ld16(ocur, obase + (c + 3) * 16);
      }
    }
    mbar_arrive(b_dealloc);
    return;
  }
  // ==== ANCHOR:ROLE_MMA ====
  // ================= Role: MMA leader (warp 12) =================
  if (warp == 12) {
    const TileMeta m = compute_meta(p);
    mbar_wait(b_qfull + 0, 0);
    mbar_wait(b_qfull + 8, 0);
    unsigned kv_stage = 0, kv_phase = 0;
    const int q0_lo = warp_uni(((smem + kQ0Off) >> 4) & 0x3FFF);
    const int q1_lo = warp_uni(((smem + kQ1Off) >> 4) & 0x3FFF);
    const int ring_lo = warp_uni(((smem + kRingOff) >> 4) & 0x3FFF);
    mbar_wait(b_kvfull + kv_stage * 8, kv_phase);
    int kb_lo = ring_lo + kv_stage * 2048;
    mma_qk_group(q0_lo, kb_lo, taddr + 0, idesc_qk, 0);
    tc_commit(b_sfull + 0);
    mma_qk_group(q1_lo, kb_lo, taddr + 128, idesc_qk, 0);
    tc_commit(b_sfull + 8);
    tc_commit(b_kvempty + kv_stage * 8);
    kv_stage++;
    if (kv_stage == 3) {
      kv_stage = 0;
      kv_phase ^= 1;
    }
    int first_pv = 1;
    unsigned ph_p0 = 0, ph_pt0 = 0, ph_p1 = 0, ph_pt1 = 0;
    unsigned ph_r0 = 0, ph_r1 = 0;
#pragma unroll 1
    for (int it = 0; it < nb_r - 1; it++) {
      const unsigned vs = kv_stage, vp = kv_phase;
      kv_stage++;
      if (kv_stage == 3) {
        kv_stage = 0;
        kv_phase ^= 1;
      }
      mbar_wait(b_kvfull + vs * 8, vp);
      mbar_wait(b_pfull + 0, ph_p0);
      ph_p0 ^= 1;
      mbar_wait(b_rdone + 0, ph_r0);
      ph_r0 ^= 1;
      int v_lo = (ring_lo | 0x4000000) + vs * 2048;
      mma_pv_head(v_lo, taddr + 256, taddr + 64, idesc_pv, first_pv ? 0 : 1);
      mbar_wait(b_ptail + 0, ph_pt0);
      ph_pt0 ^= 1;
      mma_pv_tail(v_lo, taddr + 256, taddr + 64, idesc_pv);
      tc_commit(b_pgate + 0);
      const unsigned ks = kv_stage, kp = kv_phase;
      kv_stage++;
      if (kv_stage == 3) {
        kv_stage = 0;
        kv_phase ^= 1;
      }
      mbar_wait(b_kvfull + ks * 8, kp);
      kb_lo = ring_lo + ks * 2048;
      mma_qk_group(q0_lo, kb_lo, taddr + 0, idesc_qk, 0);
      tc_commit(b_sfull + 0);
      mbar_wait(b_pfull + 8, ph_p1);
      ph_p1 ^= 1;
      mbar_wait(b_rdone + 8, ph_r1);
      ph_r1 ^= 1;
      mma_pv_head(v_lo, taddr + 384, taddr + 192, idesc_pv, first_pv ? 0 : 1);
      mbar_wait(b_ptail + 8, ph_pt1);
      ph_pt1 ^= 1;
      mma_pv_tail(v_lo, taddr + 384, taddr + 192, idesc_pv);
      tc_commit(b_pgate + 8);
      tc_commit(b_kvempty + vs * 8);
      mma_qk_group(q1_lo, kb_lo, taddr + 128, idesc_qk, 0);
      tc_commit(b_sfull + 8);
      tc_commit(b_kvempty + ks * 8);
      first_pv = 0;
    }
    mbar_wait(b_kvfull + kv_stage * 8, kv_phase);
    mbar_wait(b_pfull + 0, ph_p0);
    mbar_wait(b_rdone + 0, ph_r0);
    const int v_lo = (ring_lo | 0x4000000) + kv_stage * 2048;
    mma_pv_head(v_lo, taddr + 256, taddr + 64, idesc_pv, first_pv ? 0 : 1);
    mbar_wait(b_ptail + 0, ph_pt0);
    mma_pv_tail(v_lo, taddr + 256, taddr + 64, idesc_pv);
    tc_commit(b_pgate + 0);
    mbar_wait(b_pfull + 8, ph_p1);
    mbar_wait(b_rdone + 8, ph_r1);
    mma_pv_head(v_lo, taddr + 384, taddr + 192, idesc_pv, first_pv ? 0 : 1);
    mbar_wait(b_ptail + 8, ph_pt1);
    mma_pv_tail(v_lo, taddr + 384, taddr + 192, idesc_pv);
    tc_commit(b_pgate + 8);
    tc_commit(b_kvempty + kv_stage * 8);
    tc_commit(b_ofull + 0);
    tc_commit(b_ofull + 8);
    mbar_wait(b_dealloc, 0);
    tc_dealloc(taddr, 512);
    tc_relinquish();
    return;
  }
  // ==== ANCHOR:ROLE_PRODUCER ====
  // ================= Role: producer (warp 14) =================
  if (warp == 14) {
    const TileMeta m = compute_meta(p);
    // q2k copy: rows [slot][16] ints, only rows < q_valid are consumed
    const int* q2k_src = p.q2k + ((long)m.kv_head * p.total_q + m.query_base) * p.topk;
    const int q2k_total = m.q_valid * p.topk;
    if ((p.topk & (p.topk - 1)) == 0) {
      // power-of-two topk: SASS-level division per element (mulhi + branchy
      // fixup, ~20 instr) was the producer's pre-pipeline hotspot; shift/mask
      // is 2 ALU ops and keeps identical (qrow, slot) mapping.
      const int sh = __ffs(p.topk) - 1;
      const int tmask = p.topk - 1;
      for (int idx = lane * 4; idx < q2k_total; idx += 128) {
        int4 v = *reinterpret_cast<const int4*>(q2k_src + idx);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int flat = idx + j;
          q2k_smem[(flat & tmask) * 16 + (flat >> sh)] = (&v.x)[j];
        }
      }
    } else {
      for (int idx = lane * 4; idx < q2k_total; idx += 128) {
        int4 v = *reinterpret_cast<const int4*>(q2k_src + idx);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int flat = idx + j;
          int qrow = flat / p.topk;
          int slot = flat - qrow * p.topk;
          q2k_smem[slot * 16 + qrow] = (&v.x)[j];
        }
      }
    }
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_arrive(b_q2kfull);
    if (elect_one()) {
      mbar_arrive_tx(b_qfull + 0, 32768);
      tma_load_4d(smem + kQ0Off, qtm, 0, m.q_head, m.query_base, 0, b_qfull + 0);
      mbar_arrive_tx(b_qfull + 8, 32768);
      tma_load_4d(smem + kQ1Off, qtm, 0, m.q_head, m.query_base + 8, 0, b_qfull + 8);
    }
    unsigned stage = 0, empty_phase = 1;
#pragma unroll 1
    for (int n_iter = 0; n_iter < nb_r; n_iter++) {
      const int n_block = first_blk - step * n_iter;
      const int token_base = m.k_start + n_block * 128;
      mbar_wait(b_kvempty + stage * 8, empty_phase);
      if (elect_one()) {
        const int dst = smem + kRingOff + stage * 32768;
        mbar_arrive_tx(b_kvfull + stage * 8, 32768);
        tma_load_4d(dst, ktm, 0, token_base, 0, m.kv_head, b_kvfull + stage * 8);
        tma_load_4d(dst + 8192, ktm, 0, token_base + 64, 0, m.kv_head, b_kvfull + stage * 8);
        tma_load_4d(dst + 16384, ktm, 0, token_base, 1, m.kv_head, b_kvfull + stage * 8);
        tma_load_4d(dst + 24576, ktm, 0, token_base + 64, 1, m.kv_head, b_kvfull + stage * 8);
      }
      stage++;
      if (stage == 3) {
        stage = 0;
        empty_phase ^= 1;
      }
      mbar_wait(b_kvempty + stage * 8, empty_phase);
      if (elect_one()) {
        const int dst = smem + kRingOff + stage * 32768;
        mbar_arrive_tx(b_kvfull + stage * 8, 32768);
        tma_load_4d(dst, vtm, 0, token_base, 0, m.kv_head, b_kvfull + stage * 8);
        tma_load_4d(dst + 8192, vtm, 0, token_base + 64, 0, m.kv_head, b_kvfull + stage * 8);
        tma_load_4d(dst + 16384, vtm, 0, token_base, 1, m.kv_head, b_kvfull + stage * 8);
        tma_load_4d(dst + 24576, vtm, 0, token_base + 64, 1, m.kv_head, b_kvfull + stage * 8);
      }
      stage++;
      if (stage == 3) {
        stage = 0;
        empty_phase ^= 1;
      }
    }
    return;
  }
  // warps 13, 15: idle
}

// ==== ANCHOR:HOST ====
// Q viewed as [total_q, q_heads, 128] -> 4D map {64, q_heads, total_q, 2},
// box {64, 16, 8, 2} = 32 KB per stage (8 queries x 16 heads x 128 dims).
static CUtensorMap encode_q_map(const void* base, int total_q, int q_heads,
                                CUtensorMapDataType dt) {
  uint64_t gdim[4] = {64ull, (uint64_t)q_heads, (uint64_t)total_q, 2ull};
  uint64_t gstride[3] = {256ull, (uint64_t)q_heads * 256ull, 128ull};
  uint32_t box[4] = {64u, 16u, 8u, 2u};
  uint32_t estride[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  cuTensorMapEncodeTiled(&tm, dt, 4, const_cast<void*>(base), gdim, gstride, box, estride,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tm;
}

// K/V viewed as [total_k, kv_heads, 128] -> 4D map {64, total_k, 2, kv_heads},
// box {64, 64, 1, 1} = 8 KB per call (4 calls per 128-token block half-tensor).
static CUtensorMap encode_kv_map(const void* base, int total_k, int kv_heads,
                                 CUtensorMapDataType dt) {
  uint64_t gdim[4] = {64ull, (uint64_t)total_k, 2ull, (uint64_t)kv_heads};
  uint64_t gstride[3] = {(uint64_t)kv_heads * 256ull, 128ull, 256ull};
  uint32_t box[4] = {64u, 64u, 1u, 1u};
  uint32_t estride[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  cuTensorMapEncodeTiled(&tm, dt, 4, const_cast<void*>(base), gdim, gstride, box, estride,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tm;
}

template <typename QT, bool IS_BF16>
static void launch_g16(const G16Params& p, const CUtensorMap& qm, const CUtensorMap& km,
                       const CUtensorMap& vm, cudaStream_t stream) {
  static bool attr_set = false;
  const int smem_bytes = kQ2KOff + p.topk * 16 * 4;
  if (!attr_set) {
    cudaFuncSetAttribute(msa_g16_umma_kernel<QT, IS_BF16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    attr_set = true;
  }
  dim3 grid(p.num_kv_heads, p.total_q / kTile + p.nbatch, 1);
  msa_g16_umma_kernel<QT, IS_BF16><<<grid, kThreads, smem_bytes, stream>>>(p, qm, km, vm);
}

void umma_g16_forward(const void* q, bool q_is_bf16, const void* k, const void* v, const int* q2k,
                      const int* cu_q, const int* cu_k, void* out, int total_q, int total_k,
                      int num_q_heads, int num_kv_heads, int topk, int nbatch, bool causal,
                      cudaStream_t stream) {
  G16Params p;
  p.q = q;
  p.k = k;
  p.v = v;
  p.q2k = q2k;
  p.cu_q = cu_q;
  p.cu_k = cu_k;
  p.out = out;
  p.total_q = total_q;
  p.total_k = total_k;
  p.num_q_heads = num_q_heads;
  p.num_kv_heads = num_kv_heads;
  p.topk = topk;
  p.nbatch = nbatch;
  p.causal = causal ? 1 : 0;
  p.scale_log2e = (float)(0.08838834764831845 * M_LOG2E);  // 128^-0.5 * log2(e)
  const CUtensorMapDataType dt =
      q_is_bf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  CUtensorMap qm = encode_q_map(p.q, p.total_q, p.num_q_heads, dt);
  CUtensorMap km = encode_kv_map(p.k, p.total_k, p.num_kv_heads, dt);
  CUtensorMap vm = encode_kv_map(p.v, p.total_k, p.num_kv_heads, dt);
  if (q_is_bf16) {
    launch_g16<__nv_bfloat16, true>(p, qm, km, vm, stream);
  } else {
    launch_g16<__half, false>(p, qm, km, vm, stream);
  }
  MSAV_CHECK(cudaGetLastError() == cudaSuccess, "msa_g16_umma_kernel launch failed");
}

// Flat-only UMMA route: g16 with dense bf16/fp16 KV, enough queries per CTA
// tile, and a topk range whose q2k smem slice fits the 227 KB budget.
bool umma_g16_eligible(int group, int seqlen_q, int topk, int kv_dtype_code, bool paged,
                       bool causal_supported) {
  return group == kGroup && seqlen_q >= kTile && !paged && kv_dtype_code != 2 && topk >= 12 &&
         topk <= 64 && (topk % 4) == 0 && causal_supported;
}

}  // namespace msa_umma_g16
