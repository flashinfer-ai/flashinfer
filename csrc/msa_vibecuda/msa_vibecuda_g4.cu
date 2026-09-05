// Block-bucketed UMMA/TMEM sparse attention for group_size==4 paged KV (SM100).
// Round 24: the per-head dense-stream design (our msa_umma_g16.cu and the
// baseline cake m128 prefill) computes QK/PV for every block every tile
// streams; on the topk << num_blocks regime (case 9: 4 of 64 blocks) that is
// ~12x MMA waste, which is why the JV HMMA PACK path lost 0.73x. This file
// implements the qualitatively different variant gated to group==4 + paged:
// a device routing pass buckets query rows by (kv_head, batch, logical KV
// block); each CTA then handles one (bucket, 256-row tile) whose rows ALL
// selected the streamed block, so no wasted MMA work at all. Softmax is
// single-block per tile; cross-block merging uses the split-KV flash
// decoding scheme ((m, l, fp16 acc) partials in a workspace + a merge
// kernel). The tensor-core pipeline reuses the proven g16 UMMA/TMEM dataflow
// (M128 stages, TMEM S/P/O layout, descriptor walks) verbatim; the KV slab
// layout matches the baseline paged m128 kernel: [elhalf 16KB][tokhalf 8KB]
// [64 tok rows x 128B] with SWIZZLE_128B.
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

#include <cstdlib>

#include "msa_vibecuda_common.h"

namespace msa_umma_g4 {

constexpr int kGroup = 4;   // q heads per kv head (structural gate)
constexpr int kQTile = 32;  // queries per routing tile (32 x 4 heads = 128 rows);
                            // the dormant 512-thread kernel assumed two such
                            // tiles per CTA and must not be dispatched now
constexpr int kHead = 128;
constexpr int kThreads = 512;
constexpr int kMaxTopk = 8;  // routing arrays + slot-rank packing budget

// ---- smem map (dynamic; offsets must stay 1024-aligned for TMA/UMMA) --
constexpr int kBarBytes = 256;  // mbarriers + tmem_hold at [0, 256)
constexpr int kMlOff = 1024;    // ml_m[256] | ml_l[256] floats = 2KB
constexpr int kRowsOff = 3072;  // rown[64] | rowrank[64] | rowvcol[64] = 768B
constexpr int kQ0Off = 4096;    // 32 KB stage 0
constexpr int kQ1Off = 36864;   // 32 KB stage 1
constexpr int kKOff = 69632;    // 32 KB K slab stage 0
constexpr int kVOff = 102400;   // 32 KB V slab stage 0
constexpr int kK1Off = 135168;  // 32 KB K slab stage 1 (paired chunk)
constexpr int kV1Off = 167936;  // 32 KB V slab stage 1 (paired chunk)
constexpr int kSmemTotal = 200704;

// mbarrier byte offsets inside the [0, kBarBytes) region
constexpr int kB_qfull = 0;      // 2
constexpr int kB_kvfull = 16;    // 2 (K, V) stage 0
constexpr int kB_sfull = 32;     // 2
constexpr int kB_pfull = 48;     // 2
constexpr int kB_ptail = 64;     // 2
constexpr int kB_ofull = 80;     // 2
constexpr int kB_dealloc = 96;   // 1
constexpr int kB_kv2full = 112;  // 2 (K, V) stage 1
constexpr int kTmemHold = 192;

struct G4Params {
  const void* q;  // [total_q, Hq, 128]
  const void* k;  // paged: [pages, Hkv, 128, 128]
  const void* v;
  const int* q2k;         // [Hkv, total_q, topk]
  const int* cu_q;        // [nbatch+1]
  const int* cu_k;        // [nbatch+1]
  const int* page_table;  // [nbatch, max_pages]
  void* out;              // [total_q, Hq, 128]
  float* ws_acc;          // [rows, topk, 128] fp16 partial accum (2B/elem)
  float* ws_ml;           // [rows, topk, 2] fp32 (m raw, l)
  int* bcnt;              // [nbuckets] zeroed
  int* bcnt2;             // [nbuckets] zeroed (scatter cursor)
  int* boff;              // [nbuckets+1] padded row offsets
  int* btoff;             // [nbuckets+1] tile offsets
  int* bmeta;             // [nbuckets*5]
  int* chunk_bkt;         // [tiles_bound]
  int* cnt_hn;            // [Hkv*total_q] zeroed
  int* rankmap;           // [Hkv*total_q*topk]
  int* rowlist;           // [rows_bound] packed (query | rank<<28)
  int* tile_total;        // [1]
  int* route_bar;         // [4] per-call grid-barrier cnt/flag pairs, host-zeroed
  int* rowcnt;            // [total_q*Hq] per-row completion counters (zeroed)
  int inline_merge;       // 1: last-arriving tile merges in-kernel (no merge k)
  long pt_stride, q_tok, q_head, o_tok, o_head;
  int total_q, num_q_heads, num_kv_heads, topk, nbatch, max_pages, nbuckets;
  int rows_bound, tiles_bound;
  int seqlen_q, causal;
  float scale_log2e;
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

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 4D TMA load (paged KV); dst mbar gets the transaction bytes.
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

__device__ __forceinline__ void tmem_ld16(float* d, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]),
        "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
        "=f"(d[14]), "=f"(d[15])
      : "r"(addr));
}

__device__ __forceinline__ void tmem_st32(int addr, const uint32_t* s) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(addr),
      "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
      "r"(s[8]), "r"(s[9]), "r"(s[10]), "r"(s[11]), "r"(s[12]), "r"(s[13]), "r"(s[14]), "r"(s[15]),
      "r"(s[16]), "r"(s[17]), "r"(s[18]), "r"(s[19]), "r"(s[20]), "r"(s[21]), "r"(s[22]),
      "r"(s[23]), "r"(s[24]), "r"(s[25]), "r"(s[26]), "r"(s[27]), "r"(s[28]), "r"(s[29]),
      "r"(s[30]), "r"(s[31])
      : "memory");
}

__device__ __forceinline__ void tmem_st16u(int addr, const uint32_t* s) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(addr),
      "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(s[4]), "r"(s[5]), "r"(s[6]), "r"(s[7]),
      "r"(s[8]), "r"(s[9]), "r"(s[10]), "r"(s[11]), "r"(s[12]), "r"(s[13]), "r"(s[14]), "r"(s[15])
      : "memory");
}

__device__ __forceinline__ float fast_exp2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
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
__device__ __forceinline__ uint64_t pack2(float lo, float hi) {
  float2 f = make_float2(lo, hi);
  return *reinterpret_cast<uint64_t*>(&f);
}

// ==== ANCHOR:MMA_GROUPS (identical descriptor walks to the g16 kernel) ====
constexpr uint32_t kIdescQK = 136316048u;
constexpr uint32_t kIdescPV = 136381584u;
constexpr uint32_t kDescHi = 0x40004040u;

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

// ==== ANCHOR:ROUTING ====
// Acceptance + dedup rule shared by the count and scatter passes, folded into
// g4_route_row (round 27): a listed block is usable iff non-negative and, when
// causal, its first token is visible; an additional kv_len filter skips blocks
// that would only contribute masked columns. Duplicates of an earlier ACCEPTED
// entry of the same (h, n) are dropped so the merge sees each block's
// contribution exactly once.
__device__ __forceinline__ int g4_batch_of(const G4Params& p, long n) {
  int lo = 0, hi = p.nbatch - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (__ldg(p.cu_q + mid) <= n)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// Per-(h, n) routing: accepted-block ranks + blocks kept in registers so both
// the legacy kernels and the fused route kernel share one implementation.
struct G4RowRoute {
  int h;
  long n;
  int bat;
  int ranks[kMaxTopk];
  int blks[kMaxTopk];
  int cnt;
};

__device__ __forceinline__ G4RowRoute g4_route_row(const G4Params& p, long hn) {
  G4RowRoute r;
  r.h = (int)(hn / p.total_q);
  r.n = hn - (long)r.h * p.total_q;
  r.bat = g4_batch_of(p, r.n);
  const int kv_len = __ldg(p.cu_k + r.bat + 1) - __ldg(p.cu_k + r.bat);
  const int qpos = kv_len - p.seqlen_q + (int)(r.n - __ldg(p.cu_q + r.bat));
  // Round 27: the whole q2k row is fetched into registers with independent
  // predicated loads and dedup runs in registers. The old per-slot
  // g4_accept_rank re-issued the same ldgs inside nested loops, giving each
  // route thread a ~10-step serial gmem chain (route kernel measured 2% SM /
  // 0.2% DRAM — pure latency).
  const int* qrow = p.q2k + hn * p.topk;
#pragma unroll
  for (int s = 0; s < kMaxTopk; s++) {
    r.blks[s] = (s < p.topk) ? __ldg(qrow + s) : -1;
  }
  bool av[kMaxTopk];
#pragma unroll
  for (int s = 0; s < kMaxTopk; s++) {
    const int blk = r.blks[s];
    const bool val =
        (blk >= 0) && ((long)blk * 128 < kv_len) && (!p.causal || (long)blk * 128 <= qpos);
    bool dup = false;
    int rank = 0;
    if (val) {
#pragma unroll
      for (int t = 0; t < kMaxTopk; t++) {
        if (t < s && av[t]) {
          if (r.blks[t] == blk)
            dup = true;
          else
            rank++;
        }
      }
    }
    av[s] = val && !dup;
    r.ranks[s] = av[s] ? rank : -1;
  }
  r.cnt = 0;
#pragma unroll
  for (int s = 0; s < kMaxTopk; s++) {
    if (r.ranks[s] >= 0) r.cnt++;
  }
  return r;
}

__device__ __forceinline__ int g4_bucket_of(const G4Params& p, const G4RowRoute& r, int s) {
  return (r.h * p.nbatch + r.bat) * p.max_pages + r.blks[s];
}

// ==== ANCHOR:INLINE_MERGE_HELPERS ====
// Round 27: the ~50 us serial g4_merge kernel (L2 read-back of every partial,
// launched after ALL tiles drain) is folded into the main kernel. Each drained
// output row bumps a device-scope release counter; the tile that observes the
// last ticket merges that row's online-softmax partials in-place after its
// dealloc arrive, so merges overlap the remaining tiles' MMA chains.
__device__ __forceinline__ int g4_ticket_add(int* ctr) {
  int old;
  asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(ctr), "r"(1)
               : "memory");
  return old;
}

__device__ __forceinline__ void g4_acquire_fence() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}

// Split-KV merge of up to 4 finished output rows: out = sum_s w_s acc_s /
// sum_s w_s l_s, w_s = exp2((m_s - M) * scale). Last-ticket lanes pool their
// rows into groups of 4 with 8 lanes per row (16 elements each), so the
// dependent-gmem-latency chains of 4 rows overlap inside one warp — the
// latency chain is what a per-lane serial merge measured (~+16 us/tile at
// 1 CTA/SM since CTA exit gates the next tile's start). topk <= kMaxTopk.
template <typename QT, bool IS_BF16>
__device__ __forceinline__ void g4_merge_rows_warp(const G4Params& p, const long* rows,
                                                   const int* cnts, int ng) {
  const int lane = threadIdx.x & 31;
  const int grp = lane >> 3;  // row slot 0..3
  const int sub = lane & 7;   // owns elements [sub*16, sub*16+16)
  if (grp >= ng) return;
  const long sbase = rows[grp] * p.topk;
  const int cnt = cnts[grp];
  // All m/l loads are independent: one latency round instead of two chains.
  float2 mlv[8];
  const float* ml = p.ws_ml + sbase * 2;
#pragma unroll
  for (int s = 0; s < 8; s++) {
    if (s < cnt) mlv[s] = *reinterpret_cast<const float2*>(ml + s * 2);
  }
  float mmax = -CUDART_INF_F;
#pragma unroll
  for (int s = 0; s < 8; s++)
    if (s < cnt) mmax = fmaxf(mmax, mlv[s].x);
  float w[8];
  float l_tot = 0.f;
#pragma unroll
  for (int s = 0; s < 8; s++) {
    if (s < cnt) {
      w[s] = fast_exp2((mlv[s].x - mmax) * p.scale_log2e);
      l_tot += w[s] * mlv[s].y;
    }
  }
  const float inv = (l_tot > 0.f) ? (1.f / l_tot) : 0.f;
  const __half* ab = reinterpret_cast<const __half*>(p.ws_acc) + sbase * 128 + sub * 16;
  float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
  for (int s = 0; s < 8; s++) {
    if (s < cnt) {
      const __half* sp = ab + s * 128;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint2 h = *reinterpret_cast<const uint2*>(sp + j * 4);
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
        float4* a = (j == 0) ? &a0 : ((j == 1) ? &a1 : ((j == 2) ? &a2 : &a3));
        a->x += w[s] * f0.x;
        a->y += w[s] * f0.y;
        a->z += w[s] * f1.x;
        a->w += w[s] * f1.y;
      }
    }
  }
  uint4 o0, o1;
  o0.x = IS_BF16 ? pack_bf16x2(a0.x * inv, a0.y * inv) : pack_f16x2(a0.x * inv, a0.y * inv);
  o0.y = IS_BF16 ? pack_bf16x2(a0.z * inv, a0.w * inv) : pack_f16x2(a0.z * inv, a0.w * inv);
  o0.z = IS_BF16 ? pack_bf16x2(a1.x * inv, a1.y * inv) : pack_f16x2(a1.x * inv, a1.y * inv);
  o0.w = IS_BF16 ? pack_bf16x2(a1.z * inv, a1.w * inv) : pack_f16x2(a1.z * inv, a1.w * inv);
  o1.x = IS_BF16 ? pack_bf16x2(a2.x * inv, a2.y * inv) : pack_f16x2(a2.x * inv, a2.y * inv);
  o1.y = IS_BF16 ? pack_bf16x2(a2.z * inv, a2.w * inv) : pack_f16x2(a2.z * inv, a2.w * inv);
  o1.z = IS_BF16 ? pack_bf16x2(a3.x * inv, a3.y * inv) : pack_f16x2(a3.x * inv, a3.y * inv);
  o1.w = IS_BF16 ? pack_bf16x2(a3.z * inv, a3.w * inv) : pack_f16x2(a3.z * inv, a3.w * inv);
  QT* orow = reinterpret_cast<QT*>(p.out) + rows[grp] * 128 + sub * 16;
  *reinterpret_cast<uint4*>(orow) = o0;
  *reinterpret_cast<uint4*>(orow + 8) = o1;
}

// Rows whose query selected no usable block get exact zeros (the reference
// zeroes the probabilities). The inline-merge path has no tail merge kernel,
// so the route stage zeroes these rows directly: the group_size heads of one
// (kv-head, query) pair are contiguous rows, 256 B each.
__device__ __forceinline__ void g4_zero_row_group(const G4Params& p, long n, int h) {
  uint4* orow = reinterpret_cast<uint4*>(reinterpret_cast<char*>(p.out) +
                                         ((long)n * p.num_q_heads + h * kGroup) * 256);
  const uint4 z = {0u, 0u, 0u, 0u};
#pragma unroll 4
  for (int j = 0; j < 64; j++) orow[j] = z;
}

// R1: per (h, n) — count accepted blocks per bucket, record per-slot ranks and
// the per-query accepted count.
__global__ void __launch_bounds__(256) g4_route_count(const __grid_constant__ G4Params p) {
  long hn = (long)blockIdx.x * 256 + threadIdx.x;
  long total = (long)p.num_kv_heads * p.total_q;
  if (hn >= total) return;
  const G4RowRoute r = g4_route_row(p, hn);
#pragma unroll
  for (int s = 0; s < kMaxTopk; s++) {
    if (s >= p.topk) break;
    p.rankmap[hn * p.topk + s] = r.ranks[s];
    if (r.ranks[s] >= 0) atomicAdd(&p.bcnt[g4_bucket_of(p, r, s)], 1);
  }
  p.cnt_hn[hn] = r.cnt;
  if (p.inline_merge && r.cnt == 0) g4_zero_row_group(p, r.n, r.h);
}

// R1+R2+R3 fused (round 26): one cooperative launch replaces the zero-fill +
// 3-kernel route stage. Ranks stay in registers (no rankmap traffic at all),
// every CTA redundantly builds the bucket scan in smem so the scatter phase
// needs no third grid round-trip, and bcnt/bcnt2 are zeroed in-kernel (no fill
// kernel). Engaged only for fully co-resident grids (host guard); the legacy
// 3-kernel chain below remains the general fallback.
// Round 35: one-shot grid barriers replace
// cooperative_groups::grid_group::sync(). The CG barrier's backoff polling
// measured ~7.4 us release latency per sync on a 96-CTA grid (46% of
// route_fused warp cycles were barrier stalls); a one-thread-per-CTA
// release-add + acquire-spin releases in ~1-2 us. The counter/flag pair lives
// in the per-call workspace (host-zeroed, 16 B) so concurrent calls on
// different streams have independent state and the kernel needs no
// generation bookkeeping.
__device__ __forceinline__ int ld_acq_gpu(const int* addr) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(addr) : "memory");
  return v;
}

__device__ __forceinline__ void route_grid_sync(int* cnt, int* flag, int expected) {
  __syncthreads();
  if (threadIdx.x == 0) {
    int old;
    asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(cnt), "r"(1)
                 : "memory");
    if (old == expected - 1) {
      asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(flag), "r"(1) : "memory");
    } else {
      while (ld_acq_gpu(flag) == 0) {
      }
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(256) g4_route_fused(const __grid_constant__ G4Params p) {
  __shared__ int s_boff[6144];
  __shared__ int s_pad[256], s_tile[256];
  const int tid = threadIdx.x;
  const int nb = p.nbuckets;
  int* const bar_cnt = p.route_bar;
  int* const bar_flag = p.route_bar + 1;
  const int nctas = (int)gridDim.x;
  // phase 0: zero the two histogram buffers (bcnt, bcnt2 are contiguous) and,
  // for the experimental inline-merge path, the per-row completion counters.
  // (Round 27: a host-side cudaMemsetAsync for bcnt/bcnt2 measured NET
  // NEUTRAL-worse — the memset launch costs more than zeroing 3 KB here.)
  for (int i = blockIdx.x * 256 + tid; i < 2 * nb; i += nctas * 256) p.bcnt[i] = 0;
  if (p.inline_merge) {
    const long rows = (long)p.total_q * p.num_q_heads;
    for (long i = blockIdx.x * 256 + tid; i < rows; i += (long)nctas * 256) p.rowcnt[i] = 0;
  }
  route_grid_sync(bar_cnt, bar_flag, nctas);
  // phase 1: count, ranks in registers.
  const long hn = (long)blockIdx.x * 256 + tid;
  const long total = (long)p.num_kv_heads * p.total_q;
  G4RowRoute r;
  const bool live = hn < total;
  // Dead lanes (last partial warp) stay converged with ranks=-1 so the
  // full-warp __match_any_sync histogram below is well defined.
#pragma unroll
  for (int s = 0; s < kMaxTopk; s++) r.ranks[s] = -1;
  if (live) {
    r = g4_route_row(p, hn);
    p.cnt_hn[hn] = r.cnt;
    if (p.inline_merge && r.cnt == 0) g4_zero_row_group(p, r.n, r.h);
  }
  {
    const int rl = tid & 31;
#pragma unroll
    for (int s = 0; s < kMaxTopk; s++) {
      if (s >= p.topk) break;
      // Warp-aggregated histogramming: lanes selecting the same bucket combine
      // into one RED, cutting the outstanding-atomic drain at the next grid
      // barrier (phase-2 release waits on the L2 atomic queue).
      const int bucket = (r.ranks[s] >= 0) ? g4_bucket_of(p, r, s) : -1;
      const unsigned grp = __match_any_sync(0xffffffffu, bucket);
      if (bucket >= 0 && rl == __ffs(grp) - 1) atomicAdd(&p.bcnt[bucket], __popc(grp));
    }
  }
  route_grid_sync(bar_cnt + 2, bar_flag + 2, nctas);
  // phase 2: redundant per-CTA scan into s_boff; CTA 0 also publishes the
  // global route tables consumed by the UMMA kernel.
  const int S = (nb + 255) >> 8;
  const int i0 = tid * S;
  int pad_sum = 0, tile_sum = 0;
  for (int i = 0; i < S && i0 + i < nb; i++) {
    int c = p.bcnt[i0 + i];
    int tiles = (c + kQTile - 1) / kQTile;
    pad_sum += tiles * kQTile;
    tile_sum += tiles;
  }
  // Round 35: warp-shuffle scan replaces the 16-__syncthreads Hillis-Steele
  // smem scan. route_fused is barrier-stall dominated (~46% of warp cycles at
  // CTA barriers with only ~2 warps/scheduler resident); the shuffle form
  // drops phase-2 to 2 barriers with identical exclusive-prefix semantics.
  const int lane = tid & 31;
  const int wp = tid >> 5;
  int sq = pad_sum, stv = tile_sum;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int q2 = __shfl_up_sync(0xffffffffu, sq, d);
    const int t2 = __shfl_up_sync(0xffffffffu, stv, d);
    if (lane >= d) {
      sq += q2;
      stv += t2;
    }
  }
  if (lane == 31) {
    s_pad[wp] = sq;
    s_tile[wp] = stv;
  }
  __syncthreads();
  if (wp == 0) {
    int q = (lane < 8) ? s_pad[lane] : 0;
    int t = (lane < 8) ? s_tile[lane] : 0;
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const int q2 = __shfl_up_sync(0xffffffffu, q, d);
      const int t2 = __shfl_up_sync(0xffffffffu, t, d);
      if (lane >= d && lane < 8) {
        q += q2;
        t += t2;
      }
    }
    if (lane < 8) {
      s_pad[lane] = q;
      s_tile[lane] = t;
    }
  }
  __syncthreads();
  int run_q = sq + ((wp > 0) ? s_pad[wp - 1] : 0) - pad_sum;
  int run_t = stv + ((wp > 0) ? s_tile[wp - 1] : 0) - tile_sum;
  const bool pub = (blockIdx.x == 0);
  for (int i = 0; i < S && i0 + i < nb; i++) {
    const int bkt = i0 + i;
    const int c = p.bcnt[bkt];
    const int tiles = (c + kQTile - 1) / kQTile;
    s_boff[bkt] = run_q;
    if (pub) {
      p.boff[bkt] = run_q;
      p.btoff[bkt] = run_t;
      int* meta = p.bmeta + bkt * 5;
      meta[0] = run_q;
      meta[1] = c;
      meta[2] = bkt / (p.nbatch * p.max_pages);
      const int rem = bkt % (p.nbatch * p.max_pages);
      const int bat = rem / p.max_pages;
      const int blk = rem - bat * p.max_pages;
      meta[3] = bat * 65536 + blk;
      meta[4] = __ldg(p.page_table + (long)bat * p.pt_stride + blk);
      for (int t = 0; t < tiles; t++) p.chunk_bkt[run_t + t] = bkt;
    }
    run_q += tiles * kQTile;
    run_t += tiles;
  }
  if (pub && tid == 255) {
    p.boff[nb] = run_q;
    p.btoff[nb] = run_t;
    p.tile_total[0] = run_t;
  }
  __syncthreads();  // s_boff visible CTA-wide before scatter
  // phase 3: warp-aggregated scatter — one atomic per distinct bucket per
  // warp; group lanes carve consecutive positions from the leader's base.
  {
    const int rl = tid & 31;
#pragma unroll
    for (int s = 0; s < kMaxTopk; s++) {
      if (s >= p.topk) break;
      const int bucket = (r.ranks[s] >= 0) ? g4_bucket_of(p, r, s) : -1;
      const unsigned grp = __match_any_sync(0xffffffffu, bucket);
      if (bucket >= 0) {
        const int leader = __ffs(grp) - 1;
        int base = -1;
        if (rl == leader) base = atomicAdd(&p.bcnt2[bucket], __popc(grp));
        base = __shfl_sync(0xffffffffu, base, leader);
        const int off = __popc(grp & ((1u << rl) - 1u));
        p.rowlist[s_boff[bucket] + base + off] = (int)r.n | (r.ranks[s] << 28);
      }
    }
  }
}

// R2: single-CTA scan — padded row offsets (multiples of kQTile queries),
// tile offsets, per-bucket meta, chunk->bucket LUT, total tile count.
__global__ void __launch_bounds__(1024) g4_route_scan(const __grid_constant__ G4Params p) {
  __shared__ int s_pad[1024];
  __shared__ int s_tile[1024];
  const int nb = p.nbuckets;
  const int tid = threadIdx.x;
  const int S = (nb + 1023) >> 10;  // buckets per thread
  const int i0 = tid * S;
  int pad_sum = 0, tile_sum = 0;
  for (int i = 0; i < S && i0 + i < nb; i++) {
    int c = p.bcnt[i0 + i];
    int tiles = (c + kQTile - 1) / kQTile;
    pad_sum += tiles * kQTile;
    tile_sum += tiles;
  }
  s_pad[tid] = pad_sum;
  s_tile[tid] = tile_sum;
  __syncthreads();
  // Kogge-Stone block scan (exclusive)
  for (int d = 1; d < 1024; d <<= 1) {
    int ap = (tid >= d) ? s_pad[tid - d] : 0;
    int at = (tid >= d) ? s_tile[tid - d] : 0;
    __syncthreads();
    s_pad[tid] += ap;
    s_tile[tid] += at;
    __syncthreads();
  }
  int run_q = (tid > 0) ? s_pad[tid - 1] : 0;
  int run_t = (tid > 0) ? s_tile[tid - 1] : 0;
  for (int i = 0; i < S && i0 + i < nb; i++) {
    const int bkt = i0 + i;
    const int c = p.bcnt[bkt];
    const int tiles = (c + kQTile - 1) / kQTile;
    p.boff[bkt] = run_q;
    p.btoff[bkt] = run_t;
    int* meta = p.bmeta + bkt * 5;
    meta[0] = run_q;                           // rowlist start (queries)
    meta[1] = c;                               // query count
    meta[2] = bkt / (p.nbatch * p.max_pages);  // kv head
    const int rem = bkt % (p.nbatch * p.max_pages);
    const int bat = rem / p.max_pages;
    const int blk = rem - bat * p.max_pages;
    meta[3] = bat * 65536 + blk;                                    // packed batch/block
    meta[4] = __ldg(p.page_table + (long)bat * p.pt_stride + blk);  // physical page
    for (int t = 0; t < tiles; t++) p.chunk_bkt[run_t + t] = bkt;
    run_q += tiles * kQTile;
    run_t += tiles;
  }
  if (tid == 1023) {
    p.boff[nb] = run_q;
    p.btoff[nb] = run_t;
    p.tile_total[0] = run_t;
  }
}

// R3: scatter accepted (h, n) entries into their bucket's rowlist region.
__global__ void __launch_bounds__(256) g4_route_scatter(const __grid_constant__ G4Params p) {
  long hn = (long)blockIdx.x * 256 + threadIdx.x;
  long total = (long)p.num_kv_heads * p.total_q;
  if (hn >= total) return;
  const int h = (int)(hn / p.total_q);
  const long n = hn - (long)h * p.total_q;
  const int bat = g4_batch_of(p, n);
  const int kv_len = __ldg(p.cu_k + bat + 1) - __ldg(p.cu_k + bat);
  const int qpos = kv_len - p.seqlen_q + (int)(n - __ldg(p.cu_q + bat));
  for (int s = 0; s < p.topk; s++) {
    int rank = __ldg(p.rankmap + hn * p.topk + s);
    if (rank < 0) continue;
    int blk = __ldg(p.q2k + hn * p.topk + s);
    int bucket = (h * p.nbatch + bat) * p.max_pages + blk;
    int pos = atomicAdd(&p.bcnt2[bucket], 1);
    p.rowlist[p.boff[bucket] + pos] = (int)n | (rank << 28);
  }
}

// ==== ANCHOR:MAIN_KERNEL ====
// Per-CTA tile state resolved once per role (redundant scalar ldgs, cached).
struct TileInfo {
  int bkt, tile_in_bkt, row_q0, row_count, kvh, batch, blk, page, kv_len;
};

__device__ __forceinline__ TileInfo load_tile(const G4Params& p, int chunk) {
  TileInfo t;
  t.bkt = __ldg(p.chunk_bkt + chunk);
  const int* meta = p.bmeta + t.bkt * 5;
  t.row_q0 = __ldg(meta + 0);
  const int cnt_total = __ldg(meta + 1);
  t.kvh = __ldg(meta + 2);
  const int bat_blk = __ldg(meta + 3);
  t.batch = bat_blk >> 16;
  t.blk = bat_blk & 0xFFFF;
  t.page = __ldg(meta + 4);
  const int t0 = __ldg(p.btoff + t.bkt);
  t.tile_in_bkt = chunk - t0;
  t.row_q0 += t.tile_in_bkt * kQTile;
  // queries valid in THIS tile (meta count is bucket-global; clamp to tile)
  t.row_count = min(kQTile, cnt_total - t.tile_in_bkt * kQTile);
  t.kv_len = __ldg(p.cu_k + t.batch + 1) - __ldg(p.cu_k + t.batch);
  return t;
}

template <typename QT, bool IS_BF16>
__global__ void __launch_bounds__(512, 1)
    g4_umma_kernel(const __grid_constant__ G4Params p, const __grid_constant__ CUtensorMap k_map,
                   const __grid_constant__ CUtensorMap v_map) {
  const CUtensorMap* ktm = &k_map;
  const CUtensorMap* vtm = &v_map;
  const int tid = threadIdx.x;
  const int warp = warp_uni(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  const int smem = smem_u32(smem_raw);

  // Round 33: pair two 32-query chunks per CTA (stages 0/1 now carry REAL,
  // independent tiles with their own K/V slabs instead of stage 1 running a
  // fully-masked padding pass over the same bucket). An odd tail chunk pairs
  // with a "fake" tile reusing chunk 0's bucket with row_count 0 — identical
  // semantics to the old padding stage, so every barrier arrival stays
  // uniform across all roles.
  const int c0 = (int)blockIdx.x * 2;
  const int total_tiles = __ldg(p.tile_total);
  if (c0 >= total_tiles) return;
  const int c1 = c0 + 1;
  const bool has1 = c1 < total_tiles;
  const int c1v = has1 ? c1 : c0;

  __syncthreads();

  const int b_qfull = smem + kB_qfull;
  const int b_kvfull = smem + kB_kvfull;
  const int b_sfull = smem + kB_sfull;
  const int b_pfull = smem + kB_pfull;
  const int b_ptail = smem + kB_ptail;
  const int b_kv2full = smem + kB_kv2full;
  const int b_ofull = smem + kB_ofull;
  const int b_dealloc = smem + kB_dealloc;

  if (warp == 0 && elect_one()) {
    mbar_init(b_qfull + 0, 1);
    mbar_init(b_qfull + 8, 1);
    mbar_init(b_kvfull + 0, 1);
    mbar_init(b_kvfull + 8, 1);
    mbar_init(b_sfull + 0, 1);
    mbar_init(b_sfull + 8, 1);
    mbar_init(b_pfull + 0, 128);
    mbar_init(b_pfull + 8, 128);
    mbar_init(b_ptail + 0, 128);
    mbar_init(b_ptail + 8, 128);
    mbar_init(b_ofull + 0, 1);
    mbar_init(b_ofull + 8, 1);
    mbar_init(b_dealloc, 128);
    mbar_init(b_kv2full + 0, 1);
    mbar_init(b_kv2full + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  volatile int* tmem_hold = (volatile int*)(smem_raw + kTmemHold);
  if (warp == 0) tc_alloc(smem + kTmemHold, 512);
  __syncthreads();
  tc_fence_after_sync();
  const int taddr = tmem_hold[0];

  int* sm_rown = reinterpret_cast<int*>(smem_raw + kRowsOff);
  int* sm_rowrank = sm_rown + 64;
  // rowvcol[64] was never consumed by any role; it now stages per-stage
  // {row_count, kvh} pairs for the chunk-paired epilogue/gather readers.
  int* sm_stile = sm_rowrank + 64;
  float* sm_mlm = reinterpret_cast<float*>(smem_raw + kMlOff);
  float* sm_mll = sm_mlm + 256;

  if (warp >= 12) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
  }

  const int trow_base = (warp % 4) * 32 << 16;
  const int my_row = (warp % 4) * 32 + lane;
  const uint32_t idesc_qk = kIdescQK - (IS_BF16 ? 0u : 0x480u);
  const uint32_t idesc_pv = kIdescPV - (IS_BF16 ? 0u : 0x480u);

  // ================= Role: softmax (warps 0-7) =================
  if (warp <= 7) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    const int stage = warp_uni(warp / 4);
    TileInfo ti = load_tile(p, stage ? c1v : c0);
    if (stage && !has1) ti.row_count = 0;  // fake tail tile: fully masked
    const int q_i = (stage * 128 + my_row) >> 2;
    // sm_rown is published by warp 13 before q_full; the s_full wait below
    // (MMA <- q_full <- row-meta) orders these reads.
    mbar_wait(b_sfull + stage * 8, 0);
    // q_i is stage-global ([32,64) in stage 1); compare the chunk-local row.
    const int n_row = (q_i - (stage << 5) < ti.row_count) ? sm_rown[q_i] : -1;
    int vcol = 0;
    if (n_row >= 0) {
      const int qpos = ti.kv_len - p.seqlen_q + (n_row - __ldg(p.cu_q + ti.batch));
      vcol = ti.kv_len - ti.blk * 128;
      if (vcol > 128) vcol = 128;
      if (p.causal) {
        int cc = qpos - ti.blk * 128 + 1;
        if (vcol > cc) vcol = cc;
      }
      if (vcol < 0) vcol = 0;
    }
    const int body_v = min(vcol, 64);
    const int tail_v = max(vcol - 64, 0);
    const float scale = p.scale_log2e;
    const int sc_base = taddr + stage * 128 + trow_base;
    const int p_base = taddr + stage * 128 + 64 + trow_base;

    float s0[64], s1[64];
    {
      // tcgen05.ld is .sync.aligned: EVERY lane must execute it. Guarding the
      // loads on per-lane body_v/tail_v deadlocks partial tiles (lanes park
      // at different WARPSYNC sites), so load unconditionally and mask in
      // registers afterwards.
      tmem_ld64(s0, sc_base);
      tmem_ld64(s1, sc_base + 64);
#pragma unroll
      for (int j = 0; j < 64; j++) {
        if (j >= body_v) s0[j] = -CUDART_INF_F;
        if (j >= tail_v) s1[j] = -CUDART_INF_F;
      }
    }
    float tmax0 = -CUDART_INF_F, tmax1 = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      tmax0 =
          fmaxf(tmax0, fmaxf(fmaxf(s0[4 * j], s0[4 * j + 1]), fmaxf(s0[4 * j + 2], s0[4 * j + 3])));
      tmax1 =
          fmaxf(tmax1, fmaxf(fmaxf(s1[4 * j], s1[4 * j + 1]), fmaxf(s1[4 * j + 2], s1[4 * j + 3])));
    }
    const float row_max = fmaxf(tmax0, tmax1);  // raw score max (>= -inf rows only)
    const uint64_t scale2 = pack2(scale, scale);
    const uint64_t bias2 = pack2(-row_max * scale, -row_max * scale);
    uint64_t sum2 = pack2(0.f, 0.f);
    uint32_t pk[32];
    const uint64_t* s0w = reinterpret_cast<const uint64_t*>(s0);
    const uint64_t* s1w = reinterpret_cast<const uint64_t*>(s1);
#pragma unroll
    for (int j = 0; j < 32; j++) {
      uint64_t x = fma2(s0w[j], scale2, bias2);
      float2 e = *(float2*)&x;
      e.x = (s0[2 * j] <= -1e37f) ? 0.f : fast_exp2(e.x);
      e.y = (s0[2 * j + 1] <= -1e37f) ? 0.f : fast_exp2(e.y);
      sum2 = add2(sum2, *(uint64_t*)&e);
      pk[j] = IS_BF16 ? pack_bf16x2(e.x, e.y) : pack_f16x2(e.x, e.y);
    }
    tmem_st32(p_base, pk);
#pragma unroll
    for (int j = 0; j < 16; j++) {
      uint64_t x = fma2(s1w[j], scale2, bias2);
      float2 e = *(float2*)&x;
      e.x = (s1[2 * j] <= -1e37f) ? 0.f : fast_exp2(e.x);
      e.y = (s1[2 * j + 1] <= -1e37f) ? 0.f : fast_exp2(e.y);
      sum2 = add2(sum2, *(uint64_t*)&e);
      pk[j] = IS_BF16 ? pack_bf16x2(e.x, e.y) : pack_f16x2(e.x, e.y);
    }
    tmem_st16u(p_base + 32, pk);
    tc_wait_st();
    mbar_arrive(b_pfull + stage * 8);
#pragma unroll
    for (int j = 0; j < 16; j++) {
      uint64_t x = fma2(s1w[16 + j], scale2, bias2);
      float2 e = *(float2*)&x;
      e.x = (s1[32 + 2 * j] <= -1e37f) ? 0.f : fast_exp2(e.x);
      e.y = (s1[33 + 2 * j] <= -1e37f) ? 0.f : fast_exp2(e.y);
      sum2 = add2(sum2, *(uint64_t*)&e);
      pk[j] = IS_BF16 ? pack_bf16x2(e.x, e.y) : pack_f16x2(e.x, e.y);
    }
    tmem_st16u(p_base + 48, pk);
    const float2 sf = *(const float2*)&sum2;
    // Publish m/l BEFORE the tail arrive so the release chain (ptail -> PV ->
    // ofull commit) makes them visible to the epilogue warps that read them.
    if (q_i - (stage << 5) < ti.row_count) {
      sm_mlm[stage * 128 + my_row] = row_max;
      sm_mll[stage * 128 + my_row] = sf.x + sf.y;
    }
    tc_wait_st();
    mbar_arrive(b_ptail + stage * 8);
    return;
  }

  // ================= Role: epilogue (warps 8-11) =================
  if (warp >= 8 && warp <= 11) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
    // Per-stage {rc, kvh} are read from smem (published by warp 13 before
    // q_full; the ofull wait below chains after it) so this role stays free
    // of global route-table walk under the 80-reg cap.
    long row_of_stage[2];
    int cnt_of_stage[2], tick_of_stage[2];
#pragma unroll 1
    for (int stage = 0; stage < 2; stage++) {
      mbar_wait(b_ofull + stage * 8, 0);
      const int q_i = (stage * 128 + my_row) >> 2;
      const int gg = my_row & 3;
      // tcgen05.ld stays warp-uniform (same deadlock class as the softmax S
      // loads); only the workspace writes are per-lane guarded.
      const bool lane_valid = (q_i - (stage << 5) < sm_stile[stage * 2]);
      const int ent = lane_valid ? sm_rown[q_i] : 0;
      const int rank = lane_valid ? sm_rowrank[q_i] : 0;
      const long out_row = (long)ent * p.num_q_heads + sm_stile[stage * 2 + 1] * kGroup + gg;
      const long slot = out_row * p.topk + rank;
      if (lane_valid) {
        p.ws_ml[slot * 2 + 0] = sm_mlm[stage * 128 + my_row];
        p.ws_ml[slot * 2 + 1] = sm_mll[stage * 128 + my_row];
      }
      __half* dst = reinterpret_cast<__half*>(p.ws_acc) + slot * 128;
      // Pipelined drain (round 25, mirrors the g16 epilogue v2): 3 rotating
      // tcgen05.ld buffers so chunk c's workspace stores overlap chunk
      // c+1/c+2's TMEM latency. Loads stay warp-uniform; stores lane-guarded.
      // Round 27: partials are stored as fp16 pairs (half the previous fp32
      // traffic: ~200 MB -> ~100 MB write + ~100 MB read on case 9). fp16's
      // 11-bit mantissa sits far below the bf16 output quantum and the merge
      // accumulates in fp32.
      const int obase = taddr + 256 + stage * 128 + trow_base;
      float o0[16], o1[16], o2[16];
      tmem_ld16(o0, obase);
      tmem_ld16(o1, obase + 16);
      tmem_ld16(o2, obase + 32);
#pragma unroll
      for (int c = 0; c < 8; c++) {
        float* ocur = (c % 3 == 0) ? o0 : ((c % 3 == 1) ? o1 : o2);
        if (lane_valid) {
#pragma unroll
          for (int j = 0; j < 2; j++) {
            uint4 pk;
            pk.x = pack_f16x2(ocur[j * 8 + 0], ocur[j * 8 + 1]);
            pk.y = pack_f16x2(ocur[j * 8 + 2], ocur[j * 8 + 3]);
            pk.z = pack_f16x2(ocur[j * 8 + 4], ocur[j * 8 + 5]);
            pk.w = pack_f16x2(ocur[j * 8 + 6], ocur[j * 8 + 7]);
            *reinterpret_cast<uint4*>(dst + c * 16 + j * 8) = pk;
          }
        }
        if (c + 3 < 8) tmem_ld16(ocur, obase + (c + 3) * 16);
      }
      // Release-publish this row's partial, then take a completion ticket.
      row_of_stage[stage] = out_row;
      if (lane_valid && p.inline_merge) {
        const int cnt = __ldg(p.cnt_hn + (long)sm_stile[stage * 2 + 1] * p.total_q + ent);
        cnt_of_stage[stage] = cnt;
        tick_of_stage[stage] = g4_ticket_add(p.rowcnt + out_row);
      } else {
        cnt_of_stage[stage] = 1;
        tick_of_stage[stage] = -1;
      }
    }
    // Dealloc BEFORE merging: the merge reads only global partials, so TMEM is
    // freed immediately and the next resident tile is never delayed behind a
    // last-ticket merge's gmem round-trips.
    mbar_arrive(b_dealloc);
    if (p.inline_merge) {
      g4_acquire_fence();  // earlier tickets' partials are visible now
#pragma unroll 1
      for (int stage = 0; stage < 2; stage++) {
        unsigned mask = __ballot_sync(0xFFFFFFFFu, tick_of_stage[stage] == cnt_of_stage[stage] - 1);
        while (mask) {
          long rs[4];
          int cs[4], ng = 0;
          while (mask && ng < 4) {
            const int src = __ffs(mask) - 1;
            mask &= mask - 1;
            rs[ng] = (long)__shfl_sync(0xFFFFFFFFu, (long long)row_of_stage[stage], src);
            cs[ng] = __shfl_sync(0xFFFFFFFFu, cnt_of_stage[stage], src);
            ng++;
          }
          g4_merge_rows_warp<QT, IS_BF16>(p, rs, cs, ng);
        }
      }
    }
    return;
  }

  // ================= Roles: producer + gather (warps 12-15) =================
  // Round 26: the MMA-leader warp now joins the Q gather before its MMA
  // duties (it only waited on qfull during this phase anyway) — 128 gather
  // threads instead of 96 shortens the front-end critical path.
  if (warp >= 12) {
    const TileInfo ti0 = load_tile(p, c0);
    TileInfo ti1 = load_tile(p, c1v);
    if (!has1) ti1.row_count = 0;
    // Only these four scalars per chunk stay live past this point (keeps the
    // 48-reg producer cap safe now that two TileInfos are materialized).
    const int t_rc[2] = {ti0.row_count, ti1.row_count};
    const int t_rq0[2] = {ti0.row_q0, ti1.row_q0};
    const int t_kvh[2] = {ti0.kvh, ti1.kvh};
    const int t_pg[2] = {ti0.page, ti1.page};
    const int t_bkt0 = ti0.bkt, t_bkt1 = ti1.bkt;
    // Same-bucket pairs (the common case: routing emits a bucket's tiles
    // consecutively) AND the fake tail tile stream identical K/V — skip the
    // duplicate 64 KB stage-1 TMA and let the MMA reuse the stage-0 slabs.
    const bool share_kv = (t_bkt0 == t_bkt1);
    // Round 27: K/V TMA no longer waits for the rown staging + 128-thread
    // barrier — the KV slabs never depend on row metadata, so issuing the 8
    // loads immediately after load_tile starts ~2 us of transfer earlier on
    // every tile's serial chain.
    if (warp == 14 && elect_one()) {
      // K + V of each chunk's physical page: 4D map
      // {64 el, 128 tok, 2 elhalf, pages*Hkv}, box {64, 64, 1, 1}.
      const int ph0 = t_pg[0] * p.num_kv_heads + t_kvh[0];
      mbar_arrive_tx(b_kvfull + 0, 32768);
      tma_load_4d(smem + kKOff + 0, ktm, 0, 0, 0, ph0, b_kvfull + 0);
      tma_load_4d(smem + kKOff + 8192, ktm, 0, 64, 0, ph0, b_kvfull + 0);
      tma_load_4d(smem + kKOff + 16384, ktm, 0, 0, 1, ph0, b_kvfull + 0);
      tma_load_4d(smem + kKOff + 24576, ktm, 0, 64, 1, ph0, b_kvfull + 0);
      mbar_arrive_tx(b_kvfull + 8, 32768);
      tma_load_4d(smem + kVOff + 0, vtm, 0, 0, 0, ph0, b_kvfull + 8);
      tma_load_4d(smem + kVOff + 8192, vtm, 0, 64, 0, ph0, b_kvfull + 8);
      tma_load_4d(smem + kVOff + 16384, vtm, 0, 0, 1, ph0, b_kvfull + 8);
      tma_load_4d(smem + kVOff + 24576, vtm, 0, 64, 1, ph0, b_kvfull + 8);
      // Round 33: stage-1 chunk gets its own K/V slabs so both pipeline
      // stages carry real, independent tiles — skipped when the paired
      // chunks share a bucket (identical K/V page). Skip via if/else, never
      // early-return: the elected lane must still reach barrier 9 below.
      if (!share_kv) {
        const int ph1 = t_pg[1] * p.num_kv_heads + t_kvh[1];
        mbar_arrive_tx(b_kv2full + 0, 32768);
        tma_load_4d(smem + kK1Off + 0, ktm, 0, 0, 0, ph1, b_kv2full + 0);
        tma_load_4d(smem + kK1Off + 8192, ktm, 0, 64, 0, ph1, b_kv2full + 0);
        tma_load_4d(smem + kK1Off + 16384, ktm, 0, 0, 1, ph1, b_kv2full + 0);
        tma_load_4d(smem + kK1Off + 24576, ktm, 0, 64, 1, ph1, b_kv2full + 0);
        mbar_arrive_tx(b_kv2full + 8, 32768);
        tma_load_4d(smem + kV1Off + 0, vtm, 0, 0, 0, ph1, b_kv2full + 8);
        tma_load_4d(smem + kV1Off + 8192, vtm, 0, 64, 0, ph1, b_kv2full + 8);
        tma_load_4d(smem + kV1Off + 16384, vtm, 0, 0, 1, ph1, b_kv2full + 8);
        tma_load_4d(smem + kV1Off + 24576, vtm, 0, 64, 1, ph1, b_kv2full + 8);
      }
    }
    if (warp == 13) {
      // Stage BOTH chunks' 32 row entries each (rown[32:64] = chunk 1) +
      // per-chunk visibility; rowvcol slots carry the {rc, kvh} pairs the
      // gather/epilogue readers need.
      for (int i = lane; i < 64; i += 32) {
        const int ch = i >> 5;
        const int loc = i & 31;
        int ent = -1;
        if (loc < t_rc[ch]) ent = __ldg(p.rowlist + t_rq0[ch] + loc);
        sm_rown[i] = (ent < 0) ? -1 : (ent & 0x0FFFFFFF);
        sm_rowrank[i] = (ent < 0) ? 0 : (ent >> 28);
      }
      if (lane < 2) {
        sm_stile[lane * 2] = t_rc[lane];
        sm_stile[lane * 2 + 1] = t_kvh[lane];
      }
    }
    asm volatile("barrier.sync 9, 128;" ::: "memory");
    // Q gather: 256 rows x 2 elhalves x 8 16B segs, rows (q_i*4 + gg), into
    // the exact TMA SWIZZLE_128B slab layout the proven QK descriptor walk
    // expects ([elhalf 16KB][128 rows x 128B]).
    const char* qbase = reinterpret_cast<const char*>(p.q);
    const int t128 = (warp - 12) * 32 + lane;  // 0..127
    // Issue BOTH stages' cp.async chunks before waiting: one commit group
    // per stage, so wait<1> retires stage 0 while stage 1 is still in flight.
    for (int stage = 0; stage < 2; stage++) {
      const int qstage = smem + (stage ? kQ1Off : kQ0Off);
      const int rc_stage = sm_stile[stage * 2];
      const int kvh_stage = sm_stile[stage * 2 + 1];
      // 2048 chunks per stage; 128 threads x 16 each
      for (int idx = t128; idx < 2048; idx += 128) {
        const int r = idx >> 4;           // row in stage (0..127)
        const int seg = idx & 7;          // 16B segment in elhalf
        const int half = (idx >> 3) & 1;  // elhalf
        const int q_i = (stage * 128 + r) >> 2;
        if (q_i - (stage << 5) >= rc_stage) continue;  // chunk-local row index
        const int n = sm_rown[q_i];
        const int gg = r & 3;
        const long src_el =
            (long)n * p.q_tok + (long)(kvh_stage * kGroup + gg) * p.q_head + half * 64 + seg * 8;
        const int dst = qstage + half * 16384 + r * 128 + (((seg ^ (r & 7))) << 4);
        cp16(dst, qbase + src_el * (long)sizeof(QT));
      }
      cp_commit();
    }
    cp_wait<1>();  // stage-0 group landed (stage 1 may still be in flight)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 9, 128;" ::: "memory");
    if (warp == 13 && elect_one()) mbar_arrive(b_qfull + 0);
    cp_wait<0>();  // stage-1 group landed
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 9, 128;" ::: "memory");
    if (warp == 13 && elect_one()) mbar_arrive(b_qfull + 8);
    if (warp != 12) return;
  }

  // ================= Role: MMA leader (warp 12) =================
  if (warp == 12) {
    const int q0_lo = warp_uni(((smem + kQ0Off) >> 4) & 0x3FFF);
    const int q1_lo = warp_uni(((smem + kQ1Off) >> 4) & 0x3FFF);
    const int kb_lo = warp_uni(((smem + kKOff) >> 4) & 0x3FFF);
    const int vb_lo = warp_uni((((smem + kVOff) >> 4) & 0x3FFF) | 0x4000000);
    // Same-bucket pairs and the fake tail tile share the stage-0 K/V slabs;
    // the producer skipped the stage-1 duplicate load under the same rule.
    const bool share_kv = __ldg(p.chunk_bkt + c0) == __ldg(p.chunk_bkt + c1v);
    const int kb1_lo = share_kv ? kb_lo : warp_uni(((smem + kK1Off) >> 4) & 0x3FFF);
    const int vb1_lo = share_kv ? vb_lo : warp_uni((((smem + kV1Off) >> 4) & 0x3FFF) | 0x4000000);
    mbar_wait(b_qfull + 0, 0);
    mbar_wait(b_kvfull + 0, 0);
    mma_qk_group(q0_lo, kb_lo, taddr + 0, idesc_qk, 0);
    tc_commit(b_sfull + 0);
    mbar_wait(b_qfull + 8, 0);
    if (!share_kv) mbar_wait(b_kv2full + 0, 0);
    mma_qk_group(q1_lo, kb1_lo, taddr + 128, idesc_qk, 0);
    tc_commit(b_sfull + 8);
    mbar_wait(b_kvfull + 8, 0);
    mbar_wait(b_pfull + 0, 0);
    mma_pv_head(vb_lo, taddr + 256, taddr + 64, idesc_pv, 0);
    mbar_wait(b_ptail + 0, 0);
    mma_pv_tail(vb_lo, taddr + 256, taddr + 64, idesc_pv);
    if (!share_kv) mbar_wait(b_kv2full + 8, 0);
    mbar_wait(b_pfull + 8, 0);
    mma_pv_head(vb1_lo, taddr + 384, taddr + 192, idesc_pv, 0);
    mbar_wait(b_ptail + 8, 0);
    mma_pv_tail(vb1_lo, taddr + 384, taddr + 192, idesc_pv);
    tc_commit(b_ofull + 0);
    tc_commit(b_ofull + 8);
    mbar_wait(b_dealloc, 0);
    tc_dealloc(taddr, 512);
    tc_relinquish();
    return;
  }
}

// ==== ANCHOR:MERGE ====
// Split-KV merge across each row's accepted blocks: out = sum_s w_s acc_s /
// sum_s w_s l_s with w_s = exp2((m_s - M) * scale). Rows with no accepted
// block produce exact zeros (matches the reference's prob-zeroing).
// Round 27: two rows per warp (16 lanes each, 16B loads/stores at the fp16
// partial layout) — the one-row/warp uint2 form measured latency-bound at
// ~3.6 TB/s effective; doubling per-lane load width doubles ILP.
template <typename QT, bool IS_BF16>
__global__ void __launch_bounds__(256, 8) g4_merge_kernel(const __grid_constant__ G4Params p) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const long row = ((long)blockIdx.x * 8 + warp) * 2 + (lane >> 4);
  const long rows = (long)p.total_q * p.num_q_heads;
  if (row >= rows) return;
  const int li = lane & 15;  // this lane owns output elements [8*li, 8*li+8)
  const int Hq = p.num_q_heads;
  const long n = row / Hq;
  const int qh = (int)(row - n * Hq);
  const int kvh = qh >> 2;
  const int cnt = min(p.cnt_hn[kvh * (long)p.total_q + n], p.topk);
  QT* out_row = reinterpret_cast<QT*>(p.out) + row * 128 + li * 8;
  if (cnt == 0) {
    uint4 z = {0u, 0u, 0u, 0u};
    *reinterpret_cast<uint4*>(out_row) = z;
    return;
  }
  const long sbase = row * p.topk;
  float2 ml[8];
#pragma unroll
  for (int s = 0; s < 8; s++) {
    if (s < cnt) ml[s] = *reinterpret_cast<const float2*>(p.ws_ml + (sbase + s) * 2);
  }
  float M = -CUDART_INF_F;
#pragma unroll
  for (int s = 0; s < 8; s++) {
    if (s < cnt) M = fmaxf(M, ml[s].x);
  }
  float w[8];
  float l_tot = 0.f;
#pragma unroll
  for (int s = 0; s < 8; s++) {
    if (s < cnt) {
      w[s] = fast_exp2(fmaf(ml[s].x - M, p.scale_log2e, 0.f));
      l_tot += w[s] * ml[s].y;
    }
  }
  const float inv = (l_tot > 0.f) ? (1.f / l_tot) : 0.f;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++) acc[j] = 0.f;
  const __half* ab = reinterpret_cast<const __half*>(p.ws_acc);
  for (int s = 0; s < cnt; s++) {
    const uint4 h = *reinterpret_cast<const uint4*>(ab + (sbase + s) * 128 + li * 8);
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
    const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&h.z));
    const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&h.w));
    const float ws = w[s];
    acc[0] += ws * f0.x;
    acc[1] += ws * f0.y;
    acc[2] += ws * f1.x;
    acc[3] += ws * f1.y;
    acc[4] += ws * f2.x;
    acc[5] += ws * f2.y;
    acc[6] += ws * f3.x;
    acc[7] += ws * f3.y;
  }
  uint4 pk;
  pk.x = IS_BF16 ? pack_bf16x2(acc[0] * inv, acc[1] * inv) : pack_f16x2(acc[0] * inv, acc[1] * inv);
  pk.y = IS_BF16 ? pack_bf16x2(acc[2] * inv, acc[3] * inv) : pack_f16x2(acc[2] * inv, acc[3] * inv);
  pk.z = IS_BF16 ? pack_bf16x2(acc[4] * inv, acc[5] * inv) : pack_f16x2(acc[4] * inv, acc[5] * inv);
  pk.w = IS_BF16 ? pack_bf16x2(acc[6] * inv, acc[7] * inv) : pack_f16x2(acc[6] * inv, acc[7] * inv);
  *reinterpret_cast<uint4*>(out_row) = pk;
}

// ==== ANCHOR:HOST ====
// Paged KV map: [pages, Hkv, 128, 128] -> 4D {64 el, 128 tok, 2 elhalf,
// pages*Hkv}, box {64, 64, 1, 1} (8 KB per call; matches the proven slab).
static CUtensorMap encode_paged_kv_map(const void* base, int num_pages, int hkv,
                                       CUtensorMapDataType dt) {
  uint64_t gdim[4] = {64ull, 128ull, 2ull, (uint64_t)(num_pages * hkv)};
  uint64_t gstride[3] = {256ull, 128ull, 128ull * 128ull * 2ull};
  uint32_t box[4] = {64u, 64u, 1u, 1u};
  uint32_t estride[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  cuTensorMapEncodeTiled(&tm, dt, 4, const_cast<void*>(base), gdim, gstride, box, estride,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tm;
}

bool umma_g4_eligible(int group, bool paged, int kv_dtype_code, int topk, int nbatch, int max_pages,
                      int num_kv_heads, int total_q) {
  if (group != kGroup || !paged || kv_dtype_code == 2) return false;
  if (topk < 1 || topk > kMaxTopk) return false;
  const long nbuckets = (long)num_kv_heads * nbatch * max_pages;
  if (nbuckets < 1 || nbuckets > 32768) return false;
  if (max_pages > 65536) return false;  // meta packs blk into 16 bits
  const long slots = (long)total_q * num_kv_heads * kGroup * topk;
  if (slots > (1l << 22)) return false;  // ws_acc workspace cap (2 GB fp32)
  return true;
}

template <typename QT, bool IS_BF16>
static void launch_g4(G4Params& p, const CUtensorMap& km, const CUtensorMap& vm,
                      cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(g4_umma_kernel<QT, IS_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemTotal);
    attr_set = true;
  }
  g4_umma_kernel<QT, IS_BF16>
      <<<dim3((unsigned)((p.tiles_bound + 1) / 2)), kThreads, kSmemTotal, stream>>>(p, km, vm);
}

void umma_g4_forward(const void* q, bool q_is_bf16, const void* k, const void* v, const int* q2k,
                     const int* cu_q, const int* cu_k, const int* page_table, void* out,
                     int total_q, int num_q_heads, int num_kv_heads, int topk, int nbatch,
                     int num_pages, int max_pages, long pt_stride, long q_tok, long q_head,
                     long o_tok, long o_head, int* ws_int, float* ws_float, int seqlen_q,
                     bool causal, cudaStream_t stream) {
  G4Params p;
  p.q = q;
  p.k = k;
  p.v = v;
  p.q2k = q2k;
  p.cu_q = cu_q;
  p.cu_k = cu_k;
  p.page_table = page_table;
  p.out = out;
  p.total_q = total_q;
  p.num_q_heads = num_q_heads;
  p.num_kv_heads = num_kv_heads;
  p.topk = topk;
  p.nbatch = nbatch;
  p.max_pages = max_pages;
  p.nbuckets = p.num_kv_heads * p.nbatch * p.max_pages;
  p.pt_stride = pt_stride;
  p.q_tok = q_tok;
  p.q_head = q_head;
  p.o_tok = o_tok;
  p.o_head = o_head;
  p.seqlen_q = seqlen_q;
  p.causal = causal ? 1 : 0;
  p.scale_log2e = (float)(0.08838834764831845 * M_LOG2E);  // 128^-0.5 * log2(e)
  // Round 27: the last-ticket in-kernel merge measured out in three variants
  // (456 / 456 / 397 us wall vs 288 us for the parallel tail kernel) because
  // the ~425 MB of partial traffic is DRAM-bound and only the massively
  // parallel tail kernel feeds DRAM at rate. The separate merge kernel is the
  // shipping path; the inert merge hooks stay in the device code with
  // inline_merge pinned to 0.
  p.inline_merge = 0;

  const long hn = (long)p.num_kv_heads * p.total_q;
  const long rows = (long)p.total_q * p.num_q_heads;
  const long slots = rows * p.topk;
  p.rows_bound = (int)(hn * p.topk + (long)p.nbuckets * kQTile);
  p.tiles_bound = (int)(hn * p.topk / kQTile + p.nbuckets);

  // Caller-provided grow-only workspace (ints + floats), exactly the offsets
  // the proven level used; bcnt/bcnt2/rowcnt are re-zeroed by the route stage
  // on every call, all other tables are fully produced before use.
  const long rowcnt_span = 0;  // inline-merge completion counters (inert)
  p.bcnt = ws_int;
  p.bcnt2 = p.bcnt + p.nbuckets;
  p.rowcnt = p.bcnt2 + p.nbuckets;
  p.cnt_hn = p.rowcnt + rowcnt_span;
  p.boff = p.cnt_hn + hn;
  p.btoff = p.boff + (p.nbuckets + 1);
  p.bmeta = p.btoff + (p.nbuckets + 1);
  p.chunk_bkt = p.bmeta + p.nbuckets * 5;
  p.rankmap = p.chunk_bkt + p.tiles_bound;
  p.rowlist = p.rankmap + hn * p.topk;
  p.tile_total = p.rowlist + p.rows_bound;
  p.route_bar = p.tile_total + 1;
  p.ws_acc = ws_float;
  p.ws_ml = p.ws_acc + slots * 64;

  cudaStream_t s = stream;
  const long hn_blocks = (hn + 255) / 256;
  // Round 26: fused cooperative route kernel for fully co-resident, small-bucket
  // grids (replaces the fill + 3 launches; ranks stay in registers); the legacy
  // memset + 3-kernel chain remains the general fallback. Round 35: in-kernel
  // one-shot grid barriers on per-call workspace state replaced
  // grid_group::sync(); the cooperative launch stays only for the driver's
  // co-residency guarantee (grid barriers deadlock if any CTA is evicted).
  const bool fused_ok = (hn_blocks <= 128) && (p.nbuckets <= 6144);
  bool routed = false;
  if (fused_ok) {
    cudaMemsetAsync(p.route_bar, 0, 4 * sizeof(int), s);
    void* kargs[] = {&p};
    const cudaError_t cerr = cudaLaunchCooperativeKernel(
        (const void*)g4_route_fused, dim3((unsigned)hn_blocks), dim3(256), kargs, 0, s);
    if (cerr == cudaSuccess) {
      routed = true;
    } else {
      (void)cudaGetLastError();  // clear the rejected cooperative launch
    }
  }
  if (!routed) {
    cudaMemsetAsync(p.bcnt, 0, (size_t)(2 * p.nbuckets + rowcnt_span) * sizeof(int), s);
    g4_route_count<<<dim3((unsigned)hn_blocks), 256, 0, s>>>(p);
    g4_route_scan<<<1, 1024, 0, s>>>(p);
    g4_route_scatter<<<dim3((unsigned)hn_blocks), 256, 0, s>>>(p);
  }

  const CUtensorMapDataType dt =
      q_is_bf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  CUtensorMap km = encode_paged_kv_map(p.k, num_pages, p.num_kv_heads, dt);
  CUtensorMap vm = encode_paged_kv_map(p.v, num_pages, p.num_kv_heads, dt);
  const unsigned merge_blocks = (unsigned)((rows + 15) / 16);
  if (q_is_bf16) {
    launch_g4<__nv_bfloat16, true>(p, km, vm, s);
    g4_merge_kernel<__nv_bfloat16, true><<<dim3(merge_blocks), 256, 0, s>>>(p);
  } else {
    launch_g4<__half, false>(p, km, vm, s);
    g4_merge_kernel<__half, false><<<dim3(merge_blocks), 256, 0, s>>>(p);
  }
  cudaError_t le = cudaGetLastError();
  MSAV_CHECK(le == cudaSuccess, "umma_g4 launch failed: %s", cudaGetErrorString(le));
}

}  // namespace msa_umma_g4
