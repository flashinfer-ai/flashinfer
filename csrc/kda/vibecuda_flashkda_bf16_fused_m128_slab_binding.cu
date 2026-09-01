// Binding TU for the compile-time slab-specialized VibeCUDA FlashKDA M128
// prefill kernel (the runtime-regime union with the combined N=160 UMMA-4
// branch removed). All
// __global__ symbols are macro-renamed at include time so this TU links
// alongside the union TU (kda_flash_binding_m128.cu) in one extension. The
// host dispatches here whenever the slab regime rule selects ft_slab=1.
//
// Split-seq (hierarchical affine prefix) orchestration: RunM128SplitFull
// issues the four-launch pipeline for small-BH ultra-long workloads:
//   (1) main pass, grid.y = P parts with zero initial state exporting each
//       part's additive transform (fp32 zero-init end state S_p);
//   (2) map pass, grid.y = P parts, V := 0 and identity initial state,
//       exporting each part's exact linear operator M_p;
//   (3) dense scan composing per-part prefix (carry) states
//       c_{p+1} = S_p + M_p x c_p;
//   (4) correction pass, grid.y = P-1, relaunching the same kernel with V := 0
//       and the scanned carry as initial state, accumulating the exact carry
//       contribution onto the zero-init main-pass output. Exactness follows
//       from the state recurrence being affine in the state given the tokens.
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "vibecuda_flashkda_tma.cuh"

// The generated standalone source declares its own fixed-width typedefs;
// isolate them so they do not collide with this TU's CUDA headers.
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
// Rename every __global__ symbol in the slab copy so both TUs link together.
#define kernel_flashkda_bf16_fused_m128 kernel_flashkda_bf16_fused_m128_slab
#define kernel_flashkda_split_scan_m128 kernel_flashkda_split_scan_m128_slab
#define kernel_flashkda_split_scan_bf16_m128 kernel_flashkda_split_scan_bf16_m128_slab
#define kernel_flashkda_split_lookback_m128 kernel_flashkda_split_lookback_m128_slab
#define kernel_flashkda_split_out_add_m128 kernel_flashkda_split_out_add_m128_slab
#include "vibecuda_flashkda_bf16_fused_m128_slab.cu"
// NOTE: the kernel-name renames stay active for the whole TU so every
// reference below (launch, cudaFuncSetAttribute, template instantiation)
// resolves to this TU's slab kernels, not the union TU's symbols. They are
// #undef'd at the end of the file.
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_flash_slab {
using namespace kda_flash;

// Round-73 P=2 specialization (KDA_P2SPEC, DEFAULT ON; supervisor directive):
// at num_parts == 2 the (map + scan) middle of the split pipeline is dead
// work: the map pass's exported M panels are only consumed by the scan, whose
// only EMITTED carry is c_1 = S_0 (the bf16 register-carry scan initializes
// its register carry from split_state slot 0 and emits it BEFORE the
// discarded M_1 matvec; c_2 would be the final state, which split callers
// never store). Replace both launches with the exact fp32 slab copy of S_0
// into carry slot 0, preserving the scan's seq_order task mapping
// (split_state export slot is [bid*P + 0]; carry slot 0 is
// [seq_idx*num_heads + head_idx]; the correction pass reads part_y = 0 at
// offset task*16384 with the same mapping).
// (This kernel ALSO lives in kda_flash_binding_m128.cu; keep both in sync.)
__global__ __launch_bounds__(256) void kernel_flashkda_split_carry_copy_m128_slab(
    const float* __restrict__ split_state, float* __restrict__ carry,
    const int* __restrict__ seq_order, int num_heads) {
  const long long task = blockIdx.x;
  const long long seq_idx = static_cast<long long>(seq_order[task / num_heads]);
  const long long head_idx = task % num_heads;
  const float4* src = reinterpret_cast<const float4*>(split_state + task * 2 * 16384);
  float4* dst = reinterpret_cast<float4*>(
      carry + (seq_idx * static_cast<long long>(num_heads) + head_idx) * 16384);
#pragma unroll
  for (int i = 0; i < 16; i++) dst[threadIdx.x + i * 256] = src[threadIdx.x + i * 256];
}

static_assert(THREADS == 1024);

// Round-75 GPREP operand channel (supervisor directive): the producer-side
// preprocessing (L2 normalize + gate scan + decay scaling + beta sigmoid +
// chunk gt) runs in a standalone producer whose outputs are published once
// per forward call through these TU-scope statics. The workspace lock
// serializes forward calls and every consumed operand is recomputed on each
// call from the current inputs, so these pointers always reference freshly
// produced buffers (no result caching; capacity-grown persistent workspace).
// Only the KDA_GPREP kernel build consumes them; the default build (and the
// non-slab regime of any build) never reads them.
static const void* g_gprep_qd = nullptr;
static const void* g_gprep_kd = nullptr;
static const void* g_gprep_ki = nullptr;
static const void* g_gprep_gt = nullptr;
static const void* g_gprep_bsig = nullptr;
static const int* g_gprep_seq_chunk_base = nullptr;
static float g_gprep_rc = 1.0f;
static float g_gprep_ic = 1.0f;

// lower_bound drives the two restore constants: rc = 2^c, ic = 2^-c with
// c = lower_bound * log2(e) * 16.
void SetGprepOperands(const void* qd, const void* kd, const void* ki, const void* gt,
                      const void* bsig, const void* seq_chunk_base, double lower_bound) {
  g_gprep_qd = qd;
  g_gprep_kd = kd;
  g_gprep_ki = ki;
  g_gprep_gt = gt;
  g_gprep_bsig = bsig;
  g_gprep_seq_chunk_base = reinterpret_cast<const int*>(seq_chunk_base);
  const double c = lower_bound * 1.4426950408889634 * 16.0;
  g_gprep_rc = static_cast<float>(exp2(c));
  g_gprep_ic = static_cast<float>(exp2(-c));
}

#if defined(KDA_TF32_U2)
#if defined(KDA_PREP_PIPE2)
static_assert(SMEM_TOTAL == 218624 + 3200);
#else
static_assert(SMEM_TOTAL == 218624);
#endif
#else
#if defined(KDA_PREP_PIPE2)
static_assert(SMEM_TOTAL == 227328 + 3200 + 512);
#else
static_assert(SMEM_TOTAL == 227328 + 512);
#endif
#endif

static void LaunchM128(const TmaPointers& tma, const void* q_ptr, const void* k_ptr,
                       const void* v_ptr, const void* g_ptr, const void* beta_ptr,
                       const void* A_log_ptr, const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                       const void* seq_order_ptr, const void* initial_state_ptr, void* out_ptr,
                       void* final_state_ptr, int64_t num_seqs, int64_t num_heads,
                       int64_t use_initial_state, int64_t store_final_state, double scale,
                       double lower_bound, int64_t split_num_parts, void* split_state_ptr,
                       void* split_gamma_ptr, int64_t fixup_mode, int64_t grid_y,
                       cudaStream_t stream, int* progress_flags, void* map_state_bf16_ptr,
                       int64_t ft_slab) {
  const dim3 grid(static_cast<uint32_t>(num_seqs * num_heads), static_cast<uint32_t>(grid_y), 1);
  const dim3 block(THREADS, 1, 1);
  // Full-chunk correction stores target the scratch-out descriptor (slot 6);
  // every other launch targets out (same address for non-split callers).
  const void* out2_tma = ((fixup_mode & 1) == 1) ? tma.scratch_out : tma.out;
#if defined(KDA_PDL)
  // Round-134 programmatic dependent launch (supervisor directive, sm_103a
  // image + runtime dispatch): the launch-boundary overlap for the
  // launch-sensitive H96 fixed and packed rows. This TU's per-call chain for
  // those rows is a single fused launch (prep is producer-fused inside the
  // kernel), so the overlappable boundary is the launch itself; the split
  // path additionally benefits at the main->map->scan->correction->out_add
  // boundaries since every LaunchM128 goes through here. The PSS attribute
  // marks this launch's grid as allowed to schedule before the predecessor
  // completes; the kernel's entry-point griddepcontrol.wait (compiled under
  // KDA_PDL) restores the serialized-launch memory ordering before the first
  // dependent global/TMA read. Nobody issues griddepcontrol.launch_dependents
  // (the in-place initial_state write must stay ordered before any successor
  // read), so the trigger is the implicit primary-completion one: identical
  // ordering semantics to the legacy launch, minus the CTA-scheduling bubble.
  // Runtime A/B: default ON on the sm_103a image; KDA_PDL=0 opts back to the
  // plain <<<>>> launch below. Decision-grade A/B must interleave legs in one
  // window per the R59 ruling.
  static const bool kPdlOn = [] {
    const char* e = std::getenv("KDA_PDL");
    return e == nullptr || std::atoi(e) != 0;
  }();
  if (kPdlOn) {
    cudaLaunchConfig_t pdl_cfg = {};
    cudaLaunchAttribute pdl_attr[1];
    pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;
    pdl_cfg.gridDim = grid;
    pdl_cfg.blockDim = block;
    pdl_cfg.dynamicSmemBytes = SMEM_TOTAL;
    pdl_cfg.stream = stream;
    pdl_cfg.attrs = pdl_attr;
    pdl_cfg.numAttrs = 1;
    CheckCuda(cudaLaunchKernelEx(
                  &pdl_cfg, kernel_flashkda_bf16_fused_m128,
                  reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(q_ptr)), tma.q,
                  reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(k_ptr)), tma.k,
                  reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v_ptr)), tma.v,
                  reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(g_ptr)), tma.g,
                  reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(beta_ptr)), tma.beta,
                  reinterpret_cast<float*>(const_cast<void*>(A_log_ptr)),
                  reinterpret_cast<float*>(const_cast<void*>(dt_bias_ptr)),
                  reinterpret_cast<long long*>(const_cast<void*>(cu_seqlens_ptr)),
                  reinterpret_cast<int*>(const_cast<void*>(seq_order_ptr)),
                  reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(initial_state_ptr)),
                  reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out, out2_tma,
                  reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
                  static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
                  static_cast<float>(scale), static_cast<float>(lower_bound),
                  static_cast<int>(split_num_parts), reinterpret_cast<float*>(split_state_ptr),
                  reinterpret_cast<float*>(split_gamma_ptr), static_cast<int>(fixup_mode),
                  progress_flags, reinterpret_cast<__nv_bfloat16*>(map_state_bf16_ptr),
                  static_cast<int>(ft_slab)
#if defined(KDA_GPREP)
                  // Round-75 GPREP operand channel (see SetGprepOperands above).
                  ,
                  reinterpret_cast<const __nv_bfloat16*>(g_gprep_qd),
                  reinterpret_cast<const __nv_bfloat16*>(g_gprep_kd),
                  reinterpret_cast<const __nv_bfloat16*>(g_gprep_ki),
                  reinterpret_cast<const float*>(g_gprep_gt),
                  reinterpret_cast<const float*>(g_gprep_bsig), g_gprep_seq_chunk_base, g_gprep_rc,
                  g_gprep_ic
#endif
                  ),
              "kernel_flashkda_bf16_fused_m128 PDL launch");
    return;
  }
#endif
  kernel_flashkda_bf16_fused_m128<<<grid, block, SMEM_TOTAL, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(q_ptr)), tma.q,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(k_ptr)), tma.k,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v_ptr)), tma.v,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(g_ptr)), tma.g,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(beta_ptr)), tma.beta,
      reinterpret_cast<float*>(const_cast<void*>(A_log_ptr)),
      reinterpret_cast<float*>(const_cast<void*>(dt_bias_ptr)),
      reinterpret_cast<long long*>(const_cast<void*>(cu_seqlens_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(seq_order_ptr)),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(initial_state_ptr)),
      reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out, out2_tma,
      reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
      static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound), static_cast<int>(split_num_parts),
      reinterpret_cast<float*>(split_state_ptr), reinterpret_cast<float*>(split_gamma_ptr),
      static_cast<int>(fixup_mode), progress_flags,
      reinterpret_cast<__nv_bfloat16*>(map_state_bf16_ptr),
      static_cast<int>(ft_slab)
#if defined(KDA_GPREP)
      // Round-75 GPREP operand channel (see SetGprepOperands above).
      ,
      reinterpret_cast<const __nv_bfloat16*>(g_gprep_qd),
      reinterpret_cast<const __nv_bfloat16*>(g_gprep_kd),
      reinterpret_cast<const __nv_bfloat16*>(g_gprep_ki),
      reinterpret_cast<const float*>(g_gprep_gt), reinterpret_cast<const float*>(g_gprep_bsig),
      g_gprep_seq_chunk_base, g_gprep_rc, g_gprep_ic
#endif
  );
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_fused_m128 launch");
}

void RunM128(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
             const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
             const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
             int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
             int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream) {
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  // Round-120 (shared): exactly-once SMEM-size attribute registration —
  // see the kda_flash_binding_m128.cu RunM128 hoist for rationale. The
  // slab TU is the live route for the packed ragged and split shapes,
  // so this removes 1 (7 on split) driver calls per forward here too.
  static const bool kSmemAttrOnce = [] {
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
        "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128)");
    return true;
  }();
  (void)kSmemAttrOnce;

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<128>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out_ptr=*/nullptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);
  LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
             seq_order_ptr, initial_state_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
             use_initial_state, store_final_state, scale, lower_bound,
             /*split_num_parts=*/1, /*split_state_ptr=*/nullptr, /*split_gamma_ptr=*/nullptr,
             /*fixup_mode=*/0, /*grid_y=*/1, stream, /*progress_flags=*/nullptr,
             /*map_state_bf16_ptr=*/nullptr, ft_slab);
}

void RunM128SplitFull(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                      const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                      const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                      const void* seq_order_ptr, const void* initial_state_ptr, void* out_ptr,
                      void* final_state_ptr, void* descriptor_storage_ptr, int64_t token_count,
                      int64_t num_seqs, int64_t prepare_descriptors, int64_t num_heads,
                      int64_t use_initial_state, double scale, double lower_bound,
                      int64_t beta_tma_rows, int64_t beta_tma_dim1, int64_t num_parts,
                      void* split_state_ptr, void* map_state_ptr, void* carry_ptr,
                      void* split_out_ptr, void* lookback_flags_ptr, void* map_state_bf16_ptr,
                      int64_t ft_slab, int64_t cuda_stream) {
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  // Round-120 (shared): exactly-once SMEM-size attribute registration —
  // see the kda_flash_binding_m128.cu RunM128SplitFull hoist for
  // rationale (7 driver calls per split forward now run exactly once).
  static const bool kSplitSmemAttrOnce = [] {
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
        "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128)");
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<4, 0>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             141504),  // LDG path: 2*128*132*4 (M) + 3*4*132*4 (c, R49 CPAD=132)
        "cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<4,0>)");
    kda_flash::CheckCuda(cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<2, 0>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              138336),  // LDG path: 2*128*132*4 (M) + 3*2*132*4 (c)
                         "cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<2,0>)");
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<4, 1>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             137424),  // BULK: 2*128*128*4 (M) + 3*4*132*4 (c) + 16 (mbarriers)
        "cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<4,1>)");
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<2, 1>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             134256),  // BULK: 2*128*128*4 (M) + 3*2*132*4 (c) + 16 (mbarriers)
        "cudaFuncSetAttribute(kernel_flashkda_split_scan_m128<2,1>)");
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_split_scan_bf16_m128,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             65552),  // BF16 M: 2*32KB panels + 16B mbarriers (register carry)
        "cudaFuncSetAttribute(kernel_flashkda_split_scan_bf16_m128)");
    kda_flash::CheckCuda(
        cudaFuncSetAttribute(kernel_flashkda_split_lookback_m128,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, LOOKBACK_SMEM_BYTES),
        "cudaFuncSetAttribute(kernel_flashkda_split_lookback_m128)");
    return true;
  }();
  (void)kSplitSmemAttrOnce;
  // Round-48 supervisor directive: the R45 v2 serial scan is the baseline
  // (the R47 decoupled-lookback path regressed the authoritative average to
  // 1.468x and its smoke test hung unresolved). The round-48 fused 32-band
  // scan/correction launch exists but is OFF by default: measured +140us at
  // P=128 (131072xH1) vs the separate scan+correction launches; per-pass
  // penalty ~1.1us traced to the scan role inside the 1024-thread
  // launch_bounds (64-reg) code losing the register-batched M-prefetch
  // software pipelining. KDA_FUSED_SCAN=1 re-enables it; KDA_LOOKBACK=1
  // re-enables the R47 lookback kernel for debugging only.
  const char* _lb_env = std::getenv("KDA_LOOKBACK");
  const bool kLookbackOn = (_lb_env != nullptr) && (std::atoi(_lb_env) != 0);
  const char* _fs_env = std::getenv("KDA_FUSED_SCAN");
  const bool kFusedScan = !kLookbackOn && (_fs_env != nullptr) && (std::atoi(_fs_env) != 0);
  // Round-121 (SM103 supervisor directive, env-gated OFF): underfilled
  // fused-scan specialization. The R48 fused scan/correction launch was
  // disabled suite-wide after the +140us P=128 regression (scan role losing
  // the register-batched M-prefetch under launch_bounds). That penalty was
  // traced at HIGH part counts; for underfilled regimes (small P, tasks*P
  // within one wave) the fused launch eliminates one full launch boundary
  // (scan+correction merge) and overlaps the correction prologue with the
  // scan bands. KDA_FUSED_SMALLP=1 enables the fused path only when
  // 2 < num_parts <= 32 (the P=128/P=148 regression shapes never qualify;
  // P==2 is already covered by the cheaper P2Spec carry-copy). SM103 A/B
  // target: 65536xH8 (P=18); SM100 must stay regression-free. Default OFF
  // until retained by a same-window A/B on both archs.
  const char* _fsp_env = std::getenv("KDA_FUSED_SMALLP");
  const bool kFusedSmallP = !kLookbackOn && !kFusedScan && (_fsp_env != nullptr) &&
                            (std::atoi(_fsp_env) != 0) && (num_parts > 2) && (num_parts <= 32);
  const bool kFusedAny = kFusedScan || kFusedSmallP;

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<128>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out=*/split_out_ptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);
  const int64_t num_tasks = num_seqs * num_heads;
  // Round-118 dead-map generalization of the R73 P=2 specialization (DEFAULT
  // ON under the default bf16 scan; KDA_DEAD_MAP=0 restores the exact legacy
  // map+matvec path per part). A part is flagged when its cumulative gate-
  // decay product flushed to fp32 +0.0f on EVERY state channel. Rigor, not
  // proof: each per-token factor diag(d_t)(I - beta k k^T) has spectral norm
  // <= max_i d_i(t) (exactly-normalized k and beta in (0,1) make
  // (I - beta k k^T) a contraction), so ||M_p||_2 <= Prod_t max_i d_i(t).
  // The tracked flag quantity is the per-channel max
  // A = max_c Prod_t d_c(t) <= Prod_t max_i d_i(t): A == 0 certifies a dead
  // operator only when the same channel attains the per-token max throughout
  // the part (uniform decay). That regime is what this suite realizes: the
  // R118 decay probe measured cumulative log-decay <= -129 per 128 tokens
  // per channel and <= -1541 over full split parts, versus the ~-104 fp32
  // underflow threshold, so flagged parts drop a term orders of magnitude
  // below the 1e-2 atol for any representable carry. Under adversarial
  // epoch-separated gates (per-token max channel switching) the certificate
  // is not airtight; treat the flag as a suite-validated heuristic and use
  // KDA_DEAD_MAP=0 for the exact fallback wherever it is in doubt. For
  // flagged parts the map pass is dead work and the scan's matvec step
  // composes the carry copy c_{p+1} = S_p; the unflagged path keeps the
  // exact legacy map+matvec per part. The lookback_flags workspace
  // buffer doubles as the per-part flag channel (its fused/lookback roles
  // are mutually exclusive with this default path).
  const char* _dm_env = std::getenv("KDA_DEAD_MAP");
  const char* _sbf_env0 = std::getenv("KDA_SCAN_BF16");
  const bool kScanBf16Default =
      (_sbf_env0 == nullptr || std::atoi(_sbf_env0) != 0) && (map_state_bf16_ptr != nullptr);
  const bool kDeadMap = (_dm_env == nullptr || std::atoi(_dm_env) != 0) && (num_parts > 2) &&
                        !kLookbackOn && !kFusedAny && kScanBf16Default;
  int* dead_flags_ptr = reinterpret_cast<int*>(lookback_flags_ptr);
  // (1) Main pass: parts [0, P) with zero initial state and the real inputs;
  // the exported end states are the parts' additive transforms S_p. Under
  // kDeadMap it also publishes the per-part dead-map flags.
  LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
             seq_order_ptr, initial_state_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
             use_initial_state, /*store_final_state=*/0, scale, lower_bound, num_parts,
             split_state_ptr, /*split_gamma_ptr=*/nullptr, /*fixup_mode=*/0,
             /*grid_y=*/num_parts, stream,
             /*progress_flags=*/kDeadMap ? dead_flags_ptr : nullptr,
             /*map_state_bf16_ptr=*/nullptr, ft_slab);
  // (2) Map pass: parts [0, P), V := 0, identity initial state; the exported
  // end states are the parts' exact linear operators M_p (their row v is the
  // evolved row basis vector e_v).
  //
  // Round-73 P=2 specialization (KDA_P2SPEC, DEFAULT ON): at P == 2
  // the map pass is dead work. The scan's carry recurrence starts from
  // c_1 = S_0 (the main pass folded the real initial state into part 0; see
  // the init guard `fixup_mode != 0 || split_part == 0` in the kernel and the
  // scan's "c_1 = end state of part 0" comment), so carry slot 0 = c_1 needs
  // no M_0. The scan's only remaining M read produces c_2 = the FINAL state,
  // which split callers never store (store_final_state=0), so M_1's product
  // is discarded too. Skipping the map launch removes one full duplicated
  // q/k/g-prep walk of every token, and the scan is replaced by the exact
  // S_0->carry copy kernel (see kernel_flashkda_split_carry_copy_m128_slab).
  // Round-73 supervisor directive: the P=2 specialization supersedes the R72
  // KDA_HSPLIT2 gating; it is DEFAULT ON at num_parts == 2 and applies to
  // every split launch at P == 2, env-gated OFF via KDA_P2SPEC=0 for A/B.
  // (This gate ALSO lives in kda_flash_binding_m128.cu; keep both in sync.)
  const char* _p2s_env = std::getenv("KDA_P2SPEC");
  const bool kP2Spec = (num_parts == 2) && (_p2s_env == nullptr || std::atoi(_p2s_env) != 0);
  if (!kP2Spec) {
    LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
               seq_order_ptr, initial_state_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
               /*use_initial_state=*/0, /*store_final_state=*/0, scale, lower_bound, num_parts,
               map_state_ptr, /*split_gamma_ptr=*/nullptr, /*fixup_mode=*/2,
               /*grid_y=*/num_parts, stream,
               /*progress_flags=*/kDeadMap ? dead_flags_ptr : nullptr, map_state_bf16_ptr, ft_slab);
  }
  // (3+4) Round-48 fused scan/correction: a single launch whose grid.y =
  // 32 scan bands + (P-1) correction parts. The scan role publishes each
  // part's inclusive carry plus a release-ordered band counter; correction
  // CTAs remap (task, y>=32) to part y-32+1 and their compute warps spin on
  // the counter while the producer warps prefetch inputs, overlapping the
  // correction prologue with the scan. (Round 73: the fused path stays
  // env-gated debug-only and is bypassed under the P=2 specialization.)
  if (kFusedAny && !kP2Spec) {
    kda_flash::CheckCuda(
        cudaMemsetAsync(lookback_flags_ptr, 0, sizeof(int) * num_tasks * num_parts, stream),
        "fused scan flags memset");
    LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
               seq_order_ptr, carry_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
               /*use_initial_state=*/1,
               // Round-121 contract fix: the fused launch's correction role is
               // the same odd-mode walk as fixup_mode==1, so it must emit the
               // final state when the caller supplied a real initial_state
               // (previously hardcoded 0 — silent contract break for any
               // caller comparing the returned initial_state tensor).
               /*store_final_state=*/(use_initial_state != 0) ? 1 : 0, scale, lower_bound,
               num_parts, split_state_ptr,
               /*split_gamma_ptr=*/map_state_ptr, /*fixup_mode=*/3,
               /*grid_y=*/32 + num_parts - 1, stream, reinterpret_cast<int*>(lookback_flags_ptr),
               /*map_state_bf16_ptr=*/nullptr, ft_slab);
  } else {
    // (3) Dense scan: c_0 = initial_state, c_{p+1} = S_p + M_p x c_p; emits the
    // fp32 per-part prefix states consumed by the correction pass.
    // Round-73 P=2 specialization: at P == 2 the only emitted carry is c_1 =
    // S_0, so the whole scan launch collapses to the exact fp32 S_0->carry copy
    // (bit-identical to the bf16 register-carry scan's c_1 emission, which
    // copies split_state slot 0 into the register carry and stores it before
    // the discarded M_1 matvec). The copy also skips the map-pass M-panel
    // export above, eliminating both middle passes.
    if (kP2Spec) {
      kernel_flashkda_split_carry_copy_m128_slab<<<dim3(static_cast<uint32_t>(num_tasks)), 256, 0,
                                                   stream>>>(
          reinterpret_cast<const float*>(split_state_ptr), reinterpret_cast<float*>(carry_ptr),
          reinterpret_cast<const int*>(seq_order_ptr), static_cast<int>(num_heads));
      CheckCuda(cudaGetLastError(), "kernel_flashkda_split_carry_copy_m128_slab launch");
    } else if (kLookbackOn) {
      // Flags must start at 0 for every call (workspace buffer is reused).
      kda_flash::CheckCuda(
          cudaMemsetAsync(lookback_flags_ptr, 0, sizeof(int) * num_tasks * num_parts, stream),
          "lookback flags memset");
      kernel_flashkda_split_lookback_m128<<<dim3(static_cast<uint32_t>(num_tasks),
                                                 static_cast<uint32_t>(num_parts - 1)),
                                            256, LOOKBACK_SMEM_BYTES, stream>>>(
          reinterpret_cast<const float*>(split_state_ptr),
          reinterpret_cast<const float*>(map_state_ptr), reinterpret_cast<float*>(carry_ptr),
          reinterpret_cast<int*>(lookback_flags_ptr), reinterpret_cast<const int*>(seq_order_ptr),
          static_cast<int>(num_heads), static_cast<int>(num_parts));
      CheckCuda(cudaGetLastError(), "kernel_flashkda_split_lookback_m128 launch");
    } else {
      // Round-50 A/B matrix (math and fp32 accumulation order identical):
      //   KDA_SCAN64=1   -> ROWS=2, grid.y=64 (measured neutral at H1, regressive
      //                     at H4/H8; debug only, never default)
      //   KDA_SCAN_BULK=1-> cp.async.bulk M staging + mbarrier tracking (the NCU
      //                     warp-stall dominant: staging STS.128 scoreboard waits)
      // Default stays the R45 v2 <4, 0> 32 four-row LDG schedule.
      const char* _s64_env = std::getenv("KDA_SCAN64");
      const bool kScan64 = (_s64_env != nullptr) && (std::atoi(_s64_env) != 0);
      const char* _sb_env = std::getenv("KDA_SCAN_BULK");
      // BULK is the default since round 50 (scan -50% across all split shapes,
      // bit-exact); KDA_SCAN_BULK=0 falls back to the R45 LDG staging.
      const bool kScanBulk = (_sb_env == nullptr) || (std::atoi(_sb_env) != 0);
      // Round-51 bf16 register-carry scan is DEFAULT ON (supervisor directive
      // implementation): the map pass additionally exports M_p rounded to bf16
      // and the scan stages 32KB bf16 panels (half the BULK bytes, half the
      // matvec LDS traffic) with the carry kept in REGISTERS (no SMEM c panels,
      // one shfl broadcast per step); 128-thread CTAs, 65552B SMEM -> 3 CTAs/SM.
      // Requires the map pass launch above to have received a non-null
      // map_state_bf16 buffer (env-checked by kernel.py). KDA_SCAN_BF16=0 opts
      // back to the fp32 BULK scan. (Round 118: the predicate and env read are
      // hoisted above as kScanBf16Default for the dead-map gate; semantics
      // unchanged.)
      const bool kScanBf16 = kScanBf16Default;
      if (kScanBf16) {
        kernel_flashkda_split_scan_bf16_m128<<<dim3(static_cast<uint32_t>(num_tasks), 32), 128,
                                               65552, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(initial_state_ptr),
            reinterpret_cast<const float*>(split_state_ptr),
            reinterpret_cast<const __nv_bfloat16*>(map_state_bf16_ptr),
            reinterpret_cast<float*>(carry_ptr), reinterpret_cast<const int*>(seq_order_ptr),
            static_cast<int>(num_heads), static_cast<int>(num_parts),
            static_cast<int>(use_initial_state), kDeadMap ? dead_flags_ptr : nullptr);
        CheckCuda(cudaGetLastError(), "kernel_flashkda_split_scan_bf16_m128 launch");
      } else {
        const int grid_bands = kScan64 ? 64 : 32;
        const int smem_bytes =
            kScanBulk ? (kScan64 ? 134256 : 137424) : (kScan64 ? 138336 : 141504);
        auto* scan_kernel = kScanBulk ? (kScan64 ? kernel_flashkda_split_scan_m128<2, 1>
                                                 : kernel_flashkda_split_scan_m128<4, 1>)
                                      : (kScan64 ? kernel_flashkda_split_scan_m128<2, 0>
                                                 : kernel_flashkda_split_scan_m128<4, 0>);
        scan_kernel<<<dim3(static_cast<uint32_t>(num_tasks), static_cast<uint32_t>(grid_bands)),
                      256, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(initial_state_ptr),
            reinterpret_cast<const float*>(split_state_ptr),
            reinterpret_cast<const float*>(map_state_ptr), reinterpret_cast<float*>(carry_ptr),
            reinterpret_cast<const int*>(seq_order_ptr), static_cast<int>(num_heads),
            static_cast<int>(num_parts), static_cast<int>(use_initial_state));
        CheckCuda(cudaGetLastError(), "kernel_flashkda_split_scan_m128 launch");
      }
    }
    // (4) Correction pass: parts [1, P), V := 0, scanned carry as initial state.
    // Full chunks TMA-store the exact carry contribution into the split-out
    // scratch buffer (out2_tma); partial tail chunks accumulate into out
    // directly (scalar path), and the add-out kernel below folds the scratch
    // contribution into out over the full-chunk rows of parts [1, P). Run in
    // every non-fused launch — which under the P=2 specialization includes the
    // carry-copy path even if KDA_FUSED_SCAN is set.
    {
      LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
                 seq_order_ptr, carry_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
                 // Round-79 contract correction: the last correction part emits the
                 // exact serial final state (M_{P-1} * c_{P-1} + S_{P-1}) in place
                 // into the caller's initial_state buffer (gated in the kernel;
                 // only safe when the caller supplied a real initial_state).
                 /*use_initial_state=*/1,
                 /*store_final_state=*/(use_initial_state != 0) ? 1 : 0, scale, lower_bound,
                 num_parts, split_state_ptr, /*split_gamma_ptr=*/nullptr, /*fixup_mode=*/1,
                 /*grid_y=*/num_parts - 1, stream, /*progress_flags=*/nullptr,
                 /*map_state_bf16_ptr=*/nullptr, ft_slab);
    }
  }
  {
    // Fixed layout only (split gate): uniform tokens per sequence.
    const int64_t tokens_per_seq = token_count / num_seqs;
    const int64_t total_vec = token_count * num_heads * 16;
    const int threads = 256;
    const int gridsize =
        static_cast<int>(std::min<int64_t>((total_vec + threads - 1) / threads, 65535LL * 8));
    if (gridsize > 0) {
      kernel_flashkda_split_out_add_m128<<<gridsize, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(split_out_ptr),
          reinterpret_cast<__nv_bfloat16*>(out_ptr), tokens_per_seq, static_cast<int>(num_seqs),
          static_cast<int>(num_parts), static_cast<int>(num_heads));
      CheckCuda(cudaGetLastError(), "kernel_flashkda_split_out_add_m128 launch");
    }
  }
}

}  // namespace kda_flash_slab

// End-of-TU cleanup: release the kernel-name renames (see include block).
#undef kernel_flashkda_bf16_fused_m128
#undef kernel_flashkda_split_scan_m128
#undef kernel_flashkda_split_scan_bf16_m128
#undef kernel_flashkda_split_lookback_m128
#undef kernel_flashkda_split_out_add_m128
