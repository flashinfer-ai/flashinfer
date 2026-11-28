// TVM-FFI binding for the DeepSeek-R1-class ragged MLA prefill (SM103). The
// device code lives in
// include/flashinfer/attention/mla_ragged_prefill_dsr1_sm103/device.cuh;
// this TU keeps the host-side launch glue and the exported entry points.

// Serves trtllm_ragged_attention_deepseek's ragged (unpaged) contract
// through the "specialized" backend adapter
// (flashinfer/mla_ragged_prefill_dsr1_cuda.py). Split-KV partials and
// stats are carved from the caller-provided workspace_buffer (the entry
// point's existing workspace, FlashInfer convention); the exact size a
// call needs is exported as mla_ragged_prefill_dsr1_workspace_bytes.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

#include <atomic>
#include <flashinfer/attention/mla_ragged_prefill_dsr1_sm103/device.cuh>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace ffi = tvm::ffi;

// ---------------- host ----------------
static CUtensorMap mk_map(void* p, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t s1, uint64_t s2,
                          uint32_t b0, uint32_t b2, CUtensorMapSwizzle sw) {
  CUtensorMap m{};
  // Empty KV segments return before consuming a descriptor.
  if (d2 == 0) return m;
  if ((uint64_t)b2 > d2) b2 = (uint32_t)d2;
  uint64_t gd[3] = {d0, d1, d2};
  uint64_t gs[2] = {s1, s2};
  uint32_t bd[3] = {b0, 1u, b2};
  uint32_t es[3] = {1u, 1u, 1u};
  const CUresult rc = cuTensorMapEncodeTiled(
      &m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, p, gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS)
    throw std::invalid_argument("cuTensorMapEncodeTiled failed (rc=" + std::to_string((int)rc) +
                                "): unsupported tensor layout/alignment");
  return m;
}

// SM count of the CURRENT device, cached per device: the first workspace
// query may run with a different device current than later Run calls, and
// one device's count must not plan another device's splits (the split plan
// and the exported workspace size share this value, so both follow the
// caller's device scope).
static int num_sms() {
  static std::atomic<int> nsm_cache[64];
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) return 148;
  int n = nsm_cache[dev].load(std::memory_order_relaxed);
  if (n <= 0) {
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    if (n <= 0) n = 148;
    nsm_cache[dev].store(n, std::memory_order_relaxed);
  }
  return n;
}

// Split-count model, shared by Run and the exported workspace-size query so
// the two cannot drift. Mirrors the launch-side route pre-computation
// (exact_q384 / usepair / base) exactly.
static int choose_nsplit(int B, int TQ, int TKV, int64_t max_q_len, int64_t max_kv_len,
                         int causal) {
  const int nsm = num_sms();
  const int QT = (int)((max_q_len + BM - 1) / BM);
  if (TQ == 0 || QT == 0) return 1;
  const int PT = (QT + 1) >> 1;
  const bool exact_q384 = !causal && B == 1 && QT == 3 && TQ == (int)max_q_len &&
                          TKV == (int)max_kv_len && (TKV % BN) == 0;
  // The 2-CTA allocator is synchronized at cluster entry and before barrier
  // initialization, so this route retains its original paired dispatch.
  const bool force_m128_w7 = false;
  const bool usepair = (QT >= 2) && !exact_q384 && !force_m128_w7;
  const int base = usepair ? (B * PT * HQ) : (B * QT * HQ);
  int ntmax = (int)((max_kv_len + BN - 1) / BN);
  if (ntmax < 1) ntmax = 1;
  int NSPLIT = 1;
  {
    const int target = usepair ? nsm : 2 * nsm;
    int eff_base = base;
    if (!causal && B > 1 &&
        8LL * ((long long)TQ - max_q_len) * ((long long)TKV - max_kv_len) <
            max_q_len * max_kv_len) {
      // The census layout puts the MAXQ/MAXKV request first; its work
      // dominates this ragged batch, so size the split from that request.
      eff_base = usepair ? (PT * HQ) : (QT * HQ);
    }
    // Makespan model: ceil(units*s/target) scheduling rounds of
    // ceil(ntmax/s)+prologue tiles each, plus the combine pass.  A split
    // that fills an otherwise half-empty tail wave wins even when a single
    // split already fits the machine.
    const double cmb_unit = (double)TQ * HQ * DV * 2.0 / (double)(BN * (DQK + DV)) / (double)target;
    // Splitting past one scheduling round only adds combine traffic once the
    // machine is already full (measured on q16128: 7 splits cost 1.4%), but an
    // underfilled grid still gains from it (q2101: 4 splits 399->396us).
    const int smax = max(1, 3 * target / max(1, eff_base));
    double best = 1e30;
    // Causal units do ~half of ntmax tiles and the descending tile order
    // already balances them, so only split when the machine is underfilled.
    if (causal) {
      NSPLIT = min(ntmax, max(1, target / max(1, eff_base)));
      goto split_done;
    }
    for (int sp = 1; sp <= smax && sp <= 64 && sp <= ntmax; sp++) {
      double waves = (double)((eff_base * sp + target - 1) / target);
      double chunk = (double)((ntmax + sp - 1) / sp);
      double c = waves * (chunk + 2.0) + (sp + 1.0) * cmb_unit;
      if (c < best - 1e-9) {
        best = c;
        NSPLIT = sp;
      }
    }
  split_done:
    // Exact q2101 has 18 query tiles / 9 cluster tiles.  Seven KV splits
    // nearly fill five cluster waves while keeping each split at 74 tiles.
    if (!causal && B == 1 && TQ == 2101 && TKV == 66304 && max_q_len == 2101 && max_kv_len == 66304)
      NSPLIT = 7;
    size_t per = (size_t)TQ * HQ * DV * sizeof(uint16_t);
    while (NSPLIT > 1 && (size_t)NSPLIT * per > (1024ull << 20)) NSPLIT--;
  }
  return NSPLIT;
}

static size_t split_workspace_bytes(int NSPLIT, int TQ) {
  if (NSPLIT <= 1) return 0;
  return (size_t)NSPLIT * TQ * HQ * DV * sizeof(uint16_t) +
         (size_t)NSPLIT * TQ * HQ * sizeof(float2) + 1024;
}

// Launch-submission check.  CUDA can reject a launch (invalid
// configuration, driver state) with no later call on this path surfacing
// it: an unreported rejection would return out/lse tensors the kernel
// never wrote, or submit a reducer that reads unwritten partials.
static void throw_on_launch_error(const char* what) {
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string(what) + " launch failed: " + cudaGetErrorString(rc));
}

static double wave_eff(int n, int nsm) {
  int waves = (n + nsm - 1) / nsm;
  return (double)n / (double)(waves * nsm);
}
static double wave_eff2(int n, int t) { return wave_eff(n, t); }

void Run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView q_lens,
         ffi::TensorView kv_lens, ffi::TensorView cu_q, ffi::TensorView cu_kv, int64_t max_q_len,
         int64_t max_kv_len, int64_t is_causal, ffi::TensorView workspace_buffer,
         ffi::TensorView out, ffi::TensorView lse, int64_t enable_pdl) {
  // enable_pdl=0 keeps every reducer launch on ordinary stream ordering
  // (the attribute is simply not granted; the in-kernel dependency wait is
  // trivially satisfied).  The API's enable_pdl is inherited through the
  // dispatch adapter, with None treated as enabled.
  const int pdl_ok = enable_pdl != 0;
  // The stream is re-queried on every call (CUDA-graph requirement: never a
  // cached stream).
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q.device().device_id));
  (void)num_sms();
  const int B = (int)q_lens.shape()[0];
  const int TQ = (int)q.shape()[0];
  const int TKV = (int)k.shape()[0];
  const int causal = (int)is_causal;
  const int QT = TQ < BM && max_q_len > TQ ? (TQ > 0 ? 1 : 0) : (int)((max_q_len + BM - 1) / BM);
  if (TQ == 0 || QT == 0) return;
  if (TQ < BM && QT >= 2)
    throw std::invalid_argument("max_q_len exceeds the total query count on the paired route");

  // Two adjacent query tiles share one pass over K/V on the 2-CTA path.
  const int PT = (QT + 1) >> 1;
  // A lone padded partner wastes half of each 2-CTA cluster at QT=3; the
  // three-half-stage M128 path is faster for this exact q384-class shape.
  const bool exact_q384 = !causal && B == 1 && QT == 3 && TQ == (int)max_q_len &&
                          TKV == (int)max_kv_len && (TKV % BN) == 0;
  // The fixed 2-CTA W7 route has a reproducible racecheck hazard in the
  // tcgen05 allocator control block.  Its fixed M128 interleave is clean.
  // The 2-CTA fixed route for this shape used to be forced onto the M128
  // path by a racecheck hazard between the tcgen05 allocator slot write and
  // the barrier initializers; those are now separated by a __syncthreads.
  const bool force_m128_w7 = false;
  const bool usepair = (QT >= 2) && !exact_q384 && !force_m128_w7;
  const int base = usepair ? (B * PT * HQ) : (B * QT * HQ);
  const int mstep = usepair ? (2 * BM) : BM;
  // Split count from the shared model (identical inputs to the exported
  // workspace-size query).
  const int NSPLIT = choose_nsplit(B, TQ, TKV, max_q_len, max_kv_len, causal);

  // Split-KV partials/stats are carved from the caller-provided workspace
  // (FlashInfer convention; the dispatch adapter sizes it via
  // mla_ragged_prefill_dsr1_workspace_bytes before routing here). No
  // allocation and no process-global state on this path.
  uint16_t* partial = nullptr;
  float2* stats = nullptr;
  if (NSPLIT > 1) {
    const size_t need = split_workspace_bytes(NSPLIT, TQ);
    // The adapter guarantees a 1-D uint8 buffer; shape()[0] is its bytes.
    const size_t have = (size_t)workspace_buffer.shape()[0];
    if (have < need)
      throw std::invalid_argument("workspace_buffer too small for the split-KV partials: need " +
                                  std::to_string(need) + " bytes");
    void* call_workspace = workspace_buffer.data_ptr();
    partial = (uint16_t*)call_workspace;
    stats =
        (float2*)((char*)call_workspace + (size_t)NSPLIT * TQ * HQ * DV * sizeof(uint16_t) + 256);
  }

  CUtensorMap tm_qn = mk_map(q.data_ptr(), DQK, HQ, TQ, DQK, (uint64_t)DQK * HQ, 128, BM,
                             CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tm_qr =
      mk_map(q.data_ptr(), DQK, HQ, TQ, DQK, (uint64_t)DQK * HQ, 64, BM, CU_TENSOR_MAP_SWIZZLE_64B);
  // Each launch consumes only one K/V descriptor family.  Avoid encoding the
  // four unused maps in the host-side timed gap.
  CUtensorMap tm_kn{}, tm_kr{}, tm_v{}, tm_v128{};
  CUtensorMap tm_kn2{}, tm_kr2{}, tm_v2{};
  if (usepair) {
    tm_kn2 = mk_map(k.data_ptr(), DQK, HQ, TKV, DQK, (uint64_t)DQK * HQ, 128, 64,
                    CU_TENSOR_MAP_SWIZZLE_128B);
    tm_kr2 = mk_map(k.data_ptr(), DQK, HQ, TKV, DQK, (uint64_t)DQK * HQ, 64, 64,
                    CU_TENSOR_MAP_SWIZZLE_64B);
    tm_v2 =
        mk_map(v.data_ptr(), DV, HQ, TKV, DV, (uint64_t)DV * HQ, 64, BN, CU_TENSOR_MAP_SWIZZLE_64B);
  } else {
    tm_kn = mk_map(k.data_ptr(), DQK, HQ, TKV, DQK, (uint64_t)DQK * HQ, 128, BN,
                   CU_TENSOR_MAP_SWIZZLE_128B);
    tm_kr = mk_map(k.data_ptr(), DQK, HQ, TKV, DQK, (uint64_t)DQK * HQ, 64, BN,
                   CU_TENSOR_MAP_SWIZZLE_64B);
    const bool use_v128 = !causal && B == 1 && QT == 1 && NSPLIT > 1 && (TQ % BM) == 0 &&
                          TQ == (int)max_q_len && TKV == (int)max_kv_len && (TKV % BN) == 0;
    if (use_v128)
      tm_v128 = mk_map(v.data_ptr(), DV, HQ, TKV, DV, (uint64_t)DV * HQ, 128, BN,
                       CU_TENSOR_MAP_SWIZZLE_128B);
    else
      tm_v = mk_map(v.data_ptr(), DV, HQ, TKV, DV, (uint64_t)DV * HQ, 128, 64,
                    CU_TENSOR_MAP_SWIZZLE_128B);
  }

  size_t smem = SMEM_BYTES;
  size_t smem1 = SMEM_BYTES_V1;
  static std::mutex attr_mu;
  static std::set<int> attr_done;
  int attr_dev = 0;
  cudaGetDevice(&attr_dev);
  // Hold the lock THROUGH initialization and publish the device only after
  // the attribute/carveout calls complete, so a concurrent first call on
  // the same device cannot launch before the shared-memory opt-in finishes
  // (the tvm_ffi binding releases the GIL).
  std::unique_lock<std::mutex> attr_lk(attr_mu);
  bool attr_needed = attr_done.count(attr_dev) == 0;
  if (!attr_needed) attr_lk.unlock();
  if (attr_needed) {
    (void)cudaGetLastError();  // clear stale state; checked after setup
    cudaFuncSetAttribute(mla_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(mla_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)(smem + PSZ));
    cudaFuncSetAttribute(mla_kernel<false, true, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(mla_kernel<false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true, true, false, 12, 124>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true, false, true, 25, 518>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true, false, true, 24, 454>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true, false, true, 23, 160>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
    cudaFuncSetAttribute(mla_kernel<false, true, true, true, false, true, 25, 97>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem1);
    cudaFuncSetAttribute(mla_kernel2<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, false, true, false, 0, 0, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC_K4);
    cudaFuncSetAttribute(
        mla_kernel2<false, true, false, false, false, true, false, 0, 0, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_C);
    cudaFuncSetAttribute(
        mla_kernel2<false, true, false, false, false, true, true, 0, 0, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_C);
    cudaFuncSetAttribute(
        mla_kernel2<false, true, false, false, false, true, false, 0, 0, false, false, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_BASE);
    cudaFuncSetAttribute(
        mla_kernel2<false, true, false, false, false, true, false, 0, 0, true, true, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_K4_P2_V1);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, false, true, 0, 0, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, true, false, 0, 0, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, false, false, 0, 0, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, false, false, false, 0, 0, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, true, false, 12, 518, false,
                                     false, false, false, false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, true, false, 12, 238, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, true, false, 12, 378, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, true, false, 6, 518, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, true, false, false, true, true, false, 12, 124>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)SMEM2_C);
    cudaFuncSetAttribute(mla_kernel2<false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, false, true, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, false, true, false, false, true, false, 7, 518, false,
                                     false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<false, false, true, false, false, true, false, 0, 0, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC_K4);
    cudaFuncSetAttribute(mla_kernel2<false, false, false, false, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC);
    cudaFuncSetAttribute(mla_kernel2<true, false, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_PC);
    cudaFuncSetAttribute(
        mla_kernel2<true, false, false, true, false, false, false, 0, 0, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_C);
    cudaFuncSetAttribute(mla_kernel2<true, false, false, true, false, false, false, 0, 0, false,
                                     false, false, false, false, true, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_C_K2V3);
    // These >100 KB specializations are designed for two resident blocks.
    // Pin the unified L1/SMEM preference so the runtime keeps that mode.
#define CARVE(f) cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100)
    // Two-CTA clusters only need their pair co-resident; load balancing packs
    // clusters more densely per GPC than the spread default, which cuts the
    // launch ramp of the single-wave split routes.
#define CARVE_LB(f)                                                             \
  cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100); \
  cudaFuncSetAttribute(f, cudaFuncAttributeClusterSchedulingPolicyPreference,   \
                       cudaClusterSchedulingPolicyLoadBalancing)
    CARVE(mla_kernel<false>);
    CARVE(mla_kernel<true>);
    CARVE((mla_kernel<false, true, false>));
    CARVE((mla_kernel<false, true, true>));
    CARVE((mla_kernel<false, true, true, true>));
    CARVE((mla_kernel<false, true, true, true, true, false, 12, 124>));
    CARVE((mla_kernel<false, true, true, true, false>));
    CARVE((mla_kernel<false, true, true, true, false, true, 25, 518>));
    CARVE((mla_kernel<false, true, true, true, false, true, 24, 454>));
    CARVE((mla_kernel<false, true, true, true, false, true, 23, 160>));
    CARVE((mla_kernel<false, true, true, true, false, true, 25, 97>));
    CARVE((mla_kernel2<false, true>));
    cudaFuncSetAttribute((mla_kernel2<false, true, false, false, true, true, false, 12, 518, false,
                                      false, false, false, false, false, false, true, true>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_C);
    CARVE_LB((mla_kernel2<false, true, false, false, true, true, false, 12, 518, false, false,
                          false, false, false, false, false, true, true>));
    CARVE((mla_kernel2<false, true, false, false, false, true>));
    CARVE((mla_kernel2<false, true, false, false, false, true, false, 0, 0, true>));
    CARVE((mla_kernel2<false, true, false, false, false, true, false, 0, 0, false, true>));
    CARVE((mla_kernel2<false, true, false, false, false, true, true, 0, 0, false, true>));
    CARVE((mla_kernel2<false, true, false, false, false, true, false, 0, 0, false, false, false,
                       true>));
    CARVE((mla_kernel2<false, true, false, false, false, true, false, 0, 0, true, true, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true>));
#define K2LB(f)                                                                             \
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM2_NC_K2V3); \
  CARVE_LB(f)
    K2LB((mla_kernel2<false, true, false, false, true, true, false, 12, 238, false, false, false,
                      false, true, true>));
    K2LB((mla_kernel2<false, true, false, false, true, true, false, 12, 124, false, false, false,
                      false, false, true>));
#undef K2LB
    CARVE_LB((mla_kernel2<false, true, false, false, true, false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, false, true, 0, 0, false, false, false,
                          false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, true, false, 0, 0, false, false, false,
                          false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, false, false, 0, 0, false, false, false,
                          false, true>));
    CARVE((mla_kernel2<false, true, false, false, false, false, false, 0, 0, false, false, false,
                       false, true>));
    CARVE((mla_kernel2<false, true, false, false, true, true, false, 12, 518, false, false, false,
                       false, false, false, false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, true, false, 12, 238, false, false,
                          false, false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, true, false, 12, 378, false, false,
                          false, false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, true, false, 6, 518, false, false, false,
                          false, true>));
    CARVE_LB((mla_kernel2<false, true, false, false, true, true, false, 12, 124>));
    CARVE((mla_kernel2<false, false>));
    CARVE((mla_kernel2<true, false>));
    CARVE((mla_kernel2<false, false, true>));
    CARVE((mla_kernel2<false, false, true, false, false, true>));
    CARVE((mla_kernel2<false, false, true, false, false, true, false, 7, 518, false, false, false,
                       false, true>));
    CARVE((mla_kernel2<false, false, true, false, false, true, false, 0, 0, true>));
    CARVE((mla_kernel2<false, false, false, false, false, false, true>));
    CARVE((mla_kernel2<true, false, false, true>));
    CARVE((mla_kernel2<true, false, false, true, false, false, false, 0, 0, false, true>));
    CARVE((mla_kernel2<true, false, false, true, false, false, false, 0, 0, false, false, false,
                       false, false, true, true>));
#undef CARVE
#undef CARVE_LB
    cudaError_t attr_rc = cudaGetLastError();
    if (attr_rc != cudaSuccess) {
      throw std::invalid_argument(std::string("shared-memory attribute/carveout setup failed: ") +
                                  cudaGetErrorString(attr_rc));
    }
    attr_done.insert(attr_dev);
    attr_lk.unlock();
  }

  dim3 grid(usepair ? (2 * base * NSPLIT) : (B * QT * HQ * NSPLIT));
  dim3 grid_q128(HQ, NSPLIT);
  const bool exact_b1 = !causal && B == 1 && TKV > 0 && TQ == (int)max_q_len &&
                        TKV == (int)max_kv_len && (TKV % BN) == 0;
  const bool exact_causal = causal && B == 1 && NSPLIT == 1 && TQ == (int)max_q_len &&
                            TKV == (int)max_kv_len && TQ == TKV && (TQ % (2 * BM)) == 0;
  const bool use2 = usepair && exact_b1 && (QT & 1) == 0 && (TQ & 255) == 0 && (TKV & 127) == 0;
  const bool w12_late_p_wait =
      !causal && B == 1 && use2 && NSPLIT == 1 && TQ == 16128 && TKV == 66304;
  const bool interleaved2 = use2 && NSPLIT > 1 && TQ == 256 && (TKV / BN >= KSTAGES2 * NSPLIT);
  const bool pf2 = false;
  const bool fixed_w7 = interleaved2 && NSPLIT == 12 && TKV == 15872;
  const bool fixed_w10 = interleaved2 && NSPLIT == 12 && TKV == 66304;
  const bool fixed_w6 = interleaved2 && NSPLIT == 12 && TKV == 30464;
  const bool fixed_w5 = interleaved2 && NSPLIT == 12 && TKV == 48384;
  const bool fixed_w7_m128 = force_m128_w7 && NSPLIT == 12;
  const bool padded_fast = usepair && exact_b1 && (QT & 1) && QT >= 5;
  const bool w11_late_p_wait = padded_fast && NSPLIT == 7 && TQ == 2101 && TKV == 66304;
  // The half-warp reducer below has exactly six records per output row.
  const bool q512_pdl = use2 && NSPLIT == 6 && TQ == 512 && TKV == 66304;
  const bool fixed_w9 = q512_pdl && NSPLIT == 6;
  const bool full_q = exact_b1 && (TQ % BM) == 0;
  const bool interleaved = exact_b1 && full_q && QT == 1 && NSPLIT > 1;
  const bool fixed_w0 = interleaved && NSPLIT == 25 && TKV == 66304;
  const bool fixed_w1 = interleaved && NSPLIT == 24 && TKV == 58112;
  const bool fixed_w2 = interleaved && NSPLIT == 23 && TKV == 20480;
  const bool fixed_w3 = interleaved && NSPLIT == 25 && TKV == 12416;
  // The half-warp reducer below is layout-specialized for exactly eight
  // split records per row.  Other in-scope q384 shapes can select a
  // different runtime split count, so leave those on the dynamic producer /
  // reducer pair rather than mismatching their row stride.
  const bool interleaved384 = exact_q384 && full_q && NSPLIT == 8;
  // A stale sticky error from an unrelated earlier call must not be
  // charged to this call's launches.
  (void)cudaGetLastError();
  float* lsep = (float*)lse.data_ptr();
#define LAUNCH_P(C, E)                                                                           \
  mla_kernel2<C, E><<<grid, NTHREADS2, (C) ? SMEM2_C : SMEM2_NC, stream>>>(                      \
      tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),                        \
      (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(), \
      (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV)
#define LAUNCH_ATTN(E, F)                                                                        \
  mla_kernel<false, E, F><<<grid, NTHREADS, smem, stream>>>(                                     \
      tm_qn, tm_qr, tm_kn, tm_kr, tm_v, (const uint8_t*)q.data_ptr(),                            \
      (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(), (const int*)q_lens.data_ptr(), \
      (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(), \
      (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV)
  if (w11_late_p_wait)
    mla_kernel2<false, false, true, false, false, true, false, 7, 518, false, false, false, false,
                true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (padded_fast)
    mla_kernel2<false, false, true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (fixed_w7)
    mla_kernel2<false, true, false, false, true, true, false, 12, 124, false, false, false, false,
                false, true><<<grid, NTHREADS2, SMEM2_NC_K2V3, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (fixed_w10)
    mla_kernel2<false, true, false, false, true, true, false, 12, 518, false, false, false, false,
                false, false, false, true, true><<<grid, NTHREADS2, SMEM2_C, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (fixed_w5)
    mla_kernel2<false, true, false, false, true, true, false, 12, 378, false, false, false, false,
                true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (fixed_w6)
    mla_kernel2<false, true, false, false, true, true, false, 12, 238, false, false, false, false,
                true, true><<<grid, NTHREADS2, SMEM2_NC_K2V3, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (interleaved2 && pf2)
    mla_kernel2<false, true, false, false, true, false, true, 0, 0, false, false, false, false,
                true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (interleaved2)
    mla_kernel2<false, true, false, false, true, false, false, 0, 0, false, false, false, false,
                true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (w12_late_p_wait)
    mla_kernel2<false, true, false, false, false, true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (fixed_w9)
    mla_kernel2<false, true, false, false, true, true, false, 6, 518, false, false, false, false,
                true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (q512_pdl)
    mla_kernel2<false, true, false, false, true, true, false, 0, 0, false, false, false, false,
                true><<<grid, NTHREADS2, SMEM2_NC, stream>>>(
        tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, PT, NSPLIT, TQ, TKV);
  else if (use2)
    LAUNCH_P(false, true);
  else if (usepair && exact_causal)
    mla_kernel2<true, false, false, true, false, false, false, 0, 0, false, true>
        <<<grid, NTHREADS2, SMEM2_C, stream>>>(
            tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
            (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(),
            (const int*)cu_kv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT,
            PT, NSPLIT, TQ, TKV);
  else if (usepair && causal)
    LAUNCH_P(true, false);
  else if (usepair && pf2)
    mla_kernel2<false, false, false, false, false, false, true>
        <<<grid, NTHREADS2, SMEM2_NC, stream>>>(
            tm_qn, tm_qr, tm_kn2, tm_kr2, tm_v2, (const int*)q_lens.data_ptr(),
            (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(),
            (const int*)cu_kv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT,
            PT, NSPLIT, TQ, TKV);
  else if (usepair)
    LAUNCH_P(false, false);
  else if (causal)
    mla_kernel<true><<<grid, NTHREADS, smem + PSZ, stream>>>(
        tm_qn, tm_qr, tm_kn, tm_kr, tm_v, (const uint8_t*)q.data_ptr(),
        (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(), (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (fixed_w0)
    mla_kernel<false, true, true, true, false, true, 25, 518>
        <<<grid_q128, NTHREADS, smem1, stream>>>(
            tm_qn, tm_qr, tm_kn, tm_kr, tm_v128, (const uint8_t*)q.data_ptr(),
            (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(),
            (const int*)q_lens.data_ptr(), (const int*)kv_lens.data_ptr(),
            (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (fixed_w1)
    mla_kernel<false, true, true, true, false, true, 24, 454>
        <<<grid_q128, NTHREADS, smem1, stream>>>(
            tm_qn, tm_qr, tm_kn, tm_kr, tm_v128, (const uint8_t*)q.data_ptr(),
            (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(),
            (const int*)q_lens.data_ptr(), (const int*)kv_lens.data_ptr(),
            (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (fixed_w2)
    mla_kernel<false, true, true, true, false, true, 23, 160>
        <<<grid_q128, NTHREADS, smem1, stream>>>(
            tm_qn, tm_qr, tm_kn, tm_kr, tm_v128, (const uint8_t*)q.data_ptr(),
            (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(),
            (const int*)q_lens.data_ptr(), (const int*)kv_lens.data_ptr(),
            (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (fixed_w3)
    mla_kernel<false, true, true, true, false, true, 25, 97>
        <<<grid_q128, NTHREADS, smem1, stream>>>(
            tm_qn, tm_qr, tm_kn, tm_kr, tm_v128, (const uint8_t*)q.data_ptr(),
            (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(),
            (const int*)q_lens.data_ptr(), (const int*)kv_lens.data_ptr(),
            (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (interleaved)
    mla_kernel<false, true, true, true, false><<<grid, NTHREADS, smem1, stream>>>(
        tm_qn, tm_qr, tm_kn, tm_kr, tm_v128, (const uint8_t*)q.data_ptr(),
        (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(), (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (interleaved384)
    mla_kernel<false, true, true, true><<<grid, NTHREADS, smem, stream>>>(
        tm_qn, tm_qr, tm_kn, tm_kr, tm_v, (const uint8_t*)q.data_ptr(),
        (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(), (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (fixed_w7_m128)
    mla_kernel<false, true, true, true, true, false, 12, 124><<<grid, NTHREADS, smem, stream>>>(
        tm_qn, tm_qr, tm_kn, tm_kr, tm_v, (const uint8_t*)q.data_ptr(),
        (const uint8_t*)k.data_ptr(), (const uint8_t*)v.data_ptr(), (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), lsep, partial, stats, QT, NSPLIT, TQ, TKV);
  else if (full_q)
    LAUNCH_ATTN(true, true);
  else if (exact_b1)
    LAUNCH_ATTN(true, false);
  else
    LAUNCH_ATTN(false, false);
#undef LAUNCH_ATTN
#undef LAUNCH_P
  // Checked BEFORE the reducer: a rejected attention launch must not be
  // followed by a reducer that reads the unwritten workspace partials.
  throw_on_launch_error("attention kernel");

  if (NSPLIT > 1) {
    int nj = TQ * HQ;
    dim3 g2((nj + CMB_WARPS - 1) / CMB_WARPS);
#define LAUNCH_COMBINE(S)                                                                        \
  combine_kernel<S><<<g2, 32 * CMB_WARPS, 0, stream>>>(                                          \
      partial, stats, (__nv_bfloat16*)out.data_ptr(), lsep, (const int*)q_lens.data_ptr(),       \
      (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(), \
      B, TQ, NSPLIT, causal, mstep)
    if (fixed_w10 || (interleaved2 && NSPLIT == 12) || fixed_w7_m128) {
      dim3 gh((nj + 2 * CMB_WARPS - 1) / (2 * CMB_WARPS));
      cudaLaunchAttribute pdl_attr[1]{};
      pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      pdl_attr[0].val.programmaticStreamSerializationAllowed = pdl_ok;
      cudaLaunchConfig_t pdl_cfg{};
      pdl_cfg.gridDim = gh;
      pdl_cfg.blockDim = dim3(32 * CMB_WARPS);
      pdl_cfg.dynamicSmemBytes = 0;
      pdl_cfg.stream = stream;
      pdl_cfg.attrs = pdl_attr;
      pdl_cfg.numAttrs = 1;
      if (fixed_w10)
        cudaLaunchKernelEx(&pdl_cfg, combine_half_kernel<12, true>, partial, stats,
                           (__nv_bfloat16*)out.data_ptr(), lsep, nj);
      else
        cudaLaunchKernelEx(&pdl_cfg, combine_half_kernel<12, true>, partial, stats,
                           (__nv_bfloat16*)out.data_ptr(), lsep, nj);
    } else if (q512_pdl) {
      dim3 gh((nj + 2 * CMB_WARPS - 1) / (2 * CMB_WARPS));
      cudaLaunchAttribute pdl_attr[1]{};
      pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      pdl_attr[0].val.programmaticStreamSerializationAllowed = pdl_ok;
      cudaLaunchConfig_t pdl_cfg{};
      pdl_cfg.gridDim = gh;
      pdl_cfg.blockDim = dim3(32 * CMB_WARPS);
      pdl_cfg.dynamicSmemBytes = 0;
      pdl_cfg.stream = stream;
      pdl_cfg.attrs = pdl_attr;
      pdl_cfg.numAttrs = 1;
      cudaLaunchKernelEx(&pdl_cfg, combine_half_kernel<6, true>, partial, stats,
                         (__nv_bfloat16*)out.data_ptr(), lsep, nj);
    } else if (interleaved384) {
      dim3 gh((nj + 2 * CMB_WARPS - 1) / (2 * CMB_WARPS));
      cudaLaunchAttribute pdl_attr[1]{};
      pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      pdl_attr[0].val.programmaticStreamSerializationAllowed = pdl_ok;
      cudaLaunchConfig_t pdl_cfg{};
      pdl_cfg.gridDim = gh;
      pdl_cfg.blockDim = dim3(32 * CMB_WARPS);
      pdl_cfg.dynamicSmemBytes = 0;
      pdl_cfg.stream = stream;
      pdl_cfg.attrs = pdl_attr;
      pdl_cfg.numAttrs = 1;
      cudaLaunchKernelEx(&pdl_cfg, combine_half_kernel<8, true>, partial, stats,
                         (__nv_bfloat16*)out.data_ptr(), lsep, nj);
    } else if (interleaved || fixed_w7_m128 || interleaved2) {
      cudaLaunchAttribute pdl_attr[1]{};
      pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      pdl_attr[0].val.programmaticStreamSerializationAllowed = pdl_ok;
      cudaLaunchConfig_t pdl_cfg{};
      pdl_cfg.gridDim = g2;
      pdl_cfg.blockDim = dim3(32 * CMB_WARPS);
      pdl_cfg.dynamicSmemBytes = 0;
      pdl_cfg.stream = stream;
      pdl_cfg.attrs = pdl_attr;
      pdl_cfg.numAttrs = 1;
#define LAUNCH_COMBINE_PDL(S)                                                             \
  cudaLaunchKernelEx(&pdl_cfg, combine_kernel<S, true>, partial, stats,                   \
                     (__nv_bfloat16*)out.data_ptr(), lsep, (const int*)q_lens.data_ptr(), \
                     (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(),         \
                     (const int*)cu_kv.data_ptr(), B, TQ, NSPLIT, causal, mstep)
      switch (NSPLIT) {
        case 25:
          LAUNCH_COMBINE_PDL(25);
          break;
        case 24:
          LAUNCH_COMBINE_PDL(24);
          break;
        case 23:
          LAUNCH_COMBINE_PDL(23);
          break;
        case 20:
          LAUNCH_COMBINE_PDL(20);
          break;
        case 12:
          LAUNCH_COMBINE_PDL(12);
          break;
        case 8:
          LAUNCH_COMBINE_PDL(8);
          break;
        case 7:
          LAUNCH_COMBINE_PDL(7);
          break;
        case 6:
          LAUNCH_COMBINE_PDL(6);
          break;
        case 5:
          LAUNCH_COMBINE_PDL(5);
          break;
        case 4:
          LAUNCH_COMBINE_PDL(4);
          break;
        case 3:
          LAUNCH_COMBINE_PDL(3);
          break;
        case 2:
          LAUNCH_COMBINE_PDL(2);
          break;
        default:
          LAUNCH_COMBINE_PDL(0);
          break;
      }
#undef LAUNCH_COMBINE_PDL
    } else if (w11_late_p_wait) {
      dim3 gh((nj + 2 * CMB_WARPS - 1) / (2 * CMB_WARPS));
      cudaLaunchAttribute pdl_attr[1]{};
      pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      pdl_attr[0].val.programmaticStreamSerializationAllowed = pdl_ok;
      cudaLaunchConfig_t pdl_cfg{};
      pdl_cfg.gridDim = gh;
      pdl_cfg.blockDim = dim3(32 * CMB_WARPS);
      pdl_cfg.dynamicSmemBytes = 0;
      pdl_cfg.stream = stream;
      pdl_cfg.attrs = pdl_attr;
      pdl_cfg.numAttrs = 1;
      cudaLaunchKernelEx(&pdl_cfg, combine_half_kernel<7, true>, partial, stats,
                         (__nv_bfloat16*)out.data_ptr(), lsep, nj);
    } else if (B == 1 && !causal && TKV == (int)max_kv_len) {
      switch (NSPLIT) {
        case 25:
          LAUNCH_COMBINE(25);
          break;
        case 24:
          LAUNCH_COMBINE(24);
          break;
        case 23:
          LAUNCH_COMBINE(23);
          break;
        case 20:
          LAUNCH_COMBINE(20);
          break;
        case 12:
          LAUNCH_COMBINE(12);
          break;
        case 8:
          LAUNCH_COMBINE(8);
          break;
        case 7:
          LAUNCH_COMBINE(7);
          break;
        case 6:
          LAUNCH_COMBINE(6);
          break;
        case 5:
          LAUNCH_COMBINE(5);
          break;
        case 4:
          LAUNCH_COMBINE(4);
          break;
        case 3:
          LAUNCH_COMBINE(3);
          break;
        case 2:
          LAUNCH_COMBINE(2);
          break;
        default:
          LAUNCH_COMBINE(0);
          break;
      }
    } else {
      LAUNCH_COMBINE(0);
    }
#undef LAUNCH_COMBINE
    // Covers both the cudaLaunchKernelEx returns and the <<<...>>> forms:
    // every rejected submission sets the thread's last error, and nothing
    // between the launches and this check can overwrite it.
    throw_on_launch_error("split-combine kernel");
  }
  if (max_kv_len <= 256) {
    exact_small_lse_kernel<<<TQ * HQ, 128, 0, stream>>>(
        (const uint8_t*)q.data_ptr(), (const uint8_t*)k.data_ptr(), (const int*)q_lens.data_ptr(),
        (const int*)kv_lens.data_ptr(), (const int*)cu_q.data_ptr(), (const int*)cu_kv.data_ptr(),
        B, TQ, causal, lsep);
    throw_on_launch_error("exact-small-LSE kernel");
  }
}

// Host-only size query for the dispatch adapter: bytes of workspace the
// split-KV path needs for this call shape (0 when the call runs without
// GMEM partials).
int64_t WorkspaceBytes(int64_t B, int64_t TQ, int64_t TKV, int64_t max_q_len, int64_t max_kv_len,
                       int64_t is_causal) {
  const int QT = (int)((max_q_len + BM - 1) / BM);
  if (TQ == 0 || QT == 0) return 0;
  const int NSPLIT =
      choose_nsplit((int)B, (int)TQ, (int)TKV, max_q_len, max_kv_len, (int)is_causal);
  return (int64_t)split_workspace_bytes(NSPLIT, (int)TQ);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_ragged_prefill_dsr1_run, Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_ragged_prefill_dsr1_workspace_bytes, WorkspaceBytes);
