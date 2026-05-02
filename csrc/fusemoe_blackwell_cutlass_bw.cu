/*
 * Originally contributed to the FlashInfer-Bench MLSys'26 kernel-generation
 * contest (https://github.com/flashinfer-ai/mlsys26-contest) by the IFKernel
 * team. Adapted into flashinfer's JIT pipeline.
 *
 * TODO(license): the upstream merge is conditional on the contest team
 * granting an Apache-2.0 license for this file. Until that grant is on file
 * the file must NOT be redistributed.
 */
// Extracted from the embedded build_cutlass_so() in fusemoe_blackwell.cu.
// Compiled directly via flashinfer JIT (was previously built into a .so via
// system("nvcc ...") + dlopen at first call).

// Stub for headers in tilelang-bundled CUTLASS that may reference TRT-LLM.
// flashinfer-bundled CUTLASS does not need them, but we define them as no-ops
// in case a future header pulls them in.
#define TLLM_CHECK(x)
#define TLLM_CHECK_WITH_INFO(x, ...)

// CUTLASS blockwise FP8 GEMM with dual tile configs: 64x128 + 128x128
#include <cuda_runtime.h>

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "fusemoe_blackwell_log.h"
using namespace cute;
using EA = cutlass::float_e4m3_t;
using EB = cutlass::float_e4m3_t;
using EAcc = float;
using EC = cutlass::bfloat16_t;
using LA = cutlass::layout::RowMajor;
using LB = cutlass::layout::ColumnMajor;
using LC = cutlass::layout::RowMajor;
using SC = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, UMMA::Major::K, UMMA::Major::K>;
using LSFA = decltype(SC::deduce_layoutSFA());
using LSFB = decltype(SC::deduce_layoutSFB());
using PS = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using Tl128 = Shape<_128, _128, _128>;
using Cl1 = Shape<_1, _1, _1>;
using Ep128 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tl128, Cl1,
    cutlass::epilogue::collective::EpilogueTileAuto, EAcc, EAcc, EC, LC*, 8, EC, LC*, 8,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
using Ml128 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, EA, cute::tuple<LA*, LSFA*>, 16, EB,
    cute::tuple<LB*, LSFB*>, 16, EAcc, Tl128, Cl1,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Ep128::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;
using Kn128 = cutlass::gemm::kernel::GemmUniversal<PS, Ml128, Ep128>;
using Gm128 = cutlass::gemm::device::GemmUniversalAdapter<Kn128>;
using Tl64 = Shape<_64, _128, _128>;
using Ep64 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tl64, Cl1,
    cutlass::epilogue::collective::EpilogueTileAuto, EAcc, EAcc, EC, LC*, 8, EC, LC*, 8,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
using Ml64 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, EA, cute::tuple<LA*, LSFA*>, 16, EB,
    cute::tuple<LB*, LSFB*>, 16, EAcc, Tl64, Cl1,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Ep64::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;
using Kn64 = cutlass::gemm::kernel::GemmUniversal<PS, Ml64, Ep64>;
using Gm64 = cutlass::gemm::device::GemmUniversalAdapter<Kn64>;
using StA = typename Gm128::GemmKernel::InternalStrideA;
using StB = typename Gm128::GemmKernel::InternalStrideB;
using StD = typename Gm128::GemmKernel::InternalStrideD;
using ILSFA = cute::remove_pointer_t<typename Ml128::LayoutSFA>;
using ILSFB = cute::remove_pointer_t<typename Ml128::LayoutSFB>;
__global__ void prep(EA* A, EB* B, EAcc* SFA, EAcc* SFB, EC* D, int* mi, int* expert_ids, int n,
                     int k, int ng, PS::UnderlyingProblemShape* ps, const EA** pA, const EB** pB,
                     const EAcc** pSFA, const EAcc** pSFB, EC** pD, StA* sA, StB* sB, StD* sD,
                     ILSFA* lA, ILSFB* lB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ng) return;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int sk = k / 128, sn = n / 128, off = mi[i], m = mi[i + 1] - off;
  int eid = expert_ids[i];
  ps[i] = PS::UnderlyingProblemShape(m, n, k);
  sA[i] = cutlass::make_cute_packed_stride(StA{}, {m, k, 1});
  sB[i] = cutlass::make_cute_packed_stride(StB{}, {n, k, 1});
  sD[i] = cutlass::make_cute_packed_stride(StD{}, {m, n, 1});
  pA[i] = A + int64_t(off) * k;
  pB[i] = B + int64_t(eid) * n * k;
  pD[i] = D + int64_t(off) * n;
  lA[i] = SC::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  pSFA[i] = SFA + int64_t(off) * sk;
  lB[i] = SC::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  pSFB[i] = SFB + int64_t(eid) * sn * sk;
}
static Gm128 g_gemm128;
static Gm64 g_gemm64;
static cutlass::KernelHardwareInfo g_hw;
static void* g_ws128 = nullptr;
static size_t g_ws128_sz = 0;
static void* g_ws64 = nullptr;
static size_t g_ws64_sz = 0;
static bool g_init = false;
static int g_mx = 0;
static PS::UnderlyingProblemShape* d_ps = nullptr;
static EA const** d_pA = nullptr;
static EB const** d_pB = nullptr;
static EC** d_pD = nullptr;
static EAcc const** d_pSFA = nullptr;
static EAcc const** d_pSFB = nullptr;
static StA* d_sA = nullptr;
static StB* d_sB = nullptr;
static StD* d_sD = nullptr;
static ILSFA* d_lA = nullptr;
static ILSFB* d_lB = nullptr;
// Duplicate set of arrays for GEMM2 (so both can be pre-filled simultaneously)
static PS::UnderlyingProblemShape* d_ps2 = nullptr;
static EA const** d_pA2 = nullptr;
static EB const** d_pB2 = nullptr;
static EC** d_pD2 = nullptr;
static EAcc const** d_pSFA2 = nullptr;
static EAcc const** d_pSFB2 = nullptr;
static StA* d_sA2 = nullptr;
static StB* d_sB2 = nullptr;
static StD* d_sD2 = nullptr;
static ILSFA* d_lA2 = nullptr;
static ILSFB* d_lB2 = nullptr;
// Returns 0 on success, non-zero on cudaMalloc failure (in which case the
// argument arrays remain in their previous state and g_mx is unchanged).
static int grow(int G) {
  if (G <= g_mx) return 0;
  int n = std::max(G, 64);
  if (d_ps) {
    cudaFree(d_ps);
    cudaFree(d_pA);
    cudaFree(d_pB);
    cudaFree(d_pD);
    cudaFree(d_pSFA);
    cudaFree(d_pSFB);
    cudaFree(d_sA);
    cudaFree(d_sB);
    cudaFree(d_sD);
    cudaFree(d_lA);
    cudaFree(d_lB);
    cudaFree(d_ps2);
    cudaFree(d_pA2);
    cudaFree(d_pB2);
    cudaFree(d_pD2);
    cudaFree(d_pSFA2);
    cudaFree(d_pSFB2);
    cudaFree(d_sA2);
    cudaFree(d_sB2);
    cudaFree(d_sD2);
    cudaFree(d_lA2);
    cudaFree(d_lB2);
  }
  // Allocate every array; if any fail, free what we got and bail without
  // touching g_mx so the next call retries cleanly.
  cudaError_t e = cudaSuccess;
#define _ALLOC(p)                                                         \
  do {                                                                    \
    if (e == cudaSuccess) e = FUSEMOE_CUDA_MALLOC((p), n * sizeof(*(p))); \
  } while (0)
  _ALLOC(d_ps);
  _ALLOC(d_pA);
  _ALLOC(d_pB);
  _ALLOC(d_pD);
  _ALLOC(d_pSFA);
  _ALLOC(d_pSFB);
  _ALLOC(d_sA);
  _ALLOC(d_sB);
  _ALLOC(d_sD);
  _ALLOC(d_lA);
  _ALLOC(d_lB);
  _ALLOC(d_ps2);
  _ALLOC(d_pA2);
  _ALLOC(d_pB2);
  _ALLOC(d_pD2);
  _ALLOC(d_pSFA2);
  _ALLOC(d_pSFB2);
  _ALLOC(d_sA2);
  _ALLOC(d_sB2);
  _ALLOC(d_sD2);
  _ALLOC(d_lA2);
  _ALLOC(d_lB2);
#undef _ALLOC
  if (e != cudaSuccess) {
    void** ptrs[] = {(void**)&d_ps,    (void**)&d_pA,   (void**)&d_pB,  (void**)&d_pD,
                     (void**)&d_pSFA,  (void**)&d_pSFB, (void**)&d_sA,  (void**)&d_sB,
                     (void**)&d_sD,    (void**)&d_lA,   (void**)&d_lB,  (void**)&d_ps2,
                     (void**)&d_pA2,   (void**)&d_pB2,  (void**)&d_pD2, (void**)&d_pSFA2,
                     (void**)&d_pSFB2, (void**)&d_sA2,  (void**)&d_sB2, (void**)&d_sD2,
                     (void**)&d_lA2,   (void**)&d_lB2};
    for (auto* pp : ptrs) {
      if (*pp) {
        cudaFree(*pp);
        *pp = nullptr;
      }
    }
    return -1;
  }
  g_mx = n;
  return 0;
}
#include "fusemoe_blackwell_gemm_args.h"

extern "C" int cutlass_blockwise_fp8_gemm(GemmArgs* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  {
    cudaLaunchConfig_t c;
    c.gridDim = (G + 255) / 256;
    c.blockDim = std::min(G, 256);
    c.dynamicSmemBytes = 0;
    c.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = true;
    c.numAttrs = 1;
    c.attrs = at;
    cudaLaunchKernelEx(&c, prep, (EA*)a->A, (EB*)a->B, (EAcc*)a->SFA, (EAcc*)a->SFB, (EC*)a->D,
                       a->m_indptr, a->expert_ids, a->N, a->K, G, d_ps, (const EA**)d_pA,
                       (const EB**)d_pB, (const EAcc**)d_pSFA, (const EAcc**)d_pSFB, (EC**)d_pD,
                       d_sA, d_sB, d_sD, d_lA, d_lB);
  }
  typename Gm64::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {G, d_ps, nullptr},
                                {(const EA**)d_pA, d_sA, (const EB**)d_pB, d_sB,
                                 (const EAcc**)d_pSFA, d_lA, (const EAcc**)d_pSFB, d_lB},
                                {{}, nullptr, nullptr, (EC**)d_pD, d_sD},
                                g_hw};
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = 4;

  static bool s_validated64 = false;
  if (!s_validated64) {
    auto st = g_gemm64.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1;
    size_t need = Gm64::get_workspace_size(args);
    if (need > g_ws64_sz) {
      if (g_ws64) cudaFree(g_ws64);
      if (FUSEMOE_CUDA_MALLOC(g_ws64, need) != cudaSuccess) return -4;
      g_ws64_sz = need;
    }
    s_validated64 = true;
  }
  auto st = g_gemm64.initialize(args, g_ws64, stream);
  if (st != cutlass::Status::kSuccess) return -2;
  st = g_gemm64.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : -3;
}
extern "C" int cutlass_blockwise_fp8_gemm_128(GemmArgs* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  {
    cudaLaunchConfig_t c;
    c.gridDim = (G + 255) / 256;
    c.blockDim = std::min(G, 256);
    c.dynamicSmemBytes = 0;
    c.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = true;
    c.numAttrs = 1;
    c.attrs = at;
    cudaLaunchKernelEx(&c, prep, (EA*)a->A, (EB*)a->B, (EAcc*)a->SFA, (EAcc*)a->SFB, (EC*)a->D,
                       a->m_indptr, a->expert_ids, a->N, a->K, G, d_ps, (const EA**)d_pA,
                       (const EB**)d_pB, (const EAcc**)d_pSFA, (const EAcc**)d_pSFB, (EC**)d_pD,
                       d_sA, d_sB, d_sD, d_lA, d_lB);
  }
  typename Gm128::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                 {G, d_ps, nullptr},
                                 {(const EA**)d_pA, d_sA, (const EB**)d_pB, d_sB,
                                  (const EAcc**)d_pSFA, d_lA, (const EAcc**)d_pSFB, d_lB},
                                 {{}, nullptr, nullptr, (EC**)d_pD, d_sD},
                                 g_hw};
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = 4;

  static bool s_validated128 = false;
  if (!s_validated128) {
    auto st = g_gemm128.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1;
    size_t need = Gm128::get_workspace_size(args);
    if (need > g_ws128_sz) {
      if (g_ws128) cudaFree(g_ws128);
      if (FUSEMOE_CUDA_MALLOC(g_ws128, need) != cudaSuccess) return -4;
      g_ws128_sz = need;
    }
    s_validated128 = true;
  }
  auto st = g_gemm128.initialize(args, g_ws128, stream);
  if (st != cutlass::Status::kSuccess) return -2;
  st = g_gemm128.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : -3;
}
// Combined prep for both GEMM1 and GEMM2 in a single kernel launch
__global__ void prep_dual(EA* A1, EB* B1, EAcc* SFA1, EAcc* SFB1, EC* D1, int n1, int k1, EA* A2,
                          EB* B2, EAcc* SFA2, EAcc* SFB2, EC* D2, int n2, int k2, int* mi,
                          int* expert_ids, int ng, PS::UnderlyingProblemShape* ps1, const EA** pA1,
                          const EB** pB1, const EAcc** pSFA1, const EAcc** pSFB1, EC** pD1,
                          StA* sA1, StB* sB1, StD* sD1, ILSFA* lA1, ILSFB* lB1,
                          PS::UnderlyingProblemShape* ps2, const EA** pA2, const EB** pB2,
                          const EAcc** pSFA2, const EAcc** pSFB2, EC** pD2, StA* sA2, StB* sB2,
                          StD* sD2, ILSFA* lA2, ILSFB* lB2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ng) return;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int off = mi[i], m = mi[i + 1] - off;
  int eid = expert_ids[i];
  // GEMM1 prep
  {
    int sk = k1 / 128, sn = n1 / 128;
    ps1[i] = PS::UnderlyingProblemShape(m, n1, k1);
    sA1[i] = cutlass::make_cute_packed_stride(StA{}, {m, k1, 1});
    sB1[i] = cutlass::make_cute_packed_stride(StB{}, {n1, k1, 1});
    sD1[i] = cutlass::make_cute_packed_stride(StD{}, {m, n1, 1});
    pA1[i] = A1 + int64_t(off) * k1;
    pB1[i] = B1 + int64_t(eid) * n1 * k1;
    pD1[i] = D1 + int64_t(off) * n1;
    lA1[i] = SC::tile_atom_to_shape_SFA(make_shape(m, n1, k1, 1));
    pSFA1[i] = SFA1 + int64_t(off) * sk;
    lB1[i] = SC::tile_atom_to_shape_SFB(make_shape(m, n1, k1, 1));
    pSFB1[i] = SFB1 + int64_t(eid) * sn * sk;
  }
  // GEMM2 prep
  {
    int sk = k2 / 128, sn = n2 / 128;
    ps2[i] = PS::UnderlyingProblemShape(m, n2, k2);
    sA2[i] = cutlass::make_cute_packed_stride(StA{}, {m, k2, 1});
    sB2[i] = cutlass::make_cute_packed_stride(StB{}, {n2, k2, 1});
    sD2[i] = cutlass::make_cute_packed_stride(StD{}, {m, n2, 1});
    pA2[i] = A2 + int64_t(off) * k2;
    pB2[i] = B2 + int64_t(eid) * n2 * k2;
    pD2[i] = D2 + int64_t(off) * n2;
    lA2[i] = SC::tile_atom_to_shape_SFA(make_shape(m, n2, k2, 1));
    pSFA2[i] = SFA2 + int64_t(off) * sk;
    lB2[i] = SC::tile_atom_to_shape_SFB(make_shape(m, n2, k2, 1));
    pSFB2[i] = SFB2 + int64_t(eid) * sn * sk;
  }
}
// Launch dual prep kernel for both GEMM1 and GEMM2 at once (saves one kernel launch)
extern "C" int cutlass_prep_dual(GemmArgsDual* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  cudaLaunchConfig_t c;
  c.gridDim = (G + 255) / 256;
  c.blockDim = std::min(G, 256);
  c.dynamicSmemBytes = 0;
  c.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = true;
  c.numAttrs = 1;
  c.attrs = at;
  cudaLaunchKernelEx(&c, prep_dual, (EA*)a->A1, (EB*)a->B1, (EAcc*)a->SFA1, (EAcc*)a->SFB1,
                     (EC*)a->D1, a->N1, a->K1, (EA*)a->A2, (EB*)a->B2, (EAcc*)a->SFA2,
                     (EAcc*)a->SFB2, (EC*)a->D2, a->N2, a->K2, a->m_indptr, a->expert_ids, G, d_ps,
                     (const EA**)d_pA, (const EB**)d_pB, (const EAcc**)d_pSFA, (const EAcc**)d_pSFB,
                     (EC**)d_pD, d_sA, d_sB, d_sD, d_lA, d_lB, d_ps2, (const EA**)d_pA2,
                     (const EB**)d_pB2, (const EAcc**)d_pSFA2, (const EAcc**)d_pSFB2, (EC**)d_pD2,
                     d_sA2, d_sB2, d_sD2, d_lA2, d_lB2);
  return 0;
}
// No-prep GEMM: skip prep kernel, use pre-filled arrays (set 1 for GEMM1)
extern "C" int cutlass_blockwise_fp8_gemm_noprep(GemmArgs* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  typename Gm64::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {G, d_ps, nullptr},
                                {(const EA**)d_pA, d_sA, (const EB**)d_pB, d_sB,
                                 (const EAcc**)d_pSFA, d_lA, (const EAcc**)d_pSFB, d_lB},
                                {{}, nullptr, nullptr, (EC**)d_pD, d_sD},
                                g_hw};
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = 4;

  static bool s_validated64np = false;
  if (!s_validated64np) {
    auto st = g_gemm64.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1;
    size_t need = Gm64::get_workspace_size(args);
    if (need > g_ws64_sz) {
      if (g_ws64) cudaFree(g_ws64);
      if (FUSEMOE_CUDA_MALLOC(g_ws64, need) != cudaSuccess) return -4;
      g_ws64_sz = need;
    }
    s_validated64np = true;
  }
  auto st = g_gemm64.initialize(args, g_ws64, stream);
  if (st != cutlass::Status::kSuccess) return -2;
  st = g_gemm64.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : -3;
}
extern "C" int cutlass_blockwise_fp8_gemm_128_noprep(GemmArgs* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  typename Gm128::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                 {G, d_ps, nullptr},
                                 {(const EA**)d_pA, d_sA, (const EB**)d_pB, d_sB,
                                  (const EAcc**)d_pSFA, d_lA, (const EAcc**)d_pSFB, d_lB},
                                 {{}, nullptr, nullptr, (EC**)d_pD, d_sD},
                                 g_hw};
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = 4;

  static bool s_validated128np = false;
  if (!s_validated128np) {
    auto st = g_gemm128.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1;
    size_t need = Gm128::get_workspace_size(args);
    if (need > g_ws128_sz) {
      if (g_ws128) cudaFree(g_ws128);
      if (FUSEMOE_CUDA_MALLOC(g_ws128, need) != cudaSuccess) return -4;
      g_ws128_sz = need;
    }
    s_validated128np = true;
  }
  auto st = g_gemm128.initialize(args, g_ws128, stream);
  if (st != cutlass::Status::kSuccess) return -2;
  st = g_gemm128.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : -3;
}
// No-prep GEMM using array set 2 (for GEMM2 after dual prep)
extern "C" int cutlass_blockwise_fp8_gemm_noprep2(GemmArgs* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  typename Gm64::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {G, d_ps2, nullptr},
                                {(const EA**)d_pA2, d_sA2, (const EB**)d_pB2, d_sB2,
                                 (const EAcc**)d_pSFA2, d_lA2, (const EAcc**)d_pSFB2, d_lB2},
                                {{}, nullptr, nullptr, (EC**)d_pD2, d_sD2},
                                g_hw};
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = 4;

  static bool s_v64np2 = false;
  if (!s_v64np2) {
    auto st = g_gemm64.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1;
    size_t need = Gm64::get_workspace_size(args);
    if (need > g_ws64_sz) {
      if (g_ws64) cudaFree(g_ws64);
      if (FUSEMOE_CUDA_MALLOC(g_ws64, need) != cudaSuccess) return -4;
      g_ws64_sz = need;
    }
    s_v64np2 = true;
  }
  auto st = g_gemm64.initialize(args, g_ws64, stream);
  if (st != cutlass::Status::kSuccess) return -2;
  st = g_gemm64.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : -3;
}
extern "C" int cutlass_blockwise_fp8_gemm_128_noprep2(GemmArgs* a, cudaStream_t stream) {
  if (!g_init) {
    g_hw.device_id = 0;
    g_hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    g_init = true;
  }
  int G = a->num_groups;
  if (grow(G)) return -4;
  typename Gm128::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                 {G, d_ps2, nullptr},
                                 {(const EA**)d_pA2, d_sA2, (const EB**)d_pB2, d_sB2,
                                  (const EAcc**)d_pSFA2, d_lA2, (const EAcc**)d_pSFB2, d_lB2},
                                 {{}, nullptr, nullptr, (EC**)d_pD2, d_sD2},
                                 g_hw};
  args.epilogue.thread.alpha = 1.f;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = 4;

  static bool s_v128np2 = false;
  if (!s_v128np2) {
    auto st = g_gemm128.can_implement(args);
    if (st != cutlass::Status::kSuccess) return -1;
    size_t need = Gm128::get_workspace_size(args);
    if (need > g_ws128_sz) {
      if (g_ws128) cudaFree(g_ws128);
      if (FUSEMOE_CUDA_MALLOC(g_ws128, need) != cudaSuccess) return -4;
      g_ws128_sz = need;
    }
    s_v128np2 = true;
  }
  auto st = g_gemm128.initialize(args, g_ws128, stream);
  if (st != cutlass::Status::kSuccess) return -2;
  st = g_gemm128.run(stream);
  return st == cutlass::Status::kSuccess ? 0 : -3;
}
