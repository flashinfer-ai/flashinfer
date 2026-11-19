// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
//
// Host entry of the DeepSeek-V4-Flash small-M fused TP MoE kernel
// (include/flashinfer/tp_moe/sm100/flash_kernel.cuh). One module per (TP size, token range, entry);
// flashinfer/jit/tp_moe.py passes the build flags.
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>

#include "flashinfer/tp_moe/sm100/flash_kernel.cuh"
#include "tp_moe_host.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

namespace K = flashinfer::tp_moe::sm100::flash;
using flashinfer::tp_moe::sm100::host::encode_tiled;
using flashinfer::tp_moe::sm100::host::WorkspaceLayout;

// The per-group workspace (zero-initialized once): the kernel's epoch-parity counters and its
// per-call scratch.  Every build of a TP size has the same layout (the host checks it), so all
// builds and all layers share one workspace.
enum Buf {
  kXq,
  kHimg,
  kHsf,
  kPart,
  kXgs,
  kXgsfs,
  kHsimg,
  kHssf,
  kShout,
  kXsf2,
  kXflag,
  kGcnt,
  kPanCnt,
  kRdy,
  kFc2Next,
  kRtCnt,
  kWork,
  kOutp,
  kIds,
  kRw,
  kNumBufs
};

constexpr WorkspaceLayout<kNumBufs> make_layout() {
  using namespace K;
  const size_t bytes[kNumBufs] = {
      size_t(MAXM) * DIM,                     // xq: x e4m3
      size_t(MAXP) * NCH2 * NMAX * 128,       // himg
      size_t(MAXP) * NSTG2 * 512,             // hsf
      size_t(MAXG) * 4 * NMAX * 128 * 4,      // part (fp32)
      size_t(MAXPS) * NCH1 * NMAX * 128,      // xgs
      size_t(MAXPS) * NSTG1 * 512,            // xgsfs
      size_t(MAXPS) * NCH2 * NMAX * 128,      // hsimg
      size_t(MAXPS) * NSTG2 * 512,            // hssf
      size_t(MAXM) * DIM * 2,                 // shout (bf16)
      size_t(MAXM) * 16 * 16,                 // xsf2
      2 * size_t(MAXM) * 4,                   // xflag
      2 * 4,                                  // gcnt
      2 * size_t(MAXP) * 4,                   // pan_cnt
      2 * size_t(MAXP + 32) * 4,              // rdy
      4 * 4,                                  // fc2_next
      2 * size_t(NRT) * 4,                    // rt_cnt
      4 * 4,                                  // work
      size_t(MAXM) * TOPK_MAX * OUTP_LD * 2,  // outp (bf16)
      size_t(MAXM) * 6 * 4,                   // ids (gate entry: the gate's top-6)
      size_t(MAXM) * 6 * 4,                   // rw
  };
  return WorkspaceLayout<kNumBufs>(bytes);
}
constexpr WorkspaceLayout<kNumBufs> kLayout = make_layout();
constexpr size_t kRsBytes =
    2 * K::RS_PAR * 16;  // two epoch-parity halves of 16-B all-reduce messages
constexpr size_t kAgBytes = 2 * K::AG_PAR * 16;

// (workspace bytes, reduce-scatter bytes, all-gather bytes, bytes of the rs buffer to fill with
// 0xFF)
Array<int64_t> layout() {
  return Array<int64_t>({static_cast<int64_t>(kLayout.total), static_cast<int64_t>(kRsBytes),
                         static_cast<int64_t>(kAgBytes), 0});
}

template <typename T>
T* ws(TensorView workspace, Buf b) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(workspace.data_ptr()) + kLayout.offset[b]);
}

// Loads the kernel's code and sets its shared-memory limit on the current device, once per device.
// The workspace calls it for every build: under lazy loading the code would otherwise be loaded at
// the first launch, and code first loaded after other modules were loaded without the driver's code
// cache can run slower for the life of the process.
void load() {
  static std::atomic<uint64_t> loaded{0};  // one bit per device
  int dev;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&dev), cudaSuccess);
  TVM_FFI_ICHECK(dev >= 0 && dev < 64);
  const uint64_t bit = uint64_t{1} << dev;
  if (loaded.load(std::memory_order_acquire) & bit) return;
  cudaFuncAttributes attr;
  TVM_FFI_ICHECK_EQ(cudaFuncGetAttributes(&attr, K::tp_moe_kernel), cudaSuccess);
  TVM_FFI_ICHECK_EQ(
      cudaFuncSetAttribute(K::tp_moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           K::SMEM_ALLOC),
      cudaSuccess);
  loaded.fetch_or(bit, std::memory_order_release);
}

void run(TensorView x, Optional<TensorView> router_logits, Optional<TensorView> gate_bias,
         Optional<TensorView> topk_ids, Optional<TensorView> topk_weights, double factor,
         TensorView w13, TensorView w13_sf, TensorView w2, TensorView w2_sf, TensorView lanes,
         TensorView shared_w13, TensorView shared_w2, TensorView shared_sfa1,
         TensorView shared_sfa2, TensorView workspace, Array<int64_t> rs, Array<int64_t> ag,
         TensorView out, int64_t rank, bool enable_pdl) {
  using namespace K;
  const int M = static_cast<int>(x.size(0));
  TVM_FFI_ICHECK(M >= TPMOE_MMIN && M <= TPMOE_MMAX)
      << "tp_moe: M = " << M << " outside this build's [" << TPMOE_MMIN << ", " << TPMOE_MMAX
      << "]";
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == DIM && x.dtype() == dl_bfloat16 && x.IsContiguous());
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) == M && out.size(1) == DIM &&
                 out.dtype() == dl_bfloat16 && out.IsContiguous());
  TVM_FFI_ICHECK(w13.ndim() == 3 && w13.size(0) == NEXP && w13.size(1) == 2 * INTER &&
                 w13.size(2) == DIM / 2);
  TVM_FFI_ICHECK(w2.ndim() == 3 && w2.size(0) == NEXP && w2.size(1) == DIM &&
                 w2.size(2) == INTER / 2);
  TVM_FFI_ICHECK(shared_w13.ndim() == 2 && shared_w13.size(0) == 2 * INTER &&
                 shared_w13.size(1) == DIM);
  TVM_FFI_ICHECK(shared_w2.ndim() == 2 && shared_w2.size(0) == DIM && shared_w2.size(1) == INTER);
  TVM_FFI_ICHECK(lanes.numel() == 128);
  TVM_FFI_ICHECK(workspace.numel() >= static_cast<int64_t>(kLayout.total));
  TVM_FFI_ICHECK(static_cast<int>(rs.size()) == NDEV && static_cast<int>(ag.size()) == NDEV);
  TVM_FFI_ICHECK(rank >= 0 && rank < NDEV);
  int device;
  cudaGetDevice(&device);
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  TVM_FFI_ICHECK_EQ(num_sms, 148) << "tp_moe: the kernel is compiled for a 148-SM B200 grid";

  Params P;
  std::memset(&P, 0, sizeof(P));
  encode_tiled(&P.map1, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, w13.data_ptr(),
               {128, uint64_t(NEXP) * 2 * INTER, NCH1}, {DIM / 2, 64}, {128, 128, KC},
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  encode_tiled(&P.map2, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, w2.data_ptr(),
               {128, uint64_t(NEXP) * DIM, NCH2}, {INTER / 2, 64}, {128, 128, KC},
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  P.w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr());
  P.w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr());
  P.lanes = static_cast<const uint8_t*>(lanes.data_ptr());
  P.xb = static_cast<const __nv_bfloat16*>(x.data_ptr());
#if TPMOE_GATE
  TVM_FFI_ICHECK(router_logits.has_value() && gate_bias.has_value())
      << "gate-entry build: router_logits and gate_bias";
  TensorView lg = router_logits.value();
  TVM_FFI_ICHECK(lg.ndim() == 2 && lg.size(0) == M && lg.size(1) == NEXP &&
                 lg.dtype() == dl_float32 && lg.IsContiguous());
  TVM_FFI_ICHECK(gate_bias.value().numel() == NEXP && gate_bias.value().dtype() == dl_float32);
  P.glog = static_cast<const float*>(lg.data_ptr());
  P.gbias = static_cast<const float*>(gate_bias.value().data_ptr());
  P.ids = ws<int>(workspace, kIds);
  P.rw = ws<float>(workspace, kRw);
#else
  TVM_FFI_ICHECK(topk_ids.has_value() && topk_weights.has_value())
      << "ids-entry build: topk_ids and topk_weights";
  TensorView ids = topk_ids.value(), rw = topk_weights.value();
  TVM_FFI_ICHECK(ids.ndim() == 2 && ids.size(0) == M && ids.size(1) == 6 &&
                 ids.dtype() == dl_int32 && ids.IsContiguous());
  TVM_FFI_ICHECK(rw.ndim() == 2 && rw.size(0) == M && rw.size(1) == 6 && rw.dtype() == dl_float32 &&
                 rw.IsContiguous());
  P.ids = static_cast<int*>(ids.data_ptr());
  P.rw = static_cast<float*>(rw.data_ptr());
#endif
  P.factor = static_cast<float>(factor);
  P.xq = ws<uint8_t>(workspace, kXq);
  P.himg = ws<uint8_t>(workspace, kHimg);
  P.hsf = ws<uint8_t>(workspace, kHsf);
  P.part = ws<float>(workspace, kPart);
  P.xflag = ws<unsigned>(workspace, kXflag);
  P.gcnt = ws<unsigned>(workspace, kGcnt);
  P.pan_cnt = ws<unsigned>(workspace, kPanCnt);
  P.rdy = ws<unsigned>(workspace, kRdy);
  P.fc2_next = ws<unsigned>(workspace, kFc2Next);
  P.rt_cnt = ws<unsigned>(workspace, kRtCnt);
  P.work = ws<unsigned>(workspace, kWork);
  P.outp = ws<__nv_bfloat16>(workspace, kOutp);
  for (int d = 0; d < NDEV; ++d) {
    P.rs[d] = reinterpret_cast<uint4*>(rs[d]);
    P.ag[d] = reinterpret_cast<uint4*>(ag[d]);
  }
  P.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  P.M = M;
  P.my_rank = static_cast<int>(rank);
  P.oneshot = M <= TPMOE_ONESHOT_MAXM ? 1 : 0;
  encode_tiled(&P.map1s, CU_TENSOR_MAP_DATA_TYPE_UINT8, shared_w13.data_ptr(),
               {128, 2 * INTER, NCH1}, {DIM, 128}, {128, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  encode_tiled(&P.map2s, CU_TENSOR_MAP_DATA_TYPE_UINT8, shared_w2.data_ptr(), {128, DIM, NCH2},
               {INTER, 128}, {128, 128, KC}, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  P.sfa1s = static_cast<const uint8_t*>(shared_sfa1.data_ptr());
  P.sfa2s = static_cast<const uint8_t*>(shared_sfa2.data_ptr());
  P.xgs = ws<uint8_t>(workspace, kXgs);
  P.xgsfs = ws<uint8_t>(workspace, kXgsfs);
  P.hsimg = ws<uint8_t>(workspace, kHsimg);
  P.hssf = ws<uint8_t>(workspace, kHssf);
  P.shout = ws<__nv_bfloat16>(workspace, kShout);
  P.xsf2 = ws<uint8_t>(workspace, kXsf2);
  // x codes, token-major, for the gather4 B-operand loads (box height 1)
  encode_tiled(&P.mapxq, CU_TENSOR_MAP_DATA_TYPE_UINT8, P.xq, {DIM, MAXM}, {DIM}, {128, 1},
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);

  load();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(148);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM_ALLOC;
  cfg.stream = get_stream(x.device());
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  TVM_FFI_ICHECK_EQ(cudaLaunchKernelEx(&cfg, tp_moe_kernel, P), cudaSuccess)
      << "tp_moe: kernel launch failed";
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(layout, layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(load, load);
