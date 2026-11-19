// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
//
// Host entry of the shape-generic fused TP MoE kernel
// (include/flashinfer/tp_moe/sm100/generic_kernel.cuh): DeepSeek-V4.1-Flash.  One module per
// (TP size, token range); flashinfer/jit/tp_moe.py passes the build flags.  Both entries (fused
// gate / given top-k ids) share a build.
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>

#include "flashinfer/tp_moe/sm100/generic_kernel.cuh"
#include "tp_moe_host.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

namespace K = flashinfer::tp_moe::sm100::generic;
using flashinfer::tp_moe::sm100::host::encode_tiled;
using flashinfer::tp_moe::sm100::host::WorkspaceLayout;

enum Buf {
  kXq,
  kHimg,
  kHsf,
  kPart,
  kXflag,
  kGcnt,
  kPanCnt,
  kRdy,
  kFc2Next,
  kRtCnt,
  kWork,
  kOutp,
  kXsf2,
  kXgs,
  kXgsfs,
  kHsimg,
  kHssf,
  kShout,
  kIds,
  kRw,
  kNumBufs
};

// Sizes depend on the model shape and TP size only (not on the build's token range or panel width):
// every build of one (model, TP size) shares the workspace.
constexpr WorkspaceLayout<kNumBufs> make_layout() {
  using namespace K;
  constexpr size_t kPanelRows = 32;  // the widest panel of any build (NMAX 24 or 32)
  constexpr size_t kSharedPanels = MAXM / 16;
  const size_t bytes[kNumBufs] = {
      size_t(MAXM) * DIM,                                   // xq
      size_t(MAXP) * NCH2 * kPanelRows * 128 + 4096,        // himg
      size_t(MAXP) * (NSTG2 > 4 ? NSTG2 : 4) * 512 + 4096,  // hsf
      size_t(MAXG) * 4 * kPanelRows * 128 * 4,              // part (fp32)
      2 * size_t(MAXM) * 4,                                 // xflag
      2 * 4,                                                // gcnt
      2 * size_t(MAXP) * 4,                                 // pan_cnt
      2 * size_t(MAXP + 32) * 4,                            // rdy
      4 * 4,                                                // fc2_next
      2 * size_t(NRT) * 4,                                  // rt_cnt
      (64 + 256) * 4,                                       // work
      size_t(MAXM) * TOPK_MAX * OUTP_LD * 2,                // outp (bf16)
      size_t(MAXM) * NSTG1 * 16,                            // xsf2
      size_t(MAXM) * NCH1 * 128 + 4096,                     // xgs
      kSharedPanels * NSTG1 * 512,                          // xgsfs
      size_t(MAXM) * NCH2 * 128 + 4096,                     // hsimg
      kSharedPanels * (NSTG2 > 4 ? NSTG2 : 4) * 512,        // hssf
      size_t(MAXM) * DIM * 2,                               // shout (bf16)
      size_t(MAXM) * TOPK_MAX * 4,                          // ids (gate entry: the gate's top-k)
      size_t(MAXM) * TOPK_MAX * 4,                          // rw
  };
  return WorkspaceLayout<kNumBufs>(bytes);
}
constexpr WorkspaceLayout<kNumBufs> kLayout = make_layout();
constexpr size_t kLlBytes = 2 * K::RS_PAR * 16;  // all-reduce message slots (0xFF = empty)
constexpr size_t kRsBytes = kLlBytes;

Array<int64_t> layout() {
  return Array<int64_t>({static_cast<int64_t>(kLayout.total), static_cast<int64_t>(kRsBytes), 0,
                         static_cast<int64_t>(kLlBytes)});
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
         Optional<TensorView> gate_bias_vision, Optional<TensorView> input_ids,
         int64_t image_token_id, double gate_scale, Optional<TensorView> topk_ids,
         Optional<TensorView> topk_weights, double factor, TensorView w13, TensorView w13_sf,
         TensorView w2, TensorView w2_sf, TensorView w2t, TensorView w2t_sf, TensorView lanes,
         TensorView shared_w13, TensorView shared_w2, TensorView shared_sfa1,
         TensorView shared_sfa2, TensorView workspace, Array<int64_t> rs, TensorView out,
         int64_t rank, bool enable_pdl) {
  using namespace K;
  const int M = static_cast<int>(x.size(0));
  TVM_FFI_ICHECK(M >= TPMOE_MMIN && M <= TPMOE_MMAX)
      << "tp_moe: M = " << M << " outside this build's [" << TPMOE_MMIN << ", " << TPMOE_MMAX
      << "]";
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == DIM && x.dtype() == dl_bfloat16 && x.IsContiguous());
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) == M && out.size(1) == DIM &&
                 out.dtype() == dl_bfloat16 && out.IsContiguous());
  TVM_FFI_ICHECK(w13.ndim() == 3 && w13.size(0) == NEXP && w13.size(1) == 2 * IPAD &&
                 w13.size(2) == DIM / 2);
  TVM_FFI_ICHECK(w2.ndim() == 3 && w2.size(0) == NEXP && w2.size(1) == DIM &&
                 w2.size(2) == IPAD / 2);
  TVM_FFI_ICHECK(shared_w13.ndim() == 2 && shared_w13.size(0) == 2 * SHI &&
                 shared_w13.size(1) == DIM);
  TVM_FFI_ICHECK(shared_w2.ndim() == 2 && shared_w2.size(0) == DIM && shared_w2.size(1) == SHI);
  TVM_FFI_ICHECK(lanes.numel() == 128);
  if (RAGI)  // (F2T) the w2 K tails: [E][128 rows][NRT2 * TAILK / 2 bytes] + [E][NTL][512] images
    TVM_FFI_ICHECK(w2t.ndim() == 3 && w2t.size(0) == NEXP && w2t.size(1) == 128 &&
                   w2t.size(2) == NRT2 * TAILK / 2 &&
                   w2t_sf.numel() == static_cast<int64_t>(NEXP) * NTL * 512);
  TVM_FFI_ICHECK(workspace.numel() >= static_cast<int64_t>(kLayout.total));
  TVM_FFI_ICHECK(static_cast<int>(rs.size()) == NDEV);
  TVM_FFI_ICHECK(rank >= 0 && rank < NDEV);
  int device;
  cudaGetDevice(&device);
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  TVM_FFI_ICHECK_EQ(num_sms, 148) << "tp_moe: the kernel is compiled for a 148-SM B200 grid";

  Params P;
  std::memset(&P, 0, sizeof(P));
  // routed weights: w13 per expert (4D: rows, k chunks, experts; the 2 INTER valid rows of the
  // expert's 2 IPAD), w2 [E * DIM rows][IPAD / 2 bytes] in k chunks (ragged I: only the whole
  // chunks are read through it, the K tails from w2t)
  encode_tiled(&P.map1, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, w13.data_ptr(),
               {128, uint64_t(2 * INTER), NCH1, NEXP},
               {DIM / 2, 64, uint64_t(2 * IPAD) * (DIM / 2)}, {128, 128, KC, 1},
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  encode_tiled(&P.map2, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, w2.data_ptr(),
               {128, uint64_t(NEXP) * DIM, NCH2}, {IPAD / 2, 64}, {128, 128, KC},
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  P.w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr());
  P.w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr());
  if (RAGI) {
    for (int nv = 1; nv <= 2; ++nv)
      encode_tiled(nv == 2 ? &P.map2t2 : &P.map2t1, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                   w2t.data_ptr(), {128, uint64_t(NEXP) * 128, NTL}, {NRT2 * TAILK / 2, 64},
                   {128, 128, static_cast<uint32_t>(nv)}, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    P.w2t_sf = static_cast<const uint8_t*>(w2t_sf.data_ptr());
  }
  P.lanes = static_cast<const uint8_t*>(lanes.data_ptr());
  P.xb = static_cast<const __nv_bfloat16*>(x.data_ptr());
  if (router_logits
          .has_value()) {  // fused gate: the kernel writes the top-k ids / weights to the workspace
    TVM_FFI_ICHECK(gate_bias.has_value()) << "the gate entry needs gate_bias";
    TensorView lg = router_logits.value();
    TVM_FFI_ICHECK(lg.ndim() == 2 && lg.size(0) == M && lg.size(1) == TPMOE_NEXP &&
                   lg.dtype() == dl_float32 && lg.IsContiguous());
    TVM_FFI_ICHECK(gate_bias.value().numel() == TPMOE_NEXP &&
                   gate_bias.value().dtype() == dl_float32);
    P.glog = static_cast<const float*>(lg.data_ptr());
    P.gbias = static_cast<const float*>(gate_bias.value().data_ptr());
    P.ids = ws<int>(workspace, kIds);
    P.rw = ws<float>(workspace, kRw);
  } else {
    TVM_FFI_ICHECK(topk_ids.has_value() && topk_weights.has_value());
    TensorView ids = topk_ids.value(), rw = topk_weights.value();
    TVM_FFI_ICHECK(ids.ndim() == 2 && ids.size(0) == M && ids.size(1) == 6 &&
                   ids.dtype() == dl_int32 && ids.IsContiguous());
    TVM_FFI_ICHECK(rw.ndim() == 2 && rw.size(0) == M && rw.size(1) == 6 &&
                   rw.dtype() == dl_float32 && rw.IsContiguous());
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(ids.data_ptr()) % 16 == 0)
        << "topk_ids: 16-B aligned";
    P.ids = static_cast<int*>(ids.data_ptr());
    P.rw = static_cast<float*>(rw.data_ptr());
  }
  if (router_logits
          .has_value()) {  // image tokens (input_ids == image_token_id) use the vision bias
    TVM_FFI_ICHECK(gate_bias_vision.has_value() && input_ids.has_value());
    TVM_FFI_ICHECK(gate_bias_vision.value().numel() == TPMOE_NEXP &&
                   gate_bias_vision.value().dtype() == dl_float32);
    TVM_FFI_ICHECK(input_ids.value().numel() >= M && input_ids.value().dtype() == dl_int32);
    P.gbias_alt = static_cast<const float*>(gate_bias_vision.value().data_ptr());
    P.in_ids = static_cast<const int*>(input_ids.value().data_ptr());
    P.img_tok = static_cast<int>(image_token_id);
    P.gscale = static_cast<float>(gate_scale);
  } else {
    P.img_tok = -1;
    P.gscale = static_cast<float>(gate_scale);  // (ids entry: applied to the given weights)
  }
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
  for (int d = 0; d < NDEV;
       ++d) {  // one IPC buffer per rank holds both the reduce-scatter and the all-gather messages
    P.rs[d] = reinterpret_cast<uint4*>(rs[d]);
    P.ag[d] = reinterpret_cast<uint4*>(rs[d]);
  }
  P.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  P.M = M;
  P.my_rank = static_cast<int>(rank);
  P.ebase = TPMOE_ISPLIT < NDEV ? static_cast<int>(rank / TPMOE_ISPLIT) * NEXP
                                : 0;  // hybrid: this rank's expert group
  P.oneshot = M <= 32 ? 1 : 0;
  encode_tiled(&P.map1s, CU_TENSOR_MAP_DATA_TYPE_UINT8, shared_w13.data_ptr(), {128, 2 * SHI, NCH1},
               {DIM, 128}, {128, 128, 1}, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  encode_tiled(&P.map2s, CU_TENSOR_MAP_DATA_TYPE_UINT8, shared_w2.data_ptr(), {128, DIM, NCH2S},
               {SHI, 128}, {128, 128, KC}, CU_TENSOR_MAP_SWIZZLE_128B,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
  P.sfa1s = static_cast<const uint8_t*>(shared_sfa1.data_ptr());
  P.sfa2s = static_cast<const uint8_t*>(shared_sfa2.data_ptr());
  P.xgs = ws<uint8_t>(workspace, kXgs);
  P.xgsfs = ws<uint8_t>(workspace, kXgsfs);
  P.hsimg = ws<uint8_t>(workspace, kHsimg);
  P.hssf = ws<uint8_t>(workspace, kHssf);
  P.shout = ws<__nv_bfloat16>(workspace, kShout);
  P.xsf2 = ws<uint8_t>(workspace, kXsf2);
  encode_tiled(&P.mapxq, CU_TENSOR_MAP_DATA_TYPE_UINT8, P.xq, {DIM, MAXM}, {DIM}, {128, 1},
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  TVM_FFI_ICHECK(M * TOPK_MAX <= MAXE);
  TVM_FFI_ICHECK(((M + NMAX - 1) / NMAX) * (SHI / 128) <= 148)
      << "shared-expert pieces exceed the grid";

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
