// SM120 MXFP4 MegaMoE launch/binding. The symmetric buffer layout is computed by the nosync
// backend's workspace.py (buffer_layout); `offsets` and `dims` carry it here. Built twice:
// the P2P module, and with FLASHINFER_MEGAMOE_SM120_GIN=1 (NCCL Device API) for buffers with NIC
// (GIN) pairs, whose NCCL window and device communicator come from megamoe_sm120_gin_session.cu.
#include <cuda_runtime.h>

#include <flashinfer/megamoe/sm120_mxfp4_mega_moe.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe {
namespace s = sm120;

// offsets: byte offsets into the symmetric buffer, in Sm120Offset order (index 0: total bytes)
enum Sm120Offset : int {
  kOffTotal = 0,
  kOffX,
  kOffXSf,
  kOffTopkIdx,
  kOffTopkWeights,
  kOffPool,
  kOffRecvCnt,
  kOffRecvFlag,
  kOffComb,
  kOffCombFlag,
  kOffRoute,
  kOffRouteFlag,
  kOffInter,
  kOffInterS,
  kOffCtr,
  kOffStageRow,
  kOffInbox,
  kOffStageComb,
  kOffCombGin,
  kOffSlotMap,
  kOffStageSlot,
  kOffCidx,
  kNumSm120Offsets
};

// Every CTA of the grid must be resident at once (CTAs wait on each other's flags), so the grid is
// the device's full occupancy and the launch is cooperative.
constexpr int kTile16Rows = 3;  // routed rows per local expert from which items cover 16 tokens

template <int kTileN>
static int sm120_grid_size(int dev) {
  static int grids[64] = {};
  TVM_FFI_ICHECK(dev >= 0 && dev < 64);
  if (grids[dev] > 0) return grids[dev];
  int sms = 0, occ = 0, coop = 0;
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev), cudaSuccess);
  TVM_FFI_ICHECK(coop) << "SM120 MegaMoE needs cooperative launch support";
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &occ, s::sm120_mega_moe_kernel<kTileN>, s::kWarps * 32, 0),
                    cudaSuccess);
  return grids[dev] = occ * sms;
}

template <int kTileN>
static void sm120_launch(s::Args& a, int dev, cudaStream_t stream) {
  void* params[] = {&a};
  TVM_FFI_ICHECK_EQ(
      cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(s::sm120_mega_moe_kernel<kTileN>),
                                  dim3(sm120_grid_size<kTileN>(dev)), dim3(s::kWarps * 32), params,
                                  0, stream),
      cudaSuccess);
}

void megamoe_sm120_mxfp4_fused(tvm::ffi::TensorView y, tvm::ffi::TensorView symm_buffer,
                               tvm::ffi::Array<int64_t> symm_buffer_ptrs, int64_t rank_idx,
                               tvm::ffi::TensorView l1_weights, tvm::ffi::TensorView l2_weights,
                               tvm::ffi::TensorView l1_scales, tvm::ffi::TensorView l2_scales,
                               tvm::ffi::Array<int64_t> offsets, tvm::ffi::Array<int64_t> dims,
                               int64_t num_tokens, double activation_clamp,
                               tvm::ffi::Array<int64_t> gin, int64_t tokens_per_item) {
  // dims: [num_experts, num_topk, hidden, intermediate_hidden, num_max_tokens_per_rank]
  // l1/l2_weights: row-major packed E2M1 [EL][2I][H/2] / [EL][H][I/2] (the cake / cutedsl layout);
  // l1/l2_scales: [EL][bytes] UE8M0 words per 16-row block and K128
  // gin: [] (every pair P2P) or [device ncclDevComm*, ncclWindow_t, GIN rank mask, GIN contexts]
  TVM_FFI_ICHECK_EQ(offsets.size(), kNumSm120Offsets);
  TVM_FFI_ICHECK_EQ(dims.size(), 5);
  const int world = static_cast<int>(symm_buffer_ptrs.size());
  const int experts = static_cast<int>(dims[0]), topk = static_cast<int>(dims[1]);
  const int hidden = static_cast<int>(dims[2]), inter = static_cast<int>(dims[3]);
  const int capacity = static_cast<int>(dims[4]);
  TVM_FFI_ICHECK(world >= 1 && world <= s::kMaxWorld);
  TVM_FFI_ICHECK(rank_idx >= 0 && rank_idx < world);
  TVM_FFI_ICHECK(num_tokens >= 1 && num_tokens <= capacity);
  TVM_FFI_ICHECK_EQ(experts % world, 0);

  s::Args a{};
  char* local = static_cast<char*>(symm_buffer.data_ptr());
  a.x = reinterpret_cast<const uint8_t*>(local + offsets[kOffX]);
  a.x_sf = reinterpret_cast<const uint8_t*>(local + offsets[kOffXSf]);
  a.topk_ids = reinterpret_cast<const int64_t*>(local + offsets[kOffTopkIdx]);
  a.topk_w = reinterpret_cast<const float*>(local + offsets[kOffTopkWeights]);
  a.out = static_cast<__nv_bfloat16*>(y.data_ptr());
  a.l1 = static_cast<const uint8_t*>(l1_weights.data_ptr());
  a.l2 = static_cast<const uint8_t*>(l2_weights.data_ptr());
  a.l1s = static_cast<const uint8_t*>(l1_scales.data_ptr());
  a.l2s = static_cast<const uint8_t*>(l2_scales.data_ptr());
  a.l1_stride = static_cast<size_t>(2 * inter) * hidden / 2;
  a.l2_stride = static_cast<size_t>(hidden) * inter / 2;
  a.l1s_stride = a.l1_stride / 16;
  a.l2s_stride = a.l2_stride / 16;
  TVM_FFI_ICHECK_EQ(l1_weights.numel(), int64_t(experts / world) * int64_t(a.l1_stride));
  TVM_FFI_ICHECK_EQ(l2_weights.numel(), int64_t(experts / world) * int64_t(a.l2_stride));
  TVM_FFI_ICHECK_EQ(l1_scales.numel(), int64_t(experts / world) * int64_t(a.l1s_stride));
  TVM_FFI_ICHECK_EQ(l2_scales.numel(), int64_t(experts / world) * int64_t(a.l2s_stride));
  a.stamps = nullptr;
  a.inter = reinterpret_cast<uint8_t*>(local + offsets[kOffInter]);
  a.inter_s = reinterpret_cast<uint32_t*>(local + offsets[kOffInterS]);
  a.ctr = reinterpret_cast<int*>(local + offsets[kOffCtr]);
  if (gin.size() > 0) {
#if FLASHINFER_MEGAMOE_SM120_GIN
    TVM_FFI_ICHECK_EQ(gin.size(), 4);
    a.gin.comm = reinterpret_cast<const void*>(gin[0]);
    a.gin.win = reinterpret_cast<void*>(gin[1]);
    a.gin.mask = static_cast<uint32_t>(gin[2]);
    a.gin.nctx = static_cast<int>(gin[3]);
    TVM_FFI_ICHECK(a.gin.nctx >= 1);
    TVM_FFI_ICHECK(!((a.gin.mask >> rank_idx) & 1u)) << "a rank cannot reach itself over GIN";
    a.gin.off_route = offsets[kOffRoute];
    a.gin.off_inbox = offsets[kOffInbox];
    a.gin.off_comb_gin = offsets[kOffCombGin];
    a.gin.off_stage_row = offsets[kOffStageRow];
    a.gin.off_stage_comb = offsets[kOffStageComb];
    a.gin.off_slotmap = offsets[kOffSlotMap];
    a.gin.off_stage_slot = offsets[kOffStageSlot];
    a.gin.stage_row = reinterpret_cast<uint8_t*>(local + offsets[kOffStageRow]);
    a.gin.inbox = reinterpret_cast<uint8_t*>(local + offsets[kOffInbox]);
    a.gin.stage_comb = reinterpret_cast<__nv_bfloat16*>(local + offsets[kOffStageComb]);
    a.gin.comb_gin = reinterpret_cast<__nv_bfloat16*>(local + offsets[kOffCombGin]);
    a.gin.slotmap = reinterpret_cast<int*>(local + offsets[kOffSlotMap]);
    a.gin.stage_slot = reinterpret_cast<int*>(local + offsets[kOffStageSlot]);
    a.gin.cidx = reinterpret_cast<int*>(local + offsets[kOffCidx]);
#else
    TVM_FFI_ICHECK(false) << "GIN pairs need the FLASHINFER_MEGAMOE_SM120_GIN module";
#endif
  }
  for (int w = 0; w < world; ++w) {
    char* b = reinterpret_cast<char*>(symm_buffer_ptrs[w]);
    if (b == nullptr) continue;  // GIN peer: no load/store mapping
    a.peers.pool[w] = reinterpret_cast<uint8_t*>(b + offsets[kOffPool]);
    a.peers.recv_cnt[w] = reinterpret_cast<int*>(b + offsets[kOffRecvCnt]);
    a.peers.recv_flag[w] = reinterpret_cast<int*>(b + offsets[kOffRecvFlag]);
    a.peers.comb[w] = reinterpret_cast<__nv_bfloat16*>(b + offsets[kOffComb]);
    a.peers.comb_flag[w] = reinterpret_cast<int*>(b + offsets[kOffCombFlag]);
    a.peers.route[w] = reinterpret_cast<int*>(b + offsets[kOffRoute]);
    a.peers.route_flag[w] = reinterpret_cast<int*>(b + offsets[kOffRouteFlag]);
  }
  a.T = static_cast<int>(num_tokens);
  a.H = hidden, a.I = inter, a.EL = experts / world, a.W = world, a.TOPK = topk, a.TCAP = capacity;
  a.rank = static_cast<int>(rank_idx);
  a.swiglu_limit = static_cast<float>(activation_clamp);

  ffi::CUDADeviceGuard device_guard(y.device().device_id);
  const int dev = y.device().device_id;
  cudaStream_t stream = get_stream(y.device());
  // Work-item width (tokens_per_item: 8, 16, 24 or 32; 0 = default). Default: by the routed rows
  // each local expert averages, T * TOPK / EL (the rows a rank receives, spread over its experts),
  // 16 tokens (two MMAs per weight fragment) from 3 rows, else 8. Measured with cold L2 (decode
  // reads every layer's weights from DRAM) on RTX 6000D, EP4 / EP8, DeepSeek-V4-Flash,
  // DeepSeek-V4.1-Flash and GLM-5.3-Flash (top-6 / top-8), 16-128 tokens per rank: 16 matches 8
  // below 3 rows and is up to 12% faster above; 24 and 32 load more activation fragments per
  // weight fragment and are 0-6% slower than 16 (one point 1% faster: V4.1-Flash EP8, 128 tokens).
  const int width = tokens_per_item ? static_cast<int>(tokens_per_item)
                                    : (a.T * a.TOPK >= kTile16Rows * a.EL ? 16 : 8);
  switch (width) {
    case 8:
      sm120_launch<8>(a, dev, stream);
      break;
    case 16:
      sm120_launch<16>(a, dev, stream);
      break;
    case 24:
      sm120_launch<24>(a, dev, stream);
      break;
    case 32:
      sm120_launch<32>(a, dev, stream);
      break;
    default:
      TVM_FFI_ICHECK(false) << "tokens_per_item must be 8, 16, 24 or 32, got " << width;
  }
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm120_mxfp4_fused,
                              flashinfer::megamoe::megamoe_sm120_mxfp4_fused);
