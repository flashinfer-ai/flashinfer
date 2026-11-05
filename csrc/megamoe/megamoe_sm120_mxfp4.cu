// SM120 MXFP4 MegaMoE launch/binding. The symmetric buffer layout is computed by
// flashinfer/megamoe/__init__.py (_sm120_layout); `offsets` and `dims` carry it here.
#include <cuda_runtime.h>

#include <flashinfer/megamoe/sm120_mxfp4_mega_moe.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe {
namespace s = sm120;

// offsets: [total, x, x_sf, topk_idx, topk_weights, pool_x, pool_xs, pool_meta, recv_cnt,
// recv_flag,
//           comb, comb_flag, route, route_flag, inter, inter_s, ctr]
enum Sm120Offset : int {
  kOffTotal = 0,
  kOffX,
  kOffXSf,
  kOffTopkIdx,
  kOffTopkWeights,
  kOffPoolX,
  kOffPoolXs,
  kOffPoolMeta,
  kOffRecvCnt,
  kOffRecvFlag,
  kOffComb,
  kOffCombFlag,
  kOffRoute,
  kOffRouteFlag,
  kOffInter,
  kOffInterS,
  kOffCtr,
  kNumSm120Offsets
};

static int sm120_grid_size() {
  int dev = 0, sms = 0, occ = 0;
  TVM_FFI_ICHECK_EQ(cudaGetDevice(&dev), cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, s::sm120_mega_moe_kernel,
                                                                  s::kWarps * 32, 0),
                    cudaSuccess);
  return occ * sms;
}

void megamoe_sm120_mxfp4_fused(
    tvm::ffi::TensorView y, tvm::ffi::TensorView symm_buffer,
    tvm::ffi::Array<int64_t> symm_buffer_ptrs, int64_t rank_idx, tvm::ffi::TensorView l1_weights,
    tvm::ffi::TensorView l2_weights, tvm::ffi::Array<int64_t> offsets,
    tvm::ffi::Array<int64_t> dims, int64_t num_tokens, double activation_clamp,
    tvm::ffi::Optional<tvm::ffi::TensorView> cumulative_local_expert_recv_stats,
    tvm::ffi::Optional<tvm::ffi::TensorView> l1_global_scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> l2_global_scales) {
  // dims: [num_experts, num_topk, hidden, intermediate_hidden, num_max_tokens_per_rank]
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
  a.x_sf = reinterpret_cast<const float*>(local + offsets[kOffXSf]);
  a.topk_ids = reinterpret_cast<const int64_t*>(local + offsets[kOffTopkIdx]);
  a.topk_w = reinterpret_cast<const float*>(local + offsets[kOffTopkWeights]);
  a.out = static_cast<__nv_bfloat16*>(y.data_ptr());
  a.l1 = static_cast<const uint8_t*>(l1_weights.data_ptr());
  a.l2 = static_cast<const uint8_t*>(l2_weights.data_ptr());
  a.l1_stride = static_cast<size_t>(l1_weights.size(1));
  a.l2_stride = static_cast<size_t>(l2_weights.size(1));
  a.l1_w_bytes = static_cast<size_t>(2 * inter) * hidden / 2;
  a.l2_w_bytes = static_cast<size_t>(hidden) * inter / 2;
  auto opt = [](const tvm::ffi::Optional<tvm::ffi::TensorView>& t) -> void* {
    return t.has_value() ? t.value().data_ptr() : nullptr;
  };
  a.gs1 = static_cast<const float*>(opt(l1_global_scales));
  a.gs2 = static_cast<const float*>(opt(l2_global_scales));
  a.stats = static_cast<int*>(opt(cumulative_local_expert_recv_stats));
  a.inter = reinterpret_cast<uint8_t*>(local + offsets[kOffInter]);
  a.inter_s = reinterpret_cast<uint32_t*>(local + offsets[kOffInterS]);
  a.ctr = reinterpret_cast<int*>(local + offsets[kOffCtr]);
  a.stamps = nullptr;
  for (int w = 0; w < world; ++w) {
    char* b = reinterpret_cast<char*>(symm_buffer_ptrs[w]);
    a.peers.pool_x[w] = reinterpret_cast<uint8_t*>(b + offsets[kOffPoolX]);
    a.peers.pool_xs[w] = reinterpret_cast<uint32_t*>(b + offsets[kOffPoolXs]);
    a.peers.pool_meta[w] = reinterpret_cast<int*>(b + offsets[kOffPoolMeta]);
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

  static const int grid = sm120_grid_size();
  s::sm120_mega_moe_kernel<<<grid, s::kWarps * 32, 0, get_stream(y.device())>>>(a);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace flashinfer::megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(megamoe_sm120_mxfp4_fused,
                              flashinfer::megamoe::megamoe_sm120_mxfp4_fused);
