/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Entry points of the SM90 fused TP-MoE kernel for DeepSeek-V4.1-Flash
// (flashinfer/tp_moe/sm90/dsv41_kernel.cuh); one JIT module per TP size.  See
// flashinfer/fused_moe/_tp_moe_sm90.py for the Python side.
#include "flashinfer/tp_moe/sm90/dsv41_kernel.cuh"
#include "tp_moe_sm90_host.cuh"

namespace K = flashinfer::tp_moe::sm90::dsv41;
namespace H = flashinfer::tp_moe::sm90::host;
using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

#ifndef TP_MOE_W_L2PROMO
#define TP_MOE_W_L2PROMO \
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B  // weight / scale maps (streamed once per call)
#endif

static_assert(K::TAIL,
              "DeepSeek-V4.1: 2304 / TP is not a multiple of 128 (the tail item has its own maps)");
// per-expert bytes of each routed weight tensor (Humming layout)
constexpr int64_t W13_SLICE =
    (int64_t)(2 * K::INTER_M / 64) * K::HLB;  // one k32 slice of the main gate + up rows
constexpr int64_t FC1_W_EXPERT = (int64_t)K::NKT1 * 4 * W13_SLICE,
                  FC1_S_EXPERT = (int64_t)K::NKT1 * 4 * (2 * K::INTER_M);
constexpr int64_t FC1T_W_EXPERT = (int64_t)K::NKT1 * 4 * 2 * K::HLB,
                  FC1T_S_EXPERT = (int64_t)K::NKT1 * 4 * 128;  // tail: 1 block / stack
constexpr int64_t FC1T4_W_EXPERT = (int64_t)K::NKT1 * 4 * K::HLB,
                  FC1T4_S_EXPERT = (int64_t)K::NKT1 * 256;  // TP8 split tail
constexpr int64_t FC2_W_EXPERT = (int64_t)K::NSL2 * K::DIM * 16,
                  FC2_S_EXPERT = (int64_t)K::NSL2 * K::DIM;
constexpr int64_t RS_BYTES = 2 * (int64_t)K::RS_REG * 16,
                  AG_BYTES = 2 * (int64_t)K::AG_REG * 16;  // two epoch-parity halves

H::Dims dims() {
  return H::Dims{K::MAXM, K::MAXE,  K::MAXP, K::INTER,    K::NQG, K::NKT2,
                 K::NMAX, K::NCONS, K::NRT,  K::TOPK_MAX, K::DIM, K::NEXP};
}

int64_t nbytes(const TensorView& t) { return t.numel() * get_element_size(t); }

template <class T>
T* at(const TensorView& ws, int64_t off) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(ws.data_ptr()) + off);
}

void check_weight(const TensorView& t, int64_t n, int64_t per_expert, const char* what) {
  CHECK_CUDA(t);
  CHECK_CONTIGUOUS(t);
  TVM_FFI_ICHECK(nbytes(t) == n * per_expert)
      << what << ": " << nbytes(t) << " bytes, expected " << n << " x " << per_expert;
  TVM_FFI_ICHECK((reinterpret_cast<uint64_t>(t.data_ptr()) & 15) == 0)
      << what << " must be 16-B aligned";
}

}  // namespace

// [MAXM, NDEV, DIM, NEXP, INTER, TOPK, RS_BYTES, AG_BYTES]
Array<int64_t> tp_moe_sm90_constants() {
  return Array<int64_t>{K::MAXM,  K::NDEV,      K::DIM,   K::NEXP,
                        K::INTER, K::GATE_TOPK, RS_BYTES, AG_BYTES};
}

int64_t tp_moe_sm90_workspace_bytes(int64_t num_sms) {
  return H::workspace_layout(dims(), num_sms).bytes;
}

// The routed weights' TMA maps (Humming layout, used in place) -> maps (CPU uint8 [8 * 128]): [0]
// w13 5-D, [1] s13 4-D, [2] w2 4-D, [3] s2 4-D, [4] tail w13 4-D, [5] tail s13 3-D, (TP8) [6] split
// tail w13 4-D, [7] split tail s13 3-D.  The shared expert's weight maps (gate_up [2 INTER][DIM],
// down [DIM][INTER] e4m3, 128B swizzle) -> sh_maps (CPU uint8 [2 * 128]).  Returns the expert
// count.
int64_t tp_moe_sm90_encode_weight_maps(TensorView w1, TensorView s1, TensorView w2, TensorView s2,
                                       TensorView sh_w1, TensorView sh_w2, TensorView w1t,
                                       TensorView s1t, TensorView w1t4, TensorView s1t4,
                                       TensorView maps, TensorView sh_maps) {
  TVM_FFI_ICHECK(maps.device().device_type == kDLCPU &&
                 nbytes(maps) == 8 * (int64_t)sizeof(CUtensorMap))
      << "maps: CPU uint8 [1024]";
  TVM_FFI_ICHECK(sh_maps.device().device_type == kDLCPU &&
                 nbytes(sh_maps) == 2 * (int64_t)sizeof(CUtensorMap))
      << "sh_maps: CPU uint8 [256]";
  const int64_t n = nbytes(w1) / FC1_W_EXPERT;
  TVM_FFI_ICHECK(n >= 1 && n <= K::NEXP) << "expert count out of range";
  check_weight(w1, n, FC1_W_EXPERT, "w13");
  check_weight(s1, n, FC1_S_EXPERT, "w13 scales");
  check_weight(w2, n, FC2_W_EXPERT, "w2");
  check_weight(s2, n, FC2_S_EXPERT, "w2 scales");
  check_weight(w1t, n, FC1T_W_EXPERT, "tail w13");
  check_weight(s1t, n, FC1T_S_EXPERT, "tail w13 scales");
  if (K::T4) {
    check_weight(w1t4, n, FC1T4_W_EXPERT, "split tail w13");
    check_weight(s1t4, n, FC1T4_S_EXPERT, "split tail w13 scales");
  }
  alignas(64) CUtensorMap m[8];
  std::memset(m, 0, sizeof(m));
  const cuuint64_t ne = static_cast<cuuint64_t>(n);
  const cuuint64_t d_w13[5] = {256, 2, K::INTER_M / 128, 2, ne * K::NKT1 * 4};
  const cuuint64_t s_w13[4] = {K::HLB, 2 * K::HLB, (K::INTER_M / 64) * K::HLB,
                               (cuuint64_t)W13_SLICE};
  const cuuint32_t b_w13[5] = {256, 2, 1, 2, 4};
  const cuuint64_t d_s13[4] = {128, K::INTER_M / 128, 2, ne * K::NKT1 * 4},
                   s_s13[3] = {128, K::INTER_M, 2 * K::INTER_M};
  const cuuint32_t b_s13[4] = {128, 1, 2, 4};
  H::encode_map(&m[0], w1.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 5, d_w13, s_w13, b_w13,
                TP_MOE_W_L2PROMO);
  H::encode_map(&m[1], s1.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, d_s13, s_s13, b_s13,
                TP_MOE_W_L2PROMO);
  // w2 / s2 per expert (the k32 slices past NSL2 in the last k-tile read as zeros)
  const cuuint64_t d_w2[4] = {256, K::DIM / 64, K::NSL2, ne},
                   s_w2[3] = {K::HLB, (cuuint64_t)K::DIM * 16, (cuuint64_t)FC2_W_EXPERT};
  const cuuint32_t b_w2[4] = {256, 4, 4, 1};
  const cuuint64_t d_s2[4] = {256, K::DIM / 256, K::NSL2, ne},
                   s_s2[3] = {256, K::DIM, (cuuint64_t)FC2_S_EXPERT};
  const cuuint32_t b_s2[4] = {256, 1, 4, 1};
  H::encode_map(&m[2], w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 4, d_w2, s_w2, b_w2,
                TP_MOE_W_L2PROMO);
  H::encode_map(&m[3], s2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, d_s2, s_s2, b_s2,
                TP_MOE_W_L2PROMO);
  // tail w13 [slice][stack][1 block] (block 1 of each stack out of bounds: zeros), s13
  // [slice][stack][64 rows] (rows 64.. zeros)
  const cuuint64_t d_w1t[4] = {256, 1, 2, ne * K::NKT1 * 4},
                   s_w1t[3] = {K::HLB, K::HLB, 2 * K::HLB};
  const cuuint32_t b_w1t[4] = {256, 2, 2, 4};
  const cuuint64_t d_s1t[3] = {64, 2, ne * K::NKT1 * 4}, s_s1t[2] = {64, 128};
  const cuuint32_t b_s1t[3] = {128, 2, 4};
  H::encode_map(&m[4], w1t.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 4, d_w1t, s_w1t, b_w1t,
                TP_MOE_W_L2PROMO);
  H::encode_map(&m[5], s1t.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, d_s1t, s_s1t, b_s1t,
                TP_MOE_W_L2PROMO);
  if (K::T4) {  // split tail block: [slice][stack][blk][1 KB] (block w = k-tiles 4 j + w), scales
                // [slice][stack][128]
    const cuuint64_t d_w[4] = {256, 2, 2, ne * K::NKT1}, s_w[3] = {K::HLB, 2 * K::HLB, 4 * K::HLB};
    const cuuint32_t b_w[4] = {256, 2, 2, 4};
    const cuuint64_t d_s[3] = {128, 2, ne * K::NKT1}, s_s[2] = {128, 256};
    const cuuint32_t b_s[3] = {128, 2, 4};
    H::encode_map(&m[6], w1t4.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 4, d_w, s_w, b_w,
                  TP_MOE_W_L2PROMO);
    H::encode_map(&m[7], s1t4.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, d_s, s_s, b_s,
                  TP_MOE_W_L2PROMO);
  }
  std::memcpy(maps.data_ptr(), m, sizeof(m));
  for (const TensorView* p : {&sh_w1, &sh_w2}) {
    const TensorView& t = *p;
    CHECK_CUDA(t);
    CHECK_CONTIGUOUS(t);
  }
  TVM_FFI_ICHECK(sh_w1.ndim() == 2 && sh_w1.size(0) == 2 * K::INTER && sh_w1.size(1) == K::DIM &&
                 get_element_size(sh_w1) == 1)
      << "shared gate_up: e4m3 [2 * " << K::INTER << ", " << K::DIM
      << "] (zero-padded to the kernel's INTER)";
  TVM_FFI_ICHECK(sh_w2.ndim() == 2 && sh_w2.size(0) == K::DIM && sh_w2.size(1) == K::INTER &&
                 get_element_size(sh_w2) == 1)
      << "shared down: e4m3 [" << K::DIM << ", " << K::INTER
      << "] (zero-padded to the kernel's INTER)";
  alignas(64) CUtensorMap sm[2];
  H::encode_rows_sw128(&sm[0], sh_w1.data_ptr(), K::DIM, 2 * K::INTER, 128);
  H::encode_rows_sw128(&sm[1], sh_w2.data_ptr(), K::INTER, K::DIM, 128);
  std::memcpy(sh_maps.data_ptr(), sm, sizeof(sm));
  return n;
}

// cudaMemset of a raw device allocation (the CUDA-IPC exchange buffers start zeroed)
void tp_moe_sm90_zero_device_memory(int64_t ptr, int64_t bytes) {
  TVM_FFI_ICHECK(cudaMemset(reinterpret_cast<void*>(ptr), 0, static_cast<size_t>(bytes)) ==
                 cudaSuccess)
      << "cudaMemset";
}

void tp_moe_sm90_run(TensorView x, TensorView maps, TensorView f1, TensorView f2, int64_t n_exp,
                     TensorView sh_maps, TensorView sh_s1, TensorView sh_s2, TensorView f1t,
                     TensorView f1t4, Optional<TensorView> router_logits,
                     Optional<TensorView> gate_bias, Optional<TensorView> gate_bias_vision,
                     Optional<TensorView> input_ids, int64_t image_token_id,
                     Optional<TensorView> topk_ids, Optional<TensorView> topk_weights,
                     TensorView workspace, Array<int64_t> rs_ptrs, Array<int64_t> ag_ptrs,
                     TensorView out, int64_t rank, double factor, bool pdl) {
  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == K::DIM && x.dtype() == dl_bfloat16)
      << "x: bf16 [M, " << K::DIM << "]";
  const int M = static_cast<int>(x.size(0));
  TVM_FFI_ICHECK(M >= 1 && M <= K::MAX_TOKENS) << "M out of range";
  TVM_FFI_ICHECK(out.dtype() == dl_bfloat16 && out.IsContiguous() &&
                 out.numel() == (int64_t)M * K::DIM)
      << "out: bf16 [M, " << K::DIM << "]";
  TVM_FFI_ICHECK((int64_t)rs_ptrs.size() == K::NDEV && (int64_t)ag_ptrs.size() == K::NDEV &&
                 rank >= 0 && rank < K::NDEV)
      << "this build is TP" << K::NDEV;
  const int dev = x.device().device_id;
  const int grid = H::sm_count(dev);
  TVM_FFI_ICHECK(grid <= K::MAXG) << "more SMs than the stream-K bound table";
  const H::Workspace L = H::workspace_layout(dims(), grid);
  TVM_FFI_ICHECK(nbytes(workspace) >= L.bytes) << "workspace too small";
  TVM_FFI_ICHECK(maps.device().device_type == kDLCUDA &&
                 nbytes(maps) == 8 * (int64_t)sizeof(CUtensorMap))
      << "maps";
  TVM_FFI_ICHECK(sh_maps.device().device_type == kDLCPU &&
                 nbytes(sh_maps) == 2 * (int64_t)sizeof(CUtensorMap))
      << "sh_maps";
  TVM_FFI_ICHECK(f1.dtype() == dl_float32 && f2.dtype() == dl_float32 &&
                 f1t.dtype() == dl_float32 && f1.numel() >= n_exp && f2.numel() >= n_exp &&
                 f1t.numel() >= n_exp)
      << "global scales";
  TVM_FFI_ICHECK(!K::T4 || (f1t4.dtype() == dl_float32 && f1t4.numel() >= n_exp))
      << "split tail global scales";
  TVM_FFI_ICHECK(sh_s1.dtype() == dl_float32 && sh_s1.IsContiguous() &&
                 sh_s1.numel() == (2 * K::INTER / 32) * (K::DIM / 32))
      << "shared gate_up scales: fp32 [2 INTER / 32, DIM / 32]";
  TVM_FFI_ICHECK(sh_s2.dtype() == dl_float32 && sh_s2.IsContiguous() &&
                 sh_s2.numel() == (K::DIM / 32) * (K::INTER / 32))
      << "shared down scales: fp32 [DIM / 32, INTER / 32]";

  K::Params P{};
  P.tmaps = reinterpret_cast<const CUtensorMap*>(maps.data_ptr());
  P.fc1_s2 = static_cast<const float*>(f1.data_ptr());
  P.fc1t_s2 = static_cast<const float*>(f1t.data_ptr());
  P.fc1t4_s2 = K::T4 ? static_cast<const float*>(f1t4.data_ptr()) : nullptr;
  P.fc2_s2 = static_cast<const float*>(f2.data_ptr());
  P.n_exp = static_cast<int>(n_exp);
  // fused input quantization: x (BF16) is quantized in the kernel into the workspace (xq / xs),
  // per-token flags tagged with the launch epoch (in-kernel)
  P.xq = at<uint8_t>(workspace, L.xq);
  P.xs = at<float>(workspace, L.xs);
  P.xb = static_cast<const __nv_bfloat16*>(x.data_ptr());
  P.xflag = at<unsigned>(workspace, L.tok_flags);
  P.dense = M <= 16 ? 1 : 0;
  P.oneshot = (M <= TP_MOE_ONESHOT_MAXM_DEF && M <= K::OS_MAXM) ? 1 : 0;
  P.groups =
      M >= 16 ? 2
              : 1;  // two FC1 panel groups: group 0's FC1 -> FC2 handoff hides under group 1's FC1
  P.md = M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 24 ? 24 : 32));
  if (P.dense) {
    const cuuint64_t d[2] = {K::DIM, (cuuint64_t)M}, s[1] = {K::DIM};
    const cuuint32_t box[2] = {128, (cuuint32_t)P.md}, estr[2] = {1, 1};
    const CUresult r =
        H::encode_fn()(&P.xmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(P.xq), d, s,
                       box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(r == CUDA_SUCCESS) << "x tensor map: " << static_cast<int>(r);
  }
  // routing: the fused gate on router logits (ids / weights are then workspace outputs) or the
  // given ids / weights
  P.gcnt = at<unsigned>(workspace, L.gcnt);
  if (router_logits.has_value()) {
    TVM_FFI_ICHECK(gate_bias.has_value() && !topk_ids.has_value())
        << "gate entry: router_logits + gate_bias";
    const TensorView lg = router_logits.value(), gb = gate_bias.value();
    TVM_FFI_ICHECK(P.n_exp == K::NEXP) << "fused gate: all " << K::NEXP << " experts";
    TVM_FFI_ICHECK(lg.dtype() == dl_float32 && lg.IsContiguous() && lg.ndim() == 2 &&
                   lg.size(0) == M && lg.size(1) == K::NEXP)
        << "router_logits: fp32 [M, " << K::NEXP << "]";
    TVM_FFI_ICHECK(gb.dtype() == dl_float32 && gb.IsContiguous() && gb.numel() == K::NEXP)
        << "gate_bias: fp32 [" << K::NEXP << "]";
    P.glog = static_cast<const float*>(lg.data_ptr());
    P.gbias = static_cast<const float*>(gb.data_ptr());
    TVM_FFI_ICHECK(gate_bias_vision.has_value() == input_ids.has_value())
        << "vision gate: gate_bias_vision + input_ids";
    if (input_ids.has_value()) {  // image tokens (input_ids[t] == image_token_id) select their
                                  // experts with the vision bias
      const TensorView vb = gate_bias_vision.value(), ids = input_ids.value();
      TVM_FFI_ICHECK(vb.dtype() == dl_float32 && vb.IsContiguous() && vb.numel() == K::NEXP)
          << "gate_bias_vision: fp32 [" << K::NEXP << "]";
      TVM_FFI_ICHECK(ids.dtype() == dl_int32 && ids.IsContiguous() && ids.numel() == M)
          << "input_ids: int32 [M]";
      P.gbias_v = static_cast<const float*>(vb.data_ptr());
      P.vids = static_cast<const int*>(ids.data_ptr());
      P.image_id = static_cast<int>(image_token_id);
    }
    P.ids = at<int>(workspace, L.gids);
    P.rw = at<float>(workspace, L.grw);
    P.topk = K::GATE_TOPK;
  } else {
    TVM_FFI_ICHECK(topk_ids.has_value() && topk_weights.has_value())
        << "ids entry: topk_ids + topk_weights";
    const TensorView ids = topk_ids.value(), rw = topk_weights.value();
    TVM_FFI_ICHECK(ids.dtype() == dl_int32 && rw.dtype() == dl_float32 && ids.IsContiguous() &&
                   rw.IsContiguous() && ids.ndim() == 2 && ids.size(0) == M && rw.ndim() == 2 &&
                   rw.size(0) == M && rw.size(1) == ids.size(1) && ids.size(1) >= 1 &&
                   ids.size(1) <= K::TOPK_MAX)
        << "topk_ids int32 / topk_weights fp32 [M, k <= 8]";
    P.ids = static_cast<const int*>(ids.data_ptr());
    P.rw = static_cast<const float*>(rw.data_ptr());
    P.topk = static_cast<int>(ids.size(1));
  }
  TVM_FFI_ICHECK((int64_t)P.topk * K::NQG * K::NKT1 >= grid)
      << "stream-K: every CTA needs >= 1 FC1 k-tile";
  // fused shared expert: FP8 gate_up / down with 32 x 32 block scales; its BF16 output is added in
  // T1
  std::memcpy(&P.w1s, static_cast<const uint8_t*>(sh_maps.data_ptr()), sizeof(CUtensorMap));
  std::memcpy(&P.w2s, static_cast<const uint8_t*>(sh_maps.data_ptr()) + sizeof(CUtensorMap),
              sizeof(CUtensorMap));
  P.sb1 = static_cast<const float*>(sh_s1.data_ptr());
  P.sb2 = static_cast<const float*>(sh_s2.data_ptr());
  P.xqg = at<uint8_t>(workspace, L.xqg);
  P.xsT = at<float>(workspace, L.xsT);
  P.hs4 = at<float>(workspace, L.hs4);
  P.shout = at<__nv_bfloat16>(workspace, L.shout);
  H::encode_rows_sw128(&P.xgs, P.xqg, K::DIM, (uint64_t)M, K::NMAX);
  P.factor = static_cast<float>(factor);
  P.act = at<__nv_bfloat16>(workspace, L.act);
  P.amax = at<float>(workspace, L.amax);
  P.part = at<float>(workspace, L.part);
  P.part_flags = at<unsigned>(workspace, L.part_flags);
  P.rt_cnt = at<unsigned>(workspace, L.rt_cnt);
  P.work = at<unsigned>(workspace, L.work);
  P.outp = at<__nv_bfloat16>(workspace, L.outp);
  P.hq = at<uint8_t>(workspace, L.hq);
  P.hs = at<float>(workspace, L.hs);
  P.h_flags = at<unsigned>(workspace, L.h_flags);
  P.pan_cnt = at<unsigned>(workspace, L.pan_cnt);
  P.xg = at<uint8_t>(workspace, L.xg);
  P.xg_cnt = at<unsigned>(workspace, L.xg_cnt);
  P.fc2_next = at<unsigned>(workspace, L.fc2_next);
  for (int d = 0; d < K::NDEV; ++d) {
    P.rs[d] = reinterpret_cast<uint4*>(rs_ptrs[d]);
    P.ag[d] = reinterpret_cast<uint4*>(ag_ptrs[d]);
  }
  P.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  P.M = M;
  P.my_rank = static_cast<int>(rank);

  static bool attr[64] = {false};
  if (!attr[dev]) {
    TVM_FFI_ICHECK(cudaFuncSetAttribute(K::tp_moe_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        K::SMEM_ALLOC) == cudaSuccess)
        << "smem attribute";
    attr[dev] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(K::NTHREADS);
  cfg.dynamicSmemBytes = K::SMEM_ALLOC;
  cfg.stream = get_stream(x.device());
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = la;
  cfg.numAttrs = pdl ? 1 : 0;
  TVM_FFI_ICHECK(cudaLaunchKernelEx(&cfg, K::tp_moe_kernel, P) == cudaSuccess) << "launch";
  const cudaError_t e = cudaGetLastError();
  TVM_FFI_ICHECK(e == cudaSuccess) << "launch failed: " << cudaGetErrorString(e);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(constants, tp_moe_sm90_constants);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(workspace_bytes, tp_moe_sm90_workspace_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(encode_weight_maps, tp_moe_sm90_encode_weight_maps);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(zero_device_memory, tp_moe_sm90_zero_device_memory);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tp_moe_sm90_run);
