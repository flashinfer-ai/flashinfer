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
 *
 * Launcher of the SM90 NVFP4 W4A16 fused MoE large batch (prefill) kernel and
 * its finalize kernel. See sm90_nvfp4_w4a16_moe_decode.cu for the generated
 * configuration include.
 */
#include <cuda.h>

#include <algorithm>
#include <cstring>

#include "flashinfer/fused_moe/sm90_nvfp4_w4a16/common.cuh"
// Generated: the problem shape macros and the wgmma_rs specializations, which must
// precede the kernels' instantiations below.
#include "sm90_nvfp4_w4a16_moe_config.inc"
// clang-format off
#include "flashinfer/fused_moe/sm90_nvfp4_w4a16/prefill.cuh"
// clang-format on
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm90_w4a16_moe {
namespace {

constexpr int kH = FI_W4A16_HIDDEN, kI = FI_W4A16_INTER;
constexpr bool kPN = FI_W4A16_POST_NORM != 0;
using Shape = pf::Shape<kH, kI, kPN>;
// item widths in widx() order
constexpr int kWidths[pf::NW] = {32, 64, 96, 128, 160, 192, 224, 256, 240};
// activation / intermediate tensor maps in the device map buffer
constexpr int kMapTx2 = 0, kMapTh = 2, kMapTh2 = kMapTh + pf::NW, kMapTxh = kMapTh2 + 2,
              kMapThh = kMapTxh + pf::NW, kNumMaps = kMapThh + pf::NW;

void encode_bf16_map(CUtensorMap* m, void* ptr, int64_t rows, int64_t cols, int box_rows) {
  CUresult res;
  if (box_rows < 0) {
    // 3D view [cols / 64][rows][64]: two 64-column chunks per load
    cuuint64_t dims[3] = {64, (cuuint64_t)rows, (cuuint64_t)(cols / 64)};
    cuuint64_t strides[2] = {(cuuint64_t)(cols * 2), 128};
    cuuint32_t box[3] = {64, (cuuint32_t)(-box_rows), 2};
    cuuint32_t es[3] = {1, 1, 1};
    res = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box,
                                 es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
    cuuint64_t strides[1] = {(cuuint64_t)(cols * 2)};
    cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
    cuuint32_t es[2] = {1, 1};
    res = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box,
                                 es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  TVM_FFI_ICHECK(res == CUDA_SUCCESS) << "cuTensorMapEncodeTiled (activations) failed: " << res;
}

// Packed weights as 32-bit words: [block][row group 16][k sub-block 4][128 words].
void encode_weight_map(CUtensorMap* m, const void* ptr, int64_t nblocks, int kblocks, int groups) {
  cuuint64_t dims[4] = {128, 4, 16, (cuuint64_t)nblocks};
  cuuint64_t strides[3] = {512, 2048, (cuuint64_t)kWeightBlockBytes};
  cuuint32_t box[4] = {128, (cuuint32_t)kblocks, (cuuint32_t)groups, 1};
  cuuint32_t es[4] = {1, 1, 1, 1};
  CUresult res = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, const_cast<void*>(ptr), dims, strides, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS) << "cuTensorMapEncodeTiled (weights) failed: " << res;
}

int num_sms(int device) {
  int sms = 0;
  cudaError_t status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaDeviceGetAttribute failed: " << cudaGetErrorString(status);
  return sms;
}

}  // namespace

int64_t sm90_nvfp4_w4a16_moe_prefill_counter_words() { return pf::kCounterWords; }
int64_t sm90_nvfp4_w4a16_moe_prefill_map_bytes() { return kNumMaps * sizeof(pf::Tmap); }
// Rows of the expert-grouped activation (`xt`, [rows, hidden]) and intermediate
// ([rows, intermediate]) workspaces for chunks of `chunk_tokens` tokens.
int64_t sm90_nvfp4_w4a16_moe_prefill_slots(int64_t chunk_tokens) {
  return 8 * chunk_tokens + pf::kSlotPad;
}

// Encodes the tensor maps of the persistent `xt` / `h` workspaces into `maps`
// (device memory). Call once per workspace, outside CUDA graph capture.
void sm90_nvfp4_w4a16_moe_prefill_init(TensorView xt, TensorView h, TensorView maps) {
  check_cuda_tensor(xt, "xt");
  check_dtype(xt, dl_bfloat16, "xt");
  check_dtype(h, dl_bfloat16, "h");
  check_dtype(maps, dl_uint8, "maps");
  TVM_FFI_ICHECK(xt.size(1) == kH && h.size(1) == kI && xt.size(0) == h.size(0))
      << "xt must be [slots, hidden], h [slots, intermediate]";
  TVM_FFI_ICHECK_GE(maps.numel(), (int64_t)(kNumMaps * sizeof(pf::Tmap)));
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(maps.data_ptr()) % 64, 0u)
      << "maps must be 64-byte aligned";
  const int64_t rows = xt.size(0);
  CUtensorMap host[kNumMaps];
  std::memset(host, 0, sizeof(host));
  encode_bf16_map(&host[kMapTx2 + 0], xt.data_ptr(), rows, kH, -32);
  encode_bf16_map(&host[kMapTx2 + 1], xt.data_ptr(), rows, kH, -64);
  encode_bf16_map(&host[kMapTh2 + 0], h.data_ptr(), rows, kI, -32);
  encode_bf16_map(&host[kMapTh2 + 1], h.data_ptr(), rows, kI, -64);
  for (int i = 0; i < pf::NW; ++i) {
    encode_bf16_map(&host[kMapTh + i], h.data_ptr(), rows, kI, kWidths[i]);
    encode_bf16_map(&host[kMapTxh + i], xt.data_ptr(), rows, kH, kWidths[i] / 2);
    encode_bf16_map(&host[kMapThh + i], h.data_ptr(), rows, kI, kWidths[i] / 2);
  }
  static_assert(sizeof(CUtensorMap) == sizeof(pf::Tmap), "tensor map size");
  cudaError_t status = cudaMemcpy(maps.data_ptr(), host, sizeof(host), cudaMemcpyHostToDevice);
  TVM_FFI_ICHECK(status == cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(status);
}

void sm90_nvfp4_w4a16_moe_prefill_run(TensorView hidden_states, TensorView topk_ids,
                                      TensorView topk_weights, TensorView w13, TensorView w13_scale,
                                      TensorView w13_up_scale, TensorView w2, TensorView w2_scale,
                                      TensorView norm_weight, TensorView out, TensorView recv,
                                      TensorView xt, TensorView maps, TensorView rev,
                                      TensorView unit_counts, TensorView counters,
                                      int64_t num_experts, int64_t chunk_tokens, double eps,
                                      bool enable_pdl) {
  check_cuda_tensor(hidden_states, "hidden_states");
  check_dtype(hidden_states, dl_bfloat16, "hidden_states");
  check_dtype(topk_ids, dl_int32, "topk_ids");
  check_dtype(topk_weights, dl_float32, "topk_weights");
  check_dtype(w13, dl_uint8, "w13");
  check_dtype(w2, dl_uint8, "w2");
  check_dtype(out, dl_bfloat16, "out");
  check_dtype(recv, dl_bfloat16, "recv");
  check_dtype(counters, dl_int32, "counters");
  for (const TensorView* t :
       {&hidden_states, &topk_ids, &topk_weights, &w13, &w13_scale, &w13_up_scale, &w2, &w2_scale,
        &norm_weight, &out, &recv, &xt, &maps, &rev, &unit_counts, &counters}) {
    check_contiguous(*t, "sm90_nvfp4_w4a16_moe tensor");
    check_same_device(*t, hidden_states, "sm90_nvfp4_w4a16_moe tensor", "hidden_states");
  }
  const int64_t M = hidden_states.size(0);
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), kH) << "hidden size of the module";
  TVM_FFI_ICHECK(topk_ids.size(0) == M && topk_ids.size(1) == kTopKSlots)
      << "topk_ids must be [num_tokens, 8]";
  TVM_FFI_ICHECK(topk_weights.size(0) == M && topk_weights.size(1) == kTopKSlots)
      << "topk_weights must be [num_tokens, 8]";
  TVM_FFI_ICHECK(out.size(0) == M && out.size(1) == kH);
  TVM_FFI_ICHECK_GE(recv.numel(), (int64_t)kTopKSlots * M * kH) << "recv must hold [8, M, hidden]";
  TVM_FFI_ICHECK(num_experts >= 1 && num_experts <= kMaxLocalExperts);
  TVM_FFI_ICHECK(chunk_tokens >= 128 && chunk_tokens % 128 == 0);
  TVM_FFI_ICHECK_GE(xt.size(0), sm90_nvfp4_w4a16_moe_prefill_slots(chunk_tokens))
      << "xt has fewer rows than the chunk needs";
  TVM_FFI_ICHECK_GE(rev.numel(), (int64_t)(pf::NE + kTopKSlots) * M);
  TVM_FFI_ICHECK_GE(unit_counts.numel(), (M + 127) / 128 * 64);
  TVM_FFI_ICHECK_GE(counters.numel(), (int64_t)pf::kCounterWords);
  TVM_FFI_ICHECK_LT(M, (int64_t)1 << 28) << "token count";
  if (M == 0) return;
  const int device = hidden_states.device().device_id;
  const int grid = num_sms(device) & ~1;  // CTA pairs
  cudaStream_t stream = get_stream(hidden_states.device());

  pf::Params p;
  std::memset(&p, 0, sizeof(p));
  const int64_t nb13 = w13.numel() / kWeightBlockBytes, nb2 = w2.numel() / kWeightBlockBytes;
  const int kb_rg[3][2] = {{1, 8}, {1, 16}, {2, 16}};
  for (int v = 0; v < 3; ++v) {
    encode_weight_map(reinterpret_cast<CUtensorMap*>(&p.wm[2 * v]), w13.data_ptr(), nb13,
                      kb_rg[v][0], kb_rg[v][1]);
    encode_weight_map(reinterpret_cast<CUtensorMap*>(&p.wm[2 * v + 1]), w2.data_ptr(), nb2,
                      kb_rg[v][0], kb_rg[v][1]);
  }
  const pf::Tmap* mp = static_cast<const pf::Tmap*>(maps.data_ptr());
  p.tx2 = mp + kMapTx2;
  p.th = mp + kMapTh;
  p.th2 = mp + kMapTh2;
  p.txh = mp + kMapTxh;
  p.thh = mp + kMapThh;
  p.ids = static_cast<const int*>(topk_ids.data_ptr());
  p.w13 = static_cast<const u8*>(w13.data_ptr());
  p.w2 = static_cast<const u8*>(w2.data_ptr());
  p.gs13 = static_cast<const float*>(w13_scale.data_ptr());
  p.gs13_up = static_cast<const float*>(w13_up_scale.data_ptr());
  p.gs2 = static_cast<const float*>(w2_scale.data_ptr());
  p.nw = static_cast<const float*>(norm_weight.data_ptr());
  p.xt = static_cast<u16*>(xt.data_ptr());
  p.ctr = static_cast<int*>(counters.data_ptr());
  p.rev = static_cast<int*>(rev.data_ptr());
  p.ucnt = static_cast<int*>(unit_counts.data_ptr());
  p.M = static_cast<int>(M);
  p.cap = static_cast<int>(sm90_nvfp4_w4a16_moe_prefill_slots(chunk_tokens));
  p.ne = static_cast<int>(num_experts);
  p.x = static_cast<const u16*>(hidden_states.data_ptr());
  p.recv = static_cast<u16*>(recv.data_ptr());
  // one launch covers the batch: count and list routes in one pass, longer GEMM2 lag
  p.cl = M <= chunk_tokens;
  p.lag = M <= chunk_tokens ? 16 : 6;

  auto kernel = pf::moe_prefill_kernel<Shape>;
  cudaError_t status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kSmemBytes);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);
  cudaLaunchConfig_t config{};
  cudaLaunchAttribute attrs[2];
  config.gridDim = dim3(grid);
  config.blockDim = dim3(384);
  config.dynamicSmemBytes = pf::kSmemBytes;
  config.stream = stream;
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  // A launch covers the units whose routes fit `xt`; the next one resumes at the cursor the
  // last CTA leaves in the counters. ceil(M / chunk) launches cover any routing.
  const int64_t launches = (M + chunk_tokens - 1) / chunk_tokens;
  for (int64_t i = 0; i < launches; ++i) {
    p.first = i == 0;
    config.numAttrs = (enable_pdl && i > 0) ? 2 : 1;
    status = cudaLaunchKernelEx(&config, kernel, p);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "sm90_nvfp4_w4a16_moe prefill launch failed: " << cudaGetErrorString(status);
  }

  auto fin = pf::moe_finalize_kernel<kH, kPN>;
  constexpr int fin_smem = pf::finalize_smem_bytes<kH>();
  status = cudaFuncSetAttribute(fin, cudaFuncAttributeMaxDynamicSharedMemorySize, fin_smem);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);
  const int fin_grid = static_cast<int>(std::min<int64_t>(M, 2 * num_sms(device)));
  pf::moe_finalize_kernel<kH, kPN><<<fin_grid, 256, fin_smem, stream>>>(
      static_cast<const u16*>(recv.data_ptr()), static_cast<const int*>(topk_ids.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()), static_cast<u16*>(out.data_ptr()),
      static_cast<int>(M), static_cast<int>(num_experts), static_cast<float>(eps));
  status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "sm90_nvfp4_w4a16_moe finalize launch failed: " << cudaGetErrorString(status);
}

}  // namespace sm90_w4a16_moe
}  // namespace flashinfer
