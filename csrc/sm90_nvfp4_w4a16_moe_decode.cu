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
 * Launcher of the SM90 NVFP4 W4A16 fused MoE small / medium batch kernel.
 * The JIT generator (flashinfer/jit/sm90_nvfp4_w4a16_moe.py) writes
 * sm90_nvfp4_w4a16_moe_config.inc next to this file: the problem shape
 * (FI_W4A16_HIDDEN, FI_W4A16_INTER, FI_W4A16_POST_NORM) and the wgmma_rs
 * specializations.
 */
#include <algorithm>
#include <type_traits>

#include "flashinfer/fused_moe/sm90_nvfp4_w4a16/common.cuh"
// Generated: the problem shape macros and the wgmma_rs specializations, which must
// precede the kernels' instantiations below.
#include "sm90_nvfp4_w4a16_moe_config.inc"
// clang-format off
#include "flashinfer/fused_moe/sm90_nvfp4_w4a16/decode.cuh"
// clang-format on
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm90_w4a16_moe {
namespace {

constexpr int kH = FI_W4A16_HIDDEN, kI = FI_W4A16_INTER;
constexpr bool kPN = FI_W4A16_POST_NORM != 0;
constexpr int kMaxSmem = 232448;
constexpr int kNkc1 = kH / 256, kNkc2 = kI / 256;

// Ring depth of a configuration, reduced when the K extent is short: every work
// item needs at least NS stages (the first item of the 1..32-token kernel one more).
template <int RW, int NT, int NS, int MW>
constexpr int ring_depth() {
  constexpr int KW = 8 / RW;
  constexpr bool IT2K = NT == 4 && MW == 1;
  return std::min({NS, kNkc1 / KW - (IT2K ? 1 : 0), kNkc2 / KW});
}
// The 128-row-block configuration (two consumer warps per 16-row group, K split
// across them) needs an even number of 256-wide K chunks in both GEMMs.
constexpr bool kUseHalfBlocks =
    kNkc1 % 2 == 0 && kNkc2 % 2 == 0 && kNkc1 / 2 - 1 >= 2 && kNkc2 / 2 >= 2;

// Tuned configurations (measured on H200 for hidden = intermediate = 4096, 32 local
// experts, top-8). Columns: max tokens, RW, NT, NS, MW, dynamic shared memory.
struct Config {
  int max_tokens, rw, nt, ns, mw, smem;
};

template <int RW, int NT, int NS, int MW>
struct Cfg {
  static constexpr int kNs = ring_depth<RW, NT, NS, MW>();
  static_assert(kNs >= 2, "hidden / intermediate size too small for this configuration");
  using Layout = dec::DecodeLayout<RW, NT, kNs, MW>;
  using Split = dec::DecodeSplit<kH, kI, RW, kNs>;
  static constexpr int T = 8 * NT;
  static constexpr int KW = 8 / RW;
  static constexpr int XP = 2 * 256 * KW + 16;

  // Transient buffers of one launch with M tokens (bytes, 256-aligned each).
  static void buffers(int M, int64_t* sizes) {
    const int64_t pmax = 8LL * M, tiles = kMaxLocalExperts + (pmax + T - 1) / T;
    auto al = [](int64_t b) { return (b + 255) / 256 * 256; };
    sizes[0] = al(tiles * Split::ST1H * T * XP);                          // inter
    sizes[1] = al(tiles * Split::ST1X * T * XP);                          // xs
    sizes[2] = al(MW < 32 && Split::KSX > 1 ? 4LL * pmax * kH * 4 : 16);  // ypart
    sizes[3] = al(pmax * kH * 2);                                         // yh
  }

  static void launch(const dec::Params& p, int smem_cfg, int grid, bool pdl, cudaStream_t stream) {
    auto kernel = dec::moe_decode_kernel<kH, kI, kPN, RW, NT, kNs, MW>;
    static_assert(Layout::kSmemBytes <= kMaxSmem, "shared memory");
    const int smem = std::max(smem_cfg, Layout::kSmemBytes);
    cudaError_t status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "cudaFuncSetAttribute failed: " << cudaGetErrorString(status);
    cudaLaunchConfig_t config{};
    cudaLaunchAttribute attrs[1];
    config.gridDim = dim3(grid);
    config.blockDim = dim3(384);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = pdl ? 1 : 0;
    status = cudaLaunchKernelEx(&config, kernel, p);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "sm90_nvfp4_w4a16_moe decode launch failed: " << cudaGetErrorString(status);
  }
};

// <= 16 tokens: the 128-row-block configuration where the shape allows it. Chosen by type
// so that the configuration that is not used is never instantiated.
using SmallestCfg = std::conditional_t<kUseHalfBlocks, Cfg<4, 4, 4, 1>, Cfg<8, 4, 4, 1>>;

// Dispatch by token count; F(Cfg<...>, smem) is called with the chosen configuration.
template <class F>
void dispatch(int M, F&& f) {
  if (M <= 16) {
    f(SmallestCfg{}, 232448);
  } else if (M <= 32) {
    f(Cfg<8, 4, 4, 1>{}, 232448);
  } else if (M <= 64) {
    f(Cfg<8, 4, 4, 2>{}, 232448);
  } else if (M <= 128) {
    f(Cfg<8, 8, 3, 4>{}, 228352);
  } else if (M <= 256) {
    f(Cfg<8, 8, 3, 8>{}, 232448);
  } else if (M <= 512) {
    f(Cfg<8, 8, 3, 16>{}, 228352);
  } else if (M <= 1024) {
    f(Cfg<8, 8, 3, 32>{}, 229376);
  } else {
    TVM_FFI_THROW(ValueError) << "sm90_nvfp4_w4a16_moe decode kernel takes at most 1024 tokens per "
                                 "launch, got "
                              << M;
  }
}

int num_sms(int device) {
  int sms = 0;
  cudaError_t status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaDeviceGetAttribute failed: " << cudaGetErrorString(status);
  return sms;
}

}  // namespace

// Bytes of the transient buffer one launch with `num_tokens` tokens needs.
int64_t sm90_nvfp4_w4a16_moe_decode_buffer_bytes(int64_t num_tokens) {
  int64_t sizes[4] = {0, 0, 0, 0};
  dispatch(static_cast<int>(num_tokens),
           [&](auto cfg, int) { decltype(cfg)::buffers(static_cast<int>(num_tokens), sizes); });
  return sizes[0] + sizes[1] + sizes[2] + sizes[3];
}

int64_t sm90_nvfp4_w4a16_moe_decode_counter_words() { return dec::kCounterWords; }

int64_t sm90_nvfp4_w4a16_moe_decode_scratch_bytes(int64_t device) {
  return static_cast<int64_t>(num_sms(static_cast<int>(device))) * 64 * 1040;
}

void sm90_nvfp4_w4a16_moe_decode_run(TensorView hidden_states, TensorView topk_ids,
                                     TensorView topk_weights, TensorView w13, TensorView w13_scale,
                                     TensorView w13_up_scale, TensorView w2, TensorView w2_scale,
                                     TensorView norm_weight, TensorView out, TensorView buffer,
                                     TensorView counters, TensorView scratch, int64_t num_experts,
                                     double eps, bool enable_pdl) {
  check_cuda_tensor(hidden_states, "hidden_states");
  check_dtype(hidden_states, dl_bfloat16, "hidden_states");
  check_dtype(topk_ids, dl_int32, "topk_ids");
  check_dtype(topk_weights, dl_float32, "topk_weights");
  check_dtype(w13, dl_uint8, "w13");
  check_dtype(w2, dl_uint8, "w2");
  check_dtype(out, dl_bfloat16, "out");
  check_dtype(counters, dl_int32, "counters");
  for (const TensorView* t :
       {&hidden_states, &topk_ids, &topk_weights, &w13, &w13_scale, &w13_up_scale, &w2, &w2_scale,
        &norm_weight, &out, &buffer, &counters, &scratch}) {
    check_contiguous(*t, "sm90_nvfp4_w4a16_moe tensor");
    check_same_device(*t, hidden_states, "sm90_nvfp4_w4a16_moe tensor", "hidden_states");
  }
  const int M = static_cast<int>(hidden_states.size(0));
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), kH) << "hidden size of the module";
  TVM_FFI_ICHECK(topk_ids.size(0) == M && topk_ids.size(1) == kTopKSlots)
      << "topk_ids must be [num_tokens, 8] (pad smaller top-k with -1)";
  TVM_FFI_ICHECK(topk_weights.size(0) == M && topk_weights.size(1) == kTopKSlots)
      << "topk_weights must be [num_tokens, 8]";
  TVM_FFI_ICHECK(out.size(0) == M && out.size(1) == kH) << "out must be [num_tokens, hidden]";
  TVM_FFI_ICHECK(num_experts >= 1 && num_experts <= kMaxLocalExperts)
      << "num_experts must be in [1, 32], got " << num_experts;
  TVM_FFI_ICHECK_GE(counters.numel(), dec::kCounterWords) << "counter workspace too small";
  if (M == 0) return;
  const int device = hidden_states.device().device_id;
  const int sms = num_sms(device);
  TVM_FFI_ICHECK_GE(scratch.numel() * get_element_size(scratch), (int64_t)sms * 64 * 1040)
      << "scratch workspace too small";
  cudaStream_t stream = get_stream(hidden_states.device());

  dispatch(M, [&](auto cfg, int smem_cfg) {
    using C = decltype(cfg);
    int64_t sizes[4];
    C::buffers(M, sizes);
    TVM_FFI_ICHECK_GE(buffer.numel() * get_element_size(buffer),
                      sizes[0] + sizes[1] + sizes[2] + sizes[3])
        << "buffer too small (see decode_buffer_bytes)";
    uint8_t* base = static_cast<uint8_t*>(buffer.data_ptr());
    dec::Params p{};
    p.x = static_cast<const u16*>(hidden_states.data_ptr());
    p.ids = static_cast<const int*>(topk_ids.data_ptr());
    p.topk_w = static_cast<const float*>(topk_weights.data_ptr());
    p.M = M;
    p.w13 = static_cast<const u32*>(w13.data_ptr());
    p.gs13 = static_cast<const float*>(w13_scale.data_ptr());
    p.w2 = static_cast<const u32*>(w2.data_ptr());
    p.gs2 = static_cast<const float*>(w2_scale.data_ptr());
    p.nw1 = static_cast<const float*>(norm_weight.data_ptr());
    p.inter = reinterpret_cast<u16*>(base);
    p.xs = reinterpret_cast<u16*>(base + sizes[0]);
    p.ypart = reinterpret_cast<float*>(base + sizes[0] + sizes[1]);
    p.yh = reinterpret_cast<u16*>(base + sizes[0] + sizes[1] + sizes[2]);
    p.out = static_cast<u16*>(out.data_ptr());
    p.ctr = static_cast<int*>(counters.data_ptr());
    p.pmax = 8 * M;
    p.scr = static_cast<float*>(scratch.data_ptr());
    p.gs13_up = static_cast<const float*>(w13_up_scale.data_ptr());
    p.num_experts = static_cast<int>(num_experts);
    p.eps = static_cast<float>(eps);
    // programmatic dependent launch only pays off for the short (<= 128 token) launches
    C::launch(p, smem_cfg, sms, enable_pdl && C::Layout::kMaxTokens <= 128, stream);
  });
}

}  // namespace sm90_w4a16_moe
}  // namespace flashinfer
