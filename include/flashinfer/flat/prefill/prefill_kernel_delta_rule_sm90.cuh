/*
 * Copyright (c) 2025 by FlashInfer team.
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
#pragma once

#include <cstdio>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/device_memory.h"
#include "flashinfer/flat/common.hpp"
#include "flashinfer/flat/hopper/device/device_universal.hpp"
#include "flashinfer/flat/hopper/kernel/flat_kernel_builder_delta_rule.hpp"

namespace flat {

using namespace cute;

template <bool IsGVA, bool NeedsBeta, bool NeedsAlpha, bool InitStateFromInput, typename ArchTag,
          typename TO, typename TQKV, typename TState>
void launch_delta_rule_prefill_kernel_gbai(
    cudaStream_t stream, TO* output, TState* output_state, TQKV const* q, TQKV const* k,
    TQKV const* v, TState const* input_state, float const* alpha, float const* beta,
    int64_t const* cu_seqlens, uint8_t* workspace_buffer, int32_t num_seqs, int32_t num_q_heads,
    int32_t num_k_heads, int32_t num_v_heads, int32_t num_o_heads, int32_t head_size,
    int64_t total_seqlen, float scale, int32_t sm_count) {
#if defined(FLAT_SM90A_ENABLED)
  constexpr bool HopperSupported = true;
#else
  constexpr bool HopperSupported = false;
#endif

  if constexpr (HopperSupported) {
    static_assert(std::is_same_v<TQKV, TO>);

    using namespace flat::kernel;
    using T = map_to_cutlass_t<TQKV>;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = sm_count;

    using Options = decltype([&]() {
      constexpr auto options_0 = DefaultOptions{};
      constexpr auto options_1 =
          add_option(Option<Tag::kIsDeltaRule, cute::true_type>{}, options_0);
      constexpr auto options_2 = add_option(
          Option<Tag::kIsGVA, std::conditional_t<IsGVA, cute::true_type, cute::false_type>>{},
          options_1);
      constexpr auto options_3 =
          add_option(Option<Tag::kNeedsBeta,
                            std::conditional_t<NeedsBeta, cute::true_type, cute::false_type>>{},
                     options_2);
      constexpr auto options_4 =
          add_option(Option<Tag::kNeedsAlpha,
                            std::conditional_t<NeedsAlpha, cute::true_type, cute::false_type>>{},
                     options_3);
      constexpr auto options_5 = add_option(
          Option<Tag::kInitStateFromInput,
                 std::conditional_t<InitStateFromInput, cute::true_type, cute::false_type>>{},
          options_4);
      return options_5;
    }());

    using TileShape = Shape<_64, _64, _128>;
    using Scheduler = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using Operation = cutlass::device::Universal<typename flat::kernel::FlatBuilderDeltaRule<
        T, float, float, TileShape,
        /*LayoutQ=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutK=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutV=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutO=*/cute::tuple<int64_t, _1, int32_t>, Scheduler, Options>::Kernel>;
    using Arguments = typename Operation::Arguments;

    // NOTE: LayoutQ/K/V in (seq, head_size, (b,h)) coordinate semantics

    int32_t num_sab_heads = std::max(num_q_heads, num_v_heads);

    int32_t q_tok_stride = num_q_heads * head_size;
    int32_t o_tok_stride = num_o_heads * head_size;
    int32_t k_tok_stride = num_k_heads * head_size;
    int32_t v_tok_stride = num_v_heads * head_size;

    int32_t q_head_stride = head_size;
    int32_t o_head_stride = head_size;
    int32_t k_head_stride = head_size;
    int32_t v_head_stride = head_size;

    Operation op;
    Arguments arguments{.problem_size =
                            {
                                .cu_seqlens = cu_seqlens,
                                .total_seqlen = total_seqlen,
                                .num_seqs = num_seqs,
                                .num_q_heads = num_q_heads,
                                .num_k_heads = num_k_heads,
                                .num_v_heads = num_v_heads,
                                .num_o_heads = num_o_heads,
                                .num_sab_heads = num_sab_heads,
                                .head_size = head_size,
                            },
                        .mainloop =
                            {
                                // clang-format off
                .ptr_Q = (T*)q,      .dQ = {q_tok_stride, _1{}, q_head_stride},
                .ptr_K = (T*)k,      .dK = {k_tok_stride, _1{}, k_head_stride},
                .ptr_V = (T*)v,      .dV = {v_tok_stride, _1{}, v_head_stride},
                .ptr_O = (T*)output, .dO = {o_tok_stride, _1{}, o_head_stride},
                .ptr_output_state = (float*)output_state,
                .ptr_input_state  = (float*)input_state,
                .scale = scale,
                .alpha_ptr = alpha, .alpha_stride = {num_sab_heads, 1},
                .beta_ptr  = beta,  .beta_stride  = {num_sab_heads, 1},
        },  // clang-format on
                        .hw_info = hw_info};

    cutlass::Status status;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("can_implement failed");
    }

    status = op.initialize(arguments, workspace_buffer, stream);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("initialize failed");
    }

    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("run failed");
    }
  } else {
    throw std::runtime_error("hopper not supported");
  }
}

}  // namespace flat
