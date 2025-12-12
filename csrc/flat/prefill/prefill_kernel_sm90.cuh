#pragma once

#include <cstdio>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/device_memory.h"
#include "flat/common.hpp"
#include "flat/hopper/device/device_universal.hpp"
#include "flat/hopper/kernel/flat_kernel_builder.hpp"

namespace flat {

using namespace cute;

template <bool NeedsScale, bool NeedsDecay, bool InitStateFromInput, typename ArchTag, typename TO,
          typename TQKV, typename TState>
void launch_linear_attention_prefill_kernel_sdi(
    cudaStream_t stream, TO* output, TState* output_state, TQKV const* q, TQKV const* k,
    TQKV const* v, TState const* input_state, int64_t const* cu_seqlens, int32_t num_seqs,
    int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_size, int64_t total_seqlen,
    float scale, float decay, float const* per_head_decay, int32_t decay_exponent_offset,
    int32_t sm_count) {
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
          add_option(Option<Tag::kNeedsScale,
                            std::conditional_t<NeedsScale, cute::true_type, cute::false_type>>{},
                     options_0);
      constexpr auto options_2 =
          add_option(Option<Tag::kNeedsDecay,
                            std::conditional_t<NeedsDecay, cute::true_type, cute::false_type>>{},
                     options_1);
      constexpr auto options_3 =
          add_option(Option<Tag::kIsLinearAttn, cute::true_type>{}, options_2);
      constexpr auto options_4 = add_option(
          Option<Tag::kInitStateFromInput,
                 std::conditional_t<InitStateFromInput, cute::true_type, cute::false_type>>{},
          options_3);
      return options_4;
    }());

    using TileShape = Shape<_64, _64, _128>;
    using Scheduler = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using Operation = cutlass::device::Universal<typename flat::kernel::FlatBuilder<
        T, float, float, TileShape,
        /*LayoutQ=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutK=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutV=*/cute::tuple<int64_t, _1, int32_t>,
        /*LayoutO=*/cute::tuple<int64_t, _1, int32_t>, Scheduler, Options>::Kernel>;
    using Arguments = typename Operation::Arguments;

    // NOTE: LayoutQ/K/V in (seq, head_size, (b,h)) coordinate semantics

    int32_t q_tok_stride = num_qo_heads * head_size;
    int32_t o_tok_stride = num_qo_heads * head_size;
    int32_t k_tok_stride = num_kv_heads * head_size;
    int32_t v_tok_stride = num_kv_heads * head_size;

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
                                .num_q_heads = num_qo_heads,
                                .num_k_heads = num_kv_heads,
                                .num_v_heads = num_kv_heads,
                                .num_o_heads = num_qo_heads,
                                .head_size = head_size,
                            },
                        .mainloop =
                            {
                                // clang-format off
                .ptr_Q = (T*)q,      .dQ = {q_tok_stride, _1{}, q_head_stride},
                .ptr_K = (T*)k,      .dK = {k_tok_stride, _1{}, k_head_stride},
                .ptr_V = (T*)v,      .dV = {v_tok_stride, _1{}, v_head_stride},
                .ptr_O = (T*)output, .dO = {o_tok_stride, _1{}, o_head_stride},
                .ptr_output_state = output_state,
                .ptr_input_state  = input_state,
                .scale = scale,
                .decay = decay,
                .per_head_decay = per_head_decay,
                .decay_exponent_offset = decay_exponent_offset,
        },  // clang-format on
                        .hw_info = hw_info};

    size_t workspace_size = op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error("can_implement failed");
    }

    status = op.initialize(arguments, workspace.get(), stream);
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
