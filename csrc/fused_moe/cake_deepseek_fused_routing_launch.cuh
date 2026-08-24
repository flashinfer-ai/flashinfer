#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::cake_deepseek_fused_routing {

template <typename ScoreT, typename BiasT>
cudaError_t launch(ScoreT* scores, BiasT* bias, ScoreT* topk_values, int32_t* topk_indices,
                   int16_t* routing_replay_out, int64_t num_tokens, int64_t num_experts,
                   int64_t n_group, int64_t topk_group, int64_t topk, double routed_scaling_factor,
                   bool launch_with_pdl, cudaStream_t stream);

#define FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(ScoreT, BiasT, ScoreTag, BiasTag)           \
  template <>                                                                                      \
  inline cudaError_t launch<ScoreT, BiasT>(                                                        \
      ScoreT * scores, BiasT * bias, ScoreT * topk_values, int32_t* topk_indices,                  \
      int16_t* routing_replay_out, int64_t num_tokens, int64_t num_experts, int64_t n_group,       \
      int64_t topk_group, int64_t topk, double routed_scaling_factor, bool launch_with_pdl,        \
      cudaStream_t stream) {                                                                       \
    cudaLaunchConfig_t config{};                                                                   \
    config.gridDim = dim3(static_cast<uint32_t>(num_tokens), 1, 1);                                \
    config.blockDim =                                                                              \
        dim3(static_cast<uint32_t>(n_group > 1 ? 256 : (num_experts <= 128 ? 128 : 384)), 1, 1);   \
    config.dynamicSmemBytes =                                                                      \
        static_cast<size_t>(n_group > 1 ? 2176 : (num_experts <= 128 ? 128 : 256));                \
    config.stream = stream;                                                                        \
    cudaLaunchAttribute attr{};                                                                    \
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;                                  \
    attr.val.programmaticStreamSerializationAllowed = launch_with_pdl;                             \
    config.attrs = &attr;                                                                          \
    config.numAttrs = 1;                                                                           \
    auto replay_bytes = reinterpret_cast<uint8_t*>(routing_replay_out);                            \
    int32_t const tokens32 = static_cast<int32_t>(num_tokens);                                     \
    int32_t const experts32 = static_cast<int32_t>(num_experts);                                   \
    int32_t const topk32 = static_cast<int32_t>(topk);                                             \
    int32_t const groups32 = static_cast<int32_t>(n_group);                                        \
    int32_t const top_groups32 = static_cast<int32_t>(topk_group);                                 \
    float const scaling32 = static_cast<float>(routed_scaling_factor);                             \
    int32_t const has_replay = routing_replay_out != nullptr;                                      \
    if (n_group > 1) {                                                                             \
      if (num_experts == 256 && n_group == 8 && topk_group == 4 && topk == 8) {                    \
        return cudaLaunchKernelEx(                                                                 \
            &config, kernel_cake_deepseek_routing_grouped_k8g4_##ScoreTag##_##BiasTag, scores,     \
            bias, topk_values, topk_indices, replay_bytes, tokens32, experts32, topk32, groups32,  \
            top_groups32, scaling32, has_replay);                                                  \
      }                                                                                            \
      return cudaLaunchKernelEx(&config,                                                           \
                                kernel_cake_deepseek_routing_grouped_##ScoreTag##_##BiasTag,       \
                                scores, bias, topk_values, topk_indices, replay_bytes, tokens32,   \
                                experts32, topk32, groups32, top_groups32, scaling32, has_replay); \
    }                                                                                              \
    if (num_experts <= 128) {                                                                      \
      return cudaLaunchKernelEx(&config,                                                           \
                                kernel_cake_deepseek_routing_single128_##ScoreTag##_##BiasTag,     \
                                scores, bias, topk_values, topk_indices, replay_bytes, tokens32,   \
                                experts32, topk32, groups32, top_groups32, scaling32, has_replay); \
    }                                                                                              \
    return cudaLaunchKernelEx(&config,                                                             \
                              kernel_cake_deepseek_routing_single384_##ScoreTag##_##BiasTag,       \
                              scores, bias, topk_values, topk_indices, replay_bytes, tokens32,     \
                              experts32, topk32, groups32, top_groups32, scaling32, has_replay);   \
  }

FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(float, float, f32, f32)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(float, half, f32, f16)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(float, __nv_bfloat16, f32, bf16)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(half, float, f16, f32)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(half, half, f16, f16)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(half, __nv_bfloat16, f16, bf16)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(__nv_bfloat16, float, bf16, f32)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(__nv_bfloat16, half, bf16, f16)
FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH(__nv_bfloat16, __nv_bfloat16, bf16, bf16)

#undef FLASHINFER_DEFINE_CAKE_DEEPSEEK_ROUTING_LAUNCH

}  // namespace flashinfer::cake_deepseek_fused_routing
