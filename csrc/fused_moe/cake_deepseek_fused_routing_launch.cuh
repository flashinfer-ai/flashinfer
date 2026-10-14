#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

namespace flashinfer::cake_deepseek_fused_routing {

// Dispatches one NoAuxTc call to the matching kernel-template instantiation of
// cake_deepseek_fused_routing/cake_deepseek_fused_routing_kernels.cu:
//   n_group > 1              -> grouped<ScoreT, BiasT, 4, 8> for E=256 G=8 topk_group=4 topk=8,
//                               grouped<ScoreT, BiasT, 1, 1> otherwise (256 threads);
//   n_group == 1, E <= 128   -> single<ScoreT, BiasT, 128>;
//   n_group == 1, E <= 384   -> single<ScoreT, BiasT, 384>.
// On compute capability 10.7 the E=256 G=8 topk_group=4 topk=8 route above kMultiTokenMaxCtas
// tokens uses the multi-token CTA members of
// cake_deepseek_fused_routing/cake_deepseek_fused_routing_kernels_sm107a_mt.cu instead: one
// 256-thread CTA routes tokens_per_cta tokens, the smallest of {2, 4, 8} that keeps the grid at or
// below kMultiTokenMaxCtas CTAs (else 8).  Every other launch keeps the one-token kernels.
constexpr int64_t kMultiTokenMaxCtas = 848;
constexpr int kMultiTokenMajor = 10;
constexpr int kMultiTokenMinor = 7;
constexpr int kMultiTokenChoices[] = {2, 4, 8};

inline bool is_specialized_k8g4_route(int64_t num_experts, int64_t n_group, int64_t topk_group,
                                      int64_t topk) {
  return num_experts == 256 && n_group == 8 && topk_group == 4 && topk == 8;
}

// Compute capability check for the multi-token members, cached per device (one attribute query
// per device per process); reached only above kMultiTokenMaxCtas tokens on the specialized route.
inline bool device_has_multi_token_members() {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices] = {};  // 0 = unknown, 1 = no, 2 = yes
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) return false;
  int state = cached[device].load(std::memory_order_relaxed);
  if (state == 0) {
    int major = 0, minor = 0;
    bool const ok =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess;
    state = (ok && major == kMultiTokenMajor && minor == kMultiTokenMinor) ? 2 : 1;
    cached[device].store(state, std::memory_order_relaxed);
  }
  return state == 2;
}

// Tokens one CTA routes for this launch: 1 (the one-token kernels) unless the device has the
// multi-token members, the route is the specialized one and num_tokens > kMultiTokenMaxCtas.
inline int tokens_per_cta_for(int64_t num_tokens, int64_t num_experts, int64_t n_group,
                              int64_t topk_group, int64_t topk) {
  if (n_group <= 1 || num_tokens <= kMultiTokenMaxCtas) return 1;
  if (!is_specialized_k8g4_route(num_experts, n_group, topk_group, topk)) return 1;
  if (!device_has_multi_token_members()) return 1;
  for (int choice : kMultiTokenChoices) {
    if ((num_tokens + choice - 1) / choice <= kMultiTokenMaxCtas) return choice;
  }
  return kMultiTokenChoices[2];
}

template <typename ScoreT, typename BiasT>
cudaError_t launch(ScoreT* scores, BiasT* bias, ScoreT* topk_values, int32_t* topk_indices,
                   int16_t* routing_replay_out, int64_t num_tokens, int64_t num_experts,
                   int64_t n_group, int64_t topk_group, int64_t topk, double routed_scaling_factor,
                   bool launch_with_pdl, cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(num_tokens), 1, 1);
  config.blockDim =
      dim3(static_cast<uint32_t>(n_group > 1 ? 256 : (num_experts <= 128 ? 128 : 384)), 1, 1);
  config.dynamicSmemBytes =
      static_cast<size_t>(n_group > 1 ? 2176 : (num_experts <= 128 ? 128 : 256));
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = launch_with_pdl;
  config.attrs = &attr;
  config.numAttrs = 1;
  auto replay_bytes = reinterpret_cast<uint8_t*>(routing_replay_out);
  int32_t const tokens32 = static_cast<int32_t>(num_tokens);
  int32_t const experts32 = static_cast<int32_t>(num_experts);
  int32_t const topk32 = static_cast<int32_t>(topk);
  int32_t const groups32 = static_cast<int32_t>(n_group);
  int32_t const top_groups32 = static_cast<int32_t>(topk_group);
  float const scaling32 = static_cast<float>(routed_scaling_factor);
  int32_t const has_replay = routing_replay_out != nullptr;
  if (n_group > 1) {
    if (is_specialized_k8g4_route(num_experts, n_group, topk_group, topk)) {
      int const tokens_per_cta =
          tokens_per_cta_for(num_tokens, num_experts, n_group, topk_group, topk);
      if (tokens_per_cta > 1) {
        config.gridDim =
            dim3(static_cast<uint32_t>((num_tokens + tokens_per_cta - 1) / tokens_per_cta), 1, 1);
        config.dynamicSmemBytes = static_cast<size_t>(
            cake_deepseek_routing_sm107a_mt::grouped_k8g4_mt_smem_bytes(tokens_per_cta));
        switch (tokens_per_cta) {
          case 2:
            return cudaLaunchKernelEx(
                &config,
                cake_deepseek_routing_sm107a_mt::kernel_cake_deepseek_routing_grouped_k8g4_mt2<
                    ScoreT, BiasT>,
                scores, bias, topk_values, topk_indices, replay_bytes, tokens32, experts32, topk32,
                groups32, top_groups32, scaling32, has_replay);
          case 4:
            return cudaLaunchKernelEx(
                &config,
                cake_deepseek_routing_sm107a_mt::kernel_cake_deepseek_routing_grouped_k8g4_mt4<
                    ScoreT, BiasT>,
                scores, bias, topk_values, topk_indices, replay_bytes, tokens32, experts32, topk32,
                groups32, top_groups32, scaling32, has_replay);
          default:
            return cudaLaunchKernelEx(
                &config,
                cake_deepseek_routing_sm107a_mt::kernel_cake_deepseek_routing_grouped_k8g4_mt8<
                    ScoreT, BiasT>,
                scores, bias, topk_values, topk_indices, replay_bytes, tokens32, experts32, topk32,
                groups32, top_groups32, scaling32, has_replay);
        }
      }
      return cudaLaunchKernelEx(&config, kernel_cake_deepseek_routing_grouped<ScoreT, BiasT, 4, 8>,
                                scores, bias, topk_values, topk_indices, replay_bytes, tokens32,
                                experts32, topk32, groups32, top_groups32, scaling32, has_replay);
    }
    return cudaLaunchKernelEx(&config, kernel_cake_deepseek_routing_grouped<ScoreT, BiasT, 1, 1>,
                              scores, bias, topk_values, topk_indices, replay_bytes, tokens32,
                              experts32, topk32, groups32, top_groups32, scaling32, has_replay);
  }
  if (num_experts <= 128) {
    return cudaLaunchKernelEx(&config, kernel_cake_deepseek_routing_single<ScoreT, BiasT, 128>,
                              scores, bias, topk_values, topk_indices, replay_bytes, tokens32,
                              experts32, topk32, groups32, top_groups32, scaling32, has_replay);
  }
  return cudaLaunchKernelEx(&config, kernel_cake_deepseek_routing_single<ScoreT, BiasT, 384>,
                            scores, bias, topk_values, topk_indices, replay_bytes, tokens32,
                            experts32, topk32, groups32, top_groups32, scaling32, has_replay);
}

}  // namespace flashinfer::cake_deepseek_fused_routing
