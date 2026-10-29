/*
 * Plan selection for the SM90 NVFP4 MegaMoE split (BN128) family.
 *
 * The split family runs two kernels per call (L1 then L2), each with its own
 * pipeline depth and shared-memory budget. Those two numbers are template
 * parameters, so they must be known before the module is built -- this header
 * is compiled into a small plan module that Python queries first.
 *
 * The sizing arithmetic is carried over unchanged from the reference
 * implementation: it mirrors the kernel's shared-memory layout exactly, and any
 * paraphrase of it would silently mis-size the pipeline.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <flashinfer/megamoe/layout.cuh>
#include <utility>

#include "tvm_ffi_utils.h"

namespace flashinfer::megamoe {

template <typename T>
constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

struct SM90ArchSpec {
  static constexpr int smem_capacity = 232448;
};

static int get_num_experts_per_wave_for_mega_moe(const int& num_experts_per_rank,
                                                 const int& num_tokens, const int& num_topk,
                                                 const int& intermediate_hidden, const int& block_m,
                                                 const int& block_n, const int& num_sms) {
  float expected_tokens_per_expert =
      static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
  if (expected_tokens_per_expert < 1) {
    // Most experts don't have tokens, calculate all experts at once
    return num_experts_per_rank;
  }

  // Reduce per-expert block count by this factor since uneven routing leaves some experts with
  // fewer tokens
  constexpr int kImbalanceFactor = 2;

  // Count L1 blocks per expert assuming tokens are evenly spread across experts
  const int num_m_blocks = flashinfer::megamoe::ceil_div(
      static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);
  const int num_n_blocks = (2 * intermediate_hidden) / block_n;
  const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;

  // Pick the smallest value whose total blocks (after imbalance reduction) can keep all SMs busy
  int num_experts_per_wave =
      num_l1_blocks_per_expert > 0
          ? flashinfer::megamoe::ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert)
          : 1;
  num_experts_per_wave = std::min(num_experts_per_wave, num_experts_per_rank);

  // Round up to the nearest divisor of num_experts_per_rank so every wave processes the same count
  while (num_experts_per_wave < num_experts_per_rank and
         num_experts_per_rank % num_experts_per_wave != 0)
    ++num_experts_per_wave;

  return num_experts_per_wave;
}

struct SM90NVFP4MegaMoEConfig {
  static constexpr int kBlockM = 128;
  static constexpr int kBlockN = 128;
  static constexpr int kBlockK = 128;
  static constexpr int kWeightStoragePerKBlock = 80;
  static constexpr int kSwizzleActsMode = 128;
  static constexpr int kL1ClusterSize = 2;
  static constexpr int kL2ClusterSize = 1;
  static constexpr int kL1NumDispatchThreads = 128;
  static constexpr int kL2NumDispatchThreads = 0;
  static constexpr int kL1NumActiveDispatchWarps = 2;
  static constexpr int kL2NumActiveDispatchWarps = 0;
  static constexpr int kL1ActsSFReservationGranK = 64;
  static constexpr int kL2ActsSFReservationGranK = 128;
  static constexpr int kNumNonEpilogueThreads = 128;
  static constexpr int kNumEpilogueThreads = 256;

  int cluster_size;

  int num_max_pool_tokens;
  int num_padded_sf_pool_tokens;

  int num_experts_per_wave;

  int num_stages, smem_size;

  int num_dispatch_threads;

  friend std::ostream& operator<<(std::ostream& os, const SM90NVFP4MegaMoEConfig& config) {
    os << "SM90NVFP4MegaMoEConfig("
       << "block_m=" << kBlockM << ", block_n=" << kBlockN << ", block_k=" << kBlockK
       << ", cluster_size=" << config.cluster_size
       << ", num_max_pool_tokens=" << config.num_max_pool_tokens
       << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
       << ", swizzle_acts_mode=" << kSwizzleActsMode
       << ", num_experts_per_wave=" << config.num_experts_per_wave
       << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
       << ", num_dispatch_threads=" << config.num_dispatch_threads
       << ", num_non_epilogue_threads=" << kNumNonEpilogueThreads
       << ", num_epilogue_threads=" << kNumEpilogueThreads << ")";
    return os;
  }
};

struct SM90NVFP4MegaMoEInput {
  int launch_num_sms;

  int num_ranks;
  int num_experts;
  int num_experts_per_rank;
  int num_max_tokens_per_rank;
  int num_tokens;
  int num_topk;
  int hidden;
  int intermediate_hidden;
  int num_padded_sf_pool_tokens;
};

struct SM90NVFP4MegaMoELoad {
  int64_t routed_tokens;
  int64_t local_experts;
  float expected_tokens_per_local_expert;

  bool valid() const { return routed_tokens > 0 && local_experts > 0; }

  bool greater_equal(const int64_t value) const { return routed_tokens >= value * local_experts; }
};

struct SM90NVFP4MegaMoEPlan {
  SM90NVFP4MegaMoEConfig l1_config;
  SM90NVFP4MegaMoEConfig l2_config;
  bool dispatch_dequant;
  bool l2_arrival_counter;
};

enum class SM90NVFP4MegaMoEPhase {
  L1,
  L2,
};

static SM90NVFP4MegaMoELoad get_sm90_nvfp4_mega_moe_load(const SM90NVFP4MegaMoEInput& input) {
  const int64_t routed_tokens = static_cast<int64_t>(input.num_tokens) * input.num_topk;
  const SM90NVFP4MegaMoELoad load{
      routed_tokens,
      input.num_experts_per_rank,
      static_cast<float>(routed_tokens) / input.num_experts_per_rank,
  };
  TVM_FFI_ICHECK(load.valid());
  return load;
}

static std::pair<int, int> get_sm90_nvfp4_mega_moe_pipeline_config(
    const SM90NVFP4MegaMoEInput& input, const SM90NVFP4MegaMoELoad& load,
    const SM90NVFP4MegaMoEPhase phase) {
  const auto align = [](int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
  };
  constexpr int kSmemAlignment = 1024;
  const bool is_l2 = phase == SM90NVFP4MegaMoEPhase::L2;
  const int num_dispatch_threads = is_l2 ? SM90NVFP4MegaMoEConfig::kL2NumDispatchThreads
                                         : SM90NVFP4MegaMoEConfig::kL1NumDispatchThreads;
  const int num_active_dispatch_warps = is_l2 ? SM90NVFP4MegaMoEConfig::kL2NumActiveDispatchWarps
                                              : SM90NVFP4MegaMoEConfig::kL1NumActiveDispatchWarps;
  // L1 retains only the proven per-64 physical reservation. Kernel
  // activation-scale semantics remain per-128.
  const int acts_sf_gran_k = is_l2 ? SM90NVFP4MegaMoEConfig::kL2ActsSFReservationGranK
                                   : SM90NVFP4MegaMoEConfig::kL1ActsSFReservationGranK;

  TVM_FFI_ICHECK(num_active_dispatch_warps <= num_dispatch_threads / 32);
  const int num_dispatch_warps = num_active_dispatch_warps;
  constexpr int kNumEpilogueWarps = 8;
  constexpr int kNumEpilogueWarpgroups = 2;
  constexpr int kWGBlockM = 64;
  constexpr int kWGL1OutBlockN = 64;

  const int smem_expert_count_size =
      align(input.num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
  const int smem_send_buffers_size =
      align(static_cast<int>(
                layout::Buffer(layout::Data(input.hidden), num_dispatch_warps, 1).get_num_bytes()),
            kSmemAlignment);
  const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;
  const int smem_nvfp4_lut = align(128 * 8, kSmemAlignment);

  const int smem_cd_l1 = kNumEpilogueWarpgroups * kWGBlockM * kWGL1OutBlockN;
  // L2 stages its remote scatter through one warp-private 8-row half tile so
  // that each destination row leaves as a single contiguous 16-byte-wide
  // burst. Must match `kL2StageRows` / `kL2StageRowStride` in the kernel.
  constexpr int kWGBlockN = 128;
  constexpr int kL2StageRows = 8;
  constexpr int kL2StageRowPad = 8;
  const int smem_cd_l2 = kNumEpilogueWarps * kL2StageRows * (kWGBlockN + kL2StageRowPad) *
                         static_cast<int>(sizeof(uint16_t));
  const int smem_cd = align(is_l2 ? smem_cd_l2 : smem_cd_l1, kSmemAlignment);

  const int num_sfa_groups_per_bk = SM90NVFP4MegaMoEConfig::kBlockK / acts_sf_gran_k;
  const int smem_sfa_per_stage = align(
      num_sfa_groups_per_bk * SM90NVFP4MegaMoEConfig::kBlockM * static_cast<int>(sizeof(float)),
      128);
  const int smem_per_stage = SM90NVFP4MegaMoEConfig::kBlockM * SM90NVFP4MegaMoEConfig::kBlockK +
                             SM90NVFP4MegaMoEConfig::kBlockN * SM90NVFP4MegaMoEConfig::kBlockK +
                             smem_sfa_per_stage;
  const int smem_barriers_fixed = (num_dispatch_warps + 2 * kNumEpilogueWarps) * 8;
  // Both phases own full, empty, and dequant barriers. Production L1 is
  // always half-stream mode4 and therefore owns one additional K[64:128]
  // publication mbarrier per stage. Account for it while selecting stages,
  // rather than appending bytes after a max-stage decision.
  const int smem_barriers_per_stage = (is_l2 ? 3 : 4) * 8;
  const int smem_fixed = smem_dispatch_size + smem_nvfp4_lut + smem_cd + smem_barriers_fixed;
  const int max_num_stages =
      (SM90ArchSpec::smem_capacity - smem_fixed) / (smem_per_stage + smem_barriers_per_stage);
  const int num_stages =
      !is_l2 && load.expected_tokens_per_local_expert > 8.0f && max_num_stages > 6 ? 6
                                                                                   : max_num_stages;
  TVM_FFI_ICHECK(max_num_stages >= 2);
  TVM_FFI_ICHECK(num_stages >= 2 && num_stages <= max_num_stages);
  return {
      num_stages,
      smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage),
  };
}

static int get_num_experts_per_wave_for_sm90_nvfp4_mega_moe(const SM90NVFP4MegaMoEInput& input) {
  return get_num_experts_per_wave_for_mega_moe(
      input.num_experts_per_rank, input.num_tokens, input.num_topk, input.intermediate_hidden,
      SM90NVFP4MegaMoEConfig::kBlockM, SM90NVFP4MegaMoEConfig::kBlockN, input.launch_num_sms);
}

static SM90NVFP4MegaMoEConfig materialize_sm90_nvfp4_mega_moe_phase(
    const SM90NVFP4MegaMoEInput& input, const SM90NVFP4MegaMoELoad& load,
    const SM90NVFP4MegaMoEPhase phase) {
  const bool is_l2 = phase == SM90NVFP4MegaMoEPhase::L2;
  SM90NVFP4MegaMoEConfig config{
      is_l2 ? SM90NVFP4MegaMoEConfig::kL2ClusterSize : SM90NVFP4MegaMoEConfig::kL1ClusterSize,
      layout::get_num_max_pool_tokens(input.num_ranks, input.num_max_tokens_per_rank,
                                      input.num_topk, input.num_experts_per_rank),
      input.num_padded_sf_pool_tokens,
      get_num_experts_per_wave_for_sm90_nvfp4_mega_moe(input),
      0,
      0,
      is_l2 ? SM90NVFP4MegaMoEConfig::kL2NumDispatchThreads
            : SM90NVFP4MegaMoEConfig::kL1NumDispatchThreads,
  };
  std::tie(config.num_stages, config.smem_size) =
      get_sm90_nvfp4_mega_moe_pipeline_config(input, load, phase);
  return config;
}

static bool is_sm90_nvfp4_mega_moe_plan_legal(const SM90NVFP4MegaMoEInput& input,
                                              const SM90NVFP4MegaMoEPlan& plan) {
  const auto valid_phase = [&](const SM90NVFP4MegaMoEConfig& config) {
    return config.num_experts_per_wave > 0 &&
           config.num_experts_per_wave <= input.num_experts_per_rank &&
           input.num_experts_per_rank % config.num_experts_per_wave == 0 &&
           config.num_stages >= 2 && config.smem_size > 0 &&
           config.smem_size <= SM90ArchSpec::smem_capacity;
  };
  return valid_phase(plan.l1_config) && valid_phase(plan.l2_config) &&
         plan.l1_config.cluster_size == SM90NVFP4MegaMoEConfig::kL1ClusterSize &&
         plan.l1_config.num_dispatch_threads == SM90NVFP4MegaMoEConfig::kL1NumDispatchThreads &&
         plan.l2_config.cluster_size == SM90NVFP4MegaMoEConfig::kL2ClusterSize &&
         plan.l2_config.num_dispatch_threads == SM90NVFP4MegaMoEConfig::kL2NumDispatchThreads;
}

static SM90NVFP4MegaMoEPlan select_sm90_nvfp4_split_mega_moe(const SM90NVFP4MegaMoEInput& input) {
  TVM_FFI_ICHECK(input.launch_num_sms > 0);
  TVM_FFI_ICHECK(input.num_ranks > 0);
  TVM_FFI_ICHECK(input.num_experts_per_rank > 0);
  TVM_FFI_ICHECK(input.num_experts == input.num_experts_per_rank * input.num_ranks);
  TVM_FFI_ICHECK(input.num_max_tokens_per_rank > 0);
  TVM_FFI_ICHECK(input.num_tokens > 0 && input.num_tokens <= input.num_max_tokens_per_rank);
  TVM_FFI_ICHECK(input.num_topk > 0);
  TVM_FFI_ICHECK(input.hidden > 0 && input.hidden % 128 == 0);
  TVM_FFI_ICHECK(input.intermediate_hidden > 0 && input.intermediate_hidden % 128 == 0);
  TVM_FFI_ICHECK(input.num_padded_sf_pool_tokens > 0);

  const auto load = get_sm90_nvfp4_mega_moe_load(input);
  const SM90NVFP4MegaMoEPlan plan{
      materialize_sm90_nvfp4_mega_moe_phase(input, load, SM90NVFP4MegaMoEPhase::L1),
      materialize_sm90_nvfp4_mega_moe_phase(input, load, SM90NVFP4MegaMoEPhase::L2),
      // Every production BN128 split uses the accepted L1 mode4+remap
      // implementation. That implementation requires the independent odd-K
      // dispatch dequant team, including in the former rho<256 interval.
      true,
      load.expected_tokens_per_local_expert <= 32.0f ||
          load.expected_tokens_per_local_expert >= 128.0f,
  };
  TVM_FFI_ICHECK(is_sm90_nvfp4_mega_moe_plan_legal(input, plan));
  return plan;
}

}  // namespace flashinfer::megamoe
