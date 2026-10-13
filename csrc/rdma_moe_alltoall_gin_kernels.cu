/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#include <cooperative_groups.h>

#include "flashinfer/comm/rdma_moe_alltoall/kernels/launch.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/precision.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/utils.cuh"
namespace cg = cooperative_groups;
#include "flashinfer/comm/rdma_moe_alltoall/kernels/gin_device.cuh"
#define EP_GIN_KERNEL_ARG , deep_ep::GinLaunch gin_launch
#define EP_GIN_HOST_ARG , const deep_ep::GinLaunch& gin_launch
#define EP_GIN_PASS , gin_launch

namespace deep_ep {
namespace extensions {
#define SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(case_macro) \
  switch (hidden) {                                     \
    case 4096:                                          \
      case_macro(4096);                                 \
    case 6144:                                          \
      case_macro(6144);                                 \
    case 7168:                                          \
      case_macro(7168);                                 \
    default:                                            \
      EP_HOST_ASSERT(false && "Unsupported hidden");    \
  }                                                     \
  while (false)

#include "flashinfer/comm/rdma_moe_alltoall/gin_local_partial.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/gin_planned.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/gin_channels.cuh"
template <typename Precision, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch_low_precision(
    void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* cumulative_local_expert_recv_stats, void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
    const void* x, const void* x_scales, const int* topk_idx, int* atomic_counter_per_expert,
    int* atomic_finish_counter_per_expert, int* next_clean, int num_next_clean_int, int num_tokens,
    int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts, int rank, int num_ranks,
    int num_warp_groups, int num_warps_per_group, int phases EP_GIN_KERNEL_ARG) {
  EP_STATIC_ASSERT(kHidden % 32 == 0, "Invalid hidden size");
  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto num_warps = num_warp_groups * num_warps_per_group;
  const auto num_local_experts = num_experts / num_ranks;
  const auto warp_group_id = warp_id / num_warps_per_group;
  const auto sub_warp_id = warp_id % num_warps_per_group;
  const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

  constexpr size_t kHiddenBytes = kHidden / Precision::kNumElemsPerUnit * Precision::kUnitBytes;
  constexpr size_t kHiddenVecAccessNum = kHiddenBytes / sizeof(int4);
  constexpr size_t kScalesBytes =
      kHidden / Precision::kNumElemsPerScale * Precision::kBytesPerScale;
  constexpr size_t kScalesVecAccessNum = kScalesBytes / sizeof(int4);

  const size_t header_bytes = gin_launch.bf16_local_partial_planned ? 80 : sizeof(int4);
  const size_t kNumBytesPerMsg = header_bytes + kHiddenBytes + kScalesBytes;
  EP_DEVICE_ASSERT(kNumBytesPerMsg % sizeof(int4) == 0);

  // Preserve BF16's predicate and the FP8/coalesced paths. NVFP4 reuses
  // the same immutable 16-byte follower reference only above its auto-policy
  // coalesced capacity boundary. A leader owns both payload and block scales.
  // Its registered send tail fits even for E=2, K=8 and minimum H4096:
  // C*(2320+8*16) <= E*C*2320. Larger H increases that headroom.
  // Untyped buffers retain the full-message path. Combine is unchanged.
  const bool bf16_rank_dedup =
      Precision::kBytesPerScale == 0 &&
      (num_max_dispatch_tokens_per_rank > 8 || gin_launch.bf16_local_partial_planned);
  const bool nvfp4_rank_dedup = Precision::kUseGlobalScale && gin_launch.nvfp4_legacy_rank_dedup &&
                                num_max_dispatch_tokens_per_rank > 32;
  const bool rank_dedup = num_experts >= 2 && (bf16_rank_dedup || nvfp4_rank_dedup);
  __shared__ int shared_route_slots[32];

  // Expert counts
  constexpr int kNumMaxWarpGroups = 32;
  __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_DISPATCH_RECV;

  // There are 2 kinds of warps in this part:
  // 1. The first-kind warps for sending top-k tokens
  // 2. The last warp for reading `topk_idx` and count for per-expert information
  if (warp_id < num_warps - 1) {
    const auto num_threads = (num_warps - 1) * 32;

    for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
      const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * kHiddenVecAccessNum;
      const auto rdma_x_src_idx =
          reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * kNumBytesPerMsg);
      const auto rdma_x_vec =
          reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + header_bytes);
      const auto rdma_x_scales =
          reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + kHiddenBytes);

      // Overlap top-k index read and source token index writes
      auto dst_expert_idx = warp_id < num_topk
                                ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id))
                                : -1;
      int route_slot = 0;
      if (rank_dedup && dst_expert_idx >= 0) {
        EP_DEVICE_ASSERT(dst_expert_idx < num_experts);
        if (lane_id == 0) {
          route_slot = atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1);
          shared_route_slots[warp_id] = route_slot;
        }
        route_slot = __shfl_sync(0xffffffff, route_slot, 0);
        EP_DEVICE_ASSERT(route_slot < num_max_dispatch_tokens_per_rank);
      }
      if (thread_id == 0) {
        if (gin_launch.bf16_local_partial_planned)
          planned_pack_header(rdma_x_src_idx, token_idx, num_tokens, num_topk, topk_idx,
                              gin_launch);
        else if (rank_dedup)
          *reinterpret_cast<int4*>(rdma_x_src_idx) = make_int4(token_idx, 0, 0, 0);
        else
          *rdma_x_src_idx = token_idx;
      }

#pragma unroll
      for (int i = thread_id; i < kHiddenVecAccessNum; i += num_threads) {
        auto int4_value = __ldg(x_int4 + i);
        rdma_x_vec[i] = *reinterpret_cast<int4*>(&int4_value);
      }
      if constexpr (kScalesBytes > 0) {
        const auto x_scales_int4 =
            reinterpret_cast<const int4*>(x_scales) + token_idx * kScalesVecAccessNum;
#pragma unroll
        for (int i = thread_id; i < kScalesVecAccessNum; i += num_threads) {
          auto int4_value = __ldg(x_scales_int4 + i);
          rdma_x_scales[i] = *reinterpret_cast<int4*>(&int4_value);
        }
      }
      asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

      // Issue IBGDA sends
      if (dst_expert_idx >= 0) {
        int slot_idx = route_slot;
        if (!rank_dedup) {
          slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
          slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
        }
        const auto dst_rank = dst_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
        auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
        size_t bytes = kNumBytesPerMsg;
        if (rank_dedup && dst_rank != rank) {
          int leader_warp = warp_id;
          int leader_expert = dst_expert_local_idx;
          for (int previous = 0; previous < warp_id; ++previous) {
            const int expert = __ldg(topk_idx + token_idx * num_topk + previous);
            if (expert >= 0 && expert / num_local_experts == dst_rank) {
              leader_warp = previous;
              leader_expert = expert % num_local_experts;
              break;
            }
          }
          if (leader_warp != warp_id) {
            auto* reference = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(rdma_x) +
                                                      size_t(num_max_dispatch_tokens_per_rank) *
                                                          kNumBytesPerMsg) +
                              size_t(token_idx) * num_topk + warp_id;
            if (lane_id == 0)
              *reference = make_int4(token_idx, leader_expert, shared_route_slots[leader_warp], 1);
            __syncwarp();
            src_ptr = reinterpret_cast<uint64_t>(reference);
            bytes = sizeof(int4);
          }
        }
        const auto dst_ptr =
            reinterpret_cast<uint64_t>(rdma_recv_x) +
            dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg +
            rank * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg + slot_idx * kNumBytesPerMsg;
        gin_put_warp(gin_launch, reinterpret_cast<void*>(dst_ptr),
                     reinterpret_cast<const void*>(src_ptr), bytes, dst_rank, dst_expert_local_idx);
        // Increase counter after finishing
        __syncwarp();
        lane_id == 0
            ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1)
            : 0;
      }
      // A faster warp must not overwrite the next token's shared slot before
      // every follower has copied its current leader slot into immutable RDMA memory.
      if (rank_dedup) asm volatile("bar.sync 1, %0;" ::"r"(num_threads));
    }
  } else if (warp_id == num_warps - 1) {
    EP_DEVICE_ASSERT(num_sms > 1);
    if (sm_id == 0) {
      // The first SM is also responsible for checking QPs

      // Notify before executing `int_p`
      __syncwarp();
#pragma unroll
      for (int i = lane_id; i < num_experts; i += 32)
        atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
    }

    // This SM should be responsible for some destination experts, read `topk_idx` for them
    int expert_count[kNumMaxWarpGroups] = {0};
    const auto expert_begin_idx = sm_id * num_warp_groups;
    const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

// Per lane count
#pragma unroll 8
    for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
      auto idx = static_cast<int>(__ldg(topk_idx + i));
      if (idx >= expert_begin_idx and idx < expert_end_idx) expert_count[idx - expert_begin_idx]++;
    }

// Warp reduce
#pragma unroll
    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
      auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
      if (lane_id == 0) {
        shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
        atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
      }
    }
  }
  __syncthreads();

  // Issue count sends
  if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
    const auto num_tokens_sent =
        shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

    // Wait local sends issued and send expert counts
    while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) !=
           FINISHED_SUM_TAG * 2);
    auto dst_ptr =
        reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
    // The release/acquire scoreboard above joins every producer CTA before
    // this strong signal is posted on the same expert context and peer.
    if (dst_rank == rank) {
      st_release_sys_global(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1);
    } else {
      auto gin = gin_context(gin_launch, dst_expert_local_idx);
      gin.putValue(ncclTeamWorld(gin_launch.comm), dst_rank, gin_launch.window,
                   gin_offset(gin_launch, reinterpret_cast<void*>(dst_ptr)), -num_tokens_sent - 1,
                   ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(
                       gin_launch.signal_base + rank * num_local_experts + dst_expert_local_idx)});
      gin_finish_peer(gin_launch, dst_rank, dst_expert_local_idx);
    }
    // Clean workspace for next use
    atomic_counter_per_expert[responsible_expert_idx] = 0;
    atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

    // Clean `packed_recv_count`
    if (dst_rank == 0) packed_recv_count[dst_expert_local_idx] = 0;
  }
  __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
  if (phases & LOW_LATENCY_SEND_PHASE) cg::this_grid().sync();

  // References can target another expert's row. Join every expert's strong
  // signal before any CTA follows one, including separate receive-hook kernels.
  // Each signal/epoch retains its original single waiter and context.
  if (rank_dedup) {
    if (responsible_expert_idx < num_experts && sub_warp_id == 1 && lane_id == 0) {
      const int source = responsible_expert_idx / num_local_experts;
      if (source != rank)
        gin_wait_epoch(gin_launch, responsible_expert_idx % num_local_experts,
                       responsible_expert_idx);
    }
    cg::this_grid().sync();
  }

  // Receiving and packing
  if (responsible_expert_idx < num_experts) {
    const auto src_rank = responsible_expert_idx / num_local_experts;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto rdma_recv_x_uint8 =
        reinterpret_cast<uint8_t*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg +
        src_rank * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg;
    const auto recv_x_int4 =
        reinterpret_cast<int4*>(packed_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kHiddenVecAccessNum;
    const auto recv_src_info =
        packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;

    // Shared between sub-warps in warp groups
    __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups],
        shared_recv_token_begin_idx[kNumMaxWarpGroups];

    // Wait tokens to arrive
    // NOTES: using sub-warp 1 to overlap with sub-warp 0
    int num_recv_tokens, recv_token_begin_idx;
    EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
    if (sub_warp_id == 1 and lane_id == 0) {
      if (!rank_dedup && src_rank != rank)
        gin_wait_epoch(gin_launch, local_expert_idx, responsible_expert_idx);
      num_recv_tokens =
          ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank);
      num_recv_tokens = -num_recv_tokens - 1;
      recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
      shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
      recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
      if (cumulative_local_expert_recv_stats != nullptr)
        atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
    }
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
    num_recv_tokens = shared_num_recv_tokens[warp_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

    // Copy tokens
    for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
      // Copy source info
      const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * kNumBytesPerMsg);
      if (lane_id == 0) recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
      __syncwarp();

      // Copy data
      // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
      const auto* payload = reinterpret_cast<const uint8_t*>(src_src_idx);
      if (rank_dedup) {
        const int4 header = ld_nc_global(reinterpret_cast<const int4*>(src_src_idx));
        if (header.w != 0) {
          EP_DEVICE_ASSERT(header.w == 1 && header.y >= 0 && header.y < num_local_experts);
          EP_DEVICE_ASSERT(header.z >= 0 && header.z < num_max_dispatch_tokens_per_rank);
          payload = reinterpret_cast<const uint8_t*>(rdma_recv_x) +
                    ((size_t(header.y) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank +
                     header.z) *
                        kNumBytesPerMsg;
          EP_DEVICE_ASSERT(ld_nc_global(reinterpret_cast<const int*>(payload)) == header.x);
        }
      }
      if (gin_launch.bf16_local_partial_planned && lane_id == 0)
        planned_record_row(payload, src_rank, rank * num_local_experts + local_expert_idx,
                           size_t(local_expert_idx) * num_ranks * num_max_dispatch_tokens_per_rank +
                               recv_token_begin_idx + i,
                           num_max_dispatch_tokens_per_rank, num_experts, gin_launch);
      const auto src_data = reinterpret_cast<const int4*>(payload + header_bytes);
      const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * kHiddenVecAccessNum;
      constexpr int kUnrollFactorData = (kHiddenVecAccessNum + 31) / 32;
      EP_STATIC_ASSERT(kUnrollFactorData > 0, "Invalid unroll factor");
      UNROLLED_WARP_COPY(kUnrollFactorData, lane_id, kHiddenVecAccessNum, dst_data, src_data,
                         ld_nc_global, st_na_global);
      if constexpr (kScalesBytes > 0) {
        const auto src_scales = reinterpret_cast<const int4*>(
            reinterpret_cast<const uint8_t*>(src_data) + kHiddenBytes);
        const auto recv_scales_int4 =
            reinterpret_cast<int4*>(packed_recv_x_scales) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kScalesVecAccessNum;
        const auto dst_scales = recv_scales_int4 + (recv_token_begin_idx + i) * kScalesVecAccessNum;
        constexpr int kUnrollFactorScales = (kScalesVecAccessNum + 31) / 32;
        EP_STATIC_ASSERT(kUnrollFactorScales > 0, "Invalid scales unroll factor");
        UNROLLED_WARP_COPY(kUnrollFactorScales, lane_id, kScalesVecAccessNum, dst_scales,
                           src_scales, ld_nc_global, st_na_global);
      }
    }
  }
}

void dispatch(int precision, void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range, int* packed_recv_count,
              int* cumulative_local_expert_recv_stats, void* rdma_recv_x, int* rdma_recv_count,
              void* rdma_x, const void* x, const void* x_scales, const int* topk_idx,
              int* next_clean, int num_next_clean_int, int num_tokens, int hidden,
              int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts, int rank,
              int num_ranks, void* workspace, int num_device_sms, cudaStream_t stream,
              int phases EP_GIN_HOST_ARG) {
  EP_HOST_ASSERT(precision >= 0 && precision <= 2);
  EP_HOST_ASSERT(precision == 2 ||
                 ((num_tokens == 0 || x_scales != nullptr) && packed_recv_x_scales != nullptr));
  if (gin_launch.nvfp4_rank_packet) {
    EP_HOST_ASSERT(precision == 1 && num_max_dispatch_tokens_per_rank > 32 &&
                   (num_ranks == 4 || num_ranks == 8) && gin_launch.num_ctas > 0 &&
                   gin_launch.num_ctas <= num_device_sms && num_topk > 0 && num_topk <= 8 &&
                   gin_launch.send_counts != nullptr && gin_launch.recv_counts != nullptr &&
                   gin_launch.planned_counts != nullptr &&
                   gin_launch.planned_fanout_cursors != nullptr);
#define NVFP4_RANK_PACKET_LAUNCH_CASE(hidden)                                                      \
  {                                                                                                \
    auto dispatch_func = dispatch_channels_nvfp4_rank_packet<hidden>;                              \
    LAUNCH_KERNEL(&cfg, dispatch_func, packed_recv_x, packed_recv_x_scales, packed_recv_src_info,  \
                  packed_recv_layout_range, packed_recv_count, cumulative_local_expert_recv_stats, \
                  rdma_recv_x, rdma_x, x, x_scales, topk_idx, num_tokens,                          \
                  num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank, num_ranks,        \
                  phases, gin_launch);                                                             \
  }                                                                                                \
  break
    SETUP_LAUNCH_CONFIG(gin_launch.num_ctas, kGinChannelThreads, stream);
    SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(NVFP4_RANK_PACKET_LAUNCH_CASE);
#undef NVFP4_RANK_PACKET_LAUNCH_CASE
    return;
  }
  if (gin_launch.coalesced) {
    EP_HOST_ASSERT(gin_launch.num_ctas > 0 && gin_launch.num_ctas <= num_device_sms);
    EP_HOST_ASSERT(num_topk > 0 && num_topk <= kGinChannelThreads / 32);
    EP_HOST_ASSERT(gin_launch.send_counts != nullptr && gin_launch.recv_counts != nullptr);
#define DISPATCH_CHANNEL_LAUNCH_CASE(hidden)                                                       \
  {                                                                                                \
    auto dispatch_func =                                                                           \
        dispatch_channels<LowPrecisionTypeTraits<LowPrecisionType::NVFP4>, hidden>;                \
    if (precision == 0)                                                                            \
      dispatch_func = dispatch_channels<LowPrecisionTypeTraits<LowPrecisionType::FP8>, hidden>;    \
    else if (precision == 2)                                                                       \
      dispatch_func = dispatch_channels<LowPrecisionTypeTraits<LowPrecisionType::BF16>, hidden>;   \
    if (gin_launch.bf16_local_partial_planned) {                                                   \
      EP_HOST_ASSERT(precision == 2 && gin_launch.planned_fanout_cursors != nullptr);              \
      dispatch_func = dispatch_channels_planned_rank_fanout<hidden>;                               \
    }                                                                                              \
    LAUNCH_KERNEL(&cfg, dispatch_func, packed_recv_x, packed_recv_x_scales, packed_recv_src_info,  \
                  packed_recv_layout_range, packed_recv_count, cumulative_local_expert_recv_stats, \
                  rdma_recv_x, rdma_x, x, x_scales, topk_idx, num_tokens,                          \
                  num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank, num_ranks,        \
                  phases, gin_launch);                                                             \
  }                                                                                                \
  break
    SETUP_LAUNCH_CONFIG(gin_launch.num_ctas, kGinChannelThreads, stream);
    SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(DISPATCH_CHANNEL_LAUNCH_CASE);
#undef DISPATCH_CHANNEL_LAUNCH_CASE
    return;
  }
  constexpr int kNumMaxTopK = 9;
  const int num_warp_groups = ceil_div(num_experts, num_device_sms);
  const int num_warps_per_group = 32 / num_warp_groups;
  EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
  EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

  const auto num_warps = num_warp_groups * num_warps_per_group;
  const auto num_sms = ceil_div(num_experts, num_warp_groups);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
  EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

#define DISPATCH_LAUNCH_CASE(hidden)                                                               \
  {                                                                                                \
    auto dispatch_func =                                                                           \
        dispatch_low_precision<LowPrecisionTypeTraits<LowPrecisionType::NVFP4>, hidden>;           \
    if (precision == 0)                                                                            \
      dispatch_func =                                                                              \
          dispatch_low_precision<LowPrecisionTypeTraits<LowPrecisionType::FP8>, hidden>;           \
    else if (precision == 2)                                                                       \
      dispatch_func =                                                                              \
          dispatch_low_precision<LowPrecisionTypeTraits<LowPrecisionType::BF16>, hidden>;          \
    LAUNCH_KERNEL(&cfg, dispatch_func, packed_recv_x, packed_recv_x_scales, packed_recv_src_info,  \
                  packed_recv_layout_range, packed_recv_count, cumulative_local_expert_recv_stats, \
                  rdma_recv_x, rdma_recv_count, rdma_x, x, x_scales, topk_idx,                     \
                  atomic_counter_per_expert, atomic_finish_counter_per_expert, next_clean,         \
                  num_next_clean_int, num_tokens, num_max_dispatch_tokens_per_rank, num_topk,      \
                  num_experts, rank, num_ranks, num_warp_groups, num_warps_per_group,              \
                  phases EP_GIN_PASS);                                                             \
  }                                                                                                \
  break

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
  SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <LowPrecisionType kPrecision, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(1024, 1) void low_precision_combine(
    void* combined_x, void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x, const void* x,
    const float* global_scale_per_token, const int* topk_idx, const float* topk_weights,
    const int* src_info, const int64_t* layout_range, int* next_clean, int num_next_clean_int,
    int* atomic_clean_flag, int num_combined_tokens, int hidden, int num_topk,
    int num_max_dispatch_tokens_per_rank, int num_experts, int rank, int num_ranks,
    int num_warp_groups, int num_warps_per_group, int phases EP_GIN_KERNEL_ARG) {
  using CommVecType = typename LowPrecisionTypeTraits<kPrecision>::CommVecType;
  using ScaleType = typename LowPrecisionTypeTraits<kPrecision>::ScaleType;
  EP_DEVICE_ASSERT(num_topk <= 32);
  constexpr int kBF16ElemsNumPerVecAccess = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kThreadsPerScale =
      LowPrecisionTypeTraits<kPrecision>::kNumElemsPerScale / kBF16ElemsNumPerVecAccess;
  EP_STATIC_ASSERT(kHidden % (32 * kBF16ElemsNumPerVecAccess) == 0, "Invalid vectorization");
  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x);
  const auto warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_local_experts = num_experts / num_ranks;
  const auto warp_group_id = warp_id / num_warps_per_group;
  const auto sub_warp_id = warp_id % num_warps_per_group;
  const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

  // Data type staffs
  constexpr size_t kHiddenBf16VecAccessNum = kHidden / kBF16ElemsNumPerVecAccess;
  constexpr size_t kHiddenCommBytes = kHidden /
                                      LowPrecisionTypeTraits<kPrecision>::kNumElemsPerUnit *
                                      LowPrecisionTypeTraits<kPrecision>::kUnitBytes;
  constexpr size_t kScalesBytes = kHidden / LowPrecisionTypeTraits<kPrecision>::kNumElemsPerScale *
                                  LowPrecisionTypeTraits<kPrecision>::kBytesPerScale;
  constexpr size_t kGlobalScaleBytes =
      LowPrecisionTypeTraits<kPrecision>::kUseGlobalScale ? sizeof(float) : 0;

  // Message package
  constexpr size_t kNumBytesPerSlot =
      (kHiddenCommBytes + kScalesBytes + kGlobalScaleBytes + sizeof(int4) - 1) / sizeof(int4) *
      sizeof(int4);

  // Sending phase
  if ((phases & LOW_LATENCY_SEND_PHASE) == 0) goto LOW_LATENCY_COMBINE_RECV;

  // Clean up next buffer
  if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
    // Notify before executing `int_p`
    __syncwarp();
    if (lane_id == 0) atomic_add_release_global(atomic_clean_flag, num_experts);
  }
  // Issue IBGDA sends
  if (responsible_expert_idx < num_experts) {
    const auto dst_rank = responsible_expert_idx / num_local_experts;
    const auto local_expert_idx = responsible_expert_idx % num_local_experts;
    const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
    const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
    const auto local_x = reinterpret_cast<const int4*>(x) + local_expert_idx * num_ranks *
                                                                num_max_dispatch_tokens_per_rank *
                                                                kHiddenBf16VecAccessNum;
    const auto local_src_info =
        src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
    auto rdma_send_x_current_expert =
        reinterpret_cast<uint8_t*>(rdma_send_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kNumBytesPerSlot;

    // Unpack layout
    int offset, num_tokens_to_send;
    unpack2(layout, num_tokens_to_send, offset);

    // Issue IBGDA send
    for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send;
         token_idx += num_warps_per_group) {
      auto rdma_send_x_vec =
          reinterpret_cast<CommVecType*>(rdma_send_x_current_expert + token_idx * kNumBytesPerSlot);
      auto rdma_send_x_scales_vec = reinterpret_cast<ScaleType*>(
          rdma_send_x_current_expert + token_idx * kNumBytesPerSlot + kHiddenCommBytes);
      float global_scale_val = 0.f;
      if constexpr (LowPrecisionTypeTraits<kPrecision>::kUseGlobalScale) {
        auto rdma_send_x_global_scale_vec =
            reinterpret_cast<float*>(rdma_send_x_current_expert + token_idx * kNumBytesPerSlot +
                                     kHiddenCommBytes + kScalesBytes);
        global_scale_val =
            __ldg(global_scale_per_token +
                  local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank + token_idx);
        rdma_send_x_global_scale_vec[0] = global_scale_val;
      }
      const auto x_int4 = local_x + token_idx * kHiddenBf16VecAccessNum;
      for (int i = lane_id; i < kHiddenBf16VecAccessNum; i += 32) {
        auto int4_value = __ldg(x_int4 + i);
        auto [comm_vec, scale_vec] =
            LowPrecisionTypeTraits<kPrecision>::quantize(int4_value, global_scale_val);
        rdma_send_x_vec[i] = comm_vec;
        if constexpr (LowPrecisionTypeTraits<kPrecision>::kBytesPerScale > 0) {
          if (i % kThreadsPerScale == 0) rdma_send_x_scales_vec[i / kThreadsPerScale] = scale_vec;
        }
      }

      // Copy directly to local rank, or copy to buffer and issue RDMA
      auto src_idx = __ldg(local_src_info + token_idx);
      auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec);
      auto dst_ptr =
          reinterpret_cast<uint64_t>(rdma_recv_x) +
          (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * kNumBytesPerSlot;
      gin_put_warp(gin_launch, reinterpret_cast<void*>(dst_ptr),
                   reinterpret_cast<const void*>(buf_ptr), kNumBytesPerSlot, dst_rank,
                   local_expert_idx);
    }

    // Put the finishing flag
    EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
    asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
    if (sub_warp_id == 1 and lane_id == 0) {
      while (ld_acquire_global(atomic_clean_flag) == 0);
      if (dst_rank != rank) {
        gin_context(gin_launch, local_expert_idx)
            .signal(ncclTeamWorld(gin_launch.comm), dst_rank,
                    ncclGin_StrongSignalInc{
                        static_cast<ncclGinSignal_t>(gin_launch.signal_base + global_expert_idx)});
        gin_finish_peer(gin_launch, dst_rank, local_expert_idx);
      }
      atomic_add_release_global(atomic_clean_flag, -1);
    }
    __syncwarp();
  }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
  if ((phases & LOW_LATENCY_RECV_PHASE) == 0) return;

  // Wait all ranks to arrive
  if (responsible_expert_idx < num_experts) {
    EP_DEVICE_ASSERT(num_warps_per_group > 1);
    if (sub_warp_id == 0 and lane_id == 0) {
      if (responsible_expert_idx / num_local_experts != rank)
        gin_wait_epoch(gin_launch, responsible_expert_idx % num_local_experts,
                       responsible_expert_idx);
    }
  }
  cg::this_grid().sync();
  for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
    int reg_topk_idx[kNumMaxTopk];
    float reg_topk_weights[kNumMaxTopk];
#pragma unroll
    for (int i = 0; i < num_topk; ++i) {
      reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
      reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
    }
    for (auto vec_id = thread_id; vec_id < kHiddenBf16VecAccessNum; vec_id += num_threads) {
      float combined_values[kBF16ElemsNumPerVecAccess] = {0.f};
#pragma unroll
      for (int i = 0; i < num_topk; ++i)
        if (reg_topk_idx[i] >= 0) {
          auto rdma_x_buffer =
              reinterpret_cast<uint8_t*>(rdma_recv_x) +
              (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * kNumBytesPerSlot;
          auto rdma_x_scales_buffer = rdma_x_buffer + kHiddenCommBytes;

          float global_scale_val = 0.f;
          if constexpr (LowPrecisionTypeTraits<kPrecision>::kUseGlobalScale) {
            auto rdma_global_scale_ptr =
                reinterpret_cast<float*>(rdma_x_scales_buffer + kScalesBytes);
            global_scale_val = __ldg(rdma_global_scale_ptr);
          }
          auto x_vec = ld_nc_global(reinterpret_cast<CommVecType*>(rdma_x_buffer) + vec_id);
          ScaleType x_scale{};
          if constexpr (LowPrecisionTypeTraits<kPrecision>::kBytesPerScale > 0)
            x_scale = ld_nc_global(reinterpret_cast<ScaleType*>(rdma_x_scales_buffer) +
                                   vec_id / kThreadsPerScale);
          float x_fp32[kBF16ElemsNumPerVecAccess];
          LowPrecisionTypeTraits<kPrecision>::dequantize(x_vec, global_scale_val, x_scale, x_fp32,
                                                         reg_topk_weights[i]);
#pragma unroll
          for (int j = 0; j < kBF16ElemsNumPerVecAccess; ++j) {
            combined_values[j] += x_fp32[j];
          }
        }

      int4 combined_vec_bf16;
#pragma unroll
      for (int j = 0; j < kBF16ElemsNumPerVecAccess; ++j)
        reinterpret_cast<nv_bfloat16*>(&combined_vec_bf16)[j] =
            static_cast<nv_bfloat16>(combined_values[j]);
      (reinterpret_cast<int4*>(combined_x) + token_idx * kHiddenBf16VecAccessNum)[vec_id] =
          combined_vec_bf16;
    }
  }
}

void low_precision_combine(int precision, void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
                           void* rdma_send_x, const void* x, const float* global_scale_per_token,
                           const int* topk_idx, const float* topk_weights, const int* src_info,
                           const int64_t* layout_range, int* next_clean, int num_next_clean_int,
                           int num_combined_tokens, int hidden, int num_topk,
                           int num_max_dispatch_tokens_per_rank, int num_experts, int rank,
                           int num_ranks, void* workspace, int num_device_sms, cudaStream_t stream,
                           int phases EP_GIN_HOST_ARG) {
  EP_HOST_ASSERT(precision >= 0 && precision <= 2);
  if (gin_launch.bf16_local_partial_planned) {
    EP_HOST_ASSERT(precision == 2 && num_topk >= 1 && num_topk <= 8);
    EP_HOST_ASSERT(gin_launch.num_ctas > 0 && gin_launch.num_ctas <= num_device_sms);
#define PLANNED_LAUNCH_CASE(hidden)                                                             \
  {                                                                                             \
    auto combine_func = combine_bf16_local_partial_planned<hidden>;                             \
    LAUNCH_KERNEL(&cfg, combine_func, combined_x, rdma_recv_x, rdma_send_x, x,                  \
                  num_combined_tokens, num_max_dispatch_tokens_per_rank, num_topk, num_experts, \
                  rank, num_ranks, phases, gin_launch);                                         \
  }                                                                                             \
  break
    SETUP_LAUNCH_CONFIG(gin_launch.num_ctas, 256, stream);
    SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(PLANNED_LAUNCH_CASE);
#undef PLANNED_LAUNCH_CASE
    return;
  }
  if (gin_launch.bf16_local_partial) {
    EP_HOST_ASSERT(precision == 2 && num_topk >= 1 && num_topk <= 8);
    EP_HOST_ASSERT(gin_launch.num_ctas > 0 && gin_launch.num_ctas <= num_device_sms);
#define COMBINE_PARTIAL_LAUNCH_CASE(hidden)                                                 \
  {                                                                                         \
    auto combine_func = combine_bf16_local_partial<hidden>;                                 \
    LAUNCH_KERNEL(&cfg, combine_func, combined_x, rdma_recv_x, rdma_send_x, x, topk_idx,    \
                  topk_weights, src_info, layout_range, num_combined_tokens,                \
                  num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank, num_ranks, \
                  phases, gin_launch);                                                      \
  }                                                                                         \
  break
    SETUP_LAUNCH_CONFIG(gin_launch.num_ctas, 256, stream);
    SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(COMBINE_PARTIAL_LAUNCH_CASE);
#undef COMBINE_PARTIAL_LAUNCH_CASE
    return;
  }
  if (gin_launch.coalesced) {
    EP_HOST_ASSERT(gin_launch.num_ctas > 0 && gin_launch.num_ctas <= num_device_sms);
    EP_HOST_ASSERT(num_topk > 0 && num_topk <= 8);
    EP_HOST_ASSERT(precision != 1 || global_scale_per_token != nullptr);
#define COMBINE_CHANNEL_LAUNCH_CASE(hidden)                                                        \
  {                                                                                                \
    auto combine_func =                                                                            \
        combine_channels<LowPrecisionTypeTraits<LowPrecisionType::FP8>, hidden, 8>;                \
    if (precision == 1)                                                                            \
      combine_func = combine_channels<LowPrecisionTypeTraits<LowPrecisionType::NVFP4>, hidden, 8>; \
    else if (precision == 2)                                                                       \
      combine_func = combine_channels<LowPrecisionTypeTraits<LowPrecisionType::BF16>, hidden, 8>;  \
    LAUNCH_KERNEL(&cfg, combine_func, combined_x, rdma_recv_x, rdma_send_x, x,                     \
                  global_scale_per_token, topk_idx, topk_weights, src_info, layout_range,          \
                  num_combined_tokens, num_topk, num_max_dispatch_tokens_per_rank, num_experts,    \
                  rank, num_ranks, phases, gin_launch);                                            \
  }                                                                                                \
  break
    SETUP_LAUNCH_CONFIG(gin_launch.num_ctas, kGinChannelThreads, stream);
    SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(COMBINE_CHANNEL_LAUNCH_CASE);
#undef COMBINE_CHANNEL_LAUNCH_CASE
    return;
  }
  constexpr int kNumMaxTopk = 9;
  const int num_warp_groups = ceil_div(num_experts, num_device_sms);
  const int num_warps_per_group = 32 / num_warp_groups;
  EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

  const auto num_warps = num_warp_groups * num_warps_per_group;
  const auto num_sms = ceil_div(num_experts, num_warp_groups);

  // Check workspace
  auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
  EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
  EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden)                                                               \
  {                                                                                               \
    auto combine_func = low_precision_combine<LowPrecisionType::FP8, hidden, kNumMaxTopk>;        \
    if (precision == 1) {                                                                         \
      combine_func = low_precision_combine<LowPrecisionType::NVFP4, hidden, kNumMaxTopk>;         \
      EP_HOST_ASSERT(global_scale_per_token != nullptr);                                          \
    } else if (precision == 2) {                                                                  \
      combine_func = low_precision_combine<LowPrecisionType::BF16, hidden, kNumMaxTopk>;          \
    } else {                                                                                      \
      EP_HOST_ASSERT(global_scale_per_token == nullptr);                                          \
    }                                                                                             \
    LAUNCH_KERNEL(&cfg, combine_func, combined_x, rdma_recv_x, rdma_recv_flag, rdma_send_x, x,    \
                  global_scale_per_token, topk_idx, topk_weights, src_info, layout_range,         \
                  next_clean, num_next_clean_int, atomic_clean_flag, num_combined_tokens, hidden, \
                  num_topk, num_max_dispatch_tokens_per_rank, num_experts, rank, num_ranks,       \
                  num_warp_groups, num_warps_per_group, phases EP_GIN_PASS);                      \
  }                                                                                               \
  break

  SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
  SWITCH_HIDDEN_FOR_EXTENSION_KERNELS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}
#undef SWITCH_HIDDEN_FOR_EXTENSION_KERNELS

}  // namespace extensions
}  // namespace deep_ep
