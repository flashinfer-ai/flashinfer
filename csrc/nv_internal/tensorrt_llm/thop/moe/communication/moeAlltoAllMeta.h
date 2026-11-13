/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace moe_comm
{

// Per-rank workspace layout of the NVLink one-sided all-to-all, computed once at allocation from
// the configuration fields below. OFFSET fields are byte offsets from the rank's workspace slice,
// SIZE fields byte capacities; every extent follows the allocation-time capacities, never a
// step's token count. Two groups of regions:
//   fence     round state, dispatch counters, flags and routing tables, the dispatch receive
//             region (token slices packed per step), combine flags and the combine input region
//             the peers pull from;
//   sentinel  round state, the landing rows the dispatch and the combine push into and the
//             receive buffers the dispatch copies the landing rows into (rows packed per step,
//             like the fence receive region, each buffer at a fixed offset); all zero when
//             SENTINEL_MAX_NUM_TOKENS is zero.
// A sentinel step also fills and reads the fence routing table TOPK_TARGET_RANKS, and its expert
// output may sit in the combine input region.
enum MoeA2AMetaInfoIndex : int64_t
{
    // Configuration.
    // Number of ranks in the EP group.
    EP_SIZE_INDEX = 0,
    // Maximum input token count per rank.
    MAX_NUM_TOKENS_INDEX = 1,
    // Experts selected per token.
    TOP_K_INDEX = 2,
    // Expert statistics gathered from each rank; zero disables EPLB storage.
    EPLB_STATS_NUM_EXPERTS_INDEX = 3,
    // Largest token count per rank of a sentinel step; zero: no sentinel regions.
    SENTINEL_MAX_NUM_TOKENS_INDEX = 4,
    // Payload format (moe_alltoall_sentinel::PayloadFormat) and hidden size of the landing rows.
    SENTINEL_PAYLOAD_FORMAT_INDEX = 5,
    SENTINEL_HIDDEN_SIZE_INDEX = 6,
    // Bytes per rank, every region and alignment padding included.
    WORKSPACE_SIZE_INDEX = 7,

    // Fence protocol: round state and dispatch.
    // uint32_t round counter advanced by dispatch and prepare-combine: sync epochs and parity.
    FLAG_VAL_OFFSET_INDEX = 8,
    // int32_t counting token CTAs to elect the last CTA that publishes dispatch counts.
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = 9,
    // int32_t[ep_size]: outgoing token counts / slot allocators, by destination rank.
    SEND_COUNTERS_OFFSET_INDEX = 10,
    // int32_t[2][ep_size]: incoming counts by round parity and source rank.
    RECV_COUNTERS_OFFSET_INDEX = 11,
    // uint32_t[ep_size]: per-peer epoch flags of the dispatch.
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = 12,
    // int32_t[max_tokens][top_k]: destination rank of each local token's slot k.
    TOPK_TARGET_RANKS_OFFSET_INDEX = 13,
    // int32_t[max_tokens][top_k]: its receive-slot index within the destination's source slice.
    TOPK_TARGET_INDICES_OFFSET_INDEX = 14,
    // int32_t[ep_size][eplb_stats_num_experts]: gathered per-rank expert statistics.
    EPLB_GATHERED_STATS_OFFSET_INDEX = 15,
    // Receive region of the dispatched payloads (hidden states, scales, expert ids, weights,
    // extra payloads), each payload starting on a kWorkspaceAlignment boundary.
    DISPATCH_PAYLOAD_OFFSET_INDEX = 16,
    DISPATCH_PAYLOAD_SIZE_INDEX = 17,

    // Fence protocol: combine.
    // uint32_t[ep_size]: per-peer readiness epochs of the combine.
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 18,
    // Expert outputs the peers pull from; the split layer's expert computation may write into it.
    COMBINE_INPUT_OFFSET_INDEX = 19,
    COMBINE_INPUT_SIZE_INDEX = 20,

    // Sentinel protocol (flashinfer/comm/moe_alltoall_sentinel.cuh). Every row group holds
    // sentinel_max_tokens * ep_size rows (source rank, token): landing row token * ep_size + src,
    // fixed across steps; receive row src * tokens_per_rank + token, tokens_per_rank being the
    // step's.
    // int32 round-state words, then a uint64 receive-source mask per token slot.
    SENTINEL_STATE_OFFSET_INDEX = 21,
    // Landing rows of the hidden states and of their scales, in the wire format.
    SENTINEL_PAYLOAD_ROWS_OFFSET_INDEX = 22,
    SENTINEL_SCALE_ROWS_OFFSET_INDEX = 23,
    // Landing rows of the routing weights, then two sets of expert-id rows (ID_ROW_SET_SIZE each).
    SENTINEL_WEIGHT_ROWS_OFFSET_INDEX = 24,
    SENTINEL_ID_ROWS_OFFSET_INDEX = 25,
    SENTINEL_ID_ROW_SET_SIZE_INDEX = 26,
    // Combine landing rows: one bf16 partial row per (source rank, token).
    SENTINEL_COMBINE_ROWS_OFFSET_INDEX = 27,
    // Receive buffers of the dispatched hidden states and scales, in the payload format, and of
    // the expert ids and weights (int32_t / 4-byte [rows][top_k]).
    SENTINEL_RECV_PAYLOAD_OFFSET_INDEX = 28,
    SENTINEL_RECV_SCALE_OFFSET_INDEX = 29,
    SENTINEL_RECV_IDS_OFFSET_INDEX = 30,
    SENTINEL_RECV_WEIGHTS_OFFSET_INDEX = 31,

    // Number of int64_t entries in the metadata tensor; not a stored layout field.
    NUM_METAINFO_FIELDS = 32
};

using MoeA2AWorkspaceLayout = std::array<int64_t, NUM_METAINFO_FIELDS>;
static constexpr int64_t kWorkspaceAlignment = 256;

inline std::vector<std::pair<char const*, int64_t>> getMoeA2AMetaInfoIndexPairs()
{
    using namespace tensorrt_llm::kernels::moe_comm;
    return {
        {"MOE_A2A_EP_SIZE_INDEX", EP_SIZE_INDEX},
        {"MOE_A2A_MAX_NUM_TOKENS_INDEX", MAX_NUM_TOKENS_INDEX},
        {"MOE_A2A_TOP_K_INDEX", TOP_K_INDEX},
        {"MOE_A2A_EPLB_STATS_NUM_EXPERTS_INDEX", EPLB_STATS_NUM_EXPERTS_INDEX},
        {"MOE_A2A_SENTINEL_MAX_NUM_TOKENS_INDEX", SENTINEL_MAX_NUM_TOKENS_INDEX},
        {"MOE_A2A_SENTINEL_PAYLOAD_FORMAT_INDEX", SENTINEL_PAYLOAD_FORMAT_INDEX},
        {"MOE_A2A_SENTINEL_HIDDEN_SIZE_INDEX", SENTINEL_HIDDEN_SIZE_INDEX},
        {"MOE_A2A_WORKSPACE_SIZE_INDEX", WORKSPACE_SIZE_INDEX},
        {"MOE_A2A_FLAG_VAL_OFFSET_INDEX", FLAG_VAL_OFFSET_INDEX},
        {"MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX", LOCAL_TOKEN_COUNTER_OFFSET_INDEX},
        {"MOE_A2A_SEND_COUNTERS_OFFSET_INDEX", SEND_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_RECV_COUNTERS_OFFSET_INDEX", RECV_COUNTERS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX", DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX", TOPK_TARGET_RANKS_OFFSET_INDEX},
        {"MOE_A2A_TOPK_TARGET_INDICES_OFFSET_INDEX", TOPK_TARGET_INDICES_OFFSET_INDEX},
        {"MOE_A2A_EPLB_GATHERED_STATS_OFFSET_INDEX", EPLB_GATHERED_STATS_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_PAYLOAD_OFFSET_INDEX", DISPATCH_PAYLOAD_OFFSET_INDEX},
        {"MOE_A2A_DISPATCH_PAYLOAD_SIZE_INDEX", DISPATCH_PAYLOAD_SIZE_INDEX},
        {"MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX", COMBINE_COMPLETION_FLAGS_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_INPUT_OFFSET_INDEX", COMBINE_INPUT_OFFSET_INDEX},
        {"MOE_A2A_COMBINE_INPUT_SIZE_INDEX", COMBINE_INPUT_SIZE_INDEX},
        {"MOE_A2A_SENTINEL_STATE_OFFSET_INDEX", SENTINEL_STATE_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_PAYLOAD_ROWS_OFFSET_INDEX", SENTINEL_PAYLOAD_ROWS_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_SCALE_ROWS_OFFSET_INDEX", SENTINEL_SCALE_ROWS_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_WEIGHT_ROWS_OFFSET_INDEX", SENTINEL_WEIGHT_ROWS_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_ID_ROWS_OFFSET_INDEX", SENTINEL_ID_ROWS_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_ID_ROW_SET_SIZE_INDEX", SENTINEL_ID_ROW_SET_SIZE_INDEX},
        {"MOE_A2A_SENTINEL_COMBINE_ROWS_OFFSET_INDEX", SENTINEL_COMBINE_ROWS_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_RECV_PAYLOAD_OFFSET_INDEX", SENTINEL_RECV_PAYLOAD_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_RECV_SCALE_OFFSET_INDEX", SENTINEL_RECV_SCALE_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_RECV_IDS_OFFSET_INDEX", SENTINEL_RECV_IDS_OFFSET_INDEX},
        {"MOE_A2A_SENTINEL_RECV_WEIGHTS_OFFSET_INDEX", SENTINEL_RECV_WEIGHTS_OFFSET_INDEX},
        {"MOE_A2A_NUM_METAINFO_FIELDS", NUM_METAINFO_FIELDS},
        {"MOE_A2A_MAX_RANKS", kMaxRanks},
        {"MOE_A2A_MAX_TOP_K", kMaxTopK},
        {"MOE_A2A_MAX_PAYLOADS", kMaxPayloads},
        {"MOE_A2A_WORKSPACE_ALIGNMENT", kWorkspaceAlignment},
    };
}

} // namespace moe_comm
} // namespace torch_ext

TRTLLM_NAMESPACE_END
