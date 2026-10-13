/*
 * Derived from DeepEP; SM120 adaptation starting-point revision:
 * be2582ffe69b5e7d61c3bc9bf7a5316bc48261f9 (fork revision).
 * Copyright (c) 2025 DeepSeek. SPDX-License-Identifier: MIT.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#include <nccl.h>
#include <nccl_device.h>

#include <algorithm>
#include <cstring>

#include "flashinfer/comm/rdma_moe_alltoall/gin_layout.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/configs.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/extension_api.cuh"
#include "flashinfer/comm/rdma_moe_alltoall/kernels/gin_device.cuh"
#include "rdma_moe_alltoall_runtime_utils.h"
#define GIN_CUDA LL_CUDA
#define GIN_NCCL(call)                                                        \
  do {                                                                        \
    auto result = (call);                                                     \
    LL_CHECK(result == ncclSuccess, #call, ": ", ncclGetErrorString(result)); \
  } while (0)
namespace deep_ep {
struct PlannedViews {
  TensorView rows, weights, tokens, counts, source_ids, source_weights, source_slots;
  std::vector<TensorView> tensors() const {
    return {rows, weights, tokens, counts, source_ids, source_weights, source_slots};
  }
};
class GinBuffer {
 public:
  GinBuffer(int rank, int world_size, const std::string& unique_id, int max_tokens, int hidden,
            int num_experts, bool require_gdaki, int traffic_class,
            const std::string& communication_mode, int num_sms, int num_contexts,
            int communication_precision, const std::string& dispatch_mode,
            const std::string& combine_mode, int dispatch_num_sms, int combine_num_sms,
            int dispatch_num_contexts, int combine_num_contexts,
            const std::string& combine_reduction);
  void dispatch(int precision, TensorView x, Optional<TensorView> scales, TensorView topk_idx,
                Optional<TensorView> recv_stats, TensorView recv_x, TensorView recv_scales,
                TensorView counts, TensorView src_info, TensorView ranges, int phases, int slot,
                const PlannedViews* plan = nullptr, const TensorView* planned_weights = nullptr);
  void combine(int precision, TensorView x, Optional<TensorView> global_scale, TensorView topk_idx,
               TensorView topk_weights, TensorView src_info, TensorView layout_range,
               TensorView output, int phases, int slot, const PlannedViews* plan = nullptr);
  Info backend_info() const;
  void close();
  void abort_initialization();

 private:
  void check_ready() const { LL_CHECK(!closed_, "GinBuffer is closed"); }
  void check_tensor(TensorView t, const char* name, DLDataType dtype, int ndim) const {
    deep_ep::check_tensor(t, name, dtype, ndim, device_);
  }
  void check_plan(const PlannedViews& plan, int tokens, int topk) const;
  void bind_plan(GinLaunch& launch, const PlannedViews& plan) const;
  // Resolved once at construction. Each operation has a disjoint signal
  // namespace, so its expert/channel-to-context mapping never changes while
  // device epochs or captured graphs remain live.
  struct OperationStrategy {
    std::string mode;
    int num_sms_requested = -1, num_contexts_requested = -1;
    int num_sms = 0, num_ctas = 0, num_contexts = 0;
  };
  OperationStrategy dispatch_strategy_, combine_strategy_;
  int rank_, world_size_, max_tokens_, hidden_, num_experts_, local_experts_;
  int device_ = -1, device_num_sms_ = 0;
  int num_sms_requested_ = -1, num_contexts_requested_ = -1, num_contexts_ = 0;
  int runtime_version_ = 0;
  int traffic_class_ = -1;
  int communication_precision_ = -1;
  bool bf16_local_partial_ = false, bf16_local_partial_planned_ = false;
  bool compact_rank_arena_ = false, nvfp4_rank_packet_ = false;
  size_t rank_packet_ready_offset_ = 0, rank_packet_ready_bytes_ = 0;
  size_t partial_arena_offset_ = 0, partial_meta_stride_ = 0, partial_scratch_bytes_ = 0;
  int signal_count_ = 0;
  std::string communication_mode_;
  std::string communication_mode_requested_;
  size_t arena_bytes_ = 0, count_arena_offset_ = 0, count_array_bytes_ = 0;
  size_t fanout_cursor_offset_ = 0, fanout_cursor_bytes_ = 0;
  bool closed_ = false, devcomm_created_ = false, operations_started_ = false;
  ncclComm_t comm_ = nullptr;
  ncclWindow_t window_ = nullptr;
  ncclDevComm devcomm_{};
  ncclCommProperties properties_ = NCCL_COMM_PROPERTIES_INITIALIZER;
  void* arena_ = nullptr;
  void* workspace_ = nullptr;
  uint64_t* epochs_ = nullptr;
};
GinBuffer::GinBuffer(int rank, int world_size, const std::string& unique_id, int max_tokens,
                     int hidden, int num_experts, bool require_gdaki, int traffic_class,
                     const std::string& communication_mode, int num_sms, int num_contexts,
                     int communication_precision, const std::string& dispatch_mode,
                     const std::string& combine_mode, int dispatch_num_sms, int combine_num_sms,
                     int dispatch_num_contexts, int combine_num_contexts,
                     const std::string& combine_reduction)
    : rank_(rank),
      world_size_(world_size),
      max_tokens_(max_tokens),
      hidden_(hidden),
      num_experts_(num_experts),
      local_experts_(world_size > 0 ? num_experts / world_size : 0),
      num_sms_requested_(num_sms),
      num_contexts_requested_(num_contexts),
      traffic_class_(traffic_class),
      communication_precision_(communication_precision),
      bf16_local_partial_(combine_reduction != "fp32"),
      bf16_local_partial_planned_(combine_reduction == "bf16_local_partial_planned"),
      communication_mode_(communication_mode),
      communication_mode_requested_(communication_mode) {
  LL_CHECK(communication_mode == "legacy" || communication_mode == "coalesced",
           "communication_mode must be 'legacy' or 'coalesced'");
  LL_CHECK(communication_precision >= -1 && communication_precision <= 2,
           "communication precision must be -1 (untyped), 0 (FP8), 1 (NVFP4), or 2 (BF16)");
  LL_CHECK(combine_reduction == "fp32" || combine_reduction == "bf16_local_partial" ||
               combine_reduction == "bf16_local_partial_planned",
           "unsupported combine_reduction");
  LL_CHECK(!bf16_local_partial_ || communication_precision == 2,
           "local-partial reduction requires typed BF16");
  LL_CHECK(traffic_class >= -1 && traffic_class <= 255,
           "traffic_class must be -1 (inherit) or 0..255");
  LL_CHECK(world_size > 0 && rank >= 0 && rank < world_size, "invalid EP rank/world size");
  LL_CHECK(num_experts > 0 && num_experts % world_size == 0, "experts must divide EP size");
  LL_CHECK(local_experts_ <= NUM_MAX_LOCAL_EXPERTS, "too many local experts");
  LL_CHECK(max_tokens > 0 && int64_t(max_tokens) * world_size <= std::numeric_limits<int>::max(),
           "invalid dispatch capacity");
  LL_CHECK((int64_t(max_tokens) * world_size) % 4 == 0,
           "EP size * capacity must be divisible by 4");
  LL_CHECK(hidden == 4096 || hidden == 6144 || hidden == 7168,
           "the inherited dispatch/combine kernels support hidden 4096, 6144, or 7168");
  LL_CHECK(unique_id.size() == sizeof(ncclUniqueId), "invalid NCCL unique ID");
  GIN_CUDA(cudaGetDevice(&device_));
  cudaDeviceProp prop{};
  GIN_CUDA(cudaGetDeviceProperties(&prop, device_));
  LL_CHECK(prop.major == 12 && prop.minor == 0, "RDMA MoE AlltoAll currently targets SM120");
  LL_CHECK(prop.cooperativeLaunch, "GIN protocol requires cooperative CUDA launches");
  device_num_sms_ = prop.multiProcessorCount;
  LL_CHECK(num_sms == -1 || (num_sms >= 1 && num_sms <= device_num_sms_),
           "num_sms must be -1 (automatic) or in [1, ", device_num_sms_, "]");
  LL_CHECK(num_contexts == -1 || (num_contexts >= 1 && num_contexts <= local_experts_),
           "num_contexts must be -1 (automatic) or in [1, ", local_experts_, "]");
  auto resolve_strategy = [&](const char* operation, const std::string& mode_override,
                              int sms_override, int contexts_override) {
    OperationStrategy strategy;
    strategy.mode = mode_override.empty() ? communication_mode : mode_override;
    LL_CHECK(strategy.mode == "legacy" || strategy.mode == "coalesced", operation,
             " mode must be legacy or coalesced");
    LL_CHECK(sms_override == -1 || (sms_override >= 1 && sms_override <= device_num_sms_),
             operation, " num_sms must be -1 or in [1, ", device_num_sms_, "]");
    LL_CHECK(
        contexts_override == -1 || (contexts_override >= 1 && contexts_override <= local_experts_),
        operation, " num_contexts must be -1 or in [1, ", local_experts_, "]");
    strategy.num_sms_requested = sms_override == -1 ? num_sms : sms_override;
    strategy.num_contexts_requested = contexts_override == -1 ? num_contexts : contexts_override;
    const bool coalesced = strategy.mode == "coalesced";
    strategy.num_sms = strategy.num_sms_requested == -1
                           ? (coalesced ? std::min(32, device_num_sms_) : device_num_sms_)
                           : strategy.num_sms_requested;
    strategy.num_contexts = strategy.num_contexts_requested == -1
                                ? (coalesced ? std::min(8, local_experts_) : local_experts_)
                                : strategy.num_contexts_requested;
    strategy.num_ctas = strategy.num_sms;
    if (!coalesced) {
      const int num_warp_groups = (num_experts_ + strategy.num_sms - 1) / strategy.num_sms;
      LL_CHECK(num_warp_groups < 15, operation,
               " legacy mode requires ceil(num_experts / num_sms) < 15; use coalesced mode for "
               "fewer SMs");
      strategy.num_ctas = (num_experts_ + num_warp_groups - 1) / num_warp_groups;
      LL_CHECK(strategy.num_ctas > 1, operation, " legacy mode requires more than one CTA");
    }
    return strategy;
  };
  dispatch_strategy_ =
      resolve_strategy("dispatch", dispatch_mode, dispatch_num_sms, dispatch_num_contexts);
  combine_strategy_ =
      resolve_strategy("combine", combine_mode, combine_num_sms, combine_num_contexts);
  // The resolved strategies are immutable. Only this path dispatches rank
  // records and combines rank partials; legacy/default arenas retain E*C.
  compact_rank_arena_ = communication_precision_ == 2 && bf16_local_partial_planned_ &&
                        dispatch_strategy_.mode == "coalesced";
  nvfp4_rank_packet_ = communication_precision_ == 1 && dispatch_strategy_.mode == "legacy" &&
                       max_tokens_ > 32 && num_experts_ >= 2 &&
                       (world_size_ == 4 || world_size_ == 8);
  communication_mode_ =
      dispatch_strategy_.mode == combine_strategy_.mode ? dispatch_strategy_.mode : "mixed";
  const bool any_coalesced =
      dispatch_strategy_.mode == "coalesced" || combine_strategy_.mode == "coalesced";
  const bool audited_completion = any_coalesced || communication_precision_ >= 0;
  num_contexts_ = std::max(dispatch_strategy_.num_contexts, combine_strategy_.num_contexts);
  GIN_NCCL(ncclGetVersion(&runtime_version_));
  // Keep the verified-version admission policy separate from GIN capability
  // checks below. This is not a claim about NCCL's first GIN-capable release.
  LL_CHECK(runtime_version_ >= NCCL_VERSION(2, 30, 7),
           "RDMA MoE AlltoAll does not admit NCCL versions older than 2.30.7");
  LL_CHECK(runtime_version_ == NCCL_VERSION_CODE,
           "NCCL runtime/header mismatch: runtime=", runtime_version_,
           " headers=", NCCL_VERSION_CODE);
  LL_CHECK(!audited_completion || runtime_version_ == NCCL_VERSION(2, 30, 7),
           "RDMA MoE AlltoAll currently verifies NCCL 2.30.7 GDAKI for coalesced mode "
           "and typed shared-send source completion");
  ncclUniqueId id;
  std::memcpy(&id, unique_id.data(), sizeof(id));
  try {
    GIN_NCCL(ncclCommInitRank(&comm_, world_size_, id, rank_));
    GIN_NCCL(ncclCommQueryProperties(comm_, &properties_));
    LL_CHECK(properties_.deviceApiSupport, "NCCL device API is unavailable");
    LL_CHECK(properties_.ginType != NCCL_GIN_TYPE_NONE, "full-connectivity GIN is unavailable");
    LL_CHECK(!require_gdaki || properties_.ginType == NCCL_GIN_TYPE_GDAKI,
             "GDAKI required; actual NCCL GIN type=", int(properties_.ginType),
             ". Proxy transport must not be mistaken for GPU-initiated RDMA.");
    LL_CHECK(
        !audited_completion || properties_.ginType == NCCL_GIN_TYPE_GDAKI,
        "coalesced mode and typed shared-send arenas require GDAKI even when require_gdaki=False; ",
        "its shared-context source-completion protocol has only been audited for GDAKI");
    arena_bytes_ = GinArenaLayout(nullptr, max_tokens_, hidden_, world_size_, num_experts_,
                                  communication_precision_, bf16_local_partial_planned_,
                                  compact_rank_arena_, nvfp4_rank_packet_)
                       .total_bytes;
    if (any_coalesced || nvfp4_rank_packet_) {
      // Each slot owns distinct send/recv vectors. Planned rank fanout adds
      // one packet count per rank to the E expert counts; other modes retain E.
      const bool rank_fanout =
          (bf16_local_partial_planned_ && dispatch_strategy_.mode == "coalesced") ||
          nvfp4_rank_packet_;
      const size_t count_entries = size_t(num_experts_) + (rank_fanout ? world_size_ : 0);
      count_arena_offset_ = arena_bytes_;
      count_array_bytes_ = GinArenaLayout::aligned(sizeof(int) * count_entries, 128);
      arena_bytes_ += 4 * count_array_bytes_;
      if (rank_fanout) {
        fanout_cursor_offset_ = arena_bytes_;
        fanout_cursor_bytes_ = GinArenaLayout::aligned(sizeof(int) * size_t(num_experts_), 128);
        arena_bytes_ += 2 * fanout_cursor_bytes_;
        if (nvfp4_rank_packet_) {
          rank_packet_ready_offset_ = arena_bytes_;
          rank_packet_ready_bytes_ =
              GinArenaLayout::aligned(sizeof(int) * size_t(world_size_), 128);
          arena_bytes_ += 2 * rank_packet_ready_bytes_;
        }
      }
    }
    signal_count_ = 4 * num_experts_;
    if (bf16_local_partial_) {
      partial_arena_offset_ = GinArenaLayout::aligned(arena_bytes_, 128);
      partial_meta_stride_ = GinArenaLayout::aligned(16 + size_t(max_tokens_) * 64, 128);
      partial_scratch_bytes_ = GinArenaLayout::aligned(
          (size_t(num_experts_) + 2 * world_size_) * max_tokens_ * sizeof(int), 128);
      arena_bytes_ = partial_arena_offset_ + (1 + 2 * size_t(world_size_)) * partial_meta_stride_ +
                     2 * partial_scratch_bytes_;
      LL_CHECK(int64_t(signal_count_) + 4LL * world_size_ <= std::numeric_limits<int>::max(),
               "local-partial signal count overflow");
      signal_count_ += 4 * world_size_;
    }
    GIN_NCCL(ncclMemAlloc(&arena_, arena_bytes_));
    GIN_CUDA(cudaMemset(arena_, 0, arena_bytes_));
    GIN_CUDA(cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES));
    GIN_CUDA(cudaMemset(workspace_, 0, NUM_WORKSPACE_BYTES));
    GIN_CUDA(cudaMalloc(reinterpret_cast<void**>(&epochs_), sizeof(uint64_t) * signal_count_));
    GIN_CUDA(cudaMemset(epochs_, 0, sizeof(uint64_t) * signal_count_));
    // Complete local initialization before the collective registration
    // publishes addresses which another rank may immediately write.
    GIN_CUDA(cudaDeviceSynchronize());
    GIN_NCCL(
        ncclCommWindowRegister(comm_, arena_, arena_bytes_, &window_, NCCL_WIN_COLL_SYMMETRIC));
    ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    req.ginContextCount = num_contexts_;
    req.ginSignalCount = signal_count_;
    req.ginExclusiveContexts = true;
    req.ginQueueDepth = 1024;
    // GIN does not inherit the legacy NCCL_IB_TC environment setting.
    if (traffic_class_ >= 0) req.ginTrafficClass = traffic_class_;
    req.ginStrongSignalsRequired = true;
    GIN_NCCL(ncclDevCommCreate(comm_, &req, &devcomm_));
    devcomm_created_ = true;
    LL_CHECK(devcomm_.ginContextCount > 0, "NCCL returned zero GIN contexts");
    LL_CHECK(devcomm_.ginContextCount <= local_experts_,
             "NCCL returned more contexts than the reserved per-phase signal capacity");
    LL_CHECK(devcomm_.ginContextCount >= num_contexts_,
             "NCCL allocated fewer contexts than the resolved dispatch/combine strategies require");
  } catch (...) {
    // Constructor failures and successful peers of a collectively failed
    // initialization use the same local cleanup. Preserve the original
    // error if best-effort cleanup itself encounters a CUDA/NCCL failure.
    try {
      abort_initialization();
    } catch (...) {
    }
    throw;
  }
}

void GinBuffer::check_plan(const PlannedViews& p, int tokens, int topk) const {
  LL_CHECK(tokens >= 0 && tokens <= max_tokens_ && topk >= 1 && topk <= 8,
           "planned source dimensions invalid");
  check_tensor(p.rows, "plan.rows", dl_int64, 3);
  check_tensor(p.weights, "plan.weights", dl_float32, 3);
  check_tensor(p.tokens, "plan.tokens", dl_int32, 2);
  check_tensor(p.counts, "plan.counts", dl_int32, 1);
  check_tensor(p.source_ids, "plan.source_ids", dl_int32, 2);
  check_tensor(p.source_weights, "plan.source_weights", dl_float32, 2);
  check_tensor(p.source_slots, "plan.source_slots", dl_int32, 2);
  LL_CHECK(p.rows.size(0) == world_size_ && p.rows.size(1) == max_tokens_ && p.rows.size(2) == 8 &&
               same_shape(p.rows, p.weights),
           "planned owner route shape mismatch");
  LL_CHECK(p.tokens.size(0) == world_size_ && p.tokens.size(1) == max_tokens_ &&
               same_shape(p.tokens, p.source_slots) && p.counts.size(0) == world_size_,
           "planned compact shape mismatch");
  LL_CHECK(p.source_ids.size(0) == tokens && p.source_ids.size(1) == topk &&
               same_shape(p.source_ids, p.source_weights),
           "planned snapshot shape mismatch");
  auto tensors = p.tensors();
  for (size_t i = 0; i < tensors.size(); ++i)
    for (size_t j = i + 1; j < tensors.size(); ++j)
      LL_CHECK(!overlaps(tensors[i], tensors[j]), "plan tensors must be disjoint");
}
void GinBuffer::bind_plan(GinLaunch& launch, const PlannedViews& p) const {
  launch.bf16_local_partial_planned = true;
  launch.planned_rows = pointer<int64_t>(p.rows);
  launch.planned_owner_weights = pointer<float>(p.weights);
  launch.planned_tokens = pointer<int>(p.tokens);
  launch.planned_counts = pointer<int>(p.counts);
  launch.planned_source_ids = pointer<int>(p.source_ids);
  launch.planned_source_weights = pointer<float>(p.source_weights);
  launch.planned_source_slots = pointer<int>(p.source_slots);
}
void GinBuffer::dispatch(int precision, TensorView x, Optional<TensorView> scales,
                         TensorView topk_idx, Optional<TensorView> recv_stats, TensorView recv_x,
                         TensorView recv_scales, TensorView counts, TensorView src_info,
                         TensorView ranges, int phases, int slot, const PlannedViews* plan,
                         const TensorView* planned_weights) {
  check_ready();
  check_phases(phases);
  LL_CHECK(slot == 0 || slot == 1, "slot must be zero or one");
  ffi::CUDADeviceGuard guard(device_);
  LL_CHECK(precision >= 0 && precision <= 2, "precision must be FP8, NVFP4, or BF16");
  LL_CHECK(communication_precision_ < 0 || precision == communication_precision_,
           "dispatch precision must match this typed GinBuffer's communication_dtype");
  const auto payload_dtype = precision == 2 ? dl_bfloat16 : dl_uint8;
  const auto scale_dtype = precision == 0 ? dl_float32 : dl_uint8;
  const int payload_width = precision == 1 ? hidden_ / 2 : hidden_;
  const int scale_width = precision == 0 ? hidden_ / 128 : precision == 1 ? hidden_ / 16 : 0;
  check_tensor(x, "x", payload_dtype, 2);
  check_vector_alignment(x, "x");
  LL_CHECK(x.size(1) == payload_width, "dispatch hidden mismatch");
  if (precision != 2) {
    LL_CHECK(scales.has_value(), "FP8/NVFP4 dispatch requires block scales");
    check_tensor(scales.value(), "scales", scale_dtype, 2);
    check_vector_alignment(scales.value(), "scales");
    LL_CHECK(scales.value().size(0) == x.size(0) && scales.value().size(1) == scale_width,
             "dispatch block scale shape mismatch");
  } else {
    LL_CHECK(!scales.has_value(), "BF16 dispatch requires scales=None");
  }
  check_tensor(topk_idx, "topk_idx", dl_int32, 2);
  LL_CHECK(x.size(0) <= max_tokens_ && topk_idx.size(0) == x.size(0),
           "dispatch token count mismatch/capacity exceeded");
  LL_CHECK(topk_idx.size(1) > 0 && topk_idx.size(1) <= 8, "top-k must be in [1,8]");
  LL_CHECK(bool(plan) == bf16_local_partial_planned_,
           "planned buffers require gin_dispatch_planned; default buffers reject a plan");
  if (plan) {
    LL_CHECK(precision == 2 && planned_weights, "planned dispatch requires BF16 weights");
    check_tensor(*planned_weights, "topk_weights", dl_float32, 2);
    LL_CHECK(same_shape(topk_idx, *planned_weights), "planned weights shape mismatch");
    check_plan(*plan, int(x.size(0)), int(topk_idx.size(1)));
  }
  if (recv_stats.has_value()) {
    check_tensor(recv_stats.value(), "recv_stats", dl_int32, 1);
    LL_CHECK(recv_stats.value().numel() == local_experts_, "recv_stats shape mismatch");
  }
  const int64_t cap = int64_t(world_size_) * max_tokens_;
  check_tensor(recv_x, "out.recv_x", payload_dtype, 3);
  check_vector_alignment(recv_x, "out.recv_x");
  check_tensor(recv_scales, "out.recv_scales", scale_dtype, 3);
  check_vector_alignment(recv_scales, "out.recv_scales");
  check_tensor(counts, "out.counts", dl_int32, 1);
  check_tensor(src_info, "out.src_info", dl_int32, 2);
  check_tensor(ranges, "out.ranges", dl_int64, 2);
  LL_CHECK(
      recv_x.size(0) == local_experts_ && recv_x.size(1) == cap && recv_x.size(2) == payload_width,
      "out.recv_x shape mismatch");
  LL_CHECK(recv_scales.size(0) == local_experts_ && recv_scales.size(1) == cap &&
               recv_scales.size(2) == scale_width,
           "out.recv_scales shape mismatch");
  LL_CHECK(counts.size(0) == local_experts_, "out.counts shape mismatch");
  LL_CHECK(src_info.size(0) == local_experts_ && src_info.size(1) == cap,
           "out.src_info shape mismatch");
  LL_CHECK(ranges.size(0) == local_experts_ && ranges.size(1) == world_size_,
           "out.ranges shape mismatch");
  std::vector<TensorView> inputs{x, topk_idx};
  if (scales.has_value()) inputs.push_back(scales.value());
  if (recv_stats.has_value()) inputs.push_back(recv_stats.value());
  std::vector<TensorView> outputs{recv_x, recv_scales, counts, src_info, ranges};
  if (plan) {
    inputs.push_back(*planned_weights);
    for (auto tensor : plan->tensors()) outputs.push_back(tensor);
  }
  check_dispatch_overlap(outputs, inputs);
  GinArenaLayout layout(arena_, max_tokens_, hidden_, world_size_, num_experts_,
                        communication_precision_, bf16_local_partial_planned_, compact_rank_arena_,
                        nvfp4_rank_packet_);
  auto buffer = layout.buffers[slot];
  GinLaunch gin{devcomm_, window_, arena_, epochs_, slot * (2 * num_experts_)};
  gin.coalesced = dispatch_strategy_.mode == "coalesced";
  gin.nvfp4_rank_packet = nvfp4_rank_packet_;
  gin.nvfp4_legacy_rank_dedup = !nvfp4_rank_packet_ && communication_precision_ == 1 &&
                                !gin.coalesced && max_tokens_ > 32 && num_experts_ >= 2;
  gin.num_ctas = dispatch_strategy_.num_ctas;
  gin.active_context_count = dispatch_strategy_.num_contexts;
  if (gin.coalesced || gin.nvfp4_rank_packet) {
    auto* counts = static_cast<char*>(arena_) + count_arena_offset_ + slot * 2 * count_array_bytes_;
    gin.send_counts = reinterpret_cast<int*>(counts);
    gin.recv_counts = reinterpret_cast<int*>(counts + count_array_bytes_);
    if (fanout_cursor_bytes_ != 0)
      gin.planned_fanout_cursors = reinterpret_cast<int*>(
          static_cast<char*>(arena_) + fanout_cursor_offset_ + slot * fanout_cursor_bytes_);
  }
  if (gin.nvfp4_rank_packet) {
    // Only native count/range helpers reuse these pointers. No BF16 plan is
    // allocated, exposed in a handle, or read by NVFP4 combine.
    gin.planned_counts = reinterpret_cast<int*>(
        static_cast<char*>(arena_) + rank_packet_ready_offset_ + slot * rank_packet_ready_bytes_);
  }
  auto next_clean = layout.buffers[slot ^ 1].clean_meta();
  auto stream = get_stream(x.device());
  if (plan) {
    bind_plan(gin, *plan);
    gin.planned_input_weights = pointer<float>(*planned_weights);
  }
  operations_started_ = true;
  // Planned rank fanout initializes its private rows inside the SEND kernel,
  // before that kernel's existing initial grid join. Legacy planned dispatch
  // retains the separate memset; all nonplanned paths remain unchanged.
  if (plan && (phases & LOW_LATENCY_SEND_PHASE) && gin.planned_fanout_cursors == nullptr)
    GIN_CUDA(cudaMemsetAsync(gin.planned_rows, 0xff,
                             size_t(world_size_) * max_tokens_ * 8 * sizeof(int64_t), stream));
  extensions::dispatch(precision, recv_x.data_ptr(),
                       precision == 2 ? nullptr : recv_scales.data_ptr(), pointer<int>(src_info),
                       pointer<int64_t>(ranges), pointer<int>(counts),
                       recv_stats.has_value() ? pointer<int>(recv_stats.value()) : nullptr,
                       buffer.dispatch_rdma_recv_data_buffer,
                       buffer.dispatch_rdma_recv_count_buffer, buffer.dispatch_rdma_send_buffer,
                       x.data_ptr(), scales.has_value() ? scales.value().data_ptr() : nullptr,
                       pointer<int>(topk_idx), next_clean.first, next_clean.second, int(x.size(0)),
                       hidden_, max_tokens_, int(topk_idx.size(1)), num_experts_, rank_,
                       world_size_, workspace_, dispatch_strategy_.num_sms, stream, phases, gin);
  GIN_CUDA(cudaGetLastError());
  // Rank fanout writes the bidirectional wire-slot plan in its existing
  // SEND/RECV phases. Legacy planned dispatch still needs the sorted epilogue.
  if (plan && (phases & LOW_LATENCY_RECV_PHASE) && gin.planned_fanout_cursors == nullptr) {
    extensions::finalize_planned_dispatch(int(x.size(0)), int(topk_idx.size(1)), max_tokens_,
                                          world_size_, local_experts_, stream, gin);
    GIN_CUDA(cudaGetLastError());
  }
}
void GinBuffer::combine(int precision, TensorView x, Optional<TensorView> global_scale,
                        TensorView topk_idx, TensorView topk_weights, TensorView src_info,
                        TensorView layout_range, TensorView output, int phases, int slot,
                        const PlannedViews* plan) {
  check_ready();
  check_phases(phases);
  LL_CHECK(slot == 0 || slot == 1, "slot must be zero or one");
  ffi::CUDADeviceGuard guard(device_);
  LL_CHECK(precision >= 0 && precision <= 2, "precision must be FP8, NVFP4, or BF16");
  LL_CHECK(communication_precision_ < 0 || precision == communication_precision_,
           "combine precision must match this typed GinBuffer's communication_dtype");
  LL_CHECK(precision != 1 || global_scale.has_value(), "NVFP4 combine requires global_scale");
  check_tensor(x, "x", dl_bfloat16, 3);
  check_vector_alignment(x, "x");
  check_tensor(topk_idx, "topk_idx", dl_int32, 2);
  check_tensor(topk_weights, "topk_weights", dl_float32, 2);
  LL_CHECK(bool(plan) == bf16_local_partial_planned_,
           "planned buffers require gin_combine_planned; default buffers reject a plan");
  if (plan) check_plan(*plan, int(topk_idx.size(0)), int(topk_idx.size(1)));
  check_tensor(src_info, "src_info", dl_int32, 2);
  check_tensor(layout_range, "layout_range", dl_int64, 2);
  LL_CHECK(x.size(0) == local_experts_ && x.size(1) == int64_t(world_size_) * max_tokens_ &&
               x.size(2) == hidden_,
           "combine expert tensor shape mismatch");
  LL_CHECK(same_shape(topk_idx, topk_weights) && topk_idx.size(0) <= max_tokens_ &&
               topk_idx.size(1) > 0 && topk_idx.size(1) <= 8,
           "combine routing shape mismatch");
  LL_CHECK(src_info.size(0) == local_experts_ && src_info.size(1) == x.size(1),
           "src_info shape mismatch");
  LL_CHECK(layout_range.size(0) == local_experts_ && layout_range.size(1) == world_size_,
           "layout_range shape mismatch");
  if (global_scale.has_value()) {
    check_tensor(global_scale.value(), "global_scale", dl_float32, 3);
    LL_CHECK(global_scale.value().size(0) == x.size(0) &&
                 global_scale.value().size(1) == x.size(1) && global_scale.value().size(2) == 1,
             "global_scale shape mismatch");
  }
  check_tensor(output, "out", dl_bfloat16, 2);
  check_vector_alignment(output, "out");
  LL_CHECK(output.size(0) == topk_idx.size(0) && output.size(1) == hidden_, "out shape mismatch");
  std::vector<TensorView> inputs{x, topk_idx, topk_weights, src_info, layout_range};
  if (plan)
    for (auto tensor : plan->tensors()) inputs.push_back(tensor);
  if (global_scale.has_value()) inputs.push_back(global_scale.value());
  for (auto input : inputs)
    LL_CHECK(!overlaps(output, input), "combine output must not overlap inputs");
  GinArenaLayout layout(arena_, max_tokens_, hidden_, world_size_, num_experts_,
                        communication_precision_, bf16_local_partial_planned_, compact_rank_arena_,
                        nvfp4_rank_packet_);
  auto buffer = layout.buffers[slot];
  GinLaunch gin{devcomm_, window_, arena_, epochs_, slot * (2 * num_experts_) + num_experts_};
  gin.coalesced = combine_strategy_.mode == "coalesced";
  gin.bf16_local_partial = bf16_local_partial_;
  if (plan) bind_plan(gin, *plan);
  if (bf16_local_partial_) {
    auto* partial_base = static_cast<uint8_t*>(arena_) + partial_arena_offset_;
    gin.partial_meta_send = partial_base;
    gin.partial_meta_recv = partial_base + (1 + size_t(slot) * world_size_) * partial_meta_stride_;
    gin.partial_lookup =
        reinterpret_cast<int*>(partial_base + (1 + 2 * size_t(world_size_)) * partial_meta_stride_ +
                               size_t(slot) * partial_scratch_bytes_);
    gin.partial_meta_stride = partial_meta_stride_;
    gin.partial_signal_base = 4 * num_experts_ + slot * 2 * world_size_;
  }
  gin.num_ctas = combine_strategy_.num_ctas;
  gin.active_context_count = combine_strategy_.num_contexts;
  if (gin.coalesced) {
    auto* counts = static_cast<char*>(arena_) + count_arena_offset_ + slot * 2 * count_array_bytes_;
    gin.send_counts = reinterpret_cast<int*>(counts);
    gin.recv_counts = reinterpret_cast<int*>(counts + count_array_bytes_);
    if (fanout_cursor_bytes_ != 0)
      gin.planned_fanout_cursors = reinterpret_cast<int*>(
          static_cast<char*>(arena_) + fanout_cursor_offset_ + slot * fanout_cursor_bytes_);
  }
  auto next_clean = layout.buffers[slot ^ 1].clean_meta();
  auto stream = get_stream(x.device());
  operations_started_ = true;
  extensions::low_precision_combine(
      precision, output.data_ptr(), buffer.combine_rdma_recv_data_buffer,
      buffer.combine_rdma_recv_flag_buffer, buffer.combine_rdma_send_buffer, x.data_ptr(),
      global_scale.has_value() ? pointer<float>(global_scale.value()) : nullptr,
      pointer<int>(topk_idx), pointer<float>(topk_weights), pointer<int>(src_info),
      pointer<int64_t>(layout_range), next_clean.first, next_clean.second, int(topk_idx.size(0)),
      hidden_, int(topk_idx.size(1)), max_tokens_, num_experts_, rank_, world_size_, workspace_,
      combine_strategy_.num_sms, stream, phases, gin);
  GIN_CUDA(cudaGetLastError());
}
Info GinBuffer::backend_info() const {
  Info info;
  info.Set("backend", "nccl_gin");
  info.Set("gin_type", properties_.ginType == NCCL_GIN_TYPE_GDAKI   ? "GDAKI"
                       : properties_.ginType == NCCL_GIN_TYPE_PROXY ? "PROXY"
                       : properties_.ginType == NCCL_GIN_TYPE_GPI   ? "GPI"
                                                                    : "NONE");
  info.Set("gin_type_id", int(properties_.ginType));
  info.Set("device_api_support", properties_.deviceApiSupport);
  info.Set("nccl_header_version", NCCL_VERSION_CODE);
  info.Set("nccl_runtime_version", runtime_version_);
  info.Set("communication_mode", communication_mode_);
  info.Set("communication_mode_requested", communication_mode_requested_);
  auto strategy_info = [&](const OperationStrategy& strategy) {
    Info result;
    result.Set("mode", strategy.mode);
    result.Set("num_sms_requested", optional_int(strategy.num_sms_requested));
    result.Set("num_sms_budget", strategy.num_sms);
    result.Set("num_ctas", strategy.num_ctas);
    result.Set("num_contexts_requested", optional_int(strategy.num_contexts_requested));
    result.Set("num_contexts", strategy.num_contexts);
    return result;
  };
  info.Set("dispatch_strategy", strategy_info(dispatch_strategy_));
  info.Set("combine_strategy", strategy_info(combine_strategy_));
  info.Set("strategy_selection", "fixed_at_initialization");
  info.Set("dispatch_mode", dispatch_strategy_.mode);
  info.Set("combine_mode", combine_strategy_.mode);
  info.Set("transport", "rdma");
  info.Set("lsa_size", devcomm_.lsaSize);
  info.Set("combine_reduction", bf16_local_partial_planned_ ? "bf16_local_partial_planned"
                                : bf16_local_partial_       ? "bf16_local_partial"
                                                            : "fp32");
  info.Set("dispatch_full_header_bytes", bf16_local_partial_planned_ ? 80
                                         : nvfp4_rank_packet_        ? 48
                                                                     : 16);
  const bool planned_rank_major =
      bf16_local_partial_planned_ && dispatch_strategy_.mode == "coalesced";
  info.Set("dispatch_planned_rank_major", planned_rank_major);
  info.Set("dispatch_planned_rank_fanout", planned_rank_major);
  const bool planned_pair2 = planned_rank_major && world_size_ <= 8;
  info.Set("dispatch_planned_put_batching",
           planned_rank_major ? Any(planned_pair2 ? "adjacent_token_pair_contiguous_subset"
                                                  : "single_token_fallback")
                              : Any(nullptr));
  info.Set("dispatch_planned_max_records_per_put",
           planned_rank_major ? Any(planned_pair2 ? 2 : 1) : Any(nullptr));
  info.Set("dispatch_planned_qp_seed",
           planned_rank_major ? Any(planned_pair2 ? "pair_index_plus_owner" : "global_warp")
                              : Any(nullptr));
  const bool planned_source_ready =
      planned_rank_major && world_size_ <= 8 && dispatch_strategy_.num_ctas > 1;
  const int fused_flush_ctas =
      planned_rank_major && dispatch_strategy_.num_ctas > 1 && !planned_source_ready
          ? std::min((dispatch_strategy_.num_contexts * world_size_ + 7) / 8,
                     dispatch_strategy_.num_ctas - 1)
          : 0;
  info.Set("dispatch_planned_source_completion_location",
           planned_rank_major
               ? Any(planned_source_ready ? "inline_publish_flush_overlaps_source_ready_receive"
                                          : "fused_same_kernel_tail_ctas_hooks_inline")
               : Any(nullptr));
  info.Set("dispatch_planned_fused_flush_ctas",
           planned_rank_major ? Any(fused_flush_ctas) : Any(nullptr));
  info.Set("dispatch_planned_fused_data_ctas",
           planned_rank_major
               ? Any(dispatch_strategy_.num_ctas - (planned_source_ready ? 1 : fused_flush_ctas))
               : Any(nullptr));
  info.Set("dispatch_planned_source_ready", planned_source_ready);
  info.Set("dispatch_planned_source_ready_leader_ctas",
           planned_rank_major ? Any(planned_source_ready ? 1 : 0) : Any(nullptr));
  info.Set("dispatch_planned_inline_publisher_ctas",
           planned_rank_major
               ? Any(planned_source_ready
                         ? std::min((dispatch_strategy_.num_contexts * world_size_ + 7) / 8,
                                    dispatch_strategy_.num_ctas)
                         : 0)
               : Any(nullptr));
  info.Set("dispatch_planned_receive_sync",
           planned_rank_major ? Any(planned_source_ready ? "per_source_release_acquire_counts"
                                                         : "all_context_grid_join")
                              : Any(nullptr));
  info.Set("dispatch_planned_source_ready_hook_fallback",
           planned_rank_major ? Any("original_all_context_wait_and_grid_join") : Any(nullptr));
  info.Set(
      "dispatch_planned_row_initialization",
      bf16_local_partial_planned_
          ? Any(planned_rank_major ? "send_kernel_before_initial_grid_join" : "separate_memset")
          : Any(nullptr));
  info.Set("combine_planned_grid_joins", bf16_local_partial_planned_ ? Any(2) : Any(nullptr));
  info.Set("combine_planned_self_transport",
           bf16_local_partial_planned_ ? Any("direct_producer_store_to_receive") : Any(nullptr));
  info.Set("dispatch_kernel", nvfp4_rank_packet_   ? "dispatch_channels_nvfp4_rank_packet"
                              : planned_rank_major ? "dispatch_channels_planned_rank_fanout"
                              : dispatch_strategy_.mode == "coalesced" ? "dispatch_channels"
                                                                       : "dispatch_low_precision");
  info.Set("dispatch_nvfp4_rank_dedup", !nvfp4_rank_packet_ && communication_precision_ == 1 &&
                                            dispatch_strategy_.mode == "legacy" &&
                                            max_tokens_ > 32 && num_experts_ >= 2);
  info.Set("dispatch_nvfp4_rank_dedup_min_capacity", 33);
  info.Set("dispatch_nvfp4_reference_bytes", nvfp4_rank_packet_ ? 0 : 16);
  info.Set("dispatch_nvfp4_rank_packet", nvfp4_rank_packet_);
  info.Set("dispatch_nvfp4_rank_packet_min_capacity", 33);
  info.Set("dispatch_nvfp4_packet_payload",
           nvfp4_rank_packet_ ? Any("one_full_record_per_token_owner_payload_and_scales_unchanged")
                              : Any(nullptr));
  info.Set("dispatch_nvfp4_packet_wire_layout",
           nvfp4_rank_packet_ ? Any("source_rank_capacity_48B_header_full_records") : Any(nullptr));
  info.Set("dispatch_nvfp4_packet_max_records_per_put", nvfp4_rank_packet_ ? Any(1) : Any(nullptr));
  info.Set("dispatch_nvfp4_packet_qp_seed",
           nvfp4_rank_packet_ ? Any("first_route_global_warp") : Any(nullptr));
  info.Set("dispatch_nvfp4_packet_count_stride",
           nvfp4_rank_packet_ ? Any(local_experts_ + 1) : Any(nullptr));
  info.Set("dispatch_nvfp4_packet_source_ready",
           nvfp4_rank_packet_ && dispatch_strategy_.num_ctas > 1);
  info.Set("dispatch_nvfp4_packet_hook_receive",
           nvfp4_rank_packet_ ? Any("all_context_wait_reserve_then_grid_join") : Any(nullptr));
  info.Set("dispatch_nvfp4_packet_source_completion",
           nvfp4_rank_packet_ ? Any("inline_publish_flush_kernel_completion_before_send_reuse")
                              : Any(nullptr));
  info.Set("rank_packet_ready_bytes_per_slot", rank_packet_ready_bytes_);
  info.Set("dispatch_planned_payload",
           planned_rank_major ? Any("one_full_record_per_destination_token") : Any(nullptr));
  info.Set("dispatch_planned_wire_layout",
           planned_rank_major ? Any("source_rank_capacity_full_records") : Any(nullptr));
  info.Set("dispatch_planned_reference_bytes", planned_rank_major ? Any(0) : Any(nullptr));
  info.Set("dispatch_planned_counts",
           planned_rank_major ? Any("per_peer_packet_count_then_all_local_expert_counts")
                              : Any(nullptr));
  info.Set("dispatch_planned_completion",
           planned_rank_major ? Any("original_per_context_strong_signal_and_source_completion")
                              : Any(nullptr));
  info.Set("dispatch_planned_receive_scans", planned_rank_major ? Any(1) : Any(nullptr));
  info.Set("dispatch_planned_scan_packets_per_warp", planned_rank_major ? Any(1) : Any(nullptr));
  info.Set("dispatch_planned_count_stride",
           planned_rank_major ? Any(local_experts_ + 1) : Any(nullptr));
  info.Set("dispatch_planned_range_reservation",
           planned_rank_major
               ? Any(planned_source_ready ? "source_leader_all_channels_wait_then_release_counts"
                                          : "channel_zero_count_acquire_then_existing_grid_join")
               : Any(nullptr));
  info.Set("dispatch_planned_receive_copy",
           planned_rank_major ? Any("one_packet_load_multiple_local_expert_stores") : Any(nullptr));
  info.Set("dispatch_planned_receive_row_broadcast",
           planned_rank_major ? Any("eight_named_64bit_bases_once_per_packet_original_k_order")
                              : Any(nullptr));
  info.Set("dispatch_planned_cursor_bytes_per_slot", fanout_cursor_bytes_);
  info.Set("dispatch_plan_epilogue", bf16_local_partial_planned_ && !planned_rank_major);
  info.Set("dispatch_planned_slot_order",
           bf16_local_partial_planned_
               ? Any(planned_rank_major ? "wire_compact_slot" : "ascending_source_token")
               : Any(nullptr));
  info.Set("dispatch_planned_slot_bijection",
           planned_rank_major ? Any("source_slots_owner_token_inverse_tokens_source_wire_slot")
                              : Any(nullptr));
  info.Set("planned_owner_row_slots_per_token", bf16_local_partial_planned_ ? 8 : 0);
  info.Set("planned_tensors_owned_by_handle", bf16_local_partial_planned_);
  info.Set("partial_metadata_stride", partial_meta_stride_);
  info.Set("partial_scratch_bytes_per_slot", partial_scratch_bytes_);
  info.Set("combine_kernel", bf16_local_partial_planned_ ? "combine_bf16_local_partial_planned"
                             : bf16_local_partial_       ? "combine_bf16_local_partial"
                                                         : (combine_strategy_.mode == "coalesced"
                                                                ? "combine_channels"
                                                                : "low_precision_combine"));
  info.Set("partial_threads_per_cta", bf16_local_partial_ ? Any(256) : Any(nullptr));
  info.Set("partial_original_tokens_per_chunk", bf16_local_partial_planned_ ? Any(2)
                                                : bf16_local_partial_       ? Any(8)
                                                                            : Any(nullptr));
  info.Set("partial_planned_route_cache",
           bf16_local_partial_planned_
               ? Any("first_warp_two_token_eight_lane_groups_original_k_prefix")
               : Any(nullptr));
  const bool planned_combine_defer = bf16_local_partial_planned_ && combine_strategy_.num_ctas > 1;
  info.Set("combine_planned_fused_source_completion",
           bf16_local_partial_planned_
               ? Any(planned_combine_defer ? "same_kernel_tail_cta_overlaps_final_reduction"
                                           : "inline_before_receive")
               : Any(nullptr));
  info.Set("combine_planned_fused_flush_ctas",
           bf16_local_partial_planned_ ? Any(planned_combine_defer ? 1 : 0) : Any(nullptr));
  info.Set("combine_planned_fused_reduce_ctas",
           bf16_local_partial_planned_
               ? Any(combine_strategy_.num_ctas - (planned_combine_defer ? 1 : 0))
               : Any(nullptr));
  info.Set("combine_planned_hook_source_completion",
           bf16_local_partial_planned_ ? Any("inline_before_receive_hook") : Any(nullptr));
  info.Set("partial_payload_bytes_per_row",
           bf16_local_partial_ ? Any(size_t(hidden_) * 2) : Any(nullptr));
  info.Set("partial_metadata_bytes_per_token", bf16_local_partial_planned_ ? Any(0)
                                               : bf16_local_partial_       ? Any(64)
                                                                           : Any(nullptr));
  info.Set("partial_metadata_header_bytes", bf16_local_partial_planned_ ? Any(0)
                                            : bf16_local_partial_       ? Any(16)
                                                                        : Any(nullptr));
  info.Set("partial_metadata_remote_signals",
           bf16_local_partial_ && !bf16_local_partial_planned_ ? Any(world_size_ - 1) : Any(0));
  info.Set("partial_payload_remote_signals", bf16_local_partial_ ? Any(world_size_ - 1) : Any(0));
  info.Set("partial_metadata_context",
           bf16_local_partial_ && !bf16_local_partial_planned_ ? Any(0) : Any(nullptr));
  info.Set("partial_payload_context",
           bf16_local_partial_ ? Any(rank_ % combine_strategy_.num_contexts) : Any(nullptr));
  info.Set("communication_dtype", communication_precision_ < 0
                                      ? Any(nullptr)
                                      : Any(communication_precision_ == 0   ? "fp8"
                                            : communication_precision_ == 1 ? "nvfp4"
                                                                            : "bf16"));
  info.Set("typed_shared_send", communication_precision_ >= 0);
  info.Set("arena_layout",
           communication_precision_ < 0 ? "legacy_bf16_upper_bound_double_send_double_recv"
           : compact_rank_arena_        ? "typed_planned_rank_capacity_single_send_double_recv"
                                        : "typed_single_send_double_recv");
  info.Set("registered_arena_compact_rank_capacity", compact_rank_arena_);
  GinArenaLayout layout(nullptr, max_tokens_, hidden_, world_size_, num_experts_,
                        communication_precision_, bf16_local_partial_planned_, compact_rank_arena_,
                        nvfp4_rank_packet_);
  info.Set("registered_data_capacity_rows", layout.data_capacity_rows);
  info.Set("send_buffer_bytes", layout.send_bytes);
  info.Set("recv_buffer_bytes_per_slot", layout.recv_bytes);
  info.Set("send_buffer_slots", communication_precision_ < 0 ? 2 : 1);
  info.Set("recv_buffer_slots", 2);
  info.Set("metadata_bytes_per_slot", layout.metadata_bytes);
  info.Set("dispatch_msg_bytes",
           communication_precision_ < 0 ? Any(nullptr) : Any(layout.dispatch_msg_bytes));
  info.Set("combine_msg_bytes",
           communication_precision_ < 0 ? Any(nullptr) : Any(layout.combine_msg_bytes));
  info.Set("device_num_sms", device_num_sms_);
  info.Set("num_sms_requested", num_sms_requested_ < 0 ? Any(nullptr) : Any(num_sms_requested_));
  info.Set("num_sms_budget", optional_int(dispatch_strategy_.num_sms == combine_strategy_.num_sms
                                              ? dispatch_strategy_.num_sms
                                              : -1));
  info.Set("num_sms", optional_int(dispatch_strategy_.num_ctas == combine_strategy_.num_ctas
                                       ? dispatch_strategy_.num_ctas
                                       : -1));
  info.Set("dispatch_num_ctas", dispatch_strategy_.num_ctas);
  info.Set("combine_num_ctas", combine_strategy_.num_ctas);
  info.Set("dispatch_num_sms_budget", dispatch_strategy_.num_sms);
  info.Set("combine_num_sms_budget", combine_strategy_.num_sms);
  info.Set("dispatch_num_contexts", dispatch_strategy_.num_contexts);
  info.Set("combine_num_contexts", combine_strategy_.num_contexts);
  info.Set("num_contexts_requested",
           num_contexts_requested_ < 0 ? Any(nullptr) : Any(num_contexts_requested_));
  info.Set("num_contexts", devcomm_.ginContextCount);
  info.Set("gin_context_count_requested", num_contexts_);
  info.Set("gin_context_count", devcomm_.ginContextCount);
  info.Set("gin_signal_count", signal_count_);
  const int rdma_peers = world_size_ - 1;
  auto gin_signals = [&](const OperationStrategy& strategy) {
    return rdma_peers * (strategy.mode == "coalesced" ? strategy.num_contexts : local_experts_);
  };
  const int dispatch_gin_signals = nvfp4_rank_packet_ ? rdma_peers * dispatch_strategy_.num_contexts
                                                      : gin_signals(dispatch_strategy_);
  const int combine_gin_signals = bf16_local_partial_
                                      ? (bf16_local_partial_planned_ ? rdma_peers : 2 * rdma_peers)
                                      : gin_signals(combine_strategy_);
  info.Set("gin_active_remote_signals_per_phase",
           optional_int(dispatch_gin_signals == combine_gin_signals ? dispatch_gin_signals : -1));
  info.Set("dispatch_gin_active_remote_signals", dispatch_gin_signals);
  info.Set("combine_gin_active_remote_signals", combine_gin_signals);
  info.Set("gin_traffic_class_requested", traffic_class_);
  info.Set("arena_bytes", arena_bytes_);
  info.Set("count_arena_bytes", 4 * count_array_bytes_);
  info.Set("fanout_cursor_arena_bytes", 2 * fanout_cursor_bytes_);
  info.Set("rank", rank_);
  info.Set("world_size", world_size_);
  info.Set("closed", closed_);
  return info;
}
void GinBuffer::close() {
  if (closed_) return;
  check_ready();
  ffi::CUDADeviceGuard guard(device_);
  GIN_CUDA(cudaDeviceSynchronize());
  if (devcomm_created_) {
    GIN_NCCL(ncclDevCommDestroy(comm_, &devcomm_));
    devcomm_created_ = false;
  }
  if (window_) {
    GIN_NCCL(ncclCommWindowDeregister(comm_, window_));
    window_ = nullptr;
  }
  if (comm_) {
    GIN_NCCL(ncclCommDestroy(comm_));
    comm_ = nullptr;
  }
  if (epochs_) {
    GIN_CUDA(cudaFree(epochs_));
    epochs_ = nullptr;
  }
  if (workspace_) {
    GIN_CUDA(cudaFree(workspace_));
    workspace_ = nullptr;
  }
  if (arena_) {
    GIN_NCCL(ncclMemFree(arena_));
    arena_ = nullptr;
  }
  closed_ = true;
}
void GinBuffer::abort_initialization() {
  if (closed_) return;
  LL_CHECK(!operations_started_,
           "gin_abort is restricted to initialization before any dispatch/combine");
  ffi::CUDADeviceGuard guard(device_);
  std::string first_error;
  auto record_nccl = [&](ncclResult_t status, const char* operation) {
    if (status != ncclSuccess && first_error.empty())
      first_error = std::string(operation) + ": " + ncclGetErrorString(status);
    return status == ncclSuccess;
  };
  // In the audited NCCL 2.30.7 implementation these two APIs are local:
  // DevCommDestroy releases its local GIN contexts and resource window;
  // WindowDeregister unmaps local imports. Neither waits for another rank.
  // Do this before abort invalidates the parent communicator. CommDestroy
  // is deliberately skipped because initialization need not succeed on all
  // ranks. No user kernels or captured graphs may refer to these resources.
  if (devcomm_created_ && comm_ &&
      record_nccl(ncclDevCommDestroy(comm_, &devcomm_), "ncclDevCommDestroy"))
    devcomm_created_ = false;
  if (window_ && comm_ &&
      record_nccl(ncclCommWindowDeregister(comm_, window_), "ncclCommWindowDeregister"))
    window_ = nullptr;
  if (comm_) {
    const auto status = ncclCommAbort(comm_);
    // Keep the native handle and owned allocations intact if abort fails,
    // so a caller can retry instead of freeing still-registered memory.
    LL_CHECK(status == ncclSuccess, "ncclCommAbort: ", ncclGetErrorString(status));
    comm_ = nullptr;
    window_ = nullptr;
    devcomm_created_ = false;
  }
  auto free_cuda = [&](auto& allocation) {
    if (!allocation) return;
    const auto status = cudaFree(allocation);
    if (status == cudaSuccess)
      allocation = nullptr;
    else if (first_error.empty())
      first_error = std::string("cudaFree: ") + cudaGetErrorString(status);
  };
  free_cuda(epochs_);
  free_cuda(workspace_);
  if (arena_ && record_nccl(ncclMemFree(arena_), "ncclMemFree")) arena_ = nullptr;
  LL_CHECK(first_error.empty(), first_error);
  closed_ = true;
}

RuntimeRegistry<GinBuffer> gin_registry;

Array<int64_t> gin_get_unique_id() {
  ncclUniqueId id;
  GIN_NCCL(ncclGetUniqueId(&id));
  return encode_bytes(&id, sizeof(id));
}
int64_t gin_create(int rank, int world_size, Array<int64_t> unique_id, int max_tokens, int hidden,
                   int num_experts, bool require_gdaki, int traffic_class, String mode, int num_sms,
                   int num_contexts, int precision, String dispatch_mode, String combine_mode,
                   int dispatch_sms, int combine_sms, int dispatch_contexts, int combine_contexts,
                   String combine_reduction) {
  auto runtime = std::make_shared<GinBuffer>(
      rank, world_size, decode_bytes(unique_id, sizeof(ncclUniqueId)), max_tokens, hidden,
      num_experts, require_gdaki, traffic_class, std::string(mode), num_sms, num_contexts,
      precision, std::string(dispatch_mode), std::string(combine_mode), dispatch_sms, combine_sms,
      dispatch_contexts, combine_contexts, std::string(combine_reduction));
  return gin_registry.insert(std::move(runtime));
}
Info gin_info(int64_t handle) { return gin_registry.get(handle)->backend_info(); }
void gin_close(int64_t handle) {
  auto runtime = gin_registry.get(handle);
  runtime->close();
  gin_registry.erase(handle);
}
void gin_abort(int64_t handle) {
  auto runtime = gin_registry.get(handle);
  runtime->abort_initialization();
  gin_registry.erase(handle);
}
void gin_dispatch(int64_t handle, int precision, TensorView x, Optional<TensorView> scales,
                  TensorView topk, Optional<TensorView> stats, TensorView recv_x,
                  TensorView recv_scales, TensorView counts, TensorView src, TensorView ranges,
                  int phases, int slot) {
  gin_registry.get(handle)->dispatch(precision, x, scales, topk, stats, recv_x, recv_scales, counts,
                                     src, ranges, phases, slot);
}
void gin_combine(int64_t handle, int precision, TensorView x, Optional<TensorView> global_scale,
                 TensorView topk, TensorView weights, TensorView src, TensorView ranges,
                 TensorView out, int phases, int slot) {
  gin_registry.get(handle)->combine(precision, x, global_scale, topk, weights, src, ranges, out,
                                    phases, slot);
}
void gin_dispatch_planned(int64_t handle, int precision, TensorView x, Optional<TensorView> scales,
                          TensorView topk, Optional<TensorView> stats, TensorView recv_x,
                          TensorView recv_scales, TensorView counts, TensorView src,
                          TensorView ranges, TensorView weights, TensorView owner_rows,
                          TensorView owner_weights, TensorView owner_tokens,
                          TensorView owner_counts, TensorView source_ids, TensorView source_weights,
                          TensorView source_slots, int phases, int slot) {
  PlannedViews plan{owner_rows, owner_weights,  owner_tokens, owner_counts,
                    source_ids, source_weights, source_slots};
  gin_registry.get(handle)->dispatch(precision, x, scales, topk, stats, recv_x, recv_scales, counts,
                                     src, ranges, phases, slot, &plan, &weights);
}
void gin_combine_planned(int64_t handle, TensorView x, TensorView src, TensorView ranges,
                         TensorView out, TensorView owner_rows, TensorView owner_weights,
                         TensorView owner_tokens, TensorView owner_counts, TensorView source_ids,
                         TensorView source_weights, TensorView source_slots, int phases, int slot) {
  PlannedViews plan{owner_rows, owner_weights,  owner_tokens, owner_counts,
                    source_ids, source_weights, source_slots};
  gin_registry.get(handle)->combine(2, x, Optional<TensorView>(), source_ids, source_weights, src,
                                    ranges, out, phases, slot, &plan);
}
}  // namespace deep_ep
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_get_unique_id, deep_ep::gin_get_unique_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_create, deep_ep::gin_create);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_info, deep_ep::gin_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_close, deep_ep::gin_close);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_abort, deep_ep::gin_abort);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_dispatch, deep_ep::gin_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_combine, deep_ep::gin_combine);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_dispatch_planned, deep_ep::gin_dispatch_planned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gin_combine_planned, deep_ep::gin_combine_planned);
