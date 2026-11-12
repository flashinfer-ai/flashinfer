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
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, MIT licensed;
 * see LICENSE.DeepGEMM at the root of this package.
 */

// Host planner of the experimental MOSCA MegaMoE backend (SM100, FP8 x FP4).
//
// This module holds no kernel. It is compiled once per process and exports
// two pure-host functions that the Python backend calls before it allocates a
// workspace and before every launch:
//
//   * `layout`: the byte layout of one rank's symmetric buffer. It builds the
//     same `layout::MegaMoEBuffer` that the kernel derives from its template
//     arguments, from the vendored header, so host and kernel cannot disagree.
//     With `gather` set, the in-kernel all-gather's output region
//     [num_ranks * num_max_tokens_per_rank, hidden] BF16 follows the MegaMoE
//     buffer at align(bytes, 256), the address the kernel computes under
//     -DDGX_BCAST.
//   * `plan`: the launch configuration of one call (tiles, store block,
//     pipeline stages, shared memory, thread split) for a per-rank token
//     count: DeepGEMM's `get_mega_moe_config` heuristic, optionally replaced
//     by a MOSCA tile rule that is validated against the kernel's static
//     constraints exactly as upstream does (a violation raises; it never falls
//     back silently).
//
// Upstream functions this file reproduces (DeepGEMM revision in ../../VENDOR.md):
// `csrc/apis/mega_moe.hpp` (the ring sizing of
// `get_symm_buffer_size_for_mega_moe`) and
// `csrc/jit_kernels/heuristics/mega_moe.hpp` (`get_block_config_for_mega_moe`,
// `get_pipeline_config_for_mega_moe`, `get_mega_moe_config`), restricted to
// the FP8 x FP4 MMA kind, no shared experts, BF16 combine and no per-token x
// scales (the contract of this backend). Nothing is printed. MOSCA's host
// additions are code of this file, not of upstream: the tile rule (an
// argument of `plan`) and the gathered output region of `layout`.

// clang-format off
// The common shim defines DG_UNIFIED_ASSERT and must precede every deep_gemm header.
#include "common/deep_gemm_host_shim.h"
// clang-format on

#include <tvm/ffi/container/array.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tvm_ffi_utils.h"

namespace mosca_megamoe_host {

using deep_gemm::MmaKind;
namespace layout = deep_gemm::layout;
namespace sched = deep_gemm::sched;

constexpr MmaKind kMmaKind = MmaKind::MXFP8FP4;
// `SM100ArchSpec::smem_capacity` of the upstream revision.
constexpr int kSmemCapacity = 232448;
constexpr int kNumUtccpAlignedElems = 128;
constexpr int kGatherAlignment = 256;

int align_up(int x, int a) { return (x + a - 1) / a * a; }
int64_t align_up64(int64_t x, int64_t a) { return (x + a - 1) / a * a; }
int ceil_div(int a, int b) { return (a + b - 1) / b; }

bool is_candidate_block_m(int block_m) {
  return std::any_of(layout::kCandidateBlockM,
                     layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
                     [=](int c) { return c == block_m; });
}

[[noreturn]] void fail(const std::string& what) {
  TVM_FFI_THROW(ValueError) << "MOSCA MegaMoE: " << what;
  __builtin_unreachable();
}

[[noreturn]] void fail_rule(const std::string& what) { fail("tile_rule: " + what); }

// ---------------------------------------------------------------------------------------------
// Ring sizing (upstream `get_ring_tokens_for_mega_moe`).
// ---------------------------------------------------------------------------------------------
std::pair<int, int> ring_tokens(int num_ranks, int num_experts, int num_max_tokens_per_rank,
                                int num_topk, int hidden, int intermediate_hidden, int num_sms) {
  const int num_experts_per_rank = num_experts / num_ranks;
  const int num_active_topk = std::min(num_topk, num_experts_per_rank);
  const int num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;

  int num_ring_tokens = 0;
  for (const int block_m : layout::kCandidateBlockM) {
    const int num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
    const int num_live_pool_blocks =
        sched::get_num_max_live_pool_blocks(num_pool_blocks, num_sms, hidden, intermediate_hidden);
    num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
  }
  num_ring_tokens = align_up(num_ring_tokens, layout::kLCMCandidateBlockM);
  if (num_max_tokens_per_rank >= 6144) {
    // Upstream keeps the non-wrapping prefill path for large prefill allocations.
    num_ring_tokens = layout::get_num_max_pool_tokens<int>(num_ranks, num_max_tokens_per_rank,
                                                           num_topk, num_experts_per_rank);
  }

  int num_sf_ring_tokens = 0;
  for (const int block_m : layout::kCandidateBlockM)
    num_sf_ring_tokens =
        std::max(num_sf_ring_tokens, layout::get_num_sf_ring_tokens<int>(num_ring_tokens, block_m));
  return {num_ring_tokens, num_sf_ring_tokens};
}

void check_geometry(int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
                    int64_t num_topk, int64_t hidden, int64_t intermediate_hidden,
                    int64_t num_sms) {
  if (num_ranks < 1 || num_ranks > static_cast<int64_t>(layout::kNumMaxRanks))
    fail("num_ranks must be in [1, " + std::to_string(layout::kNumMaxRanks) + "], got " +
         std::to_string(num_ranks));
  if (num_experts <= 0 || num_experts % num_ranks != 0)
    fail("num_experts (" + std::to_string(num_experts) +
         ") must be a positive multiple of num_ranks");
  if (num_max_tokens_per_rank <= 0 || num_max_tokens_per_rank % layout::kLCMCandidateBlockM != 0)
    fail("num_max_tokens_per_rank must be a positive multiple of " +
         std::to_string(layout::kLCMCandidateBlockM) + " (align it first)");
  if (num_topk < 1 || num_topk > 32 || num_topk > num_experts)
    fail("top_k must be in [1, min(32, num_experts)], got " + std::to_string(num_topk));
  if (hidden % 256 != 0 || intermediate_hidden % 128 != 0 || (2 * intermediate_hidden) % 256 != 0)
    fail("hidden must be a multiple of 256 and intermediate a multiple of 128");
  if (num_sms < 2 || num_sms % 2 != 0)
    fail("num_sms must be even (2-CTA clusters), got " + std::to_string(num_sms));
}

// Byte layout of one rank's symmetric buffer. Returns, in order:
//   [0] total bytes to allocate (including the gathered region when `gather`)
//   [1] MegaMoE buffer bytes (the kernel's `buffer.get_num_bytes()`)
//   [2] gathered region offset (0 when not `gather`)
//   [3] num_ring_tokens, [4] num_sf_ring_tokens
//   [5] x offset, [6] x_sf offset, [7] topk_idx offset, [8] topk_weights offset,
//   [9] l1_acts offset, [10] l1_acts_sf offset, [11] l2_acts offset, [12] l2_acts_sf offset
// All offsets are bytes from the start of the buffer. Shapes follow upstream's slicing:
//   x [T_max, hidden] e4m3; x_sf [T_max, hidden / 128] int32 (K-major);
//   topk_idx [T_max, top_k] int64; topk_weights [T_max, top_k] float32;
//   l1_acts [ring, hidden] e4m3; l1_acts_sf [hidden / 128, sf_ring] int32 viewed with strides (1,
//   sf_ring); l2_acts [ring, intermediate] e4m3; l2_acts_sf [intermediate / 128, sf_ring] int32
//   likewise.
tvm::ffi::Array<int64_t> Layout(int64_t num_ranks, int64_t num_experts,
                                int64_t num_max_tokens_per_rank, int64_t num_topk, int64_t hidden,
                                int64_t intermediate_hidden, int64_t num_sms, bool gather) {
  check_geometry(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden,
                 intermediate_hidden, num_sms);
  const auto [num_ring_tokens, num_sf_ring_tokens] = ring_tokens(
      static_cast<int>(num_ranks), static_cast<int>(num_experts),
      static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
      static_cast<int>(hidden), static_cast<int>(intermediate_hidden), static_cast<int>(num_sms));
  if (num_sf_ring_tokens % 4 != 0) fail("internal: num_sf_ring_tokens must be a multiple of 4");

  const auto buffer = layout::MegaMoEBuffer(
      nullptr, static_cast<uint32_t>(hidden), static_cast<uint32_t>(intermediate_hidden),
      static_cast<uint32_t>(num_ranks), static_cast<uint32_t>(num_experts),
      static_cast<uint32_t>(num_max_tokens_per_rank), static_cast<uint32_t>(num_topk),
      static_cast<uint32_t>(num_ring_tokens), static_cast<uint32_t>(num_sf_ring_tokens), kMmaKind,
      /*num_shared_experts=*/0u, /*use_fp8_combine=*/false);
  const auto offset = [](const layout::Buffer& b) { return reinterpret_cast<int64_t>(b.base); };

  const int64_t megamoe_bytes = buffer.get_num_bytes();
  int64_t total_bytes = megamoe_bytes;
  int64_t gathered_offset = 0;
  if (gather) {
    gathered_offset = align_up64(megamoe_bytes, kGatherAlignment);
    total_bytes = gathered_offset + num_ranks * num_max_tokens_per_rank * hidden * 2;
  }
  return tvm::ffi::Array<int64_t>(std::vector<int64_t>{
      total_bytes, megamoe_bytes, gathered_offset, num_ring_tokens, num_sf_ring_tokens,
      offset(buffer.input_token_buffer), offset(buffer.input_sf_buffer),
      offset(buffer.input_topk_idx_buffer), offset(buffer.input_topk_weights_buffer),
      offset(buffer.l1_token_buffer), offset(buffer.l1_sf_buffer), offset(buffer.l2_token_buffer),
      offset(buffer.l2_sf_buffer)});
}

// ---------------------------------------------------------------------------------------------
// Heuristic (upstream `get_block_config_for_mega_moe`, `get_smem_breakdown_for_mega_moe`,
// `get_mega_moe_config`).
// ---------------------------------------------------------------------------------------------
struct Config {
  int block_m, block_n, block_k, load_block_m, load_block_n, store_block_m;
  int sf_block_m, sf_block_n, gran_k;
  int num_ring_tokens, num_sf_ring_tokens;
  int swizzle_acts_mode, swizzle_weights_mode;
  int num_stages, smem_size;
  int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
  int num_bytes_per_pull;
};

int get_store_block_m(int block_m) {
  return block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 : 40;
}

struct BlockConfig {
  int block_m, store_block_m, block_k, num_epilogue_threads;
};

BlockConfig block_config(int num_ranks, int num_experts, int num_topk, int num_tokens) {
  const float num_expected_tokens =
      static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
  const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);
  int block_m = num_expected_tokens <= 10 ? 16 : 32;
  if (num_expected_tokens > 24) {
    int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
    if (num_blocks == 1 && num_covered_tokens > 192 && num_experts / num_ranks < 14) num_blocks = 2;
    for (const int candidate : {64, 128, 192, 240}) {
      block_m = candidate;
      if (num_blocks * candidate >= num_covered_tokens) break;
    }
  }
  const int block_k = 128 * 8 / deep_gemm::get_element_bits(kMmaKind);
  if (!is_candidate_block_m(block_m)) fail("internal: heuristic block_m is not a candidate");
  return {block_m, get_store_block_m(block_m), block_k, 2 * 128};
}

struct SmemBreakdown {
  int smem_fixed, smem_size_per_stage, smem_non_reusable_fixed, smem_stage_barriers;
};

SmemBreakdown smem_breakdown(int num_experts, int block_m, int block_n, int block_k,
                             int num_bytes_per_pull, int store_block_m, int sf_block_m,
                             int sf_block_n, int gran_k, int num_dispatch_warps,
                             int num_epilogue_warps) {
  constexpr int kSmemAlignment = 1024;
  constexpr int kNumEpilogueStages = 2;
  constexpr int kNumTMAStoreStages = 2;
  constexpr int kNumScheduleStages = 2;
  const int elem_bits = deep_gemm::get_element_bits(kMmaKind);
  const int load_block_m = block_m / 2;

  const int smem_expert_count_size =
      align_up(num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
  const int smem_send_buffers_size = align_up(
      static_cast<int>(
          layout::Buffer(layout::Data(num_bytes_per_pull), num_dispatch_warps, 1).get_num_bytes()),
      kSmemAlignment);
  const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

  const int num_epilogue_warpgroups = num_epilogue_warps / 4;
  const int smem_cd_l1 =
      num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages * elem_bits / 8;
  const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * 2;
  const int smem_cd = align_up(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

  const int smem_task_info = kNumScheduleStages * static_cast<int>(sizeof(sched::TaskInfo<true>));
  const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2 +
                             kNumScheduleStages * 2) *
                            8;
  const int smem_amax_reduction =
      store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));
  const int smem_tmem_ptr = 4;

  const int smem_sfa_per_stage = sf_block_m * (block_k / gran_k);
  const int smem_sfb_per_stage = sf_block_n * (block_k / gran_k);
  const int smem_a_size_per_stage = load_block_m * block_k * elem_bits / 8;
  const int smem_b_size_per_stage = block_n * block_k * elem_bits / 8;
  if (smem_a_size_per_stage % kSmemAlignment != 0 || smem_b_size_per_stage % kSmemAlignment != 0)
    fail("internal: per-stage A/B tiles must be 1024-byte aligned");
  const int smem_stage_barriers = 2 * 8;
  const int smem_size_per_stage = smem_a_size_per_stage + smem_b_size_per_stage +
                                  smem_sfa_per_stage + smem_sfb_per_stage + smem_stage_barriers;
  const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers +
                         smem_task_info + smem_tmem_ptr;
  return {smem_fixed, smem_size_per_stage, smem_barriers + smem_tmem_ptr, smem_stage_barriers};
}

Config heuristic_config(int num_ranks, int num_experts, int num_tokens, int num_topk, int hidden,
                        int num_ring_tokens, int num_sf_ring_tokens) {
  const auto bc = block_config(num_ranks, num_experts, num_topk, num_tokens);
  if (num_ring_tokens % bc.block_m != 0)
    fail("internal: num_ring_tokens must be a multiple of block_m");
  Config c{};
  c.block_m = bc.block_m;
  c.block_n = 128;
  c.block_k = bc.block_k;
  c.load_block_m = c.block_m / 2;
  c.load_block_n = c.block_n;
  c.store_block_m = bc.store_block_m;
  c.sf_block_m = align_up(c.block_m, kNumUtccpAlignedElems);
  c.sf_block_n = align_up(c.block_n, kNumUtccpAlignedElems);
  c.gran_k = deep_gemm::get_sf_gran_k(kMmaKind);
  c.num_ring_tokens = num_ring_tokens;
  c.num_sf_ring_tokens = num_sf_ring_tokens;
  c.swizzle_acts_mode = 128;
  c.swizzle_weights_mode = 128;
  c.num_dispatch_threads = 128;
  c.num_non_epilogue_threads = 128;
  c.num_epilogue_threads = bc.num_epilogue_threads;
  c.num_bytes_per_pull = hidden * deep_gemm::get_element_bits(kMmaKind) / 8;
  while (c.num_bytes_per_pull > 8192) {
    if (c.num_bytes_per_pull % 2 != 0) fail("internal: odd pull size");
    c.num_bytes_per_pull /= 2;
  }
  const auto bd = smem_breakdown(num_experts, c.block_m, c.block_n, c.block_k, c.num_bytes_per_pull,
                                 c.store_block_m, c.sf_block_m, c.sf_block_n, c.gran_k,
                                 c.num_dispatch_threads / 32, c.num_epilogue_threads / 32);
  c.num_stages = (kSmemCapacity - bd.smem_fixed) / bd.smem_size_per_stage;
  if (c.num_stages < 2) fail("internal: fewer than 2 pipeline stages fit in shared memory");
  c.smem_size = bd.smem_fixed + c.num_stages * bd.smem_size_per_stage;
  return c;
}

// ---------------------------------------------------------------------------------------------
// Tile rule (upstream `get_mega_moe_override`): `rule;rule;...`, rule = `[cond:]key=value,...`,
// cond = `atom&atom...`, atom = `t<=N | t>=N | t<N | t>N | t==N | t=N|M|...`. `t` is
// max(num_tokens, 1). The first rule whose condition holds wins; no match keeps the heuristic.
// Upstream also accepts a `shared` condition variable; this backend has no shared experts, so
// `shared` evaluates to 0. Keys accepted here: block_m, store_block_m, num_stages (alias stages).
// Upstream's remaining keys (block_n, block_k, bytes_per_pull, thread counts, fp8_combine) are
// rejected: they were not part of any measured MOSCA configuration and fp8_combine conflicts with
// the in-kernel all-gather.
// ---------------------------------------------------------------------------------------------
struct Override {
  std::optional<int> block_m, store_block_m, num_stages;
  bool has_any() const { return block_m || store_block_m || num_stages; }
};

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\n\r");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\n\r");
  return s.substr(b, e - b + 1);
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (true) {
    const auto pos = s.find(sep, start);
    parts.push_back(
        trim(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start)));
    if (pos == std::string::npos) break;
    start = pos + 1;
  }
  return parts;
}

int parse_int(const std::string& s, const std::string& what) {
  int value = 0;
  const auto t = trim(s);
  const auto [ptr, ec] = std::from_chars(t.data(), t.data() + t.size(), value);
  if (t.empty() || ec != std::errc() || ptr != t.data() + t.size())
    fail_rule("non-integer value '" + s + "' for " + what);
  return value;
}

bool match_atom(const std::string& atom, int t) {
  for (const std::string op : {"<=", ">=", "==", "<", ">", "="}) {
    const auto pos = atom.find(op);
    if (pos == std::string::npos) continue;
    const auto lhs = trim(atom.substr(0, pos));
    const auto rhs = trim(atom.substr(pos + op.size()));
    int x;
    if (lhs == "t")
      x = t;
    else if (lhs == "shared")
      x = 0;
    else
      fail_rule("unknown condition variable '" + lhs + "' in '" + atom + "' (use t)");
    if (op == "=" || op == "==") {
      for (const auto& v : split(rhs, '|'))
        if (parse_int(v, "condition '" + atom + "'") == x) return true;
      return false;
    }
    const int v = parse_int(rhs, "condition '" + atom + "'");
    if (op == "<=") return x <= v;
    if (op == ">=") return x >= v;
    if (op == "<") return x < v;
    return x > v;
  }
  fail_rule("invalid condition '" + atom + "'");
}

Override parse_rule(const std::string& rules, int num_tokens) {
  Override result;
  bool found = false;
  if (trim(rules).empty()) return result;
  const int t = std::max(num_tokens, 1);
  for (const auto& rule : split(rules, ';')) {
    if (rule.empty()) continue;
    std::string body = rule;
    bool matched = true;
    if (const auto colon = rule.find(':'); colon != std::string::npos) {
      body = rule.substr(colon + 1);
      for (const auto& atom : split(rule.substr(0, colon), '&'))
        matched = match_atom(atom, t) && matched;
    }
    // Parse every body, matched or not, so that a typo is reported on every call.
    Override candidate;
    for (const auto& kv : split(body, ',')) {
      if (kv.empty()) continue;
      const auto eq = kv.find('=');
      if (eq == std::string::npos)
        fail_rule("expected key=value, got '" + kv + "' in rule '" + rule + "'");
      const auto key = trim(kv.substr(0, eq));
      const int value = parse_int(kv.substr(eq + 1), "key " + key);
      if (key == "block_m")
        candidate.block_m = value;
      else if (key == "store_block_m")
        candidate.store_block_m = value;
      else if (key == "num_stages" || key == "stages")
        candidate.num_stages = value;
      else
        fail_rule("unsupported key '" + key + "' in rule '" + rule +
                  "' (this backend accepts block_m, store_block_m and num_stages)");
    }
    if (matched && !found) {
      result = candidate;
      found = true;
    }
  }
  return result;
}

// Launch configuration of one call. Returns, in order:
//   block_m, block_n, block_k, load_block_m, load_block_n, store_block_m, sf_block_m, sf_block_n,
//   gran_k, num_ring_tokens, num_sf_ring_tokens, swizzle_acts_mode, swizzle_weights_mode,
//   num_stages, smem_size, num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
//   num_bytes_per_pull, rule_applied
tvm::ffi::Array<int64_t> Plan(int64_t num_ranks, int64_t num_experts,
                              int64_t num_max_tokens_per_rank, int64_t num_tokens, int64_t num_topk,
                              int64_t hidden, int64_t intermediate_hidden, int64_t num_sms,
                              tvm::ffi::String tile_rule) {
  check_geometry(num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden,
                 intermediate_hidden, num_sms);
  if (num_tokens < 0 || num_tokens > num_max_tokens_per_rank)
    fail("num_tokens must be in [0, num_max_tokens_per_rank], got " + std::to_string(num_tokens));
  const auto [num_ring_tokens, num_sf_ring_tokens] = ring_tokens(
      static_cast<int>(num_ranks), static_cast<int>(num_experts),
      static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_topk),
      static_cast<int>(hidden), static_cast<int>(intermediate_hidden), static_cast<int>(num_sms));
  const Config base = heuristic_config(
      static_cast<int>(num_ranks), static_cast<int>(num_experts), static_cast<int>(num_tokens),
      static_cast<int>(num_topk), static_cast<int>(hidden), num_ring_tokens, num_sf_ring_tokens);
  const Override ov = parse_rule(std::string(tile_rule), static_cast<int>(num_tokens));

  Config c = base;
  if (ov.has_any()) {
    // Upstream `get_fp8_fp4_mega_moe_plan` validation for the keys accepted here.
    const auto check = [](bool cond, const std::string& key, int value, const std::string& reason) {
      if (!cond) fail_rule("invalid " + key + "=" + std::to_string(value) + ": " + reason);
    };
    const int block_m = ov.block_m.value_or(base.block_m);
    check(is_candidate_block_m(block_m), "block_m", block_m,
          "not in {8, 16, 32, 64, 96, 128, 192, 240}");
    check(block_m % 16 == 0 && block_m <= 256, "block_m", block_m,
          "the kernel asserts BLOCK_M % 16 == 0");
    check(num_ring_tokens % block_m == 0, "block_m", block_m,
          "must divide num_ring_tokens=" + std::to_string(num_ring_tokens));
    const int num_epilogue_wgs = base.num_epilogue_threads / 128;
    const int store_block_m =
        ov.store_block_m.value_or(ov.block_m ? get_store_block_m(block_m) : base.store_block_m);
    check(block_m % num_epilogue_wgs == 0, "block_m", block_m,
          "must be a multiple of the epilogue warpgroups");
    check(store_block_m > 0 && store_block_m % 8 == 0 &&
              (block_m / num_epilogue_wgs) % store_block_m == 0,
          "store_block_m", store_block_m,
          "must be a multiple of 8 dividing BLOCK_M / epilogue warpgroups = " +
              std::to_string(block_m / num_epilogue_wgs));
    const int sf_block_m = align_up(block_m, kNumUtccpAlignedElems);
    check(block_m * 2 + sf_block_m / 32 + base.sf_block_n / 32 <= 512, "block_m", block_m,
          "tensor memory columns exceed 512");
    const auto bd =
        smem_breakdown(static_cast<int>(num_experts), block_m, base.block_n, base.block_k,
                       base.num_bytes_per_pull, store_block_m, sf_block_m, base.sf_block_n,
                       base.gran_k, base.num_dispatch_threads / 32, base.num_epilogue_threads / 32);
    const int max_num_stages =
        std::min((kSmemCapacity - bd.smem_fixed) / bd.smem_size_per_stage, 32);
    const int num_stages = ov.num_stages.value_or(max_num_stages);
    check(max_num_stages >= 2, ov.num_stages ? "num_stages" : "block_m",
          ov.num_stages ? num_stages : block_m, "tiles leave room for fewer than 2 stages");
    check(num_stages >= 2 && num_stages <= max_num_stages, "num_stages", num_stages,
          "must be in [2, " + std::to_string(max_num_stages) + "]");
    const int smem_size = bd.smem_fixed + num_stages * bd.smem_size_per_stage;
    const int smem_reusable =
        smem_size - bd.smem_non_reusable_fixed - num_stages * bd.smem_stage_barriers;
    const int num_combine_hidden_bytes = static_cast<int>(hidden) * 2;
    const int num_epilogue_warps = base.num_epilogue_threads / 32;
    const int num_combine_chunks =
        (3 * num_epilogue_warps * num_combine_hidden_bytes <= smem_reusable && hidden <= 32 * 128)
            ? 1
            : 2;
    const int smem_combine_required =
        3 * num_epilogue_warps * num_combine_hidden_bytes / num_combine_chunks;
    check(smem_combine_required <= smem_reusable, ov.num_stages ? "num_stages" : "block_m",
          ov.num_stages ? num_stages : block_m,
          "combine needs " + std::to_string(smem_combine_required) +
              " bytes of reusable shared memory, the pipeline "
              "provides " +
              std::to_string(smem_reusable));
    const int combine_chunk_elems = num_combine_hidden_bytes / num_combine_chunks / 16;
    check(combine_chunk_elems % 32 == 0, ov.num_stages ? "num_stages" : "block_m",
          ov.num_stages ? num_stages : block_m,
          "combine chunk of " + std::to_string(combine_chunk_elems) +
              " 16-byte elements must be a multiple of 32");
    c.block_m = block_m;
    c.load_block_m = block_m / 2;
    c.store_block_m = store_block_m;
    c.sf_block_m = sf_block_m;
    c.num_stages = num_stages;
    c.smem_size = smem_size;
  }
  return tvm::ffi::Array<int64_t>(std::vector<int64_t>{c.block_m,
                                                       c.block_n,
                                                       c.block_k,
                                                       c.load_block_m,
                                                       c.load_block_n,
                                                       c.store_block_m,
                                                       c.sf_block_m,
                                                       c.sf_block_n,
                                                       c.gran_k,
                                                       c.num_ring_tokens,
                                                       c.num_sf_ring_tokens,
                                                       c.swizzle_acts_mode,
                                                       c.swizzle_weights_mode,
                                                       c.num_stages,
                                                       c.smem_size,
                                                       c.num_dispatch_threads,
                                                       c.num_non_epilogue_threads,
                                                       c.num_epilogue_threads,
                                                       c.num_bytes_per_pull,
                                                       ov.has_any() ? 1 : 0});
}

// Token alignment of `num_max_tokens_per_rank` (upstream `get_token_alignment_for_mega_moe`).
int64_t TokenAlignment() { return layout::kLCMCandidateBlockM; }

}  // namespace mosca_megamoe_host

TVM_FFI_DLL_EXPORT_TYPED_FUNC(layout, mosca_megamoe_host::Layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, mosca_megamoe_host::Plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(token_alignment, mosca_megamoe_host::TokenAlignment);
