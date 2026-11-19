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
 */
#pragma once
// Host-side description of one attention call and the gates of the three Nemotron 3.5 Super
// NVFP4-KV kernels (sm_100a):
//
//   decode "mtp"   2, 4, 6 or 8 query tokens per request (fuses 2, 4 or 8 tokens per KV page load)
//   decode "base"  1, 3 or 5 query tokens per request (fused per KV page load), or a call with
//                  cum_seq_lens_q (requests of different query lengths), one token at a time
//   prefill        one request of up to 32768 query tokens
//
// All three serve one geometry only: FP8 E4M3 query and output, NVFP4 K/V with one E4M3 scale per
// 16 values, 32 query heads, 2 KV heads, head dim 128, 128-token pages in HND layout.
//
// Everything here is pure C++: no CUDA calls, no device memory reads, no allocation that outlives
// the call. A gate returns an empty reason when its kernel can serve the call. Which kernel is
// *preferred* for a shape is decided in Python (flashinfer/cake_nvfp4.py).
#include <cstdint>
#include <initializer_list>
#include <string>

namespace flashinfer {
namespace cake_nvfp4 {

constexpr int kNumHeadsQ = 32;
constexpr int kNumHeadsKv = 2;
constexpr int kHeadsPerKv = kNumHeadsQ / kNumHeadsKv;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
// One head of one page: 128 tokens x 64 bytes of packed FP4, and 128 x 8 bytes of block scales.
constexpr int kFp4HeadBytes = kPageSize * kHeadDim / 2;
constexpr int kScaleHeadBytes = kPageSize * kHeadDim / 16;

// One call, as the FFI entry points read it from the tensors. Strides are in bytes.
struct Call {
  void const* q = nullptr;  // [sumQ, 32, 128] E4M3, packed
  void const* k = nullptr;  // [pages, 2, 128, 64] packed FP4
  void const* v = nullptr;
  void const* kSf = nullptr;  // [pages, 2, 128, 8] E4M3 block scales
  void const* vSf = nullptr;
  void* out = nullptr;  // [sumQ, 32, 128] E4M3
  void* workspace = nullptr;
  int64_t workspaceBytes = 0;
  int32_t const* pageTable = nullptr;          // [batch, maxPagesPerSeq]
  int32_t const* seqLensKv = nullptr;          // [batch]
  int32_t const* cumSeqLensQ = nullptr;        // [batch + 1], starts at 0; optional for decode
  float const* scaleSoftmaxLog2Ptr = nullptr;  // device scalars; override the values below
  float const* outputScalePtr = nullptr;
  float scaleSoftmaxLog2 = 0.f;
  float outputScale = 1.f;
  int batch = 0;
  int maxQ = 0;       // query tokens of the longest request
  int64_t sumQ = 0;   // query rows
  int64_t maxKv = 0;  // KV tokens of the longest request (prefill: cached prefix + query chunk)
  int64_t maxPagesPerSeq = 0;
  int64_t poolPages = 0;
  int64_t kStrideHead = 0, kStridePage = 0;
  int64_t vStrideHead = 0, vStridePage = 0;
  int64_t kSfStrideHead = 0, kSfStridePage = 0;
  int64_t vSfStrideHead = 0, vSfStridePage = 0;
  bool enablePdl = false;
};

// Device facts the gates need.
struct Device {
  int major = 0;
  int minor = 0;
  int smCount = 0;
};

inline bool aligned(void const* ptr, uintptr_t alignment) {
  return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

inline std::string deviceReason(Device const& dev) {
  if (dev.major == 10 && dev.minor == 0) return "";
  return "device sm_" + std::to_string(dev.major) + std::to_string(dev.minor) + " (needs sm_100)";
}

// Checks shared by both decode kernels and the prefill kernel: page table, pointers and strides.
inline std::string layoutReason(Call const& c) {
  if (c.sumQ <= 0 || c.maxPagesPerSeq <= 0 || c.poolPages <= 0)
    return "empty query, page table or page pool";
  if (c.maxKv > c.maxPagesPerSeq * kPageSize)
    return "max KV length " + std::to_string(c.maxKv) + " exceeds the page table";
  if (!c.out || !c.pageTable || !c.seqLensKv) return "missing output, page table or KV lengths";
  // The decode combine kernels store the output as 4-byte words.
  if (!aligned(c.out, 4)) return "output pointer not 4-byte aligned";
  // TMA descriptors and vector loads need 16-byte aligned bases and strides.
  for (void const* ptr : {c.q, c.k, c.v, c.kSf, c.vSf})
    if (!aligned(ptr, 16)) return "Q/KV pointer null or not 16-byte aligned";
  for (int64_t stride : {c.kStrideHead, c.vStrideHead, c.kStridePage, c.vStridePage})
    if (stride <= 0 || stride % 16) return "KV stride not 16-byte aligned";
  for (int64_t stride : {c.kSfStrideHead, c.vSfStrideHead, c.kSfStridePage, c.vSfStridePage})
    if (stride <= 0 || stride % 16) return "KV scale stride not 16-byte aligned";
  if (c.kStrideHead < kFp4HeadBytes || c.vStrideHead < kFp4HeadBytes ||
      c.kSfStrideHead < kScaleHeadBytes || c.vSfStrideHead < kScaleHeadBytes)
    return "KV head stride too small";
  return "";
}

// ---------------------------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------------------------

constexpr int kDecodeMaxBatch = 256;
constexpr int kDecodeMaxQ = 8;
// Longest request of a ragged call the "base" kernel accepts. The routing rules send it ragged
// calls of at most 5 query tokens.
constexpr int kDecodeBaseMaxRaggedQ = 6;
constexpr int kDecodeBaseMaxCtas = 512;  // the combine kernel keeps at most 512 piece weights

// Launch geometry and workspace layout of a decode call. The workspace holds, in floats:
// piece O | piece (max, sum) | unit piece ranges | per-CTA KV segment lists.
struct DecodePlan {
  std::string reason;   // empty when the kernel serves the call
  bool ragged = false;  // the call passes cum_seq_lens_q, and the kernel reads it
  int g = 0;            // query tokens fused per virtual request
  int groups = 0;       // virtual requests per request = maxQ / g
  int ctas = 0;         // one CTA per SM, rounded down to a multiple of the 2 KV heads
  int64_t virt = 0;     // virtual requests
  int64_t stats = 0;    // workspace offsets, in floats
  int64_t ranges = 0;
  int64_t segments = 0;
  int64_t requiredBytes = 0;
};

// Tokens the "mtp" kernel fuses for q query tokens per request; 0 when it does not serve q.
// q = 6 runs the 2-token variant over three groups.
inline int decodeMtpFusedTokens(int q) {
  switch (q) {
    case 2:
      return 2;
    case 4:
      return 4;
    case 6:
      return 2;
    case 8:
      return 8;
    default:
      return 0;
  }
}

// Tokens the "base" kernel fuses for q query tokens per request; 0 when it does not serve q.
// Ragged calls fuse one token, so that no fused tile crosses a request boundary.
inline int decodeBaseFusedTokens(bool ragged, int q) {
  if (ragged) return 1;
  return q == 1 || q == 3 || q == 5 ? q : 0;
}

inline void fillDecodePlan(DecodePlan& p, Call const& c, Device const& dev) {
  p.ctas = dev.smCount & ~1;
  p.groups = c.maxQ / p.g;
  p.virt = int64_t(c.batch) * p.groups;
  int64_t const slots = (p.virt + p.ctas / 2) * 2;
  p.stats = slots * kHeadsPerKv * p.g * kHeadDim;
  p.ranges = p.stats + slots * kHeadsPerKv * p.g * 2;
  p.segments = (p.ranges + p.virt * 4 + 3) / 4 * 4;
  p.requiredBytes = (p.segments + int64_t(p.ctas) * (p.virt + 1) * 4) * int64_t(sizeof(float));
}

// Query rows equal to batch * max q_len prove that every request has max q_len tokens, so
// request i owns rows i * max q_len onwards. They prove nothing when the call passes
// cum_seq_lens_q: a CUDA graph captured with uniform offsets replays whatever they hold then.
inline bool uniformQuery(Call const& c) {
  return !c.cumSeqLensQ && int64_t(c.batch) * c.maxQ == c.sumQ;
}

inline std::string decodeCommonReason(Call const& c, Device const& dev) {
  std::string reason = deviceReason(dev);
  if (!reason.empty()) return reason;
  if (c.batch <= 0 || c.batch > kDecodeMaxBatch)
    return "batch " + std::to_string(c.batch) + " outside 1.." + std::to_string(kDecodeMaxBatch);
  return layoutReason(c);
}

inline void checkDecodeWorkspace(DecodePlan& p, Call const& c) {
  if (p.ctas < 2)
    p.reason = "fewer than 2 SMs";
  else if (!aligned(c.workspace, 16))
    p.reason = "workspace null or not 16-byte aligned";
  else if (p.requiredBytes > c.workspaceBytes)
    p.reason = "workspace " + std::to_string(c.workspaceBytes) + " bytes < required " +
               std::to_string(p.requiredBytes);
}

inline DecodePlan planDecodeMtp(Call const& c, Device const& dev) {
  DecodePlan p;
  p.reason = decodeCommonReason(c, dev);
  if (!p.reason.empty()) return p;
  p.g = decodeMtpFusedTokens(c.maxQ);
  if (p.g == 0) {
    p.reason = "q_len " + std::to_string(c.maxQ) + " unsupported (serves 2, 4, 6, 8)";
    return p;
  }
  if (!uniformQuery(c)) {
    p.reason = "ragged q_len";
    return p;
  }
  fillDecodePlan(p, c, dev);
  checkDecodeWorkspace(p, c);
  return p;
}

inline DecodePlan planDecodeBase(Call const& c, Device const& dev) {
  DecodePlan p;
  p.reason = decodeCommonReason(c, dev);
  if (!p.reason.empty()) return p;
  p.ragged = !uniformQuery(c);
  if (c.maxQ < 1 || c.maxQ > kDecodeMaxQ) {
    p.reason = "q_len " + std::to_string(c.maxQ) + " outside 1.." + std::to_string(kDecodeMaxQ);
    return p;
  }
  if (p.ragged && c.maxQ > kDecodeBaseMaxRaggedQ) {
    p.reason =
        "ragged q_len " + std::to_string(c.maxQ) + " > " + std::to_string(kDecodeBaseMaxRaggedQ);
    return p;
  }
  if (p.ragged && !c.cumSeqLensQ) {
    p.reason = "query rows != batch * q_len without cum_seq_lens_q";
    return p;
  }
  p.g = decodeBaseFusedTokens(p.ragged, c.maxQ);
  if (p.g == 0) {
    p.reason = "q_len " + std::to_string(c.maxQ) + " unsupported (serves 1, 3, 5)";
    return p;
  }
  fillDecodePlan(p, c, dev);
  if (p.ctas > kDecodeBaseMaxCtas)
    p.reason = "CTA count " + std::to_string(p.ctas) + " > " + std::to_string(kDecodeBaseMaxCtas);
  else
    checkDecodeWorkspace(p, c);
  return p;
}

// ---------------------------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------------------------

constexpr int kPrefillQTile = 256;
constexpr int kPrefillMaxQ = 32768;
constexpr int kPrefillMaxSms = 256;
// The kernel first converts every visible NVFP4 page to FP8 in the workspace. One FP8 page of
// one tensor (K or V) is 2 heads x 128 tokens x 128 bytes.
constexpr int64_t kPrefillFp8PageBytes = int64_t(kNumHeadsKv) * kPageSize * kHeadDim;
// Upper bound of the split-schedule helper buffers for up to 256 SMs.
// flashinfer/cake_nvfp4.py repeats this formula to size a workspace before this module is
// loaded: change both together.
constexpr int64_t kPrefillSplitReserveBytes = int64_t(72) << 20;
constexpr uintptr_t kPrefillWorkspaceAlign = 128;

// Pages of the compact FP8 cache: every page the longest visible KV range can touch.
inline int64_t prefillCachePages(int64_t maxKv) { return (maxKv + kPageSize - 1) / kPageSize; }

inline int64_t prefillWorkspaceBytes(int64_t maxKv) {
  return 2 * prefillCachePages(maxKv) * kPrefillFp8PageBytes + kPrefillSplitReserveBytes;
}

struct PrefillSchedule {
  int nctas = 0;
  int qf = 1;
  int nrem = 0;
  int pchunks = 1;
  float xFrac = 0.f;
  int tileBase = 0;       // tail-only: first split tile (= tiles in complete waves)
  bool tailOnly = false;  // complete waves on the plain kernel, sparse tail split
  // Tiles run by the plain kernel (none when the whole grid uses split scheduling).
  int plainTiles(int tiles) const { return tailOnly ? tileBase : (nrem == 0 ? tiles : 0); }
};

// Schedule of one request. A tile is 256 query rows of one head. With one CTA per SM:
//   * fewer than 6 complete waves and a tail: whole waves plus a helper-split tail;
//   * 6 to 10 complete waves and a tail filling at most half the SMs: complete waves on the plain
//     kernel, only the sparse tail split;
//   * otherwise the plain kernel runs every tile.
inline PrefillSchedule makePrefillSchedule(int maxQ, int sms) {
  int const tiles = (maxQ + kPrefillQTile - 1) / kPrefillQTile * kNumHeadsQ;
  PrefillSchedule s;
  s.nctas = tiles;
  int const fullWaves = sms > 0 ? tiles / sms : 0;
  int const tailTiles = sms > 0 ? tiles % sms : 0;
  if (sms > 0 && tailTiles != 0 && fullWaves < 6) {
    s.nctas = sms;
    s.qf = fullWaves;
    s.nrem = tailTiles;
  } else if (sms > 0 && tailTiles != 0 && fullWaves < 11 && 2 * tailTiles <= sms) {
    s.nctas = sms;
    s.qf = 0;
    s.nrem = tailTiles;
    s.tileBase = fullWaves * sms;
    s.tailOnly = true;
  }
  if (s.nrem > 0) {
    int const helpers = sms - s.nrem;
    int const hmax = (s.nrem + helpers - 1) / helpers;
    s.xFrac = 1.f / (float(hmax) + 1.f);
    double best = 1e30;
    for (int p = 1; p <= 16; ++p) {
      double const waves = (s.nrem * p + sms - 1) / sms;
      double const time = waves / p + 0.002 * p;
      if (time < best) {
        best = time;
        s.pchunks = p;
      }
    }
  }
  return s;
}

inline std::string prefillReason(Call const& c, Device const& dev) {
  std::string reason = deviceReason(dev);
  if (!reason.empty()) return reason;
  if (dev.smCount < 1 || dev.smCount > kPrefillMaxSms)
    return "SM count " + std::to_string(dev.smCount) + " outside 1.." +
           std::to_string(kPrefillMaxSms);
  // One request per call bounds the FP8 copy of its KV; the launcher relies on it.
  if (c.batch != 1) return "batch " + std::to_string(c.batch) + " (serves one request)";
  if (c.maxQ < 1 || c.maxQ > kPrefillMaxQ)
    return "max q_len " + std::to_string(c.maxQ) + " outside 1.." + std::to_string(kPrefillMaxQ);
  if (c.sumQ != c.maxQ) return "query rows != max q_len";
  if (c.maxKv < c.maxQ) return "max KV length shorter than max q_len";
  if (!c.cumSeqLensQ) return "missing cum_seq_lens_q";
  reason = layoutReason(c);
  if (!reason.empty()) return reason;
  // The conversion kernel reads K and V, and their scales, with one shared set of strides.
  if (c.kStrideHead != c.vStrideHead || c.kStridePage != c.vStridePage ||
      c.kSfStrideHead != c.vSfStrideHead || c.kSfStridePage != c.vSfStridePage)
    return "K and V strides differ";
  if (!aligned(c.out, 16)) return "output pointer not 16-byte aligned";
  if (!aligned(c.workspace, kPrefillWorkspaceAlign))
    return "workspace null or not 128-byte aligned";
  int64_t const need = prefillWorkspaceBytes(c.maxKv);
  if (need > c.workspaceBytes)
    return "workspace " + std::to_string(c.workspaceBytes) + " bytes < required " +
           std::to_string(need);
  return "";
}

// Launchers, one per kernel translation unit (csrc/cake_nvfp4_*.cu). Each expects a call
// its gate accepted and returns once every kernel is enqueued on `stream`.
void launchDecodeMtp(Call const& c, DecodePlan const& plan, void* stream);
void launchDecodeBase(Call const& c, DecodePlan const& plan, void* stream);
void launchPrefill(Call const& c, Device const& dev, void* stream);

}  // namespace cake_nvfp4
}  // namespace flashinfer
