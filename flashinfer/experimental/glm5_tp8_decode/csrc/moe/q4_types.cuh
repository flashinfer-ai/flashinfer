/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define MOE_HD __host__ __device__
#else
#define MOE_HD
#endif

namespace flashinfer::glm5_tp8::moe_q4 {

constexpr int kGrid = 148;
constexpr int kBlock = 320;
constexpr int kConsumerThreads = 256;
constexpr int kConsumerWarps = 8;
constexpr int kAuxBase = 288;
constexpr int kPrefetchWarps = 1;
constexpr int kRoutedCtas = 128;
constexpr int kSharedBase = 128;
constexpr int kSharedCtas = 16;
constexpr int kSharedNormCtas = 4;
constexpr int kWorkCtas = kRoutedCtas + kSharedCtas;
constexpr int kSentinelHelperCtas = kGrid - kWorkCtas;
constexpr int kWorld = 8;

constexpr int kSeqLen = 4;
static_assert(kSentinelHelperCtas == kSeqLen, "one sentinel helper CTA per token");
constexpr int kDim = 6144;
constexpr int kDimWords = kDim / 2;
constexpr int kWordsPerThread = kDimWords / kConsumerThreads;
constexpr int kTopk = 8;
constexpr int kExpertSlots = 1 + kTopk;
constexpr int kInterPerRank = 256;
constexpr int kRowsPerGroup = 16;
constexpr int kRowGroupsPerCta = 4;
constexpr int kGateUpPages = 6;
constexpr int kPageK = 1024;
constexpr int kDownRowsPerCta = 48;
constexpr int kDownSlices = 128;
constexpr int kDownBlocksPerCta = 4;
constexpr int kDownPages = 5;
constexpr int kDownColsPerCta = kDownBlocksPerCta * kDownRowsPerCta;
constexpr int kCtasPerToken = 32;
constexpr int kExpertsPerRouterCta = 8;
constexpr int kRouterPages = 4;
constexpr float kRouteScale = 2.5f;
constexpr float kRmsEps = 1e-5f;
constexpr float kRmsDenominator = 6144.0f;

constexpr uint32_t kGammaPageBytes = 0x6000;
constexpr uint32_t kRouterPageBytes = 0x6000;
constexpr uint32_t kRouterRowBytes = 0x3000;
constexpr uint32_t kRouterCtaStride = 0x18000;
constexpr uint32_t kGateUpPageStride = 0x8080;
constexpr uint32_t kGateUpPageCopyBytes = 0x8040;
constexpr uint32_t kGateUpRowGroupStride = 0x30300;
constexpr uint32_t kGateUpExpertStride = 0x303000;
constexpr uint32_t kGateUpUpHalf = 0x4000;
constexpr uint32_t kGateUpScaleGate = 0x8000;
constexpr uint32_t kGateUpScaleUp = 0x8020;
constexpr uint32_t kGateUpWarpKBytes = 0x800;
constexpr uint32_t kGateUpKStepBytes = 0x100;
constexpr uint32_t kDownExpertStride = 0x180000;
constexpr uint32_t kDownSliceBytes = 0x3000;
constexpr uint32_t kDownScaleExpertStride = 0x2000;
constexpr uint32_t kDownScaleSliceBytes = 0x40;
constexpr uint32_t kDownTileBytes = 0x1000;
constexpr uint32_t kDownWarpKBytes = 0x200;
constexpr int kDownHalfBlocks = 2;
constexpr int kDownHalfGroups = kDownBlocksPerCta / kDownHalfBlocks;
constexpr uint32_t kDownHalfMatBytes = kDownHalfBlocks * kDownSliceBytes;
constexpr uint32_t kDownHalfScaleBytes = kDownHalfBlocks * kDownScaleSliceBytes;
constexpr uint32_t kDownHalfStageBytes = kDownHalfMatBytes + kDownHalfScaleBytes;
constexpr int kRoutedPersistentDownStages = kDownHalfGroups * kTopk;
constexpr int kSharedPersistentDownStages = 4;
static_assert(kDownHalfGroups == 2 && kDownHalfMatBytes == 0x6000 && kDownHalfScaleBytes == 0x80 &&
                  kDownHalfStageBytes == 0x6080,
              "two-slice expert-major down stage geometry");
constexpr int kRoutedItems =
    1 + kRouterPages + kRowGroupsPerCta * kGateUpPages + kRoutedPersistentDownStages;
constexpr int kSharedItems = 1 + kGateUpPages + kSharedPersistentDownStages;
constexpr uint32_t kRoutedStreamBytes = kGammaPageBytes + kRouterPages * kRouterPageBytes +
                                        kRowGroupsPerCta * kGateUpPages * kGateUpPageCopyBytes +
                                        kRoutedPersistentDownStages * kDownHalfStageBytes;
constexpr uint32_t kSharedStreamBytes = kGammaPageBytes + kGateUpPages * kGateUpPageCopyBytes +
                                        kSharedPersistentDownStages * kDownHalfStageBytes;
static_assert(kRoutedItems == 45 && kSharedItems == 11, "persistent item sequences");
static_assert(kGammaPageBytes % 64 == 0 && kRouterPageBytes % 64 == 0 &&
                  kGateUpPageCopyBytes % 64 == 0,
              "single-segment pages split into 4 x 16 B-multiples (0)");
static_assert(kRoutedStreamBytes == 1306112u && kSharedStreamBytes == 320384u, "bytes per CTA");

constexpr uint32_t kScoresSlotBytes = 128;
constexpr uint32_t kHiddenMidSlotBytes = kInterPerRank * 2;
constexpr uint32_t kHiddenMidTokenBytes = kExpertSlots * kHiddenMidSlotBytes;
constexpr uint32_t kXRowBytes = kDim * 2;

constexpr uint32_t kInboxTokenStride = 0x38000;
constexpr uint32_t kInboxSlotStride = 0x8000;
constexpr uint32_t kInboxParityStride = 0x4000;
constexpr uint32_t kInboxChunkStride = 0x80;
constexpr uint32_t kInboxLineStride = 0x10;

static_assert(kInboxTokenStride == 7 * kInboxSlotStride &&
                  kInboxSlotStride == 2 * kInboxParityStride &&
                  kInboxParityStride == kDownSlices * kInboxChunkStride,
              "inbox geometry");

constexpr uint32_t kSyncW1 = 0x38;
constexpr uint32_t kSyncInit = 0x3c;
constexpr uint32_t kSyncSharedMid = 0x40;
constexpr uint32_t kSyncSharedDown = 0x44;
constexpr uint32_t kSyncSharedDownFlags = kCtasPerToken;
constexpr uint32_t kSyncProgress = 0x100;
constexpr uint32_t kSyncProgressBankStride = 0x10;
constexpr uint32_t kSyncBlockBytes = 0xd0;
constexpr uint32_t kSyncWordsBytes = kSyncProgress + 2 * kSyncProgressBankStride;
constexpr uint32_t kW1Arrivers = kSharedNormCtas;
static_assert(kSyncSharedDown + kSyncSharedDownFlags * sizeof(uint32_t) <= kSyncBlockBytes,
              "shared Down flags exceed sync block");
static_assert(kSyncProgress % 128 == 0 && kSyncProgress >= (kSyncBlockBytes + 127) / 128 * 128 &&
                  2 * kSyncProgressBankStride <= 128 &&
                  kSeqLen * sizeof(uint32_t) == kSyncProgressBankStride,
              "two progress banks in a line of their own");
MOE_HD constexpr uint32_t w1_target(uint32_t epoch) { return kW1Arrivers * epoch; }

namespace state {
constexpr size_t kWords = 0x000;
constexpr size_t kLaunchCtr = 0x200;
constexpr size_t kBytes = 0x500;
static_assert(kWords + kSyncWordsBytes <= kLaunchCtr &&
                  kLaunchCtr + size_t(kGrid) * sizeof(uint32_t) <= kBytes &&
                  kLaunchCtr % 256 == 0 && kBytes % 256 == 0,
              "per-device state layout");
}  // namespace state

struct CtxRow0 {
  const uint16_t* x;
  const float* gamma;
  uint64_t pad[14];
};
struct CtxRow1 {
  const uint16_t* router_w;
  float* scores_out;
  uint64_t pad[14];
};
struct CtxRow2 {
  const uint8_t* gate_up_w;
  const float* bias;
  const float* scores_in;
  float* sel_probs;
  int32_t* sel_indices;
  uint16_t* hidden_mid_out;
  uint64_t pad[10];
};
struct CtxRow3 {
  const uint8_t* gate_up_w_shared;
  uint16_t* hidden_mid_shared_out;
  uint16_t* x_norm_buf;
  const uint16_t* x_shared;
  uint64_t pad[12];
};
struct CtxRow4 {
  const uint8_t* down_w;
  const uint16_t* hidden_mid_in;
  uint16_t* hidden_out;
  const uint16_t* x_residual;
  uint64_t epoch;
  const float* down_scale;
  uint64_t pad[10];
};
struct CtxTable {
  CtxRow0 r0;
  CtxRow1 r1;
  CtxRow2 r2;
  CtxRow3 r3;
  CtxRow4 r4;
};
constexpr int kCtxRows = 5;
constexpr int kCtxRowBytes = 128;
static_assert(sizeof(CtxRow0) == kCtxRowBytes && sizeof(CtxRow1) == kCtxRowBytes &&
                  sizeof(CtxRow2) == kCtxRowBytes && sizeof(CtxRow3) == kCtxRowBytes &&
                  sizeof(CtxRow4) == kCtxRowBytes && sizeof(CtxTable) == kCtxRows * kCtxRowBytes,
              "ctx rows are 128 B");
static_assert(offsetof(CtxRow0, gamma) == 0x08, "row 0");
static_assert(offsetof(CtxRow1, scores_out) == 0x08, "row 1");
static_assert(offsetof(CtxRow2, bias) == 0x08 && offsetof(CtxRow2, scores_in) == 0x10 &&
                  offsetof(CtxRow2, sel_probs) == 0x18 && offsetof(CtxRow2, sel_indices) == 0x20 &&
                  offsetof(CtxRow2, hidden_mid_out) == 0x28,
              "row 2");
static_assert(offsetof(CtxRow3, hidden_mid_shared_out) == 0x08 &&
                  offsetof(CtxRow3, x_norm_buf) == 0x10 && offsetof(CtxRow3, x_shared) == 0x18,
              "row 3");
static_assert(offsetof(CtxRow4, hidden_mid_in) == 0x08 && offsetof(CtxRow4, hidden_out) == 0x10 &&
                  offsetof(CtxRow4, x_residual) == 0x18 && offsetof(CtxRow4, epoch) == 0x20 &&
                  offsetof(CtxRow4, down_scale) == 0x28,
              "row 4");
static_assert(offsetof(CtxTable, r1) == 0x80 && offsetof(CtxTable, r2) == 0x100 &&
                  offsetof(CtxTable, r3) == 0x180 && offsetof(CtxTable, r4) == 0x200,
              "ctx table rows");

struct DirectCtx {
  const uint16_t* r0_x;
  const float* r0_gamma;
  const uint16_t* r1_router_w;
  float* r1_scores_out;
  const uint8_t* r2_gate_up_w;
  const float* r2_bias;
  const float* r2_scores_in;
  float* r2_sel_probs;
  int32_t* r2_sel_indices;
  uint16_t* r2_hidden_mid_out;
  const uint8_t* r3_gate_up_w_shared;
  uint16_t* r3_hidden_mid_shared_out;
  uint16_t* r3_x_norm_buf;
  const uint16_t* r3_x_shared;
  const uint8_t* r4_down_w;
  const uint16_t* r4_hidden_mid_in;
  uint16_t* r4_hidden_out;
  const uint16_t* r4_x_residual;
  uint64_t r4_epoch;
  const float* r4_down_scale;
};
struct ContextPayload {
  DirectCtx direct;
};
static_assert(sizeof(DirectCtx) == 0xa0, "twenty context fields");

struct MoeArgs {
  int32_t rank;
  uint32_t reserved;
  uint32_t* sync_words;
  uint8_t* peers[kWorld];
  ContextPayload context;
  float* shared_down_scratch;
  uint32_t* launch_ctr;
};
static_assert(sizeof(MoeArgs) == 0x100 && offsetof(MoeArgs, context) == 0x50,
              "MoE parameter layout");

namespace smem {
constexpr uint32_t kRouterRedScratch = 0x0300;
constexpr uint32_t kCtx = 0x0400;
constexpr uint32_t kCtxRowStride = 0x01c0;
constexpr uint32_t kSelAllBar = 0x08c0;
constexpr uint32_t kSharedReadyBar = 0x08c8;
constexpr uint32_t kMidReadyBar = 0x1000;
constexpr uint32_t kUnitBar = 0x1040;
constexpr uint32_t kFoldsDoneBar = 0x1060;
constexpr uint32_t kProgOwn = 0x1100;
constexpr uint32_t kCapFlag = 0x1104;
constexpr uint32_t kPfDone = 0x1108;
constexpr uint32_t kSharedEpoch = 0x110c;
constexpr uint32_t kRedScratch = 0x3c00;
constexpr uint32_t kFullBar = 0x3d00;
constexpr uint32_t kEmptyBar = 0x3d80;

constexpr uint32_t kDownFullBar = 0x3e00;
constexpr uint32_t kDownEmptyBar = 0x3e40;
constexpr uint32_t kDownRingSlots = 5;
constexpr uint32_t kRing = 0x4000;
constexpr uint32_t kRingSlots = 4;
constexpr uint32_t kRingLog2 = kRingSlots == 4 ? 2 : (kRingSlots == 2 ? 1 : 0);

constexpr uint32_t kRingSlotBytes = 0x8080;
constexpr uint32_t kRingEnd = kRing + kRingSlots * kRingSlotBytes;
static_assert(kRingSlots == 1 || kRingSlots == 2 || kRingSlots == 4,
              "kRingSlots must be 1, 2 or 4 (deeper rings need a smem relayout)");
static_assert(kPrefetchWarps == 1, "the ordered prefetch path uses exactly one producer warp");
static_assert((1u << kRingLog2) == kRingSlots, "ring log2");

constexpr uint32_t kWorkShift = 0x3e00;
constexpr uint32_t kRxRow = 0x28000 - kWorkShift;

constexpr uint32_t kDownActCodes = kRxRow;
constexpr uint32_t kDownActSlotBytes = kInterPerRank * 2;
constexpr uint32_t kDownActScales = kDownActCodes + kExpertSlots * kDownActSlotBytes;
static_assert(kDownActScales + kExpertSlots * 2 * sizeof(float) <= kRxRow + kXRowBytes,
              "Down quantization must fit the x-norm buffer, which is dead in this phase");
constexpr uint32_t kRSelIdx = 0x2b000 - kWorkShift;
constexpr uint32_t kRSelProb = 0x2b080 - kWorkShift;
constexpr uint32_t kRHiddenMid = 0x2b100 - kWorkShift;

constexpr uint32_t kRSharedTerm = kRHiddenMid;
constexpr uint32_t kRSharedTermReady = kRHiddenMid + 0x300;
constexpr uint32_t kRScores = 0x2b500 - kWorkShift;
constexpr uint32_t kRBias = 0x2b900 - kWorkShift;
constexpr uint32_t kRGateUp16 = 0x2bd00 - kWorkShift;

constexpr uint32_t kRQScales = kRGateUp16;
constexpr uint32_t kRProbCopy = 0x2c900 - kWorkShift;
constexpr uint32_t kRDownOut = 0x2c980 - kWorkShift;
constexpr uint32_t kRResidual = 0x2cb00 - kWorkShift;
constexpr uint32_t kRDownPartial = 0x2cc80 - kWorkShift;

constexpr uint32_t kRDownPartialWarpStride = 0x300;
constexpr uint32_t kRDownPartialBlockStride = 0xc0;
constexpr uint32_t kRCollectiveRx = 0x2e480 - kWorkShift;
constexpr uint32_t kRRxSubStride = 0x2a0;
constexpr uint32_t kRRxPeerStride = 0x60;
constexpr uint32_t kRCollectiveRxEnd = kRCollectiveRx + kSeqLen * kRRxSubStride;

constexpr uint32_t kRDownActCodesP = kRCollectiveRxEnd;
constexpr uint32_t kRDownActScalesP = kRDownActCodesP + kExpertSlots * kDownActSlotBytes;

constexpr uint32_t kSxRow = 0x2b100 - kWorkShift;
constexpr uint32_t kSxNorm = 0x2e100 - kWorkShift;
constexpr uint32_t kSxNormBufStride = 0x2000;
constexpr uint32_t kSxNormTokStride = 0x800;
constexpr uint32_t kSPartial = 0x32100 - kWorkShift;
constexpr uint32_t kSPartialEnd = kSPartial + 2 * kSeqLen * kRowsPerGroup * kConsumerWarps * 4;
constexpr uint32_t kSelMineBar = 0x33140 - kWorkShift;
constexpr uint32_t kDynamicBytes = 0x2f400;
static_assert(kDownRingSlots * kDownHalfStageBytes <= kRingSlots * kRingSlotBytes &&
                  kDownHalfStageBytes % 16 == 0,
              "five Down half-stage slots fit the ring memory");
static_assert(kEmptyBar + 8 * kRingSlots <= kDownFullBar &&
                  kDownEmptyBar + 8 * kDownRingSlots <= kRing,
              "Down ring barriers");
static_assert(kRDownActScalesP + kExpertSlots * 2 * sizeof(float) <= kSelMineBar,
              "routed Down activation area below sel_mine");
static_assert(kCtx + kCtxRows * kCtxRowStride <= kRedScratch &&
                  kSelAllBar >= kCtx + 2 * kCtxRowStride + 0x80 &&
                  kSelAllBar + 8 <= kCtx + 3 * kCtxRowStride,
              "sel_all sits in the host-unfilled tail of ctx row 2");
static_assert(kRedScratch + 0x100 <= kFullBar && kEmptyBar + 8 * kRingSlots <= kRing,
              "static region layout");
static_assert(kDownHalfStageBytes <= kRingSlotBytes,
              "expert-major half stage must fit one ring slot");
static_assert(kRingEnd <= kRxRow && kRingEnd <= kSxRow,
              "ring must end below the compacted work areas");
static_assert(kRxRow + kXRowBytes <= kRSelIdx &&
                  kRHiddenMid + kExpertSlots * kHiddenMidSlotBytes <= kRProbCopy &&
                  kRDownPartial + kConsumerWarps * kRDownPartialWarpStride <= kRCollectiveRx &&
                  kRCollectiveRxEnd <= kSPartial,
              "routed work area");
static_assert(kSxRow + kXRowBytes <= kSxNorm && kSxNorm + 2 * kSxNormBufStride <= kSPartial &&
                  kSPartialEnd <= kSelMineBar && kSelMineBar + 8 <= kDynamicBytes,
              "shared work area");
static_assert(kRing % 1024 == 0 && kRingSlotBytes % 16 == 0,
              "ring alignment (cp.async.bulk needs 16 B)");
static_assert(kDynamicBytes <= 232448, "exceeds sm_100 max dynamic smem per block (227 KB)");
}  // namespace smem

}  // namespace flashinfer::glm5_tp8::moe_q4
