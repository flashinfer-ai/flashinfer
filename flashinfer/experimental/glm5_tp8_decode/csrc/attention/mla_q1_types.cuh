/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "q1_launch.cuh"

namespace flashinfer::glm5_tp8::mla_q1 {

constexpr int kWorld = 8;
constexpr int kGrid = 148;  // launch grid; CTA roles are by blockIdx.x, so the grid MUST be 148
constexpr int kDefaultBlock = 384;  // nominal block size; the launcher uses kFastBlock (mla_q1.cu)
constexpr int kLaunchDynSmem =
    214016;  // dynamic smem requested at launch (opted in by initialize_mla_q1); the kernel needs
             // only FastSmem::kBytes of it
constexpr int kNRows = 8;
constexpr int kRowBytes = 128;  // op-table row stride in global memory (16 x u64)

// ---------------------------------------------------------------------------------------------------------------
// 1. Compact operator parameters; mutable state is owned by the caller.
using LaunchArgs = attention_q1::Q1Args;

// ---------------------------------------------------------------------------------------------------------------
// 2. Attention-rank descriptor table: 8 rows x 128 B, filled by _descriptors.prepare_q1_attention.
// +0xNN is a field's byte offset in its row; alias_* fields repeat a pointer of an earlier row.
// reserved_N and pad words are left zero and never read.
// ---------------------------------------------------------------------------------------------------------------
struct MlaRow0 {  // P1: x-RMSNorm + qkv_a_proj GEMV (CTAs 0..81), k_pe raw -> pe_cache[cur_pos + t]
  const uint16_t* x;
  const float* x_gamma;       // +0x08 f32 [6144]
  const uint8_t* qkv_a_proj;  // +0x10 fp8 pages [82 CTA][6][0x8080] (16,184,832 B)
  const int32_t* cur_pos;
  uint16_t* post_q;    // +0x20 bf16 [S][2048]   q_a rows of the qkv_a_proj output
  uint16_t* post_kv;   // +0x28 bf16 [S][512]    c_kv rows of the qkv_a_proj output
  uint16_t* pe_cache;  // +0x30 bf16 [rows][64]  row cur_pos + t receives the raw k_pe (RoPE'd in
                       // place by the tail CTA)
  uint64_t pad[9];     // +0x38..+0x7f
};
struct MlaRow1 {                 // P2: q-RMSNorm + q_b_proj (CTAs 0..79)
  const uint16_t* alias_post_q;  // +0x00 = row0.post_q (Q in)
  const uint8_t* q_b_proj;       // +0x08 fp8 pages [80 CTA][2][0x8080] (5,263,360 B)
  uint64_t reserved_0;           // +0x10
  const float* q_gamma;          // +0x18 f32 [2048]
  uint16_t* post_qnd;
  uint16_t* post_qpe;
  uint64_t pad[10];  // +0x30..+0x7f
};
struct MlaRow2 {  // tail RoPE (CTA 132 + t): Q_PE in place, pe_cache[cur_pos + t] in place
  uint16_t* alias_post_qpe;  // +0x00 = row1.post_qpe
  uint16_t* alias_pe_cache;  // +0x08 = row0.pe_cache
  const float* rope;  // +0x10 f32 [rows][64] RoPE table of interleaved (cos, sin) pairs; query t
                      // uses row cur_pos + t
  const int32_t* alias_cur_pos;  // +0x18 = row0.cur_pos
  uint64_t heads_rope;           // +0x20 = 10 (not read by this kernel)
  uint64_t pad[11];              // +0x28..+0x7f
};
struct MlaRow3 {  // tail kv-norm (CTA 128 + t): kv_cache[cur_pos + t] = bf16(kv * (gamma *
                  // rsqrt(fma(sumsq, 2^-9, 1e-5))))
  const uint16_t* alias_post_kv;  // +0x00 = row0.post_kv (c_kv in)
  const float* kv_gamma;          // +0x08 f32 [512]
  uint16_t* kv_cache;             // +0x10 bf16 [rows][512]
  const int32_t* alias_cur_pos;   // +0x18 = row0.cur_pos
  uint64_t pad[12];               // +0x20..+0x7f
};
struct MlaRow4 {                   // P3: W_UK (CTAs 0..79; 64 rows each)
  const uint16_t* alias_post_qnd;  // +0x00 = row1.post_qnd
  const uint8_t* w_uk;             // +0x08 fp8 pages [80 CTA][0x3080] (993,280 B)
  uint64_t reserved_0;             // +0x10
  uint16_t* post_qnope;
  uint64_t pad[12];  // +0x20..+0x7f
};
struct MlaRow5 {  // P4/P5: split-KV attention (q1: CTA = split 0..31) + combine (q1: CTA = head
                  // 0..9)
  const uint16_t* alias_post_qnope;  // +0x00 = row4.post_qnope
  const uint16_t* alias_post_qpe;    // +0x08 = row1.post_qpe (after the RoPE tail)
  const uint16_t* alias_kv_cache;    // +0x10 = row3.kv_cache
  const uint16_t* alias_pe_cache;    // +0x18 = row0.pe_cache
  const int32_t* ids;  // +0x20 int32 [S][2048] local selected_ids table: read only if ll_recv
                       // == 0 and kv_len > 2048
  uint16_t* post_o;    // +0x28 bf16 [S][10][512] attention output (q4 token stride 0x2800 B)
  float* o_acc;        // +0x30 f32 split partial outputs
  float* lse;          // +0x38 (not read by this kernel)
  float* lse_acc;      // +0x40 f32 [S][32 split][12] split log-sum-exp: 10 live + 2 pad (pad never
                       // written); token stride 0x600 B
  const int32_t* alias_cur_pos;  // +0x48 = row0.cur_pos
  uint64_t heads_attn;           // +0x50 = 10 (not read by this kernel)
  uint64_t tmap_idx;             // +0x58 (not read by this kernel)
  const int32_t* ll_recv;  // +0x60 LL receive buffer [8*2048*2] int32 = {id0, tag, id1, tag} 16
                           // B records, record r = (q*32 + split)*32 + lane;
                           //       0 -> use the local ids table when kv_len > 2048
  uint64_t ll_tag;
  uint64_t pad[2];  // +0x70..+0x7f
};
struct MlaRow6 {                 // P6: W_UV (CTAs 0..79; 32 rows each)
  const uint16_t* alias_post_o;  // +0x00 = row5.post_o
  const uint8_t* w_uv;           // +0x08 fp8 pages [80 CTA][0x4080] (1,320,960 B)
  uint64_t reserved_0;           // +0x10
  uint16_t* post_uv;
  uint64_t pad[12];  // +0x20..+0x7f
};
struct MlaRow7 {                  // P7/P8: o_proj (128 CTAs x 48 rows) + push collective + residual
  const uint16_t* alias_post_uv;  // +0x00 = row6.post_uv
  const uint8_t* o_proj;          // +0x08 fp8 pages [128 CTA][5][0x6080] (15,810,560 B)
  const float* out_scale;         // +0x10 (not read by this kernel)
  uint16_t* hidden_out;           // +0x18 bf16 [S][6144] layer output
  const uint16_t* residual;       // +0x20 bf16 [S][6144] residual input x
  uint64_t epoch;                 // +0x28 (not read by this kernel)
  uint64_t reserved_fill_f32;     // +0x30 reserved; not read by this kernel
  uint64_t reserved_fill_bf16;    // +0x38 reserved; not read by this kernel
  uint64_t pad[8];                // +0x40..+0x7f
};
static_assert(sizeof(MlaRow0) == kRowBytes && sizeof(MlaRow1) == kRowBytes &&
                  sizeof(MlaRow2) == kRowBytes && sizeof(MlaRow3) == kRowBytes &&
                  sizeof(MlaRow4) == kRowBytes && sizeof(MlaRow5) == kRowBytes &&
                  sizeof(MlaRow6) == kRowBytes && sizeof(MlaRow7) == kRowBytes,
              "attention descriptor rows are 128 B");
static_assert(offsetof(MlaRow0, pe_cache) == 0x30 && offsetof(MlaRow1, post_qpe) == 0x28 &&
                  offsetof(MlaRow2, rope) == 0x10 && offsetof(MlaRow3, kv_cache) == 0x10 &&
                  offsetof(MlaRow4, post_qnope) == 0x18 && offsetof(MlaRow5, ids) == 0x20 &&
                  offsetof(MlaRow5, o_acc) == 0x30 && offsetof(MlaRow5, lse_acc) == 0x40 &&
                  offsetof(MlaRow5, ll_recv) == 0x60 && offsetof(MlaRow5, ll_tag) == 0x68 &&
                  offsetof(MlaRow6, post_uv) == 0x18 && offsetof(MlaRow7, hidden_out) == 0x18 &&
                  offsetof(MlaRow7, epoch) == 0x28 && offsetof(MlaRow7, reserved_fill_bf16) == 0x38,
              "attention descriptor field offsets");
struct MlaTable {
  MlaRow0 r0;
  MlaRow1 r1;
  MlaRow2 r2;
  MlaRow3 r3;
  MlaRow4 r4;
  MlaRow5 r5;
  MlaRow6 r6;
  MlaRow7 r7;
};
static_assert(sizeof(MlaTable) == kNRows * kRowBytes, "8 rows x 128 B");

// ---------------------------------------------------------------------------------------------------------------
// 3. Rank 0 runs no q1 indexer computation.  For q1 it republishes the accepted selection
//    (indexer_q1.cu, descriptor rows from backend._shared_context) as LL records
//    {id0, tag, id1, tag} into every attention rank's ll_recv buffer, and its output CTAs join
//    the push collective of section 6 (slot = rank - (rank > dst)).
// ---------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------
// 4. Sync words: byte offsets in the 256-byte q1 sync block.  CTA 0 clears the block at the start
//    of every launch and then releases reset_token; terminal values after a launch (q = 1):
//      C0 +0x00 = 0x100 (128 Group1 CTAs x +2 after P1)
//      C1 +0x10 = 0x100 (128 x +2 after q_b_proj)
//      C2 +0x20 = 0x100 (16 RoPE tails 132..147 x +0x10)
//      C3 +0x30 = 0x100 (4 kv-norm tails 128..131 x +0x40)
//      C4 +0x40 = 0x100 (128 x +2 after W_UK)
//      C6 +0x50 = 0x100 (q1: 32 attention CTAs x +8; q4: 128 x +2)
//      C5 +0x54 = 0x80000000 (attention CTAs: CTA 0 adds kC5Head<Q>, the others +1; waiters
//                 test the sign bit)
//      C7 +0x60 = 0x100 (128 x +2 after W_UV)
//    Waiters test `>= 0x100` / sign bit (C5). Arrive = block barrier, then thread 0: fence + atomic
//    add.
// ---------------------------------------------------------------------------------------------------------------
constexpr int kWordC0 = 0x00 / 4, kWordC1 = 0x10 / 4, kWordC2 = 0x20 / 4, kWordC3 = 0x30 / 4,
              kWordC4 = 0x40 / 4;
constexpr int kWordC6 = 0x50 / 4, kWordC5 = 0x54 / 4, kWordC7 = 0x60 / 4;
constexpr uint32_t kSyncFull = 0x100u;
constexpr uint32_t kWeightGroup1 = 2u;     // C0, C1, C4, C7 (and C6 for q4)
constexpr uint32_t kWeightRope = 0x10u;    // C2
constexpr uint32_t kWeightKvNorm = 0x40u;  // C3
template <int Q>
constexpr uint32_t kWeightC6 = (Q == 1) ? 8u : 2u;
template <int Q>
constexpr uint32_t kC5Head = (Q == 1) ? 0x7fffffe1u : 0x7fffff81u;  // + (Q*32 - 1) x 1 = 0x80000000
constexpr uint32_t kSyncBlockBytes = 0x100;  // 8 rows x 16 B (the host snapshots 0x100 B)

// ---------------------------------------------------------------------------------------------------------------
// 5. Geometry shared by both instantiations.
// ---------------------------------------------------------------------------------------------------------------
constexpr int kHidden = 6144, kQlora = 2048, kKvlora = 512, kNope = 192, kRope = 64, kHeads = 10,
              kVdim = 256, kSelected = 2048;
constexpr int kGroup1Ctas = 128;
constexpr int kQkvAProjCtas =
    82;  // 32-row blocks: rows 0..2047 -> post_q, 2048..2559 -> post_kv, 2560..2623 -> k_pe raw
constexpr int kQkvAProjRows = kQlora + kKvlora + kRope;  // 2624
constexpr int kQbProjCtas = 80;  // 32 packed rows each: 0..1919 = head*192 + dim (nope), 1920..2559
                                 // = 1920 + head*64 + dim (pe)
constexpr int kQbProjRows = kHeads * (kNope + kRope);
constexpr int kQbProjNopeRows = kHeads * kNope;  // 1920
constexpr int kUkCtas = 80;                      // 64 rows each of head*512 + out
constexpr int kUvCtas = 80;                      // 32 rows each of head*256 + out
constexpr int kOProjCtas = 128;                  // 48 rows each of 6144
constexpr int kOProjRows = 48;
constexpr int kSplitRows = 64;
constexpr int kSplits =
    32;  // q1: attention CTA = split (0..31); q4: (query = cta >> 5, split = cta & 31)
constexpr int kKvNormTail =
    128;  // CTAs 128..131: kv-norm of token t = cta - 128 (t < Q works, the rest only arrive C3)
constexpr int kRopeTail =
    132;  // CTAs 132..147: RoPE of token t = cta - 132 (t < Q works, the rest only arrive C2)
constexpr int kLseStride = 12;  // f32 per split in lse_acc (10 live + 2 pad)
constexpr size_t kOAccSplitElems = size_t(kHeads) * kKvlora;           // 5120 f32 = 0x5000 B
constexpr size_t kOAccTokenElems = size_t(kSplits) * kOAccSplitElems;  // 0xa0000 B
constexpr size_t kLseTokenElems = size_t(kSplits) * kLseStride;        // 0x600 B
constexpr int32_t kIdSentinel = 0x32000;  // dense ids past min(kv_len - 1, 2047)
constexpr float kExp2Scale = 0.09016844f;
constexpr float kRmsEps = 1e-5f;
constexpr uint32_t kNanF32 = 0x7fc00000u;
constexpr uint32_t kNanBf16x2 = 0x7fc07fc0u;
constexpr int kNanF32Lines = 1024;  // 16 KB of 16 B lines
template <int Q>
constexpr int kNanBf16Lines =
    Q * 288;  // [q][9][256] bf16 = q x 4608 B (16-byte lines: q1 288, q4 1152)

// ---------------------------------------------------------------------------------------------------------------
// 6. K5 push collective
//    destination block of (token t, sender rank, receiver dst, CTA c):
//        peers[dst] + base + t*0x38000 + slot*0x8000 + parity*0x4000 + c*0x80,   slot = rank -
//        (rank > dst),   parity = epoch & 1,

//    line k = 6 bf16 (this CTA's o_proj rows 48c+6k..48c+6k+5) in dwords 0..2 + E (epoch low 32
//    bits) in dword 3. Receiver (own inbox peers[rank], slot s <- rank s + (s >= rank)) polls every
//    line until its dword 3 == E. hidden[t][48c + i] = bf16(f32(bf16(sum_{rank 0..7}
//    f32(slice_rank[i]) )) + f32(residual[t][48c + i]) ), rank order, own slice inserted at its
//    rank.  The q1 instance only touches token slice 0 (224 KB of the 896 KB inbox).
// ---------------------------------------------------------------------------------------------------------------
constexpr size_t kInboxTokenStride = 0x38000;  // 7 slots x 0x8000
constexpr size_t kInboxSlotStride = 0x8000;    // 2 parities x 0x4000
constexpr size_t kInboxParityStride = 0x4000;  // 128 CTAs x 0x80
constexpr size_t kInboxCtaStride = 0x80;       // 8 lines x 16 B
template <int Q>
constexpr size_t kInboxBytes = size_t(Q) * kInboxTokenStride;  // q1 0x38000, q4 0xe0000
constexpr int kLlRecordsPerCta =
    32;  // LL ids records: record r = (q*32 + split)*32 + lane, 16 B {id0, tag, id1, tag}
constexpr size_t kLlRecvInts = size_t(8) * 2048 * 2;

// Dynamic shared memory of one instantiation (bytes): the larger of the staged activation rows
// and the per-head latent/PE buffers, plus the fixed scratch terms.
template <int Q>
constexpr int kModeSmemBytes =
    ((Q * kHidden * 2 > kHeads * (kKvlora + kRope) * 4) ? Q * kHidden * 2
                                                        : kHeads * (kKvlora + kRope) * 4) +
    Q * 32 * 8 * 4 + 128 + kHeads * kSplitRows * 4 + kSplitRows * 4 + 512;

}  // namespace flashinfer::glm5_tp8::mla_q1
