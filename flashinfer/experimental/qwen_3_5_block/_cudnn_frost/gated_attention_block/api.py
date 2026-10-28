# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated attention block, forward -- projection, QK-norm/RoPE, SDPA, gate, out
projection as one FROST block (bf16 / FP8 / MXFP8, optional MXFP4 weights and fp4 O).

FROST's first MODEL-LEVEL block: a SET of FROST kernels behind ONE API, one
workspace, one call. Every stage is a FROST kernel; there are no cuBLAS or
cuDNN-graph call-outs for the GEMMs (an explicit scoping decision).

Geometry is Qwen3.5's gated attention (provenance in comments only — the shipped
name is by op geometry, per the FROST engine contract § 8).

.. note::

   **MAINTENANCE.** This docstring is the block's design record in the code, and
   it makes claims that stop being true the moment a stage lands or two stages
   merge. Update it in the SAME commit as: any stage becoming real, any fusion
   that deletes a stage or a workspace region, any dtype the block starts to
   serve, and any change to the ``qkvg_offsets`` / :class:`SavedForBackward`
   contracts. A stale design record is worse than none, because it gets trusted.
   The per-stage **Fusion status** paragraphs below are the parts that go stale
   first.

The op graph this file implements, in pipeline order::

    h [B, S, d_model]                                    (post input_layernorm)
     |
     |  (1) QKV+GATE projection   h @ W_qkvg^T
     +----------------------------------------------------------------+
     |  Q [B,S,H_q,D] | GATE [B,S,H_q,D] | K [B,S,H_kv,D] | V [B,S,H_kv,D]
     |
     |  (2)+(3) ONE kernel: QK-RMSNorm per head over D, then partial mRoPE on
     |          the first ROPE_DIM. Q and K only -- V is NOT normed. Dims
     |          [ROPE_DIM, D) pass through. fp32 throughout, ONE rounding.
     |          ``geometry.qk_norm=False`` drops the RMSNorm: RoPE-only Q/K, no
     |          norm weights (pass None), no rstd; [ROPE_DIM, D) copied bit-exactly.
     |
     |  (4) SDPA        O = softmax(QK^T * scale + mask) V        GQA H_q/H_kv
     |
     |  (5) gate        O_gated = O * sigmoid(GATE)     elementwise, BEFORE (6)
     |
     |  (6) out projection   O_gated @ W_o^T
     v
    out [B, S, d_model]

**FUSION KNOB (2026-09-11): ``fuse_gate=True`` folds stage (5) into stage (4)'s
epilogue.**  The sigmoid gate is a PRODUCTION feature of the Rubin d256 SDPA
kernels (``sdpa/fwd/kernels/sm107/prefill_d256_{f16,fp8}.py`` behind
``TemplateParams.epilogue_gate``; the engine rows claim it through
``epilogue_gate_d_shapes``): a TMA-staged GATE read + ``O *= sigmoid(GATE)``
after the dead-row select.  ``_Sdpa`` reaches it through the shipped adapter
(``SdpaFwdDslSm100(sample_gate=...)`` / ``execute(gate=...)``) -- the block
owns no SDPA kernel in ANY configuration.  Inference only (no pre-gate ``O``
for ``dG``); composes with ``fuse_norm_rope`` into the 3-launch block
proj(+norm+rope) -> sdpa(+gate) -> out_proj.

**FUSION KNOB (2026-09-11): ``fuse_norm_rope=True`` folds stages (2)+(3) into
stage (1)'s epilogue** (``_FusedQkvProjection`` over the fork
``kernels/proj_gemm_norm_rope.py``): Q/K tiles are normed + rotated on the fp32
accumulator and the slab is written once. Inference / in-place only (no
``q_pre``); OFF by default until the perf node ranks it.

**FUSION KNOB (2026-10-08): ``fuse_prepare=True`` folds stages (2)+(3) and the
serving write-through (4w) into ONE launch** (:class:`_QsaPrepare` over
``kernels/qsa_prepare.py``): Q / K are normed + rotated in place and the post-RoPE
K, the V and the raw indexer key land in the paged pools from the same registers,
so a pool row IS the row attention consumes and every output is BITWISE the
unfused chain's.  A performance knob (same function); needs ``paged_kv_page_size
> 0`` and the in-place bf16 / f16 pipeline, OFF by default.  The kernel also
carries the indexer's decode-step arms (the queries' norm + RoPE, the raw-key
ring + the incremental compress) for the serving step; the block binds them when
its decode mode lands.

**STATUS (2026-09-11): every stage is REAL.** The two projections drive the
shipped FROST GEMM, stages (2)+(3) are one FROST kernel of this block's own,
stage (4) is the shipped FROST SDPA, stage (5) is this block's elementwise
kernel. Two OPTIONAL fusions, both off by default until the perf node ranks
them: ``fuse_norm_rope`` (stages (2)+(3) into (1)'s epilogue) and ``fuse_gate``
(stage (5) into (4)'s epilogue); with both on the block is THREE launches,
``proj(+norm+rope) -> sdpa(+gate) -> out_proj``. The block targets **Rubin
(SM107) only** for now — every other arch is declined explicitly rather than
served slowly.

**The default pipeline is deliberately UNFUSED**: five stages, five launches
(four under ``inplace_qkv``), every intermediate materialised in the caller's
workspace. That is the honest baseline; it is also the point. Per the design
record, the API boundary is the expensive thing to change later and the
partitioning behind it is cheap, so the stage cuts below move behind knobs
(see "Fusion status" per stage) while this file's signature does not.

Two structural facts that will not move, and that shape everything here:

* **(6) can never share a kernel with (4).** ``o_proj`` contracts over
  ``H_q * D`` — ALL heads — while an SDPA CTA owns one head. Fusing it needs a
  cross-CTA split-K reduction. This is why the block is necessarily a *set* of
  kernels and why the API, not any one kernel, is the deliverable.
* **(5) must run AFTER the SDPA epilogue's dead-row substitution.** A row with
  no unmasked column gets ``O := 0`` by SELECT (``sdpa-invariants.md`` § 2);
  multiplying accumulator residue by ``sigmoid(GATE)`` before that substitution
  propagates NaN. Whenever (5) folds into (4)'s epilogue, it goes after the
  select, never before.

Precision roadmap
-----------------

**bf16 throughout by default** (fp32 accumulate in every GEMM and in the
softmax).  **FP8 (2026-09-11): an E4M3 pipeline with static per-tensor scales
is wired in TWO configurations** -- pass an e4m3 ``h``/weights and a
:class:`QuantSpec`:

* **UNFUSED** (both fusion knobs off; 7 stages = 9 kernel launches, the Q/K/V
  quantize stage being three launches): FP8 projections with the descale
  folded into a scalar-multiply epilogue (bf16 out), bf16 norm+RoPE in place,
  two quantize passes (Q/K/V, then gated O) feeding the Rubin per-tensor FP8
  SDPA and the FP8 out projection.  The quantize passes are the visible price
  of "unfused".
* **FULLY FUSED** (``fuse_norm_rope=True, fuse_gate=True``, 3 launches): the
  projection fork ``kernels/proj_gemm_norm_rope_fp8.py`` descales, norms,
  rotates AND quantizes in its epilogue -- e4m3 Q / K / V into COMPACT
  per-tensor ``q8`` / ``k8`` / ``v8`` (each == compact BSHD ``[B, S, H, d]``,
  exactly what the unfused FP8 SDPA reads), the GATE as bf16 into ``gate16``
  -- and the production FP8 d256 SDPA (``prefill_d256_fp8.py`` with
  ``epilogue_gate`` on and ``has_amax_o=False``: the block runs static scales
  and reads no ``Amax_O``, so the atomic is compiled out) reads those compact
  buffers, multiplies by ``sigmoid(gate16)`` after the dead-row select and
  writes e4m3 ``o8`` for the out projection.  No bf16 slab, no quantize
  passes, no bf16 O: 33792 B/token of workspace against 68608 unfused
  (-51 %).  (Round 1 wrote ONE token-major ``qkv8 [T, n_qkv]`` slab and read
  it STRIDED: measured +7.2 % on the SDPA at 32K -- K/V lines refetched at a
  9216 B stride once the gate stream evicts them -- where compact reads are
  -3.5 %; STATUS.md "SDPA decomposition".  Hence compact per tensor.)  The two
  are NOT bit-identical (the unfused path rounds to bf16 twice); both are
  scored against the fake-quant fp32 oracle.
* Anything in between (one knob) is a typed decline naming both knobs.  The
  FULLY FUSED FP8 pipeline is inference-only; the UNFUSED one trains (see
  "Training under FP8 / MXFP8" below).  FP8 + ``seq_lens_present`` (a dense
  padding mask, incl. an
  EMPTY entry) is SERVED since 2026-09-15: the Rubin FP8 d256 SDPA's
  empty-KV-entry hang is gone (8/8 fresh processes at S=1000 / 512), and the
  block's dead-entry oracle test pins ``out[dead] == 0`` exactly.

**MXFP8 (2026-09-15, PR-B): an E4M3 + per-32-block E8M0 pipeline is wired in
the same TWO configurations** -- pass e4m3 ``h`` / ``W_qkvg`` codes, their
F8_128x4 scale-factor blobs (``sample_h_sf`` / ``sample_w_qkvg_sf`` at
declaration, ``h_sf`` / ``w_qkvg_sf`` at execute) and an :class:`MxQuantSpec`:

* **UNFUSED** (9 stages = 9 kernel launches -- the same launch count as
  unfused FP8, whose 7 stages also issue 9): the block-scale FROST GEMM (``block_scale=True``,
  the E8M0 dequant is exact and happens IN the MMA, so there is no ``alpha``)
  writes the bf16 slab; norm+RoPE in place; THREE ``quantize_mxfp8`` launches
  (Q / K ROWWISE along D, V COLUMNWISE along S -- ``kernels/quantize_mxfp8.py``)
  write compact e4m3 ``q8`` / ``k8`` / ``v8`` + the SDPA's own F8_128x4 SF
  blobs (``sf_q`` / ``sf_k`` / ``sf_v``, one 1 KiB tile per ``(b, h, 128-row
  tile)``; V's is D-PLANE-MAJOR, ``mma-tma-matrix.md`` § 7); the production
  ``sm107/prefill_d256_mxfp8.py`` (``SdpaFwdDslSm100(pertensor_fp8=False)``,
  NATURAL at (256, 256), the 2-CTA pair on dense legs) writes bf16 O; the bf16 gate; a PER-TENSOR
  ``quantize_o`` (``MxQuantSpec.scale_o``; D1 -- never the rowwise recipe) and
  the per-tensor FP8 ``out_proj`` with ``alpha_o = descale_w_o / scale_o``.
* **FULLY FUSED** (``fuse_norm_rope=True, fuse_gate=True``, 3 launches): the
  MXFP8 GEMM fork twin (``kernels/proj_gemm_norm_rope_mxfp8.py`` via
  ``run_fused_proj_gemm_mxfp8``) norms / rotates / BLOCK-quantizes in its
  epilogue and writes ``q8`` / ``k8`` / ``v8`` + ``sf_q`` / ``sf_k`` / ``sf_v``
  + bf16 ``gate16``; the gated production MXFP8 SDPA writes e4m3 O UNSCALED
  (the kernel has no per-tensor ``scale_o`` -- D8: ``MxQuantSpec.scale_o``
  must be 1.0 on this path, typed decline otherwise); FP8 ``out_proj``.  The
  fork twin is feature-detected on the runner name (``_fork_supports_field``),
  so the typed ``NotImplementedError`` returns only on a checkout without the
  twin -- never a silently un-quantized path.
* Same envelope as FP8: both knobs or neither (the fully fused twin is
  inference-only, the unfused pipeline trains), e4m3 only
  (E5M2 is a typed decline), ``d_model % 128 == 0`` (whole SF atoms along K),
  padding (``seq_lens_present``) served with the same dead-entry contract.

**fp4 (2026-09-17): two modes RIDE the MXFP8 pipeline as appended
:class:`MxQuantSpec` fields** (unrepresentable on :class:`QuantSpec` / bf16
rather than declined); every existing caller and every existing workspace
offset is byte-identical (pinned by a frozen layout snapshot):

* **MXFP4 weights** (``w_qkvg_dtype=torch.float4_e2m1fn_x2``): ``W_qkvg``
  arrives as packed e2m1 codes ``[n_qkvg, d_model // 2]`` (two per byte, LOW
  nibble = even k; the STORAGE shape is what ``check_support`` checks) with the
  UNCHANGED E8M0 / 32 ``w_qkvg_sf``; stage (1) runs the FROST catalog's MIXED
  block-scale row (``fp8_e4m3 x fp4_e2m1``, E8M0 per 32 on both sides) --
  the same 9 launches, no new stage, no new slot.  FULLY FUSED too: the fused
  MXFP8 projection fork's e2m1-B arm (``NormRopeFusionParams.weight_fp4``,
  feature-detected -- a checkout whose fork lacks the field declines typed)
  reads the same packed codes and the same blob inside the norm+RoPE+quant
  epilogue, so ``fuse_norm_rope`` with an e2m1 ``W_qkvg`` is the same 3
  launches as the e4m3 fused pipeline (inference only, dense only, like it).
* **fp4 O** (``o_fp4=Fp4Format.NVFP4 | Fp4Format.MXFP4``; ONE enum member =
  e2m1 codes x scale dtype x block, e4m3 / 16 or E8M0 / 32, so an illegal
  pairing cannot be spelled): the per-tensor tail (``quantize_o`` +
  ``alpha_o`` FP8 ``out_proj``) is replaced by ``kernels/quantize_fp4.py``
  (bf16 gated O -> e2m1 codes ``o4`` + the out-projection GEMM's PADDED
  F8_128x4 blob ``sf_o``, sized by ``proj_gemm.sf_blob_bytes`` -- the GEMM
  contract, never the SDPA's ``_sf_slot_bytes``) and the fp4 x fp4
  block-scale ``out_proj`` against an e2m1 ``W_o`` ``[d_model, H_q*D // 2]``
  of the SAME format with its blob ``sample_w_o_sf`` / ``w_o_sf`` (appended
  arguments, required iff ``o_fp4``, refused otherwise).  No per-tensor scale
  on either side: ``scale_o`` and ``descale_w_o`` MUST be 1.0 (typed
  ``ValueError`` -- the block-scale GEMM has no alpha to carry them).
  UNFUSED: still 9 launches (``quantize_fp4_o`` in ``quantize_o``'s place);
  FULLY FUSED: **4 launches** -- the gated MXFP8 SDPA writes **bf16** O
  (``sdpa_o_dtype = act``; no SDPA ``Capabilities`` change, so the
  support-matrix tracker is untouched), then the fp4 quantize, then the fp4
  ``out_proj``.  Workspace: ``o8`` is not reserved (-1); ``o4`` / ``sf_o`` are
  appended at the END of the arm.  ``d_head % (4 * block) == 0`` (whole
  4-block scale words per head; d=256 passes both formats), else a typed
  ``NotImplementedError``.  A dead ragged entry quantizes to codes 0 exactly
  (NVFP4 scale = the ``2^-9`` e4m3 floor, MXFP4 scale byte ``0x00``).
* Both compose (row 9: the mixed GEMM at (1), the fp4 tail at (5q')/(6')); both
  TRAIN on the unfused pipeline (``save_for_backward=True`` writes the MXFP8
  record byte for byte -- the fp4 tail changes only the workspace's ``o8`` into
  ``o4`` / ``sf_o`` -- and ``GatedAttentionBlockBwd(quant=MxQuantSpec)`` differentiates
  it over the caller's transposed e2m1 artifacts, ``api_bwd.py`` "The fp4 weight
  modes"); the fused forks stay inference-only.  NOT served: an fp4 ``h``, an fp4
  ``W_o`` against an e4m3 O, e4m3 scales at block 32 / E8M0 at block 16, a global
  (per-tensor) scale on either fp4 side.

**Training under FP8 / MXFP8 (2026-10-01): the UNFUSED quantized pipelines
write the bf16 training record.**  ``save_for_backward=True`` with a
:class:`QuantSpec` / :class:`MxQuantSpec` runs the same 9 launches as the
quantized inference forward and writes the SAME :class:`SavedForBackward` the
bf16 training forward writes -- the bf16 slab (stage (1)'s dequantized product)
with PRE-norm Q/K bands, the bf16 pre-gate ``O``, the exact fp32 LSE,
``rstd_q`` / ``rstd_k`` -- with ``h`` the caller's e4m3 codes (``h_sf`` is a
forward input the record never carries).  What differs from quantized
inference is ONE routing: norm+RoPE writes the normed Q/K OUT of place into
compact bf16 workspace slots (``q`` / ``k``, +17 KiB/token at the 397B
geometry) instead of back over the slab, so the slab's Q/K columns stay
pre-norm for the record, and the quantize stages read those slots.  The hazard
this guards: the inference pipeline norms IN PLACE, so lifting the training
decline without the slots would hand the backward POST-norm bands that it
differentiates as pre-norm -- wrong ``dQ``-side gradients and norm-weight
gradients, no crash, and ``out`` still bitwise the inference block's.  ``out``,
``O``, ``LSE`` and the GATE / V bands are bitwise the inference forward's (same
kernels, different buffers).  The FULLY FUSED quantized pipelines (no slab,
e4m3 ``O``) stay behind the fusion knobs' own training guards and the fp4 modes
are a typed decline.  The bf16 backward consumes this record given the
dequantized bf16 ``h`` and weights (``test_block_backward.py``); the native
fp8 / mxfp8 backward is the follow-up.

Three facts the MXFP8 design rests on, kept here because a port will drop them:

* V is quantized COLUMNWISE for the BMM2, and its MXFP8 scale factors are
  D-PLANE-MAJOR in GMEM while Q/K's are per-tile contiguous -- reusing one SF
  layout for all three is a silent wrong answer that only shows up at
  ``d > 128`` and ``S > TILE_N`` (``mma-tma-matrix.md`` § 7).  The SF-order
  S-sweep in ``test_block_mxfp8.py`` is the detector.
* ``TILE_K_HW`` / idesc ``k_dim`` are **arch-opposite** for FP8: Blackwell wants
  ``k_dim=0`` at ``TILE_K_HW=32``, Rubin ``k_dim=1`` at 64. Read
  ``mma-tma-matrix.md`` § 1 before picking either.
* the RMSNorm in stage (2) and the sigmoid in stage (5) stay fp32-compute
  regardless of the storage dtype.

Public signatures grow APPEND-ONLY (a ``QuantSpec`` argument lands at the end of
the argument list with a ``None`` default), so nothing below needs a placeholder
field today.

Design record (fusion analysis, backward graph, multi-GPU/CP, measurement
protocol): an internal design record (not part of this tree).
Backward: :mod:`cudnn.gated_attention_block.api_bwd` — read this file first, the
backward is defined against the :class:`SavedForBackward` contract below.
PyTorch oracle and perf baseline:
``test/python/gated_attention_block/cutedsl/reference.py``.

**THD / packed sequences (``thd=True``):** ``h`` / ``cos`` / ``sin`` / ``out``
are PACKED token matrices ``[T, .]`` (or ``[1, T, .]``) holding ``num_sequences``
sequences of at most ``max_seq_len`` tokens each, and ``execute(seq_lens=)``
carries the per-sequence lengths -- ``[B]`` int32 lengths, or ``[B+1]`` int32
prefix sums under ``cu_seqlens=True`` -- for the Q and the KV side alike.  Every
stage but the SDPA is token-wise and runs unchanged over the ``T`` rows; the
SDPA runs the Rubin d256 kernels' varlen arm (packed ``(T, H, D)`` operands, the
lengths read on device, the natural tile order) and writes the head-major packed
LSE ``[1, H_q, T]`` -- the dense record's ``[B, H_q, S]`` at ``B = 1, S = T``, so
the :class:`SavedForBackward` contract is unchanged (``seq_lens`` REQUIRED and
``seq_lens_form`` naming its form).  ``cos`` / ``sin`` are PER-TOKEN tables whose
positions restart at every sequence; the caller packs them.  Served: bf16 / fp16
(inference and training), the UNFUSED per-tensor FP8 and MXFP8 pipelines
(inference and training; the fp4 modes that ride MXFP8 included -- the three
``quantize_mxfp8`` stages run their PACKED arm and write the SDPA row's
per-sequence-tile-padded scale-factor layout at the slot capacity
``_sf_slot_bytes_packed``; ``kernels/quantize_mxfp8.py`` "PACKED sequences"),
``fuse_norm_rope`` and ``fuse_gate`` for bf16 / fp16 inference (the gate rides the
packed d256 prefill's epilogue through the SDPA's THD gate descriptor: one launch
fewer, bitwise the dense gated block at ``B = 1``), block-sparse attention (``QsaSpec``
with caller lists: ``block_ids`` ``[T, top_k]`` with every id relative to the token's
OWN sequence, the sparse core's packed arm over the same views).  Typed declines: the
fully fused quantized pipelines (the quantized THD SDPA bodies have no THD gate
descriptor, and the fused MXFP8 projection fork decodes ``(b, s_tile)`` once per
128-row GEMM tile, which a packed tile may straddle), ``seq_lens_present`` (the
two length contracts are mutually exclusive) and the in-block indexer under ``QsaSpec``
(the paged write-through composes with the packed sparse block: the stage addresses tokens).
The lengths are device data, never read on the host:
the caller's contract is every length in ``[0, max_seq_len]`` and
``sum(lengths) == T`` -- the SDPA leaves rows past the live total UNWRITTEN, and
the gate and the backward's weight-gradient GEMMs read every one of the ``T``
rows (a contract violation, not a checked error).  A batch whose sequence count
varies pads with zero-length sequences.

Not in scope for v1, in the order they are likely to land: an MXFP8 (e4m3
block-scaled) O / ``out_proj`` (D1 keeps the e4m3 O per-tensor; the block-scaled
out projection exists only in the fp4 formats above); the graph-API engine row;
the packed-sequence arm of the fully fused MXFP8 pipeline. This is a
frontend-only OSS API first; a manifest family + ``Capabilities`` comes when the
stages exist to be honest about.
"""

from __future__ import annotations

import math
import os
import warnings
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from cuda.bindings import driver as cuda

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import as_torch_stream
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import APIBase, TensorDesc, TupleDict

# ---------------------------------------------------------------------------
# 1. Stage (1) layout — exactly what the QKV+GATE matmul computes and where it lands
# ---------------------------------------------------------------------------
#
# ONE GEMM, four logical outputs. This section is the contract every other stage
# is written against, so it is spelled out in full rather than left to the
# kernel.
#
# THE GEMM
# --------
#   A  = h        [M, K]   M = B*S, K = d_model      row-major, bf16
#   B  = W_qkvg   [N, K]   N = (2*H_q + 2*H_kv) * D  row-major, bf16   (read transposed)
#   C  =          [M, N]   fp32 accumulate, bf16 epilogue store
#
# i.e. the ordinary ``nn.Linear`` TN form, ``out = h @ W_qkvg.T``. At the 397B
# full-attention layer: M = B*S, N = 17408, K = 4096.
#
# THE N AXIS
# ----------
# N is four contiguous blocks, each head-major with D innermost:
#
#     col   0                  H_q*D            2*H_q*D    +H_kv*D   +H_kv*D
#           |------- Q -------|------ GATE -----|--- K ---|--- V ---|
#   397B:   0                 8192              16384     16896     17408
#           |<--- 32 heads -->|<--- 32 heads --->|<- 2 ->|<- 2 ->|
#
# A geometry that declares block-sparse attention WITH an indexer band
# (``GatedAttentionBlockGeometry.qsa.index_band``) appends a FIFTH block,
# INDEX, below V: ``(index_heads + index_kv_heads) * index_head_dim`` columns
# (640 at the Flash-Next geometry: 4 indexer query heads then 1 raw key head,
# 128 each), so N = 13952 there.  Every geometry without it has exactly the
# four blocks above and the same numbers it always had.
#
# Within a block, head ``j`` owns columns ``[j*D, (j+1)*D)``. Q and GATE are
# adjacent because that is how the model produces them (one double-width
# ``q_proj``), not for any kernel reason.
#
# THE ALIGNMENT INVARIANT, AND WHY IT IS LOAD-BEARING
# --------------------------------------------------
# Every block boundary is a multiple of ``QKVG_TILE_ALIGN`` (64), so for any
# supported ``TILE_N`` **no GEMM output tile ever straddles two blocks**
# (``qkvg_tile_plan`` proves it, ``validate`` enforces it). Consequences:
#
#   * the epilogue can specialize PER BLOCK on a per-tile constant rather than
#     predicating column-by-column -- which is what makes the future
#     quantization epilogue cheap (Q/K rowwise, V columnwise, GATE passthrough,
#     each with its own scale-factor layout);
#   * each block's compact row stride is 16-byte aligned at bf16, so a TMA
#     descriptor over it is legal.
#
# At 397B the boundaries are 8192 / 16384 / 16896, all multiples of 256, so any
# TILE_N in {64, 128, 256} works. The invariant is checked, not assumed: a
# geometry with, say, H_kv*D = 96 raises at build time instead of producing a
# straddling tile nobody notices.
#
# WHERE THE TILES ARE STORED: FOUR COMPACT BUFFERS, NOT ONE FUSED ONE
# -------------------------------------------------------------------
# DESIGN RECORD OF THE STAGE-(1) FORK -- NOT what the shipped UNFUSED path does
# (2026-09-29).  Today stage (1) is the unforked FROST GEMM and writes ONE fused
# ``[T, N]`` slab -- the workspace ``proj`` slot, or under ``save_for_backward``
# the caller-owned ``SavedForBackward.proj_slab`` (its bands are the strided
# views ``saved_slab_views`` spells) -- and stage (3b) compacts V out of it.  The
# four-buffer epilogue below is what the fork writes when it lands; the
# reasoning is kept because the fork is still the plan.
#
# The epilogue selects one of FOUR output descriptors from the N-tile index and
# writes a BSHD-COMPACT buffer per block:
#
#     Q     [B, S, H_q,  D]   strides (S*H_q*D,  H_q*D,  D, 1)
#     GATE  [B, S, H_q,  D]   strides (S*H_q*D,  H_q*D,  D, 1)
#     K     [B, S, H_kv, D]   strides (S*H_kv*D, H_kv*D, D, 1)
#     V     [B, S, H_kv, D]   strides (S*H_kv*D, H_kv*D, D, 1)
#
# The tempting alternative -- one fused ``[B, S, N]`` buffer, with Q/K/V as
# strided views -- is REJECTED, and the reason is specific rather than
# aesthetic. Such a view has a padded token stride (N, not H*D). The SDPA's
# dense path accepts that layout (``sdpa/graph_analyzer.dense_layout_ok``
# explicitly allows padded strides). The historical tensor adapter normalized
# layouts outside its direct-binding domain with an implicit gather. Current
# plans bind TMA-compatible strides directly and prepare any required copies
# with explicit scratch; the block still chooses a layout requiring no copies.
# A **gather copy of Q**,
# at 1M tokens is 16 GiB moved twice, silently, inside a block whose whole
# premise is that it does not do that (AGENTS.md Rule 2).
#
# The SDPA's THD arm does address such strides natively -- its own docstring
# names this exact case, "a K/V view of a kv-interleaved [T, 2, H, D] buffer
# ... the layout torch.nn.attention.varlen users produce by slicing a fused KV
# projection" (``api_dsl.py`` ``_thd_check_strides_native``). So the fused-buffer
# layout is reachable, just not on the dense path today. Writing four compact
# buffers gets zero-copy on BOTH arms with no SDPA change, and it is also
# strictly less memory: 17 GiB of Q+GATE+K+V rather than a 34 GiB fused slab
# plus per-tensor copies.
#
# LAYOUT VARIANTS CONSIDERED AND NOT TAKEN (revisit with a measurement, not an
# argument):
#   * **Q/GATE interleaved per head** (``[q_0 g_0 q_1 g_1 ...]``). Still a legal
#     rank-4 view (head stride 2*D), and it is what a fused epilogue would want
#     if one CTA produced a head's Q and its gate together. Buys nothing while
#     the two land in separate buffers anyway, and costs a checkpoint permute.
#   * **K/V interleaved** (``[T, 2, H_kv, D]``). Better KV-tile locality for the
#     SDPA, and a shape the THD path already handles. Worth measuring once the
#     SDPA is the bottleneck; K and V are 1/16 of Q here, so it is a small
#     lever.
#
# THE CALLER'S SIDE
# -----------------
# ``qkvg_from_hf`` is the documented entry point: it assembles ``W_qkvg`` and
# the two norm weights from a HF Qwen checkpoint's five tensors ONCE, at load
# time, applying the two conventions every Qwen checkpoint from Qwen3-Next on
# shares -- the double-width ``q_proj`` is split PER HEAD (``view(..., H_q,
# 2*D).chunk(2, dim=-1)``), and the QK-norm weight is zero-centered (applied as
# ``1 + w``). ``build_fused_qkvg_weight`` is the layout-explicit assembler
# underneath it. It takes ``q_gate_layout`` because the model's own convention
# for splitting ``q_proj`` is not something to guess: "flat" chunks
# ``[..., 2*H_q*D]`` into all-Q then all-GATE, "per_head" views it as
# ``[..., H_q, 2*D]`` first and chunks the head dim. The two differ by a row
# permutation of ``q_proj.weight``, and picking wrong is a silent wrong answer
# -- so it is a parameter, checked by a round-trip test, never a hot-path
# conversion either way, and an OMITTED layout (which still means "flat")
# emits a ``FutureWarning`` rather than guessing silently. FutureWarning, not
# DeprecationWarning: CPython's default filters show a ``DeprecationWarning``
# only to a ``__main__`` caller, so from a model loader living in a library
# module -- where every serving stack's model file lives -- it would be
# dropped unseen, and the wrong gate layout it announces is a finite,
# plausible, silently wrong output.


QKVG_TILE_ALIGN = 64
_QK_NORM_ROPE_THREADS = 128
_ELEMENTWISE_THREADS = 128
_ELEMENTWISE_ROWS_PER_GROUP = 2
# H is fixed per compiled artifact either way (it is in the compile-cache key),
# so baking it into the address math costs no generality -- it only turns a
# software integer divide per row into a shift. Runtime arm kept for A/B.
_ELEMENTWISE_CONST_HEAD_COUNT = True
_QK_NORM_ROPE_ROWS_PER_GROUP = 2
_QK_NORM_ROPE_DEFER_SECONDARY_LOADS = True  # keep in step with kernels/qk_norm_rope.py DEFAULT_DEFER_SECONDARY_LOADS
_QK_NORM_ROPE_TILE_ROWS = 16  # keep in step with kernels/qk_norm_rope_tma.py DEFAULT_TILE_ROWS
_QK_NORM_ROPE_STAGES = 2  # keep in step with kernels/qk_norm_rope_tma.py DEFAULT_STAGES
_QUANTIZE_MXFP8_THREADS = 256  # keep in step with kernels/quantize_mxfp8.py DEFAULT_THREADS_PER_CTA (>= the 64 burst lanes of a 1 KiB SF tile)
_QUANTIZE_FP4_THREADS = 256  # keep in step with kernels/quantize_fp4.py DEFAULT_THREADS_PER_CTA (>= the 128 burst lanes of a 2 KiB nvfp4 SF tile at d=256)
"""Smallest GEMM ``TILE_N`` the block intends to support.

Every ``qkvg`` block boundary must be a multiple of this so no output tile
straddles two blocks -- see the alignment invariant above.
"""


class ProjBlock(IntEnum):
    """Which of stage (1)'s outputs an N column belongs to.

    The values are the block ORDER along N and are part of the layout contract:
    ``qkvg_offsets`` returns them in this order and ``dQKVG`` in the backward
    reuses it, so one wgrad GEMM produces ``dW_qkvg`` in the layout the forward
    consumes.

    ``INDEX`` (appended) is the FIFTH band, present ONLY on a geometry that
    declares it (``GatedAttentionBlockGeometry.qsa.index_band``): the sparse
    attention indexer's projection -- its query heads then its single raw key
    head, ``index_head_dim`` columns each -- below V.  Every geometry without it
    keeps exactly the four bands, so its layout tuples are what they always
    were.  A consumer that enumerates the bands iterates ``geometry.qkvg_blocks``
    (never ``ProjBlock`` itself) and touches ``INDEX`` only when it addresses
    the indexer columns on purpose: the norm / RoPE / gate / quantize stages and
    the backward never do.
    """

    Q = 0
    GATE = 1
    K = 2
    V = 3
    INDEX = 4


# ---------------------------------------------------------------------------
# 2a. QsaSpec -- block-sparse attention as a DECLARATION attribute
# ---------------------------------------------------------------------------

QSA_BLOCK_SIZE = 4
"""Tokens per selectable KV block: the sparse loader fetches one 4-token block per gather transaction."""
QSA_TOP_K_MAX = 512
"""The most blocks a query's list may carry (a 2048-token budget at block size 4): the index staging is sized at it."""
QSA_TOP_K_ALIGN = 4
"""The per-query id list is copied in 16-byte units, so ``top_k`` int32 ids must be a multiple of 4."""
# What the sparse core SERVES (head dim, block size, the top_k range, the GQA group it packs on its N tile, the dtypes,
# the arms it carries) is read off the sparse adapter's capabilities record (``cudnn.sdpa.fwd.sparse_gqa_sm107
# .SPARSE_CAPABILITIES`` through ``_sparse_record()``), never transcribed here: the adapter's ``check_support`` enforces
# that record, and two copies of one fact drift.  The three constants above are the API's own contract (the list's
# unit, width and alignment); a declaration also has to satisfy the record (``_check_qsa_geometry_against_record``).

# Two more DECLARATION ATTRIBUTES of the serving surface sit beside QsaSpec, on ``GatedAttentionBlockFwd`` itself, as
# compile-key fields that never enter a knob dataclass or an autotuner: ``paged_kv_page_size`` (landed: it changes the
# INPUT CONTRACT -- the post-RoPE K / V are written through into paged pools at a slot mapping, so ``execute`` takes
# pools and a slot mapping it otherwise refuses) and ``kv_cache_dtype`` (forthcoming with the e4m3 pools: a NUMERICS
# change -- fp8 rounding of the cached K / V -- so never a knob either).
PAGED_KV_PAGE_ALIGN = 16
"""``paged_kv_page_size`` must be a positive multiple of this: the serving stacks allocate pages in multiples of 16 tokens,
and 4 | 16 keeps every 4-token sparse block inside one page (a block never straddles a page boundary)."""

# The SECOND serving attribute: where stage (4) reads K / V from.  ``"self"`` (the default) is the block as it always was --
# attention over the step's own K / V (the slab bands or the compact buffers).  ``"paged"`` is the EXTEND form: the step's
# tokens are written through into the pools first (the write-through above, which it therefore requires) and the SDPA reads
# every sequence's K / V -- its prefix AND the step's new tokens -- from the POOLS through a block table and the per-sequence
# logical KV lengths, on the bottom-right causal diagonal.  A declaration attribute (it changes the input contract: ``execute``
# then REQUIRES ``block_table`` / ``kv_lens`` and the pools' layout is compiled in from the samples), never a knob.
KV_SOURCE_SELF = "self"
"""``kv_source``: stage (4) attends over the block's own K / V (the default; the dense and the packed block as before)."""
KV_SOURCE_PAGED = "paged"
"""``kv_source``: the EXTEND form -- stage (4) reads K / V from the paged pools via ``execute(block_table=, kv_lens=)``."""
_KV_SOURCES = (KV_SOURCE_SELF, KV_SOURCE_PAGED)

# The THIRD serving attribute: the POOLS' dtype (``kv_cache_dtype``; appended 2026-10-14).  ``None`` = the activation dtype -- the
# pools as before, every row written bitwise.  ``torch.float8_e4m3fn`` = the serving stacks' ``--kv-cache-dtype fp8`` recipe: the
# cache holds e4m3 CODES under ONE fp32 per-tensor DEscale per operand (``execute(k_scale=, v_scale=)``: value = code x scale) and a
# bf16 Q attends over the dequantized rows.  NUMERICS-CHANGING (the cached K / V are rounded to e4m3), so an attribute, never a
# knob: it changes the write-through (the rows are quantized on the way in, ``code = sat_e4m3(row x (1 / scale))``) and, on the
# block-sparse decode mode, stage (4) (the listed rows are gathered and cast back to bf16 by ``kernels/qsa_kv_upcast.py`` and the
# bf16 d256 DECODE tile runs over that scratch; the sparse core is never taught fp8).  bf16 Q over an e4m3 pool ONLY: an fp8 Q is
# a different function (the per-tensor FP8 SDPA rows take e4m3 Q / K / V together) and is not a form of this block.
KV_CACHE_E4M3 = torch.float8_e4m3fn
INDEX_CACHE_E4M3 = torch.float8_e4m3fn  # the fp8 INDEXER cache's element type (QsaSpec.index_cache_dtype): plain e4m3, no scales
QSA_INDEX_LIST_FORM_SHARED = "shared"
QSA_INDEX_LIST_FORM_PER_ROW = "per_row"
QSA_INDEX_LIST_FORMS = (None, QSA_INDEX_LIST_FORM_SHARED, QSA_INDEX_LIST_FORM_PER_ROW)
"""The vocabulary of ``QsaSpec.index_list_form`` (the block-sparse decode mode's list form): ``None`` = the ``S_q`` rule (``"shared"``
under caller lists at ``S_q <= 4``, ``"per_row"`` past it and under the in-block indexer over the pools), ``"shared"`` = one list per
sequence anchored at the step-0 position, ``"per_row"`` = one list per row at the row's own position."""
"""``kv_cache_dtype``: e4m3 pools with per-tensor fp32 ``k_scale`` / ``v_scale`` at execute (the serving stacks' fp8 KV cache)."""
_UPCAST_ROWS = (
    2176  # rows per (token, KV head) item of the gathered-cast scratch: 17 x the decode tile's 128-row KV tile (qsa_kv_upcast.ROWS, pinned at check_support)
)
_UPCAST_D = 256  # the scratch row width (qsa_kv_upcast.D, pinned at check_support)
_UPCAST_MAX_S_Q = 4  # the cast serves the SHARED list only: one list per sequence, S_q <= 4 (qsa_kv_upcast.MTP_MAX_S_Q, pinned at check_support)
SERVING_PAGE_SIZES = (16, 32, 64, 128)
"""The page sizes the serving forms are VALIDATED at (the kernel block sizes the serving stacks allocate); the attribute itself
takes any positive multiple of :data:`PAGED_KV_PAGE_ALIGN` and the SDPA adapter's own page rule still applies."""
SERVING_POOL_LAYOUTS = ("hnd", "nhd", "unified")
"""The three pool stride patterns the serving forms are validated on: HND compact ``(H*P*D, P*D, D, 1)``, NHD by strides
``(P*H*D, D, H*D, 1)``, and a unified ``[num_pages, H_kv, P, 2D]`` content row (K at ``+0``, V at ``+D``) bound as its two halves."""

DECODE_CLAIMED_MAX_S_Q = 4
"""The CLAIMED decode envelope: ``1 <= S_q <= 4`` (one new token plus up to three speculative ones) on the Rubin d256 decode tile
PACKED (the whole GQA group in one tile) -- the measured cells (``DecodeRoute.claimed``).  Every other served cell is routed and
declared (``DecodeRoute.tile`` / ``pack_gqa`` / ``kv_streams``) and unmeasured until its perf cell."""

DECODE_CLAIMED_MAX_N_Q = 32
"""The widest decode-tile width inside the CLAIMED envelope: the 16- and 32-column tiles are the measured ones; the 64-column tile
(``config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES[-1]``, routed since 2026-10-15 for a packed step of 33 .. 64 rows -- 24/2 at ``S_q``
3 .. 5, 32/2 at ``S_q`` 3 / 4) is compiled, tested and bitwise the two-unit form, but unmeasured on the perf class: a cell on it is
routed and declared, never ``claimed``, until its perf row lands (then this bound moves to 64)."""

DECODE_MAX_ROWS = 32
"""The widest step the DECODE form serves, in tokens per sequence: the 32-column decode tile UNPACKED (one head per CTA, ``S_q`` live
rows) -- the arm every group falls back to past its packed routes (:func:`decode_route`), as before the table-driven route landed (the
same name, the same meaning, the same value: a caller's ``served iff S_q <= DECODE_MAX_ROWS`` holds).  A dense step with more new
tokens per sequence has no kernel over the pools on this part -- a typed decline (the ``decode_rows`` row), the EXTEND form's (packed)
business."""

DECODE_MAX_S_Q = DECODE_MAX_ROWS
"""The served decode envelope in tokens: ``1 <= S_q <= 32`` (== :data:`DECODE_MAX_ROWS`, the unpacked arm's widest step).  The CLAIMED
bound is :data:`DECODE_CLAIMED_MAX_S_Q` (4, the measured cells); an MTP verify step of up to eight tokens rides the one- and two-stream
routes (the decode tile in one unit, the paged prefill tile with the group packed, two token units -- the routes validated to ``S_q =
8``); past a group's packed routes the step rides the decode tile UNPACKED (``group`` KV streams: declared, unmeasured) up to 32 tokens."""

DECODE_PREFILL_TILE_ROWS = 128
"""The packed rows ``S_q x (H_q / H_kv)`` one Q tile of the paged d256 PREFILL tile holds (``CfgD256.TILE_M``, pinned by the MTP route
tests): the bound of the DECODE form's prefill route (one Q tile per sequence and KV head, the KV streamed once).  Past it a step rides
the decode tile's token-unit arm or its unpacked arm (:func:`decode_route`) -- never a decline by itself."""

DECODE_PREFILL_ROUTE_MIN_CTAS = 64
"""The fill the DECODE form's prefill route needs before the default policy takes it: ``batch x H_kv`` CTAs.  The paged d256 prefill
tile runs ONE unsplit CTA per (sequence, KV head), so a step of fewer CTAs than this leaves most of the part idle while the decode
tile's token-unit arm (two units of the 64-column tile, split over the part by the decode split model) streams the same KV twice but
fills it -- measured on the Rubin performance class (212 SMs) at 32 / 2, ``S_q = 8``, kv 32K: at B 1 the prefill route's step took
330.6 us against the two-unit route's 58.7; at B 32 (64 CTAs) the prefill route's 389.9 beat the two-unit route's 472.6.  So below
this fill the default policy rides the two-unit arm wherever it holds the step (:func:`decode_route`), the prefill tile at or above
it (and wherever no two-unit route holds the step, whatever the batch).  ``decode_route(batch=)`` / ``DecodeRoute.batch`` read the
batch the route was decided at; without a batch (the envelope's ``routes``) the route is the small-batch one."""

DECODE_TILE = "decode_d256_f16"
"""``kernel_template`` of the Rubin d256 decode tile (``sdpa/fwd/kernels/sm107/decode_d256_f16.py``: the swap-AB tile)."""

PREFILL_TILE = "prefill_d256_f16"
"""``kernel_template`` of the paged d256 prefill tile the DECODE form's prefill route runs: the shared SM100 half d256 body
(``sdpa/fwd/kernels/sm100/prefill_d256_f16.py``) on its paged THD arm with PackGQA -- the EXTEND form's kernel."""

DECODE_ROUTE_SINGLE_UNIT = "single_unit"
"""The default route policy: one KV stream per sequence and KV head wherever a kernel offers it AND the step fills the part -- the
decode tile in ONE token unit (``config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES``), else the paged prefill tile with the group packed (a
group dividing its 128 rows) where the step fills :data:`DECODE_PREFILL_ROUTE_MIN_CTAS` CTAs or no two-unit decode route holds it;
below that fill, and for a group the prefill tile cannot pack, the token-unit arm (two streams, split over the part) and the unpacked
arm past it -- the fewest streams available to a step that keeps the part busy."""

DECODE_ROUTE_TOKEN_UNITS = "token_units"
"""The SDPA adapter's own route, verbatim (``config_sm107.decode_d256_q_tile``): the widest decode tile in up to two TOKEN UNITS for a
packed group past the 16-column tile (each unit one more KV stream), then the unpacked tile -- the other arm of the two routes' A/B."""

DECODE_ROUTE_POLICIES = (DECODE_ROUTE_SINGLE_UNIT, DECODE_ROUTE_TOKEN_UNITS)
DEFAULT_DECODE_ROUTE_POLICY = DECODE_ROUTE_SINGLE_UNIT

DECODE_PDL_STAGES = ("qk_norm_rope", "cache_write", "sigmoid_gate", "sdpa")
"""The DECODE form's stages that take the ``use_pdl`` switch (by stage name): the norm + RoPE pass (the unfused chain), the separate
cache write, the sigmoid gate when it is its own launch, and stage (4) on a DECODE-TILE route only (``DecodeRoute.pdl_capable``: the
tile and its split combine carry the griddepcontrol pair; the prefill route's paged d256 prefill tile does not and launches plain --
the link serializes, which PDL's semantics keep correct) or, under a ``QsaSpec`` geometry, the block-sparse core and its split combine
(stage name ``sdpa_sparse``: the ``sdpa`` kind; the core's every role waits before its first dependent read and the core triggers its
dependent at its completion -- no release above its mainloop, the measured placement --, the combine's every form carries the pair --
appended 2026-10-21).  The FROST GEMM stages launch with the attribute by their
own constant and are not switched.  Per declaration ``GatedAttentionBlockFwd.pdl_stages`` names the ones the chain holds and
``use_pdl_effective`` whether every switched stage took it; ``DecodeEnvelope.pdl_stages`` carries this tuple on both decode forms."""

EXTEND_PDL_STAGES = ("qk_norm_rope", "cache_write", "sigmoid_gate")
"""The EXTEND form's stages that take the switch: stage (4), the paged d256 prefill tile, carries no sites and launches plain (the
link serializes; the chain stays correct); ``ExtendEnvelope.pdl_stages`` carries this tuple."""

DECODE_PDL_DEFAULT_CCS: Tuple[Tuple[int, int], ...] = ((10, 7),)
"""The parts on which the AUTO reading of ``use_pdl`` (``None``, the default) resolves True on the dense DECODE form's decode-tile
routes for the bf16 / f16 block (no ``quant``, no e4m3 cache, no ``QsaSpec``): programmatic dependent launch over the block's own chain
-- the norm + RoPE pass, the separate cache write, the decode tile and its split combine.  Measured on a 212-SM cc 10.7 part as a
1.1-1.5 us saving per decode step at B 1 (B 1 x KV 4K: +5.0 %, 32K +4.0 %, 128K +2.5 %; +2.8 % geomean over the B 1 .. 4 cells) and flat
within the control pair where the tile dominates, no cell losing beyond its pair; every ``griddepcontrol`` site of the chain audited
(each role's first dependent read behind its own wait, no global write above a wait, every release after its kernel's wait).  Every
other part keeps False until measured there; an explicit ``use_pdl=True`` / ``False`` is honoured on every part.  The FROST GEMM stages
(the slab GEMM, the out projection, its split-K reducer) launch with the attribute regardless of the switch (their templates' own
constant) and are neither listed by ``pdl_stages`` nor counted by ``use_pdl_effective``."""

# Measured 2026-10-21 (the end-to-end decode gate): Rubin cc 10.7, 212 SMs at the 2376 MHz lock, CUDA-graph replay A/B/A 3 x 100 --
# unfused / fused - 1 = +9.6 .. +14.5 % at B <= 4 and +4.4 .. +10.0 % at B 32 over the 19 decode cells (geomean +11.9 % over B <= 4),
# bitwise the two-stage form on out AND the list everywhere; the control pairs -0.5 .. -1.3 % (a capture-order bias AGAINST the fused
# arm: the band-corrected lever >= +9.1 % on every B <= 4 cell).  Hence cc 10.7 reads True; every other part keeps False until measured.
SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS: Tuple[Tuple[int, int], ...] = ((10, 7),)
"""The parts on which the AUTO reading of ``fuse_prepare`` (``None``, the default) resolves True on the in-block indexer's DECODE step
over the pools (``QsaSpec(index_source="indexer")`` with ``kv_source="paged"``): the ONE prepare launch -- Q / K norm + RoPE, the K / V
write-through, the indexer queries' norm + RoPE into the workspace, the raw key into the paged ring and the block a committed row
completes into the paged compressed-key cache -- in place of the four launches of the unfused twin (the norm + RoPE pass, the K / V
cache write, the pool-1 query norm, the paged compress step), bitwise.  EMPTY until the fused form is measured against the unfused twin
on an exclusive part (the measured-default rule: a default ships with its table, in the change that measures it); until then ``None``
reads False -- the unfused twin -- and an explicit ``fuse_prepare=True`` is served.  Every other form keeps ``None`` = False."""
SPARSE_DECODE_PDL_DEFAULT_CCS: Tuple[Tuple[int, int], ...] = ()
"""The parts on which the AUTO reading of ``use_pdl`` (``None``) resolves True on the BLOCK-SPARSE decode mode (``kv_source="paged"`` on a
dense ``[B, S_q]`` ``QsaSpec`` declaration over bf16 / f16 pools): the sparse core and its split combine carry the griddepcontrol sites since
2026-10-21 and an explicit ``use_pdl=True`` is served there (bitwise the plain chain), but the default stays OFF: the measured-default rule
of :data:`DECODE_PDL_DEFAULT_CCS` admits a part only with a measurement meeting the flip bar (>= 1.0 us AND >= 5 % per step at B 1 x 8K
and B 1 x 32K, no loss beyond the control band on the 128-row cells), and the 212-SM cc 10.7 class, measured 2026-10-22 (CUDA-graph
replay, the switched chain vs the plain one in one process), read +2.1 % / 0.80 us and +4.1 % / 1.54 us there (+2.6 % geomean over the
B <= 4 cells, no cell slower): a gain, below the bar -- so cc 10.7 stays OUT and the explicit switch is the way to take it.  EMPTY today."""

CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS = 128
"""The AUTO reading of ``fuse_cache_write_small_m`` (``None``, the default) folds the paged K / V write-through into the norm + RoPE
launch (``kernels/qk_norm_rope_tma.py``'s pool-write arm: one launch fewer, the pools bitwise the separate write) wherever the arm is
served and the step's token rows ``T = B x S_q`` are below this bound OR ``S_q == 1`` (a one-token-per-sequence step, measured to B 128 --
past it the rule is reachable only under an explicit ``fuse_norm_rope=False``, since the C0 lowering hands rows > 128 to the fused fork,
which owns its own write-through); at
``T >= 128`` with ``S_q > 1`` the separate cache-write launch runs (the chain as before).  The rule is the measurement on a 212-SM cc 10.7
part: the arm wins on every B 1 .. 4 cell (+1.1 .. +3.2 % per decode step on the plain chain, +2.4 .. +4.9 % with PDL on; B 1 x KV 4K
32.2 -> 31.4 us), at B 32 x S_q 1 and B 128 x S_q 1 (the PDL + write-through step +6.9 % / +2.5 % against +2.2 % / +1.8 % from PDL alone),
and loses 0.7-2.7 us per step at B 32 .. 128 x S_q 4 (128-512 rows), where the per-row pool write inside the norm + RoPE kernel costs more
than the write kernel it replaces.  Performance-only (the same bytes land in the pools and the slab either way); an explicit ``True`` /
``False`` is honoured wherever the arm is served, and an explicit ``fuse_cache_write=False`` (the separate launch asked for by name) keeps
the AUTO fold off."""


def _resolve_decode_route_policy(policy: Optional[str]) -> str:
    """``None`` = the default policy; a known policy name as is; anything else a typed ``ValueError`` naming the vocabulary."""
    if policy is None:
        return DEFAULT_DECODE_ROUTE_POLICY
    if not isinstance(policy, str) or policy not in DECODE_ROUTE_POLICIES:
        raise ValueError(f"decode_route_policy must be one of {DECODE_ROUTE_POLICIES} or None (= {DEFAULT_DECODE_ROUTE_POLICY!r}), got {policy!r}")
    return policy


@dataclass(frozen=True)
class DecodeRoute:
    """Which SDPA kernel serves a DECODE step of ``s_q`` tokens at a geometry under a route policy -- the Rubin decode tile
    (``sdpa/fwd/kernels/sm107/decode_d256_f16.py``, the swap-AB tile: ``config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES`` names the
    widths it takes in ONE token unit, ``config_sm107.decode_d256_q_tile`` the adapter's token-unit arm past them) or the paged d256
    PREFILL tile with the group packed (the EXTEND form's kernel at the dense step's envelope); ``_Sdpa.compile`` pins the compiled
    ``kernel_template`` against this record.  ``kv_streams`` is the figure the HBM-bound decode step is priced by: how many times the
    KV range of a (sequence, KV head) is streamed -- the token units on the packed decode tile, the group on the unpacked arm (one
    head per CTA), one on the prefill tile and on the single-unit decode tile."""

    s_q: int
    group: int  # h_q // h_kv: the heads a tile packs per token (1 = MHA, unpacked)
    tile: Optional[str]  # DECODE_TILE / PREFILL_TILE (the ``kernel_template`` the adapter compiles), or None: the step is not served over the pools
    n_q: int  # the decode tile's N extent (the packed Q rows of ONE token unit: 16, 32 or 64); 0 on the prefill tile and when not served
    token_units: int  # the decode tile's CTAs per (KV-head group, batch, split): ceil(s_q / (n_q // group)); 0 on the prefill tile / not served
    pack_gqa: bool  # the whole group packed into the tile; False = MHA or the decode tile's unpacked arm
    claimed: bool  # inside the CLAIMED envelope (the PACKED decode tile at s_q <= DECODE_CLAIMED_MAX_S_Q); False = routed, declared, unmeasured
    reason: str = ""  # why the step is not served (``tile`` None); empty when served
    # APPENDED (2026-10-15): the KV streams per (sequence, KV head) -- 1 on a single-unit decode tile and on the prefill tile, the token
    # units on the packed decode tile, the group on the unpacked arm; 0 when not served -- and the policy the route was read under.
    kv_streams: int = 0
    policy: str = ""
    # APPENDED (2026-10-15): the cap the route was read under (``decode_route(max_q_tile=)`` / ``GatedAttentionBlockFwd(decode_q_tile_max=)``):
    # None = the whole single-unit table (16, 32, 64); 32 = the route before the 64-column width landed.  ``_Sdpa`` hands it to the adapter
    # (``SdpaFwdDslSm100(decode_q_tile_max=)``) so the compiled width is the declared one.
    q_tile_max: Optional[int] = None
    # APPENDED (2026-10-15): whether stage (4) takes the programmatic-dependent-launch switch (``GatedAttentionBlockFwd(use_pdl=True)``)
    # on this route -- the decode tile and its split combine carry the griddepcontrol pair (True); the paged d256 prefill tile does not
    # (False: the block launches it plain and the link serializes, correct by PDL's semantics).  ``_Sdpa.pdl_capable`` is the same rule.
    pdl_capable: bool = False
    # APPENDED (2026-10-16): the batch the route was decided at (``decode_route(batch=)``: the block passes its declared batch, the
    # envelope's ``routes`` none) -- the prefill route needs the step to fill the part, ``batch x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS``,
    # so a step past the single-unit table rides two token units of the decode tile below that fill and the paged prefill tile at or
    # above it; ``None`` = no batch given, the small-batch reading.  A readback, not part of the route's identity (``compare=False``):
    # two records of one route read at different batches are equal when they name the same kernel.
    batch: Optional[int] = dataclasses.field(default=None, compare=False)

    @property
    def decode_tile(self) -> bool:
        return self.tile == DECODE_TILE

    @property
    def prefill_tile(self) -> bool:
        return self.tile == PREFILL_TILE

    @property
    def served(self) -> bool:
        return self.tile is not None


def _resolve_decode_q_tile_max(max_q_tile: Optional[int]) -> Optional[int]:
    """``None`` = the whole single-unit table; a width of the table as is; anything else a typed ``ValueError`` naming the table."""
    if max_q_tile is None:
        return None
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import D256_DECODE_SINGLE_UNIT_Q_TILES

    if isinstance(max_q_tile, bool) or not isinstance(max_q_tile, int) or max_q_tile not in D256_DECODE_SINGLE_UNIT_Q_TILES:
        raise ValueError(
            f"decode_q_tile_max must be one of {D256_DECODE_SINGLE_UNIT_Q_TILES} (the Rubin d256 decode tile's single-unit widths) or None, got {max_q_tile!r}"
        )
    return int(max_q_tile)


def decode_route(
    geometry: "GatedAttentionBlockGeometry", s_q: int, policy: Optional[str] = None, max_q_tile: Optional[int] = None, batch: Optional[int] = None
) -> DecodeRoute:
    """The DECODE route of ``s_q`` tokens at ``geometry`` under ``policy`` (``None`` = :data:`DEFAULT_DECODE_ROUTE_POLICY`), read
    here so the block, the envelope and the tests name the kernel BEFORE anything compiles; ``batch`` (appended) the step's sequences,
    which the default policy's prefill route needs to fill the part (``None`` = no batch given: the small-batch reading).

    The SERVED envelope first: ``1 <= s_q <= DECODE_MAX_S_Q`` (32 tokens per sequence: the unpacked arm's widest step), else NOT served
    (``tile`` None, ``reason`` set: a longer dense step over the pools is the EXTEND form's packed business); a head dim other than 256
    has no paged read (the d256 flavor's).  Then ONE rule per policy over the decode tile's single-unit width table
    (``config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES`` = (16, 32, 64), read for the DECODE form's facts -- a PAGED cache under the
    bottom-right CAUSAL mask, so the 64-column width's form predicate admits every packed group; ``max_q_tile`` (appended) caps the
    table: 32 = the route before the 64-column width landed, value for value):

    * ``"single_unit"`` (the default): packed rows within the table ride the narrowest such width in ONE token unit
      (``config_sm107.decode_d256_single_unit_q_tile``: the KV streamed once -- 24/2 at ``S_q`` 3 .. 5 and 32/2 at ``S_q`` 3 / 4 on the
      64-column tile, 36 .. 64 rows); past the table the step rides the paged d256 PREFILL tile with the whole group packed -- one Q
      tile per (sequence, KV head), the KV streamed once -- wherever that body packs the group within its tile
      (``pack_gqa_group_size(group, DECODE_PREFILL_TILE_ROWS)``: a group dividing its 128 rows, ``s_q x group <= 128``; the cc 10.7
      paged THD leg takes whole-group PackGQA only) AND the step fills the part: that tile is one UNSPLIT CTA per (sequence, KV head),
      so it needs ``batch x H_kv >= DECODE_PREFILL_ROUTE_MIN_CTAS`` (64) CTAs -- measured at 32 / 2, ``S_q = 8``, kv 32K on the Rubin
      performance class: 330.6 us at B 1 against 58.7 on two 64-column units, 389.9 at B 32 against the two units' 472.6 -- or no
      two-unit decode route to hold the step; below the fill, and for a group it cannot pack (12 = 24 / 2, 6 = 24 / 4) or a step past
      its 128 rows, the step keeps the adapter's token-unit arm (up to two units of the widest width: two streams, split over the part
      by the decode split model -- 24/2 at ``S_q`` 6 .. 10 and 32/2 at ``S_q`` 5 .. 8 = two units of the 64-column tile) and the
      UNPACKED arm past it (one head per CTA, ``S_q`` rows on the 16- or 32-column tile: ``group`` streams -- 32 / 2 from ``S_q`` 9,
      the arm every group had before the table landed) -- the fewest streams any kernel offers a step that keeps the part busy, never
      more than the ``"token_units"`` route's.
    * ``"token_units"``: the adapter's own route verbatim (``config_sm107.decode_d256_q_tile``: the single-unit table, then the widest
      width in up to two token units for a packed group), then the unpacked arm -- the A/B's other arm.

    MHA (``h_q == h_kv``) stays the unpacked tile (the bit-exact unpacked fold, one stream: 16 columns to 16 tokens, 32 past).  A cap
    below the width an unpacked step needs (``decode_q_tile_max=16`` at 17 .. 32 tokens) is NOT served either -- the cap is honoured,
    never widened, and the reason names it.  ``claimed`` marks the measured cells: the PACKED decode tile at ``s_q <=
    DECODE_CLAIMED_MAX_S_Q`` on a width up to :data:`DECODE_CLAIMED_MAX_N_Q`."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import decode_d256_q_units, pack_gqa_group_size
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import decode_d256_q_tile, decode_d256_single_unit_q_tile

    policy = _resolve_decode_route_policy(policy)
    cap = _resolve_decode_q_tile_max(max_q_tile)
    if batch is not None:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError(f"decode_route: batch must be >= 1 (the step's sequences) or None (no batch: the small-batch reading), got {batch!r}")
        batch = int(batch)
    g = int(geometry.h_q) // int(geometry.h_kv)
    s_q = int(s_q)
    rows = s_q * g
    # The DECODE form's facts for the width table: the pools (paged) under the bottom-right causal mask (masked) -- the constructor
    # refuses the form without them, so the 64-column width's predicate turns on the packing alone.
    form = dict(paged=True, masked=True, max_q_tile=cap)

    def route(tile, n_q, units, packed, streams, reason=""):
        return DecodeRoute(
            s_q=s_q,
            group=g,
            tile=tile,
            n_q=int(n_q),
            token_units=int(units),
            pack_gqa=bool(packed),
            claimed=(tile == DECODE_TILE) and bool(packed) and s_q <= DECODE_CLAIMED_MAX_S_Q and int(n_q) <= DECODE_CLAIMED_MAX_N_Q,
            reason=reason,
            kv_streams=int(streams),
            policy=policy,
            q_tile_max=cap,
            pdl_capable=(tile == DECODE_TILE),
            batch=batch,
        )

    if int(geometry.d_head) != 256:
        return route(None, 0, 0, False, 0, f"the paged read rides the Rubin d256 SDPA flavor; got d_head={geometry.d_head}")
    if s_q < 1 or g < 1:
        return route(None, 0, 0, False, 0, f"a dense step needs S_q >= 1 and a whole GQA group, got S_q={s_q} at group {g}")
    if s_q > DECODE_MAX_S_Q:
        return route(None, 0, 0, False, 0, _decode_rows_decline(s_q))
    if g == 1:
        # MHA: the unpacked fold on the decode tile (S_q <= 32 rows fit its 16- or 32-column width), one stream
        n_q = decode_d256_single_unit_q_tile(s_q, 1, **form)
        if n_q:
            return route(DECODE_TILE, n_q, 1, False, 1)
        return route(None, 0, 0, False, 0, _decode_cap_decline(s_q, g, cap))
    n_q = decode_d256_single_unit_q_tile(s_q, g, **form)
    if n_q:
        return route(DECODE_TILE, n_q, 1, True, 1)
    if policy == DECODE_ROUTE_SINGLE_UNIT and rows <= DECODE_PREFILL_TILE_ROWS and pack_gqa_group_size(g, DECODE_PREFILL_TILE_ROWS) == g:
        # past the table: the paged prefill tile with the whole group packed (one Q tile per sequence and KV head; rows <= 128) -- ONE
        # unsplit CTA per (sequence, KV head), so only where the step fills the part (DECODE_PREFILL_ROUTE_MIN_CTAS CTAs) or where no
        # two-unit decode route holds it; below the fill the token-unit arm streams the KV twice but splits it over the part
        filled = batch is not None and batch * int(geometry.h_kv) >= DECODE_PREFILL_ROUTE_MIN_CTAS
        if filled or not decode_d256_q_tile(s_q, g, **form):
            return route(PREFILL_TILE, 0, 0, True, 1)
    # the adapter's token-unit arm (the widest width in up to two units), then its unpacked arm (one head per CTA, one unit: the
    # fallback every group had before the table landed, up to 32 tokens)
    n_q = decode_d256_q_tile(s_q, g, **form)
    if n_q:
        units = decode_d256_q_units(s_q, g, n_q)
        return route(DECODE_TILE, n_q, units, True, units)
    n_q = decode_d256_q_tile(s_q, 1, **form)
    if n_q:
        return route(DECODE_TILE, n_q, 1, False, g)
    # reachable under a cap only (decode_q_tile_max=16 holds the table below the 32-column tile a step of 17 .. 32 tokens needs
    # unpacked): the cap is honoured, never widened -- the honest last word names it
    return route(None, 0, 0, False, 0, _decode_cap_decline(s_q, g, cap))


_DECODE_ROWS_DECLINE = (
    f"kv_source='paged' on a dense [B, S_q] declaration (the DECODE form) serves S_q <= {DECODE_MAX_ROWS} new tokens per sequence: the Rubin "
    "d256 decode tile in one token unit, the paged d256 prefill tile with the group packed or two token units of the decode tile (an MTP "
    f"verify step), then the decode tile UNPACKED (one head per CTA) up to {DECODE_MAX_ROWS} rows; a longer dense step over the pools is the "
    "EXTEND form's PACKED (thd=True) business (a QsaSpec geometry has no such cap: the sparse core runs one work item per query row)"
)
"""The ``decode_rows`` row's message (:data:`_EXTEND_DECLINES`): the one text the declaration raises, the envelope publishes and the
route's record carries (with the step appended) for a dense step past :data:`DECODE_MAX_ROWS` tokens."""


def _decode_rows_decline(s_q: int) -> str:
    return f"{_DECODE_ROWS_DECLINE} (got S_q={s_q})"


def _decode_cap_decline(s_q: int, group: int, cap: Optional[int]) -> str:
    return (
        f"kv_source='paged' on a dense [B, S_q] declaration (the DECODE form): decode_q_tile_max={cap} holds the Rubin d256 decode tile's "
        f"width table below the {16 if s_q <= 16 else 32} columns an UNPACKED step of S_q={s_q} tokens needs, and the packed routes of group "
        f"{group} are past the capped table too; declare decode_q_tile_max=None (the whole table) or a width of at least that, or a shorter "
        "step -- the cap is honoured, never widened"
    )


def decode_split_choice(route: DecodeRoute, *, batch: int, h_kv: int, max_kv_len: int, sm_count: int, gated: bool, h_q: Optional[int] = None) -> int:
    """The split-KV the DECODE form lowers with when the declaration leaves ``decode_split_kv=None``: the SDPA heuristics' decode
    split model (``cudnn.sdpa.fwd.heuristics.choose_decode_tile_split_kv`` -- an under-full launch splits until its K / V streams
    saturate HBM, a launch past that never splits, the eager caller's second launch counted against the saving) over
    ``batch x KV-head groups x token units`` CTAs (``batch x Q heads x token units`` on the unpacked arm: one head per CTA)
    and the declared ``max_kv_len`` in 128-key tiles at this part's SM count, the route's token units named so the model counts
    ONE KV stream per split for the units of a (sequence, KV head) (they stream the same range); floored at 2 under ``gated``
    (the gate rides the split COMBINE -- the tile has no gate seams); 1 for a route that is not served (declined at declaration)
    and when ``sm_count`` is 0 (a non-CUDA declaration, declined at ``check_support`` anyway)."""
    if not route.decode_tile or int(sm_count) <= 0:
        return 1
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import choose_decode_tile_split_kv

    heads = int(h_kv) if (route.pack_gqa or h_q is None) else int(h_q)
    units = int(batch) * heads * int(route.token_units)
    kv_tiles = -(-int(max_kv_len) // 128)
    # The combine's grid (B, H_q, S_q) over the REAL Q heads (the group x h_kv when the caller passes no h_q): it decides which
    # combine kernel the split launch takes (the decode-grid form on a small grid, flat in the split count) and so the model's
    # price for the combine -- the same input the SDPA heuristics hand the model for the standalone graph.
    real_heads = int(h_q) if h_q is not None else int(route.group) * int(h_kv)
    split = int(
        choose_decode_tile_split_kv(
            units=units,
            kv_tiles=kv_tiles,
            sm_count=int(sm_count),
            q_tile=route.n_q,
            combine_grid=(int(batch), real_heads, int(route.s_q)),
            token_units=int(route.token_units),
        )
    )
    return max(split, 2) if gated else split


def sparse_decode_split_choice(
    *, batch: int, h_kv: int, s_q: int, top_k: int, sm_count: int, tile_form: str = "tcgen05", sub_tiles_per_cta: Optional[int] = None, group: int = 1
) -> int:
    """The split-KV the block-sparse DECODE mode lowers with when the declaration leaves ``decode_split_kv=None`` -- on the TCGEN05 core
    (``tile_form="tcgen05"``, the rule below; the default) or on the LEAN tile (``tile_form="lean"``, appended: the LEAN split
    ``ceil(n_sub / k)`` of ``heuristics.sparse_lean_splits`` -- 65 at ``top_k`` 512 and k 1 -- where k is ``sub_tiles_per_cta`` when pinned,
    else the default of ``heuristics.sparse_lean_sub_tiles_per_cta`` (1 today; ``group`` = H_q / H_kv feeds its fill model); the lean
    tile always splits and the combine reads its slab as a ``split``-way split, the same contract).  The tcgen05 core's rule: the sparse core
    runs ONE work item per (sequence, KV head, query row) -- ``B x H_kv x S_q`` items of at most ``ceil((top_k + 1) / 32)`` 32-row
    tiles each -- so an under-full launch cuts every item into enough chunks to put one on each SM (the FILL rule, ``ceil(SMs / items)``),
    never more than the item's tile count (a larger split has an empty chunk, a dead work item, on every item -- the adapter's own bound)
    and never more than :data:`SPARSE_DECODE_SPLIT_CAP` = 16 chunks (MEASURED with the pinned combine forms -- Rubin cc 10.7, 212 SMs at
    the 2376 MHz lock, CUDA-graph replay, the decode-grid combine pinned per grid by ``sparse_gqa_sm107._sparse_combine_decode_grid``:
    the B 1 x S_kv 32K x S_q 1 step 39.0 us at split 8 -> 38.1 at 16 -> 37.7 at 17, the B 4 x S_kv 32K x S_q 1 step 37.1 -> 36.2 (16)
    -> 36.4 (17), the former combine entry's step turning UP past 16 (its cost grows with the partial count: 3.3 / 4.5 / 6.5 us at
    8 / 16 / 17 at B 1, where the decode-grid kernel stays at 2.8-3.1 us); 16 is within 1 % of the best on both cells, and the adapter's
    bound at ``top_k`` 512 is 17.  The sweep behind the former cap of 8 (B 4 x 32K x 1: 69.2 / 64.3 / 59.9 / 59.8 / 65.0 us at split
    1 / 2 / 4 / 8 / 17) ran the combine module's OWN forms, the entry compiled without a form -- its knee was that combine's growth
    (2.9 us at 4, 12.2 us at 17); the pinned forms moved the knee and the cap moved with the measurement, never ahead of it), 1 for a
    non-CUDA reading (``sm_count`` 0, declined at ``check_support`` anyway).

    THE WAVE RULE (2026-10-21): where the fill rule's pick overflows ONE wave of SMs (``B x H_kv x S_q x split > SMs``) the launch pays
    its tail wave in full -- a CTA streams its chunk serially, so 256 equal items on 212 SMs cost two item durations, not 1.2 (the
    block's end-to-end census on a Rubin cc 10.7 part, 212 SMs at the 2376 MHz lock, CUDA-graph replay, the 24 / 2 geometry at
    ``top_k`` 512: the T 128 step (B 32 x S_q 4) runs its 256 unsplit 17-tile items in two waves, 64.7 / 69.0 / 70.9 us at KV 8K / 32K /
    128K; B 32 x S_q 1 at the fill rule's 4 runs 256 chunk-items of 5 / 4 / 4 / 4 tiles in 1.2 waves, 19.2-19.3 us; B 4 x S_q 4 at its
    7 runs 224 chunk-items of 3 / 3 / 3 / 2 / 2 / 2 / 2 tiles in 1.06 waves, 13.3-14.7 us) -- so the split is then the one in
    ``[1, min(ceil((top_k + 1) / 32), SPARSE_DECODE_SPLIT_CAP)]`` that minimizes
    ``ceil(items x split / SMs) x (ceil(tiles / split) + SPARSE_DECODE_SPLIT_ITEM_COST_TILES)`` (:func:`sparse_decode_split_rounds`: the
    waves the launch takes, times the tiles a chunk streams plus the chunk-item's fixed cost in tile units -- its Q and ids loads, the
    KV ring's first fill and the epilogue, about one tile's streaming time: the dense decode model's per-CTA term), ties to the SMALLER
    split (fewer partials for the combine, fewer item ramps).  What falls out on the measured cells: T 128 1 -> 3 (768 chunk-items of
    6 / 6 / 5 tiles: 4 waves x 7 = 28 against 2 x 18 = 36; the bare waves x tiles count would pick 9 -- 2304 two-tile items in 11 waves
    -- which the item term rules out), B 32 x S_q 1 4 -> 3 (192 chunk-items, ONE wave of 6 / 6 / 5: 7 against 2 x 6), B 4 x S_q 4 7 -> 6
    (192 chunk-items, one wave of 3 / 3 / 3 / 3 / 3 / 2: 4 against 2 x 4); the rule can also UNDO a split (150 items: the fill rule's 2 =
    300 chunk-items in two waves, 20; unsplit one wave of 17, 18).  Every launch the fill rule keeps in one wave -- every B <= 4 x S_q 1
    cell and B 1 x S_q 4, the measured 16s -- is untouched, bitwise.  The combine form a moved cell takes is the pinned one
    (``sparse_gqa_sm107._sparse_combine_decode_grid``): the lean group walk at B 32 x 24 heads at split 3 as at 4, the combine module's
    own forms at B 4 x S_q 4 at split 6 as at 7.  A MODEL until the end-to-end gate measures the moved cells against their former split
    (``decode_split_kv=`` pins either).

    A PERFORMANCE choice only (the fp32 partials ride the workspace, the combine is one more launch): the same function under every
    value, so a caller may pin any split in ``[1, ceil((top_k + 1) / 32)]`` (the cap binds the DEFAULT only)."""
    if tile_form == "lean":
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import lean_sparse_split
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import sparse_lean_sub_tiles_per_cta

        k = (
            int(sub_tiles_per_cta)
            if sub_tiles_per_cta is not None
            else sparse_lean_sub_tiles_per_cta(batch=batch, h_kv=h_kv, s_q=s_q, group=group, top_k=top_k, sm_count=sm_count)
        )
        return int(lean_sparse_split(int(top_k), k))
    max_tiles = -(-(int(top_k) + 1) // 32)
    if int(sm_count) <= 0:
        return 1
    sms = int(sm_count)
    units = max(int(batch) * int(h_kv) * int(s_q), 1)
    fill = max(1, min(max_tiles, SPARSE_DECODE_SPLIT_CAP, -(-sms // units)))
    if units * fill <= sms:
        return fill  # ONE wave: the fill rule as measured, bitwise
    # a SECOND wave: the split that costs the fewest tile rounds over the launch's waves; ties to the smaller split
    cap = max(1, min(max_tiles, SPARSE_DECODE_SPLIT_CAP))
    return min(range(1, cap + 1), key=lambda s: (sparse_decode_split_rounds(units=units, split=s, max_tiles=max_tiles, sm_count=sms), s))


def sparse_decode_split_rounds(*, units: int, split: int, max_tiles: int, sm_count: int) -> int:
    """The wave rule's price of ``split`` (:func:`sparse_decode_split_choice`), in 32-row-tile units (MODEL): ``ceil(units x split /
    sm_count)`` waves of SMs -- a CTA streams its chunk serially, so a launch past one wave pays every wave in full --, each streaming
    ``ceil(max_tiles / split)`` tiles plus :data:`SPARSE_DECODE_SPLIT_ITEM_COST_TILES` per chunk-item."""
    waves = -(-(int(units) * int(split)) // int(sm_count))
    return waves * (-(-int(max_tiles) // int(split)) + SPARSE_DECODE_SPLIT_ITEM_COST_TILES)


SPARSE_DECODE_SPLIT_CAP = (
    16  # the fill rule's cap on the DEFAULT split of the block-sparse decode mode: the B 1 / B 4 x 32K x S_q 1 sweeps above, taken with
    # the pinned decode-grid combine on a 212-SM cc 10.7 part (the former cap of 8 came from the combine module's own forms, whose cost
    # grew with the partial count); a pinned split may exceed it, up to the item's tile count; the wave rule's candidates stop at it too
)
SPARSE_DECODE_SPLIT_ITEM_COST_TILES = (
    1  # the wave rule's fixed cost per chunk-item in 32-row-tile units (MODEL): the item's Q and ids loads, the KV ring's first fill and
    # the epilogue -- about one tile's streaming time (the sparse core's own B = 1 calibration: ~2900-3400 clocks per K + V tile pair) --
    # the term that keeps the rule from cutting a 17-tile item into two-tile chunks (T 128: split 9's 2304 chunk-items in 11 waves
    # against split 3's 768 in 4); the gate's A/B of the moved cells against their former split is what refits it
)


def _qsa_shared_list_form(qsa, s_q: int) -> bool:
    """The block-sparse decode mode's LIST FORM over the pools: ``QsaSpec.index_list_form`` as declared, else derived from the step's rows under
    caller lists -- ONE list per sequence (the shared step-0 list) at ``s_q <=`` the record's shared-list cap, one per row past it and under the
    in-block indexer.  The ONE derivation the constructor (:meth:`GatedAttentionBlockFwd._qsa_shared_list_derived`) and the static envelope
    (:meth:`DecodeEnvelope.sparse_tile`) read, so the two readings of the route cannot drift."""
    if qsa.index_list_form is not None:
        return qsa.index_list_form == QSA_INDEX_LIST_FORM_SHARED
    return qsa.index_source != "indexer" and int(s_q) <= int(_sparse_record().index_shared_list_max_tokens)


@dataclass(frozen=True)
class DecodeEnvelope:
    """What :meth:`GatedAttentionBlockFwd.serving_envelope` says about the DECODE form (``kv_source="paged"`` on a dense
    ``[B, S_q]`` declaration) for one ``(geometry, dtype, device)`` -- the shape of :class:`ExtendEnvelope`, plus the route per
    ``S_q`` (:func:`decode_route`: the kernel, its packing and token units, whether the cell is inside the CLAIMED envelope
    ``S_q <= claimed_max_s_q``) and the widest dense step served over the pools (``max_rows`` = ``max_s_q`` = 32 tokens per
    sequence, the decode tile unpacked; beyond it a typed decline).  ``declines``
    maps every typed decline the form raises AT DECLARATION to its message -- the rows of the ONE table the constructor reads
    (:data:`_EXTEND_DECLINES`) that a dense declaration can hit.

    ``sparse_core`` (appended): on a :class:`QsaSpec` geometry stage (4) is the index-list SPARSE core reading the pools -- the
    block-sparse DECODE mode (one list per sequence at ``S_q <= 4``, one list per row past it, any ``S_q``: no row cap, the LSE
    served); ``routes`` is then empty and ``max_s_q`` / ``max_rows`` keep the DENSE tile's facts, which do not apply."""

    served: bool
    reason: Optional[str]
    requires: Tuple[str, ...]
    dtypes: Tuple[torch.dtype, ...]
    page_sizes: Tuple[int, ...]
    pool_layouts: Tuple[str, ...]
    declines: Dict[str, str]
    max_s_q: int
    max_rows: int
    routes: Tuple[
        DecodeRoute, ...
    ]  # S_q = 1 .. max_s_q + 1 at this geometry's group under ``policy`` (the served envelope and its first decline; empty under QsaSpec)
    inference_only: bool = True
    writes_lse: bool = False

    def route(self, s_q: int, batch: Optional[int] = None) -> DecodeRoute:
        """The route of an ``s_q``-token step at this geometry under this record's ``policy`` (any ``s_q >= 1``; ``routes`` holds
        1 .. ``max_s_q`` + 1) -- the DENSE form's rule; under ``sparse_core`` the step runs the sparse core whatever this says.
        ``batch`` (appended) is the step's sequences: the default policy's prefill route needs the step to fill the part
        (``DECODE_PREFILL_ROUTE_MIN_CTAS``), and ``routes`` is the reading without one (the small-batch route)."""
        return decode_route(self._geometry, s_q, policy=self.policy or None, max_q_tile=self.q_tile_max, batch=batch)

    _geometry: Optional["GatedAttentionBlockGeometry"] = None
    sparse_core: bool = False  # appended: stage (4) is the index-list sparse core over the pools (a QsaSpec geometry)
    # appended (2026-10-14): the POOL dtypes this form serves at this geometry / activation dtype -- the activation dtype always; plus
    # torch.float8_e4m3fn (``kv_cache_dtype``: the gathered cast + the decode tile, S_q <= 4, no LSE) on a QsaSpec geometry with bf16
    # activations.  An e4m3 pool on a DENSE geometry is a typed decline (the decode tile reads bf16 / f16 pools).
    kv_cache_dtypes: Tuple[torch.dtype, ...] = ()
    # appended (2026-10-15): the route policy ``routes`` / ``route()`` were read under (``serving_envelope(decode_route_policy=)``) and
    # the CLAIMED bound (``DECODE_CLAIMED_MAX_S_Q``; ``max_s_q`` is the DECLARED one).
    policy: str = ""
    claimed_max_s_q: int = 0
    # appended (2026-10-15): the width cap ``routes`` / ``route()`` were read under (``serving_envelope(decode_q_tile_max=)``; None = the whole table).
    q_tile_max: Optional[int] = None
    # appended (2026-10-15): the stage kinds that take the ``use_pdl`` switch on this form (:data:`DECODE_PDL_STAGES`; stage (4) on a
    # decode-tile route only -- ``DecodeRoute.pdl_capable`` per route -- or, under ``sparse_core``, the block-sparse core + its split
    # combine, since 2026-10-21; the ``sdpa`` kind names the sparse stage ``sdpa_sparse`` too).
    pdl_stages: Tuple[str, ...] = ()
    # appended (2026-10-19): the AUTO reading of ``use_pdl`` (None) on THIS part for a SERVED form -- True on the parts of
    # :data:`DECODE_PDL_DEFAULT_CCS` when ``served`` (the bf16 / f16 block over pools in the activation dtype), False wherever the form is
    # not served (another part, an unserved dtype, a geometry row, no DSL target); under ``sparse_core`` the parts of
    # :data:`SPARSE_DECODE_PDL_DEFAULT_CCS` (empty until the switched sparse chain is measured: False everywhere today, the explicit
    # ``True`` served).  A FORM-level reading: the per-step one is ``pdl_default and route(s_q, batch).pdl_capable`` (stage (4) takes the
    # switch on a decode-tile route only; a prefill-route step launches plain and resolves False).  An explicit ``use_pdl`` outranks it.
    pdl_default: bool = False
    # appended (the list form, ``QsaSpec.index_list_form``; a QsaSpec geometry only, empty otherwise): the forms the block-sparse decode
    # mode serves at this geometry -- ``("shared", "per_row")`` under caller lists (``"shared"`` at ``S_q <= 4``: the record's shared-list
    # cap, ``requires`` spells it), ``("per_row",)`` under the in-block indexer over the pools -- and what ``index_list_form=None`` resolves to
    # (``"shared"`` under caller lists at ``S_q <= 4``, ``"per_row"`` past it and under the indexer).
    index_list_forms: Tuple[str, ...] = ()
    index_list_form_default: str = ""
    # appended (the in-block indexer's DECODE step over the pools; a QsaSpec geometry only, empty / 0 otherwise) -- the facts a serving
    # stack's dispatch predicate READS instead of re-spelling them (one copy of every rule): the index sources the geometry serves at decode
    # (``("caller", "indexer")`` with bf16 activations and the scorer's head dim, ``("caller",)`` otherwise), the indexer cache dtypes served
    # at decode (bf16 and -- the e4m3 compressed-key cache, served 2026-10-21 -- float8_e4m3fn), the scorer's head dim / KV heads / block size, the ``top_k`` range
    # ``(min, max, align)``, the decode scorer's query-row envelope (``S_q x next_pow2(index_heads) <= index_query_rows_max``), the rows the
    # raw-key ring needs beyond the step's (``cap >= S_q + index_state_rows_extra``), the page sizes the paged scorer addresses, and the AUTO
    # reading of ``fuse_prepare`` on this part for the in-place pipeline (:data:`SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS`; a block declared
    # out of place reads False -- the AUTO reading is never True where an explicit True is declined).
    index_sources: Tuple[str, ...] = ()
    index_cache_dtypes: Tuple[torch.dtype, ...] = ()
    index_head_dims: Tuple[int, ...] = ()
    index_kv_heads: Tuple[int, ...] = ()
    index_block_sizes: Tuple[int, ...] = ()
    index_top_k_range: Tuple[int, int, int] = (0, 0, 0)
    index_query_rows_max: int = 0
    index_state_rows_extra: int = 0
    index_page_sizes: Tuple[int, ...] = ()
    fuse_prepare_default: bool = False
    # appended (the lean block-sparse decode tile): the sparse core's BODIES this part serves on a QsaSpec geometry, by ``kernel_template``
    # (``heuristics.SPARSE_TILE_TCGEN05`` everywhere the form is served; ``SPARSE_TILE_LEAN`` too on cc 10.7, the lean tile's one part), and
    # the AUTO pin's reading on this part (``heuristics.SPARSE_LEAN_TILE_DEFAULT_CCS``: cc 10.7 since the gate measured the lean route, so
    # ``sparse_tile_form="auto"`` follows the placement rule there and resolves the tcgen05 core on every other part; the explicit ``"lean"``
    # is served where the body admits the declaration).  The per-CELL reading is :meth:`sparse_tile` (the route reads the batch -- the CTAs
    # per chunk, B x H_kv -- and the step width through the admissibility, which a static envelope cannot carry in one field).
    sparse_tiles: Tuple[str, ...] = ()
    sparse_tile_lean_default: bool = False
    device_cc: Optional[Tuple[int, int]] = dataclasses.field(default=None, compare=False)  # the part :meth:`sparse_tile` reads the AUTO pin and the rule on

    def sparse_tile(
        self, s_q: int, batch: int, *, sparse_tile_form: str = "auto", decode_split_kv: Optional[int] = None, kv_cache_dtype: Optional[torch.dtype] = None
    ) -> Optional[str]:
        """The sparse core's body (its ``kernel_template``) a ``[batch, s_q]`` block-sparse decode step at this geometry takes on this part
        under ``sparse_tile_form`` (``"auto"`` = the AUTO pin + the placement rule + the admissibility, with a pinned ``decode_split_kv`` kept
        on the body that honours it; ``"lean"`` / ``"tcgen05"`` as named where the body admits the step, else None = a typed decline at
        declaration); None off a QsaSpec geometry.  ``kv_cache_dtype`` (appended) is the POOLS' dtype the step is declared over (None = the
        activations' bf16 / f16): over e4m3 pools stage (4) is the gathered cast + the dense decode tile, NO body of the sparse core, so every
        form reads None there (the constructor declines an explicit body by name).  The same arithmetic the constructor runs (one copy:
        ``heuristics``, the list form through :func:`_qsa_shared_list_form`), so a serving stack's dispatch predicate reads the route instead
        of catching the block's exceptions."""
        if not self.sparse_core or self._geometry is None or not self.served:
            return None
        if kv_cache_dtype is not None and kv_cache_dtype not in (torch.bfloat16, torch.float16):
            return None  # e4m3 pools: the gathered cast + the dense decode tile run, no body of the sparse core
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import sparse_tile_rules as _H

        g = self._geometry
        q = g.qsa
        shared = _qsa_shared_list_form(q, int(s_q))
        reason = _H.sparse_lean_tile_decline(
            s_q=int(s_q), group=g.gqa_ratio, top_k=q.top_k, thd=False, bottom_right=True, list_per_sequence=shared, seq_kv_lens_present=True
        )
        if sparse_tile_form == _H.SPARSE_TILE_FORM_AUTO:
            form = _H.sparse_tile_form_auto(
                batch=int(batch), h_kv=g.h_kv, s_q=int(s_q), top_k=q.top_k, device_cc=self.device_cc, admissible=reason is None, split_pinned=decode_split_kv
            )
        elif sparse_tile_form == _H.SPARSE_TILE_FORM_LEAN:
            if reason is not None or (decode_split_kv is not None and int(decode_split_kv) not in _H.sparse_lean_splits(q.top_k)):
                return None
            form = _H.SPARSE_TILE_FORM_LEAN
        else:
            form = _H.SPARSE_TILE_FORM_TCGEN05
        if form == _H.SPARSE_TILE_FORM_TCGEN05 and decode_split_kv is not None and not 1 <= int(decode_split_kv) <= _H.sparse_tcgen05_max_split(q.top_k):
            return None  # the core's split bound [1, ceil((top_k + 1) / 32)]: outside it the declaration is rejected, so no body is read
        return _H.SPARSE_TILE_LEAN if form == _H.SPARSE_TILE_FORM_LEAN else _H.SPARSE_TILE_TCGEN05


@dataclass(frozen=True)
class ExtendEnvelope:
    """What :meth:`GatedAttentionBlockFwd.serving_envelope` says about the EXTEND form (``kv_source="paged"`` on a packed
    declaration) for one ``(geometry, dtype, device)``.

    ``served`` is True iff a block declared as ``requires`` spells it would pass ``check_support`` on that device for that
    geometry and dtype -- the device / DSL facts (``reason`` names the first one that fails), the dtype (``dtypes``), the head
    dim and the geometry rows below.  ``declines`` maps every typed decline the form raises AT DECLARATION to the message it
    raises with -- the SAME table the constructor reads (:data:`_EXTEND_DECLINES`), so a caller's dispatch predicate and the
    block's declines cannot drift: no decline is reachable at declaration that this record did not predict.  ``page_sizes``
    / ``pool_layouts`` are the validated matrix (every entry an accept cell on the target body).
    """

    served: bool
    reason: Optional[str]
    requires: Tuple[str, ...]
    dtypes: Tuple[torch.dtype, ...]
    page_sizes: Tuple[int, ...]
    pool_layouts: Tuple[str, ...]
    declines: Dict[str, str]
    inference_only: bool = True
    writes_lse: bool = False
    # appended (2026-10-15): the stage kinds that take the ``use_pdl`` switch on this form (:data:`EXTEND_PDL_STAGES`: never stage (4), the
    # paged prefill tile launches plain and the link serializes).
    pdl_stages: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ServingEnvelope:
    """The static serving envelope of the block for one ``(geometry, dtype, device)``: the device / DSL facts and one record per
    serving form.  Read by a serving stack's INIT-time dispatch predicate instead of catching the block's exceptions; derived from
    the same code ``check_support`` runs.  ``forms`` lists the forms this record serves; ``extend`` is the EXTEND form's record,
    ``decode`` the DECODE form's (:class:`DecodeEnvelope`: the route per ``S_q`` included)."""

    device_cc: Optional[Tuple[int, int]]
    cutedsl_ok: bool
    cutedsl_reason: Optional[str]
    forms: Tuple[str, ...]
    extend: ExtendEnvelope
    decode: Optional[DecodeEnvelope] = None


@dataclass(frozen=True)
class QsaSpec:
    """Block-sparse attention (Qwen Sparse Attention) as the restriction of stage (4): every query attends to the keys
    of its SELECTED ``block_size``-token blocks AND to the open tail block of its visible range, under the causal mask.

    A DECLARATION ATTRIBUTE, never a knob: it changes the function -- which keys a query sees, who selects them, the
    ``W_qkvg`` contract -- so a block declared with it is a different plan from the dense block, and no knob value may
    route around it.  Attached as ``GatedAttentionBlockGeometry.qsa``.  The model provenance (Qwen3.8-Flash-Next: a
    2048-token budget, block ratio 4, a 4-head x 128 MQA indexer over one raw key head) lives in comments only; the
    fields are named by op geometry.

    Fields
    ------
    block_size
        Tokens per selectable block.  4 only: the sparse loader fetches one 4-token block per gather transaction;
        another block size has no loader.
    top_k
        Blocks per query the caller's list carries: a multiple of 4 in ``[4, 512]``.  It SIZES the kernel's per-query
        index staging (every ids byte count is ``top_k x 4``, copied in 16-byte units), so a longer list would be
        silently truncated and an unaligned one over-read -- both refused by :meth:`validate`.
    index_source
        ``"caller"``: ``execute(block_ids=)`` carries the selection -- ``[T, top_k]`` int32, per query the ids of its
        selected complete blocks (block ``b`` = tokens ``[4b, 4b + 4)`` of the query's own sequence), the valid prefix
        then ``-1`` padding; ``block_lens`` optional.  ``"indexer"``: the block runs the indexer over its fifth band
        itself (``index_band`` required; bf16 activations): the band's query heads are RMSNormed and rotated per token,
        its raw key head is mean-pooled per complete block, RMSNormed and rotated at the block START, and the scorer
        keeps the ``top_k`` blocks per query -- ``execute`` then takes ``w_iq_norm`` / ``w_ik_norm`` and refuses
        ``block_ids``; the selection, its counts and the compressed keys come out through the optional caller buffers
        ``block_ids_out`` / ``block_lens_out`` / ``index_k_compressed``.  At ``S <= identity_bound`` the block launches
        no scorer: the selection is the identity, a plan-time constant (:class:`_Indexer`).
    index_band
        ``W_qkvg`` carries a FIFTH band, ``ProjBlock.INDEX``, of ``(index_heads + index_kv_heads) * index_head_dim``
        columns below V: the indexer's query heads first, its single raw key head last (the checkpoint's own row
        order, so the indexer weight concatenates unchanged).  Served on the UNFUSED projection (the FROST GEMM writes
        the wider slab; :func:`index_k_raw_view` exposes the raw key); the fused projection fork renders 256-column
        tiles and declines a 640-column band, typed.  At decode row counts (``T = B x S_q <= 128``) the slab GEMM takes the
        small-M route's 128-row swap-AB tile (N 13952 = 109 x 128: the route's overflow branch admits the one N off the 256
        grid whose 128-row tiles divide it), so a band declaration over the pools compiles without the opt-in flag and is
        capture-safe like the four-band one; past 128 rows the band slab keeps the graph route.
    index_heads, index_kv_heads, index_head_dim, index_norm_eps
        The indexer's geometry.  Under ``index_source="caller"`` they SIZE THE BAND ONLY (no kernel reads them); under
        ``"indexer"`` they are the scorer's: ``index_heads`` a head group the indexer scorer packs, ``index_head_dim``
        its head dim, ``index_norm_eps`` the epsilon of both indexer RMSNorms (``cudnn.QSA.selection``
        names the two).  ``index_kv_heads`` is 1 (one raw key head, the checkpoint's own validator pins it);
        ``index_head_dim`` is a multiple of ``QKVG_TILE_ALIGN`` so every indexer head keeps the band tile-aligned.

    index_cache_dtype
        The INDEXER CACHE's element type under ``index_source="indexer"`` (appended): ``None`` -- the activation dtype, bf16:
        the compressed keys and the indexer queries as the compress kernel writes them -- or ``torch.float8_e4m3fn``, the fp8
        indexer cache: both as plain e4m3 of the same RMSNormed + rotated values (no scales), scored through the scorer's
        MXFP8 arm with unit scale factors (``cudnn.QSA.selection``'s e4m3 arm), the caller's
        ``index_k_compressed`` then an e4m3 buffer.  NUMERICS-CHANGING (3 mantissa bits move the weakest selected blocks at
        the k-th boundary: the selection agrees with the bf16 one on 99.0 % of the blocks at S = 4096, 97.4 % at 32768 and
        96.5 % at 131072 on synthetic activations, every flip a rank-boundary flip), so an attribute of ``QsaSpec`` itself, never
        a knob; bf16 stays the default and the reference.  Refused under ``index_source="caller"``: nothing of the block
        reads or writes an indexer cache there.  Measured (2026-10-22/23, one exclusive Rubin cc 10.7 GPU, 212 SMs, A/B/A in one process
        per cell): the e4m3 cache's lever over bf16 is the scorer's bytes -- +10.8 % of the decode step at B 32 x 128K x S_q 1, +1.6 %
        geomean over 13 cells, within +-1 % at B <= 4 (the two B 1 x 8K cells -1.1 / -1.0 %: the e4m3 path's fixed cost at a 0.26 MB
        cache) -- an opt-in for long contexts at batch; the docs carry the table.

    index_list_form
        The LIST FORM of the block-sparse DECODE mode (``kv_source="paged"`` on a dense ``[B, S_q]`` declaration; appended):
        ``"shared"`` -- ONE list per sequence, ``[B, top_k]``, selected at the step-0 position ``kv_lens[b] - S_q`` and
        shared by the step's rows (the MTP form; ``S_q`` at most the sparse core's shared-list cap, 4); ``"per_row"`` -- one
        list per row, ``[B, S_q, top_k]`` / ``[B x S_q, top_k]`` at positions ``kv_lens[b] - S_q + j`` (the serving stacks'
        per-row selection), at ANY ``S_q``; ``None`` (the default) -- ``"shared"`` under caller lists at ``S_q <= 4`` and
        ``"per_row"`` past it (the former derivation), ``"per_row"`` under the in-block indexer over the pools.  A
        DECLARATION ATTRIBUTE (numerics-changing: which keys the speculative rows see), never a knob.  ``"shared"`` under the
        indexer over the pools and under caller lists past four rows are typed declines; on the block's own K / V
        (``kv_source="self"``) every row carries its own list, so ``"shared"`` is refused there by name.

    Fixed semantics that are deliberately NOT fields: the open tail block is always visible, the causal mask is always
    applied, and the id dtype is int32 (one legal value; an int64 list is refused at ``execute``).  A dense route below
    the identity bound (:attr:`identity_bound` visible tokens -- 2051 at the defaults -- below which every query's
    complete blocks fit the list and a full list reproduces dense causal attention) is a legal performance knob ONLY
    under ``index_source="indexer"``, where the block derives the selection itself; under caller lists it would ignore
    the list: numerics-changing, never a knob.
    """

    block_size: int = QSA_BLOCK_SIZE
    top_k: int = QSA_TOP_K_MAX
    index_source: str = "caller"
    index_band: bool = False
    index_heads: int = 4
    index_kv_heads: int = 1
    index_head_dim: int = 128
    index_norm_eps: float = 1e-6
    index_cache_dtype: Optional[torch.dtype] = None  # appended: None = the activation dtype; torch.float8_e4m3fn = the fp8 indexer cache
    index_list_form: Optional[str] = None  # appended: the decode mode's list form -- None (the S_q rule / per row under the indexer), "shared", "per_row"

    @property
    def index_cache_e4m3(self) -> bool:
        """True under the fp8 indexer cache (``index_cache_dtype=torch.float8_e4m3fn``): the scorer reads e4m3 casts of the compressed keys
        and the queries, and ``index_k_compressed`` is an e4m3 buffer."""
        return self.index_cache_dtype == INDEX_CACHE_E4M3

    @property
    def index_band_cols(self) -> int:
        """Columns of the fifth band when declared: ``(index_heads + index_kv_heads) * index_head_dim`` (640 at the defaults)."""
        return (int(self.index_heads) + int(self.index_kv_heads)) * int(self.index_head_dim)

    @property
    def identity_bound(self) -> int:
        """The largest visible-token count at which every query's complete blocks still fit the list (2051 at the
        defaults): ``floor(n / block_size) <= top_k`` for every ``n`` up to it, so a full list reproduces dense causal
        attention exactly; one token more and the oldest complete block of that query is not representable."""
        return int(self.top_k) * int(self.block_size) + int(self.block_size) - 1

    def validate(self) -> None:
        """Raise ``ValueError`` on a declaration no stage can express; each message names what the failure would have
        LOOKED like (the geometry validator's own rule)."""
        if self.block_size != QSA_BLOCK_SIZE:
            raise ValueError(
                f"QsaSpec.block_size must be {QSA_BLOCK_SIZE}: the sparse loader gathers one {QSA_BLOCK_SIZE}-token block per transaction, "
                f"and block_size={self.block_size} has no loader"
            )
        if not (QSA_TOP_K_ALIGN <= self.top_k <= QSA_TOP_K_MAX) or self.top_k % QSA_TOP_K_ALIGN:
            raise ValueError(
                f"QsaSpec.top_k must be a multiple of {QSA_TOP_K_ALIGN} in [{QSA_TOP_K_ALIGN}, {QSA_TOP_K_MAX}], got {self.top_k}: the per-query "
                f"index staging is sized at top_k blocks (at most {QSA_TOP_K_MAX}, the {QSA_TOP_K_MAX * QSA_BLOCK_SIZE}-token budget over "
                f"{QSA_BLOCK_SIZE}-token blocks) and copied in 16-byte units -- a longer list would be silently truncated, an unaligned one over-read"
            )
        if self.index_source not in ("caller", "indexer"):
            raise ValueError(
                f"QsaSpec.index_source must be 'caller' (execute(block_ids=) carries the selection) or 'indexer' (the block runs the indexer), "
                f"got {self.index_source!r}"
            )
        if self.index_source == "indexer" and not self.index_band:
            raise ValueError(
                "QsaSpec.index_source='indexer' requires index_band=True: the indexer projects its queries and keys from the fifth band of "
                "W_qkvg; without the band there is nothing to score"
            )
        if self.index_heads < 1:
            raise ValueError(f"QsaSpec.index_heads must be >= 1 (the indexer's query heads), got {self.index_heads}")
        if self.index_kv_heads != 1:
            raise ValueError(f"QsaSpec.index_kv_heads must be 1 (one raw key head; the checkpoint's own validator pins it), got {self.index_kv_heads}")
        if self.index_head_dim < QKVG_TILE_ALIGN or self.index_head_dim % QKVG_TILE_ALIGN:
            raise ValueError(
                f"QsaSpec.index_head_dim must be a positive multiple of QKVG_TILE_ALIGN={QKVG_TILE_ALIGN} (every indexer head keeps the fifth band "
                f"tile-aligned, else a GEMM output tile straddles the band and its neighbour), got {self.index_head_dim}"
            )
        if not self.index_norm_eps > 0.0:
            raise ValueError(f"QsaSpec.index_norm_eps must be > 0, got {self.index_norm_eps}")
        if self.index_cache_dtype is not None and self.index_cache_dtype not in (torch.bfloat16, INDEX_CACHE_E4M3):
            raise ValueError(
                f"QsaSpec.index_cache_dtype must be None (the activation dtype), torch.bfloat16 or torch.float8_e4m3fn (the fp8 indexer cache: plain "
                f"e4m3 compressed keys and indexer queries through the scorer's MXFP8 arm with unit scales), got {self.index_cache_dtype}"
                + (" -- e5m2 has no scorer arm (a different storage numerics)" if self.index_cache_dtype == torch.float8_e5m2 else "")
            )
        if self.index_cache_dtype is not None and self.index_source != "indexer":
            raise ValueError(
                f"QsaSpec.index_cache_dtype={self.index_cache_dtype} requires index_source='indexer': under caller lists nothing of the block reads or "
                "writes an indexer cache (the compressed keys and the queries are the stack's), so the declaration would be silently meaningless -- "
                "refused by name, whatever the dtype"
            )
        if self.index_list_form not in QSA_INDEX_LIST_FORMS:
            raise ValueError(
                f"QsaSpec.index_list_form must be None (the S_q rule: 'shared' at S_q <= 4 under caller lists, 'per_row' past it and under the "
                f"in-block indexer over the pools), 'shared' (one list per sequence, [B, top_k], anchored at the step-0 position) or 'per_row' "
                f"(one list per row, [B, S_q, top_k]), got {self.index_list_form!r}"
            )


# ---------------------------------------------------------------------------
# 2. Geometry — the block's compile-time contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatedAttentionBlockGeometry:
    """Everything the stages specialize on. Plan-time only, never runtime data.

    Compile keys are plan-time-only (``python/cudnn/AGENTS.md`` Rule 4): every
    field here is derivable from the declaration — shapes, dtypes, flags — so
    one compiled artifact re-binds any batch/sequence.

    Qwen3.5-397B (TP=1) fills this as ``d_model=4096, h_q=32, h_kv=2,
    d_head=256, rope_dim=64`` — GQA 16x, ``scale = 256 ** -0.5 = 1/16``.
    """

    d_model: int
    h_q: int
    h_kv: int
    d_head: int  # d_qk == d_v; the SDPA flavor is picked from this
    rope_dim: int  # leading dims of d_head that rotate; [rope_dim, d_head) pass through

    qk_norm_eps: float = 1e-6
    attn_scale: Optional[float] = None  # None -> d_head ** -0.5

    # Mask. NOT hardcoded causal: ColQwen3.5 is bidirectional, and the mask is a
    # SCHEDULING-REGIME selector here, not just a correctness knob -- under a
    # window the SDPA collapses to a constant ~9 KV tiles while the gate GEMM is
    # fixed, which can invert the partitioning the causal arm wants.
    is_causal: bool = True
    causal_bottom_right: bool = False
    window_left: int = -1
    window_right: int = -1

    # QK-RMSNorm on/off. False: RoPE-only Q/K -- no per-head RMSNorm, so no
    # norm weights (the block takes None in their slots, both directions
    # checked) and no rstd (SavedForBackward.rstd_q/rstd_k are None). The
    # kernels fold the norm out at trace time (presence of the weight tensors,
    # like want_rstd); the dims [rope_dim, d_head) are then a bit-exact copy.
    # ``qk_norm_eps`` stays validated > 0 regardless (D10): the fused fork's
    # own validator re-checks it after eligibility, and relaxing it here would
    # leak a post-eligibility ValueError.
    qk_norm: bool = True

    # APPENDED: block-sparse attention.  None = the dense block.  A QsaSpec changes
    # the FUNCTION (which keys a query sees, who selects them, the W_qkvg contract --
    # a fifth band under `index_band`), so it is a declaration attribute, never a
    # knob, and every layout tuple below grows its fifth entry exactly when the band
    # is declared.  validate() runs QsaSpec's own rows and the cross-field ones.
    qsa: Optional[QsaSpec] = None

    # -- derived scalars ----------------------------------------------------

    @property
    def scale(self) -> float:
        """Softmax scale actually used."""
        return self.attn_scale if self.attn_scale is not None else float(self.d_head) ** -0.5

    @property
    def gqa_ratio(self) -> int:
        """Query heads per KV head."""
        return self.h_q // self.h_kv

    @property
    def n_qkvg(self) -> int:
        """Output width of the fused stage-(1) projection."""
        return sum(self.qkvg_block_widths)

    # -- the N-axis map (see "Stage (1) layout" above) ----------------------

    @property
    def index_band(self) -> bool:
        """``True`` iff ``W_qkvg`` carries the fifth (indexer) band: ``qsa`` declared with ``index_band``."""
        return isinstance(self.qsa, QsaSpec) and bool(self.qsa.index_band)

    @property
    def qkvg_blocks(self) -> tuple[ProjBlock, ...]:
        """The bands this geometry's ``W_qkvg`` has, in N order: ``(Q, GATE, K, V)``, plus ``INDEX`` iff declared.

        Enumerate THIS, never ``ProjBlock`` itself -- the enum carries the fifth member for every geometry, the band
        exists only where it is declared.
        """
        dense = (ProjBlock.Q, ProjBlock.GATE, ProjBlock.K, ProjBlock.V)
        return dense + (ProjBlock.INDEX,) if self.index_band else dense

    @property
    def qkvg_block_widths(self) -> tuple[int, ...]:
        """Column count of each band, in ``qkvg_blocks`` order (four entries; five with the indexer band)."""
        widths = (
            self.h_q * self.d_head,
            self.h_q * self.d_head,
            self.h_kv * self.d_head,
            self.h_kv * self.d_head,
        )
        return widths + (self.qsa.index_band_cols,) if self.index_band else widths

    @property
    def qkvg_offsets(self) -> tuple[int, ...]:
        """Starting column of each band, in ``qkvg_blocks`` order.

        **This ordering is API, append-only forever** — it is what a caller
        concatenates its checkpoint weights into, ONCE at load time (never per
        execute: Rule 1 bans conversions on the hot path).
        """
        offs, acc = [], 0
        for w in self.qkvg_block_widths:
            offs.append(acc)
            acc += w
        return tuple(offs)

    # -- the fully fused FP8 pipeline: Q + K + V width (no GATE band) -----------

    @property
    def n_qkv(self) -> int:
        """The Q + K + V width (no GATE band, no indexer band): the e4m3 bytes
        per token the fused FP8 projection writes across its three COMPACT
        outputs ``q8`` / ``k8`` / ``v8`` (``h_q*d + 2*h_kv*d``).  A width, not a
        slab: since round 2 nothing in the block addresses a ``[T, n_qkv]``
        buffer."""
        w = self.qkvg_block_widths
        return w[ProjBlock.Q] + w[ProjBlock.K] + w[ProjBlock.V]

    @property
    def qkvg_heads(self) -> tuple[int, ...]:
        """Head count of each band, in ``qkvg_blocks`` order (the INDEX band's heads are the indexer's: its query
        heads plus its raw key head, of ``qkvg_head_dims[INDEX]`` columns each, not ``d_head``)."""
        heads = (self.h_q, self.h_q, self.h_kv, self.h_kv)
        return heads + (self.qsa.index_heads + self.qsa.index_kv_heads,) if self.index_band else heads

    @property
    def qkvg_head_dims(self) -> tuple[int, ...]:
        """Per-band head dim, in ``qkvg_blocks`` order: ``d_head`` for Q / GATE / K / V, ``index_head_dim`` for INDEX."""
        dims = (self.d_head,) * 4
        return dims + (self.qsa.index_head_dim,) if self.index_band else dims

    @property
    def index_k_raw_offset(self) -> int:
        """First slab column of the RAW indexer key: the INDEX band's last ``index_kv_heads * index_head_dim`` columns
        (pre-norm, un-rotated -- what a serving cache keeps for the indexer).  ``ValueError`` without the band."""
        if not self.index_band:
            raise ValueError("index_k_raw_offset: this geometry declares no indexer band (GatedAttentionBlockGeometry.qsa.index_band)")
        return self.qkvg_offsets[ProjBlock.INDEX] + self.qsa.index_heads * self.qsa.index_head_dim

    def block_for_column(self, col: int) -> tuple[ProjBlock, int, int]:
        """``col`` in ``[0, N)`` -> ``(block, head, column within head)`` -- the head and its width per band
        (``qkvg_head_dims``: the INDEX band's heads are ``index_head_dim`` wide)."""
        if not 0 <= col < self.n_qkvg:
            raise ValueError(f"column {col} out of range [0, {self.n_qkvg})")
        for block, off, width, hd in zip(self.qkvg_blocks, self.qkvg_offsets, self.qkvg_block_widths, self.qkvg_head_dims):
            if col < off + width:
                local = col - off
                return block, local // hd, local % hd
        raise AssertionError("unreachable: widths sum to n_qkvg")

    def qkvg_tile_plan(self, tile_n: int) -> tuple[ProjBlock, ...]:
        """Destination block of every stage-(1) output tile, at this ``TILE_N``.

        This is what the epilogue's descriptor select decodes at run time, and
        it exists as plain Python so the alignment invariant is checkable
        without a GPU: the plan is well-defined exactly when no tile straddles a
        block boundary, which :meth:`validate` requires.
        """
        if tile_n <= 0 or self.n_qkvg % tile_n != 0:
            raise ValueError(f"TILE_N={tile_n} must be positive and divide N={self.n_qkvg}")
        plan = []
        for t in range(self.n_qkvg // tile_n):
            first = self.block_for_column(t * tile_n)[0]
            last = self.block_for_column((t + 1) * tile_n - 1)[0]
            if first != last:
                raise ValueError(
                    f"TILE_N={tile_n} tile {t} straddles {first.name} and {last.name}; " f"block widths {self.qkvg_block_widths} are not {tile_n}-aligned"
                )
            plan.append(first)
        return tuple(plan)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on any geometry the stages cannot express.

        Never a module-level ``assert`` (contract § 7): anything derived from
        user input raises, so it survives ``python -O`` and names itself. Each
        message says what the failure would have LOOKED like, because a config
        error that reaches a kernel does not announce itself.
        """
        if self.qsa is not None and not isinstance(self.qsa, QsaSpec):
            raise TypeError(f"geometry.qsa must be a QsaSpec or None, got {type(self.qsa).__name__}")
        for label, value in (("d_model", self.d_model), ("h_q", self.h_q), ("h_kv", self.h_kv), ("d_head", self.d_head)):
            if value <= 0:
                raise ValueError(f"{label} must be > 0, got {value}")

        if self.h_q % self.h_kv != 0:
            raise ValueError(f"h_q ({self.h_q}) must be divisible by h_kv ({self.h_kv}) for GQA/MQA broadcast")

        if not 0 <= self.rope_dim <= self.d_head:
            raise ValueError(f"rope_dim must be in [0, d_head={self.d_head}], got {self.rope_dim}")
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even (rotate_half pairs i with i + rope_dim//2), got {self.rope_dim}")

        if not self.qk_norm_eps > 0.0:
            raise ValueError(f"qk_norm_eps must be > 0, got {self.qk_norm_eps}")
        if not self.qk_norm and self.rope_dim == 0:
            raise ValueError("qk_norm=False with rope_dim=0 leaves stage (2)+(3) an identity copy; drop the stage instead")
        if self.attn_scale is not None and not self.attn_scale > 0.0:
            raise ValueError(f"attn_scale must be > 0 when given, got {self.attn_scale}")

        # The stage-(1) alignment invariant. Without it an output tile can
        # straddle two blocks, and the epilogue would have to predicate per
        # column instead of specializing per tile -- which is not a correctness
        # bug today but forecloses the quantization epilogue entirely.
        for block, width in zip(self.qkvg_blocks, self.qkvg_block_widths):
            if width % QKVG_TILE_ALIGN != 0:
                raise ValueError(
                    f"{block.name} block width {width} (= heads * head dim) must be a multiple of QKVG_TILE_ALIGN={QKVG_TILE_ALIGN}, "
                    f"else a GEMM output tile straddles two of Q/GATE/K/V"
                )

        # Mask knobs. -1 means "unbounded on that side"; anything else must be a
        # real, non-negative distance.
        for label, value in (("window_left", self.window_left), ("window_right", self.window_right)):
            if value < -1:
                raise ValueError(f"{label} must be >= 0, or -1 for unbounded; got {value}")
        if self.causal_bottom_right and not self.is_causal:
            raise ValueError("causal_bottom_right=True requires is_causal=True")
        # The SDPA lowers ONE band: window_right widens the causal diagonal, so
        # it is meaningless without a diagonal to widen. Caught here for a
        # message that names the block's own field rather than the adapter's.
        if self.window_right >= 0 and not self.is_causal:
            raise ValueError("window_right requires is_causal=True (it widens the causal diagonal, it does not create one)")

        # Block-sparse attention: the QsaSpec's own rows, then the cross-field ones.
        # The selection IS the sparsity, so a window or a bidirectional mask has no
        # definition under it; the indexer rotates the same leading rope_dim as Q / K.
        if self.qsa is not None:
            self.qsa.validate()
            if self.rope_dim > self.qsa.index_head_dim:
                raise ValueError(
                    f"rope_dim ({self.rope_dim}) must be <= QsaSpec.index_head_dim ({self.qsa.index_head_dim}): the indexer rotates the same "
                    "leading rope_dim of its heads, and a rotation wider than the head has no definition"
                )
            if not self.is_causal:
                raise ValueError(
                    "QsaSpec requires is_causal=True: block-sparse attention is defined under the causal mask (causal AND selected); a "
                    "bidirectional sparse block is not a thing the model computes"
                )
            if self.window_left != -1 or self.window_right != -1:
                raise ValueError(
                    f"QsaSpec has no sliding window (the selection IS the sparsity): window_left / window_right must both be -1, got "
                    f"{self.window_left} / {self.window_right}"
                )


def build_fused_qkvg_weight(
    w_q_gate: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    q_gate_layout: Optional[str] = None,
    index_qk_proj_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assemble ``W_qkvg [N, d_model]`` from the checkpoint's three matrices (four under an indexer band).

    **Load time only.** The result is what stage (1) reads; nothing here ever
    runs on the execute path. Loading a HF Qwen checkpoint? Use
    :func:`qkvg_from_hf`, which fixes the layout below to the one those
    checkpoints use and also prepares the two norm weights.

    Parameters
    ----------
    w_q_gate
        ``q_proj.weight``, ``[2*H_q*D, d_model]`` — the model's double-width
        query projection, holding both Q and the output gate.
    w_k, w_v
        ``k_proj.weight`` / ``v_proj.weight``, ``[H_kv*D, d_model]``.
    q_gate_layout
        How ``w_q_gate``'s rows split into Q and GATE, which is a property of
        the checkpoint and is NOT inferable from the tensor:

        ``"flat"``
            ``[all Q heads | all GATE heads]`` — the split a
            ``chunk(2, dim=-1)`` on the flat ``[..., 2*H_q*D]`` output performs.
        ``"per_head"``
            ``[Q_0 | GATE_0 | Q_1 | GATE_1 | ...]`` — the split a
            ``view(..., H_q, 2*D)`` followed by ``chunk(2, dim=-1)`` performs.

        Get this wrong and every gate is applied to the wrong head: the output
        is finite, plausible, and wrong, with no error anywhere. Verify it
        against the model you are loading rather than trusting a default.

        ``None`` (the default) is DEPRECATED: it still means ``"flat"`` so no
        existing caller changes behaviour, but it emits a ``FutureWarning``
        -- the deprecation category Python shows under its default warning
        filters, so a caller inside a library module sees it too, where a
        ``DeprecationWarning`` is shown only to ``__main__``. Every HF Qwen
        checkpoint from Qwen3-Next on is ``"per_head"``
        (``q_proj(x).view(..., H_q, 2*D).chunk(2, dim=-1)``), so the silent
        default was the trap the paragraph above describes. Pass the layout
        explicitly, or load through :func:`qkvg_from_hf`.
    index_qk_proj_weight
        APPENDED.  The indexer projection ``[(index_heads + index_kv_heads) *
        index_head_dim, d_model]`` of a block-sparse layer -- its query heads
        then its raw key head, the checkpoint's own row order -- REQUIRED iff
        ``geometry.qsa.index_band`` (it becomes the fifth band, below V) and
        REFUSED otherwise (a geometry without the band has exactly four).
    """
    geometry.validate()
    d = geometry.d_head
    hq, hkv = geometry.h_q, geometry.h_kv
    if w_q_gate.shape[0] != 2 * hq * d:
        raise ValueError(f"w_q_gate must have {2 * hq * d} rows (2*h_q*d_head), got {w_q_gate.shape[0]}")
    for name, w in (("w_k", w_k), ("w_v", w_v)):
        if w.shape[0] != hkv * d:
            raise ValueError(f"{name} must have {hkv * d} rows (h_kv*d_head), got {w.shape[0]}")
    for name, w in (("w_q_gate", w_q_gate), ("w_k", w_k), ("w_v", w_v)):
        if w.shape[1] != geometry.d_model:
            raise ValueError(f"{name} must have {geometry.d_model} columns (d_model), got {w.shape[1]}")

    if q_gate_layout is None:
        warnings.warn(
            "build_fused_qkvg_weight: q_gate_layout was not given and defaults to 'flat' (all Q heads, then all GATE heads). "
            "HF Qwen checkpoints from Qwen3-Next on split q_proj PER HEAD ([q_h | gate_h]): load those through qkvg_from_hf(...) "
            "or pass q_gate_layout='per_head'. The implicit default is deprecated; pass the layout explicitly.",
            # FutureWarning, not DeprecationWarning: the default filters drop a DeprecationWarning raised from library code.
            FutureWarning,
            stacklevel=2,
        )
        q_gate_layout = "flat"
    if q_gate_layout == "flat":
        w_q, w_gate = w_q_gate[: hq * d], w_q_gate[hq * d :]
    elif q_gate_layout == "per_head":
        per_head = w_q_gate.view(hq, 2 * d, geometry.d_model)
        w_q = per_head[:, :d].reshape(hq * d, geometry.d_model)
        w_gate = per_head[:, d:].reshape(hq * d, geometry.d_model)
    else:
        raise ValueError(f"q_gate_layout must be 'flat' or 'per_head', got {q_gate_layout!r}")

    bands = [w_q, w_gate, w_k, w_v]
    if geometry.index_band:
        q = geometry.qsa
        if index_qk_proj_weight is None:
            raise ValueError(
                f"index_qk_proj_weight is required: geometry.qsa.index_band=True puts a fifth band of {q.index_band_cols} indexer columns "
                f"([{q.index_heads} query heads | {q.index_kv_heads} raw key head] x {q.index_head_dim}) below V in W_qkvg"
            )
        if tuple(index_qk_proj_weight.shape) != (q.index_band_cols, geometry.d_model):
            raise ValueError(
                f"index_qk_proj_weight must be [{q.index_band_cols}, {geometry.d_model}] (the indexer's {q.index_heads} query heads then its "
                f"{q.index_kv_heads} raw key head, {q.index_head_dim} rows each, over d_model), got {tuple(index_qk_proj_weight.shape)}"
            )
        if index_qk_proj_weight.dtype != w_q_gate.dtype:
            raise ValueError(f"index_qk_proj_weight must have the other matrices' dtype {w_q_gate.dtype}, got {index_qk_proj_weight.dtype}")
        bands.append(index_qk_proj_weight)
    elif index_qk_proj_weight is not None:
        raise ValueError(
            "index_qk_proj_weight: this geometry declares no indexer band, so W_qkvg has exactly the four bands Q | GATE | K | V and the "
            "indexer projection is not part of it; declare GatedAttentionBlockGeometry(qsa=QsaSpec(index_band=True)) for a fifth band, or pass None"
        )
    return torch.cat(bands, dim=0).contiguous()


def qkvg_from_hf(
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    geometry: GatedAttentionBlockGeometry,
    *,
    index_qk_proj_weight: Optional[torch.Tensor] = None,
    act_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(w_qkvg, w_q_norm, w_k_norm)`` for the block from a HF Qwen attention layer's tensors.

    **Load time only** -- the three results are what the block reads on every
    ``execute``; nothing here runs on the hot path. "From HF" means the two
    conventions every Qwen checkpoint from Qwen3-Next on shares (Qwen3-Next,
    Qwen3.5, the Qwen3.8 family), applied here so the caller cannot get them
    wrong:

    * **The double-width ``q_proj`` is split PER HEAD.** The model computes
      ``q, gate = q_proj(x).view(..., H_q, 2*D).chunk(2, dim=-1)``, so the
      rows of ``q_proj.weight`` are ``[q_0 | gate_0 | q_1 | gate_1 | ...]`` --
      :func:`build_fused_qkvg_weight` with ``q_gate_layout="per_head"``. The
      other layout applies every gate to the wrong head with no error anywhere.
    * **The QK-norm weights are zero-centered.** The model's RMSNorm is
      ``x * rsqrt(mean(x^2) + eps) * (1 + w)`` with ``w`` initialised to zeros,
      while the block multiplies by the ``[D]`` vector it is given. The form
      handed over is DERIVED from the geometry's ``norm_weight_offset`` (0.0
      when the geometry has no such field): at ``0.0`` the block adds nothing,
      so it receives ``(1 + w)``; at ``1.0`` the block's norm kernels add the
      ``1`` themselves in fp32, so it receives ``w`` as is. Deriving it from
      the same geometry the block is declared with is what keeps the two
      mechanisms from composing into ``1 + (1 + w)`` -- finite, plausible,
      wrong -- which is why there is no free-standing argument for it.

    ``(1 + w)`` is formed in fp32 and rounded ONCE into ``act_dtype``. bf16
    has 8 significand bits, so near 1.0 that rounding is at most ``2^-8`` =
    0.39 % relative per channel (``2^-11`` = 0.049 % in f16), and a trained
    ``-2^-9 < w < 2^-8`` rounds to exactly 1.0 (an asymmetric interval: bf16's
    spacing is ``2^-8`` just below 1.0 and ``2^-7`` just above; f16 loses
    ``-2^-12 < w < 2^-11``); a systematic per-channel scale error inside the
    block's accuracy budget but not HF-faithful -- the fp32 in-kernel offset
    form is the faithful one once a geometry declares it.

    Parameters
    ----------
    q_proj_weight
        ``q_proj.weight``, ``[2*H_q*D, d_model]`` (Q and the output gate, per head).
    k_proj_weight, v_proj_weight
        ``k_proj.weight`` / ``v_proj.weight``, ``[H_kv*D, d_model]``.
    q_norm_weight, k_norm_weight
        ``q_norm.weight`` / ``k_norm.weight``, ``[D]``, zero-centered; both
        ``None`` iff ``geometry.qk_norm`` is ``False`` (then the block takes no
        norm weights and ``None`` is returned in their slots).
    geometry
        The block's geometry; also the source of ``norm_weight_offset``.
    index_qk_proj_weight
        The indexer projection ``index_qk_proj.weight`` of a block-sparse layer,
        ``[(index_heads + index_kv_heads) * index_head_dim, d_model]``: REQUIRED
        iff ``geometry.qsa.index_band`` (appended below V as the fifth band of
        ``w_qkvg``, rows as the checkpoint stores them), REFUSED otherwise.
    act_dtype
        The block's ACTIVATION dtype (bf16 by default, f16 also served). All
        three results are returned in it, contiguous. A quantized pipeline
        quantizes the returned ``w_qkvg`` afterwards and keeps the norm weights
        in this dtype, exactly as the block's descriptor contract requires.
    """
    if not isinstance(act_dtype, torch.dtype) or not act_dtype.is_floating_point:
        raise ValueError(f"act_dtype must be a floating-point torch dtype (the block's activation dtype), got {act_dtype!r}")
    offset = float(getattr(geometry, "norm_weight_offset", 0.0))
    if offset not in (0.0, 1.0):
        raise ValueError(
            f"geometry.norm_weight_offset must be 0.0 (the block multiplies by the weight as given, so it receives 1 + w) or 1.0 "
            f"(the block adds the 1 itself, so it receives w); got {offset}"
        )
    w_qkvg = (
        build_fused_qkvg_weight(q_proj_weight, k_proj_weight, v_proj_weight, geometry, q_gate_layout="per_head", index_qk_proj_weight=index_qk_proj_weight)
        .to(act_dtype)
        .contiguous()
    )

    norms = []
    for name, w in (("q_norm_weight", q_norm_weight), ("k_norm_weight", k_norm_weight)):
        if not geometry.qk_norm:
            if w is not None:
                raise ValueError(f"{name}: geometry.qk_norm=False -- the block applies no QK-RMSNorm and takes no norm weights; pass None")
            norms.append(None)
            continue
        if w is None:
            raise ValueError(f"{name} is required when geometry.qk_norm=True (a zero-centered [d_head] vector; HF Qwen initialises it to zeros)")
        if tuple(w.shape) != (geometry.d_head,):
            raise ValueError(f"{name} must be [d_head={geometry.d_head}] (one RMSNorm weight per head dim), got {tuple(w.shape)}")
        # (1 + w) in fp32 -- the model's own `output * (1.0 + self.weight.float())` -- then ONE rounding into the
        # activation dtype; at offset 1.0 the kernel adds the 1 in fp32 and `w` travels as is.
        norms.append(((1.0 + w.float()) if offset == 0.0 else w.float()).to(act_dtype).contiguous())
    return w_qkvg, norms[0], norms[1]


def qkvg_from_vllm(
    qkv_proj_weight: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    geometry: GatedAttentionBlockGeometry,
    *,
    act_dtype: torch.dtype = torch.bfloat16,
    index_qk_proj_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(w_qkvg, w_q_norm, w_k_norm)`` for the block from a serving stack's MERGED QKV linear at THIS rank's geometry.

    **Load time only.** vLLM's ``QKVParallelLinear`` (SGLang's likewise) holds one ``qkv_proj.weight`` per tensor-parallel
    rank whose rows are the HF attention layer's three matrices, cut to the rank's heads and concatenated:
    ``[q_proj | k_proj | v_proj]`` -- ``2 * H_q * D`` rows of ``q_proj`` in the checkpoint's per-head ``[q_h | gate_h]``
    order (the layer declares ``2 * H_q`` query heads to carry the gate), then ``H_kv * D`` rows of ``k_proj`` and
    ``H_kv * D`` of ``v_proj`` -- all at the RANK's ``H_q`` / ``H_kv``: ``geometry`` describes the shard, not the whole
    model (8 / 1 for a 32 / 2 model at TP 4, the KV head replicated across the ranks that share it).  This twin cuts those
    three row ranges and hands them to :func:`qkvg_from_hf`, so all three results are BITWISE what ``qkvg_from_hf`` returns
    for the same HF tensors: the per-head split is fixed to the one those checkpoints use (there is no layout argument to get
    wrong), and the two QK-norm weights -- replicated on every rank, zero-centered -- take the form ``qkvg_from_hf`` derives
    from ``geometry.norm_weight_offset`` (``(1 + w)`` formed in fp32 and rounded once into ``act_dtype`` at the default
    ``0.0``; ``w`` as is at ``1.0``).  ``w_qkvg`` is a row PERMUTATION of the input into the block's ``Q | GATE | K | V``
    bands with the same byte count, so a layer may replace its merged weight by it in place and run the block against the
    model's own ``o_proj.weight`` as ``W_o``; nothing is left to convert on the execute path.

    Parameters
    ----------
    qkv_proj_weight
        The rank's merged weight, ``[2*H_q*D + 2*H_kv*D, d_model]`` at the rank's head counts.  Under an indexer band
        (``geometry.qsa.index_band``) it may instead carry the indexer projection's rows APPENDED -- ``[..., +
        (index_heads + index_kv_heads) * index_head_dim]``, the form of a serving stack that packs the (replicated) indexer
        projection into the same GEMM -- with ``index_qk_proj_weight`` left ``None``.
    q_norm_weight, k_norm_weight
        ``q_norm.weight`` / ``k_norm.weight``, ``[D]``, zero-centered; both ``None`` iff ``geometry.qk_norm`` is ``False``.
    geometry
        The block's geometry AT THIS RANK (``h_q``, ``h_kv`` = the shard's heads); also the source of ``norm_weight_offset``.
    act_dtype
        The block's activation dtype (bf16 by default, f16 also served); all three results come back in it, contiguous.
    index_qk_proj_weight
        The indexer projection ``[(index_heads + index_kv_heads) * index_head_dim, d_model]`` of a block-sparse layer -- its
        query heads then its raw key head, the checkpoint's row order: REQUIRED under ``geometry.qsa.index_band`` unless
        those rows ride ``qkv_proj_weight`` (above; giving both is refused), REFUSED without the band -- exactly
        :func:`qkvg_from_hf`'s arm.
    """
    geometry.validate()
    if not isinstance(qkv_proj_weight, torch.Tensor) or qkv_proj_weight.dim() != 2:
        got = tuple(qkv_proj_weight.shape) if isinstance(qkv_proj_weight, torch.Tensor) else type(qkv_proj_weight).__name__
        raise ValueError(f"qkv_proj_weight must be a 2-D [rows, d_model] weight (a merged QKV linear's weight at this rank), got {got}")
    d, hq, hkv, dm = geometry.d_head, geometry.h_q, geometry.h_kv, geometry.d_model
    rows_q, rows_kv = 2 * hq * d, hkv * d
    rows_qkv = rows_q + 2 * rows_kv
    rows_band = geometry.qsa.index_band_cols if geometry.index_band else 0
    n_rows = qkv_proj_weight.shape[0]
    if qkv_proj_weight.shape[1] != dm:
        raise ValueError(f"qkv_proj_weight must have {dm} columns (d_model), got {qkv_proj_weight.shape[1]}")
    layout = (
        f"[{rows_q} q_proj rows (per-head [q_h | gate_h]) | {rows_kv} k_proj | {rows_kv} v_proj] at this rank's geometry " f"(h_q={hq}, h_kv={hkv}, d_head={d})"
    )
    if geometry.index_band and n_rows == rows_qkv + rows_band:
        # The packed form: the indexer projection's rows ride the merged weight, below v_proj.
        if index_qk_proj_weight is not None:
            raise ValueError(
                f"index_qk_proj_weight: the indexer band's {rows_band} rows were given twice -- appended to qkv_proj_weight "
                f"({n_rows} = {rows_qkv} + {rows_band} rows) AND as index_qk_proj_weight; pass one of the two"
            )
        index_qk_proj_weight = qkv_proj_weight[rows_qkv:]
    elif n_rows != rows_qkv:
        packed = f", or {rows_qkv + rows_band} with the indexer band's {rows_band} rows appended" if geometry.index_band else ""
        raise ValueError(f"qkv_proj_weight must have {rows_qkv} rows = {layout}{packed}; got {n_rows}")
    elif geometry.index_band and index_qk_proj_weight is None:
        raise ValueError(
            f"index_qk_proj_weight is required: geometry.qsa.index_band=True puts a fifth band of {rows_band} indexer columns below V "
            f"in W_qkvg; pass the indexer projection, or hand qkv_proj_weight with its {rows_band} rows appended ({rows_qkv + rows_band} rows)"
        )
    w_q_gate = qkv_proj_weight[:rows_q]
    w_k = qkv_proj_weight[rows_q : rows_q + rows_kv]
    w_v = qkv_proj_weight[rows_q + rows_kv : rows_qkv]
    return qkvg_from_hf(w_q_gate, w_k, w_v, q_norm_weight, k_norm_weight, geometry, index_qk_proj_weight=index_qk_proj_weight, act_dtype=act_dtype)


# ---------------------------------------------------------------------------
# RoPE table contract
# ---------------------------------------------------------------------------
#
# ``cos`` / ``sin`` are PRECOMPUTED per-token tables of shape
# ``[B, S, ROPE_DIM]`` (B may be 1 and broadcast), in the block's storage dtype
# or fp32. Element ``i`` of the rotated slice pairs with element
# ``i + ROPE_DIM // 2`` -- the NeoX / ``rotate_half`` convention that HF
# ``apply_rotary_pos_emb`` and vLLM's default ``RotaryEmbedding`` use, with the
# halves duplicated so ``cos[..., i] == cos[..., i + ROPE_DIM // 2]``:
#
#     x_rot = x[..., :ROPE_DIM]
#     out   = x_rot * cos + rotate_half(x_rot) * sin
#     where rotate_half(v) = cat(-v[..., H:], v[..., :H]), H = ROPE_DIM // 2
#
# The duplication is redundant (2x a 128 KiB table at S=1K) and kept anyway
# because it makes the tables byte-identical to what a HF or vLLM caller already
# holds -- so no conversion happens on the hot path (Rule 1).
#
# **Taking a TABLE rather than position ids is the load-bearing choice.** mRoPE
# (per-section positions), plain RoPE, YaRN and any NTK scaling all differ only
# in how the table is built; the caller builds it (``cudnn.yarn`` already ships
# the YaRN half) and the kernel never learns which. This is also what makes
# ColQwen3.5 and the vision tower reachable without touching stage (3).
#
# NOT served, and declined explicitly rather than silently mis-rotated:
# the GPT-J / INTERLEAVED pairing (``i`` with ``i+1``). DeepSeek's
# interleaved-in / halves-out variant lives in
# ``gemm/cutedsl/dense/proj_rope_mxfp8`` if it is ever needed here.


# ---------------------------------------------------------------------------
# 3. Workspace — every intermediate the unfused chain materialises
# ---------------------------------------------------------------------------


_WS_ALIGN = 256


def _align_up(n: int, a: int = _WS_ALIGN) -> int:
    return (n + a - 1) // a * a


_ITEMSIZE: dict = {}


def _itemsize(dtype: torch.dtype) -> int:
    """Bytes per element of ``dtype`` -- memoised: the view builders asked torch for an empty tensor on every call."""
    n = _ITEMSIZE.get(dtype)
    if n is None:
        n = _ITEMSIZE[dtype] = torch.empty((), dtype=dtype).element_size()
    return n


def _view(ws: torch.Tensor, offset: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """A typed, shaped VIEW of the caller's uint8 workspace at ``offset``.

    Views only — never a copy, never an allocation (Rule 1). Offsets are
    ``_WS_ALIGN``-aligned so the dtype reinterpretation is always legal.
    """
    n = 1
    for x in shape:
        n *= int(x)
    nbytes = n * _itemsize(dtype)
    return ws[offset : offset + nbytes].view(dtype).view(*shape)


def _cols(proj: torch.Tensor, col_offset: int, h: int, d: int) -> torch.Tensor:
    """A ``[T, h, d]`` view of ``h*d`` COLUMNS of the fused ``[T, N]`` projection.

    Heads are contiguous within a token (stride ``d``, elem 1) but the token
    stride is ``N``, not ``h*d`` — a padded stride. Every consumer in this block
    addresses that natively; the one that would not is the SDPA, which is why V
    gets its own compaction stage.
    """
    t, n = int(proj.shape[0]), int(proj.shape[1])
    return torch.as_strided(proj, (t, h, d), (n, d, 1), storage_offset=proj.storage_offset() + col_offset)


@dataclass(frozen=True)
class _Intermediates:
    """Byte offsets into the caller's workspace, one per materialised tensor.

    ``get_workspace_size()`` must be honest and ``execute()`` must not allocate
    (contract § 10 / Rule 1), so every intermediate is reserved at build time and
    carved as a view at execute time.

    At Qwen3.5-397B, B=1 S=1Mi bf16 these are NOT small: PROJ alone is 34 GiB and
    Q/O are 16 GiB each, while K and V are 1 GiB (GQA is 16x and ``h_kv`` is 2).
    That asymmetry decides both the fusion order here and the context-parallel
    strategy in the design record § 9.2.

    ``-1`` means the tensor never lands in HBM: ``gate`` lives in ``proj``'s
    columns and ``o_gated`` aliases ``o`` (stage (5) gates in place).  Under the
    FULLY FUSED FP8 pipeline ``proj`` and ``o`` are ``-1`` too: the fused
    projection writes the COMPACT e4m3 ``q8`` / ``k8`` / ``v8`` (the same three
    slots the unfused FP8 quantize passes fill) + ``gate16`` (bf16 GATE), and
    the gated FP8 SDPA writes ``o8`` directly.

    Under the TRAINING forward (``_plan_workspace(want_saved=True)``, i.e.
    ``save_for_backward``) the caller's :class:`SavedForBackward` takes two slots
    over: ``o`` is ``-1`` (the SDPA writes the PRE-gate ``saved.o`` directly) and
    ``o_gated`` is RESERVED (stage (5) gates OUT of place into it, stage (6) reads
    it); ``proj`` is ``-1`` in the proj_slab save mode (stage (1) writes
    ``saved.proj_slab``) and reserved as usual in the gate-copy mode
    (``saved_gate_copy``: the GATE band is copied out of it into ``saved.gate``).
    Under a QUANTIZED training forward (FP8 / MXFP8) the bf16 compact ``q`` /
    ``k`` are reserved as well: norm+RoPE writes the normed Q/K there out of
    place (the slab's Q/K bands stay PRE-norm for the record) and the quantize
    stages read them.
    """

    proj: int  # [T, N]            stage (1) output; holds Q | GATE | K | V  (-1 when FP8 fully fused, or when it is saved.proj_slab)
    q: int  # [T, H_q,  D]     compact, post-norm, post-RoPE (bf16 out of place; and FP8 / MXFP8 TRAINING, where the quantize stages read it)
    gate: int  # -1: a column slice of proj
    k: int  # [T, H_kv, D]     compact (same rule as q)
    v: int  # [T, H_kv, D]     compact (stage 3b)
    o: int  # [T, H_q,  D]     SDPA output, gated in place by (5)  (-1 when FP8 fully fused, or when it is saved.o under training)
    o_gated: int  # -1: aliases o (inference).  Reserved under training: stage (5) gates saved.o OUT of place into it, stage (6) reads it
    engine_scratch: int  # the sub-engines' own workspace (GEMM / SDPA)

    total_bytes: int
    base_align: int
    # FP8 pipelines only (-1 otherwise): compact e4m3 Q/K/V the FP8 SDPA reads
    # -- written by the quantize stages (unfused) or by the projection fork's
    # epilogue (fully fused) -- and the e4m3 O the FP8 out_proj reads.
    q8: int = -1  # [T, H_q,  D]
    k8: int = -1  # [T, H_kv, D]
    v8: int = -1  # [T, H_kv, D]
    o8: int = -1  # [T, H_q * D]
    # FULLY FUSED FP8 only (-1 otherwise): the fused projection's bf16 GATE.
    gate16: int = -1  # [T, H_q, D]  bf16, compact GATE
    # MXFP8 pipelines only (-1 otherwise): the SDPA's F8_128x4 E8M0 scale-factor
    # blobs for Q / K / V -- ``b * h * ceil(s/128) * (128 * d/32)`` bytes each
    # (the adapter's ``_reshape_sf`` count; 1 KiB per (b, h, 128-row tile) at
    # d=256).  Written by the quantize_mxfp8 stages (unfused) or by the MXFP8
    # projection fork's epilogue (fully fused).
    sf_q: int = -1
    sf_k: int = -1
    sf_v: int = -1
    # fp4 O (``MxQuantSpec.o_fp4``) only (-1 otherwise): the packed e2m1 gated O the fp4 out_proj reads
    # -- ``[T, H_q*D/2]`` bytes, viewed ``float4_e2m1fn_x2`` -- and its PADDED F8_128x4 scale blob over
    # ``(rows=T, K=H_q*D)``, ``proj_gemm.sf_blob_bytes(T, H_q*D, block)`` bytes: the GEMM's contract, NOT the
    # SDPA's ``_sf_slot_bytes`` (a different byte ORDER and count).  ``o8`` is -1 under o_fp4 (never written,
    # so never reserved); on the fused arm ``o`` (bf16) takes its place, since the gated SDPA then writes bf16 O.
    o4: int = -1
    sf_o: int = -1
    # The in-block indexer (``QsaSpec.index_source="indexer"``, ``S`` past the identity bound) only (-1 otherwise), at the END of
    # the layout: the normed + rotated indexer queries, the compressed keys the scorer reads, the selection (ids + scores) and
    # the scorer's compact-logits scratch -- :func:`_qsa_indexer_slots`.  Every layout without the indexer is byte-identical.
    ix_q: int = (
        -1
    )  # [T, index_heads, index_head_dim] activation dtype (the decode form over the pools: the indexer CACHE dtype -- e4m3 at 1 B under the attribute)
    ix_kbar: int = -1  # [B, floor(S / block_size), index_head_dim] activation dtype
    ix_ids: int = -1  # [T, top_k] int32: the top-k's output (its slot order unspecified)
    ix_scores: int = -1  # [T, top_k] fp32
    ix_cand: int = -1  # [cand_floats] fp32
    ix_ids_sorted: int = -1  # [T, top_k] int32: the canonical list the sparse core consumes (ids descending, then -1)
    ix_sort_idx: int = -1  # [min(T, _QSA_SORT_ROWS), top_k] int64: the row sort's index output, reused per row window
    # e4m3 KV cache (``kv_cache_dtype``) only (-1 otherwise): the compact e4m3 K / V rows the quantizing write-through stages into
    # before the bit copy into the pools, the two fp32 reciprocals of ``k_scale`` / ``v_scale``, and -- on the block-sparse decode mode
    # -- the gathered-cast bf16 scratch ``[items, 2, 2176, 256]`` + its per-item ``kv_len_t`` (``kernels/qsa_kv_upcast.py``).
    kc8: int = -1  # [T, H_kv, D] e4m3
    vc8: int = -1  # [T, H_kv, D] e4m3
    kv_inv: int = -1  # [2] fp32: 1 / k_scale, 1 / v_scale
    kv_gathered: int = -1  # [B x S_q x H_kv, 2, 2176, 256] bf16
    kv_len_t: int = -1  # [B x S_q x H_kv] int32
    # fp8 indexer cache (``QsaSpec.index_cache_dtype=torch.float8_e4m3fn``) only (-1 otherwise): the e4m3 indexer queries and compressed
    # keys the scorer's MXFP8 arm reads, cast from ``ix_q`` / ``ix_kbar`` (below the identity bound only ``ix_kbar`` is reserved, as the
    # cache cast's bf16 staging slot).  Appended LAST: every layout without the attribute is byte-identical to before.
    ix_q8: int = -1  # [T, index_heads, index_head_dim] e4m3
    ix_kbar8: int = -1  # [B, floor(S / block_size), index_head_dim] e4m3
    # the DECODE form's PREFILL route only (-1 otherwise): the paged THD leg's per-sequence Q lengths -- the constant S_q of every
    # sequence of the dense step -- a [B] int32 slot the block FILLS on the launch stream at every execute (Rule 8 / R2 + R4: never a
    # plan-owned tensor, never written once behind a flag)
    decode_q_lens: int = -1  # [B] int32
    # the in-block indexer's DECODE step over the pools (``QsaSpec(index_source="indexer")`` with ``kv_source="paged"``) only (-1
    # otherwise), appended at the very END (:func:`_qsa_indexer_decode_slots`; ``ix_q`` / ``ix_ids`` above are reused): the scorer's fp32
    # logits ``[T, n_blocks_max]`` (the columns ``[0, counts[t])`` written per row, the rest never read), its visible counts ``[T]``, and
    # the compress arm's per-row slot report ``[B, S_q]`` (device data nothing on the execute path reads).  Every layout without the form
    # is byte-identical.
    ix_logits: int = -1  # [T, n_blocks_max] fp32
    ix_counts: int = -1  # [T] int32
    ix_cslot: int = -1  # [B, S_q] int32


_SF_TILE_ROWS = 128  # rows of one F8_128x4 scale-factor atom == the SDPA's Q / KV tile height (keep in step with kernels/quantize_mxfp8.py SF_TILE_ROWS)
_SF_BLOCK = 32  # elements per E8M0 scale (MXFP8 block size)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _sf_slot_bytes(b: int, h: int, s: int, d: int) -> int:
    """Bytes of ONE of the SDPA's F8_128x4 scale-factor blobs (Q, K or V) over ``[B, H, S, D]``.

    ``B * H * ceil(S/128) * (128 * D/32)`` -- exactly the adapter's ``_reshape_sf``
    count (``b*h*n_tiles*SF_SMEM_SIZE``, 1 KiB per tile at d=256) and the
    ``kernels/quantize_mxfp8.py`` ``sf_bytes`` contract the quantize stages
    write; Q/K (rowwise) and V (columnwise, D-plane-major) have the SAME byte
    count, only the byte ORDER differs.  Derived, never a literal.
    """
    return b * h * _ceil_div(s, _SF_TILE_ROWS) * (_SF_TILE_ROWS * d // _SF_BLOCK)


def _sf_slot_bytes_packed(num_sequences: int, h: int, t: int, d: int) -> int:
    """Bytes of ONE SDPA scale-factor blob under THD (``thd=True``): the PACKED per-sequence-tile-padded layout the Rubin
    d256 MXFP8 row reads, sized at its CAPACITY -- ``H * n_cap * (128 * D/32)`` with ``n_cap = (T + 127 * B) // 128``, the
    host bound on the live tile total ``sum_b ceil(s_b / 128)`` (the lengths' VALUES are never read; ``B = num_sequences``).
    The dense ``(1, T)`` count ``H * ceil(T/128)`` is short by up to ``B - 1`` tiles per head -- a slot sized that way is
    an out-of-bounds write by the packed quantizer.  ``kernels/quantize_mxfp8.py::sf_bytes_packed`` is the ONE source of
    the formula; K and V (both ``h_kv``) carry the same count by construction, which the SDPA's binder requires (it
    derives the tile extent from the byte count, so the view handed over must be the WHOLE slot, never the live total).
    """
    from .kernels.quantize_mxfp8 import sf_bytes_packed

    return sf_bytes_packed(num_sequences, h, t, d)


def _plan_workspace(
    geom: GatedAttentionBlockGeometry,
    b: int,
    s: int,
    dtype: torch.dtype,
    want_lse: bool,
    want_rstd: bool,
    inplace_qkv: bool = False,
    fp8: bool = False,
    fp8_fused: bool = False,
    mxfp8: bool = False,
    o_fp4: Optional[Fp4Format] = None,
    want_saved: bool = False,
    saved_gate_copy: bool = False,
    thd_num_sequences: int = 0,
    kv_cache_e4m3: bool = False,
    upcast_items: int = 0,
    decode_q_lens: int = 0,
    qsa_indexer_cand_floats: Optional[int] = None,
    qsa_indexer_e4m3: bool = False,
    qsa_indexer_decode: Optional[int] = None,
    qsa_indexer_decode_cache_bytes: Optional[int] = None,
) -> _Intermediates:
    """Reserve every intermediate, in stage order, and report the total.

    ``qsa_indexer_decode`` (appended; ``None`` = no decode indexer) is the in-block indexer's DECODE step over the pools: the
    logits width ``n_blocks_max`` (the main block table's reach in compressed rows, ``max_pages x P / block_size``) -- the five slots
    of :func:`_qsa_indexer_decode_slots` go at the very END of the layout (after ``decode_q_lens``), so every layout without the
    form is byte-identical to before; a form of the bf16 inference pipeline, never beside the prefill indexer's slots.
    ``qsa_indexer_decode_cache_bytes`` (appended; ``None`` = the activation dtype's size) is the indexer cache dtype's element size: the
    ``ix_q`` slot is carved at it (1 under the e4m3 indexer cache -- the e4m3 queries; every bf16 carve byte-identical).

    ``kv_cache_e4m3`` / ``upcast_items`` (appended; the e4m3 KV cache): the quantizing write-through's compact e4m3 ``kc8`` /
    ``vc8`` rows (``t * h_kv * d_head`` bytes each) and the two inverse scales ``kv_inv``, and -- ``upcast_items > 0``, the
    block-sparse decode mode over e4m3 pools -- the gathered-cast bf16 scratch ``kv_gathered`` (``items x 2 x 2176 x 256`` bf16)
    plus ``kv_len_t`` (``items`` int32), all at the END of the non-fused layout; every layout without the attribute is
    byte-identical to before.  A form of the bf16 pipeline (``fp8`` with it is a typed ``ValueError`` here, mirroring the
    declaration).

    ``want_saved`` / ``saved_gate_copy`` (appended): the TRAINING forward
    (``save_for_backward=True``; out of place, on the UNFUSED bf16 / fp16 /
    per-tensor FP8 / MXFP8 pipelines, the fp4 modes included -- a fully fused
    quantized or an in-place training carve is a typed ``ValueError`` here,
    mirroring the block's own declaration declines; under ``o_fp4`` the carve
    reserves ``o_gated`` and appends ``o4`` / ``sf_o`` exactly as the inference
    carve does, and never ``o8``).  Under ``fp8`` (per-tensor FP8 or MXFP8) the bf16
    compact ``q`` / ``k`` are reserved as well: norm+RoPE writes the normed Q/K
    there OUT of place so the slab's Q/K bands stay PRE-norm for the record, and
    the quantize stages read them (+17 KiB/token at the 397B geometry; the
    inference carve is untouched).  ``o`` is NOT reserved (the SDPA writes the
    caller's PRE-gate ``saved.o``) and ``o_gated`` IS (stage (5) gates OUT of
    place into it; stage (6) reads it); ``proj`` is not reserved in the
    proj_slab save mode (stage (1) writes ``saved.proj_slab``) and reserved as
    today under ``saved_gate_copy`` (the GATE band is copied out of it into the
    compact ``saved.gate``).  Every ``want_saved=False`` layout is byte-identical
    to before (pinned by ``test_block_training_forward.py::
    test_workspace_layout_is_byte_identical_without_want_saved``).

    ``mxfp8`` (appended) adds the three SDPA scale-factor blobs ``sf_q`` /
    ``sf_k`` / ``sf_v`` (:func:`_sf_slot_bytes`) at the END of either layout, so
    every FP8 offset is byte-identical to before MXFP8 existed.
    ``thd_num_sequences`` (appended; 0 = dense) sizes those three slots at the
    PACKED capacity (:func:`_sf_slot_bytes_packed`, ``B = thd_num_sequences``,
    ``T = b * s``) for a ``thd=True`` block -- the ONLY difference between the
    packed MXFP8 carve and the dense ``(1, T)`` one (pinned by
    ``test_block_thd_mxfp8.py``).

    ``o_fp4`` (appended; an :class:`Fp4Format`) appends ``o4`` (packed e2m1 gated O,
    ``t*h_q*d_head // 2`` bytes) and ``sf_o`` (its padded F8_128x4 blob,
    ``proj_gemm.sf_blob_bytes(t, h_q*d_head, block)`` bytes -- the GEMM contract)
    at the END of either layout and does NOT reserve ``o8`` (a slot never written
    is not reserved); on the fused arm the bf16 ``o`` the gated SDPA then writes
    takes ``o8``'s place.  Every layout with ``o_fp4=None`` is byte-identical to
    before (pinned by ``test_workspace_layout_is_byte_identical_without_fp4``).

    ``qsa_indexer_cand_floats`` (appended; ``None`` = no in-block indexer) appends
    the indexer's five slots (:func:`_qsa_indexer_slots`, the scorer's compact-logits
    scratch sized at that many fp32) at the END of the unfused layout; every layout
    without it is byte-identical to before.

    ``qsa_indexer_e4m3`` (appended; the fp8 indexer cache, ``QsaSpec.index_cache_dtype=torch.float8_e4m3fn``): past the identity
    bound the two e4m3 slots ``ix_q8`` / ``ix_kbar8`` follow the indexer's seven; below it ONE bf16 staging slot ``ix_kbar`` (the
    cache cast's source) is reserved.  Only a declaration with the attribute reaches either: every layout without it is
    byte-identical to before.

    ``decode_q_lens`` (appended; 0 = none) reserves, at the very END, the DECODE
    form's PREFILL route's per-sequence Q lengths: ``decode_q_lens`` (= ``B``)
    int32 words the block fills with the constant ``S_q`` on the launch stream
    at every execute (Rule 8 recipes R2 + R4: the paged THD leg reads them on
    device, the caller never hands them over, and a plan-owned tensor written
    once behind a flag would leave an eager execute after a captured first
    execute reading an unwritten vector).  Every layout with ``decode_q_lens=0``
    is byte-identical to before.

    Reserved in the order the stages write them, so a future fusion that deletes
    one leaves a contiguous prefix rather than a hole — forking stage (1) to
    write four compact buffers deletes ``proj`` and stage (3b) together.

    Scratch only. Anything the BACKWARD needs leaves through
    :class:`SavedForBackward`, into storage the caller owns, because a workspace
    is dead the moment ``execute()`` returns. ``lse`` is likewise a caller
    tensor: it is an OUTPUT, not an intermediate.
    """
    del want_lse, want_rstd
    if want_saved:
        # The training carve must agree with the body that fills it.  The UNFUSED bf16 / fp16 / per-tensor FP8 / MXFP8
        # pipelines write the record (a bf16 slab with PRE-norm Q/K bands, a bf16 pre-gate O, the LSE) -- the fp4 modes
        # included: the fp4 tail replaces `o8` by `o4` / `sf_o` in THIS carve and the record it writes is byte for byte the
        # MXFP8 record (the SDPA writes bf16 `o`, stage (5) gates out of place into `o_gated`, the fp4 quantize reads it);
        # the FULLY FUSED quantized pipelines write no slab and no bf16 O, and in-place Q/K would destroy the slab's
        # pre-norm columns the record hands over.
        if fp8_fused:
            raise ValueError(
                "want_saved (the training forward's workspace carve) needs the UNFUSED pipeline: the FULLY FUSED FP8 / MXFP8 pipelines write "
                "no bf16 slab and no pre-gate O (no q_pre / k_pre / pre-gate O contract), so they are inference-only"
            )
        if inplace_qkv:
            raise ValueError(
                "want_saved requires inplace_qkv=False: the training forward keeps the slab's Q/K columns as q_pre / k_pre and writes the "
                "normed Q/K out of place (see SavedForBackward)"
            )
    elif saved_gate_copy:
        raise ValueError("saved_gate_copy=True selects the gate-copy SAVE mode of a training forward and has no meaning without want_saved=True")
    if qsa_indexer_cand_floats is not None and (fp8_fused or fp8 or want_saved):
        raise ValueError(
            "qsa_indexer_cand_floats (the in-block indexer's workspace slots) belongs to the bf16 inference pipeline: the block declines the "
            "indexer under quant / save_for_backward at declaration, so a quantized or training carve never reserves them"
        )
    if qsa_indexer_e4m3 and (fp8_fused or fp8 or want_saved):
        raise ValueError("qsa_indexer_e4m3 (the fp8 indexer cache's slots) belongs to the bf16 inference pipeline, like the indexer's own slots")
    if qsa_indexer_decode is not None:
        if fp8_fused or fp8 or want_saved:
            raise ValueError(
                "qsa_indexer_decode (the decode indexer's slots over the pools) belongs to the bf16 inference pipeline, like the indexer's own slots"
            )
        if qsa_indexer_cand_floats is not None or qsa_indexer_e4m3:
            raise ValueError("qsa_indexer_decode (the decode step over the pools) and the prefill indexer's slots never share a declaration")
        if int(qsa_indexer_decode) < 1:
            raise ValueError(f"qsa_indexer_decode must be the logits width n_blocks_max >= 1, got {qsa_indexer_decode}")
    if kv_cache_e4m3 and fp8:
        raise ValueError(
            "kv_cache_e4m3 (the e4m3 KV cache) is a form of the bf16 pipeline: the quantized pipelines' K / V are e4m3 codes of another "
            "recipe (the per-tensor FP8 / MXFP8 SDPA rows take e4m3 Q / K / V together)"
        )
    if upcast_items and not kv_cache_e4m3:
        raise ValueError("upcast_items belongs to the e4m3 KV cache (kv_cache_e4m3=True): the gathered-cast scratch exists only there")
    e = _itemsize(dtype)
    t = b * s
    off = 0
    offsets = {}
    if fp8_fused:
        # FULLY FUSED FP8 (fuse_norm_rope + fuse_gate under a QuantSpec): the
        # projection fork writes COMPACT e4m3 Q / K / V into `q8` / `k8` / `v8`
        # (the unfused pipeline's own slots -- the gated FP8 SDPA reads exactly
        # what the ungated one reads, compact BSHD at token_stride 0) and the
        # bf16 GATE into `gate16`; the gated FP8 SDPA writes e4m3 `o8`.  No bf16 slab,
        # no bf16 O, no quantize passes: 33792 B/token at the 397B geometry
        # against 68608 unfused (-51 %).  Same byte total as round 1's
        # [T, n_qkv] slab (q8 + k8 + v8 == t * n_qkv), laid out per tensor.
        slots = [
            ("q8", t * geom.h_q * geom.d_head),
            ("k8", t * geom.h_kv * geom.d_head),
            ("v8", t * geom.h_kv * geom.d_head),
            ("gate16", t * geom.h_q * geom.d_head * e),
            # fp4 O (row 10): the gated SDPA writes bf16 `o` here instead of e4m3 `o8`; quantize_fp4 reads it.
            ("o", t * geom.h_q * geom.d_head * e) if o_fp4 is not None else ("o8", t * geom.h_q * geom.d_head),
        ]
        if mxfp8:
            slots += _sf_slots(geom, b, s, thd_num_sequences)
        if o_fp4 is not None:
            slots += _o_fp4_slots(geom, t, o_fp4)
        for name, nbytes in slots:
            offsets[name] = off
            off += _align_up(nbytes)
        return _Intermediates(
            proj=-1,
            q=-1,
            gate=-1,
            k=-1,
            v=-1,
            o=offsets.get("o", -1),
            o_gated=-1,
            engine_scratch=off,
            total_bytes=off,
            base_align=_WS_ALIGN,
            q8=offsets["q8"],
            k8=offsets["k8"],
            v8=offsets["v8"],
            o8=offsets.get("o8", -1),
            gate16=offsets["gate16"],
            sf_q=offsets.get("sf_q", -1),
            sf_k=offsets.get("sf_k", -1),
            sf_v=offsets.get("sf_v", -1),
            o4=offsets.get("o4", -1),
            sf_o=offsets.get("sf_o", -1),
        )
    # IN-PLACE: Q and K are normed back over their own columns of `proj`, and V
    # is never moved, so the SDPA reads all three straight out of the slab at
    # its PADDED token stride. The three compact buffers stop existing -- 26% of
    # the block's workspace at every shape (18 GiB at 1Mi tokens). What makes it
    # legal is that the forward's own kernels are row-local (each lane holds its
    # whole [D] row in registers before storing) and the SDPA now compiles its
    # TMA descriptors AT the declared strides rather than repacking.
    # It is NOT legal under save_for_backward: the slab IS q_pre/k_pre, which
    # the RMSNorm backward needs and cannot safely reconstruct -- see
    # SavedForBackward. The caller-facing guard is in GatedAttentionBlock.
    #
    # TRAINING, proj_slab save mode: stage (1) writes the caller-owned
    # `saved.proj_slab` (bound by pointer as the GEMM output), so the slab is not
    # reserved here at all -- 34 KiB/token at the 397B geometry that the caller
    # keeps for the backward instead of the block scratching it.
    slots = [] if (want_saved and not saved_gate_copy) else [("proj", t * geom.n_qkvg * e)]
    if not inplace_qkv and not fp8:
        slots += [
            ("q", t * geom.h_q * geom.d_head * e),
            ("k", t * geom.h_kv * geom.d_head * e),
            ("v", t * geom.h_kv * geom.d_head * e),
        ]
    if fp8:
        # FP8 / MXFP8 INFERENCE: norm+RoPE stays in place on the bf16 slab; the
        # quantize stages then write COMPACT e4m3 Q/K/V (1 B/elem -- half the size
        # of the bf16 compact buffers they replace, and they double as V's
        # compaction), the SDPA writes bf16 O, the gate is in place, and O is
        # quantized into o8 for the FP8 out_proj.
        if want_saved:
            # FP8 / MXFP8 TRAINING: the slab's Q/K bands are the record's PRE-norm
            # q_pre / k_pre, so norm+RoPE writes the normed Q/K OUT of place into
            # these bf16 compact slots and the quantize stages read them (the same
            # bytes the in-place norm would have written, to a different place:
            # launch count and traffic unchanged).  V needs no slot: it is never
            # normed, and its quantize reads the slab's V band.  +17 KiB/token at
            # the 397B geometry; the inference carve (want_saved=False) is untouched.
            slots += [
                ("q", t * geom.h_q * geom.d_head * e),
                ("k", t * geom.h_kv * geom.d_head * e),
            ]
        slots += [
            ("q8", t * geom.h_q * geom.d_head),
            ("k8", t * geom.h_kv * geom.d_head),
            ("v8", t * geom.h_kv * geom.d_head),
        ]
    if want_saved:
        # TRAINING: the SDPA writes the caller's PRE-gate `saved.o` (the backward's dG
        # operand -- never gated in place); stage (5) gates OUT of place into `o_gated`,
        # which stage (6) reads.  Same bytes as the inference `o` slot, one slot later.
        slots += [("o_gated", t * geom.h_q * geom.d_head * e)]
    else:
        slots += [("o", t * geom.h_q * geom.d_head * e)]
    if fp8 and o_fp4 is None:
        slots += [("o8", t * geom.h_q * geom.d_head)]
    if mxfp8:
        # MXFP8: the SDPA's F8_128x4 SF blobs, written by the three quantize_mxfp8 stages.
        slots += _sf_slots(geom, b, s, thd_num_sequences)
    if o_fp4 is not None:
        # fp4 O (rows 8 / 9): the quantize_fp4 stage's packed codes + the out_proj GEMM's scale blob, at the END.
        slots += _o_fp4_slots(geom, t, o_fp4)
    if qsa_indexer_cand_floats is not None:
        # The in-block indexer: its queries, compressed keys, selection and the scorer's scratch, after everything else (+ the two e4m3
        # slots of the fp8 indexer cache).
        slots += _qsa_indexer_slots(geom, b, s, e, qsa_indexer_cand_floats, e4m3=qsa_indexer_e4m3)
    elif qsa_indexer_e4m3:
        # The fp8 indexer cache below the identity bound: the bf16 staging slot of the cache cast only (the compress writes bf16, the e4m3
        # cache is its cast); there is no scorer to feed.
        slots += [("ix_kbar", b * (s // geom.qsa.block_size) * geom.qsa.index_head_dim * e)]
    if kv_cache_e4m3:
        # e4m3 KV cache (kv_cache_dtype): the quantizing write-through's compact e4m3 K / V rows and the two inverse scales; on the
        # block-sparse decode mode the gathered-cast scratch + its per-item lengths.  At the END: every layout without the attribute
        # is byte-identical to before.
        slots += [("kc8", t * geom.h_kv * geom.d_head), ("vc8", t * geom.h_kv * geom.d_head), ("kv_inv", 2 * 4)]
        if upcast_items:
            slots += [("kv_gathered", int(upcast_items) * 2 * _UPCAST_ROWS * _UPCAST_D * 2), ("kv_len_t", int(upcast_items) * 4)]
    if decode_q_lens:
        # The DECODE form's PREFILL route: the paged THD leg's per-sequence Q lengths ([B] int32, the constant S_q), carved here and
        # filled on the launch stream by every execute (_fill_decode_q_lens).  At the END: every other layout is byte-identical to before.
        slots += [("decode_q_lens", int(decode_q_lens) * 4)]
    if qsa_indexer_decode is not None:
        # The in-block indexer's DECODE step over the pools: its queries, the scorer's logits and counts, the list, the slot report -- at the
        # very END, so every layout without the form is byte-identical to before.
        slots += _qsa_indexer_decode_slots(geom, b, s, e, int(qsa_indexer_decode), qsa_indexer_decode_cache_bytes)
    for name, nbytes in slots:
        offsets[name] = off
        off += _align_up(nbytes)
    engine = off
    return _Intermediates(
        proj=offsets.get("proj", -1),
        q=offsets.get("q", -1),
        gate=-1,
        k=offsets.get("k", -1),
        v=offsets.get("v", -1),
        o=offsets.get("o", -1),
        o_gated=offsets.get("o_gated", -1),
        engine_scratch=engine,
        total_bytes=engine,
        base_align=_WS_ALIGN,
        q8=offsets.get("q8", -1),
        k8=offsets.get("k8", -1),
        v8=offsets.get("v8", -1),
        o8=offsets.get("o8", -1),
        sf_q=offsets.get("sf_q", -1),
        sf_k=offsets.get("sf_k", -1),
        sf_v=offsets.get("sf_v", -1),
        o4=offsets.get("o4", -1),
        sf_o=offsets.get("sf_o", -1),
        ix_q=offsets.get("ix_q", -1),
        ix_kbar=offsets.get("ix_kbar", -1),
        ix_ids=offsets.get("ix_ids", -1),
        ix_scores=offsets.get("ix_scores", -1),
        ix_cand=offsets.get("ix_cand", -1),
        ix_ids_sorted=offsets.get("ix_ids_sorted", -1),
        ix_sort_idx=offsets.get("ix_sort_idx", -1),
        kc8=offsets.get("kc8", -1),
        vc8=offsets.get("vc8", -1),
        kv_inv=offsets.get("kv_inv", -1),
        kv_gathered=offsets.get("kv_gathered", -1),
        kv_len_t=offsets.get("kv_len_t", -1),
        ix_q8=offsets.get("ix_q8", -1),
        ix_kbar8=offsets.get("ix_kbar8", -1),
        decode_q_lens=offsets.get("decode_q_lens", -1),
        ix_logits=offsets.get("ix_logits", -1),
        ix_counts=offsets.get("ix_counts", -1),
        ix_cslot=offsets.get("ix_cslot", -1),
    )


def _sf_slots(geom: GatedAttentionBlockGeometry, b: int, s: int, thd_num_sequences: int = 0) -> list:
    """The three MXFP8 SDPA scale-factor slots, in (Q, K, V) order.  ``thd_num_sequences`` (appended; 0 = dense) selects
    the PACKED capacity count of :func:`_sf_slot_bytes_packed` at ``T = b * s`` for a ``thd=True`` block."""

    def nbytes(h: int) -> int:
        if thd_num_sequences:
            return _sf_slot_bytes_packed(thd_num_sequences, h, b * s, geom.d_head)
        return _sf_slot_bytes(b, h, s, geom.d_head)

    return [("sf_q", nbytes(geom.h_q)), ("sf_k", nbytes(geom.h_kv)), ("sf_v", nbytes(geom.h_kv))]


def _o_fp4_code_bytes(geom: GatedAttentionBlockGeometry, t: int) -> int:
    """Bytes of the packed e2m1 gated O: ``T * H_q * D / 2`` (two codes per byte along K = H_q*D)."""
    return t * geom.h_q * geom.d_head // 2


def _o_fp4_slots(geom: GatedAttentionBlockGeometry, t: int, o_fp4: Fp4Format) -> list:
    """The two fp4-O slots, in the order quantize_fp4 writes them: ``o4`` (codes), ``sf_o`` (the out_proj GEMM's
    PADDED F8_128x4 blob over ``(rows=T, K=H_q*D)`` at the format's block -- ``proj_gemm.sf_blob_bytes``)."""
    from .kernels.proj_gemm import sf_blob_bytes

    return [
        ("o4", _o_fp4_code_bytes(geom, t)),
        ("sf_o", sf_blob_bytes(t, geom.h_q * geom.d_head, o_fp4.block_size)),
    ]


def _qsa_cand_floats(batch: int, seq_len: int, block_size: int) -> int:
    """fp32 elements of the indexer scorer's compact-logits scratch for ``batch`` dense sequences of ``seq_len`` tokens.

    The scorer writes, per query at position ``p``, its ``floor((p + 1) / block_size)`` ratio-causal candidate scores
    compactly, so one sequence needs ``sum_{m=1}^{S} floor(m / bs)`` floats -- the closed form the DSA sizing helper
    (``compress_topk_cand_buffer_size_thd``) evaluates on device; :meth:`_Indexer.compile` cross-checks the two and
    raises on a drift.  Quadratic in ``S`` (``~ S^2 / (2 bs)``): 512 MiB per 32K-token sequence at ``bs = 4``.
    """
    q, r = divmod(int(seq_len), int(block_size))
    return int(batch) * (int(block_size) * q * (q - 1) // 2 + q * (r + 1))


def _qsa_indexer_slots(geom: GatedAttentionBlockGeometry, b: int, s: int, e: int, cand_floats: int, e4m3: bool = False) -> list:
    """The in-block indexer's workspace slots, in the order its launches write them: the normed + rotated indexer
    queries ``ix_q`` ``[T, index_heads, index_head_dim]``, the compressed keys ``ix_kbar`` ``[B, floor(S / bs), index_head_dim]``
    (the scorer's key operand), the selection ``ix_ids`` / ``ix_scores`` ``[T, top_k]`` int32 / fp32 and the scorer's
    compact-logits scratch ``ix_cand`` (``cand_floats`` fp32).  Reserved only when the scorer runs (``S`` past the
    identity bound); below it the block has nothing to select and the carve is the dense one.  ``e4m3`` (appended; the fp8 indexer
    cache) adds the two e4m3 slots ``ix_q8`` / ``ix_kbar8`` the scorer's MXFP8 arm reads, after the seven."""
    q = geom.qsa
    t = b * s
    slots = [
        ("ix_q", t * q.index_heads * q.index_head_dim * e),
        ("ix_kbar", b * (s // q.block_size) * q.index_head_dim * e),
        ("ix_ids", t * q.top_k * _itemsize(torch.int32)),
        ("ix_scores", t * q.top_k * _itemsize(torch.float32)),
        ("ix_cand", int(cand_floats) * _itemsize(torch.float32)),
        ("ix_ids_sorted", t * q.top_k * _itemsize(torch.int32)),
        ("ix_sort_idx", min(t, _QSA_SORT_ROWS) * q.top_k * _itemsize(torch.int64)),
    ]
    if e4m3:
        slots += [("ix_q8", t * q.index_heads * q.index_head_dim), ("ix_kbar8", b * (s // q.block_size) * q.index_head_dim)]
    return slots


def _qsa_indexer_decode_slots(geom: GatedAttentionBlockGeometry, b: int, s: int, e: int, n_blocks_max: int, cache_e: Optional[int] = None) -> list:
    """The in-block indexer's DECODE-step slots over the pools, in the order its launches write them: the normed + rotated indexer
    queries ``ix_q`` ``[T, index_heads, index_head_dim]`` (the prepare's query arm / the pool-1 compress) in the indexer CACHE dtype --
    ``cache_e`` bytes per element (appended; ``None`` = ``e``, the activation dtype; 1 under ``QsaSpec.index_cache_dtype=torch.float8_e4m3fn``:
    the e4m3 queries the e4m3 scorer arm reads, half the slot; every bf16 carve is byte-identical) --, the scorer's fp32 logits
    ``ix_logits`` ``[T, n_blocks_max]`` (rows 16-byte aligned: ``n_blocks_max`` is a multiple of 4, the page a multiple of 16) and visible
    counts ``ix_counts`` ``[T]``, the top-k's canonical list ``ix_ids`` ``[T, top_k]`` int32 (skipped when a caller hands ``block_ids_out``
    in -- the top-k writes the caller's buffer directly) and the compress arm's slot report ``ix_cslot`` ``[B, S_q]`` int32.  No
    compressed-key slot (the paged cache is the caller's), no scorer scratch (the logits are the scratch), no sort scratch (the top-k
    writes the canonical order)."""
    q = geom.qsa
    t = b * s
    ce = e if cache_e is None else int(cache_e)
    return [
        ("ix_q", t * q.index_heads * q.index_head_dim * ce),
        ("ix_logits", t * int(n_blocks_max) * _itemsize(torch.float32)),
        ("ix_counts", t * _itemsize(torch.int32)),
        ("ix_ids", t * q.top_k * _itemsize(torch.int32)),
        ("ix_cslot", b * s * _itemsize(torch.int32)),
    ]


_QSA_SORT_ROWS = 4096
"""Rows per launch of the in-block indexer's canonical row sort: bounds the int64 index scratch the sort writes (16 MiB at
``top_k = 512``) at the price of ``ceil(T / 4096)`` launches."""


# ---------------------------------------------------------------------------
# 4. Saved-tensor contract — the forward/backward boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedForBackward:
    """What the forward hands the backward. **Append-only forever**, like a
    manifest slot: a field's meaning cannot change once a checkpoint or an
    autograd graph has been built against it.

    Sizes are per token; the 1M-token column is one full-attention layer of the
    397B at B=1 bf16, which is the regime that makes the trade real.

    ============ ================= ============ ===================================
    field         shape             at S=1Mi     why
    ============ ================= ============ ===================================
    ``h``         [B,S,d_model]     8 GiB        stage (1) wgrad AND recompute source
    ``gate``      [B,S,H_q,D]       16 GiB       stage (5) backward; NOT recomputable
                                                 without half the stage-(1) GEMM
    ``o``         [B,S,H_q,D]       16 GiB       ``dG`` needs pre-gate O; SDPA bwd
                                                 needs O too
    ``lse``       [B,H_q,S] fp32    134 MiB      SDPA bwd cannot run without it
    ``rstd_q``    [B,S,H_q] fp32    134 MiB      RMSNorm bwd; tiny, always save
                                                 (None iff geometry.qk_norm is False)
    ``rstd_k``    [B,S,H_kv] fp32   8 MiB        idem
    ``q_pre``     [B,S,H_q,D]       16 GiB       pre-norm Q -- SAVE or RECOMPUTE
    ``k_pre``     [B,S,H_kv,D]      1 GiB        idem
    ``proj_slab`` [B*S, n_qkvg]     34 GiB       APPENDED: the stage-(1) slab itself;
                                                 gate / q_pre / k_pre / V are its bands
    ``seq_lens``  [B] int32          --          APPENDED: dense -- the padding the
                                                 forward RAN with (identity, never
                                                 read); THD -- the packed lengths,
                                                 [B] or [B+1] prefix sums, REQUIRED,
                                                 bound by the backward
    ``seq_lens_form`` str            --          APPENDED: None (dense) / "lengths" /
                                                 "prefix" -- what seq_lens IS, verified
                                                 against the block's declaration
    ============ ================= ============ ===================================

    **Two SAVE modes, chosen at declaration** (``GatedAttentionBlockFwd(saved_gate_copy=)``,
    because the workspace carve differs; ``execute`` verifies the record agrees):

    * **proj_slab** (the default): the caller allocates ``proj_slab`` and the
      projection GEMM writes it directly (a per-call pointer bind, zero copies);
      ``gate`` / ``q_pre`` / ``k_pre`` are its column bands -- pass them as
      :func:`saved_slab_views` or leave them ``None`` and let the backward derive
      them.  The whole 34 KiB/token slab is saved (the fastest backward: nothing
      to recompute, and V comes for free), and the workspace loses its own slab.
    * **gate-copy** (``saved_gate_copy=True``): ``proj_slab`` is ``None``, the slab
      stays in the workspace and ONE elementwise launch copies the GATE band into
      the compact ``gate`` (16 KiB/token saved); ``q_pre`` / ``k_pre`` are copied
      too only if the caller passed buffers, else the backward recomputes them
      from ``h`` (``RecomputePolicy.RECOMPUTE_QK_PRE``).

    **This dataclass is the block's one genuinely novel value proposition, and
    the reason to build it even if the first A/B is flat.** ``gate`` is the same
    size as ``o``; saving it roughly doubles the attention activation footprint,
    and recomputing it re-runs half the stage-(1) GEMM. Likewise ``q_pre`` /
    ``k_pre`` are either saved (17 GiB) or recomputed from ``h`` by re-running
    the Q and K slices of the projection. **Both trades are invisible at the op
    level and a decomposed graph cannot make either one** — a block can, and can
    expose them as a knob.

    Do NOT try to reconstruct ``q_pre`` from the normed Q by dividing out
    ``w_q_norm``: it is undefined wherever a norm weight is zero, and it is
    numerically hostile wherever one is small.

    ``o_gated`` is deliberately absent: it is ``o * sigmoid(gate)``, so the
    backward's ``dW_o`` stage recomputes it elementwise from two tensors it
    already holds rather than storing a third 16 GiB copy.

    A field left ``None`` means "recompute me", and the backward decides how
    from :class:`~cudnn.gated_attention_block.api_bwd.RecomputePolicy` --
    with two exceptions.  ``rstd_q`` / ``rstd_k`` are ``None`` iff
    ``geometry.qk_norm`` is False: there is no RMSNorm, stage (B6) does not
    exist, and nothing could recompute them; the forward REQUIRES them to be
    ``None`` in that case (and tensors otherwise), both directions typed.  And
    in the proj_slab save mode ``gate`` / ``q_pre`` / ``k_pre`` left ``None``
    mean "a band of ``proj_slab``" (:func:`saved_slab_views` derives them, no
    recompute) -- the only mode in which ``gate`` may be ``None`` at all.
    """

    # The forward's input, verified to BE execute's `h` (same storage): bf16 / fp16, or the caller's e4m3 codes under a
    # QuantSpec / MxQuantSpec (the MXFP8 `h_sf` blob is a forward input, never a record field).
    h: torch.Tensor
    # Compact [B,S,H_q,D] (the gate-copy save mode), or the GATE column VIEW of proj_slab (see saved_slab_views).  None is
    # legal ONLY in the proj_slab save mode, where the band is derivable -- saved_slab_views(proj_slab, geometry, B, S)[1]
    # -- so the record ALWAYS carries the GATE one way or the other (api_bwd.RecomputePolicy.RECOMPUTE_GATE stays reserved).
    # Type widened from torch.Tensor: append-only compatible, the positional slot is unchanged.
    gate: Optional[torch.Tensor]
    o: torch.Tensor  # PRE-gate O, compact [B,S,H_q,D], 16-B aligned; the SDPA writes it directly under save_for_backward
    lse: torch.Tensor  # [B,H_q,S] fp32 natural log
    rstd_q: Optional[torch.Tensor]  # None iff geometry.qk_norm is False (positional slot kept: append-only)
    rstd_k: Optional[torch.Tensor]  # idem
    q_pre: Optional[torch.Tensor] = None
    k_pre: Optional[torch.Tensor] = None
    # APPENDED. The caller-owned stage-(1) output, [B*S, n_qkvg] or [B, S, n_qkvg], act dtype, contiguous, 16-B
    # aligned.  The SAVE MODE is declared on the block -- GatedAttentionBlockFwd(saved_gate_copy=), because the workspace
    # carve differs -- and execute() verifies the record agrees with it (a typed ValueError either way):
    #   * proj_slab mode (saved_gate_copy=False, the default): proj_slab is REQUIRED.  The forward's projection GEMM
    #     writes it (per-call pointer bind, zero copies) and gate / q_pre / k_pre / V are its column bands at
    #     geometry.qkvg_offsets -- pass them as saved_slab_views(...) or leave them None and let the backward derive them.
    #   * gate-copy mode (saved_gate_copy=True): proj_slab must be None.  The slab stays in the workspace and the forward
    #     copies the GATE band into `gate` (compact); q_pre / k_pre stay None unless the caller passes buffers for them
    #     (then they are copied too).
    proj_slab: Optional[torch.Tensor] = None
    # APPENDED. The `seq_lens` tensor the forward was EXECUTED with (the same object; no copy, no D2H read), or None
    # for a dense forward.  Dense: the per-batch KV padding mask -- lets the backward decline padding at check_support()
    # time instead of silently consuming a save set whose dead entries carry O = 0 / LSE = -inf.  THD (thd=True): the
    # packed per-sequence lengths ([B] int32, or [B+1] int32 prefix sums under cu_seqlens=True), REQUIRED -- the backward
    # binds this very tensor to rebuild the packed metadata.  The record is frozen, so the CALLER puts it here and the
    # forward VERIFIES identity (execute: `saved.seq_lens is seq_lens`), the way saved.h and lse are verified.
    seq_lens: Optional[torch.Tensor] = None
    # APPENDED. What `seq_lens` IS, so a record says which length contract it carries: None for a dense record (seq_lens is
    # the per-batch KV padding mask, or None), "lengths" for a THD record whose seq_lens is the [B] int32 per-sequence
    # lengths, "prefix" for one whose seq_lens is the [B+1] int32 prefix sums (GatedAttentionBlockFwd(thd=True, cu_seqlens=)).
    # The CALLER writes it; the forward VERIFIES it against its declaration at execute (typed ValueError, like the identity
    # check above) and the backward checks it against its own -- a padded dense record can never be consumed by a THD
    # backward, nor a packed one by a dense backward, without a typed decline.
    seq_lens_form: Optional[str] = None


def saved_slab_views(proj_slab: torch.Tensor, geometry: GatedAttentionBlockGeometry, batch: int, seq_len: int) -> tuple:
    """``(q_pre, gate, k_pre, v)`` as strided VIEWS of ``proj_slab`` (``_cols`` at ``qkvg_offsets``; token stride ``n_qkvg``,
    head stride ``d_head``), each ``[B, S, heads, D]``.  No copy, no allocation.  The block's execute verifies that a
    SavedForBackward built from these aliases proj_slab (data_ptr + shape + strides), typed ValueError otherwise.

    ``proj_slab`` is the caller's ``[B*S, n_qkvg]`` (or ``[B, S, n_qkvg]``) contiguous, 16-B-aligned stage-(1) output in
    the activation dtype -- the slab :class:`GatedAttentionBlockFwd` TMA-stores under ``save_for_backward`` in the
    proj_slab save mode.  A wrong element count, a non-contiguous or a misaligned slab is a typed ``ValueError``.
    """
    proj, b, s = _slab_2d(proj_slab, geometry, batch, seq_len)
    d = geometry.d_head
    # `_cols` gives the [T, h, d] band at token stride n; splitting T into (B, S) is a legal `.view` on it (dim-0 stride n
    # -> (S*n, n)), so every band keeps proj_slab's storage.  The four DENSE bands only: an indexer band (ProjBlock.INDEX,
    # a block-sparse geometry) is not part of the training record -- the forward declines save_for_backward under it.
    dense = (ProjBlock.Q, ProjBlock.GATE, ProjBlock.K, ProjBlock.V)
    return tuple(_cols(proj, geometry.qkvg_offsets[blk], geometry.qkvg_heads[blk], d).view(b, s, geometry.qkvg_heads[blk], d) for blk in dense)


def _slab_2d(proj_slab: torch.Tensor, geometry: GatedAttentionBlockGeometry, batch: int, seq_len: int) -> tuple[torch.Tensor, int, int]:
    """The stage-(1) slab contract, typed: ``proj_slab`` has ``B*S x n_qkvg`` elements, is contiguous and 16-B aligned.
    Returns ``(the [T, n_qkvg] view, B, S)`` -- what every band view is cut from."""
    b, s = int(batch), int(seq_len)
    t, n = b * s, geometry.n_qkvg
    if proj_slab.numel() != t * n:
        raise ValueError(
            f"proj_slab has {proj_slab.numel()} elements; the stage-(1) slab over B*S={t} tokens x n_qkvg={n} columns is {t * n} "
            "([B*S, n_qkvg] or [B, S, n_qkvg])"
        )
    if not proj_slab.is_contiguous():
        raise ValueError(
            f"proj_slab must be contiguous (the projection GEMM writes it as ONE row-major [B*S, n_qkvg] slab), got shape "
            f"{tuple(proj_slab.shape)} strides {tuple(proj_slab.stride())}"
        )
    if proj_slab.data_ptr() % 16:
        raise ValueError(f"proj_slab must be 16-byte aligned (the projection GEMM TMA-stores it), got data_ptr={proj_slab.data_ptr():#x}")
    return proj_slab.view(t, n), b, s


def index_k_raw_view(proj_slab: torch.Tensor, geometry: GatedAttentionBlockGeometry, batch: int, seq_len: int) -> torch.Tensor:
    """The RAW indexer key as a ``[B, S, index_kv_heads, index_head_dim]`` strided VIEW of the stage-(1) slab.

    It is the INDEX band's last ``index_kv_heads * index_head_dim`` columns (``geometry.index_k_raw_offset``), PRE-norm and
    un-rotated -- exactly what a serving cache keeps for the indexer -- at token stride ``n_qkvg``.  No copy, no allocation:
    a serving caller keeps ONE projection GEMM and reads the key off the slab the block already wrote
    (:meth:`GatedAttentionBlockFwd.index_k_raw` locates that slab inside the workspace).  The indexer's query columns sit
    just before it in the band and are computed and left in place.

    ``proj_slab`` is the ``[B*S, n_qkvg]`` (or ``[B, S, n_qkvg]``) contiguous, 16-B-aligned slab in the activation dtype;
    the contract of :func:`saved_slab_views` applies (typed ``ValueError``), as does a geometry without the band.
    """
    if not geometry.index_band:
        raise ValueError(
            "index_k_raw_view: this geometry declares no indexer band (GatedAttentionBlockGeometry.qsa.index_band), so the slab holds no raw indexer key"
        )
    proj, b, s = _slab_2d(proj_slab, geometry, batch, seq_len)
    q = geometry.qsa
    return _cols(proj, geometry.index_k_raw_offset, q.index_kv_heads, q.index_head_dim).view(b, s, q.index_kv_heads, q.index_head_dim)


class _SavedBinding(NamedTuple):
    """What ``GatedAttentionBlockFwd._check_saved_set`` hands ``execute`` once a :class:`SavedForBackward` record
    passed every contract: the caller tensors the training forward writes THROUGH.  Views of caller storage, never
    copies (Rule 1)."""

    lse: torch.Tensor  # the LSE the SDPA writes -- saved.lse (an explicit `lse` must be that same storage)
    rstd_q: Optional[torch.Tensor]  # saved.rstd_q / rstd_k (tensors iff geometry.qk_norm)
    rstd_k: Optional[torch.Tensor]
    proj: Optional[torch.Tensor]  # proj_slab mode: the [T, n_qkvg] view of saved.proj_slab stage (1) writes; None = the workspace slab (gate-copy mode)
    o: torch.Tensor  # [T, H_q, D] view of saved.o -- the SDPA's PRE-gate output
    gate_dst: Optional[torch.Tensor]  # gate-copy mode: the [T, H_q, D] view of saved.gate the GATE band is copied into; None in proj_slab mode
    q_pre_dst: Optional[torch.Tensor]  # gate-copy mode with a caller q_pre buffer: its [T, H_q, D] view; None = not copied (recompute)
    k_pre_dst: Optional[torch.Tensor]  # idem, [T, H_kv, D]


def _check_norm_weights_agree(qk_norm: bool, w_q_norm, w_k_norm, *, prefix: str = "") -> None:
    """``geometry.qk_norm`` and the two norm-weight slots must agree, BOTH ways.

    Typed ``ValueError`` naming the knob. Load-bearing rather than cosmetic:
    ``APIBase._make_tensor_desc(None)`` and ``_check_tensor_shape(None)`` both
    return ``None`` SILENTLY, so a norm-on block declared with ``None`` weights
    would sail through declaration and die in ``check_support``'s dtype loop with
    an untyped ``AttributeError`` -- or, worse, hand the kernel a null weight
    pointer. Called at declaration (``sample_*``) and again at every ``execute``.
    """
    have_q, have_k = w_q_norm is not None, w_k_norm is not None
    q_nm, k_nm = f"{prefix}w_q_norm", f"{prefix}w_k_norm"
    if have_q != have_k:
        raise ValueError(
            f"{q_nm} and {k_nm} must be given together or both be None (geometry.qk_norm={qk_norm}); "
            f"got {q_nm}={'tensor' if have_q else 'None'}, {k_nm}={'tensor' if have_k else 'None'}"
        )
    if qk_norm and not have_q:
        raise ValueError(
            f"geometry.qk_norm=True (QK-RMSNorm on) requires both [D] norm weights, but {q_nm} / {k_nm} are None. "
            "Pass them, or declare GatedAttentionBlockGeometry(qk_norm=False) for RoPE-only Q/K."
        )
    if not qk_norm and have_q:
        raise ValueError(
            f"geometry.qk_norm=False (RoPE-only Q/K, no RMSNorm) takes no norm weights, but {q_nm} / {k_nm} were given. "
            "Pass None for both, or declare GatedAttentionBlockGeometry(qk_norm=True)."
        )


def _check_index_side_cache(x, name: str, *, rows: int, exact_rows: bool, dtype, d: int, device, what: str, rows_contiguous: bool) -> None:
    """FORM checks of one paged side cache of the decode indexer (the compressed-key cache or the raw-key ring; a declaration sample or
    the execute tensor; Rule 3: never a value, never a sync): a 4-D ``[blocks >= 1, rows, 1, d]`` tensor of ``dtype`` -- the cache in the
    DECLARED indexer cache dtype (``QsaSpec.index_cache_dtype``: bf16, or e4m3 since 2026-10-21; refused both ways), the ring in the
    activation dtype (an e4m3 ring is refused by the ring's rule: a bit copy of the step's raw keys, never quantized) -- with ``D``
    contiguous, every other stride a whole number of 16-byte vectors (the kernels move rows in 128-bit accesses: 8 elements at 16 bits, 16 at
    e4m3), ``rows`` exactly (the cache: ``P / block_size`` rows per page)
    or at least (the ring: the ring rule), the rows contiguous inside a page when ``rows_contiguous`` (the scorer reads a page's compressed
    rows as one block), the page stride below 2 GiB (the scorer's 32-bit page stride), on ``device``.  Typed ``ValueError`` naming the
    tensor; the strides beyond these rules are the stack's (never compiled in)."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a 4-D [blocks >= 1, {'rows' if not exact_rows else rows}, 1, {d}] {dtype} tensor ({what}), got {type(x).__name__}")
    if x.dtype != dtype:
        is_ring = "state" in name
        why = (
            " (the raw-key ring is a bit copy of the step's raw keys and is never quantized; the compressed-key cache takes the dtype QsaSpec.index_cache_dtype declares)"
            if (x.dtype == torch.float8_e4m3fn and is_ring)
            else (
                " (the compressed-key cache is in the dtype QsaSpec.index_cache_dtype declares -- bf16, or torch.float8_e4m3fn for the e4m3 indexer cache -- never a silent cast)"
                if not is_ring
                else ""
            )
        )
        raise ValueError(f"{name} must be {dtype} ({'the activation dtype' if is_ring else 'the declared indexer cache dtype'}), got {x.dtype}{why}")
    shape = tuple(int(n) for n in x.shape)
    if x.dim() != 4 or shape[0] < 1 or shape[2] != 1 or shape[3] != d or (shape[1] != rows if exact_rows else shape[1] < rows):
        raise ValueError(f"{name} must be [blocks >= 1, {'exactly' if exact_rows else 'at least'} {rows}, 1, {d}] ({what}), got shape {shape}")
    strides = tuple(int(s) for s in x.stride())
    if strides[3] != 1:
        raise ValueError(f"{name} must be contiguous along its last dim (D), got strides {strides}")
    if rows_contiguous and strides[1] != d:
        raise ValueError(
            f"{name} must hold a page's compressed rows contiguously (row stride {d}: the scorer reads a page as one block), got strides {strides}"
        )
    per_access = 16 // _itemsize(dtype)  # elements per 16-byte vector (the kernels' ld / st .v4 granule)
    for dim in range(3):
        if shape[dim] > 1 and strides[dim] % per_access:
            raise ValueError(
                f"{name} stride {strides[dim]} (dim {dim}) must be a multiple of {per_access} elements so every row starts 16-byte aligned; got strides {strides}"
            )
    if strides[0] * _itemsize(dtype) >= 2**31:
        raise ValueError(f"{name}: the page / block stride must be below 2 GiB (the scorer's 32-bit page stride), got {strides[0] * _itemsize(dtype)} bytes")
    if x.device != torch.device(device):
        raise ValueError(f"{name} must live on {device}, got {x.device}")


def _check_index_tensor(x, name: str, flat_shape: tuple, dense_shape: tuple, device, *, hint: str = "") -> None:
    """FORM checks of an index tensor of a block-sparse block (Rule 3: never a value read, never a sync): a contiguous
    int32 ``torch.Tensor`` on ``device`` of exactly ``flat_shape`` (``[T, ...]``) or ``dense_shape`` (``[B, S, ...]``).
    Typed ``ValueError`` naming the tensor; an int64 list is refused, not converted (the sparse core reads 32-bit ids and
    a conversion would be a hot-path copy)."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != torch.int32:
        why = (
            " (int64 ids are refused, not converted: the sparse core reads 32-bit ids and a conversion would be a hot-path copy)"
            if x.dtype == torch.int64
            else ""
        )
        raise ValueError(f"{name} must be int32, got {x.dtype}{why}")
    shape = tuple(int(n) for n in x.shape)
    if shape not in (tuple(flat_shape), tuple(dense_shape)):
        raise ValueError(f"{name} must be {list(flat_shape)} or {list(dense_shape)}, got {list(shape)}{hint}")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous (the sparse core's index staging copies whole rows), got strides {tuple(x.stride())}")
    if x.device != device:
        raise ValueError(f"{name} must live on h's device {device}, got {x.device}")


_THD_FORM_LENGTHS = "lengths"  # SavedForBackward.seq_lens_form of a THD record whose seq_lens is the [B] int32 lengths
_THD_FORM_PREFIX = "prefix"  # ... the [B+1] int32 prefix sums (cu_seqlens=True)


def _thd_seq_lens_form(cu_seqlens: bool) -> str:
    """The ``SavedForBackward.seq_lens_form`` a THD block declared with ``cu_seqlens`` writes and verifies."""
    return _THD_FORM_PREFIX if cu_seqlens else _THD_FORM_LENGTHS


def _thd_token_matrix(sample: torch.Tensor, name: str, width: str) -> torch.Tensor:
    """A PACKED sample as the rank-3 ``[1, T, width]`` the block declares internally (``batch = 1, seq_len = T``).

    ``[T, width]`` gains a leading extent-1 axis (``unsqueeze(0)``: a view at any stride, never a copy); ``[1, T, width]``
    passes as is; anything else -- a dense ``[B, S, .]`` with ``B > 1``, a wrong rank -- is a typed ``ValueError`` naming
    the two packed forms.  ``execute`` takes the caller's tensors at either rank unchanged: every stage addresses them
    through ``.view(T, ...)``.
    """
    if not isinstance(sample, torch.Tensor):
        raise TypeError(f"thd=True: {name} must be a torch.Tensor sample ([T, {width}] or [1, T, {width}]), got {type(sample).__name__}")
    if sample.ndim == 2:
        return sample.unsqueeze(0)
    if sample.ndim == 3 and int(sample.shape[0]) == 1:
        return sample
    raise ValueError(f"thd=True: {name} is the packed token matrix [T, {width}] (or [1, T, {width}]), got {tuple(sample.shape)}")


# ---------------------------------------------------------------------------
# 4b. FP8 (E4M3, per-tensor STATIC scales) — the QuantSpec contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSpec:
    """Static per-tensor FP8 scales for the block (inference-style quantization).

    Passing a ``QuantSpec`` (and FP8 ``h`` / weights) selects the FP8 pipeline;
    ``None`` is the bf16/f16 block.  All scales are Python floats fixed at plan
    time -- calibrated offline, like the weights' own scales -- so the execute
    path does NO amax pass and NO host readback; the block allocates them as
    1-element fp32 device tensors ONCE in ``compile()`` and writes their values on
    the launch stream at its first ``execute`` (the SDPA adapter materialises its
    identity descales at compile time), and
    :meth:`GatedAttentionBlockFwd.update_quant_scales` rewrites those tensors IN
    PLACE on the launch stream for a per-step recalibration -- no recompile, no
    new plan: the TE-style caller contract (quantize ``h`` / the weights each
    step, recalibrate the activation scales from the previous step's record).

    Conventions (``x_real = x_fp8 * descale``; ``x_fp8 = sat_e4m3(x_real * scale)``):

    ============== ============================================================
    ``descale_h``      dequant multiplier of the FP8 input ``h``
    ``descale_w_qkvg`` dequant multiplier of the FP8 ``W_qkvg``
    ``descale_w_o``    dequant multiplier of the FP8 ``W_o``
    ``scale_q/k/v``    quant scales applied to the bf16 post-norm Q, post-norm K
                       and V before the SDPA; the SDPA receives ``1/scale`` as
                       its descales
    ``scale_o``        quant scale applied to the bf16 gated O before ``out_proj``
    ============== ============================================================

    UNFUSED: the two projections fold ``descale_a * descale_w`` into ONE
    scalar-multiply epilogue of the FROST GEMM (fp32 accumulate, bf16 out), so
    the slab and O stay bf16 and the norm+RoPE and gate kernels run unchanged.
    FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``): the same scalars ride
    INSIDE the two kernels -- ``alpha_qkvg`` and ``scale_q/k/v`` as one fp32 ``[4]``
    device vector read by the projection fork's epilogue (which writes e4m3
    Q/K/V and a bf16 GATE), ``1/scale_q/k/v`` and ``scale_o`` as the production
    FP8 SDPA's ``descale_q/k/v`` / ``scale_o`` execute tensors (it folds
    ``descale_v * scale_o`` into ``inv_sum`` in-kernel and writes e4m3 O).  Only
    E4M3 is served today;
    E5M2 is a knob away once a use case asks for it.
    """

    descale_h: float
    descale_w_qkvg: float
    descale_w_o: float
    scale_q: float
    scale_k: float
    scale_v: float
    scale_o: float
    dtype: torch.dtype = torch.float8_e4m3fn

    def validate(self) -> None:
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"QuantSpec: only torch.float8_e4m3fn is served, got {self.dtype}")
        for name in ("descale_h", "descale_w_qkvg", "descale_w_o", "scale_q", "scale_k", "scale_v", "scale_o"):
            v = float(getattr(self, name))
            if not (v > 0.0) or math.isnan(v) or v in (float("inf"),):
                raise ValueError(f"QuantSpec.{name} must be a finite positive float, got {v}")

    @property
    def alpha_qkvg(self) -> float:
        return self.descale_h * self.descale_w_qkvg

    @property
    def alpha_o(self) -> float:
        return (1.0 / self.scale_o) * self.descale_w_o


# ---------------------------------------------------------------------------
# 4c. MXFP8 (E4M3 codes + per-32-block E8M0 scales, F8_128x4) — the MxQuantSpec contract
# ---------------------------------------------------------------------------


MXFP8_BLOCK_SIZE = 32
_E8M0 = getattr(torch, "float8_e8m0fnu", None)
_SF_DTYPES = tuple(t for t in (torch.uint8, _E8M0) if t is not None)
_FP4_X2 = getattr(torch, "float4_e2m1fn_x2", None)  # packed e2m1: ONE byte = two codes along the contiguous axis
# The W_qkvg code dtypes an MxQuantSpec may name: e4m3 (MXFP8 x MXFP8) or e2m1 (the catalog's MIXED
# block-scale row, MXFP8 h x MXFP4 W -- `kernel_registry._BLOCK_SCALE_CASES` fp8_e4m3 x fp4_e2m1, E8M0 / 32).
_W_QKVG_DTYPES = tuple(t for t in (torch.float8_e4m3fn, _FP4_X2) if t is not None)


class Fp4Format(Enum):
    """The two fp4 OUTPUT formats the block quantizes its gated O to (``MxQuantSpec.o_fp4``) -- and, by
    construction, the format of the fp4 ``W_o`` the block-scale out projection multiplies it with.

    ONE member = (e2m1 codes, scale dtype, scale block): an illegal pairing cannot be spelled.  Exactly the
    two rows the FROST block-scale catalog serves for two e2m1 sides (``kernel_registry._BLOCK_SCALE_CASES``):

    ========= ============================ ======= ======================================================
    member    scale dtype                  block   catalog row
    ========= ============================ ======= ======================================================
    ``NVFP4`` ``torch.float8_e4m3fn``      16      ``fp4_e2m1 x fp4_e2m1`` with ``fp8_e4m3`` scales per 16
    ``MXFP4`` ``torch.float8_e8m0fnu``     32      ``fp4_e2m1 x fp4_e2m1`` with ``fp8_e8m0`` scales per 32
    ========= ============================ ======= ======================================================

    Members are NAMED after ``kernels/quantize_fp4.py``'s format keys (``"nvfp4"`` / ``"mxfp4"``): the
    quantize stage takes the member itself and ``fp4_format(member)`` cross-checks ``block_size`` against
    the kernel's table, so the enum and the kernel cannot drift apart silently.  No global (per-tensor)
    scale in either format: ``MxQuantSpec.scale_o`` / ``descale_w_o`` are pinned to 1.0 under ``o_fp4``.
    """

    NVFP4 = ("nvfp4", 16)  # e2m1 x e4m3 scales per 16 along K
    MXFP4 = ("mxfp4", 32)  # e2m1 x E8M0 scales per 32 along K

    @property
    def fmt_name(self) -> str:
        """The ``kernels/quantize_fp4.py`` format key (``"nvfp4"`` / ``"mxfp4"``)."""
        return self.value[0]

    @property
    def block_size(self) -> int:
        """Elements per scale along K (16 for NVFP4, 32 for MXFP4)."""
        return self.value[1]

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        """The scale blob's torch storage dtype: ``float8_e4m3fn`` (NVFP4) / ``float8_e8m0fnu`` (MXFP4); a blob may
        also arrive as plain ``uint8`` bytes."""
        return torch.float8_e4m3fn if self is Fp4Format.NVFP4 else _E8M0

    @property
    def sf_cudnn_dtype(self):
        """The ``cudnn.data_type`` the out-projection GEMM DECLARES its scale tensors in -- which selects the MMA's
        scale format (``FP8_E4M3`` for NVFP4, ``FP8_E8M0`` for MXFP4)."""
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        return cudnn.data_type.FP8_E4M3 if self is Fp4Format.NVFP4 else cudnn.data_type.FP8_E8M0


@dataclass(frozen=True)
class MxQuantSpec:
    """MXFP8 pipeline: ``h`` / ``W_qkvg`` / Q / K / V carry per-32-block E8M0 scales
    (cuDNN's F8_128x4 order), so only ``out_proj``'s per-tensor pair survives.

    A SIBLING of :class:`QuantSpec`, not a flag on it (PR-B D3): five of
    ``QuantSpec``'s seven floats are meaningless under block scales, and a
    separate dataclass makes the illegal combinations unrepresentable while
    leaving every FP8 caller byte-identical.  Passing an ``MxQuantSpec`` (with
    e4m3 ``h`` / weights AND the two scale-factor blobs ``sample_h_sf`` /
    ``sample_w_qkvg_sf``) selects the MXFP8 pipeline.

    ============== ============================================================
    ``descale_w_o``  dequant multiplier of the per-tensor FP8 ``W_o`` (D1: the
                     out projection stays per-tensor FP8)
    ``scale_o``      quant scale applied to the bf16 gated O before ``out_proj``
                     (UNFUSED path).  On the FULLY FUSED path the production
                     MXFP8 SDPA writes e4m3 O UNSCALED (its ABI has no
                     ``scale_o``), so it MUST be 1.0 there -- D8, typed decline.
    ``dtype``        the code dtype; only E4M3 is served (E5M2 declines typed)
    ``block_size``   32 -- the MX block; anything else is a ``ValueError``
    ``w_qkvg_dtype`` the ``W_qkvg`` code dtype (appended, default e4m3 = today's
                     MXFP8 x MXFP8 GEMM).  ``torch.float4_e2m1fn_x2`` selects an
                     MXFP4 weight: e2m1 codes stored ``[n_qkvg, d_model // 2]``
                     (two per byte, LOW nibble = even k) against the e4m3 ``h``
                     -- the FROST catalog's MIXED block-scale row (``fp8_e4m3 x
                     fp4_e2m1``, E8M0 scales per 32); ``w_qkvg_sf`` is UNCHANGED
                     (the same E8M0 / 32 F8_128x4 blob over ``n_qkvg x d_model``).
                     Served on both pipelines: unfused, and fully fused through
                     the MXFP8 projection fork's e2m1-B arm (3 launches).
    ``o_fp4``        (appended, default ``None`` = today's per-tensor e4m3 O).  An
                     :class:`Fp4Format` member selects the fp4 OUTPUT mode: the gated
                     O is block-quantized to e2m1 codes + that format's scale blob by
                     a ``quantize_fp4`` launch, and the out projection becomes the
                     fp4 x fp4 block-scale GEMM against an e2m1 ``W_o`` of the SAME
                     format -- stored ``[d_model, h_q*d_head // 2]`` as
                     ``torch.float4_e2m1fn_x2`` with its F8_128x4 blob
                     (``sample_w_o_sf`` / ``w_o_sf``,
                     ``proj_gemm.sf_blob_bytes(d_model, h_q*d_head, block)`` bytes).
                     Neither side carries a per-tensor scale: ``scale_o`` and
                     ``descale_w_o`` MUST be 1.0 (typed ``ValueError``).  Served on
                     the unfused pipeline (config row 8 / 9: ``quantize_o`` becomes
                     ``quantize_fp4_o``, still 9 launches) and on the fully fused one
                     (row 10: the gated MXFP8 SDPA writes bf16 O, then the fp4
                     quantize, then the fp4 out projection -- 4 launches).
    ============== ============================================================

    ``h`` / ``W_qkvg`` arrive PRE-quantized by the caller (D2: static, offline,
    like the weights' own scales -- the block runs no amax pass): e4m3 codes plus
    ONE F8_128x4 E8M0 blob each, PADDED to whole 128-row x 4-block atoms --
    ``kernels.proj_gemm.sf_blob_bytes(rows, d_model)`` bytes over ``rows = B*S``
    (``h``) / ``n_qkvg`` (``W_qkvg``), pad rows / blocks ``0x00``.
    ``descale_w_o * (1/scale_o)`` is ``alpha_o``, the out projection's epilogue.
    """

    descale_w_o: float
    scale_o: float = 1.0
    dtype: torch.dtype = torch.float8_e4m3fn
    block_size: int = MXFP8_BLOCK_SIZE
    w_qkvg_dtype: torch.dtype = torch.float8_e4m3fn  # appended: torch.float4_e2m1fn_x2 -> MXFP4 W_qkvg (the mixed row)
    o_fp4: Optional[Fp4Format] = None  # appended: Fp4Format.NVFP4 | MXFP4 -> fp4 gated O + fp4 W_o of the same format, block-scale out_proj

    @property
    def w_qkvg_fp4(self) -> bool:
        """``W_qkvg`` is packed e2m1 (``torch.float4_e2m1fn_x2``): the mixed MXFP8 x MXFP4 stage (1)."""
        return _FP4_X2 is not None and self.w_qkvg_dtype == _FP4_X2

    def validate(self, *, fused: bool = False) -> None:
        """Typed declines; ``fused=True`` adds the D8 unit-``scale_o`` rule of the fully fused path."""
        if self.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"MxQuantSpec: only torch.float8_e4m3fn codes are served (E5M2 is a knob away), got {self.dtype}")
        if int(self.block_size) != MXFP8_BLOCK_SIZE:
            raise ValueError(f"MxQuantSpec.block_size must be {MXFP8_BLOCK_SIZE} (one E8M0 scale per 32-element MX block), got {self.block_size}")
        if self.w_qkvg_dtype not in _W_QKVG_DTYPES:
            raise NotImplementedError(
                f"MxQuantSpec.w_qkvg_dtype: W_qkvg codes are torch.float8_e4m3fn (MXFP8 x MXFP8) or torch.float4_e2m1fn_x2 (the FROST "
                f"block-scale catalog's mixed row, kernel_registry._BLOCK_SCALE_CASES fp8_e4m3 x fp4_e2m1 with E8M0 scales per 32); got "
                f"{self.w_qkvg_dtype}. A uint8 blob of packed e2m1 codes is spelled .view(torch.float4_e2m1fn_x2), never uint8."
            )
        if self.o_fp4 is not None and not isinstance(self.o_fp4, Fp4Format):
            raise TypeError(f"MxQuantSpec.o_fp4 must be an Fp4Format member (Fp4Format.NVFP4 | Fp4Format.MXFP4) or None, got {self.o_fp4!r}")
        if self.o_fp4 is not None and _FP4_X2 is None:
            # Mirrors quantize_fp4.check_torch_fp4_dtypes: the o4 buffer and W_o are VIEWED as the packed e2m1 dtype, so a
            # torch without it cannot serve rows 8-10 -- declined here, before _expected_weight_dtypes reports a None dtype.
            raise NotImplementedError(f"MxQuantSpec.o_fp4 needs torch.float4_e2m1fn_x2 for the packed e2m1 O and W_o (torch {torch.__version__} has none)")
        for name in ("descale_w_o", "scale_o"):
            v = float(getattr(self, name))
            if not (v > 0.0) or math.isnan(v) or v in (float("inf"),):
                raise ValueError(f"MxQuantSpec.{name} must be a finite positive float, got {v}")
        if self.o_fp4 is not None:
            # No global scale on either fp4 side (plan section 1): the O quantizer writes local block scales
            # only, and W_o dequantizes through its own blob in the MMA.  A non-unit value here would be
            # silently dropped by the block-scale GEMM (no alpha epilogue) -- refused instead.
            if float(self.scale_o) != 1.0:
                raise ValueError(f"MxQuantSpec.scale_o: a block-scaled O has no per-tensor scale; pass scale_o=1.0 under o_fp4 (got {self.scale_o})")
            if float(self.descale_w_o) != 1.0:
                raise ValueError(
                    f"MxQuantSpec.descale_w_o: under o_fp4 W_o dequantizes through w_o_sf (its F8_128x4 scale blob); pass descale_w_o=1.0 "
                    f"(got {self.descale_w_o})"
                )
        elif fused and float(self.scale_o) != 1.0:
            # The per-tensor e4m3 O on the fully fused path.  Skipped under o_fp4: scale_o == 1.0 is already pinned above.
            raise NotImplementedError(
                f"MxQuantSpec.scale_o must be 1.0 on the FULLY FUSED MXFP8 path (got {self.scale_o}): the production MXFP8 SDPA writes "
                "e4m3 O UNSCALED (its ABI has no per-tensor scale_o -- PR-B D8). Use scale_o=1.0, or the unfused pipeline."
            )

    @property
    def alpha_o(self) -> float:
        return (1.0 / self.scale_o) * self.descale_w_o


def _check_sf_blob(sf: torch.Tensor, name: str, rows: int, k: int, *, block: int = MXFP8_BLOCK_SIZE, sf_dtypes: Tuple[torch.dtype, ...] = _SF_DTYPES) -> None:
    """A caller-supplied F8_128x4 scale-factor blob: dtype, PADDED byte count
    (``proj_gemm.sf_blob_bytes(rows, k, block)``), contiguity, 16-B alignment -- typed, before any kernel.

    ``block`` / ``sf_dtypes`` (appended, defaults = the MXFP8 E8M0 / 32 contract) generalise it to the fp4
    blobs: one scale byte per ``block`` elements along K, stored as any of ``sf_dtypes`` (``uint8`` plus
    the format's own fp8 storage dtype -- ``float8_e8m0fnu`` for MX scales, ``float8_e4m3fn`` for NVFP4)."""
    from .kernels.proj_gemm import sf_blob_bytes, sf_padded_dims

    if sf.dtype not in sf_dtypes:
        want = " or ".join(str(t) for t in sf_dtypes)
        raise ValueError(f"{name} must be {want} (scale bytes in F8_128x4 order), got {sf.dtype}")
    need = sf_blob_bytes(rows, k, block)
    if sf.numel() != need:
        rows_pad, k4 = sf_padded_dims(rows, k, block)
        raise ValueError(
            f"{name} has {sf.numel()} bytes; the F8_128x4 blob over {rows} rows x K={k} at block {block} is {need} = {rows_pad} (rows padded to 128) "
            f"x {k4} (K/{block} blocks padded to 4) -- whole 512-B atoms, pad rows / blocks 0x00 (kernels.proj_gemm.sf_blob_bytes)"
        )
    if not sf.is_contiguous():
        raise ValueError(f"{name} must be contiguous (an opaque F8_128x4 byte blob bound by storage order)")
    if sf.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned (TMA-fed)")


# ---------------------------------------------------------------------------
# 5. Stages — one FROST kernel each, in pipeline order
# ---------------------------------------------------------------------------


class _Stage(ABC):
    """One kernel of the block.

    Deliberately the same three-phase shape as ``APIBase`` (support / compile /
    execute) so a stage can be promoted to a standalone public API, or absorbed
    into its neighbour, without the block's own API moving.
    """

    name: str

    @abstractmethod
    def check_support(self) -> None:
        """Raise ``NotImplementedError`` / ``ValueError`` if this stage cannot
        serve the declaration. Never degrade silently, never adapt (Rule 2)."""

    @abstractmethod
    def compile(self) -> None:
        """Build the artifact. Plan-time keys only (Rule 4)."""

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Launch. No allocation, no D2H read, no implicit conversion
        (Rules 1 / 3), and everything ordered on the LAUNCH stream (Rule 5)."""


class _Projection(_Stage):
    """(1) and (6) — the block's two dense projections, ONE implementation.

    They are the same op at different shapes, so there is one stage class and
    two factories rather than two kernels::

        (1) qkv_gate_proj   h       [M, d_model]  @ W_qkvg^T  ->  [M, N_qkvg]
        (6) out_proj        O_gated [M, H_q*D]    @ W_o^T     ->  [M, d_model]

    **Neither writes a kernel.** Both drive the shipped FROST GEMM
    (``gemm/frost/kernel_templates/sm100_matmul.py`` — persistent, double-TMEM,
    CLC-scheduled, tuned tile catalog), pinned by name in the graph's ranked
    plan list. Details and the layout convention: ``kernels/proj_gemm.py``.

    **Fusion status — both are FORK candidates, and the template is built to be
    forked** (the SDPA backward already forked it for a 2-D batch, with a
    bidirectional "apply fixes both ways" note). Stage (1) will need a fork for
    the four-output-buffer epilogue of § 1 and, later, the FP8/MXFP8
    quantization epilogue; stage (1)'s GATE columns are also the block's one
    piece of independent MMA work, dependency-legal to run during the SDPA's
    softmax phase below ~12K tokens. **Measure against the unforked engine
    first** — it is the baseline any fork has to beat, and this stage is how you
    get that number.

    Stage (6) can never fuse into the SDPA: it contracts over all heads.
    """

    def __init__(
        self,
        *,
        m: int,
        k: int,
        n: int,
        dtype: torch.dtype,
        name: str,
        out_dtype: Optional[torch.dtype] = None,
        alpha: bool = False,
        block_scale: bool = False,
        w_dtype: Optional[torch.dtype] = None,
        block_size: int = MXFP8_BLOCK_SIZE,
        sf_dtype=None,
    ) -> None:
        self.name = name
        self.m = int(m)
        self.k = int(k)
        self.n = int(n)
        self.dtype = dtype
        # FP8: inputs e4m3, fp32 accumulate, `out_dtype` (bf16) out, and `alpha`
        # = descale_a * descale_w folded into the GEMM's scalar-multiply epilogue.
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.alpha = bool(alpha)
        # MXFP8: e4m3 codes + per-32-block E8M0 scale factors for A (over M) and W
        # (over N), dequantized IN the MMA (`block_scale_dequantize`) -- no alpha.
        # The caller hands the two F8_128x4 blobs to execute(sf_a=, sf_w=).
        self.block_scale = bool(block_scale)
        # fp4 (append-only, default = today's behaviour): `w_dtype` names W's dtype apart from
        # A's (an e2m1 weight against an e4m3 activation is the catalog's MIXED block-scale
        # row; two e2m1 sides are NVFP4 / MXFP4), `block_size` the scale block along K (32,
        # or 16 for NVFP4) and `sf_dtype` the scale dtype (a `cudnn.data_type`; None = the
        # pair's own default).  The served pairs are `kernels.proj_gemm.block_scale_pairing`.
        self.w_dtype = w_dtype if w_dtype is not None else dtype
        self.block_size = int(block_size)
        self.sf_dtype = sf_dtype
        self._plan = None

    def check_support(self) -> None:
        from .kernels.proj_gemm import block_scale_pairing

        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        a_dtypes = (torch.bfloat16, torch.float16, torch.float8_e4m3fn) + ((fp4,) if (self.block_scale and fp4 is not None) else ())
        if self.dtype not in a_dtypes:
            raise NotImplementedError(f"{self.name}: bf16/f16/fp8-e4m3 (and fp4-e2m1 under block_scale) only, got {self.dtype}")
        if self.dtype == torch.float8_e4m3fn and self.k % 16:
            raise NotImplementedError(f"{self.name}: FP8 needs K % 16 == 0 (TMA 16-byte rule at 1 B/elem), got K={self.k}")
        if self.out_dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: output dtype must be bf16/f16, got {self.out_dtype}")
        if self.block_scale:
            # The pairing table is the GEMM driver's (one source); its ValueError is the block's typed decline.
            try:
                block_scale_pairing(dtype=self.dtype, w_dtype=self.w_dtype, sf_dtype=self.sf_dtype, block_size=self.block_size, label=self.name)
            except ValueError as exc:
                raise NotImplementedError(str(exc)) from None
            if self.alpha:
                raise ValueError(f"{self.name}: block_scale=True carries its descale in the per-block scale factors; alpha must be False")
            if self.k % MXFP8_BLOCK_SIZE:
                # 32 also for NVFP4 (two 16-blocks): it is the fp4 TMA rule (16 bytes = 32 codes) as much as the scale block.
                raise NotImplementedError(
                    f"{self.name}: block_scale=True needs K % {MXFP8_BLOCK_SIZE} == 0 (whole scale blocks; the fp4 TMA rule), got K={self.k}"
                )
        elif self.w_dtype != self.dtype:
            raise NotImplementedError(
                f"{self.name}: a per-weight dtype (A {self.dtype}, W {self.w_dtype}) exists only as a block-scale row; the dense GEMM takes one dtype"
            )

    def resolves_a_jit_tile(self) -> bool:
        """Whether ``tile_config="auto"`` names a JIT tile for this stage's own ``(M, N, K)`` -- asked of the driver's ONE resolver
        (``kernels.proj_gemm._auto_tile_name``: the small-M route at ``M <= SMALL_M_MAX``, else the forced 256-wide tile where 256
        divides N) with the operands :meth:`compile` hands ``build_proj_gemm`` (K-major A and B, this stage's dtype / alpha /
        block_scale).  True is the JIT-only route (no backend plan, no cuDNN handle at execute: what a ``capture_safe`` declaration
        needs to compile); False is the graph route, where ``capture_safe=True`` would be the driver's typed decline at ``compile()``
        -- so the block sets the flag only where this reads True.  The indexer band's slab (N 13952, off the 256 grid) reads True
        at decode row counts through the small-M route's 128-row swap-AB tile and False past 128 rows.  A host query (no device
        sync): the SM count and the compiler family it reads are device properties."""
        from .kernels.proj_gemm import _auto_tile_name

        return _auto_tile_name(self.m, self.n, self.k, dtype=self.dtype, a_major="k", b_major="k", block_scale=self.block_scale, alpha=self.alpha) is not None

    def compile(self) -> None:
        from .kernels.proj_gemm import build_proj_gemm

        self._plan = build_proj_gemm(
            m=self.m,
            k=self.k,
            n=self.n,
            dtype=self.dtype,
            label=self.name,
            out_dtype=self.out_dtype,
            alpha=self.alpha,
            block_scale=self.block_scale,
            sf_dtype=self.sf_dtype,
            w_dtype=self.w_dtype,
            block_size=self.block_size,
            # capture_safe: the serving declaration's flag (default off); a capture-safe block sets it on both projection stages before compile()
            capture_safe=bool(getattr(self, "capture_safe", False)),
        )

    def workspace_bytes(self) -> int:
        """Bytes the FROST GEMM needs. Its own, separate from the block's
        intermediates — the block reserves a region for it (contract § 10)."""
        if self._plan is None:
            raise RuntimeError("call compile() before workspace_bytes()")
        return self._plan.workspace_bytes

    def flops(self) -> int:
        """``2*M*N*K`` — the denominator for an MMA SOL number."""
        return 2 * self.m * self.n * self.k

    def execute(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        handle=None,
        alpha: Optional[torch.Tensor] = None,
        sf_a: Optional[torch.Tensor] = None,
        sf_w: Optional[torch.Tensor] = None,
        *,
        stream=None,
    ) -> None:
        """``sf_a`` / ``sf_w`` (block-scale plans only, both required): the PADDED
        F8_128x4 E8M0 blobs of ``a`` (over its M rows) and ``w`` (over its N rows).
        ``stream`` is the block's launch stream (a raw ``CUstream`` int); the
        runner carries it onto both GEMM routes -- see ``run_proj_gemm`` (Rule 5)."""
        from .kernels.proj_gemm import run_proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute()")
        if self.block_scale and (sf_a is None or sf_w is None):
            raise ValueError(f"{self.name}: block_scale=True needs both scale-factor blobs (sf_a=, sf_w=); no silent unit scale (Rule 1)")
        if not self.block_scale and (sf_a is not None or sf_w is not None):
            raise ValueError(f"{self.name}: this projection has no block-scale dequant; refusing to drop sf_a / sf_w silently")
        run_proj_gemm(self._plan, a, w, out, workspace, handle, alpha=alpha, sf_a=sf_a, sf_w=sf_w, stream=stream)


def _qkv_gate_projection(
    geom: GatedAttentionBlockGeometry,
    *,
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    out_dtype: Optional[torch.dtype] = None,
    alpha: bool = False,
    block_scale: bool = False,
    w_dtype: Optional[torch.dtype] = None,
) -> _Projection:
    """Stage (1). At the 397B full-attention layer: ``M x 17408 x 4096``.  ``w_dtype`` (default:
    ``dtype``) is the weight's dtype -- ``torch.float4_e2m1fn_x2`` for an MXFP4 ``W_qkvg`` against
    e4m3 ``h`` (the block-scale MIXED row; the E8M0 / 32 scale blob is unchanged)."""
    return _Projection(
        m=batch * seq_len,
        k=geom.d_model,
        n=geom.n_qkvg,
        dtype=dtype,
        name="qkv_gate_proj",
        out_dtype=out_dtype,
        alpha=alpha,
        block_scale=block_scale,
        w_dtype=w_dtype,
    )


def _out_projection(
    geom: GatedAttentionBlockGeometry,
    *,
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    out_dtype: Optional[torch.dtype] = None,
    alpha: bool = False,
    block_scale: bool = False,
    w_dtype: Optional[torch.dtype] = None,
    block_size: int = MXFP8_BLOCK_SIZE,
    sf_dtype=None,
) -> _Projection:
    """Stage (6). At the 397B full-attention layer: ``M x 4096 x 8192``.  ``block_scale`` + the fp4
    trio (``dtype=w_dtype=torch.float4_e2m1fn_x2``, ``block_size`` 16 | 32, ``sf_dtype`` E4M3 | E8M0)
    is the fp4 gated-O x fp4 ``W_o`` projection; the defaults are the per-tensor / bf16 path as before."""
    return _Projection(
        m=batch * seq_len,
        k=geom.h_q * geom.d_head,
        n=geom.d_model,
        dtype=dtype,
        name="out_proj",
        out_dtype=out_dtype,
        alpha=alpha,
        block_scale=block_scale,
        w_dtype=w_dtype,
        block_size=block_size,
        sf_dtype=sf_dtype,
    )


class _Quantize(_Stage):
    """(3q) / (5q) per-tensor FP8 cast: ``dst = sat_e4m3(src * scale)`` over ``[T, H, D]``.

    Kernel: ``kernels/quantize.py`` (a streaming pass in the mould of
    ``elementwise.py``; the source may be a strided slab column slice, the
    destination is compact, so for Q/K/V this stage IS the compaction).  The
    scale is a 1-element fp32 device tensor read in-kernel -- no host readback.
    It exists because the UNFUSED FP8 pipeline needs FP8 operands for the SDPA
    and the out projection while the norm+RoPE and gate kernels stay bf16.  The
    fully fused FP8 pipeline (``fuse_norm_rope`` + ``fuse_gate``) folds both
    casts into the producing epilogues and does not build this stage at all.
    """

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype_in: torch.dtype, heads: int, name: str) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.quantize import validate_shape

        if self.dtype_in not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: the quantize source must be bf16/f16, got {self.dtype_in}")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        from .kernels.quantize import compile_quantize

        self._recipe = compile_quantize(dtype_in=self.dtype_in, h=self.heads, d=self.geom.d_head, threads_per_cta=_ELEMENTWISE_THREADS)

    def moved_bytes(self) -> int:
        from .kernels.quantize import moved_bytes

        return moved_bytes(self.batch * self.seq_len, self.heads, self.geom.d_head, src_elem_bytes=_itemsize(self.dtype_in))

    def execute(self, src: torch.Tensor, dst: torch.Tensor, scale: torch.Tensor, current_stream=None) -> None:
        from .kernels.quantize import run_quantize

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_quantize(self._recipe, src, dst, scale, stream=stream)


class _QuantizeMxfp8(_Stage):
    """(3q, MXFP8) block quantize: bf16 ``[T, H, D]`` -> compact e4m3 ``[T, H, D]`` + an F8_128x4 E8M0 SF blob.

    Kernel: ``kernels/quantize_mxfp8.py``.  ONE stage class, TWO arms selected by
    ``axis`` -- ``"row"`` (Q / K: 32-element blocks along D, the BMM1 contraction;
    SF tile ``(b, h, s_tile)`` = 1024 contiguous bytes) and ``"col"`` (V: blocks
    along S, the BMM2 contraction; SF D-PLANE-MAJOR, plane stride
    ``B*H*n_tiles*512``) -- the exact byte layouts the production MXFP8 SDPA's
    ``_build_sf_desc`` reads (``mma-tma-matrix.md`` § 7).  The kernel writes EVERY
    SF byte of every 128-row tile (pad rows -> ``0x00``), so a KV tail is served
    exactly like the torch oracle pads it (D6).  The source may be a strided slab
    column slice; the destination is compact, so for Q/K/V this stage IS the
    compaction.  The fully fused MXFP8 pipeline folds all three into the
    projection fork's epilogue and builds none of them.

    **The canonical arm** (``sf_layout="gemm"``, appended; the block-scale GEMMs
    of the MXFP8 backward): the SF blob is cuDNN's PADDED F8_128x4 matrix over
    ``[rows, K]`` with the batch folded into the rows -- the layout
    ``build_proj_gemm`` declares for its SFA / SFB (``proj_gemm.sf_padded_dims``),
    sized by ``proj_gemm.sf_blob_bytes``, NOT by ``_sf_slot_bytes``.  Rowwise
    (``axis="row"``) it quantizes the ``[T, H, D]`` view of a ``[T, N]`` gradient
    (blocks along N; ``(rows, K) = (T, H*D)``: the dgrad's A operand); with
    ``transposed=True`` (``axis="col"`` only) it quantizes along the TOKENS and
    stores the codes PHYSICALLY TRANSPOSED as the contiguous e4m3 ``[H*D, T]``
    matrix (``(rows, K) = (H*D, T)``: the wgrad's K-major A operand), which needs
    ``T % 32 == 0`` (whole 32-token blocks).  ``moved_bytes`` is the same count in
    every mode; ``execute`` forwards the same operands (``dst`` is the ``[H*D, T]``
    matrix when transposed).

    **The packed arm** (``packed=True``, appended; the block's THD pipeline): the
    unit is a ``(head, packed tile)`` of the PER-SEQUENCE-TILE-PADDED SF layout the
    Rubin d256 MXFP8 SDPA's THD row reads (``kernels/quantize_mxfp8.py`` "PACKED
    sequences": Q/K slabs at ``(h*n_cap + tile) * 1024``, V's two D-planes ADJACENT
    inside the tile), the grid ``(H, n_cap)`` with ``n_cap = (T + 127 *
    num_sequences) // 128`` the capacity bound on ``sum_b ceil(s_b/128)``, every
    slack tile zero-filled; the kernel resolves each tile's sequence from the
    lengths tensor on device.  ``execute(..., seq_lens=)`` then takes that tensor
    (``[B]`` int32 lengths, or ``[B+1]`` prefix sums under ``cu_seqlens``) INSTEAD of
    ``batch`` / ``seq_len`` -- REQUIRED under ``packed``, refused otherwise (Rule 1,
    both ways; the kernel host checks the same).  ``sf_bytes()`` is then
    :func:`_sf_slot_bytes_packed`.  ``batch`` / ``seq_len`` stay the block's ``(1,
    T)`` so ``rows()`` / ``moved_bytes()`` read ``T``.
    """

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype_in: torch.dtype,
        heads: int,
        axis: str,
        name: str,
        sf_layout: str = "sdpa",
        transposed: bool = False,
        dual: bool = False,
        transposed_second: bool = False,
        packed: bool = False,
        num_sequences: int = 0,
        cu_seqlens: bool = False,
    ) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self.axis = str(axis)
        self.sf_layout = str(sf_layout)
        self.transposed = bool(transposed)
        # Appended: the DUAL-AXIS arm (``kernels/quantize_mxfp8.py``, "The dual-axis arm") -- the rowwise AND the columnwise
        # quantization from ONE read: under "sdpa" the second half is the row-major columnwise payload + the D-plane-major SF atoms
        # (the row's do8 + do_T8); under "gemm" it is the TRANSPOSED [H*D, T] store + the blob over (H*D, T) (dqkvg8 + dqkvg_t8),
        # ``transposed_second=True`` REQUIRED there (the canonical columnwise blob's one form).  ``axis`` must be "row" (the first
        # half); ``execute`` then takes ``dst_T`` / ``sf_T`` (appended) -- REQUIRED under ``dual``, refused otherwise (Rule 1).
        self.dual = bool(dual)
        self.transposed_second = bool(transposed_second)
        # Appended: the PACKED (THD) arm -- see the class docstring.  ``num_sequences`` is the packing's B (the lengths tensor has B
        # entries, or B+1 prefix sums under ``cu_seqlens``); the lengths themselves ride ``execute(seq_lens=)``.
        self.packed = bool(packed)
        self.num_sequences = int(num_sequences)
        self.cu_seqlens = bool(cu_seqlens)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.quantize_mxfp8 import (
            AXES,
            SF_LAYOUTS,
            validate_dual_mode,
            validate_dual_shape,
            validate_mode,
            validate_packed_mode,
            validate_shape,
        )

        if self.axis not in AXES:
            raise ValueError(f"{self.name}: axis must be one of {AXES} ('row' for Q/K, 'col' for V), got {self.axis!r}")
        if self.sf_layout not in SF_LAYOUTS:
            raise ValueError(f"{self.name}: sf_layout must be one of {SF_LAYOUTS}, got {self.sf_layout!r}")
        if self.transposed and (self.axis != "col" or self.sf_layout != "gemm"):
            raise ValueError(
                f"{self.name}: transposed=True is the columnwise arm's GEMM-canonical [H*D, T] store -- it needs axis='col' and sf_layout='gemm', "
                f"got axis={self.axis!r} sf_layout={self.sf_layout!r}"
            )
        if self.dtype_in not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: the quantize source must be bf16/f16, got {self.dtype_in}")
        if self.packed:
            # The packed arm serves the SDPA layouts only (no canonical blob, no transposed store) and needs the packing's B.
            validate_packed_mode(self.sf_layout, self.transposed, True)
            if self.num_sequences < 1:
                raise ValueError(
                    f"{self.name}: packed=True needs num_sequences >= 1 (the packing's B: the lengths tensor has B entries, or B+1 prefix sums "
                    f"under cu_seqlens), got {self.num_sequences}"
                )
        elif self.num_sequences or self.cu_seqlens:
            raise ValueError(f"{self.name}: num_sequences / cu_seqlens are the packed (THD) arm's fields (packed=True); this stage is dense")
        if self.dual:
            if self.axis != "row" or self.transposed:
                raise ValueError(
                    f"{self.name}: dual=True is the rowwise arm plus its columnwise twin from one read -- it needs axis='row' and transposed=False (the "
                    f"second half's form is transposed_second), got axis={self.axis!r} transposed={self.transposed}"
                )
            validate_dual_mode(self.sf_layout, self.transposed_second)
            validate_dual_shape(self.geom.d_head, _QUANTIZE_MXFP8_THREADS)
            return
        if self.transposed_second:
            raise ValueError(f"{self.name}: transposed_second=True is the dual arm's second-half form; it needs dual=True")
        validate_mode(self.axis, self.sf_layout, self.transposed)
        validate_shape(self.geom.d_head, _QUANTIZE_MXFP8_THREADS, self.axis)

    def compile(self) -> None:
        from .kernels.quantize_mxfp8 import compile_quantize_mxfp8, compile_quantize_mxfp8_dual

        if self.dual:
            self._recipe = compile_quantize_mxfp8_dual(
                dtype_in=self.dtype_in,
                h=self.heads,
                d=self.geom.d_head,
                threads_per_cta=_QUANTIZE_MXFP8_THREADS,
                sf_layout=self.sf_layout,
                transposed_second=self.transposed_second,
                packed=self.packed,
            )
            return
        self._recipe = compile_quantize_mxfp8(
            dtype_in=self.dtype_in,
            h=self.heads,
            d=self.geom.d_head,
            axis=self.axis,
            threads_per_cta=_QUANTIZE_MXFP8_THREADS,
            sf_layout=self.sf_layout,
            transposed=self.transposed,
            packed=self.packed,
        )

    def rows(self) -> int:
        return self.batch * self.seq_len

    def sf_bytes(self) -> int:
        """Bytes of the SF blob this stage writes, BY LAYOUT: ``"sdpa"`` -> ``_sf_slot_bytes`` (== the SDPA adapter's
        ``_reshape_sf`` count); ``"gemm"`` -> ``proj_gemm.sf_blob_bytes(T, H*D)`` rowwise / ``sf_blob_bytes(H*D, T)``
        transposed (the block-scale GEMM's padded F8_128x4 blob; the batch folds into the rows); under ``packed`` the
        capacity count :func:`_sf_slot_bytes_packed` (``T = rows()``)."""
        if self.sf_layout == "gemm":
            from .kernels.proj_gemm import sf_blob_bytes

            k = self.heads * self.geom.d_head
            return sf_blob_bytes(k, self.rows()) if self.transposed else sf_blob_bytes(self.rows(), k)
        if self.packed:
            return _sf_slot_bytes_packed(self.num_sequences, self.heads, self.rows(), self.geom.d_head)
        return _sf_slot_bytes(self.batch, self.heads, self.seq_len, self.geom.d_head)

    def sf_bytes_second(self) -> int:
        """Bytes of the dual arm's SECOND SF blob: the SDPA columnwise atoms (the same count as the rowwise tiles) under ``"sdpa"``,
        ``proj_gemm.sf_blob_bytes(H*D, T)`` (the transposed canonical blob) under ``"gemm"``.  Needs ``dual``."""
        if not self.dual:
            raise ValueError(f"{self.name}: sf_bytes_second() is the dual arm's second blob; this stage is dual=False")
        if self.sf_layout == "gemm":
            from .kernels.proj_gemm import sf_blob_bytes

            return sf_blob_bytes(self.heads * self.geom.d_head, self.rows())
        if self.packed:
            return _sf_slot_bytes_packed(self.num_sequences, self.heads, self.rows(), self.geom.d_head)
        return _sf_slot_bytes(self.batch, self.heads, self.seq_len, self.geom.d_head)

    def moved_bytes(self) -> int:
        """HBM traffic of one launch: 2 B in, 1 B code + 1/32 B SF out per element (the dual arm: one read, two codes + two SF)."""
        from .kernels.quantize_mxfp8 import moved_bytes, moved_bytes_dual

        f = moved_bytes_dual if self.dual else moved_bytes
        return f(self.batch * self.seq_len, self.heads, self.geom.d_head, src_elem_bytes=_itemsize(self.dtype_in))

    def execute(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        sf: torch.Tensor,
        *,
        batch: Optional[int] = None,
        seq_len: Optional[int] = None,
        current_stream=None,
        dst_T: Optional[torch.Tensor] = None,
        sf_T: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """``src`` ``[T, H, D]`` (strided ok), ``dst`` compact e4m3 ``[T, H, D]`` (the contiguous ``[H*D, T]`` matrix when
        ``transposed``), ``sf`` uint8 flat (``sf_bytes()`` bytes).  Under ``dual`` (appended) ``dst_T`` / ``sf_T`` are the second
        half's outputs -- a compact e4m3 ``[T, H, D]`` + the D-plane-major blob under ``"sdpa"``, the contiguous ``[H*D, T]`` matrix +
        ``sf_bytes_second()`` under ``"gemm"`` -- REQUIRED; without ``dual`` they are refused (Rule 1, both ways).

        Under ``packed`` (appended; THD) ``seq_lens`` -- the ``[B]`` int32 lengths, or the ``[B+1]`` int32 prefix sums under
        ``cu_seqlens``, on ``src``'s device -- REPLACES ``batch`` / ``seq_len`` (REQUIRED, and those two are refused; a dense stage
        refuses ``seq_lens``: Rule 1, both ways).  The lengths are validated for FORM only and never read on the host (CUDA-graph
        capturable): the caller's contract is every length in ``[0, max_seq_len]`` and ``sum(lengths) == T`` -- a violation writes
        payload rows past ``T`` (the SF writes are bounded by the slot's capacity in every case)."""
        from .kernels.quantize_mxfp8 import run_quantize_mxfp8, run_quantize_mxfp8_dual

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        if self.packed:
            if seq_lens is None:
                raise ValueError(
                    f"{self.name}: this stage is packed=True (THD): execute needs seq_lens (the [B] int32 lengths, or [B+1] int32 prefix sums under "
                    "cu_seqlens) in place of batch / seq_len"
                )
            if batch is not None or seq_len is not None:
                raise ValueError(
                    f"{self.name}: packed=True: batch / seq_len are the dense geometry; the packed stage runs over (T, num_sequences, seq_lens) (Rule 1)"
                )
            geo = dict(seq_lens=seq_lens, num_sequences=self.num_sequences, cu_seqlens=self.cu_seqlens)
        else:
            if seq_lens is not None:
                raise ValueError(f"{self.name}: seq_lens is the packed (THD) stage's lengths tensor; this stage is packed=False (Rule 1)")
            geo = dict(batch=self.batch if batch is None else int(batch), seq_len=self.seq_len if seq_len is None else int(seq_len))
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        if self.dual:
            if dst_T is None or sf_T is None:
                raise ValueError(f"{self.name}: this stage is dual=True (rowwise + columnwise from one read): dst_T and sf_T must be bound at execute (Rule 1)")
            run_quantize_mxfp8_dual(self._recipe, src, dst, sf, dst_T, sf_T, stream=stream, **geo)
            return
        if dst_T is not None or sf_T is not None:
            raise ValueError(f"{self.name}: this stage is dual=False; passing dst_T / sf_T would silently ignore them (Rule 1)")
        run_quantize_mxfp8(self._recipe, src, dst, sf, stream=stream, **geo)


class _QuantizeFp4(_Stage):
    """(5q') fp4 block quantize of the gated O: compact bf16/f16 ``[T, H_q, D]`` -> e2m1 codes ``[T, H_q*D/2]``
    (two per byte, bound as ``float4_e2m1fn_x2``) + the out-projection GEMM's PADDED F8_128x4 scale blob
    over ``(rows=T, K=H_q*D)``.

    Kernel: ``kernels/quantize_fp4.py``.  ONE stage class, TWO formats selected by ``fmt`` (the
    ``Fp4Format`` member, or its name): ``NVFP4`` -- e4m3 scale per 16 (``e4m3_rn(max(amax/6, 2^-9))``,
    codes by ``div.rn.f32``) -- and ``MXFP4`` -- E8M0 scale per 32 (``cvt.rp.ue8m0(amax/6)``, codes by the
    exact power-of-two reciprocal).  The blob is sized by ``proj_gemm.sf_blob_bytes`` (the GEMM contract,
    NOT the SDPA's ``_sf_slot_bytes``) and the kernel writes EVERY byte of it (pad rows ``0x00``), so the
    block-scale out projection reads it with no re-layout.  The source is the compact gated ``o``; the
    stage is what turns it into the fp4 A operand of ``o4 @ W_o^T``.  Built under ``MxQuantSpec.o_fp4`` (config
    rows 8-10) in ``quantize_o``'s place, on the unfused and the fully fused MXFP8 pipeline.

    ``scale_in`` (appended, default ``False`` = the forward's stage, byte-identical artifact): the kernel's
    pre-scale slot read -- every element is multiplied by a 1-element fp32 device slot BEFORE the block amax
    (``kernels/quantize_fp4.py``: the two-level NVFP4 cast of a GRADIENT, whose raw magnitude would otherwise fall
    under the e4m3 scale floor).  The quantized backward builds the NVFP4 cast of ``dY`` (viewed ``[T, d_model / D,
    D]``) with it, fed the live power-of-two ``scale_dy``; ``execute(scale_in=)`` is then REQUIRED and otherwise refused
    (Rule 1, both ways, here and in the kernel's host wrapper).
    """

    def __init__(
        self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype_in: torch.dtype, heads: int, fmt, name: str, scale_in: bool = False
    ) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype_in = dtype_in
        self.heads = int(heads)
        self.fmt = fmt
        self.scale_in = bool(scale_in)
        self._recipe = None

    def _format(self) -> tuple:
        """``(name, block, sf_e4m3)`` of ``fmt`` -- ``ValueError`` for anything but the two served formats."""
        from .kernels.quantize_fp4 import fp4_format

        return fp4_format(self.fmt)

    def check_support(self) -> None:
        from .kernels.quantize_fp4 import validate_shape

        _, block, _ = self._format()
        if self.dtype_in not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: the quantize source must be bf16/f16, got {self.dtype_in}")
        if self.geom.d_head % (4 * block):
            raise NotImplementedError(
                f"{self.name}: d_head={self.geom.d_head} is not a multiple of 4*block = {4 * block}, so one head's scales are not whole F8_128x4 atoms"
            )
        validate_shape(self.geom.d_head, _QUANTIZE_FP4_THREADS, block)

    def compile(self) -> None:
        from .kernels.quantize_fp4 import compile_quantize_fp4

        self._recipe = compile_quantize_fp4(
            dtype_in=self.dtype_in, h=self.heads, d=self.geom.d_head, fmt=self.fmt, threads_per_cta=_QUANTIZE_FP4_THREADS, scale_in=self.scale_in
        )

    def rows(self) -> int:
        return self.batch * self.seq_len

    def code_bytes(self) -> int:
        """Bytes of the packed e2m1 codes: ``T * H_q * D / 2``."""
        return self.rows() * self.heads * self.geom.d_head // 2

    def sf_bytes(self) -> int:
        """Bytes of the padded F8_128x4 blob this stage writes == what the block-scale out projection binds."""
        from .kernels.proj_gemm import sf_blob_bytes

        _, block, _ = self._format()
        return sf_blob_bytes(self.rows(), self.heads * self.geom.d_head, block)

    def moved_bytes(self) -> int:
        """HBM traffic of one launch: 2 B in, 1/2 B code + 1/block B SF out per element."""
        from .kernels.quantize_fp4 import moved_bytes

        _, block, _ = self._format()
        return moved_bytes(self.rows(), self.heads, self.geom.d_head, block, src_elem_bytes=_itemsize(self.dtype_in))

    def execute(self, src: torch.Tensor, dst4: torch.Tensor, sf: torch.Tensor, *, current_stream=None, scale_in: Optional[torch.Tensor] = None) -> None:
        """``src`` compact ``[T, H_q, D]``; ``dst4`` uint8 / ``float4_e2m1fn_x2`` of ``code_bytes()``; ``sf`` uint8 of ``sf_bytes()``;
        ``scale_in`` (appended) the 1-element fp32 device slot of a ``scale_in=True`` stage -- required then, refused otherwise (Rule 1)."""
        from .kernels.quantize_fp4 import run_quantize_fp4

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        if self.scale_in != (scale_in is not None):
            raise ValueError(
                f"{self.name}: scale_in "
                + (
                    "is required: this stage was declared with scale_in=True (the pre-scale of the two-level fp4 cast) and never assumes 1.0 -- pass the "
                    "slot's 1-element fp32 view"
                    if self.scale_in
                    else "was given but this stage was declared without scale_in (the forward's single-level cast); refusing to drop the value silently"
                )
            )
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_quantize_fp4(self._recipe, src, dst4, sf, stream=stream, scale_in=scale_in)


class _FusedQkvProjection(_Stage):
    """(1)+(2)+(3) in ONE kernel: the QKV+GATE projection with per-head RMSNorm
    and partial RoPE applied to the Q and K tiles INSIDE the GEMM epilogue.

    **This one IS a fork** -- ``kernels/proj_gemm_norm_rope.py``, the rendered
    shipped GEMM at the block's tile config plus an epilogue arm.  What makes the
    fusion tile-local, and therefore cheap, is the S1 layout contract: at
    ``d_head == 256`` one CTA output tile is exactly one head, so the RMSNorm
    reduction over D is thread-local (one epilogue thread owns one row of the
    tile) and the rotate_half partner sits in the same thread.  No new barrier,
    no new SMEM, no cross-CTA exchange; the kernel docstring has the design.

    It writes the SAME ``[T, N]`` slab the unfused chain norms in place, so the
    SDPA reads it exactly as under ``inplace_qkv`` -- which is why the fused
    path REQUIRES in-place: there is no pre-norm Q/K anywhere.  Training
    (``save_for_backward``) therefore keeps the unfused chain until the fork
    also writes ``q_pre``/``k_pre`` (the same trade every overwriting fusion in
    this block makes; see :class:`SavedForBackward`).  ``rstd`` IS emitted.

    Numerics: the norm runs on the fp32 ACCUMULATOR with one bf16 rounding,
    where the unfused chain norms the bf16-rounded projection -- the fused
    result is the more accurate one, so the oracle is the fp32 reference, never
    bit-identity with the pair.

    Rubin (SM107) only: the fork is a rendering for the sm_107a ``128x256``
    2-CTA config and its dtype constants are baked in -- ONE rendering per
    dtype family.  A different ``d_head`` or ``rope_dim`` the epilogue cannot
    tile is declined here, with the tile constraint in the message, before the
    template is ever loaded.

    **FP8 arm (``quant`` given, e4m3 ``h``/``W_qkvg``):** the second rendering,
    ``kernels/proj_gemm_norm_rope_fp8.py`` (selected by
    ``NormRopeFusionParams.quant_fp8``), whose epilogue also QUANTIZES: Q/K tiles
    are descaled (``alpha_qkvg``), normed + rotated on fp32 and written as e4m3
    (``* scale_q`` / ``* scale_k``) into COMPACT per-tensor ``q8`` / ``k8``
    buffers, V tiles as ``e4m3(alpha * acc * scale_v)`` into ``v8``, GATE tiles
    as bf16 into ``gate16`` -- four TMA-store descriptors, tile class -> buffer
    + column remap (round 2; round 1's single ``qkv8`` slab made the SDPA read
    strided and cost +7.2 % at 32K).  The four scalars ride as ONE fp32 ``[4]``
    device vector ``[alpha_qkvg, scale_q, scale_k, scale_v]`` (``qscal``)
    materialised in :meth:`compile`, read in-kernel -- never module-param floats.
    Inference only (``want_rstd`` must be False).  Launched through
    ``run_fused_proj_gemm_fp8`` (the bf16 runner is not overloaded).

    **MXFP8 arm (``quant`` is an :class:`MxQuantSpec`, e4m3 codes + F8_128x4 SF
    blobs for ``h`` / ``W_qkvg``):** the THIRD rendering, the block-scale twin
    ``kernels/proj_gemm_norm_rope_mxfp8.py`` (``NormRopeFusionParams.quant_mxfp8``;
    PR-B section 3.1), whose epilogue norms / rotates the DEQUANTIZED fp32
    accumulator (the E8M0 dequant happens in the MMA -- no alpha, no qscal) and
    BLOCK-quantizes: Q / K rowwise along D, V columnwise along S, e4m3 codes into
    compact ``q8`` / ``k8`` / ``v8`` and E8M0 scale factors into the SDPA's own
    F8_128x4 blobs ``sf_q`` / ``sf_k`` / ``sf_v`` (Q/K per-tile contiguous, V
    D-plane-major); GATE as bf16 ``gate16``.  Launched through
    ``run_fused_proj_gemm_mxfp8`` (frozen ABI, plan 3.1).  Feature-detected on
    the runner name + the fork file: until slice S6 lands the twin this arm is a
    typed ``NotImplementedError`` at ``check_support``, never a silently
    un-quantized path.  Declines ``S % 128 != 0 and B > 1`` (GEMM M-tiles
    straddle sequences; the unfused path serves it).
    """

    name = "qkv_gate_proj_norm_rope"
    _TILE_N = 256  # the rendering's per-CTA output width; one head per tile needs d_head == this
    _SUBTILE_N = 32  # epilogue subtile; rope_dim must be a whole number of subtile PAIRS
    _FORK_PATH_FP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "proj_gemm_norm_rope_fp8.py")
    _FORK_PATH_MXFP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "proj_gemm_norm_rope_mxfp8.py")
    _MXFP8_RUNNER = "run_fused_proj_gemm_mxfp8"
    _WEIGHT_FP4_FIELD = "weight_fp4"  # the NormRopeFusionParams field the fork's e2m1-B arm will append (feature-detected)

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_rstd: bool,
        norm_source: str = "ldg_early",
        quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,
        device=None,
        # APPENDED (serving): the paged KV-cache write-through FOLDED into this fork's epilogue -- the pools' tokens per page (0 = off,
        # the separate `_CacheWrite` launch).  bf16 fork only; the stage compiles both slot dtypes (int32 / int64 are the two the serving
        # stacks hand over; the dtype is a compile key, as in the separate kernel) so execute dispatches on a guaranteed cache hit.
        kv_pool_page_size: int = 0,
    ) -> None:
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.m = int(batch * seq_len)
        self.k = int(geometry.d_model)
        self.n = int(geometry.n_qkvg)
        self.dtype = dtype
        self.want_rstd = bool(want_rstd)
        self.norm_source = str(norm_source)
        self.kv_pool_page_size = int(kv_pool_page_size)
        self._plan_i64 = None  # the int64-slot twin of _plan under the write-through arm
        # FP8: the QuantSpec whose alpha_qkvg / scale_q / scale_k / scale_v the
        # fork's epilogue applies; `device` is where `qscal` is materialised.
        # MXFP8: an MxQuantSpec selects the block-scale twin (no scalars ride in).
        if quant is not None and not isinstance(quant, (QuantSpec, MxQuantSpec)):
            raise TypeError(f"{self.name}: quant must be a QuantSpec or an MxQuantSpec, got {type(quant).__name__}")
        self.quant = quant
        self.device = device
        self._plan = None
        self._qscal = None
        self._qscal_written = False
        self._qscal_ready = None  # the allocation stream's event, recorded in compile(); every write waits on it

    @property
    def fp8(self) -> bool:
        """The per-tensor FP8 fork (a ``QuantSpec``)."""
        return isinstance(self.quant, QuantSpec)

    @property
    def mxfp8(self) -> bool:
        """The block-scale MXFP8 fork twin (an ``MxQuantSpec``)."""
        return isinstance(self.quant, MxQuantSpec)

    def params(self):
        from .kernels.proj_gemm import NormRopeFusionParams

        g = self.geom
        kw = {}
        if self.fp8:
            # Only the FP8 arm names the field, so the bf16 compile key (and the
            # bf16 rendering's cache) is byte-identical to before the FP8 fork.
            kw["quant_fp8"] = True
        if self.mxfp8:
            # Same only-when-set idiom for the block-scale twin; a fork without
            # the field is a typed decline, never a norm-only bf16 artifact.
            if not self._fork_supports_field("quant_mxfp8"):
                raise NotImplementedError(f"{self.name}: {self._mxfp8_fork_available()}")
            kw["quant_mxfp8"] = True
            if self.quant.w_qkvg_fp4:
                # An e2m1 B is its own rendering (Uint8 B SMEM, the B4X16_P64 form, half the
                # expect-tx bytes); a fork without the field must never render an e4m3-B artifact
                # for an fp4 weight, so the same only-when-set idiom guards the key too.
                if not self._fork_supports_field(self._WEIGHT_FP4_FIELD):
                    raise NotImplementedError(self._weight_fp4_unsupported_msg())
                kw[self._WEIGHT_FP4_FIELD] = True
        if not g.qk_norm:
            # Same only-when-set idiom: norm-on keys are spelled identically to
            # today.  The field lands with the fork edits (PR-B slice S2); until
            # then a RoPE-only fused projection is a typed decline, never a
            # silently norm-ON artifact.
            if not self._fork_supports_qk_norm():
                raise NotImplementedError(self._qk_norm_off_unsupported_msg())
            kw["qk_norm"] = False
        if self.kv_pool_page_size:
            # Same only-when-set idiom: the write-through arm names its two fields (the page size; the slot dtype = this plan's, the int64
            # twin is built with kv_slot_i64=True in compile()); every key without the arm is spelled identically to today.
            kw["kv_pool_page_size"] = self.kv_pool_page_size
        return NormRopeFusionParams(
            d_head=g.d_head, rope_dim=g.rope_dim, h_q=g.h_q, h_kv=g.h_kv, eps=g.qk_norm_eps, want_rstd=self.want_rstd, norm_source=self.norm_source, **kw
        )

    @staticmethod
    def _fork_supports_qk_norm() -> bool:
        """True once ``NormRopeFusionParams`` carries ``qk_norm`` (the GEMM forks'
        RoPE-only epilogue, PR-B slice S2). Feature-detected so either landing
        order works: this slice can ship before or after the fork edits."""
        import dataclasses

        from .kernels import proj_gemm

        return "qk_norm" in {f.name for f in dataclasses.fields(proj_gemm.NormRopeFusionParams)}

    def _qk_norm_off_unsupported_msg(self) -> str:
        return (
            f"{self.name}: geometry.qk_norm=False (RoPE-only Q/K) with fuse_norm_rope=True needs the GEMM forks' RoPE-only "
            "epilogue (NormRopeFusionParams.qk_norm), which has not landed in this checkout. Use the unfused chain "
            "(fuse_norm_rope=False) for qk_norm=False."
        )

    def _fp8_fork_available(self) -> Optional[str]:
        """None when the FP8 fork + its runner ABI are present; else the reason they are not."""
        import dataclasses

        from .kernels import proj_gemm

        if "quant_fp8" not in {f.name for f in dataclasses.fields(proj_gemm.NormRopeFusionParams)}:
            return "NormRopeFusionParams has no `quant_fp8` field"
        if not hasattr(proj_gemm, "run_fused_proj_gemm_fp8"):
            return "kernels/proj_gemm.py has no `run_fused_proj_gemm_fp8`"
        if not self._runner_writes_compact_qkv():
            return "`run_fused_proj_gemm_fp8` still has the round-1 slab ABI (no `out_k8`): the compact q8/k8/v8 GEMM fork has not landed"
        if not os.path.exists(self._FORK_PATH_FP8):
            return f"{os.path.basename(self._FORK_PATH_FP8)} is not in kernels/"
        return None

    def _weight_fp4_unsupported_msg(self) -> str:
        return (
            f"{self.name}: the fused MXFP8 projection fork is rendered for an e4m3 B; W_qkvg is torch.float4_e2m1fn_x2 "
            f"(MxQuantSpec.w_qkvg_dtype) and NormRopeFusionParams has no `{self._WEIGHT_FP4_FIELD}` arm in this checkout. "
            "Use the unfused pipeline (fuse_norm_rope=False, fuse_gate=False): its block-scale GEMM serves the mixed MXFP8 x MXFP4 row."
        )

    @staticmethod
    def _fork_supports_field(field: str) -> bool:
        import dataclasses

        from .kernels import proj_gemm

        return field in {f.name for f in dataclasses.fields(proj_gemm.NormRopeFusionParams)}

    def _mxfp8_fork_available(self) -> Optional[str]:
        """None when the MXFP8 fork twin + its runner ABI (plan 3.1) are present; else the reason they are not.

        Feature-detected (the ``_fp8_fork_available`` idiom) so the block ships
        before the twin: ``NormRopeFusionParams.quant_mxfp8``, the runner
        ``run_fused_proj_gemm_mxfp8`` carrying the frozen SF-output ABI
        (``out_sf_q`` / ``out_sf_k`` / ``out_sf_v``), and the fork file."""
        import inspect

        from .kernels import proj_gemm

        if not self._fork_supports_field("quant_mxfp8"):
            return "NormRopeFusionParams has no `quant_mxfp8` field"
        fn = getattr(proj_gemm, self._MXFP8_RUNNER, None)
        if fn is None:
            return f"kernels/proj_gemm.py has no `{self._MXFP8_RUNNER}` (the MXFP8 GEMM fork twin, PR-B slice S6, has not landed)"
        params = inspect.signature(fn).parameters
        if not {"out_sf_q", "out_sf_k", "out_sf_v", "sf_a", "sf_w"} <= set(params):
            return f"`{self._MXFP8_RUNNER}` does not carry the frozen PR-B 3.1 ABI (sf_a, sf_w, out_sf_q/k/v)"
        if not os.path.exists(self._FORK_PATH_MXFP8):
            return f"{os.path.basename(self._FORK_PATH_MXFP8)} is not in kernels/"
        return None

    @staticmethod
    def _runner_writes_compact_qkv() -> bool:
        """True when the FP8 runner carries the round-2 ABI
        ``(plan, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, qscal, *, stream)``.
        Round 1 took one ``out_qkv8`` slab; handing it the compact buffers
        positionally would bind ``out_k8`` as the gate, so the block declines
        (check_support) and refuses (execute_fp8) rather than mis-binding."""
        import inspect

        from .kernels import proj_gemm

        fn = getattr(proj_gemm, "run_fused_proj_gemm_fp8", None)
        return fn is not None and "out_k8" in inspect.signature(fn).parameters

    def check_support(self) -> None:
        from .kernels.proj_gemm import validate_norm_rope_params

        g = self.geom
        # Geometry first, so the declines below read the same on every device.
        if g.index_band:
            # The fork classifies every output tile by the band its first column falls in and renders _TILE_N-wide
            # tiles; a 640-column indexer band is 2.5 of them, so its last tile would straddle the band's end.  The
            # UNFUSED projection (the FROST GEMM over any N) serves the band; a band padded to whole tiles is the
            # fork's own later arm, measured before it ships.
            pad = -(-g.qsa.index_band_cols // self._TILE_N) * self._TILE_N
            raise NotImplementedError(
                f"{self.name}: the fused projection renders {self._TILE_N}-column tiles; a {g.qsa.index_band_cols}-column indexer band "
                f"(geometry.qsa.index_band) is not a whole number of them (N={g.n_qkvg} = {g.n_qkvg / self._TILE_N:g} tiles). Use the "
                f"unfused projection (fuse_norm_rope=False), or a band padded to {pad} columns once that arm exists."
            )
        if g.d_head != self._TILE_N:
            raise NotImplementedError(
                f"{self.name}: the fused epilogue needs one GEMM output tile == one head, i.e. d_head == {self._TILE_N} "
                f"(the rendering's per-CTA N tile); got d_head={g.d_head}. Use the unfused chain (fuse_norm_rope=False)."
            )
        if g.rope_dim % (2 * self._SUBTILE_N) or not 0 < g.rope_dim < g.d_head:
            raise NotImplementedError(
                f"{self.name}: rope_dim must be a positive multiple of {2 * self._SUBTILE_N} and < d_head so each rotate_half pair "
                f"spans whole epilogue subtiles; got rope_dim={g.rope_dim}"
            )
        if not g.qk_norm:
            # RoPE-only epilogue: no RMSNorm, so no rstd can be wanted, and the
            # fork must know the knob (typed decline until a later slice lands it).
            if self.want_rstd:
                raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and emits no rstd; want_rstd must be False")
            if not self._fork_supports_qk_norm():
                raise NotImplementedError(self._qk_norm_off_unsupported_msg())
        if self.fp8:
            if self.dtype != torch.float8_e4m3fn:
                raise NotImplementedError(f"{self.name}: the FP8 fork is rendered for e4m3 h / W_qkvg, got {self.dtype}")
            if self.k % 16:
                # TMA 16-byte rule at 1 B/elem (mirrors _Projection's FP8 gate).
                raise NotImplementedError(f"{self.name}: FP8 needs K % 16 == 0 (TMA 16-byte rule at 1 B/elem), got K={self.k}")
            if self.want_rstd:
                raise NotImplementedError(f"{self.name}: the FP8 fork is inference-only (no rstd output)")
            missing = self._fp8_fork_available()
            if missing is not None:
                raise NotImplementedError(f"{self.name}: the FP8 fused projection fork has not landed in this checkout ({missing})")
        elif self.mxfp8:
            if self.dtype != torch.float8_e4m3fn:
                raise NotImplementedError(f"{self.name}: the MXFP8 fork twin is rendered for e4m3 codes, got {self.dtype}")
            if self.quant.w_qkvg_fp4 and not self._fork_supports_field(self._WEIGHT_FP4_FIELD):
                # Config row 11: MXFP4 W_qkvg + fuse_norm_rope.  Feature-detected like
                # `quant_mxfp8`, so this decline INVERTS the day the fork's e2m1-B arm lands.
                raise NotImplementedError(self._weight_fp4_unsupported_msg())
            if self.k % MXFP8_BLOCK_SIZE:
                raise NotImplementedError(f"{self.name}: MXFP8 needs K % {MXFP8_BLOCK_SIZE} == 0 (one E8M0 scale per block), got K={self.k}")
            if self.want_rstd:
                raise NotImplementedError(f"{self.name}: the MXFP8 fork twin is inference-only (no rstd output)")
            if self.seq_len % _SF_TILE_ROWS and self.batch > 1:
                # The fork's SF stores are decoded from the flat GEMM row; a 128-row
                # M-tile straddling two sequences would have to scatter its SF
                # atom across two (b, s_tile) units.  v1 declines (plan 3.1 / Q9).
                raise NotImplementedError(
                    f"{self.name}: the fully fused MXFP8 pipeline needs S % {_SF_TILE_ROWS} == 0 when B > 1 (a GEMM M-tile must not straddle two "
                    f"sequences' scale-factor tiles); got B={self.batch}, S={self.seq_len}. Use the unfused MXFP8 pipeline."
                )
            missing = self._mxfp8_fork_available()
            if missing is not None:
                raise NotImplementedError(f"{self.name}: the MXFP8 fused projection fork twin has not landed in this checkout ({missing})")
        elif self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: the fork is rendered for bf16 only (e4m3 needs a QuantSpec / MxQuantSpec), got {self.dtype}")
        validate_norm_rope_params(self.params())
        if torch.cuda.is_available():
            cc = tuple(torch.cuda.get_device_capability())
            if cc != _SM107_CC:
                raise NotImplementedError(f"{self.name}: rendered for sm_107a (Rubin); this device is SM{cc[0]}{cc[1]}")

    def compile(self) -> None:
        from .kernels.proj_gemm import build_fused_proj_gemm

        # params() itself declines qk_norm=False on a fork without the knob, so
        # a compile() reached without check_support() cannot build a norm-ON
        # artifact for a norm-OFF geometry.
        self._plan = build_fused_proj_gemm(self.params())
        if self.kv_pool_page_size:
            # The int64-slot twin (the slot dtype is a compile key): both artifacts exist before the first execute, so whichever dtype
            # the serving stack hands over dispatches on a cache hit (Rule 4) -- the separate `_CacheWrite` stage's own contract.
            import dataclasses as _dc

            self._plan_i64 = build_fused_proj_gemm(_dc.replace(self.params(), kv_slot_i64=True))
        if self.fp8:
            # The fork reads these four once per epilogue warp from a device vector the execute path never allocates or converts.
            # ALLOCATED here and written on a launch stream only (write_qscal: the block's first execute or update_quant_scales, or
            # this stage's own first execute_fp8): a fill enqueued on the stream ambient at compile() would have no ordering against
            # a later cross-stream recalibration -- pending work ahead of it could let it land after the update and restore the
            # declared scales.
            dev = self.device if self.device is not None else torch.device("cuda")
            self._qscal = torch.empty(4, dtype=torch.float32, device=dev)
            self._qscal_written = False
            # The allocation is not free of stream work either: under torch.use_deterministic_algorithms(True) (with
            # fill_uninitialized_memory, the default) torch.empty enqueues a NaN fill on the ambient stream.  Record it, so every
            # writer orders itself behind whatever the allocation enqueued (write_qscal waits on this event).
            self._qscal_ready = torch.cuda.Event()
            self._qscal_ready.record(torch.cuda.current_stream(dev))

    def workspace_bytes(self) -> int:
        """None: no split-K, no scratch -- the kernel writes the slab directly."""
        return 0

    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def execute(self, a, w, out, w_q_norm, w_k_norm, cos, sin, rstd_q=None, rstd_k=None, *, stream, k_cache=None, v_cache=None, slot_mapping=None) -> None:
        """bf16 arm: ONE ``[M, N]`` slab out (Q/K columns normed + rotated).

        ``w_q_norm`` / ``w_k_norm`` are ``None`` (both) under ``geometry.qk_norm=False``
        and tensors otherwise -- checked here, both directions, before the runner.
        ``k_cache`` / ``v_cache`` / ``slot_mapping`` (appended): the write-through arm's pools and slot map -- REQUIRED on a stage declared
        with ``kv_pool_page_size`` (the epilogue writes the post-RoPE K / V rows there too), REFUSED otherwise; the slot dtype picks the
        artifact (int32 / int64, both compiled)."""
        from .kernels.proj_gemm import run_fused_proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute()")
        if self.fp8:
            raise ValueError(f"{self.name}: this stage was declared FP8; use execute_fp8(...)")
        if self.mxfp8:
            raise ValueError(f"{self.name}: this stage was declared MXFP8; use execute_mxfp8(...)")
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        plan = self._plan
        if self.kv_pool_page_size:
            if slot_mapping is None:
                raise ValueError(
                    f"{self.name}: this stage folds the paged KV-cache write-through into its epilogue; k_cache, v_cache and slot_mapping are required"
                )
            if slot_mapping.dtype == torch.int64:
                plan = self._plan_i64
            elif slot_mapping.dtype != torch.int32:
                raise ValueError(f"{self.name}: slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
        run_fused_proj_gemm(
            plan, a, w, out, w_q_norm, w_k_norm, cos, sin, rstd_q, rstd_k, stream=stream, k_cache=k_cache, v_cache=v_cache, slot_mapping=slot_mapping
        )

    def execute_fp8(self, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, *, stream) -> None:
        """FP8 arm: COMPACT e4m3 ``q8 [M, h_q*d]`` / ``k8 [M, h_kv*d]`` / ``v8 [M, h_kv*d]``
        + bf16 ``gate16 [M, h_q*d]`` out (all 2-D, contiguous; the SDPA views the
        same bytes ``[B, S, H, d]``).

        The positional order is the runner's leading order on purpose --
        an internal probe script captures these arguments and
        replays them through ``run_fused_proj_gemm_fp8(plan, *args, qscal, **kw)``.
        ``a`` is the e4m3 ``[M, K]`` view of ``h`` (the runner binds it rank-3
        ``[1, M, K]`` exactly as the bf16 runner does), ``w`` the e4m3
        ``[N_qkvg, K]`` checkpoint-layout weight; ``qscal`` is the fp32 ``[4]``
        ``[alpha_qkvg, scale_q, scale_k, scale_v]`` allocated in :meth:`compile` and
        written on the launch stream (:meth:`write_qscal`; the stage's own first
        use writes the declared spec's values when nobody wrote them before).
        """
        from .kernels.proj_gemm import run_fused_proj_gemm_fp8

        if self._plan is None or self._qscal is None:
            raise RuntimeError("call compile() before execute_fp8()")
        if not self.fp8:
            raise ValueError(f"{self.name}: this stage was declared bf16; use execute(...)")
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        if not self._runner_writes_compact_qkv():
            # Never hand the round-1 slab runner three compact buffers positionally.
            raise NotImplementedError(f"{self.name}: {self._fp8_fork_available()}")
        if not self._qscal_written:
            self.write_qscal(self.quant, stream)  # a standalone first use: the declared values, on this launch stream
        run_fused_proj_gemm_fp8(self._plan, a, w, out_q8, out_k8, out_v8, out_gate16, w_q_norm, w_k_norm, cos, sin, self._qscal, stream=stream)

    def write_qscal(self, spec, stream) -> None:
        """Write ``spec``'s ``[alpha_qkvg, scale_q, scale_k, scale_v]`` into the fork's fp32 vector IN PLACE on ``stream`` (a raw
        CUstream handle: the launch stream, so the next execute on it is ordered behind the write) and make ``spec`` the stage's spec.
        The ONLY writer of that vector -- :meth:`compile` allocates it and never fills it -- so no compile-time write can land after a
        later cross-stream recalibration and restore the declared scales; the write first waits on the allocation stream's event
        (:meth:`compile` records it), so even a deterministic-mode fill of the allocation lands before it.  Inside a stream capture
        the wait is skipped (an external event cannot be waited on there): warm the block up before capturing."""
        if self._qscal is None:
            raise RuntimeError("call compile() before write_qscal()")
        ts = as_torch_stream(stream, self._qscal.device)
        with torch.cuda.stream(ts):
            if self._qscal_ready is not None and not torch.cuda.is_current_stream_capturing():
                ts.wait_event(self._qscal_ready)  # the allocation's own stream work (a deterministic-mode fill) lands first
            for i, v in enumerate((spec.alpha_qkvg, spec.scale_q, spec.scale_k, spec.scale_v)):
                self._qscal[i : i + 1].fill_(float(v))
        self.quant = spec
        self._qscal_written = True

    def execute_mxfp8(
        self, a, sf_a, w, sf_w, out_q8, out_k8, out_v8, out_gate16, out_sf_q, out_sf_k, out_sf_v, w_q_norm, w_k_norm, cos, sin, *, stream
    ) -> None:
        """MXFP8 arm (frozen runner ABI, PR-B plan 3.1): e4m3 codes ``a [M, K]`` + its
        F8_128x4 blob ``sf_a``, ``w [N_qkvg, K]`` + ``sf_w``; COMPACT e4m3 ``q8`` / ``k8`` /
        ``v8`` + their SDPA F8_128x4 blobs ``sf_q`` / ``sf_k`` / ``sf_v`` (Q/K rowwise
        per-(b, h, s_tile) 1024-B tiles; V columnwise D-plane-major) + bf16 ``gate16`` out.
        ``batch`` / ``seq_len`` ride as keywords: the SF tiles are decoded per sequence."""
        from .kernels import proj_gemm

        if self._plan is None:
            raise RuntimeError("call compile() before execute_mxfp8()")
        if not self.mxfp8:
            raise ValueError(f"{self.name}: this stage was not declared MXFP8; use execute(...) / execute_fp8(...)")
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        missing = self._mxfp8_fork_available()
        if missing is not None:
            raise NotImplementedError(f"{self.name}: {missing}")
        getattr(proj_gemm, self._MXFP8_RUNNER)(
            self._plan,
            a,
            sf_a,
            w,
            sf_w,
            out_q8,
            out_k8,
            out_v8,
            out_gate16,
            out_sf_q,
            out_sf_k,
            out_sf_v,
            w_q_norm,
            w_k_norm,
            cos,
            sin,
            batch=self.batch,
            seq_len=self.seq_len,
            stream=stream,
        )


class _QkNormRope(_Stage):
    """(2)+(3) per-head RMSNorm over D then partial RoPE, on Q and K, ONE kernel.

    **V is not normed and is not touched here.** Q and K ride one launch over a
    flat row space, so K's 1/16-of-Q traffic costs no second launch and the
    cos/sin tables stay hot across a token's heads.

    Emits ``rstd_q`` / ``rstd_k`` (fp32, one per (token, head)) when the block
    saves for backward — 134 MiB + 8 MiB at 1M tokens, cheap enough that
    recomputing them would be the odd choice.

    Norm and rotation both run in fp32 with a SINGLE rounding at the end. An
    unfused torch chain rounds twice, so this is a slightly different (and more
    accurate) function; the oracle is written to match
    (``reference.qk_norm_rope_reference``).

    **Fusion status: this stage IS the fusion.** Stages (2)+(3) were the
    largest single stage at every sequence length in the torch baseline, which
    is why they were built before the projections. Measured on Rubin at the 397B
    geometry: **18.9x the torch chain, taking the stage from 36-53% of the block
    to 1.5-2.6%** — and **36.7% of a cold 10777 GB/s copy ceiling** at the
    shipped knobs, so there is real headroom left. Being pure bandwidth, the
    harness reports achieved GB/s and fraction-of-ceiling, never TFLOP/s.

    **Quote no fraction for this stage that was not L2-FLUSHED.** The earlier
    "58%" was hot-cache and against another node's ceiling; re-measured cold on
    2026-09-10 it is 36.7%. The kernel docstring carries the full table and the
    register-pressure diagnosis; an internal probe script reproduces
    it with hot and cold columns side by side.

    Q/K may be updated IN PLACE (``q_out is q``): every lane reads its whole row
    before any lane stores, and the RoPE partner shuffle stays inside the row's
    own lane group.

    **``geometry.qk_norm=False`` -- RoPE only.** Both kernels fold the RMSNorm
    out at trace time (``compile_qk_norm_rope[_tma](apply_norm=False)``: the
    weight slots are traced as ``None``, exactly the presence switch ``want_rstd``
    uses), so there is no sum-of-squares pass, no rsqrt, no weight load and no
    rstd; the dims ``[rope_dim, d_head)`` come out BIT-EXACT. The stage name and
    position do not change (``"qk_norm_rope"`` stays in ``_stages``); ``execute``
    takes ``None`` for both weights and refuses tensors, both directions typed.

    Kernel: ``kernels/qk_norm_rope.py`` — at ``kernels/`` level, not under an
    arch package, because it is plain vectorized LDG/STG with no tcgen05 and no
    arch-specific path (engine contract § 8).
    """

    name = "qk_norm_rope"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        want_rstd: bool,
        threads_per_cta: int = _QK_NORM_ROPE_THREADS,
        rows_per_group: int = _QK_NORM_ROPE_ROWS_PER_GROUP,
        defer_secondary_loads: bool = _QK_NORM_ROPE_DEFER_SECONDARY_LOADS,
        const_head_counts: bool = True,
        impl: str = "auto",
        tile_rows: Optional[int] = None,  # None = fit it to the geometry
        stages: int = _QK_NORM_ROPE_STAGES,
        use_pdl: bool = False,
        dynamic_token_stride: bool = True,
        page_size: int = 0,  # APPENDED: > 0 = the pool-write arm (the paged K / V write-through folded into this launch; the TMA kernel only)
    ) -> None:
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.want_rstd = bool(want_rstd)
        self.page_size = int(page_size)
        self._recipes = None  # the pool-write arm: {slot dtype: recipe} (both slot widths compiled, Rule 4)
        self.threads_per_cta = int(threads_per_cta)
        self.rows_per_group = int(rows_per_group)
        self.defer_secondary_loads = bool(defer_secondary_loads)
        self.const_head_counts = bool(const_head_counts)
        self.impl = str(impl)
        self.tile_rows = None if tile_rows is None else int(tile_rows)
        self.stages = int(stages)
        self.use_pdl = bool(use_pdl)
        self.dynamic_token_stride = bool(dynamic_token_stride)
        self._recipe = None
        self._impl = None

    def resolve_tile_rows(self) -> int:
        """Rows per TMA tile: the caller's value, or the best one for this shape.

        A TMA tile is ``tile_rows`` rows of the ``[T, H, D]`` operand, so the
        tiling is only expressible when ``tile_rows`` divides ``h_q``, is a
        multiple of ``h_kv``, and spreads evenly over the CTA's warps. The
        measured optimum on Rubin is 16, but ``h_q`` is 8 on plenty of models and
        16 simply does not tile there.

        So ``tile_rows=None`` (the default) FITS it: the largest legal value not
        exceeding the measured preference. That is a selection, like
        ``impl="auto"`` -- an EXPLICIT ``tile_rows`` that does not fit raises
        instead of being quietly replaced.
        """
        g = self.geom
        warps = max(1, self.threads_per_cta // 32)
        legal = lambda r: r > 0 and g.h_q % r == 0 and r % g.h_kv == 0 and r % warps == 0
        if self.tile_rows is not None:
            if not legal(self.tile_rows):
                raise NotImplementedError(
                    f"tile_rows={self.tile_rows} cannot tile this geometry: it must divide h_q={g.h_q}, "
                    f"be a multiple of h_kv={g.h_kv}, and spread over {warps} warps"
                )
            return self.tile_rows
        fits = [r for r in range(_QK_NORM_ROPE_TILE_ROWS, 0, -1) if legal(r)]
        return fits[0] if fits else 0

    def resolve_impl(self) -> str:
        """Which of the two kernels this stage will run: ``"ldg"`` or ``"tma"``.

        Both are kept on purpose. They compute the SAME function bit-for-bit
        (asserted by ``test_qk_norm_rope.py``), and each owns an arch:

        * **tma** stages every tile through SMEM with a TMA ring. It needs
          ``cp.async.bulk.tensor``, i.e. **SM90 or newer**, and it is the faster
          kernel wherever it runs -- +45.7% on the Rubin dev node, +55.7% on the
          perf node, both cold and against the LDG kernel that already carries
          the deferred-load fix.
        * **ldg** is plain vectorized global load/store with no SMEM staging. It
          is the ONLY option on SM80, which has no TMA at all, and it is the
          fallback for any geometry the TMA tiling cannot express.

        ``impl="auto"`` picks tma when the arch AND the geometry allow it. That
        is a SELECTION, not a silent capability fallback: asking for ``"tma"``
        explicitly on a part or a shape that cannot serve it RAISES rather than
        quietly running the other kernel (Rule 1).
        """
        if self.impl not in ("auto", "ldg", "tma"):
            raise ValueError(f'impl must be one of "auto", "ldg", "tma"; got {self.impl!r}')
        if self.impl == "ldg":
            return "ldg"
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import ambient_device, compute_capability

        from .kernels.qk_norm_rope_tma import validate_shape as _tma_validate

        g = self.geom
        major, minor = compute_capability(ambient_device())
        arch_ok = major >= 9  # cp.async.bulk.tensor exists from Hopper on
        # OUTSIDE the try: an EXPLICIT tile_rows that does not fit is a knob the
        # caller asked for and we cannot honor, so it must propagate. Only the
        # fitted value is allowed to decide "tma is not expressible here".
        tile_rows = self.resolve_tile_rows()
        try:
            _tma_validate(g.d_head, g.rope_dim, g.h_q, g.h_kv, tile_rows, self.threads_per_cta)
            shape_ok, why = True, ""
        except (ValueError, NotImplementedError) as exc:
            shape_ok, why = False, str(exc)
        if self.impl == "tma":
            if not arch_ok:
                raise NotImplementedError(f"impl='tma' needs sm_90 or newer for cp.async.bulk.tensor; this device is sm_{major}{minor}")
            if not shape_ok:
                raise NotImplementedError(f"impl='tma' cannot tile this geometry: {why}")
            return "tma"
        return "tma" if (arch_ok and shape_ok) else "ldg"

    def check_support(self) -> None:
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"qk_norm_rope serves bf16/f16 only, got {self.dtype}")
        g = self.geom
        if not g.qk_norm:
            # RoPE-only: nothing to norm, so nothing to emit an rstd from, and
            # with no RoPE either the stage would be an identity copy.
            if self.want_rstd:
                raise ValueError(f"{self.name}: geometry.qk_norm=False computes no RMSNorm and emits no rstd; want_rstd must be False")
            if g.rope_dim == 0:
                raise ValueError(f"{self.name}: geometry.qk_norm=False with rope_dim=0 is an identity copy of Q/K; drop the stage instead")
        if self.resolve_impl() == "tma":
            from .kernels.qk_norm_rope_tma import validate_shape as _tma_validate

            g = self.geom
            _tma_validate(g.d_head, g.rope_dim, g.h_q, g.h_kv, self.resolve_tile_rows(), self.threads_per_cta)
            if self.page_size:
                from .kernels.cache_write import validate_cache_write_shape

                validate_cache_write_shape(g.d_head, self.page_size, self.threads_per_cta)
        elif self.page_size:
            raise NotImplementedError(
                f"{self.name}: the paged write-through folded into the norm + RoPE launch is an arm of the TMA kernel (sm_90+, a geometry its tiling "
                f"fits), and the dense DECODE form's unfused chain is served on cc 10.7, where that kernel runs; this device resolves the stage to "
                f"the LDG kernel, which has no such arm -- declare fuse_cache_write_small_m=False (the separate cache-write launch)"
            )
        else:
            from .kernels.qk_norm_rope import validate_shape

            validate_shape(self.geom.d_head, self.geom.rope_dim, self.threads_per_cta)

    def compile(self) -> None:
        g = self.geom
        self._impl = self.resolve_impl()
        if self._impl == "tma":
            from .kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma

            common = dict(
                dtype=self.dtype,
                h_q=g.h_q,
                h_kv=g.h_kv,
                d=g.d_head,
                rope_dim=g.rope_dim,
                eps=g.qk_norm_eps,
                want_rstd=self.want_rstd,
                tile_rows=self.resolve_tile_rows(),
                stages=self.stages,
                threads_per_cta=self.threads_per_cta,
                apply_norm=g.qk_norm,
                use_pdl=self.use_pdl,
            )
            if self.page_size:
                # the pool-write arm: both slot widths compiled (int32 and int64 are the two the serving stacks hand over; the dtype is a
                # compile key), so execute dispatches on a guaranteed cache hit whichever one arrives (the separate stage's rule)
                self._recipes = {
                    dt: compile_qk_norm_rope_tma(pool_write=True, page_size=self.page_size, slot_dtype=dt, **common) for dt in (torch.int32, torch.int64)
                }
                self._recipe = self._recipes[torch.int32]
            else:
                self._recipe = compile_qk_norm_rope_tma(**common)
            return
        if self.page_size:
            raise NotImplementedError(f"{self.name}: the pool-write arm exists on the TMA kernel only (check_support names it); impl={self._impl!r}")
        from .kernels.qk_norm_rope import compile_qk_norm_rope

        self._recipe = compile_qk_norm_rope(
            dtype=self.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d=g.d_head,
            rope_dim=g.rope_dim,
            eps=g.qk_norm_eps,
            want_rstd=self.want_rstd,
            threads_per_cta=self.threads_per_cta,
            rows_per_group=self.rows_per_group,
            defer_secondary_loads=self.defer_secondary_loads,
            const_head_counts=self.const_head_counts,
            use_pdl=self.use_pdl,
            dynamic_token_stride=self.dynamic_token_stride,
            apply_norm=g.qk_norm,
        )

    def moved_bytes(self) -> int:
        """HBM traffic of one launch — the denominator for the SOL number."""
        from .kernels.qk_norm_rope import moved_bytes

        g = self.geom
        t = self.batch * self.seq_len
        total = moved_bytes(t, g.h_q, g.h_kv, g.d_head, elem_bytes=2, want_rstd=self.want_rstd)
        if self.page_size:
            # the pool-write arm: the V band read once, the K and V rows written to the pools, one slot per token (int32 counted)
            total += 3 * t * g.h_kv * g.d_head * 2 + t * 4
        return total

    def execute(
        self,
        q: torch.Tensor,  # [B, S, H_q,  D]
        k: torch.Tensor,  # [B, S, H_kv, D]
        w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],  # [D]
        cos: torch.Tensor,  # [B, S, ROPE_DIM]
        sin: torch.Tensor,  # [B, S, ROPE_DIM]
        q_out: Optional[torch.Tensor] = None,  # defaults to in place
        k_out: Optional[torch.Tensor] = None,
        rstd_q: Optional[torch.Tensor] = None,  # [B, S, H_q]  fp32
        rstd_k: Optional[torch.Tensor] = None,  # [B, S, H_kv] fp32
        current_stream=None,
        flat: bool = False,  # accepted and ignored: rank decides
        # APPENDED (the pool-write arm, page_size > 0): the V band [B, S, H_kv, D] / [T, H_kv, D], the two pools [P, H_kv, page_size, D]
        # and the slot map [T] -- REQUIRED there, refused otherwise (both directions typed).
        v: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> None:
        """Flatten ``[B, S, ...]`` to ``[T, ...]`` and launch.

        ``.view()`` only — never ``reshape``: these buffers are compact by
        construction (§ 1), so a view is exact and a copy would be a silent
        extra kernel (Rule 1).

        The norm weights must agree with ``geometry.qk_norm`` in both directions
        (typed ``ValueError`` naming the knob) -- checked here, before the recipe.
        """
        from .kernels.qk_norm_rope import run_qk_norm_rope
        from .kernels.qk_norm_rope_tma import run_qk_norm_rope_tma

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        g = self.geom
        _check_norm_weights_agree(g.qk_norm, w_q_norm, w_k_norm)
        # The token rows: a flat [T, H, D] q names them (the block's strided slab views -- the LIVE rows under num_tokens_live); a
        # compact [B, S, H, D] q is the declared geometry.
        t = int(q.shape[0]) if q.ndim == 3 else self.batch * self.seq_len
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(q.device).cuda_stream

        def _flat(x, h):
            """Accept ``[B, S, H, D]`` or an already-flat ``[T, H, D]``.

            The block hands STRIDED ``[T, H, D]`` views of the fused projection,
            which cannot be ``.view()``-ed; a standalone caller hands compact
            ``[B, S, H, D]``. Never ``reshape``: it may copy (Rule 1).
            """
            if x is None:
                return None
            return x if x.ndim == 3 else x.view(t, h, g.d_head)

        pool_kw = {}
        recipe = self._recipe
        if self.page_size:
            if v is None or k_cache is None or v_cache is None or slot_mapping is None:
                raise ValueError(f"{self.name}: the pool-write arm needs v, k_cache, v_cache and slot_mapping at every execute (Rule 1: no silent fallback)")
            if slot_mapping.dtype not in self._recipes:
                raise ValueError(f"{self.name}: slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
            recipe = self._recipes[slot_mapping.dtype]
            pool_kw = dict(v=_flat(v, g.h_kv), k_cache=k_cache, v_cache=v_cache, slot_mapping=slot_mapping)
        elif v is not None or k_cache is not None or v_cache is not None or slot_mapping is not None:
            raise ValueError(
                f"{self.name}: v / k_cache / v_cache / slot_mapping belong to the pool-write arm (page_size > 0); this stage would silently ignore them"
            )
        runner = run_qk_norm_rope_tma if self._impl == "tma" else run_qk_norm_rope
        runner(
            recipe,
            _flat(q, g.h_q),
            _flat(k, g.h_kv),
            _flat(q if q_out is None else q_out, g.h_q),
            _flat(k if k_out is None else k_out, g.h_kv),
            w_q_norm,
            w_k_norm,
            cos.view(t, g.rope_dim),  # t rows explicitly (a rope_dim 0 table is [T, 0]: -1 could not infer it)
            sin.view(t, g.rope_dim),
            None if rstd_q is None else rstd_q.view(t, g.h_q),
            None if rstd_k is None else rstd_k.view(t, g.h_kv),
            stream=stream,
            **pool_kw,
        )


# Rubin. The block is Rubin-only for now: the SDPA flavor it needs
# ((256, 256) f16) exists on SM100 too, but nothing here has been validated
# there, and serving an unvalidated arch silently is worse than declining it.
_SM107_CC = (10, 7)


def _bhsd_desc(b: int, h: int, s: int, d: int, dtype: torch.dtype, device, name: str, token_stride: int = 0) -> TensorDesc:
    """Descriptor for a BSHD-COMPACT buffer presented as logical BHSD.

    The SDPA's operand contract is rank-4 ``(B, H, S, D)``; the block's buffers
    are ``[B, S, H, D]`` compact, so what it hands over is ``.transpose(1, 2)``
    — strides ``(S*H*D, D, H*D, 1)``. That is exactly the layout the SDPA's own
    prepared binder accepts directly: the pointer stays unchanged and the
    BSHD strides go to the kernel without a copy. Stage (1) writes compact
    buffers to keep that zero-copy handover explicit — see § 1.

    Built as a descriptor rather than from a sample tensor so declaring the
    block costs no device allocation.
    """
    # token_stride = 0 means COMPACT (h*d). A larger value declares a column
    # slice of a wider row -- Q/K/V inside the fused projection, token stride
    # n_qkvg. The SDPA compiles its descriptors at whatever this declares and
    # binds the view directly (config_sm100.dense_bind_strides), so a padded
    # stride costs no copy; what it must not do is OVERLAP, hence the >= check.
    ts = int(token_stride) if token_stride else h * d
    if ts < h * d:
        raise ValueError(f"{name}: token_stride {ts} is smaller than h*d={h*d}; that would alias distinct rows")
    shape = (b, h, s, d)
    stride = (s * ts, d, ts, 1)
    stride_order = tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))
    return TensorDesc(dtype=dtype, shape=shape, stride=stride, stride_order=stride_order, device=device, name=name)


def _lse_desc(b: int, h: int, s: int, device, name: str = "lse") -> TensorDesc:
    """Descriptor for the ``[B, H_q, S]`` fp32 log-sum-exp, head-major compact."""
    shape = (b, h, s)
    stride = (h * s, s, 1)
    stride_order = tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))
    return TensorDesc(dtype=torch.float32, shape=shape, stride=stride, stride_order=stride_order, device=device, name=name)


def _thd_lse_head_stride(t_total: int) -> int:
    """The PACKED (THD) LSE's head stride: exactly the packed token total ``T``.

    ONE definition for both directions of the training boundary.  The forward declares its THD Stats descriptor with it
    (:func:`_thd_lse_desc`) and the backward hands it to the SDPA backward as the packed Stats head stride, so ``saved.lse``
    is the contiguous ``[1, H_q, T]`` fp32 tensor on both sides -- the dense record's ``[B, H_q, S]`` at ``B = 1, S = T``,
    never a rounded-up capacity one side spells and the other does not.
    """
    return int(t_total)


def _thd_lse_desc(b_env: int, h: int, s_max: int, t_total: int, device, name: str = "lse") -> TensorDesc:
    """Descriptor for the PACKED (THD) ``[1, H_q, T]`` fp32 log-sum-exp, as declared to the SDPA adapter.

    Shape ``(B_env, H_q, S_max)`` -- the adapter checks the Stats declaration against the operands' envelope shape -- with
    strides ``(H_q*T, T, 1)``: the token axis contiguous and the head stride exactly ``T`` (:func:`_thd_lse_head_stride`),
    which the adapter classifies as the head-major packed Stats layout and binds at execute as the caller's contiguous
    ``[1, H_q, T]`` tensor (head stride == the declared one, ``numel >= H_q * T``).  The batch stride is never read under THD.
    """
    hs = _thd_lse_head_stride(t_total)
    shape = (b_env, h, s_max)
    stride = (h * hs, hs, 1)
    stride_order = tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))
    return TensorDesc(dtype=torch.float32, shape=shape, stride=stride, stride_order=stride_order, device=device, name=name)


class _Sdpa(_Stage):
    """(4) ``O = softmax(Q K^T * scale + mask) V``, GQA-broadcast H_q/H_kv.

    **This stage writes no kernel, in ANY configuration.** It drives the shipped
    FROST forward (``cudnn.sdpa.fwd.api_dsl.SdpaFwdDslSm100``), which routes to
    the SM107 sibling kernels automatically from the live device —
    ``rubin=(self._device_cc == (10, 7))`` in ``SdpaFwdDslSm100.compile`` — so the
    ``(256, 256)`` flavor Qwen3.5 needs lands on
    ``sdpa/fwd/kernels/sm107/prefill_d256_f16.py`` (bf16 / f16) or, under
    ``pertensor_fp8=True``, ``prefill_d256_fp8.py`` (e4m3).  There is no flag to
    pass; there IS an arch to check, which :meth:`check_support` does.

    **``fuse_gate=True`` is the SAME adapter with a gate descriptor.**  The
    sigmoid-gate epilogue -- a TMA-staged read of GATE and
    ``O := O * sigmoid(GATE)`` AFTER the dead-row select, per element -- is a
    production feature of both d256 kernels behind
    ``TemplateParams.epilogue_gate`` (``SdpaFwdDslSm100.template_params()``
    sets it from ``sample_gate``; the engine rows claim it through
    ``Capabilities.epilogue_gate_d_shapes``).  Declaring the stage with
    ``sample_gate=<GATE descriptor>`` selects that specialization and
    ``execute(gate=...)`` binds the tensor.  GATE may be a column slice of the
    ``[T, N]`` slab (``gate_token_stride``): the adapter compiles the gate's
    TMA descriptor at the declared stride exactly as it does Q/K/V, so no copy
    happens anywhere -- and stage (5) disappears.

    **MXFP8** (``mxfp8=True``, e4m3 codes + F8_128x4 E8M0 scale factors): the
    SAME adapter with ``pertensor_fp8=False`` -> ``sm107/prefill_d256_mxfp8.py``
    (the production block-scale kernel; the per-tensor fork machinery is gone,
    so nothing can route block-scaled data into the per-tensor kernel).  The
    three SF blobs ride ``execute(sf_q=, sf_k=, sf_v=)`` and are REQUIRED;
    ``descale_q/k/v`` / ``scale_o`` are REFUSED (the adapter would silently
    ignore them -- the E8M0 dequant is in-MMA and a gated e4m3 O is UNSCALED,
    D8).  ``sched_policy`` is read off the MXFP8 row
    (``engine_name(arch, mxfp8=True)``: NATURAL at (256, 256) -- the per-tensor
    row's LPT claim does NOT transfer, D5) and ``cta_mma`` is left to the
    adapter (2 on a dense unsplit leg, 1 under THD / paged / split-KV;
    ``config_sm100.d256_mxfp8_pair_leg``).  ``has_amax_o=False`` as under FP8.
    Under ``thd=True`` the three blobs are the row's PACKED per-sequence-tile-padded
    layout at the block's slot CAPACITY (``_sf_slot_bytes_packed``: ``n_cap`` tiles
    per head, K and V equal) -- the adapter's binder derives the tile extent from
    the byte count, so each view handed over is the WHOLE slot, never the live
    total; the block's packed ``quantize_mxfp8`` stages write every byte of it.

    **FP8** (``dtype == e4m3``): ``pertensor_fp8=True``, ``dtype_o`` (bf16 on the
    unfused pipeline, e4m3 on the fully fused one) and ``has_amax_o=False`` --
    the block runs static per-tensor scales and never reads ``Amax_O``, so the
    kernel's atomicMax is compiled out rather than written into a slot nobody
    reads.  ``descale_q/k/v`` and (e4m3 O only) ``scale_o`` ride ``execute``;
    the kernel folds ``descale_v * scale_o`` into ``inv_sum`` in-kernel.  The
    block feeds it COMPACT ``q8`` / ``k8`` / ``v8`` (``token_stride == 0``):
    round 1's strided slab reads cost +7.2 % on the SDPA at 32K -- K/V lines
    refetched at a 9216 B stride once the gate stream evicts them -- where
    compact reads are -3.5 % (STATUS.md "SDPA decomposition").

    Three things settled here rather than inherited:

    * **``sched_policy`` is chosen EXPLICITLY, and it is LPT at d=256 -- bf16
      AND FP8.** `sdpa_fwd_prefill_sm107` (PR #1001) and, since 2026-09-11,
      `sdpa_fwd_prefill_sm107_fp8` advertise ``{NATURAL, LPT}`` for the
      (256, 256) flavor this block uses. ``_sched_policy`` reads the live
      device's row and picks LPT exactly where that row claims it, NATURAL for
      anything else -- gated on the d-shape the row actually claims rather than
      on "is Rubin". Stated rather than left to the standalone wrapper's ``None``
      derivation, which still excludes Rubin wholesale and would quietly give
      the win back.
    * **LSE is optional here and mandatory for training.** Prefill inference
      never reads it, so a stats-less block pays nothing; ``save_for_backward``
      implies it, because the SDPA backward cannot run without it.
    * **Mask arms are ``const_expr``-folded**, so a dense PASS proves nothing
      about the causal/window path. Validate at least one config per arm.

    ``seq_lens`` is the per-batch valid KV length (dense padding mask), or --
    under ``thd=True`` -- the per-sequence PACKED lengths (``[B]`` int32, or
    ``[B+1]`` int32 prefix sums under ``cu_seqlens``), handed to the adapter as
    BOTH ``seq_q_lens`` and ``seq_kv_lens`` (self-attention; the adapter requires
    both under THD).  **THD / varlen is the adapter's own ragged arm**: the stage
    declares the ``(num_sequences, H, max_seq_len, D)`` ENVELOPE at the block's
    token strides (the batch stride is never read -- every sequence base comes
    from the on-device prefix sums), ``thd=True`` with the packed totals ``T``,
    the head-major packed LSE ``[1, H_q, T]`` with head stride exactly ``T``
    (:func:`_thd_lse_desc`) and ``SCHED_NATURAL`` -- the varlen grid is the
    kernel's own persistent claim counter over the live (sequence, q-tile, head)
    units, and the LPT decodes assume a dense rectangular tile space -- and at
    ``execute`` hands RANK-3 packed ``(T, H, D)`` views (a slab column slice at
    token stride ``n_qkvg``, or the compact buffers) that the adapter binds
    without a copy; the dense arm's rank-4 ``(1, H, T, D)`` transpose is REFUSED
    there once more than one sequence runs.  Dense Q-side trimming
    (``seq_q_lens``) is NOT wired -- a decline, not a silent no-op.
    """

    name = "sdpa"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        want_lse: bool,
        seq_lens_present: bool = False,
        token_stride: int = 0,
        fuse_gate: bool = False,
        gate_token_stride: int = 0,
        o_dtype: Optional[torch.dtype] = None,
        gate_dtype: Optional[torch.dtype] = None,
        mxfp8: bool = False,
        thd: bool = False,  # packed sequences: batch=1, seq_len=T; the adapter's varlen arm (see the class docstring)
        num_sequences: Optional[int] = None,  # THD: B, the length tensor's [B] (or [B+1] prefix-sum) entries -- the envelope's batch
        max_seq_len: Optional[int] = None,  # THD: S_max, the longest sequence the plan admits -- the envelope's S
        cu_seqlens: bool = False,  # THD: seq_lens is [B+1] int32 prefix sums (True) or [B] int32 lengths (False)
        # APPENDED (serving, the EXTEND form): K / V come from PAGED POOLS instead of the block's own operands.  The stage then
        # declares the adapter's paged THD leg -- the pools' descriptors (their layout kind and strides are compiled in), the
        # page size, the block table's declared strides, the logical S_kv = max_pages x page_size -- and execute() takes the
        # pools, the block table and the per-sequence logical KV lengths.  Only the sm107 f16 row's paged THD leg (bf16 / f16,
        # unsplit, no gate, no LSE) is bound here; everything else is a typed decline of the stage itself.
        kv_source: str = KV_SOURCE_SELF,
        paged_page_size: int = 0,  # kv_source="paged": the pools' tokens per page (the block's paged_kv_page_size)
        k_cache_desc: Optional[TensorDesc] = None,  # kv_source="paged": the K pool's [num_pages, H_kv, P, D] descriptor (shape + strides)
        v_cache_desc: Optional[TensorDesc] = None,  # kv_source="paged": the V pool's descriptor
        block_table_desc: Optional[TensorDesc] = None,  # kv_source="paged": the [num_sequences, max_pages] int32 block table's descriptor (row stride free)
        # APPENDED (the DECODE form = kv_source="paged" on a DENSE declaration): the logical KV bound (0 = max_pages x page_size)
        # and the split-KV the form lowers with (>= 2 on the decode tile routes fuse_gate into the split combine).
        max_kv_len: int = 0,
        decode_split_kv: int = 1,
        # APPENDED (the DECODE form's route): the record the block read (decode_route under its policy).  On a dense paged stage it
        # replaces the stage's own reading (the default policy); on a THD paged stage it declares the DECODE form's PREFILL route --
        # the EXTEND form's binding at the dense step's (B, S_q) envelope with the group packed -- and the compiled template is pinned
        # against it either way.
        route: Optional[DecodeRoute] = None,
        # APPENDED: the adapter's programmatic-dependent-launch form (SdpaFwdDslSm100(use_pdl=True): the decode tile and its split combine
        # launch with the attribute and carry the griddepcontrol pair).  Performance-only.  The block hands it to a decode-tile route only
        # (``pdl_capable``); a standalone stage that forces it onto a kernel without the sites (the prefill tiles) hears the adapter's
        # typed decline at compile().
        use_pdl: bool = False,
    ) -> None:
        # token_stride != 0 => Q/K/V are column slices of the fused projection
        # and are read in place at that stride (the adapter compiles its TMA
        # descriptors at the declared strides).  0 => the compact buffers.
        self.token_stride = int(token_stride)
        # FP8 (dtype == e4m3): per-tensor descales (and scale_o) ride execute();
        # O comes out in `o_dtype` -- bf16 on the unfused pipeline, e4m3 on the
        # fully fused one.  `fp8` = the fp8-CLASS dtype; `mxfp8` selects the
        # block-scale kernel (pertensor_fp8=False), `pertensor` the scalar one.
        self.o_dtype = o_dtype if o_dtype is not None else dtype
        self.fp8 = dtype == torch.float8_e4m3fn
        self.mxfp8 = bool(mxfp8)
        if self.mxfp8 and not self.fp8:
            raise ValueError(f"{self.name}: mxfp8=True needs e4m3 Q/K/V codes, got dtype={dtype}")
        # fuse_gate => the kernel's epilogue_gate specialization reads GATE (a
        # slab column slice at gate_token_stride; 0 = compact like O) in
        # `gate_dtype` -- the block's activation dtype; None = Q's dtype -- and
        # writes O gated in place of O.
        self.fuse_gate = bool(fuse_gate)
        self.gate_token_stride = int(gate_token_stride)
        self.gate_dtype = gate_dtype
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = device
        self.want_lse = bool(want_lse)
        self.seq_lens_present = bool(seq_lens_present)
        # THD (packed sequences).  The block validates the whole contract (typed, in order) before building this stage;
        # the checks here keep the STAGE honest on its own: the envelope must be declared, and the one feature whose
        # SDPA specialization has no THD arm (the dense padding mask) is declined by name rather than left to the
        # adapter's later message.  The epilogue gate rides the THD arm of the f16/bf16 d256 prefill (the SDPA's THD gate
        # descriptor: G declared at the same (B_env, H, S_max, D) envelope as O, at the slab's gate column stride, bound
        # as the packed (T, H, D) view); the quantized THD gate is the adapter's own typed decline.  MXFP8 rides: the
        # row's THD arm reads the packed per-sequence-tile-padded scale-factor layout the block's packed quantize stages
        # write (class docstring).
        self.thd = bool(thd)
        self.cu_seqlens = bool(cu_seqlens)
        self.num_sequences = None if num_sequences is None else int(num_sequences)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        if self.thd:
            if self.num_sequences is None or self.max_seq_len is None or self.num_sequences < 1 or self.max_seq_len < 1:
                raise ValueError(f"{self.name}: thd=True needs num_sequences >= 1 and max_seq_len >= 1 (the (B, H, S_max, D) envelope the adapter declares)")
            if self.seq_lens_present:
                raise ValueError(f"{self.name}: thd=True and seq_lens_present=True are mutually exclusive (the packed lengths ARE the per-sequence lengths)")
        elif self.num_sequences is not None or self.max_seq_len is not None or self.cu_seqlens:
            raise ValueError(f"{self.name}: num_sequences / max_seq_len / cu_seqlens are THD-only (thd=True)")
        # The EXTEND form (kv_source="paged"): the stage's own honesty about the ONE leg it binds -- the adapter's paged THD
        # leg of the f16 / bf16 d256 flavor: packed queries, pools with the page size compiled in, the fused gate (the shared
        # SM100 d256 body compiled for cc 10.7 carries the gate seams on its paged / THD arms; the adapter's paged x gate twin
        # admits it and the prepared THD binder binds G per call), no LSE (the form writes none), no quantized Q / K / V (the
        # FP8 THD path clamps runtime descriptors to a packed total and serves no pools).  The block's declaration-time checks
        # say the same in the caller's vocabulary first; these keep a standalone stage from binding a leg nobody validated.
        if kv_source not in _KV_SOURCES:
            raise ValueError(f"{self.name}: kv_source must be one of {_KV_SOURCES}, got {kv_source!r}")
        self.kv_source = kv_source
        self.paged_page_size = int(paged_page_size)
        self.k_cache_desc, self.v_cache_desc, self.block_table_desc = k_cache_desc, v_cache_desc, block_table_desc
        # The DECODE form (kv_source="paged" on a DENSE declaration): the adapter declared PAGED with dense Q [B, S_q, H_q, D] (a
        # view of the slab at the token stride, as on the dense arm), the pools at the CALLER'S strides, the table's strides, the
        # logical KV bound, PackGQA over the whole group and the split the form lowers with; the route (which kernel serves S_q at
        # this group) is the adapter's own rule read up front (decode_route) and pinned against the compiled module.  The LENGTH
        # BINDING is one fact on the KV side only: execute hands kv_lens as seq_kv_lens and NOTHING as seq_q_lens (uniform S_q:
        # every Q row of every sequence is live; a padded entry is a live row with ZERO keys -> the tile's empty-range SELECT ->
        # O = 0 / LSE = -inf) -- never the dense block's self-attention binding, which would read a KV length as a Q length.
        # fuse_gate rides the tile's split COMBINE (the tile has no gate seams): with split_kv >= 2 the adapter compiles the tile
        # UNGATED and the shared combine multiplies the merged fp32 value by sigmoid(G) before the single cast; an unsplit tile
        # gates in its own launch, which the block decides (GatedAttentionBlockFwd.gate_fused) -- fuse_gate=True here = the combine.
        self.max_kv_len = int(max_kv_len)
        self.decode_split_kv = int(decode_split_kv)
        self.use_pdl = bool(use_pdl)
        self.decode_route: Optional[DecodeRoute] = route
        self.paged_dense = self.kv_source == KV_SOURCE_PAGED and not self.thd
        if self.kv_source == KV_SOURCE_PAGED:
            if self.paged_page_size <= 0 or k_cache_desc is None or v_cache_desc is None or block_table_desc is None:
                raise ValueError(f"{self.name}: kv_source='paged' needs paged_page_size > 0 and the k_cache / v_cache / block_table descriptors")
            if self.want_lse:
                raise NotImplementedError(f"{self.name}: kv_source='paged' writes no LSE (the serving forms over the pools are inference forms)")
            if self.fp8:
                raise NotImplementedError(f"{self.name}: kv_source='paged' serves bf16 / f16 operands and pools only (no quantized paged leg)")
            if self.decode_route is not None and self.decode_route.s_q != self.seq_len:
                raise ValueError(f"{self.name}: route is the record of S_q={self.decode_route.s_q}, this stage's step is S_q={self.seq_len}")
            if self.thd:
                # fuse_gate rides the paged THD leg: the gate is declared at the SAME (B_env, H, S_max, D) envelope as O at the
                # slab's GATE column stride and bound as the packed (T, H, D) view at execute -- the dense THD arm's contract.
                if self.decode_route is not None:
                    # The DECODE form on the PREFILL route: the EXTEND form's binding (packed queries over the pools) at the dense step's
                    # own envelope -- num_sequences = B, max_seq_len = S_q, the group packed, unsplit -- with the route pinned at compile.
                    if not self.decode_route.prefill_tile:
                        raise ValueError(
                            f"{self.name}: a THD paged stage carries the DECODE form on its prefill route only; the route of S_q={self.seq_len} at group "
                            f"{self.decode_route.group} is {self.decode_route.tile!r} (declare the dense paged stage for the decode tile)"
                        )
                    if (self.num_sequences, self.max_seq_len) != (self.batch, self.seq_len):
                        raise ValueError(
                            f"{self.name}: the prefill route binds the dense step's own envelope (num_sequences=B={self.batch}, max_seq_len=S_q={self.seq_len}), "
                            f"got ({self.num_sequences}, {self.max_seq_len})"
                        )
                    if self.decode_split_kv != 1:
                        raise NotImplementedError(
                            f"{self.name}: the prefill route runs the paged d256 body unsplit; got decode_split_kv={self.decode_split_kv}"
                        )
            else:
                if self.decode_split_kv < 1:
                    raise ValueError(f"{self.name}: decode_split_kv must be >= 1, got {decode_split_kv}")
                self.seq_lens_present = True  # the per-sequence KV lengths are mandatory over pools (the adapter's paged contract)
                if self.decode_route is None:
                    self.decode_route = decode_route(geometry, self.seq_len, batch=self.batch)
                if not self.decode_route.served:
                    raise NotImplementedError(f"{self.name}: {self.decode_route.reason}")
                if self.decode_route.prefill_tile:
                    raise NotImplementedError(
                        f"{self.name}: the DECODE route of S_q={self.seq_len} at group {self.decode_route.group} is the paged d256 prefill tile (packed "
                        "queries): a dense paged stage cannot bind it -- declare the stage thd=True at (num_sequences=B, max_seq_len=S_q) with this "
                        "route, as GatedAttentionBlockFwd does"
                    )
                if self.fuse_gate and not (self.decode_route.decode_tile and self.decode_split_kv >= 2):
                    raise NotImplementedError(
                        f"{self.name}: fuse_gate=True over the pools rides the decode tile's split combine only (split_kv >= 2; got split_kv="
                        f"{self.decode_split_kv}); the block gates in its own launch when the tile runs unsplit"
                    )
        elif (
            self.paged_page_size
            or k_cache_desc is not None
            or v_cache_desc is not None
            or block_table_desc is not None
            or max_kv_len
            or decode_split_kv != 1
            or route is not None
        ):
            raise ValueError(
                f"{self.name}: paged_page_size / k_cache_desc / v_cache_desc / block_table_desc / max_kv_len / decode_split_kv / route belong to kv_source='paged'"
            )
        self._impl = None

    @staticmethod
    def pdl_capable(route: Optional[DecodeRoute]) -> bool:
        """Whether stage (4) takes the programmatic-dependent-launch switch on ``route``: the Rubin d256 decode tile and its split
        combine carry the griddepcontrol pair (``DecodeRoute.decode_tile``); the paged d256 prefill tile (the DECODE form's prefill
        route, the EXTEND form -- ``route`` None) and the dense prefill tiles (``kv_source="self"``: no route) do not, so the block
        launches them plain and the link serializes -- correct by PDL's semantics (the successor's wait blocks on full completion).
        ``DecodeRoute.pdl_capable`` is the same fact on the record."""
        return route is not None and route.decode_tile

    def _build_impl(self):

        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.api_dsl import SdpaFwdDslSm100

        g = self.geom
        b, s, d = self.batch, self.seq_len, g.d_head
        ts = self.token_stride
        # has_amax_o=False: static scales, no Amax_O consumer -> the FP8 kernel
        # compiles the atomicMax out (and execute() refuses an amax_o tensor).
        # pertensor_fp8 picks the kernel FAMILY: True -> prefill_d256_fp8.py
        # (scalar descales), False -> prefill_d256_mxfp8.py (block scales).
        kw = dict(pertensor_fp8=self.pertensor, dtype_o=self.o_dtype, has_amax_o=False) if self.fp8 else {}
        if self.use_pdl:
            # a decode-tile route (the block never hands the switch to a prefill tile: pdl_capable); the adapter declines at compile()
            # when a standalone stage forces it onto a kernel without the griddepcontrol sites
            kw["use_pdl"] = True
        if self.thd:
            # THD: the adapter takes the (B_env, H, S_max, D) ENVELOPE at the block's token strides -- the same `_bhsd_desc`
            # the dense arm declares, at (num_sequences, max_seq_len) instead of (1, T); its (d, h, s) stride order is what
            # the adapter's packed-layout gate reads, the batch stride never is.  The packed totals TIGHTEN the token
            # extents to T (the adapter min's them against the bound buffers' capacity), the lengths' FORM is a
            # declaration fact (n_q_lens = B or B+1), and the Stats descriptor is the head-major packed LSE whose head
            # stride is exactly T (one helper, both directions).  NATURAL, stated: the varlen grid is the kernel's
            # persistent claim counter over the live units; LPT's decodes assume a dense rectangular tile space.
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import SCHED_NATURAL

            t = self.batch * self.seq_len
            b_env, s_env = self.num_sequences, self.max_seq_len
            lse = _thd_lse_desc(b_env, g.h_q, s_env, t, self.device) if self.want_lse else None
            sched = SCHED_NATURAL
            if self.kv_source == KV_SOURCE_PAGED:
                # The EXTEND form: the adapter's PAGED THD leg.  Q is the packed envelope as above; K / V are the caller's POOLS
                # (the sample descriptors as declared: [num_pages, H_kv, P, D] with the pool's own strides -- HND / NHD /
                # a unified row's half -- which the adapter compiles in and matches at execute); the logical S_kv the block
                # tables address is max_pages x P; the block table's declared strides let a row slice of a wider table bind as
                # a view (the same tuple for the V table: the block hands ONE table for both pools); the Q lengths keep the
                # block's own form (lengths or prefix sums), the KV lengths are the serving stack's [B] logical lengths
                # (never a prefix form); per-sequence KV lengths are REQUIRED by the paged leg (seq_kv_lens_present=True --
                # the adapter forces it under THD anyway, after its paged gate reads it).  No packed KV total: pools have no
                # packed capacity to clamp a runtime descriptor to.
                bt = self.block_table_desc
                table_stride = tuple(int(x) for x in bt.stride)
                kw.update(
                    thd=True,
                    max_total_seq_len_q=t,
                    cu_seq_q_lens=self.cu_seqlens,
                    cu_seq_kv_lens=False,
                    paged_page_size=self.paged_page_size,
                    paged_max_seq_len_kv=int(bt.shape[1]) * self.paged_page_size,
                    paged_table_stride=table_stride,
                    paged_table_v_stride=table_stride,
                )
                if self.decode_route is not None and self.decode_route.pack_gqa:
                    # The DECODE form on the PREFILL route: the whole group packed into the body's 128-row Q tile (HEADS_PER_TILE = the
                    # group: one Q tile per sequence and KV head, the KV streamed once); the EXTEND form keeps the adapter's default.
                    kw["pack_gqa"] = True
                if self.fuse_gate:
                    # The gate descriptor selects the epilogue_gate specialization of the paged body (template_params().epilogue_gate):
                    # G at the slab's GATE column stride, the (B_env, H, S_max, D) envelope O takes, mapped by the adapter through the
                    # packed [1, T, H, D] form Q / O take under THD.
                    kw["sample_gate"] = _bhsd_desc(
                        b_env, g.h_q, s_env, d, self.gate_dtype or self.dtype, self.device, "gate", token_stride=self.gate_token_stride
                    )
                return SdpaFwdDslSm100(
                    _bhsd_desc(b_env, g.h_q, s_env, d, self.dtype, self.device, "q", token_stride=ts),
                    dataclasses.replace(self.k_cache_desc, name="k"),
                    dataclasses.replace(self.v_cache_desc, name="v"),
                    _bhsd_desc(b_env, g.h_q, s_env, d, self.o_dtype, self.device, "o"),
                    lse,
                    is_causal=g.is_causal,
                    causal_bottom_right=g.causal_bottom_right,
                    window_size_left=None if g.window_left < 0 else g.window_left,
                    window_size_right=None if g.window_right < 0 else g.window_right,
                    scale_softmax=g.scale,
                    seq_kv_lens_present=True,
                    sched_policy=sched,
                    **kw,
                )
            kw.update(thd=True, max_total_seq_len_q=t, max_total_seq_len_kv=t, cu_seq_q_lens=self.cu_seqlens, cu_seq_kv_lens=self.cu_seqlens)
        else:
            b_env, s_env = b, s
            lse = _lse_desc(b, g.h_q, s, self.device) if self.want_lse else None
            sched = self._sched_policy()
        if self.paged_dense:
            # The DECODE form: the pools AS DECLARED (the caller's strides -> the TMA descriptors), ONE table for both pools (its
            # strides declared for K and V, or the adapter's row-major default refuses a row-sliced table at execute), the logical
            # KV bound, the whole group packed on the decode tile (MHA unpacked), the form's split.  seq_kv_lens_present is True.
            bt = self.block_table_desc
            table_stride = tuple(int(x) for x in bt.stride)
            k_desc = dataclasses.replace(self.k_cache_desc, name="k")
            v_desc = dataclasses.replace(self.v_cache_desc, name="v")
            kw.update(
                paged_page_size=self.paged_page_size,
                paged_max_seq_len_kv=self.max_kv_len or int(bt.shape[1]) * self.paged_page_size,
                paged_table_stride=table_stride,
                paged_table_v_stride=table_stride,
                split_kv=self.decode_split_kv,
                pack_gqa=True if self.decode_route.pack_gqa else None,
                # the route's width cap (decode_q_tile_max): the adapter re-reads the route rule at compile, so the cap it was read under
                # travels with it -- the compiled width is the declared one (pinned in compile() below)
                decode_q_tile_max=self.decode_route.q_tile_max,
            )
        else:
            k_desc = _bhsd_desc(b_env, g.h_kv, s_env, d, self.dtype, self.device, "k", token_stride=ts)
            v_desc = _bhsd_desc(b_env, g.h_kv, s_env, d, self.dtype, self.device, "v", token_stride=ts)
        if self.fuse_gate:
            # The gate descriptor selects the epilogue_gate specialization
            # (template_params().epilogue_gate); its stride is compiled in like
            # Q/K/V's, so a slab column slice binds with no copy.  Declared at the
            # SAME envelope as O -- (B_env, H, S_max, D) under THD, where the adapter
            # maps G through the packed [1, T, H, D] form Q / O take.
            kw["sample_gate"] = _bhsd_desc(b_env, g.h_q, s_env, d, self.gate_dtype or self.dtype, self.device, "gate", token_stride=self.gate_token_stride)
        return SdpaFwdDslSm100(
            _bhsd_desc(b_env, g.h_q, s_env, d, self.dtype, self.device, "q", token_stride=ts),
            k_desc,
            v_desc,
            _bhsd_desc(b_env, g.h_q, s_env, d, self.o_dtype, self.device, "o"),
            lse,
            is_causal=g.is_causal,
            causal_bottom_right=g.causal_bottom_right,
            window_size_left=None if g.window_left < 0 else g.window_left,
            window_size_right=None if g.window_right < 0 else g.window_right,
            scale_softmax=g.scale,
            seq_kv_lens_present=self.seq_lens_present,
            sched_policy=sched,
            **kw,
        )

    @property
    def pertensor(self) -> bool:
        """The per-tensor FP8 kernel family (scalar descales); False for bf16 and for MXFP8."""
        return self.fp8 and not self.mxfp8

    @property
    def _family(self) -> str:
        return "MXFP8" if self.mxfp8 else "FP8" if self.fp8 else "f16/bf16"

    @classmethod
    def _row_capabilities(cls, arch: str, fp8: bool, mxfp8: bool = False):
        """The ``Capabilities`` of the SDPA-forward engine row for ``(arch, dtype family)``.

        ``fp8`` names the per-tensor FP8 row, ``mxfp8`` the block-scale one
        (``engine_name(arch, fp8=..., mxfp8=...)``); a caller passing the fp8-class
        flag with ``mxfp8=True`` gets the MXFP8 row -- the two rows make DIFFERENT
        claims (LPT, gate dtypes), so the family must never be conflated.

        A missing / renamed row is a typed decline, not a bare ``StopIteration``
        escaping ``check_support`` (engine-contract § 2): this sits on the decline
        path of every ``fuse_gate=True`` block, on every device.
        """
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import engines

        row = engines.engine_name(arch=arch, fp8=bool(fp8) and not mxfp8, mxfp8=bool(mxfp8))
        caps = next((spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == row), None)
        if caps is None:
            raise NotImplementedError(f"{cls.name}: no SDPA forward engine row {row!r} in cudnn.sdpa.fwd.engines.ENGINE_SPECS")
        return caps

    def _sched_policy(self) -> int:
        """LPT where the engine row says it is validated, NATURAL everywhere else.

        The causal load is triangular, so natural row-major order leaves the last
        CTAs of a wave with almost nothing to do. LPT rebalances it: measured
        standalone on the bf16 d256 kernel at +18.2 / +11.6 / +1.1 % of causal
        SOL at S = 4096 / 8192 / 32768, recovering 40 / 51 / 29 % of the
        causal-vs-dense gap, and dense-neutral; on the per-tensor FP8 d256 kernel
        +5.2 / +5.9 / +5.6 / +2.0 / +2.3 % at S = 2K..32K (perf node,
        launch-interleaved, O bit-identical to NATURAL).

        READ OFF THE ROW, never transcribed: the SM107 f16 and per-tensor FP8
        rows claim LPT per flavor through `sched_policies_by_d_shape` ((256, 256)
        on both, (192, 128) on FP8), and this picks LPT exactly when the row the
        live device would use has it in the effective domain for (d, d). Asking
        for LPT on another flavor would be asking for something no row claims
        (contract § 4: honoured or ineligible, never substituted). Two copies of
        one fact drift (engine-contract § 8b'), so there is no shape literal
        here. The policy is compile-time; the arm not chosen is never traced.
        """
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import ambient_device, compute_capability
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import SCHED_LPT, SCHED_NATURAL

        d = self.geom.d_head
        arch = "sm107" if compute_capability(ambient_device()) == (10, 7) else "sm100"
        # MXFP8 reads the MXFP8 row (D5): at (256, 256) it claims NATURAL only --
        # the per-tensor row's LPT does not transfer, and `sched_policy=None`
        # would let the adapter's auto knobs pick LPT for a masked fp8-class d256.
        caps = self._row_capabilities(arch, self.fp8, self.mxfp8)
        domain = dict(caps.sched_policies_by_d_shape).get((d, d), caps.sched_policies)
        return SCHED_LPT if SCHED_LPT in domain else SCHED_NATURAL

    def _check_gate_geometry(self) -> None:
        """Decline ``fuse_gate`` on a head dim -- or a gate dtype -- the Rubin row does not claim.

        The adapter picks its kernel flavor from d_head (d128 / d192x128 / d256
        / d512) and the gate epilogue is wired on ONE of them.  READ OFF THE ROW
        (``Capabilities.epilogue_gate`` / ``epilogue_gate_d_shapes`` of the SM107
        engine the block runs on), the same way ``_sched_policy`` reads LPT: two
        copies of one fact drift (engine-contract § 8b'), so there is no shape
        literal here.  The adapter's standalone twin re-checks the same claim
        after the cc gate; THIS pin runs first so a d64 block declines with the
        head dim in the message on EVERY device (a d128 block would otherwise
        pass eligibility here and, on Rubin, be declined only by the adapter --
        or, before the twin existed, run a gate nobody wired at 2-4x the cost).
        Mirrors ``engines.mismatch()``: a row without the claim declines every
        head dim; a row claiming it with no shape set claims every flavor; a row
        with no ``epilogue_gate_dtypes`` reads GATE in Q's dtype.
        """
        g = self.geom
        caps = self._row_capabilities("sm107", self.fp8, self.mxfp8)
        if not caps.epilogue_gate:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True is not served by the {self._family} SM107 SDPA engine row. "
                "Use fuse_gate=False (stage (5) runs as its own launch)"
            )
        shapes = caps.epilogue_gate_d_shapes
        if shapes is not None and (g.d_head, g.d_head) not in shapes:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True is served at (d_head, d_head) in {sorted(shapes)}; got {(g.d_head, g.d_head)}. "
                "Use fuse_gate=False for other head dims"
            )
        # The gate's dtype is a row claim too (``epilogue_gate_dtypes``; None =
        # "G in Q's dtype", exactly ``engines.mismatch()``'s reading -- the f16
        # row; the FP8 row claims bf16).  Pinned HERE, before the cc gate, so a
        # standalone fp8 ``_Sdpa(fuse_gate=True)`` that forgot ``gate_dtype=``
        # (the block always passes its activation dtype) names the knob instead
        # of reading as the cc decline off Rubin or as the adapter's dtype
        # ValueError on it.  No dtype literal: the domain is the row's.
        gate_dtype = self.gate_dtype or self.dtype
        if caps.epilogue_gate_dtypes is None:
            served = (self.dtype,)
        else:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import to_torch_dtype

            served = tuple(to_torch_dtype(dt) for dt in sorted(caps.epilogue_gate_dtypes, key=str))
        if gate_dtype not in served:
            raise NotImplementedError(
                f"{self.name}: fuse_gate=True reads GATE in {' / '.join(str(t) for t in served)} on the "
                f"{self._family} SM107 SDPA engine row; got gate_dtype={gate_dtype}. "
                "Pass gate_dtype= (the block's activation dtype) or use fuse_gate=False"
            )

    def check_support(self) -> None:
        # Geometry first, so the decline reads the same on every device (the
        # block's own stages ahead of this one are cc-independent too).
        if self.fuse_gate:
            self._check_gate_geometry()
        if self.mxfp8 and self.fuse_gate:
            # The block-scale kernel is reached ONLY through the production
            # adapter's `sample_gate` (PR-A / S7).  An adapter without it has no
            # gated MXFP8 path at all -- decline rather than reach for any
            # per-tensor fork, which cannot take block scales.
            import inspect

            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.api_dsl import SdpaFwdDslSm100

            if "sample_gate" not in inspect.signature(SdpaFwdDslSm100.__init__).parameters:
                raise NotImplementedError(
                    f"{self.name}: MXFP8 + fuse_gate needs the production adapter's `sample_gate` (the shared epilogue_gate hook); "
                    "this checkout's SdpaFwdDslSm100 has none. Use fuse_gate=False."
                )
        cc = torch.cuda.get_device_capability(self.device)
        if tuple(cc) != _SM107_CC:
            raise NotImplementedError(f"gated_attention_block targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found SM{cc[0]}{cc[1]}")
        self._impl = self._build_impl()
        # The adapter's own contract check: the dense S % 128 split-KV decline and the
        # FP8 envelope, the gate descriptor's shape / dtype / TMA-expressible
        # stride, and the standalone twins of the rows' gate claims (arch,
        # head dims, MXFP8, THD, paged, split, PackGQA -- engine-contract § 8b).
        self._impl.check_support()

    def compile(self) -> None:
        if self._impl is None:
            raise RuntimeError("call check_support() before compile()")
        self._impl.compile()
        if self.decode_route is not None:
            # The route this stage DECLARED (decode_route: the block's rule on its shapes -- the decode tile, or the prefill tile through
            # the THD paged binding) against the module the adapter COMPILED -- two readings of one rule; a drift is the "config and body
            # must agree" class, so it raises rather than letting an envelope name a tile the plan does not run.
            got = getattr(self._impl, "kernel_template", None)
            if got != self.decode_route.tile:
                raise RuntimeError(
                    f"{self.name}: the DECODE route named {self.decode_route.tile} for S_q={self.seq_len} at group {self.decode_route.group} "
                    f"(policy={self.decode_route.policy}, split_kv={self.decode_split_kv}, pack_gqa={self.decode_route.pack_gqa}, thd={self.thd}) but the "
                    f"adapter compiled {got!r}: decode_route and SdpaFwdDslSm100._decode_q_tile disagree -- fix the drift, neither reading is the truth alone"
                )
            if self.decode_route.pack_gqa != bool(getattr(self._impl, "pack_gqa", False)):
                raise RuntimeError(
                    f"{self.name}: the DECODE route of S_q={self.seq_len} at group {self.decode_route.group} packs the group ({self.decode_route.pack_gqa}) but the "
                    f"adapter compiled pack_gqa={getattr(self._impl, 'pack_gqa', None)}"
                )
            if self.decode_route.decode_tile:
                # The WIDTH too: every width of the swap-AB tile is the same kernel_template, so the template pin cannot see the route
                # naming 64 columns while the adapter compiled 32 (the adapter re-reads the rule at compile with its own facts).
                got_n_q = getattr(getattr(getattr(self._impl, "_k_mod", None), "CFG", None), "N_Q", None)
                if got_n_q is not None and int(got_n_q) != int(self.decode_route.n_q):
                    raise RuntimeError(
                        f"{self.name}: the DECODE route of S_q={self.seq_len} at group {self.decode_route.group} names the {self.decode_route.n_q}-column decode "
                        f"tile (policy={self.decode_route.policy}, q_tile_max={self.decode_route.q_tile_max}) but the adapter compiled N_Q={got_n_q}: decode_route "
                        "and SdpaFwdDslSm100._decode_q_tile disagree on the width -- fix the drift, neither reading is the truth alone"
                    )

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch the SDPA carves from the block's workspace.

        0 on every DENSE block configuration (unsplit: ``api_dsl.py``
        ``SdpaFwdDslSm100.scratch_workspace_bytes``).  NON-ZERO under ``thd=True``:
        the adapter's packed metadata -- the per-sequence lengths / prefix sums the
        setup launch normalizes on device, the per-sequence runtime descriptors and
        (f16 / bf16) the sinks dummy, or (per-tensor FP8) the same three terms plus
        its identity-scale words -- about 1 KiB at a handful of sequences (1024 B
        bf16 / 1152 B FP8 at ``num_sequences=3``), growing 128 B per sequence.
        The block folds it into ``get_workspace_size()`` through the same
        ``max(proj, out_proj, sdpa, 1)`` as the GEMMs' scratch, so the size stays
        honest either way.
        """
        if self._impl is None:
            raise RuntimeError("call check_support() before scratch_workspace_bytes()")
        return int(self._impl.scratch_workspace_bytes())

    def execute(
        self,
        q: torch.Tensor,  # [B, S, H_q,  D]  compact or a slab column slice (token_stride)
        k: torch.Tensor,  # [B, S, H_kv, D]
        v: torch.Tensor,  # [B, S, H_kv, D]
        o: torch.Tensor,  # [B, S, H_q,  D]  compact, written
        lse: Optional[torch.Tensor] = None,  # [B, H_q, S] fp32; THD: [1, H_q, T] (head-major packed, head stride T)
        seq_lens: Optional[torch.Tensor] = None,  # [B] int32, per-batch valid KV length; THD (REQUIRED): [B] lengths or [B+1] prefix sums
        workspace: Optional[torch.Tensor] = None,
        current_stream=None,
        gate: Optional[torch.Tensor] = None,  # [B, S, H_q, D] slab slice or compact gate16; fuse_gate only
        descale_q: Optional[torch.Tensor] = None,  # FP8 only: 1-element fp32 device tensors
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,  # FP8 with e4m3 O only: 1-element fp32 device tensor
        sf_q: Optional[torch.Tensor] = None,  # MXFP8 only (all three REQUIRED): F8_128x4 E8M0 blobs, uint8, the adapter's _reshape_sf byte count
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        # APPENDED (kv_source="paged", all four REQUIRED there, REFUSED otherwise): the pools stage (4) reads K / V from (the
        # declared descriptors' shape and strides exactly), the [num_sequences, >= max_pages] int32 block table at the declared
        # strides, the [num_sequences] int32 LOGICAL KV lengths (prefix + the step's tokens).  `k` / `v` above are then unread.
        block_table: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> None:
        """Hand the BSHD buffers over as BHSD views. No copy — see :func:`_bhsd_desc`.

        ONE call into the adapter for every configuration; the FP8 scales and
        the gate ride as keyword arguments the adapter validates against the
        specialization it compiled (``gate`` <-> ``sample_gate``, ``amax_o``
        refused under ``has_amax_o=False``).

        MXFP8 (``mxfp8=True``): ``sf_q`` / ``sf_k`` / ``sf_v`` are REQUIRED and
        ``descale_q/k/v`` / ``scale_o`` are REFUSED -- the adapter's MXFP8 path
        accepts and silently ignores the scalars (``_execute_mxfp8`` never reads
        them), and a silently-ignored scale is a wrong answer nobody reports.
        A non-MXFP8 stage refuses the SF blobs for the mirror-image reason.
        """
        if self._impl is None:
            raise RuntimeError("call compile() before execute()")
        if self.mxfp8:
            if sf_q is None or sf_k is None or sf_v is None:
                raise ValueError(f"{self.name}: the MXFP8 SDPA needs sf_q/sf_k/sf_v (F8_128x4 E8M0 scale-factor blobs, uint8)")
            if descale_q is not None or descale_k is not None or descale_v is not None or scale_o is not None:
                raise ValueError(
                    f"{self.name}: descale_q/k/v and scale_o are per-tensor FP8 scalars; the MXFP8 kernel dequantizes with its block "
                    "scale factors in the MMA and writes an e4m3 O UNSCALED (the adapter would silently ignore them -- refused instead)"
                )
        elif sf_q is not None or sf_k is not None or sf_v is not None:
            raise ValueError(f"{self.name}: sf_q/sf_k/sf_v are the MXFP8 scale-factor blobs; this stage was declared {self._family}")
        if self.pertensor:
            if descale_q is None or descale_k is None or descale_v is None:
                raise ValueError(f"{self.name}: the FP8 SDPA needs descale_q/k/v (1-element fp32 device tensors)")
            # The kernel folds scale_o into inv_sum REGARDLESS of O's dtype, so
            # on a bf16 O it would scale the output the block's own quantize
            # pass then scales again.  None binds the adapter's cached 1.0.
            if scale_o is not None and self.o_dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"{self.name}: scale_o is consumed only when O is e4m3 (this stage writes {self.o_dtype}); "
                    "the unfused FP8 pipeline applies scale_o in its quantize pass"
                )
            if scale_o is None and self.o_dtype == torch.float8_e4m3fn:
                # A caller bug, not a case to paper over with a per-execute fill (Rule 1).
                raise ValueError(f"{self.name}: an e4m3 O needs scale_o (1-element fp32 device tensor)")
        elif not self.mxfp8 and (descale_q is not None or descale_k is not None or descale_v is not None or scale_o is not None):
            raise ValueError(f"{self.name}: descales / scale_o are only consumed by the FP8 pipeline")
        if self.fuse_gate and gate is None:
            raise ValueError(f"{self.name}: fuse_gate=True requires the GATE tensor at execute")
        if not self.fuse_gate and gate is not None:
            raise ValueError(f"{self.name}: gate is only consumed under fuse_gate=True (the SDPA's epilogue gate); this stage was declared without it")
        paged_args = (block_table, kv_lens, k_cache, v_cache)
        if self.kv_source == KV_SOURCE_PAGED:
            if any(x is None for x in paged_args):
                raise ValueError(f"{self.name}: kv_source='paged' requires block_table, kv_lens, k_cache and v_cache at execute")
            if self.paged_dense:
                # The DECODE form: the pools as K / V, kv_lens as seq_kv_lens and NOTHING as seq_q_lens (one fact, the KV side only:
                # uniform S_q, every Q row live); seq_lens (the dense block's self-attention length, a Q-side length) is REFUSED.
                if seq_lens is not None:
                    raise ValueError(
                        f"{self.name}: seq_lens is the dense block's self-attention length (a Q-side length); over the pools the per-sequence KV "
                        "length is kv_lens, bound to the KV side only (every Q row is live) -- refused rather than read as a Q length"
                    )
                self._impl.execute(
                    q.transpose(1, 2),
                    k_cache,
                    v_cache,
                    o.transpose(1, 2),
                    lse_tensor=lse,
                    seq_q_lens=None,
                    seq_kv_lens=kv_lens,
                    block_table=block_table,
                    workspace=workspace,
                    current_stream=current_stream,
                    **({"gate": gate.transpose(1, 2)} if self.fuse_gate else {}),
                )
                return
            if seq_lens is None:
                raise ValueError(f"{self.name}: kv_source='paged' requires seq_lens at execute (the packed per-sequence Q lengths)")
            # The adapter's paged THD leg: the packed (T, H, D) Q / O views as on the THD arm, the POOLS as K / V (the adapter
            # matches their shape and strides against the declared descriptors), the Q lengths in the block's form, the
            # LOGICAL KV lengths as [B] lengths, the block table (one table for both pools).  `k` / `v` are not read.
            g, t = self.geom, self.batch * self.seq_len
            self._impl.execute(
                q.view(t, g.h_q, g.d_head),
                k_cache,
                v_cache,
                o.view(t, g.h_q, g.d_head),
                lse_tensor=lse,
                seq_q_lens=seq_lens,
                seq_kv_lens=kv_lens,
                workspace=workspace,
                current_stream=current_stream,
                block_table=block_table,
                # The fused gate: the rank-3 packed (T, H, D) view of the slab's GATE columns, as on the dense THD arm (an exact
                # `.view` at the slab's token stride, no copy).
                **({"gate": gate.view(t, g.h_q, g.d_head)} if self.fuse_gate else {}),
            )
            return
        if any(x is not None for x in paged_args):
            raise ValueError(
                f"{self.name}: block_table / kv_lens / k_cache / v_cache are consumed under kv_source='paged' only; this stage was declared kv_source='self'"
            )
        kw = {}
        if self.pertensor:
            kw.update(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o)
        if self.mxfp8:
            kw.update(sf_q=sf_q, sf_k=sf_k, sf_v=sf_v)
        if self.fuse_gate:
            # Dense: the BHSD transpose like Q / K / V / O; THD: the rank-3 packed (T, H, D) view like them (below) -- an
            # exact `.view` at the slab's token stride, no copy.
            kw["gate"] = gate.view(self.batch * self.seq_len, self.geom.h_q, self.geom.d_head) if self.thd else gate.transpose(1, 2)
        if self.thd:
            # RANK-3 PACKED (T, H, D) views, never the rank-4 transpose: the adapter's THD binder admits a rank-4 operand
            # only with the RUNTIME sequence count as its leading extent (so (1, H, T, D) is refused once B > 1) and a
            # rank-3 (T, H, D) buffer at any TMA-expressible token / head stride.  Dropping the extent-1 leading axis is an
            # exact `.view` at any stride, so the slab column slice keeps its token stride n_qkvg and a compact buffer its
            # H*D -- no copy, as on the dense arm.  ONE lengths tensor serves both sides (self-attention).
            if seq_lens is None:
                raise ValueError(f"{self.name}: thd=True requires seq_lens at execute (the packed per-sequence lengths, for the Q and the KV side alike)")
            g, t = self.geom, self.batch * self.seq_len
            self._impl.execute(
                q.view(t, g.h_q, g.d_head),
                k.view(t, g.h_kv, g.d_head),
                v.view(t, g.h_kv, g.d_head),
                o.view(t, g.h_q, g.d_head),
                lse_tensor=lse,
                seq_q_lens=seq_lens,
                seq_kv_lens=seq_lens,
                workspace=workspace,
                current_stream=current_stream,
                **kw,
            )
            return
        self._impl.execute(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            o.transpose(1, 2),
            lse_tensor=lse,
            seq_kv_lens=seq_lens,
            workspace=workspace,
            current_stream=current_stream,
            **kw,
        )


def _sparse_record():
    """The sparse core's claims: the adapter-owned ``SparseCapabilities`` record of ``cudnn.sdpa.fwd.sparse_gqa_sm107``
    (Form A -- no engine row, no manifest slot: no graph form carries a block-index list).  Read lazily (the adapter
    module imports no framework at import time) and never transcribed: the adapter's ``check_support`` enforces this same
    record, and two copies of one fact drift (engine-contract s 8b')."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_gqa_sm107 import SPARSE_CAPABILITIES

    return SPARSE_CAPABILITIES


_SHORT_DTYPE = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "fp32"}


def _sparse_record_torch_dtypes(rec) -> Tuple[torch.dtype, ...]:
    """The record's ``dtypes`` (public ``cudnn.data_type`` members) as torch dtypes, in a stable order."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import to_torch_dtype

    return tuple(to_torch_dtype(dt) for dt in sorted(rec.dtypes, key=str))


def _check_qsa_geometry_against_record(geom: GatedAttentionBlockGeometry, dtype: torch.dtype) -> None:
    """A ``QsaSpec`` geometry's claims on the sparse core against the adapter's record -- the head dim (``d_shapes``), the
    block size, the ``top_k`` range, the GQA group the core packs on its N tile, the activation dtype -- as typed
    ``NotImplementedError`` naming the feature and the record's bound.  Device-free; called at declaration
    (``GatedAttentionBlockFwd._check_qsa_declaration``) and again by the sparse stage's ``check_support`` so the stage stays
    honest on its own; the adapter re-checks every one of them (it is the enforcement point)."""
    rec = _sparse_record()
    g, q = geom, geom.qsa
    if (g.d_head, g.d_head) not in rec.d_shapes:
        raise NotImplementedError(
            f"QsaSpec needs d_head == {' / '.join(str(d) for d, _ in sorted(rec.d_shapes))} (the sparse core's head dim: the d256 swap-AB body), got d_head={g.d_head}"
        )
    if q.block_size not in rec.index_block_sizes:
        raise NotImplementedError(
            f"QsaSpec.block_size={q.block_size}: the sparse core serves block sizes {sorted(rec.index_block_sizes)} (one gather transaction = one block's rows)"
        )
    if not (rec.index_top_k_min <= q.top_k <= rec.index_top_k_max) or q.top_k % QSA_TOP_K_ALIGN:
        raise NotImplementedError(
            f"QsaSpec.top_k={q.top_k}: the sparse core serves top_k as a multiple of {QSA_TOP_K_ALIGN} in [{rec.index_top_k_min}, {rec.index_top_k_max}] "
            "(its index staging is sized by it)"
        )
    if not (rec.index_gqa_group_min <= g.gqa_ratio <= rec.index_gqa_group_max):
        raise NotImplementedError(
            f"QsaSpec needs h_q // h_kv <= {rec.index_gqa_group_max} (the sparse core packs the GQA group on an N tile of at most "
            f"{rec.index_gqa_group_max} rows), got {g.h_q} // {g.h_kv} = {g.gqa_ratio}"
        )
    served = _sparse_record_torch_dtypes(rec)
    if dtype not in served:
        raise NotImplementedError(
            f"QsaSpec needs {' / '.join(_SHORT_DTYPE.get(t, str(t)) for t in served)} activations (the sparse core's dtypes), got {dtype}"
        )


class _SparseSdpa(_Stage):
    """(4) under a :class:`QsaSpec`: ``O = softmax(Q K^T * scale + mask) V`` over each query's SELECTED 4-token blocks
    and its open tail block, causal, GQA-broadcast -- the index-list sparse attention core
    (``sdpa/fwd/kernels/sm107/sparse_d256_f16.py`` through its standalone adapter
    ``sdpa/fwd/sparse_gqa_sm107.SparseGqaFwdDslSm107``).

    **This stage writes no kernel.**  Like :class:`_Sdpa` it drives an adapter and reads every claim off ONE record --
    the adapter-owned ``SparseCapabilities`` (Form A: no engine row, no manifest slot; no graph form carries a
    block-index list) through :func:`_sparse_record`, never a literal: the served dtypes, head dim, block size, ``top_k``
    range and GQA group, the arms the body does not carry.  The adapter's ``check_support`` is the ENFORCEMENT point and
    runs inside this stage's: the CuTe DSL ``sm_107a`` gate FIRST (Rule 7, before any kernel import), then the arch range
    off the record (no ``_SM107_CC`` literal here -- that would be a second copy of the record's ``sm_lo`` / ``sm_hi``),
    then every unserved arm by name, the dtypes, the shapes, the stride contract and the index tensors' form -- so a
    sparse decline reads the same through the block and through the adapter.

    Declared at the block's strides with storage-free ``SparseOperandDesc`` stand-ins (the buffers exist only at execute):
    Q / K / V as BSHD column slices of the slab (``token_stride = n_qkvg``, the in-place pipeline) or the compact buffers
    (``inplace_qkv=False``), O compact, LSE ``[B, H_q, S]`` fp32, the index tensors FLAT (``[T, top_k]`` / ``[T]`` int32),
    ``seq_kv_lens`` ``[B]`` when declared.  ``execute`` binds the real views AS THEY ARE -- the adapter's contract is BSHD,
    so unlike :class:`_Sdpa` there is no transpose -- plus the index tensors (a ``[B, S, top_k]`` list is viewed flat: an
    exact view of a contiguous tensor, no copy).  ``block_lens`` is optional PER CALL, so :meth:`compile` builds BOTH
    ``has_block_lens`` variants at plan time and the adapter dispatches on the tensor it is handed (plan-time keys only,
    Rule 4).  The dense kernel carries no GMEM scratch (the count, the tail block and the dead items are derived on
    device), so :meth:`scratch_workspace_bytes` folds 0 into the block's workspace -- a dense ``QsaSpec`` block's workspace
    is the dense block's.

    **THD / packed sequences (``thd=True``)**: the adapter's packed arm (the kernel's persistent claim-counter form).  The
    block is internally ``B = 1, S = T``, and the adapter's packed contract IS that shape -- Q / O ``[1, T, H, D]``, K / V
    ``[1, T, H_kv, D]`` with batch extent 1, the token totals as CAPACITIES -- so ``execute`` binds the very same views the
    dense arm binds (no transpose, no copy; the slab column slice keeps its token stride).  ``execute(seq_lens=)`` is
    REQUIRED and serves BOTH length sides (self-attention): the ``[B]`` int32 lengths or the ``[B+1]`` int32 prefix sums
    (``cu_seqlens=True``), declared as ``seq_q_lens`` / ``seq_kv_lens`` descriptors of exactly that extent and never read
    on the host (the setup launch normalizes them on device).  ``block_ids`` keeps its ``[T, top_k]`` form with every id
    RELATIVE TO ITS SEQUENCE (block ``j`` of sequence ``b`` = its tokens ``[4j, 4j + 4)``); the LSE is the head-major packed
    ``[1, H_q, T]`` the block already carries at ``B = 1``.  The ONE scratch the arm needs -- the adapter's int32 THD
    metadata (``THD_META_WORDS(B)`` = 4 B + 4 words, 16-byte rounded: the layout the sm107 sparse adapter sizes it with) -- enters the block's engine arm through
    :meth:`scratch_workspace_bytes` and is handed back at ``execute(workspace=)``.  A one-sequence packing computes the
    dense ``B = 1`` function bitwise; an empty sequence owns no rows.

    ``fuse_gate`` (stage (5) inside this stage's epilogue): the GATE is declared like the other operands -- O's shape at
    the slab's GATE columns (``gate_token_stride``) or compact -- as the adapter's ``epilogue_gate`` operand, which selects
    the kernel's gated specialization (``TemplateParams.epilogue_gate``); ``execute(gate=)`` binds the view and the kernel
    writes ``O * sigmoid(GATE)`` in place of O after its own dead-row select (the LSE is untouched).

    **The DECODE / extend-with-prefix mode over the pools (``kv_source="paged"``, appended; 2026-10-14)**: K / V are the
    serving stack's PAGED POOLS instead of the block's own operands -- the adapter's paged read (``paged_kv=True``: the pools
    AS SAMPLED, ``[num_pages, H_kv, P, D]`` HND compact / NHD by strides / a unified row's halves, their strides compiled in;
    ONE contiguous ``(B, max_pages)`` int32 block table; ``seq_kv_lens = kv_lens``, the per-sequence LOGICAL KV length, the
    pools' only extent) on the BOTTOM-RIGHT diagonal (``bottom_right=True``: row ``j`` of sequence ``b`` sits at
    ``kv_lens[b] - S_q + j``; a negative position is a dead row).  The list form is a DECLARATION fact the block derives from
    ``S_q``: ``list_per_sequence=True`` at ``S_q <= 4`` -- ONE list per sequence, ``block_ids [B, top_k]`` (+ ``block_lens
    [B]``), anchored at the step-0 position (the MTP form: the listed blocks plus every token from the step-0 tail start to
    the row's own position), ``False`` past it -- one list per row, ``[T, top_k]`` at bottom-right positions (a chunked
    prefill over a cached prefix).  ``split_kv = S`` (``1 <= S <= ceil((top_k + 1) / 32)``) cuts every item's tiles into
    ``S`` chunks with fp32 partials in the block's engine workspace arm (:meth:`scratch_workspace_bytes` grows by the adapter's
    two slabs) and the split combine as a second launch; the gate rides the core's epilogue unsplit and the combine under a
    split (the adapter's own routing), so ``fuse_gate`` is honoured either way.  ``execute(block_table=, kv_lens=, k_cache=,
    v_cache=)`` are REQUIRED on the paged declaration and REFUSED on the prefill form; ``seq_lens`` is REFUSED over the pools
    (the KV length is ``kv_lens``, bound to the KV side only: every row of the step is live unless its position is
    negative).  Never together with ``thd`` (the core serves packed sequences and paged pools one at a time -- the block's
    serving table declines the packed form by name).

    Not here (typed declines at the block's declaration, each naming its feature): ``causal_bottom_right`` on the block's
    OWN K / V (the diagonal is the paged mode's); the quantized pipelines and training; THD together with the in-block
    indexer (the scorer reads dense ``[B, S]`` prompts).  The in-block indexer over the pools is served (2026-10-21: its
    decode step, :class:`_IndexerDecode`, hands this stage one list per row and no ``block_lens``; the shared step-0 form
    under the indexer is a typed decline).  THD together with the paged write-through is served (2026-10-14: the write-through stage addresses
    tokens, so the packed sparse block builds the same ``(4w)`` stage).  Each lands by flipping the record, the adapter AND
    this stage in the same change; the declaration decline names the record's state so the flip is visible (``thd`` and
    ``fuse_gate`` flipped 2026-10-08, the paged read / decode form 2026-10-14).
    """

    name = "sdpa_sparse"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        want_lse: bool,
        seq_lens_present: bool = False,
        token_stride: int = 0,
        gate_token_stride: int = 0,
        thd: bool = False,  # packed sequences: batch=1, seq_len=T; the adapter's THD arm (see the class docstring)
        num_sequences: Optional[int] = None,  # THD: B, the length tensor's [B] (or [B+1] prefix-sum) entries
        max_seq_len: Optional[int] = None,  # THD: S_max, the longest sequence the block admits (recorded; the core derives on device)
        cu_seqlens: bool = False,  # THD: seq_lens is [B+1] int32 prefix sums (True) or [B] int32 lengths (False)
        fuse_gate: bool = False,
        # APPENDED (the DECODE / extend-with-prefix mode over the pools, class docstring): where stage (4) reads K / V from --
        # "self" = the block's own K / V (as before); "paged" = the serving stack's pools through the adapter's paged read, the
        # three descriptors (the declaration SAMPLES: layout and strides compiled in), the list form and the split.
        kv_source: str = KV_SOURCE_SELF,
        paged_page_size: int = 0,  # kv_source="paged": the pools' tokens per page (the block's paged_kv_page_size)
        k_cache_desc: Optional[TensorDesc] = None,  # kv_source="paged": the K pool's [num_pages, H_kv, P, D] descriptor (shape + strides)
        v_cache_desc: Optional[TensorDesc] = None,  # kv_source="paged": the V pool's descriptor
        block_table_desc: Optional[TensorDesc] = None,  # kv_source="paged": the (B, max_pages) int32 block table's descriptor (contiguous)
        list_per_sequence: bool = False,  # kv_source="paged": block_ids [B, top_k] shared by the sequence's S_q <= 4 rows (else [T, top_k])
        split_kv: int = 1,  # kv_source="paged": the KV split (fp32 partials in the engine workspace arm + the combine); 1 = unsplit
        # APPENDED (2026-10-21): the adapter's programmatic-dependent-launch form -- the sparse core's launch attribute + griddepcontrol sites
        # and the split combine's pair (SparseGqaFwdDslSm107(use_pdl=True)); the block's resolved switch; THD x use_pdl is the adapter's typed
        # decline (the block declines it at declaration first, carrying the adapter's one statement of the fact, `pdl_decline_reason`).
        use_pdl: bool = False,
        # APPENDED (the lean block-sparse decode tile): the adapter's BODY knob -- the block passes its RESOLVED form ("lean" / "tcgen05":
        # one decision point, the declaration; "auto" lets the adapter resolve, the standalone callers' reading) and the lean body's k pin;
        # `kernel_template` reads the compiled body back and `check_support` pins it against the form the block resolved (a route is pinned
        # by the kernel that compiled, never transcribed).
        tile_form: str = "auto",
        sub_tiles_per_cta: Optional[int] = None,
    ) -> None:
        if geometry.qsa is None:
            raise ValueError(f"{self.name}: the geometry declares no QsaSpec (geometry.qsa is None); the dense stage serves it")
        self.tile_form, self.sub_tiles_per_cta = tile_form, sub_tiles_per_cta
        self.kernel_template: Optional[str] = None  # the adapter's readback after check_support
        self.use_pdl = bool(use_pdl)  # read back by the block's pdl_stages / use_pdl_effective like every switched stage's
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = device
        self.want_lse = bool(want_lse)
        self.seq_lens_present = bool(seq_lens_present)
        # token_stride != 0 => Q/K/V are column slices of the fused projection, read in place at that stride (the
        # adapter validates the stride contract on the declaration).  0 => the compact buffers.
        self.token_stride = int(token_stride)
        # fuse_gate => the sparse core's epilogue_gate specialization reads GATE (a slab column slice at gate_token_stride; 0 =
        # compact like O) in the activation dtype and writes O gated in place of O; the gate operand is declared like Q / K / V
        # (storage-free) and bound at execute.
        self.fuse_gate = bool(fuse_gate)
        self.gate_token_stride = int(gate_token_stride)
        # THD (packed sequences).  The block validates the whole contract (typed, in order) before building this stage; the
        # checks here keep the STAGE honest on its own: the sequence count must be declared (the lengths' extent is a
        # declaration fact the adapter binds exactly), the two length contracts are exclusive, and the record's claim is
        # read, never assumed -- a record that drops the arm declines here by name (the declaration's decline reads the same).
        self.thd = bool(thd)
        self.cu_seqlens = bool(cu_seqlens)
        self.num_sequences = None if num_sequences is None else int(num_sequences)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        if self.thd:
            if self.batch != 1:
                raise ValueError(f"{self.name}: thd=True is the packed form -- batch 1, seq_len T; got batch={self.batch}")
            if self.num_sequences is None or self.num_sequences < 1:
                raise ValueError(f"{self.name}: thd=True needs num_sequences >= 1 (the length tensor's [B] / [B+1] extent the adapter binds)")
            if self.seq_lens_present:
                raise ValueError(f"{self.name}: thd=True and seq_lens_present=True are mutually exclusive (the packed lengths ARE the per-sequence lengths)")
            if not self._record().thd:
                raise NotImplementedError(f"{self.name}: thd=True -- the sparse adapter's record declines THD (the kernel body does not carry the packed arm)")
        elif self.num_sequences is not None or self.max_seq_len is not None or self.cu_seqlens:
            raise ValueError(f"{self.name}: num_sequences / max_seq_len / cu_seqlens are THD-only (thd=True)")
        # The paged read (the DECODE / extend-with-prefix mode).  The block validates the whole contract first (the serving
        # table, the samples); these checks keep the STAGE honest on its own: the three descriptors and the page size exist
        # exactly on the paged declaration, never with THD (the core serves the two arms one at a time), on the bottom-right
        # diagonal only (the step's rows sit at the END of the sequence), with the per-sequence KV length declared (the pools'
        # only extent), and every arm read off the RECORD, never assumed -- a record that drops one declines here by name.
        if kv_source not in _KV_SOURCES:
            raise ValueError(f"{self.name}: kv_source must be one of {_KV_SOURCES}, got {kv_source!r}")
        self.kv_source = kv_source
        self.paged = kv_source == KV_SOURCE_PAGED
        self.paged_page_size = int(paged_page_size)
        self.k_cache_desc, self.v_cache_desc, self.block_table_desc = k_cache_desc, v_cache_desc, block_table_desc
        self.list_per_sequence = bool(list_per_sequence)
        self.split_kv = int(split_kv)
        if self.paged:
            rec = self._record()
            if self.paged_page_size <= 0 or k_cache_desc is None or v_cache_desc is None or block_table_desc is None:
                raise ValueError(f"{self.name}: kv_source='paged' needs paged_page_size > 0 and the k_cache / v_cache / block_table descriptors")
            if self.thd:
                raise NotImplementedError(
                    f"{self.name}: kv_source='paged' with thd=True -- the sparse core serves packed sequences and paged pools one at a time (the packed "
                    "sequence's K / V row offset composes with a dense tensor only, never with a page pool); declare the dense [B, S_q] form"
                )
            if not geometry.causal_bottom_right:
                raise ValueError(
                    f"{self.name}: kv_source='paged' is the bottom-right form (the step's rows sit at the end of the sequence); got causal_bottom_right=False"
                )
            if not self.seq_lens_present:
                raise ValueError(f"{self.name}: kv_source='paged' needs seq_lens_present=True (the per-sequence KV length is the pools' only extent)")
            if self.split_kv < 1:
                raise ValueError(f"{self.name}: split_kv must be >= 1, got {split_kv}")
            for arm, claimed in (("paged_kv", rec.paged_kv), ("bottom_right", rec.bottom_right), ("decode", rec.decode)):
                if not claimed:
                    raise NotImplementedError(
                        f"{self.name}: kv_source='paged' -- the sparse adapter's record declines {arm} (the kernel body does not carry the arm)"
                    )
            if self.list_per_sequence and not rec.index_list_per_sequence:
                raise NotImplementedError(f"{self.name}: list_per_sequence=True -- the sparse adapter's record declines the shared per-sequence list")
            if self.list_per_sequence and self.seq_len > rec.index_shared_list_max_tokens:
                raise NotImplementedError(
                    f"{self.name}: list_per_sequence=True serves S_q <= {rec.index_shared_list_max_tokens} rows per sequence (the record's shared-list cap); got S_q={self.seq_len}"
                )
            if self.split_kv > 1 and not rec.split_kv:
                raise NotImplementedError(f"{self.name}: split_kv={self.split_kv} -- the sparse adapter's record declines the KV split")
        elif (
            self.paged_page_size
            or k_cache_desc is not None
            or v_cache_desc is not None
            or block_table_desc is not None
            or self.list_per_sequence
            or self.split_kv != 1
        ):
            raise ValueError(
                f"{self.name}: paged_page_size / k_cache_desc / v_cache_desc / block_table_desc / list_per_sequence / split_kv belong to kv_source='paged'"
            )
        self._impl = None
        self._ops = None
        self._compiled = False

    @staticmethod
    def _record():
        return _sparse_record()

    def _operands(self) -> dict:
        """The declared operands as storage-free descriptors at the block's strides -- what ``check_support`` validates
        and what every ``execute`` binding has to match exactly."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_gqa_sm107 import SparseOperandDesc

        g, b, s, d = self.geom, self.batch, self.seq_len, self.geom.d_head
        t = b * s
        ts_q = self.token_stride or g.h_q * d
        ts_kv = self.token_stride or g.h_kv * d

        def bshd(h: int, ts: int) -> SparseOperandDesc:
            return SparseOperandDesc((b, s, h, d), (s * ts, ts, d, 1), self.dtype, self.device)

        # THD: ONE lengths tensor of exactly num_sequences (+1 under cu_seqlens) int32 entries serves both sides; the block's
        # execute-time form check (`_check_thd_seq_lens`) and the adapter's `_bind` both hold the caller to this extent.
        lens = SparseOperandDesc((self.num_sequences + (1 if self.cu_seqlens else 0),), (1,), torch.int32, self.device) if self.thd else None
        if self.paged:
            # The pools AS SAMPLED (the serving stack's layout and strides, compiled into the gather maps; the bound tensors must match
            # them exactly) and the WHOLE block table at the sample's row stride (contiguous: the core bulk-reads it).
            def pool(desc: TensorDesc) -> SparseOperandDesc:
                return SparseOperandDesc(tuple(int(x) for x in desc.shape), tuple(int(x) for x in desc.stride), self.dtype, self.device)

            k_desc, v_desc = pool(self.k_cache_desc), pool(self.v_cache_desc)
            bt = self.block_table_desc
            table = SparseOperandDesc((b, int(bt.shape[1])), (int(bt.stride[0]), 1), torch.int32, self.device)
        else:
            k_desc, v_desc, table = bshd(g.h_kv, ts_kv), bshd(g.h_kv, ts_kv), None
        # the index tensors: one row per query token, or ONE row per sequence under the shared list (the decode mode at S_q <= 4)
        rows = b if self.list_per_sequence else t
        return dict(
            q=bshd(g.h_q, ts_q),
            k=k_desc,
            v=v_desc,
            o=bshd(g.h_q, g.h_q * d),
            # fuse_gate: GATE in O's shape at the slab's GATE columns (gate_token_stride) or compact -- the adapter's gate operand
            gate=bshd(g.h_q, self.gate_token_stride or g.h_q * d) if self.fuse_gate else None,
            block_ids=SparseOperandDesc((rows, g.qsa.top_k), (g.qsa.top_k, 1), torch.int32, self.device),
            block_lens=SparseOperandDesc((rows,), (1,), torch.int32, self.device),
            lse=SparseOperandDesc((b, g.h_q, s), (g.h_q * s, s, 1), torch.float32, self.device) if self.want_lse else None,
            seq_kv_lens=lens if self.thd else (SparseOperandDesc((b,), (1,), torch.int32, self.device) if self.seq_lens_present else None),
            seq_q_lens=lens,
            block_table=table,
        )

    def check_support(self) -> None:
        """The geometry against the record (device-free: the rows the declaration checked, re-run so the stage is honest on
        its own), then the adapter -- the enforcement point: the DSL ``sm_107a`` gate before any kernel import (Rule 7),
        the arch range off the record, the unserved arms by name, dtypes, shapes, the stride contract, the index tensors."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_gqa_sm107 import SparseGqaFwdDslSm107

        g = self.geom
        _check_qsa_geometry_against_record(g, self.dtype)
        cc = tuple(torch.cuda.get_device_capability(self.device))
        ops = self._operands()
        self._ops = ops  # the declared forms: execute re-strides the unit dims of the bound views to them (below)
        self._impl = SparseGqaFwdDslSm107(
            q=ops["q"],
            k=ops["k"],
            v=ops["v"],
            o=ops["o"],
            block_ids=ops["block_ids"],
            lse=ops["lse"],
            block_lens=ops["block_lens"],
            seq_kv_lens=ops["seq_kv_lens"],
            top_k=g.qsa.top_k,
            block_size=g.qsa.block_size,
            scale=g.scale,
            bottom_right=g.causal_bottom_right,
            epilogue_gate=ops["gate"],
            device_cc=cc,
            # THD: the packed arm -- both length descriptors, the lengths' FORM (a declaration fact: [B] or [B+1]); the
            # metadata workspace is bound per call (the block's engine arm, `execute(workspace=)`).
            thd=self.thd,
            seq_q_lens=ops["seq_q_lens"],
            cu_seq_q_lens=self.cu_seqlens,
            cu_seq_kv_lens=self.cu_seqlens,
            # The paged read (the DECODE / extend-with-prefix mode): the pools as sampled, the whole table, the list form, the
            # split -- the adapter validates the arms against each other and the pools' strides (its own paged contract); the
            # partials workspace is bound per call (the block's engine arm).
            paged_kv=self.paged,
            page_size=self.paged_page_size if self.paged else 0,
            block_table=ops["block_table"],
            list_per_sequence=self.list_per_sequence,
            split_kv=self.split_kv,
            # the programmatic-dependent-launch form of the core and its combine (the block's resolved switch; the adapter declines THD x PDL)
            use_pdl=self.use_pdl,
            # the BODY: the block's resolved form (or "auto" for a standalone reading) and the lean tile's k pin
            tile_form=self.tile_form,
            sub_tiles_per_cta=self.sub_tiles_per_cta,
        )
        self._impl.check_support()
        # the route PINNED by the kernel that will compile (a route is pinned, never transcribed): the adapter's resolved body must be the one the block declared
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import SPARSE_TILE_FORM_AUTO

        if self.tile_form != SPARSE_TILE_FORM_AUTO and self._impl.tile_form != self.tile_form:
            raise RuntimeError(
                f"{self.name}: the block resolved sparse_tile_form={self.tile_form!r} but the adapter compiled {self._impl.tile_form!r} (kernel_template "
                f"{self._impl.kernel_template}); the two readings of the route disagree -- a bug, not a decline"
            )
        self.kernel_template = self._impl.kernel_template

    def compile(self) -> None:
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call check_support() before compile()")
        # BOTH block_lens variants at plan time: the caller passes block_lens per call or not, and nothing may compile on
        # the execute path (Rule 4).  One template load, two pointer-ABI artifacts.
        self._impl.compile(has_block_lens=True)
        self._impl.compile(has_block_lens=False)
        self._compiled = True

    def scratch_workspace_bytes(self) -> int:
        """The adapter's ``scratch_workspace_bytes``: 0 on a dense declaration (the kernel has no GMEM scratch), the int32
        THD metadata (``THD_META_WORDS(B)`` = 4 B + 4 words, 16-byte rounded: the sm107 sparse adapter's own sizing) under ``thd=True``; folded into the block's workspace
        like the dense stage's so the size stays honest either way."""
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call check_support() before scratch_workspace_bytes()")
        return int(self._impl.scratch_workspace_bytes())

    def _declared_strides(self, t: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        """The bound view re-strided on its UNIT dims to the declared strides -- an exact view of the same storage (a unit dim's
        stride is unobservable by the kernel, but ``torch.view`` picks its own value for it: ``[T, H, D].view(B, 1, H, D)`` at
        ``S_q = 1`` gives the seq dim ``H x D`` where the declaration says the token stride, and the adapter's ``_bind`` holds
        every stride to the declaration).  A non-unit dim is never touched: a real mismatch stays the adapter's typed refusal."""
        if t is None or self._ops is None or self._ops.get(name) is None:
            return t
        declared = tuple(self._ops[name].strides)
        if len(declared) != t.dim():
            return t
        want = tuple(int(d) if int(n) == 1 else int(s) for n, s, d in zip(t.shape, t.stride(), declared))
        if want == tuple(int(s) for s in t.stride()):
            return t
        return t.as_strided(tuple(int(n) for n in t.shape), want, t.storage_offset())

    def execute(
        self,
        q: torch.Tensor,  # [B, S, H_q,  D]  compact or a slab column slice (token_stride)
        k: torch.Tensor,  # [B, S, H_kv, D]
        v: torch.Tensor,  # [B, S, H_kv, D]
        o: torch.Tensor,  # [B, S, H_q,  D]  compact, written
        lse: Optional[torch.Tensor] = None,  # [B, H_q, S] fp32, iff declared want_lse; THD: [1, H_q, T] (head-major packed)
        seq_lens: Optional[
            torch.Tensor
        ] = None,  # [B] int32 per-batch visible KV length, iff declared seq_lens_present; THD (REQUIRED): [B] lengths / [B+1] prefix sums
        workspace: Optional[torch.Tensor] = None,  # dense: unused (no GMEM scratch); THD (REQUIRED): the engine arm holding the packed metadata
        current_stream=None,
        gate: Optional[torch.Tensor] = None,  # [B, S, H_q, D] slab slice or compact; fuse_gate only (refused otherwise)
        descale_q: Optional[torch.Tensor] = None,  # refused: the quantized pipelines' operands
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,  # [T, top_k] or [B, S, top_k] int32 contiguous (the block's form check ran); paged + shared list: [B, top_k]
        block_lens: Optional[torch.Tensor] = None,  # [T] or [B, S] int32 contiguous, optional per call; paged + shared list: [B]
        # APPENDED (kv_source="paged", all four REQUIRED there, REFUSED otherwise): the pools stage (4) gathers K / V from (the
        # declared descriptors' shape and strides exactly), the whole (B, max_pages) int32 block table at the declared strides, the
        # [B] int32 LOGICAL KV lengths (prefix + the step's tokens; bound to the KV side only).  `k` / `v` above are then unread.
        block_table: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> None:
        """Hand the block's BSHD views over AS THEY ARE (no transpose, no copy) plus the index tensors viewed flat.  The
        dense stage's gate / quantization operands have no sparse arm and are refused rather than ignored; the LSE and the
        KV-length presence must match the declaration (both are compiled into the specialization).  Under ``thd=True`` the
        same ``[1, T, H, D]`` views ARE the adapter's packed operands; ``seq_lens`` (REQUIRED) binds both length sides and
        ``workspace`` (REQUIRED) is the engine arm the adapter's packed metadata lands in.  Under ``kv_source="paged"`` the
        pools replace ``k`` / ``v``, ``kv_lens`` binds the KV side (``seq_lens`` REFUSED), the table rides along, and
        ``workspace`` (REQUIRED under a split) is the engine arm the fp32 partials land in."""
        if self._impl is None or not self._compiled:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        if self.fuse_gate and gate is None:
            raise ValueError(f"{self.name}: fuse_gate=True requires the GATE tensor at execute")
        if not self.fuse_gate and gate is not None:
            raise ValueError(f"{self.name}: gate is the fused epilogue gate's operand; this stage was declared without fuse_gate")
        if any(x is not None for x in (descale_q, descale_k, descale_v, scale_o, sf_q, sf_k, sf_v)):
            raise ValueError(f"{self.name}: descales / scale_o / scale-factor blobs are the quantized pipelines' operands; the sparse core is bf16 / f16")
        if block_ids is None:
            raise ValueError(f"{self.name}: block_ids is required at execute (the per-query block list)")
        paged_args = (block_table, kv_lens, k_cache, v_cache)
        if self.paged:
            # The DECODE / extend-with-prefix mode: the pools as K / V, kv_lens as seq_kv_lens (one fact, the KV side only: every
            # row of the step is live unless its position is negative), the whole table, the partials workspace under a split.
            if any(x is None for x in paged_args):
                raise ValueError(f"{self.name}: kv_source='paged' requires block_table, kv_lens, k_cache and v_cache at execute")
            if seq_lens is not None:
                raise ValueError(
                    f"{self.name}: seq_lens is the block's own self-attention length (a Q-side length); over the pools the per-sequence KV length is "
                    "kv_lens, bound to the KV side only -- refused rather than read as a Q length"
                )
            if self.split_kv > 1 and workspace is None:
                raise ValueError(f"{self.name}: split_kv={self.split_kv} requires workspace at execute (the engine arm the fp32 partials land in)")
            if self.want_lse and lse is None:
                raise ValueError(f"{self.name}: declared with return_lse=True; execute needs lse ([B, H_q, S] fp32)")
            if not self.want_lse and lse is not None:
                raise ValueError(f"{self.name}: lse given, but this stage was declared without return_lse (the LSE presence is compiled in)")
            g = self.geom
            rows = self.batch if self.list_per_sequence else self.batch * self.seq_len
            q, o, gate = self._declared_strides(q, "q"), self._declared_strides(o, "o"), self._declared_strides(gate, "gate")
            self._impl.execute(
                stream=current_stream,
                q=q,
                k=k_cache,
                v=v_cache,
                o=o,
                lse=lse,
                block_ids=block_ids.view(rows, g.qsa.top_k),
                block_lens=None if block_lens is None else block_lens.view(rows),
                seq_kv_lens=kv_lens,
                block_table=block_table,
                gate=gate,
                workspace=workspace if self.split_kv > 1 else None,
            )
            return
        if any(x is not None for x in paged_args):
            raise ValueError(
                f"{self.name}: block_table / kv_lens / k_cache / v_cache are consumed under kv_source='paged' only; this stage was declared kv_source='self'"
            )
        if self.thd:
            if seq_lens is None:
                raise ValueError(f"{self.name}: thd=True requires seq_lens at execute (the packed per-sequence lengths, for the Q and the KV side alike)")
            if workspace is None:
                raise ValueError(f"{self.name}: thd=True requires workspace at execute (the engine arm the packed metadata is written to)")
        else:
            if self.seq_lens_present and seq_lens is None:
                raise ValueError(f"{self.name}: declared with seq_lens_present=True; execute needs seq_lens ([B] int32, the per-batch visible KV length)")
            if not self.seq_lens_present and seq_lens is not None:
                raise ValueError(f"{self.name}: seq_lens given, but this stage was declared without seq_lens_present (the KV-length presence is compiled in)")
        if self.want_lse and lse is None:
            raise ValueError(f"{self.name}: declared with return_lse=True; execute needs lse ([B, H_q, S] fp32)")
        if not self.want_lse and lse is not None:
            raise ValueError(f"{self.name}: lse given, but this stage was declared without return_lse (the LSE presence is compiled in)")
        g, t = self.geom, self.batch * self.seq_len
        q, k, v, o, gate = (self._declared_strides(x, nm) for x, nm in ((q, "q"), (k, "k"), (v, "v"), (o, "o"), (gate, "gate")))
        # THD: ONE lengths tensor serves both sides (self-attention) and the metadata lands in the engine arm; the adapter
        # holds each to its declaration (`_bind`: shape / strides / dtype) and sizes the workspace (`_check_workspace`).
        thd_kw = dict(seq_q_lens=seq_lens, workspace=workspace) if self.thd else {}
        self._impl.execute(
            stream=current_stream,
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            block_ids=block_ids.view(t, g.qsa.top_k),
            block_lens=None if block_lens is None else block_lens.view(t),
            seq_kv_lens=seq_lens,
            gate=gate,
            **thd_kw,
        )


class _UpcastSparseSdpa(_Stage):
    """(4) under ``kv_cache_dtype=torch.float8_e4m3fn`` on the block-sparse DECODE mode (``kv_source="paged"`` on a dense
    ``[B, S_q <= 4]`` QsaSpec declaration): the e4m3 pools' LISTED rows gathered and cast back to bf16, then the bf16 d256 DECODE
    tile over the result as a plain padded dense SDPA -- the sparse core untouched by fp8.

    Two launches, no new kernel: (a) ``kernels/qsa_kv_upcast.py`` reads, per (token, KV head) ITEM, the sequence's SHARED list
    (``block_ids [B, top_k]`` + ``block_lens [B]``, anchored at the step-0 position) and the open tail over ``[0, kv_lens[b])``
    through ``block_table[b]`` on the bottom-right diagonal, dequantizes each e4m3 row (``bf16(fp32(code) x scale)``; ``k_scale`` /
    ``v_scale`` read on device) and writes the visible rows COMPACTED in list order into the workspace scratch ``kv_gathered
    [items, 2, 2176, 256]`` -- every row past the item's count 0.0 (``zero_fill="full"``), so a tile the decode kernel reads past
    the length is FINITE whatever the workspace held -- plus ``kv_len_t [items]`` (0 = a dead item: a dead entry, or a row at a
    negative position); (b) the Rubin d256 decode tile (``SdpaFwdDslSm100``; its ``kernel_template`` pinned to ``decode_d256_f16``
    at compile) over Q as ``[items, G, 1, D]`` (the COMPACT normed Q viewed per item: the G query heads of the item's KV head -- a
    view of the compact Q only, which is why the mode norms + rotates out of place), K / V ``[items, 1, 2176, D]`` (the scratch's two
    halves), O as ``[items, G, 1, D]`` (the workspace ``o`` slot: the same bytes as ``[T, H_q, D]``), NO causal mask (the gathered
    rows ARE the visible set), ``seq_kv_lens = kv_len_t`` on the KV side only (a dead item is the tile's empty-range select: O = 0),
    the group packed (``pack_gqa``), unsplit.  No LSE, no epilogue gate (stage (5) runs as its own launch), no split: typed
    declines at the block's declaration.

    Both ``has_block_lens`` artifacts of the cast are compiled at plan time (Rule 4); the cast kernel's DSL / arch gate runs inside
    ``check_support`` BEFORE its module is imported (Rule 7: the module imports the DSL at import time).  The scratch is the block's
    workspace (``_plan_workspace(kv_cache_e4m3=True, upcast_items=)``); the decode tile's own scratch (0 unsplit) folds through
    :meth:`scratch_workspace_bytes` like every other SDPA stage's.  The list contract (``top_k``, the 4-token block, the group of at
    most 16 heads) is the sparse record's, re-checked here so the stage is honest on its own.
    """

    name = "sdpa_sparse_e4m3"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        page_size: int,
        k_cache_desc: Optional[TensorDesc],
        v_cache_desc: Optional[TensorDesc],
        block_table_desc: Optional[TensorDesc],
    ) -> None:
        if geometry.qsa is None:
            raise ValueError(f"{self.name}: the geometry declares no QsaSpec (geometry.qsa is None); the e4m3 read is the block-sparse decode mode's")
        if dtype != torch.bfloat16:
            raise NotImplementedError(
                f"{self.name}: bf16 activations only (the gathered cast writes bf16 rows and the decode tile reads a bf16 Q), got {dtype}"
            )
        if not (1 <= int(seq_len) <= _UPCAST_MAX_S_Q):
            raise NotImplementedError(f"{self.name}: the shared-list form serves 1 <= S_q <= {_UPCAST_MAX_S_Q} rows per sequence, got S_q={seq_len}")
        if int(page_size) <= 0 or int(page_size) % geometry.qsa.block_size:
            raise ValueError(f"{self.name}: page_size must be a positive multiple of {geometry.qsa.block_size}, got {page_size}")
        if k_cache_desc is None or v_cache_desc is None or block_table_desc is None:
            raise ValueError(f"{self.name}: the k_cache / v_cache / block_table descriptors are required")
        for nm, dsc in (("k_cache", k_cache_desc), ("v_cache", v_cache_desc)):
            if dsc.dtype != KV_CACHE_E4M3:
                raise ValueError(f"{self.name}: sample_{nm} must be torch.float8_e4m3fn (the e4m3 KV cache), got {dsc.dtype}")
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = device
        self.page_size = int(page_size)
        self.k_cache_desc, self.v_cache_desc, self.block_table_desc = k_cache_desc, v_cache_desc, block_table_desc
        self.items = self.batch * self.seq_len * int(geometry.h_kv)
        self.group = int(geometry.gqa_ratio)
        self._recipes = None  # {has_block_lens: QsaKvUpcastRecipe}
        self._impl = None
        self._compiled = False

    def _device_index(self) -> int:
        dev = torch.device(self.device)
        return int(dev.index) if dev.index is not None else int(torch.cuda.current_device())

    def _scratch_descs(self):
        """The decode tile's operands over the scratch, storage-free: Q / O per item ``(items, G, 1, D)`` at the compact strides (the
        unit S dim at the token stride ``G x D``: ``execute`` binds an exact ``as_strided`` view), K / V ``(items, 1, 2176, D)`` at the
        scratch's strides ``(2 x 2176 x D, 2176 x D, D, 1)``."""
        g, d, items, G = self.geom, self.geom.d_head, self.items, self.group
        kv_shape, kv_stride = (items, 1, _UPCAST_ROWS, d), (2 * _UPCAST_ROWS * d, _UPCAST_ROWS * d, d, 1)
        order = tuple(i for i, _ in sorted(enumerate(kv_stride), key=lambda x: (x[1], kv_shape[x[0]])))
        k = TensorDesc(dtype=self.dtype, shape=kv_shape, stride=kv_stride, stride_order=order, device=self.device, name="k")
        v = TensorDesc(dtype=self.dtype, shape=kv_shape, stride=kv_stride, stride_order=order, device=self.device, name="v")
        return _bhsd_desc(items, G, 1, d, self.dtype, self.device, "q"), k, v, _bhsd_desc(items, G, 1, d, self.dtype, self.device, "o")

    def check_support(self) -> None:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

        too_old = cutedsl_requirement_error("the gated attention block's e4m3 KV-cache decode mode")
        if too_old is not None:
            raise NotImplementedError(too_old)
        g = self.geom
        _check_qsa_geometry_against_record(g, self.dtype)
        if g.d_head != _UPCAST_D:
            raise NotImplementedError(f"{self.name}: the gathered cast serves d_head == {_UPCAST_D}, got {g.d_head}")
        cc = tuple(torch.cuda.get_device_capability(self.device))
        if cc != _SM107_CC:
            raise NotImplementedError(f"gated_attention_block targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found SM{cc[0]}{cc[1]}")
        from .kernels import qsa_kv_upcast as up  # imports the DSL: after the floor gate above

        if (up.ROWS, up.D, up.MTP_MAX_S_Q) != (_UPCAST_ROWS, _UPCAST_D, _UPCAST_MAX_S_Q):
            raise RuntimeError(
                f"{self.name}: the workspace carve's scratch geometry ({_UPCAST_ROWS} x {_UPCAST_D}, S_q <= {_UPCAST_MAX_S_Q}) and kernels/qsa_kv_upcast.py's "
                f"({up.ROWS} x {up.D}, S_q <= {up.MTP_MAX_S_Q}) disagree -- fix the drift, neither reading is the truth alone"
            )
        why = up.dsl_decline_reason(self._device_index())
        if why is not None:
            raise NotImplementedError(why)
        up.validate_params(h_kv=g.h_kv, top_k=g.qsa.top_k, page_size=self.page_size, zero_fill="full")
        # the pools' DECLARED form against the cast kernel's contract (e4m3, D contiguous, 16-byte strides); the real tensors are held to
        # it again by run_qsa_kv_upcast at execute
        for nm, dsc in (("k_cache", self.k_cache_desc), ("v_cache", self.v_cache_desc)):
            shape, stride = tuple(int(x) for x in dsc.shape), tuple(int(x) for x in dsc.stride)
            if len(shape) != 4 or shape[1:] != (g.h_kv, self.page_size, g.d_head) or stride[3] != 1 or any(st % 16 for st in stride[:3]):
                raise ValueError(
                    f"{self.name}: sample_{nm} must be an e4m3 [num_pages, {g.h_kv}, {self.page_size}, {g.d_head}] pool with D contiguous and every other "
                    f"stride a multiple of 16 (the cast reads rows in 16-byte vectors), got shape {shape} strides {stride}"
                )
        bt = self.block_table_desc
        if int(bt.stride[1]) != 1 or (int(bt.shape[0]) > 1 and int(bt.stride[0]) != int(bt.shape[1])):
            raise ValueError(f"{self.name}: sample_block_table must be contiguous (the cast bulk-reads the whole table), got strides {tuple(bt.stride)}")
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import SCHED_NATURAL
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.api_dsl import SdpaFwdDslSm100

        q, k, v, o = self._scratch_descs()
        self._impl = SdpaFwdDslSm100(
            q,
            k,
            v,
            o,
            None,
            is_causal=False,
            causal_bottom_right=False,
            scale_softmax=g.scale,
            seq_kv_lens_present=True,
            sched_policy=SCHED_NATURAL,
            split_kv=1,
            pack_gqa=True if self.group > 1 else None,
        )
        self._impl.check_support()

    def compile(self) -> None:
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call check_support() before compile()")
        from .kernels import qsa_kv_upcast as up

        g, dev = self.geom, self._device_index()
        # BOTH block_lens variants at plan time: the caller passes block_lens per call or not, and nothing may compile on the execute
        # path (Rule 4).
        self._recipes = {
            hb: up.compile_qsa_kv_upcast(
                h_kv=g.h_kv, top_k=g.qsa.top_k, page_size=self.page_size, list_per_sequence=True, has_block_lens=hb, zero_fill="full", device=dev
            )
            for hb in (True, False)
        }
        self._impl.compile()
        got = getattr(self._impl, "kernel_template", None)
        if got != DECODE_TILE:
            raise RuntimeError(
                f"{self.name}: the dense core over the gathered scratch must be the Rubin d256 decode tile ({DECODE_TILE}); the adapter compiled {got!r} for "
                f"Q [{self.items}, {self.group}, 1, {g.d_head}] over K / V [{self.items}, 1, {_UPCAST_ROWS}, {g.d_head}] -- fix the drift"
            )
        self._compiled = True

    def scratch_workspace_bytes(self) -> int:
        """The decode tile's own scratch over the scratch (0 unsplit), folded into the block's workspace like every SDPA stage's."""
        if self._impl is None:
            raise RuntimeError(f"{self.name}: call check_support() before scratch_workspace_bytes()")
        return int(self._impl.scratch_workspace_bytes())

    def execute(
        self,
        q: torch.Tensor,  # [T, H_q, D] the COMPACT normed Q
        o: torch.Tensor,  # [T, H_q, D] the O slot, written
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        block_ids: torch.Tensor,
        block_lens: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        scratch: torch.Tensor,  # [items, 2, 2176, D] bf16 (the workspace slot)
        kv_len_t: torch.Tensor,  # [items] int32 (the workspace slot)
        workspace: Optional[torch.Tensor] = None,
        current_stream=None,
    ) -> None:
        from .kernels import qsa_kv_upcast as up

        if not self._compiled:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        g = self.geom
        items, G, d, t = self.items, self.group, g.d_head, self.batch * self.seq_len
        for nm, x in (("q", q), ("o", o)):
            if tuple(int(s) for s in x.shape) != (t, g.h_q, d) or tuple(int(s) for s in x.stride()) != (g.h_q * d, d, 1):
                raise ValueError(
                    f"{self.name}: {nm} must be the compact [{t}, {g.h_q}, {d}] buffer (the per-item view is a view of it alone), got {tuple(x.shape)} / {tuple(x.stride())}"
                )
        if tuple(int(s) for s in scratch.shape) != (items, 2, _UPCAST_ROWS, d) or scratch.dtype != self.dtype:
            raise ValueError(
                f"{self.name}: scratch must be the [{items}, 2, {_UPCAST_ROWS}, {d}] {self.dtype} workspace slot, got {tuple(scratch.shape)} {scratch.dtype}"
            )
        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(q.device).cuda_stream
        # (a) the gathered cast: the artifact of this call's block_lens presence (both compiled at plan time)
        up.run_qsa_kv_upcast(
            self._recipes[block_lens is not None],
            k_pool=k_cache,
            v_pool=v_cache,
            block_table=block_table,
            block_ids=block_ids,
            block_lens=block_lens,
            seq_kv_lens=kv_lens,
            k_scale=k_scale,
            v_scale=v_scale,
            out_k=scratch[:, 0],
            out_v=scratch[:, 1],
            kv_len_t=kv_len_t,
            s_q=self.seq_len,
            stream=stream,
        )
        # (b) the decode tile over the scratch: Q / O per item at the DECLARED strides (an exact as_strided view of the compact buffer;
        # the unit S dim's stride is unobservable), K / V the scratch halves, the lengths bound on the KV side only (every Q row live;
        # a dead item has zero keys -> the tile's empty-range select -> O = 0)
        qv = q.as_strided((items, G, 1, d), (G * d, d, G * d, 1), q.storage_offset())
        ov = o.as_strided((items, G, 1, d), (G * d, d, G * d, 1), o.storage_offset())
        self._impl.execute(
            qv,
            scratch[:, 0].unsqueeze(1),
            scratch[:, 1].unsqueeze(1),
            ov,
            lse_tensor=None,
            seq_q_lens=None,
            seq_kv_lens=kv_len_t,
            workspace=workspace,
            current_stream=cuda.CUstream(stream),
        )


class _ElementwiseStage(_Stage):
    """Shared body of stages (5) and (3b): one pass over ``[T, H, D]``.

    Both are the same kernel with a ``const_expr`` on whether a gate operand
    exists — see ``kernels/elementwise.py``. Being pure streaming with no
    reduction and no shuffle, this is the closest thing in the block to a copy
    and should sit closest to the HBM ceiling.
    """

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        heads: int,
        has_gate: bool,
        name: str,
        use_pdl: bool = False,
    ) -> None:
        self.name = name
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.heads = int(heads)
        self.has_gate = bool(has_gate)
        self.use_pdl = bool(use_pdl)  # the kernel's programmatic-dependent-launch form (kernels/elementwise.py: the launch attribute + the griddepcontrol pair)
        self._recipe = None

    def check_support(self) -> None:
        from .kernels.elementwise import validate_shape

        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: bf16/f16 only, got {self.dtype}")
        validate_shape(self.geom.d_head, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        from .kernels.elementwise import compile_elementwise_gate

        self._recipe = compile_elementwise_gate(
            dtype=self.dtype,
            h=self.heads,
            d=self.geom.d_head,
            has_gate=self.has_gate,
            threads_per_cta=_ELEMENTWISE_THREADS,
            rows_per_group=_ELEMENTWISE_ROWS_PER_GROUP,
            const_head_count=_ELEMENTWISE_CONST_HEAD_COUNT,
            use_pdl=self.use_pdl,
        )

    def moved_bytes(self) -> int:
        """HBM traffic of one launch — the denominator for the SOL number."""
        from .kernels.elementwise import moved_bytes

        return moved_bytes(self.batch * self.seq_len, self.heads, self.geom.d_head, elem_bytes=_itemsize(self.dtype), has_gate=self.has_gate)

    def _run(self, src, gate, dst, current_stream):
        from .kernels.elementwise import run_elementwise_gate

        if self._recipe is None:
            raise RuntimeError("call compile() before execute()")
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(src.device).cuda_stream
        run_elementwise_gate(self._recipe, src, gate, dst, stream=stream)


class _SigmoidGate(_ElementwiseStage):
    """(5) ``O_gated = O * sigmoid(GATE)``, elementwise over ``[T, H_q, D]``.

    **Hazard, and the reason ordering matters:** it must consume the SDPA's
    *substituted* O. For a fully-masked row the epilogue selects ``O := 0``;
    gating accumulator residue instead of that zero propagates NaN, and residue
    really can be a NaN bit pattern (``* 0`` is not a fix -- the substitution is
    a SELECT). Under context parallelism a rank can legitimately hold a chunk
    whose KV range is empty, so this is reachable, not theoretical.

    GATE is read as a strided COLUMN SLICE of the fused projection — no repack.
    Under inference O is gated IN PLACE, so ``O_gated`` costs no buffer; under
    training (``save_for_backward``) the SDPA wrote the PRE-gate O into the
    caller's ``saved.o`` (the backward's ``dG`` operand, which must survive), so
    the same kernel gates OUT of place into the workspace ``o_gated`` slot that
    stage (6) then reads.

    **Fusion status: folding this into the SDPA epilogue is the obvious next
    increment** — a pure epilogue change, one extra input in O's exact layout,
    removing a full HBM round trip of O. It pulls against the backward, which
    wants pre-gate ``O`` for ``dG``: a fused epilogue would have to write both,
    or training keeps this stage split. Decide that deliberately.
    """

    def __init__(self, geometry, *, batch, seq_len, dtype, use_pdl: bool = False):
        super().__init__(geometry, batch=batch, seq_len=seq_len, dtype=dtype, heads=geometry.h_q, has_gate=True, name="sigmoid_gate", use_pdl=use_pdl)

    def execute(self, o: torch.Tensor, gate: torch.Tensor, out: torch.Tensor, current_stream=None) -> None:
        self._run(o, gate, out, current_stream)


class _VCompaction(_ElementwiseStage):
    """(3b) copy V out of the fused projection into a compact buffer.

    **This stage exists only because stage (1) is the UNFORKED FROST GEMM.** Q
    and K are de-interleaved for free by (2)+(3), which already read and write
    them; V is untouched between the projection and the SDPA, so without this it
    would reach the SDPA as a padded-stride view. The historical adapter copied
    such layouts implicitly; this block keeps its explicit compaction stage
    and hands compact storage to the prepared SDPA plan (Rule 2).

    Forking stage (1) to write four compact buffers (§ 1) deletes this outright.
    Until then it is 1/16 of Q's traffic, which is why it is an acceptable v1.
    """

    def __init__(self, geometry, *, batch, seq_len, dtype):
        super().__init__(geometry, batch=batch, seq_len=seq_len, dtype=dtype, heads=geometry.h_kv, has_gate=False, name="compact_v")

    def execute(self, v_src: torch.Tensor, v_dst: torch.Tensor, current_stream=None) -> None:
        self._run(v_src, None, v_dst, current_stream)


class _BandCopy(_ElementwiseStage):
    """(3g) TRAINING, gate-copy save mode only: copy one ``h_q``-head band of the
    fused projection into a compact caller buffer -- the GATE into ``saved.gate``
    (always), the PRE-norm Q into ``saved.q_pre`` when the caller passed one
    (``saved.k_pre`` rides :class:`_VCompaction`'s ``h_kv`` recipe on the bf16 pipeline; the
    quantized pipelines build no ``_VCompaction`` -- their quantize stages compact V -- so a
    quantized gate-copy block builds a second instance at ``h_kv`` heads, ``k_pre_compaction``).

    The elementwise kernel's ``has_gate=False`` arm at ``h_q`` heads: zero new
    kernel code, one launch per band, ``2 x 16 KiB/token`` moved at the 397B
    geometry.  It exists because the gate-copy mode keeps the slab in the
    WORKSPACE (dead the moment ``execute`` returns) and the backward needs the
    GATE from caller-owned storage; the proj_slab save mode has no such stage
    (the slab itself is the record).  Built only under ``saved_gate_copy``.
    """

    def __init__(self, geometry, *, batch, seq_len, dtype, heads: Optional[int] = None, name: str = "gate_compaction"):
        """Build the band copy as a ``has_gate=False`` elementwise stage at ``h_q`` heads; ``heads`` / ``name`` (appended) select the
        ``h_kv`` twin that copies ``k_pre`` on a quantized gate-copy block."""
        super().__init__(geometry, batch=batch, seq_len=seq_len, dtype=dtype, heads=geometry.h_q if heads is None else int(heads), has_gate=False, name=name)

    def execute(self, src: torch.Tensor, dst: torch.Tensor, current_stream=None) -> None:
        self._run(src, None, dst, current_stream)


class _QsaPrepare(_Stage):
    """(2+3+4w) The PREPARE fusion: stages (2)+(3) and the serving write-through (4w) as ONE launch
    (``kernels/qsa_prepare.py``), built under ``fuse_prepare=True`` in place of :class:`_QkNormRope` + :class:`_CacheWrite`.

    Q / K are RMSNormed and rotated in place in the slab (``qk_norm_rope.py``'s chain, op for op) and the post-RoPE K, the
    V and -- when the geometry declares the indexer band -- the raw indexer key are written into the paged pools at
    ``slot_mapping`` from the same registers: a pool row IS the row attention consumes, and every output is BITWISE the two
    stages it replaces (the K pool = the slab's K band, the V pool = the V band, the raw pool = ``index_k_raw_view``).  The
    gate split costs nothing in either form (the GATE band is addressed in place).  A PERFORMANCE knob of the block:
    the same function, one launch fewer (two with the band).

    The kernel also carries the indexer's decode-step arms -- the queries' norm + RoPE (the compress kernel at pool 1),
    the raw-key ring and the incremental compress (its step form) -- traced by presence (``index_arms``, the block's
    :class:`QsaPrepareIndexArms` record).  Under the in-block indexer's DECODE step over the pools
    (``QsaSpec(index_source="indexer")`` with ``kv_source="paged"``) the block binds the three PAGED arms in ONE launch with
    the norm + RoPE and the K / V write-through: the indexer queries normed + rotated into the workspace slot ``ix_q``, the raw
    key into the stack's ring (``index_state_cache`` behind ``index_state_block_table``, or at the stack's own ring slots) and
    the block a committed row completes -- pooled, normed and rotated at the block START from the position-indexed
    ``cos_pos`` / ``sin_pos`` -- into the paged compressed-key cache ``index_kv_cache`` on the MAIN block table, the positions
    from ``kv_lens`` on device, the per-row slot report into ``ix_cslot``; under ``rope_position_tables_only`` the Q / K and
    indexer-query rotations read the same tables (``cos`` / ``sin`` then ``None``).  Bitwise the unfused twin (the norm + RoPE
    pass, the K / V write, the pool-1 query norm, the paged compress step: :class:`_IndexerDecode`'s unfused arm).  On every
    other band declaration the stage binds the raw-key pool alone (the write-through), so the knob changes no input there.

    No SMEM, no mbarrier (the kernel's module docstring: both pre-development tables empty by construction).  Needs no
    workspace -- the pools and the slot mapping are the caller's -- so ``get_workspace_size`` is the unfused block's.
    ``geometry.qk_norm=False`` traces the RoPE-only arm (the weight slots ``None``, as ``_QkNormRope`` does); ``geometry.rope_dim=0`` is a
    typed decline at ``check_support`` with or without ``qk_norm`` (the tables would be zero-width operands).  ``use_pdl`` (the
    block's resolved switch) launches it with the programmatic-dependent-launch attribute and traces the kernel's griddepcontrol
    pair, as the two stages it replaces do.
    """

    name = "qsa_prepare"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        page_size: int,
        use_pdl: bool = False,
        index_arms=None,  # appended: the QsaPrepareIndexArms record the block builds (None = the band's raw-key pool arm alone, as before)
        index_cache_dtype=None,  # appended: the indexer cache dtype of the decode arms (None = the activation dtype; e4m3 = the e4m3 store arm)
    ) -> None:
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.page_size = int(page_size)
        # The indexer cache dtype the decode arms write (QsaSpec.index_cache_dtype, resolved by the block): the query arm's ix_q and the compress
        # arm's paged cache in e4m3 under the attribute (kernels/qsa_prepare.py's e4m3 store arm: the e4m3 cast of the bf16 row); bf16 as before.
        self.index_cache_dtype = dtype if index_cache_dtype is None else index_cache_dtype
        # The kernel's programmatic-dependent-launch form (kernels/qsa_prepare.py: the launch attribute + the griddepcontrol pair -- the wait
        # before the first global read, the release after the body's last store), taken from the block as the two stages it replaces take it.
        self.use_pdl = bool(use_pdl)
        self.index_band = bool(geometry.index_band)
        self._arms = index_arms
        self._recipes = None  # {slot dtype: recipe}

    def _index_arms(self):
        """The indexer arms the launch traces: the block's record (the decode step's three paged arms), else -- on a band declaration --
        the raw-key pool alone; ``None`` without the band."""
        from .cache_write_contract import QsaPrepareIndexArms  # DSL-free (the kernel module re-exports it): callable at declaration

        if self._arms is not None:
            return self._arms
        q = self.geom.qsa
        return QsaPrepareIndexArms(heads=q.index_heads, head_dim=q.index_head_dim, raw_pool=True, query=False, compress=False) if self.index_band else None

    @property
    def decode_arms(self) -> bool:
        """True when the launch carries the decode step's compress arm (the ring + the paged compressed-key cache)."""
        return self._arms is not None and bool(self._arms.compress)

    def check_support(self) -> None:
        # Rule 7: a too-old CuTe DSL reads as a VERSION decline here, before the kernel module (which imports the DSL) is loaded.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

        too_old = cutedsl_requirement_error("the gated attention block's fused prepare launch")
        if too_old is not None:
            raise NotImplementedError(too_old)
        from .kernels.qsa_prepare import validate_prepare_shape

        g = self.geom
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: bf16 / f16 only (the activation dtype of the slab and the pools), got {self.dtype}")
        if g.rope_dim == 0:
            # Every rope_dim == 0 geometry, with or without qk_norm: the kernel binds cos / sin as [T, rope_dim] operands and its artifact
            # is traced with one-column tables (`table_dim = rope_dim if rope_dim else 1`), so the [T, 0] tables execute would hand it are
            # not operands it takes -- a typed decline here, not a shape mismatch at the launch.
            raise ValueError(
                f"{self.name}: fuse_prepare=True with geometry.rope_dim=0: the fused launch would bind the RoPE tables cos / sin as zero-width "
                "[T, 0] operands, which its artifact (traced with one-column tables) does not take -- the knob needs rope_dim > 0 (with or without qk_norm)"
            )
        validate_prepare_shape(g.d_head, g.rope_dim, self.page_size, _ELEMENTWISE_THREADS, g.qsa.index_head_dim if self.index_band else None)
        if self.decode_arms:
            # the compress arm's row geometry at the indexer head dim (the step form's own rule: 16-byte lanes over D_i, the RoPE half inside it)
            from .kernels.qsa_compress import validate_compress_shape

            validate_compress_shape(g.qsa.index_head_dim, g.rope_dim, self._arms.pool, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        """Both slot dtypes are compiled (int32 and int64 are the two the serving stacks hand over; the dtype is a compile
        key), so ``execute`` dispatches on a guaranteed cache hit (Rule 4) whichever one arrives."""
        from .kernels.qsa_prepare import compile_qsa_prepare

        g = self.geom
        self._recipes = {
            slot_dtype: compile_qsa_prepare(
                dtype=self.dtype,
                h_q=g.h_q,
                h_kv=g.h_kv,
                d=g.d_head,
                rope_dim=g.rope_dim,
                eps=g.qk_norm_eps,
                page_size=self.page_size,
                slot_dtype=slot_dtype,
                apply_norm=g.qk_norm,
                index=self._index_arms(),
                threads_per_cta=_ELEMENTWISE_THREADS,
                use_pdl=self.use_pdl,
                index_cache_dtype=self.index_cache_dtype,
            )
            for slot_dtype in (torch.int32, torch.int64)
        }

    def moved_bytes(self, slot_dtype=torch.int32) -> int:
        """HBM traffic of the one launch -- the denominator for the SOL number."""
        from .kernels.qsa_prepare import moved_bytes

        g = self.geom
        return moved_bytes(
            self.batch * self.seq_len,
            g.h_q,
            g.h_kv,
            g.d_head,
            elem_bytes=_itemsize(self.dtype),
            slot_bytes=_itemsize(slot_dtype),
            index=self._index_arms(),
            index_cache_bytes=_itemsize(self.index_cache_dtype),
        )

    def execute(
        self,
        q: torch.Tensor,  # [T, H_q,  D] the slab's Q band (rewritten in place)
        k: torch.Tensor,  # [T, H_kv, D] the K band (rewritten in place; its post-RoPE rows land in k_cache)
        v: torch.Tensor,  # [T, H_kv, D] the V band (read; its rows land in v_cache)
        w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],
        cos: torch.Tensor,  # [B, S, ROPE_DIM] (or [T, ROPE_DIM]) the attention's per-token tables
        sin: torch.Tensor,
        k_cache: torch.Tensor,  # [num_pages, H_kv, page_size, D]
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # [T] int32 / int64
        index_src: Optional[torch.Tensor] = None,  # [T, 1, index_head_dim] the raw indexer key band; given iff the band is declared
        index_cache: Optional[torch.Tensor] = None,  # [num_pages, page_size, index_head_dim] its pool (the raw-pool arm)
        current_stream=None,
        # APPENDED -- the decode step's three paged arms (given iff the stage was built with the decode arms record; refused otherwise):
        index_q_src: Optional[torch.Tensor] = None,  # [T, index_heads, index_head_dim] the band's indexer-query columns (read in place)
        index_q_out: Optional[torch.Tensor] = None,  # [T, index_heads, index_head_dim] compact: the normed + rotated queries (the workspace slot)
        w_iq_norm: Optional[torch.Tensor] = None,  # [index_head_dim] the queries' RMSNorm weight
        ring: Optional[torch.Tensor] = None,  # [num_ring_blocks, cap, 1, index_head_dim] the stack's raw-key ring
        index_k_compressed: Optional[torch.Tensor] = None,  # [num_pages, P / block_size, 1, index_head_dim] the paged compressed-key cache
        w_ik_norm: Optional[torch.Tensor] = None,  # [index_head_dim] the compressed keys' RMSNorm weight
        cos_pos: Optional[torch.Tensor] = None,  # [1 or B, S_pos, rope_dim] the position-indexed tables (the block START of a completing block)
        sin_pos: Optional[torch.Tensor] = None,
        compressed_slot: Optional[torch.Tensor] = None,  # [B, S_q] int32: the per-row slot report (the workspace slot)
        kv_lens: Optional[torch.Tensor] = None,  # [B] int32: the KV length AFTER the step (pos0 = kv_lens - S_q on device)
        block_table: Optional[torch.Tensor] = None,  # [B, max_pages] int32: the MAIN block table (the compressed-key cache's pages)
        index_state_block_table: Optional[torch.Tensor] = None,  # [B, 1] int32: the ring block per sequence
        index_state_slot_mapping: Optional[torch.Tensor] = None,  # [T] of slot_mapping's dtype iff the ring-slots arm was declared
    ) -> None:
        """``cos`` / ``sin`` are ``None`` under the RoPE-table arm (``rope_table`` in the arms record: every rotation reads ``cos_pos`` /
        ``sin_pos`` at the row's device-derived position); the kernel's runner holds every arm to its artifact in both directions."""
        from .kernels.qsa_prepare import run_qsa_prepare

        if self._recipes is None:
            raise RuntimeError("call compile() before execute()")
        g = self.geom
        _check_norm_weights_agree(g.qk_norm, w_q_norm, w_k_norm)
        arms = self._index_arms()
        raw_pool = arms is not None and arms.raw_pool
        if (index_src is None) != (arms is None) or (index_cache is None) != (not raw_pool):
            raise ValueError(
                f"{self.name}: index_src is given exactly when the geometry declares the indexer band, index_cache exactly when the launch carries "
                "the raw-key POOL arm (the decode step keeps the raw key in the ring instead)"
            )
        decode_kw = dict(
            index_q_src=index_q_src,
            index_q_out=index_q_out,
            w_iq_norm=w_iq_norm,
            ring=ring,
            index_k_compressed=index_k_compressed,
            w_ik_norm=w_ik_norm,
            cos_pos=cos_pos,
            sin_pos=sin_pos,
            compressed_slot=compressed_slot,
            kv_lens=kv_lens,
            block_table=block_table,
            index_state_block_table=index_state_block_table,
            index_state_slot_mapping=index_state_slot_mapping,
        )
        if not self.decode_arms and any(x is not None for x in decode_kw.values()):
            raise ValueError(
                f"{self.name}: the decode step's arms ({', '.join(nm for nm, x in decode_kw.items() if x is not None)}) belong to a stage built with them"
            )
        if slot_mapping.dtype not in self._recipes:
            raise ValueError(f"{self.name}: slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
        t = self.batch * self.seq_len
        table_arm = arms is not None and arms.rope_table
        if table_arm != (cos is None and sin is None):
            raise ValueError(
                f"{self.name}: cos / sin are None exactly under the RoPE-table arm (rope_position_tables_only: every rotation reads cos_pos / sin_pos)"
            )
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(q.device).cuda_stream
        run_qsa_prepare(
            self._recipes[slot_mapping.dtype],
            q,
            k,
            v,
            None,
            None,
            w_q_norm,
            w_k_norm,
            None if table_arm else cos.view(t, g.rope_dim),
            None if table_arm else sin.view(t, g.rope_dim),
            k_cache,
            v_cache,
            slot_mapping,
            index_k_raw_src=index_src,
            index_k_raw_pool=index_cache,
            stream=stream,
            **(
                dict(
                    decode_kw,
                    seq_q=self.seq_len,
                    index_eps=float(g.qsa.index_norm_eps),
                    index_norm_weight_offset=0.0,
                    pos0=None,
                    n_commit=None,
                )
                if self.decode_arms
                else {}
            ),
        )


# ---------------------------------------------------------------------------
# 5b. The in-block indexer (block-sparse attention, QsaSpec.index_source="indexer")
# ---------------------------------------------------------------------------


def _cu_check(err, what: str) -> None:
    """Raise on a CUDA driver error from one of the block's own stream-ordered driver calls (a memset or a device copy)."""
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


def _index_state_ring_extra_rows(block_size: int) -> int:
    """The rows the raw-key ring needs BEYOND the step's rows: ``cap >= S_q + block_size - 1`` -- a completing block pools the ``block_size - 1``
    positions below the step while the step writes its own rows (``kernels.qsa_compress.min_ring_rows``, the kernel's host rule).  Spelled
    DSL-free here so :meth:`GatedAttentionBlockFwd.serving_envelope` reads it on any host (the kernel module imports the DSL at import);
    :meth:`_IndexerDecode.check_support` holds the two spellings equal at plan time (a drift raises there)."""
    return int(block_size) - 1


def _check_qsa_indexer_geometry(
    geom: GatedAttentionBlockGeometry,
    dtype: torch.dtype,
    *,
    thd: bool = False,
    paged: bool = False,
    s_q: Optional[int] = None,
    page_size: Optional[int] = None,
    kv_cache_dtype=None,
) -> None:
    """``QsaSpec(index_source="indexer")`` against the indexer scorer's contract -- device-free, typed
    ``NotImplementedError`` naming the feature (the texts of :func:`_qsa_decode_decline_texts`, the envelope's rows): the activation
    dtype the scorer packs, its head dim, the packed form; then, on the PREFILL form, the prefill scorer's head groups (read off the
    DSA tree behind the DSL floor), and on the PAGED form (``paged``, appended: the decode step over the pools) the decode scorer's
    own rows instead -- its query-row envelope at ``s_q`` (``S_q x next_pow2(index_heads) <= 64``), the page's power-of-two row count
    (``page_size``), the e4m3 pool (``kv_cache_dtype``) and the shared list form -- all DSL-free (the e4m3 indexer CACHE is served there
    since 2026-10-21: the compress arms' e4m3 store, the paged scorer's e4m3 read arm); (the
    scorer's host facts live in ``cudnn.QSA.selection_paged``, never the kernel module).  Called at declaration
    (``GatedAttentionBlockFwd._check_qsa_declaration``) and again by the stage's ``check_support`` so the stage stays honest on its own."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import INDEXER_SCORER_HEAD_DIM, indexer_scorer_head_groups

    q = geom.qsa
    texts = _qsa_decode_decline_texts(geom, dtype)
    if dtype != torch.bfloat16:
        raise NotImplementedError(texts["indexer_f16"])
    if q.index_head_dim != INDEXER_SCORER_HEAD_DIM:
        raise NotImplementedError(texts["indexer_head_dim"])
    if thd:
        raise NotImplementedError(texts["indexer_thd"])
    if paged:
        # The DECODE step over the pools: the paged scorer's rows (DSL-free: cudnn.QSA.selection_paged owns the kernel's host facts).  The prefill
        # scorer's head-group rule below is NOT applied here -- it reads the DSA tree behind a floor gate, and a sub-floor DSL would decline
        # the decode form for the prefill's reason.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection_paged import MAX_QUERY_ROWS, padded_heads, validate_params

        if s_q is not None and int(s_q) * padded_heads(q.index_heads) > MAX_QUERY_ROWS:
            raise NotImplementedError(texts["indexer_query_rows"])
        if isinstance(page_size, int) and not isinstance(page_size, bool) and page_size > 0:
            try:
                validate_params(h_i=q.index_heads, s_q=1, page_size=int(page_size), units_per_warp=1)
            except ValueError as exc:
                raise NotImplementedError(texts["indexer_page_size"]) from exc
        if kv_cache_dtype == KV_CACHE_E4M3:
            raise NotImplementedError(texts["indexer_e4m3_pool"])
        if q.index_list_form == QSA_INDEX_LIST_FORM_SHARED:
            raise NotImplementedError(texts["indexer_shared_list"])
        return
    # The scorer's head groups are read off the DSA tree, whose package imports its kernel modules: the CuTe DSL floor first
    # (Rule 7), so a too-old DSL reads as a version decline here and never as an error from inside the DSL.
    too_old = cutedsl_requirement_error("the in-block indexer (the indexer scorer)")
    if too_old is not None:
        raise NotImplementedError(too_old)
    groups = indexer_scorer_head_groups()
    if q.index_heads not in groups:
        raise NotImplementedError(
            f"QsaSpec(index_source='indexer', index_heads={q.index_heads}): the indexer scorer packs head groups {groups} on its MMA tile; "
            "another head count has no scorer"
        )


class _Indexer(_Stage):
    """(4i) The in-block indexer of a block-sparse (``QsaSpec``) block declared with ``index_source="indexer"``: the
    per-query selection stage (4) consumes, derived from the slab's fifth band instead of handed in by the caller.

    **Five launches at ``T <= 4096`` -- two compress, the scorer GEMM, the radix top-k, the row sort -- one more sort window
    per further 4096 rows (:meth:`launches_per_execute`); no kernel of its own.**  Two launches more under the fp8 indexer cache
    (the paragraph below).

    1. The indexer QUERIES -- the band's ``index_heads`` x ``index_head_dim`` query columns of every token, RMSNormed
       (``w_iq_norm``, ``index_norm_eps``) and partially rotated at the token's position -- through the block-compress
       kernel (``kernels/qsa_compress.py``) at ``pool = 1`` over a ``[T, index_heads, index_head_dim]`` view of the slab
       (batch = token, block = head; the tables broadcast over the heads with a zero stride): bitwise the block's own
       norm + RoPE chain on those rows (one fp32 pass, one rounding), written compact into the ``ix_q`` slot.
    2. The COMPRESSED KEYS -- the band's raw key head mean-pooled per complete ``block_size``-token block, RMSNormed
       (``w_ik_norm``) and rotated at the block START position ``block_size * j`` -- the same kernel at
       ``pool = block_size`` over the slab's raw-key column slice (token stride ``n_qkvg``, no copy), into the
       ``ix_kbar`` slot ``[B, floor(S / bs), D_i]`` and, when the caller passes ``index_k_compressed``, into that buffer
       as well (its rows at or past a sequence's complete blocks are never written: a cache sized for a longer context
       keeps them).
    3. The SELECTION -- :func:`qsa_select_forward` (the indexer scorer ``sum_h relu(q_h . kbar_j) / sqrt(D_i)`` over the blocks
       below ``floor((pos + 1) / bs)`` + the fused radix top-k, ties to the smaller id) into ``ix_ids`` / ``ix_scores``:
       ``[T, top_k]`` int32, the ``min(top_k, floor((pos + 1) / bs))`` selected ids then ``-1`` -- exactly the count rule
       the sparse core derives, so stage (4) needs no ``block_lens``.  The top-k's SET is reproducible but its slot order
       is not (the radix pass assigns slots with atomics), and the core accumulates in list order, so
    4. the CANONICAL ORDER -- each row sorted descending (the selected ids descending, the ``-1`` padding last; a
       row-windowed ``torch.sort`` into preallocated workspace slots, no temporaries) into ``ix_ids_sorted`` or the
       caller's ``block_ids_out``: two executes of the block are bitwise, and the list a caller gets back is canonical.

    **The RoPE tables are the attention's own** ``cos`` / ``sin`` ``[B, S, rope_dim]``: the key compress reads row
    ``bs * j`` for block ``j`` (every block start is a position of the prefill), the query norm reads row ``pos``.  No
    table of its own, no position ids; the decode step, whose per-token tables do not reach a block start, carries its
    own table rule in the compress kernel's step form.

    **The identity below the bound.**  At ``S <= QsaSpec.identity_bound`` every query's complete blocks fit the list, so
    the selection is the identity: the block launches NO scorer and NO query norm, hands stage (4) a plan-time constant
    list (``ids[t, j] = j`` below the row's count, ``-1`` past it) and reserves no indexer slot; the key compress runs
    only into a caller's ``index_k_compressed`` (a serving cache wants the prompt's compressed keys whatever its length).

    **Padding rows under ``seq_lens``.**  A query at or past its sequence's KV length scores blocks past the sequence's
    complete ones; the ``ix_kbar`` slot is zero-filled per execute there (one memset, only under ``seq_lens_present``),
    so those blocks score 0 -- a positive visible block always wins, a zero-score tie goes to the smaller (visible) id
    -- and the sparse core masks every key past the length anyway.  The caller's ``index_k_compressed`` is NOT
    zero-filled: its rows past a sequence's blocks are the caller's.

    **The fp8 indexer cache (``QsaSpec.index_cache_dtype=torch.float8_e4m3fn``).**  The two compress launches keep writing bf16
    (the one-rounding chain above); the FP8 pipeline's per-tensor cast (``kernels/quantize.py``, ``sat_e4m3(row x 1.0)`` -- the
    plan-constant scale word 1.0 makes the code exactly the saturating round-to-nearest-even e4m3 of the bf16 value, never a NaN
    from an out-of-range value) then writes the e4m3 queries into ``ix_q8`` and the e4m3 compressed keys into ``ix_kbar8``, and
    :func:`qsa_select_forward` scores the e4m3 pair through the DSA MXFP8 scorer with unit scale blobs
    (``sum_h relu(e4m3(q_h) . e4m3(kbar_j)) / sqrt(D_i)``).  The caller's ``index_k_compressed`` is an e4m3 buffer written by one
    more cast from the bf16 slot over the ``floor(S / bs)`` rows of every sequence -- under ``seq_lens`` the rows
    ``[floor(S_b / bs), floor(S / bs))`` of a padded sequence receive ZEROS (the slot is zero-filled per execute, so the cast
    writes a defined value, never stale bytes; the bf16 form leaves them untouched), rows at or past ``floor(S / bs)`` stay the
    caller's in both forms.  Below the identity bound only the cache runs: the bf16 compress into a staging ``ix_kbar`` slot, the
    cast into the cache (two launches where the bf16 form runs one).  The compress kernel's e4m3 store arm would fold the casts
    away (a lever, not built).  No new kernel: the cast kernel has no SMEM and no mbarrier (its module docstring).

    **What it costs.**  ``ix_q`` ``T x H_i x D_i x 2`` B, ``ix_kbar`` ``B x S / bs x D_i x 2`` B, the list
    ``T x top_k x 8`` B, and the scorer's compact-logits scratch ``~ B x S^2 / (2 bs)`` fp32 (512 MiB per 32K-token
    sequence) -- the price of never materializing the dense ``[T, n_blocks]`` score tensor in one launch pair; a
    row-windowed select bounds it and is the lever for longer prompts.  ``compile`` warms the scorer GEMM and the top-k
    at the real ``(B, S, n_blocks)`` (their compile keys carry the shape) on temporaries of the slot sizes dropped after
    one run on the ambient stream, so ``execute`` compiles nothing and allocates nothing of its own on either cache dtype
    (Rules 1 / 4: the e4m3 scorer's padded scale prefix is a plan constant, never rebuilt per call); the ONE allocator
    transient on the path is torch's own -- ``torch.sort`` builds its 4 KB index template (``arange(top_k)`` int64) per
    4096-row window even into preallocated outputs, served from the caching allocator's pool and, under a capture, the
    graph's; a block-local in-kernel row sort would remove it with the sort's int64 index scratch and four of its five
    launches (a follow-up lever).  The plan constants (the query
    prefix, the scratch offsets, that scale prefix, the counts, the identity list, the casts' scale word) are built on the
    host at ``compile``, which only ALLOCATES their device tensors; their values are UPLOADED on a launch stream -- the
    warm-up's, and the FIRST execute's own (one async copy each from a pinned host copy, behind the event ``compile``
    records; :meth:`_upload_const`) -- so no execute reads a constant written on another stream (Rule 5; a capture of a
    block's first execute records those uploads with their values: warm the block up before capturing, the tree's rule).

    Barrier / SMEM tables: the compress kernel has no mbarrier and no SMEM (its module docstring: both tables empty by
    construction); the scorer and the top-k are the DSA kernels with their own suites.  Declines, typed: at declaration
    (:func:`_check_qsa_indexer_geometry`) f16 activations, an ``index_head_dim`` other than the scorer's, an
    ``index_heads`` outside its packed groups, THD, and the CuTe DSL floor as well (ahead of the head-group read, which
    imports the scorer's package); at ``check_support`` the floor again and the ``sm_107a`` target, both before any kernel
    import, and a device outside the scorer's family (cc 10.x).
    """

    name = "qsa_indexer"

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, device, seq_lens_present: bool) -> None:
        q = geometry.qsa
        if q is None or q.index_source != "indexer":
            raise ValueError(f"{self.name}: the geometry declares no QsaSpec(index_source='indexer')")
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = device
        self.seq_lens_present = bool(seq_lens_present)
        self.n_blocks = self.seq_len // q.block_size  # compressed-key rows per sequence the scorer reads
        self.selects = self.seq_len > q.identity_bound  # False: the identity -- no scorer launch, no workspace slot
        self.cand_floats = _qsa_cand_floats(self.batch, self.seq_len, q.block_size) if self.selects else 0
        self._q_recipe = self._k_recipe = None
        self._const = None  # the plan constants' DEVICE tensors (allocated at compile, written on a launch stream by _upload_const): counts [T];
        # identity_ids [T, top_k] | cu_seqlens_q [B+1], cand_batch_offsets [B+1] (+ cu_seqlens_q_scale_padded [B+1] under fp8); one [1] under fp8
        self._host = None  # their values: pinned host tensors built at compile, the source of every upload
        self._const_ready = None  # the event compile() records behind the allocations and the warm-up; the first execute's upload waits on it
        self._const_on_launch_stream = False  # True once the first execute after compile() uploaded the constants on ITS launch stream
        self.launches = 0  # launches issued so far (a test pin of the short-circuit; the block never reads it)
        # The indexer cache's dtype (QsaSpec.index_cache_dtype; None = the activation dtype).  e4m3 = the fp8 indexer cache (the class
        # docstring): the scorer reads e4m3 casts of the bf16 slots and the caller's cache is e4m3.
        self.cache_dtype = q.index_cache_dtype if q.index_cache_dtype is not None else dtype
        self.fp8 = self.cache_dtype == INDEX_CACHE_E4M3
        self._q8 = self._k8 = None  # the two cast recipes (fp8 only; _k8 None when S < block_size: no complete block)
        self._one = None  # the casts' scale word, the plan constant 1.0 (fp8 only): self._const["one"] once compiled

    def launches_per_execute(self, cache: bool) -> int:
        """Launches one ``execute`` issues: the cache compress iff a caller passes ``index_k_compressed``; past the identity
        bound also the key compress, the query norm, the scorer GEMM, the radix top-k and the row sort's windows."""
        if not self.selects:
            if not cache:
                return 0
            return (2 if self.n_blocks > 0 else 1) if self.fp8 else 1  # fp8: the staging compress + the cache cast
        return (1 if cache else 0) + (6 if self.fp8 else 4) + -(-(self.batch * self.seq_len) // _QSA_SORT_ROWS)

    def check_support(self) -> None:
        """The scorer's contract (device-free, re-run so the stage is honest on its own), the CuTe DSL floor / target
        BEFORE any kernel import (Rule 7), the device family the scorer runs on, the compress kernel's row geometry."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

        g, q = self.geom, self.geom.qsa
        _check_qsa_indexer_geometry(g, self.dtype)
        too_old = cutedsl_requirement_error("the in-block indexer (block compress + the indexer scorer)")
        if too_old is not None:
            raise NotImplementedError(too_old)
        cc = tuple(torch.cuda.get_device_capability(self.device))
        no_target = cutedsl_arch_requirement_error(cc)
        if no_target is not None:
            raise NotImplementedError(no_target)
        # cc 10.x = the family the scorer has run on (cc 10.0 / 10.3 / 10.7); the sparse record's 11.9 upper bound is the attention
        # CORE's, not the scorer's -- a cc 11.x part is a typed decline here until the scorer is run on one (widen from that run).
        if cc[0] != 10:
            raise NotImplementedError(f"{self.name}: the indexer scorer is an SM100-family (cc 10.x) tcgen05 kernel; found cc {cc[0]}.{cc[1]}")
        from .kernels.qsa_compress import DEFAULT_THREADS_PER_CTA, validate_compress_shape

        for pool in (1, q.block_size):
            validate_compress_shape(q.index_head_dim, g.rope_dim, pool, DEFAULT_THREADS_PER_CTA)
        if self.fp8:
            from .kernels.quantize import validate_shape as validate_cast_shape

            validate_cast_shape(q.index_head_dim, _ELEMENTWISE_THREADS)  # the per-tensor cast's row geometry at the indexer head dim

    def compile(self) -> None:
        """The two compress recipes, the plan constants (built on the host; their device tensors allocated here, their values uploaded on a
        launch stream by :meth:`_upload_const`), and -- when the scorer runs -- its warm-up at the real shape on the ambient stream."""
        from .kernels.qsa_compress import compile_qsa_compress

        g, q = self.geom, self.geom.qsa
        b, s, t, top_k, dev = self.batch, self.seq_len, self.batch * self.seq_len, q.top_k, self.device
        # The key compress: block_size rows per compressed key; the per-batch lengths traced iff declared (the artifact folds the load).
        self._k_recipe = compile_qsa_compress(
            dtype=self.dtype, d=q.index_head_dim, rope_dim=g.rope_dim, pool=q.block_size, has_seq_lens=self.seq_lens_present, table_dtype=self.dtype
        )
        if self.fp8:
            from .kernels.quantize import compile_quantize

            # The casts of the fp8 indexer cache: the queries as [T, H_i, D_i] (past the bound) and the compressed keys as [B, n_blocks, D_i]
            # (one recipe serves the workspace slot and the caller's cache, whose token stride is the cache's own; the cast kernel reads
            # both in place); the scale word is the plan constant 1.0 (host["one"] below) -- x * 1.0 is exact, so the code is sat_e4m3(bf16 value).
            if self.n_blocks > 0:
                self._k8 = compile_quantize(dtype_in=self.dtype, h=self.n_blocks, d=q.index_head_dim, threads_per_cta=_ELEMENTWISE_THREADS)
            if self.selects:
                self._q8 = compile_quantize(dtype_in=self.dtype, h=q.index_heads, d=q.index_head_dim, threads_per_cta=_ELEMENTWISE_THREADS)
        # The plan constants, built on the HOST: nothing of the plan is FILLED on the ambient stream here (frost-engine-contract s10 / Rule 5 --
        # a torch fill at compile() sits on whatever stream is ambient then, unordered against a first execute on another stream).  Their
        # device tensors are only allocated (_set_const); every value lands through _upload_const on a launch stream: the warm-up's below,
        # and the first execute's own.
        pos = torch.arange(s, dtype=torch.int64).repeat(b)  # [T]: row s of every batch entry sits at position s
        counts = torch.minimum(torch.div(pos + 1, q.block_size, rounding_mode="floor"), torch.full_like(pos, top_k))
        host = dict(counts=counts.to(torch.int32).contiguous())
        if self.fp8:
            host["one"] = torch.ones(1, dtype=torch.float32)
        if not self.selects:
            j = torch.arange(top_k, dtype=torch.int64)
            ids = torch.where(j[None, :] < counts[:, None], j[None, :].expand(t, top_k), torch.full((t, top_k), -1, dtype=torch.int64))
            host["identity_ids"] = ids.to(torch.int32).contiguous()
            self._set_const(host)
            self._const_ready.record(torch.cuda.current_stream(dev))  # behind the allocations (a deterministic-mode fill included)
            return
        # The query norm + RoPE: pool = 1, batch = token, block = head; no per-batch lengths (every token's queries are normed).
        self._q_recipe = compile_qsa_compress(dtype=self.dtype, d=q.index_head_dim, rope_dim=g.rope_dim, pool=1, has_seq_lens=False, table_dtype=self.dtype)
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward import compress_topk_cand_buffer_size_thd

        cu_q = (torch.arange(b + 1, dtype=torch.int64) * s).to(torch.int32)
        cu_k = (torch.arange(b + 1, dtype=torch.int64) * self.n_blocks).to(torch.int32)
        # The scorer's per-sequence scratch offsets by the DSA sizing helper (device inputs; the two plan-time host reads of this compile).
        offsets, total = compress_topk_cand_buffer_size_thd(cu_q.to(dev), cu_k.to(dev), q.block_size, None)
        if total != self.cand_floats:
            raise RuntimeError(
                f"{self.name}: the scorer's compact-logits scratch is {total} floats by its sizing helper but {self.cand_floats} by the block's "
                f"closed form (B={b}, S={s}, block_size={q.block_size}) -- the two derivations drifted; the workspace would be mis-sized"
            )
        host.update(cu_seqlens_q=cu_q, cand_batch_offsets=offsets.cpu())
        if self.fp8:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import _fp8_q_scale_prefix

            # The e4m3 scorer's padded Q scale prefix (every sequence rounded up to the scale span of its heads): a plan constant, so
            # qsa_select_forward builds nothing per execute (Rule 1).
            host["cu_seqlens_q_scale_padded"] = _fp8_q_scale_prefix(cu_q, q.index_heads)
        self._set_const(host)
        # Warm the scorer GEMM and the radix top-k at the REAL (B, S, n_blocks): their compile keys carry the shape, so a smaller
        # warm-up would leave a compile on the execute path.  On the ambient stream, the constants uploaded there first; temporaries of
        # the slot sizes, allocated and consumed on that one stream and dropped after the run -- no host sync (Rule 8).
        ambient = torch.cuda.current_stream(dev).cuda_stream
        self._upload_const(ambient, wait=False)
        qi = torch.zeros(t, q.index_heads, q.index_head_dim, dtype=self.cache_dtype, device=dev)  # e4m3 under the fp8 indexer cache
        kbar = torch.zeros(b, self.n_blocks, q.index_head_dim, dtype=self.cache_dtype, device=dev)
        ids = torch.empty(t, top_k, dtype=torch.int32, device=dev)
        scores = torch.empty(t, top_k, dtype=torch.float32, device=dev)
        cand = torch.empty(self.cand_floats, dtype=torch.float32, device=dev)
        self._select(qi, kbar, ids, scores, cand, stream=ambient)
        del qi, kbar, ids, scores, cand
        self._const_ready.record(torch.cuda.current_stream(dev))  # behind the allocations, the warm-up's upload and its reads of the constants

    def _set_const(self, host: dict) -> None:
        """The plan constants: ``host``'s values as PINNED host tensors (the source of every upload) and their device tensors, ALLOCATED here
        and never filled here (:meth:`_upload_const` writes them on a launch stream); the readiness event, recorded by the caller once
        everything the ambient stream owes them is enqueued; the first-execute flag reset."""
        self._host = {name: val.contiguous().pin_memory() for name, val in host.items()}
        self._const = {name: torch.empty(val.shape, dtype=val.dtype, device=self.device) for name, val in host.items()}
        self._one = self._const.get("one")
        self._const_ready = torch.cuda.Event()
        self._const_on_launch_stream = False

    def _upload_const(self, stream: int, *, wait: bool) -> None:
        """Write every plan constant's VALUE into its device tensor on ``stream`` (a raw CUstream handle, the launch stream: the consumers that
        follow on it are ordered behind the write by construction, Rule 5) -- one async H2D copy from the pinned host copy each; nothing
        allocated, nothing read back.  ``wait=True`` (the first execute) first waits on the event :meth:`compile` recorded behind its
        allocations and its warm-up, so stream work the allocation itself enqueued (``torch.use_deterministic_algorithms(True)`` fills
        ``torch.empty`` with NaN) and the warm-up's reads land before any value written on another stream.  Inside a stream capture the
        wait and the allocator's stream record are skipped (an external event cannot be waited on there; a ``torch.cuda.graph`` capture
        synchronised the device at its entry) -- the documented case: warm the block up before capturing."""
        ts = as_torch_stream(stream, self.device)
        with torch.cuda.stream(ts):
            capturing = torch.cuda.is_current_stream_capturing()
            if wait and self._const_ready is not None and not capturing:
                ts.wait_event(self._const_ready)
            for name, dst in self._const.items():
                dst.copy_(self._host[name], non_blocking=True)
                if not capturing:
                    dst.record_stream(ts)  # the allocator reuses the block only once this stream's work on it is done (R1)

    def _select(self, qi: torch.Tensor, kbar: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor, cand: torch.Tensor, *, stream: int) -> torch.Tensor:
        """The scorer + top-k on the block's stream, every buffer preallocated (the CUDA-graph form of :func:`qsa_select_forward`)."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import qsa_select_forward

        q = self.geom.qsa
        return qsa_select_forward(
            qi,
            kbar,
            self._const["cu_seqlens_q"],
            top_k=q.top_k,
            block_size=q.block_size,
            max_seqlen_q=self.seq_len,
            deterministic=True,
            block_ids_out=ids,
            scores_out=scores,
            cand_buffer=cand,
            cand_batch_offsets=self._const["cand_batch_offsets"],
            stream=cuda.CUstream(int(stream)),
            cu_seqlens_q_scale_padded=self._const.get("cu_seqlens_q_scale_padded"),  # the e4m3 arm's plan constant (None on the bf16 arm)
        )["block_ids"]

    def views(self, workspace: torch.Tensor, lay: "_Intermediates") -> Optional[dict]:
        """The five indexer slots as typed views of the workspace; ``None`` below the identity bound (nothing reserved)."""
        if not self.selects:
            return None
        q, b, t = self.geom.qsa, self.batch, self.batch * self.seq_len
        v = dict(
            qi=_view(workspace, lay.ix_q, (t, q.index_heads, q.index_head_dim), self.dtype),
            kbar=_view(workspace, lay.ix_kbar, (b, self.n_blocks, q.index_head_dim), self.dtype),
            ids=_view(workspace, lay.ix_ids, (t, q.top_k), torch.int32),
            scores=_view(workspace, lay.ix_scores, (t, q.top_k), torch.float32),
            cand=_view(workspace, lay.ix_cand, (self.cand_floats,), torch.float32),
            ids_sorted=_view(workspace, lay.ix_ids_sorted, (t, q.top_k), torch.int32),
            sort_idx=_view(workspace, lay.ix_sort_idx, (min(t, _QSA_SORT_ROWS), q.top_k), torch.int64),
        )
        if self.fp8:
            v.update(
                qi8=_view(workspace, lay.ix_q8, (t, q.index_heads, q.index_head_dim), INDEX_CACHE_E4M3),
                kbar8=_view(workspace, lay.ix_kbar8, (b, self.n_blocks, q.index_head_dim), INDEX_CACHE_E4M3),
            )
        return v

    def _staging(self, workspace: torch.Tensor, lay: "_Intermediates") -> torch.Tensor:
        """The fp8 indexer cache BELOW the identity bound: the bf16 ``ix_kbar`` staging slot the cache cast reads (the only slot reserved)."""
        q = self.geom.qsa
        return _view(workspace, lay.ix_kbar, (self.batch, self.n_blocks, q.index_head_dim), self.dtype)

    def _cast_cache(self, kbar16: torch.Tensor, cache: torch.Tensor, stream: int) -> None:
        """The caller's e4m3 ``index_k_compressed``: ``sat_e4m3`` of the bf16 compressed keys over the ``n_blocks`` rows of every sequence,
        the destination the cache's own ``[B, n_blocks, D_i]`` prefix at the cache's token stride (read and written in place)."""
        from .kernels.quantize import run_quantize

        run_quantize(self._k8, kbar16, cache[:, : self.n_blocks], self._one, stream=stream)

    def execute(
        self,
        proj: torch.Tensor,  # [T, n_qkvg] the slab stage (1) wrote; the band's columns are read in place
        cos: torch.Tensor,  # [B, S, rope_dim] the attention's tables (block starts and token positions alike)
        sin: torch.Tensor,
        w_iq_norm: torch.Tensor,  # [index_head_dim] the indexer queries' RMSNorm weight (pre-folded (1 + w))
        w_ik_norm: torch.Tensor,  # [index_head_dim] the compressed keys' RMSNorm weight
        *,
        seq_lens: Optional[torch.Tensor],  # [B] int32 iff declared seq_lens_present
        workspace: torch.Tensor,
        lay: "_Intermediates",
        index_k_compressed: Optional[torch.Tensor] = None,  # [B, >= floor(S / bs), index_head_dim]: the caller's compressed-key cache
        block_ids_out: Optional[torch.Tensor] = None,  # [T, top_k] int32 (or [B, S, top_k]): the selection, written in place
        block_lens_out: Optional[torch.Tensor] = None,  # [T] int32 (or [B, S]): the counts
        current_stream,
    ) -> torch.Tensor:
        """Run the indexer and return the ``[T, top_k]`` int32 list stage (4) consumes -- the caller's ``block_ids_out``
        when given, else the workspace slot, else the plan-time identity list (``S <= identity_bound``).  Every buffer was
        form-checked by the block; nothing is allocated, nothing is read back; the first execute after ``compile`` uploads the
        plan constants on its launch stream (:meth:`_upload_const`) ahead of everything that reads them."""
        from .kernels.qsa_compress import run_qsa_compress

        if self._const is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        g, q = self.geom, self.geom.qsa
        b, s, t = self.batch, self.seq_len, self.batch * self.seq_len
        n, d_i, h_i = g.n_qkvg, q.index_head_dim, q.index_heads
        stream = int(current_stream)
        hs = cuda.CUstream(stream)
        if not self._const_on_launch_stream:
            # The FIRST execute writes the plan constants' VALUES on ITS launch stream (compile() only allocated their device tensors; the
            # warm-up's upload was the ambient stream's): the DtoD copies, the casts and the sparse core read them in stream order.  Once per
            # compile(); a stream capture of this first execute records the uploads WITH their values (constants: every replay re-writes the
            # same ones) -- warm up before capturing, the tree's convention.
            self._upload_const(stream, wait=True)
            self._const_on_launch_stream = True
        eps = float(q.index_norm_eps)
        # The raw indexer key: the band's last head, [B, S, D_i] at the slab's token stride (no copy); block j pools rows 4j .. 4j + 3.
        k_raw = torch.as_strided(proj, (b, s, d_i), (s * n, n, 1), proj.storage_offset() + g.index_k_raw_offset)
        if index_k_compressed is not None and not self.fp8:
            # The caller's bf16 cache: rows [0, floor(S_b / bs)) of every sequence; the rest stays the caller's.
            run_qsa_compress(self._k_recipe, k_raw, index_k_compressed, w_ik_norm, cos, sin, seq_lens, eps=eps, norm_weight_offset=0.0, stream=stream)
            self.launches += 1
        if not self.selects:
            if index_k_compressed is not None and self.fp8:
                # The caller's e4m3 cache below the bound: the bf16 compress into the staging slot (zero-filled under seq_lens, so a padded
                # sequence's tail rows cast to ZEROS, never to stale bytes), then the cast into the cache -- two launches.
                kb = self._staging(workspace, lay)
                if self.seq_lens_present:
                    _cu_check(cuda.cuMemsetD8Async(kb.data_ptr(), 0, kb.numel() * _itemsize(self.dtype), hs)[0], f"{self.name}: staging slot memset")
                run_qsa_compress(self._k_recipe, k_raw, kb, w_ik_norm, cos, sin, seq_lens, eps=eps, norm_weight_offset=0.0, stream=stream)
                self.launches += 1
                if self._k8 is not None:
                    self._cast_cache(kb, index_k_compressed, stream)
                    self.launches += 1
            ids = self._const["identity_ids"]
            if block_ids_out is not None:
                dst = block_ids_out.view(t, q.top_k)
                _cu_check(
                    cuda.cuMemcpyDtoDAsync(dst.data_ptr(), ids.data_ptr(), ids.numel() * _itemsize(torch.int32), hs)[0], f"{self.name}: identity list copy"
                )
                ids = dst
        else:
            v = self.views(workspace, lay)
            if self.seq_lens_present:
                # Rows past a padded sequence's complete blocks are never written by the compress: zero them, so a padding row
                # scores 0 there (never stale workspace -- a select over NaN is undefined, over zeros it is the visible set).
                kb = v["kbar"]
                _cu_check(cuda.cuMemsetD8Async(kb.data_ptr(), 0, kb.numel() * _itemsize(self.dtype), hs)[0], f"{self.name}: compressed-key slot memset")
            run_qsa_compress(self._k_recipe, k_raw, v["kbar"], w_ik_norm, cos, sin, seq_lens, eps=eps, norm_weight_offset=0.0, stream=stream)
            # The queries: pool = 1 over [T, H_i, D_i] (batch = token, block = head), the tables broadcast over the heads.
            q_raw = torch.as_strided(proj, (t, h_i, d_i), (n, d_i, 1), proj.storage_offset() + g.qkvg_offsets[ProjBlock.INDEX])
            cos_q = cos.view(t, 1, g.rope_dim).expand(t, h_i, g.rope_dim)
            sin_q = sin.view(t, 1, g.rope_dim).expand(t, h_i, g.rope_dim)
            run_qsa_compress(self._q_recipe, q_raw, v["qi"], w_iq_norm, cos_q, sin_q, None, eps=eps, norm_weight_offset=0.0, stream=stream)
            if self.fp8:
                # The fp8 indexer cache: the e4m3 casts of both slots feed the scorer's MXFP8 arm; the caller's cache is one more cast.
                from .kernels.quantize import run_quantize

                run_quantize(self._q8, v["qi"], v["qi8"], self._one, stream=stream)
                run_quantize(self._k8, v["kbar"], v["kbar8"], self._one, stream=stream)
                self.launches += 2
                if index_k_compressed is not None:
                    self._cast_cache(v["kbar"], index_k_compressed, stream)
                    self.launches += 1
                self._select(v["qi8"], v["kbar8"], v["ids"], v["scores"], v["cand"], stream=stream)
            else:
                self._select(v["qi"], v["kbar"], v["ids"], v["scores"], v["cand"], stream=stream)
            # The canonical order: the top-k's slot order is unspecified (atomics), the core accumulates in list order, so each
            # row is sorted descending (ids descending, -1 last) in row windows of _QSA_SORT_ROWS into preallocated outputs --
            # torch's small-segment sort runs in place on the output (its one transient is the 4 KB int64 index template torch
            # builds per call -- the sort's own, not a workspace matter; the R9 census pins it as the path's only allocation), on
            # the block's LAUNCH stream as a torch stream through as_torch_stream (recipe R1: a default-stream handle maps to
            # torch's default stream, a side stream to its ExternalStream -- never a hand-rolled ExternalStream(handle), which
            # torch before 2.13 turns into an unordered pool stream for the handle 0 every eager caller hands in).
            ids = v["ids_sorted"] if block_ids_out is None else block_ids_out.view(t, q.top_k)
            with torch.cuda.stream(as_torch_stream(stream, proj.device)):
                for r0 in range(0, t, _QSA_SORT_ROWS):
                    r1 = min(t, r0 + _QSA_SORT_ROWS)
                    torch.sort(v["ids"][r0:r1], dim=-1, descending=True, out=(ids[r0:r1], v["sort_idx"][: r1 - r0]))
            self.launches += 4 + -(-t // _QSA_SORT_ROWS)  # the key compress, the query norm, the scorer GEMM, the radix top-k, the sort windows
        if block_lens_out is not None:
            c = self._const["counts"]
            _cu_check(
                cuda.cuMemcpyDtoDAsync(block_lens_out.view(t).data_ptr(), c.data_ptr(), c.numel() * _itemsize(torch.int32), hs)[0], f"{self.name}: counts copy"
            )
        return ids


class _IndexerDecode(_Stage):
    """(4i) The in-block indexer's DECODE step over the serving stack's PAGED side caches: a block-sparse (``QsaSpec``) block
    declared ``index_source="indexer"`` with ``kv_source="paged"`` (a dense ``[B, S_q]`` declaration) derives the step's per-row
    selection itself -- the list stage (4) consumes -- from the band the slab GEMM wrote, the stack's raw-key ring and its paged
    compressed-key cache.  The stage name stays ``qsa_indexer`` (the prefill :class:`_Indexer` is untouched: its constants, sort
    and casts have no decode form).

    **Two launches (``fuse_prepare=True``: the scorer and the top-k; the prepare launch carries the indexer arms) or four
    (``fuse_prepare=False``: the pool-1 query norm, the paged compress step, the scorer, the top-k), no kernel of its own;
    both forms BITWISE.**  Per step, in stream order:

    1. (fused: inside :class:`_QsaPrepare`; unfused: here) the indexer QUERIES -- the band's ``index_heads x index_head_dim`` query
       columns of every token, RMSNormed (``w_iq_norm``, ``index_norm_eps``) and partially rotated at the token's position -- into
       the workspace slot ``ix_q`` (``kernels/qsa_compress.py`` at pool 1 over the slab's query columns; the prepare's query arm is
       the same chain); and the RAW KEY of every committed row into the stack's ring ``index_state_cache [num_ring_blocks, cap, 1,
       D_i]`` at row ``(index_state_block_table[b, 0], pos % cap)`` (or at the stack's own ``index_state_slot_mapping`` row), the
       block a row COMPLETES (``(pos + 1) % block_size == 0``; ``pos = kv_lens[b] - S_q + j`` on device) mean-pooled from the ring's
       rows below the step and this step's band rows, RMSNormed (``w_ik_norm``) and rotated at the block START ``block_size x blk``
       from the position-indexed ``cos_pos`` / ``sin_pos`` into the paged compressed-key cache ``index_kv_cache [num_pages,
       P / block_size, 1, D_i]`` at ``(block_table[b, blk // (P / bs)], blk % (P / bs))`` -- the step form of the compress kernel
       (``frost_qsa_compress_step``, its PAGED arms), every row committed (the serving stacks' regime: a rejected draft's rows are
       rewritten by the next step at the same positions); the per-row slot report lands in ``ix_cslot`` (device data nothing here
       reads).
    2. the SCORER (``kernels/qsa_score_paged.py``, PR 4b): ``logits[t, j] = scale x sum_h relu(qi[t, h] . kbar[b, j])`` for the
       ``n_vis(t) = clamp(floor((pos_t + 1) / bs), 0, n_blocks_max)`` complete blocks row ``t`` can see -- the block its own row
       completed INCLUDED (the prepare / the step ran first on the same stream: the serving stacks' order, prepare then select) --
       read from the paged cache through the MAIN block table once, in page order, into the workspace ``ix_logits [T, n_blocks_max]``
       fp32 (the columns at or past ``n_vis`` are never written and never read) and ``ix_counts [T] = n_vis``; ``block_lens_out``,
       when given, receives ``min(top_k, n_vis)``.
    3. the TOP-K (``kernels/qsa_topk_decode.py``, PR 5; the DSL-free facade ``qsa_topk`` gates its import): per row the exact
       top-``min(top_k, counts[t])`` of the fp32 logits (ties to the smaller id), written in CANONICAL order -- the ids strictly
       descending, ``-1`` past them -- into ``ix_ids [T, top_k]`` or the caller's ``block_ids_out`` directly; ``counts <= top_k`` is
       the identity list.  No sort, no atomic decides a slot: two executes are bitwise and the list a caller reads back is canonical.
       The knob pair ``(cluster_ctas, threads_per_cta)`` is ``default_knobs_for(T, n_blocks_max, cc)`` at plan time (the row-split
       cluster form only where the kernel's own ranking earned it: rows wider than 8192 blocks at up to 32 rows).

    The sparse core then consumes the list PER ROW (``list_per_sequence=False`` at every ``S_q``, ``block_lens=None``: the ``-1``
    padding carries the count, which equals the core's own ``min(top_k, floor((pos + 1) / bs))``), the step's rows at the end of
    each sequence on the bottom-right diagonal.

    **The execute path binds and launches only** (Rules 1 / 3 / 5 / 8): every buffer is the caller's or a workspace view carved at
    plan time (:func:`_qsa_indexer_decode_slots`: ``ix_q``, ``ix_logits``, ``ix_counts``, ``ix_ids``, ``ix_cslot`` -- no compressed-key
    slot, the paged cache is the caller's; no scorer scratch, the logits are the scratch; no sort scratch); nothing is allocated, no
    plan constant exists (nothing to upload on a first execute, so a fresh declaration's FIRST execute captures with no warm-up), no
    host read (positions, counts, ring rows and page ids are device data), no torch op (the launches take the raw launch stream).
    Compile keys are plan-time only (Rule 4): the prepare's arms, the scorer's ``(index_heads, S_q, P, units, has_block_lens x 2, pdl)``
    (both ``block_lens`` variants at plan time, the call dispatches on the optional output), the top-k's ``(cols_cap, top_k, C, threads,
    pdl)``, the step form's ``(paged, ring_slots, slot dtype x 2, pdl)``; the side caches' strides are SYMBOLIC in every kernel, so
    ``execute`` matches their dims / dtype / the 16-byte rule, never the strides.

    PDL: the stage carries the block's resolved ``use_pdl`` (``pdl_names`` = the kernels that take the pair: the step form on the
    unfused arm, the scorer, the top-k) -- a typed decline under ``QsaSpec`` today, so every launch runs plain and the links serialize
    in stream order.  The scorer releases its dependents BEFORE it writes the logits (its docstring's obligation); the top-k waits
    before its first dependent read, the sparse core per role, so the chain stays correct the day the switch is served.

    Degenerate rows: ``kv_lens[b] == 0`` (a padding sequence on the null ring block / page 0) and ``kv_lens[b] < S_q`` (rows at a
    negative position) write no ring row, no block, score nothing (``counts = 0``, every list slot ``-1``) and attend nothing (``out =
    0``, ``LSE = -inf`` exactly: the core's dead row); a ``-1`` table entry under a visible block reads as a zero key; a completing block
    past the table's width or a row at or past the RoPE table reports ``-2`` in ``ix_cslot`` and writes nothing (the kernels' capacity
    rules; the host refuses a table shorter than ``max_pages x P`` positions by form).  Barrier / SMEM tables: the prepare and the
    compress kernels have neither (their module docstrings); the scorer's and the top-k's are in their modules.

    **The e4m3 indexer cache (``QsaSpec(index_cache_dtype=torch.float8_e4m3fn)``; served 2026-10-21).**  ``self.cache_dtype`` (the
    declared cache dtype; ``self.fp8``) threads through every kernel of the step: the prepare's query + compress arms (fused) or the pool-1
    compress + the step form (unfused) STORE e4m3 directly -- the e4m3 cast of the bf16 row the bf16 form stores (the compress kernels' e4m3
    store arm; no cast kernel, the launch census unchanged), so ``ix_q`` (carved at 1 B per element) and ``index_kv_cache`` are plain e4m3 at
    unit scale (the serving stacks' fp8 indexer-cache recipe and the prefill form's convention: a cache primed from the prefill form's e4m3
    rows and appended by this step is ONE chain, bitwise); the paged scorer runs its e4m3 READ arm (``mma.sync.m16n8k32 .e4m3``, 128-B rows,
    half the SMEM ring; every product exact in fp32).  A different selection function (3 mantissa bits), chosen at declaration; bf16 stays
    the default and the reference; the ring stays bf16 (a bit copy, never quantized).

    Typed declines: at declaration (:func:`_check_qsa_indexer_geometry`, ``paged=True``) f16 activations, another head dim, the
    query-row envelope (``S_q x next_pow2(index_heads) <= 64``), a page off the powers of two, the e4m3 pool, the shared list form --
    each the envelope's row; at ``check_support`` the CuTe DSL floor and the device's target BEFORE any kernel module is imported (Rule
    7: the facades ``cudnn.QSA.selection_paged`` / ``qsa_topk`` gate first), a device outside the cc 10.x family, a top-k geometry past the
    kernel's reach, the shared-memory opt-in limit.
    """

    name = "qsa_indexer"

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device,
        page_size: int,
        n_blocks_max: int,
        fused: bool,
        ring_slots: bool = False,
        rope_table: bool = False,
        use_pdl: bool = False,
    ) -> None:
        q = geometry.qsa
        if q is None or q.index_source != "indexer":
            raise ValueError(f"{self.name}: the geometry declares no QsaSpec(index_source='indexer')")
        if int(page_size) <= 0 or int(page_size) % q.block_size:
            raise ValueError(f"{self.name}: page_size must be a positive multiple of block_size={q.block_size}, got {page_size}")
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.device = device
        # the indexer cache dtype (QsaSpec.index_cache_dtype; None = the activation dtype): e4m3 = the e4m3 indexer cache (the class docstring) --
        # the compress arms store it, the scorer reads it, ix_q is carved at it
        self.cache_dtype = q.index_cache_dtype if q.index_cache_dtype is not None else dtype
        self.fp8 = self.cache_dtype == INDEX_CACHE_E4M3
        self.page_size = int(page_size)
        self.rows_per_page = self.page_size // q.block_size  # compressed rows per attention page
        self.n_blocks_max = int(n_blocks_max)  # the logits width = the main table's reach in compressed rows (max_pages x rows_per_page)
        self.fused = bool(fused)  # the prepare launch carries the query + compress arms; else this stage launches the two
        self.ring_slots = bool(ring_slots)  # the ring WRITE rows from the stack's own slot mapping (a compile key of the step form)
        self.rope_table = bool(rope_table)  # the fused launch rotates from the position tables (no per-token cos / sin); unfused: declined by the block
        self.use_pdl = bool(use_pdl)
        self.topk_knobs: Optional[Tuple[int, int]] = None  # (cluster_ctas, threads_per_cta), decided at check_support from the device
        self.topk_cols_cap: Optional[int] = None
        self._score = None  # {has_block_lens: the scorer recipe}
        self._topk = None  # the top-k recipe
        self._run_topk = None  # the kernel module's runner (imported behind the facade's gate)
        self._q_recipe = None  # unfused: the pool-1 query norm + RoPE
        self._step = None  # unfused: {slot dtype: the paged step recipe}
        self.launches = 0  # launches issued so far (a test pin; the block never reads it)

    @property
    def pdl_names(self) -> Tuple[str, ...]:
        """The kernels of this stage that carry the ``use_pdl`` pair (what ``GatedAttentionBlockFwd.pdl_stages`` lists when the switch is on)."""
        return (() if self.fused else ("qsa_compress_step",)) + ("qsa_score", "qsa_topk")

    def launches_per_execute(self) -> int:
        """Launches one ``execute`` issues: the scorer and the top-k, plus the query norm and the paged step on the unfused arm."""
        return 2 if self.fused else 4

    def _dev_index(self) -> int:
        dev = torch.device(self.device)
        return dev.index if dev.index is not None else torch.cuda.current_device()

    def check_support(self) -> None:
        """The geometry's rows again (honest on its own), then -- Rule 7, BEFORE any kernel module is imported -- the CuTe DSL floor
        and the device's target, the device family, the top-k facade's gate, the scorer's own gate; then the top-k's geometry against
        the kernel's reach and the shared-memory limit, the ring rule's two spellings, the compress kernels' row geometry."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import shared_memory_per_block_optin

        g, q = self.geom, self.geom.qsa
        _check_qsa_indexer_geometry(g, self.dtype, paged=True, s_q=self.seq_len, page_size=self.page_size)
        too_old = cutedsl_requirement_error("the in-block indexer's decode step (the paged scorer + the decode top-k)")
        if too_old is not None:
            raise NotImplementedError(too_old)
        cc = tuple(torch.cuda.get_device_capability(self.device))
        no_target = cutedsl_arch_requirement_error(cc)
        if no_target is not None:
            raise NotImplementedError(no_target)
        # cc 10.x = the family the paged scorer and the decode top-k have run on (the kernels themselves run from sm_80 / sm_90; the
        # block serves Rubin, and its sparse core's record is the attention bound) -- another family is a typed decline until run there.
        if cc[0] != 10:
            raise NotImplementedError(f"{self.name}: the decode indexer's kernels are validated on the SM100 family (cc 10.x); found cc {cc[0]}.{cc[1]}")
        # the DSL-free facades gate the floor and import the kernel modules only now
        from .qsa_topk import qsa_topk_decode_kernel_module

        topk = qsa_topk_decode_kernel_module()
        from .kernels.qsa_score_paged import dsl_decline_reason

        dev = self._dev_index()
        why = dsl_decline_reason(dev)
        if why is not None:
            raise NotImplementedError(why)
        t = self.batch * self.seq_len
        if t > topk.MAX_ROWS:
            raise NotImplementedError(f"{self.name}: the decode top-k launches one row block per step of at most {topk.MAX_ROWS} rows; got T = {t}")
        self.topk_knobs = tuple(int(x) for x in topk.default_knobs_for(t, self.n_blocks_max, cc))
        c, threads = self.topk_knobs
        self.topk_cols_cap = int(topk.cols_cap_for(self.n_blocks_max, c, threads))
        try:
            topk.validate_topk_geometry(self.topk_cols_cap, q.top_k, c, threads)
        except ValueError as exc:
            raise NotImplementedError(
                f"{self.name}: the decode top-k cannot address a block table of {self.n_blocks_max} compressed rows per sequence ({exc})"
            ) from exc
        smem = int(topk.smem_bytes_for(self.topk_cols_cap, c, threads, q.top_k))
        limit = int(shared_memory_per_block_optin(dev))
        if smem > limit:
            raise NotImplementedError(
                f"{self.name}: the decode top-k at {self.n_blocks_max} compressed rows per sequence needs {smem} B of shared memory per CTA; "
                f"the device allows {limit} B"
            )
        from .kernels.qsa_compress import DEFAULT_THREADS_PER_CTA, min_ring_rows, validate_compress_shape

        # the envelope's DSL-free spelling of the ring rule (index_state_rows_extra) against the kernel's host rule: a drift raises here
        if min_ring_rows(self.seq_len, q.block_size) != self.seq_len + _index_state_ring_extra_rows(q.block_size):
            raise RuntimeError(
                f"{self.name}: the ring rule drifted -- the kernel needs {min_ring_rows(self.seq_len, q.block_size)} rows at S_q={self.seq_len}, the block "
                f"spells S_q + {_index_state_ring_extra_rows(q.block_size)}"
            )
        for pool in ((1, q.block_size) if not self.fused else (q.block_size,)):
            validate_compress_shape(q.index_head_dim, g.rope_dim, pool, DEFAULT_THREADS_PER_CTA)

    def compile(self) -> None:
        """Both scorer variants (``block_lens_out`` is optional per call: Rule 4), the top-k at the plan-time knob pair, and -- on the
        unfused arm -- the pool-1 query norm and the paged step form (both slot dtypes under the ring-slots arm).  No warm-up: nothing
        of this stage holds a plan constant."""
        from .kernels.qsa_score_paged import compile_qsa_score_paged
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection_paged import default_units_per_warp
        from .qsa_topk import qsa_topk_decode_kernel_module

        if self.topk_knobs is None:
            raise RuntimeError(f"{self.name}: call check_support() before compile()")
        g, q = self.geom, self.geom.qsa
        dev = self._dev_index()
        units = int(default_units_per_warp(self.n_blocks_max, self.batch, torch.cuda.get_device_properties(self.device).multi_processor_count, self.seq_len))
        self._score = {
            hb: compile_qsa_score_paged(
                h_i=q.index_heads,
                s_q=self.seq_len,
                page_size=self.page_size,
                units_per_warp=units,
                has_block_lens=hb,
                use_pdl=self.use_pdl,
                device=dev,
                cache_dtype=self.cache_dtype,
            )
            for hb in (True, False)
        }
        topk = qsa_topk_decode_kernel_module()
        c, threads = self.topk_knobs
        self._topk = topk.compile_qsa_topk_decode(cols_cap=self.topk_cols_cap, top_k=q.top_k, cluster_ctas=c, threads_per_cta=threads, use_pdl=self.use_pdl)
        self._run_topk = topk.run_qsa_topk_decode
        if not self.fused:
            from .kernels.qsa_compress import compile_qsa_compress, compile_qsa_compress_step

            self._q_recipe = compile_qsa_compress(
                dtype=self.dtype, d=q.index_head_dim, rope_dim=g.rope_dim, pool=1, has_seq_lens=False, table_dtype=self.dtype, out_dtype=self.cache_dtype
            )
            self._step = {
                sd: compile_qsa_compress_step(
                    dtype=self.dtype,
                    d=q.index_head_dim,
                    rope_dim=g.rope_dim,
                    pool=q.block_size,
                    has_n_commit=False,
                    table_dtype=self.dtype,
                    use_pdl=self.use_pdl,
                    paged=True,
                    ring_slots=self.ring_slots,
                    ring_slot_dtype=sd,
                    index_cache_dtype=self.cache_dtype,
                )
                for sd in (torch.int32, torch.int64)
            }

    def views(self, workspace: torch.Tensor, lay: "_Intermediates") -> dict:
        """The five slots as typed views of the workspace."""
        q, b, s = self.geom.qsa, self.batch, self.seq_len
        t = b * s
        return dict(
            qi=_view(workspace, lay.ix_q, (t, q.index_heads, q.index_head_dim), self.cache_dtype),  # the indexer cache dtype (e4m3 at 1 B under the attribute)
            logits=_view(workspace, lay.ix_logits, (t, self.n_blocks_max), torch.float32),
            counts=_view(workspace, lay.ix_counts, (t,), torch.int32),
            ids=_view(workspace, lay.ix_ids, (t, q.top_k), torch.int32),
            cslot=_view(workspace, lay.ix_cslot, (b, s), torch.int32),
        )

    def execute(
        self,
        proj: torch.Tensor,  # [T, n_qkvg] the slab stage (1) wrote; the band's columns are read in place (the unfused arm)
        *,
        workspace: torch.Tensor,
        lay: "_Intermediates",
        w_iq_norm: torch.Tensor,  # [index_head_dim]
        w_ik_norm: torch.Tensor,  # [index_head_dim]
        cos: Optional[torch.Tensor],  # [B, S_q, rope_dim] per-token tables (the unfused query norm); None under the table arm (fused only)
        sin: Optional[torch.Tensor],
        cos_pos: torch.Tensor,  # [1 or B, S_pos, rope_dim] the position-indexed tables (the compress arm)
        sin_pos: torch.Tensor,
        index_kv_cache: torch.Tensor,  # [num_pages, P / bs, 1, D_i] the paged compressed-key cache
        index_state_cache: torch.Tensor,  # [num_ring_blocks, cap, 1, D_i] the raw-key ring
        index_state_block_table: torch.Tensor,  # [B, 1] int32
        index_state_slot_mapping: Optional[torch.Tensor],  # [T] iff the ring-slots arm
        kv_lens: torch.Tensor,  # [B] int32: the KV length after the step
        block_table: torch.Tensor,  # [B, max_pages] int32: the main table
        slot_dtype: torch.dtype,  # slot_mapping's dtype: the step form's artifact under the ring-slots arm
        block_ids_out: Optional[torch.Tensor] = None,  # [T, top_k] int32 (or [B, S_q, top_k]): the list, written in place
        block_lens_out: Optional[torch.Tensor] = None,  # [T] int32 (or [B, S_q]): min(top_k, the visible count)
        current_stream,
    ) -> torch.Tensor:
        """Run the step's indexer and return the ``[T, top_k]`` int32 list stage (4) consumes -- the caller's ``block_ids_out`` when
        given, else the workspace slot.  Every buffer was form-checked by the block; nothing is allocated, nothing is read back."""
        from .kernels.qsa_score_paged import run_qsa_score_paged

        if self._score is None or self._topk is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        g, q = self.geom, self.geom.qsa
        b, s, t = self.batch, self.seq_len, self.batch * self.seq_len
        n, d_i, h_i = g.n_qkvg, q.index_head_dim, q.index_heads
        stream = int(current_stream)
        v = self.views(workspace, lay)
        if not self.fused:
            # The unfused twin of the prepare's indexer arms: the queries (pool 1, batch = token, block = head, the tables broadcast over the
            # heads) into ix_q, then the step form over the band's raw key -- the ring and the block a row completes, the paged tables.
            from .kernels.qsa_compress import run_qsa_compress, run_qsa_compress_step

            eps = float(q.index_norm_eps)
            q_raw = torch.as_strided(proj, (t, h_i, d_i), (n, d_i, 1), proj.storage_offset() + g.qkvg_offsets[ProjBlock.INDEX])
            cos_q = cos.view(t, 1, g.rope_dim).expand(t, h_i, g.rope_dim)
            sin_q = sin.view(t, 1, g.rope_dim).expand(t, h_i, g.rope_dim)
            run_qsa_compress(self._q_recipe, q_raw, v["qi"], w_iq_norm, cos_q, sin_q, None, eps=eps, norm_weight_offset=0.0, stream=stream)
            k_new = torch.as_strided(proj, (b, s, d_i), (s * n, n, 1), proj.storage_offset() + g.index_k_raw_offset)
            run_qsa_compress_step(
                self._step[slot_dtype],
                k_new,
                index_state_cache,
                index_kv_cache,
                w_ik_norm,
                cos_pos,
                sin_pos,
                None,
                v["cslot"],
                None,
                eps=eps,
                norm_weight_offset=0.0,
                stream=stream,
                kv_lens=kv_lens,
                block_table=block_table,
                index_state_block_table=index_state_block_table,
                index_state_slot_mapping=index_state_slot_mapping,
            )
            self.launches += 2
        lens = None if block_lens_out is None else block_lens_out.view(t)
        run_qsa_score_paged(
            self._score[lens is not None],
            v["qi"],
            index_kv_cache,
            block_table,
            kv_lens,
            v["logits"],
            v["counts"],
            lens,
            top_k=q.top_k,
            scale=float(d_i) ** -0.5,
            stream=stream,
        )
        ids = v["ids"] if block_ids_out is None else block_ids_out.view(t, q.top_k)
        self._run_topk(self._topk, v["logits"], v["counts"], ids, stream=stream)
        self.launches += 2
        return ids


# ---------------------------------------------------------------------------
# 6. The public API
# ---------------------------------------------------------------------------


class _CacheWrite(_Stage):
    """(4w) SERVING write-through: the post-norm, post-RoPE K and the V of every token into PAGED pools
    ``[num_pages, H_kv, page_size, D]`` at ``slot_mapping [T]`` (flat slot = page x page_size + offset), and the RAW
    indexer key (``index_k_raw``: the fifth band's key head, pre-norm, un-rotated) into its own
    ``[num_pages, page_size, index_head_dim]`` pool when the geometry declares the band.

    Built only under ``paged_kv_page_size > 0`` -- a declaration ATTRIBUTE (it changes the input contract), never a knob.
    ONE launch for K and V (``kernels/cache_write.py`` writes two operands per launch), a second for the indexer key.
    Runs AFTER norm + RoPE and reads the very operands stage (4) consumes -- the slab's K / V column bands in place, or
    the compact buffers of the out-of-place layout -- so a pool row is bitwise the row attention saw.  The pools are
    the activation dtype (bf16 / f16); the e4m3 pools of a quantized cache arrive with the cache-dtype attribute and
    its cast kernel.  Needs no workspace: the pools and the slot mapping are the caller's.  Token-addressed: under
    ``thd=True`` (the block is ``B = 1, S = T`` inside) it is the same launch over the same ``[T, H_kv, D]`` views, so
    the packed form needs no arm of its own -- a pad token's slot is the caller's ``-1``.

    The pool contract is the SDPA adapter's own paged contract: a 4-D view ``[num_pages, H_kv, page_size, D]`` with ``D``
    contiguous and the other three strides FREE -- HND compact ``(H x P x D, P x D, D, 1)``, NHD by strides
    ``(P x H x D, D, H x D, 1)``, or a serving stack's unified ``[num_pages, H_kv, P, 2D]`` content row with K at ``+0``
    and V at ``+D`` (token stride ``2D``; the two halves bound as the two pool views) -- nothing is repacked, and every
    non-innermost stride has to be a whole number of 16-byte vectors (a typed ``ValueError`` otherwise).

    The row the kernel addresses is a copy, so the two pre-development tables of a FROST kernel are empty here (no SMEM,
    no mbarrier).  A negative slot is the serving stacks' padding convention and writes nothing; a slot past the pool's
    capacity writes nothing either (the kernel range-checks on device; slot VALUES are never read on the host).
    """

    def __init__(
        self,
        geometry: GatedAttentionBlockGeometry,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        page_size: int,
        use_pdl: bool = False,
        index_raw_pool: bool = True,  # appended: the raw-key arm (False under the in-block indexer's decode step: the raw key lives in the RING)
    ) -> None:
        self.name = "cache_write"
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.page_size = int(page_size)
        self.use_pdl = bool(use_pdl)  # the kernel's programmatic-dependent-launch form (kernels/cache_write.py: the launch attribute + the griddepcontrol pair)
        # the raw indexer key's pool arm: the band's second launch -- not under the decode indexer over the pools, whose raw key enters the
        # paged ring (the prepare / the step form) and whose `index_k_raw` is refused by name
        self.index_band = bool(geometry.index_band) and bool(index_raw_pool)
        self._recipes = None  # {slot dtype: (K / V recipe, indexer recipe or None)}

    def check_support(self) -> None:
        # Rule 7: a too-old CuTe DSL reads as a VERSION decline here, before the kernel module (which imports the DSL) is loaded.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

        too_old = cutedsl_requirement_error("the gated attention block's paged KV-cache write-through")
        if too_old is not None:
            raise NotImplementedError(too_old)
        from .kernels.cache_write import validate_cache_write_shape

        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: bf16 / f16 pools only (the activation dtype), got {self.dtype}")
        validate_cache_write_shape(self.geom.d_head, self.page_size, _ELEMENTWISE_THREADS)
        if self.index_band:
            validate_cache_write_shape(self.geom.qsa.index_head_dim, self.page_size, _ELEMENTWISE_THREADS)

    def compile(self) -> None:
        """Both slot dtypes are compiled (int32 and int64 are the two the serving stacks hand over; the dtype is a
        compile key), so ``execute`` dispatches on a guaranteed cache hit (Rule 4) whichever one arrives."""
        from .kernels.cache_write import compile_cache_write

        g = self.geom
        self._recipes = {}
        for slot_dtype in (torch.int32, torch.int64):
            kv = compile_cache_write(
                dtype=self.dtype,
                h=g.h_kv,
                d=g.d_head,
                page_size=self.page_size,
                slot_dtype=slot_dtype,
                two_operands=True,
                threads_per_cta=_ELEMENTWISE_THREADS,
                rows_per_group=_ELEMENTWISE_ROWS_PER_GROUP,
                const_head_count=_ELEMENTWISE_CONST_HEAD_COUNT,
                use_pdl=self.use_pdl,
            )
            idx = (
                compile_cache_write(
                    dtype=self.dtype,
                    h=g.qsa.index_kv_heads,
                    d=g.qsa.index_head_dim,
                    page_size=self.page_size,
                    slot_dtype=slot_dtype,
                    two_operands=False,
                    threads_per_cta=_ELEMENTWISE_THREADS,
                    rows_per_group=_ELEMENTWISE_ROWS_PER_GROUP,
                    const_head_count=_ELEMENTWISE_CONST_HEAD_COUNT,
                    use_pdl=self.use_pdl,
                )
                if self.index_band
                else None
            )
            self._recipes[slot_dtype] = (kv, idx)

    def moved_bytes(self, slot_dtype=torch.int32) -> int:
        """HBM traffic of the K / V launch (+ the indexer launch when the band is declared) -- the denominator for the SOL number."""
        from .kernels.cache_write import moved_bytes

        g = self.geom
        t, es, sb = self.batch * self.seq_len, _itemsize(self.dtype), _itemsize(slot_dtype)
        total = moved_bytes(t, g.h_kv, g.d_head, elem_bytes=es, n_ops=2, slot_bytes=sb)
        if self.index_band:
            total += moved_bytes(t, g.qsa.index_kv_heads, g.qsa.index_head_dim, elem_bytes=es, n_ops=1, slot_bytes=sb)
        return total

    def execute(
        self,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        index_src: Optional[torch.Tensor] = None,
        index_cache: Optional[torch.Tensor] = None,
        current_stream=None,
    ) -> None:
        """``k_src`` / ``v_src`` are the ``[T, H_kv, D]`` operands stage (4) reads (any token stride); ``k_cache`` /
        ``v_cache`` the ``[num_pages, H_kv, page_size, D]`` pools; ``slot_mapping`` ``[T]`` int32 / int64; ``index_src``
        the ``[T, index_kv_heads, index_head_dim]`` raw-key slice and ``index_cache`` its ``[num_pages, page_size, index_head_dim]``
        pool -- both given iff the band is declared."""
        from .kernels.cache_write import run_cache_write

        if self._recipes is None:
            raise RuntimeError("call compile() before execute()")
        if (index_src is None) != (index_cache is None) or (index_cache is not None) != self.index_band:
            raise ValueError(
                f"{self.name}: index_src / index_cache are given together, and exactly when the stage carries the raw-key pool arm (the indexer band; "
                "not under the decode indexer over the pools, whose raw key lives in the ring)"
            )
        if slot_mapping.dtype not in self._recipes:
            raise ValueError(f"{self.name}: slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
        kv, idx = self._recipes[slot_mapping.dtype]
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(k_src.device).cuda_stream
        run_cache_write(kv, k_src, v_src, k_cache, v_cache, slot_mapping, stream=stream)
        if idx is not None:
            # The 3-D indexer pool as the kernel's 4-D [P, 1, page_size, D] view (one raw key head; its head stride is never scaled).
            run_cache_write(idx, index_src, None, index_cache.unsqueeze(1), None, slot_mapping, stream=stream)


def _token_rows(x: torch.Tensor, name: str, width: int, t: int, t_live: Optional[int] = None) -> torch.Tensor:
    """The caller's token matrix as its ``[rows, width]`` view -- ``.view`` only (a copy would be a silent extra kernel, Rule 1), the row
    count read from the LEADING shape (never inferred with ``-1``: a RoPE table of ``rope_dim`` 0, the RMSNorm-only geometry, is a
    ``[T, 0]`` tensor whose row count ``-1`` cannot infer).  Admitted forms: ``[T, width]`` / ``[1, T, width]`` / a dense ``[B, S, width]``
    with ``B x S == T`` (the first ``t_live`` rows when a live count is given), or -- under ``num_tokens_live`` -- exactly ``[T_live, width]``
    / ``[1, T_live, width]`` (a serving stack's ``hidden_states[rows]`` / ``output[rows]`` with no staging copy).  Any other row count is a
    typed ``ValueError`` naming the admitted counts.  ``rows`` = ``t`` without a live count, else ``t_live``."""
    if x.ndim == 2:
        n = int(x.shape[0])
    elif x.ndim == 3 and int(x.shape[0]) == 1:
        n = int(x.shape[1])
    elif x.ndim == 3:
        n = int(x.shape[0]) * int(x.shape[1])
    else:
        raise ValueError(f"{name} must be a [T, {width}] / [1, T, {width}] / [B, S, {width}] token matrix, got shape {tuple(x.shape)}")
    if n == t:
        x2 = x.view(t, width)
        return x2 if t_live is None else x2[:t_live]
    if t_live is not None and n == t_live:
        return x.view(t_live, width)
    admitted = f"{t}" if t_live is None else f"{t} (the declared capacity) or {t_live} (num_tokens_live)"
    raise ValueError(f"{name} must carry {admitted} token rows of width {width}, got {tuple(x.shape)}")


class _QuantizedCacheWrite(_Stage):
    """(4w) under ``kv_cache_dtype=torch.float8_e4m3fn``: the SERVING write-through into e4m3 pools -- the post-norm, post-RoPE K
    and the V of every token QUANTIZED per tensor (``code = sat_e4m3(row x (1 / scale))``: the serving stacks' fp8 KV-cache recipe,
    ``k_scale`` / ``v_scale`` DEscales, value = code x scale) and written at ``slot_mapping``; the raw indexer key keeps its bf16 pool
    (a serving indexer cache's dtype is its own question).

    NO new kernel: two launches of the FP8 pipeline's per-tensor cast (``kernels/quantize.py``: the source a slab band or a compact
    buffer at any token stride, the destination the compact e4m3 ``kc8`` / ``vc8`` workspace slots; the scale a 1-element fp32 device
    word -- here the RECIPROCAL of the caller's scale, written on the launch stream by the block) and ONE launch of the bf16 cache
    write (``kernels/cache_write.py``) copying those rows BITWISE into the e4m3 pools THROUGH THEIR bf16 VIEWS: an e4m3 ``[T, H_kv, D]``
    row is a bf16 ``[T, H_kv, D / 2]`` row and an e4m3 pool ``[num_pages, H_kv, P, D]`` a bf16 pool ``[num_pages, H_kv, P, D / 2]``
    (every non-innermost stride is a whole number of 16-byte vectors, so ``Tensor.view(torch.bfloat16)`` is an exact view of HND /
    NHD / unified-row pools alike; the kernel copies bit patterns and never casts; ``D / 2 = 128`` is the indexer pool's own row
    width).  The two pre-development tables of both kernels stay empty (no SMEM, no mbarrier).

    Built only under ``paged_kv_page_size > 0`` with the attribute; ``check_support`` gates the DSL floor before any kernel import
    (Rule 7).  Three launches where the bf16 write-through runs one -- the fusion into the projection epilogue is a later lever.
    """

    def __init__(self, geometry: GatedAttentionBlockGeometry, *, batch: int, seq_len: int, dtype: torch.dtype, page_size: int) -> None:
        self.name = "cache_write_e4m3"
        self.geom = geometry
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self.dtype = dtype  # the activation dtype: the source rows and the indexer pool
        self.page_size = int(page_size)
        self.index_band = bool(geometry.index_band)
        # ONE per-tensor cast recipe at h_kv heads serves K and V (the same [T, H_kv, D] form); the scale slot is the reciprocal
        self._quant = _Quantize(geometry, batch=batch, seq_len=seq_len, dtype_in=dtype, heads=geometry.h_kv, name="quantize_kv_cache")
        self._recipes = None  # {slot dtype: (K / V bit-copy recipe at D / 2 bf16 pairs, indexer recipe or None)}

    def check_support(self) -> None:
        # Rule 7: a too-old CuTe DSL reads as a VERSION decline here, before the kernel modules (which import the DSL) are loaded.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

        too_old = cutedsl_requirement_error("the gated attention block's e4m3 paged KV-cache write-through")
        if too_old is not None:
            raise NotImplementedError(too_old)
        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: bf16 activations only (the e4m3 KV cache is a form of the bf16 pipeline), got {self.dtype}")
        if self.geom.d_head % 16:
            raise NotImplementedError(
                f"{self.name}: the e4m3 rows are copied as bf16 pairs in 16-byte vectors; d_head must be a multiple of 16, got {self.geom.d_head}"
            )
        from .kernels.cache_write import validate_cache_write_shape

        validate_cache_write_shape(self.geom.d_head // 2, self.page_size, _ELEMENTWISE_THREADS)
        if self.index_band:
            validate_cache_write_shape(self.geom.qsa.index_head_dim, self.page_size, _ELEMENTWISE_THREADS)
        self._quant.check_support()

    def compile(self) -> None:
        """Both slot dtypes (int32 / int64) are compiled like the bf16 write-through's, so ``execute`` dispatches on a guaranteed cache
        hit (Rule 4); the per-tensor cast recipe once."""
        from .kernels.cache_write import compile_cache_write

        self._quant.compile()
        g = self.geom
        self._recipes = {}
        for slot_dtype in (torch.int32, torch.int64):
            kv = compile_cache_write(
                dtype=torch.bfloat16,
                h=g.h_kv,
                d=g.d_head // 2,
                page_size=self.page_size,
                slot_dtype=slot_dtype,
                two_operands=True,
                threads_per_cta=_ELEMENTWISE_THREADS,
                rows_per_group=_ELEMENTWISE_ROWS_PER_GROUP,
                const_head_count=_ELEMENTWISE_CONST_HEAD_COUNT,
            )
            idx = (
                compile_cache_write(
                    dtype=self.dtype,
                    h=g.qsa.index_kv_heads,
                    d=g.qsa.index_head_dim,
                    page_size=self.page_size,
                    slot_dtype=slot_dtype,
                    two_operands=False,
                    threads_per_cta=_ELEMENTWISE_THREADS,
                    rows_per_group=_ELEMENTWISE_ROWS_PER_GROUP,
                    const_head_count=_ELEMENTWISE_CONST_HEAD_COUNT,
                )
                if self.index_band
                else None
            )
            self._recipes[slot_dtype] = (kv, idx)

    def moved_bytes(self, slot_dtype=torch.int32) -> int:
        """HBM traffic of the three launches (+ the indexer launch when the band is declared) -- the denominator for a SOL number."""
        from .kernels.cache_write import moved_bytes

        g = self.geom
        t, es, sb = self.batch * self.seq_len, _itemsize(self.dtype), _itemsize(slot_dtype)
        total = 2 * self._quant.moved_bytes() + moved_bytes(t, g.h_kv, g.d_head, elem_bytes=1, n_ops=2, slot_bytes=sb)
        if self.index_band:
            total += moved_bytes(t, g.qsa.index_kv_heads, g.qsa.index_head_dim, elem_bytes=es, n_ops=1, slot_bytes=sb)
        return total

    @staticmethod
    def _as_bf16_pairs(x: torch.Tensor) -> torch.Tensor:
        """An e4m3 tensor with a contiguous last axis of even extent as the bf16 tensor over the SAME bytes (``[..., D]`` ->
        ``[..., D / 2]``): an exact view (torch halves the last extent and every other stride; a stride that is not a whole number
        of bf16 elements is torch's own refusal -- the pool contract's 16-byte rule makes every pool and slot pass)."""
        return x.view(torch.bfloat16)

    def execute(
        self,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
        kc8: torch.Tensor,
        vc8: torch.Tensor,
        inv_k_scale: torch.Tensor,
        inv_v_scale: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        index_src: Optional[torch.Tensor] = None,
        index_cache: Optional[torch.Tensor] = None,
        current_stream=None,
    ) -> None:
        """``k_src`` / ``v_src`` the ``[T, H_kv, D]`` post-RoPE operands (any token stride); ``kc8`` / ``vc8`` the compact e4m3 workspace
        slots; ``inv_k_scale`` / ``inv_v_scale`` 1-element fp32 device words holding ``1 / k_scale`` / ``1 / v_scale``; ``k_cache`` /
        ``v_cache`` the e4m3 pools; ``slot_mapping`` ``[T]`` int32 / int64; the indexer pair iff the band is declared."""
        from .kernels.cache_write import run_cache_write

        if self._recipes is None:
            raise RuntimeError("call compile() before execute()")
        if (index_src is None) != (index_cache is None) or (index_cache is not None) != self.index_band:
            raise ValueError(f"{self.name}: index_src / index_cache are given together, and exactly when the geometry declares the indexer band")
        if slot_mapping.dtype not in self._recipes:
            raise ValueError(f"{self.name}: slot_mapping must be int32 or int64, got {slot_mapping.dtype}")
        for nm, x in (("k_cache", k_cache), ("v_cache", v_cache), ("kc8", kc8), ("vc8", vc8)):
            if x.dtype != KV_CACHE_E4M3:
                raise ValueError(f"{self.name}: {nm} must be torch.float8_e4m3fn (the e4m3 KV cache), got {x.dtype}")
        kv, idx = self._recipes[slot_mapping.dtype]
        stream = current_stream if current_stream is not None else torch.cuda.current_stream(k_src.device).cuda_stream
        # the per-tensor casts: code = sat_e4m3(row x (1 / scale)) -- one fp32 rounding, the FP8 pipeline's own kernel
        self._quant.execute(k_src, kc8, inv_k_scale, current_stream=stream)
        self._quant.execute(v_src, vc8, inv_v_scale, current_stream=stream)
        # the bit copy into the pools through the bf16 views (D / 2 pairs per row); the indexer key into its bf16 pool as before
        run_cache_write(
            kv, self._as_bf16_pairs(kc8), self._as_bf16_pairs(vc8), self._as_bf16_pairs(k_cache), self._as_bf16_pairs(v_cache), slot_mapping, stream=stream
        )
        if idx is not None:
            run_cache_write(idx, index_src, None, index_cache.unsqueeze(1), None, slot_mapping, stream=stream)


@dataclass(frozen=True)
class _ServingDecl:
    """The declaration facts the serving forms' typed declines read -- ONE record, built by the constructor from its arguments and
    by :meth:`GatedAttentionBlockFwd.serving_envelope` from a hypothetical EXTEND declaration, so both read the same table."""

    kv_source: str
    thd: bool
    paged_kv_page_size: int
    token_capacity: Optional[int]
    save_for_backward: bool
    return_lse: bool
    geometry: GatedAttentionBlockGeometry
    s_q: int = 0  # the DENSE declaration's S (the DECODE form's step width); 0 under thd
    fuse_gate: bool = False


def _is_paged(d: _ServingDecl) -> bool:
    return d.kv_source == KV_SOURCE_PAGED


# (name, exception class, predicate over the declaration facts, the message) -- the typed declines of the EXTEND form and of
# the bounded token form, in the order a caller hears them.  A QsaSpec geometry under the THD-only EXTEND form is not a row: the
# sparse declaration's own rows run first and refuse thd=True only through a record without the packed arm (:data:`_QSA_THD_DECLINE`);
# with the arm claimed the packed QsaSpec declaration reaches the table and meets its ``sparse`` row like the dense one;
# the dense DECODE form reaches the table's ``sparse`` row.  The constructor raises the FIRST row whose predicate holds
# (``_check_serving_declaration``); ``serving_envelope`` publishes every row's name and message (``ExtendEnvelope.declines``)
# and evaluates the geometry rows for its ``served`` verdict -- one table, so no decline is reachable at declaration that the
# envelope did not predict.  Rows the write-through attribute already declines for every ``kv_source`` (``quant``,
# ``save_for_backward`` with ``paged_kv_page_size``) are not repeated here: the EXTEND form requires that attribute and hears
# those declines from it first.
_EXTEND_DECLINES: Tuple[Tuple[str, type, Callable[[_ServingDecl], bool], str], ...] = (
    (
        "write_through",
        ValueError,
        lambda d: _is_paged(d) and d.paged_kv_page_size <= 0,
        "kv_source='paged' needs paged_kv_page_size > 0: the EXTEND form writes the step's tokens through into the pools first and then "
        "attends over the pools, so the write-through attribute is part of its declaration",
    ),
    (
        "decode_rows",
        NotImplementedError,
        lambda d: _is_paged(d) and not d.thd and d.geometry.qsa is None and d.s_q > DECODE_MAX_ROWS,
        _DECODE_ROWS_DECLINE,
    ),
    (
        "sparse",
        NotImplementedError,
        lambda d: _is_paged(d) and d.geometry.qsa is not None and d.thd,
        "kv_source='paged' with geometry.qsa on a packed (thd=True) declaration: the sparse core serves packed sequences and paged pools one "
        "at a time (the packed sequence's K / V row offset composes with a dense tensor only, never with a page pool), so the block-sparse "
        "EXTEND form over the pools is not served -- declare the dense [B, S_q] form: the block-sparse DECODE / extend-with-prefix mode (one "
        "list per sequence at S_q <= 4, one list per row past it, the step's rows at the end of each sequence)",
    ),
    (
        "diagonal",
        ValueError,
        lambda d: _is_paged(d) and not (d.geometry.is_causal and d.geometry.causal_bottom_right),
        "kv_source='paged' needs geometry.is_causal=True and geometry.causal_bottom_right=True: a sequence's new tokens sit at the END of its "
        "logical KV range [0, kv_len), so query row i of a sequence with q_len new tokens attends keys k <= i + kv_len - q_len (the "
        "bottom-right diagonal; top-left would let the first new token see one key only)",
    ),
    (
        "window",
        NotImplementedError,
        lambda d: _is_paged(d) and (d.geometry.window_left >= 0 or d.geometry.window_right > 0),
        "kv_source='paged' is validated on the plain bottom-right causal diagonal only (window_left=-1, window_right in (-1, 0)); a sliding "
        "window or a right-widened band over the pools is a follow-up",
    ),
    (
        "head_dim",
        NotImplementedError,
        lambda d: _is_paged(d) and d.geometry.d_head != 256,
        "kv_source='paged' is validated at d_head == 256 only (the Rubin d256 paged THD leg); other head dims are a follow-up",
    ),
    # (retired) "fuse_gate": the paged THD body stage (4) binds -- the shared SM100 d256 half body compiled for cc 10.7 -- carries the
    # epilogue-gate seams (the paged gate), so fuse_gate=True rides the EXTEND form's SDPA epilogue (5 -> 4 stages) and
    # the row is gone from the table; a hypothetical declaration that asked for it hears nothing (the envelope lists no such row).
    (
        "return_lse",
        NotImplementedError,
        lambda d: _is_paged(d) and d.return_lse and d.geometry.qsa is None,
        "kv_source='paged' writes no LSE: the EXTEND form is an inference form (return_lse=False); an LSE over the pools is a follow-up "
        "(the block-sparse mode writes it: the sparse core's LSE over the pools, [B, H_q, S_q] fp32)",
    ),
    (
        "token_capacity_dense",
        ValueError,
        lambda d: d.token_capacity is not None and not d.thd,
        "token_capacity is THD-only (thd=True): it bounds the PACKED token rows a step may leave as pad rows; a dense [B, S, d_model] block "
        "has no packed capacity",
    ),
    (
        "token_capacity_training",
        NotImplementedError,
        lambda d: d.token_capacity is not None and d.save_for_backward,
        "token_capacity with save_for_backward=True: the backward's weight-gradient GEMMs contract over every one of the T rows, so a step "
        "with fewer live tokens than the capacity would contaminate dW_qkvg / dW_o; the bounded token form is inference-only",
    ),
)


def _device_cc_of(device) -> Optional[Tuple[int, int]]:
    """``(major, minor)`` of a CUDA device (a ``torch.device`` / string / index), None for a non-CUDA device or no CUDA."""
    if device is None:
        return None
    try:
        dev = torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        return None
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    return tuple(int(x) for x in torch.cuda.get_device_capability(dev))


def _first_serving_decline(decl: _ServingDecl) -> Optional[Tuple[str, type, str]]:
    """The first row of :data:`_EXTEND_DECLINES` whose predicate holds for ``decl`` -- ``(name, exception class, message)`` -- or None."""
    for name, exc, pred, msg in _EXTEND_DECLINES:
        if pred(decl):
            return name, exc, msg
    return None


_QSA_THD_DECLINE = (
    "QsaSpec with thd=True: the sparse adapter's record declines THD (the kernel body does not carry the packed-sequence arm); "
    "declare the dense [B, S, d_model] form"
)
"""The sparse declaration's own THD row (``_check_qsa_declaration``), RECORD-DRIVEN: the sparse core's record claims ``thd`` (served
since 2026-10-08), so the row is reachable only through a record that drops the arm -- then it is the reason the envelope gives a
QsaSpec geometry for the EXTEND form.  With the arm claimed, a packed QsaSpec declaration reaches the serving table like a dense
one and meets its ``sparse`` row (the same row the DECODE form reaches)."""

_DECODE_REQUIRES = (
    "a DENSE declaration: sample_h [B, S_q, d_model] with 1 <= S_q <= DECODE_MAX_S_Q = DECODE_MAX_ROWS = 32 (the claimed envelope S_q <= 4 on the packed "
    "16- / 32-column decode tile; an MTP verify step to S_q = 8 on the decode tile in one unit, the paged prefill tile with the group packed within its "
    "DECODE_PREFILL_TILE_ROWS = 128 or two token units; past a group's packed routes the decode tile UNPACKED, one head per CTA, up to 32 tokens; "
    "decode_route_policy picks the single-unit route or the adapter's token units, decode_q_tile_max caps the tile's width table), the batch uniform",
    "paged_kv_page_size=P (a positive multiple of 16; validated at page_sizes)",
    "kv_source='paged'",
    "geometry.is_causal=True, geometry.causal_bottom_right=True, window_left=-1, window_right in (-1, 0), qsa=None, d_head=256",
    "sample_k_cache / sample_v_cache (the pools' [num_pages, H_kv, P, D] shape / dtype / strides) and sample_block_table ([B, max_pages] int32, any row stride)",
    "bf16 / f16 activations (quant=None), save_for_backward=False, return_lse=False, token_capacity=None; fuse_gate rides the decode tile's split combine",
    "execute(block_table=, kv_lens=, k_cache=, v_cache=, slot_mapping=) with seq_lens=None (kv_lens binds the KV side only)",
)
_DECODE_DECLINE_NAMES = (
    "write_through",
    "decode_rows",
    "diagonal",
    "window",
    "head_dim",
    "return_lse",
    "token_capacity_dense",
)  # `sparse` is the packed form's
_SPARSE_DECODE_REQUIRES = (
    "a DENSE declaration on a QsaSpec geometry: sample_h [B, S_q, d_model], any S_q >= 1 (the sparse core runs one work item per query row), the batch uniform",
    "paged_kv_page_size=P (a positive multiple of 16; a 4-token block never straddles a page)",
    "kv_source='paged'",
    "geometry.is_causal=True, geometry.causal_bottom_right=True, window_left=-1, window_right in (-1, 0), qsa=QsaSpec(index_source='caller' | 'indexer'), d_head=256",
    "sample_k_cache / sample_v_cache (the pools' [num_pages, H_kv, P, D] shape / dtype / strides) and sample_block_table ([B, max_pages] int32, CONTIGUOUS: the sparse core binds the whole table)",
    "index_source='indexer' (the in-block indexer's DECODE step over the pools; bf16, index_band=True, S_q x next_pow2(index_heads) <= 64, a power-of-two page, bf16 main pools "
    "and raw-key ring; the compressed-key cache bf16 or e4m3 per QsaSpec.index_cache_dtype): the three side-cache samples at declaration -- sample_index_kv_cache [num_pages, P / 4, 1, 128] (the paged compressed-key cache on the MAIN block "
    "table), sample_index_state_cache [num_ring_blocks, cap >= S_q + 3, 1, 128] (the raw-key ring), sample_index_state_block_table [B, 1] int32 -- and at execute "
    "index_kv_cache, index_state_cache, index_state_block_table, cos_pos / sin_pos [1 or B, >= max_pages x P, rope_dim] (the position-indexed tables), w_iq_norm / w_ik_norm; "
    "index_state_slot_mapping [T] iff index_state_slot_mapping_present, cos / sin None iff rope_position_tables_only; the list comes back through block_ids_out / block_lens_out "
    "(one list per row); index_k_raw / index_k_compressed / block_ids / n_commit refused; fuse_prepare None (the measured default) / True (one prepare launch) / False (the unfused twin), bitwise",
    "bf16 / f16 activations (quant=None), save_for_backward=False, token_capacity=None; return_lse served ([B, H_q, S_q] fp32); fuse_gate served (the core's epilogue unsplit, the split combine otherwise)",
    "the list form (QsaSpec.index_list_form; None = the S_q rule below under caller lists, 'per_row' under the in-block indexer): 'shared' -- S_q <= 4, ONE list per "
    "sequence -- execute(block_ids=) [B, top_k] int32 (+ block_lens [B]) selected at the step-0 position kv_lens[b] - S_q and shared by the step's rows "
    "(row j sees the listed blocks plus every token from the step-0 tail start to its own position); 'per_row' -- any S_q, one list per row, block_ids [B, S_q, top_k] / "
    "[B x S_q, top_k] (+ block_lens [B, S_q] / [B x S_q]) at positions kv_lens[b] - S_q + j (the serving stacks' per-row selection; the extend-with-prefix form); "
    "None = 'shared' at S_q <= 4 and 'per_row' past it",
    "decode_split_kv=None (the fill rule: ceil(SMs / (B x H_kv x S_q)) chunks, at most min(16, ceil((top_k + 1) / 32)); where that fill leaves a second wave of SMs, "
    "the wave rule's split: the fewest waves x (tiles per chunk + 1) under the same cap) or an int in [1, ceil((top_k + 1) / 32)]: performance only",
    "execute(block_table=, kv_lens=, k_cache=, v_cache=, slot_mapping=, block_ids=[, block_lens=]) with seq_lens=None (kv_lens binds the KV side only; a row at a negative position is dead)",
    "kv_cache_dtype=None (bf16 / f16 pools) or torch.float8_e4m3fn (bf16 activations, S_q <= 4, return_lse=False, decode_split_kv None / 1, fuse_norm_rope=False, "
    "inplace_qkv None / False: the pools' e4m3 codes under per-tensor fp32 k_scale / v_scale at execute -- the write-through quantizes, the step reads through the "
    "gathered cast + the d256 decode tile)",
)


def _qsa_decode_decline_texts(geometry: GatedAttentionBlockGeometry, dtype: torch.dtype) -> Dict[str, str]:
    """The typed declines of the block-sparse DECODE mode -- the list form and the in-block indexer's rows -- keyed by name: ONE
    spelling for the constructor (which raises them) and :meth:`serving_envelope` (which publishes them under ``decode.declines``),
    formatted from the geometry and the activation dtype alone (no ``S_q``, no device: a raised text equals its envelope entry byte
    for byte).  Each row carries ONE shared token the tests pin: ``shared_list_rows`` ``tail covers at most two blocks``;
    ``indexer_f16`` ``needs bf16 activations``; ``indexer_head_dim`` ``packs 128-wide heads``; ``indexer_thd`` ``scores dense [B, S]
    prompts``; ``indexer_query_rows`` ``query rows per sequence (at most 64``; ``indexer_page_size`` ``a power-of-two count of compressed
    rows``; ``indexer_e4m3_pool`` ``in-kernel dequant arm``; ``indexer_shared_list`` ``selects per row``; ``fuse_prepare_rope_tables_unfused``
    ``the unfused query norm reads the per-token``.  (The ``indexer_e4m3_cache`` row retired 2026-10-21: the e4m3 compressed-key cache is
    SERVED at decode -- a served form has no decline row; ``decode.index_cache_dtypes`` lists it.)  The host facts come from the DSL-free
    modules (``cudnn.QSA.selection`` / ``cudnn.QSA.selection_paged``) and the sparse record, never from a kernel module."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import INDEXER_SCORER_HEAD_DIM
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection_paged import MAX_QUERY_ROWS, MMA_M, padded_heads

    q = geometry.qsa
    cap = int(_sparse_record().index_shared_list_max_tokens)
    h_i, d_i = int(q.index_heads), int(q.index_head_dim)
    max_s_q = MAX_QUERY_ROWS // padded_heads(h_i)
    return {
        "shared_list_rows": (
            f"QsaSpec(index_list_form='shared') past {cap} rows per sequence: the shared per-sequence list is anchored at the step-0 position and "
            f"its tail covers at most two blocks (S_q <= {cap}); declare index_list_form='per_row' (one list per row, [B, S_q, top_k])"
        ),
        "indexer_f16": (
            f"QsaSpec(index_source='indexer') needs bf16 activations: the indexer scorer packs bf16 queries and compressed keys, got {dtype}; "
            "declare the bf16 pipeline, or hand the selection in under index_source='caller'"
        ),
        "indexer_head_dim": (
            f"QsaSpec(index_source='indexer', index_head_dim={d_i}): the indexer scorer packs {INDEXER_SCORER_HEAD_DIM}-wide heads; "
            "another head dim has no scorer (the caller-list form serves any tile-aligned band)"
        ),
        "indexer_thd": (
            "QsaSpec(index_source='indexer') with thd=True: the in-block indexer scores dense [B, S] prompts; its packed-sequence form (the "
            "scorer over a query prefix, one compressed-key row range per sequence) is a follow-up -- hand the selection in under "
            "index_source='caller'"
        ),
        "indexer_query_rows": (
            f"QsaSpec(index_source='indexer') over the pools at index_heads={h_i}: the decode scorer stages S_q x next_pow2(index_heads) = "
            f"S_q x {padded_heads(h_i)} query rows per sequence (at most {MAX_QUERY_ROWS}: {MAX_QUERY_ROWS // MMA_M} {MMA_M}-row MMA tiles), so a "
            f"step holds at most {max_s_q} rows per sequence -- hand a longer step's lists in under index_source='caller'"
        ),
        "indexer_page_size": (
            "QsaSpec(index_source='indexer') over the pools: the decode scorer splits a block id into its page and row by a shift, so "
            "paged_kv_page_size must hold a power-of-two count of compressed rows (16 / 32 / 64 / 128 at block size 4) -- declare one of "
            "those pages, or hand the step's lists in under index_source='caller'"
        ),
        "indexer_e4m3_pool": (
            "QsaSpec(index_source='indexer') with kv_cache_dtype=torch.float8_e4m3fn: the block-sparse decode mode over e4m3 pools runs the "
            "gathered cast + the d256 decode tile over a caller list; its composition with the in-block indexer waits for the sparse core's "
            "in-kernel dequant arm -- declare bf16 pools or index_source='caller'"
        ),
        "indexer_shared_list": (
            "QsaSpec(index_source='indexer', index_list_form='shared') over the pools: the decode indexer selects per row (one list per query "
            "row, the serving stacks' function); the step-0 shared selection is a follow-up arm of the paged scorer -- declare "
            "index_list_form='per_row' or None"
        ),
        "fuse_prepare_rope_tables_unfused": (
            "fuse_prepare=False with rope_position_tables_only=True: the unfused query norm reads the per-token cos / sin (the prefill compress at "
            "pool 1); the position-table arm is the fused launch's -- declare fuse_prepare=True or hand cos / sin"
        ),
    }


_EXTEND_REQUIRES = (
    "thd=True (the packed declaration: h / cos / sin / out as [T_max, .] token matrices)",
    "num_sequences=B_max, max_seq_len=token_capacity (one sequence may own the whole chunk), cu_seqlens as the stack hands the Q lengths",
    "token_capacity=T_max (the samples' packed row count; a step hands lengths summing to T_live <= T_max)",
    "paged_kv_page_size=P (a positive multiple of 16; validated at page_sizes)",
    "kv_source='paged'",
    "geometry.is_causal=True, geometry.causal_bottom_right=True, window_left=-1, window_right in (-1, 0), qsa=None, d_head=256",
    "sample_k_cache / sample_v_cache (the pools' [num_pages, H_kv, P, D] shape / dtype / strides) and sample_block_table ([B_max, max_pages] int32)",
    "bf16 / f16 activations (quant=None), save_for_backward=False, return_lse=False; fuse_gate as declared (True folds stage (5) into the paged SDPA's epilogue)",
)


class GatedAttentionBlockFwd(APIBase):
    """Gated attention block, forward. One call, one workspace, three to five launches.

    No ``Sm1xx`` suffix on the class: arch is a directory axis under ``kernels/``
    and a per-stage dispatch, not part of the user-visible name. A shipped name
    cannot be renamed, so the arch stays out of it.

    **The default pipeline (``inplace_qkv``, no fusion knobs) is FOUR launches;
    the two fusion knobs take it to three**::

        (1) proj         h            -> PROJ [T, N]            FROST GEMM        | fuse_norm_rope: (1)+(2)+(3) in ONE
        (2+3) norm+rope  PROJ[Q],[K]  -> in place  (+rstd)      this block's kernel| launch (the GEMM fork's epilogue)
                         geometry.qk_norm=False: RoPE only -- same stage, same launch, no norm weights (None), no rstd
        (3b) compact     PROJ[V]      -> V_c                    only when inplace_qkv=False
        (4) sdpa         PROJ[Q,K,V]  -> O  (+LSE)              FROST SDPA        | fuse_gate: (4)+(5) in ONE launch
        (5) gate         O, PROJ[G]   -> O in place             this block's kernel| (the SDPA's epilogue_gate)
        (6) out_proj     O            -> out                    FROST GEMM

    Both fusions are inference-only: each overwrites a tensor the backward needs
    (``q_pre``/``k_pre``, pre-gate ``O``) and is declined with ``save_for_backward``.

    **Serving write-through (``paged_kv_page_size > 0``; bf16 / f16 inference, dense or packed)** adds ONE launch
    between (2+3) and (4) -- ``(4w) cache write: PROJ[K], PROJ[V] -> k_cache / v_cache at slot_mapping`` (and the raw
    indexer key into ``index_k_raw`` when the geometry declares the band) -- and changes nothing stage (4) reads, so
    ``out`` is bitwise the block's without it.  A declaration ATTRIBUTE: ``execute`` then REQUIRES the pools and the
    slot mapping and refuses them otherwise.  The stage addresses TOKENS, so under ``thd=True`` (the block is
    ``B = 1, S = T`` inside) it is the same launch over the same ``[T, H_kv, D]`` views; a pad token's slot is ``-1``.
    The knob ``fuse_prepare=True`` (in-place bf16 / f16 only, the dense declaration with the write-through alone) runs (2+3)
    and (4w) as ONE launch,
    ``(2+3+4w) prepare: PROJ[Q], PROJ[K] -> in place; PROJ[K], PROJ[V] (and the raw key) -> the pools`` --
    bitwise the two-stage form, one launch fewer (two with the band).

    **Block-sparse attention (``geometry.qsa``, a :class:`QsaSpec`; bf16 / f16 inference, dense ``[B, S, d_model]`` or
    packed ``thd=True``)** swaps stage (4) for the index-list sparse core -- ``(4) sdpa_sparse  PROJ[Q,K,V], block_ids[,
    block_lens] -> O (+LSE)``, ``sdpa/fwd/kernels/sm107/sparse_d256_f16.py`` through its adapter (:class:`_SparseSdpa`) -- at
    the same operands, the same strides and the same launch count; ``execute`` then REQUIRES ``block_ids`` (``[T, top_k]``
    int32: each query's selected 4-token blocks, the open tail block always visible; under ``thd=True`` the ids are relative
    to the token's OWN sequence) and refuses it otherwise.  Every claim of that core is read off the adapter's capabilities
    record, never transcribed; what the core does not serve is a typed decline at declaration (the in-block indexer under
    ``thd`` among them; the paged write-through composes with the packed sparse block -- the stage addresses tokens).
    ``fuse_gate=True`` folds stage (5) into the sparse core's epilogue exactly as on the dense block (the GATE slice read in
    place, O gated after the dead-row select, the LSE untouched; stage (5) is not built).

    **The EXTEND form (``kv_source="paged"``; bf16 / f16 inference, PACKED, bottom-right)** -- every serving prefill of a
    sequence with a cached prefix (a chunked prefill, a prefix-cache hit, a speculative verify): the step's ``T_max``-row
    token matrices hold ``B_max`` sequences' NEW tokens (``execute(seq_lens=)`` as under THD), the write-through puts them
    into the pools, and stage (4) then reads EVERY sequence's K / V -- its prefix and the new tokens -- from the POOLS
    over ``[0, kv_lens[b])`` through ``block_table[b]`` on the bottom-right diagonal (``k <= i + kv_len - q_len``)::

        (1)+(2)+(3) proj + norm + rope   h [T_max, d_model]        -> slab          (or (1) then (2+3) unfused)
        (4w) cache write                  slab[K], slab[V]          -> k_cache / v_cache at slot_mapping
        (4)  sdpa (paged THD leg)         slab[Q], k_cache, v_cache -> O             block_table, kv_lens, seq_lens
        (5)  gate                         O, slab[G]                -> O in place    | fuse_gate: (4)+(5) in ONE launch
        (6)  out_proj                     O                         -> out [T_max, d_model]

    Five launches -- four with ``fuse_gate=True`` (the paged SDPA body carries the epilogue gate) -- one stream (the write is
    ordered before the read by STREAM order alone -- no programmatic dependent launch on the chain).  Declared as ``thd=True, num_sequences=B_max, max_seq_len=token_capacity, token_capacity=T_max,
    paged_kv_page_size=P, kv_source="paged", geometry.causal_bottom_right=True`` with the pools' and the block table's
    SAMPLES (their layout and row stride are compiled in; ``serving_envelope`` lists the rest).  ``token_capacity`` makes
    the packed row count a CAPACITY: a step's lengths sum to ``T_live <= T_max`` and rows ``[T_live, T_max)`` are pad rows
    (slot ``-1``, finite ``cos`` / ``sin``, never attended; their ``out`` rows unspecified but finite); ``execute(num_tokens_live=T_live)``
    (appended) launches the token-indexed stages over the live rows alone -- no recompile, ``h`` / ``cos`` / ``sin`` / ``slot_mapping`` /
    ``out`` at ``T_max`` or exactly ``T_live`` rows, rows ``[T_live, T_max)`` of ``out`` untouched.  Typed declines, in
    :data:`_EXTEND_DECLINES`'s order: a missing bottom-right diagonal, a window, another head dim, an LSE (a fused gate
    is SERVED: the shared SM100 d256 body compiled for cc 10.7 carries the gate seams on its paged / THD arms);
    ``quant`` / ``save_for_backward`` are the write-through's, a sparse geometry the table's ``sparse`` row (the
    block-sparse forms' attention over the pools is its own binding; the sparse stage reads the block's own K / V,
    dense or packed).  Dense declarations use the DECODE form described below.

    **The DECODE form (``kv_source="paged"`` on a DENSE ``[B, S_q, d_model]`` declaration, ``1 <= S_q`` -- the claimed
    envelope ``S_q <= 4``: one new token plus up to three speculative ones -- over ``paged_kv_page_size > 0``; bf16 / f16
    inference, the batch UNIFORM)**: the step's tokens are written through (4w) and stage (4) attends over the first
    ``kv_lens[b]`` keys of every sequence's ``block_table`` row -- the Rubin d256 DECODE tile (PackGQA over the whole group,
    a split-KV plan with fp32 partials in the workspace and one combine launch when the decode split model splits) inside
    its packed route, the same tile UNPACKED (one head per CTA) up to ``DECODE_MAX_ROWS`` rows past it (:func:`decode_route`
    names the kernel per ``(group, S_q)``; a wider dense step is a typed decline -- the EXTEND form's packed business) -- so
    the pipeline is ``(1)+(2)+(3) -> (4w) -> (4) [+ combine] -> (5) -> (6)``, five launches with ``fuse_norm_rope`` -- past decode row
    counts: at ``B x S_q <= 128`` (``kernels.proj_gemm.SMALL_M_MAX``) the declaration LOWERS stage (1) to the unfused chain, ``(1) -> (2)+(3)
    -> (4w) -> (4) [+ combine] -> (5) -> (6)`` (the small-M route's slab GEMM, the norm + RoPE pass with the paged write-through folded in
    -- ``fuse_cache_write_small_m``, by default below :data:`CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS` rows or at ``S_q = 1``, the separate
    cache write past it --: the measured faster step; ``fuse_norm_rope_effective`` / ``cache_write_in_norm_rope`` read it back,
    ``fuse_norm_rope_small_m=True`` keeps the fused fork for the A/B, as does an explicit ``fuse_cache_write=True`` -- the write-through arm
    lives in the fork and the request outranks the lowering).  Programmatic dependent launch is the form's default on its decode-tile routes
    on cc 10.7 (``use_pdl=None`` -> True, :data:`DECODE_PDL_DEFAULT_CCS`; ``use_pdl=False`` the plain chain): the norm + RoPE pass, the tile
    and its combine launch with the attribute, the FROST GEMM stages do regardless; ``pdl_stages`` / ``use_pdl_effective`` read it back.
    ``fuse_gate`` then rides the split COMBINE on the decode tile (``gate_fused``; the tile has no gate seams, the combine
    multiplies the merged fp32 value by ``sigmoid(G)`` before the single cast) and is its own launch on an unsplit tile.
    The pools' and the table's strides are the same declaration samples as the EXTEND form's (a row-sliced table at any row
    stride); ``execute`` REQUIRES ``block_table`` + ``kv_lens`` and REFUSES ``seq_lens``: the per-sequence KV length is ONE
    fact bound to the KV side only (``seq_kv_lens = kv_lens``, ``seq_q_lens = None`` -- every Q row live; a serving stack's
    padded entry, ``kv_lens = 0`` + block ``0`` + slot ``-1``, is a live row with zero keys and yields ``out = 0`` through the
    tile's empty-range SELECT, nothing written to block ``0``).  The tail tile READS the pool rows past ``kv_lens[b]`` (the
    stale rows of the last page, and block ``0`` for the pages a short row does not list) and masks them to ``P = 0``: they
    contribute nothing as long as they are FINITE, as a zero-initialized cache's are (a NaN there is ``0 x NaN``).  The
    typed declines are the same table's rows (:data:`_EXTEND_DECLINES`) a dense declaration can hit -- the diagonal, a
    window, another head dim, an LSE, a step wider than ``DECODE_MAX_ROWS`` -- and :meth:`serving_envelope`
    publishes them with the route per ``S_q`` (``decode``).

    **The block-sparse DECODE / extend-with-prefix mode (``kv_source="paged"`` on a dense ``[B, S_q]`` declaration of a
    ``QsaSpec`` geometry; bf16 / f16 inference, ``causal_bottom_right=True``; 2026-10-14)**: the same pipeline with stage (4)
    the index-list sparse core READING THE POOLS (:class:`_SparseSdpa`'s paged form: the pools as sampled, the whole
    ``[B, max_pages]`` table, ``kv_lens`` the per-sequence logical length, the step's rows at the END of each sequence).
    The list form is a declaration fact derived from ``S_q``: at ``S_q <= 4`` ONE list per sequence (``execute(block_ids=)``
    ``[B, top_k]`` + ``block_lens [B]``), selected at the step-0 position ``kv_lens[b] - S_q`` and shared by the step's rows
    (row ``j`` sees the listed blocks plus every token from the step-0 tail start to its own position -- the contract page's
    MTP definition); past four rows one list per row (``[B, S_q, top_k]``) at positions ``kv_lens[b] - S_q + j`` -- a chunked
    prefill over a cached prefix -- at any ``S_q`` (no 32-row cap: one sparse work item per row).  ``decode_split_kv``
    (``None`` = the fill rule :func:`sparse_decode_split_choice`, at most ``min(16, ceil((top_k + 1) / 32))``) is the one performance
    knob (fp32 partials in the workspace, the split combine as a second launch); ``fuse_gate`` rides the core's epilogue
    unsplit and the combine under a split; ``return_lse`` is served (``[B, H_q, S_q]`` fp32).  A dead sequence
    (``kv_lens = 0``) and a row at a negative position (a sequence shorter than the step) yield ``out = 0`` / ``LSE = -inf``
    exactly.  Typed declines: the packed form (the table's ``sparse`` row: the core serves THD and paged pools one at a
    time), a block-table sample that is not contiguous (the core binds the whole table), a split past the item's tile
    count, ``causal_bottom_right`` on the block's own K / V; the envelope says ``decode.sparse_core``.  The list form is
    ``QsaSpec.index_list_form`` (``None`` = the ``S_q`` rule above; ``"per_row"`` at any ``S_q``).

    **The in-block indexer's DECODE step over the pools (``QsaSpec(index_source="indexer", index_band=True)`` with
    ``kv_source="paged"``; bf16; 2026-10-21)**: the same pipeline with the step's selection derived IN the block from the band
    and the serving stack's two PAGED side caches, per row -- ``(1) -> (2+3+4w: the prepare launch with the three indexer arms)
    -> (4i: the paged scorer, the decode top-k) -> (4) [+ combine] -> (6)`` under ``fuse_prepare=True`` (8 launches at a split,
    6 unsplit), the unfused twin ``(1) -> (2)+(3) -> (4w) -> (4i: the pool-1 query norm, the paged compress step, the scorer,
    the top-k) -> ...`` (11 / 9) under ``fuse_prepare=False``, bitwise; ``fuse_prepare=None`` is the MEASURED default
    (:data:`SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS`: True on cc 10.7 since the 2026-10-21 end-to-end gate measured the fused step
    +9.6 .. +14.5 % at B <= 4 and +4.4 .. +10.0 % at B 32, bitwise; False on every other part until measured, and False out of place --
    ``inplace_qkv=False`` -- on every part: the AUTO reading is never True where an explicit True is declined).  The declaration adds the three
    side-cache SAMPLES (``sample_index_kv_cache`` ``[num_pages, P / 4, 1, 128]``, ``sample_index_state_cache`` ``[num_ring_blocks,
    cap >= S_q + 3, 1, 128]``, ``sample_index_state_block_table`` ``[B, 1]`` int32: their FORM, never their strides) and three
    input-contract flags (``index_state_slot_mapping_present``, ``index_commit_present`` -- a typed decline --,
    ``rope_position_tables_only``); ``execute`` adds ``index_kv_cache`` / ``index_state_cache`` / ``index_state_block_table`` /
    ``index_state_slot_mapping`` / ``cos_pos`` / ``sin_pos`` / ``n_commit`` (REQUIRED / iff declared / refused) and hands the
    per-row list back through ``block_ids_out`` / ``block_lens_out``; ``index_k_raw`` (the raw key lives in the ring) and
    ``index_k_compressed`` (the cache is paged) are refused by name.  The scorer sees the block the step's own row completes
    (the prepare runs first on the same stream), the list is canonical (ids descending, ``-1`` last) and the first execute of a
    fresh declaration captures with no warm-up (no plan constant, no allocation, no host read, no torch op on the path).  Typed
    declines (the envelope's rows): f16 activations, another head dim, ``S_q x next_pow2(index_heads) > 64``, a page off the
    powers of two, the e4m3 MAIN pool together with the indexer (``indexer_e4m3_pool``; the compressed-key cache itself may be e4m3
    through ``QsaSpec.index_cache_dtype``, served since 2026-10-21), the shared list form, ``index_commit_present=True``,
    ``rope_position_tables_only`` with the unfused twin; the envelope publishes the indexer facts (``decode.index_*``,
    ``decode.fuse_prepare_default``) so a serving stack reads them instead of re-spelling them.

    **The e4m3 KV cache (``kv_cache_dtype=torch.float8_e4m3fn``; bf16 inference; 2026-10-14)** -- the serving stacks'
    ``--kv-cache-dtype fp8`` recipe: the pools hold e4m3 codes under per-tensor fp32 DEscales ``execute(k_scale=, v_scale=)``
    (value = code x scale), a bf16 Q attends over the dequantized rows.  A declaration ATTRIBUTE beside ``paged_kv_page_size``
    (it changes the write-through's numerics), never a knob.  The WRITE-THROUGH then quantizes the post-RoPE K / V
    (:class:`_QuantizedCacheWrite`: two per-tensor casts ``sat_e4m3(row x (1 / scale))`` + the bit copy through the pools' bf16
    views; the raw indexer key keeps its bf16 pool) on every DENSE form that writes through over the block's own K / V (QsaSpec or
    dense; the packed prefill's quantizing write-through is a typed decline until its accept cell lands).  OVER THE POOLS it is served on the block-sparse DECODE mode at ``S_q <= 4`` (the shared list):
    stage (4) is :class:`_UpcastSparseSdpa` -- ``kernels/qsa_kv_upcast.py`` gathers the listed rows and the open tail, casts them
    to bf16 into the workspace scratch (``[B x S_q x H_kv, 2, 2176, 256]`` + ``kv_len_t``), and the Rubin d256 DECODE tile runs
    over it as a plain padded dense SDPA (no causal mask, ``seq_kv_lens = kv_len_t``, the group packed, unsplit); the mode
    norms + rotates OUT of place (the tile's per-item Q is a view of the compact Q alone: ``inplace_qkv=None`` reads as False,
    ``fuse_norm_rope`` is declined) and gates in its own launch.  Typed declines: e5m2; another dtype; the attribute without
    ``paged_kv_page_size``; ``quant`` / ``save_for_backward`` (the write-through's rows); f16 activations; over the pools the
    EXTEND form and the dense DECODE form (the paged THD body and the decode tile read bf16 / f16 pools: a dense e4m3 pool needs
    the in-kernel dequant arm), ``S_q > 4``, ``return_lse``, ``decode_split_kv > 1``, ``fuse_norm_rope`` / ``inplace_qkv=True``;
    pools of the wrong dtype (at declaration and at execute, both ways); the scales missing under the attribute or given
    without it.  The envelope says ``decode.kv_cache_dtypes``.

    **The fp8 indexer cache (``QsaSpec(index_source="indexer", index_cache_dtype=torch.float8_e4m3fn)``; 2026-10-14)** -- the
    in-block indexer's compressed keys and queries as plain e4m3 of the RMSNormed + rotated values (no scales; the caller's
    ``index_k_compressed`` is then an e4m3 buffer), scored through :func:`qsa_select_forward`'s e4m3 arm (the DSA MXFP8 scorer with unit
    scale blobs).  A declaration attribute of ``QsaSpec`` itself, never a knob (3 mantissa bits move the weakest selected blocks at the
    k-th boundary: 99.0 / 97.4 / 96.5 % set agreement with the bf16 selection at S = 4096 / 32768 / 131072 on synthetic
    activations; bf16 stays the default and the reference).  No new kernel: the compress launches write bf16 as before and the
    FP8 pipeline's per-tensor cast writes the e4m3 operands (two launches more; :class:`_Indexer`).  Typed declines: e5m2 and any
    other dtype; the attribute under ``index_source="caller"``; f16 activations (the indexer's row); a cache of the other dtype at
    execute, both ways; THD as for the bf16 indexer.  OVER THE POOLS (the in-block indexer's DECODE step, served 2026-10-21): the e4m3
    compressed-key cache ``sample_index_kv_cache`` / ``execute(index_kv_cache=)`` ``[num_pages, P / 4, 1, 128]`` in e4m3 (REQUIRED in the
    declared dtype, both ways), the ring bf16 -- the prepare / compress kernels STORE e4m3 directly (the e4m3 cast of the bf16 row, one
    rounding convention with the prefill form's rows: a cache a stack primes from them and this step appends is one chain, bitwise), the
    paged scorer reads it on its fp8 ``mma.sync`` arm, no cast kernel (the launch census is the bf16 form's); ``ix_q`` is carved at 1 B;
    ``decode.index_cache_dtypes`` reads ``(bf16, e4m3)``.  The selection is pinned EXACTLY against the fp64 top-k of the kernel's own e4m3
    operands (up to ties); the bf16-vs-e4m3 agreement is a statistic, never a gate.

    **MXFP8** (an :class:`MxQuantSpec` + e4m3 codes + F8_128x4 SF blobs for ``h`` /
    ``W_qkvg``), UNFUSED (9 stages = 9 kernel launches)::

        (1)  proj          h8+sf_h, W8+sf_w -> PROJ [T, N] bf16   block-scale FROST GEMM (E8M0 dequant in-MMA, no alpha)
        (2+3) norm+rope    in place (out of place into compact bf16 Q/K under save_for_backward)   this block's kernel
        (3q) quantize x3   PROJ[Q]/[K] rowwise, PROJ[V] columnwise -> q8/k8/v8 + sf_q/sf_k/sf_v   kernels/quantize_mxfp8.py
        (4)  sdpa          q8,k8,v8 + SF -> O bf16                 production prefill_d256_mxfp8.py (NATURAL, pair on dense legs)
        (5)  gate          O, PROJ[G] -> O in place                this block's kernel
        (5q) quantize_o    O -> o8 (PER-TENSOR scale_o, D1)        kernels/quantize.py
        (6)  out_proj      o8 -> out (alpha_o = descale_w_o / scale_o)   per-tensor FP8 FROST GEMM

    and FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``, 3 launches):
    ``proj(+norm+rope+block-quant -> q8/k8/v8 + sf_q/k/v + gate16) -> sdpa(+gate, e4m3 O UNSCALED;
    scale_o must be 1.0, D8) -> out_proj`` (the ``run_fused_proj_gemm_mxfp8`` fork twin is
    feature-detected; a checkout without it gets a typed decline).  Padding (``seq_lens_present``) is served under
    FP8 and MXFP8 like bf16: a dead entry (``seq_lens[b] == 0``) yields ``out[b] == 0`` exactly.

    **MXFP4 W_qkvg** (``MxQuantSpec.w_qkvg_dtype = torch.float4_e2m1fn_x2``: e2m1 codes ``[N, d_model // 2]``,
    two per byte, ``w_qkvg_sf`` unchanged): the SAME nine unfused stages with stage (1) on the mixed
    MXFP8 x MXFP4 block-scale row::

        (1)  proj          h8+sf_h, W4+sf_w -> PROJ [T, N] bf16   block-scale FROST GEMM, mixed row (fp8_e4m3 x fp4_e2m1, E8M0/32)

    and FULLY FUSED (``fuse_norm_rope=True, fuse_gate=True``, the SAME 3 launches as the e4m3 fused
    pipeline: inference only, dense only, ``S % 128 == 0`` at ``B > 1``, ``scale_o == 1.0``) with stage (1)
    on the fork's e2m1-B arm::

        (1)  proj+norm+rope+quant  h8+sf_h, W4+sf_w -> q8/k8/v8 + sf_q/sf_k/sf_v + gate16   the fused MXFP8 fork, mixed row
                                   (``NormRopeFusionParams.weight_fp4``: the same packed codes and the same blob; the
                                   fused stage also checks the packed TMA format's 32-byte base / row-stride rules)

    (feature-detected on ``NormRopeFusionParams.weight_fp4``: a checkout whose fork lacks the field declines typed).

    **fp4 O** (``MxQuantSpec.o_fp4 = Fp4Format.NVFP4 | MXFP4`` + an e2m1 ``W_o`` ``[d_model, H_q*D // 2]``
    with its F8_128x4 blob ``sample_w_o_sf`` / ``w_o_sf``): the per-tensor tail of BOTH MXFP8
    configurations is replaced -- UNFUSED (still 9 launches)::

        (5q') quantize_fp4_o  O bf16 -> o4 [T, H_q*D/2] e2m1 + sf_o (the GEMM's padded blob)   kernels/quantize_fp4.py
        (6')  out_proj        o4, W_o4 -> out                        fp4 x fp4 block-scale FROST GEMM (scales dequant in-MMA, no alpha)

    and FULLY FUSED (4 launches: the gated MXFP8 SDPA writes **bf16** O, then (5q') and (6')).
    ``scale_o`` / ``descale_w_o`` are pinned to 1.0 (no global scale in either format); ``o8`` is not
    reserved, ``o4`` / ``sf_o`` are appended at the end of the workspace arm.  Composes with the MXFP4
    ``W_qkvg`` above on the unfused pipeline (the mixed GEMM at (1), the fp4 tail at (5q') / (6')).

    Stage (3b) exists only because stage (1) is the UNFORKED FROST GEMM, which
    writes one fused ``[T, N]``. Q and K are de-interleaved for FREE by (2+3),
    which already reads and writes them; V is untouched between the projection
    and the SDPA. Stage (3b) makes V compaction explicit and hands compact
    storage to the prepared SDPA plan (Rule 2).
    **Forking stage (1) to write four compact
    buffers (§ 1) deletes (3b) outright**; until then it is the measurable price
    of not having forked, and V is 1/16 of Q so the price is small.

    Under inference ``O_gated`` needs no buffer: stage (5) gates O in place.
    Under training it does.

    **THD / packed sequences (``thd=True, num_sequences=B, max_seq_len=S_max,
    cu_seqlens=``)**: ``h`` / ``cos`` / ``sin`` / ``out`` are ``[T, .]`` (or
    ``[1, T, .]``) token matrices and ``execute(seq_lens=)`` is REQUIRED -- the
    ``[B]`` int32 lengths (or ``[B+1]`` prefix sums) of the packed sequences, read
    on device only.  Internally ``B = 1, S = T``: the same stages, the same
    launches, the same workspace carve as the dense ``B=1, S=T`` block (plus the
    SDPA's small packed-metadata scratch), the same record -- ``saved.lse`` is
    the head-major ``[1, H_q, T]``, ``saved.seq_lens`` the lengths tensor itself,
    ``saved.seq_lens_form`` its form.  Stage (4) runs the SDPA's varlen arm over
    packed ``(T, H, D)`` views in the natural tile order (a ``QsaSpec`` block
    runs the sparse core's packed arm over the same ``[1, T, H, D]`` views, the
    block ids relative to each token's sequence).  bf16 / fp16 (inference
    and training), the UNFUSED per-tensor FP8 and MXFP8 pipelines (the fp4 modes
    included; the three MXFP8 quantize stages write the SDPA row's PACKED
    per-sequence-tile-padded scale-factor layout into slots sized at the capacity
    ``_sf_slot_bytes_packed``), ``fuse_norm_rope`` and ``fuse_gate`` (bf16 / fp16
    inference; the gate rides the packed d256 prefill's epilogue through the
    SDPA's THD gate descriptor) and block-sparse attention with caller lists are
    served, the paged write-through composing with them; the fully fused quantized
    pipelines, ``seq_lens_present`` and the in-block indexer are typed declines.  Module
    docstring, "THD".

    **TRAINING (``save_for_backward=True``; bf16 / fp16, out of place, no fusion
    knob)** writes THROUGH the caller's :class:`SavedForBackward` instead of the
    workspace wherever the backward needs the tensor, same kernels, different
    buffers (``out`` is bitwise the inference block's)::

        (1) proj         h            -> saved.proj_slab [T, N]   (proj_slab mode)  |  workspace slab (gate-copy mode)
        (2+3) norm+rope  slab[Q],[K]  -> compact q, k (workspace) + saved.rstd_q / rstd_k  (slab columns stay PRE-norm = q_pre / k_pre)
        (3g) gate copy   slab[GATE]   -> saved.gate (compact)     gate-copy mode only; q_pre / k_pre likewise if buffers were passed
        (3b) compact     slab[V]      -> V_c
        (4) sdpa         q, k, V_c    -> saved.o (PRE-gate) + saved.lse
        (5) gate         saved.o, slab[G] -> o_gated (workspace, OUT of place)
        (6) out_proj     o_gated      -> out

    Six launches (seven in gate-copy mode), the inference chain's six.  The save
    mode is a declaration knob (``saved_gate_copy``) because the carve differs;
    ``execute`` checks the record against it -- and that ``saved.h`` IS ``h``,
    ``saved.seq_lens`` IS the ``seq_lens`` it runs with, and ``lse`` (optional)
    IS ``saved.lse`` -- before any launch (:meth:`_check_saved_set`).
    """

    def __init__(
        self,
        sample_h: torch.Tensor,  # [B, S, d_model]
        sample_w_qkvg: torch.Tensor,  # [N, d_model], N = (2*H_q + 2*H_kv) * D
        sample_w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False -- same positions
        sample_w_k_norm: Optional[torch.Tensor],  # [D]
        sample_cos: torch.Tensor,  # [B, S, ROPE_DIM] -- see "RoPE table contract"
        sample_sin: torch.Tensor,  # [B, S, ROPE_DIM]
        sample_w_o: torch.Tensor,  # [d_model, H_q * D]
        sample_out: torch.Tensor,  # [B, S, d_model]
        geometry: GatedAttentionBlockGeometry,
        *,
        return_lse: bool = False,
        save_for_backward: bool = False,  # implies return_lse; see SavedForBackward
        seq_lens_present: bool = False,
        inplace_qkv: Optional[bool] = None,  # None -> not save_for_backward
        fuse_norm_rope: bool = False,  # stages (2)+(3) inside stage (1)'s epilogue; needs inplace_qkv
        fuse_gate: bool = False,  # stage (5) inside stage (4)'s epilogue; inference only (no pre-gate O)
        quant: Optional[Union[QuantSpec, MxQuantSpec]] = None,  # FP8 (E4M3) per-tensor static scales, or MXFP8 (MxQuantSpec); None = bf16/f16
        sample_h_sf: Optional[torch.Tensor] = None,  # MXFP8 only: F8_128x4 E8M0 blob of h, proj_gemm.sf_blob_bytes(B*S, d_model) bytes (uint8 / e8m0)
        sample_w_qkvg_sf: Optional[torch.Tensor] = None,  # MXFP8 only: F8_128x4 E8M0 blob of W_qkvg, sf_blob_bytes(n_qkvg, d_model) bytes
        sample_w_o_sf: Optional[torch.Tensor] = None,  # fp4 O only (MxQuantSpec.o_fp4): the e2m1 W_o's F8_128x4 blob, sf_blob_bytes(d_model, h_q*d_head, block)
        saved_gate_copy: bool = False,  # training SAVE mode: False = the GEMM writes saved.proj_slab; True = the GATE band is copied into a compact saved.gate
        # APPENDED (THD): packed / ragged sequences.  h / cos / sin / out are PACKED token matrices [T, .] (or [1, T, .]) and
        # execute(seq_lens=) carries the per-sequence lengths for the Q and the KV side alike -- module docstring, "THD".
        thd: bool = False,
        num_sequences: Optional[int] = None,  # B, REQUIRED under thd: the length tensor has B entries ([B] lengths) or B+1 ([B+1] prefix sums)
        max_seq_len: Optional[int] = None,  # S_max, REQUIRED under thd: the longest sequence the plan admits (the SDPA's envelope)
        cu_seqlens: bool = False,  # the FORM of execute(seq_lens=): False = [B] int32 lengths, True = [B+1] int32 prefix sums
        # APPENDED (serving): write the post-RoPE K / V (and the raw indexer key, when the band is declared) THROUGH into paged
        # pools at execute(slot_mapping=).  A declaration ATTRIBUTE (it changes the input contract: execute then REQUIRES
        # k_cache / v_cache / slot_mapping and refuses them otherwise), never a knob.  0 = off.  Positive: a multiple of 16.
        paged_kv_page_size: int = 0,
        # APPENDED (serving, the EXTEND form): where stage (4) reads K / V from.  "self" = the block's own K / V (as before).
        # "paged" = the pools: every sequence's prefix AND the step's new tokens (written through first) are read from
        # k_cache / v_cache via execute(block_table=, kv_lens=) on the bottom-right diagonal -- the serving prefill of every
        # sequence with a cached prefix.  Requires paged_kv_page_size > 0, thd=True and geometry.causal_bottom_right=True; the
        # three samples below (REQUIRED there, REFUSED otherwise) carry the pools' layout and the block table's row stride,
        # which are compiled in.  A declaration ATTRIBUTE, never a knob.  Class docstring, "The EXTEND form".
        kv_source: str = KV_SOURCE_SELF,
        # APPENDED (serving, THD only): the BOUNDED token form.  The samples' T packed rows are a CAPACITY: a step may hand
        # lengths summing to T_live <= T (rows [T_live, T) are PAD rows -- finite inputs, out rows unspecified but finite, never
        # written to the cache under slot -1, never attended).  Must equal the samples' row count when given; None = the tight
        # contract (sum == T) as before.  Inference only (the backward's weight-gradient GEMMs contract over every row).
        token_capacity: Optional[int] = None,
        sample_k_cache: Optional[torch.Tensor] = None,  # kv_source="paged": the K pool [num_pages, H_kv, P, D] (shape / dtype / strides only)
        sample_v_cache: Optional[torch.Tensor] = None,  # kv_source="paged": the V pool (same contract; the same HND / NHD kind as K)
        sample_block_table: Optional[torch.Tensor] = None,  # kv_source="paged": the [num_sequences, max_pages] int32 block table (row stride free)
        # APPENDED (the DECODE form = kv_source="paged" on a DENSE [B, S_q] declaration; refused on the packed form and under "self"):
        # the logical KV bound (kv_lens[b] <= max_kv_len: the masks' clamp and the split model's extent; None = max_pages x P) and
        # the split-KV the form lowers with (None = the SDPA heuristics' decode split model; the one performance knob of the form:
        # fp32 partials in the workspace + one combine launch, the same function under every value).
        max_kv_len: Optional[int] = None,
        decode_split_kv: Optional[int] = None,
        # APPENDED (serving): WHERE the paged KV-cache write-through runs under paged_kv_page_size > 0.  None = AUTO: folded into the fused
        # projection's epilogue wherever that arm serves (fuse_norm_rope=True, bf16 / f16 activations, no indexer band -- both serving forms as
        # declared today: EXTEND 4 -> 3 stages with fuse_gate, DECODE 5 -> 4), the separate cache-write launch everywhere else; False = the
        # separate launch always (the A/B control and the fallback; on the DECODE form's unfused chain it also keeps the AUTO small-M fold of
        # fuse_cache_write_small_m=None off -- an explicit fuse_cache_write_small_m names that arm itself and is followed either way); True =
        # the fused arm, a typed decline on a form it cannot serve.  A
        # performance-only knob: the pools hold the SAME bytes either way (the pool row is the slab row, bitwise).  At decode row counts
        # an explicit True is the REQUEST that keeps the fused fork the AUTO lowering (fuse_norm_rope_small_m=None) would otherwise replace
        # (user request > heuristic proposal); only the explicit contradiction fuse_norm_rope_small_m=False + fuse_cache_write=True declines.
        fuse_cache_write: Optional[bool] = None,
        # APPENDED (serving): run stages (2)+(3) and the write-through (4w) as ONE launch (kernels/qsa_prepare.py).  A
        # PERFORMANCE knob -- the same function, every output bitwise the two-stage form; needs paged_kv_page_size > 0 and
        # the in-place bf16 / f16 pipeline (typed otherwise).  None = AUTO (the type widened from bool, the position kept; False keeps
        # its meaning everywhere): the MEASURED default -- True on the in-block indexer's DECODE step over the pools on the parts of
        # SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS (cc 10.7 since the 2026-10-21 measurement; `fuse_prepare_default` on the envelope) where the
        # fused form is admissible -- False out of place (inplace_qkv=False: the unfused twin serves that layout) and on every other form, never
        # True where an explicit True is declined; `fuse_prepare_requested` keeps the declaration, `fuse_prepare` the resolved knob.
        fuse_prepare: Optional[bool] = None,
        # APPENDED (serving, the e4m3 KV cache): the POOLS' dtype.  None = the activation dtype (the pools as before).  torch.float8_e4m3fn =
        # the serving stacks' fp8 KV cache: the write-through quantizes the post-RoPE K / V into e4m3 codes under per-tensor fp32 DEscales
        # (execute(k_scale=, v_scale=), REQUIRED there, refused otherwise) and the block-sparse DECODE mode (S_q <= 4) reads the pools
        # through kernels/qsa_kv_upcast.py + the bf16 d256 decode tile.  Needs paged_kv_page_size > 0 and bf16 activations.  A declaration
        # ATTRIBUTE (numerics-changing), never a knob.  Class docstring, "The e4m3 KV cache".
        kv_cache_dtype: Optional[torch.dtype] = None,
        # APPENDED (the DECODE form's route policy; refused on the packed form and under "self"): None = DECODE_ROUTE_SINGLE_UNIT -- the
        # decode tile in ONE token unit where its width table holds the packed rows, else the paged d256 prefill tile with the group
        # packed (the KV streamed once) wherever that tile packs the group; DECODE_ROUTE_TOKEN_UNITS = the SDPA adapter's own route (the
        # widest tile in up to two token units, then the unpacked tile).  A PERFORMANCE-ONLY knob: the same function under either (the
        # two routes' A/B); `decode_route` / `DecodeRoute.policy` read it back.  Module docstring, "The DECODE form".
        decode_route_policy: Optional[str] = None,
        # APPENDED (2026-10-15; the DECODE form's width cap; refused on the packed form and under "self"): None = the decode tile's whole
        # single-unit width table (config_sm107.D256_DECODE_SINGLE_UNIT_Q_TILES = (16, 32, 64)); 32 = the route before the 64-column width
        # landed (the measured widths: 24/2 at S_q 3 / 4 in two units of the 32-column tile), value for value.  A PERFORMANCE-ONLY knob
        # (the same function under every width -- the SDPA suite's cross-form bitwise pins); `DecodeRoute.q_tile_max` reads it back and
        # the stage hands it to the adapter so the compiled width is the declared one.  Module docstring, "The DECODE form".
        decode_q_tile_max: Optional[int] = None,
        # APPENDED (performance-only): programmatic dependent launch over the block's own launch chain.  None = AUTO (2026-10-19): True on the
        # dense DECODE form's decode-tile routes for the bf16 / f16 block on the parts of DECODE_PDL_DEFAULT_CCS (cc 10.7, where the default
        # was measured: 1.1-1.5 us per decode step at B 1, no cell losing beyond its control pair), False everywhere else -- the EXTEND form,
        # kv_source="self", the DECODE form's prefill route, every other part -- until measured there; `use_pdl_requested` keeps the value as
        # declared, `use_pdl` the resolved one.  True launches the norm + RoPE, cache-write, sigmoid-gate, d256 decode-tile and split-combine
        # kernels with the programmatic-stream-serialization attribute and traces their griddepcontrol pair (a wait before the first global
        # read of a predecessor's output, a release after the prologue); False is the plain chain on every part.  The FROST GEMM templates
        # launch with the attribute regardless (their module constant USE_PDL).  The same kernels, the same math: `out` is BITWISE the
        # use_pdl=False block's, eagerly and under CUDA-graph capture.  An explicit True is served for the bf16 / f16 block only (no quant, no
        # e4m3 KV cache: a typed decline; pools in the activation dtype, kv_cache_dtype None or that dtype, are the bf16 / f16 block); a QsaSpec
        # geometry takes the switch on its dense [B, S_q] forms (the block-sparse core's waits -- it triggers its dependent at its completion,
        # not at its prologue: the measured placement -- and its combine's pair, 2026-10-21; AUTO False
        # there until measured, SPARSE_DECODE_PDL_DEFAULT_CCS), thd=True x QsaSpec declined by name; AUTO reads False on the declined pipelines
        # and never True where the explicit True is refused.  A PER-STAGE fact (2026-10-15): a stage whose
        # kernel carries no sites -- the paged d256 prefill tile of the DECODE form's prefill route and of the EXTEND form, the dense prefill
        # tiles under kv_source="self" -- launches plain and the link serializes there (correct by PDL's semantics: its successor's wait
        # blocks on full completion); the switch is never forwarded to such a kernel.  `pdl_stages` reads back the stages launching with the
        # attribute, `use_pdl_effective` whether every switched stage of the chain does; `DecodeRoute.pdl_capable`, `DecodeEnvelope.pdl_stages`
        # / `.pdl_default` and `ExtendEnvelope.pdl_stages` say so before anything is declared.
        use_pdl: Optional[bool] = None,
        # APPENDED (2026-10-15; the DECODE form only; performance-only): WHETHER the fused norm + RoPE projection runs at DECODE ROW COUNTS
        # (T = B x S_q <= kernels.proj_gemm.SMALL_M_MAX = 128 on the dense DECODE form, bf16 / f16, no indexer band, 16-bit pools) when
        # fuse_norm_rope=True.  None = AUTO = False: stage (1) is LOWERED to the unfused chain there -- the small-M route's slab GEMM, the
        # norm + RoPE pass with the cache write folded in (`fuse_cache_write_small_m`'s default; six launches with the split combine) --
        # because the 256-wide fused fork pads one token tile and streams its weight at a fraction of the machine (the measured "route +
        # unfused" step is the faster one at B 1 ..
        # 32), unless an explicit fuse_cache_write=True asks for the fork's write-through arm (the request outranks the lowering);
        # True = the fused fork as declared (the former behaviour; the A/B's control arm).  `fuse_norm_rope` stays what was declared,
        # `fuse_norm_rope_effective` reads back what runs (`cache_write_fused` follows it: the write-through arm lives in the fork).  Refused
        # (typed) with fuse_norm_rope=False and off the dense DECODE form.  Module docstring, "The DECODE form".
        fuse_norm_rope_small_m: Optional[bool] = None,
        # APPENDED (2026-10-16; the DECODE form's unfused chain; performance-only): WHERE the paged KV-cache write-through runs when stage (1)
        # is the unfused chain at decode row counts (the C0 lowering above, or fuse_norm_rope=False) on the dense DECODE form, bf16 / f16,
        # no quant, no indexer band, 16-bit pools, the TMA norm + RoPE kernel.  None = AUTO (2026-10-19): the MEASURED rule -- folded in
        # where the arm is served and T = B x S_q < CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS (128) or S_q == 1, the separate cache-write launch
        # (the chain as before: seven launches with the split combine) at T >= 128 with S_q > 1, where the arm measured slower, and wherever
        # an explicit fuse_cache_write=False asked for the separate launch by name (user request > heuristic proposal); True = the
        # write-through FOLDED INTO the norm + RoPE launch (kernels/qk_norm_rope_tma.py's pool-write arm: the slab's V band copied and the
        # post-RoPE K / V rows written to the pools from the same tiles, the separate cache-write stage not built -- six launches), a typed
        # decline on a form that cannot serve it (the fused fork, a quantized pipeline, the e4m3 cache, the indexer band, the EXTEND / packed
        # forms, a geometry the TMA tiling does not fit); False = the separate launch always.  The same bytes land in the pools and the slab
        # either way (the pools bitwise the separate write).  `cache_write_in_norm_rope` reads back what runs.
        fuse_cache_write_small_m: Optional[bool] = None,
        # APPENDED (the in-block indexer's DECODE step over the pools: QsaSpec(index_source="indexer") with kv_source="paged" on a dense
        # [B, S_q] declaration; REQUIRED there, REFUSED everywhere else): the serving stack's two side caches and the ring's block table as
        # SAMPLES (their FORM is compiled in -- dims, dtype, the ring's cap, the page's rows -- never their strides: the kernels bind them
        # symbolically, so a stack may re-allocate them with other strides without a new plan).  sample_index_kv_cache: the paged
        # compressed-key cache [num_pages, P / 4, 1, 128] in the activation dtype on the MAIN block table (one compressed row per 4-token
        # block; written by the step for the block a committed row completes, read by the scorer); sample_index_state_cache: the raw-key
        # ring [num_ring_blocks, cap, 1, 128] (block 0 the null block of a padding sequence; cap >= S_q + 3: a completing block pools the
        # three positions below the step while the step writes S_q rows); sample_index_state_block_table: [B, 1] int32, the ring block per
        # sequence (any row stride).
        sample_index_kv_cache: Optional[torch.Tensor] = None,
        sample_index_state_cache: Optional[torch.Tensor] = None,
        sample_index_state_block_table: Optional[torch.Tensor] = None,
        # APPENDED (the decode indexer's input-contract flags; each a COMPILE KEY of the prepare launch / the step form, so each value is
        # its own artifact and execute matches it in both directions): index_state_slot_mapping_present -- True = execute(index_state_slot_mapping=)
        # names the ring WRITE rows (the stack's own CircularBuffer slots, block x cap + pos % cap, -1 = no write; the same dtype as
        # slot_mapping; the ring's blocks then contiguous), False = the rows derive from index_state_block_table and kv_lens (the two forms
        # are bitwise); index_commit_present -- the two-phase commit (execute(n_commit=): the accepted prefix of the step's rows) -- a typed
        # decline (the serving stacks commit every row; a rejected draft's rows are rewritten by the next step at the same positions);
        # rope_position_tables_only -- True = every rotation of the step (Q / K, the indexer queries, the compressed keys) reads the
        # position-indexed cos_pos / sin_pos at the row's device-derived position and execute(cos=, sin=) are REFUSED (a stack hands the
        # model's table once instead of gathering per-token rows every step; fuse_prepare=True only: the unfused query norm reads per-token
        # rows), False = cos / sin per token as everywhere else, cos_pos / sin_pos for the block starts alone.
        index_state_slot_mapping_present: bool = False,
        index_commit_present: bool = False,
        rope_position_tables_only: bool = False,
        # APPENDED (the lean block-sparse decode tile; performance-only, the block-sparse stage's knob): WHICH BODY of the index-list sparse
        # core serves stage (4) on a QsaSpec geometry -- "auto" (the default): the placement rule (cc 10.7, B x H_kv <= 4 CTAs per chunk --
        # the lean grid's own unit: the step's S_q rows ride inside the CTA, so S_q is not in the bound --, the lean body admissible) on the
        # parts of heuristics.SPARSE_LEAN_TILE_DEFAULT_CCS (cc 10.7 since the end-to-end gate measured the lean route; the bound
        # LEAN_SPARSE_MAX_UNITS = 4 refit from its table, its S_q 4 rows fixing the unit), the tcgen05 core everywhere else; "lean": the lean decode
        # tile (mma.sync + cp.async, 128 threads per m16 tile, no TMEM / TMA / CLC; the DECODE mode over the pools at S_q <= 4 under the
        # shared list -- one list per sequence, every row of the step in one CTA -- or the per-row list at one row per sequence, bf16 /
        # f16, G <= 16, top_k >= 8; it ALWAYS splits: decode_split_kv None =
        # the lean split of the default k, or one of heuristics.sparse_lean_splits(top_k), which SELECTS k), a typed decline naming the arm
        # elsewhere -- never a silent fallback; "tcgen05": the tcgen05 core as before.  The same O / LSE under either body within the
        # fixed-order-sum budget (never bitwise ACROSS the bodies; bitwise WITHIN each: two launches, replay == eager, paged == dense).  A
        # pinned decode_split_kv that is not a lean split keeps "auto" on the tcgen05 core (the pin is the stronger knob; the resolved form
        # reads back).  `sparse_tile_form_requested` keeps the declaration, `sparse_tile_form` the resolved body, `sparse_tile` its
        # kernel_template (pinned against the compiled kernel at check_support); the envelope's `decode.sparse_tiles` / `.sparse_tile(...)`
        # / `.sparse_tile_lean_default` say so before anything is declared.  Refused (ValueError) off a QsaSpec geometry unless "auto".
        sparse_tile_form: str = "auto",
        # APPENDED: the lean body's k -- its 32-key sub-tiles per CTA (1 .. min(8, n_sub - 1); the split is ceil(n_sub / k)): None = the default
        # (heuristics.sparse_lean_sub_tiles_per_cta: 1 today, the k >= 2 renderings a measured lever), an int with sparse_tile_form="lean" only
        # (a lean-only pin: refused by name under "auto" and "tcgen05" so it is never silently ignored); with a pinned decode_split_kv the two
        # must agree.  Performance-only: the same function under every k.
        sparse_sub_tiles_per_cta: Optional[int] = None,
    ):
        """Validate the declaration -- the dtype / ``quant`` / scale-blob halves, the fusion and save-mode knobs against
        ``save_for_backward``, the THD envelope, the paged-cache write-through -- normalise the samples, record the knobs and
        build the stage list in pipeline order (declared, not compiled: ``check_support`` / ``compile`` follow)."""
        super().__init__()
        self._warn_experimental_api()
        self.geom = geometry
        self.dtype = sample_h.dtype
        self.device = sample_h.device
        # THE SPARSE TILE FORM (the constructor's note): the vocabulary and the k pin's shape here, device-free; the body is RESOLVED once the
        # form is known (`_check_serving_declaration` on the block-sparse decode mode, `_resolve_sparse_tile_form_off_decode` on the dense /
        # packed QsaSpec forms) -- `sparse_tile_form_requested` keeps the declaration, `sparse_tile_form` / `sparse_tile` the resolution.
        self._check_sparse_tile_form_declaration(sparse_tile_form, sparse_sub_tiles_per_cta, geometry)
        # Programmatic dependent launch over the chain (the constructor's note): the bf16 / f16 block only -- pools in the activation dtype
        # (`kv_cache_dtype` None or that dtype: the same block) -- because the quantized pipelines' quantize stages and the e4m3 write-through
        # carry no griddepcontrol sites yet, so neither of those chains overlaps at its attention link and the switch is a typed decline there
        # (the per-stage form below is the bf16 / f16 chains', whose one site-less link is a prefill tile).  The block-sparse core takes the
        # switch since 2026-10-21 (its sites + its combine's pair) on the dense BSHD and paged chains; under the packed form (`thd=True`) it
        # is declined by name below (the adapter's rule: the packed rendering reads the setup launch's metadata above every role wait).  The
        # FIRST decline of such a declaration, so the message names the switch rather than a later check's vocabulary; a 16-bit kv_cache_dtype
        # that is not the activation dtype passes here and hears the attribute's own ValueError below.  None = AUTO: nothing to decline here;
        # resolved from the form's facts below (`_pdl_default`), before any stage is built, and only INSIDE the domain this decline admits
        # (the two readings of the knob answer for one domain) -- `use_pdl_requested` keeps the declaration, `use_pdl` the resolved switch
        # every stage reads.
        if use_pdl is not None and not isinstance(use_pdl, bool):
            raise TypeError(f"use_pdl must be None (auto), True or False, got {type(use_pdl).__name__}")
        self.use_pdl_requested: Optional[bool] = use_pdl
        self.use_pdl = bool(use_pdl)
        if self.use_pdl and (quant is not None or kv_cache_dtype not in (None, torch.bfloat16, torch.float16)):
            raise NotImplementedError(
                "use_pdl=True is served for the bf16 / f16 block only (no quant / QuantSpec, no e4m3 KV cache -- kv_cache_dtype None or the "
                "activation dtype --): those pipelines' stages carry no programmatic-dependent-launch sites; declare use_pdl=False"
            )
        if self.use_pdl and geometry.qsa is not None and thd:
            # ONE statement of the kernel fact (the adapter's `pdl_decline_reason`): raised verbatim here at declaration and again by the
            # adapter's check_support on the stage; both suites pin the two sites equal, so neither text drifts alone when the scope moves.
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_gqa_sm107 import pdl_decline_reason

            raise NotImplementedError(f"use_pdl=True on a QsaSpec geometry: the block-sparse core's rule -- {pdl_decline_reason(thd=True)}")
        if self.use_pdl and geometry.qsa is not None and geometry.qsa.index_source == "indexer":
            # the in-block indexer (prefill: the band's scorer / top-k; decode: the paged step, the scorer, the top-k) threads the block's
            # resolved switch into its own kernels' compile keys, and that switched chain is not measured yet (the measured chain is the
            # caller-list block + band): declined by name here, under the pinned prefix, so the stage is only ever handed False.
            raise NotImplementedError(
                "use_pdl=True is served for the bf16 / f16 block only on the caller-list sparse chain, not under QsaSpec(index_source='indexer'): "
                "the in-block indexer's scorer / top-k / step kernels would take the switch unmeasured; declare use_pdl=False (or None = auto)"
            )
        # FP8: `h` and both weights arrive as e4m3 with a QuantSpec; every
        # activation the block's own kernels touch (slab, O, out, cos/sin, norm
        # weights) is bf16 -- the "activation dtype".  bf16/f16: the two coincide.
        # MXFP8: an MxQuantSpec instead, plus the two scale-factor blobs.
        if quant is not None and not isinstance(quant, (QuantSpec, MxQuantSpec)):
            raise TypeError(f"quant must be a QuantSpec (per-tensor FP8) or an MxQuantSpec (MXFP8), got {type(quant).__name__}")
        self.quant = quant
        self.mxfp8 = isinstance(quant, MxQuantSpec)
        if (self.dtype == torch.float8_e4m3fn) != (quant is not None):
            raise ValueError(
                "FP8 needs both halves: an e4m3 `h` AND a QuantSpec / MxQuantSpec (static scales). "
                f"Got h.dtype={self.dtype}, quant={'set' if quant is not None else 'None'}."
            )
        have_sf = sample_h_sf is not None or sample_w_qkvg_sf is not None
        if self.mxfp8 != have_sf or (self.mxfp8 and (sample_h_sf is None or sample_w_qkvg_sf is None)):
            raise ValueError(
                "MXFP8 needs all three halves: e4m3 `h` / `W_qkvg` codes, an MxQuantSpec AND both F8_128x4 E8M0 scale-factor blobs "
                f"(sample_h_sf, sample_w_qkvg_sf). Got quant={type(quant).__name__ if quant is not None else 'None'}, "
                f"sample_h_sf={'tensor' if sample_h_sf is not None else 'None'}, sample_w_qkvg_sf={'tensor' if sample_w_qkvg_sf is not None else 'None'}."
            )
        if self.mxfp8:
            # D8 rides the declaration: a fully fused MXFP8 block writes e4m3 O
            # UNSCALED, so scale_o != 1.0 is refused HERE (typed), not silently dropped.
            quant.validate(fused=bool(fuse_gate) and bool(fuse_norm_rope))
        elif quant is not None:
            quant.validate()
        # fp4 O (MxQuantSpec.o_fp4): the e2m1 W_o needs its scale blob, and a blob needs the mode -- both
        # halves or neither, at declaration (the field lives on MxQuantSpec only, so a QuantSpec / bf16
        # block can name the blob but never the mode: refused here, typed).
        self.o_fp4: Optional[Fp4Format] = quant.o_fp4 if self.mxfp8 else None
        if (self.o_fp4 is not None) != (sample_w_o_sf is not None):
            raise ValueError(
                "fp4 O needs both halves: MxQuantSpec.o_fp4 (Fp4Format.NVFP4 | MXFP4) AND sample_w_o_sf (the F8_128x4 scale blob of the e2m1 W_o, "
                f"proj_gemm.sf_blob_bytes(d_model, h_q*d_head, block) bytes). Got o_fp4={self.o_fp4}, "
                f"sample_w_o_sf={'tensor' if sample_w_o_sf is not None else 'None'}" + ("" if self.mxfp8 else " (only an MxQuantSpec carries o_fp4)") + "."
            )
        # geometry.qk_norm vs the two weight slots, BOTH directions, at
        # declaration (again at every execute).  A None sample would otherwise
        # be swallowed by _make_tensor_desc and surface as an untyped
        # AttributeError in check_support's dtype loop.
        _check_norm_weights_agree(geometry.qk_norm, sample_w_q_norm, sample_w_k_norm, prefix="sample_")
        self.act_dtype = torch.bfloat16 if quant is not None else self.dtype
        # THD (packed sequences).  The four appended knobs are one unit: a dense block takes none of them; a packed one
        # takes h / cos / sin / out as [T, .] (or [1, T, .]) token matrices and is INTERNALLY B = 1, S = T -- every stage
        # addresses tokens, so the dense carve, the dense record and the dense kernels serve it unchanged, and only the
        # SDPA stage grows a THD arm.  The samples are normalized to rank 3 HERE so every later shape check speaks (1, T, .).
        self.thd = bool(thd)
        self.cu_seqlens = bool(cu_seqlens)
        self.num_sequences = None if num_sequences is None else int(num_sequences)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        if not self.thd and (self.num_sequences is not None or self.max_seq_len is not None or self.cu_seqlens):
            raise ValueError("num_sequences / max_seq_len / cu_seqlens are THD-only (thd=True); a dense [B, S, d_model] block takes none of them")
        if self.thd:
            sample_h = _thd_token_matrix(sample_h, "sample_h", "d_model")
            sample_cos = _thd_token_matrix(sample_cos, "sample_cos", "rope_dim")
            sample_sin = _thd_token_matrix(sample_sin, "sample_sin", "rope_dim")
            sample_out = _thd_token_matrix(sample_out, "sample_out", "d_model")
        elif sample_h.ndim != 3:
            raise ValueError(f"sample_h must be [B, S, d_model], got {tuple(sample_h.shape)}")
        self.batch, self.seq_len, d_model = (int(x) for x in sample_h.shape)
        if d_model != geometry.d_model:
            raise ValueError(f"sample_h last dim {d_model} != geometry.d_model {geometry.d_model}")
        if self.thd and self.seq_len == 0:
            raise ValueError(
                "thd=True needs T >= 1 packed tokens (sample_h has 0 rows): the SDPA adapters refuse a zero packed capacity ('the packed token "
                "capacities must be positive') and a GEMM over M = 0 has nothing to launch -- an empty step is the caller's early-out"
            )
        self.save_for_backward = bool(save_for_backward)
        self.return_lse = bool(return_lse) or self.save_for_backward
        self.seq_lens_present = bool(seq_lens_present)
        # SAVE MODE (training only; appended, default = the proj_slab mode).  False:
        # stage (1) writes the caller-owned SavedForBackward.proj_slab and gate /
        # q_pre / k_pre / V are its column bands (saved_slab_views) -- zero copies,
        # the whole 34 KiB/token slab kept at the 397B geometry, and the workspace
        # loses its own slab.  True: the slab stays in the workspace and ONE
        # elementwise launch copies the GATE band into a compact saved.gate (q_pre /
        # k_pre only if the caller passed buffers) -- 16 KiB/token kept, the backward
        # recomputes the rest (RecomputePolicy.RECOMPUTE_QK_PRE).  A DECLARATION
        # knob because the workspace carve differs (_plan_workspace(saved_gate_copy=));
        # execute() verifies the record agrees (proj_slab given iff proj_slab mode).
        self.saved_gate_copy = bool(saved_gate_copy)
        if self.saved_gate_copy and not self.save_for_backward:
            raise ValueError("saved_gate_copy=True selects the gate-copy SAVE mode of a training forward and needs save_for_backward=True")
        # IN-PLACE Q/K/V. Norm+RoPE writes back over its own columns of the
        # fused projection and V is never moved, so stage (3b) disappears and
        # the three compact buffers with it -- 26% of the workspace.
        # Default: ON for inference, OFF the moment the backward needs q_pre /
        # k_pre. Explicit True with save_for_backward RAISES rather than
        # silently costing the caller a tensor the backward cannot rebuild.
        self.inplace_qkv = (not self.save_for_backward) if inplace_qkv is None else bool(inplace_qkv)
        # The two training guards are KEPT under qk_norm=False (D11): the
        # SavedForBackward contract still writes q_pre/k_pre out of place and
        # no block-level save_for_backward=True execute has validated a
        # relaxation (PR-B plan § 6 Q10).  Only the REASON differs, so the
        # message must not claim an RMSNorm backward that does not exist.
        _why_pre = (
            "norming in place destroys q_pre/k_pre, which the RMSNorm backward needs and cannot reconstruct (dividing out "
            "the norm weight is undefined at a zero weight and hostile at a small one -- see SavedForBackward)"
            if geometry.qk_norm
            else "rotating in place overwrites q_pre/k_pre, which the SavedForBackward contract still hands the backward "
            "out of place under qk_norm=False (RoPE-only; relaxing this needs a block-level training validation first)"
        )
        if self.inplace_qkv and self.save_for_backward:
            raise ValueError(
                f"inplace_qkv=True is incompatible with save_for_backward=True: {_why_pre}. Pass inplace_qkv=False, or recompute "
                "q_pre/k_pre from h by re-running the Q and K slices of the projection."
            )

        # FUSED norm+RoPE: stage (1)'s fork norms the Q/K tiles in its epilogue
        # and writes the slab the in-place path already reads.  OFF by default
        # until the perf node has ranked it against the unfused chain; it is
        # never a silent choice.  It has no pre-norm Q/K to hand a backward, so
        # it rides the same guard as inplace_qkv.
        self.fuse_norm_rope = bool(fuse_norm_rope)
        if self.fuse_norm_rope and not self.inplace_qkv:
            raise ValueError(
                f"fuse_norm_rope=True writes {'normed' if geometry.qk_norm else 'rotated'} Q/K straight into the projection slab "
                "(the in-place layout) and never materialises q_pre/k_pre, so it requires inplace_qkv=True and is incompatible "
                "with save_for_backward=True. Pass fuse_norm_rope=False for training."
            )

        # FUSED GATE: the production d256 SDPA's epilogue_gate specialization
        # reads GATE in its epilogue and writes O_gated, so stage (5) is not
        # built.  It overwrites the one tensor the backward's dG needs (pre-gate
        # O -- SavedForBackward), so training declines it here, the same way
        # inplace_qkv / fuse_norm_rope do.
        self.fuse_gate = bool(fuse_gate)
        # FP8 / MXFP8 serve exactly TWO configurations: UNFUSED (7 FP8 stages or
        # 9 MXFP8 stages -- 9 kernel launches either way, the FP8 Q/K/V quantize
        # stage being three launches) and FULLY FUSED (3 launches:
        # proj(+norm+rope+quant) -> sdpa(+gate, e4m3 O) -> out_proj).  Each
        # half-fused combination would be its own specialization (bf16 O + a
        # quantize pass, or a gate read from the bf16 slab) that nobody has
        # validated -- declined, typed, naming both knobs.
        _family = "MXFP8" if self.mxfp8 else "FP8"
        if quant is not None and self.fuse_gate != self.fuse_norm_rope:
            raise NotImplementedError(
                f"the {_family} pipeline is either fully fused or unfused: fuse_norm_rope and fuse_gate must BOTH be True "
                f"(3 launches) or BOTH be False ({'9' if self.mxfp8 else '7'} stages, 9 kernel launches); "
                f"got fuse_norm_rope={self.fuse_norm_rope}, fuse_gate={self.fuse_gate}"
            )
        # TRAINING under a quant spec (2026-10-01) serves the UNFUSED per-tensor FP8 and
        # MXFP8 pipelines: they write the bf16 training record (the slab with PRE-norm
        # Q/K bands, the bf16 pre-gate O, the exact fp32 LSE, rstd) exactly like the
        # bf16 forward, with norm+RoPE routed OUT of place into compact bf16 Q/K slots
        # (_plan_workspace: "FP8 / MXFP8 TRAINING").  The fp4 modes of the MXFP8 pipeline
        # (an e2m1 W_qkvg, an fp4 gated O with an e2m1 W_o) train on the UNFUSED pipeline
        # too: the record they write is byte for byte the MXFP8 record (the fp4 tail only
        # replaces the per-tensor o8 by o4 / sf_o in the workspace), and the MXFP8 backward
        # differentiates it over the caller's transposed e2m1 artifacts (api_bwd.py, "The
        # fp4 weight modes").  The fused forks are caught by the fuse_norm_rope /
        # inplace_qkv guards above and the fuse_gate guard below (typed, naming the knob).
        # seq_lens_present (a dense padding mask, incl. an EMPTY entry) is SERVED
        # under FP8 and MXFP8 since 2026-09-15.  The decline that used to sit here
        # ("the Rubin FP8 d256 SDPA hangs on seq_kv_lens == 0") is retired: the
        # rebased kernels pass 8/8 fresh processes at S=1000 and S=512 with a dead
        # entry, and the MXFP8 d256 leading-zero-length-KV L0 test passes on Rubin.
        # The block's own dead-entry oracle tests (test_block_fp8.py /
        # test_block_mxfp8.py: seq_lens=[s, 0] -> out[1] == 0 EXACTLY) pin it.
        if self.fuse_gate and self.save_for_backward:
            raise ValueError(
                "fuse_gate=True writes O_gated in place of O and never materialises the pre-gate O, which the backward's "
                "dG needs (see SavedForBackward: `o` is saved because dG needs pre-gate O; o_gated is recomputable, o is not). "
                "Pass fuse_gate=False for training."
            )
        # THD, typed, in this order: the length contract, the PIPELINE (so a fully fused quantized request hears about the
        # pipeline, not the knob), the knob, then the sizes.  Everything else -- the unfused per-tensor FP8 and MXFP8
        # pipelines (the fp4 modes included: the three quantize_mxfp8 stages run their PACKED arm, module docstring "THD"),
        # fuse_norm_rope for bf16 / fp16 inference, training -- is served at (1, T) with no further branch.
        if self.thd:
            if self.seq_lens_present:
                raise ValueError(
                    "thd=True and seq_lens_present=True are mutually exclusive: under THD execute(seq_lens=) carries the per-sequence packed "
                    "lengths ([B] int32 lengths, or [B+1] int32 prefix sums with cu_seqlens=True) for the Q and the KV side alike, and there "
                    "is no per-batch KV padding mask (the SDPA adapter sets its own seq_kv_lens_present under THD); pass seq_lens_present=False"
                )
            if quant is not None and (self.fuse_gate or self.fuse_norm_rope):
                raise NotImplementedError(
                    f"the fully fused {_family} pipeline (fuse_norm_rope + fuse_gate) is dense-only: its quantized gated SDPA specialization "
                    "has no THD gate descriptor (the SDPA's THD gate descriptor is the f16/bf16 d256 prefill's), and the fused MXFP8 projection "
                    "fork decodes (b, s_tile) once per 128-row GEMM tile, which a packed tile may straddle; under thd=True run the UNFUSED "
                    "quantized pipeline (fuse_norm_rope=False, fuse_gate=False)"
                )
            if self.num_sequences is None or self.max_seq_len is None:
                raise ValueError(
                    "thd=True needs num_sequences (B: the length tensor has B entries, or B+1 prefix sums under cu_seqlens=True) and max_seq_len "
                    "(S_max, the longest sequence the plan admits): the SDPA's unit grid, metadata and the backward's kv-blocked workspace are "
                    "sized from them at build time"
                )
            _t = self.batch * self.seq_len
            if not (self.num_sequences >= 1 and 2 <= self.max_seq_len <= _t and self.num_sequences * self.max_seq_len >= _t):
                # The product bound is THE guard against a silently truncated chain: the SDPA's packed capacity is
                # min(num_sequences * max_seq_len, T), so a smaller product processes only the first B * S_max tokens --
                # units past the plan envelope never run, their O / LSE rows stay unwritten -- with no message anywhere.
                raise ValueError(
                    f"thd=True: need num_sequences >= 1, 2 <= max_seq_len <= T and num_sequences * max_seq_len >= T (every length is <= "
                    "max_seq_len and the lengths sum to T; S_q = 1 is the DECODE form -- a dense [B, S_q <= 4] declaration with kv_source='paged' -- out "
                    "of the prefill bodies' scope; a smaller product would cap the "
                    f"SDPA backward's packed capacity below T); got num_sequences={self.num_sequences}, max_seq_len={self.max_seq_len}, T={_t}"
                )
        # BLOCK-SPARSE ATTENTION (geometry.qsa): the sparse path's declaration-time declines, typed, in one place --
        # the geometry's own rows (ValueError) and then every pipeline knob the sparse core has no arm for
        # (NotImplementedError naming the feature) -- so a sparse request hears about its feature, never a knob.
        self.qsa: Optional[QsaSpec] = geometry.qsa
        if self.qsa is not None:
            self._check_qsa_declaration(quant, kv_source, kv_cache_dtype=kv_cache_dtype, page_size=paged_kv_page_size)
        # PAGED-CACHE WRITE-THROUGH (serving): the attribute's own rows (type, the multiple-of-16 contract) and the
        # pipelines the stage has no arm for, typed and naming the feature -- device-free, so a test pins every row anywhere.
        self.paged_kv_page_size = self._check_cache_write_declaration(paged_kv_page_size, quant)
        # THE e4m3 KV CACHE (kv_cache_dtype): the attribute's own rows (type, e5m2 by name, the write-through it belongs to, bf16
        # activations), typed and naming the feature -- device-free; the pools' samples below are checked in THIS dtype.
        self.kv_cache_dtype = self._check_kv_cache_dtype_declaration(kv_cache_dtype, quant, kv_source)
        self.kv_cache_e4m3 = self.kv_cache_dtype == KV_CACHE_E4M3
        # THE EXTEND FORM (kv_source) and the BOUNDED token form (token_capacity): the typed declines of the table every caller's
        # dispatch predicate can read back (serving_envelope), then the three samples' contract -- device-free, before any descriptor.
        self.decode_route_policy = _resolve_decode_route_policy(decode_route_policy)
        self.decode_q_tile_max = _resolve_decode_q_tile_max(decode_q_tile_max)
        self.kv_source, self.token_capacity, self.decode_route, self.max_kv_len, self.decode_split_kv = self._check_serving_declaration(
            kv_source,
            token_capacity,
            quant,
            sample_k_cache,
            sample_v_cache,
            sample_block_table,
            sample_h.device,
            max_kv_len,
            decode_split_kv,
            decode_route_policy=decode_route_policy,
            decode_q_tile_max=decode_q_tile_max,
            sparse_tile_form=sparse_tile_form,
            sparse_sub_tiles_per_cta=sparse_sub_tiles_per_cta,
        )
        # the dense / packed QsaSpec forms (kv_source="self"): the tcgen05 core; an explicit "lean" there is the lean body's own decline
        self._resolve_sparse_tile_form_off_decode()
        if self.kv_source == KV_SOURCE_PAGED and not self.thd:
            # The DECODE form: over the pools the per-sequence KV length (kv_lens) is mandatory, so the declaration reads
            # seq_lens_present=True whatever was passed (the dense padding mask it names IS the per-sequence KV length -- bound to
            # the KV side only here).
            self.seq_lens_present = True
        # PDL BY DEFAULT (2026-10-19): the AUTO reading of `use_pdl` resolves here, once the form is known and before any stage is built --
        # True on the dense DECODE form's decode-tile routes for the bf16 / f16 block on the parts of DECODE_PDL_DEFAULT_CCS (its docstring:
        # the measured default), False everywhere else; an explicit value was kept above.  A device-side fact read once (the sample's
        # device), like the stages' own arch gates; a CPU sample (a host-side declaration) reads False.
        if use_pdl is None:
            self.use_pdl = self._pdl_default(quant)
        # C0 (2026-10-15): at DECODE ROW COUNTS the declaration `fuse_norm_rope=True` LOWERS stage (1) to the unfused chain.  The 256-wide
        # fused fork pads a decode step's handful of rows to one 256-row token tile and streams the slab weight from a quarter of the SMs
        # (the measured step at B 1 x KV 4K: the fork ~25 us against the small-M route's slab GEMM + norm / RoPE + cache write ~12 us of
        # kernels), so the fewer launches lose; the unfused chain is what the route measured as the faster decode step.  The lowering is a
        # declaration fact (`fuse_norm_rope_effective`), decided here from the dense DECODE form's facts -- the dense paged route
        # (`decode_route`), bf16 / f16 (no quant), no indexer band, 16-bit pools, T = B x S_q within the small-M route's bound -- and kept
        # OFF by `fuse_norm_rope_small_m=True` (the fused fork as declared: the A/B's control arm) and by an explicit `fuse_cache_write=True`
        # (the RAW argument, read before its own validator runs below: the write-through arm lives in the fork, so the request outranks the
        # lowering -- user request > heuristic proposal; the contradiction `fuse_norm_rope_small_m=False` + `fuse_cache_write=True` is that
        # validator's typed decline).  The EXTEND form, the packed form and the quantized pipelines are untouched (`fuse_norm_rope` runs as
        # declared there).
        if fuse_norm_rope_small_m is not None and not isinstance(fuse_norm_rope_small_m, bool):
            raise TypeError(f"fuse_norm_rope_small_m must be None (auto), True or False, got {type(fuse_norm_rope_small_m).__name__}")
        self.fuse_norm_rope_small_m: Optional[bool] = fuse_norm_rope_small_m
        if fuse_norm_rope_small_m is not None and (self.decode_route is None or not self.fuse_norm_rope):
            raise ValueError(
                f"fuse_norm_rope_small_m={fuse_norm_rope_small_m}: the knob decides whether the FUSED norm + RoPE projection runs at decode row counts on "
                f"the dense DECODE form (kv_source='paged' on a dense declaration with fuse_norm_rope=True); got kv_source={kv_source!r}, "
                f"thd={self.thd}, fuse_norm_rope={self.fuse_norm_rope}, qsa={'set' if geometry.qsa is not None else None}"
            )
        from .kernels.proj_gemm import SMALL_M_MAX

        self.small_m_decode = self.decode_route is not None and self.batch * self.seq_len <= SMALL_M_MAX
        lowering_applies = (
            self.small_m_decode and quant is None and not geometry.index_band and not self.kv_cache_e4m3 and self.dtype in (torch.bfloat16, torch.float16)
        )
        # the knob's three readings where the lowering applies: None = AUTO, the lowering -- unless an explicit fuse_cache_write=True asks for
        # the fork's write-through arm (the request outranks the proposal); True = the fork as declared; False = the lowering by name (with
        # fuse_cache_write=True the contradiction the write-through validator declines below)
        self.fuse_norm_rope_effective = self.fuse_norm_rope and not (
            lowering_applies and (self.fuse_norm_rope_small_m is False or (self.fuse_norm_rope_small_m is None and fuse_cache_write is not True))
        )
        # The BLOCK-SPARSE decode / extend-with-prefix mode (a QsaSpec geometry over the pools; the packed form met the table's
        # `sparse` row above): stage (4) is the sparse core reading the pools, and the LIST FORM is a declaration fact --
        # `QsaSpec.index_list_form` as declared, or (None) derived from S_q under caller lists: ONE list per sequence at S_q <= the
        # record's shared-list cap (the MTP form, anchored at the step-0 position), one list per row past it; under the in-block indexer
        # over the pools the rows select for themselves (per row at every S_q: the serving stacks' function).  `_check_qsa_declaration`
        # declined the forms the core does not serve ("shared" past the cap, "shared" under the indexer); the cap is read off the record.
        self.sparse_decode = self.kv_source == KV_SOURCE_PAGED and self.qsa is not None
        self.indexer_decode = self.sparse_decode and self.qsa.index_source == "indexer"
        # the indexer CACHE dtype over the pools (QsaSpec.index_cache_dtype; None = the activation dtype): the compressed-key cache sample / tensor
        # and the ix_q slot are in it (e4m3 = the e4m3 indexer cache, served 2026-10-21); the ring stays the activation dtype
        self.index_cache_dtype = (self.qsa.index_cache_dtype if self.qsa.index_cache_dtype is not None else self.act_dtype) if self.indexer_decode else None
        self.qsa_shared_list = self.sparse_decode and self._qsa_shared_list_derived()
        self.qsa_list_form = (QSA_INDEX_LIST_FORM_SHARED if self.qsa_shared_list else QSA_INDEX_LIST_FORM_PER_ROW) if self.sparse_decode else None
        # The e4m3 KV cache OVER THE POOLS: stage (4) is the gathered cast + the dense decode tile (the block-sparse decode mode at
        # S_q <= 4 only); every other paged form over e4m3 pools, the LSE and a split are typed declines, and the mode norms +
        # rotates OUT of place (the tile's per-item Q is a view of the compact Q only), so inplace_qkv / fuse_norm_rope are settled here.
        self.upcast_decode = self.sparse_decode and self.kv_cache_e4m3
        if self.kv_cache_e4m3 and self.kv_source == KV_SOURCE_PAGED:
            self.decode_split_kv = self._check_kv_cache_dtype_serving(decode_split_kv, inplace_qkv, fuse_norm_rope)
        # Where the sigmoid gate runs: the SDPA epilogue (fuse_gate on the prefill forms), the split COMBINE of the decode tile
        # (fuse_gate on the DECODE form with split >= 2: the tile has no gate seams), or its own launch, stage (5), everywhere
        # else (an unsplit decode tile).  `fuse_gate` stays what was declared; `gate_fused` is what runs.
        self.gate_fused = (
            self.fuse_gate
            and not self.upcast_decode
            and (self.decode_route is None or self.decode_route.prefill_tile or (self.decode_route.decode_tile and self.decode_split_kv >= 2))
        )
        # Where the paged KV-cache write-through runs: the fused projection's epilogue (the arm serves the bf16 / f16 fused norm + RoPE
        # pipeline without an indexer band) or the separate cache-write launch.  `fuse_cache_write` stays what was declared;
        # `cache_write_fused` is what runs.  Never under kv_cache_dtype=e4m3: the epilogue writes 16-bit rows and the e4m3 write-through
        # QUANTIZES (the separate _QuantizedCacheWrite launches); AUTO stays off there, an explicit True is the typed decline below.
        self.fuse_cache_write = self._check_fuse_cache_write_declaration(fuse_cache_write, quant)
        self.cache_write_fused = bool(self.paged_kv_page_size) and (
            self.fuse_cache_write is True
            or (self.fuse_cache_write is None and self.fuse_norm_rope_effective and quant is None and not geometry.index_band and not self.kv_cache_e4m3)
        )
        # The IN-BLOCK INDEXER's DECODE STEP over the pools (QsaSpec(index_source="indexer") with kv_source="paged", a dense declaration;
        # `indexer_decode` was settled with the list form above): the three side-cache samples and the three input-contract flags --
        # REQUIRED there, REFUSED everywhere else -- their FORM only (the kernels bind the strides symbolically).
        self.index_state_slot_mapping_present, self.index_commit_present, self.rope_position_tables_only = self._check_indexer_decode_declaration(
            sample_index_kv_cache,
            sample_index_state_cache,
            sample_index_state_block_table,
            index_state_slot_mapping_present,
            index_commit_present,
            rope_position_tables_only,
            sample_h.device,
        )
        # The PREPARE fusion (stages (2)+(3) + (4w) in one launch): a performance knob over the write-through declaration;
        # its own rows (type, the attribute it needs, the pipelines its kernel has an arm for), typed and naming the knob.  None = the
        # MEASURED default (`_fuse_prepare_default`: the decode indexer's form on the parts that measured it; False everywhere else).
        self.fuse_prepare_requested: Optional[bool] = fuse_prepare
        self.fuse_prepare = self._check_fuse_prepare_declaration(fuse_prepare)
        if self.fuse_prepare and self.kv_cache_e4m3:
            raise NotImplementedError(
                "fuse_prepare=True with kv_cache_dtype=torch.float8_e4m3fn: the prepare kernel writes the activation dtype's K / V rows into the pools; "
                "the e4m3 cache's write-through quantizes in its own stage (two per-tensor casts + the bit copy) -- pass fuse_prepare=False"
            )
        if self.rope_position_tables_only and not self.fuse_prepare:
            # the position-table arm is the fused launch's: the unfused twin's pool-1 query norm reads per-token rows
            raise NotImplementedError(_qsa_decode_decline_texts(geometry, self.act_dtype)["fuse_prepare_rope_tables_unfused"])
        self._descs = {
            "w_qkvg": self._make_tensor_desc(sample_w_qkvg, name="w_qkvg"),
            "w_q_norm": self._make_tensor_desc(sample_w_q_norm, name="w_q_norm"),
            "w_k_norm": self._make_tensor_desc(sample_w_k_norm, name="w_k_norm"),
            "cos": self._make_tensor_desc(sample_cos, name="cos"),
            "sin": self._make_tensor_desc(sample_sin, name="sin"),
            "w_o": self._make_tensor_desc(sample_w_o, name="w_o"),
            "out": self._make_tensor_desc(sample_out, name="out"),
            # MXFP8 only (None otherwise): the caller's F8_128x4 blobs, validated in check_support.
            "h_sf": self._make_tensor_desc(sample_h_sf, name="h_sf"),
            "w_qkvg_sf": self._make_tensor_desc(sample_w_qkvg_sf, name="w_qkvg_sf"),
            # fp4 O only (None otherwise): the e2m1 W_o's F8_128x4 blob, validated in check_support.
            "w_o_sf": self._make_tensor_desc(sample_w_o_sf, name="w_o_sf"),
            # kv_source="paged" only (None otherwise): the pools' and the block table's shape / dtype / strides, validated above;
            # the SDPA stage compiles the paged THD leg against them and execute matches the real tensors to them.
            "k_cache": self._make_tensor_desc(sample_k_cache, name="k_cache"),
            "v_cache": self._make_tensor_desc(sample_v_cache, name="v_cache"),
            "block_table": self._make_tensor_desc(sample_block_table, name="block_table"),
            # the in-block indexer's DECODE step only (None otherwise): the side caches' FORM (dims / dtype; the strides are never matched --
            # the kernels bind them symbolically) and the ring block table's; validated above, re-read by the execute checks
            "index_kv_cache": self._make_tensor_desc(sample_index_kv_cache, name="index_kv_cache"),
            "index_state_cache": self._make_tensor_desc(sample_index_state_cache, name="index_state_cache"),
            "index_state_block_table": self._make_tensor_desc(sample_index_state_block_table, name="index_state_block_table"),
        }

        fp8 = quant is not None  # fp8-CLASS pipeline (per-tensor FP8 or MXFP8): e4m3 h / weights, bf16 activations
        mxfp8 = self.mxfp8
        # FULLY FUSED: both knobs under a quant spec (the both-or-neither decline
        # above makes `fp8 and fuse_gate` == `fp8 and fuse_norm_rope`).
        # `fp8_fused` names the per-tensor pipeline (test-visible, unchanged);
        # `mxfp8_fused` the block-scale one; `quant_fused` either.
        self.fp8_fused = fp8 and not mxfp8 and self.fuse_gate and self.fuse_norm_rope
        self.mxfp8_fused = mxfp8 and self.fuse_gate and self.fuse_norm_rope
        self.quant_fused = self.fp8_fused or self.mxfp8_fused
        act = self.act_dtype
        self._quant_dev = None  # the QuantSpec / MxQuantSpec as device scalars: ALLOCATED in compile(), written on a launch stream
        # compile() only allocates those scalars and enqueues nothing on the stream ambient then; the FIRST execute writes their
        # VALUES on its own launch stream (_write_quant_dev), as does every update_quant_scales() on the caller's.  A compile-time
        # fill would have no ordering against a cross-stream recalibration: pending work ahead of it on the compile stream would let
        # it land AFTER the update and silently restore the declared scales for every later execute.  Reset at compile().
        self._quant_dev_on_launch_stream = False
        # The allocation itself is stream work under torch.use_deterministic_algorithms(True) (torch.empty enqueues a NaN fill on
        # the ambient stream): compile() records an event behind it and every writer waits on that event first.
        self._quant_dev_ready = None
        # The DECODE form's prefill route: the per-sequence Q lengths of the paged THD leg -- the constant S_q for every sequence of the
        # dense step -- are NOT plan-owned memory (Rule 8 recipes R2 + R4): a [B] int32 slot of the caller's workspace
        # (_Intermediates.decode_q_lens, reserved by _plan_workspace on that route only) that EVERY execute fills on its launch stream
        # (_fill_decode_q_lens) -- no first-execute flag, so an eager execute after a captured first execute reads its own fill.
        # rstd exists only where a norm exists: under qk_norm=False a training
        # block saves lse / q_pre / k_pre but no rstd (SavedForBackward.rstd_*
        # are None -- required, both directions, at execute).
        want_rstd = self.save_for_backward and geometry.qk_norm
        if self.fuse_norm_rope_effective:
            # bf16: writes the [T, N] slab.  FP8: the second rendering writes the
            # compact e4m3 q8/k8/v8 + the bf16 gate16 buffer, quantizing in its
            # epilogue.  MXFP8: the block-scale twin (typed decline until S6 lands).
            self._proj = _FusedQkvProjection(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=self.dtype,
                want_rstd=want_rstd,
                quant=quant,
                device=self.device,
                # the serving write-through folded into the epilogue (cache_write_fused): the pools' page size, else 0 = the fork as before
                kv_pool_page_size=self.paged_kv_page_size if self.cache_write_fused else 0,
            )
            self._norm_rope = None  # lives in the fused epilogue
        else:
            # FP8: e4m3 x e4m3 -> fp32 -> * (descale_h * descale_w_qkvg) -> bf16 slab.
            # MXFP8: e4m3 x e4m3 with the E8M0 block scales dequantized IN the MMA -> bf16 slab (no alpha).
            # MXFP8 with an e2m1 W_qkvg (MxQuantSpec.w_qkvg_dtype): the same GEMM on the catalog's
            # MIXED row (e4m3 h x e2m1 W, E8M0 / 32) -- same stage list, same workspace (config row 7).
            self._proj = _qkv_gate_projection(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=self.dtype,
                out_dtype=act,
                alpha=fp8 and not mxfp8,
                block_scale=mxfp8,
                w_dtype=quant.w_qkvg_dtype if mxfp8 else None,
            )
            self._norm_rope = _QkNormRope(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, want_rstd=want_rstd, use_pdl=self.use_pdl)
        # (4w) on the unfused chain at decode row counts: the paged write-through folded into the norm + RoPE launch (the constructor's
        # `fuse_cache_write_small_m` note).  Decided here, after the norm + RoPE stage exists (its TMA tiling must fit the geometry) and
        # before the separate cache-write stage is built (not built when this arm runs: a stage that never runs is not in `_stages`).
        self.fuse_cache_write_small_m: Optional[bool] = fuse_cache_write_small_m
        self.cache_write_in_norm_rope: bool = self._resolve_cache_write_small_m(fuse_cache_write_small_m, quant)
        if self.cache_write_in_norm_rope:
            self._norm_rope.page_size = self.paged_kv_page_size
        # Stage (3b) exists ONLY to give the SDPA a compact V. In-place needs no
        # such thing, so the stage is not built at all rather than built and
        # skipped -- a stage that is never run should not be in `_stages`,
        # where it would still be compiled and still report support.  Under
        # FP8 / MXFP8 the quantize stages compact V (and Q, K) on the way to e4m3.
        # The e4m3 decode mode norms Q / K out of place (the compact slots) but never compacts V: the write-through reads the slab's
        # V band and stage (4) reads the pools, so the stage is not built there either.
        self._compact_v = None if (self.inplace_qkv or fp8 or self.upcast_decode) else _VCompaction(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act)
        # (3q) UNFUSED FP8 only: bf16 slab slices -> compact e4m3 Q/K/V.  Two
        # recipes (h_q and h_kv); K and V share the h_kv one.  Fully fused FP8
        # quantizes in the projection fork's epilogue and builds none of them.
        # (3q) UNFUSED MXFP8: THREE block-quantize stages -- Q and K ROWWISE
        # (two recipes, h_q / h_kv), V COLUMNWISE (its own recipe: a different
        # kernel arm AND a different SF byte order) -- each writing compact e4m3
        # + the SDPA's F8_128x4 SF blob.
        quantize = fp8 and not self.quant_fused
        self._quant_q = self._quant_kv = self._quant_k = self._quant_v = None
        if quantize and not mxfp8:
            self._quant_q = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, name="quantize_q")
            self._quant_kv = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_kv, name="quantize_kv")
        elif quantize:
            # THD: the PACKED arm -- the per-sequence-tile-padded SF layout the MXFP8 row's THD arm reads, at the slot capacity
            # (num_sequences is the packing's B; the lengths ride execute(seq_lens=) into each of the three launches).
            _mxq = lambda heads, axis, name: _QuantizeMxfp8(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype_in=act,
                heads=heads,
                axis=axis,
                name=name,
                packed=self.thd,
                num_sequences=self.num_sequences if self.thd else 0,
                cu_seqlens=self.cu_seqlens,
            )  # noqa: E731
            self._quant_q = _mxq(geometry.h_q, "row", "quantize_mxfp8_q")
            self._quant_k = _mxq(geometry.h_kv, "row", "quantize_mxfp8_k")
            self._quant_v = _mxq(geometry.h_kv, "col", "quantize_mxfp8_v")
        if self.quant_fused:
            # Q/K/V are the COMPACT e4m3 q8/k8/v8 the projection fork writes
            # (token_stride 0 -- what the unfused quantized SDPA reads too), the
            # GATE is the compact bf16 gate16, O comes out e4m3 (o8).  The gated
            # FP8 / MXFP8 SDPA specialization (epilogue_gate) is compiled at
            # compact strides.
            # fp4 O (row 10): the gated MXFP8 SDPA writes bf16 O (the adapter serves a bf16 O for fp8-class input --
            # no SDPA row changes) and the quantize_fp4 stage turns it into the fp4 out_proj's A operand.
            sdpa_token_stride, sdpa_gate_token_stride = 0, geometry.h_q * geometry.d_head
            sdpa_o_dtype = torch.float8_e4m3fn if self.o_fp4 is None else act
        else:
            # bf16: in-place reads the slab at its padded stride; FP8 / MXFP8
            # unfused read the compact e4m3 buffers.  GATE is always a slab column slice.
            sdpa_token_stride, sdpa_gate_token_stride, sdpa_o_dtype = (geometry.n_qkvg if self.inplace_qkv else 0) if not fp8 else 0, geometry.n_qkvg, act
        if self.upcast_decode:
            # Stage (4) under QsaSpec over e4m3 POOLS: the gathered cast + the bf16 d256 decode tile over the scratch (the class
            # docstring); the compact normed Q (out of place) is its Q.
            self._sdpa = _UpcastSparseSdpa(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=self.dtype,
                device=self.device,
                page_size=self.paged_kv_page_size,
                k_cache_desc=self._descs["k_cache"],
                v_cache_desc=self._descs["v_cache"],
                block_table_desc=self._descs["block_table"],
            )
        elif self.qsa is not None:
            # Stage (4) under QsaSpec: the index-list sparse core -- built or not built per declaration like every other
            # stage.  Reads the slab (or the compact buffers) at the SAME strides the dense stage would; declines, typed,
            # until the core lands (its own docstring).
            self._sdpa = _SparseSdpa(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=self.dtype,
                device=self.device,
                want_lse=self.return_lse,
                seq_lens_present=self.seq_lens_present,
                token_stride=sdpa_token_stride,
                gate_token_stride=sdpa_gate_token_stride,
                # THD: the packed form (B = 1, S = T), the sequence count and the lengths' form -- the adapter's THD arm.
                thd=self.thd,
                num_sequences=self.num_sequences,
                max_seq_len=self.max_seq_len,
                cu_seqlens=self.cu_seqlens,
                fuse_gate=self.fuse_gate,
                # The DECODE / extend-with-prefix mode over the pools: the pools' and the table's descriptors as declared, the
                # page size, the list form derived from S_q, the split the declaration settled on (the defaults otherwise).
                kv_source=self.kv_source,
                paged_page_size=self.paged_kv_page_size if self.sparse_decode else 0,
                k_cache_desc=self._descs["k_cache"] if self.sparse_decode else None,
                v_cache_desc=self._descs["v_cache"] if self.sparse_decode else None,
                block_table_desc=self._descs["block_table"] if self.sparse_decode else None,
                list_per_sequence=self.qsa_shared_list,
                split_kv=self.decode_split_kv if self.sparse_decode else 1,
                # the switch reaches the sparse core and its combine on every arm the adapter serves it on (THD declined above by name)
                use_pdl=self.use_pdl,
                # the BODY the declaration resolved (never "auto" here: one decision point) and the lean tile's k pin
                tile_form=self.sparse_tile_form,
                sub_tiles_per_cta=self.sparse_sub_tiles_per_cta,
            )
        else:
            # The DECODE form's PREFILL route: stage (4) is the EXTEND form's binding -- the adapter's paged THD leg with the group packed
            # -- at the dense step's own envelope (num_sequences = B, max_seq_len = S_q, [B] int32 lengths all S_q: a workspace slot the
            # block fills on the launch stream at every execute, _fill_decode_q_lens), unsplit, the gate in the body's epilogue.
            prefill_route = self.decode_route is not None and self.decode_route.prefill_tile
            self._sdpa = _Sdpa(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=self.dtype,
                device=self.device,
                want_lse=self.return_lse,
                seq_lens_present=self.seq_lens_present and not prefill_route,
                token_stride=sdpa_token_stride,
                fuse_gate=self.gate_fused,  # the SDPA gates (epilogue, or the decode tile's split combine over the pools); else stage (5)
                gate_token_stride=sdpa_gate_token_stride,
                o_dtype=sdpa_o_dtype,
                gate_dtype=act,  # gate16 / the slab's GATE columns are the activation dtype (bf16 under FP8 / MXFP8)
                mxfp8=mxfp8,  # pertensor_fp8=False -> the production block-scale kernel; NATURAL read off the MXFP8 row (D5)
                # THD: the (num_sequences, max_seq_len) envelope, the packed total T = batch * seq_len, the lengths' form.
                thd=self.thd or prefill_route,
                num_sequences=self.batch if prefill_route else self.num_sequences,
                max_seq_len=self.seq_len if prefill_route else self.max_seq_len,
                cu_seqlens=False if prefill_route else self.cu_seqlens,
                # The EXTEND form: K / V from the pools (their descriptors as declared), the block table's strides, the page size.
                kv_source=self.kv_source,
                paged_page_size=self.paged_kv_page_size if self.kv_source == KV_SOURCE_PAGED else 0,
                k_cache_desc=self._descs["k_cache"],
                v_cache_desc=self._descs["v_cache"],
                block_table_desc=self._descs["block_table"],
                # The DECODE form: the KV bound and the split the form lowers with (the defaults otherwise), the route read under the policy.
                max_kv_len=self.max_kv_len if self.decode_route is not None else 0,
                decode_split_kv=self.decode_split_kv if self.decode_route is not None else 1,
                route=self.decode_route,
                # the switch reaches stage (4) on a decode-tile route only (_Sdpa.pdl_capable): the prefill tiles carry no sites and
                # launch plain -- the link serializes, the chain stays correct; `pdl_stages` reads it back
                use_pdl=self.use_pdl and _Sdpa.pdl_capable(self.decode_route),
            )
        # Stage (5) lives in the SDPA kernel's gate epilogue under fuse_gate --
        # not built rather than built and skipped (it would still compile).
        # gate_fused, not fuse_gate: on the DECODE form a declared fuse_gate runs as this launch when the tile runs unsplit.
        self._gate = None if self.gate_fused else _SigmoidGate(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, use_pdl=self.use_pdl)
        # (4w) SERVING write-through of the post-RoPE K / V (+ the raw indexer key) into paged pools -- built only when
        # the declaration carries a page size; not built rather than built and skipped (it would still compile).
        # Not built when the fused projection's epilogue writes the pools (cache_write_fused): a stage that never runs is not in `_stages`.
        # Under kv_cache_dtype=e4m3 the write-through QUANTIZES (two per-tensor casts + the bit copy through the pools' bf16 views).
        # Nor when the norm + RoPE launch writes them (cache_write_in_norm_rope).
        if not self.paged_kv_page_size or self.cache_write_fused or self.cache_write_in_norm_rope:
            self._cache_write = None
        elif self.kv_cache_e4m3:
            self._cache_write = _QuantizedCacheWrite(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, page_size=self.paged_kv_page_size)
        else:
            # under the decode indexer over the pools the raw key enters the RING (the prepare / the step form), so the band's raw-key
            # pool arm is not built there (index_k_raw refused by name)
            self._cache_write = _CacheWrite(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=act,
                page_size=self.paged_kv_page_size,
                use_pdl=self.use_pdl,
                index_raw_pool=not self.indexer_decode,
            )
        # (2+3+4w) FUSED under fuse_prepare: ONE launch norms + rotates Q / K in place and writes the post-RoPE K, the V (and the
        # raw indexer key) into the pools from the same registers -- bitwise the two stages it replaces, which are then not
        # built (a stage that never runs is not in `_stages`).  Under the decode indexer over the pools the launch carries the three
        # PAGED indexer arms (the query norm into the workspace, the raw key into the ring, the completing block into the paged
        # compressed-key cache) in place of the raw-key pool arm: `_IndexerDecode` then launches the scorer and the top-k alone.
        self._prepare = None
        if self.fuse_prepare:
            arms = None
            if self.indexer_decode:
                # the record's home is the DSL-free contract module: a DECLARATION never imports the kernel module (Rule 7 -- the stage's
                # check_support gates the DSL floor before the kernel is loaded; test_import_boundaries pins this declaration)
                from .cache_write_contract import QsaPrepareIndexArms

                q = self.qsa
                arms = QsaPrepareIndexArms(
                    heads=q.index_heads,
                    head_dim=q.index_head_dim,
                    raw_pool=False,
                    query=True,
                    compress=True,
                    n_commit=False,  # the two-phase commit is a typed decline (index_commit_present)
                    pool=q.block_size,
                    paged=True,
                    ring_slots=self.index_state_slot_mapping_present,
                    rope_table=self.rope_position_tables_only,
                )
            self._prepare = _QsaPrepare(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=act,
                page_size=self.paged_kv_page_size,
                use_pdl=self.use_pdl,
                index_arms=arms,
                index_cache_dtype=self.index_cache_dtype if self.indexer_decode else None,
            )
            self._norm_rope = None
            self._cache_write = None
        # (4i) the IN-BLOCK INDEXER -- built only under QsaSpec(index_source="indexer"): on the prefill form (the block's own K / V) the
        # band's queries normed + rotated, the raw key compressed per block, the scorer's top-k -> the list stage (4) consumes (its
        # workspace slots exist only when the scorer runs -- S past the identity bound; below it the stage hands over a plan-time identity
        # list); over the POOLS (the decode step) the paged scorer + the decode top-k over the stack's side caches, per row.
        self._indexer = (
            _Indexer(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, device=self.device, seq_lens_present=self.seq_lens_present)
            if (self.qsa is not None and self.qsa.index_source == "indexer" and not self.indexer_decode)
            else None
        )
        self._indexer_decode = (
            _IndexerDecode(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=act,
                device=self.device,
                page_size=self.paged_kv_page_size,
                n_blocks_max=int(self._descs["block_table"].shape[1]) * (self.paged_kv_page_size // self.qsa.block_size),
                fused=self.fuse_prepare,
                ring_slots=self.index_state_slot_mapping_present,
                rope_table=self.rope_position_tables_only,
                use_pdl=self.use_pdl,
            )
            if self.indexer_decode
            else None
        )
        # (3g) TRAINING, gate-copy save mode only: ONE strided copy of the slab's
        # GATE band into the compact caller `saved.gate` (the elementwise kernel's
        # has_gate=False arm at h_q heads; the same artifact serves an optional
        # q_pre copy, k_pre rides `_compact_v`'s h_kv recipe).  Not built otherwise
        # -- a stage that never runs is not in `_stages`.
        self._gate_copy = _BandCopy(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act) if (self.save_for_backward and self.saved_gate_copy) else None
        # (3g, K) the k_pre copy rides `_compact_v`'s h_kv recipe on the bf16 pipeline; the quantized pipelines build no
        # `_compact_v` (their quantize stages compact V), so a quantized gate-copy block gets an h_kv band copy of its own.
        # Runs only when the record carries a k_pre buffer (like the q_pre copy above); None wherever `_compact_v` serves.
        self._kpre_copy = (
            _BandCopy(geometry, batch=self.batch, seq_len=self.seq_len, dtype=act, heads=geometry.h_kv, name="k_pre_compaction")
            if (self._gate_copy is not None and self._compact_v is None)
            else None
        )
        # (5q) UNFUSED FP8 only: bf16 gated O -> compact e4m3 for the out
        # projection (same [T, H_q, D] shape as Q, so the Q recipe serves it).
        # UNFUSED MXFP8: an EXPLICIT per-tensor recipe (D1) -- never the rowwise
        # MX recipe, whose SF blob nothing downstream would read.
        # (5q') fp4 O (MXFP8, unfused OR fused): ONE block-quantize launch (bf16 gated O -> e2m1 codes + the
        # out_proj GEMM's padded F8_128x4 blob) replaces the per-tensor quantize_o; the out projection becomes
        # the fp4 x fp4 block-scale GEMM (no alpha: both sides dequantize through their blobs in the MMA).
        quant_o_distinct = mxfp8 and (quantize or self.o_fp4 is not None)
        if self.o_fp4 is not None:
            self._quant_o = _QuantizeFp4(
                geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, fmt=self.o_fp4, name="quantize_fp4_o"
            )
            self._out_proj = _out_projection(
                geometry,
                batch=self.batch,
                seq_len=self.seq_len,
                dtype=_FP4_X2,
                out_dtype=act,
                alpha=False,
                block_scale=True,
                w_dtype=_FP4_X2,
                block_size=self.o_fp4.block_size,
                sf_dtype=self.o_fp4.sf_cudnn_dtype,
            )
        else:
            if quantize and mxfp8:
                self._quant_o = _Quantize(geometry, batch=self.batch, seq_len=self.seq_len, dtype_in=act, heads=geometry.h_q, name="quantize_o")
            else:
                self._quant_o = self._quant_q
            self._out_proj = _out_projection(geometry, batch=self.batch, seq_len=self.seq_len, dtype=self.dtype, out_dtype=act, alpha=fp8)
        # The SERVING declaration (kv_source="paged") is captured by a serving stack with vLLM's defaults -- the FIRST execute
        # of a fresh declaration inside a CUDA-graph capture on a new stream, no eager warm-up -- so both dense projection
        # stages are declared capture-safe here.  The flag is consumed by the projection GEMM driver's capture-safe route
        # (build_proj_gemm(capture_safe=True): the JIT-only route by construction, never a ranked list, no cuDNN handle at the
        # first execute; a shape the driver resolves no JIT tile for is a typed NotImplementedError at compile()), which
        # `_Projection.compile` forwards it to when the driver takes the kwarg (the CUDA-graph capture-refusal change of
        # kernels/proj_gemm.py); on a tree without that route the flag is recorded on the stage and the projections take the
        # driver's existing route (the serving suite's capture cell skips there).  The fused norm + RoPE projection builds no
        # build_proj_gemm plan, so the flag is set on the two dense projection stages only.
        # Per stage, iff the driver's "auto" resolver names a JIT tile for the stage's OWN (M, N, K) (`_Projection.resolves_a_jit_tile`):
        # the forced 256-wide tile where 256 divides N (every dense served geometry: n_qkvg, d_model and h_q * d_head), or the small-M
        # route's tile at T = B x S_q <= 128 -- which is how the indexer band's slab (QsaSpec.index_band: n_qkvg = 13312 + 640 = 13952,
        # off the 256 grid) is capture-safe at decode row counts: its 218 64-row tiles overflow a wave and the route takes the 128-row
        # swap-AB tile that divides it exactly (109 CTAs), the same JIT-only route as the four-band slab, no opt-in flag at compile().
        # A band declaration past 128 rows resolves no tile and keeps the driver's own route -- its steps run eagerly and a capture is
        # the driver's typed refusal at the first captured execute (the band's prefill / extend forms' documented limit); setting the
        # flag there would decline the whole block at compile.  Read off the resolver `build_proj_gemm` itself runs, never re-spelled,
        # so this predicate cannot drift from the route -- and read AGAIN at compile() on the compiling device
        # (`_flag_capture_safe_projections`): the SM count the small-M route reads is a device property, so a declaration and its
        # compile on different parts would otherwise carry a stale flag into the driver.
        self._flag_capture_safe_projections()
        # Stage order == pipeline order.  `_quant_o` is listed only where it is a
        # DISTINCT stage (MXFP8: quantize_o, or quantize_fp4_o under o_fp4); under
        # FP8 it aliases `_quant_q`, listed once.
        self._stages = tuple(
            st
            for st in (
                self._proj,
                self._norm_rope,
                self._prepare,  # (2+3+4w) in the (2+3) slot under fuse_prepare; `_norm_rope` / `_cache_write` are None then
                self._gate_copy,
                self._kpre_copy,
                self._compact_v,
                self._quant_q,
                self._quant_kv,
                self._quant_k,
                self._quant_v,
                self._indexer,
                self._indexer_decode,  # the decode step over the pools, in the indexer's slot (one of the two is built, never both)
                self._cache_write,
                self._sdpa,
                self._gate,
                self._quant_o if quant_o_distinct else None,
                self._out_proj,
            )
            if st is not None
        )
        # The programmatic-dependent-launch readback (the switch is per stage): the stages that launch with the attribute, and whether
        # every switched stage of the chain does (False with a plain link: the prefill tile of the DECODE form's prefill route or of the
        # EXTEND form, or a stage the switch is not threaded to -- the V compaction of inplace_qkv=False, the training copies).  The
        # FROST GEMM stages launch with the attribute by their own constant and are not switched: neither listed nor counted.  `use_pdl`
        # is the RESOLVED switch (None resolved by `_pdl_default`), `use_pdl_requested` the declaration as given.  A stage that launches
        # several kernels names them through `pdl_names` (the decode indexer: the step form, the scorer, the top-k); every other stage is
        # listed under its own name, as before.
        switched = tuple(st for st in self._stages if hasattr(st, "use_pdl"))
        self.pdl_stages: Tuple[str, ...] = tuple(n for st in switched if st.use_pdl for n in getattr(st, "pdl_names", (st.name,)))
        self.use_pdl_effective: bool = self.use_pdl and bool(switched) and all(st.use_pdl for st in switched)
        self._ws = None

    # -- block-sparse attention (QsaSpec) -----------------------------------

    def _check_qsa_declaration(self, quant, kv_source=KV_SOURCE_SELF, kv_cache_dtype=None, page_size=None) -> None:
        """Every sparse request this block cannot serve, declined at DECLARATION with the feature named: the geometry's
        own rows first (``validate()``: block size, top_k range, index source, the indexer band, causal, no window --
        ``ValueError``), then the pipeline knobs the sparse core has no arm for (``NotImplementedError``), in the order
        a caller is most likely to have asked for them.  Device-free, so a test pins every row on any GPU; the arch gate
        is the stages' own (``check_support``).  ``kv_source`` (appended) is the constructor's argument as given: under
        ``"paged"`` the bottom-right diagonal is the DECODE / extend-with-prefix mode's (served through the sparse stage;
        the serving table validates the rest), under ``"self"`` it stays a decline naming that mode.  ``kv_cache_dtype`` /
        ``page_size`` (appended; the raw arguments, validated by their own rows afterwards) feed the in-block indexer's rows over the
        pools (the e4m3 pool, the page's power-of-two row count)."""
        g = self.geom
        g.validate()
        q = g.qsa
        rec = _sparse_record()
        paged = kv_source == KV_SOURCE_PAGED
        if self.thd and not rec.thd:
            # Served since 2026-10-08 (the record claims `thd`; the stage binds the packed operands): this decline stays
            # reachable so a record that drops the arm declines by name instead of launching a body without it.
            raise NotImplementedError(_QSA_THD_DECLINE)
        if self.save_for_backward:
            raise NotImplementedError(
                "QsaSpec with save_for_backward=True: sparse-attention training (the sparse backward and the indexer loss) is out of scope; the "
                "block's backward differentiates the dense record only"
            )
        if quant is not None:
            raise NotImplementedError(
                f"QsaSpec with quant={type(quant).__name__}: the sparse core is bf16 / f16; the quantized pipelines (per-tensor FP8, MXFP8, the "
                "fp4 modes) have no sparse arm"
            )
        if self.fuse_gate and not rec.epilogue_gate:
            # Served while the record claims the arm (it does: the sparse core's epilogue gate, bound through _SparseSdpa);
            # the decline stays reachable for a record that retires it.
            raise NotImplementedError(
                "QsaSpec with fuse_gate=True: the sparse adapter's record declines the epilogue gate; stage (5) runs as its own launch (fuse_gate=False)"
            )
        if self.fuse_norm_rope and q.index_band:
            tile = _FusedQkvProjection._TILE_N
            raise NotImplementedError(
                f"QsaSpec(index_band=True) with fuse_norm_rope=True: the fused projection renders {tile}-column tiles; a {q.index_band_cols}-column "
                f"indexer band is not a whole number of them -- use the unfused projection (fuse_norm_rope=False), or a band padded to "
                f"{-(-q.index_band_cols // tile) * tile} columns once that arm exists"
            )
        if g.causal_bottom_right and not paged:
            # Served over the pools since 2026-10-14 (the block-sparse DECODE / extend-with-prefix mode: kv_source="paged"); on the
            # block's OWN K / V the diagonal has no serving meaning (the prefill form is top-left), so the decline names the mode.
            raise NotImplementedError(
                "QsaSpec with causal_bottom_right=True on the block's own K / V (kv_source='self'): the bottom-right diagonal (the decode step's "
                "rows and its speculative verify rows at the END of a cached sequence) is the block-sparse DECODE / extend-with-prefix mode's "
                "over the pools -- declare kv_source='paged' with paged_kv_page_size and the pool / block-table samples; the prefill form over "
                "the block's own K / V serves the top-left causal diagonal"
            )
        if paged and not rec.bottom_right:
            # the mode's arm, read off the record: a record that retires it declines here by name instead of binding a body without it
            raise NotImplementedError("QsaSpec with kv_source='paged': the sparse adapter's record declines bottom_right (the decode mode's diagonal)")
        # The list form (QsaSpec.index_list_form): the shared per-sequence list is the decode mode's MTP form -- over the pools, at most the
        # record's shared-list cap of rows per sequence, under caller lists (the in-block indexer selects per row: its own row below); on the
        # block's own K / V every row carries its own list, so "shared" is refused by name rather than read as the per-row form.
        if q.index_list_form == QSA_INDEX_LIST_FORM_SHARED:
            if not paged:
                raise ValueError(
                    "QsaSpec(index_list_form='shared') is the block-sparse decode mode's shared per-sequence list (kv_source='paged', S_q <= "
                    f"{rec.index_shared_list_max_tokens}: one list per sequence anchored at the step-0 position); on the block's own K / V "
                    "(kv_source='self') every row carries its own list -- declare index_list_form=None or 'per_row'"
                )
            if q.index_source != "indexer" and self.seq_len > rec.index_shared_list_max_tokens:
                raise NotImplementedError(_qsa_decode_decline_texts(g, self.dtype)["shared_list_rows"])
        # The head dim, the block size, the top_k range, the GQA group and the dtype: the RECORD's claims, read, not transcribed.
        _check_qsa_geometry_against_record(g, self.dtype)
        if q.index_source == "indexer":
            # The in-block indexer's own rows (device-free): the scorer's dtype / head dim, the packed form; on the prefill form the
            # prefill scorer's head groups, over the POOLS (the decode step) the paged scorer's envelope -- the query rows per sequence,
            # the page's power-of-two row count, the e4m3 cache, the e4m3 pool, the shared list -- each the envelope's row.
            _check_qsa_indexer_geometry(g, self.dtype, thd=self.thd, paged=paged, s_q=self.seq_len, page_size=page_size, kv_cache_dtype=kv_cache_dtype)

    def _check_qsa_execute_args(
        self, block_ids, block_lens, h: torch.Tensor, w_iq_norm=None, w_ik_norm=None, index_k_compressed=None, block_ids_out=None, block_lens_out=None
    ) -> None:
        """The index tensors against the declaration -- FORM only (dtype / rank / shape / contiguity / device), never a
        value and never a sync (Rule 3): a QsaSpec block with caller lists needs ``block_ids``; one with the in-block
        indexer needs the indexer's two norm weights, refuses a list and form-checks its optional outputs; a dense block
        refuses every one of them rather than ignoring a tensor it cannot consume."""
        indexer_args = [
            nm
            for nm, x in (
                ("w_iq_norm", w_iq_norm),
                ("w_ik_norm", w_ik_norm),
                ("index_k_compressed", index_k_compressed),
                ("block_ids_out", block_ids_out),
                ("block_lens_out", block_lens_out),
            )
            if x is not None
        ]
        if self.qsa is None:
            if block_ids is not None or block_lens is not None:
                raise ValueError(
                    "block_ids / block_lens are the index lists of a block-sparse (QsaSpec) block; this block was declared without geometry.qsa "
                    "and attends densely -- a list it cannot consume is refused rather than silently ignored"
                )
            if indexer_args:
                raise ValueError(
                    f"{', '.join(indexer_args)}: the in-block indexer's inputs and outputs belong to a block declared with "
                    "geometry.qsa.index_source='indexer'; this block was declared without geometry.qsa and attends densely -- refused rather "
                    "than silently ignored"
                )
            return
        t, top_k = self.batch * self.seq_len, self.qsa.top_k
        if self.qsa.index_source == "indexer":
            q = self.qsa
            if block_ids is not None or block_lens is not None:
                raise ValueError(
                    "QsaSpec(index_source='indexer') derives the selection in the block: block_ids / block_lens are refused -- read the block's "
                    "own list back through block_ids_out / block_lens_out, or declare index_source='caller' to hand one in"
                )
            if self.indexer_decode and index_k_compressed is not None:
                raise ValueError(
                    "index_k_compressed is the prefill form's dense compressed-key output ([B, NB, 128]); over the pools the compressed keys live in the "
                    "PAGED cache execute(index_kv_cache=) (read and written in place through the block table) -- refused rather than silently ignored"
                )
            for nm, w in (("w_iq_norm", w_iq_norm), ("w_ik_norm", w_ik_norm)):
                if w is None:
                    raise ValueError(
                        f"QsaSpec(index_source='indexer') needs {nm} at execute: the indexer's [{q.index_head_dim}] RMSNorm weight in the activation "
                        f"dtype {self.act_dtype} (the pre-folded (1 + w) form the attention's norm weights use)"
                    )
                if (
                    not isinstance(w, torch.Tensor)
                    or tuple(int(x) for x in w.shape) != (q.index_head_dim,)
                    or w.dtype != self.act_dtype
                    or not w.is_contiguous()
                    or w.device != h.device
                ):
                    got = (tuple(w.shape), w.dtype, w.device, w.is_contiguous()) if isinstance(w, torch.Tensor) else type(w).__name__
                    raise ValueError(f"{nm} must be a contiguous [{q.index_head_dim}] {self.act_dtype} tensor on h's device {h.device}, got {got}")
            if index_k_compressed is not None:
                n_blocks = self.seq_len // q.block_size
                x = index_k_compressed
                shape = tuple(int(n) for n in x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
                if not isinstance(x, torch.Tensor) or x.dim() != 3 or shape[0] != self.batch or shape[1] < n_blocks or shape[2] != q.index_head_dim:
                    raise ValueError(
                        f"index_k_compressed must be [B={self.batch}, >= {n_blocks} (the complete {q.block_size}-token blocks of S={self.seq_len}), "
                        f"{q.index_head_dim}] -- the compressed-key cache whose rows [0, floor(S_b / {q.block_size})) the block writes; got {shape}"
                    )
                cache_dtype = q.index_cache_dtype if q.index_cache_dtype is not None else self.act_dtype
                if x.dtype != cache_dtype or not x.is_contiguous() or x.device != h.device:
                    raise ValueError(
                        f"index_k_compressed must be a contiguous {cache_dtype} tensor (QsaSpec.index_cache_dtype={q.index_cache_dtype}: "
                        f"{'the fp8 indexer cache' if cache_dtype == INDEX_CACHE_E4M3 else 'the activation dtype'}) on h's device {h.device}, got {x.dtype} on "
                        f"{x.device}, "
                        f"strides {tuple(x.stride())}"
                    )
            if block_ids_out is not None:
                _check_index_tensor(block_ids_out, "block_ids_out", (t, top_k), (self.batch, self.seq_len, top_k), h.device)
            if block_lens_out is not None:
                _check_index_tensor(block_lens_out, "block_lens_out", (t,), (self.batch, self.seq_len), h.device)
            # The 16-byte rule of the caller-list form's block_ids (below), on every buffer the indexer reads or writes in 16-byte vectors:
            # the compress kernel moves the two norm weights and the compressed-key cache in 128-bit accesses (its own check would refuse
            # a storage-offset view, but only at stage (4i), after the projection, the norm + RoPE and a declared write-through have run),
            # the cast kernel writes the e4m3 cache the same way, and block_ids_out becomes the list the sparse core binds as a TMA /
            # bulk-copy base.  Refused HERE, before any stage, so the refusal modifies nothing; block_lens_out is a plain row copy.
            for nm, x in (("w_iq_norm", w_iq_norm), ("w_ik_norm", w_ik_norm), ("index_k_compressed", index_k_compressed), ("block_ids_out", block_ids_out)):
                if x is not None and int(x.data_ptr()) % 16:
                    raise ValueError(
                        f"{nm} must start at a 16-byte aligned address (the indexer reads / writes it in 16-byte vectors; block_ids_out becomes "
                        f"the sparse core's TMA / bulk-copy base) -- a storage-offset view is refused before any stage runs; got data_ptr=0x{int(x.data_ptr()):x}"
                    )
            return
        if indexer_args:
            raise ValueError(
                f"{', '.join(indexer_args)}: the in-block indexer's inputs and outputs belong to a block declared with "
                "QsaSpec(index_source='indexer'); this block was declared with index_source='caller' and takes the selection as block_ids -- "
                "refused rather than silently ignored"
            )
        if block_ids is None:
            raise ValueError(
                "QsaSpec(index_source='caller') needs block_ids at execute: [T, top_k] int32 (or [B, S, top_k]) -- per query the ids of its "
                f"selected complete {self.qsa.block_size}-token blocks (block b = tokens [{self.qsa.block_size}b, {self.qsa.block_size}b + "
                f"{self.qsa.block_size}) of the query's own sequence), the valid prefix then -1 padding"
                + (" -- ONE list per sequence, [B, top_k], on this decode-mode declaration (S_q <= 4 over the pools)" if self.qsa_shared_list else "")
            )
        t, top_k = self.batch * self.seq_len, self.qsa.top_k
        if self.qsa_shared_list:
            # The decode mode's SHARED list (S_q <= 4 over the pools): ONE row per sequence, selected at the step-0 position and shared by
            # the step's rows -- never replicated per row (the per-token spelling is refused by name: the list form is compiled in).
            b = self.batch
            _check_index_tensor(
                block_ids,
                "block_ids",
                (b, top_k),
                (b, top_k),
                h.device,
                hint=f" (the decode mode's SHARED list: one row per sequence selected at the step-0 position kv_lens[b] - {self.seq_len}; a per-token "
                f"[{t}, {top_k}] / [{b}, {self.seq_len}, {top_k}] list is the form of a declaration with S_q > 4 or QsaSpec(index_list_form='per_row'))",
            )
        else:
            _check_index_tensor(
                block_ids,
                "block_ids",
                (t, top_k),
                (self.batch, self.seq_len, top_k),
                h.device,
                hint=" (a [B, top_k] list shared by a sequence's rows is the decode mode's shared form: a dense declaration with S_q <= 4 over the pools, "
                "kv_source='paged', under QsaSpec(index_list_form=None or 'shared'))",
            )
        # The sparse core binds block_ids as a TMA / bulk-copy base (assumed_align=16): the adapter refuses a view that starts off a
        # 16-byte boundary (a storage offset of a few ids) -- but at stage (4), after the projection, norm + RoPE and a declared
        # write-through have already run.  The same rule here, before any stage and on BOTH list forms (a review found the shared form
        # returning above it), so the refusal modifies nothing.
        ptr = int(block_ids.data_ptr())
        if ptr % 16:
            raise ValueError(
                "block_ids must start at a 16-byte aligned address (the sparse core binds it as a TMA / bulk-copy base; a storage-offset view "
                f"is refused before any stage runs); got data_ptr=0x{ptr:x}"
            )
        if block_lens is not None:
            if self.qsa_shared_list:
                _check_index_tensor(block_lens, "block_lens", (self.batch,), (self.batch,), h.device, hint=" (one count per sequence under the shared list)")
            else:
                _check_index_tensor(block_lens, "block_lens", (t,), (self.batch, self.seq_len), h.device)

    # -- the in-block indexer's DECODE step over the pools ---------------------

    def _check_indexer_decode_declaration(
        self, sample_index_kv_cache, sample_index_state_cache, sample_index_state_block_table, slot_mapping_present, commit_present, rope_tables_only, device
    ) -> Tuple[bool, bool, bool]:
        """The decode indexer's three side-cache SAMPLES and three input-contract flags at declaration, device-free: each flag a bool
        (``TypeError`` otherwise); on the form (``indexer_decode``) the samples REQUIRED -- the paged compressed-key cache ``[num_pages >= 1,
        P / block_size, 1, index_head_dim]`` with its rows contiguous inside a page (the scorer's read) in the DECLARED indexer cache dtype
        (``QsaSpec.index_cache_dtype``: bf16, or e4m3 -- refused both ways), the raw-key ring ``[num_ring_blocks >= 1, cap >= S_q + block_size - 1,
        1, index_head_dim]`` (``cap`` is the ring rule, a ``ValueError`` naming the bound; its blocks contiguous under the ring-slots arm) in the
        activation dtype with 16-byte-vector strides (an e4m3 ring is refused by its own rule: a bit copy, never quantized), the ring block
        table ``[B, 1]`` int32 -- and ``index_commit_present=True`` a typed decline (the
        two-phase commit is a follow-up: the serving stacks commit every row of the step); off the form every one of the six REFUSED
        (``ValueError`` naming them).  Returns the three flags.  The samples' strides are never compiled in (the kernels bind them
        symbolically), so ``execute`` matches dims and dtype, never strides."""
        flags = (
            ("index_state_slot_mapping_present", slot_mapping_present),
            ("index_commit_present", commit_present),
            ("rope_position_tables_only", rope_tables_only),
        )
        for nm, val in flags:
            if not isinstance(val, bool):
                raise TypeError(f"{nm} must be a bool (an input-contract flag of the in-block indexer's decode step), got {type(val).__name__}")
        samples = (
            ("sample_index_kv_cache", sample_index_kv_cache),
            ("sample_index_state_cache", sample_index_state_cache),
            ("sample_index_state_block_table", sample_index_state_block_table),
        )
        if not self.indexer_decode:
            given = [nm for nm, x in samples if x is not None] + [nm for nm, val in flags if val]
            if given:
                raise ValueError(
                    f"{', '.join(given)}: the side caches and the input-contract flags of the in-block indexer's DECODE step over the pools "
                    "(QsaSpec(index_source='indexer') with kv_source='paged' on a dense [B, S_q] declaration); this block was declared "
                    f"{'with index_source=' + repr(self.qsa.index_source) if self.qsa is not None else 'without geometry.qsa'} and kv_source="
                    f"{self.kv_source!r}{' (the packed form)' if self.thd else ''} -- refused rather than silently ignored"
                )
            return False, False, False
        missing = [nm for nm, x in samples if x is None]
        if missing:
            raise ValueError(
                f"QsaSpec(index_source='indexer') over the pools needs {', '.join(missing)} at declaration: sample_index_kv_cache [num_pages, "
                f"{self.paged_kv_page_size // self.qsa.block_size}, 1, {self.qsa.index_head_dim}] (the paged compressed-key cache on the main block table) in "
                f"the declared indexer cache dtype {self.index_cache_dtype}, sample_index_state_cache [num_ring_blocks, cap >= "
                f"{self.seq_len + _index_state_ring_extra_rows(self.qsa.block_size)}, 1, {self.qsa.index_head_dim}] (the raw-key ring) in the activation dtype, "
                f"and sample_index_state_block_table [B={self.batch}, 1] int32"
            )
        if commit_present:
            raise NotImplementedError(
                "index_commit_present=True (the two-phase commit: execute(n_commit=) the accepted prefix of the step's rows) is a follow-up of the "
                "in-block indexer's decode step; the serving stacks commit every row of the step -- a rejected draft's ring row and block are rewritten "
                "by the next step at the same positions (the ring is position-indexed, a block recompute lands on its own row) -- declare "
                "index_commit_present=False"
            )
        self._check_index_side_caches(
            sample_index_kv_cache,
            sample_index_state_cache,
            sample_index_state_block_table,
            device,
            prefix="sample_",
            ring_slots=bool(slot_mapping_present),
        )
        return bool(slot_mapping_present), bool(commit_present), bool(rope_tables_only)

    def _check_index_side_caches(self, index_kv_cache, index_state_cache, index_state_block_table, device, *, prefix: str, ring_slots: bool) -> None:
        """The two side caches and the ring block table by FORM (the declaration samples and the execute tensors alike; Rule 3: never a
        value, never a sync): dims, dtype, the 16-byte-vector strides the kernels move rows in, the cache's rows contiguous inside a page
        (the scorer's row stride), the ring's cap against the ring rule and its blocks contiguous under the ring-slots arm, the table
        ``[B, 1]`` int32 with a unit column stride, every one on ``device``.  Strides beyond these rules are the stack's (never compiled in)."""
        q, act = self.qsa, self.act_dtype
        d_i, bs = q.index_head_dim, q.block_size
        dev = torch.device(device)
        rpp = self.paged_kv_page_size // bs
        _check_index_side_cache(
            index_kv_cache,
            f"{prefix}index_kv_cache",
            rows=rpp,
            exact_rows=True,
            dtype=self.index_cache_dtype,  # the DECLARED cache dtype (QsaSpec.index_cache_dtype): bf16, or e4m3 -- both ways
            d=d_i,
            device=dev,
            what=f"the paged compressed-key cache: P / {bs} = {rpp} compressed rows per attention page, one row per {bs}-token block",
            rows_contiguous=True,
        )
        need = self.seq_len + _index_state_ring_extra_rows(bs)
        _check_index_side_cache(
            index_state_cache,
            f"{prefix}index_state_cache",
            rows=need,
            exact_rows=False,
            dtype=act,
            d=d_i,
            device=dev,
            what=(
                f"the raw-key ring needs at least S_q + {bs} - 1 = {need} rows per block: a completing block pools the {bs - 1} positions below the "
                f"step while the step writes S_q = {self.seq_len} rows"
            ),
            rows_contiguous=False,
        )
        if ring_slots and int(index_state_cache.stride(0)) != int(index_state_cache.shape[1]) * int(index_state_cache.stride(1)):
            raise ValueError(
                f"{prefix}index_state_cache: index_state_slot_mapping names flat ring rows (block x cap + pos % cap), which needs the ring's blocks "
                f"contiguous (stride[0] == cap x stride[1]); got shape {tuple(index_state_cache.shape)} strides {tuple(index_state_cache.stride())}"
            )
        bt = index_state_block_table
        if (
            not isinstance(bt, torch.Tensor)
            or bt.dtype != torch.int32
            or bt.dim() != 2
            or tuple(int(x) for x in bt.shape) != (self.batch, 1)
            or int(bt.stride(1)) != 1
        ):
            got = f"{bt.dtype} {tuple(bt.shape)} strides {tuple(bt.stride())}" if isinstance(bt, torch.Tensor) else type(bt).__name__
            raise ValueError(
                f"{prefix}index_state_block_table must be a [B={self.batch}, 1] int32 tensor (the ring block per sequence; any row stride), got {got}"
            )
        if bt.device != dev:
            raise ValueError(f"{prefix}index_state_block_table must live on {dev}, got {bt.device}")

    def _check_indexer_decode_execute_args(
        self, h, cos, sin, cos_pos, sin_pos, index_kv_cache, index_state_cache, index_state_block_table, index_state_slot_mapping, n_commit, slot_mapping
    ) -> None:
        """The decode indexer's execute inputs against the declaration -- FORM only, before any stage (Rule 3): on the form the two side
        caches, the ring block table and the position tables REQUIRED (``cos_pos`` / ``sin_pos`` ``[1 or B, >= max_pages x P, rope_dim]``: the
        RoPE-table ceiling by form -- every position the cache can address --, 16-byte bases), ``index_state_slot_mapping`` iff declared and
        of ``slot_mapping``'s dtype, ``cos`` / ``sin`` None iff ``rope_position_tables_only``, ``n_commit`` always refused (the two-phase commit
        is a typed decline at declaration); off the form every one of the seven REFUSED.  The 16-byte bases are checked HERE so a refusal
        modifies nothing (the kernels move the caches and the tables in 16-byte vectors)."""
        given = [
            nm
            for nm, x in (
                ("index_kv_cache", index_kv_cache),
                ("index_state_cache", index_state_cache),
                ("index_state_block_table", index_state_block_table),
                ("index_state_slot_mapping", index_state_slot_mapping),
                ("cos_pos", cos_pos),
                ("sin_pos", sin_pos),
                ("n_commit", n_commit),
            )
            if x is not None
        ]
        if not self.indexer_decode:
            if given:
                raise ValueError(
                    f"{', '.join(given)}: the execute inputs of the in-block indexer's DECODE step over the pools (QsaSpec(index_source='indexer') "
                    "with kv_source='paged'); this block was declared without that form and would never read them -- refused rather than silently ignored"
                )
            return
        if n_commit is not None:
            raise ValueError(
                "n_commit (the two-phase commit's accepted prefix) is refused: index_commit_present=True is a typed decline at declaration, so every row "
                "of the step commits -- pass n_commit=None"
            )
        g, q = self.geom, self.qsa
        missing = [
            nm
            for nm, x in (("index_kv_cache", index_kv_cache), ("index_state_cache", index_state_cache), ("index_state_block_table", index_state_block_table))
            if x is None
        ]
        if missing:
            raise ValueError(
                f"QsaSpec(index_source='indexer') over the pools needs {', '.join(missing)} at execute: the paged compressed-key cache, the raw-key ring "
                "and the ring block table the step reads and writes in place (the declaration's three samples' form)"
            )
        dev = torch.device(h.device)
        self._check_index_side_caches(
            index_kv_cache, index_state_cache, index_state_block_table, dev, prefix="", ring_slots=self.index_state_slot_mapping_present
        )
        for nm, x in (("index_kv_cache", index_kv_cache), ("index_state_cache", index_state_cache)):
            if int(x.data_ptr()) % 16:
                raise ValueError(
                    f"{nm} must start at a 16-byte aligned address (the kernels move its rows in 16-byte vectors), got data_ptr=0x{int(x.data_ptr()):x}"
                )
        if self.index_state_slot_mapping_present:
            if index_state_slot_mapping is None:
                raise ValueError(
                    "index_state_slot_mapping_present=True: execute needs index_state_slot_mapping [T] (the stack's ring slots block x cap + pos % cap, -1 = no write) "
                    "of slot_mapping's dtype -- no silent derived fallback"
                )
            x = index_state_slot_mapping
            t = self.batch * self.seq_len
            if not isinstance(x, torch.Tensor) or x.dim() != 1 or int(x.numel()) != t or not x.is_contiguous() or not x.is_cuda or x.device != dev:
                got = f"{x.dtype} {tuple(x.shape)} on {x.device}" if isinstance(x, torch.Tensor) else type(x).__name__
                raise ValueError(f"index_state_slot_mapping must be a contiguous [T={t}] int32 / int64 CUDA tensor on {dev}, got {got}")
            if x.dtype != slot_mapping.dtype:
                raise ValueError(
                    f"index_state_slot_mapping is {x.dtype} but slot_mapping is {slot_mapping.dtype}: the ring slots share the slot mapping's dtype (one "
                    "artifact per slot dtype) -- hand both in one dtype, never a cast"
                )
        elif index_state_slot_mapping is not None:
            raise ValueError(
                "index_state_slot_mapping is given, but this block was declared index_state_slot_mapping_present=False (the ring rows derive from "
                "index_state_block_table and kv_lens; the flag is a compile key) -- declare the flag, or pass None"
            )
        if self.rope_position_tables_only:
            if cos is not None or sin is not None:
                raise ValueError(
                    "cos / sin are refused under rope_position_tables_only=True: every rotation of the step reads cos_pos / sin_pos at the row's position "
                    "-- pass cos=None, sin=None"
                )
        elif cos is None or sin is None:
            raise ValueError(
                "cos / sin (the per-token RoPE tables) are required; rope_position_tables_only=True is the form that takes the position tables alone"
            )
        max_pages = int(self._descs["block_table"].shape[1])
        n_pos = max_pages * self.paged_kv_page_size
        for nm, x in (("cos_pos", cos_pos), ("sin_pos", sin_pos)):
            if x is None:
                raise ValueError(
                    f"QsaSpec(index_source='indexer') over the pools needs {nm} at execute: the position-indexed RoPE table [1 or B, >= {n_pos}, {g.rope_dim}] "
                    "(the block START of a completing block lies up to three positions below the step, which the per-token rows do not hold)"
                )
            if not isinstance(x, torch.Tensor) or x.dim() != 3 or int(x.shape[0]) not in (1, self.batch) or int(x.shape[2]) != g.rope_dim:
                got = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
                raise ValueError(f"{nm} must be [1 or B={self.batch}, S_pos, {g.rope_dim}] (position-indexed), got {got}")
            if int(x.shape[1]) < n_pos:
                raise ValueError(
                    f"{nm} holds {int(x.shape[1])} positions, but the main block table addresses {max_pages} pages x {self.paged_kv_page_size} tokens = {n_pos} "
                    "positions: the table is the ceiling on positions, so hand one of at least that many rows (the model's position table) -- a live row past "
                    "it would be reported -2 and rotated nowhere"
                )
            if x.dtype != self.act_dtype or int(x.stride(2)) != 1 or any(int(s) % 8 for s in x.stride()[:2]) or int(x.data_ptr()) % 16 or x.device != dev:
                raise ValueError(
                    f"{nm} must be a {self.act_dtype} table with a unit last stride, 16-byte-aligned row strides and base on {dev}; got {x.dtype}, strides "
                    f"{tuple(x.stride())}, data_ptr=0x{int(x.data_ptr()):x}, {x.device}"
                )
        if tuple(cos_pos.shape) != tuple(sin_pos.shape):
            raise ValueError(f"cos_pos and sin_pos must share a shape, got {tuple(cos_pos.shape)} and {tuple(sin_pos.shape)}")

    # -- paged-cache write-through (serving) ---------------------------------

    def _check_cache_write_declaration(self, page_size, quant) -> int:
        """``paged_kv_page_size`` at declaration: an ``int >= 0`` (``ValueError`` otherwise), a positive one a multiple of
        :data:`PAGED_KV_PAGE_ALIGN` (the serving stacks' page granularity; it also keeps every 4-token sparse block inside
        one page); then the pipelines the write-through stage has no arm for, typed, naming the feature.  Returns the
        normalised page size (0 = off).  Device-free.  The packed form (``thd=True``) is SERVED: the stage addresses tokens,
        so a packed block runs the same launch over its ``[T, H_kv, D]`` views."""
        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise TypeError(f"paged_kv_page_size must be an int (0 = no write-through; else the pools' tokens per page), got {type(page_size).__name__}")
        if page_size < 0:
            raise ValueError(f"paged_kv_page_size must be >= 0 (0 = no write-through), got {page_size}")
        if page_size == 0:
            return 0
        if page_size % PAGED_KV_PAGE_ALIGN:
            raise ValueError(
                f"paged_kv_page_size must be a positive multiple of {PAGED_KV_PAGE_ALIGN} (the serving stacks' page granularity; a 4-token "
                f"sparse block then never straddles a page), got {page_size}"
            )
        if self.qsa is not None and page_size % self.qsa.block_size:
            raise ValueError(
                f"paged_kv_page_size={page_size} must be a multiple of QsaSpec.block_size={self.qsa.block_size}: a selectable block is gathered "
                "as one unit and may not straddle two pages"
            )
        if quant is not None:
            raise NotImplementedError(
                f"paged_kv_page_size with quant={type(quant).__name__}: the write-through pools are bf16 / f16 (the activation dtype of the "
                "post-RoPE K / V); a quantized cache (e4m3 pools) arrives with the cache-dtype attribute and its cast kernel -- declare "
                "paged_kv_page_size on the bf16 / f16 pipeline"
            )
        if self.save_for_backward:
            raise NotImplementedError(
                "paged_kv_page_size with save_for_backward=True: the paged-cache write-through is a serving (inference) feature; a training "
                "forward keeps its K / V in the SavedForBackward record, not in a paged pool"
            )
        return page_size

    def _check_sparse_tile_form_declaration(self, sparse_tile_form, sparse_sub_tiles_per_cta, geometry) -> None:
        """The two knobs' argument contract at declaration (device-free): ``sparse_tile_form`` one of the heuristics' vocabulary
        (``"auto"`` / ``"lean"`` / ``"tcgen05"``), ``sparse_sub_tiles_per_cta`` None or an int >= 1 and ONLY with the explicit ``"lean"`` (a
        lean-only pin is never silently ignored); off a QsaSpec geometry only ``"auto"`` and no pin (the knobs are the block-sparse stage's).
        Records the declaration; the resolution follows once the form is known."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import SPARSE_TILE_FORM_AUTO, SPARSE_TILE_FORM_LEAN, SPARSE_TILE_FORMS

        if not isinstance(sparse_tile_form, str) or sparse_tile_form not in SPARSE_TILE_FORMS:
            raise ValueError(
                f"sparse_tile_form must be one of {SPARSE_TILE_FORMS} ('auto' = the placement rule on the measured parts, the tcgen05 core elsewhere; "
                f"'lean' / 'tcgen05' explicit), got {sparse_tile_form!r}"
            )
        k = sparse_sub_tiles_per_cta
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError(f"sparse_sub_tiles_per_cta must be None (the default k) or an int >= 1 (the lean tile's 32-key sub-tiles per CTA), got {k!r}")
        if geometry.qsa is None:
            if sparse_tile_form != SPARSE_TILE_FORM_AUTO or k is not None:
                raise ValueError(
                    f"sparse_tile_form={sparse_tile_form!r} / sparse_sub_tiles_per_cta={k!r} are the block-sparse stage's knobs (a QsaSpec geometry); this "
                    "geometry declares no QsaSpec, so stage (4) is the dense SDPA -- leave both at their defaults"
                )
        if k is not None and sparse_tile_form != SPARSE_TILE_FORM_LEAN:
            raise ValueError(
                f"sparse_sub_tiles_per_cta={k} is the lean tile's pin (its sub-tiles per CTA) and needs sparse_tile_form='lean'; under "
                f"{sparse_tile_form!r} the pin would be ignored where the route resolves the tcgen05 core -- declare 'lean' to force the body"
            )
        self.sparse_tile_form_requested: str = sparse_tile_form
        self.sparse_sub_tiles_per_cta_requested: Optional[int] = k
        # the RESOLVED body of a QsaSpec block ("lean" / "tcgen05") and its kernel_template (heuristics.SPARSE_TILE_LEAN / SPARSE_TILE_TCGEN05):
        # None off QsaSpec and over e4m3 pools, where NO body of the sparse core runs (the gathered cast + the dense decode tile)
        self.sparse_tile_form: Optional[str] = None
        self.sparse_sub_tiles_per_cta: Optional[int] = None  # the lean body's k; None otherwise
        self.sparse_tile: Optional[str] = None

    def _resolve_sparse_tile_form_off_decode(self) -> None:
        """The body of a QsaSpec block OFF the block-sparse decode mode (the dense ``[B, S]`` and the packed forms over the block's own K / V):
        the tcgen05 core -- the lean tile serves the DECODE form only, so ``"auto"`` and ``"tcgen05"`` resolve to it and an explicit ``"lean"``
        hears the lean body's own decline (``heuristics.sparse_lean_tile_decline``: THD, or the top-left prefill form).  The decode mode
        resolved its body in ``_check_serving_declaration`` (``sparse_tile_form`` set -- or, over e4m3 pools, settled that NO body runs:
        ``sparse_tile_form`` None, so the paged form never comes here); a dense geometry has no body."""
        if self.qsa is None or self.sparse_tile_form is not None or self.kv_source == KV_SOURCE_PAGED:
            return
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import sparse_tile_rules as _H

        if self.sparse_tile_form_requested == _H.SPARSE_TILE_FORM_LEAN:
            reason = _H.sparse_lean_tile_decline(
                s_q=self.seq_len,
                group=self.geom.gqa_ratio,
                top_k=self.qsa.top_k,
                thd=self.thd,
                bottom_right=False,
                list_per_sequence=False,
                seq_kv_lens_present=self.seq_lens_present,
            )
            raise NotImplementedError(f"sparse_tile_form='lean' -- {reason}")
        self.sparse_tile_form = _H.SPARSE_TILE_FORM_TCGEN05
        self.sparse_tile = _H.SPARSE_TILE_TCGEN05

    def _qsa_shared_list_derived(self, kv_source: Optional[str] = None) -> bool:
        """The block-sparse decode mode's LIST FORM as a declaration fact (``QsaSpec.index_list_form`` as declared, or derived from ``S_q``
        under caller lists: ONE list per sequence at ``S_q <= `` the record's shared-list cap, one per row past it and under the in-block
        indexer) -- ONE derivation (:func:`_qsa_shared_list_form`, the static envelope's too) for the serving declaration (the body's
        admissibility; ``kv_source`` as given there, before the attribute exists) and the stage's ``list_per_sequence``."""
        q = self.qsa
        if q is None or (self.kv_source if kv_source is None else kv_source) != KV_SOURCE_PAGED:
            return False
        return _qsa_shared_list_form(q, self.seq_len)

    def _pdl_default(self, quant) -> bool:
        """The AUTO reading of ``use_pdl`` (None): True iff this is the dense DECODE form on a decode-tile route (``DecodeRoute.pdl_capable``:
        the tile and its split combine carry the griddepcontrol pair), the bf16 / f16 block (no ``quant``, no e4m3 cache, no ``QsaSpec``) and
        the sample's device is one of :data:`DECODE_PDL_DEFAULT_CCS` (cc 10.7, where the default was measured) -- OR the block-sparse DECODE
        mode (``sparse_decode``: a caller-list ``QsaSpec`` geometry over bf16 / f16 pools, whose core and combine carry the sites; never
        ``index_source="indexer"``, whose explicit True is declined) on a part of
        :data:`SPARSE_DECODE_PDL_DEFAULT_CCS`, EMPTY until that chain is measured, so the sparse arm reads False everywhere today; False
        everywhere else -- the EXTEND form, ``kv_source="self"``, the DECODE form's prefill route, the quantized pipelines, every other part
        and no CUDA device.  Never True where an explicit True is refused: the two readings of the knob answer for ONE domain (no ``quant``,
        pools in the activation dtype -- ``kv_cache_dtype`` None or that dtype), the AUTO one narrowed further by route, form and part."""
        if quant is not None or self.kv_cache_e4m3 or self.dtype not in (torch.bfloat16, torch.float16):
            return False
        if self.qsa is not None:
            # the decode mode's fact from the two attributes already set here (`sparse_decode` itself is derived from them later in the
            # constructor): the paged form on a dense [B, S_q] QsaSpec declaration -- and the CALLER-LIST form only: the in-block indexer
            # (`index_source="indexer"`) refuses the explicit True above, so AUTO never resolves True there either, whatever the tuple holds
            return bool(
                self.kv_source == KV_SOURCE_PAGED
                and not self.thd
                and self.qsa.index_source != "indexer"
                and _device_cc_of(self.device) in SPARSE_DECODE_PDL_DEFAULT_CCS
            )
        return bool(self.decode_route is not None and self.decode_route.pdl_capable and _device_cc_of(self.device) in DECODE_PDL_DEFAULT_CCS)

    def _resolve_cache_write_small_m(self, requested: Optional[bool], quant) -> bool:
        """Whether the paged write-through runs INSIDE the norm + RoPE launch (``fuse_cache_write_small_m``: None = AUTO -- the measured rule
        of :data:`CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS`: folded in at ``T = B x S_q < 128`` rows or at ``S_q == 1``, the separate launch at
        ``T >= 128`` with ``S_q > 1``, and the separate launch wherever an explicit ``fuse_cache_write=False`` asked for it by name: the request
        outranks the AUTO fold --, True / False by name -- this arm's own knob, the strongest request, followed under either reading of
        ``fuse_cache_write``; ``TypeError`` otherwise).  Served on the dense DECODE form's UNFUSED chain with
        paged pools in the activation dtype (bf16 / f16, no quant, no indexer band, no e4m3 cache) when the TMA norm + RoPE kernel tiles the
        geometry; AUTO reads False everywhere else and an explicit True is a typed decline naming the reason.  Device-free (the arch half --
        sm_90+ for the TMA kernel -- is the stage's ``check_support``; the dense DECODE form itself is served on cc 10.7 only)."""
        if requested is not None and not isinstance(requested, bool):
            raise TypeError(f"fuse_cache_write_small_m must be None (auto), True or False, got {type(requested).__name__}")
        if requested is None:
            # an explicit fuse_cache_write=False asks for the separate cache-write launch by name: the request outranks the AUTO fold
            # (user request > heuristic proposal, as the constructor applies it to fuse_cache_write=True); the row rule decides otherwise
            want = self.fuse_cache_write is not False and (self.seq_len == 1 or self.batch * self.seq_len < CACHE_WRITE_SMALL_M_DEFAULT_MAX_ROWS)
        else:
            want = requested
        if not want:
            return False
        why = None
        if not self.paged_kv_page_size:
            why = "needs paged_kv_page_size > 0 (there is no write-through to fold)"
        elif self.decode_route is None:
            why = (
                "is the dense DECODE form's arm (kv_source='paged' on a dense bf16 / f16 declaration without a QsaSpec); the EXTEND / packed forms, the "
                "write-through on the block's own K / V and the block-sparse decode mode keep their separate cache-write launch"
            )
        elif self._norm_rope is None:
            why = "folds the write-through into the UNFUSED chain's norm + RoPE launch, and this declaration runs the fused projection fork (which owns the write-through: fuse_cache_write)"
        # The three rows below are GUARDS no declaration reaches through the constructor today: the write-through's own quant decline
        # (`_check_cache_write_declaration`) and the no-dense-route row above (a QsaSpec geometry, with or without the indexer band, and the
        # e4m3 cache ride the block-sparse decode mode) speak first.  Kept so the resolver stays honest on its own facts should a form admit
        # one of those pipelines onto a decode route; the host cell pins their text on the resolver directly.
        elif quant is not None:
            why = f"serves the bf16 / f16 pipeline only, got quant={type(quant).__name__}"
        elif self.geom.index_band:
            why = "with an indexer band the raw indexer key's pool is the separate cache-write launch's; declare fuse_cache_write_small_m=False (or None)"
        elif self.kv_cache_e4m3:
            why = "with kv_cache_dtype=torch.float8_e4m3fn the write-through QUANTIZES (its own stage); declare fuse_cache_write_small_m=False (or None)"
        elif self.act_dtype not in (torch.bfloat16, torch.float16):
            why = f"serves bf16 / f16 activations, got {self.act_dtype}"
        else:
            from .kernels.qk_norm_rope_tma import validate_shape as _tma_validate

            g = self.geom
            try:
                _tma_validate(g.d_head, g.rope_dim, g.h_q, g.h_kv, self._norm_rope.resolve_tile_rows(), self._norm_rope.threads_per_cta)
            except (ValueError, NotImplementedError) as exc:
                why = f"is an arm of the TMA norm + RoPE kernel, whose tiling does not fit this geometry ({exc}); the LDG kernel has no such arm"
        if why is None:
            return True
        if requested is True:
            raise NotImplementedError(f"fuse_cache_write_small_m=True {why}")
        return False  # AUTO: the separate launch where the arm is not served (the fused fork, the EXTEND / packed forms, a quantized pipeline, ...)

    def _check_fuse_cache_write_declaration(self, fuse_cache_write, quant) -> Optional[bool]:
        """``fuse_cache_write`` at declaration: None / True / False (``TypeError`` otherwise); True on a pipeline the fused projection's
        write-through arm does not serve is a typed ``NotImplementedError`` naming the feature (the arm: ``paged_kv_page_size > 0``,
        ``fuse_norm_rope=True``, bf16 / f16 activations, no indexer band, 16-bit pools -- the bf16 fork's epilogue; the quantizing forks, the
        unfused chain, the indexer band and the e4m3 KV cache keep the separate cache-write launch).  Device-free."""
        if fuse_cache_write is not None and not isinstance(fuse_cache_write, bool):
            raise TypeError(f"fuse_cache_write must be None (auto), True or False, got {type(fuse_cache_write).__name__}")
        if fuse_cache_write is True:
            if not self.paged_kv_page_size:
                raise ValueError("fuse_cache_write=True needs paged_kv_page_size > 0 (there is no write-through to fold)")
            if quant is not None:
                raise NotImplementedError(
                    f"fuse_cache_write=True with quant={type(quant).__name__}: the write-through arm lives in the bf16 fused projection's epilogue; the "
                    "quantizing forks keep the separate cache-write launch (the paged pools are bf16 / f16 either way)"
                )
            if not self.fuse_norm_rope:
                raise NotImplementedError(
                    "fuse_cache_write=True needs fuse_norm_rope=True: the write-through is an epilogue arm of the fused norm + RoPE projection; the "
                    "unfused chain norms the slab in place after the GEMM and keeps the separate cache-write launch"
                )
            if not self.fuse_norm_rope_effective:
                # reachable through fuse_norm_rope_small_m=False only: under AUTO the explicit request keeps the fork (the effective flag reads
                # the raw argument), so the one way to ask for the arm AND for the chain it does not live in is to spell both
                raise NotImplementedError(
                    "fuse_cache_write=True with fuse_norm_rope_small_m=False: the write-through arm lives in the fused norm + RoPE projection's epilogue, "
                    "and fuse_norm_rope_small_m=False lowers stage (1) to the unfused chain at decode row counts (T = B x S_q within the small-M route's "
                    "bound), which keeps the separate cache-write launch -- declare fuse_norm_rope_small_m=None (the request keeps the fork) or True, or "
                    "fuse_cache_write=None / False"
                )
            if self.geom.index_band:
                raise NotImplementedError(
                    "fuse_cache_write=True with an indexer band: the fused projection declines the band's 640-column tile, and the raw indexer key's "
                    "pool is the separate cache-write launch's; declare fuse_cache_write=False (or None)"
                )
            if self.kv_cache_e4m3:
                raise NotImplementedError(
                    "fuse_cache_write=True with kv_cache_dtype=torch.float8_e4m3fn: the fused projection's epilogue writes bf16 / f16 rows, and the e4m3 "
                    "cache's write-through QUANTIZES (two per-tensor casts + the bit copy through the pools' bf16 views) -- declare fuse_cache_write=False (or None)"
                )
        return fuse_cache_write

    def _fuse_prepare_default(self) -> bool:
        """The AUTO reading of ``fuse_prepare`` (None): True iff this is the in-block indexer's DECODE step over the pools (the form the
        fused launch's three paged arms serve), the fused form is ADMISSIBLE on this declaration, and the sample's device is one of
        :data:`SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS` -- cc 10.7 since the 2026-10-21 end-to-end gate measured the fused step +9.6 .. +14.5 %
        at B <= 4 against the unfused twin on an exclusive part (the measured-default rule; every other part waits for its own measurement)
        --; False everywhere else (the knob's former default).  Never True where an explicit True is declined: the reading walks the explicit
        rows of :meth:`_check_fuse_prepare_declaration` (the write-through attribute, ``fuse_norm_rope``, the in-place pipeline, ``thd``), the
        e4m3 cache's row and the fused stage's own ``rope_dim`` row (``_QsaPrepare.check_support``), and reads False -- the unfused twin --
        where any one holds, so a legal out-of-place declaration (``inplace_qkv=False``) constructs under the measured default instead of
        being refused for a knob it never set (an explicit True keeps its typed decline there).  A performance knob: both readings compute
        the same function.
        """
        admissible = (
            bool(self.paged_kv_page_size)
            and not self.fuse_norm_rope
            and self.inplace_qkv
            and not self.thd
            and not self.kv_cache_e4m3
            and self.geom.rope_dim > 0
        )
        return bool(self.indexer_decode and admissible and _device_cc_of(self.device) in SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS)

    def _check_fuse_prepare_declaration(self, fuse_prepare) -> bool:
        """``fuse_prepare`` at declaration: ``None`` (the measured default, :meth:`_fuse_prepare_default`), ``True`` or ``False``
        (``TypeError`` otherwise); ``True`` needs the write-through attribute (there is nothing to fold without it -- ``ValueError``
        naming both), the in-place pipeline (the fused kernel rewrites Q / K in the slab; the out-of-place layout keeps its two stages --
        ``ValueError``) and not ``fuse_norm_rope`` (the norm + RoPE already live in the projection epilogue there; the prepare kernel has
        no cache-write-only arm -- ``NotImplementedError``); then the knob's own arms: not ``thd=True`` (the prepare kernel addresses the
        dense ``[B, S]`` slab) and, over the pools, the block-sparse decode mode only (the caller-list band form and the in-block indexer's
        decode step: the dense-geometry DECODE form's chain folds its write-through into the norm + RoPE launch) -- ``NotImplementedError``.
        ``quant`` / ``save_for_backward`` are excluded by the attribute's own rows, which ran first.  Device-free.  A performance knob:
        whatever it returns, the block computes the same function."""
        if fuse_prepare is not None and not isinstance(fuse_prepare, bool):
            raise TypeError(f"fuse_prepare must be a bool (the prepare-fusion knob) or None (the measured default), got {type(fuse_prepare).__name__}")
        if fuse_prepare is None:
            fuse_prepare = self._fuse_prepare_default()
        if not fuse_prepare:
            return False
        if not self.paged_kv_page_size:
            raise ValueError(
                "fuse_prepare=True folds stages (2)+(3) and the paged-cache write-through (4w) into one launch, so it needs paged_kv_page_size > 0; "
                "this block was declared without the write-through (paged_kv_page_size=0) -- there is nothing to fold (pass fuse_prepare=False)"
            )
        if self.fuse_norm_rope:
            raise NotImplementedError(
                "fuse_prepare=True with fuse_norm_rope=True: the norm + RoPE already run in the projection epilogue there and the prepare kernel "
                "has no cache-write-only arm; pick one fusion (fuse_norm_rope=True keeps the write-through as its own stage)"
            )
        if not self.inplace_qkv:
            raise ValueError(
                "fuse_prepare=True rewrites Q / K in place in the projection slab and writes the pools from the same registers; the out-of-place "
                "layout (inplace_qkv=False) keeps its two stages -- pass inplace_qkv=True (the inference default) or fuse_prepare=False"
            )
        if self.thd:
            raise NotImplementedError(
                "fuse_prepare=True with thd=True: the prepare kernel addresses the dense [B, S] slab (one RoPE table row per (b, s)); its "
                "packed-sequence arm is a follow-up -- run the two-stage form (fuse_prepare=False: the norm + RoPE stage and the write-through "
                "both serve thd=True)"
            )
        if self.kv_source == KV_SOURCE_PAGED and not self.sparse_decode:
            raise NotImplementedError(
                "fuse_prepare=True with kv_source='paged' on a dense geometry (the DECODE form): over the pools the prepare fusion serves the block-sparse "
                "decode mode (a QsaSpec geometry -- the caller-list band form and the in-block indexer's decode step, each pinned bitwise its unfused "
                "twin); the dense decode form's unfused chain folds the write-through into the norm + RoPE launch (fuse_cache_write_small_m) -- run "
                "the two-stage form (fuse_prepare=False)"
            )
        return True

    # -- the e4m3 KV cache (kv_cache_dtype) ---------------------------------

    def _check_kv_cache_dtype_declaration(self, kv_cache_dtype, quant, kv_source=KV_SOURCE_SELF) -> torch.dtype:
        """``kv_cache_dtype`` at declaration, device-free: ``None`` / the activation dtype = the pools as before; ``torch.float8_e4m3fn``
        = the e4m3 KV cache, which needs the write-through attribute (``paged_kv_page_size > 0``: the attribute changes what the
        write-through writes) and bf16 activations (the cast writes bf16 rows, the decode tile reads a bf16 Q); ``quant`` and
        ``save_for_backward`` with the write-through are the write-through's own declines (heard first).  e5m2 is declined BY NAME; any
        other dtype is a ``ValueError``; the packed (``thd=True``) PREFILL (``kv_source="self"``) with the attribute is a typed decline (its
        quantizing write-through has no accept cell yet) -- a packed declaration OVER THE POOLS is the serving table's EXTEND row, which
        speaks for itself later.  Returns the pools' dtype.  Runs AFTER ``_check_cache_write_declaration`` and before the serving table."""
        act = self.act_dtype
        if kv_cache_dtype is None:
            return act
        if not isinstance(kv_cache_dtype, torch.dtype):
            raise TypeError(
                f"kv_cache_dtype must be a torch.dtype (None = the activation dtype; torch.float8_e4m3fn = the e4m3 KV cache), got {type(kv_cache_dtype).__name__}"
            )
        if kv_cache_dtype == torch.float8_e5m2:
            raise NotImplementedError(
                "kv_cache_dtype=torch.float8_e5m2: the e4m3 KV cache is the one served (the serving stacks' fp8 KV cache is e4m3fn; the gathered cast and "
                "the quantizing write-through are e4m3 kernels) -- declare torch.float8_e4m3fn"
            )
        if kv_cache_dtype in (torch.bfloat16, torch.float16):
            if kv_cache_dtype != act:
                raise ValueError(
                    f"kv_cache_dtype={kv_cache_dtype} differs from the activation dtype {act}: a 16-bit pool holds the post-RoPE K / V rows bitwise (no "
                    f"cast is written for a 16-bit pool of the other width) -- pass None, {act}, or torch.float8_e4m3fn"
                )
            return act
        if kv_cache_dtype != KV_CACHE_E4M3:
            raise ValueError(f"kv_cache_dtype must be None, the activation dtype ({act}) or torch.float8_e4m3fn (the e4m3 KV cache), got {kv_cache_dtype}")
        if not self.paged_kv_page_size:
            raise ValueError(
                "kv_cache_dtype=torch.float8_e4m3fn belongs to the paged KV-cache write-through: declare paged_kv_page_size > 0 (the pools the e4m3 rows "
                "are written into and read from); without a cache there is nothing to quantize"
            )
        if quant is not None:
            # unreachable through the constructor (the write-through's quant row speaks first); kept so the attribute is honest on its own
            raise NotImplementedError(
                f"kv_cache_dtype=torch.float8_e4m3fn with quant={type(quant).__name__}: the e4m3 KV cache is a form of the bf16 pipeline -- the quantized "
                "pipelines' K / V are e4m3 codes of another recipe (the per-tensor FP8 / MXFP8 SDPA rows take e4m3 Q / K / V together)"
            )
        if act != torch.bfloat16:
            raise NotImplementedError(
                f"kv_cache_dtype=torch.float8_e4m3fn with {act} activations: the gathered cast writes bf16 rows and the d256 decode tile over them reads a "
                "bf16 Q (one dtype per plan) -- declare bf16 activations (f16 activations over an e4m3 cache are a follow-up)"
            )
        if self.thd and kv_source != KV_SOURCE_PAGED:
            raise NotImplementedError(
                "kv_cache_dtype=torch.float8_e4m3fn with thd=True: the packed prefill's quantizing write-through (the same token-addressed launches over "
                "the packed slab) has no accept cell yet, so it is declared as a typed decline rather than claimed -- run the dense prefill over e4m3 "
                "pools, or the packed prefill over bf16 / f16 pools (kv_cache_dtype=None)"
            )
        return KV_CACHE_E4M3

    def _check_kv_cache_dtype_serving(self, decode_split_kv, inplace_qkv, fuse_norm_rope) -> int:
        """The e4m3 cache OVER THE POOLS (``kv_source="paged"``), device-free, after the serving table: served on the block-sparse
        DECODE mode at ``S_q <= 4`` only -- the EXTEND form and the dense DECODE form read bf16 / f16 pools (the paged THD body and the
        decode tile have no dequant arm; the gathered cast is list-driven), the per-token list form would gather 2176 x 256 bf16 rows
        per (row, KV head) item, the decode tile over the scratch writes no LSE the block hands back and runs unsplit, and the mode
        norms + rotates OUT of place (the tile's per-item Q is a view of the compact Q alone), so ``fuse_norm_rope`` and an explicit
        ``inplace_qkv=True`` are declined and ``inplace_qkv=None`` reads as False.  Returns the split (1)."""
        e4 = "kv_cache_dtype=torch.float8_e4m3fn"
        if self.thd:
            raise NotImplementedError(
                f"kv_source='paged' with {e4} on a packed (thd=True) declaration -- the EXTEND form over e4m3 pools: the paged d256 THD body reads bf16 / "
                "f16 pools (no dequant arm); the e4m3 read is the block-sparse DECODE mode's (a dense [B, S_q <= 4] QsaSpec declaration: the gathered cast + "
                "the decode tile)"
            )
        if self.qsa is None:
            raise NotImplementedError(
                f"kv_source='paged' with {e4} on a dense geometry (the DECODE form): the Rubin d256 decode tile reads bf16 / f16 pools, and the gathered-cast "
                "composition is list-driven (a QsaSpec geometry: the listed blocks and the open tail, at most 2176 rows per item); a dense e4m3 pool needs the "
                "tile's in-kernel dequant arm -- a follow-up.  Declare a QsaSpec geometry (the block-sparse DECODE mode over e4m3 pools) or bf16 / f16 pools"
            )
        if self.seq_len > _UPCAST_MAX_S_Q:
            raise NotImplementedError(
                f"{e4} over the pools serves the SHARED-list form, S_q <= {_UPCAST_MAX_S_Q} rows per sequence (the decode step and its speculative rows): the "
                f"gathered cast lays out {_UPCAST_ROWS} x {_UPCAST_D} bf16 rows per (row, KV head) item ({2 * _UPCAST_ROWS * _UPCAST_D * 2} bytes), so the "
                f"per-token list form (S_q > {_UPCAST_MAX_S_Q}: the extend-with-prefix mode) over e4m3 pools waits for the in-kernel dequant arm; got S_q={self.seq_len}"
            )
        if self.qsa.index_list_form == QSA_INDEX_LIST_FORM_PER_ROW:
            raise NotImplementedError(
                f"QsaSpec(index_list_form='per_row') with {e4} over the pools: the gathered cast reads ONE list per sequence (the shared form anchored at the "
                "step-0 position) per (row, KV head) item; the per-row list over e4m3 pools waits for the sparse core's in-kernel dequant arm -- declare "
                "index_list_form=None or 'shared', or bf16 pools"
            )
        if self.return_lse:
            raise NotImplementedError(
                f"return_lse=True with {e4} over the pools: the decode tile over the gathered scratch writes its Stats item-major ([B x S_q x H_kv, G, 1]) and "
                "the block hands no LSE back on this form -- a follow-up; declare return_lse=False"
            )
        if decode_split_kv is not None and int(decode_split_kv) != 1:
            raise NotImplementedError(
                f"decode_split_kv={decode_split_kv} with {e4}: the e4m3 mode runs the d256 decode tile UNSPLIT over the gathered scratch (at most 17 KV tiles "
                "per item); a split there is a follow-up -- pass None or 1"
            )
        why_oop = (
            "the decode tile over the gathered scratch reads the step's Q per (token, KV head) item, a view of the COMPACT normed Q only (the slab's token "
            "stride cannot fold token and KV head into one batch stride), so the mode norms + rotates OUT of place"
        )
        if fuse_norm_rope:
            raise NotImplementedError(
                f"fuse_norm_rope=True with {e4} over the pools: {why_oop} -- declare fuse_norm_rope=False (the fused projection writes the slab in place)"
            )
        if inplace_qkv:
            raise NotImplementedError(f"inplace_qkv=True with {e4} over the pools: {why_oop} -- leave inplace_qkv=None (read as False here) or pass False")
        self.inplace_qkv = False
        return 1

    def _check_cache_write_args(
        self,
        h: torch.Tensor,
        k_cache,
        v_cache,
        block_table,
        kv_lens,
        slot_mapping,
        index_k_raw,
        seq_lens=None,
        num_tokens_live: Optional[int] = None,
        k_scale=None,
        v_scale=None,
    ) -> None:
        """The serving inputs against the declaration -- FORM only (dtype / rank / shape / strides / alignment / device),
        never a value and never a sync (Rule 3).  A block declared with ``paged_kv_page_size`` REQUIRES ``k_cache`` /
        ``v_cache`` / ``slot_mapping`` (and ``index_k_raw`` iff its geometry declares the indexer band); a block declared
        without it REFUSES every one of them rather than ignoring a pool it would never write (the ``h_sf`` pattern).
        ``block_table`` / ``kv_lens`` are the EXTEND form's inputs (``kv_source="paged"``): REQUIRED there -- and the pools
        must then match the declared samples' shape and strides exactly (their layout is compiled in) -- and refused on a
        block declared ``kv_source="self"`` (the write-through alone attends over the block's own K / V)."""
        from .cache_write_contract import check_pool, check_slot_mapping

        given = [nm for nm, x in (("k_cache", k_cache), ("v_cache", v_cache), ("slot_mapping", slot_mapping), ("index_k_raw", index_k_raw)) if x is not None]
        read_mode = [nm for nm, x in (("block_table", block_table), ("kv_lens", kv_lens)) if x is not None]
        scales = [nm for nm, x in (("k_scale", k_scale), ("v_scale", v_scale)) if x is not None]
        if not self.paged_kv_page_size:
            given += scales
            if given or read_mode:
                raise ValueError(
                    f"{', '.join(given + read_mode)}: the paged KV-cache inputs belong to a block declared with paged_kv_page_size > 0 (the "
                    "write-through of the post-RoPE K / V into paged pools); this block was declared without it (paged_kv_page_size=0) and "
                    "would never write them -- refused rather than silently ignored"
                )
            return
        paged = self.kv_source == KV_SOURCE_PAGED
        if read_mode and not paged:
            raise ValueError(
                f"{', '.join(read_mode)}: the EXTEND form's inputs (attention over the pools through a block table and the per-sequence logical "
                "KV lengths) belong to a block declared kv_source='paged'; this block was declared kv_source='self' and attends over its own "
                "K / V after the write-through -- refused rather than silently ignored"
            )
        g, ps = self.geom, self.paged_kv_page_size
        if self.kv_cache_e4m3:
            # the per-tensor DEscales of the e4m3 cache: read on device by the write-through's casts (as reciprocals) and by the gathered cast
            dev = torch.device(h.device)
            for nm, x in (("k_scale", k_scale), ("v_scale", v_scale)):
                if not (isinstance(x, torch.Tensor) and x.dtype == torch.float32 and int(x.numel()) == 1 and x.is_cuda and x.device == dev):
                    got = f"{x.dtype} x {int(x.numel())} on {x.device}" if isinstance(x, torch.Tensor) else type(x).__name__
                    raise ValueError(
                        f"kv_cache_dtype=torch.float8_e4m3fn needs {nm} at execute: a 1-element fp32 CUDA tensor on h's device {dev} (the pool's per-tensor "
                        f"DEscale: value = code x scale; the write-through quantizes with its reciprocal, the gathered cast multiplies it in -- both on device), got {got}"
                    )
        elif scales:
            raise ValueError(
                f"{', '.join(scales)}: the per-tensor scales of an e4m3 KV cache (kv_cache_dtype=torch.float8_e4m3fn); this block's pools are {self.kv_cache_dtype} "
                "and no stage would read them -- refused rather than silently ignored"
            )
        for nm, x in (("k_cache", k_cache), ("v_cache", v_cache), ("slot_mapping", slot_mapping)):
            if x is None:
                raise ValueError(
                    f"paged_kv_page_size={ps} needs {nm} at execute: k_cache / v_cache [num_pages, {g.h_kv}, {ps}, {g.d_head}] {self.kv_cache_dtype} pools "
                    f"(HND compact, NHD by strides, or the two halves of a unified [num_pages, {g.h_kv}, {ps}, {2 * g.d_head}] content row) and "
                    f"slot_mapping [{self.batch * self.seq_len}] int32 / int64 (one flat slot per token, page * page_size + offset; a negative "
                    "slot writes nothing)"
                )
        if paged:
            bt = self._descs["block_table"]
            for nm, x in (("block_table", block_table), ("kv_lens", kv_lens)):
                if x is None:
                    raise ValueError(
                        f"kv_source='paged' needs {nm} at execute: block_table [num_sequences={self.num_sequences}, >= max_pages={int(bt.shape[1])}] int32 "
                        f"(every sequence's page ids, at the row stride sample_block_table declared) and kv_lens [{self.num_sequences}] int32 (each "
                        "sequence's LOGICAL KV length INCLUDING this step's tokens; 0 = a padded entry)"
                    )
        t = self.batch * self.seq_len
        check_pool(k_cache, "k_cache", h=g.h_kv, page_size=ps, d=g.d_head, dtype=self.kv_cache_dtype, device=h.device)
        check_pool(v_cache, "v_cache", h=g.h_kv, page_size=ps, d=g.d_head, dtype=self.kv_cache_dtype, device=h.device)
        # Under num_tokens_live (appended) the slot mapping may carry the declared T_max slots (the first T_live are read) or exactly T_live.
        check_slot_mapping(slot_mapping, "slot_mapping", t=t, device=h.device, t_live=num_tokens_live)
        if paged:
            self._check_extend_args(h, k_cache, v_cache, block_table, kv_lens, seq_lens=seq_lens)
        if g.index_band and self.indexer_decode:
            # the decode indexer over the pools: the raw key enters the stack's RING (the prepare / the step form); a raw-key pool is refused
            # by name rather than silently ignored
            if index_k_raw is not None:
                raise ValueError(
                    "index_k_raw is the raw indexer key's pool of a caller-list band declaration; under QsaSpec(index_source='indexer') over the pools "
                    "the raw key lives in the ring (index_state_cache) and nothing writes a raw-key pool -- refused rather than silently ignored"
                )
        elif g.index_band:
            if index_k_raw is None:
                raise ValueError(
                    f"paged_kv_page_size={ps} with the indexer band needs index_k_raw at execute: the raw indexer key's pool "
                    f"[num_pages, {ps}, {g.qsa.index_head_dim}] {self.act_dtype} (D contiguous; written at the same slot_mapping)"
                )
            if not isinstance(index_k_raw, torch.Tensor) or index_k_raw.dim() != 3:
                raise ValueError(
                    f"index_k_raw must be a 3-D [num_pages, {ps}, {g.qsa.index_head_dim}] tensor (one raw key head per token), got "
                    f"{tuple(index_k_raw.shape) if isinstance(index_k_raw, torch.Tensor) else type(index_k_raw).__name__}"
                )
            check_pool(
                index_k_raw.unsqueeze(1), "index_k_raw", h=g.qsa.index_kv_heads, page_size=ps, d=g.qsa.index_head_dim, dtype=self.act_dtype, device=h.device
            )
        elif index_k_raw is not None:
            raise ValueError(
                "index_k_raw is the raw indexer key's pool of a geometry that declares the indexer band (QsaSpec.index_band=True); this block's "
                "geometry has no band, so there is no raw key to write -- refused rather than silently ignored"
            )

    def _check_num_tokens_live(self, num_tokens_live) -> Optional[int]:
        """``execute(num_tokens_live=)`` against the declaration -- host-side, typed, callable on a declared (uncompiled) block, before
        any launch: ``None`` = the declared capacity; an int belongs to the BOUNDED packed form (``token_capacity=T_max``) and must lie in
        ``[1, T_max]`` -- ``T_live > T_max`` is the typed decline (a plan declared at ``T_max`` launches at ``T_live <= T_max``; more
        tokens are another plan); a quantized pipeline declines (its scale-factor blobs and quantize stages are sized by the capacity).
        Returns the normalised live count, or ``None`` when it equals the capacity (identical launches)."""
        if num_tokens_live is None:
            return None
        if isinstance(num_tokens_live, bool) or not isinstance(num_tokens_live, int):
            raise TypeError(f"num_tokens_live must be an int (the step's live token count T_live) or None, got {type(num_tokens_live).__name__}")
        if self.token_capacity is None:
            raise ValueError(
                f"num_tokens_live={num_tokens_live} belongs to a block declared with token_capacity=T_max (the bounded packed form: thd=True, the "
                "samples' T rows a capacity); this block was declared without it, so its lengths sum to T exactly -- refused rather than silently ignored"
            )
        if num_tokens_live < 1:
            raise ValueError(f"num_tokens_live must be >= 1 (a step with no live token launches nothing), got {num_tokens_live}")
        if num_tokens_live > self.token_capacity:
            raise ValueError(
                f"num_tokens_live={num_tokens_live} exceeds the declared token_capacity={self.token_capacity}: a plan declared at T_max launches at "
                "T_live <= T_max; a longer step needs a declaration with a larger capacity (another plan)"
            )
        if self.quant is not None:
            raise NotImplementedError(
                f"num_tokens_live={num_tokens_live} on a quantized pipeline ({type(self.quant).__name__}): its scale-factor blobs and quantize stages "
                "are sized by the declared capacity; the live token count serves the bf16 / f16 pipelines (the serving forms)"
            )
        return None if num_tokens_live == self.token_capacity else int(num_tokens_live)

    def _check_extend_args(self, h: torch.Tensor, k_cache, v_cache, block_table, kv_lens, seq_lens=None) -> None:
        """The paged forms' execute inputs against the declaration, FORM only (Rule 3): the pools match the declared samples'
        shape and strides EXACTLY (the SDPA stage compiled the paged leg against them -- a pool of another page count or layout
        is a different plan, never a view the adapter would silently re-bind); ``block_table`` is a 2-D int32 CUDA tensor
        ``[rows, >= max_pages]`` (``rows`` = ``num_sequences`` on the packed EXTEND form, ``B`` on the dense DECODE form) whose page
        axis is contiguous and whose row stride is the declared one (a row slice of a wider table binds as a view; the SDPA reads
        page ids on device); ``kv_lens`` a contiguous 1-D int32 CUDA tensor of exactly ``rows`` entries (values -- the logical
        lengths -- are device data, never read here).  On the DECODE form ``seq_lens`` is REFUSED: the dense block's self-attention
        length is a Q-side length, and over the pools the per-sequence KV length is ``kv_lens``, bound to the KV side only."""
        if not self.thd and seq_lens is not None:
            raise ValueError(
                "seq_lens with kv_source='paged' on a dense declaration (the DECODE form): the per-sequence KV length over the pools is kv_lens "
                "(bound to the KV side only; every Q row of the step is live), and seq_lens is the dense block's self-attention length -- a "
                "Q-side length -- refused rather than read as one"
            )
        for nm, x in (("k_cache", k_cache), ("v_cache", v_cache)):
            d = self._descs[nm]
            if tuple(int(s) for s in x.shape) != tuple(d.shape) or tuple(int(s) for s in x.stride()) != tuple(d.stride):
                raise ValueError(
                    f"{nm} must match sample_{nm} exactly -- shape {tuple(d.shape)} / strides {tuple(d.stride)} (the pool's page count and layout are "
                    f"compiled into the paged read); got shape {tuple(x.shape)} / strides {tuple(x.stride())}. Declare a block per pool geometry"
                )
        bt = self._descs["block_table"]
        b, max_pages, row_stride = (self.num_sequences if self.thd else self.batch), int(bt.shape[1]), int(bt.stride[0])
        label = "num_sequences" if self.thd else "B"
        dev = torch.device(h.device)
        if not isinstance(block_table, torch.Tensor) or block_table.dtype != torch.int32 or block_table.dim() != 2:
            got = f"{block_table.dtype} {tuple(block_table.shape)}" if isinstance(block_table, torch.Tensor) else type(block_table).__name__
            raise ValueError(f"block_table must be a 2-D int32 tensor [{label}={b}, >= max_pages={max_pages}], got {got}")
        if self.sparse_decode and (
            tuple(int(s) for s in block_table.shape) != (b, max_pages) or tuple(int(s) for s in block_table.stride()) != (row_stride, 1)
        ):
            # the sparse core binds the WHOLE declared table (shape and strides exactly, like the pools): a wider table or another row
            # stride is not a view it can take -- declare the block on the table it will read
            raise ValueError(
                f"block_table must be exactly the declared sample_block_table under QsaSpec over the pools -- shape [B={b}, max_pages={max_pages}] "
                f"with strides ({row_stride}, 1) (the sparse core binds the whole table); got shape {tuple(block_table.shape)} strides "
                f"{tuple(block_table.stride())}"
            )
        if int(block_table.shape[0]) != b or int(block_table.shape[1]) < max_pages:
            raise ValueError(
                f"block_table must be [{label}={b}, >= max_pages={max_pages}] (the declared sample_block_table's page axis bounds the logical "
                f"S_kv = {max_pages * self.paged_kv_page_size}); got {tuple(block_table.shape)}"
            )
        if int(block_table.stride(1)) != 1 or (b > 1 and int(block_table.stride(0)) != row_stride):
            raise ValueError(
                f"block_table strides must be the declared (row stride {row_stride}, 1) -- the page axis contiguous, the row stride as sample_block_table "
                f"declared it (a row slice of a wider table); got {tuple(block_table.stride())}"
            )
        if not block_table.is_cuda or block_table.device != dev:
            raise ValueError(f"block_table must live on h's device {dev}, got {block_table.device}")
        ok = (
            isinstance(kv_lens, torch.Tensor)
            and kv_lens.dtype == torch.int32
            and kv_lens.dim() == 1
            and kv_lens.is_contiguous()
            and int(kv_lens.numel()) == b
            and kv_lens.is_cuda
            and kv_lens.device == dev
        )
        if not ok:
            got = (
                f"dtype {kv_lens.dtype}, shape {tuple(kv_lens.shape)}, strides {tuple(kv_lens.stride())}, device {kv_lens.device}"
                if isinstance(kv_lens, torch.Tensor)
                else type(kv_lens).__name__
            )
            raise ValueError(
                f"kv_lens must be a contiguous 1-D int32 CUDA tensor of {b} entries on {dev} (each sequence's logical KV length INCLUDING this "
                f"step's tokens; 0 = a padded entry), got {got}"
            )

    # -- the serving forms: the declaration table and the envelope ------------

    def _check_serving_declaration(
        self,
        kv_source,
        token_capacity,
        quant,
        sample_k_cache,
        sample_v_cache,
        sample_block_table,
        device,
        max_kv_len=None,
        decode_split_kv=None,
        decode_route_policy=None,
        decode_q_tile_max=None,
        sparse_tile_form="auto",
        sparse_sub_tiles_per_cta=None,
    ):
        """``kv_source`` / ``token_capacity`` at declaration -- the argument contract first (``kv_source`` one of
        :data:`_KV_SOURCES`; ``token_capacity`` None or a positive int equal to the samples' packed row count), then the
        typed declines of :data:`_EXTEND_DECLINES` in order (the ONE table :meth:`serving_envelope` publishes for both paged
        forms), then the three samples' contract under ``kv_source="paged"`` (REQUIRED there, REFUSED otherwise; each pool as
        :func:`check_pool` spells it, in the activation dtype, on ``device``, K and V of the SAME HND / NHD kind; the block table a
        2-D int32 ``[rows, max_pages >= 1]`` tensor with a contiguous page axis, ``rows`` = ``num_sequences`` on the packed EXTEND
        form and ``B`` on the dense DECODE form), then the DECODE form's own facts (``max_kv_len``: default ``max_pages * P``, never
        past the table's reach; ``decode_split_kv``: None = the decode split model at THIS device's SM count -- 1 on a non-CUDA
        device, which ``check_support`` declines anyway -- or an int >= 1; the route from :func:`decode_route`), both REFUSED on the
        packed form and under ``"self"``.  Device-free: a test pins every row anywhere.  Returns ``(kv_source, token_capacity,
        route, max_kv_len, split)`` -- the last three None off the DECODE form.  Runs AFTER the write-through's own validator, so
        ``quant`` / ``save_for_backward`` with the attribute are heard from there."""
        from .cache_write_contract import check_pool

        if not isinstance(kv_source, str) or kv_source not in _KV_SOURCES:
            raise ValueError(
                f"kv_source must be one of {_KV_SOURCES} ('self' = the block's own K / V, 'paged' = the EXTEND / DECODE forms over the pools), got {kv_source!r}"
            )
        t = self.batch * self.seq_len
        if token_capacity is not None:
            if isinstance(token_capacity, bool) or not isinstance(token_capacity, int):
                raise TypeError(f"token_capacity must be an int (the packed token capacity T_max) or None, got {type(token_capacity).__name__}")
            if token_capacity < 1:
                raise ValueError(f"token_capacity must be >= 1, got {token_capacity}")
        decl = _ServingDecl(
            kv_source=kv_source,
            thd=self.thd,
            paged_kv_page_size=self.paged_kv_page_size,
            token_capacity=token_capacity,
            save_for_backward=self.save_for_backward,
            return_lse=self.return_lse,
            geometry=self.geom,
            s_q=0 if self.thd else self.seq_len,
            fuse_gate=self.fuse_gate,
        )
        hit = _first_serving_decline(decl)
        if hit is not None:
            _name, exc, msg = hit
            raise exc(msg)
        if token_capacity is not None and token_capacity != t:
            raise ValueError(
                f"token_capacity={token_capacity} must equal the samples' packed token rows T={t}: the samples ARE the capacity (declare sample_h / "
                "sample_cos / sample_sin / sample_out with T_max rows); a step then hands lengths summing to T_live <= T_max"
            )
        decode_extras = [
            nm
            for nm, x in (
                ("max_kv_len", max_kv_len),
                ("decode_split_kv", decode_split_kv),
                ("decode_route_policy", decode_route_policy),
                ("decode_q_tile_max", decode_q_tile_max),
            )
            if x is not None
        ]
        samples = (("sample_k_cache", sample_k_cache), ("sample_v_cache", sample_v_cache), ("sample_block_table", sample_block_table))
        if kv_source != KV_SOURCE_PAGED:
            given = [nm for nm, x in samples if x is not None] + decode_extras
            if given:
                raise ValueError(
                    f"{', '.join(given)}: the pool / block-table samples and the DECODE form's facts belong to the EXTEND form or the DECODE form "
                    f"(kv_source='paged'); this block was declared kv_source={kv_source!r} and never reads the pools -- refused rather than silently ignored"
                )
            return kv_source, token_capacity, None, None, None
        if self.thd and decode_extras:
            raise ValueError(
                f"{', '.join(decode_extras)}: the DECODE form's facts (a dense [B, S_q] declaration over the pools); the packed EXTEND form addresses "
                "max_pages x P keys per sequence and runs the paged THD body unsplit -- refused rather than silently ignored"
            )
        g, ps = self.geom, self.paged_kv_page_size
        rows = self.num_sequences if self.thd else self.batch
        label = "num_sequences" if self.thd else "B"
        missing = [nm for nm, x in samples if x is None]
        if missing:
            raise ValueError(
                f"kv_source='paged' needs {', '.join(missing)} at declaration: sample_k_cache / sample_v_cache [num_pages, {g.h_kv}, {ps}, {g.d_head}] "
                f"{self.kv_cache_dtype} (the pools' shape / dtype / strides -- HND compact, NHD by strides, or the two halves of a unified row; the "
                f"layout is compiled in) and sample_block_table [{label}={rows}, max_pages] int32 (its row stride is compiled in too)"
            )
        for nm, x in samples[:2]:
            check_pool(x, nm, h=g.h_kv, page_size=ps, d=g.d_head, dtype=self.kv_cache_dtype, device=device)
        if tuple(sample_k_cache.shape) != tuple(sample_v_cache.shape):
            raise ValueError(
                f"sample_k_cache {tuple(sample_k_cache.shape)} and sample_v_cache {tuple(sample_v_cache.shape)} must have the same [num_pages, H_kv, P, D] shape"
            )
        if (sample_k_cache.stride(2) < sample_k_cache.stride(1)) != (sample_v_cache.stride(2) < sample_v_cache.stride(1)):
            raise ValueError(
                "sample_k_cache and sample_v_cache must share one layout kind (both HND -- the page-row stride below the head stride -- or both "
                f"NHD): got K strides {tuple(sample_k_cache.stride())}, V strides {tuple(sample_v_cache.stride())}"
            )
        bt = sample_block_table
        if not isinstance(bt, torch.Tensor) or bt.dtype != torch.int32 or bt.dim() != 2:
            got = f"{bt.dtype} {tuple(bt.shape)}" if isinstance(bt, torch.Tensor) else type(bt).__name__
            raise ValueError(f"sample_block_table must be a 2-D int32 tensor [{label}={rows}, max_pages >= 1], got {got}")
        if int(bt.shape[0]) != rows or int(bt.shape[1]) < 1:
            raise ValueError(
                f"sample_block_table must be [{label}={rows}, max_pages >= 1] ({label} = one row per sequence of the declaration; "
                f"max_pages x {ps} = the logical S_kv the block tables may address), got {tuple(bt.shape)}"
            )
        if int(bt.stride(1)) != 1:
            raise ValueError(f"sample_block_table's page axis must be contiguous (stride 1; the row stride is free), got strides {tuple(bt.stride())}")
        if bt.device != torch.device(device):
            raise ValueError(f"sample_block_table must live on sample_h's device {device}, got {bt.device}")
        if self.thd:
            return kv_source, token_capacity, None, None, None
        # The DECODE form: the KV bound, the route (served or declined -- the decode_rows row above already raised past DECODE_MAX_ROWS;
        # this is the record the stage pins its compiled module against), the split.
        capacity = int(bt.shape[1]) * ps
        if max_kv_len is None:
            max_kv_len = capacity
        if isinstance(max_kv_len, bool) or not isinstance(max_kv_len, int) or max_kv_len < 1:
            raise ValueError(f"max_kv_len must be a positive int (the logical KV bound kv_lens[b] <= max_kv_len), got {max_kv_len!r}")
        if max_kv_len > capacity:
            raise ValueError(
                f"max_kv_len={max_kv_len} exceeds the block table's reach max_pages * P = {capacity}: a sequence longer than its table row addresses cannot be read"
            )
        if decode_split_kv is not None and (isinstance(decode_split_kv, bool) or not isinstance(decode_split_kv, int) or decode_split_kv < 1):
            raise ValueError(f"decode_split_kv must be None (the decode split model) or an int >= 1, got {decode_split_kv!r}")
        if self.qsa is not None:
            # The BLOCK-SPARSE decode / extend-with-prefix mode: no dense route (the sparse core runs one work item per query row), the
            # WHOLE table bound (the core bulk-reads it: the sample must be contiguous), the split against the item's tile count
            # (ceil((top_k + 1) / 32); a larger split has an empty chunk on every item -- the adapter's own bound, read here so the
            # declaration declines) or the fill rule.
            if not bt.is_contiguous():
                raise ValueError(
                    f"sample_block_table must be CONTIGUOUS under QsaSpec over the pools (the sparse core binds the whole [B, max_pages] table; a row "
                    f"slice of a wider table is not a view it can take -- declare the block on the table it will read), got strides {tuple(bt.stride())}"
                )
            max_tiles = -(-(self.qsa.top_k + 1) // 32)
            dev = torch.device(device)
            sm = torch.cuda.get_device_properties(dev).multi_processor_count if (dev.type == "cuda" and torch.cuda.is_available()) else 0
            # THE BODY of the sparse core (the constructor's note on `sparse_tile_form`): resolved HERE, before the split, because the split
            # is the body's -- the lean tile's splits SELECT its k.  The admissibility and the placement rule are the heuristics' (one copy;
            # the adapter runs the same functions at check_support and `_SparseSdpa` pins its readback against this resolution).
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import sparse_tile_rules as _H

            top_k = self.qsa.top_k
            reason = _H.sparse_lean_tile_decline(
                s_q=self.seq_len,
                group=g.gqa_ratio,
                top_k=top_k,
                thd=False,
                bottom_right=True,
                list_per_sequence=self._qsa_shared_list_derived(kv_source),
                seq_kv_lens_present=True,
                # the POOLS' dtype, not the activations' alone: over e4m3 pools no half body runs (`kv_cache_e4m3` is settled above)
                half_dtype=self.dtype in (torch.bfloat16, torch.float16) and not self.kv_cache_e4m3,
            )
            if self.kv_cache_e4m3:
                # THE e4m3 POOLS (kv_cache_dtype): stage (4) is the gathered cast + the DENSE decode tile over the scratch (`upcast_decode`),
                # NO body of the sparse core -- the body knob has nothing to honour: an explicit body is a typed decline naming the arm (the
                # lean body's own e4m3 text; the core's named here), never the readback of a kernel that does not run, and "auto" reads back
                # NO body (`sparse_tile_form` / `sparse_tile` None); the split below is the core's rule, which the e4m3 serving then settles
                # unsplit (`_check_kv_cache_dtype_serving`).
                if sparse_tile_form == _H.SPARSE_TILE_FORM_LEAN:
                    raise NotImplementedError(f"sparse_tile_form='lean' -- {reason}")
                if sparse_tile_form == _H.SPARSE_TILE_FORM_TCGEN05:
                    raise NotImplementedError(
                        f"sparse_tile_form='tcgen05' over e4m3 pools (kv_cache_dtype={KV_CACHE_E4M3}): the block-sparse decode mode over e4m3 pools runs "
                        "the gathered cast + the dense decode tile, no body of the sparse core -- leave sparse_tile_form='auto' (sparse_tile reads None there)"
                    )
                form = None
            elif sparse_tile_form == _H.SPARSE_TILE_FORM_AUTO:
                form = _H.sparse_tile_form_auto(
                    batch=self.batch,
                    h_kv=g.h_kv,
                    s_q=self.seq_len,
                    top_k=top_k,
                    device_cc=_device_cc_of(device),
                    admissible=reason is None,
                    split_pinned=decode_split_kv,
                )
            else:
                form = sparse_tile_form
            if form == _H.SPARSE_TILE_FORM_LEAN:
                if reason is not None:
                    raise NotImplementedError(f"sparse_tile_form='lean' -- {reason}")
                from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.sparse_tile_rules import lean_sparse_n_sub, lean_sparse_split

                n_sub, splits, k_max = lean_sparse_n_sub(top_k), _H.sparse_lean_splits(top_k), _H.sparse_lean_max_sub_tiles_per_cta(top_k)
                if decode_split_kv is None:
                    k = (
                        int(sparse_sub_tiles_per_cta)
                        if sparse_sub_tiles_per_cta is not None
                        else _H.sparse_lean_sub_tiles_per_cta(batch=self.batch, h_kv=g.h_kv, s_q=self.seq_len, group=g.gqa_ratio, top_k=top_k, sm_count=sm)
                    )
                else:
                    if int(decode_split_kv) not in splits:
                        raise NotImplementedError(
                            f"decode_split_kv={decode_split_kv} with sparse_tile_form='lean': the lean tile's splits at top_k {top_k} are {splits} (ceil(n_sub / k) "
                            f"at n_sub = {n_sub}; the split SELECTS k, the sub-tiles per CTA) -- pin one of them, or None for the default k; a split of 1 is the "
                            "tcgen05 core's (the lean tile always splits), as is any other value (sparse_tile_form='tcgen05' / 'auto')"
                        )
                    # the explicit k is THE k when given (the adapter's rule: ceil(n_sub / split) is not its inverse); the split must be ITS lean split
                    k = int(sparse_sub_tiles_per_cta) if sparse_sub_tiles_per_cta is not None else -(-n_sub // int(decode_split_kv))
                    if sparse_sub_tiles_per_cta is not None and lean_sparse_split(top_k, k) != int(decode_split_kv):
                        raise ValueError(
                            f"sparse_sub_tiles_per_cta={k} and decode_split_kv={decode_split_kv} disagree: the lean split of k {k} at top_k {top_k} is "
                            f"ceil({n_sub} / {k}) = {lean_sparse_split(top_k, k)} -- pin one of the two"
                        )
                if not 1 <= k <= k_max:
                    raise NotImplementedError(
                        f"sparse_sub_tiles_per_cta={k} must be in [1, {k_max}] at top_k {top_k} (n_sub {n_sub}: the lean tile's sub-tile loop is unrolled at trace "
                        "time, and k = n_sub would be one chunk = split 1 -- the lean tile always splits)"
                    )
                # the lean split ceil(n_sub / k) at this k through the ONE function the test pins too (never a second spelling here)
                split = sparse_decode_split_choice(
                    batch=self.batch,
                    h_kv=g.h_kv,
                    s_q=self.seq_len,
                    top_k=top_k,
                    sm_count=sm,
                    tile_form=_H.SPARSE_TILE_FORM_LEAN,
                    sub_tiles_per_cta=k,
                    group=g.gqa_ratio,
                )
                self.sparse_sub_tiles_per_cta = k
            else:
                if decode_split_kv is None:
                    split = sparse_decode_split_choice(batch=self.batch, h_kv=g.h_kv, s_q=self.seq_len, top_k=top_k, sm_count=sm)
                else:
                    split = int(decode_split_kv)
                    if split > max_tiles:
                        raise NotImplementedError(
                            f"decode_split_kv={split} exceeds the sparse core's item tile count ceil((top_k + 1) / 32) = {max_tiles} at top_k={top_k}: a larger "
                            "split has an empty chunk (a dead work item) on every item -- pass a split in [1, that count], or None for the fill rule"
                        )
                self.sparse_sub_tiles_per_cta = None
            self.sparse_tile_form = form
            self.sparse_tile = None if form is None else (_H.SPARSE_TILE_LEAN if form == _H.SPARSE_TILE_FORM_LEAN else _H.SPARSE_TILE_TCGEN05)
            return kv_source, token_capacity, None, int(max_kv_len), split
        route = decode_route(g, self.seq_len, policy=self.decode_route_policy, max_q_tile=self.decode_q_tile_max, batch=self.batch)
        if not route.served:
            raise NotImplementedError(f"kv_source='paged': {route.reason}")
        if route.prefill_tile and decode_split_kv is not None and int(decode_split_kv) > 1:
            raise NotImplementedError(
                f"decode_split_kv={decode_split_kv} on the DECODE form's prefill route (S_q={self.seq_len} at group {route.group} rides the paged d256 "
                f"prefill tile with the group packed, which runs unsplit): the split-KV plan is the decode tile's -- pass decode_split_kv=None or 1, or "
                f"declare decode_route_policy={DECODE_ROUTE_TOKEN_UNITS!r} for the decode tile in token units"
            )
        if decode_split_kv is None:
            dev = torch.device(device)
            sm = torch.cuda.get_device_properties(dev).multi_processor_count if (dev.type == "cuda" and torch.cuda.is_available()) else 0
            split = decode_split_choice(route, batch=self.batch, h_kv=g.h_kv, max_kv_len=max_kv_len, sm_count=sm, gated=self.fuse_gate, h_q=g.h_q)
        else:
            split = int(decode_split_kv)
        return kv_source, token_capacity, route, int(max_kv_len), split

    @classmethod
    def serving_envelope(
        cls,
        geometry: GatedAttentionBlockGeometry,
        dtype: torch.dtype,
        device,
        *,
        device_cc: Optional[Tuple[int, int]] = None,
        decode_route_policy: Optional[str] = None,
        decode_q_tile_max: Optional[int] = None,
    ) -> ServingEnvelope:
        """The static serving envelope for ``(geometry, dtype, device)`` -- what a serving stack's INIT-time dispatch predicate
        reads instead of catching the block's exceptions.  Derived from the SAME code the constructor and ``check_support`` run:
        the device fact (Rubin, cc 10.7, the one arch every stage binds), the CuTe DSL facts (the version floor and the Rubin
        target, exactly as the stages' ``check_support`` gates them), the dtype (bf16 / f16 activations), and the geometry rows of
        :data:`_EXTEND_DECLINES` (``diagonal``, ``window``, ``sparse``, ``head_dim``) evaluated on ``geometry`` for EACH paged form.
        The declaration rows (including ``return_lse`` and the EXTEND-only ``fuse_gate`` refusal) are listed in each form's
        ``declines``; a QsaSpec geometry meets the table's ``sparse`` row on both forms (:data:`_QSA_THD_DECLINE` only
        through a record without the packed arm).  Thus
        ``extend.served`` / ``decode.served`` are True iff a block declared as ``requires`` spells it passes its declaration and
        ``check_support`` here, and each form's ``declines`` lists every typed decline it raises at declaration with its message;
        ``decode.routes`` names the kernel per ``S_q`` (:func:`decode_route`).  ``forms`` holds the served forms.  No allocation, no
        launch; safe to call on any device (``served`` is simply False off Rubin; a non-CUDA ``device`` reads as no CUDA device).
        ``device_cc`` (appended) overrides the device's compute capability for a host-side reading of another part's envelope;
        ``decode_route_policy`` (appended) is the policy ``decode.routes`` / ``decode.route()`` are read under (``None`` = the default,
        :data:`DEFAULT_DECODE_ROUTE_POLICY`; ``decode.policy`` names it); ``decode_q_tile_max`` (appended) the width cap they are read
        under (``None`` = the whole single-unit table; ``decode.q_tile_max`` names it); ``decode.pdl_default`` (appended) is the AUTO
        reading of ``use_pdl`` on this part for a served form (:data:`DECODE_PDL_DEFAULT_CCS` on the dense form,
        :data:`SPARSE_DECODE_PDL_DEFAULT_CCS` under ``sparse_core`` -- empty until measured; False where ``decode.served`` is False) -- the
        per-step reading is ``decode.pdl_default and decode.route(s_q, batch).pdl_capable``; ``decode.pdl_stages`` names the switched stage
        kinds on both forms."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

        policy = _resolve_decode_route_policy(decode_route_policy)
        q_tile_max = _resolve_decode_q_tile_max(decode_q_tile_max)
        cc = tuple(int(x) for x in device_cc) if device_cc is not None else _device_cc_of(device)
        dsl_reason = cutedsl_requirement_error("the gated attention block's serving forms")
        if dsl_reason is None and cc == _SM107_CC:
            dsl_reason = cutedsl_arch_requirement_error(cc)
        common = None
        if cc != _SM107_CC:
            common = (
                f"gated_attention_block targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only; found {'no CUDA device' if cc is None else f'SM{cc[0]}{cc[1]}'}"
            )
        elif dsl_reason is not None:
            common = dsl_reason
        elif dtype not in (torch.bfloat16, torch.float16):
            common = f"the serving forms over the pools serve bf16 / f16 activations and pools only (quant=None), got {dtype}"

        def probe(thd: bool, s_q: int):
            if common is not None:
                return common
            if thd and geometry.qsa is not None and not _sparse_record().thd:
                # a record without the packed arm: the THD-only form meets the sparse declaration's own row before any row of the
                # table (_check_qsa_declaration runs first); with the arm claimed (the shipped record) the table's `sparse` row speaks
                return _QSA_THD_DECLINE
            hit = _first_serving_decline(
                _ServingDecl(
                    kv_source=KV_SOURCE_PAGED,
                    thd=thd,
                    paged_kv_page_size=SERVING_PAGE_SIZES[0],
                    token_capacity=None,
                    save_for_backward=False,
                    return_lse=False,
                    geometry=geometry,
                    s_q=s_q,
                )
            )
            return None if hit is None else hit[2]

        extend_reason = probe(True, 0)
        extend = ExtendEnvelope(
            served=extend_reason is None,
            reason=extend_reason,
            requires=_EXTEND_REQUIRES,
            dtypes=(torch.bfloat16, torch.float16),
            page_sizes=SERVING_PAGE_SIZES,
            pool_layouts=SERVING_POOL_LAYOUTS,
            declines={name: msg for name, _exc, _pred, msg in _EXTEND_DECLINES},  # the whole table: every row a declaration can hit
            pdl_stages=EXTEND_PDL_STAGES,
        )
        decode_reason = probe(False, 1)
        has_group = int(geometry.h_kv) >= 1 and int(geometry.h_q) % int(geometry.h_kv) == 0
        sparse = geometry.qsa is not None  # stage (4) over the pools is the index-list sparse core: its own requires, no dense route
        indexer = sparse and geometry.qsa.index_source == "indexer"
        decode_declines = {name: msg for name, _exc, _pred, msg in _EXTEND_DECLINES if name in _DECODE_DECLINE_NAMES}
        if sparse:
            # the block-sparse decode mode's own rows (the list form, the in-block indexer over the pools): the constructor raises these
            # very texts, so a stack's dispatch predicate and the block's declines cannot drift
            decode_declines.update(_qsa_decode_decline_texts(geometry, dtype))
        decode = DecodeEnvelope(
            served=decode_reason is None,
            reason=decode_reason,
            requires=_SPARSE_DECODE_REQUIRES if sparse else _DECODE_REQUIRES,
            dtypes=(torch.bfloat16, torch.float16),
            page_sizes=SERVING_PAGE_SIZES,
            pool_layouts=SERVING_POOL_LAYOUTS,
            declines=decode_declines,
            max_s_q=DECODE_MAX_S_Q,
            max_rows=DECODE_MAX_ROWS,
            # the declared envelope and its first decline, under the policy asked for
            routes=(
                tuple(decode_route(geometry, s, policy=policy, max_q_tile=q_tile_max) for s in range(1, DECODE_MAX_S_Q + 2))
                if (has_group and not sparse)
                else ()
            ),
            _geometry=geometry,
            sparse_core=sparse,
            # the e4m3 cache (the gathered cast + the decode tile) is a form of the block-sparse decode mode with bf16 activations only --
            # under caller lists: its composition with the in-block indexer waits for the sparse core's in-kernel dequant arm
            kv_cache_dtypes=(dtype,) + ((KV_CACHE_E4M3,) if (sparse and dtype == torch.bfloat16 and not indexer) else ()),
            policy=policy,
            claimed_max_s_q=DECODE_CLAIMED_MAX_S_Q,
            q_tile_max=q_tile_max,
            # the stage kinds the switch reaches on this form: the dense chain's (stage (4) on a decode-tile route) and, under the sparse
            # core on a caller list, the same four kinds (the sparse core + its combine are the `sdpa` kind; appended 2026-10-21); NONE under
            # the in-block indexer, where the explicit switch is a typed decline (its kernels would take it unmeasured)
            pdl_stages=() if indexer else DECODE_PDL_STAGES,
            # the AUTO reading of the switch on this part for a SERVED form (the measured default on cc 10.7 for the dense form; the sparse
            # core's own tuple, empty until measured; never under the indexer): False wherever the form is not served (`decode_reason`:
            # another part, an unserved dtype, a geometry row, no DSL target); the per-step reading is `pdl_default and route(s_q, batch).pdl_capable`
            pdl_default=(decode_reason is None) and not indexer and cc in (SPARSE_DECODE_PDL_DEFAULT_CCS if sparse else DECODE_PDL_DEFAULT_CCS),
            # the list form (QsaSpec.index_list_form) at this geometry: the indexer over the pools selects per row; caller lists take the
            # shared step-0 list at S_q <= the record's cap (requires spells it) or the per-row list at any S_q
            index_list_forms=((QSA_INDEX_LIST_FORM_PER_ROW,) if indexer else (QSA_INDEX_LIST_FORM_SHARED, QSA_INDEX_LIST_FORM_PER_ROW)) if sparse else (),
            index_list_form_default=(QSA_INDEX_LIST_FORM_PER_ROW if indexer else QSA_INDEX_LIST_FORM_SHARED) if sparse else "",
            **cls._indexer_envelope_fields(geometry, dtype, cc, decode_reason is None),
            **cls._sparse_tile_envelope_fields(sparse and decode_reason is None, cc),
        )
        forms = tuple(nm for nm, env in (("extend", extend), ("decode", decode)) if env.served)
        return ServingEnvelope(
            device_cc=cc,
            cutedsl_ok=dsl_reason is None,
            cutedsl_reason=dsl_reason,
            forms=forms,
            extend=extend,
            decode=decode,
        )

    @staticmethod
    def _sparse_tile_envelope_fields(sparse_served: bool, cc) -> dict:
        """The lean tile's rows of :class:`DecodeEnvelope` for a SERVED block-sparse decode form: the bodies this part serves by
        ``kernel_template`` (the tcgen05 core everywhere; the lean tile on cc 10.7, its one part -- ``heuristics._SPARSE_LEAN_TILE_RULE_CC``),
        the AUTO pin's reading on this part (``SPARSE_LEAN_TILE_DEFAULT_CCS``: cc 10.7, the measured part) and the part itself for the per-cell
        :meth:`DecodeEnvelope.sparse_tile`; empty off the sparse form."""
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import sparse_tile_rules as _H

        if not sparse_served:
            return dict(sparse_tiles=(), sparse_tile_lean_default=False, device_cc=cc)
        tiles = (_H.SPARSE_TILE_TCGEN05,) + ((_H.SPARSE_TILE_LEAN,) if cc is not None and tuple(cc)[:2] == _H._SPARSE_LEAN_TILE_RULE_CC else ())
        return dict(sparse_tiles=tiles, sparse_tile_lean_default=bool(cc is not None and tuple(cc)[:2] in _H.SPARSE_LEAN_TILE_DEFAULT_CCS), device_cc=cc)

    @staticmethod
    def _indexer_envelope_fields(geometry: GatedAttentionBlockGeometry, dtype: torch.dtype, cc, decode_served: bool) -> dict:
        """The in-block indexer's rows of :class:`DecodeEnvelope` for a QsaSpec geometry (empty for a dense one) -- every value READ off the
        DSL-free host modules (``cudnn.QSA.selection`` / ``cudnn.QSA.selection_paged``) and the block's own rules, never a second literal: the index sources
        the geometry serves at decode (the indexer needs bf16 activations and the scorer's head dim: the ``indexer_f16`` / ``indexer_head_dim``
        rows), the indexer cache dtypes at decode (bf16 and e4m3 -- the e4m3 compressed-key cache served since 2026-10-21 on every part the
        indexer is served on: the fp8 ``cvt`` and ``mma.sync`` are sm_89+ instructions and the block serves cc 10.x), the scorer's head dim / KV
        heads / block size, the ``top_k`` range, the query-row envelope, the ring's extra rows, the pages the paged scorer addresses
        (``SERVING_PAGE_SIZES`` through its own validator), and the AUTO reading of ``fuse_prepare`` on this part."""
        if geometry.qsa is None:
            return {}
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import INDEXER_SCORER_HEAD_DIM
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection_paged import MAX_QUERY_ROWS, validate_params

        q = geometry.qsa
        indexer_served = dtype == torch.bfloat16 and int(q.index_head_dim) == INDEXER_SCORER_HEAD_DIM

        def page_ok(p: int) -> bool:
            try:
                validate_params(h_i=q.index_heads, s_q=1, page_size=int(p), units_per_warp=1)
            except ValueError:
                return False
            return True

        return dict(
            index_sources=("caller", "indexer") if indexer_served else ("caller",),
            index_cache_dtypes=(torch.bfloat16, INDEX_CACHE_E4M3) if indexer_served else (),
            index_head_dims=(INDEXER_SCORER_HEAD_DIM,),
            index_kv_heads=(1,),  # QsaSpec.validate pins one raw key head (the checkpoint's own validator)
            index_block_sizes=(QSA_BLOCK_SIZE,),
            index_top_k_range=(QSA_TOP_K_ALIGN, QSA_TOP_K_MAX, QSA_TOP_K_ALIGN),
            index_query_rows_max=int(MAX_QUERY_ROWS),
            index_state_rows_extra=_index_state_ring_extra_rows(q.block_size),
            index_page_sizes=tuple(p for p in SERVING_PAGE_SIZES if page_ok(p)),
            fuse_prepare_default=bool(decode_served and indexer_served and q.index_source == "indexer" and cc in SPARSE_INDEXER_FUSE_PREPARE_DEFAULT_CCS),
        )

    def index_k_raw(self, workspace: torch.Tensor) -> torch.Tensor:
        """The RAW indexer key the last ``execute`` left in ``workspace``, as a ``[B, S, index_kv_heads, index_head_dim]``
        VIEW of the stage-(1) slab (:func:`index_k_raw_view` at this block's own slab offset; no copy, no allocation).
        Valid after an ``execute`` with that workspace until the next one overwrites the slab.  ``ValueError`` on a
        block whose geometry has no indexer band, or on a workspace too small to hold the slab."""
        g = self.geom
        if not g.index_band:
            raise ValueError("index_k_raw: this block's geometry declares no indexer band (GatedAttentionBlockGeometry.qsa.index_band)")
        lay = self._layout()
        t = self.batch * self.seq_len
        need = lay.proj + t * g.n_qkvg * _itemsize(self.act_dtype)
        if workspace.numel() < need:
            raise ValueError(f"workspace is {workspace.numel()} bytes; the stage-(1) slab ends at byte {need}")
        return index_k_raw_view(_view(workspace, lay.proj, (t, g.n_qkvg), self.act_dtype), g, self.batch, self.seq_len)

    # -- support ------------------------------------------------------------

    def check_support(self) -> bool:
        """Validate the declaration, then ask every stage in turn.

        Whatever this ACCEPTS, the stages must address natively — acceptance is a
        promise about the execute path, not about what the adapter can patch up
        (Rule 2). A ``ValueError`` escaping a stage's own config builder after
        this returned True is a bug HERE, not user error.
        """
        self._check_declaration()
        for st in self._stages:
            st.check_support()
        self._is_supported = True
        return True

    def _expected_weight_dtypes(self) -> dict:
        """``{"w_qkvg": dtype, "w_o": dtype}`` -- what each weight descriptor must carry.  ``h``'s dtype
        (bf16 / f16, or e4m3 under a QuantSpec / MxQuantSpec) unless the spec names a packed e2m1 weight:
        ``MxQuantSpec.w_qkvg_dtype`` for ``w_qkvg`` (the mixed block-scale row).  ONE place for the
        per-weight contract, so a further fp4 weight is one more entry here, not a second loop."""
        w_qkvg = self.quant.w_qkvg_dtype if self.mxfp8 else self.dtype
        w_o = _FP4_X2 if self.o_fp4 is not None else self.dtype
        return {"w_qkvg": w_qkvg, "w_o": w_o}

    @staticmethod
    def _fp4_storage_shape(desc: TensorDesc) -> Tuple[int, ...]:
        """The STORAGE shape of a ``float4_e2m1fn_x2`` descriptor.  ``_make_tensor_desc`` reports the LOGICAL
        shape of a packed-fp4 tensor (the innermost, stride-1 extent DOUBLED -- pinned by
        ``test_make_tensor_desc_reports_the_logical_k_of_an_fp4x2_tensor``); this halves that one axis back,
        so the caller-facing check speaks the shape the caller allocated (``[N, K // 2]``)."""
        inner = desc.stride_order[0]  # ascending stride: [0] is the stride-1 axis the descriptor doubled
        return tuple(int(n) // 2 if i == inner else int(n) for i, n in enumerate(desc.shape))

    def _check_weight(self, nm: str, rows: int, k: int, expect: torch.dtype) -> None:
        """One weight, dtype THEN shape, typed.  A packed e2m1 weight is checked against its STORAGE shape
        ``(rows, k // 2)``; every other dtype against ``(rows, k)`` exactly as before."""
        d = self._descs[nm]
        fp4_expected = _FP4_X2 is not None and expect == _FP4_X2
        if not fp4_expected:
            self._check_tensor_shape(d, (rows, k), nm)
            if d.dtype == _FP4_X2:
                need = (
                    "declare it with MxQuantSpec(w_qkvg_dtype=torch.float4_e2m1fn_x2)"
                    if nm == "w_qkvg"
                    else "an e2m1 w_o is the block's fp4 O output mode: declare it with MxQuantSpec(o_fp4=Fp4Format.NVFP4 | MXFP4) plus sample_w_o_sf"
                )
                raise ValueError(
                    f"{nm} is torch.float4_e2m1fn_x2 but this block expects {expect}: a packed e2m1 {nm} rides the block-scale "
                    f"MXFP8 pipeline only -- {need}; a QuantSpec / bf16 block has no GEMM row for it"
                )
            if d.dtype != expect:
                raise ValueError(f"{nm} must have h's dtype {expect}, got {d.dtype}")
            return
        if d.dtype != _FP4_X2:
            hint = (
                " -- torch 2.13 can VIEW but not cast to fp4: hand over the packed codes as storage.view(torch.float4_e2m1fn_x2), not uint8"
                if d.dtype == torch.uint8
                else ""
            )
            field = "MxQuantSpec.w_qkvg_dtype" if nm == "w_qkvg" else "MxQuantSpec.o_fp4"
            raise ValueError(f"{nm} must be torch.float4_e2m1fn_x2 ({field} names an e2m1 {nm}), got {d.dtype}{hint}")
        storage = self._fp4_storage_shape(d)
        if storage != (rows, k // 2):
            raise ValueError(
                f"{nm} is fp4 storage {storage} (logical {tuple(int(x) for x in d.shape)}); a packed e2m1 {nm} is [{rows}, {k} // 2 = {k // 2}] "
                f"-- two codes per byte along K, LOW nibble = even k.  A LOGICAL [{rows}, {k}] fp4 tensor holds twice the data the GEMM declares."
            )

    def _check_declaration(self) -> None:
        """Everything ``check_support`` verifies BEFORE asking the stages (geometry, every descriptor's shape
        and dtype, the MXFP8 blob contract) -- device-agnostic, so a test can pin the caller contract on
        any GPU while the stages' own ``check_support`` still gates the arch."""
        self.geom.validate()
        if self.save_for_backward and not self.return_lse:
            raise ValueError("save_for_backward requires the LSE: the SDPA backward cannot run without it")
        g = self.geom
        want = self._expected_weight_dtypes()
        self._check_weight("w_qkvg", g.n_qkvg, g.d_model, want["w_qkvg"])
        self._check_weight("w_o", g.d_model, g.h_q * g.d_head, want["w_o"])
        # Under qk_norm=False the two norm-weight descriptors are None (the
        # constructor checked both directions), so they are skipped here: the
        # shape check would pass silently and the dtype loop would raise an
        # untyped AttributeError.
        norm_names = ("w_q_norm", "w_k_norm") if g.qk_norm else ()
        for nm in norm_names:
            self._check_tensor_shape(self._descs[nm], (g.d_head,), nm)
        # under rope_position_tables_only the per-token tables are refused at execute (every rotation reads cos_pos / sin_pos), so a declaration
        # may omit their samples; a sample given is held to the form like every other
        table_names = tuple(nm for nm in ("cos", "sin") if not (self.rope_position_tables_only and self._descs[nm] is None))
        for nm in table_names:
            self._check_tensor_shape(self._descs[nm], (self.batch, self.seq_len, g.rope_dim), nm)
        self._check_tensor_shape(self._descs["out"], (self.batch, self.seq_len, g.d_model), "out")
        # dtype contract: bf16/f16 everywhere, or (FP8 / MXFP8) e4m3 h + weights (an e2m1 W_qkvg
        # under MxQuantSpec.w_qkvg_dtype -- checked per weight above) with every activation-side
        # tensor in the bf16 activation dtype.
        for nm in norm_names + table_names + ("out",):
            if self._descs[nm].dtype != self.act_dtype:
                raise ValueError(f"{nm} must be the activation dtype {self.act_dtype}, got {self._descs[nm].dtype}")
        if self.mxfp8:
            self._check_mxfp8_declaration()

    def _check_mxfp8_declaration(self) -> None:
        """The MXFP8 caller contract, typed, before any stage: whole SF atoms along
        K (``d_model % 128``), and the two F8_128x4 blobs' dtype / PADDED byte count
        / contiguity (``proj_gemm.sf_blob_bytes``).  16-B alignment needs a data
        pointer, so it is checked on the real tensors at ``execute``."""
        from .kernels.proj_gemm import sf_blob_bytes

        g = self.geom
        if g.d_model % _SF_TILE_ROWS:
            # v1 contract: K = d_model in whole 128-element F8_128x4 atoms (4 blocks of
            # 32), so `h_sf` / `w_qkvg_sf` carry no partially-used 4-block words and
            # the quantizer / GEMM / oracle agree on every byte.  Rows (T, n_qkvg) ARE
            # padded (sf_padded_dims); only the K axis is pinned.
            raise NotImplementedError(
                f"MXFP8 needs geometry.d_model % {_SF_TILE_ROWS} == 0 (whole F8_128x4 scale-factor atoms along the contraction: 4 blocks of "
                f"{MXFP8_BLOCK_SIZE}); got d_model={g.d_model}. Use the per-tensor FP8 or bf16 pipeline for this width."
            )
        t = self.batch * self.seq_len
        # (name, rows, K, block, accepted storage dtypes) of every caller blob: the two MXFP8 ones (E8M0 / 32),
        # plus the e2m1 W_o's under o_fp4 (the format's own scale dtype and block; K = the out_proj's H_q*D).
        blobs = [("h_sf", t, g.d_model, MXFP8_BLOCK_SIZE, _SF_DTYPES), ("w_qkvg_sf", g.n_qkvg, g.d_model, MXFP8_BLOCK_SIZE, _SF_DTYPES)]
        if self.o_fp4 is not None:
            block = self.o_fp4.block_size
            if g.d_head % (4 * block):
                # One head's scales must be whole 4-block F8_128x4 words, so the per-head quantize CTA owns whole
                # atoms of the out_proj blob and no two heads share a byte (d=256 passes both formats).
                raise NotImplementedError(
                    f"fp4 O ({self.o_fp4.name}) needs geometry.d_head % {4 * block} == 0 (whole 4-block scale words per head at block {block}); "
                    f"got d_head={g.d_head}. Use the per-tensor e4m3 O (o_fp4=None) for this head dim."
                )
            blobs.append(("w_o_sf", g.d_model, g.h_q * g.d_head, block, (torch.uint8, self.o_fp4.sf_torch_dtype)))
        for nm, rows, k, block, dtypes in blobs:
            d = self._descs[nm]
            if d.dtype not in dtypes:
                want = " or ".join(str(x) for x in dtypes)
                raise ValueError(f"sample_{nm} must be {want} (scale bytes in F8_128x4 order), got {d.dtype}")
            numel = 1
            for x in d.shape:
                numel *= int(x)
            need = sf_blob_bytes(rows, k, block)
            if numel != need:
                raise ValueError(
                    f"sample_{nm} has {numel} bytes; the PADDED F8_128x4 blob over {rows} rows x K={k} at block {block} is {need} "
                    f"(ceil(rows/128)*128 x ceil(K/{block}/4)*4 -- kernels.proj_gemm.sf_blob_bytes; pad rows / blocks 0x00)"
                )
            # Contiguity from the descriptor strides (an opaque blob bound by storage order).
            expect, ok = 1, True
            for size, stride in zip(reversed(d.shape), reversed(d.stride)):
                if int(size) != 1 and int(stride) != expect:
                    ok = False
                expect *= int(size)
            if not ok:
                raise ValueError(f"sample_{nm} must be contiguous, got shape {tuple(d.shape)} strides {tuple(d.stride)}")

    # -- workspace ----------------------------------------------------------

    def _layout(self) -> "_Intermediates":
        return _plan_workspace(
            self.geom,
            self.batch,
            self.seq_len,
            self.act_dtype,
            self.return_lse,
            self.save_for_backward,
            self.inplace_qkv,
            fp8=self.quant is not None,
            fp8_fused=self.quant_fused,
            mxfp8=self.mxfp8,
            o_fp4=self.o_fp4,
            want_saved=self.save_for_backward,
            saved_gate_copy=self.saved_gate_copy,
            thd_num_sequences=self.num_sequences if self.thd else 0,
            qsa_indexer_cand_floats=self._indexer.cand_floats if (self._indexer is not None and self._indexer.selects) else None,
            kv_cache_e4m3=self.kv_cache_e4m3,
            upcast_items=(self.batch * self.seq_len * self.geom.h_kv) if self.upcast_decode else 0,
            qsa_indexer_e4m3=(self._indexer is not None and self._indexer.fp8),
            # the prefill route's [B] int32 Q lengths (a declaration fact: the route is read at declaration, so get_workspace_size()
            # before compile() already counts the slot)
            decode_q_lens=self.batch if (self.decode_route is not None and self.decode_route.prefill_tile) else 0,
            # the decode indexer over the pools: its five slots at the end (the logits width = the main table's reach in compressed rows;
            # the ix_q slot at the indexer cache dtype's element size -- 1 under the e4m3 cache)
            qsa_indexer_decode=self._indexer_decode.n_blocks_max if self._indexer_decode is not None else None,
            qsa_indexer_decode_cache_bytes=_itemsize(self._indexer_decode.cache_dtype) if self._indexer_decode is not None else None,
        )

    def get_workspace_size(self) -> int:
        """Bytes the caller must provide: every intermediate, plus each
        sub-engine's own scratch at a reserved offset.

        Honest and never exceeded (contract § 10) — that is what keeps the block
        CUDA-graph capturable with stable pointers.
        """
        self._ensure_support_checked()
        lay = self._layout()
        engine = max(self._proj.workspace_bytes(), self._out_proj.workspace_bytes(), self._sdpa.scratch_workspace_bytes(), 1)
        return lay.total_bytes + _align_up(engine)

    # -- compile ------------------------------------------------------------

    def _flag_capture_safe_projections(self) -> None:
        """Per paged projection stage (``kv_source="paged"``), ``capture_safe`` iff the driver's "auto" resolver names a JIT tile for
        the stage's OWN (M, N, K) (:meth:`_Projection.resolves_a_jit_tile`; the predicate the declaration documents at its stage list).
        Set at declaration (the host facts) and read AGAIN at :meth:`compile`, on the compiling device: the SM count the small-M route
        reads is a device property, so a declaration and its compile on different parts would otherwise carry a stale flag into
        ``build_proj_gemm`` -- its typed refusal where the flag says True and the route resolves no tile, the eager graph route where
        it says False and the route has one.  A host query, no device sync."""
        if self.kv_source != KV_SOURCE_PAGED:
            return
        for st in (self._proj, self._out_proj):
            if isinstance(st, _Projection):
                st.capture_safe = st.resolves_a_jit_tile()

    def compile(self) -> None:
        """Build all five artifacts. Plan-time keys only, so every execute-path
        dispatch is a guaranteed cache hit (Rule 4)."""
        self._ensure_support_checked()
        self._flag_capture_safe_projections()  # on the compiling device (the method's docstring)
        for st in self._stages:
            st.compile()
        self._ws = self._layout()
        self._quant_dev = self._make_quant_dev()
        self._quant_dev_on_launch_stream = False
        self._quant_dev_ready = None
        if self._quant_dev:
            # whatever the allocations above (and the fused fork's, in its stage compile) enqueued on the ambient stream -- under
            # deterministic mode a NaN fill -- is behind this event; _write_quant_dev waits on it before writing on any stream
            self._quant_dev_ready = torch.cuda.Event()
            self._quant_dev_ready.record(torch.cuda.current_stream(self.device))
        # The DECODE form's PREFILL route's per-sequence Q lengths are a workspace slot (_Intermediates.decode_q_lens, in self._ws above),
        # filled by every execute on its launch stream (_fill_decode_q_lens): nothing to allocate or to record here.

    def _quant_dev_values(self, spec: Optional[Union[QuantSpec, MxQuantSpec]]) -> Optional[dict]:
        """The per-tensor scalars of ``spec`` the block holds as 1-element fp32 device tensors, as ``{name: float}`` -- the ONE
        spelling ``compile()`` materialises (:meth:`_make_quant_dev`) and :meth:`update_quant_scales` rewrites in place.  ``None``
        for bf16; ``{}`` under ``o_fp4`` -- neither ``alpha_o`` nor ``scale_o`` exists there (both pinned 1.0: the fp4 out_proj has
        no alpha epilogue and the fp4 quantizer takes no scale); under MXFP8 only the out projection's per-tensor pair survives
        (D1) -- ``alpha_o`` for the GEMM epilogue, ``scale_o`` for the unfused quantize_o; under per-tensor FP8 the two GEMM alphas,
        the four quantize scales and the fp8 SDPA's three descales."""
        if spec is None:
            return None
        if self.o_fp4 is not None:
            return {}
        if self.mxfp8:
            return dict(alpha_o=float(spec.alpha_o), scale_o=float(spec.scale_o))
        return dict(
            alpha_qkvg=float(spec.alpha_qkvg),
            alpha_o=float(spec.alpha_o),
            scale_q=float(spec.scale_q),
            scale_k=float(spec.scale_k),
            scale_v=float(spec.scale_v),
            scale_o=float(spec.scale_o),
            descale_q=1.0 / float(spec.scale_q),
            descale_k=1.0 / float(spec.scale_k),
            descale_v=1.0 / float(spec.scale_v),
        )

    def _make_quant_dev(self) -> Optional[dict]:
        """The quant spec's scalars as 1-element fp32 device tensors (plan-time constants; the execute path never allocates):
        one ``torch.empty`` per name of :meth:`_quant_dev_values` -- ALLOCATED here, never filled here.  Their VALUES are written
        on a launch stream only: by the FIRST ``execute`` on ITS launch stream (:meth:`_write_quant_dev`) and by every
        :meth:`update_quant_scales` on the caller's.  A fill enqueued on the stream ambient at ``compile()`` would have no ordering
        against a recalibration on another stream: pending work ahead of it would let it land after the update and restore the
        declared values while ``self.quant`` already named the new spec.  The allocation is not free of stream work either
        (``torch.use_deterministic_algorithms(True)`` makes ``torch.empty`` fill with NaN), which is why ``compile()`` records an
        event behind it that every writer waits on."""
        vals = self._quant_dev_values(self.quant)
        if vals is None:
            return None
        return {name: torch.empty((1,), dtype=torch.float32, device=self.device) for name in vals}

    def _check_quant_update(self, spec) -> None:
        """The typed refusals of :meth:`update_quant_scales`, every one BEFORE any write: a block declared without ``quant``
        (``ValueError``); a spec of the other class (``TypeError``: the two classes select different pipelines); a differing plan
        fact -- ``dtype``, and under MXFP8 ``block_size`` / ``w_qkvg_dtype`` / ``o_fp4`` -- (``ValueError`` naming the field: those
        select kernels and the workspace carve, only the scales may move); ``spec``'s own ``validate`` EXACTLY as the declaration
        applied it (``MxQuantSpec.validate(fused=...)`` -- the fully fused MXFP8 path pins ``scale_o == 1.0`` --, the bare
        ``QuantSpec.validate()``: a zero / inf / NaN scale is refused there); a block not yet compiled (``RuntimeError``)."""
        if self.quant is None:
            raise ValueError(
                "update_quant_scales() belongs to a quantized block (quant=QuantSpec / MxQuantSpec): this block was declared without quant and "
                "holds no per-tensor scale to update"
            )
        if type(spec) is not type(self.quant):
            raise TypeError(
                f"update_quant_scales(): spec must be a {type(self.quant).__name__}, the class this block was declared with (QuantSpec and "
                f"MxQuantSpec select different pipelines), got {type(spec).__name__}"
            )
        for name in ("dtype", "block_size", "w_qkvg_dtype", "o_fp4") if self.mxfp8 else ("dtype",):
            if getattr(spec, name) != getattr(self.quant, name):
                raise ValueError(
                    f"update_quant_scales(): {type(spec).__name__}.{name} is a plan fact (it selects kernels and the workspace carve): declared "
                    f"{getattr(self.quant, name)!r}, got {getattr(spec, name)!r}; only the scales may change -- declare a new block for a new {name}"
                )
        if self.mxfp8:
            spec.validate(fused=bool(self.fuse_gate) and bool(self.fuse_norm_rope))
        else:
            spec.validate()
        if self._ws is None:
            raise RuntimeError("call compile() before update_quant_scales(): the device scalars it rewrites are materialised there")

    def _fill_decode_q_lens(self, workspace: torch.Tensor, stream: int) -> torch.Tensor:
        """The prefill route's per-sequence Q lengths -- the constant ``S_q`` of every sequence of the dense step -- as the ``[B]`` int32
        VIEW of the caller's workspace slot ``_Intermediates.decode_q_lens`` (Rule 8 recipe R2: carved, never allocated), FILLED on
        ``stream`` (a raw CUstream handle: the launch stream, so the SDPA that reads them is ordered behind the fill by construction,
        Rule 5) on EVERY execute (recipe R4: one fill of ``4 x B`` bytes, nothing read back, no first-execute flag).  A stream capture
        of an execute records the fill WITH its value and every replay re-fills the slot before the attention kernel; an eager execute
        after a capture -- replayed or not -- fills its own, so no execute ever hands the paged THD leg an unwritten vector."""
        q_lens = _view(workspace, self._ws.decode_q_lens, (self.batch,), torch.int32)
        with torch.cuda.stream(as_torch_stream(stream, self.device)):
            q_lens.fill_(self.seq_len)
        return q_lens

    def _write_quant_dev(self, spec, stream: int) -> None:
        """Write ``spec``'s scalars into the device tensors of :meth:`_make_quant_dev` -- and, on the fully fused fp8 pipeline, into
        the projection fork's fp32 ``[alpha_qkvg, scale_q, scale_k, scale_v]`` vector (:meth:`_FusedQkvProjection.write_qscal`) -- IN PLACE on ``stream`` (a raw CUstream
        handle: the launch stream, so the next execute on it is ordered behind the writes by construction, Rule 5).  One fill
        per value: nothing allocated, nothing read back, no recompile.  Nothing to write under ``o_fp4`` (``{}``).  The write
        first waits on the event ``compile()`` recorded behind its allocations, so stream work the allocation itself enqueued (under
        ``torch.use_deterministic_algorithms(True)`` ``torch.empty`` fills the tensors with NaN) lands before any value written on
        another stream.  Inside a stream capture the wait is skipped (an external event cannot be waited on there) -- the documented
        case: warm the block up before capturing."""
        qd = self._quant_dev
        if not qd:
            return
        vals = self._quant_dev_values(spec)
        ts = as_torch_stream(stream, self.device)
        with torch.cuda.stream(ts):
            if self._quant_dev_ready is not None and not torch.cuda.is_current_stream_capturing():
                ts.wait_event(self._quant_dev_ready)
            for name, ten in qd.items():
                ten.fill_(vals[name])
        if self.fp8_fused:
            # the fused fork's epilogue reads the four from the stage's own vector (allocated at its compile(), written only on a
            # launch stream); the stage's spec follows, so a later recompile declares the same values
            self._proj.write_qscal(spec, stream)

    def update_quant_scales(self, spec: Union[QuantSpec, MxQuantSpec], *, current_stream: Optional[cuda.CUstream] = None) -> None:
        """Re-point a COMPILED quantized block's per-tensor scales at ``spec`` without recompiling: an in-place write of the
        device scalars of :meth:`_make_quant_dev` (``alpha_qkvg`` / ``alpha_o`` / ``scale_q`` / ``scale_k`` / ``scale_v`` /
        ``scale_o`` / ``descale_q`` / ``descale_k`` / ``descale_v`` under a ``QuantSpec``; ``alpha_o`` / ``scale_o`` under an
        ``MxQuantSpec``; and the fused fp8 fork's ``[alpha_qkvg, scale_q, scale_k, scale_v]`` vector where it exists) on the
        launch stream -- ``current_stream`` (a raw ``CUstream``), else torch's current stream on the block's device (Rule 5) --
        then ``self.quant = spec``.  Every consumer binds those tensors at ``execute``, so the next execute on that stream runs
        at the new scales, and a CUDA graph that captured an execute of a WARMED-UP block replays with the LIVE values (the
        scalars are read, never baked into the graph) -- unless the captured execute was the block's FIRST: its one-time scalar
        write (:meth:`_write_quant_dev`, one fill kernel per value) is captured WITH the capture-time values, so every replay
        re-writes them and overrides a later eager ``update_quant_scales``.  Run one eager execute on the capture stream before
        capturing (every capture of the block does), or treat a graph that captured the first execute as pinned to the scales
        it captured and re-capture after a recalibration.
        No allocation, no host readback, no new plan: the training-loop recipe is one call per layer
        right before its forward (``descale_h`` / ``descale_w_*`` from the tensors quantized this step, the activation scales
        from the previous step's record), and the same ``spec`` handed to ``GatedAttentionBlockBwd.update_quant_scales`` so the
        backward rebuilds the SDPA operands at the forward's scales.

        Typed refusals, every one BEFORE any write (:meth:`_check_quant_update`): a block declared without ``quant``
        (``ValueError``); a spec of the other class (``TypeError``); a differing plan fact -- ``dtype``, ``block_size``,
        ``w_qkvg_dtype``, ``o_fp4`` -- (``ValueError`` naming the field: those select kernels and the carve); ``spec``'s own
        ``validate`` exactly as the declaration applied it (the fully fused MXFP8 path keeps its ``scale_o == 1.0`` rule; a zero
        / inf / NaN scale is its own ``ValueError``); a block not yet compiled (``RuntimeError``).  Under ``o_fp4``
        there is nothing to write (both per-tensor scales are pinned 1.0): the call validates and records ``spec``.
        """
        self._check_quant_update(spec)
        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(self.device).cuda_stream
        self._write_quant_dev(spec, stream)
        self.quant = spec

    # -- the training record --------------------------------------------------

    @staticmethod
    def _check_saved_tensor(name: str, ten, shape: tuple, dtype: torch.dtype, device, *, contiguous: bool = True) -> None:
        """One caller-owned buffer of the record: a tensor of exactly ``shape`` / ``dtype`` on ``device`` (compact when
        ``contiguous``) whose base is 16-B aligned -- every one of them is a TMA-store or a 16-B vector-store target, and a
        misaligned base (a ``[1:]`` slice of a caller arena passes count / dtype / contiguity) would fail UNTYPED at the
        tensor-map encode or the launch, after earlier stages ran; the package precedent is ``kernels/proj_gemm.py``'s
        output checks and :func:`_check_sf_blob`.  Typed and naming the field -- no launch, no device read."""
        if not isinstance(ten, torch.Tensor):
            raise ValueError(f"{name} must be a caller-owned {list(shape)} {dtype} tensor on {device}, got {type(ten).__name__}")
        if tuple(int(x) for x in ten.shape) != tuple(shape):
            raise ValueError(f"{name} must be {list(shape)}, got {list(ten.shape)}")
        if ten.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {ten.dtype}")
        if ten.device != device:
            raise ValueError(f"{name} must live on h's device {device}, got {ten.device}")
        if contiguous and not ten.is_contiguous():
            raise ValueError(f"{name} must be contiguous (compact, the layout the kernels write), got strides {tuple(ten.stride())}")
        if ten.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned (a TMA-store / 16-B vector-store target), got data_ptr={ten.data_ptr():#x}")

    def _check_thd_seq_lens(self, seq_lens, device) -> None:
        """The packed lengths' FORM, host-side and before any launch: a contiguous 1-D int32 CUDA tensor on ``h``'s device
        with EXACTLY ``num_sequences`` (``[B]`` lengths) or ``num_sequences + 1`` (``[B+1]`` prefix sums, ``cu_seqlens``)
        elements -- stricter than the forward adapter's ``1..B`` admission because the backward binds exactly ``B`` or
        ``B+1`` and the training record must round-trip.  The VALUES are never read (Rule 3)."""
        n = self.num_sequences + (1 if self.cu_seqlens else 0)
        form = "[B+1] prefix sums" if self.cu_seqlens else "[B] lengths"
        device = torch.device(device)
        ok = (
            isinstance(seq_lens, torch.Tensor)
            and seq_lens.dtype == torch.int32
            and seq_lens.dim() == 1
            and seq_lens.is_contiguous()
            and seq_lens.numel() == n
            and seq_lens.is_cuda
            and device.type == "cuda"
            and (device.index is None or seq_lens.device.index == device.index)
        )
        if not ok:
            got = (
                f"dtype {seq_lens.dtype}, shape {tuple(seq_lens.shape)}, strides {tuple(seq_lens.stride())}, device {seq_lens.device}"
                if isinstance(seq_lens, torch.Tensor)
                else type(seq_lens).__name__
            )
            raise ValueError(f"thd=True: seq_lens must be a contiguous 1-D int32 CUDA tensor of {n} elements on {device} ({form}), got {got}")

    def _check_saved_set(
        self, h: torch.Tensor, seq_lens: Optional[torch.Tensor], lse: Optional[torch.Tensor], saved: Optional[SavedForBackward]
    ) -> _SavedBinding:
        """Validate the :class:`SavedForBackward` record against this declaration and bind the caller tensors the stages
        write through.  Every miss is a ``ValueError`` naming the field; nothing here launches, allocates or reads the
        device, and it runs on a DECLARED (uncompiled) block, so the contract is testable on any device.  ``execute``
        calls it BEFORE its first launch.

        The contracts:

        1. ``saved.rstd_q`` / ``rstd_k`` are tensors iff ``geometry.qk_norm`` (both directions) -- and then ``[B, S, H_q]`` /
           ``[B, S, H_kv]`` fp32 compact: the norm kernel writes them through raw fp32 pointer arithmetic over ``[T, H]``.
        2. ``saved.h`` IS ``h`` (same storage) -- the backward reads it for the projection wgrad and the Q/K recompute.
        3. ``saved.seq_lens`` IS the ``seq_lens`` this execute runs with (``is``, or both ``None``): the backward's
           declaration-time padding decline rests on this identity, without a D2H read.  Under ``thd=True`` a tensor is
           REQUIRED (it is the packed-lengths tensor the backward binds), and ``saved.seq_lens_form`` must name its form
           (``"lengths"`` / ``"prefix"`` per ``cu_seqlens``); a dense record carries ``seq_lens_form=None``.  Both typed.
        4. ``saved.lse`` is ``[B, H_q, S]`` fp32 compact; ``lse=None`` defaults to it and an explicit ``lse`` must be that
           same storage (the SDPA adapter requires an LSE tensor once compiled with one).
        5. ``saved.o`` is compact ``[B, S, H_q, D]`` in the activation dtype: the SDPA writes the PRE-gate O there.
        6. proj_slab mode (``saved_gate_copy=False``): ``saved.proj_slab`` is REQUIRED (``B*S x n_qkvg`` elements, act dtype,
           contiguous, on ``h``'s device) and becomes the stage-(1) output; ``saved.gate`` / ``q_pre`` / ``k_pre``, when
           given, must alias it exactly as :func:`saved_slab_views` spells (data_ptr, dtype, shape AND strides) -- each may
           be ``None`` here, the backward derives it from the slab.
        7. gate-copy mode (``saved_gate_copy=True``): ``saved.proj_slab`` must be ``None``, ``saved.gate`` is a compact
           ``[B, S, H_q, D]`` caller buffer the GATE band is copied into; ``q_pre`` / ``k_pre`` are copied only if given.
        8. Every caller buffer a kernel WRITES -- ``proj_slab``, ``o``, ``lse``, ``rstd_*``, the gate-copy targets -- is
           16-B aligned (:meth:`_check_saved_tensor`): they are TMA-store / 16-B vector-store targets, and a misaligned
           base fails untyped at the tensor-map encode or the launch, after stage (1) already ran.
        """
        if saved is None:
            raise ValueError("save_for_backward=True requires a SavedForBackward to write through")
        g, b, s = self.geom, self.batch, self.seq_len
        t, act, dev = b * s, self.act_dtype, h.device
        if g.qk_norm:
            if saved.rstd_q is None or saved.rstd_k is None:
                raise ValueError("save_for_backward=True with geometry.qk_norm=True needs SavedForBackward.rstd_q and rstd_k tensors to write through")
            # The norm kernel writes rstd by raw fp32 pointer arithmetic over [T, H]: a wrong count, dtype or layout here
            # is a device OOB write or silently misplaced values, never an error.
            self._check_saved_tensor("saved.rstd_q", saved.rstd_q, (b, s, g.h_q), torch.float32, dev)
            self._check_saved_tensor("saved.rstd_k", saved.rstd_k, (b, s, g.h_kv), torch.float32, dev)
        elif saved.rstd_q is not None or saved.rstd_k is not None:
            raise ValueError(
                "geometry.qk_norm=False (RoPE-only) computes no RMSNorm and writes no rstd: SavedForBackward.rstd_q and rstd_k must be None "
                "(stage B6 does not exist)"
            )
        sh = saved.h
        if not isinstance(sh, torch.Tensor) or sh.data_ptr() != h.data_ptr() or sh.numel() != h.numel() or sh.dtype != h.dtype:
            raise ValueError(
                "saved.h must be the SAME storage as the h this forward runs on (data_ptr, element count and dtype equal): the backward reads "
                "saved.h for the projection wgrad and the Q/K recompute, so a copy or another tensor there would silently differentiate a "
                "different input"
            )
        if saved.seq_lens is not seq_lens:
            raise ValueError(
                "saved.seq_lens must be the very tensor passed to execute(seq_lens=) -- the same object, or both None; got "
                f"saved.seq_lens={'None' if saved.seq_lens is None else 'a tensor'}, seq_lens={'None' if seq_lens is None else 'a tensor'}. "
                "The backward declines padding at declaration from this field (no device read), so the record must say what the forward ran with."
                + (" -- under thd=True saved.seq_lens is REQUIRED: it is the packed-lengths tensor the backward binds" if self.thd else "")
            )
        # The record must SAY which length contract it carries (a frozen, caller-written field verified like the identity
        # above): the backward checks it against its own declaration, so a padded dense record never reaches a THD backward
        # and a packed one never reaches a dense backward without a typed decline.
        want_form = _thd_seq_lens_form(self.cu_seqlens) if self.thd else None
        if saved.seq_lens_form != want_form:
            if self.thd:
                raise ValueError(
                    f"saved.seq_lens_form must be {want_form!r} for this block (thd=True, cu_seqlens={self.cu_seqlens}: seq_lens is the "
                    f"{'[B+1] int32 prefix sums' if self.cu_seqlens else '[B] int32 lengths'}), got {saved.seq_lens_form!r}: the record must "
                    "say which form of packed lengths it carries, so the backward can check it against its own declaration"
                )
            raise ValueError(
                f"saved.seq_lens_form must be None for a dense block (thd=False): seq_lens there is the per-batch KV padding mask (or None), "
                f"not packed lengths; got {saved.seq_lens_form!r}"
            )
        self._check_saved_tensor("saved.lse", saved.lse, (b, g.h_q, s), torch.float32, dev)
        if lse is not None and lse.data_ptr() != saved.lse.data_ptr():
            raise ValueError(
                "lse and saved.lse are different storage; under save_for_backward the SDPA writes saved.lse -- pass lse=None (the default) "
                "or saved.lse itself"
            )
        self._check_saved_tensor("saved.o", saved.o, (b, s, g.h_q, g.d_head), act, dev)
        o = saved.o.view(t, g.h_q, g.d_head)
        if self.saved_gate_copy:
            if saved.proj_slab is not None:
                raise ValueError(
                    "this block was declared saved_gate_copy=True (the gate-copy save mode: the projection slab stays in the workspace and the "
                    "GATE band is copied into a compact saved.gate), but saved.proj_slab was given -- declare saved_gate_copy=False for the "
                    "proj_slab save mode, or pass proj_slab=None"
                )
            self._check_saved_tensor("saved.gate", saved.gate, (b, s, g.h_q, g.d_head), act, dev)
            for nm, ten, hh in (("saved.q_pre", saved.q_pre, g.h_q), ("saved.k_pre", saved.k_pre, g.h_kv)):
                if ten is not None:
                    self._check_saved_tensor(nm, ten, (b, s, hh, g.d_head), act, dev)
            return _SavedBinding(
                lse=saved.lse,
                rstd_q=saved.rstd_q,
                rstd_k=saved.rstd_k,
                proj=None,
                o=o,
                gate_dst=saved.gate.view(t, g.h_q, g.d_head),
                q_pre_dst=None if saved.q_pre is None else saved.q_pre.view(t, g.h_q, g.d_head),
                k_pre_dst=None if saved.k_pre is None else saved.k_pre.view(t, g.h_kv, g.d_head),
            )
        ps = saved.proj_slab
        if ps is None:
            raise ValueError(
                "saved.proj_slab is required: this block was declared saved_gate_copy=False (the proj_slab save mode -- stage (1) writes the "
                "caller-owned [B*S, n_qkvg] slab and gate / q_pre / k_pre / V are its column bands, saved_slab_views). Pass it, or declare "
                "saved_gate_copy=True for the gate-copy save mode"
            )
        if not isinstance(ps, torch.Tensor) or ps.numel() != t * g.n_qkvg:
            raise ValueError(
                f"saved.proj_slab has {ps.numel() if isinstance(ps, torch.Tensor) else 'no'} elements; the stage-(1) slab over B*S={t} tokens x "
                f"n_qkvg={g.n_qkvg} columns is {t * g.n_qkvg} ([B*S, n_qkvg] or [B, S, n_qkvg])"
            )
        if ps.dtype != act:
            raise ValueError(f"saved.proj_slab must be the activation dtype {act} (the GEMM's output dtype), got {ps.dtype}")
        if not ps.is_contiguous():
            raise ValueError(
                f"saved.proj_slab must be contiguous (bound by pointer as ONE row-major [B*S, n_qkvg] GEMM output), got strides {tuple(ps.stride())}"
            )
        if ps.device != dev:
            raise ValueError(f"saved.proj_slab must live on h's device {dev}, got {ps.device}")
        if ps.data_ptr() % 16:
            raise ValueError(f"saved.proj_slab must be 16-byte aligned (the projection GEMM TMA-stores it), got data_ptr={ps.data_ptr():#x}")
        want_q, want_gate, want_k, _want_v = saved_slab_views(ps, g, b, s)
        for nm, given, want in (("gate", saved.gate, want_gate), ("q_pre", saved.q_pre, want_q), ("k_pre", saved.k_pre, want_k)):
            if given is None:
                continue
            ok = (
                isinstance(given, torch.Tensor)
                and given.data_ptr() == want.data_ptr()
                and given.dtype == want.dtype
                and tuple(given.shape) == tuple(want.shape)
                and tuple(given.stride()) == tuple(want.stride())
            )
            if not ok:
                got = (
                    f"data_ptr {given.data_ptr():#x} {given.dtype} shape {tuple(given.shape)} strides {tuple(given.stride())}"
                    if isinstance(given, torch.Tensor)
                    else type(given).__name__
                )
                raise ValueError(
                    f"saved.{nm} must alias saved.proj_slab's {nm.upper() if nm == 'gate' else nm[0].upper()} band exactly as "
                    "saved_slab_views(proj_slab, geometry, batch, seq_len) spells it ([B, S, heads, D], token stride n_qkvg, storage offset at "
                    f"qkvg_offsets): expected data_ptr {want.data_ptr():#x} {want.dtype} shape {tuple(want.shape)} strides {tuple(want.stride())}, "
                    f"got {got}. Leave it None to let the backward derive it from the slab."
                )
        return _SavedBinding(
            lse=saved.lse, rstd_q=saved.rstd_q, rstd_k=saved.rstd_k, proj=ps.view(t, g.n_qkvg), o=o, gate_dst=None, q_pre_dst=None, k_pre_dst=None
        )

    # -- execute ------------------------------------------------------------

    def execute(
        self,
        h: torch.Tensor,
        w_qkvg: torch.Tensor,
        w_q_norm: Optional[torch.Tensor],  # [D]; None (both) iff geometry.qk_norm is False
        w_k_norm: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        w_o: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        saved: Optional[SavedForBackward] = None,
        current_stream: Optional[cuda.CUstream] = None,
        h_sf: Optional[torch.Tensor] = None,  # MXFP8 only (both REQUIRED): the F8_128x4 E8M0 blobs of h and W_qkvg (sample_* byte counts)
        w_qkvg_sf: Optional[torch.Tensor] = None,
        w_o_sf: Optional[torch.Tensor] = None,  # fp4 O only (REQUIRED there): the F8_128x4 blob of the e2m1 W_o (sample_w_o_sf's byte count)
        # APPENDED (block-sparse attention, geometry.qsa with index_source="caller"): the per-query selection.  REQUIRED there, REFUSED
        # on a dense block.  Checked for FORM only (dtype / rank / shape / contiguity / device), never read on the host (Rule 3).
        block_ids: Optional[torch.Tensor] = None,  # [T, top_k] int32 (or [B, S, top_k]): the ids of each query's selected complete blocks, valid prefix then -1
        block_lens: Optional[torch.Tensor] = None,  # [T] int32 (or [B, S]), optional: the valid-prefix length per query (lowering-only; clamped on device)
        # APPENDED (serving, paged_kv_page_size > 0): the paged KV cache.  k_cache / v_cache / slot_mapping REQUIRED there (index_k_raw
        # too iff the geometry declares the indexer band), every one of them REFUSED on a block declared without the attribute.
        # Checked for FORM only (dtype / rank / shape / strides / alignment / device), never read on the host (Rule 3).
        k_cache: Optional[
            torch.Tensor
        ] = None,  # [num_pages, H_kv, page_size, D] activation dtype (HND compact, NHD by strides, or a unified-row half): post-RoPE K lands here
        v_cache: Optional[torch.Tensor] = None,  # same shape: V lands here
        block_table: Optional[
            torch.Tensor
        ] = None,  # [num_sequences, >= max_pages] int32 -- the EXTEND form's page table (kv_source="paged"); refused otherwise
        kv_lens: Optional[torch.Tensor] = None,  # [num_sequences] int32 -- the EXTEND form's LOGICAL KV lengths incl. this step's tokens; refused otherwise
        slot_mapping: Optional[torch.Tensor] = None,  # [T] int32 / int64: the flat slot (page * page_size + offset) of every token; negative = no write
        index_k_raw: Optional[torch.Tensor] = None,  # [num_pages, page_size, index_head_dim] activation dtype: the raw indexer key's pool (index band only)
        # APPENDED (serving, the bounded packed form -- token_capacity=T_max): the step's LIVE token count T_live, a HOST int.  None (the
        # default) = the declared capacity, every stage over all T_max rows as before.  An int launches the token-indexed stages over
        # T_live <= T_max rows WITHOUT a recompile (the block kernels take a symbolic token count): h / cos / sin / slot_mapping / out
        # may then carry EITHER the declared T_max rows (the first T_live are read / written) OR exactly T_live rows (a serving stack's
        # hidden_states[rows] / output[rows] directly -- no staging copy); rows [T_live, T_max) of `out` are UNTOUCHED.  The SDPA keeps the
        # declared geometry and takes the live count from seq_lens on device, so the lengths must sum to T_live.  Typed declines: a
        # block declared without token_capacity, T_live outside [1, T_max], a quantized pipeline, a projection on the graph route.
        # Under a CUDA-graph capture the host int is baked into the captured launches: ONE capture per live-count bucket (docs).
        num_tokens_live: Optional[int] = None,
        # APPENDED (the e4m3 KV cache, kv_cache_dtype=torch.float8_e4m3fn): the pools' per-tensor DEscales (value = code x scale), 1-element fp32
        # CUDA tensors read on device -- REQUIRED there, REFUSED on a block whose pools are the activation dtype.
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        # APPENDED (the in-block indexer, geometry.qsa with index_source="indexer") AFTER v_scale -- the end of the signature, so every
        # positional caller of the earlier parameters is unchanged (the append-only rule): the indexer's two norm weights (REQUIRED there)
        # and its three OPTIONAL outputs -- every one of the five REFUSED on a block declared without index_source="indexer".  Form checks only.
        w_iq_norm: Optional[torch.Tensor] = None,  # [index_head_dim] activation dtype: the indexer queries' RMSNorm weight (pre-folded (1 + w), as w_q_norm)
        w_ik_norm: Optional[torch.Tensor] = None,  # [index_head_dim] activation dtype: the compressed keys' RMSNorm weight
        index_k_compressed: Optional[
            torch.Tensor
        ] = None,  # [B, >= floor(S / 4), index_head_dim] activation dtype: the compressed-key cache, rows [0, floor(S_b / 4)) written
        block_ids_out: Optional[
            torch.Tensor
        ] = None,  # [T, top_k] int32 (or [B, S, top_k]): the block's own selection, -1 padded (the step-0 list a speculative caller reuses)
        block_lens_out: Optional[torch.Tensor] = None,  # [T] int32 (or [B, S]): the valid-prefix length of every row of that list
        # APPENDED (the in-block indexer's DECODE step over the pools -- QsaSpec(index_source="indexer") with kv_source="paged"): the serving
        # stack's two side caches, the ring block table and the position-indexed RoPE tables, REQUIRED there; index_state_slot_mapping iff
        # index_state_slot_mapping_present; every one of the seven REFUSED on any other declaration; n_commit always refused
        # (index_commit_present is a typed decline).  Form checks only -- dims, dtype, the 16-byte rule, the ring's cap, the tables' reach
        # (never the strides: the kernels bind them symbolically; never a value).
        index_kv_cache: Optional[torch.Tensor] = None,  # [num_pages, P / 4, 1, 128] the paged compressed-key cache on the MAIN block table (read + written)
        index_state_cache: Optional[torch.Tensor] = None,  # [num_ring_blocks, cap >= S_q + 3, 1, 128] the raw-key ring (read + written)
        index_state_block_table: Optional[torch.Tensor] = None,  # [B, 1] int32: the ring block per sequence (0 = the null block of a padding entry)
        index_state_slot_mapping: Optional[
            torch.Tensor
        ] = None,  # [T] int32 / int64 (slot_mapping's dtype): the ring WRITE rows, block x cap + pos % cap, -1 = no write
        cos_pos: Optional[
            torch.Tensor
        ] = None,  # [1 or B, >= max_pages x P, rope_dim]: the position-indexed tables (block starts; every rotation under rope_position_tables_only)
        sin_pos: Optional[torch.Tensor] = None,
        n_commit: Optional[torch.Tensor] = None,  # refused: the two-phase commit is a typed decline at declaration (every row of the step commits)
    ) -> None:
        """Launch the five stages in pipeline order.

        No allocation, no D2H read, no implicit conversion: every intermediate is
        a strided VIEW of the caller's workspace.

        ``w_q_norm`` / ``w_k_norm`` must agree with ``geometry.qk_norm`` in both
        directions (typed ``ValueError``), and so must ``saved.rstd_q`` /
        ``saved.rstd_k`` under ``save_for_backward`` (tensors iff qk_norm).

        ``h_sf`` / ``w_qkvg_sf`` (appended): REQUIRED under MXFP8 (both, checked
        for dtype / byte count / contiguity / 16-B alignment, typed), REFUSED
        otherwise.  ``w_o_sf`` (appended): REQUIRED under ``MxQuantSpec.o_fp4``
        (same checks at the format's block and scale dtype), REFUSED otherwise.

        ``saved``: REQUIRED under ``save_for_backward=True`` (the whole record is
        validated by :meth:`_check_saved_set` before any launch), REFUSED on a
        block declared without it -- an inference forward writes none of the
        record's tensors, so a silently ignored ``saved=`` would hand the
        backward an uninitialised save set.

        ``seq_lens`` under ``thd=True``: REQUIRED -- the per-sequence packed lengths
        (``[B]`` int32, or ``[B+1]`` int32 prefix sums under ``cu_seqlens=True``) on
        ``h``'s device, validated for form (never read) and handed to the SDPA for
        the Q and the KV side alike; ``h`` / ``cos`` / ``sin`` / ``out`` are the
        packed ``[T, .]`` (or ``[1, T, .]``) tensors the block was declared with.
        A training record then carries that very tensor as ``saved.seq_lens`` and
        names its form in ``saved.seq_lens_form``.

        ``block_ids`` / ``block_lens`` (appended): REQUIRED under ``geometry.qsa``
        with ``index_source="caller"`` (``block_ids``; ``block_lens`` optional),
        REFUSED on a block declared without ``qsa``.  Form checks only -- the ids
        are device data the sparse core reads (block ``b`` = tokens ``[4b, 4b + 4)``
        of the query's own sequence -- under ``thd=True`` numbered from that
        sequence's first token, ``-1`` = padding; the open tail block is always
        visible).

        ``k_cache`` / ``v_cache`` / ``slot_mapping`` (appended): REQUIRED under
        ``paged_kv_page_size > 0`` -- after norm + RoPE the block writes every
        token's post-RoPE K and V row into the pools at ``slot_mapping[token]``
        (flat slot = ``page * page_size + offset``; a negative slot writes nothing,
        the serving stacks' padding convention; a slot past the pool writes
        nothing either), then attends over its own K / V as before, so ``out`` is
        bitwise the block's without write-through.  Dense and packed (``thd=True``:
        ``slot_mapping`` is ``[T]`` over the packed tokens) alike.  ``index_k_raw``
        (appended): the raw indexer key's pool, REQUIRED there iff the geometry
        declares the indexer band.  ``block_table`` / ``kv_lens`` (appended): the
        EXTEND form's inputs, REQUIRED under ``kv_source="paged"`` -- the SDPA then
        reads every sequence's K / V from the pools (the step's tokens written
        through first, on the same stream) over ``[0, kv_lens[b])`` through
        ``block_table[b]`` on the bottom-right diagonal; the pools must match the
        declared samples exactly -- and refused under ``kv_source="self"``.  Every
        one of the six is REFUSED on a block declared without the attribute.  Form
        checks only; slot VALUES, page ids and lengths are device data the kernels
        read.  Under ``token_capacity`` the lengths may sum to ``T_live <= T``; rows
        ``[T_live, T)`` of ``out`` are then unspecified but finite given finite
        inputs (hand them slot ``-1`` and finite ``cos`` / ``sin`` rows).

        ``w_iq_norm`` / ``w_ik_norm`` (appended): REQUIRED under ``geometry.qsa`` with
        ``index_source="indexer"`` -- the indexer's ``[index_head_dim]`` RMSNorm
        weights (the pre-folded ``(1 + w)`` form, like ``w_q_norm``); the block then
        REFUSES ``block_ids`` / ``block_lens`` and derives the selection itself
        (:class:`_Indexer`: the band's queries normed and rotated per token, the raw
        key mean-pooled per complete block, normed and rotated at the block start
        with the attention's own ``cos`` / ``sin``, the scorer's top-k; at
        ``S <= identity_bound`` no scorer launches and the list is the identity).
        ``index_k_compressed`` / ``block_ids_out`` / ``block_lens_out`` (appended,
        optional there): the compressed keys (``[B, >= floor(S / 4), index_head_dim]``,
        rows ``[0, floor(S_b / 4))`` written -- a cache sized for a longer context
        keeps the rest; under the identity it is the only indexer launch), the block's
        own list (``[T, top_k]`` int32, ``-1`` padded: the step-0 list a speculative
        caller reuses) and its per-row counts ``min(top_k, floor((pos + 1) / 4))``.
        Every one of the five is REFUSED on a block declared without
        ``index_source="indexer"``.  Form checks only.
        ``num_tokens_live`` (appended; the bounded packed form only): the step's
        LIVE token count ``T_live`` as a HOST int.  ``None`` = the declared
        capacity (every stage over all ``T_max`` rows, bitwise as before).  With
        an int the projection, norm + RoPE, the cache write, the gate and the out
        projection run over ``T_live <= T_max`` rows with NO recompile (their
        kernels take the token count symbolically); ``h`` / ``cos`` / ``sin`` /
        ``slot_mapping`` / ``out`` may carry the declared ``T_max`` rows (the
        first ``T_live`` are read / written) or exactly ``T_live`` rows -- a
        serving stack hands ``hidden_states[rows]`` and ``output[rows]`` directly,
        with no staging copy -- and rows ``[T_live, T_max)`` of ``out`` are left
        UNTOUCHED.  The SDPA keeps the declared geometry and reads the live total
        from ``seq_lens`` on device, so the lengths must sum to ``T_live``
        (device data, never read here).  Typed, before any launch: a block
        declared without ``token_capacity`` refuses it (``ValueError`` naming the
        attribute), ``T_live`` outside ``[1, T_max]`` is a ``ValueError`` naming
        ``T_max``, a quantized pipeline a ``NotImplementedError`` (its scale-factor
        blobs are sized by the capacity), a dense projection on the graph route
        (a bounded block declared without ``kv_source="paged"``) a
        ``NotImplementedError`` (the graph binds the declared ``M``).  A CUDA-graph
        capture bakes the host int into the captured launches: a replay serves
        the live count it was captured with -- one capture per live-count bucket.

        ``k_scale`` / ``v_scale`` (appended): under ``kv_cache_dtype=torch.float8_e4m3fn``
        REQUIRED -- the pools' per-tensor DEscales (value = code x scale), 1-element
        fp32 CUDA tensors on ``h``'s device, read on device: the write-through
        quantizes the post-RoPE K / V with their reciprocals (two fp32 words of the
        workspace, written on the launch stream), the block-sparse decode step's
        gathered cast multiplies them back in; REFUSED on a block whose pools are
        the activation dtype.

        ``index_kv_cache`` / ``index_state_cache`` / ``index_state_block_table`` /
        ``index_state_slot_mapping`` / ``cos_pos`` / ``sin_pos`` / ``n_commit`` (appended):
        the in-block indexer's DECODE step over the pools (``QsaSpec(index_source="indexer")``
        with ``kv_source="paged"``; :class:`_IndexerDecode`).  The paged compressed-key cache
        (``[num_pages, P / 4, 1, 128]`` on the MAIN block table: the block a committed
        row completes is written, the scorer reads every visible block), the raw-key
        ring (``[num_ring_blocks, cap >= S_q + 3, 1, 128]`` behind the ``[B, 1]`` ring
        block table: the step's raw keys enter it, a completing block pools from it)
        and the position-indexed tables (``[1 or B, >= max_pages x P, rope_dim]``: a
        completing block's START lies up to three positions below the step) are
        REQUIRED there; ``index_state_slot_mapping`` iff ``index_state_slot_mapping_present``
        (``slot_mapping``'s dtype), ``cos`` / ``sin`` None iff ``rope_position_tables_only``;
        the step's selection comes back through ``block_ids_out`` / ``block_lens_out``
        (one list per row); ``index_k_raw`` / ``index_k_compressed`` / ``block_ids`` /
        ``n_commit`` are refused by name.  Every one of the seven is REFUSED on any
        other declaration.  Form checks only; the side caches' strides are never
        matched (the kernels bind them symbolically).
        """
        # The bounded packed form's LIVE token count (appended): validated FIRST -- host-side, typed, before the compile check, so a
        # declared block pins the contract on any device -- and handed to the slot-mapping form check (the two admitted row counts).
        t_live = self._check_num_tokens_live(num_tokens_live)
        # Block-sparse attention: the index tensors are a DECLARATION-vs-argument contract and need no plan, so they are
        # checked first and the refusal reads the same on every device (Rule 3: form only, never a value, never a sync).
        self._check_qsa_execute_args(
            block_ids,
            block_lens,
            h,
            w_iq_norm=w_iq_norm,
            w_ik_norm=w_ik_norm,
            index_k_compressed=index_k_compressed,
            block_ids_out=block_ids_out,
            block_lens_out=block_lens_out,
        )
        # The paged-cache inputs: the same kind of contract (declared with paged_kv_page_size -> required; without -> refused).
        self._check_cache_write_args(
            h, k_cache, v_cache, block_table, kv_lens, slot_mapping, index_k_raw, seq_lens=seq_lens, num_tokens_live=t_live, k_scale=k_scale, v_scale=v_scale
        )
        # The decode indexer's side caches, ring table and position tables (REQUIRED on its form, REFUSED elsewhere): form only, up front.
        self._check_indexer_decode_execute_args(
            h, cos, sin, cos_pos, sin_pos, index_kv_cache, index_state_cache, index_state_block_table, index_state_slot_mapping, n_commit, slot_mapping
        )
        if self._ws is None:
            raise RuntimeError("call compile() before execute()")
        if t_live is not None:
            # The dense projections must take M from the buffers: the forced-tile JIT route does (the serving declaration's capture-safe
            # route, kv_source="paged"); the graph route binds the declared M = T_max and would read T_live rows as T_max.
            for st in (self._proj, self._out_proj):
                if isinstance(st, _Projection) and (st._plan is None or getattr(st._plan, "jit", None) is None):
                    raise NotImplementedError(
                        f"num_tokens_live={t_live}: the {st.name} projection runs on the graph route (route={getattr(st._plan, 'route', None)!r}, "
                        "no JIT), which binds the declared M=T_max; a live token count below the capacity needs the forced-tile JIT route both "
                        "dense projections take under the serving declaration (kv_source='paged'). Declare the EXTEND form, or launch at the capacity"
                    )
        _check_norm_weights_agree(self.geom.qk_norm, w_q_norm, w_k_norm)
        # `saved=` is the training forward's write-through record.  On an inference
        # block no stage targets saved.o / lse / rstd_* / proj_slab / gate, so
        # accepting it would leave the caller holding an uninitialised record that
        # the backward then consumes -- refused up front, like every other argument
        # this declaration cannot consume (h_sf / w_qkvg_sf outside MXFP8, w_o_sf
        # outside fp4 O).  The training-side validation stays in _check_saved_set.
        if saved is not None and not self.save_for_backward:
            raise ValueError(
                "saved= is the SavedForBackward record a TRAINING forward writes through, but this block was declared "
                "save_for_backward=False (inference): no stage would write saved.o / lse / rstd_q / rstd_k / proj_slab / gate, so the "
                "record would stay uninitialised. Declare GatedAttentionBlockFwd(..., save_for_backward=True) for a training forward, "
                "or drop saved="
            )
        if self.thd:
            # THD: the packed lengths are REQUIRED and host-validated for FORM only (dtype / rank / contiguity / count /
            # device) -- never read (Rule 3): their values -- each in [0, max_seq_len], summing to T -- are the caller's
            # contract, normalized and consumed on device by the SDPA's setup launch.
            if seq_lens is None:
                raise ValueError(
                    "thd=True: execute(seq_lens=) is required -- the per-sequence packed lengths ([B] int32 lengths, or [B+1] int32 prefix sums "
                    "under cu_seqlens=True) on h's device; the SDPA builds its packed metadata from them"
                )
            self._check_thd_seq_lens(seq_lens, h.device)
        if self.mxfp8:
            if h_sf is None or w_qkvg_sf is None:
                raise ValueError("MXFP8 execute needs both scale-factor blobs: h_sf (over B*S rows) and w_qkvg_sf (over n_qkvg rows); no silent unit scale")
            _check_sf_blob(h_sf, "h_sf", self.batch * self.seq_len, self.geom.d_model)
            _check_sf_blob(w_qkvg_sf, "w_qkvg_sf", self.geom.n_qkvg, self.geom.d_model)
            for nm, sf in (("h_sf", h_sf), ("w_qkvg_sf", w_qkvg_sf)):
                if sf.device != h.device:
                    raise ValueError(f"{nm} must live on h's device {h.device}, got {sf.device}")
        elif h_sf is not None or w_qkvg_sf is not None:
            raise ValueError("h_sf / w_qkvg_sf are the MXFP8 scale-factor blobs; this block was declared without an MxQuantSpec")
        if self.o_fp4 is not None:
            if w_o_sf is None:
                raise ValueError(
                    f"fp4 O ({self.o_fp4.name}) execute needs w_o_sf (the F8_128x4 scale blob of the e2m1 W_o, over d_model rows x K=h_q*d_head); "
                    "no silent unit scale"
                )
            _check_sf_blob(
                w_o_sf,
                "w_o_sf",
                self.geom.d_model,
                self.geom.h_q * self.geom.d_head,
                block=self.o_fp4.block_size,
                sf_dtypes=(torch.uint8, self.o_fp4.sf_torch_dtype),
            )
            if w_o_sf.device != h.device:
                raise ValueError(f"w_o_sf must live on h's device {h.device}, got {w_o_sf.device}")
        elif w_o_sf is not None:
            raise ValueError("w_o_sf is the e2m1 W_o's scale blob of the fp4 O mode; this block was declared without MxQuantSpec.o_fp4")
        g = self.geom
        t = self.batch * self.seq_len
        # The token rows this step's stages address: the declared capacity, or the live count (the SDPA alone keeps T: its Q / O views
        # are the declared slab / workspace geometry and the live total is seq_lens' on device).
        rows = t if t_live is None else t_live

        def tok(x: torch.Tensor, name: str, width: int) -> torch.Tensor:
            """The caller's token matrix as its ``[rows, width]`` view (:func:`_token_rows`: the row count from the leading shape, so a
            ``rope_dim`` 0 table of width 0 flattens too; the first ``rows`` rows of a ``T_max``-row tensor, or exactly ``T_live`` rows)."""
            return _token_rows(x, name, width, t, t_live)

        # THE launch stream (Rule 5): the caller's ``current_stream``, else torch's
        # current stream on h's device -- resolved once, here, and handed to EVERY
        # stage.  The CuTe-DSL kernels take the raw CUstream int directly; the two
        # FROST GEMMs take it through ``run_proj_gemm(stream=)`` (the JIT plan's
        # own ``stream=``, or a cached per-(device, stream) cuDNN handle bound to
        # it on the graph route); the SDPA adapter takes it as ``current_stream``.
        # No stage derives its own stream from torch's current one, so a block
        # run under ``with torch.cuda.stream(s):`` -- or with an explicit stream --
        # cannot split across two streams (the GEMMs used to launch on the
        # default stream regardless, and the SDPA read an unwritten slab).
        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(h.device).cuda_stream
        ws = self._ws
        req = self.get_workspace_size()
        if workspace.numel() < req:
            raise ValueError(f"workspace is {workspace.numel()} bytes, need {req}")
        if self._quant_dev and not self._quant_dev_on_launch_stream:
            # The FIRST execute writes the per-tensor scalars' VALUES on ITS launch stream: compile() only ALLOCATED them (a fill
            # enqueued on the stream ambient at compile time would have no ordering against this stream, nor against a later
            # cross-stream update_quant_scales(), which pending work on that stream could let it undo).  Once per compile(); a later
            # update_quant_scales() writes on the caller's stream itself.  Nothing under o_fp4 ({}), nothing for bf16 (None).  A stream capture of this first
            # execute records these fills WITH their values, so its replays re-write them: warm up before capturing
            # (update_quant_scales's docstring).
            self._write_quant_dev(self.quant, stream)
            self._quant_dev_on_launch_stream = True
        # The prefill route's Q lengths (the constant S_q per sequence): the workspace slot, filled on this launch stream on EVERY execute
        # before the SDPA that reads it (Rule 5; a capture records the fill with its value, an eager execute fills its own).  None on
        # every other form (no slot reserved).
        decode_q_lens = self._fill_decode_q_lens(workspace, stream) if ws.decode_q_lens >= 0 else None

        fp8 = self.quant is not None
        mxfp8 = self.mxfp8
        act = self.act_dtype
        if self.fp8_fused:
            self._execute_fp8_fused(h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream)
            return
        if self.mxfp8_fused:
            self._execute_mxfp8_fused(h, h_sf, w_qkvg, w_qkvg_sf, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream, w_o_sf=w_o_sf)
            return
        # TRAINING (save_for_backward): validate the whole SavedForBackward record
        # FIRST -- before any launch -- and bind what the stages write THROUGH: the
        # caller's slab (proj_slab mode), the PRE-gate O, the LSE, rstd, and the
        # gate-copy destinations.  `_check_saved_set` is the contract (typed,
        # device-free, callable on a declared block).  An inference block reached
        # this line with saved=None: a record there was refused at the top.
        sv = self._check_saved_set(h, seq_lens, lse, saved) if self.save_for_backward else None
        rstd_q, rstd_k = (sv.rstd_q, sv.rstd_k) if sv is not None else (None, None)
        if sv is not None:
            lse = sv.lse
        # Stage (1)'s output: the workspace slab, or the caller-owned saved.proj_slab
        # (proj_slab save mode -- the GEMM binds `out` by pointer, so the slab the
        # backward reads is written exactly once, in place of the workspace one).
        proj = sv.proj if (sv is not None and sv.proj is not None) else _view(workspace, ws.proj, (t, g.n_qkvg), act)
        # The live-row views of the caller's token matrices and of the slab (the first `rows` rows; the whole buffers when rows == t).
        h_live, out_live = tok(h, "h", g.d_model), tok(out, "out", g.d_model)
        # under rope_position_tables_only (the decode indexer's fused form) the per-token tables are None: every stage that would read them
        # is not built (the fused prepare rotates from cos_pos / sin_pos)
        cos_live = None if cos is None else tok(cos, "cos", g.rope_dim)
        sin_live = None if sin is None else tok(sin, "sin", g.rope_dim)
        proj_live = proj if rows == t else proj[:rows]
        slot_live = slot_mapping if (slot_mapping is None or rows == t or int(slot_mapping.numel()) == rows) else slot_mapping[:rows]
        # In-place: there ARE no compact Q/K/V buffers -- the SDPA reads the
        # slab columns directly, so these are the same views stage (2)+(3)
        # normed over. Bound below, after `q_src`/`k_src`/`v_src` exist.
        q_c = k_c = v_c = None
        if not self.inplace_qkv and not fp8:
            q_c = _view(workspace, ws.q, (t, g.h_q, g.d_head), act)
            k_c = _view(workspace, ws.k, (t, g.h_kv, g.d_head), act)
            v_c = _view(workspace, ws.v, (t, g.h_kv, g.d_head), act)
        elif fp8 and ws.q >= 0:
            # FP8 / MXFP8 TRAINING: the carve reserved compact bf16 Q/K, so stage (2)+(3)
            # writes the normed Q/K OUT of place into them (the slab's Q/K bands stay
            # PRE-norm for the record) and the quantize stages read them.  Inference
            # (ws.q == -1) keeps the in-place norm and quantizes the slab bands.
            q_c = _view(workspace, ws.q, (t, g.h_q, g.d_head), act)
            k_c = _view(workspace, ws.k, (t, g.h_kv, g.d_head), act)
        # O: the SDPA's output -- the workspace slot under inference (stage (5)
        # then gates it IN PLACE: `o_gated is o`), the caller's PRE-gate saved.o
        # under training, where stage (5) gates OUT of place into the workspace
        # `o_gated` slot and stage (6) reads that.
        o = sv.o if sv is not None else _view(workspace, ws.o, (t, g.h_q, g.d_head), act)
        o_gated = _view(workspace, ws.o_gated, (t, g.h_q, g.d_head), act) if ws.o_gated >= 0 else o
        engine_ws = workspace[ws.engine_scratch :]
        sfq = sfk = sfv = None
        if fp8:
            e4 = torch.float8_e4m3fn
            q8 = _view(workspace, ws.q8, (t, g.h_q, g.d_head), e4)
            k8 = _view(workspace, ws.k8, (t, g.h_kv, g.d_head), e4)
            v8 = _view(workspace, ws.v8, (t, g.h_kv, g.d_head), e4)
            o8 = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4) if ws.o8 >= 0 else None  # not reserved under o_fp4
            qd = self._quant_dev
        o4 = sfo = None
        if self.o_fp4 is not None:
            o4, sfo = self._fp4_o_views(workspace, ws, t)
        if mxfp8:
            # The SDPA's own F8_128x4 SF blobs (flat uint8), written by the three quantize stages -- under THD the WHOLE packed
            # slot (the capacity count; the SDPA's binder derives the tile extent from the byte count), never the live total.
            sfq, sfk, sfv = self._sf_views(workspace, ws)

        o_q, o_g, o_k, o_v = g.qkvg_offsets[:4]  # the four DENSE bands; a five-band (indexer) slab's band is never an SDPA operand
        # Column slices of the fused projection, as strided views. Every consumer
        # addresses these strides natively -- no repack anywhere (Rule 2).
        q_src = _cols(proj, o_q, g.h_q, g.d_head)
        gate_src = _cols(proj, o_g, g.h_q, g.d_head)
        k_src = _cols(proj, o_k, g.h_kv, g.d_head)
        v_src = _cols(proj, o_v, g.h_kv, g.d_head)

        if self.fuse_norm_rope_effective:
            # (1)+(2)+(3): the fork norms + rotates the Q/K tiles in its
            # epilogue and writes the slab; rstd (if wanted) comes out of the
            # same launch (at decode row counts the declaration lowers to the
            # unfused chain below: fuse_norm_rope_effective, the constructor's C0 note).
            self._proj.execute(
                h_live,
                w_qkvg,
                proj_live,
                w_q_norm,
                w_k_norm,
                cos_live,
                sin_live,
                rstd_q if rstd_q is None else rstd_q.view(t, g.h_q),
                rstd_k if rstd_k is None else rstd_k.view(t, g.h_kv),
                stream=stream,
                # (4w) folded in (cache_write_fused): the post-RoPE K / V rows go to the pools from the epilogue's registers -- the same
                # bytes the slab gets, so the pool row is bitwise the row stage (4) consumes; stream order still orders this launch
                # before the paged read (nothing on the chain launches as a programmatic dependent launch).  The slot map is the
                # LIVE-ROW view (`slot_live`, as the separate cache-write stage takes it): the fused projection runs over `rows` rows and
                # its runner checks the slot map against that count, so under num_tokens_live a capacity-length slot map is sliced
                # to the live rows here, not rejected there.
                **({"k_cache": k_cache, "v_cache": v_cache, "slot_mapping": slot_live} if self.cache_write_fused else {}),
            )
        else:
            if mxfp8:
                # (1) [MXFP8: e4m3 codes x e4m3 codes with the E8M0 block scales
                # dequantized IN the MMA (block_scale_dequantize) -> bf16 slab; no alpha]
                self._proj.execute(h_live, w_qkvg, proj_live, engine_ws, sf_a=h_sf, sf_w=w_qkvg_sf, stream=stream)
            else:
                # (1) [FP8: e4m3 x e4m3, epilogue * alpha_qkvg, bf16 slab]
                self._proj.execute(h_live, w_qkvg, proj_live, engine_ws, alpha=qd["alpha_qkvg"] if fp8 else None, stream=stream)
            # (2)+(3) -- on EVERY unfused pipeline (bf16 / FP8 / MXFP8; the bring-up
            # bisect caught an MXFP8 arm that skipped this call: GEMM exact, SDPA
            # exact, end-to-end cos 0.73 -- the normed-Q/K comparison is the tell).
            # q_out/k_out=None means IN PLACE, which is the stage's own
            # default and is safe by construction: every lane holds its whole [D]
            # row in registers before it stores, and no lane touches another's.
            if self._prepare is not None:
                # (2+3+4w) FUSED: the same norm + RoPE in place AND the write-through of the post-RoPE K / V (+ the raw
                # indexer key) into the pools at slot_mapping, one launch -- bitwise the two stages below (`_cache_write`
                # is None under the knob, so the write-through block further down does not run again).  Under the decode indexer over
                # the pools the launch carries the three paged indexer arms instead of the raw-key pool: the queries into the workspace's
                # ix_q, the raw key into the ring, the completing block into the paged compressed-key cache, the slot report into ix_cslot.
                dec = self._indexer_decode
                ixv = dec.views(workspace, ws) if dec is not None else None
                self._prepare.execute(
                    q_src,
                    k_src,
                    v_src,
                    w_q_norm,
                    w_k_norm,
                    cos,
                    sin,
                    k_cache,
                    v_cache,
                    slot_mapping,
                    index_src=_cols(proj, g.index_k_raw_offset, g.qsa.index_kv_heads, g.qsa.index_head_dim) if g.index_band else None,
                    index_cache=index_k_raw if (g.index_band and dec is None) else None,
                    current_stream=stream,
                    **(
                        dict(
                            index_q_src=_cols(proj, g.qkvg_offsets[ProjBlock.INDEX], g.qsa.index_heads, g.qsa.index_head_dim),
                            index_q_out=ixv["qi"],
                            w_iq_norm=w_iq_norm,
                            ring=index_state_cache,
                            index_k_compressed=index_kv_cache,
                            w_ik_norm=w_ik_norm,
                            cos_pos=cos_pos,
                            sin_pos=sin_pos,
                            compressed_slot=ixv["cslot"],
                            kv_lens=kv_lens,
                            block_table=block_table,
                            index_state_block_table=index_state_block_table,
                            index_state_slot_mapping=index_state_slot_mapping,
                        )
                        if dec is not None
                        else {}
                    ),
                )
            else:
                self._norm_rope.execute(
                    q_src[:rows],
                    k_src[:rows],
                    w_q_norm,
                    w_k_norm,
                    cos_live,
                    sin_live,
                    q_out=None if q_c is None else q_c[:rows],
                    k_out=None if k_c is None else k_c[:rows],
                    rstd_q=rstd_q,
                    rstd_k=rstd_k,
                    current_stream=stream,
                    flat=True,
                    # (4w) folded in (cache_write_in_norm_rope): the slab's V band is copied and the post-RoPE K / V rows are written to the
                    # pools from the same tiles -- the pool row is bitwise the slab row stage (4) reads; the slot map is the LIVE-ROW view, as
                    # the separate stage takes it.
                    **({"v": v_src[:rows], "k_cache": k_cache, "v_cache": v_cache, "slot_mapping": slot_live} if self.cache_write_in_norm_rope else {}),
                )
        if sv is not None and sv.gate_dst is not None:
            # (3g) TRAINING, gate-copy save mode: the slab's GATE band -> the compact
            # caller `saved.gate` (ONE strided copy).  q_pre / k_pre likewise, only
            # when the caller passed buffers -- legal here because training is out
            # of place, so the slab's Q/K columns are still PRE-norm after (2)+(3).
            self._gate_copy.execute(gate_src, sv.gate_dst, current_stream=stream)
            if sv.q_pre_dst is not None:
                self._gate_copy.execute(q_src, sv.q_pre_dst, current_stream=stream)
            if sv.k_pre_dst is not None:
                (self._compact_v if self._compact_v is not None else self._kpre_copy).execute(k_src, sv.k_pre_dst, current_stream=stream)
        # The NORMED Q/K the quantize stages read: the compact slots when stage (2)+(3)
        # wrote out of place (training), else the slab bands it normed in place
        # (inference).  Both quantize artifacts take a dynamic token stride, so one
        # recipe serves either source.
        q_n, k_n = (q_c, k_c) if q_c is not None else (q_src, k_src)
        if mxfp8:
            # (3q) bf16 normed Q/K (+ the slab's V band) -> compact e4m3 + the SDPA's
            # F8_128x4 SF blobs: Q / K ROWWISE (blocks along D), V COLUMNWISE (blocks
            # along S, D-plane-major SF).  This IS the compaction the MXFP8 SDPA needs.
            # THD: the PACKED arm takes the lengths tensor (the very one the SDPA reads) in place of the (1, T) geometry.
            qkw = dict(seq_lens=seq_lens) if self.thd else dict(batch=self.batch, seq_len=self.seq_len)
            self._quant_q.execute(q_n, q8, sfq, current_stream=stream, **qkw)
            self._quant_k.execute(k_n, k8, sfk, current_stream=stream, **qkw)
            self._quant_v.execute(v_src, v8, sfv, current_stream=stream, **qkw)
            q_c, k_c, v_c = q8, k8, v8
        elif fp8:
            # (3q) bf16 normed Q/K (+ the slab's V band) -> compact e4m3.  This IS the
            # compaction the FP8 SDPA needs (its adapter path takes no declared slab strides).
            self._quant_q.execute(q_n, q8, qd["scale_q"], current_stream=stream)
            self._quant_kv.execute(k_n, k8, qd["scale_k"], current_stream=stream)
            self._quant_kv.execute(v_src, v8, qd["scale_v"], current_stream=stream)
            q_c, k_c, v_c = q8, k8, v8
        elif self.inplace_qkv:
            # The normed Q/K are the slab columns; V was never moved. All three
            # go to the SDPA at the slab's token stride -- no stage (3b), no
            # compact buffers, no copy anywhere.
            q_c, k_c, v_c = q_src, k_src, v_src
        elif self._compact_v is None:
            # The e4m3 decode mode: Q / K normed OUT of place (the compact slots -- the tile's per-item Q is a view of them), V never
            # compacted: the write-through reads the slab's V band and stage (4) reads the pools.
            v_c = v_src
        else:
            # (3b) -- exists only to hand the SDPA a compact V.
            self._compact_v.execute(v_src[:rows], v_c[:rows], current_stream=stream)
        if self._indexer is not None:
            # (4i) the IN-BLOCK INDEXER: the slab's fifth band -> the per-query block list stage (4) consumes (the caller's
            # block_ids_out when given, else the workspace slot, else the plan-time identity list below the bound); the
            # compress reads the band's raw key in place, the scorer the normed queries and compressed keys in the workspace.
            # The list carries the derived count as its -1 padding, so stage (4) takes no block_lens.
            block_ids = self._indexer.execute(
                proj,
                cos,
                sin,
                w_iq_norm,
                w_ik_norm,
                seq_lens=seq_lens,
                workspace=workspace,
                lay=ws,
                index_k_compressed=index_k_compressed,
                block_ids_out=block_ids_out,
                block_lens_out=block_lens_out,
                current_stream=stream,
            )
            block_lens = None
        elif self._indexer_decode is not None:
            # (4i) the decode step over the pools: (unfused) the indexer queries into ix_q and the paged compress step -- the ring, the block a
            # row completes --, then the paged scorer over the stack's compressed-key cache and the decode top-k: the per-row list stage (4)
            # consumes (the caller's block_ids_out when given, else the workspace slot), its -1 padding the count, so stage (4) takes no block_lens.
            block_ids = self._indexer_decode.execute(
                proj,
                workspace=workspace,
                lay=ws,
                w_iq_norm=w_iq_norm,
                w_ik_norm=w_ik_norm,
                cos=cos,
                sin=sin,
                cos_pos=cos_pos,
                sin_pos=sin_pos,
                index_kv_cache=index_kv_cache,
                index_state_cache=index_state_cache,
                index_state_block_table=index_state_block_table,
                index_state_slot_mapping=index_state_slot_mapping,
                kv_lens=kv_lens,
                block_table=block_table,
                slot_dtype=slot_mapping.dtype,
                block_ids_out=block_ids_out,
                block_lens_out=block_lens_out,
                current_stream=stream,
            )
            block_lens = None
        if self._cache_write is not None:
            # (4w) SERVING write-through: the VERY operands stage (4) is about to read -- the post-RoPE K and the V (the
            # slab bands in place, or the compact buffers out of place) -- into the paged pools at slot_mapping, so a
            # pool row is bitwise the row attention consumes; the raw indexer key (the fifth band's key head, pre-norm,
            # un-rotated: what a serving indexer cache keeps) into its own pool when the band is declared.  Nothing the
            # SDPA reads is touched, so `out` is bitwise the block's without write-through.  Only the bf16 / f16
            # pipelines reach here (the quantized ones declined at declaration); their k_c / v_c are 16-bit views -- the
            # packed block's too (B = 1, S = T: the same [T, H_kv, D] views at the slab's token stride).
            # the raw-key arm: the band's pool (not under the decode indexer over the pools, whose raw key entered the ring -- the stage was
            # built without the arm and `index_k_raw` was refused up front)
            raw_arm = bool(getattr(self._cache_write, "index_band", g.index_band))
            cw_kw = dict(
                index_src=_cols(proj_live, g.index_k_raw_offset, g.qsa.index_kv_heads, g.qsa.index_head_dim) if raw_arm else None,
                index_cache=index_k_raw if raw_arm else None,
                current_stream=stream,
            )
            if self.kv_cache_e4m3:
                # The e4m3 cache: the reciprocals of the caller's DEscales into the workspace's two fp32 words ON THE LAUNCH STREAM (Rule 5;
                # a device op, no host read, capture-safe), then the quantizing write (two casts + the bit copy through the bf16 views).
                inv = _view(workspace, ws.kv_inv, (2,), torch.float32)
                with torch.cuda.stream(as_torch_stream(stream, h.device)):
                    torch.reciprocal(k_scale.reshape(1), out=inv[0:1])
                    torch.reciprocal(v_scale.reshape(1), out=inv[1:2])
                self._cache_write.execute(
                    k_c[:rows],
                    v_c[:rows],
                    _view(workspace, ws.kc8, (t, g.h_kv, g.d_head), KV_CACHE_E4M3)[:rows],
                    _view(workspace, ws.vc8, (t, g.h_kv, g.d_head), KV_CACHE_E4M3)[:rows],
                    inv[0:1],
                    inv[1:2],
                    k_cache,
                    v_cache,
                    slot_live,
                    **cw_kw,
                )
            else:
                self._cache_write.execute(k_c[:rows], v_c[:rows], k_cache, v_cache, slot_live, **cw_kw)
        # (4) [+ (5) under fuse_gate: the SDPA gates the SUBSTITUTED O inside
        # its epilogue, after the dead-row select, and writes O_gated.]
        # Under QsaSpec stage (4) is the index-list sparse core: the SAME operands at the SAME strides plus the caller's
        # block list (form-checked at the top of execute; the kernel reads its values on device).
        sparse_kw = dict(block_ids=block_ids, block_lens=block_lens) if self.qsa is not None else {}
        # The EXTEND form (kv_source="paged"): K / V are read from the POOLS the write-through just filled -- every sequence's
        # prefix and this step's tokens -- through the block table over [0, kv_lens[b]); the slab's K / V bands are unread by
        # the SDPA.  On the plain chain stream order is the ONLY ordering between the write and the read (the cache write launches
        # without the PDL attribute; the prefill bodies wait on no dependent grid); under use_pdl=True the cache write launches with
        # the attribute and the decode tile waits on its dependent grid before its first K / V load, while a prefill tile -- which
        # carries no sites -- launches plain, so the link serializes and the read still follows the write either way.
        paged = self.kv_source == KV_SOURCE_PAGED
        # The pools and the table reach the stage that reads them: the dense SDPA stage (the EXTEND / DECODE forms) or, on a QsaSpec
        # geometry, the sparse stage (the block-sparse DECODE / extend-with-prefix mode: the same paged read through the sparse core,
        # kv_lens bound to the KV side only, the list in the form the declaration derived from S_q).
        paged_kw = dict(block_table=block_table, kv_lens=kv_lens, k_cache=k_cache, v_cache=v_cache) if paged else {}
        if self.upcast_decode:
            # The e4m3 cache over the pools: the gathered cast into the workspace scratch, then the d256 decode tile over it -- the compact
            # normed Q and the O slot per (token, KV head) item; the scales read on device.
            items = t * g.h_kv
            self._sdpa.execute(
                q_c,
                o,
                k_cache=k_cache,
                v_cache=v_cache,
                block_table=block_table,
                kv_lens=kv_lens,
                block_ids=block_ids,
                block_lens=block_lens,
                k_scale=k_scale,
                v_scale=v_scale,
                scratch=_view(workspace, ws.kv_gathered, (items, 2, _UPCAST_ROWS, _UPCAST_D), act),
                kv_len_t=_view(workspace, ws.kv_len_t, (items,), torch.int32),
                workspace=engine_ws,
                current_stream=stream,
            )
        else:
            self._sdpa.execute(
                q_c.view(self.batch, self.seq_len, g.h_q, g.d_head),
                k_c.view(self.batch, self.seq_len, g.h_kv, g.d_head),
                v_c.view(self.batch, self.seq_len, g.h_kv, g.d_head),
                o.view(self.batch, self.seq_len, g.h_q, g.d_head),
                lse=lse,
                # the prefill route hands the paged THD leg the block's own constant Q lengths (the workspace slot, [B] int32, all S_q);
                # every other form the caller's
                seq_lens=decode_q_lens if decode_q_lens is not None else seq_lens,
                workspace=engine_ws,
                current_stream=cuda.CUstream(stream),
                gate=gate_src.view(self.batch, self.seq_len, g.h_q, g.d_head) if self.gate_fused else None,
                # per-tensor FP8: scalar descales; MXFP8: the three SF blobs (and NO scalars -- _Sdpa refuses them)
                descale_q=qd["descale_q"] if (fp8 and not mxfp8) else None,
                descale_k=qd["descale_k"] if (fp8 and not mxfp8) else None,
                descale_v=qd["descale_v"] if (fp8 and not mxfp8) else None,
                sf_q=sfq,
                sf_k=sfk,
                sf_v=sfv,
                **sparse_kw,
                **paged_kw,
            )
        if not self.gate_fused:
            # (5) -- gates the SUBSTITUTED O: the SDPA epilogue already selected
            # O := 0 on dead rows, so no residue reaches the sigmoid.  In place
            # under inference (`o_gated is o`); OUT of place under training, where
            # `o` is the caller's pre-gate saved.o and must survive for dG.
            self._gate.execute(o[:rows], gate_src[:rows], o_gated[:rows], current_stream=stream)
        if self.o_fp4 is not None:
            # (5q') bf16 gated O -> e2m1 codes + the out_proj GEMM's F8_128x4 blob (block scales, no per-tensor
            # scale); (6') fp4 x fp4 block-scale out projection, both blobs dequantized IN the MMA (no alpha).
            self._quant_o.execute(o_gated, o4, sfo, current_stream=stream)
            self._out_proj.execute(o4, w_o, out.view(t, g.d_model), engine_ws, sf_a=sfo, sf_w=w_o_sf, stream=stream)
        elif fp8:
            # (5q) bf16 gated O -> e4m3 for the FP8 out projection (PER-TENSOR
            # scale_o under both families, D1); (6) folds (1/scale_o) *
            # descale_w_o into its epilogue.
            self._quant_o.execute(o_gated, o8, qd["scale_o"], current_stream=stream)
            self._out_proj.execute(o8.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)
        else:
            # (6) -- reads the gated O: the workspace `o` (inference) or `o_gated` (training); the live rows under num_tokens_live.
            self._out_proj.execute(o_gated[:rows].view(rows, g.h_q * g.d_head), w_o, out_live, engine_ws, stream=stream)

    def _sf_views(self, workspace: torch.Tensor, ws: "_Intermediates") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three MXFP8 scale-factor slots as flat uint8 views, sized exactly as the carve reserved them: ``_sf_slot_bytes``
        dense, ``_sf_slot_bytes_packed`` under ``thd`` (the capacity ``n_cap`` tiles per head -- the packed quantize stages write
        every byte and the SDPA's binder reads the tile extent off the byte count, so Q / K / V are each handed the WHOLE slot)."""
        g = self.geom

        def nbytes(h: int) -> int:
            if self.thd:
                return _sf_slot_bytes_packed(self.num_sequences, h, self.batch * self.seq_len, g.d_head)
            return _sf_slot_bytes(self.batch, h, self.seq_len, g.d_head)

        return (
            _view(workspace, ws.sf_q, (nbytes(g.h_q),), torch.uint8),
            _view(workspace, ws.sf_k, (nbytes(g.h_kv),), torch.uint8),
            _view(workspace, ws.sf_v, (nbytes(g.h_kv),), torch.uint8),
        )

    def _fp4_o_views(self, workspace: torch.Tensor, ws: "_Intermediates", t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two fp4-O workspace views: ``o4`` -- the packed e2m1 gated O as ``float4_e2m1fn_x2 [T, H_q*D/2]``
        (a re-view of the uint8 bytes, no copy; what ``run_proj_gemm`` binds as the fp4 A operand) -- and ``sfo``, the
        flat uint8 blob of ``proj_gemm.sf_blob_bytes(T, H_q*D, block)`` bytes the quantize stage writes and the
        block-scale GEMM reads."""
        from .kernels.proj_gemm import sf_blob_bytes

        g = self.geom
        o4 = _view(workspace, ws.o4, (t, g.h_q * g.d_head // 2), torch.uint8).view(_FP4_X2)
        sfo = _view(workspace, ws.sf_o, (sf_blob_bytes(t, g.h_q * g.d_head, self.o_fp4.block_size),), torch.uint8)
        return o4, sfo

    def _execute_fp8_fused(self, h, w_qkvg, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream) -> None:
        """The FULLY FUSED FP8 pipeline: three launches, three workspace buffers.

        ::

            (1') proj+norm+rope+quant  h8, W8              -> q8 [T, H_q, D] / k8, v8 [T, H_kv, D] e4m3 (COMPACT), gate16 [T, H_q, D] bf16
            (4') sdpa+gate             q8, k8, v8, gate16  -> o8 [T, H_q, D] e4m3   (descale_v*scale_o folded, gate after the select)
            (6)  out_proj              o8                  -> out                    (alpha_o epilogue)

        No bf16 slab, no bf16 O, no quantize pass: every intermediate is a view
        of the five slots ``_plan_workspace`` reserved.  Q/K/V are compact BSHD
        (``token_stride == 0``), the SAME layout the unfused FP8 SDPA reads --
        the 2-D ``[T, h*d]`` views go to the GEMM runner, the ``[B, S, H, d]``
        views of the same bytes to the gated FP8 SDPA; nothing is strided,
        nothing is repacked (Rule 2).

        ``w_q_norm`` / ``w_k_norm`` are ``None`` (both) under ``geometry.qk_norm=False``
        -- already checked by ``execute``; the fork's runner receives them as-is.
        """
        g = self.geom
        b, s = self.batch, self.seq_len
        t = b * s
        ws = self._ws
        qd = self._quant_dev
        e4 = torch.float8_e4m3fn
        q8 = _view(workspace, ws.q8, (t, g.h_q, g.d_head), e4)
        k8 = _view(workspace, ws.k8, (t, g.h_kv, g.d_head), e4)
        v8 = _view(workspace, ws.v8, (t, g.h_kv, g.d_head), e4)
        gate16 = _view(workspace, ws.gate16, (t, g.h_q, g.d_head), self.act_dtype)
        o8 = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4)
        engine_ws = workspace[ws.engine_scratch :]
        # (1'): alpha_qkvg / scale_q / scale_k / scale_v ride the stage's qscal vector.
        # The runner takes the four outputs 2-D ([T, h*d]); the [T, H, D] / [B, S, H, D] views are for the SDPA.
        self._proj.execute_fp8(
            h.view(t, g.d_model),
            w_qkvg,
            q8.view(t, g.h_q * g.d_head),
            k8.view(t, g.h_kv * g.d_head),
            v8.view(t, g.h_kv * g.d_head),
            gate16.view(t, g.h_q * g.d_head),
            w_q_norm,
            w_k_norm,
            cos,
            sin,
            stream=stream,
        )
        # (4'): e4m3 in, e4m3 out; the SDPA gates the SUBSTITUTED O (after the
        # dead-row select, per element) and casts once, with saturation.
        self._sdpa.execute(
            q8.view(b, s, g.h_q, g.d_head),
            k8.view(b, s, g.h_kv, g.d_head),
            v8.view(b, s, g.h_kv, g.d_head),
            o8.view(b, s, g.h_q, g.d_head),
            lse=lse,
            seq_lens=seq_lens,
            workspace=engine_ws,
            current_stream=cuda.CUstream(stream),
            gate=gate16.view(b, s, g.h_q, g.d_head),
            descale_q=qd["descale_q"],
            descale_k=qd["descale_k"],
            descale_v=qd["descale_v"],
            scale_o=qd["scale_o"],
        )
        # (6): e4m3 O_gated @ W_o^T with (1/scale_o) * descale_w_o in the epilogue.
        self._out_proj.execute(o8.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)

    def _execute_mxfp8_fused(self, h, h_sf, w_qkvg, w_qkvg_sf, w_q_norm, w_k_norm, cos, sin, w_o, out, workspace, seq_lens, lse, stream, w_o_sf=None) -> None:
        """The FULLY FUSED MXFP8 pipeline: three launches, eight workspace slots.

        ::

            (1'') proj+norm+rope+block-quant  h8+sf_h, W8+sf_w      -> q8 / k8 / v8 e4m3 (COMPACT) + sf_q / sf_k / sf_v (F8_128x4) + gate16 bf16
            (4'') sdpa+gate                   q8, k8, v8, SF, gate16 -> o8 e4m3 UNSCALED (block scales dequant in-MMA; gate after the select)
            (6)   out_proj                    o8                    -> out   (alpha_o = descale_w_o / 1.0)

        Mirrors :meth:`_execute_fp8_fused` with the three SF slots added: the
        fork's runner (``run_fused_proj_gemm_mxfp8``, frozen ABI) writes the
        SDPA's own SF layouts (Q/K per-(b, h, s_tile) 1024-B tiles, V
        D-plane-major), so the gated production MXFP8 SDPA reads exactly what
        the unfused pipeline's quantize stages would have written.  No scalar
        rides into the SDPA (``_Sdpa`` refuses ``descale_*`` / ``scale_o`` under
        MXFP8); ``MxQuantSpec.scale_o == 1.0`` was pinned at declaration (D8).

        ``w_o_sf`` (appended) -- fp4 O (config row 10, FOUR launches): the gated SDPA writes **bf16** ``o``
        (``o8`` is not reserved), ``quantize_fp4_o`` turns it into ``o4`` + ``sf_o``, and the out projection
        is the fp4 x fp4 block-scale GEMM over ``sf_o`` / ``w_o_sf`` (no alpha).
        """
        g = self.geom
        b, s = self.batch, self.seq_len
        t = b * s
        ws = self._ws
        qd = self._quant_dev
        e4 = torch.float8_e4m3fn
        q8 = _view(workspace, ws.q8, (t, g.h_q, g.d_head), e4)
        k8 = _view(workspace, ws.k8, (t, g.h_kv, g.d_head), e4)
        v8 = _view(workspace, ws.v8, (t, g.h_kv, g.d_head), e4)
        gate16 = _view(workspace, ws.gate16, (t, g.h_q, g.d_head), self.act_dtype)
        # e4m3 O for the per-tensor out_proj, or (o_fp4) bf16 O for the quantize_fp4 stage -- one slot exists, never both.
        o_sdpa = _view(workspace, ws.o8, (t, g.h_q, g.d_head), e4) if self.o_fp4 is None else _view(workspace, ws.o, (t, g.h_q, g.d_head), self.act_dtype)
        sfq, sfk, sfv = self._sf_views(workspace, ws)  # the fully fused pipeline is dense-only: the dense slot sizes
        engine_ws = workspace[ws.engine_scratch :]
        # (1''): the runner takes the four data outputs 2-D ([T, h*d]) + the three SF blobs flat.
        self._proj.execute_mxfp8(
            h.view(t, g.d_model),
            h_sf,
            w_qkvg,
            w_qkvg_sf,
            q8.view(t, g.h_q * g.d_head),
            k8.view(t, g.h_kv * g.d_head),
            v8.view(t, g.h_kv * g.d_head),
            gate16.view(t, g.h_q * g.d_head),
            sfq,
            sfk,
            sfv,
            w_q_norm,
            w_k_norm,
            cos,
            sin,
            stream=stream,
        )
        # (4''): e4m3 in (+ block scales), e4m3 out UNSCALED (or bf16 out under o_fp4); the SDPA gates the
        # SUBSTITUTED O (after the dead-row select, per element) and casts once.
        self._sdpa.execute(
            q8.view(b, s, g.h_q, g.d_head),
            k8.view(b, s, g.h_kv, g.d_head),
            v8.view(b, s, g.h_kv, g.d_head),
            o_sdpa.view(b, s, g.h_q, g.d_head),
            lse=lse,
            seq_lens=seq_lens,
            workspace=engine_ws,
            current_stream=cuda.CUstream(stream),
            gate=gate16.view(b, s, g.h_q, g.d_head),
            sf_q=sfq,
            sf_k=sfk,
            sf_v=sfv,
        )
        if self.o_fp4 is not None:
            # (5q') bf16 O_gated -> e2m1 codes + the out_proj blob; (6') fp4 x fp4 block-scale out projection (no alpha).
            o4, sfo = self._fp4_o_views(workspace, ws, t)
            self._quant_o.execute(o_sdpa, o4, sfo, current_stream=stream)
            self._out_proj.execute(o4, w_o, out.view(t, g.d_model), engine_ws, sf_a=sfo, sf_w=w_o_sf, stream=stream)
        else:
            # (6): e4m3 O_gated @ W_o^T with descale_w_o (scale_o == 1.0) in the epilogue.
            self._out_proj.execute(o_sdpa.view(t, g.h_q * g.d_head), w_o, out.view(t, g.d_model), engine_ws, alpha=qd["alpha_o"], stream=stream)


# ---------------------------------------------------------------------------
# 7. Convenience wrapper — allocates, then delegates
# ---------------------------------------------------------------------------


def gated_attention_block_forward(
    h: torch.Tensor,
    w_qkvg: torch.Tensor,
    w_q_norm: Optional[torch.Tensor],  # None (both) iff geometry.qk_norm is False
    w_k_norm: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    geometry: GatedAttentionBlockGeometry,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    save_for_backward: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate outputs + workspace, cache the compiled block, and run it.

    Returns ``{"out": ..., "lse": ..., "saved": ...}``; the optional entries are
    ``None`` unless requested. Allocation happens HERE and only here — the class
    API above never allocates.
    """
    raise NotImplementedError("gated_attention_block_forward")
