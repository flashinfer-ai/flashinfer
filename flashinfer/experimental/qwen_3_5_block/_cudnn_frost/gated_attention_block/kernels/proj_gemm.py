# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stages (1) and (6) of the gated attention block: the two dense projections.

**This writes no kernel.** Both projections are ordinary dense ``nn.Linear``-shaped
GEMMs, and this repo already ships a production FROST one —
``gemm/frost/kernel_templates/sm100_matmul.py``: persistent, double-TMEM, CLC
dynamic scheduler, ``cta_group`` 1 or 2, a tuned tile-config catalog and a
codegen'd epilogue, behind an engine that competes in the graph's ranked plan
list. Writing a second tcgen05 GEMM to do the same thing would be weeks of work
and a second thing to maintain.

So this module is a thin driver: build the two-operand matmul graph, pin the
``frost_gemm`` plan, and hand it the block's buffers.

    stage (1)   h        [M, d_model]  @ W_qkvg^T  ->  [M, N_qkvg]
    stage (6)   O_gated  [M, H_q*D]    @ W_o^T     ->  [M, d_model]

with ``M = B*S``. One code path serves both: they are the same op at different
shapes, which is why there is one ``_Projection`` stage in ``api.py`` and not
two kernels.

Operand layout follows the graph API's matmul convention, and it happens to be
exactly what a checkpoint already holds: ``A`` is ``[1, M, K]`` row-major, and
``B`` is declared ``[1, K, N]`` with stride ``[K*N, 1, K]`` — i.e. the weight
stored ``[N, K]`` row-major (``nn.Linear``), read transposed. No repacking, at
plan time or execute time.

**When this gets FORKED — and the fork already has prior art in this repo.**
``sdpa/bwd/kernels/bprop_matmul_blackwell.py`` forked this same template to give its
stage-3 gradient GEMMs a 2-D ``(batch, head)`` batch: it takes a **rendered
dense-bf16 expansion** of the template and widens the TMA descriptors to 4-D
``[k, m, h, b]``, keeping the mainloop, the CLC scheduler, the TMEM pipeline and
the epilogue identical. It carries a bidirectional "apply fixes both ways" note.

**Start our fork from that one, not from the template and not from scratch**, for
two things the unforked engine cannot express:

* **the four-output-buffer epilogue** (§ 1 of ``api.py``): stage (1)'s N axis is
  ``[Q | GATE | K | V]`` and each block wants its own compact ``[B, S, H, D]``
  destination. That is a per-N-tile choice among several output TMA descriptors
  — the same *kind* of descriptor surgery the backward's fork already did, and
  its 4-D output descriptors are close to the shape Q/K/V want.
* **the quantization epilogue** (FP8/MXFP8): Q/K rowwise, V columnwise, GATE
  passthrough — three scale-factor layouts off one GEMM. Note V's MXFP8 scale
  factors are D-PLANE-major while Q/K's are per-tile contiguous
  (``mma-tma-matrix.md`` § 7); reusing one SF descriptor builder for all three
  is a silent wrong answer that only shows up at ``d > 128``.

Two process rules that apply the moment the fork starts, both learned the hard
way here: **diff the two RENDERINGS, not your edit** (a constants swap that
misses half a descriptor pair produces ``cos ~= 0.006``, not a crash —
``frost-tile-dsl.md`` § 5), and add the matching cross-reference note in BOTH
files so a fix in one is not silently lost in the other.

Until then the unforked engine is the honest baseline, and any fork starts by
measuring against it — which is what this module exists to make cheap.

**Rubin (sm_107) perf note.** The tile catalog already sizes its K-pipeline from
the LIVE device budget (``tile_config._sm_smem_budget_bytes_of`` takes
``max(optin, oversized)``), and on Rubin that is **327 KiB, not SM100's 227** —
so deeper staging is available to the unforked engine today, capped at 16
stages. Two things to check before assuming it is taken: whether the chosen
config is SMEM-limited or already at the 16-stage cap, and — the landmine — that
**no MMA-operand SMEM buffer starts at or past 262144 bytes**. These templates
carry NO ``desc_version`` handling, and a version-0 tcgen05 descriptor's
``start_address`` is 14 bits: past 256 KiB it wraps and the accumulator comes out
EXACTLY ZERO, with no crash (``mma-tma-matrix.md`` § 6). That is why
``test_proj_gemm.py::test_output_is_not_silently_zero`` exists and asserts
non-zero explicitly rather than only a cosine, which would be NaN against a zero
tensor.

**The FROST GEMM engine is opt-in**: without ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``
in the environment the backend's ranked list has no ``frost_gemm`` plan.  The flag is
read LIVE, when the ranked list is built (``engines/manifest.py``: ``opt_in_engines_enabled``),
not once at ``import cudnn`` -- so it has to be set before :func:`build_proj_gemm` runs,
and a host module must NEVER set it for the process: it flips every opt-in engine into
every FE graph the process builds.  :func:`build_proj_gemm` checks for the plan by name
and raises rather than letting an unpinned graph run a cuDNN backend plan quietly; a
dense bf16 / f16 projection whose N the forced 256-wide tile divides needs no flag at
all -- it takes the forced-tile JIT-only route (``plan.route == "jit-only"``, no backend
plan, no cuDNN handle at execute), which is what a serving stack compiles under.

**CUDA-graph capture** (``cudnn.frost.capture``): :func:`build_proj_gemm` is refused
under capture (it builds a cuDNN graph -- a handle in the ``pygraph`` constructor -- and
JIT-compiles a kernel), and so is the GRAPH route's lazy cuDNN handle creation
(:func:`handle_for_stream`, the first execute per (device, stream)); both are typed
``RuntimeError``s naming the route and the remedy.  ``build_proj_gemm(capture_safe=True)``
is the serving declaration: the plan never enters the ranked list (the graph is built only
to be analyzed -- no heuristics query, no backend plan, no FROST-engine compile, whatever
the opt-in flag says), so it ends on the forced-tile JIT route by construction, and a
refused forced compile is a typed ``NotImplementedError`` at compile time, never the
warning + graph-heuristic fallback the plain declaration takes.

**The BACKWARD's four projection GEMMs -- transposed operands as VIEWS.** The
``nn.Linear`` orientation (``api_bwd.py``: ``dW = dY^T @ X``, ``dX = dY @ W``) needs
an M-major A and an N-major B, which the shipped engine already renders (it
infers each operand's major from the graph's strides). :func:`build_proj_gemm`
takes them as the appended ``a_major`` / ``b_major`` kwargs -- the dims stay
``[1, m, k]`` / ``[1, k, n]``, only the strides move -- and the two drivers bind
views, no repack:

    build                          A bind (view)                      B bind (view)                 C
    a_major="k", b_major="k"       X [T, K] row-major (the forward)   W [N, K] row-major, read ^T   [T, N]
    a_major="m", b_major="n"       dY[T, rows]^T  -> run_wgrad_gemm   X [T, cols] row-major         dW [rows, cols]
    a_major="k", b_major="n"       dY[T, K] row-major -> run_dgrad_gemm   W [K, N] row-major (un-transposed)   dX [T, N]

    B1  dW_o    = dY^T    @ O_gated   ("m", "n"; m=dm, k=T, n=HD)     B2  dO_gated = dY    @ W_o     ("k", "n"; m=T, k=dm, n=HD)
    B7  dW_qkvg = dQKVG^T @ h         ("m", "n"; m=N,  k=T, n=dm)     B8  dh       = dQKVG @ W_qkvg  ("k", "n"; m=T, k=N,  n=dm)

A view binds ONLY against a plan declared with its majors, and ONLY with exactly
the declared strides (size-1 axes aside): the graph fallback binds pointers
against the DECLARED strides with no check (a padded view is a silent
reinterpretation), and the JIT route re-labels B from graph order into kernel
order only when the runtime ``(shape, stride)`` EQUALS the declaration
(``sm100/compiler.py`` ``_lower``: a padded N-major B trips its generic
``gave_up["input layout"]``, a padded M-major A would be accepted) -- so
:func:`run_wgrad_gemm` / :func:`run_dgrad_gemm` assert the whole declared layout
of every view before binding, on both routes (a typed ``ValueError`` naming the
operand and both strides). All four backward GEMMs take
the FORCED ``..._cluster2x1_2ctamma`` tile (every ``n`` is ``d_model`` or
``h_q*d_head``, both ``% 256 == 0``); its MN-major rendering on cc 10.7 is pinned
by ``test_proj_gemm_bwd.py::test_forced_tile_renders_mn_major_on_cc107`` (bf16) and
``test_fp8_mn_major_matches_fp64_on_cc107`` (e4m3, K32 and K64 -- the triples in
:data:`FP8_MN_MAJOR_VALIDATED`). The appended ``split_k`` kwarg (0 / 1 / S >= 2) is
documented on :func:`build_proj_gemm`.

**The QUANTIZED backward's GEMMs** ride the same drivers: per-tensor e4m3 operands take
the MN-major renderings above with the descale product bound as the ``[1, 1, 1]`` fp32
``alpha`` epilogue (:func:`device_alpha` writes it into the caller's workspace slot on the
launch stream), and the 64-byte MMA K form when the stage asks for it explicitly
(``mma_tile_k_bytes=64``; the forward's plans stay at K32).  Block-scale (MXFP8) operands
keep every operand K-major -- the TE "columnwise" artifacts ``dQKVG^T [N, T]`` / ``h^T [dm, T]``
with their F8_128x4 scale factors along T -- through :func:`run_wgrad_gemm_block_scale`
/ :func:`run_dgrad_gemm_block_scale`, which are the forward's K-major binding at the
backward's shapes (``block_scale=True`` needs ``K % 32 == 0``, so a token-axis wgrad needs
``T % 32 == 0``: a typed decline at plan time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Any, Optional

import torch

_FROST_GEMM_PLAN = "frost_gemm"


class _NoFrostPlan(RuntimeError):
    """``frost_gemm`` is not in the graph's ranked plan list (opt-in flag unset, arch outside
    the template family, or the engine declined the shape).  A ``RuntimeError`` so a dense
    caller sees it as before; a DISTINCT type so the block-scale fallback can catch exactly
    this and the typed backend decline, and let a genuine backend failure propagate."""


def _frost_plan_index(names) -> Optional[int]:
    """Index of the top-ranked FROST GEMM plan in ``names``, or None.

    A plan name is the engine name alone (older builds) or the engine name
    followed by its knob bracket, ``frost_gemm[CTA_GROUP=2, ...]`` (the shared
    backend/python knob vocabulary).  Both spell the same engine.
    """
    for i, n in enumerate(names):
        if n == _FROST_GEMM_PLAN or n.startswith(_FROST_GEMM_PLAN + "["):
            return i
    return None


# ---------------------------------------------------------------------------
# The FORK: stage (1) with stages (2)+(3) fused into its epilogue
# ---------------------------------------------------------------------------
#
# ``kernels/proj_gemm_norm_rope.py`` is the rendered dense-bf16 expansion of the
# shipped template for the block's tile config, plus a norm+RoPE arm in the
# epilogue for the Q and K tiles.  It is a FROST template: the loader executes
# it once per ``NormRopeFusionParams`` with the params injected as
# ``FROST_TEMPLATE_PARAMS`` (engine contract S6), so geometries coexist in one
# process and nothing is read from the environment.  The kernel's own docstring
# carries the design; this module carries the params and the launch.
#
# ``kernels/proj_gemm_norm_rope_fp8.py`` is the SAME fusion on the FP8 pipeline
# (``NormRopeFusionParams(quant_fp8=True)``): the rendered e4m3 x e4m3 -> fp32
# GEMM with the per-tensor descale ``alpha``, whose epilogue also QUANTIZES --
# Q/K normed + rotated then written as e4m3 into COMPACT per-tensor outputs
# ``q8 [T, h_q*d]`` / ``k8 [T, h_kv*d]``, V as e4m3 into ``v8 [T, h_kv*d]``, the
# GATE as bf16 into ``gate16 [T, h_q*d]`` (each == compact BSHD, exactly what
# the unfused path hands the FP8 SDPA; there is no ``[T, N_QKV]`` slab any
# more -- round 2 of the FP8 fusion design).  It has its own runner
# (``run_fused_proj_gemm_fp8``): four outputs and a ``[4]`` fp32 scale vector
# (``[alpha, scale_q, scale_k, scale_v]``, read IN-KERNEL) replace the bf16
# runner's single slab, and the bf16 runner refuses an FP8 plan.
#
# ``kernels/proj_gemm_norm_rope_mxfp8.py`` is the SAME fusion on the MXFP8 pipeline
# (``NormRopeFusionParams(quant_mxfp8=True)``): the rendered BLOCK-SCALE GEMM (e4m3
# codes + one E8M0 scale per 32-element K block on both operands, dequantized IN the
# MMA -- no alpha) whose epilogue [norms +] rotates Q/K and BLOCK-quantizes Q/K
# rowwise and V columnwise, writing the compact e4m3 ``q8 / k8 / v8`` + bf16
# ``gate16`` AND the three F8_128x4 E8M0 scale-factor blobs ``sf_q / sf_k / sf_v``
# exactly as ``kernels/quantize_mxfp8.py`` lays them down for
# ``sm107/prefill_d256_mxfp8.py``.  Runner ``run_fused_proj_gemm_mxfp8`` (PR-B
# section 3.1 ABI): it takes the block-scale GEMM's own SF blobs (``sf_a`` /
# ``sf_w``, PADDED F8_128x4 -- ``sf_padded_dims``) and ``batch`` / ``seq_len`` (the
# kernel decodes ``(b, s_tile)`` per 128-row tile), and DECLINES (typed
# ``NotImplementedError``) ``seq_len % 128 != 0 and batch > 1`` -- a GEMM tile would
# straddle two sequences; the unfused path serves that shape.
# ``NormRopeFusionParams.weight_fp4`` selects that fork's e2m1-B arm -- an MXFP4 ``W_qkvg``
# (packed ``[N, K // 2]`` codes, the UNCHANGED E8M0 / 32 blob) on the SAME runner ABI; the
# runner then also checks the padded sub-byte TMA format's K / base-address / row-stride rules.


@dataclass(frozen=True)
class NormRopeFusionParams:
    """Compile-time geometry + knobs of the fused stage-(1) kernel.

    Defaults are the Qwen3.5-397B geometry so a plain import of the template
    runs standalone.  ``offsets`` / ``n_qkvg`` follow ``api.py`` S1 exactly
    (Q | GATE | K | V, each ``heads * d_head`` wide) -- the fused epilogue
    classifies a tile by comparing its first column against these.

    ``norm_source``:
      ``"ldg"``   per-lane global loads of cos/sin (before the TMEM pass) and of
                  the norm weight (per subtile), issued in the epilogue arm.
                  Kept as the A/B reference for ``ldg_early``.
      ``"ldg_early"`` DEFAULT.  The cos/sin loads issued BEFORE the epilogue's wait on the
                  accumulator barrier, so they land while the MMA drains the
                  tile (non-norm tiles read the tables' row 0 -- one L1 line);
                  the weight stays per subtile.  Hoisting the weight too
                  (+32 live words across the sum-of-squares pass) SPILLED at
                  ``epi_reg_count=232`` -- 44 STL / 88 LDL -- do not retry.
      ``"const"`` the SAME arithmetic with every load deleted -- the diagnostic
                  floor (is the cost latency, or ALU/registers?).  NOT correct.
      ``"const_w"`` / ``"const_cs"`` FP8 fork only: ``const`` split by load class.
                  ``const_w`` = cos/sin loaded exactly as ``ldg_early``, the per-subtile
                  norm WEIGHT replaced by ``const``'s substitute; ``const_cs`` = the
                  weight loaded exactly as ``ldg_early``, cos/sin replaced by ``const``'s
                  substitutes.  Diagnostic floors (which load class costs?).  NOT correct.
      ``"off"``   the epilogue arm is not traced at all: the fork degenerates to
                  the rendered GEMM.  The A/B control.  bf16 only.

    ``quant_fp8`` (LAST field, default False so every bf16 key is unchanged)
    selects the FP8 fork ``proj_gemm_norm_rope_fp8.py``: e4m3 inputs, the
    descale ``alpha`` and the quant scales read in-kernel from a ``[4]`` fp32
    tensor, Q/K/V written as COMPACT e4m3 tensors (``q8 [T, h_q*d]``,
    ``k8 / v8 [T, h_kv*d]``) and the GATE as bf16 (``[T, n_gate]``).
    Inference only: ``want_rstd`` must be False, and
    ``norm_source="off"`` does not exist there (there is no single bf16 slab to
    degenerate to; the loads-deleted floor is ``"const"``).

    ``qk_norm`` (appended after ``quant_fp8``; default True so every existing key
    is spelled identically) selects the RoPE-ONLY epilogue when False: no pass A
    (sum of squares), no ``rsqrt``, no norm-weight loads, no ``rstd`` -- the Q/K
    tiles are rotated on the fp32 accumulator (FP8 fork: descaled + scaled, then
    quantized) and stored.  The kernel switch is ``_QK_NORM`` (``const_expr``) on
    both forks, and the weights are ``None`` at the ABI (the artifact's fakes are
    None too, so a norm-on artifact can never be handed None weights: the key
    differs).  ``want_rstd`` must be False (nothing to emit) and ``norm_source``
    ``"const_w"`` has no meaning (there are no weight loads to delete;
    ``"const_cs"`` == ``"const"`` there).

    ``quant_mxfp8`` (appended after ``qk_norm``) selects the MXFP8 GEMM fork twin
    ``proj_gemm_norm_rope_mxfp8.py``: the BLOCK-SCALE rendering (e4m3 codes + F8_128x4
    E8M0 scale factors on both operands, no alpha), norm optional + RoPE + ROWWISE
    (Q/K) / COLUMNWISE (V) block quantization with the F8_128x4 E8M0 scale factors
    written in the epilogue; outputs the compact e4m3 ``q8 / k8 / v8``, the bf16
    ``gate16`` and the three ``sf_q / sf_k / sf_v`` blobs.  Inference only
    (``want_rstd`` False), no ``"off"`` arm, exclusive with ``quant_fp8``; launched
    with :func:`run_fused_proj_gemm_mxfp8`.

    ``weight_fp4`` (appended after ``quant_mxfp8``; default False so every existing key is
    spelled identically) selects the e2m1-B arm of the MXFP8 fork: ``W_qkvg`` arrives as
    packed e2m1 codes (``torch.float4_e2m1fn_x2``, two per byte along K, storage
    ``[N, K // 2]``) with the UNCHANGED E8M0 / 32 scale-factor blob -- the block-scale
    catalog's MIXED row (e4m3 A x e2m1 B) rendered into the SAME template at
    ``mma_tile_k_bytes=32``, i.e. the padded E2M1 SMEM form: the fork's B-side constants
    change (the B dtypes, a ``Uint8`` B SMEM ring of the same footprint, the B TMA
    descriptor's padded sub-byte format, half the B expect-tx bytes), nothing in the kernel
    body.  Needs ``quant_mxfp8=True``; exclusive with ``quant_fp8``.  An NVFP4 ``W_qkvg``
    (e4m3 scales per 16) is not expressible here: the catalog has no mixed row with e4m3
    scales, so ``weight_fp4`` means MXFP4 exactly.  Same runner (:func:`run_fused_proj_gemm_mxfp8`).
    """

    d_head: int = 256
    rope_dim: int = 64
    h_q: int = 32
    h_kv: int = 2
    eps: float = 1e-6
    want_rstd: bool = False
    norm_source: str = "ldg_early"
    quant_fp8: bool = False  # the FP8 fork (e4m3 in, e4m3 Q/K/V + bf16 GATE out); appended, bf16 keys unchanged
    qk_norm: bool = True  # False: RoPE-only epilogue (no pass A / rsqrt / weight loads / rstd); appended, keys unchanged
    quant_mxfp8: bool = False  # the MXFP8 fork twin (block-scale rendering, PR-B section 3.1); appended, keys unchanged
    weight_fp4: bool = (
        False  # e2m1 W_qkvg (torch.float4_e2m1fn_x2) on the MXFP8 fork: the catalog's mixed row at K32, the padded E2M1 SMEM form; appended, keys unchanged
    )
    # The paged KV-cache WRITE-THROUGH arm of the bf16 fork (appended; 0 = every existing key spelled identically): the epilogue
    # also stores every K / V tile's post-norm, post-RoPE row into the pools ``[P, H_kv, kv_pool_page_size, D]`` at
    # ``slot_mapping[token]`` (a negative slot or one past the pools writes nothing) -- the gated attention block's serving
    # cache write folded into stage (1): one more st.global.v4 stream per row from the registers the slab store reads, a
    # per-row slot read before the accumulator wait; no new barrier, no new SMEM.  The page size is a compile key (the
    # in-kernel divide), the slot dtype too (``kv_slot_i64``: int64 slots as two 32-bit halves), exactly as in the separate
    # ``cache_write.py`` kernel the arm replaces bitwise.  bf16 fork only, ``norm_source`` in ("ldg", "ldg_early").
    kv_pool_page_size: int = 0
    kv_slot_i64: bool = False

    @property
    def offsets(self) -> tuple[int, int, int, int]:
        q = 0
        g = self.h_q * self.d_head
        k = 2 * self.h_q * self.d_head
        v = k + self.h_kv * self.d_head
        return (q, g, k, v)

    @property
    def n_qkvg(self) -> int:
        return (2 * self.h_q + 2 * self.h_kv) * self.d_head

    @property
    def n_gate(self) -> int:
        """Width of the bf16 GATE slab of the FP8 fork (= ``h_q * d_head``)."""
        return self.h_q * self.d_head

    @property
    def n_qkv(self) -> int:
        """``n_qkvg - n_gate`` = the Q + K + V columns.  Geometry bookkeeping only: since
        round 2 the FP8 fork writes q8 / k8 / v8 as separate compact tensors (no slab)."""
        return self.n_qkvg - self.n_gate

    @property
    def n_q(self) -> int:
        """Width of the compact e4m3 ``q8`` output of the FP8 fork (= ``h_q * d_head``)."""
        return self.h_q * self.d_head

    @property
    def n_kv(self) -> int:
        """Width of the compact e4m3 ``k8`` / ``v8`` outputs of the FP8 fork (= ``h_kv * d_head``)."""
        return self.h_kv * self.d_head


def validate_norm_rope_params(p: NormRopeFusionParams) -> None:
    """Raise ``ValueError`` on anything the fused epilogue cannot express.

    The tile-config-dependent facts (per-CTA N tile == d_head, subtile width
    vs rope_dim) are checked by the template itself against its rendered
    constants; this is the geometry-only half, usable without the DSL.
    """
    if p.d_head <= 0 or p.d_head & (p.d_head - 1):
        raise ValueError(f"d_head must be a power of two (the head index is a shift), got {p.d_head}")
    if p.h_q <= 0 or p.h_kv <= 0 or p.h_q % p.h_kv:
        raise ValueError(f"h_q={p.h_q} must be a positive multiple of h_kv={p.h_kv}")
    if not 0 < p.rope_dim < p.d_head or p.rope_dim % 2:
        raise ValueError(f"rope_dim must be even and in (0, d_head={p.d_head}), got {p.rope_dim}")
    if not p.eps > 0.0:
        raise ValueError(f"eps must be > 0, got {p.eps}")
    if p.norm_source not in ("ldg", "ldg_early", "const", "const_w", "const_cs", "off"):
        raise ValueError(f"norm_source must be 'ldg', 'ldg_early', 'const', 'const_w', 'const_cs' or 'off', got {p.norm_source!r}")
    if p.norm_source in ("const_w", "const_cs") and not (p.quant_fp8 or p.quant_mxfp8):
        raise ValueError(
            f"norm_source={p.norm_source!r} is a diagnostic arm of the quantizing forks only (quant_fp8=True or quant_mxfp8=True); the bf16 fork has 'const'"
        )
    if p.quant_fp8:
        if p.want_rstd:
            raise ValueError("quant_fp8: the FP8 fused projection is inference-only and emits no rstd; want_rstd must be False")
        if p.norm_source == "off":
            raise ValueError(
                "quant_fp8: norm_source='off' has no meaning on the FP8 fork (no single bf16 slab to degenerate to); use 'const' for the loads-deleted floor"
            )
    if not p.qk_norm:
        if p.want_rstd:
            raise ValueError("qk_norm=False emits no rstd; want_rstd must be False")
        if p.norm_source == "const_w":
            raise ValueError("qk_norm=False has no weight loads: const_w == ldg_early; use 'const_cs' (== 'const') for the floor")
    if p.quant_mxfp8:
        if p.quant_fp8:
            raise ValueError("quant_fp8 and quant_mxfp8 name two different forks; set at most one")
        if p.want_rstd:
            raise ValueError("quant_mxfp8: the MXFP8 fused projection is inference-only and emits no rstd; want_rstd must be False")
        if p.norm_source == "off":
            raise ValueError(
                "quant_mxfp8: norm_source='off' has no meaning on the MXFP8 fork (no single bf16 slab to degenerate to); use 'const' for the floor"
            )
    if p.kv_pool_page_size or p.kv_slot_i64:
        if p.kv_pool_page_size < 1:
            raise ValueError(f"kv_slot_i64 belongs to the write-through arm; set kv_pool_page_size > 0 (got {p.kv_pool_page_size})")
        if p.quant_fp8 or p.quant_mxfp8:
            raise NotImplementedError(
                "kv_pool_page_size: the paged KV-cache write-through arm is wired on the bf16 fork only (the pools are bf16 / f16, the activation dtype); the "
                "quantizing forks keep the separate cache-write launch"
            )
        if p.norm_source not in ("ldg", "ldg_early"):
            raise ValueError(f"kv_pool_page_size needs the production epilogue arm (norm_source 'ldg' / 'ldg_early'), got {p.norm_source!r}")
    if p.weight_fp4:
        if p.quant_fp8:
            raise ValueError(
                "weight_fp4 and quant_fp8 name two different forks: the e2m1 W_qkvg arm belongs to the MXFP8 fork (quant_mxfp8=True); "
                "the per-tensor FP8 fork takes e4m3 operands only"
            )
        if not p.quant_mxfp8:
            raise ValueError(
                "weight_fp4 (an e2m1 W_qkvg with the E8M0 / 32 scale-factor blob) is an arm of the MXFP8 fork only; set quant_mxfp8=True "
                "-- the bf16 fork has no fp4 operand"
            )


_FUSED_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope.py")
_FUSED_TEMPLATE_FP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope_fp8.py")
_FUSED_TEMPLATE_MXFP8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proj_gemm_norm_rope_mxfp8.py")


@dataclass
class FusedProjGemmPlan:
    """One compiled fused projection.  Plan-time only; ``run_fused_proj_gemm`` launches."""

    params: NormRopeFusionParams
    module: Any
    launch: Any  # the cute-compiled ``_host``
    fp8: bool = False  # True: the FP8 fork; launch with ``run_fused_proj_gemm_fp8`` (q8/k8/v8/gate16 + qscal)
    mxfp8: bool = False  # True: the MXFP8 fork twin; launch with ``run_fused_proj_gemm_mxfp8`` (q8/k8/v8/gate16 + sf_q/sf_k/sf_v)
    # W's STORAGE dtype the MXFP8 artifact was compiled for -- torch.float8_e4m3fn, or torch.float4_e2m1fn_x2 under
    # ``params.weight_fp4`` -- so the runner checks the PLAN it is handed, not a re-derivation; None on the other two
    # forks and on a hand-built plan (then derived from the params).
    w_dtype: Any = None

    @property
    def workspace_bytes(self) -> int:
        return 0  # no split-K, no scratch: the kernel writes the slab directly


def build_fused_proj_gemm(params: NormRopeFusionParams) -> FusedProjGemmPlan:
    """Load the fork specialized for ``params`` and cute-compile it.

    No shape here: M/N/K are runtime (symbolic in the artifact), so one plan
    serves every batch/sequence of a geometry.  N is checked at launch against
    ``params.n_qkvg`` -- the tile classification is only meaningful for the
    slab the params describe.
    """
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.template_loader import load_template

    validate_norm_rope_params(params)
    if params.quant_mxfp8:
        w_dtype = _fused_mx_w_dtype_of_params(params)  # a typed decline on a torch without the fp4 storage dtype, BEFORE the template loads
        mod = load_template(_FUSED_TEMPLATE_MXFP8, params, tag="proj_gemm_norm_rope_mxfp8")
        return FusedProjGemmPlan(params=params, module=mod, launch=mod.compile(), mxfp8=True, w_dtype=w_dtype)
    if params.quant_fp8:
        mod = load_template(_FUSED_TEMPLATE_FP8, params, tag="proj_gemm_norm_rope_fp8")
        return FusedProjGemmPlan(params=params, module=mod, launch=mod.compile(), fp8=True)
    mod = load_template(_FUSED_TEMPLATE, params, tag="proj_gemm_norm_rope")
    return FusedProjGemmPlan(params=params, module=mod, launch=mod.compile())


def _check_norm_weights(p: NormRopeFusionParams, w_q_norm, w_k_norm, dtype, dtype_name: str) -> None:
    """Norm weights present iff ``params.qk_norm`` -- BOTH directions, before any tensor is touched.

    The artifact was compiled with the weights as ``None`` under ``qk_norm=False``
    (and as ``[D]`` fakes otherwise), so a mismatch here would be a tvm-ffi ABI
    error at launch rather than a typed decline; check it on the host first."""
    if p.qk_norm:
        for name, t_ in (("w_q_norm", w_q_norm), ("w_k_norm", w_k_norm)):
            if t_ is None:
                raise ValueError(f"this artifact was compiled with qk_norm=True; {name} must be a [{p.d_head}] {dtype_name} tensor, got None")
            if tuple(t_.shape) != (p.d_head,) or t_.dtype != dtype:
                raise ValueError(f"{name} must be [{p.d_head}] {dtype_name}, got {tuple(t_.shape)} {t_.dtype}")
    elif w_q_norm is not None or w_k_norm is not None:
        raise ValueError("this artifact was compiled with qk_norm=False (RoPE only, no weight loads); w_q_norm / w_k_norm must both be None")


def _rope_table_2d(t: torch.Tensor, name: str, m: int, rope_dim: int, dtype: torch.dtype, dtype_name: str) -> torch.Tensor:
    """The kernel's ``[M, ROPE_DIM]`` view of a RoPE table -- a VIEW, never a copy.

    A ``[B, S, R]`` table is flattened with ``.view``: ``reshape`` would silently COPY a non-viewable table
    and the contiguity check below would then pass on the copy -- a hidden per-execute allocation (engine
    contract rule 10).  The result must be a contiguous ``[M, R]`` ``dtype`` table (the same contract the
    unfused stage applies with ``cos.view(t, rope_dim)``)."""
    if t.ndim == 3:
        try:
            t = t.view(-1, rope_dim)
        except RuntimeError as e:
            raise ValueError(
                f"{name} must be a contiguous [M={m}, {rope_dim}] {dtype_name} table; a [B, S, R] table is flattened as a VIEW (never copied) "
                f"and {tuple(t.shape)} with strides {tuple(t.stride())} is not viewable"
            ) from e
    if tuple(t.shape) != (m, rope_dim) or not t.is_contiguous() or t.dtype != dtype:
        raise ValueError(f"{name} must be a contiguous [M={m}, {rope_dim}] {dtype_name} table, got {tuple(t.shape)} {t.dtype}")
    return t


def run_fused_proj_gemm(
    plan: FusedProjGemmPlan,
    a: torch.Tensor,  # [M, K]   bf16
    w: torch.Tensor,  # [N, K]   bf16 (checkpoint layout, read transposed)
    out: torch.Tensor,  # [M, N]   bf16 -- Q/K columns land normed + rotated (qk_norm=False: rotated only)
    w_q_norm: Optional[torch.Tensor],  # [D]; None iff params.qk_norm is False
    w_k_norm: Optional[torch.Tensor],  # [D]
    cos: torch.Tensor,  # [M, ROPE_DIM] (or [B, S, ROPE_DIM] contiguous)
    sin: torch.Tensor,
    rstd_q: Optional[torch.Tensor] = None,  # [M, H_q]  fp32, required iff params.want_rstd
    rstd_k: Optional[torch.Tensor] = None,  # [M, H_kv] fp32
    *,
    stream,
    # APPENDED (the write-through arm, params.kv_pool_page_size > 0 -- all three REQUIRED there, REFUSED otherwise): the K / V pools
    # [P, H_kv, page_size, D] (D contiguous, the other three strides free: HND compact, NHD, a unified row's half) and the [M] int32 /
    # int64 slot map (its dtype = the artifact's kv_slot_i64).
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    """Launch.  No allocation, no conversion; every check is cheap host arithmetic.

    The problem tuple mirrors the shipped ``compiler._lower`` for the plain
    flavor: ``(m, n, k, batch, a strides (m, k, l), b strides (n, k, l), out
    strides (m, n, l))`` in ELEMENTS, taken from rank-3 ``[1, M, K]``-shaped views.
    """
    from cuda.bindings import driver as cuda

    if plan.fp8:
        raise ValueError("this plan is the FP8 fork (q8/k8/v8/gate16 outputs + qscal); launch it with run_fused_proj_gemm_fp8")
    if plan.mxfp8:
        raise ValueError("this plan is the MXFP8 fork (q8/k8/v8/gate16 + sf_q/sf_k/sf_v outputs); launch it with run_fused_proj_gemm_mxfp8")
    p = plan.params
    pools = (k_cache, v_cache, slot_mapping)
    if p.kv_pool_page_size:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block.cache_write_contract import check_pool, check_slot_mapping

        if any(x is None for x in pools):
            raise ValueError(
                "this artifact folds the paged KV-cache write-through into its epilogue: k_cache, v_cache and slot_mapping are required (Rule 1: no silent fallback)"
            )
        m_rows = int(_rank3(a, "a").shape[1])
        check_pool(k_cache, "k_cache", h=p.h_kv, page_size=p.kv_pool_page_size, d=p.d_head, dtype=a.dtype)
        check_pool(v_cache, "v_cache", h=p.h_kv, page_size=p.kv_pool_page_size, d=p.d_head, dtype=a.dtype)
        check_slot_mapping(slot_mapping, "slot_mapping", t=m_rows)
        want = torch.int64 if p.kv_slot_i64 else torch.int32
        if slot_mapping.dtype != want:
            raise ValueError(f"slot_mapping is {slot_mapping.dtype} but this artifact was compiled for {want}; the slot dtype is a compile key (kv_slot_i64)")
    elif any(x is not None for x in pools):
        raise ValueError("k_cache / v_cache / slot_mapping are consumed by the write-through arm only (params.kv_pool_page_size > 0); this artifact has none")
    a3, w3, o3 = _rank3(a, "a"), _rank3(w, "w"), _rank3(out, "out")
    m, k = int(a3.shape[1]), int(a3.shape[2])
    n = int(w3.shape[1])
    if n != p.n_qkvg:
        raise ValueError(f"the fused projection classifies tiles for N={p.n_qkvg} (Q|GATE|K|V at {p.offsets}); the weight has N={n}")
    if tuple(o3.shape) != (1, m, n) or int(w3.shape[2]) != k:
        raise ValueError(f"shape mismatch: a {tuple(a3.shape)}, w {tuple(w3.shape)}, out {tuple(o3.shape)}")
    cos2 = _rope_table_2d(cos, "cos", m, p.rope_dim, a.dtype, str(a.dtype))
    sin2 = _rope_table_2d(sin, "sin", m, p.rope_dim, a.dtype, str(a.dtype))
    _check_norm_weights(p, w_q_norm, w_k_norm, a.dtype, str(a.dtype).removeprefix("torch."))
    if p.want_rstd:
        if rstd_q is None or rstd_k is None:
            raise ValueError("this artifact was compiled with rstd outputs; both must be bound (Rule 1: no silent fallback)")
        if tuple(rstd_q.shape) != (m, p.h_q) or tuple(rstd_k.shape) != (m, p.h_kv) or rstd_q.dtype != torch.float32 or rstd_k.dtype != torch.float32:
            raise ValueError("rstd_q / rstd_k must be fp32 [M, h_q] / [M, h_kv]")
    else:
        rstd_q = rstd_k = None

    def _st(t3):
        st = t3.stride()
        return (st[1], st[2], st[0])

    problem = (m, n, k, 1, *_st(a3), *_st(w3), *_st(o3))
    # The artifact's fakes are (M|N, K, L) / (M, N, L): the same axis relabel the
    # shipped lowering applies (`compiler._lower`: `v.permute(1, 2, 0)`).
    plan.launch(
        problem,
        a3.permute(1, 2, 0),
        w3.permute(1, 2, 0),
        o3.permute(1, 2, 0),
        w_q_norm,
        w_k_norm,
        cos2,
        sin2,
        rstd_q,
        rstd_k,
        cuda.CUstream(int(stream)),
        *((k_cache, v_cache, slot_mapping) if p.kv_pool_page_size else ()),
    )


def run_fused_proj_gemm_fp8(
    plan: FusedProjGemmPlan,
    a: torch.Tensor,  # [M, K]        e4m3 (rank-2, or a rank-3 [1, M, K] view)
    w: torch.Tensor,  # [N_qkvg, K]   e4m3 (checkpoint layout, read transposed)
    out_q8: torch.Tensor,  # [M, h_q*d]   e4m3 contiguous (== compact BSHD [B, S, h_q, d])
    out_k8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_v8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_gate16: torch.Tensor,  # [M, h_q*d] bf16 contiguous
    w_q_norm: Optional[torch.Tensor],  # [D] bf16; None iff params.qk_norm is False
    w_k_norm: Optional[torch.Tensor],  # [D] bf16
    cos: torch.Tensor,  # [M, ROPE_DIM] bf16 (or [B, S, ROPE_DIM] contiguous)
    sin: torch.Tensor,
    qscal: torch.Tensor,  # [4] fp32 CUDA: [alpha, scale_q, scale_k, scale_v], read in-kernel
    *,
    stream,
) -> None:
    """Launch the FP8 fork.  No allocation, no conversion; every check is cheap host arithmetic.

    Problem tuple: ``(m, n, k, batch, a strides (m, k, l), b strides (n, k, l),
    q8 strides (m, n, l), k8 strides, v8 strides, gate16 strides)`` in ELEMENTS,
    from rank-3 ``[1, M, K]``-shaped views -- the bf16 runner's tuple plus three
    more output triples.  The cos/sin and norm-weight dtype checks compare
    against bf16 (the ACTIVATION dtype of the tables), not against ``a.dtype``
    (e4m3).  Every output is a TMA-store target: contiguous and 16-B aligned.
    """
    from cuda.bindings import driver as cuda

    if plan.mxfp8:
        raise ValueError("this plan is the MXFP8 fork (q8/k8/v8/gate16 + sf_q/sf_k/sf_v outputs); launch it with run_fused_proj_gemm_mxfp8")
    if not plan.fp8:
        raise ValueError("this plan is the bf16 fork (one bf16 slab); launch it with run_fused_proj_gemm")
    p = plan.params
    a3, w3 = _rank3(a, "a"), _rank3(w, "w")
    q3, k3, v3, g3 = _rank3(out_q8, "out_q8"), _rank3(out_k8, "out_k8"), _rank3(out_v8, "out_v8"), _rank3(out_gate16, "out_gate16")
    m, k = int(a3.shape[1]), int(a3.shape[2])
    n = int(w3.shape[1])
    if _FP8_E4M3 is None or a.dtype != _FP8_E4M3 or w.dtype != _FP8_E4M3:
        raise ValueError(f"the FP8 fused projection takes e4m3 operands, got a {a.dtype}, w {w.dtype}")
    if n != p.n_qkvg:
        raise ValueError(f"the fused projection classifies tiles for N={p.n_qkvg} (Q|GATE|K|V at {p.offsets}); the weight has N={n}")
    if int(w3.shape[2]) != k:
        raise ValueError(f"shape mismatch: a {tuple(a3.shape)}, w {tuple(w3.shape)}")
    for name, t3, t_, width, dt, dt_name in (
        ("out_q8", q3, out_q8, p.n_q, _FP8_E4M3, "e4m3"),
        ("out_k8", k3, out_k8, p.n_kv, _FP8_E4M3, "e4m3"),
        ("out_v8", v3, out_v8, p.n_kv, _FP8_E4M3, "e4m3"),
        ("out_gate16", g3, out_gate16, p.n_gate, torch.bfloat16, "bf16"),
    ):
        if tuple(t3.shape) != (1, m, width) or t3.dtype != dt or not t3.is_contiguous() or t3.data_ptr() % 16:
            raise ValueError(f"{name} must be a contiguous, 16-B-aligned {dt_name} [M={m}, {width}] tensor, got {tuple(t_.shape)} {t_.dtype}")
    cos2 = _rope_table_2d(cos, "cos", m, p.rope_dim, torch.bfloat16, "bf16")
    sin2 = _rope_table_2d(sin, "sin", m, p.rope_dim, torch.bfloat16, "bf16")
    _check_norm_weights(p, w_q_norm, w_k_norm, torch.bfloat16, "bf16")
    if tuple(qscal.shape) != (4,) or qscal.dtype != torch.float32 or not qscal.is_cuda or not qscal.is_contiguous() or qscal.data_ptr() % 16:
        raise ValueError(
            f"qscal must be a contiguous, 16-B-aligned fp32 CUDA tensor [4] = [alpha, scale_q, scale_k, scale_v], got {tuple(qscal.shape)} {qscal.dtype}"
        )

    def _st(t3):
        st = t3.stride()
        return (st[1], st[2], st[0])

    problem = (m, n, k, 1, *_st(a3), *_st(w3), *_st(q3), *_st(k3), *_st(v3), *_st(g3))
    # The artifact's fakes are (M|N, K, L) / (M, N, L): the same axis relabel the
    # shipped lowering applies (`compiler._lower`: `v.permute(1, 2, 0)`).
    plan.launch(
        problem,
        a3.permute(1, 2, 0),
        w3.permute(1, 2, 0),
        q3.permute(1, 2, 0),
        k3.permute(1, 2, 0),
        v3.permute(1, 2, 0),
        g3.permute(1, 2, 0),
        w_q_norm,
        w_k_norm,
        cos2,
        sin2,
        qscal,
        cuda.CUstream(int(stream)),
    )


def _check_fused_outputs(m: int, outs) -> None:
    """Every fused output is a TMA-store target: contiguous, 16-B aligned, its OWN width and dtype."""
    for name, t3, t_, width, dt, dt_name in outs:
        if tuple(t3.shape) != (1, m, width) or t3.dtype != dt or not t3.is_contiguous() or t3.data_ptr() % 16:
            raise ValueError(f"{name} must be a contiguous, 16-B-aligned {dt_name} [M={m}, {width}] tensor, got {tuple(t_.shape)} {t_.dtype}")


def _check_sf_out(name: str, sf: torch.Tensor, need: int, *, batch: int, heads: int, seq_len: int, d: int) -> torch.Tensor:
    """One F8_128x4 E8M0 scale-factor OUTPUT blob of the MXFP8 fork: uint8, contiguous, 16-B aligned, CUDA,
    ``B*H*ceil(S/128)*4*D`` bytes (the SDPA adapter's ``_reshape_sf`` count) -- returned flat."""
    if sf.dtype != torch.uint8:
        raise ValueError(f"{name} must be torch.uint8 (E8M0 bytes in F8_128x4 order), got {sf.dtype}")
    if not sf.is_cuda or not sf.is_contiguous() or sf.data_ptr() % 16:
        raise ValueError(f"{name} must be a contiguous, 16-B-aligned CUDA tensor")
    if sf.numel() != need:
        raise ValueError(f"{name} must hold B*H*ceil(S/128)*{4 * d} = {need} bytes for batch={batch} H={heads} S={seq_len} D={d}, got {sf.numel()}")
    return sf.view(-1)


def run_fused_proj_gemm_mxfp8(
    plan: FusedProjGemmPlan,
    a8: torch.Tensor,  # [M, K]        e4m3 codes (rank-2, or a rank-3 [1, M, K] view), M == batch * seq_len
    sf_a: torch.Tensor,  # F8_128x4 E8M0 blob of a8 over its M rows: sf_blob_bytes(M, K) bytes, uint8 / float8_e8m0fnu
    w8: torch.Tensor,  # [N_qkvg, K]   e4m3 codes, or packed e2m1 codes [N_qkvg, K // 2] under params.weight_fp4 (checkpoint layout, read transposed)
    sf_w: torch.Tensor,  # F8_128x4 E8M0 blob of w8 over its N rows: sf_blob_bytes(N, K) bytes
    out_q8: torch.Tensor,  # [M, h_q*d]   e4m3 contiguous (== compact BSHD [B, S, h_q, d])
    out_k8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_v8: torch.Tensor,  # [M, h_kv*d]  e4m3 contiguous
    out_gate16: torch.Tensor,  # [M, h_q*d] bf16 contiguous
    out_sf_q: torch.Tensor,  # uint8, B*h_q*ceil(S/128)*4*d bytes: the SDPA's rowwise Q scale factors
    out_sf_k: torch.Tensor,  # uint8, B*h_kv*ceil(S/128)*4*d bytes: rowwise K
    out_sf_v: torch.Tensor,  # uint8, B*h_kv*ceil(S/128)*4*d bytes: COLUMNWISE V (D-plane-major)
    w_q_norm: Optional[torch.Tensor],  # [D] bf16; None iff params.qk_norm is False
    w_k_norm: Optional[torch.Tensor],  # [D] bf16
    cos: torch.Tensor,  # [M, ROPE_DIM] bf16 (or [B, S, ROPE_DIM] contiguous)
    sin: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    stream,
) -> None:
    """Launch the MXFP8 fork.  No allocation, no conversion; every check is cheap host arithmetic.

    Problem tuple: the block-scale rendering's ``(m, n, k, batch, a strides (m, k, l), b strides
    (n, k, l), q8 strides (m, n, l))`` plus the k8 / v8 / gate16 stride triples, in ELEMENTS,
    from rank-3 ``[1, M, K]``-shaped views.  ``sf_a`` / ``sf_w`` are the block-scale GEMM's OWN
    operand scale factors (what ``run_proj_gemm(..., sf_a=, sf_w=)`` binds): PADDED F8_128x4
    blobs (``sf_blob_bytes``), bound as the ``(1, rows_pad, 4*k4)`` e8m0 view -- the kernel
    reads a base pointer and rebuilds the layout from the problem size.  The three SF OUTPUTS
    are the SDPA's: ``B*H*ceil(S/128)*4*D`` bytes each, written in full (pad rows -> 0x00).

    **Typed decline**: ``seq_len % 128 != 0 and batch > 1`` raises ``NotImplementedError`` -- a
    128-row GEMM tile would straddle two sequences and the kernel's once-per-tile
    ``(b, s_tile)`` decode would be wrong for part of it.  The unfused block-scale path
    (``build_proj_gemm(block_scale=True)`` + ``quantize_mxfp8``) serves that shape.

    **The e2m1 ``W_qkvg`` arm** (``plan.params.weight_fp4``; ``plan.w_dtype`` is ``torch.float4_e2m1fn_x2``): ``w8`` is
    the PACKED ``[N, K // 2]`` storage -- two codes per byte along K; a uint8 blob is declined with the
    ``.view(torch.float4_e2m1fn_x2)`` hint, a LOGICAL ``[N, K]`` fp4 tensor by name -- and ``sf_w`` the UNCHANGED
    E8M0 / 32 blob over the LOGICAL ``K`` (read off ``a8``).  Typed ``ValueError``s before any launch: an e4m3
    ``w8`` on an e2m1 plan and the reverse; ``K % 128 != 0``; a weight base address that is not 32-byte aligned;
    a row stride that is not a multiple of 32 bytes.  The last three are the padded sub-byte TMA format's own
    rules, which the descriptor builder checks on STATIC operands only -- the fork's weight descriptor carries a
    symbolic K, a dynamic base and symbolic strides, so unchecked they would surface as a malformed tensor map
    at launch (an illegal-instruction error), never as a decline.
    """
    from cuda.bindings import driver as cuda

    from .quantize_mxfp8 import n_sf_tiles, sf_bytes

    if not plan.mxfp8:
        raise ValueError("this plan is not the MXFP8 fork; launch the bf16 fork with run_fused_proj_gemm and the FP8 fork with run_fused_proj_gemm_fp8")
    p = plan.params
    if batch <= 0 or seq_len <= 0:
        raise ValueError(f"batch and seq_len must be positive, got batch={batch} seq_len={seq_len}")
    if seq_len % 128 and batch > 1:
        raise NotImplementedError(
            f"the fused MXFP8 projection decodes (b, s_tile) once per 128-row GEMM tile, so at batch={batch} > 1 the sequence length must be a "
            f"multiple of 128 (got {seq_len}: a tile would straddle two sequences); use the unfused block-scale path for this shape"
        )
    a3, w3 = _rank3(a8, "a8"), _rank3(w8, "w8")
    q3, k3, v3, g3 = _rank3(out_q8, "out_q8"), _rank3(out_k8, "out_k8"), _rank3(out_v8, "out_v8"), _rank3(out_gate16, "out_gate16")
    m, k = int(a3.shape[1]), int(a3.shape[2])
    n = int(w3.shape[1])
    w_dtype = _fused_mx_w_dtype(plan)  # e4m3, or the packed e2m1 storage dtype under params.weight_fp4
    fp4_w = _is_fp4(w_dtype)
    if _FP8_E4M3 is None or a8.dtype != _FP8_E4M3:
        raise ValueError(f"the MXFP8 fused projection takes e4m3 codes for the activation, got a8 {a8.dtype}")
    if w8.dtype != w_dtype:
        hint = " -- e2m1 codes travel as torch.float4_e2m1fn_x2: .view(torch.float4_e2m1fn_x2) the uint8 storage" if fp4_w and w8.dtype == torch.uint8 else ""
        raise ValueError(
            f"this MXFP8 fused projection was compiled for {_dtype_word(w_dtype)} W_qkvg codes (NormRopeFusionParams.weight_fp4={p.weight_fp4}), "
            f"got w8 {w8.dtype}; refusing to reinterpret the bytes{hint}"
        )
    if m != batch * seq_len:
        raise ValueError(f"M must equal batch*seq_len: a8 has M={m}, batch*seq_len={batch * seq_len}")
    if n != p.n_qkvg:
        raise ValueError(f"the fused projection classifies tiles for N={p.n_qkvg} (Q|GATE|K|V at {p.offsets}); the weight has N={n}")
    k_store = storage_k(k, w_dtype)  # the STORAGE extent along K: K/2 bytes for packed e2m1 codes, K for e4m3
    if int(w3.shape[2]) != k_store:
        if fp4_w and int(w3.shape[2]) == k:
            raise ValueError(
                f"w8 is fp4 storage {tuple(w8.shape)} -- two e2m1 codes per byte along K -- so its last extent must be K/2 = {k_store} for the "
                f"K={k} read off a8, not {k} (a LOGICAL [N, K] fp4 tensor holds twice the data)"
            )
        raise ValueError(f"shape mismatch: a8 {tuple(a3.shape)} (K={k}), w8 {tuple(w3.shape)} (a {_dtype_word(w_dtype)} weight stores K as {k_store} elements)")
    if int(a3.stride(2)) != 1 or int(w3.stride(2)) != 1:
        # Both operands are K-major: the kernel's tensor maps carry the ROW and BATCH strides only (the K stride is one element by
        # construction), so a K-strided view (a transposed [K, N] checkpoint viewed .t(), an every-other-column slice) would be read
        # as if contiguous -- silent wrong output on either arm.  Refuse it here, typed, like the shape checks above.
        raise ValueError(
            "the fused projection reads a8 and w8 K-major with a UNIT stride along K (its tensor maps carry the row and batch strides only, "
            f"so a K-strided view would be read as if it were contiguous): got a8 strides {tuple(int(s) for s in a3.stride())}, "
            f"w8 strides {tuple(int(s) for s in w3.stride())}"
        )
    if k % 32:
        raise ValueError(f"the block-scale GEMM needs K % 32 == 0 (one E8M0 scale per 32-element block), got K={k}")
    if fp4_w:
        # The e2m1 B is read through the PADDED sub-byte TMA format (64-byte boxes expanded to 128-byte SMEM rows).  Its three
        # tensor-map rules are checked by the descriptor builder on STATIC operands only; the fork's weight descriptor carries a
        # symbolic K, a dynamic base address and symbolic strides, so a violation would surface as a malformed tensor map at launch
        # (an illegal-instruction error), not as a typed decline.  Check them here, in STORAGE bytes (one byte = one
        # float4_e2m1fn_x2 element).  A contiguous [N, K // 2] weight with K % 128 == 0 satisfies the stride rule by construction;
        # only a padded-row view, or a 16-byte-but-not-32-byte-aligned view of a larger buffer, can trip the last two.
        if k % 128:
            raise ValueError(
                f"the e2m1 W_qkvg arm needs K % 128 == 0 (the padded sub-byte TMA format's Tensor-Size[0] rule: whole 64-byte = 128-code rows), got K={k}"
            )
        if w8.data_ptr() % 32:
            raise ValueError(
                "the e2m1 W_qkvg arm needs a 32-byte-aligned weight base address (the padded sub-byte TMA format's rule; 16-byte alignment is not "
                f"enough), got data_ptr() % 32 = {w8.data_ptr() % 32}"
            )
        row_stride_bytes = int(w3.stride(1))  # elements of float4_e2m1fn_x2 == storage bytes
        if row_stride_bytes % 32:
            raise ValueError(
                "the e2m1 W_qkvg arm needs every weight row stride to be a multiple of 32 bytes (the padded sub-byte TMA format's rule), got "
                f"{row_stride_bytes} bytes between rows (a padded-row view of a wider buffer)"
            )
    _check_fused_outputs(
        m,
        (
            ("out_q8", q3, out_q8, p.n_q, _FP8_E4M3, "e4m3"),
            ("out_k8", k3, out_k8, p.n_kv, _FP8_E4M3, "e4m3"),
            ("out_v8", v3, out_v8, p.n_kv, _FP8_E4M3, "e4m3"),
            ("out_gate16", g3, out_gate16, p.n_gate, torch.bfloat16, "bf16"),
        ),
    )
    cos2 = _rope_table_2d(cos, "cos", m, p.rope_dim, torch.bfloat16, "bf16")
    sin2 = _rope_table_2d(sin, "sin", m, p.rope_dim, torch.bfloat16, "bf16")
    _check_norm_weights(p, w_q_norm, w_k_norm, torch.bfloat16, "bf16")
    # The GEMM's own operand scale factors: the same padded F8_128x4 blobs / e8m0 views the unfused driver binds.
    sfa3 = _sf_view_of(sf_a, "sf_a", m, k, "fused_mxfp8")
    sfw3 = _sf_view_of(sf_w, "sf_w", n, k, "fused_mxfp8")
    n_tiles = n_sf_tiles(seq_len)
    sfq = _check_sf_out("out_sf_q", out_sf_q, sf_bytes(batch, p.h_q, seq_len, p.d_head), batch=batch, heads=p.h_q, seq_len=seq_len, d=p.d_head)
    sfk = _check_sf_out("out_sf_k", out_sf_k, sf_bytes(batch, p.h_kv, seq_len, p.d_head), batch=batch, heads=p.h_kv, seq_len=seq_len, d=p.d_head)
    sfv = _check_sf_out("out_sf_v", out_sf_v, sf_bytes(batch, p.h_kv, seq_len, p.d_head), batch=batch, heads=p.h_kv, seq_len=seq_len, d=p.d_head)

    def _st(t3):
        st = t3.stride()
        return (st[1], st[2], st[0])

    problem = (m, n, k, 1, *_st(a3), *_st(w3), *_st(q3), *_st(k3), *_st(v3), *_st(g3))
    # The artifact's fakes are (M|N, K, L) / (M, N, L): the same axis relabel the
    # shipped lowering applies (`compiler._lower`: `v.permute(1, 2, 0)`).  The SF
    # operands reach the kernel as base pointers only (fully symbolic fakes).
    plan.launch(
        problem,
        a3.permute(1, 2, 0),
        w3.permute(1, 2, 0),
        sfa3,
        sfw3,
        q3.permute(1, 2, 0),
        k3.permute(1, 2, 0),
        v3.permute(1, 2, 0),
        g3.permute(1, 2, 0),
        w_q_norm,
        w_k_norm,
        cos2,
        sin2,
        sfq,
        sfk,
        sfv,
        _cutlass_int32(seq_len),
        _cutlass_int32(batch * p.h_kv * n_tiles),
        cuda.CUstream(int(stream)),
    )


def _cutlass_int32(v: int):
    """A ``cutlass.Int32`` runtime scalar for the fused-kernel ABI (imported lazily: ``cutlass`` is a DSL import)."""
    import cutlass

    return cutlass.Int32(int(v))


def _why_no_frost_plan() -> str:
    """The two reasons the plan is absent, checked in the order that costs least.

    Worth spelling out because the two look identical from the call site and one
    of them is a real capability limit, not a setup mistake.
    """
    import os

    reasons = []
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        arch = cc[0] * 10 + cc[1]
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.kernel_registry import PIPELINE_ARCH_RANGES

        spans = PIPELINE_ARCH_RANGES.get("sm100", ())
        if spans and not any(lo <= arch < hi for lo, hi in spans):
            reasons.append(
                f"the sm100 GEMM template family runs on "
                + " or ".join(f"sm_{lo}..sm_{hi - 1}" for lo, hi in spans)
                + f", and this device is sm_{arch} -- so FROST cannot serve this graph here at all"
            )
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").strip().lower() not in ("1", "true", "yes", "on"):
        reasons.append(
            "CUDNN_FRONTEND_ENABLE_FROST_ENGINES is not set -- frost_gemm is an opt-in engine; the flag is read live when the ranked list is "
            "built, so it must be in the environment before compile() (a dense bf16 / f16 plan whose N the forced 256-wide tile divides needs "
            "no flag: it takes the forced-tile JIT-only route)"
        )
    if not reasons:
        reasons.append("the engine declined this shape/dtype; run with CUDNN_FRONTEND_LOG_INFO=1 CUDNN_FRONTEND_LOG_FILE=stderr for its reason")
    return (
        " AND ".join(reasons)
        + ". Refusing to fall back: an unpinned graph runs a cuDNN backend plan, and any measurement off it would be of that kernel, not FROST's."
    )


_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)  # None on a torch without fp8 storage types
_FP4_X2 = getattr(torch, "float4_e2m1fn_x2", None)  # e2m1 codes, TWO per byte along the contiguous (K) axis
_E8M0 = getattr(torch, "float8_e8m0fnu", None)


def _is_fp8(dtype: torch.dtype) -> bool:
    return _FP8_E4M3 is not None and dtype == _FP8_E4M3


# The (dtype, a_major, b_major) triples whose fp8 MN-major rendering this driver ADMITS -- a table, never a
# blanket lift.  Each row was validated on cc 10.7 at the block's forced tile in BOTH MMA K forms (32 and 64
# bytes) against an fp64 reference of the dequantized e4m3 products under test_proj_gemm_bwd.py's bf16-output
# bound (rtol 2^-7, atol rtol*max|ref|), with a two-launch bitwise check and a sentinel-filled output:
#
#   (e4m3, "m", "n")   the wgrads  dW = dY^T @ X   (B1 dW_o, B7 dW_qkvg)   -- M-major A, N-major B
#   (e4m3, "k", "n")   the dgrads  dX = dY @ W     (B2 dO_gated, B8 dh)    -- N-major B
#
# `test_fp8_mn_major_matches_fp64_on_cc107` is the pin; `test_fp8_mn_major_outside_the_table_is_a_typed_decline`
# inverts it.  An fp8 major outside the table (an M-major A against a K-major B, nobody's GEMM) and a MIXED fp8
# dtype pair stay typed NotImplementedErrors.  Both e4m3 at every triple: the driver's only fp8 dtype (_is_fp8).
# The rows hold at the tile they were validated at and nowhere else: `build_proj_gemm` admits an fp8 MN-major triple only
# when the plan takes the forced tile (`_FORCED_TILE_NAME`) on the FROST JIT -- an N that 256 does not divide (the
# heuristic's tile), another explicit `tile_config`, or `pin_frost=False` (no JIT) is the same typed decline, naming the tile;
# so is a forced compile the compiler refuses (never the graph-heuristic fallback the bf16 plans take).
# "Another explicit tile_config" includes the forced tile's own K64 twin named as `tile_config`: the 64-byte form is reached
# through `mma_tile_k_bytes=64` ONLY (the one spelling the device cells validated), never by its config name.
FP8_MN_MAJOR_VALIDATED: frozenset = frozenset({(_FP8_E4M3, "m", "n"), (_FP8_E4M3, "k", "n")}) if _FP8_E4M3 is not None else frozenset()


def _fp8_mn_major_menu() -> str:
    """The admitted fp8 MN-major triples (``FP8_MN_MAJOR_VALIDATED``) spelled for the decline's message --
    ``(e4m3, a_major='m', b_major='n'), ...`` ordered by majors -- or ``none`` on a torch without e4m3."""
    return (
        ", ".join(f"({_dtype_word(dt)}, a_major={am!r}, b_major={bm!r})" for dt, am, bm in sorted(FP8_MN_MAJOR_VALIDATED, key=lambda t: (t[1], t[2]))) or "none"
    )


def _is_fp4(dtype: torch.dtype) -> bool:
    """``torch.float4_e2m1fn_x2`` -- the ONLY fp4 storage dtype the driver accepts.  A uint8 blob
    holding packed e2m1 codes is declined with the ``.view(torch.float4_e2m1fn_x2)`` hint (torch
    2.13 can VIEW but not cast to fp4), so the dtype carries the packing through every check."""
    return _FP4_X2 is not None and dtype == _FP4_X2


# Two e2m1 codes per byte: the packing factor of ``torch.float4_e2m1fn_x2`` storage along K (low nibble = even k).  ONE source for
# every "K/2" of this module (``_check_operand``, ``storage_k`` -> the block-scale drivers' K-major checks) and of ``quantize_fp4``.
FP4_CODES_PER_BYTE = 2


def storage_k(k: int, dtype) -> int:
    """The STORAGE extent along K of a K-major operand of ``dtype`` whose LOGICAL K is ``k``: ``k // FP4_CODES_PER_BYTE`` for e2m1
    codes (``torch.float4_e2m1fn_x2``, two per byte), ``k`` for every 1- or 2-byte dtype -- and for ``None`` (a hand-built plan
    without dtypes declares the logical extent)."""
    return k // FP4_CODES_PER_BYTE if _is_fp4(dtype) else k


def _fused_mx_w_dtype_of_params(p: NormRopeFusionParams):
    """W's STORAGE dtype of the MXFP8 fork specialized for ``p``: ``torch.float4_e2m1fn_x2`` under ``weight_fp4`` (a typed
    ``NotImplementedError`` on a torch without that dtype -- the arm's operand cannot be bound), else ``torch.float8_e4m3fn``."""
    if p.weight_fp4:
        if _FP4_X2 is None:
            raise NotImplementedError("NormRopeFusionParams.weight_fp4 needs torch.float4_e2m1fn_x2 (the packed e2m1 storage dtype) and this torch has none")
        return _FP4_X2
    return _FP8_E4M3


def _fused_mx_w_dtype(plan: FusedProjGemmPlan):
    """The W dtype recorded on an MXFP8 fused plan; a hand-built plan (``w_dtype=None``) derives it from its params."""
    return plan.w_dtype if plan.w_dtype is not None else _fused_mx_w_dtype_of_params(plan.params)


def _w_dtype_of(plan: "ProjGemmPlan"):
    """W's dtype as declared at build time: ``w_dtype`` when built with a per-weight dtype, else ``dtype`` (``None`` on
    a hand-built plan without dtypes, which checks nothing)."""
    return plan.w_dtype if plan.w_dtype is not None else plan.dtype


def _cudnn_dtype(dtype: torch.dtype):
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    table = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF}
    if _FP8_E4M3 is not None:
        table[_FP8_E4M3] = cudnn.data_type.FP8_E4M3
    if _FP4_X2 is not None:
        table[_FP4_X2] = cudnn.data_type.FP4_E2M1
    try:
        return table[dtype]
    except KeyError:
        raise ValueError(f"proj_gemm serves bf16/f16/fp8-e4m3/fp4-e2m1 (float4_e2m1fn_x2) only, got {dtype}") from None


def _dtype_word(dtype: torch.dtype) -> str:
    return "e2m1" if _is_fp4(dtype) else "e4m3" if _is_fp8(dtype) else str(dtype).replace("torch.", "")


# The block-scale operand PAIRS this driver claims -- one row each, exactly as the FROST GEMM
# catalog spells them (``gemm/frost/kernel_registry.py`` ``_BLOCK_SCALE_CASES``; every row shares
# reorder F8_128x4, fp32 dequant, fp32 accumulate):
#
#   A e4m3  x  W e4m3   with E8M0 scales per 32   -- MXFP8 x MXFP8, today's stage (1)
#   A e4m3  x  W e2m1   with E8M0 scales per 32   -- the MIXED row: MXFP8 activations x MXFP4 weight
#   A e2m1  x  W e4m3   with E8M0 scales per 32   -- the mixed row the other way round
#   A e2m1  x  W e2m1   with E4M3 scales per 16   -- NVFP4 x NVFP4
#   A e2m1  x  W e2m1   with E8M0 scales per 32   -- MXFP4 x MXFP4
#
# Nothing else the catalog lists (e5m2 sides, E5M3 scales, e4m3 scales at block 32 -- SM 10.7-only)
# is claimed: the block never quantizes to it, and an unlisted pair is a typed error, never a
# fallback.  The pairing is a RULE and not a default because the DECLARED scale dtype selects the
# MMA's scale format (``gemm/frost/fusion_ir.py`` ``sf_scale_format``): an e4m3 blob declared E8M0
# does not fail, it miscomputes.
SF_BLOCK_SIZES = (16, 32)


def _sf_torch_dtype(sf_dtype) -> torch.dtype:
    """The torch STORAGE dtype of a scale-factor blob declared as ``sf_dtype`` (a ``cudnn.data_type``;
    ``None`` = the E8M0 default) -- the dtype ``_sf_view_of`` re-views a uint8 blob as."""
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    if sf_dtype is None or sf_dtype == cudnn.data_type.FP8_E8M0:
        return _E8M0
    if sf_dtype == cudnn.data_type.FP8_E4M3:
        return _FP8_E4M3
    raise ValueError(f"block-scale scale factors are FP8_E8M0 or FP8_E4M3 bytes (cudnn.data_type), got {sf_dtype}")


def check_sf_torch_dtype(sf_dtype) -> torch.dtype:
    """:func:`_sf_torch_dtype` plus the torch-version check: a typed ``ValueError`` at PLAN construction
    (``block_scale_pairing`` calls it) rather than at the first ``_sf_view`` of a launch, so a plan whose
    scale dtype this torch cannot even name never exists."""
    dt = _sf_torch_dtype(sf_dtype)
    if dt is None:
        raise ValueError(f"this torch ({torch.__version__}) has no storage dtype for {sf_dtype} scale factors (needs torch.float8_e8m0fnu / float8_e4m3fn)")
    return dt


def block_scale_pairing(*, dtype: torch.dtype, w_dtype: torch.dtype, sf_dtype, block_size: int, label: str):
    """Resolve and check the ``(sf_dtype, block_size)`` of a block-scale plan over ``A: dtype`` x ``W: w_dtype``
    against the claimed pairs above; returns the resolved pair.

    ``sf_dtype=None`` resolves from the operands and the block: E4M3 for two e2m1 sides at block 16,
    E8M0 otherwise -- every claimed pair has exactly one scale dtype per block size, so the default is
    derived, never guessed.  ``block_size`` is never defaulted here (the caller's 32 is the MX contract).
    Raises ``ValueError`` naming the pair; ``_Projection.check_support`` re-raises it as the block's
    typed ``NotImplementedError``, ``build_proj_gemm`` lets it propagate.
    """
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    e8m0, e4m3 = cudnn.data_type.FP8_E8M0, cudnn.data_type.FP8_E4M3
    for side, dt in (("dtype (A)", dtype), ("w_dtype (W)", w_dtype)):
        if not (_is_fp8(dt) or _is_fp4(dt)):
            raise ValueError(
                f"{label}: block_scale=True needs e4m3 or e2m1 codes on each operand (torch.float8_e4m3fn | torch.float4_e2m1fn_x2); {side} is {dt}"
            )
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size not in SF_BLOCK_SIZES:
        # An int, not merely ``== 16``: ``16.0`` would flow float dims into ``sf_padded_dims`` and the graph.
        raise ValueError(f"{label}: block_size must be one of {SF_BLOCK_SIZES} (an int; one scale per block along K), got {block_size!r}")
    pair = f"A {_dtype_word(dtype)} x W {_dtype_word(w_dtype)}"
    if _is_fp4(dtype) and _is_fp4(w_dtype):
        served = {(e4m3, 16): "NVFP4 x NVFP4", (e8m0, 32): "MXFP4 x MXFP4"}
        if sf_dtype is None:
            sf_dtype = e4m3 if block_size == 16 else e8m0
    else:
        served = {(e8m0, 32): "MXFP8 x MXFP8" if _is_fp8(dtype) and _is_fp8(w_dtype) else "the mixed MXFP8 x MXFP4 row"}
        if sf_dtype is None:
            sf_dtype = e8m0
    if (sf_dtype, block_size) not in served:
        menu = "; ".join(f"{name}: {sf} per {blk}" for (sf, blk), name in served.items())
        raise ValueError(
            f"{label}: no block-scale GEMM row for {pair} with {sf_dtype} scales per {block_size} elements "
            f"(kernel_registry._BLOCK_SCALE_CASES); for this operand pair the driver serves {menu}"
        )
    check_sf_torch_dtype(sf_dtype)  # the blob's storage dtype exists on this torch -- decline here, not at the first launch
    return sf_dtype, block_size


_LOG = logging.getLogger(__name__)


# cuDNN handles for the GRAPH route of :func:`run_proj_gemm`, one per
# (device, stream).  The graph API binds a plan's launch stream to a HANDLE
# (``cudnn.set_stream``), not to an execute argument, so threading a stream
# through ``graph.execute`` means owning a handle bound to it.  Created on first
# use and kept for the process -- ``cudnnCreate`` is not free and Rule 1 bans
# per-execute resource creation (same shape as ``sdpa/fwd/torch_op.py``'s
# per-device cache).  Keyed by stream too, and ``set_stream`` is never re-issued
# on a cached handle, so two streams driving a block concurrently never share
# one (the ``cudnn.set_stream`` docstring's "own handle per stream" caveat).
# Bounded by the number of distinct launch streams a caller uses.
_GRAPH_HANDLES: dict = {}


def handle_for_stream(device, stream) -> Any:
    """The cached ``cudnn.Handle`` bound to ``stream`` on ``device`` (created on first use).

    ``stream`` is a raw ``CUstream`` int (``torch.cuda.current_stream(dev).cuda_stream``;
    0 = the legacy default stream).  Only the graph route of :func:`run_proj_gemm`
    needs this -- the JIT route takes ``stream=`` directly.
    """
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (idx, int(stream))
    h = _GRAPH_HANDLES.get(key)
    if h is None:
        # The FIRST execute of a graph-route plan on a stream creates its handle HERE (cudnnCreate: host and device
        # resources) -- illegal inside a CUDA-graph capture, and a serving stack captures the first execute on a fresh
        # stream with no eager warm-up.  Refused typed, naming the route and the remedy, before the driver sees it.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.capture import check_not_capturing

        check_not_capturing(
            int(stream),
            f"frost gemm graph route (cuDNN handle creation for device {idx}, stream {int(stream):#x})",
            remedy=(
                "run one eager execute of the block on the capture stream before capturing (it creates the handle this route binds), "
                "or declare the block's GEMMs capture-safe (build_proj_gemm(capture_safe=True)) so they take the forced-tile JIT route, "
                "which needs no handle"
            ),
        )
        # create_handle() binds to the CURRENT device -- pin it so a tensor on
        # cuda:1 never gets a handle created against cuda:0.
        with torch.cuda.device(idx):
            h = cudnn.create_handle()
        cudnn.set_stream(handle=h, stream=int(stream))
        _GRAPH_HANDLES[key] = h
    return h


def _refuse_compile_under_capture(label: str) -> None:
    """``build_proj_gemm`` under a CUDA-graph capture is refused typed: it builds a cuDNN graph (a cuDNN handle in the
    handle-less ``cudnn.pygraph`` constructor) and JIT-compiles a kernel -- neither can happen inside a capture, and a serving
    stack that compiles lazily at its first (captured) execute would otherwise see the driver's error from the middle of it.
    The stream asked is torch's current one on the current device: the stream a ``torch.cuda.graph`` capture makes current."""
    if not torch.cuda.is_available():
        return  # nothing can be capturing without a device; the host-only probes build plans here too
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.capture import check_not_capturing

    check_not_capturing(
        torch.cuda.current_stream().cuda_stream,
        f"{label}: FROST GEMM compile (build_proj_gemm builds the cuDNN graph and JIT-compiles the kernel)",
        remedy="compile() at warm-up, before the capture begins; a capture holds launches only",
    )


@dataclass
class ProjGemmPlan:
    """One compiled projection. Built at plan time, called per execute."""

    graph: Any
    a: Any
    b: Any
    c: Any
    m: int
    k: int
    n: int
    label: str
    jit: Any = None  # set only when a tile config was forced; see _forced_tile_config
    jit_binding: Any = None
    # FP8 / epilogue additions (append-only). `alpha` is the graph's scalar
    # multiplier tensor when the plan was built with `alpha=True`, else None.
    dtype: Any = None  # input torch dtype
    out_dtype: Any = None  # output torch dtype (bf16 by default for fp8 inputs)
    alpha: Any = None
    tile_config_name: Optional[str] = None  # the JIT config that actually runs, or "heuristic (graph engine)"
    mma_tile_k_bytes: Optional[int] = None  # 32 = the K=32 MMA form (half the Rubin 2xFP8 rate); 64 = K=64
    # Block-scale (MXFP8) additions (append-only). `sfa` / `sfb` are the graph's
    # F8_128x4 scale-factor tensors when the plan was built with `block_scale=True`,
    # else None; `route` records which path serves the launch:
    #   "graph+jit"  the backend admitted the graph, `frost_gemm` is pinned in its
    #                ranked list (workspace / plan-name honesty) AND the JIT at the
    #                resolved tile config is what `run_proj_gemm` launches;
    #   "jit-only"   the backend declined the graph (a typed cudnnGraphNotSupportedError,
    #                or no `frost_gemm` in its ranked list -- `_NoFrostPlan`); the JIT
    #                never touches the backend, so the kernel still runs.  Taken by the
    #                block-scale plans, and by a DENSE bf16 / f16 plan with a forced (or
    #                explicitly named) tile -- the route a serving stack compiles under with
    #                CUDNN_FRONTEND_ENABLE_FROST_ENGINES unset (no backend plan, no cuDNN
    #                handle at execute).  Any OTHER backend error propagates (it is a bug:
    #                e.g. the unpadded F8_128x4 declaration this driver used to make, which
    #                read as "declined" until the catch was narrowed);
    #   "graph"      dense, no forced tile: the graph engine's own plan runs.
    block_scale: bool = False
    sfa: Any = None
    sfb: Any = None
    sf_dtype: Any = None  # cudnn.data_type of the scale factors (FP8_E8M0 | FP8_E4M3)
    route: Optional[str] = None
    # fp4 additions (append-only). `w_dtype` is W's torch dtype (== `dtype` unless the plan was built
    # with a per-weight `w_dtype`; None on a hand-built plan means "same as dtype"); `block_size` is the
    # scale block along K (32 for every E8M0 pair, 16 for NVFP4) -- what `_sf_view` sizes the blobs by.
    w_dtype: Any = None
    block_size: int = 32
    # Backward-driver additions (append-only; the defaults spell the forward's declaration).
    # `a_major` / `b_major` name the axis the graph DECLARED contiguous for A / B:
    #   A "k": dim [1, m, k] stride [m*k, k, 1]  (a row-major [m, k])
    #     "m": dim [1, m, k] stride [m*k, 1, m]  (the TRANSPOSE of a row-major [k, m] -- dY^T of a wgrad)
    #   B "k": dim [1, k, n] stride [k*n, 1, k]  (a row-major [n, k] weight, read transposed -- the forward)
    #     "n": dim [1, k, n] stride [k*n, n, 1]  (a row-major [k, n] -- X of a wgrad, W of a dgrad)
    # `split_k` is the REQUESTED knob (0 = the driver's pick, `build_proj_gemm`); the slices
    # that run are `plan.jit.config.split_k_slices` on the JIT route, and `tile_config_name`
    # spells them the catalog's way (`..._cluster2x1_2ctamma_splitK2` for S=2; no suffix at 1).
    a_major: str = "k"
    b_major: str = "k"
    split_k: int = 0
    # Serving-declaration addition (append-only): True = the plan was built `capture_safe` -- it never entered the ranked list
    # (no backend plan, no FROST-engine compile, whatever the opt-in flag said) and ends on the forced-tile JIT route by
    # construction (never the graph route, which creates a cuDNN handle at the first execute per (device, stream)).
    capture_safe: bool = False
    # The forced-tile JIT route's pack resolution (`_resolved_pack`): `(jit, {id(graph tensor key): id(bound tensor)})`, taken
    # once BY ROLE (the graph tensors the plan declares, never the buffers of a call) and OWNED BY THE PLAN, so it lives exactly
    # as long as the JIT it describes.  A cache slot, not a plan fact: no constructor argument (init=False), outside eq / repr.
    # Its predecessor was a module-level table keyed by `id(plan.jit)` that held a strong reference to every JIT ever resolved
    # (to keep the id valid) and never evicted -- every dropped block left its four forced-tile JITs, their bindings and graph
    # tensors alive for the life of the process.
    _resolved_keys: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def workspace_bytes(self) -> int:
        # The JIT artifact's own scratch (split-K fp32 partials) counts on EVERY route that has
        # one: `run_proj_gemm` launches `plan.jit` whenever it exists, and the graph's number is
        # the backend plan's, which knows nothing of a `split_k=` the driver pinned on the JIT
        # config.  The max of both keeps `gemm_scratch = max(plan.workspace_bytes)` honest under
        # either route (a forgotten reducer workspace does not fail the launch).
        jit_ws = int(getattr(self.jit, "workspace_bytes", 0) or 0) if self.jit is not None else 0
        if self.route == "jit-only":
            # No backend plan exists to ask; the JIT artifact knows its own (split-K only).
            return max(jit_ws, 1)
        return max(int(self.graph.get_workspace_size()), jit_ws, 1)

    def flops(self) -> int:
        """``2*M*N*K`` — the denominator for an MMA SOL number."""
        return 2 * self.m * self.n * self.k


# The shipped tile scorer has a REAL BUG, and this is the workaround for it.
#
# `gemm/frost/tile_config._tile_score` multiplies an efficiency factor
# `n_eff = N / (n_tiles * cta_n)` by `total_ctas`, which contains `n_tiles`.
# The product `n_tiles * n_eff` is `N / cta_n` -- independent of the tiling --
# so the raggedness penalty CANCELS EXACTLY and the score reduces to
# `1 / (sqrt(cta_n) * waves)`. At equal wave count it therefore always prefers
# the NARROWEST tile. Verified numerically: score(224)/score(256) is
# sqrt(256/224) to six digits at the out_proj shape.
#
# Cost here: `out_proj` is M x 4096 x 8192, and the scorer picks a 224-wide N
# tile against N = 4096. 256 divides 4096 into 16 tiles; 224 gives 18 full
# tiles plus a ragged 64, so the last tile does a full-width MMA for a quarter
# of a tile of useful work. Measured on Rubin: 2189 -> 2937 TFLOP/s at M=4096
# (+34%), and 256 also wins at M=8192 (3402 -> 3573) and M=32768 (3706 -> 3775).
#
# Fixing the scorer is the right long-term answer, but it is SHIPPED code used
# by every GEMM caller and the corrected score changes the pick on ~80% of a
# (M, N, K) grid -- that needs a measured sweep, not a drive-by edit. So the
# block names a config for the shape it owns, and the upstream fix is filed
# separately.
# The block's forced tile: a 256-wide N tile, a 128-tall CTA in a 2-CTA pair, the catalog's K=32 spelling.  The 64-byte MMA
# K form is the same geometry through `tile_config.as_mma_tile_k`, reached ONLY by an explicit `mma_tile_k_bytes=64`; the fp8
# MN-major table (`FP8_MN_MAJOR_VALIDATED`) is admitted at this tile and nowhere else.
_FORCED_TILE_NAME = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"


def _forced_tile_config(n: int) -> Optional[str]:
    """A catalog config whose N tile DIVIDES ``n``, or None to take the heuristic.

    Only claims the case it measured: a 256-wide N tile when 256 divides N.
    Anything else falls through to the scorer rather than guessing.
    """
    if n % 256 == 0:
        return _FORCED_TILE_NAME
    return None


# The SMALL-M route: the serving DECODE / MTP projections (T = B x S_q <= SMALL_M_MAX rows).
#
# At T <= 128 a projection GEMM is a weight-streaming problem, not an MMA one: the forced 256-wide tile pads every M tile to
# 256 rows (128 per CTA of the pair), launches ONE M tile (N / 256 cluster pairs: 52 for the 24/2 fused slab, 10 for its
# out projection) and spends 4-5x cuBLAS's time at T 1..128 (measured on the Rubin perf class, 212 SMs: 25.2 vs 8.4 us at
# T 1 on the 24/2 fused slab).  The catalog already carries the two axes that fix it, so no new kernel: ``swap_ab``
# computes ``C^T = W A^T`` with the WEIGHT rows on the MMA M axis (every 64- or 128-row weight tile is a CTA, so the
# machine is filled by N / tile_m CTAs) and the TOKENS on the MMA N axis, sized to T (the catalog's N tiles run 8..256 in
# steps of 8, so an 8-token step issues 64x8 / 128x8 MMAs instead of 256x256 ones); the two-kernel ``split_k_slices``
# fills the machine where the weight has too few rows for one wave (the out projection).  Measured on the Rubin perf
# class (one process, shuffled A/B/A x 3 x 100, CUPTI device time, control pair <= 0.6 %):
#
#   24/2 fused slab (N 13312, K 2560):    T 1   cuBLAS 8.4 | forced 25.3 | 64x8 cluster2x1 swapAB   6.1 us
#                                         T 4          7.1 |        16.1 | 64x8 cluster2x1 swapAB   6.1
#                                         T 32         7.2 |        13.5 | 64x32 cluster2x1 swapAB  6.3
#                                         T 128        9.7 |        13.7 | 128x64 (plain)           9.0
#   24/2 out projection (N 2560, K 6144): T 1   cuBLAS 7.3 | forced 24.6 | 128x8 splitK8 swapAB    6.3
#                                         T 4          7.2 |        24.6 | 128x8 splitK8 swapAB    6.4
#                                         T 32         7.6 |        24.5 | 128x32 splitK8 swapAB   7.0
#                                         T 128        8.4 |        24.5 | 64x32 (plain)           9.3
#
#   32/2 fused slab (N 17408, K 4096):    T 1   cuBLAS 13.3 | forced 45.6 | 128x8 swapAB            13.1
#                                         T 32        15.8 |        19.9 | 128x32 swapAB           13.3
#                                         T 128       21+  |        20.3 | 128x128 swapAB          15.5
#   32/2 out projection (N 4096, K 8192): T 1   cuBLAS 9.5  | forced 32.2 | 128x8 splitK6 swapAB    7.7
#                                         T 128       14.1 |        32.1 | 128x128 splitK6 swapAB  16.1
#   8/1 fused slab (N 4608, K 4096):      T 1   cuBLAS 5.8  | forced 21.4 | 64x8 cluster1x1 swapAB  6.6
#   8/1 out projection (N 4096, K 2048):  T 1   cuBLAS 3.9  | forced 11.2 | 64x8 swapAB (no split)  4.9   (cuBLAS stays ahead on this shard)
#
# The rule below spells those picks from the shape and the SM count alone (one-wave arithmetic, no measurement at plan
# time): the token tile is the power of two >= T; a NARROW weight (fewer than a third of a wave of 64-row tiles) takes
# 128-row weight tiles with the split-K slice count that fills one wave (capped at 8, every slice at least eight K tiles) up to
# 64 tokens and the plain 64x32 tile past it while that grid fits one wave -- a shallow K (fewer than 64 K tiles) takes the
# 64-row swap-AB tile instead of any split; a weight whose 64-row tiles fit one wave takes
# them (in a 2x1 cluster on a full machine) up to 64 tokens and the plain 128x64 tile past it; a weight whose 64-row tiles
# overflow a wave takes 128-row weight tiles (one wave).  Dense bf16 / f16, K-major A and B, no epilogue -- the forward's two
# projections; everything else keeps today's resolution.  N is a multiple of 256 (the forced tile's grid) on every branch but
# the last: a weight whose 64-row tiles overflow a wave is admitted at N % 128 == 0 too, because its 128-row tiles divide such an
# N exactly -- the indexer band's slab (QsaSpec.index_band: 13312 + 640 = 13952 = 109 x 128 columns at the 24/2 geometry, whose
# 218 64-row tiles overflow every served part) is the case, and it takes the SAME 128-row swap-AB names the 32/2 slab does (109
# CTAs: the half-wave launch a band padded to 768 columns would make, with no pad in the weight).  An N that 128 divides but 256
# does not whose 64-row tiles FIT one wave keeps today's resolution (the narrow and one-wave branches are unmeasured off the 256 grid).
SMALL_M_MAX = 128
_SMALL_M_SPLIT_K_MAX = 8
_SMALL_M_CTA_K_ELEMS = 64  # the catalog's 128-byte K tile at 2 bytes per element
_SMALL_M_SPLIT_K_MIN_K_TILES = 64  # split K only past this depth (K 4096 at 2 bytes): below it the reducer costs more than it buys
_SMALL_M_MIN_K_TILES_PER_SLICE = 8


def small_m_tile_config(
    m: int,
    n: int,
    k: int,
    *,
    dtype: torch.dtype,
    a_major: str = "k",
    b_major: str = "k",
    block_scale: bool = False,
    alpha: bool = False,
    sm_count: Optional[int] = None,
) -> Optional[str]:
    """The catalog config name of the small-M route for a ``[m, k] x [k, n]`` projection, or None when the route does not
    apply (a non-positive extent, ``m > SMALL_M_MAX``, a dtype other than bf16 / f16, a block-scale or fp8 plan, an MN-major
    operand, an extent the TMA 16-byte rule refuses, an N off the route's tile grids: 256 must divide N, or 128 where the 64-row
    tiles overflow a wave -- the indexer band's 13952-column slab).  ``sm_count`` is the target's SM count (the active device when None); the SAME shape resolves
    to the same name on every part of one SM-count class, so a recorded plan name replays."""
    if m <= 0 or n <= 0 or k <= 0:
        # a non-positive extent never reaches the one-wave arithmetic below (``n = 0`` would divide by a zero CTA count, ``m`` or
        # ``k`` at 0 would still spell a tile): the route declines and ``build_proj_gemm``'s own extent check names the extent
        return None
    if m > SMALL_M_MAX or block_scale or alpha:
        return None
    if dtype not in (torch.bfloat16, torch.float16):
        return None
    if a_major != "k" or b_major != "k":
        return None
    if n % 8 or k % 8:
        # The swapped problem reads the weight as the K-contiguous A operand and stores the output M-major: both extents
        # fall under TMA's 16-byte contiguous-extent rule at 2 bytes per element.
        return None
    if n % 128:
        # An N that 128 does not divide keeps today's resolution (the graph route; under ``capture_safe`` the typed refusal at
        # compile): the overflow branch's 128-row tiles would leave a partial tile, and the 64-row-tile branches below are gated
        # to the 256 grid (``n % 256`` and one wave), so 13376 = 209 x 64 is declined as well.
        return None
    if sm_count is None:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.tile_config import _sm_count

        sm_count = _sm_count()
    tn = 8
    while tn < m:
        tn *= 2
    ctas_64 = -(-n // 64)
    ctas_128 = -(-n // 128)
    if n % 256 and ctas_64 <= sm_count:
        # An N that 128 divides but 256 does not, whose 64-row tiles FIT one wave: the narrow and the one-wave branches below were
        # measured on 256-multiples only, so it keeps today's resolution (the graph route; under ``capture_safe`` the typed refusal
        # at compile).  The one served off-grid N -- the indexer band's slab, ``QsaSpec.index_band``: 13312 + 640 = 13952 = 109 x 128
        # -- overflows the wave on every served part (218 64-row tiles) and takes the overflow branch's 128-row tiles, which divide
        # it exactly (no partial tile).
        return None
    if 3 * ctas_64 < sm_count:
        # NARROW weight (fewer than a third of a wave of 64-row tiles).  With a DEEP K (>= 64 K tiles) 128-row weight tiles
        # and the split-K slice count that fills ONE wave (a slice count that overflows the wave loses its tail: 32/2 out
        # projection, 32 tiles, S 6 = 192 CTAs 7.7 us vs S 7 = 224 CTAs 9.0 us; every slice at least 8 K tiles); past 64
        # tokens the plain 64x32 tile where its grid fits one wave, else the same 128x128 split.  With a SHALLOW K the reducer's
        # fixed cost outweighs the split (8/1 out projection, K 2048: 4.9 us unsplit vs 5.9 split) -- the 64-row swap-AB tile.
        k_tiles = -(-k // _SMALL_M_CTA_K_ELEMS)
        slices = (
            max(1, min(_SMALL_M_SPLIT_K_MAX, sm_count // ctas_128, k_tiles // _SMALL_M_MIN_K_TILES_PER_SLICE)) if k_tiles >= _SMALL_M_SPLIT_K_MIN_K_TILES else 1
        )
        if m > 64 and (slices == 1 or -(-m // 64) * -(-n // 32) <= sm_count):
            return "CONFIG_sm100_64x32x128_64x32x32_cluster1x1_1ctamma"
        if slices > 1:
            return f"CONFIG_sm100_128x{tn}x128_128x{tn}x32_cluster1x1_1ctamma_splitK{slices}_swapAB"
        cluster = "2x1" if 2 * ctas_64 >= sm_count else "1x1"
        return f"CONFIG_sm100_64x{tn}x128_64x{tn}x32_cluster{cluster}_1ctamma_swapAB"
    if ctas_64 <= sm_count:
        # The 64-row weight tile fits one wave.  The 2x1 cluster (the token tile multicast across the pair) pays on a FULL
        # machine only (24/2 slab, 208 CTAs: +18 %; 8/1 slab, 72 CTAs: -10 %).
        if m <= 64:
            cluster = "2x1" if 2 * ctas_64 >= sm_count else "1x1"
            return f"CONFIG_sm100_64x{tn}x128_64x{tn}x32_cluster{cluster}_1ctamma_swapAB"
        # 65..128 tokens: the plain 128-token x 64-weight tile (full-rate MMA M) while its grid fits one wave.
        if -(-m // 128) * ctas_64 <= sm_count:
            return "CONFIG_sm100_128x64x128_128x64x32_cluster1x1_1ctamma"
        return "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_swapAB"
    # The 64-row tiles overflow a wave: 128-row weight tiles (one wave up to N = 128 x SMs; 32/2 slab, 136 CTAs; the indexer
    # band's 13952-column slab, 109 CTAs -- the one N off the 256 grid the route names a tile for).
    if m <= 64:
        return f"CONFIG_sm100_128x{tn}x128_128x{tn}x32_cluster1x1_1ctamma_swapAB"
    return "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma_swapAB"


def _auto_tile_name(m: int, n: int, k: int, *, dtype: torch.dtype, a_major: str, b_major: str, block_scale: bool, alpha: bool) -> Optional[str]:
    """``tile_config="auto"``: the small-M route where it applies (:func:`small_m_tile_config`; on the ``sm100`` compiler family
    only -- its names are the tcgen05 tree's catalog, so a process on the ``sm120`` family keeps the family-aware fallback below),
    else the forced 256-wide tile where 256 divides N (:func:`_forced_tile_config`), else None (the graph heuristic's own tile).
    A None here IS the graph route: ``build_proj_gemm(capture_safe=True)`` declines it typed, and the block sets ``capture_safe`` on a
    projection stage only where this resolver names a tile for the stage's own shape (``_Projection.resolves_a_jit_tile``) -- the
    indexer band's slab (N 13952, off the 256 grid) resolves here at ``m <= SMALL_M_MAX`` only, so a band declaration past 128 rows
    keeps the graph route and its documented capture refusal."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost import arch_family

    name = (
        small_m_tile_config(m, n, k, dtype=dtype, a_major=a_major, b_major=b_major, block_scale=block_scale, alpha=alpha)
        if arch_family.active_family() == "sm100"
        else None
    )
    return name if name is not None else _forced_tile_config(n)


def build_proj_gemm(
    *,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    label: str,
    pin_frost: bool = True,
    tile_config: Optional[str] = "auto",
    out_dtype: Optional[torch.dtype] = None,
    alpha: bool = False,
    block_scale: bool = False,
    sf_dtype: Any = None,
    mma_tile_k_bytes: Optional[int] = None,
    w_dtype: Optional[torch.dtype] = None,
    block_size: int = 32,
    a_major: str = "k",
    b_major: str = "k",
    split_k: int = 0,
    capture_safe: bool = False,
) -> ProjGemmPlan:
    """Compile one projection GEMM and pin the FROST plan.

    Plan-time only: shapes and dtypes come from the block's declaration, never
    from a runtime value (AGENTS.md Rule 4), and nothing here allocates a data
    tensor.

    ``pin_frost=False`` selects whatever the heuristic ranks first — used by the
    tests to get a native-cuDNN reference through the identical graph.

    **FP8 (E4M3) inputs.** ``dtype=torch.float8_e4m3fn`` builds the same graph
    with FP8 A/B, fp32 accumulate, and ``out_dtype`` output (default bf16 for
    fp8 inputs; for bf16/f16 inputs the output keeps the input dtype unless
    ``out_dtype`` says otherwise). ``alpha=True`` adds the per-tensor descale as
    a SCALAR-multiply epilogue -- ``c = matmul(A, B) * alpha`` with ``alpha`` a
    ``[1, 1, 1]`` fp32 graph tensor bound at execute (``run_proj_gemm(...,
    alpha=)``); the FROST engine recognises the ``[1,1,1]`` broadcast as its
    "scalar" fusion mode, so it is one fused epilogue, not a second kernel.
    A FORCED dense FP8 tile runs the K=32 MMA form (the named config's own width;
    half the Rubin 2xFP8 rate) unless ``mma_tile_k_bytes=64`` asks for the 64-byte
    form EXPLICITLY -- the quantized backward's GEMM stage does, the forward's plans
    stay at K32 (pinned by ``test_proj_gemm_bwd.py::test_forward_fp8_gemm_plans_stay_k32``)
    -- which is why the plan records the resolved tile config and its
    ``mma_tile_k_bytes`` for the perf table to state.  The knob is never keyed on the
    dtype here: ``_forced_tile_config(n)`` stays dtype-agnostic.

    **Block-scale (MXFP8) inputs.** ``block_scale=True`` (e4m3 ``dtype`` required,
    ``alpha`` must be False -- the E8M0 dequant is exact and happens IN the MMA,
    so there is nothing to descale -- and ``K % 32 == 0``) declares two
    per-32-block E8M0 scale-factor tensors in cuDNN's ``F8_128x4`` order --
    declared at the PADDED atom extents ``(1, ceil(rows/128)*128, ceil(K/32/4)*4)``
    (:func:`sf_padded_dims`), which the backend requires for that reorder -- and
    routes A / B through ``block_scale_dequantize`` before the matmul, as
    ``test/python/gemm/frost/test_block_scale_matmul.py`` builds it.  The
    caller binds the same PADDED blobs at execute (``run_proj_gemm(..., sf_a=, sf_w=)``).
    ``sf_dtype`` defaults to ``cudnn.data_type.FP8_E8M0`` (spelled ``None`` here so
    ``cudnn`` stays a lazy import).  The graph goes to the backend under
    ``[heur_mode.A, heur_mode.FALLBACK]``; if the backend declines it (a typed
    ``cudnnGraphNotSupportedError``, or no ``frost_gemm`` in the ranked list) the
    plan takes the JIT-ONLY route -- ``jit_from_cudnn_graph`` never touches the
    backend -- and says so in ``plan.route``.  Under ``pin_frost`` the block-scale
    kernel is ALWAYS the JIT at the resolved tile config (the forced 256-wide
    tile when 256 divides N, else the auto pick), so ``plan.tile_config_name`` /
    ``plan.mma_tile_k_bytes`` name what runs.

    Without ``mma_tile_k_bytes`` a FORCED block-scale tile follows the engine's
    ``preferred_mma_tile_k_bytes`` (64 on SM 10.7, else the named config's 32).
    ``mma_tile_k_bytes`` (8-bit operands only -- block-scale, and dense e4m3 since the
    backward GEMM drivers; 32 or 64) re-targets the resolved config's MMA-instruction
    K width through ``tile_config.as_mma_tile_k`` -- the width is A/B'd, never
    assumed.  ``None`` keeps the config's own width (the named 256-wide config is the
    K=32 form; a block-scale forced tile follows the engine's preference, below).  On a
    bf16 / f16 dense GEMM the knob is a ``ValueError`` (one MMA K width exists), and with
    ``pin_frost=False`` it is one too (no JIT to re-target -- a knob is honoured or
    refused, never dropped).  The dense 64-byte form exists for a 128-tall CTA in a
    2-CTA pair on SM 10.7-class silicon (``kernel_registry.dense_k64_envelope``); the
    forced 256-wide tile qualifies, and a shape that resolves elsewhere surfaces the
    compiler's own typed decline.

    **fp4 (E2M1) operands -- block-scale only.** ``w_dtype`` (default: ``dtype``)
    names W's dtype separately from A's, so stage (1) can run the catalog's MIXED
    row -- e4m3 ``h`` against a ``torch.float4_e2m1fn_x2`` weight -- and the fp4
    out projection can run the both-fp4 rows; ``block_size`` (32 | 16) is the scale
    block along K.  The served pairs are exactly :func:`block_scale_pairing`'s table
    (``kernel_registry._BLOCK_SCALE_CASES``): both e4m3 or exactly one e2m1 side ->
    E8M0 per 32; both e2m1 -> E4M3 per 16 (NVFP4) or E8M0 per 32 (MXFP4); any other
    ``(sf_dtype, block_size)`` is a ``ValueError`` -- the declared scale dtype selects
    the MMA's scale format, so a wrong one miscomputes rather than fails.  The graph
    keeps LOGICAL dims (``A [1, M, K]``, ``B [1, K, N]``) and declares the fp4 side's
    ``data_type``; the caller binds the PACKED storage (``[M, K/2]`` / ``[N, K/2]``,
    two codes per byte, low nibble = even k), which the kernel accepts as
    ``shape[-1] * kpack == K``.  An e4m3 scale blob is re-viewed as
    ``torch.float8_e4m3fn`` at bind (``_sf_view``).  ``alpha`` stays forbidden with
    ``block_scale`` for every pair, and an fp4 side without ``block_scale`` is a
    ``ValueError`` (there is no dense fp4 MMA).

    **Operand majors -- the backward's transposed operands (appended kwargs).**
    ``a_major`` (``"k"`` | ``"m"``) and ``b_major`` (``"k"`` | ``"n"``) pick the graph
    STRIDES of A / B; the dims stay ``[1, m, k]`` / ``[1, k, n]`` and the FROST GEMM
    engine infers each operand's major from them (``graph_analyzer._infer_a_major`` /
    ``_infer_b_major``) and renders its M-major-A / N-major-B template arms.  The
    defaults are the forward's declaration, byte-identical graph.  ``nn.Linear``
    backward (``api_bwd.py``): ``dW = dY^T @ X`` is ``a_major="m", b_major="n"`` with
    ``m=rows, k=T, n=cols`` and ``dX = dY @ W`` is ``a_major="k", b_major="n"`` with the
    UN-transposed row-major weight -- :func:`run_wgrad_gemm` / :func:`run_dgrad_gemm`
    build the views and refuse a mismatch.  The TMA 16-byte contiguous-extent rule
    now falls on the MN extent (``M % (16/BPE)`` for an M-major A, ``N % (16/BPE)``
    for an N-major B) and is a typed ``ValueError`` HERE, before any graph exists; the
    fp8 ``K % 16`` rule stays on the operands whose contiguous axis IS K (the forward's
    two, a dgrad's A), so a wgrad's ragged ``K = T`` is admitted -- validated on the
    device at ``T = 4104`` in both MMA K forms, like the bf16 twins.
    Served on the dense path: bf16 / f16 at any major, and e4m3 at exactly the
    ``(dtype, a_major, b_major)`` triples of :data:`FP8_MN_MAJOR_VALIDATED` -- the wgrad
    (``"m", "n"``) and dgrad (``"k", "n"``) renderings, validated on cc 10.7 at the forced
    tile in both MMA K forms against an fp64 reference of the dequantized e4m3 products
    (the bf16 twin's bound, a two-launch bitwise check, a sentinel-filled output); any
    other fp8 major, and a MIXED fp8 dtype pair, is a typed ``NotImplementedError`` naming
    the table -- and the table holds at that tile ONLY: an fp8 MN-major plan whose N 256
    does not divide (the heuristic's tile), an explicit ``tile_config`` other than the
    forced one, or ``pin_frost=False`` (the graph route, no JIT) is the same typed
    ``NotImplementedError``, naming the tile -- and so is a forced compile the compiler
    refuses: the graph-heuristic fallback the bf16 / f16 and K-major fp8 plans take would
    run the table at a tile nobody validated, so it is refused with the compiler's decline
    chained, never a silent route change.  The block-scale rows lay their scale-factor blobs over K-major operand
    rows and are refused with a non-K major (``ValueError``) -- the backward keeps every
    block-scale operand K-major (the transposed "columnwise" artifacts) and binds them
    through :func:`run_wgrad_gemm_block_scale` / :func:`run_dgrad_gemm_block_scale`.
    The forward's fp8 plans keep the K-major defaults and are untouched.

    **``split_k`` -- three values.** ``0`` (default) is the driver's pick:
    the FORCED tile's catalog ``split_k_slices=1`` on the JIT route -- the heuristic's
    ``_auto_split_k`` (``compiler.py``) is reached ONLY if the forced compile fell back
    to the graph route.  ``1`` PINS ``replace(cfg, split_k_slices=1)`` on the JIT config
    and REFUSES the graph fallback (:class:`SplitKPinRefused`): the recompute plans
    of the backward need this for their bit-identical claim, since a fallback would
    change config, route and slice count.  ``S >= 2`` is ``replace(cfg,
    split_k_slices=S)`` on the JIT config -- the two-kernel split with the FIXED-ORDER
    reducer (``kernel_templates/split_k_reduction_epilogue_fusion.py``; deterministic)
    and fp32 partials of ``S*M*N*4`` bytes, reported through ``plan.workspace_bytes``
    and handed to the JIT by :func:`run_proj_gemm` (``compiler._check_splitk_supported``
    gates S).  A pinned split without a forced name takes the engine's auto pick as
    the base geometry (``_auto_tile_config``), so it is always a JIT plan -- which is
    why ``split_k >= 1`` needs ``pin_frost=True``: with the FROST engine deselected there
    is no JIT to pin, and the combination is a ``ValueError`` (a knob is honoured or
    refused, never silently dropped).

    **The dense bf16 / f16 JIT-only route (no opt-in flag).** ``frost_gemm`` is an
    opt-in engine: without ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` the backend's ranked
    list has no FROST plan and the pin raises ``_NoFrostPlan``.  A dense bf16 / f16 plan
    whose tile is forced (``tile_config="auto"`` with ``N % 256 == 0``) or named
    explicitly then takes the SAME ``jit-only`` route the block-scale plans take: the
    graph is built only to be analyzed, ``jit_from_cudnn_graph`` never asks the backend,
    and ``run_proj_gemm`` launches the JIT before any handle exists -- so the block
    compiles with the flag UNSET (a host module must never set it: the flag is read live
    and flips every opt-in engine into every FE graph of the process).  Without a forced
    tile the graph route IS the plan, so that decline stays; the per-tensor fp8 dense
    plans keep it too (their alpha epilogue is validated on the pinned ``graph+jit``
    route).  When the flag IS set nothing changes for a PLAIN declaration: the backend
    admits the graph, the plan is pinned by name (``graph+jit``) and the same JIT runs.

    **CUDA-graph capture.** ``build_proj_gemm`` is refused under capture (a typed
    ``RuntimeError`` naming the route and the remedy, before the ``cudnn.pygraph``
    constructor creates its handle), and so is the GRAPH route's first execute on a
    stream (:func:`handle_for_stream`: no cuDNN handle is created inside a capture).
    ``capture_safe=True`` is the serving declaration (appended, default off): the plan
    NEVER ENTERS THE RANKED LIST -- the graph is validated and built only to be analyzed;
    ``create_execution_plans`` (the heuristics query that reads the flag), ``check_support``
    and ``build_plans`` are never called, so there is no backend plan and no FROST-engine
    compile whatever the flag says, and the plan is ``jit-only`` by construction.  A shape
    the forced tile does not divide, or ``pin_frost=False``, is a typed decline at compile
    time, and a forced compile the compiler refuses is a typed ``NotImplementedError``
    naming the tile and the reason, NEVER the warning + graph-heuristic fallback the plain
    declaration takes (a fallback would turn a capture-safe plan into one that creates a
    handle inside the capture).  The pin is skipped on purpose: with the flag set, a plain
    declaration's ``build_plans`` compiles the same ``jit_from_cudnn_graph`` inside the
    FROST GEMM engine, and a refusal there is neither typed for the tile nor caught by the
    fallback arm -- the driver's forced-tile compile has to be the ONLY compile for the
    refusal to be the typed one on every host, under either flag state.
    """
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    if w_dtype is None:
        w_dtype = dtype
    if a_major not in _A_MAJORS:
        raise ValueError(f"{label}: a_major must be one of {_A_MAJORS} (the axis of A [1, m, k] the graph declares contiguous), got {a_major!r}")
    if b_major not in _B_MAJORS:
        raise ValueError(f"{label}: b_major must be one of {_B_MAJORS} (the axis of B [1, k, n] the graph declares contiguous), got {b_major!r}")
    if isinstance(split_k, bool) or not isinstance(split_k, int) or split_k < 0:
        raise ValueError(
            f"{label}: split_k must be an int >= 0 (0 = the driver's pick, 1 = pin one slice on the JIT, S >= 2 = split-K with S slices), got {split_k!r}"
        )
    if split_k >= 1 and not pin_frost:
        raise ValueError(
            f"{label}: split_k={split_k} pins a FROST JIT config and cannot be combined with pin_frost=False (the deselected-FROST graph route "
            "has no JIT to pin, so the knob would be silently dropped)"
        )
    _refuse_compile_under_capture(label)
    if capture_safe and not pin_frost:
        raise ValueError(
            f"{label}: capture_safe=True needs the forced-tile JIT route, and pin_frost=False is the graph route by request (its first execute "
            "per stream creates a cuDNN handle, which cannot happen inside a CUDA-graph capture); drop one of the two"
        )
    # The tile the JIT route runs: the small-M route at m <= SMALL_M_MAX on a dense bf16 / f16 K-major plan, else the forced
    # 256-wide config when it divides N, an explicit name as given, else None (the graph heuristic's own tile -- no JIT
    # unless a split or an MMA K width is pinned below).
    name = (
        _auto_tile_name(m, n, k, dtype=dtype, a_major=a_major, b_major=b_major, block_scale=block_scale, alpha=alpha) if tile_config == "auto" else tile_config
    )
    if capture_safe and name is None and not (block_scale or split_k >= 1 or mma_tile_k_bytes is not None):
        # No JIT would be built: the plan would run the GRAPH route, whose first execute per (device, stream) creates a cuDNN
        # handle -- inside the capture a serving stack records with no warm-up.  Typed, before any graph or backend call.
        raise NotImplementedError(_capture_safe_graph_route_message(label, m, k, n, tile_config, "the graph heuristic's tile"))
    if mma_tile_k_bytes is not None:
        # The MMA K width's contract, checked ONCE and up front for the block-scale and the dense path alike, so an invalid
        # value is always named as such whatever else the call combines it with.  (1) 32 or 64, the tcgen05 MMA K widths.
        # (2) An 8-bit knob: e4m3 operands issue the 32- or 64-byte form (the 64-byte one is the backward GEMM stage's
        # explicit request; the forward's plans never pass it and stay at the named config's 32), a bf16 / f16 GEMM has
        # one width, so the knob is refused rather than ignored.  (3) The width lives on a FROST JIT config, which
        # pin_frost=False never builds -- a knob is honoured or refused, never silently dropped.
        if mma_tile_k_bytes not in (32, 64):
            raise ValueError(f"{label}: mma_tile_k_bytes must be None, 32 or 64 (the tcgen05 MMA K widths), got {mma_tile_k_bytes!r}")
        if not block_scale and not (_is_fp8(dtype) or _is_fp4(dtype) or _is_fp4(w_dtype)):
            raise ValueError(
                f"{label}: mma_tile_k_bytes is a knob of the 8-bit MMA paths (block-scale, and dense e4m3); a dense {dtype} GEMM issues one "
                "MMA K width -- pass None"
            )
        if not pin_frost:
            raise ValueError(
                f"{label}: mma_tile_k_bytes={mma_tile_k_bytes} re-targets a FROST JIT config and cannot be combined with pin_frost=False (the "
                "deselected-FROST graph route has no JIT to re-target, so the knob would be silently dropped)"
            )
    if (a_major != "k" or b_major != "k") and (block_scale or _is_fp4(dtype) or _is_fp4(w_dtype)):
        raise ValueError(
            f"{label}: M-major A / N-major B are served on the dense path only (got a_major={a_major!r}, b_major={b_major!r} with "
            f"block_scale={block_scale}, dtype={dtype}, w_dtype={w_dtype}); the block-scale rows' F8_128x4 scale-factor blobs are laid out over K-major operand rows"
        )
    if (
        (a_major != "k" or b_major != "k")
        and (_is_fp8(dtype) or _is_fp8(w_dtype))
        and (w_dtype != dtype or (dtype, a_major, b_major) not in FP8_MN_MAJOR_VALIDATED)
    ):
        # A TABLE of validated renderings, never a blanket lift: each admitted triple ran on cc 10.7 against the
        # fp64 reference at the forced tile (K32 and K64).  Everything else is declined by name.
        raise NotImplementedError(
            f"{label}: an fp8 (e4m3) operand with a_major={a_major!r}, b_major={b_major!r} (dtype={dtype}, w_dtype={w_dtype}) is not a rendering "
            f"this driver has validated on cc 10.7; the validated (dtype, a_major, b_major) triples are {_fp8_mn_major_menu()} -- "
            "use one of them, bf16 / f16 operands, or the K-major defaults"
        )
    if (a_major != "k" or b_major != "k") and _is_fp8(dtype):
        # The table holds at the ONE tile its rows were validated at: the forced 256-wide config on the FROST JIT (K32, and
        # K64 through mma_tile_k_bytes=64).  An N the forced tile does not divide resolves to the heuristic's tile, an
        # explicit other tile_config renders a tile nobody validated, and pin_frost=False builds no JIT at all; the compiler
        # itself admits any 8-bit MN-major tile whose slices are 128-element multiples, so the driver declines these by name.
        resolved = _forced_tile_config(n) if tile_config == "auto" else tile_config
        if not pin_frost or resolved != _FORCED_TILE_NAME:
            where = "the graph heuristic's tile" if resolved is None else f"the tile {resolved!r}"
            if not pin_frost:
                where += " on the graph route (no JIT)"
            raise NotImplementedError(
                f"{label}: the fp8 (e4m3) a_major={a_major!r}, b_major={b_major!r} rendering is validated at the forced tile "
                f"{_FORCED_TILE_NAME!r} on the FROST JIT only (K32, and K64 through mma_tile_k_bytes=64); n={n}, tile_config={tile_config!r}, "
                f"pin_frost={pin_frost} resolves to {where} -- use an N that 256 divides, or name that K32 tile_config explicitly, with "
                "pin_frost=True; the K64 form is reached through mma_tile_k_bytes=64 only, never by its config name"
            )
    if split_k and block_scale:
        raise ValueError(
            f"{label}: split_k={split_k} on the block-scale GEMM is not served by this driver (its reducer is verified on the dense path only); use split_k=0"
        )
    _check_mn_major_tma_rule(label, "A", a_major, m, "M", dtype)
    _check_mn_major_tma_rule(label, "B", b_major, n, "N", w_dtype)
    if block_scale:
        sf_dtype, block_size = block_scale_pairing(dtype=dtype, w_dtype=w_dtype, sf_dtype=sf_dtype, block_size=block_size, label=label)
        if alpha:
            raise ValueError(f"{label}: block_scale=True carries its descale in the per-block scale factors; alpha must be False")
        if k % 32:
            # Covers both the scale blocks (32, or 2 x 16) and TMA's 16-byte contiguous-extent rule
            # for a packed e2m1 operand (16 bytes = 32 codes; `gemm/frost/recipe.py` contiguous_modulus).
            raise ValueError(
                f"{label}: block_scale=True needs K % 32 == 0 (one E8M0 scale per 32-element block, or two E4M3 ones; the fp4 TMA rule), got K={k}"
            )
        if isinstance(tile_config, str) and tile_config != "auto" and not tile_config.startswith("CONFIG_sm100_"):
            # The sm103 block-scale template declares its rings A, B, SFA, SFB, so on Rubin's 327 KiB
            # budget its scale-factor roots cross the 256 KiB tcgen05-descriptor line and the kernel
            # returns NaN (pre-existing; `mma-tma-matrix.md` section 6).  A block-scale plan takes the
            # sm100 pipeline's catalog only -- refused HERE, before any backend or JIT sees the graph.
            raise ValueError(
                f"{label}: a block-scale plan takes an sm100-pipeline tile config (CONFIG_sm100_*), got {tile_config!r}; "
                "the sm103 block-scale template's SF rings sit past the 256 KiB descriptor line on Rubin and return NaN"
            )
    else:
        if _is_fp4(dtype) or _is_fp4(w_dtype):
            raise ValueError(
                f"{label}: fp4 (e2m1) operands ride the block-scale GEMM only -- pass block_scale=True with their scale factors; got dtype={dtype}, w_dtype={w_dtype}"
            )
        if w_dtype != dtype:
            raise ValueError(f"{label}: the dense GEMM takes one operand dtype (A {dtype} vs W {w_dtype}); mixed dtypes exist only as block-scale rows")
        if sf_dtype is None:
            sf_dtype = cudnn.data_type.FP8_E8M0
    # The op-recording hook the JIT-only route reads the graph through is installed by this
    # import; the graph engine imports it too, but a backend that declines the graph early
    # would otherwise leave the JIT nothing to analyze.
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost; import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn  # noqa: F401

    for lbl, v in (("m", m), ("k", k), ("n", n)):
        if v <= 0:
            raise ValueError(f"{label}: {lbl} must be > 0, got {v}")
    fp8 = _is_fp8(dtype)
    fp4_a, fp4_w = _is_fp4(dtype), _is_fp4(w_dtype)
    if fp8 and k % 16 and (a_major == "k" or b_major == "k"):
        # TMA's 16-byte contiguous-extent rule at 1 B/elem (compiler._tma_alignment_reject), on the operands whose
        # CONTIGUOUS axis is K: the forward's two, a dgrad's A.  A wgrad (M-major A, N-major B) has no K-contiguous
        # operand -- its rule fell on M / N above (`_check_mn_major_tma_rule`) and its ragged K = T tail is TMA
        # zero-fill, as for the bf16 twins; the device cells run it at T = 4104 (32 CTA K tiles of 128 e4m3 elements
        # plus an 8-element tail) in both MMA K forms, so the rule is scoped rather than blanket.
        which = " and ".join(f"{op} ({mj}-major)" for op, mj in (("A", a_major), ("B", b_major)) if mj == "k")
        raise ValueError(f"{label}: FP8 operands need K % 16 == 0 (16-byte TMA rule at 1 B/elem on {which}: K is its contiguous axis), got K={k}")
    if out_dtype is None:
        out_dtype = torch.bfloat16 if (fp8 or fp4_a) else dtype
    out_dt = _cudnn_dtype(out_dtype)
    # The graph's io dtype is what A / B take when they carry no explicit `data_type=`.  An fp4 side
    # ALWAYS declares its own (below), so with an fp4 A the io dtype names no tensor at all and the
    # output's dtype stands in; every non-fp4 plan keeps the io dtype it always had (byte-identical graph).
    io_dt = out_dt if fp4_a else _cudnn_dtype(dtype)

    g = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # Dims are LOGICAL for an fp4 side too (K codes, not K/2 bytes): the analyzer reads the
    # element type off `data_type=` and the binding's `shape[-1] * kpack == K` check accepts the
    # packed [.., K/2] storage -- exactly how test_block_scale_matmul.py declares its fp4 operands.
    # A is a row-major [M, K] (K-major, the forward) or the transpose of a row-major [K, M]
    # (M-major: stride[1] == 1, a wgrad's dY^T); the engine reads the major off the stride.
    a = g.tensor(name="A", dim=[1, m, k], stride=[m * k, k, 1] if a_major == "k" else [m * k, 1, m], **({"data_type": _cudnn_dtype(dtype)} if fp4_a else {}))
    # B is the weight as a checkpoint stores it -- [N, K] row-major -- declared
    # transposed. stride[1] == 1 is what makes it the K-contiguous operand (the forward);
    # an N-major B (stride[2] == 1) is a row-major [K, N] -- a wgrad's X or a dgrad's
    # un-transposed W.  It names its dtype only when it differs from the graph's io dtype
    # (an fp4 W, or W of the other side's dtype).
    b = g.tensor(
        name="B",
        dim=[1, k, n],
        stride=[k * n, 1, k] if b_major == "k" else [k * n, n, 1],
        **({"data_type": _cudnn_dtype(w_dtype)} if (fp4_w or w_dtype != dtype) else {}),
    )
    sfa_t = sfb_t = None
    if block_scale:
        # One E8M0 scale per 32-element K block, in cuDNN's F8_128x4 (128 rows x 4 blocks
        # = 512 B atoms, row-major) order: SFA over A's rows, SFB over B's N (the weight's
        # rows).  Declared at the PADDED atom extents -- rows to 128, K-blocks to 4 -- which
        # is (i) what the backend's block_scale_dequantize finalize REQUIRES for F8_128x4
        # ("the scale tensor ... must be (1, 1024, 128) but found (1, 1000, 128)",
        # CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH -- a bare RuntimeError, not a typed decline;
        # measured 2026-09-15 on the dev node at M=1000, K=544 (sf_k 17 -> 20) and N=392), and
        # (ii) exactly the padded blob `run_proj_gemm` binds (`sf_blob_bytes`).  The FROST JIT
        # never reads these dims (the analyzer takes the SF dtype / reorder; the lowering
        # derives ceil(K/32/4) x ceil(rows/128) from A / B), so the declaration is inert there.
        # test_block_scale_matmul.py:81-112 declares the UNPADDED [1, M, K/32] -- fine for its
        # JIT-only path, rejected by the backend the moment M % 128 != 0.
        m_pad, sf_k = sf_padded_dims(m, k, block_size)
        n_pad, _ = sf_padded_dims(n, k, block_size)
        sfa_t = g.tensor(name="SFA", dim=[1, m_pad, sf_k], stride=[m_pad * sf_k, sf_k, 1], data_type=sf_dtype, reordering_type=cudnn.tensor_reordering.F8_128x4)
        sfb_t = g.tensor(name="SFB", dim=[1, sf_k, n_pad], stride=[sf_k * n_pad, 1, sf_k], data_type=sf_dtype, reordering_type=cudnn.tensor_reordering.F8_128x4)
        ad = g.block_scale_dequantize(input=a, descale=sfa_t, block_size=[1, block_size])
        bd = g.block_scale_dequantize(input=b, descale=sfb_t, block_size=[block_size, 1])
        mm = g.matmul(A=ad, B=bd, name=label)
    else:
        mm = g.matmul(A=a, B=b, name=label)
    alpha_t = None
    if alpha:
        # dim [1,1,1] == graph_analyzer "scalar" broadcast mode -> fused epilogue multiply.
        # The graph tensor / op names land in the GENERATED kernel source as
        # Python identifiers, so a label with a space or a dash ("1 qkv_gate_proj")
        # is a SyntaxError at JIT time -- sanitise to an identifier.
        _ident = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(label)) or "proj"
        if _ident[0].isdigit():
            _ident = "p_" + _ident
        alpha_t = g.tensor(name=f"{_ident}_alpha", dim=[1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
        c = g.mul(a=mm, b=alpha_t, name=f"{_ident}_scale")
    else:
        c = mm
    c.set_output(True).set_data_type(out_dt)

    plan = ProjGemmPlan(
        graph=g,
        a=a,
        b=b,
        c=c,
        m=m,
        k=k,
        n=n,
        label=label,
        dtype=dtype,
        out_dtype=out_dtype,
        alpha=alpha_t,
        block_scale=block_scale,
        sfa=sfa_t,
        sfb=sfb_t,
        w_dtype=w_dtype,
        block_size=block_size,
        a_major=a_major,
        b_major=b_major,
        split_k=split_k,
    )
    plan.sf_dtype = sf_dtype if block_scale else None
    plan.tile_config_name = "heuristic (graph engine)"
    plan.route = "graph"

    def _backend_pin() -> None:
        """validate -> build -> rank -> pin (or deselect) -> check_support -> build_plans on the backend."""
        g.validate()
        g.build_operation_graph()
        # The block-scale graph is ranked under [A, FALLBACK] (as the FROST GEMM suite's own
        # `_pin_frost` does); the dense graph keeps the A-only list it always had.
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK] if block_scale else [cudnn.heur_mode.A])
        names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
        if pin_frost:
            # Plan names carry the plan's knobs since the shared knob vocabulary
            # landed ("frost_gemm[CTA_GROUP=2, MMA_TILE_K=32, ...]"), so the pin
            # matches the engine NAME (exact, or followed by its knob bracket) and
            # takes the top-ranked FROST configuration; deselect_engines already
            # matches by substring (the engine name is the user-visible API).
            frost_idx = _frost_plan_index(names)
            if frost_idx is None:
                raise _NoFrostPlan(f"{label}: the {_FROST_GEMM_PLAN!r} plan is not in the ranked list {names}. {_why_no_frost_plan()}")
            g.select_plan(frost_idx)
        else:
            g.deselect_engines([_FROST_GEMM_PLAN])
        g.check_support()
        g.build_plans()

    # A DENSE bf16 / f16 plan with a forced (or named) tile may run with no backend plan at all: the JIT never asks the
    # backend, `run_proj_gemm` launches it before any handle, so an unset CUDNN_FRONTEND_ENABLE_FROST_ENGINES (frost_gemm is
    # opt-in; a host module never sets the flag) is not a decline here.  Without a forced tile the graph route IS the plan,
    # so the decline stays; the per-tensor fp8 plans keep it too (their alpha epilogue is validated on the pinned route).
    dense_jit_only_ok = pin_frost and not block_scale and name is not None and dtype in (torch.bfloat16, torch.float16)
    backend_decline: Optional[BaseException] = None
    if capture_safe:
        # The SERVING declaration never enters the ranked list.  The graph is validated and built only to be ANALYZED (the
        # analyzer reads the tensor metadata the build fills in); `create_execution_plans` -- the heuristics query that reads
        # the opt-in flag and would put `frost_gemm` on the list -- `check_support` and `build_plans` are never called, so
        # there is no backend plan and no FROST-engine compile WHATEVER the flag says.  With the flag set, a plain
        # declaration's `build_plans` compiles the SAME `jit_from_cudnn_graph` inside the FROST GEMM engine, and a refusal
        # there reaches the caller as the engine's own error (or a bare one): neither typed for the tile nor caught by the
        # fallback arm below.  Skipping the pin makes the driver's forced-tile compile below the ONLY compile, so its refusal
        # is the one typed decline, on every host, under either flag state.  `workspace_bytes` reads the JIT's own number on
        # this route (there is no backend plan to ask).
        g.validate()
        g.build_operation_graph()
        plan.route = "jit-only"
        _LOG.info(
            "%s: capture_safe declaration -- no ranked list, no backend plan, no engine-side compile; the forced-tile JIT-only route at %s",
            label,
            name if name else "the engine's auto-picked config",
        )
    elif pin_frost and (block_scale or dense_jit_only_ok):
        # A backend that does not admit the e4m3 + E8M0 two-dequant graph is a typed decline
        # (cudnnGraphNotSupportedError), and "no frost_gemm in the list" is the `_NoFrostPlan`
        # above; both leave the JIT-only route, which never asks the backend.  Anything else --
        # a backend failure inside check_support / build_plans, a bare RuntimeError -- is a
        # bug and PROPAGATES (engine contract rule 2); the catch is deliberately this narrow.
        # WARNING, not INFO, on the block-scale route: there the plan-name pin is absent and `workspace_bytes`
        # is the JIT artifact's, which a caller reading the perf table should see without -v.  INFO on the
        # dense route: compiling without the opt-in flag is the serving stack's intended path, not a surprise.
        try:
            _backend_pin()
        except (cudnn.cudnnGraphNotSupportedError, _NoFrostPlan) as exc:
            plan.route = "jit-only"
            backend_decline = exc
            if block_scale:
                _LOG.warning("%s: backend declined the block-scale graph (%s: %s); taking the JIT-only route", label, type(exc).__name__, str(exc)[:200])
            else:
                _LOG.info(
                    "%s: no backend plan for the dense %s graph (%s: %s); taking the forced-tile JIT-only route at %s (no backend plan, no cuDNN handle at execute)",
                    label,
                    _dtype_word(dtype),
                    type(exc).__name__,
                    str(exc)[:200],
                    name,
                )
    else:
        _backend_pin()

    # Forced tile: JIT the SAME graph at a named config. The graph engine picks
    # its config internally with no override hook (graph_analyzer.py:571-574),
    # so taking a specific tile means going through the JIT entry point. The
    # graph is still built above -- it is what gets analyzed, and it keeps
    # `workspace_bytes` and the plan-name pin honest.  A block-scale plan under
    # `pin_frost` ALWAYS takes the JIT (forced name, else the auto pick) so one
    # binding path (`bd.sfa_operands` / `bd.sfb_operands`) serves every shape.  So does
    # an explicit `mma_tile_k_bytes`: the width lives on a JIT config, and a dense shape
    # outside the forced tile's N would otherwise drop the knob on the graph route.
    pinned_split = split_k >= 1  # split_k=1 pins ONE slice (and refuses the fallback); S >= 2 splits
    if pin_frost and (name or block_scale or pinned_split or mma_tile_k_bytes is not None):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.compiler import jit_from_cudnn_graph
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.tile_config import as_mma_tile_k, by_name

        cfg = by_name(name) if name else _auto_tile_config(g)
        if mma_tile_k_bytes is not None:
            cfg = as_mma_tile_k(cfg, mma_tile_k_bytes)
            if cfg.mma_tile_k_bytes != mma_tile_k_bytes:
                raise ValueError(f"{label}: config {cfg.name!r} cannot issue mma_tile_k_bytes={mma_tile_k_bytes} (it stays at {cfg.mma_tile_k_bytes})")
        elif block_scale and name:
            # A FORCED name carries the catalog's K=32 spelling, which would silently pin the block-scale GEMM to the
            # narrow MMA form the engine's own auto path abandons on Rubin (`preferred_mma_tile_k_bytes`: 64 wherever the
            # active GPU issues the 64-byte block-scale MMA, 32 elsewhere -- so SM100 is unchanged).  Measured on the
            # 397B geometry, perf node c09 @2376 MHz, S=16K: MXFP8 qkv+gate proj 0.344 -> 0.298 ms (82 -> 95 % of the fp8
            # MMA peak), the mixed fp8 x fp4 row 0.332 -> 0.268 ms, the fp4 x fp4 out proj 0.114 -> 0.107 (NVFP4) /
            # 0.113 -> 0.102 ms (MXFP4); every S in {4K, 16K, 32K} and every mode moved the same way, so the forced tile
            # follows the engine's preference and an explicit `mma_tile_k_bytes` stays the override.
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.graph_analyzer import analyze
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.kernel_registry import preferred_mma_tile_k_bytes

            cfg = as_mma_tile_k(cfg, preferred_mma_tile_k_bytes(analyze(g)))
        if pinned_split:
            # The catalog entry carries split_k_slices=1 and `jit_from_cudnn_graph` never calls
            # `_auto_split_k`, so this replace is the ONLY way a slice count reaches the JIT.
            from dataclasses import replace

            cfg = replace(cfg, split_k_slices=split_k)
        # A forced compile the compiler refuses is a FALLBACK to the graph heuristic -- not a failure -- only where the plan's
        # contract allows one: an implicit tile (tile_config="auto"), the driver's own slice count (split_k=0), no explicit MMA
        # K width, and a dense bf16 / f16 or K-major fp8 GEMM.  Everything else is a typed REFUSAL, never a silent route change:
        # an explicit request (tile_config, mma_tile_k_bytes) and the block-scale path surface the compiler's own reason; a
        # pinned split refuses the fallback as SplitKPinRefused; an admitted fp8 MN-major triple (FP8_MN_MAJOR_VALIDATED)
        # refuses it with the admission's own NotImplementedError -- the table is validated at the forced tile on the FROST
        # JIT only, and the graph heuristic's tile is not that tile.
        try:
            compiled = jit_from_cudnn_graph(g, config=cfg)
        except Exception as exc:
            if capture_safe:
                # The serving declaration: the graph-heuristic fallback would create a cuDNN handle at the first execute per
                # stream -- inside the capture a serving stack records with no warm-up -- so a refused forced compile is a
                # typed decline naming the tile and the compiler's reason, never a warning and a silent route change.
                raise NotImplementedError(
                    f"{label}: capture_safe=True pins the forced-tile JIT route at {cfg.name!r}, and that compile was refused "
                    f"({type(exc).__name__}: {str(exc)[:300]}); the graph-heuristic fallback is refused too -- its first execute on a stream "
                    "creates a cuDNN handle, which cannot happen inside a CUDA-graph capture"
                ) from exc
            if tile_config != "auto" or block_scale or mma_tile_k_bytes is not None:
                raise  # explicit requests (and the block-scale path, which has no graph fallback) surface their reason
            if pinned_split:
                # The graph heuristic would change the config, the route and the slice count -- exactly
                # what a pinned split forbids -- so the compiler's own decline is surfaced, typed, instead.
                raise SplitKPinRefused(
                    f"{label}: split_k={split_k} pins the JIT at {cfg.name!r}; its compile was refused ({type(exc).__name__}: {str(exc)[:300]}) "
                    "and the graph-heuristic fallback is refused too (it would change the tile config, the route and the slice count)"
                ) from exc
            if (dtype, a_major, b_major) in FP8_MN_MAJOR_VALIDATED:
                # The admission above holds the table at the forced tile on the FROST JIT only; the fallback would run e4m3
                # MN-major operands at the heuristic's tile (on cc 10.7 also its K64 MMA form), which nobody validated.
                raise NotImplementedError(
                    f"{label}: the fp8 (e4m3) a_major={a_major!r}, b_major={b_major!r} rendering is validated at the forced tile {name!r} on the "
                    f"FROST JIT only (K32, and K64 through mma_tile_k_bytes=64), and that compile was refused ({type(exc).__name__}: {str(exc)[:300]}); "
                    "the graph heuristic's tile is not validated for e4m3 MN-major operands, so the fallback the bf16 / f16 and K-major fp8 plans "
                    "take is refused too"
                ) from exc
            if plan.route == "jit-only":
                # The dense jit-only route has NO backend plan to fall back onto (the pin above raised): today's typed decline
                # stands, with the compiler's refusal chained -- nothing can serve this plan without the opt-in flag.
                raise _NoFrostPlan(
                    f"{label}: {backend_decline}; and the forced-tile JIT compile at {cfg.name!r} was refused too "
                    f"({type(exc).__name__}: {str(exc)[:300]}) -- no route can serve this plan"
                ) from exc
            compiled = None
            # WARNING, not DEBUG (like the block-scale twin above): the fallback changes the tile config, the
            # route and possibly the slice count (`_auto_split_k`), under a different plan name -- a perf table
            # or a bit-identical claim built on it must see the switch without -v.
            _LOG.warning(
                "%s: forced tile %s rejected (%s: %s); falling back to the graph heuristic (tile config, route and slice count change)",
                label,
                name,
                type(exc).__name__,
                str(exc)[:200],
            )
        if compiled is not None:
            plan.jit, plan.jit_binding = compiled, compiled.binding
            plan.tile_config_name = getattr(compiled.config, "name", name)
            plan.mma_tile_k_bytes = getattr(compiled.config, "mma_tile_k_bytes", None)
            if plan.route == "graph":
                plan.route = "graph+jit"
    plan.capture_safe = bool(capture_safe)
    if capture_safe and plan.jit is None:
        # Unreachable by the checks above (the graph route was declined up front, a refused compile raised typed); kept so the
        # invariant the serving stack relies on -- a capture_safe plan HAS a JIT -- is asserted where the plan is finished.
        raise NotImplementedError(_capture_safe_graph_route_message(label, m, k, n, tile_config, str(plan.tile_config_name)))
    return plan


def _capture_safe_graph_route_message(label: str, m: int, k: int, n: int, tile_config, resolved: str) -> str:
    return (
        f"{label}: capture_safe=True needs the forced-tile JIT route, but this plan resolves to the graph route ({resolved}; m={m}, k={k}, "
        f"n={n}, tile_config={tile_config!r}): the graph route creates a cuDNN handle at the first execute per stream, which cannot happen "
        f"inside a CUDA-graph capture -- use an N that 256 divides (the forced tile {_FORCED_TILE_NAME!r}), a shape the small-M route "
        f"names a tile for (m <= {SMALL_M_MAX} on a dense bf16 / f16 K-major projection), or name a tile_config explicitly"
    )


def _auto_tile_config(g):
    """The tile config the FROST graph engine's automatic strategy would pick for ``g``
    (``select_config`` + ``preferred_strategy``), for a JIT with no forced name."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.compiler import _graph_dynamic_shapes, plan_config
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.graph_analyzer import analyze

    return plan_config(analyze(g), dynamic_shapes=_graph_dynamic_shapes(g))


def run_proj_gemm(
    plan: ProjGemmPlan,
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    alpha: Optional[torch.Tensor] = None,
    sf_a: Optional[torch.Tensor] = None,
    sf_w: Optional[torch.Tensor] = None,
    *,
    stream=None,
) -> None:
    """Launch. No allocation, no conversion — the caller owns every buffer.

    ``a`` is ``[M, K]``-shaped storage, ``w`` is ``[N, K]``, ``out`` is
    ``[M, N]``; any leading batch dim of 1 is accepted since the graph carries
    the layout and only the pointer is bound.

    **The launch stream (Rule 5).** ``stream`` is a raw ``CUstream`` int (or a
    ``cuda.CUstream``) -- the same value the block's CuTe-DSL stages take.  It
    reaches BOTH routes: the forced-tile JIT plan's own ``stream=``, and, on the
    graph route, a cached per-(device, stream) cuDNN handle bound to it
    (:func:`handle_for_stream`; the graph API carries a stream on a HANDLE, not
    on an execute argument).  ``None`` means torch's current stream on
    ``out.device`` -- so a standalone call under ``with torch.cuda.stream(s):``
    stays ordered with the caller's torch work, exactly as every other stage of
    the block does.  Before this the two GEMMs launched on the default stream
    regardless, and a block run on a side stream had its SDPA read a slab the
    projection had not written yet (all-zero output on SM107).

    ``handle`` remains the graph API's classic way of naming the stream: given
    alone, its stream is the launch stream; given WITH ``stream``, the two must
    agree -- a handle bound to another stream would run this GEMM off the stream
    the caller ordered its other work on, which is the race this guards, so it
    is a ``ValueError`` rather than a silent pick.

    ``sf_a`` / ``sf_w`` (block-scale plans only, both required): the F8_128x4
    E8M0 scale-factor blobs of ``a`` (over its M rows) and ``w`` (over its N
    rows), uint8 or ``float8_e8m0fnu``, contiguous, PADDED to whole atoms --
    ``ceil(rows/128)*128 * ceil(K/32/4)*4`` bytes, exactly what
    ``swizzle_sf_rowwise`` / ``to_blocked`` of the padded logical ``[rows, K/32]``
    scale matrix produces.  They are bound as rank-3 ``(1, rows_pad, 4*k4)``
    VIEWS (a uint8 blob is re-viewed as e8m0, no copy) -- the kernel reads a base
    pointer and rebuilds the layout from the problem size.

    Both routes take the SAME variant pack keyed by the plan's graph tensors
    (``plan.a`` / ``plan.b`` / ``plan.c`` / ``plan.sfa`` / ``plan.sfb`` / ``plan.alpha``):
    the JIT resolves by tensor identity as the graph does, and graph tensors are
    invariant under a ``swap_ab`` tile config (which swaps the binding's LISTS).

    **Operand dtype and fp4 extent are CHECKED here** (``a.dtype == plan.dtype``,
    ``w.dtype == plan.w_dtype``, ``out.dtype == plan.out_dtype``; an e2m1 side's storage
    ``shape[-1] * 2 == K``).  The graph carries the dtypes and the kernel binds a pointer,
    so before this a bf16 ``w`` handed to an fp8 plan was silently REINTERPRETED
    byte-for-byte -- the one deliberate tightening the fp4 widening brings, because a
    ``[N, K]`` fp4 tensor (logical shape) and a ``[N, K/2]`` one (storage) are one wrong
    bind apart.  The output gate is the same rule on C: its stores are typed by the plan,
    so a same-size buffer of another dtype would hold wrong bit patterns and a narrower
    one would be overrun -- refused by name before any route, for the forward and both
    backward drivers alike.
    """
    _check_operand(plan, a, "a", plan.dtype)
    _check_operand(plan, w, "w", _w_dtype_of(plan))
    # The OUTPUT too: the graph carries C's dtype and the kernel's stores use it, so a same-size f16
    # buffer on a bf16 plan would hold bf16 bit patterns (wrong values, no error) and a narrower one
    # (an e4m3 slab) would be overrun by 2-byte stores.  Shared by the forward and both backward
    # drivers; a hand-built plan with no out_dtype (the routing probes) checks nothing, as for A / W.
    if out is not None and plan.out_dtype is not None and out.dtype != plan.out_dtype:
        raise ValueError(f"{plan.label}: out is {out.dtype} but this plan was built for {plan.out_dtype}; refusing to reinterpret the bytes")
    if plan.block_scale:
        sf_a3, sf_w3 = _sf_view(plan, sf_a, "sf_a", plan.m), _sf_view(plan, sf_w, "sf_w", plan.n)
    elif sf_a is not None or sf_w is not None:
        raise ValueError(f"{plan.label}: this plan has no block-scale dequant (built with block_scale=False); refusing to drop the scale factors silently")
    # `alpha`: the per-tensor descale for an fp8 plan built with alpha=True --
    # a 1-element fp32 CUDA tensor, bound as the [1,1,1] scalar aux (a VIEW).
    if plan.has_alpha:
        if alpha is None:
            raise ValueError(f"{plan.label}: this plan was built with alpha=True; pass alpha= (1-element fp32 CUDA tensor). No silent 1.0 (Rule 1).")
        if alpha.numel() != 1 or alpha.dtype != torch.float32 or not alpha.is_cuda:
            raise ValueError(f"{plan.label}: alpha must be a 1-element fp32 CUDA tensor, got {tuple(alpha.shape)} {alpha.dtype} on {alpha.device}")
        alpha3 = alpha.reshape(1, 1, 1)  # a view: 1 element is always contiguous
    elif alpha is not None:
        raise ValueError(f"{plan.label}: this plan has no alpha epilogue (built with alpha=False); refusing to drop the value silently")
    # Resolve the launch stream FIRST (Rule 5), before any route is taken.
    if handle is not None:
        import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

        handle_stream = int(cudnn.get_stream(handle) or 0)
        if stream is not None and int(stream) != handle_stream:
            raise ValueError(
                f"{plan.label}: handle is bound to stream {handle_stream:#x} but stream={int(stream):#x} was requested; "
                "a GEMM off the launch stream races the block's other stages (Rule 5). Pass one or make them agree (cudnn.set_stream)."
            )
        stream = handle_stream
    elif stream is None:
        stream = torch.cuda.current_stream(out.device).cuda_stream
    stream = int(stream)
    # ONE variant pack, keyed by the GRAPH tensors, for both routes.  The JIT resolves keys
    # by tensor identity too (`resolve_variant_pack`), and the graph tensors are the
    # swap-INVARIANT spelling: `jit_from_cudnn_graph` applies `swap_ab_binding` under a
    # `swap_ab` tile config, which trades the binding's a/b (and sfa/sfb) LISTS while
    # keeping the tensor objects -- so keying by `binding.a_operands[0]` would hand A's
    # buffer to B under any `_swapAB` config a caller passes as `tile_config=`.
    vp = {plan.a: _rank3(a, "a"), plan.b: _rank3(w, "w"), plan.c: _rank3(out, "out")}
    if plan.block_scale:
        vp[plan.sfa] = sf_a3
        vp[plan.sfb] = sf_w3
    if plan.has_alpha:
        vp[plan.alpha] = alpha3
    if plan.jit is not None:
        need = int(getattr(plan.jit, "workspace_bytes", 0) or 0)
        # The pack's KEY -> bound-tensor resolution (`resolve_variant_pack`: four dicts over the binding per call) is a plan-time
        # fact -- the keys are the plan's own graph tensors, i.e. its ROLES -- so it is taken ONCE per plan (kept on it,
        # `plan._resolved_keys`) over the graph tensors themselves, never over this call's buffers (one read-only tensor may be
        # bound as both A and W), and the per-call pack maps straight to `{id(bound): buffer}` for `run_resolved`, the same launch
        # path `__call__` takes after resolving.  This driver sits on the host-bound path of the fp8 block backward at short
        # sequences.
        resolved = _resolved_pack(plan, vp)
        if need:
            # Split-K partials: the JIT carves them out of the CALLER's workspace (never allocates).
            # `Workspace` validates presence, contiguity, size and 128-B alignment and raises with
            # the required size -- the same carver the graph engine's execute path uses.  `device=`
            # pins it to the launch device: without it a host (CPU) buffer of the right size and
            # alignment passes every other check and reaches the launch boundary as a bogus pointer.
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.workspace import Workspace

            ws_obj = Workspace(workspace, need, plan.label, device=out.device.index)
            if resolved is None:
                plan.jit(vp, stream=stream, workspace=ws_obj)
            else:
                plan.jit.run_resolved(resolved, stream=stream, workspace=ws_obj)
        elif resolved is None:
            plan.jit(vp, stream=stream)
        else:
            plan.jit.run_resolved(resolved, stream=stream)
        return
    if plan.route == "jit-only":
        raise RuntimeError(f"{plan.label}: the backend declined this graph (route=jit-only) and no JIT artifact was built -- nothing can launch it")
    if plan.capture_safe:
        raise RuntimeError(
            f"{plan.label}: a capture_safe plan reached the graph route (route={plan.route!r}, no JIT) -- build_proj_gemm refuses that at "
            "compile time, so this plan was altered after it was built"
        )
    # The graph route names its stream through the handle: the plan's engine
    # reads `ExecutionContext.stream` off it (`_pygraph.execute` -> `cudnn.get_stream`).
    # Creating that handle inside a CUDA-graph capture is refused typed in `handle_for_stream`.
    if handle is None:
        handle = handle_for_stream(out.device, stream)
    plan.graph.execute(vp, workspace, handle)


def _resolved_pack(plan: ProjGemmPlan, vp: dict) -> Optional[dict]:
    """``{id(bound tensor): buffer}`` for the JIT's ``run_resolved`` from the graph-tensor-keyed pack.

    The KEY resolution -- which of the JIT's bound tensors each graph tensor of the pack names -- is a property of the plan's
    ROLES (``plan.a`` / ``plan.b`` / ``plan.c`` / ``plan.sfa`` / ``plan.sfb`` / ``plan.alpha`` are the roles the compiled plan
    declares), so it is taken once per plan through the compiler's own ``resolve_variant_pack`` (the two routes agree on every
    key) over a PROBE pack that binds every graph tensor to ITSELF -- ``{graph tensor: graph tensor}`` comes back as
    ``{id(bound tensor): graph tensor}``, no buffer involved -- and kept ON THE PLAN (``plan._resolved_keys``, whose lifetime is
    the JIT's own: a dropped plan takes its JIT, binding and graph tensors with it; nothing module-level holds them).  Per call
    only a dict over the pack's buffers.  A key the entry has not seen, or a JIT swapped under the plan, re-resolves.  ``None``
    for a JIT without ``run_resolved`` (the caller takes the plain ``__call__`` route).

    Never derive the roles from the BUFFERS.  The predecessor resolved the real pack and inverted it over ``id(buffer)``, which
    assumes every role holds a distinct Python object; a square projection legally binds ONE read-only rank-3 tensor as both A
    and W (``_rank3`` passes rank-3 through unchanged), the inversion kept one of the two roles, ``run_resolved`` raised
    ``KeyError: 'variant pack is missing a buffer for A operand[0]'`` -- and the two-role mapping had been cached before the
    failure, so a later call with distinct buffers on the same plan failed the same way."""
    jit = plan.jit
    if not hasattr(jit, "run_resolved") or getattr(jit, "binding", None) is None:
        return None
    entry = plan._resolved_keys
    if entry is None or entry[0] is not jit or any(id(k) not in entry[1] for k in vp):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.graph_analyzer import resolve_variant_pack

        roles = resolve_variant_pack({k: k for k in vp}, jit.binding)  # {id(bound tensor): graph tensor} -- the roles themselves
        keys = {id(k): bid for bid, k in roles.items()}
        if len(keys) != len(vp):
            # Two graph tensors of the pack named ONE bound tensor of the binding: a mapping the per-call pack cannot express.
            # Refused here, never cached as the partial mapping that would fail every later launch of this plan.
            raise RuntimeError(
                f"{plan.label}: {len(vp)} variant-pack keys resolved to {len(keys)} bound tensors of the JIT's binding; "
                "the plan's graph tensors are not one-to-one with the roles its JIT binds"
            )
        entry = plan._resolved_keys = (jit, keys)
    km = entry[1]
    return {km[id(k)]: buf for k, buf in vp.items()}


def _check_operand(plan: ProjGemmPlan, t: Optional[torch.Tensor], name: str, expect: Any) -> None:
    """``t`` carries the dtype the plan was built for, and an e2m1 side the PACKED K extent
    (``shape[-1] * 2 == K``).  A hand-built plan with no dtype (``ProjGemmPlan(graph=None, ...)``
    in the routing tests) checks nothing; so does a missing tensor -- the launch refuses those itself."""
    if t is None or expect is None:
        return
    if t.dtype != expect:
        hint = (
            " -- e2m1 codes travel as torch.float4_e2m1fn_x2: .view(torch.float4_e2m1fn_x2) the uint8 storage"
            if _is_fp4(expect) and t.dtype == torch.uint8
            else ""
        )
        raise ValueError(f"{plan.label}: {name} is {t.dtype} but this plan was built for {expect}; refusing to reinterpret the bytes{hint}")
    if _is_fp4(expect) and int(t.shape[-1]) * FP4_CODES_PER_BYTE != plan.k:
        raise ValueError(
            f"{plan.label}: {name} is fp4 storage {tuple(t.shape)} -- two e2m1 codes per byte along K -- so its last extent must be K/2 = "
            f"{plan.k // FP4_CODES_PER_BYTE}, not {int(t.shape[-1])} (a LOGICAL [.., K] fp4 tensor holds twice the data the graph declared)"
        )


def sf_padded_dims(rows: int, k: int, block: int = 32) -> tuple[int, int]:
    """``(rows_pad, sf_k_pad)`` of one F8_128x4 scale-factor tensor over ``rows`` rows and ``k``
    contraction elements at ``block`` (32, or 16 for NVFP4): whole 128-row x 4-block (512 B) atoms, i.e.
    ``(ceil(rows/128)*128, ceil((k/block)/4)*4)``.  ONE source for the graph declaration
    (``build_proj_gemm``), the blob size (``sf_blob_bytes``) and the bound view (``_sf_view``):
    the backend rejects an unpadded F8_128x4 declaration outright (``CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH``),
    and the JIT refuses a smaller blob (``compiler._lower``: ``count >= 512 * k4 * ceil(rows/128)``)."""
    if block not in SF_BLOCK_SIZES:
        raise ValueError(f"block must be one of {SF_BLOCK_SIZES}, got {block!r}")
    if rows <= 0:
        raise ValueError(f"rows must be > 0, got {rows}")
    if k <= 0 or k % block:
        raise ValueError(f"K={k} is not a positive multiple of the {block}-element block")
    return -(-rows // 128) * 128, -(-(k // block) // 4) * 4


def sf_blob_bytes(rows: int, k: int, block: int = 32) -> int:
    """Bytes of one F8_128x4 scale-factor blob = ``rows_pad * sf_k_pad`` of :func:`sf_padded_dims`
    (one E8M0 / E4M3 byte per scale, pad rows / pad blocks ``0x00``)."""
    rows_pad, sf_k_pad = sf_padded_dims(rows, k, block)
    return rows_pad * sf_k_pad


def _sf_view(plan: ProjGemmPlan, sf: Optional[torch.Tensor], name: str, rows: int) -> torch.Tensor:
    """The padded F8_128x4 blob as the rank-3 view (e8m0, or e4m3 for NVFP4) the kernel binds -- checked, never copied."""
    if sf is None:
        raise ValueError(f"{plan.label}: this plan was built with block_scale=True; pass {name}= (the F8_128x4 scale-factor blob). No silent 1.0 (Rule 1).")
    return _sf_view_of(sf, name, rows, plan.k, plan.label, block=plan.block_size, sf_torch_dtype=_sf_torch_dtype(plan.sf_dtype))


def _sf_view_of(sf: torch.Tensor, name: str, rows: int, k: int, label: str, *, block: int = 32, sf_torch_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``_sf_view`` without the plan: the same checks and the same ``(1, rows_pad, 4*k4)`` view, shared by
    the unfused block-scale driver and the fused MXFP8 runner (which keeps the E8M0 / 32 defaults).

    ``sf`` is ``uint8`` or ``sf_torch_dtype`` (``float8_e8m0fnu`` by default; ``float8_e4m3fn`` for the
    NVFP4 pair) and is RE-VIEWED as ``sf_torch_dtype`` -- the JIT binds a pointer and a byte count, so the
    view's dtype only has to be one byte wide, but it is kept honest because the graph tensor it feeds
    was DECLARED in that dtype and that declaration is what picks the MMA scale format."""
    if sf_torch_dtype is None:
        sf_torch_dtype = _E8M0
    sf_word = str(sf_torch_dtype).replace("torch.", "")
    if sf is None:
        raise ValueError(f"{label}: pass {name}= (the F8_128x4 {sf_word} scale-factor blob of the operand). No silent 1.0 (Rule 1).")
    if sf.dtype not in (torch.uint8, sf_torch_dtype):
        raise ValueError(f"{label}: {name} must be uint8 or {sf_word} ({sf_word} bytes), got {sf.dtype}")
    if not sf.is_cuda or not sf.is_contiguous() or sf.data_ptr() % 16:
        raise ValueError(f"{label}: {name} must be a contiguous, 16-B-aligned CUDA tensor (TMA-fed)")
    need = sf_blob_bytes(rows, k, block)
    if sf.numel() != need:
        raise ValueError(
            f"{label}: {name} has {sf.numel()} bytes; the F8_128x4 blob over {rows} rows x K={k} at block {block} is "
            f"{need} = ceil({rows}/128)*128 * ceil({k // block}/4)*4 (whole 512-B atoms, pad rows/blocks 0x00)"
        )
    rows_pad, sf_k_pad = sf_padded_dims(rows, k, block)
    v = sf.view(sf_torch_dtype) if (sf.dtype == torch.uint8 and sf_torch_dtype is not None) else sf  # a re-view, no copy
    return v.view(1, rows_pad, sf_k_pad)  # == the graph's declared SFA / SFB dims


def _rank3(t: torch.Tensor, name: str) -> torch.Tensor:
    """Bind the operand as the rank-3 buffer the graph declared.

    The kernel reads three axes off every operand, so a rank-2 ``[M, K]`` bind
    raises ``"expected a rank-3 buffer"``. ``unsqueeze(0)`` is always a VIEW —
    never a ``reshape`` here, which may copy, silently allocate per execute and
    (for the OUTPUT) swallow the kernel's write entirely (Rule 1).
    """
    if t.ndim == 3:
        return t
    if t.ndim == 2:
        return t.unsqueeze(0)
    raise ValueError(f"{name} must be rank 2 or 3, got shape {tuple(t.shape)}")


# ---------------------------------------------------------------------------
# The BACKWARD's projection GEMMs: transposed operands as VIEWS (api_bwd.py B1 / B2 / B7 / B8)
# ---------------------------------------------------------------------------

_A_MAJORS = ("k", "m")
_B_MAJORS = ("k", "n")


class SplitKPinRefused(RuntimeError):
    """``build_proj_gemm(split_k >= 1)``: the pinned JIT config was refused by the compiler and
    the graph-heuristic fallback -- which would change the tile config, the route and the slice
    count -- is refused too.  The compiler's own decline is the ``__cause__``."""


def _check_mn_major_tma_rule(label: str, operand: str, major: str, extent: int, axis: str, dtype: torch.dtype) -> None:
    """TMA encodes an operand's contiguous extent in 16-byte units (``compiler._tma_alignment_reject``).
    A K-major operand's K is checked where it always was (fp8 ``k % 16``; bf16 by the engine); an
    MN-major operand moves the rule onto M / N, and the driver names it BEFORE the graph exists --
    the engine's own decline would surface only as 'no frost_gemm plan' after the build."""
    if major == "k":
        return
    bpe = torch.empty((), dtype=dtype).element_size()
    elems16 = 16 // bpe
    if extent % elems16:
        raise ValueError(
            f"{label}: {operand} is {major}-major, so TMA reads its {axis} axis contiguously and needs {axis} % {elems16} == 0 "
            f"(the 16-byte contiguous-extent rule at {bpe} B/elem), got {axis}={extent}"
        )


def _declared_layout(plan: ProjGemmPlan, operand: str) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """``(dims, strides, contiguous axis)`` the plan DECLARED for ``operand`` (``"A"`` | ``"B"``)."""
    m, k, n = plan.m, plan.k, plan.n
    if operand == "A":
        return ((1, m, k), (m * k, k, 1), 2) if plan.a_major == "k" else ((1, m, k), (m * k, 1, m), 1)
    return ((1, k, n), (k * n, 1, k), 1) if plan.b_major == "k" else ((1, k, n), (k * n, n, 1), 2)


def _check_view_against_declaration(plan: ProjGemmPlan, operand: str, what: str, view: torch.Tensor) -> None:
    """The rank-3 VIEW about to be bound as ``operand`` has the declared dims and its stride-1 axis
    IS the declared major -- else a typed ``ValueError`` naming the operand and both strides.

    Why the DRIVER checks, and why the WHOLE stride tuple on BOTH routes: the graph
    fallback binds pointers against the DECLARED strides with no check -- a wrong or padded view
    there is a silent reinterpretation of the bytes.  The JIT route is not uniform either: its
    lowered call (``sm100/compiler.py`` ``_lower``, the ``renorm`` loop) re-labels B from graph
    order ``[1, K, N]`` into kernel order ONLY when the runtime ``(shape, stride)`` EQUALS the
    declared layout (``graph_order`` is None on the ``__call__`` path), so a padded N-major B is
    refused by its generic ``gave_up["input layout"]`` message, while A -- already in kernel
    order -- takes any leading dimension.  One rule the block can rely on, exactly the declared
    layout, and one typed message naming the operand and both strides.  A size-1 axis' stride is
    arbitrary in torch and is never compared (the drivers build the views from rank-2 inputs, so
    the batch stride follows from the checked axes)."""
    dims, declared, contig = _declared_layout(plan, operand)
    shape, stride = tuple(int(x) for x in view.shape), tuple(int(x) for x in view.stride())
    if shape != dims:
        raise ValueError(f"{plan.label}: {operand} ({what}) view has shape {shape}; the plan declared {operand} as dims {dims}")
    if stride[contig] != 1:
        raise ValueError(
            f"{plan.label}: {operand} ({what}) view has strides {stride}, but the plan declared {operand} "
            f"{plan.a_major if operand == 'A' else plan.b_major}-major with strides {declared}: the view's stride-1 axis is not the declared "
            f"contiguous axis {contig}. Build the plan with the matching a_major / b_major, or hand the driver the un-transposed storage"
        )
    if any(s != d for s, d, e in zip(stride, declared, shape, strict=True) if e != 1):  # all rank 3: shape == dims was checked above
        raise ValueError(
            f"{plan.label}: {operand} ({what}) view has strides {stride} but the plan declared {declared}; the drivers bind exactly the declared "
            "layout on every route -- the graph route would read a padded / strided view as the declared layout, and the JIT re-labels B into "
            "kernel order only on an exact match. Hand the driver a contiguous [rows, cols] storage (or its plain transpose), not a slice of a wider slab"
        )


def _check_output_view(plan: ProjGemmPlan, what: str, view: torch.Tensor) -> None:
    dims = (1, plan.m, plan.n)
    shape = tuple(int(x) for x in view.shape)
    if shape != dims:
        raise ValueError(f"{plan.label}: {what} view has shape {shape}; the plan declared C as dims {dims}")
    if not view.is_contiguous():
        raise ValueError(
            f"{plan.label}: {what} must be a contiguous row-major [{plan.m}, {plan.n}] (the graph declared C with strides {(plan.m * plan.n, plan.n, 1)}), got strides {tuple(view.stride())}"
        )


def _rank2(t: torch.Tensor, label: str, name: str) -> torch.Tensor:
    if t.ndim != 2:
        raise ValueError(f"{label}: {name} must be a rank-2 matrix, got shape {tuple(t.shape)}")
    return t


def run_wgrad_gemm(
    plan: ProjGemmPlan,
    dy_like: torch.Tensor,
    x: torch.Tensor,
    dw: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    *,
    stream=None,
    alpha: Optional[torch.Tensor] = None,
) -> None:
    """``dW[rows, cols] = dy_like[T, rows]^T @ x[T, cols]`` -- ``nn.Linear``'s weight gradient
    (B1 ``dW_o = dY^T @ O_gated``, B7 ``dW_qkvg = dQKVG^T @ h``).

    ``plan`` was built with ``a_major="m", b_major="n", m=rows, k=T, n=cols``.  Binds
    ``A = dy_like.unsqueeze(0).transpose(1, 2)`` (== the declared ``[1, rows, T]`` stride
    ``[T*rows, 1, rows]`` VIEW), ``B = x.unsqueeze(0)`` (== ``[1, T, cols]`` stride
    ``[T*cols, cols, 1]``) and ``C = dw.unsqueeze(0)`` -- all views, no copy, no allocation;
    :func:`run_proj_gemm`'s ``_rank3`` passes rank-3 through.  BEFORE binding it asserts
    ``plan.a_major == "m"`` and ``plan.b_major == "n"`` and that every view carries EXACTLY
    the plan's declared strides, size-1 axes aside (a typed ``ValueError`` naming the operand
    and both strides -- the graph fallback would read the declared strides with no check, and
    the JIT re-labels B into kernel order only on an exact match; a column slice of a
    wider slab is refused, not reinterpreted).
    ``stream`` / ``handle`` as :func:`run_proj_gemm` (Rule 5: one launch stream).
    ``alpha`` (appended): the per-tensor fp8 plan's ``[1, 1, 1]`` fp32 descale product --
    required when the plan was built with ``alpha=True``, refused otherwise, exactly as
    :func:`run_proj_gemm` spells it; :func:`device_alpha` writes it into the caller's slot."""
    if (plan.a_major, plan.b_major) != ("m", "n"):
        raise ValueError(
            f"{plan.label}: run_wgrad_gemm binds A = dy_like^T (M-major) and B = x (N-major), but this plan was built with "
            f"a_major={plan.a_major!r}, b_major={plan.b_major!r}; build it with a_major='m', b_major='n' (m=rows, k=T, n=cols)"
        )
    a3 = _rank2(dy_like, plan.label, "dy_like").unsqueeze(0).transpose(1, 2)  # [1, rows, T], stride [T*rows, 1, rows] for a row-major dy_like
    b3 = _rank2(x, plan.label, "x").unsqueeze(0)  # [1, T, cols]
    c3 = _rank2(dw, plan.label, "dw").unsqueeze(0)  # [1, rows, cols]
    _check_view_against_declaration(plan, "A", "dy_like^T", a3)
    _check_view_against_declaration(plan, "B", "x", b3)
    _check_output_view(plan, "dw", c3)
    run_proj_gemm(plan, a3, b3, c3, workspace, handle, alpha=alpha, stream=stream)


def run_dgrad_gemm(
    plan: ProjGemmPlan,
    dy_like: torch.Tensor,
    w: torch.Tensor,
    dx: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    *,
    stream=None,
    alpha: Optional[torch.Tensor] = None,
) -> None:
    """``dX[T, N] = dy_like[T, K] @ w[K, N]`` with ``w`` the UN-transposed row-major weight --
    ``nn.Linear``'s input gradient (B2 ``dO_gated = dY @ W_o``, B8 ``dh = dQKVG @ W_qkvg``).

    ``plan`` was built with ``a_major="k", b_major="n", m=T, k=K, n=N``.  Binds
    ``A = dy_like.unsqueeze(0)`` (``[1, T, K]`` row-major), ``B = w.unsqueeze(0)`` (== the
    declared ``[1, K, N]`` stride ``[K*N, N, 1]``: the weight exactly as the checkpoint holds
    it, ``[out_features, in_features]`` for the forward's ``x @ W^T``) and ``C = dx.unsqueeze(0)``
    -- views only, with the same declaration check as :func:`run_wgrad_gemm`.  ``alpha``
    (appended) as there: the fp8 plan's descale product, required iff ``alpha=True``."""
    if (plan.a_major, plan.b_major) != ("k", "n"):
        raise ValueError(
            f"{plan.label}: run_dgrad_gemm binds A = dy_like (K-major) and B = w (N-major, the un-transposed row-major weight), but this plan "
            f"was built with a_major={plan.a_major!r}, b_major={plan.b_major!r}; build it with a_major='k', b_major='n' (m=T, k=K, n=N)"
        )
    a3 = _rank2(dy_like, plan.label, "dy_like").unsqueeze(0)  # [1, T, K]
    b3 = _rank2(w, plan.label, "w").unsqueeze(0)  # [1, K, N]
    c3 = _rank2(dx, plan.label, "dx").unsqueeze(0)  # [1, T, N]
    _check_view_against_declaration(plan, "A", "dy_like", a3)
    _check_view_against_declaration(plan, "B", "w", b3)
    _check_output_view(plan, "dx", c3)
    run_proj_gemm(plan, a3, b3, c3, workspace, handle, alpha=alpha, stream=stream)


# ---------------------------------------------------------------------------
# The QUANTIZED backward: the device alpha slot, and the K-major block-scale drivers (B7 / B8 under MXFP8)
# ---------------------------------------------------------------------------


def _check_scalar(label: str, name: str, t: torch.Tensor, device) -> torch.Tensor:
    """``t`` is a 1-element fp32 CUDA tensor on ``device`` (the alpha slot's), else a typed ``ValueError`` naming ``name``;
    returns its ``[1]`` view -- a view, never a copy -- for the device-side multiply."""
    if not isinstance(t, torch.Tensor) or t.numel() != 1 or t.dtype != torch.float32 or not t.is_cuda:
        raise ValueError(
            f"{label}: {name} must be a 1-element fp32 CUDA tensor (a slot of the caller's workspace, or a plan-time constant), got "
            f"{type(t).__name__}{' ' + str(tuple(t.shape)) + ' ' + str(t.dtype) + ' on ' + str(t.device) if isinstance(t, torch.Tensor) else ''}"
        )
    if t.device != device:
        raise ValueError(f"{label}: {name} is on {t.device} but the alpha slot is on {device}; every scalar of one GEMM lives on the launch device")
    return t.reshape(1)  # one element is always contiguous: a view, never a copy


def device_alpha(
    alpha_out: torch.Tensor,
    descale_a: torch.Tensor,
    descale_b: torch.Tensor,
    scale_out: Optional[torch.Tensor] = None,
    *,
    stream=None,
    label: str = "proj_gemm",
) -> torch.Tensor:
    """``alpha_out[:] = descale_a * descale_b`` (``* scale_out`` when a quantized output wants it) ON THE DEVICE,
    on the launch stream; returns the ``[1, 1, 1]`` fp32 view :func:`run_proj_gemm` binds as the fp8 plan's
    ``alpha`` epilogue.

    The contract of the quantized backward's descale products (B1 ``descale_dY * (1 / scale_o)``, B2
    ``descale_dY * descale_w_o``, B7 ``descale_dQKVG * descale_h``, B8 ``descale_dQKVG * descale_w_qkvg``):

    * ``alpha_out`` is the CALLER's slot -- one fp32 element of a workspace region the caller carved at plan
      time (a view into it is fine; the write is in place and the returned view aliases it), so nothing is
      allocated per execute (Rule 1) and a CUDA-graph capture sees stable pointers;
    * every input is a 1-element fp32 CUDA tensor on the slot's device -- a plan-time constant or another
      slot a quantize kernel wrote this step (the "current" recipe's device-side scales) -- and is READ on
      the device: no ``.item()``, no host round trip, no host-side float ever reaches the product;
    * the two multiplies run under ``stream`` (a raw ``CUstream`` int or a ``torch.cuda.Stream``; ``None`` =
      torch's current stream on the slot's device) -- the SAME launch stream the GEMM takes (Rule 5), so a
      scale written by an earlier kernel on that stream is ordered before the product, and the product
      before the GEMM that reads it.

    Numerics: one fp32 product (two with ``scale_out``), rounded to nearest -- the same value a host-side
    ``descale_a * descale_b`` in fp32 would give, so the oracle may compute it either way.
    """
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import stream_context

    if not isinstance(alpha_out, torch.Tensor) or alpha_out.numel() != 1 or alpha_out.dtype != torch.float32 or not alpha_out.is_cuda:
        raise ValueError(
            f"{label}: alpha_out must be a 1-element fp32 CUDA tensor -- the caller's workspace slot the GEMM's alpha epilogue reads -- got "
            f"{type(alpha_out).__name__}{' ' + str(tuple(alpha_out.shape)) + ' ' + str(alpha_out.dtype) + ' on ' + str(alpha_out.device) if isinstance(alpha_out, torch.Tensor) else ''}"
        )
    dev = alpha_out.device
    out1 = alpha_out.reshape(1)
    a1 = _check_scalar(label, "descale_a", descale_a, dev)
    b1 = _check_scalar(label, "descale_b", descale_b, dev)
    s1 = _check_scalar(label, "scale_out", scale_out, dev) if scale_out is not None else None
    with stream_context(stream, dev):
        torch.mul(a1, b1, out=out1)
        if s1 is not None:
            out1.mul_(s1)
    return alpha_out.reshape(1, 1, 1)


def _check_k_major_block_scale_plan(plan: ProjGemmPlan, driver: str) -> None:
    """The block-scale drivers' plan gate: ``plan`` was built with ``block_scale=True`` at the K-major defaults, else a
    typed ``ValueError`` naming ``driver`` (a dense plan has no dequant; the block-scale rows take no other major)."""
    if not plan.block_scale:
        raise ValueError(f"{plan.label}: {driver} serves block-scale plans (build_proj_gemm(block_scale=True)); this plan has no block-scale dequant")
    if (plan.a_major, plan.b_major) != ("k", "k"):
        raise ValueError(
            f"{plan.label}: {driver} binds K-major TRANSPOSED operands (the block-scale rows take no other major), but this plan was built with "
            f"a_major={plan.a_major!r}, b_major={plan.b_major!r}; build it with the K-major defaults"
        )


def _check_k_major_operand(plan: ProjGemmPlan, what: str, t: torch.Tensor, rows: int, k: int, expect: Any = None) -> torch.Tensor:
    """``t`` is the contiguous row-major ``[rows, k]`` storage the plan declared (a transposed artifact as the caller
    STORES it -- never ``.t()`` of a row-major ``[k, rows]``, whose stride-1 axis would be the wrong one).

    ``expect`` (appended; ``None`` = no dtype check and the LOGICAL extent, as a hand-built plan without dtypes declares it): the
    dtype this side was built for.  The dtype is checked FIRST, by :func:`_check_operand` -- a mismatch is its typed
    refusal, with the ``.view(torch.float4_e2m1fn_x2)`` hint for uint8 codes handed to an e2m1 side, and a LOGICAL ``[rows, k]``
    fp4 tensor its "last extent must be K/2" refusal -- and an e2m1 ``expect`` makes the declared storage the PACKED
    ``[rows, k // 2]`` (:func:`storage_k`: two codes per byte along K, low nibble = even k), so the shape and the strides are
    checked against the bytes the caller holds: the ``_check_operand`` rule of :func:`run_proj_gemm`, applied to the K-major extent
    and the row stride here, for either operand of either block-scale driver (both are packed under the NVFP4 x NVFP4 row)."""
    t = _rank2(t, plan.label, what)
    if expect is not None:
        _check_operand(plan, t, what, expect)  # the dtype (+ the view hint) and, for e2m1 codes, ``shape[-1] * 2 == K``
    k_store = storage_k(k, expect)
    packed = f" -- the PACKED e2m1 storage, {FP4_CODES_PER_BYTE} codes per byte along K={k}" if _is_fp4(expect) else ""
    shape, stride = tuple(int(x) for x in t.shape), tuple(int(x) for x in t.stride())
    if shape != (rows, k_store):
        raise ValueError(f"{plan.label}: {what} has shape {shape}; this plan declared it as the K-major [{rows}, {k_store}] (rows x K, K contiguous{packed})")
    if stride != (k_store, 1):
        raise ValueError(
            f"{plan.label}: {what} has strides {stride} but this plan declared a contiguous row-major [{rows}, {k_store}] (strides {(k_store, 1)}{packed}): "
            "the block-scale rows read K-major operands whose F8_128x4 scale factors run along K -- hand the driver the transposed storage itself, "
            "not a view of the un-transposed tensor and not a slice of a wider slab"
        )
    return t


def _check_driver_sf_blob(plan: ProjGemmPlan, name: str, sf: Optional[torch.Tensor], rows: int, of_what: str) -> None:
    """The block-scale drivers' scale-factor gate, by the DRIVER's keyword: a missing ``sf_dy_t`` / ``sf_x_t`` /
    ``sf_dy`` / ``sf_w_t`` is named as such (``run_proj_gemm``'s own gate would name its ``sf_a`` / ``sf_w``), with
    the operand it scales and the byte count it needs; dtype, device, alignment and size are ``_sf_view``'s checks
    under the same name.  ``run_proj_gemm`` re-derives the bound views when it launches (views, never copies)."""
    if sf is None:
        raise ValueError(
            f"{plan.label}: this plan was built with block_scale=True; pass {name}= (the padded F8_128x4 scale-factor blob of {of_what}, "
            f"{sf_blob_bytes(rows, plan.k, plan.block_size)} bytes for {rows} rows x K={plan.k}). No silent 1.0 (Rule 1)."
        )
    _sf_view(plan, sf, name, rows)


def run_wgrad_gemm_block_scale(
    plan: ProjGemmPlan,
    dy_t: torch.Tensor,
    x_t: torch.Tensor,
    dw: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    *,
    sf_dy_t: torch.Tensor,
    sf_x_t: torch.Tensor,
    stream=None,
) -> None:
    """``dW[rows, cols] = dy_t[rows, T] @ x_t[cols, T]^T`` -- ``nn.Linear``'s weight gradient over block-scaled operands
    that are BOTH K-major transposed artifacts (B7 ``dW_qkvg = dQKVG^T @ h``: ``dy_t`` is the quantizer's transposed
    ``dQKVG^T [N, T]`` with its scale factors along T, ``x_t`` the caller's TE "columnwise" ``h^T [dm, T]`` with its
    ``h_t_sf``).

    ``plan`` was built with ``block_scale=True`` and the K-major defaults: ``m=rows, k=T, n=cols`` -- the forward's own
    declaration (``A [m, k]`` row-major, ``W [n, k]`` row-major read transposed) at the backward's shapes, so the binding
    is the forward's proven one (:func:`run_proj_gemm` with ``a=dy_t, w=x_t``), and the contraction over tokens needs
    ``T % 32 == 0`` (whole scale blocks along K -- ``build_proj_gemm`` declines a ragged T at plan time, typed).  The rows
    this driver binds are :func:`block_scale_pairing`'s: the block's weight gradients run the **MXFP8 x MXFP8** row (e4m3
    codes on both sides, one E8M0 scale per 32 tokens) under every quantized mode -- the fp4 weight modes keep their weight
    gradients 8-bit -- and an e2m1 side (the **mixed** e4m3 x e2m1 row at E8M0 per 32, the **NVFP4 x NVFP4** row at e4m3
    per 16) takes the same binding under the packed-storage rule below.  ``sf_dy_t`` / ``sf_x_t`` are the PADDED F8_128x4
    blobs over ``rows`` x ``T`` and ``cols`` x ``T`` at its ``block_size`` (``sf_blob_bytes``), both required (Rule
    1: no silent unit scale) and checked HERE under these keywords (a missing or wrong-sized blob is a ``ValueError``
    naming ``sf_dy_t`` / ``sf_x_t``, the operand it scales and the byte count).  Every operand is checked against the
    declaration BEFORE the launch: its dtype (its ``dtype`` / ``w_dtype``; uint8 bytes handed to an e2m1 side get
    the ``.view(torch.float4_e2m1fn_x2)`` hint), contiguous row-major ``[rows, T]`` / ``[cols, T]`` storage -- for an e2m1
    side the PACKED ``[.., T // 2]`` (two codes per byte along K, low nibble = even k; a LOGICAL ``[.., T]`` fp4 tensor is
    refused as holding twice the data) -- where a ``.t()`` view of the un-transposed tensor, or a slice of a wider slab, is
    a typed ``ValueError`` naming the operand and both strides, and a contiguous ``[rows, cols]`` output.  There is no
    ``alpha`` (the block dequant is exact and happens in the MMA) and no ``split_k`` (refused at plan time)."""
    _check_k_major_block_scale_plan(plan, "run_wgrad_gemm_block_scale")
    a = _check_k_major_operand(plan, "dy_t (dy_like^T, [rows, T])", dy_t, plan.m, plan.k, expect=plan.dtype)
    w = _check_k_major_operand(plan, "x_t (x^T, [cols, T])", x_t, plan.n, plan.k, expect=_w_dtype_of(plan))
    _check_output_view(plan, "dw", _rank2(dw, plan.label, "dw").unsqueeze(0))
    _check_driver_sf_blob(plan, "sf_dy_t", sf_dy_t, plan.m, "dy_t over its rows x T")
    _check_driver_sf_blob(plan, "sf_x_t", sf_x_t, plan.n, "x_t over its cols x T")
    run_proj_gemm(plan, a, w, dw, workspace, handle, sf_a=sf_dy_t, sf_w=sf_x_t, stream=stream)


def run_dgrad_gemm_block_scale(
    plan: ProjGemmPlan,
    dy_like: torch.Tensor,
    w_t: torch.Tensor,
    dx: torch.Tensor,
    workspace: torch.Tensor,
    handle: Optional[Any] = None,
    *,
    sf_dy: torch.Tensor,
    sf_w_t: torch.Tensor,
    stream=None,
) -> None:
    """``dX[T, N] = dy_like[T, K] @ w_t[N, K]^T`` -- ``nn.Linear``'s input gradient over block-scaled operands with the
    weight TRANSPOSED into K-major storage (``w_t`` = ``W^T`` re-quantized along the contraction axis by the caller, once
    per weight update, with its blob).  Three rows, each the forward's own rendering at the dgrad's ``(m, k, n)``
    (:func:`block_scale_pairing`):

    * **MXFP8 x MXFP8** (``dtype = w_dtype = e4m3``, E8M0 per 32): B8 ``dh = dQKVG @ W_qkvg`` -- ``dy_like`` the rowwise
      ``dQKVG [T, N]`` with its scale factors along N, ``w_t`` the caller's ``W_qkvg^T [dm, N]`` with ``w_qkvg_t_sf``;
    * **the mixed row** (``dtype = e4m3``, ``w_dtype = torch.float4_e2m1fn_x2``, E8M0 per 32 on both sides): B8 under an
      MXFP4 ``W_qkvg`` -- ``w_t`` the e2m1 ``W_qkvg^T`` stored PACKED ``[dm, N // 2]`` with its E8M0 blob over ``(dm, N)`` --
      and B2 ``dO_gated = dY @ W_o`` under an MXFP4 ``W_o`` -- ``dy_like`` an MX-rowwise e4m3 ``dY [T, dm]``, ``w_t`` the e2m1
      ``W_o^T`` packed ``[HD, dm // 2]``;
    * **NVFP4 x NVFP4** (``dtype = w_dtype = torch.float4_e2m1fn_x2``, ``block_size = 16``, e4m3 scales): B2 under an NVFP4
      ``W_o`` -- ``dy_like`` the NVFP4 cast of ``dY`` packed ``[T, dm // 2]`` with its e4m3 blob over ``(T, dm)``, ``w_t`` the
      e2m1 ``W_o^T`` packed ``[HD, dm // 2]`` with its e4m3 blob over ``(HD, dm)`` -- BOTH operands packed.

    ``plan``: ``block_scale=True``, K-major defaults, ``m=T, k=K, n=N`` -- again the forward's declaration, so ``w_t`` is
    bound exactly as the forward binds its ``[N, K]`` weight.  ``sf_dy`` / ``sf_w_t``: the PADDED F8_128x4 blobs over ``T``
    x ``K`` and ``N`` x ``K`` at its ``block_size`` (16 for NVFP4), checked here under these keywords.  Operand
    checks as :func:`run_wgrad_gemm_block_scale`: the dtype per side (uint8 codes handed to an e2m1 side get the
    ``.view(torch.float4_e2m1fn_x2)`` hint), contiguous row-major storage at the DECLARED extent -- ``[T, K]`` / ``[N, K]``,
    or the PACKED ``[.., K // 2]`` for an e2m1 side (a LOGICAL ``[.., K]`` fp4 tensor is a typed refusal: twice the data) --
    a ``.t()`` view or a slab slice refused by name with both strides, a contiguous ``[T, N]`` output; no ``alpha``, no
    ``split_k``."""
    _check_k_major_block_scale_plan(plan, "run_dgrad_gemm_block_scale")
    a = _check_k_major_operand(plan, "dy_like ([T, K])", dy_like, plan.m, plan.k, expect=plan.dtype)
    w = _check_k_major_operand(plan, "w_t (w^T, [N, K])", w_t, plan.n, plan.k, expect=_w_dtype_of(plan))
    _check_output_view(plan, "dx", _rank2(dx, plan.label, "dx").unsqueeze(0))
    _check_driver_sf_blob(plan, "sf_dy", sf_dy, plan.m, "dy_like over its T rows x K")
    _check_driver_sf_blob(plan, "sf_w_t", sf_w_t, plan.n, "w_t over its N rows x K")
    run_proj_gemm(plan, a, w, dx, workspace, handle, sf_a=sf_dy, sf_w=sf_w_t, stream=stream)
