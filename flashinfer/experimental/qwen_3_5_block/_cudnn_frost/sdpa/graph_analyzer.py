# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Engine-agnostic SDPA graph analysis: ``graph.nodes`` -> :class:`SdpaGraphFacts`.

``cudnn.pygraph`` is the Python-native graph IR: it records its op DAG
directly, exposed via ``graph.nodes`` — no construction-time hooks are needed.
This module extracts *facts* (what the graph asks for) without judging
supportedness; each registered engine matches the facts against its own
capability declaration (see ``cudnn.sdpa.fwd.engines``). :func:`analyze` is the
callable the SDPA family names in ``engines/manifest.py``; PLANNING runs it once
per graph -- after the backend's layout inference has landed and the graph is
frozen -- and attaches the record, so the ranking and the engine share it rather
than each parsing the graph.

Also hosts the graph-side runtime helpers shared by every SDPA engine:
variant-pack resolution and TensorDesc construction.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._sdpa_tail import GateTail, match_gate_tail

_LOG = logging.getLogger(__name__)


# The facts vocabulary is cudnn.data_type, not torch.dtype. Facts are what every
# engine of a family reads, so expressing them in one framework's types would
# make the whole dispatch path require that framework; lowering converts at its
# own boundary, where a torch tensor is actually being allocated.
_TORCH_FROM_CUDNN = {
    cudnn.data_type.HALF: "float16",
    cudnn.data_type.BFLOAT16: "bfloat16",
    cudnn.data_type.FP8_E4M3: "float8_e4m3fn",
    cudnn.data_type.FP8_E5M2: "float8_e5m2",
    cudnn.data_type.FLOAT: "float32",
    cudnn.data_type.INT32: "int32",
    # Block-scaled O of the per-tensor FP8 forward: two E2M1 per byte, and the
    # UE8M0 scale factors of its MXFP8 output.
    cudnn.data_type.FP4_E2M1: "float4_e2m1fn_x2",
    cudnn.data_type.FP8_E8M0: "float8_e8m0fnu",
}
_KNOWN_DTYPES = frozenset(_TORCH_FROM_CUDNN)


def _torch_cuda_device():
    """The lowering boundary's torch device — imported here, not at module level."""
    import torch

    return torch.device("cuda", torch.cuda.current_device())


def to_torch_dtype(dt):
    """cudnn.data_type -> torch.dtype, for the lowering boundary only.

    Declines rather than raising KeyError on a type this family has no mapping
    for: only Q's dtype is capability-checked, so O / Stats / a side output can
    still carry one, and tensor_desc_from_ir() runs on every bound tensor.
    """
    import torch

    if dt not in _TORCH_FROM_CUDNN:
        raise NotImplementedError(f"cudnn.sdpa: no lowering for data type {dt}")
    name = _TORCH_FROM_CUDNN[dt]
    dtype = getattr(torch, name, None)
    if dtype is None:
        # The packed FP4 dtype arrived in torch 2.8: on an older build a graph
        # that declares it is DECLINED, like any type without a lowering.
        raise NotImplementedError(f"cudnn.sdpa: this torch build has no torch.{name} for data type {dt}")
    return dtype


# BHSD logical / BSHD physical, size-1 dims wildcarded.
_REQ_STRIDE_ORDER = (3, 1, 2, 0)


def _device_props():
    """The current device as cuDNN describes it, or None without a device.

    cudnn.create_device_properties() is the backend's OWN device descriptor —
    the same object the C++ deviceless-AoT path serializes and replays. Reading
    it here instead of torch.cuda keeps the facts path framework-neutral, and
    is the step that makes a deviceless python engine possible at all: facts
    computed from a serialized descriptor need no live device.

    Cached per device id: the descriptor costs a backend call and the device
    does not change under a process.
    """
    import json

    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import current_device_id

    dev = current_device_id()
    if dev is None:
        return None
    cached = _DEVICE_PROPS_CACHE.get(dev)
    if cached is None:
        try:
            cached = json.loads(cudnn.create_device_properties(dev).serialize())
        except Exception:  # noqa: BLE001 — no device / older backend: facts stay device-less
            cached = {}
        _DEVICE_PROPS_CACHE[dev] = cached
    return cached or None


_DEVICE_PROPS_CACHE: dict = {}


def _device_cc() -> Optional[tuple]:
    """Compute capability of the current CUDA device, or None without CUDA."""
    props = _device_props()
    if not props or "deviceVer" not in props:
        return None
    ver = int(props["deviceVer"])  # 1000 == SM 10.0
    return (ver // 100, (ver % 100) // 10)


def _device_sm_count() -> Optional[int]:
    """SM count of the current CUDA device, or None without CUDA."""
    props = _device_props()
    return int(props["multiProcessorCount"]) if props and "multiProcessorCount" in props else None


def _stride_order(dim: tuple, stride: tuple) -> tuple:
    return tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], dim[x[0]])))


def thd_stats_packing(stride_h: int, stride_s: int, h_q: int) -> Optional[str]:
    """Classify a ragged (packed) Stats declaration by its strides (Rule S1).

    ``"token_major"``: ``(stride_h, stride_s) == (1, H_q)`` -- cuDNN's ragged
    Stats recipe, ``(T, H)`` in memory.  ``"head_major"``: ``stride_s == 1``
    with ``stride_h >= 1`` the declared head stride, ``(H, head_stride)`` in
    memory -- what the FROST forwards emit, rounded up to a 64-token capacity.
    ``None`` is a packing no FROST row reads.  The head stride must cover the
    packed token total; that total is a device value, so callers can only
    bound it against a plan-time capacity (see bwd/engines.py mismatch()).

    One classifier for every probe and adapter site, so the two packings
    cannot drift apart between the forward that writes Stats and the
    backward that reads it.
    """
    if (int(stride_h), int(stride_s)) == (1, int(h_q)):
        return "token_major"
    if int(stride_s) == 1 and int(stride_h) >= 1:
        return "head_major"
    return None


def bshd_layout_ok(dim: tuple, stride: tuple) -> bool:
    """Strict BSHD physical stride order for a rank-4 (B, H, S, D) tensor:
    axes sorted by ascending stride must come out D, H, S, B, with size-1
    dims wildcarded. Order only — padded (oversized) strides are allowed;
    compactness and aliasing are dense_layout_ok's concern. Used by the
    facts extraction below and by engine gates that require true-BSHD views
    (e.g. THD ragged, where the kernel addresses packed rows directly)."""
    order = _stride_order(dim, stride)
    act = tuple(ax for ax in order if dim[ax] != 1)
    exp = tuple(ax for ax in _REQ_STRIDE_ORDER if dim[ax] != 1)
    return act == exp


def packed_layout_ok(dim: tuple, stride: tuple) -> bool:
    """BSHD order over (H, S, D) only, for a ragged (THD) tensor. With ragged
    offsets every sequence's base comes from the offset table and the batch
    axis is never stepped (the THD lowering binds it at extent 1), so its
    declared stride carries no information -- FlashInfer declares it equal to
    the token stride. The per-token layout the kernels address natively is
    what has to hold: D innermost, then H, then S."""
    order = tuple(ax for ax in _stride_order(dim, stride) if ax != 0)
    act = tuple(ax for ax in order if dim[ax] != 1)
    exp = tuple(ax for ax in _REQ_STRIDE_ORDER if ax != 0 and dim[ax] != 1)
    return act == exp


def dense_layout_ok(dim: tuple, stride: tuple) -> bool:
    """Relaxed DENSE layout soundness for a rank-4 (B, H, S, D) tensor: the
    real requirement of the SM100 DSL lowering, which normalizes any such
    layout to the kernel's canonical BSHD-compact buffers (zero-copy when the
    tensor already is BSHD-compact, one gather/scatter copy otherwise) — so
    the B/H/S physical stride ORDER is free, and strides may be padded
    (oversized) arbitrarily. What must hold:

      * head dim (axis 3) innermost-contiguous: stride 1 (size-1 wildcarded);
      * no zero stride on a size>1 dim — a broadcast view aliases distinct
        logical elements onto one address, which is ill-defined for the O
        write-back and rejected uniformly;
      * non-overlapping: visiting dims by ascending stride, each stride must
        cover the span of every dim below it (dense == equality; padded is
        larger; SMALLER than dense would alias, so it is rejected).

    Size-1 dims are wildcards: any stride, 0 and garbage included (cuDNN's
    own harnesses corrupt them deliberately).
    """
    if dim[3] != 1 and stride[3] != 1:
        return False
    active = sorted((int(st), int(sz)) for st, sz in zip(stride, dim) if sz != 1)
    span = 1
    for st, sz in active:
        if st < span:  # st == 0 (broadcast) or overlapping / sub-dense
            return False
        span = st * sz
    return True


# Element widths of the facts vocabulary, for the 16-byte TMA stride rule below.
# An unmapped dtype gets the STRICTEST width (1 B -> strides must be multiples
# of 16 elements); the row's dtype check declines such a G before layout anyway.
_ELEM_BYTES = {
    cudnn.data_type.FLOAT: 4,
    cudnn.data_type.INT32: 4,
    cudnn.data_type.HALF: 2,
    cudnn.data_type.BFLOAT16: 2,
    cudnn.data_type.FP8_E4M3: 1,
    cudnn.data_type.FP8_E5M2: 1,
}


def gate_layout_ok(dim: tuple, stride: tuple, elem_bytes: int) -> bool:
    """Whether a rank-4 logical (B, H, S, D) tensor is one the fused-gate kernels
    can TMA-load ZERO-COPY: BSHD-compact, or a declared BSHD layout TMA can
    express. Uses the shared ``config_sm100.bshd_zero_copy_stride`` predicate,
    distinguishing compact from inexpressible geometry so the row declines
    what the adapter would reject:

      * head dim innermost-contiguous (stride 1);
      * every stepped batch, seq and head stride is a 16-byte multiple;
      * the declaration COVERS the tensor (head >= d, seq >= h*head,
        batch >= s*seq) -- an overlapping declaration aliases distinct rows.

    Stricter than ``bshd_layout_ok`` + ``dense_layout_ok`` on purpose: size-1
    dims are NOT wildcarded, because the adapter's TensorDesc keeps the IR's
    unit-dim strides verbatim and its cover check reads them.  Unlike Q/K/V/O,
    G has no normalisation-copy fallback in the lowering.
    """
    if len(dim) != 4 or len(stride) != 4:
        return False
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import bshd_compact, bshd_zero_copy_stride

    dim, stride = tuple(int(x) for x in dim), tuple(int(x) for x in stride)
    return bshd_compact(dim, stride) or bshd_zero_copy_stride(dim, stride, max(1, int(elem_bytes))) is not None


@dataclass(frozen=True)
class SdpaGraphFacts:
    """What a single-SDPA graph asks for. Pure description — no support judgment.

    Why extraction never judges: supportedness is per-engine knowledge (one
    kernel has paged-KV, another has sinks). The moment a shared parser
    starts rejecting, it becomes an if-ladder that must know every kernel —
    which stops scaling at the second kernel. So this record only DESCRIBES
    (``has_bias=True`` is a fact, not an error) and each engine's
    Capabilities row does the judging in ``fwd/engines.mismatch``.

    ``invalid`` is the one exception: a graph-consistency error (malformed
    regardless of which kernel would run — K/V shape mismatch, padding mask
    without seq_len_kv, ...); when set, every engine is ineligible.
    """

    invalid: Optional[str] = None

    # geometry
    b: int = 0
    h_q: int = 0
    h_kv: int = 0
    s_q: int = 0
    s_kv: int = 0
    d_qk: int = 0
    d_v: int = 0

    # dtypes / layout
    dtype: Optional[Any] = None  # Q dtype as cudnn.data_type (None if unrecognized)
    uniform_dtype: bool = True  # K/V (and O, for half; O/dO too on the per-tensor FP8 backward) dtypes equal Q's
    # Quantized graphs only: the half-precision operands (O; and dO_f16 / dQ /
    # dK / dV on the MXFP8 backward) share ``dtype_o``. Always True on half
    # graphs, where ``uniform_dtype`` already covers them.  On the per-tensor
    # FP8 backward the non-payload side is the GRADIENT triple: dK / dV share
    # dQ's dtype (O rides the payload set, see ``uniform_dtype``).
    uniform_out_dtype: bool = True
    bshd_layout: bool = True  # all of Q/K/V/O in BSHD-physical order
    # BSHD order over (H, S, D) only -- what a ragged (THD) tensor has to
    # satisfy, its batch stride being unread under ragged offsets (packed_layout_ok).
    packed_layout: bool = True
    # Relaxed dense-layout soundness: every one of Q/K/V/O has the head dim
    # innermost-contiguous (stride 1) with non-broadcast, non-overlapping
    # strides — any B/H/S order, padded strides allowed (see dense_layout_ok).
    # The actual per-tensor stride tuples stay available via q_t/k_t/v_t/o_t.
    dense_layout: bool = True
    # Backward only: (name, dim, stride) per rank-4 layout port, with the
    # K/V transposed-port rewrite undone; () on forward graphs.
    port_layouts: tuple = ()
    is_mxfp8: bool = False  # block-scale MXFP8 (FP8 Q/K/V + per-32-block E8M0 SF)
    is_fp8: bool = False  # per-tensor FP8 (FP8 Q/K/V + scalar descales; sdpa_fp8 / sdpa_fp8_backward)
    # O dtype as cudnn.data_type -- the non-payload side of a quantized graph
    # (what a row's ``out_dtypes`` gates).  On the per-tensor FP8 BACKWARD that
    # side is the gradients, so ``dtype_o`` is dQ's dtype there: O is an FP8
    # payload with its own descale (``descale_o_t``) and belongs to the payload
    # set, while dQ / dK / dV carry ``scale_dQ/dK/dV`` and may be FP8 (the
    # contract) or half (what the backend allows on Blackwell).
    dtype_o: Optional[Any] = None
    # Block-scaled O (sdpa_fp8 with an ``sf_o`` output): scale-factor block
    # along d_v — 16 = E2M1 O + E4M3 SF, 32 = E4M3 O + UE8M0 SF, 0 = plain O.
    o_block_scale: int = 0

    # masks (resolved cuDNN semantics)
    causal: bool = False  # effective causal upper bound (right band == 0)
    bottom_right: bool = False  # diagonal anchored bottom-right
    window_left: Optional[int] = None  # SWA offset W (cuDNN length - 1)
    right_band_widening: bool = False  # diagonal_band_right_bound > 0

    # requested features
    has_bias: bool = False
    has_dropout: bool = False
    has_score_mod: bool = False
    has_paged_kv: bool = False
    # Paged KV (paged_attention_k_table / v_table): K/V are page pools
    # [num_pages, H_kv, page_size, D]; ``s_kv`` is the declared
    # paged_attention_max_seq_len_kv (else what the block tables address). The
    # in-page layout (HND/NHD) is only the pools' strides, and the tables'
    # extents live on their IR refs — neither is a separate fact.
    page_size: int = 0
    has_alibi: bool = False
    has_unfuse_fma: bool = False
    # sdpa(stats_use_log2=True): Stats requested as (max + ln(sum_exp)) * log2(e) instead of
    # the natural-log form. A convention on the Stats output, not a math change;
    # an engine that writes natural-log stats must decline, not ignore it.
    has_stats_log2: bool = False
    has_block_mask: bool = False
    has_rng_dump: bool = False
    # sdpa_backward() / sdpa_fp8_backward() / sdpa_mxfp8_backward() node
    # (NodeType.SDPA_BWD / SDPA_FP8_BWD / SDPA_MXFP8_BWD).
    is_backward: bool = False
    # Quantized backward: any of amax_dQ / amax_dK / amax_dV (and, on the
    # per-tensor FP8 backward, amax_dP) requested as a real output. The op
    # returns the ports unconditionally; only set_output(True) ones count (same
    # convention as amax_s_t on the FP8 forward).
    has_amax_dgrad: bool = False
    right_bound: Optional[int] = None  # raw resolved right band (0 == causal)
    deterministic: bool = False  # sdpa_backward(use_deterministic_algorithm=True)
    has_dbias: bool = False  # dBias output requested (backward)
    has_dsink: bool = False  # dSink_token output requested (backward)
    has_score_max: bool = False  # per-row/tile score-max side output requested
    has_score_sum_exp: bool = False  # per-row/tile sum-of-exp side output requested
    dynamic_scale: bool = False  # attn_scale passed as a tensor
    seq_q_trim: bool = False  # seq_len_q given without a padding mask

    padded: bool = False  # per-batch KV lengths present (padding mask or THD)
    thd: bool = False  # ragged (THD) Q/K/V
    # Caller-declared packed token totals (sdpa(max_total_seq_len_q/kv=...)).
    # Ragged graphs describe extents as (B, H, S_max, D) plus device ragged
    # offsets, so the packed total is not otherwise expressible; when supplied
    # it bounds the token axis EXACTLY instead of being inferred from buffer
    # geometry. None = not declared (infer).
    max_total_seq_len_q: Optional[int] = None
    max_total_seq_len_kv: Optional[int] = None
    # cu_seq_len_q / cu_seq_len_kv (cuDNN 9.24+): (B+1,) prefix sums, a
    # contract of their own — neither seq_len_* nor ragged_offset. A fact,
    # not a verdict: engines that don't consume the form must decline (reading
    # the graph as plain padded gave wrong output — 14.9% of O on
    # test_sdpa_mixed_seq_len_forms_L0[cu_q_brcm]).
    has_cu_seq_len: bool = False
    has_sink: bool = False
    wants_stats: bool = False

    scale: Optional[float] = None
    device_cc: Optional[tuple] = None  # current CUDA device capability
    device_sm_count: Optional[int] = None  # current CUDA device SM count

    # IR tensor refs for binding
    q_t: Any = None
    k_t: Any = None
    v_t: Any = None
    o_t: Any = None
    stats_t: Any = None
    sink_t: Any = None
    seq_kv_t: Any = None
    seq_q_t: Any = None
    # Paged KV block tables, (B, 1, max_pages, 1) int32; when the graph declares
    # only one of the pair both refs point at it.
    paged_k_table_t: Any = None
    paged_v_table_t: Any = None
    # (B+1,) prefix-sum IR refs (cuDNN 9.24+ cu_seq_len form); None when the
    # graph carries the per-batch seq_len_* form on that side instead.
    cu_seq_q_t: Any = None
    cu_seq_kv_t: Any = None
    # Feature operands (bias / block-mask / score-stat outputs).
    bias_t: Any = None
    block_mask_t: Any = None
    score_max_t: Any = None
    score_sum_exp_t: Any = None
    # Backward-only refs.
    do_t: Any = None
    dq_t: Any = None
    dk_t: Any = None
    dv_t: Any = None
    dbias_t: Any = None
    dsink_t: Any = None
    # MXFP8 backward-only refs: the transposed-quantization payloads (each
    # contraction axis needs its own 1x32 quantization), the half-precision dO,
    # their block-scale tensors, and the (declined-by-default) amax outputs.
    q_T_t: Any = None
    k_T_t: Any = None
    dO_T_t: Any = None
    dO_f16_t: Any = None
    sf_q_T_t: Any = None
    sf_k_T_t: Any = None
    sf_dO_t: Any = None
    sf_dO_T_t: Any = None
    amax_dq_t: Any = None
    amax_dk_t: Any = None
    amax_dv_t: Any = None
    # Per-tensor FP8 backward (sdpa_fp8_backward) contract: the scalar descales
    # of the forward's O, of dO and of the intermediate dP, the scales the FP8
    # gradients (dQ / dK / dV) and dP are quantized with, and the
    # requested-only Amax_dP output.  descale_q/k/v/s and scale_s below are
    # shared with the FP8 forward; amax_dQ/dK/dV above with the MXFP8 backward.
    descale_o_t: Any = None
    descale_do_t: Any = None
    descale_dp_t: Any = None
    scale_dq_t: Any = None
    scale_dk_t: Any = None
    scale_dv_t: Any = None
    scale_dp_t: Any = None
    amax_dp_t: Any = None
    # MXFP8 block-scale (descale) tensors + Amax_O output.
    sf_q_t: Any = None
    sf_k_t: Any = None
    sf_v_t: Any = None
    amax_o_t: Any = None
    # Block-scaled O scale-factor output (sdpa_fp8 ``sf_o``; see o_block_scale).
    sf_o_t: Any = None
    # Per-tensor FP8 scalar descale tensors + Amax_S output.
    descale_q_t: Any = None
    descale_k_t: Any = None
    descale_v_t: Any = None
    scale_o_t: Any = None
    # cuDNN's Scale_S/Descale_S: they quantize P, the softmax OUTPUT. SM120
    # applies them; SM100 converts P unscaled (exact for a reciprocal pair) and
    # declines a non-reciprocal one (api_dsl._require_reciprocal_s_scales).
    descale_s_t: Any = None
    scale_s_t: Any = None
    amax_s_t: Any = None
    # Softmax accumulation precision the GRAPH asks for, as a cudnn.data_type:
    # HALF when the graph's intermediate_data_type is HALF (the f16x2 exponent
    # arm; numerics-changing, so it is a graph attribute and never a tuning
    # knob), None otherwise (= the f32 pipeline every row runs). Engines whose
    # capability row does not list the requested precision decline.
    softmax_precision: Optional[Any] = None
    # The GRAPH's statement that Q already carries attn_scale * log2(e) (the
    # python-only op attribute sdpa(attn_scale_prefolded=True)): the engine
    # applies no softmax scale and the kernel's pre-folded arm runs exp2(S - m)
    # on the raw QK^T.  A fact the capability rows gate on
    # (attn_scale_prefolded_d_shapes); attn_scale must be unset alongside it.
    attn_scale_prefolded: bool = False

    # Epilogue gate: the three-node tail ``sdpa(virtual O_v) -> sigmoid(G) ->
    # mul(O_v, s)`` (cudnn._sdpa_tail.match_gate_tail).  A FACT, not a verdict:
    # the Rubin d256 kernels fuse it into their epilogue (TemplateParams.
    # epilogue_gate) and their rows claim it through Capabilities.epilogue_gate*;
    # every other row declines in mismatch().  When set, ``o_t`` is the MUL
    # output (the graph's real O the fused kernel writes) and the virtual O_v /
    # s are never bound.
    has_epilogue_gate: bool = False
    epilogue_gate_t: Any = None  # G: a graph INPUT, dims == O's (B, H_q, S_q, D_v)
    epilogue_gate_dtype: Optional[Any] = None  # cudnn.data_type of G
    # dims(G) == dims(O).  A broadcast G (pointwise mul broadcasts) is a LEGAL
    # graph the fused kernels do not serve -- recorded, so the row declines it
    # with a message, never _invalid.
    epilogue_gate_shape_ok: bool = True
    # dtype the IR gave the virtual O_v: the graph's intermediate dtype, which
    # defaults to the IO dtype (_pygraph.py: ``intermediate_data_type or
    # io_data_type``).  The fused kernel never materialises O_v -- it gates the
    # fp32 pre-cast accumulator -- so a row accepts FLOAT or Q's dtype only.
    sdpa_o_virtual_dtype: Optional[Any] = None
    # O_v.dim_assigned and O_v.stride_assigned (graph_types.Tensor): the classic
    # C++ pre-validation needs a rank-4 dim + stride on the sdpa node's O, and
    # only USER-assigned values are pushed at lowering, so an undeclared O_v
    # would fail planning with a bare ValueError instead of a typed decline.
    sdpa_o_virtual_declared: bool = True
    # G's layout is one the fused kernel TMA-loads ZERO-COPY: BSHD-compact, or
    # a declared BSHD layout TMA can express (D innermost-contiguous, seq/head
    # strides 16-byte multiples, non-overlapping).  G has NO normalisation-copy
    # fallback (Q/K/V/O do), so the standalone adapter's check_support raises on
    # anything else -- this fact lets the row DECLINE the same G by message
    # instead of admitting a plan that dies in the lowering (rule 8b).  Mirrors
    # the shared config_sm100 layout predicate (gate_layout_ok below).
    epilogue_gate_layout_ok: bool = True
    shape_overrides: bool = False  # graph permits execute-time geometry; the chosen plan must consume it
    # caller replays execute under a CUDA graph (pygraph(is_cuda_graph_replay_expected=True)):
    # per-execute host costs are paid once at capture, so heuristics may lead with the GPU-time optimum
    cuda_graph_replay: bool = False


_SDPA_NODE_TYPES = (
    cudnn.NodeType.SDPA,
    cudnn.NodeType.SDPA_BWD,
    cudnn.NodeType.SDPA_MXFP8,
    cudnn.NodeType.SDPA_FP8,
    cudnn.NodeType.SDPA_FP8_BWD,
    cudnn.NodeType.SDPA_MXFP8_BWD,
)


def _sdpa_node_and_tail(graph: "cudnn.pygraph") -> tuple:
    """``(node, tail)``: the graph's SDPA node and, when the graph is the
    three-node epilogue-gate tail, its :class:`GateTail`; ``(None, None)`` when
    the graph is anything else.

    Two shapes are recognised and nothing in between: a SINGLE SDPA node of any
    flavor (forward, backward, FP8, MXFP8), or exactly ``sdpa -> sigmoid(G) ->
    mul`` as :func:`cudnn._sdpa_tail.match_gate_tail` defines it.  A second op
    of any other kind still makes the graph "not ours".
    """
    try:
        nodes = list(graph.nodes)
    except Exception:  # noqa: BLE001 — non-IR graph objects
        return None, None
    if len(nodes) == 1:
        node = nodes[0]
        return (node, None) if node.node_type in _SDPA_NODE_TYPES else (None, None)
    tail = match_gate_tail(nodes)
    return (tail.sdpa, tail) if tail is not None else (None, None)


def _single_sdpa_node(graph: "cudnn.pygraph") -> Optional[Any]:
    """The graph's SDPA node (forward, backward, or an FP8/MXFP8 flavor) --
    alone on the graph or heading the epilogue-gate tail -- or None if the
    graph is anything else.  See :func:`_sdpa_node_and_tail`."""
    return _sdpa_node_and_tail(graph)[0]


def _record_from_node(node: Any, tail: Optional[GateTail] = None) -> dict:
    """Flatten an SDPA node into one kwargs-style dict.

    ``node.params`` holds the scalar sdpa() kwargs verbatim; tensor kwargs are
    named ports in ``node.inputs`` (port name == kwarg name); O / Stats (and
    the output-style kwargs like rng_dump) live in ``node.outputs``.

    With an epilogue-gate ``tail`` the record's ``o`` is the MUL output (the
    real O the fused kernel writes), ``_o_virtual`` the sdpa node's own virtual
    O and ``gate`` the graph input G.
    """
    rec: dict = dict(node.params)
    for port, t in node.inputs.items():
        rec.setdefault(port, t)
    # Forward: O / Stats are node outputs. Backward: o / stats are INPUT
    # ports (already folded above) — don't clobber them with the absent
    # forward output ports.
    if node.outputs.get("O") is not None:
        rec["o"] = node.outputs.get("O")
    if node.outputs.get("Stats") is not None:
        rec["stats"] = node.outputs.get("Stats")
    is_mxfp8_bwd = node.node_type == cudnn.NodeType.SDPA_MXFP8_BWD
    is_fp8_bwd = node.node_type == cudnn.NodeType.SDPA_FP8_BWD
    rec["_is_backward"] = node.node_type == cudnn.NodeType.SDPA_BWD or is_mxfp8_bwd or is_fp8_bwd
    rec["_is_mxfp8_bwd"] = is_mxfp8_bwd
    rec["_is_fp8_bwd"] = is_fp8_bwd
    if rec["_is_backward"]:
        for port in ("dQ", "dK", "dV", "dBias", "dSink_token", "amax_dQ", "amax_dK", "amax_dV", "amax_dP"):
            if rec.get(port) is None:
                rec[port] = node.outputs.get(port)
    if is_mxfp8_bwd:
        # sdpa_mxfp8_backward names its half-precision forward output ``o_f16``
        # (and its half-precision gradient ``dO_f16``); the plain ``dO`` port is
        # the FP8 payload. Alias O so the shared shape/layout logic sees it.
        if rec.get("o") is None:
            rec["o"] = rec.get("o_f16")
    # Output-style kwargs (passed as sdpa() arguments but recorded in
    # node.outputs): fold each one in so engines see every requested output.
    # Missing one here lets an engine that never writes it pass the probe and
    # silently leave that output buffer as garbage (see score_max/score_sum_exp).
    for out_kwarg in ("rng_dump", "score_max", "score_sum_exp", "sf_o"):
        if rec.get(out_kwarg) is None:
            rec[out_kwarg] = node.outputs.get(out_kwarg)
    # MXFP8 / per-tensor FP8: descale_q/k/v (+ scale_o, etc. for FP8; the
    # descale_o/dO/dP + scale_dQ/dK/dV/dP set on the FP8 backward) arrive via
    # node.inputs above; Amax_S / Amax_O are outputs. The FP8 input dtype + these ports
    # distinguish these ops from plain sdpa().
    rec["_is_mxfp8"] = node.node_type in (cudnn.NodeType.SDPA_MXFP8, cudnn.NodeType.SDPA_MXFP8_BWD)
    rec["_is_fp8"] = node.node_type in (cudnn.NodeType.SDPA_FP8, cudnn.NodeType.SDPA_FP8_BWD)
    rec["amax_o"] = node.outputs.get("Amax_O")
    rec["amax_s"] = node.outputs.get("Amax_S")
    if node.params.get("_dropout_n"):
        rec["dropout"] = True  # any dropout spec (tensors or probability) is requested
    if tail is not None:
        rec["o"] = tail.o_final
        rec["_o_virtual"] = tail.o_virtual
        rec["gate"] = tail.gate
    return rec


def _invalid(reason: str) -> SdpaGraphFacts:
    return SdpaGraphFacts(invalid=f"cudnn.sdpa: {reason}")


def _first_not_none(*vals):
    """First non-None value (used to fold the op family's several mask spellings)."""
    for v in vals:
        if v is not None:
            return v
    return None


def _extract_facts(rec: dict) -> SdpaGraphFacts:
    is_backward = bool(rec.get("_is_backward"))
    q, k, v, o = rec.get("q"), rec.get("k"), rec.get("v"), rec.get("o")
    if q is None or k is None or v is None or o is None:
        return _invalid("missing q/k/v/o on the sdpa node")

    is_mxfp8_bwd = bool(rec.get("_is_mxfp8_bwd"))
    is_fp8_bwd = bool(rec.get("_is_fp8_bwd"))
    rank4_ports = [("q", q), ("k", k), ("v", v), ("o", o)]
    if is_backward:
        for name in ("dO", "dQ", "dK", "dV"):
            t = rec.get(name)
            if t is None:
                return _invalid(f"missing {name} on the sdpa_backward node")
            rank4_ports.append((name, t))
    if is_mxfp8_bwd:
        # The transposed-quantization payloads and the half-precision dO are
        # first-class operands of the MXFP8 backward (each contraction axis
        # needs its own 1x32 quantization; a transposed payload is not the
        # transpose of the payload).
        for name in ("q_T", "k_T", "dO_T", "dO_f16"):
            t = rec.get(name)
            if t is None:
                return _invalid(f"missing {name} on the sdpa_mxfp8_backward node")
            rank4_ports.append((name, t))

    dims = {}
    strides = {}
    for name, t in rank4_ports:
        d = tuple(t.get_dim())
        if len(d) != 4:
            return _invalid(f"{name} must be rank-4 (B, H, S, D); got rank {len(d)}")
        dims[name] = d
        strides[name] = tuple(t.get_stride())
    q_dim, q_stride = dims["q"], strides["q"]
    k_dim, k_stride = dims["k"], strides["k"]
    v_dim, v_stride = dims["v"], strides["v"]
    o_dim, o_stride = dims["o"], strides["o"]

    b, h_q, s_q, d_qk = q_dim

    # ``build_operation_graph`` rewrites the BACKWARD node's K / V ports to
    # transposed (B, H, D, S) views (the K^T / V^T the lowering consumes);
    # the underlying buffer keeps the user's (B, H, S, D) shape.  Canonicalize
    # so probing works both before and after the native build.
    if is_backward:

        def _square_transposed(dim: tuple, stride: tuple) -> bool:
            """Square (S == D) rewritten views are extent-ambiguous; the stride
            order disambiguates: the transposed view keeps the buffer's unit
            stride, which lands on axis 2 instead of axis 3."""
            return dim[2] == dim[3] != 1 and stride[2] == 1 and stride[3] != 1

        if (k_dim[3] != d_qk and k_dim[2] == d_qk) or _square_transposed(k_dim, k_stride):
            k_dim = (k_dim[0], k_dim[1], k_dim[3], k_dim[2])
            k_stride = (k_stride[0], k_stride[1], k_stride[3], k_stride[2])
        _, h_kv, s_kv, _ = k_dim
        if (v_dim[2] != s_kv and v_dim[3] == s_kv) or _square_transposed(v_dim, v_stride):
            v_dim = (v_dim[0], v_dim[1], v_dim[3], v_dim[2])
            v_stride = (v_stride[0], v_stride[1], v_stride[3], v_stride[2])
        dims["k"], dims["v"] = k_dim, v_dim
        strides["k"], strides["v"] = k_stride, v_stride
    table_k, table_v = rec.get("paged_attention_k_table"), rec.get("paged_attention_v_table")
    paged = table_k is not None or table_v is not None
    page_size = 0
    d_v = v_dim[-1]
    if paged:
        # cuDNN paged-cache contract: K/V are page POOLS [num_pages, H_kv,
        # page_size, D] (the in-page layout rides the strides), addressed
        # through (B, 1, max_pages, 1) int32 block tables; the logical S_kv is
        # the declared maximum, else what the tables can address.
        if is_backward:
            return _invalid("paged KV caches are forward-only")
        _, h_kv, page_size, _ = k_dim
        if k_dim[3] != d_qk:
            return _invalid(f"paged K container head dim must equal Q's (q_dim={q_dim}, k_dim={k_dim})")
        if v_dim[:3] != k_dim[:3]:
            return _invalid(f"paged K/V containers must share [num_pages, H_kv, page_size] (k_dim={k_dim}, v_dim={v_dim})")
        tables = [t for t in (table_k, table_v) if t is not None]
        for t in tables:
            td = tuple(t.get_dim())
            if len(td) != 4 or td[0] != b or td[1] != 1 or td[3] != 1:
                return _invalid(f"paged block table must be (B, 1, max_pages, 1) = ({b}, 1, *, 1); got {td}")
            if t.get_data_type() != cudnn.data_type.INT32:
                return _invalid(f"paged block table must be int32; got {t.get_data_type()}")
            if getattr(t, "ragged_offset", None) is not None:
                return _invalid("packed (ragged-offset) paged block tables are not supported")
        max_pages = tuple(tables[0].get_dim())[2]
        if any(tuple(t.get_dim())[2] != max_pages for t in tables):
            return _invalid("paged K and V block tables must have the same max_pages extent")
        declared_max = rec.get("paged_attention_max_seq_len_kv")
        s_kv = int(declared_max) if declared_max is not None else max_pages * page_size
        if s_kv > max_pages * page_size:
            return _invalid(f"paged_attention_max_seq_len_kv ({s_kv}) exceeds what the block table addresses ({max_pages} pages x {page_size})")
        if (k_stride[1] > k_stride[2]) != (v_stride[1] > v_stride[2]):
            return _invalid("paged K and V containers must share an in-page layout (both HND or both NHD)")
        table_k = table_k if table_k is not None else table_v
        table_v = table_v if table_v is not None else table_k
    else:
        _, h_kv, s_kv, _ = k_dim
        if k_dim != (b, h_kv, s_kv, d_qk):
            return _invalid(f"K shape mismatch (q_dim={q_dim}, k_dim={k_dim})")
        if v_dim != (b, h_kv, s_kv, d_v):
            return _invalid(f"V shape mismatch (k_dim={k_dim}, v_dim={v_dim})")
    if o_dim != (b, h_q, s_q, d_v):
        return _invalid(f"O shape mismatch (q_dim={q_dim}, o_dim={o_dim})")
    if is_backward:
        if dims["dO"] != (b, h_q, s_q, d_v):
            return _invalid("dO shape mismatch")
        if dims["dQ"] != dims["q"] or dims["dK"] != dims["k"] or dims["dV"] != dims["v"]:
            return _invalid("dQ/dK/dV must match Q/K/V shapes")
    if is_mxfp8_bwd:
        if dims["q_T"] != dims["q"] or dims["k_T"] != dims["k"]:
            return _invalid("q_T / k_T must match the Q / K shapes")
        if dims["dO_T"] != dims["dO"] or dims["dO_f16"] != dims["dO"]:
            return _invalid("dO_T / dO_f16 must match the dO shape")
    if any(x <= 0 for x in (b, h_q, h_kv, s_q, s_kv, d_qk, d_v)):
        return _invalid("B/H/S/D must all be > 0")
    if h_q % h_kv != 0:
        return _invalid("H_q must be divisible by H_kv (GQA / MQA)")

    # Epilogue gate (the sdpa -> sigmoid(G) -> mul tail).  G is described, never
    # judged: its shape / dtype / layout become facts the gate-claiming rows
    # match in mismatch(); a G that is not rank-4 or not O-shaped is a legal
    # (broadcasting) pointwise graph every FROST row declines by message.
    gate = rec.get("gate")
    has_gate = gate is not None
    gate_dim = tuple(gate.get_dim()) if has_gate else ()
    gate_stride = tuple(gate.get_stride()) if has_gate else ()
    gate_shape_ok = (not has_gate) or (len(gate_dim) == 4 and gate_dim == o_dim and len(gate_stride) == 4)
    gate_dtype = gate.get_data_type() if has_gate else None
    # G is judged by ITS OWN layout fact (the zero-copy TMA rule the adapter
    # enforces), not folded into the Q/K/V/O layout lists below: the kernel
    # has no normalisation copy for G, and a G-only failure must name G rather
    # than fire the generic "Q/K/V/O must be BSHD-physical" decline.
    gate_layout_ok_fact = (not has_gate) or gate_layout_ok(gate_dim, gate_stride, _ELEM_BYTES.get(gate_dtype, 1))
    o_virtual = rec.get("_o_virtual")
    o_virtual_dtype = o_virtual.get_data_type() if o_virtual is not None else None
    o_virtual_declared = bool(getattr(o_virtual, "dim_assigned", True) and getattr(o_virtual, "stride_assigned", True)) if o_virtual is not None else True

    is_mxfp8 = bool(rec.get("_is_mxfp8"))
    is_fp8 = bool(rec.get("_is_fp8"))
    _fp8_family = is_mxfp8 or is_fp8
    q_dtype = q.get_data_type() if q.get_data_type() in _KNOWN_DTYPES else None
    o_dtype = o.get_data_type() if o.get_data_type() in _KNOWN_DTYPES else None
    # Half-precision side of a quantized graph: O (and, for the MXFP8 backward,
    # dO_f16 / dQ / dK / dV) share one dtype that is independent of the FP8
    # payloads. ``uniform_out_dtype`` records whether they agree; the half rows
    # fold this into ``uniform_dtype`` (everything equals Q there).
    uniform_out = True
    grad_dtype = None
    if _fp8_family:
        # FP8 in: O dtype is independent of the input; only K/V must match Q.
        # The per-tensor FP8 BACKWARD consumes O as an FP8 payload too (it
        # carries descale_o), and its non-payload side is the gradient triple.
        if is_mxfp8_bwd:
            _fp8_ports = [k, v, rec["dO"], rec["q_T"], rec["k_T"], rec["dO_T"]]
        elif is_fp8_bwd:
            _fp8_ports = [k, v, o, rec["dO"]]
        else:
            _fp8_ports = [k, v]
        uniform = all(t.get_data_type() == q_dtype for t in _fp8_ports)
        if is_mxfp8_bwd:
            uniform_out = all(t.get_data_type() == o_dtype for t in (rec["dO_f16"], rec["dQ"], rec["dK"], rec["dV"]))
        elif is_fp8_bwd:
            grad_dtype = rec["dQ"].get_data_type() if rec["dQ"].get_data_type() in _KNOWN_DTYPES else None
            uniform_out = all(t.get_data_type() == grad_dtype for t in (rec["dK"], rec["dV"]))
    else:
        _uniform_ports = [k, v, o] + ([rec["dO"], rec["dQ"], rec["dK"], rec["dV"]] if is_backward else [])
        uniform = all(t.get_data_type() == q_dtype for t in _uniform_ports)
    # Block-scaled O (the quantized forwards, sdpa_fp8 and sdpa_mxfp8): FP4 O
    # needs the sf_o output (E4M3 scale per 16 d); an E4M3 O with sf_o is the
    # MXFP8 output (UE8M0 scale per 32 d). Any other pairing is not a graph any
    # engine serves. sdpa_mxfp8 carries no per-tensor O scale otherwise, so its
    # scale_o (the FP4 global scale) is legal only with sf_o and mandatory for FP4.
    _quant_fwd = (is_fp8 or is_mxfp8) and not is_backward
    sf_o = rec.get("sf_o") if _quant_fwd else None
    o_block_scale = 0
    if _quant_fwd:
        _op = "sdpa_fp8" if is_fp8 else "sdpa_mxfp8"
        if o.get_data_type() == cudnn.data_type.FP4_E2M1:
            if sf_o is None:
                return _invalid(f"{_op}: an FP4_E2M1 O requires the sf_o output (E4M3 scale factors, one per 16 d elements)")
            if is_mxfp8 and rec.get("scale_o") is None:
                return _invalid("sdpa_mxfp8: an FP4_E2M1 O requires scale_o (the FP4 global scale)")
            o_block_scale = 16
        elif sf_o is not None:
            if o.get_data_type() != cudnn.data_type.FP8_E4M3:
                return _invalid(f"{_op}: sf_o with a non-FP4 O requires an FP8_E4M3 O (MXFP8 output, UE8M0 scale per 32 d elements)")
            o_block_scale = 32
        if is_mxfp8 and rec.get("scale_o") is not None and sf_o is None:
            return _invalid("sdpa_mxfp8: scale_o is accepted only together with the sf_o output (block-scaled O)")
    elif rec.get("sf_o") is not None:
        return _invalid("sf_o is an output of the quantized forwards (sdpa_fp8 / sdpa_mxfp8) only")
    _layout_ports = [(q_dim, q_stride), (k_dim, k_stride), (v_dim, v_stride), (o_dim, o_stride)]
    if is_backward:
        _layout_ports += [(dims[name], strides[name]) for name in ("dO", "dQ", "dK", "dV")]
    if is_mxfp8_bwd:
        _layout_ports += [(dims[name], strides[name]) for name in ("q_T", "k_T", "dO_T", "dO_f16")]
    # Paged pools are not BSHD tensors (an HND pool has its head stride above
    # the row stride); the BSHD fact — what the THD lowering gates on — is
    # about the ragged Q/O only.
    _bshd_ports = [(q_dim, q_stride), (o_dim, o_stride)] if paged else _layout_ports
    bshd = all(bshd_layout_ok(d, s) for d, s in _bshd_ports)
    packed = all(packed_layout_ok(d, s) for d, s in _bshd_ports)
    dense_layout = all(dense_layout_ok(d, s) for d, s in _layout_ports)

    # descale_q/k/v are the block-scale SF tensors for MXFP8, or scalar per-tensor
    # descales for FP8. Both arrive on the same-named node.inputs ports.
    dsc_q, dsc_k, dsc_v = rec.get("descale_q"), rec.get("descale_k"), rec.get("descale_v")
    if _fp8_family and (dsc_q is None or dsc_k is None or dsc_v is None):
        op = "sdpa_mxfp8_backward" if is_mxfp8_bwd else "sdpa_mxfp8" if is_mxfp8 else "sdpa_fp8_backward" if is_fp8_bwd else "sdpa_fp8"
        return _invalid(f"{op} requires descale_q / descale_k / descale_v")
    # The MXFP8 backward additionally carries one block-scale tensor per
    # transposed payload and for dO (see the port table in _pygraph.py).
    dsc_q_T, dsc_k_T, dsc_dO, dsc_dO_T = (rec.get(n) for n in ("descale_q_T", "descale_k_T", "descale_dO", "descale_dO_T"))
    if is_mxfp8_bwd and any(t is None for t in (dsc_q_T, dsc_k_T, dsc_dO, dsc_dO_T)):
        return _invalid("sdpa_mxfp8_backward requires descale_q_T / descale_k_T / descale_dO / descale_dO_T")
    # The per-tensor FP8 backward carries the rest of cuDNN's sdpa_fp8_backward
    # scalar set: every one is a REQUIRED positional of the op (C++ builder and
    # pybind alike), so a graph without one is malformed for every engine.
    _FP8_BWD_SCALARS = ("descale_o", "descale_dO", "descale_s", "descale_dP", "scale_s", "scale_dQ", "scale_dK", "scale_dV", "scale_dP")
    if is_fp8_bwd and any(rec.get(n) is None for n in _FP8_BWD_SCALARS):
        return _invalid("sdpa_fp8_backward requires " + " / ".join(_FP8_BWD_SCALARS))

    def _real_output(t):
        """The op RETURNS its amax ports unconditionally; only a real
        (non-virtual, set_output(True)) tensor is a requested output."""
        return t if (t is not None and not getattr(t, "is_virtual", True)) else None

    _quant_bwd = is_mxfp8_bwd or is_fp8_bwd
    amax_dq, amax_dk, amax_dv = (_real_output(rec.get(n)) if _quant_bwd else None for n in ("amax_dQ", "amax_dK", "amax_dV"))
    amax_dp = _real_output(rec.get("amax_dP")) if is_fp8_bwd else None

    # Masks: resolve cuDNN's several spellings to (causal, bottom_right, window_left).
    # Conflicting spellings follow the Python binding (python/pygraph/sdpa.cpp): every
    # entry point except sdpa_backward rejects them; sdpa_backward applies them in order,
    # so the later setter (the explicit left / right bound) wins.
    binding_rejects_conflicts = not (is_backward and not is_fp8_bwd and not is_mxfp8_bwd)
    use_causal = bool(rec.get("use_causal_mask", False))
    use_causal_br = bool(rec.get("use_causal_mask_bottom_right", False))
    # Left window (== length; window offset is length-1). The op family uses several
    # spellings for the same knob: sdpa / sdpa_backward → sliding_window_length or
    # diagonal_band_left_bound; sdpa_mxfp8 → diagonal_band_left_bound; sdpa_fp8 →
    # sliding_window or left_bound; fp8 / mxfp8 backward → left_bound.
    window = _first_not_none(rec.get("sliding_window_length"), rec.get("sliding_window"))
    explicit_left = _first_not_none(rec.get("diagonal_band_left_bound"), rec.get("left_bound"))
    if window is not None and explicit_left is not None and binding_rejects_conflicts:
        return _invalid("sliding window and left_bound cannot be set at the same time")
    left_bound = explicit_left if explicit_left is not None else window
    # Right band: diagonal_band_right_bound (sdpa/mxfp8) or right_bound (fp8).
    explicit_right = _first_not_none(rec.get("diagonal_band_right_bound"), rec.get("right_bound"))
    if use_causal or use_causal_br:
        if binding_rejects_conflicts:
            if use_causal and use_causal_br:
                return _invalid("use_causal_mask and use_causal_mask_bottom_right cannot both be true")
            if explicit_right is not None:
                flag = "use_causal_mask" if use_causal else "use_causal_mask_bottom_right"
                return _invalid(f"{flag} and diagonal_band_right_bound cannot be set at the same time")
        resolved_right = 0 if explicit_right is None else explicit_right
        align_is_br = use_causal_br
    else:
        resolved_right = explicit_right
        # Alignment is a property OF the diagonal band; with no band bound at all
        # there is no diagonal, so BOTTOM_RIGHT is inert — recording it as a fact
        # would make every engine reject an effectively-unmasked graph.
        has_band = resolved_right is not None or left_bound is not None
        align_is_br = has_band and rec.get("diagonal_alignment") == cudnn.diagonal_alignment.BOTTOM_RIGHT
    if resolved_right is not None and resolved_right < 0:
        return _invalid(f"diagonal-band right bound must be >= 0; got {resolved_right}")
    right_widening = resolved_right not in (0, None)
    causal = (resolved_right is not None) and not right_widening
    if left_bound is not None and left_bound < 1:
        return _invalid(f"sliding-window length must be >= 1; got {left_bound}")
    window_left = (left_bound - 1) if left_bound is not None else None

    cu_seq_q = rec.get("cu_seq_len_q")
    cu_seq_kv = rec.get("cu_seq_len_kv")
    has_cu_seq_len = cu_seq_q is not None or cu_seq_kv is not None

    # Padding / THD. Per-batch lengths arrive as seq_len_* ((B,) lengths) or
    # cu_seq_len_* ((B+1,) prefix sums, cuDNN 9.24+) per side; either form
    # satisfies the length requirement. Both-on-one-side is NOT flagged
    # invalid here (invalid means malformed-for-everyone; the backend accepts
    # it with its own precedence) — an engine that serves the cu form must
    # decline the ambiguous combination itself.
    thd = getattr(q, "ragged_offset", None) is not None
    use_padding_mask = bool(rec.get("use_padding_mask", False))
    seq_len_kv = rec.get("seq_len_kv")
    seq_len_q = rec.get("seq_len_q")
    q_lens_given = seq_len_q is not None or cu_seq_q is not None
    kv_lens_given = seq_len_kv is not None or cu_seq_kv is not None
    seq_q_trim = False
    if thd:
        # Paged KV: the pools are addressed through the block tables, so only
        # Q (and O) are ragged — chunked prefill over a paged cache.
        if not paged and (getattr(k, "ragged_offset", None) is None or getattr(v, "ragged_offset", None) is None):
            return _invalid("THD (ragged) requires ragged Q, K, and V")
        if not q_lens_given or not kv_lens_given:
            return _invalid("THD (ragged) requires seq_len_q/cu_seq_len_q and seq_len_kv/cu_seq_len_kv")
        padded = True
    else:
        if use_padding_mask and not kv_lens_given:
            return _invalid("use_padding_mask requires seq_len_kv or cu_seq_len_kv")
        seq_q_trim = q_lens_given and not use_padding_mask
        padded = use_padding_mask and kv_lens_given

    # The kernels consume per-batch lengths / prefix sums as int32 directly;
    # there is no implicit conversion anywhere on the execute path (it would
    # allocate and launch a cast kernel).
    for name, t in (("seq_len_q", seq_len_q), ("seq_len_kv", seq_len_kv), ("cu_seq_len_q", cu_seq_q), ("cu_seq_len_kv", cu_seq_kv)):
        if t is not None:
            if t.get_data_type() != cudnn.data_type.INT32:
                return _invalid(f"{name} must be int32; got {t.get_data_type()}")

    sink_token = rec.get("sink_token")
    if sink_token is not None:
        sink_dim = tuple(sink_token.get_dim())
        if sink_dim != (1, h_q, 1, 1):
            return _invalid(f"sink_token must be (1, H_q, 1, 1); got {sink_dim}")
        # The kernels consume fp32 sink logits directly; there is no implicit
        # conversion anywhere on the execute path (it would allocate and
        # launch a cast kernel).
        if sink_token.get_data_type() != cudnn.data_type.FLOAT:
            return _invalid(f"sink_token must be float32; got {sink_token.get_data_type()}")

    stats = rec.get("stats")
    if is_backward:
        wants_stats = False
        if stats is None:
            return _invalid("sdpa_backward requires the forward stats tensor")
    else:
        generate_stats = rec.get("generate_stats")
        is_inference = rec.get("is_inference")
        if generate_stats is not None:
            wants_stats = bool(generate_stats)
        elif is_inference is not None:
            wants_stats = not bool(is_inference)
        else:
            wants_stats = False
        if wants_stats and stats is None:
            return _invalid("generate_stats=True but no Stats tensor was returned")

    attn_scale = rec.get("attn_scale")
    dynamic_scale = attn_scale is not None and not isinstance(attn_scale, (int, float))
    # An omitted attn_scale is no scaling (1.0), as the backend lowers it. None reaching an adapter selects its
    # direct-API default (1/sqrt(d)), so a static graph scale is never None.
    scale = None if dynamic_scale else (1.0 if attn_scale is None else float(attn_scale))

    return SdpaGraphFacts(
        b=b,
        h_q=h_q,
        h_kv=h_kv,
        s_q=s_q,
        s_kv=s_kv,
        d_qk=d_qk,
        d_v=d_v,
        dtype=q_dtype,
        uniform_dtype=uniform,
        uniform_out_dtype=uniform_out,
        bshd_layout=bshd,
        packed_layout=packed,
        dense_layout=dense_layout,
        port_layouts=(tuple((name, dims[name], strides[name]) for name, _ in rank4_ports) if is_backward else ()),
        is_mxfp8=is_mxfp8,
        is_fp8=is_fp8,
        dtype_o=(grad_dtype if is_fp8_bwd else o_dtype if _fp8_family else q_dtype),
        o_block_scale=o_block_scale,
        causal=causal,
        bottom_right=bool(align_is_br),
        window_left=window_left,
        right_band_widening=right_widening,
        is_backward=is_backward,
        right_bound=resolved_right,
        deterministic=bool(rec.get("use_deterministic_algorithm", False)),
        has_dbias=rec.get("dBias") is not None,
        has_dsink=rec.get("dSink_token") is not None,
        has_bias=rec.get("bias") is not None,
        has_dropout=rec.get("dropout") is not None,
        has_score_mod=rec.get("fn") is not None,
        has_paged_kv=paged,
        page_size=page_size,
        paged_k_table_t=table_k,
        paged_v_table_t=table_v,
        has_alibi=bool(rec.get("use_alibi_mask")),
        has_unfuse_fma=bool(rec.get("unfuse_fma")),
        has_stats_log2=bool(rec.get("stats_use_log2")) and wants_stats,
        has_block_mask=rec.get("block_mask") is not None,
        has_rng_dump=rec.get("rng_dump") is not None,
        has_score_max=rec.get("score_max") is not None,
        has_score_sum_exp=rec.get("score_sum_exp") is not None,
        dynamic_scale=dynamic_scale,
        seq_q_trim=seq_q_trim,
        padded=padded,
        thd=thd,
        max_total_seq_len_q=rec.get("max_total_seq_len_q"),
        max_total_seq_len_kv=rec.get("max_total_seq_len_kv"),
        has_cu_seq_len=has_cu_seq_len,
        has_sink=sink_token is not None,
        wants_stats=wants_stats,
        scale=scale,
        device_cc=_device_cc(),
        device_sm_count=_device_sm_count(),
        q_t=q,
        k_t=k,
        v_t=v,
        o_t=o,
        stats_t=(stats if (wants_stats or is_backward) else None),
        bias_t=rec.get("bias"),
        block_mask_t=rec.get("block_mask"),
        score_max_t=rec.get("score_max"),
        score_sum_exp_t=rec.get("score_sum_exp"),
        do_t=rec.get("dO"),
        dq_t=rec.get("dQ"),
        dk_t=rec.get("dK"),
        dv_t=rec.get("dV"),
        dbias_t=rec.get("dBias"),
        dsink_t=rec.get("dSink_token"),
        q_T_t=(rec.get("q_T") if is_mxfp8_bwd else None),
        k_T_t=(rec.get("k_T") if is_mxfp8_bwd else None),
        dO_T_t=(rec.get("dO_T") if is_mxfp8_bwd else None),
        dO_f16_t=(rec.get("dO_f16") if is_mxfp8_bwd else None),
        sf_q_T_t=(dsc_q_T if is_mxfp8_bwd else None),
        sf_k_T_t=(dsc_k_T if is_mxfp8_bwd else None),
        sf_dO_t=(dsc_dO if is_mxfp8_bwd else None),
        sf_dO_T_t=(dsc_dO_T if is_mxfp8_bwd else None),
        amax_dq_t=amax_dq,
        amax_dk_t=amax_dk,
        amax_dv_t=amax_dv,
        amax_dp_t=amax_dp,
        has_amax_dgrad=any(t is not None for t in (amax_dq, amax_dk, amax_dv, amax_dp)),
        descale_o_t=(rec.get("descale_o") if is_fp8_bwd else None),
        descale_do_t=(rec.get("descale_dO") if is_fp8_bwd else None),
        descale_dp_t=(rec.get("descale_dP") if is_fp8_bwd else None),
        scale_dq_t=(rec.get("scale_dQ") if is_fp8_bwd else None),
        scale_dk_t=(rec.get("scale_dK") if is_fp8_bwd else None),
        scale_dv_t=(rec.get("scale_dV") if is_fp8_bwd else None),
        scale_dp_t=(rec.get("scale_dP") if is_fp8_bwd else None),
        sink_t=sink_token,
        seq_kv_t=seq_len_kv,
        seq_q_t=seq_len_q,
        cu_seq_q_t=cu_seq_q,
        cu_seq_kv_t=cu_seq_kv,
        sf_q_t=(dsc_q if is_mxfp8 else None),
        sf_k_t=(dsc_k if is_mxfp8 else None),
        sf_v_t=(dsc_v if is_mxfp8 else None),
        # Amax_O: like Amax_S, the op RETURNS the port unconditionally; only a
        # real (non-virtual, set_output(True)) tensor is a requested output.  A
        # virtual one used to enter SdpaBinding and make the plan demand a buffer
        # for it (engine._FrostSdpaFwdPlan: "missing buffers").
        amax_o_t=_real_output(rec.get("amax_o")),
        sf_o_t=sf_o,
        descale_q_t=(dsc_q if is_fp8 else None),
        descale_k_t=(dsc_k if is_fp8 else None),
        descale_v_t=(dsc_v if is_fp8 else None),
        scale_o_t=(rec.get("scale_o") if (is_fp8 or is_mxfp8) else None),
        descale_s_t=(rec.get("descale_s") if is_fp8 else None),
        scale_s_t=(rec.get("scale_s") if is_fp8 else None),
        # Amax_S: the op RETURNS the port unconditionally; only a real
        # (non-virtual, set_output(True)) tensor is a requested output.
        amax_s_t=(rec.get("amax_s") if (is_fp8 and rec.get("amax_s") is not None and not getattr(rec.get("amax_s"), "is_virtual", True)) else None),
        has_epilogue_gate=has_gate,
        epilogue_gate_t=gate,
        epilogue_gate_dtype=gate_dtype,
        epilogue_gate_shape_ok=gate_shape_ok,
        sdpa_o_virtual_dtype=o_virtual_dtype,
        sdpa_o_virtual_declared=o_virtual_declared,
        epilogue_gate_layout_ok=gate_layout_ok_fact,
    )


def _as_bool_flag(value):
    """A bool-valued op attribute as Python bool: bool, numpy bool (``numpy.bool_`` -- named ``bool`` on numpy 2),
    or the 0 / 1 a config file carries.  None for None and for anything else (the caller reports the latter)."""
    if value is None or isinstance(value, bool):
        return value
    t = type(value)
    if t.__module__ == "numpy" and t.__name__ in ("bool", "bool_"):
        return bool(value)
    if isinstance(value, numbers.Integral) and int(value) in (0, 1):
        return bool(int(value))
    return None


def analyze(graph: "cudnn.pygraph") -> Optional[SdpaGraphFacts]:
    """Facts for a single-SDPA graph -- optionally followed by the epilogue-gate
    tail ``O * sigmoid(G)`` -- or None if the graph is anything else.

    Pure: attaching and caching is the graph's job (validate() ->
    _attach_facts), so the ranking and the engine share one record instead of
    each holding a private one. This is the callable the manifest names in the
    SDPA family's ``analyzer``.
    """
    node, tail = _sdpa_node_and_tail(graph)
    if node is None:
        return None
    facts = _extract_facts(_record_from_node(node, tail))
    if getattr(graph, "_cpp_graph_kwargs", {}).get("is_override_shape_enabled", False):
        facts = replace(facts, shape_overrides=True)
    if getattr(graph, "is_cuda_graph_replay_expected", False):
        facts = replace(facts, cuda_graph_replay=True)
    if facts.invalid is not None:
        return facts
    # sdpa(..., softmax_precision=...) is a python-only op attribute (see
    # _pygraph._CAPTURED_OPS): FLOAT / None is the f32 pipeline every row runs,
    # HALF asks for the f16 softmax accumulator arm. Numerics-changing, so it
    # is a fact the capability rows gate on, never a tuning knob.
    requested = node.params.get("softmax_precision")
    # attn_scale_prefolded: a bool, or a bool-like flag from a config file (0 / 1, a numpy bool); anything else is
    # malformed.  Normalized ONCE here so every rule below (forward-only, the fold itself) sees the same value that
    # _pygraph._python_only_attr_is_set saw -- a false-valued 0 / numpy False is the default, exactly like False.
    prefolded_raw = node.params.get("attn_scale_prefolded")
    prefolded = _as_bool_flag(prefolded_raw)
    if prefolded_raw is not None and prefolded is None:
        return replace(facts, invalid=f"cudnn.sdpa: attn_scale_prefolded must be a bool; got {prefolded_raw!r}")
    if facts.is_backward and (requested is not None or prefolded):
        # Both attributes describe the FORWARD softmax (its exponent arm, its scale contract); the backward
        # recomputes P from the forward's unscaled contract and no backward engine carries either arm.
        return replace(facts, invalid="cudnn.sdpa_backward: softmax_precision / attn_scale_prefolded are forward-only op attributes; leave them unset")
    if requested == cudnn.data_type.HALF:
        facts = replace(facts, softmax_precision=cudnn.data_type.HALF)
    elif requested is not None and requested != cudnn.data_type.FLOAT:
        return replace(facts, invalid=f"cudnn.sdpa: softmax_precision must be cudnn.data_type.FLOAT or HALF; got {requested}")
    # sdpa(..., attn_scale_prefolded=True) is the second python-only op attribute:
    # Q already carries attn_scale * log2(e), so the engine applies no softmax
    # scale (the kernel's pre-folded arm traces no per-score multiply).  A
    # contract on the graph's Q, hence a fact, never a knob; False / None is the
    # default (the engine scales).  attn_scale set alongside it is a malformed
    # request: the scale would be applied twice or silently dropped.
    if not prefolded:
        return facts
    # "Unset" is the node's own record (no attn_scale kwarg, no attn_scale tensor port): facts.scale cannot
    # tell, since an OMITTED attn_scale is reported as the backend's 1.0 (see _extract_facts).
    if node.params.get("attn_scale") is not None or "attn_scale" in node.inputs or facts.dynamic_scale:
        return replace(
            facts,
            invalid="cudnn.sdpa: attn_scale_prefolded=True: leave attn_scale unset -- Q already carries attn_scale * log2(e) and the engine applies no scale",
        )
    # scale=None here means "the engine applies no scale" (the adapters receive scale_softmax=None together with
    # softmax_scale_prefolded=True); it is the one static-graph case where the fact is None rather than a float.
    return replace(facts, attn_scale_prefolded=True, scale=None)


# ---------------------------------------------------------------------------
# Graph-side runtime helpers shared by SDPA engines
# ---------------------------------------------------------------------------


@dataclass
class SdpaBinding:
    q: Any
    k: Any
    v: Any
    o: Any
    stats: Any = None
    sink_token: Any = None
    seq_len_kv: Any = None
    seq_len_q: Any = None
    # (B+1,) prefix-sum form (cuDNN 9.24+); at most one form per side.
    cu_seq_len_q: Any = None
    cu_seq_len_kv: Any = None
    # Paged KV block tables.
    paged_k_table: Any = None
    paged_v_table: Any = None
    # MXFP8 block-scale (descale) tensors + Amax_O output.
    sf_q: Any = None
    sf_k: Any = None
    sf_v: Any = None
    amax_o: Any = None
    # Block-scaled O scale-factor output (sdpa_fp8 sf_o).
    sf_o: Any = None
    # Per-tensor FP8 scalar descales + Amax_S output.
    descale_q: Any = None
    descale_k: Any = None
    descale_v: Any = None
    scale_o: Any = None
    descale_s: Any = None
    scale_s: Any = None
    # SM80 feature operands + backward ports.
    bias: Any = None
    block_mask: Any = None
    score_max: Any = None
    score_sum_exp: Any = None
    do: Any = None
    dq: Any = None
    dk: Any = None
    dv: Any = None
    dbias: Any = None
    dsink: Any = None
    # MXFP8 backward: transposed payloads, half-precision dO, their SF tensors.
    q_T: Any = None
    k_T: Any = None
    dO_T: Any = None
    dO_f16: Any = None
    sf_q_T: Any = None
    sf_k_T: Any = None
    sf_dO: Any = None
    sf_dO_T: Any = None
    # Per-tensor FP8 backward: the remaining sdpa_fp8_backward scalars (the
    # descale_q/k/v/s + scale_s slots above are shared with the FP8 forward)
    # and the four requested-only amax outputs.
    descale_o: Any = None
    descale_dO: Any = None
    descale_dP: Any = None
    scale_dQ: Any = None
    scale_dK: Any = None
    scale_dV: Any = None
    scale_dP: Any = None
    amax_dQ: Any = None
    amax_dK: Any = None
    amax_dV: Any = None
    amax_dP: Any = None
    # Epilogue gate G (the sdpa -> sigmoid(G) -> mul tail): a REQUIRED bound
    # operand of the fused kernel.  The tail's virtual O_v / s are never bound.
    gate: Any = None
    # THD ragged-offset tensors ((B+1,) int32 / int64): bound operands of the
    # lowerings that read them on device -- Q / O / Stats for the SM100 decode
    # tile's ragged-Q leg, every port for the SM80 backward (caller-buffer token
    # origins).  Every other lowering leaves them unbound.
    ragged_q: Any = None
    ragged_o: Any = None
    ragged_stats: Any = None
    ragged_k: Any = None
    ragged_v: Any = None
    ragged_do: Any = None
    ragged_dq: Any = None
    ragged_dk: Any = None
    ragged_dv: Any = None

    # Built once on first use and reused. Rebuilding it per execute cost ~1.3 us
    # per bound operand: three passes over the bound list and five dict
    # constructions, not any one expensive getter.
    #
    # What makes the cache safe is the graph, not this class: a binding is
    # constructed by the engine's lowering AFTER the graph is frozen, and a
    # frozen graph can no longer re-uid or rename a tensor, so the names and
    # uids indexed here cannot move. The binding itself is still an ordinary
    # mutable dataclass -- reassigning a field after the first index() would go
    # unnoticed. Nothing does; an ordered-slot binding would remove the question.
    # init=False so a replace()d binding rebuilds rather than inheriting a
    # cache for the operands it no longer has; compare/repr excluded so the
    # cache cannot change how a binding prints or compares.
    _index: Optional[tuple] = field(default=None, init=False, repr=False, compare=False)

    def _build_index(self) -> tuple:
        bound = [
            t
            for t in (
                self.q,
                self.k,
                self.v,
                self.o,
                self.stats,
                self.sink_token,
                self.seq_len_kv,
                self.seq_len_q,
                self.cu_seq_len_q,
                self.cu_seq_len_kv,
                self.paged_k_table,
                self.paged_v_table,
                self.sf_q,
                self.sf_k,
                self.sf_v,
                self.amax_o,
                self.sf_o,
                self.descale_q,
                self.descale_k,
                self.descale_v,
                self.scale_o,
                self.descale_s,
                self.scale_s,
                self.bias,
                self.block_mask,
                self.score_max,
                self.score_sum_exp,
                self.do,
                self.dq,
                self.dk,
                self.dv,
                self.dbias,
                self.dsink,
                self.q_T,
                self.k_T,
                self.dO_T,
                self.dO_f16,
                self.sf_q_T,
                self.sf_k_T,
                self.sf_dO,
                self.sf_dO_T,
                self.descale_o,
                self.descale_dO,
                self.descale_dP,
                self.scale_dQ,
                self.scale_dK,
                self.scale_dV,
                self.scale_dP,
                self.amax_dQ,
                self.amax_dK,
                self.amax_dV,
                self.amax_dP,
                self.gate,
                self.ragged_q,
                self.ragged_o,
                self.ragged_stats,
                self.ragged_k,
                self.ragged_v,
                self.ragged_do,
                self.ragged_dq,
                self.ragged_dk,
                self.ragged_dv,
            )
            if t is not None
        ]
        name_counts: dict = {}
        uid_counts: dict = {}
        names, uids = [], []
        for t in bound:
            nm = _safe_name(t)
            names.append(nm)
            if nm is not None:
                name_counts[nm] = name_counts.get(nm, 0) + 1
            uid = _safe_uid(t)
            uids.append(uid)
            if uid is not None:
                uid_counts[uid] = uid_counts.get(uid, 0) + 1
        # A name or uid carried by two bound tensors identifies neither.
        by_obj = {id(t): t for t in bound}
        by_name = {nm: t for nm, t in zip(names, bound) if nm is not None and name_counts[nm] == 1}
        by_uid = {uid: t for uid, t in zip(uids, bound) if uid is not None and uid_counts[uid] == 1}
        self._index = (tuple(bound), by_obj, by_uid, by_name)
        return self._index

    def index(self) -> tuple:
        """``(bound, by_obj, by_uid, by_name)`` — the resolution tables."""
        return self._index or self._build_index()

    def bound_tensors(self) -> tuple:
        """The bound tensors, in slot order. A tuple: this is the binding's own
        record, not a working list for a caller to edit."""
        return self.index()[0]


def _safe_name(t: Any) -> Optional[str]:
    try:
        nm = t.get_name()
    except Exception:  # noqa: BLE001
        return None
    return nm or None


def _safe_uid(t: Any) -> Optional[int]:
    """The tensor's user-assigned uid, or None. ``uid_assigned`` distinguishes an
    explicit ``set_uid`` (any value, including 0) from the provisional uid that
    ``get_uid`` mints on demand — matching by a provisional uid would misbind
    variant packs keyed by the user's own numbering."""
    try:
        if not getattr(t, "uid_assigned", False):
            return None
        uid = t.get_uid()
    except Exception:  # noqa: BLE001
        return None
    return uid if isinstance(uid, int) and uid >= 0 else None


def resolve_variant_pack(variant_pack: dict, binding: SdpaBinding) -> dict:
    if not isinstance(variant_pack, dict):
        raise TypeError(
            f"cudnn.sdpa: compiled plans are called with a variant-pack dict {{cudnn_tensor | uid | name: buffer}}; got {type(variant_pack).__name__}"
        )
    _bound, by_obj, by_uid, by_name = binding.index()

    resolved: dict = {}
    for key, buf in variant_pack.items():
        if id(key) in by_obj:
            t = by_obj[id(key)]
        elif isinstance(key, int) and key in by_uid:
            t = by_uid[key]
        elif isinstance(key, str) and key in by_name:
            t = by_name[key]
        else:
            continue
        resolved[id(t)] = buf
    return resolved


def tensor_desc_from_ir(t: Any, name: str = "") -> "TensorDesc":
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import TensorDesc

    shape = tuple(t.get_dim())
    stride = tuple(t.get_stride())
    dtype = to_torch_dtype(t.get_data_type())
    return TensorDesc(
        dtype=dtype,
        shape=shape,
        stride=stride,
        stride_order=_stride_order(shape, stride),
        device=_torch_cuda_device(),
        name=name,
    )


def adapter_mask_args(facts: "SdpaGraphFacts") -> dict:
    """Map resolved graph mask facts onto the standalone adapters'
    (is_causal, window_size, causal_bottom_right) vocabulary."""
    is_causal = facts.right_bound is not None
    win_left = facts.window_left if facts.window_left is not None else -1
    win_right = facts.right_bound if (facts.right_bound is not None and facts.right_bound > 0) else -1
    # A BOTTOM_RIGHT alignment only means something combined with a causal
    # bound and/or a left window; bare BR on a dense graph is a no-op.
    bottom_right = facts.bottom_right and (is_causal or facts.window_left is not None)
    return dict(
        is_causal=is_causal,
        window_size=(win_left, win_right),
        causal_bottom_right=bottom_right,
    )
