"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cake backend: fused DSA indexer scoring + non-deterministic exact top-k
selection for training on SM100 / SM103 / SM107 (flashinfer-ai/flashinfer#5676
under the set-equality semantics of flashinfer-ai/flashinfer#8946, point 4;
target geometry: 32 indexer heads, head dimension 128, top-k 2048).

Contract (the issue text is authoritative)
------------------------------------------
* ``q [T, 32, 128]`` BF16 indexer queries after RoPE (contiguous); ``k [Tkv,
  128]`` BF16 packed keys after normalization and RoPE -- a row-strided view
  (the trainer's ``[:, :128]`` view of a packed ``[Tkv, 704]`` tensor) is
  accepted without a copy; ``w [T, 32]`` FP32 signed head weights already
  scaled by ``32 ** -0.5``; ``cu_seqlens_q`` / ``cu_seqlens_k [S + 1]`` int32
  independent query / key segment boundaries on the device; optional
  ``q_causal_offsets [S]`` int64 on the device; ``top_k`` in ``[1, 4096]``
  (default 2048); ``softmax_scale > 0`` (default ``128 ** -0.5``); ``ratio
  >= 1``; ``max_seqlen_q`` is an accepted host mirror that no launch decision
  reads; ``max_seqlen_k`` (the caller's bound on every key segment, optional)
  lets the dispatch tell a packed call of equal key segments, whose scan runs
  without the longest-first segment order -- never the selected set; a value
  above ``Tkv`` or below ``ceil(Tkv / S)`` is rejected.  The package offers
  the BF16 form only: float8_e4m3fn ``q`` / ``k`` are rejected
  (:data:`BF16_ONLY`).
* Outputs ``indices [T, top_k]`` int32 segment-local key ids and ``scores [T,
  top_k]`` FP32: per row, the ``min(top_k, visible)`` selected entries and
  ``-1`` / ``-inf`` padding in the other slots, in no specified slot order --
  the order of a row's entries and the positions of its padding slots may
  vary between calls (no launch orders a row); an entirely empty row holds
  only padding.
* Score ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] *
  k[j, d])`` with FP32 products, accumulation and score arithmetic (no BF16
  intermediate rounding).  The FP32 reduction scheme of the generated program
  is documented in the package README and is fixed: the score bits do not
  depend on the internal query / key partition.
* Visibility: key ``j`` is visible to the query at segment-local position ``u``
  iff ``0 <= j < Lk`` and ``j < floor((offset + u + 1) / ratio)``; ``offset`` is
  ``q_causal_offsets[s]`` when supplied, else ``Lk - Lq`` for ``ratio == 1`` and
  ``0`` otherwise.  Keys of other segments or outside the visible prefix are
  never selected; negative offsets, empty segments and rows without visible
  keys are defined (all padding).
* Selection: the exact global top ``min(top_k, visible)`` keys per row, ranked
  by score descending then key id descending (``+0.0 == -0.0`` for ranking);
  the returned score bits are the computed FP32 bits.  The scan programs are
  the deterministic package's, so a row holds the ids and score bits that
  package returns for it, in another slot order.
* Repeatable as a set: for identical inputs and configuration (same device,
  SM count and arguments) the set of ``(id, score bits)`` pairs of every row,
  and with it the row's count of valid (non-``-1``) slots, repeats bit for
  bit.  The program the host dispatches (unit geometry, key-range split, CTA
  pair, unroll, unit order, sampled threshold) and the partition knobs
  (``grid_ctas``, ``candidate_multiplier``, ``check_period``, the
  sampled-threshold knobs) never change that set, only the slot order.
* Deterministic algorithms: under ``torch.use_deterministic_algorithms(True)``
  a call that launches programs (``T > 0`` and ``Tkv > 0``) raises PyTorch's
  ``RuntimeError`` after its argument checks and before it allocates or
  launches anything (``warn_only=True``: a ``UserWarning``, then the call
  runs); nothing falls back to the deterministic package.  A prepared runner
  (:func:`prepare_dsa_indexer_nondet_topk`) consults the setting on every run
  that launches, before its first launch; preparing one never alerts, and
  :func:`cake_dsa_indexer_nondet_train_topk_workspace_size` never does.
* Memory: no ``[T, Tkv]`` score matrix.  The workspace is
  :func:`cake_dsa_indexer_nondet_train_topk_workspace_size` bytes for the call's geometry -- the
  persistent candidate buffers ``grid x queries_per_unit x
  candidate_capacity(top_k) x entry_bytes`` plus, when the dispatch splits the
  key ranges, the staging ``T x n_split x top_k x entry_bytes``, the
  deterministic package's bound for the same call -- separate from the inputs
  and the two outputs; no host synchronization inside the operator.
* Non-finite inputs (NaN / inf / overflow) are outside the normal domain: the
  operator returns with the output structure intact and no loop bound depends
  on a score value, so the kernels cannot hang.

Every allocation happens in :func:`prepare_dsa_indexer_nondet_topk`; the returned
runner launches with no CUDA allocation and no host synchronization, so a
runner (or a CUDA graph capturing it) replays for new values written into the
bound tensors.  The host decides the program from host-known integers only
(:mod:`cake_policy`, the registry's ``POLICY`` record of the architecture) and
fails closed when the decided program is not registered for the device's
architecture.  Kernels are reached through the argument plans of the registry
(``cake_jit.ARG_PLANS``); ``cake_jit.ABI`` names the keyword set its kernels
expect (:data:`ABI_CONTRACT`).
"""

from __future__ import annotations

import functools
import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    ABI,
    ARG_PLANS,
    FFI_ENTRY,
    NUMERICS,
    POLICY,
    PROGRAM_KEYS,
    PROGRAMS,
    STAGES,
    TICKET_TAIL_ARCHES,
    TRACKING_ISSUE,
    load_program,
)
from .cake_policy import (
    MERGE_KEY,
    SAMPLE_SHIFT_PERMILLE_BOUNDS,
    TICKET_TAIL_BYTES,
    DispatchPolicy,
    ProgramChoice,
    select_program,
    ticket_static_path,
)

NUM_HEADS = 32
HEAD_DIM = 128
MAX_TOP_K = 4096
DEFAULT_TOP_K = 2048
PAD_ID = -1
PAD_SCORE = float("-inf")
TMA_ALIGNMENT_BYTES = 16
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The package offers the BF16 input form only: float8_e4m3fn q / k (the opt-in FP8 form of the deterministic package)
# are rejected with this text.
BF16_ONLY = (
    "this package offers the BF16 form only (bfloat16 q / k); the float8_e4m3fn form is the opt-in FP8 form of "
    "flashinfer.cake_dsa_indexer_det_train"
)

# Host binding profile: the keyword set of the generated kernels (the names
# below are what the host provides; a kernel's argument plan selects from
# them).  Tensor kinds in the argument plan are ``buffer`` / ``tma_buffer``
# (the binding encodes the tensor map itself from the view it receives),
# scalars are ``parameter``; ``grid_x/y/z`` are ``grid``; ``zero_region`` (kind
# ``zero``, scan only) is the byte range the scan binding zeroes on the launch
# stream right before its launch: the caller-owned ticket tail, else empty.
# ``TicketWord`` (scan only, right before ``zero_region``) is the 16-byte block
# a scan program draws its work units from where its registry record lists the
# architecture in ``ticket_arg_arches``; every other scan program takes it and
# never reads it.  ``K_scale`` (scan only, right before ``TicketWord``) is the
# per-key FP32 scale vector of the deterministic package's FP8 form, which the
# BF16 programs of this package never read: the host passes them a
# never-dereferenced placeholder.
ABI_CONTRACT = "dsa_indexer_v5"
SUPPORTED_ABIS = (ABI_CONTRACT,)
CONTRACT_TENSORS = (
    "Q",  # q viewed as [T * 32, 128] (TMA source)
    "K",  # k [Tkv, 128], any row stride that is a multiple of 16 bytes (TMA source)
    "W",  # w [T, 32] fp32 (TMA source)
    "cu_seqlens_q",
    "cu_seqlens_k",
    "q_offsets",  # q_causal_offsets, or a never-dereferenced int64 placeholder
    "Indices",
    "Scores",
    "Cand",  # the workspace viewed as packed int64 candidate entries (buffers, then the split staging)
    "Staging",  # the split staging [T, n_split, top_k] behind the candidate buffers (merge stage only)
    "zero_region",  # the scan's caller-owned ticket tail, zeroed by its binding before the launch, or an empty view
    "TicketWord",  # the scan's 16-byte ticket block (uint32 view): its own, or a zeroed placeholder it never reads
    "K_scale",  # a never-dereferenced float32 placeholder: the BF16 programs never read per-key scales
)
SCAN_SCALARS = (
    "num_segments",
    "top_k",
    "ratio",
    "has_offsets",
    "cand_cap",
    "first_cap",
    "sample_tiles_max",
    "sample_shift_permille",
    "far_sample",
    "check_period",
    "grid_ctas",
    "softmax_scale",
    "n_split",
)
MERGE_SCALARS = ("top_k", "n_split")
CONTRACT_SCALARS = SCAN_SCALARS
# Kernel-side spellings of the contract names (a renamed kernel argument
# resolves through this table; any other name fails closed at bind time).
CONTRACT_ALIASES = {
    "q": "Q",
    "k": "K",
    "w": "W",
    "indices": "Indices",
    "scores": "Scores",
    "cand": "Cand",
    "candidates": "Cand",
    "workspace": "Cand",
    "staging": "Staging",
    "q_causal_offsets": "q_offsets",
    "offsets": "q_offsets",
    "num_segs": "num_segments",
    "topk": "top_k",
    "scale": "softmax_scale",
    "sm_scale": "softmax_scale",
    "grid": "grid_ctas",
    "num_splits": "n_split",
}

# Documented zero-sign behaviour of the head reduction (registry field
# ``NUMERICS["zero_sign_policy"]``): ``positive_accumulator`` -- the FMA chains
# start from ``+0.0``, so a row whose head terms are all zero scores ``+0.0``;
# ``ieee_sum`` -- the sum starts from the ``h = 0`` term, so an all-zero sum is
# ``-0.0`` iff every term is ``-0.0``.  Either way the computed bits are
# returned unchanged and ``+0.0 == -0.0`` for ranking.
ZERO_SIGN_POLICIES = ("positive_accumulator", "ieee_sum")


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is not None and torch.device(device).type != "cuda":
        # the refusal torch.cuda.get_device_capability raised here before the facts were memoised
        raise ValueError(f"Expected a cuda device, but got: {device}")
    capability, _ = _device_facts(_device_index(device))
    return SUPPORTED_COMPUTE_CAPABILITIES.get(capability)


def default_softmax_scale() -> float:
    return HEAD_DIM**-0.5


def _check_bf16(dtype: torch.dtype) -> None:
    """The q / k dtype of a plan: ``torch.bfloat16``, the only input form of this package (:data:`BF16_ONLY`)."""
    if dtype == torch.float8_e4m3fn:
        raise ValueError(f"dtype float8_e4m3fn: {BF16_ONLY}")
    if dtype != torch.bfloat16:
        raise ValueError(f"dtype must be torch.bfloat16, got {dtype}")


def registry_abi() -> str:
    if ABI not in SUPPORTED_ABIS:
        raise NotImplementedError(
            f"unsupported host binding profile {ABI!r} (this host binds {SUPPORTED_ABIS})"
        )
    return ABI


def policy_for(arch: str) -> DispatchPolicy:
    """The dispatch policy of ``arch``: the registry's ``POLICY`` record.  The registry is immutable for the life of
    the process, so a record is validated once per architecture; an architecture without programs raises on every
    call."""
    return _policy_for(arch)


@functools.cache
def _policy_for(arch: str) -> DispatchPolicy:
    record = POLICY.get(arch)
    if record is None:
        raise NotImplementedError(
            f"The generated DSA indexer top-k programs for {arch} are not registered in this checkout "
            f"(registered: {sorted(POLICY)}; see {TRACKING_ISSUE})"
        )
    return DispatchPolicy.from_record(record)


@functools.cache
def ticket_tail_skips(arch: str) -> Optional[frozenset]:
    """None when ``arch`` is not a ticket architecture (``TICKET_TAIL_ARCHES``); else the physical scan kinds of the
    record's ``static_scan_kinds`` (``policy_for``), whose programs keep the static enumeration and get no ticket
    tail.  Fixed for the life of the process (like ``policy_for``), so a call decides its tail with one lookup and
    one set test."""
    if arch not in TICKET_TAIL_ARCHES:
        return None
    return frozenset(policy_for(arch).static_scan_kinds)


@functools.cache
def reads_ticket_word(program: str, arch: str) -> bool:
    """Whether the scan ``program`` draws its work units on ``arch`` from its ``TicketWord`` argument (the registry
    record's ``ticket_arg_arches``, which the exporter writes from the program's source build) instead of the
    workspace's ticket tail."""
    return arch in PROGRAMS[program].get("ticket_arg_arches", ())


def program_for(arch: str, key: str) -> str:
    """The registered program of ``key`` on ``arch``; fails closed by name when the dispatch reaches an unshipped form."""
    programs = PROGRAM_KEYS.get(arch)
    if programs is None:
        raise NotImplementedError(
            f"The generated DSA indexer top-k programs for {arch} are not registered in this checkout "
            f"(registered: {sorted(PROGRAM_KEYS)}; see {TRACKING_ISSUE})"
        )
    name = programs.get(key)
    if name is None:
        raise NotImplementedError(
            f"The host dispatch selected program {key!r} on {arch}, which this checkout does not register "
            f"(registered on {arch}: {sorted(programs)}; see {TRACKING_ISSUE})"
        )
    return name


def generated_program_available(device: Optional[torch.device] = None) -> bool:
    """True when this checkout registers programs for ``device`` under the host binding profile this module binds:
    scan programs and the plain split merge."""
    arch = arch_for(device)
    if (
        arch is None
        or arch not in PROGRAM_KEYS
        or arch not in POLICY
        or ABI not in SUPPORTED_ABIS
    ):
        return False
    keys = PROGRAM_KEYS[arch]
    return any(k.startswith("scan:") for k in keys) and MERGE_KEY in keys


def record_zero_sign_policy() -> str:
    """The documented zero-sign policy of the program's head reduction."""
    policy = NUMERICS.get("zero_sign_policy") if isinstance(NUMERICS, dict) else None
    if policy not in ZERO_SIGN_POLICIES:
        raise ValueError(
            f"registry declares numerics {NUMERICS!r}; zero_sign_policy must be one of {ZERO_SIGN_POLICIES}"
        )
    return str(policy)


# ---------------------------------------------------------------------------
# Deterministic algorithms
# ---------------------------------------------------------------------------

# torch.use_deterministic_algorithms(True): this is the non-deterministic package (the slot order of a row and the
# positions of its padding may vary between identical calls), so the eager entry raises PyTorch's alert before its
# first allocation and a prepared runner before the first launch of every run; preparing a runner and the workspace
# bound never do.
TOPK_ENTRY = "cake_dsa_indexer_nondet_train_topk"
RUN_ENTRY = "DSAIndexerNondetRunner.run (prepare_dsa_indexer_nondet_topk)"

# This package's module prefix (the vendored name of a vendored copy).
_PACKAGE = __name__.rpartition(".")[0]


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside this package and flashinfer, for a warning issued by the
    function that calls this one: the caller of the entry point, also through the facade and its decorators (an
    autograd backward reports the autograd frame that called it)."""
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None:
            name = str(parent.f_globals.get("__name__", ""))
            if not (
                name == _PACKAGE
                or name.startswith(_PACKAGE + ".")
                or name == "flashinfer"
                or name.startswith("flashinfer.")
            ):
                return level
        frame = parent
    return 2


def _alert_nondeterministic(entry: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, when torch's deterministic algorithms
    are enabled: a ``RuntimeError``, or a ``UserWarning`` under ``torch.use_deterministic_algorithms(True,
    warn_only=True)`` after which the call runs; nothing otherwise.  Called once per call that launches programs
    (``T > 0`` and ``Tkv > 0``): by the eager entry (:data:`TOPK_ENTRY`) after the call's argument checks, its plan
    and the registry lookup of its programs and before its first allocation or launch, and by every run of a prepared
    runner (:data:`RUN_ENTRY`) before its first launch: the whole call is the step whose slot order varies.  A call
    without queries or keys launches nothing (its rows are all padding) and does not alert; preparing a runner and the
    workspace bound never do.  There is no fallback: nothing here calls the deterministic package.  The alert
    is raised here because PyTorch exposes no public Python entry for it."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{entry} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
        return
    raise RuntimeError(
        f"{entry} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'; use the 'warn_only=True' option if that is acceptable for your "
        "application"
    )


# ---------------------------------------------------------------------------
# Visibility (host helpers; the kernels evaluate the same rule on device)
# ---------------------------------------------------------------------------


def effective_offsets(
    seg_q_len: list[int],
    seg_k_len: list[int],
    q_causal_offsets: Optional[list[int]],
    ratio: int,
) -> list[int]:
    """Per-segment effective causal offset: the supplied value, else ``Lk - Lq`` for ``ratio == 1``, else ``0``."""
    if q_causal_offsets is not None:
        if len(q_causal_offsets) != len(seg_q_len):
            raise ValueError("q_causal_offsets must hold one entry per segment")
        return [int(o) for o in q_causal_offsets]
    if int(ratio) == 1:
        return [int(lk) - int(lq) for lq, lk in zip(seg_q_len, seg_k_len, strict=True)]
    return [0] * len(seg_q_len)


def visible_key_count(offset: int, position: int, ratio: int, num_keys: int) -> int:
    """Visible keys of one query: ``max(0, min(Lk, floor((offset + u + 1) / ratio)))`` (true floor)."""
    return max(0, min(int(num_keys), (int(offset) + int(position) + 1) // int(ratio)))


def visible_key_counts(
    seg_q_len: list[int],
    seg_k_len: list[int],
    *,
    ratio: int = 1,
    q_causal_offsets: Optional[list[int]] = None,
    device=None,
) -> torch.Tensor:
    """int64 ``[T]`` visible-key count of every query row from host segment metadata (floor division)."""
    offsets = effective_offsets(seg_q_len, seg_k_len, q_causal_offsets, ratio)
    pieces = []
    for lq, lk, off in zip(seg_q_len, seg_k_len, offsets, strict=True):
        if lq == 0:
            continue
        u = torch.arange(int(lq), dtype=torch.int64, device=device)
        vis = torch.div(u + (int(off) + 1), int(ratio), rounding_mode="floor")
        pieces.append(vis.clamp_(min=0, max=int(lk)))
    if not pieces:
        return torch.zeros(0, dtype=torch.int64, device=device)
    return torch.cat(pieces)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_int(name: str, value: Any, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < lo or (hi is not None and value > hi):
        bound = f"in [{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return int(value)


def _check_max_seqlen_k(value: int, num_keys: int, num_segments: int) -> int:
    """``max_seqlen_k`` as a host bound on every key segment: an int in ``[ceil(Tkv / S), Tkv]``."""
    v = _check_int("max_seqlen_k", value, 0)
    if v > int(num_keys) or v * max(1, int(num_segments)) < int(num_keys):
        raise ValueError(
            f"max_seqlen_k={v} is not an upper bound of every key segment: {int(num_keys)} keys in {int(num_segments)} segments"
        )
    return v


def validate_dsa_indexer_nondet_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_causal_offsets: Optional[torch.Tensor],
    top_k: int,
    softmax_scale: float,
    ratio: int,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> tuple[int, int, int]:
    """Check the contract's shapes, dtypes, strides, alignment and scalars; return ``(T, Tkv, S)``.

    bfloat16 q / k only: float8_e4m3fn q / k raise with :data:`BF16_ONLY`
    (this package offers the BF16 form only).  Shape and dtype rules are
    checked before the device rule so that host tensors exercise them in
    CPU-only tests.  No host synchronization: the segment boundaries are not
    read.
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"q must be [T, {NUM_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}"
        )
    if torch.float8_e4m3fn in (q.dtype, k.dtype):
        raise ValueError(f"float8_e4m3fn q / k: {BF16_ONLY}")
    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bfloat16, got {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if k.ndim != 2 or k.shape[1] != HEAD_DIM:
        raise ValueError(f"k must be [Tkv, {HEAD_DIM}], got {tuple(k.shape)}")
    if k.dtype != torch.bfloat16:
        raise ValueError(f"k must be bfloat16, got {k.dtype}")
    if (
        k.stride(1) != 1
        or k.stride(0) < HEAD_DIM
        or (k.stride(0) * k.element_size()) % TMA_ALIGNMENT_BYTES
    ):
        raise ValueError(
            f"k needs a unit inner stride and a row stride >= {HEAD_DIM} elements that is a multiple of "
            f"{TMA_ALIGNMENT_BYTES // k.element_size()} elements, got strides {tuple(k.stride())}"
        )
    T = int(q.shape[0])
    if tuple(w.shape) != (T, NUM_HEADS):
        raise ValueError(f"w must be [T = {T}, {NUM_HEADS}], got {tuple(w.shape)}")
    if w.dtype != torch.float32:
        raise ValueError(f"w must be float32, got {w.dtype}")
    if not w.is_contiguous():
        raise ValueError("w must be contiguous")
    for name, t in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if t.ndim != 1 or t.numel() < 1:
            raise ValueError(f"{name} must be a one-dimensional [S + 1] tensor")
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must both have S + 1 entries")
    S = int(cu_seqlens_q.numel()) - 1
    if q_causal_offsets is not None:
        if q_causal_offsets.ndim != 1 or q_causal_offsets.numel() != S:
            raise ValueError(
                f"q_causal_offsets must be [S = {S}], got {tuple(q_causal_offsets.shape)}"
            )
        if q_causal_offsets.dtype != torch.int64:
            raise ValueError(
                f"q_causal_offsets must be int64, got {q_causal_offsets.dtype}"
            )
        if not q_causal_offsets.is_contiguous():
            raise ValueError("q_causal_offsets must be contiguous")
    _check_int("top_k", top_k, 1, MAX_TOP_K)
    _check_int("ratio", ratio, 1)
    if max_seqlen_q is not None:
        _check_int("max_seqlen_q", max_seqlen_q, 0)
    if (
        max_seqlen_k is not None
    ):  # a host bound on every key segment: validated before the device rule, read by the dispatch only
        _check_max_seqlen_k(max_seqlen_k, int(k.shape[0]), S)
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise ValueError(
            f"softmax_scale must be a positive finite number, got {softmax_scale!r}"
        )
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0.0:
        raise ValueError(
            f"softmax_scale must be a positive finite number, got {softmax_scale!r}"
        )
    device = q.device
    if device.type != "cuda":
        raise ValueError("inputs must be CUDA tensors")
    tensors = [k, w, cu_seqlens_q, cu_seqlens_k] + (
        [q_causal_offsets] if q_causal_offsets is not None else []
    )
    if any(t.device != device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if (
        (q.data_ptr() % TMA_ALIGNMENT_BYTES)
        or (k.data_ptr() % TMA_ALIGNMENT_BYTES)
        or (w.data_ptr() % TMA_ALIGNMENT_BYTES)
    ):
        raise ValueError(
            f"q, k and w must be {TMA_ALIGNMENT_BYTES}-byte aligned (TMA sources)"
        )
    return T, int(k.shape[0]), S


def _check_output(
    t: Optional[torch.Tensor], name: str, shape: tuple, dtype, device
) -> None:
    if t is None:
        return
    if tuple(t.shape) != shape or t.dtype != dtype:
        raise ValueError(
            f"{name} must be a {dtype} tensor of shape {shape}, got {tuple(t.shape)} {t.dtype}"
        )
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.device != device:
        raise ValueError(f"{name} must be on {device}")


# ---------------------------------------------------------------------------
# Program selection and workspace bound
# ---------------------------------------------------------------------------

_DEVICE_FACTS: dict[int, tuple[tuple[int, int], int]] = {}


def _device_index(device: Optional[torch.device]) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    device = torch.device(device)
    return int(
        device.index if device.index is not None else torch.cuda.current_device()
    )


def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    """``((major, minor), multi_processor_count)`` of ``cuda:index``: immutable for the life of the process, so queried
    once per device index; the index itself (``None`` / an index-less device = torch's current device) is resolved on
    every call by ``_device_index``."""
    facts = _DEVICE_FACTS.get(index)
    if facts is None:
        properties = torch.cuda.get_device_properties(index)
        facts = _DEVICE_FACTS[index] = (
            (int(properties.major), int(properties.minor)),
            int(properties.multi_processor_count),
        )
    return facts


def _num_sms(device: Optional[torch.device]) -> int:
    return _device_facts(_device_index(device))[1]


def plan_dsa_indexer_nondet_topk(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = DEFAULT_TOP_K,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ProgramChoice:
    """The host dispatch of a call from host-known integers (no tensor is read).

    ``arch`` defaults to the architecture of ``device`` (the current device) and
    ``num_sms`` to its SM count; both may be given explicitly for host-only use.
    The persistent grid is ``grid_ctas`` when given, else the SM count.  The
    plan's programs (``ProgramChoice.keys``) are the scan and, when the plan
    splits the key ranges (``n_split > 1``), the plain split merge.  ``dtype``:
    the q / k dtype, ``torch.bfloat16`` (the only form of this package;
    float8_e4m3fn raises ``ValueError`` with :data:`BF16_ONLY`).
    """
    _check_int("num_queries", num_queries, 0)
    _check_int("num_keys", num_keys, 0)
    _check_int("num_segments", num_segments, 0)
    _check_int("top_k", top_k, 1, MAX_TOP_K)
    _check_int("ratio", ratio, 1)
    _check_bf16(dtype)
    if arch is None:
        arch = arch_for(device)
        if arch is None:
            raise ValueError(
                "DSA indexer top-k requires compute capability 10.0, 10.3 or 10.7"
            )
    policy = policy_for(arch)
    if grid_ctas is not None:
        grid = _check_int("grid_ctas", grid_ctas, 1)
    elif num_sms is not None:
        grid = _check_int("num_sms", num_sms, 1)
    else:
        grid = _num_sms(device)
    if candidate_multiplier is not None:
        _check_int("candidate_multiplier", candidate_multiplier, 1)
    if sample_tiles_max is not None:
        _check_int("sample_tiles_max", sample_tiles_max, 0, 256)
    if sample_shift_permille is not None:
        _check_int(
            "sample_shift_permille",
            sample_shift_permille,
            *SAMPLE_SHIFT_PERMILLE_BOUNDS,
        )
    if far_sample is not None:
        _check_int("far_sample", far_sample, 0, 1)
    if check_period is not None:
        _check_int("check_period", check_period, 1)
    if max_seqlen_k is not None:
        _check_max_seqlen_k(max_seqlen_k, int(num_keys), int(num_segments))
    call: dict[str, Any] = dict(
        num_queries=int(num_queries),
        num_keys=int(num_keys),
        num_segments=int(num_segments),
        ratio=int(ratio),
        top_k=int(top_k),
        grid=grid,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        max_seqlen_k=max_seqlen_k,
        scan_keys=PROGRAM_KEYS.get(arch, {}),
    )
    return select_program(policy, **call)


def cake_dsa_indexer_nondet_train_topk_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = DEFAULT_TOP_K,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Explicit workspace bound in bytes of one call's geometry on ``device``.

    ``grid_ctas x queries_per_unit x candidate_capacity(top_k) x entry_bytes``
    for the dispatched unit geometry and capacity multiplier (``grid_ctas`` =
    the SM count, made even for the CTA-pair program), plus the split staging
    ``T x n_split x top_k x entry_bytes`` when the dispatch splits the key
    ranges.  The dispatch, and with it the bound, depends on ``T``, ``Tkv``,
    ``S``, ``ratio`` and ``top_k``; prepare a call's buffer for its own geometry
    (or for the largest bound over the geometries it will serve).  The bound
    covers both sampling modes: where the record's ``l6_solo_rule``,
    ``l6_solo_unsampled_max_queries``, ``split_wave_scope`` or ``wide_pair_snake_only`` gives a call with the
    sampled first threshold off (``sample_tiles_max=0``) another unit
    geometry, pair or split than the sampled default, the larger of the two
    plans.  ``dtype`` is the q / k dtype: ``torch.bfloat16``, the only form of
    this package (float8_e4m3fn raises ``ValueError`` with :data:`BF16_ONLY`).
    The bound is the deterministic package's for the same call, and computing
    it never consults torch's deterministic algorithms setting.
    """
    return max(
        int(
            plan_dsa_indexer_nondet_topk(
                num_queries,
                num_keys,
                num_segments,
                top_k=top_k,
                ratio=ratio,
                device=device,
                arch=arch,
                grid_ctas=grid_ctas,
                num_sms=num_sms,
                candidate_multiplier=candidate_multiplier,
                sample_tiles_max=sample_tiles_max,
                dtype=dtype,
            ).workspace_bytes
        )
        for sample_tiles_max in (None, 0)
    )


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    role: str
    program: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.entry(*self.arguments)


def bind_program(
    program: str,
    arch: str,
    role: str,
    values: dict[str, Any],
    grid: tuple[int, int, int],
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``role`` and load ``program`` for ``arch``.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.
    """
    if role not in STAGES:
        raise ValueError(f"role must be one of {STAGES}, got {role!r}")
    if PROGRAMS[program]["role"] != role:
        raise ValueError(
            f"program {program!r} is a {PROGRAMS[program]['role']} program, not {role!r}"
        )
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in ARG_PLANS[role]:
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif key in values and values[key] is not None:
            arguments.append(values[key])
        else:
            raise KeyError(
                f"generated program {program!r} ({role}) expects argument {name!r} ({kind}); the host binding "
                f"provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    entry = getattr(load_program(program, arch), FFI_ENTRY)
    return _Launch(role, program, entry, tuple(arguments), grid)


_FFI_DEVICES: dict[int, Any] = {}
_EMPTY_REGIONS: dict[int, Any] = {}
_TICKET_PLACEHOLDERS: dict[int, Any] = {}
_QUIET_SCAN_ARGS: dict[int, dict[str, Any]] = {}
_UNREAD_PLACEHOLDERS: dict[tuple[int, torch.dtype], torch.Tensor] = {}
# the ticket block a ``TicketWord`` argument points at: the ticket word (the low u32 of its first 8 bytes) and padding
TICKET_BLOCK_BYTES = 16
# the bytes a binding's ``zero_region`` zeroes to restore the ticket word of a block
TICKET_WORD_ZERO_BYTES = 8


def _empty_region(index: int) -> Any:
    """The scan's empty ``zero_region`` on device ``index`` (its binding zeroes nothing): one ``tvm_ffi.Tensor``
    per device, made on first use, so a call that zeroes no tail neither makes a view nor converts one per call."""
    region = _EMPTY_REGIONS.get(index)
    if region is None:
        region = _EMPTY_REGIONS[index] = tvm_ffi.from_dlpack(
            torch.empty(0, dtype=torch.uint8, device=f"cuda:{index}")
        )
    return region


def _ticket_placeholder(index: int) -> Any:
    """The ``TicketWord`` of a scan launch that never reads it (a program drawing from the workspace's tail, a
    ``static_scan_kinds`` program, a static call): one zeroed 16-byte block per device, made on first use and never
    written (a program that reads its ``TicketWord`` never gets it)."""
    block = _TICKET_PLACEHOLDERS.get(index)
    if block is None:
        block = _TICKET_PLACEHOLDERS[index] = tvm_ffi.from_dlpack(
            torch.zeros(
                TICKET_BLOCK_BYTES, dtype=torch.uint8, device=f"cuda:{index}"
            ).view(torch.uint32)
        )
    return block


def _quiet_scan_args(index: int) -> dict[str, Any]:
    """The ticket arguments of a scan launch that zeroes nothing and draws from no block of its own (a static call, a
    ``static_scan_kinds`` program, a program drawing from a tail it does not share): the device's empty
    ``zero_region`` and ``TicketWord`` placeholder, one dict per device, so such a call adds them with one lookup."""
    args = _QUIET_SCAN_ARGS.get(index)
    if args is None:
        args = _QUIET_SCAN_ARGS[index] = {
            "zero_region": _empty_region(index),
            "TicketWord": _ticket_placeholder(index),
        }
    return args


def _unread_placeholder(index: int, dtype: torch.dtype) -> torch.Tensor:
    """A one-element ``dtype`` tensor on device ``index`` for an argument no launched program dereferences: one per
    device and dtype, allocated without a device operation and never written, so such a call launches nothing before
    its scan.  Made during a CUDA graph capture, it is the call's own (the capture's pool) and not kept."""
    holder = _UNREAD_PLACEHOLDERS.get((index, dtype))
    if holder is None:
        holder = torch.empty(1, dtype=dtype, device=f"cuda:{index}")
        if not torch.cuda.is_current_stream_capturing():
            _UNREAD_PLACEHOLDERS[(index, dtype)] = holder
    return holder


def _offsets_placeholder(index: int) -> torch.Tensor:
    """The ``q_offsets`` of a call without causal offsets (``has_offsets = 0``: no program dereferences it)."""
    return _unread_placeholder(index, torch.int64)


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``."""
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = (
        getter(index)
        if getter is not None
        else torch.cuda.current_stream(index).cuda_stream
    )
    return tvm_ffi.use_raw_stream(device, raw)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DSAIndexerNondetRunner:
    """The prepared launches of one tensor binding.

    ``__call__()`` runs the dispatched scan program and, when the dispatch
    splits the key ranges, the plain split merge on torch's current stream and
    returns ``(indices, scores)``: per row the selected entries and the
    padding, in no specified slot order (no launch orders a row).  No launch
    allocates or synchronizes; capture into a CUDA graph belongs to the
    caller.  Prepare a new runner when a shape, dtype, stride or tensor
    binding changes; values may change freely.  Every run that launches
    consults torch's deterministic algorithms setting before its first launch
    (:func:`_alert_nondeterministic`; host only, so a captured graph's replays
    do not); the eager entry's one-shot runner was checked before it allocated.
    """

    arch: str
    abi: str
    choice: ProgramChoice
    programs: dict[str, str]  # launched role -> registered program name
    tensors: dict[str, torch.Tensor] = field(repr=False)
    scalars: dict[str, Any] = field(repr=False)
    launches: tuple[_Launch, ...] = field(repr=False)
    stages: tuple[str, ...]  # launched roles in order
    device_index: int = 0
    workspace: torch.Tensor = field(default=None, repr=False)
    workspace_bytes: int = 0
    grid: tuple[int, int, int] = (1, 1, 1)
    # the workspace's ticket tail on the ticket architectures (``TICKET_TAIL_ARCHES``;
    # None for a program of the record's ``static_scan_kinds``).  Invariant: the ticket word reads 0 when a scan launches.  Every scan
    # launch that completes leaves the word at 0 (the kernel's end-of-launch reset, which
    # kernel-tests check as the output ``ticket_word`` on every ticket-drawing config).
    # Backend-owned workspace (no ``workspace_buffer`` passed to prepare): nothing else writes
    # the tail, so it is zeroed once, when the runner is prepared; ``run()`` issues no memset
    # and a CUDA graph capturing it carries none.  Caller-owned workspace (``_tail_shared``):
    # the buffer may be written between two runs by another runner of a different geometry
    # (its candidate region covers this tail) or by any other kernel the caller lends it to,
    # so the scan launch passes the tail as its ``zero_region``, which the scan binding zeroes
    # with ``cudaMemsetAsync`` on the launch stream right before every scan launch, in the same
    # host call (no host work between the two submissions); a graph capturing it carries that
    # memset node.  Static calls (``_tail_static``): where the
    # shape alone bounds the work units by the launched grid (``ticket_static_path``), every
    # scan launch takes the static fast path, which gives each CTA its unit by index and never
    # reads or writes the ticket word, so neither prepare nor a run zeroes the tail (no
    # memset on either workspace kind; the scan's ``zero_region`` is empty).  A ``run()`` that raised while launching may have
    # left the word non-zero: it marks the tail dirty and the next ``run()`` zeroes it
    # first (static or not).  The view (``ticket_tail``) is made on first use: a static call
    # that never zeroes its tail makes none on the launch path.
    # A scan program that reads its ``TicketWord`` argument on this architecture (``reads_ticket_word``) has no
    # ticket tail: a drawing call gets a 16-byte block of the runner's own (``ticket_block``), which no other work
    # writes, zeroed once here; every completed scan launch leaves it 0, so neither kind of workspace adds a memset
    # to ``run()``.  After a run that raised while launching, the next ``run()`` launches the scan through a binding
    # call whose ``zero_region`` is the block's ticket word (8 bytes), on the run's stream.  The eager entry draws from
    # the first 16 bytes of the workspace's tail instead, which its scan binding zeroes (8 bytes) right before the
    # launch.  Any other scan launch (static, or a program drawing from the tail) gets the device's zeroed
    # placeholder, which it never reads.  The :lsf copy of a TicketWord-reading program (sm_107a's l6 ticket
    # programs) still keeps its longest-first segment order in the tail's bytes behind the ticket word, which every
    # launch writes before it reads them, so stale bytes there never matter and nothing zeroes them.
    # byte offset of the tail in the workspace; -1: no tail
    _tail_at: int = field(default=-1, repr=False)
    _tail_view: Optional[torch.Tensor] = field(default=None, repr=False)
    _tail_shared: bool = field(default=False, repr=False)
    _tail_static: bool = field(default=False, repr=False)
    _tail_dirty: bool = field(default=False, repr=False)
    # the bytes a TicketWord-reading drawing scan draws from (uint8); None: the call draws from no ticket block
    _ticket_block: Optional[torch.Tensor] = field(default=None, repr=False)
    # the scan launch that zeroes the ticket block's word first (a prepared TicketWord runner after a failed run)
    _ticket_restore: Optional[_Launch] = field(default=None, repr=False)
    # the entry name every run alerts under (torch's deterministic algorithms); None: no run alerts (nothing launches,
    # or the eager entry's one-shot runner, which alerted before it allocated)
    _alert_entry: Optional[str] = field(default=None, repr=False)

    @property
    def ticket_tail(self) -> Optional[torch.Tensor]:
        if self._tail_at < 0:
            return None
        if self._tail_view is None:
            self._tail_view = self.workspace[
                self._tail_at : self._tail_at + TICKET_TAIL_BYTES
            ]
        return self._tail_view

    @property
    def ticket_block(self) -> Optional[torch.Tensor]:
        """The 16 bytes the scan draws its tickets from where its program reads ``TicketWord`` and the call draws."""
        return self._ticket_block

    @property
    def indices(self) -> torch.Tensor:
        return self.tensors["Indices"]

    @property
    def scores(self) -> torch.Tensor:
        return self.tensors["Scores"]

    @property
    def num_queries(self) -> int:
        return self.choice.num_queries

    @property
    def num_keys(self) -> int:
        return self.choice.num_keys

    @property
    def top_k(self) -> int:
        return self.choice.top_k

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._alert_entry is not None:
            _alert_nondeterministic(self._alert_entry)
        with _ffi_stream_context(self.device_index):
            launches = self.launches
            if self._tail_dirty:
                # a previous run raised while launching: restore the ticket word before the scan
                if self._ticket_restore is not None:
                    # the scan binding zeroes the block's word on this stream right before the scan launch
                    launches = (self._ticket_restore, *launches[1:])
                elif self._tail_at >= 0:
                    # (a caller-owned tail is zeroed by the scan binding anyway)
                    self.ticket_tail.zero_()
                self._tail_dirty = False
            try:
                for launch in launches:
                    launch()
            except BaseException:
                self._tail_dirty = (
                    self._tail_at >= 0 or self._ticket_restore is not None
                )
                raise
        return self.tensors["Indices"], self.tensors["Scores"]

    __call__ = run


def _padded_outputs(
    num_queries: int,
    top_k: int,
    device: torch.device,
    indices: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outputs of a call without keys: all padding (two fills, no synchronization)."""
    if indices is None:
        indices = torch.empty((num_queries, top_k), dtype=torch.int32, device=device)
    if scores is None:
        scores = torch.empty((num_queries, top_k), dtype=torch.float32, device=device)
    indices.fill_(PAD_ID)
    scores.fill_(PAD_SCORE)
    return indices, scores


def _prepare_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    backend: str = "cake",
    one_shot: bool = False,
) -> DSAIndexerNondetRunner:
    """:func:`prepare_dsa_indexer_nondet_topk`; ``one_shot``: the runner runs once (the eager entry), so a scan program
    that reads its ``TicketWord`` draws from the first bytes of the workspace's tail, which its scan binding zeroes,
    instead of a block of its own."""
    if backend != "cake":
        raise ValueError("DSA indexer top-k supports backend='cake'")
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    num_queries, num_keys, num_segments = validate_dsa_indexer_nondet_inputs(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        q_causal_offsets,
        top_k,
        softmax_scale,
        ratio,
        max_seqlen_q,
        max_seqlen_k,
    )
    device = q.device
    for name, tensor in (
        ("workspace_buffer", workspace_buffer),
        ("indices", indices),
        ("scores", scores),
    ):
        if tensor is not None and (not tensor.is_cuda or tensor.device != device):
            raise ValueError(f"{name} must be on {device}")
    _check_output(indices, "indices", (num_queries, int(top_k)), torch.int32, device)
    _check_output(scores, "scores", (num_queries, int(top_k)), torch.float32, device)
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "DSA indexer top-k requires compute capability 10.0, 10.3 or 10.7"
        )
    abi = registry_abi()
    device_index = _device_index(device)
    choice = plan_dsa_indexer_nondet_topk(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        device=device,
        arch=arch,
        grid_ctas=grid_ctas,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        max_seqlen_k=max_seqlen_k,
        dtype=q.dtype,
    )
    # every program of the call must be registered before anything is allocated (fail closed by name): the scan and, on
    # a split call, the plain merge (no program orders a row)
    programs = {"scan": program_for(arch, choice.scan_key)}
    if choice.merge_key is not None:
        programs["merge"] = program_for(arch, choice.merge_key)
    workspace_bytes = int(choice.workspace_bytes)
    caller_owned_workspace = workspace_buffer is not None
    if caller_owned_workspace:
        # the caller's buffer is checked before the alert (a view, no allocation)
        flat = workspace_buffer.view(-1).view(torch.uint8)
        if flat.numel() < workspace_bytes:
            raise ValueError(
                f"workspace_buffer needs {workspace_bytes} bytes, got {flat.numel()}"
            )
        if flat.data_ptr() % 8:
            raise ValueError("workspace_buffer must be 8-byte aligned")
    if one_shot and num_queries > 0 and num_keys > 0:
        # an eager call that launches: PyTorch's alert after every check above, before its first allocation or launch
        # (a prepared runner alerts on every run instead: ``DSAIndexerNondetRunner.run``)
        _alert_nondeterministic(TOPK_ENTRY)
    # the ticket-tail decision (see the comment where the tail is passed) is host arithmetic only: it is made here,
    # before the call's first device operation, so an eager call does not hold the stream for it
    tail_at, tail_static, drawing, reads_word = -1, False, False, False
    tail_skips = ticket_tail_skips(arch)
    if (
        num_queries > 0
        and num_keys > 0
        and tail_skips is not None
        and choice.physical_kind not in tail_skips
    ):
        tail_static = ticket_static_path(choice)
        drawing = not tail_static
        # the scan program draws from its TicketWord argument instead of the tail (registry ``ticket_arg_arches``)
        reads_word = reads_ticket_word(programs["scan"], arch)
        if not reads_word:
            tail_at = workspace_bytes - TICKET_TAIL_BYTES
    if not caller_owned_workspace:
        flat = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    cand = flat[:workspace_bytes].view(torch.int64)

    t: dict[str, torch.Tensor] = {
        "Q": q.view(num_queries * NUM_HEADS, HEAD_DIM),
        "K": k,
        "W": w,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        # the device's never-dereferenced placeholder when no offsets are supplied (has_offsets = 0): no fill per call
        "q_offsets": q_causal_offsets
        if q_causal_offsets is not None
        else _offsets_placeholder(device_index),
        # the BF16 programs never read the per-key scales: the device's never-dereferenced placeholder
        "K_scale": _unread_placeholder(device_index, torch.float32),
        "Indices": indices
        if indices is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.int32, device=device),
        "Scores": scores
        if scores is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.float32, device=device),
        "Cand": cand,
        # the split staging [T, n_split, top_k] lives behind the persistent candidate buffers (merge stage only)
        "Staging": cand[
            choice.scan_entries : choice.scan_entries + choice.staging_entries
        ]
        if choice.split
        else None,
    }
    scalars: dict[str, Any] = dict(
        num_segments=num_segments,
        top_k=int(top_k),
        ratio=int(ratio),
        has_offsets=int(q_causal_offsets is not None),
        cand_cap=int(choice.cand_cap),
        first_cap=int(
            choice.first_cap
        ),  # regular trigger only: the first selection fires at the capacity rule
        sample_tiles_max=int(choice.sample_tiles_max),
        sample_shift_permille=int(choice.sample_shift_permille),
        far_sample=int(choice.far_sample),
        check_period=int(choice.check_period),
        grid_ctas=int(choice.grid_ctas),
        softmax_scale=float(softmax_scale),
        n_split=int(choice.n_split),
        num_queries=num_queries,
        num_keys=num_keys,
    )
    grid = (int(choice.grid_ctas), 1, 1)
    row_grid = (max(1, num_queries), 1, 1)
    launches: list[_Launch] = []
    stages: list[str] = []
    # the ticket tail the scan binding zeroes right before every scan launch (``zero_region``): a caller-owned tail
    # the call draws tickets from; any other call passes the device's empty region (nothing zeroed, no per-call view)
    shared_tail = caller_owned_workspace and tail_at >= 0 and not tail_static
    # the scan's TicketWord: a program that reads it and draws gets a block no other work writes -- the runner's own
    # (zeroed here, once) or, for the eager entry's one launch, the first bytes of the workspace's tail, whose ticket
    # word the scan binding zeroes right before the launch; any other launch gets the device's placeholder.  A call
    # that neither shares its tail nor draws from a block (every static call) passes the device's quiet pair as is.
    ticket_block: Optional[torch.Tensor] = None
    scan_ticket = _quiet_scan_args(device_index)
    if shared_tail:
        scan_ticket = {
            "zero_region": flat[tail_at:workspace_bytes],
            "TicketWord": scan_ticket["TicketWord"],
        }
    elif reads_word and drawing:
        zero_region = scan_ticket["zero_region"]
        if one_shot:
            block_at = workspace_bytes - TICKET_TAIL_BYTES
            ticket_block = flat[block_at : block_at + TICKET_BLOCK_BYTES]
            zero_region = ticket_block[:TICKET_WORD_ZERO_BYTES]
        else:
            ticket_block = torch.zeros(
                TICKET_BLOCK_BYTES, dtype=torch.uint8, device=device
            )
        ticket_word = ticket_block.view(torch.uint32)
        if ticket_word is _ticket_placeholder(device_index):
            raise RuntimeError(
                f"{programs['scan']} ({arch}) draws from its TicketWord argument but was given the shared placeholder"
            )
        scan_ticket = {"zero_region": zero_region, "TicketWord": ticket_word}
    ticket_restore: Optional[_Launch] = None
    if num_queries > 0 and num_keys > 0:
        values: dict[str, Any] = dict(t)
        values.update(scalars)
        values.update(scan_ticket)
        launches.append(bind_program(programs["scan"], arch, "scan", values, grid))
        stages.append("scan")
        if ticket_block is not None and not one_shot:
            # the scan launch of the run after a failed one: its binding zeroes the block's ticket word first
            ticket_restore = bind_program(
                programs["scan"],
                arch,
                "scan",
                {**values, "zero_region": ticket_block[:TICKET_WORD_ZERO_BYTES]},
                grid,
            )
        if "merge" in programs:
            launches.append(
                bind_program(programs["merge"], arch, "merge", values, row_grid)
            )
            stages.append("merge")
    elif num_queries > 0:
        _padded_outputs(num_queries, int(top_k), device, t["Indices"], t["Scores"])
    # on the ticket architectures the scan program draws every work unit from the workspace's
    # ticket tail (behind the candidate buffers and the staging); every completed scan launch
    # resets the word to 0.  Backend-owned workspace: zeroed here once, the only memset, a run
    # adds none.  Caller-owned ``workspace_buffer``: the buffer may be written between runs by
    # others (another runner's candidate region, any kernel it is lent to), so nothing is zeroed
    # here; the scan launch passes the tail as its ``zero_region`` and the scan binding zeroes it
    # right before every scan launch (see ``DSAIndexerNondetRunner.ticket_tail``).
    # A static call (``ticket_static_path``: the shape bounds the work units by the launched
    # grid, so every scan launch takes the static fast path and never touches the word) is
    # zeroed neither here nor per run.  A program of the record's ``static_scan_kinds``
    # (``policy_for(arch)``) keeps the static enumeration on every call: no tail view, nothing zeroed.
    # A program that reads its TicketWord argument (``reads_ticket_word``) has no tail here: it draws from its
    # ticket block (above), so neither a caller-owned nor a backend-owned workspace is zeroed for it (its :lsf copy
    # writes the segment order behind the tail's ticket word before it reads it).
    # (``tail_at`` / ``tail_static`` were decided above, before the first device operation; a call launches
    # iff it has queries and keys.)
    tail_view = scan_ticket["zero_region"] if shared_tail else None
    if tail_at >= 0 and not tail_static and not caller_owned_workspace:
        tail_view = flat[tail_at:workspace_bytes]
        tail_view.zero_()
    return DSAIndexerNondetRunner(
        arch=arch,
        abi=abi,
        choice=choice,
        programs=programs,
        tensors=t,
        scalars=scalars,
        launches=tuple(launches),
        stages=tuple(stages),
        device_index=device_index,
        workspace=flat,
        workspace_bytes=workspace_bytes,
        grid=grid,
        _tail_at=tail_at,
        _tail_view=tail_view,
        _tail_shared=caller_owned_workspace and tail_at >= 0,
        _tail_static=tail_static,
        _ticket_block=ticket_block,
        _ticket_restore=ticket_restore,
        _alert_entry=RUN_ENTRY if launches and not one_shot else None,
    )


def prepare_dsa_indexer_nondet_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    backend: str = "cake",
) -> DSAIndexerNondetRunner:
    """Validate one binding, decide its programs and prepare its launches.

    Missing outputs and the workspace are allocated here (the only allocations
    of the backend).  Pass ``workspace_buffer`` of at least
    :func:`cake_dsa_indexer_nondet_train_topk_workspace_size` bytes for the call's geometry (same
    ``grid_ctas`` / ``candidate_multiplier``) to reuse storage across calls.
    A caller-owned buffer may be written by other runners or kernels between
    two runs of the returned runner, so where the scan program draws from the
    buffer's ticket tail the scan binding zeroes it before every run (one small
    memset on the current stream, captured into a CUDA graph with the
    launches); the backend-owned workspace is zeroed once, here.  A scan
    program that reads its ``TicketWord`` argument draws from a 16-byte block
    of the runner's own instead, zeroed once, here, on either workspace.  ``indices`` / ``scores`` may
    be caller-owned ``[T, top_k]`` int32 / float32 tensors.

    The keyword-only knobs ``grid_ctas`` (persistent CTA count; default: the SM
    count), ``candidate_multiplier`` (candidate buffer per query in multiples
    of ``top_k``), ``check_period`` (key tiles between selection-trigger
    checks), ``sample_tiles_max`` / ``sample_shift_permille`` (sampled first
    threshold; ``0`` disables sampling and sizes the candidate buffers by the
    record's unsampled capacity values; the shift defaults to the record's
    value for the call's ``top_k``, ``sample_shift_rule`` where the record sets
    one) and ``far_sample`` (``1``: the sampled first threshold of the
    ascending-tile-order scan programs ranks the far sample tiles only,
    leaving out the sample tile nearest the diagonal; ``0``: every sample
    tile; the descending-order and wide programs ignore it) change only the
    internal partition and scheduling of the work: every admissible value
    selects the same set of ids and score bits per row (the slot order may
    differ).  ``grid_ctas`` also enters the key-range split rule
    (``cake_policy.DispatchPolicy.n_split_for``: a grid of more CTAs per query
    unit splits longer key ranges into more parts).  ``T == 0`` returns a
    runner whose launches are empty; a call without keys (``Tkv == 0``) fills
    the outputs with padding instead of launching.

    Preparing never consults ``torch.use_deterministic_algorithms``; every run
    of a runner that launches programs (``T > 0`` and ``Tkv > 0``) does, before
    its first launch: PyTorch's ``RuntimeError`` (``warn_only=True``: a
    ``UserWarning``, then the run launches).  float8_e4m3fn ``q`` / ``k`` raise
    ``ValueError`` with :data:`BF16_ONLY`.
    """
    return _prepare_runner(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        top_k=top_k,
        softmax_scale=softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        workspace_buffer=workspace_buffer,
        indices=indices,
        scores=scores,
        grid_ctas=grid_ctas,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        backend=backend,
    )


def cake_dsa_indexer_nondet_train_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager entry point: prepare a binding and run it once; returns ``(indices, scores)``.

    See :func:`prepare_dsa_indexer_nondet_topk` for the arguments.  The call allocates
    the outputs (unless given) and the workspace (unless given) through the
    caching allocator and performs no host synchronization, so it may be
    captured into a CUDA graph like any allocation-free launch sequence.  The
    ticket word the scan draws from is zeroed once per call by the scan
    binding, right before the scan launch: the ticket tail of a caller-owned
    ``workspace_buffer``, or the first 8 bytes of the tail of either workspace
    where the scan program reads its ``TicketWord`` argument.  Nothing else is
    launched before the scan: a call without ``q_causal_offsets`` passes the
    device's offsets placeholder, which no program reads.  Under
    ``torch.use_deterministic_algorithms(True)`` a call with queries and keys
    raises PyTorch's ``RuntimeError`` before it allocates or launches anything
    (``warn_only=True``: a ``UserWarning``, then the call runs).
    """
    runner = _prepare_runner(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        top_k=top_k,
        softmax_scale=softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        workspace_buffer=workspace_buffer,
        indices=indices,
        scores=scores,
        grid_ctas=grid_ctas,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        backend=backend,
        one_shot=True,
    )
    return runner.run()


__all__ = [
    "ABI_CONTRACT",
    "BF16_ONLY",
    "CONTRACT_ALIASES",
    "CONTRACT_SCALARS",
    "CONTRACT_TENSORS",
    "DEFAULT_TOP_K",
    "HEAD_DIM",
    "MAX_TOP_K",
    "MERGE_SCALARS",
    "NUM_HEADS",
    "PAD_ID",
    "PAD_SCORE",
    "SCAN_SCALARS",
    "SUPPORTED_ABIS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "ZERO_SIGN_POLICIES",
    "DSAIndexerNondetRunner",
    "DispatchPolicy",
    "ProgramChoice",
    "arch_for",
    "bind_program",
    "default_softmax_scale",
    "cake_dsa_indexer_nondet_train_topk",
    "cake_dsa_indexer_nondet_train_topk_workspace_size",
    "effective_offsets",
    "generated_program_available",
    "plan_dsa_indexer_nondet_topk",
    "policy_for",
    "prepare_dsa_indexer_nondet_topk",
    "program_for",
    "reads_ticket_word",
    "record_zero_sign_policy",
    "registry_abi",
    "validate_dsa_indexer_nondet_inputs",
    "visible_key_count",
    "visible_key_counts",
]
