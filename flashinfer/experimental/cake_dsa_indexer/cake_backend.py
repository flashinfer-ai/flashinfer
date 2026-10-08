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

Cake backend: fused DSA indexer scoring + deterministic exact top-k selection
for training on SM100 / SM103 / SM107 (flashinfer-ai/flashinfer#5676; GLM-5.2
geometry: 32 indexer heads, head dimension 128, top-k 2048).

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
  sizes the rank finalize's bitmap pool and so selects its program variant
  (``rank_seg_window``) -- never a result; a value above ``Tkv`` or below
  ``ceil(Tkv / S)`` is rejected.
* Outputs ``indices [T, top_k]`` int32 segment-local key ids and ``scores [T,
  top_k]`` FP32, both in ascending id order with the tail padded by ``-1`` /
  ``-inf``; an entirely empty row holds only padding.
* Score ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] *
  k[j, d])`` with FP32 products, accumulation and score arithmetic (no BF16
  intermediate rounding).  The FP32 reduction scheme of the generated program
  is documented in the package README and is fixed: the result does not depend
  on the internal query / key partition.
* Visibility: key ``j`` is visible to the query at segment-local position ``u``
  iff ``0 <= j < Lk`` and ``j < floor((offset + u + 1) / ratio)``; ``offset`` is
  ``q_causal_offsets[s]`` when supplied, else ``Lk - Lq`` for ``ratio == 1`` and
  ``0`` otherwise.  Keys of other segments or outside the visible prefix are
  never selected; negative offsets, empty segments and rows without visible
  keys are defined (all padding).
* Selection: the exact global top ``min(top_k, visible)`` keys per row, ranked
  by score descending then key id descending (``+0.0 == -0.0`` for ranking);
  the returned score bits are the computed FP32 bits.
* Repeatable: identical inputs give identical ids and score bits; the program
  the host dispatches (unit geometry, key-range split, CTA pair, unroll, unit
  order, sampled threshold; the CUB block radix sort or the prefix-popcount
  rank scatter as the finalize) and the partition knobs (``grid_ctas``,
  ``candidate_multiplier``, ``check_period``, the sampled-threshold knobs, the
  finalize program) never change a result.
* Memory: no ``[T, Tkv]`` score matrix.  The workspace is
  :func:`dsa_indexer_workspace_size` bytes for the call's geometry -- the
  persistent candidate buffers ``grid x queries_per_unit x
  candidate_capacity(top_k) x entry_bytes`` plus, when the dispatch splits the
  key ranges, the staging ``T x n_split x top_k x entry_bytes`` -- separate
  from the inputs and the two outputs; no host synchronization inside the
  operator.
* Non-finite inputs (NaN / inf / overflow) are outside the normal domain: the
  operator returns with the output structure intact and no loop bound depends
  on a score value, so the kernels cannot hang.

Every allocation happens in :func:`prepare_dsa_indexer_topk`; the returned
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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

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
    TRACKING_ISSUE,
    load_program,
)
from .cake_policy import (
    MERGE_KEY,
    DispatchPolicy,
    ProgramChoice,
    finalize_grid,
    select_program,
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

# Host binding profile: the keyword set of the generated kernels (the names
# below are what the host provides; a kernel's argument plan selects from
# them).  Tensor kinds in the argument plan are ``buffer`` / ``tma_buffer``
# (the binding encodes the tensor map itself from the view it receives),
# scalars are ``parameter``; ``grid_x/y/z`` are ``grid``.
ABI_CONTRACT = "dsa_indexer_v2"
SUPPORTED_ABIS = (ABI_CONTRACT,)
CONTRACT_TENSORS = (
    "Q",  # q viewed as [T * 32, 128] (TMA source)
    "K",  # k [Tkv, 128], any row stride that is a multiple of 8 elements (TMA source)
    "W",  # w [T, 32] fp32 (TMA source)
    "cu_seqlens_q",
    "cu_seqlens_k",
    "q_offsets",  # q_causal_offsets, or a never-dereferenced int64 placeholder
    "Indices",
    "Scores",
    "Cand",  # the workspace viewed as packed int64 candidate entries (buffers, then the split staging)
    "Staging",  # the split staging [T, n_split, top_k] behind the candidate buffers (merge stage only)
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
    "check_period",
    "grid_ctas",
    "softmax_scale",
    "n_split",
)
MERGE_SCALARS = ("top_k", "n_split")
FINALIZE_SCALARS = ("top_k", "key_bits")  # the CUB block radix sort programs
FINALIZE_RANK_SCALARS = (
    "top_k",
    "num_segments",
)  # the prefix-popcount rank finalize programs (+ cu_seqlens_q / cu_seqlens_k)
CONTRACT_SCALARS = SCAN_SCALARS + ("key_bits",)
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


def registry_abi() -> str:
    if ABI not in SUPPORTED_ABIS:
        raise NotImplementedError(
            f"unsupported host binding profile {ABI!r} (this host binds {SUPPORTED_ABIS})"
        )
    return ABI


def policy_for(arch: str) -> DispatchPolicy:
    """The dispatch policy registered for ``arch``; raises when the architecture has no program."""
    record = POLICY.get(arch)
    if record is None:
        raise NotImplementedError(
            f"The generated DSA indexer top-k programs for {arch} are not registered in this checkout "
            f"(registered: {sorted(POLICY)}; see {TRACKING_ISSUE})"
        )
    return DispatchPolicy.from_record(record)


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
    """True when this checkout registers programs for ``device`` under the host binding profile this module binds."""
    arch = arch_for(device)
    if (
        arch is None
        or arch not in PROGRAM_KEYS
        or arch not in POLICY
        or ABI not in SUPPORTED_ABIS
    ):
        return False
    keys = PROGRAM_KEYS[arch]
    return (
        any(k.startswith("scan:") for k in keys)
        and MERGE_KEY in keys
        and any(k.startswith(("finalize:", "finalize_rank:")) for k in keys)
    )


def record_zero_sign_policy() -> str:
    """The documented zero-sign policy of the program's head reduction."""
    policy = NUMERICS.get("zero_sign_policy") if isinstance(NUMERICS, dict) else None
    if policy not in ZERO_SIGN_POLICIES:
        raise ValueError(
            f"registry declares numerics {NUMERICS!r}; zero_sign_policy must be one of {ZERO_SIGN_POLICIES}"
        )
    return str(policy)


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


def validate_dsa_indexer_inputs(
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

    Shape and dtype rules are checked before the device rule so that host
    tensors exercise them in CPU-only tests.  No host synchronization: the
    segment boundaries are not read.
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"q must be [T, {NUM_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}"
        )
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


def _outputs_aligned(tensors: Iterable[Optional[torch.Tensor]], align: int) -> bool:
    """True when every given output tensor's address is a multiple of ``align`` bytes (None = allocated here, aligned by the
    caching allocator): the staged rank finalize's ``cp.async.bulk`` row I/O (lever FRB) needs ``rank_bulk_align_bytes``."""
    align = int(align)
    return all(t is None or int(t.data_ptr()) % align == 0 for t in tensors)


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


def plan_dsa_indexer_topk(
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
    finalize_threads: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    outputs_aligned: bool = True,
) -> ProgramChoice:
    """The host dispatch of a call from host-known integers (no tensor is read).

    ``arch`` defaults to the architecture of ``device`` (the current device) and
    ``num_sms`` to its SM count; both may be given explicitly for host-only use.
    The persistent grid is ``grid_ctas`` when given, else the SM count.
    ``outputs_aligned`` (default True: outputs the backend allocates) says whether
    the ``indices`` / ``scores`` addresses are multiples of the policy's
    ``rank_bulk_align_bytes``; the staged rank finalize's bulk row I/O (lever FRB)
    is dispatched only then, else its plain twin program (same results).
    """
    _check_int("num_queries", num_queries, 0)
    _check_int("num_keys", num_keys, 0)
    _check_int("num_segments", num_segments, 0)
    _check_int("top_k", top_k, 1, MAX_TOP_K)
    _check_int("ratio", ratio, 1)
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
        _check_int("sample_shift_permille", sample_shift_permille, -1000, 10000)
    if check_period is not None:
        _check_int("check_period", check_period, 1)
    if finalize_threads is not None:
        _check_int("finalize_threads", finalize_threads, 1)
    if max_seqlen_k is not None:
        _check_max_seqlen_k(max_seqlen_k, int(num_keys), int(num_segments))
    return select_program(
        policy,
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
        finalize_threads=finalize_threads,
        max_seqlen_k=max_seqlen_k,
        outputs_aligned=bool(outputs_aligned),
    )


def dsa_indexer_workspace_size(
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
) -> int:
    """Explicit workspace bound in bytes of one call's geometry on ``device``.

    ``grid_ctas x queries_per_unit x candidate_capacity(top_k) x entry_bytes``
    for the dispatched unit geometry and capacity multiplier (``grid_ctas`` =
    the SM count, made even for the CTA-pair program), plus the split staging
    ``T x n_split x top_k x entry_bytes`` when the dispatch splits the key
    ranges.  The dispatch, and with it the bound, depends on ``T``, ``Tkv``,
    ``S``, ``ratio`` and ``top_k``; prepare a call's buffer for its own geometry
    (or for the largest bound over the geometries it will serve).
    """
    choice = plan_dsa_indexer_topk(
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
    )
    return int(choice.workspace_bytes)


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
class DSAIndexerRunner:
    """The prepared launches of one tensor binding.

    ``__call__()`` runs the dispatched scan program, the split merge when the
    dispatch splits the key ranges, and the finalize program (the CUB block
    radix sort, or the prefix-popcount rank scatter where the policy's
    ``finalize_rank_for`` names a bitmap window; stage role ``finalize`` /
    ``finalize_rank``) on torch's current stream and returns ``(indices,
    scores)``.  No launch allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype, stride or tensor binding changes; values
    may change freely.
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
        with _ffi_stream_context(self.device_index):
            for launch in self.launches:
                launch()
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


def prepare_dsa_indexer_topk(
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
    finalize_threads: Optional[int] = None,
    backend: str = "cake",
) -> DSAIndexerRunner:
    """Validate one binding, decide its programs and prepare its launches.

    Missing outputs and the workspace are allocated here (the only allocations
    of the backend).  Pass ``workspace_buffer`` of at least
    :func:`dsa_indexer_workspace_size` bytes for the call's geometry (same
    ``grid_ctas`` / ``candidate_multiplier``) to reuse storage across calls;
    ``indices`` / ``scores`` may be caller-owned ``[T, top_k]`` int32 / float32
    tensors.

    The keyword-only knobs ``grid_ctas`` (persistent CTA count; default: the SM
    count), ``candidate_multiplier`` (candidate buffer per query in multiples
    of ``top_k``), ``check_period`` (key tiles between selection-trigger
    checks), ``sample_tiles_max`` / ``sample_shift_permille`` (sampled first
    threshold; ``0`` disables sampling) and ``finalize_threads`` (an admissible
    registered finalize program) change only the internal partition and
    scheduling of the work: results are bitwise identical for every admissible
    value.  ``T == 0`` returns a runner whose launches are empty; a call
    without keys (``Tkv == 0``) fills the outputs with padding instead of
    launching.
    """
    if backend != "cake":
        raise ValueError("DSA indexer top-k supports backend='cake'")
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    num_queries, num_keys, num_segments = validate_dsa_indexer_inputs(
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
    # lever FRB: the staged rank finalize's bulk row I/O needs output addresses on the policy's 16-B granule; caller-owned
    # outputs may be arbitrary views (the plain twin program is registered alongside), the backend's own allocations are aligned
    outputs_aligned = _outputs_aligned(
        (indices, scores), policy_for(arch).rank_bulk_align_bytes
    )
    choice = plan_dsa_indexer_topk(
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
        finalize_threads=finalize_threads,
        max_seqlen_k=max_seqlen_k,
        outputs_aligned=outputs_aligned,
    )
    # every program of the call must be registered before anything is allocated (fail closed by name)
    fin_role = choice.finalize_role
    programs = {
        "scan": program_for(arch, choice.scan_key),
        fin_role: program_for(arch, choice.finalize_key),
    }
    if choice.merge_key is not None:
        programs["merge"] = program_for(arch, choice.merge_key)
    workspace_bytes = int(choice.workspace_bytes)
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < workspace_bytes:
        raise ValueError(
            f"workspace_buffer needs {workspace_bytes} bytes, got {flat.numel()}"
        )
    if flat.data_ptr() % 8:
        raise ValueError("workspace_buffer must be 8-byte aligned")
    cand = flat[:workspace_bytes].view(torch.int64)

    t: dict[str, torch.Tensor] = {
        "Q": q.view(num_queries * NUM_HEADS, HEAD_DIM),
        "K": k,
        "W": w,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        # a never-dereferenced placeholder when no offsets are supplied (has_offsets = 0)
        "q_offsets": q_causal_offsets
        if q_causal_offsets is not None
        else torch.zeros(1, dtype=torch.int64, device=device),
        "Indices": indices
        if indices is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.int32, device=device),
        "Scores": scores
        if scores is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.float32, device=device),
        "Cand": cand,
        # the split staging [T, n_split, top_k] lives behind the persistent candidate buffers (merge stage only)
        "Staging": cand[choice.scan_entries :] if choice.split else None,
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
        check_period=int(choice.check_period),
        grid_ctas=int(choice.grid_ctas),
        softmax_scale=float(softmax_scale),
        n_split=int(choice.n_split),
        key_bits=int(choice.key_bits),
        num_queries=num_queries,
        num_keys=num_keys,
    )
    grid = (int(choice.grid_ctas), 1, 1)
    row_grid = (max(1, num_queries), 1, 1)
    launches: list[_Launch] = []
    stages: list[str] = []
    if num_queries > 0 and num_keys > 0:
        values: dict[str, Any] = dict(t)
        values.update(scalars)
        launches.append(bind_program(programs["scan"], arch, "scan", values, grid))
        stages.append("scan")
        if "merge" in programs:
            launches.append(
                bind_program(programs["merge"], arch, "merge", values, row_grid)
            )
            stages.append("merge")
        # lever FRP-K: the persistent rank finalize launches min(rows, CTAs per SM x SMs) CTAs, each walking rows bid, bid + grid, ...
        fin_grid = (finalize_grid(choice, _num_sms(device)), 1, 1)
        launches.append(
            bind_program(programs[fin_role], arch, fin_role, values, fin_grid)
        )
        stages.append(fin_role)
    elif num_queries > 0:
        _padded_outputs(num_queries, int(top_k), device, t["Indices"], t["Scores"])
    return DSAIndexerRunner(
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
    )


def dsa_indexer_topk(
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
    finalize_threads: Optional[int] = None,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager entry point: prepare a binding and run it once; returns ``(indices, scores)``.

    See :func:`prepare_dsa_indexer_topk` for the arguments.  The call allocates
    the outputs (unless given) and the workspace (unless given) through the
    caching allocator and performs no host synchronization, so it may be
    captured into a CUDA graph like any allocation-free launch sequence.
    """
    runner = prepare_dsa_indexer_topk(
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
        finalize_threads=finalize_threads,
        backend=backend,
    )
    return runner.run()


__all__ = [
    "ABI_CONTRACT",
    "CONTRACT_ALIASES",
    "CONTRACT_SCALARS",
    "CONTRACT_TENSORS",
    "DEFAULT_TOP_K",
    "FINALIZE_RANK_SCALARS",
    "FINALIZE_SCALARS",
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
    "DSAIndexerRunner",
    "DispatchPolicy",
    "ProgramChoice",
    "arch_for",
    "bind_program",
    "default_softmax_scale",
    "dsa_indexer_topk",
    "dsa_indexer_workspace_size",
    "effective_offsets",
    "generated_program_available",
    "plan_dsa_indexer_topk",
    "policy_for",
    "prepare_dsa_indexer_topk",
    "program_for",
    "record_zero_sign_policy",
    "registry_abi",
    "validate_dsa_indexer_inputs",
    "visible_key_count",
    "visible_key_counts",
]
