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
  >= 1``; ``max_seqlen_q`` / ``max_seqlen_k`` are accepted host mirrors that
  no launch decision reads.
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
* Repeatable: identical inputs give identical ids and score bits; the
  internal partition knobs (``grid_ctas``, ``candidate_multiplier``,
  ``check_period``, the sampled-threshold knobs) never change a result.
* Memory: no ``[T, Tkv]`` score matrix.  The workspace is
  :func:`dsa_indexer_workspace_size` bytes -- ``grid x queries_per_cta x
  candidate_capacity(top_k) x entry_bytes`` from the record's gate policy,
  independent of ``T``, ``Tkv`` and ``S`` -- separate from the inputs and the
  two outputs; no host synchronization inside the operator.
* Non-finite inputs (NaN / inf / overflow) are outside the normal domain: the
  computed score bits are returned, NaN scores rank below every other score
  (also below ``-inf``), ``+-inf`` rank by value, and no loop bound depends on
  a score value, so the kernels cannot hang.

Every allocation happens in :func:`prepare_dsa_indexer_topk`; the returned
runner launches with no CUDA allocation and no host synchronization, so a
runner (or a CUDA graph capturing it) replays for new values written into the
bound tensors.  Kernels are reached through the argument plans of the registry
records in ``cake_jit.MODULES``; the ``abi`` field of a record names the
keyword set its kernels expect (:data:`ABI_CONTRACT`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    FINALIZE_STAGES,
    MODULES,
    load_cake_dsa_indexer_module,
    registered_stages,
    select_module,
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
ABI_CONTRACT = "dsa_indexer_v1"
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
    "Cand",  # the workspace viewed as packed int64 candidate entries
)
CONTRACT_SCALARS = (
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
)
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
    "q_causal_offsets": "q_offsets",
    "offsets": "q_offsets",
    "num_segs": "num_segments",
    "topk": "top_k",
    "scale": "softmax_scale",
    "sm_scale": "softmax_scale",
    "grid": "grid_ctas",
}

# Documented zero-sign behaviour of the head reduction (registry field
# ``numerics["zero_sign_policy"]``): ``positive_accumulator`` -- the FMA chains
# start from ``+0.0``, so a row whose head terms are all zero scores ``+0.0``;
# ``ieee_sum`` -- the sum starts from the ``h = 0`` term, so an all-zero sum is
# ``-0.0`` iff every term is ``-0.0``.  Either way the computed bits are
# returned unchanged and ``+0.0 == -0.0`` for ranking.
ZERO_SIGN_POLICIES = ("positive_accumulator", "ieee_sum")

# ---------------------------------------------------------------------------
# Candidate-gate policy (host-evaluated; carried by the registry record)
# ---------------------------------------------------------------------------

GATE_POLICY_FIELDS = (
    "queries_per_cta",  # queries one persistent CTA scores at a time (one candidate buffer each)
    "candidate_entry_bytes",  # packed (score bits, key id) entry size
    "candidate_multiplier",  # default capacity multiplier: cand_cap ~ multiplier * top_k
    "candidate_slack",  # minimum headroom above top_k before a buffer can be compacted
    "tile_keys",  # keys per key tile (capacity granule)
    "check_period_max",  # cap on the selection-trigger check period (tiles)
    "check_period_cap_divisor",  # default period = cand_cap // divisor (keys; an architecture-specific constant)
    "sample_tiles_max",  # cap on the sample tiles of the sampled first threshold (0 = none)
    "sample_shift_permille",  # conservative shift of the sampled rank
    "finalize_small_max_top_k",  # largest top_k the finalize_small stage sorts
)


@dataclass(frozen=True)
class GatePolicy:
    """Candidate-gate constants of a generated program, evaluated on the host.

    ``candidate_capacity(top_k)`` is the per-query candidate buffer (entries),
    ``check_period(top_k, cap)`` the default tile period between selection
    checks, ``workspace_bytes(top_k, grid)`` the explicit workspace bound.
    Every value is a plain integer from the registry record; the host does
    not guess kernel constants.
    """

    queries_per_cta: int
    candidate_entry_bytes: int
    candidate_multiplier: int
    candidate_slack: int
    tile_keys: int
    check_period_max: int
    check_period_cap_divisor: int
    sample_tiles_max: int
    sample_shift_permille: int
    finalize_small_max_top_k: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GatePolicy":
        raw = record.get("gate_policy")
        if not isinstance(raw, dict) or set(raw) != set(GATE_POLICY_FIELDS):
            raise ValueError(
                f"registry record declares gate_policy {raw!r}; expected the fields {GATE_POLICY_FIELDS}"
            )
        values = {}
        for name in GATE_POLICY_FIELDS:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"gate_policy[{name!r}] must be an integer, got {value!r}"
                )
            if (
                name != "sample_shift_permille"
                and name != "sample_tiles_max"
                and value <= 0
            ):
                raise ValueError(
                    f"gate_policy[{name!r}] must be positive, got {value!r}"
                )
            values[name] = int(value)
        if values["sample_tiles_max"] < 0:
            raise ValueError("gate_policy['sample_tiles_max'] must be >= 0")
        return cls(**values)

    def candidate_capacity(self, top_k: int, multiplier: Optional[int] = None) -> int:
        """Per-query candidate buffer: ``max(multiplier * top_k, top_k + slack)`` rounded up to whole tiles."""
        mult = self.candidate_multiplier if multiplier is None else int(multiplier)
        if mult < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        wanted = max(mult * int(top_k), int(top_k) + self.candidate_slack)
        return -(-wanted // self.tile_keys) * self.tile_keys

    def check_period(self, top_k: int, cand_cap: int) -> int:
        """Default tiles between two selection-trigger checks (exact for the given capacity).

        The capacity divisor is an architecture-specific constant of the program
        (the record of each architecture carries its own value).
        """
        return max(
            1,
            min(
                self.check_period_max,
                (int(cand_cap) - int(top_k)) // self.tile_keys,
                int(cand_cap) // self.check_period_cap_divisor,
            ),
        )

    def check_period_limit(self, top_k: int, cand_cap: int) -> int:
        """Largest ``check_period`` that is still exact for the capacity (the kernel's headroom rule)."""
        return max(
            1,
            min(
                self.check_period_max,
                (int(cand_cap) - int(top_k)) // self.tile_keys,
                int(cand_cap) // (2 * self.tile_keys),
            ),
        )

    def workspace_bytes(
        self, top_k: int, grid: int, multiplier: Optional[int] = None
    ) -> int:
        """Explicit workspace bound: ``grid x queries_per_cta x candidate_capacity x entry_bytes``."""
        return (
            int(grid)
            * self.queries_per_cta
            * self.candidate_capacity(top_k, multiplier)
            * self.candidate_entry_bytes
        )


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def default_softmax_scale() -> float:
    return HEAD_DIM**-0.5


def record_for(device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    """``(module_name, record)`` registered for ``device``; raises when absent."""
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "DSA indexer top-k requires compute capability 10.0, 10.3 or 10.7"
        )
    name = select_module(arch)
    return name, MODULES[name]


def generated_program_available(device: Optional[torch.device] = None) -> bool:
    """True when this checkout registers a complete program for ``device``."""
    arch = arch_for(device)
    if arch is None:
        return False
    names = [n for n, r in MODULES.items() if r["arch"] == arch]
    if len(names) != 1:
        return False
    stages = registered_stages(names[0])
    return "scan" in stages and any(s in stages for s in FINALIZE_STAGES)


def record_abi(record: dict[str, Any]) -> str:
    abi = str(record.get("abi", ABI_CONTRACT))
    if abi not in SUPPORTED_ABIS:
        raise NotImplementedError(f"unsupported host binding profile {abi!r}")
    return abi


def record_gate_policy(record: dict[str, Any]) -> GatePolicy:
    return GatePolicy.from_record(record)


def record_zero_sign_policy(record: dict[str, Any]) -> str:
    """The documented zero-sign policy of the program's head reduction."""
    numerics = record.get("numerics")
    policy = numerics.get("zero_sign_policy") if isinstance(numerics, dict) else None
    if policy not in ZERO_SIGN_POLICIES:
        raise ValueError(
            f"registry record declares numerics {numerics!r}; zero_sign_policy must be one of {ZERO_SIGN_POLICIES}"
        )
    return str(policy)


def finalize_stage_for(
    stages: tuple[str, ...],
    policy: GatePolicy,
    top_k: int,
    override: Optional[str] = None,
) -> str:
    """The finalize stage serving ``top_k``: ``finalize_small`` up to its limit when registered, else ``finalize``."""
    if override is not None:
        if override not in FINALIZE_STAGES:
            raise ValueError(
                f"finalize_stage must be one of {FINALIZE_STAGES}, got {override!r}"
            )
        if override not in stages:
            raise NotImplementedError(
                f"the registered program has no {override!r} stage (registered: {stages})"
            )
        if (
            override == "finalize_small"
            and int(top_k) > policy.finalize_small_max_top_k
        ):
            raise ValueError(
                f"finalize_small sorts at most top_k = {policy.finalize_small_max_top_k}, got {top_k}"
            )
        return override
    if "finalize_small" in stages and int(top_k) <= policy.finalize_small_max_top_k:
        return "finalize_small"
    if "finalize" in stages:
        return "finalize"
    raise NotImplementedError(
        f"the registered program has no finalize stage for top_k = {top_k} (registered: {stages})"
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
    if max_seqlen_k is not None:
        _check_int("max_seqlen_k", max_seqlen_k, 0)
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
# Workspace bound
# ---------------------------------------------------------------------------


def _num_sms(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def dsa_indexer_workspace_size(
    top_k: int,
    device: Optional[torch.device] = None,
    *,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    policy: Optional[GatePolicy] = None,
) -> int:
    """Explicit workspace bound in bytes for ``top_k`` on ``device``.

    ``grid x queries_per_cta x candidate_capacity(top_k) x entry_bytes`` with
    ``grid`` = the device's SM count (the persistent grid) unless ``grid_ctas``
    overrides it; independent of ``T``, ``Tkv`` and the segment count.  The
    gate policy comes from the program registered for ``device`` unless
    ``policy`` is given explicitly.
    """
    _check_int("top_k", top_k, 1, MAX_TOP_K)
    if policy is None:
        _, record = record_for(device)
        policy = record_gate_policy(record)
    if grid_ctas is None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        grid = _num_sms(device)
    else:
        grid = _check_int("grid_ctas", grid_ctas, 1)
    return policy.workspace_bytes(top_k, grid, candidate_multiplier)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def grid_dims(rule, scalars: dict[str, Any], num_sms: int) -> tuple[int, int, int]:
    """Evaluate a registry grid rule.

    Each of the three entries is an integer, ``"sms"``, ``"sms*<n>"`` or
    ``"<name>[*<a>][/<b>]"``: a scalar name (``"num_queries"``) optionally
    multiplied by ``a`` and then divided by ``b`` with rounding up.
    """

    def term(value) -> int:
        if isinstance(value, bool):
            raise ValueError("grid rule entries must be integers or expressions")
        if isinstance(value, int):
            return int(value)
        text = str(value)
        if text == "sms":
            return int(num_sms)
        if text.startswith("sms*"):
            return int(num_sms) * int(text[4:])
        name, _, divisor = text.partition("/")
        name, _, factor = name.partition("*")
        value = int(scalars[name]) * (int(factor) if factor else 1)
        return -(-value // int(divisor)) if divisor else value

    if len(rule) != 3:
        raise ValueError("grid rule must have three entries")
    x, y, z = (term(v) for v in rule)
    return max(1, x), max(1, y), max(1, z)


@dataclass(frozen=True)
class _Launch:
    stage: str
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]
    # Descriptor-preparation entry of a pointer-ABI module (same arguments); the
    # indexer programs pass their tensor maps by value and have none.
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)


def bind_stage(
    module_name: str, stage: str, values: dict[str, Any], grid: tuple[int, int, int]
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``stage`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.
    """
    physical = MODULES[module_name][stage]
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in physical["arg_plan"]:
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif key in values and values[key] is not None:
            arguments.append(values[key])
        else:
            raise KeyError(
                f"generated module {module_name!r} stage {stage!r} expects argument "
                f"{name!r} ({kind}); the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_dsa_indexer_module(module_name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return _Launch(
        stage,
        module_name,
        getattr(module, physical["ffi_entry"]),
        tuple(arguments),
        grid,
        prepare,
    )


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

    ``__call__()`` runs the scan and the finalize stage on torch's current
    stream and returns ``(indices, scores)``.  No launch allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype, stride or tensor binding changes; values
    may change freely.
    """

    module_name: str
    abi: str
    num_queries: int
    num_keys: int
    num_segments: int
    top_k: int
    softmax_scale: float
    ratio: int
    tensors: dict[str, torch.Tensor] = field(repr=False)
    scalars: dict[str, Any] = field(repr=False)
    launches: tuple[_Launch, ...] = field(repr=False)
    stages: tuple[str, ...]
    device_index: int
    workspace: torch.Tensor = field(repr=False)
    workspace_bytes: int = 0
    grid: tuple[int, int, int] = (1, 1, 1)
    _tma_prepared: bool = False

    @property
    def indices(self) -> torch.Tensor:
        return self.tensors["Indices"]

    @property
    def scores(self) -> torch.Tensor:
        return self.tensors["Scores"]

    def prepare_tma(self) -> None:
        """Encode the descriptors of pointer-ABI stages once (idempotent; no-op for by-value tensor maps)."""
        if self._tma_prepared:
            return
        with _ffi_stream_context(self.device_index):
            for launch in self.launches:
                if launch.prepare is not None:
                    launch.prepare(*launch.arguments)
        self._tma_prepared = True

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.prepare_tma()
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
    finalize_stage: Optional[str] = None,
    backend: str = "cake",
) -> DSAIndexerRunner:
    """Validate one binding and prepare its launches.

    Missing outputs and the workspace are allocated here (the only allocations
    of the backend).  Pass ``workspace_buffer`` of
    :func:`dsa_indexer_workspace_size` bytes (same ``grid_ctas`` /
    ``candidate_multiplier``) to reuse storage across calls; ``indices`` /
    ``scores`` may be caller-owned ``[T, top_k]`` int32 / float32 tensors.

    The keyword-only knobs ``grid_ctas`` (persistent CTA count; default: the SM
    count), ``candidate_multiplier`` (candidate buffer per query in multiples
    of ``top_k``), ``check_period`` (key tiles between selection-trigger
    checks), ``sample_tiles_max`` / ``sample_shift_permille`` (sampled first
    threshold; ``0`` disables sampling) and ``finalize_stage`` change only the
    internal partition and scheduling of the work: results are bitwise
    identical for every admissible value.  ``T == 0`` returns a runner whose
    launches are empty; a call without keys (``Tkv == 0``) fills the outputs
    with padding instead of launching.
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
    for name, t in (
        ("workspace_buffer", workspace_buffer),
        ("indices", indices),
        ("scores", scores),
    ):
        if t is not None and (not t.is_cuda or t.device != device):
            raise ValueError(f"{name} must be on {device}")
    _check_output(indices, "indices", (num_queries, int(top_k)), torch.int32, device)
    _check_output(scores, "scores", (num_queries, int(top_k)), torch.float32, device)
    module_name, record = record_for(device)
    abi = record_abi(record)
    stages = registered_stages(module_name)
    if "scan" not in stages:
        raise NotImplementedError(
            f"the registered DSA indexer program {module_name!r} has no scan stage (registered: {stages})"
        )
    policy = record_gate_policy(record)
    fin_stage = finalize_stage_for(stages, policy, top_k, finalize_stage)
    device_index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    num_sms = _num_sms(device)

    scan_physical = record["scan"]
    if grid_ctas is None:
        grid = grid_dims(scan_physical.get("grid", ["sms", 1, 1]), {}, num_sms)
    else:
        grid = (_check_int("grid_ctas", grid_ctas, 1), 1, 1)
    cand_cap = policy.candidate_capacity(top_k, candidate_multiplier)
    period = (
        policy.check_period(top_k, cand_cap)
        if check_period is None
        else int(check_period)
    )
    _check_int("check_period", period, 1, policy.check_period_limit(top_k, cand_cap))
    tiles_max = (
        policy.sample_tiles_max
        if sample_tiles_max is None
        else _check_int("sample_tiles_max", sample_tiles_max, 0, 256)
    )
    shift = (
        policy.sample_shift_permille
        if sample_shift_permille is None
        else _check_int("sample_shift_permille", sample_shift_permille, -1000, 10000)
    )
    workspace_bytes = policy.workspace_bytes(top_k, grid[0], candidate_multiplier)
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
        "Cand": flat[:workspace_bytes].view(torch.int64),
    }
    scalars: dict[str, Any] = dict(
        num_segments=num_segments,
        top_k=int(top_k),
        ratio=int(ratio),
        has_offsets=int(q_causal_offsets is not None),
        cand_cap=int(cand_cap),
        first_cap=int(
            cand_cap
        ),  # regular trigger only: the first selection fires at the capacity rule
        sample_tiles_max=int(tiles_max),
        sample_shift_permille=int(shift),
        check_period=int(period),
        grid_ctas=int(grid[0]),
        softmax_scale=float(softmax_scale),
        num_queries=num_queries,
        num_keys=num_keys,
    )
    launches: list[_Launch] = []
    if num_queries > 0 and num_keys > 0:
        values: dict[str, Any] = dict(t)
        values.update(scalars)
        launches.append(bind_stage(module_name, "scan", values, grid))
        fin_physical = record[fin_stage]
        fin_grid = grid_dims(
            fin_physical.get("grid", ["num_queries", 1, 1]), scalars, num_sms
        )
        launches.append(bind_stage(module_name, fin_stage, values, fin_grid))
    elif num_queries > 0:
        _padded_outputs(num_queries, int(top_k), device, t["Indices"], t["Scores"])
    return DSAIndexerRunner(
        module_name=module_name,
        abi=abi,
        num_queries=num_queries,
        num_keys=num_keys,
        num_segments=num_segments,
        top_k=int(top_k),
        softmax_scale=float(softmax_scale),
        ratio=int(ratio),
        tensors=t,
        scalars=scalars,
        launches=tuple(launches),
        stages=("scan", fin_stage),
        device_index=device_index,
        workspace=flat,
        workspace_bytes=int(workspace_bytes),
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
    finalize_stage: Optional[str] = None,
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
        finalize_stage=finalize_stage,
        backend=backend,
    )
    return runner.run()


__all__ = [
    "ABI_CONTRACT",
    "CONTRACT_ALIASES",
    "CONTRACT_SCALARS",
    "CONTRACT_TENSORS",
    "DEFAULT_TOP_K",
    "GATE_POLICY_FIELDS",
    "HEAD_DIM",
    "MAX_TOP_K",
    "NUM_HEADS",
    "PAD_ID",
    "PAD_SCORE",
    "SUPPORTED_ABIS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "ZERO_SIGN_POLICIES",
    "DSAIndexerRunner",
    "GatePolicy",
    "arch_for",
    "bind_stage",
    "default_softmax_scale",
    "dsa_indexer_topk",
    "dsa_indexer_workspace_size",
    "effective_offsets",
    "finalize_stage_for",
    "generated_program_available",
    "grid_dims",
    "prepare_dsa_indexer_topk",
    "record_abi",
    "record_for",
    "record_gate_policy",
    "record_zero_sign_policy",
    "validate_dsa_indexer_inputs",
    "visible_key_count",
    "visible_key_counts",
]
