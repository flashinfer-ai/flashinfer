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
"""

# Host backend of the Cake DSA sparse-attention training program (SM100 /
# SM103 / SM107): input validation, workspace layout, varlen index offsetting,
# the prepared runner, the eager entry points and the autograd Function.
#
# * Forward writes ``out [T, 64, 512]`` BF16, ``lse [T, 64]`` FP32 (natural
#   log over the valid keys; ``-inf`` for a fully masked row, whose ``out`` is
#   zero) and the BF16 output residual ``o_lo = bf16(fp32(O) - bf16(O))``.
# * Backward returns ``dq_latent [T, 64, 512]``, ``dq_rope [T, 64, 64]`` BF16
#   (written once per row: bitwise deterministic) and ``dkv_latent [S, 512]``,
#   ``dk_rope [S, 64]`` (FP32 ``red.global`` accumulation in the kernels'
#   internal permuted layout, un-permuted by the cast stage into natural-layout
#   BF16 or, with ``dkv_fp32=True``, FP32 outputs) -- or, with a caller-owned
#   ``dkv_acc``, adds them into that packed FP32 buffer (``+=``, optional
#   destination-row map ``dkv_dst_map``) and returns ``None`` for both.
# * Strided inputs are consumed in place: the query operands reach the kernels
#   through TMA descriptors encoded from the tensor's own head and token
#   strides, the key operands by row stride (see ``_check_head_tensor`` /
#   ``_check_key_tensor``).
# * Without ``topk_length`` the forward kernel derives each row's length (last
#   valid slot + 1) itself while it runs and skips the trailing invalid key
#   blocks; ``derive_topk_length=True`` makes it write the lengths into the
#   caller's ``[T]`` tensor so a training step can hand them to its backward
#   (the autograd entry does so; no separate derivation launch).
# * A program registering the natural-layout main stages (``bwd_main_natural``,
#   ``bwd_main_pass_natural``) adds the dK/dV gradients of a ``dkv_acc`` binding
#   straight into the caller's packed rows when the record's size rule selects
#   it (``plan_dkv_direct``): no FP32 accumulators, no zero fill, no cast launch.
# * Every launch goes through ``cake_launch`` (generated next to the registry:
#   one positional launcher per stage over the kernel's own argument names)
#   with a grid computed in Python; the bindings encode the tensor maps by
#   value, so no launch allocates, synchronizes or touches a descriptor
#   workspace.

from __future__ import annotations

import functools
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from . import cake_jit, cake_launch
from .cake_jit import FORWARD_STAGES

NUM_HEADS = 64
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE
LOG2E = 1.4426950408889634
WORKSPACE_ALIGN = 256
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# Raw-pointer operands are read in 16-byte vectors: eight int32 index slots,
# eight BF16 rope elements.  ``indices`` may start anywhere (the kernels take
# the element offset below a vector-aligned base and fall back to scalar loads
# when the row stride or the offset is not a multiple of eight); ``k_rope``
# rows and their start must be vector aligned.
POINTER_GRANULE = 8
# The head-dimension operands (q_latent, q_rope, dout, dq_*) reach the kernels
# through 4-D TMA descriptors whose global strides are the tensor's own head and
# token strides; the key operands are 2-D TMA gather maps over the row stride
# (kv_latent, and k_rope in the backward) or 16-byte row loads (k_rope in the
# forward).  ``cuTensorMapEncodeTiled`` requires every global stride to be a
# positive multiple of 16 bytes and a 16-byte-aligned global address: for BF16
# a multiple of 8 elements.  Nothing else is assumed about the layout: a head
# stride of 576 (a packed [T, 64, 576] query), 256 (the 192:256 rope channels of
# the [T, 64, 256] pre-absorption query) and key row strides of 576 or 704 (a
# frozen 128-channel indexer key stored alongside) are consumed in place.
TMA_STRIDE_ELEMENTS = 8  # 16 bytes of BF16: the descriptor stride granule
TMA_ADDRESS_ALIGN = 16  # bytes: the descriptor's global address

# Key-range passes of the backward (the DRAM regime of the main stage).  A
# program with ``bwd_compact`` and ``bwd_main_pass`` can run the backward of
# every token in P passes over disjoint key ranges: per pass the compaction
# stage writes the token's keys inside the range (slot order; the validity
# rules of the main kernel) to ``key_scratch`` and their number to
# ``pass_counts``, and the pass variant of the main stage consumes them,
# carrying the FP32 dQ / dQ_rope partial of every token in ``dq_partial``
# between passes (dq_mode 1 store, 2 load-add-store, 3 load-add and BF16
# output) -- ``dq`` stays bitwise deterministic; the dK/dV reductions are
# unchanged (a pass only selects which keys a tile carries).  The pass count
# follows the policy the record carries (``key_pass_policy``, evaluated by
# ``KeyPassPolicy``): the L2 formula P = ceil(S * key_bytes / l2_budget_bytes)
# (the FP32 accumulator slice one pass touches fits the L2), the token chunk =
# the largest multiple of ``token_chunk_multiple`` whose pass scratch (147,456
# + 4 * topk + 4 B per token) fits ``workspace_budget_bytes`` (4224 tokens at
# top-k 2048), and the rule of the device's architecture from the paired sweeps
# of the second optimisation round (``rules[arch]``: ``tail_rows``,
# ``fixed_passes``, ``min_kv``, ``tail_min_kv``, ``tail_cap``): passes apply to
# a one-segment key row when P > 1 and the row is one chunk (``T <= chunk``) or
# -- with ``tail_rows`` -- ``2 T <= S``; one-chunk rows below ``min_kv`` keys
# and the other tail rows below ``tail_min_kv`` keys stay single-pass; the
# count is ``fixed_passes`` when set, else P, capped at ``tail_cap`` for tail
# rows that are not one chunk.  The tokens of a multi-pass backward run in
# chunks of ``chunk`` tokens, every chunk through every pass in order (grid =
# the chunk's tokens, ``token_base`` = its first token), exactly as the Cake
# launcher, so the workspace grows by min(T, chunk) * (147,456 + 4 * topk + 4)
# bytes.  The passes split the whole key row ``[0, S)`` into equal ranges,
# which matches an index row whose keys spread over the whole row (one
# document); in a packed multi-segment row every token's keys lie inside its
# own segment, so whole-row ranges leave most passes empty for most tokens
# while the per-pass fixed cost is still paid.  The policy therefore applies
# to one-segment rows only: ``num_segments > 1`` (the varlen entry passes
# ``len(cu_seqlens_k) - 1``, host metadata, no device sync) plans one pass
# unless ``key_passes`` overrides.  A record without ``rules`` (the first
# release) takes the R1 rule everywhere: one-chunk rows, the formula, no tail
# rows.  A count above ``max_passes`` (4) runs the single-pass stage: beyond it
# the per-pass FP32 dQ-partial round trip and pipeline fill outweigh the L2
# benefit (B200, 4k queries: five and six passes lose 7-15 % against one).
KEY_PASS_STAGES = ("bwd_compact", "bwd_main_pass")
DQ_PARTIAL_BYTES_PER_TOKEN = (
    NUM_HEADS * D_QK * 4
)  # one FP32 dQ / dQ_rope partial row (147,456 B)
KEY_PASS_POLICY_FIELDS = (
    "l2_budget_bytes",
    "key_bytes",
    "workspace_budget_bytes",
    "token_chunk_multiple",
    "max_passes",
)
# The per-target rule fields of ``key_pass_policy["rules"][arch]`` (absent in
# first-release records: the R1 rule).
KEY_PASS_RULE_FIELDS = (
    "tail_rows",
    "fixed_passes",
    "min_kv",
    "tail_min_kv",
    "tail_cap",
)
_R1_RULE: dict[str, int] = dict.fromkeys(KEY_PASS_RULE_FIELDS, 0)


@dataclass(frozen=True)
class KeyPassPolicy:
    """The record's key-range-pass policy (see the comment above)."""

    l2_budget_bytes: int  # FP32 accumulator bytes one pass may touch (100 MiB)
    key_bytes: int  # FP32 accumulator bytes per key row (576 * 4 = 2304)
    workspace_budget_bytes: int  # pass scratch the whole row may need at most (640 MiB)
    token_chunk_multiple: (
        int  # the token chunk is a multiple of this and at least this (128)
    )
    max_passes: int  # the formula's count is taken only up to this many passes, else one pass (4)
    # the per-target rules: ((arch, (tail_rows, fixed_passes, min_kv, tail_min_kv, tail_cap)), ...)
    rules: tuple[tuple[str, tuple[int, ...]], ...] = ()

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Optional["KeyPassPolicy"]:
        raw = record.get("key_pass_policy")
        if raw is None:
            return None
        missing = [name for name in KEY_PASS_POLICY_FIELDS if name not in raw]
        if missing:
            raise ValueError(f"registry record: key_pass_policy lacks {missing}")
        values = {name: int(raw[name]) for name in KEY_PASS_POLICY_FIELDS}
        if any(v <= 0 for v in values.values()):
            raise ValueError(
                f"registry record: key_pass_policy needs positive values, got {raw}"
            )
        rules = []
        for arch, rule in (raw.get("rules") or {}).items():
            if not isinstance(rule, dict) or set(rule) != set(KEY_PASS_RULE_FIELDS):
                raise ValueError(
                    f"registry record: key_pass_policy.rules[{arch!r}] must map exactly "
                    f"{KEY_PASS_RULE_FIELDS}, got {rule!r}"
                )
            for name in KEY_PASS_RULE_FIELDS:
                value = rule[name]
                if isinstance(value, bool) or int(value) != value or int(value) < 0:
                    raise ValueError(
                        f"registry record: key_pass_policy.rules[{arch!r}].{name} must be "
                        f"a non-negative integer, got {value!r}"
                    )
            rules.append(
                (str(arch), tuple(int(rule[name]) for name in KEY_PASS_RULE_FIELDS))
            )
        return cls(**values, rules=tuple(rules))

    def rule_for(self, arch: Optional[str]) -> dict[str, int]:
        """The rule of ``arch`` (an architecture without one, or ``None``: the R1 rule)."""
        for name, fields in self.rules:
            if name == arch:
                return dict(zip(KEY_PASS_RULE_FIELDS, fields, strict=True))
        return dict(_R1_RULE)

    def formula_passes(self, num_kv: int) -> int:
        return max(1, -(-int(num_kv) * self.key_bytes // self.l2_budget_bytes))

    def token_chunk(self, topk: int) -> int:
        """Tokens whose pass scratch (dQ partial, key scratch, pass count) fits the workspace budget."""
        per_token = DQ_PARTIAL_BYTES_PER_TOKEN + 4 * int(topk) + 4
        m = self.token_chunk_multiple
        return max(m, (self.workspace_budget_bytes // per_token) // m * m)

    def token_chunks(self, num_queries: int, topk: int) -> tuple[tuple[int, int], ...]:
        """``(token_base, num_tokens)`` of every token chunk of a multi-pass backward, in launch order."""
        T, chunk = int(num_queries), self.token_chunk(topk)
        return tuple((base, min(chunk, T - base)) for base in range(0, T, chunk))

    def passes(
        self,
        num_queries: int,
        num_kv: int,
        topk: int,
        *,
        num_segments: int = 1,
        arch: Optional[str] = None,
    ) -> int:
        """Passes of a binding under the rule of ``arch`` (see the comment above
        ``KEY_PASS_STAGES``): 1 for a multi-segment key row, a formula of 1, a
        row that is neither one chunk nor (with ``tail_rows``) ``2 T <= S``, or
        fewer keys than the row's floor (``min_kv`` for a one-chunk row,
        ``tail_min_kv`` for a tail row of more than one chunk); else
        ``fixed_passes`` or the formula, capped at ``tail_cap`` for a tail row
        of more than one chunk; a count above ``max_passes`` (beyond it the
        per-pass dQ-partial round trip and pipeline fill outweigh the L2
        benefit) is one pass."""
        if int(num_segments) < 1:
            raise ValueError(f"num_segments must be >= 1, got {num_segments}")
        if int(num_segments) > 1:
            return 1
        T, S = int(num_queries), int(num_kv)
        formula = self.formula_passes(S)
        if formula == 1:
            return 1
        rule = self.rule_for(arch)
        one_chunk = self.token_chunk(topk) >= T
        tail = bool(rule["tail_rows"]) and 2 * T <= S
        if not (one_chunk or tail):
            return 1
        if (rule["min_kv"] if one_chunk else rule["tail_min_kv"]) > S:
            return 1
        passes = rule["fixed_passes"] or formula
        if tail and not one_chunk and rule["tail_cap"]:
            passes = min(passes, rule["tail_cap"])
        return passes if passes <= self.max_passes else 1


def plan_key_passes(
    record: dict[str, Any],
    stages: tuple[str, ...],
    num_queries: int,
    num_kv: int,
    topk: int,
    key_passes: Optional[int] = None,
    *,
    num_segments: int = 1,
    arch: Optional[str] = None,
) -> int:
    """Number of key-range passes of one backward binding.

    ``key_passes`` overrides the record's policy (1 = the single-pass stage);
    a program without the pass stages serves one pass only, and a record
    without a ``key_pass_policy`` never takes the pass path by default.
    ``num_segments`` is the packed segment count of the key row (see the
    comment above ``KEY_PASS_STAGES``): the policy plans one pass for more
    than one segment.  ``arch`` selects the record's per-target rule (``None``:
    the R1 rule).
    """
    multi_pass = all(stage in stages for stage in KEY_PASS_STAGES)
    if key_passes is not None:
        if isinstance(key_passes, bool) or int(key_passes) != key_passes:
            raise ValueError("key_passes must be a positive integer or None")
        passes = int(key_passes)
        if passes < 1:
            raise ValueError("key_passes must be >= 1")
        if passes > int(num_kv):
            raise ValueError(
                f"key_passes ({passes}) must not exceed the number of keys ({int(num_kv)})"
            )
        if passes > 1 and not multi_pass:
            raise NotImplementedError(
                "the registered DSA training program has no key-range-pass stages "
                f"({KEY_PASS_STAGES}); key_passes > 1 is unavailable"
            )
        return passes
    if not multi_pass:
        return 1
    policy = KeyPassPolicy.from_record(record)
    if policy is None:
        return 1
    return policy.passes(
        num_queries, num_kv, topk, num_segments=num_segments, arch=arch
    )


def _record_token_chunk(record: dict[str, Any], topk: int) -> Optional[int]:
    """Tokens per launch chunk of a multi-pass backward under the record's policy (``None``: no policy, the whole row)."""
    policy = KeyPassPolicy.from_record(record)
    return None if policy is None else policy.token_chunk(topk)


def _pass_token_chunks(
    record: dict[str, Any], num_queries: int, topk: int
) -> tuple[tuple[int, int], ...]:
    """``(token_base, num_tokens)`` of the token chunks a multi-pass backward runs (one whole-row chunk without a policy)."""
    policy = KeyPassPolicy.from_record(record)
    if policy is None:
        return ((0, int(num_queries)),)
    return policy.token_chunks(num_queries, topk)


def key_pass_ranges(num_kv: int, passes: int) -> tuple[tuple[int, int], ...]:
    """``[pass_lo, pass_hi)`` of every pass: ``R = ceil(S / P)`` keys, the last one clipped to ``S``."""
    S, P = int(num_kv), int(passes)
    R = -(-S // P)
    return tuple((p * R, min(S, (p + 1) * R)) for p in range(P))


def key_pass_dq_mode(index: int, passes: int) -> int:
    """``dq_mode`` of pass ``index``: 1 first (store the FP32 partial), 2 middle, 3 last (BF16 output)."""
    if passes == 1:
        return 0
    return 1 if index == 0 else (2 if index < passes - 1 else 3)


# Direct accumulation.  A record may register natural-layout variants of the
# backward main stage (``bwd_main_natural``; ``bwd_main_pass_natural`` next to
# the key-range-pass form) whose reduce warps add the dK/dV contributions
# straight into the caller's packed FP32 rows (``dkv_acc`` through
# ``dkv_dst_map``): no private accumulator, no zero fill and no ``bwd_cast``.
# The natural drain costs the reduce-bound rows a lane transpose that only rows
# with many keys per query token pay back (B200: S / T >= 16 -> 0.5..4 % faster
# backward, S = T -> 3.7 % slower), so the host takes it by the size rule the
# record carries (``dkv_direct``: ``min_keys_per_query``; direct iff
# S >= min_keys_per_query * T) whenever the caller passes ``dkv_acc``;
# ``FLASHINFER_CAKE_DSA_DKV_DIRECT=1|0`` forces or disables it (``auto`` = the
# rule).  Without the natural stages every ``dkv_acc`` call takes the cast's
# accumulate path.  Both paths add the same FP32 contributions into the same
# rows (another summation order).
DIRECT_STAGES = ("bwd_main_natural", "bwd_main_pass_natural")
DKV_DIRECT_ENV = "FLASHINFER_CAKE_DSA_DKV_DIRECT"
DKV_DIRECT_POLICY_FIELDS = ("min_keys_per_query",)


@dataclass(frozen=True)
class DkvDirectPolicy:
    """The record's direct-accumulation size rule (see the comment above)."""

    min_keys_per_query: int  # direct iff num_kv >= min_keys_per_query * num_queries (4)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Optional["DkvDirectPolicy"]:
        raw = record.get("dkv_direct")
        if raw is None:
            return None
        missing = [name for name in DKV_DIRECT_POLICY_FIELDS if name not in raw]
        if missing:
            raise ValueError(f"registry record: dkv_direct lacks {missing}")
        values = {name: int(raw[name]) for name in DKV_DIRECT_POLICY_FIELDS}
        if any(v <= 0 for v in values.values()):
            raise ValueError(
                f"registry record: dkv_direct needs positive values, got {raw}"
            )
        return cls(**values)

    def wants(self, num_queries: int, num_kv: int) -> bool:
        return int(num_kv) >= self.min_keys_per_query * int(num_queries)


def dkv_direct_mode() -> str:
    """``FLASHINFER_CAKE_DSA_DKV_DIRECT``: ``auto`` (default: the record's size rule), ``1`` (direct whenever the
    program registers the natural stages) or ``0`` (never: the cast's accumulate path)."""
    mode = os.environ.get(DKV_DIRECT_ENV, "auto").strip() or "auto"
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"{DKV_DIRECT_ENV} must be auto, 0 or 1, got {mode!r}")
    return mode


def record_direct_stages(record: dict[str, Any], stages: tuple[str, ...]) -> bool:
    """True when the record registers the natural-layout main stage (with its pass form next to the pass stages)."""
    direct = "bwd_main_natural" in stages
    if direct:
        if "bwd_main" not in stages:
            raise ValueError(
                "registry record: bwd_main_natural needs the bwd_main stage next to it"
            )
        if ("bwd_main_pass_natural" in stages) != all(
            s in stages for s in KEY_PASS_STAGES
        ):
            raise ValueError(
                "registry record: bwd_main_pass_natural must accompany the key-range-pass stages"
            )
        if DkvDirectPolicy.from_record(record) is None:
            raise ValueError(
                "registry record: the natural-layout stages need a dkv_direct policy"
            )
    elif "bwd_main_pass_natural" in stages:
        raise ValueError(
            "registry record: bwd_main_pass_natural without bwd_main_natural"
        )
    return direct


def plan_dkv_direct(
    record: dict[str, Any],
    stages: tuple[str, ...],
    num_queries: int,
    num_kv: int,
    *,
    accumulate: bool,
) -> bool:
    """Does this backward binding add ``dkv_acc`` directly from the main stage?  Only with ``dkv_acc``
    (``accumulate``), a program registering the natural stages, and the size rule or the forced mode."""
    if not accumulate or not record_direct_stages(record, stages):
        return False
    mode = dkv_direct_mode()
    if mode == "0":
        return False
    return mode == "1" or DkvDirectPolicy.from_record(record).wants(num_queries, num_kv)


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


@functools.cache
def _arch_of(device_index: int) -> Optional[str]:
    return SUPPORTED_COMPUTE_CAPABILITIES.get(
        torch.cuda.get_device_capability(device_index)
    )


def _device_index(device: Optional[torch.device]) -> int:
    if device is None or device.index is None:
        return torch.cuda.current_device()
    return int(device.index)


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return _arch_of(_device_index(device))


def default_softmax_scale() -> float:
    return D_QK**-0.5


def record_for(device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    """``(program name, registry record)`` serving ``device``."""
    arch = arch_for(device)
    if arch is None:
        raise NotImplementedError(
            "DSA sparse-attention training needs a compute capability 10.0 / 10.3 / 10.7 device"
        )
    record = cake_jit.record()
    if arch not in record["arches"]:
        raise NotImplementedError(
            f"the generated DSA training program is registered for {record['arches']}, not {arch}"
        )
    return cake_jit.PROGRAM, record


def generated_program_available(
    device: Optional[torch.device] = None, *, backward: bool = False
) -> bool:
    """Whether ``device`` is served (with the backward stages when ``backward``)."""
    try:
        _, record = record_for(device)
    except NotImplementedError:
        return False
    stages = cake_jit.registered_stages()
    needed = ("fwd", "bwd_delta", "bwd_main", "bwd_cast") if backward else ("fwd",)
    return all(stage in stages for stage in needed) and (
        not backward or "key_pass_policy" in record
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_head_tensor(t: torch.Tensor, name: str, last: int) -> None:
    """``[T, 64, last]`` BF16 with a unit channel stride, TMA-admissible head and token strides and base address."""
    if t.ndim != 3 or t.shape[1] != NUM_HEADS or t.shape[2] != last:
        raise ValueError(f"{name} must be a BF16 [T, {NUM_HEADS}, {last}] tensor")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if t.stride(2) != 1:
        raise ValueError(
            f"{name} must be contiguous along its last dimension, got strides {tuple(t.stride())}"
        )
    head, token = int(t.stride(1)), int(t.stride(0))
    if head < last or head % TMA_STRIDE_ELEMENTS:
        raise ValueError(
            f"{name} head stride must be >= {last} elements and a multiple of {TMA_STRIDE_ELEMENTS} elements "
            f"(16 bytes: TMA descriptor stride), got {head}; slices of a packed [T, {NUM_HEADS}, {D_QK}] query "
            f"or of the [T, {NUM_HEADS}, 256] pre-absorption query are admitted"
        )
    if token <= 0 or token % TMA_STRIDE_ELEMENTS:
        raise ValueError(
            f"{name} token stride must be a positive multiple of {TMA_STRIDE_ELEMENTS} elements "
            f"(16 bytes: TMA descriptor stride), got {token}"
        )
    if t.data_ptr() % TMA_ADDRESS_ALIGN:
        raise ValueError(
            f"{name} base address must be {TMA_ADDRESS_ALIGN}-byte aligned (TMA global address), "
            f"got {t.data_ptr() % TMA_ADDRESS_ALIGN} bytes past an aligned address"
        )


def _check_key_tensor(t: torch.Tensor, name: str, last: int) -> None:
    """``[S, last]`` BF16 with a unit column stride, a TMA-admissible row stride and base address."""
    if t.ndim != 2 or t.shape[1] != last:
        raise ValueError(f"{name} must be a BF16 [S, {last}] tensor")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if t.stride(1) != 1:
        raise ValueError(
            f"{name} rows must be contiguous (a view of a packed [S, {D_QK}] tensor is allowed)"
        )
    row = int(t.stride(0))
    if row < last or row % TMA_STRIDE_ELEMENTS:
        raise ValueError(
            f"{name} row stride must be >= {last} elements and a multiple of {TMA_STRIDE_ELEMENTS} elements "
            f"(16 bytes: TMA gather descriptor stride / 16-byte rope loads), got {row}; column slices of a "
            f"packed [S, {D_QK}] or [S, 704] row are admitted"
        )
    if t.data_ptr() % TMA_ADDRESS_ALIGN:
        raise ValueError(
            f"{name} base address must be {TMA_ADDRESS_ALIGN}-byte aligned (TMA global address), "
            f"got {t.data_ptr() % TMA_ADDRESS_ALIGN} bytes past an aligned address"
        )


def validate_dsa_train_inputs(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    *,
    dout: Optional[torch.Tensor] = None,
    allow_empty_queries: bool = False,
) -> tuple[int, int, int]:
    """Shape / dtype / layout validation shared by the entry points.

    Returns ``(T, S, topk)``.  Device placement is checked separately so this
    runs on host tensors.  Layouts: see :func:`_check_head_tensor` and
    :func:`_check_key_tensor` (strided query slices and packed key rows are
    consumed in place); ``indices`` rows must be contiguous (any row stride;
    a column slice of a wider index buffer is accepted).  Zero key rows
    (``S == 0``) are rejected: the kernels index at least one key row and the
    FP32 dK/dV accumulators would be empty.  Zero query rows (``T == 0``) are
    rejected unless ``allow_empty_queries``: the launch grids clamp to one CTA
    that would read a query row that does not exist, so only the eager entry
    points, which return empty outputs for ``T == 0`` without launching,
    accept it.
    """
    _check_head_tensor(q_latent, "q_latent", D_LATENT)
    _check_head_tensor(q_rope, "q_rope", D_ROPE)
    _check_key_tensor(kv_latent, "kv_latent", D_LATENT)
    _check_key_tensor(k_rope, "k_rope", D_ROPE)
    num_queries = int(q_latent.shape[0])
    num_kv = int(kv_latent.shape[0])
    if int(q_rope.shape[0]) != num_queries:
        raise ValueError("q_latent and q_rope must have the same number of rows")
    if int(k_rope.shape[0]) != num_kv:
        raise ValueError("kv_latent and k_rope must have the same number of rows")
    if num_kv == 0:
        raise ValueError("kv_latent and k_rope must hold at least one key row (S > 0)")
    if num_queries == 0 and not allow_empty_queries:
        raise ValueError(
            "q_latent holds no query rows (T == 0): a prepared runner needs at least one "
            "query row; the eager entry points return empty outputs for T == 0"
        )
    if (
        indices.ndim != 2
        or indices.dtype != torch.int32
        or int(indices.shape[0]) != num_queries
    ):
        raise ValueError("indices must be an int32 [T, topk] tensor")
    topk = int(indices.shape[1])
    if topk <= 0:
        raise ValueError("topk must be positive")
    if indices.stride(1) != 1 or (num_queries > 1 and indices.stride(0) < topk):
        raise ValueError("indices rows must be contiguous (any row stride)")
    if topk_length is not None and (
        topk_length.shape != (num_queries,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be a contiguous int32 [T] tensor")
    if dout is not None:
        _check_head_tensor(dout, "dout", D_LATENT)
        if int(dout.shape[0]) != num_queries:
            raise ValueError("dout must have T rows")
        if not dout.is_contiguous():
            # bwd_delta reads dout as a dense [T * 64, 512] array (the main stage reads it
            # through a stride-carrying tensor map); the eager entry copies a strided dout.
            raise ValueError("dout must be contiguous")
    return num_queries, num_kv, topk


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}"
        )


def _check_dkv_acc(
    dkv_acc: torch.Tensor, dkv_dst_map: Optional[torch.Tensor], num_kv: int
) -> tuple[torch.Tensor, int]:
    """Validate the caller's packed FP32 dK/dV accumulator and its optional destination-row map.

    ``dkv_acc``: FP32, 2-D ``[S_dst, >= 576]`` (the latent gradient is added into columns ``0:512``, the rope gradient
    into ``512:576``; further columns are never touched), ``stride(1) == 1``, row stride >= 576 elements and a multiple
    of 4 (16-byte vectors), 16-byte-aligned base; without a map at least ``num_kv`` rows.  ``dkv_dst_map``: ``None``
    (identity) or a contiguous int32 ``[num_kv]`` tensor on the device of ``dkv_acc`` -- the destination row of every
    source key row, values in ``[0, S_dst)``, duplicates allowed.  The kernel does not range-check the map (a value
    outside ``[0, S_dst)`` would add into memory outside ``dkv_acc``); the caller owns that invariant.  Setting the
    environment variable ``FLASHINFER_CAKE_DSA_CHECK_DST_MAP=1`` validates the values on every call (a device
    synchronization) and raises ``ValueError`` on a violation.  Returns ``(operand, row_stride)``: the raw-pointer
    operand of ``bwd_cast`` (``dkv_acc`` itself when contiguous, else a flat stride-1 alias from its first element over
    ``(S_dst - 1) * row_stride + cols`` elements -- the FFI boundary takes contiguous tensors only, and the kernel
    addresses ``row * row_stride + col``) and the row stride.  Device placement is checked by the callers, so this
    runs on host tensors.
    """
    if not isinstance(dkv_acc, torch.Tensor) or dkv_acc.dtype != torch.float32:
        raise ValueError("dkv_acc must be a float32 tensor")
    if dkv_acc.ndim != 2 or int(dkv_acc.shape[1]) < D_QK:
        raise ValueError(
            f"dkv_acc must be a 2-D [S_dst, >= {D_QK}] tensor (latent columns 0:{D_LATENT}, "
            f"rope columns {D_LATENT}:{D_QK}), got shape {tuple(dkv_acc.shape)}"
        )
    if dkv_acc.stride(1) != 1:
        raise ValueError(
            f"dkv_acc rows must be contiguous (stride(1) == 1), got strides {tuple(dkv_acc.stride())}"
        )
    row_stride = int(dkv_acc.stride(0))
    if row_stride < D_QK or row_stride % 4:
        raise ValueError(
            f"dkv_acc row stride must be >= {D_QK} elements and a multiple of 4 (16-byte vectors), got {row_stride}"
        )
    if dkv_acc.data_ptr() % 16:
        raise ValueError("dkv_acc must start at a 16-byte-aligned address")
    if dkv_dst_map is None:
        if int(dkv_acc.shape[0]) < int(num_kv):
            raise ValueError(
                f"dkv_acc needs at least S = {num_kv} rows without a destination map, got {dkv_acc.shape[0]}"
            )
    elif (
        not isinstance(dkv_dst_map, torch.Tensor)
        or dkv_dst_map.dtype != torch.int32
        or tuple(dkv_dst_map.shape) != (int(num_kv),)
        or not dkv_dst_map.is_contiguous()
        or dkv_dst_map.device != dkv_acc.device
    ):
        raise ValueError(
            f"dkv_dst_map must be a contiguous int32 [S] tensor (S = {num_kv}) on the device of dkv_acc"
        )
    elif os.environ.get("FLASHINFER_CAKE_DSA_CHECK_DST_MAP", "0") not in ("", "0"):
        num_rows = int(dkv_acc.shape[0])
        if bool(((dkv_dst_map < 0) | (dkv_dst_map >= num_rows)).any().item()):
            raise ValueError(
                f"dkv_dst_map values must lie in [0, {num_rows}) (the rows of dkv_acc); "
                f"got min {int(dkv_dst_map.min())}, max {int(dkv_dst_map.max())}"
            )
    if dkv_acc.is_contiguous():
        return dkv_acc, row_stride
    span = (int(dkv_acc.shape[0]) - 1) * row_stride + int(dkv_acc.shape[1])
    return dkv_acc.as_strided((span,), (1,), dkv_acc.storage_offset()), row_stride


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return -(-int(nbytes) // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


def workspace_layout(
    num_queries: int,
    num_kv: int,
    topk: int,
    *,
    backward: bool = True,
    key_passes: int = 1,
    dkv_direct: bool = False,
    token_chunk: Optional[int] = None,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    The ``topk_length`` region backs a full-length vector when the caller
    passes none; ``delta`` and the FP32 dK/dV accumulators exist for the
    backward (a direct binding, ``dkv_direct``, adds into the caller's
    ``dkv_acc`` and has no accumulators).  A backward with more than one
    key-range pass adds the FP32 dQ partials (147,456 B per token), the
    compacted keys of one pass (``topk`` int32 per token) and the per-token
    counts for the tokens of one launch chunk: ``min(num_queries,
    token_chunk)`` rows (the whole row without a ``token_chunk``).
    """
    sizes = [("topk_length", num_queries * 4)]
    if backward:
        sizes.append(("delta", num_queries * NUM_HEADS * 4))
        if not dkv_direct:
            sizes += [
                ("dkv_latent_acc", num_kv * D_LATENT * 4),
                ("dk_rope_acc", num_kv * D_ROPE * 4),
            ]
        if int(key_passes) > 1:
            rows = (
                int(num_queries)
                if token_chunk is None
                else min(int(num_queries), int(token_chunk))
            )
            sizes += [
                ("dq_partial", rows * DQ_PARTIAL_BYTES_PER_TOKEN),
                ("key_scratch", rows * int(topk) * 4),
                ("pass_counts", rows * 4),
            ]
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def dsa_train_workspace_size(
    num_queries: int,
    num_kv: int,
    topk: int,
    device: Optional[torch.device] = None,
    *,
    backward: bool = True,
    key_passes: Optional[int] = None,
    num_segments: int = 1,
    dkv_acc: bool = False,
) -> int:
    """Workspace bytes :func:`prepare_dsa_train` needs for ``(T, S, topk)`` on ``device``.

    ``key_passes`` / ``num_segments`` as in :func:`prepare_dsa_train`
    (``None`` = the record's policy; the packed segment count of the key row);
    ``dkv_acc`` = the binding will accumulate into a caller-provided ``dkv_acc``
    (a direct binding needs no FP32 accumulators; see :func:`plan_dkv_direct`).
    """
    _, record = record_for(device)
    arch = arch_for(device)
    stages = cake_jit.registered_stages()
    direct = bool(backward) and plan_dkv_direct(
        record, stages, num_queries, num_kv, accumulate=bool(dkv_acc)
    )
    passes = (
        plan_key_passes(
            record,
            stages,
            num_queries,
            num_kv,
            topk,
            key_passes,
            num_segments=num_segments,
            arch=arch,
        )
        if backward
        else 1
    )
    return int(
        workspace_layout(
            num_queries,
            num_kv,
            topk,
            backward=backward,
            key_passes=passes,
            dkv_direct=direct,
            token_chunk=_record_token_chunk(record, topk),
        )["total"]
    )


def _alloc(shape, dtype, device, *, zero: bool = False) -> torch.Tensor:
    """The backend's allocation site: outputs of the functional entry points, the
    per-call backward scratch of the eager path and the workspace of a prepared
    runner come from the caching allocator through it (``zero``: filled on the
    stream in the same op)."""
    if zero:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * dtype.itemsize
    if needed > nbytes:
        raise ValueError(
            f"workspace region {name!r} holds {nbytes} bytes, {needed} needed"
        )
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Varlen index offsetting (host glue of the varlen entry point)
# ---------------------------------------------------------------------------


def offset_gather_kv_indices(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    return_topk_length: bool = False,
):
    """Turn per-document key indices into global key rows.

    ``gather_kv_indices [T, topk]`` holds, for query row ``t`` of document
    ``d``, key positions relative to the document's first key
    (``cu_seqlens_k[d]``); ``-1`` or a position ``>= seqlen_k[d]`` is invalid.
    Query and key lengths of a document may differ: the query segment is the
    tail of its key prefix, query ``local_q = t - cu_seqlens_q[d]`` sitting at
    key position ``(seqlen_k[d] - seqlen_q[d]) + local_q``.  With
    ``causal=True`` a slot is also invalid when it selects a key after that
    position (``idx <= (seqlen_k[d] - seqlen_q[d]) + local_q`` is required);
    the default ``causal=False`` is the plain offsetting of the first release:
    the index row is taken as is and only ``-1`` / out-of-range slots are
    dropped.  The result addresses the packed ``kv_*`` tensors; invalid slots
    become ``-1``.  A zero-length query segment contributes no rows.

    This glue counts in the step time of the varlen entry and is launch-bound
    (a ``[T, topk]`` pass is 5-10 us of GPU time against ~10 us of dispatch),
    so it runs with as few launches as the rule allows: a handful of ``[T]``
    int32 ops for the per-row document data, then two compares into one bool
    mask (``0 <= idx`` and ``idx <= bound`` -- the bound is the row's own key
    position with ``causal``, itself ``< seqlen_k``, and ``seqlen_k - 1``
    otherwise) and one ``where`` over ``idx + key_base``.  With
    ``return_topk_length=True`` the per-row ``topk_length`` (last valid slot
    + 1; bitwise what :func:`derive_topk_length` computes from the result) is
    derived from the same mask with one ``where`` and one ``amax`` and the
    function returns ``(indices, topk_length)``.  Runs on device without a
    host synchronization, in int32 throughout.
    """
    if gather_kv_indices.ndim != 2 or gather_kv_indices.dtype != torch.int32:
        raise ValueError("gather_kv_indices must be an int32 [T, topk] tensor")
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.ndim != 1 or cu.dtype != torch.int32 or cu.numel() < 2:
            raise ValueError(f"{name} must be an int32 [num_docs + 1] tensor")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must describe the same documents"
        )
    total_q, topk = (int(d) for d in gather_kv_indices.shape)
    device = gather_kv_indices.device
    # per-row document data: [T]-sized, int32 like cu_seqlens (key rows and positions fit int32)
    rows = torch.arange(total_q, dtype=torch.int32, device=device)
    doc_of_row = torch.bucketize(rows, cu_seqlens_q[1:], right=True)
    key_base = cu_seqlens_k[:-1][doc_of_row]
    if causal:
        # Own key position of row t: (seqlen_k[d] - seqlen_q[d]) + (t - cu_seqlens_q[d])
        # == (cu_seqlens_k[d + 1] - cu_seqlens_q[d + 1]) + t - key_base.
        bound = (cu_seqlens_k[1:] - cu_seqlens_q[1:])[doc_of_row] + rows - key_base
    else:
        bound = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])[doc_of_row] - 1
    local = gather_kv_indices
    # [T, topk] passes: two int32 compares into one bool mask, one add, one where
    valid = local >= 0
    valid &= local <= bound[:, None]
    if out is None:
        result = torch.where(valid, local + key_base[:, None], -1)
    else:
        # the out= overload takes tensors only
        result = torch.where(
            valid, local + key_base[:, None], local.new_full((), -1), out=out
        )
    if not return_topk_length:
        return result
    topk_length = torch.where(valid, _slot_positions(topk, device), 0).amax(dim=-1)
    return result, topk_length


_SLOT_POSITIONS: dict[tuple, torch.Tensor] = {}


def _slot_positions(topk: int, device: torch.device) -> torch.Tensor:
    """Cached ``[1 .. topk]`` int32 on ``device`` (constant per problem)."""
    key = (int(topk), str(device))
    pos = _SLOT_POSITIONS.get(key)
    if pos is None:
        pos = torch.arange(1, int(topk) + 1, device=device, dtype=torch.int32)
        _SLOT_POSITIONS[key] = pos
    return pos


def derive_topk_length(indices: torch.Tensor, num_kv: int) -> torch.Tensor:
    """``[T]`` int32: position of the last valid slot + 1 per row (0 for a fully masked row).

    The public entries' default when the caller passes no ``topk_length``: a slot
    is valid iff ``0 <= idx < num_kv`` (``idx == clamp(idx, 0, num_kv - 1)`` for
    int32 ``idx``), every slot past the last valid one is invalid, so the kernels
    may skip the trailing invalid key blocks while the masking semantics -- and
    the results -- stay those of the full row.  Four small launches (clamp, eq,
    where, amax) on device, no host synchronization.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("indices must be an int32 [T, topk] tensor")
    T, topk = indices.shape
    S = int(num_kv)
    if S < 1:
        return torch.zeros((T,), dtype=torch.int32, device=indices.device)
    pos = _slot_positions(topk, indices.device)
    valid = indices == indices.clamp(0, S - 1)
    return torch.where(valid, pos, 0).amax(dim=-1)


# ---------------------------------------------------------------------------
# Launches
# ---------------------------------------------------------------------------


def _pointer_operand(
    tensor: torch.Tensor, granule: int = POINTER_GRANULE
) -> tuple[torch.Tensor, int]:
    """``(view, element offset)`` of a raw-pointer operand.

    The kernels' vector loads are gated on the row stride and the element
    offset they receive (``((stride | offset) & 7) == 0``) and assume a vector
    aligned pointer.  A tensor that starts inside its storage is therefore
    passed as the zero-copy view that starts at the nearest aligned element
    below it, plus the offset the kernel adds; a tensor whose start is aligned
    is passed as is.
    """
    offset = tensor.storage_offset() % granule
    if offset == 0:
        return tensor, 0
    view = tensor.as_strided(
        tensor.shape, tensor.stride(), tensor.storage_offset() - offset
    )
    return view, offset


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


@dataclass(frozen=True)
class _StageEntry:
    """The loaded entry of one stage for one architecture: generated launcher, binding entry, grid."""

    stage: str
    launcher: Callable[..., None] = field(repr=False)
    run: Callable[..., Any] = field(repr=False)
    grid: tuple[int, int, int]


@dataclass(frozen=True)
class _Launch:
    """One stage launch: the generated positional launcher over bound host values."""

    stage: str
    launcher: Callable[..., None] = field(repr=False)
    run: Callable[..., Any] = field(repr=False)
    values: dict[str, Any] = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.launcher(self.run, self.values, self.grid)


def _stage_entry(stage: str, arch: str, scalars: dict) -> _StageEntry:
    """Resolve ``stage`` for ``arch``: the generated launcher, its grid and the entry of the loaded module."""
    grid = cake_launch.GRID[stage](
        scalars["num_queries"], scalars["num_kv"], scalars["topk"]
    )
    cluster = cake_jit.record()[stage]["launch"]["cluster"]
    if any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape "
            f"{tuple(cluster)} baked into the module"
        )
    module = cake_jit.load_cake_dsa_train_module(stage, arch)
    entry = getattr(module, cake_jit.record()[stage]["ffi_entry"])
    return _StageEntry(stage, cake_launch.LAUNCH[stage], entry, grid)


# ---------------------------------------------------------------------------
# Plans: the tensor-free part of a binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Plan:
    """Everything a binding of one input geometry needs that does not depend on
    which storage the tensors live in: the validated sizes, the key-range
    passes, the workspace layout, the launch constants, and the resolved stage
    entries.  :func:`prepare_dsa_train` binds a plan to the caller's tensors;
    the eager entry points keep plans in :data:`BINDING_CACHE` and bind them to
    the tensors of every call.
    """

    module_name: str
    arch: str
    stages: tuple[str, ...]
    num_queries: int
    num_kv: int
    topk: int
    softmax_scale: float
    backward: bool
    dkv_fp32: bool
    has_topk_length: bool
    key_passes: int
    # The backward adds the dK/dV gradients into the caller's packed FP32 ``dkv_acc``
    # (row stride and map presence baked into the accumulating launch's constants).
    accumulate_dkv: bool
    # ... from the natural-layout main stage itself (no accumulators, no cast launch).
    dkv_direct: bool
    # The forward kernel derives the row lengths into the ``topk_length`` vector it receives.
    derive_length: bool
    num_segments: int
    device_index: int
    layout: dict = field(repr=False)
    scalars: dict = field(repr=False)
    # Launch values that follow from the geometry alone (every stage selects from them).
    constants: dict[str, Any] = field(repr=False)
    # ``(pass_lo, pass_hi, dq_mode, token_base, num_tokens)`` overrides of every
    # (token chunk, key-range pass) of a multi-pass backward (empty for one pass).
    pass_overrides: tuple[tuple[dict[str, int], ...], ...]
    # stage name -> entry; the pass stages of a multi-pass backward are keyed ``(stage, chunk index)``
    entries: dict[Any, _StageEntry] = field(repr=False)
    backward_order: tuple

    def launch_keys(self) -> tuple:
        return tuple(FORWARD_STAGES) + self.backward_order

    @property
    def pass_rows(self) -> int:
        """Token rows of the pass scratch regions (one launch chunk; 0 without passes)."""
        if "dq_partial" not in self.layout:
            return 0
        return self.layout["dq_partial"][1] // DQ_PARTIAL_BYTES_PER_TOKEN


def _geometry(t: Optional[torch.Tensor]) -> Optional[tuple]:
    """What validation and the launch constants read from a tensor: shape, strides,
    dtype, device and the vector alignment of its start (``None`` for an absent one)."""
    if t is None:
        return None
    return (
        t.shape,
        t.stride(),
        t.dtype,
        t.get_device(),
        t.storage_offset() % POINTER_GRANULE,
    )


def _main_accumulator_constants(
    direct: bool, dst_row_stride: int, has_map: bool
) -> dict[str, int]:
    """The main stage's accumulator operands (every main-stage program declares them; the permuted program reads
    them inert): the private permuted accumulators -- 512 / 64 FP32 elements per row, rope column 0, no map -- or,
    for a direct binding, the caller's packed rows: their row stride for both, rope column 512, the map flag."""
    if direct:
        return dict(
            dkv_stride=int(dst_row_stride),
            dkr_stride=int(dst_row_stride),
            dkr_col0=D_LATENT,
            dkv_has_map=int(bool(has_map)),
        )
    return dict(dkv_stride=D_LATENT, dkr_stride=D_ROPE, dkr_col0=0, dkv_has_map=0)


def _plan(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor],
    dout: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    outputs: dict[str, Optional[torch.Tensor]],
    dkv_fp32: bool,
    backward: bool,
    key_passes: Optional[int],
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    derive_topk_length: bool = False,
) -> _Plan:
    """Validate one geometry and resolve its plan (no allocation, no launch)."""
    if derive_topk_length and topk_length is None:
        raise ValueError(
            "derive_topk_length=True needs a topk_length [T] int32 tensor for the forward to fill"
        )
    num_queries, num_kv, topk = validate_dsa_train_inputs(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length, dout=dout
    )
    if backward and dout is None:
        raise ValueError("a backward binding needs dout")
    accumulate = dkv_acc is not None
    dst_row_stride = 0
    if accumulate:
        if not backward:
            raise ValueError(
                "dkv_acc belongs to the backward: prepare the binding with dout (backward=True)"
            )
        if dkv_fp32:
            raise ValueError(
                "dkv_acc accumulates the dK/dV gradients in place; dkv_fp32 outputs are not produced with it"
            )
        if outputs.get("dkv_latent") is not None or outputs.get("dk_rope") is not None:
            raise ValueError(
                "dkv_acc accumulates in place: no dkv_latent / dk_rope output tensors are written with it"
            )
        _, dst_row_stride = _check_dkv_acc(dkv_acc, dkv_dst_map, num_kv)
    elif dkv_dst_map is not None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
    device = q_latent.device
    tensors = [q_latent, q_rope, kv_latent, k_rope, indices, topk_length, dout]
    tensors += [dkv_acc, dkv_dst_map]
    tensors += list(outputs.values())
    if not all(t.is_cuda and t.device == device for t in tensors if t is not None):
        raise ValueError("Expected all tensors on one CUDA device")
    module_name, record = record_for(device)
    arch = arch_for(device)
    stages = cake_jit.registered_stages()
    if backward and not generated_program_available(device, backward=True):
        raise NotImplementedError(
            f"the registered DSA training program {module_name!r} has no backward stages "
            f"(registered: {stages}); forward-only use is available"
        )
    # direct accumulation: the natural-layout main stage adds into dkv_acc when the record registers it and its
    # size rule (or the forced mode) selects it; otherwise the cast's accumulate path
    direct = bool(backward) and plan_dkv_direct(
        record, stages, num_queries, num_kv, accumulate=accumulate
    )
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    passes = (
        plan_key_passes(
            record,
            stages,
            num_queries,
            num_kv,
            topk,
            key_passes,
            num_segments=num_segments,
            arch=arch,
        )
        if backward
        else 1
    )
    _check_output(
        outputs.get("out"), "out", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16
    )
    _check_output(outputs.get("lse"), "lse", (num_queries, NUM_HEADS), torch.float32)
    _check_output(
        outputs.get("o_lo"), "o_lo", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16
    )
    _check_output(
        outputs.get("dq_latent"),
        "dq_latent",
        (num_queries, NUM_HEADS, D_LATENT),
        torch.bfloat16,
    )
    _check_output(
        outputs.get("dq_rope"),
        "dq_rope",
        (num_queries, NUM_HEADS, D_ROPE),
        torch.bfloat16,
    )
    _check_output(
        outputs.get("dkv_latent"), "dkv_latent", (num_kv, D_LATENT), torch.bfloat16
    )
    _check_output(outputs.get("dk_rope"), "dk_rope", (num_kv, D_ROPE), torch.bfloat16)
    layout = workspace_layout(
        num_queries,
        num_kv,
        topk,
        backward=backward,
        key_passes=passes,
        dkv_direct=direct,
        token_chunk=_record_token_chunk(record, topk),
    )
    has_topk_length = topk_length is not None
    # the forward derives the row lengths into the vector when the caller gives none (or asks for it); the backward
    # stages of a binding without caller lengths keep the full-row semantics (has_topk_length = 0)
    derive_length = topk_length is None or bool(derive_topk_length)
    scalars = dict(
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=float(softmax_scale),
        has_topk_length=int(has_topk_length),
        out_f32=int(bool(backward and dkv_fp32)),
        # bwd_cast packed-accumulate scalars (0 / 0 / 0 unless the binding adds into dkv_acc)
        dst_row_stride=int(dst_row_stride),
        has_dst_map=int(accumulate and dkv_dst_map is not None),
        accumulate=int(accumulate),
    )
    scale = float(softmax_scale)
    T, S = num_queries, num_kv
    constants: dict[str, Any] = dict(
        indices_offset=indices.storage_offset() % POINTER_GRANULE,
        idx_stride=int(indices.stride(0)),
        k_rope_stride=int(k_rope.stride(0)),
        k_rope_offset=0,  # k_rope starts vector aligned (validated)
        num_queries=T,
        num_kv=S,
        topk=topk,
        has_topk_length=int(has_topk_length),
        sm_scale=scale,
        scale_log2=scale * LOG2E,
        num_rows=T * NUM_HEADS,  # bwd_delta: (token, head) rows
        latent_groups=S * D_LATENT // 32,  # bwd_cast: permuted 32-element groups
        rope_groups=S * D_ROPE // 32,
        out_f32=scalars["out_f32"],
        dst_row_stride=scalars["dst_row_stride"],
        has_dst_map=scalars["has_dst_map"],
        accumulate=scalars["accumulate"],
        derive_length=int(derive_length),
        # main-stage accumulator geometry: the private permuted accumulators unless the binding accumulates
        # directly (the caller's packed row stride, rope column 512, the map flag)
        **_main_accumulator_constants(direct, dst_row_stride, dkv_dst_map is not None),
        token_base=0,  # one CTA per token, token = blockIdx.x
        token_step=1,
        num_tokens=T,  # bwd_compact: the whole row; a multi-pass backward overrides it per token chunk
    )
    # per (token chunk, key-range pass) overrides of a multi-pass backward: the key range, ``dq_mode``,
    # the chunk's first token and its token count
    pass_overrides: tuple[tuple[dict[str, int], ...], ...] = ()
    token_chunks: tuple[tuple[int, int], ...] = ()
    if backward and passes == 1:
        # The single-pass kernel never reads the pass operands; it receives, like its
        # production launcher, the whole key range and dq_mode 0.
        constants.update(pass_lo=0, pass_hi=S, dq_mode=0)
    elif backward:
        token_chunks = _pass_token_chunks(record, num_queries, topk)
        ranges = key_pass_ranges(num_kv, passes)
        pass_overrides = tuple(
            tuple(
                dict(
                    pass_lo=lo,
                    pass_hi=hi,
                    dq_mode=key_pass_dq_mode(index, passes),
                    token_base=int(base),
                    num_tokens=int(num_tokens),
                )
                for index, (lo, hi) in enumerate(ranges)
            )
            for base, num_tokens in token_chunks
        )
    entries: dict[Any, _StageEntry] = {"fwd": _stage_entry("fwd", arch, scalars)}
    backward_order: list = []
    if backward:
        # the natural-layout variants of the main stage replace the permuted ones on a direct binding
        main_stage = "bwd_main_natural" if direct else "bwd_main"
        pass_stages = (
            "bwd_compact",
            "bwd_main_pass_natural" if direct else "bwd_main_pass",
        )
        entries["bwd_delta"] = _stage_entry("bwd_delta", arch, scalars)
        backward_order.append("bwd_delta")
        if passes == 1:
            entries[main_stage] = _stage_entry(main_stage, arch, scalars)
            backward_order.append(main_stage)
        else:
            # token chunks in order, every chunk through every pass (the Cake launcher's loop);
            # the grid rules (``num_queries`` / ``num_queries/W``) take the chunk's token count
            for chunk_index, (_base, num_tokens) in enumerate(token_chunks):
                chunk_scalars = dict(scalars, num_queries=int(num_tokens))
                for stage in pass_stages:
                    entries[(stage, chunk_index)] = _stage_entry(
                        stage, arch, chunk_scalars
                    )
                for index in range(passes):
                    backward_order += [
                        (stage, chunk_index, index) for stage in pass_stages
                    ]
        # a direct binding's gradients are complete after the main stage: no cast
        if not direct:
            entries["bwd_cast"] = _stage_entry("bwd_cast", arch, scalars)
            backward_order.append("bwd_cast")
    return _Plan(
        module_name=module_name,
        arch=arch,
        stages=stages,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=scale,
        backward=bool(backward),
        dkv_fp32=bool(dkv_fp32),
        has_topk_length=has_topk_length,
        key_passes=int(passes),
        accumulate_dkv=bool(accumulate),
        dkv_direct=bool(direct),
        derive_length=bool(derive_length),
        num_segments=int(num_segments),
        device_index=_device_index(device),
        layout=layout,
        scalars=scalars,
        constants=constants,
        pass_overrides=pass_overrides,
        entries=entries,
        backward_order=tuple(backward_order),
    )


def _bound_values(plan: _Plan, t: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Launch values of ``plan`` over the tensors ``t`` (the kernels' own argument names)."""
    indices, _ = _pointer_operand(t["indices"])
    values: dict[str, Any] = dict(plan.constants)
    values.update(t, indices=indices)
    if plan.backward:
        # the destination-row map of the main stage (and the cast): the caller's map, else a
        # never-dereferenced int32 placeholder (dkv_has_map = has_dst_map = 0), as the
        # production launcher passes it
        dst_map = (
            t["dkv_dst_map"] if t.get("dkv_dst_map") is not None else t["topk_length"]
        )
        values["dkv_dst_map"] = dst_map
        if plan.dkv_direct:
            # the natural-layout main stage's accumulator operands are the caller's packed
            # rows (the flat alias of dkv_acc, addressed through dkv_stride / dkr_stride /
            # dkr_col0); no cast is launched, so its operands are not bound
            values.update(dkv_f32=t["dkv_acc"], dkr_f32=t["dkv_acc"])
        else:
            acc_latent, acc_rope = t["dkv_latent_acc"], t["dk_rope_acc"]
            values.update(
                dkv_f32=acc_latent,
                dkr_f32=acc_rope,
                src_latent=acc_latent,
                src_rope=acc_rope,
                dst_latent=t["dkv_latent"],
                dst_rope=t["dk_rope"],
                dst_latent_f32=t["dkv_latent_fp32"],
                dst_rope_f32=t["dk_rope_fp32"],
                # bwd_cast packed-accumulate operands: the caller's dkv_acc (its flat alias) and
                # destination map, or never-dereferenced placeholders of the right dtypes
                # (accumulate = 0 / has_dst_map = 0), as the production launcher passes them
                dst_packed=t["dkv_acc"] if plan.accumulate_dkv else acc_latent,
                dst_map=dst_map,
            )
        if plan.key_passes == 1:
            # never-dereferenced placeholders of the single-pass kernel: contiguous
            # tensors of the argument dtypes
            values.update(
                dq_partial=t["delta"],
                key_scratch=t["topk_length"],
                pass_counts=t["topk_length"],
            )
    return values


def _launches(plan: _Plan, values: dict[str, Any]) -> dict[Any, _Launch]:
    """The launches of ``plan`` over ``values``, keyed as :class:`DSATrainRunner` documents."""
    launches: dict[Any, _Launch] = {}
    for key in plan.launch_keys():
        stage, stage_values = key, values
        if isinstance(key, tuple):
            stage, chunk, index = key
            stage_values = dict(values, **plan.pass_overrides[chunk][index])
            e = plan.entries[(stage, chunk)]
        else:
            e = plan.entries[stage]
        launches[key] = _Launch(stage, e.launcher, e.run, stage_values, e.grid)
    return launches


def _scratch(plan: _Plan, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """The backward scratch regions of ``plan`` carved from the workspace ``flat``."""
    T, S, topk, layout = plan.num_queries, plan.num_kv, plan.topk, plan.layout
    t = dict(delta=_carve(flat, layout, "delta", torch.float32, (T, NUM_HEADS)))
    # a direct binding adds into dkv_acc: no private accumulators
    if not plan.dkv_direct:
        t["dkv_latent_acc"] = _carve(
            flat, layout, "dkv_latent_acc", torch.float32, (S, D_LATENT)
        )
        t["dk_rope_acc"] = _carve(
            flat, layout, "dk_rope_acc", torch.float32, (S, D_ROPE)
        )
    if plan.key_passes > 1:
        rows = plan.pass_rows  # the tokens of one launch chunk
        t.update(
            dq_partial=_carve(
                flat,
                layout,
                "dq_partial",
                torch.float32,
                (rows, DQ_PARTIAL_BYTES_PER_TOKEN // 4),
            ),
            key_scratch=_carve(flat, layout, "key_scratch", torch.int32, (rows, topk)),
            pass_counts=_carve(flat, layout, "pass_counts", torch.int32, (rows,)),
        )
    return t


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DSATrainRunner:
    """The prepared forward / backward launches of one tensor binding.

    ``forward()`` writes ``out``, ``lse`` and ``o_lo``; ``backward()`` writes
    ``dq_latent``, ``dq_rope`` and ``dkv_latent`` / ``dk_rope`` (BF16 casts of
    the FP32 accumulators, or natural-layout FP32 outputs when prepared with
    ``dkv_fp32=True``) -- or, prepared with ``dkv_acc``, adds the dK/dV
    gradients into that buffer and returns ``None`` for both (``accumulate_dkv``);
    ``step()`` runs both.  No launch allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype or tensor binding changes; values may change
    freely.

    ``launches`` is keyed by stage name; the launches of a multi-pass backward
    are keyed ``(stage, chunk index, pass index)`` and ``backward_order`` lists
    every backward launch key in launch order (``bwd_delta``, then per token
    chunk and per pass ``bwd_compact`` and ``bwd_main_pass`` -- or the single
    ``bwd_main`` --, then ``bwd_cast``; a direct binding, ``dkv_direct``,
    launches the natural-layout variants of the main stage and no cast).
    """

    module_name: str
    num_queries: int
    num_kv: int
    topk: int
    softmax_scale: float
    tensors: dict[str, torch.Tensor] = field(repr=False)
    launches: dict[Any, _Launch] = field(repr=False)
    stages: tuple[str, ...]
    dkv_fp32: bool
    has_topk_length: bool  # False: ``tensors["topk_length"]`` is the workspace vector filled with ``topk``
    device_index: int
    # The flat workspace the scratch regions of ``tensors`` are carved from, and their byte layout.
    workspace: torch.Tensor = field(repr=False)
    layout: dict = field(repr=False)
    # Key-range passes of the backward main stage (1 = the single-pass stage) and the backward launch keys in order.
    key_passes: int = 1
    backward_order: tuple = ()
    # True when the dK/dV gradients are added into the caller's packed FP32 rows (``tensors["dkv_acc"]``).
    accumulate_dkv: bool = False
    # True when the natural-layout main stage adds into ``dkv_acc`` itself (no accumulators, no zero fill, no cast).
    dkv_direct: bool = False

    @property
    def out(self) -> torch.Tensor:
        return self.tensors["out"]

    @property
    def lse(self) -> torch.Tensor:
        return self.tensors["lse"]

    @property
    def o_lo(self) -> torch.Tensor:
        return self.tensors["o_lo"]

    @property
    def has_backward(self) -> bool:
        return bool(self.backward_order)

    def _run(self, keys: tuple) -> None:
        launches = self.launches
        with _ffi_stream_context(self.device_index):
            for key in keys:
                launches[key]()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._run(FORWARD_STAGES)
        t = self.tensors
        return t["out"], t["lse"], t["o_lo"]

    def backward(
        self,
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        if not self.has_backward:
            raise NotImplementedError(
                "this runner was prepared without the backward (pass dout / backward=True)"
            )
        t = self.tensors
        if not self.dkv_direct:
            t["dkv_latent_acc"].zero_()
            t["dk_rope_acc"].zero_()
        self._run(self.backward_order)
        if self.accumulate_dkv:
            # the gradients were added into the caller's dkv_acc rows: no dK/dV output tensors
            return t["dq_latent"], t["dq_rope"], None, None
        if self.dkv_fp32:
            return t["dq_latent"], t["dq_rope"], t["dkv_latent_fp32"], t["dk_rope_fp32"]
        return t["dq_latent"], t["dq_rope"], t["dkv_latent"], t["dk_rope"]

    def step(self):
        self.forward()
        return self.backward()

    __call__ = step


def prepare_dsa_train(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    dout: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    o_lo: Optional[torch.Tensor] = None,
    dq_latent: Optional[torch.Tensor] = None,
    dq_rope: Optional[torch.Tensor] = None,
    dkv_latent: Optional[torch.Tensor] = None,
    dk_rope: Optional[torch.Tensor] = None,
    dkv_fp32: bool = False,
    backward: Optional[bool] = None,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    derive_topk_length: bool = False,
    backend: str = "cake",
) -> DSATrainRunner:
    """Validate one binding and prepare its launches.

    ``backward`` defaults to ``dout is not None``.  Missing outputs and the
    workspace are allocated here (the only allocations of a prepared runner).
    Pass ``workspace_buffer`` of :func:`dsa_train_workspace_size` bytes to
    reuse storage across steps.  ``dkv_fp32=True`` makes ``backward()`` return
    natural-layout FP32 dK/dV gradients.  ``key_passes`` overrides the record's
    key-range-pass policy for the backward (``None`` = policy, 1 = the
    single-pass stage; see :func:`plan_key_passes`); ``num_segments`` is the
    packed segment count of the key row the policy plans with.  ``dkv_acc``
    (FP32 ``[S_dst, >= 576]``, see :func:`_check_dkv_acc`) makes ``backward()``
    ADD the dK/dV gradients into it -- latent columns ``0:512``, rope columns
    ``512:576`` of row ``dkv_dst_map[s]`` (optional int32 ``[S]``; identity by
    default; repeated rows sum) -- and return ``None`` for ``dkv_latent`` /
    ``dk_rope``; the caller owns zeroing.  It excludes ``dkv_fp32``.  A program
    registering the natural-layout main stages accumulates directly from the
    main stage when the record's size rule selects it (:func:`plan_dkv_direct`:
    no FP32 accumulators, no cast launch); otherwise the cast adds the
    gradients.

    Row lengths: with ``topk_length`` the kernels use the caller's lengths.
    Without it the forward kernel derives them itself (last valid slot + 1 per
    row) into the binding's own ``[T]`` vector while it runs and skips the
    trailing invalid key blocks; the backward stages of such a binding treat
    every slot as potentially valid (the results are those of the full row
    either way).  ``derive_topk_length=True`` makes the forward derive the
    lengths into the caller's ``topk_length`` tensor instead (its contents are
    ignored on entry), so a training step can hand them to its backward.
    """
    if backend != "cake":
        raise ValueError("DSA sparse-attention training supports backend='cake'")
    if backward is None:
        backward = dout is not None
    outputs = dict(
        out=out,
        lse=lse,
        o_lo=o_lo,
        dq_latent=dq_latent,
        dq_rope=dq_rope,
        dkv_latent=dkv_latent,
        dk_rope=dk_rope,
    )
    plan = _plan(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        dout=dout,
        softmax_scale=softmax_scale,
        outputs=outputs,
        dkv_fp32=dkv_fp32,
        backward=backward,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=num_segments,
        derive_topk_length=derive_topk_length,
    )
    device = q_latent.device
    layout = plan.layout
    if workspace_buffer is None:
        workspace_buffer = _alloc(layout["total"], torch.uint8, device)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}"
        )
    num_queries, num_kv = plan.num_queries, plan.num_kv

    def fresh(shape, dtype):
        return _alloc(shape, dtype, device)

    t: dict[str, torch.Tensor] = dict(
        q_latent=q_latent,
        q_rope=q_rope,
        kv_latent=kv_latent,
        k_rope=k_rope,
        indices=indices,
    )
    if topk_length is None:
        # the forward derives the row lengths into this vector while it runs; nothing reads them back (the
        # backward of a binding without caller lengths keeps the full-row semantics), the fill is the full row
        topk_length = _carve(flat, layout, "topk_length", torch.int32, (num_queries,))
        topk_length.fill_(plan.topk)
    t["topk_length"] = topk_length
    t["out"] = (
        out
        if out is not None
        else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    )
    t["lse"] = (
        lse if lse is not None else fresh((num_queries, NUM_HEADS), torch.float32)
    )
    t["o_lo"] = (
        o_lo
        if o_lo is not None
        else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    )
    if backward:
        t["dout"] = dout
        t.update(_scratch(plan, flat))
        t["dq_latent"] = (
            dq_latent
            if dq_latent is not None
            else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
        )
        t["dq_rope"] = (
            dq_rope
            if dq_rope is not None
            else fresh((num_queries, NUM_HEADS, D_ROPE), torch.bfloat16)
        )
        if plan.accumulate_dkv:
            t["dkv_acc"], _row_stride = _check_dkv_acc(dkv_acc, dkv_dst_map, num_kv)
            if dkv_dst_map is not None:
                t["dkv_dst_map"] = dkv_dst_map
        t.update(_dkv_outputs(plan, t, dkv_latent, dk_rope, fresh))
    values = _bound_values(plan, t)
    return DSATrainRunner(
        module_name=plan.module_name,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=plan.topk,
        softmax_scale=plan.softmax_scale,
        tensors=t,
        launches=_launches(plan, values),
        stages=plan.stages,
        dkv_fp32=plan.dkv_fp32,
        has_topk_length=plan.has_topk_length,
        device_index=plan.device_index,
        workspace=flat,
        layout=layout,
        key_passes=plan.key_passes,
        backward_order=plan.backward_order,
        accumulate_dkv=plan.accumulate_dkv,
        dkv_direct=plan.dkv_direct,
    )


def _dkv_outputs(
    plan: _Plan,
    t: dict[str, torch.Tensor],
    dkv_latent: Optional[torch.Tensor],
    dk_rope: Optional[torch.Tensor],
    fresh: Callable[..., torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The dK/dV output tensors of a backward binding (BF16 casts, or natural-layout FP32)."""
    S = plan.num_kv
    if plan.dkv_direct:
        # the natural-layout main stage adds into the caller's rows; no cast runs, so there
        # are no dK/dV outputs at all
        return dict(
            dkv_latent=fresh((0,), torch.bfloat16),
            dk_rope=fresh((0,), torch.bfloat16),
        )
    if plan.accumulate_dkv:
        # the cast adds into the caller's rows (accumulate = 1): its BF16 / FP32 output
        # pointers are not dereferenced
        return dict(
            dkv_latent=fresh((0,), torch.bfloat16),
            dk_rope=fresh((0,), torch.bfloat16),
            dkv_latent_fp32=t["dkv_latent_acc"],
            dk_rope_fp32=t["dk_rope_acc"],
        )
    if plan.dkv_fp32:
        # natural-layout FP32 outputs written by bwd_cast (out_f32 = 1); its BF16 output
        # pointers are not dereferenced
        return dict(
            dkv_latent_fp32=fresh((S, D_LATENT), torch.float32),
            dk_rope_fp32=fresh((S, D_ROPE), torch.float32),
            dkv_latent=fresh((0,), torch.bfloat16),
            dk_rope=fresh((0,), torch.bfloat16),
        )
    # the cast's FP32 output pointers are not dereferenced when out_f32 = 0: alias the accumulators
    return dict(
        dkv_latent=dkv_latent
        if dkv_latent is not None
        else fresh((S, D_LATENT), torch.bfloat16),
        dk_rope=dk_rope if dk_rope is not None else fresh((S, D_ROPE), torch.bfloat16),
        dkv_latent_fp32=t["dkv_latent_acc"],
        dk_rope_fp32=t["dk_rope_acc"],
    )


# ---------------------------------------------------------------------------
# Eager entry points: a bounded cache of plans, bound to the tensors of every call
# ---------------------------------------------------------------------------

# Plans the eager entry points remember, keyed by the input geometry (see
# :func:`_geometry`) and the call options.  A remembered plan owns no
# problem-sized storage: only the full-length ``topk_length`` vector it
# materializes when the caller passes none (4 bytes per query row).  Outputs
# and the backward scratch come from the caching allocator per call, so a
# call never shares storage with another call that is still in flight.
BINDING_CACHE_DEFAULT_CAPACITY = 64
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY"


def binding_cache_capacity() -> int:
    """Capacity of :data:`BINDING_CACHE`: ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY``
    when set, else :data:`BINDING_CACHE_DEFAULT_CAPACITY`."""
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    if raw is None or not raw.strip():
        return BINDING_CACHE_DEFAULT_CAPACITY
    try:
        capacity = int(raw)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(
            f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
        )
    return capacity


def forward_binding_key(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    derive_topk_length: bool = False,
) -> tuple:
    """Cache key of a forward plan: the :func:`_geometry` of every input (``None`` for
    an absent ``topk_length``), the scale and whether the forward derives the lengths
    into the caller's tensor.  Data pointers are not part of it."""
    return (
        "fwd",
        _geometry(q_latent),
        _geometry(q_rope),
        _geometry(kv_latent),
        _geometry(k_rope),
        _geometry(indices),
        _geometry(topk_length),
        float(softmax_scale),
        bool(derive_topk_length),
    )


def backward_binding_key(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    dkv_fp32: bool,
    key_passes: Optional[int] = None,
    num_segments: int = 1,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
) -> tuple:
    """Cache key of a backward plan: the forward key's inputs plus the saved forward
    outputs, ``dout``, the ``dkv_fp32`` option, the ``key_passes`` override, the
    segment count the pass policy saw and the geometry of the packed accumulator /
    destination map (``None`` when absent; their row stride and presence are baked
    into the accumulating launch) and, with ``dkv_acc``, the direct-accumulation
    mode (:func:`dkv_direct_mode`)."""
    return (
        "bwd",
        _geometry(q_latent),
        _geometry(q_rope),
        _geometry(kv_latent),
        _geometry(k_rope),
        _geometry(indices),
        _geometry(out),
        _geometry(o_lo),
        _geometry(lse),
        _geometry(dout),
        _geometry(topk_length),
        float(softmax_scale),
        bool(dkv_fp32),
        None if key_passes is None else int(key_passes),
        int(num_segments),
        _geometry(dkv_acc),
        _geometry(dkv_dst_map),
        None if dkv_acc is None else dkv_direct_mode(),
    )


class _Binding:
    """A remembered plan plus the one value it owns (the materialized ``topk_length``)."""

    __slots__ = ("plan", "topk_length")

    def __init__(self, plan: _Plan, topk_length: Optional[torch.Tensor]):
        self.plan = plan
        self.topk_length = topk_length

    def _run(self, values: dict[str, Any], keys: tuple) -> None:
        plan = self.plan
        with _ffi_stream_context(plan.device_index):
            for key in keys:
                if isinstance(key, tuple):
                    stage, chunk, index = key
                    e = plan.entries[(stage, chunk)]
                    e.launcher(
                        e.run,
                        dict(values, **plan.pass_overrides[chunk][index]),
                        e.grid,
                    )
                else:
                    e = plan.entries[key]
                    e.launcher(e.run, values, e.grid)

    def forward(self, q_latent, q_rope, kv_latent, k_rope, indices, topk_length):
        T, device = self.plan.num_queries, q_latent.device
        out = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        lse = _alloc((T, NUM_HEADS), torch.float32, device)
        o_lo = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        t = dict(
            q_latent=q_latent,
            q_rope=q_rope,
            kv_latent=kv_latent,
            k_rope=k_rope,
            indices=indices,
            topk_length=topk_length if topk_length is not None else self.topk_length,
            out=out,
            lse=lse,
            o_lo=o_lo,
        )
        self._run(_bound_values(self.plan, t), FORWARD_STAGES)
        return out, lse, o_lo

    def backward(
        self,
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        dkv_acc=None,
        dkv_dst_map=None,
    ):
        plan = self.plan
        T, S, device = plan.num_queries, plan.num_kv, q_latent.device

        def fresh(shape, dtype):
            return _alloc(shape, dtype, device)

        t = dict(
            q_latent=q_latent,
            q_rope=q_rope,
            kv_latent=kv_latent,
            k_rope=k_rope,
            indices=indices,
            topk_length=topk_length if topk_length is not None else self.topk_length,
            out=out,
            lse=lse,
            o_lo=o_lo,
            dout=dout,
            dq_latent=fresh((T, NUM_HEADS, D_LATENT), torch.bfloat16),
            dq_rope=fresh((T, NUM_HEADS, D_ROPE), torch.bfloat16),
        )
        # The backward scratch is allocated per region (the prepared runner carves it
        # from one workspace): the FP32 dK/dV accumulators zero-filled, the rest plain.
        t["delta"] = fresh((T, NUM_HEADS), torch.float32)
        # a direct binding adds into the caller's dkv_acc: no accumulators, no fill
        if not plan.dkv_direct:
            t["dkv_latent_acc"] = _alloc(
                (S, D_LATENT), torch.float32, device, zero=True
            )
            t["dk_rope_acc"] = _alloc((S, D_ROPE), torch.float32, device, zero=True)
        if plan.key_passes > 1:
            rows = plan.pass_rows  # the tokens of one launch chunk
            t["dq_partial"] = fresh(
                (rows, DQ_PARTIAL_BYTES_PER_TOKEN // 4), torch.float32
            )
            t["key_scratch"] = fresh((rows, plan.topk), torch.int32)
            t["pass_counts"] = fresh((rows,), torch.int32)
        if plan.accumulate_dkv:
            if dkv_acc is None:
                raise RuntimeError(
                    "this binding accumulates the dK/dV gradients into dkv_acc and the call provides none"
                )
            # the same key implies the validated shape / stride / dtype; the flat alias is rebuilt per call
            t["dkv_acc"], _row_stride = _check_dkv_acc(dkv_acc, dkv_dst_map, S)
            if dkv_dst_map is not None:
                t["dkv_dst_map"] = dkv_dst_map
        t.update(_dkv_outputs(plan, t, None, None, fresh))
        self._run(_bound_values(plan, t), plan.backward_order)
        if plan.accumulate_dkv:
            return t["dq_latent"], t["dq_rope"], None, None
        if plan.dkv_fp32:
            return t["dq_latent"], t["dq_rope"], t["dkv_latent_fp32"], t["dk_rope_fp32"]
        return t["dq_latent"], t["dq_rope"], t["dkv_latent"], t["dk_rope"]


class _BindingCache:
    """Bounded, lock-protected LRU of :class:`_Binding` by call key."""

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = binding_cache_capacity() if capacity is None else int(capacity)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[tuple, _Binding]" = OrderedDict()

    def get(self, key: tuple) -> Optional[_Binding]:
        with self._lock:
            binding = self._entries.get(key)
            if binding is not None:
                self._entries.move_to_end(key)
            return binding

    def put(self, key: tuple, binding: _Binding) -> None:
        with self._lock:
            self._entries[key] = binding
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


BINDING_CACHE = _BindingCache()


def _binding(key: tuple, make_plan: Callable[[], _Plan]) -> _Binding:
    """The cached binding of ``key``, planned on a miss.

    During CUDA-graph capture the plan is built privately and neither read from
    nor stored in the cache: its ``topk_length`` vector would live in the graph's
    memory pool, and a replay must not share storage with eager calls.
    """
    capturing = torch.cuda.is_current_stream_capturing()
    binding = None if capturing else BINDING_CACHE.get(key)
    if binding is None:
        plan = make_plan()
        topk_length = None
        if not plan.has_topk_length:
            # Everything a cached entry materializes is a function of the key and the
            # call options alone, never of tensor contents: this vector holds ``topk``
            # (a shape) in every row, so calls with other index / activation contents
            # of the same geometry share it.  The forward kernel overwrites it with
            # the lengths it derives (``derive_length``); nothing reads them back --
            # the backward of a binding without caller lengths keeps the full-row
            # semantics -- so concurrent calls may share the vector.
            topk_length = _alloc(
                (plan.num_queries,),
                torch.int32,
                torch.device("cuda", plan.device_index),
            )
            topk_length.fill_(plan.topk)
        binding = _Binding(plan, topk_length)
        if not capturing:
            BINDING_CACHE.put(key, binding)
    return binding


def _no_query_rows(q_latent: torch.Tensor) -> bool:
    return q_latent.ndim >= 1 and int(q_latent.shape[0]) == 0


def _empty_forward(
    q_latent, q_rope, kv_latent, k_rope, indices, topk_length
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(out, lse, o_lo)`` of a call without query rows: validated, empty, neither bound nor launched."""
    validate_dsa_train_inputs(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        allow_empty_queries=True,
    )
    device = q_latent.device
    out = _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)
    lse = _alloc((0, NUM_HEADS), torch.float32, device)
    return out, lse, _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)


def _empty_backward(
    q_latent,
    q_rope,
    kv_latent,
    k_rope,
    indices,
    out,
    o_lo,
    lse,
    dout,
    topk_length,
    dkv_fp32: bool,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Gradients of a call without query rows: empty ``dq``, zero ``dkv`` in the requested dtype (``None`` with
    ``dkv_acc``: nothing to accumulate); nothing launched."""
    validate_dsa_train_inputs(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        dout=dout,
        allow_empty_queries=True,
    )
    _check_output(out, "out", (0, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(o_lo, "o_lo", (0, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(lse, "lse", (0, NUM_HEADS), torch.float32)
    device, num_kv = q_latent.device, int(kv_latent.shape[0])
    dq_latent = _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)
    dq_rope = _alloc((0, NUM_HEADS, D_ROPE), torch.bfloat16, device)
    if dkv_acc is not None:
        _check_dkv_acc(dkv_acc, dkv_dst_map, num_kv)  # nothing to accumulate
        return dq_latent, dq_rope, None, None
    dtype = torch.float32 if dkv_fp32 else torch.bfloat16
    return (
        dq_latent,
        dq_rope,
        _alloc((num_kv, D_LATENT), dtype, device).zero_(),
        _alloc((num_kv, D_ROPE), dtype, device).zero_(),
    )


def forward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    derive_topk_length: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass: ``(out, lse, o_lo)``.

    Allocates the outputs and launches over the remembered plan of the input
    geometry (validated and resolved on the first call of that geometry); a
    call without query rows returns empty outputs without launching.  Without
    ``topk_length`` the kernel derives the row lengths itself;
    ``derive_topk_length=True`` makes it write them into the given
    ``topk_length`` tensor (see :func:`prepare_dsa_train`).
    """
    if derive_topk_length and topk_length is None:
        raise ValueError(
            "derive_topk_length=True needs a topk_length [T] int32 tensor for the forward to fill"
        )
    if _no_query_rows(q_latent):
        return _empty_forward(q_latent, q_rope, kv_latent, k_rope, indices, topk_length)
    scale = default_softmax_scale() if softmax_scale is None else float(softmax_scale)
    key = forward_binding_key(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        scale,
        derive_topk_length,
    )
    binding = _binding(
        key,
        lambda: _plan(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            dout=None,
            softmax_scale=scale,
            outputs={},
            dkv_fp32=False,
            backward=False,
            key_passes=None,
            derive_topk_length=derive_topk_length,
        ),
    )
    return binding.forward(q_latent, q_rope, kv_latent, k_rope, indices, topk_length)


def backward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dkv_fp32: bool = False,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward pass from the saved forward outputs: ``(dq_latent, dq_rope, dkv_latent, dk_rope)``.

    With ``dkv_fp32=True`` the dK/dV gradients are natural-layout FP32 tensors.
    ``key_passes`` overrides the key-range-pass policy of the main stage
    (``None`` = the registered policy; see :func:`plan_key_passes`);
    ``num_segments`` is the packed segment count of the key row the policy
    plans with (the varlen entry passes ``len(cu_seqlens_k) - 1``; more than
    one segment plans one pass).  ``dkv_acc`` (FP32 ``[S_dst, >= 576]``)
    receives the dK/dV gradients in place instead -- latent columns ``0:512``,
    rope ``512:576`` of row ``dkv_dst_map[s]`` (optional int32 ``[S]``, identity
    by default, repeated rows sum); the caller owns zeroing and the result is
    ``(dq_latent, dq_rope, None, None)`` (see :func:`prepare_dsa_train`).  A
    strided ``dout`` is copied once (the delta stage reads it densely); a call
    without query rows returns empty ``dq`` and zero ``dkv`` gradients without
    launching.
    """
    if dkv_acc is not None and dkv_fp32:
        raise ValueError(
            "dkv_acc accumulates the dK/dV gradients in place; dkv_fp32 outputs are not produced with it"
        )
    if dkv_dst_map is not None and dkv_acc is None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
    if not dout.is_contiguous():
        dout = dout.contiguous()
    if _no_query_rows(q_latent):
        return _empty_backward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            out,
            o_lo,
            lse,
            dout,
            topk_length,
            dkv_fp32,
            dkv_acc,
            dkv_dst_map,
        )
    scale = default_softmax_scale() if softmax_scale is None else float(softmax_scale)
    key = backward_binding_key(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        scale,
        dkv_fp32,
        key_passes,
        num_segments,
        dkv_acc,
        dkv_dst_map,
    )
    binding = _binding(
        key,
        lambda: _plan(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            dout=dout,
            softmax_scale=scale,
            outputs=dict(out=out, lse=lse, o_lo=o_lo),
            dkv_fp32=dkv_fp32,
            backward=True,
            key_passes=key_passes,
            dkv_acc=dkv_acc,
            dkv_dst_map=dkv_dst_map,
            num_segments=num_segments,
        ),
    )
    return binding.backward(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        dkv_acc,
        dkv_dst_map,
    )


class DSASparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper: saves ``out``, ``o_lo``, ``lse`` and the inputs for the backward.

    ``dkv_acc`` / ``dkv_dst_map`` (see :func:`backward`): the backward accumulates the dK/dV gradients into the
    caller's packed FP32 buffer and returns no gradient for ``kv_latent`` / ``k_rope``.
    """

    @staticmethod
    def forward(
        ctx,
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        softmax_scale,
        key_passes=None,
        dkv_acc=None,
        dkv_dst_map=None,
        num_segments=1,
        derive_topk_length=False,
    ):
        out, lse, o_lo = forward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            derive_topk_length=derive_topk_length,
        )
        ctx.set_materialize_grads(
            False
        )  # no zero-filled grad for an unused lse; dout is None when out is unused
        ctx.softmax_scale = softmax_scale
        ctx.key_passes = key_passes
        ctx.num_segments = int(num_segments)
        ctx.has_topk_length = topk_length is not None
        # The packed accumulator is mutated in place by the backward (+=), so it is kept on ctx rather than saved: a
        # saved tensor's version check would reject that mutation when the graph is retained for another backward.
        ctx.dkv_acc = dkv_acc
        ctx.dkv_dst_map = dkv_dst_map
        saved = [q_latent, q_rope, kv_latent, k_rope, indices, out, lse, o_lo]
        saved.append(topk_length if topk_length is not None else indices.new_empty(0))
        ctx.save_for_backward(*saved)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        if dlse is not None:
            raise NotImplementedError(
                "gradients through lse are not supported; only out is differentiable"
            )
        if dout is None:  # out unused downstream
            return (None,) * 12
        q_latent, q_rope, kv_latent, k_rope, indices, out, lse, o_lo, topk_length = (
            ctx.saved_tensors
        )
        dq_latent, dq_rope, dkv_latent, dk_rope = backward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            out,
            o_lo,
            lse,
            dout,
            topk_length=topk_length if ctx.has_topk_length else None,
            softmax_scale=ctx.softmax_scale,
            key_passes=ctx.key_passes,
            dkv_acc=ctx.dkv_acc,
            dkv_dst_map=ctx.dkv_dst_map,
            num_segments=ctx.num_segments,
        )
        # dkv_latent / dk_rope are None when the gradients went into dkv_acc; no gradient for the eight other inputs
        return (dq_latent, dq_rope, dkv_latent, dk_rope) + (None,) * 8


def dsa_sparse_attention(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
):
    """Autograd entry of the flat form (see :mod:`flashinfer.dsa_sparse_attention`).

    Strided query slices and packed key rows are consumed in place (see
    :func:`_check_head_tensor` / :func:`_check_key_tensor`).  ``key_passes``
    overrides the backward's key-range-pass policy (``None`` = the registered
    policy, 1 = single pass; see :func:`plan_key_passes`).  ``dkv_acc`` (caller
    FP32 ``[S_dst, >= 576]`` accumulated in place: latent columns ``0:512``,
    rope ``512:576``) and ``dkv_dst_map`` (int32 ``[S]`` destination row per
    key row, default identity) are handed to the backward; with ``dkv_acc``
    the ``kv_latent`` / ``k_rope`` gradients are ``None``.  ``num_segments``
    is the packed segment count of the key row (``len(cu_seqlens_k) - 1`` for
    a packed batch; :func:`dsa_sparse_attention_varlen` passes it): the
    backward's whole-row key-range passes apply to one-segment rows only.
    Without ``topk_length`` the per-row lengths (last valid slot + 1, what
    :func:`derive_topk_length` computes) are derived once per step by the
    forward kernel itself while it runs -- no separate launch -- into a fresh
    ``[T]`` tensor that is saved for the backward, so both kernels skip the
    trailing invalid key blocks of short rows; the results are those of the
    full row.
    """
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    derive = False
    if topk_length is None and q_latent.shape[0] > 0:
        topk_length = torch.empty(
            (int(q_latent.shape[0]),), dtype=torch.int32, device=q_latent.device
        )
        derive = True
    out, lse = DSASparseAttentionFunction.apply(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length,
        float(softmax_scale),
        key_passes,
        dkv_acc,
        dkv_dst_map,
        int(num_segments),
        derive,
    )
    return (out, lse) if return_lse else out


def dsa_sparse_attention_varlen(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    causal: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
):
    """Packed multi-document form: per-document key indices are offset by
    ``cu_seqlens_k`` on device (this glue counts in the step time), then the flat
    kernels run over the packed rows.  Query and key segment lengths may differ
    (the query segment is the tail of its key prefix); with ``causal=True`` the
    offsetting also drops selected keys after the query's own position
    ``(seqlen_k - seqlen_q) + local_q`` (:func:`offset_gather_kv_indices`); the
    default ``causal=False`` keeps the index rows as is (the behaviour of the
    first release).  Without an explicit ``topk_length`` the per-row lengths
    come out of the same glue pass (no separate derivation).
    ``max_seqlen_q/k`` are accepted for signature parity and not used on the
    host.  ``dkv_acc`` / ``dkv_dst_map`` as in :func:`dsa_sparse_attention`.
    The segment count ``len(cu_seqlens_k) - 1`` (host metadata, no device
    sync) is passed on as ``num_segments``: with more than one segment the
    backward plans the single-pass stage unless ``key_passes`` overrides."""
    del max_seqlen_q, max_seqlen_k
    if topk_length is None:
        # one glue pass: global rows and the per-row lengths from the same validity mask
        indices, topk_length = offset_gather_kv_indices(
            gather_kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
            return_topk_length=True,
        )
    else:
        indices = offset_gather_kv_indices(
            gather_kv_indices, cu_seqlens_q, cu_seqlens_k, causal=causal
        )
    return dsa_sparse_attention(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=int(cu_seqlens_k.numel()) - 1,
    )
