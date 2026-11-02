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
# * Strided inputs are consumed in place: the query operands and ``dout``
#   reach the kernels through TMA descriptors encoded from the tensor's own
#   head and token strides (the delta stage takes the same strides as
#   scalars), the key operands by row stride (see ``_check_head_tensor`` /
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
# * The native zero-RoPE (NoPE) program (``cake_jit.NOPE_PROGRAM``) serves the
#   same contract without rope operands: ``forward_nope`` / ``backward_nope``,
#   ``prepare_dsa_nope_train`` and ``DSASparseAttentionNopeFunction`` take
#   ``q [T, 64, 512]`` and the shared latent ``kv [S, 512]`` (K = V), default
#   softmax scale 1/16, and return one combined natural-layout FP32 ``dkv = dK
#   + dV [S, 512]`` -- fresh (a new buffer per call) or accumulated (``+=``)
#   into the caller's ``dkv_acc``.  The RoPE entry points below are untouched;
#   the NoPE entries share every rope-independent helper (key-range passes,
#   direct accumulation, workspace carving, index offsetting, the plan cache).

from __future__ import annotations

import functools
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Union, overload

import torch
import tvm_ffi

from . import cake_jit, cake_launch
from .cake_jit import FORWARD_STAGES

NUM_HEADS = 64
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE
# The NoPE program scores ``q . kv`` over the 512 shared latent dimensions with
# an explicit scale; the model this program serves fixes it at 1/16 (exact in
# every floating-point format), not ``1 / sqrt(512)``.
NOPE_SOFTMAX_SCALE = 1.0 / 16.0
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
# The NoPE program carries no rope part: one FP32 dQ partial row is 64 x 512 x 4 B
# (131,072 B); its record states the value (``key_pass_policy.dq_partial_bytes_per_token``).
NOPE_DQ_PARTIAL_BYTES_PER_TOKEN = NUM_HEADS * D_LATENT * 4
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
    # FP32 dQ partial bytes per token of the pass scratch (147,456 with the rope part; a
    # record states another value, the NoPE program 131,072)
    dq_partial_bytes_per_token: int = DQ_PARTIAL_BYTES_PER_TOKEN
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
        if "dq_partial_bytes_per_token" in raw:
            values["dq_partial_bytes_per_token"] = int(
                raw["dq_partial_bytes_per_token"]
            )
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
        per_token = self.dq_partial_bytes_per_token + 4 * int(topk) + 4
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


def _program_stages(names: tuple[str, ...], program: str) -> tuple[str, ...]:
    """The RoPE program's stage ``names`` spelled for ``program`` (``nope_`` prefix on the NoPE program)."""
    return tuple(cake_jit.stage_name(name, program) for name in names)


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
    program: str = cake_jit.PROGRAM,
) -> int:
    """Number of key-range passes of one backward binding.

    ``key_passes`` overrides the record's policy (1 = the single-pass stage);
    a program without the pass stages serves one pass only, and a record
    without a ``key_pass_policy`` never takes the pass path by default.
    ``num_segments`` is the packed segment count of the key row (see the
    comment above ``KEY_PASS_STAGES``): the policy plans one pass for more
    than one segment.  ``arch`` selects the record's per-target rule (``None``:
    the R1 rule).  ``program`` names the pass stages (``KEY_PASS_STAGES`` of the
    RoPE program, their ``nope_`` forms of the NoPE program).
    """
    pass_stages = _program_stages(KEY_PASS_STAGES, program)
    multi_pass = all(stage in stages for stage in pass_stages)
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
                f"({pass_stages}); key_passes > 1 is unavailable"
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
# S >= min_keys_per_query * T) whenever the caller passes ``dkv_acc``; the
# ``dkv_direct`` keyword of the entry points forces (``True``) or disables
# (``False``) it (``None``, the default, = the rule).  Without the natural
# stages every ``dkv_acc`` call takes the cast's accumulate path.  Both paths
# add the same FP32 contributions into the same rows (another summation order).
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


def record_direct_stages(
    record: dict[str, Any], stages: tuple[str, ...], program: str = cake_jit.PROGRAM
) -> bool:
    """True when the record registers the natural-layout main stage (with its pass form next to the pass stages).

    ``program`` names the stages (``bwd_main_natural`` ... of the RoPE program, their ``nope_`` forms of the NoPE
    program)."""
    main, natural, pass_natural = _program_stages(
        ("bwd_main", "bwd_main_natural", "bwd_main_pass_natural"), program
    )
    pass_stages = _program_stages(KEY_PASS_STAGES, program)
    direct = natural in stages
    if direct:
        if main not in stages:
            raise ValueError(
                f"registry record: {natural} needs the {main} stage next to it"
            )
        if (pass_natural in stages) != all(s in stages for s in pass_stages):
            raise ValueError(
                f"registry record: {pass_natural} must accompany the key-range-pass stages"
            )
        if DkvDirectPolicy.from_record(record) is None:
            raise ValueError(
                "registry record: the natural-layout stages need a dkv_direct policy"
            )
    elif pass_natural in stages:
        raise ValueError(f"registry record: {pass_natural} without {natural}")
    return direct


def plan_dkv_direct(
    record: dict[str, Any],
    stages: tuple[str, ...],
    num_queries: int,
    num_kv: int,
    *,
    accumulate: bool,
    dkv_direct: Optional[bool] = None,
    program: str = cake_jit.PROGRAM,
) -> bool:
    """Does this backward binding add ``dkv_acc`` directly from the main stage?  Only with ``dkv_acc``
    (``accumulate``) and a program registering the natural stages; then ``dkv_direct`` decides: ``None`` (the
    default) applies the record's size rule, ``True`` takes the direct path, ``False`` the cast's accumulate path.
    Any other value raises ``ValueError``.  ``program`` names the stages (see :func:`record_direct_stages`)."""
    if dkv_direct is not None and not isinstance(dkv_direct, bool):
        raise ValueError(f"dkv_direct must be None, True or False, got {dkv_direct!r}")
    if not accumulate or not record_direct_stages(record, stages, program):
        return False
    if dkv_direct is None:
        return DkvDirectPolicy.from_record(record).wants(num_queries, num_kv)
    return dkv_direct


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


# The NoPE main stages run a persistent token loop: CTA ``b`` takes tokens ``b, b + grid, ...`` of the launch's
# ``num_queries`` tokens (``token = token_base + token_step * (blockIdx.x + k * gridDim.x)``), so every grid
# <= ``num_queries`` is legal.  One persistent CTA per SM overlaps each token's prologue / epilogue with its
# predecessor's tail and measured 0.5-19 % faster than a CTA per token while the launch's FP32 dkv window
# (``num_kv`` x 2048 B) is within 64 MiB; beyond it the CTAs' token drift widens the set of keys live in L2
# (3-9 % slower at 268-553 MB windows), so those launches keep a CTA per token.
NOPE_PERSISTENT_DKV_BYTES = 64 << 20
NOPE_DKV_BYTES_PER_KEY = 2048


def nope_bwd_launch_grid(num_queries: int, num_kv_in_pass: int, sm_count: int) -> int:
    """Grid (x) of a NoPE main-stage launch over ``num_queries`` tokens whose key pass covers ``num_kv_in_pass``
    keys -- the kernel's own rule, mirrored: one persistent CTA per SM (``min(num_queries, sm_count)``, at least one)
    while the pass's FP32 dkv window (``num_kv_in_pass`` x 2048 B) is within :data:`NOPE_PERSISTENT_DKV_BYTES`, else
    a CTA per token (``num_queries``).  The host never launches zero tokens (the empty paths return before planning).
    """
    if int(num_kv_in_pass) * NOPE_DKV_BYTES_PER_KEY <= NOPE_PERSISTENT_DKV_BYTES:
        return max(1, min(int(num_queries), int(sm_count)))
    return int(num_queries)


_SM_COUNTS: dict[int, int] = {}


def _sm_count(device_index: int) -> int:
    """Streaming multiprocessors of CUDA device ``device_index``, read once per device."""
    count = _SM_COUNTS.get(device_index)
    if count is None:
        count = int(
            torch.cuda.get_device_properties(device_index).multi_processor_count
        )
        _SM_COUNTS[device_index] = count
    return count


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return _arch_of(_device_index(device))


def default_softmax_scale() -> float:
    return D_QK**-0.5


def record_for(
    device: Optional[torch.device] = None, program: str = cake_jit.PROGRAM
) -> tuple[str, dict[str, Any]]:
    """``(program name, registry record)`` of ``program`` (the RoPE program by default) serving ``device``."""
    arch = arch_for(device)
    if arch is None:
        raise NotImplementedError(
            "DSA sparse-attention training needs a compute capability 10.0 / 10.3 / 10.7 device"
        )
    record = cake_jit.record(program)
    if arch not in record["arches"]:
        raise NotImplementedError(
            f"the generated DSA training program {program!r} is registered for {record['arches']}, not {arch}"
        )
    return program, record


def generated_program_available(
    device: Optional[torch.device] = None,
    *,
    backward: bool = False,
    program: str = cake_jit.PROGRAM,
) -> bool:
    """Whether ``device`` is served by ``program`` (with the backward stages when ``backward``): its record is
    registered for the device's architecture and carries the forward stage (and the delta, main and cast stages
    plus a key-range-pass policy for the backward)."""
    try:
        _, record = record_for(device, program)
    except NotImplementedError:
        return False
    stages = cake_jit.registered_stages(program)
    needed = _program_stages(
        ("fwd", "bwd_delta", "bwd_main", "bwd_cast") if backward else ("fwd",), program
    )
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


def _dout_admissible(dout: torch.Tensor) -> bool:
    """Whether the backward kernels can read ``dout`` in place (:func:`_check_head_tensor` as a predicate)."""
    try:
        _check_head_tensor(dout, "dout", D_LATENT)
    except ValueError:
        return False
    return True


def _dout_stride_constants(dout: Optional[torch.Tensor]) -> dict[str, int]:
    """The delta stage's ``dout`` operands: its token / head strides in elements and the element offset of the
    pointer the launcher passes (the validated view itself, 16-byte aligned: offset 0).  Both strides are multiples
    of 8 (validated) and must fit the kernel's int32 arguments (a head-major view's head stride is ``T * 512``)."""
    if dout is None:
        return {}
    return dict(
        dout_token_stride=_check_int32("the dout token stride", dout.stride(0)),
        dout_head_stride=_check_int32("the dout head stride", dout.stride(1)),
        dout_offset=0,
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
    accept it.  ``dout`` follows the layout rules of ``q_latent``
    (:func:`_check_head_tensor`: unit channel stride, token and head strides
    positive multiples of 8 elements, head stride >= 512, 16-byte base): a
    dense array or a strided view such as the head-major ``[512, T * 512, 1]``
    layout is consumed in place (the main stage reads it through a
    stride-carrying tensor map, the delta stage through the strides it
    receives).
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
    return num_queries, num_kv, topk


def validate_dsa_nope_inputs(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    *,
    dout: Optional[torch.Tensor] = None,
    allow_empty_queries: bool = False,
) -> tuple[int, int, int]:
    """Shape / dtype / layout validation of the NoPE entry points.  Returns ``(T, S, K_slots)``.

    ``q_latent``: BF16 ``[T, 64, 512]`` (:func:`_check_head_tensor`: strided token / head views are consumed in
    place); ``kv_latent``: BF16 ``[S, 512]`` (:func:`_check_key_tensor`: contiguous rows, any admissible row
    stride); ``indices``: int32 ``[T, K_slots]`` global key rows with contiguous rows (any row stride; ``-1`` or
    ``>= S`` invalid), any ``K_slots >= 1`` -- no width alignment is required (rows whose stride and start are
    multiples of eight elements take the kernels' vector index path, the others the scalar path);
    ``topk_length``: contiguous int32 ``[T]`` or ``None`` (the kernel-level entries read the rows as given; without
    it the public autograd entries compact rows with ``-1`` holes through the index-preparation stage first at or
    above :data:`NOPE_CONVERT_SLOT_LIMIT` slots and read them as given below it; with it the varlen entry runs the
    stage at every size and the flat entry reads rows and lengths as given, see :func:`nope_compact_indices`);
    ``dout``: BF16 ``[T, 64, 512]`` with ``T`` rows, under
    the same layout rules as ``q_latent`` (:func:`_check_head_tensor`: unit channel stride, token and head strides
    positive multiples of 8 elements, head stride >= 512, 16-byte base) -- a dense array or a strided view such as
    the head-major ``[512, T * 512, 1]`` layout is consumed in place (the main stage reads it through a
    stride-carrying tensor map, the delta stage through the strides it receives).  ``S == 0`` is rejected;
    ``T == 0`` only without ``allow_empty_queries`` (the eager entries return empty results for it).  Device
    placement is checked by the callers, so this runs on host tensors.
    """
    _check_head_tensor(q_latent, "q_latent", D_LATENT)
    _check_key_tensor(kv_latent, "kv_latent", D_LATENT)
    num_queries = int(q_latent.shape[0])
    num_kv = int(kv_latent.shape[0])
    if num_kv == 0:
        raise ValueError("kv_latent must hold at least one key row (S > 0)")
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
        raise ValueError("indices must be an int32 [T, K_slots] tensor")
    topk = int(indices.shape[1])
    if topk <= 0:
        raise ValueError("indices must have at least one slot per row (K_slots >= 1)")
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
    return num_queries, num_kv, topk


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}"
        )


def _check_dkv_acc(
    dkv_acc: torch.Tensor,
    dkv_dst_map: Optional[torch.Tensor],
    num_kv: int,
    *,
    check_dst_map: bool = False,
    min_cols: int = D_QK,
) -> tuple[torch.Tensor, int]:
    """Validate the caller's packed FP32 dK/dV accumulator and its optional destination-row map.

    ``dkv_acc``: FP32, 2-D ``[S_dst, >= min_cols]`` (``min_cols`` 576 for the RoPE program: the latent gradient is
    added into columns ``0:512``, the rope gradient into ``512:576``; 512 for the NoPE program: the combined dK + dV
    into columns ``0:512``; further columns are never touched), ``stride(1) == 1``, row stride >= ``min_cols``
    elements and a multiple of 4 (16-byte vectors), 16-byte-aligned base; without a map at least ``num_kv`` rows.
    ``dkv_dst_map``: ``None`` (identity) or a contiguous int32 ``[num_kv]`` tensor on the device of ``dkv_acc`` --
    the destination row of every source key row, values in ``[0, S_dst)``, duplicates allowed.  The kernel does not
    range-check the map (a value outside ``[0, S_dst)`` would add into memory outside ``dkv_acc``); the caller owns
    that invariant.  ``check_dst_map=True`` (default ``False``) validates the values here (a device synchronization)
    and raises ``ValueError`` on a violation.  Returns ``(operand, row_stride)``: the raw-pointer operand of the
    accumulating stage (``dkv_acc`` itself when contiguous, else a flat stride-1 alias from its first element over
    ``(S_dst - 1) * row_stride + cols`` elements -- the FFI boundary takes contiguous tensors only, and the kernel
    addresses ``row * row_stride + col``) and the row stride.  Device placement is checked by the callers, so this
    runs on host tensors.
    """
    if not isinstance(dkv_acc, torch.Tensor) or dkv_acc.dtype != torch.float32:
        raise ValueError("dkv_acc must be a float32 tensor")
    columns = (
        f"latent columns 0:{D_LATENT}, rope columns {D_LATENT}:{D_QK}"
        if min_cols == D_QK
        else f"combined dK + dV columns 0:{min_cols}"
    )
    if dkv_acc.ndim != 2 or int(dkv_acc.shape[1]) < min_cols:
        raise ValueError(
            f"dkv_acc must be a 2-D [S_dst, >= {min_cols}] tensor ({columns}), got shape {tuple(dkv_acc.shape)}"
        )
    if dkv_acc.stride(1) != 1:
        raise ValueError(
            f"dkv_acc rows must be contiguous (stride(1) == 1), got strides {tuple(dkv_acc.stride())}"
        )
    row_stride = int(dkv_acc.stride(0))
    if row_stride < min_cols or row_stride % 4:
        raise ValueError(
            f"dkv_acc row stride must be >= {min_cols} elements and a multiple of 4 (16-byte vectors), got {row_stride}"
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
    elif check_dst_map:
        num_rows = int(dkv_acc.shape[0])
        if bool(((dkv_dst_map < 0) | (dkv_dst_map >= num_rows)).any().item()):
            raise ValueError(
                f"dkv_dst_map values must lie in [0, {num_rows}) (the rows of dkv_acc); "
                f"got min {int(dkv_dst_map.min())}, max {int(dkv_dst_map.max())}"
            )
    return _dkv_acc_operand(dkv_acc), row_stride


def _dkv_acc_operand(dkv_acc: torch.Tensor) -> torch.Tensor:
    """The raw-pointer operand of a validated accumulator (:func:`_check_dkv_acc`): the tensor itself when
    contiguous, else the flat stride-1 alias over the rows it spans."""
    if dkv_acc.is_contiguous():
        return dkv_acc
    row_stride = int(dkv_acc.stride(0))
    span = (int(dkv_acc.shape[0]) - 1) * row_stride + int(dkv_acc.shape[1])
    return dkv_acc.as_strided((span,), (1,), dkv_acc.storage_offset())


# The autograd entries take ``dkv_acc`` as a tensor or as a zero-argument callable that produces it at backward
# time (``DkvAccSource``): under activation checkpointing a tensor passed at forward time exists for every layer
# at once, a deferred accumulator only while its backward runs.
DkvAccSource = Union[
    torch.Tensor,
    Callable[[], Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]],
]


def _resolve_dkv_acc(
    dkv_acc: Optional[DkvAccSource], dkv_dst_map: Optional[torch.Tensor]
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(dkv_acc, dkv_dst_map)`` as tensors: a tensor ``dkv_acc`` passes through; a callable is called now (at
    backward time) and returns the accumulator or an ``(accumulator, dkv_dst_map)`` pair -- a pair's map must not be
    given twice.  Validation (:func:`_check_dkv_acc`) is the caller's."""
    if dkv_acc is None or isinstance(dkv_acc, torch.Tensor):
        return dkv_acc, dkv_dst_map
    if not callable(dkv_acc):
        raise TypeError(
            "dkv_acc must be a float32 tensor or a zero-argument callable returning it"
        )
    produced = dkv_acc()
    if isinstance(produced, tuple):
        if len(produced) != 2:
            raise ValueError(
                "a dkv_acc callable returns the accumulator or an (accumulator, dkv_dst_map) pair"
            )
        produced, companion = produced
        if companion is not None:
            if dkv_dst_map is not None:
                raise ValueError(
                    "dkv_dst_map was given both as an argument and by the dkv_acc callable"
                )
            dkv_dst_map = companion
    if not isinstance(produced, torch.Tensor):
        raise ValueError("the dkv_acc callable must return a float32 tensor")
    return produced, dkv_dst_map


def _saved_optional(t: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """``t`` for ``ctx.save_for_backward``, or a zero-size placeholder on ``like``'s device when absent: the saved
    list keeps one fixed length and order (non-reentrant checkpointing matches recomputed saved tensors by
    position), and the presence flags travel separately."""
    return t if t is not None else like.new_empty(0)


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
    return _layout_of(sizes)


def _layout_of(sizes: list[tuple[str, int]]) -> dict:
    """``name -> (offset, size)`` of the regions ``sizes`` in order, each at a ``WORKSPACE_ALIGN`` offset, plus ``"total"``."""
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def workspace_layout_nope(
    num_queries: int,
    num_kv: int,
    topk: int,
    *,
    backward: bool = True,
    key_passes: int = 1,
    dkv_direct: bool = False,
    token_chunk: Optional[int] = None,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region of a NoPE binding plus ``"total"``.

    As :func:`workspace_layout` without the rope accumulator: the ``topk_length`` vector, and for the backward
    ``delta`` and -- unless the natural-layout main stage adds straight into the natural FP32 ``dkv`` buffer
    (``dkv_direct``: the caller's ``dkv_acc`` or the fresh output) -- the kernels' permuted FP32 accumulator
    ``dkv_latent_acc`` (``S * 2048`` B), plus the pass scratch of a multi-pass backward (``dq_partial`` at 131,072 B
    per token, ``key_scratch``, ``pass_counts`` for the tokens of one launch chunk).
    """
    sizes = [("topk_length", num_queries * 4)]
    if backward:
        sizes.append(("delta", num_queries * NUM_HEADS * 4))
        if not dkv_direct:
            sizes.append(("dkv_latent_acc", num_kv * D_LATENT * 4))
        if int(key_passes) > 1:
            rows = (
                int(num_queries)
                if token_chunk is None
                else min(int(num_queries), int(token_chunk))
            )
            sizes += [
                ("dq_partial", rows * NOPE_DQ_PARTIAL_BYTES_PER_TOKEN),
                ("key_scratch", rows * int(topk) * 4),
                ("pass_counts", rows * 4),
            ]
    return _layout_of(sizes)


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
    dkv_direct: Optional[bool] = None,
) -> int:
    """Workspace bytes :func:`prepare_dsa_train` needs for ``(T, S, topk)`` on ``device``.

    ``key_passes`` / ``num_segments`` as in :func:`prepare_dsa_train`
    (``None`` = the record's policy; the packed segment count of the key row);
    ``dkv_acc`` = the binding will accumulate into a caller-provided ``dkv_acc``
    (a direct binding needs no FP32 accumulators; see :func:`plan_dkv_direct`),
    with ``dkv_direct`` as in :func:`prepare_dsa_train` (``None``, the default,
    = the record's size rule; ``True`` / ``False`` force / disable the direct
    path).
    """
    _, record = record_for(device)
    arch = arch_for(device)
    stages = cake_jit.registered_stages()
    direct = bool(backward) and plan_dkv_direct(
        record,
        stages,
        num_queries,
        num_kv,
        accumulate=bool(dkv_acc),
        dkv_direct=dkv_direct,
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


def _plan_nope_direct(
    record: dict[str, Any],
    stages: tuple[str, ...],
    num_queries: int,
    num_kv: int,
    dkv_direct: Optional[bool],
) -> bool:
    """Does a NoPE backward add its dK + dV straight into the natural FP32 buffer from the main stage?

    Every NoPE backward accumulates into a natural-layout FP32 ``[S_dst, >= 512]`` buffer -- the caller's
    ``dkv_acc`` or the fresh ``dkv`` the backend allocates and zero-fills -- so the record's rule (and the
    ``dkv_direct`` keyword) applies to both forms: :func:`plan_dkv_direct` with ``accumulate=True``.  The NoPE
    record's rule (``min_keys_per_query`` 1) takes the natural stage whenever the row has at least one key per
    query (every packed causal problem; measured faster than the permuted accumulator plus cast on every row);
    a problem with more query rows than keys, or ``dkv_direct=False``, uses the permuted accumulator and the cast.
    Without the natural-layout stages the kernels' permuted accumulator and the cast serve both forms.
    """
    return plan_dkv_direct(
        record,
        stages,
        num_queries,
        num_kv,
        accumulate=True,
        dkv_direct=dkv_direct,
        program=cake_jit.NOPE_PROGRAM,
    )


def dsa_nope_train_workspace_size(
    num_queries: int,
    num_kv: int,
    topk: int,
    device: Optional[torch.device] = None,
    *,
    backward: bool = True,
    key_passes: Optional[int] = None,
    num_segments: int = 1,
    dkv_direct: Optional[bool] = None,
) -> int:
    """Workspace bytes :func:`prepare_dsa_nope_train` needs for ``(T, S, K_slots)`` on ``device``.

    ``key_passes`` / ``num_segments`` as in :func:`prepare_dsa_nope_train` (``None`` = the NoPE record's policy;
    the packed segment count of the key row); ``dkv_direct`` as there (``None``, the default, = the record's size
    rule; ``True`` / ``False`` force / disable the direct path).  Unlike :func:`dsa_train_workspace_size` the
    layout does not depend on whether the caller passes ``dkv_acc``: a fresh ``dkv`` is accumulated under the same
    rule (see :func:`_plan_nope_direct`).
    """
    program = cake_jit.NOPE_PROGRAM
    _, record = record_for(device, program)
    arch = arch_for(device)
    stages = cake_jit.registered_stages(program)
    direct = bool(backward) and _plan_nope_direct(
        record, stages, num_queries, num_kv, dkv_direct
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
            program=program,
        )
        if backward
        else 1
    )
    return int(
        workspace_layout_nope(
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
    compact: bool = False,
    topk_length: Optional[torch.Tensor] = None,
):
    """Turn per-document key indices into global key rows (the torch twin of the index-preparation stage).

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

    This is the reference formulation the tests hold the index-preparation
    stage to, and the fallback of the rope varlen entry where the stage is not
    available (host tensors, a device the NoPE record does not serve); the
    varlen entries themselves launch the stage.  It is launch-bound (a
    ``[T, topk]`` pass is 5-10 us of GPU time against ~10 us of dispatch), so
    it runs with as few launches as the rule allows: a handful of ``[T]``
    int32 ops for the per-row document data, then two compares into one bool
    mask (``0 <= idx`` and ``idx <= bound`` -- the bound is the row's own key
    position with ``causal``, itself ``< seqlen_k``, and ``seqlen_k - 1``
    otherwise) and one ``where`` over ``idx + key_base``.  With
    ``return_topk_length=True`` the per-row ``topk_length`` (last valid slot
    + 1 of the result row under the rule ``0 <= idx < num_kv``, 0 if none) is
    derived from the same mask with one ``where`` and one ``amax`` and the
    function returns ``(indices, topk_length)``.  Runs on device without a
    host synchronization, in int32 throughout.

    ``compact=True`` (the stage's ``compact = 1`` mode) is the stable compaction
    of the same rows: every row holds its valid globalized keys first, in
    ascending slot order, then ``-1``, and the returned ``topk_length`` is the
    valid count (a stable ``argsort`` of the validity mask over the default
    result).  Rows whose valid slots already come first are byte-identical to
    the default form, and the count equals their last valid slot + 1.  A
    caller ``topk_length`` (int32 ``[T]``; the stage's ``length_in``) is a
    position cut applied before compaction: slots at positions ``>=
    topk_length[t]`` are invalid regardless of content.

    Caller-owned invariants of the prefix sums (checked by
    :func:`validate_cu_seqlens`, which the NoPE varlen entries run by default;
    this function checks structure only): both int32 ``[num_docs + 1]``, start
    at 0, non-decreasing, ``cu_seqlens_q[-1] == T``, ``cu_seqlens_k[-1] == S``
    and ``seqlen_q <= seqlen_k`` per document.  A violation that passes here
    is not detected on device: a document with ``seqlen_k < seqlen_q`` gives
    negative own positions and all-invalid rows under ``causal``, a
    non-monotonic ``cu_seqlens_q`` an asynchronous device-side index error.
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
    if topk_length is not None:
        if topk_length.shape != (total_q,) or topk_length.dtype != torch.int32:
            raise ValueError("topk_length must be an int32 [T] tensor")
        valid &= _slot_positions(topk, device) <= topk_length[:, None]
    if compact:
        # valid slots first in ascending slot order (a stable sort of the validity mask), -1 after
        order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
        globalized = torch.where(valid, local + key_base[:, None], -1)
        result = (
            globalized.gather(-1, order)
            if out is None
            else torch.gather(globalized, -1, order, out=out)
        )
        if not return_topk_length:
            return result
        return result, valid.sum(dim=-1, dtype=torch.int32)
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


def validate_cu_seqlens(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_queries: int,
    num_kv: int,
    *,
    values: bool = True,
) -> None:
    """Check the packed-batch invariants of a varlen call; raises ``ValueError`` naming the argument.

    Structure (host side, always): both contiguous int32 1-D ``[num_docs + 1]`` tensors of one length (at least two
    entries) on one device.  Values (one blocking device-to-host copy; ``values=False`` skips them, and they are
    skipped during CUDA-graph capture, where a synchronization is not possible -- a capture replays the geometry of
    the eager call that was validated before it): both start at 0 and are non-decreasing, ``cu_seqlens_q[-1] == num_queries``
    (the packed query rows ``T``) and ``cu_seqlens_k[-1] == num_kv`` (the packed key rows ``S``; a shorter or longer
    key window is not accepted), and every document has ``seqlen_q <= seqlen_k`` (the query segment is the tail of
    its key prefix).
    """
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if (
            not isinstance(cu, torch.Tensor)
            or cu.ndim != 1
            or cu.dtype != torch.int32
            or cu.numel() < 2
        ):
            raise ValueError(f"{name} must be an int32 [num_docs + 1] tensor")
        if not cu.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must describe the same documents"
        )
    if cu_seqlens_q.device != cu_seqlens_k.device:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be on one device")
    if not values or (
        cu_seqlens_q.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        return
    q, k = torch.stack([cu_seqlens_q, cu_seqlens_k]).tolist()  # one copy to the host
    for name, cu in (("cu_seqlens_q", q), ("cu_seqlens_k", k)):
        if cu[0] != 0:
            raise ValueError(f"{name} must start at 0, got {cu[0]}")
        if any(b < a for a, b in zip(cu, cu[1:], strict=False)):
            raise ValueError(f"{name} must be non-decreasing")
    if q[-1] != int(num_queries):
        raise ValueError(
            f"cu_seqlens_q[-1] ({q[-1]}) must equal the packed query rows T ({int(num_queries)})"
        )
    if k[-1] != int(num_kv):
        raise ValueError(
            f"cu_seqlens_k[-1] ({k[-1]}) must equal the packed key rows S ({int(num_kv)}); "
            "a key window shorter or longer than the packed keys is not accepted"
        )
    for d, (q0, q1, k0, k1) in enumerate(zip(q, q[1:], k, k[1:], strict=False)):
        if q1 - q0 > k1 - k0:
            raise ValueError(
                f"document {d} has seqlen_q ({q1 - q0}) > seqlen_k ({k1 - k0}): the query segment "
                "must be the tail of its key prefix"
            )


_SLOT_POSITIONS: dict[tuple, torch.Tensor] = {}


def _slot_positions(topk: int, device: torch.device) -> torch.Tensor:
    """Cached ``[1 .. topk]`` int32 on ``device`` (constant per problem)."""
    key = (int(topk), str(device))
    pos = _SLOT_POSITIONS.get(key)
    if pos is None:
        pos = torch.arange(1, int(topk) + 1, device=device, dtype=torch.int32)
        _SLOT_POSITIONS[key] = pos
    return pos


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


def _ffi_tensor(tensor: torch.Tensor) -> Any:
    """The tvm-ffi view of ``tensor`` (zero-copy: the same pointer, shape, strides and dtype the binding reads when
    it converts a torch tensor itself).  Module attribute so tests can stand it in."""
    return tvm_ffi.from_dlpack(tensor.detach() if tensor.requires_grad else tensor)


def _ffi_values(values: dict[str, Any]) -> dict[str, Any]:
    """``values`` with every tensor converted to its tvm-ffi view now: a launch issued later over the result costs
    its binding call alone (no per-operand conversion), which is how the packed entry issues the forward right
    after the index-preparation launch."""
    return {
        name: _ffi_tensor(value) if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }


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


def _stage_entry(
    stage: str,
    arch: str,
    scalars: dict,
    program: str = cake_jit.PROGRAM,
    grid: Optional[tuple[int, int, int]] = None,
) -> _StageEntry:
    """Resolve ``stage`` of ``program`` for ``arch``: the generated launcher, its grid (``grid`` replaces the generated
    rule: the NoPE main stages' persistent grid, :func:`nope_bwd_launch_grid`) and the entry of the loaded module."""
    if grid is None:
        grid = cake_launch.GRID[stage](
            scalars["num_queries"], scalars["num_kv"], scalars["topk"]
        )
    record = cake_jit.record(program)
    cluster = record[stage]["launch"]["cluster"]
    if any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape "
            f"{tuple(cluster)} baked into the module"
        )
    module = cake_jit.load_cake_dsa_train_module(stage, arch, program)
    entry = getattr(module, record[stage]["ffi_entry"])
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
    # The program's forward stage(s) (``module_name`` is the program) and the FP32 dQ partial bytes per
    # token of its pass scratch; the NoPE plans set both.
    forward_stages: tuple[str, ...] = FORWARD_STAGES
    dq_partial_bytes_per_token: int = DQ_PARTIAL_BYTES_PER_TOKEN
    # NoPE: the backward returns a fresh natural FP32 ``dkv`` (no caller ``dkv_acc``).
    fresh_dkv: bool = False

    def launch_keys(self) -> tuple:
        return tuple(self.forward_stages) + self.backward_order

    @property
    def pass_rows(self) -> int:
        """Token rows of the pass scratch regions (one launch chunk; 0 without passes)."""
        if "dq_partial" not in self.layout:
            return 0
        return self.layout["dq_partial"][1] // self.dq_partial_bytes_per_token


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
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
) -> _Plan:
    """Validate one geometry and resolve its plan (no allocation, no launch).  ``dkv_direct`` / ``check_dst_map``
    as in :func:`prepare_dsa_train`."""
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
        _, dst_row_stride = _check_dkv_acc(
            dkv_acc, dkv_dst_map, num_kv, check_dst_map=check_dst_map
        )
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
    # size rule (or the caller's ``dkv_direct``) selects it; otherwise the cast's accumulate path
    direct = bool(backward) and plan_dkv_direct(
        record,
        stages,
        num_queries,
        num_kv,
        accumulate=accumulate,
        dkv_direct=dkv_direct,
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
        # bwd_delta: dout's token / head strides and the pointer's element offset (the strided delta kernel reads
        # a head-major or any other admissible view in place; with contiguous strides it is bitwise the dense form)
        **_dout_stride_constants(dout),
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
    T, S, layout = plan.num_queries, plan.num_kv, plan.layout
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
        t.update(_pass_scratch(plan, flat))
    return t


def _pass_scratch(plan: _Plan, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """The key-range-pass scratch of ``plan`` (one launch chunk of tokens) carved from the workspace ``flat``."""
    rows, layout = plan.pass_rows, plan.layout
    return dict(
        dq_partial=_carve(
            flat,
            layout,
            "dq_partial",
            torch.float32,
            (rows, plan.dq_partial_bytes_per_token // 4),
        ),
        key_scratch=_carve(flat, layout, "key_scratch", torch.int32, (rows, plan.topk)),
        pass_counts=_carve(flat, layout, "pass_counts", torch.int32, (rows,)),
    )


def _scratch_nope(plan: _Plan, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """The backward scratch regions of a NoPE ``plan`` carved from the workspace ``flat`` (no rope accumulator)."""
    T, S, layout = plan.num_queries, plan.num_kv, plan.layout
    t = dict(delta=_carve(flat, layout, "delta", torch.float32, (T, NUM_HEADS)))
    if not plan.dkv_direct:
        t["dkv_latent_acc"] = _carve(
            flat, layout, "dkv_latent_acc", torch.float32, (S, D_LATENT)
        )
    if plan.key_passes > 1:
        t.update(_pass_scratch(plan, flat))
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
    # The program's forward launch key(s) (``("fwd",)``; ``("nope_fwd",)`` on the NoPE runner).
    forward_stages: tuple[str, ...] = FORWARD_STAGES

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
        self._run(self.forward_stages)
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
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
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
    gradients.  ``dkv_direct`` overrides that rule: ``None`` (default) applies
    it, ``True`` takes the direct path whenever the program registers the
    natural stages, ``False`` always takes the cast's accumulate path.
    ``check_dst_map=True`` (default ``False``) validates the values of
    ``dkv_dst_map`` (every destination row in ``[0, S_dst)``) while the binding
    is prepared, at the cost of one device synchronization; the kernel does not
    range-check the map.

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
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
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
# call never shares storage with another call that is still in flight.  The
# cache keeps ``BINDING_CACHE_DEFAULT_CAPACITY`` entries unless
# ``BINDING_CACHE.configure(capacity=...)`` resizes it.
BINDING_CACHE_DEFAULT_CAPACITY = 64


def _binding_cache_capacity(capacity: Optional[int]) -> int:
    """``capacity`` as a positive integer; ``None`` is :data:`BINDING_CACHE_DEFAULT_CAPACITY`."""
    if capacity is None:
        return BINDING_CACHE_DEFAULT_CAPACITY
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError(
            f"binding cache capacity must be a positive integer, got {capacity!r}"
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
    dkv_direct: Optional[bool] = None,
) -> tuple:
    """Cache key of a backward plan: the forward key's inputs plus the saved forward
    outputs, ``dout``, the ``dkv_fp32`` option, the ``key_passes`` override, the
    segment count the pass policy saw and the geometry of the packed accumulator /
    destination map (``None`` when absent; their row stride and presence are baked
    into the accumulating launch) and, with ``dkv_acc``, the ``dkv_direct`` option
    (``None`` / ``True`` / ``False``; see :func:`plan_dkv_direct`)."""
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
        None if dkv_acc is None else dkv_direct,
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
        check_dst_map=False,
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
                (rows, plan.dq_partial_bytes_per_token // 4), torch.float32
            )
            t["key_scratch"] = fresh((rows, plan.topk), torch.int32)
            t["pass_counts"] = fresh((rows,), torch.int32)
        if plan.accumulate_dkv:
            if dkv_acc is None:
                raise RuntimeError(
                    "this binding accumulates the dK/dV gradients into dkv_acc and the call provides none"
                )
            # the same key implies the validated shape / stride / dtype; the flat alias is rebuilt per call (and
            # the map's values are checked per call when the caller asks for it)
            t["dkv_acc"], _row_stride = _check_dkv_acc(
                dkv_acc, dkv_dst_map, S, check_dst_map=check_dst_map
            )
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
    """Bounded, lock-protected LRU of :class:`_Binding` by call key.

    ``capacity`` (``None`` = :data:`BINDING_CACHE_DEFAULT_CAPACITY`, 64) bounds the entries; :meth:`configure`
    changes it on a live cache.
    """

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = _binding_cache_capacity(capacity)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[tuple, _Binding]" = OrderedDict()

    def configure(self, *, capacity: Optional[int] = None) -> None:
        """Set the capacity (``None`` = :data:`BINDING_CACHE_DEFAULT_CAPACITY`); entries beyond it are evicted,
        least recently used first."""
        with self._lock:
            self.capacity = _binding_cache_capacity(capacity)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

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


_BindingT = TypeVar("_BindingT", bound="_Binding")


@overload
def _binding(key: tuple, make_plan: Callable[[], _Plan]) -> _Binding: ...


@overload
def _binding(
    key: tuple, make_plan: Callable[[], _Plan], binding_cls: type[_BindingT]
) -> _BindingT: ...


def _binding(key, make_plan, binding_cls=_Binding):
    """The cached binding of ``key``, planned on a miss (a ``binding_cls`` instance: :class:`_Binding`, or
    :class:`_NopeBinding` for the NoPE entries; the overloads give the call site that class).

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
        binding = binding_cls(plan, topk_length)
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
    check_dst_map: bool = False,
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
        # nothing to accumulate; the buffer and the map are validated all the same
        _check_dkv_acc(dkv_acc, dkv_dst_map, num_kv, check_dst_map=check_dst_map)
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
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
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
    ``(dq_latent, dq_rope, None, None)`` (see :func:`prepare_dsa_train`).
    ``dkv_direct`` (default ``None`` = the record's size rule, ``True`` / ``False`` =
    force / disable the direct path of a ``dkv_acc`` binding) and
    ``check_dst_map`` (default ``False``; ``True`` validates the values of
    ``dkv_dst_map`` on every call, one device synchronization) as in
    :func:`prepare_dsa_train`.  An admissible strided ``dout`` (the head-major
    ``[512, T * 512, 1]`` view an output projection hands back, or any view
    under the layout rules of ``q_latent``) is read in place by the delta stage
    (token / head strides) and the main stage (stride-carrying tensor map); a
    view the TMA descriptor cannot address (base not 16-byte aligned, strides
    not multiples of 8 elements, channel stride != 1) is copied once.  A call
    without query rows returns empty ``dq`` and zero ``dkv`` gradients without
    launching.
    """
    if dkv_acc is not None and dkv_fp32:
        raise ValueError(
            "dkv_acc accumulates the dK/dV gradients in place; dkv_fp32 outputs are not produced with it"
        )
    if dkv_dst_map is not None and dkv_acc is None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
    if not _dout_admissible(dout):
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
            check_dst_map,
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
        dkv_direct,
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
            dkv_direct=dkv_direct,
            check_dst_map=check_dst_map,
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
        check_dst_map,
    )


class DSASparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper: saves ``out``, ``o_lo``, ``lse`` and the inputs for the backward.

    Every tensor the forward produces or consumes for the backward goes through ``ctx.save_for_backward`` (nine
    tensors in a fixed order, a zero-size placeholder for an absent ``topk_length``): under non-reentrant
    activation checkpointing they are released after the forward and recomputed, and nothing else of size is kept
    on ``ctx``.  ``dkv_acc`` / ``dkv_dst_map`` (see :func:`backward`): the backward accumulates the dK/dV gradients
    into the caller's packed FP32 buffer and returns no gradient for ``kv_latent`` / ``k_rope``; ``dkv_acc`` is a
    tensor (caller-owned: validated here and kept on ``ctx`` as before, free to be zeroed in place between the
    forward and the backward) or a zero-argument callable producing it (or an ``(accumulator, dkv_dst_map)`` pair)
    at backward time, so the accumulator need not exist at forward time (:data:`DkvAccSource`; a callable's result
    is validated in the backward).  ``dkv_direct`` (default ``None``: the record's size rule; ``True`` /
    ``False`` force / disable the direct path) and ``check_dst_map`` (default ``False``; ``True`` validates the
    map's values in the backward) are handed to :func:`backward`.
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
        dkv_direct=None,
        check_dst_map=False,
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
        ctx.dkv_direct = dkv_direct
        ctx.check_dst_map = check_dst_map
        if dkv_dst_map is not None and dkv_acc is None:
            raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
        # A tensor accumulator is validated now and kept on ``ctx`` as before (caller-owned, outside the saved list:
        # the checkpoint frame holds the Function's inputs anyway, and no version counter is involved).  A callable
        # (no tensor yet) is kept as the zero-size state it is and resolved in the backward; ``check_dst_map`` then
        # synchronizes there.
        acc_tensor = dkv_acc if isinstance(dkv_acc, torch.Tensor) else None
        if acc_tensor is not None:
            _check_dkv_acc(acc_tensor, dkv_dst_map, int(kv_latent.shape[0]))
        elif dkv_acc is not None and not callable(dkv_acc):
            raise TypeError(
                "dkv_acc must be a float32 tensor or a zero-argument callable returning it"
            )
        ctx.dkv_acc = acc_tensor
        ctx.dkv_dst_map = dkv_dst_map
        ctx.dkv_acc_source = None if acc_tensor is not None else dkv_acc
        ctx.save_for_backward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            out,
            lse,
            o_lo,
            _saved_optional(topk_length, indices),
        )
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        if dlse is not None:
            raise NotImplementedError(
                "gradients through lse are not supported; only out is differentiable"
            )
        if dout is None:  # out unused downstream
            return (None,) * 14
        (
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            out,
            lse,
            o_lo,
            topk_length,
        ) = ctx.saved_tensors
        dkv_acc, dkv_dst_map = _resolve_dkv_acc(
            ctx.dkv_acc if ctx.dkv_acc is not None else ctx.dkv_acc_source,
            ctx.dkv_dst_map,
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
            dkv_acc=dkv_acc,
            dkv_dst_map=dkv_dst_map,
            num_segments=ctx.num_segments,
            dkv_direct=ctx.dkv_direct,
            check_dst_map=ctx.check_dst_map,
        )
        # dkv_latent / dk_rope are None when the gradients went into dkv_acc; no gradient for the ten other inputs
        return (dq_latent, dq_rope, dkv_latent, dk_rope) + (None,) * 10


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
    dkv_acc: Optional[DkvAccSource] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
):
    """Autograd entry of the flat form (see :mod:`flashinfer.dsa_sparse_attention`).

    Strided query slices and packed key rows are consumed in place (see
    :func:`_check_head_tensor` / :func:`_check_key_tensor`).  ``key_passes``
    overrides the backward's key-range-pass policy (``None`` = the registered
    policy, 1 = single pass; see :func:`plan_key_passes`).  ``dkv_acc`` (caller
    FP32 ``[S_dst, >= 576]`` accumulated in place: latent columns ``0:512``,
    rope ``512:576``) and ``dkv_dst_map`` (int32 ``[S]`` destination row per
    key row, default identity) are handed to the backward; with ``dkv_acc``
    the ``kv_latent`` / ``k_rope`` gradients are ``None``.  ``dkv_acc`` may
    also be a zero-argument callable that the backward calls for the
    accumulator -- or for an ``(accumulator, dkv_dst_map)`` pair -- so that it
    need not exist at forward time (:data:`DkvAccSource`; under activation
    checkpointing a tensor passed at forward time is alive for every layer at
    once, a deferred one only during its backward).  Every tensor this call
    produces for the backward is saved through ``ctx.save_for_backward``; a
    tensor ``dkv_acc`` stays caller-owned on the node: a checkpointed forward
    retains nothing of this call.  ``dkv_direct``
    (default ``None``: the record's size rule; ``True`` / ``False`` force /
    disable the direct accumulation of a ``dkv_acc`` binding) and
    ``check_dst_map`` (default ``False``; ``True`` validates the values of
    ``dkv_dst_map`` in the backward, one device synchronization) are handed to
    the backward as well (see :func:`backward`).  ``num_segments``
    is the packed segment count of the key row (``len(cu_seqlens_k) - 1`` for
    a packed batch; :func:`dsa_sparse_attention_varlen` passes it): the
    backward's whole-row key-range passes apply to one-segment rows only.
    Without ``topk_length`` the per-row lengths (last valid slot + 1 under
    the rule ``0 <= idx < num_kv``, 0 if none) are derived once per step by the
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
        dkv_direct,
        check_dst_map,
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
    dkv_acc: Optional[DkvAccSource] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
):
    """Packed multi-document form: per-document key indices are turned into
    global key rows on device in one launch of the NoPE record's
    index-preparation stage (:func:`_rope_varlen_index_mode`; the torch twin
    ``offset_gather_kv_indices(..., compact=True)`` where the stage is not
    available: host tensors, a device the record does not serve, a
    non-contiguous prefix sum -- both routes agree bitwise), then the flat
    kernels run over the packed rows.  Each row is compacted: its valid keys
    come first in ascending slot order, then ``-1``, and the row's length is
    its valid count, so rows with ``-1`` holes cost their valid slots in the
    forward and the backward.  Query and key segment lengths may differ (the
    query segment is the tail of its key prefix); with ``causal=True`` the
    offsetting also drops selected keys after the query's own position
    ``(seqlen_k - seqlen_q) + local_q``; the default ``causal=False`` keeps the
    index rows as is (the behaviour of the first release).  A caller
    ``topk_length`` is a position cut applied before compaction (slots at
    positions ``>= topk_length[t]`` are invalid regardless of content); the
    kernels then read the compacted rows and their counts.  ``max_seqlen_q/k``
    are accepted for signature parity and not used on the host.  ``dkv_acc`` /
    ``dkv_dst_map`` / ``dkv_direct`` (default ``None``) / ``check_dst_map``
    (default ``False``) as in :func:`dsa_sparse_attention`.  The segment count
    ``len(cu_seqlens_k) - 1`` (host metadata, no device sync) is passed on as
    ``num_segments``: with more than one segment the backward plans the
    single-pass stage unless ``key_passes`` overrides.  The autograd function
    saves the compacted rows and counts for the backward."""
    del max_seqlen_q, max_seqlen_k
    num_queries = int(q_latent.shape[0])
    if (
        _rope_varlen_index_mode(
            gather_kv_indices, cu_seqlens_q, cu_seqlens_k, num_queries
        )
        == "index_prep"
    ):
        arch = _nope_arch(gather_kv_indices)
        topk = int(gather_kv_indices.shape[1])
        device = gather_kv_indices.device
        indices = _alloc((num_queries, topk), torch.int32, device)
        lengths = _alloc((num_queries,), torch.int32, device)
        _launch_index_prep(
            arch,
            gather_kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
            out=indices,
            topk_length=lengths,
            compact=True,
            length_in=topk_length,
        )
    else:
        indices, lengths = offset_gather_kv_indices(
            gather_kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
            return_topk_length=True,
            compact=True,
            topk_length=topk_length,
        )
    return dsa_sparse_attention(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=lengths,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=int(cu_seqlens_k.numel()) - 1,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


# ---------------------------------------------------------------------------
# NoPE program (``cake_jit.NOPE_PROGRAM``): plans, runner, eager entries, autograd
# ---------------------------------------------------------------------------
#
# The native zero-RoPE kernels score ``q . kv`` over the 512 shared latent
# dimensions (K = V = ``kv_latent``), take an explicit softmax scale (1/16 by
# default) and produce ONE combined gradient per key row: ``dkv = dK + dV``,
# natural layout, FP32, ``[S, 512]``.  Two output forms, both FP32 and both
# accumulated in FP32 across the kernels' atomics (no per-call BF16 rounding):
#
# * fresh (no ``dkv_acc``): the backend allocates the ``[S, 512]`` buffer and
#   returns it filled -- through the natural-layout main stage adding into the
#   zero-filled buffer when the record's direct rule (``dkv_direct``) selects
#   it, else through the kernels' permuted accumulator and the cast stage
#   writing the buffer densely (``out_f32``);
# * accumulate (``dkv_acc`` FP32 ``[S_dst, >= 512]``, optional ``dkv_dst_map``):
#   ``+=`` into the caller's rows (the direct main stage or the accumulating
#   cast, by the same rule); the call returns ``None`` for ``dkv``.  Chunked
#   backward passes of one long row accumulate into one such buffer with no
#   intermediate rounding.
#
# ``dout`` may arrive head-major (``[512, T * 512, 1]`` strides, the layout an
# output projection hands back) or in any other admissible strided layout: the
# main stage reads it through a stride-carrying tensor map and the delta stage
# through the token / head strides and element offset the launcher passes, so no
# host copy is made.
#
# Index rows with ``-1`` holes: the kernels work up to the last valid slot of a
# row (or its ``topk_length``), so the public autograd entries compact every
# row on device through the index-preparation stage before the forward (stable:
# valid keys first in ascending slot order, then ``-1``; the length becomes the
# valid count) -- the varlen entry in its document mode, the flat entry without
# caller lengths in the stage's flat mode.  The kernel-level entries and the
# prepared runner read rows as given; ``nope_compact_indices`` is their recipe.


def _main_accumulator_constants_nope(
    direct: bool, dst_row_stride: int, has_map: bool
) -> dict[str, int]:
    """The NoPE main stage's accumulator operands: the natural FP32 rows it adds into on a direct binding (their
    row stride and the map flag), else the private permuted ``[S, 512]`` accumulator (stride 512, no map)."""
    if direct:
        return dict(dkv_stride=int(dst_row_stride), dkv_has_map=int(bool(has_map)))
    return dict(dkv_stride=D_LATENT, dkv_has_map=0)


def _kv_latent_rows(kv_latent: torch.Tensor) -> torch.Tensor:
    """``kv_latent`` as the 2-D ``[S, 512]`` operand: the one-head form ``[S, 1, 512]`` of the shared latent is
    squeezed (a view; autograd routes the gradient back through it), anything else is passed through to
    validation."""
    if kv_latent.ndim == 3 and int(kv_latent.shape[1]) == 1:
        return kv_latent.squeeze(1)
    return kv_latent


def _plan_nope(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor],
    dout: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    outputs: dict[str, Optional[torch.Tensor]],
    backward: bool,
    key_passes: Optional[int],
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    derive_topk_length: bool = False,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
) -> _Plan:
    """Validate one NoPE geometry and resolve its plan (no allocation, no launch).  ``dkv_direct`` /
    ``check_dst_map`` as in :func:`prepare_dsa_nope_train`."""
    program = cake_jit.NOPE_PROGRAM
    if derive_topk_length and topk_length is None:
        raise ValueError(
            "derive_topk_length=True needs a topk_length [T] int32 tensor for the forward to fill"
        )
    num_queries, num_kv, topk = validate_dsa_nope_inputs(
        q_latent, kv_latent, indices, topk_length, dout=dout
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
        if outputs.get("dkv") is not None:
            raise ValueError(
                "dkv_acc accumulates in place: no dkv output tensor is written with it"
            )
        _, dst_row_stride = _check_dkv_acc(
            dkv_acc,
            dkv_dst_map,
            num_kv,
            check_dst_map=check_dst_map,
            min_cols=D_LATENT,
        )
    elif dkv_dst_map is not None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
    device = q_latent.device
    tensors = [q_latent, kv_latent, indices, topk_length, dout, dkv_acc, dkv_dst_map]
    tensors += list(outputs.values())
    if not all(t.is_cuda and t.device == device for t in tensors if t is not None):
        raise ValueError("Expected all tensors on one CUDA device")
    module_name, record = record_for(device, program)
    arch = arch_for(device)
    stages = cake_jit.registered_stages(program)
    if not generated_program_available(device, backward=backward, program=program):
        raise NotImplementedError(
            f"the registered NoPE DSA training program {module_name!r} has no "
            f"{'backward' if backward else 'forward'} stages for {arch} (registered: {stages})"
        )
    # every NoPE backward accumulates into a natural FP32 buffer (the caller's dkv_acc or the fresh dkv):
    # the natural-layout main stage adds into it when the record's size rule (or ``dkv_direct``) selects it,
    # otherwise the permuted accumulator and the cast serve it
    direct = bool(backward) and _plan_nope_direct(
        record, stages, num_queries, num_kv, dkv_direct
    )
    scale = NOPE_SOFTMAX_SCALE if softmax_scale is None else float(softmax_scale)
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
            program=program,
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
        "dq",
        (num_queries, NUM_HEADS, D_LATENT),
        torch.bfloat16,
    )
    _check_output(outputs.get("dkv"), "dkv", (num_kv, D_LATENT), torch.float32)
    layout = workspace_layout_nope(
        num_queries,
        num_kv,
        topk,
        backward=backward,
        key_passes=passes,
        dkv_direct=direct,
        token_chunk=_record_token_chunk(record, topk),
    )
    has_topk_length = topk_length is not None
    derive_length = topk_length is None or bool(derive_topk_length)
    cast = bool(backward) and not direct
    scalars = dict(
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=scale,
        has_topk_length=int(has_topk_length),
        # the cast writes the fresh natural FP32 dkv densely (out_f32) or adds into dkv_acc (accumulate);
        # a direct binding launches no cast and these are inert
        out_f32=int(cast and not accumulate),
        dst_row_stride=int(dst_row_stride),
        has_dst_map=int(accumulate and dkv_dst_map is not None),
        accumulate=int(cast and accumulate),
    )
    T, S = num_queries, num_kv
    constants: dict[str, Any] = dict(
        indices_offset=indices.storage_offset() % POINTER_GRANULE,
        idx_stride=int(indices.stride(0)),
        num_queries=T,
        num_kv=S,
        topk=topk,
        has_topk_length=int(has_topk_length),
        sm_scale=scale,
        scale_log2=scale * LOG2E,
        num_rows=T * NUM_HEADS,  # delta: (token, head) rows
        # delta: dout's token / head strides in elements and the element offset of the pointer the launcher
        # passes (the validated view itself, 16-byte aligned: offset 0); both strides are multiples of 8
        **_dout_stride_constants(dout),
        latent_groups=S * D_LATENT // 32,  # cast: permuted 32-element groups
        rope_groups=0,  # cast: no rope part
        out_f32=scalars["out_f32"],
        dst_row_stride=scalars["dst_row_stride"],
        has_dst_map=scalars["has_dst_map"],
        accumulate=scalars["accumulate"],
        derive_length=int(derive_length),
        # a forward program that converts per-document index rows itself (NOPE_FWD_CONVERT_ARGS) takes these
        # too; a plan over global rows passes ``convert = 0`` (the per-document operands are never read; the
        # launcher of a program without them ignores them)
        num_docs=1,
        search_iters=1,
        causal=0,
        convert=0,
        # main-stage accumulator geometry: the caller's rows (their stride, the map flag) or the fresh
        # contiguous [S, 512] buffer on a direct binding, the private permuted accumulator otherwise
        **_main_accumulator_constants_nope(
            direct,
            dst_row_stride if accumulate else D_LATENT,
            dkv_dst_map is not None,
        ),
        token_base=0,  # token = token_base + token_step * (blockIdx.x + k * gridDim.x): the persistent loop
        token_step=1,
        num_tokens=T,  # compaction: the whole row; a multi-pass backward overrides it per token chunk
    )
    pass_overrides: tuple[tuple[dict[str, int], ...], ...] = ()
    token_chunks: tuple[tuple[int, int], ...] = ()
    if backward and passes == 1:
        constants.update(pass_lo=0, pass_hi=S, dq_mode=0)
    elif backward:
        token_chunks = _pass_token_chunks(record, num_queries, topk)
        ranges = key_pass_ranges(num_kv, passes)
        # per chunk launch: the compaction reads ``num_tokens``; the main pass kernel's token count is
        # ``num_queries`` -- the launch's tokens (CTA b takes tokens b, b + grid, ... below it), so it is the
        # chunk's count, not the row's, and its scratch rows (dq_partial, key_scratch, pass_counts) are the
        # launch-relative token index into the one-chunk scratch
        pass_overrides = tuple(
            tuple(
                dict(
                    pass_lo=lo,
                    pass_hi=hi,
                    dq_mode=key_pass_dq_mode(index, passes),
                    token_base=int(base),
                    num_tokens=int(num_tokens),
                    num_queries=int(num_tokens),
                )
                for index, (lo, hi) in enumerate(ranges)
            )
            for base, num_tokens in token_chunks
        )
    fwd, delta, main, natural, compact, main_pass, pass_natural, cast_stage = (
        _program_stages(
            (
                "fwd",
                "bwd_delta",
                "bwd_main",
                "bwd_main_natural",
                "bwd_compact",
                "bwd_main_pass",
                "bwd_main_pass_natural",
                "bwd_cast",
            ),
            program,
        )
    )
    entries: dict[Any, _StageEntry] = {fwd: _stage_entry(fwd, arch, scalars, program)}
    backward_order: list = []
    if backward:
        main_stage = natural if direct else main
        pass_stages = (compact, pass_natural if direct else main_pass)
        # the main stages' persistent grid (nope_bwd_launch_grid): the launch's token count is ``num_queries`` --
        # the whole row, or the chunk's tokens -- and its key window the launch's ``num_kv``
        sm_count = _sm_count(_device_index(device))
        entries[delta] = _stage_entry(delta, arch, scalars, program)
        backward_order.append(delta)
        if passes == 1:
            grid = (nope_bwd_launch_grid(T, S, sm_count), 1, 1)
            entries[main_stage] = _stage_entry(main_stage, arch, scalars, program, grid)
            backward_order.append(main_stage)
        else:
            for chunk_index, (_base, num_tokens) in enumerate(token_chunks):
                chunk_scalars = dict(scalars, num_queries=int(num_tokens))
                grid = (nope_bwd_launch_grid(int(num_tokens), S, sm_count), 1, 1)
                for stage in pass_stages:
                    entries[(stage, chunk_index)] = _stage_entry(
                        stage,
                        arch,
                        chunk_scalars,
                        program,
                        None if stage == compact else grid,
                    )
                for index in range(passes):
                    backward_order += [
                        (stage, chunk_index, index) for stage in pass_stages
                    ]
        if cast:
            entries[cast_stage] = _stage_entry(cast_stage, arch, scalars, program)
            backward_order.append(cast_stage)
    return _Plan(
        module_name=module_name,
        arch=arch,
        stages=stages,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=scale,
        backward=bool(backward),
        dkv_fp32=False,
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
        forward_stages=cake_jit.NOPE_FORWARD_STAGES,
        dq_partial_bytes_per_token=NOPE_DQ_PARTIAL_BYTES_PER_TOKEN,
        fresh_dkv=bool(backward and not accumulate),
    )


_INERT_CU_SEQLENS: dict[str, torch.Tensor] = {}


def _inert_cu_seqlens(device: torch.device) -> torch.Tensor:
    """Cached int32 ``[2]`` constant on ``device``: the prefix-sum operands of a forward launch over global rows
    (``convert = 0``), where a converting program needs valid pointers it never reads."""
    key = str(device)
    t = _INERT_CU_SEQLENS.get(key)
    if t is None:
        t = _INERT_CU_SEQLENS[key] = torch.zeros(2, dtype=torch.int32, device=device)
    return t


_INERT_PLACEHOLDERS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _inert_placeholders(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached zero-size ``(FP32, BF16)`` placeholders on ``device``: the cast stage's rope operands
    (``rope_groups = 0``: never dereferenced), shared by every call instead of two allocations per backward."""
    key = str(device)
    pair = _INERT_PLACEHOLDERS.get(key)
    if pair is None:
        pair = _INERT_PLACEHOLDERS[key] = (
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.bfloat16, device=device),
        )
    return pair


def _bound_values_nope(plan: _Plan, t: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Launch values of a NoPE ``plan`` over the tensors ``t`` (the kernels' own argument names).

    A direct binding's main stage adds into ``t["dkv_acc"]`` (the caller's rows, flat alias) or the fresh
    ``t["dkv"]``; otherwise the permuted accumulator ``t["dkv_latent_acc"]`` feeds the cast, whose rope operands are
    the zero-size placeholders ``t["inert_f32"]`` / ``t["inert_bf16"]`` (``rope_groups = 0``: never dereferenced) and
    whose natural output is ``t["dkv"]`` (fresh, ``out_f32``) or the caller's rows (``dst_packed``, ``accumulate``).
    """
    indices, _ = _pointer_operand(t["indices"])
    values: dict[str, Any] = dict(plan.constants)
    values.update(t, indices=indices)
    # the converting forward's pointer operands under ``convert = 0``: the index operand stands as ``indices_out``
    # -- as the contiguous span of its storage when the rows are a strided view (a column slice of a wider
    # buffer): the binding takes contiguous pointer operands and the kernel never reads it -- and a per-device
    # int32 constant as the prefix sums (valid pointers the kernel never reads)
    inert_cu = _inert_cu_seqlens(t["indices"].device)
    indices_out = (
        indices if indices.is_contiguous() else _index_storage_alias(t["indices"])[0]
    )
    values.update(indices_out=indices_out, cu_seqlens_q=inert_cu, cu_seqlens_k=inert_cu)
    if plan.backward:
        dst_map = (
            t["dkv_dst_map"] if t.get("dkv_dst_map") is not None else t["topk_length"]
        )
        values["dkv_dst_map"] = dst_map
        if plan.dkv_direct:
            values["dkv_f32"] = t["dkv_acc"] if plan.accumulate_dkv else t["dkv"]
        else:
            acc = t["dkv_latent_acc"]
            values.update(
                dkv_f32=acc,
                src_latent=acc,
                src_rope=t["inert_f32"],
                dst_latent=t["inert_bf16"],
                dst_rope=t["inert_bf16"],
                dst_latent_f32=t["dkv"] if plan.fresh_dkv else acc,
                dst_rope_f32=t["inert_f32"],
                dst_packed=t["dkv_acc"] if plan.accumulate_dkv else acc,
                dst_map=dst_map,
            )
        if plan.key_passes == 1:
            values.update(
                dq_partial=t["delta"],
                key_scratch=t["topk_length"],
                pass_counts=t["topk_length"],
            )
    return values


@dataclass
class DSANopeTrainRunner(DSATrainRunner):
    """The prepared forward / backward launches of one NoPE tensor binding.

    ``forward()`` writes ``out``, ``lse`` and ``o_lo``; ``backward()`` writes ``dq`` and returns
    ``(dq, dkv)`` -- ``dkv`` the natural-layout FP32 ``[S, 512]`` combined dK + dV of this binding
    (``fresh_dkv``: zero-filled and accumulated by the main stage on a direct binding, written densely by the cast
    otherwise), or ``None`` when the binding adds into the caller's ``dkv_acc`` (``accumulate_dkv``).  ``step()``
    runs both.  No launch allocates or synchronizes; capture into a CUDA graph belongs to the caller.
    """

    fresh_dkv: bool = False

    # The NoPE runner returns ``(dq, dkv)`` by contract (class docstring); the base runner's
    # ``(dq_latent, dq_rope, dkv_latent, dk_rope)`` tuple has no rope members here.
    def backward(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:  # type: ignore[override]
        if not self.has_backward:
            raise NotImplementedError(
                "this runner was prepared without the backward (pass dout / backward=True)"
            )
        t = self.tensors
        if self.dkv_direct:
            if self.fresh_dkv:
                # the fresh natural dkv is the buffer the main stage adds into
                t["dkv"].zero_()
        else:
            t["dkv_latent_acc"].zero_()
        self._run(self.backward_order)
        return t["dq_latent"], (t["dkv"] if self.fresh_dkv else None)


def prepare_dsa_nope_train(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    dout: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    o_lo: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    backward: Optional[bool] = None,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    derive_topk_length: bool = False,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
    backend: str = "cake",
) -> DSANopeTrainRunner:
    """Validate one NoPE binding and prepare its launches.

    ``q_latent`` BF16 ``[T, 64, 512]`` (strided token / head views in place), ``kv_latent`` BF16 ``[S, 512]`` (the
    shared latent, K = V), ``indices`` int32 ``[T, K_slots]`` global key rows (``K_slots >= 1``, any width).
    ``backward`` defaults to ``dout is not None``; ``dout`` is read in place in any layout
    :func:`validate_dsa_nope_inputs` admits (dense or strided, head-major included).  ``softmax_scale`` defaults to
    ``NOPE_SOFTMAX_SCALE`` (1/16).  Missing outputs and the workspace are allocated here (the only allocations of a
    prepared runner); pass ``workspace_buffer`` of :func:`dsa_nope_train_workspace_size` bytes to reuse storage.
    ``backward()`` returns ``(dq, dkv)`` with ``dkv`` the natural-layout FP32 ``[S, 512]`` combined dK + dV (the
    ``dkv`` tensor when given, else a fresh one) -- or, with ``dkv_acc`` (FP32 ``[S_dst, >= 512]``, see
    :func:`_check_dkv_acc`), ADDS the gradient into columns ``0:512`` of row ``dkv_dst_map[s]`` (optional int32
    ``[S]``; identity by default; repeated rows sum) and returns ``None`` for ``dkv``; the caller owns the zeroing
    of ``dkv_acc``.  ``key_passes`` / ``num_segments`` as in :func:`prepare_dsa_train` (the NoPE record's policy:
    2048 B per key).  The natural-layout main stage adds straight into the FP32 buffer (``dkv_acc`` or the fresh
    ``dkv``, which the runner zero-fills per backward) when the record's size rule selects it, otherwise the
    permuted accumulator and the cast serve the same rows; ``dkv_direct`` overrides the rule (``None`` = the rule,
    ``True`` / ``False`` = force / disable the direct path).  ``check_dst_map``, ``derive_topk_length`` and the row
    length semantics as in :func:`prepare_dsa_train`.
    """
    if backend != "cake":
        raise ValueError("DSA sparse-attention training supports backend='cake'")
    if backward is None:
        backward = dout is not None
    outputs = dict(out=out, lse=lse, o_lo=o_lo, dq_latent=dq, dkv=dkv)
    plan = _plan_nope(
        q_latent,
        kv_latent,
        indices,
        topk_length=topk_length,
        dout=dout,
        softmax_scale=softmax_scale,
        outputs=outputs,
        backward=backward,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=num_segments,
        derive_topk_length=derive_topk_length,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
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
        q_latent=q_latent, kv_latent=kv_latent, indices=indices
    )
    if topk_length is None:
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
        t.update(_scratch_nope(plan, flat))
        t["dq_latent"] = (
            dq
            if dq is not None
            else fresh((num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
        )
        # the cast's rope operands: zero-size placeholders of the declared dtypes (rope_groups = 0)
        t["inert_f32"] = fresh((0,), torch.float32)
        t["inert_bf16"] = fresh((0,), torch.bfloat16)
        if plan.accumulate_dkv:
            t["dkv_acc"], _row_stride = _check_dkv_acc(
                dkv_acc, dkv_dst_map, num_kv, min_cols=D_LATENT
            )
            if dkv_dst_map is not None:
                t["dkv_dst_map"] = dkv_dst_map
        else:
            # the natural FP32 combined dK + dV: the accumulator of a direct binding (zero-filled by
            # ``backward()``), the cast's dense output otherwise
            t["dkv"] = (
                dkv if dkv is not None else fresh((num_kv, D_LATENT), torch.float32)
            )
    values = _bound_values_nope(plan, t)
    return DSANopeTrainRunner(
        module_name=plan.module_name,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=plan.topk,
        softmax_scale=plan.softmax_scale,
        tensors=t,
        launches=_launches(plan, values),
        stages=plan.stages,
        dkv_fp32=False,
        has_topk_length=plan.has_topk_length,
        device_index=plan.device_index,
        workspace=flat,
        layout=layout,
        key_passes=plan.key_passes,
        backward_order=plan.backward_order,
        accumulate_dkv=plan.accumulate_dkv,
        dkv_direct=plan.dkv_direct,
        forward_stages=plan.forward_stages,
        fresh_dkv=plan.fresh_dkv,
    )


def forward_binding_key_nope(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    derive_topk_length: bool = False,
) -> tuple:
    """Cache key of a NoPE forward plan (tag ``"nope_fwd"``: disjoint from the RoPE keys): the :func:`_geometry`
    of every input, the scale and the derive flag.  Data pointers are not part of it."""
    return (
        "nope_fwd",
        _geometry(q_latent),
        _geometry(kv_latent),
        _geometry(indices),
        _geometry(topk_length),
        float(softmax_scale),
        bool(derive_topk_length),
    )


def backward_binding_key_nope(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    key_passes: Optional[int] = None,
    num_segments: int = 1,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
) -> tuple:
    """Cache key of a NoPE backward plan (tag ``"nope_bwd"``): the forward key's inputs plus the saved outputs,
    ``dout``, the ``key_passes`` override, the segment count, the geometry of ``dkv_acc`` / ``dkv_dst_map``
    (``None`` when absent) and the ``dkv_direct`` option -- part of the key with and without ``dkv_acc``, since a
    fresh ``dkv`` is accumulated under the same rule."""
    return (
        "nope_bwd",
        _geometry(q_latent),
        _geometry(kv_latent),
        _geometry(indices),
        _geometry(out),
        _geometry(o_lo),
        _geometry(lse),
        _geometry(dout),
        _geometry(topk_length),
        float(softmax_scale),
        None if key_passes is None else int(key_passes),
        int(num_segments),
        _geometry(dkv_acc),
        _geometry(dkv_dst_map),
        dkv_direct,
    )


class _NopeBinding(_Binding):
    """A remembered NoPE plan plus the one value it owns (the materialized ``topk_length``)."""

    __slots__ = ()

    def prepared_forward(
        self, q_latent, kv_latent, indices, topk_length, extra=None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], Callable[[], None]]:
        """``((out, lse, o_lo), launch)``: the forward's outputs allocated and its launch values bound; ``launch()``
        issues the forward stage.  The varlen entry binds first, launches its index preparation and then calls
        ``launch()``, so the forward kernel follows that launch with no host work in between.  ``extra``: further
        launch values (the operands of a forward program that converts per-document index rows itself)."""
        (out, lse, o_lo), values = self._forward_values(
            q_latent, kv_latent, indices, topk_length, extra
        )
        stages = self.plan.forward_stages

        def launch() -> None:
            self._run(values, stages)

        return (out, lse, o_lo), launch

    def _forward_values(
        self, q_latent, kv_latent, indices, topk_length, extra=None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
        """The forward's fresh outputs and its bound launch values."""
        T, device = self.plan.num_queries, q_latent.device
        out = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        lse = _alloc((T, NUM_HEADS), torch.float32, device)
        o_lo = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        t = dict(
            q_latent=q_latent,
            kv_latent=kv_latent,
            indices=indices,
            topk_length=topk_length if topk_length is not None else self.topk_length,
            out=out,
            lse=lse,
            o_lo=o_lo,
        )
        values = _bound_values_nope(self.plan, t)
        if extra:
            values.update(extra)
        return (out, lse, o_lo), values

    def prebound_forward(
        self, q_latent, kv_latent, indices, topk_length, extra=None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[_Launch, ...]]:
        """``((out, lse, o_lo), launches)``: as :meth:`prepared_forward`, with every tensor operand converted to its
        tvm-ffi view now and the forward's stage entries bound to those values.  Each launch, issued by the caller
        inside its own FFI stream context, costs one binding call and nothing else: the packed entry issues the
        forward this way right after the index-preparation launch."""
        (out, lse, o_lo), values = self._forward_values(
            q_latent, kv_latent, indices, topk_length, extra
        )
        plan, values = self.plan, _ffi_values(values)
        launches = []
        for key in plan.forward_stages:
            if isinstance(key, tuple):
                stage, chunk, index = key
                e = plan.entries[(stage, chunk)]
                bound = dict(values, **_ffi_values(plan.pass_overrides[chunk][index]))
                launches.append(_Launch(stage, e.launcher, e.run, bound, e.grid))
            else:
                e = plan.entries[key]
                launches.append(_Launch(key, e.launcher, e.run, values, e.grid))
        return (out, lse, o_lo), tuple(launches)

    def forward(self, q_latent, kv_latent, indices, topk_length):
        outputs, launch = self.prepared_forward(
            q_latent, kv_latent, indices, topk_length
        )
        launch()
        return outputs

    def _launch_stages(self, values: dict[str, Any], keys: tuple) -> None:
        """The stage launches of ``keys`` over ``values``, inside the caller's FFI stream context."""
        plan = self.plan
        for key in keys:
            if isinstance(key, tuple):
                stage, chunk, index = key
                e = plan.entries[(stage, chunk)]
                e.launcher(
                    e.run, dict(values, **plan.pass_overrides[chunk][index]), e.grid
                )
            else:
                e = plan.entries[key]
                e.launcher(e.run, values, e.grid)

    def backward_operands(
        self,
        q_latent,
        kv_latent,
        indices,
        out,
        o_lo,
        lse,
        topk_length,
        dkv_acc=None,
        dkv_dst_map=None,
        check_dst_map=False,
    ) -> dict[str, torch.Tensor]:
        """The backward's launch operands that are settled before its call: the saved tensors, the lengths (the
        binding's own vector without caller lengths), the cast's zero-size placeholders and the caller's accumulator
        validated once (:func:`_check_dkv_acc`; ``check_dst_map`` synchronizes here).  :meth:`launch_backward` adds
        ``dout`` and the fresh buffers."""
        plan = self.plan
        inert_f32, inert_bf16 = _inert_placeholders(q_latent.device)
        t = dict(
            q_latent=q_latent,
            kv_latent=kv_latent,
            indices=indices,
            topk_length=topk_length if topk_length is not None else self.topk_length,
            out=out,
            lse=lse,
            o_lo=o_lo,
            inert_f32=inert_f32,
            inert_bf16=inert_bf16,
        )
        if plan.accumulate_dkv:
            if dkv_acc is None:
                raise RuntimeError(
                    "this binding accumulates the dK + dV gradient into dkv_acc and the call provides none"
                )
            t["dkv_acc"], _row_stride = _check_dkv_acc(
                dkv_acc,
                dkv_dst_map,
                plan.num_kv,
                check_dst_map=check_dst_map,
                min_cols=D_LATENT,
            )
            if dkv_dst_map is not None:
                t["dkv_dst_map"] = dkv_dst_map
        return t

    def launch_backward(
        self, operands: dict[str, torch.Tensor], dout: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``(dq, dkv)`` over the settled ``operands``: the fresh buffers are allocated (``torch.empty``) and the
        launch values bound before the first launch; then the zero fills of the FP32 buffers this call owns and the
        stages follow back to back, with no host work between the launches."""
        plan = self.plan
        T, S, device = plan.num_queries, plan.num_kv, dout.device
        dq = _alloc((T, NUM_HEADS, D_LATENT), torch.bfloat16, device)
        delta = _alloc((T, NUM_HEADS), torch.float32, device)
        t = dict(operands, dout=dout, dq_latent=dq, delta=delta)
        fills = []
        if not plan.dkv_direct:
            t["dkv_latent_acc"] = acc = _alloc((S, D_LATENT), torch.float32, device)
            fills.append(acc)
        if plan.key_passes > 1:
            rows = plan.pass_rows
            t["dq_partial"] = _alloc(
                (rows, plan.dq_partial_bytes_per_token // 4), torch.float32, device
            )
            t["key_scratch"] = _alloc((rows, plan.topk), torch.int32, device)
            t["pass_counts"] = _alloc((rows,), torch.int32, device)
        dkv = None
        if not plan.accumulate_dkv:
            # the fresh natural dkv: zero-filled when the main stage adds into it (direct), written densely by
            # the cast otherwise
            t["dkv"] = dkv = _alloc((S, D_LATENT), torch.float32, device)
            if plan.dkv_direct:
                fills.append(dkv)
        values = _bound_values_nope(plan, t)
        with _ffi_stream_context(plan.device_index):
            for buffer in fills:
                buffer.zero_()
            self._launch_stages(values, plan.backward_order)
        return dq, dkv

    def backward(
        self,
        q_latent,
        kv_latent,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        dkv_acc=None,
        dkv_dst_map=None,
        check_dst_map=False,
    ):
        operands = self.backward_operands(
            q_latent,
            kv_latent,
            indices,
            out,
            o_lo,
            lse,
            topk_length,
            dkv_acc,
            dkv_dst_map,
            check_dst_map,
        )
        return self.launch_backward(operands, dout)


def _empty_forward_nope(
    q_latent, kv_latent, indices, topk_length
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(out, lse, o_lo)`` of a NoPE call without query rows: validated, empty, neither bound nor launched."""
    validate_dsa_nope_inputs(
        q_latent, kv_latent, indices, topk_length, allow_empty_queries=True
    )
    device = q_latent.device
    out = _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)
    lse = _alloc((0, NUM_HEADS), torch.float32, device)
    return out, lse, _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)


def _empty_backward_nope(
    q_latent,
    kv_latent,
    indices,
    out,
    o_lo,
    lse,
    dout,
    topk_length,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    check_dst_map: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Gradients of a NoPE call without query rows: empty ``dq``, a zero FP32 ``dkv`` (``None`` with ``dkv_acc``:
    nothing to accumulate); nothing launched."""
    validate_dsa_nope_inputs(
        q_latent, kv_latent, indices, topk_length, dout=dout, allow_empty_queries=True
    )
    _check_output(out, "out", (0, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(o_lo, "o_lo", (0, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(lse, "lse", (0, NUM_HEADS), torch.float32)
    device, num_kv = q_latent.device, int(kv_latent.shape[0])
    dq = _alloc((0, NUM_HEADS, D_LATENT), torch.bfloat16, device)
    if dkv_acc is not None:
        _check_dkv_acc(
            dkv_acc,
            dkv_dst_map,
            num_kv,
            check_dst_map=check_dst_map,
            min_cols=D_LATENT,
        )
        return dq, None
    return dq, _alloc((num_kv, D_LATENT), torch.float32, device, zero=True)


def forward_nope(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    derive_topk_length: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NoPE forward pass: ``(out, lse, o_lo)``.

    ``q_latent`` BF16 ``[T, 64, 512]``, ``kv_latent`` BF16 ``[S, 512]`` (the shared latent, K = V), ``indices``
    int32 ``[T, K_slots]`` global key rows (any ``K_slots >= 1``; ``-1`` / ``>= S`` invalid).  ``softmax_scale``
    defaults to 1/16.  A fully masked row gives ``out = 0`` and ``lse = -inf``.  Allocates the outputs and
    launches over the remembered plan of the input geometry; a call without query rows returns empty outputs
    without launching.  ``topk_length`` / ``derive_topk_length`` as in :func:`forward`.
    """
    if derive_topk_length and topk_length is None:
        raise ValueError(
            "derive_topk_length=True needs a topk_length [T] int32 tensor for the forward to fill"
        )
    if _no_query_rows(q_latent):
        return _empty_forward_nope(q_latent, kv_latent, indices, topk_length)
    scale = NOPE_SOFTMAX_SCALE if softmax_scale is None else float(softmax_scale)
    key = forward_binding_key_nope(
        q_latent, kv_latent, indices, topk_length, scale, derive_topk_length
    )
    binding = _binding(
        key,
        lambda: _plan_nope(
            q_latent,
            kv_latent,
            indices,
            topk_length=topk_length,
            dout=None,
            softmax_scale=scale,
            outputs={},
            backward=False,
            key_passes=None,
            derive_topk_length=derive_topk_length,
        ),
        _NopeBinding,
    )
    return binding.forward(q_latent, kv_latent, indices, topk_length)


def backward_nope(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """NoPE backward pass from the saved forward outputs: ``(dq, dkv)``.

    ``dq`` is BF16 ``[T, 64, 512]`` (written once per row: bitwise deterministic for one pass plan).  ``dkv`` is
    the combined gradient of the shared latent, ``dK + dV``, natural layout, FP32 ``[S, 512]``, accumulated in
    FP32 by the kernels' atomics (the summation order is not fixed: run-to-run spread within FP32 rounding):

    * fresh (default, ``dkv_acc=None``): a new buffer filled by this call;
    * accumulate (``dkv_acc`` FP32 ``[S_dst, >= 512]``, row stride a multiple of 4 and ``>= 512``, 16-byte base;
      optional int32 ``[S]`` ``dkv_dst_map`` with the destination row of every key row, identity by default,
      repeated rows sum): ``+=`` into columns ``0:512`` of the caller's rows, in FP32, and ``dkv`` is ``None``.
      The caller owns the zeroing; chunked backward passes of one row sum into one buffer without intermediate
      rounding.

    ``dout`` may be a head-major view (strides ``[512, T * 512, 1]``) or any strided ``[T, 64, 512]`` view the
    layout rules admit: it is read in place (no host copy; the delta stage takes its token / head strides).
    ``key_passes`` / ``num_segments`` as in :func:`backward` (the NoPE record's policy plans one pass; an explicit
    ``key_passes`` forces the pass form); ``dkv_direct`` (default ``None`` = the record's rule, the natural-layout
    main stage adding straight into the FP32 buffer -- fresh or ``dkv_acc`` alike -- whenever the row has at least
    one key per query; ``True`` / ``False`` = force / disable it) and ``check_dst_map`` as in
    :func:`prepare_dsa_nope_train`.
    A call without query rows returns an empty ``dq`` and a zero ``dkv`` (``None`` with ``dkv_acc``) without
    launching.
    """
    if dkv_dst_map is not None and dkv_acc is None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
    if _no_query_rows(q_latent):
        return _empty_backward_nope(
            q_latent,
            kv_latent,
            indices,
            out,
            o_lo,
            lse,
            dout,
            topk_length,
            dkv_acc,
            dkv_dst_map,
            check_dst_map,
        )
    scale = NOPE_SOFTMAX_SCALE if softmax_scale is None else float(softmax_scale)
    key = backward_binding_key_nope(
        q_latent,
        kv_latent,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        scale,
        key_passes,
        num_segments,
        dkv_acc,
        dkv_dst_map,
        dkv_direct,
    )
    binding = _binding(
        key,
        lambda: _plan_nope(
            q_latent,
            kv_latent,
            indices,
            topk_length=topk_length,
            dout=dout,
            softmax_scale=scale,
            outputs=dict(out=out, lse=lse, o_lo=o_lo),
            backward=True,
            key_passes=key_passes,
            dkv_acc=dkv_acc,
            dkv_dst_map=dkv_dst_map,
            num_segments=num_segments,
            dkv_direct=dkv_direct,
            check_dst_map=check_dst_map,
        ),
        _NopeBinding,
    )
    return binding.backward(
        q_latent,
        kv_latent,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length,
        dkv_acc,
        dkv_dst_map,
        check_dst_map,
    )


class DSASparseAttentionNopeFunction(torch.autograd.Function):
    """Autograd wrapper of the NoPE program: saves every tensor the backward reads and settles the rest of it at
    forward time (:class:`_NopeBackwardPrep`).

    Twelve forward inputs ``(q_latent, kv_latent, indices, topk_length, softmax_scale, key_passes, dkv_acc,
    dkv_dst_map, num_segments, derive_topk_length, dkv_direct, check_dst_map)``; returns ``(out, lse)``.  With
    ``derive_topk_length`` (the public entry's call without caller lengths; ``topk_length`` is then ``None``) the
    route follows the table size (:func:`nope_flat_index_mode`): below :data:`NOPE_CONVERT_SLOT_LIMIT` slots one
    launch of the forward over the caller's rows as given, deriving each row's length (the last valid slot + 1)
    into a fresh ``[T]`` vector, and the backward reads the caller's ``indices`` tensor with that vector -- the
    rows are then a saved tensor, so autograd rejects an in-place edit of ``indices`` between the forward and the
    backward (the stage route saves a fresh buffer and does not); at or above the limit the forward compacts the
    rows through the index-preparation stage's flat mode (:func:`_flat_forward_nope`) and the backward reads the
    compacted rows and their counts.  Otherwise the caller's rows and lengths reach both kernels as given
    (:func:`forward_nope`).  The backward returns ``(dq, dkv)`` for ``q_latent`` / ``kv_latent``
    (``dkv`` is ``None`` when it went into ``dkv_acc``) and no gradient for the ten other inputs.  The FP32
    ``dkv`` of :func:`backward_nope` reaches a BF16 ``kv_latent`` leaf as BF16: the autograd engine casts a
    gradient to the leaf's dtype.  The FP32 contract
    therefore holds through ``dkv_acc`` -- a tensor (validated when the forward binds it; ``check_dst_map=True``
    synchronizes there) or a zero-argument callable producing the accumulator, or an ``(accumulator,
    dkv_dst_map)`` pair, at backward time (:data:`DkvAccSource`; validated then) -- the eager :func:`backward_nope`
    and the prepared runner.  Seven tensors are saved in a fixed order (``q_latent, kv_latent, out, lse, o_lo,
    indices, topk_length`` -- a zero-size placeholder when the lengths are absent): under non-reentrant
    activation checkpointing they are released after the forward and recomputed, and the ``ctx`` keeps nothing
    else of size; a tensor ``dkv_acc`` / ``dkv_dst_map`` is caller-owned and stays on the node (the prep).
    """

    @staticmethod
    def forward(
        ctx,
        q_latent,
        kv_latent,
        indices,
        topk_length,
        softmax_scale,
        key_passes=None,
        dkv_acc=None,
        dkv_dst_map=None,
        num_segments=1,
        derive_topk_length=False,
        dkv_direct=None,
        check_dst_map=False,
    ):
        num_rows = int(q_latent.shape[0])
        if derive_topk_length and topk_length is None and num_rows > 0:
            small = (
                indices.dim() == 2
                and nope_flat_index_mode(num_rows, int(indices.shape[1])) == "mask"
            )
            if small:
                # the small-table route: one launch over the caller's rows as given; the forward derives each
                # row's length (last valid slot + 1) into the fresh vector the backward reads with those rows
                topk_length = _alloc((num_rows,), torch.int32, q_latent.device)
                out, lse, o_lo = forward_nope(
                    q_latent,
                    kv_latent,
                    indices,
                    topk_length=topk_length,
                    softmax_scale=softmax_scale,
                    derive_topk_length=True,
                )
            else:
                out, lse, o_lo, indices, topk_length = _flat_forward_nope(
                    q_latent, kv_latent, indices, softmax_scale
                )
        else:
            out, lse, o_lo = forward_nope(
                q_latent,
                kv_latent,
                indices,
                topk_length=topk_length,
                softmax_scale=softmax_scale,
                derive_topk_length=derive_topk_length,
            )
        ctx.set_materialize_grads(False)
        ctx.nope_prep = _NopeBackwardPrep(
            ctx,
            q_latent,
            kv_latent,
            indices,
            out,
            o_lo,
            lse,
            topk_length,
            softmax_scale,
            key_passes,
            num_segments,
            dkv_acc,
            dkv_dst_map,
            dkv_direct,
            check_dst_map,
        )
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        if dlse is not None:
            raise NotImplementedError(
                "gradients through lse are not supported; only out is differentiable"
            )
        if dout is None:  # out unused downstream
            return (None,) * 12
        return _nope_backward_from_ctx(ctx, dout) + (None,) * 10


class _NopeBackwardPrep:
    """What the NoPE autograd wrappers settle at forward time for their backward, so that the backward's host work
    before its first launch is the completion of the binding key with ``dout``'s geometry, the cache lookup, the
    two fresh allocations and the binding of the launch values: the options, the geometry key without ``dout`` and
    without the accumulator, and the presence flag of the lengths.  It holds no produced tensor: the saved outputs,
    the index rows, ``o_lo`` and the lengths go through ``ctx.save_for_backward`` in one fixed order (seven
    tensors), so that under activation checkpointing nothing of size outlives the forward.  The caller-owned
    accumulator stays here as before -- a tensor ``dkv_acc`` (validated here, :func:`_check_dkv_acc`;
    ``check_dst_map`` synchronizes here) with its pointer operand and ``dkv_dst_map`` in ``operands`` -- or as the
    zero-argument callable a deferred accumulator comes from, resolved and validated in the backward
    (:data:`DkvAccSource`)."""

    __slots__ = (
        "softmax_scale",
        "key_passes",
        "num_segments",
        "dkv_direct",
        "check_dst_map",
        "has_topk_length",
        "dkv_acc",
        "dkv_dst_map",
        "dkv_acc_source",
        "operands",
        "key_head",
        "key_mid",
    )

    def __init__(
        self,
        ctx,
        q_latent,
        kv_latent,
        indices,
        out,
        o_lo,
        lse,
        topk_length,
        softmax_scale,
        key_passes,
        num_segments,
        dkv_acc,
        dkv_dst_map,
        dkv_direct,
        check_dst_map,
    ):
        if dkv_dst_map is not None and dkv_acc is None:
            raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
        self.softmax_scale = float(softmax_scale)
        self.key_passes, self.num_segments = key_passes, int(num_segments)
        self.dkv_direct, self.check_dst_map = dkv_direct, bool(check_dst_map)
        acc_tensor = dkv_acc if isinstance(dkv_acc, torch.Tensor) else None
        self.operands = {}
        if acc_tensor is not None:
            # caller-owned: validated (and its map's values checked) now, the pointer operand bound now
            self.operands["dkv_acc"], _row_stride = _check_dkv_acc(
                acc_tensor,
                dkv_dst_map,
                int(kv_latent.shape[0]),
                check_dst_map=check_dst_map,
                min_cols=D_LATENT,
            )
            if dkv_dst_map is not None:
                self.operands["dkv_dst_map"] = dkv_dst_map
        elif dkv_acc is not None and not callable(dkv_acc):
            raise TypeError(
                "dkv_acc must be a float32 tensor or a zero-argument callable returning it"
            )
        self.dkv_acc, self.dkv_dst_map = acc_tensor, dkv_dst_map
        self.dkv_acc_source = None if acc_tensor is not None else dkv_acc
        self.has_topk_length = topk_length is not None
        # the backward key with dout's slot (index 7) and the accumulator's geometry (indices 12, 13) left open:
        # completed per backward call
        key = backward_binding_key_nope(
            q_latent,
            kv_latent,
            indices,
            out,
            o_lo,
            lse,
            None,
            topk_length,
            self.softmax_scale,
            key_passes,
            self.num_segments,
            None,
            None,
            dkv_direct,
        )
        self.key_head, self.key_mid = key[:7], key[8:12]
        ctx.save_for_backward(
            q_latent,
            kv_latent,
            out,
            lse,
            o_lo,
            indices,
            _saved_optional(topk_length, indices),
        )

    def backward(self, saved, dout) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_latent, kv_latent, out, lse, o_lo, indices, topk_length = saved
        topk_length = topk_length if self.has_topk_length else None
        dkv_acc, dst_map = _resolve_dkv_acc(
            self.dkv_acc if self.dkv_acc is not None else self.dkv_acc_source,
            self.dkv_dst_map,
        )
        if _no_query_rows(q_latent):
            return backward_nope(
                q_latent,
                kv_latent,
                indices,
                out,
                o_lo,
                lse,
                dout,
                topk_length=topk_length,
                softmax_scale=self.softmax_scale,
                key_passes=self.key_passes,
                dkv_acc=dkv_acc,
                dkv_dst_map=dst_map,
                num_segments=self.num_segments,
                dkv_direct=self.dkv_direct,
            )
        key = (
            self.key_head
            + (_geometry(dout),)
            + self.key_mid
            + (_geometry(dkv_acc), _geometry(dst_map), self.dkv_direct)
        )
        binding = _binding(
            key,
            lambda: _plan_nope(
                q_latent,
                kv_latent,
                indices,
                topk_length=topk_length,
                dout=dout,
                softmax_scale=self.softmax_scale,
                outputs=dict(out=out, lse=lse, o_lo=o_lo),
                backward=True,
                key_passes=self.key_passes,
                dkv_acc=dkv_acc,
                dkv_dst_map=dst_map,
                num_segments=self.num_segments,
                dkv_direct=self.dkv_direct,
            ),
            _NopeBinding,
        )
        inert_f32, inert_bf16 = _inert_placeholders(q_latent.device)
        operands = dict(
            q_latent=q_latent,
            kv_latent=kv_latent,
            indices=indices,
            out=out,
            lse=lse,
            o_lo=o_lo,
            topk_length=topk_length if topk_length is not None else binding.topk_length,
            inert_f32=inert_f32,
            inert_bf16=inert_bf16,
        )
        if self.dkv_acc is not None:
            # the tensor accumulator: validated and bound at forward time
            operands.update(self.operands)
        elif dkv_acc is not None:
            # a callable's result is validated here, including the map's values when asked for
            operands["dkv_acc"], _row_stride = _check_dkv_acc(
                dkv_acc,
                dst_map,
                int(kv_latent.shape[0]),
                check_dst_map=self.check_dst_map,
                min_cols=D_LATENT,
            )
            if dst_map is not None:
                operands["dkv_dst_map"] = dst_map
        return binding.launch_backward(operands, dout)


def _nope_backward_from_ctx(ctx, dout) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(dq, dkv)`` of the NoPE autograd wrappers: the backward the forward settled (:class:`_NopeBackwardPrep`)
    over the saved tensors."""
    return ctx.nope_prep.backward(ctx.saved_tensors, dout)


def dsa_sparse_attention_nope(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    dkv_acc: Optional[DkvAccSource] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
):
    """Autograd entry of the flat NoPE form (see :mod:`flashinfer.dsa_sparse_attention`).

    ``kv_latent`` is ``[S, 512]`` or its one-head form ``[S, 1, 512]`` (squeezed here; the gradient reaches the
    leaf through the view).  ``softmax_scale`` defaults to 1/16.  Without ``dkv_acc`` the ``kv_latent`` gradient
    of the autograd call is the FP32 ``dkv`` of :func:`backward_nope` cast by the autograd engine to the leaf's
    dtype (BF16 for a BF16 leaf); with ``dkv_acc`` the FP32 ``dK + dV`` is accumulated into it and the
    ``kv_latent`` gradient is ``None``.  ``dkv_acc`` is a tensor or a zero-argument callable producing it (or an
    ``(accumulator, dkv_dst_map)`` pair) at backward time (:data:`DkvAccSource`), so the accumulator need not exist
    at forward time; every tensor this call produces for the backward is saved through ``ctx.save_for_backward``
    (a tensor ``dkv_acc`` stays caller-owned on the node) and a checkpointed forward retains nothing of this call.
    ``key_passes``, ``dkv_dst_map``, ``num_segments``, ``dkv_direct`` and
    ``check_dst_map`` as in :func:`dsa_sparse_attention`.  Without ``topk_length`` the route follows the table
    size (:func:`nope_flat_index_mode`).  Below :data:`NOPE_CONVERT_SLOT_LIMIT` slots the forward kernel alone
    reads the index rows (any row stride) as given and derives each row's length -- the last valid slot + 1 --
    while it runs, into a ``[T]`` vector the backward reads with the caller's rows: one launch, a ``-1`` anywhere
    stays a masked slot at its position, and the caller's ``indices`` tensor is a saved tensor until the backward
    (autograd rejects an in-place edit of it in between).  At or above the limit the rows are compacted on device
    in one launch of the index-preparation stage right before the forward kernel: each row's valid keys (``0 <=
    idx < S``) first in ascending slot order, then ``-1``, with the valid count as the row's length -- into a fresh
    ``[T, K_slots]`` buffer and a ``[T]`` vector that the backward reads, so rows with ``-1`` holes cost their
    valid slots in both kernels and the caller's tensor is left untouched.  Hole-free rows give the same bytes on
    either route.  With ``topk_length`` the rows and lengths reach the kernels as given (slots ``>=
    topk_length[t]`` invalid; work up to the last valid slot): callers with holes and lengths compact first with
    :func:`nope_compact_indices`.
    """
    kv_latent = _kv_latent_rows(kv_latent)
    if softmax_scale is None:
        softmax_scale = NOPE_SOFTMAX_SCALE
    # without caller lengths the Function derives them (derive_topk_length): in-kernel below the slot limit, as
    # the counts of the flat stage at or above it
    derive = topk_length is None and q_latent.shape[0] > 0
    out, lse = DSASparseAttentionNopeFunction.apply(
        q_latent,
        kv_latent,
        indices,
        topk_length,
        float(softmax_scale),
        key_passes,
        dkv_acc,
        dkv_dst_map,
        int(num_segments),
        derive,
        dkv_direct,
        check_dst_map,
    )
    return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# Packed form: on-device index preparation
# ---------------------------------------------------------------------------

# The NoPE record's index-preparation stage (``"index_prep"``, first in the record's stage list: it runs before
# ``nope_fwd``): one warp per index row, INDEX_PREP_ROWS_PER_CTA rows per 128-thread CTA.  Under ``compact = 1`` it
# writes every row stably compacted (valid globalized keys first in ascending slot order, then ``-1``) and the
# valid count as the row's length, after the position cut of an optional ``length_in`` vector; ``compact = 0`` is
# the position-preserving rewrite with last valid slot + 1.  Its torch formulation, ``offset_gather_kv_indices``
# (``compact=True`` for the compacting mode), is the reference the tests hold the stage to and the rope varlen
# entry's fallback where the stage is not available.  The rope varlen entry launches the stage at every size; the
# NoPE entries launch it at or above :data:`NOPE_CONVERT_SLOT_LIMIT` slots without caller lengths -- the varlen
# entry also with caller lengths at every size, the flat entry with caller lengths never (the forward alone over
# the rows and lengths as given) -- and take the one-launch position-preserving forward below it
# (:func:`nope_varlen_index_mode`, :func:`nope_flat_index_mode`).
NOPE_INDEX_PREP_STAGE = "index_prep"
INDEX_PREP_ROWS_PER_CTA = 4
# Launch arguments a forward program that converts the per-document rows itself appends to its argument list
# (``cake_launch.STAGE_ARGS["nope_fwd"]``), in this order: the prefix sums, the global-row output the backward
# consumes, the document count and its search depth, the causal flag, and ``convert`` -- 0: global rows in, the
# per-document operands never read; :data:`NOPE_CONVERT_COMPACT` (1): per-document rows in, written to
# ``indices_out`` stably compacted (valid global keys first, then ``-1``) with the valid count as the length;
# :data:`NOPE_CONVERT_MASK` (2): per-document rows in, written to ``indices_out`` position-preserving (the global
# key or ``-1`` at the slot it came from) with the last valid slot + 1 as the length.  Both non-zero values derive
# the lengths in-kernel; the packed entry then needs no index-preparation launch.
NOPE_FWD_CONVERT_ARGS = (
    "cu_seqlens_q",
    "cu_seqlens_k",
    "indices_out",
    "num_docs",
    "search_iters",
    "causal",
    "convert",
)
NOPE_CONVERT_COMPACT = 1
NOPE_CONVERT_MASK = 2
# Index tables (``T x K_slots``) with fewer slots than this take the one-launch position-preserving forward: the
# rows reach the kernel as given (a ``-1`` anywhere stays a masked slot at its position) and the forward derives
# each row's length -- the last valid slot + 1 -- itself, so no host gap separates a preparation launch from the
# forward.  At or above the limit the compacting preparation stage runs first (valid keys first, the count as the
# length) and the forward follows over its rows back to back; on the varlen entry a caller ``topk_length`` takes
# the stage at every size (the cut it applies before compacting), on the flat entry it skips the stage (rows and
# lengths as given).  The rule reads the table size alone -- the host cannot see holes
# without a device sync -- so a small table with holes pays the position-preserving cost of its holes (bounded by
# its size); hole-free rows give the same bytes on either route.  The value is a measured crossover: the largest
# measured table at which the one-launch forward stays within 0.95x of the stage route on the eager forward and
# within 1.02x on the GPU-busy step, plus one, rounded up to the next power of two when the next measured table
# also stays within 1.02x on the GPU-busy step.  Measured on GB200 and GB300 (CUDA 13.4): tables up to 1024 x 512
# qualify and the two one-million-slot tables (4096 x 256, 1024 x 1024) meet the step gate, so the limit is 1 << 20.
NOPE_CONVERT_SLOT_LIMIT = 1 << 20
_INT32_MAX = 2**31 - 1


def index_prep_grid(num_rows: int) -> tuple[int, int, int]:
    """Launch grid of the index-preparation stage over ``num_rows`` index rows: one CTA per four rows."""
    return (max(1, -(-int(num_rows) // INDEX_PREP_ROWS_PER_CTA)), 1, 1)


def index_prep_search_iters(num_docs: int) -> int:
    """Bisection steps of the index-preparation kernel that always converge on a row's document among
    ``num_docs``: ``ceil(log2(num_docs)) + 1`` (``1`` for one document)."""
    return (max(int(num_docs), 1) - 1).bit_length() + 1


def nope_index_prep_available(device: Optional[torch.device] = None) -> bool:
    """Whether the NoPE record serves ``device`` with the index-preparation stage."""
    try:
        _, record = record_for(device, cake_jit.NOPE_PROGRAM)
    except NotImplementedError:
        return False
    return NOPE_INDEX_PREP_STAGE in record


def _rope_varlen_index_mode(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_queries: int,
) -> str:
    """The rope varlen entry's index route: ``"index_prep"`` -- one launch of the NoPE record's index-preparation
    stage -- when the rows are int32 CUDA ``[T, K_slots]`` with ``T >= 1`` on a device the record serves with the
    stage and the prefix sums are contiguous int32 ``[num_docs + 1]`` tensors of one length (the stage's binding
    takes contiguous pointer operands); ``"torch"`` otherwise -- the twin ``offset_gather_kv_indices(...,
    compact=True)``, which accepts host tensors, ``T == 0`` and strided prefix sums and agrees with the stage
    bitwise.  Not :func:`varlen_index_mode`: its ``"kernel"`` branch is the NoPE forward's own conversion, which the
    frozen rope forward does not have."""
    if not gather_kv_indices.is_cuda or int(num_queries) < 1:
        return "torch"
    try:
        _check_gather_kv_indices(
            gather_kv_indices, cu_seqlens_q, cu_seqlens_k, int(num_queries)
        )
    except ValueError:
        return "torch"
    if int(gather_kv_indices.shape[1]) < 1 or any(
        cu.device != gather_kv_indices.device for cu in (cu_seqlens_q, cu_seqlens_k)
    ):
        return "torch"
    if not nope_index_prep_available(gather_kv_indices.device):
        return "torch"
    return "index_prep"


def nope_fwd_converts_indices() -> bool:
    """Whether the registered NoPE forward program converts per-document index rows itself: its launch argument
    list carries :data:`NOPE_FWD_CONVERT_ARGS`."""
    args = cake_launch.STAGE_ARGS.get(cake_jit.NOPE_FORWARD_STAGES[0], ())
    return all(name in args for name in NOPE_FWD_CONVERT_ARGS)


def _nope_arch(tensor: torch.Tensor) -> str:
    """Architecture tag of the NoPE record serving ``tensor``'s device; ``NotImplementedError`` for a host tensor or a
    device the record does not serve."""
    if not tensor.is_cuda:
        raise NotImplementedError(
            "the NoPE DSA sparse-attention training entries need CUDA tensors on a compute capability "
            "10.0 / 10.3 / 10.7 device"
        )
    record_for(tensor.device, cake_jit.NOPE_PROGRAM)
    return arch_for(tensor.device)


def _small_index_table(num_queries: int, topk: int) -> bool:
    """Whether a ``[num_queries, topk]`` index table is below :data:`NOPE_CONVERT_SLOT_LIMIT` slots."""
    return int(num_queries) * int(topk) < NOPE_CONVERT_SLOT_LIMIT


def nope_varlen_index_mode(
    num_queries: int, topk: int, *, caller_lengths: bool = False
) -> str:
    """The packed entry's index route for a ``[num_queries, topk]`` per-document index table, from its size alone
    (no device or registry access): ``"mask"`` -- the forward alone, in one launch, globalizes the rows
    position-preserving (the global key or ``-1`` at the slot it came from) into the rows the backward reads and
    derives each row's length as the last valid slot + 1 -- below :data:`NOPE_CONVERT_SLOT_LIMIT` slots without a
    caller ``topk_length``; ``"index_prep"`` -- one launch of the index-preparation stage, which compacts the rows
    (valid keys first, the count as the length), then the forward over its rows -- at or above the limit and always
    with caller lengths (the cut the stage applies before compacting; the one-launch forward takes no cut).  The
    compacting one-launch mode ``"kernel"`` is never selected by size: callers name it explicitly."""
    if caller_lengths:
        return "index_prep"
    return "mask" if _small_index_table(num_queries, topk) else "index_prep"


def nope_flat_index_mode(num_queries: int, topk: int) -> str:
    """The flat autograd entry's index route without caller lengths for a ``[num_queries, topk]`` table of global
    rows, the same size rule as :func:`nope_varlen_index_mode`: ``"mask"`` -- one launch of the forward over the
    caller's rows as given, deriving each row's length (the last valid slot + 1) into a fresh vector the backward
    reads with those rows -- below :data:`NOPE_CONVERT_SLOT_LIMIT` slots; ``"index_prep"`` -- the stage's flat mode
    compacts the rows into a fresh buffer with the valid counts, then the forward follows over them -- at or above
    it.  With caller lengths the flat entry launches the forward alone over the rows and lengths as given."""
    return "mask" if _small_index_table(num_queries, topk) else "index_prep"


def varlen_index_mode(
    gather_kv_indices: torch.Tensor, *, caller_lengths: bool = False
) -> str:
    """The route :func:`nope_varlen_index_mode` selects for ``gather_kv_indices`` (``[T, K_slots]`` on a served
    device), restricted to what the registered program offers: a program without the converting forward
    (:func:`nope_fwd_converts_indices`) takes ``"index_prep"`` at every size, one without the index-preparation
    stage takes ``"mask"`` at every size (never with caller lengths: only the stage applies a caller cut).  Raises
    ``NotImplementedError`` for a host tensor, a device the NoPE record does not serve, or a program with neither.
    The torch formulation :func:`offset_gather_kv_indices` is the tests' reference and the rope entry's fallback,
    not a runtime mode of the NoPE entries."""
    _nope_arch(gather_kv_indices)
    if gather_kv_indices.dim() != 2:
        raise ValueError("gather_kv_indices must be [T, K_slots]")
    num_rows, topk = (int(d) for d in gather_kv_indices.shape)
    preferred = nope_varlen_index_mode(num_rows, topk, caller_lengths=caller_lengths)
    converts = nope_fwd_converts_indices()
    record = cake_jit.record(cake_jit.NOPE_PROGRAM)
    if preferred == "mask" and converts:
        return "mask"
    if NOPE_INDEX_PREP_STAGE in record:
        return "index_prep"
    if converts and not caller_lengths:
        return "mask"
    if converts:
        raise NotImplementedError(
            "a caller-given topk_length with the converting NoPE forward needs the index-preparation stage "
            f"{NOPE_INDEX_PREP_STAGE!r}, which the registered program {cake_jit.NOPE_PROGRAM!r} does not carry"
        )
    raise NotImplementedError(
        f"the registered NoPE DSA training program {cake_jit.NOPE_PROGRAM!r} has neither the index-preparation "
        f"stage {NOPE_INDEX_PREP_STAGE!r} nor a forward that converts per-document index rows "
        f"(registered stages: {tuple(record.get('stages', ()))})"
    )


def _index_storage_alias(indices: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``(flat view, element offset, row stride)`` of an int32 ``[T, K_slots]`` index tensor for the
    index-preparation kernel, which addresses its index operand as flat storage (row ``t`` at
    ``indices_offset + t * idx_stride``): the stride-1 view of the storage span the rows occupy, from the
    vector-aligned element at or below the tensor's start (:data:`POINTER_GRANULE`), and the offset of row 0 within
    it.  Any row stride is read in place -- a column slice of a wider index buffer included -- and the view is
    contiguous by construction (the FFI boundary takes contiguous tensors)."""
    num_rows, topk = (int(d) for d in indices.shape)
    stride = int(indices.stride(0)) if num_rows > 1 else topk
    offset = indices.storage_offset() % POINTER_GRANULE
    span = offset + (num_rows - 1) * stride + topk
    view = indices.as_strided((span,), (1,), indices.storage_offset() - offset)
    return view, offset, stride


def _check_int32(name: str, value: int) -> int:
    if not -_INT32_MAX - 1 <= int(value) <= _INT32_MAX:
        raise ValueError(f"{name} = {value} does not fit the kernel's int32 argument")
    return int(value)


def _check_gather_kv_indices(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_rows: Optional[int] = None,
) -> None:
    """Structure of the packed entry's index operands: int32 ``[T, K_slots]`` rows (``T == num_rows`` when given)
    and contiguous int32 ``[num_docs + 1]`` prefix sums of one length."""
    if (
        gather_kv_indices.ndim != 2
        or gather_kv_indices.dtype != torch.int32
        or (num_rows is not None and int(gather_kv_indices.shape[0]) != num_rows)
    ):
        raise ValueError("gather_kv_indices must be an int32 [T, topk] tensor")
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.ndim != 1 or cu.dtype != torch.int32 or cu.numel() < 2:
            raise ValueError(f"{name} must be an int32 [num_docs + 1] tensor")
        if not cu.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must describe the same documents"
        )


def _check_index_rows(
    indices: torch.Tensor, name: str, topk_length: Optional[torch.Tensor] = None
) -> tuple[int, int]:
    """Structure of a global index operand the flat stage reads: int32 ``[T, K_slots]`` with contiguous rows (any
    row stride), ``K_slots >= 1``; an optional contiguous int32 ``[T]`` cut.  Returns ``(T, K_slots)``."""
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError(f"{name} must be an int32 [T, K_slots] tensor")
    num_rows, topk = (int(d) for d in indices.shape)
    if topk <= 0:
        raise ValueError(f"{name} must have at least one slot per row (K_slots >= 1)")
    if indices.stride(1) != 1 or (num_rows > 1 and indices.stride(0) < topk):
        raise ValueError(f"{name} rows must be contiguous (any row stride)")
    if topk_length is not None and (
        topk_length.shape != (num_rows,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be a contiguous int32 [T] tensor")
    return num_rows, topk


def _index_prep_launch(
    arch: str,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    *,
    causal: bool,
    out: torch.Tensor,
    topk_length: torch.Tensor,
    compact: bool = False,
    length_in: Optional[torch.Tensor] = None,
    flat: bool = False,
    num_kv: Optional[int] = None,
) -> _Launch:
    """The bound launch (not yet issued; the caller provides the FFI stream context) of one index-preparation
    stage call over the index rows ``gather_kv_indices`` (int32 ``[T, K_slots]``, ``T >= 1``, any row stride, on a
    device the stage serves): the global key rows (invalid slots ``-1``) go to the caller's contiguous int32
    ``out [T, K_slots]``, the row lengths to the caller's int32 ``topk_length [T]``.

    Modes.  Document mode (``flat=False``): row ``t`` belongs to the last document ``n`` with ``cu_seqlens_q[n] <=
    t``; a slot is valid iff ``0 <= idx <= bound`` with the query's own key position as the bound under ``causal``
    and ``seqlen_k - 1`` otherwise, and maps to ``idx + cu_seqlens_k[n]``.  Flat mode (``flat=True``, ``has_docs =
    0``): the rows are global already, a slot is valid iff ``0 <= idx <= num_kv - 1``, the prefix sums are never
    read (the per-device inert constant stands in for them; ``causal`` is ignored).  ``compact=False``: the
    position-preserving rewrite, ``topk_length`` = last valid slot + 1 (bitwise :func:`offset_gather_kv_indices`).
    ``compact=True``: stable compaction -- the valid globalized keys first in ascending slot order, then ``-1`` to
    the end of the row -- and ``topk_length`` = the valid count (bitwise ``offset_gather_kv_indices(...,
    compact=True)``); a valid-first row is a byte-identical fixed point.  ``length_in`` (contiguous int32 ``[T]``,
    with ``compact`` only) is a position cut applied before compaction: slots at positions ``>= length_in[t]`` are
    invalid; without it the kernel receives the ``topk_length`` output pointer as the placeholder it never reads
    (``has_length_in = 0``)."""
    num_rows, topk = (int(d) for d in gather_kv_indices.shape)
    alias, offset, stride = _index_storage_alias(gather_kv_indices)
    for name, value in (
        ("T", num_rows),
        ("K_slots", topk),
        ("the index row stride", stride),
    ):
        _check_int32(name, value)
    if length_in is not None and not compact:
        raise ValueError("length_in is a cut before compaction: it needs compact=True")
    if flat:
        if num_kv is None:
            raise ValueError("the flat index-preparation mode needs num_kv")
        inert = _inert_cu_seqlens(gather_kv_indices.device)
        cu_seqlens_q = cu_seqlens_k = inert
        num_docs, search_iters, causal_flag = 1, 1, 0
        has_docs, flat_bound = 0, _check_int32("S", int(num_kv)) - 1
    else:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(
                "the document index-preparation mode needs both prefix sums"
            )
        num_docs = int(cu_seqlens_q.numel()) - 1
        search_iters, causal_flag = index_prep_search_iters(num_docs), int(bool(causal))
        has_docs, flat_bound = 1, 0
    values = dict(
        indices=alias,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        out=out,
        topk_length=topk_length,
        num_rows=num_rows,
        topk=topk,
        idx_stride=stride,
        indices_offset=offset,
        num_docs=num_docs,
        search_iters=search_iters,
        causal=causal_flag,
        has_docs=has_docs,
        flat_bound=flat_bound,
        write_out=1,
        compact=int(bool(compact)),
        length_in=length_in if length_in is not None else topk_length,
        has_length_in=int(length_in is not None),
    )
    # the grid is the stage's own rule (one CTA per four rows), given explicitly
    entry = _stage_entry(
        NOPE_INDEX_PREP_STAGE,
        arch,
        dict(num_queries=num_rows, num_kv=0, topk=topk),
        cake_jit.NOPE_PROGRAM,
        index_prep_grid(num_rows),
    )
    return _Launch(NOPE_INDEX_PREP_STAGE, entry.launcher, entry.run, values, entry.grid)


def _launch_index_prep(
    arch: str,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    *,
    causal: bool,
    out: torch.Tensor,
    topk_length: torch.Tensor,
    compact: bool = False,
    length_in: Optional[torch.Tensor] = None,
    flat: bool = False,
    num_kv: Optional[int] = None,
) -> None:
    """:func:`_index_prep_launch` issued inside the FFI stream context of the index rows' device."""
    launch = _index_prep_launch(
        arch,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        causal=causal,
        out=out,
        topk_length=topk_length,
        compact=compact,
        length_in=length_in,
        flat=flat,
        num_kv=num_kv,
    )
    with _ffi_stream_context(_device_index(gather_kv_indices.device)):
        launch()


def _require_index_prep_stage() -> None:
    if NOPE_INDEX_PREP_STAGE not in cake_jit.record(cake_jit.NOPE_PROGRAM):
        raise NotImplementedError(
            f"the registered NoPE DSA training program {cake_jit.NOPE_PROGRAM!r} has no "
            f"index-preparation stage {NOPE_INDEX_PREP_STAGE!r}"
        )


def nope_globalize_indices(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = False,
    compact: bool = False,
    topk_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices, topk_length)`` of per-document index rows in one launch of the index-preparation stage: the
    global key rows of ``gather_kv_indices`` (int32 ``[T, K_slots]``, any row stride; invalid slots ``-1``) under
    the rule of :func:`offset_gather_kv_indices` -- the query segment is the tail of its key prefix, ``causal``
    drops keys after the query's own position -- and the per-row ``topk_length``.  Pure integer work.  By default
    (``compact=False``) the rows keep their slot positions and the length is the last valid slot + 1: bitwise what
    ``offset_gather_kv_indices(..., causal=causal, return_topk_length=True)`` returns.  ``compact=True`` compacts
    every row stably (valid keys first in ascending slot order, then ``-1``) and returns the valid counts, after the
    position cut of an optional caller ``topk_length`` (contiguous int32 ``[T]``; slots at positions ``>=
    topk_length[t]`` are invalid): bitwise ``offset_gather_kv_indices(..., causal=causal, compact=True,
    topk_length=topk_length, return_topk_length=True)``.  The public varlen entries take the compacting form.
    ``T == 0`` returns empty results without a launch; otherwise the device must be served by the NoPE record with
    the stage (``NotImplementedError``).  The prefix sums are checked for structure here (contiguous int32
    ``[num_docs + 1]``) and for their values by :func:`validate_cu_seqlens`; a violation that passes is not
    detected on device (a row past ``cu_seqlens_q[-1]`` is assigned to the last document).
    """
    _check_gather_kv_indices(gather_kv_indices, cu_seqlens_q, cu_seqlens_k)
    num_rows, topk = (int(d) for d in gather_kv_indices.shape)
    if topk_length is not None and not compact:
        raise ValueError(
            "topk_length is a cut before compaction: it needs compact=True"
        )
    if topk_length is not None and (
        topk_length.shape != (num_rows,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be a contiguous int32 [T] tensor")
    device = gather_kv_indices.device
    if num_rows == 0:
        return _alloc((0, topk), torch.int32, device), _alloc((0,), torch.int32, device)
    arch = _nope_arch(gather_kv_indices)
    _require_index_prep_stage()
    out = _alloc((num_rows, topk), torch.int32, device)
    lengths = _alloc((num_rows,), torch.int32, device)
    _launch_index_prep(
        arch,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        causal=causal,
        out=out,
        topk_length=lengths,
        compact=compact,
        length_in=topk_length,
    )
    return out, lengths


def nope_compact_indices(
    indices: torch.Tensor,
    num_kv: int,
    *,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    counts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, counts)``: global index rows stably compacted in one launch of the index-preparation stage's flat
    mode -- the recipe for the kernel-level entries (:func:`forward_nope` / :func:`backward_nope`,
    :func:`prepare_dsa_nope_train`) and the flat autograd entry with caller lengths, which consume rows as given.

    ``indices``: int32 ``[T, K_slots]`` global key rows (any row stride, read in place; ``-1`` or ``>= num_kv``
    invalid).  Row ``t`` of ``rows`` (contiguous int32 ``[T, K_slots]``) holds the valid keys of the input row in
    ascending slot order, then ``-1``; ``counts[t]`` (int32 ``[T]``) is their number, the ``topk_length`` to pass
    with the rows.  A caller ``topk_length`` (contiguous int32 ``[T]``) is a position cut applied first: slots at
    positions ``>= topk_length[t]`` are invalid.  Valid-first rows come back byte-identical with ``counts`` = last
    valid slot + 1.  ``out`` / ``counts`` receive the results in place when given (contiguous, the right shape and
    dtype); otherwise fresh tensors are allocated before the launch.  Nothing synchronizes with the device and the
    launch is a plain stage launch, so a capturing caller passes its own buffers.  ``T == 0`` returns (the) empty
    results without a launch; otherwise the device must be served by the NoPE record with the stage
    (``NotImplementedError``).  Bitwise ``offset_gather_kv_indices(indices, [0, T], [0, num_kv], compact=True,
    topk_length=topk_length, return_topk_length=True)``.
    """
    num_rows, topk = _check_index_rows(indices, "indices", topk_length)
    if int(num_kv) < 1:
        raise ValueError("num_kv must hold at least one key row (S > 0)")
    _check_output(out, "out", (num_rows, topk), torch.int32)
    _check_output(counts, "counts", (num_rows,), torch.int32)
    device = indices.device
    if out is None:
        out = _alloc((num_rows, topk), torch.int32, device)
    if counts is None:
        counts = _alloc((num_rows,), torch.int32, device)
    if num_rows == 0:
        return out, counts
    arch = _nope_arch(indices)
    _require_index_prep_stage()
    _launch_index_prep(
        arch,
        indices,
        None,
        None,
        causal=False,
        out=out,
        topk_length=counts,
        compact=True,
        length_in=topk_length,
        flat=True,
        num_kv=int(num_kv),
    )
    return out, counts


def _packed_forward_nope(
    q_latent,
    kv_latent,
    gather_kv_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    causal,
    topk_length,
    softmax_scale,
    mode,
):
    """The packed forward over per-document index rows, ``(out, lse, o_lo, indices, lengths)``: the plan, the
    outputs and the launch values are settled before the first launch.  ``mode == "index_prep"`` issues the
    preparation launch (``compact = 1``: every row stably compacted -- its valid globalized keys first in ascending
    slot order, then ``-1`` -- after the position cut of a caller ``topk_length`` passed as ``length_in``) into the
    fresh ``indices`` / ``lengths`` (the valid counts) and then the forward from pre-converted operands
    (:meth:`_NopeBinding.prebound_forward`) with nothing else in between; the forward reads the stage's rows and
    counts whether or not the caller gave lengths.  ``mode == "mask"`` lets the forward convert the rows itself in
    its single launch (``convert = NOPE_CONVERT_MASK``): its scan lanes write the position-preserving global rows
    to ``indices`` -- the global key or ``-1`` at the slot it came from, so a hole stays a masked slot -- and the
    last valid slot + 1 to ``lengths``; no caller cut.  ``mode == "kernel"`` is the compacting single launch
    (``convert = NOPE_CONVERT_COMPACT``): the scan lanes write the same stable compaction and valid counts as the
    stage (bitwise); no caller cut, never selected by size.  ``indices`` (global key rows in the order the forward
    processed them) and ``lengths`` are what the backward reads on every route."""
    arch = _nope_arch(gather_kv_indices)
    num_rows, topk = int(q_latent.shape[0]), int(gather_kv_indices.shape[1])
    device = q_latent.device
    scale = float(softmax_scale)
    indices = _alloc(
        (num_rows, topk), torch.int32, device
    )  # the global key rows: the backward's operand
    derived = _alloc(
        (num_rows,), torch.int32, device
    )  # the row lengths the conversion writes (valid counts, or last valid + 1 in mask mode)
    if mode in ("mask", "kernel"):
        # the forward reads the per-document rows (any row stride) and writes the global rows and the row
        # lengths itself (a non-zero ``convert`` derives the lengths: the entry selects these modes without
        # caller lengths)
        if topk_length is not None:
            raise ValueError(
                "the converting NoPE forward derives the row lengths itself: no caller topk_length in "
                f"{mode} mode"
            )
        num_docs = int(cu_seqlens_q.numel()) - 1
        plan_indices, derive = gather_kv_indices, True
        extra = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            indices_out=indices,
            num_docs=num_docs,
            search_iters=index_prep_search_iters(num_docs),
            causal=int(bool(causal)),
            convert=NOPE_CONVERT_MASK if mode == "mask" else NOPE_CONVERT_COMPACT,
        )
    elif mode == "index_prep":
        plan_indices, derive, extra = indices, False, None
        if topk_length is not None and (
            topk_length.shape != (num_rows,)
            or topk_length.dtype != torch.int32
            or not topk_length.is_contiguous()
        ):
            raise ValueError("topk_length must be a contiguous int32 [T] tensor")
    else:
        raise ValueError(
            f"unknown NoPE index mode {mode!r}: expected 'mask', 'index_prep' or 'kernel'"
        )
    lengths = derived
    key = forward_binding_key_nope(
        q_latent, kv_latent, plan_indices, lengths, scale, derive
    )
    binding = _binding(
        key,
        lambda: _plan_nope(
            q_latent,
            kv_latent,
            plan_indices,
            topk_length=lengths,
            dout=None,
            softmax_scale=scale,
            outputs={},
            backward=False,
            key_passes=None,
            derive_topk_length=derive,
        ),
        _NopeBinding,
    )
    if extra is not None:  # the one-launch modes
        (out, lse, o_lo), launch = binding.prepared_forward(
            q_latent, kv_latent, plan_indices, lengths, extra
        )
        launch()
        return out, lse, o_lo, indices, lengths
    # the preparation route: every allocation, launch value and operand conversion of both launches first, then
    # -- inside one stream context -- the stage launch and the pre-bound forward launch with nothing in between
    # (the forward's binding call is all that separates them, so it hides behind the stage's kernel)
    (out, lse, o_lo), launches = binding.prebound_forward(
        q_latent, kv_latent, indices, lengths
    )
    prep = _index_prep_launch(
        arch,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        causal=causal,
        out=indices,
        topk_length=derived,
        compact=True,
        length_in=topk_length,
    )
    with _ffi_stream_context(_device_index(device)):
        prep()
        for launch in launches:
            launch()
    return out, lse, o_lo, indices, lengths


def _flat_forward_nope(q_latent, kv_latent, indices, softmax_scale):
    """The flat autograd entry's forward without caller lengths, ``(out, lse, o_lo, indices_c, counts)``: the
    stage branch of :func:`_packed_forward_nope` in the stage's flat mode -- one launch compacts the caller's global
    rows (any row stride, read in place) stably into the fresh contiguous ``indices_c [T, K_slots]`` with the valid
    counts in ``counts [T]``, then the pre-bound forward follows over them with nothing in between; the backward
    reads the same two tensors.  Rows with ``-1`` holes therefore cost their valid slots in both kernels; valid-first
    rows are byte-identical to the input with ``counts`` = last valid slot + 1.  Both buffers are allocated before
    the first launch and nothing synchronizes with the device.  The forward plan is that of a kernel-level call over
    contiguous rows with lengths (``has_topk_length = 1, derive_length = 0``): the same geometry shares the binding,
    since neither the key nor the plan depends on tensor contents."""
    arch = _nope_arch(indices)
    _require_index_prep_stage()
    num_rows, topk = _check_index_rows(indices, "indices")
    if num_rows != int(q_latent.shape[0]):
        raise ValueError("indices must be an int32 [T, K_slots] tensor")
    device = q_latent.device
    scale = float(softmax_scale)
    indices_c = _alloc((num_rows, topk), torch.int32, device)
    counts = _alloc((num_rows,), torch.int32, device)
    key = forward_binding_key_nope(q_latent, kv_latent, indices_c, counts, scale, False)
    binding = _binding(
        key,
        lambda: _plan_nope(
            q_latent,
            kv_latent,
            indices_c,
            topk_length=counts,
            dout=None,
            softmax_scale=scale,
            outputs={},
            backward=False,
            key_passes=None,
            derive_topk_length=False,
        ),
        _NopeBinding,
    )
    (out, lse, o_lo), launches = binding.prebound_forward(
        q_latent, kv_latent, indices_c, counts
    )
    prep = _index_prep_launch(
        arch,
        indices,
        None,
        None,
        causal=False,
        out=indices_c,
        topk_length=counts,
        compact=True,
        flat=True,
        num_kv=int(kv_latent.shape[0]),
    )
    with _ffi_stream_context(_device_index(device)):
        prep()
        for launch in launches:
            launch()
    return out, lse, o_lo, indices_c, counts


class DSASparseAttentionNopeVarlenFunction(torch.autograd.Function):
    """Autograd wrapper of the packed NoPE form with on-device index preparation.

    Fifteen forward inputs ``(q_latent, kv_latent, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, causal,
    topk_length, softmax_scale, key_passes, dkv_acc, dkv_dst_map, num_segments, dkv_direct, check_dst_map,
    mode)``; returns ``(out, lse)``.  The forward turns the per-document rows into global key rows on device --
    ``mode == "mask"`` (below :data:`NOPE_CONVERT_SLOT_LIMIT` slots without caller lengths): the forward kernel
    alone, in one launch, writes the position-preserving global rows (the global key or ``-1`` at the slot it
    came from) to a fresh ``indices [T, K_slots]`` and the last valid slot + 1 to the lengths; ``mode ==
    "index_prep"``: one launch of the index-preparation stage into the fresh ``indices`` and the row lengths,
    every row stably compacted (valid keys first, then ``-1``; the length is the valid count) after the position
    cut of a caller ``topk_length``, then the forward kernel with no host work in between; ``mode == "kernel"``
    (explicit only): the forward kernel writes the same compacted rows and counts as the stage itself.  On every
    route the plan lookup, the output allocation and the forward's launch values are prepared before the first
    launch, and nothing on the path synchronizes with the device.  The backward is that of
    :class:`DSASparseAttentionNopeFunction` over the saved global rows and lengths -- the rows in the order the
    forward processed them, so forward and backward see the same slots (the stage's rows and counts on the
    ``"index_prep"`` route, also with caller lengths): every produced
    operand of the backward goes through ``ctx.save_for_backward`` and a tensor ``dkv_acc`` stays caller-owned on
    the node, so a checkpointed forward retains nothing; ``dkv_acc`` may be a callable resolved at backward time
    (:data:`DkvAccSource`).
    """

    @staticmethod
    def forward(
        ctx,
        q_latent,
        kv_latent,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        causal,
        topk_length,
        softmax_scale,
        key_passes=None,
        dkv_acc=None,
        dkv_dst_map=None,
        num_segments=1,
        dkv_direct=None,
        check_dst_map=False,
        mode="index_prep",
    ):
        out, lse, o_lo, indices, lengths = _packed_forward_nope(
            q_latent,
            kv_latent,
            gather_kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            causal,
            topk_length,
            softmax_scale,
            mode,
        )
        scale = float(softmax_scale)
        ctx.set_materialize_grads(False)
        ctx.nope_prep = _NopeBackwardPrep(
            ctx,
            q_latent,
            kv_latent,
            indices,
            out,
            o_lo,
            lse,
            lengths,
            scale,
            key_passes,
            num_segments,
            dkv_acc,
            dkv_dst_map,
            dkv_direct,
            check_dst_map,
        )
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        if dlse is not None:
            raise NotImplementedError(
                "gradients through lse are not supported; only out is differentiable"
            )
        if dout is None:  # out unused downstream
            return (None,) * 15
        return _nope_backward_from_ctx(ctx, dout) + (None,) * 13


def dsa_sparse_attention_nope_varlen(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
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
    dkv_acc: Optional[DkvAccSource] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: bool = False,
    check_cu_seqlens: bool = False,
):
    """Packed multi-document NoPE form: per-document key indices are turned into global key rows on device and
    the flat kernels run over the packed rows.  The route follows the table size (:func:`nope_varlen_index_mode`).
    Below :data:`NOPE_CONVERT_SLOT_LIMIT` slots, without caller lengths, the forward kernel alone converts the
    rows in its single launch, position-preserving: the global key or ``-1`` at the slot it came from -- bitwise
    ``offset_gather_kv_indices(...)`` -- and each row's length is its last valid slot + 1, so a ``-1`` before the
    last valid key stays a masked slot the kernels process (small tables pay that for their holes).  At or above
    the limit, and always with a caller ``topk_length``, one launch of the record's index-preparation stage runs
    right before the forward kernel and compacts each row stably: its valid keys come first in ascending slot
    order, then ``-1``, and the row's length is its valid count -- bitwise ``offset_gather_kv_indices(...,
    compact=True)`` -- so rows with ``-1`` holes cost their valid slots in the forward and the backward.  Rows whose
    valid slots already come first give the same bytes on both routes.  A caller ``topk_length`` is a position cut
    applied before compaction (slots at positions ``>= topk_length[t]`` are invalid regardless of content); the
    kernels then read the compacted rows and their counts.  Query and key
    segment lengths may differ (the query segment is the tail of its key prefix); ``causal=True`` also drops
    selected keys after the query's own position ``(seqlen_k - seqlen_q) + local_q``.  ``max_seqlen_q/k`` are
    accepted for signature parity and not used on the host.  The segment count ``len(cu_seqlens_k) - 1`` is passed
    on as ``num_segments``.  Every call checks the structure of the prefix sums (contiguous int32
    ``[num_docs + 1]`` of one length) and nothing on the path synchronizes with the device; ``check_cu_seqlens=True``
    also checks their values (:func:`validate_cu_seqlens`: start at 0, non-decreasing, ``cu_seqlens_q[-1] == T``,
    ``cu_seqlens_k[-1] == S``, ``seqlen_q <= seqlen_k`` per document) at the cost of one blocking device-to-host
    copy before any launch (skipped during CUDA-graph capture); the default leaves those invariants to the caller,
    as a violation is otherwise not detected on device.  Other arguments as in :func:`dsa_sparse_attention_nope`.
    """
    del max_seqlen_q, max_seqlen_k
    kv_latent = _kv_latent_rows(kv_latent)
    num_queries = int(q_latent.shape[0])
    validate_cu_seqlens(
        cu_seqlens_q,
        cu_seqlens_k,
        num_queries,
        int(kv_latent.shape[0]),
        values=check_cu_seqlens,
    )
    _check_gather_kv_indices(gather_kv_indices, cu_seqlens_q, cu_seqlens_k, num_queries)
    num_segments = int(cu_seqlens_k.numel()) - 1
    if softmax_scale is None:
        softmax_scale = NOPE_SOFTMAX_SCALE
    if (
        num_queries == 0
    ):  # no rows: nothing to offset, nothing launched downstream either
        return dsa_sparse_attention_nope(
            q_latent,
            kv_latent,
            gather_kv_indices,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
            key_passes=key_passes,
            dkv_acc=dkv_acc,
            dkv_dst_map=dkv_dst_map,
            num_segments=num_segments,
            dkv_direct=dkv_direct,
            check_dst_map=check_dst_map,
        )
    mode = varlen_index_mode(gather_kv_indices, caller_lengths=topk_length is not None)
    out, lse = DSASparseAttentionNopeVarlenFunction.apply(
        q_latent,
        kv_latent,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        bool(causal),
        topk_length,
        float(softmax_scale),
        key_passes,
        dkv_acc,
        dkv_dst_map,
        num_segments,
        dkv_direct,
        check_dst_map,
        mode,
    )
    return (out, lse) if return_lse else out


def forward_nope_varlen(
    q_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    check_cu_seqlens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kernel-level packed NoPE forward without autograd: the validation and the index route of
    :func:`dsa_sparse_attention_nope_varlen` (:func:`nope_varlen_index_mode`), returning
    ``(out, lse, o_lo, indices, topk_length)`` -- the forward outputs plus the global key rows and row lengths the
    forward read: below :data:`NOPE_CONVERT_SLOT_LIMIT` slots without caller lengths the position-preserving rows
    (the global key or ``-1`` at the slot it came from) and the last valid slot + 1 per row, written by the
    forward itself; on the stage route the stably compacted rows (valid keys first, then ``-1``) and their valid
    counts, after the cut of a caller ``topk_length``; the pair the kernel-level backward consumes::

        dq, _ = backward_nope(q_latent, kv_latent, indices, out, o_lo, lse, dout, topk_length=topk_length,
                              softmax_scale=softmax_scale, dkv_acc=dkv_acc,
                              num_segments=cu_seqlens_k.numel() - 1)

    with ``dkv_acc`` an FP32 ``[S, 512]`` buffer the caller allocates and zero-fills (or ``dkv_acc=None`` for a fresh
    ``dkv``).  A call without query rows returns empty outputs and the (empty) rows without launching."""
    kv_latent = _kv_latent_rows(kv_latent)
    num_queries = int(q_latent.shape[0])
    validate_cu_seqlens(
        cu_seqlens_q,
        cu_seqlens_k,
        num_queries,
        int(kv_latent.shape[0]),
        values=check_cu_seqlens,
    )
    _check_gather_kv_indices(gather_kv_indices, cu_seqlens_q, cu_seqlens_k, num_queries)
    scale = NOPE_SOFTMAX_SCALE if softmax_scale is None else float(softmax_scale)
    if num_queries == 0:
        out, lse, o_lo = forward_nope(
            q_latent,
            kv_latent,
            gather_kv_indices,
            topk_length=topk_length,
            softmax_scale=scale,
        )
        lengths = (
            topk_length
            if topk_length is not None
            else gather_kv_indices.new_empty((0,))
        )
        return out, lse, o_lo, gather_kv_indices, lengths
    mode = varlen_index_mode(gather_kv_indices, caller_lengths=topk_length is not None)
    return _packed_forward_nope(
        q_latent,
        kv_latent,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        bool(causal),
        topk_length,
        scale,
        mode,
    )
