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

Cake backend: native 64-query-head DeepSeek Sparse Attention (top-k sparse
MLA with absorbed queries) forward and backward for training on SM100 / SM103
(flashinfer-ai/flashinfer#5657; GLM-5.2 geometry).

Contract
--------
* ``q_latent [T, 64, 512]`` and ``q_rope [T, 64, 64]`` BF16 -- may be views of
  a packed ``q [T, 64, 576]`` (row stride 576); ``kv_latent [S, 512]`` (K = V)
  and ``k_rope [S, 64]`` BF16 -- may be views of a packed ``kv [S, 576]``.
* ``indices [T, topk]`` int32: **global** key rows into ``kv_*``; ``-1`` or
  ``>= S`` marks an invalid slot, anywhere in the row.  ``topk_length [T]``
  int32 (optional) invalidates slots ``>= topk_length[t]``.  ``topk`` is any
  positive integer.
* ``softmax_scale`` defaults to ``576 ** -0.5``.
* Forward writes ``out [T, 64, 512]`` BF16, ``lse [T, 64]`` FP32 = natural-log
  logsumexp over the valid keys, and ``o_lo [T, 64, 512]`` BF16 =
  ``bf16(fp32(O) - bf16(O))``, the output residual the backward uses to form
  ``delta = rowsum(dO * (O + O_lo))`` (FA-style exact delta).  Fully masked
  rows give ``out = 0``, ``lse = -inf``, ``o_lo = 0``.
* Backward returns ``dq_latent [T, 64, 512]`` and ``dq_rope [T, 64, 64]`` BF16
  (computed once per row: no atomics, bitwise deterministic) and
  ``dkv_latent [S, 512]`` / ``dk_rope [S, 64]`` BF16 (FP32 ``red.global``
  accumulation then a cast).  ``dkv_fp32=True`` returns natural-layout FP32
  gradients instead: the accumulators themselves when the program accumulates
  in natural row-major layout, or the cast's FP32 outputs when it accumulates
  in an internal permuted layout (registry field ``dkv_acc_layout``); the raw
  accumulators of a permuted program never cross the API boundary.
* BF16 operands into the MMAs (Q, K, V, P, dS), FP32 accumulation; S is
  recomputed in the backward from the BF16 Q and K.

Every allocation happens in :func:`prepare_dsa_train`; the returned runner
launches with no CUDA allocation and no host synchronization, so a runner
(or a CUDA graph capturing it) replays for new values written into the bound
tensors.  Kernels are reached through the argument plans of the registry
records in ``cake_jit.MODULES``; the ``abi`` field of a record names the
keyword set its kernels expect (see :data:`ABI_CONTRACT`).

The eager entry points (:func:`forward`, :func:`backward`, the autograd
``Function`` behind ``dsa_sparse_attention``) validate and bind once per
*input binding* and remember the result in :data:`BINDING_CACHE` (see
:class:`BindingCache`): a later call whose inputs have the same
``(data_ptr, shape, stride, dtype)`` and scale slots the current tensors,
freshly allocated outputs and per-call scratch into the remembered argument
plans and launches.  A call without query rows (``T == 0``) returns empty
outputs and zero gradients from the eager entry points without binding or
launching anything.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    FORWARD_STAGES,
    MODULES,
    load_cake_dsa_train_module,
    registered_stages,
    select_module,
)

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

# Host binding profiles.  ``ABI_CONTRACT`` is the keyword set of the native
# training kernels (the names below are what the host provides; a kernel's
# argument plan selects from them).  ``ABI_SEED`` is the profile of the
# forward-only FlashMLA-derived prefill program that stands in while the
# native kernels are being written: it needs packed ``q``/``kv`` operands, a
# top-k that is a multiple of 64, produces no output residual and reports
# ``+inf`` LSE for empty rows (rewritten to ``-inf`` here).
ABI_CONTRACT = "dsa_h64_v1"
ABI_SEED = "flashmla_v41_prefill_seed"
SUPPORTED_ABIS = (ABI_CONTRACT, ABI_SEED)

# Keyword names the contract profile offers to every stage.  Tensor kinds in
# the argument plan are ``buffer`` / ``tma_buffer`` (the binding encodes the
# tensor map itself), scalars are ``parameter``; ``grid_x/y/z`` are ``grid``.
CONTRACT_TENSORS = (
    "q_latent",
    "q_rope",
    "kv_latent",
    "k_rope",
    "indices",
    "topk_length",
    "out",
    "o_lo",
    "lse",
    "dout",
    "delta",
    "dq_latent",
    "dq_rope",
    "dkv_latent_acc",
    "dk_rope_acc",
    "dkv_latent",
    "dk_rope",
    "dkv_latent_fp32",  # bwd_cast: natural-layout FP32 outputs of the dkv_fp32 mode (out_f32 = 1); otherwise a placeholder
    "dk_rope_fp32",
    # bwd_cast packed-accumulate mode (accumulate = 1; flashinfer-ai/flashinfer#5675): the caller's FP32 [S_dst, >= 576]
    # rows the un-permuted dK/dV gradients are ADDED into (``dst_packed``: latent columns 0:512, rope columns 512:576) and
    # the optional int32 [S] destination-row map (``dst_map``, ``has_dst_map``).  Inert placeholders of the right dtypes
    # otherwise (_inert_cast_values): the FP32 accumulator and the indices storage.
    "dkv_acc",
    "dkv_dst_map",
    # Key-range passes (bwd_compact / bwd_main_pass): the FP32 dQ partials, the per-pass compacted keys and their
    # counts, carved from the workspace when the plan has more than one pass.  The single-pass kernel (bwd_main) never
    # reads them and receives, like its production launcher, ``delta`` and the indices storage as inert placeholders
    # (see _inert_pass_values).
    "dq_partial",
    "key_scratch",
    "pass_counts",
    "workspace",  # kernel-private scratch of the backward main stage (record field workspace_bytes)
    "tma_descriptor_workspace",
)
CONTRACT_SCALARS = (
    "num_queries",
    "num_kv",
    "topk",
    "softmax_scale",
    "softmax_scale_log2",
    "idx_stride",  # indices row stride (elements)
    "indices_offset",  # element offset of indices inside the storage alias the host passes (0 when the tensor starts its storage)
    "k_rope_stride",  # k_rope row stride (elements); k_rope is passed as its storage alias
    "k_rope_offset",  # element offset of k_rope inside that storage (0 when the tensor starts its storage)
    "has_topk_length",  # 1 when the caller supplied topk_length
    "q_latent_row_stride",
    "q_rope_row_stride",
    "kv_latent_row_stride",
    "num_rows",  # bwd_delta: num_queries * 64 (token, head) rows
    "latent_vecs",  # bwd_cast: num_kv * 512 / 16 sixteen-element vectors
    "rope_vecs",  # bwd_cast: num_kv * 64 / 16
    "latent_groups",  # bwd_cast: num_kv * 512 / 32 permuted 32-element groups
    "rope_groups",  # bwd_cast: num_kv * 64 / 32
    "out_f32",  # bwd_cast: 1 when the cast writes natural-layout FP32 outputs (dkv_fp32 mode of a permuted program)
    "token_base",  # bwd_main: token = token_base + token_step * blockIdx.x (0, 1: one CTA per token in row order)
    "token_step",
    # bwd_main / bwd_main_pass: key-range-pass controls.  The single-pass kernel is launched with the whole key range and
    # dq_mode 0 (direct BF16 dQ output) and does not read them; the pass stages receive the pass's key range [pass_lo,
    # pass_hi) and dq_mode 1 (first pass: store the FP32 dQ partial) / 2 (middle: load-add-store) / 3 (last: load-add,
    # BF16 output).
    "pass_lo",
    "pass_hi",
    "dq_mode",
    "num_tokens",  # bwd_compact: tokens of the launch (= num_queries: passes run over the whole row)
    "dst_row_stride",  # bwd_cast: elements per dkv_acc row (>= 576, a multiple of 4); 0 when accumulate = 0
    "has_dst_map",  # bwd_cast: 1 when dkv_dst_map is given
    "accumulate",  # bwd_cast: 1 = add the gradients into dkv_acc (no BF16 / FP32 dK/dV outputs); 0 = fresh natural outputs
)
# FP32 dK/dV accumulator layouts a backward record declares (``dkv_acc_layout``).
DKV_ACC_LAYOUTS = ("natural", "permuted")
# Key-range passes of the backward (the DRAM regime of the main stage).  A
# program that registers ``bwd_compact`` and ``bwd_main_pass`` can run the
# backward of every token in P passes over disjoint key ranges: per pass the
# compaction stage writes the token's keys inside the range (slot order; the
# validity rules of the main kernel) to ``key_scratch`` and their number to
# ``pass_counts``, and the pass variant of the main stage consumes them,
# carrying the FP32 dQ / dQ_rope partial of every token in ``dq_partial``
# between passes (dq_mode 1 store, 2 load-add-store, 3 load-add and BF16
# output) -- ``dq`` stays bitwise deterministic; the dK/dV reductions are
# unchanged (a pass only selects which keys a tile carries).  The pass count
# follows the policy the record carries (``key_pass_policy``): P = ceil(S *
# key_bytes / l2_budget_bytes), the FP32 accumulator slice one pass touches
# fitting the L2, when P > 1 AND the whole row runs as one launch
# (num_queries <= the token chunk the workspace budget allows); otherwise the
# single-pass stage.  Passes run over the whole row only (grid = num_queries
# per pass; no token chunking), so the workspace grows by num_queries *
# (147,456 + 4 * topk + 4) bytes.  The passes split the whole key row
# ``[0, S)`` into equal ranges, which matches an index row whose keys spread
# over the whole row (one document); in a packed multi-segment row every
# token's keys lie inside its own segment, so whole-row ranges leave most
# passes empty for most tokens while the per-pass fixed cost (Q/dO reload,
# FP32 dQ partial round trip, compaction) is still paid.  The policy therefore
# applies to one-segment rows only: ``num_segments > 1`` (the varlen entry
# passes ``len(cu_seqlens_k) - 1``, host metadata, no device sync) plans one
# pass unless ``key_passes`` overrides.
KEY_PASS_STAGES = ("bwd_compact", "bwd_main_pass")
DQ_PARTIAL_BYTES_PER_TOKEN = (
    NUM_HEADS * D_QK * 4
)  # one FP32 dQ / dQ_rope partial row (147,456 B)
KEY_PASS_POLICY_FIELDS = (
    "l2_budget_bytes",
    "key_bytes",
    "workspace_budget_bytes",
    "token_chunk_multiple",
)


@dataclass(frozen=True)
class KeyPassPolicy:
    """The record's key-range-pass policy (see the comment above)."""

    l2_budget_bytes: int  # FP32 accumulator bytes one pass may touch (100 MiB)
    key_bytes: int  # FP32 accumulator bytes per key row (576 * 4 = 2304)
    workspace_budget_bytes: int  # pass scratch the whole row may need at most (640 MiB)
    token_chunk_multiple: (
        int  # the token chunk is a multiple of this and at least this (128)
    )

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
        return cls(**values)

    def formula_passes(self, num_kv: int) -> int:
        return max(1, -(-int(num_kv) * self.key_bytes // self.l2_budget_bytes))

    def token_chunk(self, topk: int) -> int:
        """Tokens whose pass scratch (dQ partial, key scratch, pass count) fits the workspace budget."""
        per_token = DQ_PARTIAL_BYTES_PER_TOKEN + 4 * int(topk) + 4
        m = self.token_chunk_multiple
        return max(m, (self.workspace_budget_bytes // per_token) // m * m)

    def passes(
        self, num_queries: int, num_kv: int, topk: int, *, num_segments: int = 1
    ) -> int:
        """Passes of a binding: the formula when the key row is one segment, the
        whole row is one launch and the formula gives more than one; else 1."""
        if int(num_segments) < 1:
            raise ValueError(f"num_segments must be >= 1, got {num_segments}")
        if int(num_segments) > 1:
            return 1
        formula = self.formula_passes(num_kv)
        if formula == 1:
            return 1
        return formula if int(num_queries) <= self.token_chunk(topk) else 1


def plan_key_passes(
    record: dict[str, Any],
    stages: tuple[str, ...],
    num_queries: int,
    num_kv: int,
    topk: int,
    key_passes: Optional[int] = None,
    *,
    num_segments: int = 1,
) -> int:
    """Number of key-range passes of one backward binding.

    ``key_passes`` overrides the record's policy (1 = the single-pass stage);
    a program without the pass stages serves one pass only, and a record
    without a ``key_pass_policy`` never takes the pass path by default.
    ``num_segments`` is the packed segment count of the key row (see the
    comment above ``KEY_PASS_STAGES``): the policy plans one pass for more
    than one segment.
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
    return policy.passes(num_queries, num_kv, topk, num_segments=num_segments)


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


# Accepted spellings of the same host value (kernel side -> host side).
CONTRACT_ALIASES = {
    "sm_scale": "softmax_scale",
    "scale": "softmax_scale",
    "scale_log2": "softmax_scale_log2",
    "sm_scale_log2": "softmax_scale_log2",
    "total_q": "num_queries",
    "total_kv": "num_kv",
    "num_keys": "num_kv",
    "seqlen_kv": "num_kv",
    "lengths": "topk_length",
    "indices_stride": "idx_stride",
    "k_rope_row_stride": "k_rope_stride",
    "dkv_latent_f32": "dkv_latent_acc",
    "dk_rope_f32": "dk_rope_acc",
    "dkv_f32": "dkv_latent_acc",
    "dkr_f32": "dk_rope_acc",
    "src_latent": "dkv_latent_acc",
    "src_rope": "dk_rope_acc",
    "dst_latent": "dkv_latent",
    "dst_rope": "dk_rope",
    "dst_latent_f32": "dkv_latent_fp32",
    "dst_rope_f32": "dk_rope_fp32",
    "dst_packed": "dkv_acc",
    "dst_map": "dkv_dst_map",
    "do": "dout",
    "d_out": "dout",
}


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def default_softmax_scale() -> float:
    return D_QK**-0.5


def record_for(device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    """``(module_name, record)`` registered for ``device``; raises when absent."""
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "DSA sparse-attention training requires compute capability 10.0, 10.3 or 10.7"
        )
    name = select_module(arch)
    return name, MODULES[name]


def generated_program_available(
    device: Optional[torch.device] = None, *, backward: bool = False
) -> bool:
    """True when this checkout registers the program for ``device`` (and its
    backward stages when ``backward``)."""
    arch = arch_for(device)
    if arch is None:
        return False
    names = [n for n, r in MODULES.items() if r["arch"] == arch]
    if len(names) != 1:
        return False
    stages = registered_stages(names[0])
    if "fwd" not in stages:
        return False
    return not backward or ("bwd_main" in stages and "bwd_delta" in stages)


def record_abi(record: dict[str, Any]) -> str:
    abi = str(record.get("abi", ABI_CONTRACT))
    if abi not in SUPPORTED_ABIS:
        raise NotImplementedError(f"unsupported host binding profile {abi!r}")
    return abi


def record_dkv_acc_layout(record: dict[str, Any]) -> str:
    """Layout of the backward's FP32 dK/dV accumulators: ``"natural"`` (row-major
    ``[S, 512]`` / ``[S, 64]``, returnable as FP32 gradients) or ``"permuted"``
    (internal to the kernels; only ``bwd_cast`` yields natural-layout outputs).
    A record with a backward must declare it."""
    layout = record.get("dkv_acc_layout")
    if layout not in DKV_ACC_LAYOUTS:
        raise ValueError(
            f"registry record declares dkv_acc_layout {layout!r}; expected one of {DKV_ACC_LAYOUTS}"
        )
    return str(layout)


# Operands a ``bwd_cast`` argument plan declares when the cast can add the un-permuted dK/dV gradients into a
# caller-provided packed FP32 buffer with an optional destination-row map (``dkv_acc`` / ``dkv_dst_map``).
CAST_ACCUMULATE_OPERANDS = (
    "dst_packed",
    "dst_row_stride",
    "dst_map",
    "has_dst_map",
    "accumulate",
)


def record_cast_accumulates(record: dict[str, Any]) -> bool:
    """True when the record's ``bwd_cast`` plan declares :data:`CAST_ACCUMULATE_OPERANDS`, i.e. the program serves
    ``dkv_acc`` / ``dkv_dst_map``; a program exported from an older cast does not."""
    physical = record.get("bwd_cast")
    if physical is None:
        return False
    names = {name for _, name in physical["arg_plan"]}
    return set(CAST_ACCUMULATE_OPERANDS) <= names


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# Layout rules of the model operands (packed / strided trainer inputs, flashinfer-ai/flashinfer#5675).
#
# The head-dimension operands (q_latent, q_rope, dout, dq_*) reach the kernels through 4-D TMA descriptors whose
# global strides are the tensor's own head stride and token stride (the generated ``EncodeTma_*`` reads
# ``t.stride(ndim - 2)`` / ``t.stride(ndim - 3)`` of the TensorView); the key operands are 2-D TMA gather maps over the
# row stride (kv_latent, and k_rope in the backward) or 16-byte ``cp.async`` row loads at ``base + offset + row *
# stride`` (k_rope in the forward).  ``cuTensorMapEncodeTiled`` requires every global stride to be a positive
# multiple of 16 bytes below 2^40 and the global address to be 16-byte aligned; for BF16 that is a multiple of 8
# elements.  Nothing else is assumed about the layout: a head stride of 576 (a packed [T, 64, 576] query), 256 (the
# 192:256 rope channels of the [T, 64, 256] pre-absorption query) or any other admissible value and key row strides
# of 576 or 704 (a frozen 128-channel indexer key stored alongside) are consumed in place, without a host copy.
TMA_STRIDE_ELEMENTS = 8  # 16 bytes of BF16: the descriptor stride granule
TMA_ADDRESS_ALIGN = 16  # bytes: the descriptor's global address


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
    consumed in place); ``indices`` and ``topk_length`` are contiguous.  Zero key rows (``S == 0``) are rejected: the
    kernels index at least one key row and the FP32 dK/dV accumulators would
    be empty.  Zero query rows (``T == 0``) are rejected unless
    ``allow_empty_queries``: the launch grids clamp to one CTA that would read
    a query row that does not exist, so only the eager entry points, which
    return empty outputs for ``T == 0`` without launching, accept it.
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
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")
    topk = int(indices.shape[1])
    if topk <= 0:
        raise ValueError("topk must be positive")
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
        # ``bwd_delta`` addresses dout as a flat pointer of T * 64 contiguous 512-element rows (one (token, head)
        # row per warp, 16-byte vector loads at row * 512 + column), so a strided dout would be mis-read there;
        # it therefore stays contiguous although the main stage reads it through a TMA descriptor.  The eager
        # ``backward`` and the autograd Function call ``dout.contiguous()`` before validating; only
        # ``prepare_dsa_train`` callers see this error.
        if not dout.is_contiguous():
            raise ValueError("dout must be contiguous (call .contiguous() first)")
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
    synchronization) and raises ``ValueError`` on a violation.  Returns ``(operand,
    row_stride)``: the raw-pointer operand of ``bwd_cast`` (``dkv_acc`` itself when contiguous, else a flat stride-1
    alias from its first element over ``(S_dst - 1) * row_stride + cols`` elements -- the FFI boundary takes contiguous
    tensors only, and the kernel addresses ``row * row_stride + col``) and the row stride.  Device placement is checked
    by the callers, so this runs on host tensors.
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
    return (nbytes + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def workspace_layout(
    num_queries: int,
    num_kv: int,
    topk: int,
    *,
    abi: str = ABI_CONTRACT,
    backward: bool = True,
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
    key_passes: int = 1,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    ``delta`` and the FP32 dK/dV accumulators exist for the backward; the
    ``topk_length`` region backs a full-length vector when the caller passes
    none; ``tma_descriptor_workspace`` is the caller-owned descriptor storage
    of pointer-ABI programs.  A backward with more than one key-range pass adds
    the FP32 dQ partials (``num_queries`` x 147,456 B), the compacted keys of
    one pass (``num_queries`` x ``topk`` int32) and their per-token counts.  The
    seed profile adds the packed ``q``/``kv`` operands, an auxiliary row-max
    buffer and an (unused) sink vector.
    """
    sizes = [("topk_length", num_queries * 4)]
    if backward:
        sizes += [
            ("delta", num_queries * NUM_HEADS * 4),
            ("dkv_latent_acc", num_kv * D_LATENT * 4),
            ("dk_rope_acc", num_kv * D_ROPE * 4),
        ]
        if int(key_passes) > 1:
            sizes += [
                ("dq_partial", num_queries * DQ_PARTIAL_BYTES_PER_TOKEN),
                ("key_scratch", num_queries * int(topk) * 4),
                ("pass_counts", num_queries * 4),
            ]
    if abi == ABI_SEED:
        sizes += [
            ("packed_q", num_queries * NUM_HEADS * D_QK * 2),
            ("packed_kv", num_kv * D_QK * 2),
            ("aux_logits", num_queries * NUM_HEADS * 4),
            ("sinks", NUM_HEADS * 4),
        ]
    if backward and scratch_bytes:
        sizes.append(("workspace", int(scratch_bytes)))
    if tma_workspace_bytes:
        sizes.append(("tma_descriptor_workspace", int(tma_workspace_bytes)))
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def _record_tma_bytes(record: dict[str, Any], stages: tuple[str, ...]) -> int:
    return max(
        (int(record[s].get("tma_workspace_bytes", 0)) for s in stages), default=0
    )


def _record_scratch_bytes(record: dict[str, Any], stages: tuple[str, ...]) -> int:
    """Kernel-private scratch declared by the backward stages (``workspace_bytes``; 0 when none)."""
    return max((int(record[s].get("workspace_bytes", 0)) for s in stages), default=0)


def dsa_train_workspace_size(
    num_queries: int,
    num_kv: int,
    topk: int,
    device: Optional[torch.device] = None,
    *,
    backward: bool = True,
    key_passes: Optional[int] = None,
    num_segments: int = 1,
) -> int:
    """Workspace bytes :func:`prepare_dsa_train` needs for ``(T, S, topk)`` on ``device``.

    ``key_passes`` / ``num_segments`` as in :func:`prepare_dsa_train`
    (``None`` = the record's policy; the packed segment count of the key row).
    """
    name, record = record_for(device)
    stages = registered_stages(name)
    passes = (
        plan_key_passes(
            record,
            stages,
            num_queries,
            num_kv,
            topk,
            key_passes,
            num_segments=num_segments,
        )
        if backward
        else 1
    )
    return int(
        workspace_layout(
            num_queries,
            num_kv,
            topk,
            abi=record_abi(record),
            backward=backward,
            tma_workspace_bytes=_record_tma_bytes(record, stages),
            scratch_bytes=_record_scratch_bytes(record, stages) if backward else 0,
            key_passes=passes,
        )["total"]
    )


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
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
) -> torch.Tensor:
    """Turn per-document key indices into global key rows.

    ``gather_kv_indices [T, topk]`` holds, for query row ``t`` of document
    ``d``, key positions relative to the document's first key
    (``cu_seqlens_k[d]``); ``-1`` or a position ``>= seqlen_k[d]`` is invalid.
    Query and key lengths of a document may differ: the query segment is the
    tail of its key prefix, query ``local_q = t - cu_seqlens_q[d]`` sitting at
    key position ``(seqlen_k[d] - seqlen_q[d]) + local_q``.  With
    ``causal=True`` a slot is also invalid when it selects a key after that
    position (``idx <= (seqlen_k[d] - seqlen_q[d]) + local_q`` is required);
    the rule is the one of the Cake facade (``globalize_topk_indices``).  The
    default ``causal=False`` is the plain offsetting of the first release: the
    index row is taken as is and only ``-1`` / out-of-range slots are dropped.
    The result
    addresses the packed ``kv_*`` tensors; invalid slots become ``-1``.  A
    zero-length query segment contributes no rows.  Runs on device without a
    host synchronization, in int32 throughout (the ``[T, topk]`` temporaries
    are the dominant cost of the varlen entry: int64 copies of the index row
    tripled its traffic).
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
    total_q = int(gather_kv_indices.shape[0])
    num_docs = int(cu_seqlens_q.numel()) - 1
    device = gather_kv_indices.device
    # per-row document data: [T]-sized, int32 like cu_seqlens (key rows and positions fit int32)
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    doc_of_row = torch.repeat_interleave(
        torch.arange(num_docs, device=device),
        seqlens_q.to(torch.int64),
        output_size=total_q,
    )
    key_base = cu_seqlens_k[:-1][doc_of_row][:, None]
    key_len = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])[doc_of_row][:, None]
    local = gather_kv_indices
    # [T, topk] passes: three int32 compares into one bool mask, one int32 add, one masked fill
    valid = local >= 0
    valid &= local < key_len
    if causal:
        # Own key position of row t: (seqlen_k[d] - seqlen_q[d]) + (t - cu_seqlens_q[d]).
        own_position = (key_len[:, 0] - seqlens_q[doc_of_row]) + (
            torch.arange(total_q, device=device, dtype=torch.int32)
            - cu_seqlens_q[:-1][doc_of_row]
        )
        valid &= local <= own_position[:, None]
    result = local + key_base  # invalid slots hold arbitrary sums until the fill below
    result.masked_fill_(~valid, -1)
    if out is None:
        return result
    out.copy_(result)
    return out


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def grid_dims(rule, scalars: dict[str, Any], num_sms: int) -> tuple[int, int, int]:
    """Evaluate a registry grid rule.

    Each of the three entries is an integer, ``"sms"``, ``"sms*<n>"`` or
    ``"<name>[*<a>][/<b>]"``: a scalar name (``"num_queries"``) optionally
    multiplied by ``a`` and then divided by ``b`` with rounding up
    (``"num_kv*36/256"``).
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


# Host values that a remembered binding re-supplies at every launch: the
# contract tensors (caller tensors, fresh outputs and cache-owned scratch) and
# the storage alias / element offset pairs of the raw-pointer operands.
REBIND_KEYS = frozenset(CONTRACT_TENSORS) | frozenset(
    ("indices_storage", "indices_offset", "k_rope_storage", "k_rope_offset")
)


@dataclass(frozen=True)
class _Launch:
    stage: str
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]
    # Descriptor-preparation entry of a pointer-ABI module (same arguments).
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)
    # ``(argument index, host value name)`` of every argument in REBIND_KEYS.
    slots: tuple[tuple[int, str], ...] = ()

    def __call__(self) -> None:
        self.entry(*self.arguments)

    def templated(self) -> "_Launch":
        """Copy whose re-bindable arguments are ``None`` placeholders (holds no tensor)."""
        arguments = list(self.arguments)
        for index, _ in self.slots:
            arguments[index] = None
        return replace(self, arguments=tuple(arguments))

    def arguments_for(self, values: dict[str, Any]) -> list:
        """The argument list with every slot filled from ``values`` (fails closed on a missing one)."""
        arguments = list(self.arguments)
        for index, key in self.slots:
            value = values.get(key)
            if value is None:
                raise RuntimeError(
                    f"stage {self.stage!r} of {self.module!r}: the remembered argument plan needs "
                    f"{key!r} (argument {index}) and the call provides no value for it"
                )
            arguments[index] = value
        return arguments


def bind_stage(
    module_name: str, stage: str, values: dict[str, Any], grid: tuple[int, int, int]
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``stage`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.  The returned launch records which
    argument positions hold re-bindable host values (:data:`REBIND_KEYS`).
    """
    physical = MODULES[module_name][stage]
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    slots = []
    for kind, name in physical["arg_plan"]:
        # A profile may provide the kernel's own name (seed profile) or the contract name behind an alias.
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif (
            key in values and values[key] is not None
        ):  # buffer / tma_buffer / workspace / parameter
            value = values[key]
            if kind == "buffer" and f"{key}_storage" in values:
                key = f"{key}_storage"  # raw pointer: whole storage + <name>_offset elements
                value = values[key]
            if key in REBIND_KEYS:
                slots.append((len(arguments), key))
            arguments.append(value)
        else:
            raise KeyError(
                f"generated module {module_name!r} stage {stage!r} expects argument "
                f"{name!r} ({kind}); the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_dsa_train_module(module_name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return _Launch(
        stage,
        module_name,
        getattr(module, physical["ffi_entry"]),
        tuple(arguments),
        grid,
        prepare,
        tuple(slots),
    )


_FFI_DEVICES: dict[int, Any] = {}


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``.

    Same effect as ``tvm_ffi.use_torch_stream()`` without its per-entry
    ``torch.cuda.Stream`` wrapper and device-string parse: the raw handle comes
    from ``torch._C._cuda_getCurrentRawStream`` and the FFI device is cached.
    """
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
class DSATrainRunner:
    """The prepared forward / backward launches of one tensor binding.

    ``forward()`` writes ``out``, ``lse`` and ``o_lo``; ``backward()`` writes
    ``dq_latent``, ``dq_rope``, ``dkv_latent`` and ``dk_rope`` (the BF16 casts
    of the FP32 accumulators ``dkv_latent_acc`` / ``dk_rope_acc``; prepared
    with ``dkv_fp32=True`` it returns natural-layout FP32 gradients -- the
    accumulators themselves for a ``natural`` program, the cast's FP32 outputs
    ``dkv_latent_fp32`` / ``dk_rope_fp32`` for a ``permuted`` one); ``step()``
    runs both.  No launch allocates or synchronizes; capture into a CUDA graph
    belongs to the caller.  Prepare a new runner when a shape, dtype or tensor
    binding changes; values may change freely.

    ``launches`` is keyed by stage name; the launches of a multi-pass backward
    are keyed ``(stage, pass index)`` and ``backward_order`` lists every
    backward launch key in launch order (``bwd_delta``, then per pass
    ``bwd_compact`` and ``bwd_main_pass`` -- or the single ``bwd_main`` --,
    then ``bwd_cast``).
    """

    module_name: str
    abi: str
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
    # True when the program accumulates dK/dV in an internal permuted layout (the dkv_fp32 outputs come from the cast).
    dkv_acc_permuted: bool = False
    # Key-range passes of the backward main stage (1 = the single-pass stage) and the backward launch keys in order.
    key_passes: int = 1
    backward_order: tuple = ()
    # True when bwd_cast adds the dK/dV gradients into the caller's packed FP32 rows (``dkv_acc``): backward() then
    # returns None for dkv_latent / dk_rope.
    accumulate_dkv: bool = False
    _tma_prepared: bool = False

    @property
    def out(self) -> torch.Tensor:
        return self.tensors["out"]

    @property
    def lse(self) -> torch.Tensor:
        return self.tensors["lse"]

    @property
    def o_lo(self) -> Optional[torch.Tensor]:
        return self.tensors.get("o_lo")

    @property
    def has_backward(self) -> bool:
        return bool(self.backward_order)

    def prepare_tma(self) -> None:
        """Encode the descriptors of pointer-ABI stages once (idempotent)."""
        if self._tma_prepared:
            return
        with _ffi_stream_context(self.device_index):
            for launch in self.launches.values():
                if launch.prepare is not None:
                    launch.prepare(*launch.arguments)
        self._tma_prepared = True

    def _run(self, keys: tuple) -> None:
        self.prepare_tma()
        with _ffi_stream_context(self.device_index):
            for key in keys:
                launch = self.launches.get(key)
                if launch is not None:
                    launch()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        t = self.tensors
        if self.abi == ABI_SEED:
            # Placeholder profile: packed operands, no residual, +inf LSE on
            # empty rows.  The copies are host glue and count in its timing.
            t["packed_q"][:, :, :D_LATENT].copy_(t["q_latent"])
            t["packed_q"][:, :, D_LATENT:].copy_(t["q_rope"])
            t["packed_kv"][:, :D_LATENT].copy_(t["kv_latent"])
            t["packed_kv"][:, D_LATENT:].copy_(t["k_rope"])
        self._run(FORWARD_STAGES)
        if self.abi == ABI_SEED:
            lse = t["lse"]
            torch.isposinf(
                lse, out=t["lse_mask"]
            )  # empty rows; no temporaries (launch path allocates nothing)
            lse.masked_fill_(t["lse_mask"], float("-inf"))
        return t["out"], t["lse"], t.get("o_lo")

    def backward(
        self,
    ) -> tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        if not self.has_backward:
            raise NotImplementedError(
                "the registered DSA training program has no backward stages "
                f"(registered: {self.stages}); backward is unavailable in this checkout"
            )
        t = self.tensors
        t["dkv_latent_acc"].zero_()
        t["dk_rope_acc"].zero_()
        self._run(self.backward_order)
        if self.accumulate_dkv:
            # the cast added the gradients into the caller's dkv_acc rows: no dK/dV output tensors
            return t["dq_latent"], t["dq_rope"], None, None
        if self.dkv_fp32:
            if (
                self.dkv_acc_permuted
            ):  # natural-layout FP32 written by bwd_cast (out_f32 = 1)
                return (
                    t["dq_latent"],
                    t["dq_rope"],
                    t["dkv_latent_fp32"],
                    t["dk_rope_fp32"],
                )
            return t["dq_latent"], t["dq_rope"], t["dkv_latent_acc"], t["dk_rope_acc"]
        if "bwd_cast" not in self.launches:
            t["dkv_latent"].copy_(t["dkv_latent_acc"])
            t["dk_rope"].copy_(t["dk_rope_acc"])
        return t["dq_latent"], t["dq_rope"], t["dkv_latent"], t["dk_rope"]

    def step(self):
        self.forward()
        return self.backward()

    __call__ = step


def _pointer_alias(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Contiguous zero-copy alias of ``tensor``'s storage plus its element offset.

    Raw pointer arguments are checked for contiguity at the FFI boundary, and
    the kernels take the pointer they receive for the aligned base of the
    storage: their vector index-tile loads are gated on ``((idx_stride |
    indices_offset) & 7) == 0``, not on the pointer itself.  A contiguous
    tensor that starts its storage is therefore passed as is with offset 0;
    a strided view (the rope columns of a packed ``[S, 576]`` tensor) or a
    contiguous view that starts inside its storage (a slice of a larger
    buffer) is passed as the whole storage viewed flat plus the element offset
    the kernel adds.
    """
    if tensor.is_contiguous() and tensor.storage_offset() == 0:
        return tensor, 0
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    flat.set_(tensor.untyped_storage())
    return flat, int(tensor.storage_offset())


def _inert_pass_values(values: dict[str, Any], num_kv: int) -> dict[str, Any]:
    """Key-range-pass operands of the backward main stage for the single-pass program: the whole key range,
    ``dq_mode`` 0, and never-dereferenced placeholders (``delta`` for the FP32 dQ partials, the indices storage for
    the compacted keys and their counts) -- exactly what the production launcher passes for one pass."""
    values["pass_lo"], values["pass_hi"], values["dq_mode"] = 0, int(num_kv), 0
    if values.get("delta") is not None:
        values["dq_partial"] = values["delta"]
    if values.get("indices_storage") is not None:
        values["key_scratch"] = values["pass_counts"] = values["indices_storage"]
    return values


def _inert_cast_values(values: dict[str, Any]) -> dict[str, Any]:
    """Packed-accumulate operands of ``bwd_cast`` a call does not use: never-dereferenced placeholders of the right
    dtypes -- the FP32 accumulator for ``dst_packed`` (``accumulate = 0``) and the indices storage for ``dst_map``
    (``has_dst_map = 0``) -- as the production launcher passes them.  A call with ``dkv_acc`` keeps its own value; the
    map placeholder stands in for an absent ``dkv_dst_map``."""
    if values.get("dkv_acc") is None and values.get("dkv_latent_acc") is not None:
        values["dkv_acc"] = values["dkv_latent_acc"]
    if values.get("dkv_dst_map") is None and values.get("indices_storage") is not None:
        values["dkv_dst_map"] = values["indices_storage"]
    return values


def _pass_values(
    values: dict[str, Any], index: int, passes: int, key_range: tuple[int, int]
) -> dict[str, Any]:
    """The host values of one key-range pass: its key range and ``dq_mode``."""
    lo, hi = key_range
    return dict(
        values,
        pass_lo=int(lo),
        pass_hi=int(hi),
        dq_mode=key_pass_dq_mode(index, passes),
    )


def _contract_values(
    t: dict[str, torch.Tensor], scalars: dict[str, Any], *, key_passes: int = 1
) -> dict[str, Any]:
    values: dict[str, Any] = {name: t.get(name) for name in CONTRACT_TENSORS}
    values.update(scalars)
    values["softmax_scale_log2"] = float(scalars["softmax_scale"]) * LOG2E
    for name, key in (
        ("q_latent", "q_latent_row_stride"),
        ("q_rope", "q_rope_row_stride"),
        ("kv_latent", "kv_latent_row_stride"),
    ):
        values[key] = int(t[name].stride(0))
    values["idx_stride"] = int(t["indices"].stride(0))
    values["indices_storage"], values["indices_offset"] = _pointer_alias(t["indices"])
    values["k_rope_stride"] = int(t["k_rope"].stride(0))
    values["k_rope_storage"], values["k_rope_offset"] = _pointer_alias(t["k_rope"])
    values["num_rows"] = int(scalars["num_queries"]) * NUM_HEADS
    values["latent_vecs"] = int(scalars["num_kv"]) * D_LATENT // 16
    values["rope_vecs"] = int(scalars["num_kv"]) * D_ROPE // 16
    values["latent_groups"] = int(scalars["num_kv"]) * D_LATENT // 32
    values["rope_groups"] = int(scalars["num_kv"]) * D_ROPE // 32
    values["out_f32"] = int(scalars.get("out_f32", 0))
    values["token_base"], values["token_step"] = (
        0,
        1,
    )  # one CTA per token, token = blockIdx.x
    values["num_tokens"] = int(
        scalars["num_queries"]
    )  # bwd_compact: the whole row per pass
    # bwd_cast packed-accumulate scalars (0 / 0 / 0 unless the binding accumulates into dkv_acc)
    values["dst_row_stride"] = int(scalars.get("dst_row_stride", 0))
    values["has_dst_map"] = int(scalars.get("has_dst_map", 0))
    values["accumulate"] = int(scalars.get("accumulate", 0))
    _inert_cast_values(values)
    if int(key_passes) > 1:
        return values  # dq_partial / key_scratch / pass_counts are the carved regions; pass_lo/hi and dq_mode per launch
    return _inert_pass_values(values, int(scalars["num_kv"]))


def _seed_values(t: dict[str, torch.Tensor], scalars: dict[str, Any]) -> dict[str, Any]:
    packed_q = t["packed_q"]
    return dict(
        q=packed_q,
        kv=t["packed_kv"],
        q_rope=packed_q[:, :, D_LATENT:],
        kv_ptr=t["packed_kv"],
        out=t["out"],
        indices=t["indices"],
        lengths=t["topk_length"],
        sinks=t["sinks"],
        max_logits=t["aux_logits"],
        lse=t["lse"],
        num_queries=scalars["num_queries"],
        num_kv=scalars["num_kv"],
        topk=scalars["topk"],
        has_sinks=0,
        scale_log2=float(scalars["softmax_scale"]) * LOG2E,
        tma_descriptor_workspace=t.get("tma_descriptor_workspace"),
    )


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
    backend: str = "cake",
    num_segments: int = 1,
) -> DSATrainRunner:
    """Validate one binding and prepare its launches.

    ``backward`` defaults to ``dout is not None``; a backward binding requires
    the backward stages to be registered.  Missing outputs and the workspace
    are allocated here (the only allocations of the backend).  Pass
    ``workspace_buffer`` of :func:`dsa_train_workspace_size` bytes to reuse
    storage across steps.  ``dkv_fp32=True`` makes ``backward()`` return
    natural-layout FP32 dK/dV gradients (see :func:`record_dkv_acc_layout`).
    ``key_passes`` overrides the record's key-range-pass policy for the
    backward (``None`` = policy, 1 = the single-pass stage; see
    :func:`plan_key_passes`).  ``dkv_acc`` (FP32 ``[S_dst, >= 576]``, see
    :func:`_check_dkv_acc`) makes ``backward()`` ADD the dK/dV gradients into
    it -- latent columns ``0:512``, rope columns ``512:576`` of row
    ``dkv_dst_map[s]`` (optional int32 ``[S]``; identity by default; repeated
    rows sum) -- and return ``None`` for ``dkv_latent`` / ``dk_rope``; the caller
    owns zeroing.  It requires a program whose cast declares
    :data:`CAST_ACCUMULATE_OPERANDS` and excludes ``dkv_fp32``.
    """
    if backend != "cake":
        raise ValueError("DSA sparse-attention training supports backend='cake'")
    num_queries, num_kv, topk = validate_dsa_train_inputs(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length, dout=dout
    )
    if backward is None:
        backward = dout is not None
    if backward and dout is None:
        raise ValueError("a backward binding needs dout")
    device = q_latent.device
    tensors = [q_latent, q_rope, kv_latent, k_rope, indices]
    tensors += [
        t
        for t in (
            topk_length,
            dout,
            workspace_buffer,
            out,
            lse,
            o_lo,
            dq_latent,
            dq_rope,
            dkv_latent,
            dk_rope,
            dkv_acc,
            dkv_dst_map,
        )
        if t is not None
    ]
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    module_name, record = record_for(device)
    abi = record_abi(record)
    stages = registered_stages(module_name)
    if backward and ("bwd_main" not in stages or "bwd_delta" not in stages):
        raise NotImplementedError(
            f"the registered DSA training program {module_name!r} has no backward stages "
            f"(registered: {stages}); forward-only use is available"
        )
    if abi == ABI_SEED and (topk < 128 or topk % 64):
        raise NotImplementedError(
            "the placeholder forward program serves top-k values that are multiples of 64 and >= 128"
        )
    permuted = backward and record_dkv_acc_layout(record) == "permuted"
    if backward and dkv_fp32 and permuted and "bwd_cast" not in stages:
        raise NotImplementedError(
            f"program {module_name!r} accumulates dK/dV in a permuted layout and registers no cast stage; "
            "dkv_fp32 outputs are unavailable"
        )
    accumulate = dkv_acc is not None
    dkv_acc_operand: Optional[torch.Tensor] = None
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
        if dkv_latent is not None or dk_rope is not None:
            raise ValueError(
                "dkv_acc accumulates in place: no dkv_latent / dk_rope output tensors are written with it"
            )
        if "bwd_cast" not in stages or not record_cast_accumulates(record):
            raise NotImplementedError(
                f"program {module_name!r} registers no bwd_cast stage declaring the packed-accumulate operands "
                f"{CAST_ACCUMULATE_OPERANDS}; dkv_acc / dkv_dst_map need a program exported from a cast that declares them"
            )
        dkv_acc_operand, dst_row_stride = _check_dkv_acc(dkv_acc, dkv_dst_map, num_kv)
    elif dkv_dst_map is not None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
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
        )
        if backward
        else 1
    )
    if passes > 1 and "bwd_dq" in stages:
        raise NotImplementedError(
            "key-range passes are defined for programs whose main stage carries the dQ pass (no bwd_dq stage)"
        )

    _check_output(out, "out", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(lse, "lse", (num_queries, NUM_HEADS), torch.float32)
    _check_output(o_lo, "o_lo", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(
        dq_latent, "dq_latent", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16
    )
    _check_output(dq_rope, "dq_rope", (num_queries, NUM_HEADS, D_ROPE), torch.bfloat16)
    _check_output(dkv_latent, "dkv_latent", (num_kv, D_LATENT), torch.bfloat16)
    _check_output(dk_rope, "dk_rope", (num_kv, D_ROPE), torch.bfloat16)

    layout = workspace_layout(
        num_queries,
        num_kv,
        topk,
        abi=abi,
        backward=backward,
        tma_workspace_bytes=_record_tma_bytes(record, stages),
        scratch_bytes=_record_scratch_bytes(record, stages) if backward else 0,
        key_passes=passes,
    )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            layout["total"], dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}"
        )

    t: dict[str, torch.Tensor] = dict(
        q_latent=q_latent,
        q_rope=q_rope,
        kv_latent=kv_latent,
        k_rope=k_rope,
        indices=indices,
    )
    has_topk_length = int(topk_length is not None)
    if topk_length is None:
        topk_length = _carve(flat, layout, "topk_length", torch.int32, (num_queries,))
        topk_length.fill_(topk)
    t["topk_length"] = topk_length
    t["out"] = (
        out
        if out is not None
        else torch.empty(
            (num_queries, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=device
        )
    )
    t["lse"] = (
        lse
        if lse is not None
        else torch.empty((num_queries, NUM_HEADS), dtype=torch.float32, device=device)
    )
    if abi == ABI_CONTRACT:
        t["o_lo"] = o_lo if o_lo is not None else torch.empty_like(t["out"])
    else:
        t["packed_q"] = _carve(
            flat, layout, "packed_q", torch.bfloat16, (num_queries, NUM_HEADS, D_QK)
        )
        t["packed_kv"] = _carve(
            flat, layout, "packed_kv", torch.bfloat16, (num_kv, D_QK)
        )
        t["aux_logits"] = _carve(
            flat, layout, "aux_logits", torch.float32, (num_queries, NUM_HEADS)
        )
        t["sinks"] = _carve(flat, layout, "sinks", torch.float32, (NUM_HEADS,))
        t["lse_mask"] = torch.empty(
            (num_queries, NUM_HEADS), dtype=torch.bool, device=device
        )  # +inf -> -inf rewrite scratch
    if backward:
        t["dout"] = dout
        t["delta"] = _carve(
            flat, layout, "delta", torch.float32, (num_queries, NUM_HEADS)
        )
        t["dkv_latent_acc"] = _carve(
            flat, layout, "dkv_latent_acc", torch.float32, (num_kv, D_LATENT)
        )
        t["dk_rope_acc"] = _carve(
            flat, layout, "dk_rope_acc", torch.float32, (num_kv, D_ROPE)
        )
        t["dq_latent"] = (
            dq_latent
            if dq_latent is not None
            else torch.empty(
                (num_queries, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=device
            )
        )
        t["dq_rope"] = (
            dq_rope
            if dq_rope is not None
            else torch.empty(
                (num_queries, NUM_HEADS, D_ROPE), dtype=torch.bfloat16, device=device
            )
        )
        if accumulate:
            # the cast adds into the caller's rows (accumulate = 1): its BF16 / FP32 output pointers are not dereferenced
            t["dkv_acc"] = dkv_acc_operand
            if dkv_dst_map is not None:
                t["dkv_dst_map"] = dkv_dst_map
            t["dkv_latent"] = torch.empty((0,), dtype=torch.bfloat16, device=device)
            t["dk_rope"] = torch.empty((0,), dtype=torch.bfloat16, device=device)
            t["dkv_latent_fp32"], t["dk_rope_fp32"] = (
                t["dkv_latent_acc"],
                t["dk_rope_acc"],
            )
        elif not dkv_fp32:
            t["dkv_latent"] = (
                dkv_latent
                if dkv_latent is not None
                else torch.empty(
                    (num_kv, D_LATENT), dtype=torch.bfloat16, device=device
                )
            )
            t["dk_rope"] = (
                dk_rope
                if dk_rope is not None
                else torch.empty((num_kv, D_ROPE), dtype=torch.bfloat16, device=device)
            )
            # the cast's FP32 output pointers are not dereferenced when out_f32 = 0: alias the accumulators
            t["dkv_latent_fp32"], t["dk_rope_fp32"] = (
                t["dkv_latent_acc"],
                t["dk_rope_acc"],
            )
        elif permuted:
            # natural-layout FP32 outputs written by bwd_cast (out_f32 = 1); its BF16 output pointers are not dereferenced
            t["dkv_latent_fp32"] = torch.empty(
                (num_kv, D_LATENT), dtype=torch.float32, device=device
            )
            t["dk_rope_fp32"] = torch.empty(
                (num_kv, D_ROPE), dtype=torch.float32, device=device
            )
            t["dkv_latent"] = torch.empty((0,), dtype=torch.bfloat16, device=device)
            t["dk_rope"] = torch.empty((0,), dtype=torch.bfloat16, device=device)
    if passes > 1:
        t["dq_partial"] = _carve(
            flat,
            layout,
            "dq_partial",
            torch.float32,
            (num_queries, DQ_PARTIAL_BYTES_PER_TOKEN // 4),
        )
        t["key_scratch"] = _carve(
            flat, layout, "key_scratch", torch.int32, (num_queries, topk)
        )
        t["pass_counts"] = _carve(
            flat, layout, "pass_counts", torch.int32, (num_queries,)
        )
    if layout.get("workspace"):
        t["workspace"] = _carve(
            flat, layout, "workspace", torch.uint8, (layout["workspace"][1],)
        )
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = _carve(
            flat,
            layout,
            "tma_descriptor_workspace",
            torch.uint8,
            (layout["tma_descriptor_workspace"][1],),
        )

    scalars = dict(
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=float(softmax_scale),
        has_topk_length=has_topk_length,
        out_f32=int(bool(backward and dkv_fp32 and permuted)),
        dst_row_stride=int(dst_row_stride),
        has_dst_map=int(accumulate and dkv_dst_map is not None),
        accumulate=int(accumulate),
    )
    values = (
        _seed_values(t, scalars)
        if abi == ABI_SEED
        else _contract_values(t, scalars, key_passes=passes)
    )
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    launches: dict[Any, _Launch] = {}

    def bind(stage: str, key: Any, stage_values: dict[str, Any]) -> None:
        physical = record[stage]
        grid = grid_dims(physical.get("grid", ["num_queries", 1, 1]), scalars, num_sms)
        cluster = physical.get("launch", {}).get("cluster")
        if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
            raise ValueError(
                f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
            )
        launches[key] = bind_stage(module_name, stage, stage_values, grid)

    for stage in FORWARD_STAGES:
        if stage in stages:
            bind(stage, stage, values)
    backward_order: list = []
    if backward:
        bind("bwd_delta", "bwd_delta", values)
        backward_order.append("bwd_delta")
        if passes == 1:
            for stage in ("bwd_main", "bwd_dq"):
                if stage in stages:
                    bind(stage, stage, values)
                    backward_order.append(stage)
        else:
            for index, key_range in enumerate(key_pass_ranges(num_kv, passes)):
                pass_values = _pass_values(values, index, passes, key_range)
                for stage in KEY_PASS_STAGES:
                    bind(stage, (stage, index), pass_values)
                    backward_order.append((stage, index))
        # with natural-layout FP32 accumulators as the outputs the cast is skipped
        if "bwd_cast" in stages and not (dkv_fp32 and not permuted):
            bind("bwd_cast", "bwd_cast", values)
            backward_order.append("bwd_cast")
    return DSATrainRunner(
        module_name=module_name,
        abi=abi,
        num_queries=num_queries,
        num_kv=num_kv,
        topk=topk,
        softmax_scale=float(softmax_scale),
        tensors=t,
        launches=launches,
        stages=stages,
        dkv_fp32=bool(dkv_fp32),
        has_topk_length=bool(has_topk_length),
        device_index=int(
            device.index if device.index is not None else torch.cuda.current_device()
        ),
        workspace=flat,
        layout=layout,
        dkv_acc_permuted=bool(permuted),
        key_passes=int(passes),
        backward_order=tuple(backward_order),
        accumulate_dkv=bool(accumulate),
    )


# ---------------------------------------------------------------------------
# Binding cache of the eager entry points
# ---------------------------------------------------------------------------

BINDING_CACHE_ENV = (
    "FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE"  # "0" disables the cache at import
)
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY"  # bindings kept (a positive integer)
# Enough for the forward and the backward binding of every layer of a deep model (a binding holds a few kilobytes).
BINDING_CACHE_DEFAULT_CAPACITY = 256


def binding_cache_capacity() -> int:
    """Capacity of :data:`BINDING_CACHE`: ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY`` when set, else
    :data:`BINDING_CACHE_DEFAULT_CAPACITY`."""
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


def _meta(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), t.shape, t.stride(), t.dtype)


def forward_binding_key(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> tuple:
    """Cache key of a forward binding: ``(data_ptr, shape, stride, dtype)`` of
    every input (``None`` for an absent ``topk_length``) plus the scale.  Two
    tensors at one address with another shape, stride or dtype (a buffer freed
    and re-allocated for another problem) never share a key."""
    return (
        "fwd",
        _meta(q_latent),
        _meta(q_rope),
        _meta(kv_latent),
        _meta(k_rope),
        _meta(indices),
        None if topk_length is None else _meta(topk_length),
        float(softmax_scale),
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
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    num_segments: int = 1,
) -> tuple:
    """Cache key of a backward binding: the forward key's inputs plus the saved
    forward outputs, ``dout``, the ``dkv_fp32`` option, the ``key_passes``
    override (``None`` = policy), the packed accumulator / destination map
    (``None`` when absent; their row stride and presence are baked into the
    cast's launch) and the segment count the pass policy saw."""
    return (
        "bwd",
        _meta(q_latent),
        _meta(q_rope),
        _meta(kv_latent),
        _meta(k_rope),
        _meta(indices),
        _meta(out),
        _meta(o_lo),
        _meta(lse),
        _meta(dout),
        None if topk_length is None else _meta(topk_length),
        float(softmax_scale),
        bool(dkv_fp32),
        None if key_passes is None else int(key_passes),
        None if dkv_acc is None else _meta(dkv_acc),
        None if dkv_dst_map is None else _meta(dkv_dst_map),
        int(num_segments),
    )


# Values a remembered binding owns (the caller never sees them): the caller-owned descriptor workspace of the
# pointer-ABI stages (every launch re-encodes its tensor maps into it), kernel-private scratch a record declares
# (``workspace_bytes``) and, when the caller passes no ``topk_length``, the materialized full-length vector.  Every
# other scratch of a launch -- ``delta``, the FP32 dK/dV accumulators, the key-range-pass regions -- is allocated
# per call from the caching allocator like the outputs, so a binding holds a few kilobytes whatever the problem size.
_OWNED_VALUES = ("workspace", "tma_descriptor_workspace")


@dataclass
class _Binding:
    """One remembered input binding of the contract profile.

    Holds the templated launches (argument plans with ``None`` at every
    re-bindable position: no caller tensor is pinned), the few values the
    binding owns (:data:`_OWNED_VALUES` and a materialized ``topk_length``;
    no problem-sized scratch) and the plain facts a launch needs.  ``forward``
    / ``backward`` allocate the outputs and the per-call scratch, slot them
    and the current tensors into the templates and launch; storage aliases
    and element offsets of the raw-pointer operands are re-read from the
    current tensors every call.
    """

    num_queries: int
    num_kv: int
    topk: int
    device: torch.device
    device_index: int
    dkv_fp32: bool
    launches: dict[Any, _Launch] = field(repr=False)
    owned: dict[str, torch.Tensor] = field(repr=False)
    owned_bytes: int = 0
    dkv_acc_permuted: bool = False
    key_passes: int = 1
    backward_order: tuple = ()
    accumulate_dkv: bool = (
        False  # the cast adds into the caller's dkv_acc rows (re-supplied per call)
    )

    @classmethod
    def from_runner(cls, runner: DSATrainRunner) -> "_Binding":
        if runner.abi != ABI_CONTRACT:
            raise ValueError(f"only the {ABI_CONTRACT!r} profile can be remembered")
        t = runner.tensors
        # Own separate allocations, never views of the runner's workspace: a view would keep the whole
        # workspace alive.  The descriptor workspace needs no content (every launch encodes into it).
        owned = {name: torch.empty_like(t[name]) for name in _OWNED_VALUES if name in t}
        if not runner.has_topk_length:
            owned["topk_length"] = t["topk_length"].clone()
        if runner.accumulate_dkv or (runner.dkv_fp32 and runner.dkv_acc_permuted):
            # zero-element BF16 placeholders of the cast (not dereferenced when out_f32 = 1 or accumulate = 1)
            for name in ("dkv_latent", "dk_rope"):
                owned[name] = t[name]
        return cls(
            num_queries=runner.num_queries,
            num_kv=runner.num_kv,
            topk=runner.topk,
            device=t["q_latent"].device,
            device_index=runner.device_index,
            dkv_fp32=runner.dkv_fp32,
            launches={
                key: launch.templated() for key, launch in runner.launches.items()
            },
            owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
            dkv_acc_permuted=runner.dkv_acc_permuted,
            key_passes=runner.key_passes,
            backward_order=runner.backward_order,
            accumulate_dkv=runner.accumulate_dkv,
        )

    def holds_no_tensor(self) -> bool:
        return not any(
            isinstance(a, torch.Tensor)
            for launch in self.launches.values()
            for a in launch.arguments
        )

    def rebind_values(self, current: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Host values of one launch: the owned values, the current tensors (inputs,
        outputs and per-call scratch) and the storage alias / element offset of the
        raw-pointer operands, re-read now."""
        values: dict[str, Any] = dict(self.owned)
        values.update(current)
        values["indices_storage"], values["indices_offset"] = _pointer_alias(
            current["indices"]
        )
        values["k_rope_storage"], values["k_rope_offset"] = _pointer_alias(
            current["k_rope"]
        )
        _inert_cast_values(values)
        if self.key_passes > 1:
            return values  # the owned dq_partial / key_scratch / pass_counts regions; pass scalars are baked per launch
        return _inert_pass_values(values, self.num_kv)

    def _launch(self, current: dict[str, torch.Tensor], keys: tuple) -> None:
        values = self.rebind_values(current)
        with _ffi_stream_context(self.device_index):
            for key in keys:
                launch = self.launches.get(key)
                if launch is None:
                    continue
                arguments = launch.arguments_for(values)
                if (
                    launch.prepare is not None
                ):  # descriptors of a pointer-ABI stage see the fresh outputs
                    launch.prepare(*arguments)
                launch.entry(*arguments)

    def forward(self, q_latent, q_rope, kv_latent, k_rope, indices, topk_length):
        shape = (self.num_queries, NUM_HEADS, D_LATENT)
        out = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        o_lo = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        lse = torch.empty(shape[:2], dtype=torch.float32, device=self.device)
        current = dict(
            q_latent=q_latent,
            q_rope=q_rope,
            kv_latent=kv_latent,
            k_rope=k_rope,
            indices=indices,
            out=out,
            o_lo=o_lo,
            lse=lse,
        )
        if topk_length is not None:
            current["topk_length"] = topk_length
        self._launch(current, FORWARD_STAGES)
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
        T, S, device = self.num_queries, self.num_kv, self.device
        dq_latent = torch.empty(
            (T, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=device
        )
        dq_rope = torch.empty(
            (T, NUM_HEADS, D_ROPE), dtype=torch.bfloat16, device=device
        )
        # Per-call scratch from the caching allocator: the FP32 dK/dV accumulators are one zero-filled span
        # (a single fill kernel, as on the prepared runner's path); delta and the key-range-pass regions are
        # fully written before they are read.
        acc = torch.zeros((S * D_QK,), dtype=torch.float32, device=device)
        acc_latent = acc[: S * D_LATENT].view(S, D_LATENT)
        acc_rope = acc[S * D_LATENT :].view(S, D_ROPE)
        current = dict(
            q_latent=q_latent,
            q_rope=q_rope,
            kv_latent=kv_latent,
            k_rope=k_rope,
            indices=indices,
            out=out,
            o_lo=o_lo,
            lse=lse,
            dout=dout,
            dq_latent=dq_latent,
            dq_rope=dq_rope,
            delta=torch.empty((T, NUM_HEADS), dtype=torch.float32, device=device),
            dkv_latent_acc=acc_latent,
            dk_rope_acc=acc_rope,
        )
        if topk_length is not None:
            current["topk_length"] = topk_length
        if self.key_passes > 1:
            current["dq_partial"] = torch.empty(
                (T, DQ_PARTIAL_BYTES_PER_TOKEN // 4), dtype=torch.float32, device=device
            )
            current["key_scratch"] = torch.empty(
                (T, self.topk), dtype=torch.int32, device=device
            )
            current["pass_counts"] = torch.empty((T,), dtype=torch.int32, device=device)
        if self.accumulate_dkv:
            if dkv_acc is None:
                raise RuntimeError(
                    "this binding accumulates the dK/dV gradients into dkv_acc and the call provides none"
                )
            # the same binding key implies the validated shape / stride / dtype; the alias is rebuilt per call
            current["dkv_acc"], _row_stride = _check_dkv_acc(dkv_acc, dkv_dst_map, S)
            if dkv_dst_map is not None:
                current["dkv_dst_map"] = dkv_dst_map
            # the cast's FP32 output pointers are not dereferenced when accumulate = 1: alias the accumulators
            current["dkv_latent_fp32"], current["dk_rope_fp32"] = acc_latent, acc_rope
            self._launch(current, self.backward_order)
            return dq_latent, dq_rope, None, None
        if self.dkv_fp32:
            if (
                self.dkv_acc_permuted
            ):  # fresh natural-layout FP32 outputs from the cast (out_f32 = 1)
                current["dkv_latent_fp32"] = torch.empty(
                    (S, D_LATENT), dtype=torch.float32, device=device
                )
                current["dk_rope_fp32"] = torch.empty(
                    (S, D_ROPE), dtype=torch.float32, device=device
                )
                self._launch(current, self.backward_order)
                return (
                    dq_latent,
                    dq_rope,
                    current["dkv_latent_fp32"],
                    current["dk_rope_fp32"],
                )
            self._launch(
                current, self.backward_order
            )  # natural layout: the accumulators are the outputs
            return dq_latent, dq_rope, acc_latent, acc_rope
        dkv_latent = torch.empty((S, D_LATENT), dtype=torch.bfloat16, device=device)
        dk_rope = torch.empty((S, D_ROPE), dtype=torch.bfloat16, device=device)
        current["dkv_latent"] = dkv_latent
        current["dk_rope"] = dk_rope
        # the cast's FP32 output pointers are not dereferenced when out_f32 = 0: alias the accumulators
        current["dkv_latent_fp32"], current["dk_rope_fp32"] = acc_latent, acc_rope
        self._launch(current, self.backward_order)
        if "bwd_cast" not in self.launches:
            dkv_latent.copy_(acc_latent)
            dk_rope.copy_(acc_rope)
        return dq_latent, dq_rope, dkv_latent, dk_rope


class BindingCache:
    """Remembered input bindings of the eager entry points.

    The first call for a binding goes through :func:`prepare_dsa_train` (full
    validation, workspace allocation, argument-plan binding) and launches
    through its runner; the runner's launches are then remembered as
    templates keyed by :func:`forward_binding_key` /
    :func:`backward_binding_key`.  A later call with the same key skips the
    Python-level validation and binding: it allocates fresh outputs, slots
    them and the current tensors into the templates and launches (the module,
    its descriptor encoding and the cubin are those of the validating path).
    A binding pins no caller tensor and holds no problem-sized scratch: the
    outputs, ``delta``, the FP32 dK/dV accumulators and the key-range-pass
    regions are allocated per call from the caching allocator, and a binding
    owns only the descriptor workspace of the pointer-ABI stages and, when the
    caller passes no ``topk_length``, the materialized vector (a few
    kilobytes; see :class:`_Binding`).  The cache keeps at most ``capacity``
    bindings (:data:`BINDING_CACHE_DEFAULT_CAPACITY`, or the value of
    ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY``) and evicts the least
    recently used one first -- a hit refreshes its binding -- so a model whose
    layers cycle through up to ``capacity`` bindings per step binds each of
    them once.  ``FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE=0`` or
    ``enabled = False`` routes every call through the validating path.
    """

    def __init__(self, capacity: Optional[int] = None, enabled: bool = True):
        self.capacity = binding_cache_capacity() if capacity is None else int(capacity)
        if self.capacity < 1:
            raise ValueError(
                "BindingCache needs a capacity of at least one binding (use enabled=False to bypass it)"
            )
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0
        self._bindings: OrderedDict[tuple, _Binding] = OrderedDict()

    def __len__(self) -> int:
        return len(self._bindings)

    @property
    def owned_bytes(self) -> int:
        return sum(b.owned_bytes for b in self._bindings.values())

    def clear(self) -> None:
        self._bindings.clear()

    def lookup(self, key: tuple) -> Optional[_Binding]:
        """The binding remembered for ``key`` (counts a hit or a miss); a hit becomes the most recently used."""
        binding = self._bindings.get(key)
        if binding is None:
            self.misses += 1
        else:
            self.hits += 1
            self._bindings.move_to_end(key)
        return binding

    def peek(self, key: tuple) -> Optional[_Binding]:
        """The binding remembered for ``key`` without touching the counters or the recency order (inspection)."""
        return self._bindings.get(key)

    def remember(self, key: tuple, binding: _Binding) -> _Binding:
        """Remember ``binding`` as the most recently used; evict the least recently used beyond ``capacity``."""
        self._bindings.pop(key, None)
        self._bindings[key] = binding
        while len(self._bindings) > self.capacity:
            self._bindings.popitem(last=False)
        return binding


BINDING_CACHE = BindingCache(enabled=os.environ.get(BINDING_CACHE_ENV, "1") != "0")


# ---------------------------------------------------------------------------
# Eager entry points (allocate, launch, return)
# ---------------------------------------------------------------------------


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
    out = torch.empty(
        (0, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=q_latent.device
    )
    lse = torch.empty((0, NUM_HEADS), dtype=torch.float32, device=q_latent.device)
    return out, lse, torch.empty_like(out)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of a call without query rows: empty ``dq``, zero ``dkv`` in the requested dtype; nothing launched."""
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
    dtype = torch.float32 if dkv_fp32 else torch.bfloat16
    return (
        torch.empty((0, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=device),
        torch.empty((0, NUM_HEADS, D_ROPE), dtype=torch.bfloat16, device=device),
        torch.zeros((num_kv, D_LATENT), dtype=dtype, device=device),
        torch.zeros((num_kv, D_ROPE), dtype=dtype, device=device),
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
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass: ``(out, lse, o_lo)``; ``o_lo`` is ``None`` for the placeholder program.

    The first call for an input binding validates and binds through
    :func:`prepare_dsa_train`; later calls with the same binding take the
    remembered launch (:data:`BINDING_CACHE`).  A call without query rows
    returns empty outputs without binding or launching.
    """
    if _no_query_rows(q_latent):
        return _empty_forward(q_latent, q_rope, kv_latent, k_rope, indices, topk_length)
    scale = (
        float(softmax_scale) if softmax_scale is not None else default_softmax_scale()
    )
    cache = BINDING_CACHE
    key = None
    if cache.enabled:
        key = forward_binding_key(
            q_latent, q_rope, kv_latent, k_rope, indices, topk_length, scale
        )
        binding = cache.lookup(key)
        if binding is not None:
            return binding.forward(
                q_latent, q_rope, kv_latent, k_rope, indices, topk_length
            )
    runner = prepare_dsa_train(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=scale,
        backward=False,
    )
    result = runner.forward()
    if key is not None and runner.abi == ABI_CONTRACT:
        cache.remember(key, _Binding.from_runner(runner))
    return result


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

    Validates and binds once per input binding like :func:`forward`.  With
    ``dkv_fp32=True`` the dK/dV gradients are natural-layout FP32 tensors
    (fresh per call, never the kernels' internal accumulators).  ``key_passes``
    overrides the key-range-pass policy of the main stage (``None`` = the
    registered policy; see :func:`plan_key_passes`); ``num_segments`` is the
    packed segment count of the key row the policy plans with (the varlen
    entry passes ``len(cu_seqlens_k) - 1``; more than one segment plans one
    pass).  A call without query
    rows returns empty ``dq`` and zero ``dkv`` gradients without binding or
    launching.  ``dkv_acc`` (FP32 ``[S_dst, >= 576]``) receives the dK/dV
    gradients in place instead -- latent columns ``0:512``, rope ``512:576`` of
    row ``dkv_dst_map[s]`` (optional int32 ``[S]``, identity by default,
    repeated rows sum); the caller owns zeroing and the result is
    ``(dq_latent, dq_rope, None, None)`` (see :func:`prepare_dsa_train`).
    """
    if o_lo is None:
        raise NotImplementedError(
            "the forward produced no output residual (placeholder program); backward is unavailable"
        )
    if dkv_acc is not None and dkv_fp32:
        raise ValueError(
            "dkv_acc accumulates the dK/dV gradients in place; dkv_fp32 outputs are not produced with it"
        )
    if dkv_dst_map is not None and dkv_acc is None:
        raise ValueError("dkv_dst_map is only meaningful together with dkv_acc")
    dout = dout.contiguous()
    if _no_query_rows(q_latent):
        empty = _empty_backward(
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
        )
        if dkv_acc is None:
            return empty
        _check_dkv_acc(
            dkv_acc, dkv_dst_map, int(kv_latent.shape[0])
        )  # nothing to accumulate
        return empty[0], empty[1], None, None
    scale = (
        float(softmax_scale) if softmax_scale is not None else default_softmax_scale()
    )
    cache = BINDING_CACHE
    key = None
    if cache.enabled:
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
            dkv_acc,
            dkv_dst_map,
            num_segments,
        )
        binding = cache.lookup(key)
        if binding is not None:
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
                dkv_acc=dkv_acc,
                dkv_dst_map=dkv_dst_map,
            )
    _check_output(out, "out", (q_latent.shape[0], NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(
        o_lo, "o_lo", (q_latent.shape[0], NUM_HEADS, D_LATENT), torch.bfloat16
    )
    _check_output(lse, "lse", (q_latent.shape[0], NUM_HEADS), torch.float32)
    runner = prepare_dsa_train(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        dout=dout,
        softmax_scale=scale,
        out=out,
        lse=lse,
        o_lo=o_lo,
        dkv_fp32=dkv_fp32,
        backward=True,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=num_segments,
    )
    result = runner.backward()
    if key is not None and runner.abi == ABI_CONTRACT:
        cache.remember(key, _Binding.from_runner(runner))
    return result


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
    ):
        out, lse, o_lo = forward(
            q_latent,
            q_rope,
            kv_latent,
            k_rope,
            indices,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
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
        saved = [q_latent, q_rope, kv_latent, k_rope, indices, out, lse]
        saved.append(o_lo if o_lo is not None else out.new_empty(0))
        saved.append(topk_length if topk_length is not None else indices.new_empty(0))
        ctx.has_o_lo = o_lo is not None
        ctx.save_for_backward(*saved)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        if dlse is not None:
            raise NotImplementedError(
                "gradients through lse are not supported; only out is differentiable"
            )
        if dout is None:  # out unused downstream
            return (None,) * 11
        q_latent, q_rope, kv_latent, k_rope, indices, out, lse, o_lo, topk_length = (
            ctx.saved_tensors
        )
        if not ctx.has_o_lo:
            raise NotImplementedError(
                "backward is unavailable: the registered program is forward-only (placeholder)"
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
        # dkv_latent / dk_rope are None when the gradients went into dkv_acc; no gradient for the seven other inputs
        return (
            dq_latent,
            dq_rope,
            dkv_latent,
            dk_rope,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    """Differentiable sparse attention over global key indices (see the module docstring).

    Inputs are validated on the first call for a binding (see :class:`BindingCache`);
    strided query slices and packed key rows are consumed in place (see
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
    """
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
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
    """Packed multi-document form: per-document ``gather_kv_indices`` are offset by
    ``cu_seqlens_k`` on device (this glue counts in the step time), then the flat
    kernels run over the packed rows.  Query and key segment lengths may differ
    (the query segment is the tail of its key prefix); with ``causal=True`` the
    offsetting also drops selected keys after the query's own position
    ``(seqlen_k - seqlen_q) + local_q`` (:func:`offset_gather_kv_indices`); the
    default ``causal=False`` keeps the index rows as is (the behaviour of the
    first release).
    ``max_seqlen_q/k`` are accepted for signature parity and not used on the
    host.  ``dkv_acc`` / ``dkv_dst_map`` as in :func:`dsa_sparse_attention`.
    The segment count ``len(cu_seqlens_k) - 1`` (host metadata, no device
    sync) is passed on as ``num_segments``: with more than one segment the
    backward plans the single-pass stage unless ``key_passes`` overrides."""
    del max_seqlen_q, max_seqlen_k
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
