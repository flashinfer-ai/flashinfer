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

Cake backend: chunked large-vocabulary LM-head projection + loss with a
memory-bounded backward for training on SM100 / SM103
(flashinfer-ai/flashinfer#5680).

Contract
--------
* ``X [T, H]`` BF16 hidden states (row-major; any leading stride ``ld_X >= H``
  whose byte pitch is a multiple of 16 -- otherwise a contiguous copy is made
  and reported), ``W [V, H]`` BF16 output weight (contiguous), ``labels [T]``
  int64 with ``-100`` = ignored.  ``V`` and ``H`` must be multiples of 256
  (the GEMM tiles; GLM-class geometry ``H = 6144``, ``V = 154880``).  ``T`` is
  arbitrary (irregular, not a multiple of the chunk); the token chunk ``C``
  (default 4096) bounds every vocabulary-sized intermediate.
* Precision boundary: BF16 GEMM output ``z = bf16(X @ W^T)``, promoted to FP32
  for max / log-sum-exp / loss arithmetic; ``dlogits`` BF16; ``dW`` accumulated
  in FP32 across chunks and cast ONCE at the output boundary (BF16 by default,
  FP32 on request); ``dX`` accumulated in FP32 and cast once to BF16.
* ``logp_t = z[t, y_t] - logsumexp_v z[t, v]`` for valid rows, 0 for ignored
  rows.  Cross-entropy: ``loss = -sum(logp[valid]) / loss_div`` with a
  caller-supplied positive ``loss_div`` (never a local token mean);
  ``d_t = -1 / loss_div`` on valid rows.  Policy: ``loss = -sum(w_t * min(
  exp(logp_t - infer_logp_t), 2))`` over valid rows (``w`` FP32 signed, already
  masked and normalized); ``d_t = -w_t * ratio_t`` when ``ratio_t <= 2`` and
  ``0`` when ``ratio_t > 2``.  ``dz[t, v] = d_t * (1[v = y_t] - softmax(z_t)_v)``;
  ignored rows: zero loss, zero gradient, ``logp = 0``.  ``dX = dz @ W``,
  ``dW = dz^T @ X`` chunk by chunk.
* Entry (a) ``chunked_lm_head_loss``: the forward produces ``dX_acc`` (FP32
  ``[T, H]``) and ``dW_acc`` (FP32 ``[V, H]``) directly per chunk (three GEMMs,
  no logits recomputation); the backward applies the incoming scalar
  gradient ``g`` and performs the single cast ``dX = bf16(g * dX_acc)``,
  ``dW = cast(g * dW_acc)`` -- never mutating the saved accumulators, so a
  retained graph may run the backward repeatedly.  Entry (b)
  ``chunked_lm_head_logprob``: the forward saves only the FP32 row statistic
  ``lse[T]``; the backward recomputes each chunk's logits (four GEMMs) with the
  incoming ``dlogp`` as ``d_t`` (masked to valid rows).
* Memory rule: no logits / dlogits / probability buffer spans more than ``C``
  tokens (a batch smaller than ``C`` is one chunk).  :func:`memory_report`
  states the peak temporary bytes separately from the model weights, the
  required outputs and the FP32 gradient accumulators.
* ``deterministic=True`` (the only mode this backend serves): fixed sequential
  chunk order, no atomics -- bitwise reproducible ``loss``, ``logp``, ``dX``
  and ``dW`` across runs.

Every allocation happens in :func:`prepare_lm_head_loss`; the returned runner
launches with no CUDA allocation and no host synchronization, so a runner (or
a CUDA graph capturing it) replays for new values written into the bound
tensors.  Kernels are reached through the argument plans of the registry
records in ``cake_jit.MODULES``; the ``abi`` field of a record names the
keyword set its kernels expect (see :data:`ABI_CONTRACT` and
:func:`stage_values`).  ``backend="reference"`` runs the same chunk loop with
PyTorch operators at the same rounding boundaries (a host-layer development
aid that needs no generated program and no CUDA device).

The eager entry points (:func:`forward_loss`, :func:`backward_loss`,
:func:`forward_logprob`, :func:`backward_logprob` and the autograd Functions
behind the public API) validate and bind once per *input binding* and remember
the result in :data:`BINDING_CACHE` (see :class:`BindingCache`): a later call
whose inputs have the same ``(data_ptr, shape, stride, dtype)`` and options
slots the current tensors, freshly allocated outputs and per-call scratch into
the remembered argument plans and launches.  A call without rows (``T == 0``)
returns loss 0, an empty ``logp`` and zero gradients without binding or
launching anything.
"""

from __future__ import annotations

import ast
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import torch

from .cake_jit import (
    MODULES,
    STAGES,
    load_cake_lm_head_loss_module,
    registered_stages,
    select_module,
)

IGNORE_INDEX = -100
DEFAULT_CHUNK_SIZE = 4096
RATIO_CLIP = 2.0
OBJECTIVES = ("ce", "policy")
ENTRIES = ("loss", "logprob")
BACKENDS = ("cake", "reference")
GRAD_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)
# ``mode`` scalar of ``row_finalize`` / ``loss_reduce``: which per-row quantities the stage forms.
MODE_CE, MODE_POLICY, MODE_EXTERNAL, MODE_NONE = 0, 1, 2, 3
WORKSPACE_ALIGN = 256
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

# Host binding profile: the keyword set of the generated kernels (see
# :func:`stage_values`: the host provides, per stage, the kernel's own argument
# names; a kernel's argument plan selects from them).  A record with another
# ``abi`` (a plumbing placeholder program) is not served by this backend.
ABI_CONTRACT = "lm_head_loss_v1"
SUPPORTED_ABIS = (ABI_CONTRACT,)

# Geometry the kernels of a record were built for (record field ``geometry``;
# the defaults are the GLM-class contract).
GEOMETRY_DEFAULTS = dict(
    stats_tile=256,  # vocabulary columns per (max, sum-exp) partial: stats[C, V / stats_tile, 2]
    row_tile=128,  # output rows per CTA of the GEMMs (m_tiles = ceil(rows / row_tile), rounded to the CTA pair)
    cta_group=2,  # CTAs per GEMM cluster (the pair takes adjacent row tiles)
    k_block=64,  # K step of the GEMMs (k_iters of the weight-gradient GEMM = ceil(rows_c / k_block))
    cast_vec=8,  # elements per vector of the scale-cast kernels (num_vecs = numel / cast_vec)
    vocab_multiple=256,  # V % vocab_multiple == 0
    hidden_multiple=256,  # H % hidden_multiple == 0
    ld_multiple=8,  # row stride of X in elements (16 B TMA pitch)
    labels_dtype="int64",  # element type the row kernels read (``int32`` = one host cast per call)
    hidden=None,  # the GEMM instances are specialized to one H (None = any multiple of hidden_multiple)
    vocab=None,  # ... and one V (None = any multiple of vocab_multiple)
)
LABEL_DTYPES = {"int64": torch.int64, "int32": torch.int32}
_GEOMETRY_OPTIONAL = ("hidden", "vocab")

# Values the host provides to every stage of one chunk (the kernels' own
# argument names; the argument plan of a stage selects from them).  Tensors:
# ``A`` / ``B`` / ``C`` / ``STATS_OUT`` of the GEMMs (per stage: X_c / W / z_c /
# stats; dz_c / W / dX_acc rows; dz_c / X_c / dW_acc), ``stats``, ``z`` (the
# ``[C, V]`` BF16 workspace: ``z_c`` then ``dz_c`` in place), full-length ``[T]``
# vectors ``labels`` / ``lse`` / ``logp`` / ``infer_logp`` / ``loss_weights`` /
# ``d_in`` (indexed by ``row0 + r``), chunk-local ``d`` / ``term`` (``[C]``), the
# ``loss_acc`` / ``loss_out`` cells and, for the casts, ``acc`` / ``g`` / ``out``.
# Scalars: ``M`` / ``m_tiles`` / ``k_iters`` / ``first_chunk`` of the GEMMs,
# ``rows_c`` / ``row0`` / ``V`` / ``num_tiles`` / ``mode`` / ``loss_div`` /
# ``last_chunk`` / ``d_off`` of the row kernels, ``num_vecs`` of the casts, and
# ``T`` / ``H`` / ``C`` for grid rules.
STAGE_TENSORS = {
    "gemm_logits": ("A", "B", "C", "STATS_OUT"),
    "gemm_logits_nostats": ("A", "B", "C", "STATS_OUT"),
    "gemm_dx": ("A", "B", "C", "STATS_OUT"),
    "gemm_dw_acc": ("A", "B", "C", "STATS_OUT"),
    "row_finalize": ("stats", "z", "labels", "infer_logp", "loss_weights", "d_in", "lse", "logp", "d", "term"),
    "loss_reduce": ("term", "loss_acc", "loss_out"),
    "row_grad": ("z", "labels", "lse", "d"),
    "scale_cast_bf16": ("acc", "g", "out"),
    "scale_cast_f32": ("acc", "g", "out"),
}
COMMON_TENSORS = ("workspace", "tma_descriptor_workspace")
COMMON_SCALARS = ("rows_c", "row0", "T", "H", "V", "C", "num_tiles", "mode", "loss_div", "first_chunk", "last_chunk", "d_off")
# Accepted spellings of the same host value (kernel side -> host side).
CONTRACT_ALIASES = {
    "num_rows": "T",
    "total_rows": "T",
    "hidden": "H",
    "vocab": "V",
    "chunk": "C",
    "chunk_size": "C",
    "num_vocab_tiles": "num_tiles",
    "vocab_tiles": "num_tiles",
    "objective": "mode",
    "grad_scale": "g",
    "scale": "g",
    "src": "acc",
    "dst": "out",
    "z_c": "z",
    "logits": "z",
    "dz": "z",
    "dz_c": "z",
    "dlogits": "z",
    "dlogp": "d_in",
    "y": "labels",
    "targets": "labels",
    "weights": "loss_weights",
    "w": "loss_weights",
}


@dataclass(frozen=True)
class Geometry:
    stats_tile: int
    row_tile: int
    cta_group: int
    k_block: int
    cast_vec: int
    vocab_multiple: int
    hidden_multiple: int
    ld_multiple: int
    labels_dtype: torch.dtype
    hidden: Optional[int] = None
    vocab: Optional[int] = None

    @classmethod
    def from_record(cls, record: Optional[dict[str, Any]]) -> "Geometry":
        raw = dict(GEOMETRY_DEFAULTS)
        if record is not None:
            declared = record.get("geometry", {})
            unknown = sorted(set(declared) - set(GEOMETRY_DEFAULTS))
            if unknown:
                raise ValueError(f"registry record: unknown geometry fields {unknown}")
            raw.update(declared)
        for name in GEOMETRY_DEFAULTS:
            if name == "labels_dtype" or (name in _GEOMETRY_OPTIONAL and raw[name] is None):
                continue
            if isinstance(raw[name], bool) or int(raw[name]) < 1:
                raise ValueError(f"registry record: geometry {name} must be a positive integer")
        if raw["labels_dtype"] not in LABEL_DTYPES:
            raise ValueError(f"registry record: geometry labels_dtype must be one of {sorted(LABEL_DTYPES)}")
        return cls(
            **{name: int(raw[name]) for name in GEOMETRY_DEFAULTS if name != "labels_dtype" and name not in _GEOMETRY_OPTIONAL},
            labels_dtype=LABEL_DTYPES[raw["labels_dtype"]],
            hidden=None if raw["hidden"] is None else int(raw["hidden"]),
            vocab=None if raw["vocab"] is None else int(raw["vocab"]),
        )

    def row_tiles(self, rows: int) -> int:
        """``ceil(rows / row_tile)`` rounded up to a multiple of ``cta_group`` (the pair takes adjacent row tiles)."""
        tiles = -(-int(rows) // self.row_tile)
        return -(-tiles // self.cta_group) * self.cta_group

    def k_iters(self, rows_c: int) -> int:
        return -(-int(rows_c) // self.k_block)


DEFAULT_GEOMETRY = Geometry.from_record(None)


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def record_for(device: Optional[torch.device] = None) -> tuple[str, dict[str, Any]]:
    """``(module_name, record)`` registered for ``device``; raises when absent."""
    arch = arch_for(device)
    if arch is None:
        raise ValueError("the chunked LM-head + loss kernels require compute capability 10.0 or 10.3")
    name = select_module(arch)
    return name, MODULES[name]


def record_abi(record: dict[str, Any]) -> str:
    abi = str(record.get("abi", ABI_CONTRACT))
    if abi not in SUPPORTED_ABIS:
        raise NotImplementedError(f"unsupported host binding profile {abi!r}")
    return abi


def stages_for_entry(entry: str, *, need_dx: bool = True, need_dw: bool = True, grad_weight_dtype=torch.bfloat16) -> tuple[str, ...]:
    """Stages an entry point launches (``need_dx`` / ``need_dw`` drop the GEMMs of frozen inputs)."""
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    stages = ["gemm_logits", "row_finalize"]
    if entry == "loss":
        stages.append("loss_reduce")
    else:
        stages.append("gemm_logits_nostats")
    if need_dx or need_dw:
        stages.append("row_grad")
    if need_dx:
        stages += ["gemm_dx", "scale_cast_bf16"]
    if need_dw:
        stages += ["gemm_dw_acc", "scale_cast_f32" if (entry == "loss" and grad_weight_dtype == torch.float32) else "scale_cast_bf16"]
    return tuple(s for s in STAGES if s in stages)


def generated_program_available(device: Optional[torch.device] = None, *, entry: str = "loss") -> bool:
    """True when this checkout registers a contract program for ``device`` with every stage the entry point needs."""
    arch = arch_for(device)
    if arch is None:
        return False
    names = [n for n, r in MODULES.items() if r["arch"] == arch]
    if len(names) != 1:
        return False
    record = MODULES[names[0]]
    if str(record.get("abi", ABI_CONTRACT)) != ABI_CONTRACT:
        return False
    stages = registered_stages(names[0])
    needed = set(stages_for_entry(entry)) | set(stages_for_entry(entry, grad_weight_dtype=torch.float32))
    return all(stage in stages for stage in needed)


# ---------------------------------------------------------------------------
# Validation and planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """Validated geometry and options of one call."""

    num_rows: int  # T
    hidden: int  # H
    vocab: int  # V
    chunk: int  # C
    ld_x: int  # leading stride of X as launched (after a copy: H)
    x_copy: bool  # X is materialized contiguously (leading stride not a TMA pitch)
    objective: str
    loss_div: Optional[float]
    grad_weight_dtype: torch.dtype
    entry: str = "loss"

    @property
    def mode(self) -> int:
        """``mode`` of ``row_finalize`` in the forward."""
        if self.entry == "logprob":
            return MODE_NONE
        return MODE_CE if self.objective == "ce" else MODE_POLICY


def _positive_scalar(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if value.is_cuda:
            raise ValueError(f"{name} must be a Python number or a CPU scalar (a device value would synchronize)")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


def validate_lm_head_inputs(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    entry: str = "loss",
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> Problem:
    """Shape / dtype / stride validation shared by the entry points.

    Returns the :class:`Problem`.  Device placement is checked separately so
    this runs on host tensors.  ``T == 0`` is accepted (the eager entry points
    return zeros for it without launching).
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    if deterministic is not True:
        raise NotImplementedError("only deterministic=True is available: fixed chunk order, no atomics")
    if X.ndim != 2 or X.dtype != torch.bfloat16:
        raise ValueError("X must be a BF16 [T, H] tensor")
    if W.ndim != 2 or W.dtype != torch.bfloat16:
        raise ValueError("W must be a BF16 [V, H] tensor")
    if not W.is_contiguous():
        raise ValueError("W must be contiguous")
    num_rows, hidden = (int(s) for s in X.shape)
    vocab, w_hidden = (int(s) for s in W.shape)
    if w_hidden != hidden:
        raise ValueError(f"W has {w_hidden} columns, X has {hidden}: the hidden sizes differ")
    if hidden < 1 or hidden % geometry.hidden_multiple:
        raise ValueError(f"H must be a positive multiple of {geometry.hidden_multiple}, got {hidden}")
    if vocab < 1 or vocab % geometry.vocab_multiple:
        raise ValueError(f"V must be a positive multiple of {geometry.vocab_multiple}, got {vocab}")
    if geometry.hidden is not None and hidden != geometry.hidden:
        raise ValueError(f"the registered program is specialized to H = {geometry.hidden}, got {hidden}")
    if geometry.vocab is not None and vocab != geometry.vocab:
        raise ValueError(f"the registered program is specialized to V = {geometry.vocab}, got {vocab}")
    if X.stride(1) != 1:
        raise ValueError("X rows must be contiguous (stride(1) == 1)")
    ld_x = int(X.stride(0))
    if num_rows > 1 and ld_x < hidden:
        raise ValueError("X must be row-major with a leading stride of at least H")
    # The TMA operands need a 16 B row pitch and base; otherwise X is materialized contiguously (reported).
    x_copy = bool(num_rows and (ld_x < hidden or ld_x % geometry.ld_multiple or X.data_ptr() % 16))
    if x_copy or num_rows == 0:
        ld_x = hidden
    if labels.ndim != 1 or int(labels.shape[0]) != num_rows:
        raise ValueError("labels must be a [T] tensor")
    if labels.dtype != torch.int64:
        raise ValueError("labels must be int64 (-100 marks ignored rows)")
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    if int(chunk_size) > 65535:
        raise ValueError("chunk_size must not exceed 65535 (the row kernels index the chunk row by blockIdx.y)")
    if grad_weight_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("grad_weight_dtype must be torch.bfloat16 or torch.float32")
    if entry == "logprob":
        if objective != "ce" or loss_div is not None or infer_logp is not None or loss_weights is not None:
            raise ValueError("the log-probability entry point takes no objective arguments")
        return Problem(num_rows, hidden, vocab, int(chunk_size), ld_x, x_copy, "ce", None, grad_weight_dtype, entry)
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "ce":
        if loss_div is None:
            raise ValueError("objective='ce' needs loss_div (a caller-supplied positive scalar)")
        if infer_logp is not None or loss_weights is not None:
            raise ValueError("infer_logp and loss_weights belong to objective='policy'")
        loss_div = _positive_scalar(loss_div, "loss_div")
    else:
        if loss_div is not None:
            raise ValueError("loss_div belongs to objective='ce'")
        if infer_logp is None or loss_weights is None:
            raise ValueError("objective='policy' needs infer_logp and loss_weights")
        for name, t in (("infer_logp", infer_logp), ("loss_weights", loss_weights)):
            if t.ndim != 1 or int(t.shape[0]) != num_rows or t.dtype != torch.float32:
                raise ValueError(f"{name} must be an FP32 [T] tensor")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    return Problem(num_rows, hidden, vocab, int(chunk_size), ld_x, x_copy, objective, loss_div, grad_weight_dtype, entry)


def plan_chunks(num_rows: int, chunk: int) -> tuple[tuple[int, int], ...]:
    """``(row0, rows_c)`` of every chunk in launch order: full chunks first, the
    tail (``T % C``) last, never dropped, never padded; empty for ``T == 0``."""
    if int(chunk) < 1:
        raise ValueError("chunk must be positive")
    return tuple((row0, min(int(chunk), int(num_rows) - row0)) for row0 in range(0, int(num_rows), int(chunk)))


@dataclass(frozen=True)
class Plan:
    """The chunk schedule of one binding."""

    problem: Problem
    chunks: tuple[tuple[int, int], ...]
    need_dx: bool
    need_dw: bool
    geometry: Geometry = DEFAULT_GEOMETRY

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_tiles(self) -> int:
        return -(-self.problem.vocab // self.geometry.stats_tile)

    @property
    def stages(self) -> tuple[str, ...]:
        return stages_for_entry(
            self.problem.entry, need_dx=self.need_dx, need_dw=self.need_dw, grad_weight_dtype=self.problem.grad_weight_dtype
        )

    @property
    def dw_cast_stage(self) -> str:
        fp32 = self.problem.entry == "loss" and self.problem.grad_weight_dtype == torch.float32
        return "scale_cast_f32" if fp32 else "scale_cast_bf16"


def make_plan(problem: Problem, *, need_dx: bool, need_dw: bool, geometry: Geometry = DEFAULT_GEOMETRY) -> Plan:
    return Plan(problem, plan_chunks(problem.num_rows, problem.chunk), bool(need_dx), bool(need_dw), geometry)


# ---------------------------------------------------------------------------
# Workspace and memory accounting
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return (nbytes + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def workspace_layout(
    num_rows: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    The workspace holds the temporaries only the launches read: the
    vocabulary buffer ``logits`` (``min(T, C) x V`` BF16, ``z_c`` then ``dz_c``
    in place), its row-statistics partials ``stats``, the chunk-local per-row
    gradient scale ``d`` and loss terms ``term``, the loss accumulator and the
    ``grad_scale`` cell, plus the descriptor storage of pointer-ABI programs
    and kernel-private scratch when a record declares them.  The ``O(T)`` row
    vectors a caller or the autograd graph keeps (``lse``, ``logp``, the
    ``loss`` cell) and the FP32 accumulators ``dX_acc`` / ``dW_acc`` are
    allocated separately so that no returned or saved tensor pins the
    vocabulary buffer.
    """
    rows = min(int(chunk), max(int(num_rows), 1))  # the logits workspace never exceeds T rows either
    tiles = -(-int(vocab) // int(stats_tile))
    sizes = [
        ("logits", rows * int(vocab) * 2),
        ("stats", rows * tiles * 2 * 4),
        ("d", rows * 4),
        ("term", rows * 4),
        ("loss_acc", 4),
        ("grad_scale", 4),
    ]
    if scratch_bytes:
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


def _record_tma_bytes(record: dict[str, Any], stages) -> int:
    return max((int(record[s].get("tma_workspace_bytes", 0)) for s in stages), default=0)


def _record_scratch_bytes(record: dict[str, Any], stages) -> int:
    return max((int(record[s].get("workspace_bytes", 0)) for s in stages), default=0)


def memory_report(
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    return_logp: bool = False,
    x_copy: bool = False,
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
) -> dict[str, Any]:
    """The reporting buckets of the memory rule (bytes).

    ``temporary``: the workspace regions of :func:`workspace_layout`, the
    ``O(T)`` row vectors (``lse``; ``logp`` unless it is returned; the
    contiguous ``dlogp`` of the log-probability backward) and a contiguous copy
    of ``X`` when its leading stride forced one; ``outputs``: what the caller
    receives (``loss``, ``logp`` when requested, ``dX`` BF16, ``dW`` in
    ``grad_weight_dtype``); ``accumulators``: the FP32 ``dX_acc`` / ``dW_acc``
    the loss entry keeps alive between forward and backward (the
    log-probability entry allocates them in its backward only) and the row
    statistic the log-probability entry saves; ``weights``: the model tensors
    (``W``, ``X``).  ``vocab_rows_max`` is the largest token extent of any
    vocabulary-sized buffer (``min(T, C)``).
    """
    layout = workspace_layout(
        num_rows, vocab, chunk, stats_tile=stats_tile, entry=entry,
        tma_workspace_bytes=tma_workspace_bytes, scratch_bytes=scratch_bytes,
    )
    temporary = {name: size for name, (_, size) in ((k, v) for k, v in layout.items() if k != "total")}
    temporary["lse"] = int(num_rows) * 4
    if entry == "loss" and not return_logp:
        temporary["logp"] = int(num_rows) * 4
    if entry == "logprob":
        temporary["dlogp"] = int(num_rows) * 4
    if x_copy:
        temporary["x_copy"] = int(num_rows) * int(hidden) * 2
    outputs = {"loss": 4 if entry == "loss" else 0, "logp": int(num_rows) * 4 if (return_logp or entry == "logprob") else 0}
    if need_dx:
        outputs["dX"] = int(num_rows) * int(hidden) * 2
    if need_dw:
        outputs["dW"] = int(vocab) * int(hidden) * (4 if (entry == "loss" and grad_weight_dtype == torch.float32) else 2)
    accumulators = {}
    if need_dx:
        accumulators["dX_acc"] = int(num_rows) * int(hidden) * 4
    if need_dw:
        accumulators["dW_acc"] = int(vocab) * int(hidden) * 4
    if entry == "logprob":
        accumulators["saved_lse"] = int(num_rows) * 4
    weights = {"W": int(vocab) * int(hidden) * 2, "X": int(num_rows) * int(hidden) * 2}
    return dict(
        temporary_bytes=sum(temporary.values()),
        temporary=temporary,
        outputs_bytes=sum(outputs.values()),
        outputs=outputs,
        accumulator_bytes=sum(accumulators.values()),
        accumulators=accumulators,
        weights_bytes=sum(weights.values()),
        weights=weights,
        vocab_rows_max=min(int(chunk), max(int(num_rows), 1)),
        chunk=int(chunk),
        num_chunks=len(plan_chunks(num_rows, chunk)),
    )


def lm_head_loss_workspace_size(
    num_rows: int,
    vocab: int,
    chunk: int = DEFAULT_CHUNK_SIZE,
    device: Optional[torch.device] = None,
    *,
    entry: str = "loss",
    backend: str = "cake",
) -> int:
    """Workspace bytes :func:`prepare_lm_head_loss` needs for ``(T, V, C)`` on ``device``."""
    if backend == "reference":
        return int(workspace_layout(num_rows, vocab, chunk, entry=entry)["total"])
    name, record = record_for(device)
    stages = registered_stages(name)
    geometry = Geometry.from_record(record)
    return int(
        workspace_layout(
            num_rows, vocab, chunk, stats_tile=geometry.stats_tile, entry=entry,
            tma_workspace_bytes=_record_tma_bytes(record, stages), scratch_bytes=_record_scratch_bytes(record, stages),
        )["total"]
    )


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if needed > nbytes:
        raise ValueError(f"workspace region {name!r} holds {nbytes} bytes, {needed} needed")
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------

_GRID_FUNCTIONS = {"min": min, "max": max}


def grid_dims(rule, scalars: dict[str, Any], num_sms: int) -> tuple[int, int, int]:
    """Evaluate a registry grid rule.

    Each of the three entries is an integer or an integer expression over the
    scalar names of the stage's host values (``rows_c``, ``m_tiles``, ``V``,
    ``num_vecs``, ...) and ``sms`` (the SM count) with ``+``, ``-``, ``*``,
    ``/`` (rounds up), ``//`` (rounds down), parentheses and ``min`` / ``max``:
    ``"rows_c/8"`` (one CTA per eight rows), ``"max(1, min(m_tiles//2*605,
    sms//2))*2"`` (a persistent cluster grid capped by the SM pairs).
    """
    names = {"sms": int(num_sms)}

    def evaluate(node) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ValueError(f"grid rule constants must be integers, got {node.value!r}")
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id not in scalars or scalars[node.id] is None or isinstance(scalars[node.id], torch.Tensor):
                raise KeyError(f"grid rule names the unknown scalar {node.id!r}")
            return int(scalars[node.id])
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                if right == 0:
                    raise ValueError("grid rule divides by zero")
                return -(-left // right) if isinstance(node.op, ast.Div) else left // right
            raise ValueError(f"grid rule operator {type(node.op).__name__} is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _GRID_FUNCTIONS or node.keywords:
                raise ValueError("grid rule calls must be min(...) or max(...)")
            if len(node.args) < 2:
                raise ValueError("grid rule min()/max() take at least two terms")
            return _GRID_FUNCTIONS[node.func.id](evaluate(arg) for arg in node.args)
        raise ValueError(f"grid rule syntax {type(node).__name__} is not allowed")

    def term(value) -> int:
        if isinstance(value, bool):
            raise ValueError("grid rule entries must be integers or expressions")
        if isinstance(value, int):
            return int(value)
        try:
            tree = ast.parse(str(value).strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"grid rule entry {value!r} is not an expression") from exc
        return evaluate(tree)

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
    # Descriptor-preparation entry of a pointer-ABI module (same arguments).
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)
    # ``(argument index, host value name)`` of every tensor argument (re-bound per call by a remembered binding).
    slots: tuple[tuple[int, str], ...] = ()

    def __call__(self) -> None:
        self.entry(*self.arguments)

    def templated(self) -> "_Launch":
        """Copy whose tensor arguments are ``None`` placeholders (holds no tensor)."""
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


def bind_stage(module_name: str, stage: str, values: dict[str, Any], grid: tuple[int, int, int]) -> _Launch:
    """Order ``values`` by the generated argument plan of ``stage`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.  The returned launch records which
    argument positions hold tensors (re-bound per call by a remembered binding).
    """
    physical = MODULES[module_name][stage]
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    slots = []
    for kind, name in physical["arg_plan"]:
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif key in values and values[key] is not None:  # buffer / tma_buffer / workspace / parameter
            value = values[key]
            if isinstance(value, torch.Tensor):
                slots.append((len(arguments), key))
            arguments.append(value)
        else:
            raise KeyError(
                f"generated module {module_name!r} stage {stage!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_lm_head_loss_module(module_name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return _Launch(stage, module_name, getattr(module, physical["ffi_entry"]), tuple(arguments), grid, prepare, tuple(slots))


_FFI_DEVICES: dict[int, Any] = {}


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``."""
    import tvm_ffi

    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = getter(index) if getter is not None else torch.cuda.current_stream(index).cuda_stream
    return tvm_ffi.use_raw_stream(device, raw)


# ---------------------------------------------------------------------------
# Host values of the stages
# ---------------------------------------------------------------------------


def stage_values(stage: str, t: dict[str, Any], plan: Plan, index: int, *, order_key: Any = None) -> dict[str, Any]:
    """Host values of ``stage`` for chunk ``index`` (the kernel's own argument names).

    ``t`` holds the bound tensors: ``X`` (as launched), ``W``, ``labels``,
    ``lse``, ``logp``, ``infer_logp`` / ``loss_weights`` (policy), ``d_in``
    (the ``[T]`` incoming ``dlogp`` of the log-probability backward), the
    workspace regions ``logits`` / ``stats`` / ``d`` / ``term`` / ``loss_acc`` /
    ``grad_scale``, ``loss`` (the finished-loss cell), ``dx_acc`` / ``dw_acc``
    and the cast outputs ``dx_out`` / ``dw_out``.  For the casts ``index`` is
    ``"dx"`` or ``"dw"``.
    """
    p, g = plan.problem, plan.geometry
    values: dict[str, Any] = {name: t.get(name) for name in COMMON_TENSORS}
    values.update(T=int(p.num_rows), H=int(p.hidden), V=int(p.vocab), C=int(p.chunk), num_tiles=int(plan.num_tiles),
                  mode=int(p.mode), loss_div=float(p.loss_div) if p.loss_div is not None else 1.0)
    if stage in ("scale_cast_bf16", "scale_cast_f32"):
        acc = t["dx_acc"] if index == "dx" else t["dw_acc"]
        out = t["dx_out"] if index == "dx" else t["dw_out"]
        if acc.numel() != out.numel() or acc.numel() % g.cast_vec:
            raise ValueError("scale-cast operands must match in size and hold whole vectors")
        values.update(acc=acc.reshape(-1), g=t["grad_scale"], out=out.reshape(-1), num_vecs=int(acc.numel() // g.cast_vec),
                      rows_c=0, row0=0, first_chunk=0, last_chunk=0, d_off=0)
        return values
    row0, rows_c = plan.chunks[index]
    stop = row0 + rows_c
    first, last = int(index == 0), int(index == plan.num_chunks - 1)
    values.update(rows_c=int(rows_c), row0=int(row0), first_chunk=first, last_chunk=last, d_off=0)
    x_chunk = t["X"][row0:stop]
    logits = t["logits"]
    dz_chunk = logits[:rows_c]
    if stage in ("gemm_logits", "gemm_logits_nostats"):
        values.update(A=x_chunk, B=t["W"], C=logits, STATS_OUT=t["stats"], M=int(rows_c), m_tiles=g.row_tiles(rows_c), k_iters=1,
                      first_chunk=0)
    elif stage == "gemm_dx":
        values.update(A=dz_chunk, B=t["W"], C=t["dx_acc"][row0:stop], STATS_OUT=t["stats"], M=int(rows_c),
                      m_tiles=g.row_tiles(rows_c), k_iters=1, first_chunk=0)
    elif stage == "gemm_dw_acc":
        values.update(A=dz_chunk, B=x_chunk, C=t["dw_acc"], STATS_OUT=t["stats"], M=int(p.vocab), m_tiles=g.row_tiles(p.vocab),
                      k_iters=g.k_iters(rows_c), first_chunk=first)
    elif stage == "row_finalize":
        d = t["d"]

        def present(name):  # policy / external operands: the chunk-local ``d`` stands in when the mode never reads them
            return d if t.get(name) is None else t[name]

        values.update(stats=t["stats"], z=logits, labels=t["labels"], infer_logp=present("infer_logp"),
                      loss_weights=present("loss_weights"), d_in=present("d_in"), lse=t["lse"], logp=t["logp"], d=d, term=t["term"])
    elif stage == "loss_reduce":
        values.update(term=t["term"], loss_acc=t["loss_acc"], loss_out=t["loss"])
    elif stage == "row_grad":
        external = p.entry == "logprob"  # the recompute reads the caller's [T] dlogp at d[row0 + r]
        values.update(z=logits, labels=t["labels"], lse=t["lse"], d=t["d_in"] if external else t["d"], d_off=int(row0) if external else 0)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return values


def forward_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys ``(stage, chunk index)`` of the forward chunk loop."""
    keys = []
    for index in range(plan.num_chunks):
        keys += [("gemm_logits", index), ("row_finalize", index)]
        if plan.problem.entry == "loss":
            keys.append(("loss_reduce", index))
            if plan.need_dx or plan.need_dw:
                keys.append(("row_grad", index))
            if plan.need_dx:
                keys.append(("gemm_dx", index))
            if plan.need_dw:
                keys.append(("gemm_dw_acc", index))
    return tuple(keys)


def recompute_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the log-probability backward (recompute the logits, then the gradient GEMMs)."""
    keys = []
    for index in range(plan.num_chunks):
        keys += [("gemm_logits_nostats", index), ("row_grad", index)]
        if plan.need_dx:
            keys.append(("gemm_dx", index))
        if plan.need_dw:
            keys.append(("gemm_dw_acc", index))
    return tuple(keys)


def cast_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    keys = []
    if plan.need_dx:
        keys.append(("scale_cast_bf16", "dx"))
    if plan.need_dw:
        keys.append((plan.dw_cast_stage, "dw"))
    return tuple(keys)


# ---------------------------------------------------------------------------
# Reference engine: the stages as PyTorch operators over the same host values
# ---------------------------------------------------------------------------


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of BF16 operands with FP32 accumulation and an FP32 result (no intermediate rounding)."""
    if a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
    return torch.mm(a.float(), b.float())


class ReferenceEngine:
    """PyTorch implementation of every stage at the kernels' rounding boundaries.

    Each method consumes the host values a generated stage is bound to
    (:func:`stage_values`), so the chunk loop, the plans and the autograd
    Functions are exercised without a generated program.  Not
    allocation-free (a development aid).
    """

    def run(self, stage: str, values: dict[str, Any]) -> None:
        getattr(self, stage)(values)

    @staticmethod
    def _rows(values: dict[str, Any]) -> tuple[int, int]:
        return int(values["row0"]), int(values["rows_c"])

    @staticmethod
    def _valid(values: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        row0, rows_c = ReferenceEngine._rows(values)
        labels = values["labels"][row0 : row0 + rows_c].long()
        return labels, labels >= 0

    @staticmethod
    def gemm_logits(values: dict[str, Any], stats: bool = True) -> None:
        rows = int(values["M"])
        z = values["C"][:rows]
        torch.mm(values["A"], values["B"].t(), out=z)  # BF16 result of an FP32 accumulation
        if stats:
            tiles = int(values["num_tiles"])
            zf = z.float().view(rows, tiles, -1)
            pmax = zf.amax(-1)
            psum = torch.exp(zf - pmax[..., None]).sum(-1)
            out = values["STATS_OUT"][:rows]
            out[..., 0] = pmax
            out[..., 1] = psum

    @staticmethod
    def gemm_logits_nostats(values: dict[str, Any]) -> None:
        ReferenceEngine.gemm_logits(values, stats=False)

    @staticmethod
    def row_finalize(values: dict[str, Any]) -> None:
        row0, rows_c = ReferenceEngine._rows(values)
        stop = row0 + rows_c
        stats = values["stats"][:rows_c]
        pmax, psum = stats[..., 0], stats[..., 1]
        m = pmax.amax(-1)
        lse = m + torch.log((psum * torch.exp(pmax - m[:, None])).sum(-1))
        labels, valid = ReferenceEngine._valid(values)
        index = torch.where(valid, labels, torch.zeros_like(labels)).clamp_max(int(values["V"]) - 1)
        zy = values["z"][:rows_c].gather(1, index[:, None]).squeeze(1).float()
        zero = torch.zeros_like(lse)
        logp = torch.where(valid, zy - lse, zero)
        values["lse"][row0:stop] = lse
        values["logp"][row0:stop] = logp
        mode = int(values["mode"])
        d, term = zero, zero
        if mode == MODE_CE:
            d = torch.where(valid, torch.full_like(lse, -1.0 / float(values["loss_div"])), zero)
            term = logp
        elif mode == MODE_POLICY:
            ratio = torch.exp(logp - values["infer_logp"][row0:stop])
            w = values["loss_weights"][row0:stop]
            d = torch.where(valid & (ratio <= RATIO_CLIP), -w * ratio, zero)
            term = torch.where(valid, w * torch.clamp_max(ratio, RATIO_CLIP), zero)
        elif mode == MODE_EXTERNAL:
            d = torch.where(valid, values["d_in"][row0:stop], zero)
        values["d"][:rows_c] = d
        values["term"][:rows_c] = term

    @staticmethod
    def loss_reduce(values: dict[str, Any]) -> None:
        rows_c = int(values["rows_c"])
        total = values["term"][:rows_c].sum().reshape(1)
        acc = values["loss_acc"]
        if int(values["first_chunk"]):
            acc.copy_(total)
        else:
            acc.add_(total)
        if int(values["last_chunk"]):
            neg = -acc
            values["loss_out"].copy_(neg / float(values["loss_div"]) if int(values["mode"]) == MODE_CE else neg)

    @staticmethod
    def row_grad(values: dict[str, Any]) -> None:
        row0, rows_c = ReferenceEngine._rows(values)
        z = values["z"][:rows_c]
        labels, valid = ReferenceEngine._valid(values)
        lse = values["lse"][row0 : row0 + rows_c]
        d_off = int(values["d_off"])
        d = torch.where(valid, values["d"][d_off : d_off + rows_c], torch.zeros_like(lse))
        p = torch.exp(z.float() - lse[:, None])
        index = torch.where(valid, labels, torch.zeros_like(labels)).clamp_max(int(values["V"]) - 1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, index[:, None], 1.0)
        dz = d[:, None] * (onehot - p)
        dz[~valid] = 0.0
        z.copy_(dz)  # the BF16 dlogits boundary

    @staticmethod
    def gemm_dx(values: dict[str, Any]) -> None:
        values["C"].copy_(_mm_fp32(values["A"], values["B"]))

    @staticmethod
    def gemm_dw_acc(values: dict[str, Any]) -> None:
        product = _mm_fp32(values["A"].t(), values["B"])
        if int(values["first_chunk"]):
            values["C"].copy_(product)
        else:
            values["C"].add_(product)

    @staticmethod
    def scale_cast_bf16(values: dict[str, Any]) -> None:
        values["out"].copy_(values["acc"] * values["g"])  # one rounding at the copy

    scale_cast_f32 = scale_cast_bf16


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class LmHeadLossRunner:
    """The prepared launches of one tensor binding.

    Loss entry: ``forward()`` runs the chunk loop (three GEMMs per chunk) and
    returns ``(loss [1], logp [T])`` while filling ``dx_acc`` (FP32 ``[T, H]``)
    and ``dw_acc`` (FP32 ``[V, H]``); ``backward(grad)`` casts ``dX = bf16(g *
    dX_acc)`` into ``dx_out`` and ``dW = cast(g * dW_acc)`` into ``dw_out``
    (``g`` from the ``[1]`` FP32 ``grad_scale`` cell; ``None`` = 1).
    Log-probability entry: ``forward()`` returns ``(logp, lse)``;
    ``backward()`` reads ``dlogp`` (the bound FP32 ``[T]`` tensor
    ``tensors["d_in"]``, masked to valid rows by the row kernel), recomputes
    the logits per chunk and produces ``dx_out`` / ``dw_out`` (BF16) through
    the accumulators.  No launch of the ``cake`` backend allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype, stride, option or tensor binding changes;
    values may change freely.
    """

    backend: str
    module_name: Optional[str]
    plan: Plan
    tensors: dict[str, Any] = field(repr=False)
    values: dict[Any, dict[str, Any]] = field(repr=False)
    launches: dict[Any, _Launch] = field(repr=False)
    forward_order: tuple = ()
    backward_order: tuple = ()
    device_index: int = 0
    workspace: Optional[torch.Tensor] = field(default=None, repr=False)
    layout: dict = field(default_factory=dict, repr=False)
    memory: dict = field(default_factory=dict, repr=False)
    engine: Optional[ReferenceEngine] = field(default=None, repr=False)
    _tma_prepared: bool = False

    @property
    def problem(self) -> Problem:
        return self.plan.problem

    @property
    def loss(self) -> torch.Tensor:
        return self.tensors["loss"]

    @property
    def logp(self) -> torch.Tensor:
        return self.tensors["logp"]

    @property
    def lse(self) -> torch.Tensor:
        return self.tensors["lse"]

    @property
    def dx_acc(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dx_acc")

    @property
    def dw_acc(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dw_acc")

    @property
    def dx_out(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dx_out")

    @property
    def dw_out(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dw_out")

    @property
    def dlogp(self) -> Optional[torch.Tensor]:
        return self.tensors.get("d_in")

    @property
    def stages(self) -> tuple[str, ...]:
        return self.plan.stages

    def prepare_tma(self) -> None:
        """Encode the descriptors of pointer-ABI stages once (idempotent)."""
        if self._tma_prepared or self.backend != "cake":
            return
        with _ffi_stream_context(self.device_index):
            for launch in self.launches.values():
                if launch.prepare is not None:
                    launch.prepare(*launch.arguments)
        self._tma_prepared = True

    def _run(self, keys: tuple) -> None:
        if self.backend == "cake":
            self.prepare_tma()
            with _ffi_stream_context(self.device_index):
                for key in keys:
                    self.launches[key]()
        else:
            for key in keys:
                self.engine.run(key[0], self.values[key])

    def forward(self):
        """Loss entry: ``(loss [1], logp [T])``; log-probability entry: ``(logp, lse)``."""
        t = self.tensors
        self._run(self.forward_order)
        return (t["loss"], t["logp"]) if self.problem.entry == "loss" else (t["logp"], t["lse"])

    def backward(self, grad: Optional[torch.Tensor] = None):
        """``(dx_out, dw_out)`` (``None`` for a frozen input).

        Loss entry: ``grad`` is the incoming scalar gradient (``None`` = 1);
        the saved accumulators are read, never written.  Log-probability
        entry: the caller has written ``dlogp`` into the bound ``dlogp``
        tensor; ``grad`` must be ``None`` (the cast scale is 1).
        """
        t = self.tensors
        if self.problem.entry == "logprob" and grad is not None:
            raise ValueError("the log-probability runner takes its gradient from the bound dlogp tensor")
        if grad is None:
            t["grad_scale"].fill_(1.0)
        else:
            t["grad_scale"].copy_(grad.reshape(1).to(torch.float32))
        self._run(self.backward_order)
        return t.get("dx_out"), t.get("dw_out")

    def step(self, grad: Optional[torch.Tensor] = None):
        self.forward()
        return self.backward(grad)

    __call__ = step


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}")


def _prepare_x(X: torch.Tensor, problem: Problem) -> torch.Tensor:
    return X.contiguous() if problem.x_copy else X


def _prepare_labels(labels: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    labels = labels.contiguous()
    if geometry.labels_dtype != labels.dtype:
        labels = labels.to(geometry.labels_dtype)  # one O(T) cast per call when the kernels read int32
    return labels


def _bind_all(record, module_name, keys, values, device) -> dict[Any, _Launch]:
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    launches: dict[Any, _Launch] = {}
    for key in keys:
        stage = key[0]
        physical = record[stage]
        grid = grid_dims(physical.get("grid", ["rows_c", 1, 1]), values[key], num_sms)
        cluster = physical.get("launch", {}).get("cluster")
        if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
            raise ValueError(
                f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
            )
        launches[key] = bind_stage(module_name, stage, values[key], grid)
    return launches


def prepare_lm_head_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    entry: str = "loss",
    dlogp: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    dx_acc: Optional[torch.Tensor] = None,
    dw_acc: Optional[torch.Tensor] = None,
    dx_out: Optional[torch.Tensor] = None,
    dw_out: Optional[torch.Tensor] = None,
    logp: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> LmHeadLossRunner:
    """Validate one binding and prepare its launches.

    Missing outputs, accumulators and the workspace are allocated here (the
    only allocations of the ``cake`` backend).  Pass ``workspace_buffer`` of
    :func:`lm_head_loss_workspace_size` bytes to reuse storage across steps.
    For the log-probability entry ``dlogp`` is the FP32 ``[T]`` tensor the
    backward reads (allocated here when absent) and ``lse`` the statistic the
    forward writes / the backward reads.  ``T == 0`` is rejected here (the
    eager entry points return zeros for it without a runner).
    ``backend="reference"`` builds the same runner over PyTorch operators.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    device = X.device
    record = None
    module_name = None
    if backend == "cake":
        if not X.is_cuda:
            raise ValueError("the cake backend needs CUDA tensors")
        module_name, record = record_for(device)
        record_abi(record)
    geometry = Geometry.from_record(record)
    problem = validate_lm_head_inputs(
        X, W, labels, objective=objective, loss_div=loss_div, infer_logp=infer_logp, loss_weights=loss_weights,
        chunk_size=chunk_size, grad_weight_dtype=grad_weight_dtype, deterministic=deterministic, entry=entry, geometry=geometry,
    )
    if problem.num_rows == 0:
        raise ValueError("a prepared runner needs at least one row (T > 0); the eager entry points return zeros for T == 0")
    given = [t for t in (W, labels, infer_logp, loss_weights, dlogp, workspace_buffer, dx_acc, dw_acc, dx_out, dw_out, logp, lse) if t is not None]
    if not all(t.device == device for t in given):
        raise ValueError("Expected all tensors on one device")
    plan = make_plan(problem, need_dx=need_dx, need_dw=need_dw, geometry=geometry)
    stages = registered_stages(module_name) if record is not None else tuple(STAGES)
    missing = [s for s in plan.stages if s not in stages]
    if missing:
        raise NotImplementedError(f"the registered program {module_name!r} lacks the stages {missing} (registered: {stages})")
    T, H, V, C = problem.num_rows, problem.hidden, problem.vocab, problem.chunk
    tma_bytes = _record_tma_bytes(record, stages) if record is not None else 0
    scratch_bytes = _record_scratch_bytes(record, stages) if record is not None else 0
    layout = workspace_layout(T, V, C, stats_tile=geometry.stats_tile, entry=entry, tma_workspace_bytes=tma_bytes, scratch_bytes=scratch_bytes)
    if workspace_buffer is None:
        workspace_buffer = torch.empty(layout["total"], dtype=torch.uint8, device=device)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}")

    def output(given, name, shape, dtype):
        _check_output(given, name, shape, dtype)
        return given if given is not None else torch.empty(shape, dtype=dtype, device=device)

    t: dict[str, Any] = dict(X=_prepare_x(X, problem), W=W, labels=_prepare_labels(labels, geometry))
    rows = layout["logits"][1] // (V * 2)
    t["logits"] = _carve(flat, layout, "logits", torch.bfloat16, (rows, V))
    t["stats"] = _carve(flat, layout, "stats", torch.float32, (rows, plan.num_tiles, 2))
    t["d"] = _carve(flat, layout, "d", torch.float32, (rows,))
    t["term"] = _carve(flat, layout, "term", torch.float32, (rows,))
    t["loss_acc"] = _carve(flat, layout, "loss_acc", torch.float32, (1,))
    t["grad_scale"] = _carve(flat, layout, "grad_scale", torch.float32, (1,))
    # O(T) row vectors and the loss cell are separate allocations: a caller (or the autograd graph) keeping
    # ``loss`` / ``logp`` / ``lse`` alive must not pin the vocabulary workspace.
    t["lse"] = output(lse, "lse", (T,), torch.float32)
    t["logp"] = output(logp, "logp", (T,), torch.float32)
    t["loss"] = torch.empty((1,), dtype=torch.float32, device=device)
    if entry == "loss":
        if infer_logp is not None:
            t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
    else:
        t["d_in"] = output(dlogp, "dlogp", (T,), torch.float32)
    if plan.need_dx:
        t["dx_acc"] = output(dx_acc, "dx_acc", (T, H), torch.float32)
        t["dx_out"] = output(dx_out, "dx_out", (T, H), torch.bfloat16)
    if plan.need_dw:
        t["dw_acc"] = output(dw_acc, "dw_acc", (V, H), torch.float32)
        t["dw_out"] = output(dw_out, "dw_out", (V, H), torch.float32 if plan.dw_cast_stage == "scale_cast_f32" else torch.bfloat16)
    if layout.get("workspace"):
        t["workspace"] = _carve(flat, layout, "workspace", torch.uint8, (layout["workspace"][1],))
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = _carve(flat, layout, "tma_descriptor_workspace", torch.uint8, (layout["tma_descriptor_workspace"][1],))

    fwd = forward_keys(plan)
    bwd = (recompute_keys(plan) if entry == "logprob" else ()) + cast_keys(plan)
    values = {key: stage_values(key[0], t, plan, key[1]) for key in fwd + bwd}
    launches: dict[Any, _Launch] = {}
    engine = None
    if backend == "cake":
        launches = _bind_all(record, module_name, fwd + bwd, values, device)
    else:
        engine = ReferenceEngine()
    memory = memory_report(
        T, H, V, C, stats_tile=geometry.stats_tile, entry=entry, need_dx=plan.need_dx, need_dw=plan.need_dw,
        grad_weight_dtype=problem.grad_weight_dtype, return_logp=True, x_copy=problem.x_copy,
        tma_workspace_bytes=tma_bytes, scratch_bytes=scratch_bytes,
    )
    if device.type == "cuda":
        device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    else:
        device_index = 0
    return LmHeadLossRunner(
        backend=backend, module_name=module_name, plan=plan, tensors=t, values=values, launches=launches,
        forward_order=fwd, backward_order=bwd, device_index=device_index, workspace=flat, layout=layout, memory=memory, engine=engine,
    )


# ---------------------------------------------------------------------------
# Binding cache of the eager entry points (cake backend)
# ---------------------------------------------------------------------------

BINDING_CACHE_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE"  # "0" disables the cache at import
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY"
BINDING_CACHE_DEFAULT_CAPACITY = 64


def binding_cache_capacity() -> int:
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    if raw is None or not raw.strip():
        return BINDING_CACHE_DEFAULT_CAPACITY
    try:
        capacity = int(raw)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}")
    return capacity


def _meta(t: Optional[torch.Tensor]):
    return None if t is None else (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def forward_binding_key(X, W, labels, *, objective, loss_div, infer_logp, loss_weights, chunk_size, need_dx, need_dw, grad_weight_dtype, entry) -> tuple:
    """Cache key of a forward binding: ``(data_ptr, shape, stride, dtype)`` of every
    input plus every option that shapes the argument plans."""
    return ("fwd", entry, _meta(X), _meta(W), _meta(labels), _meta(infer_logp), _meta(loss_weights), objective,
            None if loss_div is None else float(loss_div), int(chunk_size), bool(need_dx), bool(need_dw), grad_weight_dtype)


def logprob_backward_binding_key(X, W, labels, lse, dlogp, *, chunk_size, need_dx, need_dw) -> tuple:
    return ("bwd", "logprob", _meta(X), _meta(W), _meta(labels), _meta(lse), _meta(dlogp), int(chunk_size), bool(need_dx), bool(need_dw))


# Values a remembered binding owns: the descriptor workspace of pointer-ABI stages and kernel-private scratch (a few
# kilobytes).  The vocabulary workspace, the row vectors, the accumulators and the outputs are allocated per call from
# the caching allocator (their sizes are part of the plan; the allocator reuses the blocks).
_OWNED_VALUES = ("workspace", "tma_descriptor_workspace")


@dataclass
class _Binding:
    """One remembered input binding: templated launches (no tensor pinned) plus the plan."""

    plan: Plan
    device: torch.device
    device_index: int
    launches: dict[Any, _Launch] = field(repr=False)
    forward_order: tuple = ()
    backward_order: tuple = ()
    owned: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    owned_bytes: int = 0
    layout: dict = field(default_factory=dict, repr=False)
    memory: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_runner(cls, runner: LmHeadLossRunner) -> "_Binding":
        if runner.backend != "cake":
            raise ValueError("only cake-backend runners are remembered")
        t = runner.tensors
        owned = {name: torch.empty_like(t[name]) for name in _OWNED_VALUES if name in t}
        return cls(
            plan=runner.plan, device=t["X"].device, device_index=runner.device_index,
            launches={key: launch.templated() for key, launch in runner.launches.items()},
            forward_order=runner.forward_order, backward_order=runner.backward_order, owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()), layout=runner.layout, memory=runner.memory,
        )

    def holds_no_tensor(self) -> bool:
        return not any(isinstance(a, torch.Tensor) for launch in self.launches.values() for a in launch.arguments)

    def _scratch(self, t: dict[str, Any]) -> None:
        """Per-call temporaries from the caching allocator into ``t`` (the runner's regions, minus the owned ones)."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        t["logits"] = torch.empty((rows, p.vocab), dtype=torch.bfloat16, device=self.device)
        t["stats"] = torch.empty((rows, self.plan.num_tiles, 2), dtype=torch.float32, device=self.device)
        t["d"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        t["term"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        t["loss_acc"] = torch.empty((1,), dtype=torch.float32, device=self.device)
        t["loss"] = torch.empty((1,), dtype=torch.float32, device=self.device)
        t["grad_scale"] = torch.ones((1,), dtype=torch.float32, device=self.device)
        for name, tensor in self.owned.items():
            t[name] = tensor

    def _launch(self, keys: tuple, t: dict[str, Any]) -> None:
        with _ffi_stream_context(self.device_index):
            for key in keys:
                launch = self.launches[key]
                arguments = launch.arguments_for(stage_values(key[0], t, self.plan, key[1]))
                if launch.prepare is not None:  # descriptors of a pointer-ABI stage see the fresh tensors
                    launch.prepare(*arguments)
                launch.entry(*arguments)

    def forward(self, X, W, labels, infer_logp, loss_weights):
        p, plan = self.plan.problem, self.plan
        T, H, V = p.num_rows, p.hidden, p.vocab
        t: dict[str, Any] = dict(X=_prepare_x(X, p), W=W, labels=_prepare_labels(labels, plan.geometry))
        self._scratch(t)
        t["lse"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        t["logp"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        if p.entry == "loss":
            if infer_logp is not None:
                t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
            if plan.need_dx:
                t["dx_acc"] = torch.empty((T, H), dtype=torch.float32, device=self.device)
            if plan.need_dw:
                t["dw_acc"] = torch.empty((V, H), dtype=torch.float32, device=self.device)
        self._launch(self.forward_order, t)
        if p.entry == "loss":
            return t["loss"], t["logp"], t.get("dx_acc"), t.get("dw_acc")
        return t["logp"], t["lse"]

    def backward_logprob(self, X, W, labels, lse, dlogp):
        p, plan = self.plan.problem, self.plan
        T, H, V = p.num_rows, p.hidden, p.vocab
        t: dict[str, Any] = dict(X=_prepare_x(X, p), W=W, labels=_prepare_labels(labels, plan.geometry), lse=lse, d_in=dlogp)
        self._scratch(t)
        if plan.need_dx:
            t["dx_acc"] = torch.empty((T, H), dtype=torch.float32, device=self.device)
            t["dx_out"] = torch.empty((T, H), dtype=torch.bfloat16, device=self.device)
        if plan.need_dw:
            t["dw_acc"] = torch.empty((V, H), dtype=torch.float32, device=self.device)
            t["dw_out"] = torch.empty((V, H), dtype=torch.bfloat16, device=self.device)
        self._launch(self.backward_order, t)
        return t.get("dx_out"), t.get("dw_out")


class BindingCache:
    """Remembered input bindings of the eager entry points (least recently used, bounded)."""

    def __init__(self, capacity: Optional[int] = None, enabled: bool = True):
        self.capacity = binding_cache_capacity() if capacity is None else int(capacity)
        if self.capacity < 1:
            raise ValueError("BindingCache needs a capacity of at least one binding (use enabled=False to bypass it)")
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
        binding = self._bindings.get(key)
        if binding is None:
            self.misses += 1
        else:
            self.hits += 1
            self._bindings.move_to_end(key)
        return binding

    def peek(self, key: tuple) -> Optional[_Binding]:
        return self._bindings.get(key)

    def remember(self, key: tuple, binding: _Binding) -> _Binding:
        self._bindings.pop(key, None)
        self._bindings[key] = binding
        while len(self._bindings) > self.capacity:
            self._bindings.popitem(last=False)
        return binding


BINDING_CACHE = BindingCache(enabled=os.environ.get(BINDING_CACHE_ENV, "1") != "0")


# ---------------------------------------------------------------------------
# Eager entry points (allocate, launch, return)
# ---------------------------------------------------------------------------


@dataclass
class ForwardResult:
    loss: Optional[torch.Tensor]  # FP32 [] (loss entry)
    logp: torch.Tensor  # FP32 [T]
    lse: Optional[torch.Tensor] = None  # FP32 [T] (log-probability entry: the saved statistic)
    dx_acc: Optional[torch.Tensor] = None  # FP32 [T, H] (loss entry, X trainable)
    dw_acc: Optional[torch.Tensor] = None  # FP32 [V, H] (loss entry, W trainable)
    memory: Optional[dict] = None
    backend: str = "cake"


def _empty_forward(problem: Problem, X: torch.Tensor, *, need_dx: bool, need_dw: bool) -> ForwardResult:
    device = X.device
    memory = memory_report(0, problem.hidden, problem.vocab, problem.chunk, entry=problem.entry, need_dx=need_dx, need_dw=need_dw,
                           grad_weight_dtype=problem.grad_weight_dtype, return_logp=True)
    empty = torch.zeros((0,), dtype=torch.float32, device=device)
    if problem.entry == "loss":
        return ForwardResult(
            loss=torch.zeros((), dtype=torch.float32, device=device), logp=empty,
            dx_acc=torch.zeros((0, problem.hidden), dtype=torch.float32, device=device) if need_dx else None,
            dw_acc=torch.zeros((problem.vocab, problem.hidden), dtype=torch.float32, device=device) if need_dw else None, memory=memory,
        )
    return ForwardResult(loss=None, logp=empty, lse=empty.clone(), memory=memory)


def forward_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    backend: str = "cake",
) -> ForwardResult:
    """Forward of the loss entry: ``loss`` (FP32 scalar), ``logp`` (FP32 ``[T]``) and the
    FP32 gradient accumulators ``dx_acc`` / ``dw_acc`` of the trainable inputs.

    The first call for an input binding validates and binds through
    :func:`prepare_lm_head_loss`; later calls with the same binding take the
    remembered launches (:data:`BINDING_CACHE`).  A call without rows returns
    loss 0, an empty ``logp`` and zero accumulators without binding or launching.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X, W, labels, objective=objective, loss_div=loss_div, infer_logp=infer_logp, loss_weights=loss_weights,
        chunk_size=chunk_size, grad_weight_dtype=grad_weight_dtype, deterministic=deterministic, entry="loss",
    )
    if problem.num_rows == 0:
        return _empty_forward(problem, X, need_dx=need_dx, need_dw=need_dw)
    common = dict(objective=objective, loss_div=problem.loss_div, infer_logp=infer_logp, loss_weights=loss_weights,
                  chunk_size=chunk_size, need_dx=need_dx, need_dw=need_dw, grad_weight_dtype=grad_weight_dtype)
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(X, W, labels, entry="loss", **common)
        binding = cache.lookup(key)
        if binding is not None:
            loss, logp, dx_acc, dw_acc = binding.forward(X, W, labels, infer_logp, loss_weights)
            return ForwardResult(loss=loss.reshape(()), logp=logp, dx_acc=dx_acc, dw_acc=dw_acc, memory=binding.memory, backend=backend)
    runner = prepare_lm_head_loss(X, W, labels, deterministic=deterministic, entry="loss", backend=backend, **common)
    loss, logp = runner.forward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return ForwardResult(loss=loss.reshape(()), logp=logp, dx_acc=runner.dx_acc, dw_acc=runner.dw_acc, memory=runner.memory, backend=backend)


def scale_cast(acc: torch.Tensor, grad: Optional[torch.Tensor], out_dtype: torch.dtype, *, backend: str = "cake") -> torch.Tensor:
    """``cast(g * acc)`` into a new tensor of ``out_dtype`` (one rounding); ``acc`` is never written.

    The single fp32 -> output cast of the backward, for ``dW`` and ``dX`` alike
    (``g`` = the incoming scalar gradient, ``None`` = 1).
    """
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    if not acc.is_contiguous() or acc.dtype != torch.float32:
        raise ValueError("the accumulator must be a contiguous FP32 tensor")
    out = torch.empty(acc.shape, dtype=out_dtype, device=acc.device)
    if acc.numel() == 0:
        return out
    g = torch.ones((1,), dtype=torch.float32, device=acc.device) if grad is None else grad.detach().reshape(1).to(device=acc.device, dtype=torch.float32)
    if backend == "reference":
        out.reshape(-1).copy_(acc.reshape(-1) * g)
        return out
    module_name, record = record_for(acc.device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    stage = "scale_cast_f32" if out_dtype == torch.float32 else "scale_cast_bf16"
    if acc.numel() % geometry.cast_vec:
        raise ValueError(f"the cast needs a multiple of {geometry.cast_vec} elements")
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(acc=acc.reshape(-1), g=g, out=out.reshape(-1), num_vecs=int(acc.numel() // geometry.cast_vec), loss_div=1.0)
    launches = _bind_all(record, module_name, ((stage, "eager"),), {(stage, "eager"): values}, acc.device)
    launch = launches[(stage, "eager")]
    index = acc.device.index if acc.device.index is not None else torch.cuda.current_device()
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*launch.arguments)
        launch()
    return out


def backward_loss(
    dx_acc: Optional[torch.Tensor],
    dw_acc: Optional[torch.Tensor],
    grad: Optional[torch.Tensor] = None,
    *,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cake",
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the loss entry from the saved accumulators: ``(dX, dW)`` = ``bf16(g * dX_acc)``,
    ``cast(g * dW_acc)``; the accumulators are read, never written (repeatable)."""
    dx = None if dx_acc is None else scale_cast(dx_acc, grad, torch.bfloat16, backend=backend)
    dw = None if dw_acc is None else scale_cast(dw_acc, grad, grad_weight_dtype, backend=backend)
    return dx, dw


def forward_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    deterministic: bool = True,
    backend: str = "cake",
) -> ForwardResult:
    """Forward of the log-probability entry: ``logp`` (FP32 ``[T]``, 0 on ignored rows) plus the saved statistic ``lse``."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(X, W, labels, chunk_size=chunk_size, deterministic=deterministic, entry="logprob")
    if problem.num_rows == 0:
        return _empty_forward(problem, X, need_dx=False, need_dw=False)
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(X, W, labels, objective="ce", loss_div=None, infer_logp=None, loss_weights=None, chunk_size=chunk_size,
                                  need_dx=False, need_dw=False, grad_weight_dtype=torch.bfloat16, entry="logprob")
        binding = cache.lookup(key)
        if binding is not None:
            logp, lse = binding.forward(X, W, labels, None, None)
            return ForwardResult(loss=None, logp=logp, lse=lse, memory=binding.memory, backend=backend)
    runner = prepare_lm_head_loss(X, W, labels, chunk_size=chunk_size, need_dx=False, need_dw=False, deterministic=deterministic,
                                  entry="logprob", backend=backend)
    logp, lse = runner.forward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return ForwardResult(loss=None, logp=logp, lse=lse, memory=runner.memory, backend=backend)


def backward_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    deterministic: bool = True,
    backend: str = "cake",
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the log-probability entry: recompute each chunk's logits, ``dz = dlogp_t *
    (1[v = y_t] - softmax)`` (ignored rows zero), then ``dX`` (BF16) and ``dW`` (BF16) through
    the FP32 accumulators.  ``(None, None)`` when neither input is trainable."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(X, W, labels, chunk_size=chunk_size, deterministic=deterministic, entry="logprob")
    if not (need_dx or need_dw):
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    if T == 0:
        return (torch.zeros((0, H), dtype=torch.bfloat16, device=X.device) if need_dx else None,
                torch.zeros((V, H), dtype=torch.bfloat16, device=X.device) if need_dw else None)
    _check_output(lse, "lse", (T,), torch.float32)
    dlogp = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = logprob_backward_binding_key(X, W, labels, lse, dlogp, chunk_size=chunk_size, need_dx=need_dx, need_dw=need_dw)
        binding = cache.lookup(key)
        if binding is not None:
            return binding.backward_logprob(X, W, labels, lse, dlogp)
    runner = prepare_lm_head_loss(X, W, labels, chunk_size=chunk_size, need_dx=need_dx, need_dw=need_dw, deterministic=deterministic,
                                  entry="logprob", dlogp=dlogp, lse=lse, backend=backend)
    result = runner.backward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return result


# ---------------------------------------------------------------------------
# Autograd Functions and the backend-level entry points
# ---------------------------------------------------------------------------


class ChunkedLmHeadLossFunction(torch.autograd.Function):
    """Entry (a): the forward produces the FP32 gradient accumulators; the backward scales and casts them."""

    @staticmethod
    def forward(ctx, X, W, labels, objective, loss_div, infer_logp, loss_weights, chunk_size, return_logp, grad_weight_dtype, deterministic, backend):
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        result = forward_loss(
            X, W, labels, objective=objective, loss_div=loss_div, infer_logp=infer_logp, loss_weights=loss_weights,
            chunk_size=chunk_size, need_dx=need_dx, need_dw=need_dw, grad_weight_dtype=grad_weight_dtype,
            deterministic=deterministic, backend=backend,
        )
        ctx.set_materialize_grads(False)
        ctx.dx_acc, ctx.dw_acc = result.dx_acc, result.dw_acc  # saved state, never mutated
        ctx.grad_weight_dtype = grad_weight_dtype
        ctx.backend = backend
        ctx.mark_non_differentiable(result.logp)
        return result.loss, result.logp

    @staticmethod
    def backward(ctx, grad_loss, grad_logp=None):
        none = (None,) * 12
        if grad_loss is None:
            return none
        dx, dw = backward_loss(ctx.dx_acc, ctx.dw_acc, grad_loss, grad_weight_dtype=ctx.grad_weight_dtype, backend=ctx.backend)
        return (dx, dw) + none[2:]


class ChunkedLmHeadLogprobFunction(torch.autograd.Function):
    """Entry (b): the forward saves the row statistic; the backward recomputes the logits."""

    @staticmethod
    def forward(ctx, X, W, labels, chunk_size, deterministic, backend):
        result = forward_logprob(X, W, labels, chunk_size=chunk_size, deterministic=deterministic, backend=backend)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(X, W, labels, result.lse)
        ctx.chunk_size, ctx.deterministic, ctx.backend = chunk_size, deterministic, backend
        return result.logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return None, None, None, None, None, None
        X, W, labels, lse = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        dx, dw = backward_logprob(X, W, labels, lse, dlogp, chunk_size=ctx.chunk_size, need_dx=need_dx, need_dw=need_dw,
                                  deterministic=ctx.deterministic, backend=ctx.backend)
        return dx, dw, None, None, None, None


def chunked_lm_head_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_logp: bool = False,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    backend: str = "cake",
):
    """Differentiable chunked LM-head + loss (see the module docstring); returns the FP32 scalar
    ``loss`` and, with ``return_logp``, the detached FP32 ``logp [T]``.

    ``grad_weight_dtype`` must equal ``W.dtype`` here: PyTorch's autograd engine
    casts every gradient to its leaf's dtype, so an FP32 ``dW`` cannot leave
    this entry through ``W.grad`` for a BF16 ``W``.  Use ``forward_loss(...,
    need_dw=True)`` + ``backward_loss(dx_acc, dw_acc, g, grad_weight_dtype=
    torch.float32)`` for an FP32 weight gradient.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if grad_weight_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("grad_weight_dtype must be torch.bfloat16 or torch.float32")
    if grad_weight_dtype != W.dtype:
        raise ValueError(
            f"grad_weight_dtype={grad_weight_dtype} differs from W.dtype={W.dtype}: the autograd engine casts "
            "every gradient to its leaf's dtype, so this entry cannot return it; use forward_loss(..., need_dw=True) + "
            "backward_loss(dx_acc, dw_acc, g, grad_weight_dtype=torch.float32) for an FP32 dW"
        )
    loss, logp = ChunkedLmHeadLossFunction.apply(
        X, W, labels, objective, loss_div, infer_logp, loss_weights, int(chunk_size), bool(return_logp), grad_weight_dtype, deterministic, backend,
    )
    return (loss, logp.detach()) if return_logp else loss


def chunked_lm_head_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    deterministic: bool = True,
    backend: str = "cake",
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` (0 on ignored rows) for arbitrary downstream losses."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    return ChunkedLmHeadLogprobFunction.apply(X, W, labels, int(chunk_size), deterministic, backend)
