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
  accumulation then a cast; ``dkv_fp32=True`` returns the FP32 accumulators).
* BF16 operands into the MMAs (Q, K, V, P, dS), FP32 accumulation; S is
  recomputed in the backward from the BF16 Q and K.

Every allocation happens in :func:`prepare_dsa_train`; the returned runner
launches with no CUDA allocation and no host synchronization, so a runner
(or a CUDA graph capturing it) replays for new values written into the bound
tensors.  Kernels are reached through the argument plans of the registry
records in ``cake_jit.MODULES``; the ``abi`` field of a record names the
keyword set its kernels expect (see :data:`ABI_CONTRACT`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    BACKWARD_STAGES,
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
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

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
    "indices_offset",  # element offset of indices inside the storage alias the host passes (0 when contiguous)
    "k_rope_stride",  # k_rope row stride (elements); k_rope is passed as its storage alias
    "k_rope_offset",  # element offset of k_rope inside that storage (0 when contiguous)
    "has_topk_length",  # 1 when the caller supplied topk_length
    "q_latent_row_stride",
    "q_rope_row_stride",
    "kv_latent_row_stride",
    "num_rows",  # bwd_delta: num_queries * 64 (token, head) rows
    "latent_vecs",  # bwd_cast: num_kv * 512 / 16 sixteen-element vectors
    "rope_vecs",  # bwd_cast: num_kv * 64 / 16
)
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
            "DSA sparse-attention training requires compute capability 10.0 or 10.3"
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_head_tensor(t: torch.Tensor, name: str, last: int) -> None:
    if t.ndim != 3 or t.shape[1] != NUM_HEADS or t.shape[2] != last:
        raise ValueError(f"{name} must be a BF16 [T, {NUM_HEADS}, {last}] tensor")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    # Heads are ``last`` apart in a contiguous tensor and ``D_QK`` apart in a view of a packed [T, 64, 576] tensor.
    if t.stride(2) != 1 or t.stride(1) not in (last, D_QK):
        raise ValueError(
            f"{name} must be contiguous within a token row (a view of a packed "
            f"[T, {NUM_HEADS}, {D_QK}] tensor is allowed)"
        )


def _check_key_tensor(t: torch.Tensor, name: str, last: int) -> None:
    if t.ndim != 2 or t.shape[1] != last:
        raise ValueError(f"{name} must be a BF16 [S, {last}] tensor")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if t.stride(1) != 1:
        raise ValueError(f"{name} rows must be contiguous (a view of a packed [S, {D_QK}] tensor is allowed)")


def validate_dsa_train_inputs(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    *,
    dout: Optional[torch.Tensor] = None,
) -> tuple[int, int, int]:
    """Shape / dtype validation shared by the entry points.

    Returns ``(T, S, topk)``.  Device placement is checked separately so this
    runs on host tensors.
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
    if indices.ndim != 2 or indices.dtype != torch.int32 or int(indices.shape[0]) != num_queries:
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
        if not dout.is_contiguous():
            raise ValueError("dout must be contiguous (call .contiguous() first)")
    return num_queries, num_kv, topk


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}")


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
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    ``delta`` and the FP32 dK/dV accumulators exist for the backward; the
    ``topk_length`` region backs a full-length vector when the caller passes
    none; ``tma_descriptor_workspace`` is the caller-owned descriptor storage
    of pointer-ABI programs.  The seed profile adds the packed ``q``/``kv``
    operands, an auxiliary row-max buffer and an (unused) sink vector.
    """
    del topk  # every region is independent of the top-k width
    sizes = [("topk_length", num_queries * 4)]
    if backward:
        sizes += [
            ("delta", num_queries * NUM_HEADS * 4),
            ("dkv_latent_acc", num_kv * D_LATENT * 4),
            ("dk_rope_acc", num_kv * D_ROPE * 4),
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
    return max((int(record[s].get("tma_workspace_bytes", 0)) for s in stages), default=0)


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
) -> int:
    """Workspace bytes :func:`prepare_dsa_train` needs for ``(T, S, topk)`` on ``device``."""
    name, record = record_for(device)
    stages = registered_stages(name)
    return int(
        workspace_layout(
            num_queries,
            num_kv,
            topk,
            abi=record_abi(record),
            backward=backward,
            tma_workspace_bytes=_record_tma_bytes(record, stages),
            scratch_bytes=_record_scratch_bytes(record, stages) if backward else 0,
        )["total"]
    )


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if needed > nbytes:
        raise ValueError(f"workspace region {name!r} holds {nbytes} bytes, {needed} needed")
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Varlen index offsetting (host glue of the varlen entry point)
# ---------------------------------------------------------------------------


def offset_gather_kv_indices(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Turn per-document key indices into global key rows.

    ``gather_kv_indices [T, topk]`` holds, for query row ``t`` of document
    ``d``, key positions relative to the document's first key
    (``cu_seqlens_k[d]``); ``-1`` or a position ``>= seqlen_k[d]`` is invalid.
    The result addresses the packed ``kv_*`` tensors; invalid slots become
    ``-1``.  Runs on device without a host synchronization.
    """
    if gather_kv_indices.ndim != 2 or gather_kv_indices.dtype != torch.int32:
        raise ValueError("gather_kv_indices must be an int32 [T, topk] tensor")
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.ndim != 1 or cu.dtype != torch.int32 or cu.numel() < 2:
            raise ValueError(f"{name} must be an int32 [num_docs + 1] tensor")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same documents")
    total_q = int(gather_kv_indices.shape[0])
    num_docs = int(cu_seqlens_q.numel()) - 1
    device = gather_kv_indices.device
    seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    doc_of_row = torch.repeat_interleave(
        torch.arange(num_docs, device=device), seqlens_q, output_size=total_q
    )
    key_base = cu_seqlens_k[:-1].to(torch.int64)[doc_of_row][:, None]
    key_len = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int64)[doc_of_row][:, None]
    local = gather_kv_indices.to(torch.int64)
    valid = (local >= 0) & (local < key_len)
    result = torch.where(valid, local + key_base, torch.full_like(local, -1)).to(torch.int32)
    if out is None:
        return result
    out.copy_(result)
    return out


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def grid_dims(rule, scalars: dict[str, int], num_sms: int) -> tuple[int, int, int]:
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


@dataclass(frozen=True)
class _Launch:
    stage: str
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]
    # Descriptor-preparation entry of a pointer-ABI module (same arguments).
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
        # A profile may provide the kernel's own name (seed profile) or the contract name behind an alias.
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif key in values and values[key] is not None:  # buffer / tma_buffer / workspace / parameter
            value = values[key]
            if kind == "buffer" and f"{key}_storage" in values:
                value = values[f"{key}_storage"]  # raw pointer: whole storage + <name>_offset elements
            arguments.append(value)
        else:
            raise KeyError(
                f"generated module {module_name!r} stage {stage!r} expects argument "
                f"{name!r} ({kind}); the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_dsa_train_module(module_name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return _Launch(stage, module_name, getattr(module, physical["ffi_entry"]), tuple(arguments), grid, prepare)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DSATrainRunner:
    """The prepared forward / backward launches of one tensor binding.

    ``forward()`` writes ``out``, ``lse`` and ``o_lo``; ``backward()`` writes
    ``dq_latent``, ``dq_rope``, ``dkv_latent`` and ``dk_rope`` (the BF16 casts
    of the FP32 accumulators ``dkv_latent_acc`` / ``dk_rope_acc``, or those
    accumulators themselves when prepared with ``dkv_fp32=True``); ``step()``
    runs both.  No launch allocates or synchronizes; capture into a CUDA graph
    belongs to the caller.  Prepare a new runner when a shape, dtype or tensor
    binding changes; values may change freely.
    """

    module_name: str
    abi: str
    num_queries: int
    num_kv: int
    topk: int
    softmax_scale: float
    tensors: dict[str, torch.Tensor] = field(repr=False)
    launches: dict[str, _Launch] = field(repr=False)
    stages: tuple[str, ...]
    dkv_fp32: bool
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
        return "bwd_main" in self.launches

    def prepare_tma(self) -> None:
        """Encode the descriptors of pointer-ABI stages once (idempotent)."""
        if self._tma_prepared:
            return
        with tvm_ffi.use_torch_stream():
            for launch in self.launches.values():
                if launch.prepare is not None:
                    launch.prepare(*launch.arguments)
        self._tma_prepared = True

    def _run(self, stages: tuple[str, ...]) -> None:
        self.prepare_tma()
        with tvm_ffi.use_torch_stream():
            for stage in stages:
                launch = self.launches.get(stage)
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
            torch.isposinf(lse, out=t["lse_mask"])  # empty rows; no temporaries (launch path allocates nothing)
            lse.masked_fill_(t["lse_mask"], float("-inf"))
        return t["out"], t["lse"], t.get("o_lo")

    def backward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_backward:
            raise NotImplementedError(
                "the registered DSA training program has no backward stages "
                f"(registered: {self.stages}); backward is unavailable in this checkout"
            )
        t = self.tensors
        t["dkv_latent_acc"].zero_()
        t["dk_rope_acc"].zero_()
        self._run(BACKWARD_STAGES)
        if self.dkv_fp32:
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

    Raw pointer arguments are checked for contiguity at the FFI boundary; a
    strided view (the rope columns of a packed ``[S, 576]`` tensor) is passed
    as the whole storage viewed flat plus the element offset the kernel adds.
    """
    if tensor.is_contiguous():
        return tensor, 0
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    flat.set_(tensor.untyped_storage())
    return flat, int(tensor.storage_offset())


def _contract_values(t: dict[str, torch.Tensor], scalars: dict[str, Any]) -> dict[str, Any]:
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
    return values


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
    backend: str = "cake",
) -> DSATrainRunner:
    """Validate one binding and prepare its launches.

    ``backward`` defaults to ``dout is not None``; a backward binding requires
    the backward stages to be registered.  Missing outputs and the workspace
    are allocated here (the only allocations of the backend).  Pass
    ``workspace_buffer`` of :func:`dsa_train_workspace_size` bytes to reuse
    storage across steps.
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
    tensors += [t for t in (topk_length, dout, workspace_buffer, out, lse, o_lo, dq_latent, dq_rope, dkv_latent, dk_rope) if t is not None]
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
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()

    _check_output(out, "out", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(lse, "lse", (num_queries, NUM_HEADS), torch.float32)
    _check_output(o_lo, "o_lo", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(dq_latent, "dq_latent", (num_queries, NUM_HEADS, D_LATENT), torch.bfloat16)
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
    )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(layout["total"], dtype=torch.uint8, device=device)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}")

    t: dict[str, torch.Tensor] = dict(
        q_latent=q_latent, q_rope=q_rope, kv_latent=kv_latent, k_rope=k_rope, indices=indices
    )
    has_topk_length = int(topk_length is not None)
    if topk_length is None:
        topk_length = _carve(flat, layout, "topk_length", torch.int32, (num_queries,))
        topk_length.fill_(topk)
    t["topk_length"] = topk_length
    t["out"] = out if out is not None else torch.empty((num_queries, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=device)
    t["lse"] = lse if lse is not None else torch.empty((num_queries, NUM_HEADS), dtype=torch.float32, device=device)
    if abi == ABI_CONTRACT:
        t["o_lo"] = o_lo if o_lo is not None else torch.empty_like(t["out"])
    else:
        t["packed_q"] = _carve(flat, layout, "packed_q", torch.bfloat16, (num_queries, NUM_HEADS, D_QK))
        t["packed_kv"] = _carve(flat, layout, "packed_kv", torch.bfloat16, (num_kv, D_QK))
        t["aux_logits"] = _carve(flat, layout, "aux_logits", torch.float32, (num_queries, NUM_HEADS))
        t["sinks"] = _carve(flat, layout, "sinks", torch.float32, (NUM_HEADS,))
        t["lse_mask"] = torch.empty((num_queries, NUM_HEADS), dtype=torch.bool, device=device)  # +inf -> -inf rewrite scratch
    if backward:
        t["dout"] = dout
        t["delta"] = _carve(flat, layout, "delta", torch.float32, (num_queries, NUM_HEADS))
        t["dkv_latent_acc"] = _carve(flat, layout, "dkv_latent_acc", torch.float32, (num_kv, D_LATENT))
        t["dk_rope_acc"] = _carve(flat, layout, "dk_rope_acc", torch.float32, (num_kv, D_ROPE))
        t["dq_latent"] = dq_latent if dq_latent is not None else torch.empty((num_queries, NUM_HEADS, D_LATENT), dtype=torch.bfloat16, device=device)
        t["dq_rope"] = dq_rope if dq_rope is not None else torch.empty((num_queries, NUM_HEADS, D_ROPE), dtype=torch.bfloat16, device=device)
        if not dkv_fp32:
            t["dkv_latent"] = dkv_latent if dkv_latent is not None else torch.empty((num_kv, D_LATENT), dtype=torch.bfloat16, device=device)
            t["dk_rope"] = dk_rope if dk_rope is not None else torch.empty((num_kv, D_ROPE), dtype=torch.bfloat16, device=device)
    if layout.get("workspace"):
        t["workspace"] = _carve(flat, layout, "workspace", torch.uint8, (layout["workspace"][1],))
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = _carve(
            flat, layout, "tma_descriptor_workspace", torch.uint8, (layout["tma_descriptor_workspace"][1],)
        )

    scalars = dict(
        num_queries=num_queries, num_kv=num_kv, topk=topk, softmax_scale=float(softmax_scale),
        has_topk_length=has_topk_length,
    )
    values = _seed_values(t, scalars) if abi == ABI_SEED else _contract_values(t, scalars)
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    wanted = FORWARD_STAGES + (BACKWARD_STAGES if backward else ())
    launches = {}
    for stage in stages:
        if stage not in wanted:
            continue
        physical = record[stage]
        grid = grid_dims(physical.get("grid", ["num_queries", 1, 1]), scalars, num_sms)
        launches[stage] = bind_stage(module_name, stage, values, grid)
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
    )


# ---------------------------------------------------------------------------
# Eager entry points (allocate, launch, return)
# ---------------------------------------------------------------------------


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
    """Forward pass: ``(out, lse, o_lo)``; ``o_lo`` is ``None`` for the placeholder program."""
    runner = prepare_dsa_train(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length=topk_length,
        softmax_scale=softmax_scale, backward=False,
    )
    return runner.forward()


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass from the saved forward outputs: ``(dq_latent, dq_rope, dkv_latent, dk_rope)``."""
    if o_lo is None:
        raise NotImplementedError(
            "the forward produced no output residual (placeholder program); backward is unavailable"
        )
    _check_output(out, "out", (q_latent.shape[0], NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(o_lo, "o_lo", (q_latent.shape[0], NUM_HEADS, D_LATENT), torch.bfloat16)
    _check_output(lse, "lse", (q_latent.shape[0], NUM_HEADS), torch.float32)
    runner = prepare_dsa_train(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length=topk_length,
        dout=dout.contiguous(), softmax_scale=softmax_scale, out=out, lse=lse, o_lo=o_lo,
        dkv_fp32=dkv_fp32, backward=True,
    )
    return runner.backward()


class DSASparseAttentionFunction(torch.autograd.Function):
    """Autograd wrapper: saves ``out``, ``o_lo``, ``lse`` and the inputs for the backward."""

    @staticmethod
    def forward(ctx, q_latent, q_rope, kv_latent, k_rope, indices, topk_length, softmax_scale):
        out, lse, o_lo = forward(
            q_latent, q_rope, kv_latent, k_rope, indices,
            topk_length=topk_length, softmax_scale=softmax_scale,
        )
        ctx.softmax_scale = softmax_scale
        ctx.has_topk_length = topk_length is not None
        saved = [q_latent, q_rope, kv_latent, k_rope, indices, out, lse]
        saved.append(o_lo if o_lo is not None else out.new_empty(0))
        saved.append(topk_length if topk_length is not None else indices.new_empty(0))
        ctx.has_o_lo = o_lo is not None
        ctx.save_for_backward(*saved)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse=None):
        q_latent, q_rope, kv_latent, k_rope, indices, out, lse, o_lo, topk_length = ctx.saved_tensors
        if not ctx.has_o_lo:
            raise NotImplementedError(
                "backward is unavailable: the registered program is forward-only (placeholder)"
            )
        dq_latent, dq_rope, dkv_latent, dk_rope = backward(
            q_latent, q_rope, kv_latent, k_rope, indices, out, o_lo, lse, dout.contiguous(),
            topk_length=topk_length if ctx.has_topk_length else None,
            softmax_scale=ctx.softmax_scale,
        )
        return dq_latent, dq_rope, dkv_latent, dk_rope, None, None, None


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
):
    """Differentiable sparse attention over global key indices (see the module docstring)."""
    validate_dsa_train_inputs(q_latent, q_rope, kv_latent, k_rope, indices, topk_length)
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    out, lse = DSASparseAttentionFunction.apply(
        q_latent, q_rope, kv_latent, k_rope, indices, topk_length, float(softmax_scale)
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
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
):
    """Packed multi-document form: per-document ``gather_kv_indices`` are offset by
    ``cu_seqlens_k`` on device (this glue counts in the step time), then the flat
    kernels run over the packed rows.  ``max_seqlen_q/k`` are accepted for
    signature parity and not used on the host."""
    del max_seqlen_q, max_seqlen_k
    indices = offset_gather_kv_indices(gather_kv_indices, cu_seqlens_q, cu_seqlens_k)
    return dsa_sparse_attention(
        q_latent, q_rope, kv_latent, k_rope, indices,
        topk_length=topk_length, softmax_scale=softmax_scale, return_lse=return_lse,
    )
