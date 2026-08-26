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

"""MoE routing prologue and weighted-sum finalize.

These are the two halves of the non-GEMM glue a serving engine runs around the
routed-expert GEMMs of one MoE block.  They are two entry points because that
is the dataflow: the prologue produces the block-aligned descriptors the expert
GEMM needs, and the finalize consumes the tensor that GEMM produces::

    moe_routing_prologue(hidden_states, gate_weight, shared_gate_weight)
        -> topk_weights, topk_ids,
           sorted_token_ids, expert_ids, num_tokens_post_pad,
           shared_gate, router_logits

        w13 GEMM -> activation -> w2 GEMM     (reads the descriptors,
                                               produces ``expert_out``)

    moe_routing_finalize(expert_out, shared_out, topk_weights, shared_gate)
        -> output

A third entry point, ``moe_routing_align``, is the prologue's descriptor stage
on its own, for an engine that runs its own router and only wants the
block-aligned descriptor build fused.  It shares the prologue's kernel, so its
descriptors are byte-identical.

All of them always work: the default implementation of each is a portable composition
of torch ops with exactly these semantics.  On SM120, for an allowlisted
problem size, each is served instead by specialized CUDA kernels -- three
launches in total, replacing the ten a serving engine spends on this glue.

**The finalize owns the routing weights.**  ``expert_out`` is the routed-expert
down-projection output with ``topk_weights`` NOT yet applied.  A caller whose
expert GEMM folds the routing weights into its own epilogue must turn that off
(vLLM: ``mul_topk_weights=False``), or they are applied twice -- which presents
as an accuracy regression, not as a crash.

Semantics that are load bearing and are shared by both implementations:

* scoring is **softmax** over all experts (not sigmoid); only the shared-expert
  gate is a sigmoid;
* the router logits are rounded to bfloat16 between the GEMV and the scoring,
  and are returned so a caller can see exactly what was scored;
* selection is by descending score with ties broken toward the **lower** expert
  id, and the selected weights are renormalised in fp32;
* the shared gate is bf16(sigmoid(bf16(hidden @ shared_gate_weight.T)));
* the finalize accumulates in fp32 and rounds to bfloat16 exactly once;
* ``topk_weights`` stays float32 and the descriptors stay int32.

Dispatch contract for the specialized path:

* the allowlist lives in ``moe_routing_sm120_workloads.json`` (package data,
  next to this module) and is matched exactly; nothing is inferred from it;
* ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` is read at *call* time and
  restores the composable path for all three entry points;
* **nothing is prebuilt.**  This op is not in the AOT pass, so the single
  translation unit under ``kernel/`` JIT-compiles on the first *non-capturing*
  dispatch, or on an explicit :func:`moe_routing_precompile` -- which is what a
  serving engine's eager profile pass amounts to;
* during CUDA-graph capture an entry point dispatches only if that module is
  already compiled and loaded; it never compiles, queries devices or
  synchronizes under capture.  A capture that arrives cold therefore records
  the (capture-safe) composable path for that shape instead of compiling inside
  the graph: correct, and slower until something warms the module;
* one translation unit holds all three entry points, so there is exactly one
  compiled variant and capture readiness is a single check;
* the kernels hold **no persistent device state** and contain no inter-CTA
  rendezvous -- every launch is independent of every other launch.

Import cost: this module imports :mod:`torch` and nothing else.  The allowlist,
the JIT toolchain and the kernel are all reached lazily, on the first probe or
call, so ``flashinfer/__init__.py`` re-exports these names unconditionally and a
consumer's ``getattr(flashinfer, "moe_routing_finalize", None)`` costs nothing
and never compiles.
"""

import functools
import json
import os
from importlib import resources
from typing import Optional, Tuple

import torch

# NOTE: torch is the only import at module scope, and that is a contract, not a
# coincidence -- see the "import cost" paragraph of the module docstring.  The
# JIT toolchain, the allowlist and the kernel are reached through the lazy
# helpers below, on the first probe or call.


@functools.cache
def _jit_logger():
    """FlashInfer's JIT logger, imported on first use."""
    from ...jit.core import logger

    return logger


@functools.cache
def _sm120_module_generator():
    """The specialized module's JitSpec factory, or None on an install without it.

    Imported lazily so that ``import flashinfer`` -- and therefore a
    consumer's ``getattr(flashinfer, "moe_routing_finalize", None)``
    capability check -- never pulls in the JIT toolchain.  Constructing the
    spec compiles nothing; that happens in :func:`_get_module`.
    """
    try:
        from ...jit.moe_routing import gen_moe_routing_sm120_module
    except (ImportError, RuntimeError):  # pragma: no cover - packaging fallback
        return None
    return gen_moe_routing_sm120_module


def _sm120_available() -> bool:
    """Can this install build the specialized kernels at all?"""
    return _sm120_module_generator() is not None


_WORKLOAD_PACKAGE = "flashinfer.fused_moe.experimental"
_WORKLOAD_FILE = "moe_routing_sm120_workloads.json"
_WORKLOAD_FIELDS = ("m", "hidden_size", "num_experts", "top_k")

# (12, 0) only: the architecture the kernels were written and validated for.
_SUPPORTED_COMPUTE_CAPABILITY = (12, 0)

# BLOCK_M of the block-aligned descriptor the MoE GEMM consumes.  Not a free
# parameter: it has to match what the expert GEMM was built for.
BLOCK_SIZE_M = 8
_INACTIVE_EXPERT_ID = -1
# The descriptor pass keeps every (token, slot) assignment inside one CTA.
_MAX_TOKENS = 32

_MODULE = None
_PROLOGUE_LAUNCH_COUNT = 0
_ALIGN_LAUNCH_COUNT = 0
_FINALIZE_LAUNCH_COUNT = 0
_DISPATCH_LOGGED = set()


@functools.cache
def load_moe_routing_sm120_workloads() -> frozenset:
    """Exact (m, hidden_size, num_experts, top_k) sizes the kernels may serve."""
    try:
        payload = json.loads(
            resources.files(_WORKLOAD_PACKAGE).joinpath(_WORKLOAD_FILE).read_text()
        )
        fields = tuple(payload["fields"])
        if fields != _WORKLOAD_FIELDS:
            raise ValueError(f"unexpected workload fields: {fields}")
        return frozenset(
            tuple(int(value) for value in row) for row in payload["workloads"]
        )
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        _jit_logger().warning_once(
            "Unable to load the MoE routing allowlist: %s", type(exc).__name__
        )
        return frozenset()


def _get_module():
    """Build + load the module, caching it. Never called under graph capture."""
    global _MODULE
    if _MODULE is None:
        generator = _sm120_module_generator()
        if generator is None:
            raise RuntimeError("MoE routing module generator unavailable")
        _MODULE = generator().build_and_load()
    return _MODULE


def moe_routing_ready_for_graph_capture() -> bool:
    """True when a dispatch cannot trigger a compile.

    One module holds both halves and every supported size, so readiness does
    not depend on which half is called or on the problem size: once loaded,
    every allowlisted call is capture-ready.
    """
    return _MODULE is not None


def moe_routing_precompile() -> bool:
    """Compile + load the specialized module ahead of CUDA-graph capture.

    This op ships no AOT build, so the module is *always* compiled here or by
    the first non-capturing dispatch -- never by a dispatch under capture, which
    checks :func:`moe_routing_ready_for_graph_capture` instead and falls back if
    the module is cold.  Serving engines run an eager profile pass before they
    capture graphs and warm it implicitly; callers that would rather be explicit
    (or that capture without a profile pass) invoke this directly.  Returns True
    when the module is loaded afterwards.
    """
    if not _sm120_available():
        return False
    try:
        _get_module()
    except (ImportError, OSError, RuntimeError) as exc:
        # The message, not just the class. A build failure here is silent by
        # design -- every call then takes the composable path and only a launch
        # counter notices -- so the compiler's own error is the single most
        # useful thing this line can carry.
        _jit_logger().warning_once(
            "Unable to build the specialized MoE routing kernels: %s: %s",
            type(exc).__name__,
            str(exc)[:4000],
        )
        return False
    return _MODULE is not None


def _is_current_stream_capturing() -> bool:
    # Same convention as the rest of the package (see flashinfer/page.py): on a
    # torch build without the query there is no capture to protect against.
    if not hasattr(torch.cuda, "is_current_stream_capturing"):
        return False
    return bool(torch.cuda.is_current_stream_capturing())


def _device_capability(device: torch.device) -> Optional[Tuple[int, int]]:
    try:
        return torch.cuda.get_device_capability(device)
    except (RuntimeError, AssertionError):
        return None


def _kill_switch() -> bool:
    return os.environ.get("FLASHINFER_SPECIALIZED_KERNEL_DISABLE") == "1"


def _device_ok(device: Optional[torch.device]) -> bool:
    if device is None:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _device_capability(device) == _SUPPORTED_COMPUTE_CAPABILITY


def moe_routing_supported(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    block_size_m: int = BLOCK_SIZE_M,
) -> bool:
    """Are there specialized kernels for this problem size on this device?

    One predicate covers both halves on purpose: the prologue's ``shared_gate``
    feeds the finalize, so a caller routes both or neither.  Callers that have a
    fast native implementation of their own should use this to decide whether to
    route the op to FlashInfer at all -- the composable fallback is a
    correctness guarantee, not a performance one.  The answer depends only on
    values known at layer-construction time, so it can be cached there; it
    deliberately does NOT depend on anything per-call.
    """
    if _kill_switch():
        return False
    if not _sm120_available():
        return False
    if dtype is not torch.bfloat16:
        return False
    if int(block_size_m) != BLOCK_SIZE_M:
        return False
    if not _device_ok(device):
        return False
    key = (int(num_tokens), int(hidden_size), int(num_experts), int(top_k))
    return key in load_moe_routing_sm120_workloads()


def _prologue_allowed(m: int, hidden_size: int, num_experts: int, top_k: int) -> bool:
    return (
        int(m),
        int(hidden_size),
        int(num_experts),
        int(top_k),
    ) in load_moe_routing_sm120_workloads()


def _align_allowed(m: int, num_experts: int, top_k: int) -> bool:
    """The align entry point never sees ``hidden_size``: it takes ``topk_ids``.

    So it matches on the three axes it can observe and accepts a size that any
    allowlisted row covers.  The kernel's baked-in geometry here is
    ``num_experts`` and ``top_k``; both appear.
    """
    return any(
        row[0] == int(m) and row[2] == int(num_experts) and row[3] == int(top_k)
        for row in load_moe_routing_sm120_workloads()
    )


def _finalize_allowed(m: int, hidden_size: int, top_k: int) -> bool:
    """The finalize never sees ``num_experts``: it takes ``expert_out``.

    So it matches on the three axes it can observe and accepts a size that any
    allowlisted row covers.  The geometry constants baked into the kernel are
    ``hidden_size`` and ``top_k``; both appear here.
    """
    return any(
        row[0] == int(m) and row[1] == int(hidden_size) and row[3] == int(top_k)
        for row in load_moe_routing_sm120_workloads()
    )


def _dispatch_ready() -> bool:
    """Resolve capture-safety and module readiness for a call about to dispatch."""
    if _is_current_stream_capturing():
        # Compiling, or anything else host-side and un-captured, is not allowed
        # here: fall back so the capture-safe composable path is what gets baked
        # into the graph.
        return moe_routing_ready_for_graph_capture()
    return moe_routing_precompile()


def _note_dispatch(which: str, detail: str) -> None:
    if which not in _DISPATCH_LOGGED:
        _DISPATCH_LOGGED.add(which)
        _jit_logger().info(
            "flashinfer: specialized SM120 MoE routing %s kernel dispatched (%s)",
            which,
            detail,
        )


# ---------------------------------------------------------------- prologue
def _should_use_specialized_prologue(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    router_logits: torch.Tensor,
    shared_gate: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> bool:
    """Exact-match guard. Any doubt at all returns False (composable path)."""
    if _kill_switch() or not _sm120_available():
        return False

    tensors = (
        hidden_states,
        gate_weight,
        shared_gate_weight,
        router_logits,
        shared_gate,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    device = hidden_states.device
    if any((not t.is_cuda) or t.device != device for t in tensors):
        return False
    if _device_capability(device) != _SUPPORTED_COMPUTE_CAPABILITY:
        return False
    # The kernels index every operand by a row stride derived from the shape,
    # so anything but a densely packed tensor is out.
    if any(not t.is_contiguous() for t in tensors):
        return False

    if hidden_states.ndim != 2 or gate_weight.ndim != 2:
        return False
    m, hidden_size = hidden_states.shape
    num_experts = gate_weight.shape[0]
    if m > _MAX_TOKENS:
        return False
    if gate_weight.shape[1] != hidden_size:
        return False
    if tuple(shared_gate_weight.shape) != (1, hidden_size):
        return False
    if tuple(router_logits.shape) != (m, num_experts):
        return False
    if tuple(shared_gate.shape) != (m,):
        return False
    if topk_weights.ndim != 2 or topk_weights.shape[0] != m:
        return False
    top_k = topk_weights.shape[1]
    if tuple(topk_ids.shape) != (m, top_k):
        return False
    if sorted_token_ids.ndim != 1 or sorted_token_ids.shape[0] != 64 * m:
        return False
    if expert_ids.ndim != 1 or expert_ids.shape[0] != BLOCK_SIZE_M * m:
        return False
    if num_tokens_post_pad.numel() != 1:
        return False

    bf16 = (
        hidden_states,
        gate_weight,
        shared_gate_weight,
        router_logits,
        shared_gate,
    )
    if any(t.dtype is not torch.bfloat16 for t in bf16):
        return False
    if topk_weights.dtype is not torch.float32:
        return False
    if any(
        t.dtype is not torch.int32
        for t in (topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad)
    ):
        return False

    return _prologue_allowed(m, hidden_size, num_experts, top_k)


def _reference_descriptors(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size_m: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Block-aligned sort descriptors, written in place.

    Padding entries of ``sorted_token_ids`` carry ``topk_ids.numel()`` and
    blocks past ``num_tokens_post_pad`` carry -1, matching what the expert GEMM
    expects (it reads neither, bounded by ``num_tokens_post_pad``).
    """
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = int(flat.numel())
    # scatter_add rather than bincount: bincount reduces on device and reads the
    # result on the host to size its output, which is a synchronization and
    # would make the composable path un-capturable.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum[1:] = torch.cumsum(padded, dim=0)
    total = cumsum[num_experts]

    sorted_token_ids.fill_(numel)
    order = torch.argsort(flat, stable=True)
    experts_in_order = flat[order]
    dense_starts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    dense_starts[1:] = torch.cumsum(counts, dim=0)
    rank = (
        torch.arange(numel, device=device, dtype=torch.int64)
        - dense_starts[experts_in_order]
    )
    sorted_token_ids[cumsum[experts_in_order] + rank] = order.to(torch.int32)

    num_blocks = expert_ids.shape[0]
    block_start = torch.arange(num_blocks, device=device, dtype=torch.int64) * (
        block_size_m
    )
    owner = torch.searchsorted(cumsum, block_start, right=True) - 1
    expert_ids.copy_(
        torch.where(
            block_start < total, owner, torch.full_like(owner, _INACTIVE_EXPERT_ID)
        ).to(torch.int32)
    )
    num_tokens_post_pad.copy_(total.reshape(1).to(torch.int32))


def _reference_prologue(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    router_logits: torch.Tensor,
    shared_gate: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size_m: int,
) -> None:
    """Portable composition with the documented semantics, written in place."""
    m, hidden_size = hidden_states.shape
    num_experts = gate_weight.shape[0]
    top_k = topk_weights.shape[1]

    # Router GEMV, rounded to bf16 before the scoring (load bearing: the
    # rounding can change which expert wins a near tie).
    router_logits.copy_(
        (hidden_states.to(torch.float32) @ gate_weight.to(torch.float32).t()).to(
            torch.bfloat16
        )
    )

    scores = torch.softmax(router_logits.to(torch.float32), dim=-1)
    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    # Stable descending sort == "strictly greater wins, else the lower id wins".
    values, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    weights = values[..., :top_k].contiguous()
    ids = indices[..., :top_k].to(torch.int32).contiguous()
    denom = weights.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    topk_weights.copy_(weights / denom)
    topk_ids.copy_(ids)

    # Block-aligned sort descriptors (shared with moe_routing_align, exactly as
    # the two entry points share one kernel).
    _reference_descriptors(
        topk_ids,
        num_experts,
        block_size_m,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    # Shared-expert scalar gate: bf16 dot -> bf16 logit -> sigmoid -> bf16.
    gate_logit = (
        hidden_states.to(torch.float32) @ shared_gate_weight.to(torch.float32).t()
    ).to(torch.bfloat16)
    shared_gate.copy_(
        torch.sigmoid(gate_logit.to(torch.float32)).to(torch.bfloat16).reshape(m)
    )


def moe_routing_prologue(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    sorted_token_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    num_tokens_post_pad: Optional[torch.Tensor] = None,
    shared_gate: Optional[torch.Tensor] = None,
    router_logits: Optional[torch.Tensor] = None,
    block_size_m: int = BLOCK_SIZE_M,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Router GEMV, top-k, block-aligned descriptors and the shared-expert gate.

    Runs **before** the expert GEMMs; everything it returns is an input to them
    or to :func:`moe_routing_finalize`.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[m, hidden_size]`` bfloat16, the MoE block input (post norm).
    gate_weight : torch.Tensor
        ``[num_experts, hidden_size]`` bfloat16 router weight, out-major.
    shared_gate_weight : torch.Tensor
        ``[1, hidden_size]`` bfloat16 shared-expert gate weight.

    Other Parameters
    ----------------
    top_k : int
        Experts per token.  Inferred from ``topk_weights`` when that is given.
    topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, shared_gate, router_logits
        Pre-allocated destinations.  Serving engines pass persistent buffers so
        the call can be recorded into a CUDA graph; when omitted they are
        allocated here.
    block_size_m : int
        Block size of the aligned descriptor the expert GEMM consumes.

    Returns
    -------
    tuple
        ``(topk_weights, topk_ids, sorted_token_ids, expert_ids,
        num_tokens_post_pad, shared_gate, router_logits)``.
    """
    global _PROLOGUE_LAUNCH_COUNT

    m, hidden_size = hidden_states.shape
    num_experts = gate_weight.shape[0]
    device = hidden_states.device
    if top_k is None:
        if topk_weights is not None:
            top_k = topk_weights.shape[1]
        elif topk_ids is not None:
            top_k = topk_ids.shape[1]
        else:
            raise ValueError(
                "moe_routing_prologue needs top_k when no destination buffer carries it"
            )
    top_k = int(top_k)

    max_num_tokens_padded = m * top_k + num_experts * (block_size_m - 1)
    max_num_tokens_padded = min(m * top_k * block_size_m, max_num_tokens_padded)
    num_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

    if topk_weights is None:
        topk_weights = torch.empty(m, top_k, dtype=torch.float32, device=device)
    if topk_ids is None:
        topk_ids = torch.empty(m, top_k, dtype=torch.int32, device=device)
    if sorted_token_ids is None:
        sorted_token_ids = torch.empty(
            max_num_tokens_padded, dtype=torch.int32, device=device
        )
    if expert_ids is None:
        expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=device)
    if num_tokens_post_pad is None:
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    if shared_gate is None:
        shared_gate = torch.empty(m, dtype=torch.bfloat16, device=device)
    if router_logits is None:
        router_logits = torch.empty(m, num_experts, dtype=torch.bfloat16, device=device)

    outputs = (
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        shared_gate,
        router_logits,
    )

    dispatched = False
    try:
        if block_size_m == BLOCK_SIZE_M and _should_use_specialized_prologue(
            hidden_states,
            gate_weight,
            shared_gate_weight,
            router_logits,
            shared_gate,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        ):
            if _dispatch_ready():
                _MODULE.moe_routing_prologue_sm120(
                    hidden_states,
                    gate_weight,
                    shared_gate_weight,
                    router_logits,
                    shared_gate,
                    topk_weights,
                    topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_pad,
                )
                dispatched = True
    except Exception as exc:  # noqa: BLE001 - a guard must never break the op
        _jit_logger().warning_once(
            "Specialized MoE routing prologue unavailable, using the composable "
            "path: %s: %s",
            type(exc).__name__,
            str(exc)[:1000],
        )
        dispatched = False

    if dispatched:
        _PROLOGUE_LAUNCH_COUNT += 1
        _note_dispatch(
            "prologue",
            f"m={int(m)}, hidden={int(hidden_size)}, experts={int(num_experts)}, "
            f"top_k={top_k}",
        )
    else:
        _reference_prologue(
            hidden_states,
            gate_weight,
            shared_gate_weight,
            router_logits,
            shared_gate,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            block_size_m,
        )

    return outputs


# ------------------------------------------------------------------- align
def _should_use_specialized_align(
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    num_experts: int,
    block_size_m: int,
) -> bool:
    """Exact-match guard. Any doubt at all returns False (composable path)."""
    if _kill_switch() or not _sm120_available():
        return False
    if int(block_size_m) != BLOCK_SIZE_M:
        return False

    tensors = (topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad)
    device = topk_ids.device
    if any((not t.is_cuda) or t.device != device for t in tensors):
        return False
    if _device_capability(device) != _SUPPORTED_COMPUTE_CAPABILITY:
        return False
    if any(not t.is_contiguous() for t in tensors):
        return False
    if any(t.dtype is not torch.int32 for t in tensors):
        return False

    if topk_ids.ndim != 2:
        return False
    m, top_k = topk_ids.shape
    if m > _MAX_TOKENS:
        return False
    if sorted_token_ids.ndim != 1 or sorted_token_ids.shape[0] != 64 * m:
        return False
    if expert_ids.ndim != 1 or expert_ids.shape[0] != BLOCK_SIZE_M * m:
        return False
    if num_tokens_post_pad.numel() != 1:
        return False

    return _align_allowed(m, num_experts, top_k)


def moe_routing_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    *,
    block_size_m: int = BLOCK_SIZE_M,
    sorted_token_ids: Optional[torch.Tensor] = None,
    expert_ids: Optional[torch.Tensor] = None,
    num_tokens_post_pad: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-aligned sort descriptors for an expert GEMM, from ``topk_ids``.

    This is the second stage of :func:`moe_routing_prologue` as a standalone
    entry point, for an engine that runs its own router and only wants the
    descriptor build fused.  It uses the same kernel and therefore produces
    byte-identical descriptors.

    Parameters
    ----------
    topk_ids : torch.Tensor
        ``[m, top_k]`` int32 selected expert ids.
    num_experts : int
        Total expert count (the global count under expert parallelism).

    Other Parameters
    ----------------
    block_size_m : int
        Block size of the aligned descriptor the expert GEMM consumes.
    sorted_token_ids, expert_ids, num_tokens_post_pad
        Pre-allocated destinations, sized ``64*m``, ``8*m`` and 1.  Serving
        engines pass persistent buffers so the call can be recorded into a CUDA
        graph; when omitted they are allocated here with those sizes.

    Returns
    -------
    tuple
        ``(sorted_token_ids, expert_ids, num_tokens_post_pad)``.  Padding
        entries of ``sorted_token_ids`` are ``topk_ids.numel()``; blocks past
        ``num_tokens_post_pad`` are -1 and must not be read.
    """
    global _ALIGN_LAUNCH_COUNT

    m, top_k = topk_ids.shape
    device = topk_ids.device
    max_num_tokens_padded = m * top_k + num_experts * (block_size_m - 1)
    max_num_tokens_padded = min(m * top_k * block_size_m, max_num_tokens_padded)
    num_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

    if sorted_token_ids is None:
        sorted_token_ids = torch.empty(
            max_num_tokens_padded, dtype=torch.int32, device=device
        )
    if expert_ids is None:
        expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=device)
    if num_tokens_post_pad is None:
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    outputs = (sorted_token_ids, expert_ids, num_tokens_post_pad)

    dispatched = False
    try:
        if _should_use_specialized_align(topk_ids, *outputs, num_experts, block_size_m):
            if _dispatch_ready():
                _MODULE.moe_routing_align_sm120(
                    topk_ids, *outputs, int(num_experts), int(block_size_m)
                )
                dispatched = True
    except Exception as exc:  # noqa: BLE001 - a guard must never break the op
        _jit_logger().warning_once(
            "Specialized MoE routing align unavailable, using the composable "
            "path: %s: %s",
            type(exc).__name__,
            str(exc)[:1000],
        )
        dispatched = False

    if dispatched:
        _ALIGN_LAUNCH_COUNT += 1
        _note_dispatch(
            "align",
            f"m={int(m)}, experts={int(num_experts)}, top_k={int(top_k)}",
        )
    else:
        _reference_descriptors(topk_ids, int(num_experts), int(block_size_m), *outputs)

    return outputs


# ---------------------------------------------------------------- finalize
def _should_use_specialized_finalize(
    expert_out: torch.Tensor,
    shared_out: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    shared_gate: Optional[torch.Tensor],
    output: torch.Tensor,
) -> bool:
    """Exact-match guard. Any doubt at all returns False (composable path)."""
    if _kill_switch() or not _sm120_available():
        return False

    tensors = [expert_out, topk_weights, output]
    if shared_out is not None:
        tensors += [shared_out, shared_gate]
    device = expert_out.device
    if any((not t.is_cuda) or t.device != device for t in tensors):
        return False
    if _device_capability(device) != _SUPPORTED_COMPUTE_CAPABILITY:
        return False
    if any(not t.is_contiguous() for t in tensors):
        return False

    if expert_out.ndim != 3:
        return False
    m, top_k, hidden_size = expert_out.shape
    if tuple(topk_weights.shape) != (m, top_k):
        return False
    if tuple(output.shape) != (m, hidden_size):
        return False
    if shared_out is not None:
        if tuple(shared_out.shape) != (m, hidden_size):
            return False
        if tuple(shared_gate.shape) != (m,):
            return False

    if any(t.dtype is not torch.bfloat16 for t in tensors if t is not topk_weights):
        return False
    if topk_weights.dtype is not torch.float32:
        return False

    return _finalize_allowed(m, hidden_size, top_k)


def _reference_finalize(
    expert_out: torch.Tensor,
    shared_out: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    shared_gate: Optional[torch.Tensor],
    output: torch.Tensor,
) -> None:
    """Portable composition with the documented semantics, written in place."""
    acc = (
        topk_weights.to(torch.float32).unsqueeze(-1) * expert_out.to(torch.float32)
    ).sum(dim=1)
    if shared_out is not None:
        acc = acc + shared_gate.to(torch.float32).unsqueeze(-1) * shared_out.to(
            torch.float32
        )
    output.copy_(acc.to(torch.bfloat16))


def moe_routing_finalize(
    expert_out: torch.Tensor,
    shared_out: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    shared_gate: Optional[torch.Tensor],
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expert-weighted sum plus the gated shared expert.

    Runs **after** the routed-expert w2 GEMM.

    .. warning::

       ``expert_out`` must be the routed-expert output with the routing weights
       **not** applied -- this op owns them.  vLLM's Marlin MoE folds them into
       the w2 epilogue by default; a caller must pass ``mul_topk_weights=False``
       there, or the weights are applied twice.

    Parameters
    ----------
    expert_out : torch.Tensor
        ``[m, top_k, hidden_size]`` bfloat16 routed-expert output, unweighted.
    shared_out : torch.Tensor or None
        ``[m, hidden_size]`` bfloat16 shared-expert output, **before** its gate.
        ``None`` (together with ``shared_gate``) when the engine combines the
        shared expert somewhere else -- the result is then exactly the routed
        weighted sum, which is still the fused half of this op.
    topk_weights : torch.Tensor
        ``[m, top_k]`` float32 renormalised routing weights, as returned by
        :func:`moe_routing_prologue`.
    shared_gate : torch.Tensor or None
        ``[m]`` bfloat16 shared-expert scalar gate, as returned by
        :func:`moe_routing_prologue`.  Must be given if and only if
        ``shared_out`` is.

    Other Parameters
    ----------------
    output : torch.Tensor
        Pre-allocated ``[m, hidden_size]`` bfloat16 destination; allocated here
        when omitted.

    Returns
    -------
    torch.Tensor
        ``output``.
    """
    global _FINALIZE_LAUNCH_COUNT

    if (shared_out is None) != (shared_gate is None):
        raise ValueError(
            "moe_routing_finalize needs shared_out and shared_gate together or "
            "not at all"
        )

    m, top_k, hidden_size = expert_out.shape
    if output is None:
        output = torch.empty(
            m, hidden_size, dtype=expert_out.dtype, device=expert_out.device
        )

    dispatched = False
    try:
        if _should_use_specialized_finalize(
            expert_out, shared_out, topk_weights, shared_gate, output
        ):
            if _dispatch_ready():
                _MODULE.moe_routing_finalize_sm120(
                    expert_out, shared_out, topk_weights, shared_gate, output
                )
                dispatched = True
    except Exception as exc:  # noqa: BLE001 - a guard must never break the op
        _jit_logger().warning_once(
            "Specialized MoE routing finalize unavailable, using the composable "
            "path: %s: %s",
            type(exc).__name__,
            str(exc)[:1000],
        )
        dispatched = False

    if dispatched:
        _FINALIZE_LAUNCH_COUNT += 1
        _note_dispatch(
            "finalize",
            f"m={int(m)}, hidden={int(hidden_size)}, top_k={int(top_k)}",
        )
    else:
        _reference_finalize(expert_out, shared_out, topk_weights, shared_gate, output)

    return output


def moe_routing_stats() -> dict:
    """Compile footprint + dispatch counters for the specialized SM120 kernels.

    The launch counts are host-side dispatches, per half; a CUDA-graph capture
    counts once and replays do not count, so they prove *which* implementation
    was recorded into a graph rather than how often the graph ran.

    ``persistent_device_state_bytes`` is 0 and is reported explicitly: the
    split has no cross-launch state and no inter-CTA rendezvous, so a launch's
    result cannot depend on any earlier launch.
    """
    workloads = load_moe_routing_sm120_workloads()
    return {
        "available": _sm120_available(),
        "allowlist_rows": len(workloads),
        # One translation unit holds both halves and every supported size, so
        # the whole allowlist is covered by a single compile.
        "compiled_variants": 1 if _MODULE is not None else 0,
        "distinct_kernels_for_allowlist": 3,
        "entry_points": 3,
        "compile_cache_key": (
            "arch=sm120a; source=moe_routing_sm120.cu (no problem size in the key)"
        ),
        "persistent_device_state_bytes": 0,
        # Capability flags, so a consumer can gate on what this build's API can
        # actually do instead of sniffing a version.  `shared_out`/`shared_gate`
        # became optional after the first revision of this op.
        "finalize_optional_shared_expert": True,
        "has_align_entry_point": True,
        "precompiled": _MODULE is not None,
        "ready_for_graph_capture": moe_routing_ready_for_graph_capture(),
        "prologue_launch_count": _PROLOGUE_LAUNCH_COUNT,
        "align_launch_count": _ALIGN_LAUNCH_COUNT,
        "finalize_launch_count": _FINALIZE_LAUNCH_COUNT,
        "launch_count": (
            _PROLOGUE_LAUNCH_COUNT + _ALIGN_LAUNCH_COUNT + _FINALIZE_LAUNCH_COUNT
        ),
    }
