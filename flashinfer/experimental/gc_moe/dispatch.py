"""Architecture dispatch for the experimental GC MoE API."""

from __future__ import annotations

import threading

from . import plan
from . import support
from .kernel_src.gc_moe import run, translated
from .workspace import GCMoEWorkspace

# Upstream keys its SM90/SM120 Green context pairs by CUDA context alone, so
# the switches of the first launch in a thread stay in force for every later
# workspace in that thread. SM100 keys by the dispatch request as well.
_CONTEXT_KEYED_BACKENDS = frozenset({"sm90", "sm120"})
_LAUNCH_LATCH = threading.local()


def _check_launch_latch(backend: str, device: int, fingerprint: tuple) -> None:
    """Reject switch changes the thread-local context cache cannot honor.

    Upstream reuses one pair per CUDA context and thread; on SM90/SM120 that
    key ignores the switch values, so the first launch on a device fixes them
    for every later workspace in the same thread and device.
    """
    if backend not in _CONTEXT_KEYED_BACKENDS:
        return
    latched = getattr(_LAUNCH_LATCH, "fingerprints", None)
    if latched is None:
        latched = _LAUNCH_LATCH.fingerprints = {}
    previous = latched.setdefault((backend, device), fingerprint)
    if previous != fingerprint:
        raise RuntimeError(
            "flashinfer.experimental.gc_moe: GC MoE runtime switches changed "
            f"after an earlier {backend} workspace launched on device {device} "
            "in this thread, and the upstream Green context cache is keyed by "
            "CUDA context alone; keep one GcMoePlan per thread and device"
        )


def _call_shared_experts(shared_l2_weights, intermediate: int) -> int | None:
    """Shared experts carried by one transformed W2 pair, or None if unclear.

    Upstream re-slices the staging views to the call's shared count, but this
    API exposes only the views sized at workspace creation; a count mismatch
    would stage into one layout and launch with another.
    """
    payload = shared_l2_weights[0]
    shape = getattr(payload, "shape", None)
    if shape is None or len(shape) not in (2, 3):
        return None
    rows = int(shape[0]) * (128 if len(shape) == 3 else 1)
    if rows <= 0 or rows % intermediate:
        return None
    return rows // intermediate


def run_gc_moe(
    y,
    l1_weights,
    l2_weights,
    workspace,
    *,
    shared_l1_weights=None,
    shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1, 1, 32),
    activation="swiglu",
    activation_clamp=None,
    fast_math=True,
    shared_recipe=None,
    activation_alpha=None,
    num_tokens_bound=None,
    backend=None,
):
    if not isinstance(workspace, GCMoEWorkspace):
        raise TypeError(
            "workspace must come from flashinfer.GCMoEWorkspace(...); "
            f"got {type(workspace).__name__}"
        )
    resolved = workspace.backend
    if backend is not None and backend != resolved:
        raise ValueError(
            f"backend={backend!r} does not match the workspace backend {resolved!r}"
        )
    if workspace._closed:
        raise ValueError("the GC MoE workspace has been destroyed")
    support.validate_launch(activation, activation_alpha, num_tokens_bound, resolved)
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError(
            "shared_l1_weights and shared_l2_weights must be given together"
        )
    staged_shared = getattr(workspace, "num_shared_experts", 0)
    if (staged_shared > 0) != (shared_l1_weights is not None):
        raise ValueError(
            f"the workspace stages {staged_shared} shared experts but "
            f"{'no' if shared_l1_weights is None else 'a'} shared weight pair was "
            "passed; create the workspace with the same num_shared_experts as the call"
        )
    if shared_l1_weights is not None:
        call_shared = _call_shared_experts(
            shared_l2_weights, workspace.intermediate_hidden
        )
        if call_shared is not None and call_shared != staged_shared:
            raise ValueError(
                f"the workspace stages {staged_shared} shared experts but the shared "
                f"weight pair carries {call_shared}; the staging views are sized for "
                "the workspace count, so create the workspace for the call's shared "
                "experts"
            )

    fingerprint = plan.environment_fingerprint(resolved)
    if not workspace._lock.acquire(blocking=False):
        raise RuntimeError(
            "flashinfer.experimental.gc_moe: another gc_moe call is in flight on "
            "this workspace; calls that share one workspace must be serialized"
        )
    try:
        if workspace._fingerprint is None:
            workspace._fingerprint = fingerprint
        elif workspace._fingerprint != fingerprint:
            raise RuntimeError(
                "flashinfer.experimental.gc_moe: GC MoE runtime switches changed "
                "after this workspace launched, and the upstream Green Context "
                "cache cannot pick them up; use one GcMoePlan for the lifetime of "
                "the workspace"
            )
        _check_launch_latch(resolved, workspace.buffer.device.index, fingerprint)
        with translated("running gc_moe"):
            run(
                resolved,
                y=y,
                l1_weights=l1_weights,
                l2_weights=l2_weights,
                buffer=workspace._buffer,
                shared_l1_weights=shared_l1_weights,
                shared_l2_weights=shared_l2_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                recipe=recipe,
                activation=activation,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
                shared_recipe=shared_recipe,
                activation_alpha=activation_alpha,
                num_tokens_bound=num_tokens_bound,
            )
    finally:
        workspace._lock.release()
