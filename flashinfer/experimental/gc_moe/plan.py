"""Execution-plan knobs for GC MoE.

The vendored runtime reads its tuning switches from process environment
variables. They are applied here, explicitly, at workspace/call time instead
of being inherited silently:

======================================  ==========  ==========  ==========
field                                   sm90        sm100       sm120
======================================  ==========  ==========  ==========
``dispatch_sms``                        SM90_DISP   DISPATCH    SM120_DISP
``nvlink_barrier_timeout_secs``         NVLINK      NVLINK      NVLINK
``ptxas``                               PTXAS       PTXAS       PTXAS
======================================  ==========  ==========  ==========

(``GC_MOE_`` prefixes the right-hand names; ``NVLINK`` is read by the SM100
path only.) The primary-context and balanced-workqueue switches are
deliberately not plan fields: the shim pins ``GC_MOE_PRIMARY_CTX`` to ``1``
before the vendored sources are imported, and with every role kernel in the
primary context the ``GC_MOE_*_BALANCED_WORKQUEUES`` switches can never take
effect. Upstream caches its Green Context pairs thread-locally, keyed by
(CUDA context, ``dispatch_sms``) on SM100 but by the CUDA context alone on
SM90/SM120. On SM90/SM120 the switches therefore latch at the first launch in
a thread, and the dispatcher rejects a later workspace in that thread that
changes them; SM100 builds a separate pair for a changed ``dispatch_sms``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading

_FIELD_ENV = {
    "sm90": {
        "dispatch_sms": "GC_MOE_SM90_DISPATCH_SMS",
        "nvlink_barrier_timeout_secs": "GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS",
        "ptxas": "GC_MOE_PTXAS",
    },
    "sm100": {
        "dispatch_sms": "GC_MOE_DISPATCH_SMS",
        "nvlink_barrier_timeout_secs": "GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS",
        "ptxas": "GC_MOE_PTXAS",
    },
    "sm120": {
        "dispatch_sms": "GC_MOE_SM120_DISPATCH_SMS",
        "nvlink_barrier_timeout_secs": "GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS",
        "ptxas": "GC_MOE_PTXAS",
    },
}

_FIELDS = tuple(_FIELD_ENV["sm100"])

_LOCK = threading.RLock()


def _env_names(backend: str) -> dict[str, str]:
    names = _FIELD_ENV.get(backend)
    if names is None:
        raise ValueError(
            f"unknown GC MoE backend {backend!r}; expected one of {tuple(_FIELD_ENV)}"
        )
    return names


@dataclass(frozen=True)
class GcMoePlan:
    """Per-backend overrides for the vendored runtime switches.

    Use it as a context manager around the workspace creation and the calls
    that should share one configuration; leaving the block restores the
    previous environment. The switches are read at the first launch, not at
    workspace creation, so keep the block open through that call. Values must
    stay fixed for the lifetime of a workspace because the upstream runtime
    caches its Green Context pairs.

    ``applied`` mutates ``os.environ``, which is process-global: blocks are
    serialized by a process-wide lock, and code in other threads observes the
    block's values for its duration.
    """

    backend: str | None = None
    dispatch_sms: int | None = None
    nvlink_barrier_timeout_secs: int | None = None
    ptxas: str | None = None

    def _resolved(self, backend: str) -> dict[str, str]:
        names = _env_names(backend)
        if self.backend is not None and self.backend != backend:
            raise ValueError(
                f"plan targets backend {self.backend!r} but the call uses {backend!r}"
            )
        values = {}
        for field in _FIELDS:
            value = getattr(self, field)
            if value is not None:
                values[names[field]] = str(value)
        return values

    @contextmanager
    def applied(self, backend: str):
        """Temporarily export this plan's switches for ``backend``."""
        values = self._resolved(backend)
        with _LOCK:
            previous = {name: os.environ.get(name) for name in values}
            try:
                os.environ.update(values)
                yield self
            finally:
                for name, old in previous.items():
                    if old is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = old


def environment_fingerprint(backend: str) -> tuple:
    """Switches that cannot change once a workspace has launched."""
    names = _env_names(backend)
    return tuple(os.environ.get(names[field]) for field in _FIELDS)


def block_m(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
) -> int:
    """Upstream tile height for one call; useful to size staging buffers."""
    from .kernel_src.gc_moe import block_m as upstream_block_m

    return upstream_block_m(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_tokens,
        num_topk,
    )
