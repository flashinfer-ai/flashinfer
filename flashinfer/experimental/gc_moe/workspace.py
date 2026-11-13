"""Owned symmetric workspace for the experimental GC MoE API."""

from __future__ import annotations

import threading

from . import support
from .kernel_src.gc_moe import create_buffer, destroy_buffer, translated


class GCMoEWorkspace:
    """Symmetric GC MoE workspace shared by every rank of ``group``.

    Exposes the upstream staging views (``x``, ``x_sf``, ``topk_idx``,
    ``topk_weights``, ``shared_l1_acts``, ...) by delegation. Calls that share
    one workspace must be serialized; captured graphs must be retired before
    :meth:`destroy`, which also releases the cached TensorMaps and their weight
    references.
    """

    def __init__(
        self,
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts=0,
        *,
        base=None,
        backend=None,
        ptxas=None,
    ):
        support.check_cutlass_dsl()
        support.validate_group(group)
        support.check_symmetric_memory()
        self.backend = support.resolve_backend(backend)
        if base is not None:
            if self.backend != "sm100":
                raise ValueError(
                    "base= is only supported by the SM100 backend; the SM90 and "
                    "SM120 workspace classes do not implement it"
                )
            if isinstance(base, GCMoEWorkspace):
                base = base._buffer
        self._lock = threading.Lock()
        self._fingerprint = None
        self._closed = False
        with translated("creating the GC MoE workspace"):
            self._buffer = create_buffer(
                self.backend,
                group,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                num_shared_experts=num_shared_experts,
                base=base,
                ptxas=ptxas,
            )

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        buffer = self.__dict__.get("_buffer")
        if buffer is None:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}; "
                "the workspace may have been destroyed"
            )
        return getattr(buffer, name)

    def destroy(self) -> None:
        buffer = self.__dict__.get("_buffer")
        if buffer is None:
            return
        self._buffer = None
        self._closed = True
        with translated("destroying the GC MoE workspace"):
            destroy_buffer(buffer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.destroy()
        return False


def create_workspace(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    num_shared_experts=0,
    *,
    base=None,
    backend=None,
    ptxas=None,
) -> GCMoEWorkspace:
    return GCMoEWorkspace(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        base=base,
        backend=backend,
        ptxas=ptxas,
    )
