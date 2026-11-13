"""Symmetric workspace lifecycle for the vendored sources."""

from __future__ import annotations

from . import launch

_BUFFER_CLASS = {
    "sm90": "SM90GCSymmBuffer",
    "sm100": "GCSymmBuffer",
    "sm120": "SM120GCSymmBuffer",
}


def create(
    backend: str,
    group,
    *,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    base=None,
    ptxas: str | None = None,
):
    source = launch.module(backend, ptxas)
    buffer_type = getattr(source, _BUFFER_CLASS[backend])
    arguments = dict(
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        activation="swiglu",
    )
    if backend == "sm100":
        return buffer_type(
            group,
            num_shared_experts=num_shared_experts,
            mma_type="fp8xfp4",
            base=base,
            **arguments,
        )
    return buffer_type(group, num_shared_experts=num_shared_experts, **arguments)


def destroy(buffer) -> None:
    buffer.destroy()
