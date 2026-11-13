"""The only module that imports the vendored sources.

``env.prepare`` must run before the first ``..src`` import because the
vendored JIT resolves its PTX assembler at import time. Everything else in the
integration reaches the kernels through here.
"""

from __future__ import annotations

import os
import sys

from . import env

_SOURCES: dict | None = None
_PTXAS: str | None = None
_USED: set[str] = set()

_RUNTIME_MODULES = {
    "sm90": "sm90_gc_moe",
    "sm100": "gc_moe",
    "sm120": "sm120_gc_moe",
}


def _prepare(ptxas: str | None) -> None:
    global _PTXAS
    if _PTXAS is None:
        _PTXAS = env.prepare(ptxas)
        return
    if ptxas is not None and os.path.realpath(ptxas) != os.path.realpath(_PTXAS):
        raise RuntimeError(
            "flashinfer.experimental.gc_moe: the GC MoE kernels already "
            f"compiled against {_PTXAS!r}; ptxas cannot change to {ptxas!r} "
            "after the first launch"
        )


def sources(ptxas: str | None = None) -> dict:
    global _SOURCES
    _prepare(ptxas)
    if _SOURCES is None:
        from ..src import gc as sm100
        from ..src import sm120_gc, sm90_gc

        _SOURCES = {"sm90": sm90_gc, "sm100": sm100, "sm120": sm120_gc}
    return _SOURCES


def module(backend: str, ptxas: str | None = None):
    _USED.add(backend)
    return sources(ptxas)[backend]


def run(
    backend: str,
    *,
    y,
    l1_weights,
    l2_weights,
    buffer,
    shared_l1_weights,
    shared_l2_weights,
    cumulative_local_expert_recv_stats,
    recipe,
    activation,
    activation_clamp,
    fast_math,
    shared_recipe,
    activation_alpha,
    num_tokens_bound,
    ptxas=None,
):
    source = module(backend, ptxas)
    if backend == "sm100":
        return source.fp8_fp4_gc_moe(
            y,
            l1_weights,
            l2_weights,
            buffer,
            shared_l1_weights=shared_l1_weights,
            shared_l2_weights=shared_l2_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            recipe=recipe,
            activation=activation,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            shared_recipe=shared_recipe,
            activation_alpha=activation_alpha,
        )
    entry = (
        source.sm90_fp8_fp4_gc_moe if backend == "sm90" else source.sm120_fp8_fp4_gc_moe
    )
    return entry(
        y,
        l1_weights,
        l2_weights,
        buffer,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        num_tokens_bound=num_tokens_bound,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        shared_recipe=shared_recipe,
    )


def transform(
    backend: str,
    l1_weights,
    l2_weights,
    *,
    l1_scale_2=None,
    l2_scale_2=None,
    shared_recipe=None,
    activation="swiglu",
    ptxas=None,
):
    source = module(backend, ptxas)
    if backend == "sm100":
        return source.transform_weights_for_gc_moe(
            l1_weights, l2_weights, activation=activation
        )
    if backend == "sm90":
        return source.transform_weights_for_gc_moe_sm90(
            l1_weights,
            l2_weights,
            l1_scale_2,
            l2_scale_2,
            (1, 1, 32) if shared_recipe is None else shared_recipe,
        )
    return source.transform_weights_for_gc_moe_sm120(
        l1_weights, l2_weights, activation=activation, shared_recipe=shared_recipe
    )


def block_m(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk):
    return sources()["sm100"].get_block_m_for_gc_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk
    )


def close_runtime() -> list[str]:
    """Retire every Green Context pair created through this shim.

    Terminal for launches in this process: the vendored runtime refuses new
    work afterwards. Every captured graph that references the workspace must
    be retired first. Returns the failures instead of hiding them.
    """
    prefix = __package__.rsplit(".", 1)[0] + ".src.runtime."
    failures = []
    for backend in sorted(_USED):
        handle = sys.modules.get(prefix + _RUNTIME_MODULES[backend])
        close = getattr(handle, "_close_runtime", None) if handle else None
        if close is None:
            continue
        try:
            close()
        except Exception as exc:  # reported, not swallowed
            failures.append(f"{backend}: {exc}")
    return failures
