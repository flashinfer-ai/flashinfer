"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

"""Automatic MLA planning: architecture dispatch, ranking and typed fallback.

Ranking only orders preferences from request metadata. Concrete backend
preflight owns eligibility; candidate preparation may compile a native plan.
"""

from typing import TYPE_CHECKING, cast

import torch

from ...api_logging import (
    _warn_from_external_caller,
    experimental_auto_backends_allowed,
)
from ...utils import (
    determine_mla_backend,
    get_compute_capability as _get_compute_capability,
)
from ._backends._capabilities import _BackendPlanUnsupportedError
from ._backends.cutile_backend import _CUTILE_SUPPORTED_COMPUTE_CAPABILITIES
from ._planning import _MLAPlanArguments

if TYPE_CHECKING:
    from ._wrapper import _ConcreteWrapperBackendType, _PlannedBackend


class _BatchMLAPagedAttentionAutoBackend:
    """Resolve the automatic policy to a prepared concrete backend."""

    _blackwell_auto_fallback_warned: bool = False

    @classmethod
    def _maybe_warn_blackwell_auto_fallback(
        cls, device: torch.device, selected_backend: str
    ) -> None:
        if cls._blackwell_auto_fallback_warned:
            return
        major, minor = _get_compute_capability(device)
        if major < 10:
            return
        cls._blackwell_auto_fallback_warned = True
        if (major, minor) in _CUTILE_SUPPORTED_COMPUTE_CAPABILITIES:
            in_wrapper_alternative = (
                "backend='cutile' is the native in-wrapper cuda.tile alternative."
            )
        else:
            in_wrapper_alternative = (
                "backend='cutlass' is the closest in-wrapper alternative but may be "
                "slower than this fallback for decode shapes."
            )
        _warn_from_external_caller(
            f"BatchMLAPagedAttentionWrapper: backend='auto' selected "
            f"'{selected_backend}' on SM{major}{minor}, which is not Blackwell-native "
            f"and gives poor MLA decode performance. For decode, use "
            f"flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla "
            f"(Blackwell-native trtllm-gen); {in_wrapper_alternative}",
            UserWarning,
        )

    @classmethod
    def plan_from_wrapper(cls, plan_args: _MLAPlanArguments) -> "_PlannedBackend":
        # The wrapper registers this class during import; resolve its shared
        # machinery only when planning starts, after initialization is complete.
        from ._wrapper import _BACKEND_TYPES

        plan_args.csr()  # Malformed metadata is never a candidate support refusal.
        device = plan_args._float_workspace_buffer.device
        if plan_args._use_cuda_graph and plan_args._previous_backend_name is not None:
            # Retain the executable family used by the existing graph plan.
            candidates: tuple[str, ...] = (plan_args._previous_backend_name,)
        elif _get_compute_capability(device) == (10, 0):
            candidates = ordered_sm100_backends(plan_args)
        else:
            backend = determine_mla_backend(device)
            cls._maybe_warn_blackwell_auto_fallback(device, backend)
            candidates = (backend,)
        rejections: list[str] = []
        for candidate in candidates:
            try:
                backend_type = cast(
                    "type[_ConcreteWrapperBackendType]", _BACKEND_TYPES[candidate]
                )
                if (
                    backend_type._plan_capabilities.is_experimental
                    and not experimental_auto_backends_allowed()
                ):
                    raise _BackendPlanUnsupportedError(
                        "experimental backend requires "
                        "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1"
                    )
                return backend_type.plan_from_wrapper(plan_args)
            except _BackendPlanUnsupportedError as error:
                if len(candidates) == 1:
                    raise
                rejections.append(f"{candidate}: {error}")
        raise _BackendPlanUnsupportedError(
            "No supported planned MLA backend: " + "; ".join(rejections)
        )


_AUTO_BACKEND_CANDIDATES = (
    "cute-dsl-monolithic",
    "trtllm-gen",
    "fa2",
    "cute-dsl-modular",
    "cutile",
    "cutlass",
)


def _prefer(*backends):
    return backends + tuple(b for b in _AUTO_BACKEND_CANDIDATES if b not in backends)


# SM100 warmed planned-wrapper measurements favor monolithic CuTe at larger
# work sizes. Keep FA2 before cuTile for native independent-split/page-1 fallback.
def ordered_sm100_backends(args):
    """Order every candidate from actual lengths, without deciding support.

    These coarse work boundaries approximate measured launch/throughput
    crossovers, not exact optima. Metadata transfers are plan-time setup only;
    re-read on each plan because callers may mutate the metadata in place.
    """
    csr = args.csr()
    offsets = csr.qo_indptr.cpu().tolist()
    kv_lens = csr.kv_len_arr.cpu().tolist()
    q_lens = [end - begin for begin, end in zip(offsets, offsets[1:], strict=False)]
    # Legacy flat metadata can have extra query offsets. Leave its interpretation
    # to the backend instead of imposing a new validity restriction here.
    if not q_lens or len(q_lens) != len(kv_lens):
        return _AUTO_BACKEND_CANDIDATES
    total_q, total_kv = sum(q_lens), sum(kv_lens)
    if not total_q or not total_kv:
        return _AUTO_BACKEND_CANDIDATES
    heads = args.num_heads
    work = heads * sum(q * kv for q, kv in zip(q_lens, kv_lens, strict=True))
    output_work = heads * total_q
    max_q, min_kv, max_kv = max(q_lens), min(kv_lens), max(kv_lens)
    batch = len(q_lens)
    small_work = work <= 2**21 and output_work <= 32768
    if not args._use_cuda_graph:
        # Packed tensors and adjacent split views share the same plan facts.
        # Retain FA2 where conversion/launch costs erase the packed-input gain.
        if small_work and max_kv <= 4096:
            return _prefer("fa2", "trtllm-gen")
        within_eager_limit = (
            work < 5 * 2**20
            if heads >= 128 and (max_q > 1 or output_work <= 2048 or max_kv <= 2048)
            else work <= 2**22
        )
        # Larger aggregate KV volume amortizes conversion even for adjacent
        # views. Share this exemption across both secondary eager guards.
        if (
            within_eager_limit
            and output_work <= 32768
            and max_kv < 32768
            and total_kv < 5 * 2**14
        ):
            return _prefer("fa2", "trtllm-gen")
        if (
            max_q == 1
            and work < 5 * 2**20
            and max_kv <= 4096
            and output_work <= 2048
            and batch < 32
            and total_kv < 5 * 2**14
        ):
            return _prefer("fa2", "trtllm-gen")
    # Base-2 LSE needs a prefix-aware preference in both execution modes.
    # The per-request margin is empirical, not a native support restriction.
    query_factor = 4 if heads <= 8 else 2
    near_prefill = query_factor * total_q + 32 * batch >= total_kv
    if (
        args.lse_mode == "base2"
        and work <= 2**23
        and output_work <= 32768
        and max_q > 1
        and max_kv <= 1024
        and near_prefill
    ):
        return _prefer("fa2", "trtllm-gen")
    if args._use_cuda_graph:
        # Large short-context prefill amortizes TRT's launch cost. Preserve
        # CuTe for longer prefixes and lower head counts with different crossovers.
        if heads >= 64 and max_q >= 128 and output_work >= 65536 and max_kv <= 512:
            return _prefer("trtllm-gen")
        # Batched short prefill amortizes TRT's launch cost at larger head
        # counts. Lower-head requests retain CuTe even at the same output work.
        if heads >= 64 and output_work >= 24576 and 16 <= max_q <= 32 and max_kv <= 32:
            return _prefer("trtllm-gen")
        # TRT wins small short-context query tiles; CuTe wins as those tiles
        # grow. Preserve the older narrow tile rule for larger query counts.
        query_tile = max_q * heads
        if work <= 2**20 and max_kv <= 512 and (query_tile <= 128 or max_q <= 16):
            return (
                _prefer("trtllm-gen") if query_tile <= 576 else _AUTO_BACKEND_CANDIDATES
            )
        # Larger short-context decode batches favor TRT. Count actual queries
        # so empty metadata requests cannot trigger the throughput preference.
        if max_q == 1 and heads >= 64 and total_q >= 96 and max_kv <= 512:
            return _prefer("trtllm-gen")
        if heads >= 128:
            # Distinguish long KV from short-KV multi-query work with the same
            # aggregate size. Their measured TRT/CuTe crossovers differ.
            if min_kv >= 2048:
                graph_limit = 6 * 2**20 if max_kv <= 4096 else 2**22
                if 1 < max_q <= 8 and min_kv > 4096:
                    graph_limit = 2**23
                if work <= graph_limit:
                    return _AUTO_BACKEND_CANDIDATES
            return _prefer("trtllm-gen")
        if max_q == 1 and (
            min_kv >= 8192
            or (output_work > 384 and max_kv > 512)
            or (16 < heads < 32 and min_kv > 512 and max_kv <= 2048)
        ):
            return _AUTO_BACKEND_CANDIDATES
        if work <= 2**20 and max_q == 1:
            return _prefer("trtllm-gen")
        return _AUTO_BACKEND_CANDIDATES
    if heads >= 128:
        order = _prefer("trtllm-gen")
    elif small_work:
        # Long-context decode keeps CuTe before FA2 after typed TRT rejection.
        order = _prefer("trtllm-gen") if max_q == 1 else _prefer("trtllm-gen", "fa2")
    else:
        order = _AUTO_BACKEND_CANDIDATES
    # Larger multi-query work or aggregate KV volume can favor modular CuTe
    # after earlier implementations reject. Keep the eager guards authoritative.
    if max_q > 1 and (work >= 5 * 2**20 or total_kv >= 5 * 2**14):
        remaining = tuple(backend for backend in order if backend != "cute-dsl-modular")
        position = remaining.index("fa2")
        return remaining[:position] + ("cute-dsl-modular",) + remaining[position:]
    return order
