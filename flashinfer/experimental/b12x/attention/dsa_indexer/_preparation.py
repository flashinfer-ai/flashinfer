"""Prepared native DSA indexer plan.

Declarations retain tensor ABI metadata only; scratch layouts and launch carriers are
created only after a session chose and admitted the immutable configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compiler import observe_launchers
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation.types import require_prepared

from ._tuning import DsaIndexerConfig, DsaIndexerQuery, TUNING
from .mxfp4 import MXFP4PreparedState, materialize_mxfp4
from .paged import index_topk_fp8
from .tiled_topk import run_row_topk
from .scratch import (
    B12XIndexerScratchCaps,
    INDEXER_SOURCE_LAYOUT_PAGED,
    plan_indexer_scratch,
)


@dataclass(frozen=True)
class _Mxfp4Programs:
    launchers: Mapping

    @property
    def __b12x_dependencies__(self):
        return tuple(self.launchers.values())

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys

        return program_keys(self.__b12x_dependencies__)


class _DsaIndexerState:
    """FP8 prepared state retaining its scratch layout and native launchers."""

    def __init__(self, layout, config, launchers):
        self.layout = layout
        self.config = config
        self.launchers = MappingProxyType(dict(launchers))

    def bind(self, **kwargs):
        return self.layout.bind(**kwargs)

    def run(
        self,
        runtime,
        *,
        q_fp8,
        query_weights,
        index_k_cache,
        output_indices,
        output_scores=None,
        **_ignored,
    ):
        return index_topk_fp8(
            q_fp8=q_fp8,
            weights=query_weights,
            index_k_cache=index_k_cache,
            binding=runtime,
            page_size=self.layout.caps.page_size,
            topk=self.layout.caps.topk,
            expected_num_q_heads=self.layout.caps.num_q_heads,
            out_indices=output_indices,
            out_scores=output_scores,
            allow_transient_fold_buffers=False,
            launchers=self.launchers,
        )


_OPERANDS = (
    "q_fp8",
    "query_weights",
    "index_k_cache",
    "page_table",
    "cache_lengths",
    "active_width",
    "output_indices",
    "output_scores",
)


def _alignment(tensor):
    pointer = int(tensor.data_ptr())
    return min(16, pointer & -pointer) if pointer else 16


def invocation_from_descriptors(
    caps, *, operands: Mapping[str, Mapping[str, object] | None]
) -> FrozenMapping:
    """Build declaration metadata from an owner-provided tensor ABI.

    Providers use this before temporary priming tensors exist.  It deliberately
    records only immutable shape/stride/dtype/alignment properties, never a
    storage address or a live runtime value.
    """
    if not hasattr(caps, "num_q_heads"):
        raise TypeError("invocation metadata requires dsa_indexer.Caps")
    normalized = {}
    for name in _OPERANDS:
        descriptor = operands.get(name)
        if descriptor is None:
            normalized[name] = None
            continue
        fields = FrozenMapping(descriptor)
        required = {"shape", "strides", "dtype", "alignment"}
        if set(fields) != required:
            raise ValueError(f"DSA {name} ABI descriptor fields do not match schema")
        if name == "index_k_cache":
            # Physical pool size is a runtime bound, not a kernel specialization.
            fields = FrozenMapping({**dict(fields), "shape": (1, *fields["shape"][1:])})
        normalized[name] = fields
    return FrozenMapping(
        {
            "operands": FrozenMapping(normalized),
            "route": getattr(caps, "route", "auto"),
            "page_size": int(getattr(caps, "page_size", 64)),
            "requested_supertile_k": int(getattr(caps, "supertile_k", 0)),
            "requested_prefill_block_k": int(getattr(caps, "prefill_block_k", 256)),
            "reserve_paged_logits": bool(getattr(caps, "reserve_paged_logits", False)),
            "paged_logits_k_rows": int(getattr(caps, "paged_logits_k_rows", 0)),
            "score_mode": str(getattr(caps, "score_mode", "dsa")),
            "num_idx_heads": int(getattr(caps, "num_idx_heads", 1)),
        }
    )


def invocation_from_tensors(caps, **tensors) -> FrozenMapping:
    """Describe all static native ABI properties from caller-owned tensors."""
    return invocation_from_descriptors(
        caps,
        operands={
            name: None
            if (value := tensors.get(name)) is None
            else {
                "shape": tuple(int(v) for v in value.shape),
                "strides": tuple(int(v) for v in value.stride()),
                "dtype": str(value.dtype).removeprefix("torch."),
                "alignment": _alignment(value),
            }
            for name in _OPERANDS
        },
    )


def _fake(descriptor, device):
    if descriptor is None:
        return None
    shape = tuple(descriptor["shape"])
    strides = tuple(descriptor["strides"])
    dtype = getattr(torch, descriptor["dtype"])
    alignment = int(descriptor["alignment"])
    if alignment >= 16:
        return torch.empty_strided(shape, strides, dtype=dtype, device=device)
    offset = alignment // dtype.itemsize
    span = (
        1
        + sum((size - 1) * stride for size, stride in zip(shape, strides, strict=True))
        if all(shape)
        else 0
    )
    storage = torch.empty((span + offset,), dtype=dtype, device=device)
    return storage.as_strided(shape, strides, storage_offset=offset)


def _scratch_caps(query, *, device, config):
    return B12XIndexerScratchCaps(
        device=device,
        source_layout=query.source_layout,
        num_q_heads=query.num_q_heads,
        max_q_rows=query.max_q_rows,
        max_k_rows=query.max_k_rows,
        max_page_table_width=query.max_page_table_width,
        topk=query.top_k,
        mode=query.mode,
        page_size=query.page_size,
        supertile_k=query.supertile_k,
        shared_page_table=query.shared_page_table,
        output_physical_slots=query.output_physical_slots,
        dtype=getattr(torch, query.dtype),
        kv_dtype=getattr(torch, query.kv_dtype),
        route=query.route,
        prefill_block_k=query.prefill_block_k,
        reserve_paged_logits=query.reserve_paged_logits,
        paged_logits_k_rows=query.paged_logits_k_rows,
        score_mode=query.score_mode,
        num_idx_heads=query.num_idx_heads,
    )


@program_cache(scope="preparation")
def compile_indexer(query_payload, config_payload, ordinal):
    """Return retained concrete launchers for one selected native route.

    The factory operates solely on declaration ABI metadata.  ``compile_only``
    prevents a fake invocation from executing while preserving the exact
    production launcher construction and compiler identities.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    query = DsaIndexerQuery(**dict(query_payload))
    config = DsaIndexerConfig.from_config(FrozenMapping(config_payload))
    if query.source_layout != INDEXER_SOURCE_LAYOUT_PAGED:
        raise ValueError("DSA public preparation requires paged source metadata")
    device = torch.device("cuda", ordinal)
    if query.cache_format == "mxfp4":
        from types import SimpleNamespace

        caps = SimpleNamespace(
            device=device,
            num_q_heads=query.num_q_heads,
            max_q_rows=query.max_q_rows,
            max_page_table_width=query.max_page_table_width,
            topk=query.top_k,
            mode=query.mode,
            page_size=query.page_size,
            max_candidates=query.max_candidates,
            candidate_topk_blocks=query.candidate_topk_blocks,
        )
        state = materialize_mxfp4(
            caps, device_index=ordinal, score_kind=config.mxfp4_score_kind
        )
        return _Mxfp4Programs(state._launchers)

    descriptors = query.operands
    gathered: dict[object, object] = {}
    with FakeTensorMode(), compile_only_launches(), observe_launchers() as resolved:
        values = {name: _fake(descriptors[name], device) for name in _OPERANDS}
        layout = plan_indexer_scratch(
            _scratch_caps(query, device=device, config=config),
            fused_merge=config.fused_merge,
        )
        scratch = torch.empty(
            layout.scratch_specs()[0].shape, dtype=torch.uint8, device=device
        )
        binding = layout.bind(
            scratch=scratch,
            real_page_table=values["page_table"],
            cache_seqlens_int32=values["cache_lengths"],
            active_width=values["active_width"],
            expected_num_q_heads=query.num_q_heads,
            shared_page_table=query.shared_page_table,
            output_physical_slots=query.output_physical_slots,
            _initialize=False,
        )
        index_topk_fp8(
            q_fp8=values["q_fp8"],
            weights=values["query_weights"],
            index_k_cache=values["index_k_cache"],
            binding=binding,
            page_size=query.page_size,
            topk=query.top_k,
            expected_num_q_heads=query.num_q_heads,
            out_indices=values["output_indices"],
            out_scores=values["output_scores"],
            allow_transient_fold_buffers=False,
            launcher_sink=gathered,
        )
        if binding.route != "paged_fused":
            # A single live supertile combines the first and final top-k arms.
            short_width = min(
                query.max_page_table_width,
                max(1, int(binding.supertile_k) // query.page_size),
            )
            if short_width < query.max_page_table_width:
                short_binding = layout.bind(
                    scratch=scratch,
                    real_page_table=torch.empty(
                        (values["page_table"].shape[0], short_width),
                        dtype=values["page_table"].dtype,
                        device=device,
                    ),
                    cache_seqlens_int32=values["cache_lengths"],
                    active_width=values["active_width"],
                    expected_num_q_heads=query.num_q_heads,
                    shared_page_table=query.shared_page_table,
                    output_physical_slots=query.output_physical_slots,
                    _initialize=False,
                )
                index_topk_fp8(
                    q_fp8=values["q_fp8"],
                    weights=values["query_weights"],
                    index_k_cache=values["index_k_cache"],
                    binding=short_binding,
                    page_size=query.page_size,
                    topk=query.top_k,
                    expected_num_q_heads=query.num_q_heads,
                    out_indices=values["output_indices"],
                    out_scores=values["output_scores"],
                    launchers=(
                        {"paged": resolved[0], "tiled": None}
                        if binding.route == "paged_tiled"
                        else None
                    ),
                    allow_transient_fold_buffers=False,
                    launcher_sink=gathered,
                )
            # The two-stage fold is a declared native family even when this
            # serving state uses its fixed-scratch carry path. Compile its
            # final row fold from the same exact row/top-k ABI; it remains
            # retained for prepared capacity variants rather than becoming a
            # runtime cache miss.
            row_logits = torch.empty(
                (query.max_q_rows, query.top_k), dtype=torch.float32, device=device
            )
            run_row_topk(
                row_logits=row_logits,
                lengths=values["cache_lengths"],
                topk=query.top_k,
                output_values=values["output_scores"],
                output_indices=values["output_indices"],
                output_gather_table=values["output_indices"],
                launcher_sink=gathered,
            )

    route = binding.route
    tiled_launchers = {
        key: value
        for key, value in gathered.items()
        if isinstance(key, tuple) and key[0] in ("tiled", "row")
    }
    if route == "paged_fused":
        if len(resolved) != 1:
            raise RuntimeError("prepared fused DSA route did not resolve one launcher")
        return {"fused": resolved[0]}
    if route == "packed_contiguous":
        if "gather" not in gathered or not resolved:
            raise RuntimeError(
                "prepared contiguous DSA route did not resolve every launcher"
            )
        return {
            "gather": gathered["gather"],
            "contiguous": resolved[0],
            "tiled": MappingProxyType(tiled_launchers),
        }
    if not resolved:
        raise RuntimeError("prepared paged DSA route did not resolve its scorer")
    return {"paged": resolved[0], "tiled": MappingProxyType(tiled_launchers)}


def _query(caps, invocation):
    operands = invocation.get("operands", FrozenMapping())
    if not isinstance(operands, FrozenMapping):
        raise ValueError("DSA invocation operands must be immutable metadata")
    if caps.cache_format != "mxfp4" and any(
        name not in operands or operands[name] is None for name in _OPERANDS[:-1]
    ):
        raise ValueError(
            "DSA FP8 declarations require invocation_from_tensors metadata"
        )
    return DsaIndexerQuery(
        source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
        mode=caps.mode,
        dtype="bfloat16",
        kv_dtype="uint8",
        num_q_heads=caps.num_q_heads,
        num_idx_heads=int(
            invocation.get("num_idx_heads", getattr(caps, "num_idx_heads", 1))
        ),
        max_q_rows=caps.max_q_rows,
        max_k_rows=caps.max_page_table_width
        * int(invocation.get("page_size", caps.page_size)),
        top_k=caps.topk,
        page_size=int(invocation.get("page_size", caps.page_size)),
        score_mode=str(
            invocation.get("score_mode", getattr(caps, "score_mode", "dsa"))
        ),
        shared_page_table=caps.mode == "prefill",
        max_page_table_width=caps.max_page_table_width,
        route=str(invocation.get("route", getattr(caps, "route", "auto"))),
        output_physical_slots=caps.output_index_space == "physical",
        supertile_k=int(
            invocation.get("requested_supertile_k", getattr(caps, "supertile_k", 0))
        ),
        prefill_block_k=int(
            invocation.get(
                "requested_prefill_block_k", getattr(caps, "prefill_block_k", 256)
            )
        ),
        reserve_paged_logits=bool(
            invocation.get(
                "reserve_paged_logits", getattr(caps, "reserve_paged_logits", False)
            )
        ),
        paged_logits_k_rows=int(
            invocation.get(
                "paged_logits_k_rows", getattr(caps, "paged_logits_k_rows", 0)
            )
        ),
        cache_format=caps.cache_format,
        max_candidates=caps.max_candidates,
        candidate_topk_blocks=caps.candidate_topk_blocks,
        operands=operands,
    )


def plan(
    caps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: DsaIndexerConfig | None = None,
) -> Plan:
    if not hasattr(caps, "max_page_table_width"):
        raise TypeError("plan requires dsa_indexer.Caps")
    invocation = FrozenMapping(invocation)
    query = _query(caps, invocation)

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.attention.dsa_indexer._preparation:compile_indexer",
                TUNING.encode_query(replace(query, exhaustive=False)),
                TUNING.encode_config(config),
                device.ordinal,
            ),
        )

    def memory(config, device):
        if query.cache_format == "mxfp4":
            from types import SimpleNamespace
            from .mxfp4 import plan_mxfp4

            mx_caps = SimpleNamespace(
                device=caps.device,
                num_q_heads=query.num_q_heads,
                max_q_rows=query.max_q_rows,
                max_page_table_width=query.max_page_table_width,
                topk=query.top_k,
                mode=query.mode,
                page_size=query.page_size,
                max_candidates=query.max_candidates,
                candidate_topk_blocks=query.candidate_topk_blocks,
            )
            return MemoryRequirements(scratch=plan_mxfp4(mx_caps).scratch_specs())
        layout = plan_indexer_scratch(
            _scratch_caps(query, device=caps.device, config=config),
            fused_merge=config.fused_merge,
        )
        return MemoryRequirements(scratch=layout.scratch_specs())

    def materialize(selection, device):
        programs = compile_indexer(
            TUNING.encode_query(replace(query, exhaustive=False)),
            TUNING.encode_config(selection.config),
            device.ordinal,
        )
        load_programs(programs)
        if query.cache_format == "mxfp4":
            from types import SimpleNamespace
            from .mxfp4 import plan_mxfp4

            mx_caps = SimpleNamespace(
                device=caps.device,
                num_q_heads=query.num_q_heads,
                max_q_rows=query.max_q_rows,
                max_page_table_width=query.max_page_table_width,
                topk=query.top_k,
                mode=query.mode,
                page_size=query.page_size,
                max_candidates=query.max_candidates,
                candidate_topk_blocks=query.candidate_topk_blocks,
            )
            layout = plan_mxfp4(mx_caps, score_kind=selection.config.mxfp4_score_kind)
            return MXFP4PreparedState(layout, programs.launchers)
        layout = plan_indexer_scratch(
            _scratch_caps(query, device=caps.device, config=selection.config),
            fused_merge=selection.config.fused_merge,
        )
        launchers = programs
        if not isinstance(launchers, Mapping):
            raise TypeError("FP8 DSA compiler factory did not return launchers")
        return attach_programs(
            _DsaIndexerState(layout, selection.config, launchers), launchers
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )


def bind(plan, **kwargs):
    return require_prepared(plan, "attention.dsa_indexer").bind(**kwargs)


def run(
    *,
    binding,
    plan,
    q_fp8,
    query_weights,
    index_k_cache,
    output_indices,
    output_scores=None,
):
    return require_prepared(plan, "attention.dsa_indexer").run(
        binding,
        q_fp8=q_fp8,
        query_weights=query_weights,
        index_k_cache=index_k_cache,
        output_indices=output_indices,
        output_scores=output_scores,
    )


__all__ = [
    "bind",
    "invocation_from_descriptors",
    "invocation_from_tensors",
    "plan",
    "run",
]
