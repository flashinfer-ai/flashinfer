"""Prepared plan owner for native paged attention.

The declaration is deliberately metadata-only.  The legacy scratch-plan object is
created only while a session materializes a selected plan, where its
shape-only CUDA views and fixed-address graph metadata are owned by that
plan rather than by a public declaration.
"""
from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.compile_plan import program_keys
from b12x.preparation import (
    DetectedDevice, DeviceIdentity, FrozenMapping, MemoryRequirements,
    PersistentMemory, Plan, current_plan, current_prepared_state, detect_device,
    require_prepared,
)
from b12x.preparation.types import _owned_tensor_nbytes

from ._forward import (
    compile_paged_launchers, install_paged_resources, materialize_paged_resources,
    run_paged_prepared,
)
from ._controls import paged_controls, snapshot_paged_controls
from ._scratch import (
    B12XPagedAttentionScratchCaps,
    paged_attention_scratch_specs,
    make_compile_bindings,
    variant_key,
    plan_paged_attention_scratch,
)
from ._tuning import GqaConfig, GqaQuery, TUNING
from .planner import plan_extend_graph_capacity, plan_verify_graph_capacity
_OPERANDS = (
    "q", "k_cache", "v_cache", "output", "page_table", "cache_seqlens",
    "cu_seqlens_q", "q2k_indices", "k_descale", "v_descale",
    "attention_sink_bias", "relative_attention_bias",
)


def _capturing() -> bool:
    """Report stream capture without assuming a usable CUDA context."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _alignment(tensor) -> int:
    pointer = int(tensor.data_ptr())
    return min(16, pointer & -pointer) if pointer else 16


def invocation_from_descriptors(
    caps: B12XPagedAttentionScratchCaps, *, operands: Mapping[str, Mapping[str, object] | None]
) -> FrozenMapping:
    """Capture a declaration ABI without allocating temporary CUDA tensors."""
    normalized = {}
    for name in _OPERANDS:
        descriptor = operands.get(name)
        if descriptor is None:
            normalized[name] = None
            continue
        fields = FrozenMapping(descriptor)
        if set(fields) != {"shape", "strides", "dtype", "alignment"}:
            raise ValueError(f"paged {name} ABI descriptor fields do not match schema")
        normalized[name] = fields
    return FrozenMapping({"operands": FrozenMapping(normalized)})


def invocation_from_tensors(caps: B12XPagedAttentionScratchCaps, **tensors) -> FrozenMapping:
    """Capture the complete immutable native ABI from real caller tensors."""
    return invocation_from_descriptors(
        caps,
        operands={
            name: None
            if (tensor := tensors.get(name)) is None
            else {
                "shape": tuple(int(value) for value in tensor.shape),
                "strides": tuple(int(value) for value in tensor.stride()),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "alignment": _alignment(tensor),
            }
            for name in _OPERANDS
        },
    )



def _query_from_caps(
    # Controls and capacity are declaration coordinates, never replay lookups.
    caps: B12XPagedAttentionScratchCaps, invocation: FrozenMapping
) -> GqaQuery:
    return GqaQuery(
        device=caps.device,
        mode=caps.mode,
        q_dtype=str(caps.dtype).removeprefix("torch."),
        kv_dtype=str(caps.kv_dtype).removeprefix("torch."),
        q_heads=caps.num_q_heads,
        kv_heads=caps.num_kv_heads,
        head_dim_qk=caps.head_dim_qk,
        head_dim_vo=caps.head_dim_vo,
        page_size=caps.page_size,
        kv_cache_layout=str(invocation.get("kv_cache_layout", "separate")),
        batch_size=caps.max_batch,
        query_len=int(invocation.get(
            "query_len", 1 if caps.mode in ("decode", "verify") else caps.max_total_q
        )),
        cache_tokens=int(invocation.get(
            "cache_tokens", caps.max_page_table_width * caps.page_size
        )),
        window_left=int(invocation.get("window_left", -1)),
        requested_graph_ctas_per_sm=invocation.get("graph_ctas_per_sm"),
        requested_max_work_items=invocation.get(
            "max_work_items", caps.max_work_items or None
        ),
        abi=FrozenMapping(invocation.get("operands", FrozenMapping())),
        controls=FrozenMapping(invocation["controls"]),
        requested_max_partial_rows=invocation.get(
            "max_partial_rows", caps.max_partial_rows or None
        ),
        force_split_kv=invocation.get("force_split_kv"),
    )


def _caps_payload(caps):
    return FrozenMapping({
        item.name: (
            str(value).removeprefix("torch.") if isinstance(value, torch.dtype) else value
        )
        for item in fields(caps)
        if item.name not in ("device", "config")
        for value in (getattr(caps, item.name),)
    })


def _restore_caps(payload, config, ordinal):
    """Restore already-normalized declaration metadata without re-reading env."""
    expected = {item.name for item in fields(B12XPagedAttentionScratchCaps)} - {"device", "config"}
    if set(payload) != expected:
        raise ValueError("paged capacity metadata does not match its schema")
    caps = object.__new__(B12XPagedAttentionScratchCaps)
    for name, value in payload.items():
        object.__setattr__(caps, name, getattr(torch, value) if name in ("dtype", "kv_dtype") else value)
    object.__setattr__(caps, "device", torch.device("cuda", ordinal))
    object.__setattr__(caps, "config", config)
    object.__setattr__(caps, "max_work_items", max(caps.max_work_items, config.max_work_items))
    object.__setattr__(caps, "max_partial_rows", max(caps.max_partial_rows, config.max_partial_rows))
    return caps


@program_cache(scope="preparation")
def compile_paged(caps_payload, config_payload, invocation, ordinal, identity_payload):
    """Describe and retain every native program for one immutable declaration."""
    config = GqaConfig.from_config(FrozenMapping(config_payload))
    device = DetectedDevice(ordinal, DeviceIdentity(**dict(identity_payload)))
    caps = _restore_caps(caps_payload, config, ordinal)
    with paged_controls(invocation["controls"]):
        launchers = {
            key: compile_paged_launchers(binding, controls=invocation["controls"])
            for key, binding in make_compile_bindings(caps, config, invocation, device).items()
        }
    return launchers


def _memory(caps, config, device):
    selected = _restore_caps(_caps_payload(caps), config, device.ordinal)
    # Plan-owned replay schedule and chunk LUT; numerical scratch is shared.
    metadata_ints = (
        4 * selected.max_work_items + selected.max_total_q + 1
        + 2 * (selected.max_batch + 1) + selected.max_page_table_width + 2
    )
    placeholder_nbytes = 2 * selected.dtype.itemsize + 2 * selected.kv_dtype.itemsize
    # Native plane descriptors are 128-byte records with >=64 channels/plane.
    pool_nbytes = selected.num_kv_heads * (
        128 * ((selected.head_dim_qk + 63) // 64 + (selected.head_dim_vo + 63) // 64) + 16
    ) + 4 * selected.num_q_heads + 8 * selected.max_batch
    persistent = (
        placeholder_nbytes + pool_nbytes
        + (4 * metadata_ints if selected.use_cuda_graph else 0)
    )
    resident = 0
    state = current_prepared_state()
    if isinstance(state, _PagedState):
        owner = state.scratch_plan
        tensors = [owner._plan_q, owner._plan_output, owner._plan_k_cache, owner._plan_v_cache,
                   owner._decode_graph_chunk_pages_lut]
        if owner._plan_metadata_cache is not None:
            tensors.extend(
                value for item in fields(owner._plan_metadata_cache)
                if isinstance(value := getattr(owner._plan_metadata_cache, item.name), torch.Tensor)
            )
        for resources in state.resources.values():
            tensors.extend(resources["plane_tma_descs"])
            tensors.extend(resources["optional_buffers"].values())
        resident = _owned_tensor_nbytes(tensors)
        # A state whose KV pool has not attached yet still creates that pool's
        # resources on its first bind, so keep reserving them.
        persistent = resident + (0 if state.resources else pool_nbytes)
    return MemoryRequirements(
        scratch=paged_attention_scratch_specs(selected),
        persistent=(PersistentMemory(
            key=("attention.gqa.prepared", current_plan()),
            required_nbytes=persistent,
            resident_nbytes=resident,
        ),),
    )


def memory_requirements(caps, *, invocation, override=None):
    """Bound all eligible native choices before the serving KV pool is published."""
    invocation = FrozenMapping({
        **dict(invocation), "controls": FrozenMapping(snapshot_paged_controls()),
        "caps": _caps_payload(caps),
    })
    device = detect_device(caps.device)
    configured = TUNING.configure(_query_from_caps(caps, invocation), device=device.identity, override=override)
    configs = (
        (configured.pinned,) if configured.pinned is not None
        else (configured.default, *(config for _, config in TUNING.iterate(configured)))
    )
    return MemoryRequirements.sequential(_memory(caps, config, device) for config in configs)


@dataclass(frozen=True)
class _PagedState:
    """Private materialized layout and retained graph replay metadata."""

    def prepare_decode_graph_replay_state(self, **kwargs):
        with paged_controls(self.controls):
            return self.scratch_plan.prepare_decode_graph_replay_state(**kwargs)

    def prepare_graph_replay_state(self, **kwargs):
        with paged_controls(self.controls):
            return self.scratch_plan.prepare_graph_replay_state(**kwargs)

    scratch_plan: object
    config: GqaConfig

    programs: object
    metadata: object
    controls: FrozenMapping
    resources: dict = field(default_factory=dict)

    def bind(self, *, _preparing=True, **kwargs):
        with paged_controls(self.controls):
            binding = self.scratch_plan.bind(metadata_launchers=self.metadata, **kwargs)
        key = variant_key(binding)
        if key not in self.programs:
            raise ValueError("paged binding requests an undeclared native specialization")
        storage_key = (key, binding.k_cache.data_ptr(), binding.v_cache.data_ptr())
        resources = self.resources.get(storage_key)
        if resources is None:
            # A plan materialized on first use meets its KV pool here; the
            # pool's resources are built once per pool, never under capture.
            if not _preparing and _capturing():
                raise ValueError(
                    "a paged KV pool's resources cannot be created during CUDA "
                    "graph capture"
                )
            resources = materialize_paged_resources(
                k_cache=binding.k_cache, v_cache=binding.v_cache,
                plan=binding.scratch.plan, launchers=self.programs[key],
                k_descale=binding.k_descale, v_descale=binding.v_descale,
                attention_sink_bias=binding.attention_sink_bias,
            )
            self.resources[storage_key] = resources
        install_paged_resources(binding, resources)
        return binding

    def run(self, binding):
        if binding.scratch._owner_scratch_plan is not self.scratch_plan:
            raise ValueError("paged binding belongs to a different prepared plan")
        key = variant_key(binding)
        try:
            launchers = self.programs[key]
        except KeyError:
            raise ValueError("paged binding requests an undeclared native specialization") from None
        return run_paged_prepared(binding, launchers)


def _install_graph_replay_state(scratch_plan, window_left: int) -> None:
    """Install the fixed-address replay schedule a capture binds against.

    Scratch views are rematerialized on every bind, so the schedule belongs to
    the plan; the first bind may already be inside a capture.
    """
    caps = scratch_plan.caps
    if not caps.use_cuda_graph:
        return
    if caps.mode == "decode":
        scratch_plan.prepare_decode_graph_replay_state(
            batch=caps.max_batch, total_q_capacity=caps.max_total_q,
            max_page_table_width=caps.max_page_table_width,
            max_cache_page_count=caps.max_page_table_width,
            window_left=window_left,
        )
        return
    page_table = (
        (torch.arange(caps.max_page_table_width, dtype=torch.int32, device=caps.device)
         % caps.num_cache_pages)
        .unsqueeze(0).expand(caps.max_batch, -1).contiguous()
    )
    geometry = dict(
        device=caps.device, q_dtype=caps.dtype, kv_dtype=caps.kv_dtype,
        num_q_heads=caps.num_q_heads, num_kv_heads=caps.num_kv_heads,
        head_dim_qk=caps.head_dim_qk, head_dim_vo=caps.head_dim_vo,
        page_size=caps.page_size, batch=caps.max_batch,
        max_cache_page_count=caps.max_page_table_width, window_left=window_left,
    )
    if caps.mode == "verify":
        if caps.max_total_q % caps.max_batch:
            raise ValueError(
                "verify graph capacity requires an equal query count per request: "
                f"max_total_q={caps.max_total_q} is not a multiple of "
                f"max_batch={caps.max_batch}"
            )
        query_len = caps.max_total_q // caps.max_batch
        capacity = plan_verify_graph_capacity(**geometry, query_len=query_len)
        cu_seqlens_q = torch.arange(
            0, caps.max_total_q + 1, query_len, dtype=torch.int32, device=caps.device)
    else:
        capacity = plan_extend_graph_capacity(
            **geometry, total_q_capacity=caps.max_total_q)
        cu_seqlens_q = torch.arange(
            caps.max_batch + 1, dtype=torch.int32, device=caps.device)
        cu_seqlens_q[-1] = caps.max_total_q
    cache_tokens = int(capacity.representative_cache_seqlen)
    scratch_plan.prepare_graph_replay_state(
        page_table=page_table,
        cache_seqlens=torch.full(
            (caps.max_batch,), cache_tokens, dtype=torch.int32, device=caps.device),
        cu_seqlens_q=cu_seqlens_q,
        active_total_q=caps.max_total_q,
        window_left=window_left,
    )


def plan(
    caps: B12XPagedAttentionScratchCaps,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: GqaConfig | None = None,
) -> Plan:
    """Declare one prepared paged-attention plan without CUDA work."""
    if not isinstance(caps, B12XPagedAttentionScratchCaps):
        raise TypeError("plan requires B12XPagedAttentionScratchCaps")
    invocation = FrozenMapping({
        **dict(invocation), "controls": FrozenMapping(snapshot_paged_controls()),
        "caps": _caps_payload(caps),
    })
    if "operands" not in invocation:
        raise ValueError("paged declarations require invocation_from_tensors metadata")

    if caps.config is not None:
        raise ValueError("prepared config belongs in plan(..., override=...), not Caps")
    query = _query_from_caps(caps, invocation)
    operands = invocation["operands"]
    if not isinstance(operands, FrozenMapping) or any(name not in operands for name in _OPERANDS):
        raise ValueError("paged invocation operands must come from invocation_from_tensors")
    for name in ("q", "k_cache", "v_cache", "output", "page_table", "cache_seqlens", "cu_seqlens_q"):
        if operands[name] is None:
            raise ValueError(f"paged invocation requires {name} metadata")

    payload = _caps_payload(caps)

    def compile_jobs(config, device):
        return (CompileJob.create(
            "b12x.attention.paged._preparation:compile_paged",
            payload, TUNING.config_payload(config), invocation,
            device.ordinal, asdict(device.identity),
        ),)

    def memory(config, device):
        return _memory(caps, config, device)

    def materialize(selection, device):
        programs = compile_paged(
            payload, TUNING.config_payload(selection.config), invocation,
            device.ordinal, asdict(device.identity),
        )
        metadata = {}
        for launcher in programs.values():
            for name, program in launcher.metadata.items():
                previous = metadata.setdefault(name, program)
                if program_keys(previous) != program_keys(program):
                    raise ValueError("paged variants require incompatible scheduler specializations")
        scratch_plan = plan_paged_attention_scratch(
            _restore_caps(payload, selection.config, device.ordinal)
        )
        with paged_controls(invocation["controls"]):
            _install_graph_replay_state(
                scratch_plan, int(invocation.get("window_left", -1))
            )
        return _PagedState(
            scratch_plan=scratch_plan,
            config=selection.config,
            programs=programs,
            metadata=metadata,
            controls=invocation["controls"],
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=False,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )


def bind(plan, **kwargs):
    """Bind caller tensors to a session-prepared paged plan."""
    state = require_prepared(plan, "attention.gqa")
    return replace(state.bind(_preparing=False, **kwargs), plan=plan)


def run(*, binding, plan=None):
    """Run through the selected state; declarations and raw scratch plans fail."""
    if plan is None:
        plan = binding.plan
    elif binding.plan is not None and binding.plan is not plan:
        raise ValueError("paged binding belongs to a different prepared plan")
    return require_prepared(plan, "attention.gqa").run(binding)


__all__ = ["bind", "invocation_from_descriptors", "invocation_from_tensors", "memory_requirements", "plan", "run"]
