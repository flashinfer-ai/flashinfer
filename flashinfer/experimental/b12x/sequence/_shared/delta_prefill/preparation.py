"""Metadata-only lowering and resolved prepared state for both delta-prefill recipes."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan


_PARAMETER_FIELDS = frozenset(("a_log_dtype", "dt_bias_dtype", "state_indices_dtype"))


def invocation_from_tensors(*, A_log, dt_bias, initial_state_indices, **unused):
    """Extract the parameter/index dtypes that actually specialize these kernels."""
    return FrozenMapping(
        {
            "a_log_dtype": str(A_log.dtype).removeprefix("torch."),
            "dt_bias_dtype": str(dt_bias.dtype).removeprefix("torch."),
            "state_indices_dtype": str(initial_state_indices.dtype).removeprefix(
                "torch."
            ),
        }
    )


def _modules(component):
    if component not in ("gdn_prefill", "kda_prefill"):
        raise ValueError("unknown delta-prefill component")
    root = f"b12x.sequence.{component}"
    return importlib.import_module(root + "._impl"), importlib.import_module(
        root + "._tuning"
    )


def _caps_from_query(impl, query, ordinal, *, is_gdn):
    geometry = (
        {"key_heads": query.key_heads, "value_heads": query.value_heads}
        if is_gdn
        else {"heads": query.heads}
    )
    return impl.Caps(
        device=torch.device("cuda", ordinal),
        max_tokens=query.max_tokens,
        max_seqs=query.max_seqs,
        max_state_slots=query.max_state_slots,
        head_dim=query.head_dim,
        model_dtype=getattr(torch, query.model_dtype),
        state_dtype=getattr(torch, query.state_dtype),
        qk_l2norm=query.qk_l2norm,
        checkpoint_export=query.checkpoint_export,
        null_state_index=query.null_state_index,
        **geometry,
    )


def _metadata_binding(layout, query):
    """Supply only host tensor metadata consumed by the existing compile factories."""
    device = layout.caps.device

    def tensor(dtype):
        return SimpleNamespace(dtype=dtype, device=device)

    integers = (
        "band_base",
        "sorted_seq",
        "rank_of",
        "pos_seq",
        "pos_local",
        "window_table",
        "ready_flags",
        "cu_seqlens",
        "checkpoint_offsets",
        "num_seqs",
        "num_tokens",
    )
    fields = {name: tensor(torch.int32) for name in integers}
    fields.update(
        {
            name: tensor(torch.bfloat16)
            for name in ("q", "k", "v", "raw_g", "raw_beta", "output")
        }
    )
    fields.update(
        {
            name: tensor(getattr(torch, query.state_indices_dtype))
            for name in (
                "initial_state_indices",
                "final_state_indices",
                "checkpoint_state_indices",
            )
        }
    )
    fields.update(
        A_log=tensor(getattr(torch, query.a_log_dtype)),
        dt_bias=tensor(getattr(torch, query.dt_bias_dtype)),
        recurrent_state=tensor(torch.float32),
        scratch=tensor(torch.uint8),
        ws=tensor(torch.uint8),
    )
    binding = SimpleNamespace(
        _state=layout,
        token_capacity=layout.caps.max_tokens,
        seq_capacity=layout.caps.max_seqs,
        parallel=None,
        **fields,
    )
    parallel = getattr(layout, "parallel", None)
    if parallel is not None:
        inner_query = replace(query, state_indices_dtype="int32")
        inner = _metadata_binding(parallel.segment_plan, inner_query)
        binding.parallel = SimpleNamespace(
            _state=parallel,
            transfer=inner,
            local_state=inner,
            output=inner,
            seq_segments=tensor(torch.int32),
            pool=tensor(torch.float32),
            packed_transfer=tensor(torch.bfloat16),
            transfer_flags=tensor(torch.int32),
        )
    return binding


@program_cache(scope="preparation")
def compile_prefill(component, query_payload, config_payload, ordinal):
    impl, tuning = _modules(component)
    query_type = (
        tuning.GdnPrefillQuery if component == "gdn_prefill" else tuning.KdaPrefillQuery
    )
    query = query_type(**dict(query_payload))
    config = tuning.TUNING.decode_config(FrozenMapping(config_payload))
    caps = _caps_from_query(impl, query, ordinal, is_gdn=component == "gdn_prefill")
    layout = impl._materialize_layout(caps, config)
    metadata = _metadata_binding(layout, query)
    with torch.cuda.device(ordinal):
        if metadata.parallel is not None:
            from ...gdn_prefill._parallel import compile_binding
        else:
            from ._cute_kernels import compile_binding
        return compile_binding(metadata)


@dataclass(frozen=True)
class _PrefillState:
    query: object
    layout: object
    programs: tuple
    resources: object
    impl: object
    component_id: str
    parameter_dtypes: tuple
    parallel: bool

    def _check(self, binding):
        if binding._state is not self.layout:
            raise ValueError("prefill binding belongs to another prepared layout")
        actual = (
            binding.A_log.dtype,
            binding.dt_bias.dtype,
            binding.initial_state_indices.dtype,
        )
        if actual != self.parameter_dtypes:
            raise ValueError("prefill parameter/index dtypes differ from preparation")

    def bind(self, **kwargs):
        binding = self.impl._bind(self.layout, **kwargs)
        self._check(binding)
        return binding

    def run(
        self,
        binding,
        *,
        lower_bound=None,
        scale=None,
        eps=1e-6,
        max_live_tokens=None,
        max_live_seqs=None,
    ):
        self._check(binding)
        if self.layout.caps.is_gdn:
            scale, eps = self.impl._check_run_scalars(scale, eps)
            lower_bound = 0.0
        else:
            if lower_bound is None:
                raise TypeError("KDA prefill requires lower_bound")
            lower_bound, scale, eps = self.impl._check_run_scalars(
                lower_bound, scale, eps
            )
        windows = self.layout.launched_windows(max_live_tokens, max_live_seqs)
        if self.parallel:
            from ...gdn_prefill._parallel import run

            run(binding, programs=self.programs, scale=scale, eps=eps)
        else:
            from ._cute_kernels import run_prefill

            run_prefill(
                binding,
                programs=self.programs,
                resources=self.resources,
                lower_bound=lower_bound,
                scale=scale,
                eps=eps,
                windows=windows,
            )
        return binding.output


def make_plan(caps, impl, tuning, *, invocation, override):
    invocation = FrozenMapping(invocation)
    if set(invocation) - _PARAMETER_FIELDS:
        raise ValueError("unknown delta-prefill invocation metadata")
    query = impl._query(caps, invocation)
    component = caps.op_name
    layouts = {}

    def layout(config):
        if config not in layouts:
            layouts[config] = impl._materialize_layout(caps, config)
        return layouts[config]

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.sequence._shared.delta_prefill.preparation:compile_prefill",
                component,
                replace(query, exhaustive=False).to_dict(),
                tuning.TUNING.encode_config(config),
                device.ordinal,
            ),
        )

    def memory(config, device):
        native = MemoryRequirements(scratch=layout(config).scratch_specs())
        staging = impl.staging_memory(caps)
        return MemoryRequirements(
            scratch=native.scratch,
            persistent=staging.persistent,
        )

    def materialize(selection, device):
        native_layout = layout(selection.config)
        programs = compile_prefill(
            component,
            replace(query, exhaustive=False).to_dict(),
            tuning.TUNING.encode_config(selection.config),
            device.ordinal,
        )
        parallel = getattr(native_layout, "parallel", None) is not None
        resources = None
        if not parallel:
            from ._cute_kernels import _side_resources

            with torch.cuda.device(caps.device):
                resources = _side_resources(caps.device, native_layout.max_windows)
        return _PrefillState(
            query,
            native_layout,
            programs,
            resources,
            impl,
            tuning.TUNING.component_id,
            tuple(
                getattr(torch, value)
                for value in (
                    query.a_log_dtype,
                    query.dt_bias_dtype,
                    query.state_indices_dtype,
                )
            ),
            parallel,
        )

    return Plan(
        contract=tuning.TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=False,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )
