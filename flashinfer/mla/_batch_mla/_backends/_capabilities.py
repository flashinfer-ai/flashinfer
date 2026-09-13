"""Declarative plan capabilities shared by concrete Batch MLA backends."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MLAPlanCapabilities:
    backend_name: str
    lse_modes: frozenset[str]
    kv_layouts: frozenset[str]
    output_scales: frozenset[str]
    scale_modes: frozenset[str]
    supports_skip_softmax: bool = False
    supports_skip_softmax_with_lse: bool = False
    supports_enable_pdl: bool = False
    supports_sinks: bool = False
    supports_cuda_graph_replan: bool = False
    requires_packed_query: bool = False
    requires_packed_kv_cache: bool = False


class _BackendPlanUnsupportedError(RuntimeError):
    """Typed signal for backend preflight rejection before launch/compile."""


class _CapabilityPlanArguments(Protocol):
    @property
    def query_layout(self) -> str: ...

    @property
    def kv_cache_layout(self) -> str: ...

    @property
    def lse_mode(self) -> str: ...

    @property
    def kv_layout(self) -> str: ...

    @property
    def output_scale(self) -> str: ...

    @property
    def scale_mode(self) -> str: ...

    @property
    def skip_softmax(self) -> bool: ...

    @property
    def enable_pdl(self) -> bool | None: ...

    @property
    def use_sinks(self) -> bool: ...

    @property
    def query_kind(self) -> str | None: ...

    @property
    def kv_kind(self) -> str | None: ...


def structural_eligibility_rejection_reason(
    args: _CapabilityPlanArguments,
    capabilities: MLAPlanCapabilities,
) -> str | None:
    query_kind = args.query_kind
    kv_kind = args.kv_kind
    query_layout = args.query_layout
    kv_cache_layout = args.kv_cache_layout

    if capabilities.requires_packed_query and (
        query_kind == "independent-split" or query_layout == "split"
    ):
        return (
            f"{capabilities.backend_name} backend requires a packed query view; "
            "representative query is independent split-only."
        )
    if capabilities.requires_packed_kv_cache and (
        kv_kind == "independent-split" or kv_cache_layout == "split"
    ):
        return (
            f"{capabilities.backend_name} backend requires a packed KV-cache view; "
            "representative kv_cache is independent split-only."
        )
    return None


def plan_capability_rejection_reason(
    args: _CapabilityPlanArguments,
    capabilities: MLAPlanCapabilities,
) -> str | None:
    if reason := structural_eligibility_rejection_reason(args, capabilities):
        return reason

    lse_mode = args.lse_mode
    kv_layout = args.kv_layout
    output_scale = args.output_scale
    scale_mode = args.scale_mode
    skip_softmax = args.skip_softmax
    enable_pdl = args.enable_pdl
    use_sinks = args.use_sinks

    backend_name = capabilities.backend_name
    if lse_mode not in capabilities.lse_modes:
        return f"{backend_name} backend does not support this LSE contract."
    if (
        skip_softmax
        and lse_mode != "none"
        and not capabilities.supports_skip_softmax_with_lse
    ):
        return (
            f"{backend_name} backend does not support the combined LSE and "
            "skip-softmax contract."
        )
    if kv_layout not in capabilities.kv_layouts:
        return f"{backend_name} backend does not support this KV layout contract."
    if output_scale not in capabilities.output_scales:
        return f"{backend_name} backend does not support this output contract."
    if scale_mode not in capabilities.scale_modes:
        return f"{backend_name} backend does not support this scale contract."
    if skip_softmax and not capabilities.supports_skip_softmax:
        return f"{backend_name} backend does not support the skip-softmax contract."
    if enable_pdl is True and not capabilities.supports_enable_pdl:
        return f"{backend_name} backend does not support enable_pdl=True."
    if use_sinks and not capabilities.supports_sinks:
        return f"{backend_name} backend does not support sink inputs."
    return None
