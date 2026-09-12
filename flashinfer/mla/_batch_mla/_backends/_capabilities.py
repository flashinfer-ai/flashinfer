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
    requires_packed_query: bool = False
    requires_packed_kv_cache: bool = False


class _CapabilityPlanArguments(Protocol):
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
    def query_kind(self) -> str | None: ...

    @property
    def kv_kind(self) -> str | None: ...


def structural_eligibility_rejection_reason(
    args: _CapabilityPlanArguments,
    capabilities: MLAPlanCapabilities,
) -> str | None:
    if capabilities.requires_packed_query and args.query_kind == "independent-split":
        return (
            f"{capabilities.backend_name} backend requires a packed query view; "
            "representative query is independent split-only."
        )
    if capabilities.requires_packed_kv_cache and args.kv_kind == "independent-split":
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
    backend_name = capabilities.backend_name
    if args.lse_mode not in capabilities.lse_modes:
        return f"{backend_name} backend does not support this LSE contract."
    if (
        args.skip_softmax
        and args.lse_mode != "none"
        and not capabilities.supports_skip_softmax_with_lse
    ):
        return (
            f"{backend_name} backend does not support the combined LSE and "
            "skip-softmax contract."
        )
    if args.kv_layout not in capabilities.kv_layouts:
        return f"{backend_name} backend does not support this KV layout contract."
    if args.output_scale not in capabilities.output_scales:
        return f"{backend_name} backend does not support this output contract."
    if args.scale_mode not in capabilities.scale_modes:
        return f"{backend_name} backend does not support this scale contract."
    if args.skip_softmax and not capabilities.supports_skip_softmax:
        return f"{backend_name} backend does not support the skip-softmax contract."
    return None
