"""Declarative plan capabilities shared by concrete Batch MLA backends."""

from dataclasses import dataclass
from typing import Optional, Protocol

from flashinfer._backend import _BackendPlanUnsupportedError


@dataclass(frozen=True, slots=True)
class MLAPlanCapabilities:
    """Cross-backend plan options accepted by one concrete backend."""

    backend_name: str
    lse_modes: frozenset[str]
    kv_layouts: frozenset[str]
    output_scales: frozenset[str]
    scale_modes: frozenset[str]
    supports_skip_softmax: bool = False
    supports_skip_softmax_with_lse: bool = False
    supports_enable_pdl: bool = False
    supports_is_var_seq: bool = False
    supports_sinks: bool = False
    supports_qk_nope_head_dim: bool = False
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
    def enable_pdl(self) -> Optional[bool]: ...

    @property
    def is_var_seq(self) -> Optional[bool]: ...

    @property
    def use_sinks(self) -> bool: ...

    @property
    def qk_nope_head_dim(self) -> Optional[int]: ...

    @property
    def query_kind(self) -> Optional[str]: ...

    @property
    def kv_kind(self) -> Optional[str]: ...


def validate_plan_capabilities(
    args: _CapabilityPlanArguments,
    capabilities: MLAPlanCapabilities,
) -> None:
    """Reject unsupported cross-backend options before backend construction."""

    backend_name = capabilities.backend_name
    if (
        reason := structural_eligibility_rejection_reason(args, capabilities)
    ) is not None:
        raise _BackendPlanUnsupportedError(reason)
    if args.lse_mode not in capabilities.lse_modes:
        raise _BackendPlanUnsupportedError(
            f"{backend_name} backend does not support this LSE contract."
        )
    if (
        args.skip_softmax
        and args.lse_mode != "none"
        and not capabilities.supports_skip_softmax_with_lse
    ):
        raise _BackendPlanUnsupportedError(
            f"{backend_name} backend does not support the combined LSE and "
            "skip-softmax contract."
        )
    if args.kv_layout not in capabilities.kv_layouts:
        raise _BackendPlanUnsupportedError(
            f"{backend_name} backend does not support this KV layout contract."
        )
    if args.output_scale not in capabilities.output_scales:
        raise _BackendPlanUnsupportedError(
            f"{backend_name} backend does not support this output contract."
        )
    if args.scale_mode not in capabilities.scale_modes:
        raise _BackendPlanUnsupportedError(
            f"{backend_name} backend does not support this scale contract."
        )
    if args.skip_softmax and not capabilities.supports_skip_softmax:
        raise _BackendPlanUnsupportedError(
            f"{backend_name} backend does not support the skip-softmax contract."
        )

    if args.enable_pdl is not None and type(args.enable_pdl) is not bool:
        raise ValueError(f"enable_pdl must be bool or None, got {args.enable_pdl!r}.")
    if args.is_var_seq is not None and type(args.is_var_seq) is not bool:
        raise ValueError(f"is_var_seq must be bool or None, got {args.is_var_seq!r}.")
    if type(args.use_sinks) is not bool:
        raise ValueError(f"use_sinks must be bool, got {args.use_sinks!r}.")
    if args.qk_nope_head_dim is not None and (
        not isinstance(args.qk_nope_head_dim, int)
        or isinstance(args.qk_nope_head_dim, bool)
        or args.qk_nope_head_dim <= 0
    ):
        raise ValueError(
            "qk_nope_head_dim must be a positive int or None, got "
            f"{args.qk_nope_head_dim!r}."
        )

    if args.enable_pdl and not capabilities.supports_enable_pdl:
        raise _BackendPlanUnsupportedError(
            f"enable_pdl is not supported by the {backend_name} backend contract."
        )
    if args.is_var_seq is not None and not capabilities.supports_is_var_seq:
        raise _BackendPlanUnsupportedError(
            f"is_var_seq is not supported by the {backend_name} backend contract."
        )
    if args.use_sinks and not capabilities.supports_sinks:
        raise _BackendPlanUnsupportedError(
            f"use_sinks is not supported by the {backend_name} backend contract."
        )
    if args.qk_nope_head_dim is not None and not capabilities.supports_qk_nope_head_dim:
        raise _BackendPlanUnsupportedError(
            f"qk_nope_head_dim is not supported by the {backend_name} backend contract."
        )


def structural_eligibility_rejection_reason(
    args: _CapabilityPlanArguments,
    capabilities: MLAPlanCapabilities,
) -> str | None:
    """Return why representative inputs cannot meet native layout requirements."""

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
