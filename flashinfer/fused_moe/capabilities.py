"""Internal capability introspection for the Unified MoE runner registry."""

from __future__ import annotations

from dataclasses import dataclass

from .api import ActivationConfig, QuantVariant


@dataclass(frozen=True)
class MoEBackendCapability:
    """One registered backend's capability for one quantization variant."""

    backend_key: str
    config_type: type
    quant_variant: QuantVariant
    activation_classes: tuple[type[ActivationConfig], ...]


def get_moe_backend_capabilities() -> tuple[MoEBackendCapability, ...]:
    """Return deterministic records derived from the registered runner classes."""
    from .layer import _BACKEND_RUNNERS

    rows = []
    for config_type, runner_type in _BACKEND_RUNNERS.items():
        variants = runner_type.supported_quant_variants
        by_quant = runner_type.supported_activation_classes_by_quant
        if by_quant and set(by_quant) != set(variants):
            raise ValueError(
                f"{runner_type.__name__} activation mapping must cover exactly "
                f"{tuple(variant.name for variant in variants)}."
            )

        for variant in variants:
            activations = (
                by_quant[variant]
                if by_quant
                else runner_type.supported_activation_classes
            )
            if not activations:
                raise ValueError(
                    f"{runner_type.__name__} declares no activations for "
                    f"QuantVariant.{variant.name}."
                )
            rows.append(
                MoEBackendCapability(
                    backend_key=runner_type.backend_key,
                    config_type=config_type,
                    quant_variant=variant,
                    activation_classes=activations,
                )
            )

    return tuple(
        sorted(rows, key=lambda row: (row.backend_key, row.quant_variant.name))
    )


def render_moe_activation_matrix(
    rows: tuple[MoEBackendCapability, ...] | None = None,
) -> str:
    """Render the generated backend × quantization × activation Markdown table."""
    rows = get_moe_backend_capabilities() if rows is None else rows
    lines = [
        "| Backend | Config | Quantization | Typed activations |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        activations = ", ".join(
            f"`{activation.__name__}`" for activation in row.activation_classes
        )
        lines.append(
            f"| `{row.backend_key}` | `{row.config_type.__name__}` | "
            f"`{row.quant_variant.name}` | {activations} |"
        )
    return "\n".join(lines)
