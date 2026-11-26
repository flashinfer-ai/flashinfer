"""Weight preparation for the SM103 MXFP8 x MXFP4 NVLinkOneSided MegaMoE.

Canonical inputs (one rank's local experts):

- BF16/FP16/FP32 ``w13`` ``[E, 2I, H]`` (gate rows, then up rows) and ``w2``
  ``[E, H, I]``, quantized here to MXFP4 with UE8M0 scales per 32 values,
  rounded up to powers of two.
- Packed MXFP4: ``w13`` ``[E, 2I, H/2]`` and ``w2`` ``[E, H, I/2]`` E2M1
  nibbles (low nibble = even column; ``uint8``, ``int8`` or
  ``float4_e2m1fn_x2`` storage) with E8M0 scales ``w13_scale`` ``[E, 2I, H/32]``
  and ``w2_scale`` ``[E, H, I/32]`` (``float8_e8m0fnu`` or raw ``uint8``
  bytes). The payload and scales are kept bit for bit.

Kernel layout: FC1 rows interleaved in 16-row gate/up stripes, scales in the
32x4x4 atom swizzle flattened per expert, and the packed weights stored in
the engine config's ``weight_layout`` (block-major by default), tagged so the
launch path rejects weights prepared for another layout. Under die-local
weights each leg is stored half-major, ``[2E, K, rows / 2]`` with half ``h``
of expert ``e`` at entry ``h * E + e``, every half in the HBM of its die.

With a shared expert (``shared_expert="fp8_in_kernel"``) the weights are a
:class:`PrequantizedMoEWeightsWithSharedExpert` whose shared expert has the
routed intermediate width: E4M3 ``shared_w13`` ``[2I, H]`` and ``shared_w2``
``[H, I]`` with UE8M0 scales per 32 values (``float8_e8m0fnu`` or raw ``uint8``
bytes; E4M3 data may be raw ``uint8`` too), kept bit for bit. They become two
more legs, the shared expert's FC1 and FC2, as one K-major expert entry with
the same 16-row gate/up interleave and scale swizzle as the routed legs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from ......core.validation.common import MoEEpConfigError
from ......weights import (
    MoEWeightPack,
    PrequantizedMoEWeights,
    PrequantizedMoEWeightsWithSharedExpert,
)

if TYPE_CHECKING:
    import torch

# The routed experts' (FC1, FC2) legs, each (weight, scale), followed with a
# shared expert by its (FC1, FC2) legs.
TransformedMegaWeights = Tuple[Tuple["torch.Tensor", "torch.Tensor"], ...]

__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]


def _as_fp4_bytes(tensor: "torch.Tensor", name: str) -> "torch.Tensor":
    import torch

    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if tensor.dtype not in (torch.uint8, torch.int8) and tensor.dtype != fp4:
        raise MoEEpConfigError(
            f"packed MXFP4 {name} must be uint8, int8 or float4_e2m1fn_x2, "
            f"got {tensor.dtype}"
        )
    return tensor.view(torch.uint8)


def _as_e4m3(tensor: "torch.Tensor", name: str) -> "torch.Tensor":
    import torch

    if tensor.dtype == torch.float8_e4m3fn:
        return tensor
    if tensor.dtype == torch.uint8:
        return tensor.view(torch.float8_e4m3fn)
    raise MoEEpConfigError(
        f"{name} must hold E4M3 values (float8_e4m3fn or uint8), got {tensor.dtype}"
    )


def _as_e8m0(tensor: "torch.Tensor", name: str) -> "torch.Tensor":
    import torch

    if tensor.dtype == torch.float8_e8m0fnu:
        return tensor
    if tensor.dtype == torch.uint8:
        return tensor.view(torch.float8_e8m0fnu)
    raise MoEEpConfigError(
        f"{name} must hold E8M0 scales (float8_e8m0fnu or uint8), got {tensor.dtype}"
    )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    weight_layout: Optional[str] = None,
    die_local_weights: bool = False,
    shared_expert: Optional[str] = None,
) -> TransformedMegaWeights:
    """Canonical floating-point or packed MXFP4 weights -> kernel layout.

    ``weight_layout`` must be the engine config's ``resolved_weight_layout``;
    ``None`` selects the frontend default for MXFP4 weights (block-major).
    ``die_local_weights`` (the engine config's ``resolved_die_local_weights``)
    stores both routed legs die-local, which needs a device with two locality
    domains and block-major weights. ``shared_expert`` (the config's) appends
    the shared expert's two legs.
    """
    if (shared_expert is not None) != isinstance(
        weights, PrequantizedMoEWeightsWithSharedExpert
    ):
        raise MoEEpConfigError(
            "shared_expert='fp8_in_kernel' requires PrequantizedMoEWeightsWithSharedExpert "
            "weights, and a layer without a shared expert does not take them"
        )
    transformed = _preprocess_mega_weights(
        weights,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        weight_layout=weight_layout,
    )
    shared: TransformedMegaWeights = ()
    if isinstance(weights, PrequantizedMoEWeightsWithSharedExpert):
        shared = _preprocess_shared_expert_weights(
            weights, intermediate_size=intermediate_size, hidden_size=hidden_size
        )
    if not die_local_weights:
        return transformed + shared
    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.die_local import (
        die_local_weight_leg,
    )

    try:
        return tuple(die_local_weight_leg(leg) for leg in transformed) + shared
    except (ValueError, RuntimeError) as exc:
        raise MoEEpConfigError(f"die-local weights: {exc}") from exc


def _preprocess_shared_expert_weights(
    weights: PrequantizedMoEWeightsWithSharedExpert,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """The shared expert's E4M3 weights and UE8M0 scales -> its (FC1, FC2) kernel legs, K-major."""
    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        preprocess_genphase_prequantized_weights,
    )

    tensors = {
        name: getattr(weights, name)
        for name in (
            "shared_w13",
            "shared_w2",
            "shared_w13_scale",
            "shared_w2_scale",
        )
    }
    expected = {
        "shared_w13": (2 * intermediate_size, hidden_size),
        "shared_w2": (hidden_size, intermediate_size),
        "shared_w13_scale": (2 * intermediate_size, hidden_size // 32),
        "shared_w2_scale": (hidden_size, intermediate_size // 32),
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name]:
            raise MoEEpConfigError(
                f"{name} must have shape {expected[name]} (the shared expert's "
                f"intermediate width is intermediate_size), got {tuple(tensor.shape)}"
            )
    try:
        return preprocess_genphase_prequantized_weights(
            _as_e4m3(tensors["shared_w13"], "shared_w13").unsqueeze(0),
            _as_e4m3(tensors["shared_w2"], "shared_w2").unsqueeze(0),
            _as_e8m0(tensors["shared_w13_scale"], "shared_w13_scale").unsqueeze(0),
            _as_e8m0(tensors["shared_w2_scale"], "shared_w2_scale").unsqueeze(0),
            quant_kind="mxfp8_e4m3",
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout="kmajor",
        )
    except ValueError as exc:
        raise MoEEpConfigError(str(exc)) from exc


def _preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    weight_layout: Optional[str],
) -> TransformedMegaWeights:
    import torch

    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        preprocess_genphase_prequantized_weights,
        preprocess_genphase_weights,
    )

    try:
        if isinstance(weights, PrequantizedMoEWeights):
            return preprocess_genphase_prequantized_weights(
                _as_fp4_bytes(weights.w13, "w13"),
                _as_fp4_bytes(weights.w2, "w2"),
                _as_e8m0(weights.w13_scale, "w13_scale"),
                _as_e8m0(weights.w2_scale, "w2_scale"),
                quant_kind="mxfp4_mxfp8",
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                weight_layout=weight_layout,
            )
        w13, w2 = weights.w13, weights.w2
        for name, tensor in (("w13", w13), ("w2", w2)):
            if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise MoEEpConfigError(
                    f"unquantized {name} must be bf16, fp16 or fp32, got "
                    f"{tensor.dtype}; pass both scale planes for packed MXFP4"
                )
        if w13.ndim != 3 or tuple(w13.shape[1:]) != (
            2 * intermediate_size,
            hidden_size,
        ):
            raise MoEEpConfigError(
                f"w13 must have shape [E, {2 * intermediate_size}, {hidden_size}], "
                f"got {tuple(w13.shape)}"
            )
        return preprocess_genphase_weights(
            w13,
            w2,
            quant_kind="mxfp4_mxfp8",
            intermediate_size=intermediate_size,
            weight_layout=weight_layout,
        )
    except MoEEpConfigError:
        raise
    except ValueError as exc:
        raise MoEEpConfigError(str(exc)) from exc


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights, engine_config
) -> None:
    """Check caller-supplied kernel-layout weights against an engine config."""
    import torch

    from ......cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.genphase import (
        validate_genphase_weights,
    )

    try:
        validate_genphase_weights(
            transformed,
            engine_config,
            torch.device("cuda", torch.cuda.current_device()),
        )
    except ValueError as exc:
        raise MoEEpConfigError(str(exc)) from exc
