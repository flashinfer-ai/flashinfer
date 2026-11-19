"""User-facing config for the SM103 MXFP8 x MXFP4 NVLinkOneSided MegaMoE kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

KERNEL_NAME = "sm103_mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl"


@dataclass
class Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig:
    """Blackwell MegaMoE: MXFP8 E4M3 activations x MXFP4 E2M1 weights, BF16 output.

    Runs on SM100 (B200) and SM103 (B300) devices.

    One compiled call launches the EP dispatch, the fused FC1 / SwiGLU / MXFP8
    requantization / FC2 kernel with its combine pushes, and the top-k reduce,
    over PyTorch symmetric memory on the EP process group. Activations and
    weights carry UE8M0 scales per 32 values. The dispatch is a Sentinel
    (data-as-flag) protocol up to 128 tokens per rank and a counter protocol
    (per-peer payload counters) above that; the combine is Sentinel up to 256
    tokens per rank and above that a counter protocol (one all-rank barrier at
    the tail of the fused kernel).

    The top-k reduce needs ``token_hidden_size`` to be a multiple of 1024 and
    ``top_k`` to be at most 32. The engine's tiling and shared-memory limits
    are checked when the layer is constructed.

    ``intermediate_size`` is the width after SwiGLU; FC1 produces
    ``2 * intermediate_size`` columns (gate rows first, then up rows, in the
    canonical ``w13`` layout).

    ``gate_up_clamp`` clamps the gate from above and the up projection on both
    sides before SwiGLU; ``activation_clamp`` is the same clamp under its
    DeepGEMM name. Setting both requires equal values.

    ``knobs`` overrides the engine geometry, schedule and protocols that
    ``nvlink_one_sided_tiling`` derives from the capacity, EP size and routing shape.
    Keys are the frontend's ``TUNING_FIELDS``. The backend has no autotuner.

    ``die_local_weights`` homes the first and second half of every weight
    matrix's M tiles in the HBM of one die each (B300's two locality domains)
    and has each die's clusters claim the tiles of their own half. It changes
    the weight layout, so it is a field of its own rather than a knob, and the
    backend's weight preparation produces that layout. ``None`` takes the
    engine's default: on from 64 expected routes per local expert where the
    device has two locality domains.

    ``reduce_topk_in_kernel`` sums the top-k partials in the fused kernel: under
    the counter combine every target rank's FC2 epilogue adds each route's BF16
    partial straight into the source token's output row on the source rank
    (system-scope BF16 reductions over NVLink), the fused kernel zeroes the
    output at the start of every call, and the tail barrier publishes the sums,
    so the combine slots and the separate top-k reduce go. The adds round to BF16
    one at a time in arrival order instead of one FP32 sum per token, so the
    output changes and is not reproducible from run to run; it is opt-in. It
    applies where the combine is the counter protocol (above 256 tokens per rank
    unless knobs pick it); the Sentinel combine's slots announce their own
    arrival, which a sum cannot, so there the separate reduce stays.

    ``fc1_token_pool_capacity`` bounds the token pool of the pull dispatch (from
    256 expected routes per local expert), which holds one row of
    ``token_hidden_size`` bytes per routed token in processing order: the token
    tiles whose routes end within that many rows load from the pool, and the
    rest fall back to gathering from the landing slots, which then stay
    allocated. ``None`` takes the engine's default, twice the routes a rank
    receives under balanced routing; 0 sizes the pool for any routing (one row
    per possible route, no landing slots).

    ``shared_expert="fp8_in_kernel"`` runs one MXFP8 (E4M3, UE8M0 per 32 values)
    shared expert over every token in the fused kernel, as a third weight
    stream through the routed experts' grouped GEMMs, and adds its output to the
    routed sum: the layer's output is routed + shared. Its weights come with the
    routed ones in a :class:`~flashinfer.moe_ep.PrequantizedMoEWeightsWithSharedExpert`
    (its intermediate width must be ``intermediate_size``). Its FP32 FC2 partial
    is rounded to BF16 like a route's and summed after the routes in the same
    FP32 sum; under ``reduce_topk_in_kernel`` it is added into the output like a
    route. Its FC1 tiles run from the fused kernel's entry, so where they fill
    the counter dispatch's window before the first routed tile, the engine takes
    that dispatch over the pull (``nvlink_one_sided_tiling``; knobs may pin either).
    ``shared_gate_up_clamp`` is its own SwiGLU clamp (``None``: none; the
    routed ``gate_up_clamp`` never applies to it). ``shared_bf16_rounding``
    rounds its gate, up and SwiGLU product to BF16 and the routes' FP32 sum to
    BF16 before its output joins it, the order of a BF16 shared expert added to
    a BF16 routed output; it changes the output and is opt-in, and the in-kernel
    top-k reduce, which adds in arrival order, cannot round that way.

    ``routing_weights_after_fc2`` applies each route's routing weight to its FC2
    output in FP32 instead of to its SwiGLU output before the MXFP8
    requantization: the top-k reduce multiplies each route's BF16 partial by its
    weight as it sums them (the order of a reference that weights the BF16 FC2
    rows in the sum), and under ``reduce_topk_in_kernel`` the FC2 epilogue
    multiplies each route's FP32 partial before it rounds and adds it. It
    changes the output and is opt-in; the shared expert's output is never
    weighted.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = KERNEL_NAME
    gate_up_clamp: Optional[float] = None
    activation_clamp: Optional[float] = None
    knobs: Optional[dict[str, Any]] = None
    die_local_weights: Optional[bool] = None
    reduce_topk_in_kernel: bool = False
    fc1_token_pool_capacity: Optional[int] = None
    shared_expert: Optional[str] = None
    shared_gate_up_clamp: Optional[float] = None
    shared_bf16_rounding: bool = False
    routing_weights_after_fc2: bool = False

    def __post_init__(self) -> None:
        for name in ("intermediate_size", "top_k"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        for name in ("gate_up_clamp", "activation_clamp"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative, got {value}")
        if (
            self.gate_up_clamp is not None
            and self.activation_clamp is not None
            and self.gate_up_clamp != self.activation_clamp
        ):
            raise ValueError(
                "gate_up_clamp and activation_clamp name the same clamp; "
                f"got {self.gate_up_clamp} and {self.activation_clamp}"
            )
        if self.die_local_weights is not None and not isinstance(
            self.die_local_weights, bool
        ):
            raise ValueError(
                f"die_local_weights must be None or a bool, got {self.die_local_weights!r}"
            )
        for name in ("reduce_topk_in_kernel", "routing_weights_after_fc2"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.fc1_token_pool_capacity is not None and (
            type(self.fc1_token_pool_capacity) is not int
            or self.fc1_token_pool_capacity < 0
        ):
            raise ValueError(
                "fc1_token_pool_capacity must be None or a nonnegative int, "
                f"got {self.fc1_token_pool_capacity!r}"
            )
        if self.shared_expert not in (None, "fp8_in_kernel"):
            raise ValueError(
                f"shared_expert must be None or 'fp8_in_kernel', got {self.shared_expert!r}"
            )
        value = self.shared_gate_up_clamp
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"shared_gate_up_clamp must be finite and nonnegative, got {value!r}"
            )
        if not isinstance(self.shared_bf16_rounding, bool):
            raise ValueError(
                f"shared_bf16_rounding must be a bool, got {self.shared_bf16_rounding!r}"
            )
        if self.shared_expert is None and (
            self.shared_gate_up_clamp is not None or self.shared_bf16_rounding
        ):
            raise ValueError(
                "shared_gate_up_clamp and shared_bf16_rounding need shared_expert='fp8_in_kernel'"
            )
        if self.shared_bf16_rounding and self.reduce_topk_in_kernel:
            raise ValueError(
                "shared_bf16_rounding rounds the routes' sum before the shared expert's output "
                "joins it, which reduce_topk_in_kernel (BF16 adds in arrival order) cannot do"
            )
        if self.knobs is not None and not isinstance(self.knobs, dict):
            raise ValueError(
                f"{KERNEL_NAME} knobs must be None or a dictionary of engine "
                f"overrides (there is no autotuner), got {self.knobs!r}"
            )

    @property
    def resolved_gate_up_clamp(self) -> Optional[float]:
        if self.gate_up_clamp is not None:
            return self.gate_up_clamp
        return self.activation_clamp
