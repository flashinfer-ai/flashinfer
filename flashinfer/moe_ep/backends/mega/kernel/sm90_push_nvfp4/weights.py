"""Checkpoint conversion and weight validation for SM90 push NVFP4."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias, cast

from .....core.validation.common import MoEEpConfigError
from .....weights import MoEWeightPack

if TYPE_CHECKING:
    import torch

    from .....kernel_src.sm90.push_style_megamoe import (
        Sm90PushNvFp4DualWeights,
        Sm90PushNvFp4HotFoldedWeights,
        Sm90PushNvFp4Weights,
    )

    TransformedMegaWeights: TypeAlias = (
        Sm90PushNvFp4Weights | Sm90PushNvFp4HotFoldedWeights | Sm90PushNvFp4DualWeights
    )


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .....kernel_src.sm90.push_style_megamoe import (
            Sm90PushNvFp4DualWeights,
            Sm90PushNvFp4HotFoldedWeights,
            Sm90PushNvFp4Weights,
        )

        return (
            Sm90PushNvFp4Weights
            | Sm90PushNvFp4HotFoldedWeights
            | Sm90PushNvFp4DualWeights
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _e2m1_codes(values: "torch.Tensor") -> "torch.Tensor":
    import torch

    magnitude = values.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold, inclusive in (
        (0.25, False),
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        crossed = magnitude >= threshold if inclusive else magnitude > threshold
        code.add_(crossed.to(torch.uint8))
    return code | ((values < 0).to(torch.uint8) << 3)


_NVFP4_QUANT_CHUNK_VALUES = 1 << 20


def quantize_bf16_to_nvfp4_checkpoint(
    weights: "torch.Tensor",
    *,
    source_format_version: str = "flashinfer.bf16_to_nvfp4.v1",
):
    """Quantize canonical BF16 expert weights into the linear NVFP4 contract."""

    import torch
    from ......fused_moe.nvfp4_checkpoint import NVFP4Checkpoint

    if weights.dtype != torch.bfloat16 or weights.ndim != 3:
        raise MoEEpConfigError("weights must be contiguous BF16 [E,N,K]")
    if not weights.is_cuda or not weights.is_contiguous():
        raise MoEEpConfigError("weights must be contiguous CUDA tensors")
    weights = weights.detach()
    experts, rows, logical_k = weights.shape
    if experts <= 0 or rows <= 0 or logical_k <= 0:
        raise MoEEpConfigError("weights dimensions must be positive")
    padded_k = math.ceil(logical_k / 16) * 16
    chunk_rows = max(1, min(rows, _NVFP4_QUANT_CHUNK_VALUES // padded_k))
    max_e4m3 = 448.0
    payload = torch.zeros(
        (experts, rows, padded_k // 2), dtype=torch.uint8, device=weights.device
    )
    scales = torch.zeros(
        (experts, rows, padded_k // 16),
        dtype=torch.float8_e4m3fn,
        device=weights.device,
    )
    alpha = torch.empty(experts, dtype=torch.float32, device=weights.device)
    for expert in range(experts):
        global_amax = torch.zeros((), dtype=torch.float32, device=weights.device)
        for begin in range(0, rows, chunk_rows):
            chunk = weights[expert, begin : begin + chunk_rows]
            global_amax = torch.maximum(global_amax, chunk.abs().amax().float())
        if not bool(torch.isfinite(global_amax)):
            raise MoEEpConfigError(f"weights for expert {expert} must be finite")
        expert_alpha = torch.where(
            global_amax > 0,
            global_amax / (max_e4m3 * 6.0),
            torch.ones_like(global_amax),
        )
        alpha[expert] = expert_alpha
        for begin in range(0, rows, chunk_rows):
            end = min(begin + chunk_rows, rows)
            values = weights[expert, begin:end].float()
            if padded_k != logical_k:
                padded = torch.zeros(
                    (end - begin, padded_k),
                    dtype=torch.float32,
                    device=weights.device,
                )
                padded[:, :logical_k] = values
                values = padded
            blocks = values.view(end - begin, padded_k // 16, 16)
            raw_scale = (blocks.abs().amax(dim=-1) / (6.0 * expert_alpha)).clamp(
                0, max_e4m3
            )
            chunk_scales = raw_scale.to(torch.float8_e4m3fn)
            scales[expert, begin:end] = chunk_scales
            effective = chunk_scales.float() * expert_alpha
            normalized = torch.where(
                effective.unsqueeze(-1) > 0,
                blocks / effective.unsqueeze(-1),
                torch.zeros_like(blocks),
            ).clamp(-6.0, 6.0)
            codes = _e2m1_codes(normalized).view(end - begin, padded_k)
            payload[expert, begin:end] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return NVFP4Checkpoint(
        payload,
        scales.contiguous(),
        alpha.contiguous(),
        (experts, rows, logical_k),
        tuple(range(experts)),
        source_format_version,
    )


def make_transformed_weights_from_checkpoints(
    w13_checkpoint,
    w2_checkpoint,
    *,
    nvfp4_mode: Literal["w4a8", "w4a16_rs"],
    group_size: int,
    residual_scheme: str,
    payload_layout: Literal[3, 4] = 4,
):
    from .....kernel_src.sm90.push_style_megamoe import (
        make_sm90_push_nvfp4_weights_from_checkpoints,
    )

    return make_sm90_push_nvfp4_weights_from_checkpoints(
        w13_checkpoint,
        w2_checkpoint,
        nvfp4_mode=nvfp4_mode,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )


def make_folded_fp8_weights_from_checkpoints(
    w13_checkpoint,
    w2_checkpoint,
    *,
    interleave_gate_up: bool = False,
):
    """Fold NVFP4 checkpoints once for the SM90 FP8 runner.

    Returns resident ``Sm90PushWeights`` with E4M3 payloads and FP32 scales per
    128x128 block. The fold belongs in loading or initialization; forward calls
    through ``Sm90PushFp8MegaMoeConfig`` do not fold.

    ``interleave_gate_up`` must match the config's ``fuse_fc1_epilogue`` value.
    """

    from .....kernel_src.sm90.push_style_megamoe import (
        make_sm90_push_folded_fp8_weights_from_checkpoints,
    )

    return make_sm90_push_folded_fp8_weights_from_checkpoints(
        w13_checkpoint,
        w2_checkpoint,
        interleave_gate_up=interleave_gate_up,
    )


def make_hot_folded_weights_from_checkpoints(
    w13_checkpoint,
    w2_checkpoint,
    *,
    hot_experts: int,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
):
    """Build a static hot-prefix FP8 and cold-suffix online-W4A8 bundle.

    Folding happens once in this constructor. The folded prefix and packed
    suffix remain resident, and forward calls select them by the frozen local-
    expert prefix without host-side routing decisions.

    Use ``MegaConfig(megakernel=Sm90PushNvFp4MegaMoeConfig(
    nvfp4_mode="w4a8"), preprocess_weights=False,
    transformed_weights=bundle)`` and construct the layer with ``weights=None``.
    """

    from .....kernel_src.sm90.push_style_megamoe import (
        make_sm90_push_nvfp4_hot_folded_weights_from_checkpoints,
    )

    return make_sm90_push_nvfp4_hot_folded_weights_from_checkpoints(
        w13_checkpoint,
        w2_checkpoint,
        hot_experts=hot_experts,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )


def make_dual_weights_from_checkpoints(
    w13_checkpoint,
    w2_checkpoint,
    *,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
):
    """Keep full packed W4A8 and folded FP8 weights resident together.

    Use this bundle with ``weight_policy="dual"`` and
    ``acknowledge_dual_residency=True``. The explicit acknowledgement prevents
    selecting the doubled representation accidentally.
    """

    from .....kernel_src.sm90.push_style_megamoe import (
        make_sm90_push_nvfp4_dual_weights_from_checkpoints,
    )

    return make_sm90_push_nvfp4_dual_weights_from_checkpoints(
        w13_checkpoint,
        w2_checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )


def estimate_residency(
    local_experts: int,
    policy: Literal["packed", "folded", "hot_folded", "dual"],
    *,
    hidden_size: int,
    intermediate_size: int,
    group_size: int = 128,
    residual_scheme: str = "generic",
    hot_expert_count: int = 0,
):
    """Return geometry-derived packed and folded resident byte counts."""

    from .....kernel_src.sm90.push_style_megamoe import estimate_nvfp4_residency

    return estimate_nvfp4_residency(
        local_experts,
        policy,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        group_size=group_size,
        residual_scheme=residual_scheme,
        hot_expert_count=hot_expert_count,
    )


def load_modelopt_transformed_weights(
    state_dict: Mapping[str, "torch.Tensor"],
    *,
    w13_prefix: str,
    w2_prefix: str,
    nvfp4_mode: Literal["w4a8", "w4a16_rs"] = "w4a8",
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
    device: "torch.device | str | None" = None,
):
    from .....kernel_src.sm90.push_style_megamoe import (
        load_sm90_push_nvfp4_modelopt_weights,
    )

    return load_sm90_push_nvfp4_modelopt_weights(
        state_dict,
        w13_prefix=w13_prefix,
        w2_prefix=w2_prefix,
        nvfp4_mode=nvfp4_mode,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
        device=device,
    )


def load_modelopt_folded_fp8_weights(
    state_dict: Mapping[str, "torch.Tensor"],
    *,
    w13_prefix: str,
    w2_prefix: str,
    interleave_gate_up: bool = False,
    device: "torch.device | str | None" = None,
):
    """Load and fold ModelOpt NVFP4 tensors once for the SM90 FP8 runner.

    Returns resident ``Sm90PushWeights`` with E4M3 payloads and FP32 scales per
    128x128 block. Forward calls through ``Sm90PushFp8MegaMoeConfig`` do not
    fold.

    ``interleave_gate_up`` must match the config's ``fuse_fc1_epilogue`` value.
    CPU tensors are moved and folded one layer at a time when ``device`` is set.
    CUDA source tensors remain owned by the caller.
    """

    from .....kernel_src.sm90.push_style_megamoe import (
        load_sm90_push_nvfp4_modelopt_folded_fp8_weights,
    )

    return load_sm90_push_nvfp4_modelopt_folded_fp8_weights(
        state_dict,
        w13_prefix=w13_prefix,
        w2_prefix=w2_prefix,
        interleave_gate_up=interleave_gate_up,
        device=device,
    )


def load_modelopt_hot_folded_weights(
    state_dict: Mapping[str, "torch.Tensor"],
    *,
    w13_prefix: str,
    w2_prefix: str,
    hot_experts: int,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
    device: "torch.device | str | None" = None,
):
    """Load a static hot-prefix FP8 and cold-suffix online-W4A8 bundle.

    Source layers are moved and transformed one at a time when ``device`` is
    provided. The returned folded prefix and packed suffix remain resident;
    forward calls perform no folding or host-side expert selection.

    Use ``MegaConfig(megakernel=Sm90PushNvFp4MegaMoeConfig(
    nvfp4_mode="w4a8"), preprocess_weights=False,
    transformed_weights=bundle)`` and construct the layer with ``weights=None``.
    """

    from .....kernel_src.sm90.push_style_megamoe import (
        load_sm90_push_nvfp4_modelopt_hot_folded_weights,
    )

    return load_sm90_push_nvfp4_modelopt_hot_folded_weights(
        state_dict,
        w13_prefix=w13_prefix,
        w2_prefix=w2_prefix,
        hot_experts=hot_experts,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
        device=device,
    )


def load_modelopt_dual_weights(
    state_dict: Mapping[str, "torch.Tensor"],
    *,
    w13_prefix: str,
    w2_prefix: str,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
    device: "torch.device | str | None" = None,
):
    """Load ModelOpt weights into explicit dual-residency W4A8/FP8 storage.

    The matching backend config requires ``weight_policy="dual"`` and
    ``acknowledge_dual_residency=True``.
    """

    from .....kernel_src.sm90.push_style_megamoe import (
        load_sm90_push_nvfp4_modelopt_dual_weights,
    )

    return load_sm90_push_nvfp4_modelopt_dual_weights(
        state_dict,
        w13_prefix=w13_prefix,
        w2_prefix=w2_prefix,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
        device=device,
    )


def validate_transformed_mega_weights(
    transformed_weights: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    nvfp4_mode: str,
    group_size: int,
    residual_scheme: str,
    payload_layout: int = 4,
    weight_policy: str = "packed",
    hot_expert_count: int = 0,
) -> None:
    from ......fused_moe.sm90_nvfp4_repack import (
        NVFP4RSWeightView,
        NVFP4SM90WeightViewV3,
        NVFP4SM90WeightViewV4,
    )
    from .....kernel_src.sm90.push_style_megamoe import (
        Sm90PushNvFp4DualWeights,
        Sm90PushNvFp4HotFoldedWeights,
        Sm90PushNvFp4Weights,
    )

    w4a8_view_types = (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)

    if isinstance(transformed_weights, Sm90PushNvFp4DualWeights):
        if weight_policy != "dual":
            raise MoEEpConfigError("dual weights require weight_policy='dual'")
        if transformed_weights.total_experts != num_local_experts:
            raise MoEEpConfigError("dual weights do not match the local expert count")
        validate_transformed_mega_weights(
            transformed_weights.packed_nvfp4,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            num_local_experts=num_local_experts,
            nvfp4_mode=nvfp4_mode,
            group_size=group_size,
            residual_scheme=residual_scheme,
            payload_layout=payload_layout,
            weight_policy="packed",
        )
        from ..sm90_push_fp8.weights import (
            validate_transformed_mega_weights as validate_fp8_weights,
        )

        validate_fp8_weights(
            transformed_weights.folded_fp8,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            num_local_experts=num_local_experts,
            fuse_fc1_epilogue=False,
        )
        return

    if isinstance(transformed_weights, Sm90PushNvFp4HotFoldedWeights):
        if weight_policy not in ("folded", "hot_folded"):
            raise MoEEpConfigError(
                "hot-prefix weights require folded or hot_folded weight_policy"
            )
        if nvfp4_mode != "w4a8":
            raise MoEEpConfigError("hot-folded weights require nvfp4_mode='w4a8'")
        if transformed_weights.total_experts != num_local_experts:
            raise MoEEpConfigError(
                "hot-folded total_experts does not match the local expert count"
            )
        expected_hot = (
            num_local_experts if weight_policy == "folded" else hot_expert_count
        )
        if transformed_weights.hot_experts != expected_hot:
            raise MoEEpConfigError(
                "hot-folded weight prefix does not match the configured policy"
            )
        try:
            transformed_weights.__post_init__()
        except (TypeError, ValueError) as exc:
            raise MoEEpConfigError(f"invalid hot-folded weight bundle: {exc}") from exc
        if transformed_weights.hot_fp8 is not None:
            hot = transformed_weights.hot_fp8
            expected_hot = transformed_weights.hot_experts
            from ..sm90_push_fp8.weights import (
                validate_transformed_mega_weights as validate_fp8_weights,
            )

            validate_fp8_weights(
                hot,
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                num_local_experts=expected_hot,
                fuse_fc1_epilogue=False,
            )
        if transformed_weights.cold_nvfp4 is not None:
            cold = transformed_weights.cold_nvfp4
            expected_cold = num_local_experts - transformed_weights.hot_experts
            for label, view, shape in zip(
                ("cold w13", "cold w2"),
                (cold.w13, cold.w2),
                (
                    (expected_cold, 2 * intermediate_size, hidden_size),
                    (expected_cold, hidden_size, intermediate_size),
                ),
                strict=True,
            ):
                if not isinstance(view, w4a8_view_types):
                    raise MoEEpConfigError(
                        f"sm90_push_nvfp4 {label} is not a W4A8 view"
                    )
                if view.manifest.layout_version != payload_layout:
                    raise MoEEpConfigError(
                        f"sm90_push_nvfp4 {label} payload layout does not match config"
                    )
                if tuple(view.manifest.logical_shape) != shape:
                    raise MoEEpConfigError(
                        f"sm90_push_nvfp4 {label} logical shape must be {shape}"
                    )
                if view.manifest.group_size != group_size:
                    raise MoEEpConfigError(
                        f"sm90_push_nvfp4 {label} group_size does not match config"
                    )
                if view.manifest.residual_scheme != residual_scheme:
                    raise MoEEpConfigError(
                        f"sm90_push_nvfp4 {label} residual_scheme does not match config"
                    )
                if not view.packed_e2m1.is_cuda:
                    raise MoEEpConfigError(
                        f"sm90_push_nvfp4 {label} must be on a CUDA device"
                    )
            cold_w13 = cast("NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4", cold.w13)
            cold_w2 = cast("NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4", cold.w2)
            if cold_w13.packed_e2m1.device != cold_w2.packed_e2m1.device:
                raise MoEEpConfigError(
                    "sm90_push_nvfp4 cold weights must share a CUDA device"
                )
            if (
                transformed_weights.hot_fp8 is not None
                and cold_w13.packed_e2m1.device
                != transformed_weights.hot_fp8.w13_fp8.device
            ):
                raise MoEEpConfigError(
                    "sm90_push_nvfp4 hot and cold weights must share a CUDA device"
                )
        return

    if not isinstance(transformed_weights, Sm90PushNvFp4Weights):
        raise MoEEpConfigError(
            "sm90_push_nvfp4 transformed weights must be "
            f"Sm90PushNvFp4Weights, got {type(transformed_weights).__name__}"
        )
    if weight_policy != "packed":
        raise MoEEpConfigError("packed NVFP4 weights require weight_policy='packed'")
    if transformed_weights.nvfp4_mode != nvfp4_mode:
        raise MoEEpConfigError(
            "sm90_push_nvfp4 weight mode does not match config: "
            f"{transformed_weights.nvfp4_mode!r} != {nvfp4_mode!r}"
        )
    expected_shapes = (
        (num_local_experts, 2 * intermediate_size, hidden_size),
        (num_local_experts, hidden_size, intermediate_size),
    )
    if nvfp4_mode == "w4a8":
        for label, view, shape in zip(
            ("w13", "w2"),
            (transformed_weights.w13, transformed_weights.w2),
            expected_shapes,
            strict=True,
        ):
            if not isinstance(view, w4a8_view_types):
                raise MoEEpConfigError(f"sm90_push_nvfp4 {label} is not a W4A8 view")
            if view.manifest.layout_version != payload_layout:
                raise MoEEpConfigError(
                    f"sm90_push_nvfp4 {label} payload layout does not match config"
                )
            if tuple(view.manifest.logical_shape) != shape:
                raise MoEEpConfigError(
                    f"sm90_push_nvfp4 {label} logical shape must be {shape}, got "
                    f"{tuple(view.manifest.logical_shape)}"
                )
            if view.manifest.group_size != group_size:
                raise MoEEpConfigError(
                    f"sm90_push_nvfp4 {label} group_size does not match config"
                )
            if view.manifest.residual_scheme != residual_scheme:
                raise MoEEpConfigError(
                    f"sm90_push_nvfp4 {label} residual_scheme does not match config"
                )
            if tuple(view.manifest.expert_mapping) != tuple(range(num_local_experts)):
                raise MoEEpConfigError(
                    f"sm90_push_nvfp4 {label} must map every local expert in order"
                )
            if not view.packed_e2m1.is_cuda:
                raise MoEEpConfigError(
                    f"sm90_push_nvfp4 {label} must be on a CUDA device"
                )
        w13_view = cast(
            "NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4",
            transformed_weights.w13,
        )
        w2_view = cast(
            "NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4",
            transformed_weights.w2,
        )
        if w13_view.packed_e2m1.device != w2_view.packed_e2m1.device:
            raise MoEEpConfigError("sm90_push_nvfp4 weights must share a CUDA device")
        return
    if nvfp4_mode != "w4a16_rs":
        raise MoEEpConfigError(f"unsupported nvfp4_mode {nvfp4_mode!r}")
    w13, w2 = transformed_weights.w13, transformed_weights.w2
    if not isinstance(w13, NVFP4RSWeightView) or not isinstance(w2, NVFP4RSWeightView):
        raise MoEEpConfigError("sm90_push_nvfp4 RS mode requires two RS views")
    if tuple(w13.payload.shape[:3]) != (
        num_local_experts,
        (2 * intermediate_size) // 64,
        hidden_size // 16,
    ):
        raise MoEEpConfigError("sm90_push_nvfp4 w13 RS shape mismatch")
    if tuple(w2.payload.shape[:3]) != (
        num_local_experts,
        hidden_size // 64,
        intermediate_size // 16,
    ):
        raise MoEEpConfigError("sm90_push_nvfp4 w2 RS shape mismatch")
    if not w13.payload.is_cuda or w13.payload.device != w2.payload.device:
        raise MoEEpConfigError("sm90_push_nvfp4 RS weights must share a CUDA device")


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    nvfp4_mode: str,
    group_size: int,
    residual_scheme: str,
    payload_layout: int = 4,
    weight_policy: str = "packed",
    hot_expert_count: int = 0,
) -> Any:
    import torch

    if nvfp4_mode not in ("w4a8", "w4a16_rs"):
        raise MoEEpConfigError(
            "sm90_push_nvfp4 nvfp4_mode must be 'w4a8' or 'w4a16_rs'"
        )
    if not isinstance(weights, MoEWeightPack):
        raise MoEEpConfigError(
            "sm90_push_nvfp4 weights must be MoEWeightPack, got "
            f"{type(weights).__name__}"
        )
    if weights.w13_scale is not None or weights.w2_scale is not None:
        raise MoEEpConfigError(
            "sm90_push_nvfp4 canonical preprocessing accepts BF16 weights only; "
            "pass Sm90PushNvFp4Weights through MegaConfig.transformed_weights "
            "for a quantized checkpoint"
        )
    expected = (
        ("w13", weights.w13, (num_local_experts, 2 * intermediate_size, hidden_size)),
        ("w2", weights.w2, (num_local_experts, hidden_size, intermediate_size)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm90_push_nvfp4 {name} must be BF16 with shape {shape}"
            )
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise MoEEpConfigError(
                f"sm90_push_nvfp4 {name} must be a contiguous CUDA tensor"
            )
    if weights.w13.device != weights.w2.device:
        raise MoEEpConfigError("sm90_push_nvfp4 w13 and w2 must share a device")
    w13_checkpoint = quantize_bf16_to_nvfp4_checkpoint(weights.w13)
    w2_checkpoint = quantize_bf16_to_nvfp4_checkpoint(weights.w2)
    if weight_policy == "packed":
        transformed = make_transformed_weights_from_checkpoints(
            w13_checkpoint,
            w2_checkpoint,
            nvfp4_mode=cast("Literal['w4a8', 'w4a16_rs']", nvfp4_mode),
            group_size=group_size,
            residual_scheme=residual_scheme,
            payload_layout=cast("Literal[3, 4]", payload_layout),
        )
    elif weight_policy in ("folded", "hot_folded"):
        hot_experts = (
            num_local_experts if weight_policy == "folded" else hot_expert_count
        )
        transformed = make_hot_folded_weights_from_checkpoints(
            w13_checkpoint,
            w2_checkpoint,
            hot_experts=hot_experts,
            group_size=group_size,
            residual_scheme=residual_scheme,
            payload_layout=cast("Literal[3, 4]", payload_layout),
        )
    elif weight_policy == "dual":
        transformed = make_dual_weights_from_checkpoints(
            w13_checkpoint,
            w2_checkpoint,
            group_size=group_size,
            residual_scheme=residual_scheme,
            payload_layout=cast("Literal[3, 4]", payload_layout),
        )
    else:
        raise MoEEpConfigError(f"unsupported NVFP4 weight_policy {weight_policy!r}")
    validate_transformed_mega_weights(
        transformed,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_local_experts=num_local_experts,
        nvfp4_mode=nvfp4_mode,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
        weight_policy=weight_policy,
        hot_expert_count=hot_expert_count,
    )
    return transformed


__all__ = [
    "TransformedMegaWeights",
    "estimate_residency",
    "load_modelopt_dual_weights",
    "load_modelopt_folded_fp8_weights",
    "load_modelopt_hot_folded_weights",
    "load_modelopt_transformed_weights",
    "make_folded_fp8_weights_from_checkpoints",
    "make_dual_weights_from_checkpoints",
    "make_hot_folded_weights_from_checkpoints",
    "make_transformed_weights_from_checkpoints",
    "preprocess_mega_weights",
    "quantize_bf16_to_nvfp4_checkpoint",
    "validate_transformed_mega_weights",
]
