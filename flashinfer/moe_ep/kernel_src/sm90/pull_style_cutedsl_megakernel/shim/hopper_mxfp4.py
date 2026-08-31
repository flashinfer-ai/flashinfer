# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM90 Humming MXFP4-weight x FP8-activation fused MegaMoE frontend.

This module is deliberately separate from :mod:`.hopper_fp8`: ordinary FP8
and Humming MXFP4 have different weight, scale, and cache identities even
though they share the same communication/session lifecycle and launch ABI.
The implementation reuses the established FP8 frontend's launch cache,
workspace teardown, CUDA-graph guards, tensor conversion, and high-level
symmetric-heap conventions while selecting only
``Sm90MegaMoESwapABMxfp4Fp8Kernel``.

Numerical/storage contract (E=local experts, T=tokens, H=hidden, I=post-SwiGLU
width):

* activation: E4M3 ``[T, H]``;
* activation scale wire: one FP32 scale replicated to ``[T, 4]``;
* FC1 packed E2M1 payload: uint8 ``[E, H/2, 2I]``, storage-K stride 1;
* FC2 packed E2M1 payload: uint8 ``[E, I/2, H]``, storage-K stride 1;
* folded K32 Humming offsets: uint8
  ``[E, N/64, logical_K/128, 16, 16]``;
* per-expert weight scale: FP32 Humming residual multiplied by 64.

The packed weights are never materialized as persistent E4M3 tensors.  The
kernel forms E4M3 bit patterns from E2M1 payloads and folded exponent offsets
in the WGMMA input path, then applies the common residual compensation in the
epilogue.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Tuple

import torch

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    normalize_sm90_routing_profile,
)

from .comm import (
    _CompiledMega,
    _compute_peer_offsets,
    ensure_not_capturing,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .hopper_fp8 import (
    MegaMoEHopperFp8Config,
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
    _kind_to_cutlass_dtype,
    _sym_zeros_byte_view_1b,
)


_MXFP4_MMA_TILER_DEFAULT = (128, 32, 128)
_MXFP4_CLUSTER_SHAPE_DEFAULT = (1, 1, 1)

_MXFP4_GROUP_SIZE = 32
_MXFP4_FOLD_M = 64
_MXFP4_FOLD_K = 128
_MXFP4_GATE_UP_INTERLEAVE = 8
_MXFP4_HUMMING_MAX_RANGE = 11
_MXFP4_EPILOGUE_COMPENSATION = 64

# This tuple is intentionally human-readable and embedded verbatim in every
# compile key.  Changing preprocessing/layout semantics must create a new
# identity rather than reusing an ordinary FP8 or older MXFP4 specialization.
_MXFP4_COMPILE_IDENTITY = (
    "e2m1_k32",
    "humming_v1",
    "fold64x128",
    "residual64",
    "gateup8",
    "fused",
)

# The generic knob-cache schema has a single string ``dtype`` axis. Encode
# every fixed Humming/layout semantic in that axis so a future MXFP4 layout
# cannot consume a winner recorded for this Phase-A ABI. Tactic geometry
# remains in the cached knob value and problem geometry in the normal key.
_MXFP4_TUNING_DTYPE_ID = (
    "sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_"
    "humming_v1_fold_m64_k128_gateup8_packedk2_residual64_"
    "swapab_fused"
)

_SUPPORTED_MXFP4_KNOBS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "group_hint",
        "flag_batch",
        "epi_flag_batch",
        "load_balance_mode",
        "token_back_mode",
        "in_kernel_fc2_reduce",
        "clc_bundle_size",
        "num_sched_stages",
    }
)
_REQUIRED_CACHED_MXFP4_GEOMETRY = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
    }
)


def _validate_mxfp4_knobs(
    knobs: Dict[str, Any],
    *,
    source: str,
    require_swap_ab: bool,
    require_complete_geometry: bool = False,
) -> Dict[str, Any]:
    """Validate and normalize one MXFP4-only tactic without dropping fields."""
    if not isinstance(knobs, dict):
        raise ValueError(f"{source} must provide an MXFP4 knob dict.")
    unknown = set(knobs).difference(_SUPPORTED_MXFP4_KNOBS)
    if unknown:
        rendered = ", ".join(sorted(map(repr, unknown)))
        raise ValueError(
            f"{source} contains unsupported MXFP4 knob field(s): {rendered}; "
            "refusing to silently ignore them"
        )
    if require_complete_geometry:
        missing = _REQUIRED_CACHED_MXFP4_GEOMETRY.difference(knobs)
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ValueError(
                f"{source} is missing required MXFP4 geometry field(s): {rendered}"
            )
    if require_swap_ab and knobs.get("swap_ab") is not True:
        raise ValueError(f"{source} must include swap_ab=True.")
    if "swap_ab" in knobs and knobs["swap_ab"] is not True:
        raise ValueError(f"{source} cannot select native A/B; swap_ab must be True.")
    if knobs.get("fp8_accum_mode", "1xacc") != "1xacc":
        raise ValueError(f"{source} must use fp8_accum_mode='1xacc'.")
    if knobs.get("in_kernel_fc2_reduce", False) is not False:
        raise ValueError(f"{source} must use the standalone top-k reduce.")

    normalized = dict(knobs)
    for name in ("mma_tiler_mnk", "cluster_shape_mnk", "epi_flag_batch"):
        if name in normalized:
            value = normalized[name]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{source} field {name!r} must be a list or tuple.")
            normalized[name] = tuple(value)
    return normalized


@dataclasses.dataclass(frozen=True)
class MegaMoEHopperMxfp4Config(MegaMoEHopperFp8Config):
    """Compile/launch configuration for fused Humming MXFP4 x FP8.

    The inherited communication and scheduling knobs remain available, but
    the numerical format is fixed: E4M3 activation, packed E2M1/K32 weights,
    hybrid per-token activation scaling, one FP32 accumulator, and swap-AB.
    """

    kind: Literal["fp8_e4m3"] = "fp8_e4m3"
    fp8_scale_mode: Literal["mxfp4_hybrid"] = "mxfp4_hybrid"
    fp8_accum_mode: Literal["1xacc"] = "1xacc"
    swap_ab: bool = True
    mma_tiler_mnk: Tuple[int, int, int] = _MXFP4_MMA_TILER_DEFAULT
    routing_profile: str = field(
        default=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "routing_profile",
            normalize_sm90_routing_profile(self.routing_profile),
        )
        if self.kind != "fp8_e4m3":
            raise ValueError(
                f"Hopper MXFP4 x FP8 requires kind='fp8_e4m3'; got {self.kind!r}."
            )
        if self.fp8_scale_mode != "mxfp4_hybrid":
            raise ValueError(
                "Hopper MXFP4 x FP8 requires "
                "fp8_scale_mode='mxfp4_hybrid'; "
                f"got {self.fp8_scale_mode!r}."
            )
        if self.fp8_accum_mode != "1xacc":
            raise ValueError(
                "Hopper MXFP4 x FP8 supports only fp8_accum_mode='1xacc'; "
                f"got {self.fp8_accum_mode!r}."
            )
        if not self.swap_ab:
            raise ValueError(
                "Hopper MXFP4 x FP8 supports only the swap-AB kernel; "
                "ordinary FP8/native-layout fallback is forbidden."
            )

        # Reuse the target FP8 config's rank, shape, scheduler, cluster,
        # ping-pong, token-back, and flag validation.  The generic base does
        # not recognize the dedicated hybrid string, so validate an otherwise
        # identical blockwise proxy (the common divisibility requirements are
        # the same) and apply MXFP4-only constraints below.
        base_kwargs = {
            item.name: getattr(self, item.name)
            for item in dataclasses.fields(MegaMoEHopperFp8Config)
        }
        base_kwargs["fp8_scale_mode"] = "blockwise"
        MegaMoEHopperFp8Config(**base_kwargs)

        tile_m, _tile_n, tile_k = self.mma_tiler_mnk
        if tile_m not in (128, 256):
            raise ValueError(
                "Hopper MXFP4 x FP8 requires swap-AB MMA tile M=128 or 256; "
                f"got mma_tiler_mnk={self.mma_tiler_mnk}."
            )
        if tile_k not in (128, 256):
            raise ValueError(
                "Hopper MXFP4 x FP8 requires MMA tile K=128 or 256; "
                f"got mma_tiler_mnk={self.mma_tiler_mnk}."
            )
        for name, logical_k in (
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
        ):
            if logical_k % tile_k:
                raise ValueError(
                    f"Hopper MXFP4 x FP8 requires {name} ({logical_k}) "
                    f"divisible by MMA tile K={tile_k}."
                )

    @property
    def mxfp4_hybrid(self) -> bool:
        return True


@dataclasses.dataclass
class MegaMoEHopperMxfp4Inputs(MegaMoEHopperFp8Inputs):
    """Per-rank tensors for one fused Humming MXFP4 x FP8 launch.

    Field names intentionally match :class:`MegaMoEHopperFp8Inputs` because
    the composite kernel keeps the same thirteen-tensor runtime ABI.  Their
    dtypes/shapes have the MXFP4 meanings documented at module scope.
    """


class MegaMoEHopperMxfp4Frontend(MegaMoEHopperFp8Frontend):
    """Lazy-compile frontend for ``Sm90MegaMoESwapABMxfp4Fp8Kernel`` only."""

    _mega_key: Optional[tuple]
    _mega: Optional[_CompiledMega]

    def __init__(self, config: MegaMoEHopperMxfp4Config) -> None:
        super().__init__(config)

    def apply_knobs(self, knobs: Optional[dict]) -> None:
        """Apply only declared MXFP4 tactics; reject stale/cache-only fields."""
        if not knobs:
            return
        validated = _validate_mxfp4_knobs(
            knobs,
            source="MXFP4 frontend.apply_knobs()",
            require_swap_ab=False,
        )
        new_config = dataclasses.replace(self.config, **validated)
        if new_config == self.config:
            return
        ensure_not_capturing("MXFP4 apply_knobs (config change)")
        self._release_workspace()
        self._config = new_config

    @property
    def config(self) -> MegaMoEHopperMxfp4Config:
        config = super().config
        if not isinstance(config, MegaMoEHopperMxfp4Config):
            raise RuntimeError("MXFP4 frontend lost its format-specific config.")
        return config

    def _mega_compile_key(self) -> tuple:
        c = self.config
        return (
            "sm90_mxfp4_fp8_megamoe",
            *_MXFP4_COMPILE_IDENTITY,
            # Make logical-K versus packed storage-K explicit.  Shape fields
            # later in the inherited key would distinguish these in practice,
            # but naming both protects against future generic packed formats.
            ("fc1_logical_storage_k", c.hidden, c.hidden // 2),
            ("fc2_logical_storage_k", c.intermediate, c.intermediate // 2),
            *super()._mega_compile_key(),
        )

    @staticmethod
    def _assert_mirrored_constants() -> None:
        """Fail closed if the clean vendor drop changes the Humming layout."""
        MegaMoEHopperFp8Frontend._assert_mirrored_constants()

        from moe_hopper_fp8.mxfp4_cutedsl import (
            MXFP4_FOLD_M,
            MXFP4_GROUP_SIZE,
            MXFP4_K_TILE,
        )

        mirrored = (
            ("MXFP4_GROUP_SIZE", MXFP4_GROUP_SIZE, _MXFP4_GROUP_SIZE),
            ("MXFP4_FOLD_M", MXFP4_FOLD_M, _MXFP4_FOLD_M),
            ("MXFP4_K_TILE", MXFP4_K_TILE, _MXFP4_FOLD_K),
        )
        for name, source_value, expected in mirrored:
            if source_value != expected:
                raise RuntimeError(
                    f"kernel drop changed {name} ({source_value} != {expected}); "
                    "bump the MXFP4 compile identity and re-audit preprocessing, "
                    "folded-offset shapes, and launch validation."
                )

    def _ensure_mega_compiled(self, inputs: MegaMoEHopperFp8Inputs) -> _CompiledMega:
        if not isinstance(inputs, MegaMoEHopperMxfp4Inputs):
            raise TypeError("MXFP4 frontend requires MegaMoEHopperMxfp4Inputs.")
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        ensure_not_capturing("MXFP4 cute.compile + symmetric-heap allocation")
        self._release_workspace()

        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils

        from common.megamoe_constants import Fp8E8M0SfVecSize, SfPaddingBlock
        from moe_hopper_fp8.megamoe_kernel_fp8 import (
            Sm90MegaMoESwapABMxfp4Fp8Kernel,
        )

        self._assert_mirrored_constants()
        c = self.config
        if not c.swap_ab:
            # Redundant with config validation by design: never let a mutated
            # object silently select the ordinary FP8/native class.
            raise RuntimeError("MXFP4 frontend requires swap_ab=True.")

        static_expert_shape = (
            c.num_experts_per_rank,
            c.fc1_out,
            c.hidden,
        )
        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        group_hint = c.group_hint if c.group_hint is not None else max_active_clusters
        token_padding_block = c.mma_tiler_mnk[1]

        # There is intentionally no conditional class selection or fallback.
        kernel = Sm90MegaMoESwapABMxfp4Fp8Kernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=c.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=c.force_static_sched,
            clc_bundle_size=c.clc_bundle_size,
            num_sched_stages=c.num_sched_stages,
            ab_dtype=_kind_to_cutlass_dtype(c.kind),
            sf_vec_size=Fp8E8M0SfVecSize,
            fp8_scale_mode="mxfp4_hybrid",
            fp8_accum_mode="1xacc",
            pingpong=c.pingpong,
            world_size=c.world_size,
            local_rank=c.rank,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_in_kernel_topk_reduce=c.in_kernel_fc2_reduce,
            apply_topk_in_fc1=c.apply_topk_in_fc1,
            token_back_mode=c.resolved_token_back_mode,
            epi_flag_batch=c.epi_flag_batch,
            flag_batch=c.flag_batch,
            gate_up_clamp=self._gate_up_clamp,
        )

        local_ws_bytes, shared_ws_bytes = kernel.get_workspace_sizes()
        local_workspace = torch.zeros(
            (local_ws_bytes,), dtype=torch.uint8, device="cuda"
        )
        shared_workspace = sym_zeros((shared_ws_bytes,), torch.uint8)
        symmetric_base, peer_offsets_list = _compute_peer_offsets(
            shared_workspace, c.world_size
        )
        mega = _CompiledMega(
            compiled=None,
            kernel=kernel,
            local_workspace=local_workspace,
            shared_workspace=shared_workspace,
            symmetric_base=symmetric_base,
            peer_offsets_list=peer_offsets_list,
        )
        compile_kwargs = self._build_mega_runtime_kwargs(inputs, mega)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if c.enable_iket:
            compile_kwargs["options"] = "iket"
        mega.compiled = cute.compile(kernel, **compile_kwargs)
        self._mega_key = key
        self._mega = mega
        return mega

    @staticmethod
    def _slice_inputs(
        inputs: MegaMoEHopperFp8Inputs,
        num_tokens: int,
    ) -> MegaMoEHopperMxfp4Inputs:
        if not isinstance(inputs, MegaMoEHopperMxfp4Inputs):
            raise TypeError("MXFP4 frontend requires MegaMoEHopperMxfp4Inputs.")
        tok = slice(None, num_tokens)
        return MegaMoEHopperMxfp4Inputs(
            activation=inputs.activation[tok],
            activation_sf=inputs.activation_sf[tok],
            topk_idx=inputs.topk_idx[tok],
            topk_weights=inputs.topk_weights[tok],
            fc1_weight=inputs.fc1_weight,
            fc1_weight_sf=inputs.fc1_weight_sf,
            fc1_activation_dequant_scale=inputs.fc1_activation_dequant_scale,
            fc1_weight_dequant_scale=inputs.fc1_weight_dequant_scale,
            fc2_weight=inputs.fc2_weight,
            fc2_weight_sf=inputs.fc2_weight_sf,
            fc2_activation_dequant_scale=inputs.fc2_activation_dequant_scale,
            fc2_weight_dequant_scale=inputs.fc2_weight_dequant_scale,
            output_activation=inputs.output_activation[tok],
        )

    def _validate_inputs(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: int,
    ) -> None:
        if not isinstance(inputs, MegaMoEHopperMxfp4Inputs):
            raise TypeError("MXFP4 frontend requires MegaMoEHopperMxfp4Inputs.")
        c = self.config
        buf_tokens = int(inputs.activation.shape[0])
        if num_tokens > buf_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds activation buffer size "
                f"({buf_tokens})."
            )
        if num_tokens > c.num_tokens_per_rank:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds config.num_tokens_per_rank "
                f"({c.num_tokens_per_rank})."
            )

        current_device = torch.cuda.current_device()

        def require_cuda(name: str, tensor: torch.Tensor) -> None:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor.")
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor.")
            if tensor.device.index != current_device:
                raise ValueError(
                    f"{name} must be on current cuda:{current_device}, "
                    f"got {tensor.device}."
                )

        require_cuda("activation", inputs.activation)
        if tuple(inputs.activation.shape) != (buf_tokens, c.hidden):
            raise ValueError(
                f"activation must have shape ({buf_tokens}, {c.hidden}), "
                f"got {tuple(inputs.activation.shape)}."
            )
        if inputs.activation.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "MXFP4 MegaMoE activation must be torch.float8_e4m3fn, "
                f"got {inputs.activation.dtype}."
            )
        if inputs.activation.stride(-1) != 1:
            raise ValueError("activation must have contiguous hidden dimension.")

        require_cuda("activation_sf", inputs.activation_sf)
        if tuple(inputs.activation_sf.shape) != (buf_tokens, 4):
            raise ValueError(
                "MXFP4 hybrid activation_sf must have exact shape "
                f"({buf_tokens}, 4), got {tuple(inputs.activation_sf.shape)}."
            )
        if inputs.activation_sf.dtype != torch.float32:
            raise ValueError(
                "MXFP4 hybrid activation_sf must be float32, "
                f"got {inputs.activation_sf.dtype}."
            )
        if not inputs.activation_sf.is_contiguous():
            raise ValueError("activation_sf [T,4] must be contiguous.")
        if inputs.activation_sf.data_ptr() % 16:
            raise ValueError("activation_sf rows must start at 16-byte alignment.")

        token_contracts = (
            (
                "topk_idx",
                inputs.topk_idx,
                (buf_tokens, c.num_topk),
                torch.int64,
            ),
            (
                "topk_weights",
                inputs.topk_weights,
                (buf_tokens, c.num_topk),
                torch.float32,
            ),
            (
                "output_activation",
                inputs.output_activation,
                (buf_tokens, c.hidden),
                torch.bfloat16,
            ),
        )
        for name, tensor, token_shape, dtype in token_contracts:
            require_cuda(name, tensor)
            if tuple(tensor.shape) != token_shape:
                raise ValueError(
                    f"{name} must have shape {token_shape}, got {tuple(tensor.shape)}."
                )
            if tensor.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}.")

        e = c.num_experts_per_rank
        packed_weight_contracts = (
            (
                "fc1_weight",
                inputs.fc1_weight,
                (e, c.hidden // 2, c.fc1_out),
            ),
            (
                "fc2_weight",
                inputs.fc2_weight,
                (e, c.intermediate // 2, c.hidden),
            ),
        )
        for name, tensor, weight_shape in packed_weight_contracts:
            require_cuda(name, tensor)
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != weight_shape:
                raise ValueError(
                    f"{name} must be packed E2M1 uint8 with shape "
                    f"{weight_shape}; "
                    f"got {tensor.dtype} {tuple(tensor.shape)}."
                )
            if tensor.stride(1) != 1:
                raise ValueError(
                    f"{name} must have packed storage-K stride 1 on dim 1; "
                    f"got strides {tuple(tensor.stride())}."
                )
            if tensor.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned.")

        offset_contracts = (
            (
                "fc1_weight_sf",
                inputs.fc1_weight_sf,
                (e, c.fc1_out // 64, c.hidden // 128, 16, 16),
            ),
            (
                "fc2_weight_sf",
                inputs.fc2_weight_sf,
                (e, c.hidden // 64, c.intermediate // 128, 16, 16),
            ),
        )
        for name, tensor, offset_shape in offset_contracts:
            require_cuda(name, tensor)
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != offset_shape:
                raise ValueError(
                    f"{name} must be folded Humming uint8 offsets with shape "
                    f"{offset_shape}; got {tensor.dtype} {tuple(tensor.shape)}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")
            if tensor.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned.")

        scale_contracts = (
            (
                "fc1_activation_dequant_scale",
                inputs.fc1_activation_dequant_scale,
                (1,),
            ),
            (
                "fc1_weight_dequant_scale",
                inputs.fc1_weight_dequant_scale,
                (e,),
            ),
            (
                "fc2_activation_dequant_scale",
                inputs.fc2_activation_dequant_scale,
                (1,),
            ),
            (
                "fc2_weight_dequant_scale",
                inputs.fc2_weight_dequant_scale,
                (e,),
            ),
        )
        for name, tensor, scale_shape in scale_contracts:
            require_cuda(name, tensor)
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != scale_shape:
                raise ValueError(
                    f"{name} must be float32 with shape {scale_shape}; "
                    f"got {tensor.dtype} {tuple(tensor.shape)}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")


# Kernel-ready leg: processed packed payload, folded offset, static unit
# activation placeholder, and per-expert Humming residual * 64.
TransformedMxfp4Weights = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass
class MegaMoEHopperMxfp4SymmBuffer:
    """Symmetric staging/workspace owner for one fused MXFP4 session."""

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int

    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor

    _frontend: MegaMoEHopperMxfp4Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    @property
    def kind(self) -> str:
        return "fp8_e4m3"

    @property
    def fp8_scale_mode(self) -> str:
        return "mxfp4_hybrid"

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    def destroy(self) -> None:
        """Release compiled workspaces before symmetric input buffers."""
        if self._destroyed:
            return
        self._frontend.release()
        for root in self._sym_roots:
            free_sym_tensor(root)
        self._sym_roots.clear()
        self._destroyed = True


def _resolve_mxfp4_knobs(
    knobs: Optional[Any],
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_total_experts: int,
    num_topk: int,
    num_max_tokens: int,
    gate_up_clamp: Optional[float] = None,
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
) -> Dict[str, Any]:
    """Resolve only MXFP4-keyed knobs; never consult FP8 heuristics/cache."""
    routing_profile = normalize_sm90_routing_profile(routing_profile)
    if isinstance(knobs, dict):
        return _validate_mxfp4_knobs(
            knobs,
            source="explicit MXFP4 knobs",
            require_swap_ab=True,
        )
    if knobs == "auto":
        raise ValueError(
            "direct MXFP4 knob resolution cannot execute collective autotune; "
            "pass knobs='auto' through Sm90PullMxfp4MegaKernelBackend"
        )
    if knobs is not None:
        raise ValueError("knobs must be None or an MXFP4 knob dict.")

    from .knob_cache import lookup_knobs
    from .mxfp4_tuner import (
        hopper_mxfp4_ordered_candidates,
        is_hopper_mxfp4_tactic_shape_compatible,
        require_hopper_mxfp4_tuning_device,
        validate_hopper_mxfp4_tactic,
    )

    require_hopper_mxfp4_tuning_device()
    cached = lookup_knobs(
        dtype=_MXFP4_TUNING_DTYPE_ID,
        fp8_scale_mode="mxfp4_hybrid",
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_total_experts,
        topk=num_topk,
        max_tokens=num_max_tokens,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
    )
    if cached is not None:
        normalized = _validate_mxfp4_knobs(
            cached,
            source="MXFP4 tuning-cache entry",
            require_swap_ab=True,
            require_complete_geometry=True,
        )
        tactic = validate_hopper_mxfp4_tactic(
            normalized,
            execution_mode="fused",
        )
        if not is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode="fused",
            hidden=hidden,
            intermediate=intermediate,
        ):
            raise ValueError(
                "MXFP4 tuning-cache tactic is incompatible with "
                f"hidden={hidden}, intermediate={intermediate}"
            )
        return tactic

    # Prefer the bucket winner when legal; smaller shapes may require the
    # first legal tactic in the stable manifest union instead.
    return hopper_mxfp4_ordered_candidates(
        num_max_tokens,
        execution_mode="fused",
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=routing_profile,
    )[0]


def get_symm_buffer_for_hopper_mxfp4_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: Literal["fp8_e4m3"] = "fp8_e4m3",
    fp8_scale_mode: Literal["mxfp4_hybrid"] = "mxfp4_hybrid",
    fp8_accum_mode: Literal["1xacc"] = "1xacc",
    knobs: Optional[Any] = None,
    swap_ab: Optional[bool] = None,
    pingpong: Optional[bool] = None,
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mnk: Optional[Tuple[int, int, int]] = None,
    load_balance_mode: Literal["static", "atomic_counter"] = "static",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None,
    apply_topk_in_fc1: bool = True,
    group_hint: Optional[int] = None,
    clc_bundle_size: Optional[int] = None,
    num_sched_stages: Optional[int] = None,
    flag_batch: int = 1,
    epi_flag_batch: Tuple[int, int] = (2, 4),
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
) -> MegaMoEHopperMxfp4SymmBuffer:
    """Allocate symmetric inputs and a fused Humming MXFP4 frontend.

    ``knobs=None`` performs a lookup using a dedicated, versioned MXFP4
    fused identity, then uses the manifest-derived per-token heuristic on a
    cache miss. Collective ``knobs="auto"`` is owned by the production
    backend because it requires live weights, staged routing, and every EP
    rank in lockstep. Explicit geometry and ``knobs=`` are mutually exclusive.
    """
    routing_profile = normalize_sm90_routing_profile(routing_profile)
    manual_geometry = any(
        value is not None
        for value in (swap_ab, pingpong, mma_tiler_mnk, cluster_shape_mnk)
    )
    if manual_geometry and knobs is not None:
        raise ValueError("pass either explicit launch geometry or knobs=, not both.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if manual_geometry:
        # Manual geometry is completely local: do not consult a cache and
        # accidentally inherit non-geometry choices from a tuned session.
        resolved: Dict[str, Any] = dict(
            swap_ab=True if swap_ab is None else swap_ab,
            pingpong=False if pingpong is None else pingpong,
            mma_tiler_mnk=(
                _MXFP4_MMA_TILER_DEFAULT
                if mma_tiler_mnk is None
                else tuple(mma_tiler_mnk)
            ),
            cluster_shape_mnk=(
                _MXFP4_CLUSTER_SHAPE_DEFAULT
                if cluster_shape_mnk is None
                else tuple(cluster_shape_mnk)
            ),
        )
    else:
        resolved = _resolve_mxfp4_knobs(
            knobs,
            world_size=world_size,
            hidden=hidden,
            intermediate=intermediate,
            num_total_experts=num_total_experts,
            num_topk=num_topk,
            num_max_tokens=num_max_tokens,
            gate_up_clamp=clamp,
            routing_profile=routing_profile,
        )
        # Match the established FP8 shim's precedence rule: an explicit
        # caller token-back choice wins over a cache entry.
        if token_back_mode is not None:
            resolved.pop("token_back_mode", None)

    config = MegaMoEHopperMxfp4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        kind=kind,
        fp8_scale_mode=fp8_scale_mode,
        fp8_accum_mode=fp8_accum_mode,
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=_MXFP4_MMA_TILER_DEFAULT,
        cluster_shape_mnk=_MXFP4_CLUSTER_SHAPE_DEFAULT,
        load_balance_mode=load_balance_mode,
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        apply_topk_in_fc1=apply_topk_in_fc1,
        group_hint=group_hint,
        clc_bundle_size=clc_bundle_size,
        num_sched_stages=num_sched_stages,
        flag_batch=flag_batch,
        epi_flag_batch=epi_flag_batch,
        routing_profile=routing_profile,
    )
    if resolved:
        validated = _validate_mxfp4_knobs(
            resolved,
            source="resolved MXFP4 knobs",
            require_swap_ab=True,
        )
        config = dataclasses.replace(config, **validated)
        if not isinstance(config, MegaMoEHopperMxfp4Config):
            raise RuntimeError("MXFP4 knob application changed config type.")

    frontend = MegaMoEHopperMxfp4Frontend(config)
    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view_1b((num_max_tokens, hidden), torch.float8_e4m3fn)
    sym_roots.append(x_root)

    # One logical token scale repeated into four FP32 words.  The staging
    # layer writes all four lanes; keeping the allocation exact makes an
    # accidental generic blockwise [T,H/128] wire fail validation.
    x_sf = sym_zeros((num_max_tokens, 4), torch.float32)
    sym_roots.append(x_sf)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    topk_idx.fill_(-1)
    sym_roots.append(topk_idx)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    sym_roots.append(topk_weights)
    output_activation = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(output_activation)

    return MegaMoEHopperMxfp4SymmBuffer(
        num_total_experts=num_total_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        output_activation=output_activation,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def _resolve_transformed_mxfp4_weights(
    transformed: TransformedMxfp4Weights,
    leg: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        weight, folded_offset, activation_placeholder, residual_x64 = transformed
    except (TypeError, ValueError):
        raise ValueError(
            f"transformed_{leg} must be a strict four-slot tuple "
            "(packed_weight, folded_offset, unit_activation_placeholder, "
            "residual_x64)."
        ) from None
    if any(
        item is None
        for item in (weight, folded_offset, activation_placeholder, residual_x64)
    ):
        raise ValueError(
            f"transformed_{leg} cannot contain None; MXFP4 metadata does not "
            "use ordinary-FP8 unit-scale substitution."
        )
    return weight, folded_offset, activation_placeholder, residual_x64


class _Mxfp4InputBuffer(Protocol):
    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor


def _build_mxfp4_inputs(
    symm_buffer: _Mxfp4InputBuffer,
    transformed_l1: TransformedMxfp4Weights,
    transformed_l2: TransformedMxfp4Weights,
) -> MegaMoEHopperMxfp4Inputs:
    (
        fc1_weight,
        fc1_weight_sf,
        fc1_activation_dequant_scale,
        fc1_weight_dequant_scale,
    ) = _resolve_transformed_mxfp4_weights(transformed_l1, "l1")
    (
        fc2_weight,
        fc2_weight_sf,
        fc2_activation_dequant_scale,
        fc2_weight_dequant_scale,
    ) = _resolve_transformed_mxfp4_weights(transformed_l2, "l2")
    return MegaMoEHopperMxfp4Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc1_activation_dequant_scale=fc1_activation_dequant_scale,
        fc1_weight_dequant_scale=fc1_weight_dequant_scale,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        fc2_activation_dequant_scale=fc2_activation_dequant_scale,
        fc2_weight_dequant_scale=fc2_weight_dequant_scale,
        output_activation=symm_buffer.output_activation,
    )


def hopper_mxfp4_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedMxfp4Weights,
    transformed_l2: TransformedMxfp4Weights,
    symm_buffer: MegaMoEHopperMxfp4SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch fused dispatch + MXFP4 FC1/FC2 + combine with no fallback."""
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM90 MXFP4 path.",
            UserWarning,
            stacklevel=2,
        )
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")

    n = num_tokens if num_tokens is not None else symm_buffer.num_max_tokens
    if n < 0 or n > symm_buffer.num_max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {symm_buffer.num_max_tokens}], got {n}."
        )
    if n == 0 and symm_buffer._frontend.config.in_kernel_fc2_reduce:
        return symm_buffer.output_activation[:0] if y is None else None
    if y is not None:
        if tuple(y.shape) != (n, symm_buffer.hidden):
            raise ValueError(
                f"y must have shape ({n}, {symm_buffer.hidden}), got {tuple(y.shape)}."
            )
        if y.dtype != torch.bfloat16:
            raise ValueError(f"y must be bfloat16, got {y.dtype}.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)

    inputs = _build_mxfp4_inputs(symm_buffer, transformed_l1, transformed_l2)
    # Launch the full padded workspace.  topk_idx[n:] == -1 masks tail rows;
    # this matches the kernel-team fused driver and keeps compile shape stable.
    out = symm_buffer._frontend.run(inputs, num_tokens=None, sync=False)
    if y is None:
        result = out[:n] if out is not None else symm_buffer.output_activation[:0]
    else:
        result = None
        if out is not None:
            y.copy_(out[:n])
    if sync and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
    return result


def hopper_mxfp4_mega_launch_thunk(
    transformed_l1: TransformedMxfp4Weights,
    transformed_l2: TransformedMxfp4Weights,
    symm_buffer: MegaMoEHopperMxfp4SymmBuffer,
) -> Callable[[], None]:
    """Build a zero-argument steady-state fused MXFP4 launch thunk."""
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    inputs = _build_mxfp4_inputs(symm_buffer, transformed_l1, transformed_l2)
    return symm_buffer._frontend.make_launch_thunk(inputs)


_SPLIT_EXPORTS = (
    "MegaMoEHopperMxfp4SplitConfig",
    "MegaMoEHopperMxfp4SplitSession",
    "MegaMoEHopperMxfp4SplitSymmBuffer",
    "Mxfp4SplitError",
    "Mxfp4SplitLifecycleError",
    "Mxfp4SplitSessionPoisonedError",
    "Mxfp4SplitUnavailableError",
    "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
    "hopper_mxfp4_split_mega_launch_thunk",
    "hopper_mxfp4_split_mega_moe",
)


def __getattr__(name):  # PEP 562
    if name not in _SPLIT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import hopper_mxfp4_split

    value = getattr(hopper_mxfp4_split, name)
    globals()[name] = value
    return value


__all__ = [
    *_SPLIT_EXPORTS,
    "MegaMoEHopperMxfp4Config",
    "MegaMoEHopperMxfp4Frontend",
    "MegaMoEHopperMxfp4Inputs",
    "MegaMoEHopperMxfp4SymmBuffer",
    "TransformedMxfp4Weights",
    "get_symm_buffer_for_hopper_mxfp4_mega_moe",
    "hopper_mxfp4_mega_launch_thunk",
    "hopper_mxfp4_mega_moe",
]
