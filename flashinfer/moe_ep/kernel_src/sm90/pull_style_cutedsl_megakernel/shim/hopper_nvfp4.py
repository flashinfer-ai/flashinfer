# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM90 NVFP4-weight x FP8-activation fused MegaMoE frontend.

This module is deliberately separate from :mod:`.hopper_fp8`: ordinary FP8
and NVFP4 have different weight, scale, and cache identities even
though they share the same communication/session lifecycle and launch ABI.
The implementation reuses the established FP8 frontend's launch cache,
workspace teardown, CUDA-graph guards, tensor conversion, and high-level
symmetric-heap conventions while selecting only
``Sm90MegaMoESwapABNvfp4Fp8Kernel``.

Numerical/storage contract (E=local experts, T=tokens, H=hidden, I=post-SwiGLU
width):

* activation: E4M3 ``[T, H]``;
* activation scale wire: one FP32 scale replicated to ``[T, 4]``;
* FC1 packed E2M1 payload: uint8 ``[E, H/2, 2I]``, storage-K stride 1;
* FC2 packed E2M1 payload: uint8 ``[E, I/2, H]``, storage-K stride 1;
* folded K16 E4M3 scales: uint8
  ``[E, N/64, logical_K/128, 32, 16]``;
* per-expert weight scale: FP32 global scale multiplied by 8.

The packed weights are never materialized as persistent E4M3 tensors.  The
kernel forms E4M3 bit patterns from E2M1 payloads and folded K16 scale bytes
in the WGMMA input path, then applies the common residual compensation in the
epilogue.

``nvfp4_kperm`` (0/32/64) selects a K-permuted weight layout: payload K, W1
output rows and FC1 activations are permuted inside each K32/K64 chunk, and
scale bytes are swapped after the gate/up interleave (see the backend's weight
preparation). Weights, staged activations and kernel must use the same value.
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
from ..src.moe_hopper_fp8.mxfp4_policy import resolve_mxfp4_optimizations
from .nvfp4_policy import (
    NVFP4_FC1_SCHEDULE_VERSION,
    Nvfp4ConversionLayout,
    nvfp4_fc1_auto_shape_eligible,
    resolve_nvfp4_fc1_overlap,
    split_nvfp4_conversion_tactic,
    validate_nvfp4_fc1_overlap_override,
    validate_nvfp4_kperm,
)


_NVFP4_MMA_TILER_DEFAULT = (128, 32, 128)
_NVFP4_CLUSTER_SHAPE_DEFAULT = (1, 1, 1)

_NVFP4_GROUP_SIZE = 16
_NVFP4_FOLD_M = 64
_NVFP4_FOLD_K = 128
_NVFP4_GATE_UP_INTERLEAVE = 8
_NVFP4_EPILOGUE_COMPENSATION = 8

# This tuple is intentionally human-readable and embedded verbatim in every
# compile key.  Changing preprocessing/layout semantics must create a new
# identity rather than reusing an ordinary FP8 or older NVFP4 specialization.
_NVFP4_COMPILE_IDENTITY = (
    "e2m1_k16",
    "e4m3_sf_single_v1",
    "fold64x128",
    "global8",
    "gateup8",
    "fused_layout_v2",
    "shared_lut_grouped_single_v1",
    NVFP4_FC1_SCHEDULE_VERSION,
)

# Encode the packed weight/activation layout in the cache dtype so
# incompatible layouts cannot share winners. Tactic geometry remains in
# the cached value and problem geometry in the key.
_NVFP4_TUNING_DTYPE_ID = (
    "sm90_w_nvfp4_e2m1_k16_a_fp8_e4m3_per_token_full_hidden_"
    "e4m3_sf_single_v1_fold_m64_k128_gateup8_packedk2_global8_"
    "swapab_fused_layout_v2_nvfp4_single_v1_" + NVFP4_FC1_SCHEDULE_VERSION
)


def _validate_complete_nvfp4_tactic(
    knobs: Dict[str, Any], *, source: str
) -> Dict[str, Any]:
    """Validate one complete external fused tactic through the tuner schema.

    The MXFP4 schema fields are followed by the NVFP4 conversion fields
    (``compact_pull_buffer``, ``nvfp4_lut_replicas``, ``nvfp4_lut_map``); a tactic without them selects the full pull buffer and the
    8-replica lane-quad table.
    """

    if not isinstance(knobs, dict):
        raise ValueError(f"{source} must provide an NVFP4 knob dict.")
    from .mxfp4_optimization import normalize_mxfp4_optimization_tactic

    try:
        core, conversion = split_nvfp4_conversion_tactic(knobs)
        return dict(normalize_mxfp4_optimization_tactic(core), **conversion)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class MegaMoEHopperNvfp4Config(MegaMoEHopperFp8Config):
    """Compile/launch configuration for fused NVFP4 x FP8.

    The inherited communication and scheduling knobs remain available, but
    the numerical format is fixed: E4M3 activation, packed E2M1/K16 weights,
    hybrid per-token activation scaling, one FP32 accumulator, and swap-AB.
    """

    kind: Literal["fp8_e4m3"] = "fp8_e4m3"
    # The frozen dataclass subclass narrows the inherited runtime contract,
    # but mypy treats dataclass field overrides as mutable/invariant.
    fp8_scale_mode: Literal["mxfp4_hybrid"] = "mxfp4_hybrid"  # type: ignore[assignment]
    fp8_accum_mode: Literal["1xacc"] = "1xacc"
    swap_ab: bool = True
    # Preserve the default NVFP4 layout when the FP8 base gains new features.
    # The compact pull buffer (one dispatch slot per active warp) is a tactic
    # field: it frees shared memory for a larger LUT or another AB stage.
    compact_pull_buffer: bool = False
    generate_c: Literal[False] = False
    mma_tiler_mnk: Tuple[int, int, int] = _NVFP4_MMA_TILER_DEFAULT
    # Local layout improvements are derived defaults, not tuning axes.
    # Only these two bounded performance strategies are optional.
    fc2_tail_n8: bool = field(default=False, kw_only=True)
    fc1_ready_mode: Literal["tile", "k256"] = field(default="tile", kw_only=True)
    # None selects the default FC1 K256 schedule; explicit booleans enable
    # or disable conversion/MMA overlap without changing the tactic schema.
    nvfp4_fc1_k256_overlap: Optional[bool] = field(default=None, kw_only=True)
    # K-permuted weight layout; owned by the weights, not by the tactic.
    nvfp4_kperm: int = field(default=64, kw_only=True)
    # Conversion-table replicas and lane map (tactic fields; values identical).
    nvfp4_lut_replicas: int = field(default=8, kw_only=True)
    nvfp4_lut_map: Literal["quad", "pair", "lane"] = field(default="quad", kw_only=True)
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
        validate_nvfp4_kperm(self.nvfp4_kperm)
        _ = self.nvfp4_conversion
        if self.kind != "fp8_e4m3":
            raise ValueError(
                f"Hopper NVFP4 x FP8 requires kind='fp8_e4m3'; got {self.kind!r}."
            )
        if self.fp8_scale_mode != "mxfp4_hybrid":
            raise ValueError(
                "Hopper NVFP4 x FP8 requires "
                "fp8_scale_mode='mxfp4_hybrid'; "
                f"got {self.fp8_scale_mode!r}."
            )
        if self.fp8_accum_mode != "1xacc":
            raise ValueError(
                "Hopper NVFP4 x FP8 supports only fp8_accum_mode='1xacc'; "
                f"got {self.fp8_accum_mode!r}."
            )
        if not isinstance(self.compact_pull_buffer, bool):
            raise ValueError("compact_pull_buffer must be a bool.")
        if self.generate_c is not False:
            raise ValueError("NVFP4 requires generate_c=False")
        if self.tail_split_pairs and self.fc1_ready_mode != "tile":
            raise ValueError("NVFP4 tail_split_pairs requires fc1_ready_mode='tile'")
        if not self.swap_ab:
            raise ValueError(
                "Hopper NVFP4 x FP8 supports only the swap-AB kernel; "
                "ordinary FP8/native-layout fallback is forbidden."
            )
        for field_name in (
            "swap_ab",
            "pingpong",
            "in_kernel_fc2_reduce",
            "dedup_dispatch",
            "grouped_token_back",
            "fc1_store_offload",
            "fc1_early_done_publish",
            "fold_producer_warps",
            "tail_split_pairs",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool.")
        if type(
            self.active_dispatch_warps
        ) is not int or self.active_dispatch_warps not in (1, 2, 4):
            raise ValueError("active_dispatch_warps must be 1, 2, or 4.")
        if self.fold_producer_warps and self.active_dispatch_warps != 1:
            raise ValueError(
                "fold_producer_warps=True requires active_dispatch_warps=1."
            )
        if self.grouped_token_back:
            raise ValueError(
                "the production NVFP4 BF16-combine profile does not support "
                "grouped_token_back."
            )
        if self.combine_format != "bf16":
            raise ValueError(
                "the production NVFP4 profile requires combine_format='bf16'."
            )

        # Reuse the target FP8 config's rank, shape, scheduler, cluster,
        # ping-pong, token-back, and flag validation.  The generic base does
        # not recognize the dedicated hybrid string, so validate an otherwise
        # identical blockwise proxy (the common divisibility requirements are
        # the same) and apply NVFP4-only constraints below.
        base_kwargs = {
            item.name: getattr(self, item.name)
            for item in dataclasses.fields(MegaMoEHopperFp8Config)
        }
        base_kwargs["fp8_scale_mode"] = "blockwise"
        MegaMoEHopperFp8Config(**base_kwargs)

        tile_m, _tile_n, tile_k = self.mma_tiler_mnk
        if tile_m not in (128, 256):
            raise ValueError(
                "Hopper NVFP4 x FP8 requires swap-AB MMA tile M=128 or 256; "
                f"got mma_tiler_mnk={self.mma_tiler_mnk}."
            )
        if tile_k not in (128, 256):
            raise ValueError(
                "Hopper NVFP4 x FP8 requires MMA tile K=128 or 256; "
                f"got mma_tiler_mnk={self.mma_tiler_mnk}."
            )
        for name, logical_k in (
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
        ):
            if logical_k % tile_k:
                raise ValueError(
                    f"Hopper NVFP4 x FP8 requires {name} ({logical_k}) "
                    f"divisible by MMA tile K={tile_k}."
                )
        validate_nvfp4_fc1_overlap_override(
            self.nvfp4_fc1_k256_overlap,
            mma_tiler_mnk=self.mma_tiler_mnk,
            fp8_scale_mode=self.fp8_scale_mode,
            pingpong=self.pingpong,
        )
        # Validate the final model/layout/protocol before any symmetric input
        # or kernel workspace is allocated.
        _ = self.optimizations

    @property
    def optimizations(self):
        folded = self.fold_producer_warps and self.active_dispatch_warps == 1
        return resolve_mxfp4_optimizations(
            fp8_scale_mode=self.fp8_scale_mode,
            mma_tiler_mnk=self.mma_tiler_mnk,
            cluster_shape_mnk=self.cluster_shape_mnk,
            static_expert_shape=(self.num_experts_per_rank, self.fc1_out, self.hidden),
            world_size=self.world_size,
            pingpong=self.pingpong,
            token_back_by_dispatch=self.resolved_token_back_mode != "epi_warps",
            fc2_in_kernel_topk_reduce=self.in_kernel_fc2_reduce,
            fc1_early_done_publish=folded or self.fc1_early_done_publish,
            fc1_store_offload=False if folded else self.fc1_store_offload,
            dedup_dispatch=self.dedup_dispatch,
            fc2_tail_n8=self.fc2_tail_n8,
            fc1_ready_mode=self.fc1_ready_mode,
        )

    @property
    def mxfp4_hybrid(self) -> bool:
        return True

    @property
    def nvfp4_conversion(self) -> Nvfp4ConversionLayout:
        return Nvfp4ConversionLayout(
            kperm=self.nvfp4_kperm,
            lut_replicas=self.nvfp4_lut_replicas,
            lut_map=self.nvfp4_lut_map,
        )


@dataclasses.dataclass
class MegaMoEHopperNvfp4Inputs(MegaMoEHopperFp8Inputs):
    """Per-rank tensors for one fused NVFP4 x FP8 launch.

    Field names intentionally match :class:`MegaMoEHopperFp8Inputs` because
    the composite kernel keeps the same thirteen-tensor runtime ABI.  Their
    dtypes/shapes have the NVFP4 meanings documented at module scope.
    """


class MegaMoEHopperNvfp4Frontend(MegaMoEHopperFp8Frontend):
    """Lazy-compile frontend for ``Sm90MegaMoESwapABNvfp4Fp8Kernel`` only."""

    _mega_key: Optional[tuple]
    _mega: Optional[_CompiledMega]

    def __init__(self, config: MegaMoEHopperNvfp4Config) -> None:
        super().__init__(config)
        self._nvfp4_actual_num_tokens: Optional[int] = None
        self._nvfp4_fc1_k256_overlap = False
        self._nvfp4_device_identity: Optional[tuple] = None
        self._nvfp4_workspace_owner: Optional[_CompiledMega] = None
        self._nvfp4_workspace_key: Optional[tuple] = None
        self._nvfp4_workspace_signature: Optional[tuple] = None
        self._nvfp4_schedule_cache: Dict[bool, _CompiledMega] = {}
        self._nvfp4_workspace_generation = 0

    def _select_nvfp4_fc1_schedule(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int],
        selection_tokens: Optional[int],
    ) -> None:
        launch_tokens = self._resolve_num_tokens(inputs, num_tokens)
        actual_tokens = launch_tokens if selection_tokens is None else selection_tokens
        if (
            isinstance(actual_tokens, bool)
            or not isinstance(actual_tokens, int)
            or not 0 <= actual_tokens <= launch_tokens
        ):
            raise ValueError(
                "selection_tokens must be an integer in [0, launch token count]"
            )
        c = self.config
        shape: Dict[str, Any] = dict(
            num_tokens=actual_tokens,
            hidden=c.hidden,
            intermediate=c.intermediate,
            num_total_experts=c.num_total_experts,
            num_topk=c.num_topk,
            world_size=c.world_size,
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            fp8_scale_mode=c.fp8_scale_mode,
            pingpong=c.pingpong,
        )
        device_name, sm_count = None, None
        if c.nvfp4_fc1_k256_overlap is None and nvfp4_fc1_auto_shape_eligible(**shape):
            device = inputs.activation.device
            identity = self._nvfp4_device_identity
            if identity is None or identity[0] != device:
                properties = torch.cuda.get_device_properties(device)
                identity = (device, properties.name, properties.multi_processor_count)
                self._nvfp4_device_identity = identity
            _, device_name, sm_count = identity
        enabled = resolve_nvfp4_fc1_overlap(
            override=c.nvfp4_fc1_k256_overlap,
            device_name=device_name,
            sm_count=sm_count,
            **shape,
        )
        workspace_key = self._nvfp4_workspace_compile_key()
        if (
            self._nvfp4_workspace_owner is not None
            and self._nvfp4_workspace_key != workspace_key
        ):
            # Actual config changes retain the established all-rank lifecycle.
            ensure_not_capturing("NVFP4 workspace config change")
            self._release_workspace()
        if enabled != self._nvfp4_fc1_k256_overlap:
            ensure_not_capturing("NVFP4 FC1 schedule change")
            self._nvfp4_fc1_k256_overlap = enabled
            # Ranks may cross the threshold independently. Never free/allocate
            # symmetric memory here; only select a rank-local compiled entry.
            self._mega = self._nvfp4_schedule_cache.get(enabled)
            self._mega_key = self._mega_compile_key() if self._mega else None
        self._nvfp4_actual_num_tokens = actual_tokens

    def prepare_launch(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        validated: Optional[tuple[MegaMoEHopperFp8Inputs, tuple]] = None,
        selection_tokens: Optional[int] = None,
    ) -> None:
        self._select_nvfp4_fc1_schedule(
            inputs, num_tokens=num_tokens, selection_tokens=selection_tokens
        )
        super().prepare_launch(inputs, num_tokens=num_tokens, validated=validated)

    def warmup(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        selection_tokens: Optional[int] = None,
    ) -> None:
        self.prepare_launch(
            inputs, num_tokens=num_tokens, selection_tokens=selection_tokens
        )

    def run(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = False,
        selection_tokens: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        self._select_nvfp4_fc1_schedule(
            inputs, num_tokens=num_tokens, selection_tokens=selection_tokens
        )
        return super().run(
            inputs, num_tokens=num_tokens, sync=sync, reset_counters=reset_counters
        )

    def make_launch_thunk(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        selection_tokens: Optional[int] = None,
    ) -> Callable[[], None]:
        self._select_nvfp4_fc1_schedule(
            inputs, num_tokens=num_tokens, selection_tokens=selection_tokens
        )
        thunk = super().make_launch_thunk(inputs, num_tokens=num_tokens)
        generation = self._nvfp4_workspace_generation

        def launch() -> None:
            if self._nvfp4_workspace_generation != generation:
                raise RuntimeError(
                    "NVFP4 launch thunk is stale; rebuild it after workspace release"
                )
            thunk()

        return launch

    def apply_knobs(self, knobs: Optional[dict]) -> None:
        """Apply only declared NVFP4 tactics; reject stale/cache-only fields."""
        if knobs is None:
            return
        validated = _validate_complete_nvfp4_tactic(
            knobs,
            source="NVFP4 frontend.apply_knobs()",
        )
        new_config = dataclasses.replace(self.config, **validated)
        if new_config == self.config:
            return
        ensure_not_capturing("NVFP4 apply_knobs (config change)")
        self._release_workspace()
        self._config = new_config

    def effective_tactic(self) -> Dict[str, Any]:
        """Return the complete tactic of the current compiled kernel.

        The kernel may normalize warp-layout knobs while it establishes a
        register-feasible schedule. Cache and benchmark identities therefore
        have to use these effective values, not merely the requested config.
        """

        mega = self._mega
        if (
            mega is None
            or mega.compiled is None
            or self._mega_key != self._mega_compile_key()
        ):
            raise RuntimeError(
                "NVFP4 effective tactic requires the current frontend config "
                "to have an actual compiled kernel"
            )
        kernel = mega.kernel
        c = self.config
        return _validate_complete_nvfp4_tactic(
            {
                "swap_ab": c.swap_ab,
                "pingpong": c.pingpong,
                "mma_tiler_mnk": tuple(c.mma_tiler_mnk),
                "cluster_shape_mnk": tuple(c.cluster_shape_mnk),
                "fp8_accum_mode": c.fp8_accum_mode,
                "load_balance_mode": c.load_balance_mode,
                "token_back_mode": c.resolved_token_back_mode,
                "in_kernel_fc2_reduce": c.in_kernel_fc2_reduce,
                "group_hint": kernel.group_hint,
                "num_sched_stages": kernel.num_sched_stages,
                "dedup_dispatch": bool(kernel.dedup_dispatch),
                "grouped_token_back": bool(kernel.grouped_token_back),
                "combine_format": str(kernel.combine_format),
                "active_dispatch_warps": int(kernel.token_comm.active_dispatch_warps),
                "fc1_store_offload": bool(kernel.fc1_store_offload),
                "fc1_early_done_publish": bool(kernel.fc1_early_done_publish),
                "fold_producer_warps": bool(kernel.fold_producer_warps),
                "tail_split_pairs": bool(kernel.tail_split_pairs),
                "fc2_tail_n8": kernel.mxfp4_optimizations.fc2_tail_n8,
                "fc1_ready_mode": kernel.mxfp4_optimizations.fc1_ready_mode,
                "compact_pull_buffer": bool(kernel.token_comm.compact_pull_buffer),
                "nvfp4_lut_replicas": kernel.nvfp4_conversion.lut_replicas,
                "nvfp4_lut_map": kernel.nvfp4_conversion.lut_map,
            },
            source="NVFP4 frontend effective tactic",
        )

    @property
    def config(self) -> MegaMoEHopperNvfp4Config:
        config = super().config
        if not isinstance(config, MegaMoEHopperNvfp4Config):
            raise RuntimeError("NVFP4 frontend lost its format-specific config.")
        return config

    def _mega_compile_key(self) -> tuple:
        return self._nvfp4_workspace_compile_key() + (
            ("nvfp4_fc1_k256_overlap", self._nvfp4_fc1_k256_overlap),
        )

    def _nvfp4_workspace_compile_key(self) -> tuple:
        c = self.config
        return (
            "sm90_nvfp4_fp8_megamoe",
            *_NVFP4_COMPILE_IDENTITY,
            ("optimization_policy", c.optimizations.identity()),
            # Make logical-K versus packed storage-K explicit.  Shape fields
            # later in the inherited key would distinguish these in practice,
            # but naming both protects against future generic packed formats.
            ("fc1_logical_storage_k", c.hidden, c.hidden // 2),
            ("fc2_logical_storage_k", c.intermediate, c.intermediate // 2),
            ("nvfp4_conversion", dataclasses.astuple(c.nvfp4_conversion)),
            ("compact_pull_buffer", c.compact_pull_buffer),
            *super()._mega_compile_key(),
        )

    @staticmethod
    def _nvfp4_workspace_layout_signature(kernel) -> tuple:
        def regions(specs):
            return tuple(
                (r.name, r.cute_dtype, tuple(r.shape), r.align, r.nbytes) for r in specs
            )

        geometry = tuple(
            (name, getattr(kernel, name, None))
            for name in (
                "world_size",
                "num_topk",
                "max_tokens_per_rank",
                "hidden",
                "group_hint",
                "num_sched_stages",
                "num_ab_stage",
                "threads_per_cta",
                "wgmma_tile_m",
                "wgmma_n_splits",
            )
        )
        return (
            tuple(kernel.get_workspace_sizes()),
            regions(kernel._local_region_specs),
            regions(kernel._shared_region_specs),
            tuple(sorted(kernel._local_offsets.items())),
            tuple(sorted(kernel._shared_offsets.items())),
            geometry,
        )

    def _get_nvfp4_workspace(self, kernel) -> _CompiledMega:
        """Allocate once collectively; FC1 schedules share the same live regions."""
        key = self._nvfp4_workspace_compile_key()
        signature = self._nvfp4_workspace_layout_signature(kernel)
        owner = self._nvfp4_workspace_owner
        if owner is not None:
            if (
                key != self._nvfp4_workspace_key
                or signature != self._nvfp4_workspace_signature
            ):
                raise RuntimeError("NVFP4 FC1 overlap workspace layout mismatch")
            return _CompiledMega(
                compiled=None,
                kernel=kernel,
                local_workspace=owner.local_workspace,
                shared_workspace=owner.shared_workspace,
                symmetric_base=owner.symmetric_base,
                peer_offsets_list=owner.peer_offsets_list,
            )

        ensure_not_capturing("NVFP4 symmetric workspace allocation")
        local_ws_bytes, shared_ws_bytes = kernel.get_workspace_sizes()
        local_workspace = torch.zeros(
            (local_ws_bytes,), dtype=torch.uint8, device="cuda"
        )
        shared_workspace = sym_zeros((shared_ws_bytes,), torch.uint8)
        owner = _CompiledMega(
            compiled=None,
            kernel=kernel,
            local_workspace=local_workspace,
            shared_workspace=shared_workspace,
            symmetric_base=0,
            peer_offsets_list=(),
        )
        # Publish before peer setup/compile so coordinated failure cleanup owns
        # the allocation even when no callable was produced.
        self._nvfp4_workspace_owner = owner
        self._nvfp4_workspace_key = key
        self._nvfp4_workspace_signature = signature
        self._mega = owner
        self._mega_key = None
        symmetric_base, peer_offsets = _compute_peer_offsets(
            shared_workspace, self.config.world_size
        )
        owner.symmetric_base = symmetric_base
        owner.peer_offsets_list = tuple(peer_offsets)
        return owner

    def _release_workspace(self) -> None:
        owner = self._nvfp4_workspace_owner
        if owner is not None:
            ensure_not_capturing("NVFP4 workspace release (symmetric-heap free)")
            free_sym_tensor(owner.shared_workspace)
        self._nvfp4_schedule_cache.clear()
        self._nvfp4_workspace_owner = None
        self._nvfp4_workspace_key = None
        self._nvfp4_workspace_signature = None
        self._nvfp4_workspace_generation += 1
        self._invalidate_compile_cache()

    @staticmethod
    def _assert_mirrored_constants() -> None:
        """Fail closed if the clean vendor drop changes the Humming layout."""
        MegaMoEHopperFp8Frontend._assert_mirrored_constants()

        from .nvfp4_conversion import (
            MXFP4_FOLD_M as NVFP4_FOLD_M,
            MXFP4_GROUP_SIZE as NVFP4_GROUP_SIZE,
            MXFP4_K_TILE as NVFP4_K_TILE,
        )

        mirrored = (
            ("NVFP4_GROUP_SIZE", NVFP4_GROUP_SIZE, _NVFP4_GROUP_SIZE),
            ("NVFP4_FOLD_M", NVFP4_FOLD_M, _NVFP4_FOLD_M),
            ("NVFP4_K_TILE", NVFP4_K_TILE, _NVFP4_FOLD_K),
        )
        for name, source_value, expected in mirrored:
            if source_value != expected:
                raise RuntimeError(
                    f"kernel drop changed {name} ({source_value} != {expected}); "
                    "bump the NVFP4 compile identity and re-audit preprocessing, "
                    "folded-offset shapes, and launch validation."
                )

    def _ensure_mega_compiled(self, inputs: MegaMoEHopperFp8Inputs) -> _CompiledMega:
        if not isinstance(inputs, MegaMoEHopperNvfp4Inputs):
            raise TypeError("NVFP4 frontend requires MegaMoEHopperNvfp4Inputs.")
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        ensure_not_capturing("NVFP4 cute.compile")
        workspace_key = self._nvfp4_workspace_compile_key()
        if (
            self._nvfp4_workspace_owner is not None
            and self._nvfp4_workspace_key != workspace_key
        ):
            self._release_workspace()
        cached = self._nvfp4_schedule_cache.get(self._nvfp4_fc1_k256_overlap)
        if cached is not None:
            self._mega, self._mega_key = cached, key
            return cached

        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils

        from common.megamoe_constants import Fp8E8M0SfVecSize, SfPaddingBlock
        from .nvfp4_kernels import Sm90MegaMoESwapABNvfp4Fp8Kernel

        self._assert_mirrored_constants()
        c = self.config
        if not c.swap_ab:
            # Redundant with config validation by design: never let a mutated
            # object silently select the ordinary FP8/native class.
            raise RuntimeError("NVFP4 frontend requires swap_ab=True.")

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
        kernel = Sm90MegaMoESwapABNvfp4Fp8Kernel(
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
            dedup_dispatch=c.dedup_dispatch,
            grouped_token_back=c.grouped_token_back,
            combine_format=c.combine_format,
            active_dispatch_warps=c.active_dispatch_warps,
            compact_pull_buffer=c.compact_pull_buffer,
            fc1_store_offload=c.fc1_store_offload,
            fc1_early_done_publish=c.fc1_early_done_publish,
            fold_producer_warps=c.fold_producer_warps,
            generate_c=False,
            mxfp4_fc2_tail_n8=c.fc2_tail_n8,
            mxfp4_fc1_ready_mode=c.fc1_ready_mode,
            nvfp4_fc1_k256_overlap=self._nvfp4_fc1_k256_overlap,
            nvfp4_conversion=c.nvfp4_conversion,
            tail_split_pairs=c.tail_split_pairs,
        )

        mega = self._get_nvfp4_workspace(kernel)
        self._mega = mega
        self._mega_key = None
        compile_kwargs = self._build_mega_runtime_kwargs(inputs, mega)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if c.enable_iket:
            compile_kwargs["options"] = "iket"
        mega.compiled = cute.compile(kernel, **compile_kwargs)
        self._nvfp4_schedule_cache[self._nvfp4_fc1_k256_overlap] = mega
        self._mega_key = key
        return mega

    @staticmethod
    def _slice_inputs(
        inputs: MegaMoEHopperFp8Inputs,
        num_tokens: int,
    ) -> MegaMoEHopperNvfp4Inputs:
        if not isinstance(inputs, MegaMoEHopperNvfp4Inputs):
            raise TypeError("NVFP4 frontend requires MegaMoEHopperNvfp4Inputs.")
        tok = slice(None, num_tokens)
        return MegaMoEHopperNvfp4Inputs(
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
        if not isinstance(inputs, MegaMoEHopperNvfp4Inputs):
            raise TypeError("NVFP4 frontend requires MegaMoEHopperNvfp4Inputs.")
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
                "NVFP4 MegaMoE activation must be torch.float8_e4m3fn, "
                f"got {inputs.activation.dtype}."
            )
        if inputs.activation.stride(-1) != 1:
            raise ValueError("activation must have contiguous hidden dimension.")

        require_cuda("activation_sf", inputs.activation_sf)
        if tuple(inputs.activation_sf.shape) != (buf_tokens, 4):
            raise ValueError(
                "NVFP4 hybrid activation_sf must have exact shape "
                f"({buf_tokens}, 4), got {tuple(inputs.activation_sf.shape)}."
            )
        if inputs.activation_sf.dtype != torch.float32:
            raise ValueError(
                "NVFP4 hybrid activation_sf must be float32, "
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
                (e, c.fc1_out // 64, c.hidden // 128, 32, 16),
            ),
            (
                "fc2_weight_sf",
                inputs.fc2_weight_sf,
                (e, c.hidden // 64, c.intermediate // 128, 32, 16),
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


# Kernel-ready leg: packed payload, transposed E4M3 scale pairs, static unit
# activation placeholder, and per-expert global scale * 8.
TransformedNvfp4Weights = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass
class MegaMoEHopperNvfp4SymmBuffer:
    """Symmetric staging/workspace owner for one fused NVFP4 session."""

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

    _frontend: MegaMoEHopperNvfp4Frontend
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


def _default_nvfp4_profile(
    *,
    num_max_tokens: int,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_total_experts: int,
    num_topk: int,
    gate_up_clamp: Optional[float],
    routing_profile: str,
    nvfp4_kperm: int,
):
    """Use calibrated defaults only for their exact model and device domain."""
    from .nvfp4_tuner import hopper_nvfp4_default_profile

    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return hopper_nvfp4_default_profile(
        num_max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_total_experts,
        world_size=world_size,
        num_topk=num_topk,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
        kperm=nvfp4_kperm,
        device_name=properties.name,
        sm_count=int(properties.multi_processor_count),
    )


def resolve_hopper_nvfp4_knobs(
    knobs: Optional[Any],
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_total_experts: int,
    num_topk: int,
    num_max_tokens: int,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    nvfp4_kperm: int = 64,
    nvfp4_fc1_k256_overlap: Optional[bool] = None,
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    use_tuning_cache: bool = False,
) -> Dict[str, Any]:
    """Resolve initial geometry; NV cache reads require explicit opt-in."""
    if type(use_tuning_cache) is not bool:
        raise ValueError("use_tuning_cache must be a bool")
    routing_profile = normalize_sm90_routing_profile(routing_profile)
    if isinstance(knobs, dict):
        return _validate_complete_nvfp4_tactic(
            knobs,
            source="explicit NVFP4 knobs",
        )
    if knobs == "auto":
        raise ValueError(
            "direct NVFP4 resolution cannot execute collective autotune; "
            "pass knobs='auto' through Sm90PullNvfp4MegaKernelBackend"
        )
    if knobs is not None:
        raise ValueError("knobs must be None or an NVFP4 knob dict.")

    profile = _default_nvfp4_profile(
        num_max_tokens=num_max_tokens,
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_total_experts=num_total_experts,
        num_topk=num_topk,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
        nvfp4_kperm=nvfp4_kperm,
    )
    effective_overlap = nvfp4_fc1_k256_overlap
    if effective_overlap is None and profile is not None:
        effective_overlap = profile["requested_config"]["nvfp4_fc1_k256_overlap"]

    if use_tuning_cache:
        # Only a live NV autotune winner can populate this separate namespace.
        from .knob_cache import lookup_knobs
        from .nvfp4_tuner import (
            hopper_nvfp4_cache_provenance_sha256,
            hopper_nvfp4_candidates,
            filter_hopper_nvfp4_candidates,
        )

        if num_tokens is not None and (
            type(num_tokens) is not int or not 0 <= num_tokens <= num_max_tokens
        ):
            raise ValueError("NVFP4 cache live tokens must be within capacity")
        cached = lookup_knobs(
            dtype=_NVFP4_TUNING_DTYPE_ID,
            fp8_scale_mode="mxfp4_hybrid",
            world_size=world_size,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_total_experts,
            topk=num_topk,
            max_tokens=num_max_tokens,
            live_tokens=num_tokens,
            gate_up_clamp=gate_up_clamp,
            routing_profile=routing_profile,
            tuning_provenance_sha256=hopper_nvfp4_cache_provenance_sha256(
                routing_profile=routing_profile,
                kperm=nvfp4_kperm,
                overlap=effective_overlap,
                max_tokens=num_max_tokens,
                hidden=hidden,
                intermediate=intermediate,
                num_experts=num_total_experts,
                world_size=world_size,
            ),
        )
        if cached is not None:
            try:
                cached = _validate_complete_nvfp4_tactic(
                    cached, source="NVFP4 tuning-cache entry"
                )
            except (TypeError, ValueError):
                cached = None
            if cached is not None:
                properties = (
                    torch.cuda.get_device_properties(torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                )
                allowed = hopper_nvfp4_candidates(
                    num_max_tokens,
                    hidden=hidden,
                    intermediate=intermediate,
                    num_experts=num_total_experts,
                    world_size=world_size,
                    routing_profile=routing_profile,
                    kperm=nvfp4_kperm,
                )
                allowed = filter_hopper_nvfp4_candidates(
                    allowed,
                    max_tokens=num_max_tokens,
                    live_tokens=num_tokens,
                    hidden=hidden,
                    intermediate=intermediate,
                    num_experts=num_total_experts,
                    world_size=world_size,
                    num_topk=num_topk,
                    gate_up_clamp=gate_up_clamp,
                    routing_profile=routing_profile,
                    kperm=nvfp4_kperm,
                    overlap=effective_overlap,
                    device_name=properties.name if properties else None,
                    sm_count=int(properties.multi_processor_count)
                    if properties
                    else None,
                )
                if cached in allowed:
                    return cached
            # Unknown live context, a neighboring bucket or an unsupported profile
            # is a cache miss; its timing does not apply to this request.

    # Default resolution is independent of timing caches. The H200 table
    # applies only to its thirteen exact capacity buckets and model domain.
    if profile is not None:
        return _validate_complete_nvfp4_tactic(
            profile["requested_tactic"], source="NVFP4 tuned default"
        )
    from .mxfp4_tuner import _ordered_base_candidates

    # Prefer the bucket winner when legal; smaller shapes may require the
    # first legal tactic in the stable manifest union instead.
    return _validate_complete_nvfp4_tactic(
        _ordered_base_candidates(
            num_max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            routing_profile=routing_profile,
        )[0],
        source="NVFP4 initial geometry",
    )


def _resolve_hopper_nvfp4_mega_moe_config(
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
    load_balance_mode: Optional[Literal["static", "atomic_counter"]] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None,
    dedup_dispatch: Optional[bool] = None,
    grouped_token_back: bool = False,
    combine_format: str = "bf16",
    active_dispatch_warps: Optional[int] = None,
    fc1_store_offload: Optional[bool] = None,
    fc1_early_done_publish: Optional[bool] = None,
    fold_producer_warps: Optional[bool] = None,
    apply_topk_in_fc1: bool = True,
    group_hint: Optional[int] = None,
    clc_bundle_size: Optional[int] = None,
    num_sched_stages: Optional[int] = None,
    flag_batch: int = 1,
    epi_flag_batch: Tuple[int, int] = (2, 4),
    nvfp4_kperm: int = 64,
    nvfp4_fc1_k256_overlap: Optional[bool] = None,
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
) -> MegaMoEHopperNvfp4Config:
    """Resolve the complete NVFP4 constructor request without allocation."""

    routing_profile = normalize_sm90_routing_profile(routing_profile)
    manual_geometry = any(
        value is not None
        for value in (
            swap_ab,
            pingpong,
            mma_tiler_mnk,
            cluster_shape_mnk,
            group_hint,
            num_sched_stages,
        )
    )
    manual_execution = any(
        value is not None
        for value in (
            load_balance_mode,
            dedup_dispatch,
            active_dispatch_warps,
            fc1_store_offload,
            fc1_early_done_publish,
            fold_producer_warps,
        )
    )
    manual_selector = manual_geometry or manual_execution
    if manual_selector and knobs is not None:
        raise ValueError(
            "pass either explicit fused geometry/execution axes or knobs=, not both."
        )

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if manual_selector:
        # Every explicit geometry/execution selector is completely local: do
        # not consult a cache and accidentally inherit the remaining choices
        # from a tuned session. Omitted fields use the manual-mode
        # defaults below.
        resolved: Dict[str, Any] = dict(
            swap_ab=True if swap_ab is None else swap_ab,
            pingpong=False if pingpong is None else pingpong,
            mma_tiler_mnk=(
                _NVFP4_MMA_TILER_DEFAULT
                if mma_tiler_mnk is None
                else tuple(mma_tiler_mnk)
            ),
            cluster_shape_mnk=(
                _NVFP4_CLUSTER_SHAPE_DEFAULT
                if cluster_shape_mnk is None
                else tuple(cluster_shape_mnk)
            ),
        )
    else:
        resolved = resolve_hopper_nvfp4_knobs(
            knobs,
            world_size=world_size,
            hidden=hidden,
            intermediate=intermediate,
            num_total_experts=num_total_experts,
            num_topk=num_topk,
            num_max_tokens=num_max_tokens,
            gate_up_clamp=clamp,
            routing_profile=routing_profile,
            nvfp4_kperm=nvfp4_kperm,
            nvfp4_fc1_k256_overlap=nvfp4_fc1_k256_overlap,
        )
        # Match the established FP8 shim's precedence rule: an explicit
        # caller token-back choice wins over a cache entry.
        if token_back_mode is not None:
            resolved.pop("token_back_mode", None)

    config = MegaMoEHopperNvfp4Config(
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
        mma_tiler_mnk=_NVFP4_MMA_TILER_DEFAULT,
        cluster_shape_mnk=_NVFP4_CLUSTER_SHAPE_DEFAULT,
        load_balance_mode=(
            "static" if load_balance_mode is None else load_balance_mode
        ),
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        dedup_dispatch=False if dedup_dispatch is None else dedup_dispatch,
        grouped_token_back=grouped_token_back,
        combine_format=combine_format,
        active_dispatch_warps=(
            1 if active_dispatch_warps is None else active_dispatch_warps
        ),
        fc1_store_offload=(True if fc1_store_offload is None else fc1_store_offload),
        fc1_early_done_publish=(
            False if fc1_early_done_publish is None else fc1_early_done_publish
        ),
        fold_producer_warps=(
            active_dispatch_warps in (None, 1)
            if fold_producer_warps is None
            else fold_producer_warps
        ),
        apply_topk_in_fc1=apply_topk_in_fc1,
        group_hint=group_hint,
        clc_bundle_size=clc_bundle_size,
        num_sched_stages=num_sched_stages,
        flag_batch=flag_batch,
        epi_flag_batch=epi_flag_batch,
        nvfp4_kperm=nvfp4_kperm,
        routing_profile=routing_profile,
    )
    if resolved:
        # External knob/cache/heuristic dicts were already validated as a
        # complete tactic (legacy 17/19-field inputs reset omitted strategies).
        # Manual selection uses an internal geometry
        # mapping assembled above; the concrete config performs value
        # validation without pretending it is a complete external tactic.
        config = dataclasses.replace(config, **resolved)
        if not isinstance(config, MegaMoEHopperNvfp4Config):
            raise RuntimeError("NVFP4 knob application changed config type.")
    if not manual_selector and knobs is None and nvfp4_fc1_k256_overlap is None:
        profile = _default_nvfp4_profile(
            num_max_tokens=num_max_tokens,
            world_size=world_size,
            hidden=hidden,
            intermediate=intermediate,
            num_total_experts=num_total_experts,
            num_topk=num_topk,
            gate_up_clamp=clamp,
            routing_profile=routing_profile,
            nvfp4_kperm=nvfp4_kperm,
        )
        if profile is not None:
            nvfp4_fc1_k256_overlap = profile["requested_config"][
                "nvfp4_fc1_k256_overlap"
            ]
    if nvfp4_fc1_k256_overlap is not None:
        # The override is checked against the final geometry: a forced K256
        # schedule with a knob tactic must not meet the K128 default above.
        config = dataclasses.replace(
            config, nvfp4_fc1_k256_overlap=nvfp4_fc1_k256_overlap
        )
    return config


def get_symm_buffer_for_hopper_nvfp4_mega_moe(
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
    load_balance_mode: Optional[Literal["static", "atomic_counter"]] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None,
    dedup_dispatch: Optional[bool] = None,
    grouped_token_back: bool = False,
    combine_format: str = "bf16",
    active_dispatch_warps: Optional[int] = None,
    fc1_store_offload: Optional[bool] = None,
    fc1_early_done_publish: Optional[bool] = None,
    fold_producer_warps: Optional[bool] = None,
    apply_topk_in_fc1: bool = True,
    group_hint: Optional[int] = None,
    clc_bundle_size: Optional[int] = None,
    num_sched_stages: Optional[int] = None,
    flag_batch: int = 1,
    epi_flag_batch: Tuple[int, int] = (2, 4),
    nvfp4_kperm: int = 64,
    nvfp4_fc1_k256_overlap: Optional[bool] = None,
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
) -> MegaMoEHopperNvfp4SymmBuffer:
    """Allocate symmetric inputs and a fused NVFP4 frontend.

    ``knobs=None`` performs a lookup using a dedicated, versioned NVFP4
    fused identity, then uses the manifest-derived per-token heuristic on a
    cache miss. Supplying any fused geometry or execution axis instead uses
    cache-free manual defaults. Collective ``knobs="auto"`` is owned by the production
    backend because it requires live weights, staged routing, and every EP
    rank in lockstep. Explicit geometry/execution axes and ``knobs=`` are
    mutually exclusive. An explicit token-back mode retains its established
    precedence over a cached tactic. ``nvfp4_kperm`` must equal the value the
    transformed weights were prepared with; FC1 activations must be staged in
    the same permuted hidden order.
    ``nvfp4_fc1_k256_overlap`` is an independent schedule override, allowed
    with a complete ``knobs`` tactic. None preserves automatic selection;
    True requires hybrid K256 without pingpong and False always disables it.
    """
    config = _resolve_hopper_nvfp4_mega_moe_config(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        kind=kind,
        fp8_scale_mode=fp8_scale_mode,
        fp8_accum_mode=fp8_accum_mode,
        knobs=knobs,
        swap_ab=swap_ab,
        pingpong=pingpong,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        load_balance_mode=load_balance_mode,
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        dedup_dispatch=dedup_dispatch,
        grouped_token_back=grouped_token_back,
        combine_format=combine_format,
        active_dispatch_warps=active_dispatch_warps,
        fc1_store_offload=fc1_store_offload,
        fc1_early_done_publish=fc1_early_done_publish,
        fold_producer_warps=fold_producer_warps,
        apply_topk_in_fc1=apply_topk_in_fc1,
        group_hint=group_hint,
        clc_bundle_size=clc_bundle_size,
        num_sched_stages=num_sched_stages,
        flag_batch=flag_batch,
        epi_flag_batch=epi_flag_batch,
        nvfp4_kperm=nvfp4_kperm,
        nvfp4_fc1_k256_overlap=nvfp4_fc1_k256_overlap,
        routing_profile=routing_profile,
    )
    return _get_symm_buffer_for_hopper_nvfp4_mega_moe_from_resolved_config(config)


def _get_symm_buffer_for_hopper_nvfp4_mega_moe_from_resolved_config(
    config: MegaMoEHopperNvfp4Config,
) -> MegaMoEHopperNvfp4SymmBuffer:
    """Allocate a fused workspace from one already-resolved immutable config."""
    frontend = MegaMoEHopperNvfp4Frontend(config)
    num_total_experts = config.num_total_experts
    num_max_tokens = config.num_tokens_per_rank
    num_topk = config.num_topk
    hidden = config.hidden
    intermediate = config.intermediate
    rank = config.rank
    world_size = config.world_size
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

    return MegaMoEHopperNvfp4SymmBuffer(
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


def _resolve_transformed_nvfp4_weights(
    transformed: TransformedNvfp4Weights,
    leg: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        weight, folded_offset, activation_placeholder, global_x8 = transformed
    except (TypeError, ValueError):
        raise ValueError(
            f"transformed_{leg} must be a strict four-slot tuple "
            "(packed_weight, folded_offset, unit_activation_placeholder, "
            "global_x8)."
        ) from None
    if any(
        item is None
        for item in (weight, folded_offset, activation_placeholder, global_x8)
    ):
        raise ValueError(
            f"transformed_{leg} cannot contain None; NVFP4 metadata does not "
            "use ordinary-FP8 unit-scale substitution."
        )
    return weight, folded_offset, activation_placeholder, global_x8


class _Nvfp4InputBuffer(Protocol):
    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor


def _build_nvfp4_inputs(
    symm_buffer: _Nvfp4InputBuffer,
    transformed_l1: TransformedNvfp4Weights,
    transformed_l2: TransformedNvfp4Weights,
) -> MegaMoEHopperNvfp4Inputs:
    (
        fc1_weight,
        fc1_weight_sf,
        fc1_activation_dequant_scale,
        fc1_weight_dequant_scale,
    ) = _resolve_transformed_nvfp4_weights(transformed_l1, "l1")
    (
        fc2_weight,
        fc2_weight_sf,
        fc2_activation_dequant_scale,
        fc2_weight_dequant_scale,
    ) = _resolve_transformed_nvfp4_weights(transformed_l2, "l2")
    return MegaMoEHopperNvfp4Inputs(
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


def hopper_nvfp4_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedNvfp4Weights,
    transformed_l2: TransformedNvfp4Weights,
    symm_buffer: MegaMoEHopperNvfp4SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch fused dispatch + NVFP4 FC1/FC2 + combine with no fallback."""
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM90 NVFP4 path.",
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

    inputs = _build_nvfp4_inputs(symm_buffer, transformed_l1, transformed_l2)
    # Launch the full padded workspace.  topk_idx[n:] == -1 masks tail rows;
    # this matches the kernel-team fused driver and keeps compile shape stable.
    out = symm_buffer._frontend.run(
        inputs, num_tokens=None, sync=False, selection_tokens=n
    )
    if y is None:
        result = out[:n] if out is not None else symm_buffer.output_activation[:0]
    else:
        result = None
        if out is not None:
            y.copy_(out[:n])
    if sync and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
    return result


def hopper_nvfp4_mega_launch_thunk(
    transformed_l1: TransformedNvfp4Weights,
    transformed_l2: TransformedNvfp4Weights,
    symm_buffer: MegaMoEHopperNvfp4SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
) -> Callable[[], None]:
    """Build a padded launch thunk; num_tokens selects the FC1 schedule.

    Rebuild the thunk after changing its live token count or frontend config.
    Omitting num_tokens retains the full-capacity launch convention.
    """
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if not 0 <= n <= symm_buffer.num_max_tokens:
        raise ValueError("num_tokens is outside the symmetric buffer capacity")
    inputs = _build_nvfp4_inputs(symm_buffer, transformed_l1, transformed_l2)
    return symm_buffer._frontend.make_launch_thunk(inputs, selection_tokens=n)


__all__ = [
    "MegaMoEHopperNvfp4Config",
    "MegaMoEHopperNvfp4Frontend",
    "MegaMoEHopperNvfp4Inputs",
    "MegaMoEHopperNvfp4SymmBuffer",
    "TransformedNvfp4Weights",
    "get_symm_buffer_for_hopper_nvfp4_mega_moe",
    "hopper_nvfp4_mega_launch_thunk",
    "hopper_nvfp4_mega_moe",
]
