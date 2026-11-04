# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM90 NVFP4-weight x FP8-activation fused MegaMoE frontend.

This module is deliberately separate from :mod:`.hopper_fp8`: ordinary FP8
and NVFP4 have different weight, scale, and cache identities even
though they share the same communication/session lifecycle and launch ABI.
The implementation extends the MXFP4 frontend's common validation, compilation,
and activation staging, retaining NVFP4 cache identities and overlap-workspace
ownership while selecting only
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
from typing import Any, Callable, ClassVar, Dict, Literal, Optional, Protocol, Tuple

import torch

from flashinfer.moe_ep._compat import deprecated_aliases
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
from .hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
    MegaMoEHopperMxfp4Inputs,
    MegaMoEHopperMxfp4SymmBuffer,
    _allocate_fp4_symm_buffer,
    _build_fp4_inputs,
    _Mxfp4InputBuffer,
)
from .hopper_fp8 import (
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
)
from .nvfp4_policy import (
    NVFP4_FC1_SCHEDULE_VERSION,
    Nvfp4ConversionLayout,
    nvfp4_fc1_auto_shape_eligible,
    resolve_nvfp4_fc1_overlap,
    validate_nvfp4_fc1_overlap_override,
    validate_nvfp4_kperm,
)


_NVFP4_MMA_TILER_DEFAULT = (128, 32, 128)
_NVFP4_CLUSTER_SHAPE_DEFAULT = (1, 1, 1)

_NVFP4_GROUP_SIZE = 16
_NVFP4_FOLD_M = 64
_NVFP4_FOLD_K = 128

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
    (``compact_pull_buffer``, ``nvfp4_lut_replicas``, ``nvfp4_lut_map``,
    ``nvfp4_sf_stages``); a tactic without them selects the full pull buffer,
    8-replica lane-quad table and lockstep scale slots.
    """

    if not isinstance(knobs, dict):
        raise ValueError(f"{source} must provide an NVFP4 knob dict.")
    from .nvfp4_tuner import normalize_hopper_nvfp4_tactic

    try:
        return normalize_hopper_nvfp4_tactic(knobs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {exc}") from exc


@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
@dataclasses.dataclass(frozen=True)
class MegaMoEHopperNvfp4Config(MegaMoEHopperMxfp4Config):
    """Compile/launch configuration for fused NVFP4 x FP8.

    The inherited communication and scheduling knobs remain available, but
    the numerical format is fixed: E4M3 activation, packed E2M1/K16 weights,
    hybrid per-token activation scaling, one FP32 accumulator, and swap-AB.
    """

    # None selects the default FC1 K256 schedule; explicit booleans enable
    # or disable conversion/MMA overlap without changing the tactic schema.
    nvfp4_fc1_k256_overlap: Optional[bool] = field(default=None, kw_only=True)
    # K-permuted weight layout; owned by the weights, not by the tactic.
    nvfp4_kperm: int = field(default=64, kw_only=True)
    # Conversion-table replicas and lane map (tactic fields; values identical).
    nvfp4_lut_replicas: int = field(default=8, kw_only=True)
    nvfp4_lut_map: Literal["quad", "pair", "lane"] = field(default="quad", kw_only=True)
    # Zero preserves lockstep weight scales; one or two use a separate ring.
    nvfp4_sf_stages: int = field(default=0, kw_only=True)
    _weight_format = "NVFP4"

    def _validate_conversion_layout(self) -> None:
        validate_nvfp4_kperm(self.nvfp4_kperm)
        _ = self.nvfp4_conversion

    def _validate_active_dispatch_warps(self) -> None:
        if type(
            self.active_dispatch_warps
        ) is not int or self.active_dispatch_warps not in (1, 2, 4):
            raise ValueError("active_dispatch_warps must be 1, 2, or 4.")

    def _validate_weight_schedule(self) -> None:
        if self.nvfp4_sf_stages and (self.mma_tiler_mnk[2] != 256 or self.pingpong):
            raise ValueError("nvfp4_sf_stages requires MMA tile K=256 without pingpong")
        validate_nvfp4_fc1_overlap_override(
            self.nvfp4_fc1_k256_overlap,
            mma_tiler_mnk=self.mma_tiler_mnk,
            fp8_scale_mode=self.fp8_scale_mode,
            pingpong=self.pingpong,
        )

    @property
    def nvfp4_conversion(self) -> Nvfp4ConversionLayout:
        return Nvfp4ConversionLayout(
            kperm=self.nvfp4_kperm,
            lut_replicas=self.nvfp4_lut_replicas,
            lut_map=self.nvfp4_lut_map,
            sf_stages=self.nvfp4_sf_stages,
        )


@dataclasses.dataclass
class MegaMoEHopperNvfp4Inputs(MegaMoEHopperMxfp4Inputs):
    """Per-rank tensors for one fused NVFP4 x FP8 launch.

    Field names intentionally match :class:`MegaMoEHopperFp8Inputs` because
    the composite kernel keeps the same thirteen-tensor runtime ABI.  Their
    dtypes/shapes have the NVFP4 meanings documented at module scope.
    """

    _weight_format: ClassVar[str] = "NVFP4"


class MegaMoEHopperNvfp4Frontend(MegaMoEHopperMxfp4Frontend):
    """Lazy-compile frontend for ``Sm90MegaMoESwapABNvfp4Fp8Kernel`` only."""

    _weight_format = "NVFP4"
    _config_type = MegaMoEHopperNvfp4Config
    _inputs_type = MegaMoEHopperNvfp4Inputs
    _folded_scale_rows = 32
    _validate_tactic = staticmethod(_validate_complete_nvfp4_tactic)

    def _conversion_tactic(self, kernel) -> Dict[str, Any]:
        return {
            "compact_pull_buffer": bool(kernel.token_comm.compact_pull_buffer),
            "nvfp4_lut_replicas": kernel.nvfp4_conversion.lut_replicas,
            "nvfp4_lut_map": kernel.nvfp4_conversion.lut_map,
            "nvfp4_sf_stages": kernel.nvfp4_conversion.sf_stages,
        }

    _mega_key: Optional[tuple]
    _mega: Optional[_CompiledMega]

    def __init__(self, config: MegaMoEHopperNvfp4Config) -> None:
        super().__init__(config)
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
            compiled = self._nvfp4_schedule_cache.get(enabled)
            if compiled is None:
                ensure_not_capturing("NVFP4 FC1 schedule change")
            self._nvfp4_fc1_k256_overlap = enabled
            # A warm schedule switch is rank-local and capture-safe. Only a
            # cache miss requires compilation outside CUDA Graph capture.
            self._mega = compiled
            self._mega_key = self._mega_compile_key() if self._mega else None

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
            *MegaMoEHopperFp8Frontend._mega_compile_key(self),
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

    def _prepare_compile(self, key: tuple) -> Optional[_CompiledMega]:
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

    @staticmethod
    def _kernel_class():
        from .nvfp4_kernels import Sm90MegaMoESwapABNvfp4Fp8Kernel

        return Sm90MegaMoESwapABNvfp4Fp8Kernel

    def _kernel_extra_kwargs(self) -> Dict[str, Any]:
        return {
            "nvfp4_fc1_k256_overlap": self._nvfp4_fc1_k256_overlap,
            "nvfp4_conversion": self.config.nvfp4_conversion,
        }

    def _allocate_compiled_workspace(self, kernel) -> _CompiledMega:
        mega = self._get_nvfp4_workspace(kernel)
        self._mega = mega
        self._mega_key = None
        return mega

    def _after_compile(self, mega: _CompiledMega, key: tuple) -> None:
        self._nvfp4_schedule_cache[self._nvfp4_fc1_k256_overlap] = mega
        self._mega_key = key


# Kernel-ready leg: packed payload, transposed E4M3 scale pairs, static unit
# activation placeholder, and per-expert global scale * 8.
TransformedNvfp4Weights = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass
class MegaMoEHopperNvfp4SymmBuffer(MegaMoEHopperMxfp4SymmBuffer):
    """NVFP4 session with the same symmetric activation wire as MXFP4."""

    _frontend: MegaMoEHopperNvfp4Frontend


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
    """Select a reference-model heuristic without querying CUDA state."""
    from .nvfp4_tuner import hopper_nvfp4_default_profile

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
    use_tuning_cache: bool = True,
) -> Dict[str, Any]:
    """Read a compatible NV tuning entry, then use the reference heuristic."""
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

    def compatible(tactic: Dict[str, Any]) -> bool:
        tile = tactic["mma_tiler_mnk"]
        if hidden % tile[2] or intermediate % tile[2]:
            return False
        try:
            validate_nvfp4_fc1_overlap_override(
                effective_overlap,
                mma_tiler_mnk=tuple(tile),
                fp8_scale_mode="mxfp4_hybrid",
                pingpong=tactic["pingpong"],
            )
        except ValueError:
            return False
        return True

    if use_tuning_cache:
        from .knob_cache import lookup_knobs
        from .nvfp4_tuner import hopper_nvfp4_cache_provenance_sha256

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
            if cached is not None and compatible(cached):
                return cached

    from .mxfp4_tuner import _ordered_base_candidates

    choices = ([] if profile is None else [profile["requested_tactic"]]) + list(
        _ordered_base_candidates(
            num_max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            routing_profile=routing_profile,
        )
    )
    for candidate in choices:
        candidate = _validate_complete_nvfp4_tactic(
            candidate, source="NVFP4 initial geometry"
        )
        if compatible(candidate):
            return candidate
    raise ValueError(
        "no NVFP4 initial tactic is compatible with the model shape and "
        "explicit FC1 overlap setting"
    )


@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
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
    apply_routing_weights_before_fc2: bool = True,
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
    manual_selector = (
        manual_geometry
        or manual_execution
        or (token_back_mode is not None and knobs is None)
    )
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
                (
                    (128, 32, 256)
                    if nvfp4_fc1_k256_overlap is True
                    else _NVFP4_MMA_TILER_DEFAULT
                )
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
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
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


@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
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
    apply_routing_weights_before_fc2: bool = True,
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
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
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
    return _allocate_fp4_symm_buffer(
        config, MegaMoEHopperNvfp4Frontend, MegaMoEHopperNvfp4SymmBuffer
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


class _Nvfp4InputBuffer(_Mxfp4InputBuffer, Protocol):
    pass


def _build_nvfp4_inputs(
    symm_buffer: _Nvfp4InputBuffer,
    transformed_l1: TransformedNvfp4Weights,
    transformed_l2: TransformedNvfp4Weights,
) -> MegaMoEHopperNvfp4Inputs:
    return _build_fp4_inputs(
        symm_buffer,
        transformed_l1,
        transformed_l2,
        MegaMoEHopperNvfp4Inputs,
        _resolve_transformed_nvfp4_weights,
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
