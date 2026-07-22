# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile SM90 (Hopper) FP8 MegaMoE API for ``Sm90MegaMoEFp8Kernel``.

Structural mirror of the SM100 tree's ``shim/mxfp8.py``, retargeted at the
Hopper FP8 kernels (``Sm90MegaMoEFp8Kernel`` / ``Sm90MegaMoESwapABFp8Kernel``
in ``src/moe_hopper_fp8/megamoe_kernel_fp8.py``).  The construct/launch recipe
follows the drop driver's ``moe_hopper_fp8/mega_runner.py run_kernel()``.

Hopper-specific deltas vs the SM100 MXFP8 frontend:

* FP8 dequant-scale ABI: four extra fp32 launch tensors
  (``fc{1,2}_{activation,weight}_dequant_scale``) whose semantics flip with
  ``fp8_scale_mode`` (see :class:`MegaMoEHopperFp8Inputs`).
* ``fp8_scale_mode`` (``per_tensor`` legacy E8M0 wire / ``blockwise`` FP32
  scales) and ``fp8_accum_mode`` (``1xacc`` / ``2xacc``) compile knobs.
* Native (``M=64``) and swap-AB (``M in (128, 256)``) kernel geometries,
  selected by ``swap_ab``; the token padding block is the physical token tile
  (mma M native, mma N when swapped) instead of the SM100 EpilogueTokenTile.
* Opaque workspaces are passed as raw ``cute`` Uint8 POINTERS
  (``cute.runtime.make_ptr``), not tensor views: the internal combine plane
  can push ``shared_workspace`` beyond 2 GiB and the kernel partitions the
  base pointer with Int64 byte offsets.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple  # noqa: F401

import torch

from .comm import (
    _CompiledMega,
    _compute_peer_offsets,
    bootstrap_dist,
    ensure_not_capturing,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)

_KIND_TO_TORCH_DTYPE = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

# Legacy per-tensor E8M0 scale wire dtype (moe_nvfp4_swapab.runner_common
# Mxfp8ScaleDtype; the FP8 driver aliases it Fp8E8M0ScaleDtype).
_E8M0_SCALE_DTYPE = torch.float8_e8m0fnu

# ABI constants mirrored from common/megamoe_constants.py so this module
# imports without cutlass (megamoe_constants pulls cutlass.cutlass_dsl at
# import).  ``_ensure_mega_compiled`` re-asserts the buffer-sizing ones
# against the vendored source on first compile, so a kernel drop that moves
# them fails loudly instead of silently mis-sizing staging buffers.
_FP8_E8M0_SF_VEC_SIZE = 32  # Fp8E8M0SfVecSize: elements per E8M0 SF byte
_FP8_BLOCK_SCALE_K = 128  # Fp8BlockScaleK: elements per blockwise FP32 act scale
_FP8_WEIGHT_SCALE_BLOCK_N = 128  # Fp8WeightScaleBlockN
_FP8_WEIGHT_SCALE_BLOCK_K = 128  # Fp8WeightScaleBlockK
_FP8_DISPATCH_SCALE_ATOM_K = 128  # Fp8DispatchScaleAtomK = 4 * Fp8E8M0SfVecSize

# Kernel-geometry choices mirrored from moe_hopper_fp8/epilogue_fp8.py and
# epilogue_fp8_swapab.py (pre-checks only -- the kernel ctor re-validates and
# is authoritative).  The Hopper FP8 fork is 1-CTA-only.
_NONSWAP_TILE_M_CHOICES = (64,)
_NONSWAP_TILE_N_CHOICES = (128, 256)
_SWAPAB_TILE_M_CHOICES = (128, 256)
_SWAPAB_TILE_N_CHOICES = (16, 32, 64, 128)

# Drop-driver defaults (moe_hopper_fp8/mega_runner.py CLI): (64, 128, 128)
# native, remapped to (256, 32, 128) under --swap_ab.
_DEFAULT_MMA_TILER_NATIVE = (64, 128, 128)
_DEFAULT_MMA_TILER_SWAPAB = (256, 32, 128)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(a: int, b: int) -> int:
    return -(-a // b) * b


def _kind_to_cutlass_dtype(kind: str):
    import cutlass

    return {
        "fp8_e4m3": cutlass.Float8E4M3FN,
        "fp8_e5m2": cutlass.Float8E5M2,
    }[kind]


@dataclasses.dataclass(frozen=True)
class MegaMoEHopperFp8Config:
    """Compile-time / launch-time SM90 FP8 MegaMoE configuration.

    ``intermediate`` is the post-SwiGLU width (matching the SM100 MXFP8
    config and SGLang).  The kernel's FC1 gate+up width is derived as
    ``2 * intermediate`` (= ``static_expert_shape[1]``, the drop driver's
    ``problem.intermediate``).
    """

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int

    kind: Literal["fp8_e4m3", "fp8_e5m2"] = "fp8_e4m3"
    fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor"
    fp8_accum_mode: Literal["1xacc", "2xacc"] = "1xacc"
    swap_ab: bool = False
    mma_tiler_mnk: Tuple[int, int, int] = _DEFAULT_MMA_TILER_NATIVE
    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    # Drop-driver defaults (mega_runner.py __main__): flag_batch=1,
    # epi_flag_batch=(2, 4).
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (2, 4)
    in_kernel_fc2_reduce: bool = False
    token_back_by_dispatch: bool = False
    # deepgemm compute graph: routing weights folded into the SwiGLU output
    # before FC1-output quantization (the driver's ref_compute_graph switch).
    # False leaves the staged FC2 terms unweighted and applies scores in the
    # standalone TopkReduce; Form B (ikr) requires True.
    apply_topk_in_fc1: bool = True
    gate_up_clamp: Optional[float] = None
    enable_iket: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(
                f"kind must be one of {sorted(_KIND_TO_TORCH_DTYPE)}, "
                f"got {self.kind!r}."
            )
        if self.fp8_scale_mode not in ("per_tensor", "blockwise"):
            raise ValueError(
                "fp8_scale_mode must be 'per_tensor' or 'blockwise', "
                f"got {self.fp8_scale_mode!r}."
            )
        if self.fp8_accum_mode not in ("1xacc", "2xacc"):
            raise ValueError(
                "fp8_accum_mode must be '1xacc' or '2xacc', "
                f"got {self.fp8_accum_mode!r}."
            )
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank must be in [0, world_size), got rank={self.rank}, "
                f"world_size={self.world_size}."
            )
        if self.num_tokens_per_rank <= 0:
            raise ValueError(
                f"num_tokens_per_rank must be positive, got {self.num_tokens_per_rank}."
            )
        if self.num_topk <= 0:
            raise ValueError(f"num_topk must be positive, got {self.num_topk}.")
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                "num_total_experts must be divisible by world_size "
                f"({self.num_total_experts} % {self.world_size} != 0)."
            )
        # 64 covers the TMA 16B fp8 row alignment, the gate/up interleave
        # (Fp8GateUpInterleave = 8), and the E8M0 SF-word packing.
        # PORT NOTE: tail behaviour for hidden/intermediate that are 64- but
        # not 128-multiples has not been hardware-verified on SM90 yet; the
        # drop's functional tests use 128-multiples.
        if self.hidden % 64 != 0 or self.intermediate % 64 != 0:
            raise ValueError(
                "hidden and intermediate must be multiples of 64 "
                f"(got hidden={self.hidden}, intermediate={self.intermediate})."
            )
        if self.fp8_scale_mode == "blockwise":
            # Driver divisibility contract (mega_runner.generate_inputs):
            # hidden % Fp8BlockScaleK, gate+up % Fp8WeightScaleBlockN, and
            # downproj % {Fp8Fc2ActivationScaleK, Fp8WeightScaleBlockK}
            # collapse to hidden % 128 and intermediate % 128.
            if self.hidden % _FP8_BLOCK_SCALE_K != 0:
                raise ValueError(
                    f"blockwise FP8 requires hidden ({self.hidden}) divisible "
                    f"by {_FP8_BLOCK_SCALE_K}."
                )
            if self.intermediate % _FP8_WEIGHT_SCALE_BLOCK_K != 0:
                raise ValueError(
                    f"blockwise FP8 requires intermediate ({self.intermediate}) "
                    f"divisible by {_FP8_WEIGHT_SCALE_BLOCK_K}."
                )
        if self.in_kernel_fc2_reduce and self.token_back_by_dispatch:
            raise ValueError(
                "in_kernel_fc2_reduce and token_back_by_dispatch cannot both be True."
            )
        if self.in_kernel_fc2_reduce and not self.apply_topk_in_fc1:
            # Kernel invariant: the Form B REDG path collapses topk before a
            # separate reducer could apply routing weights.
            raise ValueError(
                "in_kernel_fc2_reduce requires apply_topk_in_fc1=True."
            )
        if not self.force_static_sched:
            raise ValueError(
                "The Hopper FP8 v1 kernel only implements "
                "force_static_sched=True (dynamic CLC is future work)."
            )
        m, n, k = self.mma_tiler_mnk
        if self.use_2cta_instrs or self.cluster_shape_mnk != (1, 1, 1):
            raise ValueError(
                "The Hopper FP8 MegaMoE fork is 1-CTA-only: use_2cta_instrs "
                "must be False and cluster_shape_mnk must be (1, 1, 1); got "
                f"use_2cta_instrs={self.use_2cta_instrs}, "
                f"cluster_shape_mnk={self.cluster_shape_mnk}."
            )
        if self.swap_ab:
            if m not in _SWAPAB_TILE_M_CHOICES or n not in _SWAPAB_TILE_N_CHOICES:
                raise ValueError(
                    "swap-AB Hopper FP8 requires mma_tiler M in "
                    f"{_SWAPAB_TILE_M_CHOICES} and N in {_SWAPAB_TILE_N_CHOICES}; "
                    f"got mma_tiler_mnk={self.mma_tiler_mnk}."
                )
        else:
            if m not in _NONSWAP_TILE_M_CHOICES or n not in _NONSWAP_TILE_N_CHOICES:
                raise ValueError(
                    "native (non-swap) Hopper FP8 requires mma_tiler M in "
                    f"{_NONSWAP_TILE_M_CHOICES} and N in {_NONSWAP_TILE_N_CHOICES}; "
                    f"got mma_tiler_mnk={self.mma_tiler_mnk}."
                )
        if k % _FP8_DISPATCH_SCALE_ATOM_K != 0:
            raise ValueError(
                f"mma_tiler K ({k}) must be a multiple of the FP8 dispatch "
                f"scale atom K = {_FP8_DISPATCH_SCALE_ATOM_K}."
            )
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"load_balance_mode must be 'static' or 'atomic_counter'; "
                f"got {self.load_balance_mode!r}."
            )
        if self.group_hint is not None and self.group_hint <= 0:
            raise ValueError(
                f"group_hint must be positive when set, got {self.group_hint}."
            )
        if self.flag_batch < 1 or self.flag_batch > 32:
            raise ValueError(
                f"flag_batch must be in [1, 32], got {self.flag_batch}."
            )
        eb = self.epi_flag_batch
        if len(eb) != 2:
            raise ValueError(
                f"epi_flag_batch must be a (fc1, fc2) pair, got {self.epi_flag_batch}."
            )
        for leg, val in (("fc1", eb[0]), ("fc2", eb[1])):
            # The epilogue clamps into [1, 32] silently; validate instead so a
            # typo'd knob fails loudly.
            if val < 1 or val > 32:
                raise ValueError(
                    f"epi_flag_batch[{leg}] must be in [1, 32], got {val}."
                )

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def torch_ab_dtype(self) -> torch.dtype:
        return _KIND_TO_TORCH_DTYPE[self.kind]

    @property
    def fc1_out(self) -> int:
        return 2 * self.intermediate

    @property
    def blockwise(self) -> bool:
        return self.fp8_scale_mode == "blockwise"


@dataclasses.dataclass
class MegaMoEHopperFp8Inputs:
    """Per-rank tensors for one SM90 FP8 MegaMoE launch.

    Scale ABI (T=tokens, E=local experts, H=hidden, I=post-SwiGLU width;
    FC1 produces the gate+up width 2I) -- per kernel ``__call__`` docstring:

    ============================  ==============================  =========================
    tensor                        per_tensor                      blockwise
    ============================  ==============================  =========================
    activation_sf                 (T, round_up(ceil(H/32), 4))    (T, round_up(H/128, 4))
                                  E8M0, dispatched but unused     FP32; first H/128 cols
                                  by GEMM dequantization          feed FC1
    fc1_weight_sf                 (E, flat) swizzled E8M0         (E, 2I/128, H/128) FP32
                                  placeholder, unused             weight scales, used
    fc2_weight_sf                 (E, flat) swizzled E8M0         (E, H/128, I/128) FP32
                                  placeholder, unused             weight scales, used
    fc1_activation_dequant_scale  (1,) FP32, used by FC1          (1,) ones, unused
    fc1_weight_dequant_scale      (E,) FP32, used by FC1          (E,) ones, unused
    fc2_activation_dequant_scale  (1,) FP32, quantizes FC2 input  (1,) ones, unused
                                  and dequantizes FC2
    fc2_weight_dequant_scale      (E,) FP32, used by FC2          (E,) ones, unused
    ============================  ==============================  =========================
    """

    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc1_activation_dequant_scale: torch.Tensor
    fc1_weight_dequant_scale: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    fc2_activation_dequant_scale: torch.Tensor
    fc2_weight_dequant_scale: torch.Tensor
    # Single 2D (T, hidden) bf16 output; the kernel reduces top-k internally
    # (Form B REDG or the fused standalone TopkReduce tail).
    output_activation: torch.Tensor


class MegaMoEHopperFp8Frontend:
    """Lazy-compile host wrapper for ``Sm90MegaMoE(SwapAB)Fp8Kernel``."""

    def __init__(self, config: MegaMoEHopperFp8Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoEHopperFp8Config:
        if self._gate_up_clamp == self._config.gate_up_clamp:
            return self._config
        return dataclasses.replace(self._config, gate_up_clamp=self._gate_up_clamp)

    def set_gate_up_clamp(self, clamp: Optional[float]) -> None:
        if self._gate_up_clamp == clamp:
            return
        ensure_not_capturing("set_gate_up_clamp (clamp change)")
        self._release_workspace()
        self._gate_up_clamp = clamp
        self._invalidate_compile_cache()

    def release(self) -> None:
        self._release_workspace()
        self._invalidate_compile_cache()

    def warmup(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> None:
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return None
        self._ensure_mega_compiled(inputs)

    def run(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = False,
    ) -> Optional[torch.Tensor]:
        """Launch SM90 FP8 MegaMoE and return the 2D ``(T, hidden)`` bf16 output.

        The kernel reduces the top-k combine internally (Form B REDG under
        ``in_kernel_fc2_reduce``, otherwise the fused standalone TopkReduce
        tail inside ``__call__``), so the result is always the reduced
        ``output_activation``.

        ``reset_counters=False`` (default): workspaces are allocated zeroed and
        the kernel tail-cleans its own counters/flags after every launch (the
        kernel-team drivers never host-reset), so no per-launch reset is
        needed.  Pass ``True`` only to recover after an aborted / interrupted
        launch left the workspaces dirty.

        Steady state (same session buffers, same token count, same stream) is
        a validated-once fast path: validation and cute-tensor construction
        run only when the launch cache misses.
        """
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        if resolved == 0:
            return None
        key = self._launch_cache_key(inputs, resolved)
        mega = self._mega
        if mega is None or mega.compiled is None or mega.launch_key != key:
            # Slow path: full validation + (re)compile + launch-kwargs build.
            # Any config change (set_gate_up_clamp) nulls self._mega, so a
            # live cache entry always matches the config.
            launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
            if launch_inputs is None:
                return None
            mega = self._ensure_mega_compiled(inputs)
            mega.launch_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
            mega.launch_key = key
            mega.launch_output = launch_inputs.output_activation
        if reset_counters:
            reset_compiled_mega_workspaces(mega)
        if self.config.in_kernel_fc2_reduce:
            # ikr accumulate-from-zero contract: output_activation is the
            # cross-rank REDG atomic-add target, so it must be zeroed before
            # every launch.  Zero the full raw buffer so stale rows beyond a
            # partial num_tokens can't leak from an earlier, larger launch.
            inputs.output_activation.zero_()
        mega.compiled(**mega.launch_kwargs)
        # Zero-break capture gate: a device synchronize would abort stream
        # capture, so skip it there (the graph replays under stream semantics).
        if sync and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return mega.launch_output

    def make_launch_thunk(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> Callable[[], None]:
        """Zero-arg launcher with args prebuilt (compiles if needed).

        Steady-state fast path for timing loops and tuners: no per-call Python
        arg rebuild, no workspace reset (the kernel tail-cleans its own
        counters/flags), no sync.  Output lands in
        ``inputs.output_activation``.  Invalid after the compile cache is
        invalidated (clamp change) or the buffers are freed.

        With ``in_kernel_fc2_reduce`` the thunk is two stream-ordered nodes --
        ``output_activation.zero_()`` then the kernel launch (accumulate-from-
        zero contract of the REDG target); both are CUDA-graph capturable.
        """
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return lambda: None
        mega = self._ensure_mega_compiled(inputs)
        runtime_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
        compiled = mega.compiled

        if self.config.in_kernel_fc2_reduce:
            output_activation = inputs.output_activation

            def thunk() -> None:
                output_activation.zero_()
                compiled(**runtime_kwargs)

        else:

            def thunk() -> None:
                compiled(**runtime_kwargs)

        return thunk

    @staticmethod
    def _launch_cache_key(inputs: MegaMoEHopperFp8Inputs, num_tokens: int) -> tuple:
        # Keyed on the RAW (pre-slice) input pointers + the resolved token
        # count: _slice_inputs slices from row 0, so the sliced views keep
        # these data_ptrs and the count captures the shape.
        t = inputs
        return (
            t.activation.data_ptr(),
            t.activation_sf.data_ptr(),
            t.topk_idx.data_ptr(),
            t.topk_weights.data_ptr(),
            t.fc1_weight.data_ptr(),
            t.fc1_weight_sf.data_ptr(),
            t.fc1_activation_dequant_scale.data_ptr(),
            t.fc1_weight_dequant_scale.data_ptr(),
            t.fc2_weight.data_ptr(),
            t.fc2_weight_sf.data_ptr(),
            t.fc2_activation_dequant_scale.data_ptr(),
            t.fc2_weight_dequant_scale.data_ptr(),
            t.output_activation.data_ptr(),
            num_tokens,
            torch.cuda.current_stream().cuda_stream,
        )

    def _mega_compile_key(self) -> tuple:
        c = self.config
        return (
            c.kind,
            c.fp8_scale_mode,
            c.fp8_accum_mode,
            c.swap_ab,
            c.world_size,
            c.rank,
            c.num_tokens_per_rank,
            c.num_topk,
            c.num_total_experts,
            c.hidden,
            c.intermediate,
            c.mma_tiler_mnk,
            c.cluster_shape_mnk,
            c.use_2cta_instrs,
            c.load_balance_mode,
            c.group_hint,
            c.force_static_sched,
            c.clc_bundle_size,
            c.num_sched_stages,
            c.flag_batch,
            c.epi_flag_batch,
            c.in_kernel_fc2_reduce,
            c.token_back_by_dispatch,
            c.apply_topk_in_fc1,
            self._gate_up_clamp,
            c.enable_iket,
        )

    @staticmethod
    def _assert_mirrored_constants() -> None:
        """Fail loudly if a kernel drop moved the mirrored ABI constants."""
        from common.megamoe_constants import (
            Fp8BlockScaleK,
            Fp8DispatchScaleAtomK,
            Fp8E8M0SfVecSize,
            Fp8WeightScaleBlockK,
            Fp8WeightScaleBlockN,
        )

        mirrored = (
            ("Fp8E8M0SfVecSize", Fp8E8M0SfVecSize, _FP8_E8M0_SF_VEC_SIZE),
            ("Fp8BlockScaleK", Fp8BlockScaleK, _FP8_BLOCK_SCALE_K),
            ("Fp8WeightScaleBlockN", Fp8WeightScaleBlockN, _FP8_WEIGHT_SCALE_BLOCK_N),
            ("Fp8WeightScaleBlockK", Fp8WeightScaleBlockK, _FP8_WEIGHT_SCALE_BLOCK_K),
            (
                "Fp8DispatchScaleAtomK",
                Fp8DispatchScaleAtomK,
                _FP8_DISPATCH_SCALE_ATOM_K,
            ),
        )
        for name, source_val, mirror_val in mirrored:
            if source_val != mirror_val:
                raise RuntimeError(
                    f"kernel drop changed common.megamoe_constants.{name} "
                    f"({source_val} != mirrored {mirror_val}); update the "
                    "mirrors in shim/hopper_fp8.py and re-audit staging-buffer "
                    "shapes."
                )

    def _ensure_mega_compiled(self, inputs: MegaMoEHopperFp8Inputs) -> _CompiledMega:
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        ensure_not_capturing("cute.compile + symmetric-heap allocation")
        self._release_workspace()

        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils

        from common.megamoe_constants import Fp8E8M0SfVecSize, SfPaddingBlock
        from moe_hopper_fp8.megamoe_kernel_fp8 import (
            Sm90MegaMoEFp8Kernel,
            Sm90MegaMoESwapABFp8Kernel,
        )

        self._assert_mirrored_constants()

        c = self.config
        # static_expert_shape binds (experts, intermediate_gateup, hidden) at
        # codegen time and is REQUIRED by the FP8 mega kernel.
        static_expert_shape = (
            c.num_experts_per_rank,
            c.fc1_out,
            c.hidden,
        )

        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        # Driver recipe: occupancy-aware count from the DSL (not the SM100
        # shim's sm_count // cluster_size heuristic).
        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        group_hint = c.group_hint if c.group_hint is not None else max_active_clusters

        # Keep the dispatch pool and scheduler on the physical token tile: M
        # for the native layout and N after swapping A/B (driver recipe).
        token_padding_block = (
            c.mma_tiler_mnk[1] if c.swap_ab else c.mma_tiler_mnk[0]
        )

        kernel_cls = (
            Sm90MegaMoESwapABFp8Kernel if c.swap_ab else Sm90MegaMoEFp8Kernel
        )
        kernel = kernel_cls(
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
            fp8_scale_mode=c.fp8_scale_mode,
            fp8_accum_mode=c.fp8_accum_mode,
            world_size=c.world_size,
            local_rank=c.rank,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_in_kernel_topk_reduce=c.in_kernel_fc2_reduce,
            apply_topk_in_fc1=c.apply_topk_in_fc1,
            # The SM90 drop keeps the bool (no token_back_mode enum yet).
            token_back_by_dispatch=c.token_back_by_dispatch,
            epi_flag_batch=c.epi_flag_batch,
            flag_batch=c.flag_batch,
            gate_up_clamp=self._gate_up_clamp,
        )

        local_ws_bytes, shared_ws_bytes = kernel.get_workspace_sizes()
        local_workspace = torch.zeros(
            (local_ws_bytes,),
            dtype=torch.uint8,
            device="cuda",
        )
        shared_workspace = sym_zeros((shared_ws_bytes,), torch.uint8)
        symmetric_base, peer_offsets_list = _compute_peer_offsets(
            shared_workspace,
            c.world_size,
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
        return self._mega

    def _invalidate_compile_cache(self) -> None:
        self._mega_key = None
        self._mega = None

    def _release_workspace(self) -> None:
        if self._mega is not None:
            ensure_not_capturing("workspace release (symmetric-heap free)")
            free_sym_tensor(self._mega.shared_workspace)

    @staticmethod
    def _resolve_num_tokens(
        inputs: MegaMoEHopperFp8Inputs,
        num_tokens: Optional[int],
    ) -> int:
        buf_tokens = inputs.activation.shape[0]
        if num_tokens is None:
            return buf_tokens
        if num_tokens < 0 or num_tokens > buf_tokens:
            raise ValueError(
                f"num_tokens must be in [0, {buf_tokens}], got {num_tokens}."
            )
        return num_tokens

    def _prepare_launch_inputs(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        *,
        num_tokens: Optional[int],
    ) -> Optional[MegaMoEHopperFp8Inputs]:
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        if resolved == 0:
            return None
        self._validate_inputs(inputs, num_tokens=resolved)
        buf_tokens = inputs.activation.shape[0]
        if not self.config.in_kernel_fc2_reduce and resolved < buf_tokens:
            raise ValueError(
                "Partial num_tokens is not supported when in_kernel_fc2_reduce=False "
                f"(kernel compiles for the full buffer of {buf_tokens} tokens). "
                f"Got num_tokens={resolved}."
            )
        if resolved == buf_tokens:
            return inputs
        return self._slice_inputs(inputs, resolved)

    @staticmethod
    def _slice_inputs(
        inputs: MegaMoEHopperFp8Inputs,
        num_tokens: int,
    ) -> MegaMoEHopperFp8Inputs:
        tok = slice(None, num_tokens)
        return MegaMoEHopperFp8Inputs(
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
        c = self.config
        ab_dtype = c.torch_ab_dtype
        buf_tokens = inputs.activation.shape[0]
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

        e = c.num_experts_per_rank

        current_device = torch.cuda.current_device()

        def _require_cuda(name: str, tensor: torch.Tensor) -> None:
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor.")
            # Workspace and stream are bound to the current device; a tensor
            # from another GPU would launch with an invalid pointer.
            if tensor.device.index != current_device:
                raise ValueError(
                    f"{name} must be on the current CUDA device "
                    f"(cuda:{current_device}), got {tensor.device}."
                )

        _require_cuda("activation", inputs.activation)
        if inputs.activation.ndim != 2 or inputs.activation.shape[0] != buf_tokens:
            raise ValueError(
                f"activation must be 2-D with leading dim {buf_tokens}, "
                f"got {tuple(inputs.activation.shape)}."
            )
        if inputs.activation.shape[-1] != c.hidden:
            raise ValueError(
                f"activation last dim must equal config.hidden ({c.hidden}), "
                f"got shape {tuple(inputs.activation.shape)}."
            )
        if inputs.activation.dtype != ab_dtype:
            raise ValueError(
                f"activation must have dtype {ab_dtype}, got {inputs.activation.dtype}."
            )

        token_tensors = (
            ("activation_sf", inputs.activation_sf),
            ("topk_idx", inputs.topk_idx),
            ("topk_weights", inputs.topk_weights),
            ("output_activation", inputs.output_activation),
        )
        for name, tensor in token_tensors:
            _require_cuda(name, tensor)
            if tensor.shape[0] != buf_tokens:
                raise ValueError(
                    f"{name}.shape[0] ({tensor.shape[0]}) must match "
                    f"activation.shape[0] ({buf_tokens})."
                )

        if inputs.output_activation.shape != (buf_tokens, c.hidden):
            raise ValueError(
                "output_activation must have shape "
                f"({buf_tokens}, {c.hidden}), "
                f"got {tuple(inputs.output_activation.shape)}."
            )
        if inputs.output_activation.dtype != torch.bfloat16:
            raise ValueError(
                "output_activation must be bfloat16, got "
                f"{inputs.output_activation.dtype}."
            )
        if inputs.topk_idx.shape != (buf_tokens, c.num_topk):
            raise ValueError(
                f"topk_idx must have shape ({buf_tokens}, {c.num_topk}), "
                f"got {tuple(inputs.topk_idx.shape)}."
            )
        if inputs.topk_idx.dtype != torch.int64:
            raise ValueError(f"topk_idx must be int64, got {inputs.topk_idx.dtype}.")
        if inputs.topk_weights.shape != (buf_tokens, c.num_topk):
            raise ValueError(
                f"topk_weights must have shape ({buf_tokens}, {c.num_topk}), "
                f"got {tuple(inputs.topk_weights.shape)}."
            )
        if inputs.topk_weights.dtype != torch.float32:
            raise ValueError(
                f"topk_weights must be float32, got {inputs.topk_weights.dtype}."
            )

        if c.blockwise:
            sf_cols = c.hidden // _FP8_BLOCK_SCALE_K
            if inputs.activation_sf.dtype != torch.float32:
                raise ValueError(
                    "blockwise activation_sf must be float32, "
                    f"got {inputs.activation_sf.dtype}."
                )
        else:
            sf_cols = _ceil_div(c.hidden, _FP8_E8M0_SF_VEC_SIZE)
            if inputs.activation_sf.dtype != _E8M0_SCALE_DTYPE:
                raise ValueError(
                    f"per-tensor activation_sf must have dtype {_E8M0_SCALE_DTYPE}, "
                    f"got {inputs.activation_sf.dtype}."
                )
        # Both modes pad the K_sf storage axis to a multiple of 4 words:
        # per-tensor packs 4 E8M0 SFs per uint32 dispatch LDG.32 wire word,
        # blockwise needs a 16-byte FP32 row stride for TMA (the kernel reads
        # round_up(cols, 4) words per token either way).
        if inputs.activation_sf.shape[-1] % 4 != 0:
            raise ValueError(
                f"activation_sf.shape[-1] ({inputs.activation_sf.shape[-1]}) "
                "must be a multiple of 4."
            )
        if inputs.activation_sf.shape[-1] < sf_cols:
            raise ValueError(
                f"activation_sf.shape[-1] ({inputs.activation_sf.shape[-1]}) "
                f"must be >= {sf_cols} (hidden={c.hidden}, "
                f"fp8_scale_mode={c.fp8_scale_mode!r})."
            )

        weight_checks = (
            ("fc1_weight", inputs.fc1_weight, (e, c.hidden, c.fc1_out)),
            ("fc2_weight", inputs.fc2_weight, (e, c.intermediate, c.hidden)),
        )
        for name, tensor, shape in weight_checks:
            _require_cuda(name, tensor)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
                )
            if tensor.dtype != ab_dtype:
                raise ValueError(
                    f"{name} must have dtype {ab_dtype}, got {tensor.dtype}."
                )
            # GEMM K must be the stride-1 axis (dim 1 for both legs after the
            # driver's permute).  A plain .contiguous() tensor would silently
            # compute garbage, so reject it here instead of in the kernel.
            if tensor.stride(1) != 1:
                raise ValueError(
                    f"{name} must be K-major (stride-1 along dim 1; got "
                    f"strides {tuple(tensor.stride())}). Permute the logical "
                    "row-major weight instead of calling .contiguous()."
                )

        if c.blockwise:
            wn = _FP8_WEIGHT_SCALE_BLOCK_N
            wk = _FP8_WEIGHT_SCALE_BLOCK_K
            weight_sf_checks = (
                (
                    "fc1_weight_sf",
                    inputs.fc1_weight_sf,
                    (e, c.fc1_out // wn, c.hidden // wk),
                ),
                (
                    "fc2_weight_sf",
                    inputs.fc2_weight_sf,
                    (e, c.hidden // wn, c.intermediate // wk),
                ),
            )
            for name, tensor, shape in weight_sf_checks:
                _require_cuda(name, tensor)
                if tuple(tensor.shape) != shape:
                    raise ValueError(
                        f"blockwise {name} must have shape {shape}, "
                        f"got {tuple(tensor.shape)}."
                    )
                if tensor.dtype != torch.float32:
                    raise ValueError(
                        f"blockwise {name} must be float32, got {tensor.dtype}."
                    )
                # The launch builds this view with assumed_align=16; a
                # mid-allocation slice (e.g. global[rank]) can start off-base.
                if tensor.data_ptr() % 16 != 0:
                    raise ValueError(
                        f"{name} data_ptr must be 16-byte aligned "
                        f"(got 0x{tensor.data_ptr():x}); clone the slice."
                    )
        else:
            for name, tensor in (
                ("fc1_weight_sf", inputs.fc1_weight_sf),
                ("fc2_weight_sf", inputs.fc2_weight_sf),
            ):
                _require_cuda(name, tensor)
                if tensor.ndim != 2 or tensor.shape[0] != e or tensor.shape[1] <= 0:
                    raise ValueError(
                        f"{name} must be 2-D with shape ({e}, <swizzled_sf_cols>), "
                        f"got {tuple(tensor.shape)}."
                    )

        scale_checks = (
            (
                "fc1_activation_dequant_scale",
                inputs.fc1_activation_dequant_scale,
                (1,),
            ),
            ("fc1_weight_dequant_scale", inputs.fc1_weight_dequant_scale, (e,)),
            (
                "fc2_activation_dequant_scale",
                inputs.fc2_activation_dequant_scale,
                (1,),
            ),
            ("fc2_weight_dequant_scale", inputs.fc2_weight_dequant_scale, (e,)),
        )
        for name, tensor, shape in scale_checks:
            _require_cuda(name, tensor)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
                )
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"{name} must be float32, got {tensor.dtype}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")

    @staticmethod
    def _to_cute(tensor: torch.Tensor, assumed_align: int = 16):
        import cutlass.torch as cutlass_torch

        cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        leading_dim = cutlass_torch.get_leading_dim(tensor)
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    @staticmethod
    def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
        """Opaque Uint8 gmem base pointer for the byte workspaces.

        The internal combine plane can push shared_workspace beyond 2 GiB, so
        no tensor view is materialized (a from_dlpack view would fold the size
        into 32-bit shape/stride arithmetic); the kernel partitions the base
        with Int64 byte offsets (driver recipe).
        """
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.typing import AddressSpace

        return cute.runtime.make_ptr(
            cutlass.Uint8,
            tensor.data_ptr(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )

    def _build_mega_runtime_kwargs(
        self,
        inputs: MegaMoEHopperFp8Inputs,
        mega: _CompiledMega,
    ) -> dict:
        import cuda.bindings.driver as cuda
        from src.sym_buffer import SymBufferHost

        c = self.config
        if inputs.activation_sf.shape[-1] % 4 != 0:
            raise ValueError(
                f"activation_sf.shape[-1] ({inputs.activation_sf.shape[-1]}) "
                "must be a multiple of 4."
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        peer_rank_ptr_mapper_host = SymBufferHost(
            base_addr=mega.symmetric_base,
            offsets=tuple(mega.peer_offsets_list),
            rank_idx=c.rank,
            num_max_ranks=c.world_size,
        )

        return dict(
            activation=self._to_cute(inputs.activation),
            activation_sf=self._to_cute(inputs.activation_sf),
            topk_idx=self._to_cute(inputs.topk_idx),
            topk_weights=self._to_cute(inputs.topk_weights),
            fc1_weight=self._to_cute(inputs.fc1_weight),
            fc1_weight_sf=self._to_cute(inputs.fc1_weight_sf),
            # Dequant scales ride assumed_align=4 (driver recipe): they are
            # tiny fp32 vectors that may be unpadded views.
            fc1_activation_dequant_scale=self._to_cute(
                inputs.fc1_activation_dequant_scale, assumed_align=4
            ),
            fc1_weight_dequant_scale=self._to_cute(
                inputs.fc1_weight_dequant_scale, assumed_align=4
            ),
            fc2_weight=self._to_cute(inputs.fc2_weight),
            fc2_weight_sf=self._to_cute(inputs.fc2_weight_sf),
            fc2_activation_dequant_scale=self._to_cute(
                inputs.fc2_activation_dequant_scale, assumed_align=4
            ),
            fc2_weight_dequant_scale=self._to_cute(
                inputs.fc2_weight_dequant_scale, assumed_align=4
            ),
            output_activation=self._to_cute(inputs.output_activation),
            local_workspace=self._to_cute_ptr(mega.local_workspace),
            shared_workspace=self._to_cute_ptr(mega.shared_workspace),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )


# ---------------------------------------------------------------------------
# High-level MegaMoE API (symm buffers + launch + dummy inputs)
# ---------------------------------------------------------------------------

# Kernel-ready weight leg: (weight, weight_sf, activation_dequant_scale,
# weight_dequant_scale).  The two dequant scales may be None ONLY in blockwise
# mode (the kernel ignores them there; unit tensors are substituted and cached
# on the symm buffer so the launch cache stays warm).
TransformedFp8Weights = Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]

HopperFp8Kind = Literal["fp8_e4m3", "fp8_e5m2"]


def _sym_zeros_byte_view_1b(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 / E8M0 symmetric heap via uint8 reinterpret (matches mega_runner).

    nvshmem4py doesn't natively support the 1-byte fp8/E8M0 dtypes, so the
    allocation is a uint8 byte buffer re-viewed to ``target_dtype``.  Returns
    ``(view, root_uint8_buffer)``; free the root via :func:`free_sym_tensor`.
    """
    total_bytes = 1
    for dim_size in logical_shape:
        total_bytes *= dim_size
    root = sym_zeros((total_bytes,), torch.uint8)
    view = root.view(target_dtype).reshape(logical_shape)
    return view, root


def init_dist() -> Tuple[int, int]:
    """Initialize torch.distributed + NVSHMEM (or single-rank when ``MEGA_NO_DIST=1``).

    Returns ``(rank, world_size)``.
    """
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


@dataclass
class MegaMoEHopperFp8SymmBuffer:
    """Symmetric-heap staging buffers for one SM90 FP8 MegaMoE session.

    Mirrors the SM100 :class:`MegaMoEMxfp8SymmBuffer`: exposes ``x``,
    ``x_sf``, ``topk_idx``, and ``topk_weights`` views sized for
    ``num_max_tokens``.  ``x_sf`` is mode-dependent:

    * ``per_tensor``: ``(T, round_up(ceil(hidden/32), 4))`` E8M0
      (``torch.float8_e8m0fnu``) legacy SF wire.
    * ``blockwise``: ``(T, round_up(hidden/128, 4))`` float32 per-token block
      scales (first ``hidden/128`` columns are live; the tail is TMA padding).

    Expert weights are **not** stored here -- pass ``transformed_l1`` /
    ``transformed_l2`` to :func:`hopper_fp8_mega_moe` each launch.
    """

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int
    kind: HopperFp8Kind
    fp8_scale_mode: str

    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor

    _frontend: MegaMoEHopperFp8Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    # Blockwise-mode unit dequant scales, allocated lazily and cached so the
    # launch cache keys (data_ptrs) stay stable across launches.
    _unit_scales: Dict[Tuple[str, int], torch.Tensor] = field(default_factory=dict)
    _destroyed: bool = False

    def destroy(self) -> None:
        """Release symmetric-heap allocations and compiled kernel workspaces."""
        if self._destroyed:
            return
        self._frontend.release()
        for root in self._sym_roots:
            free_sym_tensor(root)
        self._sym_roots.clear()
        self._unit_scales.clear()
        self._destroyed = True

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    def _unit_scale(self, name: str, numel: int) -> torch.Tensor:
        key = (name, numel)
        cached = self._unit_scales.get(key)
        if cached is None:
            cached = torch.ones((numel,), dtype=torch.float32, device="cuda")
            self._unit_scales[key] = cached
        return cached


def get_symm_buffer_for_hopper_fp8_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: HopperFp8Kind = "fp8_e4m3",
    fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor",
    fp8_accum_mode: Literal["1xacc", "2xacc"] = "1xacc",
    swap_ab: bool = False,
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_by_dispatch: bool = False,
    apply_topk_in_fc1: bool = True,
    load_balance_mode: Literal["static", "atomic_counter"] = "static",
    group_hint: Optional[int] = None,
    clc_bundle_size: Optional[int] = None,
    num_sched_stages: Optional[int] = None,
    flag_batch: int = 1,
    epi_flag_batch: Tuple[int, int] = (2, 4),
) -> MegaMoEHopperFp8SymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one SM90 FP8 session.

    Argument order follows the SM100 frontends (problem sizes first).  Pass
    ``rank`` / ``world_size`` from :func:`init_dist`.

    ``kind`` selects the fp8 element format (``fp8_e4m3`` or ``fp8_e5m2``).
    ``fp8_scale_mode`` picks the scale ABI (legacy per-tensor E8M0 wire vs
    blockwise FP32 scales); ``fp8_accum_mode`` picks 1x vs 2x FP8 WGMMA
    accumulation.  ``swap_ab`` selects the swap-A/B kernel geometry;
    ``mma_tiler_mnk=None`` uses the drop driver's default for the geometry
    ((64, 128, 128) native, (256, 32, 128) swap-AB).
    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is
    a deprecated alias for ``gate_up_clamp``.
    ``intermediate`` is the post-SwiGLU width, matching the SM100 frontends
    and SGLang (the kernel's gate+up width is ``2 * intermediate``).

    Expert weights are not allocated here; supply kernel-ready
    ``(weight, weight_sf, activation_dequant_scale, weight_dequant_scale)``
    tuples to :func:`hopper_fp8_mega_moe` instead.

    PORT NOTE: no ``knobs=`` dict / offline knob-cache path yet -- the SM90
    tree has no tuner module; kernel knobs are the explicit keyword args
    above.  Wire ``resolve_knobs``-style lookup when an SM90 tuner lands.
    """
    if hidden % 64 != 0 or intermediate % 64 != 0:
        raise ValueError(
            "MegaMoE requires hidden and intermediate to be multiples of 64."
        )
    if num_total_experts % world_size != 0:
        raise ValueError("num_total_experts must be divisible by world_size.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    if mma_tiler_mnk is None:
        mma_tiler_mnk = (
            _DEFAULT_MMA_TILER_SWAPAB if swap_ab else _DEFAULT_MMA_TILER_NATIVE
        )

    cfg = MegaMoEHopperFp8Config(
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
        swap_ab=swap_ab,
        mma_tiler_mnk=mma_tiler_mnk,
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_by_dispatch=token_back_by_dispatch,
        apply_topk_in_fc1=apply_topk_in_fc1,
        load_balance_mode=load_balance_mode,
        group_hint=group_hint,
        clc_bundle_size=clc_bundle_size,
        num_sched_stages=num_sched_stages,
        flag_batch=flag_batch,
        epi_flag_batch=epi_flag_batch,
    )
    frontend = MegaMoEHopperFp8Frontend(cfg)

    data_dtype = cfg.torch_ab_dtype

    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view_1b((num_max_tokens, hidden), data_dtype)
    sym_roots.append(x_root)
    if cfg.blockwise:
        # FP32 per-token block scales; storage padded to 4 words (16B TMA
        # row stride), matching the driver's sym staging.
        sf_storage_cols = _round_up(hidden // _FP8_BLOCK_SCALE_K, 4)
        x_sf = sym_zeros((num_max_tokens, sf_storage_cols), torch.float32)
        sym_roots.append(x_sf)
    else:
        # Legacy E8M0 wire: pad K_sf to a multiple of 4 SFs so dispatch_pull's
        # LDG.32 byte stride matches the host row stride; the trailing padded
        # SFs pair with fp8 data in TMA's OOB-fill-0 region.
        sf_storage_cols = _round_up(
            _ceil_div(hidden, _FP8_E8M0_SF_VEC_SIZE), 4
        )
        x_sf, x_sf_root = _sym_zeros_byte_view_1b(
            (num_max_tokens, sf_storage_cols),
            _E8M0_SCALE_DTYPE,
        )
        sym_roots.append(x_sf_root)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    # The kernel treats -1 as the pad-row mask; zero-filled rows would dispatch
    # as live tokens routed to expert 0. Stagers overwrite [:n] and re-fill the
    # tail, but start from the masked state so a partial first staging is safe.
    topk_idx.fill_(-1)
    sym_roots.append(topk_idx)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    sym_roots.append(topk_weights)
    # Single 2D (T, hidden) bf16 output; the kernel reduces top-k internally.
    # Allocated on the symmetric heap unconditionally: under
    # in_kernel_fc2_reduce it IS the cross-rank REDG atomic-add target (a
    # rank-local buffer here would be a latent crash for ikr sessions), and in
    # explicit-reduce mode a sym allocation behaves like plain CUDA memory
    # locally.
    output_activation = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(output_activation)

    return MegaMoEHopperFp8SymmBuffer(
        num_total_experts=num_total_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        kind=kind,
        fp8_scale_mode=fp8_scale_mode,
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        output_activation=output_activation,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def _resolve_transformed_weights(
    symm_buffer: MegaMoEHopperFp8SymmBuffer,
    transformed: TransformedFp8Weights,
    leg: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack one weight leg, substituting cached unit scales in blockwise mode."""
    try:
        weight, weight_sf, activation_scale, weight_scale = transformed
    except (TypeError, ValueError):
        raise ValueError(
            f"transformed_{leg} must be a (weight, weight_sf, "
            "activation_dequant_scale, weight_dequant_scale) tuple; got "
            f"{type(transformed).__name__} of length "
            f"{len(transformed) if hasattr(transformed, '__len__') else '?'}."
        ) from None
    blockwise = symm_buffer._frontend.config.blockwise
    if activation_scale is None or weight_scale is None:
        if not blockwise:
            raise ValueError(
                f"transformed_{leg}: per-tensor fp8_scale_mode requires both "
                "dequant scales; only blockwise mode may pass None."
            )
        # Blockwise ignores the per-tensor scales but the kernel ABI still
        # takes them (driver passes ones); substitute stable cached units.
        if activation_scale is None:
            activation_scale = symm_buffer._unit_scale(f"{leg}_act", 1)
        if weight_scale is None:
            weight_scale = symm_buffer._unit_scale(
                f"{leg}_wgt", symm_buffer.num_experts_per_rank
            )
    return weight, weight_sf, activation_scale, weight_scale


def _build_inputs(
    symm_buffer: MegaMoEHopperFp8SymmBuffer,
    transformed_l1: TransformedFp8Weights,
    transformed_l2: TransformedFp8Weights,
) -> MegaMoEHopperFp8Inputs:
    (
        fc1_weight,
        fc1_weight_sf,
        fc1_activation_dequant_scale,
        fc1_weight_dequant_scale,
    ) = _resolve_transformed_weights(symm_buffer, transformed_l1, "l1")
    (
        fc2_weight,
        fc2_weight_sf,
        fc2_activation_dequant_scale,
        fc2_weight_dequant_scale,
    ) = _resolve_transformed_weights(symm_buffer, transformed_l2, "l2")
    return MegaMoEHopperFp8Inputs(
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


def hopper_fp8_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedFp8Weights,
    transformed_l2: TransformedFp8Weights,
    symm_buffer: MegaMoEHopperFp8SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch the fused SM90 FP8 MegaMoE kernel (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / ``x_sf`` / routing slices before
    calling.

    ``transformed_l1`` / ``transformed_l2`` are kernel-ready
    ``(weight, weight_sf, activation_dequant_scale, weight_dequant_scale)``
    tuples:

    * ``transformed_l1 = (w13, w13_sf, fc1_activation_dequant_scale,
      fc1_weight_dequant_scale)`` -- ``w13`` is the K-major fp8 gate+up weight
      ``(E, hidden, 2 * intermediate)`` (hidden stride-1).
    * ``transformed_l2 = (w2, w2_sf, fc2_activation_dequant_scale,
      fc2_weight_dequant_scale)`` -- ``w2`` is the K-major fp8 down-proj
      weight ``(E, intermediate, hidden)`` (intermediate stride-1).

    Per-mode shapes/semantics of ``*_sf`` and the dequant scales are in the
    :class:`MegaMoEHopperFp8Inputs` docstring.  In blockwise mode the two
    dequant-scale entries may be ``None`` (unit tensors are substituted; the
    kernel ignores them).

    ``y`` receives the top-k-reduced bf16 output for ``[:num_tokens]``.
    ``gate_up_clamp`` updates the kernel clamp for this session when set.
    ``activation_clamp`` is a deprecated alias for ``gate_up_clamp``.
    ``fast_math`` is accepted for DeepGEMM API parity and has no effect here.

    ``sync=False`` (default): the kernel launch and the ``y`` copy are
    enqueued on the current stream and this function returns without a host
    sync -- ``y`` is ready under normal stream semantics (synchronize or
    stream-order before reading it on the host).  Pass ``True`` for a
    blocking call (e.g. host-side timing).
    """
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM90 FP8 MegaMoE path.",
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
        if y.shape != (n, symm_buffer.hidden):
            raise ValueError(
                f"y must be ({n}, {symm_buffer.hidden}), got {tuple(y.shape)}."
            )
        if y.dtype != torch.bfloat16:
            raise ValueError(f"y must be bfloat16, got {y.dtype}.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)

    inputs = _build_inputs(symm_buffer, transformed_l1, transformed_l2)

    # The kernel reduces the top-k combine internally and writes the final 2D
    # (T, hidden) output; no host-side reduction is needed.  Launch the full
    # padded buffer (topk_idx[n:] == -1 marks the pad rows) and copy the live
    # [:n] rows out -- matches the reference driver, which does not slice.
    out = symm_buffer._frontend.run(inputs, num_tokens=None, sync=False)
    if y is None:
        # Zero-copy: the caller consumes the workspace view under stream
        # ordering (valid until the next launch on this session's buffers).
        result = out[:n] if out is not None else symm_buffer.output_activation[:0]
    else:
        result = None
        if out is not None:
            y.copy_(out[:n])
    if sync and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
    return result


def hopper_fp8_mega_launch_thunk(
    transformed_l1: TransformedFp8Weights,
    transformed_l2: TransformedFp8Weights,
    symm_buffer: MegaMoEHopperFp8SymmBuffer,
) -> Callable[[], None]:
    """Prebuilt zero-arg SM90 FP8 mega launcher for steady-state timing loops.

    The returned thunk is a bare compiled-kernel launch -- args prebuilt once,
    no per-call Python, no workspace reset (the kernel tail-cleans), no sync,
    no output copy.  The reduced bf16 output lands in
    ``symm_buffer.output_activation``.  Compiles on this call if needed.
    Rebuild the thunk after clamp changes or buffer destruction.
    """
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    inputs = _build_inputs(symm_buffer, transformed_l1, transformed_l2)
    return symm_buffer._frontend.make_launch_thunk(inputs)


def _create_dummy_weights(
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
    *,
    kind: HopperFp8Kind,
    fp8_scale_mode: str,
) -> Tuple[TransformedFp8Weights, TransformedFp8Weights]:
    """Random FP8 weights + per-mode scales for local smoke scripts.

    Follows the drop driver's perf-run weight assembly
    (``mega_runner.generate_inputs``): logical row-major weights permuted to
    K-major (never ``.contiguous()``), per-tensor SFs atom-swizzled via
    ``to_blocked``, blockwise scales as constant FP32 blocks.
    """
    from moe_hopper_fp8.hopper_moe_utils import (
        create_fp8_tensor,
        make_constant_block_scale,
        make_fp8_per_tensor_dequant_scale,
    )
    from moe_nvfp4_swapab.runner_common import (
        _stack_byte_reinterpretable_tensors,
        to_blocked,
    )

    data_dtype = _KIND_TO_TORCH_DTYPE[kind]
    fc1_out = 2 * intermediate  # gate+up width

    def _weight(shape: Tuple[int, ...]) -> torch.Tensor:
        return create_fp8_tensor(
            shape,
            data_dtype,
            perf_run=True,
            nonzero_value=0.5,
            generator=generator,
            perf_positive_only=True,
        )

    # fc1: logical (E, gate+up, hidden) permuted to (E, hidden, gate+up) with
    # hidden stride-1 (K-major); fc2: logical (E, hidden, intermediate)
    # permuted to (E, intermediate, hidden) with intermediate stride-1.
    fc1_weight = _weight((num_local_experts, fc1_out, hidden)).permute(0, 2, 1)
    fc2_weight = _weight((num_local_experts, hidden, intermediate)).permute(0, 2, 1)

    if fp8_scale_mode == "blockwise":
        fc1_weight_sf = make_constant_block_scale(
            data_dtype,
            (
                num_local_experts,
                fc1_out // _FP8_WEIGHT_SCALE_BLOCK_N,
                hidden // _FP8_WEIGHT_SCALE_BLOCK_K,
            ),
        )
        fc2_weight_sf = make_constant_block_scale(
            data_dtype,
            (
                num_local_experts,
                hidden // _FP8_WEIGHT_SCALE_BLOCK_N,
                intermediate // _FP8_WEIGHT_SCALE_BLOCK_K,
            ),
        )
        # Blockwise ignores the per-tensor dequant scales (None -> units).
        return (
            (fc1_weight, fc1_weight_sf, None, None),
            (fc2_weight, fc2_weight_sf, None, None),
        )

    # Per-tensor: unit E8M0 SF planes, atom-swizzled per expert into the flat
    # layout the TMA SFA descriptor expects (placeholder -- GEMM dequant uses
    # the per-tensor fp32 scales below).
    hidden_sf_cols = _ceil_div(hidden, _FP8_E8M0_SF_VEC_SIZE)
    intermediate_sf_cols = _ceil_div(intermediate, _FP8_E8M0_SF_VEC_SIZE)

    fc1_sf_plain = torch.ones(
        (num_local_experts, fc1_out, hidden_sf_cols),
        dtype=_E8M0_SCALE_DTYPE,
        device="cuda",
    )
    fc1_sf_swizzled = [to_blocked(fc1_sf_plain[e]) for e in range(num_local_experts)]
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_local_experts, fc1_sf_swizzled[0].numel()
    )

    fc2_sf_plain = torch.ones(
        (num_local_experts, hidden, intermediate_sf_cols),
        dtype=_E8M0_SCALE_DTYPE,
        device="cuda",
    )
    fc2_sf_swizzled = [to_blocked(fc2_sf_plain[e]) for e in range(num_local_experts)]
    fc2_weight_sf = _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0).view(
        num_local_experts, fc2_sf_swizzled[0].numel()
    )

    # Perf-style constant per-tensor dequant scales (driver perf_run branch).
    fc1_activation_dequant_scale = make_fp8_per_tensor_dequant_scale(
        data_dtype, (1,)
    )
    fc1_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
        data_dtype, (num_local_experts,)
    )
    fc2_activation_dequant_scale = make_fp8_per_tensor_dequant_scale(
        data_dtype, (1,)
    )
    fc2_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
        data_dtype, (num_local_experts,)
    )

    return (
        (
            fc1_weight,
            fc1_weight_sf,
            fc1_activation_dequant_scale,
            fc1_weight_dequant_scale,
        ),
        (
            fc2_weight,
            fc2_weight_sf,
            fc2_activation_dequant_scale,
            fc2_weight_dequant_scale,
        ),
    )


def create_dummy_inputs(
    rank: int,
    world_size: int,
    num_total_experts: int,
    num_max_tokens: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    *,
    kind: HopperFp8Kind = "fp8_e4m3",
    fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor",
    swap_ab: bool = False,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedFp8Weights,
    TransformedFp8Weights,
    MegaMoEHopperFp8SymmBuffer,
]:
    """Allocate symm buffer, FP8 weights, and stage activations + routing."""
    if num_tokens < 0 or num_tokens > num_max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {num_max_tokens}], got {num_tokens}."
        )

    from moe_hopper_fp8.hopper_moe_utils import (
        create_fp8_tensor,
        make_constant_block_scale,
    )

    num_local_experts = num_total_experts // world_size
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + rank)

    symm_buffer = get_symm_buffer_for_hopper_fp8_mega_moe(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        kind=kind,
        fp8_scale_mode=fp8_scale_mode,
        swap_ab=swap_ab,
        gate_up_clamp=clamp,
    )

    transformed_l1, transformed_l2 = _create_dummy_weights(
        num_local_experts,
        hidden,
        intermediate,
        gen,
        kind=kind,
        fp8_scale_mode=fp8_scale_mode,
    )

    data_dtype = symm_buffer._frontend.config.torch_ab_dtype
    activation = create_fp8_tensor(
        (num_tokens, hidden),
        data_dtype,
        perf_run=True,
        nonzero_value=0.5,
        generator=gen,
        perf_positive_only=True,
    )

    scores = torch.randn(
        num_tokens,
        num_total_experts,
        device="cuda",
        dtype=torch.float32,
    )
    topk_weights, topk_idx = torch.topk(
        scores,
        num_topk,
        dim=-1,
        largest=True,
        sorted=False,
    )

    symm_buffer.x[:num_tokens].view(torch.uint8).copy_(
        activation.view(torch.uint8),
    )
    if fp8_scale_mode == "blockwise":
        sf_cols = hidden // _FP8_BLOCK_SCALE_K
        activation_sf = make_constant_block_scale(
            data_dtype, (num_tokens, sf_cols)
        )
        symm_buffer.x_sf[:num_tokens, :sf_cols].copy_(activation_sf)
    else:
        sf_cols = _ceil_div(hidden, _FP8_E8M0_SF_VEC_SIZE)
        activation_sf = torch.ones(
            (num_tokens, sf_cols), dtype=_E8M0_SCALE_DTYPE, device="cuda"
        )
        symm_buffer.x_sf[:num_tokens, :sf_cols].view(torch.uint8).copy_(
            activation_sf.view(torch.uint8),
        )
    symm_buffer.topk_idx[:num_tokens].copy_(topk_idx.to(torch.int64))
    # Mask pad rows (and stale routes from a previous larger staging): the
    # launch covers the full buffer and relies on topk_idx[n:] == -1.
    symm_buffer.topk_idx[num_tokens:].fill_(-1)
    symm_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    y = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    return y, transformed_l1, transformed_l2, symm_buffer


def _main() -> None:
    """Minimal torchrun smoke for the SM90 FP8 MegaMoE thin API."""
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 or not bool(int(os.environ.get("MEGA_NO_DIST", "0"))):
        torch.cuda.set_device(local_rank)

    HIDDEN = 2048
    INTERMEDIATE = 1024
    NUM_TOKENS = 128
    NUM_MAX_TOKENS = 128
    NUM_TOPK = 4
    NUM_EXPERTS = 32
    GATE_UP_CLAMP = 10.0
    FP8_SCALE_MODE = os.environ.get("MEGA_FP8_SCALE_MODE", "per_tensor")
    SWAP_AB = bool(int(os.environ.get("MEGA_FP8_SWAP_AB", "0")))

    rank, world_size = init_dist()
    symm_buffer = None

    try:
        y, transformed_l1, transformed_l2, symm_buffer = create_dummy_inputs(
            rank,
            world_size,
            NUM_EXPERTS,
            NUM_MAX_TOKENS,
            NUM_TOKENS,
            NUM_TOPK,
            HIDDEN,
            INTERMEDIATE,
            fp8_scale_mode=FP8_SCALE_MODE,
            swap_ab=SWAP_AB,
            gate_up_clamp=GATE_UP_CLAMP,
            seed=0,
        )

        hopper_fp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=NUM_TOKENS,
            gate_up_clamp=GATE_UP_CLAMP,
        )
        torch.cuda.synchronize()

        if rank == 0:
            print("ok")
            print("y:", y.shape, y.dtype)
    finally:
        if symm_buffer is not None:
            symm_buffer.destroy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        no_dist = bool(int(os.environ.get("MEGA_NO_DIST", "0")))
        if not no_dist and dist.is_initialized():
            from src.bootstrap import finalize_dist_and_nvshmem

            finalize_dist_and_nvshmem()


if __name__ == "__main__":
    _main()
