# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile SM90 (Hopper) BF16 MegaMoE API for ``Sm90MegaMoEBf16Kernel``.

Structural mirror of :mod:`.hopper_fp8`, retargeted at the Hopper BF16
kernels (``Sm90MegaMoEBf16Kernel`` / ``Sm90MegaMoESwapABBf16Kernel`` in
``src/moe_hopper_bf16/megamoe_kernel_bf16.py``).  The construct/launch recipe
follows the drop driver's ``moe_hopper_bf16/mega_runner.py run_kernel()``.

Deltas vs the FP8 frontend (everything else is shared structure):

* No scale ABI at all: activations and weights are plain BF16, the dispatch
  wire carries no scale-factor plane, and the launch takes exactly five
  user tensors (activation, routing x2, fc1/fc2 weights) plus the output.
  ``TransformedBf16Weights`` is therefore ONE K-major bf16 tensor per leg.
* No ``kind`` / ``fp8_scale_mode`` / ``fp8_accum_mode`` compile knobs; the
  drop's token-bucket heuristic table (``moe_hopper_bf16/heuristic_config.py``)
  is keyed on the token count only.
* Tile K is 64 (two-byte operands: the 128-byte swizzle atom spans 64
  elements, see the table's K=64 note); the kernel accepts any multiple.
* ``compact_pull_buffer`` is not a knob (the BF16 kernel always compacts).
* Native (``M=64``) and swap-AB (``M in (128, 256)``) geometries, selected
  by ``swap_ab``; the token padding block is the physical token tile.
* Opaque workspaces are raw ``cute`` Uint8 POINTERS (``cute.runtime.make_ptr``),
  not tensor views, for the same >2 GiB reason as the FP8 frontend.
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

# Kernel-geometry choices mirrored from moe_hopper_bf16/epilogue_bf16.py and
# epilogue_bf16_swapab.py (pre-checks only -- the kernel ctor re-validates and
# is authoritative).
_NONSWAP_TILE_M_CHOICES = (64,)
_NONSWAP_TILE_N_CHOICES = (128, 256)
_SWAPAB_TILE_M_CHOICES = (128, 256)
_SWAPAB_TILE_N_CHOICES = (8, 16, 32, 64, 128)

# BF16 TMA/SMEM swizzle atom along K (kernel_bf16_glu_fc12.Bf16TmaAtomK).
_BF16_TMA_ATOM_K = 64

# CGA shapes accepted by the drop's _validate_mma_tiler_and_cluster_shape
# (both geometries).  Cluster K must stay 1.
_SUPPORTED_CLUSTER_SHAPES_MN = ((1, 1), (2, 1), (1, 2), (2, 2))

# Token-back placement enum mirrored from megamoe_kernel_bf16.py.
_TOKEN_BACK_MODES = ("epi_warps", "standalone_warps", "reuse_dispatch_warps")

# Drop-driver manual-mode default (heuristic_config.DEFAULT_MMA_TILER_MNK);
# swap-AB manual defaults live in moe_hopper_bf16/heuristic_config.py.
_DEFAULT_MMA_TILER_NATIVE = (64, 128, 64)

# Shape contract documented by the drop's harnesses (run_mega_tests.sh):
# hidden is a multiple of the fc2 N tile (256), the post-SwiGLU width a
# multiple of 64 (fc2 K tile / fc1 N tile 128 over the gate+up width).
_HIDDEN_ALIGN = 256
_INTERMEDIATE_ALIGN = 64

# Knob-cache key for this frontend (shared JSON file with the FP8 tree; the
# FP8 entries carry a real scale mode, so the keys never cross-match).
BF16_KNOB_CACHE_DTYPE = "bf16"
BF16_KNOB_CACHE_SCALE_MODE = "none"


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclasses.dataclass(frozen=True)
class MegaMoEHopperBf16Config:
    """Compile-time / launch-time SM90 BF16 MegaMoE configuration.

    ``intermediate`` is the post-SwiGLU width (matching the FP8 config and
    SGLang).  The kernel's FC1 gate+up width is derived as
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

    swap_ab: bool = False
    # Ping-pong task-tile scheduling: two WGMMA+epilogue warpgroups alternate
    # complete task tiles (requires N=128 native / M=128 swap-AB).
    pingpong: bool = False
    mma_tiler_mnk: Tuple[int, int, int] = _DEFAULT_MMA_TILER_NATIVE
    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    # Drop-driver defaults (mega_runner.py main()): flag_batch=1,
    # epi_flag_batch=(2, 4).
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (2, 4)
    in_kernel_fc2_reduce: bool = False
    # Legacy alias: True maps to token_back_mode="reuse_dispatch_warps",
    # False to "epi_warps".
    token_back_by_dispatch: bool = False
    # Explicit token-back placement; overrides token_back_by_dispatch when
    # set.  See megamoe_kernel_bf16.Sm90MegaMoEBf16Kernel token_back_mode.
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None
    # How many of the 4 dispatch warps do token-comm work at all (1/2/4);
    # the physical layout stays at 4.  Output-invariant work partitioning.
    active_dispatch_warps: int = 1
    # FC1 store offload to the empty warp; self-gating (non-swap non-ping-pong
    # with register headroom), output-invariant.
    fc1_store_offload: bool = True
    # Early fc1_done publication; output-invariant tuner knob (default off,
    # the fold layout forces it on since the store server warp is gone).
    fc1_early_done_publish: bool = False
    # Fold TMA-A / TMA-B / scheduler into the idle dispatch-warpgroup slots
    # (active_dispatch_warps == 1 only) and drop the producer warpgroup.
    fold_producer_warps: bool = True
    # Training forward: also write the raw pre-SwiGLU fc1 gate+up accumulator
    # (BF16, kernel gate/up-interleaved column order) to an expert-major pool
    # tensor with 128-row-aligned expert segments (``fc1_c``).  Off by
    # default; the kernel store path is compiled out when off.
    generate_c: bool = False
    # Tail-split pair tasks (see fc1_fc2_fuse_sched); legal only with a
    # 2-CTA token cluster: swap-AB cga (1, 2, 1) or non-swap cga (2, 1, 1).
    # Output-invariant; the heuristic table turns it on per bucket.
    tail_split_pairs: bool = False
    # deepgemm compute graph: routing weights folded into the SwiGLU output
    # before the BF16 FC1-output store.  False leaves the staged FC2 terms
    # unweighted and applies scores in the standalone TopkReduce; Form B
    # (ikr) requires True.
    apply_topk_in_fc1: bool = True
    gate_up_clamp: Optional[float] = None
    enable_iket: bool = False

    def __post_init__(self) -> None:
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
        # Drop-harness contract (moe_hopper_bf16/run_mega_tests.sh): hidden a
        # multiple of the fc2 N tile, the post-SwiGLU width a multiple of 64
        # (gate+up width a multiple of the fc1 N tile 128 / the gate-up
        # interleave).  Smaller multiples are not hardware-verified for BF16.
        if self.hidden % _HIDDEN_ALIGN != 0:
            raise ValueError(
                f"hidden must be a multiple of {_HIDDEN_ALIGN} (got hidden={self.hidden})."
            )
        if self.intermediate % _INTERMEDIATE_ALIGN != 0:
            raise ValueError(
                f"intermediate must be a multiple of {_INTERMEDIATE_ALIGN} "
                f"(got intermediate={self.intermediate})."
            )
        if (
            self.token_back_mode is not None
            and self.token_back_mode not in _TOKEN_BACK_MODES
        ):
            raise ValueError(
                f"token_back_mode must be one of {_TOKEN_BACK_MODES}, "
                f"got {self.token_back_mode!r}."
            )
        if self.active_dispatch_warps not in (1, 2, 4):
            raise ValueError(
                f"active_dispatch_warps must be 1, 2, or 4; got "
                f"{self.active_dispatch_warps!r}."
            )
        if self.in_kernel_fc2_reduce and not self.apply_topk_in_fc1:
            # Kernel invariant: the Form B REDG path collapses topk before a
            # separate reducer could apply routing weights.
            raise ValueError("in_kernel_fc2_reduce requires apply_topk_in_fc1=True.")
        if not self.force_static_sched:
            raise ValueError(
                "The Hopper BF16 kernel only implements "
                "force_static_sched=True (dynamic CLC is future work)."
            )
        m, n, k = self.mma_tiler_mnk
        if self.use_2cta_instrs:
            raise ValueError(
                "The Hopper BF16 MegaMoE fork has no 2-CTA WGMMA path: "
                "use_2cta_instrs must be False; got "
                f"use_2cta_instrs={self.use_2cta_instrs}."
            )
        cm, cn, ck = self.cluster_shape_mnk
        if ck != 1 or (cm, cn) not in _SUPPORTED_CLUSTER_SHAPES_MN:
            raise ValueError(
                "Hopper BF16 cluster_shape_mnk (m, n) must be one of "
                f"{_SUPPORTED_CLUSTER_SHAPES_MN} with k == 1; got "
                f"cluster_shape_mnk={self.cluster_shape_mnk}."
            )
        if self.pingpong:
            # Ping-pong assigns one physical warpgroup per complete task
            # tile: one N=128 WGMMA fragment native, one M=128 fragment
            # swap-AB (kernel _setup ValueError otherwise).
            if self.swap_ab and m != 128:
                raise ValueError(
                    "swap-AB Hopper BF16 ping-pong requires mma_tiler M=128; "
                    f"got mma_tiler_mnk={self.mma_tiler_mnk}."
                )
            if not self.swap_ab and n != 128:
                raise ValueError(
                    "native Hopper BF16 ping-pong requires mma_tiler N=128; "
                    f"got mma_tiler_mnk={self.mma_tiler_mnk}."
                )
        if self.swap_ab:
            if m not in _SWAPAB_TILE_M_CHOICES or n not in _SWAPAB_TILE_N_CHOICES:
                raise ValueError(
                    "swap-AB Hopper BF16 requires mma_tiler M in "
                    f"{_SWAPAB_TILE_M_CHOICES} and N in {_SWAPAB_TILE_N_CHOICES}; "
                    f"got mma_tiler_mnk={self.mma_tiler_mnk}."
                )
        else:
            if m not in _NONSWAP_TILE_M_CHOICES or n not in _NONSWAP_TILE_N_CHOICES:
                raise ValueError(
                    "native (non-swap) Hopper BF16 requires mma_tiler M in "
                    f"{_NONSWAP_TILE_M_CHOICES} and N in {_NONSWAP_TILE_N_CHOICES}; "
                    f"got mma_tiler_mnk={self.mma_tiler_mnk}."
                )
        if k <= 0 or k % _BF16_TMA_ATOM_K != 0:
            raise ValueError(
                f"mma_tiler K ({k}) must be a positive multiple of the BF16 "
                f"TMA/SMEM swizzle atom K = {_BF16_TMA_ATOM_K}."
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
        if self.tail_split_pairs:
            token_cluster, weight_cluster = (cn, cm) if self.swap_ab else (cm, cn)
            if token_cluster != 2 or weight_cluster != 1:
                raise ValueError(
                    "tail_split_pairs requires a token-side cluster of 2 and a "
                    "weight-side cluster of 1 (swap-AB cga (1, 2, 1) or non-swap "
                    f"cga (2, 1, 1)); got cluster_shape_mnk={self.cluster_shape_mnk} "
                    f"with swap_ab={self.swap_ab}."
                )
        if self.flag_batch < 1 or self.flag_batch > 32:
            raise ValueError(f"flag_batch must be in [1, 32], got {self.flag_batch}.")
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
        return torch.bfloat16

    @property
    def fc1_out(self) -> int:
        return 2 * self.intermediate

    @property
    def resolved_token_back_mode(self) -> str:
        """Explicit ``token_back_mode`` wins; else map the legacy bool."""
        if self.token_back_mode is not None:
            return self.token_back_mode
        return "reuse_dispatch_warps" if self.token_back_by_dispatch else "epi_warps"


@dataclasses.dataclass
class MegaMoEHopperBf16Inputs:
    """Per-rank tensors for one SM90 BF16 MegaMoE launch.

    T=tokens, E=local experts, H=hidden, I=post-SwiGLU width (FC1 produces
    the gate+up width 2I).  Weights are K-major: ``fc1_weight`` is
    ``(E, H, 2I)`` with H stride-1, ``fc2_weight`` is ``(E, I, H)`` with I
    stride-1 (the kernel's GEMM K axis must be the stride-1 axis).
    """

    activation: torch.Tensor  # (T, H) bf16
    topk_idx: torch.Tensor  # (T, K) int64
    topk_weights: torch.Tensor  # (T, K) fp32
    fc1_weight: torch.Tensor  # (E, H, 2I) bf16, K-major
    fc2_weight: torch.Tensor  # (E, I, H) bf16, K-major
    # Single 2D (T, hidden) bf16 output; the kernel reduces top-k internally
    # (Form B REDG or the fused standalone TopkReduce tail).
    output_activation: torch.Tensor


class MegaMoEHopperBf16Frontend:
    """Lazy-compile host wrapper for ``Sm90MegaMoE(SwapAB)Bf16Kernel``."""

    def __init__(self, config: MegaMoEHopperBf16Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoEHopperBf16Config:
        if self._gate_up_clamp == self._config.gate_up_clamp:
            return self._config
        return dataclasses.replace(self._config, gate_up_clamp=self._gate_up_clamp)

    def set_gate_up_clamp(self, clamp: Optional[float]) -> None:
        if self._gate_up_clamp == clamp:
            return
        ensure_not_capturing("set_gate_up_clamp (clamp change)")
        self._release_workspace()
        self._gate_up_clamp = clamp

    def apply_knobs(self, knobs: Optional[dict]) -> None:
        """Apply tuner knobs (see :mod:`.tuner_bf16`) to the session config.

        Invalidates the compile cache when the effective config changes; the
        next ``run()``/``warmup()`` recompiles.  Used by :mod:`.autotune_bf16`.
        """
        from .tuner import with_knobs

        new_config = with_knobs(self.config, knobs)
        if new_config == self._config:
            return
        ensure_not_capturing("apply_knobs (config change)")
        self._release_workspace()
        self._config = new_config

    def release(self) -> None:
        self._release_workspace()

    def warmup(
        self,
        inputs: MegaMoEHopperBf16Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> None:
        self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        self._ensure_mega_compiled(inputs)

    def run(
        self,
        inputs: MegaMoEHopperBf16Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = False,
    ) -> torch.Tensor:
        """Launch SM90 BF16 MegaMoE and return the 2D ``(T, hidden)`` bf16 output.

        Same contract as ``MegaMoEHopperFp8Frontend.run``: the kernel reduces
        the top-k combine internally, workspaces are allocated zeroed and the
        kernel tail-cleans its own counters/flags (``reset_counters=True``
        only to recover after an aborted launch), and steady state is a
        validated-once launch-kwargs fast path.

        ``num_tokens == 0`` (empty local batch) is NOT skipped: the launch is
        collective, so the rank launches its full buffer with every row
        marked pad and returns the full output view (no live rows).
        """
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        key = self._launch_cache_key(inputs, resolved)
        mega = self._mega
        if mega is None or mega.compiled is None or mega.launch_key != key:
            launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
            mega = self._ensure_mega_compiled(inputs)
            mega.launch_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
            mega.launch_key = key
            mega.launch_output = launch_inputs.output_activation
        elif resolved == 0:
            # Cache hit on an empty batch: the caller may have re-staged
            # routing rows since the last launch, so re-apply the pad mask.
            self._mask_empty_batch(inputs)
        if reset_counters:
            reset_compiled_mega_workspaces(mega)
        if self.config.in_kernel_fc2_reduce:
            # ikr accumulate-from-zero contract: output_activation is the
            # cross-rank REDG atomic-add target, so it must be zeroed before
            # every launch (full raw buffer, so stale rows beyond a partial
            # num_tokens can't leak from an earlier, larger launch).
            inputs.output_activation.zero_()
        if mega.fc1_c is not None:
            # generate_c pad-rows-zero contract: the kernel writes only the
            # live rows of each expert segment, so when an expert receives
            # fewer tokens than on the previous launch the rows that became
            # padding would keep stale activations.  Re-zero the pool before
            # every launch.
            mega.fc1_c.zero_()
        mega.compiled(**mega.launch_kwargs)
        if sync and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return mega.launch_output

    def make_launch_thunk(
        self,
        inputs: MegaMoEHopperBf16Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> Callable[[], None]:
        """Zero-arg launcher with args prebuilt (compiles if needed).

        Steady-state fast path for timing loops and tuners; see the FP8
        frontend for the contract.  With ``in_kernel_fc2_reduce`` the thunk
        is two stream-ordered nodes (output zero + launch); ``generate_c``
        adds the per-launch ``fc1_c`` zero (pad-rows-zero contract); an
        empty batch (``num_tokens == 0``) re-applies the all-pad routing mask
        before every launch (full-buffer collective launch, never a no-op).
        """
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        mega = self._ensure_mega_compiled(inputs)
        runtime_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
        compiled = mega.compiled

        pre_launch: list[Callable[[], None]] = []
        if resolved == 0:
            pre_launch.append(lambda: self._mask_empty_batch(inputs))
        if self.config.in_kernel_fc2_reduce:
            pre_launch.append(inputs.output_activation.zero_)
        if mega.fc1_c is not None:
            pre_launch.append(mega.fc1_c.zero_)

        if pre_launch:

            def thunk() -> None:
                for op in pre_launch:
                    op()
                compiled(**runtime_kwargs)

        else:

            def thunk() -> None:
                compiled(**runtime_kwargs)

        return thunk

    @staticmethod
    def _launch_cache_key(inputs: MegaMoEHopperBf16Inputs, num_tokens: int) -> tuple:
        # Keyed on the RAW (pre-slice) input pointers + the resolved token
        # count: _slice_inputs slices from row 0, so the sliced views keep
        # these data_ptrs and the count captures the shape.
        t = inputs
        return (
            t.activation.data_ptr(),
            t.topk_idx.data_ptr(),
            t.topk_weights.data_ptr(),
            t.fc1_weight.data_ptr(),
            t.fc2_weight.data_ptr(),
            t.output_activation.data_ptr(),
            num_tokens,
            torch.cuda.current_stream().cuda_stream,
        )

    def _mega_compile_key(self) -> tuple:
        c = self.config
        return (
            c.swap_ab,
            c.pingpong,
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
            c.resolved_token_back_mode,
            c.active_dispatch_warps,
            c.fc1_store_offload,
            c.fc1_early_done_publish,
            c.fold_producer_warps,
            c.generate_c,
            c.tail_split_pairs,
            c.apply_topk_in_fc1,
            self._gate_up_clamp,
            c.enable_iket,
        )

    def _ensure_mega_compiled(self, inputs: MegaMoEHopperBf16Inputs) -> _CompiledMega:
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        ensure_not_capturing("cute.compile + symmetric-heap allocation")
        self._release_workspace()

        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils

        from moe_hopper_bf16.megamoe_kernel_bf16 import (
            Sm90MegaMoEBf16Kernel,
            Sm90MegaMoESwapABBf16Kernel,
        )

        c = self.config
        # static_expert_shape binds (experts, intermediate_gateup, hidden) at
        # codegen time and is REQUIRED by the BF16 mega kernel.
        static_expert_shape = (
            c.num_experts_per_rank,
            c.fc1_out,
            c.hidden,
        )

        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        # Driver recipe: occupancy-aware count from the DSL.
        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        group_hint = c.group_hint if c.group_hint is not None else max_active_clusters

        # Keep the dispatch pool and scheduler on the physical token tile: M
        # for the native layout and N after swapping A/B (driver recipe).
        token_padding_block = c.mma_tiler_mnk[1] if c.swap_ab else c.mma_tiler_mnk[0]
        if c.generate_c:
            # fc1_c consumers (weight-gradient GEMMs) want 128-aligned expert
            # segments; same round-up as the FP8 frontend.
            token_padding_block = 128

        kernel_cls = Sm90MegaMoESwapABBf16Kernel if c.swap_ab else Sm90MegaMoEBf16Kernel
        kernel = kernel_cls(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=c.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=c.force_static_sched,
            clc_bundle_size=c.clc_bundle_size,
            num_sched_stages=c.num_sched_stages,
            ab_dtype=cutlass.BFloat16,
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
            fc1_store_offload=c.fc1_store_offload,
            fc1_early_done_publish=c.fc1_early_done_publish,
            active_dispatch_warps=c.active_dispatch_warps,
            fold_producer_warps=c.fold_producer_warps,
            generate_c=c.generate_c,
            tail_split_pairs=c.tail_split_pairs,
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
        if c.generate_c:
            # Expert-major pool rows (kernel.pool_token_capacity already counts
            # the 128-row padding per local expert).  Allocated zeroed AND
            # re-zeroed before every launch (run / make_launch_thunk): the
            # kernel only writes live rows, so pad rows are zero by host fill.
            mega.fc1_c = torch.zeros(
                (kernel.pool_token_capacity, c.fc1_out),
                dtype=torch.bfloat16,
                device="cuda",
            )
        compile_kwargs = self._build_mega_runtime_kwargs(inputs, mega)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if c.enable_iket:
            compile_kwargs["options"] = "iket"

        mega.compiled = cute.compile(kernel, **compile_kwargs)
        self._mega_key = key
        self._mega = mega
        return self._mega

    @property
    def fc1_c(self) -> Optional[torch.Tensor]:
        """generate_c output: raw pre-SwiGLU fc1 gate+up of the last launch.

        Same layout contract as ``MegaMoEHopperFp8Frontend.fc1_c``:
        ``(pool_rows, 2 * intermediate)`` BF16, expert-major 128-row-aligned
        segments in dispatch arrival order, pad rows zero (the pool is
        re-zeroed before every launch, so the contract holds across launches
        with shrinking per-expert counts).  None unless ``config.generate_c``;
        valid until the next launch.
        """
        return None if self._mega is None else self._mega.fc1_c

    def _invalidate_compile_cache(self) -> None:
        self._mega_key = None
        self._mega = None

    def _release_workspace(self) -> None:
        if self._mega is not None:
            ensure_not_capturing("workspace release (symmetric-heap free)")
            free_sym_tensor(self._mega.shared_workspace)
        # The cache entry dies with the workspace: if a recompile fails after
        # the free, a stale (_mega, _mega_key) hit would launch against freed
        # symmetric memory.
        self._invalidate_compile_cache()

    @staticmethod
    def _resolve_num_tokens(
        inputs: MegaMoEHopperBf16Inputs,
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

    @staticmethod
    def _mask_empty_batch(inputs: MegaMoEHopperBf16Inputs) -> None:
        """Mark the whole routing plane as pad for an empty local batch.

        ``num_tokens == 0`` is still a collective launch (peers pull this
        rank's experts and wait at every cross-rank barrier), so the rank
        launches its FULL buffer; with no live rows every row must carry
        the ``topk_idx == -1`` pad mask or the peers would route stale
        entries.  The caller's routing rows are meaningless for an empty
        batch, so overwriting them is the pad mask, not data loss.
        """
        inputs.topk_idx.fill_(-1)

    def _prepare_launch_inputs(
        self,
        inputs: MegaMoEHopperBf16Inputs,
        *,
        num_tokens: Optional[int],
    ) -> MegaMoEHopperBf16Inputs:
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        buf_tokens = inputs.activation.shape[0]
        if resolved == 0:
            # Empty local batch: full-buffer launch, all rows pad (see
            # _mask_empty_batch).  Never an early return -- skipping the
            # launch on one rank hangs the peers.
            self._validate_inputs(inputs, num_tokens=buf_tokens)
            self._mask_empty_batch(inputs)
            return inputs
        self._validate_inputs(inputs, num_tokens=resolved)
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
        inputs: MegaMoEHopperBf16Inputs,
        num_tokens: int,
    ) -> MegaMoEHopperBf16Inputs:
        tok = slice(None, num_tokens)
        return MegaMoEHopperBf16Inputs(
            activation=inputs.activation[tok],
            topk_idx=inputs.topk_idx[tok],
            topk_weights=inputs.topk_weights[tok],
            fc1_weight=inputs.fc1_weight,
            fc2_weight=inputs.fc2_weight,
            output_activation=inputs.output_activation[tok],
        )

    def _validate_inputs(
        self,
        inputs: MegaMoEHopperBf16Inputs,
        *,
        num_tokens: int,
    ) -> None:
        c = self.config
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
        if inputs.activation.dtype != torch.bfloat16:
            raise ValueError(
                f"activation must be bfloat16, got {inputs.activation.dtype}."
            )
        if inputs.activation.stride(-1) != 1:
            raise ValueError(
                "activation must be row-major (stride-1 along hidden); got "
                f"strides {tuple(inputs.activation.stride())}."
            )

        token_tensors = (
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

        weight_checks: Tuple[Tuple[str, torch.Tensor, Tuple[int, ...]], ...] = (
            ("fc1_weight", inputs.fc1_weight, (e, c.hidden, c.fc1_out)),
            ("fc2_weight", inputs.fc2_weight, (e, c.intermediate, c.hidden)),
        )
        for name, tensor, shape in weight_checks:
            _require_cuda(name, tensor)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
                )
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}.")
            # GEMM K must be the stride-1 axis (dim 1 for both legs after the
            # driver's permute).  A plain .contiguous() tensor would silently
            # compute garbage, so reject it here instead of in the kernel.
            if tensor.stride(1) != 1:
                raise ValueError(
                    f"{name} must be K-major (stride-1 along dim 1; got "
                    f"strides {tuple(tensor.stride())}). Permute the logical "
                    "row-major weight instead of calling .contiguous()."
                )
            # TMA needs 16-byte aligned rows along the stride-1 axis.
            if tensor.data_ptr() % 16 != 0:
                raise ValueError(
                    f"{name} data_ptr must be 16-byte aligned "
                    f"(got 0x{tensor.data_ptr():x}); clone the slice."
                )

    @staticmethod
    def _to_cute(tensor: torch.Tensor, assumed_align: int = 16):
        import cutlass.torch as cutlass_torch

        cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        leading_dim = cutlass_torch.get_leading_dim(tensor)
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    @staticmethod
    def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
        """Opaque Uint8 gmem base pointer for the byte workspaces (see hopper_fp8)."""
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
        inputs: MegaMoEHopperBf16Inputs,
        mega: _CompiledMega,
    ) -> dict:
        import cuda.bindings.driver as cuda
        from src.sym_buffer import SymBufferHost

        c = self.config
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        peer_rank_ptr_mapper_host = SymBufferHost(
            base_addr=mega.symmetric_base,
            offsets=tuple(mega.peer_offsets_list),
            rank_idx=c.rank,
            num_max_ranks=c.world_size,
        )

        return dict(
            activation=self._to_cute(inputs.activation),
            topk_idx=self._to_cute(inputs.topk_idx),
            topk_weights=self._to_cute(inputs.topk_weights),
            fc1_weight=self._to_cute(inputs.fc1_weight),
            fc2_weight=self._to_cute(inputs.fc2_weight),
            # Positional kernel argument (Blackwell training ABI): always
            # present, None on the inference path.
            fc1_c=(self._to_cute(mega.fc1_c) if mega.fc1_c is not None else None),
            output_activation=self._to_cute(inputs.output_activation),
            local_workspace=self._to_cute_ptr(mega.local_workspace),
            shared_workspace=self._to_cute_ptr(mega.shared_workspace),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )


# ---------------------------------------------------------------------------
# High-level MegaMoE API (symm buffers + launch + dummy inputs)
# ---------------------------------------------------------------------------

# Kernel-ready weight leg: ONE K-major bf16 tensor (no scale planes).
# fc1: (E, hidden, 2 * intermediate) with hidden stride-1;
# fc2: (E, intermediate, hidden) with intermediate stride-1.
TransformedBf16Weights = torch.Tensor


def init_dist() -> Tuple[int, int]:
    """Initialize torch.distributed + NVSHMEM (or single-rank when ``MEGA_NO_DIST=1``).

    Returns ``(rank, world_size)``.
    """
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


@dataclass
class MegaMoEHopperBf16SymmBuffer:
    """Symmetric-heap staging buffers for one SM90 BF16 MegaMoE session.

    Mirrors :class:`.hopper_fp8.MegaMoEHopperFp8SymmBuffer` without the
    scale plane: exposes ``x`` (bf16), ``topk_idx``, and ``topk_weights``
    views sized for ``num_max_tokens``.  Expert weights are **not** stored
    here -- pass ``transformed_l1`` / ``transformed_l2`` to
    :func:`hopper_bf16_mega_moe` each launch.
    """

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int

    x: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor

    _frontend: MegaMoEHopperBf16Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    @property
    def fc1_c(self) -> Optional[torch.Tensor]:
        """generate_c output of the last launch (``MegaMoEHopperBf16Frontend.fc1_c``)."""
        return self._frontend.fc1_c

    def destroy(self) -> None:
        """Release symmetric-heap allocations and compiled kernel workspaces."""
        if self._destroyed:
            return
        self._frontend.release()
        for root in self._sym_roots:
            free_sym_tensor(root)
        self._sym_roots.clear()
        self._destroyed = True

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size


def get_symm_buffer_for_hopper_bf16_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    # knobs=None resolves the knob cache (offline-tuned winners) and falls
    # back to the heuristic table; a dict applies those knobs directly;
    # "auto" starts from the resolved knobs and re-tunes at the first
    # compute (backend-driven collective sweep).  Mutually exclusive with
    # the explicit geometry arguments below.
    knobs: Optional[Any] = None,
    swap_ab: Optional[bool] = None,
    pingpong: Optional[bool] = None,
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mnk: Optional[Tuple[int, int, int]] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_by_dispatch: bool = False,
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None,
    active_dispatch_warps: int = 1,
    fc1_store_offload: bool = True,
    fc1_early_done_publish: bool = False,
    fold_producer_warps: bool = True,
    generate_c: bool = False,
    apply_topk_in_fc1: bool = True,
    load_balance_mode: Literal["static", "atomic_counter"] = "static",
    # None = the heuristic table's per-bucket value (manual geometry: the
    # kernel default / BF16_TAIL_SPLIT override); an explicit value wins.
    group_hint: Optional[int] = None,
    tail_split_pairs: Optional[bool] = None,
    clc_bundle_size: Optional[int] = None,
    num_sched_stages: Optional[int] = None,
    flag_batch: int = 1,
    epi_flag_batch: Tuple[int, int] = (2, 4),
) -> MegaMoEHopperBf16SymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one SM90 BF16 session.

    Argument order follows the FP8 frontend (problem sizes first).  Pass
    ``rank`` / ``world_size`` from :func:`init_dist`.

    Launch geometry (``swap_ab`` / ``pingpong`` / ``mma_tiler_mnk`` /
    ``cluster_shape_mnk``) follows the drop driver's token heuristics
    (``moe_hopper_bf16/heuristic_config.py``, keyed on ``num_max_tokens``)
    when ALL four are ``None``.  Setting any one of them switches to manual
    mode: unset knobs fall back to the drop driver's defaults
    (``swap_ab=False``, ``pingpong=False``, tiler (64, 128, 64) native /
    (256, 32, 64) swap-AB / (128, 32, 64) swap-AB ping-pong, cluster
    (1, 1, 1)).
    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is
    a deprecated alias for ``gate_up_clamp``.
    ``intermediate`` is the post-SwiGLU width (the kernel's gate+up width is
    ``2 * intermediate``).

    Expert weights are not allocated here; supply the kernel-ready K-major
    bf16 tensors to :func:`hopper_bf16_mega_moe` instead.
    """
    if hidden % _HIDDEN_ALIGN != 0 or intermediate % _INTERMEDIATE_ALIGN != 0:
        raise ValueError(
            f"SM90 BF16 MegaMoE requires hidden % {_HIDDEN_ALIGN} == 0 and "
            f"intermediate % {_INTERMEDIATE_ALIGN} == 0 (got hidden={hidden}, "
            f"intermediate={intermediate})."
        )
    if num_total_experts % world_size != 0:
        raise ValueError("num_total_experts must be divisible by world_size.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    manual_geometry = any(
        value is not None
        for value in (swap_ab, pingpong, mma_tiler_mnk, cluster_shape_mnk)
    )
    knob_overrides: Dict[str, Any] = {}
    if manual_geometry:
        if isinstance(knobs, dict) and knobs:
            raise ValueError(
                "pass either explicit geometry arguments (swap_ab / pingpong "
                "/ mma_tiler_mnk / cluster_shape_mnk) or knobs=, not both."
            )
        # Drop-driver recipe (mega_runner.py main()): manual mode fills the
        # unset geometry knobs with the driver defaults.  heuristic_config
        # imports without cutlass; bootstrap_paths already ran at package
        # import.
        from moe_hopper_bf16.heuristic_config import resolve_hopper_bf16_config

        selection = resolve_hopper_bf16_config(
            num_max_tokens,
            swap_ab=swap_ab,
            pingpong=pingpong,
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
        )
        launch = selection.config
        swap_ab = launch.swap_ab
        pingpong = launch.pingpong
        mma_tiler_mnk = launch.mma_tiler_mnk
        cluster_shape_mnk = launch.cluster_shape_mnk
        if tail_split_pairs is None:
            tail_split_pairs = launch.tail_split_pairs
    else:
        # knobs=None (or "auto"): pure lookup -- offline-tuned cache entry for
        # this session key when present, else the kernel drop's token-bucket
        # heuristic table.  An explicit knobs= dict overrides both entirely;
        # geometry knobs the dict omits keep the table value.
        from .tuner_bf16 import GEOMETRY_KNOBS, default_knobs, resolve_knobs

        geometry = default_knobs(num_max_tokens)
        if isinstance(knobs, dict):
            resolved = dict(knobs)
        else:
            resolved, _ = resolve_knobs(
                world_size=world_size,
                hidden=hidden,
                intermediate=intermediate,
                num_experts=num_total_experts,
                topk=num_topk,
                max_tokens=num_max_tokens,
            )
        # The table row's scheduler companions (group_hint / tail_split_pairs)
        # were tuned together with its geometry: keep them only while the
        # geometry stays on the table row; a cache / dict entry that moves the
        # geometry brings its own values (or the kernel defaults).
        table_defaults = {
            "group_hint": geometry.pop("group_hint"),
            "tail_split_pairs": geometry.pop("tail_split_pairs"),
        }
        cached_geometry = {k: resolved[k] for k in GEOMETRY_KNOBS if k in resolved}
        if any(geometry[k] != v for k, v in cached_geometry.items()):
            table_defaults = {}
        geometry.update(cached_geometry)
        swap_ab = bool(geometry["swap_ab"])
        pingpong = bool(geometry["pingpong"])
        mma_tiler_mnk = tuple(geometry["mma_tiler_mnk"])
        cluster_shape_mnk = tuple(geometry["cluster_shape_mnk"])
        knob_overrides = {
            **table_defaults,
            **{k: v for k, v in resolved.items() if k not in GEOMETRY_KNOBS},
        }
        # An explicit caller choice wins over the heuristic table's / cache's
        # per-bucket pick (token-back placement, scheduler group, tail split).
        if token_back_mode is not None or token_back_by_dispatch:
            knob_overrides.pop("token_back_mode", None)
        if group_hint is not None:
            knob_overrides.pop("group_hint", None)
        if tail_split_pairs is not None:
            knob_overrides.pop("tail_split_pairs", None)

    cfg = MegaMoEHopperBf16Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        swap_ab=swap_ab,
        pingpong=pingpong,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mnk=cluster_shape_mnk,
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_by_dispatch=token_back_by_dispatch,
        token_back_mode=token_back_mode,
        active_dispatch_warps=active_dispatch_warps,
        fc1_store_offload=fc1_store_offload,
        fc1_early_done_publish=fc1_early_done_publish,
        fold_producer_warps=fold_producer_warps,
        generate_c=generate_c,
        apply_topk_in_fc1=apply_topk_in_fc1,
        load_balance_mode=load_balance_mode,
        group_hint=group_hint,
        tail_split_pairs=bool(tail_split_pairs),
        clc_bundle_size=clc_bundle_size,
        num_sched_stages=num_sched_stages,
        flag_batch=flag_batch,
        epi_flag_batch=epi_flag_batch,
    )
    if knob_overrides:
        # Non-geometry knobs from the cache/dict (token_back_mode,
        # load_balance_mode, flag_batch, epi_flag_batch, group_hint, ...).
        from .tuner import with_knobs

        cfg = with_knobs(cfg, knob_overrides)
    frontend = MegaMoEHopperBf16Frontend(cfg)

    sym_roots: list[torch.Tensor] = []
    # nvshmem4py allocates bf16 natively -- no byte-view trick needed.
    x = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(x)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    # The kernel treats -1 as the pad-row mask; zero-filled rows would dispatch
    # as live tokens routed to expert 0.  Stagers overwrite [:n] and re-fill
    # the tail, but start from the masked state so a partial first staging is
    # safe.
    topk_idx.fill_(-1)
    sym_roots.append(topk_idx)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    sym_roots.append(topk_weights)
    # Single 2D (T, hidden) bf16 output on the symmetric heap unconditionally
    # (under in_kernel_fc2_reduce it IS the cross-rank REDG target).
    output_activation = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(output_activation)

    return MegaMoEHopperBf16SymmBuffer(
        num_total_experts=num_total_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        output_activation=output_activation,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def _require_weight_leg(transformed: Any, leg: str) -> torch.Tensor:
    if not isinstance(transformed, torch.Tensor):
        raise ValueError(
            f"transformed_{leg} must be the kernel-ready K-major bf16 weight "
            f"tensor (no scale legs on the BF16 path); got "
            f"{type(transformed).__name__}."
        )
    return transformed


def _build_inputs(
    symm_buffer: MegaMoEHopperBf16SymmBuffer,
    transformed_l1: TransformedBf16Weights,
    transformed_l2: TransformedBf16Weights,
) -> MegaMoEHopperBf16Inputs:
    return MegaMoEHopperBf16Inputs(
        activation=symm_buffer.x,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=_require_weight_leg(transformed_l1, "l1"),
        fc2_weight=_require_weight_leg(transformed_l2, "l2"),
        output_activation=symm_buffer.output_activation,
    )


def hopper_bf16_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedBf16Weights,
    transformed_l2: TransformedBf16Weights,
    symm_buffer: MegaMoEHopperBf16SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch the fused SM90 BF16 MegaMoE kernel (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / routing slices before calling.

    ``transformed_l1`` is the K-major bf16 gate+up weight
    ``(E, hidden, 2 * intermediate)`` (hidden stride-1) with gate/up rows
    interleaved in blocks of 8 (``Fp8GateUpInterleave``, shared with the FP8
    kernel); ``transformed_l2`` is the K-major bf16 down-proj weight
    ``(E, intermediate, hidden)`` (intermediate stride-1).

    ``y`` receives the top-k-reduced bf16 output for ``[:num_tokens]``
    (``None`` returns the zero-copy workspace view instead).
    ``gate_up_clamp`` updates the kernel clamp for this session when set.
    ``activation_clamp`` is a deprecated alias for ``gate_up_clamp``.
    ``fast_math`` is accepted for API parity with the FP8 entry point and has
    no effect here.

    ``sync=False`` (default): the kernel launch and the ``y`` copy are
    enqueued on the current stream and this function returns without a host
    sync.  Pass ``True`` for a blocking call (e.g. host-side timing).
    """
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM90 BF16 MegaMoE path.",
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
    # NOTE: n == 0 is NOT an early return.  The launch is a collective: a rank
    # with no local tokens still serves its experts to the other ranks' pulls
    # and must reach every cross-rank barrier, so it launches the full padded
    # buffer (all topk_idx == -1) like any other rank.  Skipping the launch on
    # the empty rank hangs the peers (observed with in_kernel_fc2_reduce).
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

    # Launch the full padded buffer (topk_idx[n:] == -1 marks the pad rows)
    # and copy the live [:n] rows out -- matches the reference driver.
    out = symm_buffer._frontend.run(inputs, num_tokens=None, sync=False)
    if y is None:
        # Zero-copy: the caller consumes the workspace view under stream
        # ordering (valid until the next launch on this session's buffers).
        result = out[:n]
    else:
        result = None
        y.copy_(out[:n])
    if sync and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
    return result


def hopper_bf16_mega_launch_thunk(
    transformed_l1: TransformedBf16Weights,
    transformed_l2: TransformedBf16Weights,
    symm_buffer: MegaMoEHopperBf16SymmBuffer,
) -> Callable[[], None]:
    """Prebuilt zero-arg SM90 BF16 mega launcher for steady-state timing loops.

    Bare compiled-kernel launch -- args prebuilt once, no per-call Python, no
    workspace reset (the kernel tail-cleans), no sync, no output copy.  The
    reduced bf16 output lands in ``symm_buffer.output_activation``.
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
) -> Tuple[TransformedBf16Weights, TransformedBf16Weights]:
    """Random K-major BF16 weights for local smoke scripts / the tuner.

    Follows the drop driver's perf-run weight assembly
    (``mega_runner.generate_inputs``): dense uniform bf16 payloads, logical
    row-major weights permuted to K-major (never ``.contiguous()``).
    """
    from moe_hopper_bf16.hopper_moe_utils import create_bf16_tensor

    fc1_out = 2 * intermediate  # gate+up width

    def _weight(shape: Tuple[int, ...]) -> torch.Tensor:
        return create_bf16_tensor(
            shape,
            perf_run=True,
            generator=generator,
            perf_positive_only=True,
        )

    # fc1: logical (E, gate+up, hidden) permuted to (E, hidden, gate+up) with
    # hidden stride-1 (K-major); fc2: logical (E, hidden, intermediate)
    # permuted to (E, intermediate, hidden) with intermediate stride-1.
    fc1_weight = _weight((num_local_experts, fc1_out, hidden)).permute(0, 2, 1)
    fc2_weight = _weight((num_local_experts, hidden, intermediate)).permute(0, 2, 1)
    return fc1_weight, fc2_weight


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
    swap_ab: Optional[bool] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedBf16Weights,
    TransformedBf16Weights,
    MegaMoEHopperBf16SymmBuffer,
]:
    """Allocate symm buffer, BF16 weights, and stage activations + routing.

    ``swap_ab=None`` keeps the heuristic launch config (the tuner's starting
    point); an explicit bool switches to manual geometry like the FP8 twin.
    """
    if num_tokens < 0 or num_tokens > num_max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {num_max_tokens}], got {num_tokens}."
        )

    from moe_hopper_bf16.hopper_moe_utils import create_bf16_tensor

    num_local_experts = num_total_experts // world_size
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + rank)

    symm_buffer = get_symm_buffer_for_hopper_bf16_mega_moe(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        swap_ab=swap_ab,
        gate_up_clamp=clamp,
    )

    transformed_l1, transformed_l2 = _create_dummy_weights(
        num_local_experts,
        hidden,
        intermediate,
        gen,
    )

    activation = create_bf16_tensor(
        (num_tokens, hidden),
        perf_run=True,
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

    symm_buffer.x[:num_tokens].copy_(activation)
    symm_buffer.topk_idx[:num_tokens].copy_(topk_idx.to(torch.int64))
    # Mask pad rows (and stale routes from a previous larger staging): the
    # launch covers the full buffer and relies on topk_idx[n:] == -1.
    symm_buffer.topk_idx[num_tokens:].fill_(-1)
    symm_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    y = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    return y, transformed_l1, transformed_l2, symm_buffer


def _main() -> None:
    """Minimal torchrun smoke for the SM90 BF16 MegaMoE thin API."""
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
    SWAP_AB = bool(int(os.environ.get("MEGA_BF16_SWAP_AB", "0")))

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
            swap_ab=SWAP_AB,
            gate_up_clamp=GATE_UP_CLAMP,
            seed=0,
        )

        hopper_bf16_mega_moe(
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
