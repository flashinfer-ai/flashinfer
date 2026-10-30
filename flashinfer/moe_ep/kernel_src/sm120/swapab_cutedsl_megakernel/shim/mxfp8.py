# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile MXFP8 MegaMoE API for ``Sm120MegaMoEMxfp8SwapABKernel``.

API shape follows the SM100 tree's ``shim/mxfp8.py``; fork mechanics (mirrored
ABI constants, derived token tile, native ``token_back_mode`` enum) follow the
SM90 ``shim/hopper_fp8.py`` precedent.  SM120 deltas vs the SM100 frontend:

- the kernel is E4M3-only (``ab_dtype`` is hard-coded in the drop), takes a
  required ``fc2_output_dtype``, and renames ``fc2_in_kernel_topk_reduce`` to
  ``in_kernel_fc2_reduce``;
- weights are consumed K-major (fc1 ``(E, hidden, 2*intermediate)`` with
  hidden stride-1, fc2 ``(E, intermediate, hidden)`` with intermediate
  stride-1) — permuted views of contiguous ``(N, K)`` storage, never
  ``.contiguous()`` after the permute;
- three per-expert epilogue-arg tensors (``fc1_alpha`` / ``fc2_alpha`` /
  ``fc1_norm_const``) join the launch ABI.  The drop's kernel does not apply
  them yet ("a kernel that ignores them validates correctly" — runner
  ``_init_global_scales_and_norm``); the shim pins all-ones so the math stays
  neutral if/when the kernel wires them through;
- the kernel writes a caller-allocated 3-D symmetric ``combine_output``
  ``(T, num_topk, hidden)`` (or ``(T, 1, hidden)`` under REDG) instead of
  reducing top-k internally; form A needs a second ``topk_reduce`` launch;
- the kernel does not tail-clean its local counters, so every launch is
  preceded by :func:`~.comm.zero_local_counter_regions`.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple  # noqa: F401

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
    zero_local_counter_regions,
)

# The SM120 kernel hard-codes ab_dtype = Float8E4M3FN (no e5m2 selection at the
# mega level, unlike the SM100 drop).
_KIND_TO_TORCH_DTYPE = {
    "mxfp8_e4m3": torch.float8_e4m3fn,
}

# Mirrored ABI constants.  This drop's ``common.megamoe_constants`` imports
# cutlass at module load, so the CPU-safe layers mirror the values and
# ``_assert_mirrored_constants`` (run at first compile) fails loudly if a new
# drop moves them (SM90-shim precedent).
_MXFP8_BLOCK_SIZE = 32  # common.megamoe_constants.Mxfp8BlockSize
_SF_PADDING_BLOCK = 128  # common.megamoe_constants.SfPaddingBlock
_CTA_TOKEN_TILE = 64  # moe_sm120_mxfp8_swapab.sm120_mma.CTA_TOKEN_TILE
_SWAP_AB_INTERLEAVE = 8  # moe_sm120_mxfp8_swapab.sm120_mma.SWAP_AB_INTERLEAVE

# Swap-AB GEMM-domain tile constraints (runner_fc12_common validation): M must
# be 64; N ∈ {32, 64, 128}, but the C3 pool constraint (cluster_tile_tokens =
# N * cluster_n must be a multiple of token_padding_block = 64, with cluster_n
# pinned to 1) rules out N = 32.
_TILE_M_CHOICES = (64,)
_TILE_N_CHOICES = (64, 128)

_TOKEN_BACK_MODES = ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
_DISPATCH_PULL_MODES = ("auto", "token_strided", "tile_cooperative")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


@dataclasses.dataclass(frozen=True)
class MegaMoESm120Mxfp8Config:
    """Compile-time / launch-time SM120 swap-AB MXFP8 MegaMoE configuration.

    ``intermediate`` is the post-SwiGLU width. The kernel's full FC1 gate+up
    width is derived as ``2 * intermediate``.
    """

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int

    kind: Literal["mxfp8_e4m3"] = "mxfp8_e4m3"
    mma_tiler_mnk: Tuple[int, int, int] = (64, 128, 128)
    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (1, 1)
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    non_ubulk_fc2_store: bool = True
    gate_up_clamp: Optional[float] = None
    dispatch_pull_mode: Literal["auto", "token_strided", "tile_cooperative"] = "auto"
    dispatch_warps_per_tile: int = 8
    dispatch_compute_overlap: Optional[bool] = None
    enable_iket: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(
                f"kind must be one of {sorted(_KIND_TO_TORCH_DTYPE)}, "
                f"got {self.kind!r} (the SM120 kernel is E4M3-only)."
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
        # Runner-mirrored problem constraints: hidden % 32 (MXFP8 SF blocks),
        # intermediate % 32 (=> gate+up width % 64, per-32 fc1-out SF blocks,
        # and the swap-AB register interleave of 8 divides evenly).
        if self.hidden % _MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"hidden must be a multiple of {_MXFP8_BLOCK_SIZE}, got {self.hidden}."
            )
        if self.intermediate % _MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"intermediate must be a multiple of {_MXFP8_BLOCK_SIZE}, "
                f"got {self.intermediate}."
            )
        m, n, _k = self.mma_tiler_mnk
        if m not in _TILE_M_CHOICES or n not in _TILE_N_CHOICES:
            raise ValueError(
                "SM120 swap-AB MXFP8 MegaMoE supports mma_tiler (M, N) with "
                f"M in {_TILE_M_CHOICES} and N in {_TILE_N_CHOICES}; "
                f"got mma_tiler_mnk={self.mma_tiler_mnk}."
            )
        # The kernel ctor nominally accepts cluster_m in [1, 16], but the
        # drop never tests > 1 and its own runner fails to compile there
        # ("expects num_multicast to be 1 for non multicast G2S copies",
        # verified 2026-08-06 on sm_120 / DSL 4.6.1) — see VENDOR.md.  Loosen
        # when a drop validates multi-CTA clusters.
        if self.cluster_shape_mnk != (1, 1, 1):
            raise ValueError(
                "cluster_shape_mnk must be (1, 1, 1) on the current SM120 "
                f"drop (multi-CTA clusters do not compile upstream); "
                f"got {self.cluster_shape_mnk}."
            )
        # C3 pool constraint: cluster_tile_tokens (= N * cluster_n under
        # swap-AB; cluster_n is pinned to 1 above) must be a multiple of the
        # token padding block (64).
        if n % _CTA_TOKEN_TILE != 0:
            raise ValueError(
                f"mma_tiler N ({n}) must be a multiple of the token padding "
                f"block ({_CTA_TOKEN_TILE})."
            )
        if self.token_back_mode not in _TOKEN_BACK_MODES:
            raise ValueError(
                f"token_back_mode must be one of {_TOKEN_BACK_MODES}, "
                f"got {self.token_back_mode!r}."
            )
        if self.in_kernel_fc2_reduce and self.token_back_mode != "epi_warps":
            raise ValueError(
                "in_kernel_fc2_reduce requires token_back_mode='epi_warps' "
                f"(got {self.token_back_mode!r})."
            )
        if self.token_back_mode != "epi_warps" and not self.non_ubulk_fc2_store:
            raise ValueError(
                "non-epi token_back_mode requires non_ubulk_fc2_store=True "
                "(the workspace-staged push-back path has no UBLK store)."
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
        if self.flag_batch < 1:
            raise ValueError(f"flag_batch must be >= 1, got {self.flag_batch}.")
        eb = self.epi_flag_batch
        if len(eb) != 2:
            raise ValueError(
                f"epi_flag_batch must be a (fc1, fc2) pair, got {self.epi_flag_batch}."
            )
        for leg, val in (("fc1", eb[0]), ("fc2", eb[1])):
            if val < 1 or val > 32:
                raise ValueError(
                    f"epi_flag_batch[{leg}] must be in [1, 32], got {val}."
                )
        if self.dispatch_pull_mode not in _DISPATCH_PULL_MODES:
            raise ValueError(
                f"dispatch_pull_mode must be one of {_DISPATCH_PULL_MODES}, "
                f"got {self.dispatch_pull_mode!r}."
            )
        if self.dispatch_warps_per_tile not in (4, 8, 16, 32):
            raise ValueError(
                "dispatch_warps_per_tile must be one of (4, 8, 16, 32), "
                f"got {self.dispatch_warps_per_tile}."
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
    def combine_k(self) -> int:
        return 1 if self.in_kernel_fc2_reduce else self.num_topk


@dataclasses.dataclass
class MegaMoESm120Mxfp8Inputs:
    """Per-rank tensors for one SM120 MXFP8 MegaMoE launch.

    Weights are K-major permuted views: ``fc1_weight`` ``(E, hidden, 2I)``
    with ``stride(1) == 1``, ``fc2_weight`` ``(E, I, hidden)`` with
    ``stride(1) == 1``.  ``output_activation`` is the final reduced 2-D
    ``(T, hidden)`` bf16 buffer this frontend fills (the kernel's own output
    is the frontend-owned 3-D ``combine_output``).
    """

    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    output_activation: torch.Tensor


class MegaMoESm120Mxfp8Frontend:
    """Lazy-compile host wrapper for ``Sm120MegaMoEMxfp8SwapABKernel``."""

    def __init__(self, config: MegaMoESm120Mxfp8Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoESm120Mxfp8Config:
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

    def apply_knobs(self, knobs: Optional[dict]) -> None:
        """Apply tuning-knob overrides (config field names) to the session.

        Invalidates the compile cache when the effective config changes; the
        next ``run()``/``warmup()`` recompiles.  ``token_back_mode`` is a
        native config field here, so the knob vocabulary passes straight
        through.  Unknown keys raise (this tree has no cross-dtype knob
        vocabulary to silently subset).
        """
        if not knobs:
            return
        fields = {f.name for f in dataclasses.fields(self._config)}
        unknown = sorted(set(knobs) - fields)
        if unknown:
            raise ValueError(f"unknown knob(s) for SM120 MXFP8 config: {unknown}")
        new_config = dataclasses.replace(self.config, **knobs)
        if new_config == self._config:
            return
        ensure_not_capturing("apply_knobs (config change)")
        self._release_workspace()
        self._config = new_config
        self._invalidate_compile_cache()

    def release(self) -> None:
        self._release_workspace()
        self._invalidate_compile_cache()

    def warmup(
        self,
        inputs: MegaMoESm120Mxfp8Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> None:
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return None
        self._ensure_mega_compiled(inputs)

    def run(
        self,
        inputs: MegaMoESm120Mxfp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = False,
        reduce_topk: bool = True,
    ) -> Optional[torch.Tensor]:
        """Launch SM120 MXFP8 MegaMoE and return the 2D ``(T, hidden)`` bf16 output.

        Per-launch sequence (all stream-ordered, CUDA-graph capturable):
        local counter zero (this drop's kernel does not tail-clean) →
        [ikr: combine zero (REDG accumulate-from-zero)] → mega kernel →
        [form A: compiled topk reduce | ikr: combine→output copy].

        ``reduce_topk`` is accepted for API parity with the SM100 frontend and
        ignored (the reduce is always performed into ``output_activation``).
        ``reset_counters=True`` additionally runs the recovery reset for
        workspaces left dirty by an aborted launch.

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
            # Any config change (apply_knobs / set_gate_up_clamp) nulls
            # self._mega, so a live cache entry always matches the config.
            launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
            if launch_inputs is None:
                return None
            mega = self._ensure_mega_compiled(inputs)
            mega.launch_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
            if not self.config.in_kernel_fc2_reduce:
                mega.reduce_kwargs = self._build_reduce_kwargs(launch_inputs, mega)
            mega.launch_key = key
            mega.launch_output = launch_inputs.output_activation
        if reset_counters:
            reset_compiled_mega_workspaces(mega)
        self._launch(mega)
        # Zero-break capture gate: a device synchronize would abort stream
        # capture, so skip it there (the graph replays under stream semantics).
        if sync and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return mega.launch_output

    def _launch(self, mega: _CompiledMega) -> None:
        zero_local_counter_regions(mega)
        if self.config.in_kernel_fc2_reduce:
            # REDG accumulate-from-zero contract: combine_output is the
            # cross-rank atomic-add target, so it must be zeroed pre-launch.
            mega.combine_output.zero_()
        mega.compiled(**mega.launch_kwargs)
        if self.config.in_kernel_fc2_reduce:
            # (T, 1, hidden) -> (T, hidden); strided device copy, capturable.
            mega.launch_output.copy_(mega.combine_output.squeeze(1))
        else:
            mega.reduce_compiled(**mega.reduce_kwargs)

    def make_launch_thunk(
        self,
        inputs: MegaMoESm120Mxfp8Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> Callable[[], None]:
        """Zero-arg launcher with args prebuilt (compiles if needed).

        Steady-state fast path for timing loops and tuners: no per-call Python
        arg rebuild, no validation, no sync.  Unlike the SM100 thunk this is
        not a bare kernel launch — the SM120 pre-launch counter zero and the
        post-launch reduce/copy are part of the per-launch contract, so the
        thunk enqueues the same stream-ordered node sequence as ``run()``.
        Output lands in ``inputs.output_activation``.  Invalid after the
        compile cache is invalidated (knobs/clamp change) or buffers are freed.
        """
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return lambda: None
        mega = self._ensure_mega_compiled(inputs)
        mega.launch_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
        if not self.config.in_kernel_fc2_reduce:
            mega.reduce_kwargs = self._build_reduce_kwargs(launch_inputs, mega)
        mega.launch_output = launch_inputs.output_activation
        mega.launch_key = self._launch_cache_key(
            inputs, self._resolve_num_tokens(inputs, num_tokens)
        )

        def thunk() -> None:
            self._launch(mega)

        return thunk

    @staticmethod
    def _launch_cache_key(inputs: MegaMoESm120Mxfp8Inputs, num_tokens: int) -> tuple:
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
            t.fc2_weight.data_ptr(),
            t.fc2_weight_sf.data_ptr(),
            t.output_activation.data_ptr(),
            num_tokens,
            torch.cuda.current_stream().cuda_stream,
        )

    def _mega_compile_key(self) -> tuple:
        c = self.config
        return (
            c.kind,
            c.world_size,
            c.rank,
            c.num_tokens_per_rank,
            c.num_topk,
            c.num_total_experts,
            c.hidden,
            c.intermediate,
            c.mma_tiler_mnk,
            c.cluster_shape_mnk,
            c.load_balance_mode,
            c.group_hint,
            c.flag_batch,
            c.epi_flag_batch,
            c.in_kernel_fc2_reduce,
            c.token_back_mode,
            c.non_ubulk_fc2_store,
            c.dispatch_pull_mode,
            c.dispatch_warps_per_tile,
            c.dispatch_compute_overlap,
            self._gate_up_clamp,
            c.enable_iket,
        )

    @staticmethod
    def _assert_mirrored_constants() -> None:
        """Fail loudly if a kernel drop moved the mirrored ABI constants."""
        from common.megamoe_constants import Mxfp8BlockSize, SfPaddingBlock
        from moe_sm120_mxfp8_swapab.sm120_mma import (
            CTA_TOKEN_TILE,
            SWAP_AB_INTERLEAVE,
        )

        mirrors = (
            ("Mxfp8BlockSize", Mxfp8BlockSize, _MXFP8_BLOCK_SIZE),
            ("SfPaddingBlock", SfPaddingBlock, _SF_PADDING_BLOCK),
            ("CTA_TOKEN_TILE", CTA_TOKEN_TILE, _CTA_TOKEN_TILE),
            ("SWAP_AB_INTERLEAVE", SWAP_AB_INTERLEAVE, _SWAP_AB_INTERLEAVE),
        )
        for name, source_val, mirror_val in mirrors:
            if int(source_val) != mirror_val:
                raise RuntimeError(
                    f"kernel drop changed {name} ({source_val} != mirrored "
                    f"{mirror_val}); update the mirrors in shim/mxfp8.py and "
                    "re-audit staging-buffer shapes and weight preprocessing."
                )

    def _ensure_mega_compiled(self, inputs: MegaMoESm120Mxfp8Inputs) -> _CompiledMega:
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        ensure_not_capturing("cute.compile + symmetric-heap allocation")
        self._release_workspace()
        self._assert_mirrored_constants()

        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils

        from common.megamoe_constants import Mxfp8BlockSize, SfPaddingBlock
        from moe_sm120_mxfp8_swapab.megamoe_kernel import (
            Sm120MegaMoEMxfp8SwapABKernel,
        )
        from moe_sm120_mxfp8_swapab.sm120_mma import CTA_TOKEN_TILE

        c = self.config
        static_expert_shape = (
            c.num_experts_per_rank,
            c.fc1_out,
            c.hidden,
        )

        # Driver recipe: occupancy-aware count from the DSL (the SM120 runner
        # uses HardwareInfo, not the SM100 shim's sm_count // cluster_size
        # heuristic).
        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            max(cluster_size, 1)
        )
        group_hint = c.group_hint if c.group_hint is not None else max_active_clusters

        kernel = Sm120MegaMoEMxfp8SwapABKernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            # The SM120 warp-level MMA path has no 2-CTA mode.
            use_2cta_instrs=False,
            group_hint=group_hint,
            token_padding_block=CTA_TOKEN_TILE,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=True,
            sf_vec_size=Mxfp8BlockSize,
            world_size=c.world_size,
            local_rank=c.rank,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_output_dtype=cutlass.BFloat16,
            non_ubulk_fc2_store=c.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=c.in_kernel_fc2_reduce,
            token_back_mode=c.token_back_mode,
            # Mirrors the kernel-team driver: top-k weighting is applied by the
            # fc2 epilogue via the dispatched topk_scores, not folded into fc1
            # (the torch reference wrapper models the fc1-folded equivalent).
            apply_topk_in_fc1=False,
            gate_up_clamp=self._gate_up_clamp,
            epi_flag_batch=c.epi_flag_batch,
            flag_batch=c.flag_batch,
            dispatch_pull_mode=c.dispatch_pull_mode,
            dispatch_warps_per_tile=c.dispatch_warps_per_tile,
            dispatch_compute_overlap=c.dispatch_compute_overlap,
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

        # The SM120 shared workspace holds only the dispatch/barrier regions;
        # combine staging is a separate caller-allocated symmetric tensor
        # (peer-written under token-back / REDG), owned per-compile because
        # its K axis depends on in_kernel_fc2_reduce.
        combine_output, combine_root = _sym_zeros_byte_view_1b(
            (c.num_tokens_per_rank, c.combine_k, c.hidden),
            torch.bfloat16,
        )

        # Per-expert epilogue args.  The drop's kernel currently ignores them;
        # all-ones keeps the math neutral if a future drop wires them through
        # (runner ``_init_global_scales_and_norm`` pins exactly this).
        ones = torch.ones((c.num_experts_per_rank,), dtype=torch.float32, device="cuda")
        mega = _CompiledMega(
            compiled=None,
            kernel=kernel,
            local_workspace=local_workspace,
            shared_workspace=shared_workspace,
            symmetric_base=symmetric_base,
            peer_offsets_list=peer_offsets_list,
            combine_output=combine_output,
            combine_root=combine_root,
            fc1_alpha=ones,
            fc2_alpha=ones.clone(),
            fc1_norm_const=ones.clone(),
        )
        compile_kwargs = self._build_mega_runtime_kwargs(inputs, mega)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if c.enable_iket:
            compile_kwargs["options"] = "iket"

        mega.compiled = cute.compile(kernel, **compile_kwargs)

        if not c.in_kernel_fc2_reduce:
            # Form A: the kernel leaves (T, K, hidden) bf16 terms in
            # combine_output; a compiled second launch reduces them into the
            # 2-D output buffer (topk weights were already applied in-kernel,
            # so topk_score stays None — deepgemm reference graph).
            import cuda.bindings.driver as cuda_driver

            from moe_sm120_mxfp8_swapab.topk_reduce import compile_topk_reduce

            stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
            (
                reduce_compiled,
                _combine_cute,
                _reduced_cute,
                _topk_score_cute,
                _reduce_stream,
            ) = compile_topk_reduce(
                combine_output,
                inputs.output_activation,
                None,
                stream=stream,
            )
            mega.reduce_compiled = reduce_compiled
            # Runtime kwargs are (re)built per launch-cache miss so a caller
            # swapping output buffers or streams retargets the reduce too.

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
            free_sym_tensor(self._mega.combine_root)

    @staticmethod
    def _resolve_num_tokens(
        inputs: MegaMoESm120Mxfp8Inputs,
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
        inputs: MegaMoESm120Mxfp8Inputs,
        *,
        num_tokens: Optional[int],
    ) -> Optional[MegaMoESm120Mxfp8Inputs]:
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        if resolved == 0:
            return None
        self._validate_inputs(inputs, num_tokens=resolved)
        buf_tokens = inputs.activation.shape[0]
        if resolved < buf_tokens:
            # The kernel, the combine buffer, and the compiled topk reduce are
            # all shaped for the full session buffer; padded rows are masked
            # by topk_idx == -1, so a partial launch is always full-buffer.
            raise ValueError(
                "Partial num_tokens is not supported (kernel and topk reduce "
                f"compile for the full buffer of {buf_tokens} tokens; pad and "
                f"mask with topk_idx == -1 instead). Got num_tokens={resolved}."
            )
        return inputs

    def _validate_inputs(
        self,
        inputs: MegaMoESm120Mxfp8Inputs,
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

        hidden_sf_cols = _ceil_div(c.hidden, _MXFP8_BLOCK_SIZE)
        if inputs.activation_sf.dtype != torch.float8_e8m0fnu:
            raise ValueError(
                "activation_sf must have dtype torch.float8_e8m0fnu, "
                f"got {inputs.activation_sf.dtype}."
            )
        if inputs.activation_sf.shape[-1] % 4 != 0:
            raise ValueError(
                f"activation_sf.shape[-1] ({inputs.activation_sf.shape[-1]}) "
                "must be a multiple of 4."
            )
        if inputs.activation_sf.shape[-1] < hidden_sf_cols:
            raise ValueError(
                f"activation_sf.shape[-1] ({inputs.activation_sf.shape[-1]}) "
                f"must be >= {hidden_sf_cols} (hidden={c.hidden})."
            )

        # Weights are K-major permuted views: the GEMM K axis (dim 1) carries
        # stride 1.  A contiguous (SM100-layout) tensor here would silently
        # feed transposed data to the TMA descriptors.
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
            if tensor.stride(1) != 1:
                raise ValueError(
                    f"{name} must be K-major (stride(1) == 1, a permuted view "
                    f"of contiguous (experts, N, K) storage); got strides "
                    f"{tuple(tensor.stride())}."
                )

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

    @staticmethod
    def _to_cute(
        tensor: torch.Tensor,
        assumed_align: int = 16,
        *,
        static_layout: bool = False,
    ):
        import cutlass.torch as cutlass_torch

        cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        if static_layout:
            return cute_tensor
        leading_dim = cutlass_torch.get_leading_dim(tensor)
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    def _build_mega_runtime_kwargs(
        self,
        inputs: MegaMoESm120Mxfp8Inputs,
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
            fc2_weight=self._to_cute(inputs.fc2_weight),
            fc2_weight_sf=self._to_cute(inputs.fc2_weight_sf),
            fc1_alpha=self._to_cute(mega.fc1_alpha, assumed_align=4),
            fc2_alpha=self._to_cute(mega.fc2_alpha, assumed_align=4),
            fc1_norm_const=self._to_cute(mega.fc1_norm_const, assumed_align=4),
            combine_output=self._to_cute(mega.combine_output),
            local_workspace=self._to_cute(
                mega.local_workspace,
                static_layout=True,
            ),
            shared_workspace=self._to_cute(mega.shared_workspace),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )

    def _build_reduce_kwargs(
        self,
        inputs: MegaMoESm120Mxfp8Inputs,
        mega: _CompiledMega,
    ) -> dict:
        # Mirror compile_topk_reduce's own tensor construction so the runtime
        # views match the compiled launcher's sample-arg layouts.
        import cuda.bindings.driver as cuda

        from moe_sm120_mxfp8_swapab.topk_reduce import _to_cute_tensor

        return dict(
            combine_cute=_to_cute_tensor(mega.combine_output),
            reduced_cute=_to_cute_tensor(inputs.output_activation),
            topk_score_cute=None,
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


# ---------------------------------------------------------------------------
# High-level MegaMoE API (symm buffers + launch + dummy inputs)
# ---------------------------------------------------------------------------

TransformedWeights = Tuple[torch.Tensor, torch.Tensor]

Sm120Mxfp8Kind = Literal["mxfp8_e4m3"]


def _sym_zeros_byte_view_1b(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-byte / 2-byte dtype symmetric heap via uint8 reinterpret.

    nvshmem4py has no fp8 / E8M0 dtype (and views keep bf16 uniform), so
    allocate uint8 and reinterpret.  Returns ``(view, root_uint8_buffer)``;
    free the root via :func:`free_sym_tensor`.
    """
    total_bytes = target_dtype.itemsize
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
class MegaMoESm120Mxfp8SymmBuffer:
    """Symmetric-heap staging buffers for one SM120 MXFP8 MegaMoE session.

    Mirrors the SM100 :class:`MegaMoEMxfp8SymmBuffer`: exposes ``x``,
    ``x_sf``, ``topk_idx``, and ``topk_weights`` views sized for
    ``num_max_tokens``, plus the reduced ``output_activation``.

    Expert weights are **not** stored here — pass ``transformed_l1`` /
    ``transformed_l2`` to :func:`sm120_mxfp8_mega_moe` each launch.
    """

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int
    kind: Sm120Mxfp8Kind

    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor

    _frontend: MegaMoESm120Mxfp8Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

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


def get_symm_buffer_for_sm120_mxfp8_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: Sm120Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps",
    knobs: Optional[dict] = None,
) -> MegaMoESm120Mxfp8SymmBuffer:
    """Allocate symmetric-heap inputs for one SM120 MXFP8 MegaMoE session.

    Argument order follows the SM100 frontend (problem sizes first).  Pass
    ``rank`` / ``world_size`` from :func:`init_dist`.

    ``kind`` is E4M3-only on this kernel.  ``gate_up_clamp`` sets the kernel
    gate-up clamp; ``activation_clamp`` is a deprecated alias.
    ``intermediate`` is the post-SwiGLU width, matching the other trees.
    ``knobs`` is an optional dict of config-field overrides (e.g.
    ``mma_tiler_mnk``, ``flag_batch``); this tree has no knob cache or
    heuristic tables yet, so ``None`` means kernel defaults.

    Expert weights are not allocated here; supply kernel-ready ``(weight, scale)``
    tuples to :func:`sm120_mxfp8_mega_moe` instead.
    """
    if num_total_experts % world_size != 0:
        raise ValueError("num_total_experts must be divisible by world_size.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    cfg = MegaMoESm120Mxfp8Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        kind=kind,
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
    )
    if knobs:
        fields_ = {f.name for f in dataclasses.fields(cfg)}
        unknown = sorted(set(knobs) - fields_)
        if unknown:
            raise ValueError(f"unknown knob(s) for SM120 MXFP8 config: {unknown}")
        overrides = dict(knobs)
        # Caller-owned correctness choices win over knob dicts (matches the
        # SM100 factory's re-pin of in_kernel_fc2_reduce).
        overrides.pop("in_kernel_fc2_reduce", None)
        cfg = dataclasses.replace(cfg, **overrides)
    frontend = MegaMoESm120Mxfp8Frontend(cfg)

    hidden_sf_cols = _ceil_div(hidden, _MXFP8_BLOCK_SIZE)
    hidden_sf_cols_padded = _round_up(hidden_sf_cols, 4)
    data_dtype = cfg.torch_ab_dtype

    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view_1b((num_max_tokens, hidden), data_dtype)
    sym_roots.append(x_root)
    x_sf, x_sf_root = _sym_zeros_byte_view_1b(
        (num_max_tokens, hidden_sf_cols_padded),
        torch.float8_e8m0fnu,
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
    # Reduced 2-D (T, hidden) bf16 output.  The kernel's own write target is
    # the frontend-owned symmetric combine_output; this buffer is filled by
    # the topk-reduce second stage (form A) or a combine copy (ikr), both
    # rank-local, so a plain CUDA allocation would do — sym keeps the
    # allocation lifecycle uniform with the sibling trees.
    output_activation = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(output_activation)

    return MegaMoESm120Mxfp8SymmBuffer(
        num_total_experts=num_total_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        kind=kind,
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        output_activation=output_activation,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def sm120_mxfp8_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoESm120Mxfp8SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Launch the fused SM120 CuTeDSL MXFP8 MegaMoE (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / routing slices before calling.

    ``transformed_l1`` / ``transformed_l2`` are ``(weight, scale)`` tuples in
    the **kernel-ready** layout: K-major permuted fp8 weight views plus
    32x4x4 atom-swizzled flat 2-D scale factors (see the backend's
    ``preprocess_mega_weights``).  Weights are always caller-supplied here —
    they are not owned by the symm buffer.

    ``y`` receives the top-k-reduced bf16 output for ``[:num_tokens]``.
    ``gate_up_clamp`` updates the kernel clamp for this session when set.
    ``activation_clamp`` is a deprecated alias for ``gate_up_clamp``.
    ``fast_math`` is accepted for DeepGEMM API parity and has no effect here.

    ``sync=False`` (default): the kernel pipeline and the ``y`` copy are
    enqueued on the current stream and this function returns without a host
    sync — ``y`` is ready under normal stream semantics.  Pass ``True`` for a
    blocking call (e.g. host-side timing).
    """
    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM120 MXFP8 MegaMoE path.",
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
    if n == 0:
        return symm_buffer.output_activation[:0] if y is None else None
    if y is not None:
        if y.shape != (n, symm_buffer.hidden):
            raise ValueError(
                f"y must be ({n}, {symm_buffer.hidden}), got {tuple(y.shape)}."
            )
        if y.dtype != torch.bfloat16:
            raise ValueError(f"y must be bfloat16, got {y.dtype}.")

    fc1_weight, fc1_weight_sf = transformed_l1
    fc2_weight, fc2_weight_sf = transformed_l2

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)

    inputs = MegaMoESm120Mxfp8Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        output_activation=symm_buffer.output_activation,
    )

    # Launch the full padded buffer (topk_idx[n:] == -1 marks the pad rows)
    # and copy the live [:n] rows out — matches the reference driver.
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


def sm120_mxfp8_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoESm120Mxfp8SymmBuffer,
) -> Callable[[], None]:
    """Prebuilt zero-arg SM120 MXFP8 mega launcher for steady-state timing loops.

    The thunk enqueues the full per-launch node sequence (counter zero →
    kernel → reduce/copy) with args prebuilt once — no per-call Python arg
    rebuild, no validation, no sync, no output copy.  The reduced bf16 output
    lands in ``symm_buffer.output_activation``.  Compiles on this call if
    needed.  Rebuild the thunk after knob/clamp changes or buffer destruction.
    """
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    fc1_weight, fc1_weight_sf = transformed_l1
    fc2_weight, fc2_weight_sf = transformed_l2
    inputs = MegaMoESm120Mxfp8Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        output_activation=symm_buffer.output_activation,
    )
    return symm_buffer._frontend.make_launch_thunk(inputs)


def _create_dummy_weights(
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
    *,
    kind: Sm120Mxfp8Kind,
) -> Tuple[TransformedWeights, TransformedWeights]:
    """Random MXFP8 weights + swizzled SF in the SM120 K-major layout."""
    from moe_sm120_mxfp8_swapab.mega_runner import (
        _make_e8m0_scale_tensor,
        _make_fp8_tensor,
    )
    from moe_sm120_mxfp8_swapab.runner_common import (
        _stack_byte_reinterpretable_tensors,
        to_blocked,
    )

    data_dtype = _KIND_TO_TORCH_DTYPE[kind]

    fc1_out = 2 * intermediate
    hidden_sf_cols = _ceil_div(hidden, _MXFP8_BLOCK_SIZE)
    intermediate_sf_cols = _ceil_div(intermediate, _MXFP8_BLOCK_SIZE)

    # Contiguous (E, N, K) storage, permuted to the kernel's K-major logical
    # (E, K, N) views — never .contiguous() after the permute.
    fc1_weight = _make_fp8_tensor(
        generator,
        (num_local_experts, fc1_out, hidden),
        data_dtype,
        perf_run=True,
    ).permute(0, 2, 1)
    fc1_weight_sf_plain = _make_e8m0_scale_tensor(
        generator,
        num_local_experts * fc1_out,
        hidden,
        blocksize=_MXFP8_BLOCK_SIZE,
    ).reshape(num_local_experts, fc1_out, hidden_sf_cols)
    fc1_sf_swizzled = [
        to_blocked(fc1_weight_sf_plain[e]) for e in range(num_local_experts)
    ]
    fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_local_experts, fc1_flat_sf_size
    )

    fc2_weight = _make_fp8_tensor(
        generator,
        (num_local_experts, hidden, intermediate),
        data_dtype,
        perf_run=True,
    ).permute(0, 2, 1)
    fc2_weight_sf_plain = _make_e8m0_scale_tensor(
        generator,
        num_local_experts * hidden,
        intermediate,
        blocksize=_MXFP8_BLOCK_SIZE,
    ).reshape(num_local_experts, hidden, intermediate_sf_cols)
    fc2_sf_swizzled = [
        to_blocked(fc2_weight_sf_plain[e]) for e in range(num_local_experts)
    ]
    fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
    fc2_weight_sf = _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0).view(
        num_local_experts, fc2_flat_sf_size
    )

    return (fc1_weight, fc1_weight_sf), (fc2_weight, fc2_weight_sf)


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
    kind: Sm120Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedWeights,
    TransformedWeights,
    MegaMoESm120Mxfp8SymmBuffer,
]:
    """Allocate symm buffer, MXFP8 weights, and stage activations + routing."""
    if num_tokens < 0 or num_tokens > num_max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {num_max_tokens}], got {num_tokens}."
        )

    num_local_experts = num_total_experts // world_size
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + rank)

    symm_buffer = get_symm_buffer_for_sm120_mxfp8_mega_moe(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        kind=kind,
        gate_up_clamp=clamp,
    )

    transformed_l1, transformed_l2 = _create_dummy_weights(
        num_local_experts,
        hidden,
        intermediate,
        gen,
        kind=kind,
    )

    from moe_sm120_mxfp8_swapab.mega_runner import (
        _make_e8m0_scale_tensor,
        _make_fp8_tensor,
    )

    data_dtype = symm_buffer._frontend.config.torch_ab_dtype
    activation = _make_fp8_tensor(
        gen,
        (num_tokens, hidden),
        data_dtype,
        perf_run=True,
    )
    activation_sf = _make_e8m0_scale_tensor(
        gen,
        num_tokens,
        hidden,
        blocksize=_MXFP8_BLOCK_SIZE,
    ).reshape(num_tokens, _ceil_div(hidden, _MXFP8_BLOCK_SIZE))

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
    hidden_sf_cols = _ceil_div(hidden, _MXFP8_BLOCK_SIZE)
    symm_buffer.x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(
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
    """Minimal torchrun smoke for the SM120 MXFP8 MegaMoE thin API."""
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 or not bool(int(os.environ.get("MEGA_NO_DIST", "0"))):
        # Fold onto the physical GPUs (rank-sharing single-GPU boxes; the
        # kernel drop's bootstrap does the same).
        torch.cuda.set_device(local_rank % max(torch.cuda.device_count(), 1))

    HIDDEN = 2048
    INTERMEDIATE = 1024
    NUM_TOKENS = 128
    NUM_MAX_TOKENS = 128
    NUM_TOPK = 4
    NUM_EXPERTS = 32
    GATE_UP_CLAMP = 10.0

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
            gate_up_clamp=GATE_UP_CLAMP,
            seed=0,
        )

        sm120_mxfp8_mega_moe(
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
