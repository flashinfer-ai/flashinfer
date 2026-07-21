# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile NVFP4 MegaMoE API for ``Sm100MegaMoEKernel``.

NVFP4 activations and NVFP4 expert weights (fp8 plain scale-factor planes).
Compiles on first ``run()`` (or explicit ``warmup()``) and reuses the compiled
launchers for subsequent calls with the same :class:`MegaMoENvfp4Config` and tensor
shapes.  Input generation and reference checks live in
``moe_nvfp4_swapab.mega_runner``.

Example (multi-rank; launch with ``torchrun``)::

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import (
        MegaMoENvfp4Config,
        MegaMoENvfp4Frontend,
        MegaMoENvfp4Inputs,
        bootstrap_dist,
    )

    local_rank, rank, world_size, _ = bootstrap_dist()
    cfg = MegaMoENvfp4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=256,
        num_topk=4,
        num_total_experts=32,
        hidden=2048,
        intermediate=1024,
    )
    runner = MegaMoENvfp4Frontend(cfg)
    # ... allocate sym-heap + local tensors ...
    out = runner.run(MegaMoENvfp4Inputs(...))

Single-rank smoke (no NVSHMEM)::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 python -u \\
        -m flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim \\
        --num_tokens_per_rank 128 ...
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import torch

from .comm import (
    _CompiledMega,
    _compute_peer_offsets,
    bootstrap_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from common.megamoe_constants import Nvfp4BlockSize
from moe_nvfp4_swapab.runner_common import (
    _DataDtype,
    _ScaleDtype,
    ceil_div,
    round_up,
)


# Config combine_dtype -> kernel CombineFormat spelling.  The quantized wire
# formats shrink the cross-rank combine traffic 2x (fp8+e8m0 SF) / 4x
# (fp4+bf16 SF) at a numerics cost (see MegaMoENvfp4Config.combine_dtype).
COMBINE_FORMAT_NAMES = {
    "bf16": "bf16",
    "mxfp8": "32e4m3xe8m0",
    "nvfp4": "16e2m1xbf16",
}


@dataclasses.dataclass(frozen=True)
class MegaMoENvfp4Config:
    """Compile-time / launch-time NVFP4 MegaMoE configuration."""

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int

    mma_tiler_mnk: Tuple[int, int, int] = (128, 128, 256)
    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    flag_batch: int = 4
    epi_flag_batch: Tuple[int, int] = (1, 1)
    non_ubulk_fc2_store: bool = True
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16"
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
        if self.hidden % 128 != 0 or self.intermediate % 128 != 0:
            raise ValueError(
                "hidden and intermediate must be multiples of 128 "
                f"(got hidden={self.hidden}, intermediate={self.intermediate})."
            )
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(
                f"token_back_mode must be 'epi_warps', 'standalone_warps', "
                f"or 'reuse_dispatch_warps'; got {self.token_back_mode!r}."
            )
        if self.combine_dtype not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(
                f"combine_dtype must be 'bf16', 'mxfp8', or 'nvfp4'; "
                f"got {self.combine_dtype!r}."
            )
        if self.combine_dtype != "bf16":
            if self.in_kernel_fc2_reduce:
                raise ValueError(
                    "combine_dtype != 'bf16' requires in_kernel_fc2_reduce=False "
                    "(low-precision combine is a form-A path)."
                )
            if self.token_back_mode != "reuse_dispatch_warps":
                raise ValueError(
                    "combine_dtype != 'bf16' is currently only wired for "
                    "token_back_mode='reuse_dispatch_warps'; got "
                    f"{self.token_back_mode!r}."
                )
        if self.in_kernel_fc2_reduce and not self.apply_topk_in_fc1:
            # Mirrors the kernel ctor check; fail at config build, not compile.
            raise ValueError(
                "in_kernel_fc2_reduce requires apply_topk_in_fc1=True; the REDG "
                "path can only atomic-add terms whose topk score was already "
                "absorbed before fc2."
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

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def fc2_reduces_topk(self) -> bool:
        return self.in_kernel_fc2_reduce

    @property
    def token_back_by_dispatch(self) -> bool:
        return self.token_back_mode != "epi_warps"


@dataclasses.dataclass
class MegaMoENvfp4Inputs:
    """Per-rank tensors for one NVFP4 MegaMoE launch.

    Symmetric-heap tensors (``activation``, ``activation_sf``, ``topk_idx``,
    ``topk_weights``) must be allocated via NVSHMEM (or plain CUDA when
    ``MEGA_NO_DIST=1``).  Weights and epilogue scalars are rank-local.

    ``output_activation`` is the kernel's single 2D ``(T, hidden)`` bf16 output;
    the kernel reduces the top-k combine internally (the drop replaced the old
    form-A ``combine_output`` + separate reduce with this in-kernel reduce).
    """

    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor
    output_activation: torch.Tensor


class MegaMoENvfp4Frontend:
    """Lazy-compile host wrapper for ``Sm100MegaMoEKernel``."""

    def __init__(self, config: MegaMoENvfp4Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoENvfp4Config:
        """Effective config (includes runtime ``gate_up_clamp`` updates)."""
        if self._gate_up_clamp == self._config.gate_up_clamp:
            return self._config
        return dataclasses.replace(self._config, gate_up_clamp=self._gate_up_clamp)

    def set_gate_up_clamp(self, clamp: Optional[float]) -> None:
        """Update ``gate_up_clamp`` and invalidate compile cache when it changes."""
        if self._gate_up_clamp == clamp:
            return
        self._release_workspace()
        self._gate_up_clamp = clamp
        self._invalidate_compile_cache()

    def apply_knobs(self, knobs: Optional[dict]) -> None:
        """Apply tuner knobs (see :mod:`.tuner`) to the session config.

        Invalidates the compile cache when the effective config changes; the
        next ``run()``/``warmup()`` recompiles.  Used by :mod:`.autotune`.
        """
        from .tuner import with_knobs

        new_config = with_knobs(self.config, knobs)
        if new_config == self._config:
            return
        self._release_workspace()
        self._config = new_config
        self._invalidate_compile_cache()

    def release(self) -> None:
        """Free compiled workspaces (symmetric heap when using NVSHMEM)."""
        self._release_workspace()
        self._invalidate_compile_cache()

    def warmup(
        self,
        inputs: MegaMoENvfp4Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> None:
        """Compile (if needed) without launching."""
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return None
        self._ensure_mega_compiled(inputs)

    def run(
        self,
        inputs: MegaMoENvfp4Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = False,
        reduce_topk: bool = True,
    ) -> Optional[torch.Tensor]:
        """Launch NVFP4 MegaMoE and return the 2D ``(T, hidden)`` bf16 output.

        ``num_tokens`` limits the active token rows when the input buffers are
        sized for ``config.num_tokens_per_rank`` but fewer tokens are live.

        The kernel drop reduces the top-k combine internally, so the result is
        always the reduced ``output_activation`` (the old form-A + separate
        top-k-reduce path is gone).  ``reduce_topk`` is accepted for backward
        API compatibility and ignored.

        ``reset_counters=False`` (default): workspaces are allocated zeroed and
        the kernel tail-cleans its own counters/flags after every launch (the
        kernel-team drivers and tester never host-reset), so no per-launch
        reset is needed.  Pass ``True`` only to recover after an aborted /
        interrupted launch left the workspaces dirty.

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
            mega.launch_key = key
            mega.launch_output = launch_inputs.output_activation

        if reset_counters:
            self._reset_workspaces(mega)

        if self.config.fc2_reduces_topk:
            # ikr accumulate-from-zero contract: output_activation is the
            # cross-rank REDG atomic-add target, so it must be zeroed before
            # every launch (stream-ordered; ~10 us at 2048 tokens).  Zero the
            # full raw buffer so stale rows beyond a partial num_tokens can't
            # leak from an earlier, larger launch.
            inputs.output_activation.zero_()
        mega.compiled(**mega.launch_kwargs)

        if sync:
            torch.cuda.synchronize()
        return mega.launch_output

    def make_launch_thunk(
        self,
        inputs: MegaMoENvfp4Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> Callable[[], None]:
        """Zero-arg launcher with args prebuilt (compiles if needed).

        Steady-state fast path for timing loops and tuners, mirroring the
        kernel tester's ``launch_plan`` contract: no per-call Python arg
        rebuild, no workspace reset (the kernel tail-cleans its own
        counters/flags), no sync.  Output lands in
        ``inputs.output_activation``.  Invalid after the compile cache is
        invalidated (knobs/clamp change) or the buffers are freed.

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

        if self.config.fc2_reduces_topk:
            output_activation = inputs.output_activation

            def thunk() -> None:
                output_activation.zero_()
                compiled(**runtime_kwargs)

        else:

            def thunk() -> None:
                compiled(**runtime_kwargs)

        return thunk

    @staticmethod
    def _launch_cache_key(inputs: MegaMoENvfp4Inputs, num_tokens: int) -> tuple:
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
            t.fc1_alpha.data_ptr(),
            t.fc2_alpha.data_ptr(),
            t.fc1_norm_const.data_ptr(),
            t.output_activation.data_ptr(),
            num_tokens,
            torch.cuda.current_stream().cuda_stream,
        )

    # ------------------------------------------------------------------
    # Compile cache
    # ------------------------------------------------------------------

    def _mega_compile_key(self) -> tuple:
        c = self.config
        return (
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
            c.non_ubulk_fc2_store,
            c.in_kernel_fc2_reduce,
            c.token_back_mode,
            c.combine_dtype,
            c.apply_topk_in_fc1,
            self._gate_up_clamp,
            c.enable_iket,
        )

    def _ensure_mega_compiled(self, inputs: MegaMoENvfp4Inputs) -> _CompiledMega:
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        self._release_workspace()

        import cutlass
        import cutlass.cute as cute

        from common.megamoe_constants import SfPaddingBlock
        from moe_nvfp4_swapab.epilogue_refactor import SwapABSwigluFp4Epilogue
        from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel
        from src.token_comm import CombineFormat

        c = self.config
        # The kernel now takes a CombineFormat object (was a combine_dtype string)
        # and derives local_rank from the peer mapper (was a ctor arg).
        combine_format = CombineFormat.parse(COMBINE_FORMAT_NAMES[c.combine_dtype])
        token_padding_block = SwapABSwigluFp4Epilogue._EpilogueTokenTileSize
        static_expert_shape = (
            c.num_experts_per_rank,
            c.intermediate,
            c.hidden,
        )

        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        max_active_clusters = max(1, sm_count // max(cluster_size, 1))
        group_hint = c.group_hint if c.group_hint is not None else max_active_clusters

        kernel = Sm100MegaMoEKernel(
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
            world_size=c.world_size,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_output_dtype=cutlass.BFloat16,
            non_ubulk_fc2_store=c.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=c.in_kernel_fc2_reduce,
            token_back_mode=c.token_back_mode,
            apply_topk_in_fc1=c.apply_topk_in_fc1,
            gate_up_clamp=self._gate_up_clamp,
            flag_batch=c.flag_batch,
            epi_flag_batch=c.epi_flag_batch,
            combine_format=combine_format,
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

    # NOTE: the new kernel drop reduces the top-k combine INSIDE the mega kernel
    # (Sm100MegaMoEKernel.__call__), so there is no separate topk-reduce launch.
    # The drop's moe_nvfp4_swapab.topk_reduce.TopkReduce class covers the
    # standalone combine-reduce path, which moe_ep does not use; port it here
    # (with a test) if a caller ever needs in_kernel_fc2_reduce=False combine.

    # ------------------------------------------------------------------
    # Launch helpers
    # ------------------------------------------------------------------

    def _invalidate_compile_cache(self) -> None:
        self._mega_key = None
        self._mega = None

    def _release_workspace(self) -> None:
        if self._mega is not None:
            free_sym_tensor(self._mega.shared_workspace)

    @staticmethod
    def _resolve_num_tokens(
        inputs: MegaMoENvfp4Inputs,
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
        inputs: MegaMoENvfp4Inputs,
        *,
        num_tokens: Optional[int],
    ) -> Optional[MegaMoENvfp4Inputs]:
        resolved = self._resolve_num_tokens(inputs, num_tokens)
        if resolved == 0:
            return None
        self._validate_inputs(inputs, num_tokens=resolved)
        buf_tokens = inputs.activation.shape[0]
        if not self.config.fc2_reduces_topk and resolved < buf_tokens:
            raise ValueError(
                "Partial num_tokens is not supported when in_kernel_fc2_reduce=False "
                f"(top-k reduce compiles for the full buffer of {buf_tokens} tokens). "
                f"Got num_tokens={resolved}."
            )
        if resolved == buf_tokens:
            return inputs
        return self._slice_inputs(inputs, resolved)

    @staticmethod
    def _slice_inputs(
        inputs: MegaMoENvfp4Inputs, num_tokens: int
    ) -> MegaMoENvfp4Inputs:
        tok = slice(None, num_tokens)

        return MegaMoENvfp4Inputs(
            activation=inputs.activation[tok],
            activation_sf=inputs.activation_sf[tok],
            topk_idx=inputs.topk_idx[tok],
            topk_weights=inputs.topk_weights[tok],
            fc1_weight=inputs.fc1_weight,
            fc1_weight_sf=inputs.fc1_weight_sf,
            fc2_weight=inputs.fc2_weight,
            fc2_weight_sf=inputs.fc2_weight_sf,
            fc1_alpha=inputs.fc1_alpha,
            fc2_alpha=inputs.fc2_alpha,
            fc1_norm_const=inputs.fc1_norm_const,
            output_activation=inputs.output_activation[tok],
        )

    def _validate_inputs(
        self,
        inputs: MegaMoENvfp4Inputs,
        *,
        num_tokens: int,
    ) -> None:
        from common.megamoe_constants import Nvfp4BlockSize
        from moe_nvfp4_swapab.runner_common import _DataDtype, _ScaleDtype

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

        intermediate_down = c.intermediate // 2
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
        activation_hidden_packed = inputs.activation.shape[-1]
        if activation_hidden_packed * 2 != c.hidden:
            raise ValueError(
                f"activation packed hidden dim ({activation_hidden_packed}) * 2 "
                f"must equal config.hidden ({c.hidden}); "
                f"got shape {tuple(inputs.activation.shape)}."
            )
        if inputs.activation.dtype != _DataDtype:
            raise ValueError(
                f"activation must have dtype {_DataDtype}, got {inputs.activation.dtype}."
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
        hidden_sf_cols = (c.hidden + Nvfp4BlockSize - 1) // Nvfp4BlockSize
        if inputs.activation_sf.dtype != _ScaleDtype:
            raise ValueError(
                f"activation_sf must have dtype {_ScaleDtype}, "
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

        weight_checks = (
            (
                "fc1_weight",
                inputs.fc1_weight,
                (e, c.hidden // 2, c.intermediate),
                _DataDtype,
            ),
            (
                "fc2_weight",
                inputs.fc2_weight,
                (e, intermediate_down // 2, c.hidden),
                _DataDtype,
            ),
        )
        for name, tensor, shape, dtype in weight_checks:
            _require_cuda(name, tensor)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
                )
            if tensor.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}.")

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

        for name, tensor in (
            ("fc1_alpha", inputs.fc1_alpha),
            ("fc2_alpha", inputs.fc2_alpha),
            ("fc1_norm_const", inputs.fc1_norm_const),
        ):
            _require_cuda(name, tensor)
            if tensor.shape != (c.num_experts_per_rank,):
                raise ValueError(
                    f"{name} must have shape ({c.num_experts_per_rank},), "
                    f"got {tuple(tensor.shape)}."
                )
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be float32, got {tensor.dtype}.")

    @staticmethod
    def _to_cute(
        tensor: torch.Tensor, assumed_align: int = 16, *, static_layout: bool = False
    ):
        import cutlass.torch as cutlass_torch

        cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        if static_layout:
            return cute_tensor
        leading_dim = cutlass_torch.get_leading_dim(tensor)
        return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    def _build_mega_runtime_kwargs(
        self,
        inputs: MegaMoENvfp4Inputs,
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
            fc1_alpha=self._to_cute(inputs.fc1_alpha, assumed_align=4),
            fc2_alpha=self._to_cute(inputs.fc2_alpha, assumed_align=4),
            fc1_norm_const=self._to_cute(inputs.fc1_norm_const, assumed_align=4),
            output_activation=self._to_cute(inputs.output_activation),
            # Opaque byte workspaces are passed as raw uint8 gmem base pointers
            # (not cute tensors): the kernel addresses them by base + Int64 byte
            # offset, and a tensor shape would overflow cute's 32-bit memref
            # field once internalized combine staging pushes shared_workspace
            # past 2 GiB (mirrors the drop's mega_runner).
            local_workspace=self._to_cute_ptr(mega.local_workspace),
            shared_workspace=self._to_cute_ptr(mega.shared_workspace),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )

    @staticmethod
    def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
        """Raw uint8 gmem base pointer for an opaque byte workspace."""
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.typing import AddressSpace

        return cute.runtime.make_ptr(
            cutlass.Uint8,
            tensor.data_ptr(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )

    @staticmethod
    def _reset_workspaces(mega: _CompiledMega) -> None:
        reset_compiled_mega_workspaces(mega)


# ---------------------------------------------------------------------------
# High-level MegaMoE API (symm buffers + launch + dummy inputs)
# ---------------------------------------------------------------------------

TransformedWeights = Tuple[torch.Tensor, torch.Tensor]


PerExpertEpilogue = Union[torch.Tensor, int, float]


def _resolve_per_expert_epilogue(
    name: str,
    value: Optional[PerExpertEpilogue],
    num_experts_per_rank: int,
) -> torch.Tensor:
    """Build a per-local-expert fp32 CUDA vector (default 1.0)."""
    out = torch.ones(
        (num_experts_per_rank,),
        dtype=torch.float32,
        device="cuda",
    )
    if value is None:
        return out
    if isinstance(value, (int, float)):
        out.fill_(float(value))
        return out
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device {value.device}.")
    if value.shape != (num_experts_per_rank,):
        raise ValueError(
            f"{name} must have shape ({num_experts_per_rank},), "
            f"got {tuple(value.shape)}."
        )
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}.")
    out.copy_(value)
    return out


def _sym_zeros_byte_view(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 / fp8 symmetric heap via uint8 reinterpret (matches mega_runner).

    Returns ``(view, root_uint8_buffer)``; free the root via :func:`free_sym_tensor`.
    """
    if target_dtype == _DataDtype:
        if not logical_shape or logical_shape[-1] % 2 != 0:
            raise ValueError(
                f"NVFP4 sym view needs non-empty logical_shape with even last dim, "
                f"got {logical_shape}."
            )
        storage_shape = (*logical_shape[:-1], logical_shape[-1] // 2)
    elif target_dtype == _ScaleDtype:
        storage_shape = tuple(logical_shape)
    else:
        raise ValueError(
            f"_sym_zeros_byte_view: dtype must be {_DataDtype} or {_ScaleDtype}, "
            f"got {target_dtype}."
        )
    total_bytes = 1
    for dim_size in storage_shape:
        total_bytes *= dim_size
    root = sym_zeros((total_bytes,), torch.uint8)
    view = root.view(target_dtype).reshape(storage_shape)
    return view, root


def init_dist() -> Tuple[int, int]:
    """Initialize torch.distributed + NVSHMEM (or single-rank when ``MEGA_NO_DIST=1``).

    Returns ``(rank, world_size)``.
    """
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


@dataclass
class MegaMoESymmBuffer:
    """Symmetric-heap staging buffers for one MegaMoE session.

    Mirrors DeepGEMM's symm-buffer object: exposes ``x``, ``x_sf``,
    ``topk_idx``, and ``topk_weights`` views sized for ``num_max_tokens``.
    Internal combine / epilogue tensors are owned here but not part of the
    public DeepGEMM surface.

    Expert weights are **not** stored here — pass ``transformed_l1`` /
    ``transformed_l2`` to :func:`nvfp4_mega_moe` each launch.
    """

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
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor

    _frontend: MegaMoENvfp4Frontend
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


def get_symm_buffer_for_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    apply_topk_in_fc1: bool = True,
    in_kernel_fc2_reduce: bool = False,
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16",
    fc1_alpha: Optional[PerExpertEpilogue] = None,
    fc2_alpha: Optional[PerExpertEpilogue] = None,
    fc1_norm_const: Optional[PerExpertEpilogue] = None,
    knobs: Optional[dict] = None,
) -> MegaMoESymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one MegaMoE session.

    Argument order follows ``deep_gemm.get_symm_buffer_for_mega_moe`` (problem
    sizes first).  Pass ``rank`` / ``world_size`` from :func:`init_dist` instead
    of a ``ProcessGroup`` — NVSHMEM bootstrap is handled internally.

    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is a
    deprecated alias for ``gate_up_clamp``.

    ``apply_topk_in_fc1`` mirrors ``mega_runner``'s
    ``ref_compute_graph == "deepgemm"`` behaviour when ``True`` (default).

    ``in_kernel_fc2_reduce`` collapses the top-k combine in flight via
    cross-rank REDG atomic-add instead of staging the per-topk ``(T, K, H)``
    tensor + explicit tail reduce: ~1-2% faster end to end and the multi-GB
    internal combine staging disappears from ``shared_workspace``.  Requires
    ``apply_topk_in_fc1=True`` and a bf16 combine wire; the accumulation order
    is nondeterministic (compare with a tolerance, not bit-exact).

    ``combine_dtype`` selects the cross-rank combine wire format: ``"bf16"``
    (default, exact), ``"mxfp8"`` (fp8+e8m0 SF, 2x less combine traffic), or
    ``"nvfp4"`` (fp4+bf16 SF, 4x less).  Quantized wires are a numerics
    tradeoff and require the explicit-reduce path
    (``in_kernel_fc2_reduce=False``) with dispatch-warp token-back (applied
    automatically to the default knobs; explicit ``knobs`` must be compatible).

    ``fc1_alpha``, ``fc2_alpha``, and ``fc1_norm_const`` are per-local-expert
    fp32 epilogue scalars with shape ``(num_total_experts // world_size,)``.
    Pass a scalar to broadcast one value to all local experts, or pass a CUDA
    float32 tensor with that shape.  When omitted, each defaults to ``1.0``.

    Expert weights are not allocated here; supply kernel-ready
    ``(weight, scale)`` tuples to :func:`nvfp4_mega_moe` instead.
    """
    if hidden % 128 != 0 or intermediate % 128 != 0:
        raise ValueError(
            "MegaMoE requires hidden and intermediate to be multiples of 128."
        )
    if num_total_experts % world_size != 0:
        raise ValueError("num_total_experts must be divisible by world_size.")

    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    num_experts_per_rank = num_total_experts // world_size

    from .tuner import default_knobs, with_knobs

    # Token-count heuristic picks the perf/tile tactic by compile-time buffer
    # size (num_max_tokens); an explicit knobs= dict overrides it entirely.
    resolved_knobs = dict(knobs) if knobs is not None else default_knobs(num_max_tokens)
    if knobs is None and combine_dtype != "bf16":
        # The measured profiles pick the token-back mode freely, but a
        # quantized combine wire is only wired for dispatch-warp token-back;
        # explicit knobs are the caller's contract and are left untouched (the
        # config validation rejects incompatible combos).
        resolved_knobs["token_back_mode"] = "reuse_dispatch_warps"

    cfg = MegaMoENvfp4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=clamp,
        apply_topk_in_fc1=apply_topk_in_fc1,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        combine_dtype=combine_dtype,
        # Constructed valid even before knobs land: quantized combine rejects
        # the default epi_warps token-back in __post_init__.
        token_back_mode=(
            "reuse_dispatch_warps" if combine_dtype != "bf16" else "epi_warps"
        ),
    )
    cfg = with_knobs(cfg, resolved_knobs)
    frontend = MegaMoENvfp4Frontend(cfg)

    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)

    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view((num_max_tokens, hidden), _DataDtype)
    sym_roots.append(x_root)
    x_sf, x_sf_root = _sym_zeros_byte_view(
        (num_max_tokens, hidden_sf_cols_padded),
        _ScaleDtype,
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
    # Single 2D (T, hidden) bf16 output; the kernel reduces the top-k combine
    # internally.  Allocated on the symmetric heap unconditionally: under
    # in_kernel_fc2_reduce it IS the cross-rank REDG atomic-add target (hard
    # requirement), and in explicit-reduce mode a sym allocation behaves like
    # plain CUDA memory locally.  Always-sym keeps the session ikr-capable so
    # the knob can flip per-compile (autotune / apply_knobs) without
    # reallocating; the cost ((T, hidden) bf16) is negligible next to the
    # internal combine staging.
    output_activation = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(output_activation)
    fc1_alpha = _resolve_per_expert_epilogue(
        "fc1_alpha",
        fc1_alpha,
        num_experts_per_rank,
    )
    fc2_alpha = _resolve_per_expert_epilogue(
        "fc2_alpha",
        fc2_alpha,
        num_experts_per_rank,
    )
    fc1_norm_const = _resolve_per_expert_epilogue(
        "fc1_norm_const",
        fc1_norm_const,
        num_experts_per_rank,
    )

    return MegaMoESymmBuffer(
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
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        _frontend=frontend,
        _sym_roots=sym_roots,
    )


def nvfp4_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoESymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> None:
    """Launch the fused CuTeDSL NVFP4 MegaMoE kernel (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / routing slices before calling.

    ``transformed_l1`` / ``transformed_l2`` are ``(weight, scale)`` tuples in
    the **kernel-ready** NVFP4 + swizzled-SF layout (see ``mega_runner`` weight
    assembly, not ``deep_gemm.transform_weights_for_mega_moe``).  Weights are
    always caller-supplied here — unlike epilogue scalars in
    :func:`get_symm_buffer_for_mega_moe`, they are not owned by the symm buffer.

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
            "fast_math=False has no effect in the CuTeDSL NVFP4 MegaMoE path.",
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
    if n == 0 and symm_buffer._frontend.config.fc2_reduces_topk:
        return
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

    inputs = MegaMoENvfp4Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        fc1_alpha=symm_buffer.fc1_alpha,
        fc2_alpha=symm_buffer.fc2_alpha,
        fc1_norm_const=symm_buffer.fc1_norm_const,
        output_activation=symm_buffer.output_activation,
    )

    # The kernel reduces the top-k combine internally and writes the final 2D
    # (T, hidden) output; no host-side form-A reduction is needed.  Launch the
    # full padded buffer (topk_idx[n:] == -1 marks the pad rows) and copy the
    # live [:n] rows out -- matches the reference driver, which does not slice.
    out = symm_buffer._frontend.run(inputs, num_tokens=None, sync=False)
    if out is not None:
        y.copy_(out[:n])
    if sync:
        torch.cuda.synchronize()


def nvfp4_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoESymmBuffer,
) -> Callable[[], None]:
    """Prebuilt zero-arg NVFP4 mega launcher for steady-state timing loops.

    Tester-parity timed region (``tester/solver.py perf_run``): the returned
    thunk is a bare compiled-kernel launch -- args prebuilt once, no per-call
    Python, no workspace reset (the kernel tail-cleans), no sync, no output
    copy.  The reduced bf16 output lands in ``symm_buffer.output_activation``.
    Compiles on this call if needed.  Rebuild the thunk after knob/clamp
    changes or buffer destruction.
    """
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    fc1_weight, fc1_weight_sf = transformed_l1
    fc2_weight, fc2_weight_sf = transformed_l2
    inputs = MegaMoENvfp4Inputs(
        activation=symm_buffer.x,
        activation_sf=symm_buffer.x_sf,
        topk_idx=symm_buffer.topk_idx,
        topk_weights=symm_buffer.topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
        fc1_alpha=symm_buffer.fc1_alpha,
        fc2_alpha=symm_buffer.fc2_alpha,
        fc1_norm_const=symm_buffer.fc1_norm_const,
        output_activation=symm_buffer.output_activation,
    )
    return symm_buffer._frontend.make_launch_thunk(inputs)


def make_dummy_epilogue_params(
    num_local_experts: int,
    *,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random per-local-expert epilogue scalars (matches ``mega_runner``)."""
    fc1_alpha = (
        torch.randint(
            1,
            5,
            (num_local_experts,),
            generator=generator,
            device="cuda",
        ).to(torch.float32)
        * 0.5
    )
    fc2_alpha = (
        torch.randint(
            1,
            5,
            (num_local_experts,),
            generator=generator,
            device="cuda",
        ).to(torch.float32)
        * 0.5
    )
    fc1_norm_const = (
        torch.randint(
            2,
            5,
            (num_local_experts,),
            generator=generator,
            device="cuda",
        ).to(torch.float32)
        * 0.5
    )
    return fc1_alpha, fc2_alpha, fc1_norm_const


def _create_dummy_weights(
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    generator: torch.Generator,
) -> Tuple[TransformedWeights, TransformedWeights]:
    """Random NVFP4 weights + swizzled SF for local smoke scripts."""
    from moe_nvfp4_swapab.mega_runner import (
        _stack_byte_reinterpretable_tensors,
    )
    from moe_nvfp4_swapab.runner_common import (
        make_nvfp4_tensor_from_torch_rng,
        make_raw_scale_tensor_from_torch_rng,
        to_blocked,
    )

    intermediate_down = intermediate // 2
    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    intermediate_down_sf_cols = ceil_div(intermediate_down, Nvfp4BlockSize)

    fc1_weight = make_nvfp4_tensor_from_torch_rng(
        generator,
        (num_local_experts, hidden, intermediate),
        packed_dim=1,
        perf_run=True,
    )
    fc1_weight_sf_plain = make_raw_scale_tensor_from_torch_rng(
        generator,
        num_local_experts * intermediate,
        hidden,
        blocksize=Nvfp4BlockSize,
        strict=True,
    ).reshape(num_local_experts, intermediate, hidden_sf_cols)
    fc1_sf_swizzled = [
        to_blocked(fc1_weight_sf_plain[e]) for e in range(num_local_experts)
    ]
    fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_local_experts, fc1_flat_sf_size
    )

    fc2_weight = make_nvfp4_tensor_from_torch_rng(
        generator,
        (num_local_experts, intermediate_down, hidden),
        packed_dim=1,
        perf_run=True,
    )
    fc2_weight_sf_plain = make_raw_scale_tensor_from_torch_rng(
        generator,
        num_local_experts * hidden,
        intermediate_down,
        blocksize=Nvfp4BlockSize,
        strict=True,
    ).reshape(num_local_experts, hidden, intermediate_down_sf_cols)
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
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fc1_alpha: Optional[PerExpertEpilogue] = None,
    fc2_alpha: Optional[PerExpertEpilogue] = None,
    fc1_norm_const: Optional[PerExpertEpilogue] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedWeights,
    TransformedWeights,
    MegaMoESymmBuffer,
]:
    """Allocate symm buffer, NVFP4 weights, and stage activations + routing.

    Mirrors ``dummy_fp8_fp4_mega_moe.create_dummy_inputs`` for the NVFP4 path.
    When ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` are omitted, random
    per-local-expert values are generated from ``seed`` (see
    :func:`make_dummy_epilogue_params`).
    """
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
    if fc1_alpha is None and fc2_alpha is None and fc1_norm_const is None:
        fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
            num_local_experts,
            generator=gen,
        )

    symm_buffer = get_symm_buffer_for_mega_moe(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        gate_up_clamp=clamp,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )

    transformed_l1, transformed_l2 = _create_dummy_weights(
        num_local_experts,
        hidden,
        intermediate,
        gen,
    )

    from moe_nvfp4_swapab.runner_common import (
        make_nvfp4_tensor_from_torch_rng,
        make_raw_scale_tensor_from_torch_rng,
    )

    activation = make_nvfp4_tensor_from_torch_rng(
        gen,
        (num_tokens, hidden),
        packed_dim=-1,
        perf_run=True,
    )
    activation_sf = make_raw_scale_tensor_from_torch_rng(
        gen,
        num_tokens,
        hidden,
        blocksize=Nvfp4BlockSize,
        strict=True,
    ).reshape(num_tokens, ceil_div(hidden, Nvfp4BlockSize))

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
    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    symm_buffer.x_sf[:num_tokens, :hidden_sf_cols].copy_(activation_sf)
    symm_buffer.topk_idx[:num_tokens].copy_(topk_idx.to(torch.int64))
    # Mask pad rows (and stale routes from a previous larger staging): the
    # launch covers the full buffer and relies on topk_idx[n:] == -1.
    symm_buffer.topk_idx[num_tokens:].fill_(-1)
    symm_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

    y = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    return y, transformed_l1, transformed_l2, symm_buffer


def _main() -> None:
    """Minimal torchrun smoke for the NVFP4 MegaMoE thin API."""
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

    rank, world_size = init_dist()
    symm_buffer = None
    num_local_experts = NUM_EXPERTS // world_size

    try:
        epilogue_gen = torch.Generator(device="cuda")
        epilogue_gen.manual_seed(0 + rank)
        fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
            num_local_experts,
            generator=epilogue_gen,
        )

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
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            seed=0,
        )

        nvfp4_mega_moe(
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
