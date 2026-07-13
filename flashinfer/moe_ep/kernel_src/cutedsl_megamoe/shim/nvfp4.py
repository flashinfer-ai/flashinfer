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
from typing import Any, Literal, Optional, Tuple, Union

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

    @property
    def combine_is_quantized(self) -> bool:
        return self.token_back_by_dispatch and self.combine_dtype != "bf16"


@dataclasses.dataclass
class MegaMoENvfp4Inputs:
    """Per-rank tensors for one NVFP4 MegaMoE launch.

    Symmetric-heap tensors (``activation``, ``activation_sf``, ``topk_idx``,
    ``topk_weights``, ``combine_output``, and optional quantized combine staging)
    must be allocated via NVSHMEM (or plain CUDA when ``MEGA_NO_DIST=1``).
    Weights and epilogue scalars are rank-local.
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
    combine_output: torch.Tensor
    combine_reduced_output: Optional[torch.Tensor] = None
    combine_output_q: Optional[torch.Tensor] = None
    combine_sf_q: Optional[torch.Tensor] = None
    combine_global_q: Optional[torch.Tensor] = None
    topk_score_for_reduce: Optional[torch.Tensor] = None


@dataclasses.dataclass
class _CompiledTopk:
    compiled: Any
    combine_cute: Any
    reduced_cute: Any
    topk_score_cute: Any
    mxfp8_scale_cute: Any
    nvfp4_sfc_scale_cute: Any
    nvfp4_global_scale_cute: Any


class MegaMoENvfp4Frontend:
    """Lazy-compile host wrapper for ``Sm100MegaMoEKernel`` (+ optional topk reduce)."""

    def __init__(self, config: MegaMoENvfp4Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None
        self._topk_key: Optional[tuple] = None
        self._topk: Optional[_CompiledTopk] = None

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
        if not self.config.fc2_reduces_topk:
            self._ensure_topk_compiled(inputs)

    def run(
        self,
        inputs: MegaMoENvfp4Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = True,
        reduce_topk: bool = True,
    ) -> Optional[torch.Tensor]:
        """Launch NVFP4 MegaMoE (+ topk reduce when configured).

        ``num_tokens`` limits the active token rows when the input buffers are
        sized for ``config.num_tokens_per_rank`` but fewer tokens are live.

        Returns ``combine_reduced_output`` (bf16 ``(T, hidden)``) when the
        separate top-k reduce runs; otherwise ``combine_output.squeeze(1)``
        when ``in_kernel_fc2_reduce=True``. If ``reduce_topk=False`` with
        ``in_kernel_fc2_reduce=False``, returns form-A ``combine_output`` with
        shape ``(T, top_k, hidden)`` and skips the separate reduce kernel.
        """
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return None
        if (
            not self.config.fc2_reduces_topk
            and reduce_topk
            and launch_inputs.combine_reduced_output is None
        ):
            raise ValueError(
                "combine_reduced_output is required when in_kernel_fc2_reduce=False."
            )
        mega = self._ensure_mega_compiled(inputs)
        runtime_kwargs = self._build_mega_runtime_kwargs(launch_inputs, mega)
        topk_kwargs = None
        if not self.config.fc2_reduces_topk and reduce_topk:
            topk = self._ensure_topk_compiled(inputs)
            topk_kwargs = self._build_topk_runtime_kwargs(topk)

        if reset_counters:
            self._reset_workspaces(mega)

        mega.compiled(**runtime_kwargs)
        if topk_kwargs is not None:
            topk.compiled(**topk_kwargs)

        if sync:
            torch.cuda.synchronize()
        if self.config.fc2_reduces_topk:
            return launch_inputs.combine_output.squeeze(1)
        if not reduce_topk:
            return launch_inputs.combine_output
        return launch_inputs.combine_reduced_output

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

    def _topk_compile_key(self, inputs: MegaMoENvfp4Inputs) -> tuple:
        c = self.config
        combine_k = 1 if c.fc2_reduces_topk else c.num_topk
        return (
            c.num_tokens_per_rank,
            combine_k,
            c.hidden,
            c.combine_dtype,
            inputs.topk_score_for_reduce is not None,
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

        c = self.config
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
            local_rank=c.rank,
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
            combine_dtype=c.combine_dtype,
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

    def _ensure_topk_compiled(self, inputs: MegaMoENvfp4Inputs) -> _CompiledTopk:
        key = self._topk_compile_key(inputs)
        if self._topk is not None and self._topk_key == key:
            return self._topk

        import cuda.bindings.driver as cuda
        from moe_nvfp4_swapab.topk_reduce import compile_topk_reduce, nvfp4_pack6_views

        c = self.config
        if inputs.combine_reduced_output is None:
            raise ValueError(
                "combine_reduced_output is required when in_kernel_fc2_reduce=False."
            )

        has_topk_score = inputs.topk_score_for_reduce is not None  # noqa: F841
        has_combine_sf_q = inputs.combine_sf_q is not None
        has_combine_global_q = inputs.combine_global_q is not None
        if c.combine_dtype == "mxfp8":
            if not has_combine_sf_q:
                raise ValueError(
                    f"combine_dtype={c.combine_dtype!r} requires combine_sf_q."
                )
        elif c.combine_dtype == "nvfp4":
            if not has_combine_sf_q or not has_combine_global_q:
                raise ValueError(
                    f"combine_dtype={c.combine_dtype!r} requires combine_sf_q "
                    "and combine_global_q."
                )

        if c.combine_dtype == "mxfp8":
            combine_input = inputs.combine_output_q
            mxfp8_scale = inputs.combine_sf_q
            nvfp4_sfc = None
            nvfp4_global = None
        elif c.combine_dtype == "nvfp4":
            combine_input = inputs.combine_output_q.view(torch.uint8)
            mxfp8_scale = None
            nvfp4_global, nvfp4_sfc = nvfp4_pack6_views(inputs.combine_global_q)
        else:
            combine_input = inputs.combine_output
            mxfp8_scale = None
            nvfp4_sfc = None
            nvfp4_global = None

        if combine_input is None:
            raise ValueError(
                f"combine_dtype={c.combine_dtype!r} requires combine staging tensors."
            )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        plan = compile_topk_reduce(
            combine_input,
            inputs.combine_reduced_output,
            inputs.topk_score_for_reduce,
            mxfp8_scale=mxfp8_scale,
            nvfp4_sfc_scale=nvfp4_sfc,
            nvfp4_global_scale=nvfp4_global,
            stream=stream,
        )
        self._topk_key = key
        self._topk = _CompiledTopk(
            compiled=plan[0],
            combine_cute=plan[1],
            reduced_cute=plan[2],
            topk_score_cute=plan[3],
            mxfp8_scale_cute=plan[4],
            nvfp4_sfc_scale_cute=plan[5],
            nvfp4_global_scale_cute=plan[6],
        )
        return self._topk

    # ------------------------------------------------------------------
    # Launch helpers
    # ------------------------------------------------------------------

    def _invalidate_compile_cache(self) -> None:
        self._mega_key = None
        self._mega = None
        self._topk_key = None
        self._topk = None

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

        def _slice_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t[tok]

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
            combine_output=inputs.combine_output[tok],
            combine_reduced_output=_slice_optional(inputs.combine_reduced_output),
            combine_output_q=_slice_optional(inputs.combine_output_q),
            combine_sf_q=_slice_optional(inputs.combine_sf_q),
            combine_global_q=_slice_optional(inputs.combine_global_q),
            topk_score_for_reduce=_slice_optional(inputs.topk_score_for_reduce),
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

        def _require_cuda(name: str, tensor: torch.Tensor) -> None:
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor.")

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
            ("combine_output", inputs.combine_output),
        )
        for name, tensor in token_tensors:
            _require_cuda(name, tensor)
            if tensor.shape[0] != buf_tokens:
                raise ValueError(
                    f"{name}.shape[0] ({tensor.shape[0]}) must match "
                    f"activation.shape[0] ({buf_tokens})."
                )

        combine_k = 1 if c.fc2_reduces_topk else c.num_topk
        if inputs.combine_output.shape != (buf_tokens, combine_k, c.hidden):
            raise ValueError(
                "combine_output must have shape "
                f"({buf_tokens}, {combine_k}, {c.hidden}), "
                f"got {tuple(inputs.combine_output.shape)}."
            )
        if inputs.combine_output.dtype != torch.bfloat16:
            raise ValueError(
                f"combine_output must be bfloat16, got {inputs.combine_output.dtype}."
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
        if not c.fc2_reduces_topk and inputs.combine_reduced_output is None:
            raise ValueError(
                "combine_reduced_output is required when in_kernel_fc2_reduce=False."
            )
        if inputs.combine_reduced_output is not None:
            expected = (buf_tokens, c.hidden)
            _require_cuda("combine_reduced_output", inputs.combine_reduced_output)
            if tuple(inputs.combine_reduced_output.shape) != expected:
                raise ValueError(
                    "combine_reduced_output must have shape "
                    f"{expected}, got {tuple(inputs.combine_reduced_output.shape)}."
                )
            if inputs.combine_reduced_output.dtype != torch.bfloat16:
                raise ValueError(
                    "combine_reduced_output must be bfloat16, got "
                    f"{inputs.combine_reduced_output.dtype}."
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

        if inputs.topk_score_for_reduce is not None:
            expected = (buf_tokens, c.num_topk)
            _require_cuda("topk_score_for_reduce", inputs.topk_score_for_reduce)
            if tuple(inputs.topk_score_for_reduce.shape) != expected:
                raise ValueError(
                    "topk_score_for_reduce must have shape "
                    f"{expected}, got {tuple(inputs.topk_score_for_reduce.shape)}."
                )
            if inputs.topk_score_for_reduce.dtype != torch.float32:
                raise ValueError(
                    "topk_score_for_reduce must be float32, got "
                    f"{inputs.topk_score_for_reduce.dtype}."
                )

        if c.combine_is_quantized:
            if inputs.combine_output_q is None or inputs.combine_sf_q is None:
                raise ValueError(
                    f"combine_dtype={c.combine_dtype!r} requires combine_output_q "
                    "and combine_sf_q."
                )
            _require_cuda("combine_output_q", inputs.combine_output_q)
            _require_cuda("combine_sf_q", inputs.combine_sf_q)
            if inputs.combine_output_q.shape[0] != buf_tokens:
                raise ValueError(
                    "combine_output_q.shape[0] must match activation buffer size "
                    f"({buf_tokens}), got {inputs.combine_output_q.shape[0]}."
                )
            if c.combine_dtype == "nvfp4" and inputs.combine_global_q is None:
                raise ValueError("combine_dtype='nvfp4' requires combine_global_q.")
            if inputs.combine_global_q is not None:
                _require_cuda("combine_global_q", inputs.combine_global_q)
                if inputs.combine_global_q.shape[0] != buf_tokens:
                    raise ValueError(
                        "combine_global_q.shape[0] must match activation buffer size "
                        f"({buf_tokens}), got {inputs.combine_global_q.shape[0]}."
                    )

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

        if c.combine_is_quantized:
            if inputs.combine_output_q is None or inputs.combine_sf_q is None:
                raise ValueError(
                    f"combine_dtype={c.combine_dtype!r} requires combine_output_q "
                    "and combine_sf_q."
                )
            combine_output_q_cute = self._to_cute(
                inputs.combine_output_q.view(torch.uint8)
            )
            combine_sf_q_cute = self._to_cute(inputs.combine_sf_q.view(torch.uint8))
            if c.combine_dtype == "nvfp4":
                if inputs.combine_global_q is None:
                    raise ValueError("combine_dtype='nvfp4' requires combine_global_q.")
                combine_global_q_cute = self._to_cute(inputs.combine_global_q)
            else:
                combine_global_q_cute = self._to_cute(inputs.combine_output)
        else:
            combine_output_q_cute = self._to_cute(inputs.combine_output)
            combine_sf_q_cute = self._to_cute(inputs.combine_output)
            combine_global_q_cute = self._to_cute(inputs.combine_output)

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
            combine_output=self._to_cute(inputs.combine_output),
            combine_output_q=combine_output_q_cute,
            combine_sf_q=combine_sf_q_cute,
            combine_global_q=combine_global_q_cute,
            local_workspace=self._to_cute(
                mega.local_workspace,
                static_layout=True,
            ),
            shared_workspace=self._to_cute(mega.shared_workspace),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )

    @staticmethod
    def _build_topk_runtime_kwargs(topk: _CompiledTopk) -> dict:
        import cuda.bindings.driver as cuda

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return dict(
            combine_cute=topk.combine_cute,
            reduced_cute=topk.reduced_cute,
            topk_score_cute=topk.topk_score_cute,
            mxfp8_scale_cute=topk.mxfp8_scale_cute,
            nvfp4_sfc_scale_cute=topk.nvfp4_sfc_scale_cute,
            nvfp4_global_scale_cute=topk.nvfp4_global_scale_cute,
            stream=stream,
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
    combine_output: torch.Tensor
    combine_reduced_output: torch.Tensor
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
    fc1_alpha: Optional[PerExpertEpilogue] = None,
    fc2_alpha: Optional[PerExpertEpilogue] = None,
    fc1_norm_const: Optional[PerExpertEpilogue] = None,
) -> MegaMoESymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one MegaMoE session.

    Argument order follows ``deep_gemm.get_symm_buffer_for_mega_moe`` (problem
    sizes first).  Pass ``rank`` / ``world_size`` from :func:`init_dist` instead
    of a ``ProcessGroup`` — NVSHMEM bootstrap is handled internally.

    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is a
    deprecated alias for ``gate_up_clamp``.

    ``apply_topk_in_fc1`` mirrors ``mega_runner``'s
    ``ref_compute_graph == "deepgemm"`` behaviour when ``True`` (default).

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
    )
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
    sym_roots.append(topk_idx)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    sym_roots.append(topk_weights)
    combine_output = sym_zeros(
        (num_max_tokens, num_topk, hidden),
        torch.bfloat16,
    )
    sym_roots.append(combine_output)
    combine_reduced_output = torch.empty(
        (num_max_tokens, hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
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
        combine_output=combine_output,
        combine_reduced_output=combine_reduced_output,
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
        combine_output=symm_buffer.combine_output,
        combine_reduced_output=symm_buffer.combine_reduced_output,
    )

    if symm_buffer._frontend.config.fc2_reduces_topk:
        out = symm_buffer._frontend.run(inputs, num_tokens=n)
    else:
        out = symm_buffer._frontend.run(
            inputs,
            num_tokens=None,
            reduce_topk=False,
        )
    if out is not None:
        if symm_buffer._frontend.config.fc2_reduces_topk:
            y.copy_(out[:n])
        else:
            active_form_a = out[:n]
            active_form_a_fp32 = active_form_a.to(torch.float32)
            if symm_buffer._frontend.config.apply_topk_in_fc1:
                reduced = active_form_a_fp32.sum(dim=1).to(y.dtype)
            else:
                reduced = (
                    (
                        active_form_a_fp32
                        * symm_buffer.topk_weights[:n, :, None].to(torch.float32)
                    )
                    .sum(dim=1)
                    .to(y.dtype)
                )
            y.copy_(reduced)


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
