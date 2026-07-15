# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile MXFP8 MegaMoE API for ``Sm100MegaMoEMxfp8Kernel``."""

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
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from common.megamoe_constants import Mxfp8BlockSize
from moe_nvfp4_swapab.runner_common import ceil_div, round_up

_KIND_TO_TORCH_DTYPE = {
    "mxfp8_e4m3": torch.float8_e4m3fn,
    "mxfp8_e5m2": torch.float8_e5m2,
}


def _kind_to_cutlass_dtype(kind: str):
    import cutlass

    return {
        "mxfp8_e4m3": cutlass.Float8E4M3FN,
        "mxfp8_e5m2": cutlass.Float8E5M2,
    }[kind]


@dataclasses.dataclass(frozen=True)
class MegaMoEMxfp8Config:
    """Compile-time / launch-time MXFP8 MegaMoE configuration.

    ``intermediate`` is the post-SwiGLU width. The MXFP8 kernel's full FC1
    gate+up width is derived as ``2 * intermediate``.
    """

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int

    kind: Literal["mxfp8_e4m3", "mxfp8_e5m2"] = "mxfp8_e4m3"
    mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 128)
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    flag_batch: int = 4
    epi_flag_batch: Tuple[int, int] = (1, 1)
    in_kernel_fc2_reduce: bool = False
    token_back_by_dispatch: bool = False
    gate_up_clamp: Optional[float] = None
    enable_iket: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KIND_TO_TORCH_DTYPE:
            raise ValueError(
                f"kind must be one of {sorted(_KIND_TO_TORCH_DTYPE)}, "
                f"got {self.kind!r}."
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
        if self.hidden % 128 != 0 or self.intermediate % 128 != 0:
            raise ValueError(
                "hidden and intermediate must be multiples of 128 "
                f"(got hidden={self.hidden}, intermediate={self.intermediate})."
            )
        if self.in_kernel_fc2_reduce and self.token_back_by_dispatch:
            raise ValueError(
                "in_kernel_fc2_reduce and token_back_by_dispatch cannot both be True."
            )
        m, n, _k = self.mma_tiler_mnk
        if (m, n) != (256, 256) or not self.use_2cta_instrs:
            raise ValueError(
                "MXFP8 MegaMoE only supports mma_tiler (M, N) = (256, 256) with "
                "use_2cta_instrs=True; "
                f"got mma_tiler_mnk={self.mma_tiler_mnk}, "
                f"use_2cta_instrs={self.use_2cta_instrs}."
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

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def torch_ab_dtype(self) -> torch.dtype:
        return _KIND_TO_TORCH_DTYPE[self.kind]

    @property
    def fc1_out(self) -> int:
        return 2 * self.intermediate


@dataclasses.dataclass
class MegaMoEMxfp8Inputs:
    """Per-rank tensors for one MXFP8 MegaMoE launch."""

    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    # Single 2D (T, hidden) bf16 output; the kernel reduces top-k internally
    # (the drop replaced the old form-A ``combine_output`` with this).
    output_activation: torch.Tensor


class MegaMoEMxfp8Frontend:
    """Lazy-compile host wrapper for ``Sm100MegaMoEMxfp8Kernel``."""

    def __init__(self, config: MegaMoEMxfp8Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoEMxfp8Config:
        if self._gate_up_clamp == self._config.gate_up_clamp:
            return self._config
        return dataclasses.replace(self._config, gate_up_clamp=self._gate_up_clamp)

    def set_gate_up_clamp(self, clamp: Optional[float]) -> None:
        if self._gate_up_clamp == clamp:
            return
        self._release_workspace()
        self._gate_up_clamp = clamp
        self._invalidate_compile_cache()

    def apply_knobs(self, knobs: Optional[dict]) -> None:
        """Apply tuner knobs (see :mod:`.tuner`) to the session config.

        Invalidates the compile cache when the effective config changes; the
        next ``run()``/``warmup()`` recompiles.  Used by :mod:`.autotune`.
        ``token_back_mode`` is translated to this config's
        ``token_back_by_dispatch`` bool by ``with_knobs``.
        """
        from .tuner import with_knobs

        new_config = with_knobs(self.config, knobs)
        if new_config == self._config:
            return
        self._release_workspace()
        self._config = new_config
        self._invalidate_compile_cache()

    def release(self) -> None:
        self._release_workspace()
        self._invalidate_compile_cache()

    def warmup(
        self,
        inputs: MegaMoEMxfp8Inputs,
        *,
        num_tokens: Optional[int] = None,
    ) -> None:
        launch_inputs = self._prepare_launch_inputs(inputs, num_tokens=num_tokens)
        if launch_inputs is None:
            return None
        self._ensure_mega_compiled(inputs)

    def run(
        self,
        inputs: MegaMoEMxfp8Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = True,
        reset_counters: bool = False,
        reduce_topk: bool = True,
    ) -> Optional[torch.Tensor]:
        """Launch MXFP8 MegaMoE and return the 2D ``(T, hidden)`` bf16 output.

        The kernel drop reduces the top-k combine internally, so the result is
        always the reduced ``output_activation`` (the old form-A path is gone).
        ``reduce_topk`` is accepted for backward API compatibility and ignored.

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
            reset_compiled_mega_workspaces(mega)
        if self.config.in_kernel_fc2_reduce:
            # ikr accumulate-from-zero contract: output_activation is the
            # cross-rank REDG atomic-add target, so it must be zeroed before
            # every launch.  Zero the full raw buffer so stale rows beyond a
            # partial num_tokens can't leak from an earlier, larger launch.
            inputs.output_activation.zero_()
        mega.compiled(**mega.launch_kwargs)
        if sync:
            torch.cuda.synchronize()
        return mega.launch_output

    def make_launch_thunk(
        self,
        inputs: MegaMoEMxfp8Inputs,
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
    def _launch_cache_key(inputs: MegaMoEMxfp8Inputs, num_tokens: int) -> tuple:
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
            self._gate_up_clamp,
            c.enable_iket,
        )

    def _ensure_mega_compiled(self, inputs: MegaMoEMxfp8Inputs) -> _CompiledMega:
        key = self._mega_compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        self._release_workspace()

        import cutlass.cute as cute

        from common.megamoe_constants import Mxfp8BlockSize, SfPaddingBlock
        from moe_mxfp8_glu.megamoe_kernel_mxfp8 import Sm100MegaMoEMxfp8Kernel
        from moe_nvfp4_swapab.epilogue import EpilogueTokenTile

        c = self.config
        static_expert_shape = (
            c.num_experts_per_rank,
            c.fc1_out,
            c.hidden,
        )

        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        max_active_clusters = max(1, sm_count // max(cluster_size, 1))
        group_hint = c.group_hint if c.group_hint is not None else max_active_clusters

        kernel = Sm100MegaMoEMxfp8Kernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=c.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=EpilogueTokenTile,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=c.force_static_sched,
            clc_bundle_size=c.clc_bundle_size,
            num_sched_stages=c.num_sched_stages,
            ab_dtype=_kind_to_cutlass_dtype(c.kind),
            sf_vec_size=Mxfp8BlockSize,
            world_size=c.world_size,
            local_rank=c.rank,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_in_kernel_topk_reduce=c.in_kernel_fc2_reduce,
            # kernel renamed the bool token_back_by_dispatch -> token_back_mode enum:
            # dispatch-reuse maps to "reuse_dispatch_warps", default to "epi_warps".
            token_back_mode=(
                "reuse_dispatch_warps" if c.token_back_by_dispatch else "epi_warps"
            ),
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
            free_sym_tensor(self._mega.shared_workspace)

    @staticmethod
    def _resolve_num_tokens(
        inputs: MegaMoEMxfp8Inputs,
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
        inputs: MegaMoEMxfp8Inputs,
        *,
        num_tokens: Optional[int],
    ) -> Optional[MegaMoEMxfp8Inputs]:
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
        inputs: MegaMoEMxfp8Inputs,
        num_tokens: int,
    ) -> MegaMoEMxfp8Inputs:
        tok = slice(None, num_tokens)
        return MegaMoEMxfp8Inputs(
            activation=inputs.activation[tok],
            activation_sf=inputs.activation_sf[tok],
            topk_idx=inputs.topk_idx[tok],
            topk_weights=inputs.topk_weights[tok],
            fc1_weight=inputs.fc1_weight,
            fc1_weight_sf=inputs.fc1_weight_sf,
            fc2_weight=inputs.fc2_weight,
            fc2_weight_sf=inputs.fc2_weight_sf,
            output_activation=inputs.output_activation[tok],
        )

    def _validate_inputs(
        self,
        inputs: MegaMoEMxfp8Inputs,
        *,
        num_tokens: int,
    ) -> None:
        from common.megamoe_constants import Mxfp8BlockSize
        from moe_nvfp4_swapab.runner_common import Mxfp8ScaleDtype

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

        hidden_sf_cols = (c.hidden + Mxfp8BlockSize - 1) // Mxfp8BlockSize
        if inputs.activation_sf.dtype != Mxfp8ScaleDtype:
            raise ValueError(
                f"activation_sf must have dtype {Mxfp8ScaleDtype}, "
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
        inputs: MegaMoEMxfp8Inputs,
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
            output_activation=self._to_cute(inputs.output_activation),
            local_workspace=self._to_cute(
                mega.local_workspace,
                static_layout=True,
            ),
            shared_workspace=self._to_cute(mega.shared_workspace),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            # fc1_c is the optional in-kernel fc1-out C buffer (generate_c path);
            # None uses the kernel's internal staging (matches the drop default).
            fc1_c=None,
            stream=stream,
        )


# ---------------------------------------------------------------------------
# High-level MegaMoE API (symm buffers + launch + dummy inputs)
# ---------------------------------------------------------------------------

TransformedWeights = Tuple[torch.Tensor, torch.Tensor]

Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]


def _sym_zeros_byte_view_1b(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 / E8M0 symmetric heap via uint8 reinterpret (matches mega_runner).

    Returns ``(view, root_uint8_buffer)``; free the root via :func:`free_sym_tensor`.
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
class MegaMoEMxfp8SymmBuffer:
    """Symmetric-heap staging buffers for one MXFP8 MegaMoE session.

    Mirrors the NVFP4 :class:`MegaMoESymmBuffer`: exposes ``x``, ``x_sf``,
    ``topk_idx``, and ``topk_weights`` views sized for ``num_max_tokens``.

    Expert weights are **not** stored here — pass ``transformed_l1`` /
    ``transformed_l2`` to :func:`mxfp8_mega_moe` each launch.
    """

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int
    kind: Mxfp8Kind

    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor

    _frontend: MegaMoEMxfp8Frontend
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


def get_symm_buffer_for_mxfp8_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_by_dispatch: bool = False,
    knobs: Optional[dict] = None,
) -> MegaMoEMxfp8SymmBuffer:
    """Allocate symmetric-heap inputs + combine staging for one MXFP8 session.

    Argument order follows the NVFP4 frontend (problem sizes first).  Pass
    ``rank`` / ``world_size`` from :func:`init_dist`.

    ``kind`` selects the fp8 element format (``mxfp8_e4m3`` or ``mxfp8_e5m2``).
    ``gate_up_clamp`` sets the kernel gate-up clamp.  ``activation_clamp`` is a
    deprecated alias for ``gate_up_clamp``.
    ``intermediate`` is the post-SwiGLU width, matching NVFP4 and SGLang.

    Expert weights are not allocated here; supply kernel-ready ``(weight, scale)``
    tuples to :func:`mxfp8_mega_moe` instead.
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

    cfg = MegaMoEMxfp8Config(
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
        token_back_by_dispatch=token_back_by_dispatch,
    )
    from .tuner import default_knobs, with_knobs

    # Default tactic: the measured MXFP8 schedule (dtype="mxfp8"; one profile
    # for all token counts, no tile knob -- MXFP8's mma_tiler is kernel-fixed
    # at (256, 256)).  An explicit knobs= dict overrides it entirely.
    cfg = with_knobs(
        cfg,
        knobs if knobs is not None else default_knobs(num_max_tokens, dtype="mxfp8"),
    )
    frontend = MegaMoEMxfp8Frontend(cfg)

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
    data_dtype = cfg.torch_ab_dtype

    from moe_nvfp4_swapab.runner_common import Mxfp8ScaleDtype

    sym_roots: list[torch.Tensor] = []
    x, x_root = _sym_zeros_byte_view_1b((num_max_tokens, hidden), data_dtype)
    sym_roots.append(x_root)
    x_sf, x_sf_root = _sym_zeros_byte_view_1b(
        (num_max_tokens, hidden_sf_cols_padded),
        Mxfp8ScaleDtype,
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
    # rank-local buffer here was a latent crash for ikr sessions), and in
    # explicit-reduce mode a sym allocation behaves like plain CUDA memory
    # locally.  Always-sym also lets apply_knobs flip the ikr knob per-compile
    # without reallocating.
    output_activation = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    sym_roots.append(output_activation)

    return MegaMoEMxfp8SymmBuffer(
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


def mxfp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEMxfp8SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> None:
    """Launch the fused CuTeDSL MXFP8 MegaMoE kernel (dispatch + fc1 + fc2 + combine).

    Caller must stage ``symm_buffer.x`` / routing slices before calling.

    ``transformed_l1`` / ``transformed_l2`` are ``(weight, scale)`` tuples in
    the **kernel-ready** fp8 + swizzled-SF layout (see ``mega_runner`` weight
    assembly).  Weights are always caller-supplied here — they are not owned by
    the symm buffer.

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
            "fast_math=False has no effect in the CuTeDSL MXFP8 MegaMoE path.",
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

    inputs = MegaMoEMxfp8Inputs(
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

    # The kernel reduces the top-k combine internally and writes the final 2D
    # (T, hidden) output; no host-side form-A reduction is needed.  Launch the
    # full padded buffer (topk_idx[n:] == -1 marks the pad rows) and copy the
    # live [:n] rows out -- matches the reference driver, which does not slice.
    out = symm_buffer._frontend.run(inputs, num_tokens=None, sync=False)
    if out is not None:
        y.copy_(out[:n])
    if sync:
        torch.cuda.synchronize()


def mxfp8_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEMxfp8SymmBuffer,
) -> Callable[[], None]:
    """Prebuilt zero-arg MXFP8 mega launcher for steady-state timing loops.

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
    inputs = MegaMoEMxfp8Inputs(
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
    kind: Mxfp8Kind,
) -> Tuple[TransformedWeights, TransformedWeights]:
    """Random MXFP8 weights + swizzled SF for local smoke scripts."""
    from moe_mxfp8_glu.mega_runner import (
        _make_e8m0_scale_tensor,
        _make_fp8_tensor,
    )
    from moe_nvfp4_swapab.mega_runner import _stack_byte_reinterpretable_tensors
    from moe_nvfp4_swapab.runner_common import to_blocked

    data_dtype = _KIND_TO_TORCH_DTYPE[kind]

    fc1_out = 2 * intermediate
    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    intermediate_sf_cols = ceil_div(intermediate, Mxfp8BlockSize)

    fc1_weight = _make_fp8_tensor(
        generator,
        (num_local_experts, hidden, fc1_out),
        data_dtype,
        perf_run=True,
    )
    fc1_weight_sf_plain = _make_e8m0_scale_tensor(
        generator,
        num_local_experts * fc1_out,
        hidden,
        blocksize=Mxfp8BlockSize,
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
        (num_local_experts, intermediate, hidden),
        data_dtype,
        perf_run=True,
    )
    fc2_weight_sf_plain = _make_e8m0_scale_tensor(
        generator,
        num_local_experts * hidden,
        intermediate,
        blocksize=Mxfp8BlockSize,
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
    kind: Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    seed: int = 0,
) -> tuple[
    torch.Tensor,
    TransformedWeights,
    TransformedWeights,
    MegaMoEMxfp8SymmBuffer,
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

    symm_buffer = get_symm_buffer_for_mxfp8_mega_moe(
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

    from moe_mxfp8_glu.mega_runner import _make_e8m0_scale_tensor, _make_fp8_tensor

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
        blocksize=Mxfp8BlockSize,
    ).reshape(num_tokens, ceil_div(hidden, Mxfp8BlockSize))

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
    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
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
    """Minimal torchrun smoke for the MXFP8 MegaMoE thin API."""
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

        mxfp8_mega_moe(
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
