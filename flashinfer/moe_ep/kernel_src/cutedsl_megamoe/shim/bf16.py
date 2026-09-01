# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile BF16 MegaMoE API."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import torch

from .comm import (
    _CompiledMega,
    _compute_peer_offsets,
    bootstrap_dist,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)


@dataclasses.dataclass(frozen=True)
class MegaMoEBf16Config:
    """Compile-time and launch-time BF16 MegaMoE configuration."""

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 64)
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (1, 1)
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    gate_up_clamp: Optional[float] = None
    apply_topk_in_fc1: bool = True
    enable_iket: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size).")
        if self.num_tokens_per_rank <= 0:
            raise ValueError("num_tokens_per_rank must be positive.")
        if not 1 <= self.num_topk <= 32:
            raise ValueError("num_topk must be in [1, 32].")
        if self.num_total_experts % self.world_size:
            raise ValueError("num_total_experts must be divisible by world_size.")
        if self.hidden % 32:
            raise ValueError("hidden must be divisible by 32.")
        if self.intermediate % 64:
            raise ValueError("intermediate must be divisible by 64.")
        if self.mma_tiler_mnk != (256, 256, 64):
            raise ValueError("BF16 MegaMoE requires mma_tiler_mnk=(256, 256, 64).")
        if self.cluster_shape_mnk != (2, 1, 1) or not self.use_2cta_instrs:
            raise ValueError(
                "BF16 MegaMoE requires cluster_shape_mnk=(2, 1, 1) "
                "and use_2cta_instrs=True."
            )
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(f"unsupported token_back_mode={self.token_back_mode!r}.")
        if self.in_kernel_fc2_reduce and self.token_back_mode == "epi_warps":
            raise ValueError(
                "in_kernel_fc2_reduce requires standalone or reused dispatch token-back."
            )

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size


@dataclass
class MegaMoEBf16Inputs:
    activation: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    combine_output: torch.Tensor


class MegaMoEBf16Frontend:
    """Host wrapper for ``Sm100MegaMoEBf16Kernel``."""

    def __init__(self, config: MegaMoEBf16Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoEBf16Config:
        if self._gate_up_clamp == self._config.gate_up_clamp:
            return self._config
        return dataclasses.replace(self._config, gate_up_clamp=self._gate_up_clamp)

    def set_gate_up_clamp(self, clamp: Optional[float]) -> None:
        if self._gate_up_clamp != clamp:
            self._release_workspace()
            self._gate_up_clamp = clamp
            self._mega_key = None
            self._mega = None

    def apply_knobs(self, knobs: dict) -> None:
        """Apply a validated BF16 tuning configuration and invalidate its compile."""
        from .tuner import is_valid_bf16, with_knobs

        if not is_valid_bf16(knobs):
            raise ValueError(f"unsupported BF16 MegaMoE knobs: {knobs}.")
        new_config = with_knobs(self.config, knobs)
        if new_config != self._config:
            self._release_workspace()
            self._config = new_config
            self._mega_key = None
            self._mega = None

    def release(self) -> None:
        self._release_workspace()
        self._mega_key = None
        self._mega = None

    @staticmethod
    def _to_cute(tensor: torch.Tensor, *, static_layout: bool = False):
        import cutlass.torch as cutlass_torch

        result = cutlass_torch.from_dlpack(tensor, assumed_align=16)
        if static_layout:
            return result
        return result.mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(tensor)
        )

    def _compile_key(self) -> tuple:
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
            c.load_balance_mode,
            c.group_hint,
            c.force_static_sched,
            c.clc_bundle_size,
            c.num_sched_stages,
            c.flag_batch,
            c.epi_flag_batch,
            c.in_kernel_fc2_reduce,
            c.token_back_mode,
            self._gate_up_clamp,
            c.apply_topk_in_fc1,
            c.enable_iket,
        )

    def _ensure_compiled(self, inputs: MegaMoEBf16Inputs) -> _CompiledMega:
        key = self._compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        self._release_workspace()
        import cutlass
        import cutlass.cute as cute
        from moe_bf16_glu.megamoe_kernel_bf16 import Sm100MegaMoEBf16Kernel

        c = self.config
        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        max_active_clusters = max(1, sm_count // cluster_size)
        kernel = Sm100MegaMoEBf16Kernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=c.use_2cta_instrs,
            group_hint=c.group_hint or max_active_clusters,
            token_padding_block=128,
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=(
                c.num_experts_per_rank,
                2 * c.intermediate,
                c.hidden,
            ),
            force_static_sched=c.force_static_sched,
            clc_bundle_size=c.clc_bundle_size,
            num_sched_stages=c.num_sched_stages,
            ab_dtype=cutlass.BFloat16,
            world_size=c.world_size,
            local_rank=c.rank,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_in_kernel_topk_reduce=c.in_kernel_fc2_reduce,
            token_back_by_dispatch=c.token_back_mode != "epi_warps",
            token_back_mode=c.token_back_mode,
            epi_flag_batch=c.epi_flag_batch,
            flag_batch=c.flag_batch,
            gate_up_clamp=self._gate_up_clamp,
            apply_topk_in_fc1=c.apply_topk_in_fc1,
        )
        local_bytes, shared_bytes = kernel.get_workspace_sizes()
        local_workspace = torch.zeros(local_bytes, dtype=torch.uint8, device="cuda")
        shared_workspace = sym_zeros((shared_bytes,), torch.uint8)
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
        kwargs = self._runtime_kwargs(inputs, mega)
        kwargs["max_active_clusters"] = max_active_clusters
        if c.enable_iket:
            kwargs["options"] = "iket"
        mega.compiled = cute.compile(kernel, **kwargs)
        self._mega = mega
        self._mega_key = key
        return mega

    def _runtime_kwargs(self, inputs: MegaMoEBf16Inputs, mega: _CompiledMega) -> dict:
        import cuda.bindings.driver as cuda
        from src.sym_buffer import SymBufferHost

        c = self.config
        mapper = SymBufferHost(
            base_addr=mega.symmetric_base,
            offsets=tuple(mega.peer_offsets_list),
            rank_idx=c.rank,
            num_max_ranks=c.world_size,
        )
        return {
            "activation": self._to_cute(inputs.activation),
            "topk_idx": self._to_cute(inputs.topk_idx),
            "topk_weights": self._to_cute(inputs.topk_weights),
            "fc1_weight": self._to_cute(inputs.fc1_weight),
            "fc2_weight": self._to_cute(inputs.fc2_weight),
            "fc1_c": None,
            "combine_output": self._to_cute(inputs.combine_output),
            "local_workspace": self._to_cute(mega.local_workspace, static_layout=True),
            "shared_workspace": self._to_cute(mega.shared_workspace),
            "peer_rank_ptr_mapper_host": mapper,
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def run(
        self,
        inputs: MegaMoEBf16Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = False,
    ) -> torch.Tensor:
        n = inputs.activation.shape[0] if num_tokens is None else num_tokens
        self._validate(inputs, n)
        mega = self._ensure_compiled(inputs)
        key = (
            inputs.activation.data_ptr(),
            inputs.topk_idx.data_ptr(),
            inputs.topk_weights.data_ptr(),
            inputs.fc1_weight.data_ptr(),
            inputs.fc2_weight.data_ptr(),
            inputs.combine_output.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if mega.launch_key != key:
            mega.launch_kwargs = self._runtime_kwargs(inputs, mega)
            mega.launch_key = key
        if self.config.in_kernel_fc2_reduce:
            inputs.combine_output.zero_()
        mega.compiled(**mega.launch_kwargs)
        if sync:
            torch.cuda.synchronize()
        return inputs.combine_output[:n]

    def make_launch_thunk(self, inputs: MegaMoEBf16Inputs) -> Callable[[], None]:
        self._validate(inputs, inputs.activation.shape[0])
        mega = self._ensure_compiled(inputs)
        kwargs = self._runtime_kwargs(inputs, mega)
        compiled = mega.compiled
        if self.config.in_kernel_fc2_reduce:

            def thunk():
                inputs.combine_output.zero_()
                compiled(**kwargs)

            return thunk
        else:

            def thunk():
                compiled(**kwargs)

            return thunk

    def _validate(self, inputs: MegaMoEBf16Inputs, num_tokens: int) -> None:
        c = self.config
        if not 0 <= num_tokens <= c.num_tokens_per_rank:
            raise ValueError(
                f"num_tokens must be in [0, {c.num_tokens_per_rank}], got {num_tokens}."
            )
        expected = (
            (c.num_tokens_per_rank, c.hidden),
            (c.num_tokens_per_rank, c.num_topk),
            (c.num_tokens_per_rank, c.num_topk),
            (c.num_experts_per_rank, c.hidden, 2 * c.intermediate),
            (c.num_experts_per_rank, c.intermediate, c.hidden),
        )
        tensors = (
            inputs.activation,
            inputs.topk_idx,
            inputs.topk_weights,
            inputs.fc1_weight,
            inputs.fc2_weight,
        )
        for name, tensor, shape in zip(
            ("activation", "topk_idx", "topk_weights", "fc1_weight", "fc2_weight"),
            tensors,
            expected,
            strict=True,
        ):
            if not tensor.is_cuda or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be CUDA with shape {shape}.")
        if inputs.activation.dtype != torch.bfloat16:
            raise ValueError("activation must be bfloat16.")
        if (
            inputs.topk_idx.dtype != torch.int64
            or inputs.topk_weights.dtype != torch.float32
        ):
            raise ValueError("topk_idx/topk_weights must be int64/float32.")
        if (
            inputs.fc1_weight.dtype != torch.bfloat16
            or inputs.fc2_weight.dtype != torch.bfloat16
        ):
            raise ValueError("BF16 MegaMoE weights must be bfloat16.")
        topk_dim = 1 if c.in_kernel_fc2_reduce else c.num_topk
        if inputs.combine_output.shape != (c.num_tokens_per_rank, topk_dim, c.hidden):
            raise ValueError("combine_output has an invalid shape.")

    def _release_workspace(self) -> None:
        if self._mega is not None:
            free_sym_tensor(self._mega.shared_workspace)


@dataclass
class MegaMoEBf16SymmBuffer:
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
    combine_output: torch.Tensor
    _frontend: MegaMoEBf16Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    def destroy(self) -> None:
        if not self._destroyed:
            self._frontend.release()
            for root in self._sym_roots:
                free_sym_tensor(root)
            self._sym_roots.clear()
            self._destroyed = True


TransformedWeights = Tuple[torch.Tensor, None]


def init_dist() -> Tuple[int, int]:
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


def get_symm_buffer_for_bf16_mega_moe(
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
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps",
    knobs: Optional[dict] = None,
) -> MegaMoEBf16SymmBuffer:
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    cfg = MegaMoEBf16Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=clamp,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        **(knobs or {}),
    )
    x = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    topk_idx.fill_(-1)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    combine_output = sym_zeros(
        (num_max_tokens, 1 if in_kernel_fc2_reduce else num_topk, hidden),
        torch.bfloat16,
    )
    return MegaMoEBf16SymmBuffer(
        num_total_experts,
        num_max_tokens,
        num_topk,
        hidden,
        intermediate,
        rank,
        world_size,
        x,
        topk_idx,
        topk_weights,
        combine_output,
        MegaMoEBf16Frontend(cfg),
        [x, topk_idx, topk_weights, combine_output],
    )


def bf16_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEBf16SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> None:
    del fast_math
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if y.shape != (n, symm_buffer.hidden) or y.dtype != torch.bfloat16:
        raise ValueError(f"y must be bfloat16 with shape ({n}, {symm_buffer.hidden}).")
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)
    result = symm_buffer._frontend.run(
        MegaMoEBf16Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l2[0],
            symm_buffer.combine_output,
        ),
        num_tokens=n,
        sync=sync,
    )
    if symm_buffer._frontend.config.in_kernel_fc2_reduce:
        y.copy_(result[:, 0])
    else:
        y.copy_(result.sum(dim=1))


def bf16_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEBf16SymmBuffer,
) -> Callable[[], None]:
    return symm_buffer._frontend.make_launch_thunk(
        MegaMoEBf16Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l2[0],
            symm_buffer.combine_output,
        )
    )


__all__ = [
    "MegaMoEBf16Config",
    "MegaMoEBf16Frontend",
    "MegaMoEBf16Inputs",
    "MegaMoEBf16SymmBuffer",
    "TransformedWeights",
    "bf16_mega_launch_thunk",
    "bf16_mega_moe",
    "get_symm_buffer_for_bf16_mega_moe",
    "init_dist",
]
