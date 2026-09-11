# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile NVFP4-weight, BF16-activation MegaMoE API."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import torch

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    _CompiledMega,
    _compute_peer_offsets,
    bootstrap_dist,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)


@dataclasses.dataclass(frozen=True)
class MegaMoEW4A16Config:
    """Compile-time and launch-time W4A16 MegaMoE configuration."""

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    mma_tiler_mnk: Tuple[int, int, int] = (256, 128, 256)
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    force_static_sched: bool = True
    num_sched_stages: Optional[int] = None
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (1, 1)
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    gate_up_clamp: Optional[float] = None
    apply_topk_in_fc1: bool = False
    enable_iket: bool = False

    def __post_init__(self) -> None:
        # Geometry knobs also arrive as JSON arrays from the benchmark.
        object.__setattr__(self, "mma_tiler_mnk", tuple(self.mma_tiler_mnk))
        object.__setattr__(self, "cluster_shape_mnk", tuple(self.cluster_shape_mnk))
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"Unsupported load_balance_mode={self.load_balance_mode!r}."
            )
        if self.apply_topk_in_fc1 or self.in_kernel_fc2_reduce:
            raise ValueError("W4A16 routing scores are applied after FC2.")
        if self.token_back_mode == "standalone_warps":
            raise ValueError("W4A16 standalone token-back is not supported.")
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
        if self.mma_tiler_mnk not in (
            (128, 64, 256),
            (128, 128, 256),
            (256, 64, 256),
            (256, 128, 256),
        ):
            raise ValueError(
                "W4A16 MegaMoE requires mma_tiler_mnk=M128/M256, N64/N128, K256."
            )
        if self.cluster_shape_mnk != (2, 1, 1) and not (
            self.cluster_shape_mnk == (1, 1, 1) and self.mma_tiler_mnk == (128, 64, 256)
        ):
            raise ValueError(
                "W4A16 requires cluster (2,1,1), or (1,1,1) for M128/N64/K256."
            )
        if self.use_2cta_instrs != (self.mma_tiler_mnk[0] == 256):
            raise ValueError("W4A16 MMA M128/M256 requires one/two-CTA instructions.")
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
class MegaMoEW4A16Inputs:
    activation: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc1_alpha: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    fc2_alpha: torch.Tensor
    combine_output: torch.Tensor


class MegaMoEW4A16Frontend:
    """Host wrapper for ``Sm100W4A16MegaMoEKernel``."""

    def __init__(self, config: MegaMoEW4A16Config) -> None:
        self._config = config
        self._gate_up_clamp = config.gate_up_clamp
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None
        self._reduce = None

    @property
    def config(self) -> MegaMoEW4A16Config:
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
        """Apply a validated swapped-MMA tuning configuration and invalidate its compile."""
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import is_valid, with_knobs

        if not is_valid(
            {
                "mma_tiler_mnk": self.config.mma_tiler_mnk,
                "cluster_shape_mnk": self.config.cluster_shape_mnk,
                **knobs,
            }
        ):
            raise ValueError(f"unsupported W4A16 MegaMoE knobs: {knobs}.")
        new_config = with_knobs(self.config, knobs)
        if new_config != self._config:
            from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                ensure_not_capturing,
            )

            ensure_not_capturing("apply_knobs (config change)")
            self._release_workspace()
            self._config = new_config
            self._mega_key = None
            self._mega = None

    def release(self) -> None:
        self._release_workspace()
        self._mega_key = None
        self._mega = None

    @staticmethod
    def _to_cute(
        tensor: torch.Tensor, *, static_layout: bool = False, assumed_align: int = 16
    ):
        import cutlass.torch as cutlass_torch

        result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
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
            c.num_sched_stages,
            c.flag_batch,
            c.epi_flag_batch,
            c.in_kernel_fc2_reduce,
            c.token_back_mode,
            self._gate_up_clamp,
            c.apply_topk_in_fc1,
            c.enable_iket,
        )

    def _ensure_compiled(self, inputs: MegaMoEW4A16Inputs) -> _CompiledMega:
        key = self._compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega

        self._release_workspace()
        import cutlass
        import cutlass.cute as cute
        from .megamoe_kernel import Sm100W4A16MegaMoEKernel

        c = self.config
        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        max_active_clusters = max(1, sm_count // cluster_size)
        kernel = Sm100W4A16MegaMoEKernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=c.use_2cta_instrs,
            group_hint=c.group_hint or max_active_clusters,
            token_padding_block=c.mma_tiler_mnk[1],
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=(
                c.num_experts_per_rank,
                2 * c.intermediate,
                c.hidden,
            ),
            force_static_sched=c.force_static_sched,
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
        # Start with 61440 registers per CTA; the kernel redistributes this
        # pool among the five warpgroup roles.
        kwargs["options"] = "--ptxas-options='-maxrregcount=96'"
        if c.enable_iket:
            kwargs["options"] += " iket"
        mega.compiled = cute.compile(kernel, **kwargs)
        self._mega = mega
        self._mega_key = key
        return mega

    def _runtime_kwargs(self, inputs: MegaMoEW4A16Inputs, mega: _CompiledMega) -> dict:
        import cuda.bindings.driver as cuda
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import SymBufferHost

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
            "fc1_weight_sf": self._to_cute(inputs.fc1_weight_sf),
            "fc1_alpha": self._to_cute(inputs.fc1_alpha, assumed_align=4),
            "fc2_weight": self._to_cute(inputs.fc2_weight),
            "fc2_weight_sf": self._to_cute(inputs.fc2_weight_sf),
            "fc2_alpha": self._to_cute(inputs.fc2_alpha, assumed_align=4),
            "fc1_c": None,
            "combine_output": self._to_cute(inputs.combine_output),
            "local_workspace": self._to_cute(mega.local_workspace, static_layout=True),
            "shared_workspace": self._to_cute(mega.shared_workspace),
            "peer_rank_ptr_mapper_host": mapper,
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def run(
        self,
        inputs: MegaMoEW4A16Inputs,
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
            inputs.fc1_weight_sf.data_ptr(),
            inputs.fc1_alpha.data_ptr(),
            inputs.fc2_weight.data_ptr(),
            inputs.fc2_weight_sf.data_ptr(),
            inputs.fc2_alpha.data_ptr(),
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

    def make_launch_thunk(self, inputs: MegaMoEW4A16Inputs) -> Callable[[], None]:
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

    def _validate(self, inputs: MegaMoEW4A16Inputs, num_tokens: int) -> None:
        c = self.config
        if not 0 <= num_tokens <= c.num_tokens_per_rank:
            raise ValueError(
                f"num_tokens must be in [0, {c.num_tokens_per_rank}], got {num_tokens}."
            )
        expected = (
            (c.num_tokens_per_rank, c.hidden),
            (c.num_tokens_per_rank, c.num_topk),
            (c.num_tokens_per_rank, c.num_topk),
            (c.num_experts_per_rank, c.hidden // 2, 2 * c.intermediate),
            (c.num_experts_per_rank, c.intermediate // 2, c.hidden),
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
        packed_dtypes = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", None))
        for weight, scale, alpha in (
            (inputs.fc1_weight, inputs.fc1_weight_sf, inputs.fc1_alpha),
            (inputs.fc2_weight, inputs.fc2_weight_sf, inputs.fc2_alpha),
        ):
            if weight.dtype not in packed_dtypes:
                raise ValueError("W4A16 packed weights must be FP4-x2 or uint8.")
            if not weight.transpose(1, 2).is_contiguous():
                raise ValueError("W4A16 packed weights must have K-major backing.")
            padded_rows = ((weight.shape[2] + 127) // 128) * 128
            padded_columns = ((weight.shape[1] // 8 + 3) // 4) * 4
            expected_sf = (weight.shape[0], padded_rows * padded_columns)
            if scale.shape != expected_sf or scale.dtype not in (
                torch.float8_e4m3fn,
                torch.uint8,
            ):
                raise ValueError(
                    "weight scales must be native per-expert E4M3 bytes "
                    f"with shape {expected_sf}."
                )
            if alpha.shape != (c.num_experts_per_rank,) or alpha.dtype != torch.float32:
                raise ValueError("weight global scales must be FP32 per expert.")
            if not all(t.is_cuda and t.is_contiguous() for t in (scale, alpha)):
                raise ValueError("weight scales must be contiguous CUDA tensors.")
        topk_dim = 1 if c.in_kernel_fc2_reduce else c.num_topk
        if inputs.combine_output.shape != (c.num_tokens_per_rank, topk_dim, c.hidden):
            raise ValueError("combine_output has an invalid shape.")

    def reduce_topk(self, combined, scores, output):
        """Apply FP32 routing scores after the BF16 FC2 cast, accumulating in FP32."""
        # Empty source ranks still execute the collective fused kernel, but
        # have no local rows to reduce and must not launch a zero-block grid.
        if output.shape[0] == 0:
            return
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
            get_cutedsl_target_arch,
            TopkReduce,
            CombineFormat,
        )

        def compact(tensor):
            return self._to_cute(tensor, static_layout=True).mark_compact_shape_dynamic(
                mode=0, stride_order=tensor.dim_order(), divisibility=1
            )

        args = (
            compact(combined),
            None,
            compact(output),
            compact(scores),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if self._reduce is None:
            reducer = TopkReduce(
                self.config.hidden,
                self.config.num_topk,
                CombineFormat.parse("bf16"),
                sm_arch=get_cutedsl_target_arch(),
            )
            self._reduce = cute.compile(reducer, *args)
        self._reduce(*args)

    def _release_workspace(self) -> None:
        if self._mega is not None:
            free_sym_tensor(self._mega.shared_workspace)


@dataclass
class MegaMoEW4A16SymmBuffer:
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
    _frontend: MegaMoEW4A16Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    def destroy(self) -> None:
        if not self._destroyed:
            self._frontend.release()
            for root in self._sym_roots:
                free_sym_tensor(root)
            self._sym_roots.clear()
            self._destroyed = True


TransformedWeights = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def init_dist() -> Tuple[int, int]:
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


def get_symm_buffer_for_w4a16_mega_moe(
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
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None,
    knobs: Optional[dict] = None,
) -> MegaMoEW4A16SymmBuffer:
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import resolve_knobs

    # Match the existing Mega cache contract: None is a pure capacity-keyed
    # lookup; an explicit dict (including {}) bypasses cache and defaults.
    if knobs is None:
        resolved_knobs, _ = resolve_knobs(
            dtype="w4a16",
            world_size=world_size,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_total_experts,
            topk=num_topk,
            max_tokens=num_max_tokens,
            combine_dtype="bf16",
        )
    else:
        resolved_knobs = {}
    optional_config = {
        **resolved_knobs,
        "gate_up_clamp": clamp,
        "in_kernel_fc2_reduce": in_kernel_fc2_reduce,
        **({"token_back_mode": token_back_mode} if token_back_mode is not None else {}),
        **(knobs or {}),
    }
    cfg = MegaMoEW4A16Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        **optional_config,
    )
    x = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    topk_idx.fill_(-1)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    combine_output = sym_zeros(
        (num_max_tokens, 1 if cfg.in_kernel_fc2_reduce else num_topk, hidden),
        torch.bfloat16,
    )
    return MegaMoEW4A16SymmBuffer(
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
        MegaMoEW4A16Frontend(cfg),
        [x, topk_idx, topk_weights, combine_output],
    )


def w4a16_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEW4A16SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    sync: bool = False,
) -> None:
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if y.shape != (n, symm_buffer.hidden) or y.dtype != torch.bfloat16:
        raise ValueError(f"y must be bfloat16 with shape ({n}, {symm_buffer.hidden}).")
    if not y.is_cuda or not y.is_contiguous():
        raise ValueError("y must be a contiguous CUDA tensor.")
    if (
        n > 0
        and symm_buffer._frontend._reduce is None
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "W4A16 top-k reducer cannot compile during CUDA graph capture; "
            "call warmup() with its default batch on all EP ranks before "
            "capturing a nonempty forward."
        )
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)
    result = symm_buffer._frontend.run(
        MegaMoEW4A16Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            transformed_l1[2],
            transformed_l2[0],
            transformed_l2[1],
            transformed_l2[2],
            symm_buffer.combine_output,
        ),
        num_tokens=n,
    )
    if symm_buffer._frontend.config.in_kernel_fc2_reduce:
        y.copy_(result[:, 0])
    else:
        symm_buffer._frontend.reduce_topk(result, symm_buffer.topk_weights[:n], y)
    if sync:
        torch.cuda.synchronize()


def w4a16_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEW4A16SymmBuffer,
) -> Callable[[], None]:
    return symm_buffer._frontend.make_launch_thunk(
        MegaMoEW4A16Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            transformed_l1[2],
            transformed_l2[0],
            transformed_l2[1],
            transformed_l2[2],
            symm_buffer.combine_output,
        )
    )


__all__ = [
    "MegaMoEW4A16Config",
    "MegaMoEW4A16Frontend",
    "MegaMoEW4A16Inputs",
    "MegaMoEW4A16SymmBuffer",
    "TransformedWeights",
    "w4a16_mega_launch_thunk",
    "w4a16_mega_moe",
    "get_symm_buffer_for_w4a16_mega_moe",
    "init_dist",
]
