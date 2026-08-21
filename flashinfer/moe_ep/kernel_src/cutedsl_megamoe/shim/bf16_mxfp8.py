"""Lazy-compile mixed MXFP8-weight/BF16-activation MegaMoE API."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Any

import torch

from .comm import (
    _CompiledMega,
    _compute_peer_offsets,
    bootstrap_dist,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)

MixedKind = Literal["bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2"]
TransformedWeights = Tuple[torch.Tensor, torch.Tensor]

_KIND_TO_DTYPE = {
    "bf16_mxfp8_e4m3": torch.float8_e4m3fn,
    "bf16_mxfp8_e5m2": torch.float8_e5m2,
}
_VALID_IMPLS = {
    ((256, 128, 128), "tmem", False, 128),
    ((256, 256, 128), "smem", False, 128),
    ((256, 256, 128), "tmem", True, 64),
}


@dataclass(frozen=True)
class MegaMoEBf16Mxfp8Config:
    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    kind: MixedKind = "bf16_mxfp8_e4m3"
    mma_tiler_mnk: Tuple[int, int, int] = (256, 128, 128)
    transform_buffer: Literal["smem", "tmem"] = "tmem"
    accumulator_overlap: bool = False
    transform_k_tile: Literal[64, 128] = 128
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = 2
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (1, 1)
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal["epi_warps", "reuse_dispatch_warps"] = "epi_warps"
    gate_up_clamp: Optional[float] = None
    enable_iket: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KIND_TO_DTYPE:
            raise ValueError(f"unsupported mixed weight kind {self.kind!r}.")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size).")
        if self.num_tokens_per_rank <= 0 or not 1 <= self.num_topk <= 32:
            raise ValueError(
                "num_tokens_per_rank must be positive and top-k in [1, 32]."
            )
        if self.num_total_experts % self.world_size:
            raise ValueError("num_total_experts must be divisible by world_size.")
        if self.hidden <= 0 or self.hidden % 32 or self.intermediate <= 0:
            raise ValueError(
                "hidden must be divisible by 32 and intermediate positive."
            )
        if (self.intermediate // 2) % 32:
            raise ValueError("intermediate/2 must be divisible by 32.")
        if self.cluster_shape_mnk != (2, 1, 1) or not self.use_2cta_instrs:
            raise ValueError("mixed MegaMoE requires two-CTA cluster (2, 1, 1).")
        if (
            self.mma_tiler_mnk,
            self.transform_buffer,
            self.accumulator_overlap,
            self.transform_k_tile,
        ) not in _VALID_IMPLS:
            raise ValueError("unsupported mixed MXFP8/BF16 implementation tuple.")
        if self.token_back_mode not in ("epi_warps", "reuse_dispatch_warps"):
            raise ValueError(
                "mixed MegaMoE does not support standalone token-back warps."
            )
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError("unsupported load_balance_mode.")

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def weight_dtype(self) -> torch.dtype:
        return _KIND_TO_DTYPE[self.kind]


@dataclass
class MegaMoEBf16Mxfp8Inputs:
    activation: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor
    combine_output: torch.Tensor


class MegaMoEBf16Mxfp8Frontend:
    def __init__(self, config: MegaMoEBf16Mxfp8Config) -> None:
        self._config = config
        self._mega_key: Optional[tuple] = None
        self._mega: Optional[_CompiledMega] = None

    @property
    def config(self) -> MegaMoEBf16Mxfp8Config:
        return self._config

    def release(self) -> None:
        if self._mega is not None:
            free_sym_tensor(self._mega.shared_workspace)
        self._mega = None
        self._mega_key = None

    def apply_knobs(self, knobs: dict) -> None:
        """Apply mixed MegaMoE tuning knobs and invalidate the compiled kernel."""
        from .tuner import with_knobs

        new_config = with_knobs(self._config, knobs)
        if new_config != self._config:
            self.release()
            self._config = new_config

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
        return dataclasses.astuple(self._config)

    def _validate(self, inputs: MegaMoEBf16Mxfp8Inputs, num_tokens: int) -> None:
        c = self.config
        e = c.num_experts_per_rank
        if not 0 <= num_tokens <= c.num_tokens_per_rank:
            raise ValueError("num_tokens exceeds workspace capacity.")
        expected = (
            (
                "activation",
                inputs.activation,
                (c.num_tokens_per_rank, c.hidden),
                torch.bfloat16,
            ),
            (
                "topk_idx",
                inputs.topk_idx,
                (c.num_tokens_per_rank, c.num_topk),
                torch.int64,
            ),
            (
                "topk_weights",
                inputs.topk_weights,
                (c.num_tokens_per_rank, c.num_topk),
                torch.float32,
            ),
            (
                "fc1_weight",
                inputs.fc1_weight,
                (e, c.hidden, 2 * c.intermediate),
                c.weight_dtype,
            ),
            (
                "fc2_weight",
                inputs.fc2_weight,
                (e, c.intermediate, c.hidden),
                c.weight_dtype,
            ),
        )
        for name, tensor, shape, dtype in expected:
            if (
                not tensor.is_cuda
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
            ):
                raise ValueError(f"{name} must be CUDA {dtype} with shape {shape}.")
        for name, tensor in (
            ("fc1_weight_sf", inputs.fc1_weight_sf),
            ("fc2_weight_sf", inputs.fc2_weight_sf),
        ):
            if (
                not tensor.is_cuda
                or tensor.dtype != torch.float8_e8m0fnu
                or tensor.ndim != 2
                or tensor.shape[0] != e
            ):
                raise ValueError(
                    f"{name} must be a CUDA E8M0FNU tensor with one row per local expert."
                )
        combine_topk = 1 if c.in_kernel_fc2_reduce else c.num_topk
        if (
            not inputs.combine_output.is_cuda
            or inputs.combine_output.dtype != torch.bfloat16
            or tuple(inputs.combine_output.shape)
            != (c.num_tokens_per_rank, combine_topk, c.hidden)
        ):
            raise ValueError(
                "combine_output has an invalid mixed MegaMoE shape or dtype."
            )

    def _runtime_kwargs(
        self, inputs: MegaMoEBf16Mxfp8Inputs, mega: _CompiledMega
    ) -> dict:
        import cuda.bindings.driver as cuda
        from src.sym_buffer import SymBufferHost

        return {
            "activation": self._to_cute(inputs.activation),
            "topk_idx": self._to_cute(inputs.topk_idx),
            "topk_weights": self._to_cute(inputs.topk_weights),
            "fc1_weight": self._to_cute(inputs.fc1_weight),
            "fc1_weight_sf": self._to_cute(inputs.fc1_weight_sf),
            "fc2_weight": self._to_cute(inputs.fc2_weight),
            "fc2_weight_sf": self._to_cute(inputs.fc2_weight_sf),
            "combine_output": self._to_cute(inputs.combine_output),
            "local_workspace": self._to_cute(mega.local_workspace, static_layout=True),
            "shared_workspace": self._to_cute(mega.shared_workspace),
            "peer_rank_ptr_mapper_host": SymBufferHost(
                base_addr=mega.symmetric_base,
                offsets=tuple(mega.peer_offsets_list),
                rank_idx=self.config.rank,
                num_max_ranks=self.config.world_size,
            ),
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def _ensure_compiled(self, inputs: MegaMoEBf16Mxfp8Inputs) -> _CompiledMega:
        key = self._compile_key()
        if self._mega is not None and self._mega_key == key:
            return self._mega
        self.release()
        import cutlass
        import cutlass.cute as cute
        from moe_mxfp8_bf16_glu.epilogue_mxfp8_bf16 import EpilogueTokenTile
        from moe_mxfp8_bf16_glu.megamoe_kernel_mxfp8_bf16 import (
            Sm100MegaMoEMxfp8Bf16Kernel,
        )

        c = self.config
        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        max_active_clusters = max(
            1,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            // cluster_size,
        )
        kernel = Sm100MegaMoEMxfp8Bf16Kernel(
            mma_tiler_mnk=c.mma_tiler_mnk,
            cluster_shape_mnk=c.cluster_shape_mnk,
            use_2cta_instrs=True,
            group_hint=c.group_hint or max_active_clusters,
            token_padding_block=EpilogueTokenTile,
            load_balance_mode=c.load_balance_mode,
            static_expert_shape=(c.num_experts_per_rank, 2 * c.intermediate, c.hidden),
            force_static_sched=True,
            clc_bundle_size=c.clc_bundle_size,
            num_sched_stages=c.num_sched_stages,
            transform_buffer=c.transform_buffer,
            accumulator_overlap=c.accumulator_overlap,
            transform_k_tile=c.transform_k_tile,
            ab_dtype=cutlass.BFloat16,
            world_size=c.world_size,
            local_rank=c.rank,
            num_topk=c.num_topk,
            max_tokens_per_rank=c.num_tokens_per_rank,
            hidden=c.hidden,
            fc2_in_kernel_topk_reduce=c.in_kernel_fc2_reduce,
            token_back_by_dispatch=c.token_back_mode == "reuse_dispatch_warps",
            token_back_mode=c.token_back_mode,
            epi_flag_batch=c.epi_flag_batch,
            flag_batch=c.flag_batch,
            gate_up_clamp=c.gate_up_clamp,
            apply_topk_in_fc1=True,
            generate_c=False,
            use_stg_fc1=False,
        )
        local_bytes, shared_bytes = kernel.get_workspace_sizes()
        mega = _CompiledMega(
            compiled=None,
            kernel=kernel,
            local_workspace=torch.zeros(local_bytes, dtype=torch.uint8, device="cuda"),
            shared_workspace=sym_zeros((shared_bytes,), torch.uint8),
            symmetric_base=0,
            peer_offsets_list=[],
        )
        mega.symmetric_base, mega.peer_offsets_list = _compute_peer_offsets(
            mega.shared_workspace, c.world_size
        )
        kwargs = self._runtime_kwargs(inputs, mega)
        kwargs["max_active_clusters"] = max_active_clusters
        if c.enable_iket:
            kwargs["options"] = "iket"
        mega.compiled = cute.compile(kernel, **kwargs)
        self._mega, self._mega_key = mega, key
        return mega

    def run(
        self,
        inputs: MegaMoEBf16Mxfp8Inputs,
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
            inputs.fc2_weight.data_ptr(),
            inputs.fc2_weight_sf.data_ptr(),
            inputs.combine_output.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if mega.launch_key != key:
            mega.launch_kwargs = self._runtime_kwargs(inputs, mega)
            mega.launch_key = key
        if self.config.in_kernel_fc2_reduce:
            inputs.combine_output.zero_()
        mega.compiled(**mega.launch_kwargs)
        if sync and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return inputs.combine_output[:n]

    def make_launch_thunk(self, inputs: MegaMoEBf16Mxfp8Inputs) -> Callable[[], Any]:
        self._validate(inputs, inputs.activation.shape[0])
        mega = self._ensure_compiled(inputs)
        kwargs = self._runtime_kwargs(inputs, mega)
        if self.config.in_kernel_fc2_reduce:
            return lambda: (inputs.combine_output.zero_(), mega.compiled(**kwargs))
        return lambda: mega.compiled(**kwargs)


@dataclass
class MegaMoEBf16Mxfp8SymmBuffer:
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
    _frontend: MegaMoEBf16Mxfp8Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    def destroy(self) -> None:
        if not self._destroyed:
            self._frontend.release()
            for root in self._sym_roots:
                free_sym_tensor(root)
            self._destroyed = True


def init_dist() -> Tuple[int, int]:
    _, rank, world_size, _ = bootstrap_dist()
    return rank, world_size


def get_symm_buffer_for_bf16_mxfp8_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    kind: MixedKind = "bf16_mxfp8_e4m3",
    gate_up_clamp: Optional[float] = None,
    in_kernel_fc2_reduce: bool = False,
    token_back_mode: Literal["epi_warps", "reuse_dispatch_warps"] = "epi_warps",
    knobs: Optional[dict] = None,
) -> MegaMoEBf16Mxfp8SymmBuffer:
    config = MegaMoEBf16Mxfp8Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        kind=kind,
        gate_up_clamp=resolve_gate_up_clamp(
            gate_up_clamp=gate_up_clamp, activation_clamp=None
        ),
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
        **(knobs or {}),
    )
    x = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    topk_idx.fill_(-1)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    combine = sym_zeros(
        (num_max_tokens, 1 if in_kernel_fc2_reduce else num_topk, hidden),
        torch.bfloat16,
    )
    return MegaMoEBf16Mxfp8SymmBuffer(
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
        combine,
        MegaMoEBf16Mxfp8Frontend(config),
        [x, topk_idx, topk_weights, combine],
    )


def bf16_mxfp8_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEBf16Mxfp8SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> None:
    del fast_math
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if y.shape != (n, symm_buffer.hidden) or y.dtype != torch.bfloat16:
        raise ValueError(f"y must be bfloat16 with shape ({n}, {symm_buffer.hidden}).")
    if (
        gate_up_clamp is not None
        and gate_up_clamp != symm_buffer._frontend.config.gate_up_clamp
    ):
        raise ValueError("gate_up_clamp is fixed when the mixed workspace is created.")
    result = symm_buffer._frontend.run(
        MegaMoEBf16Mxfp8Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            transformed_l2[0],
            transformed_l2[1],
            symm_buffer.combine_output,
        ),
        num_tokens=n,
        sync=sync,
    )
    y.copy_(
        result[:, 0]
        if symm_buffer._frontend.config.in_kernel_fc2_reduce
        else result.sum(dim=1)
    )


def bf16_mxfp8_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEBf16Mxfp8SymmBuffer,
) -> Callable[[], None]:
    return symm_buffer._frontend.make_launch_thunk(
        MegaMoEBf16Mxfp8Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            transformed_l2[0],
            transformed_l2[1],
            symm_buffer.combine_output,
        )
    )


__all__ = [
    "MegaMoEBf16Mxfp8Config",
    "MegaMoEBf16Mxfp8Frontend",
    "MegaMoEBf16Mxfp8Inputs",
    "MegaMoEBf16Mxfp8SymmBuffer",
    "TransformedWeights",
    "get_symm_buffer_for_bf16_mxfp8_mega_moe",
    "init_dist",
    "bf16_mxfp8_mega_launch_thunk",
    "bf16_mxfp8_mega_moe",
]
