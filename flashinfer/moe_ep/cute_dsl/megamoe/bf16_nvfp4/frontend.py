# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy-compile NVFP4-weight, BF16-activation MegaMoE API."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import torch

from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
    _CompiledMega,
    _compute_peer_offsets,
    ensure_not_capturing,
    free_sym_tensor,
    init_dist,
    resolve_gate_up_clamp,
    sym_zeros,
)


@dataclasses.dataclass(frozen=True)
class MegaMoEBf16Nvfp4Config:
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
    enable_in_kernel_fc2_reduce: bool = False
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    gate_up_clamp: Optional[float] = None
    apply_topk_in_fc1: bool = False
    enable_iket: bool = False
    swiglu_alpha: Optional[float] = None
    swiglu_beta: Optional[float] = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: Optional[float] = None
    situ_linear_beta: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
        if self.activation not in ("swiglu", "situ"):
            raise ValueError(
                f"activation must be 'swiglu' or 'situ', got {self.activation!r}."
            )
        if self.activation == "situ":
            if self.swiglu_alpha is not None:
                raise ValueError("SwiGLU parameters are not supported with SiTU.")
            if self.situ_beta is None:
                raise ValueError("activation='situ' requires situ_beta.")
            if not math.isfinite(self.situ_beta) or self.situ_beta <= 0:
                raise ValueError("situ_beta must be positive and finite.")
            if self.situ_linear_beta is not None and (
                not math.isfinite(self.situ_linear_beta) or self.situ_linear_beta <= 0
            ):
                raise ValueError(
                    "situ_linear_beta must be positive and finite when set."
                )
            if self.gate_up_clamp is not None:
                raise ValueError("gate_up_clamp is not supported with SiTU.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU parameters require activation='situ'.")
        # Geometry knobs also arrive as JSON arrays from the benchmark.
        object.__setattr__(self, "mma_tiler_mnk", tuple(self.mma_tiler_mnk))
        object.__setattr__(self, "cluster_shape_mnk", tuple(self.cluster_shape_mnk))
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"Unsupported load_balance_mode={self.load_balance_mode!r}."
            )
        if self.in_kernel_fc2_reduce and not self.enable_in_kernel_fc2_reduce:
            raise ValueError(
                "in_kernel_fc2_reduce knob selected without enable_in_kernel_fc2_reduce."
            )
        if self.in_kernel_fc2_reduce and self.token_back_mode != "reuse_dispatch_warps":
            raise ValueError(
                "in_kernel_fc2_reduce requires token_back_mode='reuse_dispatch_warps'."
            )
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

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size


@dataclass
class MegaMoEBf16Nvfp4Inputs:
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
    reduced_output: Optional[torch.Tensor] = None
    staging_inputs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    fc1_norm_const: Optional[torch.Tensor] = None


class MegaMoEBf16Nvfp4Frontend:
    """Host wrapper for ``Sm100W4A16MegaMoEKernel``."""

    def __init__(self, config: MegaMoEBf16Nvfp4Config) -> None:
        self._config = config
        self._mega: Optional[_CompiledMega] = None
        self._reduce = None
        self._staging_variants: dict[tuple[torch.dtype, bool, bool], Callable] = {}
        self._unstaged_variants: dict[bool, Callable] = {}
        self._launch_inputs: Optional[MegaMoEBf16Nvfp4Inputs] = None

    @property
    def config(self) -> MegaMoEBf16Nvfp4Config:
        return self._config

    def set_gate_up_clamp(self, clamp: Optional[float]) -> None:
        if self._config.gate_up_clamp != clamp:
            new_config = dataclasses.replace(self._config, gate_up_clamp=clamp)
            ensure_not_capturing("set_gate_up_clamp (clamp change)")
            self.release()
            self._config = new_config

    def set_swiglu_params(self, alpha: Optional[float], beta: Optional[float]) -> None:
        """Change uniform activation constants outside capture, then re-warm."""
        new_config = dataclasses.replace(
            self._config, swiglu_alpha=alpha, swiglu_beta=beta
        )
        if new_config != self._config:
            ensure_not_capturing("set_swiglu_params (activation change)")
            self.release()
            self._config = new_config

    def apply_knobs(self, knobs: dict) -> None:
        """Apply a validated swapped-MMA tuning configuration and invalidate its compile."""
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import tuner, with_knobs

        if not tuner.is_valid_bf16_nvfp4_for_config(self.config, knobs):
            raise ValueError(
                f"unsupported BF16/NVFP4 MegaMoE knobs {knobs}: "
                f"{tuner.describe_invalid_knobs(self.config, knobs, tuner.is_valid_bf16_nvfp4_for_config)}."
            )
        # Numerical behavior belongs to the session, not the tuner.
        new_config = with_knobs(
            self.config,
            {
                **knobs,
                "gate_up_clamp": self.config.gate_up_clamp,
                "apply_topk_in_fc1": self.config.apply_topk_in_fc1,
                "enable_in_kernel_fc2_reduce": self.config.enable_in_kernel_fc2_reduce,
                "swiglu_alpha": self.config.swiglu_alpha,
                "swiglu_beta": self.config.swiglu_beta,
                "activation": self.config.activation,
                "situ_beta": self.config.situ_beta,
                "situ_linear_beta": self.config.situ_linear_beta,
            },
        )
        if new_config != self._config:
            ensure_not_capturing("apply_knobs (config change)")
            self.release()
            self._config = new_config

    def release(self) -> None:
        if self._mega is not None:
            ensure_not_capturing("workspace release (symmetric-heap free)")
            free_sym_tensor(self._mega.shared_workspace)
        self._mega = None
        self._staging_variants.clear()
        self._unstaged_variants.clear()
        self._launch_inputs = None

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

    def _ensure_compiled(self, inputs: MegaMoEBf16Nvfp4Inputs) -> _CompiledMega:
        # All input specializations share one collective workspace allocation.
        if self._mega is None:
            ensure_not_capturing("cute.compile + symmetric-heap allocation")
            import cutlass
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
                num_topk=c.num_topk,
                max_tokens_per_rank=c.num_tokens_per_rank,
                hidden=c.hidden,
                in_kernel_fc2_reduce=c.in_kernel_fc2_reduce,
                token_back_mode=c.token_back_mode,
                epi_flag_batch=c.epi_flag_batch,
                flag_batch=c.flag_batch,
                gate_up_clamp=c.gate_up_clamp,
                apply_topk_in_fc1=c.apply_topk_in_fc1,
                swiglu_alpha=c.swiglu_alpha,
                swiglu_beta=c.swiglu_beta,
                situ_beta=c.situ_beta,
                situ_linear_beta=c.situ_linear_beta,
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
            self._mega = mega
        mega = self._mega
        use_norm = inputs.fc1_norm_const is not None
        if inputs.staging_inputs is None:
            if use_norm not in self._unstaged_variants:
                self._unstaged_variants[use_norm] = self._compile(inputs, mega)
            mega.compiled = self._unstaged_variants[use_norm]
        else:
            key = (*self._staging_key(inputs.staging_inputs), use_norm)
            if key not in self._staging_variants:
                self._staging_variants[key] = self._compile(inputs, mega, key)
        return mega

    def _warm_staging_variants(
        self, inputs: Optional[MegaMoEBf16Nvfp4Inputs] = None
    ) -> None:
        inputs = self._launch_inputs if inputs is None else inputs
        if inputs is None or inputs.staging_inputs is None:
            return
        # Public preparation covers first-use layouts in a serving graph;
        # private autotune trials compile only the ABI they actually launch.
        assert self._mega is not None
        for dtype in (torch.int32, torch.int64):
            for vector_copy in (False, True):
                key = (dtype, vector_copy, inputs.fc1_norm_const is not None)
                if key not in self._staging_variants:
                    self._staging_variants[key] = self._compile(inputs, self._mega, key)

    def staging_variants_ready(self, use_norm: bool) -> bool:
        return self._mega is not None and all(
            (dtype, vector_copy, use_norm) in self._staging_variants
            for dtype in (torch.int32, torch.int64)
            for vector_copy in (False, True)
        )

    def _compile(
        self,
        inputs: MegaMoEBf16Nvfp4Inputs,
        mega: _CompiledMega,
        staging_key: Optional[tuple[torch.dtype, bool, bool]] = None,
    ):
        ensure_not_capturing("MegaMoE input specialization cute.compile")
        import cutlass.cute as cute

        c = self.config
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        cluster_size = c.cluster_shape_mnk[0] * c.cluster_shape_mnk[1]
        kwargs = self._runtime_kwargs(inputs, mega)
        if staging_key is not None:
            # Typed null pointers are compile descriptors, never launch inputs.
            kwargs["staging_inputs"] = self._staging_args(None, staging_key[:2])
        kwargs["max_active_clusters"] = max(1, sm_count // cluster_size)
        # Start with 61440 registers per CTA; the kernel redistributes this
        # pool among the five warpgroup roles.
        kwargs["options"] = "--ptxas-options='-maxrregcount=96'"
        if c.enable_iket:
            kwargs["options"] += " iket"
        return cute.compile(mega.kernel, **kwargs)

    def _staging_key(self, sources) -> tuple[torch.dtype, bool]:
        hidden, ids, _ = sources
        vector_copy = (
            hidden.stride() == (self.config.hidden, 1) and hidden.data_ptr() % 16 == 0
        )
        return ids.dtype, vector_copy

    def _staging_args(self, sources, key):
        import cutlass
        from cutlass.cute.runtime import make_ptr, nullptr
        from cutlass.cute.typing import AddressSpace

        ids_dtype, vector_copy = key
        dtypes = (
            cutlass.BFloat16,
            {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}[ids_dtype],
            cutlass.Float32,
        )
        alignments = (16 if vector_copy else 2, dtypes[1].width // 8, 4)
        args = [cutlass.Int32(0 if sources is None else sources[0].shape[0])]
        for index, (dtype, alignment) in enumerate(
            zip(dtypes, alignments, strict=True)
        ):
            tensor = None if sources is None else sources[index]
            pointer = (
                nullptr(dtype, AddressSpace.gmem, assumed_align=alignment)
                if tensor is None
                else make_ptr(
                    dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=alignment
                )
            )
            stride = (
                # None preserves this layout choice across the JIT boundary;
                # an unannotated tuple of Python integers becomes runtime args.
                None
                if index == 0 and vector_copy
                else tuple(
                    cutlass.Int64(s)
                    for s in ((0, 0) if tensor is None else tensor.stride())
                )
            )
            args.append((pointer, stride))
        return tuple(args)

    def _runtime_kwargs(
        self, inputs: MegaMoEBf16Nvfp4Inputs, mega: _CompiledMega
    ) -> dict:
        import cuda.bindings.driver as cuda
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import SymBufferHost

        c = self.config
        mapper = SymBufferHost(
            base_addr=mega.symmetric_base,
            offsets=tuple(mega.peer_offsets_list),
            rank_idx=c.rank,
            num_max_ranks=c.world_size,
        )
        reduced_output = inputs.reduced_output
        if reduced_output is not None:
            # Live rows vary independently of workspace capacity, including zero.
            reduced_output = self._to_cute(
                reduced_output, static_layout=True
            ).mark_compact_shape_dynamic(
                mode=0, stride_order=reduced_output.dim_order(), divisibility=1
            )
        staging_inputs = (
            None
            if inputs.staging_inputs is None
            else self._staging_args(
                inputs.staging_inputs, self._staging_key(inputs.staging_inputs)
            )
        )
        return {
            "staging_inputs": staging_inputs,
            "activation": self._to_cute(inputs.activation),
            "topk_idx": self._to_cute(inputs.topk_idx),
            "topk_weights": self._to_cute(inputs.topk_weights, static_layout=True),
            "fc1_weight": self._to_cute(inputs.fc1_weight),
            "fc1_weight_sf": self._to_cute(inputs.fc1_weight_sf),
            "fc1_alpha": self._to_cute(inputs.fc1_alpha, assumed_align=4),
            "fc2_weight": self._to_cute(inputs.fc2_weight),
            "fc2_weight_sf": self._to_cute(inputs.fc2_weight_sf),
            "fc2_alpha": self._to_cute(inputs.fc2_alpha, assumed_align=4),
            "fc1_norm_const": None
            if inputs.fc1_norm_const is None
            else self._to_cute(inputs.fc1_norm_const, assumed_align=4),
            "combine_output": self._to_cute(inputs.combine_output, static_layout=True),
            "reduced_output": reduced_output,
            "local_workspace": self._to_cute(mega.local_workspace, static_layout=True),
            "shared_workspace": self._to_cute(mega.shared_workspace),
            "peer_rank_ptr_mapper_host": mapper,
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def run(
        self,
        inputs: MegaMoEBf16Nvfp4Inputs,
        *,
        num_tokens: Optional[int] = None,
        sync: bool = False,
    ) -> torch.Tensor:
        n = inputs.activation.shape[0] if num_tokens is None else num_tokens
        self._validate(inputs, n)
        mega = self._ensure_compiled(inputs)
        key = (
            tuple(
                (t.data_ptr(), t.dtype, tuple(t.shape), t.stride())
                for t in inputs.staging_inputs
            )
            if inputs.staging_inputs is not None
            else None,
            inputs.activation.data_ptr(),
            inputs.topk_idx.data_ptr(),
            inputs.topk_weights.data_ptr(),
            inputs.fc1_weight.data_ptr(),
            inputs.fc1_weight_sf.data_ptr(),
            inputs.fc1_alpha.data_ptr(),
            inputs.fc2_weight.data_ptr(),
            inputs.fc2_weight_sf.data_ptr(),
            inputs.fc2_alpha.data_ptr(),
            None if inputs.fc1_norm_const is None else inputs.fc1_norm_const.data_ptr(),
            inputs.combine_output.data_ptr(),
            (
                inputs.reduced_output.data_ptr(),
                tuple(inputs.reduced_output.shape),
                inputs.reduced_output.stride(),
            )
            if inputs.reduced_output is not None
            else None,
            torch.cuda.current_stream().cuda_stream,
        )
        if mega.launch_key != key:
            mega.launch_kwargs = self._runtime_kwargs(inputs, mega)
            mega.launch_key = key
            self._launch_inputs = inputs
        if self.config.in_kernel_fc2_reduce:
            inputs.combine_output.zero_()
        compiled = (
            mega.compiled
            if inputs.staging_inputs is None
            else self._staging_variants[
                (
                    *self._staging_key(inputs.staging_inputs),
                    inputs.fc1_norm_const is not None,
                )
            ]
        )
        compiled(**mega.launch_kwargs)
        if sync:
            torch.cuda.synchronize()
        return inputs.combine_output[:n]

    def make_launch_thunk(self, inputs: MegaMoEBf16Nvfp4Inputs) -> Callable[[], None]:
        self._validate(
            inputs,
            inputs.reduced_output.shape[0]
            if inputs.reduced_output is not None
            else inputs.activation.shape[0],
        )
        mega = self._ensure_compiled(inputs)
        self._warm_staging_variants(inputs)
        kwargs = self._runtime_kwargs(inputs, mega)
        compiled = (
            mega.compiled
            if inputs.staging_inputs is None
            else self._staging_variants[
                (
                    *self._staging_key(inputs.staging_inputs),
                    inputs.fc1_norm_const is not None,
                )
            ]
        )
        if self.config.in_kernel_fc2_reduce:

            def thunk(_inputs=inputs):
                _inputs.combine_output.zero_()
                compiled(**kwargs)

            return thunk
        else:
            # The raw launch helper owns its scratch output through this closure.
            def thunk(_inputs=inputs):
                compiled(**kwargs)

            return thunk

    def _validate(self, inputs: MegaMoEBf16Nvfp4Inputs, num_tokens: int) -> None:
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
        norm = inputs.fc1_norm_const
        if norm is not None and (
            norm.shape != (c.num_experts_per_rank,)
            or norm.dtype != torch.float32
            or norm.device != inputs.activation.device
            or not norm.is_contiguous()
        ):
            raise ValueError(
                "fc1_norm_const must be contiguous FP32 [local_experts] on the activation device."
            )
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
                raise ValueError("fc1_alpha/fc2_alpha must be FP32 per expert.")
            if not all(t.is_cuda and t.is_contiguous() for t in (scale, alpha)):
                raise ValueError("scales and alphas must be contiguous CUDA tensors.")
        combine_topk = 1 if c.in_kernel_fc2_reduce else c.num_topk
        if inputs.combine_output.shape != (
            c.num_tokens_per_rank,
            combine_topk,
            c.hidden,
        ) or (
            not inputs.combine_output.is_cuda
            or inputs.combine_output.dtype != torch.bfloat16
        ):
            raise ValueError(
                "combine_output must be CUDA BF16 with the expected shape."
            )
        output = inputs.reduced_output
        if c.in_kernel_fc2_reduce:
            if output is not None:
                raise ValueError("in_kernel_fc2_reduce does not use reduced_output.")
        elif (
            output is None
            or output.shape != (num_tokens, c.hidden)
            or output.dtype != torch.bfloat16
            or output.device != inputs.activation.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "reduced_output must be contiguous BF16 on the activation device "
                f"with shape ({num_tokens}, {c.hidden})."
            )

    def reduce_topk(self, combined, scores, output):
        """Combine BF16 FC2 partials in FP32, weighting if FC1 did not."""
        # Empty source ranks still execute the collective fused kernel, but
        # have no local rows to reduce and must not launch a zero-block grid.
        if output.shape[0] == 0:
            return
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda
        from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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
            None if self.config.apply_topk_in_fc1 else compact(scores),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if self._reduce is None:
            ensure_not_capturing("top-k reducer cute.compile")
            reducer = TopkReduce(
                self.config.hidden,
                self.config.num_topk,
                CombineFormat.parse("bf16"),
                sm_arch=get_cutedsl_target_arch(),
            )
            self._reduce = cute.compile(reducer, *args)
        self._reduce(*args)


@dataclass
class MegaMoEBf16Nvfp4SymmBuffer:
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
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    _frontend: MegaMoEBf16Nvfp4Frontend
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False
    _staging_inputs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default=None, repr=False
    )
    fc1_norm_const: Optional[torch.Tensor] = None
    _use_fc1_norm_const: bool = False

    @property
    def kernel_combine_output(self) -> torch.Tensor:
        """A compact reduction target, or all top-k partials for external combine."""
        if self._frontend.config.in_kernel_fc2_reduce:
            # Both tactics reuse the same symmetric allocation during autotune.
            return self.combine_output.view(-1, 1, self.hidden)[: self.num_max_tokens]
        return self.combine_output

    def destroy(self) -> None:
        if not self._destroyed:
            self._staging_inputs = None
            self._frontend.release()
            for root in self._sym_roots:
                free_sym_tensor(root)
            self._sym_roots.clear()
            self._destroyed = True


TransformedWeights = Tuple[torch.Tensor, torch.Tensor]


def get_symm_buffer_for_bf16_nvfp4_mega_moe(
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
    enable_in_kernel_fc2_reduce: bool = False,
    apply_topk_in_fc1: bool = False,
    fc1_alpha: torch.Tensor | int | float | None = None,
    fc2_alpha: torch.Tensor | int | float | None = None,
    token_back_mode: Optional[
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
    ] = None,
    knobs: Optional[dict] = None,
    fc1_norm_const: torch.Tensor | int | float | None = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    activation: Literal["swiglu", "situ"] = "swiglu",
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
) -> MegaMoEBf16Nvfp4SymmBuffer:
    """Allocate stable per-expert scales and BF16 transport buffers.

    ``fc1_norm_const`` multiplies the post-activation/top-k values before their
    BF16 handoff, with no automatic reciprocal in ``fc2_alpha``. Omission keeps
    the unnormalized kernel specialization. Runtime overrides can enable the
    normalized specialization during eager warmup, using the same allocation.
    Later omission retains the last staged normalization. A captured graph
    retains its normalization mode; enabling it requires warmup and recapture.
    A normalized graph observes in-place updates to its captured source tensor.
    """
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        _resolve_per_expert_epilogue,
        resolve_knobs,
        tuner,
        with_knobs,
    )

    cfg = MegaMoEBf16Nvfp4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
        gate_up_clamp=clamp,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )

    # Match the existing Mega cache contract: None is a pure capacity-keyed
    # lookup; an explicit dict (including {}) bypasses cache and defaults.
    if knobs is None:
        resolved_knobs, _ = resolve_knobs(
            dtype="bf16_nvfp4",
            world_size=world_size,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_total_experts,
            topk=num_topk,
            max_tokens=num_max_tokens,
            combine_dtype="bf16",
            enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
            apply_topk_in_fc1=apply_topk_in_fc1,
        )
    else:
        resolved_knobs = {}
    optional_config = {
        **resolved_knobs,
        **({"token_back_mode": token_back_mode} if token_back_mode is not None else {}),
        **(knobs or {}),
        "gate_up_clamp": clamp,
        "enable_in_kernel_fc2_reduce": enable_in_kernel_fc2_reduce,
        "apply_topk_in_fc1": apply_topk_in_fc1,
        "swiglu_alpha": swiglu_alpha,
        "swiglu_beta": swiglu_beta,
        "activation": activation,
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
    }
    if not tuner.is_valid_bf16_nvfp4_for_config(cfg, optional_config):
        raise ValueError(
            f"unsupported BF16/NVFP4 MegaMoE knobs {optional_config}: "
            f"{tuner.describe_invalid_knobs(cfg, optional_config, tuner.is_valid_bf16_nvfp4_for_config)}."
        )
    cfg = with_knobs(cfg, optional_config)
    fc1_alpha = _resolve_per_expert_epilogue(
        "fc1_alpha", fc1_alpha, cfg.num_experts_per_rank
    )
    fc2_alpha = _resolve_per_expert_epilogue(
        "fc2_alpha", fc2_alpha, cfg.num_experts_per_rank
    )
    use_norm = fc1_norm_const is not None
    fc1_norm_const = _resolve_per_expert_epilogue(
        "fc1_norm_const", fc1_norm_const, cfg.num_experts_per_rank
    )
    x = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
    topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
    topk_idx.fill_(-1)
    topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
    combine_output = sym_zeros(
        (num_max_tokens, num_topk, hidden),
        torch.bfloat16,
    )
    return MegaMoEBf16Nvfp4SymmBuffer(
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
        fc1_alpha,
        fc2_alpha,
        MegaMoEBf16Nvfp4Frontend(cfg),
        [x, topk_idx, topk_weights, combine_output],
        fc1_norm_const=fc1_norm_const,
        _use_fc1_norm_const=use_norm,
    )


def bf16_nvfp4_mega_moe(
    y: torch.Tensor,
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEBf16Nvfp4SymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    sync: bool = False,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
) -> None:
    if symm_buffer._destroyed:
        raise RuntimeError("symm_buffer.destroy() was already called.")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if y.shape != (n, symm_buffer.hidden) or y.dtype != torch.bfloat16:
        raise ValueError(f"y must be bfloat16 with shape ({n}, {symm_buffer.hidden}).")
    if not y.is_cuda or not y.is_contiguous():
        raise ValueError("y must be a contiguous CUDA tensor.")
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp, activation_clamp=activation_clamp
    )
    if clamp is not None:
        symm_buffer._frontend.set_gate_up_clamp(clamp)
    if swiglu_alpha is not None or swiglu_beta is not None:
        symm_buffer._frontend.set_swiglu_params(swiglu_alpha, swiglu_beta)
    result = symm_buffer._frontend.run(
        MegaMoEBf16Nvfp4Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            symm_buffer.fc1_alpha,
            transformed_l2[0],
            transformed_l2[1],
            symm_buffer.fc2_alpha,
            symm_buffer.kernel_combine_output,
            None if symm_buffer._frontend.config.in_kernel_fc2_reduce else y,
            symm_buffer._staging_inputs,
            symm_buffer.fc1_norm_const if symm_buffer._use_fc1_norm_const else None,
        ),
        num_tokens=n,
    )
    if symm_buffer._frontend.config.in_kernel_fc2_reduce:
        y.copy_(result[:, 0])
    if sync:
        torch.cuda.synchronize()


def bf16_nvfp4_mega_launch_thunk(
    transformed_l1: TransformedWeights,
    transformed_l2: TransformedWeights,
    symm_buffer: MegaMoEBf16Nvfp4SymmBuffer,
) -> Callable[[], None]:
    reduced_output = None
    if not symm_buffer._frontend.config.in_kernel_fc2_reduce:
        ensure_not_capturing("launch thunk output allocation")
        # Keep this raw-core thunk compute-only while sharing the Tensor signature.
        reduced_output = torch.empty(
            (0, symm_buffer.hidden),
            dtype=torch.bfloat16,
            device=symm_buffer.x.device,
        )
    return symm_buffer._frontend.make_launch_thunk(
        MegaMoEBf16Nvfp4Inputs(
            symm_buffer.x,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            transformed_l1[0],
            transformed_l1[1],
            symm_buffer.fc1_alpha,
            transformed_l2[0],
            transformed_l2[1],
            symm_buffer.fc2_alpha,
            symm_buffer.kernel_combine_output,
            reduced_output,
            fc1_norm_const=(
                symm_buffer.fc1_norm_const if symm_buffer._use_fc1_norm_const else None
            ),
        )
    )


__all__ = [
    "MegaMoEBf16Nvfp4Config",
    "MegaMoEBf16Nvfp4Frontend",
    "MegaMoEBf16Nvfp4Inputs",
    "MegaMoEBf16Nvfp4SymmBuffer",
    "TransformedWeights",
    "bf16_nvfp4_mega_launch_thunk",
    "bf16_nvfp4_mega_moe",
    "get_symm_buffer_for_bf16_nvfp4_mega_moe",
    "init_dist",
]
