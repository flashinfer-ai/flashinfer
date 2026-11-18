# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""Prepared fused BF16 routing GEMM and normalized top-k weights on SM120 (CuTe DSL).

``MegaGateSm12xPlan`` keeps the contract of the SM100a/SM103a/SM107a
``deepgemm_mega_gate.mega_gate.MegaGatePlan``: preparation validates, selects
the schedule, owns every allocation and compiles the kernel (once per schedule
and input signature, cached on disk); ``run()`` launches one kernel on the
current PyTorch stream without allocating, so it can be captured in a CUDA
graph. Operand values, biases and mapping tables may change in place between
runs; new addresses or shapes need a new plan.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from .heuristics import BLOCK_K, MAX_EXPERTS, MegaGateConfig, select_config

EXPERTS_PER_LANE = 4
"""The gate reads 4 consecutive experts per lane vector (as DeepGEMM)."""

SCORING_FUNCS = ("sqrtsoftplus",)
MAX_SLOTS = 32
"""Routed plus shared slots of one token must fit the 32 lanes of its gate warp."""

SCORE_BARRIER_WORDS = 16
"""uint64 words of one token-block arrival counter line (128 B), as on SM100."""

_ARCHES = {(12, 0): "sm_120a"}
"""Compute capability -> CuTe DSL code target.

SM121 (GB10) runs the same instructions (``sm_121a``), but it is not validated
yet: add ``(12, 1): "sm_121a"`` here once the tests pass there."""

JIT_MODULE_NAME = "sm12x_mega_gate_v1"
"""On-disk CuTe DSL cache namespace; bump the suffix to drop every artifact."""


def device_facts(device_index):
    """``(arch, sm_count)`` of CUDA device ``device_index``, or ``RuntimeError``."""
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    capability = (properties.major, properties.minor)
    arch = _ARCHES.get(capability)
    if arch is None:
        raise RuntimeError(
            f"the SM12x Mega Gate backend has no validated target for compute capability "
            f"{capability}; validated: {sorted(_ARCHES)}"
        )
    from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported

    if not is_cute_dsl_arch_supported(*capability, native_only=True):
        raise RuntimeError(f"the installed CuTe DSL cannot target {arch}")
    return arch, int(properties.multi_processor_count)


@dataclass(frozen=True)
class KernelSpec:
    """Every compile-time parameter of one kernel; nothing that varies per call."""

    config: MegaGateConfig
    hidden: int
    num_experts: int
    num_topk: int
    num_shared: int
    has_bias: bool
    has_image_bias: bool
    has_mask: bool
    has_physical_map: bool
    has_unmapped: bool
    enable_pdl: bool = False
    has_fixed_routing: bool = False

    @property
    def name(self) -> str:
        c = self.config
        flags = "".join(
            "1" if flag else "0"
            for flag in (
                self.has_bias,
                self.has_image_bias,
                self.has_mask,
                self.has_physical_map,
                self.has_unmapped,
            )
        )
        return (
            f"bt{c.block_tokens}_s{c.num_split_k}_st{c.num_stages}_g{c.num_gate_warps}"
            f"{'c' if c.cooperative_gate else ''}"
            f"{f'_n{c.mma_warps_n}' if c.mma_warps_n != 1 else ''}"
            f"{'_pdl' if self.enable_pdl else ''}{'_fixed' if self.has_fixed_routing else ''}"
            f"_k{self.hidden}"
            f"_e{self.num_experts}_top{self.num_topk}_sh{self.num_shared}_f{flags}"
        )


def _fake_operands(spec: KernelSpec):
    """Compile-time stand-ins for the runtime arguments, in ``Sm12xMegaGateKernel`` order.

    Token counts and the lengths of the optional inputs stay symbolic, so one
    program serves every token count and every absent-input placeholder.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8

    fake = cute.runtime.make_fake_compact_tensor
    sym = cute.sym_int
    slots = spec.num_topk + spec.num_shared
    return (
        fake(
            cutlass.BFloat16,
            (sym(), spec.hidden),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        fake(
            cutlass.BFloat16,
            (spec.num_experts, spec.hidden),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        fake(Float32, (sym(),), assumed_align=16),  # bias, read as float4
        fake(Float32, (sym(),), assumed_align=16),  # image_bias
        fake(Uint8, (sym(),), assumed_align=1),  # image_token_mask
        fake(Uint8, (sym(),), assumed_align=1),  # mask
        fake(Uint8, (sym(),), assumed_align=1),  # fix_routing_mask
        fake(Int32, (sym(),), assumed_align=4),  # to_physical_map, flattened
        fake(Int32, (sym(),), assumed_align=16),  # logical_count, read as int4
        fake(Int64, (sym(), slots), stride_order=(1, 0), assumed_align=8),
        fake(Float32, (sym(), slots), stride_order=(1, 0), assumed_align=4),
        cute.runtime.make_fake_tensor(
            Int64, (sym(), spec.num_topk), (sym(), 1), assumed_align=8
        ),  # unmapped_topk_idx, rows may be strided
        fake(Float32, (sym(),), assumed_align=16),  # scratch, flattened
        fake(Int32, (sym(),), assumed_align=4),  # score barriers as int32 words
        Int32(1),  # num_tokens
        Int32(0),  # ep_rank
        Float32(1.0),  # routed_scale
        Int32(1),  # map_width
        Int32(1),  # num_worker_groups
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )


@functools.cache
def load_kernel(arch: str, device_index: int, spec: KernelSpec):
    """Compile (or reload from FlashInfer's on-disk CuTe DSL cache) one specialization."""
    import torch

    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    from . import heuristics
    from . import kernel as kernel_module

    def compile_fn():
        import cutlass.cute as cute

        gate = kernel_module.Sm12xMegaGateKernel(
            spec.config,
            hidden=spec.hidden,
            num_experts=spec.num_experts,
            num_topk=spec.num_topk,
            num_shared=spec.num_shared,
            has_bias=spec.has_bias,
            has_image_bias=spec.has_image_bias,
            has_mask=spec.has_mask,
            has_physical_map=spec.has_physical_map,
            has_unmapped=spec.has_unmapped,
            enable_pdl=spec.enable_pdl,
            has_fixed_routing=spec.has_fixed_routing,
        )
        # The subscript form: ``options=`` with a tuple silently drops TVM-FFI and
        # falls back to the several-times slower ctypes argument path.
        return cute.compile[(cute.EnableTVMFFI(True), cute.GPUArch(arch))](
            gate, *_fake_operands(spec)
        )

    # The cache directory follows the current device's architecture.
    with torch.cuda.device(device_index):
        return build_and_load_cute_dsl_kernel(
            JIT_MODULE_NAME,
            spec.name,
            compile_fn,
            extra_key_files=(kernel_module.__file__, heuristics.__file__, __file__),
        )


class MegaGateSm12xPlan:
    """Bind a fused routing operation on SM120; run() returns (expert_indices, weights).

    x is BF16[M,K] and weight BF16[E,K], both contiguous and 16-byte aligned (K a
    multiple of 64, E <= 1024 and a multiple of 4). The optional bias and image bias
    are FP32[E]; masks are bool[M]. Physical mapping uses int32[E+shared,width] and
    logical_count int32[E+shared]. The optional logical top-k output is
    int64[M,topk] with unit column stride. Outputs are int64[M,topk+shared] indices
    and FP32 weights: the unbiased scores normalized over the top-k and scaled by
    routed_scaling_factor, weight 1 on shared slots, index -1 and weight 0 for
    masked tokens. Deterministic mode never splits K. Tokens flagged in
    fix_routing_mask (bool[M]) skip the top-k and take the logical experts already
    in their unmapped_topk_idx row (required; read before the row is written back,
    with ids clamped into [0, E) and masked tokens set to -1), weighted from their
    unbiased scores; force_random is not supported.

    Optional scratch (FP32 split-K partial logits) and score_barriers
    (uint64[token blocks,16] arrival counters, zeroed once before first use; the
    kernel rearms them) are caller-owned. Do not use one plan concurrently across
    streams.
    """

    def __init__(
        self,
        x,
        weight,
        num_topk=6,
        *,
        scoring_func="sqrtsoftplus",
        bias=None,
        image_bias=None,
        image_token_mask=None,
        mask=None,
        to_physical_map=None,
        logical_count=None,
        fix_routing_mask=None,
        force_random=None,
        unmapped_topk_idx=None,
        use_shared_as_routed=False,
        num_shared_experts=1,
        routed_scaling_factor=1.5,
        ep_rank=0,
        out=None,
        deterministic=False,
        scratch=None,
        score_barriers=None,
        descriptor_workspace=None,
        enable_pdl=None,
        num_sms=None,
        _config=None,
    ):
        """``enable_pdl`` launches with programmatic dependent launch (default: when the
        device supports it): launch and prologue can overlap the previous kernel,
        but global memory reads wait for it to finish. ``num_sms`` caps the SMs the
        kernel uses (default: all), leaving the rest to kernels that run at the same time,
        such as communication overlapped with the gate: a worker group spins until all
        of its CTAs have arrived, so every CTA of the grid must find a free SM.
        ``_config`` (a ``MegaGateConfig``) overrides the heuristic; for tuning and tests."""
        import torch

        from flashinfer.utils import device_support_pdl

        if x.device.type != "cuda":
            raise RuntimeError("Mega Gate requires a CUDA device")
        device_index = (
            x.device.index
            if x.device.index is not None
            else torch.cuda.current_device()
        )
        arch, sms = device_facts(device_index)
        if num_sms is not None:
            if not 0 < int(num_sms) <= sms:
                raise ValueError(f"num_sms must be in [1, {sms}]")
            sms = int(num_sms)
        if (
            x.ndim != 2
            or weight.ndim != 2
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
        ):
            raise ValueError("x and weight must be contiguous BF16 matrices")
        M, K = x.shape
        E, wk = weight.shape
        if wk != K or not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("x and weight must be contiguous and share K")
        if weight.device != x.device:
            raise ValueError("x and weight must be on the same device")
        if x.data_ptr() % 16 or weight.data_ptr() % 16:
            raise ValueError("x and weight must be 16-byte aligned (TMA loads them)")
        if K <= 0 or K % BLOCK_K:
            raise ValueError(
                f"the hidden size must be a positive multiple of {BLOCK_K}"
            )
        if not 0 < E <= MAX_EXPERTS:
            raise ValueError(f"the expert count must be in [1, {MAX_EXPERTS}]")
        if E % EXPERTS_PER_LANE:
            raise ValueError(
                f"the expert count must be a multiple of {EXPERTS_PER_LANE}"
            )
        if scoring_func not in SCORING_FUNCS:
            raise NotImplementedError(
                f"the SM12x Mega Gate supports {SCORING_FUNCS} scoring, got {scoring_func!r}"
            )
        if force_random is not None:
            raise NotImplementedError(
                "random routing is not supported by the SM12x Mega Gate yet"
            )
        if fix_routing_mask is not None and unmapped_topk_idx is None:
            raise ValueError(
                "fix_routing_mask needs unmapped_topk_idx: its rows hold the fixed "
                "tokens' logical experts"
            )
        if descriptor_workspace is not None:
            raise ValueError(
                "the SM12x programs build their TMA descriptors per launch; "
                "no descriptor workspace applies"
            )
        if not 0 < num_topk <= min(E, MAX_SLOTS):
            raise ValueError(f"num_topk must be in [1, min(E, {MAX_SLOTS})]")
        shared = num_shared_experts if use_shared_as_routed else 0
        if shared and (
            shared not in (1, 2) or num_topk % shared or E % (num_topk // shared)
        ):
            raise ValueError("shared experts require the shared/top-k divisibility")
        if num_topk + shared > MAX_SLOTS:
            raise ValueError("routed and shared output slots must fit one warp")
        if not 0 <= ep_rank <= 0x7FFFFFFF:
            raise ValueError("ep_rank must fit nonnegative int32")
        if not math.isfinite(routed_scaling_factor):
            raise ValueError("routed_scaling_factor must be finite")
        if (image_bias is None) != (image_token_mask is None):
            raise ValueError(
                "image_bias and image_token_mask must be supplied together"
            )
        if (to_physical_map is None) != (logical_count is None):
            raise ValueError(
                "to_physical_map and logical_count must be supplied together"
            )

        device = x.device
        slots = num_topk + shared

        def check(tensor, name, dtype, shape):
            if (
                tensor.dtype != dtype
                or tuple(tensor.shape) != tuple(shape)
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype}{list(shape)} on {device}"
                )

        def check_vector_aligned(tensor, name):
            if tensor.data_ptr() % 16:
                raise ValueError(
                    f"{name} must be 16-byte aligned (it is read as float4)"
                )

        if bias is not None:
            check(bias, "bias", torch.float32, (E,))
            check_vector_aligned(bias, "bias")
        if image_bias is not None:
            check(image_bias, "image_bias", torch.float32, (E,))
            check_vector_aligned(image_bias, "image_bias")
            check(image_token_mask, "image_token_mask", torch.bool, (M,))
        if mask is not None:
            check(mask, "mask", torch.bool, (M,))
        if fix_routing_mask is not None:
            check(fix_routing_mask, "fix_routing_mask", torch.bool, (M,))
        if logical_count is not None:
            check(logical_count, "logical_count", torch.int32, (E + shared,))
            check_vector_aligned(logical_count, "logical_count")
            if (
                to_physical_map.dtype != torch.int32
                or to_physical_map.ndim != 2
                or to_physical_map.shape[0] != E + shared
                or to_physical_map.shape[1] < 1
                or to_physical_map.device != device
                or not to_physical_map.is_contiguous()
            ):
                raise ValueError(
                    "to_physical_map must be contiguous int32[E+shared,width]"
                )
        if unmapped_topk_idx is not None and (
            unmapped_topk_idx.dtype != torch.int64
            or tuple(unmapped_topk_idx.shape) != (M, num_topk)
            or unmapped_topk_idx.stride(1) != 1
            or (M > 1 and unmapped_topk_idx.stride(0) < num_topk)
            or unmapped_topk_idx.device != device
        ):
            raise ValueError(
                "unmapped_topk_idx must be int64[M,topk] with unit column stride "
                "and rows that do not overlap"
            )

        config = (
            _config
            if _config is not None
            else select_config(M, K, E, sms, bool(deterministic))
        )
        if config.num_split_k <= 0 or (K // BLOCK_K) % config.num_split_k:
            raise ValueError("num_split_k must be positive and divide K / BLOCK_K")
        if config.num_logical_ctas > sms:
            raise ValueError(
                f"the schedule's {config.num_logical_ctas} CTAs per token block "
                f"exceed num_sms={sms}"
            )
        # Persistent groups of num_logical_ctas CTAs; at most one CTA per SM keeps them co-resident.
        num_worker_groups = config.num_worker_groups(M, sms)
        if out is None:
            out = (
                torch.empty((M, slots), dtype=torch.int64, device=device),
                torch.empty((M, slots), dtype=torch.float32, device=device),
            )
        if len(out) != 2:
            raise ValueError("out must be a pair of tensors")
        check(out[0], "out[0]", torch.int64, (M, slots))
        check(out[1], "out[1]", torch.float32, (M, slots))
        # Physical indices, weights and logical indices are separate kernel stores.
        # Outputs are contiguous; the logical output may have padding between rows.
        outputs = [("out[0]", out[0]), ("out[1]", out[1])]
        if unmapped_topk_idx is not None:
            outputs.append(("unmapped_topk_idx", unmapped_topk_idx))
        for i, (name, tensor) in enumerate(outputs):
            if not tensor.numel():
                continue
            start = tensor.data_ptr()
            end = (
                start + (tensor.shape[0] - 1) * tensor.stride(0) * tensor.element_size()
            )
            end += tensor.shape[1] * tensor.element_size()
            for other_name, other in outputs[:i]:
                other_start = other.data_ptr()
                other_end = other_start + other.numel() * other.element_size()
                if start < other_end and other_start < end:
                    # Earlier outputs are contiguous. Find the first logical row
                    # whose end lies after their start, allowing disjoint views.
                    row_bytes = tensor.shape[1] * tensor.element_size()
                    stride_bytes = max(
                        row_bytes, tensor.stride(0) * tensor.element_size()
                    )
                    row = max(0, (other_start - start - row_bytes) // stride_bytes + 1)
                    if row < tensor.shape[0] and start + row * stride_bytes < other_end:
                        raise ValueError(f"{name} and {other_name} must not overlap")
        scratch_shape = config.scratch_shape(M, E)
        if scratch is None:
            scratch = torch.empty(scratch_shape, dtype=torch.float32, device=device)
        check(scratch, "scratch", torch.float32, scratch_shape)
        check_vector_aligned(scratch, "scratch")
        barrier_shape = (config.num_token_blocks(M), SCORE_BARRIER_WORDS)
        if score_barriers is None:
            score_barriers = torch.zeros(
                barrier_shape, dtype=torch.uint64, device=device
            )
        check(score_barriers, "score_barriers", torch.uint64, barrier_shape)

        spec = KernelSpec(
            config=config,
            hidden=K,
            num_experts=E,
            num_topk=num_topk,
            num_shared=shared,
            has_bias=bias is not None,
            has_image_bias=image_bias is not None,
            has_mask=mask is not None,
            has_physical_map=to_physical_map is not None,
            has_unmapped=unmapped_topk_idx is not None,
            has_fixed_routing=fix_routing_mask is not None,
            enable_pdl=device_support_pdl(x.device)
            if enable_pdl is None
            else bool(enable_pdl),
        )
        # Placeholders for absent inputs: never read, but every argument needs a tensor.
        dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)
        dummy_u8 = torch.zeros(1, dtype=torch.uint8, device=device)
        dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        dummy_i64 = torch.zeros((1, num_topk), dtype=torch.int64, device=device)
        self._kernel = load_kernel(arch, device_index, spec) if M else None
        self._args = (
            x,
            weight,
            bias if bias is not None else dummy_f32,
            image_bias if image_bias is not None else dummy_f32,
            image_token_mask.view(torch.uint8) if image_bias is not None else dummy_u8,
            mask.view(torch.uint8) if mask is not None else dummy_u8,
            fix_routing_mask.view(torch.uint8)
            if fix_routing_mask is not None
            else dummy_u8,
            to_physical_map.view(-1) if to_physical_map is not None else dummy_i32,
            logical_count if logical_count is not None else dummy_i32,
            out[0],
            out[1],
            unmapped_topk_idx if unmapped_topk_idx is not None else dummy_i64,
            scratch.view(-1),
            score_barriers.view(torch.int32).view(-1),
            M,
            ep_rank,
            float(routed_scaling_factor),
            to_physical_map.shape[1] if to_physical_map is not None else 1,
            num_worker_groups,
        )
        self.route = dict(
            arch=arch,
            kernel=spec.name,
            block_tokens=config.block_tokens,
            num_split_k=config.num_split_k,
            num_expert_groups=config.num_expert_groups,
            num_stages=config.num_stages,
            num_gate_warps=config.num_gate_warps,
            mma_warps_n=config.mma_warps_n,
            cooperative_gate=config.cooperative_gate,
            enable_pdl=spec.enable_pdl,
            num_worker_groups=num_worker_groups,
            launch_ctas=num_worker_groups * config.num_logical_ctas,
            num_sms=sms,
        )
        self.outputs, self.scratch, self.score_barriers = out, scratch, score_barriers

    def run(self):
        if self._kernel is not None:
            self._kernel(*self._args)
        return self.outputs
