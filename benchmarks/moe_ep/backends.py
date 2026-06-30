"""Backend builders for MoE-EP-only DeepSeek V4 MegaMoE benchmarks."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from moe_ep_common import (
    BACKEND_IDS,
    BenchmarkInputs,
    BenchmarkWeights,
    FI_MEGAKERNEL_BY_BACKEND,
)

EXPERTS_PREFIX = "bench.ffn.experts"
GATE_PREFIX = "bench.ffn.gate"


@dataclass
class VllmMegaMoeContext:
    vllm_config: Any
    config_ctx: AbstractContextManager[Any]
    experts: nn.Module


def is_flashinfer_backend(backend_id: str) -> bool:
    return backend_id.startswith("fi_")


def megakernel_for_backend(backend_id: str) -> str:
    if backend_id not in FI_MEGAKERNEL_BY_BACKEND:
        raise KeyError(
            f"backend {backend_id!r} is not a flashinfer mega backend; "
            f"expected one of {sorted(FI_MEGAKERNEL_BY_BACKEND)}"
        )
    return FI_MEGAKERNEL_BY_BACKEND[backend_id]


def _float_ue8m0_scale_to_uint8(sf: torch.Tensor) -> torch.Tensor:
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def init_distributed(local_rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=int(os.environ["RANK"]),
            device_id=device,
        )


def init_vllm_distributed(
    local_rank: int,
    world_size: int,
    *,
    num_max_tokens: int,
) -> tuple[Any, AbstractContextManager[Any]]:
    from vllm.config import (
        CompilationConfig,
        KernelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=int(os.environ["RANK"]),
            device_id=device,
        )

    parallel_config = ParallelConfig(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        is_moe_model=True,
    )
    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False
    vllm_config = VllmConfig(
        parallel_config=parallel_config,
        kernel_config=KernelConfig(moe_backend="deep_gemm_mega_moe"),
        compilation_config=compilation_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=num_max_tokens,
        ),
    )

    config_ctx = set_current_vllm_config(vllm_config)
    config_ctx.__enter__()

    init_distributed_environment(
        world_size=world_size,
        rank=int(os.environ["RANK"]),
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
    )
    return vllm_config, config_ctx


def mega_moe_ep_layout(num_experts: int) -> tuple[int, int, int]:
    from vllm.distributed import get_ep_group

    ep_group = get_ep_group()
    num_local_experts = num_experts // ep_group.world_size
    experts_start_idx = ep_group.rank_in_group * num_local_experts
    return ep_group.rank_in_group, num_local_experts, experts_start_idx


def build_vllm_mega_moe_experts(
    vllm_config,
    *,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
):
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4MegaMoEExperts

    vllm_config.scheduler_config.max_num_batched_tokens = num_max_tokens
    return DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        top_k=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        prefix=EXPERTS_PREFIX,
    )


def load_benchmark_weights_into_vllm_experts(
    experts,
    weights: BenchmarkWeights,
    *,
    experts_start_idx: int,
    intermediate: int,
) -> None:
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp4

    experts.cuda()
    num_local_experts = weights.w13.shape[0]

    for local_expert_id in range(num_local_experts):
        global_expert_id = experts_start_idx + local_expert_id
        shard_specs = (
            ("w1", weights.w13[local_expert_id, :intermediate]),
            ("w3", weights.w13[local_expert_id, intermediate:]),
            ("w2", weights.w2[local_expert_id]),
        )
        for shard_id, bf16 in shard_specs:
            packed, scale = per_token_cast_to_fp4(
                bf16,
                use_ue8m0=True,
                gran_k=32,
            )
            weight_param = (
                experts.w13_weight if shard_id in ("w1", "w3") else experts.w2_weight
            )
            scale_param = (
                experts.w13_weight_scale
                if shard_id in ("w1", "w3")
                else experts.w2_weight_scale
            )
            weight_name = (
                "experts.w13_weight"
                if shard_id in ("w1", "w3")
                else "experts.w2_weight"
            )
            scale_name = (
                "experts.w13_weight_scale"
                if shard_id in ("w1", "w3")
                else "experts.w2_weight_scale"
            )
            experts.weight_loader(
                weight_param,
                packed.view(torch.uint8),
                weight_name,
                shard_id=shard_id,
                expert_id=global_expert_id,
            )
            experts.weight_loader(
                scale_param,
                _float_ue8m0_scale_to_uint8(scale),
                scale_name,
                shard_id=shard_id,
                expert_id=global_expert_id,
            )

    experts.finalize_weights()


def build_vllm_mega_moe(
    vllm_config,
    bench_weights: BenchmarkWeights,
    *,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
) -> nn.Module:
    experts = build_vllm_mega_moe_experts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_max_tokens=num_max_tokens,
    )
    load_benchmark_weights_into_vllm_experts(
        experts,
        bench_weights,
        experts_start_idx=experts_start_idx,
        intermediate=intermediate,
    )
    return experts


def run_vllm_forward(
    *,
    vllm_config,
    experts: nn.Module,
    inputs: BenchmarkInputs,
    activation_clamp: float | None,
    fast_math: bool,
) -> torch.Tensor:
    from vllm.forward_context import set_forward_context

    num_tokens = inputs.hidden_states.shape[0]
    with set_forward_context(None, vllm_config, num_tokens=num_tokens):
        return experts(
            inputs.hidden_states,
            inputs.topk_weights,
            inputs.topk_ids,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )


def _import_vllm_stage_mega_moe_inputs():
    try:
        from vllm.model_executor.models.deepseek_v4 import (
            _stage_deepseek_v4_mega_moe_inputs,
        )
    except ImportError:
        from vllm.models.deepseek_v4.nvidia.model import (
            _stage_deepseek_v4_mega_moe_inputs,
        )
    return _stage_deepseek_v4_mega_moe_inputs


def _prepare_vllm_bench_state(
    experts: nn.Module,
    inputs: BenchmarkInputs,
    *,
    activation_clamp: float | None,
    fast_math: bool,
) -> tuple[Any, Any]:
    """Split staging vs mega_moe compute for deep_gemm bench parity with flashinfer."""
    import vllm.third_party.deep_gemm as deep_gemm

    stage_mega_moe_inputs = _import_vllm_stage_mega_moe_inputs()
    experts.finalize_weights()
    symm_buffer = experts.get_symm_buffer()
    num_tokens = inputs.hidden_states.shape[0]
    output = torch.empty(
        num_tokens,
        inputs.hidden_states.shape[1],
        dtype=torch.bfloat16,
        device=inputs.hidden_states.device,
    )
    l1_weights = experts._transformed_l1_weights
    l2_weights = experts._transformed_l2_weights
    assert l1_weights is not None and l2_weights is not None

    def stage_inputs() -> None:
        stage_mega_moe_inputs(
            inputs.hidden_states,
            inputs.topk_weights,
            inputs.topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
        )

    def run_compute() -> torch.Tensor:
        deep_gemm.fp8_fp4_mega_moe(
            output,
            l1_weights,
            l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return output

    return stage_inputs, run_compute


def run_vllm_forward_bench(
    experts: nn.Module,
    inputs: BenchmarkInputs,
    *,
    activation_clamp: float | None,
    fast_math: bool,
) -> torch.Tensor:
    """Hot path for timing: stage + kernel compute (no torch.ops / forward context)."""
    stage_inputs, run_compute = _prepare_vllm_bench_state(
        experts,
        inputs,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    stage_inputs()
    return run_compute()


def bench_vllm_forward(
    experts: nn.Module,
    inputs: BenchmarkInputs,
    *,
    activation_clamp: float | None,
    fast_math: bool,
    timing_mode: str,
    warmup: int,
    repeat: int,
    cold_start: bool,
    cold_l2_cache: bool = False,
):
    """Benchmark vLLM deep_gemm mega forward (matched scope vs ``bench_fi_forward``)."""
    from moe_ep_common import bench_deep_gemm_mega_cudagraph_ms, bench_forward

    if timing_mode == "cudagraph":
        stage_inputs, run_compute = _prepare_vllm_bench_state(
            experts,
            inputs,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return bench_deep_gemm_mega_cudagraph_ms(
            stage_inputs,
            run_compute,
            warmup=warmup,
            repeat=repeat,
        )

    def run_once() -> torch.Tensor:
        return run_vllm_forward_bench(
            experts,
            inputs,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )

    return bench_forward(
        run_once,
        timing_mode=timing_mode,
        warmup=warmup,
        repeat=repeat,
        cold_start=cold_start if timing_mode == "cuda_event" else False,
        cold_l2_cache=cold_l2_cache,
    )


def unregister_vllm_experts(vllm_config, *, prefix: str = EXPERTS_PREFIX) -> None:
    ctx = vllm_config.compilation_config.static_forward_context
    if ctx.get(prefix) is not None:
        del ctx[prefix]


def destroy_vllm_symm_buffers() -> None:
    from vllm.model_executor.models.deepseek_v4 import DeepseekV4MegaMoEExperts

    for symm_buffer in list(DeepseekV4MegaMoEExperts._symm_buffer_cache.values()):
        symm_buffer.destroy()
    DeepseekV4MegaMoEExperts._symm_buffer_cache.clear()


def release_vllm_experts(vllm_config, experts: nn.Module) -> None:
    unregister_vllm_experts(vllm_config, prefix=getattr(experts, "prefix", EXPERTS_PREFIX))
    destroy_vllm_symm_buffers()


def cleanup_vllm_distributed(vllm_config, config_ctx) -> None:
    destroy_vllm_symm_buffers()
    config_ctx.__exit__(None, None, None)
    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()


def build_fi_mega_config(
    *,
    megakernel: str,
    intermediate: int,
    topk: int,
    activation_clamp: float | None,
    fast_math: bool,
    quantize_input: bool = True,
):
    from flashinfer.moe_ep import (
        DeepGemmMegaMoeConfig,
        MegaConfig,
        Mxfp8CutedslMegaMoeConfig,
        Nvfp4CutedslMegaMoeConfig,
    )

    if megakernel == "deep_gemm_mega":
        mk = DeepGemmMegaMoeConfig(
            intermediate_size=intermediate,
            top_k=topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    elif megakernel == "nvfp4_cutedsl":
        mk = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=intermediate,
            top_k=topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    elif megakernel == "mxfp8_cutedsl":
        mk = Mxfp8CutedslMegaMoeConfig(
            intermediate_size=intermediate,
            top_k=topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    else:
        raise ValueError(
            f"unsupported megakernel {megakernel!r}; expected one of "
            "deep_gemm_mega, nvfp4_cutedsl, mxfp8_cutedsl"
        )

    return MegaConfig(
        megakernel=mk,
        quantize_input=quantize_input,
        preprocess_weights=True,
    )


def make_fi_bootstrap(
    rank: int,
    world_size: int,
    *,
    use_vllm_ep_group: bool = False,
):
    """Bootstrap config for ``MoEEpMegaLayer``.

    When vLLM EP is already initialized, mirror ``patch/fi_utils`` and bind the
    EP ``device_group`` instead of bare WORLD rank/world_size.
    """
    from flashinfer.moe_ep import BootstrapConfig

    if use_vllm_ep_group:
        from vllm.distributed import get_ep_group

        ep = get_ep_group()
        return BootstrapConfig(
            world_size=ep.world_size,
            rank=ep.rank_in_group,
            process_group=ep.device_group,
            auto_bootstrap=False,
        )
    return BootstrapConfig(world_size=world_size, rank=rank)


def ensure_fi_moe_ep_runtime(
    rank: int,
    world_size: int,
    backend_id: str,
    *,
    use_vllm_ep_group: bool = False,
) -> None:
    """Bootstrap NVSHMEM before cutedsl symm buffer allocation."""
    if backend_id not in {"fi_nvfp4", "fi_mxfp8"}:
        return
    from flashinfer.moe_ep import bootstrap_moe_ep_runtime
    from flashinfer.moe_ep.core.runtime import (
        TORCH_DIST,
        mxfp8_cutedsl_runtime_requirements,
        nvfp4_cutedsl_runtime_requirements,
    )

    megakernel = megakernel_for_backend(backend_id)
    bootstrap = make_fi_bootstrap(
        rank, world_size, use_vllm_ep_group=use_vllm_ep_group
    )
    req_by_kernel = {
        "deep_gemm_mega": frozenset({TORCH_DIST}),
        "nvfp4_cutedsl": nvfp4_cutedsl_runtime_requirements(bootstrap),
        "mxfp8_cutedsl": mxfp8_cutedsl_runtime_requirements(bootstrap),
    }
    bootstrap_moe_ep_runtime(bootstrap, req_by_kernel[megakernel])


def build_fi_mega_layer(
    rank: int,
    world_size: int,
    *,
    backend_id: str,
    num_experts: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
    weights: BenchmarkWeights,
    activation_clamp: float | None,
    fast_math: bool,
    use_vllm_ep_group: bool = False,
    quantize_input: bool = True,
):
    from flashinfer.moe_ep import (
        FleetParams,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.runtime import TORCH_DIST, nvfp4_cutedsl_runtime_requirements
    from flashinfer.moe_ep.core.runtime import mxfp8_cutedsl_runtime_requirements

    megakernel = megakernel_for_backend(backend_id)
    bootstrap = make_fi_bootstrap(
        rank, world_size, use_vllm_ep_group=use_vllm_ep_group
    )
    if use_vllm_ep_group:
        req_by_kernel = {
            "deep_gemm_mega": frozenset({TORCH_DIST}),
            "nvfp4_cutedsl": nvfp4_cutedsl_runtime_requirements(bootstrap),
            "mxfp8_cutedsl": mxfp8_cutedsl_runtime_requirements(bootstrap),
        }
        bootstrap_moe_ep_runtime(bootstrap, req_by_kernel[megakernel])

    mega = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_max_tokens,
            token_hidden_size=hidden,
            weights=MoEWeightPack(w13=weights.w13, w2=weights.w2),
        ),
        backend=build_fi_mega_config(
            megakernel=megakernel,
            intermediate=intermediate,
            topk=topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            quantize_input=quantize_input,
        ),
    )
    assert isinstance(mega, MoEEpMegaLayer)
    return mega


def to_moe_ep_tensors(inputs: BenchmarkInputs):
    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=inputs.hidden_states,
        topk_ids=inputs.topk_ids,
        topk_weights=inputs.topk_weights,
    )


FiBenchInputs = Union[BenchmarkInputs, Any]


def _as_moe_ep_tensors(inputs: FiBenchInputs):
    if isinstance(inputs, BenchmarkInputs):
        return to_moe_ep_tensors(inputs)
    return inputs


def make_prestaged_fi_tensors(
    backend_id: str,
    inputs: BenchmarkInputs,
    *,
    rank: int,
    world_size: int,
    num_experts: int,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
    topk: int,
    activation_clamp: float | None,
    use_vllm_ep_group: bool = False,
):
    """Pre-quantize activations once (outside the timed loop) for nvfp4/mxfp8."""
    from flashinfer.moe_ep import MoEEpTensors

    ensure_fi_moe_ep_runtime(
        rank,
        world_size,
        backend_id,
        use_vllm_ep_group=use_vllm_ep_group,
    )

    num_tokens = inputs.hidden_states.shape[0]

    if backend_id == "fi_nvfp4":
        from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
            get_symm_buffer_for_mega_moe,
            make_dummy_epilogue_params,
        )
        from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
            stage_mega_moe_inputs,
        )

        staging_buffer = get_symm_buffer_for_mega_moe(
            num_experts,
            num_max_tokens,
            topk,
            hidden,
            2 * intermediate,
            rank,
            world_size,
            gate_up_clamp=activation_clamp,
        )
        stage_mega_moe_inputs(
            inputs.hidden_states,
            inputs.topk_weights,
            inputs.topk_ids,
            staging_buffer.x[:num_tokens],
            staging_buffer.x_sf[:num_tokens],
            staging_buffer.topk_idx[:num_tokens],
            staging_buffer.topk_weights[:num_tokens],
        )
        t_hidden = staging_buffer.x[:num_tokens].clone()
        t_scales = staging_buffer.x_sf[:num_tokens].clone()
        staging_buffer.destroy()

        g = torch.Generator(device="cuda").manual_seed(19 + rank)
        fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
            num_local_experts,
            generator=g,
        )
        return MoEEpTensors(
            hidden_states=t_hidden,
            topk_ids=inputs.topk_ids,
            topk_weights=inputs.topk_weights,
            scales=t_scales,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
        )

    if backend_id == "fi_mxfp8":
        from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
            get_symm_buffer_for_mxfp8_mega_moe,
        )
        from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.staging import (
            stage_mega_moe_inputs,
        )

        kind = "mxfp8_e4m3"
        staging_buffer = get_symm_buffer_for_mxfp8_mega_moe(
            num_experts,
            num_max_tokens,
            topk,
            hidden,
            intermediate,
            rank,
            world_size,
            kind=kind,
            gate_up_clamp=activation_clamp,
        )
        stage_mega_moe_inputs(
            inputs.hidden_states,
            inputs.topk_weights,
            inputs.topk_ids,
            staging_buffer.x,
            staging_buffer.x_sf,
            staging_buffer.topk_idx,
            staging_buffer.topk_weights,
            kind=kind,
        )
        t_hidden = staging_buffer.x[:num_tokens].clone()
        t_scales = staging_buffer.x_sf[:num_tokens].clone()
        staging_buffer.destroy()
        return MoEEpTensors(
            hidden_states=t_hidden,
            topk_ids=inputs.topk_ids,
            topk_weights=inputs.topk_weights,
            scales=t_scales,
        )

    raise ValueError(
        f"prestaged bench scope only supports {sorted({'fi_nvfp4', 'fi_mxfp8'})}; "
        f"got {backend_id!r}"
    )


def run_fi_forward(layer, inputs: BenchmarkInputs) -> torch.Tensor:
    return layer.forward(to_moe_ep_tensors(inputs))


def run_fi_forward_bench(layer, inputs: FiBenchInputs) -> torch.Tensor:
    """Hot path for timing: stage + kernel compute (skip per-iter validation)."""
    t = _as_moe_ep_tensors(inputs)
    quantize_input = layer._mega_config.quantize_input
    if layer._transformed is None:
        layer._preprocess_weights()
    workspace = layer._ensure_workspace()
    layer._kernel.stage_inputs(
        t,
        workspace,
        quantize_input=quantize_input,
    )
    y = torch.empty(
        t.num_tokens,
        layer._fleet_params.token_hidden_size,
        dtype=torch.bfloat16,
        device=t.hidden_states.device,
    )
    return layer._kernel.compute(
        workspace,
        layer._transformed,
        output=y,
    )


def _prepare_fi_bench_state(layer, inputs: FiBenchInputs):
    """Workspace + static output buffer for graph-safe fi timing."""
    t = _as_moe_ep_tensors(inputs)
    quantize_input = layer._mega_config.quantize_input
    if layer._transformed is None:
        layer._preprocess_weights()
    workspace = layer._ensure_workspace()
    output = torch.empty(
        t.num_tokens,
        layer._fleet_params.token_hidden_size,
        dtype=torch.bfloat16,
        device=t.hidden_states.device,
    )

    def stage_inputs() -> None:
        layer._kernel.stage_inputs(
            t,
            workspace,
            quantize_input=quantize_input,
        )

    def run_compute() -> torch.Tensor:
        return layer._kernel.compute(
            workspace,
            layer._transformed,
            output=output,
        )

    return stage_inputs, run_compute


def bench_fi_forward(
    layer,
    inputs: FiBenchInputs,
    *,
    backend_id: str,
    timing_mode: str,
    warmup: int,
    repeat: int,
    cold_start: bool,
    cold_l2_cache: bool = False,
):
    """Benchmark fi mega forward with CUDA graph only when the backend supports it."""
    from moe_ep_common import (
        bench_forward,
        resolve_fi_timing_mode,
    )

    effective_mode = resolve_fi_timing_mode(backend_id, timing_mode)
    if effective_mode == "cudagraph":
        return bench_fi_forward_cudagraph_ms(
            layer,
            inputs,
            warmup=warmup,
            repeat=repeat,
        )

    def run_once() -> torch.Tensor:
        return run_fi_forward_bench(layer, inputs)

    return bench_forward(
        run_once,
        timing_mode=effective_mode,
        warmup=warmup,
        repeat=repeat,
        cold_start=cold_start if effective_mode == "cuda_event" else False,
        cold_l2_cache=cold_l2_cache,
    )


def bench_fi_forward_cudagraph_ms(
    layer,
    inputs: FiBenchInputs,
    *,
    warmup: int,
    repeat: int,
):
    """Time stage (eager) + deep_gemm mega-kernel compute (CUDA graph replay).

    Only ``fi_deep_gemm`` is supported; cutedsl backends sync inside compute.
    """
    from moe_ep_common import bench_deep_gemm_mega_cudagraph_ms

    stage_inputs, run_compute = _prepare_fi_bench_state(layer, inputs)
    return bench_deep_gemm_mega_cudagraph_ms(
        stage_inputs,
        run_compute,
        warmup=warmup,
        repeat=repeat,
    )


def destroy_fi_layer(layer) -> None:
    layer.destroy()


def backends_for_cli(backend: str) -> list[str]:
    from moe_ep_common import benchmark_backend_order

    if backend == "all":
        return benchmark_backend_order(list(BACKEND_IDS))
    if backend not in BACKEND_IDS:
        raise ValueError(f"unknown backend {backend!r}")
    return [backend]
