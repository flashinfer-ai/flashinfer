#!/usr/bin/env python3
"""MoE-EP MegaMoE benchmark with pre-quantized weights and saved inputs.

Two subcommands:

  save-fixtures  — quantize bf16 expert weights and activations once per backend/rank,
                   write ``meta.pt``, ``inputs_rank{R}.pt``,
                   ``activations_{backend}_rank{R}.pt``, and
                   ``weights_{backend}_rank{R}.pt`` under ``--fixture-dir``.

  bench          — load kernel-ready weights and prestaged activations from disk;
                   no bf16→quant at benchmark time (``preprocess_weights=False``,
                   ``quantize_input=False``; timed path is memcpy stage + kernel).
                   By default compares each backend's bf16 output against
                   ``vllm_deepgemm`` before timing (``--skip-verify`` to disable).

Launch (example, 4 GPUs on one node):
    cd /lustre/fsw/coreai_libraries_cudnn/mhoqueanik/flashinfer

    # 1) Write fixtures (bf16 quant happens here only):
    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe_pretrained.py \\
        save-fixtures --fixture-dir /path/to/moe_ep_fixtures

    # 2) Benchmark all backends from fixtures:
    torchrun --nproc_per_node=4 benchmarks/moe_ep/bench_moe_ep_mega_moe_pretrained.py \\
        bench --fixture-dir /path/to/moe_ep_fixtures --backend all

Requires SM100+ GPU per rank and the same deps as ``bench_moe_ep_mega_moe.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from backends import (
    backends_for_cli,
    bench_fi_forward,
    cleanup_vllm_distributed,
    destroy_fi_layer,
    init_distributed,
    init_vllm_distributed,
    mega_moe_ep_layout,
    release_vllm_experts,
    run_fi_forward_bench,
)
from moe_ep_common import (
    BACKEND_LABELS,
    check_problem,
    cold_start_note_for_backend,
    fi_timing_fallback_note,
    print_benchmark_header,
    print_benchmark_timing,
    require_env_rank,
    require_shared_routing_benchmark,
    require_sm100,
    resolve_fi_timing_mode,
    resolve_timing_mode,
    routing_mode_from_args,
    warn_kernel_token_sizing,
)
from moe_ep_pretrained_fixtures import (
    FixtureMeta,
    load_meta,
    load_prestaged_activations,
    load_transformed_weights,
    make_fixture_inputs,
    make_fixture_meta_from_args,
    make_fixture_weights,
    require_fixtures_exist,
    save_all_fixtures_for_rank,
)


PRETRAINED_TIMING_NOTE = (
    "kernel-ready weights + prestaged activations loaded from .pt fixtures; "
    "no bf16 weight or activation quant at bench time (memcpy stage + kernel)"
)

REFERENCE_BACKEND = "vllm_deepgemm"
QUANT_BACKEND_IDS = frozenset({"fi_nvfp4", "fi_mxfp8"})
DEFAULT_VERIFY_ATOL = 0.0
DEFAULT_VERIFY_RTOL = 0.0


def _add_shared_problem_args(parser: argparse.ArgumentParser) -> None:
    from moe_ep_common import (
        DEFAULT_HIDDEN,
        DEFAULT_INTERMEDIATE,
        DEFAULT_NUM_LOCAL_EXPERTS,
        DEFAULT_NUM_MAX_TOKENS,
        DEFAULT_NUM_TOKENS,
        DEFAULT_NUM_TOPK,
    )

    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--num-max-tokens", type=int, default=DEFAULT_NUM_MAX_TOKENS)
    parser.add_argument(
        "--num-local-experts",
        type=int,
        default=DEFAULT_NUM_LOCAL_EXPERTS,
        help="Experts owned by each EP rank.",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_NUM_TOPK)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--no-activation-clamp", action="store_true")
    parser.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fixture-dir",
        type=str,
        required=True,
        help="Directory for meta.pt, inputs_rank{R}.pt, weights_*_rank{R}.pt",
    )


def _parse_save_fixtures_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save pre-quantized MegaMoE weights and routing inputs to .pt files.",
    )
    _add_shared_problem_args(parser)
    return parser.parse_args(argv)


def _parse_bench_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MegaMoE from pre-quantized .pt fixtures.",
    )
    _add_shared_problem_args(parser)

    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument(
        "--random-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--cold-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fi_deep_gemm",
        choices=(
            "vllm_deepgemm",
            "fi_deep_gemm",
            "fi_nvfp4",
            "fi_mxfp8",
            "all",
        ),
    )
    parser.add_argument(
        "--timing",
        type=str,
        default="cudagraph",
        choices=("cudagraph", "cuda_event", "cupti"),
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Alias for --timing=cuda_event.",
    )
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip vLLM gold-standard output check before timing.",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=None,
        help="Output compare atol vs vllm_deepgemm (default: 0 for deep_gemm).",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=None,
        help="Output compare rtol vs vllm_deepgemm (default: 0 for deep_gemm).",
    )
    return parser.parse_args(argv)


def build_vllm_mega_moe_pretrained(
    vllm_config,
    *,
    fixture_dir: str,
    rank: int,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
):
    from backends import build_vllm_mega_moe_experts

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
    experts.cuda()
    transformed = load_transformed_weights(fixture_dir, "vllm_deepgemm", rank)
    experts._transformed_l1_weights = transformed[0]
    experts._transformed_l2_weights = transformed[1]
    return experts


def _build_fi_mega_layer_pretrained(
    rank: int,
    world_size: int,
    *,
    backend_id: str,
    fixture_dir: str,
    num_experts: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_max_tokens: int,
    activation_clamp: float | None,
    fast_math: bool,
    use_vllm_ep_group: bool = False,
):
    from backends import build_fi_mega_config, ensure_fi_moe_ep_runtime, make_fi_bootstrap, megakernel_for_backend
    from flashinfer.moe_ep import FleetParams, MoEEpLayer, MoEEpMegaLayer, dummy_moe_weights
    from flashinfer.moe_ep import MegaConfig

    megakernel = megakernel_for_backend(backend_id)
    ensure_fi_moe_ep_runtime(
        rank,
        world_size,
        backend_id,
        use_vllm_ep_group=use_vllm_ep_group,
    )
    transformed = load_transformed_weights(fixture_dir, backend_id, rank)
    bootstrap = make_fi_bootstrap(rank, world_size, use_vllm_ep_group=use_vllm_ep_group)
    num_local_experts = num_experts // world_size

    mk = build_fi_mega_config(
        megakernel=megakernel,
        intermediate=intermediate,
        topk=topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        quantize_input=False,
    )
    mega_config = MegaConfig(
        megakernel=mk.megakernel,
        quantize_input=False,
        preprocess_weights=False,
        transformed_weights=transformed,
    )
    mega = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_max_tokens,
            token_hidden_size=hidden,
            weights=dummy_moe_weights(
                num_local_experts=num_local_experts,
                hidden=hidden,
                intermediate=intermediate,
                device="cpu",
            ),
        ),
        backend=mega_config,
    )
    assert isinstance(mega, MoEEpMegaLayer)
    return mega


def _prepare_vllm_pretrained_bench_state(
    experts,
    prestaged_inputs,
    *,
    hidden: int,
    activation_clamp: float | None,
    fast_math: bool,
):
    import vllm.third_party.deep_gemm as deep_gemm

    symm_buffer = experts.get_symm_buffer()
    num_tokens = prestaged_inputs.num_tokens
    output = torch.empty(
        num_tokens,
        hidden,
        dtype=torch.bfloat16,
        device=prestaged_inputs.hidden_states.device,
    )
    l1_weights = experts._transformed_l1_weights
    l2_weights = experts._transformed_l2_weights
    assert l1_weights is not None and l2_weights is not None
    assert prestaged_inputs.scales is not None

    def stage_inputs() -> None:
        x_slot = symm_buffer.x[:num_tokens]
        if x_slot.dtype != torch.float8_e4m3fn:
            x_slot = x_slot.view(torch.float8_e4m3fn)
        hidden_states = prestaged_inputs.hidden_states
        if hidden_states.dtype != torch.float8_e4m3fn:
            x_slot.copy_(hidden_states.view(torch.float8_e4m3fn))
        else:
            x_slot.copy_(hidden_states)
        symm_buffer.x_sf[:num_tokens].copy_(prestaged_inputs.scales)
        symm_buffer.topk_idx[:num_tokens].copy_(prestaged_inputs.topk_ids)
        symm_buffer.topk_weights[:num_tokens].copy_(prestaged_inputs.topk_weights)

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


def bench_vllm_forward_pretrained(
    experts,
    prestaged_inputs,
    *,
    hidden: int,
    activation_clamp: float | None,
    fast_math: bool,
    timing_mode: str,
    warmup: int,
    repeat: int,
    cold_start: bool,
    cold_l2_cache: bool = False,
):
    from moe_ep_common import bench_deep_gemm_mega_cudagraph_ms, bench_forward

    if timing_mode == "cudagraph":
        stage_inputs, run_compute = _prepare_vllm_pretrained_bench_state(
            experts,
            prestaged_inputs,
            hidden=hidden,
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
        stage_inputs, run_compute = _prepare_vllm_pretrained_bench_state(
            experts,
            prestaged_inputs,
            hidden=hidden,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        stage_inputs()
        return run_compute()

    return bench_forward(
        run_once,
        timing_mode=timing_mode,
        warmup=warmup,
        repeat=repeat,
        cold_start=cold_start if timing_mode == "cuda_event" else False,
        cold_l2_cache=cold_l2_cache,
    )


def _validate_meta_matches_args(meta: FixtureMeta, args: argparse.Namespace) -> None:
    mismatches: list[str] = []
    for field_name in (
        "hidden",
        "intermediate",
        "num_tokens",
        "num_max_tokens",
        "num_local_experts",
        "topk",
    ):
        meta_val = getattr(meta, field_name)
        arg_val = getattr(args, field_name)
        if meta_val != arg_val:
            mismatches.append(f"{field_name}: fixture={meta_val} cli={arg_val}")
    if mismatches:
        raise SystemExit(
            "ERROR: CLI problem size disagrees with fixture meta.pt:\n  "
            + "\n  ".join(mismatches)
            + "\nOmit overrides or regenerate fixtures."
        )


def cmd_save_fixtures(args: argparse.Namespace) -> int:
    local_rank, world_size = require_env_rank()
    require_sm100(local_rank)

    num_experts = args.num_local_experts * world_size
    check_problem(
        args.hidden,
        args.intermediate,
        args.num_tokens,
        args.num_max_tokens,
        num_experts,
        world_size,
    )

    meta = make_fixture_meta_from_args(args, world_size=world_size)
    vllm_config, config_ctx = init_vllm_distributed(
        local_rank,
        world_size,
        num_max_tokens=args.num_max_tokens,
    )
    rank = dist.get_rank()
    try:
        _, num_local_experts, experts_start_idx = mega_moe_ep_layout(num_experts)
        bench_weights = make_fixture_weights(rank, meta)
        inputs = make_fixture_inputs(rank, meta, num_experts=num_experts)
        save_all_fixtures_for_rank(
            fixture_dir=args.fixture_dir,
            rank=rank,
            world_size=world_size,
            meta=meta,
            bench_weights=bench_weights,
            inputs=inputs,
            vllm_config=vllm_config,
            num_experts=num_experts,
            experts_start_idx=experts_start_idx,
        )
        if rank == 0:
            print(f"Saved fixtures under {args.fixture_dir}")
            print(f"  meta.pt")
            print(f"  inputs_rank*.pt (x{world_size})")
            print(f"  activations_<backend>_rank*.pt (x{world_size} each backend)")
            print(f"  weights_<backend>_rank*.pt (x{world_size} each backend)")
    finally:
        cleanup_vllm_distributed(vllm_config, config_ctx)
    return 0


def _needs_vllm(backend_ids: list[str], *, verify: bool = False) -> bool:
    return REFERENCE_BACKEND in backend_ids or verify


def _compare_pretrained_outputs(
    y_ref: torch.Tensor,
    y_cand: torch.Tensor,
    *,
    rank: int,
    reference: str,
    candidate: str,
    atol: float,
    rtol: float,
) -> None:
    if y_ref.shape != y_cand.shape:
        raise RuntimeError(
            f"rank {rank}: shape mismatch {reference}={tuple(y_ref.shape)} "
            f"{candidate}={tuple(y_cand.shape)}"
        )
    if y_ref.dtype != y_cand.dtype:
        raise RuntimeError(
            f"rank {rank}: dtype mismatch {reference}={y_ref.dtype} "
            f"{candidate}={y_cand.dtype}"
        )

    torch.testing.assert_close(
        y_ref,
        y_cand,
        atol=atol,
        rtol=rtol,
        msg=lambda _: (
            f"rank {rank}: {reference} vs {candidate} mismatch "
            f"(max_abs={(y_ref - y_cand).abs().max().item():.6g})"
        ),
    )


def _smoke_check_pretrained_output(
    y: torch.Tensor,
    *,
    rank: int,
    candidate: str,
    num_tokens: int,
    hidden: int,
) -> None:
    expected_shape = (num_tokens, hidden)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"rank {rank}: {candidate} smoke check shape mismatch "
            f"expected={expected_shape} got={tuple(y.shape)}"
        )
    if y.dtype != torch.bfloat16:
        raise RuntimeError(
            f"rank {rank}: {candidate} smoke check expected bf16, got {y.dtype}"
        )
    if not torch.isfinite(y).all():
        bad = (~torch.isfinite(y)).sum().item()
        raise RuntimeError(
            f"rank {rank}: {candidate} smoke check found {bad} non-finite values"
        )


def _verify_tolerance(
    backend_id: str,
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    if args.verify_atol is not None and args.verify_rtol is not None:
        return args.verify_atol, args.verify_rtol, "numeric"
    if backend_id in QUANT_BACKEND_IDS:
        return 0.0, 0.0, "smoke"
    return DEFAULT_VERIFY_ATOL, DEFAULT_VERIFY_RTOL, "exact"


def _run_vllm_pretrained_forward(
    *,
    vllm_config,
    fixture_dir: str,
    rank: int,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    meta: FixtureMeta,
    activation_clamp: float | None,
) -> tuple[torch.Tensor, object]:
    experts = build_vllm_mega_moe_pretrained(
        vllm_config,
        fixture_dir=fixture_dir,
        rank=rank,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        topk=meta.topk,
        hidden=meta.hidden,
        intermediate=meta.intermediate,
        num_max_tokens=meta.num_max_tokens,
    )
    prestaged_inputs = load_prestaged_activations(fixture_dir, REFERENCE_BACKEND, rank)
    stage_inputs, run_compute = _prepare_vllm_pretrained_bench_state(
        experts,
        prestaged_inputs,
        hidden=meta.hidden,
        activation_clamp=activation_clamp,
        fast_math=meta.fast_math,
    )
    stage_inputs()
    output = run_compute()
    return output, experts


def _run_fi_pretrained_forward(
    *,
    backend_id: str,
    rank: int,
    world_size: int,
    fixture_dir: str,
    num_experts: int,
    meta: FixtureMeta,
    activation_clamp: float | None,
    use_vllm_ep_group: bool,
) -> tuple[torch.Tensor, object]:
    layer = _build_fi_mega_layer_pretrained(
        rank,
        world_size,
        backend_id=backend_id,
        fixture_dir=fixture_dir,
        num_experts=num_experts,
        topk=meta.topk,
        hidden=meta.hidden,
        intermediate=meta.intermediate,
        num_max_tokens=meta.num_max_tokens,
        activation_clamp=activation_clamp,
        fast_math=meta.fast_math,
        use_vllm_ep_group=use_vllm_ep_group,
    )
    prestaged_inputs = load_prestaged_activations(fixture_dir, backend_id, rank)
    output = run_fi_forward_bench(layer, prestaged_inputs)
    return output, layer


def _release_pretrained_handle(
    backend_id: str,
    handle: object | None,
    *,
    vllm_config,
) -> None:
    if handle is None:
        return
    if backend_id == REFERENCE_BACKEND:
        release_vllm_experts(vllm_config, handle)
    else:
        destroy_fi_layer(handle)


def _verify_pretrained_backends(
    backend_ids: list[str],
    *,
    vllm_config,
    rank: int,
    world_size: int,
    meta: FixtureMeta,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    args: argparse.Namespace,
    activation_clamp: float | None,
    fixture_dir: str,
) -> None:
    candidates = [b for b in backend_ids if b != REFERENCE_BACKEND]
    if not candidates:
        return

    y_ref, ref_handle = _run_vllm_pretrained_forward(
        vllm_config=vllm_config,
        fixture_dir=fixture_dir,
        rank=rank,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        meta=meta,
        activation_clamp=activation_clamp,
    )
    torch.cuda.synchronize()
    dist.barrier()
    y_ref = y_ref.detach().clone()
    _release_pretrained_handle(REFERENCE_BACKEND, ref_handle, vllm_config=vllm_config)
    dist.barrier()

    verify_lines: list[str] = []
    for backend_id in candidates:
        y_cand, cand_handle = _run_fi_pretrained_forward(
            backend_id=backend_id,
            rank=rank,
            world_size=world_size,
            fixture_dir=fixture_dir,
            num_experts=num_experts,
            meta=meta,
            activation_clamp=activation_clamp,
            use_vllm_ep_group=vllm_config is not None,
        )
        torch.cuda.synchronize()
        dist.barrier()

        atol, rtol, mode = _verify_tolerance(backend_id, args)
        if mode == "smoke":
            _smoke_check_pretrained_output(
                y_cand,
                rank=rank,
                candidate=backend_id,
                num_tokens=meta.num_tokens,
                hidden=meta.hidden,
            )
            detail = "smoke (shape/dtype/finite; quant kernel differs from vLLM)"
        else:
            _compare_pretrained_outputs(
                y_ref,
                y_cand,
                rank=rank,
                reference=REFERENCE_BACKEND,
                candidate=backend_id,
                atol=atol,
                rtol=rtol,
            )
            detail = f"exact atol={atol} rtol={rtol}"

        verify_lines.append(
            f"  {REFERENCE_BACKEND} vs {backend_id}: {detail} "
            f"({BACKEND_LABELS[backend_id]})"
        )
        _release_pretrained_handle(backend_id, cand_handle, vllm_config=vllm_config)
        dist.barrier()

    if rank == 0:
        print("OK: pretrained correctness vs vllm_deepgemm")
        print(f"  fixture_dir={fixture_dir}")
        print(
            f"  world_size={world_size} num_tokens={meta.num_tokens} "
            f"hidden={meta.hidden}"
        )
        for line in verify_lines:
            print(line)


def _bench_one_backend_pretrained(
    backend_id: str,
    *,
    vllm_config,
    rank: int,
    world_size: int,
    meta: FixtureMeta,
    num_experts: int,
    num_local_experts: int,
    experts_start_idx: int,
    args: argparse.Namespace,
    activation_clamp: float | None,
    timing_mode: str,
    fixture_dir: str,
):
    handle = None
    use_vllm_ep_group = vllm_config is not None
    prestaged_inputs = load_prestaged_activations(fixture_dir, backend_id, rank)
    requested_timing = timing_mode
    effective_timing = (
        resolve_fi_timing_mode(backend_id, timing_mode)
        if backend_id != "vllm_deepgemm"
        else timing_mode
    )
    fallback_note = fi_timing_fallback_note(
        backend_id, requested_timing, effective_timing
    )
    try:
        if backend_id == "vllm_deepgemm":
            handle = build_vllm_mega_moe_pretrained(
                vllm_config,
                fixture_dir=fixture_dir,
                rank=rank,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=experts_start_idx,
                topk=meta.topk,
                hidden=meta.hidden,
                intermediate=meta.intermediate,
                num_max_tokens=meta.num_max_tokens,
            )
            timing = bench_vllm_forward_pretrained(
                handle,
                prestaged_inputs,
                hidden=meta.hidden,
                activation_clamp=activation_clamp,
                fast_math=meta.fast_math,
                timing_mode=timing_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                cold_start=args.cold_start,
                cold_l2_cache=args.cold_l2,
            )
        else:
            handle = _build_fi_mega_layer_pretrained(
                rank,
                world_size,
                backend_id=backend_id,
                fixture_dir=fixture_dir,
                num_experts=num_experts,
                topk=meta.topk,
                hidden=meta.hidden,
                intermediate=meta.intermediate,
                num_max_tokens=meta.num_max_tokens,
                activation_clamp=activation_clamp,
                fast_math=meta.fast_math,
                use_vllm_ep_group=use_vllm_ep_group,
            )
            timing = bench_fi_forward(
                handle,
                prestaged_inputs,
                backend_id=backend_id,
                timing_mode=timing_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                cold_start=args.cold_start,
                cold_l2_cache=args.cold_l2,
            )

        extra_lines = [
            f"  fixture_dir={fixture_dir}",
            f"  weights=pre-quantized .pt (preprocess_weights=False at init)",
            f"  activations=activations_{backend_id}_rank{rank}.pt "
            f"(quantize_input=False; memcpy stage + kernel)",
            f"  megakernel={BACKEND_LABELS[backend_id]}",
        ]
        if fallback_note is not None:
            extra_lines.append(f"  note={fallback_note}")

        print_benchmark_header(
            title="MoE-EP MegaMoE pretrained-weight benchmark",
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            args=args,
            num_experts=num_experts,
            activation_clamp=activation_clamp,
            routing_mode=routing_mode_from_args(args),
            backend_id=backend_id,
            timing_mode=effective_timing,
            extra_lines=tuple(extra_lines),
            timing_mode_note=PRETRAINED_TIMING_NOTE,
        )
        print_benchmark_timing(
            timing,
            rank=rank,
            cold_start_note=cold_start_note_for_backend(backend_id),
        )
        return timing.steady_avg_ms
    finally:
        if handle is not None:
            if backend_id == "vllm_deepgemm":
                release_vllm_experts(vllm_config, handle)
            else:
                destroy_fi_layer(handle)


def cmd_bench(args: argparse.Namespace) -> int:
    require_shared_routing_benchmark(args)
    local_rank, world_size = require_env_rank()
    require_sm100(local_rank)

    backend_ids = backends_for_cli(args.backend)
    timing_mode = resolve_timing_mode(args)
    verify = not args.skip_verify
    meta = load_meta(args.fixture_dir)
    _validate_meta_matches_args(meta, args)

    num_experts = meta.num_local_experts * world_size
    if meta.world_size != world_size:
        raise SystemExit(
            f"ERROR: fixture world_size={meta.world_size} != runtime WORLD_SIZE={world_size}"
        )

    check_problem(
        meta.hidden,
        meta.intermediate,
        meta.num_tokens,
        meta.num_max_tokens,
        num_experts,
        world_size,
    )
    warn_kernel_token_sizing(
        rank=local_rank,
        num_tokens=meta.num_tokens,
        num_max_tokens=meta.num_max_tokens,
    )

    vllm_config = None
    config_ctx = None
    if _needs_vllm(backend_ids, verify=verify):
        vllm_config, config_ctx = init_vllm_distributed(
            local_rank,
            world_size,
            num_max_tokens=meta.num_max_tokens,
        )
    else:
        init_distributed(local_rank, world_size)

    rank = dist.get_rank()
    fixture_backends = list(backend_ids)
    if verify and REFERENCE_BACKEND not in fixture_backends:
        fixture_backends.append(REFERENCE_BACKEND)
    require_fixtures_exist(args.fixture_dir, rank, fixture_backends)
    activation_clamp = meta.activation_clamp

    exit_code = 0
    summary: list[tuple[str, float]] = []
    try:
        if _needs_vllm(backend_ids, verify=verify):
            _, num_local_experts, experts_start_idx = mega_moe_ep_layout(num_experts)
        else:
            num_local_experts = meta.num_local_experts
            experts_start_idx = rank * num_local_experts

        if verify:
            if vllm_config is None:
                raise RuntimeError(
                    "correctness check requires vLLM distributed init "
                    f"(reference={REFERENCE_BACKEND})"
                )
            _verify_pretrained_backends(
                backend_ids,
                vllm_config=vllm_config,
                rank=rank,
                world_size=world_size,
                meta=meta,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=experts_start_idx,
                args=args,
                activation_clamp=activation_clamp,
                fixture_dir=args.fixture_dir,
            )

        for backend_id in backend_ids:
            dist.barrier()
            steady_avg_ms = _bench_one_backend_pretrained(
                backend_id,
                vllm_config=vllm_config,
                rank=rank,
                world_size=world_size,
                meta=meta,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=experts_start_idx,
                args=args,
                activation_clamp=activation_clamp,
                timing_mode=timing_mode,
                fixture_dir=args.fixture_dir,
            )
            summary.append((backend_id, steady_avg_ms))
            dist.barrier()

        if rank == 0 and len(summary) > 1:
            print("Summary (steady_avg_ms, vllm first / quant last):")
            for backend_id, steady_avg_ms in summary:
                print(f"  {backend_id}: {steady_avg_ms:.3f} ms")
    except Exception:
        exit_code = 1
        raise
    finally:
        if config_ctx is not None and vllm_config is not None:
            cleanup_vllm_distributed(vllm_config, config_ctx)
        elif dist.is_initialized():
            dist.destroy_process_group()

    return exit_code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    subcommand = argv[0]
    rest = argv[1:]
    if subcommand == "save-fixtures":
        return cmd_save_fixtures(_parse_save_fixtures_args(rest))
    if subcommand == "bench":
        return cmd_bench(_parse_bench_args(rest))

    raise SystemExit(
        f"unknown subcommand {subcommand!r}; use 'save-fixtures' or 'bench'"
    )


if __name__ == "__main__":
    raise SystemExit(main())
