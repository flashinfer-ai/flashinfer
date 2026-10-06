"""Regression tests for tp_moe launch arguments."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.experimental.b12x.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec


def _skip_if_no_sm120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")


def _skip_if_unavailable() -> None:
    _skip_if_no_sm120()
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    if not (MODEL_PATH / "model.safetensors.index.json").exists():
        pytest.skip(f"Indexed model weights not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _run_parameter_launch_case(case: str) -> subprocess.CompletedProcess[str]:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    spec = _make_spec()
    script = textwrap.dedent(
        f"""
        import pathlib
        import sys

        import torch
        from torch.nn import Parameter

        sys.path.insert(0, {str(repo_root)!r})

        from benchmarks.experimental.b12x.benchmark_moe import ModelSpec, load_expert_weights, make_routed_inputs
        from b12x.moe.fused_moe._impl import (
            clear_tp_moe_caches,
        )
        from b12x.testing.reference.helpers import prepare_tp_moe_fp4_experts, run_tp_moe_fp4

        case = {case!r}
        clear_tp_moe_caches()

        device = torch.device("cuda")
        spec = ModelSpec(
            hidden_size={spec.hidden_size},
            intermediate_size={spec.intermediate_size},
            num_experts={spec.num_experts},
            top_k={spec.top_k},
            tp_size={spec.tp_size},
            tp_rank={spec.tp_rank},
        )
        weights = load_expert_weights(pathlib.Path({str(MODEL_PATH)!r}), spec)
        x, topk_ids, topk_weights = make_routed_inputs(spec, 8, seed=123, device=device)

        a1_gscale = weights.w13_input_scale_quant_per_expert.clone()
        a2_gscale = weights.w2_input_scale_quant_per_expert.clone()
        w1_alphas = weights.g1_alphas_per_expert.clone()
        w2_alphas = weights.g2_alphas_per_expert.clone()

        if case in ("scales", "all"):
            a1_gscale = Parameter(a1_gscale, requires_grad=False)
            a2_gscale = Parameter(a2_gscale, requires_grad=False)
        if case in ("alphas", "all"):
            w1_alphas = Parameter(w1_alphas, requires_grad=False)
            w2_alphas = Parameter(w2_alphas, requires_grad=False)

        experts = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=a1_gscale,
            w1_fp4=weights.w13_weight,
            w1_blockscale=weights.w13_blockscale_swizzled,
            w1_alphas=w1_alphas,
            a2_gscale=a2_gscale,
            w2_fp4=weights.w2_weight,
            w2_blockscale=weights.w2_blockscale_swizzled,
            w2_alphas=w2_alphas,
        )

        out = torch.empty_like(x)
        print(f"case={{case}} start", flush=True)
        run_tp_moe_fp4(
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=out,
            input_scales_static=True,
        )
        torch.cuda.synchronize()
        print(f"case={{case}} ok", flush=True)
        """
    )
    env = os.environ.copy()
    env.setdefault("CUTE_DSL_ARCH", "sm_120a")
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )


def _direct_micro_launchable(
    quant_mode: str,
    m: int,
    n: int,
    *,
    weight_E: int = 256,
    k: int = 4096,
    num_topk: int = 10,
    compile_time_phase: int = 0,
) -> bool:
    from b12x.moe.fused_moe._impl import (
        _DIRECT_MICRO_BLOCK_DIM,
        _compiled_direct_micro_accepts_block_dim,
        _get_micro_kernel,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    torch.empty(1, device="cuda")
    compiled, _ = _get_micro_kernel(
        weight_E,
        m,
        k,
        n,
        num_topk,
        topk_ids_dtype=torch.int32,
        fast_math=True,
        activation="silu",
        quant_mode=quant_mode,
        device=torch.device("cuda"),
        compile_time_phase=compile_time_phase,
    )
    return _compiled_direct_micro_accepts_block_dim(compiled, _DIRECT_MICRO_BLOCK_DIM)


def test_nvfp4_direct_micro_launches_qwen_bs8_shape() -> None:
    _skip_if_no_sm120()

    assert _direct_micro_launchable("nvfp4", 8, 256, weight_E=512)


@pytest.mark.parametrize("compile_time_phase", [1, 2])
def test_nvfp4_split_micro_compile_signature_matches_runtime_launch(
    compile_time_phase: int,
) -> None:
    _skip_if_no_sm120()

    assert _direct_micro_launchable(
        "nvfp4",
        1,
        256,
        weight_E=512,
        compile_time_phase=compile_time_phase,
    )


@pytest.mark.parametrize("case", ["alphas", "scales"])
def test_b12x_moe_accepts_parameter_backed_launch_args(case: str) -> None:
    """The static path should not segfault on Parameter-backed scale tensors."""
    _skip_if_unavailable()

    result = _run_parameter_launch_case(case)
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"child process failed for case={case} with rc={result.returncode}\n{output}"
    )
