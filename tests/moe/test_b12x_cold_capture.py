"""Cold capture must fail before preparation; eager warm-up makes replay safe.

Run first-use checks in fresh processes and cache directories so another test
cannot hide a missing capture guard.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_cute_dsl_available()),
    reason="CUDA + CuTe-DSL required",
)

CASES = {
    "static": (320, 64),
    "aligned_static": (512, 64),
    "gated_dynamic": (320, 1024),
    "generic_dynamic": (640, 1024),
    "single_slice": (64, 64),
}


def _is_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _preparation_state(wrapper):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    folded = wrapper._folded_w1_alpha
    return (
        None if folded is None else folded.data_ptr(),
        wrapper._folded_w1_alpha_key,
        tuple(
            len(cache)
            for cache in (
                md._PADDED_SCALE_CACHE,
                md._PADDED_FP4_CACHE,
                md._WEIGHT_CACHE,
                md._STATIC_KERNEL_CACHE,
                md._DYNAMIC_KERNEL_CACHE,
                md._MICRO_KERNEL_CACHE,
                md._DIRECT_MICRO_KERNEL_CACHE,
                md._WORKSPACE_CACHE,
            )
        ),
    )


def _check_capture(
    case, prewarm, input_global_scale=False, rescale_after_prewarm=False
):
    from flashinfer import B12xMoEWrapper

    from .test_b12x_static_extent_rules import _kwargs, _reference_shape, _tensors_shape

    intermediate, tokens = CASES[case]
    t = _tensors_shape(tokens, 8, 256, intermediate, 2, seed=3)
    wrapper = B12xMoEWrapper(
        num_experts=8,
        top_k=2,
        hidden_size=256,
        intermediate_size=intermediate,
        use_cuda_graph=True,
        max_num_tokens=1024,
    )
    kwargs = _kwargs(t)
    if input_global_scale:
        # Keep the same effective FC1 alpha while exercising its folded cache.
        kwargs["input_global_scale"] = kwargs["w1_alpha"].clone()
        kwargs["w1_alpha"] = torch.ones_like(kwargs["w1_alpha"])
    if prewarm:
        eager = wrapper.run(**kwargs).clone()
        torch.cuda.synchronize()
        if input_global_scale:
            assert wrapper._folded_w1_alpha is not None
    else:
        assert wrapper._folded_w1_alpha is None
    if rescale_after_prewarm:
        kwargs["input_global_scale"] = kwargs["input_global_scale"].clone()

    before = _preparation_state(wrapper)
    graph = torch.cuda.CUDAGraph()
    if not prewarm or rescale_after_prewarm:
        match = "folded FC1 alpha" if rescale_after_prewarm else "warm-up"
        with pytest.raises(RuntimeError, match=match), torch.cuda.graph(graph):
            wrapper.run(**kwargs)
        del graph
        gc.collect()
        torch.cuda.synchronize()
    else:
        with torch.cuda.graph(graph):
            captured = wrapper.run(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, atol=2e-2, rtol=2e-2)
        ref = _reference_shape(t, tokens, 8, 256, intermediate, 2)
        assert ((captured.float() - ref).norm() / ref.norm()).item() < 0.30
    assert _preparation_state(wrapper) == before


def _run_in_fresh_process(case, prewarm, **kwargs):
    with tempfile.TemporaryDirectory(prefix="b12x_cold_capture_") as root:
        env = dict(os.environ)
        for var, sub in (
            ("CUTE_DSL_CACHE_DIR", "cute"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("TORCH_EXTENSIONS_DIR", "torch"),
            ("FLASHINFER_WORKSPACE_BASE", "fi"),
        ):
            env[var] = os.path.join(root, sub)
            os.makedirs(env[var], exist_ok=True)
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        repo_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = repo_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        code = (
            "from tests.moe.test_b12x_cold_capture import _check_capture\n"
            f"_check_capture({case!r}, {prewarm!r}, **{kwargs!r})"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize(
    "case", ("static", "gated_dynamic", "generic_dynamic", "single_slice")
)
@pytest.mark.parametrize("prewarm", [False, True])
def test_first_use_capture(case, prewarm):
    _run_in_fresh_process(case, prewarm)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("case", ("static", "aligned_static", "gated_dynamic"))
def test_folded_alpha_capture_requires_warmup(case):
    for prewarm in (False, True):
        _run_in_fresh_process(case, prewarm, input_global_scale=True)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
def test_changed_scale_tensor_inside_capture_is_refused():
    _run_in_fresh_process(
        "static", True, input_global_scale=True, rescale_after_prewarm=True
    )


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("num_tokens", [200, 1024])
@pytest.mark.parametrize("inference_scale", ["w1_alpha", "input_global_scale"])
@pytest.mark.parametrize("per_expert", [False, True])
def test_inference_scales_refresh_in_place_on_eager_and_graph_replay(
    num_tokens, inference_scale, per_expert
):
    from flashinfer import B12xMoEWrapper

    from .test_b12x_static_extent_rules import _kwargs, _tensors

    t = _tensors(num_tokens, 320, seed=79)
    kwargs = _kwargs(t)
    kwargs["input_global_scale"] = torch.full(
        (8,) if per_expert else (1,), 0.75, device="cuda"
    )
    with torch.inference_mode():
        kwargs[inference_scale] = kwargs[inference_scale].clone()

    def make_wrapper():
        return B12xMoEWrapper(
            num_experts=8,
            top_k=2,
            hidden_size=256,
            intermediate_size=320,
            use_cuda_graph=True,
            max_num_tokens=1024,
        )

    wrapper = make_wrapper()
    # The folded output can itself be an inference tensor after warm-up.
    with torch.inference_mode():
        wrapper.run(**kwargs)
    folded = wrapper._folded_w1_alpha
    views = wrapper._weight_views

    def assert_fold():
        assert wrapper._folded_w1_alpha is folded
        assert wrapper._weight_views is views
        expected = kwargs["w1_alpha"].float() * kwargs["input_global_scale"].float()
        torch.testing.assert_close(folded, expected, rtol=0, atol=0)

    with torch.inference_mode():
        kwargs[inference_scale].mul_(1.25)
    wrapper.run(**kwargs)  # also allow calls outside inference_mode
    assert_fold()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(**kwargs)
    for factor in (0.5, 1.5):
        with torch.inference_mode():
            kwargs[inference_scale].mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        assert_fold()
        expected = make_wrapper().run(**kwargs).clone()
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-3)
