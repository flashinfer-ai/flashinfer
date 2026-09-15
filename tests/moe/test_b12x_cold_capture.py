"""First-use CUDA-graph capture contract (no silent allocation inside a capture), each case in a fresh process with fresh cache roots.

A cold capture - the very first forward of a wrapper issued inside ``torch.cuda.graph`` - would have to prepare
padded scale / FP4 copies, converted scale views, kernels or a workspace inside the capture.  The dispatch refuses that
with a clear RuntimeError naming the warm-up; after one eager warm-up call the capture succeeds and the replay matches
the eager output.  The probe (``cold_capture_probe.py``) prints one JSON line per run; running it in a subprocess with
fresh ``CUTE_DSL_CACHE_DIR`` / ``CUDA_CACHE_PATH`` / ``TORCH_EXTENSIONS_DIR`` / ``FLASHINFER_WORKSPACE_BASE`` roots
keeps process-global caches from masking the cold path.
"""

from __future__ import annotations

import json
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

PROBE = Path(__file__).with_name("cold_capture_probe.py")
CASES = ("static", "gated_dynamic", "generic_dynamic", "single_slice")


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _assert_capture_prepared_nothing(r: dict) -> None:
    """The capture retained no preparation: every cache / wrapper field is unchanged and the only
    allocator growth is the graph's own private pool (captured per-call temporaries) plus the fixed
    per-graph bookkeeping torch.cuda.graph keeps (measured on a control capture in the same process)."""
    after_prewarm, after_capture = r["after_prewarm"], r["after_capture"]
    keys = set(after_prewarm) | set(after_capture)
    volatile = {"allocated_bytes", "graph_pool_bytes"}
    assert {k: after_capture.get(k) for k in keys - volatile} == {
        k: after_prewarm.get(k) for k in keys - volatile
    }, (after_prewarm, after_capture)
    assert after_prewarm["graph_pool_bytes"] == 0, after_prewarm
    assert (
        after_capture["allocated_bytes"]
        == after_prewarm["allocated_bytes"]
        + after_capture["graph_pool_bytes"]
        + r["graph_overhead_bytes"]
    ), r


def run_probe(case: str, prewarm: bool, *extra: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="cold_capture_") as root:
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
        cmd = (
            [sys.executable, str(PROBE), case]
            + (["--prewarm"] if prewarm else [])
            + list(extra)
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        assert proc.returncode == 0 and lines, (
            f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        return json.loads(lines[-1])


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("case", CASES)
def test_cold_first_use_capture_is_refused_and_leaves_no_state(case):
    r = run_probe(case, prewarm=False)
    assert r["capture"] == "refused", r
    assert r["error_type"] == "RuntimeError" and "warm-up" in r["error"], r
    # nothing was prepared inside the capture: kernel, scale, FP4 and workspace caches are as they were
    assert r["after_capture"] == r["before"], r


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("case", CASES)
def test_prewarmed_capture_succeeds_and_replays_eager(case):
    r = run_probe(case, prewarm=True)
    assert r["capture"] == "ok", r
    assert r["replay_matches_eager"] is True, r
    assert r["replay_rel_l2_vs_bf16"] < 0.30, r
    # the capture itself prepared nothing new (only its private pool grew)
    _assert_capture_prepared_nothing(r)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("case", ("static", "aligned_static", "gated_dynamic"))
def test_folded_alpha_first_use_is_refused_and_prewarmed_capture_holds_it(case):
    """A non-null input_global_scale makes the wrapper fold w1_alpha on first use: cold capture is refused with the
    warm-up error and the wrapper holds no folded alpha afterwards; after the eager warm-up the folded alpha exists and
    the capture reuses it (same pointer / key, allocator unchanged)."""
    cold = run_probe(case, False, "--input-global-scale")
    assert cold["capture"] == "refused" and "warm-up" in cold["error"], cold
    assert (
        cold["before"]["folded_w1_alpha"] is None
        and cold["after_capture"]["folded_w1_alpha"] is None
    ), cold
    assert cold["after_capture"] == cold["before"], cold
    warm = run_probe(case, True, "--input-global-scale")
    assert warm["capture"] == "ok" and warm["replay_matches_eager"] is True, warm
    assert warm["after_prewarm"]["folded_w1_alpha"] is not None
    _assert_capture_prepared_nothing(warm)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
def test_changed_scale_tensor_inside_capture_is_refused():
    """A new input_global_scale tensor (pointer / version change) after the warm-up is a folded-alpha cache miss inside
    the capture and must be refused instead of allocating; the wrapper keeps the warm-up's folded alpha."""
    r = run_probe("static", True, "--input-global-scale", "--rescale-after-prewarm")
    assert r["capture"] == "refused" and "folded FC1 alpha" in r["error"], r
    assert (
        r["after_capture"]["folded_w1_alpha"] == r["after_prewarm"]["folded_w1_alpha"]
    ), r
    assert (
        r["after_capture"]["allocated_bytes"] == r["after_prewarm"]["allocated_bytes"]
    ), r


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
