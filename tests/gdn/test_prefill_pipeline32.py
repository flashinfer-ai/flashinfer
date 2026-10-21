# SPDX-License-Identifier: Apache-2.0
"""SM120 p32 reference checks, including tails and public opt-in dispatch."""

import pytest
import torch

from flashinfer.utils import get_compute_capability


def _reference(q, k, v, alpha, beta, initial, lengths, scale):
    """Independent token recurrence in FP64, state laid out as [H,V,K]."""
    heads = v.shape[1]
    q = q.double().repeat_interleave(heads // q.shape[1], dim=1)
    k = k.double().repeat_interleave(heads // k.shape[1], dim=1)
    v = v.double()
    outputs, states = [], []
    offset = 0
    for seq, length in enumerate(lengths):
        state = initial[seq].double().clone()
        for token in range(offset, offset + length):
            state = state * alpha[token, :, None, None].double()
            prediction = torch.einsum("hvk,hk->hv", state, k[token])
            delta = beta[token, :, None].double() * (v[token] - prediction)
            state = state + delta[:, :, None] * k[token, :, None, :]
            outputs.append(scale * torch.einsum("hvk,hk->hv", state, q[token]))
        states.append(state)
        offset += length
    output = torch.stack(outputs) if outputs else v.new_empty((0, heads, 128))
    return output, torch.stack(states)


def _assert_relative(actual, expected):
    assert actual.shape == expected.shape
    diff = torch.linalg.vector_norm(actual.double() - expected)
    norm = torch.linalg.vector_norm(expected)
    error = diff / norm if norm.item() else diff
    assert torch.isfinite(error) and error.item() <= 0.02


@pytest.mark.parametrize("lengths", [(33,), (65,), (0, 31, 0, 33), (0,)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pipeline32_reference_and_dispatch(lengths, dtype, monkeypatch):
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("pipeline32 requires SM120")
    from flashinfer import gdn_prefill

    torch.manual_seed(42)
    total, hq, hv = sum(lengths), 2, 6
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(total, hq, 128, device="cuda"), dim=-1
        ).to(dtype)
        for _ in range(2)
    ]
    v = torch.randn(total, hv, 128, device="cuda", dtype=dtype)
    alpha = (
        torch.nn.functional.logsigmoid(torch.randn(total, hv, device="cuda"))
        .div(16)
        .exp()
    )
    beta = torch.randn(total, hv, device="cuda").sigmoid()
    initial = torch.randn(len(lengths), hv, 128, 128, device="cuda")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    scale = 128**-0.5
    expected = _reference(q, k, v, alpha, beta, initial, lengths, scale)
    monkeypatch.setenv("FLASHINFER_GDN_SM120_PIPELINE32", "1")

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("pipeline32 opted in but official fuse was called")

    monkeypatch.setattr(
        gdn_prefill, "chunk_gated_delta_rule_sm120", unexpected_fallback
    )
    actual = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        g=alpha,
        beta=beta,
        scale=scale,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
        backend="flashinfer",
    )
    for result, reference in zip(actual, expected, strict=False):
        _assert_relative(result, reference)
    # Replays exercise the same ping-pong slots without reusing updated state.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            g=alpha,
            beta=beta,
            scale=scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            use_cp=False,
            backend="flashinfer",
        )
    graph.replay()
    graph.replay()
    for result, reference in zip(captured, expected, strict=False):
        _assert_relative(result, reference)

    raw_q, raw_k = [torch.randn_like(q) for _ in range(2)]
    normalized = [
        (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(
            dtype
        )
        for x in (raw_q, raw_k)
    ]
    expected_normalized = _reference(
        *normalized, v, alpha, beta, initial, lengths, scale
    )
    actual_normalized = gdn_prefill.chunk_gated_delta_rule(
        raw_q,
        raw_k,
        v,
        g=alpha,
        beta=beta,
        scale=scale,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        use_cp=False,
        backend="flashinfer",
        use_qk_l2norm_in_kernel=True,
    )
    for result, reference in zip(actual_normalized, expected_normalized, strict=True):
        _assert_relative(result, reference)


@pytest.fixture
def small_inputs():
    """Stable, nonempty GDN inputs for cache, checkpoint and fallback regressions."""
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("pipeline32 requires SM120")
    torch.manual_seed(17)
    lengths, hq, hv = (64, 32), 2, 6
    total = sum(lengths)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(total, hq, 128, device="cuda"), dim=-1
        ).bfloat16()
        for _ in range(2)
    ]
    v = torch.randn(total, hv, 128, device="cuda", dtype=torch.bfloat16)
    alpha = torch.full((total, hv), 0.95, device="cuda")
    beta = torch.rand(total, hv, device="cuda")
    initial = torch.randn(2, hv, 128, 128, device="cuda")
    return q, k, v, alpha, beta, initial, lengths


@pytest.mark.parametrize("reverse", [False, True], ids=["int32-first", "int64-first"])
@pytest.mark.parametrize("interval", [0, 32], ids=["no-checkpoints", "checkpoints"])
def test_pipeline32_index_dtype_switch_and_checkpoints(small_inputs, reverse, interval):
    """Both index dtypes specialize independently, including mixed dtype pairs."""
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_prefill_pipeline32
    from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import (
        _in_mem_compile_cache,
    )

    q, k, v, alpha, beta, initial, lengths = small_inputs
    scale = 128**-0.5
    expected = _reference(q, k, v, alpha, beta, initial, lengths, scale)
    cp_expected = []
    for seq, (start, length) in enumerate(((0, 64), (64, 32))):
        for end in range(32, length + 1, 32):
            cp_expected.append(
                _reference(
                    q[start : start + end],
                    k[start : start + end],
                    v[start : start + end],
                    alpha[start : start + end],
                    beta[start : start + end],
                    initial[seq : seq + 1],
                    (end,),
                    scale,
                )[1][0]
            )
    cp_expected = torch.stack(cp_expected)
    _in_mem_compile_cache.clear()
    pairs = [
        (torch.int32, torch.int32),
        (torch.int64, torch.int64),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
    ]
    if reverse:
        pairs = [pairs[1], pairs[0], pairs[3], pairs[2]]
    for cu_dtype, cp_dtype in pairs:
        cu = torch.tensor([0, 64, 96], device="cuda", dtype=cu_dtype)
        starts = torch.tensor([0, 2, 3], device="cuda", dtype=cp_dtype)
        output, final = torch.empty_like(v), torch.empty_like(initial)
        checkpoints = (
            torch.full((3, 6, 128, 128), float("nan"), device="cuda")
            if interval
            else None
        )
        starts = starts if interval else None
        delta_rule_prefill_pipeline32(
            output,
            final,
            q,
            k,
            v,
            initial,
            alpha,
            beta,
            cu,
            scale,
            checkpoints,
            starts,
            interval,
        )
        torch.cuda.synchronize()
        actual = (output, final, checkpoints) if interval else (output, final)
        references = (*expected, cp_expected) if interval else expected
        for result, reference in zip(actual, references, strict=True):
            _assert_relative(result, reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            delta_rule_prefill_pipeline32(
                output,
                final,
                q,
                k,
                v,
                initial,
                alpha,
                beta,
                cu,
                scale,
                checkpoints,
                starts,
                interval,
            )
        graph.replay()
        torch.cuda.synchronize()
        for result, reference in zip(actual, references, strict=True):
            _assert_relative(result, reference)


def test_pipeline32_checkpoint_capacity_before_launch(small_inputs, monkeypatch):
    """An undersized checkpoint allocation fails before compilation or launch."""
    from flashinfer.gdn_kernels.delta_rule_dsl import (
        delta_rule_sm120_pipeline32 as kernel,
    )

    q, k, v, alpha, beta, initial, _ = small_inputs

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("undersized buffer reached compilation")

    monkeypatch.setattr(kernel, "cached_compile", unexpected_compile)
    with pytest.raises(ValueError, match="checkpoint buffer needs at least 3 entries"):
        kernel.delta_rule_prefill_pipeline32(
            torch.empty_like(v),
            torch.empty_like(initial),
            q,
            k,
            v,
            initial,
            alpha,
            beta,
            torch.tensor([0, 64, 96], device="cuda", dtype=torch.int32),
            128**-0.5,
            torch.empty(2, 6, 128, 128, device="cuda"),
            torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32),
            32,
        )


@pytest.mark.parametrize("reason", ["state_pool", "sm121", "kernel_unavailable"])
def test_pipeline32_opt_in_preserves_official_fallback(
    small_inputs, monkeypatch, reason
):
    """Opt-in keeps official support for pools, other SM12x and missing kernels."""
    from flashinfer import gdn_prefill

    q, k, v, alpha, beta, initial, lengths = small_inputs
    kwargs = dict(
        g=alpha,
        beta=beta,
        scale=128**-0.5,
        cu_seqlens=torch.tensor([0, 64, 96], device="cuda", dtype=torch.int32),
        output_final_state=True,
        use_cp=False,
        backend="flashinfer",
    )
    if reason == "state_pool":
        pool = torch.randn(4, 6, 128, 128, device="cuda")
        pool[[3, 1]] = initial
        initial = pool
        kwargs["state_indices"] = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    elif reason == "sm121":
        # Exercise dispatch on real SM120; this is not an SM121 hardware test.
        monkeypatch.setattr(
            gdn_prefill, "get_compute_capability", lambda device: (12, 1)
        )
    else:
        monkeypatch.setattr(gdn_prefill, "delta_rule_prefill_pipeline32", None)
    official = gdn_prefill.chunk_gated_delta_rule_sm120
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return official(*args, **kwargs)

    monkeypatch.setattr(gdn_prefill, "chunk_gated_delta_rule_sm120", spy)
    results = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("FLASHINFER_GDN_SM120_PIPELINE32", enabled)
        results.append(
            gdn_prefill.chunk_gated_delta_rule(
                q,
                k,
                v,
                initial_state=initial.clone(),
                output_state=torch.zeros_like(initial),
                **kwargs,
            )
        )
    torch.cuda.synchronize()
    assert len(calls) == 2
    for before, after in zip(*results, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    state = results[1][1][[3, 1]] if reason == "state_pool" else results[1][1]
    selected_initial = initial[[3, 1]] if reason == "state_pool" else initial
    expected = _reference(q, k, v, alpha, beta, selected_initial, lengths, 128**-0.5)
    _assert_relative(results[1][0], expected[0])
    _assert_relative(state, expected[1])


def test_pipeline32_benchmark_without_fla(small_inputs, monkeypatch, tmp_path):
    """FI-only benchmarks do not import FLA, even for accuracy checks."""
    import importlib.util
    from pathlib import Path
    import sys

    spec = importlib.util.spec_from_file_location(
        "bench_pipeline32",
        Path(__file__).resolve().parents[2]
        / "benchmarks/bench_gdn_sm120_pipeline32.py",
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    monkeypatch.setitem(sys.modules, "fla", None)
    monkeypatch.setattr(bench, "elapsed_ms", lambda fn, args: 1.0)
    args = bench.parser().parse_args(
        [
            "--backend",
            "event",
            "--seqlen",
            "1x64",
            "--fi-path",
            "p32",
            "--check-acc",
            "--json",
            str(tmp_path / "result.json"),
        ]
    )
    assert bench.run(args) == 0
