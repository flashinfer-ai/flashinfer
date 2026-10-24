# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import pytest
import torch
from flashinfer import gdn_prefill as fi
from flashinfer.utils import get_compute_capability
from .test_cute import _assert_relative


@pytest.fixture(autouse=True)
def sm120(request):
    if request.node.name == "test_import_isolation":
        return
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (12, 0):
        pytest.skip("requires SM120")
    if torch.cuda.get_device_properties("cuda").multi_processor_count != 156:
        pytest.skip("public fast-path qualification requires RTX6000D / 156 SMs")


def inputs(lengths=(64, 96), dtype=torch.bfloat16, index_dtype=torch.int32):
    torch.manual_seed(6342)
    total = sum(lengths)
    hq, hv = 2, 6
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(total, hq, 128, device="cuda"), dim=-1
        ).to(dtype)
        for _ in range(2)
    ]
    v = torch.randn(total, hv, 128, device="cuda", dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(total, hv, device="cuda")) / 16
    beta = torch.rand_like(g)
    initial = torch.randn(len(lengths), hv, 128, 128, device="cuda")
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    cu = torch.tensor(offsets, device="cuda", dtype=index_dtype)
    kwargs = dict(
        g=g,
        beta=beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=offsets,
        gate_domain="log",
        max_seqlen=max(lengths),
    )
    return (q, k, v), kwargs


@pytest.mark.parametrize("lengths", [(64, 96), (65, 95)])
@pytest.mark.parametrize("use_cp", ["auto", False, True])
@pytest.mark.parametrize("preallocated", [False, True])
@torch.inference_mode()
def test_public_forward(lengths, use_cp, preallocated, monkeypatch):
    from flashinfer.experimental.gdn_sm120_cute import _api

    args, kwargs = inputs(lengths)
    kwargs["use_cp"] = use_cp
    expected = fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)
    calls = []
    original = _api.chunk_gated_delta_rule

    def capture(*a, **kw):
        calls.append(kw["plan"].full_chunks)
        return original(*a, **kw)

    monkeypatch.setattr(_api, "chunk_gated_delta_rule", capture)
    if preallocated:
        kwargs.update(
            output=torch.empty_like(args[2]),
            output_state=torch.empty_like(kwargs["initial_state"]),
        )
    actual = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    assert calls == [all(n % 32 == 0 for n in lengths)]
    for x, y in zip(actual, expected, strict=True):
        _assert_relative(x, y.double())
    if preallocated:
        assert actual[0] is kwargs["output"] and actual[1] is kwargs["output_state"]


@pytest.mark.parametrize(
    "checkpoint_dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("lengths", [(100,), (64, 32), (200, 200), (32,)])
@torch.inference_mode()
def test_public_checkpoint_remainders(lengths, checkpoint_dtype, monkeypatch):
    from flashinfer.experimental.gdn_sm120_cute import _api

    args, kwargs = inputs(lengths)
    kwargs["use_cp"] = False
    counts = [n // 64 for n in lengths]
    starts = [0]
    for n in counts:
        starts.append(starts[-1] + n)
    kwargs.update(
        checkpoint_every_n_tokens=64,
        checkpoint_cu_starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
    )
    shape = (sum(counts), 6, 128, 128)
    reference_checkpoints = torch.full(
        shape, float("nan"), device="cuda", dtype=checkpoint_dtype
    )
    kwargs["state_checkpoints"] = reference_checkpoints
    expected = fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)

    def forbidden(*a, **kw):
        raise AssertionError("checkpoints must use official implementation")

    monkeypatch.setattr(_api, "chunk_gated_delta_rule", forbidden)
    actual_checkpoints = torch.full(
        shape, float("nan"), device="cuda", dtype=checkpoint_dtype
    )
    kwargs["state_checkpoints"] = actual_checkpoints
    actual = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y)
    torch.testing.assert_close(actual_checkpoints, reference_checkpoints)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
@torch.inference_mode()
def test_public_low_precision_state_fallback(dtype, monkeypatch):
    from flashinfer.experimental.gdn_sm120_cute import _api

    args, kwargs = inputs()
    kwargs["use_cp"] = False
    kwargs["initial_state"] = (kwargs["initial_state"] * 0.01).to(dtype)
    kwargs["output_state"] = torch.empty_like(kwargs["initial_state"])
    expected = fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)
    expected = (expected[0].clone(), expected[1].float().clone())

    def forbidden(*a, **kw):
        raise AssertionError("non-FP32 state must fall back")

    monkeypatch.setattr(_api, "chunk_gated_delta_rule", forbidden)
    actual = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1].float(), expected[1])


@torch.inference_mode()
def test_public_state_pool_fallback():
    args, kwargs = inputs()
    kwargs.update(
        use_cp=False,
        state_indices=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
    )
    initial = torch.randn(3, 6, 128, 128, device="cuda")
    kwargs["initial_state"] = initial
    kwargs["output_state"] = initial.clone()
    expected = fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)
    expected = (expected[0].clone(), expected[1].clone())
    kwargs["output_state"] = initial.clone()
    actual = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y)


@torch.inference_mode()
def test_public_output_state_without_return_and_graph():
    args, kwargs = inputs()
    kwargs.update(
        use_cp=False,
        output=torch.empty_like(args[2]),
        output_state=torch.empty_like(kwargs["initial_state"]),
        output_final_state=False,
    )
    fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)
    expected = kwargs["output_state"].clone()
    out = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    assert isinstance(out, torch.Tensor)
    _assert_relative(kwargs["output_state"], expected.double())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_relative(kwargs["output_state"], expected.double())


@torch.inference_mode()
def test_public_mutated_metadata():
    args, kwargs = inputs((64, 96))
    fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    kwargs["cu_seqlens"].copy_(
        torch.tensor([0, 65, 160], device="cuda", dtype=torch.int32)
    )
    kwargs["cu_seqlens_cpu"] = [0, 65, 160]
    expected = fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)
    actual = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    for x, y in zip(actual, expected, strict=True):
        _assert_relative(x, y.double())


def test_import_isolation():
    code = "import sys; import flashinfer.gdn_prefill; assert not any(n.startswith('flashinfer.experimental.gdn_sm120_cute') for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("mode", ["linear", "beta_logits", "no_cpu_metadata"])
@torch.inference_mode()
def test_public_gate_and_metadata_fallback(mode, monkeypatch):
    from flashinfer.experimental.gdn_sm120_cute import _api

    args, kwargs = inputs()
    if mode == "linear":
        kwargs.update(g=kwargs["g"].exp(), gate_domain="linear")
    elif mode == "beta_logits":
        kwargs["beta_is_logit"] = True
    else:
        kwargs["cu_seqlens_cpu"] = None
    expected = fi.chunk_gated_delta_rule(*args, backend="flashinfer", **kwargs)

    def forbidden(*a, **kw):
        raise AssertionError("unsupported gate/metadata call must fall back")

    monkeypatch.setattr(_api, "chunk_gated_delta_rule", forbidden)
    actual = fi.chunk_gated_delta_rule(*args, backend="sm120_cute", **kwargs)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y)
