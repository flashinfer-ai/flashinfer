"""SM90 safe quantization must stage identical bytes into reusable workspaces."""

import importlib

import pytest
import torch


_PACKAGE = "flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl"


def _modules():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 is required")
    return importlib.import_module(_PACKAGE + ".staging"), importlib.import_module(
        _PACKAGE + ".safe_quantize"
    )


def _workspace(capacity, hidden, topk, *, strided=False):
    outputs, guards = [], []
    for shape, dtype in (
        ((capacity, hidden), torch.float8_e4m3fn),
        ((capacity, 4), torch.float32),
        ((capacity, topk), torch.int64),
        ((capacity, topk), torch.float32),
    ):
        stride = 2 if strided else 1
        count = shape[0] * shape[1] * stride
        storage = torch.empty(count + 2, dtype=dtype, device="cuda")
        storage.view(torch.uint8).fill_(0x5A)
        outputs.append(storage[1:-1].view(shape[0], shape[1] * stride)[:, ::stride])
        guards.append(storage)
    return outputs, guards


def _expected(quantizer, x, ids, weights, prior):
    result = [t.clone() for t in prior]
    n = x.shape[0]
    if n:
        fp32 = x.float()
        payload, scales = quantizer(fp32, fp32.abs().amax(1, keepdim=True))
        result[0][:n].copy_(payload)
        result[1][:n].copy_(scales.expand(n, 4))
        result[2][:n].copy_(ids)
        result[3][:n].copy_(weights)
    result[2][n:].fill_(-1)
    return result


@pytest.mark.arch_hopper
@pytest.mark.parametrize("hidden,topk", [(129, 2), (4096, 4), (7168, 8)])
def test_safe_staging_reuses_workspace_across_graphs(hidden, topk):
    staging, safe = _modules()
    torch.manual_seed(20261018)
    x = torch.randn((33, hidden), device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1].mul_(1e-32)
    x[2, 0] = -128
    # Reduction must retain finite BF16 extremes, including subnormals.
    x[3].fill_(2.0**-133)
    x[4].fill_(-torch.finfo(torch.bfloat16).max)
    ids = torch.randint(-1, 64, (33, topk), device="cuda")
    weights = torch.randn((33, topk), device="cuda")
    outputs, guards = _workspace(65, hidden, topk)

    def stage(n):
        staging.stage_mega_moe_inputs(
            x[:n],
            weights[:n],
            ids[:n],
            *outputs,
            quantize_input=True,
            safe_quantization=True,
        )

    # Both large and small graphs share the same buffers. A later smaller
    # batch must clear routes written by the larger graph on every replay.
    graphs = {}
    for n in (33, 1):
        stage(n)
        assert safe._compiled_bf16_stager.cache_info().currsize > 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage(n)
        graphs[n] = graph
    original = (x.clone(), ids.clone(), weights.clone())
    for phase in ("original", "changed", "restored"):
        if phase == "changed":
            x.normal_()
            ids.random_(-1, 64)
            weights.normal_()
        elif phase == "restored":
            x.copy_(original[0])
            ids.copy_(original[1])
            weights.copy_(original[2])
        for n in (33, 1, 0, 33, 1):
            expected = _expected(
                safe.quantize_from_amax, x[:n], ids[:n], weights[:n], outputs
            )
            if n:
                graphs[n].replay()
            else:
                stage(n)
            for actual, reference in zip(outputs, expected, strict=True):
                assert torch.equal(
                    actual.view(torch.uint8), reference.view(torch.uint8)
                )
            for storage in guards:
                assert torch.all(storage[:1].view(torch.uint8) == 0x5A)
                assert torch.all(storage[-1:].view(torch.uint8) == 0x5A)


@pytest.mark.arch_hopper
def test_safe_staging_preserves_strided_workspace_fallback(monkeypatch):
    staging, safe = _modules()
    x = torch.randn((4, 129), device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(-1, 64, (4, 2), device="cuda", dtype=torch.int32)
    weights = torch.randn((4, 2), device="cuda")
    outputs, _ = _workspace(9, 129, 2, strided=True)
    expected = _expected(safe.quantize_from_amax, x, ids, weights, outputs)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported layouts must retain the existing staging path")

    monkeypatch.setattr(safe, "quantize_and_stage_bf16", forbidden)
    staging.stage_mega_moe_inputs(
        x, weights, ids, *outputs, quantize_input=True, safe_quantization=True
    )
    for actual, reference in zip(outputs, expected, strict=True):
        assert torch.equal(
            actual.contiguous().view(torch.uint8), reference.view(torch.uint8)
        )
