"""Byte-level shared staging checks, independent of the MoE numerical tolerance."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107 import shared_expert
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.kernel_helpers import (
    quantize_mxfp8_block32,
    swizzled_flat_sf_size,
    to_blocked,
)

pytestmark = pytest.mark.arch_rubin


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107")
    pytest.importorskip("triton")
    return torch.device("cuda")


def _workspace(capacity, hidden, device, kind="mxfp8_e4m3"):
    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    workspace = SimpleNamespace(
        config=SimpleNamespace(shared_expert=SimpleNamespace(kind=kind)),
        shared_x=torch.empty((capacity, hidden), device=device, dtype=dtype),
        shared_x_sf_linear=torch.empty(
            (capacity, hidden // 32), device=device, dtype=torch.float8_e8m0fnu
        ),
        shared_x_sf=torch.empty(
            swizzled_flat_sf_size(capacity, hidden // 32),
            device=device,
            dtype=torch.float8_e8m0fnu,
        ),
    )
    for name in ("shared_x", "shared_x_sf_linear", "shared_x_sf"):
        getattr(workspace, name).view(torch.uint8).fill_(197)
    return workspace


def _stage(x, workspace, *, shared_x=None, scales=None):
    tensors = SimpleNamespace(
        hidden_states=x,
        scales=scales,
        shared_hidden_states=shared_x,
        shared_scales=None,
        num_tokens=x.shape[0],
    )
    shared_expert.stage_shared_inputs(
        tensors, workspace, quantize_input=x.dtype == torch.bfloat16
    )


def _check(x, workspace, scales=None):
    data = torch.zeros_like(workspace.shared_x.view(torch.uint8))
    sf = torch.zeros_like(workspace.shared_x_sf_linear.view(torch.uint8))
    if x.shape[0]:
        if x.dtype == torch.bfloat16:
            x, scales = quantize_mxfp8_block32(x, workspace.shared_x.dtype)
        data[: x.shape[0]].copy_(x.view(torch.uint8))
        sf[: x.shape[0]].copy_(scales.view(torch.uint8))
    assert torch.equal(data, workspace.shared_x.view(torch.uint8)), "data bytes"
    assert torch.equal(sf, workspace.shared_x_sf_linear.view(torch.uint8)), (
        "scale bytes"
    )
    assert torch.equal(to_blocked(sf), workspace.shared_x_sf.view(torch.uint8)), (
        "swizzle"
    )


@pytest.mark.parametrize("count,hidden", [(0, 128), (1, 128), (32, 5120), (192, 5120)])
def test_shared_staging_exact(cuda_device, count, hidden):
    workspace = _workspace(456, hidden, cuda_device)
    x = torch.randn((count, hidden), device=cuda_device, dtype=torch.bfloat16)
    original = x.clone()
    _stage(x, workspace)
    _check(x, workspace)
    assert torch.equal(x, original)


def test_shared_staging_bf16_encodings(cuda_device):
    # Every BF16 encoding, including signed zeros, subnormals, infinities and
    # NaN payloads. Reverse block order to exercise both scale atom axes.
    x = torch.arange(65536, dtype=torch.int32, device=cuda_device)
    x = x.to(torch.uint16).view(torch.bfloat16).reshape(64, 1024)
    workspace = _workspace(129, 1024, cuda_device)
    _stage(x, workspace)
    _check(x, workspace)
    x = x.reshape(-1, 32).flip(0).reshape(64, 1024)
    _stage(x, workspace)
    _check(x, workspace)


def test_shared_staging_nonfinite_blocks(cuda_device):
    # A nonfinite block sets scale 2**-126 in the Torch reference. Other
    # values in that block can therefore overflow during division/conversion.
    x = (
        torch.tensor(
            [float("nan"), -float("nan"), float("inf"), -float("inf")],
            device=cuda_device,
            dtype=torch.bfloat16,
        )[:, None]
        .expand(4, 1024)
        .clone()
    )
    x[:, 1::32] = 1.0
    x[:, 2::32] = -1.0
    x[:, 3::32] = 0.0
    x[:, 4::32] = -0.0
    x[:, 5::32] = 2.0**-133
    x[:, 6::32] = -(2.0**-133)
    workspace = _workspace(456, 1024, cuda_device)
    _stage(x, workspace)
    _check(x, workspace)


def test_shared_staging_graphs_share_workspace(cuda_device):
    workspace = _workspace(456, 5120, cuda_device)
    inputs = [
        torch.randn((rows, 5120), device=cuda_device, dtype=torch.bfloat16)
        for rows in (32, 192, 0)
    ]
    graphs = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x in inputs:
            _stage(x, workspace)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _stage(x, workspace)
            graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)
    pointers = [
        getattr(workspace, name).data_ptr()
        for name in ("shared_x", "shared_x_sf_linear", "shared_x_sf")
    ]
    for index in (1, 0, 2, 1, 0):
        x = inputs[index]
        x.copy_(torch.randn_like(x))
        graphs[index].replay()
        _check(x, workspace)
    assert pointers == [
        getattr(workspace, name).data_ptr()
        for name in ("shared_x", "shared_x_sf_linear", "shared_x_sf")
    ]


def test_shared_staging_separate_input(cuda_device):
    routed = torch.zeros((32, 128), device=cuda_device, dtype=torch.bfloat16)
    shared = torch.randn_like(routed)
    workspace = _workspace(456, 128, cuda_device)
    _stage(routed, workspace, shared_x=shared)
    _check(shared, workspace)
    assert torch.count_nonzero(routed) == 0


@pytest.mark.parametrize(
    "mode",
    [
        "strided",
        "e5m2",
        "padded_columns",
        "prequantized",
        "disabled",
        "large_capacity",
        "large_blocked_scales",
    ],
)
def test_shared_staging_fallback(cuda_device, monkeypatch, mode):
    def unexpected_fast_path():
        raise AssertionError("unsupported input entered the fused stager")

    monkeypatch.setattr(shared_expert, "_shared_bf16_stager", unexpected_fast_path)
    if mode == "disabled":
        monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    capacity = 65_536 if mode == "large_capacity" else 137
    hidden = 160 if mode == "padded_columns" else 128
    kind = "mxfp8_e5m2" if mode == "e5m2" else "mxfp8_e4m3"
    x = torch.randn((7, hidden * 2), device=cuda_device, dtype=torch.bfloat16)[:, ::2]
    if mode != "strided":
        x = x.contiguous()
    workspace = _workspace(capacity, hidden, cuda_device, kind)
    if mode == "large_blocked_scales":
        monkeypatch.setattr(workspace.shared_x_sf, "numel", lambda: 2**31 + 1)
    sf = None
    if mode == "prequantized":
        x, sf = quantize_mxfp8_block32(x, workspace.shared_x.dtype)
    _stage(x, workspace, scales=sf)
    _check(x, workspace, scales=sf)


@pytest.mark.parametrize(
    "torch_version,cuda_version", [("2.11.0", "13.0"), ("2.12.0", "12.9")]
)
def test_shared_stager_requires_saturating_cast(
    monkeypatch, torch_version, cuda_version
):
    stager = shared_expert._shared_bf16_stager
    stager.cache_clear()
    try:
        monkeypatch.setattr(torch, "__version__", torch_version)
        monkeypatch.setattr(torch.version, "cuda", cuda_version)
        assert stager() is None
    finally:
        stager.cache_clear()
