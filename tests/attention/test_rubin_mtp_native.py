# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Contract and target-only numerical tests for the native-row prototype."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available


@pytest.fixture
def native():
    if not is_cute_dsl_available():
        pytest.skip("requires CuTe DSL")
    pytest.importorskip("cutlass.utils.rubin_helpers")
    from flashinfer.cute_dsl.attention.rubin_mtp import native

    return native


class MetadataTensor:
    """Host-only tensor metadata; attempting a data read is an error."""

    def __init__(self, shape, dtype, *, strides=None, pointer=4096):
        self.tensor = (
            torch.empty(shape, dtype=dtype, device="meta")
            if strides is None
            else torch.empty_strided(shape, strides, dtype=dtype, device="meta")
        )
        self.device = torch.device("cuda", 0)
        self.pointer = pointer

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def data_ptr(self):
        return self.pointer


def _metadata_arguments(native, splits=13):
    fp8 = torch.float8_e4m3fn
    _, _, size = native.native_geometry(96, 4, 3, splits)
    return [
        MetadataTensor((3, 4, 96, 512), fp8, strides=(221184, 55296, 576, 1)),
        MetadataTensor((3, 4, 96, 64), fp8, strides=(221184, 55296, 576, 1)),
        MetadataTensor((9, 64, 512), fp8, strides=(36864, 576, 1)),
        MetadataTensor((9, 64, 64), fp8, strides=(36864, 576, 1)),
        MetadataTensor((3, 3), torch.int32),
        MetadataTensor((3, 4, 96, 512), fp8),
        MetadataTensor((3, 4, 96), torch.float32),
        MetadataTensor((size,), torch.int8) if size else None,
        splits,
        MetadataTensor((3,), torch.int32),
        192**-0.5,
        1.0,
        1.0,
    ]


@pytest.mark.parametrize("splits", [1, 13])
def test_native_prepared_rebinds_buffers(native, monkeypatch, splits):
    calls = []
    monkeypatch.setattr(
        native, "compile_native", lambda **_: lambda *args: calls.append(args)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 7))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    run = native.prepare_native_mla_decode(heads=96, q_len=4, splits=splits)
    first, second = (_metadata_arguments(native, splits) for _ in range(2))
    run(*first)
    run(*second)
    assert len(calls) == 2
    for call, supplied in zip(calls, (first, second), strict=True):
        assert all(a is b for a, b in zip(call, supplied, strict=True))


@pytest.mark.parametrize(
    "invalid",
    [
        "affine",
        "alignment",
        "overlap",
        "kv_alignment",
        "page",
        "workspace",
        "workspace_rank",
        "workspace_alignment",
        "device",
        "dtype",
        "splits",
        "scale_tensor",
        "scale_nan",
        "lse_nan",
        "lse_zero",
        "table",
    ],
)
def test_native_rejects_unsafe_metadata(native, monkeypatch, invalid):
    def forbidden(*args):
        pytest.fail("invalid metadata reached the compiled launcher")

    monkeypatch.setattr(native, "compile_native", lambda **_: forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 7))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    args = _metadata_arguments(native)
    if invalid == "affine":
        args[0] = MetadataTensor(
            (3, 4, 96, 512), torch.float8_e4m3fn, strides=(221184, 55312, 576, 1)
        )
    elif invalid == "alignment":
        args[0].pointer += 1
    elif invalid == "overlap":
        args[0] = MetadataTensor(
            (3, 4, 96, 512), torch.float8_e4m3fn, strides=(0, 55296, 576, 1)
        )
    elif invalid == "kv_alignment":
        args[2] = MetadataTensor(
            (9, 64, 512), torch.float8_e4m3fn, strides=(36928, 577, 1)
        )
    elif invalid == "page":
        args[2] = MetadataTensor((9, 128, 512), torch.float8_e4m3fn)
    elif invalid == "workspace":
        args[7] = MetadataTensor((32,), torch.int8)
    elif invalid == "workspace_rank":
        args[7] = MetadataTensor((1, args[7].numel()), torch.int8)
    elif invalid == "workspace_alignment":
        args[7].pointer += 16
    elif invalid == "device":
        args[3].device = torch.device("cuda", 1)
    elif invalid == "dtype":
        args[5] = MetadataTensor((3, 4, 96, 512), torch.bfloat16)
    elif invalid == "splits":
        args[8] = 1
    elif invalid == "scale_tensor":
        args[10] = torch.tensor(1.0, device="meta")
    elif invalid == "scale_nan":
        args[10] = math.nan
    elif invalid == "lse_nan":
        args[12] = math.nan
    elif invalid == "lse_zero":
        args[12] = 0.0
    elif invalid == "table":
        args[4] = MetadataTensor((3, 0), torch.int32)
    run = native.prepare_native_mla_decode(heads=96, q_len=4, splits=13)
    with pytest.raises(ValueError):
        run(*args)


def test_native_prepare_refuses_graph_capture(native, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        native.prepare_native_mla_decode(heads=96, q_len=4)


def test_native_workspace_index_bounds(native):
    # Byte extent is int64; actual float32 element indexing remains signed32.
    assert native.native_geometry(96, 8, 64, 32)[2] > 2**31
    with pytest.raises(ValueError, match="indexing"):
        native.native_geometry(96, 8, 10000, 32)
    with pytest.raises(ValueError, match="integer"):
        native.native_geometry(96, 4, splits=True)


def _reference(q, kv, table, lengths, scale, output_scale, lse_scale):
    batch, q_len, heads, _ = q.shape
    outputs, lses = [], []
    for b in range(batch):
        length = int(lengths[b])
        rows = q[b].float().reshape(-1, 576)
        cache = kv[table[b].long()].reshape(-1, 576)[:length].float()
        if length == 0:
            outputs.append(torch.zeros((q_len, heads, 512), device=q.device))
            lses.append(torch.full((q_len, heads), -math.inf, device=q.device))
            continue
        logits = rows @ cache.T * scale
        bounds = (
            length
            - q_len
            + 1
            + torch.arange(q_len, device=q.device).repeat_interleave(heads)
        )
        logits.masked_fill_(
            torch.arange(length, device=q.device)[None, :] >= bounds[:, None], -math.inf
        )
        lse = torch.logsumexp(logits, -1)
        weights = torch.exp(logits - torch.where(torch.isfinite(lse), lse, 0)[:, None])
        outputs.append(
            (weights @ cache[:, :512] * output_scale).reshape(q_len, heads, 512)
        )
        lses.append((lse * (lse_scale / math.log(2))).reshape(q_len, heads))
    return torch.stack(outputs), torch.stack(lses)


@pytest.mark.parametrize("heads,q_len", [(96, 2), (96, 4), (96, 8), (128, 4)])
@pytest.mark.parametrize("branch", ["auto", "preferred", "fallback"])
@pytest.mark.parametrize(
    "splits,page_size,reducer_kind",
    [(1, 64, "shared"), (13, 128, "shared"), (6, 64, "warp"), (13, 128, "warp")],
)
def test_native_decode_live_graph(
    native, heads, q_len, branch, splits, page_size, reducer_kind
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("full native MTP execution requires SM107")
    import tvm_ffi
    from cutlass import Float32, Int32
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    torch.manual_seed(6177)
    batch, max_kv = 3, 4097
    pages = (max_kv + page_size - 1) // page_size
    q = torch.randn((batch, q_len, heads, 576), device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn((batch * pages, page_size, 576), device="cuda").to(q.dtype)
    table = (
        torch.randperm(batch * pages, device="cuda")
        .to(torch.int32)
        .reshape(batch, pages)
    )
    lengths = torch.tensor([0, 1, max_kv], device="cuda", dtype=torch.int32)
    out = torch.empty((batch, q_len, heads, 512), device="cuda", dtype=q.dtype)
    lse = torch.empty((batch, q_len, heads), device="cuda", dtype=torch.float32)
    _, _, size = native.native_geometry(heads, q_len, batch, splits)
    scratch = torch.empty(size, device="cuda", dtype=torch.int8) if size else None
    run = native.prepare_native_mla_decode(
        heads=heads,
        q_len=q_len,
        page_size=page_size,
        splits=splits,
        branch=branch,
        reducer_kind=reducer_kind,
        correction=8.0 if reducer_kind == "warp" else 0.0,
        clusters=max(get_max_active_clusters(4), get_max_active_clusters(2) // 2),
    )
    scale, output_scale = 192**-0.5, 0.75
    lse_scale = math.log(2) if splits > 1 else 1.0

    def launch():
        with tvm_ffi.use_torch_stream():
            run(
                q[..., :512],
                q[..., 512:],
                kv[..., :512],
                kv[..., 512:],
                table,
                out,
                lse,
                scratch,
                Int32(splits),
                lengths,
                Float32(scale),
                Float32(output_scale),
                Float32(lse_scale),
            )

    def validate():
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            ref, ref_lse = _reference(
                q, kv, table, lengths, scale, output_scale, lse_scale
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        assert torch.isfinite(out.float()).all()
        assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse))
        assert torch.equal(
            out.float()[torch.isneginf(ref_lse)], ref[torch.isneginf(ref_lse)]
        )
        finite = torch.isfinite(ref_lse)
        torch.testing.assert_close(lse[finite], ref_lse[finite], atol=0.03, rtol=0)
        relative_rmse = (
            (out.float() - ref).square().mean() / ref.square().mean()
        ).sqrt()
        assert relative_rmse.item() < 0.04

    out.fill_(math.nan)
    lse.fill_(math.nan)
    if scratch is not None:
        scratch.fill_(0x7F)
    launch()
    validate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    q.copy_((-q.float()).to(q.dtype))
    lengths.copy_(torch.tensor([129, 17, max_kv - 2], device="cuda", dtype=torch.int32))
    table.copy_(table.flip(0))
    out.fill_(math.nan)
    lse.fill_(math.nan)
    if scratch is not None:
        scratch.fill_(0x7F)
    graph.replay()
    validate()


def test_native_rejects_unknown_reducer(native):
    with pytest.raises(ValueError, match="native reducer"):
        native.NativeMtpKernel(
            heads=96,
            q_len=4,
            page_size=64,
            splits=4,
            clusters=52,
            reducer_kind="unknown",
        )
