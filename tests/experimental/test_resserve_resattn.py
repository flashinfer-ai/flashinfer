# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResServe grouped residue attention against decoded FP32 attention."""

import pytest
import torch

from flashinfer.resserve import paged_decode_residue


def unpack(packed):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    return table[indices].flatten(-2)


def decoded_q(q, h):
    from flashinfer.experimental.resserve_resattn.host import _quantize_q_residue_cuda

    packed, sf = _quantize_q_residue_cuda(q, h)
    b, hq, d = q.shape
    g = hq // h
    pg = 8 if g == 4 else 16
    rows = torch.arange(pg, device=q.device)[:, None]
    cols = torch.arange(d // 16, device=q.device)[None, :]
    indices = (cols // 4) * 512 + (rows % 32) * 16 + (rows // 32) * 4 + cols % 4
    scales = sf.float().reshape(b, h, -1)[..., indices]
    values = unpack(packed.reshape(b, h, pg, d // 2)) * scales.repeat_interleave(
        16, dim=-1
    )
    base, residue = values[:, :, :g], values[:, :, pg // 2 : pg // 2 + g]
    restored = (base + residue).reshape(b, hq, d)
    assert (restored - q.float()).square().mean() < (
        base.reshape_as(q) - q.float()
    ).square().mean() * 0.1
    return restored


def inputs(g, d, kv_layout="HND", length=601):
    torch.manual_seed(42)
    b, h, pages_per_seq = 2, 2, (length + 31) // 32
    q = torch.randn(b, h * g, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randint(
        0, 256, (b * pages_per_seq, h, 32, d // 2), device="cuda", dtype=torch.uint8
    )
    v = torch.randint(0, 256, k.shape, device="cuda", dtype=torch.uint8)
    if kv_layout == "NHD":
        k, v = (x.transpose(1, 2).contiguous().transpose(1, 2) for x in (k, v))
    scale_shape = (*k.shape[:-1], d // 16)
    ks = (torch.rand(scale_shape, device="cuda") * 0.1 + 0.025).to(torch.float8_e4m3fn)
    vs = (torch.rand(scale_shape, device="cuda") * 0.2 + 0.05).to(torch.float8_e4m3fn)
    vs = vs.reshape(*k.shape[:2], 8, 4, d // 16).transpose(-1, -2).contiguous()
    tables = (
        torch.randperm(b * pages_per_seq, device="cuda").int().reshape(b, pages_per_seq)
    )
    lengths = torch.tensor(
        [length, max(1, length - 82)], device="cuda", dtype=torch.int32
    )
    return q, k, v, ks, vs, tables, lengths


def reference(args, key_scale=1.0, value_scale=1.0, fp8_single_tile=False):
    q, k, v, ks, vs, tables, lengths = args
    h = k.shape[1]
    dq = decoded_q(q, h)
    dk = unpack(k) * ks.float().repeat_interleave(16, dim=-1) * key_scale
    # QMUL4 rounds/saturates each decoded V to E4M3 before the global scale.
    linear_vs = vs.transpose(-1, -2).reshape_as(ks).float()
    dv = unpack(v) * linear_vs.repeat_interleave(16, dim=-1)
    dv = dv.clamp(-448, 448).to(torch.float8_e4m3fn).float() * value_scale
    outputs, bounds = [], []
    for i in range(q.shape[0]):
        length = int(lengths[i])
        ki = dk[tables[i].long()].permute(1, 0, 2, 3).flatten(1, 2)[:, :length]
        vi = dv[tables[i].long()].permute(1, 0, 2, 3).flatten(1, 2)[:, :length]
        ki, vi = (x.repeat_interleave(q.shape[1] // h, dim=0) for x in (ki, vi))
        logits = (dq[i, :, None, :] * ki).sum(-1) * q.shape[-1] ** -0.5
        probabilities = (logits - logits.max(-1, keepdim=True).values).exp()
        denominator = probabilities.sum(-1, keepdim=True)
        if fp8_single_tile:
            assert length <= 256
            probabilities = probabilities.to(torch.float8_e4m3fn).float()
        expected = ((probabilities / denominator)[..., None] * vi).sum(1)
        outputs.append(expected)
        # E4M3 normal relative rounding <= 1/16; subnormal absolute <= 1/1024.
        # Online max corrections can only reduce these absolute error bounds.
        p_error = probabilities / 16 + 1 / 1024
        fp8_bound = (p_error[..., None] * vi.abs()).sum(1) / denominator
        bounds.append(fp8_bound + (expected.abs() + fp8_bound) / 256 + 1e-5)
    return torch.stack(outputs), torch.stack(bounds)


@pytest.fixture(autouse=True)
def require_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("ResAttn requires SM100")
    import cutlass.cutlass_dsl as dsl

    if not dsl.target_version(min_version="13.4"):
        pytest.skip("ResAttn requires CUTLASS DSL 4.8 with CUDA 13.4 PTX compiler")


@pytest.mark.parametrize(
    "g,d", [(4, 128), (6, 128), (8, 128), (4, 256), (6, 256), (8, 256)]
)
@pytest.mark.parametrize("splits", [None, 1, 2])
def test_decode(g, d, splits):
    args = inputs(g, d)
    expected, bound = reference(args, key_scale=0.75, value_scale=1.25)
    actual = paged_decode_residue(
        *args, key_scale=0.75, value_scale=1.25, num_splits=splits
    )
    assert torch.all((actual.float() - expected).abs() <= bound)


def test_changed_q_graph():
    args = inputs(4, 128)
    for _ in range(3):
        paged_decode_residue(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_decode_residue(*args)
    for multiplier in [0.0, 1.0, -2.0]:
        args[0].copy_(torch.randn_like(args[0]) * multiplier)
        graph.replay()
        eager = paged_decode_residue(*args)
        torch.testing.assert_close(out, eager, atol=0, rtol=0)


@pytest.mark.parametrize("splits", [None, 1, 2])
@pytest.mark.parametrize("length", [1, 127])
def test_single_tile_fp8_probability_reference(splits, length):
    args = inputs(8, 128)
    args[-1].fill_(length)
    expected, _ = reference(args, fp8_single_tile=True)
    actual = paged_decode_residue(*args, num_splits=splits)
    torch.testing.assert_close(actual.float(), expected, atol=3e-4, rtol=0.004)


def test_reject_linear_v_scale_layout():
    args = list(inputs(4, 128))
    args[4] = args[4].transpose(-1, -2).contiguous().view_as(args[3])
    with pytest.raises(ValueError, match="four-token interleaved"):
        paged_decode_residue(*args)


@pytest.mark.parametrize("splits", [None, 1, 2])
@pytest.mark.parametrize("g,d", [(4, 128), (8, 256)])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_changed_metadata_graph(g, d, splits, kv_layout):
    """Replay one graph while metadata and cache contents cross tile boundaries."""
    args = inputs(g, d, kv_layout)
    q, k, v, ks, vs, tables, lengths = args
    for _ in range(3):
        paged_decode_residue(*args, num_splits=splits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_decode_residue(*args, num_splits=splits)
    for length in [31, 32, 33, 127, 128, 129, 255, 256, 257]:
        q.normal_()
        k.random_(0, 256)
        v.random_(0, 256)
        tables.copy_(torch.randperm(k.shape[0], device=q.device).reshape_as(tables))
        lengths.copy_(
            torch.tensor([length, length + 1], device=q.device, dtype=lengths.dtype)
        )
        graph.replay()
        expected, bound = reference(args)
        assert torch.all((out.float() - expected).abs() <= bound)
        eager = paged_decode_residue(*args, num_splits=splits)
        torch.testing.assert_close(out, eager, atol=0, rtol=0)


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_engine_cache_views_and_output(d, kv_layout):
    """Use page-strided cache views and a padded table without repacking KV."""
    args = list(inputs(4, d))
    for i in range(1, 5):
        original = args[i]
        nhd = i in (1, 2) and kv_layout == "NHD"
        if nhd:
            original = original.transpose(1, 2)
        padded = torch.empty(
            (original.shape[0], original.shape[1] * 2, *original.shape[2:]),
            device="cuda",
            dtype=original.dtype,
        )
        view = padded[:, : original.shape[1]]
        view.copy_(original)
        args[i] = view.transpose(1, 2) if nhd else view
    table = args[-2]
    padded_table = torch.empty(
        table.shape[0], table.shape[1] + 7, device="cuda", dtype=torch.int32
    )
    args[-2] = padded_table[:, : table.shape[1]]
    args[-2].copy_(table)
    expected, bound = reference(args)
    out = torch.empty_like(args[0])
    actual = paged_decode_residue(*args, num_splits=2, out=out)
    assert actual is out
    assert torch.all((actual.float() - expected).abs() <= bound)
    args[-1][-1] = 0
    padded = paged_decode_residue(*args, num_splits=2)
    torch.testing.assert_close(padded[0], actual[0], atol=0, rtol=0)


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_reject_non_dense_inner_pages(index):
    args = list(inputs(4, 128))
    original = args[index]
    storage = torch.empty(
        (*original.shape[:-1], original.shape[-1] * 2),
        device=original.device,
        dtype=original.dtype,
    )
    args[index] = storage[..., ::2]
    with pytest.raises(ValueError, match="contiguous inner page contents"):
        paged_decode_residue(*args)


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize(
    "length,splits",
    [(1024, None), (2048, None), (4096, None), (4097, 1), (4097, 2), (4097, 64)],
)
def test_long_decode_and_empty_splits(d, length, splits, monkeypatch):
    """Exercise automatic CGA4/8/16, repeated pipeline stages and empty splits."""
    monkeypatch.delenv("RESIDUE_ATTN_REDUCTION", raising=False)
    args = inputs(6, d, length=length)
    # One active token leaves all but one split empty for the second request.
    args[-1][1] = 1
    expected, bound = reference(args)
    actual = paged_decode_residue(*args, num_splits=splits)
    assert torch.all((actual.float() - expected).abs() <= bound)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_decode_residue(*args, num_splits=splits)
    # Reuse the captured allocation as the long row shrinks and the short row grows.
    args[-1].copy_(torch.tensor([length - 1, 257], device="cuda", dtype=torch.int32))
    graph.replay()
    expected, bound = reference(args)
    assert torch.all((out.float() - expected).abs() <= bound)
    eager = paged_decode_residue(*args, num_splits=splits)
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def test_cga_wait_acquires_peer_writes(tmp_path, monkeypatch):
    from cutlass import cute

    from flashinfer.experimental.resserve_resattn import host

    compile_kernel = cute.compile

    def compile_with_ptx(*args, **kwargs):
        kwargs["options"] += f" --keep-ptx --dump-dir {tmp_path}"
        return compile_kernel(*args, **kwargs)

    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    monkeypatch.setattr(cute, "compile", compile_with_ptx)
    host._compiled.__wrapped__(12, 2, 128, "cga_sharded", 128)
    (ptx_path,) = tmp_path.glob("*.ptx")
    ptx = ptx_path.read_text()
    # The relaxed cluster wait and restricted acquire fence order peer DSMEM stores.
    wait = ptx.index("mbarrier.try_wait.parity.relaxed.cluster")
    assert "fence.acquire.sync_restrict::shared::cluster.cluster" in ptx[wait:]
