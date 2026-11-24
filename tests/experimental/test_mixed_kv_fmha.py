# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
import pytest
import torch


def _decode_v(packed, sf):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    raw = lut[codes.long()] * sf.float().repeat_interleave(16, -1)
    return raw.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def _reference(q, k, v, sf, table, lens, indptr, qk_scale, v_scale):
    output = []
    for seq, length in enumerate(lens.tolist()):
        lo, hi = indptr[seq : seq + 2].tolist()
        if hi == lo:
            continue
        rows = torch.arange(length, device=q.device)
        p = table[seq, rows // k.size(2)].long()
        t = rows % k.size(2)
        ks = k.float()[p, :, t].repeat_interleave(q.size(1) // k.size(1), 1)
        vs = _decode_v(v, sf)[p, :, t].repeat_interleave(q.size(1) // k.size(1), 1)
        scores = torch.einsum("qhd,khd->qhk", q[lo:hi].float(), ks) * qk_scale
        pos = torch.arange(length - (hi - lo), length, device=q.device)
        scores.masked_fill_(rows[None, None, :] > pos[:, None, None], -torch.inf)
        output.append(torch.einsum("qhk,khd->qhd", scores.softmax(-1), vs) * v_scale)
    return torch.cat(output)


def _writer_scale_layout(logical):
    pages, heads, tokens, groups = logical.shape
    # Independent tensor permutation: the token lane becomes the innermost axis.
    return (
        logical.reshape(pages, heads, tokens // 4, 4, 4, groups // 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape_as(logical)
    )


@pytest.mark.parametrize("dim", [128, 256])
def test_scale_addresses_match_writer_tiles(dim):
    groups = dim // 16
    token = torch.arange(64)[:, None]
    group = torch.arange(groups)[None, :]
    logical = (100 * token + group).view(1, 1, 64, groups)
    stored = _writer_scale_layout(logical)
    # Address arithmetic from vLLM's swizzle_scale_offset(t, s, scale_dim).
    t = 4 * (token // 4) + group // (groups // 4)
    s = (group % (groups // 4)) * 4 + token % 4
    torch.testing.assert_close(stored[0, 0, t, s], logical[0, 0])
    assert not torch.equal(stored, logical)


def test_reference_nibble_order_and_saturation():
    x = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2, dtype=torch.uint8
    )
    y = _decode_v(x, torch.tensor([1.0, 128.0]).to(torch.float8_e4m3fn))
    assert y[:16].tolist() == [
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        6,
        0,
        -0.5,
        -1,
        -1.5,
        -2,
        -3,
        -4,
        -6,
    ]
    assert y[23].item() == 448 and y[-1].item() == -448


@pytest.mark.parametrize("d,group", [(128, 1), (128, 4), (256, 8)])
def test_ragged_attention_and_graph_replay(d, group):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires a QMUL-capable Blackwell GPU")
    from flashinfer.attention.mixed_kv import paged_fp8_k_nvfp4_v_fmha

    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(42)
    pages, heads, page = 8, 2, 64
    # vLLM packs three differently strided views inside each physical page.
    sizes = [heads * page * n for n in (d, d // 2, d // 16)]
    storage = torch.empty((pages, sum(sizes) + 128), device=device, dtype=torch.uint8)
    k, v, sf, _ = storage.split(sizes + [128], dim=1)
    fp8 = torch.float8_e4m3fn
    k = k.view(pages, heads, page, d).view(fp8)
    v = v.view(pages, heads, page, d // 2)
    sf = sf.view(pages, heads, page, d // 16).view(fp8)
    k.copy_(torch.randn(k.shape, generator=rng, device=device))
    v.random_(0, 256, generator=rng)
    logical_sf = (torch.rand(sf.shape, generator=rng, device=device) + 0.0625).to(fp8)
    sf.copy_(_writer_scale_layout(logical_sf))
    q = torch.randn((37, heads * group, d), generator=rng, device=device).to(fp8)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    table = torch.tensor(
        [[6, 5, 2], [0, 4, 3], [7, 1, 2]], device=device, dtype=torch.int32
    )
    lens = torch.tensor([65, 1, 129], device=device, dtype=torch.int32)
    ptr = torch.tensor([0, 36, 37, 37], device=device, dtype=torch.int32)
    scales = dict(qk_scale=0.8 / d**0.5, v_scale=1.75)

    def run():
        return paged_fp8_k_nvfp4_v_fmha(
            q, k, v, sf, table, lens, ptr, max_q_len=36, out=out, **scales
        )

    def check():
        ref = _reference(q, k, v, logical_sf, table, lens, ptr, **scales)
        err = (out.float() - ref).norm() / ref.norm()
        assert out.isfinite().all() and err < 0.055, err
        # A single valid KV has unit probability; no softmax error can mask
        # a nibble, scale, or lane-order bug in this case.
        if lens[1].item() == 1:
            torch.testing.assert_close(
                out[-1], ref[-1].bfloat16(), rtol=0.008, atol=0.003
            )

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths, offsets in [
        ([73, 65, 1], [0, 15, 37, 37]),
        ([1, 2, 132], [0, 1, 1, 37]),
    ]:
        table.copy_(table.roll(1, dims=0))
        lens.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        ptr.copy_(torch.tensor(offsets, device=device, dtype=torch.int32))
        out.fill_(torch.nan)
        graph.replay()
        check()


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    ("width", "batch", "hk"),
    [(3, 4, 2), (160, 4, 2), (136, 1, 2), (3, 16, 2), (160, 64, 2), (625, 16, 8)],
)
def test_short_decode_graph(layout, width, batch, hk):
    from flashinfer.attention.mixed_kv import paged_fp8_k_nvfp4_v_fmha

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires a QMUL-capable Blackwell GPU")
    dev = "cuda"
    fp8 = torch.float8_e4m3fn
    torch.manual_seed(21)

    def allocate(d, dtype):
        if layout == "HND":
            return torch.empty((12, hk, 64, d), device=dev, dtype=dtype)
        return torch.empty((12, 64, hk, d), device=dev, dtype=dtype).transpose(1, 2)

    k = allocate(128, fp8)
    k.copy_(torch.randn(k.shape, device=dev))
    v = allocate(64, torch.uint8).random_(0, 256)
    sf = allocate(8, fp8)
    logical = (torch.rand(sf.shape, device=dev) + 0.0625).to(fp8)
    sf.copy_(_writer_scale_layout(logical))
    table = torch.tensor(
        [[6, 7, 9], [9, 1, 3], [7, 5, 2], [1, 0, 4]], device=dev, dtype=torch.int32
    )
    table = table.repeat((batch + 3) // 4, (width + 2) // 3)[
        :batch, :width
    ].contiguous()
    states = (
        [([8192], [0, 1]), ([8703], [0, 1])]
        if batch == 1
        else [
            ([129, 0, 65, 1], [0, 1, 1, 2, 3]),
            ([0, 66, 1, width * 64 - 39], [0, 0, 1, 2, 3]),
        ]
    )
    if batch > 4:
        pad = batch - 4
        (l0, p0), (l1, p1) = states
        states = [
            (l0 + [0] * pad, p0 + [p0[-1]] * pad),
            ([0] * pad + l1, [0] * pad + p1),
        ]
    lengths, offsets = states[0]
    q = torch.randn((offsets[-1], 4 * hk, 128), device=dev).to(fp8)
    if torch.cuda.get_device_capability() == (10, 3):
        from flashinfer.experimental.mixed_kv_bulk_decode import supported

        assert supported(q, k, v, sf, table) == (
            layout == "HND" and batch >= 8 and width > 32
        )
    lens = torch.tensor(lengths, device=dev, dtype=torch.int32)
    ptr = torch.tensor(offsets, device=dev, dtype=torch.int32)
    out = torch.empty_like(q, dtype=torch.bfloat16)

    def run():
        return paged_fp8_k_nvfp4_v_fmha(
            q,
            k,
            v,
            sf,
            table,
            lens,
            ptr,
            max_q_len=1,
            out=out,
            qk_scale=0.073,
            v_scale=1.8,
        )

    run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for l, p in states:
        lens.copy_(torch.tensor(l, device=dev))
        ptr.copy_(torch.tensor(p, device=dev))
        table.copy_(table.roll(1, 1))
        out.fill_(torch.nan)
        g.replay()
        ref = _reference(q, k, v, logical, table, lens, ptr, 0.073, 1.8)
        assert (
            torch.isfinite(out).all()
            and (out.float() - ref).norm() / ref.norm() < 0.055
        )
