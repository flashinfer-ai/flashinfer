# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Independent FP64 oracle and changed-input replay for CuTe DS4.1 decode."""

import builtins
import dataclasses

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_decode, deepseek_v41_window_decode


def gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("CuTe DS4.1 decode validation targets SM100")
    pytest.importorskip("cutlass.experimental.primitives")


def pack(data, scales, width):
    return torch.cat(
        (
            data.reshape(-1, 64 * data.shape[1]),
            scales.reshape(-1, 64 * scales.shape[1]),
        ),
        dim=1,
    ).view(-1, 64, 1, width)


def make_case(batch, compressed_k, seed=42323):
    torch.manual_seed(seed + batch + compressed_k)
    q = torch.randn(batch, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
    # Encoded finite bytes, with independent scale decoding in the oracle.
    wdata = (
        torch.randn(512, 512, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )
    wscale = torch.randint(124, 130, (512, 16), device="cuda", dtype=torch.uint8)
    swa = pack(wdata, wscale, 528)
    cdata = torch.randint(0, 256, (1024, 256), device="cuda", dtype=torch.uint8)
    cscale = (
        (torch.rand(1024, 32, device="cuda") + 0.125)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    main = pack(cdata, cscale, 288) if compressed_k else None
    wi = torch.randint(0, 512, (batch, 1, 128), device="cuda", dtype=torch.int32)
    ci = (
        torch.randint(
            0, 1024, (batch, 1, compressed_k), device="cuda", dtype=torch.int32
        )
        if compressed_k
        else None
    )
    sink = torch.randn(64, device="cuda")
    return q, swa, main, wi, ci, sink


def unpack(pool, fp4):
    pages = pool.view(pool.shape[0], -1)
    if not fp4:
        data = (
            pages[:, : 64 * 512]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(-1, 512)
            .double()
        )
        exponent = pages[:, 64 * 512 :].reshape(-1, 16).double() - 127
        return (
            (data * torch.exp2(exponent).repeat_interleave(32, -1)).bfloat16().double()
        )
    data = pages[:, : 64 * 256].reshape(-1, 256)
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(-1, 512)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    scales = (
        pages[:, 64 * 256 :]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(-1, 32)
        .double()
    )
    return (table[codes.long()] * scales.repeat_interleave(16, -1)).bfloat16().double()


def oracle(args):
    q, swa, main, wi, ci, sink = args
    values, masks = [], []
    for pool, ids, fp4 in ((swa, wi, False), (main, ci, True)):
        if pool is None:
            continue
        decoded = unpack(pool, fp4)
        ids = ids[:, 0].long()
        valid = (ids >= 0) & (ids < decoded.shape[0])
        gathered = decoded[ids.clamp(0, decoded.shape[0] - 1)]
        values.append(torch.where(valid[..., None], gathered, 0))
        masks.append(valid)
    kv, valid = torch.cat(values, dim=1), torch.cat(masks, dim=1)
    scores = torch.einsum("bhd,bkd->bhk", q[:, 0].double(), kv) * (512**-0.5)
    scores.masked_fill_(~valid[:, None], -float("inf"))
    lse = scores.logsumexp(-1)
    augmented = torch.cat(
        (scores, sink.double()[None, :, None].expand(q.shape[0], -1, 1)), dim=-1
    )
    probability = augmented.softmax(-1)[..., :-1].nan_to_num()
    out = torch.einsum("bhk,bkd->bhd", probability, kv)
    return out, lse


def check(args, out, lse):
    expected, expected_lse = oracle(args)
    diff = out[:, 0].double() - expected
    assert torch.isfinite(out).all()
    assert diff.norm() / expected.norm().clamp_min(1e-30) < 0.005
    assert diff.abs().max() / expected.abs().max().clamp_min(1e-30) < 0.01
    empty = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(lse[..., 0]), empty)
    torch.testing.assert_close(
        lse[..., 0].double()[~empty], expected_lse[~empty], rtol=0, atol=1e-5
    )
    assert torch.equal(out[:, 0][empty], torch.zeros_like(out[:, 0][empty]))


@pytest.mark.parametrize(
    "batch,compressed_k",
    [
        (1, 512),
        (16, 512),
        (64, 512),
        (128, 512),
        (129, 512),
        pytest.param(None, 512, id="uneven-persistent"),
        (256, 512),
        (512, 512),
        (4, 0),
        (128, 0),
        (2, 64),
        (8, 192),
    ],
)
def test_decode_changed_graph_and_fp64(batch, compressed_k, monkeypatch):
    gate()
    if batch is None:
        # Exercise CTAs with unequal request counts on the current GPU.
        batch = torch.cuda.get_device_properties("cuda").multi_processor_count + 1
    original_import = builtins.__import__

    def no_old_decode(name, *a, **kw):
        if name.startswith("flash_mla") or name.endswith("decode_fp32"):
            raise AssertionError(f"unexpected alternative decode route: {name}")
        return original_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_old_decode)
    args = make_case(batch, compressed_k)
    out, lse, plan = deepseek_v41_decode(*args)

    def run():
        if compressed_k:
            return deepseek_v41_decode(*args, plan=plan)
        return deepseek_v41_window_decode(args[0], args[1], args[3], args[5], plan=plan)

    check(args, out, lse)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(3):
        args[0].normal_()
        args[3].random_(0, 512)
        args[3][..., step::4] = -1
        if compressed_k:
            args[4].random_(0, 1024)
            args[4][..., step::3] = -1
        args[5].normal_()
        args[5][0], args[5][1], args[5][2] = 1000, -1000, -float("inf")
        if step == 1:
            # New cache bytes at the same addresses must affect graph replay.
            replacement = make_case(batch, compressed_k, seed=8)
            args[1].copy_(replacement[1])
            if compressed_k:
                args[2].copy_(replacement[2])
        if step == 2:
            args[3][0] = -1
            if compressed_k:
                args[4][0] = -1
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        check(args, out, lse)
    assert run()[0].data_ptr() == out.data_ptr()
    with pytest.raises(AssertionError):
        check(args, torch.ones_like(out), lse)
    with pytest.raises(AssertionError):
        check(args, out, torch.zeros_like(lse))
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_decode(*args, plan=dataclasses.replace(plan, out=args[0]))
    with pytest.raises(ValueError, match="mismatch"):
        deepseek_v41_decode(*args, plan=object())
    with pytest.raises(ValueError, match="inference"):
        deepseek_v41_decode(args[0].requires_grad_(), *args[1:], plan=plan)


def test_empty_uninitialized_cache_and_declaration_guards():
    gate()
    args = make_case(2, 512)
    args[1].fill_(255)
    args[2].fill_(255)
    args[3].fill_(-1)
    args[4].fill_(2**31 - 1)
    args[5].fill_(-float("inf"))
    out, lse, _ = deepseek_v41_decode(*args)
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()
    with pytest.raises(ValueError, match="Q must"):
        deepseek_v41_decode(args[0].expand(-1, 2, -1, -1).contiguous(), *args[1:])
    with pytest.raises(ValueError, match="page size"):
        deepseek_v41_decode(args[0], args[1].view(-1, 32, 1, 528), *args[2:])
    with pytest.raises(TypeError):
        deepseek_v41_decode(*args, arithmetic="bf16x3")


@pytest.mark.parametrize("batch", [128, 256])
def test_cache_offsets_above_two_gib(batch):
    gate()
    args = make_case(batch, 512)
    main = args[2]
    # Leave the prefix uninitialized: only the copied tail is a valid slot.
    # The independent oracle still reads the original small encoded pool.
    first_page = (2**31 + 64 * 288 - 1) // (64 * 288)
    large = torch.empty(
        (first_page + main.shape[0], 64, 1, 288),
        device="cuda",
        dtype=torch.uint8,
    )
    large[first_page:].copy_(main)
    shifted = args[4] + first_page * 64
    out, lse, _ = deepseek_v41_decode(
        args[0], args[1], large, args[3], shifted, args[5]
    )
    check(args, out, lse)


@pytest.mark.parametrize("pool_index,width,ids_index", [(1, 528, 3), (2, 288, 4)])
@pytest.mark.parametrize("below_limit", [True, False])
def test_persistent_metadata_capacity_boundary(
    pool_index, width, ids_index, below_limit
):
    gate()
    batch = torch.cuda.get_device_properties("cuda").multi_processor_count + 1
    args = list(make_case(batch, 512))
    small = args[pool_index]
    page_bytes = 64 * width
    limit = 2**35  # Signed Int32 offsets expressed in 16-byte units.
    pages = (
        (limit - 1) // page_bytes
        if below_limit
        else (limit + page_bytes - 1) // page_bytes + small.shape[0]
    )
    if torch.cuda.mem_get_info()[0] < pages * page_bytes + 2 * 1024**3:
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < pages * page_bytes + 2 * 1024**3:
            pytest.skip("Capacity-boundary validation needs a 32-GiB cache pool")
    # Only the tail is initialized; the oracle continues to use the small pool.
    large = torch.empty((pages, 64, 1, width), device="cuda", dtype=torch.uint8)
    first_page = pages - small.shape[0]
    large[first_page:].copy_(small)
    actual = list(args)
    actual[pool_index] = large
    actual[ids_index] = args[ids_index] + first_page * 64
    out, lse, plan = deepseek_v41_decode(*actual)
    check(args, out, lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_decode(*actual, plan=plan)
    args[0].mul_(0.75)
    args[ids_index].copy_(args[ids_index].flip(-1))
    actual[ids_index].copy_(actual[ids_index].flip(-1))
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    check(args, out, lse)


@pytest.mark.parametrize(
    "levels",
    [
        [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.75],
        [0.0625, 0.5, 0.125, 0.75, 0.125, 0.375, 0.0625, 1.0],
        [0.0625] * 7 + [0.75],
    ],
)
@pytest.mark.parametrize("batch", [16, 128, 256])
def test_normalization_anchor_crossings(levels, batch):
    gate()
    q, swa, main, wi, ci, sink = make_case(batch, 512)
    # Exact scalar products with head-dependent signs exercise differing
    # rescale decisions in the same warp. Last tile introduces a new maximum.
    head_scale = torch.linspace(-0.75, 1.25, 64, device=q.device).bfloat16()
    q.copy_(head_scale[None, None, :, None].expand_as(q))
    if batch > 128:
        # Persistent CTAs must reset row statistics and O when their next
        # request reverses which heads need normalization-anchor changes.
        q[batch // 2 :].mul_(-0.5)
    wd = (
        torch.zeros((128, 512), device=q.device)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    ws = torch.full((128, 16), 127, device=q.device, dtype=torch.uint8)
    swa = pack(wd, ws, 528)
    cd = torch.full((512, 256), 0x22, device=q.device, dtype=torch.uint8)
    cs = (
        torch.tensor(levels, device=q.device)
        .repeat_interleave(64)[:, None]
        .expand(512, 32)
        .contiguous()
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    main = pack(cd, cs, 288)
    wi = (
        torch.arange(128, device=q.device, dtype=torch.int32)[None, None]
        .expand(batch, 1, 128)
        .contiguous()
    )
    ci = (
        torch.arange(512, device=q.device, dtype=torch.int32)[None, None]
        .expand(batch, 1, 512)
        .contiguous()
    )
    sink.fill_(-float("inf"))
    args = (q, swa, main, wi, ci, sink)
    out, lse, _ = deepseek_v41_decode(*args)
    check(args, out, lse)


def test_fp4_conversion_fallback():
    gate()
    from flashinfer.experimental.deepseek_v41.decode import _compile

    args = make_case(128, 512)
    _, _, plan = deepseek_v41_decode(*args)
    fallback = dataclasses.replace(
        plan,
        kernel=_compile(args[0].device.index, 512, False, 16, 2, False, False),
    )
    out, lse, _ = deepseek_v41_decode(*args, plan=fallback)
    check(args, out, lse)


def test_native_fp4_all_finite_codes():
    gate()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.experimental import primitives as prims
    from flashinfer.experimental.deepseek_v41 import hca_v41_primitives as hw

    if (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) < (13, 2):
        pytest.skip("native BF16 conversion requires bundled CUDA >= 13.2")

    @cute.kernel
    def convert(words: cute.Tensor, scales: cute.Tensor, out: cute.Tensor):
        i = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if i < words.shape[0]:
            scale = hw.fp8x2_to_bf16_word(scales[i] * 257)
            values = hw.fp4x8_to_bf16_words(words[i])
            for j in cutlass.range_constexpr(4):
                out[i, j] = prims.mul_bf16x2(values[j], scale)

    @cute.jit
    def launch(words, scales, out, stream: cuda.CUstream):
        convert(words, scales, out).launch(
            grid=(cute.ceil_div(words.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    # Cross every pair of FP4 codes with every finite E4M3 scale. Permute
    # other bytes to expose lane-order bugs; compare signed zeros bitwise.
    ids = torch.arange(65536, device="cuda", dtype=torch.int64)
    sc, byte = ids & 255, ids >> 8
    keep = (sc != 127) & (sc != 255)
    words = (
        byte | ((byte ^ 0x53) << 8) | ((byte ^ 0x97) << 16) | ((byte ^ 0xE1) << 24)
    )[keep].int()
    scales = sc[keep].int()
    out = torch.empty((words.numel(), 4), device="cuda", dtype=torch.int32)
    tensors = [from_dlpack(x) for x in (words, scales, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *tensors, stream)
    compiled(*tensors, stream)
    lut = torch.tensor(
        [0.0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    codes = (words.long()[:, None] >> (4 * torch.arange(8, device="cuda"))) & 15
    scale_values = scales.to(torch.uint8).view(torch.float8_e4m3fn).double()
    expected = (lut[codes] * scale_values[:, None]).bfloat16()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
