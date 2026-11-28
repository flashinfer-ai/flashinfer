# SPDX-License-Identifier: Apache-2.0
"""Variable-head strided MRoPE: tails, shared storage, graphs and rejection."""

import pytest
import torch

from flashinfer import apply_mrope_with_cos_sin_cache_inplace
from flashinfer.trace.templates.mrope import _mrope_reference


def apply_strided_inplace(q, k, positions, cache, sections, *, axis_interleaved=False):
    result = apply_mrope_with_cos_sin_cache_inplace(
        positions,
        q.flatten(1),
        k.flatten(1),
        q.shape[2],
        cache,
        sections,
        axis_interleaved,
    )
    assert result is None
    return q, k


def reference(q, k, positions, cache, sections, *, axis_interleaved=False):
    qo, ko = _mrope_reference(
        positions,
        q.flatten(1),
        k.flatten(1),
        q.shape[2],
        cache,
        sections,
        axis_interleaved,
    )
    return qo.reshape_as(q), ko.reshape_as(k)


def inputs(
    n=17, hq=8, hk=2, d=128, r=64, dtype=torch.bfloat16, native=True, idtype=torch.int64
):
    torch.manual_seed(72)
    q = torch.randn(n, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(n, hk, d, device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (3, n), device="cuda", dtype=idtype)
    angles = torch.arange(2048, device="cuda")[:, None] * (
        10000.0 ** (-torch.arange(0, r, 2, device="cuda").float() / r)
    )
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(
        dtype if native else torch.float32
    )
    half = r // 2
    sections = (half - 2 * (half // 3), half // 3, half // 3)
    return q, k, positions, cache, sections


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def general_inputs(shape, dtype=torch.bfloat16, native=True, layout="packed"):
    n, hq, hk, d, r = shape
    q0, k0, p0, cache, sections = inputs(
        *shape,
        dtype=dtype,
        native=native,
        idtype=torch.int64 if native else torch.int32,
    )
    # Include V-like space, unaligned storage offsets and padding in every row.
    pitch = (hq + 2 * hk) * d + 19
    store = torch.full((max(n, 1) + 1, pitch), -9, device="cuda", dtype=dtype)
    qstart, kstart = (7 + hk * d, 7) if layout == "reverse" else (7, 7 + hq * d)
    q = store[:n, qstart : qstart + hq * d].view(n, hq, d)
    k = store[:n, kstart : kstart + hk * d].view(n, hk, d)
    mask = torch.ones_like(store, dtype=torch.bool)
    mask[:n, qstart : qstart + hq * d] = False
    guards = [(store, mask)]
    if layout == "separate":
        other = torch.full((max(n, 1) + 1, hk * d + 23), -9, device="cuda", dtype=dtype)
        k = other[:n, 5 : 5 + hk * d].view(n, hk, d)
        other_mask = torch.ones_like(other, dtype=torch.bool)
        other_mask[:n, 5 : 5 + hk * d] = False
        guards.append((other, other_mask))
    else:
        mask[:n, kstart : kstart + hk * d] = False
    q.copy_(q0)
    k.copy_(k0)
    pstore = torch.full((3, max(n, 1) + 31), -13, device="cuda", dtype=p0.dtype)
    p = pstore[:, 3 : 3 + n]
    p.copy_(p0)
    return q, k, p, cache, sections, guards, pstore


SHAPES = [
    (1, 16, 2, 128, 128),
    (17, 8, 1, 128, 128),
    (33, 8, 4, 128, 128),
    (7, 4, 2, 128, 128),
    (1, 2, 1, 128, 128),
    (117, 32, 8, 128, 128),
    (5, 32, 4, 128, 64),
    (3, 3, 5, 80, 60),
    (5, 7, 3, 82, 66),
    (9, 8, 2, 256, 64),
    (3, 3, 1, 512, 512),
    (3, 5, 1, 2, 2),
    (1, 1024, 1, 64, 32),
    (0, 2, 1, 128, 64),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("layout", ["packed", "reverse", "separate"])
def test_generalized_exact(shape, dtype, native, interleaved, layout):
    q, k, p, c, s, guards, pstore = general_inputs(shape, dtype, native, layout)
    expected = reference(q, k, p, c, s, axis_interleaved=interleaved)
    tails = [x[..., c.shape[1] :].clone() for x in (q, k)]
    p_before, c_before = pstore.clone(), c.clone()
    result = apply_strided_inplace(q, k, p, c, s, axis_interleaved=interleaved)
    assert result[0] is q and result[1] is k
    for actual, wanted, tail in zip(result, expected, tails, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert torch.equal(actual[..., c.shape[1] :], tail)
    for store, mask in guards:
        assert bool((store[mask] == -9).all())
    assert torch.equal(pstore, p_before) and torch.equal(c, c_before)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 1, 128, 128),
        (17, 8, 1, 128, 64),
        (5, 3, 5, 80, 60),
        (33, 8, 1, 128, 64),
        (117, 16, 2, 128, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("native", [False, True])
def test_generalized_changed_graph(shape, dtype, native):
    q, k, p, c, s, guards, _ = general_inputs(shape, dtype, native, "reverse")
    apply_strided_inplace(q, k, p, c, s, axis_interleaved=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_strided_inplace(q, k, p, c, s, axis_interleaved=True)
    for iteration in range(3):
        q.normal_()
        k.normal_()
        p.random_(0, c.shape[0])
        c.uniform_(-1, 1)
        if iteration == 1:
            p[1:].copy_(p[0].expand_as(p[1:]))
        expected = reference(q, k, p, c, s, axis_interleaved=True)
        graph.replay()
        for actual, wanted in zip((q, k), expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for store, mask in guards:
            assert bool((store[mask] == -9).all())


@pytest.mark.parametrize("sections", [(64, 0, 0), (0, 64, 0), (0, 0, 64), (24, 20, 20)])
def test_generalized_section_boundaries(sections):
    q, k, p, c, _, _, _ = general_inputs((13, 8, 1, 128, 128))
    expected = reference(q, k, p, c, sections)
    apply_strided_inplace(q, k, p, c, sections)
    for actual, wanted in zip((q, k), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        "qk_overlap",
        "cross_row",
        "unequal_pitch",
        "zero_heads",
        "too_many_heads",
        "odd_dim",
        "odd_rotary",
        "sections",
        "axis_slots",
        "cache_dtype",
    ],
)
def test_generalized_rejects_without_writes(bad):
    q, k, p, c, s, guards, _ = general_inputs((3, 3, 5, 80, 60))
    store = guards[0][0]
    pitch = store.stride(0)
    if bad == "qk_overlap":
        k = store[:3, 8:408].view(3, 5, 80)
    elif bad == "cross_row":
        k = torch.as_strided(store, k.shape, (pitch, 80, 1), pitch - 200)
    elif bad == "unequal_pitch":
        k = k.as_strided(k.shape, (pitch - 1, 80, 1))
    elif bad == "zero_heads":
        k = k[:, :0]
    elif bad == "too_many_heads":
        q = torch.zeros(3, 1025, 80, device="cuda", dtype=q.dtype)
    elif bad == "odd_dim":
        q, k = q[..., :79], k[..., :79]
    elif bad == "unequal_dim":
        k = k[..., :78]
    elif bad == "odd_rotary":
        c = c[:, :59].contiguous()
    elif bad == "sections":
        s = (-1, 11, 20)
    elif bad == "axis_slots":
        s = (0, 15, 15)
    elif bad == "cache_dtype":
        c = c.to(torch.float16)
    before = store.clone()
    with pytest.raises((RuntimeError, ValueError)):
        apply_strided_inplace(q, k, p, c, s, axis_interleaved=True)
    assert torch.equal(store, before)


@pytest.mark.parametrize(
    "shape",
    [
        (31, 8, 4, 128, 128),
        (32, 8, 4, 128, 128),
        (33, 8, 1, 128, 64),
        (32, 3, 5, 128, 64),
        (32, 48, 16, 128, 128),
        (32, 49, 16, 128, 128),
        (95, 8, 1, 128, 64),
        (96, 7, 1, 128, 64),
        (96, 8, 1, 128, 64),
        (95, 2, 1, 128, 128),
        (96, 2, 1, 128, 128),
        (95, 8, 4, 128, 128),
        (96, 8, 4, 128, 128),
        (127, 8, 4, 128, 128),
        (128, 8, 4, 128, 128),
        (384, 8, 4, 128, 128),
        (385, 8, 4, 128, 128),
        (223, 8, 1, 128, 64),
        (224, 8, 1, 128, 64),
        (384, 8, 1, 128, 64),
        (385, 8, 1, 128, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_prefill_dispatch_boundaries(shape, dtype, native, interleaved):
    q, k, p, c, s, guards, _ = general_inputs(shape, dtype, native, "reverse")
    expected = reference(q, k, p, c, s, axis_interleaved=interleaved)
    apply_strided_inplace(q, k, p, c, s, axis_interleaved=interleaved)
    for actual, wanted in zip((q, k), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for store, mask in guards:
        assert bool((store[mask] == -9).all())


@pytest.mark.parametrize(
    "bad",
    [
        "width",
        "rank",
        "channel_stride",
        "dtype",
        "position_dtype",
        "position_stride",
        "metadata_alias",
        "sections_type",
        "head_size",
    ],
)
def test_public_invalid_metadata_does_not_mutate(bad):
    q, k, p, c, s, guards, _ = general_inputs((3, 8, 4, 128, 64))
    q, k = q.flatten(1), k.flatten(1)
    d = 128
    if bad == "width":
        q = q[:, :-1]
    elif bad == "rank":
        q = q.view(3, 8, 128)
    elif bad == "channel_stride":
        q = q[:, ::2]
    elif bad == "dtype":
        q = q.float()
    elif bad == "position_dtype":
        p = p.float()
    elif bad == "position_stride":
        p = torch.zeros(3, 6, dtype=torch.int64, device="cuda")[:, ::2]
    elif bad == "metadata_alias":
        c = q[0].view(-1, 64)
    elif bad == "sections_type":
        s = (True, 10, 21)
    elif bad == "head_size":
        d = 0
    saved = [store.clone() for store, _ in guards]
    with pytest.raises((RuntimeError, ValueError)):
        apply_mrope_with_cos_sin_cache_inplace(p, q, k, d, c, s)
    for (store, _), original in zip(guards, saved, strict=True):
        assert torch.equal(store, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["packed", "separate"])
@pytest.mark.parametrize(
    "shape", [(117, 8, 4, 128, 64), (128, 8, 4, 128, 128), (224, 8, 1, 128, 64)]
)
@torch._dynamo.config.patch(recompile_limit=32)
def test_torch_compile_mutation_and_changing_inputs(dtype, layout, shape):
    q, k, p, c, s, guards, _ = general_inputs(shape, dtype, True, layout)
    q, k = q.flatten(1), k.flatten(1)

    def run(q, k, p, c):
        apply_mrope_with_cos_sin_cache_inplace(p, q, k, 128, c, s, True)
        return q, k

    compiled = torch.compile(run, fullgraph=True)
    for _ in range(2):
        q.normal_()
        k.normal_()
        p.random_(0, c.shape[0])
        c.uniform_(-1, 1)
        expected = _mrope_reference(p, q, k, 128, c, s, True)
        result = compiled(q, k, p, c)
        assert (
            result[0].data_ptr() == q.data_ptr()
            and result[1].data_ptr() == k.data_ptr()
        )
        for actual, wanted in zip((q, k), expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for store, mask in guards:
            assert bool((store[mask] == -9).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(1, 16, 8, 128, 128), (117, 8, 4, 128, 128), (3, 3, 5, 80, 60)]
)
def test_native_cache_overflow(dtype, shape):
    q, k, p, c, s, guards, _ = general_inputs(shape, dtype, True)
    q.fill_(torch.finfo(dtype).max)
    k.fill_(-torch.finfo(dtype).max)
    c.fill_(2**-0.5)
    expected = reference(q, k, p, c, s, axis_interleaved=True)
    apply_strided_inplace(q, k, p, c, s, axis_interleaved=True)
    for actual, wanted in zip((q, k), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    half = c.shape[1] // 2
    assert torch.isposinf(q[..., half : 2 * half]).all()
    assert torch.isneginf(k[..., half : 2 * half]).all()
    for store, mask in guards:
        assert bool((store[mask] == -9).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch._dynamo.config.patch(recompile_limit=32)
def test_torch_compile_dynamic_packed_offsets(dtype):
    def run(q, k, p, c):
        apply_mrope_with_cos_sin_cache_inplace(p, q, k, 128, c, (22, 21, 21), True)
        return q, k

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for n, layout in [
        (96, "packed"),
        (128, "reverse"),
        (213, "separate"),
        (385, "packed"),
    ]:
        q, k, p, c, s, guards, _ = general_inputs(
            (n, 8, 4, 128, 128), dtype, True, layout
        )
        q, k = q.flatten(1), k.flatten(1)
        for _ in range(2):
            q.normal_()
            k.normal_()
            p.random_(0, c.shape[0])
            expected = _mrope_reference(p, q, k, 128, c, s, True)
            result = compiled(q, k, p, c)
            assert result[0].data_ptr() == q.data_ptr()
            assert result[1].data_ptr() == k.data_ptr()
            for actual, wanted in zip((q, k), expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            for store, mask in guards:
                assert bool((store[mask] == -9).all())


@pytest.mark.parametrize("bad", ["qk_overlap", "metadata_alias"])
def test_torch_compile_rejects_aliases_without_writes(bad):
    q, k, p, c, s, guards, _ = general_inputs((128, 8, 4, 128, 128))
    q, k = q.flatten(1), k.flatten(1)
    if bad == "qk_overlap":
        k = q[:, :512]
    else:
        c = q[0].view(-1, 128)
    original = [store.clone() for store, _ in guards]

    def run(q, k, p, c):
        apply_mrope_with_cos_sin_cache_inplace(p, q, k, 128, c, s, True)

    with pytest.raises((RuntimeError, ValueError)):
        torch.compile(run, fullgraph=True)(q, k, p, c)
    for (store, _), before in zip(guards, original, strict=True):
        assert torch.equal(store, before)
