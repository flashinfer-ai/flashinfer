# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Memory contracts of the RoPE APIs in ``flashinfer.rope``; numerics over the
shape matrix are covered by test_rope.py. Every group is one ``-k`` selection:

* ``rope_output_contract``: every output element is written (outputs start as
  poison), nothing next to a caller's output view is, the non-rotated tail is
  the input bit for bit, and in place equals out of place.
* ``rope_strided``: q/k cut out of a fused projection (``qkv.split``) give
  bitwise the results of contiguous copies, in-place calls write through the
  caller's views, and a strided last dimension is rejected.
* ``rope_paged_append_contract``: the fused append writes exactly the slots of
  the real tokens, in separate and combined cache layouts.
* ``rope_pos_ids_dtype``: int32 and int64 positions agree bitwise.
* ``rope_backend_agree``: ``rope_quantize_fp8`` backends ``cuda`` and ``cutile``.
* ``rope_pdl``: PDL ordering of the three FP8 kernels, with the race harness
  of tests/utils/test_norm.py.
"""

import contextlib
import math
import os
from types import SimpleNamespace

import pytest
import torch
from tests.test_helpers.rope_reference import RotaryEmbedding, apply_scaling
from tests.utils.test_norm import (
    _PDL_REPEATS,
    _PDL_SEED,
    _PDL_SPIN_CYCLES,
    _pdl_launcher,
    _pdl_race_module,
    _skip_unless_pdl,
    _skip_unless_pdl_overlaps,
)

import flashinfer
from flashinfer.utils import get_compute_capability

_F8 = torch.float8_e4m3fn
_PAGE = 16
# Byte poisons for FP8 buffers: 0x7F is NaN in e4m3fn and in e5m2, 0x00 is +0.
_FP8_POISONS = (0x7F, 0x00)


def _randn(shape, dtype, gen):
    return torch.randn(shape, generator=gen, device="cuda").to(dtype)


def _bits(t):
    """Raw bit patterns, so NaNs and signed zeros compare exactly."""
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _assert_bitwise(actual, expected, what):
    assert actual.shape == expected.shape, f"{what}: {actual.shape} != {expected.shape}"
    bad = _bits(actual) != _bits(expected)
    if bad.any():
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {bad.numel()} elements differ bitwise; "
            f"first at {bad.nonzero()[:4].tolist()}"
        )


def _fp8_steps(a, b):
    """Distance in representable values between two tensors of one FP8 format."""

    def ordinal(t):
        code = t.view(torch.uint8).to(torch.int16)
        return torch.where(code >= 128, 128 - code, code)  # sign-magnitude codes

    return (ordinal(a) - ordinal(b)).abs()


def _assert_fp8_close(actual, expected, what):
    """``actual`` is finite and within one FP8 step of the fp32 ``expected``:
    the kernel rounds an fp32 value computed in a different order (FMA)."""
    if actual.numel() == 0:
        return
    assert torch.isfinite(actual.float()).all(), f"{what}: non-finite (unwritten?)"
    steps = int(_fp8_steps(actual, expected.to(actual.dtype)).max())
    assert steps <= 1, f"{what}: {steps} FP8 steps away from the reference"


def _cos_sin_cache(rotary_dim, max_pos, theta=1e4, llama31=False):
    """``[cos | sin]`` table of shape ``(max_pos, rotary_dim)``, fp32. With
    ``llama31`` the frequencies get the Llama 3.1 scaling with the API defaults
    (factor 8, low 1, high 4, original context 8192)."""
    if not llama31:
        return RotaryEmbedding(
            rotary_dim, rotary_dim, max_pos, theta, True, torch.float32
        ).cos_sin_cache
    inv_freq = 1.0 / theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, apply_scaling(inv_freq)).cuda()
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _rope_reference(x, pos, table, interleave):
    """fp32 RoPE of the leading ``table.shape[-1]`` features of ``x``
    (``(nnz, heads, dim)``); the remaining features pass through."""
    rotary_dim = table.shape[-1]
    cos, sin = table[pos.long()].unsqueeze(1).chunk(2, dim=-1)
    xf = x.float()
    rot, rest = xf[..., :rotary_dim], xf[..., rotary_dim:]
    if interleave:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        rot = rot.flatten(-2)
    else:
        x1, x2 = rot.chunk(2, dim=-1)
        rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((rot, rest), dim=-1)


# Ragged batch for the indptr/offsets APIs; the second request is empty.
_INDPTR = [0, 9, 9, 23]
_OFFSETS = [5, 100, 250]


def _segments(idtype=torch.int32):
    """indptr and offsets, and the per-row positions they stand for."""
    pos = [
        o + i
        for s, e, o in zip(_INDPTR[:-1], _INDPTR[1:], _OFFSETS, strict=True)
        for i in range(e - s)
    ]
    return tuple(
        torch.tensor(v, dtype=idtype, device="cuda") for v in (_INDPTR, _OFFSETS, pos)
    )


def _reference_table(api, rotary_dim):
    if "llama31" in api:
        return _cos_sin_cache(rotary_dim, 512, theta=5e5, llama31=True)
    return _cos_sin_cache(rotary_dim, 512)


def _call_fused(api, q, k, rotary_dim, interleave, inplace, segments, cos_sin_cache):
    """Run one of the bf16/fp16 RoPE APIs on 3-D ``q``/``k``. Returns the
    outputs, or ``q``/``k`` themselves when ``inplace``."""
    rope = flashinfer.rope
    indptr, offsets, pos_ids = segments
    if api == "apply_rope_with_cos_sin_cache":
        # Documented as 2-D (nnz, heads * head_size), the way vLLM passes it.
        query, key = q.flatten(1), k.flatten(1)
        assert query.data_ptr() == q.data_ptr() and key.data_ptr() == k.data_ptr()
        args = (pos_ids, query, key, q.shape[-1], cos_sin_cache, not interleave)
        if inplace:
            rope.apply_rope_with_cos_sin_cache_inplace(*args)
            return q, k
        q_out, k_out = rope.apply_rope_with_cos_sin_cache(*args)
        return q_out.view(q.shape), k_out.view(k.shape)
    lead = (pos_ids,) if api.endswith("_pos_ids") else (indptr, offsets)
    fn = getattr(rope, api + "_inplace" if inplace else api)
    out = fn(q, k, *lead, rotary_dim=rotary_dim, interleave=interleave)
    return (q, k) if inplace else out


@contextlib.contextmanager
def _poisoned_empty_like():
    """The out-of-place APIs allocate their outputs with ``torch.empty_like``,
    whose recycled memory can already hold the right answer (an earlier call's
    output, say). Fill floating allocations with NaN so an element the kernel
    never writes shows up; the yielded list records what was handed out."""
    handed_out = []
    empty_like = torch.empty_like

    def poisoned(*args, **kwargs):
        t = empty_like(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(float("nan"))
        handed_out.append(t)
        return t

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch, "empty_like", poisoned)
        yield handed_out


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

# (api, dtype, head_dim, rotary_dim, interleave)
_FUSED_CONTRACT_CASES = [
    ("apply_rope", torch.bfloat16, 128, 64, True),
    ("apply_rope", torch.float16, 64, 64, False),
    ("apply_rope_pos_ids", torch.bfloat16, 128, 128, True),
    ("apply_rope_pos_ids", torch.float16, 256, 128, False),
    ("apply_llama31_rope", torch.bfloat16, 128, 64, False),
    ("apply_llama31_rope_pos_ids", torch.float16, 128, 96, True),
    ("apply_rope_with_cos_sin_cache", torch.bfloat16, 128, 64, False),
    ("apply_rope_with_cos_sin_cache", torch.bfloat16, 128, 128, True),
    # A head_dim outside {64, 128, 256, 512} is routed to the RopeQuantize
    # kernel, whose last pass-through chunk is shorter than rotary_dim.
    ("apply_rope_with_cos_sin_cache", torch.float16, 80, 64, False),
    ("apply_rope_with_cos_sin_cache", torch.bfloat16, 192, 128, True),
]


def _fused_case_id(case):
    api, dtype, head_dim, rotary_dim, interleave = case
    style = "interleave" if interleave else "neox"
    return f"{api}-{str(dtype)[6:]}-d{head_dim}-r{rotary_dim}-{style}"


@pytest.mark.parametrize(
    "api,dtype,head_dim,rotary_dim,interleave",
    _FUSED_CONTRACT_CASES,
    ids=[_fused_case_id(c) for c in _FUSED_CONTRACT_CASES],
)
def test_rope_output_contract(api, dtype, head_dim, rotary_dim, interleave):
    """Out of place, every output element is written (the allocation starts as
    NaN), the tail past ``rotary_dim`` is the input bit for bit and the inputs
    are left alone; in place gives bitwise the same result."""
    if os.environ.get("FLASHINFER_TEST_TORCH_COMPILE", "0") == "1":
        pytest.skip("torch.compile wrappers allocate the outputs this test poisons")
    gen = torch.Generator(device="cuda").manual_seed(3605 + head_dim + rotary_dim)
    segments = _segments()
    nnz = segments[2].numel()
    q = _randn((nnz, 4, head_dim), dtype, gen)
    k = _randn((nnz, 2, head_dim), dtype, gen)
    q_before, k_before = q.clone(), k.clone()
    table = _reference_table(api, rotary_dim)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3

    with _poisoned_empty_like() as handed_out:
        outs = _call_fused(api, q, k, rotary_dim, interleave, False, segments, table)
    torch.cuda.synchronize()
    poisoned = {t.data_ptr() for t in handed_out}
    for name, x, out in zip("qk", (q, k), outs, strict=True):
        assert out.data_ptr() in poisoned, (
            f"{name}: output did not come from empty_like"
        )
        assert torch.isfinite(out).all(), f"{name}: elements left unwritten (NaN)"
        _assert_bitwise(out[..., rotary_dim:], x[..., rotary_dim:], f"{name} tail")
        ref = _rope_reference(x, segments[2], table, interleave)
        torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
    _assert_bitwise(q, q_before, "q modified by the out-of-place call")
    _assert_bitwise(k, k_before, "k modified by the out-of-place call")

    q_in, k_in = q.clone(), k.clone()
    _call_fused(api, q_in, k_in, rotary_dim, interleave, True, segments, table)
    torch.cuda.synchronize()
    _assert_bitwise(q_in, outs[0], "q in place vs out of place")
    _assert_bitwise(k_in, outs[1], "k in place vs out of place")


# (api, num_kv_heads (0: 2-D MLA key), rope_dim, no_rope_dim, dtype, quant, is_neox)
_FP8_CONTRACT_CASES = [
    ("rope_quantize_fp8", 2, 64, 64, torch.bfloat16, _F8, True),
    # no_rope_dim < rope_dim: the one no-RoPE chunk is partial.
    ("rope_quantize_fp8", 2, 64, 32, torch.float16, torch.float8_e5m2, False),
    # No no-RoPE part: the API creates the empty q_nope / k_nope it needs.
    ("rope_quantize_fp8", 2, 128, 0, torch.bfloat16, _F8, False),
    ("mla_rope_quantize_fp8", 0, 64, 512, torch.bfloat16, _F8, False),
]


def _fp8_case_id(case):
    api, hk, rope_dim, no_rope_dim, dtype, quant, is_neox = case
    layout = f"h{hk}" if hk else "mla"
    style = "neox" if is_neox else "interleave"
    return f"{api}-{layout}-r{rope_dim}-n{no_rope_dim}-{str(quant)[13:]}-{style}"


@pytest.mark.parametrize(
    "api,num_kv_heads,rope_dim,no_rope_dim,dtype,quant_dtype,is_neox",
    _FP8_CONTRACT_CASES,
    ids=[_fp8_case_id(c) for c in _FP8_CONTRACT_CASES],
)
def test_rope_output_contract_fp8(
    api, num_kv_heads, rope_dim, no_rope_dim, dtype, quant_dtype, is_neox
):
    """Caller-owned FP8 outputs that are views into row-padded buffers: every
    element of the views is written whatever they held, the padding after them
    is not touched, the views are returned as given and the inputs are not
    modified."""
    nnz, num_qo_heads, pad = 37, 4, 16
    head_dim = rope_dim + no_rope_dim
    gen = torch.Generator(device="cuda").manual_seed(3605 + no_rope_dim)
    k_shape = (nnz, num_kv_heads, head_dim) if num_kv_heads else (nnz, head_dim)
    q = _randn((nnz, num_qo_heads, head_dim), dtype, gen)
    k = _randn(k_shape, dtype, gen)
    q_before, k_before = q.clone(), k.clone()
    pos_ids = torch.randint(0, 1024, (nnz,), generator=gen, device="cuda").int()
    cache = _cos_sin_cache(rope_dim, 1024)
    scale_q, scale_kv = 0.5, 2.0
    q_ref = _rope_reference(q, pos_ids, cache, not is_neox) * scale_q
    k_ref = _rope_reference(k.view(nnz, -1, head_dim), pos_ids, cache, not is_neox)
    k_ref = k_ref.view(k_shape) * scale_kv

    previous = None
    for poison in _FP8_POISONS:
        q_store = torch.full(
            (nnz, num_qo_heads, head_dim + pad),
            poison,
            dtype=torch.uint8,
            device="cuda",
        )
        k_store = torch.full(
            (*k_shape[:-1], head_dim + pad), poison, dtype=torch.uint8, device="cuda"
        )
        q_out = q_store.view(quant_dtype)[..., :head_dim]
        k_out = k_store.view(quant_dtype)[..., :head_dim]
        outs = dict(q_rope_out=q_out[..., :rope_dim], k_rope_out=k_out[..., :rope_dim])
        if no_rope_dim:
            outs.update(
                q_nope_out=q_out[..., rope_dim:], k_nope_out=k_out[..., rope_dim:]
            )
        returned = getattr(flashinfer.rope, api)(
            q[..., :rope_dim],
            k[..., :rope_dim],
            q[..., rope_dim:] if no_rope_dim else None,
            k[..., rope_dim:] if no_rope_dim else None,
            cache,
            pos_ids,
            is_neox=is_neox,
            quant_scale_q=scale_q,
            quant_scale_kv=scale_kv,
            **outs,
        )
        torch.cuda.synchronize()
        names = ("q_rope_out", "k_rope_out", "q_nope_out", "k_nope_out")
        for name, t in zip(names, returned, strict=True):
            if name in outs:
                assert t is outs[name], f"{name} is not the caller's tensor"
            else:
                assert t.numel() == 0, f"{name} should be empty"
        _assert_fp8_close(q_out, q_ref, f"q poison={poison:#x}")
        _assert_fp8_close(k_out, k_ref, f"k poison={poison:#x}")
        assert (q_store[..., head_dim:] == poison).all(), "q row padding written"
        assert (k_store[..., head_dim:] == poison).all(), "k row padding written"
        _assert_bitwise(q, q_before, "q input modified")
        _assert_bitwise(k, k_before, "k input modified")
        if previous is not None:
            _assert_bitwise(q_out, previous[0], "q depends on the poison")
            _assert_bitwise(k_out, previous[1], "k depends on the poison")
        previous = (q_out.clone(), k_out.clone())


# ---------------------------------------------------------------------------
# Strided inputs and outputs
# ---------------------------------------------------------------------------


def _strided_qk(layout, nnz, num_qo_heads, num_kv_heads, head_dim, dtype, gen):
    """A buffer and a function cutting the q/k views a serving engine passes
    out of it (or out of any tensor shaped like it)."""
    hq, hk, d = num_qo_heads, num_kv_heads, head_dim
    if layout == "qkv_split":
        # vLLM / SGLang: q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1);
        # 8 more columns so the row stride is not a multiple of the head size.
        buf = _randn((nnz, (hq + 2 * hk) * d + 8), dtype, gen)

        def views(t):
            q = t[:, : hq * d].view(nnz, hq, d)
            return q, t[:, hq * d : (hq + hk) * d].view(nnz, hk, d)

    else:  # head_padded: heads d + 16 elements apart
        buf = _randn((nnz, hq + hk, d + 16), dtype, gen)

        def views(t):
            return t[:, :hq, :d], t[:, hq:, :d]

    return buf, views


_STRIDED_CASES = [
    ("apply_rope", "qkv_split"),
    ("apply_rope", "head_padded"),
    ("apply_rope_pos_ids", "qkv_split"),
    ("apply_rope_pos_ids", "head_padded"),
    ("apply_llama31_rope", "qkv_split"),
    ("apply_llama31_rope_pos_ids", "head_padded"),
    # The 2-D query/key of this API cannot express the head_padded layout.
    ("apply_rope_with_cos_sin_cache", "qkv_split"),
]


@pytest.mark.parametrize(
    "api,layout", _STRIDED_CASES, ids=[f"{a}-{b}" for a, b in _STRIDED_CASES]
)
def test_rope_strided(api, layout):
    """Strided q/k give bitwise the results of contiguous copies; in place, the
    results land in the caller's views and nothing else in the buffer changes."""
    head_dim = 128
    gen = torch.Generator(device="cuda").manual_seed(3605)
    segments = _segments()
    nnz = segments[2].numel()
    buf, views = _strided_qk(layout, nnz, 4, 2, head_dim, torch.bfloat16, gen)
    q, k = views(buf)
    assert not q.is_contiguous() and not k.is_contiguous()
    before = buf.clone()
    table = _reference_table(api, head_dim)

    def run(q, k, inplace):
        return _call_fused(api, q, k, head_dim, False, inplace, segments, table)

    expected = run(q.contiguous(), k.contiguous(), inplace=False)
    got = run(q, k, inplace=False)
    torch.cuda.synchronize()
    for name, g, e in zip("qk", got, expected, strict=True):
        _assert_bitwise(g, e, f"{name} from the strided view, out of place")
    _assert_bitwise(buf, before, "out-of-place call modified its input buffer")

    run(q, k, inplace=True)
    torch.cuda.synchronize()
    for name, x, e in zip("qk", views(buf), expected, strict=True):
        _assert_bitwise(x, e, f"{name} in place through the strided view")
    written = torch.zeros(buf.shape, dtype=torch.bool, device="cuda")
    for view in views(written):
        view.fill_(True)
    _assert_bitwise(buf[~written], before[~written], "bytes outside q and k changed")


@pytest.mark.parametrize("api", ["rope_quantize_fp8", "mla_rope_quantize_fp8"])
def test_rope_strided_fp8(api):
    """RoPE/no-RoPE halves cut out of fused projection outputs give bitwise the
    results of contiguous copies."""
    nnz, hq, hk, rope_dim = 37, 4, 2, 64
    gen = torch.Generator(device="cuda").manual_seed(3605)
    dtype = torch.bfloat16
    if api == "mla_rope_quantize_fp8":
        no_rope_dim = 512
        q = _randn((nnz, hq, rope_dim + no_rope_dim + 16), dtype, gen)
        # DeepSeek's kv_a projection: [compressed kv | k_pe] for each token.
        kv_a = _randn((nnz, no_rope_dim + rope_dim + 8), dtype, gen)
        k_nope, k_rope = kv_a[:, :no_rope_dim], kv_a[:, no_rope_dim:-8]
        q_rope, q_nope = q[..., :rope_dim], q[..., rope_dim:-16]
    else:
        no_rope_dim = 64
        buf, views = _strided_qk(
            "qkv_split", nnz, hq, hk, rope_dim + no_rope_dim, dtype, gen
        )
        q, k = views(buf)
        q_rope, q_nope = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nope = k[..., :rope_dim], k[..., rope_dim:]
    inputs = (q_rope, k_rope, q_nope, k_nope)
    assert not any(t.is_contiguous() for t in inputs)
    cache = _cos_sin_cache(rope_dim, 1024)
    pos_ids = torch.randint(0, 1024, (nnz,), generator=gen, device="cuda").int()
    fn = getattr(flashinfer.rope, api)
    kwargs = dict(is_neox=False, quantize_dtype=_F8)
    got = fn(*inputs, cache, pos_ids, **kwargs)
    expected = fn(*(t.contiguous() for t in inputs), cache, pos_ids, **kwargs)
    torch.cuda.synchronize()
    for name, g, e in zip(
        ("q_rope", "k_rope", "q_nope", "k_nope"), got, expected, strict=True
    ):
        _assert_bitwise(g, e, name)


def _stride2_last_dim(shape, dtype=torch.bfloat16):
    """A zero-filled view whose last dimension has stride 2, and its storage."""
    storage_dtype = torch.uint8 if dtype.itemsize == 1 else dtype
    storage = torch.zeros(
        *shape[:-1], 2 * shape[-1], dtype=storage_dtype, device="cuda"
    )
    return storage.view(dtype)[..., ::2], storage


def _rejected_call(api, which, gen):
    """A call of ``api`` whose ``which`` tensor has a strided last dimension,
    and the storages that must come out of it unchanged."""
    hq, hk, d, rope_dim, dtype = 4, 2, 128, 64, torch.bfloat16
    rope = flashinfer.rope
    if api == "apply_rope_with_cos_sin_cache":
        nnz = 8
        query, _ = _stride2_last_dim((nnz, hq * d))
        key = _randn((nnz, hk * d), dtype, gen)
        pos_ids = torch.arange(nnz, dtype=torch.int32, device="cuda")
        cache = _cos_sin_cache(d, nnz)
        return lambda: rope.apply_rope_with_cos_sin_cache(
            pos_ids, query, key, d, cache
        ), []
    if api in ("apply_rope", "apply_rope_pos_ids"):
        indptr, offsets, pos_ids = _segments()
        nnz = pos_ids.numel()
        q, k = _randn((nnz, hq, d), dtype, gen), _randn((nnz, hk, d), dtype, gen)
        if which == "q":
            q, _ = _stride2_last_dim((nnz, hq, d))
        else:
            k, _ = _stride2_last_dim((nnz, hk, d))
        lead = (pos_ids,) if api == "apply_rope_pos_ids" else (indptr, offsets)
        return lambda: getattr(rope, api)(q, k, *lead), []

    nnz = 2
    q, k, v = (_randn((nnz, h, d), dtype, gen) for h in (hq, hk, hk))
    halves = [
        q[..., :rope_dim],
        k[..., :rope_dim],
        q[..., rope_dim:],
        k[..., rope_dim:],
    ]
    cache = _cos_sin_cache(rope_dim, 64)
    pos_ids = torch.tensor([3, 20], dtype=torch.int32, device="cuda")
    if api == "rope_quantize_fp8":
        if which == "k_nope":
            halves[3], _ = _stride2_last_dim((nnz, hk, d - rope_dim))
            return lambda: rope.rope_quantize_fp8(*halves, cache, pos_ids), []
        q_rope_out, storage = _stride2_last_dim((nnz, hq, rope_dim), _F8)
        return lambda: rope.rope_quantize_fp8(
            *halves, cache, pos_ids, q_rope_out=q_rope_out
        ), [storage]
    # rope_quantize_fp8_append_paged_kv_cache, q_nope_out
    meta = _paged_metadata([3, 20], [1, 1], 0, num_pages=4)
    caches = [
        torch.zeros(4, _PAGE, hk, d, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]
    q_nope_out, storage = _stride2_last_dim((nnz, hq, d - rope_dim), _F8)
    return lambda: rope.rope_quantize_fp8_append_paged_kv_cache(
        *halves,
        v,
        cache,
        pos_ids,
        tuple(c.view(_F8) for c in caches),
        meta.kv_indices,
        meta.kv_indptr,
        meta.batch_indices,
        meta.positions,
        page_size=_PAGE,
        q_nope_out=q_nope_out,
    ), [storage, *caches]


_REJECT_CASES = [
    ("apply_rope_pos_ids", "q"),
    ("apply_rope", "k"),
    ("apply_rope_with_cos_sin_cache", "query"),
    ("rope_quantize_fp8", "k_nope"),
    ("rope_quantize_fp8", "q_rope_out"),
    ("rope_quantize_fp8_append_paged_kv_cache", "q_nope_out"),
]


@pytest.mark.parametrize(
    "api,which", _REJECT_CASES, ids=[f"{a}-{w}" for a, w in _REJECT_CASES]
)
def test_rope_strided_last_dim_rejected(api, which):
    """The kernels address the feature dimension as contiguous: a view with a
    strided last dimension is rejected before launch, and a rejected output
    buffer keeps its contents."""
    gen = torch.Generator(device="cuda").manual_seed(3605)
    call, guarded = _rejected_call(api, which, gen)
    before = [t.clone() for t in guarded]
    with pytest.raises(RuntimeError, match="contiguous"):
        call()
    torch.cuda.synchronize()
    for t, b in zip(guarded, before, strict=True):
        _assert_bitwise(t, b, "buffer written by a rejected call")


# ---------------------------------------------------------------------------
# Paged KV append
# ---------------------------------------------------------------------------


def _paged_metadata(kv_lens_before, append_lens, num_padding, num_pages, seed=3605):
    """What ``get_batch_indices_positions`` produces for appending
    ``append_lens[b]`` tokens to request ``b`` (padding tokens get batch index
    -1, as in a CUDA-graph padded batch), on a shuffled page table that leaves
    some pages of the pool unused. Built on the host, with the (page, slot)
    each real token must land in."""
    batch_indices, positions, kv_indptr = [], [], [0]
    for b, (before, n) in enumerate(zip(kv_lens_before, append_lens, strict=True)):
        batch_indices += [b] * n
        positions += range(before, before + n)
        kv_indptr.append(kv_indptr[-1] + -(-(before + n) // _PAGE))
    batch_indices += [-1] * num_padding
    positions += [0] * num_padding
    assert kv_indptr[-1] <= num_pages
    gen = torch.Generator().manual_seed(seed)
    kv_indices = torch.randperm(num_pages, generator=gen)[: kv_indptr[-1]].tolist()
    real = [t for t, b in enumerate(batch_indices) if b >= 0]
    pages = [
        kv_indices[kv_indptr[batch_indices[t]] + positions[t] // _PAGE] for t in real
    ]
    entries = [positions[t] % _PAGE for t in real]

    def cuda(v, dtype=torch.int32):
        return torch.tensor(v, dtype=dtype, device="cuda")

    return SimpleNamespace(
        batch_indices=cuda(batch_indices),
        positions=cuda(positions),
        kv_indptr=cuda(kv_indptr),
        kv_indices=cuda(kv_indices),
        real=cuda(real, torch.int64),
        pages=cuda(pages, torch.int64),
        entries=cuda(entries, torch.int64),
    )


def _paged_cache(
    arch, kv_layout, combined, num_pages, num_kv_heads, rope_dim, no_rope_dim
):
    """uint8 storages for the cache and a function mapping storages shaped like
    them to the two cache views the API takes: (k, v) or (ckv, kpe)."""
    if arch == "mla":
        if combined:  # one [ckv | kpe] row per slot, as MLA engines store it
            shapes = [(num_pages, _PAGE, no_rope_dim + rope_dim)]

            def views(s):
                return s[0][..., :no_rope_dim], s[0][..., no_rope_dim:]

        else:
            shapes = [(num_pages, _PAGE, no_rope_dim), (num_pages, _PAGE, rope_dim)]

            def views(s):
                return s[0], s[1]

    else:
        head_dim = rope_dim + no_rope_dim
        if kv_layout == "NHD":
            page = (_PAGE, num_kv_heads, head_dim)
        else:
            page = (num_kv_heads, _PAGE, head_dim)
        if combined:  # (num_pages, 2, ...) with K and V interleaved per page
            shapes = [(num_pages, 2, *page)]

            def views(s):
                return s[0][:, 0], s[0][:, 1]

        else:
            shapes = [(num_pages, *page)] * 2

            def views(s):
                return s[0], s[1]

    storages = [torch.empty(s, dtype=torch.uint8, device="cuda") for s in shapes]
    return storages, views


def _slots(cache, hnd, pages, entries):
    """The (page, slot) entries of a cache view, one row per token."""
    return cache[pages, :, entries] if hnd else cache[pages, entries]


@pytest.mark.parametrize(
    "arch,kv_layout",
    [("gqa", "NHD"), ("gqa", "HND"), ("mla", "NHD")],
    ids=["gqa-NHD", "gqa-HND", "mla"],
)
def test_rope_paged_append_contract(arch, kv_layout):
    """Of a fully poisoned cache, exactly the (page, slot) of each real token is
    written, with its rotated, scaled and FP8-rounded K (and V); padding tokens
    (batch index -1) write nothing. A combined cache (K and V, or ckv and kpe,
    in one tensor) gets bitwise the same slots."""
    mla, hnd = arch == "mla", kv_layout == "HND"
    hq, hk, rope_dim, no_rope_dim = (16, 1, 64, 512) if mla else (8, 2, 64, 64)
    head_dim = rope_dim + no_rope_dim
    num_pages = 9
    # A decode token, a chunk across a page boundary, a token in the last slot
    # of a page and a fresh prefill, then two padding tokens.
    meta = _paged_metadata([4, 13, 31, 0], [1, 5, 1, 3], 2, num_pages)
    nnz = meta.batch_indices.numel()
    gen = torch.Generator(device="cuda").manual_seed(3605)
    dtype = torch.bfloat16
    q = _randn((nnz, hq, head_dim), dtype, gen)
    k = _randn((nnz, head_dim) if mla else (nnz, hk, head_dim), dtype, gen)
    v = None if mla else _randn((nnz, hk, head_dim), dtype, gen)
    inputs_before = [t.clone() for t in (q, k, v) if t is not None]
    cache = _cos_sin_cache(rope_dim, 64)
    is_neox = kv_layout == "NHD" and not mla
    scale_q, scale_kv = 0.5, 2.0
    q_ref = _rope_reference(q, meta.positions, cache, not is_neox) * scale_q
    k_ref = _rope_reference(
        k.view(nnz, -1, head_dim), meta.positions, cache, not is_neox
    )
    k_ref = k_ref * scale_kv
    real = meta.real

    results = {}
    for combined in (False, True):
        for poison in _FP8_POISONS:
            storages, views = _paged_cache(
                arch, kv_layout, combined, num_pages, hk, rope_dim, no_rope_dim
            )
            for s in storages:
                s.fill_(poison)
            q_store = torch.full(
                (nnz, hq, head_dim), poison, dtype=torch.uint8, device="cuda"
            )
            q_out = q_store.view(_F8)
            q_rope_out, q_nope_out = q_out[..., :rope_dim], q_out[..., rope_dim:]
            returned = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q[..., :rope_dim],
                k[..., :rope_dim],
                q[..., rope_dim:],
                k[..., rope_dim:],
                v,
                cache,
                meta.positions,
                tuple(c.view(_F8) for c in views(storages)),
                meta.kv_indices,
                meta.kv_indptr,
                meta.batch_indices,
                meta.positions,
                is_neox=is_neox,
                quant_scale_q=scale_q,
                quant_scale_kv=scale_kv,
                page_size=_PAGE,
                kv_layout=kv_layout,
                q_rope_out=q_rope_out,
                q_nope_out=q_nope_out,
            )
            torch.cuda.synchronize()
            tag = f"{'combined' if combined else 'separate'} poison={poison:#x}"
            assert returned[0] is q_rope_out and returned[1] is q_nope_out

            marks = [torch.zeros_like(s) for s in storages]
            for m in views(marks):
                if hnd:
                    m[meta.pages, :, meta.entries] = 1
                else:
                    m[meta.pages, meta.entries] = 1
            for s, m in zip(storages, marks, strict=True):
                stray = s[m == 0] != poison
                assert not stray.any(), (
                    f"{tag}: {int(stray.sum())} bytes outside the target slots changed"
                )

            got = [_slots(c, hnd, meta.pages, meta.entries) for c in views(storages)]
            if mla:
                ckv_ref, kpe_ref = k_ref[real, 0, rope_dim:], k_ref[real, 0, :rope_dim]
                _assert_fp8_close(got[0].view(_F8), ckv_ref, f"{tag} ckv")
                _assert_fp8_close(got[1].view(_F8), kpe_ref, f"{tag} kpe")
            else:
                _assert_fp8_close(got[0].view(_F8), k_ref[real], f"{tag} k")
                _assert_fp8_close(
                    got[1].view(_F8), v[real].float() * scale_kv, f"{tag} v"
                )
            # The Q rows of padding tokens are left alone by the kernel today,
            # but that is not documented, so they are not checked here.
            _assert_fp8_close(q_store[real].view(_F8), q_ref[real], f"{tag} q")
            results[(combined, poison)] = [*got, q_store[real]]

    for t, b in zip(
        [t for t in (q, k, v) if t is not None], inputs_before, strict=True
    ):
        _assert_bitwise(t, b, "input modified")
    first = results[(False, _FP8_POISONS[0])]
    for key, value in results.items():
        for i, (a, b) in enumerate(zip(value, first, strict=True)):
            _assert_bitwise(
                a, b, f"target bytes differ: combined, poison={key}, part {i}"
            )


# ---------------------------------------------------------------------------
# Position index dtype
# ---------------------------------------------------------------------------

_POS_IDS_APIS = [
    "apply_rope",  # indptr / offsets
    "apply_rope_pos_ids",
    "apply_llama31_rope_pos_ids",
    "apply_rope_with_cos_sin_cache",
    "rope_quantize_fp8",
    "rope_quantize_fp8_append_paged_kv_cache",
]


def _pos_ids_case(api, idtype, gen, pos):
    """One call of ``api`` with every position / index tensor in ``idtype``."""
    dtype, nnz, rope_dim = torch.bfloat16, pos.numel(), 64
    cuda_pos = pos.to(device="cuda", dtype=idtype)
    cache = _cos_sin_cache(rope_dim, 4096)
    if api in ("rope_quantize_fp8", "rope_quantize_fp8_append_paged_kv_cache"):
        q, k = _randn((nnz, 4, 128), dtype, gen), _randn((nnz, 2, 128), dtype, gen)
        halves = (
            q[..., :rope_dim],
            k[..., :rope_dim],
            q[..., rope_dim:],
            k[..., rope_dim:],
        )
        if api == "rope_quantize_fp8":
            return flashinfer.rope.rope_quantize_fp8(
                *halves, cache, cuda_pos, quantize_dtype=_F8
            )
        # A decode step: one token per request, each on its own page.
        meta = _paged_metadata([7, 0, 20, 33], [1] * nnz, 0, num_pages=8)
        caches = [
            torch.full((8, _PAGE, 2, 128), 0x7F, dtype=torch.uint8, device="cuda")
            for _ in range(2)
        ]
        names = ("kv_indices", "kv_indptr", "batch_indices", "positions")
        q_outs = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            *halves,
            _randn((nnz, 2, 128), dtype, gen),
            cache,
            cuda_pos,
            tuple(c.view(_F8) for c in caches),
            *(getattr(meta, n).to(idtype) for n in names),
            page_size=_PAGE,
        )
        return (*q_outs, *caches)
    q, k = _randn((nnz, 4, 128), dtype, gen), _randn((nnz, 2, 128), dtype, gen)
    segments = _segments(idtype)
    if api == "apply_rope":
        segments = (segments[0], segments[1] * 10, segments[2])
    else:
        segments = (None, None, cuda_pos)
    return _call_fused(
        api, q, k, 128, False, False, segments, _cos_sin_cache(128, 4096)
    )


@pytest.mark.parametrize("api", _POS_IDS_APIS)
def test_rope_pos_ids_dtype(api):
    """int32 and int64 position tensors (pos_ids, or indptr / offsets, and the
    page metadata of the fused append) give bitwise identical outputs. The
    positions are distinct and nonzero, so a kernel that reads an int64 tensor
    as int32 sees different ones."""
    nnz = 4 if api == "rope_quantize_fp8_append_paged_kv_cache" else _INDPTR[-1]
    pos = torch.randperm(4000, generator=torch.Generator().manual_seed(3605))[:nnz] + 1
    results = {}
    for idtype in (torch.int32, torch.int64):
        gen = torch.Generator(device="cuda").manual_seed(3605)
        results[idtype] = _pos_ids_case(api, idtype, gen, pos)
    torch.cuda.synchronize()
    for i, (a, b) in enumerate(
        zip(results[torch.int32], results[torch.int64], strict=True)
    ):
        _assert_bitwise(b, a, f"output {i}: int64 vs int32")


# ---------------------------------------------------------------------------
# rope_quantize_fp8 backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,num_tokens",
    [(torch.bfloat16, 1), (torch.bfloat16, 37), (torch.float16, 130)],
    ids=["bf16-t1", "bf16-t37", "fp16-t130"],
)
def test_rope_backend_agree(dtype, num_tokens, monkeypatch):
    """``backend="cuda"`` and ``backend="cutile"`` agree on their common
    ground: interleaved RoPE, a 2-D (MLA) key, SM90+.

    Both rotate in fp32 with the same cos/sin cache entries and round once to
    e4m3 with round-to-nearest-even (``cvt.rn.satfinite`` in the CUDA kernel).
    The no-RoPE part is a power-of-two scale and that conversion, so it must
    match bitwise. In the RoPE part the compilers may contract
    ``x * cos - y * sin`` into FMAs differently; that moves the fp32 value by
    about an fp32 ulp, which changes the e4m3 rounding only within that
    distance of a rounding midpoint. So a rotated element may differ by one
    e4m3 step, and only rarely (at most 1% of them, or one; a different
    rounding mode or a wrong scale would move most elements)."""
    from flashinfer.cutile.cutile_common import is_cuda_tile_available

    if get_compute_capability(torch.device("cuda")) < (9, 0):
        pytest.skip("backend='cutile' needs SM90+")
    if not is_cuda_tile_available():
        pytest.skip("cuda.tile is not available")
    import flashinfer.quantization.kernels.cutile.rope_quantize_fp8_cutile as cutile_rope

    # One configuration per shape: autotuning would compile 15 variants.
    monkeypatch.setattr(cutile_rope, "_AUTOTUNE_DISABLED", True)
    hq, rope_dim, no_rope_dim = 8, 64, 512
    d = rope_dim + no_rope_dim
    gen = torch.Generator(device="cuda").manual_seed(3605 + num_tokens)
    q = _randn((num_tokens, hq, d), dtype, gen)
    k = _randn((num_tokens, d), dtype, gen)
    pos_ids = torch.randint(0, 4096, (num_tokens,), generator=gen, device="cuda").int()
    cache = _cos_sin_cache(rope_dim, 4096)
    outs = {}
    for backend in ("cuda", "cutile"):
        q_out = torch.full((num_tokens, hq, d), 0x7F, dtype=torch.uint8, device="cuda")
        k_out = torch.full((num_tokens, d), 0x7F, dtype=torch.uint8, device="cuda")
        q_out, k_out = q_out.view(_F8), k_out.view(_F8)
        flashinfer.rope.rope_quantize_fp8(
            q[..., :rope_dim],
            k[..., :rope_dim],
            q[..., rope_dim:],
            k[..., rope_dim:],
            cache,
            pos_ids,
            is_neox=False,
            quantize_dtype=_F8,
            quant_scale_q=0.5,
            quant_scale_kv=2.0,
            q_rope_out=q_out[..., :rope_dim],
            k_rope_out=k_out[..., :rope_dim],
            q_nope_out=q_out[..., rope_dim:],
            k_nope_out=k_out[..., rope_dim:],
            backend=backend,
        )
        outs[backend] = (q_out, k_out)
    torch.cuda.synchronize()
    for name, a, b in zip("qk", outs["cuda"], outs["cutile"], strict=True):
        assert torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()
        _assert_bitwise(b[..., rope_dim:], a[..., rope_dim:], f"{name} no-RoPE part")
        steps = _fp8_steps(a[..., :rope_dim], b[..., :rope_dim])
        assert int(steps.max()) <= 1, (
            f"{name}: up to {int(steps.max())} FP8 steps apart"
        )
        differ = int((steps > 0).sum())
        assert differ <= max(1, steps.numel() // 100), (
            f"{name}: {differ} of {steps.numel()} RoPE elements differ"
        )


# ---------------------------------------------------------------------------
# PDL ordering. With enable_pdl, rope_quantize_fp8 / mla_rope_quantize_fp8
# (RopeQuantizeKernel) and rope_quantize_fp8_append_paged_kv_cache
# (RopeQuantizeAppendPagedKVCacheKernel, GQA and MLA caches) may start while the
# previous kernel still runs; they must not read what it is still writing (RAW)
# or write what it is still reading (WAR) before griddepcontrol.wait. The
# producer and probe are the ones of tests/utils/test_norm.py.
# ---------------------------------------------------------------------------


class _Arena:
    """Tensors carved out of one contiguous byte buffer, so that a single copy
    job of the producer covers all of them."""

    def __init__(self, specs):
        self.ranges, end = {}, 0
        for name, (shape, dtype) in specs.items():
            nbytes = math.prod(shape) * dtype.itemsize
            self.ranges[name] = (end, end + nbytes)
            end += -(-nbytes // 256) * 256
        self.storage = torch.empty(end, dtype=torch.uint8, device="cuda")
        self.views = {
            name: self.storage[lo:hi].view(dtype).view(shape)
            for (name, (shape, dtype)), (lo, hi) in zip(
                specs.items(), self.ranges.values(), strict=True
            )
        }

    def assert_same(self, expected, what, actual=None):
        actual = self.storage if actual is None else actual
        bad = actual != expected
        if bad.any():
            where = [n for n, (lo, hi) in self.ranges.items() if bad[lo:hi].any()]
            raise AssertionError(f"{what}: {int(bad.sum())} bytes differ, in {where}")


_PDL_APIS = ["rope_quantize_fp8", "mla_rope_quantize_fp8", "append_gqa", "append_mla"]


class _PdlRopeCase:
    """Storage for one race case. ``inputs`` holds everything the kernel reads
    that a previous kernel could be producing: activations and the per-step
    metadata (positions, batch indices, page table). The cos/sin cache is a
    static table, like the norm weights, and stays out of the contract.
    ``outputs`` holds everything the kernel writes."""

    def __init__(self, api):
        self.api = api
        mla = "mla" in api
        self.is_neox = not mla
        hq, hk = (16, 1) if mla else (8, 2)
        self.rope_dim, no_rope_dim = 64, (512 if mla else 64)
        d = self.rope_dim + no_rope_dim
        bf16, i32 = torch.bfloat16, torch.int32
        if api.startswith("append"):
            # A decode step: one token per request, so no two tokens share a page.
            kv_before = torch.randint(
                0, 48, (64,), generator=torch.Generator().manual_seed(_PDL_SEED)
            ).tolist()
            meta = _paged_metadata(kv_before, [1] * 64, 0, num_pages=200)
            nnz = 64
        else:
            nnz = 128
        k_shape = (nnz, d) if mla else (nnz, hk, d)
        inputs = {"q": ((nnz, hq, d), bf16), "k": (k_shape, bf16)}
        inputs["pos_ids"] = ((nnz,), i32)
        outputs = {"q": ((nnz, hq, d), _F8)}
        if not api.startswith("append"):
            outputs["k"] = (k_shape, _F8)
        elif mla:
            outputs["ckv"] = ((200, _PAGE, no_rope_dim), _F8)
            outputs["kpe"] = ((200, _PAGE, self.rope_dim), _F8)
        else:
            inputs["v"] = (k_shape, bf16)
            outputs["k_cache"] = outputs["v_cache"] = ((200, _PAGE, hk, d), _F8)
        if api.startswith("append"):
            for name in ("batch_indices", "positions", "kv_indptr", "kv_indices"):
                inputs[name] = (tuple(getattr(meta, name).shape), i32)
        self.inputs, self.outputs = _Arena(inputs), _Arena(outputs)

        gen = torch.Generator(device="cuda").manual_seed(_PDL_SEED)
        views = self.inputs.views
        for name in ("q", "k", "v"):
            if name in views:
                views[name].copy_(
                    torch.randn(views[name].shape, generator=gen, device="cuda")
                )
        if api.startswith("append"):
            for name in ("batch_indices", "positions", "kv_indptr", "kv_indices"):
                views[name].copy_(getattr(meta, name))
            views["pos_ids"].copy_(meta.positions)
        else:
            views["pos_ids"].copy_(
                torch.randperm(4096, generator=gen, device="cuda")[:nnz]
            )
        self.clean = self.inputs.storage.clone()
        self.cos_sin_cache = _cos_sin_cache(self.rope_dim, 4096)

    def call(self, enable_pdl):
        i, o, r = self.inputs.views, self.outputs.views, self.rope_dim
        kwargs = dict(
            is_neox=self.is_neox,
            quant_scale_q=0.5,
            quant_scale_kv=2.0,
            q_rope_out=o["q"][..., :r],
            q_nope_out=o["q"][..., r:],
            enable_pdl=enable_pdl,
        )
        halves = (i["q"][..., :r], i["k"][..., :r], i["q"][..., r:], i["k"][..., r:])
        if self.api.startswith("append"):
            caches = (
                (o["ckv"], o["kpe"]) if "ckv" in o else (o["k_cache"], o["v_cache"])
            )
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                *halves,
                i.get("v"),
                self.cos_sin_cache,
                i["pos_ids"],
                caches,
                i["kv_indices"],
                i["kv_indptr"],
                i["batch_indices"],
                i["positions"],
                page_size=_PAGE,
                **kwargs,
            )
        else:
            getattr(flashinfer.rope, self.api)(
                *halves,
                self.cos_sin_cache,
                i["pos_ids"],
                k_rope_out=o["k"][..., :r],
                k_nope_out=o["k"][..., r:],
                **kwargs,
            )

    def reset(self):
        """Clean inputs, poisoned outputs."""
        self.inputs.storage.copy_(self.clean)
        self.outputs.storage.fill_(0x7F)

    def poison_inputs(self):
        # NaN activations; zero indices keep a premature read in bounds.
        for t in self.inputs.views.values():
            t.fill_(float("nan") if t.is_floating_point() else 0)

    def reference(self, enable_pdl):
        """The same call on synchronized inputs: the bitwise target."""
        self.reset()
        torch.cuda.synchronize()
        self.call(enable_pdl)
        torch.cuda.synchronize()
        return self.outputs.storage.clone()


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_rope_pdl_probe_sees_overlap(mode):
    """Calibration for the race tests below: they can only fail where a PDL
    dependent actually starts while the producer spins. Like them, this skips
    with the reason on setups where it never does."""
    _skip_unless_pdl_overlaps(mode)


@pytest.mark.parametrize("api", _PDL_APIS)
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_rope_pdl_race_waits_before_reading(api, mode):
    """RAW: inputs and metadata written by a still-running producer must not be
    read early."""
    _skip_unless_pdl_overlaps(mode)
    module = _pdl_race_module()
    c = _PdlRopeCase(api)
    expected = c.reference(enable_pdl=True)

    def fn():
        module.pdl_trigger_spin_copy([c.inputs.storage], [c.clean], _PDL_SPIN_CYCLES)
        c.call(enable_pdl=True)

    run = _pdl_launcher(mode, fn)
    for rep in range(_PDL_REPEATS):
        c.reset()
        c.poison_inputs()
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        c.outputs.assert_same(expected, f"{api} {mode} seed={_PDL_SEED} repeat={rep}")


@pytest.mark.parametrize("api", _PDL_APIS)
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_rope_pdl_race_waits_before_writing(api, mode):
    """WAR: outputs, including every page of the cache, that a still-running
    producer reads must not be overwritten early."""
    _skip_unless_pdl_overlaps(mode)
    module = _pdl_race_module()
    c = _PdlRopeCase(api)
    expected = c.reference(enable_pdl=True)
    snapshot = torch.empty_like(c.outputs.storage)

    def fn():
        module.pdl_trigger_spin_copy([snapshot], [c.outputs.storage], _PDL_SPIN_CYCLES)
        c.call(enable_pdl=True)

    run = _pdl_launcher(mode, fn)
    for rep in range(_PDL_REPEATS):
        c.reset()
        before = c.outputs.storage.clone()
        snapshot.zero_()
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        tag = f"{api} {mode} seed={_PDL_SEED} repeat={rep}"
        c.outputs.assert_same(before, f"{tag}: overwritten early", actual=snapshot)
        c.outputs.assert_same(expected, tag)


@pytest.mark.parametrize("api", _PDL_APIS)
def test_rope_pdl_matches_plain_launch(api):
    """The PDL and non-PDL launches give bitwise identical results."""
    _skip_unless_pdl()
    c = _PdlRopeCase(api)
    with_pdl = c.reference(enable_pdl=True)
    c.outputs.assert_same(with_pdl, api, actual=c.reference(enable_pdl=False))
