"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Dtype contract of ``merge_state``, ``merge_state_in_place`` and ``merge_states``.

The kernels are instantiated for the dtype of the first attention output and
read the logsumexp values as float32 and the mask as one byte per sequence.
A tensor of any other dtype used to be reinterpreted bit-for-bit, or (for the
in-place ``s``) converted into a temporary that received the merged logsumexp
instead of the caller's tensor. These cases must raise, and must leave the
in-place state untouched; the read-only logsumexp inputs are still converted.

The Python wrappers convert the read-only logsumexp inputs and allocate the
outputs themselves, so the FFI checks behind them are also exercised directly
through ``get_cascade_module()``.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.cascade import get_cascade_module

_SEQ_LEN = 9
_NUM_HEADS = 4
_HEAD_DIM = 128
_DEVICE = "cuda:0"


def _make_state(v_dtype, seed):
    gen = torch.Generator(device=_DEVICE).manual_seed(seed)
    v = torch.randn(_SEQ_LEN, _NUM_HEADS, _HEAD_DIM, generator=gen, device=_DEVICE).to(
        v_dtype
    )
    s = torch.randn(_SEQ_LEN, _NUM_HEADS, generator=gen, device=_DEVICE)
    return v, s


def _merge_reference(v_a, s_a, v_b, s_b):
    """Merge in fp64; the logsumexp values are base 2."""
    s_a64 = s_a.double() * math.log(2.0)
    s_b64 = s_b.double() * math.log(2.0)
    s_merged = torch.logaddexp(s_a64, s_b64)
    w_a = torch.exp(s_a64 - s_merged).unsqueeze(-1)
    w_b = torch.exp(s_b64 - s_merged).unsqueeze(-1)
    v_merged = v_a.double() * w_a + v_b.double() * w_b
    return v_merged, s_merged / math.log(2.0)


def _assert_merged(v, s, v_ref, s_ref):
    torch.testing.assert_close(v.double(), v_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(s.double(), s_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("s_dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_merge_state_in_place_rejects_non_float32_s(s_dtype):
    v, s = _make_state(torch.float16, 0)
    v_other, s_other = _make_state(torch.float16, 1)
    s = s.to(s_dtype)
    v_before, s_before = v.clone(), s.clone()

    with pytest.raises(ValueError, match="s must be float32"):
        flashinfer.merge_state_in_place(v, s, v_other, s_other)

    assert torch.equal(v, v_before)
    assert torch.equal(s, s_before)


@pytest.mark.parametrize("v_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s_other_dtype", [torch.float16, torch.float64])
def test_merge_state_in_place_converts_s_other(v_dtype, s_other_dtype):
    v, s = _make_state(v_dtype, 0)
    v_other, s_other = _make_state(v_dtype, 1)
    s_other = s_other.to(s_other_dtype)
    v_ref, s_ref = _merge_reference(v, s, v_other, s_other)

    flashinfer.merge_state_in_place(v, s, v_other, s_other)

    _assert_merged(v, s, v_ref, s_ref)


@pytest.mark.parametrize("v_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s_dtype", [torch.float16, torch.float64])
def test_merge_state_converts_s(v_dtype, s_dtype):
    v_a, s_a = _make_state(v_dtype, 0)
    v_b, s_b = _make_state(v_dtype, 1)
    s_a, s_b = s_a.to(s_dtype), s_b.to(s_dtype)
    v_ref, s_ref = _merge_reference(v_a, s_a, v_b, s_b)

    v_merged, s_merged = flashinfer.merge_state(v_a, s_a, v_b, s_b)

    assert v_merged.dtype == v_dtype
    assert s_merged.dtype == torch.float32
    _assert_merged(v_merged, s_merged, v_ref, s_ref)


@pytest.mark.parametrize(
    "v_dtype, v_other_dtype",
    [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.float32),
    ],
)
def test_merge_state_rejects_mismatched_v_dtypes(v_dtype, v_other_dtype):
    v, s = _make_state(v_dtype, 0)
    v_other, s_other = _make_state(v_other_dtype, 1)
    v_before, s_before = v.clone(), s.clone()

    with pytest.raises(RuntimeError, match="v_b dtype must match v_a dtype"):
        flashinfer.merge_state(v, s, v_other, s_other)

    with pytest.raises(RuntimeError, match="v_other dtype must match v dtype"):
        flashinfer.merge_state_in_place(v, s, v_other, s_other)

    assert torch.equal(v, v_before)
    assert torch.equal(s, s_before)


@pytest.mark.parametrize("mask_dtype", [torch.int32, torch.int64, torch.float32])
def test_merge_state_in_place_rejects_wide_mask(mask_dtype):
    v, s = _make_state(torch.float16, 0)
    v_other, s_other = _make_state(torch.float16, 1)
    mask = torch.ones(_SEQ_LEN, dtype=mask_dtype, device=_DEVICE)
    v_before, s_before = v.clone(), s.clone()

    with pytest.raises(RuntimeError, match="mask must be dl_bool or dl_uint8"):
        flashinfer.merge_state_in_place(v, s, v_other, s_other, mask=mask)

    assert torch.equal(v, v_before)
    assert torch.equal(s, s_before)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.uint8])
def test_merge_state_in_place_byte_mask(mask_dtype):
    v, s = _make_state(torch.float16, 0)
    v_other, s_other = _make_state(torch.float16, 1)
    # Odd rows are skipped; 1 and 3 are the rows a 4-byte mask element would hide.
    keep = torch.arange(_SEQ_LEN, device=_DEVICE) % 2 == 0
    v_before, s_before = v.clone(), s.clone()
    v_ref, s_ref = _merge_reference(v, s, v_other, s_other)

    flashinfer.merge_state_in_place(v, s, v_other, s_other, mask=keep.to(mask_dtype))

    _assert_merged(v[keep], s[keep], v_ref[keep], s_ref[keep])
    assert torch.equal(v[~keep], v_before[~keep])
    assert torch.equal(s[~keep], s_before[~keep])


def _ffi_dtype_error(name, anchor=None):
    """Message of the FFI check that rejects ``name``."""
    if anchor is not None:
        return f"{name} dtype must match {anchor} dtype"
    return rf"Inconsistency of Tensor type: {name}(?!\w)"


def _assert_unchanged(args, before):
    for name, tensor in args.items():
        assert torch.equal(tensor, before[name]), f"{name} was modified"


@pytest.mark.parametrize(
    "name, dtype, anchor",
    [
        ("s_a", torch.float16, None),
        ("s_a", torch.float64, None),
        ("s_b", torch.float16, None),
        ("s_b", torch.float64, None),
        ("s_merged", torch.float16, None),
        ("s_merged", torch.float64, None),
        ("v_merged", torch.bfloat16, "v_a"),
        ("v_merged", torch.float32, "v_a"),
    ],
)
def test_ffi_merge_state_rejects_dtype(name, dtype, anchor):
    v_a, s_a = _make_state(torch.float16, 0)
    v_b, s_b = _make_state(torch.float16, 1)
    args = {
        "v_a": v_a,
        "s_a": s_a,
        "v_b": v_b,
        "s_b": s_b,
        "v_merged": torch.zeros_like(v_a),
        "s_merged": torch.zeros_like(s_a),
    }
    args[name] = args[name].to(dtype)
    before = {k: t.clone() for k, t in args.items()}

    with pytest.raises(RuntimeError, match=_ffi_dtype_error(name, anchor)):
        get_cascade_module().merge_state(*args.values())

    _assert_unchanged(args, before)


@pytest.mark.parametrize("name", ["s", "s_other"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_ffi_merge_state_in_place_rejects_non_float32_lse(name, dtype):
    v, s = _make_state(torch.float16, 0)
    v_other, s_other = _make_state(torch.float16, 1)
    args = {"v": v, "s": s, "v_other": v_other, "s_other": s_other}
    args[name] = args[name].to(dtype)
    before = {k: t.clone() for k, t in args.items()}

    with pytest.raises(RuntimeError, match=_ffi_dtype_error(name)):
        get_cascade_module().merge_state_in_place(*args.values(), None)

    _assert_unchanged(args, before)


@pytest.mark.parametrize(
    "name, dtype, anchor",
    [
        ("s", torch.float16, None),
        ("s", torch.float64, None),
        ("s_merged", torch.float16, None),
        ("s_merged", torch.float64, None),
        ("v_merged", torch.bfloat16, "v"),
        ("v_merged", torch.float32, "v"),
    ],
)
def test_ffi_merge_states_rejects_dtype(name, dtype, anchor):
    num_states = 3
    gen = torch.Generator(device=_DEVICE).manual_seed(0)
    v = torch.randn(
        _SEQ_LEN, num_states, _NUM_HEADS, _HEAD_DIM, generator=gen, device=_DEVICE
    ).to(torch.float16)
    s = torch.randn(_SEQ_LEN, num_states, _NUM_HEADS, generator=gen, device=_DEVICE)
    args = {
        "v": v,
        "s": s,
        "v_merged": torch.zeros_like(v[:, 0]),
        "s_merged": torch.zeros_like(s[:, 0]),
    }
    args[name] = args[name].to(dtype)
    before = {k: t.clone() for k, t in args.items()}

    with pytest.raises(RuntimeError, match=_ffi_dtype_error(name, anchor)):
        get_cascade_module().merge_states(*args.values())

    _assert_unchanged(args, before)
