# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake SM120 DeepSeek-V4 NVFP4 fused GPT-J RoPE + quantize + paged insert writers.

Every cache case is byte exact: the fused writer's cache must equal the cache
produced by ``nvfp4_quantize_append_sparse_mla_cache`` on the torch fp32
reference's BF16 roped rows (canary ``0xA5`` everywhere else untouched), and
``q_out`` must equal the torch reference bitwise. Set
``FLASHINFER_CAKE_ROPE_INSERT_Q_OUT_ULPS=1`` to accept a documented 1-ulp
BF16 disagreement on ``q_out`` (off by default). The GPU cases skip off
SM12x and while the tree only holds the placeholder manifest.
"""

import os

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    cake_dsv4_nvfp4_kv_rope_quantize_insert,
    cake_dsv4_nvfp4_rope_insert_format_info,
    cake_dsv4_nvfp4_rope_quantize_insert,
    nvfp4_quantize_append_sparse_mla_cache,
)
from flashinfer.mla._sparse_mla_sm120._cake_dsv4_nvfp4 import (
    cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes,
)
from flashinfer.utils import is_sm12x_supported
from tests.attention.sparse_mla_test_utils import (
    _D_NOPE,
    _D_ROPE,
    _dequantize_nvfp4_cache,
    _dequantize_nvfp4_query,
    _reference_sparse_attention,
)

_D = _D_NOPE + _D_ROPE
_BYTES = 384
_CANARY = 0xA5
_MAX_POS = 4096
_OUT_TOL = dict(atol=5e-2, rtol=5e-2)
# Documented fallback: bf16 ulps of disagreement accepted on q_out (0 = bitwise, the default).
_Q_OUT_ULPS = int(os.environ.get("FLASHINFER_CAKE_ROPE_INSERT_Q_OUT_ULPS", "0"))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _require_sm120() -> None:
    _require_cuda()
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("Cake SM120 NVFP4 RoPE insert requires SM12x")
    if not cake_dsv4_nvfp4_rope_insert_format_info()["kernels_available"]:
        pytest.skip(
            "Cake SM120 NVFP4 RoPE-insert kernels are not exported into this tree"
        )


# ----------------------------------------------------------------------------- torch reference


def _fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Bit-exact fp32 ``fma(a, b, c)`` (one rounding) emulated in float64.

    ``a * b`` is exact in float64; the float64 sum carries an exactly recoverable error term,
    which decides the direction when the sum lands exactly on an fp32 rounding midpoint.
    """

    p = a.double() * b.double()
    s = p + c.double()
    bb = s - p
    err = (p - (s - bb)) + (c.double() - bb)
    r = s.float()
    rd = r.double()
    _mant, e2 = torch.frexp(s)
    tiny = s.abs() < 2.0**-126
    shift = torch.where(tiny, torch.full_like(e2, 150), 25 - e2)
    scaled = torch.ldexp(s, shift)
    is_mid = (
        (rd != s)
        & (scaled == torch.round(scaled))
        & (torch.remainder(scaled.abs(), 2.0) == 1.0)
    )
    lo = torch.where(rd <= s, r, torch.nextafter(r, torch.full_like(r, float("-inf"))))
    hi = torch.where(rd >= s, r, torch.nextafter(r, torch.full_like(r, float("inf"))))
    fixed = torch.where(err > 0, hi, torch.where(err < 0, lo, r))
    return torch.where(is_mid, fixed, r)


def _rope_gptj_fp32(x: torch.Tensor, cos_sin_rows: torch.Tensor) -> torch.Tensor:
    """GPT-J RoPE of dims 448..511 of ``x[..., 512]`` with fp32 ``cos_sin_rows[..., 64]`` -> fp32.

    Both elements are one exact fp32 FMA over one rounded product -- ``fma(x_e, cos, -(x_o * sin))``
    and ``fma(x_e, sin, x_o * cos)`` -- the shapes the generated kernel emits (explicit FMA, so the
    assembler cannot re-contract them and every variant produces identical bits).
    """

    xf = x.float()
    cos = cos_sin_rows[..., :32].float()
    sin = cos_sin_rows[..., 32:].float()
    rope = xf[..., _D_NOPE:].reshape(*xf.shape[:-1], 32, 2)
    x_even, x_odd = rope[..., 0], rope[..., 1]
    even = _fma_f32(x_even, cos, -(x_odd * sin))
    odd = _fma_f32(x_even, sin, x_odd * cos)
    out = xf.clone()
    out[..., _D_NOPE:] = torch.stack((even, odd), dim=-1).reshape(
        *xf.shape[:-1], _D_ROPE
    )
    return out


def _cos_sin_cache(max_pos: int = _MAX_POS, theta: float = 1e4) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, _D_ROPE, 2, dtype=torch.float32, device="cuda") / _D_ROPE)
    )
    freqs = torch.outer(
        torch.arange(max_pos, dtype=torch.float32, device="cuda"), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def _reference_kv_rows(
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """BF16 roped rows the writer quantizes: cos/sin row ``pos // ratio * ratio``."""

    rows = positions // compress_ratio * compress_ratio
    return _rope_gptj_fp32(kv, cos_sin[rows]).to(torch.bfloat16)


def _inserted_slots(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    *,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """int64 ``[num_tokens]`` slots with ``-1`` for every row the writer skips (padding rows, non-boundary rows)."""

    slots = torch.full((num_tokens,), -1, dtype=torch.int64, device=slot_mapping.device)
    n_ins = slot_mapping.numel()
    s = slot_mapping.long()
    keep = (s >= 0) & ((positions[:n_ins] + 1) % compress_ratio == 0)
    slots[:n_ins] = torch.where(keep, s, torch.full_like(s, -1))
    return slots


def _reference_append(
    cache: torch.Tensor,
    kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    compress_ratio: int = 1,
) -> None:
    """The pre-fusion path: torch RoPE, then the append writer (skips -1 and out-of-range slots)."""

    rows = _reference_kv_rows(kv, positions, cos_sin, compress_ratio=compress_ratio)
    slots = _inserted_slots(
        slot_mapping, positions, kv.shape[0], compress_ratio=compress_ratio
    )
    nvfp4_quantize_append_sparse_mla_cache(rows, slots, cache)


def _reference_q_out(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    q_head_padded: int,
    *,
    apply_q_rope: bool = True,
) -> torch.Tensor:
    n, h, _ = q.shape
    out = torch.zeros(n, q_head_padded, _D, dtype=torch.bfloat16, device=q.device)
    if apply_q_rope:
        rows = cos_sin[positions].unsqueeze(1).expand(n, h, _D_ROPE)
        out[:, :h] = _rope_gptj_fp32(q, rows).to(torch.bfloat16)
    else:
        out[:, :h] = q
    return out


def _assert_q_out(q_out: torch.Tensor, expected: torch.Tensor) -> None:
    assert q_out.shape == expected.shape and q_out.dtype == torch.bfloat16
    if _Q_OUT_ULPS == 0:
        assert torch.equal(q_out, expected), (
            "q_out differs from the torch reference bitwise"
        )
        return
    got = q_out.view(torch.int16).int()
    want = expected.view(torch.int16).int()
    same_sign = (got < 0) == (want < 0)
    ulps = (got - want).abs()
    assert bool((same_sign & (ulps <= _Q_OUT_ULPS)).all()), (
        f"q_out differs from the torch reference by more than {_Q_OUT_ULPS} bf16 ulp"
    )


# ----------------------------------------------------------------------------- cache construction


class _Cache:
    """A canary-filled NVFP4 cache view over its own backing storage (3-D / HND / NHD / padded)."""

    def __init__(
        self, num_pages: int, page_size: int, layout: str = "HND", pad_bytes: int = 0
    ):
        self.num_pages, self.page_size, self.layout = num_pages, page_size, layout
        page_bytes = page_size * _BYTES
        stride = page_bytes + pad_bytes
        self.storage = torch.full(
            (num_pages * stride + 16,), _CANARY, dtype=torch.uint8, device="cuda"
        )
        base = (-self.storage.data_ptr()) % 16
        self.base = base
        if layout == "HND":
            size, strides = (
                (num_pages, 1, page_size, _BYTES),
                (stride, page_bytes, _BYTES, 1),
            )
        elif layout == "NHD":
            size, strides = (
                (num_pages, page_size, 1, _BYTES),
                (stride, _BYTES, _BYTES, 1),
            )
        elif layout == "3D":
            size, strides = (num_pages, page_size, _BYTES), (stride, _BYTES, 1)
        else:
            raise ValueError(layout)
        self.view = self.storage.as_strided(size, strides, base)

    def reset(self) -> None:
        self.storage.fill_(_CANARY)

    @property
    def total_slots(self) -> int:
        return self.num_pages * self.page_size


def _assert_same_cache(got: _Cache, want: _Cache) -> None:
    """Whole-storage comparison: addressed records byte for byte and the canary everywhere else."""

    assert torch.equal(got.storage, want.storage), (
        "fused writer cache differs from the reference append"
    )


def _slot_mapping(
    total_slots: int,
    num_insert: int,
    dtype: torch.dtype,
    *,
    negatives: int = 0,
    out_of_range: int = 0,
    generator: torch.Generator,
) -> torch.Tensor:
    """Unique valid slots with ``negatives`` entries set to -1 and ``out_of_range`` entries past the pool."""

    perm = torch.randperm(total_slots, generator=generator, device="cuda")[:num_insert]
    slots = perm.clone()
    if negatives:
        slots[:negatives] = -1
    if out_of_range:
        slots[negatives : negatives + out_of_range] = total_slots + torch.arange(
            out_of_range, device="cuda"
        )
    return slots.to(dtype)


def _inputs(num_tokens: int, num_heads: int, generator: torch.Generator):
    q = (
        torch.randn(
            num_tokens,
            num_heads,
            _D,
            dtype=torch.float32,
            generator=generator,
            device="cuda",
        )
        * 2.0
    ).to(torch.bfloat16)
    kv = (
        torch.randn(
            num_tokens, _D, dtype=torch.float32, generator=generator, device="cuda"
        )
        * 3.0
    ).to(torch.bfloat16)
    positions = torch.randint(
        0, _MAX_POS, (num_tokens,), generator=generator, device="cuda"
    )
    return q, kv, positions


# ----------------------------------------------------------------------------- CPU: facts


def test_format_info_matches_record_abi() -> None:
    info = cake_dsv4_nvfp4_rope_insert_format_info()
    assert info["head_dim"] == 512 and info["rope_dim"] == 64
    assert info["bytes_per_token"] == 384
    assert info["data_bytes_per_token"] + info["scale_bytes_per_token"] == 384
    assert info["q_head_padded_choices"] == (8, 16, 32, 64, 128)
    assert info["compress_ratios"] == (1, 2)
    assert set(info["slot_dtypes"]) == {"int32", "int64"}
    assert info["threads"] == 256
    assert set(info["entries"]) == {"qkv", "kv"}
    assert isinstance(info["kernels_available"], bool)


def test_lazy_exports() -> None:
    names = dir(flashinfer.mla)
    for name in (
        "cake_dsv4_nvfp4_rope_quantize_insert",
        "cake_dsv4_nvfp4_kv_rope_quantize_insert",
        "cake_dsv4_nvfp4_rope_insert_format_info",
    ):
        assert name in names
        assert callable(getattr(flashinfer.mla, name))


# ----------------------------------------------------------------------------- GPU: validation (any CUDA device)


def test_rejects_bad_inputs() -> None:
    """Every validation error is raised before any kernel is built or launched."""

    _require_cuda()
    q = torch.zeros(3, 8, _D, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(3, _D, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(2, 1, 4, _BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.zeros(3, dtype=torch.int64, device="cuda")
    pos = torch.zeros(3, dtype=torch.int64, device="cuda")
    cs = torch.zeros(16, _D_ROPE, dtype=torch.float32, device="cuda")

    def qkv(**overrides):
        args = dict(
            q=q, kv=kv, cache=cache, slot_mapping=slots, positions=pos, cos_sin_cache=cs
        )
        kwargs = dict(q_head_padded=8)
        for key, value in overrides.items():
            (kwargs if key in ("q_head_padded", "apply_q_rope") else args)[key] = value
        return cake_dsv4_nvfp4_rope_quantize_insert(**args, **kwargs)

    def kv_only(**overrides):
        args = dict(
            kv=kv, cache=cache, slot_mapping=slots, positions=pos, cos_sin_cache=cs
        )
        kwargs = {}
        for key, value in overrides.items():
            (kwargs if key == "compress_ratio" else args)[key] = value
        return cake_dsv4_nvfp4_kv_rope_quantize_insert(**args, **kwargs)

    with pytest.raises(ValueError, match="q must have dtype torch.bfloat16"):
        qkv(q=q.half())
    with pytest.raises(ValueError, match=r"q must be \[num_tokens, num_heads, 512\]"):
        qkv(q=q[..., :256])
    with pytest.raises(ValueError, match="q must be contiguous"):
        qkv(q=q.transpose(0, 1))
    with pytest.raises(ValueError, match="kv must have dtype torch.bfloat16"):
        qkv(kv=kv.float())
    with pytest.raises(ValueError, match=r"kv must be \[num_tokens, 512\]"):
        qkv(kv=kv.unsqueeze(1))
    with pytest.raises(ValueError, match="one row per query token"):
        qkv(kv=kv[:2])
    with pytest.raises(ValueError, match="cache must have dtype torch.uint8"):
        qkv(cache=cache.view(torch.int8))
    with pytest.raises(ValueError, match=r"cache must be \[num_pages"):
        qkv(cache=cache[..., :352])
    with pytest.raises(ValueError, match="singleton latent-head"):
        qkv(cache=torch.zeros(2, 3, 4, _BYTES, dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="multiple of 16"):
        qkv(
            cache=torch.zeros(
                2 * (4 * _BYTES + 8), dtype=torch.uint8, device="cuda"
            ).as_strided((2, 4, _BYTES), (4 * _BYTES + 8, _BYTES, 1))
        )
    with pytest.raises(
        ValueError, match="slot_mapping must have dtype torch.int32 or torch.int64"
    ):
        qkv(slot_mapping=slots.to(torch.int16))
    with pytest.raises(ValueError, match="contiguous 1D"):
        qkv(slot_mapping=slots.unsqueeze(1))
    with pytest.raises(ValueError, match="never longer"):
        qkv(slot_mapping=torch.zeros(4, dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="positions must have dtype torch.int64"):
        qkv(positions=pos.to(torch.int32))
    with pytest.raises(ValueError, match="one entry per token row"):
        qkv(positions=pos[:2])
    with pytest.raises(ValueError, match="cos_sin_cache must have dtype torch.float32"):
        qkv(cos_sin_cache=cs.to(torch.bfloat16))
    with pytest.raises(ValueError, match=r"cos_sin_cache must be \[max_position, 64\]"):
        qkv(cos_sin_cache=cs[:, :32])
    with pytest.raises(ValueError, match="cos_sin_cache must be contiguous"):
        qkv(cos_sin_cache=torch.zeros(16, 128, device="cuda")[:, ::2])
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        qkv(positions=pos.cpu())
    with pytest.raises(ValueError, match="q_head_padded must be 0 or one of"):
        qkv(q_head_padded=12)
    with pytest.raises(ValueError, match="at least the number of query heads"):
        qkv(
            q=torch.zeros(3, 16, _D, dtype=torch.bfloat16, device="cuda"),
            q_head_padded=8,
        )
    with pytest.raises(ValueError, match="compress_ratio must be one of"):
        kv_only(compress_ratio=3)
    with pytest.raises(ValueError, match="positions must have dtype torch.int64"):
        kv_only(positions=pos.to(torch.int32))
    with pytest.raises(ValueError, match="never longer"):
        kv_only(slot_mapping=torch.zeros(4, dtype=torch.int64, device="cuda"))


def test_empty_batch_returns_without_launch() -> None:
    """``num_tokens == 0`` allocates the empty ``q_out`` and never touches the kernels or the cache."""

    _require_cuda()
    cache = torch.full((2, 1, 4, _BYTES), _CANARY, dtype=torch.uint8, device="cuda")
    cs = torch.zeros(16, _D_ROPE, dtype=torch.float32, device="cuda")
    empty_i64 = torch.zeros(0, dtype=torch.int64, device="cuda")
    for q_head_padded in (0, 8, 16):
        q_out = cake_dsv4_nvfp4_rope_quantize_insert(
            torch.zeros(0, 8, _D, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(0, _D, dtype=torch.bfloat16, device="cuda"),
            cache,
            empty_i64,
            empty_i64,
            cs,
            q_head_padded=q_head_padded,
        )
        assert q_out.shape == (0, q_head_padded, _D) and q_out.dtype == torch.bfloat16
    assert (
        cake_dsv4_nvfp4_kv_rope_quantize_insert(
            torch.zeros(0, _D, dtype=torch.bfloat16, device="cuda"),
            cache,
            empty_i64,
            empty_i64,
            cs,
        )
        is None
    )
    # An empty slot_mapping with live rows inserts nothing (no launch) for the KV entry.
    cake_dsv4_nvfp4_kv_rope_quantize_insert(
        torch.zeros(3, _D, dtype=torch.bfloat16, device="cuda"),
        cache,
        empty_i64,
        torch.zeros(3, dtype=torch.int64, device="cuda"),
        cs,
        compress_ratio=2,
    )
    assert bool((cache == _CANARY).all())


# ----------------------------------------------------------------------------- GPU: byte parity (SM12x)


@pytest.mark.parametrize(
    "page_size,layout,pad_bytes,slot_dtype",
    [
        (32, "HND", 0, torch.int64),
        (64, "NHD", 0, torch.int32),
        (16, "3D", 0, torch.int64),
        (32, "HND", 128, torch.int64),  # padded pool: page stride > page payload
    ],
)
@pytest.mark.parametrize("q_head_padded", [8, 16])
def test_qkv_insert_matches_reference(
    page_size: int,
    layout: str,
    pad_bytes: int,
    slot_dtype: torch.dtype,
    q_head_padded: int,
) -> None:
    """Cache bytes == append writer on the reference's BF16 roped rows; q_out == torch reference; canary intact.

    Data-parallel padding (``slot_mapping`` shorter than ``positions``), one -1 slot and one
    out-of-range slot are part of every case.
    """

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(
        20261004 + page_size + q_head_padded
    )
    num_tokens, num_insert, num_heads = 37, 29, 8
    num_pages = 4
    q, kv, positions = _inputs(num_tokens, num_heads, generator)
    cos_sin = _cos_sin_cache()
    got = _Cache(num_pages, page_size, layout, pad_bytes)
    want = _Cache(num_pages, page_size, layout, pad_bytes)
    slots = _slot_mapping(
        got.total_slots,
        num_insert,
        slot_dtype,
        negatives=1,
        out_of_range=1,
        generator=generator,
    )

    q_out = cake_dsv4_nvfp4_rope_quantize_insert(
        q, kv, got.view, slots, positions, cos_sin, q_head_padded=q_head_padded
    )
    _reference_append(want.view, kv, slots, positions, cos_sin)
    torch.cuda.synchronize()
    _assert_same_cache(got, want)
    assert not torch.equal(got.storage, torch.full_like(got.storage, _CANARY))
    _assert_q_out(q_out, _reference_q_out(q, positions, cos_sin, q_head_padded))
    if q_head_padded > num_heads:
        assert bool((q_out[:, num_heads:] == 0).all())


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_heads", [8, 16])
def test_qkv_insert_inplace_matches_reference(
    slot_dtype: torch.dtype, num_heads: int
) -> None:
    """``q_inplace=True`` rotates q's rope dims in place (returned object is q), NoPE bits untouched, cache == reference."""

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261011 + num_heads)
    num_tokens, num_insert, page_size = 41, 33, 32
    q, kv, positions = _inputs(num_tokens, num_heads, generator)
    q_orig = q.clone()
    cos_sin = _cos_sin_cache()
    got, want = _Cache(4, page_size), _Cache(4, page_size)
    slots = _slot_mapping(
        got.total_slots,
        num_insert,
        slot_dtype,
        negatives=1,
        out_of_range=1,
        generator=generator,
    )
    for q_head_padded in (0, num_heads):
        q.copy_(q_orig)
        got.storage.fill_(_CANARY)
        out = cake_dsv4_nvfp4_rope_quantize_insert(
            q,
            kv,
            got.view,
            slots,
            positions,
            cos_sin,
            q_head_padded=q_head_padded,
            q_inplace=True,
        )
        torch.cuda.synchronize()
        assert out is q
        _assert_q_out(q, _reference_q_out(q_orig, positions, cos_sin, num_heads))
        assert torch.equal(
            q[..., :_D_NOPE].view(torch.int16), q_orig[..., :_D_NOPE].view(torch.int16)
        )
    _reference_append(want.view, kv, slots, positions, cos_sin)
    torch.cuda.synchronize()
    _assert_same_cache(got, want)
    # No head padding possible in place; unsupported head counts are rejected before any launch.
    with pytest.raises(ValueError):
        cake_dsv4_nvfp4_rope_quantize_insert(
            q,
            kv,
            got.view,
            slots,
            positions,
            cos_sin,
            q_head_padded=2 * num_heads,
            q_inplace=True,
        )
    with pytest.raises(ValueError):
        cake_dsv4_nvfp4_rope_quantize_insert(
            q[:, : num_heads - 1].contiguous(),
            kv,
            got.view,
            slots,
            positions,
            cos_sin,
            q_inplace=True,
        )
    # apply_q_rope=False in place: q untouched, cache written exactly as the KV entry.
    q.copy_(q_orig)
    got.storage.fill_(_CANARY)
    out = cake_dsv4_nvfp4_rope_quantize_insert(
        q, kv, got.view, slots, positions, cos_sin, apply_q_rope=False, q_inplace=True
    )
    torch.cuda.synchronize()
    assert out is q and torch.equal(q.view(torch.int16), q_orig.view(torch.int16))
    _assert_same_cache(got, want)


def test_qkv_insert_without_q_rope() -> None:
    """``apply_q_rope=False`` copies the live heads bitwise; the cache is still rotated and quantized."""

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261005)
    num_tokens, num_heads = 12, 8
    q, kv, positions = _inputs(num_tokens, num_heads, generator)
    cos_sin = _cos_sin_cache()
    got, want = _Cache(2, 32), _Cache(2, 32)
    slots = _slot_mapping(got.total_slots, num_tokens, torch.int64, generator=generator)
    q_out = cake_dsv4_nvfp4_rope_quantize_insert(
        q, kv, got.view, slots, positions, cos_sin, q_head_padded=16, apply_q_rope=False
    )
    _reference_append(want.view, kv, slots, positions, cos_sin)
    torch.cuda.synchronize()
    _assert_same_cache(got, want)
    assert torch.equal(q_out[:, :num_heads], q)
    assert bool((q_out[:, num_heads:] == 0).all())


def test_qkv_insert_kv_only_matches_kv_entry() -> None:
    """``q_head_padded=0`` returns an empty tensor and writes exactly what the KV entry (ratio 1) writes."""

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261006)
    num_tokens = 9
    q, kv, positions = _inputs(num_tokens, 8, generator)
    cos_sin = _cos_sin_cache()
    via_qkv, via_kv, want = _Cache(2, 16), _Cache(2, 16), _Cache(2, 16)
    slots = _slot_mapping(
        via_qkv.total_slots, num_tokens, torch.int32, negatives=1, generator=generator
    )
    q_out = cake_dsv4_nvfp4_rope_quantize_insert(
        q, kv, via_qkv.view, slots, positions, cos_sin
    )
    assert q_out.shape == (num_tokens, 0, _D) and q_out.dtype == torch.bfloat16
    cake_dsv4_nvfp4_kv_rope_quantize_insert(kv, via_kv.view, slots, positions, cos_sin)
    _reference_append(want.view, kv, slots, positions, cos_sin)
    torch.cuda.synchronize()
    _assert_same_cache(via_qkv, want)
    _assert_same_cache(via_kv, want)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_kv_insert_compress_ratio(compress_ratio: int, slot_dtype: torch.dtype) -> None:
    """Boundary rows only (``(pos + 1) % ratio == 0``), cos/sin row ``pos // ratio * ratio``; others keep the canary."""

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261007 + compress_ratio)
    num_tokens, num_insert = 40, 33
    _q, kv, _positions = _inputs(num_tokens, 8, generator)
    # Half the rows on a boundary, half not (ratio 2); every row is a boundary for ratio 1.
    positions = (
        torch.randint(
            0, _MAX_POS // 2, (num_tokens,), generator=generator, device="cuda"
        )
        * 2
    )
    positions[1::2] += 1
    cos_sin = _cos_sin_cache()
    got, want = _Cache(3, 32, "NHD"), _Cache(3, 32, "NHD")
    slots = _slot_mapping(
        got.total_slots, num_insert, slot_dtype, negatives=2, generator=generator
    )
    cake_dsv4_nvfp4_kv_rope_quantize_insert(
        kv, got.view, slots, positions, cos_sin, compress_ratio=compress_ratio
    )
    _reference_append(
        want.view, kv, slots, positions, cos_sin, compress_ratio=compress_ratio
    )
    torch.cuda.synchronize()
    _assert_same_cache(got, want)
    inserted = _inserted_slots(
        slots, positions, num_tokens, compress_ratio=compress_ratio
    )
    expected_rows = int((inserted >= 0).sum())
    if compress_ratio == 2:
        assert 0 < expected_rows < num_insert - 2
    # Exactly the inserted records left the canary state.
    data, scales = _split_pages(got)
    written_rows = int((data != _CANARY).any(dim=-1).sum())
    assert written_rows == expected_rows
    assert int((scales != _CANARY).any(dim=-1).sum()) == expected_rows


def _split_pages(cache: _Cache):
    """``(data [pages, page, 352], scales [pages, page, 32])`` of a cache's logical pages."""

    stride = cache.storage.numel() - 16
    stride //= cache.num_pages
    pages = cache.storage[cache.base : cache.base + cache.num_pages * stride].view(
        cache.num_pages, stride
    )
    payload = pages[:, : cache.page_size * _BYTES]
    data = payload[:, : cache.page_size * 352].reshape(
        cache.num_pages, cache.page_size, 352
    )
    scales = payload[:, cache.page_size * 352 :].reshape(
        cache.num_pages, cache.page_size, 32
    )
    return data, scales


def test_large_batch_with_padding_int32_slots() -> None:
    """Prefill-sized batch (more than 256 rows: the append writer's owner-claim path) with DP padding."""

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261008)
    num_tokens, num_insert = 4096, 4000
    q, kv, positions = _inputs(num_tokens, 8, generator)
    cos_sin = _cos_sin_cache()
    got, want = _Cache(20, 256), _Cache(20, 256)
    slots = _slot_mapping(
        got.total_slots, num_insert, torch.int32, negatives=3, generator=generator
    )
    q_out = cake_dsv4_nvfp4_rope_quantize_insert(
        q, kv, got.view, slots, positions, cos_sin, q_head_padded=8
    )
    _reference_append(want.view, kv, slots, positions, cos_sin)
    torch.cuda.synchronize()
    _assert_same_cache(got, want)
    _assert_q_out(q_out, _reference_q_out(q, positions, cos_sin, 8))


# ----------------------------------------------------------------------------- GPU: CUDA graph


def test_cuda_graph_replay_is_bitwise() -> None:
    """Both entries capture and replay; replays equal the eager results bitwise on new inputs."""

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261009)
    num_tokens, num_insert, num_heads = 24, 18, 8
    cos_sin = _cos_sin_cache()
    q, kv, positions = _inputs(num_tokens, num_heads, generator)
    swa, compressed = _Cache(2, 64), _Cache(4, 32)
    swa_slots = _slot_mapping(
        swa.total_slots, num_insert, torch.int64, negatives=2, generator=generator
    )
    comp_slots = _slot_mapping(
        compressed.total_slots, num_insert, torch.int32, generator=generator
    )

    def run() -> torch.Tensor:
        q_out = cake_dsv4_nvfp4_rope_quantize_insert(
            q, kv, swa.view, swa_slots, positions, cos_sin, q_head_padded=16
        )
        cake_dsv4_nvfp4_kv_rope_quantize_insert(
            kv, compressed.view, comp_slots, positions, cos_sin, compress_ratio=2
        )
        return q_out

    # Eager warm-up (JIT build) and the eager reference on the first inputs.
    eager_q_out = run()
    torch.cuda.synchronize()
    eager_swa, eager_comp = swa.storage.clone(), compressed.storage.clone()

    swa.reset()
    compressed.reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q_out = run()
    swa.reset()
    compressed.reset()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(swa.storage, eager_swa) and torch.equal(
        compressed.storage, eager_comp
    )
    assert torch.equal(graph_q_out, eager_q_out)

    # New inputs through the captured addresses: replay must equal a fresh eager run.
    q2, kv2, positions2 = _inputs(num_tokens, num_heads, generator)
    q.copy_(q2)
    kv.copy_(kv2)
    positions.copy_(positions2)
    swa.reset()
    compressed.reset()
    eager_q_out2 = run()
    torch.cuda.synchronize()
    eager_swa2, eager_comp2 = swa.storage.clone(), compressed.storage.clone()
    for _ in range(2):
        swa.reset()
        compressed.reset()
        graph_q_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(swa.storage, eager_swa2) and torch.equal(
            compressed.storage, eager_comp2
        )
        assert torch.equal(graph_q_out, eager_q_out2)


# ----------------------------------------------------------------------------- GPU: decode read-back


def test_decode_reads_back_fused_cache() -> None:
    """A cache written by the fused entry decodes bitwise like one written by the pre-fusion path.

    Both go through ``trtllm_batch_decode_sparse_mla_dsv4(backend="cake", kv_cache_format="nvfp4")``;
    the output is also checked against the fp32 reference over the dequantized operands.
    """

    _require_sm120()
    generator = torch.Generator(device="cuda").manual_seed(20261010)
    num_pages, page_size = 4, 32
    num_rows = num_pages * page_size
    num_heads, topk, num_queries = 8, 128, 3
    cos_sin = _cos_sin_cache()
    _q, _kv, positions = _inputs(num_rows, num_heads, generator)
    # The Cake decode kernel is validated against the fp32 reference at kv ~ N(0, 1), q ~ N(0, 1) with _OUT_TOL
    # (test_cake_sparse_mla_sm120_dsv4_nvfp4); the read-back comparison uses the same regime.
    kv = torch.randn(
        num_rows, _D, dtype=torch.float32, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    fused, prefix = _Cache(num_pages, page_size), _Cache(num_pages, page_size)
    slots = _slot_mapping(num_rows, num_rows, torch.int64, generator=generator)
    cake_dsv4_nvfp4_kv_rope_quantize_insert(kv, fused.view, slots, positions, cos_sin)
    _reference_append(prefix.view, kv, slots, positions, cos_sin)
    torch.cuda.synchronize()
    _assert_same_cache(fused, prefix)

    q_attn = torch.randn(
        num_queries,
        num_heads,
        _D,
        dtype=torch.float32,
        generator=generator,
        device="cuda",
    ).to(torch.bfloat16)
    indices = torch.randint(
        0, num_rows, (num_queries, topk), generator=generator, device="cuda"
    ).to(torch.int32)
    lengths = torch.randint(
        topk // 2, topk + 1, (num_queries,), generator=generator, device="cuda"
    ).to(torch.int32)
    workspace = torch.empty(
        cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(num_queries, num_heads, topk),
        dtype=torch.uint8,
        device="cuda",
    )
    sm_scale = _D**-0.5

    def decode(cache: torch.Tensor) -> torch.Tensor:
        return flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q_attn,
            swa_kv_cache=cache,
            workspace_buffer=workspace,
            sparse_indices=indices,
            swa_topk_lens=lengths,
            bmm1_scale=sm_scale,
            backend="cake",
            kv_cache_format="nvfp4",
        )

    out_fused = decode(fused.view)
    out_prefix = decode(prefix.view)
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_prefix)

    ref_indices = indices.clone()
    for token in range(num_queries):
        ref_indices[token, int(lengths[token].item()) :] = -1
    reference, _ = _reference_sparse_attention(
        _dequantize_nvfp4_query(q_attn),
        _dequantize_nvfp4_cache(fused.view).reshape(1, -1, 1, _D),
        ref_indices,
        sm_scale,
    )
    torch.testing.assert_close(out_fused, reference, **_OUT_TOL)
