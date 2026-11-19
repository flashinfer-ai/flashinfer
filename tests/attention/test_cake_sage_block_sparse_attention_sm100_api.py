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

Tests of the flashinfer public surface of the SM100 / SM103 Cake Sage block-sparse
attention programs: the ``flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake`` facade
and the ``backend="cake"`` route of ``bsa_attn_sm100_blk64_fwd``.  The facade's
entry points are the package's own functions; this module checks that identity,
the dispatcher's route against the package and against the CuTe-DSL backend,
and leaves the numerical contract to
``test_cake_sage_block_sparse_attention_sm100.py``, which imports only the
backend package.
"""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer
import flashinfer.cake_sage_block_sparse_attention_backend as backend
from flashinfer.cake_sage_block_sparse_attention_backend import cake_jit as loader
from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import (
    bsa_attn_sm100_sage_fwd_cake,
    is_cake_sage_sm100_supported,
    sage_fp8_quantize_sm100,
)
from flashinfer.jit.env import has_flashinfer_jit_cache

_HEAD_DIM = 128
_BLOCK = 64


def _supported() -> bool:
    return torch.cuda.is_available() and is_cake_sage_sm100_supported()


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="Cake Sage-FP8 block-sparse attention requires SM100 or SM103",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(torch.cuda.get_device_capability())
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def _inputs(
    batch: int, heads: int, kv_heads: int, sq: int, sk: int, sel: int, seed: int
):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (batch, sq, heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    k = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    v = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    q8, k8, v8, q_scale, k_scale, v_scale = sage_fp8_quantize_sm100(q, k, v)
    q_blocks, k_blocks = -(-sq // _BLOCK), -(-sk // _BLOCK)
    scores = torch.rand((batch, heads, q_blocks, k_blocks), device=device, generator=g)
    index = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    return q8, k8, v8, q_scale, k_scale, v_scale, index


def test_facade_entry_points_are_the_package_functions() -> None:
    assert bsa_attn_sm100_sage_fwd_cake is backend.bsa_attn_sm100_sage_fwd_cake
    assert sage_fp8_quantize_sm100 is backend.sage_fp8_quantize_sm100
    assert is_cake_sage_sm100_supported is backend.is_cake_sage_sm100_supported


def test_blk64_dispatcher_routes_backend_cake_to_the_package() -> None:
    """``bsa_attn_sm100_blk64_fwd(..., backend="cake")`` returns the package's bits."""
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    q8, k8, v8, q_scale, k_scale, v_scale, index = _inputs(2, 4, 2, 257, 1000, 6, 7)
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    out_route, lse_route = bsa_attn_sm100_blk64_fwd(
        q8,
        k8,
        v8,
        index,
        6,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=scale,
        return_lse=True,
        backend="cake",
    )
    out_pkg, lse_pkg = backend.bsa_attn_sm100_sage_fwd_cake(
        q8,
        k8,
        v8,
        index,
        6,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=scale,
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_route, out_pkg) and torch.equal(lse_route, lse_pkg)


def test_cake_sage_sm100_matches_cute_backend():
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    q8, k8, v8, q_scale, k_scale, v_scale, index = _inputs(1, 8, 8, 512, 512, 4, 512)
    common = dict(
        q_scale=q_scale,
        k_scale=k_scale,
        softmax_scale=1.0 / math.sqrt(_HEAD_DIM),
        return_lse=True,
    )
    out_cake, lse_cake = bsa_attn_sm100_blk64_fwd(
        q8, k8, v8, index, 4, v_scale=v_scale, backend="cake", **common
    )
    out_cute, lse_cute = bsa_attn_sm100_blk64_fwd(
        q8, k8, v8, index, 4, v_scale=v_scale[0], backend="cute", **common
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out_cake.float(), out_cute.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_cake, lse_cute, atol=1e-2, rtol=1e-2)


def _fp8_operands(batch: int, heads: int, sq: int, sk: int, dim: int = _HEAD_DIM):
    """Contiguous FP8 Q/K/V with matching scales and a one-block index (values unused)."""
    device = torch.device("cuda")
    q8 = torch.zeros((batch, sq, heads, dim), device=device).to(torch.float8_e4m3fn)
    k8 = torch.zeros((batch, sk, heads, dim), device=device).to(torch.float8_e4m3fn)
    v8 = torch.zeros_like(k8)
    q_scale = torch.ones((batch, heads, sq), device=device)
    k_scale = torch.ones((batch, heads, -(-sk // 16)), device=device)
    v_scale = torch.ones((heads, dim), device=device)
    index = torch.zeros(
        (batch, heads, -(-sq // _BLOCK), 1), dtype=torch.int32, device=device
    )
    return q8, k8, v8, index, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


def _offset_view(t: torch.Tensor, offset_bytes: int) -> torch.Tensor:
    """A contiguous copy of ``t`` that starts ``offset_bytes`` into its storage."""
    nbytes = t.numel() * t.element_size()
    buf = torch.empty(nbytes + offset_bytes, dtype=torch.uint8, device=t.device)
    view = buf[offset_bytes:].view(t.dtype).view(t.shape)
    view.copy_(t)
    assert view.is_contiguous()
    return view


def _refused_case(name: str):
    """Batch-2 calls the Cake backend refuses (the CuTe-DSL Sage-FP8 path then raises
    its own batch_size == 1 limit, which shows the call reached it)."""
    if name == "head_dim":
        q8, k8, v8, index, scales = _fp8_operands(2, 4, 128, 128, dim=64)
        return (q8, k8, v8, index, scales, {}), "head_dim must be 128"
    q8, k8, v8, index, scales = _fp8_operands(2, 4, 128, 128)
    if name == "misaligned_q":
        return (_offset_view(q8, 8), k8, v8, index, scales, {}), "16-byte boundary"
    if name == "misaligned_out":
        out = _offset_view(torch.zeros((2, 128, 4, 128), device="cuda").bfloat16(), 2)
        return (q8, k8, v8, index, scales, {"out": out}), "16-byte boundary"
    if name == "strided_k":
        k_strided = torch.zeros((2, 128, 8, 128), device="cuda").to(
            torch.float8_e4m3fn
        )[:, :, ::2]
        return (q8, k_strided, v8, index, scales, {}), "k must be contiguous"
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name", ["head_dim", "misaligned_q", "misaligned_out", "strided_k"]
)
def test_auto_backend_falls_back_to_cute_where_cake_refuses(name: str) -> None:
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    (q8, k8, v8, index, scales, extra), reason = _refused_case(name)
    assert (
        backend.sm100_sage_fwd_cake_dispatch(
            q8, k8, v8, index, 1, **scales, **extra, fallback=True
        )
        is None
    )
    with pytest.raises(ValueError, match=reason):
        bsa_attn_sm100_blk64_fwd(
            q8, k8, v8, index, 1, **scales, **extra, backend="cake"
        )
    # backend="auto" (the default) reaches the CuTe-DSL path instead of raising the Cake reason.
    with pytest.raises(ValueError, match="batch_size == 1"):
        bsa_attn_sm100_blk64_fwd(q8, k8, v8, index, 1, **scales, **extra)


def test_auto_backend_falls_back_to_cute_beyond_the_32bit_index_limit() -> None:
    """K/V of 2**31 elements: the Cake kernels index them with 32-bit offsets."""
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    batch, heads, sq, sk = 2, 2, 64, 2**22
    device = torch.device("cuda")
    q8 = torch.empty(
        (batch, sq, heads, _HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    k8 = torch.empty(
        (batch, sk, heads, _HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    assert k8.numel() == backend.MAX_FLAT_ELEMENTS + 1
    scales = dict(
        q_scale=torch.ones((batch, heads, sq), device=device),
        k_scale=torch.ones((batch, heads, sk // 16), device=device),
        v_scale=torch.ones((heads, _HEAD_DIM), device=device),
    )
    index = torch.zeros((batch, heads, 1, 1), dtype=torch.int32, device=device)
    assert (
        backend.sm100_sage_fwd_cake_dispatch(
            q8, k8, k8, index, 1, **scales, fallback=True
        )
        is None
    )
    with pytest.raises(ValueError, match="32-bit element offsets"):
        bsa_attn_sm100_blk64_fwd(q8, k8, k8, index, 1, **scales, backend="cake")
    with pytest.raises(ValueError, match="batch_size == 1"):
        bsa_attn_sm100_blk64_fwd(q8, k8, k8, index, 1, **scales)


def test_auto_backend_runs_cake_for_a_call_it_serves() -> None:
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    q8, k8, v8, q_scale, k_scale, v_scale, index = _inputs(2, 4, 2, 257, 1000, 6, 7)
    kwargs = dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale, return_lse=True)
    out_auto, lse_auto = bsa_attn_sm100_blk64_fwd(q8, k8, v8, index, 6, **kwargs)
    out_cake, lse_cake = bsa_attn_sm100_blk64_fwd(
        q8, k8, v8, index, 6, backend="cake", **kwargs
    )
    torch.cuda.synchronize()
    assert torch.equal(out_auto, out_cake) and torch.equal(lse_auto, lse_cake)


def test_fake_mode_launches_no_cake_kernel(monkeypatch) -> None:
    """Under torch fake mode "auto" keeps the CuTe-DSL path and backend="cake"
    checks the call and returns its outputs without a launch."""
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    def no_launch(*calls):
        raise AssertionError("a Cake kernel was launched under fake mode")

    monkeypatch.setattr(backend, "_launch", no_launch)
    with FakeTensorMode():
        q8, k8, v8, index, scales = _fp8_operands(2, 4, 128, 128)
        with pytest.raises(ValueError, match="batch_size == 1"):
            bsa_attn_sm100_blk64_fwd(q8, k8, v8, index, 1, **scales)
        out, lse = bsa_attn_sm100_blk64_fwd(
            q8, k8, v8, index, 1, **scales, return_lse=True, backend="cake"
        )
        assert isinstance(out, FakeTensor) and isinstance(lse, FakeTensor)
        assert out.shape == (2, 128, 4, _HEAD_DIM) and out.dtype == torch.bfloat16
        assert lse.shape == (2, 4, 128) and lse.dtype == torch.float32


@pytest.mark.parametrize("offset", [2, 16])
def test_quantizer_refuses_inputs_off_a_32_byte_boundary(offset: int) -> None:
    q = torch.randn((1, 64, 4, _HEAD_DIM), device="cuda").bfloat16()
    k = torch.randn((1, 64, 4, _HEAD_DIM), device="cuda").bfloat16()
    for impl in ("one_launch", "two_launch", "tma"):
        with pytest.raises(ValueError, match="q must start on a 32-byte boundary"):
            sage_fp8_quantize_sm100(_offset_view(q, offset), k, k, impl=impl)
        with pytest.raises(ValueError, match="v must start on a 32-byte boundary"):
            sage_fp8_quantize_sm100(q, k, _offset_view(k, offset), impl=impl)


def _fp8(shape) -> torch.Tensor:
    return torch.zeros(shape, device="cuda").to(torch.float8_e4m3fn)


def _f32(shape) -> torch.Tensor:
    return torch.ones(shape, device="cuda")


def _i32(shape) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device="cuda")


_IMAX = backend.MAX_FLAT_ELEMENTS


def _refusal_table(ops: dict):
    """``(case, replaced operands, exception type, message)`` of pre-quantized calls the
    checks refuse.  ``ops``: valid operands with B=2, H=4, Hkv=2, Sq=128, Sk=256 (2 query
    blocks, 16 K-scale groups).  A message ending in ``...`` is a prefix (it holds an
    address).  Cases with several faults pin the order of the checks."""
    q, k, v = ops["q"], ops["k"], ops["v"]
    dev = q.device
    kv = (2, 256, 2, _HEAD_DIM)
    bf16 = torch.bfloat16
    rank4 = "{} must be a rank-4 BSHD tensor"
    ratio = "num_heads must be a positive multiple of num_kv_heads"
    v_scale_shape = (
        "v_scale must be [Hkv,128]=(2, 128) or [B,Hkv,128]=(2, 2, 128), got {}"
    )
    index_shape = "q2k_block_index must be int32 [B,H,2,capacity]"
    big = (2, 2**22, 2, _HEAD_DIM)
    big_msg = (
        "{} has 2147483648 elements (shape (2, 4194304, 2, 128)); the kernels index it "
        f"with 32-bit element offsets, so it may hold at most {_IMAX} elements"
    )
    return [
        ("q_none", dict(q=None), ValueError, rank4.format("q")),
        ("k_rank3", dict(k=k[0]), ValueError, rank4.format("k")),
        ("v_list", dict(v=[1]), ValueError, rank4.format("v")),
        ("k_rank3_q_bf16", dict(k=k[0], q=q.to(bf16)), ValueError, rank4.format("k")),
        ("q_cpu", dict(q=q.cpu()), ValueError, "q must be a CUDA tensor"),
        (
            "head_dim_64",
            dict(
                q=_fp8((2, 128, 4, 64)),
                k=_fp8((2, 256, 2, 64)),
                v=_fp8((2, 256, 2, 64)),
            ),
            ValueError,
            "head_dim must be 128, got 64",
        ),
        (
            "gqa_ratio",
            dict(k=_fp8((2, 256, 3, 128)), v=_fp8((2, 256, 3, 128))),
            ValueError,
            ratio,
        ),
        (
            "no_kv_heads",
            dict(k=_fp8((2, 256, 0, 128)), v=_fp8((2, 256, 0, 128))),
            ValueError,
            ratio,
        ),
        (
            "q_bf16",
            dict(q=q.to(bf16)),
            TypeError,
            "q must use torch.float8_e4m3fn, got torch.bfloat16",
        ),
        (
            "q_strided",
            dict(q=_fp8((2, 4, 128, 128)).transpose(1, 2)),
            ValueError,
            "q must be contiguous",
        ),
        (
            "q_bf16_k_batch",
            dict(q=q.to(bf16), k=_fp8((1, 256, 2, 128))),
            TypeError,
            "q must use torch.float8_e4m3fn, got torch.bfloat16",
        ),
        (
            "k_batch",
            dict(k=_fp8((1, 256, 2, 128))),
            ValueError,
            f"k must have shape {kv}, got (1, 256, 2, 128)",
        ),
        ("k_cpu", dict(k=k.cpu()), ValueError, f"k must live on {dev}"),
        (
            "v_seqlen",
            dict(v=_fp8((2, 255, 2, 128))),
            ValueError,
            f"v must have shape {kv}, got (2, 255, 2, 128)",
        ),
        (
            "v_bf16",
            dict(v=v.to(bf16)),
            TypeError,
            "v must use torch.float8_e4m3fn, got torch.bfloat16",
        ),
        (
            "q_scale_none",
            dict(q_scale=None),
            TypeError,
            "q_scale must be a torch.Tensor",
        ),
        (
            "q_scale_shape",
            dict(q_scale=_f32((2, 4, 127))),
            ValueError,
            "q_scale must have shape (2, 4, 128), got (2, 4, 127)",
        ),
        (
            "q_scale_fp16",
            dict(q_scale=ops["q_scale"].half()),
            TypeError,
            "q_scale must use torch.float32, got torch.float16",
        ),
        (
            "q_scale_strided",
            dict(q_scale=_f32((2, 128, 4)).transpose(1, 2)),
            ValueError,
            "q_scale must be contiguous",
        ),
        (
            "k_scale_groups",
            dict(k_scale=_f32((2, 2, 15))),
            ValueError,
            "k_scale must have shape (2, 2, 16), got (2, 2, 15)",
        ),
        (
            "k_scale_cpu",
            dict(k_scale=ops["k_scale"].cpu()),
            ValueError,
            f"k_scale must live on {dev}",
        ),
        (
            "k_scale_groups_v_scale_bf16",
            dict(k_scale=_f32((2, 2, 15)), v_scale=ops["v_scale"].to(bf16)),
            ValueError,
            "k_scale must have shape (2, 2, 16), got (2, 2, 15)",
        ),
        (
            "v_scale_bf16",
            dict(v_scale=ops["v_scale"].to(bf16)),
            TypeError,
            "v_scale must use torch.float32, got torch.bfloat16",
        ),
        (
            "v_scale_strided",
            dict(v_scale=_f32((2, 128, 2)).transpose(1, 2)),
            ValueError,
            "v_scale must be contiguous",
        ),
        (
            "v_scale_shape",
            dict(v_scale=_f32((3, 128))),
            ValueError,
            v_scale_shape.format("(3, 128)"),
        ),
        (
            "q_misaligned",
            dict(q=_offset_view(q, 8)),
            ValueError,
            "q must start on a 16-byte boundary, got address ...",
        ),
        (
            "v_scale_misaligned",
            dict(v_scale=_offset_view(ops["v_scale"], 4)),
            ValueError,
            "v_scale must start on a 16-byte boundary, got address ...",
        ),
        (
            "q_misaligned_v_scale_shape",
            dict(q=_offset_view(q, 8), v_scale=_f32((3, 128))),
            ValueError,
            "q must start on a 16-byte boundary, got address ...",
        ),
        ("index_rank3", dict(q2k_block_index=_i32((2, 4, 2))), ValueError, index_shape),
        (
            "index_q_blocks",
            dict(q2k_block_index=_i32((2, 4, 3, 2))),
            ValueError,
            index_shape,
        ),
        (
            "index_int64",
            dict(q2k_block_index=_i32((2, 4, 2, 2)).long()),
            TypeError,
            "q2k_block_index must use torch.int32, got torch.int64",
        ),
        (
            "index_strided",
            dict(q2k_block_index=_i32((2, 4, 2, 4))[..., ::2]),
            ValueError,
            "q2k_block_index must be contiguous",
        ),
        (
            "index_cpu",
            dict(q2k_block_index=_i32((2, 4, 2, 2)).cpu()),
            ValueError,
            f"q2k_block_index must live on {dev}",
        ),
        (
            "index_empty",
            dict(q2k_block_index=_i32((2, 4, 2, 0))),
            ValueError,
            "q2k_block_index must have positive capacity",
        ),
        (
            "v_scale_shape_index_int64",
            dict(v_scale=_f32((3, 128)), q2k_block_index=_i32((2, 4, 2, 2)).long()),
            ValueError,
            v_scale_shape.format("(3, 128)"),
        ),
        (
            "q_2pow31",
            dict(
                q=torch.empty(big, dtype=torch.float8_e4m3fn, device=dev),
                q_scale=_f32((2, 2, 1)),
            ),
            ValueError,
            big_msg.format("q"),
        ),
        (
            "k_2pow31_k_scale_groups",
            dict(k=torch.empty(big, dtype=torch.float8_e4m3fn, device=dev)),
            ValueError,
            big_msg.format("k"),
        ),
    ]


def _raises(exc, message: str, fn, *args, **kwargs) -> None:
    with pytest.raises(exc) as info:
        fn(*args, **kwargs)
    assert type(info.value) is exc
    if message.endswith("..."):
        assert str(info.value).startswith(message[:-3]), str(info.value)
    else:
        assert str(info.value) == message


def test_attention_refuses_invalid_operands_in_check_order() -> None:
    """Every refusal of a pre-quantized call: the exception type and full message, the
    first failing check when a call has several faults, ``None`` from the dispatcher's
    fallback and the same refusal through ``bsa_attn_sm100_blk64_fwd(backend="cake")``."""
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    q8, k8, v8, q_scale, k_scale, v_scale, index = _inputs(2, 4, 2, 128, 256, 2, 11)
    valid = dict(
        q=q8,
        k=k8,
        v=v8,
        q2k_block_index=index,
        block_sparse_num=2,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    out, _ = bsa_attn_sm100_sage_fwd_cake(**valid)
    assert out.shape == (2, 128, 4, _HEAD_DIM) and out.dtype == torch.bfloat16
    assert out.is_contiguous() and out.device == q8.device
    dispatch = backend.sm100_sage_fwd_cake_dispatch
    for case, replaced, exc, message in _refusal_table(valid):
        ops = {**valid, **replaced}
        _raises(exc, message, bsa_attn_sm100_sage_fwd_cake, **ops)
        _raises(exc, message, dispatch, **ops)
        assert dispatch(**ops, fallback=True) is None, case
        # The dispatcher reads Q/K/V shapes and requires the three scales itself first.
        if all(isinstance(ops[name], torch.Tensor) for name in ("q", "k", "v")) and (
            ops["q_scale"] is not None
        ):
            _raises(exc, message, bsa_attn_sm100_blk64_fwd, **ops, backend="cake")
    torch.cuda.empty_cache()
