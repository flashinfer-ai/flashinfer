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
"""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import (
    bsa_attn_sm100_sage_fwd_cake,
    is_cake_sage_sm100_supported,
    sage_fp8_quantize_sm100,
)

_HEAD_DIM = 128
_BLOCK = 64
_K_GROUP = 16
_FP8_MAX = 448.0
_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "cake_sage_block_sparse_attention"
    / "cake_sage_block_sparse_attention_sm100_manifest.json"
)

requires_sm100 = pytest.mark.skipif(
    not is_cake_sage_sm100_supported(),
    reason="Cake Sage-FP8 block-sparse attention requires SM100 or SM103",
)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _torch_sage_quantize(q, k, v):
    """Torch reference of the Sage-FP8 quantization (BSHD in; FlashInfer scale layouts out)."""
    batch, sq, heads, dim = q.shape
    sk, kv_heads = k.shape[1], k.shape[2]
    q32, k32, v32 = (t.float().transpose(1, 2) for t in (q, k, v))
    q_scale = q32.abs().amax(dim=-1).clamp_min(1e-12) / _FP8_MAX
    q8 = (
        (q32 / q_scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    )
    groups = _ceil_div(sk, _K_GROUP)
    k_pad = torch.nn.functional.pad(k32, (0, 0, 0, groups * _K_GROUP - sk))
    k_grouped = k_pad.view(batch, kv_heads, groups, _K_GROUP, dim)
    k_scale = k_grouped.abs().amax(dim=(-1, -2)).clamp_min(1e-12) / _FP8_MAX
    k8 = (k_grouped / k_scale[..., None, None]).clamp(-_FP8_MAX, _FP8_MAX)
    k8 = k8.view(batch, kv_heads, groups * _K_GROUP, dim)[:, :, :sk].to(
        torch.float8_e4m3fn
    )
    v_scale = v32.abs().amax(dim=-2).clamp_min(1e-12) / _FP8_MAX
    v8 = (
        (v32 / v_scale.unsqueeze(-2)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    )
    return (
        q8.transpose(1, 2).contiguous(),
        k8.transpose(1, 2).contiguous(),
        v8.transpose(1, 2).contiguous(),
        q_scale.contiguous(),
        k_scale.contiguous(),
        v_scale.contiguous(),
    )


def _reference(
    q8,
    k8,
    v8,
    q_scale,
    k_scale,
    v_scale,
    index,
    count,
    *,
    block_nums,
    block_sizes,
    scale,
):
    """Exact-P dequantized FP32 reference over the selected tokens (per (b, h, q-block))."""
    q = q8.float().transpose(1, 2)
    k = k8.float().transpose(1, 2)
    v = v8.float().transpose(1, 2)
    batch, heads, sq, dim = q.shape
    kv_heads, sk = k.shape[1], k.shape[2]
    group = heads // kv_heads
    q_blocks = _ceil_div(sq, _BLOCK)
    k_tok_scale = k_scale.repeat_interleave(_K_GROUP, dim=-1)[..., :sk]
    if v_scale.ndim == 2:
        v_scale = v_scale.unsqueeze(0).expand(batch, kv_heads, dim)
    out = torch.zeros((batch, heads, sq, dim), dtype=torch.float32, device=q.device)
    lse = torch.full(
        (batch, heads, sq), float("-inf"), dtype=torch.float32, device=q.device
    )
    for b in range(batch):
        for h in range(heads):
            kh = h // group
            for qb in range(q_blocks):
                q0, q1 = qb * _BLOCK, min(sq, (qb + 1) * _BLOCK)
                n = int(block_nums[b, h, qb]) if block_nums is not None else count
                tokens = []
                for blk in index[b, h, qb, : max(n, 0)].tolist():
                    valid = max(0, min(_BLOCK, sk - blk * _BLOCK))
                    if block_sizes is not None:
                        bs = block_sizes
                        valid = min(
                            valid,
                            max(
                                0,
                                int(
                                    bs[blk]
                                    if bs.ndim == 1
                                    else bs[b, blk]
                                    if bs.ndim == 2
                                    else bs[b, kh, blk]
                                ),
                            ),
                        )
                    if valid > 0:
                        tokens.append(
                            torch.arange(
                                blk * _BLOCK, blk * _BLOCK + valid, device=q.device
                            )
                        )
                if not tokens:
                    continue
                tok = torch.cat(tokens)
                kk = k[b, kh, tok] * k_tok_scale[b, kh, tok, None]
                vv = v[b, kh, tok] * v_scale[b, kh][None, :]
                qq = q[b, h, q0:q1] * q_scale[b, h, q0:q1, None]
                s = (qq @ kk.T) * scale
                m = s.amax(dim=-1, keepdim=True)
                p = torch.exp(s - m)
                d = p.sum(dim=-1, keepdim=True)
                out[b, h, q0:q1] = (p / d) @ vv
                lse[b, h, q0:q1] = (m + torch.log(d)).squeeze(-1)
    return out.transpose(1, 2).contiguous(), lse


_CASES = [
    # name, batch, heads, kv_heads, sq, sk, selected, per_row, empty_row, block_sizes_rank, valid_last, lse
    ("aligned_b1_h8_s256", 1, 8, 8, 256, 256, 2, False, False, 0, None, False),
    ("ragged_q_k_rank1_lse", 2, 3, 3, 257, 1000, 6, False, False, 1, 17, True),
    ("per_row_counts_empty_row", 2, 4, 4, 320, 1024, 8, True, True, 0, None, True),
    ("gqa_h4_hkv2_rank3", 1, 4, 2, 100, 257, 4, True, False, 3, 33, False),
    ("gqa_h8_hkv1_rank2", 2, 8, 1, 65, 96, 2, False, False, 2, 32, True),
    ("dense_b1_h4_s1024", 1, 4, 4, 1024, 1024, 16, False, False, 0, None, False),
]


def _make_case(case, device):
    (
        name,
        batch,
        heads,
        kv_heads,
        sq,
        sk,
        sel,
        per_row,
        empty_row,
        rank,
        valid_last,
        use_lse,
    ) = case
    g = torch.Generator(device=device).manual_seed(abs(hash(name)) % (2**31))
    q = torch.randn(
        (batch, sq, heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    k = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    v = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    q8, k8, v8, q_scale, k_scale, v_scale = _torch_sage_quantize(q, k, v)
    q_blocks, k_blocks = _ceil_div(sq, _BLOCK), _ceil_div(sk, _BLOCK)
    sel = min(sel, k_blocks)
    scores = torch.rand((batch, heads, q_blocks, k_blocks), device=device, generator=g)
    index = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    block_nums = None
    if per_row or empty_row:
        block_nums = torch.randint(
            0, sel + 1, (batch, heads, q_blocks), device=device, generator=g
        ).to(torch.int32)
        if empty_row:
            block_nums[0, 0, 0] = 0
        block_nums = block_nums.contiguous()
    block_sizes = None
    if rank:
        full = torch.full(
            (batch, kv_heads, k_blocks), _BLOCK, dtype=torch.int32, device=device
        )
        full[..., -1] = (
            sk - (k_blocks - 1) * _BLOCK if valid_last is None else valid_last
        )
        block_sizes = {1: full[0, 0], 2: full[:, 0], 3: full}[rank].contiguous()
    return dict(
        q8=q8,
        k8=k8,
        v8=v8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        index=index,
        sel=sel,
        block_nums=block_nums,
        block_sizes=block_sizes,
        use_lse=use_lse,
        scale=1.0 / math.sqrt(_HEAD_DIM),
    )


@requires_sm100
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_cake_sage_sm100_matches_reference(case):
    device = torch.device("cuda")
    c = _make_case(case, device)
    out, lse = bsa_attn_sm100_sage_fwd_cake(
        c["q8"],
        c["k8"],
        c["v8"],
        c["index"],
        c["sel"],
        q_scale=c["q_scale"],
        k_scale=c["k_scale"],
        v_scale=c["v_scale"],
        block_sizes=c["block_sizes"],
        q2k_block_nums=c["block_nums"],
        softmax_scale=c["scale"],
        return_lse=c["use_lse"],
    )
    torch.cuda.synchronize()
    ref, ref_lse = _reference(
        c["q8"],
        c["k8"],
        c["v8"],
        c["q_scale"],
        c["k_scale"],
        c["v_scale"],
        c["index"],
        c["sel"],
        block_nums=c["block_nums"],
        block_sizes=c["block_sizes"],
        scale=c["scale"],
    )
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)
    if c["use_lse"]:
        finite = torch.isfinite(ref_lse)
        assert torch.equal(torch.isinf(lse), ~finite)
        torch.testing.assert_close(lse[finite], ref_lse[finite], atol=1e-2, rtol=1e-2)
    else:
        assert lse is None


@requires_sm100
def test_cake_sage_sm100_matches_cute_backend():
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    device = torch.device("cuda")
    c = _make_case(
        ("cross_b1_h8_s512", 1, 8, 8, 512, 512, 4, False, False, 0, None, True), device
    )
    common = dict(
        q_scale=c["q_scale"],
        k_scale=c["k_scale"],
        softmax_scale=c["scale"],
        return_lse=True,
    )
    out_cake, lse_cake = bsa_attn_sm100_blk64_fwd(
        c["q8"],
        c["k8"],
        c["v8"],
        c["index"],
        c["sel"],
        v_scale=c["v_scale"],
        backend="cake",
        **common,
    )
    out_cute, lse_cute = bsa_attn_sm100_blk64_fwd(
        c["q8"],
        c["k8"],
        c["v8"],
        c["index"],
        c["sel"],
        v_scale=c["v_scale"][0],
        backend="cute",
        **common,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out_cake.float(), out_cute.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_cake, lse_cute, atol=1e-2, rtol=1e-2)


@requires_sm100
def test_cake_sage_sm100_v_scale_without_batch_axis():
    device = torch.device("cuda")
    c = _make_case(
        ("vscale_hd_b1_h4", 1, 4, 4, 192, 320, 3, False, False, 0, None, False), device
    )
    out_b, _ = bsa_attn_sm100_sage_fwd_cake(
        c["q8"],
        c["k8"],
        c["v8"],
        c["index"],
        c["sel"],
        q_scale=c["q_scale"],
        k_scale=c["k_scale"],
        v_scale=c["v_scale"],
        softmax_scale=c["scale"],
    )
    out_hd, _ = bsa_attn_sm100_sage_fwd_cake(
        c["q8"],
        c["k8"],
        c["v8"],
        c["index"],
        c["sel"],
        q_scale=c["q_scale"],
        k_scale=c["k_scale"],
        v_scale=c["v_scale"][0].contiguous(),
        softmax_scale=c["scale"],
    )
    torch.cuda.synchronize()
    assert torch.equal(out_b, out_hd)


@requires_sm100
@pytest.mark.parametrize(
    "shape", [(1, 8, 8, 256, 256), (2, 4, 2, 1000, 1500), (2, 6, 3, 65, 4000)]
)
def test_sage_fp8_quantize_sm100_matches_torch(shape):
    device = torch.device("cuda")
    batch, heads, kv_heads, sq, sk = shape
    g = torch.Generator(device=device).manual_seed(sq * 7 + sk)
    q = torch.randn(
        (batch, sq, heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    k = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    v = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    got = sage_fp8_quantize_sm100(q, k, v)
    torch.cuda.synchronize()
    ref = _torch_sage_quantize(q, k, v)
    for name, a, b in zip(
        ("q_scale", "k_scale", "v_scale"), got[3:], ref[3:], strict=True
    ):
        assert a.shape == b.shape, name
        # The kernel multiplies by 1/448 where torch divides by 448: <= 1e-5 relative.
        rel = float((a - b).abs().max() / b.abs().max())
        assert rel < 1e-5, f"{name}: max relative scale error {rel:.3e}"
    for name, a, b in zip(("q8", "k8", "v8"), got[:3], ref[:3], strict=True):
        assert a.dtype == torch.float8_e4m3fn and a.shape == b.shape, name
        af, bf = a.float(), b.float()
        # Reciprocal-multiply vs divide may flip a rounding boundary on isolated
        # elements: bound the value error by one E4M3 step at the element's
        # magnitude (spacing |x|/8 for normals, absolute 1/8 floor below 1.0).
        step = torch.maximum(af.abs(), bf.abs()).clamp_min(1.0) * 0.125
        mismatch = ((af - bf).abs() > step).sum().item()
        assert mismatch == 0, (
            f"{name}: {mismatch} elements differ by more than one E4M3 step"
        )


def test_cake_sage_sm100_manifest_is_consistent():
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "cake.library_export.v4"
    assert manifest["name"] == "cake_sage_block_sparse_attention_sm100"
    assert manifest["artifact_kind"] == "source_only"
    modules = manifest["modules"]
    keys = sorted((m["arch"], m["route"]["stage"]) for m in modules)
    assert keys == sorted(
        (arch, stage)
        for arch in ("sm_100a", "sm_103a")
        for stage in ("attention", "quantize_qk_vamax", "quantize_v")
    )
    root = _MANIFEST.parents[2]
    for item in manifest["files"]:
        path = root / item["path"]
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item[
            "path"
        ]
    for module in modules:
        units = module["translation_units"]
        assert units["device"].endswith("_kernel.cu") and units["binding"].endswith(
            "_binding.cu"
        )
        assert f"/{module['arch']}/" in units["device"]
        assert module["ffi_entry"] == "run"
        names = {name for _kind, name in module["arg_plan"]}
        if module["route"]["stage"] == "attention":
            assert {
                "q",
                "k",
                "v",
                "out",
                "lse",
                "tma_descriptor_workspace",
                "grid_x",
            } <= names
            assert module["tma_workspace_bytes"] == 384
