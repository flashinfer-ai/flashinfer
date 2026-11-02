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

Backend suite of the SM100 / SM103 programs of the Cake Sage block-sparse
attention package: the Sage-FP8 block-sparse attention forward against an
exact-P dequantized reference, the fused quantizer against the torch recipe,
the one-launch and CUDA-Graph contract of the host path, and the package's
``build.json`` contract.  It imports the package only through
``tests.test_helpers.cake_package`` and nothing else from flashinfer; the
flashinfer public routes are covered by
``test_cake_sage_block_sparse_attention_sm100_api.py``.
"""

import math
import re
import zlib
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package
from tests.test_helpers.cake_profiler import cuda_kernel_names

pkg = cake_package("cake_sage_block_sparse_attention")
cake_build = cake_package("cake_sage_block_sparse_attention", "cake_build")
loader = cake_package("cake_sage_block_sparse_attention", "cake_jit")

bsa_attn_sm100_sage_fwd_cake = pkg.bsa_attn_sm100_sage_fwd_cake
is_cake_sage_sm100_supported = pkg.is_cake_sage_sm100_supported
sage_fp8_quantize_sm100 = pkg.sage_fp8_quantize_sm100

_HEAD_DIM = 128
_BLOCK = 64
_K_GROUP = 16
_FP8_MAX = 448.0
requires_sm100 = pytest.mark.skipif(
    not is_cake_sage_sm100_supported(),
    reason="Cake Sage-FP8 block-sparse attention requires SM100 or SM103",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if is_cake_sage_sm100_supported():
        arch = loader.arch_for_capability(torch.cuda.get_device_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


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


# Relative slack of the kernel's P against the reference's: FP32 rounding of the scaled
# scores and the approximate exp2 (``ex2.approx`` or its polynomial emulation) move a P
# entry by well under this fraction, so an entry closer than this to an e4m3 rounding
# boundary may round to the other neighbour in the kernel.
_P_ROUNDING_SLACK = 1e-4
# The kernel's softmax (``cake_sage_block_sparse_attention_9884891b129880048274_kernel.cu``):
# the row max moves only when a pair's max exceeds it by more than log2(1.75), and the
# exp2 argument carries a +8 bias, so P = 2^(s - m + 8) lies in (0, 448] (e4m3's range) and
# is rounded on that grid.
_MAX_STEP_LOG2 = 0.8073549220576041  # log2(1.75)
_P_BIAS_LOG2 = 8.0
_E4M3_MAX = 448.0
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


def _e4m3(p):
    """P rounded to e4m3 as the kernel's ``cvt.rn.satfinite.e4m3x2.f32`` does (P is in (0, 448];
    the conversion saturates at 448)."""
    return p.clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn).float()


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
    """Reference of the kernel's arithmetic over the selected tokens (per (b, h, q-block)).

    FP32 scores of the dequantized operands in log2 units; the online softmax of the kernel's
    two consumer groups (group ``g`` takes every other pair of selected blocks starting at pair
    ``g``; the 128 columns of a pair share one max update).  The row max follows the kernel's
    rule: a pair takes it when the row had none, and later only when the pair's max exceeds
    it by more than ``_MAX_STEP_LOG2``; ``P = 2^(s - m + 8)`` (``_P_BIAS_LOG2``) is rounded to
    e4m3 before the PV product while the row sum keeps the exact FP32 ``P``; the two groups are
    merged at the end and the LSE drops the bias.  Returns the output, the LSE and a per-element
    budget for the entries the kernel may round to the other e4m3 neighbour
    (``_P_ROUNDING_SLACK``): their one rounding step, weighted by |V| and normalized like the
    output.
    """
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
    budget = torch.zeros_like(out)
    lse = torch.full(
        (batch, heads, sq), float("-inf"), dtype=torch.float32, device=q.device
    )
    for b in range(batch):
        for h in range(heads):
            kh = h // group
            for qb in range(q_blocks):
                q0, q1 = qb * _BLOCK, min(sq, (qb + 1) * _BLOCK)
                n = int(block_nums[b, h, qb]) if block_nums is not None else count
                blocks = []  # the valid token range of every selected block, in selection order
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
                    blocks.append(
                        torch.arange(
                            blk * _BLOCK, blk * _BLOCK + valid, device=q.device
                        )
                    )
                if not any(len(tok) for tok in blocks):
                    continue
                qq = q[b, h, q0:q1] * q_scale[b, h, q0:q1, None]
                rows = q1 - q0
                merged = []  # (m, l, acc, bud) of the two consumer groups
                for g in range(2):
                    m = torch.full((rows,), float("-inf"), device=q.device)
                    l_sum = torch.zeros((rows,), device=q.device)
                    acc = torch.zeros((rows, dim), device=q.device)
                    bud = torch.zeros((rows, dim), device=q.device)
                    for pair in range(g, _ceil_div(len(blocks), 2), 2):
                        tok = torch.cat(blocks[2 * pair : 2 * pair + 2])
                        if not len(tok):
                            continue
                        kk = k[b, kh, tok] * k_tok_scale[b, kh, tok, None]
                        vv = v[b, kh, tok] * v_scale[b, kh][None, :]
                        s = (qq @ kk.T) * (scale * _LOG2E)
                        tile_max = s.amax(dim=-1)
                        fresh = torch.isinf(m)  # the row had no column yet
                        bump = ~fresh & (tile_max - m > _MAX_STEP_LOG2)
                        alpha = torch.where(bump, torch.exp2(m - tile_max), 1.0)
                        m = torch.where(fresh | bump, tile_max, m)
                        p = torch.exp2(s + _P_BIAS_LOG2 - m[:, None])  # in (0, 448]
                        l_sum = l_sum * alpha + p.sum(dim=-1)
                        acc = acc * alpha[:, None] + _e4m3(p) @ vv
                        flip = (
                            _e4m3(p * (1.0 + _P_ROUNDING_SLACK))
                            - _e4m3(p * (1.0 - _P_ROUNDING_SLACK))
                        ).abs()
                        bud = bud * alpha[:, None] + flip @ vv.abs()
                    merged.append((m, l_sum, acc, bud))
                m_all = torch.maximum(merged[0][0], merged[1][0])
                l_all = torch.zeros((rows,), device=q.device)
                acc_all = torch.zeros((rows, dim), device=q.device)
                bud_all = torch.zeros((rows, dim), device=q.device)
                for m, l_sum, acc, bud in merged:
                    w = torch.exp2(m - m_all)  # 0 for a group without a pair
                    l_all = l_all + w * l_sum
                    acc_all = acc_all + w[:, None] * acc
                    bud_all = bud_all + w[:, None] * bud
                out[b, h, q0:q1] = acc_all / l_all[:, None]
                budget[b, h, q0:q1] = bud_all / l_all[:, None]
                lse[b, h, q0:q1] = (m_all + torch.log2(l_all) - _P_BIAS_LOG2) * _LN2
    return (
        out.transpose(1, 2).contiguous(),
        lse,
        budget.transpose(1, 2).contiguous(),
    )


_CASES = [
    # name, batch, heads, kv_heads, sq, sk, selected, per_row, empty_row, block_sizes_rank, valid_last, lse, q_gain
    ("aligned_b1_h8_s256", 1, 8, 8, 256, 256, 2, False, False, 0, None, False, 1.0),
    ("ragged_q_k_rank1_lse", 2, 3, 3, 257, 1000, 6, False, False, 1, 17, True, 1.0),
    ("per_row_counts_empty_row", 2, 4, 4, 320, 1024, 8, True, True, 0, None, True, 1.0),
    ("gqa_h4_hkv2_rank3", 1, 4, 2, 100, 257, 4, True, False, 3, 33, False, 1.0),
    ("gqa_h8_hkv1_rank2", 2, 8, 1, 65, 96, 2, False, False, 2, 32, True, 1.0),
    ("dense_b1_h4_s1024", 1, 4, 4, 1024, 1024, 16, False, False, 0, None, False, 1.0),
    # A peaked softmax: Q scaled by 3 before quantization concentrates each row on
    # one token, so a one-count row quantizes a dominant P entry (the shape of the
    # SM100 hybrid-exp2 regression); rank-2 block sizes cut the last block to 40.
    ("peaked_gqa_h6_hkv3_rank2", 3, 6, 3, 320, 1024, 5, False, False, 2, 40, True, 3.0),
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
        q_gain,
    ) = case
    # a seed that does not depend on the process's hash randomization: a failing draw reproduces
    g = torch.Generator(device=device).manual_seed(zlib.crc32(name.encode()))
    q = torch.randn(
        (batch, sq, heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    if q_gain != 1.0:
        q = (q.float() * q_gain).bfloat16()
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


def _run(c, **overrides):
    kwargs = dict(
        q_scale=c["q_scale"],
        k_scale=c["k_scale"],
        v_scale=c["v_scale"],
        block_sizes=c["block_sizes"],
        q2k_block_nums=c["block_nums"],
        softmax_scale=c["scale"],
        return_lse=c["use_lse"],
    )
    kwargs.update(overrides)
    return bsa_attn_sm100_sage_fwd_cake(
        c["q8"], c["k8"], c["v8"], c["index"], c["sel"], **kwargs
    )


@requires_sm100
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_cake_sage_sm100_matches_reference(case):
    device = torch.device("cuda")
    c = _make_case(case, device)
    out, lse = _run(c)
    torch.cuda.synchronize()
    ref, ref_lse, budget = _reference(
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
    # The reference rounds P to e4m3 on the kernel's grid and keeps its row max where the
    # kernel keeps it, so what remains is the BF16 output rounding, FP32 ordering and the
    # entries the kernel may round to the other e4m3 neighbour (``budget``).  The former
    # exact-P reference with a 5e-2 tolerance missed the rounding of a dominant entry in a
    # pair that leaves the row max where it was (P in (256, 448], a step of 32, up to 2^-4 of
    # |V| after normalization), which a peaked draw of the inputs crossed.
    error = (out.float() - ref).abs()
    allowed = 1e-2 + 1e-2 * ref.abs() + budget
    ratio = error / allowed
    worst = int(ratio.argmax())
    assert bool((ratio <= 1.0).all()), (
        f"max error / allowance {ratio.flatten()[worst].item():.3f} at flat index {worst}: "
        f"error {error.flatten()[worst].item():.4g}, reference {ref.flatten()[worst].item():.4g}, "
        f"budget {budget.flatten()[worst].item():.4g}"
    )
    if c["use_lse"]:
        finite = torch.isfinite(ref_lse)
        assert torch.equal(torch.isinf(lse), ~finite)
        torch.testing.assert_close(lse[finite], ref_lse[finite], atol=1e-2, rtol=1e-2)
    else:
        assert lse is None


@requires_sm100
def test_cake_sage_sm100_v_scale_without_batch_axis():
    device = torch.device("cuda")
    c = _make_case(
        ("vscale_hd_b1_h4", 1, 4, 4, 192, 320, 3, False, False, 0, None, False, 1.0),
        device,
    )
    out_b, _ = _run(c)
    out_hd, _ = _run(c, v_scale=c["v_scale"][0].contiguous())
    torch.cuda.synchronize()
    assert torch.equal(out_b, out_hd)


@requires_sm100
def test_cake_sage_sm100_launches_one_kernel_per_call():
    """The attention call is one kernel: the TMA descriptors travel by value in the
    kernel parameters, so no upload kernel precedes the attention kernel.  The
    quantizer is its two kernels behind the zero-fill of its V amax scratch
    (``torch.zeros``: a torch fill kernel, or a memset on torch builds that zero
    a fresh tensor that way).  The launch records come from
    ``tests.test_helpers.cake_profiler``, which retries a profiler session that
    lost its CUDA records."""
    device = torch.device("cuda")
    c = _make_case(
        ("one_launch_b2_h4", 2, 4, 2, 257, 1000, 6, True, True, 2, 40, True, 1.0),
        device,
    )
    out = torch.empty_like(c["q8"], dtype=torch.bfloat16)
    lse = torch.empty(c["q_scale"].shape, dtype=torch.float32, device=device)
    names = cuda_kernel_names(lambda: _run(c, out=out, lse=lse))
    symbol = loader.MODULES["attention"]["kernel_symbol"]
    assert names == [symbol], names
    g = torch.Generator(device=device).manual_seed(11)
    q, k, v = (
        torch.randn(s, device=device, generator=g).bfloat16()
        for s in (
            (1, 300, 4, _HEAD_DIM),
            (1, 500, 2, _HEAD_DIM),
            (1, 500, 2, _HEAD_DIM),
        )
    )
    names = cuda_kernel_names(
        lambda: sage_fp8_quantize_sm100(q, k, v), exclude=("Memcpy",)
    )
    kernels = [name for name in names if name.startswith("kernel_")]
    assert kernels == [
        loader.MODULES["quantize_qk_vamax"]["kernel_symbol"],
        loader.MODULES["quantize_v"]["kernel_symbol"],
    ], names
    fills = [name for name in names if not name.startswith("kernel_")]
    assert fills == [names[0]] and (
        "Fill" in names[0] or names[0].startswith("Memset")
    ), names


@requires_sm100
def test_cake_sage_sm100_cuda_graph_replay_matches_eager():
    """A captured call replays the descriptor bytes it captured: the replay writes
    the same bits as the eager call on the captured buffers, and again after the
    inputs are rewritten in place (the maps address the buffers, not a snapshot)."""
    device = torch.device("cuda")
    c = _make_case(
        ("graph_b2_h4_hkv2", 2, 4, 2, 320, 1024, 8, True, True, 3, 33, True, 1.0),
        device,
    )
    out_eager = torch.empty_like(c["q8"], dtype=torch.bfloat16)
    lse_eager = torch.empty(c["q_scale"].shape, dtype=torch.float32, device=device)
    out_graph = torch.empty_like(out_eager)
    lse_graph = torch.empty_like(lse_eager)
    _run(c, out=out_eager, lse=lse_eager)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _run(c, out=out_graph, lse=lse_graph)  # warm-up on the capture stream
        stream.synchronize()
        out_graph.zero_()
        lse_graph.zero_()
        with torch.cuda.graph(graph, stream=stream):
            _run(c, out=out_graph, lse=lse_graph)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out_graph).item() == 0  # capture launched nothing
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager)
    assert torch.equal(lse_graph, lse_eager)
    # Rewrite the inputs in place; the replay reads the new contents through the captured maps.
    c["q8"].copy_(c["q8"].flip(1))
    c["q_scale"].copy_(c["q_scale"].flip(-1))
    _run(c, out=out_eager, lse=lse_eager)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager)
    assert torch.equal(lse_graph, lse_eager)


@requires_sm100
def test_run_module_matches_the_host_api():
    """The generic ``run_module`` (logical names through the record's ``arg_plan``)
    launches what the host API binds positionally: the same bits."""
    device = torch.device("cuda")
    c = _make_case(
        ("run_module_b1_h4_hkv2", 1, 4, 2, 100, 257, 4, True, False, 3, 33, True, 1.0),
        device,
    )
    out_api, lse_api = _run(c)
    batch, sq, heads, _ = c["q8"].shape
    sk, kv_heads = c["k8"].shape[1], c["k8"].shape[2]
    q_blocks, k_blocks = _ceil_div(sq, _BLOCK), _ceil_div(sk, _BLOCK)
    out = torch.empty_like(out_api)
    lse = torch.empty_like(lse_api)
    values = {
        "q": c["q8"].view(torch.uint8),
        "k": c["k8"].view(torch.uint8),
        "v": c["v8"].view(torch.uint8),
        "out": out,
        "lse": lse,
        "q_scale": c["q_scale"],
        "k_scale": c["k_scale"],
        "v_scale": c["v_scale"],
        "block_index": c["index"],
        "block_nums": c["block_nums"],
        "block_sizes": c["block_sizes"],
        "seqlen_q": sq,
        "seqlen_k": sk,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "q_blocks": q_blocks,
        "k_blocks": k_blocks,
        "k_scale_groups": _ceil_div(sk, _K_GROUP),
        "capacity": int(c["index"].shape[-1]),
        "block_sparse_num": int(c["index"].shape[-1]),
        "has_block_nums": 1,
        "block_sizes_mode": 3,
        "v_scale_batch_stride": kv_heads * _HEAD_DIM,
        "total_tiles": batch * heads * q_blocks,
        "softmax_scale_log2": c["scale"] * 1.4426950408889634,
        "return_lse": 1,
    }
    loader.run_module("attention", values, device_index=torch.cuda.current_device())
    torch.cuda.synchronize()
    assert torch.equal(out, out_api) and torch.equal(lse, lse_api)


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


# Flag macros of the shared SM120 kernel source, in ``SM120_FLAGS`` order.
_SM120_FLAG_MACROS = (
    "CAKE_SAGE_HAS_BLOCK_NUMS",
    "CAKE_SAGE_BLOCK_SIZES_MODE",
    "CAKE_SAGE_FULL_K64_TILES",
    "CAKE_SAGE_UNIFORM_NONEMPTY",
    "CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES",
)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.  Every entry of this family
    is compiled with ``--use_fast_math``, the numerics its kernels were validated
    with.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES)
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    default = loader.MODULES[loader.SM120_RECORDS[(0, 0, 0, 0, 0)]]["library"]
    sm120_sources = [f"csrc/{default}_kernel.cu", f"csrc/{default}_binding.cu"]
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["role"] in loader.ROLES, name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        plan = [tuple(pair) for pair in record["arg_plan"]]
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        for kind, key in plan[:-3]:
            assert kind in ("buffer", "parameter"), (name, kind, key)
            assert record["logical"][key] in (
                loader.LOGICAL_TENSORS + loader.LOGICAL_SCALARS
            ), (name, key)
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            if record["role"] == "attention_sm120":
                # One kernel source and one launcher serve every SM120 specialization;
                # each library compiles them under its own definitions: the kernel
                # symbol, then the five flag macros in select_rule order.
                assert entry["sources"] == sm120_sources
                key = [int(record["select_rule"][flag]) for flag in loader.SM120_FLAGS]
                assert entry["nvcc_flags"] == [
                    "--use_fast_math",
                    f"-DCAKE_SAGE_SM120_KERNEL=kernel_{record['library']}",
                    *(
                        f"-D{macro}={value}"
                        for macro, value in zip(_SM120_FLAG_MACROS, key, strict=True)
                    ),
                ]
            else:
                assert entry["sources"] == [
                    f"csrc/{record['library']}_kernel.cu",
                    f"csrc/{record['library']}_binding.cu",
                ]
                assert entry["nvcc_flags"] == ["--use_fast_math"]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    sm120 = loader.records_of_role("attention_sm120")
    assert len(sm120) == 20 and len(loader.SM120_RECORDS) == 20
    # contiguous_block_indices is specialized only together with uniform_nonempty
    assert all(key[3] >= key[4] for key in loader.SM120_RECORDS)
    assert (
        len(
            {
                libraries[loader.library_name(n, "sm_120a")]["closure_sha256"]
                for n in sm120
            }
        )
        == 20
    )
    assert all(loader.MODULES[name]["arches"] == ["sm_120a"] for name in sm120)
    assert [
        loader.MODULES[name]["arches"] for name in loader.records_of_role("attention")
    ] == [["sm_100a", "sm_103a"]]
