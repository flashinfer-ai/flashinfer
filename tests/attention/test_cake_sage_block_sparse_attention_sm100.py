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
``tests.cake.test_helpers.cake_package`` and nothing else from flashinfer; the
flashinfer public routes are covered by
``test_cake_sage_block_sparse_attention_sm100_api.py``.
"""

import math
import re
import zlib
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.cake_profiler import cuda_kernel_names

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

    FP32 log2 scores (E4M3 products, scaled after the sum); the online softmax of the kernel's
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
                # The products take the E4M3 operands as they are (exact in FP32, TF32 and
                # BF16 inputs) and apply the scales after the sum, as the kernel does; a
                # product of dequantized FP32 operands would round them to TF32 wherever
                # TF32 matmuls are on (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 in NGC-based
                # images), which moves P by far more than _P_ROUNDING_SLACK.
                qq = q[b, h, q0:q1]
                q_log2 = q_scale[b, h, q0:q1, None] * (scale * _LOG2E)
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
                        kk = k[b, kh, tok]
                        vv = v[b, kh, tok]
                        s = (qq @ kk.T) * (q_log2 * k_tok_scale[b, kh, tok][None, :])
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
                vs = v_scale[b, kh][None, :]
                out[b, h, q0:q1] = acc_all * vs / l_all[:, None]
                budget[b, h, q0:q1] = bud_all * vs / l_all[:, None]
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
    kernel parameters, so no upload kernel precedes the attention kernel.  It is the
    PDL build of the record the call routes to by default on SM100/SM103 and that
    record's plain build with ``enable_pdl=False``.  The two-launch quantizer is its
    two kernels behind the zero-fill of its V amax scratch (``zero_``: a torch fill
    kernel, or a memset on torch builds that zero that way); the one-launch quantizer
    is one kernel.  The launch records come from
    ``tests.cake.test_helpers.cake_profiler``, which
    retries a profiler session that lost its CUDA records."""
    device = torch.device("cuda")
    c = _make_case(
        ("one_launch_b2_h4", 2, 4, 2, 257, 1000, 6, True, True, 2, 40, True, 1.0),
        device,
    )
    out = torch.empty_like(c["q8"], dtype=torch.bfloat16)
    lse = torch.empty(c["q_scale"].shape, dtype=torch.float32, device=device)
    index = torch.cuda.current_device()
    arch = loader.device_arch(index)
    tiles = 2 * 4 * _ceil_div(257, _BLOCK)
    for pdl in (True, False):
        role = pkg._route(arch, int(c["index"].shape[-1]), tiles, index, pdl)[0]
        assert role.endswith("_pdl") == pdl, role
        flags = {} if pdl else {"enable_pdl": False}
        names = cuda_kernel_names(
            lambda flags=flags: _run(c, out=out, lse=lse, **flags)
        )
        assert names == [loader.MODULES[role]["kernel_symbol"]], names
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
        lambda: sage_fp8_quantize_sm100(q, k, v, impl="one_launch"),
        exclude=("Memcpy",),
    )
    assert names == [loader.MODULES["quantize_qkv_c8w16"]["kernel_symbol"]], names
    # The TMA-ring quantizer is one kernel too: vsplit=8 where its V clusters fit one wave.
    names = cuda_kernel_names(
        lambda: sage_fp8_quantize_sm100(q, k, v, impl="tma"),
        exclude=("Memcpy",),
    )
    assert names == [loader.MODULES["quantize_qkv_tma_x8"]["kernel_symbol"]], names
    names = cuda_kernel_names(
        lambda: sage_fp8_quantize_sm100(q, k, v, impl="two_launch"),
        exclude=("Memcpy",),
    )
    kernels = [name for name in names if name.startswith("kernel_")]
    assert kernels == [
        loader.MODULES["quantize_qk_vamax_pdl"]["kernel_symbol"],
        loader.MODULES["quantize_v_pdl"]["kernel_symbol"],
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


#: GPU clock cycles the stream tests keep a stream busy before it rewrites the inputs
#: (about 0.1 s): a launch on any other stream runs long before the rewrite.
_SLEEP_CYCLES = 200_000_000


def _stream_cases(device):
    """``{name: (inputs, call)}``: each host entry on small rows, ``call()`` returning
    its output tensors; ``inputs`` are the tensors it reads."""
    c = _make_case(
        ("stream_b1_h4", 1, 4, 4, 320, 640, 5, False, False, 0, None, True, 1.0),
        device,
    )
    g = torch.Generator(device=device).manual_seed(17)
    # small rows take the one-launch quantizer; 8 x 8 x 8 clusters exceed one wave of
    # the device and take the TMA or the two-launch quantizer
    q, k, v = (
        torch.randn((1, 320, 4, _HEAD_DIM), device=device, generator=g).bfloat16()
        for _ in range(3)
    )
    qw, kw, vw = (
        torch.randn((8, 256, 8, _HEAD_DIM), device=device, generator=g).bfloat16()
        for _ in range(3)
    )
    index = c["index"]
    index_wide = torch.zeros((8, 8, 4, 2), dtype=torch.int32, device=device)
    index_wide[..., 1] = 1
    fp8 = [c[n] for n in ("q8", "k8", "v8", "q_scale", "k_scale", "v_scale")]
    scales = dict(q_scale=c["q_scale"], k_scale=c["k_scale"], v_scale=c["v_scale"])
    return {
        "attention": (
            fp8,
            lambda: _run(c),
        ),
        "attention_no_pdl": (
            fp8,
            lambda: _run(c, enable_pdl=False),
        ),
        "dispatch": (
            fp8,
            lambda: pkg.sm100_sage_fwd_cake_dispatch(
                c["q8"],
                c["k8"],
                c["v8"],
                index,
                c["sel"],
                return_lse=True,
                fallback=True,
                **scales,
            ),
        ),
        "dispatch_strict": (
            fp8,
            lambda: pkg.sm100_sage_fwd_cake_dispatch(
                c["q8"], c["k8"], c["v8"], index, c["sel"], **scales
            ),
        ),
        "combined_one_launch": (
            [q, k, v],
            lambda: pkg.bsa_attn_sm100_sage_quantize_fwd_cake(
                q, k, v, index, c["sel"], return_lse=True
            ),
        ),
        "combined_wide": (
            [qw, kw, vw],
            lambda: pkg.bsa_attn_sm100_sage_quantize_fwd_cake(
                qw, kw, vw, index_wide, 2, return_lse=True
            ),
        ),
        "quantize": ([q, k, v], lambda: sage_fp8_quantize_sm100(q, k, v)),
    }


def _outputs(result):
    return [t for t in result if t is not None]


def _after_stream_work(stream, inputs, call, device=None):
    """Run ``call`` on ``stream`` after the stream sleeps and then rewrites ``inputs``
    (zeroed before) with their values; return clones of its outputs taken on the stream.
    A launch on any other stream reads the zeros.  ``device``: the current device while
    ``call`` runs (another device than the stream's)."""
    inputs = [t.view(torch.uint8) for t in inputs]
    real = [t.clone() for t in inputs]
    for t in inputs:
        t.zero_()
    torch.cuda.synchronize(stream.device)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(_SLEEP_CYCLES)
        for t, r in zip(inputs, real, strict=True):
            t.copy_(r)
        if device is None:
            result = call()
        else:
            with torch.cuda.device(device):
                result = call()
        got = [t.clone() for t in _outputs(result)]
    stream.synchronize()
    return got


@requires_sm100
@pytest.mark.parametrize(
    "name",
    [
        "attention",
        "attention_no_pdl",
        "dispatch",
        "dispatch_strict",
        "combined_one_launch",
        "combined_wide",
        "quantize",
    ],
)
def test_host_entries_launch_on_the_current_stream(name):
    """Each host entry launches on torch's current stream: on a side stream it runs
    after the work queued before it there (and writes the bits of the call on the
    default stream), and it captures into a CUDA graph on a side stream."""
    device = torch.device("cuda")
    inputs, call = _stream_cases(device)[name]
    ref = [t.clone() for t in _outputs(call())]  # eager on the default stream
    torch.cuda.synchronize()
    got = _after_stream_work(torch.cuda.Stream(), inputs, call)
    for a, b in zip(got, ref, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        call()  # warm-up on the capture stream
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            captured = _outputs(call())
    for t in captured:
        t.view(torch.uint8).fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, ref, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@requires_sm100
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second CUDA device")
@pytest.mark.parametrize(
    "name",
    ["attention", "dispatch", "combined_one_launch", "combined_wide", "quantize"],
)
def test_host_entries_launch_on_the_current_stream_of_the_inputs_device(name):
    """With another device current, a call on device 1 tensors launches on device 1's
    current stream (a side stream here), as torch kernels do."""
    if not is_cake_sage_sm100_supported(torch.device("cuda", 1)):
        pytest.skip("device 1 is not an SM100 / SM103 device")
    device = torch.device("cuda", 1)
    with torch.cuda.device(device):
        inputs, call = _stream_cases(device)[name]
        ref = [t.clone() for t in _outputs(call())]
        torch.cuda.synchronize()
    got = _after_stream_work(torch.cuda.Stream(device), inputs, call, device=0)
    for a, b in zip(got, ref, strict=True):
        assert a.device == device
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


#: The build of each unsplit case of the routed-bits test: (sel, wide) -> {arch: role}.
#: A narrow case has fewer tiles than half the SMs, a wide one two to two and a half
#: times as many.
_UNSPLIT_ROLES = {
    (5, False): {"sm_100a": "attention", "sm_103a": "attention_sdp3"},
    (6, False): {"sm_100a": "attention_sdp3", "sm_103a": "attention_sdp3"},
    (12, False): {"sm_100a": "attention_sdp3", "sm_103a": "attention"},
    (24, False): {"sm_100a": "attention_sd", "sm_103a": "attention_sd"},
    (32, False): {"sm_100a": "attention_sd", "sm_103a": "attention_sd"},
    (6, True): {"sm_100a": "attention_sdp3", "sm_103a": "attention"},
    (16, True): {"sm_100a": "attention_sdp3", "sm_103a": "attention_p3"},
    (23, True): {"sm_100a": "attention_sdp3", "sm_103a": "attention_sdp3"},
    (40, True): {"sm_100a": "attention_sdp3", "sm_103a": "attention_sdp3"},
}


@requires_sm100
@pytest.mark.parametrize(("sel", "wide"), list(_UNSPLIT_ROLES))
def test_routed_attention_builds_write_the_default_bits(sel, wide, monkeypatch):
    """The builds a call routes to (split QK, double S, p3, and double S with the p3
    epilogue, each in its window of blocks a row and tiles per SM) write the default
    build's bits, with and without PDL, uniform and with per-row counts.  A call that
    takes a tail split (a partial last wave) sums its split tiles in another order: it
    holds the kernel's tolerances against the default build instead, the same bits on
    every call."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    arch = loader.arch_for_capability(torch.cuda.get_device_capability())
    sms = loader.sm_count(index)
    q_blocks = _ceil_div(640, _BLOCK)
    # A wide case has two to two and a half tiles per SM: the p3 band's rows, below the
    # SM103 window of 6 blocks a row from 2.5 tiles per SM.  A narrow case has fewer
    # tiles than half the SMs (the SM103 windows of 5 and 6 blocks a row).  Both hold on
    # a device with fewer SMs, or skip; the head count stays even for the two KV heads.
    if wide:
        heads = 2 * _ceil_div(sms, q_blocks)
        if 2 * heads * q_blocks >= 5 * sms:
            pytest.skip(f"{sms} SMs leave no wide case below 2.5 tiles per SM")
    else:
        heads = next((h for h in (4, 2) if 2 * h * q_blocks < sms), 0)
        if not heads:
            pytest.skip(f"{sms} SMs leave no narrow case below half a tile per SM")
    tiles = heads * q_blocks
    for per_row in (False, True):
        c = _make_case(
            (
                f"route_{sel}_{wide}_{per_row}",
                1,
                heads,
                2,
                640,
                4096,
                sel,
                per_row,
                per_row,
                3,
                33,
                True,
                1.0,
            ),
            device,
        )
        assert 2 * sms <= tiles and 2 * tiles < 5 * sms if wide else 2 * tiles < sms
        # A tail split (from 24 blocks a row) takes the p3 one once the grid holds a
        # tile per SM; SM103 splits without p3 from 32 blocks a row only.
        split = next(
            (
                name
                for p3, name in (
                    (True, "attention_sd_tsplit_p3"),
                    (False, "attention_sd_tsplit"),
                )
                if pkg._tail_split(arch, tiles, sel, index, p3) is not None
            ),
            None,
        )
        assert split is None or sel >= 24
        role = split or _UNSPLIT_ROLES[(sel, wide)][arch]
        for pdl in (False, True):
            record = pkg._route(arch, sel, tiles, index, pdl)[0]
            assert record == (f"{role}_pdl" if pdl else role), record
        names = cuda_kernel_names(lambda: _run(c, enable_pdl=False))
        assert names == [loader.MODULES[role]["kernel_symbol"]], names
        routed = [_run(c, enable_pdl=pdl) for pdl in (False, True)]
        with monkeypatch.context() as m:
            m.setattr(pkg, "_ATTENTION_ROUTES", {})
            m.setattr(pkg, "_P3_BANDS", {})
            m.setattr(pkg, "_SDP3_BANDS", {})
            m.setattr(pkg, "_ROUTES", {})
            names = cuda_kernel_names(lambda: _run(c, enable_pdl=False))
            assert names == [loader.MODULES["attention"]["kernel_symbol"]], names
            out, lse = _run(c, enable_pdl=False)
        torch.cuda.synchronize()
        for o, l in routed:
            if not split:
                assert torch.equal(o, out) and torch.equal(l, lse)
                continue
            assert torch.equal(o, routed[0][0]) and torch.equal(l, routed[0][1])
            error = (o.float() - out.float()).abs()
            assert bool((error <= 1e-2 + 1e-2 * out.float().abs()).all()), error.max()
            finite = torch.isfinite(lse)
            assert torch.equal(torch.isfinite(l), finite)
            torch.testing.assert_close(l[finite], lse[finite], atol=1e-2, rtol=0)


@requires_sm100
@pytest.mark.parametrize("sel", [9, 11])
@pytest.mark.parametrize(
    ("batch", "heads", "kv_heads", "sq"),
    [
        (1, 1, 1, 1536),
        (1, 1, 1, 1472),
        (1, 1, 1, 4800),
        (1, 2, 2, 1536),
        (2, 1, 1, 1500),
        (1, 2, 1, 2368),
    ],
)
def test_sm103_short_grids_route_one_long_row_to_the_p3_double_s_build(
    sel, batch, heads, kv_heads, sq, monkeypatch
):
    """Below half a wave, SM103 runs the double-S build with the p3 epilogue at 9 and 11
    blocks a row on a single (batch, head) row of at least 24 query blocks (the
    optional fields of an ``_SDP3_BANDS`` window) and the default build on a row of 23
    query blocks or on two rows; both write the default build's bits, with and without
    PDL.  A combined call after the one-launch quantizer keeps the default build."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    arch = loader.arch_for_capability(torch.cuda.get_device_capability())
    sms = loader.sm_count(index)
    q_blocks = _ceil_div(sq, _BLOCK)
    rows = batch * heads
    tiles = rows * q_blocks
    if 2 * tiles >= sms:
        pytest.skip(f"{sms} SMs leave no case below half a tile per SM")
    c = _make_case(
        (
            f"long_rows_{sel}_{batch}_{heads}_{kv_heads}_{sq}",
            batch,
            heads,
            kv_heads,
            sq,
            8192,
            sel,
            False,
            False,
            3,
            33,
            True,
            1.0,
        ),
        device,
    )
    sdp3 = q_blocks >= 24 and rows == 1
    want = {"sm_103a": "attention_sdp3" if sdp3 else "attention"}.get(arch)
    for pdl in (False, True):
        record = pkg._route(arch, sel, tiles, index, pdl, q_blocks=q_blocks)[0]
        if want is not None:
            assert record == (f"{want}_pdl" if pdl else want), record
        record = pkg._route(arch, sel, tiles, index, pdl, True, q_blocks)[0]
        if want is not None:
            assert record == ("attention_pdl" if pdl else "attention"), record
    role = pkg._route(arch, sel, tiles, index, False, q_blocks=q_blocks)[0]
    names = cuda_kernel_names(lambda: _run(c, enable_pdl=False))
    assert names == [loader.MODULES[role]["kernel_symbol"]], names
    routed = [_run(c, enable_pdl=pdl) for pdl in (False, True)]
    with monkeypatch.context() as m:
        m.setattr(pkg, "_ATTENTION_ROUTES", {})
        m.setattr(pkg, "_P3_BANDS", {})
        m.setattr(pkg, "_SDP3_BANDS", {})
        m.setattr(pkg, "_ROUTES", {})
        out, lse = _run(c, enable_pdl=False)
    torch.cuda.synchronize()
    for o, l in routed:
        assert torch.equal(o, out) and torch.equal(l, lse)


@requires_sm100
@pytest.mark.parametrize("p3", [True, False])
def test_tail_split_rows_are_deterministic_and_close_to_the_default_build(
    p3, monkeypatch
):
    """A row with at least ``_TAIL_SPLIT``'s blocks whose last wave of tiles is partial
    runs a tail-split build (with the p3 epilogue in ``_TAIL_SPLIT_P3``'s band once the
    grid holds a tile per SM): one kernel, the same bits on every call and in a CUDA
    graph, and within the kernel's tolerance of the unsplit build, whose bits are the
    default build's."""
    device = torch.device("cuda")
    sel = max(40, *pkg._TAIL_SPLIT.values())
    c = _make_case(
        ("tail_split_b1_h8", 1, 8, 8, 4096, 4096, sel, True, True, 0, None, True, 1.0),
        device,
    )
    index = torch.cuda.current_device()
    tiles = 8 * _ceil_div(4096, _BLOCK)
    arch = loader.device_arch(index)
    role = "attention_sd_tsplit_p3" if p3 else "attention_sd_tsplit"
    with monkeypatch.context() as m:
        if not p3:
            m.setattr(pkg, "_TAIL_SPLIT_P3", {})
        m.setattr(pkg, "_ROUTES", {})
        plan = pkg._tail_split(arch, tiles, int(c["index"].shape[-1]), index, p3)
        if plan is None:
            pytest.skip(f"{tiles} tiles leave no tail to split on this device")
        names = cuda_kernel_names(lambda: _run(c, enable_pdl=False))
        assert names == [loader.MODULES[role]["kernel_symbol"]], names
        runs = [_run(c, enable_pdl=pdl) for pdl in (False, True, False)]
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        out_graph = torch.empty_like(runs[0][0])
        lse_graph = torch.empty_like(runs[0][1])
        with torch.cuda.stream(stream):
            _run(c, out=out_graph, lse=lse_graph)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                _run(c, out=out_graph, lse=lse_graph)
        out_graph.zero_()
        graph.replay()
    with monkeypatch.context() as m:
        m.setattr(pkg, "_TAIL_SPLIT", {})
        m.setattr(pkg, "_TAIL_SPLIT_P3", {})
        m.setattr(pkg, "_ROUTES", {})
        unsplit, unsplit_lse = _run(c, enable_pdl=False)
        m.setattr(pkg, "_ATTENTION_ROUTES", {})
        m.setattr(pkg, "_SDP3_BANDS", {})
        m.setattr(pkg, "_ROUTES", {})
        default, default_lse = _run(c, enable_pdl=False)
    torch.cuda.synchronize()
    out, lse = runs[0]
    for o, l in [*runs[1:], (out_graph, lse_graph)]:
        assert torch.equal(o, out) and torch.equal(l, lse)
    assert torch.equal(unsplit, default) and torch.equal(unsplit_lse, default_lse)
    error = (out.float() - default.float()).abs()
    assert bool((error <= 1e-2 + 1e-2 * default.float().abs()).all()), (
        error.max().item()
    )
    finite = torch.isfinite(default_lse)
    assert torch.equal(torch.isinf(lse), ~finite)
    torch.testing.assert_close(lse[finite], default_lse[finite], atol=1e-2, rtol=0)


def test_serial_rule_thresholds():
    """``_TAIL_SPLIT_P3_SERIAL`` with its thresholds written out: after the one-launch
    quantizer, an SM100 plan ``(grid, split_from, items)`` of ``attention_sd_tsplit_p3``
    launches without PDL from three waves of the persistent grid when the split tiles
    fill at most a third of its slots, or up to half of them (the tail split's bound)
    when the call selects at most 47 blocks a row; otherwise, and below three waves,
    it gives way to ``attention_sd_tsplit``.  SM103 keeps PDL."""
    assert pkg._TAIL_SPLIT_P3_SERIAL == {"sm_100a": ((3, 3, None), (3, 2, 47))}
    grid = 148
    for waves, tail, sel, serial in [
        (2, 1, 24, False),
        (2, 44, 32, False),
        (2, 74, 24, False),
        (3, 1, 58, True),
        (3, 24, 127, True),
        (3, 49, 58, True),
        (3, 50, 24, True),
        (3, 50, 47, True),
        (3, 50, 48, False),
        (3, 56, 32, True),
        (3, 61, 48, False),
        (3, 74, 47, True),
        (3, 74, 58, False),
        (4, 68, 40, True),
        (4, 68, 58, False),
        (7, 8, 96, True),
    ]:
        split_from = (waves - 1) * grid
        plan = (grid, split_from, split_from + 2 * tail)
        assert pkg._p3_serial("sm_100a", plan, sel) is serial, (waves, tail, sel)
        assert pkg._p3_serial("sm_103a", plan, sel) is None


@requires_sm100
@pytest.mark.parametrize(
    ("heads", "sel"),
    [(3, 40), (5, 40), (7, 40), (8, 40), (3, 24), (5, 28), (8, 24)],
)
def test_combined_call_launches_the_p3_tail_split_by_the_serial_rule(heads, sel):
    """A combined call whose quantizer is the one-launch program and whose attention
    routes to ``attention_sd_tsplit_p3`` follows ``_TAIL_SPLIT_P3_SERIAL``: the build
    without programmatic dependent launch (its PDL flag is 0) where a rule holds,
    ``attention_sd_tsplit`` with PDL where it gives way, the build with PDL on the other
    rows, on SM103 and outside a combined call; the bits are those of the attention call
    on the quantizer's outputs."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    arch = loader.device_arch(index)
    sms = loader.sm_count(index)
    seqlen = 4096
    tiles = heads * _ceil_div(seqlen, _BLOCK)
    plan = pkg._tail_split(arch, tiles, sel, index, True)
    if 8 * heads > sms or plan is None:
        pytest.skip(f"{sms} SMs leave {heads} heads no one-launch p3 tail split")
    g = torch.Generator(device=device).manual_seed(5)
    q, k, v = (
        torch.randn(
            (1, seqlen, heads, _HEAD_DIM), device=device, generator=g
        ).bfloat16()
        for _ in range(3)
    )
    nb = seqlen // _BLOCK
    scores = torch.rand((1, heads, nb, nb), device=device, generator=g)
    idx = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    _, (q8, k8, v8, qs, ks, vs), impl = pkg._quantize_program(q, k, v, "auto", None)
    assert impl == "one_launch", impl
    rule = pkg._p3_serial(arch, plan, sel)
    for after in (False, True):
        record, _, args, _, _ = pkg._attention_operands(
            q8,
            k8,
            v8,
            idx,
            sel,
            qs,
            ks,
            vs,
            None,
            None,
            None,
            False,
            None,
            None,
            None,
            operands_checked=True,
            after_one_launch=after,
        )
        expect = "attention_sd_tsplit_p3_pdl", 1
        if after and rule is not None:
            expect = (
                ("attention_sd_tsplit_p3_pdl", 0)
                if rule
                else ("attention_sd_tsplit_pdl", 1)
            )
        assert (record, args[-4]) == expect, (record, args[-4:])
    out, _ = pkg.bsa_attn_sm100_sage_quantize_fwd_cake(q, k, v, idx, sel)
    q8, k8, v8, qs, ks, vs = sage_fp8_quantize_sm100(q, k, v)
    ref, _ = bsa_attn_sm100_sage_fwd_cake(
        q8, k8, v8, idx, sel, q_scale=qs, k_scale=ks, v_scale=vs
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_after_one_launch_rule_thresholds():
    """``_SDP3_GIVE_WAY`` with its thresholds written out (SM103, 152 SMs):
    ``attention_sdp3`` is given up to 22 selected blocks a row below three waves.  SM100
    has no such rule, and SM103 has no ``_TAIL_SPLIT_P3_SERIAL`` rule: its p3 tail
    splits keep PDL behind the one-launch quantizer."""
    assert pkg._TAIL_SPLIT_P3_SERIAL == {"sm_100a": ((3, 3, None), (3, 2, 47))}
    assert pkg._SDP3_GIVE_WAY == {"sm_103a": (22, 3)}
    for sel, tiles, gives in [
        (5, 32, True),
        (6, 455, True),
        (22, 455, True),
        (6, 456, False),
        (23, 152, False),
    ]:
        assert pkg._sdp3_gives_way("sm_103a", sel, tiles, 152) is gives, (sel, tiles)
        assert pkg._sdp3_gives_way("sm_100a", sel, tiles, 148) is False


@requires_sm100
@pytest.mark.parametrize(("heads", "sel"), [(1, 5), (4, 6), (3, 6), (8, 6), (4, 22)])
def test_combined_call_gives_up_sdp3_by_the_give_way_rule(heads, sel):
    """After the one-launch quantizer a combined call whose attention routes to an
    unsplit ``attention_sdp3`` row runs ``attention_p3`` or the default build with PDL
    where ``_SDP3_GIVE_WAY`` gives the build up, else the same build; the bits are those
    of the attention call on the quantizer's outputs on every call."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    arch = loader.device_arch(index)
    sms = loader.sm_count(index)
    seqlen = 4096
    tiles = heads * _ceil_div(seqlen, _BLOCK)
    if 8 * heads > sms:
        pytest.skip(f"{sms} SMs leave {heads} heads no one-launch quantizer")
    g = torch.Generator(device=device).manual_seed(7)
    q, k, v = (
        torch.randn(
            (1, seqlen, heads, _HEAD_DIM), device=device, generator=g
        ).bfloat16()
        for _ in range(3)
    )
    nb = seqlen // _BLOCK
    scores = torch.rand((1, heads, nb, nb), device=device, generator=g)
    idx = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    _, (q8, k8, v8, qs, ks, vs), impl = pkg._quantize_program(q, k, v, "auto", None)
    assert impl == "one_launch", impl
    routes = {}
    for after in (False, True):
        record, _, args, _, _ = pkg._attention_operands(
            q8,
            k8,
            v8,
            idx,
            sel,
            qs,
            ks,
            vs,
            None,
            None,
            None,
            False,
            None,
            None,
            None,
            operands_checked=True,
            after_one_launch=after,
        )
        routes[after] = record, args[-4]
    plain, comb = routes[False], routes[True]
    if plain[0] != "attention_sdp3_pdl":
        pytest.skip(f"{arch} routes {heads} heads at {sel} blocks to {plain[0]}")
    if pkg._sdp3_gives_way(arch, sel, tiles, sms):
        assert comb in (("attention_pdl", 1), ("attention_p3_pdl", 1)), comb
    else:
        assert comb == plain, (comb, plain)
    out, lse = pkg.bsa_attn_sm100_sage_quantize_fwd_cake(
        q, k, v, idx, sel, return_lse=True
    )
    again, again_lse = pkg.bsa_attn_sm100_sage_quantize_fwd_cake(
        q, k, v, idx, sel, return_lse=True
    )
    q8, k8, v8, qs, ks, vs = sage_fp8_quantize_sm100(q, k, v)
    ref, ref_lse = bsa_attn_sm100_sage_fwd_cake(
        q8, k8, v8, idx, sel, q_scale=qs, k_scale=ks, v_scale=vs, return_lse=True
    )
    torch.cuda.synchronize()
    assert torch.equal(out, again) and torch.equal(lse, again_lse)
    assert torch.equal(out, ref) and torch.equal(lse, ref_lse)


@requires_sm100
@pytest.mark.parametrize(
    ("sel", "wide"),
    [
        (5, False),
        (6, False),
        (12, False),
        (24, False),
        (40, False),
        (6, True),
        (16, True),
        (23, True),
        (40, True),
    ],
)
def test_attention_builds_write_the_same_bits_to_a_16_byte_aligned_out(
    sel, wide, monkeypatch
):
    """``out`` may start on any 16-byte boundary: a contiguous view eight BF16 elements
    into a buffer (16 bytes past a 32-byte boundary) gets the bits of a fresh, 32-byte
    aligned ``out`` from every build a call routes to (default, split QK, double S, its
    tail split, p3, double S with the p3 epilogue and its tail split), with and without
    PDL, and from the default build; nothing outside the view is written."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    arch = loader.arch_for_capability(torch.cuda.get_device_capability())
    sms = loader.sm_count(index)
    sq = 640
    q_blocks = _ceil_div(sq, _BLOCK)
    heads = 2 * _ceil_div(sms, q_blocks) if wide else 4
    c = _make_case(
        (
            f"out16_{sel}_{wide}",
            1,
            heads,
            2,
            sq,
            4096,
            sel,
            True,
            True,
            3,
            33,
            True,
            1.0,
        ),
        device,
    )
    shape = (1, sq, heads, _HEAD_DIM)
    numel = math.prod(shape)
    records = []
    for routed in (True, False):
        with monkeypatch.context() as m:
            if not routed:
                m.setattr(pkg, "_ATTENTION_ROUTES", {})
                m.setattr(pkg, "_P3_BANDS", {})
                m.setattr(pkg, "_SDP3_BANDS", {})
            m.setattr(pkg, "_ROUTES", {})
            for pdl in (False, True):
                records.append(pkg._route(arch, sel, heads * q_blocks, index, pdl)[0])
                ref, ref_lse = _run(c, enable_pdl=pdl)
                assert ref.data_ptr() % 32 == 0
                buf = torch.full(
                    (numel + 16,), float("nan"), dtype=torch.bfloat16, device=device
                )
                out = buf[8 : 8 + numel].view(shape)
                assert out.data_ptr() % 32 == 16
                o, lse = _run(c, enable_pdl=pdl, out=out)
                torch.cuda.synchronize()
                assert o.data_ptr() == out.data_ptr()
                assert torch.equal(o, ref) and torch.equal(lse, ref_lse), records[-1]
                assert bool(buf[:8].isnan().all()) and bool(
                    buf[8 + numel :].isnan().all()
                )
    assert records[2:] == ["attention", "attention_pdl"], records


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
        "q": c["q8"],
        "k": c["k8"],
        "v": c["v8"],
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
    # The attention bindings take float8_e4m3fn Q/K/V and refuse other dtypes by name,
    # before any launch.
    out.fill_(float("nan"))
    for name in ("q", "k", "v"):
        for dtype in (torch.uint8, torch.int8):
            bad = {**values, name: values[name].view(dtype)}
            with pytest.raises(TypeError, match=f"{name} dtype mismatch: expected "):
                loader.run_module(
                    "attention", bad, device_index=torch.cuda.current_device()
                )
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()


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


def _quantizer_inputs(batch, heads, kv_heads, sq, sk, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((batch, sq, heads, _HEAD_DIM), device=device, generator=g)
    k = torch.randn((batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g)
    v = torch.randn((batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g)
    v[..., 3] = 0  # a zero V channel: its scale is the amax floor
    return q.bfloat16(), k.bfloat16(), (v * 3).bfloat16()


@requires_sm100
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 8, 4096, 4096),  # vsplit=4, the longest sequence
        (1, 2, 2, 4096, 4096),  # vsplit=8
        (1, 32, 8, 4096, 4096),  # GQA, vsplit=4
        (2, 4, 4, 1000, 1500),  # ragged tails
        (1, 8, 2, 65, 257),  # tails shorter than a unit and a V box
        (1, 1, 1, 16384, 2048),  # long Q, short K/V: Q/K units on every cluster
        (3, 2, 2, 300, 17),
        (8, 8, 8, 512, 512),  # more V clusters than one wave (impl="tma" only)
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_tma_quantizer_matches_two_launches(shape, enable_pdl):
    """The TMA-ring quantizer writes the bits of the two-launch quantizer."""
    device = torch.device("cuda")
    batch, heads, kv_heads, sq, sk = shape
    q, k, v = _quantizer_inputs(batch, heads, kv_heads, sq, sk, device, sq * 7 + sk)
    ref = sage_fp8_quantize_sm100(q, k, v, impl="two_launch", enable_pdl=False)
    for _ in range(2):
        got = sage_fp8_quantize_sm100(q, k, v, impl="tma", enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        for name, a, b in zip(
            ("q8", "k8", "v8", "q_scale", "k_scale", "v_scale"), got, ref, strict=True
        ):
            assert a.shape == b.shape, name
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name


@requires_sm100
def test_tma_quantizer_refuses_sequences_beyond_its_v_boxes():
    device = torch.device("cuda")
    q, k, v = _quantizer_inputs(1, 2, 2, 64, 4097, device, 5)
    with pytest.raises(ValueError, match='impl="tma"'):
        sage_fp8_quantize_sm100(q, k, v, impl="tma")


@requires_sm100
def test_tma_quantizer_refuses_partitions_beyond_32_bits():
    """The TMA kernel multiplies its CTA count by its Q/K tile count in 32-bit integers:
    ``impl="tma"`` refuses a shape whose product reaches ``2**31``, and ``"auto"`` takes
    another quantizer for it."""
    route = ("quantize_qkv_tma_x4", 4, 37)
    assert pkg._tma_partition_fits(1, 8, 8, 4096, 4096, route)
    assert not pkg._tma_partition_fits(64, 128, 128, 64, 64, route)
    assert not pkg._tma_partition_fits(1, 8, 8, 1 << 27, 4096, route)
    q, k, v = _quantizer_inputs(64, 128, 128, 64, 64, torch.device("cuda"), 5)
    with pytest.raises(ValueError, match="32-bit"):
        sage_fp8_quantize_sm100(q, k, v, impl="tma")


@requires_sm100
def test_auto_quantizer_capture_without_an_eager_call_takes_two_launches():
    """The TMA route's occupancy query cannot run inside a CUDA-graph capture: a capture
    of a row no eager call has routed takes the two-launch quantizer (same bits), the
    next eager call queries the occupancy, and a later capture takes the TMA quantizer;
    ``impl="tma"`` in that first capture raises before anything is captured."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    arch = loader.device_arch(index)
    q, k, v = _quantizer_inputs(1, 8, 8, 4096, 4096, device, 5)
    ref = sage_fp8_quantize_sm100(q, k, v, impl="two_launch", enable_pdl=False)
    sage_fp8_quantize_sm100(q, k, v, impl="two_launch")  # load the two-launch kernels
    torch.cuda.synchronize()

    def capture(fn):
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            calls, _ = pkg._quantize_calls(q, k, v, "auto", None)
            got = fn()
        for t in got:
            t.view(torch.uint8).fill_(255)
        graph.replay()
        torch.cuda.synchronize()
        return [run for run, _ in calls], got

    pkg._TMA_RESIDENT.clear()
    runs, got = capture(lambda: sage_fp8_quantize_sm100(q, k, v))
    assert loader.entry("quantize_qkv_tma_x4", arch) not in runs
    for a, b in zip(got, ref, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="needs one eager call first"),
        torch.cuda.stream(stream),
        torch.cuda.graph(graph, stream=stream),
    ):
        sage_fp8_quantize_sm100(q, k, v, impl="tma")
    torch.cuda.synchronize()
    sage_fp8_quantize_sm100(q, k, v)  # eager: the cluster quantizer, and the query
    assert ("quantize_qkv_tma_x4", index) in pkg._TMA_RESIDENT
    runs, got = capture(lambda: sage_fp8_quantize_sm100(q, k, v))
    assert runs == [loader.entry("quantize_qkv_tma_x4", arch)]
    for a, b in zip(got, ref, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@requires_sm100
def test_auto_quantizer_routes_captured_small_rows_to_the_tma_quantizer():
    """``impl="auto"``: eager small rows take the cluster quantizer; inside a CUDA-graph
    capture the rows whose TMA V clusters fit one wave take the TMA quantizer (one kernel
    node, no zero-fill), ``vsplit=8`` where that fits, and the replay writes the
    two-launch bits."""
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    x8 = loader.resident_clusters(loader.MODULES["quantize_qkv_tma_x8"], index)
    x4 = loader.resident_clusters(loader.MODULES["quantize_qkv_tma_x4"], index)
    assert x8 >= 16 and x4 >= 16, (x8, x4)
    route = pkg._tma_route
    assert route(1, 2, 4096, 4096, index, one_wave=True)[:2] == (
        "quantize_qkv_tma_x8",
        8,
    )
    assert route(1, 8, 4096, 4096, index, one_wave=True)[:2] == (
        "quantize_qkv_tma_x4",
        4,
    )
    assert route(1, 8, 4096, 4097, index, one_wave=True) is None
    assert route(8, 8, 4096, 4096, index, one_wave=True) is None
    assert route(8, 8, 4096, 4096, index, one_wave=False)[:2] == (
        "quantize_qkv_tma_x4",
        4,
    )
    for shape, record in (
        ((1, 2, 2, 4096, 4096), "quantize_qkv_tma_x8"),
        ((1, 8, 8, 4096, 4096), "quantize_qkv_tma_x4"),
    ):
        q, k, v = _quantizer_inputs(*shape, device, 3)
        ref = sage_fp8_quantize_sm100(q, k, v, impl="two_launch", enable_pdl=False)
        calls, _ = pkg._quantize_calls(q, k, v, "auto", None)
        assert [run for run, _ in calls] == [
            loader.entry("quantize_qkv_c8w16", loader.device_arch(index))
        ]
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            sage_fp8_quantize_sm100(q, k, v)  # warm-up on the capture stream
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                calls, _ = pkg._quantize_calls(q, k, v, "auto", None)
                got = sage_fp8_quantize_sm100(q, k, v)
        assert [run for run, _ in calls] == [
            loader.entry(record, loader.device_arch(index))
        ]
        graph.replay()
        torch.cuda.synchronize()
        for name, a, b in zip(
            ("q8", "k8", "v8", "q_scale", "k_scale", "v_scale"), got, ref, strict=True
        ):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name


@requires_sm100
def test_run_module_launches_the_tma_quantizer_like_the_host_api():
    """``run_module`` grids a ``vsplit_clusters`` record as the host API does."""
    device = torch.device("cuda")
    batch, heads, kv_heads, sq, sk = 1, 4, 2, 300, 2000
    q, k, v = _quantizer_inputs(batch, heads, kv_heads, sq, sk, device, 9)
    ref = sage_fp8_quantize_sm100(q, k, v, impl="tma")
    out = [torch.empty_like(t) for t in ref]
    values = {
        "q": q,
        "k": k,
        "v": v,
        "q_out": out[0],
        "k_out": out[1],
        "v_out": out[2],
        "q_scale": out[3],
        "k_scale": out[4],
        "v_scale": out[5],
        "batch": batch,
        "seqlen_q": sq,
        "seqlen_k": sk,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "k_scale_groups": _ceil_div(sk, _K_GROUP),
        "q_chunks": _ceil_div(sq, 64),
        "k_chunks": _ceil_div(sk, 64),
        "v_boxes": _ceil_div(_ceil_div(sk, 4), 256),
        "enable_pdl": True,
    }
    index = torch.cuda.current_device()
    for name in ("quantize_qkv_tma_x4", "quantize_qkv_tma_x8"):
        for t in out:
            t.view(torch.uint8).fill_(255)
        loader.run_module(name, values, device_index=index)
        torch.cuda.synchronize()
        for a, b in zip(out, ref, strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name


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
        # A PDL record takes the per-call ``enable_pdl`` argument right before the grid.
        pdl = record["launch"]["use_pdl"] == "runtime"
        assert record["launch"]["use_pdl"] in (False, "runtime"), name
        assert (plan[-4] == ("launch", "enable_pdl")) == pdl, name
        for kind, key in plan[: -4 if pdl else -3]:
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
                # An architecture whose build launches with other dynamic shared memory
                # than the record's first architecture compiles its own binding.
                own = f"csrc/{record['library']}_{arch}_binding.cu"
                assert entry["sources"] == [
                    f"csrc/{record['library']}_kernel.cu",
                    own
                    if (package / own).is_file()
                    else f"csrc/{record['library']}_binding.cu",
                ]
                assert entry["nvcc_flags"] == ["--use_fast_math"]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
    assert referenced == set(libraries)
    # The SM103 double-S and p3 builds keep eight K/V stages (six on SM100): their
    # SM103 libraries compile their own binding with the larger dynamic shared memory.
    assert sorted(
        path.name for path in (package / "csrc").glob("*_sm_10*_binding.cu")
    ) == sorted(
        f"{loader.MODULES[name]['library']}_sm_103a_binding.cu"
        for name in (
            "attention_sd",
            "attention_sd_pdl",
            "attention_sd_tsplit",
            "attention_sd_tsplit_pdl",
            "attention_p3",
            "attention_p3_pdl",
            "attention_sdp3",
            "attention_sdp3_pdl",
            "attention_sd_tsplit_p3",
            "attention_sd_tsplit_p3_pdl",
        )
    )
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
    # The default attention build (and its PDL build) ships on both architectures; the
    # second build of each architecture (split QK on SM100, per tile on SM103) on its own;
    # the double-S build, its tail-split build, the p3 build and the double-S build with
    # the p3 epilogue and its tail-split build on both.
    assert {
        name: loader.MODULES[name]["arches"]
        for name in loader.records_of_role("attention")
    } == {
        "attention": ["sm_100a", "sm_103a"],
        "attention_pdl": ["sm_100a", "sm_103a"],
        "attention_qs": ["sm_100a"],
        "attention_qs_pdl": ["sm_100a"],
        "attention_per_tile": ["sm_103a"],
        "attention_per_tile_pdl": ["sm_103a"],
        "attention_sd": ["sm_100a", "sm_103a"],
        "attention_sd_pdl": ["sm_100a", "sm_103a"],
        "attention_sd_tsplit": ["sm_100a", "sm_103a"],
        "attention_sd_tsplit_pdl": ["sm_100a", "sm_103a"],
        "attention_p3": ["sm_100a", "sm_103a"],
        "attention_p3_pdl": ["sm_100a", "sm_103a"],
        "attention_sdp3": ["sm_100a", "sm_103a"],
        "attention_sdp3_pdl": ["sm_100a", "sm_103a"],
        "attention_sd_tsplit_p3": ["sm_100a", "sm_103a"],
        "attention_sd_tsplit_p3_pdl": ["sm_100a", "sm_103a"],
    }
    assert all(
        loader.MODULES[name]["launch"]["cluster"] == (2, 1, 1)
        and loader.MODULES[name]["grid_rule"] == {"kind": "tail_split_clusters"}
        for name in (
            "attention_sd_tsplit",
            "attention_sd_tsplit_pdl",
            "attention_sd_tsplit_p3",
            "attention_sd_tsplit_p3_pdl",
        )
    )
