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

Tests of the experimental FP8 attention backend package (SM90): BF16 -> BF16
correctness against the dequantized-operand oracle of the FP8 recipe over the
operand regimes the kernels were validated on (unit normal, V x 8, one-hot
rows, outlier channels of Q and K or of Q alone, a flat softmax with a V
channel mean), tails (``S % 128 != 0``), ``S < 128``, ``S == 1``, ``B > 1``,
``H`` in {1, 5, 8, 12, 40} and both storage orders of the logical
``[B, H, S, 128]`` view; the quantized operands the chain leaves in the
workspace against the recipe in torch; repeatability and value freedom between
launches; CUDA-graph replay; the workspace contract; the host plan of the
registry records and the ``build.json`` contract.

The oracle (:func:`oracle`) is the recipe's gate-(a) contract: the statistics,
the INT8 and E4M3 codes and the scales computed in torch with the kernels'
operations in the kernels' order (IEEE division, one FP32 product,
round-half-even INT8 codes, round-to-nearest-even saturating E4M3), then exact
FP32 softmax attention over the dequantized operands, no P quantization and no
two-level fold.  The kernels agree with it within ``atol 1.0, rtol 0.1``, the
tolerance of the kernels' own unit tests.

This suite exercises the backend package alone: it names the package once,
in the module constant ``PACKAGE``, and imports nothing else from flashinfer,
so a build that vendors the package under another import path runs it
unchanged after pointing that constant at its package.  The
``flashinfer.prefill`` public entries are covered by
``test_cake_fp8_attention_api.py``.
"""

import importlib
import math
import re
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PACKAGE = "flashinfer.experimental.cake_fp8_attention"

pkg = importlib.import_module(PACKAGE)
backend = importlib.import_module(f"{PACKAGE}.cake_backend")
loader = importlib.import_module(f"{PACKAGE}.cake_jit")
cake_build = importlib.import_module(f"{PACKAGE}.cake_build")

HEAD_DIM = 128
BLOCK = 128  # Q scale block == keys per attention tile == statistics tile
K_HALF = 64  # keys per K scale
ROWS_PER_CTA = 64  # the attention's Q rows per CTA
ATOL, RTOL = 1.0, 0.1
OUTLIER_CHANNELS = (3, 40, 77, 120)
REGIMES = ("randn", "v_x8", "onehot", "outlier32", "outlier32_q", "spread")
# The workspace regions in carving order (the ``stats_partials`` record's ``workspace_rule``).
WORKSPACE_REGIONS = [
    "partials",
    "mean_k",
    "v_scale",
    "q8",
    "k8",
    "vt8",
    "q_scale",
    "k_scale",
]
# Entries of ``build.json`` carry the compile options the program's kernel was validated with: nvcc's defaults
# alone on every program of this chain.
NVCC_FLAGS_BY_ROLE = {
    "stats_partials": [],
    "stats_finalize": [],
    "quantize": [],
    "attention": [],
}
# The recipe's constants.
E4M3_MAX = 448.0
INT8_MAX = 127.0
AMAX_FLOOR = 1.0e-12
# Hardware column ``16 h + 4 t + e`` of a 32-key chunk of the transposed V holds key ``KEY_PERM_32[16 h + 4 t + e]``.
KEY_PERM_32 = tuple(
    16 * h + 8 * (e >> 1) + 2 * t + (e & 1)
    for h in (0, 1)
    for t in range(4)
    for e in range(4)
)


def _capability():
    return (
        tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    )


def _arch():
    capability = _capability()
    return loader.arch_for_capability(capability) if capability is not None else None


def _skip_unless_supported():
    if _arch() is None:
        pytest.skip("an SM90 GPU is required")


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries when they have no prebuilt file before the first test, through the
    # spec objects the package's loader itself builds on first use; prebuilt or cached libraries load without
    # compiling.
    if _arch() is not None:
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


# --- Inputs ------------------------------------------------------------------------


def make_inputs(
    batch, heads, seqlen, *, operands="randn", layout="HND", seed=0, device="cuda"
):
    """BF16 ``q``, ``k``, ``v`` of one regime as logical ``[B, H, S, 128]`` views: ``HND`` storage is the view
    itself, ``NHD`` storage is a ``[B, S, H, 128]`` buffer viewed through ``transpose(1, 2)``."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seqlen, HEAD_DIM)
    q, k, v = (torch.randn(shape, generator=generator) for _ in range(3))
    if operands == "v_x8":
        v *= 8.0
    elif operands == "onehot":
        for b in range(batch):
            for h in range(heads):
                keys = torch.randperm(seqlen, generator=generator)
                q[b, h] = k[b, h][keys] * 8.0
    elif operands == "outlier32":
        q[..., OUTLIER_CHANNELS] *= 32.0
        k[..., OUTLIER_CHANNELS] *= 32.0
    elif operands == "outlier32_q":
        q[..., OUTLIER_CHANNELS] *= 32.0
    elif operands == "spread":
        # A flat softmax (every key carries weight) over a V with a channel mean: the long-row accumulation.
        q *= 0.1
        k *= 0.1
        v += 1.0
    elif operands != "randn":
        raise ValueError(f"unknown operand regime {operands!r}")
    tensors = []
    for x in (q, k, v):
        x = x.to(dtype=torch.bfloat16, device=device)
        if layout == "NHD":
            # L90 test-side fix: ``transpose(1, 2).contiguous()`` is a no-op when ``heads == 1`` (torch treats the
            # unit extent as contiguous), so build the [B, S, H, 128] buffer explicitly and view it.
            buffer = torch.empty(
                (batch, seqlen, heads, HEAD_DIM), dtype=x.dtype, device=x.device
            )
            buffer.transpose(1, 2).copy_(x)
            x = buffer.transpose(1, 2)
            assert x.stride(1) == HEAD_DIM and x.stride(2) == heads * HEAD_DIM
        tensors.append(x)
    return tuple(tensors)


def _empty_like_layout(x, layout):
    if layout == "NHD":
        return torch.empty_like(x.transpose(1, 2)).transpose(1, 2)
    return torch.empty_like(x)


# --- The recipe in torch and its oracle --------------------------------------------


def _ceil_div(a, b):
    return -(-a // b)


def _e4m3(x):
    """Round to nearest even, saturating, to E4M3 codes held as ``uint8`` (the kernels' conversion)."""
    return (
        x.float()
        .nan_to_num(nan=0.0)
        .clamp_(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


def _decode(codes):
    """The FP32 values of E4M3 codes held as ``uint8``."""
    return codes.view(torch.float8_e4m3fn).float()


def _int8_codes(x):
    """``rint(clamp(x, -127, 127))`` -- round half to even, the kernels' magic add -- as two's-complement bytes
    held as ``uint8``."""
    return (
        torch.round(x.float().clamp(-INT8_MAX, INT8_MAX))
        .to(torch.int8)
        .view(torch.uint8)
    )


def _decode_int8(codes):
    """The FP32 values of INT8 codes held as ``uint8``."""
    return codes.view(torch.int8).float()


def _fdiv(numerator, denominator):
    """IEEE FP32 division with both operands as tensors (torch folds a scalar divisor into a reciprocal
    product; the kernels divide)."""
    if not torch.is_tensor(numerator):
        numerator = torch.full_like(denominator, float(numerator))
    if not torch.is_tensor(denominator):
        denominator = torch.full_like(numerator, float(denominator))
    return torch.div(numerator, denominator)


def _vt_position(s_pad, device):
    """``pos[key]``: the column of the transposed-V tile that holds ``key`` (:data:`KEY_PERM_32` inside each
    32-key chunk)."""
    inverse = [0] * 32
    for column, key in enumerate(KEY_PERM_32):
        inverse[key] = column
    key = torch.arange(s_pad, device=device)
    return key - key % 32 + torch.tensor(inverse, device=device)[key % 32]


def recipe_statistics(k, v):
    """Stages 1 and 2 of the recipe in torch on logical ``[B, H, S, 128]`` BF16 views: per 128-row tile the channel
    sums of K (FP64, rounded to FP32; the kernel sums in FP32 in a fixed order, so the sums agree to that order's
    rounding) and the channel maxima of |V| (exact), then the tiles folded in order in FP32 into the K channel mean
    ``mean_k = sum / S`` and the V channel scale ``v_scale = max(amax |V|, 1e-12) / 448`` per (batch, head).
    Returns ``(partials [B * H, N, 2, 128], mean_k [B * H, 128], v_scale [B * H, 128])``."""
    batch, heads, seqlen, _dim = (int(x) for x in k.shape)
    bh = batch * heads
    nqb = _ceil_div(seqlen, BLOCK)
    device = k.device
    kf = k.reshape(bh, seqlen, HEAD_DIM).float()
    vf = v.reshape(bh, seqlen, HEAD_DIM).float()
    partials = torch.zeros(bh, nqb, 2, HEAD_DIM, device=device)
    for tile in range(nqb):
        rows = slice(tile * BLOCK, min((tile + 1) * BLOCK, seqlen))
        partials[:, tile, 0] = kf[:, rows].double().sum(dim=1).float()
        partials[:, tile, 1] = vf[:, rows].abs().amax(dim=1)
    ksum = torch.zeros(bh, HEAD_DIM, device=device)
    vmax = torch.zeros_like(ksum)
    for tile in range(nqb):
        ksum = ksum + partials[:, tile, 0]
        vmax = torch.maximum(vmax, partials[:, tile, 1])
    mean_k = _fdiv(ksum, float(seqlen))
    v_scale = _fdiv(vmax.clamp_min(AMAX_FLOOR), E4M3_MAX)
    return partials, mean_k, v_scale


def quantize_operands(q, k, v, mean_k, v_scale):
    """Stage 3 of the recipe in torch with the kernel's operations in the kernel's order, from the given
    statistics (torch's, or the kernel's own read back from the workspace): Q is not centred and gets INT8 codes
    with one scale ``amax / 127`` per 128-row block (the amax over the block's rows and channels); K is centred by
    ``mean_k`` and gets INT8 codes with one scale ``amax / 127`` per 64-key half (``k_scale[2 * tile + half]``);
    V is divided by its channel scale and gets E4M3 codes transposed to ``[B * H, 128, S_pad]`` with the keys of
    every 32-key chunk permuted by :data:`KEY_PERM_32`.  Rows and keys past ``S`` are zero codes.  Codes as
    ``uint8``, scales as FP32."""
    batch, heads, seqlen, _dim = (int(x) for x in q.shape)
    bh = batch * heads
    nqb = _ceil_div(seqlen, BLOCK)
    s_pad = nqb * BLOCK
    pad = s_pad - seqlen
    device = q.device
    qf = F.pad(q.reshape(bh, seqlen, HEAD_DIM).float(), (0, 0, 0, pad))  # zero past S
    kf = F.pad(k.reshape(bh, seqlen, HEAD_DIM).float(), (0, 0, 0, pad))
    vf = v.reshape(bh, seqlen, HEAD_DIM).float()
    # Q: not centred, one amax per 128-row block.
    q_blocks = qf.view(bh, nqb, BLOCK, HEAD_DIM)
    q_amax = q_blocks.abs().amax(dim=(2, 3)).clamp_min(AMAX_FLOOR)
    q_scale = _fdiv(q_amax, INT8_MAX)  # [bh, nqb]
    q8 = _int8_codes(q_blocks * _fdiv(INT8_MAX, q_amax)[:, :, None, None]).view(
        bh, s_pad, HEAD_DIM
    )
    # K: centred per (batch, head), one amax per 64-key half.
    kc = kf - mean_k[:, None, :]
    kc[:, seqlen:] = 0.0
    k_halves = kc.view(bh, 2 * nqb, K_HALF, HEAD_DIM)
    k_amax = k_halves.abs().amax(dim=(2, 3)).clamp_min(AMAX_FLOOR)
    k_scale = _fdiv(k_amax, INT8_MAX)  # [bh, 2 nqb]
    k8 = _int8_codes(k_halves * _fdiv(INT8_MAX, k_amax)[:, :, None, None]).view(
        bh, s_pad, HEAD_DIM
    )
    # V: one scale per channel, transposed, keys permuted inside each 32-key chunk, zero past S.
    vt = torch.zeros(bh, HEAD_DIM, s_pad, device=device)
    vt[:, :, _vt_position(s_pad, device)[:seqlen]] = _fdiv(
        vf, v_scale[:, None, :]
    ).permute(0, 2, 1)
    vt8 = _e4m3(vt)
    return types.SimpleNamespace(
        seqlen=seqlen,
        s_pad=s_pad,
        nqb=nqb,
        mean_k=mean_k,
        v_scale=v_scale,
        q8=q8,
        q_scale=q_scale,
        k8=k8,
        k_scale=k_scale,
        vt8=vt8,
    )


def recipe_operands(q, k, v):
    """The complete recipe in torch: the statistics of :func:`recipe_statistics`, then the operands of
    :func:`quantize_operands` from them (the namespace also carries ``partials``)."""
    partials, mean_k, v_scale = recipe_statistics(k, v)
    ops = quantize_operands(q, k, v, mean_k, v_scale)
    ops.partials = partials
    return ops


def _dequantized(ops, bh):
    """``Qd``, ``Kd`` and ``Vd`` of one (batch, head) in FP32, real rows and keys only."""
    seqlen, s_pad = ops.seqlen, ops.s_pad
    device = ops.q8.device
    qd = (
        _decode_int8(ops.q8[bh, :seqlen])
        * ops.q_scale[bh].repeat_interleave(BLOCK)[:seqlen, None]
    )
    kd = (
        _decode_int8(ops.k8[bh, :seqlen])
        * ops.k_scale[bh].repeat_interleave(K_HALF)[:seqlen, None]
    )
    vd = (
        _decode(ops.vt8[bh])[:, _vt_position(s_pad, device)[:seqlen]].T
        * ops.v_scale[bh][None, :]
    )
    return qd, kd, vd


def oracle(q, k, v, softmax_scale=None, chunk=1024):
    """Exact FP32 softmax attention over the dequantized operands of :func:`recipe_operands`, no P quantization
    and no two-level fold, one head and ``chunk`` query rows at a time; FP32 output in the logical
    ``[B, H, S, 128]`` shape."""
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    batch, heads, seqlen, _dim = (int(x) for x in q.shape)
    ops = recipe_operands(q, k, v)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for b in range(batch):
            for h in range(heads):
                qd, kd, vd = _dequantized(ops, b * heads + h)
                for start in range(0, seqlen, chunk):
                    scores = (qd[start : start + chunk] @ kd.T) * scale
                    out[b, h, start : start + chunk] = (
                        torch.softmax(scores, dim=-1) @ vd
                    )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return out


def _check(out, expected):
    assert torch.isfinite(out).all(), "the output has NaN or Inf"
    actual = out.float()
    rel_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    try:
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    except AssertionError as exc:
        raise AssertionError(f"relative L2 error {float(rel_l2):.4g}; {exc}") from None
    return float(rel_l2)


# --- Correctness over the validated regimes -------------------------------------

SHAPES = [
    # (batch, heads, seqlen, operands, layout): tails, S < 128, S = 1, B > 1, H in {1, 5, 8, 12, 40}, both storage
    # orders, and the flat long row of the accumulation check.
    (1, 5, 1, "randn", "HND"),
    (1, 5, 100, "randn", "NHD"),
    (2, 12, 257, "randn", "HND"),
    (1, 12, 1100, "v_x8", "NHD"),
    (1, 40, 1100, "onehot", "HND"),
    (2, 5, 1100, "outlier32", "NHD"),
    (1, 12, 1100, "outlier32_q", "HND"),
    (1, 40, 4800, "randn", "NHD"),
    (1, 5, 4800, "outlier32_q", "HND"),
    (1, 8, 32760, "randn", "HND"),
    (1, 1, 32760, "spread", "NHD"),
]


@pytest.mark.parametrize("batch,heads,seqlen,operands,layout", SHAPES)
def test_fp8_attention(batch, heads, seqlen, operands, layout):
    _skip_unless_supported()
    q, k, v = make_inputs(
        batch,
        heads,
        seqlen,
        operands=operands,
        layout=layout,
        seed=2026 + seqlen + heads,
    )
    out = _empty_like_layout(q, layout)
    attention = backend.prepare_fp8_attention(q, k, v, out, backend="cake")
    assert attention() is out
    _check(out, oracle(q, k, v))
    # Replays are bitwise repeatable.
    snapshot = out.clone()
    out.zero_()
    assert attention.launch() is out
    torch.testing.assert_close(out, snapshot, atol=0, rtol=0)


def test_fp8_attention_softmax_scale_keyword():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 1100, seed=11)
    out = torch.empty_like(q)
    backend.prepare_fp8_attention(q, k, v, out, backend="cake", softmax_scale=0.05)()
    _check(out, oracle(q, k, v, softmax_scale=0.05))
    with pytest.raises(ValueError, match="softmax_scale"):
        backend.prepare_fp8_attention(q, k, v, out, backend="cake", softmax_scale=0.0)
    with pytest.raises(ValueError, match="softmax_scale"):
        backend.prepare_fp8_attention(q, k, v, out, backend="cake", softmax_scale=-1.0)


def test_fp8_attention_quantized_operands_match_the_recipe():
    """The workspace after a launch holds the recipe's operands: the |V| maxima and the V scales exactly, the K
    sums and means to the summation order's rounding, and -- quantized in torch from the kernel's own statistics,
    as the chain feeds them to the quantizer -- the INT8 / E4M3 codes and the Q / K scales exactly; zero codes past
    ``S``."""
    _skip_unless_supported()
    batch, heads, seqlen = 2, 3, 1100
    q, k, v = make_inputs(
        batch, heads, seqlen, operands="outlier32", layout="NHD", seed=7
    )
    out = _empty_like_layout(q, "NHD")
    attention = backend.prepare_fp8_attention(q, k, v, out, backend="cake")
    attention()
    torch.cuda.synchronize()
    layout = backend.workspace_layout(_arch(), batch=batch, heads=heads, seqlen=seqlen)
    region = {name: layout.view(attention.workspace, name) for name in layout.regions}
    bh, nqb = batch * heads, _ceil_div(seqlen, BLOCK)
    s_pad = nqb * BLOCK
    # Statistics.
    partials, mean_k, v_scale = recipe_statistics(k, v)
    assert region["partials"].shape == (bh, nqb, 2, HEAD_DIM)
    torch.testing.assert_close(
        region["partials"][:, :, 1], partials[:, :, 1], atol=0, rtol=0
    )
    torch.testing.assert_close(
        region["partials"][:, :, 0], partials[:, :, 0], atol=1e-3, rtol=1e-4
    )
    torch.testing.assert_close(region["v_scale"], v_scale, atol=0, rtol=0)
    torch.testing.assert_close(region["mean_k"], mean_k, atol=1e-5, rtol=1e-5)
    # The operands, from the statistics the quantizer read (the kernel's ``mean_k`` and ``v_scale``).
    ops = quantize_operands(
        q, k, v, region["mean_k"].clone(), region["v_scale"].clone()
    )
    assert region["q_scale"].shape == (bh, nqb) and region["k_scale"].shape == (
        bh,
        2 * nqb,
    )
    torch.testing.assert_close(region["q_scale"], ops.q_scale, atol=0, rtol=0)
    torch.testing.assert_close(region["k_scale"], ops.k_scale, atol=0, rtol=0)
    assert torch.isfinite(region["q_scale"]).all()
    assert torch.isfinite(region["k_scale"]).all()
    assert torch.equal(region["q8"], ops.q8)
    assert torch.equal(region["k8"], ops.k8)
    assert torch.equal(region["vt8"], ops.vt8)
    assert not region["q8"][:, seqlen:].any() and not region["k8"][:, seqlen:].any()
    position = _vt_position(s_pad, q.device)
    assert not region["vt8"][:, :, position[seqlen:]].any()


def test_fp8_attention_strided_views_match_contiguous_storage():
    """An ``[B, S, H, 128]`` buffer viewed as ``[B, H, S, 128]`` is read in place and written in place, and gives the
    contiguous copies' output bit for bit (the same quantized operands, the same reduction orders)."""
    _skip_unless_supported()
    q, k, v = make_inputs(2, 5, 1100, seed=3)
    expected = torch.empty_like(q)
    backend.prepare_fp8_attention(q, k, v, expected, backend="cake")()
    views = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
    out = _empty_like_layout(q, "NHD")
    attention = backend.prepare_fp8_attention(*views, out, backend="cake")
    assert attention() is out
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.equal(out.transpose(1, 2).contiguous().transpose(1, 2), out)


def test_fp8_attention_values_are_free_between_launches():
    """Quantization is part of every launch: new values in the bound tensors reach the next launch."""
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 257, seed=5)
    out = torch.empty_like(q)
    attention = backend.prepare_fp8_attention(q, k, v, out, backend="cake")
    attention()
    q2, k2, v2 = make_inputs(1, 5, 257, seed=6)
    q.copy_(q2), k.copy_(k2), v.copy_(v2)
    attention()
    fresh = torch.empty_like(q)
    backend.prepare_fp8_attention(q2, k2, v2, fresh, backend="cake")()
    torch.testing.assert_close(out, fresh, atol=0, rtol=0)


def test_fp8_attention_graph_replay():
    """The runner is captured by the caller; replays reproduce the eager launch bit for bit, output and workspace
    alike (the chain carries no device-side state across launches)."""
    _skip_unless_supported()
    q, k, v = make_inputs(2, 5, 1100, seed=9)
    out = torch.empty_like(q)
    attention = backend.prepare_fp8_attention(q, k, v, out, backend="cake")
    expected = attention().clone()
    torch.cuda.synchronize()
    layout = backend.workspace_layout(_arch(), batch=2, heads=5, seqlen=1100)
    # Region by region: the slack between the 256-byte slots is never written.
    regions_after_eager = {
        name: layout.view(attention.workspace, name).clone() for name in layout.regions
    }
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        attention()  # warm the stream before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        attention()
    for _ in range(3):
        out.zero_()
        attention.workspace.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        for name, eager in regions_after_eager.items():
            assert torch.equal(layout.view(attention.workspace, name), eager), name


def test_fp8_attention_workspace_contract():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 12, 1100, seed=2)
    out = torch.empty_like(q)
    needed = backend.fp8_attention_workspace_bytes(1, 12, 1100, q.device)
    assert needed > 0 and needed % backend.ALIGN == 0
    attention = backend.prepare_fp8_attention(q, k, v, out, backend="cake")
    assert attention.workspace_bytes == needed == attention.workspace.numel()
    expected = attention().clone()
    pool = torch.empty(needed + 4096, dtype=torch.uint8, device=q.device)
    pooled = backend.prepare_fp8_attention(q, k, v, out, backend="cake", workspace=pool)
    assert pooled.workspace is pool and pooled.workspace_bytes == needed
    torch.testing.assert_close(pooled(), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="workspace holds"):
        backend.prepare_fp8_attention(
            q, k, v, out, backend="cake", workspace=pool[: needed - 1]
        )
    with pytest.raises(ValueError, match="uint8"):
        backend.prepare_fp8_attention(
            q, k, v, out, backend="cake", workspace=torch.empty(needed, device=q.device)
        )


def test_fp8_attention_route_metadata_and_stages():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 257, seed=4)
    out = torch.empty_like(q)
    attention = backend.prepare_fp8_attention(q, k, v, out, backend="cake")
    arch = _arch()
    chain = loader.records_for(arch)
    assert [stage for stage, _, _ in attention.stages] == list(loader.STAGES)
    metadata = attention.route_metadata
    assert metadata["arch"] == arch and metadata["records"] == chain
    assert metadata["recipe"] == backend.RECIPE_ID
    # The attention build spells no ``qk_format``: the host's ``QK_FORMAT`` is reported.
    assert (
        metadata["qk_format"]
        == loader.MODULES[chain["attention"]]["build"].get(
            "qk_format", backend.QK_FORMAT
        )
        == backend.QK_FORMAT
    )
    assert metadata["shape"] == (1, 5, 257, HEAD_DIM)
    assert metadata["softmax_scale"] == pytest.approx(HEAD_DIM**-0.5)
    assert metadata["workspace_bytes"] == attention.workspace_bytes
    assert set(metadata) == {
        "arch",
        "recipe",
        "qk_format",
        "records",
        "grids",
        "softmax_scale",
        "workspace_bytes",
        "shape",
    }
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    for stage, (record_stage, _entry, arguments) in zip(
        loader.STAGES, attention.stages, strict=True
    ):
        record = loader.MODULES[chain[stage]]
        grid = loader.launch_grid(
            record, batch=1, heads=5, seqlen=257, sm_count=sm_count
        )
        assert record_stage == stage and metadata["grids"][stage] == grid
        assert (
            len(arguments) == len(record["arg_plan"]) and tuple(arguments[-3:]) == grid
        )
    # A stage alone launches (the chain's order is the caller's to respect).
    attention.launch_stage("stats_partials")
    with pytest.raises(KeyError):
        attention.launch_stage("stats")


def test_fp8_attention_prepare_validation():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 257, seed=1)
    out = torch.empty_like(q)
    with pytest.raises(ValueError, match="backend"):
        backend.prepare_fp8_attention(q, k, v, out, backend="unknown")
    with pytest.raises(TypeError, match="BF16"):
        backend.prepare_fp8_attention(q.float(), k, v, out, backend="cake")
    with pytest.raises(ValueError, match="shape"):
        backend.prepare_fp8_attention(q, k[:, :4], v, out, backend="cake")
    with pytest.raises(ValueError, match=r"\[B, H, S, 128\]"):
        backend.prepare_fp8_attention(
            q[..., :64], k[..., :64], v[..., :64], out[..., :64], backend="cake"
        )
    # L90 test-side fix: ``q.transpose(2, 3)`` has shape (1, 5, 128, 257) and fails the shape check first; a unit-extent
    # strided inner view exercises the inner-stride check the case is about.
    strided = torch.empty(
        (1, 5, 257, 2 * HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )[..., ::2]
    with pytest.raises(ValueError, match="stride 1"):
        backend.prepare_fp8_attention(strided, k, v, out, backend="cake")
    padded = torch.empty(
        (1, 5, 257, HEAD_DIM + 8), dtype=torch.bfloat16, device=q.device
    )[..., :HEAD_DIM]
    with pytest.raises(ValueError, match="not a multiple of 16"):
        backend.prepare_fp8_attention(padded, k, v, out, backend="cake")
    with pytest.raises(ValueError, match="CUDA"):
        backend.prepare_fp8_attention(
            q.cpu(), k.cpu(), v.cpu(), out.cpu(), backend="cake"
        )


def test_fp8_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        backend.prepare_fp8_attention(None, None, None, None, backend="unknown")


# --- The host plan (no GPU) ------------------------------------------------------


def test_host_plan_of_the_records():
    """The registry is a role table: one record per stage, every kernel parameter named in the package
    vocabulary, the grid rules evaluated by ``launch_grid`` and the arguments bound by ``launch_args``."""
    assert loader.STAGES == (
        "stats_partials",
        "stats_finalize",
        "quantize",
        "attention",
    )
    assert loader.ARCHES == ("sm_90a",)
    assert loader.ARCH_BY_CAPABILITY == {(9, 0): "sm_90a"}
    assert set(loader.LOGICAL_TENSORS).isdisjoint(loader.LOGICAL_SCALARS)
    for arch in loader.ARCHES:
        chain = loader.records_for(arch)
        assert list(chain) == list(loader.STAGES)
        for stage, name in chain.items():
            record = loader.MODULES[name]
            assert record["role"] == stage and record["stage"] == loader.STAGES.index(
                stage
            )
            assert arch in record["arches"]
            for kind, key in record["arg_plan"]:
                if kind == "grid":
                    continue
                logical = record["logical"][key]
                assert logical in (
                    loader.LOGICAL_TENSORS
                    if kind == "buffer"
                    else loader.LOGICAL_SCALARS
                )
                if kind == "buffer":
                    assert record["dtypes"][key] in backend._TORCH_DTYPES, (name, key)
            assert set(record["dtypes"]) == {
                key for kind, key in record["arg_plan"] if kind == "buffer"
            }
            if stage == "stats_partials":
                rule = record["workspace_rule"]
                assert rule["kind"] == "prepare_carved" and rule["counters"] is None
                assert rule["regions"] == WORKSPACE_REGIONS
            else:
                assert record["workspace_rule"] is None
            assert (
                record["launch"]["use_pdl"] is (stage != "stats_partials")
                and record["launch"]["cooperative"] is False
            )
    with pytest.raises(
        RuntimeError, match="delivers 0 'stats_partials' programs for sm_100a"
    ):
        loader.records_for("sm_100a")
    chain = loader.records_for("sm_90a")
    stats_partials, stats_finalize, quantize, attention = (
        loader.MODULES[chain[stage]] for stage in loader.STAGES
    )
    # The builds: the shipped ring depth of the attention and the reduction unroll of the finalize; no record
    # spells a ``qk_format`` -- the host states it.
    assert attention["build"] == {"num_stages": 2}
    assert stats_finalize["build"] == {"unroll": 8}
    assert stats_partials["build"] == quantize["build"] == {}
    assert backend.QK_FORMAT == "int8_blk"
    # The launch records: 256 threads on the stage kernels, one 128-thread warpgroup on the attention, and the
    # dynamic shared memory of each kernel (the attention: the 73728-byte Q + two-stage K / V^T ring behind
    # the launcher's 1,024-byte control prefix = 74752 bytes at launch, the rendered leaf).
    for record, block, smem in (
        (stats_partials, 256, 16384),
        (stats_finalize, 256, 0),
        (quantize, 256, 128),
        (attention, 128, 74752),
    ):
        assert tuple(record["launch"]["block"]) == (block, 1, 1), record["role"]
        assert tuple(record["launch"]["cluster"]) == (1, 1, 1), record["role"]
        assert record["launch"]["dynamic_smem_bytes"] == smem, record["role"]
    # The caller's tensors reach the kernels through explicit element strides: a record reading or writing one
    # binds its three strides.  The statistics read K and V only (Q is not smoothed).
    for record, prefixes in (
        (stats_partials, ("k", "v")),
        (quantize, ("q", "k", "v")),
        (attention, ("o",)),
    ):
        logical = set(record["logical"].values())
        for prefix in prefixes:
            assert {
                f"{prefix}_stride_b",
                f"{prefix}_stride_h",
                f"{prefix}_stride_s",
            } <= logical
    assert "q" not in stats_partials["logical"].values()
    assert attention["logical"]["O"] == "out"
    # The attention's operands: the three byte tensor maps the quantizer writes as 32-bit words, and the FP32
    # scales; the attention binds no block count (its grid is the 64-row tile count).
    assert {attention["logical"][key] for key in ("Q8", "K8", "VT8")} == {
        "q8",
        "k8",
        "vt8",
    }
    assert all(attention["dtypes"][key] == "u8" for key in ("Q8", "K8", "VT8"))
    assert all(quantize["dtypes"][key] == "u32" for key in ("q8", "k8", "vt8"))
    assert all(
        attention["dtypes"][key] == "f32" for key in ("q_scale", "k_scale", "v_scale")
    )
    assert "num_qblocks" not in attention["logical"].values()
    # Grid rules: one CTA per (batch, head, 128-row block) for the statistics partials and the quantizer, one per
    # (batch, head) for the finalize, one per (batch, head, 64-row Q tile) for the attention.
    assert loader.launch_grid(
        attention, batch=1, heads=40, seqlen=4800, sm_count=132
    ) == (40 * 75, 1, 1)
    assert loader.launch_grid(attention, batch=1, heads=1, seqlen=1, sm_count=132) == (
        1,
        1,
        1,
    )
    assert loader.launch_grid(
        attention, batch=2, heads=3, seqlen=1100, sm_count=132
    ) == (6 * 18, 1, 1)
    assert loader.launch_grid(
        stats_partials, batch=1, heads=1, seqlen=257, sm_count=132
    ) == (3, 1, 1)
    assert loader.launch_grid(
        stats_finalize, batch=2, heads=5, seqlen=1000, sm_count=132
    ) == (10, 1, 1)
    assert loader.launch_grid(
        quantize, batch=2, heads=5, seqlen=1000, sm_count=132
    ) == (80, 1, 1)
    with pytest.raises(ValueError, match="positive extents"):
        loader.launch_grid(attention, batch=0, heads=1, seqlen=512, sm_count=132)
    with pytest.raises(ValueError, match="unknown Cake FP8 attention grid rule"):
        loader.launch_grid(
            {"grid_rule": {"kind": "linear"}},
            batch=1,
            heads=1,
            seqlen=512,
            sm_count=132,
        )
    # The argument plan: values by logical name, then the grid axes.
    values = {
        attention["logical"][key]: object()
        for kind, key in attention["arg_plan"]
        if kind != "grid"
    }
    args = loader.launch_args(attention, values, (24, 1, 1))
    assert args[-3:] == (24, 1, 1)
    assert args[:-3] == tuple(
        values[attention["logical"][key]]
        for kind, key in attention["arg_plan"]
        if kind != "grid"
    )
    with pytest.raises(ValueError, match="needs logical argument 'out'"):
        loader.launch_args(
            attention, {k: v for k, v in values.items() if k != "out"}, (2, 1, 1)
        )


def _views(batch, heads, seqlen, layout):
    """CPU logical ``[B, H, S, 128]`` views of ``layout`` buffers (the kernels never run here)."""
    tensors = []
    for _ in range(4):
        if layout == "NHD":
            x = torch.empty(
                (batch, seqlen, heads, HEAD_DIM), dtype=torch.bfloat16
            ).transpose(1, 2)
        else:
            x = torch.empty((batch, heads, seqlen, HEAD_DIM), dtype=torch.bfloat16)
        tensors.append(x)
    return tensors


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_values_bind_the_kernels_views_of_the_workspace(layout):
    """Host plan on the CPU: every record's buffers are the carved regions viewed as the record's dtypes (the
    quantizer's 32-bit code words, the attention's bytes), the caller's four tensors the contiguous span their
    view covers (same first element), the scalars the kernels' block count and the views' element strides."""
    batch, heads, seqlen = 2, 3, 1100
    bh, nqb, s_pad = 6, 9, 1152
    q, k, v, out = _views(batch, heads, seqlen, layout)
    common = backend._common_scalars(q, k, v, out, HEAD_DIM**-0.5)
    assert set(common) == set(loader.LOGICAL_SCALARS)
    assert (common["batch"], common["heads"], common["seqlen"]) == (2, 3, 1100)
    assert common["num_qblocks"] == nqb
    assert common["softmax_scale_log2"] == pytest.approx(
        HEAD_DIM**-0.5 * math.log2(math.e)
    )
    for prefix, x in (("q", q), ("k", k), ("v", v), ("o", out)):
        assert (
            common[f"{prefix}_stride_b"],
            common[f"{prefix}_stride_h"],
            common[f"{prefix}_stride_s"],
        ) == (x.stride(0), x.stride(1), x.stride(2))
    if layout == "NHD":
        assert (
            common["q_stride_h"] == HEAD_DIM
            and common["q_stride_s"] == heads * HEAD_DIM
        )
    else:
        assert (
            common["q_stride_s"] == HEAD_DIM
            and common["q_stride_h"] == seqlen * HEAD_DIM
        )
    # A unit extent's stride is spelled as the next stride.
    single = backend._common_scalars(*_views(1, 1, 1, layout), 1.0)
    assert (
        single["q_stride_b"] == single["q_stride_h"] == single["q_stride_s"] == HEAD_DIM
    )
    layout_ = backend.workspace_layout(
        "sm_90a", batch=batch, heads=heads, seqlen=seqlen
    )
    workspace = torch.zeros(layout_.total_bytes, dtype=torch.uint8)
    tensors = {name: layout_.view(workspace, name) for name in layout_.regions}
    tensors.update(q=q, k=k, v=v, out=out)
    # The regions in the ranks the attention kernel maps or reads them.
    assert tensors["partials"].shape == (bh, nqb, 2, HEAD_DIM)
    assert tensors["v_scale"].shape == tensors["mean_k"].shape == (bh, HEAD_DIM)
    assert tensors["q8"].shape == tensors["k8"].shape == (bh, s_pad, HEAD_DIM)
    assert tensors["vt8"].shape == (bh, HEAD_DIM, s_pad)
    assert tensors["q_scale"].shape == (bh, nqb)
    assert tensors["k_scale"].shape == (bh, 2 * nqb)
    expected_grids = {
        "stats_partials": (bh * nqb, 1, 1),
        "stats_finalize": (bh, 1, 1),
        "quantize": (bh * nqb, 1, 1),
        "attention": (bh * _ceil_div(seqlen, ROWS_PER_CTA), 1, 1),
    }
    for stage, name in loader.records_for("sm_90a").items():
        record = loader.MODULES[name]
        values = backend._values_for(record, tensors=tensors, common=common)
        assert set(values) == {
            record["logical"][key] for kind, key in record["arg_plan"] if kind != "grid"
        }
        for kind, key in record["arg_plan"]:
            if kind != "buffer":
                continue
            logical = record["logical"][key]
            value = values[logical]
            assert value.dtype == backend._TORCH_DTYPES[record["dtypes"][key]], (
                stage,
                key,
            )
            source = tensors[logical]
            assert value.data_ptr() == source.data_ptr(), (stage, key)
            if logical in ("q", "k", "v", "out"):
                assert value.ndim == 1 and value.is_contiguous()
                assert value.numel() == 1 + sum(
                    (d - 1) * st
                    for d, st in zip(source.shape, source.stride(), strict=True)
                )
            else:
                assert (
                    value.numel() * value.element_size()
                    == source.numel() * source.element_size()
                )
        for scalar in ("batch", "heads", "seqlen", "num_qblocks", "softmax_scale_log2"):
            if scalar in values:
                assert values[scalar] == common[scalar]
        if stage == "stats_partials":
            assert values["partials"].shape == (bh, nqb, 2, HEAD_DIM)
            assert "q" not in values
        if stage == "quantize":
            # The quantizer writes 32-bit words of codes.
            assert values["q8"].dtype == torch.uint32
            assert (
                values["q8"].shape == values["k8"].shape == (bh, s_pad, HEAD_DIM // 4)
            )
            assert values["vt8"].shape == (bh, HEAD_DIM, s_pad // 4)
            assert values["q_scale"].shape == (bh, nqb)
            assert values["k_scale"].shape == (bh, 2 * nqb)
        if stage == "attention":
            # The attention maps the codes as bytes (three tensor maps), reads the FP32 scales and takes the output
            # as a flat span; it binds no block count.
            assert values["q8"].dtype == values["k8"].dtype == torch.uint8
            assert values["vt8"].dtype == torch.uint8
            assert values["q8"].shape == values["k8"].shape == (bh, s_pad, HEAD_DIM)
            assert values["vt8"].shape == (bh, HEAD_DIM, s_pad)
            assert values["q_scale"].shape == (bh, nqb)
            assert values["k_scale"].shape == (bh, 2 * nqb)
            assert values["v_scale"].shape == (bh, HEAD_DIM)
            assert values["out"].numel() == batch * heads * seqlen * HEAD_DIM
            assert "num_qblocks" not in values
        grid = loader.launch_grid(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=132
        )
        assert grid == expected_grids[stage], stage
        args = loader.launch_args(record, values, grid)
        assert len(args) == len(record["arg_plan"]) and args[-3:] == grid
    # A record binding a dtype the host does not view as is refused before anything is loaded.
    record = dict(loader.MODULES[loader.records_for("sm_90a")["quantize"]])
    record["dtypes"] = {**record["dtypes"], "q8": "f16"}
    with pytest.raises(RuntimeError, match="states dtype 'f16' for 'q8'"):
        backend._values_for(record, tensors=tensors, common=common)
    # A record binding a scalar the host does not compute is refused too.
    record = dict(loader.MODULES[loader.records_for("sm_90a")["attention"]])
    record["logical"] = {**record["logical"], "seqlen": "num_items"}
    with pytest.raises(RuntimeError, match="binds the scalar 'num_items'"):
        backend._values_for(record, tensors=tensors, common=common)


def test_workspace_layout_is_the_vocabulary_on_aligned_slots():
    for arch in loader.ARCHES:
        layout = backend.workspace_layout(arch, batch=2, heads=5, seqlen=1100)
        offsets = [region.offset for region in layout.regions.values()]
        assert offsets == sorted(offsets) and all(
            offset % backend.ALIGN == 0 for offset in offsets
        )
        assert set(layout.regions) <= set(loader.LOGICAL_TENSORS)
        assert layout.total_bytes % backend.ALIGN == 0 and layout.total_bytes >= sum(
            r.nbytes for r in layout.regions.values()
        )
        chain = loader.records_for(arch)
        for name in chain.values():
            record = loader.MODULES[name]
            for kind, key in record["arg_plan"]:
                if kind == "buffer" and record["logical"][key] not in (
                    "q",
                    "k",
                    "v",
                    "out",
                ):
                    assert record["logical"][key] in layout.regions, (arch, name, key)
            if record["workspace_rule"] is not None:
                # The host carves exactly the registry's regions, in the registry's order.
                assert list(layout.regions) == record["workspace_rule"]["regions"]
    # The byte formulas of the README, in the carving order, on 256-byte slots.
    layout = backend.workspace_layout("sm_90a", batch=2, heads=5, seqlen=1100)
    bh, nqb, s_pad = 10, 9, 1152
    expected = {
        "partials": bh * nqb * 2 * 128 * 4,
        "mean_k": bh * 128 * 4,
        "v_scale": bh * 128 * 4,
        "q8": bh * s_pad * 128,
        "k8": bh * s_pad * 128,
        "vt8": bh * 128 * s_pad,
        "q_scale": bh * nqb * 4,
        "k_scale": bh * 2 * nqb * 4,
    }
    assert {name: region.nbytes for name, region in layout.regions.items()} == expected
    assert list(layout.regions) == list(expected) == WORKSPACE_REGIONS
    assert layout.total_bytes == sum(
        -(-nbytes // backend.ALIGN) * backend.ALIGN for nbytes in expected.values()
    )
    workspace = torch.zeros(layout.total_bytes, dtype=torch.uint8)
    for name, region in layout.regions.items():
        view = layout.view(workspace, name)
        assert view.shape == region.shape and view.dtype == region.dtype
    bytes_one = backend.workspace_layout("sm_90a", batch=1, heads=1, seqlen=1)
    assert bytes_one.regions["q8"].shape == (1, 128, 128)
    assert bytes_one.regions["q_scale"].shape == (1, 1)
    assert bytes_one.regions["k_scale"].shape == (1, 2)
    bytes_large = backend.workspace_layout(
        "sm_90a", batch=1, heads=1, seqlen=4800
    ).total_bytes
    assert 0 < bytes_one.total_bytes < bytes_large
    with pytest.raises(ValueError, match="not an exported FP8 attention architecture"):
        backend.workspace_layout("sm_100a", batch=1, heads=1, seqlen=1)
    with pytest.raises(ValueError, match="positive extents"):
        backend.workspace_layout("sm_90a", batch=0, heads=1, seqlen=1)


def test_int32_index_limit():
    """The kernels index every operand with 32-bit integers: the dense element count and every view's span stay
    below ``2**31`` (the largest supported row, ``[1, 40, 75600, 128]``, passes)."""
    backend._check_index_range(1, 40, 75600, {})
    with pytest.raises(ValueError, match="32-bit"):
        backend._check_index_range(1, 64, 262144, {})
    wide = types.SimpleNamespace(
        shape=(2, 1, 1, HEAD_DIM), stride=lambda: (2**31, HEAD_DIM, HEAD_DIM, 1)
    )
    with pytest.raises(ValueError, match="32-bit"):
        backend._check_index_range(2, 1, 1, {"q": wide})


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library, architecture) the registry
    references, each with the kernel and binding sources under ``csrc/`` of this package, their digests, a closure
    digest and the compile options the program's kernel was validated with; the registry itself records the host
    plan only.  Skipped while the package has not been exported with its manifest.
    """
    package = Path(cake_build.__file__).parent
    manifest_path = package / "build.json"
    if not manifest_path.is_file():
        pytest.skip(f"{manifest_path} is absent: the package has no build manifest yet")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES)
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    referenced = set()
    for name, record in loader.MODULES.items():
        library = record["library"]
        assert record["kernel"] == library
        assert re.fullmatch(r"cake_fp8_attention_[0-9a-f]{20}", library)
        assert record["kernel_symbol"] == f"kernel_{library}"
        assert record["ffi_entry"] == "run"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert record["arches"] and set(record["arches"]) <= set(loader.ARCHES), name
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["dynamic_smem_bytes"] >= 0
        assert len(launch["block"]) == 3 and len(launch["cluster"]) == 3
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        for arch in record["arches"]:
            entry_name = loader.library_name(name, arch)
            referenced.add(entry_name)
            entry = libraries[entry_name]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{library}_kernel.cu",
                f"csrc/{library}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert entry["nvcc_flags"] == NVCC_FLAGS_BY_ROLE[record["role"]], entry_name
            assert re.fullmatch(r"\d+\.\d+", entry["min_cuda"]), entry_name
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
