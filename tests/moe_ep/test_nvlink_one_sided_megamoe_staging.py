# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVLinkOneSided MegaMoE staging: the one-launch CuTe DSL stager against the Torch staging.

``stage_inputs`` stages a batch in one CuTe DSL launch (``staging_kernel``)
and keeps a Torch sequence (``quantize_mxfp8_block32``, copies, sentinel
rewrites, masked tail, shared expert's scale plane) for layouts the kernel
does not take. Every test here stages one batch both ways from the same
workspace state and requires equal bytes in every staging plane: data,
scales, top-k ids and weights and, with a fused shared expert, its scale
plane. The workspace is a single-GPU stand-in with the staging planes of
``BlockScaledSymmBuffer`` and its shared-expert scale methods.

BF16 inputs cover every finite block amax, every BF16 value under one amax of
each scale exponent, blocks with an infinity or a NaN (scaled by ``2**-126``,
so their finite values saturate), zero, negative-zero and subnormal blocks;
pre-quantized inputs cover ``0xFF`` data and scale bytes. Token counts run
from none to the capacity over a workspace a larger batch left behind, with
masked routes and int32 and int64 ids, at the hidden sizes of DeepSeek-V4.1
Flash (5120), V4-Flash (4096), V4-Pro (7168), Kimi-K3 (3584) and HY4 (6144).
A captured stager replays new batches to the same bytes.

Run on one SM100 / SM103 GPU::

    CUTE_DSL_ARCH=sm_103a pytest tests/moe_ep/test_nvlink_one_sided_megamoe_staging.py
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.arch_blackwell

_SF_VEC = 32
# (hidden, top_k) of the models above.
_SHAPES = [(5120, 6), (4096, 6), (7168, 6), (3584, 16), (6144, 8)]
# (token tile rows, pool slot rows) of the shared expert's token tiles.
_SHARED_TILES = [(128, 128), (240, 256)]


def _modules():
    from flashinfer.moe_ep.backends.mega.kernel.sm103.mxfp8_mxfp4_bf16_nvlink_one_sided import (
        staging,
        staging_kernel,
    )

    return staging, staging_kernel


def _workspace(capacity, hidden, top_k, *, shared_tiles=None, sf_pad=4, seed=0):
    """Staging planes of a ``BlockScaledSymmBuffer``, holding arbitrary bytes."""
    from flashinfer.moe_ep.kernels.flashinfer.mxfp8_mxfp4_nvlink_one_sided.frontend.block_scaled import (
        BlockScaledSymmBuffer,
    )

    class Workspace:
        note_staged_tokens = BlockScaledSymmBuffer.note_staged_tokens
        staged_tokens = BlockScaledSymmBuffer.staged_tokens
        shared_expert_scale_layout = BlockScaledSymmBuffer.shared_expert_scale_layout
        stage_shared_expert_scales = BlockScaledSymmBuffer.stage_shared_expert_scales

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def noise(*shape, dtype):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen
        ).view(dtype)

    ws = Workspace()
    ws._staged_tokens = None
    ws.config = SimpleNamespace(
        hidden=hidden,
        sf_vec_size=_SF_VEC,
        shared_expert=None if shared_tiles is None else "fp8_in_kernel",
    )
    hidden_sf = hidden // _SF_VEC
    ws.x = noise(capacity, hidden, dtype=torch.float8_e4m3fn)
    ws.x_sf = noise(capacity, hidden_sf + sf_pad, dtype=torch.float8_e8m0fnu)
    ws.topk_idx = noise(capacity, 4 * top_k, dtype=torch.int32)
    ws.topk_weights = noise(capacity, 4 * top_k, dtype=torch.float32)
    if shared_tiles is not None:
        tile_rows, slot_rows = shared_tiles
        tokens = torch.arange(capacity, device="cuda")
        token_tiles = -(-capacity // tile_rows)
        sf_rows = -(-token_tiles * slot_rows // 128) * 128
        ws.kernel = SimpleNamespace(
            shared_expert_tiles=SimpleNamespace(
                tile_rows=tile_rows, slot_rows=slot_rows
            )
        )
        ws._shared_expert_scale_rows = (
            tokens // tile_rows * slot_rows + tokens % tile_rows
        )
        ws._shared_expert_scale_scratch = torch.zeros(
            (sf_rows, hidden_sf), dtype=torch.uint8, device="cuda"
        )
        ws._shared_expert_scales = noise(sf_rows * hidden_sf, dtype=torch.uint8)
    return ws


def _planes(ws):
    planes = [
        ws.x.view(torch.uint8),
        ws.x_sf.view(torch.uint8),
        ws.topk_idx,
        ws.topk_weights,
    ]
    if ws.config.shared_expert is not None:
        planes.append(ws.shared_expert_scale_layout()[0])
    return planes


def _stage_both(t, ws, *, quantize):
    """Stage ``t`` in one launch, then through the Torch sequence from the same state.

    Returns whether every plane holds the same bytes; the workspace is left
    as the Torch sequence staged it.
    """
    staging, staging_kernel = _modules()
    stage = staging_kernel.stage
    launched = []

    def spy(*args, **kwargs):
        launched.append(stage(*args, **kwargs))
        return launched[-1]

    planes = _planes(ws)
    before = [plane.clone() for plane in planes]
    with mock.patch.object(staging_kernel, "stage", spy):
        staging.stage_inputs(t, ws, quantize_input=quantize)
    assert launched == [True], "the stager declined a supported layout"
    assert ws.staged_tokens() == t.hidden_states.shape[0]
    fused = [plane.clone() for plane in planes]
    for plane, saved in zip(planes, before, strict=True):
        plane.copy_(saved)
    with mock.patch.object(staging_kernel, "stage", lambda *_, **__: False):
        staging.stage_inputs(t, ws, quantize_input=quantize)
    torch.cuda.synchronize()
    return [
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(fused, planes, strict=True)
    ]


def _routing(count, top_k, *, wide_ids, gen):
    ids = torch.randint(
        -1, 384, (count, top_k), dtype=torch.int64, device="cuda", generator=gen
    )
    weights = torch.randn(count, top_k, device="cuda", generator=gen)
    return ids if wide_ids else ids.to(torch.int32), weights


def _bf16(bits):
    return bits.to(torch.int16).view(torch.bfloat16)


def _edge_bf16(rows, hidden, gen):
    """BF16 rows whose 32-value blocks each take one of the edge-case kinds.

    Scaled normal values over the whole exponent range (with overflow to
    infinity), uniformly random bit patterns (NaNs and infinities included),
    zeros, negative zeros, subnormals, and blocks with an infinity, a NaN of
    either sign or the largest finite value at a random position.
    """
    blocks = rows * hidden // _SF_VEC
    shape = (blocks, _SF_VEC)
    exponent = torch.randint(-140, 128, (blocks, 1), device="cuda", generator=gen)
    scaled = torch.randn(shape, device="cuda", generator=gen) * torch.pow(
        2.0, exponent.float()
    )
    bits = scaled.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    random_bits = torch.randint(
        0, 1 << 16, shape, dtype=torch.int32, device="cuda", generator=gen
    )
    subnormal = torch.randint(
        1, 0x80, shape, dtype=torch.int32, device="cuda", generator=gen
    ) | (
        torch.randint(0, 2, shape, dtype=torch.int32, device="cuda", generator=gen)
        << 15
    )
    kind = torch.randint(0, 10, (blocks, 1), device="cuda", generator=gen)
    bits = torch.where(kind == 1, random_bits, bits)
    bits = torch.where(kind == 2, torch.zeros_like(bits), bits)
    bits = torch.where(kind == 3, torch.full_like(bits, 0x8000), bits)
    bits = torch.where(kind == 4, subnormal, bits)
    position = torch.randint(0, _SF_VEC, (blocks, 1), device="cuda", generator=gen)
    at = torch.arange(_SF_VEC, device="cuda") == position
    specials = torch.tensor(
        [0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x7F7F, 0xFF7F],
        dtype=torch.int32,
        device="cuda",
    )
    special = specials[
        torch.randint(0, len(specials), (blocks, 1), device="cuda", generator=gen)
    ]
    bits = torch.where((kind >= 5) & at, special.expand(shape), bits)
    return _bf16(bits.reshape(rows, hidden))


def _batch(count, hidden, top_k, *, quantize, wide_ids, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ids, weights = _routing(count, top_k, wide_ids=wide_ids, gen=gen)
    if quantize:
        return SimpleNamespace(
            hidden_states=_edge_bf16(count, hidden, gen),
            scales=None,
            topk_ids=ids,
            topk_weights=weights,
        )
    data = torch.randint(
        0, 256, (count, hidden), dtype=torch.uint8, device="cuda", generator=gen
    )
    data[::5, ::7] = 0xFF
    scales = torch.randint(
        0,
        256,
        (count, hidden // _SF_VEC + 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    scales[::3, ::5] = 0xFF
    return SimpleNamespace(
        hidden_states=data.view(torch.float8_e4m3fn),
        scales=scales.view(torch.float8_e8m0fnu),
        topk_ids=ids,
        topk_weights=weights,
    )


def _sweep(hidden, top_k, capacity, *, quantize, shared_tiles, wide_ids, seed):
    ws = _workspace(capacity, hidden, top_k, shared_tiles=shared_tiles, seed=seed)
    # A whole batch through the Torch sequence first, so every row of the
    # shared expert's scale plane holds its staged scales.
    staging, staging_kernel = _modules()
    full = _batch(
        capacity, hidden, top_k, quantize=quantize, wide_ids=wide_ids, seed=seed
    )
    with mock.patch.object(staging_kernel, "stage", lambda *_, **__: False):
        staging.stage_inputs(full, ws, quantize_input=quantize)
    for count in (capacity // 2, 0, 1, 7, 37, capacity):
        t = _batch(
            count,
            hidden,
            top_k,
            quantize=quantize,
            wide_ids=wide_ids,
            seed=seed + 31 * count + 1,
        )
        same = _stage_both(t, ws, quantize=quantize)
        assert all(same), f"count {count}: planes equal {same}"


@pytest.mark.parametrize("hidden,top_k", _SHAPES)
@pytest.mark.parametrize("quantize", [True, False], ids=["bf16", "prequantized"])
def test_staging_matches_torch(hidden, top_k, quantize):
    """Routed experts only, int64 ids, counts 0..capacity over a dirty workspace."""
    _sweep(
        hidden,
        top_k,
        96,
        quantize=quantize,
        shared_tiles=None,
        wide_ids=True,
        seed=hidden + top_k,
    )


@pytest.mark.parametrize("shared_tiles", _SHARED_TILES, ids=["n128", "n240"])
@pytest.mark.parametrize("quantize", [True, False], ids=["bf16", "prequantized"])
def test_staging_shared_expert_scales(shared_tiles, quantize):
    """With a fused shared expert its scale plane gets the staged scales of the live rows."""
    _sweep(
        5120,
        6,
        300,
        quantize=quantize,
        shared_tiles=shared_tiles,
        wide_ids=False,
        seed=7 + shared_tiles[0],
    )


def test_staging_int32_ids_and_capacity_rows():
    """int32 ids and a capacity whose rows do not fill the last grid step."""
    _sweep(3584, 16, 4099, quantize=True, shared_tiles=None, wide_ids=False, seed=11)


def _blocks_to_rows(blocks, hidden):
    """Pad whole blocks of BF16 bits to whole rows of zero blocks."""
    per_row = hidden // _SF_VEC
    rows = -(-blocks.shape[0] // per_row)
    pad = rows * per_row - blocks.shape[0]
    blocks = torch.cat([blocks, blocks.new_zeros(pad, _SF_VEC)])
    return _bf16(blocks.reshape(rows, hidden))


def _stage_rows(hidden_states, hidden):
    rows = hidden_states.shape[0]
    gen = torch.Generator(device="cuda").manual_seed(rows)
    ids, weights = _routing(rows, 6, wide_ids=True, gen=gen)
    ws = _workspace(rows, hidden, 6, seed=rows)
    t = SimpleNamespace(
        hidden_states=hidden_states, scales=None, topk_ids=ids, topk_weights=weights
    )
    return _stage_both(t, ws, quantize=True)


def test_bf16_every_amax():
    """Every finite BF16 magnitude as a block amax, under 31 smaller random values."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    amax = torch.arange(0, 0x7F80, dtype=torch.int32, device="cuda")
    others = (
        (
            torch.rand(amax.shape[0], _SF_VEC - 1, device="cuda", generator=gen)
            * (amax.unsqueeze(1) + 1).float()
        )
        .to(torch.int32)
        .clamp(max=amax.unsqueeze(1))
    )
    blocks = torch.cat([amax.unsqueeze(1), others], 1)
    blocks |= (
        torch.randint(
            0, 2, blocks.shape, dtype=torch.int32, device="cuda", generator=gen
        )
        << 15
    )
    # The amax at every position of the block.
    shift = torch.randint(
        0, _SF_VEC, (blocks.shape[0], 1), device="cuda", generator=gen
    )
    index = (torch.arange(_SF_VEC, device="cuda") + shift) % _SF_VEC
    blocks = torch.gather(blocks, 1, index)
    same = _stage_rows(_blocks_to_rows(blocks, 5120), 5120)
    assert all(same), f"planes equal {same}"


def test_bf16_every_value_per_scale():
    """Every BF16 value of each sign under the largest amax of every scale code.

    Blocks hold an amax and 31 consecutive magnitudes at or below it, so each
    finite scale code sees all its values, the subnormals and both zeros
    included.
    """
    magnitudes = torch.arange(0, 0x7F80, dtype=torch.float32, device="cuda")
    bits = magnitudes.to(torch.int32)
    # Code of each magnitude as an amax: ceil(log2(amax * fl32(1/448))).
    scaled = _bf16(bits).float() * (1.0 / 448.0)
    p = scaled.clamp_min(2.0**-126).view(torch.int32)
    codes = ((p >> 23) & 0xFF) + ((p & 0x7FFFFF) != 0).int()
    blocks = []
    for code in torch.unique(codes).tolist():
        amax = int(bits[codes == code].max())
        values = torch.arange(0, amax + 1, dtype=torch.int32, device="cuda")
        values = torch.cat([values, values | 0x8000])
        chunks = -(-values.shape[0] // (_SF_VEC - 1))
        values = torch.cat(
            [values, values.new_zeros(chunks * (_SF_VEC - 1) - values.shape[0])]
        ).reshape(chunks, _SF_VEC - 1)
        head = torch.full((chunks, 1), amax, dtype=torch.int32, device="cuda")
        blocks.append(torch.cat([head, values], 1))
    blocks = torch.cat(blocks)
    for hidden in (5120, 6144):
        same = _stage_rows(_blocks_to_rows(blocks, hidden), hidden)
        assert all(same), f"hidden {hidden}: planes equal {same}"


def test_bf16_every_value_in_non_finite_blocks():
    """All 65536 BF16 bit patterns in blocks that hold an infinity or a NaN."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    values = torch.arange(0, 1 << 16, dtype=torch.int32, device="cuda")
    values = values[torch.randperm(values.shape[0], device="cuda", generator=gen)]
    chunks = -(-values.shape[0] // (_SF_VEC - 1))
    values = torch.cat(
        [values, values.new_zeros(chunks * (_SF_VEC - 1) - values.shape[0])]
    ).reshape(chunks, _SF_VEC - 1)
    specials = torch.tensor(
        [0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF],
        dtype=torch.int32,
        device="cuda",
    )
    head = specials[torch.arange(chunks, device="cuda") % len(specials)].unsqueeze(1)
    blocks = torch.cat([values, head], 1)
    same = _stage_rows(_blocks_to_rows(blocks, 4096), 4096)
    assert all(same), f"planes equal {same}"


def test_unsupported_layouts_keep_the_torch_staging():
    """Unaligned or strided activations make the stager decline, staging nothing."""
    _, staging_kernel = _modules()
    hidden, top_k, capacity = 5120, 6, 16
    ws = _workspace(capacity, hidden, top_k)
    t = _batch(8, hidden, top_k, quantize=True, wide_ids=True, seed=1)
    before = [plane.clone() for plane in _planes(ws)]
    wide = torch.empty(8, hidden + 8, dtype=torch.bfloat16, device="cuda")
    for hidden_states in (
        wide[:, 8:].copy_(t.hidden_states),
        torch.empty(8 * hidden + 1, dtype=torch.bfloat16, device="cuda")[1:]
        .view(8, hidden)
        .copy_(t.hidden_states),
    ):
        assert not staging_kernel.stage(
            hidden_states, None, t.topk_ids, t.topk_weights, ws, quantize=True
        )
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(_planes(ws), before, strict=True))


@pytest.mark.parametrize("quantize", [True, False], ids=["bf16", "prequantized"])
def test_staging_cuda_graph(quantize):
    """A captured stager replays new batches in its input buffers to the Torch bytes."""
    staging, staging_kernel = _modules()
    hidden, top_k, capacity, count = 7168, 6, 64, 21
    ws = _workspace(capacity, hidden, top_k, shared_tiles=(128, 128), seed=2)
    t = _batch(count, hidden, top_k, quantize=quantize, wide_ids=True, seed=3)
    with mock.patch.object(staging_kernel, "stage", lambda *_, **__: False):
        staging.stage_inputs(t, ws, quantize_input=quantize)
    staging.stage_inputs(t, ws, quantize_input=quantize)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        staging.stage_inputs(t, ws, quantize_input=quantize)
    for seed in (4, 5, 6):
        fresh = _batch(
            count, hidden, top_k, quantize=quantize, wide_ids=True, seed=seed
        )
        for name in ("hidden_states", "scales", "topk_ids", "topk_weights"):
            if getattr(t, name) is not None:
                getattr(t, name).copy_(getattr(fresh, name))
        planes = _planes(ws)
        before = [plane.clone() for plane in planes]
        graph.replay()
        replayed = [plane.clone() for plane in planes]
        for plane, saved in zip(planes, before, strict=True):
            plane.copy_(saved)
        with mock.patch.object(staging_kernel, "stage", lambda *_, **__: False):
            staging.stage_inputs(t, ws, quantize_input=quantize)
        torch.cuda.synchronize()
        same = [
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(replayed, planes, strict=True)
        ]
        assert all(same), f"replay {seed}: planes equal {same}"
