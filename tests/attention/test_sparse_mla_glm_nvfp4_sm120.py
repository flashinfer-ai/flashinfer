# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from flashinfer.mla._sparse_mla_glm_nvfp4_sm120 import SparseMLAGlmNvfp4Sm120Wrapper


def make_inputs(tokens, heads, topk, device="cuda"):
    torch.manual_seed(19)
    rows = 256
    packed = torch.randint(0, 256, (rows, 256), dtype=torch.uint8, device=device)
    # Include subnormal scale codes, zero blocks and different binary exponents.
    codes = torch.tensor(
        [0, 1, 4, 8, 10, 12, 24, 26, 28], dtype=torch.uint8, device=device
    )
    sf = codes[torch.randint(len(codes), (rows, 32), device=device)]
    rope = torch.randn(rows, 64, dtype=torch.bfloat16, device=device)
    cache = torch.cat((packed, sf, rope.view(torch.uint8)), dim=1).view(-1, 64, 1, 416)
    scale = torch.tensor([0.7], device=device)
    q = torch.randn(tokens, heads, 576, dtype=torch.bfloat16, device=device)
    indices = torch.randint(rows, (tokens, topk), dtype=torch.int32, device=device)
    indices[:, ::13] = -1
    lengths = torch.full((tokens,), topk, dtype=torch.int32, device=device)
    lengths[0] = 0
    if tokens > 1:
        lengths[1] = max(1, topk - 19)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device,
    )
    unpacked = torch.stack((packed & 15, packed >> 4), -1).reshape(rows, 32, 16)
    latent = (
        lut[unpacked.long()]
        * sf.contiguous().view(torch.float8_e4m3fn).float()[..., None]
        * scale
    ).reshape(rows, 512)
    decoded = torch.cat((latent, rope.float()), -1)
    return q, cache, indices, scale, lengths, decoded


def reference(q, decoded, indices, lengths, sink=None):
    kv = decoded[indices.clamp_min(0).long()]
    scores = torch.einsum("thd,tkd->thk", q.float(), kv) * 0.1
    valid = (indices >= 0) & (
        torch.arange(indices.shape[-1], device=q.device) < lengths[:, None]
    )
    scores.masked_fill_(~valid[:, None], -torch.inf)
    if sink is not None:
        scores = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1)
    lse = scores.logsumexp(-1) / torch.log(torch.tensor(2.0, device=q.device))
    p = scores.softmax(-1).nan_to_num()[..., : indices.shape[1]]
    return torch.einsum("thk,tkd->thd", p, kv[..., :512]), lse


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("route,topk", [("decode", 65), ("sg", 128)])
@pytest.mark.parametrize("with_sink", [False, True])
def test_glm_nvfp4_reference_and_graph(heads, route, topk, with_sink):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Requires SM120/SM121")
    q, cache, indices, scale, lengths, decoded = make_inputs(3, heads, topk)
    sink = torch.randn(heads, device=q.device) if with_sink else None
    expected, expected_lse = reference(q, decoded, indices, lengths, sink)
    output = torch.empty(3, heads, 512, dtype=q.dtype, device=q.device)
    runner = SparseMLAGlmNvfp4Sm120Wrapper(device=q.device)

    def run():
        return runner.run(
            q,
            cache,
            indices,
            output,
            0.1,
            kv_global_scale=scale,
            topk_length=lengths,
            attn_sink=sink,
            kernel_variant=route,
            return_lse=True,
        )

    lse = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.05)
    torch.testing.assert_close(lse, expected_lse, atol=0.05, rtol=0.01)
    before = output.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, before[0]) and torch.equal(lse, before[1])
    # Warming another shape must retain addresses used by the first graph.
    runner.run(
        q[:1],
        cache,
        indices[:1],
        output[:1],
        0.1,
        kv_global_scale=scale,
        topk_length=lengths[:1],
        kernel_variant="decode",
    )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, before[0]) and torch.equal(lse, before[1])


def test_glm_nvfp4_rejects_incompatible_layout():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Requires SM120/SM121")
    q, cache, ix, scale, lengths, _ = make_inputs(3, 8, 65)
    out = torch.empty(3, 8, 512, dtype=q.dtype, device=q.device)
    runner = SparseMLAGlmNvfp4Sm120Wrapper(device=q.device)
    with pytest.raises(ValueError, match="divisible by 64"):
        runner.run(q, cache, ix, out, 0.1, kv_global_scale=scale, kernel_variant="sg")
    with pytest.raises(ValueError, match="page_size=64"):
        runner.run(q, cache.view(-1, 32, 1, 416), ix, out, 0.1, kv_global_scale=scale)
