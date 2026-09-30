"""Numerical/layout oracles for native V4.1 vision support operations."""
import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x


def test_rectangular_rope_and_live_capacity_isolation():
    device = require_b12x()
    from b12x.attention import varlen
    from b12x.norm.vision import run_rope_qkv
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution, unfreeze_kernel_resolution,
    )

    torch.manual_seed(414)
    capacity, heads, dim = 128, 2, 64
    q, k, v = [torch.empty(capacity, heads, dim, device=device,
                           dtype=torch.bfloat16) for _ in range(3)]
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim // 2, 2, device=device,
                                         dtype=torch.float32) / (dim // 2)))
    cu = torch.zeros(2, dtype=torch.int32, device=device)
    kp = varlen.create_plan(q, k, v, cu, max_seqlen_q=capacity,
                            max_seqlen_k=capacity, causal=False)
    sp = varlen.plan(kp)
    spec, = sp.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = sp.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                      max_seqlen_q=capacity, max_seqlen_k=capacity, causal=False)
    for iteration, (height, width) in enumerate(((7, 11), (3, 5), (5, 3))):
        rows = height * width
        qkv = torch.randn(rows, 3 * heads * dim, device=device, dtype=torch.bfloat16)
        # Poison capacity tails so accidental attention to a different image
        # cannot be hidden by benign zeros or nearly identical values.
        q.fill_(float("nan"))
        k.fill_(float("nan"))
        v.fill_(float("nan"))
        if iteration:
            freeze_kernel_resolution("vision live image sizes reuse planned attention")
        try:
            run_rope_qkv(qkv, height, width, inv, q=q, k=k, v=v, cu_seqlens=cu)
            actual, _ = varlen.run(binding=binding)
        finally:
            unfreeze_kernel_resolution()
        torch.testing.assert_close(cu, torch.tensor([0, rows], device=device, dtype=torch.int32))
        pos_h = torch.arange(height, device=device)[:, None].expand(height, width)
        pos_w = torch.arange(width, device=device)[None, :].expand(height, width)
        angles = (torch.stack((pos_h, pos_w), -1).reshape(rows, 2, 1).float()
                  * inv).flatten(1)[:, None]
        expected = []
        for source in qkv.chunk(3, -1)[:2]:
            first, second = source.view(rows, heads, dim).float().chunk(2, -1)
            expected.append(torch.cat((first * angles.cos() - second * angles.sin(),
                                       second * angles.cos() + first * angles.sin()), -1).bfloat16())
        torch.testing.assert_close(q[:rows], expected[0], atol=0.008, rtol=0.008)
        torch.testing.assert_close(k[:rows], expected[1], atol=0.008, rtol=0.008)
        expected_v = qkv.chunk(3, -1)[2].reshape(rows, heads, dim)
        torch.testing.assert_close(v[:rows], expected_v, atol=0, rtol=0)
        scores = torch.einsum("thd,shd->hts", expected[0].float(), expected[1].float()) * dim**-0.5
        oracle = torch.einsum("hts,shd->thd", scores.softmax(-1), expected_v.float()).bfloat16()
        torch.testing.assert_close(actual[:rows], oracle, atol=0.016, rtol=0.02)


@pytest.mark.parametrize("height,width", [(4, 7), (1, 5)])
def test_channel_major_unfold_with_odd_padding(height, width):
    device = require_b12x()
    from b12x.norm.vision import run_spatial_merge

    channels, ratio = 5, 3
    x = torch.arange(height * width * channels, device=device).reshape(-1, channels).bfloat16()
    out = torch.empty(((height + 2) // 3 * ((width + 2) // 3), channels * 9),
                       device=device, dtype=torch.bfloat16)
    run_spatial_merge(x, height, width, ratio=ratio, out=out)
    image = x.reshape(height, width, channels).permute(2, 0, 1)
    padded = F.pad(image, (0, -width % ratio, 0, -height % ratio))
    oracle = F.unfold(padded[None].float(), ratio, stride=ratio)[0].T.bfloat16()
    torch.testing.assert_close(out, oracle, atol=0, rtol=0)


def test_default_gelu_not_tanh_and_graph_replay():
    device = require_b12x()
    from b12x.norm.vision import run_gelu

    x = torch.linspace(-5, 5, 4096, device=device).bfloat16().reshape(16, 256)
    out = torch.empty_like(x)
    run_gelu(x, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_gelu(x, out=out)
    x.neg_()
    graph.replay()
    oracle = F.gelu(x, approximate="none")
    torch.testing.assert_close(out, oracle, atol=2e-5, rtol=0.008)
    # Around the negative tail, tanh-GELU differs by orders of this tolerance.
    negative = (x < -3) & (x > -4)
    torch.testing.assert_close(out[negative], oracle[negative], atol=2e-6, rtol=0.01)
