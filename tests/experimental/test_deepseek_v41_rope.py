# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import deepseek_v41_rope


@pytest.mark.parametrize("shape", [(1, 64, 512), (5, 32, 128), (17, 1, 512)])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("token_padding", [0, 4096])
def test_rope_exact_complex_reference_graph_inplace_and_mask(
    shape, inverse, token_padding
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("V4.1 RoPE requires SM100/SM103")
    torch.manual_seed(41811)
    storage = torch.full(
        (shape[0], shape[1] * shape[2] + token_padding),
        37,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = storage[:, : shape[1] * shape[2]].view(shape)
    x.copy_(torch.randn_like(x))
    phase = torch.randn(65537, 32, device="cuda") * 100
    freqs = torch.view_as_real(torch.polar(torch.ones_like(phase), phase))
    positions = torch.randint(65537, (shape[0],), device="cuda", dtype=torch.int32)
    out = torch.full_like(x, 17)
    expected = out.clone()

    def run():
        return deepseek_v41_rope(x, freqs, positions, inverse=inverse, out=out)

    def reference(values):
        pairs = torch.view_as_complex(
            values[..., -64:].float().reshape(*values.shape[:-1], 32, 2)
        )
        rotations = torch.view_as_complex(freqs)[positions.clamp_min(0).long(), None, :]
        if inverse:
            rotations = rotations.conj()
        output = values.clone()
        output[..., -64:] = torch.view_as_real(pairs * rotations).flatten(-2).bfloat16()
        return output

    def verify():
        active = positions >= 0
        expected[active] = reference(x)[active]
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            storage[:, shape[1] * shape[2] :],
            torch.full_like(storage[:, shape[1] * shape[2] :], 37),
            atol=0,
            rtol=0,
        )

    run()
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(3):
        x.copy_(torch.randn_like(x))
        positions.copy_(
            torch.randint(65537, (shape[0],), device="cuda", dtype=torch.int32)
        )
        positions[step % shape[0]] = -1
        freqs.copy_(freqs.flip(-1))
        graph.replay()
        verify()
    positions.fill_(65536)
    allocated = deepseek_v41_rope(x, freqs, positions, inverse=inverse)
    assert allocated.is_contiguous()
    torch.testing.assert_close(allocated, reference(x), atol=0, rtol=0)
    inplace = x.contiguous()
    expected_inplace = reference(inplace)
    deepseek_v41_rope(inplace, freqs, positions, inverse=inverse, out=inplace)
    torch.testing.assert_close(inplace, expected_inplace, atol=0, rtol=0)
    if token_padding and shape[0] > 1:
        # Same data pointer with a different token stride is not in-place.
        alias = torch.as_strided(x, shape, (shape[1] * shape[2], shape[2], 1))
        with pytest.raises(ValueError, match="alias"):
            deepseek_v41_rope(x, freqs, positions, out=alias)
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    storage = torch.empty(x.numel() + 2, device="cuda", dtype=x.dtype)
    with pytest.raises(ValueError, match="alias"):
        deepseek_v41_rope(
            storage[:-2].view_as(x), freqs, positions, out=storage[2:].view_as(x)
        )


def test_rope_token_stride_beyond_two_gibibytes():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("V4.1 RoPE requires SM100/SM103")
    stride = (1 << 30) + 512
    if torch.cuda.mem_get_info()[0] < (stride + 512) * 2 + (64 << 20):
        pytest.skip("large-offset regression requires a little over2GiB free")
    storage = torch.empty(stride + 512, device="cuda", dtype=torch.bfloat16)
    x = storage.as_strided((2, 1, 512), (stride, 512, 1))
    x[0].fill_(1)
    x[1].fill_(2)
    freqs = torch.zeros(2, 32, 2, device="cuda")
    freqs[:, :, 0] = 1
    positions = torch.arange(2, device="cuda", dtype=torch.int32)
    out = deepseek_v41_rope(x, freqs, positions)
    torch.testing.assert_close(out, x.contiguous(), atol=0, rtol=0)
