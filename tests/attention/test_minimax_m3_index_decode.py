# SPDX-License-Identifier: Apache-2.0
"""Public capability, eager preparation, and changing-length graph contract."""

import pytest
import torch
import flashinfer


def test_cpu_inputs_keep_framework_stock():
    q = torch.empty((8, 1, 128), dtype=torch.bfloat16)
    cache = torch.empty((128, 128, 128), dtype=torch.bfloat16)
    table = torch.zeros((8, 128), dtype=torch.int32)
    lengths = torch.full((8,), 1280, dtype=torch.int32)
    assert not flashinfer.minimax_m3_index_decode_supported(
        q, cache, table, lengths, 16384
    )


@pytest.mark.parametrize("batch", [8, 16])
def test_prepared_graph_replays_changing_lengths(batch, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Specialized indexer requires Hopper")
    q = torch.randn((batch, 1, 128), dtype=torch.bfloat16, device="cuda")
    cache = torch.randn((128, 128, 128), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(128, dtype=torch.int32, device="cuda").repeat(batch, 1)
    lengths = torch.full((batch,), 1280, dtype=torch.int32, device="cuda")
    output = torch.empty((1, batch + 8, 16), dtype=torch.int32, device="cuda")
    args = (q, cache, table, lengths, 16384)
    assert flashinfer.minimax_m3_index_decode_supported(*args, out=output)
    flashinfer.minimax_m3_index_decode_warmup(*args, out=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flashinfer.minimax_m3_index_decode(*args, out=output)
    assert result.data_ptr() == output.data_ptr()
    for length in [0, 1, 1280, 2048, 2049, 4096, 16384, 1280]:
        lengths.fill_(length)
        graph.replay()
        torch.cuda.synchronize()
        actual = result.clone()
        monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
        assert not flashinfer.minimax_m3_index_decode_supported(*args)
        expected = flashinfer.minimax_m3_index_decode(*args)
        monkeypatch.delenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE")
        torch.testing.assert_close(actual == -1, expected == -1, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
        )
