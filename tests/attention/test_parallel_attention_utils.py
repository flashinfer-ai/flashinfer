import pytest
import torch

from flashinfer.parallel_attention import ring_varlen_config, split_varlen_input


@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_split_varlen_input_handles_sequences_shorter_than_world_size(
    tensor_layout,
):
    """Keep trailing ranks empty and pad both supported tensor layouts."""
    values = torch.arange(7, dtype=torch.float32)
    tensor = (
        values.reshape(1, 7, 1) if tensor_layout == "HND" else values.reshape(7, 1, 1)
    )
    chunk_dim = 1 if tensor_layout == "HND" else 0

    expected = (
        [0, 2, 3],
        [1, 4, 5],
        [6, 0, 0],
        [0, 0, 0],
    )
    for rank, expected_values in enumerate(expected):
        local = split_varlen_input(
            tensor,
            seq_len_list=[2, 5],
            world_size=4,
            rank=rank,
            tensor_layout=tensor_layout,
        )

        assert local.shape[chunk_dim] == 3
        assert local.flatten().tolist() == expected_values


def test_split_varlen_input_preserves_even_split_behavior():
    """Retain the existing partition for evenly divisible sequences."""
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)

    chunks = [
        split_varlen_input(tensor, [4, 8], world_size=4, rank=rank).flatten().tolist()
        for rank in range(4)
    ]

    assert chunks == [
        [0, 4, 5],
        [1, 6, 7],
        [2, 8, 9],
        [3, 10, 11],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA metadata")
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "seq_lens_q,seq_lens_kv",
    [
        ([2, 5], [1, 3]),
        ([0, 2, 5], [3, 0, 1]),
        ([4, 8], [8, 12]),
        ([1021, 1024, 1027], [750, 826, 1024]),
    ],
)
def test_ring_varlen_config_matches_split_chunks(
    monkeypatch, world_size, seq_lens_q, seq_lens_kv
):
    """Check real CUDA metadata with simulated process-group dimensions."""
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    cu_q, cu_kv, max_q, max_kv = ring_varlen_config(
        seq_lens_q, seq_lens_kv, ring_group=object()
    )

    for seq_lens, cu_seqlens, max_seqlen in (
        (seq_lens_q, cu_q, max_q),
        (seq_lens_kv, cu_kv, max_kv),
    ):
        assert cu_seqlens.device == torch.device("cuda:0")
        assert cu_seqlens.dtype == torch.int32
        assert max_seqlen == max(
            (length + world_size - 1) // world_size for length in seq_lens
        )
        values = torch.arange(1, sum(seq_lens) + 1, device="cuda:0")
        sequences = values.split(seq_lens)
        for rank, row in enumerate(cu_seqlens.cpu().tolist()):
            # Python slices provide an independent boundary reference,
            # including empty slices beyond the end of short sequences.
            expected_chunks = []
            for sequence in sequences:
                width = (len(sequence) + world_size - 1) // world_size
                expected_chunks.append(sequence[rank * width : (rank + 1) * width])
            lengths = [len(chunk) for chunk in expected_chunks]
            assert row == [0] + torch.tensor(lengths).cumsum(0).tolist()
            local = split_varlen_input(
                values.reshape(-1, 1, 1),
                seq_lens,
                world_size,
                rank,
                tensor_layout="NHD",
            ).flatten()
            torch.testing.assert_close(local[: row[-1]], torch.cat(expected_chunks))
            assert torch.count_nonzero(local[row[-1] :]).item() == 0
