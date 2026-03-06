"""Tests for the seq_chunk_cumsum CUDA kernel.

The kernel replaces the Python-side computation in _SSDKernel.run()
(ssd_combined.py:383-403) that computes per-sequence logical chunk
ranges for varlen parallelization.
"""

import torch


def seq_chunk_cumsum_reference(
    seq_idx, chunk_indices, chunk_offsets, chunk_size, num_seqs
):
    """Python reference: compute seq_chunk_cumsum from varlen metadata.

    Extracted from flashinfer/mamba/ssd_combined.py:391-402.

    Args:
        seq_idx: int32 tensor [1, total_seqlen] — sequence ID per position
        chunk_indices: int32 tensor [num_logical_chunks] — physical chunk per logical chunk
        chunk_offsets: int32 tensor [num_logical_chunks] — offset within physical chunk
        chunk_size: int — chunk size
        num_seqs: int — number of sequences

    Returns:
        seq_chunk_cumsum: int32 tensor [num_seqs + 1] — exclusive prefix sum of
            per-sequence chunk counts.  seq_chunk_cumsum[s] = number of logical
            chunks in sequences 0..s-1.
    """
    seq_ids = seq_idx[0, chunk_indices.long() * chunk_size + chunk_offsets.long()]
    counts = torch.zeros(num_seqs, dtype=torch.int32, device=seq_idx.device)
    counts.scatter_add_(0, seq_ids.int(), torch.ones_like(seq_ids, dtype=torch.int32))
    seq_chunk_cumsum = torch.zeros(
        num_seqs + 1, dtype=torch.int32, device=seq_idx.device
    )
    torch.cumsum(counts.int(), dim=0, out=seq_chunk_cumsum[1:])
    return seq_chunk_cumsum


def _make_equal_seqs(num_seqs, chunks_per_seq, chunk_size):
    """Helper: equal-length sequences, each chunk-aligned."""
    total_seqlen = num_seqs * chunks_per_seq * chunk_size
    seq_idx = torch.zeros(1, total_seqlen, dtype=torch.int32, device="cuda")
    for s in range(num_seqs):
        start = s * chunks_per_seq * chunk_size
        end = start + chunks_per_seq * chunk_size
        seq_idx[0, start:end] = s

    # Each sequence contributes chunks_per_seq logical chunks, all chunk-aligned
    chunk_indices = []
    chunk_offsets = []
    for s in range(num_seqs):
        for c in range(chunks_per_seq):
            phys = s * chunks_per_seq + c
            chunk_indices.append(phys)
            chunk_offsets.append(0)

    chunk_indices = torch.tensor(chunk_indices, dtype=torch.int32, device="cuda")
    chunk_offsets = torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda")
    return seq_idx, chunk_indices, chunk_offsets


def _make_variable_seqs(chunks_per_seq_list, chunk_size):
    """Helper: variable-length sequences (each chunk-aligned)."""
    num_seqs = len(chunks_per_seq_list)
    total_chunks = sum(chunks_per_seq_list)
    total_seqlen = total_chunks * chunk_size
    seq_idx = torch.zeros(1, total_seqlen, dtype=torch.int32, device="cuda")
    pos = 0
    for s, n in enumerate(chunks_per_seq_list):
        length = n * chunk_size
        seq_idx[0, pos : pos + length] = s
        pos += length

    chunk_indices = []
    chunk_offsets = []
    phys = 0
    for _s, n in enumerate(chunks_per_seq_list):
        for _c in range(n):
            chunk_indices.append(phys)
            chunk_offsets.append(0)
            phys += 1

    chunk_indices = torch.tensor(chunk_indices, dtype=torch.int32, device="cuda")
    chunk_offsets = torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda")
    return seq_idx, chunk_indices, chunk_offsets, num_seqs


class TestSeqChunkCumsum:
    """Test the seq_chunk_cumsum CUDA kernel against the Python reference."""

    CHUNK_SIZE = 128

    def _call_kernel(self, seq_idx, chunk_indices, chunk_offsets, chunk_size, num_seqs):
        from flashinfer.mamba.ssd_combined import _get_seq_chunk_cumsum_module

        module = _get_seq_chunk_cumsum_module()
        num_logical_chunks = len(chunk_indices)
        output = torch.zeros(num_seqs + 1, dtype=torch.int32, device="cuda")
        module.seq_chunk_cumsum(
            seq_idx,
            chunk_indices,
            chunk_offsets,
            output,
            chunk_size,
            num_logical_chunks,
            num_seqs,
        )
        return output

    def test_single_seq(self):
        """1 sequence, 4 chunks."""
        seq_idx, chunk_indices, chunk_offsets = _make_equal_seqs(1, 4, self.CHUNK_SIZE)
        ref = seq_chunk_cumsum_reference(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 1
        )
        out = self._call_kernel(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 1
        )
        torch.testing.assert_close(out, ref)

    def test_equal_seqs(self):
        """4 sequences, 8 chunks each."""
        seq_idx, chunk_indices, chunk_offsets = _make_equal_seqs(4, 8, self.CHUNK_SIZE)
        ref = seq_chunk_cumsum_reference(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 4
        )
        out = self._call_kernel(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 4
        )
        torch.testing.assert_close(out, ref)

    def test_variable_seqs(self):
        """Sequences with different chunk counts: [1, 5, 2, 8]."""
        chunks_list = [1, 5, 2, 8]
        seq_idx, chunk_indices, chunk_offsets, num_seqs = _make_variable_seqs(
            chunks_list, self.CHUNK_SIZE
        )
        ref = seq_chunk_cumsum_reference(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, num_seqs
        )
        out = self._call_kernel(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, num_seqs
        )
        torch.testing.assert_close(out, ref)

    def test_many_seqs(self):
        """128 sequences, 1 chunk each — stress-tests parallelism."""
        seq_idx, chunk_indices, chunk_offsets = _make_equal_seqs(
            128, 1, self.CHUNK_SIZE
        )
        ref = seq_chunk_cumsum_reference(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 128
        )
        out = self._call_kernel(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 128
        )
        torch.testing.assert_close(out, ref)

    def test_single_chunk_per_seq(self):
        """Each sequence has exactly 1 chunk."""
        seq_idx, chunk_indices, chunk_offsets = _make_equal_seqs(32, 1, self.CHUNK_SIZE)
        ref = seq_chunk_cumsum_reference(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 32
        )
        out = self._call_kernel(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 32
        )
        torch.testing.assert_close(out, ref)

    def test_many_chunks_few_seqs(self):
        """2 sequences, 64 chunks each — large scan range."""
        seq_idx, chunk_indices, chunk_offsets = _make_equal_seqs(2, 64, self.CHUNK_SIZE)
        ref = seq_chunk_cumsum_reference(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 2
        )
        out = self._call_kernel(
            seq_idx, chunk_indices, chunk_offsets, self.CHUNK_SIZE, 2
        )
        torch.testing.assert_close(out, ref)
