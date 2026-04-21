import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class UnevenCPConfig:
    """Configuration for uneven context parallelism.

    Handles the case where the total sequence length is not evenly divisible
    by the number of ranks. Each rank may hold a different number of tokens,
    and the last rank typically gets fewer tokens (the remainder).

    The parallel wrappers use this information to truncate padding and zero
    out extra output positions on the last rank.

    Use the :func:`uneven_cp_config` utility function to compute
    ``seq_len_cur_ring_group`` via ``all_gather``, then pass the result
    to this dataclass.

    Attributes:
        seq_len: Actual (unpadded) total sequence length.
        seq_len_padded: Padded total sequence length (divisible by
            ``world_size``).
        seq_len_cur_ring_group: Tensor of per-rank sequence lengths within
            the current ring group, shape ``[ring_size]``. ``None`` when
            ``ring_size == 1`` (no ring parallelism).

    Example::

        # Total sequence length 1023, world_size = 8
        # Each rank gets ceil(1023/8) = 128 tokens, except last rank gets 127
        ring_group, ulysses_group = get_parallel_groups(ulysses_size=2, ring_size=4)
        seq_len_cur_ring_group = uneven_cp_config(
            seq_len=1023,
            seq_len_padded=1024,
            seq_len_cur_rank=128 if rank < 7 else 127,
            ulysses_group=ulysses_group,
            ring_group=ring_group,
        )
        config = UnevenCPConfig(
            seq_len=1023,
            seq_len_padded=1024,
            seq_len_cur_ring_group=seq_len_cur_ring_group,
        )
    """

    seq_len: Optional[int] = None
    seq_len_padded: Optional[int] = None
    seq_len_cur_ring_group: Optional[torch.Tensor] = None

    def reset(self):
        self.seq_len = None
        self.seq_len_padded = None
        self.seq_len_cur_ring_group = None


@dataclass
class VarlenCPConfig:
    """Configuration for variable-length context parallelism.

    Handles the case where multiple sequences of different lengths are packed
    together (varlen). Cumulative sequence length arrays
    (``cu_seqlens``) are computed so that the attention kernel can correctly
    identify sequence boundaries.

    Supports two modes (mutually exclusive):

    - **Ulysses-only** (``ring_size == 1``): The packed sequences are treated
      as a whole and split across heads via all-to-all. No per-sequence
      splitting is needed — only the overall ``cu_seqlens`` are stored so the
      attention kernel knows where each sequence starts and ends.
    - **Ring-only** (``ulysses_size == 1``): Each individual sequence is split
      across ranks along the sequence dimension. ``cu_seqlens`` are stored as
      a 2D tensor of shape ``[ring_size, num_seqs + 1]``, one row per rank,
      because each rank holds a different slice of every sequence.

    Attributes:
        cu_seqlens_q_cur_ulysses_group: Cumulative query sequence lengths for
            the current ulysses group (shared across all ulysses ranks).
        cu_seqlens_kv_cur_ulysses_group: Cumulative key/value sequence lengths
            for the current ulysses group.
        max_seq_len_q_cur_ulysses_group: Max query sequence length in the
            current ulysses group.
        max_seq_len_kv_cur_ulysses_group: Max key/value sequence length in the
            current ulysses group.
        cu_seqlens_q_cur_ring_group: Cumulative query sequence lengths for all
            ranks in the current ring group, shape ``[ring_size, num_seqs + 1]``.
        cu_seqlens_kv_cur_ring_group: Cumulative key/value sequence lengths for
            all ranks in the current ring group.
        max_seq_len_q_cur_ring_group: Max query sequence length across all
            ranks in the ring group (per-rank padded length).
        max_seq_len_kv_cur_ring_group: Max key/value sequence length across all
            ranks in the ring group (per-rank padded length).
    """

    cu_seqlens_q_cur_ulysses_group: Optional[torch.Tensor] = None
    cu_seqlens_kv_cur_ulysses_group: Optional[torch.Tensor] = None
    max_seq_len_q_cur_ulysses_group: Optional[int] = None
    max_seq_len_kv_cur_ulysses_group: Optional[int] = None
    cu_seqlens_q_cur_ring_group: Optional[torch.Tensor] = None
    cu_seqlens_kv_cur_ring_group: Optional[torch.Tensor] = None
    max_seq_len_q_cur_ring_group: Optional[int] = None
    max_seq_len_kv_cur_ring_group: Optional[int] = None

    def set_varlen_cp_config(
        self,
        cu_seqlens_q_all_ranks,
        cu_seqlens_kv_all_ranks,
        max_seq_len_q,
        max_seq_len_kv,
        ulysses_group,
        ring_group,
    ):
        ring_size = (
            torch.distributed.get_world_size(ring_group)
            if ring_group is not None
            else 1
        )
        ulysses_size = (
            torch.distributed.get_world_size(ulysses_group)
            if ulysses_group is not None
            else 1
        )

        if ring_size == 1:
            self.cu_seqlens_q_cur_ulysses_group = cu_seqlens_q_all_ranks
            self.cu_seqlens_kv_cur_ulysses_group = cu_seqlens_kv_all_ranks
            self.max_seq_len_q_cur_ulysses_group = max_seq_len_q
            self.max_seq_len_kv_cur_ulysses_group = max_seq_len_kv
            return

        if ulysses_size == 1:
            self.cu_seqlens_q_cur_ring_group = cu_seqlens_q_all_ranks
            self.cu_seqlens_kv_cur_ring_group = cu_seqlens_kv_all_ranks
            self.max_seq_len_q_cur_ring_group = max_seq_len_q
            self.max_seq_len_kv_cur_ring_group = max_seq_len_kv
            return

        raise NotImplementedError(
            "Varlen CP only supported when ulysses_size == 1 or ring_size == 1"
        )

    def reset(self):
        self.cu_seqlens_q_cur_ulysses_group = None
        self.cu_seqlens_kv_cur_ulysses_group = None
        self.max_seq_len_q_cur_ulysses_group = None
        self.max_seq_len_kv_cur_ulysses_group = None
        self.cu_seqlens_q_cur_ring_group = None
        self.cu_seqlens_kv_cur_ring_group = None
        self.max_seq_len_q_cur_ring_group = None
        self.max_seq_len_kv_cur_ring_group = None
