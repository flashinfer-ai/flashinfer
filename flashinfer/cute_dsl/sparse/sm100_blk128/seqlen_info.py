from dataclasses import dataclass

from cutlass import Int32


"""
Simplified sequence length info for non-varlen mode.
No cu_seqlens support — seqlen is always static.
"""


@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    @staticmethod
    def create(
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
    ):
        return SeqlenInfoQK(
            seqlen_q_static,
            seqlen_k_static,
        )
