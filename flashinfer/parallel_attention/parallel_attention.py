import logging

import torch

from .attention_ops import AttentionOpManager
from .parallel_config import UnevenCPConfig, VarlenCPConfig
from .parallel_wrapper import ring_wrapper, ulysses_wrapper

logger = logging.getLogger(__name__)


class ParallelAttention:
    """Runs an attention backend with Ulysses and/or Ring parallelism.

    Wraps any registered attention implementation (see :class:`AttentionOpManager`)
    and transparently applies Ulysses (all-to-all head splitting) and Ring
    (P2P KV exchange with online softmax merging) parallelism via decorators.

    Args:
        attn_type: Name of the registered attention backend (e.g. ``"flash-attn3"``).
        ulysses_group: Ulysses process group.
        ring_group: Ring process group.
        uneven_cp_config: Configuration for uneven context parallelism where
            sequence lengths are not evenly divisible across ranks.
        varlen_cp_config: Configuration for variable-length context parallelism
            where multiple sequences of different lengths are packed together.
        fuse_qkv: If ``True``, fuse Q/K/V into a single all-to-all communication
            in Ulysses parallelism (reduces 3 NCCL calls to 1).

    Example::

        config = AttnParallelConfig()
        config.set_config(ulysses_size=2, ring_size=2)
        attn = ParallelAttention(
            attn_type="flash-attn3",
            ulysses_group=ulysses_group,
            ring_group=ring_group,
        )
        output = attn.run(query, key, value, tensor_layout="HND")
    """

    def __init__(
        self,
        attn_type: str,
        ulysses_group: torch.distributed.ProcessGroup,
        ring_group: torch.distributed.ProcessGroup,
        uneven_cp_config: UnevenCPConfig = None,
        varlen_cp_config: VarlenCPConfig = None,
        fuse_qkv: bool = False,
    ):
        self.attn_type = attn_type
        self.attn_impl = AttentionOpManager.get_impl(attn_type)
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group
        self.uneven_cp_config = uneven_cp_config
        self.varlen_cp_config = varlen_cp_config
        self.fuse_qkv = fuse_qkv

    @ulysses_wrapper
    @ring_wrapper
    def run(
        self,
        query,
        key,
        value,
        tensor_layout,
        attn_mask=None,
        is_causal=False,
        return_lse=False,
        cur_rank_cu_seqlens_q=None,
        cur_rank_cu_seqlens_k=None,
        cur_rank_max_seqlen_q=0,
        cur_rank_max_seqlen_k=0,
        **kwargs,
    ):
        """Run parallel attention on the local rank's portion of Q/K/V.

        The Ulysses and Ring wrappers transparently handle communication
        before and after this method is called.

        Args:
            query: Query tensor, shape ``[H, S, D]`` (HND) or ``[S, H, D]`` (NHD).
            key: Key tensor, same layout as query.
            value: Value tensor, same layout as query.
            tensor_layout: ``"HND"`` or ``"NHD"``.
            attn_mask: Optional attention mask (not yet supported).
            is_causal: Whether to apply causal masking (not yet supported).
            return_lse: Must be ``False``; internally managed by ring wrapper.
            cur_rank_cu_seqlens_q/ cur_rank_cu_seqlens_k/
            cur_rank_max_seqlen_q/ cur_rank_max_seqlen_k:
            please do not set this manually. This will be set by the parallel wrapper.
            The sequence lengths should be set in the uneven_cp_config or varlen_cp_config.
            **kwargs: Additional arguments forwarded to the attention backend.

        Returns:
            torch.Tensor: Attention output for the local rank, same layout as input.
        """
        if is_causal:
            raise NotImplementedError(
                "parallel attention does not support causal attention right now"
            )

        attn_inputs = {
            "query": query,
            "key": key,
            "value": value,
            "tensor_layout": tensor_layout,
            "attn_mask": attn_mask,
            "is_causal": is_causal,
            "return_lse": return_lse,
            "cur_rank_cu_seqlens_q": cur_rank_cu_seqlens_q,
            "cur_rank_cu_seqlens_k": cur_rank_cu_seqlens_k,
            "cur_rank_max_seqlen_q": cur_rank_max_seqlen_q,
            "cur_rank_max_seqlen_k": cur_rank_max_seqlen_k,
        }

        return self.attn_impl(**attn_inputs, **kwargs)
