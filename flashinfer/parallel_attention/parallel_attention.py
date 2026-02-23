import logging
from typing import Callable, Tuple, Union
from .attention_ops import AttentionOpManager
from .parallel_wrapper import ulysses_wrapper, ring_wrapper
from .parallel_config import AttnParallelConfig, UnevenCPConfig, VarlenCPConfig
import torch

logger = logging.getLogger(__name__)

class ParallelAttention:
    def __init__(self, attn_type: str, attn_parallel_config: AttnParallelConfig, uneven_cp_config: UnevenCPConfig, varlen_cp_config: VarlenCPConfig, fuse_qkv: bool = False):
        self.attn_type = attn_type
        self.attn_impl = AttentionOpManager.get_impl(attn_type)
        self.attn_parallel_config = attn_parallel_config
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
        if is_causal:
            raise NotImplementedError("parallel attention does not support causal attention right now")

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
