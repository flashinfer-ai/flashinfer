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
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        joint_seq_length=0,
        valid_joint_seq_length=None,
        joint_strategy="none",
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
    ):
        attn_inputs = {
            "query": query,
            "key": key,
            "value": value,
            "tensor_layout": tensor_layout,
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "enable_gqa": enable_gqa,
            "return_lse": return_lse,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
        }

        return self.attn_impl(**attn_inputs)
