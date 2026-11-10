# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

_SYMBOLS = {
    "sparse_attention": (".training", "sparse_attention"),
    "sparse_attention_score_recompute": (".training", "sparse_attention_score_recompute"),
    "SparseAttentionForward": (".sparse_attention_forward", "SparseAttentionForward"),
    "sparse_attention_forward_wrapper": (".sparse_attention_forward", "sparse_attention_forward_wrapper"),
    "AlignedHCABackward": (".aligned_hca_backward", "AlignedHCABackward"),
    "aligned_hca_backward_wrapper": (".aligned_hca_backward", "aligned_hca_backward_wrapper"),
    "SparseAttentionBackward": (".sparse_attention_backward", "SparseAttentionBackward"),
    "sparse_attention_backward_wrapper": (".sparse_attention_backward", "sparse_attention_backward_wrapper"),
    "IndexerForward": (".indexer_forward", "IndexerForward"),
    "indexer_forward_wrapper": (".indexer_forward", "indexer_forward_wrapper"),
    "indexer_forward_top_k_wrapper": (".indexer_forward", "indexer_forward_top_k_wrapper"),
    "compress_topk_cand_buffer_size": (".indexer_forward", "compress_topk_cand_buffer_size"),
    "compress_topk_cand_buffer_size_thd": (".indexer_forward", "compress_topk_cand_buffer_size_thd"),
    "IndexerTopK": (".indexer_top_k", "IndexerTopK"),
    "indexer_top_k_wrapper": (".indexer_top_k", "indexer_top_k_wrapper"),
    "local_to_global_wrapper": (".indexer_top_k", "local_to_global_wrapper"),
    "compactify_wrapper": (".indexer_top_k", "compactify_wrapper"),
    "SparseIndexerScoreRecompute": (".score_recompute", "SparseIndexerScoreRecompute"),
    "sparse_indexer_score_recompute_wrapper": (".score_recompute", "sparse_indexer_score_recompute_wrapper"),
    "SparseAttnScoreRecompute": (".score_recompute", "SparseAttnScoreRecompute"),
    "sparse_attn_score_recompute_wrapper": (".score_recompute", "sparse_attn_score_recompute_wrapper"),
    "DenseIndexerScoreRecompute": (".score_recompute", "DenseIndexerScoreRecompute"),
    "dense_indexer_score_recompute_wrapper": (".score_recompute", "dense_indexer_score_recompute_wrapper"),
    "DenseAttnScoreRecompute": (".score_recompute", "DenseAttnScoreRecompute"),
    "dense_attn_score_recompute_wrapper": (".score_recompute", "dense_attn_score_recompute_wrapper"),
    "IndexerBackward": (".indexer_backward", "IndexerBackward"),
    "indexer_backward_wrapper": (".indexer_backward", "indexer_backward_wrapper"),
    "DenseIndexerBackward": (".indexer_backward", "DenseIndexerBackward"),
    "dense_indexer_backward_wrapper": (".indexer_backward", "dense_indexer_backward_wrapper"),
}


def _load_symbol(name):
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    if name == "DSA":
        return DSA
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class DSANamespace:
    def __getattr__(self, name):
        if name in _SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"DSA has no attribute {name!r}")


DSA = DSANamespace()

__all__ = ["DSA", *_SYMBOLS.keys()]
