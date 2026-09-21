"""Experimental prepared sparse MQA metadata and compressed BF16 logits."""
from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_sparse_mqa_metadata(sparse_indices, *, fmt="mxfp4", sparse_block_kv=8,
                                page_kv=64, use_unaligned_ks=False, starts=None,
                                ends=None, num_kv_tokens=0, context_lens=None,
                                block_table=None, request_indices=None,
                                metadata=None, workspace=None):
    """Prepare the standalone metadata operation on caller-provided indices.

    Returns a reusable plan. plan.run() generates metadata on the current stream.
    See SparseMetadataPlan for the contiguous/paged input and packed buffer ABI.
    """
    from .experimental.deepgemm_sparse_mqa.sparse_mqa import SparseMetadataPlan
    return SparseMetadataPlan(sparse_indices, fmt=fmt, sparse_block_kv=sparse_block_kv,
        page_kv=page_kv, use_unaligned_ks=use_unaligned_ks, starts=starts, ends=ends,
        num_kv_tokens=num_kv_tokens, context_lens=context_lens, block_table=block_table,
        request_indices=request_indices, metadata=metadata, workspace=workspace)


@flashinfer_experimental_api
def prepare_sparse_mqa_logits(q, sf_q, kv, sf_kv, weights, metadata_plan, *, output=None):
    """Prepare compressed logits with metadata generation included in run().

    Packed MXFP4/MXFP8 inputs produce BF16[Q,capacity*sparse_block_kv]. Only
    selected valid slots are written. Reuse a metadata plan produced by
    prepare_sparse_mqa_metadata; caller-provided output is retained in place.
    """
    from .experimental.deepgemm_sparse_mqa.sparse_mqa import SparseMqaPlan
    return SparseMqaPlan(q, sf_q, kv, sf_kv, weights, metadata_plan, output=output)
