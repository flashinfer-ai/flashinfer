from .block_extend import (
    block_extend_attention_with_offset,
    block_extend_cascade,
    get_block_extend_module_with_offset,
    BLOCK_EXTEND_V2_WITH_OFFSET_VARIANT_DECL,
    BLOCK_EXTEND_V3_WITH_OFFSET_VARIANT_DECL,
)

from .batch_block_extend import (
    BatchBlockExtendPagedOffsetWrapper,
    BatchBlockExtendRaggedOffsetWrapper,
    batch_block_extend_cascade,
    sglang_style_cascade_attention,
    _BATCH_BE_OFFSET_VARIANT_DECL,
    _BATCH_BE_OFFSET_VARIANT_DECL_FA3,
    _check_batch_be_aot_available,
    _get_batch_be_aot_path,
    _get_batch_be_module_uri,
)

__all__ = [
    # Single Prefill with offset (FA2/FA3 auto-select)
    "block_extend_attention_with_offset",
    "get_block_extend_module_with_offset",
    "BLOCK_EXTEND_V2_WITH_OFFSET_VARIANT_DECL",
    "BLOCK_EXTEND_V3_WITH_OFFSET_VARIANT_DECL",
    # Cascade + block extend (SGLang style: causal + merge_state)
    "block_extend_cascade",
    "batch_block_extend_cascade",
    "sglang_style_cascade_attention",
    # Batch Prefill with offset versions
    "BatchBlockExtendPagedOffsetWrapper",
    "BatchBlockExtendRaggedOffsetWrapper",
    # Batch Offset variant declarations
    "_BATCH_BE_OFFSET_VARIANT_DECL",
    "_BATCH_BE_OFFSET_VARIANT_DECL_FA3",
]
