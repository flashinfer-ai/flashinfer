from .block_expanding import (
    block_expanding_attention,
    get_block_expanding_module,
    list_available_aot_kernels,
    print_kernel_status,
    BLOCK_EXPANDING_VARIANT_DECL,
)

from .block_expanding_tile_skip import (
    block_expanding_attention_v2,
    block_expanding_attention_v2_with_offset,
    block_expanding_attention_v3_with_offset,
    block_expanding_attention_with_offset,
    block_expanding_cascade,
    get_block_expanding_module_v2,
    get_block_expanding_module_with_offset,
    BLOCK_EXPANDING_V2_VARIANT_DECL,
    BLOCK_EXPANDING_V2_WITH_OFFSET_VARIANT_DECL,
    BLOCK_EXPANDING_V3_WITH_OFFSET_VARIANT_DECL,
)

from .batch_block_expanding import (
    BatchBlockExpandingAttentionWrapper,
    BatchBlockExpandingPagedAttentionWrapper,
    BatchBlockExpandingPagedQOffsetWrapper,
    BatchBlockExpandingRaggedQOffsetWrapper,
    batch_block_expanding_cascade,
    sglang_style_cascade_attention,
    BATCH_BLOCK_EXPANDING_VARIANT_DECL,
    _check_batch_be_aot_available,
    _get_batch_be_aot_path,
    _get_batch_be_module_uri,
)

__all__ = [
    # V1: Original version (Single Prefill)
    "block_expanding_attention",
    "get_block_expanding_module",
    "list_available_aot_kernels",
    "print_kernel_status",
    "BLOCK_EXPANDING_VARIANT_DECL",
    # V2: Tile-level skip optimization (Single Prefill, FA2)
    "block_expanding_attention_v2",
    "block_expanding_attention_v2_with_offset",
    "get_block_expanding_module_v2",
    "BLOCK_EXPANDING_V2_VARIANT_DECL",
    "BLOCK_EXPANDING_V2_WITH_OFFSET_VARIANT_DECL",
    # V3/with_offset: FA3 Hopper optimized (Single Prefill, FA3) / Generic with offset
    "block_expanding_attention_v3_with_offset",
    "block_expanding_attention_with_offset",
    "get_block_expanding_module_with_offset",
    "BLOCK_EXPANDING_V3_WITH_OFFSET_VARIANT_DECL",
    # Cascade + block expanding (SGLang 风格: causal + merge_state)
    "block_expanding_cascade",
    "batch_block_expanding_cascade",
    "sglang_style_cascade_attention",
    # Batch Prefill versions
    "BatchBlockExpandingAttentionWrapper",
    "BatchBlockExpandingPagedAttentionWrapper",
    "BatchBlockExpandingPagedQOffsetWrapper",
    "BatchBlockExpandingRaggedQOffsetWrapper",
    "BATCH_BLOCK_EXPANDING_VARIANT_DECL",
]
