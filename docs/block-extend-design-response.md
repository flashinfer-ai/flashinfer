# Block-diffusion prefill design

## Goal

Block diffusion uses this visibility rule:

    floor((q_offset + q_idx) / block_size)
      >= floor((kv_offset + kv_idx) / block_size)

Queries can attend to their own block and all earlier blocks.  The kernel skips
fully invisible KV tiles at CTA granularity and computes the boundary mask in
registers, so no materialized mask buffer is needed.

## Public API

There is no flashinfer.dllm package and no dLLM-specific wrapper class.  The
feature is exposed only as an option on existing prefill APIs:

    single_prefill_with_kv_cache(
        q, k, v, block_diffusion=True, block_size=block_size,
        q_offset=q_offset, kv_offset=kv_offset,
    )

    BatchPrefillWithRaggedKVCacheWrapper(
        workspace, block_diffusion=True, block_size=block_size,
    )

    BatchPrefillWithPagedKVCacheWrapper(
        workspace, block_diffusion=True, block_size=block_size,
    )

The batch wrappers receive per-request q_offsets and kv_offsets during plan().
The option is incompatible with custom masks, sliding windows, CUDA-graph
dynamic offset allocation, and non-FA2/FA3 backends.

## Compilation isolation

Existing shared customize generators compile only mask modes 0 through 3.
Block diffusion is routed to a dedicated generator only when its fixed mask
mode is kBlockExpanding.  This prevents a fifth mask-mode specialization from
being added to existing prefill URI products.

The dedicated product is intentionally small:

- fp16 and bf16 only;
- QK and value head dimensions 64 or 128 only;
- ragged and paged batch forms;
- FA2 and FA3, with FA3 requiring SM90.

The dedicated URI includes idtype for batch modules, preventing int32 and
int64 specializations from aliasing in the JIT cache.

Private implementation modules hold the variant declarations and JIT argument
construction.  They are not imported from the flashinfer package root and do
not expose a separate attention API, module cache, or AOT path.

## Correctness coverage

tests/attention/test_block_diffusion_attention.py calls only the native public
interfaces above.  It compares single, ragged, and paged executions with a
custom-mask reference and covers:

- regular and incremental offsets;
- partially visible and fully invisible KV ranges;
- heterogeneous batch offsets;
- FA2 and FA3 where Hopper is available;
- the dedicated-generator routing boundary.

The H100 regression includes fully invisible rows followed by regular rows in
the same batch.  This protects the Hopper zero-visible-KV path from both a
TMA synchronization hang and incorrect nonzero softmax output.
