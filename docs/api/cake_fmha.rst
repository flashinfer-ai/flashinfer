.. _apicake_fmha:

flashinfer.cake_fmha
====================

``cake_fmha`` is the versioned Cake implementation of FlashInfer's conventional
TensorRT-LLM paged FMHA decode and context contracts.  It is an explicit
Blackwell backend: importing FlashInfer or calling the existing APIs without a
backend continues to select the existing FlashInfer implementation.

The public functions support B200/GB200 (SM100a, CUDA 12.8+) and B300/GB300
(SM103a, CUDA 12.9+).  They accept the same arguments and return the same values
as :func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` and
:func:`flashinfer.prefill.trtllm_batch_context_with_kv_cache`.

The checked-in source product contains the optimized Cake route portfolio, a
complete-domain compatibility component, and the DCP speculative-decode
add-on.  One content-addressed manifest pins all source files, public C ABIs,
the base capability matrix, and the FlashInfer revision against which the
matrix was audited.  FlashInfer authenticates every standalone artifact before
JIT or AOT compilation.  :func:`cake_fmha_manifest` returns a defensive copy
of that product record.

All 1,798 optimized cells have authenticated high-level adapters for their
complete component chains.  The selector accepts the pinned matrix's normalized
NHD context views and device scalar FP8/NVFP4 scales.  Its numerically inert
``1e-30`` skip-softmax probe is canonicalized to ordinary softmax only after an
exact optimized match; other nonzero thresholds remain compatibility routes.
Selector misses, insufficient route workspace, and NVFP4 adapter load failures
fail closed to ``compat_v1``.

The BF16 context adapter also contains two measured single-mask profiles for
batch-four HND calls: uniform q511/KV2047 with P32 shared page tables, and
uniform q257/KV1024 with P1024 separate K/V page tables.  FlashInfer selects
these bodies only after every semantic and shape guard matches and the four KV
lengths plus five Q-indptr values confirm the exact uniform lengths.  A
nonuniform length, a near-miss shape, or CUDA graph capture retains the generic
authenticated context body; it never reuses a fixed-length specialization.

Optimized routes are fail-closed.  In particular, optimized FP8 decode is
qualified for HND pages, a shared K/V page table, and GQA group size eight;
other valid FP8 decode shapes remain Cake-owned and use the authenticated
complete-domain component.

The manifest's per-route counts are inventory metadata, not a proof that a
particular high-level selector revision reproduces the pinned matrix.  The
checked-in allocation-free replay therefore enumerates the independent pinned
capability corpus (80,768 raw cells, 57,280 valid cells), calls the actual
high-level selectors, and authenticates every case/route pair with canonical
SHA-256 ``d47bf01c2d27409c6a39759d02e30bb9df65e98c353f53d7335081dd26b3f3a8``.
It requires exactly 1,798 optimized cells and 55,482 ``compat_v1`` cells, with
the same per-route accounting as the manifest.  Both a fresh source checkout
and the installed wheel must execute this replay and then exercise selected
routes on SM100a and SM103a; representative family tests and manifest-count
accounting do not replace those gates.

The distributed-context-parallel feature remains additive.  Supplying
``causal_seqlens_kv_global`` to :func:`cake_batch_decode_with_kv_cache` selects
the authenticated ``cake_fmha_dcp_spec`` profile; ordinary calls continue to
select conventional FMHA.  The DCP JIT cache key includes the same root
manifest digest plus an authenticated FlashInfer-adapter digest and uses exact
SM100a or SM103a targets, so the add-on cannot silently drift from the base
package.

.. currentmodule:: flashinfer.cake_fmha

The SM103a request-ordered BF16-query/FP8-KV path also accepts an explicit
six-part plan for six-query decode with 32 query heads and two KV heads,
without returning LSE. Each workgroup handles the 16 query heads associated
with one KV head, so the authenticated plan has grid ``(6, 2, batch_size)``.
The logical tensor layout, completion-buffer allocation and descriptor
capture API remain unchanged. Call
``flashinfer.plan_cake_fmha_request_ordered_paged_decode`` before graph capture
to obtain the current plan; a plan carrying the previous four-workgroup grid
does not authenticate against this exported route.

.. autosummary::
    :toctree: ../generated

    cake_batch_decode_with_kv_cache
    cake_batch_context_with_kv_cache
    cake_fmha_manifest
    get_cake_fmha_module

The same implementation can also be selected on the existing APIs with
``backend="cake"``::

    output = flashinfer.trtllm_batch_decode_with_kv_cache(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_kv_len,
        backend="cake",
    )

Single-request six-query split route
------------------------------------

For one request with six query tokens, 32 query heads, two KV heads,
head dimension 256 and page size 64, explicitly setting ``num_kv_splits=76``
selects the O-only split route. This route requires at least 43,671,552 bytes
in ``workspace_buffer`` and the existing separate 24-element completion buffer.
The default route and LSE-returning routes are unchanged.

FP8-query request-ordered route
------------------------------

An explicit ``query_dtype=torch.float8_e4m3fn`` plan selects a separate
FP8 E4M3-query/FP8 E4M3-KV route with BF16 output. It supports batches
64, 128, 160, 192, 224 and 256, six query tokens per request, 32 query heads, two KV heads, head dimension
256 and page size 64 on a 152-SM SM103 device. Every KV length must be at least
six; split execution and LSE output are unavailable for this export.

Queries are contiguous ``[batch_size * 6, 32, 256]``. K/V use HND views
``[pages, 2, 64, 256]`` with strides ``[32768, 256, 512, 1]``, directly viewing
native ``[pages, 64, 2, 256]`` storage. Use shared contiguous int32 page tables,
a device int32 request permutation, and device FP32 log2 QK/output scales.
The kernel reads these caller buffers directly without external gather or
scatter. The default BF16-query planner and its existing routes are unchanged.

The existing ``CakeFmhaRequestOrderedCapture`` protocol applies. Give each live
graph binding its own 128-byte-aligned uint8 workspace of at least 388 bytes,
warm an ordinary invocation, record with the capture object, and finalize it
before replay. Keep its workspace alive with the graph. Tensor contents may
change in place while their storage, shape and minimum-length contract remain
valid.
