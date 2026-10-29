.. _apigdn_paged:

flashinfer.gdn_paged
====================

Gated Delta-Rule prefill and decode on a paged recurrent-state cache. The
caller owns one state pool ``state_pool[num_pages, H, V, K]`` and gives every
sequence a list of page ids in CSR form, as for a paged KV cache. Page ``j`` of
a sequence holds the recurrent state after the last token of that page, so a
new request that shares a full-page prefix with an earlier one can start from
the last shared page instead of recomputing the prefix.

.. currentmodule:: flashinfer.gdn_paged

.. autosummary::
    :toctree: ../generated

    chunk_gated_delta_rule_paged
    gated_delta_rule_decode_paged
    gdn_paged_prefill_slots
    gdn_paged_decode_slots
    GDNPagedPrefillSlots
    GDNPagedDecodeSlots

Page table
----------

A sequence ``b`` owns pages ``page_indices[page_indptr[b]:page_indptr[b + 1]]``
(any page ids, in any order) and has
``(num_pages_b - 1) * page_size + last_page_len[b]`` tokens after the call. Its
new tokens in ``q``/``k``/``v`` are the last ones; the tokens before them are
its prefix. With ``page_size = 1024``, a sequence of 2500 tokens owns three
pages and ``last_page_len = 452``:

.. code-block:: text

    tokens:   [0, 1024)   [1024, 2048)   [2048, 2500)
    page:     p0          p1             p2  (partial, last_page_len = 452)
    state:    after 1024  after 2048     after 2500

``chunk_gated_delta_rule_paged`` derives every pool row on the device in one
small launch, without a host synchronization. The rows depend only on the
batch, so a model can compute them once per step with
``gdn_paged_prefill_slots`` (``gdn_paged_decode_slots`` for decode) and pass
them as ``slots`` to every layer's call, which then skips that launch. The
``slots`` are contiguous int32 ``[num_seqs]`` tensors on the device of ``q``;
they are not checked against the page table. The rows come from a Triton kernel; on SM107 its compilation needs a
``ptxas`` from CUDA 13.4 or newer on ``PATH`` (or in
``TRITON_PTXAS_BLACKWELL_PATH``), since older ones do not know ``sm_107a``.
The call:

- It reads the initial state from the page that holds the last prefix token,
  or starts from the zero state when the prefix is empty.
- It writes the state at every page boundary the new tokens cross to the page
  that ends there, and the final state to the last page.
- It never writes a page that ends at or before the prefix, so full pages
  shared with other sequences stay unchanged. A prefix that ends inside a page
  continues that page in place.

Prefix lengths need not be multiples of the 64-token chunk, so chunked prefill
can split a prompt at any token. A prefix that is not a multiple of 64 costs
one more chunk of work, because the kernels pad the first chunk on the left.
``page_size`` must be a positive multiple of 64. Zero-length sequences leave
the pool unchanged.

``gated_delta_rule_decode_paged`` decodes one token per sequence (``T = 1``).
When the token opens a new page (``last_page_len[b] == 1``) it reads the
previous page and writes the new one; otherwise it updates the last page in
place. Every sequence must hold at least one token before the decoded one, so a
sequence whose token opens a new page owns at least two pages.

A pool that both entry points use must have the 4D layout
``[num_pages, H, V, K]`` (not the 5D pool of
:func:`flashinfer.gdn_decode.ugpu_state_pool`) and a state dtype both support.
The page tables hold real page ids only: ``-1`` is not a padding value. To pad
a batch, for example to a CUDA graph's batch size, give each padded sequence
its own scratch page. A padded prefill sequence has no new tokens and one
scratch page. A padded decode sequence uses a scratch page with
``last_page_len >= 2``.

.. code-block:: python

    # Prefill a 2500-token prompt whose first 2048 tokens are cached in pages
    # [7, 3] (shared with another request); page 12 is new.
    out = flashinfer.chunk_gated_delta_rule_paged(
        q, k, v, g, beta,                      # the 452 new tokens
        cu_seqlens=torch.tensor([0, 452], dtype=torch.int32, device="cuda"),
        state_pool=pool,                       # [num_pages, H, 128, 128]
        page_indices=torch.tensor([7, 3, 12], dtype=torch.int32, device="cuda"),
        page_indptr=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        page_size=1024,
        last_page_len=torch.tensor([452], dtype=torch.int32, device="cuda"),
    )
    # Reads page 3, writes page 12; pages 7 and 3 are unchanged.

Slot arguments of ``chunk_gated_delta_rule``
--------------------------------------------

The paged prefill is a thin wrapper over three keyword arguments of
:func:`flashinfer.gdn_prefill.chunk_gated_delta_rule`, which callers with other
cache layouts can use directly:

``output_state_indices``
    Pool row of each sequence's final state, separate from the read row
    ``state_indices``. A read row of ``-1`` starts from the zero state; a write
    row of ``-1`` skips the final-state store. A zero-length sequence writes
    nothing.

``checkpoint_state_indices``
    Pool rows of the checkpoints: the ``j``-th checkpoint of sequence ``i``
    goes to row ``checkpoint_state_indices[checkpoint_cu_starts[i] + j]`` of
    ``state_checkpoints``, and a row of ``-1`` skips it. This also serves one
    selected checkpoint per sequence: give every other entry ``-1``.

``prefix_lens``
    Number of tokens each sequence has processed before this call. Checkpoints
    are taken at absolute positions: after new token ``n`` when
    ``(prefix_lens[i] + n) % checkpoint_every_n_tokens == 0``. Sequence ``i``
    then has ``(prefix_lens[i] + seq_len_i) // N - prefix_lens[i] // N``
    checkpoints (``N = checkpoint_every_n_tokens``).

These arguments need ``state_indices``, an ``output_state`` buffer and
``output_final_state=True`` for ``output_state_indices``, and
``checkpoint_every_n_tokens > 0`` for ``checkpoint_state_indices`` and
``prefix_lens``. Each is a contiguous int32 or int64 tensor on the device of
``q``.

``initial_state``, ``output_state`` and ``state_checkpoints`` may be the same
pool. With ``checkpoint_state_indices`` every pool row must be one dense
``[H, V, K]`` block; the row stride may be padded. Rows written by one sequence
must not be read or written by another sequence of the same call. The native non-context-parallel kernels (SM8x,
SM90, SM100/SM103/SM107 and SM120) serve these arguments; ``use_cp=True``,
``backend="cake_gdn"`` and ``backend="cudnn"`` reject them.
