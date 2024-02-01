.. _kv-layout:

KV-Cache Layout in FlashInfer
=============================

Layout: NHD/HND
---------------

FlashInfer provides two layouts for last 3 dimensions in KV-Cache: ``NHD`` and ``HND``:

- ``NHD``: the last 3 dimensions are organized as ``(seq_len, num_heads, head_dim)``.
- ``HND``: the last 3 dimensions are organized as ``(num_heads, head_dim, seq_len)``.

The ``NHD`` layout is more natural because it's consistent with the output of
:math:`xW_k` and :math:`xW_v` without transpose. The ``HND`` layout is more friendly
for GPU implementation when KV-Cache uses low-precision data type (e.g. fp8).
In practice we don't observe significant performance difference between these two layouts
on fp16 kV-Cache and we prioritize ``NHD`` layout for better readability. FlashInfer implements
Attention kernels on both layouts and we provide an option to select between them (``NHD``
by default).

.. _ragged-layout:

Ragged Tensor
-------------

In batched inference/serving, the input sequence length may vary across different samples.
When there is no need to change the sequence length (e.g. in prefilling stage), we can use ``RaggedTensor`` to store
the key/value tensors in KV-Cache:

.. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/tutorials/ragged.png
  :width: 400
  :align: center
  :alt: Data structure of Ragged KV-Cache.

The keys (or values) of all requests are packed into a single ``data`` tensor without padding,
we use a ``indptr`` array (``num_requests+1`` elements, the first element is always zero)
to store the information of variable sequence lengths of each request
(``indptr[i+1]-indptr[i]`` is the sequence length of request ``i``), the ``data`` tensor has
shape ``(indptr[-1], num_heads, head_dim)`` when the layout is ``NHD``.

We can use ``data[indptr[i]:indptr[i+1]]`` to slice the keys (or values) of request ``i``.

.. _page-layout:

FlashInfer APIs
~~~~~~~~~~~~~~~

FlashInfer provides :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` to compute
the prefill attention between queries stored in ragged tensor and keys/values stored in ragged
KV-Cache.

Page Table
----------

When KV-Cache is dynamic (e.g. in append or decode stage), packing all keys/values is not
efficient because the sequence length per request changes over time. `vLLM <https://arxiv.org/pdf/2309.06180.pdf>`_ 
proposes to organize KV-Cache as a Page Table. In FlashInfer, we treat the page-table as
a block sparse matrix (each used page can be viewed as an non-zero block in block sparse matrix)
and uses the `CSR format <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
to index the pages in KV-Cache.

.. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/tutorials/page_layout.png
  :width: 800
  :align: center
  :alt: Data structure of Paged KV-Cache.

For each request, we keep an record of its ``page_indices``, ``last_page_len`` which
tracks the pages used by this request and the number of entries in the last page. The KV
sequence length of request ``i`` is ``page_size * (len(page_indices[i]) - 1) + last_page_length[i]``.

.. note::
  The ``last_page_len`` of each request must be greater than zero, and less than or equal to ``page_size``.

The overall ``kv_indptr`` array (with length ``num_requests+1``) can be computed as:
``[0, len(page_indices[0]), len(page_indices[0])+len(page_indices[1]), ...]``.
The overall ``kv_page_indices`` array (with length ``kv_indptr[-1]``) is the concatenation of all requests' ``page_indices``.
The overall ``kv_last_page_lens`` array (with length ``num_requests``) is the concatenation of all requests' ``last_page_length``.

The ``kv_data`` tensor is a 5-D tensor with shape (in ``NHD`` layout):

.. code::

  (max_num_pages, 2, page_size, num_heads, head_dim)

where ``max_num_pages`` is the maximum number of pages used by all requests, ``page_size`` is the number of tokens
we fit into each page. ``2`` is the number of slots in each page (first one for keys, the second one for values).

FlashInfer APIs
~~~~~~~~~~~~~~~

:meth:`flashinfer.page.append_paged_kv_cache` can append a batch of keys/values (stored as ragged tensors) to the paged KV-Cache
(the pages for these appended keys/values must be allocated prior to calling this API).

:class:`BatchDecodeWithPagedKVCacheWrapper` and :class:`BatchPrefillWithPagedKVCacheWrapper` implements the decode attention
and prefill/append attention between queries stored in ragged tensors and keys/values stored in paged KV-Cache.

FAQ
^^^

How do FlashInfer manages KV-Cache?
  FlashInfer itself is not responsible for managing the page-table (pop and allocate new pages, etc.) and we leave the strategy
  to the user: different serving engine might have different strategies to manage the page-table. FlashInfer is only responsible
  for computing the attention between queries and keys/values stored in KV-Cache.
