.. _kv-layout:

KV-Cache Layout in FlashInfer
=============================

Layout: NHD/HND
---------------

FlashInfer provides two layouts for last 3 dimensions in KV-Cache: ``NHD`` and ``HND``:

- ``NHD``: the last 3 dimensions are organized as ``(seq_len, num_heads, head_dim)``.
- ``HND``: the last 3 dimensions are organized as ``(num_heads, seq_len, head_dim)``.

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
When there is no need to change the sequence length (e.g. in prefilling stage), we can use ``RaggedTensor``
with a single ragged (variable length) dimension to store the key/value tensors in KV-Cache:

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

.. _cascade-qo-indptr-layout:

Multi-level Cascade Inference Query/Output Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using multi-level `cascade inference <https://flashinfer.ai/2024/02/02/cascade-inference.html>`_,
the query and output of each level are stored in ragged tensors, each level's ``qo_indptr`` array stores
the interval information of each node in the cascade tree at that level, the figure below shows the
``qo_indptr`` for each level in cascade inference:

.. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/tutorials/cascade_qo_indptr.png
  :width: 800
  :align: center
  :alt: The ``qo_indptr`` for each level in cascade inference.

Note that each level's ``qo_indptr`` array should start from 0, and the last element of the ``qo_indptr`` array
should be equal to the sum of length for all query/output tensors.

FlashInfer APIs
~~~~~~~~~~~~~~~

FlashInfer provides :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` to compute
the prefill attention between queries stored in ragged tensor and keys/values stored in ragged
KV-Cache.

.. _mask-layout:

Mask Layout (2D Ragged Tensor)
------------------------------

The aforementioned Ragged Tensor can be generalized to multiple "ragged" dimensions. For example,
the attention mask in FlashInfer is a 2D ragged tensor for batch size greater than 1:

.. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/tutorials/mask-layout.png
  :width: 800
  :align: center
  :alt: Data structure of Mask Layout.

When number of requests is greater than 1, different request might have different query length and kv length.
To avoid padding, we use a 2D ragged tensor to store attention mask. The input ``qo_indptr`` and
``kv_indptr`` arrays (both with length ``num_requests+1``) are used to store the information of
variable sequence lengths of each request,
``qo_indptr[i+1]-qo_indptr[i]`` is the query length of request ``i`` (``qo_len[i]``),
``kv_indptr[i+1]-kv_indptr[i]`` is the kv length of request ``i`` (``kv_len[i]``).

The mask array of all requests are flattened (with query as the first dimension, and kv as last dimension)
and concatenated into a single 1D array: ``mask_data``. FlashInfer will create a ``qk_indptr`` array implicitly
to store the start offset of each request's mask in the flattened mask array: ``qk_indptr[1:] = cumsum(qo_len * kv_len)``.

``mask_data`` has shape ``(qk_indptr[-1],)``, we can use ``mask_data[qk_indptr[i]:qk_indptr[i+1]]`` to slice the flattened
mask of request ``i``.

To save memory, we can further packes the boolean flattened boolean mask array into a bit-packed array (1 bit per element, 8 elements
are packed together as a `uint8`) with "little" bit-order (see `numpy.packbits <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>`_
for more details). FlashInfer accepts both boolean mask and bit-packed mask. If boolean mask is provided, FlashInfer will pack it into bit-packed
array internally.

FlashInfer APIs
~~~~~~~~~~~~~~~

:class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` and :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`
allow user to specify ``qo_indptr``, ``kv_indptr`` and custom attention mask ``custom_mask`` in ``begin_forward`` functions,
the mask data will be added to the attention score before softmax (and after softmax scaling) in the attention kernel.

:meth:`flashinfer.quantization.packbits` and :meth:`flashinfer.quantization.segment_packbits` are the utility functions
to pack boolean mask into bit-packed array.

.. _page-layout:

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

The ``kv_data`` tensor could either be a single 5-D tensor or a tuple of 4-D tensors,
when stored in a single tensor, ``kv_data`` has shape:

.. code:: python

  (max_num_pages, 2, page_size, num_heads, head_dim) # NHD layout
  (max_num_pages, 2, num_heads, page_size, head_dim) # HND layout

when stored in a tuple of tensors, ``kv_data = (k_data, v_data)``, and each one of them has shape:

.. code:: python

  (max_num_pages, page_size, num_heads, head_dim) # NHD layout
  (max_num_pages, num_heads, page_size, head_dim) # HND layout

where ``max_num_pages`` is the maximum number of pages used by all requests, ``page_size`` is the number of tokens
we fit into each page. ``2`` in single tensor storage means K/V (first one for keys, the second one for values).

FlashInfer APIs
~~~~~~~~~~~~~~~

:meth:`flashinfer.page.append_paged_kv_cache` can append a batch of keys/values (stored as ragged tensors) to the paged KV-Cache
(the pages for these appended keys/values must be allocated prior to calling this API).

:class:`flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper` and :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` implements the decode attention
and prefill/append attention between queries stored in ragged tensors and keys/values stored in paged KV-Cache.

FAQ
^^^

How do FlashInfer manages KV-Cache?
  FlashInfer itself is not responsible for managing the page-table (pop and allocate new pages, etc.) and we leave the strategy
  to the user: different serving engine might have different strategies to manage the page-table. FlashInfer is only responsible
  for computing the attention between queries and keys/values stored in KV-Cache.
