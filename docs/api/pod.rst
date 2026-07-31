.. _apipod:

flashinfer.pod
==============

POD (Prefill-On-Decode) attention executes a single-request prefill kernel and
a batch-decode kernel concurrently in one launch, which is useful for serving
stacks that overlap a chunked prefill with ongoing decode requests.

.. currentmodule:: flashinfer.pod

.. autoclass:: PODWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: BatchPODWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__
