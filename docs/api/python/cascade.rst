.. _apicascade:

flashinfer.cascade
==================

.. currentmodule:: flashinfer.cascade

.. _api-merge-states:

Merge Attention States
----------------------

.. autosummary::
   :toctree: ../../generated

   merge_state
   merge_state_in_place
   merge_states

.. _api-cascade-attention:

Cascade Attention
-----------------

.. autosummary::
   :toctree: ../../generated

   batch_decode_with_shared_prefix_padded_kv_cache


Cascade Attention Wrapper Classes
---------------------------------

.. autoclass:: BatchDecodeWithSharedPrefixPagedKVCacheWrapper
    :members:


.. autoclass:: BatchPrefillWithSharedPrefixPagedKVCacheWrapper
    :members:

