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

Cascade Attention Wrapper Classes
---------------------------------

.. autoclass:: MultiLevelCascadeAttentionWrapper
    :members:


.. autoclass:: BatchDecodeWithSharedPrefixPagedKVCacheWrapper
    :members:


.. autoclass:: BatchPrefillWithSharedPrefixPagedKVCacheWrapper
    :members:

