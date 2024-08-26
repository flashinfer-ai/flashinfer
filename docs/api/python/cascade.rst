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

    .. automethod:: __init__
    
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

.. autoclass:: BatchDecodeWithSharedPrefixPagedKVCacheWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BatchPrefillWithSharedPrefixPagedKVCacheWrapper
    :members:

    .. automethod:: __init__

