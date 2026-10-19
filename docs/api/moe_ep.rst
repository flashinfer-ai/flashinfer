.. _apimoe_ep:

flashinfer.moe_ep
=================

.. currentmodule:: flashinfer.moe_ep

Expert-parallel communication
-----------------------------

A :class:`MoEEpCommunication` dispatches each token to the ranks that own its
experts and combines the expert outputs back on the token's source rank.
:func:`create_communication` builds one from a :class:`BootstrapConfig`, which
describes the expert-parallel group, and a backend config; the backends below
run over NVLink (MNNVL) symmetric memory.

.. autosummary::
    :toctree: ../generated

    BootstrapConfig
    MoEEpCommunication
    MoEEpCommParams
    MoEEpDispatchResult
    create_communication
    NVLinkOneSidedAlltoAll
    NVLinkOneSidedConfig
    CakeAlltoAll
    CakeAlltoAllConfig
    NVLinkTwoSidedAlltoAll
    NVLinkTwoSidedConfig

``NVLinkOneSidedAlltoAll`` and ``CakeAlltoAll`` preserve their workspace's CUDA
virtual addresses across process checkpoint/restore.  After quiescing all work,
call ``checkpoint_prepare`` to release the non-checkpointable physical MNNVL
handles, and for ``NVLinkOneSidedAlltoAll`` its CFT logical endpoints.  Then
call ``checkpoint_restore`` with a fresh communication backend before replaying
a captured CUDA graph:

.. code-block:: python

    communication.checkpoint_prepare()
    communication.checkpoint_restore(comm_backend)

Both methods are collective.  Every rank must call them in the same order, and
``comm_backend`` must reproduce the original rank and world size.
Repeated calls are no-ops after the workspace reaches the requested state.
If an exception occurs after physical handle unmapping or remapping begins,
do not retry or reuse the workspace; restart the affected rank.
