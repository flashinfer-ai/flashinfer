.. _apimoe_ep_communication:

flashinfer.moe_ep: Expert-Parallel Communication
================================================

.. currentmodule:: flashinfer.moe_ep

A :class:`MoEEpCommunication` dispatches each token to the ranks that own its
experts and combines the expert outputs back on the token's source rank.
:func:`create_communication` builds one from a :class:`BootstrapConfig`, which
describes the expert-parallel group, and a backend config; the backends below
run over NVLink (MNNVL) symmetric memory.

.. autosummary::
    :toctree: ../generated
    :template: autosummary/class_with_members.rst

    BootstrapConfig
    MoEEpCommunication
    MoEEpCommParams
    MoEEpDispatchResult
    NVLinkOneSidedAlltoAll
    NVLinkOneSidedConfig
    CakeAlltoAll
    CakeAlltoAllConfig
    NVLinkTwoSidedAlltoAll
    NVLinkTwoSidedConfig

.. autosummary::
    :toctree: ../generated

    create_communication
