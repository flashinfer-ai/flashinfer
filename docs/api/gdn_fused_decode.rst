.. _apigdn_fused_decode:

flashinfer.gdn_fused_decode_step
================================

Fused single-token Gated Delta-Rule decode step over paged conv/ssm state
pools. It folds the whole per-layer serving chain — the b/a projection GEMV,
the depthwise causal conv1d state update, the q/k/v head split and the gated
delta-rule decode of :func:`~flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`
— into one operation, and updates both state pools in place.

The op takes **no backend argument and no environment gate**: whether a call
runs one of the specialized SM120 kernels or the composable torch path is
decided by the library from the workload registry and the device.
:func:`gdn_fused_decode_step_supported` reports that decision up front,
cheaply and capture-safely, so a framework can keep its own composition for
the shapes this op does not accelerate.

The implementation (dispatch, workload registry and kernels) lives under
``flashinfer/gdn_kernels/experimental/``; see its ``README.md`` for the
registry schema and the impl-module interface. *Experimental* describes where
that code lives, not how the op is called.

.. currentmodule:: flashinfer

.. autosummary::
    :toctree: ../generated

    gdn_fused_decode_step
    gdn_fused_decode_step_supported
