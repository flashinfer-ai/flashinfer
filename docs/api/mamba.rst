.. _apimamba:

flashinfer.mamba
================

Mamba / Mamba-2 state-space-model kernels. These wrap the selective scan and
state-update primitives used in SSM blocks.

.. currentmodule:: flashinfer.mamba

.. autosummary::
    :toctree: ../generated

    selective_state_update
    checkpointing_ssu
    ssd_combined_fwd
    replayssm_materialize

SSD scan selection
------------------

:func:`ssd_combined_fwd` accepts the keyword-only, per-call argument
``scan_algorithm``:

* ``"auto"`` (default) preserves existing dispatch, including eligible shallow
  and prefix-factorized packed-sequence specializations.
* ``"exact_scan"`` bypasses those specializations and selects the scan that
  evaluates decay from differences of cumulative exponents. This avoids the
  separately exponentiated prefix factors and reciprocals that can overflow
  or underflow under strong decay.

The selector changes the implementation, not the recurrence: ``dt_bias``,
``dt_softplus``, ``dt_limit``, initial states, and checkpoint boundaries retain
their existing meanings. In particular, keep ``dt_limit=(0.0, float("inf"))``
when nonnegative processed step sizes are required; changing the clamp to force
a dispatch route is unnecessary. ``"exact_scan"`` does not promise bitwise
agreement with other implementations or higher-precision state storage, and
may have different performance from ``"auto"``. Automatic dispatch is unchanged
and does not detect or repair every numerical overflow in the prefix route.

The functional API always uses Cake. For a reusable :class:`SSDCombined`
runner, construct it with ``backend="cake"`` and pass the selector to
:meth:`SSDCombined.run`, not to the constructor. The same per-call selector is
accepted by the underlying ``CakeSSDCombined.run``. A CuTe runner accepts only
``"auto"``; explicit ``"exact_scan"`` selection raises ``ValueError`` rather
than switching backends. Unsupported selector values also raise ``ValueError``.
Omitting the argument preserves existing calls, including positional arguments.

Example on a Cake-supported SM100/SM103 GPU, using BF16 inputs and FP16 state:

.. code-block:: python

    import torch
    from flashinfer.mamba import ssd_combined_fwd

    batch, seqlen, nheads, ngroups = 1, 128, 128, 8
    x = torch.randn(batch, seqlen, nheads, 64, device="cuda", dtype=torch.bfloat16)
    dt = torch.zeros(batch, seqlen, nheads, device="cuda", dtype=torch.float32)
    A = -torch.ones(nheads, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, seqlen, ngroups, 128, device="cuda", dtype=torch.bfloat16)
    C = torch.randn_like(B)
    initial_states = torch.zeros(
        batch, nheads, 64, 128, device="cuda", dtype=torch.float16
    )

    output, final_states = ssd_combined_fwd(
        x, dt, A, B, C,
        initial_states=initial_states,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        scan_algorithm="exact_scan",
    )

Stateful SSD runner
-------------------

.. autoclass:: SSDCombined
    :members: run
