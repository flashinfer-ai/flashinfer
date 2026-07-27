.. _experimental:

Experimental APIs and Backends
==============================

FlashInfer distinguishes stable functionality from **experimental**
functionality intended for fast-moving work such as client-GPU (e.g. SM12x)
kernels:

- an **experimental API** is a public interface that may change or disappear;
- an **experimental backend** is an implementation that is not yet ready for
  stable support.

A stable API may expose an experimental backend through explicit opt-in
(e.g. an explicit ``backend=`` argument), and experimental APIs are marked
with an experimental warning in their documentation.

Enabling experimental features
------------------------------

All experimental behavior requires an explicit opt-in:

.. code-block:: bash

   export FLASHINFER_ENABLE_EXPERIMENTAL_FEATURES=1

Without this variable, experimental APIs raise ``RuntimeError`` and stable
APIs never route to experimental backends. The variable *permits*
experimental functionality; it does not *select* it — routing to an
experimental backend additionally requires explicit selection at the call
site. The first use of an experimental API in a process emits an
``ExperimentalWarning``.

User contract
-------------

Experimental APIs and backends provide **no compatibility or long-term
support guarantees** and may change or be removed without deprecation. They
are intended primarily for main-branch users, community containers, and
local framework integrations, and should not be enabled by default in
supported framework releases.

Each experimental feature's documentation identifies whether the API, the
backend, or both are experimental; the supported use cases and limitations;
and any explicit backend-selection requirement.

Experimental features are JIT-only: they are not included in the
``flashinfer-jit-cache`` / ``flashinfer-cubin`` pre-built packages.

For contributors
----------------

The full policy — placement rules, admission criteria, ownership and
lifecycle requirements, and the graduation checklist — lives in
``flashinfer/experimental/README.md`` in the repository, with a summary in
``CONTRIBUTING.md``.
