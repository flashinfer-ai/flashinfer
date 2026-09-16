.. _experimental:

Experimental APIs and Backends
==============================

FlashInfer distinguishes stable functionality from **experimental**
functionality intended for fast-moving work — client-GPU (e.g. SM12x)
kernels, new operations from the latest models, or highly specialized kernels
for specific problem sizes:

- an **experimental API** is a public interface that may change or disappear;
- an **experimental backend** is an implementation that is not yet ready for
  stable support.

Experimental APIs are marked with a warning banner in their documentation. A
stable API exposes an experimental backend by name; it routes to one
automatically only when the user opts in (below).

Opting in to experimental features
----------------------------------

Using experimental functionality is always an explicit opt-in. Two forms need
no environment variable:

- calling an API marked experimental (``@flashinfer_experimental_api``);
- passing an experimental backend by name to a stable API, e.g.
  ``backend="sm12x_cute"``.

Both emit an ``ExperimentalWarning`` once (per API, or per API/backend pair).

Automatic selection is gated. With ``backend="auto"`` — including the dispatch
heuristics and autotuning behind it — a stable API considers only stable
backends unless you set:

.. code-block:: bash

   export FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1

With the variable set, experimental backends join the automatic candidate
list. Without it, a call whose only viable backends are experimental fails
with an error that names the variable and suggests an explicit ``backend=``.

User contract
-------------

Experimental APIs and backends provide **no compatibility or long-term
support guarantees** and may change or be removed without deprecation. They
are intended primarily for main-branch users, community containers, and
local framework integrations, and should not be enabled by default in
supported framework releases.

Each experimental feature's documentation identifies whether the API, the
backend, or both are experimental; the supported use cases and limitations;
and whether the feature participates in automatic routing once enabled.

Experimental features are JIT-only: they are not included in the
``flashinfer-jit-cache`` / ``flashinfer-cubin`` pre-built packages.

For contributors
----------------

The full policy — placement rules, admission criteria, ownership and
lifecycle requirements, and the graduation checklist — lives in
``flashinfer/experimental/README.md`` in the repository, with a summary in
``CONTRIBUTING.md``.
