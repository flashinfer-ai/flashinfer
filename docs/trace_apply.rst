.. _trace_apply:

Trace Apply
===========

Trace Apply substitutes selected FlashInfer API calls with custom *solutions* at
runtime, with no changes to the calling code (SGLang, vLLM, your own script).
It is the consumer side of the FlashInfer Trace: you collect ``fi_trace``
definitions, author solutions for them, and Trace Apply dispatches the matching
solution whenever the corresponding API is called.

It is deterministic and explicit: there is **one solution per definition** — no
workload filtering, autotuning, speed ranking, or policy selection.

Enabling
--------

Trace Apply is driven entirely by two environment variables (no other hidden
knobs):

================================ ===========================================================
Variable                         Meaning
================================ ===========================================================
``FLASHINFER_TRACE_APPLY``       ``1`` enables Trace Apply at ``import flashinfer``.
``FLASHINFER_TRACE_PATH``        Path to the folder of solutions to apply (see below).
``FLASHINFER_TRACE_APPLY_DEBUG`` ``1`` logs the extracted axes + dtypes on each newly
                                 resolved shape (for authoring/debugging solutions).
================================ ===========================================================

.. code-block:: bash

   export FLASHINFER_TRACE_APPLY=1
   export FLASHINFER_TRACE_PATH=/path/to/solutions_folder
   python my_serving_script.py    # FlashInfer calls are transparently substituted

Enabling at import is best-effort: a bad or missing config emits a warning and
``import flashinfer`` still succeeds (Trace Apply simply stays off).

Programmatic control:

.. code-block:: python

   import flashinfer.trace_apply as ta

   ta.enable_apply("/path/to/solutions_folder")  # returns # of wrapped attributes
   ...
   ta.stats()                 # per-API dispatch counts (hit / fallback / error)
   ta.explain("flashinfer.norm.rmsnorm", {"hidden_size": 4096})  # why a call routes
   ta.disable_apply()         # restore the original FlashInfer APIs

The folder layout
-----------------

``FLASHINFER_TRACE_PATH`` points at a directory in the flashinfer-bench layout::

    <root>/
      definitions/**/*.json    # problem specs: axes, inputs (incl. dtype), fi_api tag
      solutions/**/*.json       # implementations — one solution per file
      workloads/ traces/ ...    # ignored by apply

``solutions/`` provides the kernels to substitute; ``definitions/`` provides the
routing identity. Other subdirectories are ignored. A folder may hold many
solutions, and **all of them are applied at once** — one kernel substituted per
definition (this is how multiple kernels are replaced in a single model run).
The active set is exactly what is present under ``solutions/``.

Loading is strict: a malformed definition or solution file raises (Trace Apply
never silently skips something you asked it to apply). Two solutions for the
same definition keep the first and warn.

Routing
-------

A call is routed by its **definition identity** ``(fi_api, const-axes,
input-dtypes)``:

* ``const-axes`` are the compile-time shape a definition is specialized for
  (``hidden_size=4096``, ``head_dim=128`` …). *Var* axes (``batch_size``,
  sequence lengths, …) are **not** part of the identity — one solution handles
  all of them.
* ``input-dtypes`` distinguishes dtype-specialized solutions (fp16 vs bf16) for
  the same shape.

The routing decision is cached per shape, so steady-state dispatch is a dict
lookup. Inside a CUDA-graph capture, only already-resolved shapes apply (warm up
eagerly before capturing).

Solutions
---------

A solution is a :class:`flashinfer.trace.Solution`: a name, the definition it
implements, an author, a :class:`flashinfer.trace.BuildSpec`, and inlined source
files. Loaders are chosen by language *family*:

* **Python family** (``python``, ``triton``, ``cutedsl``, ``tilelang``) — the
  entry-point source is imported and the entry function is called by keyword
  (Definition input names).
* **C++/CUDA family** (``cpp``, ``cuda``, ``cutlass``, ``cutile``) — built via
  ``flashinfer.jit`` and called positionally (the ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``
  symbol).

Output adaptation reconciles the solution's outputs with the API's convention:
value-returning, ``out=`` buffers, in-place / destination-passing, and optional
outputs gated by a runtime flag (e.g. ``return_lse``).

Arch safety: if a solution's ``target_hardware`` lists explicit ``sm<NN>``
targets, it is only applied on a matching GPU (compute capability is read from
the device — there is no GPU product-name table).

Error policy
------------

Trace Apply is **strict**: a *matched* solution that fails to build or run
re-raises, so a broken registered solution surfaces immediately. A genuine miss
(no registered solution for the call's identity, or filtered out by arch/author)
falls back to the original FlashInfer API.
