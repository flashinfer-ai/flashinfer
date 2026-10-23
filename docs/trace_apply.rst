.. _trace_apply:

Trace Apply
===========

Trace Apply substitutes selected FlashInfer API calls with custom *solutions* at
runtime, with no changes to the calling code (SGLang, vLLM, or other engines).
It is the consumer side of the FlashInfer Trace: you provide a mapping from
**definition name** to the solution to run for it, and Trace Apply dispatches
that solution whenever the corresponding API is called with the matching shape.
For any single definition there is exactly one solution — the one you registered.

Enabling
--------

Register solutions explicitly with :func:`enable_apply`. The argument is a
mapping ``{definition_name: solution}``, where a solution is either a Python
callable or a first-class :class:`~flashinfer.trace.Solution`:

.. code-block:: python

   import torch
   import flashinfer
   import flashinfer.trace_apply as fi_trace_apply

   def my_rmsnorm(hidden_states, weight, eps=1e-6):
       x = hidden_states.float()
       y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
       return (y * weight.float()).to(hidden_states.dtype)

   fi_trace_apply.enable_apply({"rmsnorm_h4096": my_rmsnorm})  # returns the # of wrapped APIs
   ...
   flashinfer.rmsnorm(x, w)          # routed to my_rmsnorm when hidden_size == 4096
   fi_trace_apply.stats()            # per-API dispatch counts (hit / fallback / error)
   fi_trace_apply.disable_apply()    # restore the original FlashInfer APIs

``enable_apply`` is idempotent — calling it again replaces the previous mapping.
With no argument (``enable_apply()``) it uses the environment configuration
below; if nothing is configured it is a no-op.

There is also an import-time hook controlled by environment variables:

=================================== ==============================================================
Variable                            Meaning
=================================== ==============================================================
``FLASHINFER_TRACE_APPLY``          Set to ``1`` to enable Trace Apply when FlashInfer is imported.
``FLASHINFER_TRACE_APPLY_PATH``     Directory the deployment-configured solutions are loaded from.
=================================== ==============================================================

If the configuration is missing or invalid, Trace Apply stays disabled and
FlashInfer continues to work normally, with a warning describing the problem.

``FLASHINFER_TRACE_APPLY_PATH`` must point at a *curated* solutions folder — its
``solutions/`` subtree is scanned recursively for one solution per definition (a
duplicate definition is an error). It is **not** the raw extraction bundle, which
also contains baseline solutions and several backends per definition.

Routing
-------

A call is routed by **definition name**. On each call the wrapper extracts the
call's *const* axes and recomputes the definition name from the live
``TraceTemplate`` — the same ``name_prefix`` + const-axis convention the trace
collector uses (e.g. ``rmsnorm`` at ``hidden_size=4096`` → ``"rmsnorm_h4096"``).
If that name is in the registered mapping, the call dispatches to its solution;
otherwise it falls back to the original FlashInfer kernel.

* **Const axes** are the compile-time shape a definition is specialized for
  (``hidden_size``, ``head_dim``, …) and form the name. **Variable axes** (batch
  size, sequence length, …) are *not* part of the name, so a single solution
  serves all of their values.
* The decision is cached per name, so steady-state dispatch is a dictionary
  lookup. During CUDA-graph capture only already-resolved shapes are applied, so
  warm up eagerly before capturing.

Solutions
---------

A solution value is either:

* a **Python callable** — invoked directly with the definition's inputs by
  keyword (value-returning); or
* a first-class :class:`~flashinfer.trace.Solution` — loaded by language family:
  the **Python family** (``python``, ``triton``, ``cutedsl``, ``cutile``,
  ``tilelang``) is imported and called by keyword; the **C++/CUDA family**
  (``cpp``, ``cuda``, ``cutlass``) is built via ``flashinfer.jit`` and called
  positionally.

Trace Apply reconciles a solution's outputs with the calling API's convention:
value-returning outputs, caller-provided ``out=`` / ``lse=`` buffers, in-place
writes (e.g. ``fused_add_rmsnorm`` writing back into its input/residual buffers),
and data-dependent arity (e.g. ``return_lse``). If a caller passes ``out=`` (e.g.
``rmsnorm(x, w, out=buf)``), the substituted solution's result is written into
``buf`` and ``buf`` is returned, exactly like the original kernel.

The ``out=`` / ``lse=`` bindings are **auto-derived from the live API signature**
(a uniform FlashInfer convention), so they are *not* recorded in the trace. Only
the non-derivable in-place bindings (which input a result is written back into)
are declared in the trace via the output ``param``.

Error policy
------------

Trace Apply is strict. A matched solution that fails to build or run raises, so a
broken solution is reported immediately rather than masked. A call with no
matching registered name falls back to the original FlashInfer API.

.. _trace_apply_sealed_packages:

Sealed kernel packages
----------------------

The curated ``FLASHINFER_TRACE_APPLY_PATH`` folder is a *development* path: its
solutions are loaded as plain JSON with **no verification**. The production path
is a **sealed kernel package** — a directory whose name is the sha256 of its own
``manifest.json``, so the digest that names a package pins every byte of it
before anything is imported::

    <root>/<sha256(manifest.json)>/
        manifest.json        the directory name is the sha256 of these bytes
        <sidecar files>      verified bytes, opaque to FlashInfer (e.g. a
                             qualification receipt)
        bundle/
            solution.json    a FlashInfer Solution
            ...              exactly the regular files the manifest lists

The digest chain, all of which must hold before any code runs:

* the package directory name equals the sha256 of ``manifest.json``;
* ``files`` maps each bundle-relative file name to its sha256, with canonical
  path rules only (no absolute paths, no ``..``, no NUL bytes);
* ``bundle_sha256`` equals the sha256 of the sorted ``"name\0digest\n"`` listing,
  so the listing itself cannot be edited to match a tampered file;
* the bundle holds exactly the files the manifest lists — a missing file, an
  extra file, a symbolic link, or a special file anywhere in the package refuses
  verification;
* sidecar files outside the bundle are authenticated by ``sealed_files``
  (a mapping of package-relative names to sha256 digests). An earlier
  package format's ``qualification_sha256`` key is accepted as an alias for
  ``sealed_files = {"qualification.json": ...}``, so existing packages verify
  unchanged. FlashInfer never interprets sidecar content: a consumer with
  domain metadata (a qualification receipt, provenance, licenses) reads the
  verified path through ``SealedPackage.sealed_files``;
* the ``solution.json`` must be a FlashInfer Solution in a Python-family
  language (``python``, ``triton``, ``cutedsl``, ``cutile``, ``tilelang``) — a
  compiled family would need nvcc at serving start, and a
  SOLBench/Kernel-Factory ``spec.languages`` document is refused rather than
  silently translated.

Loading
~~~~~~~

Three environment variables drive loading at ``import flashinfer`` time:

============================================= =========================================================
Variable                                      Meaning
============================================= =========================================================
``FLASHINFER_KERNEL_PACKAGES``                Comma-separated sha256 digests of sealed packages to
                                              load and enable when FlashInfer is imported.
``FLASHINFER_KERNEL_PACKAGE_ROOT``            Root directory holding the digest-named package
                                              directories (default
                                              ``/opt/flashinfer/kernel_packages``).
``FLASHINFER_KERNEL_PLUGINS_DISABLE``         Kill switch: refuse sealed loading with a warning.
============================================= =========================================================

Loading is best-effort and never breaks ``import flashinfer``: a refused
package warns (``SealedPackageError`` is a ``RuntimeError``, never an
``ImportError``, so no engine can mistake the refusal for a missing optional
dependency) and the surviving packages install. Packages may coexist with the
curated path, with a warning that the curated solutions are unverified.

Programmatic use
~~~~~~~~~~~~~~~~

.. code-block:: python

   from flashinfer.trace_apply import load_packages, verify_package, import_entry

   result = load_packages("/opt/flashinfer/kernel_packages", ["<digest>", ...])
   result.packages        # SealedPackage objects (sealed_files, solution, ...)
   result.callables       # {(definition, solution name): imported entry point}
   result.errors          # {digest: reason} for refusals when strict=False
   result.applied         # enable_apply() return value

A verified package's entry point can also be imported on its own:
``import_entry(verify_package(root, digest))`` returns the sealed callable.
Imported callables are registered as *preloaded* (``register_preloaded``), so
the trace loaders never materialize sealed sources into the on-disk solution
cache and every deferred import re-hashes the bytes it compiles. Importing a
verified bundle executes its code: verification is content authentication, not
a sandbox.

Signatures
~~~~~~~~~~

Kernels are **not signed** anywhere in FlashInfer today, and this feature does
not add signing: the digest chain authenticates *content* (what a digest names
is what runs), and the trust anchor is the party that names the digest. The
manifest reserves ``signature`` and ``schema_version`` fields so a future
signature mechanism (e.g. keyless sigstore over the manifest) can be added
without a format break; an unknown ``schema_version`` refuses.
