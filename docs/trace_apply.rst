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

.. _trace_apply_setup:

Setup: untimed weight preparation
---------------------------------

A solution may consume weights in a layout of its own. Trace Apply never
converts a layout per call; instead the solution declares a **setup** step that
runs once per weight set, outside the call path and outside CUDA-graph capture,
and whose result reaches every later call by keyword. The declaration lives on the Solution ``spec`` and applies to the
curated path and to sealed packages alike:

========================== ====================================================================
Field                      Meaning
========================== ====================================================================
``setup_entry_point``      ``"<file>.py::<symbol>"`` of ``setup(context) -> state``. Python
                           family only; when present, ``destination_passing_style`` must be
                           spelled explicitly.
``teardown_entry_point``   ``"<file>.py::<symbol>"`` of ``teardown(state)``. Optional;
                           requires ``setup_entry_point``.
``setup_inputs``           The Definition inputs whose live tensors setup receives and whose
                           identity keys the retained state: distinct identifiers, only with
                           a setup. Empty means one state per process.
``setup_state_param``      The keyword the entry point takes the state under (default
                           ``"state"``; an identifier that is not a setup input).
========================== ====================================================================

The four fields join ``Solution.hash()`` and ``to_dict()`` only when a setup is
declared, so every existing solution, package and build-cache key is unchanged.

The ABI:

.. code-block:: python

   def setup(context):                        # once per weight set, eager, untimed
       weights = context["setup_inputs"]["gemm1_weights"]
       return {"relayout": my_layout(weights)}   # or {"state": ..., "input_buffers": {}}

   def run(hidden_states, gemm1_weights, ..., state, output):
       ...                                    # Definition inputs, the state, DPS outputs

   def teardown(state):                       # only on release_packages() / release()
       ...

``context`` is a plain dict with these keys:
``axes``, ``device`` (``"cuda:0"``), ``rank`` and ``world_size`` (from
``torch.distributed`` when initialized, else ``0`` and ``1``), ``inputs``
(``{name: {"shape", "stride", "dtype", "device"}}``: every Definition input in
declared order when a sealed ``bundle/definition.json`` is known, a scalar
input with ``shape`` None, else the call's tensors),
``input_values`` (the non-tensor inputs), ``setup_inputs`` (the live tensors)
and ``outputs``. ``axes`` and ``outputs`` are filled from the same
``definition.json``, else empty; the curated loader passes no Definition.
A ``{"state", "input_buffers"}`` result envelope is accepted with empty
``input_buffers`` and refused otherwise: serving has no harness copy phase.

A setup-bearing solution loads as a :class:`~flashinfer.trace_apply.SetupBoundCallable`
through ``import_entry``, ``load_packages`` and the curated loader alike. Its
published signature (``inspect.signature``) is the entry point's without the
state parameter, so an ABI check binds the Definition inputs exactly as before.
Per call it keys the ``setup_inputs`` tensors by **identity** — address, shape,
stride, dtype and device — and runs ``setup`` on a miss. A Definition name is a
shape class, not a layer: every MoE layer of a model shares one name and holds
its own weights, so one state per key is one state per weight set, and the view
of one storage an engine passes call after call keys the same. What setup
retains is a second resident copy, allocated at the first eager call of each
weight set and reported, never sized by Trace Apply. Identity cannot see bytes
change behind a stable address, so an engine mode that rewrites weights in
place is the consumer's to refuse.

**Eager before capture.** A weight set first seen inside CUDA-graph capture is
refused with ``SetupStateError`` (a ``RuntimeError``, never an ``ImportError``)
and counted in ``refused_in_capture``: setup allocates and synchronizes, and a
half-prepared call must never be captured. The first capture the callable
observes freezes its key set, so a weight set that appears afterwards is refused
too and a setup input that is really an activation cannot grow state without
bound (``MAX_WEIGHT_SETS`` bounds it under eager execution). Run every weight
set eagerly before capturing, as an engine's profiling and warm-up forwards do.
The check sees the calling thread's capture (CUDA offers no process-wide
query), so a first use on one thread while another thread captures is what
that warm-up rule rules out, not something the callable can detect.
A state does not survive ``fork``: a call from another process is refused. A
setup that calls its own Definition with a new weight set is refused too,
rather than left waiting on the lock it holds.

``release_packages()`` first disarms the routes it releases: every sealed
Solution whose Definition no captured binding serves leaves Trace Apply's
mapping (by identity, so a curated Solution enabled for that Definition since
stays, and the mapping is emptied in place, so a wrapper kept before its first
matching call routes nowhere; one that had already resolved keeps its
callable, which a released binding refuses), then loses its preloaded entry, so no route can fall
through to the on-disk build path at any point; curated Solutions enabled
beside them stay enabled. It then tears down every state those bindings retain
(``teardown`` when declared; every binding is released before a failed
teardown is raised) and forgets them; a released binding refuses every later
call. Callers quiesce their calls first. ``setup_stats()`` returns ``{definition: {"runs", "weight_sets",
"seconds", "retained_bytes", "refused_in_capture"}}`` for the same bindings. A
binding that a CUDA-graph capture has observed is never released, by
``release_packages()`` or by its own ``release()`` (a ``SetupStateError``):
captured graphs replay against the state addresses, so the state is kept for
the life of the process and stays in ``setup_stats()``. A setup-bearing
solution loaded through the curated path is bound the same way but registered
nowhere: ``disable_apply()`` drops it without teardown, and neither call sees
it. Nothing runs at process exit.

``setup_state_param`` is a field rather than a fixed ``state`` because Trace
Apply calls Python-family solutions by keyword, so the wrapper must know the
parameter's name; an existing kernel keeps whatever name it already uses, and
the Solution records it. ``setup_inputs`` sits on the Solution rather than on the
Definition because the Definition at this seam is a trace template with no
per-solution permission document. FlashInfer's own weight-preparation helpers
(``flashinfer.fused_moe.prepare``) may be called from a setup: a solution that
needs the layout a FlashInfer kernel expects produces it there, once.

A complete setup-bearing package is preserved under
``tests/cpu/trace_apply/data/kernel_package/48812d76b7d3073a961dc996d6968e09998bae895ff83f3a175fe7f98cb26280/``:
``bundle/kernel.py`` (``run``), ``bundle/prepare.py`` (``setup`` and
``teardown``), ``bundle/solution.json`` with the four fields,
``bundle/definition.json`` and a ``manifest.json`` in the ``sealed_files``
spelling. The test suite verifies it, binds it and serves
``flashinfer.norm.rmsnorm`` through it on the CPU.

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
  silently translated;
* every declared symbol (the entry point, and the setup and teardown of a
  setup declaration, :ref:`trace_apply_setup`) is validated at verification,
  with the standard library, before any sealed code runs: spelled
  ``<file>.py::<symbol>`` (the form ``BuildSpec`` takes), a sealed Python file
  on a canonical path, not shadowed by a package, and listed in the Solution's
  ``sources``; a sealed ``bundle/definition.json`` must name the solution's
  definition, and sidecars are plain file names beside the manifest; ``teardown_entry_point`` requires
  ``setup_entry_point``, ``destination_passing_style`` is explicit,
  ``setup_inputs`` are distinct identifiers and, when
  ``bundle/definition.json`` is sealed, every setup input is a tensor input of
  it and ``setup_state_param`` is none of its inputs or outputs.

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
``FLASHINFER_KERNEL_PLUGINS_DISABLE``         Kill switch for the import-time loading above:
                                              refuse it with a warning. A program calling
                                              ``load_packages()`` itself is not affected.
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
``import_entry(verify_package(root, digest))`` returns the sealed callable (for
a setup-bearing package, the entry point bound to its setup and teardown, both
imported from the same sealed namespace; see :ref:`trace_apply_setup`).
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
