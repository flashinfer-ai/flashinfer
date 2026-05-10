.. _apply:

Apply - Runtime Kernel Substitution
===================================

``flashinfer.apply`` lets you replace selected FlashInfer API calls with a
user-provided Python solution at runtime.  It is keyed by the same definition
names produced by :ref:`fi_trace <fi_trace>`, so a replacement is selected by
the concrete operation schema, not by the Python function name alone.

This is useful when you want to test or deploy a custom kernel implementation
for a specific FlashInfer operation shape while preserving the original
FlashInfer API surface.

Quick Start
-----------

Enable solutions with ``flashinfer.enable_apply()``.  The registration key is
the ``fi_trace`` definition name, for example ``"rmsnorm_h4096"``.

.. code-block:: python

    import torch
    import flashinfer

    hidden_states = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float16)

    def rmsnorm_solution(hidden_states, weight):
        # Replace this body with your custom implementation.
        return hidden_states * weight

    flashinfer.enable_apply({"rmsnorm_h4096": rmsnorm_solution})
    out = flashinfer.rmsnorm(hidden_states, weight)
    flashinfer.disable_apply()

When the installed apply wrapper has no matching solution, or when the solution
fails, FlashInfer falls back to the original API implementation and emits a
warning for the failure case.

Finding the Definition Name
---------------------------

Every traced API exposes ``.fi_trace()``.  Call it with representative inputs to
inspect the definition name that should be used as the apply key.

.. code-block:: python

    definition = flashinfer.rmsnorm.fi_trace(hidden_states, weight)
    print(definition["name"])  # rmsnorm_h4096

``fi_trace`` only uses tensor metadata such as shape and dtype.  It does not
dump tensor values.

Public API
----------

FlashInfer exposes top-level enable/disable APIs:

.. code-block:: python

    flashinfer.enable_apply({"rmsnorm_h4096": rmsnorm_solution})
    flashinfer.enable_apply(apply_config)
    flashinfer.disable_apply()

``enable_apply(config)``
    Installs apply wrappers for imported FlashInfer APIs.  ``config`` can be an
    ``ApplyConfig`` or a mapping from definition name to Python callable or
    ``flashinfer.trace.Solution`` object.

``disable_apply()``
    Restores patched APIs and disables apply.

Calling ``enable_apply()`` again replaces the active apply routing.  A later
``disable_apply()`` restores all patched APIs to their original implementations.

Callable Solutions
------------------

Callable solutions are invoked with positional arguments only.  The argument
order follows the corresponding ``TraceTemplate.inputs`` order.

For ``rmsnorm``, the solution receives ``hidden_states`` and ``weight``:

.. code-block:: python

    def rmsnorm_solution(hidden_states, weight):
        return hidden_states * weight

    flashinfer.enable_apply({"rmsnorm_h4096": rmsnorm_solution})
    out = flashinfer.rmsnorm(hidden_states, weight)
    flashinfer.disable_apply()

If the original FlashInfer call provides an ``out=`` buffer, the returned tensor
is copied into that buffer and the original API return convention is preserved.

flashinfer-bench Solution Objects
---------------------------------

``enable_apply`` can also consume a minimal Python Solution object compatible
with `flashinfer-bench <https://github.com/flashinfer-ai/flashinfer-bench>`_.
The current installer supports the simple no-build subset:

* ``spec.language == "python"``
* ``spec.entry_point == "<file.py>::<function>"``
* ``sources`` contains the Python source for that entry file
* calls are positional only

.. code-block:: python

    from flashinfer.trace import BuildSpec, Solution, SourceFile

    solution = Solution(
        name="rmsnorm_h4096_python",
        definition="rmsnorm_h4096",
        author="example",
        spec=BuildSpec(
            language="python",
            target_hardware=["CUDA"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "def run(hidden_states, weight):\n"
                    "    return hidden_states * weight\n"
                ),
            )
        ],
    )

    flashinfer.enable_apply({"rmsnorm_h4096": solution})
    out = flashinfer.rmsnorm(hidden_states, weight)
    flashinfer.disable_apply()

Solution source is executed in-process and should be treated as trusted code.

Serializable Apply Config
-------------------------

Use ``ApplyConfig`` when you want a serializable definition-to-solution routing
table.  The config only specifies which solution to use for each definition; it
does not contain benchmark-selection policies or file I/O.

.. code-block:: python

    config = flashinfer.ApplyConfig.from_dict(config_dict)
    flashinfer.enable_apply(config)
    out = flashinfer.rmsnorm(hidden_states, weight)
    flashinfer.disable_apply()

Example config dictionary:

.. code-block:: json

    {
      "solutions": {
        "rmsnorm_h4096": {
          "name": "rmsnorm_h4096_python",
          "definition": "rmsnorm_h4096",
          "author": "example",
          "spec": {
            "language": "python",
            "target_hardware": ["CUDA"],
            "entry_point": "main.py::run",
            "destination_passing_style": false
          },
          "sources": [
            {
              "path": "main.py",
              "content": "def run(hidden_states, weight):\n    return hidden_states * weight\n"
            }
          ]
        }
      }
    }

The ``definition`` field inside each solution is required and must match the
surrounding ``solutions`` key.

Environment Variables
---------------------

Set ``FLASHINFER_APPLY=1`` to enable apply automatically when ``flashinfer`` is
imported.  ``FLASHINFER_APPLY_CONFIG`` must point to a config JSON file.

.. code-block:: bash

    export FLASHINFER_APPLY=1
    export FLASHINFER_APPLY_CONFIG=/path/to/apply.json
    python run_engine.py

Invalid or missing env config emits a warning and leaves apply disabled.

Destination-Passing Style
-------------------------

For solutions that mutate output buffers directly, set
``destination_passing_style`` to ``True``.  FlashInfer appends output buffers
after the input arguments in ``TraceTemplate.outputs`` order.

.. code-block:: python

    solution = Solution(
        name="rmsnorm_h4096_dps",
        definition="rmsnorm_h4096",
        author="example",
        spec=BuildSpec(
            language="python",
            target_hardware=["CUDA"],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "def run(hidden_states, weight, output):\n"
                    "    output.copy_(hidden_states * weight)\n"
                ),
            )
        ],
    )

    out = torch.empty_like(hidden_states)
    flashinfer.enable_apply({"rmsnorm_h4096": solution})
    result = flashinfer.rmsnorm(hidden_states, weight, out=out)
    flashinfer.disable_apply()

    assert result is out

Limitations
-----------

The first apply implementation intentionally supports directly replaceable
kernel functions only.  Stateful wrapper replacement, automatic solution
selection from benchmark results, external build systems, and environment-file
configuration are out of scope for this runtime path.
