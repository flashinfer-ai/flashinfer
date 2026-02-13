.. _autotuning:

Autotuning
==========

FlashInfer includes an autotuner that selects the best kernel implementation
(runner and tactic) for each operation and input shape by profiling at runtime.

What Is Autotuning?
-------------------

Many FlashInfer operations -- GEMM, MoE, and others -- support multiple backend
implementations (runners).  Each runner may also expose several low-level
tactics (e.g. tile sizes, pipeline stages).  The best choice depends on the
hardware, data types, and input shapes of your workload.

Without autotuning, FlashInfer picks a default (fallback) tactic.  With
autotuning enabled, the autotuner profiles every candidate for a given shape and
automatically selects the fastest one.

Enabling Autotuning
-------------------

Wrap the portion of your code that you want to tune inside the
``flashinfer.autotune`` context manager:

.. code-block:: python

    import flashinfer

    with flashinfer.autotune():
        # All FlashInfer ops executed here will be profiled.
        output = flashinfer.gemm.bmm_fp8(A, B, A_scale, B_scale, dtype=out_dtype)

The first time an operation runs inside the context, the autotuner benchmarks
all available runners and tactics for that ``(operation, backend, shape)``
combination.  Subsequent calls with the same shape reuse the cached result
without re-profiling.

You can also pass ``tune_mode=True`` explicitly (the default):

.. code-block:: python

    with flashinfer.autotune(True):
        output = flashinfer.gemm.bmm_fp8(A, B, A_scale, B_scale, dtype=out_dtype)

When ``tune_mode=False``, the context manager enters a no-profiling mode that
only uses previously cached or loaded configs:

.. code-block:: python

    with flashinfer.autotune(False):
        # No profiling -- uses default/fallback tactic if nothing is cached.
        model(inputs)

Autotuning in the Benchmark Harness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FlashInfer benchmark harness supports autotuning via the ``--autotune``
flag:

.. code-block:: bash

    python3 flashinfer_benchmark.py \
        --routine mm_fp4 --m 4 --n 7168 --k 4608 \
        --out_dtype bfloat16 --backends cudnn cutlass trtllm \
        --use_128x4_sf_layout --use_nvfp4 \
        --autotune

Config Caching
--------------

By default, autotuning results live only in memory and are lost when the
process exits.  The ``cache`` parameter on ``flashinfer.autotune`` lets you
persist results to a JSON file and load them back in future runs, avoiding
repeated profiling.

Saving Tuned Configs
^^^^^^^^^^^^^^^^^^^^

Pass a file path to ``cache`` when tuning.  On exit, all profiled configs are
written to that file:

.. code-block:: python

    import flashinfer

    with flashinfer.autotune(True, cache="my_configs.json"):
        model(inputs)
    # On exit, tuned configs are saved to my_configs.json.

Loading Cached Configs
^^^^^^^^^^^^^^^^^^^^^^

To reuse previously tuned configs without profiling, pass ``tune_mode=False``
with the same cache path:

.. code-block:: python

    import flashinfer

    with flashinfer.autotune(False, cache="my_configs.json"):
        # Configs are loaded on entry.  No profiling occurs.
        model(inputs)

Incremental Tuning
^^^^^^^^^^^^^^^^^^

Cache files support incremental updates.  When ``autotune(True, cache=path)``
exits:

1. Previously loaded configs are used as a base.
2. Newly profiled configs are overlaid (new results take priority for duplicate keys).
3. The merged result is written back to the same file.

This means you can run multiple tuning sessions -- for example different batch
sizes or sequence lengths -- and accumulate all configs in a single file:

.. code-block:: python

    # Session 1: tune with batch_size=1
    with flashinfer.autotune(True, cache="configs.json"):
        run_model(batch_size=1)

    # Session 2: tune with batch_size=32 (configs.json now has both)
    with flashinfer.autotune(True, cache="configs.json"):
        run_model(batch_size=32)

Cache Hit Behavior During Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``autotune(True, cache=path)`` is active and a matching config is found in
the cache file, the autotuner uses it directly **without re-profiling**.  This
means:

- Shapes that were already tuned are skipped, saving time.
- Only truly new shapes trigger profiling.
- A log message is printed once per ``(operation, runner)`` pair when a cache
  hit is detected.

Caching in the Benchmark Harness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The benchmark harness supports config caching via the ``--autotune_cache`` flag.

**Tune and save during benchmarking:**

.. code-block:: bash

    python3 flashinfer_benchmark.py \
        --routine mm_fp4 --m 4 --n 7168 --k 4608 \
        --out_dtype bfloat16 --backends cudnn cutlass trtllm \
        --use_128x4_sf_layout --use_nvfp4 \
        --autotune --autotune_cache my_configs.json

**Run with cached configs (no profiling):**

.. code-block:: bash

    python3 flashinfer_benchmark.py \
        --routine mm_fp4 --m 4 --n 7168 --k 4608 \
        --out_dtype bfloat16 --backends cudnn cutlass trtllm \
        --use_128x4_sf_layout --use_nvfp4 \
        --autotune_cache my_configs.json

Cache File Format
^^^^^^^^^^^^^^^^^

The cache file is a plain JSON dictionary.  Each key is a string representation
of ``(custom_op, runner_class_name, optimization_profile)`` and each value is
``[runner_class_name, tactic]``:

.. code-block:: json

    {
      "('fp4_gemm', 'CudnnFp4GemmRunner', ...)": ["CudnnFp4GemmRunner", 3],
      "('fp4_gemm', 'CutlassFp4GemmRunner', ...)": ["CutlassFp4GemmRunner", 1]
    }

The file is human-readable and portable.  Because it stores runner class names
(not positional indices), it is robust to changes in runner ordering across
FlashInfer versions.

API Reference
-------------

``flashinfer.autotune``
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    flashinfer.autotune(tune_mode: bool = True, cache: str | None = None)

Context manager for autotuning with optional file-based caching.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``tune_mode``
     - ``bool``
     - If ``True``, profile uncovered shapes during execution.
       If ``False``, only use cached/loaded configs (no profiling).
   * - ``cache``
     - ``str | None``
     - Optional path to a JSON config file.
       On entry, configs are loaded from this file if it exists.
       On exit, configs are saved back to this file when ``tune_mode=True``.

**Behavior matrix:**

.. list-table::
   :header-rows: 1
   :widths: 15 10 20 20 35

   * - ``tune_mode``
     - ``cache``
     - Load on entry?
     - Save on exit?
     - Use case
   * - ``True``
     - path
     - Yes (if file exists)
     - Yes
     - Tune and persist results
   * - ``True``
     - ``None``
     - No
     - No
     - Tune in-memory only (results lost on exit)
   * - ``False``
     - path
     - Yes (if file exists)
     - No
     - Inference with pre-tuned configs
   * - ``False``
     - ``None``
     - No
     - No
     - No-op (default behavior)

``AutoTuner.save_configs``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    AutoTuner.get().save_configs(path: str)

Manually save the in-memory profiling cache to a JSON file.
This is called automatically on exit from ``autotune(True, cache=path)``;
direct calls are only needed for advanced use cases.

Previously loaded configs (from ``load_configs``) are merged into the output,
with in-memory profiling results taking priority for overlapping keys.

``AutoTuner.load_configs``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    AutoTuner.get().load_configs(path: str)

Manually load configs from a JSON file into the internal lookup table.
This is called automatically on entry to ``autotune(cache=path)``;
direct calls are only needed for advanced use cases.

``AutoTuner.clear_cache``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    AutoTuner.get().clear_cache()

Clear the in-memory profiling cache **and** any user-loaded file configs.

Multi-Thread / Multi-Process Considerations
--------------------------------------------

.. warning::
    Autotuner behavior in multi-threaded and multi-process environments is
    **undefined** as of now.  The ``AutoTuner`` singleton is not thread-safe,
    and concurrent writes to the same cache file from multiple processes are
    not coordinated.

If you are tuning with multiple processes (e.g. multi-GPU with ``torchrun``),
use separate output files per rank and merge them afterwards:

.. code-block:: python

    import json

    merged = {}
    for path in ["configs_rank0.json", "configs_rank1.json"]:
        with open(path) as f:
            merged.update(json.load(f))

    with open("configs_merged.json", "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
