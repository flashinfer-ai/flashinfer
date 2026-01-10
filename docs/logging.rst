.. _logging:

Logging
=======

FlashInfer provides a logging feature to help debug issues and reproduce crashes. This document describes all available logging levels and their features.

Quick Start
-----------

Enable logging using two environment variables:

.. code-block:: bash

    # Set logging level (0-10)
    export FLASHINFER_LOGLEVEL=3

    # Set log destination (default is stdout)
    export FLASHINFER_LOGDEST=stdout  # or stderr, or a file path like "flashinfer.log"

Logging Levels
--------------

.. list-table::
   :header-rows: 1
   :widths: 10 20 35 25

   * - Level
     - Name
     - Features
     - Use Case
   * - **0**
     - Disabled (Default)
     - No logging (zero overhead)
     - Production
   * - **1**
     - Function Names
     - Function names only
     - Basic tracing
   * - **3**
     - Inputs/Outputs
     - Function names + arguments + outputs with metadata
     - Standard debugging
   * - **5**
     - Statistics
     - Level 3 + tensor statistics (min, max, mean, NaN/Inf counts)
     - Numerical analysis
   * - **10**
     - Flight Recorder - Full Input/Output Dumps
     - Level 5 + dumps all input/output tensors to ``.pt`` (or ``.safetensors``) files
     - Full Reproducibility / Debugging

Environment Variables
---------------------

Main Configuration
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``FLASHINFER_LOGLEVEL``
     - int
     - 0
     - Logging level (0, 1, 3, 5, 10)
   * - ``FLASHINFER_LOGDEST``
     - str
     - ``stdout``
     - Log destination: ``stdout``, ``stderr``, or file path

Dump Configuration (Level 10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When FLASHINFER_LOGLEVEL is set to 10, the following environment variables can be used to configure the dump behavior:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``FLASHINFER_DUMP_DIR``
     - str
     - ``flashinfer_dumps``
     - Directory to save dump files
   * - ``FLASHINFER_DUMP_MAX_SIZE_GB``
     - float
     - 20
     - Maximum size of dump directory in GB
   * - ``FLASHINFER_DUMP_MAX_COUNT``
     - int
     - 1000
     - Maximum number of API calls to dump
   * - ``FLASHINFER_DUMP_INCLUDE``
     - str
     - (empty)
     - Comma-separated patterns to include (fnmatch-style)
   * - ``FLASHINFER_DUMP_EXCLUDE``
     - str
     - (empty)
     - Comma-separated patterns to exclude (fnmatch-style)
   * - ``FLASHINFER_DUMP_SAFETENSORS``
     - int
     - 0
     - Set to 1 to use safetensors format (no pickle, but loses stride info)

SafeTensors Format (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, tensors are saved using ``torch.save()`` which preserves tensor stride and contiguity information.
For faster, pickle-free serialization, you can enable safetensors format:

.. code-block:: bash

    export FLASHINFER_DUMP_SAFETENSORS=1

.. warning::
    SafeTensors does NOT preserve tensor strides or non-contiguity.
    All tensors are saved as contiguous. Use the default ``torch.save`` format
    if stride preservation is important for your debugging.

**Comparison**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - torch.save (default)
     - safetensors
   * - Speed
     - Standard
     - Faster
   * - Safety
     - Uses pickle
     - No pickle (safer)
   * - Stride preservation
     - ✅ Yes
     - ❌ No (contiguous only)
   * - File extension
     - ``.pt``
     - ``.safetensors``
   * - Dependency
     - `torch``
     - Requires ``pip install safetensors``

**Replay is format-agnostic**: The replay command automatically detects the format based on file extension.

Dump Filtering (Include/Exclude)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``FLASHINFER_DUMP_INCLUDE`` and ``FLASHINFER_DUMP_EXCLUDE`` to control which API calls are dumped.
This is especially useful when running end-to-end inference with many API calls but you only care about specific ones.

**Pattern Syntax** (fnmatch-style):

- ``*`` matches any number of characters
- ``?`` matches a single character
- Matching is case-sensitive
- For class methods, the function name is formatted as ``ClassName.method_name``

**Filter Logic**:

1. If ``FLASHINFER_DUMP_INCLUDE`` is set, only APIs matching at least one pattern are dumped
2. If ``FLASHINFER_DUMP_EXCLUDE`` is set, APIs matching any pattern are skipped
3. Both can be combined: include filter is applied first, then exclude filter

**Examples**:

.. code-block:: bash

    # Only dump decode-related APIs
    export FLASHINFER_DUMP_INCLUDE="*decode*"

    # Dump everything except __init__ and plan methods
    export FLASHINFER_DUMP_EXCLUDE="*.__init__,*.plan"

    # Only dump run() methods from wrapper classes
    export FLASHINFER_DUMP_INCLUDE="*Wrapper.run"

    # Dump all single_* APIs except prefill
    export FLASHINFER_DUMP_INCLUDE="single_*"
    export FLASHINFER_DUMP_EXCLUDE="*prefill*"

    # Only dump a specific wrapper's run method
    export FLASHINFER_DUMP_INCLUDE="BatchDecodeWithPagedKVCacheWrapper.run"

    # Dump FP8 APIs but not quantization steps
    export FLASHINFER_DUMP_INCLUDE="*fp8*,*FP8*"
    export FLASHINFER_DUMP_EXCLUDE="*quantize*"

**Common Patterns**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Pattern
     - Matches
   * - ``*decode*``
     - ``single_decode_with_kv_cache``, ``BatchDecodeWithPagedKVCacheWrapper.run``
   * - ``*Wrapper.run``
     - ``BatchDecodeWithPagedKVCacheWrapper.run``, ``BatchPrefillWithPagedKVCacheWrapper.run``
   * - ``*.__init__``
     - All wrapper ``__init__`` methods
   * - ``*.plan``
     - All wrapper ``plan`` methods
   * - ``mm_fp8``
     - Exact match for ``mm_fp8`` function
   * - ``single_*``
     - ``single_decode_with_kv_cache``, ``single_prefill_with_kv_cache``

Process ID Substitution
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``%i`` in file paths for automatic process ID substitution (useful for multi-GPU training):

.. code-block:: bash

    export FLASHINFER_LOGDEST="flashinfer_log_%i.txt"  # → flashinfer_log_12345.txt


Miscellaneous Notes and Examples
---------------------------------

CUDA Graph Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^

Level 5 statistics are **automatically skipped during CUDA graph capture** to avoid synchronization issues.

.. code-block:: python

    # This works correctly - no synchronization errors
    with torch.cuda.graph(cuda_graph):
        result = mm_fp4(a, b, scales, ...)  # Level 5 logging active
        # Statistics automatically skipped during capture

Output shows: ``[statistics skipped: CUDA graph capture in progress]``

Process IDs for Multi-GPU Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Use %i for process ID substitution
    export FLASHINFER_LOGLEVEL=3
    export FLASHINFER_LOGDEST="logs/flashinfer_api_%i.log"

    torchrun --nproc_per_node=8 awesome_script_that_uses_FlashInfer.py

    # Creates separate logs:
    # logs/flashinfer_api_12345.log (rank 0)
    # logs/flashinfer_api_12346.log (rank 1)
    # ...

Level 0 has zero overhead
^^^^^^^^^^^^^^^^^^^^^^^^^^^

At Level 0, the decorator returns the original function unchanged. No wrapper, no checks, no overhead.

Flight Recorder & Replay
------------------------

FlashInfer includes a "Flight Recorder" mode (Level 10) that captures inputs/outputs for reproducibility.

Dump Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^

When Level 10 logging is enabled, FlashInfer creates the following structure:

.. code-block:: text

    FLASHINFER_DUMP_DIR/
    ├── session.jsonl                    # Central log: one line per event (quick scanning)
    ├── 20250108_143216_802_pid12345_mm_fp8_call0001/
    │   ├── metadata.jsonl               # Per-dump metadata (JSONL format)
    │   ├── inputs.pt                    # Input tensors (or .safetensors if enabled)
    │   └── outputs.pt                   # Output tensors (or .safetensors if enabled)
    ├── 20250108_143216_868_pid12345_single_decode_call0001/
    │   ├── metadata.jsonl
    │   ├── inputs.pt                    # (or .safetensors)
    │   └── outputs.pt                   # (or .safetensors)
    └── ...

**JSONL Format**: Both ``session.jsonl`` and ``metadata.jsonl`` use `JSON Lines <https://jsonlines.org/>`_ format
(one JSON object per line). This enables:

- **Crash-safe logging**: Each API call appends two lines (inputs_saved, then completed)
- **Quick scanning**: Use ``session.jsonl`` to browse all recorded calls without reading subdirectories
- **Streaming reads**: Process records line-by-line for large sessions

**Per-dump metadata.jsonl**:

- Line 1: Written **before** execution (``execution_status: "inputs_saved"``)
- Line 2: Appended **after** successful execution (``execution_status: "completed"``)

If a crash occurs, only line 1 will be present, preserving the inputs for debugging.

**Central session.jsonl**:

One-stop log of all API calls. Use standard tools to filter and analyze:

.. code-block:: bash

    # Enable Flight Recorder (Metadata + Tensors)
    export FLASHINFER_LOGLEVEL=10
    export FLASHINFER_DUMP_DIR=./my_dumps

    # Run your application
    python3 benchmarks/flashinfer_benchmark.py --routine mm_fp4 --m 4 --n 1024 --k 7168 --out_dtype bfloat16 --backends cudnn --use_128x4_sf_layout --use_nvfp4 --refcheck -vv --generate_repro_command --use_cupti --no_cuda_graph --num_iters 5
    ... output redacted ...

    # Replay recorded calls
    export FLASHINFER_LOGLEVEL=0 # 1 for more detailed replay results.
    flashinfer replay --dir ./my_dumps
    # or
    python -m flashinfer replay --dir ./my_dumps

    [1] nvfp4_quantize (20251204_143216_802_pid12345_nvfp4_quantize_call0001): ✅ Passed
    [2] fp4_quantize (20251204_143216_868_pid12345_fp4_quantize_call0001): ✅ Passed
    [3] nvfp4_quantize (20251204_143216_949_pid12345_nvfp4_quantize_call0002): ✅ Passed
    [4] fp4_quantize (20251204_143217_003_pid12345_fp4_quantize_call0002): ✅ Passed
    [5] mm_fp4 (20251204_143217_178_pid12345_mm_fp4_call0001): ✅ Passed
    [6] mm_fp4 (20251204_143217_346_pid12345_mm_fp4_call0002): ✅ Passed
    [7] mm_fp4 (20251204_143217_427_pid12345_mm_fp4_call0003): ✅ Passed
    [8] mm_fp4 (20251204_143217_475_pid12345_mm_fp4_call0004): ✅ Passed
    [9] mm_fp4 (20251204_143217_510_pid12345_mm_fp4_call0005): ✅ Passed
    [10] mm_fp4 (20251204_143217_551_pid12345_mm_fp4_call0006): ✅ Passed
    [11] mm_fp4 (20251204_143217_591_pid12345_mm_fp4_call0007): ✅ Passed
    [12] mm_fp4 (20251204_143217_631_pid12345_mm_fp4_call0008): ✅ Passed
    [13] mm_fp4 (20251204_143217_672_pid12345_mm_fp4_call0009): ✅ Passed
    [14] mm_fp4 (20251204_143217_708_pid12345_mm_fp4_call0010): ✅ Passed
    [15] mm_fp4 (20251204_143217_769_pid12345_mm_fp4_call0011): ✅ Passed
    [16] mm_fp4 (20251204_143217_812_pid12345_mm_fp4_call0012): ✅ Passed
    [17] mm_fp4 (20251204_143217_852_pid12345_mm_fp4_call0013): ✅ Passed
    [18] mm_fp4 (20251204_143217_904_pid12345_mm_fp4_call0014): ✅ Passed
    [19] mm_fp4 (20251204_143218_153_pid12345_mm_fp4_call0015): ✅ Passed
    [20] mm_fp4 (20251204_143218_390_pid12345_mm_fp4_call0016): ✅ Passed
    [21] mm_fp4 (20251204_143218_627_pid12345_mm_fp4_call0017): ✅ Passed
    [22] mm_fp4 (20251204_143218_862_pid12345_mm_fp4_call0018): ✅ Passed

    Summary: 22 passed, 0 failed/mismatch

Python-Based Replay Examples
----------------------------

The following examples demonstrate how to use Level 10 logging to dump and replay API calls programmatically using Python.

Example 1: ``bmm_fp8`` - Simple Function Call
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Producer Script** (``bmm_fp8_producer.py``):

This script initializes tensors, calls ``bmm_fp8``, and dumps the inputs/outputs to disk.

.. code-block:: python

    """
    Producer script: Run bmm_fp8 with Level 10 logging to dump tensors.

    Usage:
        FLASHINFER_LOGLEVEL=10 FLASHINFER_DUMP_DIR=./bmm_fp8_dumps python bmm_fp8_producer.py
    """
    import torch
    from flashinfer import bmm_fp8

    def to_float8(x, dtype=torch.float8_e4m3fn):
        """Convert tensor to FP8 with per-tensor scaling."""
        finfo = torch.finfo(dtype)
        min_val, max_val = x.aminmax()
        amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
        scale = finfo.max / amax
        x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
        return x_scl_sat.to(dtype), scale.float().reciprocal()

    # Parameters
    b, m, n, k = 4, 64, 128, 256
    input_dtype = torch.float8_e4m3fn
    mat2_dtype = torch.float8_e4m3fn
    res_dtype = torch.bfloat16

    # Create input tensors
    input_bf16 = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input_bf16, dtype=input_dtype)

    # mat2: row major -> column major (transposed)
    mat2_bf16 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2_bf16, dtype=mat2_dtype)

    # Pre-allocate output
    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    # Call bmm_fp8 - this will be logged/dumped at Level 10
    bmm_fp8(input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype, res, backend="cublas")

    # Print a small portion of the output for verification
    print("Output shape:", res.shape)
    print("Output[0, :3, :3]:")
    print(res[0, :3, :3])

**Reproducer Script** (``bmm_fp8_reproducer.py``):

This script loads the dumped tensors and replays the ``bmm_fp8`` call.

.. code-block:: python

    """
    Reproducer script: Load dumped tensors and replay bmm_fp8.

    Usage:
        python bmm_fp8_reproducer.py
    """
    import torch
    from pathlib import Path
    from flashinfer import bmm_fp8
    from flashinfer.api_logging import replay_from_dump

    DUMP_DIR = "./bmm_fp8_dumps"

    # Find the bmm_fp8 dump directory (should be the only one or the latest)
    dump_path = Path(DUMP_DIR)
    bmm_dumps = sorted([d for d in dump_path.iterdir() if d.is_dir() and "bmm_fp8" in d.name])
    latest_dump = bmm_dumps[-1]  # Use the latest dump
    print(f"Loading dump from: {latest_dump}")

    # Use replay_from_dump to load inputs and optionally execute
    result = replay_from_dump(
        str(latest_dump),
        compare_outputs=True,  # Load expected outputs for comparison
        device="cuda",
        run=False,  # We'll call the function manually below
    )

    # Extract the loaded arguments - args contains all positional args including the output tensor
    args = result["args"]
    kwargs = result["kwargs"]
    expected_tensors = result.get("expected_tensors", {})

    # Replay the call - args already contains (input, mat2, input_inv_s, mat2_inv_s, dtype, out)
    res = bmm_fp8(*args, **kwargs)

    # Print the same portion for comparison
    print("Replayed output shape:", res.shape)
    print("Replayed output[0, :3, :3]:")
    print(res[0, :3, :3])

    # Compare with expected output if available
    if "result" in expected_tensors:
        expected = expected_tensors["result"]
        if torch.allclose(res, expected, rtol=1e-3, atol=1e-3):
            print("\n✅ Output matches expected result!")
        else:
            diff = (res - expected).abs().max().item()
            print(f"\n❌ Output mismatch! Max diff: {diff}")

Example 2: ``BatchDecodeWithPagedKVCacheWrapper`` - Stateful Wrapper Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Producer Script** (``batch_decode_producer.py``):

This script demonstrates logging with a stateful wrapper class that requires ``__init__``, ``plan``, and ``run`` calls.

.. code-block:: python

    """
    Producer script: Run BatchDecodeWithPagedKVCacheWrapper with Level 10 logging.

    Usage:
        FLASHINFER_LOGLEVEL=10 FLASHINFER_DUMP_DIR=./batch_decode_dumps python batch_decode_producer.py
    """
    import torch
    import flashinfer

    # Parameters
    batch_size = 4
    kv_len = 512
    page_size = 16
    num_kv_heads = 4
    num_qo_heads = 32
    head_dim = 128
    kv_layout = "NHD"

    # Create query tensor
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda", dtype=torch.float16)

    # Create paged KV cache
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]  # NHD layout
    kv_data = torch.randn(*kv_shape, device="cuda", dtype=torch.float16)

    # Create index tensors
    kv_indptr = torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages, device="cuda", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda"
    )

    # Create workspace and wrapper - __init__ will be logged
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)

    # Plan - will be logged
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )

    # Run - will be logged
    output, lse = wrapper.run(q, kv_data, return_lse=True)

    # Print a small portion of the output
    print("Output shape:", output.shape)
    print("Output[0, :3, :3]:")
    print(output[0, :3, :3])
    print("\nLSE shape:", lse.shape)
    print("LSE[0, :5]:", lse[0, :5])

**Reproducer Script** (``batch_decode_reproducer.py``):

This script demonstrates replaying a sequence of stateful API calls.

.. code-block:: python

    """
    Reproducer script: Replay BatchDecodeWithPagedKVCacheWrapper calls.

    Usage:
        python batch_decode_reproducer.py
    """
    import torch
    from pathlib import Path
    from flashinfer.api_logging import replay_sequence

    DUMP_DIR = "./batch_decode_dumps"

    # replay_sequence handles stateful objects automatically via object_registry
    # It will:
    # 1. Replay __init__ to create the wrapper instance
    # 2. Replay plan() on the same instance
    # 3. Replay run() on the same instance and compare outputs
    results = replay_sequence(DUMP_DIR, device="cuda")

    # Print summary
    passed = 0
    failed = 0
    for i, res in enumerate(results):
        func_name = res.get("metadata", {}).get("function_name", "unknown")
        dump_dir = Path(res.get("dump_dir", "")).name

        if "error" in res:
            print(f"[{i+1}] {func_name} ({dump_dir}): ❌ Error: {res['error']}")
            failed += 1
        elif res.get("comparison_match", True):
            print(f"[{i+1}] {func_name} ({dump_dir}): ✅ Passed")
            passed += 1
        else:
            print(f"[{i+1}] {func_name} ({dump_dir}): ❌ Mismatch")
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")

    # For manual inspection, you can also access individual results
    # Find the 'run' call result (usually the last non-init, non-plan call)
    for res in results:
        func_name = res.get("metadata", {}).get("function_name", "")
        if "run" in func_name and "execution_result" in res:
            output = res["execution_result"]
            if isinstance(output, tuple):
                output_tensor, lse = output
                print("\nReplayed output[0, :3, :3]:")
                print(output_tensor[0, :3, :3])
                print("Replayed LSE[0, :5]:", lse[0, :5])
            break

Manual Replay Without ``replay_from_dump``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more control, you can manually load the dumped tensors:

.. note::
    This example assumes the default ``torch.save`` format (``.pt`` files).
    If dumps were created with ``FLASHINFER_DUMP_SAFETENSORS=1``, use
    ``safetensors.torch.load_file()`` instead of ``torch.load()``.

.. code-block:: python

    """
    Manual replay: Load tensors directly from .pt files.
    """
    import json
    import torch
    from pathlib import Path
    from flashinfer import bmm_fp8

    # Path is an example, replace with the actual path.
    dump_dir = Path("./bmm_fp8_dumps/20250108_103217_012_pid12345_bmm_fp8_call0001")

    # Load metadata from JSONL (read last line for most complete state)
    with open(dump_dir / "metadata.jsonl") as f:
        lines = [line.strip() for line in f if line.strip()]
        metadata = json.loads(lines[-1])  # Last line has completed state

    print(f"Function: {metadata['function_name']}")
    print(f"Module: {metadata['module']}")
    print(f"Status: {metadata['execution_status']}")
    print(f"Input tensors: {metadata['tensor_info']['input_tensor_keys']}")

    # Load input tensors
    inputs = torch.load(dump_dir / "inputs.pt", map_location="cuda")

    # Load expected outputs (if execution completed successfully)
    outputs_path = dump_dir / "outputs.pt"
    if outputs_path.exists():
        expected = torch.load(outputs_path, map_location="cuda")
        print(f"Output tensors: {list(expected.keys())}")

    # Tensors are ready to use - reconstruct the call as needed
    for key, tensor in inputs.items():
        print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

Scanning Session History
^^^^^^^^^^^^^^^^^^^^^^^^

Use the central ``session.jsonl`` to quickly scan all recorded API calls:

.. code-block:: python

    """
    Scan session.jsonl for quick overview of recorded calls.
    """
    import json
    from pathlib import Path
    from collections import Counter

    dump_root = Path("./my_dumps")
    session_file = dump_root / "session.jsonl"

    # Read all records
    records = []
    with open(session_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter to completed calls only
    completed = [r for r in records if r["execution_status"] == "completed"]
    print(f"Total completed calls: {len(completed)}")

    # Count by function name
    func_counts = Counter(r["function_name"] for r in completed)
    print("\nCalls by function:")
    for func, count in func_counts.most_common():
        print(f"  {func}: {count}")

    # Find calls that didn't complete (potential crashes)
    inputs_only = [r for r in records if r["execution_status"] == "inputs_saved"]
    # Group by dump_dir to find incomplete calls
    completed_dirs = {r["dump_dir"] for r in completed}
    incomplete = [r for r in inputs_only if r["dump_dir"] not in completed_dirs]
    if incomplete:
        print(f"\n⚠️  Found {len(incomplete)} incomplete calls (potential crashes):")
        for r in incomplete:
            print(f"  - {r['function_name']} at {r['dump_dir']}")
