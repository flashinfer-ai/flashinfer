.. _logging:

Logging
=======

FlashInfer provides a logging feature to help debug issues and reproduce crashes. This document describes all available logging levels and their features.

Quick Start
-----------

Enable logging using two environment variables:

.. code-block:: bash

    # Set logging level (0-5)
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
     - Flight Recorder
     - Level 5 + dumps all input/output tensors to ``.pt`` files
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
     - Maximum size of dump directory in GB (Level 10 only)
   * - ``FLASHINFER_DUMP_MAX_COUNT``
     - int
     - 1000
     - Maximum number of API calls to dump

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

.. code-block:: bash

    # Enable Flight Recorder (Metadata + Tensors)
    export FLASHINFER_LOGLEVEL=10
    export FLASHINFER_DUMP_DIR=./my_dumps

    # Run your application
    python my_app.py

    # Replay recorded calls
    export FLASHINFER_LOGLEVEL=1 # 0 for pass/fail.
    flashinfer replay --dir ./my_dumps
    # or
    python -m flashinfer replay --dir ./my_dumps
