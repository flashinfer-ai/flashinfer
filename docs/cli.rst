.. _cli:

Command Line Interface
======================

FlashInfer provides a command-line interface for managing modules, artifacts, and development tools.

Quick Reference
---------------

View all available commands:

.. code-block:: bash

   flashinfer --help

Export Compile Commands
------------------------

**For developers:** Generate a ``compile_commands.json`` file for IDE integration with language servers like clangd or ccls:

.. code-block:: bash

   # Export to default location (compile_commands.json)
   flashinfer export-compile-commands

   # Export to a specific path
   flashinfer export-compile-commands my_compile_commands.json

   # Or use the --output option
   flashinfer export-compile-commands --output /path/to/output.json

This compilation database enables:

- Code completion and navigation in IDEs
- Static analysis tools integration
- Better development experience with CUDA/C++ code

Module Management
-----------------

List and inspect compilation modules:

.. code-block:: bash

   # List all available modules
   flashinfer list-modules

   # Show details for a specific module
   flashinfer list-modules module_name

   # Show compilation status for all modules
   flashinfer module-status

   # Show detailed status with filters
   flashinfer module-status --detailed
   flashinfer module-status --filter compiled
   flashinfer module-status --filter not-compiled

Configuration and Status
-------------------------

Display FlashInfer configuration and installation status:

.. code-block:: bash

   flashinfer show-config

This displays:

- FlashInfer version and installed packages
- PyTorch and CUDA version information
- Environment variables and artifact paths
- Downloaded cubin status and module compilation status

Artifact Management
-------------------

Manage pre-compiled CUDA binaries:

.. code-block:: bash

   # Download pre-compiled cubins
   flashinfer download-cubin

   # List downloaded cubins
   flashinfer list-cubins

   # Clear downloaded cubins
   flashinfer clear-cubin

Cache Management
----------------

Clear JIT compilation cache:

.. code-block:: bash

   flashinfer clear-cache
