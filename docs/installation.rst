.. _installation:

Installation
============

Python Package
--------------
FlashInfer is available as a Python package, built on top of `PyTorch <https://pytorch.org/>`_ to
easily integrate with your Python applications.

Prerequisites
^^^^^^^^^^^^^

- OS: Linux only

- Python: 3.10, 3.11, 3.12, 3.13, 3.14

- CUDA: 12.9, 13.0, 13.4 (PyTorch nightly)

.. note::
   FlashInfer strives to follow PyTorch's supported CUDA versions plus the latest CUDA release.
   CUDA 13.4 wheels are built with the preview toolkit and PyTorch nightly;
   runtime CI currently covers CUDA 12.9 and 13.0.

Quick Start
^^^^^^^^^^^

The easiest way to install FlashInfer is via pip. Please note that the package currently used by FlashInfer is named ``flashinfer-python``, not ``flashinfer``.

.. code-block:: bash

    pip install flashinfer-python

Package Options
"""""""""""""""

FlashInfer provides three packages:

- **flashinfer-python**: Core package that compiles/downloads kernels on first use

  Release wheels on PyPI leave out the kernel sources of the experimental features under ``flashinfer.experimental`` to stay under PyPI's file size limit; those features need a nightly wheel from https://flashinfer.ai/whl or an install from source.

- **flashinfer-cubin**: Pre-compiled kernel binaries for all supported GPU architectures
- **flashinfer-jit-cache**: CUDA-specific shim that installs architecture-specific pre-built kernel providers

**For faster initialization and offline usage**, install the optional packages to have most kernels pre-compiled:

.. code-block:: bash

    pip install flashinfer-python
    flashinfer download-kernels

This eliminates compilation and downloading overhead at runtime.
``flashinfer download-kernels`` installs the ``flashinfer-cubin`` and
``flashinfer-jit-cache`` wheels that match the installed ``flashinfer-python``
version and the CUDA version of the installed PyTorch. By default it installs
pre-built kernels for every supported GPU architecture (several GB); pass
``--mode minimal`` to install the jit-cache provider only for the visible GPUs,
or ``--sm`` (for example ``--sm sm90a``, repeatable) to choose architectures
explicitly. See :doc:`cli` for all options, and :ref:`upgrading` before
upgrading an existing installation.


.. _upgrading:

Upgrading
^^^^^^^^^

The pre-built kernel wheels are tied to one exact ``flashinfer-python``
version, and upgrading ``flashinfer-python`` does not upgrade them. Re-install
them after every upgrade:

.. code-block:: bash

    pip install -U flashinfer-python
    flashinfer download-kernels   # same --mode / --sm options as the first install

Until you do, FlashInfer logs a warning at import (``Ignoring incompatible
flashinfer-cubin package`` / ``flashinfer-jit-cache package``) and ignores the
old wheels: cubins are downloaded on demand and kernels are JIT-compiled, as if
the wheels were not installed. Results stay correct, but first calls are slow
again and offline machines can fail to load kernels. Set
``FLASHINFER_DISABLE_VERSION_CHECK=1`` only if you intentionally mix versions.

**flashinfer-cubin moved off PyPI.** ``flashinfer-cubin`` 0.6.13 was the last
release published to PyPI; newer releases are only on the FlashInfer wheel
index, https://flashinfer.ai/whl. Because PyPI still serves 0.6.13, these
commands silently install a stale wheel that FlashInfer then ignores:

- ``pip install flashinfer-cubin`` or ``pip install -U flashinfer-cubin`` without
  ``--index-url https://flashinfer.ai/whl``
- a ``requirements.txt`` or lock file that resolves ``flashinfer-cubin`` from
  PyPI

Use ``flashinfer download-kernels``, which picks the right index and pins the
matching version, or pin it yourself:
``pip install flashinfer-cubin==<flashinfer-python version> --index-url https://flashinfer.ai/whl``.
``flashinfer-jit-cache`` was never published to PyPI; it is served from the
CUDA-specific index (for example https://flashinfer.ai/whl/cu130).

Run ``flashinfer show-config`` to compare the installed ``flashinfer-cubin``
and ``flashinfer-jit-cache`` versions with ``flashinfer-python``.


.. _install-from-source:

Install from Source
^^^^^^^^^^^^^^^^^^^

In certain cases, you may want to install FlashInfer from source code to try out the latest features in the main branch, or to customize the library for your specific needs.

You can follow the steps below to install FlashInfer from source code:

1. Clone the FlashInfer repository:

   .. code-block:: bash

       git clone https://github.com/flashinfer-ai/flashinfer.git --recursive

2. Make sure you have installed PyTorch with CUDA support. You can check the PyTorch version and CUDA version by running:

   .. code-block:: bash

       python -c "import torch; print(torch.__version__, torch.version.cuda)"

3. Install FlashInfer:

   .. code-block:: bash

       cd flashinfer
       python -m pip install -v .

   **For development**, install in editable mode:

   .. code-block:: bash

       python -m pip install --no-build-isolation -e . -v

   .. note::
      When using ``--no-build-isolation``, pip does not automatically install build
      dependencies. FlashInfer requires ``setuptools>=77``. If you encounter an error
      like ``AttributeError: module 'setuptools.build_meta' has no attribute
      'prepare_metadata_for_build_editable'``, upgrade pip and setuptools first:

      .. code-block:: bash

          python -m pip install --upgrade pip setuptools

4. (Optional) Build optional packages:

   Build ``flashinfer-cubin``:

   .. code-block:: bash

       python -m build --no-isolation --wheel flashinfer-cubin
       python -m pip install flashinfer-cubin/dist/*.whl

   Build one JIT-cache provider for the target GPU, then build a shim that
   depends on that provider. The example below builds an SM90a provider; both
   wheels must use the same version settings:

   .. code-block:: bash

       export FLASHINFER_JIT_CACHE_PROVIDER_ARCH=9.0a
       python -m build --no-isolation --wheel flashinfer-jit-cache-provider

       export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="9.0a"
       python -m build --no-isolation --wheel flashinfer-jit-cache

       python -m pip install \
           flashinfer-jit-cache-provider/dist/*.whl \
           flashinfer-jit-cache/dist/*.whl


Install Nightly Build
^^^^^^^^^^^^^^^^^^^^^^

Nightly builds are available for testing the latest features:

.. code-block:: bash

    pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps # Install the nightly package from custom index, without installing dependencies
    pip install flashinfer-python  # Install flashinfer-python's dependencies from PyPI
    flashinfer download-kernels --nightly  # Optional: matching nightly flashinfer-cubin and flashinfer-jit-cache wheels

Verify Installation
^^^^^^^^^^^^^^^^^^^

After installation, verify that FlashInfer is correctly installed and configured:

.. code-block:: bash

    flashinfer show-config

This command displays:

- FlashInfer version and installed packages (flashinfer-python, flashinfer-cubin, flashinfer-jit-cache)
- PyTorch and CUDA version information
- Environment variables and artifact paths
- Downloaded cubin status and module compilation status
