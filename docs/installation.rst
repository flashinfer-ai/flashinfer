.. _installation:

Installation
============

Python Package
--------------
FlashInfer is available as a Python package, built on top of `PyTorch <https://pytorch.org/>`_ to
easily integrate with your python applications.

Prerequisites
^^^^^^^^^^^^^

- OS: Linux only

- Python: 3.10, 3.11, 3.12, 3.13, 3.14

- CUDA: 12.6, 12.8, 13.0, 13.1

.. note::
   FlashInfer strives to follow PyTorch's supported CUDA versions plus the latest CUDA release.

Quick Start
^^^^^^^^^^^

The easiest way to install FlashInfer is via pip. Please note that the package currently used by FlashInfer is named ``flashinfer-python``, not ``flashinfer``.

.. code-block:: bash

    pip install flashinfer-python

Package Options
"""""""""""""""

FlashInfer provides three packages:

- **flashinfer-python**: Core package that compiles/downloads kernels on first use
- **flashinfer-cubin**: Pre-compiled kernel binaries for all supported GPU architectures
- **flashinfer-jit-cache**: Pre-built kernel cache for specific CUDA versions

**For faster initialization and offline usage**, install the optional packages to have most kernels pre-compiled:

.. code-block:: bash

    pip install flashinfer-python flashinfer-cubin
    # JIT cache package: autodetects CUDA + GPU SM family and runs the right pip install.
    flashinfer install-jit-cache-wheel

This eliminates compilation and downloading overhead at runtime.

``flashinfer-jit-cache`` is published as separate per-(CUDA, SM family) wheels because a
single multi-arch wheel exceeds GitHub Releases' 2 GiB asset limit. The CLI resolves
the right one for you when all visible GPUs can be covered by one wheel. To pick
manually, override the autodetection:

.. code-block:: bash

    # Datacenter Blackwell (sm100/103/110), CUDA 13.0
    flashinfer install-jit-cache-wheel --cuda-version 13.0 --sm-family sm10x

    # Show the pip command without executing
    flashinfer install-jit-cache-wheel --dry-run

The SM families are: ``sm9x`` (Ampere/Ada/Hopper, ≤sm90), ``sm10x`` (Datacenter Blackwell,
sm100/103/110), and ``sm12x`` (Consumer Blackwell, sm120/121). Blackwell-family
wheels also retain the ``sm80`` base arch alongside their native Blackwell archs.


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

       cd flashinfer-cubin
       python -m build --no-isolation --wheel
       python -m pip install dist/*.whl

   Build ``flashinfer-jit-cache`` (customize ``FLASHINFER_CUDA_ARCH_LIST`` for your target GPUs):

   .. code-block:: bash

       export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 10.3a 11.0a 12.0f 12.1a"
       cd flashinfer-jit-cache
       python -m build --no-isolation --wheel
       python -m pip install dist/*.whl

   To build a per-SM-family wheel matching a release artifact (smaller; targets one
   GPU family), set ``FLASHINFER_JIT_CACHE_SM_FAMILY``. The local-version suffix on
   the resulting wheel will include the family (e.g. ``+cu130.sm10x``).

   .. code-block:: bash

       export FLASHINFER_CUDA_ARCH_LIST="8.0 10.0a 10.3a 11.0a"
       export FLASHINFER_JIT_CACHE_SM_FAMILY="sm10x"
       cd flashinfer-jit-cache
       python -m build --no-isolation --wheel


Install Nightly Build
^^^^^^^^^^^^^^^^^^^^^^

Nightly builds are available for testing the latest features:

.. code-block:: bash

    # Core and cubin packages
    pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps # Install the nightly package from custom index, without installing dependencies
    pip install flashinfer-python  # Install flashinfer-python's dependencies from PyPI
    pip install -U --pre flashinfer-cubin --index-url https://flashinfer.ai/whl/nightly/
    # JIT cache package: autodetect CUDA + GPU and pull from the nightly index
    flashinfer install-jit-cache-wheel --nightly

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
