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

- Python: 3.9, 3.10, 3.11, 3.12, 3.13

Quick Start
^^^^^^^^^^^

The easiest way to install FlashInfer is via pip. Please note that the package currently used by FlashInfer is named ``flashinfer-python``, not ``flashinfer``.

.. code-block:: bash

    pip install flashinfer-python


.. _install-from-source:

Install from Source
^^^^^^^^^^^^^^^^^^^

In certain cases, you may want to install FlashInfer from source code to try out the latest features in the main branch, or to customize the library for your specific needs.

``flashinfer-python`` is a source-only package and by default it will JIT compile/download kernels on-the-fly.

For fully offline deployment, we also provide two additional packages to pre-compile and download cubins ahead-of-time:

flashinfer-cubin
   - Provides pre-compiled CUDA binaries for immediate use without runtime compilation.

flashinfer-jit-cache
   - Pre-compiles kernels for specific CUDA architectures to enable fully offline deployment.

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

   For development & contribution, install in editable mode:

   .. code-block:: bash

       python -m pip install --no-build-isolation -e . -v

4. (Optional) Build additional packages for offline deployment:

   To build ``flashinfer-cubin`` package from source:

   .. code-block:: bash

       cd flashinfer-cubin
       python -m build --no-isolation --wheel
       python -m pip install dist/*.whl

   To build ``flashinfer-jit-cache`` package from source:

   .. code-block:: bash

       export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 10.0a 10.3a 12.0a"  # user can shrink the list to specific architectures
       cd flashinfer-jit-cache
       python -m build --no-isolation --wheel
       python -m pip install dist/*.whl
