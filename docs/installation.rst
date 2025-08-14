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

- Python: 3.8, 3.9, 3.10, 3.11, 3.12

Quick Start
^^^^^^^^^^^

The easiest way to install FlashInfer is via pip. Please note that the package currently used by FlashInfer is named ``flashinfer-python``, not ``flashinfer``.

.. code-block:: bash

    pip install flashinfer-python


.. _install-from-source:

Install from Source
^^^^^^^^^^^^^^^^^^^

In certain cases, you may want to install FlashInfer from source code to try out the latest features in the main branch, or to customize the library for your specific needs.

FlashInfer offers two installation modes:

JIT mode
   - CUDA kernels are compiled at runtime using PyTorch's JIT, with compiled kernels cached for future use.
   - JIT mode allows fast installation, as no CUDA kernels are pre-compiled, making it ideal for development and testing.
   - JIT version is also available as a sdist in `PyPI <https://pypi.org/project/flashinfer-python/>`_.

AOT mode
   - Core CUDA kernels are pre-compiled and included in the library, reducing runtime compilation overhead.
   - If a required kernel is not pre-compiled, it will be compiled at runtime using JIT. AOT mode is recommended for production environments.

JIT mode is the default installation mode. To enable AOT mode, see steps below.
You can follow the steps below to install FlashInfer from source code:

1. Clone the FlashInfer repository:

   .. code-block:: bash

       git clone https://github.com/flashinfer-ai/flashinfer.git --recursive

2. Make sure you have installed PyTorch with CUDA support. You can check the PyTorch version and CUDA version by running:

   .. code-block:: bash

       python -c "import torch; print(torch.__version__, torch.version.cuda)"

3. Install Ninja build system:

   .. code-block:: bash

       pip install ninja

4. Install FlashInfer:

   .. tabs::

       .. tab:: JIT mode

           .. code-block:: bash

               cd flashinfer
               pip install --no-build-isolation --verbose .

       .. tab:: AOT mode

           .. code-block:: bash

               cd flashinfer
               export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a"
               python -m flashinfer.aot  # Produces AOT kernels in aot-ops/
               python -m pip install --no-build-isolation --verbose .

5. Create FlashInfer distributions (optional):

   .. tabs::

       .. tab:: Create sdist

           .. code-block:: bash

               cd flashinfer
               python -m build --no-isolation --sdist
               ls -la dist/

       .. tab:: Create wheel for JIT mode

           .. code-block:: bash

               cd flashinfer
               python -m build --no-isolation --wheel
               ls -la dist/

       .. tab:: Create wheel for AOT mode

           .. code-block:: bash

               cd flashinfer
               export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a"
               python -m flashinfer.aot  # Produces AOT kernels in aot-ops/
               python -m build --no-isolation --wheel
               ls -la dist/

C++ API
-------

FlashInfer is a header-only library with only CUDA/C++ standard library dependency
that can be directly integrated into your C++ project without installation.
