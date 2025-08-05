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

- PyTorch: 2.2/2.3/2.4/2.5 with CUDA 11.8/12.1/12.4 (only for torch 2.4 or later)

  - Use ``python -c "import torch; print(torch.version.cuda)"`` to check your PyTorch CUDA version.

- Supported GPU architectures: ``sm75``, ``sm80``, ``sm86``, ``sm89``, ``sm90``.

Quick Start
^^^^^^^^^^^

The easiest way to install FlashInfer is via pip, we host wheels with indexed URL for different PyTorch versions and CUDA versions. Please note that the package currently used by FlashInfer is named ``flashinfer-python``, not ``flashinfer``.

.. tabs::
    .. tab:: PyTorch 2.6

        .. tabs::

            .. tab:: CUDA 12.6

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

            .. tab:: CUDA 12.4

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

    .. tab:: PyTorch 2.5

        .. tabs::

            .. tab:: CUDA 12.4

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.5/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu118/torch2.5/

    .. tab:: PyTorch 2.4

        .. tabs::

            .. tab:: CUDA 12.4

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.4/


            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.4/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu118/torch2.4/

    .. tab:: PyTorch 2.3

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.3/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer-python -i https://flashinfer.ai/whl/cu118/torch2.3/


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

Environment Variables
^^^^^^^^^^^^^^^^^^^^

FlashInfer supports the following environment variables to customize the installation process:

- ``SKIP_NVSHMEM_PIP``: When set to "1", skips installing the ``nvidia-nvshmem-cu12`` dependency via pip. This is useful when the package is already provided as a system-level package (e.g., via RPM on RHEL-based distributions via the NVIDIA CUDA repository).

  .. code-block:: bash

      export SKIP_NVSHMEM_PIP=1
      pip install --no-build-isolation --verbose .

C++ API
-------

FlashInfer is a header-only library with only CUDA/C++ standard library dependency
that can be directly integrated into your C++ project without installation.
