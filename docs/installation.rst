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

- PyTorch: 2.2/2.3/2.4 with CUDA 11.8/12.1/12.4 (only for torch 2.4)

  - Use ``python -c "import torch; print(torch.version.cuda)"`` to check your PyTorch CUDA version.

- Supported GPU architectures: ``sm75``, ``sm80``, ``sm86``, ``sm89``, ``sm90``.

Quick Start
^^^^^^^^^^^

The easiest way to install FlashInfer is via pip:

.. tabs::

    .. tab:: PyTorch 2.4

        .. tabs::

            .. tab:: CUDA 12.4

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.4/

    .. tab:: PyTorch 2.3

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.3/

    .. tab:: PyTorch 2.2

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.2/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.2/

    .. tab:: PyTorch 2.1

        Since FlashInfer version 0.1.2, support for PyTorch 2.1 has been ended. Users are encouraged to upgrade to a newer
        PyTorch version or :ref:`install FlashInfer from source code. <install-from-source>` .

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.1/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.1/

.. _install-from-source:

Install from Source
^^^^^^^^^^^^^^^^^^^

In certain cases, you may want to install FlashInfer from source code to trying out the latest features in the main branch, or to customize the library for your specific needs.

FlashInfer offers two installation modes:

JIT mode
   - CUDA kernels are compiled at runtime using PyTorch's JIT, with compiled kernels cached for future use.
   - JIT mode allows fast installation, as no CUDA kernels are pre-compiled, making it ideal for development and testing.

AOT mode
   - Core CUDA kernels are pre-compiled and included in the library, reducing runtime compilation overhead.
   - If a required kernel is not pre-compiled, it will be compiled at runtime using JIT. AOT mode is recommended for production environments.

JIT mode is the default installation mode. To enable AOT mode, set the environment variable ``FLASHINFER_ENABLE_AOT=1`` before installing FlashInfer.
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
               pip install --no-build-isolation --verbose --editable .

       .. tab:: AOT mode

           .. code-block:: bash

               cd flashinfer
               TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a" FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation --verbose --editable .

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
               TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a" FLASHINFER_ENABLE_AOT=1 python -m build --no-isolation --wheel
               ls -la dist/

C++ API
-------

FlashInfer is a header-only library with only CUDA/C++ standard library dependency
that can be directly integrated into your C++ project without installation.

You can check our `unittest and benchmarks <https://github.com/flashinfer-ai/flashinfer/tree/main/src>`_ on how to use our C++ APIs at the moment.

.. note::
    The ``nvbench`` and ``googletest`` dependency in ``3rdparty`` directory are only
    used to compile unittests and benchmarks, and are not required for the library itself.

.. _compile-cpp-benchmarks-tests:

Compile Benchmarks and Unittests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compile the C++ benchmarks (using `nvbench <https://github.com/NVIDIA/nvbench>`_) and unittests, you can follow the steps below:

1. Clone the FlashInfer repository:

   .. code-block:: bash

       git clone https://github.com/flashinfer-ai/flashinfer.git --recursive

2. Check conda is installed (you can skip this step if you have installed cmake and ninja in other ways):

   .. code-block:: bash

       conda --version

   If conda is not installed, you can install it by following the instructions on the `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
   `miniforge <https://github.com/conda-forge/miniforge>`_ websites.

2. Install CMake and Ninja build system:

   .. code-block:: bash

       conda install cmake ninja

3. Create build directory and copy configuration files

   .. code-block:: bash

       mkdir -p build
       cp cmake/config.cmake build/  # you can modify the configuration file if needed

4. Compile the benchmarks and unittests:

   .. code-block:: bash

       cd build
       cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
       ninja
