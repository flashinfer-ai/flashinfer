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

- Python: 3.10, 3.11

- PyTorch CUDA 11.8/12.1

  - Use ``python -c "import torch; print(torch.version.cuda)"`` to check your PyTorch CUDA version.

- Supported GPU architectures: ``sm80``, ``sm86``, ``sm89``, ``sm90`` (``sm75`` / ``sm70`` support is working in progress).

Quick Start
^^^^^^^^^^^

.. tabs::

    .. tab:: PyTorch 2.2

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.2/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.2/

    .. tab:: PyTorch 2.1

        .. tabs::

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.1/

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    pip install flashinfer -i https://flashinfer.ai/whl/cu118/torch2.1/


C++ API
-------

FlashInfer is a header-only library with only CUDA/C++ standard library dependency
that can be directly integrated into your C++ project without installation.

You can check our `unittest and benchmarks <https://github.com/flashinfer-ai/flashinfer/tree/main/src>`_ on how to use our C++ APIs at the moment.

.. note::
    The ``nvbench`` and ``googletest`` dependency in ``3rdparty`` directory are only
    used to compile unittests and benchmarks, and are not required for the library itself.
