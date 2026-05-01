FlashInfer on JAX with TVM FFI
==============================

These tutorials show how to call FlashInfer GPU kernels from JAX through the
`jax-tvm-ffi <https://github.com/NVIDIA/jax-tvm-ffi>`_ bridge.

The Sphinx-Gallery ``.py`` files in this directory are the canonical source:

* ``flashinfer_jax_tvm_ffi.py`` explains the core build, register, and call
  pattern for FlashInfer kernels from JAX.
* ``gemma3_flashinfer_jax.py`` applies the same pattern to Gemma 3 1B Instruct
  inference.

During the documentation build, Sphinx-Gallery renders these files into HTML
pages and creates downloadable Python and Jupyter notebook versions from the
same source files. Do not edit or commit the generated
``docs/tutorials/generated/jax_tvm_ffi/`` directory; it is produced by
Sphinx-Gallery.

The examples are not executed during the default documentation build because
they require an NVIDIA GPU, CUDA, FlashInfer JIT compilation, and in the Gemma 3
case Hugging Face credentials for a gated model.

Execution requirements
----------------------

To run the tutorials directly, use a CUDA-capable environment with:

* NVIDIA GPU with SM 7.5 or newer.
* CUDA 12.6 or newer.
* Python 3.10 or newer.
* JAX with CUDA support.
* ``flashinfer-python`` and ``jax-tvm-ffi``.

The Gemma 3 tutorial additionally requires:

* ``torch`` CPU wheels for dtype literals used by FlashInfer's JIT API.
* ``safetensors``, ``huggingface_hub``, and ``transformers``.
* Hugging Face access to ``google/gemma-3-1b-it`` and an ``HF_TOKEN``.

For example:

.. code-block:: bash

   pip install 'jax[cuda13]'
   pip install flashinfer-python -U jax-tvm-ffi \
       --no-build-isolation \
       --extra-index-url https://flashinfer.ai/whl/cu130/

   # Additional dependencies for the Gemma 3 tutorial only:
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install safetensors huggingface_hub transformers

To build the documentation locally from the repository root:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html -j auto

To run a tutorial directly, execute its canonical source file:

.. code-block:: bash

   python docs/tutorials/jax_tvm_ffi/flashinfer_jax_tvm_ffi.py
   python docs/tutorials/jax_tvm_ffi/gemma3_flashinfer_jax.py
