.. _tvm-bindings

TVM Bindings
============

.. contents:: Table of Contents
    :local:
    :depth: 2

FlashInfer also provides TVM bindings where the kernels are wrapped as `PackedFunc in TVM<https://tvm.apache.org/docs/arch/runtime.html#packedfunc>`_.
Registered functions can be used in different languages (Python/Rust/Javascript/etc) with `TVM Runtime System <https://tvm.apache.org/docs/arch/runtime.html>`_, the wrapped 
function definitions can be found at `tvm_wrapper.cu <https://github.com/flashinfer-ai/flashinfer/blob/main/src/tvm_wrapper.cu>`_.


Compile TVM Bindings
--------------------

.. code-block:: bash

    git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
    cd flashinfer
    mkdir build
    cp cmake/config.cmake build/
    cd build
    cmake .. -DFLASHINFER_TVM_BINDING=ON
    make -j12

Deploy LLM using FlashInfer kernels with TVM
--------------------------------------------

`MLC-LLM <https://github.com/mlc-ai/mlc-llm>`_ provides end-to-end examples of deploying LLM with TVM using FlashInfer kernels.

Use TVM Bindings in Python
--------------------------

FlashInfer functions registered in TVM Bindings could be used in other languages such as Python, here is an example

.. code-block:: python

    import tvm
    import numpy
    f_single_decode = tvm.get_global_func("flashinfer.single_decode")
    dev = tvm.cuda(0)
    seq_len = 18
    num_heads = 32
    head_dim = 128 
    q = tvm.nd.array(np.random.randn((num_heads, head_dim)).astype("float16"), device)
    k = tvm.nd.array(np.random.randn((seq_len, num_heads, head_dim)).astype("float16"), device)
    v = tvm.nd.array(np.random.randn((seq_len, num_heads, head_dim)).astype("float16"), device)
    tmp = tvm.nd.empty((4 * 1024 * 1024,), dtype="float32", device=dev)
    o = tvm.nd.empty((num_heads, head_dim), dtype="float16", device=dev)
    f_single_decode(q, k, v, tmp, 0, 0, 1., 1e4, o)


A more direct way to use FlashInfer in Python is through our PyTorch APIs, but with TVM binding you can export FlashInfer API to language other than Python.

