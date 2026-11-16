.. _apiqsa:

flashinfer.qsa
=============

Packed BF16 query sparse attention (QSA), backed by cuDNN Frontend 1.32 or
newer. These APIs compute attention over caller-selected four-token blocks.
They do not generate selections or register an autograd function.

.. list-table:: Supported forward configurations
   :header-rows: 1

   * - API
     - Selection
     - Query / KV heads
     - GPU
   * - ``qsa_forward``
     - uint8 block bitmap
     - 12 / 1 or 24 / 2
     - SM100, SM103, SM107
   * - ``qsa_indexed_forward``
     - int32 token indices
     - 12 / 1
     - SM100, SM103

Both require head dimension 256. Outputs are BF16 ``[T, Hq, 256]``, FP32
output before casting with the same shape, and FP32 **natural-log** LSE
``[Hq, T]``. Empty selections produce zero output and negative-infinity LSE.
All tensors must be contiguous on one CUDA device. cuDNN validates tensor
metadata and compiler support; packed sequence boundaries and selection values
are caller contracts and are not copied to the host for validation.

The indexed API consumes distinct four-token groups, with ``-1`` padding and
an optional partial causal tail. Indices are absolute packed token IDs from
the query's own sequence. Width must be divisible by four and at most 2052.
It avoids the bitmap API's int32 ``[T, NB]`` compact-index scratch, which alone
uses 16 GiB at 128K tokens and 32K blocks.

Indexed forward is not bitwise repeatable: cuDNN's parallel index compaction
can change the attention accumulation order, including BF16 output values.
Sorting caller indices does not remove this internal ordering variation.
This API has no deterministic mode; account for this in reproducibility-sensitive
workloads, including reinforcement learning.

cuDNN allocates outputs and scratch and uses the input device's current CUDA
stream. Warm up each required configuration before CUDA graph capture.
Replay reads current selection tensors; no implicit selection cache is kept.
The cuDNN package requires CuTe DSL 4.8+; SM107 also needs a compiler build
with Rubin support. These APIs are separate from the NVFP4 paged-cache
``flashinfer.sparse.qsa_sparse_paged_attention`` interface.

.. code-block:: python

   import torch
   import flashinfer

   q = torch.randn(8, 12, 256, device="cuda", dtype=torch.bfloat16)
   k = torch.randn(8, 1, 256, device="cuda", dtype=torch.bfloat16)
   v = torch.randn_like(k)
   token = torch.arange(8, device="cuda", dtype=torch.int32)
   indices = torch.where(token[None, :] <= token[:, None], token[None, :], -1)
   out, out_fp32, lse = flashinfer.qsa_indexed_forward(q, k, v, indices)

.. currentmodule:: flashinfer.qsa

.. autosummary::
   :toctree: ../generated

   qsa_forward
   qsa_indexed_forward
