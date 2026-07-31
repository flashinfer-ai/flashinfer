.. _apicudnn:

flashinfer.cudnn
================

cuDNN-backed attention kernels. These wrappers call into NVIDIA's cuDNN runtime
for batch prefill and batch decode, and are typically used as an alternative
backend for ``BatchPrefillWithPagedKVCacheWrapper`` /
``BatchDecodeWithPagedKVCacheWrapper`` when cuDNN is available on the host GPU.

.. currentmodule:: flashinfer.cudnn

.. autosummary::
    :toctree: ../generated

    cudnn_batch_decode_with_kv_cache
    cudnn_batch_prefill_with_kv_cache
