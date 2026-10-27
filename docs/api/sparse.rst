.. _apisparse:

flashinfer.sparse
=================

Kernels for block sparse flashattention.

.. currentmodule:: flashinfer.sparse

.. autoclass:: BlockSparseAttentionWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__


.. autoclass:: VariableBlockSparseAttentionWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/examples/flashinfer-variable-block-sparse.png
        :width: 600
        :alt: variable block sparse attention plan function diagram
        :align: center

    .. automethod:: __init__


Cake VSA
--------

.. currentmodule:: flashinfer.cake_vsa

.. autosummary::
    :toctree: ../generated

    plan_cake_vsa
    run_cake_vsa


.. currentmodule:: flashinfer.sparse


flashinfer.msa_ops
==================

Minimax Sparse Attention (MSA) sparse prefill, sparse decode, and top-k
selection dispatch on SM100/SM103/SM107/SM120/SM121. ``msa_proxy_score`` also
supports SM90; only ``msa_proxy_score_fp4`` is SM120/SM121-only. On
SM100/SM103/SM107, packed NVFP4 K/V is served by dedicated
paged-KV prefill and decode routes that take the four regions of a packed
NVFP4 page -- K data, K scales, V data, V scales -- as strided views of one
allocation, for the MiniMax-M3 head geometry (head dimension 128, page size
128, and 64/4, 32/2 or 16/1 query/KV heads per tensor-parallel rank);
``msa_prefill_nvfp4_specialized_stats()`` and
``msa_decode_nvfp4_specialized_stats()`` report the exact accepted set. Other
NVFP4 forms -- flat or ragged K/V, or separately allocated scale tensors --
remain SM120/SM121-only, and outside those routes the 10.0/10.3/10.7 backend
requires separate contiguous K and V tensors and does not make implicit
copies. The decode ``out=`` parameter is implemented by that 10.0/10.3/10.7 route
alone; every other route raises ``NotImplementedError`` when it is passed.
The compute capability 10.0/10.3/10.7 backend uses TopK16 as its generic contract;
non-TopK16 shapes fail closed instead of entering a generic kernel. The decode
path uses direct persistent M16 ownership for both Q1 and multi-token decode;
it does not route BF16 decode through prefill or split-K. With
``force_fused=True``, two frozen BF16-query/FP8-KV single-query serving shapes
use transformed direct kernels, and paged uniform FP8 Q/K/V supports Q1
through Q32 and returns BF16 output. Long BF16 causal prefill uses a
selected-block reverse producer and deterministic reduction when every
request of the call has at least 8192 queries (and, on flat K/V, at least
8192 keys).
The NVFP4 paged-KV prefill route on compute capability 10.0/10.3/10.7 requires
CUDA 13.0 or newer; on an older toolkit it declines with the reason, which
``msa_prefill_nvfp4_specialized_stats()`` also reports as
``toolkit_decline_reason``. The decode route has no such floor. The generated
sm_107a decode programs of the ``cake`` backend additionally require an nvcc
that lists ``compute_107a``; on a toolkit without it the backend reports no
generated program for a 10.7 device and its JIT declines those programs with
the reason instead of failing inside the build. The prefill
route is bitwise
reproducible run to run up to a 16,384-token context, and is not guaranteed to
be above it. The per-tile block union is consumed in ascending hash-slot order;
at or below 128 blocks per request the slot is a permutation of the selected
set, so no two selected ids collide, the insert is a commutative ``atomicOr``
and repeated calls return identical bits. Above that width the insert falls
back to linear probing under ``atomicCAS``, colliding ids land in whatever
order the atomics resolve in, and the accumulation order -- and so the last
bits of the output -- may differ between runs. Every result is correct either
way. ``msa_prefill_nvfp4_specialized_stats()`` reports the bound as
``run_to_run_bitwise_reproducible_up_to_context``.
Call :func:`flashinfer.msa_ops.supports_packed_kv` with the active device when
integrating a cache manager across these architectures; the legacy aggregate
``SUPPORTS_PACKED_KV`` flag describes the SM120/SM121 backend.
Per-token tensor ``num_valid_pages`` for
:func:`flashinfer.msa_ops.msa_topk_select` is likewise SM120/SM121-only;
compute capability 10.0/10.3/10.7 requires a scalar value or ``None`` and rejects
the tensor form before backend dispatch.

CUDA graph capture of sparse prefill or decode on compute capability
10.0/10.3/10.7 requires a caller-owned
:class:`flashinfer.msa_ops.MSASparseAttentionWorkspace`. Warm the workspace
eagerly with the exact tensors, options, and capture stream before capture.
The two FP8-KV single-query decode specializations are eager-only; under
capture the same inputs take the generic decode route.

.. currentmodule:: flashinfer.msa_ops

.. autosummary::
    :toctree: ../generated

    msa_proxy_score
    msa_proxy_score_fp4
    MSASparseAttentionWorkspace
    supports_packed_kv
    msa_sparse_attention
    msa_sparse_decode_attention
    msa_topk_select
    prepare_msa_nvfp4_sparse_decode
