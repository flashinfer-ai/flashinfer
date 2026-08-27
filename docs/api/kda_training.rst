.. _apikda_training:

flashinfer.kda_training
=======================

``recurrent_kda_training_forward`` and
``recurrent_kda_training_backward`` form one paired training API. The forward
returns BF16 token output, an FP32 recurrent final state, and a persistent
context containing the route-specific checkpoints, tapes, and active-beta
values consumed by the backward. An accurate full-precision-state recurrence
produces the public FP32 final state for the C16 and C32 routes. On C16, its
token output is private scratch so the selected training output remains public.
On C32, the accurate recurrence also produces the public token output, while
the chunked C32 tape and checkpoints remain saved only as backward context.
The row-split route directly produces its public token output and final state.
The paired backward consumes the saved training context directly; it does not
recompute either forward recurrence.
CUDA graph capture is not supported. Caller-provided forward outputs and
backward gradient outputs must not overlap each other or any input or saved
context storage read by the same call. Inputs and saved checkpoint or metadata
tensors must not be modified between forward and backward; tensor-version
changes are rejected before the backward launch and also prevent context reuse.

A caller-owned context may be reused only with matching shape metadata and CUDA
device, on the CUDA stream that originally created it. Reuse overwrites its
saved checkpoints and metadata. Calls sharing one context are serialized.

The frozen production dispatcher requires Blackwell compute capability 10.0
or 10.3, key/value dimensions 128, BF16 Q/K/V/raw-gate/raw-beta, and FP32
parameters and recurrent states. Q and K use ``Hqk`` heads; V, raw gate, raw
beta, and recurrent state use ``Hv`` heads, where ``Hv % Hqk == 0``. Every
semantic sequence must be non-empty. The safe gate lower bound is fixed to
``-5.0`` and the scale to ``1 / sqrt(128)``.

Fixed layout accepts contiguous ``[B, T, H, 128]`` tensors with ``B >= 1`` and
omitted ``cu_seqlens``; each physical batch row is one semantic sequence.
Packed layout accepts a physical batch dimension of one plus CUDA int64
``cu_seqlens``. Packed sequence lengths may be mixed, and neither layout
requires a 16-token-aligned length.

The dispatcher selects grouped or equal-head C16 for shapes that satisfy its
fast-route predicates. Other grouped or high-head shapes use the C32 fallback;
other low-head equal-head shapes use the row-split fallback. These predicates
select an implementation and are not public shape guards. The paired backward
consumes the exact context saved by the selected route, including C32 tails and
mixed sequence lengths, without rerunning a forward kernel.

The exact packed training shape with eight 1024-token sequences and 96 equal
heads is validated against FLA for output, final state, and all eight gradients
at ``atol=rtol=1e-2``. Regression coverage also exercises grouped C32 mixed
tails and fixed B2/B4/B8 layouts through the same public paired API.

.. currentmodule:: flashinfer.kda_training

.. autosummary::
    :toctree: generated

    RecurrentKDATrainingContext
    recurrent_kda_training_forward
    recurrent_kda_training_backward
