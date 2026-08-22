.. _apikda_training:

flashinfer.kda_training
=======================

``recurrent_kda_training_forward`` and
``recurrent_kda_training_backward`` form one paired training API. The forward
returns BF16 token output, a BF16 serving-recurrence final state promoted to an
FP32 tensor, and a persistent context containing the training checkpoints and
active-beta values consumed by the backward. The serving recurrence uses
private output/state scratch and does not overwrite the public training output.
The paired backward consumes the saved training context directly; it does not
recompute either forward recurrence.
The backward and any caller-owned context reuse must run on the same CUDA
stream that produced the context. Calls sharing one context are serialized.

The frozen route requires Blackwell compute capability 10.0 or 10.3, head and
state dimensions 128, and eight packed 1024-token sequences with 96 heads.
Inputs Q/K/V/raw-gate/raw-beta are BF16. Parameters and recurrent states are
FP32. The checkpoint-producing schedule retains BF16 checkpoint carriers. The
safe gate lower bound is fixed to ``-5.0`` and the scale to ``1 / sqrt(128)``.

.. currentmodule:: flashinfer.kda_training

.. autosummary::
    :toctree: generated

    RecurrentKDATrainingContext
    recurrent_kda_training_forward
    recurrent_kda_training_backward
