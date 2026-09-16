.. _apigdn_decode:

flashinfer.gdn_decode
=====================

Gated Delta-Rule decode-side kernels used in Mamba-2 / GDN-style sequence
models. These functions consume a pre-built KV / state cache and run the
recurrent gated delta-rule update for the current decode step.

.. currentmodule:: flashinfer.gdn_decode

.. autosummary::
    :toctree: ../generated

    gated_delta_rule_decode
    gated_delta_rule_decode_pretranspose
    gated_delta_rule_mtp
    gated_delta_rule_replayssm_commit

ReplaySSM speculative verification (SM100/SM103)
---------------------------------------------

Use ``gated_delta_rule_mtp(cache_replayssm=True, disable_state_update=True)``
with FP32 128x128 checkpoints. For each layer, pass contiguous views into
all-layer BF16 raw K/V windows and FP32 log-decay/beta windows. The windows are
indexed by the same pool slots as the checkpoints. Verify produces outputs
without changing the checkpoint or storing a full state for every token.

Once the accepted lengths are known, call ``gated_delta_rule_replayssm_commit``
once across all layers. It replays only the accepted prefix and updates each
live checkpoint in place. Zero acceptance leaves the checkpoint unchanged.
The next verify overwrites the raw windows; no circular cache or deferred
state update is retained between rounds. Use identical Q/K normalization
settings for verify and commit.

The tcgen05 verify specialization uses TMA and TF32 tensor cores for T=4/8,
B=1..256, BF16 output, int32 indices, contiguous FP32 checkpoints, normalized
Q/K and an even value-head/key-head ratio. Other supported SM100 inputs use
the FP32 MTP fallback. TF32 contraction changes numerical reduction precision.
See the API docstrings for shape, index uniqueness, padding and optional
tracked-checkpoint contracts.

Commit defaults to ``backend="auto"``: normalized T=8 without tracking uses
tcgen05; other cases use SIMT. Specify ``backend="simt"`` to keep FP32
contractions when checkpoint accuracy is preferred over tensor-core throughput.
